//! Retouching GPU paths against the CPU: the healing membrane solve
//! (Poisson), PatchMatch completion and mip pyramids.
//!
//! GPU tests skip themselves when no adapter exists or `PF_GPU=off`.

use super::tests::{edge_case, gpu, photo};
use crate::engine::grid::TileGrid;
use crate::engine::poisson::{self, Quality, Solver};
use crate::engine::px;

pub fn region(g: &TileGrid, x: i32, y: i32, w: usize, h: usize) -> Vec<u32> {
    let mut v = vec![0u32; w * h];
    g.read_region(x, y, w, h, &mut v);
    v
}

/// Max channel difference, mean ΔE2000 (unpremultiplied colour) and the
/// number of differing pixels, over `sel` (all pixels when None).
pub fn compare(a: &[u32], b: &[u32], sel: Option<&[u8]>) -> (u8, f64, usize) {
    let mut mx = 0u8;
    let mut n = 0;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if sel.is_some_and(|s| s[i] == 0) {
            continue;
        }
        let (ux, uy) = (px::unpack(x), px::unpack(y));
        let d = (0..4).map(|c| ux[c].abs_diff(uy[c])).max().unwrap();
        mx = mx.max(d);
        n += (d > 0) as usize;
    }
    let rgb = |v: &[u32]| -> Vec<[u8; 3]> {
        v.iter()
            .map(|&p| {
                let c = px::unpremultiply(p);
                [c[0], c[1], c[2]]
            })
            .collect()
    };
    (mx, crate::engine::metrics::mean_delta_e(&rgb(a), &rgb(b), sel), n)
}

fn disk(w: usize, h: usize, cx: f32, cy: f32, rx: f32, ry: f32) -> Vec<u8> {
    let mut m = vec![0u8; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let (dx, dy) = ((x as f32 - cx) / rx, (y as f32 - cy) / ry);
            m[y * w + x] = (dx * dx + dy * dy < 1.0) as u8;
        }
    }
    m
}

struct HealCase {
    name: &'static str,
    src: Vec<u32>,
    dst: Vec<u32>,
    mask: Vec<u8>,
    w: usize,
    h: usize,
    cycles: usize,
}

/// Healing-shaped problems: dabs of 60–500 px (every solver branch: plain
/// multigrid, nested iteration, the fast half-resolution extension), a
/// content-aware-fill-sized hole, translucent pixels, and a mask reaching
/// the region edge.
fn heal_cases() -> Vec<HealCase> {
    let dog = photo("dog_grass");
    let edge = edge_case();
    let mut v = Vec::new();
    for (name, d, cycles) in [("dab 60", 60usize, 4usize), ("dab 100", 100, 4), ("dab 200", 200, 4), ("dab 300", 300, 5), ("dab 500", 500, 4)] {
        let (w, h) = (d + 6, d + 6);
        let c = (w as f32 / 2.0, h as f32 / 2.0);
        v.push(HealCase {
            name,
            src: region(&dog, 40, 300 - (d as i32) / 2, w, h),
            dst: region(&dog, 470, 330 - (d as i32) / 2, w, h),
            mask: disk(w, h, c.0, c.1, d as f32 / 2.0, d as f32 / 2.0),
            w,
            h,
            cycles,
        });
    }
    // A Content-Aware-Fill-like hole: the fill is a shifted copy.
    let (w, h) = (900usize, 600usize);
    v.push(HealCase { name: "caf hole", src: region(&dog, 100, 60, w, h), dst: region(&dog, 0, 0, w, h), mask: disk(w, h, 430.0, 300.0, 380.0, 240.0), w, h, cycles: 5 });
    // Translucent and transparent pixels.
    let (w, h) = (420usize, 380usize);
    v.push(HealCase { name: "alpha", src: region(&edge, 420, 300, w, h), dst: region(&edge, 300, 200, w, h), mask: disk(w, h, 210.0, 190.0, 200.0, 180.0), w, h, cycles: 5 });
    // Unknowns up to the region edge (the solver must clear its border).
    let (w, h) = (240usize, 200usize);
    let mut mask = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            mask[y * w + x] = (x > 30 || y > 20) as u8;
        }
    }
    v.push(HealCase { name: "edge mask", src: region(&dog, 600, 400, w, h), dst: region(&dog, 100, 380, w, h), mask, w, h, cycles: 4 });
    v
}

#[test]
fn healing_blend_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    for q in [Quality::Fast, Quality::Exact] {
        for c in heal_cases() {
            let cpu = poisson::blend_rgba_cpu(&mut Solver::new(), &c.src, &c.dst, &c.mask, c.w, c.h, c.cycles, q);
            let out = poisson::blend_rgba_gpu(&g, &c.src, &c.dst, &c.mask, c.w, c.h, c.cycles, q).unwrap_or_else(|e| panic!("{} {q:?}: {e}", c.name));
            let (mx, de, n) = compare(&out, &cpu, None);
            let unknowns = c.mask.iter().filter(|&&m| m != 0).count();
            eprintln!("poisson {:<10} {q:?}: max diff {mx}, mean ΔE {de:.4}, {n} of {unknowns} unknowns differ", c.name);
            assert!(mx <= 1 && de <= 0.2, "{} {q:?}: max diff {mx}, mean ΔE {de:.4}", c.name);
        }
    }
}

// ----- PatchMatch --------------------------------------------------------------------

use crate::engine::gpu::PathTaken;
use crate::engine::patchmatch::{self, Adapt, ColorAdapt, Engine, Params, RotationAdapt};

fn straight(v: &[u32]) -> Vec<[u8; 3]> {
    v.iter()
        .map(|&p| {
            let c = px::unpremultiply(p);
            [c[0], c[1], c[2]]
        })
        .collect()
}

/// A periodic structured texture: offset bricks with mortar and a colour
/// gradient inside each brick.
pub fn bricks(w: usize, h: usize) -> Vec<u32> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            let row = y / 18;
            let bx = (x + if row % 2 == 1 { 21 } else { 0 }) % 42;
            let by = y % 18;
            if bx < 3 || by < 3 {
                px::pack(200, 196, 188, 255)
            } else {
                let t = (bx * 4 + by * 3) as u8;
                px::pack(150 + t / 2, 60 + t / 3, 40 + (row * 13 % 30) as u8, 255)
            }
        })
        .collect()
}

pub struct PmCase {
    pub name: &'static str,
    pub img: Vec<u32>,
    pub hole: Vec<u8>,
    pub w: usize,
    pub h: usize,
    pub p: Params,
}

pub fn pm_cases() -> Vec<PmCase> {
    let dog = photo("dog_grass");
    let (w, h) = (dog.w as usize, dog.h as usize);
    let img = region(&dog, 0, 0, w, h);
    let hole = disk(w, h, w as f32 * 0.62, h as f32 * 0.72, 90.0, 70.0);
    let (bw, bh) = (900usize, 640usize);
    let adapt = Adapt { color: ColorAdapt::Default, ..Default::default() };
    let rot = Adapt { rotation: RotationAdapt::Medium, scale: true, mirror: true, color: ColorAdapt::Default };
    vec![
        PmCase { name: "dog hole", img: img.clone(), hole: hole.clone(), w, h, p: Params::default() },
        PmCase { name: "dog hole, colour adapt", img: img.clone(), hole: hole.clone(), w, h, p: Params { adapt, ..Params::default() } },
        PmCase { name: "dog hole, rot+scale+mirror", img, hole, w, h, p: Params { adapt: rot, ..Params::default() } },
        PmCase { name: "bricks", img: bricks(bw, bh), hole: disk(bw, bh, 430.0, 300.0, 120.0, 90.0), w: bw, h: bh, p: Params::default() },
    ]
}

/// (mean patch SSD, SSIM over the region, SSIM over the hole) of `b` against `a`.
fn pm_compare(c: &PmCase, a: &(Vec<u32>, patchmatch::MatchStats), b: &(Vec<u32>, patchmatch::MatchStats)) -> (f64, f64) {
    let (sa, sb) = (straight(&a.0), straight(&b.0));
    let whole = crate::engine::metrics::ssim(&sa, &sb, c.w, c.h, None);
    let hole = crate::engine::metrics::ssim(&sa, &sb, c.w, c.h, Some(&c.hole));
    (whole, hole)
}

#[test]
fn patchmatch_matches_the_cpu_quality() {
    let Some(g) = gpu() else { return };
    for c in pm_cases() {
        let run = |e: Engine, seed: u32| {
            let p = Params { seed, ..c.p };
            let (o, st, path) = patchmatch::complete_with(&c.img, &c.hole, c.w, c.h, None, &p, e);
            ((o, st), path)
        };
        let (cpu, _) = run(Engine::Cpu, 7);
        let (cpu2, _) = run(Engine::Cpu, 8);
        let (gpu, path) = run(Engine::Gpu(g.clone()), 7);
        assert_eq!(path, PathTaken::Gpu, "{}", c.name);
        let (w_cc, h_cc) = pm_compare(&c, &cpu, &cpu2);
        let (w_gc, h_gc) = pm_compare(&c, &cpu, &gpu);
        let ratio = gpu.1.mean_dist / cpu.1.mean_dist;
        eprintln!(
            "patchmatch {:<28} mean SSD cpu {:.0} gpu {:.0} ({:+.1} %), cpu seed 8 {:.0} | SSIM gpu-vs-cpu: region {w_gc:.4} hole {h_gc:.4}; cpu seed 8 vs 7: region {w_cc:.4} hole {h_cc:.4}",
            c.name,
            cpu.1.mean_dist,
            gpu.1.mean_dist,
            (ratio - 1.0) * 100.0,
            cpu2.1.mean_dist
        );
        assert!(ratio <= 1.02, "{}: GPU mean patch SSD {:.0} vs CPU {:.0}", c.name, gpu.1.mean_dist, cpu.1.mean_dist);
        assert!(w_gc >= 0.98, "{}: SSIM {w_gc:.4}", c.name);
    }
}

// ----- mip pyramids ---------------------------------------------------------------------

#[test]
fn gpu_mips_are_bit_exact_with_the_same_tile_sharing() {
    let Some(g) = gpu() else { return };
    let flat = TileGrid::filled(3000, 2100, px::pack(10, 200, 30, 255));
    for (name, grid) in [("dog", photo("dog_grass")), ("edge", edge_case()), ("flat", flat)] {
        let cpu = crate::engine::mips::build_mips(&grid);
        let gm = super::mipgen::build_mips(&g, &grid).unwrap();
        assert_eq!(cpu.len(), gm.len(), "{name}");
        for (l, (c, o)) in cpu.iter().zip(&gm).enumerate() {
            if let Some(d) = super::tests::diff_exact(c, o) {
                panic!("{name} level {}: {d}", l + 1);
            }
            for (tc, to) in c.tiles.iter().zip(&o.tiles) {
                assert_eq!(tc.is_none(), to.is_none(), "{name} level {}", l + 1);
                if let (Some(a), Some(b)) = (tc, to) {
                    assert_eq!(std::sync::Arc::ptr_eq(a, b), a.uniform.is_some(), "{name} level {}: sharing", l + 1);
                    assert_eq!(a.uniform, b.uniform);
                }
            }
        }
    }
    super::resident::clear(&g);
}

// ----- switching and fallback ---------------------------------------------------------------

#[test]
fn a_failed_gpu_falls_back_to_identical_cpu_results() {
    if gpu().is_none() {
        return;
    }
    // A private context (the shared one must stay healthy for other tests).
    let g = std::sync::Arc::new(super::Gpu::new().unwrap());
    g.fail("simulated device loss");
    let c = &heal_cases()[4];
    assert!(poisson::blend_rgba_gpu(&g, &c.src, &c.dst, &c.mask, c.w, c.h, c.cycles, Quality::Fast).is_err());
    let pm = &pm_cases()[0];
    let (cpu, _, _) = patchmatch::complete_with(&pm.img, &pm.hole, pm.w, pm.h, None, &pm.p, Engine::Cpu);
    let (out, _, path) = patchmatch::complete_with(&pm.img, &pm.hole, pm.w, pm.h, None, &pm.p, Engine::Gpu(g));
    assert!(matches!(path, PathTaken::GpuThenCpu(_)), "{path:?}");
    assert!(out == cpu, "fallback differs from the CPU");
}

#[test]
fn cost_model_keeps_small_retouching_on_the_cpu() {
    use super::cost::{OP_HEAL, OP_HEAL_EXACT, OP_PATCHMATCH, veto_op};
    let Some(g) = gpu() else { return };
    assert!(veto_op(&g, OP_HEAL, 106 * 106).is_some(), "100 px dab");
    assert!(veto_op(&g, OP_HEAL_EXACT, 1006 * 1006).is_none(), "1000 px dab, Exact");
    assert!(veto_op(&g, OP_PATCHMATCH, 200 * 200).is_some(), "small spot");
    assert!(veto_op(&g, OP_PATCHMATCH, 2618 * 1729).is_none(), "1 MP content-aware fill");
}
