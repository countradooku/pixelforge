//! Filter Gallery on the GPU against the CPU: at most 1/255 per channel and
//! mean ΔE2000 <= 0.2 on the photos and the edge-case grid, commits and
//! previews at mip levels 1-3.

use std::sync::Arc;

use crate::engine::filters::Filter;
use crate::engine::filters::gallery::G;
use crate::engine::gpu::run::RunOpts;
use crate::engine::gpu::tests::{cpu, diff_stats, edge_case, gpu, on_gpu, photo};
use crate::engine::gpu::{Gpu, PathTaken, preview};
use crate::engine::grid::{RectI, TileGrid};
use crate::engine::px;

pub fn defaults(g: G) -> [f32; 3] {
    let mut p = [0f32; 3];
    for (k, d) in g.params().iter().enumerate().take(3) {
        p[k] = d.default;
    }
    p
}

pub fn effect(g: G, p: [f32; 3], img: &TileGrid) -> Filter {
    Filter::Gallery { g, p, fg: [0, 0, 0], bg: [255, 255, 255], doc: (img.w, img.h) }
}

/// Max diff, mean ΔE and the number of pixels off by more than 1.
fn stats(o: &TileGrid, c: &TileGrid) -> (u8, f64, usize) {
    let (w, h) = (o.w as usize, o.h as usize);
    let mut pa = vec![0u32; w * h];
    let mut pb = vec![0u32; w * h];
    o.read_region(0, 0, w, h, &mut pa);
    c.read_region(0, 0, w, h, &mut pb);
    let over = pa.iter().zip(&pb).filter(|(x, y)| px::unpack(**x).iter().zip(px::unpack(**y)).any(|(p, q)| p.abs_diff(q) > 1)).count();
    let (mx, de) = diff_stats(o, c);
    (mx, de, over)
}

fn first_diff(o: &TileGrid, c: &TileGrid, lim: u8) -> Option<String> {
    let (w, h) = (o.w as usize, o.h as usize);
    let mut pa = vec![0u32; w * h];
    let mut pb = vec![0u32; w * h];
    o.read_region(0, 0, w, h, &mut pa);
    c.read_region(0, 0, w, h, &mut pb);
    let i = (0..w * h).find(|&i| px::unpack(pa[i]).iter().zip(px::unpack(pb[i])).any(|(p, q)| p.abs_diff(q) > lim))?;
    Some(format!("({}, {}) GPU {:?} CPU {:?}", i % w, i / w, px::unpack(pa[i]), px::unpack(pb[i])))
}

/// Commit on the GPU vs the CPU; returns (max, ΔE, pixels over 1).
pub fn compare(gpu: &Arc<Gpu>, g: G, p: [f32; 3], img: &TileGrid) -> (u8, f64, usize, Option<String>) {
    let f = effect(g, p, img);
    let c = cpu(img, &f, None);
    let (o, path) = on_gpu(gpu, img, &f, None, &RunOpts::default());
    assert_eq!(path, PathTaken::Gpu, "{}: {path:?}", g.title());
    let (mx, de, over) = stats(&o, &c);
    (mx, de, over, first_diff(&o, &c, 1))
}

/// Preview at a mip level vs `Filter::render` there: (max, ΔE).
pub fn compare_preview(gpu: &Arc<Gpu>, f: &Filter, level: u32, rect: RectI) -> (u8, f64) {
    let mut layer = crate::engine::document::Layer::new(1, "t", edge_case());
    layer.mips = crate::engine::mips::build_mips(&layer.grid);
    let (w, h) = (rect.w() as usize, rect.h() as usize);
    let mut c = vec![0u32; w * h];
    f.render(&crate::engine::filters::Src::layer(&layer), level, rect.x0, rect.y0, w, h, &mut c, &mut Vec::new());
    let o = preview::render(gpu, f, &layer, level, rect).unwrap_or_else(|e| panic!("{f:?} level {level}: {e}"));
    let mut mx = 0u8;
    for (a, b) in o.iter().zip(&c) {
        for (x, y) in px::unpack(*a).iter().zip(px::unpack(*b)) {
            mx = mx.max(x.abs_diff(y));
        }
    }
    let rgb = |v: &[u32]| -> Vec<[u8; 3]> {
        v.iter()
            .map(|&q| {
                let s = px::unpremultiply(q);
                [s[0], s[1], s[2]]
            })
            .collect()
    };
    (mx, crate::engine::metrics::mean_delta_e(&rgb(&o), &rgb(&c), None))
}

const PREVIEWS: [(u32, (i32, i32, i32, i32)); 3] = [(1, (-3, 5, 400, 300)), (2, (37, 0, 260, 197)), (3, (0, 0, 127, 98))];

/// Every check of one effect at one parameter set; panics on failure.
pub fn check(gpu: &Arc<Gpu>, g: G, p: [f32; 3]) {
    for (name, img) in [("dog", photo("dog_grass")), ("portrait", photo("portrait_hair")), ("edge", edge_case())] {
        let (mx, de, over, at) = compare(gpu, g, p, &img);
        assert!(mx <= 1 && de <= 0.2, "{} {p:?} on {name}: max {mx}, ΔE {de:.4}, {over} px over 1, first at {at:?}", g.title());
    }
    let e = edge_case();
    let f = effect(g, p, &e);
    for (level, (x, y, w, h)) in PREVIEWS {
        let (mx, de) = compare_preview(gpu, &f, level, RectI::new(x, y, x + w, y + h));
        assert!(mx <= 1 && de <= 0.2, "{} {p:?} preview level {level}: max {mx}, ΔE {de:.4}", g.title());
    }
}

#[test]
fn gallery_effects_match_the_cpu() {
    let Some(gpu) = gpu() else { return };
    for g in G::ALL {
        if !super::gallery::fx::supported(g) {
            continue;
        }
        check(&gpu, g, defaults(g));
    }
}

/// Extreme and mixed settings of every effect on a crop of the edge-case
/// grid (alpha ramp, transparent band, flat blocks), commits and previews.
#[test]
fn gallery_extreme_settings_match_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let e = crop_of(&edge_case(), 200, 150, 640, 480);
    for g in G::ALL {
        if !super::gallery::fx::supported(g) {
            continue;
        }
        for p in param_sets(g).into_iter().skip(1).take(5) {
            let (mx, de, over, at) = compare(&gpu, g, p, &e);
            assert!(mx <= 1 && de <= 0.2, "{} {p:?}: max {mx}, ΔE {de:.4}, {over} px over 1, first at {at:?}", g.title());
            let f = effect(g, p, &e);
            for (level, (x, y, w, h)) in PREVIEWS {
                if super::gallery::plan(&gpu, &f, level, super::Target::Preview).is_none() {
                    continue;
                }
                let (mx, de) = compare_preview(&gpu, &f, level, RectI::new(x, y, x + w, y + h));
                assert!(mx <= 1 && de <= 0.2, "{} {p:?} preview level {level}: max {mx}, ΔE {de:.4}", g.title());
            }
        }
    }
}

/// A selection and one-tile-row bands (every band seam). Cutout's CPU
/// blocks are two tiles tall, so a one-row band can't hold the CPU's window:
/// its float running sums start elsewhere and a few posterization band
/// edges flip (the CPU varies the same way between tilings); it gets the
/// mean bar only there.
#[test]
fn gallery_selection_and_band_seams() {
    use crate::engine::selection::Selection;
    let Some(gpu) = gpu() else { return };
    let img = photo("dog_grass");
    let ell = Selection::ellipse(RectI::new(150, 90, 900, 640));
    for g in G::ALL {
        if !super::gallery::fx::supported(g) {
            continue;
        }
        let f = effect(g, defaults(g), &img);
        for (name, sel, opts) in [("1-row bands", None, RunOpts { max_band_rows: Some(1), ..Default::default() }), ("ellipse", Some(&ell), RunOpts::default())] {
            let c = cpu(&img, &f, sel);
            let (o, path) = on_gpu(&gpu, &img, &f, sel, &opts);
            assert_eq!(path, PathTaken::Gpu, "{} {name}", g.title());
            let (mx, de, over) = stats(&o, &c);
            let n = (img.w * img.h) as usize;
            if g == G::Cutout && sel.is_none() {
                assert!(de <= 0.2 && over * 1000 <= n, "{} {name}: max {mx}, ΔE {de:.4}, {over} px over 1", g.title());
            } else {
                assert!(mx <= 1 && de <= 0.2, "{} {name}: max {mx}, ΔE {de:.4}, {over} px over 1", g.title());
            }
        }
    }
}

/// Development table: every supported effect's numbers (`--ignored --nocapture`).
#[test]
#[ignore]
fn gallery_report() {
    let Some(gpu) = gpu() else { return };
    let only = std::env::var("PF_GALLERY_ONLY").ok();
    for g in G::ALL {
        if !super::gallery::fx::supported(g) || only.as_ref().is_some_and(|o| !g.title().to_lowercase().contains(&o.to_lowercase())) {
            continue;
        }
        let p = defaults(g);
        let mut line = format!("{:<18}", g.title());
        for (name, img) in [("dog", photo("dog_grass")), ("portrait", photo("portrait_hair")), ("edge", edge_case())] {
            let (mx, de, over, at) = compare(&gpu, g, p, &img);
            line += &format!(" | {name} max {mx} ΔE {de:.4} over {over}");
            if mx > 1 {
                line += &format!(" @ {}", at.unwrap_or_default());
            }
        }
        let e = edge_case();
        let f = effect(g, p, &e);
        for (level, (x, y, w, h)) in PREVIEWS {
            let (mx, de) = compare_preview(&gpu, &f, level, RectI::new(x, y, x + w, y + h));
            line += &format!(" | L{level} {mx} {de:.4}");
        }
        println!("{line}");
    }
}

/// Parameter sets of an effect: defaults, minima, maxima, midpoints and a
/// mix (every choice of a choice parameter is visited by the mixes).
pub fn param_sets(g: G) -> Vec<[f32; 3]> {
    let ps = g.params();
    let pick = |f: &dyn Fn(usize, &crate::engine::filters::gallery::GParam) -> f32| {
        let mut p = [0f32; 3];
        for (k, q) in ps.iter().enumerate().take(3) {
            p[k] = f(k, q);
        }
        p
    };
    let mut v = vec![
        defaults(g),
        pick(&|_, q| q.min),
        pick(&|_, q| q.max),
        pick(&|_, q| ((q.min + q.max) * 0.5).round()),
        pick(&|k, q| if k % 2 == 0 { q.min } else { q.max }),
        pick(&|k, q| if k % 2 == 1 { q.min } else { q.max }),
    ];
    for (k, q) in ps.iter().enumerate() {
        if !q.choices.is_empty() {
            for c in 0..q.choices.len() {
                let mut p = defaults(g);
                p[k] = c as f32;
                v.push(p);
            }
        }
    }
    v
}

/// Sweep (`--ignored --nocapture`, PF_GALLERY_ONLY filters): every
/// parameter set on a photo crop and the edge-case grid, plus previews.
#[test]
#[ignore]
fn gallery_sweep() {
    let Some(gpu) = gpu() else { return };
    let only = std::env::var("PF_GALLERY_ONLY").ok();
    let dog = photo("dog_grass");
    let crop = crop_of(&dog, 150, 90, 700, 520);
    let e = edge_case();
    let mut bad = 0;
    for g in G::ALL {
        if !super::gallery::fx::supported(g) || only.as_ref().is_some_and(|o| !g.title().to_lowercase().contains(&o.to_lowercase())) {
            continue;
        }
        for p in param_sets(g) {
            let mut worst = (0u8, 0f64, String::new());
            for (name, img) in [("crop", &crop), ("edge", &e)] {
                let (mx, de, over, at) = compare(&gpu, g, p, img);
                if mx > worst.0 || de > worst.1 {
                    worst = (mx.max(worst.0), de.max(worst.1), format!("{name} over {over} {}", at.unwrap_or_default()));
                }
            }
            let f = effect(g, p, &e);
            for (level, (x, y, w, h)) in PREVIEWS {
                if super::gallery::plan(&gpu, &f, level, super::Target::Preview).is_none() {
                    continue;
                }
                let (mx, de) = compare_preview(&gpu, &f, level, RectI::new(x, y, x + w, y + h));
                if mx > worst.0 || de > worst.1 {
                    worst = (mx.max(worst.0), de.max(worst.1), format!("preview L{level}"));
                }
            }
            let ok = worst.0 <= 1 && worst.1 <= 0.2;
            if !ok {
                bad += 1;
            }
            println!("{} {:<18} {p:?}: max {} ΔE {:.4} {}", if ok { "ok  " } else { "FAIL" }, g.title(), worst.0, worst.1, worst.2);
        }
    }
    println!("{bad} failing parameter sets");
}

/// A `w x h` grid holding `src`'s pixels from (x0, y0).
pub fn crop_of(src: &TileGrid, x0: i32, y0: i32, w: u32, h: u32) -> TileGrid {
    use crate::engine::grid::TS;
    let mut buf = vec![0u32; (w * h) as usize];
    src.read_region(x0, y0, w as usize, h as usize, &mut buf);
    let mut g = TileGrid::new(w, h);
    g.tiles = crate::engine::grid::build_grid(w, h, |tx, ty| {
        let mut t = vec![0u32; TS * TS].into_boxed_slice();
        for y in 0..TS {
            for x in 0..TS {
                let (gx, gy) = (tx as u32 * 256 + x as u32, ty as u32 * 256 + y as u32);
                if gx < w && gy < h {
                    t[y * TS + x] = buf[(gy * w + gx) as usize];
                }
            }
        }
        crate::engine::grid::finish_tile(t)
    })
    .tiles;
    g
}

