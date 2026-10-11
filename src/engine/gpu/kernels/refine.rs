//! Select and Mask on the GPU, bit-exact with `engine::refine` (all integer:
//! box-sum guided filter, band weights, median, `gauss_plane` feather via the
//! `blur` module, LUTs, decontamination).
//!
//! Runs in bands of tile rows covering the tiles whose apron isn't uniform,
//! each band padded by `Params::pad()` (clamped reads, like the CPU blocks),
//! so the centers equal the CPU's whatever the band height. Same switch as
//! `gpu::apply_filter`: `Feature::SelectMask` mode, the cost model in Auto,
//! `gpu::record`, and the CPU for anything the GPU can't finish.

use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use super::raw::PlaneBlur;
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{self, Buf, GResult, GpuError, STORAGE, Timer, buffer, dispatch, encoder, groups, submit, table, wait};
use crate::engine::gpu::shaders::PipeKey;
use crate::engine::gpu::{self as gpu, Feature, Mode, PathTaken, cost};
use crate::engine::grid::{TS_I, TileGrid};
use crate::engine::refine::{self, DECON_RADII, Params, Refined};
use crate::engine::selection::MaskGrid;

pub const OP: &str = "Select and Mask";

/// Rates from `--bench-gpu` on the reference machine (Apple M5 Pro), used
/// until this machine has measured its own.
pub const PRIOR: cost::Rates = cost::Rates { gpu_ns_px: 12.0, cpu_ns_px: 100.0 };

/// Select and Mask with the GPU switch: GPU when enabled and worth it, the
/// CPU otherwise or for whatever the GPU couldn't do.
pub fn apply(mask: &MaskGrid, img: &TileGrid, p: &Params) -> Refined {
    let t0 = Instant::now();
    if p.is_identity() {
        return refine::refine_cpu(mask, img, p);
    }
    let work = refine::todo_tiles(mask, p.pad()).iter().filter(|&&t| t).count() as u64 * 65536;
    let cpu = |why: String| {
        let r = refine::refine_cpu(mask, img, p);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        cost::observe_named(OP, PRIOR, work, ms, false);
        gpu::record(OP, PathTaken::Cpu(why), ms);
        r
    };
    let m = gpu::feature_mode(Feature::SelectMask);
    if m == Mode::Off {
        return cpu("GPU off".into());
    }
    let Some(g) = gpu::context(m == Mode::Force || gpu::blocking()) else {
        return cpu(format!("GPU {}", gpu::describe()));
    };
    if m == Mode::Auto && let Some(why) = cost::veto_named(&g, OP, PRIOR, work) {
        return cpu(why);
    }
    match run(&g, mask, img, p, None) {
        Ok((r, _)) => {
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            cost::observe_named(OP, PRIOR, work, ms, true);
            gpu::record(OP, PathTaken::Gpu, ms);
            r
        }
        Err(e) => {
            if e.is_fault() {
                g.fail(e.to_string());
            }
            let r = refine::refine_cpu(mask, img, p);
            gpu::record(OP, PathTaken::GpuThenCpu(e.to_string()), t0.elapsed().as_secs_f64() * 1e3);
            r
        }
    }
}

fn key(entry: &'static str) -> PipeKey {
    PipeKey { module: "refine", entry, f16: false, consts: Vec::new() }
}

/// Uniform words: w, h, r, pad, cw, ch, n, n2, eps, smart, amount, k.
type Pw = [u32; 12];

fn win(r: usize) -> u32 {
    ((2 * r + 1) * (2 * r + 1)) as u32
}

struct Bufs {
    m: Buf,
    img: Buf,
    guide: Buf,
    planes: [Buf; 4],
    color: Buf,
    v: [Buf; 5],
    out_m: Buf,
    out_c: Buf,
    read_m: Buf,
    read_c: Buf,
}

impl Bufs {
    fn new(g: &Arc<Gpu>, npx: u64, cpx: u64) -> GResult<Bufs> {
        let pl = || buffer(g, npx * 4, STORAGE);
        let vb = || buffer(g, npx * 16, STORAGE);
        let rd = || buffer(g, cpx * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        Ok(Bufs {
            m: pl()?,
            img: pl()?,
            guide: pl()?,
            planes: [pl()?, pl()?, pl()?, pl()?],
            color: pl()?,
            v: [vb()?, vb()?, vb()?, vb()?, vb()?],
            out_m: buffer(g, cpx * 4, STORAGE)?,
            out_c: buffer(g, cpx * 4, STORAGE)?,
            read_m: rd()?,
            read_c: rd()?,
        })
    }
}

/// Window sums of `src` (vec4) at radius r into `dst`, through `tmp`.
#[allow(clippy::too_many_arguments)]
fn box_sum(g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, mut timer: Option<&mut Timer>, w: u32, h: u32, r: usize, src: &wgpu::Buffer, tmp: &wgpu::Buffer, dst: &wgpu::Buffer) -> GResult<()> {
    let mut pv: Pw = [0; 12];
    pv[..3].copy_from_slice(&[w, h, r as u32]);
    // Outputs per thread: enough to amortize the 2r + 1 tap initial sum.
    let kv = (4 * r as u32).clamp(16, 256);
    pv[11] = kv;
    dispatch(g, enc, timer.as_deref_mut(), &key("bsum_v"), &pv, &[(3, src), (7, tmp)], [groups(w, 256), groups(h, kv), 1])?;
    let kh = (4 * r as u32).clamp(64, 512);
    pv[11] = kh;
    dispatch(g, enc, timer, &key("bsum_h"), &pv, &[(3, tmp), (7, dst)], [groups(w, kh), groups(h, 64), 1])
}

/// Records every stage for one padded band. Returns nothing; the centers end
/// up in `b.out_m` / `b.out_c`.
fn encode(g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, mut timer: Option<&mut Timer>, b: &Bufs, w: u32, h: u32, pad: u32, p: &Params) -> GResult<()> {
    let (cw, ch) = (w - 2 * pad, h - 2 * pad);
    let gr = [groups(w, 16), groups(h, 16), 1];
    let base: Pw = [w, h, 0, pad, cw, ch, 0, 0, 0, p.smart as u32, p.amount255(), 0];
    let un = super::point::unpremul(g);
    let v = &b.v;
    let t = &*v[4];
    dispatch(g, enc, timer.as_deref_mut(), &key("guide"), &base, &[(1, &b.img), (6, &b.guide), (8, &un)], gr)?;
    // Which buffer holds the current mask: None = the input, Some(i) = plane i.
    let mut cur: Option<usize> = None;
    let plane = |c: Option<usize>| -> &wgpu::Buffer { c.map_or(&*b.m, |i| &*b.planes[i]) };
    let er = p.edge_r();
    if er > 0 {
        let (rs, r2, rg) = (p.smart_r(), p.key_r(), p.gf_r());
        dispatch(g, enc, timer.as_deref_mut(), &key("prep_bin"), &base, &[(1, &b.m), (2, &b.guide), (7, &v[0])], gr)?;
        box_sum(g, enc, timer.as_deref_mut(), w, h, er, &v[0], t, &v[1])?;
        if p.smart {
            box_sum(g, enc, timer.as_deref_mut(), w, h, rs, &v[0], t, &v[2])?;
        }
        let mut pk = base;
        pk[6] = win(er);
        dispatch(g, enc, timer.as_deref_mut(), &key("prep_key"), &pk, &[(2, &b.guide), (3, &v[1]), (7, &v[0])], gr)?;
        box_sum(g, enc, timer.as_deref_mut(), w, h, r2, &v[0], t, &v[3])?;
        dispatch(g, enc, timer.as_deref_mut(), &key("prep_gf"), &base, &[(1, &b.m), (2, &b.guide), (3, &v[3]), (7, &v[0])], gr)?;
        box_sum(g, enc, timer.as_deref_mut(), w, h, rg, &v[0], t, &v[3])?;
        let mut pg = base;
        pg[6] = win(rg);
        pg[8] = p.eps_q() as u32;
        dispatch(g, enc, timer.as_deref_mut(), &key("gf_ab"), &pg, &[(3, &v[3]), (7, &v[0])], gr)?;
        box_sum(g, enc, timer.as_deref_mut(), w, h, rg, &v[0], t, &v[3])?;
        let mut pe = pg;
        pe[7] = win(rs);
        pe[11] = win(er);
        let vc = if p.smart { &v[2] } else { &v[1] };
        dispatch(g, enc, timer.as_deref_mut(), &key("edge_out"), &pe, &[(1, &b.m), (2, &b.guide), (3, &v[3]), (4, &v[1]), (5, vc), (6, &b.planes[0])], gr)?;
        cur = Some(0);
    }
    let rs = p.smooth_r();
    if rs > 0 {
        let mut pm = base;
        pm[2] = rs as u32;
        dispatch(g, enc, timer.as_deref_mut(), &key("median"), &pm, &[(1, plane(cur)), (6, &b.planes[1])], gr)?;
        cur = Some(1);
    }
    if p.feather >= 0.3 {
        let pb = PlaneBlur::for_sigma(p.feather);
        if let Some(i) = pb.encode(g, enc, timer.as_deref_mut(), w, h, plane(cur), [&b.planes[2], &b.planes[3]])? {
            cur = Some(2 + i);
        }
    }
    if let Some(l) = p.lut() {
        let lt = table(g, &l);
        let to = if cur == Some(0) { 1 } else { 0 };
        dispatch(g, enc, timer.as_deref_mut(), &key("lut"), &base, &[(1, plane(cur)), (6, &b.planes[to]), (8, &lt)], gr)?;
        cur = Some(to);
    }
    let cgr = [groups(cw, 16), groups(ch, 16), 1];
    dispatch(g, enc, timer.as_deref_mut(), &key("crop"), &base, &[(1, plane(cur)), (6, &b.out_m)], cgr)?;
    if p.decon() {
        dispatch(g, enc, timer.as_deref_mut(), &key("dc_prep"), &base, &[(1, &b.img), (2, plane(cur)), (6, &b.color), (7, &v[0]), (8, &un)], gr)?;
        for &r in &DECON_RADII {
            // Vertical sums, then horizontal sums fused with the pick.
            let mut pv = base;
            pv[2] = r as u32;
            let kv = (4 * r as u32).clamp(16, 256);
            pv[11] = kv;
            dispatch(g, enc, timer.as_deref_mut(), &key("bsum_v"), &pv, &[(3, &v[0]), (7, t)], [groups(w, 256), groups(h, kv), 1])?;
            let kh = (4 * r as u32).clamp(64, 512);
            pv[11] = kh;
            dispatch(g, enc, timer.as_deref_mut(), &key("bsum_h_pick"), &pv, &[(3, t), (6, &b.color)], [groups(w, kh), groups(h, 64), 1])?;
        }
        dispatch(g, enc, timer, &key("dc_out"), &base, &[(1, &b.img), (2, plane(cur)), (9, &b.color), (6, &b.out_c), (8, &un)], cgr)?;
    }
    Ok(())
}

/// The GPU run over every band. Returns the result and the summed
/// timestamp-query kernel time (when available and `timed`).
pub fn run(g: &Arc<Gpu>, mask: &MaskGrid, img: &TileGrid, p: &Params, max_rows: Option<u32>) -> GResult<(Refined, Option<f64>)> {
    run_timed(g, mask, img, p, max_rows, false)
}

pub fn run_timed(g: &Arc<Gpu>, mask: &MaskGrid, img: &TileGrid, p: &Params, max_rows: Option<u32>, timed: bool) -> GResult<(Refined, Option<f64>)> {
    if g.is_failed() {
        return Err(GpuError::Lost(g.failure().unwrap_or_default()));
    }
    let pad = p.pad();
    let todo = refine::todo_tiles(mask, pad);
    let (cols, rows) = (mask.cols as i32, mask.rows as i32);
    let row_span = |ty: i32| -> Option<(i32, i32)> {
        let xs: Vec<i32> = (0..cols).filter(|&tx| todo[(ty * cols + tx) as usize]).collect();
        Some((*xs.first()?, *xs.last()? + 1))
    };
    let spans: Vec<Option<(i32, i32)>> = (0..rows).map(row_span).collect();
    // Band height: 120 bytes per padded pixel within half the budget, vec4
    // buffers within the binding limit.
    let full_w = (cols * TS_I) as u64 + 2 * pad as u64;
    let npx_rows = |n: u32| full_w * (n as u64 * 256 + 2 * pad as u64);
    let mut cap = max_rows.unwrap_or(u32::MAX).min(rows as u32).max(1);
    while cap > 1 && (npx_rows(cap) * 16 > g.max_buffer().min(512 << 20) || npx_rows(cap) * 120 > g.budget() / 2) {
        cap = cap.div_ceil(2);
    }
    if npx_rows(cap) * 16 > g.max_buffer() {
        return Err(GpuError::Vram);
    }
    let mut bands: Vec<(i32, i32, i32, i32)> = Vec::new();
    let mut ty = 0;
    while ty < rows {
        if spans[ty as usize].is_none() {
            ty += 1;
            continue;
        }
        let (mut tx0, mut tx1) = spans[ty as usize].unwrap();
        let start = ty;
        ty += 1;
        while ty < rows && ty - start < cap as i32 {
            match spans[ty as usize] {
                Some((a, b)) => {
                    tx0 = tx0.min(a);
                    tx1 = tx1.max(b);
                    ty += 1;
                }
                None => break,
            }
        }
        bands.push((tx0, start, tx1, ty));
    }
    let max_npx = bands.iter().map(|b| ((b.2 - b.0) * TS_I) as u64 + 2 * pad as u64).max().unwrap_or(0) * npx_rows(cap) / full_w;
    let max_cpx = bands.iter().map(|b| ((b.2 - b.0) * (b.3 - b.1)) as u64 * 65536).max().unwrap_or(0);
    let mut parts = Vec::new();
    let mut kernel_ms = timed.then_some(0.0);
    if !bands.is_empty() {
        let bufs = Bufs::new(g, max_npx.max(64), max_cpx.max(64))?;
        for &(tx0, ty0, tx1, ty1) in &bands {
            let (m, im, bw, bh) = refine::gather(mask, img, pad, tx0, ty0, tx1, ty1);
            let (cw, ch) = (bw - 2 * pad, bh - 2 * pad);
            let cn = (cw * ch) as u64 * 4;
            let scopes = exec::Scopes::push(g);
            g.queue.write_buffer(&bufs.m, 0, bytemuck::cast_slice(&m));
            g.queue.write_buffer(&bufs.img, 0, bytemuck::cast_slice(&im));
            drop((m, im));
            let mut enc = encoder(g);
            let mut timer = if timed { Timer::new(g, 64) } else { None };
            encode(g, &mut enc, timer.as_mut(), &bufs, bw as u32, bh as u32, pad as u32, p)?;
            enc.copy_buffer_to_buffer(&bufs.out_m, 0, &bufs.read_m, 0, cn);
            if p.decon() {
                enc.copy_buffer_to_buffer(&bufs.out_c, 0, &bufs.read_c, 0, cn);
            }
            if let Some(t) = timer.as_mut() {
                t.resolve(g, &mut enc);
            }
            let idx = submit(g, enc);
            scopes.pop()?;
            wait(g, idx)?;
            if let (Some(t), Some(ms)) = (&timer, kernel_ms.as_mut()) {
                let passes = t.read_passes(g).unwrap_or_default();
                *ms += passes.iter().map(|p| p.1).sum::<f64>();
                if std::env::var_os("PF_GPU_TRACE").is_some() {
                    let mut agg: Vec<(String, f64)> = Vec::new();
                    for (l, v) in passes {
                        match agg.iter_mut().find(|a| a.0 == l) {
                            Some(a) => a.1 += v,
                            None => agg.push((l, v)),
                        }
                    }
                    eprintln!("select-mask band {ty0}..{ty1} ({bw} x {bh}): {}", agg.iter().map(|(l, v)| format!("{l} {v:.1}")).collect::<Vec<_>>().join(", "));
                }
            }
            let read = |b: &Buf| -> GResult<Vec<u32>> {
                b.map(cn, wgpu::MapMode::Read)?;
                let v: Vec<u32> = bytemuck::cast_slice(&b.slice(..cn).get_mapped_range()).to_vec();
                b.unmap_now();
                Ok(v)
            };
            let om = read(&bufs.read_m)?;
            let oc = if p.decon() { Some(read(&bufs.read_c)?) } else { None };
            parts.push(((tx0, ty0, tx1, ty1, cw), om, oc));
        }
    }
    let scattered: Vec<_> = parts
        .par_iter()
        .flat_map_iter(|((tx0, ty0, tx1, ty1, cw), om, oc)| refine::scatter(&todo, mask.cols, *tx0, *ty0, *tx1, *ty1, *cw, om, oc.as_deref()))
        .collect();
    Ok((refine::assemble(mask, img, p.decon(), scattered), kernel_ms))
}
