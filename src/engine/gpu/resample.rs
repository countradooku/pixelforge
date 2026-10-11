//! Image Size, Super Zoom and transform resampling on the GPU.
//!
//! These aren't [`Filter`](crate::engine::filters::Filter)s: the CPU entry
//! points ([`crate::engine::ops::resize_with`],
//! [`crate::engine::ops::upscale_bicubic`], [`crate::engine::crop::warp_affine`],
//! [`crate::engine::annot::warp_grid`]) call [`switch`], which follows the
//! same rules as [`super::apply_filter`]: the `Resample` feature mode, the
//! cost model (per operation name), the path recorded for the Info panel,
//! and the CPU (the reference) on any error, with identical pixels.
//!
//! The source image comes from the resident cache; the output is produced in
//! bands of tile rows (two slots in flight, like the filter runner) and read
//! back into new tiles.
//!
//! Equality with the CPU:
//! - Bilinear Image Size: the CPU's per-tile coordinates and 8-bit
//!   fractions depend only on the output column / row, so they're tables
//!   built here with the CPU's exact float expressions: bit-exact.
//! - Nearest Neighbor, Bicubic (all three), Lanczos, Preserve Details 1 / 2:
//!   integer tap tables from `resample::Plan`, integer passes: bit-exact.
//! - Super Zoom's Catmull-Rom: per-column / per-row taps and float weights
//!   from the CPU's formulas; the float accumulation may contract into FMAs
//!   (≤ 1/255).
//! - Rotations / affine transforms (bilinear, Catmull-Rom, Lanczos-3) and the
//!   mip-aware perspective sampler: per-pixel float positions as the CPU
//!   computes them (per-tile footprints and mip levels from the CPU), precise
//!   sin and division; ≤ 1/255, or the bilinear sampler's 2/255 on ≤ 0.01 %.

use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use super::device::Gpu;
use super::exec::{self, Buf, GResult, GpuError, STORAGE, buffer, dispatch, encoder, groups, submit, table, wait};
use super::kernels::Temps;
use super::shaders::PipeKey;
use super::{Feature, Mode, PathTaken};
use crate::engine::crop::Interp;
use crate::engine::document::Layer;
use crate::engine::grid::{TS, TS_I, TileGrid, finish_tile};
use crate::engine::resample::{Plan, Resample, Taps};

// ----- the switch -------------------------------------------------------------------------

/// Whether `op` should run on the GPU: the context, or why not.
fn decide(op: &str, px: u64, bytes: u64) -> Result<Arc<Gpu>, String> {
    let m = super::feature_mode(Feature::Resample);
    if m == Mode::Off {
        return Err("GPU off".into());
    }
    let Some(g) = super::context(m == Mode::Force || super::blocking()) else {
        return Err(format!("GPU {}", super::describe()));
    };
    if m == Mode::Auto && let Some(why) = super::cost::veto_xop(&g, op, px, bytes) {
        return Err(why);
    }
    Ok(g)
}

/// Runs a resampling operation on the GPU when it's enabled, healthy and
/// predicted to win, else (or on any GPU error) on the CPU; records the
/// path. `px` is the output size, `bytes` the host <-> device traffic.
pub fn switch(op: &str, px: u64, bytes: u64, gpu_f: impl FnOnce(&Arc<Gpu>) -> GResult<TileGrid>, cpu_f: impl FnOnce() -> TileGrid) -> TileGrid {
    let t0 = Instant::now();
    let g = match decide(op, px, bytes) {
        Ok(g) => g,
        Err(why) => {
            let r = cpu_f();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            super::cost::observe_xop(op, px, bytes, ms, false);
            super::record(op, PathTaken::Cpu(why), ms);
            return r;
        }
    };
    let (r, path) = on(&g, gpu_f, cpu_f);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    if path == PathTaken::Gpu {
        super::cost::observe_xop(op, px, bytes, ms, true);
    }
    super::record(op, path, ms);
    r
}

/// Tries the GPU on `g` (tests and benchmarks call this directly), falling
/// back to the CPU for the whole operation on any error.
pub fn on(g: &Arc<Gpu>, gpu_f: impl FnOnce(&Arc<Gpu>) -> GResult<TileGrid>, cpu_f: impl FnOnce() -> TileGrid) -> (TileGrid, PathTaken) {
    if g.is_failed() {
        let why = g.failure().unwrap_or_else(|| "GPU failed".into());
        return (cpu_f(), PathTaken::Cpu(why));
    }
    match gpu_f(g) {
        Ok(r) => (r, PathTaken::Gpu),
        Err(e) => {
            if e.is_fault() {
                g.fail(e.to_string());
            }
            // Let anything still queued drain before falling back.
            let _ = g.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(exec::watchdog()) });
            (cpu_f(), PathTaken::Cpu(e.to_string()))
        }
    }
}

/// Rates measured with `--bench-gpu` on the reference machine.
pub fn prior(op: &str) -> Option<super::cost::Rates> {
    // GPU end to end (gather, upload, kernel, readback) and CPU, ns per
    // output pixel: Apple M5 Pro, quiet machine, averaged over the
    // `--bench-gpu 4096` and `16384` rows (reductions and enlargements share
    // a key, so their rates are blended).
    let r = |g: f64, c: f64| Some(super::cost::Rates { gpu_ns_px: g, cpu_ns_px: c });
    match op {
        "Image Size (Bilinear)" => r(0.20, 0.47),
        "Image Size (Nearest Neighbor (hard edges))" => r(0.30, 0.16),
        "Image Size (Bicubic (smooth gradients))" => r(0.25, 0.95),
        "Image Size (Bicubic Smoother (enlargement))" => r(0.24, 0.95),
        "Image Size (Bicubic Sharper (reduction))" => r(0.61, 1.95),
        "Image Size (Lanczos-3)" => r(0.55, 2.0),
        "Image Size (Preserve Details (enlargement))" => r(0.51, 3.6),
        "Image Size (Preserve Details 2.0)" => r(0.58, 4.6),
        "Super Zoom upscale" => r(0.33, 1.7),
        "Transform (Bilinear)" => r(0.28, 1.25),
        "Transform (Bicubic)" => r(0.40, 3.3),
        "Transform (Lanczos)" => r(1.03, 10.5),
        "Transform (mip bilinear)" => r(0.20, 1.7),
        _ => None,
    }
}

// ----- banded output ----------------------------------------------------------------------

/// One band of output: tile-area rows `y0..y0 + rows` (`tw` wide) go to `dst`.
pub struct Band<'a> {
    pub y0: u32,
    pub rows: u32,
    pub tw: u32,
    pub dst: &'a wgpu::Buffer,
}

struct Slot {
    dst: Buf,
    down: Option<Buf>,
    temps: Temps,
}

fn trace() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var_os("PF_GPU_TRACE").is_some())
}

/// Produces a `w x h` grid band by band: `encode` records the kernels that
/// write one band's rows into `Band::dst`; `row_bytes` is the scratch it
/// needs per output row (sizes the bands). Two bands are in flight.
pub fn produce(g: &Arc<Gpu>, w: u32, h: u32, row_bytes: u64, encode: impl Fn(&mut wgpu::CommandEncoder, &Band, &mut Temps) -> GResult<()>) -> GResult<TileGrid> {
    let mut out = TileGrid::new(w, h);
    let (cols, rows) = (out.cols, out.rows);
    let tw = cols * TS as u32;
    let per_tile_row = (tw as u64 * 4 + row_bytes) * TS as u64;
    let cap = g.max_buffer().min(g.budget() / 8).min(192 << 20);
    let band = ((cap / per_tile_row.max(1)) as u32).clamp(1, rows.max(1));
    if tw as u64 * 4 * TS as u64 > g.max_buffer() {
        return Err(GpuError::Vram);
    }
    let bands: Vec<(u32, u32)> = (0..rows).step_by(band as usize).map(|b| (b, (b + band).min(rows))).collect();
    let mappable = g.caps.mappable_primary;
    let size = tw as u64 * (band * TS as u32) as u64 * 4;
    let mut slots: Vec<Slot> = Vec::new();
    for _ in 0..bands.len().min(2) {
        slots.push(if mappable {
            Slot { dst: buffer(g, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ)?, down: None, temps: Temps::default() }
        } else {
            Slot {
                dst: buffer(g, size, STORAGE)?,
                down: Some(buffer(g, size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?),
                temps: Temps::default(),
            }
        });
    }
    let start = |b: usize, slot: &mut Slot| -> GResult<wgpu::SubmissionIndex> {
        if g.is_failed() {
            return Err(GpuError::Lost(g.failure().unwrap_or_default()));
        }
        let t = Instant::now();
        let (ty0, ty1) = bands[b];
        let band = Band { y0: ty0 * TS as u32, rows: (ty1 - ty0) * TS as u32, tw, dst: &slot.dst };
        let scopes = exec::Scopes::push(g);
        let mut enc = encoder(g);
        encode(&mut enc, &band, &mut slot.temps)?;
        if let Some(d) = &slot.down {
            enc.copy_buffer_to_buffer(&slot.dst, 0, d, 0, tw as u64 * band.rows as u64 * 4);
        }
        let idx = submit(g, enc);
        scopes.pop()?;
        if trace() {
            eprintln!("resample band {b}: encode+submit {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(idx)
    };
    let finish = |b: usize, idx: wgpu::SubmissionIndex, slot: &Slot, out: &mut TileGrid| -> GResult<()> {
        let t = Instant::now();
        wait(g, idx)?;
        let waited = t.elapsed().as_secs_f64() * 1e3;
        let (ty0, ty1) = bands[b];
        let n = tw as u64 * ((ty1 - ty0) * TS as u32) as u64 * 4;
        let read: &Buf = slot.down.as_ref().unwrap_or(&slot.dst);
        read.map(n, wgpu::MapMode::Read)?;
        let tiles: Vec<(usize, crate::engine::grid::TileRef)> = {
            let view = read.slice(..n).get_mapped_range();
            let px: &[u32] = bytemuck::cast_slice(&view);
            let jobs: Vec<(u32, u32)> = (ty0..ty1).flat_map(|ty| (0..cols).map(move |tx| (tx, ty))).collect();
            jobs.par_iter()
                .map(|&(tx, ty)| {
                    let mut o = Vec::with_capacity(TS * TS);
                    let oy = ((ty - ty0) * TS as u32) as usize;
                    let ox = (tx * TS as u32) as usize;
                    for y in 0..TS {
                        let s = (oy + y) * tw as usize + ox;
                        o.extend_from_slice(&px[s..s + TS]);
                    }
                    ((ty * cols + tx) as usize, finish_tile(o.into_boxed_slice()))
                })
                .collect()
        };
        read.unmap_now();
        for (i, t) in tiles {
            out.tiles[i] = t;
        }
        if trace() {
            eprintln!("resample band {b}: wait {waited:.2} ms, map+scatter {:.2} ms", t.elapsed().as_secs_f64() * 1e3 - waited);
        }
        Ok(())
    };
    let mut inflight: std::collections::VecDeque<(usize, wgpu::SubmissionIndex)> = Default::default();
    let mut next = 0usize;
    let r: GResult<()> = (|| {
        loop {
            while next < bands.len() && inflight.len() < slots.len() {
                let idx = start(next, &mut slots[next % 2])?;
                inflight.push_back((next, idx));
                next += 1;
            }
            let Some((b, idx)) = inflight.pop_front() else { return Ok(()) };
            finish(b, idx, &slots[b % 2], &mut out)?;
        }
    })();
    r.map(|_| out)
}

fn key(entry: &'static str) -> PipeKey {
    PipeKey { module: "resample", entry, f16: false, consts: Vec::new() }
}

/// Words in the module's uniform block (`array<vec4<u32>, 6>`).
const PARAMS: usize = 24;

/// [`dispatch`] of a `resample.wgsl` entry (params padded to the block).
fn run(g: &Gpu, enc: &mut wgpu::CommandEncoder, entry: &'static str, params: &[u32], bufs: &[(u32, &wgpu::Buffer)], groups: [u32; 3]) -> GResult<()> {
    let mut p = params.to_vec();
    p.resize(PARAMS, 0);
    dispatch(g, enc, None, &key(entry), &p, bufs, groups)
}

fn f(v: f32) -> u32 {
    v.to_bits()
}

fn i32s(v: &[i32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

// ----- Image Size -------------------------------------------------------------------------

/// Image Size on the GPU (see [`crate::engine::ops::resize_with`]).
pub fn resize(g: &Arc<Gpu>, layer: &Layer, nw: u32, nh: u32, method: Resample, noise: f32) -> GResult<TileGrid> {
    let g0 = &layer.grid;
    if method.resolve(g0.w, g0.h, nw, nh) == Resample::Bilinear {
        return resize_bilinear(g, layer, nw, nh);
    }
    let p = Plan::new(layer, nw, nh, method, noise);
    if let Some((cols, rows)) = &p.nn {
        let src = super::resident::image(g, g0)?;
        let (ta, tb) = (table(g, i32s(cols)), table(g, i32s(rows)));
        return produce(g, nw, nh, 0, |enc, b, _| {
            let params = [src.tw, 0, 0, b.tw, b.y0, b.rows];
            run(g, enc, "nearest", &params, &[(1, &src.buf), (2, b.dst), (3, &ta), (4, &tb)], [groups(b.tw, 16), groups(b.rows, 16), 1])
        });
    }
    let src = super::resident::image(g, layer.level(p.level))?;
    let (x, y) = (p.x.as_ref().unwrap(), p.y.as_ref().unwrap());
    let up = |t: &Taps| (table(g, i32s(&t.start)), table(g, i32s(&t.w)));
    let (xs, xw) = up(x);
    let (ys, yw) = up(y);
    let pd = p.method.has_noise();
    let (sxt, syt) = if pd { (Some(up(p.sx.as_ref().unwrap())), Some(up(p.sy.as_ref().unwrap()))) } else { (None, None) };
    let gain = pd.then(|| table(g, i32s(&p.gain)));
    let tw = p.tw;
    // Source rows per output row (bounds the intermediate's size).
    let spr = (src.h as f64 / nh as f64).max(1.0).ceil() as u64;
    let row_bytes = tw as u64 * (8 * (spr + 1) + if pd { 8 } else { 0 });
    // Separable resample of output rows `y0..y1` into `dst` (from its row 0).
    let sep = |enc: &mut wgpu::CommandEncoder, tx: (&wgpu::Buffer, &wgpu::Buffer, usize), ty: &Taps, tyb: (&wgpu::Buffer, &wgpu::Buffer), y0: u32, y1: u32, inter: &wgpu::Buffer, dst: &wgpu::Buffer| -> GResult<()> {
        let (r0, r1) = ty.span(y0 as usize, y1 as usize);
        let nr = (r1 - r0 + 1) as u32;
        let ph = [src.tw, src.w, tx.2 as u32, tw, r0 as u32, nr];
        run(g, enc, "sep_h", &ph, &[(1, &src.buf), (2, inter), (3, tx.0), (4, tx.1)], [groups(tw, 64), groups(nr, 4), 1])?;
        let pv = [0, src.h, ty.n as u32, tw, r0 as u32, y1 - y0, y0, 0];
        run(g, enc, "sep_v", &pv, &[(1, inter), (2, dst), (3, tyb.0), (4, tyb.1)], [groups(tw, 64), groups(y1 - y0, 4), 1])
    };
    let inter_rows = |t: &Taps, y0: u32, y1: u32| {
        let (r0, r1) = t.span(y0 as usize, y1 as usize);
        (r1 - r0 + 1) as u64
    };
    produce(g, nw, nh, row_bytes, |enc, b, temps| {
        let (y0, y1) = (b.y0, b.y0 + b.rows);
        if !pd {
            temps.ensure_sizes(g, &[inter_rows(y, y0, y1) * tw as u64 * 8])?;
            return sep(enc, (&xs, &xw, x.n), y, (&ys, &yw), y0, y1, temps.at(0), b.dst);
        }
        let (sx, sy) = (p.sx.as_ref().unwrap(), p.sy.as_ref().unwrap());
        let (sxt, syt) = (sxt.as_ref().unwrap(), syt.as_ref().unwrap());
        // L and S with a one-row apron.
        let (ry0, ry1) = (y0.saturating_sub(1), (y1 + 1).min(p.th));
        let inter = inter_rows(y, ry0, ry1).max(inter_rows(sy, ry0, ry1)) * tw as u64 * 8;
        let ls = (ry1 - ry0) as u64 * tw as u64 * 4;
        temps.ensure_sizes(g, &[inter, ls, ls])?;
        sep(enc, (&xs, &xw, x.n), y, (&ys, &yw), ry0, ry1, temps.at(0), temps.at(1))?;
        sep(enc, (&sxt.0, &sxt.1, sx.n), sy, (&syt.0, &syt.1), ry0, ry1, temps.at(0), temps.at(2))?;
        let params = [nw, nh, tw, ry0, y0, b.rows, (p.method == Resample::PreserveDetails2) as u32];
        run(g, enc, "pd", &params, &[(1, temps.at(1)), (5, temps.at(2)), (2, b.dst), (3, gain.as_ref().unwrap())], [groups(tw, 16), groups(b.rows, 16), 1])
    })
}

/// The bilinear resize's per-output-column (or row) source index and 8-bit
/// fraction, computed exactly as `ops::resize_bilinear_cpu` does per tile.
pub fn bilinear_axis(fx: f32, count: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(2 * count as usize);
    for xx in 0..count as i32 {
        let x0 = (xx / TS_I) * TS_I;
        let x = xx - x0;
        let u0 = ((x0 as f32 + 0.5) * fx - 0.5).floor() as i32 - 1;
        let u1 = ((x0 as f32 + TS as f32 + 0.5) * fx - 0.5).ceil() as i32 + 2;
        let bw = (u1 - u0) as usize;
        let u = (x0 as f32 + x as f32 + 0.5) * fx - 0.5 - u0 as f32;
        let ui = (u.floor() as usize).min(bw - 2);
        let wx = ((u - ui as f32).clamp(0.0, 1.0) * 255.0) as u32;
        out.push((u0 + ui as i32) as u32);
        out.push(wx);
    }
    out
}

/// `ops::resize` (mip-aware bilinear) on the GPU: bit-exact.
pub fn resize_bilinear(g: &Arc<Gpu>, layer: &Layer, nw: u32, nh: u32) -> GResult<TileGrid> {
    // The CPU's level choice and scale factors, verbatim.
    let g0 = &layer.grid;
    let sx = g0.w as f32 / nw as f32;
    let sy = g0.h as f32 / nh as f32;
    let s = sx.min(sy);
    let level = if s >= 2.0 { (s.log2().floor() as u32).min(layer.mips.len() as u32) } else { 0 };
    let lg = layer.level(level);
    let fl = (1u32 << level) as f32;
    let (fx, fy) = (sx / fl, sy / fl);
    let tw = nw.div_ceil(TS as u32) * TS as u32;
    let th = nh.div_ceil(TS as u32) * TS as u32;
    let src = super::resident::image(g, lg)?;
    let ta = table(g, bytemuck::cast_slice(&bilinear_axis(fx, tw)));
    let tb = table(g, bytemuck::cast_slice(&bilinear_axis(fy, th)));
    produce(g, nw, nh, 0, |enc, b, _| {
        let params = [src.tw, src.w, src.h, b.tw, b.y0, b.rows];
        run(g, enc, "bilin", &params, &[(1, &src.buf), (2, b.dst), (3, &ta), (4, &tb)], [groups(b.tw, 16), groups(b.rows, 16), 1])
    })
}

// ----- Super Zoom -------------------------------------------------------------------------

/// `ops::upscale_bicubic_cpu`'s Catmull-Rom weights.
#[inline]
fn cr(t: f32) -> [f32; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [(-t3 + 2.0 * t2 - t) * 0.5, (3.0 * t3 - 5.0 * t2 + 2.0) * 0.5, (-3.0 * t3 + 4.0 * t2 + t) * 0.5, (t3 - t2) * 0.5]
}

/// Super Zoom's per-column (or row) taps: 4 clamped source indices and 4
/// weights, exactly as `ops::upscale_bicubic_cpu` computes them per tile.
pub fn cubic_axis(fx: f32, len: u32, count: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(8 * count as usize);
    for xx in 0..count as i32 {
        let x0 = (xx / TS_I) * TS_I;
        let x = xx - x0;
        let u0 = ((x0 as f32 + 0.5) * fx - 0.5).floor() as i32 - 2;
        let u1 = ((x0 as f32 + TS as f32 + 0.5) * fx).ceil() as i32 + 3;
        let bw = (u1 - u0) as isize;
        let u = (x0 as f32 + x as f32 + 0.5) * fx - 0.5 - u0 as f32;
        let ui = u.floor() as isize;
        let w = cr(u - ui as f32);
        for i in 0..4 {
            let c = (ui + i as isize - 1).clamp(0, bw - 1) as i32;
            out.push((u0 + c).clamp(0, len as i32 - 1) as u32);
        }
        out.extend(w.iter().map(|v| v.to_bits()));
    }
    out
}

/// `ops::upscale_bicubic` on the GPU (within 1/255).
pub fn upscale_bicubic(g: &Arc<Gpu>, layer: &Layer, nw: u32, nh: u32) -> GResult<TileGrid> {
    let g0 = &layer.grid;
    let (fx, fy) = (g0.w as f32 / nw as f32, g0.h as f32 / nh as f32);
    let tw = nw.div_ceil(TS as u32) * TS as u32;
    let th = nh.div_ceil(TS as u32) * TS as u32;
    let src = super::resident::image(g, g0)?;
    let ta = table(g, bytemuck::cast_slice(&cubic_axis(fx, g0.w, tw)));
    let tb = table(g, bytemuck::cast_slice(&cubic_axis(fy, g0.h, th)));
    produce(g, nw, nh, 0, |enc, b, _| {
        let params = [src.tw, 0, 0, b.tw, b.y0, b.rows];
        run(g, enc, "cubic4", &params, &[(1, &src.buf), (2, b.dst), (3, &ta), (4, &tb)], [groups(b.tw, 16), groups(b.rows, 16), 1])
    })
}

// ----- transforms -------------------------------------------------------------------------

/// `crop::warp_affine` on the GPU (within 1/255).
pub fn warp_affine(g: &Arc<Gpu>, grid: &TileGrid, m: &[f32; 9], w: u32, h: u32, interp: Interp) -> GResult<TileGrid> {
    use crate::engine::crop::affine_footprint;
    let inv = crate::engine::filters::distort::mat_inv(m);
    let rad = interp.radius();
    let srect = grid.bounds();
    let out = TileGrid::new(w, h);
    // Per-tile source footprints, as the CPU computes them (bw = 0: empty).
    let mut tiles = Vec::with_capacity(out.tiles.len() * 4);
    for ty in 0..out.rows as i32 {
        for tx in 0..out.cols as i32 {
            let fr = affine_footprint(&inv, tx, ty, rad);
            if fr.intersects(&srect) {
                tiles.extend([fr.x0, fr.y0, fr.w(), fr.h()]);
            } else {
                tiles.extend([0, 0, 0, 0]);
            }
        }
    }
    let src = super::resident::image(g, grid)?;
    let ta = table(g, i32s(&tiles));
    let kind = match interp {
        Interp::Bilinear => 0u32,
        Interp::Bicubic => 1,
        Interp::Lanczos => 2,
    };
    let cols = out.cols;
    produce(g, w, h, 0, |enc, b, _| {
        let mut params: Vec<u32> = inv[..6].iter().map(|&v| f(v)).collect();
        params.extend([src.tw, src.w, src.h, kind, b.tw, b.y0, b.rows, cols]);
        run(g, enc, "affine", &params, &[(1, &src.buf), (2, b.dst), (3, &ta)], [groups(b.tw, 16), groups(b.rows, 16), 1])
    })
}

/// `annot::warp_grid` (`SmartObject::render`: mip-aware bilinear with
/// anti-aliased edges) on the GPU.
pub fn warp_grid(g: &Arc<Gpu>, grid: &TileGrid, m: &[f32; 9], w: u32, h: u32) -> GResult<TileGrid> {
    use crate::engine::layers::SmartObject;
    let inv = crate::engine::filters::distort::mat_inv(m);
    let nmips = crate::engine::mips::mip_count(grid.w, grid.h);
    // The smart object's bounds (tiles outside stay empty).
    let (sw, sh) = (grid.w as f32, grid.h as f32);
    let corners = [(0.0, 0.0), (sw, 0.0), (sw, sh), (0.0, sh)].map(|(x, y)| crate::engine::vector::p(x, y).xform(m));
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for q in corners {
        (x0, y0, x1, y1) = (x0.min(q.x), y0.min(q.y), x1.max(q.x), y1.max(q.y));
    }
    let bounds = crate::engine::grid::RectI::new(x0.floor() as i32 - 1, y0.floor() as i32 - 1, x1.ceil() as i32 + 1, y1.ceil() as i32 + 1);
    let out = TileGrid::new(w, h);
    let (cols, rows) = (out.cols, out.rows);
    let mut tiles = Vec::with_capacity((cols * rows * 2) as usize);
    let mut used: Vec<u32> = Vec::new();
    // Levels used per band of tile rows (filled when the band is known).
    let mut tile_levels = vec![u32::MAX; (cols * rows) as usize];
    for ty in 0..rows as i32 {
        for tx in 0..cols as i32 {
            if !TileGrid::tile_rect(tx, ty).intersects(&bounds) {
                tiles.extend([u32::MAX, 0]);
                continue;
            }
            let (scale, level) = SmartObject::tile_level(&inv, tx, ty, nmips);
            tiles.extend([level, scale.max(1e-3).to_bits()]);
            tile_levels[(ty as u32 * cols + tx as u32) as usize] = level;
            if !used.contains(&level) {
                used.push(level);
            }
        }
    }
    used.sort_unstable();
    let max_level = used.last().copied().unwrap_or(0);
    let levels = super::resident::levels(g, grid, max_level)?;
    let ta = table(g, bytemuck::cast_slice(&tiles));
    produce(g, w, h, 0, |enc, b, _| {
        let (ty0, ty1) = (b.y0 / TS as u32, (b.y0 + b.rows) / TS as u32);
        let mut band_levels: Vec<u32> =
            (ty0..ty1).flat_map(|ty| (0..cols).map(move |tx| (ty * cols + tx) as usize)).map(|i| tile_levels[i]).filter(|&l| l != u32::MAX).collect();
        band_levels.sort_unstable();
        band_levels.dedup();
        // A band with no active tile still needs its zeros.
        let passes = if band_levels.is_empty() { vec![u32::MAX] } else { band_levels };
        for (i, &lv) in passes.iter().enumerate() {
            let im = &levels[lv.min(max_level) as usize];
            let mut params: Vec<u32> = inv.iter().map(|&v| f(v)).collect();
            params.extend([f(sw), f(sh), im.tw, im.w, im.h, lv, (i == 0) as u32, f(1.0 / (1u32 << lv.min(31)) as f32), b.tw, b.y0, b.rows, cols]);
            run(g, enc, "smart", &params, &[(1, &im.buf), (2, b.dst), (3, &ta)], [groups(b.tw, 16), groups(b.rows, 16), 1])?;
        }
        Ok(())
    })
}
