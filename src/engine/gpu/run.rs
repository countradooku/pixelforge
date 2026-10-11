//! The banded runner: a whole-layer filter on the GPU.
//!
//! The filtered region is cut into bands of tile rows. Each band is gathered
//! (with the filter's clamped apron) straight into mapped GPU memory, run, and
//! read back into new tiles. Two band slots alternate, so gathering band
//! k + 1 and scattering band k - 1 overlap the GPU running band k.
//!
//! Any failure (validation, out of memory, watchdog, device loss) stops the
//! GPU at that band; the caller computes the remaining rows on the CPU, so
//! the result is identical either way.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use rayon::prelude::*;

use super::device::Gpu;
use super::exec::{self, Buf, GResult, GpuError, STORAGE, Timer, buffer, encoder, submit, wait};
use super::kernels::{Input, Io, Kernel, Temps};
use crate::engine::filters::Filter;
use crate::engine::grid::{TS, TS_I, TileGrid, TileRef, finish_tile};
use crate::engine::ops::{Progress, TileJob, TilePlanner};
use crate::engine::selection::{Selection, TileCoverage};

pub(super) fn trace() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var_os("PF_GPU_TRACE").is_some())
}

/// Knobs for tests and benchmarks.
#[derive(Clone, Debug, Default)]
pub struct RunOpts {
    /// Destroy the device when this band starts (simulated device loss).
    pub lose_device_at_band: Option<usize>,
    /// Upper bound on band height in tile rows.
    pub max_band_rows: Option<u32>,
    /// Override mappable-primary-buffer use (None = when available).
    pub mappable: Option<bool>,
}

/// The GPU's part of a run: the grid with every finished band, and where it
/// stopped if it didn't finish.
pub struct Outcome {
    pub grid: TileGrid,
    /// First tile row the GPU did not finish, with the reason.
    pub stopped: Option<(i32, GpuError)>,
    pub bands: usize,
    /// Wall time of the GPU work, its upload/readback share, and the timed
    /// kernel passes (when timestamp queries exist).
    pub kernel_ms: Option<f64>,
}

struct Slot {
    /// Mappable input (staging, or the storage buffer itself); a dummy for
    /// source-reading kernels.
    up: Buf,
    /// Storage input, when `up` is a separate staging buffer.
    src: Option<Buf>,
    /// Storage output.
    dst: Buf,
    /// Mappable output, when `dst` isn't mappable.
    down: Option<Buf>,
    temps: Temps,
}

struct Band {
    ty0: i32,
    ty1: i32,
    pw: u32,
    ph: u32,
    timer: Option<Timer>,
}

#[allow(clippy::too_many_arguments)]
pub fn run_filter(
    g: &Arc<Gpu>,
    grid: &TileGrid,
    filter: &Filter,
    kernel: &dyn Kernel,
    sel: Option<&Selection>,
    progress: &Progress,
    opts: &RunOpts,
    timed: bool,
) -> Outcome {
    let img = grid.bounds();
    let region = sel.map_or(img, |s| s.effect_bounds(img));
    let mut out = grid.clone();
    if region.is_empty() {
        return Outcome { grid: out, stopped: None, bands: 0, kernel_ms: Some(0.0) };
    }
    let (tx0, ty0, tx1, ty1) = region.tile_span(grid.cols, grid.rows);
    let input = kernel.input();
    // Source kernels read the whole (resident) image; padded kernels read a
    // band, copied on the GPU when the image is already resident.
    let source: Vec<Arc<super::resident::Image>> = match input {
        Input::Source { mips } => match super::resident::levels(g, grid, mips) {
            Ok(l) => l,
            Err(e) => return Outcome { grid: out, stopped: Some((ty0, e)), bands: 0, kernel_ms: None },
        },
        Input::Padded => super::resident::cached(g, grid).into_iter().collect(),
    };
    let from_source = !source.is_empty();
    let pad = if matches!(input, Input::Source { .. }) { 0 } else { filter.pad(0) as u32 };
    let cw = ((tx1 - tx0) * TS_I) as u32;
    let pw = cw + 2 * pad;
    // Band height: the padded input, output and temporaries of two slots
    // within a quarter of the budget, and each buffer within binding limits.
    let per_row = |rows: u32| (pw as u64) * (rows as u64 * TS as u64 + 2 * pad as u64) * 4;
    let bufs_per_slot = 2 + kernel.temps() as u64 + 1;
    let mut rows = (ty1 - ty0) as u32;
    if let Some(m) = opts.max_band_rows {
        rows = rows.min(m.max(1));
    }
    let cap = g.max_buffer().min(g.budget() / (4 * bufs_per_slot)).min(256 << 20);
    while rows > 1 && per_row(rows) > cap {
        rows = rows.div_ceil(2);
    }
    // Kernels that replay the CPU's blocks of tiles want bands made of
    // whole blocks.
    let kb = kernel.band_multiple().max(1);
    if rows > kb {
        rows -= rows % kb;
    }
    if per_row(rows) > g.max_buffer() {
        return Outcome { grid: out, stopped: Some((ty0, GpuError::Vram)), bands: 0, kernel_ms: None };
    }
    let bands: Vec<(i32, i32)> = (ty0..ty1).step_by(rows as usize).map(|b| (b, (b + rows as i32).min(ty1))).collect();
    progress.total.store(((tx1 - tx0) * (ty1 - ty0)).max(1) as usize, Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);

    let mappable = opts.mappable.unwrap_or(true) && g.caps.mappable_primary;
    let size = if matches!(input, Input::Source { .. }) { 256 } else { per_row(rows) };
    let center = cw as u64 * rows as u64 * TS as u64 * 4;
    let make_slot = || -> GResult<Slot> {
        Ok(if mappable {
            Slot {
                up: buffer(g, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_WRITE)?,
                src: None,
                dst: buffer(g, center, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ)?,
                down: None,
                temps: Temps::default(),
            }
        } else {
            Slot {
                up: buffer(g, size, wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC)?,
                src: if from_source { None } else { Some(buffer(g, size, STORAGE)?) },
                dst: buffer(g, center, STORAGE)?,
                down: Some(buffer(g, center, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?),
                temps: Temps::default(),
            }
        })
    };
    let mut slots: Vec<Slot> = Vec::new();
    for _ in 0..bands.len().min(2) {
        match make_slot() {
            Ok(s) => slots.push(s),
            Err(e) => return Outcome { grid: out, stopped: Some((ty0, e)), bands: 0, kernel_ms: None },
        }
    }
    let mut kernel_ms = if timed { Some(0.0) } else { None };
    let planner = TilePlanner::new(grid, filter, sel);
    let raw_edge = kernel.raw_edge_combine().then(|| filter.combiner());

    // Gathers band `b` into its slot and submits it.
    let start = |b: usize, slot: &mut Slot| -> GResult<(Band, wgpu::SubmissionIndex)> {
        if opts.lose_device_at_band == Some(b) {
            g.fail("simulated device loss");
            g.device.destroy();
        }
        if g.is_failed() {
            return Err(GpuError::Lost(g.failure().unwrap_or_default()));
        }
        let t_start = std::time::Instant::now();
        let (by0, by1) = bands[b];
        let ph = (by1 - by0) as u32 * TS as u32 + 2 * pad;
        let n = pw as u64 * ph as u64 * 4;
        let scopes = exec::Scopes::push(g);
        let (x0, y0) = (tx0 * TS_I - pad as i32, by0 * TS_I - pad as i32);
        // Like `Filter::render`: point filters read the tiles as they are
        // (edge-tile padding included), local ones clamp to the image.
        let clamped = pad > 0 || filter.class() != crate::engine::filters::Class::Point;
        let mut enc = encoder(g);
        if !from_source {
            slot.up.map(n, wgpu::MapMode::Write)?;
            {
                let mut view = slot.up.slice(..n).get_mapped_range_mut();
                let mut wo = view.slice(..);
                // SAFETY: the mapping is `n` bytes, u32-aligned, alive until
                // `view` drops, and nothing else touches it meanwhile. The
                // gather reads back a little of what it writes (edge
                // clamping), which is fine for mapped memory, just slow on
                // write-combined heaps.
                let px: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(wo.as_raw_ptr().as_ptr() as *mut u32, n as usize / 4) };
                let strip = 32usize;
                px.par_chunks_mut(pw as usize * strip).enumerate().for_each(|(i, chunk)| {
                    let h = chunk.len() / pw as usize;
                    if clamped {
                        grid.read_region_clamped(x0, y0 + (i * strip) as i32, pw as usize, h, chunk);
                    } else {
                        grid.read_region(x0, y0 + (i * strip) as i32, pw as usize, h, chunk);
                    }
                });
            }
            slot.up.unmap_now();
        }
        let t_gather = t_start.elapsed().as_secs_f64() * 1e3;
        slot.temps.ensure(g, kernel.temps(), n)?;
        let src: &wgpu::Buffer = match (&slot.src, input, from_source) {
            (_, Input::Source { .. }, _) => &source[0].buf,
            (_, Input::Padded, true) => {
                // The image is resident: cut the padded band out on the GPU.
                let dst: &wgpu::Buffer = slot.src.as_deref().unwrap_or(&slot.up);
                let im = &source[0];
                region_copy(g, &mut enc, im, (x0, y0, pw, ph), clamped, dst)?;
                dst
            }
            (Some(s), Input::Padded, false) => {
                enc.copy_buffer_to_buffer(&slot.up, 0, s, 0, n);
                s
            }
            (None, Input::Padded, false) => &slot.up,
        };
        let io = Io { src, dst: &slot.dst, pw, ph, pad, level: 0, origin: (x0, y0), img: (grid.w, grid.h), source: &source };
        let mut timer = if timed { Timer::new(g, 16) } else { None };
        kernel.encode(g, &mut enc, &io, &mut slot.temps, timer.as_mut())?;
        let cn = io.cw() as u64 * io.ch() as u64 * 4;
        if let Some(d) = &slot.down {
            enc.copy_buffer_to_buffer(&slot.dst, 0, d, 0, cn);
        }
        if let Some(t) = timer.as_mut() {
            t.resolve(g, &mut enc);
        }
        let idx = submit(g, enc);
        scopes.pop()?;
        if trace() {
            eprintln!("gpu band {b}: gather {t_gather:.2} ms, encode+submit {:.2} ms", t_start.elapsed().as_secs_f64() * 1e3 - t_gather);
        }
        Ok((Band { ty0: by0, ty1: by1, pw, ph, timer }, idx))
    };

    // Waits for a band and writes its tiles.
    let finish = |band: &Band, idx: wgpu::SubmissionIndex, slot: &mut Slot, out: &mut TileGrid, kernel_ms: &mut Option<f64>| -> GResult<()> {
        let t_wait = std::time::Instant::now();
        wait(g, idx)?;
        let waited = t_wait.elapsed().as_secs_f64() * 1e3;
        if let (Some(t), Some(ms)) = (&band.timer, kernel_ms.as_mut()) {
            *ms += t.read_ms(g).unwrap_or(0.0);
        }
        let read: &Buf = slot.down.as_ref().unwrap_or(&slot.dst);
        let cw = band.pw - 2 * pad;
        let ch = band.ph - 2 * pad;
        let n = cw as u64 * ch as u64 * 4;
        read.map(n, wgpu::MapMode::Read)?;
        let tiles: Vec<(usize, TileRef)> = {
            let view = read.slice(..n).get_mapped_range();
            let px: &[u32] = bytemuck::cast_slice(&view);
            let jobs: Vec<(i32, i32)> = (band.ty0..band.ty1).flat_map(|ty| (tx0..tx1).map(move |tx| (tx, ty))).collect();
            jobs.par_iter()
                .filter_map(|&(tx, ty)| {
                    let idx = grid.idx(tx, ty);
                    let t = match planner.job(tx, ty) {
                        TileJob::Keep => return None,
                        TileJob::Flat(t) => t,
                        TileJob::Compute(cov) => {
                            let (ox, oy) = (((tx - tx0) * TS_I) as usize, ((ty - band.ty0) * TS_I) as usize);
                            let orig = grid.tiles[idx].as_ref().map(|t| t.read());
                            let mut o = vec![0u32; TS * TS].into_boxed_slice();
                            for y in 0..TS {
                                let s = &px[(oy + y) * cw as usize + ox..(oy + y) * cw as usize + ox + TS];
                                let d = &mut o[y * TS..(y + 1) * TS];
                                if matches!(cov, TileCoverage::Full) {
                                    d.copy_from_slice(s);
                                } else {
                                    for x in 0..TS {
                                        let i = y * TS + x;
                                        d[x] = crate::engine::px::lerp(orig.as_ref().map_or(0, |t| t[i]), s[x], cov.at(i));
                                    }
                                }
                            }
                            if let Some(comb) = &raw_edge {
                                // Edge-tile padding: combine with the raw tile pixel.
                                let (x_in, y_in) = ((grid.w as i32 - tx * TS_I).clamp(0, TS_I) as usize, (grid.h as i32 - ty * TS_I).clamp(0, TS_I) as usize);
                                if x_in < TS || y_in < TS {
                                    for y in 0..TS {
                                        for x in 0..TS {
                                            if x >= x_in || y >= y_in {
                                                let i = y * TS + x;
                                                let raw = orig.as_ref().map_or(0, |t| t[i]);
                                                let s = px[(oy + y) * cw as usize + ox + x];
                                                o[i] = crate::engine::px::lerp(raw, comb.apply(raw, s), cov.at(i));
                                            }
                                        }
                                    }
                                }
                            }
                            finish_tile(o)
                        }
                    };
                    Some((idx, t))
                })
                .collect()
        };
        read.unmap_now();
        for (i, t) in tiles {
            out.tiles[i] = t;
        }
        if trace() {
            eprintln!("gpu band {}: wait {waited:.2} ms, map+scatter {:.2} ms", band.ty0, t_wait.elapsed().as_secs_f64() * 1e3 - waited);
        }
        progress.done.fetch_add(((band.ty1 - band.ty0) * (tx1 - tx0)) as usize, Ordering::Relaxed);
        Ok(())
    };

    let mut stopped = None;
    let mut inflight: std::collections::VecDeque<(usize, Band, wgpu::SubmissionIndex)> = Default::default();
    let mut next = 0usize;
    // Two slots: band k + 1 is gathered and queued before band k is waited
    // on, and band k is scattered while the GPU runs band k + 1.
    'run: loop {
        while next < bands.len() && inflight.len() < slots.len() {
            match start(next, &mut slots[next % 2]) {
                Ok((band, idx)) => inflight.push_back((next, band, idx)),
                Err(e) => {
                    // Resume from the first band that isn't finished.
                    let row = inflight.front().map_or(bands[next].0, |b| b.1.ty0);
                    stopped = Some((row, e));
                    break 'run;
                }
            }
            next += 1;
        }
        let Some((b, band, idx)) = inflight.pop_front() else { break };
        if let Err(e) = finish(&band, idx, &mut slots[b % 2], &mut out, &mut kernel_ms) {
            stopped = Some((band.ty0, e));
            break;
        }
    }
    // Let anything still queued drain before the slots go back to the pool.
    if stopped.is_some() {
        let _ = g.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(exec::watchdog()) });
    }
    Outcome { grid: out, stopped, bands: bands.len(), kernel_ms }
}

/// Copies a region of a resident image into a band buffer, clamped like
/// `read_region_clamped` or raw like `read_region`.
pub fn region_copy(g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, im: &super::resident::Image, r: (i32, i32, u32, u32), clamped: bool, dst: &wgpu::Buffer) -> GResult<()> {
    let key = super::shaders::PipeKey { module: "util", entry: "region", f16: false, consts: Vec::new() };
    let params = [im.tw, im.th, im.w, im.h, r.0 as u32, r.1 as u32, r.2, r.3, clamped as u32];
    exec::dispatch(g, enc, None, &key, &params, &[(1, &im.buf), (2, dst)], [exec::groups(r.2, 16), exec::groups(r.3, 16), 1])
}

/// The tile-row span of the region a filter touches.
pub fn region_rows(grid: &TileGrid, sel: Option<&Selection>) -> Option<(i32, i32)> {
    let img = grid.bounds();
    let region = sel.map_or(img, |s| s.effect_bounds(img));
    if region.is_empty() {
        return None;
    }
    let (_, ty0, _, ty1) = region.tile_span(grid.cols, grid.rows);
    Some((ty0, ty1))
}


/// Kernel time with the image already on the GPU (no upload or readback):
/// the whole image as one band, best of `reps`. Returns (wall ms of the
/// submission, timestamp-query ms if available).
pub fn resident_ms(g: &Arc<Gpu>, grid: &TileGrid, filter: &Filter, kernel: &dyn Kernel, reps: usize) -> GResult<(f64, Option<f64>)> {
    let pad = filter.pad(0) as u32;
    let (cw, ch) = (grid.cols * TS as u32, grid.rows * TS as u32);
    let (pw, ph) = (cw + 2 * pad, ch + 2 * pad);
    let n = pw as u64 * ph as u64 * 4;
    if n > g.max_buffer() {
        return Err(GpuError::Vram);
    }
    let src = buffer(g, n, STORAGE)?;
    let dst = buffer(g, cw as u64 * ch as u64 * 4, STORAGE)?;
    let mut temps = Temps::default();
    temps.ensure(g, kernel.temps(), n)?;
    let mut host = vec![0u32; (pw * ph) as usize];
    grid.read_region_clamped(-(pad as i32), -(pad as i32), pw as usize, ph as usize, &mut host);
    g.queue.write_buffer(&src, 0, bytemuck::cast_slice(&host));
    drop(host);
    let source: Vec<Arc<super::resident::Image>> = match kernel.input() {
        Input::Source { mips } => super::resident::levels(g, grid, mips)?,
        Input::Padded => Vec::new(),
    };
    let io = Io { src: &src, dst: &dst, pw, ph, pad, level: 0, origin: (-(pad as i32), -(pad as i32)), img: (grid.w, grid.h), source: &source };
    let mut best = f64::MAX;
    let mut best_ts = None::<f64>;
    for _ in 0..reps + 1 {
        let mut enc = encoder(g);
        let mut timer = Timer::new(g, 64);
        let t = std::time::Instant::now();
        kernel.encode(g, &mut enc, &io, &mut temps, timer.as_mut())?;
        if let Some(t) = timer.as_mut() {
            t.resolve(g, &mut enc);
        }
        let idx = submit(g, enc);
        wait(g, idx)?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        if let Some(passes) = timer.as_ref().and_then(|t| t.read_passes(g)) {
            let ms: f64 = passes.iter().map(|p| p.1).sum();
            if trace() {
                eprintln!("  passes: {}", passes.iter().map(|(l, v)| format!("{l} {v:.2}")).collect::<Vec<_>>().join(", "));
            }
            best_ts = Some(best_ts.map_or(ms, |b: f64| b.min(ms)));
        }
    }
    Ok((best, best_ts))
}
