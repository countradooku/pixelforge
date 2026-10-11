//! Edit ▸ Perspective Warp on the GPU: the source resident, the CPU-built
//! mesh node table as a storage buffer, the output in bands of tile rows
//! (two in flight: band k + 1 runs while band k is read back into tiles).
//! Bit-identical to `perspwarp::render_cpu` (see `wgsl/perspwarp.wgsl`).

use std::sync::Arc;

use rayon::prelude::*;

use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{self, Buf, GResult, GpuError, STORAGE, Timer, buffer, dispatch, encoder, groups, submit, wait};
use crate::engine::gpu::resident;
use crate::engine::gpu::shaders::PipeKey;
use crate::engine::grid::{TS, TileGrid, finish_tile};
use crate::engine::perspwarp::Table;

/// Timings of the last [`render`] (benchmarks).
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Timestamp-query kernel time, ms.
    pub kernel_ms: Option<f64>,
}

static LAST: std::sync::Mutex<Stats> = std::sync::Mutex::new(Stats { kernel_ms: None });

pub fn last_stats() -> Stats {
    *LAST.lock().unwrap()
}

/// `src` resampled through `table`. `private`: a context other than the
/// shared one (tests), which must not use the shared resident cache.
pub fn render(g: &Arc<Gpu>, src: &TileGrid, table: &Table, private: bool) -> GResult<TileGrid> {
    render_opts(g, src, table, private, None, false)
}

/// [`render`] with a band-height cap (tests) and pass timing (benchmarks).
pub fn render_opts(g: &Arc<Gpu>, src: &TileGrid, table: &Table, private: bool, max_band_rows: Option<u32>, timed: bool) -> GResult<TileGrid> {
    if g.is_failed() {
        return Err(GpuError::Lost(g.failure().unwrap_or_default()));
    }
    let img = if private { Arc::new(resident::upload(g, src)?) } else { resident::image(g, src)? };
    let mut out = TileGrid::new(table.w, table.h);
    let (cols, rows) = (out.cols, out.rows);
    let cw = cols * TS as u32;
    let row_bytes = cw as u64 * TS as u64 * 4;
    let cap = g.max_buffer().min(256 << 20);
    let mut band = (cap / row_bytes).max(1) as u32;
    if let Some(m) = max_band_rows {
        band = band.min(m.max(1));
    }
    let band = band.min(rows);
    if row_bytes * band as u64 > g.max_buffer() || groups(cw, 16) > g.limits.max_compute_workgroups_per_dimension {
        return Err(GpuError::Vram);
    }
    let nodes = exec::table(g, bytemuck::cast_slice(&table.nodes));
    let bands: Vec<(u32, u32)> = (0..rows).step_by(band as usize).map(|b| (b, (b + band).min(rows))).collect();
    let mappable = g.caps.mappable_primary;
    let n = row_bytes * band as u64;
    let mut slots: Vec<(Buf, Option<Buf>)> = Vec::new();
    for _ in 0..bands.len().min(2) {
        slots.push(if mappable {
            (buffer(g, n, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ)?, None)
        } else {
            (buffer(g, n, STORAGE)?, Some(buffer(g, n, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?))
        });
    }
    let key = PipeKey { module: "perspwarp", entry: "warp", f16: false, consts: Vec::new() };
    let mut kernel_ms: Option<f64> = None;
    let start = |b: usize, slot: &(Buf, Option<Buf>)| -> GResult<(wgpu::SubmissionIndex, Option<Timer>)> {
        let (r0, r1) = bands[b];
        let bh = (r1 - r0) * TS as u32;
        let scopes = exec::Scopes::push(g);
        let mut enc = encoder(g);
        let mut timer = if timed { Timer::new(g, 2) } else { None };
        let params = [0, r0 * TS as u32, cw, bh, table.nx, img.tw, img.w, img.h, 0, 0, 0, 0];
        dispatch(g, &mut enc, timer.as_mut(), &key, &params, &[(1, &img.buf), (2, &nodes), (3, &slot.0)], [groups(cw, 16), groups(bh, 16), 1])?;
        if let Some(d) = &slot.1 {
            enc.copy_buffer_to_buffer(&slot.0, 0, d, 0, cw as u64 * bh as u64 * 4);
        }
        if let Some(t) = timer.as_mut() {
            t.resolve(g, &mut enc);
        }
        let idx = submit(g, enc);
        scopes.pop()?;
        Ok((idx, timer))
    };
    let mut inflight: std::collections::VecDeque<(usize, wgpu::SubmissionIndex, Option<Timer>)> = Default::default();
    let mut next = 0;
    let result: GResult<()> = (|| {
        loop {
            while next < bands.len() && inflight.len() < slots.len() {
                let (idx, t) = start(next, &slots[next % 2])?;
                inflight.push_back((next, idx, t));
                next += 1;
            }
            let Some((b, idx, timer)) = inflight.pop_front() else { break };
            wait(g, idx)?;
            // (Metal sometimes reports a single pass as 0: unknown.)
            if let Some(ms) = timer.as_ref().and_then(|t| t.read_ms(g)).filter(|&ms| ms > 0.0) {
                kernel_ms = Some(kernel_ms.unwrap_or(0.0) + ms);
            }
            let (r0, r1) = bands[b];
            let slot = &slots[b % 2];
            let read = slot.1.as_ref().unwrap_or(&slot.0);
            let nb = cw as u64 * ((r1 - r0) * TS as u32) as u64 * 4;
            read.map(nb, wgpu::MapMode::Read)?;
            {
                let view = read.slice(..nb).get_mapped_range();
                let px: &[u32] = bytemuck::cast_slice(&view);
                let first = (r0 * cols) as usize;
                out.tiles[first..(r1 * cols) as usize].par_iter_mut().enumerate().for_each(|(k, slot)| {
                    let (tx, ty) = (k % cols as usize, k / cols as usize);
                    let mut t = vec![0u32; TS * TS].into_boxed_slice();
                    for y in 0..TS {
                        let s = (ty * TS + y) * cw as usize + tx * TS;
                        t[y * TS..(y + 1) * TS].copy_from_slice(&px[s..s + TS]);
                    }
                    *slot = finish_tile(t);
                });
            }
            read.unmap_now();
        }
        Ok(())
    })();
    if let Err(e) = result {
        let _ = g.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(exec::watchdog()) });
        return Err(e);
    }
    *LAST.lock().unwrap() = Stats { kernel_ms };
    Ok(out)
}
