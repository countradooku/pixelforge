//! Liquify on the GPU: the session's displacement field as GPU tile slots,
//! brush edits of the field, and the source (resident) resampled through it.
//!
//! [`crate::engine::liquify::Session`] owns one of these while the GPU path
//! is in use and keeps the CPU field as the reference and fallback: CPU
//! edits (and every freeze-mask edit) are uploaded before the next GPU
//! stroke, GPU strokes are downloaded when they end. Field edits and the
//! resample run the CPU's exact float expressions (see `wgsl/liquify.wgsl`).

use std::sync::Arc;

use rayon::prelude::*;

use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{self, Buf, GResult, GpuError, STORAGE, buffer, dispatch, encoder, groups, submit, wait};
use crate::engine::gpu::resident::{self, Image};
use crate::engine::gpu::shaders::PipeKey;
use crate::engine::grid::{RectI, TPX, TileGrid};

const NONE: u32 = u32::MAX;
const FIELD_BYTES: u64 = TPX as u64 * 8;
const FRZ_BYTES: u64 = TPX as u64;

/// One brush dab for the GPU (`liquify::Dab` in kernel terms).
#[derive(Clone, Copy, Debug)]
pub struct DabK {
    /// 0 Forward Warp, 1 Push Left, 2 Twirl CW, 3 Twirl CCW, 4 Pucker,
    /// 5 Bloat, 6 Reconstruct, 7 Smooth.
    pub mode: u32,
    /// The dab's rectangle, already clipped to the image.
    pub rect: RectI,
    pub c: (f32, f32),
    pub d: (f32, f32),
    pub inv_r2: f32,
    pub pressure: f32,
    pub rate: f32,
    /// Smooth's neighbor offset.
    pub k: i32,
}

/// GPU timings of the last [`LiquifyGpu::run`] (ms).
#[derive(Clone, Copy, Debug, Default)]
pub struct RunStats {
    /// Timestamp-query time of the passes.
    pub kernel_ms: Option<f64>,
    /// Submission to completion, wall time (no readback).
    pub wait_ms: f64,
}

pub struct LiquifyGpu {
    pub g: Arc<Gpu>,
    base: Arc<Image>,
    cols: u32,
    w: u32,
    h: u32,
    /// Slot of each tile (`NONE` = no field there yet).
    slot_of: Vec<u32>,
    slots: u32,
    cap: u32,
    index: Option<wgpu::Buffer>,
    index_dirty: bool,
    field: Option<Buf>,
    frz: Option<Buf>,
    /// New slots with no CPU content, zeroed at the start of the next run.
    clear: Vec<u32>,
    pub last: RunStats,
    /// Timestamp queries for benchmarks.
    pub timed: bool,
}

fn zeros() -> &'static [u8] {
    static Z: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    Z.get_or_init(|| vec![0u8; FIELD_BYTES as usize])
}

impl LiquifyGpu {
    /// Makes `base` resident (or finds it) and starts an empty field.
    /// `private`: a context other than the shared one (tests), which must
    /// not put its buffers in the shared resident cache.
    pub fn new(g: &Arc<Gpu>, base: &TileGrid, private: bool) -> GResult<Self> {
        if g.is_failed() {
            return Err(GpuError::Lost(g.failure().unwrap_or_default()));
        }
        let img = if private { Arc::new(resident::upload(g, base)?) } else { resident::image(g, base)? };
        Ok(LiquifyGpu {
            g: g.clone(),
            base: img,
            cols: base.cols,
            w: base.w,
            h: base.h,
            slot_of: vec![NONE; base.tiles.len()],
            slots: 0,
            cap: 0,
            index: None,
            index_dirty: true,
            field: None,
            frz: None,
            clear: Vec::new(),
            last: RunStats::default(),
            timed: false,
        })
    }

    /// Gives every tile in `need` a slot and uploads `upload` tiles from the
    /// CPU (`content(tile)` = the CPU field and freeze mask, None = zeros).
    /// New slots get the CPU content, or are zeroed on the GPU.
    pub fn prepare<'a>(&mut self, need: &[usize], upload: &[usize], content: impl Fn(usize) -> (Option<&'a [f32]>, Option<&'a [u8]>)) -> GResult<()> {
        let mut fresh: Vec<usize> = Vec::new();
        for &t in need.iter().chain(upload) {
            if self.slot_of[t] == NONE {
                self.slot_of[t] = self.slots;
                self.slots += 1;
                self.index_dirty = true;
                fresh.push(t);
            }
        }
        if self.slots > self.cap {
            self.grow(self.slots)?;
        }
        if self.index_dirty {
            self.index = Some(exec::table(&self.g, bytemuck::cast_slice(&self.slot_of)));
            self.index_dirty = false;
        }
        let (field, frz) = (self.field.as_ref().unwrap(), self.frz.as_ref().unwrap());
        let mut todo: Vec<usize> = upload.to_vec();
        for t in fresh {
            let (f, m) = content(t);
            if f.is_none() && m.is_none() {
                self.clear.push(self.slot_of[t]);
            } else {
                todo.push(t);
            }
        }
        todo.sort_unstable();
        todo.dedup();
        for t in todo {
            let s = self.slot_of[t] as u64;
            let (f, m) = content(t);
            let fb: &[u8] = f.map_or(zeros(), |f| bytemuck::cast_slice(f));
            let mb: &[u8] = m.unwrap_or(&zeros()[..FRZ_BYTES as usize]);
            self.g.queue.write_buffer(field, s * FIELD_BYTES, fb);
            self.g.queue.write_buffer(frz, s * FRZ_BYTES, mb);
        }
        Ok(())
    }

    /// Reallocates the slot buffers for at least `n` slots, keeping their
    /// contents (submitted before any later queue write).
    fn grow(&mut self, n: u32) -> GResult<()> {
        let total = self.slot_of.len() as u32;
        let cap = n.max(self.cap * 2).max(64).min(total).max(n);
        if cap as u64 * FIELD_BYTES > self.g.max_buffer() {
            return Err(GpuError::Vram);
        }
        let field = buffer(&self.g, cap as u64 * FIELD_BYTES, STORAGE)?;
        let frz = buffer(&self.g, cap as u64 * FRZ_BYTES, STORAGE)?;
        if let (Some(of), Some(om)) = (&self.field, &self.frz)
            && self.cap > 0
        {
            let scopes = exec::Scopes::push(&self.g);
            let mut enc = encoder(&self.g);
            enc.copy_buffer_to_buffer(of, 0, &field, 0, self.cap as u64 * FIELD_BYTES);
            enc.copy_buffer_to_buffer(om, 0, &frz, 0, self.cap as u64 * FRZ_BYTES);
            let idx = submit(&self.g, enc);
            scopes.pop()?;
            wait(&self.g, idx)?;
        }
        self.field = Some(field);
        self.frz = Some(frz);
        self.cap = cap;
        Ok(())
    }

    fn params(&self, d: Option<&DabK>) -> Vec<u32> {
        let (rect, c, dd, inv, pr, rate, k, mode) = match d {
            Some(d) => (d.rect, d.c, d.d, d.inv_r2, d.pressure, d.rate, d.k, d.mode),
            None => (RectI::EMPTY, (0.0, 0.0), (0.0, 0.0), 0.0, 0.0, 0.0, 0, 0),
        };
        vec![
            rect.x0 as u32,
            rect.y0 as u32,
            rect.w().max(0) as u32,
            rect.h().max(0) as u32,
            self.cols,
            self.w,
            self.h,
            mode,
            c.0.to_bits(),
            c.1.to_bits(),
            dd.0.to_bits(),
            dd.1.to_bits(),
            inv.to_bits(),
            pr.to_bits(),
            rate.to_bits(),
            k as u32,
            self.base.tw,
            0,
            0,
            0,
        ]
    }

    /// Applies `dabs` to the field, then resamples `tiles` (whole tiles,
    /// tile-major `(tile, pixels)`). Every dab's tiles and every listed tile
    /// must have a slot ([`Self::prepare`]).
    pub fn run(&mut self, dabs: &[DabK], tiles: &[usize]) -> GResult<Vec<(usize, Box<[u32]>)>> {
        let g = self.g.clone();
        if g.is_failed() {
            return Err(GpuError::Lost(g.failure().unwrap_or_default()));
        }
        let (Some(field), Some(frz), Some(index)) = (&self.field, &self.frz, &self.index) else {
            return Err(GpuError::Unsupported);
        };
        let scopes = exec::Scopes::push(&g);
        let mut enc = encoder(&g);
        for s in self.clear.drain(..) {
            enc.clear_buffer(field, s as u64 * FIELD_BYTES, Some(FIELD_BYTES));
            enc.clear_buffer(frz, s as u64 * FRZ_BYTES, Some(FRZ_BYTES));
        }
        let mut timer = if self.timed { exec::Timer::new(&g, 2 * dabs.len() as u32 + 2) } else { None };
        // Smooth's scratch: the largest smoothing dab.
        // (A small dummy otherwise: `update` binds it either way.)
        let smooth_px = dabs.iter().filter(|d| d.mode == 7).map(|d| d.rect.w().max(0) as u64 * d.rect.h().max(0) as u64).max().unwrap_or(32);
        let avg = buffer(&g, smooth_px.max(32) * 8, STORAGE)?;
        let upd = PipeKey { module: "liquify", entry: "update", f16: false, consts: Vec::new() };
        let sm = PipeKey { module: "liquify", entry: "smooth_avg", f16: false, consts: Vec::new() };
        for d in dabs {
            if d.rect.is_empty() {
                continue;
            }
            let params = self.params(Some(d));
            let gs = [groups(d.rect.w() as u32, 16), groups(d.rect.h() as u32, 16), 1];
            if d.mode == 7 {
                dispatch(&g, &mut enc, timer.as_mut(), &sm, &params, &[(1, index), (2, field), (4, &avg)], gs)?;
            }
            dispatch(&g, &mut enc, timer.as_mut(), &upd, &params, &[(1, index), (2, field), (3, frz), (4, &avg)], gs)?;
        }
        let out = if tiles.is_empty() { None } else { Some(self.encode_resample(&mut enc, timer.as_mut(), tiles)?) };
        if let Some(t) = timer.as_mut() {
            t.resolve(&g, &mut enc);
        }
        let t_sub = std::time::Instant::now();
        let idx = submit(&g, enc);
        scopes.pop()?;
        wait(&g, idx)?;
        let t_wait = t_sub.elapsed().as_secs_f64() * 1e3;
        // (Metal sometimes reports a single pass as 0: treat that as unknown.)
        let kernel_ms = timer.as_ref().and_then(|t| t.read_ms(&g)).filter(|&ms| ms > 0.0);
        self.last = RunStats { kernel_ms, wait_ms: t_wait };
        let t_read = std::time::Instant::now();
        let r = match &out {
            Some(o) => read_tiles(o, tiles),
            None => Ok(Vec::new()),
        };
        if trace() {
            eprintln!("liquify gpu: {} dabs, {} tiles: submit+wait {t_wait:.2} ms, readback {:.2} ms", dabs.len(), tiles.len(), t_read.elapsed().as_secs_f64() * 1e3);
        }
        r
    }

    /// Records the resample of `tiles`; returns the buffer to read
    /// (mappable, or a staging copy).
    fn encode_resample(&self, enc: &mut wgpu::CommandEncoder, timer: Option<&mut exec::Timer>, tiles: &[usize]) -> GResult<(Buf, Option<Buf>)> {
        let g = &self.g;
        let n = tiles.len() as u64 * TPX as u64 * 4;
        if n > g.max_buffer() || tiles.len() > g.limits.max_compute_workgroups_per_dimension as usize {
            return Err(GpuError::Vram);
        }
        let list: Vec<u32> = tiles.iter().map(|&t| (t as u32 % self.cols) | ((t as u32 / self.cols) << 16)).collect();
        let lb = exec::table(g, bytemuck::cast_slice(&list));
        let mappable = g.caps.mappable_primary;
        let dst = if mappable { buffer(g, n, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ)? } else { buffer(g, n, STORAGE)? };
        let key = PipeKey { module: "liquify", entry: "resample", f16: false, consts: Vec::new() };
        let params = self.params(None);
        let (field, index) = (self.field.as_ref().unwrap(), self.index.as_ref().unwrap());
        dispatch(g, enc, timer, &key, &params, &[(1, index), (2, field), (5, &self.base.buf), (6, &lb), (7, &dst)], [16, 16, tiles.len() as u32])?;
        let down = if mappable {
            None
        } else {
            let d = buffer(g, n, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
            enc.copy_buffer_to_buffer(&dst, 0, &d, 0, n);
            Some(d)
        };
        Ok((dst, down))
    }

    /// Resamples `tiles` from the current field (no edits).
    pub fn resample(&mut self, tiles: &[usize]) -> GResult<Vec<(usize, Box<[u32]>)>> {
        self.run(&[], tiles)
    }

    /// Reads the field of `tiles` (each must have a slot) back to the CPU.
    pub fn download(&self, tiles: &[usize]) -> GResult<Vec<(usize, Box<[f32]>)>> {
        let g = &self.g;
        if tiles.is_empty() {
            return Ok(Vec::new());
        }
        if g.is_failed() {
            return Err(GpuError::Lost(g.failure().unwrap_or_default()));
        }
        let field = self.field.as_ref().ok_or(GpuError::Unsupported)?;
        let mut out = Vec::with_capacity(tiles.len());
        // In chunks within the buffer limit.
        let per = (g.max_buffer() / FIELD_BYTES).clamp(1, 512) as usize;
        for chunk in tiles.chunks(per) {
            let n = chunk.len() as u64 * FIELD_BYTES;
            let st = buffer(g, n, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
            let scopes = exec::Scopes::push(g);
            let mut enc = encoder(g);
            for (i, &t) in chunk.iter().enumerate() {
                let s = self.slot_of[t];
                if s == NONE {
                    return Err(GpuError::Unsupported);
                }
                enc.copy_buffer_to_buffer(field, s as u64 * FIELD_BYTES, &st, i as u64 * FIELD_BYTES, FIELD_BYTES);
            }
            let idx = submit(g, enc);
            scopes.pop()?;
            wait(g, idx)?;
            st.map(n, wgpu::MapMode::Read)?;
            {
                let view = st.slice(..n).get_mapped_range();
                let f: &[f32] = bytemuck::cast_slice(&view);
                let part: Vec<(usize, Box<[f32]>)> =
                    chunk.par_iter().enumerate().map(|(i, &t)| (t, f[i * TPX * 2..(i + 1) * TPX * 2].to_vec().into_boxed_slice())).collect();
                out.extend(part);
            }
            st.unmap_now();
        }
        Ok(out)
    }
}

fn trace() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var_os("PF_GPU_TRACE").is_some())
}

/// Maps the resample output and copies each tile out.
fn read_tiles(out: &(Buf, Option<Buf>), tiles: &[usize]) -> GResult<Vec<(usize, Box<[u32]>)>> {
    let read = out.1.as_ref().unwrap_or(&out.0);
    let n = tiles.len() as u64 * TPX as u64 * 4;
    read.map(n, wgpu::MapMode::Read)?;
    let r = {
        let view = read.slice(..n).get_mapped_range();
        let px: &[u32] = bytemuck::cast_slice(&view);
        tiles.par_iter().enumerate().map(|(i, &t)| (t, px[i * TPX..(i + 1) * TPX].to_vec().into_boxed_slice())).collect()
    };
    read.unmap_now();
    Ok(r)
}
