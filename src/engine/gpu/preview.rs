//! Live filter previews on the GPU.
//!
//! The viewer asks for the region its missing display tiles cover at the
//! display mip level; the filter runs there in one GPU pass, reading the
//! layer level resident on the GPU (uploaded once per dialog, not per
//! slider move), and the compositor copies from the result instead of
//! running `Filter::render` per tile. The pixels are identical to the CPU
//! preview, which stays the fallback.
//!
//! The result stays on the GPU: the GPU compositor copies it straight into
//! its frame (no readback), and only the CPU compositor reads it back, once.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::device::Gpu;
use super::exec::{self, Buf, GResult, STORAGE, buffer, encoder, submit, wait};
use super::kernels::{self, Input, Io, Target, Temps};
use super::run::region_copy;
use crate::engine::document::Layer;
use crate::engine::filters::{Class, Filter};
use crate::engine::grid::RectI;

struct Region {
    level: u32,
    rect: RectI,
    /// The filtered region on the GPU (`rect.w x rect.h`).
    buf: Arc<Buf>,
    /// Read back on first CPU use.
    px: std::sync::OnceLock<Option<(Vec<u32>, crate::engine::memory::Reservation)>>,
}

/// A GPU-resident preview region the GPU compositor can copy from.
pub struct GpuRegion {
    pub buf: Arc<Buf>,
    pub rect: RectI,
}

/// GPU results for one live preview (shared by the preview's clones).
#[derive(Default)]
pub struct PreviewGpu {
    regions: Mutex<Vec<Region>>,
    /// Set once the GPU declined or failed for this preview.
    off: std::sync::atomic::AtomicBool,
}

impl PreviewGpu {
    /// Copies `rect` (layer-local, at `level`) from a computed region.
    pub fn lookup(&self, level: u32, rect: RectI, out: &mut [u32]) -> bool {
        let regions = self.regions.lock().unwrap();
        let Some(r) = regions.iter().rev().find(|r| r.level == level && r.rect.contains_rect(&rect)) else { return false };
        let Some((px, _)) = r.px.get_or_init(|| { let p = readback(&r.buf, (r.rect.w() * r.rect.h()) as u64 * 4).ok()?; let charge = crate::engine::memory::global().track_cpu((p.capacity() * 4) as u64); Some((p, charge)) }) else { return false };
        let rw = r.rect.w() as usize;
        let w = rect.w() as usize;
        for y in 0..rect.h() as usize {
            let s = (rect.y0 - r.rect.y0) as usize + y;
            let o = s * rw + (rect.x0 - r.rect.x0) as usize;
            out[y * w..(y + 1) * w].copy_from_slice(&px[o..o + w]);
        }
        true
    }

    /// The GPU region covering `rect` at `level`, if any.
    pub fn gpu_region(&self, level: u32, rect: RectI) -> Option<GpuRegion> {
        let regions = self.regions.lock().unwrap();
        let r = regions.iter().rev().find(|r| r.level == level && r.rect.contains_rect(&rect))?;
        Some(GpuRegion { buf: r.buf.clone(), rect: r.rect })
    }

    pub fn has(&self, level: u32, rect: RectI) -> bool {
        self.regions.lock().unwrap().iter().any(|r| r.level == level && r.rect.contains_rect(&rect))
    }
}

/// Renders `rect` (layer-local coordinates at `level`) of the preview on the
/// GPU if that's on and worth it. Returns the GPU time, or None when the CPU
/// should render (the reason is recorded for the Info panel).
pub fn prepare(cache: &PreviewGpu, filter: &Filter, layer: &Layer, level: u32, rect: RectI) -> Option<f64> {
    use std::sync::atomic::Ordering;
    if rect.is_empty() || cache.off.load(Ordering::Relaxed) || cache.has(level, rect) {
        return None;
    }
    let t0 = Instant::now();
    let m = super::feature_mode(super::Feature::Compositing).min_with(super::feature_mode(super::Feature::of(filter)));
    if m == super::Mode::Off || !kernels::supports(filter) {
        cache.off.store(true, Ordering::Relaxed);
        return None;
    }
    let g = super::context(m == super::Mode::Force || super::blocking())?;
    if m == super::Mode::Auto && let Some(why) = super::cost::veto_preview(&g, filter, rect.w() as u64 * rect.h() as u64, level) {
        super::record(&format!("{} preview", filter.name()), super::PathTaken::Cpu(why), 0.0);
        return None;
    }
    match render_gpu(&g, filter, layer, level, rect) {
        Ok(buf) => {
            let mut r = cache.regions.lock().unwrap();
            if r.len() >= 4 {
                r.remove(0);
            }
            r.push(Region { level, rect, buf: Arc::new(buf), px: Default::default() });
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            super::record(&format!("{} preview", filter.name()), super::PathTaken::Gpu, ms);
            Some(ms)
        }
        Err(e) => {
            match &e {
                exec::GpuError::Validation(_) => super::mark_broken(filter.name()),
                e if e.is_fault() => g.fail(e.to_string()),
                _ => {}
            }
            cache.off.store(true, Ordering::Relaxed);
            super::record(&format!("{} preview", filter.name()), super::PathTaken::Cpu(e.to_string()), t0.elapsed().as_secs_f64() * 1e3);
            None
        }
    }
}

/// Renders and stores a preview region on `g` without the mode / cost
/// checks (tests).
#[cfg(test)]
pub fn prepare_on(g: &Arc<Gpu>, cache: &PreviewGpu, filter: &Filter, layer: &Layer, level: u32, rect: RectI) -> GResult<()> {
    let buf = render_gpu(g, filter, layer, level, rect)?;
    cache.regions.lock().unwrap().push(Region { level, rect, buf: Arc::new(buf), px: Default::default() });
    Ok(())
}

/// `Filter::render` of `rect` at `level` on the GPU, read back.
pub fn render(g: &Arc<Gpu>, filter: &Filter, layer: &Layer, level: u32, rect: RectI) -> GResult<Vec<u32>> {
    let buf = render_gpu(g, filter, layer, level, rect)?;
    readback(&buf, (rect.w() * rect.h()) as u64 * 4)
}

fn readback(buf: &Buf, n: u64) -> GResult<Vec<u32>> {
    let g = buf.gpu();
    let down = buffer(&g, n, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
    let mut enc = encoder(&g);
    enc.copy_buffer_to_buffer(buf, 0, &down, 0, n);
    let idx = submit(&g, enc);
    wait(&g, idx)?;
    down.map(n, wgpu::MapMode::Read)?;
    let px: Vec<u32> = bytemuck::cast_slice(&down.slice(..n).get_mapped_range()).to_vec();
    down.unmap_now();
    Ok(px)
}

/// `Filter::render` of `rect` at `level` into a GPU buffer (waited).
pub fn render_gpu(g: &Arc<Gpu>, filter: &Filter, layer: &Layer, level: u32, rect: RectI) -> GResult<Buf> {
    let kernel = kernels::plan(g, filter, level, Target::Preview).ok_or(exec::GpuError::Unsupported)?;
    let grid = layer.level(level);
    let (w, h) = (rect.w() as u32, rect.h() as u32);
    let scopes = exec::Scopes::push(g);
    let (source, pad) = match kernel.input() {
        Input::Source { mips } => (super::resident::levels(g, grid, mips)?, 0u32),
        // Mip levels of the layer, not just this one: gather filters read
        // the levels above (`Src::layer`).
        Input::Padded => (vec![super::resident::image(g, grid)?], filter.pad(level) as u32),
    };
    let (pw, ph) = (w + 2 * pad, h + 2 * pad);
    let n = pw as u64 * ph as u64 * 4;
    let src = buffer(g, n.max(256), STORAGE)?;
    let dst = buffer(g, w as u64 * h as u64 * 4, STORAGE)?;
    let mut temps = Temps::default();
    temps.ensure(g, kernel.temps(), n)?;
    let mut enc = encoder(g);
    let (x0, y0) = (rect.x0 - pad as i32, rect.y0 - pad as i32);
    if kernel.input() == Input::Padded {
        let clamped = pad > 0 || filter.class() != Class::Point;
        region_copy(g, &mut enc, &source[0], (x0, y0, pw, ph), clamped, &src)?;
    }
    let io = Io { src: &src, dst: &dst, pw, ph, pad, level, origin: (x0, y0), img: (grid.w, grid.h), source: &source };
    kernel.encode(g, &mut enc, &io, &mut temps, None)?;
    let idx = submit(g, enc);
    scopes.pop()?;
    wait(g, idx)?;
    Ok(dst)
}
