//! Liquify.
//!
//! A session keeps the layer as it was when Liquify started (`base`), a sparse
//! per-tile displacement field `D` (level-0 pixels) and a freeze mask. The
//! visible pixel at `x` is `base(x - D(x))`, sampled bilinearly, so every tool
//! is just an edit of `D` inside the brush followed by resampling those pixels
//! (tile-parallel):
//!
//! | tool          | edit of D (f = falloff × pressure × (1 − freeze)) |
//! |---------------|------------------------------------------------------|
//! | Forward Warp  | D += drag · f                                        |
//! | Push Left     | D += perp(drag) · f                                  |
//! | Twirl ↻ / ↺   | D += v − R(±θf)·v, v = x − center                    |
//! | Pucker/Bloat  | D ∓= v · k f                                         |
//! | Reconstruct   | D *= 1 − rate·f                                      |
//! | Smooth        | D → local average of D                               |
//! | Freeze / Thaw | mask ± f                                             |
//!
//! ## CPU and GPU
//!
//! A stroke is cut into [`Dab`]s (the same list on either path). The CPU
//! path (the reference and the fallback) edits `D` and resamples each dab's
//! pixels in one tile-parallel pass. The GPU path
//! ([`crate::engine::gpu::kernels::liquify`]) keeps the source resident and
//! `D` in GPU tile slots: each call edits `D` dab by dab, resamples the
//! touched tiles once and reads them back. Because the layer always equals
//! `base` resampled through `D` (an untouched pixel resamples to itself),
//! resampling whole tiles gives the CPU's pixels; the GPU kernels evaluate
//! the CPU's float expressions with fusion blocked, so for the same field the
//! pixels are identical (Twirl's sin/cos are within an ulp).
//!
//! The CPU field is the reference: the GPU's field is downloaded before a
//! CPU stroke (or at a stroke's end once enough dabs piled up), CPU edits (and freeze painting, always on the
//! CPU) are uploaded before the next GPU stroke, and the dabs since the last
//! download are journaled, so if the GPU fails they are replayed on the CPU. Auto mode (Preferences ▸ Performance ▸ Distort & Liquify)
//! uses the GPU for brushes of at least [`AUTO_GPU_MIN_DIAMETER`] pixels,
//! where it is faster.

use rayon::prelude::*;

use super::gpu::kernels::liquify::{DabK, LiquifyGpu};
use super::gpu::{self, PathTaken};
use super::grid::{RectI, SyncPtr, TS, TS_I, TileGrid, finish_tile, tile_mut};
use super::px;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    ForwardWarp,
    Reconstruct,
    Smooth,
    TwirlCw,
    TwirlCcw,
    Pucker,
    Bloat,
    PushLeft,
    Freeze,
    Thaw,
}

impl Mode {
    pub const ALL: [Mode; 10] = [
        Mode::ForwardWarp,
        Mode::Reconstruct,
        Mode::Smooth,
        Mode::TwirlCw,
        Mode::TwirlCcw,
        Mode::Pucker,
        Mode::Bloat,
        Mode::PushLeft,
        Mode::Freeze,
        Mode::Thaw,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Mode::ForwardWarp => "Forward Warp",
            Mode::Reconstruct => "Reconstruct",
            Mode::Smooth => "Smooth",
            Mode::TwirlCw => "Twirl ↻",
            Mode::TwirlCcw => "Twirl ↺",
            Mode::Pucker => "Pucker",
            Mode::Bloat => "Bloat",
            Mode::PushLeft => "Push Left",
            Mode::Freeze => "Freeze Mask",
            Mode::Thaw => "Thaw Mask",
        }
    }

    pub fn help(self) -> &'static str {
        match self {
            Mode::ForwardWarp => "Push pixels in the direction you drag",
            Mode::Reconstruct => "Paint the image back toward how it was before Liquify",
            Mode::Smooth => "Even out bumpy or jagged warps",
            Mode::TwirlCw => "Hold to spin pixels clockwise",
            Mode::TwirlCcw => "Hold to spin pixels counter-clockwise",
            Mode::Pucker => "Hold to pull pixels toward the brush center (shrink)",
            Mode::Bloat => "Hold to push pixels away from the brush center (enlarge)",
            Mode::PushLeft => "Drag up to shift pixels left, down to shift right",
            Mode::Freeze => "Paint a red mask that protects pixels from other tools",
            Mode::Thaw => "Erase the freeze mask",
        }
    }

    /// Tools that keep acting while the mouse is held still.
    pub fn continuous(self) -> bool {
        matches!(self, Mode::Reconstruct | Mode::Smooth | Mode::TwirlCw | Mode::TwirlCcw | Mode::Pucker | Mode::Bloat | Mode::Freeze | Mode::Thaw)
    }
}

/// Brushes at least this wide (pixels) run on the GPU in Auto mode.
pub const AUTO_GPU_MIN_DIAMETER: f32 = 256.0;

/// GPU dabs kept in the journal before a stroke's end copies the field
/// back to the CPU.
pub const SYNC_DABS: usize = 256;

/// Which path a session runs strokes on.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Backend {
    /// The Preferences / `PF_GPU` mode (Auto: large brushes on the GPU).
    #[default]
    Auto,
    Cpu,
    /// The GPU whenever a context exists (tests, benchmarks).
    Gpu,
}

/// One application of the brush (the unit both paths work in).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dab {
    pub mode: Mode,
    pub c: (f32, f32),
    /// Drag since the previous dab.
    pub d: (f32, f32),
    pub radius: f32,
    pub pressure: f32,
    /// Effective rate (`Brush::rate` clamped, times the time scale).
    pub rate: f32,
}

impl Dab {
    /// The pixels the dab can touch, clipped to the image.
    pub fn rect(&self, bounds: &RectI) -> RectI {
        let (c, r) = (self.c, self.radius);
        RectI::new((c.0 - r).floor() as i32, (c.1 - r).floor() as i32, (c.0 + r).ceil() as i32 + 1, (c.1 + r).ceil() as i32 + 1).intersect(bounds)
    }

    /// Smooth's neighbor offset.
    fn smooth_k(&self) -> i32 {
        (self.radius * 0.15).clamp(2.0, 24.0) as i32
    }

    fn kernel(&self, bounds: &RectI) -> Option<DabK> {
        let mode = match self.mode {
            Mode::ForwardWarp => 0,
            Mode::PushLeft => 1,
            Mode::TwirlCw => 2,
            Mode::TwirlCcw => 3,
            Mode::Pucker => 4,
            Mode::Bloat => 5,
            Mode::Reconstruct => 6,
            Mode::Smooth => 7,
            Mode::Freeze | Mode::Thaw => return None,
        };
        let r = self.radius;
        Some(DabK { mode, rect: self.rect(bounds), c: self.c, d: self.d, inv_r2: 1.0 / (r * r), pressure: self.pressure, rate: self.rate, k: self.smooth_k() })
    }
}

/// GPU state of a session (see the module docs).
struct GpuState {
    k: Box<LiquifyGpu>,
    /// Tiles whose CPU field or mask changed since they were uploaded.
    stale: Vec<bool>,
    /// Tiles the GPU edited since the CPU field was last synced.
    touched: Vec<bool>,
    touched_list: Vec<usize>,
    /// Dabs run on the GPU since the last sync.
    journal: Vec<Dab>,
}

pub struct Session {
    pub layer_id: u64,
    /// Layer version this session last wrote (detects outside edits).
    pub version: u64,
    base: TileGrid,
    disp: Vec<Option<Box<[f32]>>>,
    freeze: Vec<Option<Box<[u8]>>>,
    last: Option<(f32, f32)>,
    backend: Backend,
    gpu: Option<GpuState>,
    /// Why the GPU stopped being used for this session.
    gpu_off: Option<String>,
    /// This stroke's path, decided on its first dab.
    stroke_gpu: Option<bool>,
    /// A failed sync left GPU-only dabs to replay on the CPU.
    replay: Vec<Dab>,
    /// A specific GPU context (tests); the shared one otherwise.
    ctx: Option<std::sync::Arc<gpu::Gpu>>,
    /// Time GPU passes (benchmarks).
    timed: bool,
}

pub struct Brush {
    pub radius: f32,
    pub pressure: f32,
    pub rate: f32,
}

impl Session {
    pub fn new(layer_id: u64, version: u64, grid: &TileGrid) -> Self {
        Session {
            layer_id,
            version,
            base: grid.clone(),
            disp: (0..grid.tiles.len()).map(|_| None).collect(),
            freeze: (0..grid.tiles.len()).map(|_| None).collect(),
            last: None,
            backend: Backend::Auto,
            gpu: None,
            gpu_off: None,
            stroke_gpu: None,
            replay: Vec::new(),
            ctx: None,
            timed: false,
        }
    }

    /// Runs on this GPU context instead of the shared one (tests).
    #[cfg(test)]
    pub fn use_context(&mut self, g: std::sync::Arc<gpu::Gpu>) {
        self.ctx = Some(g);
        self.backend = Backend::Gpu;
    }

    pub fn set_backend(&mut self, b: Backend) {
        self.backend = b;
        self.stroke_gpu = None;
    }

    /// Whether the GPU holds this session's field right now.
    pub fn on_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Why the GPU was dropped for this session, if it was.
    pub fn gpu_off(&self) -> Option<&str> {
        self.gpu_off.as_deref()
    }

    pub fn begin_stroke(&mut self, x: f32, y: f32) {
        self.last = Some((x, y));
        self.stroke_gpu = None;
    }

    /// Ends the stroke. The GPU's field is copied back to the CPU lazily:
    /// before a CPU stroke, or here once [`SYNC_DABS`] dabs are journaled
    /// (which bounds the replay after a GPU failure).
    pub fn end_stroke(&mut self) {
        self.last = None;
        self.stroke_gpu = None;
        if self.gpu.as_ref().is_some_and(|g| g.journal.len() >= SYNC_DABS) {
            self.sync_field();
        }
    }

    /// Displacement at an arbitrary pixel (for Smooth's neighborhood).
    #[inline]
    fn d_at(&self, x: i32, y: i32) -> (f32, f32) {
        let (w, h) = (self.base.w as i32, self.base.h as i32);
        let (x, y) = (x.clamp(0, w - 1), y.clamp(0, h - 1));
        let idx = ((y >> 8) as u32 * self.base.cols + (x >> 8) as u32) as usize;
        match &self.disp[idx] {
            Some(f) => {
                let i = ((y & 255) as usize) * TS + (x & 255) as usize;
                (f[2 * i], f[2 * i + 1])
            }
            None => (0.0, 0.0),
        }
    }

    /// The dabs that extend the stroke to `(x, y)` (and moves the stroke
    /// there). With `tick`, continuous tools act once even without motion.
    pub fn plan(&mut self, mode: Mode, x: f32, y: f32, b: &Brush, tick: bool) -> Vec<Dab> {
        let Some((lx, ly)) = self.last else {
            self.last = Some((x, y));
            return vec![];
        };
        let (dx, dy) = (x - lx, y - ly);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 0.25 && !tick {
            return vec![];
        }
        self.last = Some((x, y));
        let steps = if len < 0.25 { 1 } else { (len / (b.radius * 0.25)).ceil().max(1.0) as usize };
        (1..=steps)
            .map(|k| {
                let t = k as f32 / steps as f32;
                let continuous_scale = if tick { 1.0 } else { 1.0 / steps as f32 };
                Dab {
                    mode,
                    c: (lx + dx * t, ly + dy * t),
                    d: (dx / steps as f32, dy / steps as f32),
                    radius: b.radius,
                    pressure: b.pressure,
                    rate: b.rate.clamp(0.01, 1.0) * continuous_scale,
                }
            })
            .collect()
    }

    /// Extends the stroke to `(x, y)`. With `tick`, applies the continuous
    /// tools once at the current position even without motion. Returns the
    /// changed pixels.
    pub fn stroke_to(&mut self, mode: Mode, x: f32, y: f32, b: &Brush, grid: &mut TileGrid, tick: bool) -> Vec<RectI> {
        let mut dirty = RectI::EMPTY;
        if !self.replay.is_empty() {
            dirty = self.replay_cpu(grid);
        }
        let dabs = self.plan(mode, x, y, b, tick);
        if !dabs.is_empty() {
            dirty = dirty.union(&self.apply(&dabs, grid));
        }
        if dirty.is_empty() { vec![] } else { vec![dirty] }
    }

    /// Applies dabs on this stroke's path. Returns the changed pixels.
    pub fn apply(&mut self, dabs: &[Dab], grid: &mut TileGrid) -> RectI {
        let t0 = std::time::Instant::now();
        let paints = !matches!(dabs[0].mode, Mode::Freeze | Mode::Thaw);
        if paints && self.stroke_gpu.is_none() {
            self.stroke_gpu = Some(self.want_gpu(dabs[0].radius));
        }
        if paints && self.stroke_gpu == Some(true) {
            match self.apply_gpu(dabs, grid) {
                Ok(r) => {
                    gpu::record("Liquify", PathTaken::Gpu, t0.elapsed().as_secs_f64() * 1e3);
                    return r;
                }
                Err(e) => {
                    // Redo everything since the last sync on the CPU.
                    let why = e.to_string();
                    self.drop_gpu(e);
                    let mut dirty = self.replay_cpu(grid);
                    for d in dabs {
                        dirty = dirty.union(&self.dab_cpu(d, grid));
                    }
                    gpu::record("Liquify", PathTaken::GpuThenCpu(why), t0.elapsed().as_secs_f64() * 1e3);
                    return dirty;
                }
            }
        }
        // CPU: the CPU field must be current first.
        self.sync_field();
        let mut dirty = if self.replay.is_empty() { RectI::EMPTY } else { self.replay_cpu(grid) };
        for d in dabs {
            dirty = dirty.union(&self.dab_cpu(d, grid));
        }
        if paints {
            let why = self.gpu_off.clone().unwrap_or_else(|| "small brush or GPU off".into());
            gpu::record("Liquify", PathTaken::Cpu(why), t0.elapsed().as_secs_f64() * 1e3);
        }
        dirty
    }

    fn want_gpu(&self, radius: f32) -> bool {
        if self.gpu_off.is_some() {
            return false;
        }
        let m = gpu::feature_mode(gpu::Feature::Distort);
        let want = match self.backend {
            Backend::Cpu => false,
            Backend::Gpu => gpu::env_mode() != Some(gpu::Mode::Off),
            Backend::Auto => match m {
                gpu::Mode::Off => false,
                gpu::Mode::Force => true,
                gpu::Mode::Auto => radius * 2.0 >= AUTO_GPU_MIN_DIAMETER,
            },
        };
        want && (self.gpu.is_some() || self.ctx.is_some() || gpu::context(gpu::blocking() || m == gpu::Mode::Force || self.backend == Backend::Gpu).is_some())
    }

    /// Creates the GPU state (uploading the CPU field) if needed.
    fn ensure_gpu(&mut self) -> Result<(), gpu::exec::GpuError> {
        if self.gpu.is_some() {
            return Ok(());
        }
        let g = match &self.ctx {
            Some(g) => g.clone(),
            None => gpu::context(true).ok_or(gpu::exec::GpuError::Unsupported)?,
        };
        let mut k = LiquifyGpu::new(&g, &self.base, self.ctx.is_some())?;
        k.timed = self.timed;
        let n = self.disp.len();
        // Everything the CPU holds goes up with the first stroke.
        let stale = (0..n).map(|t| self.disp[t].is_some() || self.freeze[t].is_some()).collect();
        self.gpu = Some(GpuState { k: Box::new(k), stale, touched: vec![false; n], touched_list: Vec::new(), journal: Vec::new() });
        Ok(())
    }

    /// Uploads stale tiles and gives `need` slots.
    fn gpu_prepare(&mut self, need: &[usize]) -> Result<(), gpu::exec::GpuError> {
        self.ensure_gpu()?;
        let (disp, freeze) = (&self.disp, &self.freeze);
        let st = self.gpu.as_mut().unwrap();
        let up: Vec<usize> = (0..st.stale.len()).filter(|&t| st.stale[t]).collect();
        st.k.prepare(need, &up, |t| (disp[t].as_deref(), freeze[t].as_deref()))?;
        for t in up {
            st.stale[t] = false;
        }
        Ok(())
    }

    fn apply_gpu(&mut self, dabs: &[Dab], grid: &mut TileGrid) -> Result<RectI, gpu::exec::GpuError> {
        let bounds = self.base.bounds();
        let (cols, rows) = (self.base.cols, self.base.rows);
        let ks: Vec<DabK> = dabs.iter().filter_map(|d| d.kernel(&bounds)).collect();
        let mut dirty = RectI::EMPTY;
        let mut mark = vec![false; self.disp.len()];
        let mut need = Vec::new();
        for k in &ks {
            dirty = dirty.union(&k.rect);
            let (tx0, ty0, tx1, ty1) = k.rect.tile_span(cols, rows);
            for ty in ty0..ty1 {
                for tx in tx0..tx1 {
                    let t = (ty as u32 * cols + tx as u32) as usize;
                    if !mark[t] {
                        mark[t] = true;
                        need.push(t);
                    }
                }
            }
        }
        let t0 = std::time::Instant::now();
        self.gpu_prepare(&need)?;
        let t_prep = t0.elapsed().as_secs_f64() * 1e3;
        let st = self.gpu.as_mut().unwrap();
        let tiles = st.k.run(&ks, &need)?;
        let t_run = t0.elapsed().as_secs_f64() * 1e3;
        st.journal.extend_from_slice(dabs);
        for &t in &need {
            if !st.touched[t] {
                st.touched[t] = true;
                st.touched_list.push(t);
            }
        }
        let gptr = SyncPtr(grid.tiles.as_mut_ptr());
        tiles.into_par_iter().for_each(|(t, px)| {
            // SAFETY: each tile index appears once.
            unsafe { *gptr.at(t) = finish_tile(px) };
        });
        if std::env::var_os("PF_GPU_TRACE").is_some() {
            eprintln!("liquify: prepare {t_prep:.2} ms, run {:.2} ms, install {:.2} ms", t_run - t_prep, t0.elapsed().as_secs_f64() * 1e3 - t_run);
        }
        Ok(dirty)
    }

    /// Copies the field the GPU edited back to the CPU. If that fails, the
    /// journaled dabs are replayed on the CPU next time.
    pub fn sync_field(&mut self) {
        let Some(st) = &mut self.gpu else { return };
        if st.touched_list.is_empty() {
            return;
        }
        let t0 = std::time::Instant::now();
        let r = st.k.download(&st.touched_list);
        if std::env::var_os("PF_GPU_TRACE").is_some() {
            eprintln!("liquify: sync {} tiles {:.2} ms", st.touched_list.len(), t0.elapsed().as_secs_f64() * 1e3);
        }
        match r {
            Ok(tiles) => {
                for (t, f) in tiles {
                    self.disp[t] = Some(f);
                    st.touched[t] = false;
                }
                st.touched_list.clear();
                st.journal.clear();
            }
            Err(e) => {
                self.drop_gpu(e);
            }
        }
    }

    /// Stops using the GPU; the journal waits in `replay`.
    fn drop_gpu(&mut self, e: gpu::exec::GpuError) {
        if let Some(st) = self.gpu.take() {
            if e.is_fault() {
                st.k.g.fail(e.to_string());
            }
            self.replay.extend(st.journal);
        }
        self.gpu_off = Some(format!("GPU failed: {e}"));
        self.stroke_gpu = Some(false);
    }

    fn replay_cpu(&mut self, grid: &mut TileGrid) -> RectI {
        let j = std::mem::take(&mut self.replay);
        let mut dirty = RectI::EMPTY;
        for d in &j {
            dirty = dirty.union(&self.dab_cpu(d, grid));
        }
        dirty
    }

    /// One dab on the CPU: the field edit and the resample of the pixels
    /// it moved, fused per pixel (tile-parallel).
    fn dab_cpu(&mut self, dab: &Dab, grid: &mut TileGrid) -> RectI {
        let (mode, c, d) = (dab.mode, dab.c, dab.d);
        let r = dab.radius;
        let rect = dab.rect(&grid.bounds());
        if rect.is_empty() {
            return rect;
        }
        let (tx0, ty0, tx1, ty1) = rect.tile_span(grid.cols, grid.rows);
        let cols = grid.cols;
        let tiles: Vec<(i32, i32)> = (ty0..ty1).flat_map(|ty| (tx0..tx1).map(move |tx| (tx, ty))).collect();
        if let Some(st) = &mut self.gpu {
            for &(tx, ty) in &tiles {
                st.stale[(ty as u32 * cols + tx as u32) as usize] = true;
            }
        }
        // Smooth reads neighbors' displacement, so snapshot what it needs first.
        let smooth_src: Option<Vec<(f32, f32)>> = (mode == Mode::Smooth).then(|| {
            let k = dab.smooth_k();
            let mut v = Vec::with_capacity((rect.w() * rect.h()) as usize);
            for y in rect.y0..rect.y1 {
                for x in rect.x0..rect.x1 {
                    let mut acc = (0.0, 0.0);
                    let mut n = 0.0;
                    for (ox, oy) in [(-k, 0), (k, 0), (0, -k), (0, k), (0, 0), (-k, -k), (k, k), (-k, k), (k, -k)] {
                        let dd = self.d_at(x + ox, y + oy);
                        acc.0 += dd.0;
                        acc.1 += dd.1;
                        n += 1.0;
                    }
                    v.push((acc.0 / n, acc.1 / n));
                }
            }
            v
        });
        let gptr = SyncPtr(grid.tiles.as_mut_ptr());
        let mut disp = std::mem::take(&mut self.disp);
        let mut freeze = std::mem::take(&mut self.freeze);
        let dptr = SyncPtr(disp.as_mut_ptr());
        let fptr = SyncPtr(freeze.as_mut_ptr());
        let me = &*self;
        let inv_r2 = 1.0 / (r * r);
        let rate = dab.rate;
        tiles.par_iter().for_each(|&(tx, ty)| {
            let sampler = super::filters::distort::Sampler::new(&me.base, None, TS * TS, false);
            let idx = (ty as u32 * cols + tx as u32) as usize;
            // SAFETY: every tile index is visited once.
            let field_slot = unsafe { dptr.at(idx) };
            let fz = unsafe { fptr.at(idx) };
            let slot = unsafe { gptr.at(idx) };
            let tr = TileGrid::tile_rect(tx, ty).intersect(&rect);
            if matches!(mode, Mode::Freeze | Mode::Thaw) {
                let m = fz.get_or_insert_with(|| vec![0u8; TS * TS].into_boxed_slice());
                for y in tr.y0..tr.y1 {
                    let fy = y as f32 + 0.5 - c.1;
                    for x in tr.x0..tr.x1 {
                        let fx = x as f32 + 0.5 - c.0;
                        let q = (fx * fx + fy * fy) * inv_r2;
                        if q >= 1.0 {
                            continue;
                        }
                        let f = (1.0 - q) * dab.pressure * rate * 255.0;
                        let i = ((y - ty * TS_I) as usize) * TS + (x - tx * TS_I) as usize;
                        let v = m[i] as f32 + if mode == Mode::Freeze { f } else { -f };
                        m[i] = v.clamp(0.0, 255.0) as u8;
                    }
                }
                return;
            }
            let field = field_slot.get_or_insert_with(|| vec![0f32; TS * TS * 2].into_boxed_slice());
            let out = tile_mut(slot);
            let mut pixels = out.write();
            for y in tr.y0..tr.y1 {
                let fy = y as f32 + 0.5 - c.1;
                for x in tr.x0..tr.x1 {
                    let fx = x as f32 + 0.5 - c.0;
                    let q = (fx * fx + fy * fy) * inv_r2;
                    if q >= 1.0 {
                        continue;
                    }
                    let i = ((y - ty * TS_I) as usize) * TS + (x - tx * TS_I) as usize;
                    let frozen = fz.as_ref().map_or(0.0, |m| m[i] as f32 / 255.0);
                    let f = (1.0 - q) * (1.0 - q) * dab.pressure * (1.0 - frozen);
                    if f <= 0.0 {
                        continue;
                    }
                    let (vx, vy) = (fx, fy);
                    let (mut ddx, mut ddy) = (field[2 * i], field[2 * i + 1]);
                    match mode {
                        Mode::ForwardWarp => {
                            ddx += d.0 * f;
                            ddy += d.1 * f;
                        }
                        Mode::PushLeft => {
                            ddx += d.1 * f;
                            ddy -= d.0 * f;
                        }
                        Mode::TwirlCw | Mode::TwirlCcw => {
                            let th = 0.06 * rate * f * if mode == Mode::TwirlCw { 1.0 } else { -1.0 };
                            let (s, co) = th.sin_cos();
                            ddx += vx - (vx * co - vy * s);
                            ddy += vy - (vx * s + vy * co);
                        }
                        Mode::Pucker | Mode::Bloat => {
                            let k = 0.04 * rate * f * if mode == Mode::Bloat { 1.0 } else { -1.0 };
                            ddx += vx * k;
                            ddy += vy * k;
                        }
                        Mode::Reconstruct => {
                            let k = (1.0 - 0.15 * rate * f).max(0.0);
                            ddx *= k;
                            ddy *= k;
                        }
                        Mode::Smooth => {
                            if let Some(src) = &smooth_src {
                                let j = ((y - rect.y0) * rect.w() + (x - rect.x0)) as usize;
                                let (ax, ay) = src[j];
                                let k = (0.3 * rate * f).min(1.0);
                                ddx += (ax - ddx) * k;
                                ddy += (ay - ddy) * k;
                            }
                        }
                        Mode::Freeze | Mode::Thaw => {}
                    }
                    field[2 * i] = ddx;
                    field[2 * i + 1] = ddy;
                    pixels[i] = sampler.sample(x as f32 + 0.5 - ddx, y as f32 + 0.5 - ddy);
                }
            }
        });
        self.disp = disp;
        self.freeze = freeze;
        if matches!(mode, Mode::Freeze | Mode::Thaw) { RectI::EMPTY } else { rect }
    }

    /// Tiles that hold a displacement field (CPU copy; [`Self::sync_field`]
    /// first for GPU edits).
    pub fn field_tiles(&self) -> Vec<usize> {
        (0..self.disp.len()).filter(|&t| self.disp[t].is_some()).collect()
    }

    /// The CPU's resample of whole tiles from the current CPU field (the
    /// reference for the GPU's `resample`): image pixels `base(x - D(x))`,
    /// edge-tile padding from `base`.
    pub fn resample_cpu(&self, tiles: &[usize]) -> Vec<(usize, Box<[u32]>)> {
        let (w, h, cols) = (self.base.w as i32, self.base.h as i32, self.base.cols);
        tiles
            .par_iter()
            .map(|&t| {
                let (tx, ty) = ((t as u32 % cols) as i32, (t as u32 / cols) as i32);
                let mut o = vec![0u32; TS * TS].into_boxed_slice();
                let sampler = super::filters::distort::Sampler::new(&self.base, None, TS * TS, false);
                let padding = self.base.get(tx, ty).map(|t| t.read());
                for y in 0..TS_I {
                    for x in 0..TS_I {
                        let (gx, gy) = (tx * TS_I + x, ty * TS_I + y);
                        let i = (y * TS_I + x) as usize;
                        o[i] = if gx >= w || gy >= h {
                            padding.as_ref().map_or(0, |p| p[i])
                        } else {
                            let (dx, dy) = self.disp[t].as_ref().map_or((0.0, 0.0), |f| (f[2 * i], f[2 * i + 1]));
                            sampler.sample(gx as f32 + 0.5 - dx, gy as f32 + 0.5 - dy)
                        };
                    }
                }
                (t, o)
            })
            .collect()
    }

    /// The GPU's resample of whole tiles from the current CPU field
    /// (uploaded first). For tests and benchmarks.
    pub fn resample_gpu(&mut self, tiles: &[usize]) -> Result<Vec<(usize, Box<[u32]>)>, gpu::exec::GpuError> {
        self.sync_field();
        self.gpu_prepare(tiles)?;
        self.gpu.as_mut().unwrap().k.resample(tiles)
    }

    /// The displacement field of a tile (CPU copy), interleaved x, y.
    #[cfg(test)]
    pub fn field(&self, tile: usize) -> Option<&[f32]> {
        self.disp[tile].as_deref()
    }

    /// GPU timings of the last GPU call (benchmarks).
    pub fn gpu_stats(&self) -> Option<gpu::kernels::liquify::RunStats> {
        self.gpu.as_ref().map(|s| s.k.last)
    }

    /// Times the GPU passes of later calls (benchmarks).
    pub fn set_gpu_timed(&mut self, on: bool) {
        self.timed = on;
        if let Some(s) = &mut self.gpu {
            s.k.timed = on;
        }
    }

    /// Red overlay tile for the freeze mask (premultiplied, 50% max).
    pub fn overlay_grid(&self) -> TileGrid {
        let mut g = TileGrid::new(self.base.w, self.base.h);
        for (i, m) in self.freeze.iter().enumerate() {
            if let Some(m) = m {
                let px: Box<[u32]> = m.iter().map(|&a| px::scale(px::pack(220, 30, 40, 255), a as u32 / 2)).collect();
                g.tiles[i] = super::grid::finish_tile(px);
            }
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn halves() -> TileGrid {
        let mut g = TileGrid::new(256, 256);
        for y in 0..256 {
            for x in 0..256 {
                let v = if x < 128 { 0 } else { 255 };
                tile_mut(&mut g.tiles[0]).write()[y * 256 + x] = px::pack(v, v, v, 255);
            }
        }
        g
    }

    #[test]
    fn forward_warp_then_reconstruct() {
        let mut g = halves();
        let orig = g.clone();
        let mut s = Session::new(1, 0, &g);
        let b = Brush { radius: 40.0, pressure: 1.0, rate: 1.0 };
        s.begin_stroke(120.0, 128.0);
        s.stroke_to(Mode::ForwardWarp, 150.0, 128.0, &b, &mut g, false);
        assert!(px::unpack(g.pixel(132, 128))[0] < 128, "content not pushed");
        s.end_stroke();
        s.begin_stroke(140.0, 128.0);
        for _ in 0..200 {
            s.stroke_to(Mode::Reconstruct, 140.0, 128.0, &b, &mut g, true);
        }
        let diff = (px::unpack(g.pixel(132, 128))[0] as i32 - px::unpack(orig.pixel(132, 128))[0] as i32).abs();
        assert!(diff < 40, "reconstruct did not restore (diff {diff})");
    }

    #[test]
    fn freeze_blocks_warp() {
        let mut g = halves();
        let mut s = Session::new(1, 0, &g);
        let b = Brush { radius: 40.0, pressure: 1.0, rate: 1.0 };
        s.begin_stroke(128.0, 128.0);
        for _ in 0..10 {
            s.stroke_to(Mode::Freeze, 128.0, 128.0, &b, &mut g, true);
        }
        s.end_stroke();
        s.begin_stroke(120.0, 128.0);
        s.stroke_to(Mode::ForwardWarp, 150.0, 128.0, &b, &mut g, false);
        assert_eq!(px::unpack(g.pixel(132, 128))[0], 255, "frozen pixels moved");
    }

    #[test]
    fn bloat_and_twirl_move_pixels() {
        for mode in [Mode::Bloat, Mode::Pucker, Mode::TwirlCw, Mode::TwirlCcw, Mode::PushLeft, Mode::Smooth] {
            let mut g = halves();
            let orig = g.clone();
            let mut s = Session::new(1, 0, &g);
            let b = Brush { radius: 60.0, pressure: 1.0, rate: 1.0 };
            s.begin_stroke(110.0, 128.0);
            for i in 0..40 {
                s.stroke_to(mode, 110.0, 128.0 - i as f32 * 0.5, &b, &mut g, true);
            }
            let changed = (90..130).any(|x| g.pixel(x, 128) != orig.pixel(x, 128)) || mode == Mode::Smooth;
            assert!(changed, "{:?} changed nothing", mode);
        }
    }
}
