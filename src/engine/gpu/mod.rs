//! GPU compute path for filters and compositing, on wgpu (Metal, DirectX 12,
//! Vulkan).
//!
//! The CPU code stays the reference and the automatic fallback. A GPU kernel
//! is an implementation detail behind the same filter entry:
//! [`crate::engine::filters::apply`] asks [`apply_filter`] first, which runs
//! the GPU only when it is enabled, healthy, has a kernel for the filter, and
//! the cost model says it wins. Everything else, and anything that fails on
//! the way (no adapter, VRAM, validation, watchdog, device loss), runs or
//! finishes on the CPU with identical pixels.
//!
//! Control: Preferences ▸ Performance (Use Graphics Processor Auto / Off /
//! Force, per-feature overrides) or `PF_GPU=off|auto|force`, which wins.

pub mod compose;
pub mod cost;
pub mod device;
pub mod exec;
pub mod kernels;
pub mod mipgen;
pub mod patchmatch;
pub mod poisson;
pub mod preview;
pub mod prog;
pub mod resample;
pub mod resident;
pub mod run;
pub mod shaders;
pub mod stats;
pub mod tiles;
#[cfg(test)]
mod compose_tests;
pub mod styles;
#[cfg(test)]
mod styles_tests;
#[cfg(test)]
pub(crate) mod tests;
#[cfg(test)]
mod retouch_tests;

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use exec::GpuError;
use std::time::Instant;

pub use device::Gpu;

use crate::engine::filters::Filter;
use crate::engine::grid::TileGrid;
use crate::engine::ops::Progress;
use crate::engine::selection::Selection;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Mode {
    #[default]
    Auto,
    Off,
    Force,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Auto => "Auto",
            Mode::Off => "Off",
            Mode::Force => "Force",
        }
    }

    pub fn parse(s: &str) -> Option<Mode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "on" | "1" => Some(Mode::Auto),
            "off" | "cpu" | "0" => Some(Mode::Off),
            "force" | "gpu" => Some(Mode::Force),
            _ => None,
        }
    }

    pub const ALL: [Mode; 3] = [Mode::Auto, Mode::Off, Mode::Force];

    /// The more restrictive of two modes (Off beats Auto beats Force).
    pub fn min_with(self, o: Mode) -> Mode {
        match (self, o) {
            (Mode::Off, _) | (_, Mode::Off) => Mode::Off,
            (Mode::Auto, _) | (_, Mode::Auto) => Mode::Auto,
            _ => Mode::Force,
        }
    }
}

/// Groups for the per-feature overrides in Preferences ▸ Performance.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Feature {
    Adjustments,
    Blur,
    BlurGallery,
    Sharpen,
    Noise,
    Distort,
    Stylize,
    FilterGallery,
    Render,
    Resample,
    CameraRaw,
    SelectMask,
    Compositing,
    /// Healing / Patch / Content-Aware Fill (Poisson blend, PatchMatch).
    Retouch,
}

impl Feature {
    pub const ALL: [Feature; 14] = [
        Feature::Adjustments,
        Feature::Blur,
        Feature::BlurGallery,
        Feature::Sharpen,
        Feature::Noise,
        Feature::Distort,
        Feature::Stylize,
        Feature::FilterGallery,
        Feature::Render,
        Feature::Resample,
        Feature::CameraRaw,
        Feature::SelectMask,
        Feature::Compositing,
        Feature::Retouch,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Feature::Adjustments => "Adjustments",
            Feature::Blur => "Blur filters",
            Feature::BlurGallery => "Blur Gallery & Lens Blur",
            Feature::Sharpen => "Sharpen",
            Feature::Noise => "Noise",
            Feature::Distort => "Distort & Liquify",
            Feature::Stylize => "Stylize & Pixelate",
            Feature::FilterGallery => "Filter Gallery",
            Feature::Render => "Render",
            Feature::Resample => "Image Size & Transform",
            Feature::CameraRaw => "Camera Raw Filter",
            Feature::SelectMask => "Select and Mask",
            Feature::Compositing => "Compositing & previews",
            Feature::Retouch => "Healing & Content-Aware Fill",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Feature::Adjustments => "adjust",
            Feature::Blur => "blur",
            Feature::BlurGallery => "blur_gallery",
            Feature::Sharpen => "sharpen",
            Feature::Noise => "noise",
            Feature::Distort => "distort",
            Feature::Stylize => "stylize",
            Feature::FilterGallery => "gallery",
            Feature::Render => "render",
            Feature::Resample => "resample",
            Feature::CameraRaw => "camera_raw",
            Feature::SelectMask => "select_mask",
            Feature::Compositing => "composite",
            Feature::Retouch => "retouch",
        }
    }

    pub fn of(f: &Filter) -> Feature {
        use Filter::*;
        match f {
            GaussianBlur { .. } | BoxBlur { .. } | Blur | BlurMore | MotionBlur { .. } | SurfaceBlur { .. } | SmartBlur { .. } | ShapeBlur { .. }
            | RadialBlur { .. } | Average { .. } => Feature::Blur,
            LensBlur { .. } | FieldBlur { .. } | IrisBlur { .. } | TiltShift { .. } | SpinBlur { .. } | PathBlur { .. } => Feature::BlurGallery,
            UnsharpMask { .. } | Sharpen | SharpenMore | SharpenEdges | SmartSharpen { .. } | HighPass { .. } => Feature::Sharpen,
            AddNoise { .. } | Median { .. } | DustScratches { .. } | Despeckle | ReduceNoise { .. } | JpegCleanup { .. } | SkinSmoothing { .. }
            | PhotoRestoration { .. } => Feature::Noise,
            Warp { .. } | Displace { .. } | LensCorrection { .. } => Feature::Distort,
            Gallery { .. } => Feature::FilterGallery,
            Clouds { .. } | Fibers { .. } | LensFlare { .. } | Lighting(_) | Gradient(_) => Feature::Render,
            CameraRaw(_) => Feature::CameraRaw,
            BrightnessContrast { .. } | Levels { .. } | Curves { .. } | Exposure { .. } | Invert | Desaturate | Threshold { .. } | Posterize { .. }
            | Solarize | Lut { .. } | Color { .. } | ShadowsHighlights { .. } => Feature::Adjustments,
            _ => Feature::Stylize,
        }
    }
}

// ----- mode -------------------------------------------------------------------------------

static MODE: AtomicU8 = AtomicU8::new(0);
static OVERRIDES: Mutex<Vec<(Feature, Mode)>> = Mutex::new(Vec::new());

fn mode_from_u8(v: u8) -> Mode {
    match v {
        1 => Mode::Off,
        2 => Mode::Force,
        _ => Mode::Auto,
    }
}

/// `PF_GPU`, read once.
pub fn env_mode() -> Option<Mode> {
    static ENV: OnceLock<Option<Mode>> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var("PF_GPU").ok().and_then(|v| Mode::parse(&v)))
}

/// Applies Preferences ▸ Performance.
pub fn set_prefs(mode: Mode, overrides: &[(Feature, Mode)]) {
    MODE.store(mode as u8, Ordering::Relaxed);
    *OVERRIDES.lock().unwrap() = overrides.to_vec();
}

/// The effective global mode (`PF_GPU` wins over the preference).
pub fn mode() -> Mode {
    env_mode().unwrap_or_else(|| mode_from_u8(MODE.load(Ordering::Relaxed)))
}

/// The effective mode for one feature.
pub fn feature_mode(f: Feature) -> Mode {
    if let Some(m) = env_mode() {
        return m;
    }
    let g = mode();
    if g == Mode::Off {
        return Mode::Off;
    }
    OVERRIDES.lock().unwrap().iter().find(|o| o.0 == f).map_or(g, |o| o.1)
}

// ----- the shared context -----------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    NotStarted,
    Initializing,
    Ready,
    /// No usable adapter.
    Unavailable(String),
    /// Failed too often on this machine; remembered in the config dir.
    Disabled(String),
}

struct State {
    status: Status,
    ctx: Option<Arc<Gpu>>,
    /// Failures of this adapter recorded so far.
    failures: u32,
}

static STATE: Mutex<State> = Mutex::new(State { status: Status::NotStarted, ctx: None, failures: 0 });
static READY: Condvar = Condvar::new();

pub fn status() -> Status {
    STATE.lock().unwrap().status.clone()
}

fn init_now() {
    let r = Gpu::new();
    let mut st = STATE.lock().unwrap();
    match r {
        Ok(g) => {
            let n = device::failures(&g.info.key());
            if n >= device::MAX_FAILURES && env_mode() != Some(Mode::Force) {
                st.status = Status::Disabled(format!("{} failed {n} times; disabled (Benchmark GPU re-enables it)", g.info.name));
                st.ctx = None;
            } else {
                st.failures = n;
                st.status = Status::Ready;
                st.ctx = Some(Arc::new(g));
            }
        }
        Err(e) => {
            st.status = Status::Unavailable(e);
            st.ctx = None;
        }
    }
    READY.notify_all();
}

/// Starts creating the context in the background (app start), then
/// precompiles every pipeline and loads the cost model.
pub fn warm_up() {
    {
        let mut st = STATE.lock().unwrap();
        if st.status != Status::NotStarted || mode() == Mode::Off {
            return;
        }
        st.status = Status::Initializing;
    }
    std::thread::Builder::new()
        .name("gpu-init".into())
        .spawn(|| {
            init_now();
            if let Some(g) = STATE.lock().unwrap().ctx.clone() {
                let _ = g.precompile();
                cost::load(&g);
            }
        })
        .ok();
}

/// The context if it's usable. `wait`: block while it's being created
/// (Force mode, the CLI and tests); otherwise the caller uses the CPU now.
pub fn context(wait: bool) -> Option<Arc<Gpu>> {
    let mut st = STATE.lock().unwrap();
    loop {
        match &st.status {
            Status::NotStarted => {
                st.status = Status::Initializing;
                drop(st);
                if wait {
                    init_now();
                    if let Some(g) = STATE.lock().unwrap().ctx.clone() {
                        cost::load(&g);
                    }
                } else {
                    std::thread::spawn(init_now);
                    return None;
                }
                st = STATE.lock().unwrap();
            }
            Status::Initializing => {
                if !wait {
                    return None;
                }
                st = READY.wait(st).unwrap();
            }
            Status::Ready => {
                let g = st.ctx.clone()?;
                if g.is_failed() {
                    // Device lost or poisoned: count it and start over (or
                    // give up on this device for good).
                    let why = g.failure().unwrap_or_default();
                    st.failures = device::record_failure(&g.info.key());
                    st.ctx = None;
                    if st.failures >= device::MAX_FAILURES && env_mode() != Some(Mode::Force) {
                        st.status = Status::Disabled(format!("{} failed {} times ({why}); disabled", g.info.name, st.failures));
                        return None;
                    }
                    st.status = Status::NotStarted;
                    continue;
                }
                return Some(g);
            }
            Status::Unavailable(_) | Status::Disabled(_) => return None,
        }
    }
}

/// Re-enables a disabled device (Preferences ▸ Benchmark GPU).
pub fn reset_health() {
    let mut st = STATE.lock().unwrap();
    if let Some(g) = &st.ctx {
        device::clear_failures(&g.info.key());
    } else if let Ok(t) = std::fs::read_to_string(crate::engine::prefs::config_dir().join("gpu_health.txt")) {
        for l in t.lines() {
            if let Some((k, _)) = l.rsplit_once('=') {
                device::clear_failures(k);
            }
        }
    }
    st.failures = 0;
    if matches!(st.status, Status::Disabled(_)) {
        st.status = Status::NotStarted;
    }
}

/// One line for the Info panel and `--bench-gpu`.
pub fn describe() -> String {
    let st = STATE.lock().unwrap();
    match (&st.status, &st.ctx) {
        (Status::Ready, Some(g)) => format!(
            "{} · {} {} · {:.1} GB{}",
            g.info.name,
            g.info.backend,
            g.info.device_type.replace("Gpu", "").to_lowercase(),
            g.info.vram as f64 / (1u64 << 30) as f64,
            if g.caps.unified_memory { " unified" } else { "" }
        ),
        (Status::Initializing, _) => "starting…".into(),
        (Status::NotStarted, _) => "not started".into(),
        (Status::Unavailable(e), _) | (Status::Disabled(e), _) => e.clone(),
        _ => "—".into(),
    }
}

/// "Apple M5 Pro · Metal · 37 GB unified" for the Info panel.
pub fn describe_short() -> String {
    let st = STATE.lock().unwrap();
    match (&st.status, &st.ctx) {
        (Status::Ready, Some(g)) => format!(
            "{} · {} · {:.0} GB{}",
            g.info.name,
            g.info.backend,
            g.info.vram as f64 / (1u64 << 30) as f64,
            if g.caps.unified_memory { " unified" } else { "" }
        ),
        _ => {
            drop(st);
            describe()
        }
    }
}

/// Whether callers wait for the context to be created. The CLI and tests
/// do; the app warms the context up at start and uses the CPU until then.
static BLOCKING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_blocking(b: bool) {
    BLOCKING.store(b, Ordering::Relaxed);
}

pub fn blocking() -> bool {
    BLOCKING.load(Ordering::Relaxed)
}

// ----- which path ran ---------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum PathTaken {
    Gpu,
    /// GPU for part, then the CPU from where it stopped.
    GpuThenCpu(String),
    Cpu(String),
}

#[derive(Clone, Debug)]
pub struct LastOp {
    pub op: String,
    pub path: PathTaken,
    pub ms: f64,
    pub adapter: String,
}

static LAST: Mutex<Option<LastOp>> = Mutex::new(None);
static LAST_COMPOSE: Mutex<Option<LastOp>> = Mutex::new(None);

/// The display compositor's last frame (kept apart from filters).
pub fn last_compose() -> Option<LastOp> {
    LAST_COMPOSE.lock().unwrap().clone()
}

pub fn record_compose(op: &str, path: PathTaken, ms: f64) {
    let adapter = STATE.lock().unwrap().ctx.as_ref().map(|g| g.info.name.clone()).unwrap_or_default();
    *LAST_COMPOSE.lock().unwrap() = Some(LastOp { op: op.to_string(), path, ms, adapter });
}

pub fn last_op() -> Option<LastOp> {
    LAST.lock().unwrap().clone()
}

/// Forgets the last operation (a new document is shown: the Info panel's
/// Path row describes work on it, not on the previous one).
pub fn clear_last_op() {
    *LAST.lock().unwrap() = None;
}

static LAST_BY_OP: Mutex<Vec<LastOp>> = Mutex::new(Vec::new());

pub fn record(op: &str, path: PathTaken, ms: f64) {
    let adapter = STATE.lock().unwrap().ctx.as_ref().map(|g| g.info.name.clone()).unwrap_or_default();
    let last = LastOp { op: op.to_string(), path, ms, adapter };
    {
        let mut by = LAST_BY_OP.lock().unwrap();
        by.retain(|o| o.op != op);
        by.push(last.clone());
    }
    *LAST.lock().unwrap() = Some(last);
}

/// The last run of the operation named `op` (a tool runs several ops; the
/// Info panel shows the last, benchmarks ask for a specific one).
pub fn last_of(op: &str) -> Option<LastOp> {
    LAST_BY_OP.lock().unwrap().iter().find(|o| o.op == op).cloned()
}

impl LastOp {
    /// "GPU", "CPU: cost model", ... (the adapter is on its own row).
    pub fn short_path(&self) -> String {
        match &self.path {
            PathTaken::Gpu => "GPU".into(),
            PathTaken::GpuThenCpu(_) => "GPU → CPU fallback".into(),
            PathTaken::Cpu(why) => format!("CPU ({})", why.split(':').next().unwrap_or(why)),
        }
    }

    /// "GPU (Apple M5 Pro)", "CPU (cost model)", ...
    pub fn path_label(&self) -> String {
        match &self.path {
            PathTaken::Gpu => format!("GPU ({})", self.adapter),
            PathTaken::GpuThenCpu(why) => format!("GPU → CPU fallback ({why})"),
            PathTaken::Cpu(why) => format!("CPU ({why})"),
        }
    }
}

/// Preferences ▸ Performance ▸ Benchmark GPU: re-enables a disabled device,
/// recalibrates the cost model and measures every listed filter on this
/// machine (CPU and GPU, 2048² image), then saves the model. Returns a
/// one-paragraph summary.
pub fn benchmark(filters: &[Filter]) -> String {
    reset_health();
    cost::reset();
    let Some(g) = context(true) else { return format!("No usable GPU: {}", describe()) };
    let mut m = cost::calibrate(&g);
    let side = 2048u32;
    let img = crate::engine::ops::procedural(side, side);
    let px = (side * side) as f64;
    let best = |f: &mut dyn FnMut()| {
        f();
        (0..3)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .fold(f64::MAX, f64::min)
    };
    let (mut faster, mut speedups) = (0, Vec::new());
    for f in filters {
        let Some(k) = kernels::plan(&g, f, 0, kernels::Target::Commit) else { continue };
        let Ok((gpu, _)) = run::resident_ms(&g, &img, f, k.as_ref(), 3) else { continue };
        let cpu = best(&mut || drop(crate::engine::ops::apply_filter_cpu(&img, f, None, &Default::default(), None)));
        let e2e = best(&mut || drop(apply_filter_on(&g, &img, f, None, &Default::default(), &Default::default())));
        m.ops.insert(cost::op_key(f, 0), cost::Rates { gpu_ns_px: gpu * 1e6 / px, cpu_ns_px: cpu * 1e6 / px });
        if e2e < cpu {
            faster += 1;
        }
        speedups.push(cpu / e2e);
    }
    cost::set_model(m.clone());
    cost::save();
    speedups.sort_by(f64::total_cmp);
    let median = speedups.get(speedups.len() / 2).copied().unwrap_or(1.0);
    format!(
        "{}: {} kernels measured, the GPU is faster on {faster} (median {median:.1}× incl. upload and readback). Round trip {:.2} ms, transfer {:.1} GB/s. Saved for Auto mode.",
        g.info.name,
        speedups.len(),
        m.fixed_ms,
        1.0 / m.transfer_ns_b
    )
}

// ----- filters ----------------------------------------------------------------------------

/// Kernels that hit a validation error: they stay on the CPU for the rest
/// of the session (one bad kernel doesn't take the device down).
static BROKEN: Mutex<Vec<String>> = Mutex::new(Vec::new());

pub fn mark_broken(op: &str) {
    let mut b = BROKEN.lock().unwrap();
    if !b.iter().any(|x| x == op) {
        b.push(op.to_string());
    }
}

pub fn is_broken(op: &str) -> bool {
    BROKEN.lock().unwrap().iter().any(|x| x == op)
}

/// Why a filter stays on the CPU, or None if the GPU should run it.
pub fn gpu_veto(filter: &Filter, px: u64) -> Option<(String, Option<Arc<Gpu>>)> {
    let m = feature_mode(Feature::of(filter));
    if m == Mode::Off {
        return Some(("GPU off".into(), None));
    }
    if !kernels::supports(filter) {
        return Some(("no GPU kernel".into(), None));
    }
    if is_broken(filter.name()) {
        return Some(("GPU kernel disabled after an error".into(), None));
    }
    let Some(g) = context(m == Mode::Force || BLOCKING.load(Ordering::Relaxed)) else {
        return Some((format!("GPU {}", describe()), None));
    };
    if m == Mode::Auto && let Some(why) = cost::veto(&g, filter, px) {
        return Some((why, Some(g)));
    }
    None
}

/// The GPU for a retouching op (`op` names its cost-model rate) on `units`
/// cells, or why the CPU runs it: the same rules as [`gpu_veto`] (feature
/// mode, context, cost model in Auto).
pub fn retouch_gate(op: &str, units: u64) -> Result<Arc<Gpu>, String> {
    let m = feature_mode(Feature::Retouch);
    if m == Mode::Off {
        return Err("GPU off".into());
    }
    let Some(g) = context(m == Mode::Force || BLOCKING.load(Ordering::Relaxed)) else {
        return Err(format!("GPU {}", describe()));
    };
    if g.is_failed() {
        return Err(g.failure().unwrap_or_else(|| "GPU failed".into()));
    }
    if m == Mode::Auto
        && let Some(why) = cost::veto_op(&g, op, units)
    {
        return Err(why);
    }
    Ok(g)
}

/// Runs a whole-layer filter on the GPU, finishing on the CPU if the GPU
/// stops part way. Returns None when the CPU should do all of it.
pub fn apply_filter(grid: &TileGrid, filter: &Filter, sel: Option<&Selection>, progress: &Progress) -> Option<TileGrid> {
    let t0 = Instant::now();
    let px = grid.w as u64 * grid.h as u64;
    if let Some((why, _)) = gpu_veto(filter, px) {
        let r = crate::engine::ops::apply_filter_cpu(grid, filter, sel, progress, None);
        record(filter.name(), PathTaken::Cpu(why), t0.elapsed().as_secs_f64() * 1e3);
        cost::observe_cpu(filter, px, t0.elapsed().as_secs_f64() * 1e3);
        return Some(r);
    }
    let g = context(true)?;
    let r = apply_filter_on(&g, grid, filter, sel, progress, &run::RunOpts::default());
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    match &r.1 {
        PathTaken::Gpu => cost::observe_gpu(&g, filter, px, ms),
        _ => {}
    }
    record(filter.name(), r.1, ms);
    Some(r.0)
}

/// [`apply_filter`] on a given context, always trying the GPU first
/// (tests and benchmarks use it directly). Returns the pixels and the path.
pub fn apply_filter_on(g: &Arc<Gpu>, grid: &TileGrid, filter: &Filter, sel: Option<&Selection>, progress: &Progress, opts: &run::RunOpts) -> (TileGrid, PathTaken) {
    if g.is_failed() {
        let why = g.failure().unwrap_or_else(|| "GPU failed".into());
        return (crate::engine::ops::apply_filter_cpu(grid, filter, sel, progress, None), PathTaken::Cpu(why));
    }
    let Some(kernel) = kernels::plan(g, filter, 0, kernels::Target::Commit) else {
        return (crate::engine::ops::apply_filter_cpu(grid, filter, sel, progress, None), PathTaken::Cpu("no GPU kernel for these settings".into()));
    };
    let out = run::run_filter(g, grid, filter, kernel.as_ref(), sel, progress, opts, run::trace());
    if let Some(ms) = out.kernel_ms.filter(|_| run::trace()) {
        eprintln!("gpu {}: {} bands, kernel passes {ms:.2} ms", filter.name(), out.bands);
    }
    match out.stopped {
        None => (out.grid, PathTaken::Gpu),
        Some((ty, err)) => {
            match &err {
                // A shader/pipeline problem: this kernel only.
                GpuError::Validation(_) => mark_broken(filter.name()),
                e if e.is_fault() => g.fail(e.to_string()),
                _ => {}
            }
            let (_, ty1) = run::region_rows(grid, sel).unwrap_or((ty, ty));
            let rest = crate::engine::ops::apply_filter_cpu(grid, filter, sel, progress, Some((ty, ty1)));
            let mut merged = out.grid;
            let cols = grid.cols as usize;
            for (i, slot) in merged.tiles.iter_mut().enumerate() {
                if (i / cols) as i32 >= ty {
                    *slot = rest.tiles[i].clone();
                }
            }
            let path = if out.bands > 0 && ty > run::region_rows(grid, sel).map_or(ty, |r| r.0) { PathTaken::GpuThenCpu(err.to_string()) } else { PathTaken::Cpu(err.to_string()) };
            (merged, path)
        }
    }
}

/// Reclaim inactive GPU caches without creating a device or disturbing work.
pub fn trim_memory() {
    let g = STATE.lock().unwrap().ctx.clone();
    if let Some(g) = g {
        resident::clear(&g);
        tiles::trim_idle(&g);
        exec::trim_idle(&g);
    }
}
