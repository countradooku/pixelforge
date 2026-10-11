//! The CPU-versus-GPU cost model.
//!
//! For an operation on `px` pixels:
//! - GPU ms = fixed submission cost + `px x 8 B` over the measured transfer
//!   rate (gather + upload + readback + new tiles) + kernel ns/px x px;
//! - CPU ms = CPU ns/px x px.
//!
//! Rates are per operation and per spatial-size bucket (the filter's padding
//! at that level: tiny, small, medium, large, huge), because GPU and CPU
//! algorithms scale differently with the radius. They start from priors
//! measured with `--bench-gpu --write-priors` on the reference machine
//! (`priors.txt`), scaled by a startup micro-benchmark of this machine's GPU
//! and CPU, and every real run refines them (moving average). The model is
//! cached per adapter + driver + app version in the config directory
//! (`gpu_cost.txt`) and recalibrated when any of them changes. Auto mode
//! runs the GPU only where it's predicted to win by 10 %.
//! End-to-end rates are also separated by power-of-two pixel-count ranges;
//! the transfer-plus-resident formula is the fallback for unmeasured sizes.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use super::device::Gpu;
use crate::engine::filters::Filter;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rates {
    pub gpu_ns_px: f64,
    pub cpu_ns_px: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// Device key + app version the numbers belong to.
    pub key: String,
    /// Fixed cost of one GPU operation (submission, mapping, waits), ms.
    pub fixed_ms: f64,
    /// Host <-> device cost (gather + upload + readback + new tiles), ns/byte.
    pub transfer_ns_b: f64,
    /// This machine's GPU / CPU time relative to the reference machine, and
    /// the raw probe times they came from (ms).
    pub gpu_scale: f64,
    pub cpu_scale: f64,
    pub gpu_probe_ms: f64,
    pub cpu_probe_ms: f64,
    /// Learned rates per op key (override the scaled priors).
    pub ops: HashMap<String, Rates>,
}

impl Default for Model {
    fn default() -> Self {
        Model { key: String::new(), fixed_ms: 0.8, transfer_ns_b: 0.03, gpu_scale: 1.0, cpu_scale: 1.0, gpu_probe_ms: 0.0, cpu_probe_ms: 0.0, ops: HashMap::new() }
    }
}

static MODEL: Mutex<Option<Model>> = Mutex::new(None);

pub fn app_key(g: &Gpu) -> String {
    format!("{}|pixelforge {}|cost-v2", g.info.key(), env!("CARGO_PKG_VERSION"))
}

fn path() -> std::path::PathBuf {
    crate::engine::prefs::config_dir().join("gpu_cost.txt")
}

/// Spatial-size bucket of a filter at a level (its padding).
pub fn bucket(pad: usize) -> &'static str {
    match pad {
        0 => "p0",
        1..=4 => "p4",
        5..=24 => "p24",
        25..=96 => "p96",
        97..=400 => "p400",
        _ => "pmax",
    }
}

/// The key rates are stored under: name (+ algorithm variant) + size bucket.
pub fn op_key(f: &Filter, level: u32) -> String {
    format!("{}{} {}", f.name(), variant(f), bucket(f.pad(level)))
}

/// Settings that switch a filter to a different algorithm with different
/// CPU and GPU costs (one name, rates an order of magnitude apart).
fn variant(f: &Filter) -> &'static str {
    use crate::engine::filters::blur::Remove;
    use crate::engine::filters::extra::{BlurFx, MapSource};
    let fx = |fx: &BlurFx| if *fx == BlurFx::default() { "" } else { " +fx" };
    match f {
        Filter::SmartSharpen { opts, .. } => match opts.remove {
            Remove::Gaussian if opts.is_plain() => "",
            Remove::Gaussian => " fades",
            Remove::Lens => " lens",
            Remove::Motion => " motion",
        },
        Filter::OilPaint { oil, .. } if oil.stylization > 0.0 => " strokes",
        Filter::LensBlur { opts, .. } if opts.blades >= 3 => " polygon",
        Filter::MotionBlur { angle, .. } if angle.rem_euclid(90.0) != 0.0 => " angled",
        Filter::Displace { map: MapSource::Clouds, .. } => " clouds",
        Filter::Displace { map: MapSource::File { .. }, .. } => " file",
        Filter::Gradient(g) => match g.kind {
            crate::engine::gradient::GradKind::Linear => "",
            crate::engine::gradient::GradKind::Radial => " radial",
            crate::engine::gradient::GradKind::Angle => " angle",
            crate::engine::gradient::GradKind::Reflected => " reflected",
            crate::engine::gradient::GradKind::Diamond => " diamond",
        },
        Filter::IrisBlur { fx: e, .. } | Filter::FieldBlur { fx: e, .. } | Filter::TiltShift { fx: e, .. } | Filter::SpinBlur { fx: e, .. } | Filter::PathBlur { fx: e, .. } => fx(e),
        _ => "",
    }
}

/// Above this many pixels an image no longer fits one band: the GPU runs in
/// pipelined bands with nothing resident, and CPU and GPU scale differently
/// from small images (cache, transfers, band aprons), so such runs have
/// their own rates (`<key> L`, GPU end to end, transfers included).
pub const LARGE_PX: u64 = 64 << 20;

pub fn large_key(f: &Filter, level: u32) -> String {
    format!("{} L", op_key(f, level))
}

/// End-to-end rates for [2^(n-1), 2^n) pixels (E0 is the empty image). Transfer and
/// banding costs change with the working set, even below the large-image
/// threshold. A preview must not overwrite a full-document observation.
pub fn area_key(f: &Filter, level: u32, px: u64) -> String {
    let exponent = u64::BITS - px.leading_zeros();
    format!("{} E{exponent}", op_key(f, level))
}

/// Reference-machine rates (Apple M5 Pro), written by
/// `pixelforge --bench-gpu --write-priors`.
fn priors() -> &'static HashMap<String, Rates> {
    static P: std::sync::OnceLock<HashMap<String, Rates>> = std::sync::OnceLock::new();
    P.get_or_init(|| parse_rates(include_str!("priors.txt")))
}

fn parse_rates(t: &str) -> HashMap<String, Rates> {
    t.lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with('@'))
        .filter_map(|l| {
            let (k, v) = l.rsplit_once('=')?;
            let mut it = v.split_whitespace().filter_map(|x| x.parse::<f64>().ok());
            Some((k.trim().to_string(), Rates { gpu_ns_px: it.next()?, cpu_ns_px: it.next()? }))
        })
        .collect()
}

pub fn rates_to_text(ops: &HashMap<String, Rates>) -> String {
    let mut v: Vec<_> = ops.iter().collect();
    v.sort_by(|a, b| a.0.cmp(b.0));
    v.into_iter().map(|(k, r)| format!("{k}={:.4} {:.4}\n", r.gpu_ns_px, r.cpu_ns_px)).collect()
}

impl Model {
    pub fn to_text(&self) -> String {
        format!(
            "key={}\nfixed_ms={}\ntransfer_ns_b={}\ngpu_scale={}\ncpu_scale={}\ngpu_probe_ms={}\ncpu_probe_ms={}\n{}",
            self.key,
            self.fixed_ms,
            self.transfer_ns_b,
            self.gpu_scale,
            self.cpu_scale,
            self.gpu_probe_ms,
            self.cpu_probe_ms,
            rates_to_text(&self.ops).lines().map(|l| format!("op {l}\n")).collect::<String>()
        )
    }

    pub fn parse(t: &str) -> Model {
        let mut m = Model::default();
        for l in t.lines() {
            if let Some(op) = l.strip_prefix("op ") {
                m.ops.extend(parse_rates(op));
                continue;
            }
            let Some((k, v)) = l.split_once('=') else {
                continue;
            };
            let f = v.parse::<f64>().ok();
            match k {
                "key" => m.key = v.to_string(),
                "fixed_ms" => m.fixed_ms = f.unwrap_or(m.fixed_ms),
                "transfer_ns_b" => m.transfer_ns_b = f.unwrap_or(m.transfer_ns_b),
                "gpu_scale" => m.gpu_scale = f.unwrap_or(1.0),
                "cpu_scale" => m.cpu_scale = f.unwrap_or(1.0),
                "gpu_probe_ms" => m.gpu_probe_ms = f.unwrap_or(0.0),
                "cpu_probe_ms" => m.cpu_probe_ms = f.unwrap_or(0.0),
                _ => {}
            }
        }
        m
    }

    /// Rates for an op key: learned, else the scaled prior.
    pub fn rates(&self, key: &str) -> Option<Rates> {
        self.ops.get(key).copied().or_else(|| priors().get(key).map(|r| Rates { gpu_ns_px: r.gpu_ns_px * self.gpu_scale, cpu_ns_px: r.cpu_ns_px * self.cpu_scale }))
    }

    /// Predicted (gpu_ms, cpu_ms) for `filter` on `px` pixels at `level`,
    /// or None when nothing is known about it yet.
    pub fn predict(&self, filter: &Filter, px: u64, level: u32) -> Option<(f64, f64)> {
        let pxf = px as f64;
        if let Some(r) = self.rates(&area_key(filter, level, px)).filter(|r| r.gpu_ns_px.is_finite() && r.cpu_ns_px.is_finite()) {
            return Some((self.fixed_ms + pxf * r.gpu_ns_px / 1e6, pxf * r.cpu_ns_px / 1e6));
        }
        if px > LARGE_PX {
            if let Some(r) = self.rates(&large_key(filter, level)).filter(|r| r.gpu_ns_px.is_finite() && r.cpu_ns_px.is_finite()) {
                return Some((self.fixed_ms + pxf * r.gpu_ns_px / 1e6, pxf * r.cpu_ns_px / 1e6));
            }
        }
        let r = self.rates(&op_key(filter, level))?;
        let gpu = self.fixed_ms + pxf * 8.0 * self.transfer_ns_b / 1e6 + pxf * r.gpu_ns_px / 1e6;
        let cpu = pxf * r.cpu_ns_px / 1e6;
        Some((gpu, cpu))
    }
}

/// Loads the cached model for this device, or calibrates a fresh one.
pub fn load(g: &Arc<Gpu>) {
    let key = app_key(g);
    let cached = std::fs::read_to_string(path()).map(|t| Model::parse(&t)).ok().filter(|m| m.key == key);
    let m = match cached {
        Some(m) => m,
        None => {
            let m = calibrate(g);
            *MODEL.lock().unwrap() = Some(m.clone());
            save();
            m
        }
    };
    *MODEL.lock().unwrap() = Some(m);
}

pub fn save() {
    if let Some(m) = MODEL.lock().unwrap().as_ref() {
        let _ = std::fs::create_dir_all(crate::engine::prefs::config_dir());
        let _ = std::fs::write(path(), m.to_text());
    }
}

pub fn model() -> Option<Model> {
    MODEL.lock().unwrap().clone()
}

pub fn set_model(m: Model) {
    *MODEL.lock().unwrap() = Some(m);
}

/// Forgets the cached model (driver or app change is detected on load; the
/// Benchmark GPU button forces it).
pub fn reset() {
    *MODEL.lock().unwrap() = None;
    let _ = std::fs::remove_file(path());
}

/// Timings of the calibration probes on the machine the priors came from
/// (ms: GPU Gaussian r=20 on 2048² resident, CPU Gaussian r=20 on 2048²),
/// the `@ref=` line `--write-priors` puts in `priors.txt`.
pub fn ref_probes() -> (f64, f64) {
    include_str!("priors.txt")
        .lines()
        .find_map(|l| {
            let mut it = l.strip_prefix("@ref=")?.split_whitespace().filter_map(|v| v.parse::<f64>().ok());
            Some((it.next()?, it.next()?))
        })
        .unwrap_or((2.7, 2.7))
}

/// This machine's speed relative to the reference. Within ±25 % it *is* the
/// reference class: the probes alone wander ~±12 % between runs, and
/// scaling by that noise flips near-parity decisions.
fn scale(probe: f64, reference: f64) -> f64 {
    let s = probe / reference;
    if (0.8..=1.25).contains(&s) { 1.0 } else { s.clamp(0.05, 50.0) }
}

/// The startup micro-benchmark (~0.1–0.3 s, off the UI thread): fixed GPU
/// round-trip cost, transfer rate, and GPU / CPU speed relative to the
/// reference machine.
pub fn calibrate(g: &Arc<Gpu>) -> Model {
    let mut m = Model { key: app_key(g), ..Model::default() };
    let side = 2048u32;
    let img = crate::engine::ops::procedural(side, side);
    let blur = Filter::GaussianBlur { radius: 20.0 };
    let inv = Filter::Invert;
    let px = (side * side) as f64;
    // Best of several back-to-back runs after a warm-up: GPU clocks ramp up
    // under load and the first runs pay for allocations.
    let best = |n: usize, f: &mut dyn FnMut() -> f64| -> f64 {
        for _ in 0..2 {
            f();
        }
        (0..n).map(|_| f()).fold(f64::MAX, f64::min)
    };
    let timed = |f: &mut dyn FnMut()| {
        let t = Instant::now();
        f();
        t.elapsed().as_secs_f64() * 1e3
    };
    // Fixed cost: a one-tile op end to end.
    let small = crate::engine::ops::procedural(256, 256);
    m.fixed_ms = best(7, &mut || timed(&mut || drop(super::apply_filter_on(g, &small, &inv, None, &Default::default(), &Default::default()))));
    // Transfer: a point op end to end on 4 MP minus its resident time.
    let e2e = best(5, &mut || timed(&mut || drop(super::apply_filter_on(g, &img, &inv, None, &Default::default(), &Default::default()))));
    let kernel = super::kernels::plan(g, &inv, 0, super::kernels::Target::Commit).and_then(|k| super::run::resident_ms(g, &img, &inv, k.as_ref(), 5).ok()).map_or(0.0, |r| r.0);
    m.transfer_ns_b = ((e2e - kernel).max(0.05) * 1e6 / (px * 8.0)).clamp(0.002, 1.0);
    // GPU speed.
    if let Some((r, _)) = super::kernels::plan(g, &blur, 0, super::kernels::Target::Commit).and_then(|k| super::run::resident_ms(g, &img, &blur, k.as_ref(), 7).ok()) {
        m.gpu_probe_ms = r;
        m.gpu_scale = scale(r, ref_probes().0);
    }
    // CPU speed.
    let cpu = best(5, &mut || timed(&mut || drop(crate::engine::ops::apply_filter_cpu(&img, &blur, None, &Default::default(), None))));
    m.cpu_probe_ms = cpu;
    m.cpu_scale = scale(cpu, ref_probes().1);
    m
}

fn with_model<T>(g: &Arc<Gpu>, f: impl FnOnce(&Model) -> T) -> Option<T> {
    let need = MODEL.lock().unwrap().as_ref().is_none_or(|m| m.key != app_key(g));
    if need {
        load(g);
    }
    MODEL.lock().unwrap().as_ref().map(f)
}

/// Why Auto mode keeps this filter on the CPU, if it does. Unknown
/// operations run on the GPU once; the observation then decides.
pub fn veto(g: &Arc<Gpu>, filter: &Filter, px: u64) -> Option<String> {
    veto_at(g, filter, px, 0)
}

pub fn veto_at(g: &Arc<Gpu>, filter: &Filter, px: u64, level: u32) -> Option<String> {
    let (gpu, cpu) = with_model(g, |m| m.predict(filter, px, level)).flatten()?;
    (gpu * 1.1 > cpu).then(|| format!("cost model: CPU {cpu:.1} ms vs GPU {gpu:.1} ms"))
}

/// Why a live preview of `px` pixels at `level` stays on the CPU. Spatial
/// parameters shrink with the level, which the size bucket captures.
pub fn veto_preview(g: &Arc<Gpu>, filter: &Filter, px: u64, level: u32) -> Option<String> {
    veto_at(g, filter, px, level)
}

const EMA: f64 = 0.3;

fn observe(filter: &Filter, px: u64, level: u32, ms: f64, gpu: bool) {
    // Tiny runs are dominated by noise and fixed costs.
    if px < 1 << 20 {
        return;
    }
    let mut lock = MODEL.lock().unwrap();
    let Some(m) = lock.as_mut() else { return };
    m.observe(filter, px, level, ms, gpu);
}

impl Model {
    pub(super) fn observe(&mut self, filter: &Filter, px: u64, level: u32, ms: f64, gpu: bool) {
        if px < 1 << 20 || !ms.is_finite() || ms < 0.0 {
            return;
        }
        let key = area_key(filter, level, px);
        let pxf = px as f64;
        // Seed the unobserved side from the prediction, including transfers.
        let prior = self.predict(filter, px, level).map(|(g, c)| Rates { gpu_ns_px: (g - self.fixed_ms).max(0.0) * 1e6 / pxf, cpu_ns_px: c * 1e6 / pxf });
        let fixed = self.fixed_ms;
        let r = self.ops.entry(key).or_insert_with(|| prior.unwrap_or(Rates { gpu_ns_px: f64::NAN, cpu_ns_px: f64::NAN }));
        let blend = |old: f64, new: f64| {
            if old.is_finite() { old * (1.0 - EMA) + new * EMA } else { new }
        };
        if gpu {
            let k = ((ms - fixed).max(0.0) * 1e6 / pxf).max(0.001);
            r.gpu_ns_px = blend(r.gpu_ns_px, k);
        } else {
            r.cpu_ns_px = blend(r.cpu_ns_px, ms * 1e6 / pxf);
        }
    }
}

pub fn observe_cpu(filter: &Filter, px: u64, ms: f64) {
    observe(filter, px, 0, ms, false);
}

pub fn observe_gpu(_g: &Gpu, filter: &Filter, px: u64, ms: f64) {
    observe(filter, px, 0, ms, true);
}

// ----- operations that aren't filters (Select and Mask, Liquify, ...) ------------------------

/// [`veto`] for a named operation with its own prior rates (reference
/// machine; scaled to this one by the calibration).
pub fn veto_named(g: &Gpu, op: &str, prior: Rates, px: u64) -> Option<String> {
    if MODEL.lock().unwrap().as_ref().is_none_or(|m| m.key != app_key(g)) {
        return None;
    }
    let lock = MODEL.lock().unwrap();
    let m = lock.as_ref()?;
    let r = m.ops.get(op).copied().unwrap_or(Rates { gpu_ns_px: prior.gpu_ns_px * m.gpu_scale, cpu_ns_px: prior.cpu_ns_px * m.cpu_scale });
    let pxf = px as f64;
    let gpu = m.fixed_ms + pxf * 8.0 * m.transfer_ns_b / 1e6 + pxf * r.gpu_ns_px / 1e6;
    let cpu = pxf * r.cpu_ns_px / 1e6;
    (gpu > cpu * 0.9).then(|| format!("cost model: CPU {cpu:.1} ms vs GPU {gpu:.1} ms"))
}

/// Refines a named operation's rates from a real run.
pub fn observe_named(op: &str, prior: Rates, px: u64, ms: f64, gpu: bool) {
    if px < 1 << 20 {
        return;
    }
    let mut lock = MODEL.lock().unwrap();
    let Some(m) = lock.as_mut() else { return };
    let (fixed, transfer) = (m.fixed_ms, m.transfer_ns_b);
    let r = m.ops.entry(op.to_string()).or_insert(prior);
    let pxf = px as f64;
    if gpu {
        let k = ((ms - fixed - pxf * 8.0 * transfer / 1e6).max(0.0) * 1e6 / pxf).max(0.001);
        r.gpu_ns_px = r.gpu_ns_px * (1.0 - EMA) + k * EMA;
    } else {
        r.cpu_ns_px = r.cpu_ns_px * (1.0 - EMA) + ms * 1e6 / pxf * EMA;
    }
}

// ----- retouching ops (not filters) ------------------------------------------------------
//
// Healing blends and PatchMatch completion are priced per *unit* (solved
// cells, region pixels): GPU ms = fixed submission cost + the op's own fixed
// cost (its dispatch count) + units x GPU ns; CPU ms = units x CPU ns. The
// rates start from `--bench-gpu` priors and are refined like the filters'.

/// (GPU fixed ms, rates per unit) measured with `--bench-gpu 4096 retouch`
/// on the reference machine (Apple M5 Pro, under the shared load of other
/// builds; units: solved crop cells for the blends, region pixels for
/// PatchMatch). Crossovers: Fast blends from ~75 k cells (a ~270 px dab),
/// Exact from ~375 k cells (~610 px), PatchMatch from a ~400 k px region.
pub fn prior_op(op: &str) -> Option<(f64, Rates)> {
    match op {
        // Fitted to the quiet `--bench-gpu` rows (106², 306², 506² cells).
        OP_HEAL => Some((1.6, Rates { gpu_ns_px: 11.0, cpu_ns_px: 40.0 })),
        OP_HEAL_EXACT => Some((3.0, Rates { gpu_ns_px: 29.0, cpu_ns_px: 42.0 })),
        OP_PATCHMATCH => Some((30.0, Rates { gpu_ns_px: 50.0, cpu_ns_px: 140.0 })),
        _ => None,
    }
}

pub const OP_HEAL: &str = "Healing blend";
pub const OP_HEAL_EXACT: &str = "Healing blend (Exact)";
pub const OP_PATCHMATCH: &str = "PatchMatch fill";

impl Model {
    /// Predicted (gpu_ms, cpu_ms) for a retouching op on `units`.
    pub fn predict_op(&self, op: &str, units: u64) -> (f64, f64) {
        let (fixed, prior) = prior_op(op).unwrap_or((1.0, Rates { gpu_ns_px: 1.0, cpu_ns_px: 1.0 }));
        let r = self.ops.get(op).copied().unwrap_or(prior);
        let u = units as f64;
        (self.fixed_ms + fixed + u * r.gpu_ns_px / 1e6, u * r.cpu_ns_px / 1e6)
    }
}

/// Why Auto mode keeps a retouching op on the CPU, if it does.
pub fn veto_op(g: &Gpu, op: &str, units: u64) -> Option<String> {
    let lock = MODEL.lock().unwrap();
    // The model is loaded with the context; until then, no opinion.
    let m = lock.as_ref().filter(|m| m.key == app_key(g))?;
    let (gpu, cpu) = m.predict_op(op, units);
    (gpu > cpu * 0.9).then(|| format!("cost model: CPU {cpu:.1} ms vs GPU {gpu:.1} ms"))
}

/// Refines a retouching op's rate from a real run of `units` (large runs only).
pub fn observe_op(op: &str, units: u64, ms: f64, gpu: bool) {
    if units < 1 << 18 {
        return;
    }
    let mut lock = MODEL.lock().unwrap();
    let Some(m) = lock.as_mut() else { return };
    let (fixed, prior) = prior_op(op).unwrap_or((1.0, Rates { gpu_ns_px: 1.0, cpu_ns_px: 1.0 }));
    let base = m.fixed_ms + fixed;
    let r = m.ops.entry(op.to_string()).or_insert(prior);
    let u = units as f64;
    if gpu {
        let k = ((ms - base).max(0.0) * 1e6 / u).max(0.01);
        r.gpu_ns_px = r.gpu_ns_px * (1.0 - EMA) + k * EMA;
    } else {
        r.cpu_ns_px = r.cpu_ns_px * (1.0 - EMA) + ms * 1e6 / u * EMA;
    }
}

// ----- non-filter operations (Image Size, Super Zoom, transforms) ---------------------------

/// Rates for a whole-image operation keyed by its name (e.g. "Image Size
/// (Bicubic Sharper (reduction))"), measured with `--bench-gpu` on the
/// reference machine (Apple M5 Pro), used until this machine has its own.
pub fn prior_xop(op: &str) -> Option<Rates> {
    super::resample::prior(op)
}

impl Model {
    /// Predicted (gpu_ms, cpu_ms) for operation `op` writing `px` pixels and
    /// moving `bytes` between host and device.
    ///
    /// Resampling GPU rates are end to end (the gathers read the tiles in
    /// shared memory, so the generic transfer rate overstates them); `bytes`
    /// is kept for the callers' accounting.
    pub fn predict_xop(&self, op: &str, px: u64, _bytes: u64) -> (f64, f64) {
        let r = self.ops.get(op).copied().or_else(|| prior_xop(op)).unwrap_or(Rates { gpu_ns_px: 1.0, cpu_ns_px: 1.0 });
        let pxf = px as f64;
        (self.fixed_ms + pxf * r.gpu_ns_px / 1e6, pxf * r.cpu_ns_px / 1e6)
    }
}

/// Why Auto mode keeps operation `op` on the CPU, if it does.
pub fn veto_xop(g: &Gpu, op: &str, px: u64, bytes: u64) -> Option<String> {
    let lock = MODEL.lock().unwrap();
    // The model loads with the context; until then, no opinion.
    let m = lock.as_ref().filter(|m| m.key == app_key(g))?;
    let (gpu, cpu) = m.predict_xop(op, px, bytes);
    (gpu > cpu * 0.9).then(|| format!("cost model: CPU {cpu:.1} ms vs GPU {gpu:.1} ms"))
}

/// Refines the rates of `op` from a real run (`gpu`: it ran on the GPU).
pub fn observe_xop(op: &str, px: u64, _bytes: u64, ms: f64, gpu: bool) {
    if px < 1 << 20 {
        return;
    }
    let mut lock = MODEL.lock().unwrap();
    let Some(m) = lock.as_mut() else { return };
    let fixed = m.fixed_ms;
    let r = m.ops.entry(op.to_string()).or_insert_with(|| prior_xop(op).unwrap_or(Rates { gpu_ns_px: 1.0, cpu_ns_px: 1.0 }));
    let pxf = px as f64;
    if gpu {
        // End to end, like the priors (see `predict_xop`).
        let k = ((ms - fixed).max(0.0) * 1e6 / pxf).max(0.01);
        r.gpu_ns_px = r.gpu_ns_px * (1.0 - EMA) + k * EMA;
    } else {
        r.cpu_ns_px = r.cpu_ns_px * (1.0 - EMA) + ms * 1e6 / pxf * EMA;
    }
}
