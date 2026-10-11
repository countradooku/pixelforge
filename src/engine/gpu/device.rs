//! The GPU context: one high-performance adapter, its device and queue, the
//! capabilities we use, and the health state that drives the CPU fallback.
//!
//! The context is created lazily off the UI thread. Device loss and any
//! uncaptured error flip it to "failed"; the next GPU request then falls back
//! to the CPU and counts the failure. A device that keeps failing is disabled
//! and remembered in the config directory (`gpu_health.txt`).

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

/// Failures (device loss, timeouts, validation errors) before a device is
/// disabled for good.
pub const MAX_FAILURES: u32 = 3;

/// Features PixelForge can use, as found on the adapter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps {
    pub f16: bool,
    pub subgroups: bool,
    pub timestamps: bool,
    pub immediates: bool,
    pub mappable_primary: bool,
    pub pipeline_cache: bool,
    pub unified_memory: bool,
    pub subgroup_min: u32,
    pub subgroup_max: u32,
}

/// What the Info panel and `--bench-gpu` show about the adapter.
#[derive(Clone, Debug, Default)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    pub driver: String,
    pub device_type: String,
    pub vendor: u32,
    pub device: u32,
    /// Memory the GPU can use without paging (Metal's recommended working
    /// set; a conservative guess elsewhere).
    pub vram: u64,
}

impl AdapterSummary {
    /// Identifies the adapter and driver for the cost-model and health caches.
    pub fn key(&self) -> String {
        format!("{}|{}|{:04x}:{:04x}|{}", self.name, self.backend, self.vendor, self.device, self.driver).replace(['\n', '='], " ")
    }
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: AdapterSummary,
    pub caps: Caps,
    pub limits: wgpu::Limits,
    failed: Arc<AtomicBool>,
    reason: Arc<Mutex<Option<String>>>,
    pub(super) pipelines: Mutex<HashMap<super::shaders::PipeKey, Arc<wgpu::ComputePipeline>>>,
    pub(super) modules: Mutex<HashMap<(&'static str, bool), Arc<wgpu::ShaderModule>>>,
    pub(super) cache: Option<wgpu::PipelineCache>,
    pub(super) pool: Mutex<super::exec::Pool>,
    /// Submissions made by this context (for the Info panel).
    pub(super) submitted: AtomicU32,
    /// Read-only tables shared by kernels (unpremultiply, zeros).
    shared: Mutex<HashMap<&'static str, Arc<wgpu::Buffer>>>,
    /// Images resident between operations, and the compositor's tile pool
    /// (per device: their buffers belong to it).
    pub(super) resident: Mutex<super::resident::Cache>,
    pub(super) tiles: Mutex<super::tiles::Pool>,
    /// One compositor frame at a time: a frame's tile uploads must run
    /// before another frame can read those slots.
    pub(super) compose_lock: Mutex<()>,
    /// Foreground composite cache (`compose::Prefix`).
    pub(super) prefix: Mutex<Option<super::compose::Prefix>>,
}

impl Gpu {
    /// Creates a context on the best adapter, or explains why there is none.
    pub fn new() -> Result<Gpu, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let ai = adapter.get_info();
        if ai.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("only a software adapter ({})", ai.name));
        }
        let af = adapter.features();
        let al = adapter.limits();
        // wgpu-hal reports Metal devices with unified memory (Apple Silicon)
        // as integrated, which is what matters here.
        let unified = ai.device_type == wgpu::DeviceType::IntegratedGpu;
        let mut want = wgpu::Features::empty();
        for f in [wgpu::Features::SHADER_F16, wgpu::Features::SUBGROUP, wgpu::Features::TIMESTAMP_QUERY, wgpu::Features::IMMEDIATES, wgpu::Features::PIPELINE_CACHE] {
            if af.contains(f) {
                want |= f;
            }
        }
        // Mapping storage buffers directly only pays off when CPU and GPU
        // share memory; on discrete GPUs it would put them in system RAM.
        if unified && af.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS) {
            want |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        let mut limits = wgpu::Limits::default().using_resolution(al.clone()).using_alignment(al.clone());
        limits.max_buffer_size = al.max_buffer_size;
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_storage_buffers_per_shader_stage = al.max_storage_buffers_per_shader_stage.min(16);
        limits.max_compute_workgroup_storage_size = al.max_compute_workgroup_storage_size;
        limits.max_compute_invocations_per_workgroup = al.max_compute_invocations_per_workgroup;
        limits.max_compute_workgroup_size_x = al.max_compute_workgroup_size_x;
        limits.max_compute_workgroup_size_y = al.max_compute_workgroup_size_y;
        limits.max_compute_workgroups_per_dimension = al.max_compute_workgroups_per_dimension;
        limits.max_immediate_size = if want.contains(wgpu::Features::IMMEDIATES) { al.max_immediate_size.min(128) } else { 0 };
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pixelforge-compute"),
            required_features: want,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .map_err(|e| format!("device creation failed: {e}"))?;
        let caps = Caps {
            f16: want.contains(wgpu::Features::SHADER_F16),
            subgroups: want.contains(wgpu::Features::SUBGROUP),
            timestamps: want.contains(wgpu::Features::TIMESTAMP_QUERY),
            immediates: want.contains(wgpu::Features::IMMEDIATES),
            mappable_primary: want.contains(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
            pipeline_cache: want.contains(wgpu::Features::PIPELINE_CACHE),
            unified_memory: unified,
            subgroup_min: ai.subgroup_min_size,
            subgroup_max: ai.subgroup_max_size,
        };
        let info = AdapterSummary {
            name: ai.name.clone(),
            backend: format!("{:?}", ai.backend),
            driver: [ai.driver.as_str(), ai.driver_info.as_str()].iter().filter(|s| !s.is_empty()).copied().collect::<Vec<_>>().join(" "),
            device_type: format!("{:?}", ai.device_type),
            vendor: ai.vendor,
            device: ai.device,
            vram: vram_estimate(&device, &ai, unified),
        };
        let failed = Arc::new(AtomicBool::new(false));
        let reason = Arc::new(Mutex::new(None));
        {
            let (f, r) = (failed.clone(), reason.clone());
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                *r.lock().unwrap() = Some(format!("uncaptured GPU error: {e}"));
                f.store(true, Ordering::SeqCst);
            }));
            let (f, r) = (failed.clone(), reason.clone());
            device.set_device_lost_callback(move |why, msg| {
                if why != wgpu::DeviceLostReason::Destroyed || !f.load(Ordering::SeqCst) {
                    *r.lock().unwrap() = Some(format!("device lost ({why:?}): {msg}"));
                }
                f.store(true, Ordering::SeqCst);
            });
        }
        let cache = caps.pipeline_cache.then(|| {
            let data = std::fs::read(pipeline_cache_path(&info)).ok();
            // SAFETY: the data was written by `PipelineCache::get_data` for
            // this adapter key; wgpu ignores it if the driver changed.
            unsafe {
                device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor { label: Some("pixelforge"), data: data.as_deref(), fallback: true })
            }
        });
        Ok(Gpu {
            device,
            queue,
            info,
            caps,
            limits: al,
            failed,
            reason,
            pipelines: Mutex::new(HashMap::new()),
            modules: Mutex::new(HashMap::new()),
            cache,
            pool: Mutex::new(super::exec::Pool::default()),
            submitted: AtomicU32::new(0),
            shared: Mutex::new(HashMap::new()),
            resident: Default::default(),
            tiles: Default::default(),
            compose_lock: Mutex::new(()),
            prefix: Mutex::new(None),
        })
    }

    /// A read-only table built once per context.
    pub fn shared(&self, name: &'static str, make: impl FnOnce() -> Vec<u8>) -> Arc<wgpu::Buffer> {
        if let Some(b) = self.shared.lock().unwrap().get(name) {
            return b.clone();
        }
        let b = Arc::new(super::exec::table(self, &make()));
        self.shared.lock().unwrap().insert(name, b.clone());
        b
    }

    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::SeqCst)
    }

    pub fn failure(&self) -> Option<String> {
        self.reason.lock().unwrap().clone()
    }

    /// Marks the context unusable (watchdog timeout, validation error).
    pub fn fail(&self, why: impl Into<String>) {
        let mut r = self.reason.lock().unwrap();
        if r.is_none() {
            *r = Some(why.into());
        }
        self.failed.store(true, Ordering::SeqCst);
    }

    /// Largest single buffer we allocate: the binding limit, kept well inside
    /// the VRAM budget.
    pub fn max_buffer(&self) -> u64 {
        self.limits.max_storage_buffer_binding_size.min(self.limits.max_buffer_size).min(self.budget() / 4).max(16 << 20)
    }

    /// VRAM we allow ourselves: the estimate minus darktable-style headroom
    /// (600 MB, or a quarter on small GPUs) for the display and other apps.
    pub fn budget(&self) -> u64 {
        // Unified memory: the working set is the same RAM the CPU-side
        // document, its history and the CPU fallback live in, so take half
        // (at 268 MP a full-budget pool pushed macOS into paging and a 250 ms
        // dispatch past the watchdog).
        let v = if self.caps.unified_memory { self.info.vram / 2 } else { self.info.vram };
        v.saturating_sub((600u64 << 20).min(v / 4)).max(64 << 20).min(crate::engine::memory::global().limit())
    }

    /// Writes the driver's pipeline cache back to disk (Vulkan only).
    pub fn save_pipeline_cache(&self) {
        if let Some(data) = self.cache.as_ref().and_then(|c| c.get_data()) {
            let p = pipeline_cache_path(&self.info);
            if let Some(d) = p.parent() {
                let _ = std::fs::create_dir_all(d);
            }
            let _ = std::fs::write(p, data);
        }
    }
}

fn pipeline_cache_path(info: &AdapterSummary) -> std::path::PathBuf {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in info.key().bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    crate::engine::prefs::config_dir().join(format!("gpu_pipelines_{h:016x}.bin"))
}

#[cfg(target_vendor = "apple")]
fn vram_estimate(device: &wgpu::Device, ai: &wgpu::AdapterInfo, unified: bool) -> u64 {
    use objc2_metal::MTLDevice;
    // SAFETY: only reads a property of the device's own MTLDevice.
    unsafe { device.as_hal::<wgpu::hal::api::Metal>() }
        .map(|d| d.raw_device().recommendedMaxWorkingSetSize())
        .unwrap_or_else(|| guess_vram(ai, unified))
}

#[cfg(not(target_vendor = "apple"))]
fn vram_estimate(_: &wgpu::Device, ai: &wgpu::AdapterInfo, unified: bool) -> u64 {
    guess_vram(ai, unified)
}

/// wgpu has no portable VRAM query: assume a mid-range card, or a typical
/// shared pool for integrated GPUs.
fn guess_vram(ai: &wgpu::AdapterInfo, unified: bool) -> u64 {
    if unified || ai.device_type == wgpu::DeviceType::IntegratedGpu { 2 << 30 } else { 4 << 30 }
}

// ----- persistent health -------------------------------------------------------------------

fn health_path() -> std::path::PathBuf {
    crate::engine::prefs::config_dir().join("gpu_health.txt")
}

/// Per-adapter failure counts, `key=failures` lines.
fn read_health() -> HashMap<String, u32> {
    std::fs::read_to_string(health_path())
        .map(|t| t.lines().filter_map(|l| l.rsplit_once('=')).filter_map(|(k, v)| Some((k.to_string(), v.trim().parse().ok()?))).collect())
        .unwrap_or_default()
}

fn write_health(h: &HashMap<String, u32>) {
    let mut keys: Vec<_> = h.iter().collect();
    keys.sort();
    let text: String = keys.into_iter().map(|(k, v)| format!("{k}={v}\n")).collect();
    let _ = std::fs::create_dir_all(crate::engine::prefs::config_dir());
    let _ = std::fs::write(health_path(), text);
}

pub fn failures(key: &str) -> u32 {
    read_health().get(key).copied().unwrap_or(0)
}

/// Counts one failure of this adapter; returns the new total.
pub fn record_failure(key: &str) -> u32 {
    let mut h = read_health();
    let n = h.entry(key.to_string()).or_insert(0);
    *n += 1;
    let n = *n;
    write_health(&h);
    n
}

/// Clears the failure count (a clean run, or the user's Benchmark GPU).
pub fn clear_failures(key: &str) {
    let mut h = read_health();
    if h.remove(key).is_some() {
        write_health(&h);
    }
}

// ----- a minimal executor for wgpu's native futures ------------------------------------------

struct ThreadWaker(std::thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Drives a future to completion on this thread. wgpu's native futures
/// (adapter, device, error scopes) resolve without an event loop.
pub fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::park_timeout(std::time::Duration::from_millis(1));
    }
}
