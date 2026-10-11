//! Buffers, dispatches and submissions.
//!
//! - A pool recycles buffers by size and usage, within the VRAM budget.
//! - Every operation runs inside error scopes and is waited on with a
//!   watchdog (`PF_GPU_WATCHDOG_MS`, default 2 s): a validation error, an
//!   out-of-memory error, a timeout or a lost device returns an error, and
//!   the caller falls back to the CPU.
//! - Passes can be timed with timestamp queries (whole passes only: Apple
//!   GPUs can't time inside a pass).

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::device::{Gpu, block_on};
use super::shaders::PipeKey;

#[derive(Clone, Debug, PartialEq)]
pub enum GpuError {
    /// The GPU path can't run this filter or these parameters.
    Unsupported,
    /// The work doesn't fit the VRAM budget.
    Vram,
    Validation(String),
    Timeout,
    Lost(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::Unsupported => write!(f, "not supported on the GPU"),
            GpuError::Vram => write!(f, "VRAM budget exceeded"),
            GpuError::Validation(e) => write!(f, "GPU validation error: {e}"),
            GpuError::Timeout => write!(f, "GPU watchdog timeout"),
            GpuError::Lost(e) => write!(f, "GPU lost: {e}"),
        }
    }
}

impl GpuError {
    /// Errors that say something about the device's health (as opposed to
    /// "this filter can't run here").
    pub fn is_fault(&self) -> bool {
        matches!(self, GpuError::Validation(_) | GpuError::Timeout | GpuError::Lost(_))
    }
}

pub type GResult<T> = Result<T, GpuError>;

pub fn watchdog() -> Duration {
    let ms = std::env::var("PF_GPU_WATCHDOG_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000u64);
    Duration::from_millis(ms)
}

// ----- buffer pool -------------------------------------------------------------------------

struct Pooled {
    reservation: Option<crate::engine::memory::Reservation>,
    buf: wgpu::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
    used: Instant,
}

#[derive(Default)]
pub struct Pool {
    free: Vec<Pooled>,
    /// Bytes in every buffer we created and haven't destroyed.
    pub live: u64,
}

pub const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE.union(wgpu::BufferUsages::COPY_SRC).union(wgpu::BufferUsages::COPY_DST);

/// A pooled buffer; goes back to the pool when dropped.
pub struct Buf {
    reservation: Option<crate::engine::memory::Reservation>,
    buf: Option<wgpu::Buffer>,
    pub size: u64,
    usage: wgpu::BufferUsages,
    gpu: Arc<Gpu>,
    /// wgpu reports unmapping an unmapped buffer as an error, so track it.
    mapped: std::sync::atomic::AtomicBool,
}

impl Buf {
    /// Maps `..n` bytes and waits.
    pub fn map(&self, n: u64, mode: wgpu::MapMode) -> GResult<()> {
        let r = map_wait(&self.gpu, self.slice(..n), mode);
        if r.is_ok() {
            self.mapped.store(true, Ordering::SeqCst);
        }
        r
    }


    /// The context the buffer belongs to.
    pub fn gpu(&self) -> Arc<Gpu> {
        self.gpu.clone()
    }

    pub fn unmap_now(&self) {
        if self.mapped.swap(false, Ordering::SeqCst) {
            self.unmap();
        }
    }
}

impl std::ops::Deref for Buf {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        self.buf.as_ref().unwrap()
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            if self.mapped.load(Ordering::SeqCst) {
                b.unmap();
            }
            let mut p = self.gpu.pool.lock().unwrap();
            p.free.push(Pooled { reservation: self.reservation.take(), buf: b, size: self.size, usage: self.usage, used: Instant::now() });
            // Idle buffers are a cache, not a reservation: past a quarter of
            // the budget the oldest go (on unified memory they are the RAM
            // the CPU-side document needs).
            let cap = (self.gpu.budget() / 4).min(crate::engine::memory::global().available());
            let mut idle: u64 = p.free.iter().map(|b| b.size).sum();
            while idle > cap {
                let i = p.free.iter().enumerate().min_by_key(|(_, b)| b.used).map(|(i, _)| i).unwrap();
                let b = p.free.swap_remove(i);
                idle -= b.size;
                p.live -= b.size;
                b.buf.destroy();
            }
        }
    }
}

fn round_size(size: u64) -> u64 {
    // 1 MB granularity above 4 MB so similar bands share buffers.
    let size = size.max(256);
    if size > 4 << 20 { size.div_ceil(1 << 20) << 20 } else { size.next_multiple_of(256) }
}

/// Takes a buffer of at least `size` bytes from the pool or creates one.
/// Mappable buffers come back unmapped.
pub fn buffer(g: &Arc<Gpu>, size: u64, usage: wgpu::BufferUsages) -> GResult<Buf> {
    let size = round_size(size);
    if size > g.limits.max_buffer_size {
        return Err(GpuError::Vram);
    }
    let memory = crate::engine::memory::global();
    if memory.available() < size { super::resident::trim(g, size); }
    let reservation;
    {
        let mut p = g.pool.lock().unwrap();
        let best = p.free.iter().enumerate().filter(|(_, b)| b.usage == usage && b.size >= size && b.size <= size * 2).min_by_key(|(_, b)| b.size).map(|(i, _)| i);
        if let Some(i) = best {
            let b = p.free.swap_remove(i);
            return Ok(Buf { reservation: b.reservation, buf: Some(b.buf), size: b.size, usage, gpu: g.clone(), mapped: Default::default() });
        }
        // Make room: drop the least recently used free buffers first.
        let budget = g.budget();
        while (p.live + size > budget || memory.available() < size) && !p.free.is_empty() {
            let i = p.free.iter().enumerate().min_by_key(|(_, b)| b.used).map(|(i, _)| i).unwrap();
            let b = p.free.swap_remove(i);
            p.live -= b.size;
            b.buf.destroy();
        }
        if p.live + size > budget {
            return Err(GpuError::Vram);
        }
        reservation = memory.reserve_gpu(size).ok_or(GpuError::Vram)?;
        p.live += size;
    }
    let scope = g.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buf = g.device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false });
    if block_on(scope.pop()).is_some() {
        g.pool.lock().unwrap().live -= size;
        return Err(GpuError::Vram);
    }
    Ok(Buf { reservation: Some(reservation), buf: Some(buf), size, usage, gpu: g.clone(), mapped: Default::default() })
}


/// A small read-only storage buffer (LUTs, kernels, tables).
pub fn table(g: &Gpu, data: &[u8]) -> wgpu::Buffer {
    upload(g, data, wgpu::BufferUsages::STORAGE, "table")
}

/// A buffer filled through the queue. Unlike a buffer mapped at creation,
/// this can't panic if the device was lost meanwhile (the error goes to
/// the uncaptured-error handler, which fails the context).
fn upload(g: &Gpu, data: &[u8], usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
    let mut v = data.to_vec();
    v.resize(data.len().next_multiple_of(16).max(16), 0);
    let b = g.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: v.len() as u64, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    g.queue.write_buffer(&b, 0, &v);
    b
}

// ----- dispatch ----------------------------------------------------------------------------

/// Pass timing with timestamp queries.
pub struct Timer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    cap: u32,
    next: u32,
    labels: Vec<String>,
    /// The last pass is the flush pass (not reported).
    flush: bool,
}

impl Timer {
    pub fn new(g: &Gpu, passes: u32) -> Option<Timer> {
        if !g.caps.timestamps {
            return None;
        }
        let cap = passes * 2;
        Some(Timer {
            set: g.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("timer"), ty: wgpu::QueryType::Timestamp, count: cap }),
            resolve: g.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: cap as u64 * 8,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            read: g.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: cap as u64 * 8, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            cap,
            next: 0,
            labels: Vec::new(),
            flush: false,
        })
    }

    fn writes(&mut self, label: &str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if self.next + 2 > self.cap {
            return None;
        }
        self.labels.push(label.to_string());
        let i = self.next;
        self.next += 2;
        Some(wgpu::ComputePassTimestampWrites { query_set: &self.set, beginning_of_pass_write_index: Some(i), end_of_pass_write_index: Some(i + 1) })
    }

    /// Timestamp writes for a pass recorded outside [`dispatch`].
    pub fn pass_writes(&mut self, label: &str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        self.writes(label)
    }

    /// Resolves the queries. A last empty timed pass makes sure the real
    /// last pass's end timestamp is written (Metal may skip it otherwise).
    pub fn resolve(&mut self, g: &Gpu, enc: &mut wgpu::CommandEncoder) {
        if self.next > 0 && self.next + 2 <= self.cap {
            let key = PipeKey { module: "probe", entry: "copy", f16: false, consts: Vec::new() };
            let pipe = g.pipeline(&key).ok();
            let w = self.writes("flush");
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("flush"), timestamp_writes: w });
            if let Some(p) = &pipe {
                pass.set_pipeline(p);
            }
            drop(pass);
            self.flush = true;
        }
        if self.next > 0 {
            enc.resolve_query_set(&self.set, 0..self.next, &self.resolve, 0);
            enc.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, self.next as u64 * 8);
        }
    }

    /// Per-pass GPU times in ms, with the pass labels (after the submission ran).
    pub fn read_passes(&self, g: &Gpu) -> Option<Vec<(String, f64)>> {
        if self.next == 0 {
            return Some(Vec::new());
        }
        let slice = self.read.slice(..self.next as u64 * 8);
        map_wait(g, slice, wgpu::MapMode::Read).ok()?;
        let ts: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        self.read.unmap();
        let period = g.queue.get_timestamp_period() as f64;
        let n = ts.len() / 2 - self.flush as usize;
        Some(ts.chunks(2).take(n).enumerate().map(|(i, p)| (self.labels.get(i).cloned().unwrap_or_default(), p[1].saturating_sub(p[0]) as f64 * period / 1e6)).collect())
    }

    /// Total GPU time of the timed passes, in ms (after the submission ran).
    pub fn read_ms(&self, g: &Gpu) -> Option<f64> {
        Some(self.read_passes(g)?.iter().map(|p| p.1).sum())
    }
}

/// Records one compute dispatch in its own pass. `params` (if any) is bound
/// as a uniform at binding 0; `bufs` are `(binding, buffer)` pairs.
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    g: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    timer: Option<&mut Timer>,
    key: &PipeKey,
    params: &[u32],
    bufs: &[(u32, &wgpu::Buffer)],
    groups: [u32; 3],
) -> GResult<()> {
    let res: Vec<(u32, wgpu::BindingResource)> = bufs.iter().map(|&(b, buf)| (b, buf.as_entire_binding())).collect();
    dispatch_res(g, enc, timer, key, params, res, groups)
}

/// [`dispatch`] with any binding resources (textures, samplers).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_res(
    g: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    timer: Option<&mut Timer>,
    key: &PipeKey,
    params: &[u32],
    res: Vec<(u32, wgpu::BindingResource)>,
    groups: [u32; 3],
) -> GResult<()> {
    if groups.contains(&0) {
        return Ok(());
    }
    let lim = g.limits.max_compute_workgroups_per_dimension;
    if groups.iter().any(|&n| n > lim) {
        return Err(GpuError::Unsupported);
    }
    let pipe = g.pipeline(key).map_err(GpuError::Validation)?;
    let ubuf;
    let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(res.len() + 1);
    if !params.is_empty() {
        let mut words = params.to_vec();
        words.resize(params.len().next_multiple_of(4), 0);
        ubuf = upload(g, bytemuck::cast_slice(&words), wgpu::BufferUsages::UNIFORM, "params");
        entries.push(wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() });
    }
    for (b, r) in res {
        entries.push(wgpu::BindGroupEntry { binding: b, resource: r });
    }
    let bg = g.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(key.entry), layout: &pipe.get_bind_group_layout(0), entries: &entries });
    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(key.entry), timestamp_writes: timer.and_then(|t| t.writes(key.entry)) });
    pass.set_pipeline(&pipe);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
    Ok(())
}

/// Error scopes around a batch of GPU calls. Fields drop in declaration
/// order, which must be the reverse of the push order.
pub struct Scopes {
    i: wgpu::ErrorScopeGuard,
    m: wgpu::ErrorScopeGuard,
    v: wgpu::ErrorScopeGuard,
}

impl Scopes {
    pub fn push(g: &Gpu) -> Scopes {
        Scopes {
            v: g.device.push_error_scope(wgpu::ErrorFilter::Validation),
            m: g.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
            i: g.device.push_error_scope(wgpu::ErrorFilter::Internal),
        }
    }

    pub fn pop(self) -> GResult<()> {
        let i = block_on(self.i.pop());
        let m = block_on(self.m.pop());
        let v = block_on(self.v.pop());
        if m.is_some() {
            return Err(GpuError::Vram);
        }
        if let Some(e) = v.or(i) {
            return Err(GpuError::Validation(e.to_string()));
        }
        Ok(())
    }
}

/// Submits and returns the submission index.
pub fn submit(g: &Gpu, enc: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
    g.submitted.fetch_add(1, Ordering::Relaxed);
    g.queue.submit([enc.finish()])
}

/// Spins on non-blocking polls until `done` is set, under the watchdog.
///
/// wgpu-hal's Metal wait sleeps in 1 ms steps, which adds ~1.3 ms to every
/// GPU round trip; polling the completion status ourselves (busy for the
/// first 0.2 ms, then 40 µs naps) brings that down to microseconds.
fn spin_until(g: &Gpu, done: &std::sync::atomic::AtomicBool, what: &str) -> GResult<()> {
    let start = Instant::now();
    let limit = watchdog();
    loop {
        if let Err(e) = g.device.poll(wgpu::PollType::Poll) {
            g.fail(e.to_string());
            return Err(GpuError::Lost(e.to_string()));
        }
        if done.load(Ordering::Acquire) {
            return Ok(());
        }
        if g.is_failed() {
            return Err(GpuError::Lost(g.failure().unwrap_or_default()));
        }
        let t = start.elapsed();
        if t >= limit {
            g.fail(format!("watchdog timeout ({what})"));
            return Err(GpuError::Timeout);
        }
        if t < Duration::from_micros(200) {
            std::hint::spin_loop();
        } else {
            std::thread::sleep(Duration::from_micros(40));
        }
    }
}

/// Waits for everything submitted so far (including `_idx`) under the
/// watchdog.
pub fn wait(g: &Gpu, _idx: wgpu::SubmissionIndex) -> GResult<()> {
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let d = done.clone();
    g.queue.on_submitted_work_done(move || d.store(true, Ordering::Release));
    spin_until(g, &done, "submission")
}

/// Maps a slice and waits for it (the GPU work using it must be submitted).
pub fn map_wait(g: &Gpu, slice: wgpu::BufferSlice, mode: wgpu::MapMode) -> GResult<()> {
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let result = Arc::new(Mutex::new(None));
    let (d, r) = (done.clone(), result.clone());
    slice.map_async(mode, move |res| {
        *r.lock().unwrap() = Some(res);
        d.store(true, Ordering::Release);
    });
    spin_until(g, &done, "map")?;
    match result.lock().unwrap().take() {
        Some(Ok(())) => Ok(()),
        Some(Err(e)) => Err(GpuError::Lost(e.to_string())),
        None => Err(GpuError::Lost("map callback missing".into())),
    }
}

pub fn encoder(g: &Gpu) -> wgpu::CommandEncoder {
    g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None })
}

/// Workgroups to cover `n` items in groups of `size`.
#[inline]
pub fn groups(n: u32, size: u32) -> u32 {
    n.div_ceil(size)
}

/// Drop cached buffers; checked-out buffers retain their reservations.
pub fn trim_idle(g: &Gpu) {
    let mut p = g.pool.lock().unwrap();
    for b in std::mem::take(&mut p.free) { p.live -= b.size; b.buf.destroy(); }
}
