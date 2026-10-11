//! One budget for pixel storage and GPU buffers. Shared tiles are charged once.
//! Readers pin a tile with a guard; reclamation uses try_write and never evicts
//! pixels being read or edited. The worker compresses cold tiles and spills LRU
//! tiles under pressure. Scratch errors preserve the in-memory source.
use super::scratch::{Record, Scratch};
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};
use std::time::{Duration, Instant};

static GLOBAL: OnceLock<Arc<MemoryBudget>> = OnceLock::new();
static CLOCK: AtomicU64 = AtomicU64::new(0);
static EPOCH: OnceLock<Instant> = OnceLock::new();
fn now() -> u64 {
    CLOCK.load(Relaxed)
}
fn tick() -> u64 {
    let t = EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64;
    CLOCK.store(t, Relaxed);
    t
}

pub fn global() -> &'static Arc<MemoryBudget> {
    GLOBAL.get_or_init(|| {
        let p = super::prefs::Prefs::load();
        let b = MemoryBudget::new(
            ram_bytes() * p.memory_percent as u64 / 100,
            p.scratch_dir.clone(),
            p.cold_seconds,
        );
        if let Some(n) = std::env::var("PF_MEMORY_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|n| *n > 0)
        {
            b.limit.store(n.saturating_mul(1 << 20), Relaxed);
        }
        let weak = Arc::downgrade(&b);
        std::thread::Builder::new()
            .name("tile-memory".into())
            .spawn(move || {
                while let Some(b) = weak.upgrade() {
                    std::thread::sleep(Duration::from_secs(1));
                    if b.available() < b.limit() / 10 {
                        super::gpu::trim_memory();
                    }
                    b.maintain();
                }
            })
            .expect("memory worker");
        b
    })
}

pub fn ram_bytes() -> u64 {
    static RAM: OnceLock<u64> = OnceLock::new();
    *RAM.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                fn sysctlbyname(
                    name: *const std::ffi::c_char,
                    old: *mut std::ffi::c_void,
                    len: *mut usize,
                    new: *mut std::ffi::c_void,
                    newlen: usize,
                ) -> i32;
            }
            let mut value = 0u64;
            let mut len = std::mem::size_of::<u64>();
            // SAFETY: a NUL-terminated constant and a correctly sized output buffer.
            if unsafe {
                sysctlbyname(
                    c"hw.memsize".as_ptr(),
                    (&mut value as *mut u64).cast(),
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                )
            } == 0
            {
                return value;
            }
        }
        #[cfg(target_os = "linux")]
        if let Ok(s) = std::fs::read_to_string("/proc/meminfo") {
            if let Some(n) = s.lines().find_map(|l| {
                l.strip_prefix("MemTotal:")
                    .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
            }) {
                return n * 1024;
            }
        }
        8 << 30
    })
}

#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Stats {
    pub used: u64,
    pub limit: u64,
    pub compressed: u64,
    pub disk: u64,
    pub gpu: u64,
    pub peak: u64,
    pub fault: Option<String>,
}
impl Stats {
    pub fn summary(&self) -> String {
        format!(
            "{:.1} / {:.0} MB · {:.1} MB compressed · {:.1} MB scratch · {:.1} MB GPU",
            self.used as f64 / 1048576.,
            self.limit as f64 / 1048576.,
            self.compressed as f64 / 1048576.,
            self.disk as f64 / 1048576.,
            self.gpu as f64 / 1048576.
        )
    }
}

pub struct MemoryBudget {
    limit: AtomicU64,
    used: AtomicU64,
    compressed: AtomicU64,
    disk: AtomicU64,
    gpu: AtomicU64,
    peak: AtomicU64,
    tiles: [Mutex<Vec<Weak<PixelStore>>>; 32],
    shared: Mutex<Vec<SharedAllocation>>,
    maintenance: Mutex<()>,
    scratch: Mutex<Option<Arc<Scratch>>>,
    directory: Mutex<PathBuf>,
    cold_ms: AtomicU64,
    fault: Mutex<Option<String>>,
}
impl MemoryBudget {
    pub fn new(limit: u64, directory: PathBuf, cold_seconds: u64) -> Arc<Self> {
        Arc::new(Self {
            limit: AtomicU64::new(limit),
            used: AtomicU64::new(0),
            compressed: AtomicU64::new(0),
            disk: AtomicU64::new(0),
            gpu: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            tiles: std::array::from_fn(|_| Mutex::new(Vec::new())),
            shared: Mutex::new(Vec::new()),
            maintenance: Mutex::new(()),
            scratch: Mutex::new(None),
            directory: Mutex::new(directory),
            cold_ms: AtomicU64::new(cold_seconds.saturating_mul(1000)),
            fault: Mutex::new(None),
        })
    }
    pub fn configure(&self, percent: u8, directory: PathBuf, cold_seconds: u64) {
        self.limit
            .store(ram_bytes() * percent.clamp(10, 90) as u64 / 100, Relaxed);
        *self.directory.lock().unwrap() = directory;
        // Existing records keep their file alive; future spills use the new path.
        self.scratch.lock().unwrap().take();
        self.cold_ms
            .store(cold_seconds.saturating_mul(1000), Relaxed);
    }
    pub fn stats(&self) -> Stats {
        Stats {
            used: self.used.load(Relaxed),
            limit: self.limit.load(Relaxed),
            compressed: self.compressed.load(Relaxed),
            disk: self.disk.load(Relaxed),
            gpu: self.gpu.load(Relaxed),
            peak: self.peak.load(Relaxed),
            fault: self.fault.lock().unwrap().clone(),
        }
    }
    pub fn limit(&self) -> u64 {
        self.limit.load(Relaxed)
    }
    pub fn available(&self) -> u64 {
        self.limit().saturating_sub(self.used.load(Relaxed))
    }
    fn charge(&self, bytes: u64) {
        let used = self.used.fetch_add(bytes, Relaxed) + bytes;
        if used > self.peak.load(Relaxed) {
            self.peak.fetch_max(used, Relaxed);
        }
    }
    pub fn reserve_gpu(self: &Arc<Self>, bytes: u64) -> Option<Reservation> {
        self.used
            .try_update(Relaxed, Relaxed, |n| {
                n.checked_add(bytes).filter(|v| *v <= self.limit())
            })
            .ok()?;
        self.gpu.fetch_add(bytes, Relaxed);
        self.peak.fetch_max(self.used.load(Relaxed), Relaxed);
        Some(Reservation {
            owner: self.clone(),
            bytes,
            gpu: true,
        })
    }
    pub fn track_cpu(self: &Arc<Self>, bytes: u64) -> Reservation {
        self.charge(bytes);
        Reservation {
            owner: self.clone(),
            bytes,
            gpu: false,
        }
    }
    fn error(&self, e: impl std::fmt::Display) {
        *self.fault.lock().unwrap() = Some(e.to_string());
    }
    fn scratch(&self) -> std::io::Result<Arc<Scratch>> {
        let mut slot = self.scratch.lock().unwrap();
        if let Some(s) = &*slot {
            return Ok(s.clone());
        }
        let s = Arc::new(Scratch::new(&self.directory.lock().unwrap())?);
        *slot = Some(s.clone());
        Ok(s)
    }
    /// Track externally owned immutable previews without retaining them. One
    /// registration follows all clones; expired charges are released by the worker.
    pub fn track_shared<T: Send + Sync + 'static>(&self, value: Arc<T>, bytes: u64) -> Arc<T> {
        let weak = Arc::downgrade(&value);
        self.charge(bytes);
        self.shared.lock().unwrap().push(SharedAllocation {
            alive: Box::new(move || weak.strong_count() != 0),
            bytes,
        });
        value
    }
    pub fn maintain(&self) {
        self.shared.lock().unwrap().retain(|a| {
            if (a.alive)() {
                true
            } else {
                self.used.fetch_sub(a.bytes, Relaxed);
                false
            }
        });
        let _exclusive = self.maintenance.lock().unwrap();
        let mut tiles = Vec::new();
        for shard in &self.tiles {
            let mut registry = shard.lock().unwrap();
            registry.retain(|w| w.strong_count() > 0);
            tiles.extend(
                registry
                    .iter()
                    .filter_map(Weak::upgrade)
                    .map(|t| (t.used.load(Relaxed), t)),
            );
        }
        // Readers update recency concurrently. Snapshot each key exactly once:
        // querying live atomics in the comparator violates sort's total order.
        tiles.sort_unstable_by_key(|(last_used, _)| *last_used);
        let time = tick();
        let pressure = self.used.load(Relaxed) > self.limit();
        let target = self.limit() * 9 / 10;
        for (_, t) in &tiles {
            if pressure && self.used.load(Relaxed) <= target {
                break;
            }
            if pressure
                || t.cold.load(Relaxed)
                || time.saturating_sub(t.used.load(Relaxed)) >= self.cold_ms.load(Relaxed)
            {
                t.reclaim(pressure, target);
            }
        }
    }
}
struct SharedAllocation {
    alive: Box<dyn Fn() -> bool + Send + Sync>,
    bytes: u64,
}
/// Charged until the underlying GPU allocation is destroyed, including cached buffers.
pub struct Reservation {
    owner: Arc<MemoryBudget>,
    bytes: u64,
    gpu: bool,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.owner.used.fetch_sub(self.bytes, Relaxed);
        if self.gpu {
            self.owner.gpu.fetch_sub(self.bytes, Relaxed);
        }
    }
}

enum Contents {
    Hot(Box<[u32]>),
    Packed(Vec<u8>),
    Disk(Record),
}
impl Contents {
    fn bytes(&self) -> (u64, u64, u64) {
        match self {
            Self::Hot(p) => ((p.len() * 4) as u64, 0, 0),
            Self::Packed(p) => (p.len() as u64, p.len() as u64, 0),
            Self::Disk(r) => (0, 0, r.bytes()),
        }
    }
}

pub struct PixelStore {
    state: RwLock<Contents>,
    owner: Arc<MemoryBudget>,
    used: AtomicU64,
    cold: AtomicBool,
    uniform: AtomicBool,
}
impl PixelStore {
    pub fn new(px: Box<[u32]>, uniform: bool) -> Arc<Self> {
        Self::with_budget(px, uniform, global().clone())
    }
    pub fn with_budget(px: Box<[u32]>, uniform: bool, owner: Arc<MemoryBudget>) -> Arc<Self> {
        owner.charge((px.len() * 4) as u64);
        let s = Arc::new(Self {
            state: RwLock::new(Contents::Hot(px)),
            owner: owner.clone(),
            used: AtomicU64::new(now()),
            cold: AtomicBool::new(false),
            uniform: AtomicBool::new(uniform),
        });
        // Independent allocator pages / workers need not serialize registration.
        let address = Arc::as_ptr(&s) as usize;
        let shard = ((address >> 6) ^ (address >> 15)) & (owner.tiles.len() - 1);
        owner.tiles[shard].lock().unwrap().push(Arc::downgrade(&s));
        if owner.used.load(Relaxed) > owner.limit() {
            owner.maintain();
        }
        s
    }
    pub fn duplicate(&self) -> Arc<Self> {
        let px = self.read().to_vec().into_boxed_slice();
        Self::with_budget(px, self.uniform.load(Relaxed), self.owner.clone())
    }
    fn replace(&self, state: &mut Contents, next: Contents) {
        let (old, packed, disk) = state.bytes();
        let (new, np, nd) = next.bytes();
        self.owner.charge(new);
        self.owner.used.fetch_sub(old, Relaxed);
        self.owner.compressed.fetch_add(np, Relaxed);
        self.owner.compressed.fetch_sub(packed, Relaxed);
        self.owner.disk.fetch_add(nd, Relaxed);
        self.owner.disk.fetch_sub(disk, Relaxed);
        *state = next;
    }
    fn materialize(&self, state: &mut Contents) -> std::io::Result<()> {
        let data = match state {
            Contents::Hot(_) => return Ok(()),
            Contents::Packed(b) => decode(b)?,
            Contents::Disk(r) => decode(&r.read()?)?,
        };
        self.replace(state, Contents::Hot(data));
        Ok(())
    }
    #[inline]
    fn touch(&self) {
        let time = now();
        // Hot reads usually share the same coarse timestamp. Avoid writing a
        // cache line that every display worker is reading on every access.
        if self.used.load(Relaxed) != time {
            self.used.store(time, Relaxed);
        }
        if self.cold.load(Relaxed) {
            self.cold.store(false, Relaxed);
        }
    }
    pub fn try_read(&self) -> std::io::Result<Pixels<'_>> {
        self.touch();
        let mut loaded = false;
        loop {
            let state = self.state.read().unwrap();
            if matches!(*state, Contents::Hot(_)) {
                if loaded && self.owner.used.load(Relaxed) > self.owner.limit() {
                    self.owner.maintain();
                }
                return Ok(Pixels(state));
            }
            drop(state);
            let mut state = self.state.write().unwrap();
            if let Err(e) = self.materialize(&mut state) {
                self.owner.error(&e);
                return Err(e);
            }
            loaded = true;
        }
    }
    pub fn read(&self) -> Pixels<'_> {
        self.try_read()
            .expect("scratch tile could not be recovered; refusing to substitute image data")
    }
    pub fn try_write(&self) -> std::io::Result<PixelsMut<'_>> {
        self.touch();
        let mut state = self.state.write().unwrap();
        if let Err(e) = self.materialize(&mut state) {
            self.owner.error(&e);
            return Err(e);
        }
        self.uniform.store(false, Relaxed);
        if self.owner.used.load(Relaxed) > self.owner.limit() {
            self.owner.maintain();
        }
        Ok(PixelsMut(state))
    }
    pub fn write(&self) -> PixelsMut<'_> {
        // Release the lock before an infallible caller aborts. A failed restore
        // must not poison the tile and cause a second panic when it is dropped.
        self.try_write().expect("scratch tile could not be recovered; edit aborted")
    }
    pub fn mark_cold(&self) {
        self.cold.store(true, Relaxed);
    }
    #[cfg(test)]
    pub fn is_cold(&self) -> bool {
        !matches!(*self.state.read().unwrap(), Contents::Hot(_))
    }
    pub fn resident_bytes(&self) -> usize {
        self.state.read().unwrap().bytes().0 as usize
    }
    fn reclaim(&self, pressure: bool, target: u64) {
        if self.uniform.load(Relaxed) {
            return;
        }
        let Ok(mut state) = self.state.try_write() else {
            return;
        };
        if let Contents::Hot(px) = &*state {
            let packed = lz4_flex::compress(bytemuck::cast_slice(px));
            if packed.len() < px.len() * 4 || pressure {
                self.replace(&mut state, Contents::Packed(packed));
            }
        }
        if pressure && self.owner.used.load(Relaxed) > target {
            if let Contents::Packed(b) = &*state {
                match self.owner.scratch().and_then(|s| s.write(b)) {
                    Ok(record) => self.replace(&mut state, Contents::Disk(record)),
                    Err(e) => self.owner.error(e),
                }
            }
        }
    }
}
impl Drop for PixelStore {
    fn drop(&mut self) {
        let (used, compressed, disk) = self.state.get_mut().unwrap().bytes();
        self.owner.used.fetch_sub(used, Relaxed);
        self.owner.compressed.fetch_sub(compressed, Relaxed);
        self.owner.disk.fetch_sub(disk, Relaxed);
    }
}
fn decode(b: &[u8]) -> std::io::Result<Box<[u32]>> {
    let mut px = vec![0u32; super::grid::TPX].into_boxed_slice();
    let n = lz4_flex::decompress_into(b, bytemuck::cast_slice_mut(&mut px))
        .map_err(std::io::Error::other)?;
    if n != px.len() * 4 {
        return Err(std::io::Error::other("invalid scratch tile length"));
    }
    Ok(px)
}
pub struct Pixels<'a>(RwLockReadGuard<'a, Contents>);
impl Deref for Pixels<'_> {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        match &*self.0 {
            Contents::Hot(p) => p,
            _ => unreachable!(),
        }
    }
}
pub struct PixelsMut<'a>(RwLockWriteGuard<'a, Contents>);
impl Deref for PixelsMut<'_> {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        match &*self.0 {
            Contents::Hot(p) => p,
            _ => unreachable!(),
        }
    }
}
impl DerefMut for PixelsMut<'_> {
    fn deref_mut(&mut self) -> &mut [u32] {
        match &mut *self.0 {
            Contents::Hot(p) => p,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::grid::TPX;
    use super::*;
    fn budget(n: u64) -> Arc<MemoryBudget> {
        MemoryBudget::new(n, std::env::temp_dir(), 30)
    }
    fn pixels(seed: u32) -> Box<[u32]> {
        let mut v = seed;
        (0..TPX)
            .map(|_| {
                v ^= v << 13;
                v ^= v >> 17;
                v ^= v << 5;
                v
            })
            .collect()
    }
    #[test]
    fn compression_and_scratch_round_trip_all_pixel_bits() {
        for seed in [0, 1, 17, u32::MAX] {
            let b = budget(32 << 10);
            let expected = pixels(seed);
            let tile = PixelStore::with_budget(expected.clone(), false, b.clone());
            assert!(b.stats().used <= b.limit());
            assert_eq!(&*tile.try_read().unwrap(), &*expected);
            tile.mark_cold();
            b.maintain();
            assert!(b.stats().used <= b.limit());
            assert_eq!(&*tile.try_read().unwrap(), &*expected);
            drop(tile);
            assert_eq!(
                (b.stats().used, b.stats().disk, b.stats().compressed),
                (0, 0, 0)
            );
        }
    }
    #[test]
    fn pinned_tiles_are_never_evicted_and_gpu_shares_the_limit() {
        let b = budget(2 << 20);
        let tile = PixelStore::with_budget(pixels(5), false, b.clone());
        let pinned = tile.read();
        b.limit.store(300 << 10, Relaxed);
        let gpu = b.reserve_gpu(32 << 10).unwrap();
        assert!(b.reserve_gpu(32 << 10).is_none());
        tile.mark_cold();
        b.maintain();
        assert_eq!(pinned[9], pixels(5)[9]);
        assert_eq!(b.stats().compressed, 0);
        drop(pinned);
        drop(gpu);
        b.maintain();
        assert_eq!(b.stats().gpu, 0);
    }
    #[test]
    fn full_or_unwritable_scratch_keeps_pixels_in_memory() {
        let path = std::env::temp_dir().join(format!("pf-no-directory-{}", std::process::id()));
        std::fs::write(&path, b"not a directory").unwrap();
        let b = MemoryBudget::new(1024, path.clone(), 0);
        let expected = pixels(9);
        let tile = PixelStore::with_budget(expected.clone(), false, b.clone());
        assert!(b.stats().fault.is_some());
        assert_eq!(&*tile.try_read().unwrap(), &*expected);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn unrecoverable_data_reports_an_error_without_poisoning_storage() {
        let b = budget(2 << 20);
        let tile = PixelStore::with_budget(pixels(9), false, b.clone());
        tile.replace(&mut tile.state.write().unwrap(), Contents::Packed(vec![0, 0]));
        assert!(tile.try_write().is_err());
        assert!(tile.try_read().is_err());
        assert!(tile.state.read().is_ok());
        assert!(b.stats().fault.is_some());
        drop(tile);
        assert_eq!(b.stats().used, 0);
    }

    #[test]
    fn idle_and_hidden_tiles_compress_without_pressure() {
        let b = budget(4 << 20);
        let t =
            PixelStore::with_budget((0..TPX).map(|i| (i % 4) as u32).collect(), false, b.clone());
        b.maintain();
        assert!(!t.is_cold());
        t.mark_cold();
        b.maintain();
        assert!(t.is_cold());
        assert!(b.stats().compressed > 0);
        let _ = t.read();
        b.cold_ms.store(0, Relaxed);
        b.maintain();
        assert!(t.is_cold());
    }
    #[test]
    fn concurrent_reads_reclaim_and_edit_are_lossless() {
        let b = budget(64 << 10);
        let expected = pixels(15);
        let t = PixelStore::with_budget(expected.clone(), false, b.clone());
        std::thread::scope(|s| {
            for _ in 0..4 {
                let t = &t;
                let expected = &expected;
                s.spawn(move || {
                    for _ in 0..20 {
                        assert_eq!(&*t.read(), &**expected);
                    }
                });
            }
            s.spawn(|| {
                for _ in 0..20 {
                    b.maintain();
                }
            });
        });
        t.write()[3] = 123;
        b.maintain();
        assert_eq!(t.read()[3], 123);
    }
    #[test]
    fn concurrent_recency_changes_cannot_invalidate_lru_sort_order() {
        let b = budget(64 << 20);
        let tiles: Vec<_> = (0..128)
            .map(|i| PixelStore::with_budget(vec![i; TPX].into_boxed_slice(), true, b.clone()))
            .collect();
        std::thread::scope(|s| {
            s.spawn(|| {
                for n in 0u64..2000 {
                    for (i, t) in tiles.iter().enumerate() {
                        t.used.store((i as u64 * 17) ^ n, Relaxed);
                    }
                }
            });
            for _ in 0..200 {
                b.maintain();
            }
        });
        assert_eq!(b.stats().used, 128 * TPX as u64 * 4);
    }
    #[test]
    fn shared_previews_are_charged_once_until_the_last_owner_drops() {
        let b = budget(1024);
        let a = b.track_shared(Arc::new(vec![1u8; 100]), 100);
        let clone = a.clone();
        assert_eq!(b.stats().used, 100);
        drop(a);
        b.maintain();
        assert_eq!(b.stats().used, 100);
        drop(clone);
        b.maintain();
        assert_eq!(b.stats().used, 0);
    }
    #[test]
    fn least_recently_used_tiles_spill_first() {
        let b = budget(4 << 20);
        let old = PixelStore::with_budget(pixels(1), false, b.clone());
        let recent = PixelStore::with_budget(pixels(2), false, b.clone());
        old.used.store(1, Relaxed);
        recent.used.store(1000, Relaxed);
        b.limit.store(300 << 10, Relaxed);
        b.maintain();
        assert!(matches!(*old.state.read().unwrap(), Contents::Disk(_)));
        assert!(matches!(*recent.state.read().unwrap(), Contents::Hot(_)));
    }
}

/// Shared byte pixels (coverage masks). Charge the allocation, not each Arc.
/// These small masks stay resident; color tiles provide reclaimable capacity.
pub struct Bytes {
    data: Box<[u8]>,
    _memory: Reservation,
}
impl Bytes {
    pub fn new(data: impl Into<Box<[u8]>>) -> Self {
        let data = data.into();
        let memory = global().track_cpu(data.len() as u64);
        if global().available() == 0 {
            global().maintain();
        }
        Self {
            data,
            _memory: memory,
        }
    }
}
impl Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.data
    }
}
impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}
