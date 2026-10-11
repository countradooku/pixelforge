//! Images kept on the GPU between operations.
//!
//! An uploaded grid stays resident (LRU, within half the VRAM budget), keyed
//! by the identity of its tiles: tiles are immutable and shared, so the same
//! tile pointers mean the same pixels. The entry holds a clone of the grid,
//! which keeps those tiles alive, so a pointer can't be reused while cached.
//! Filter chains, previews and gather filters that read the same layer don't
//! upload it again; mip levels are built on the GPU on demand.

use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use super::device::Gpu;
use super::exec::{self, Buf, GResult, STORAGE, buffer, dispatch, encoder, groups, submit, wait};
use super::shaders::PipeKey;
use crate::engine::grid::{TS, TileGrid};

/// One level of an image on the GPU, stored over its whole tile area
/// (`cols * 256` x `rows * 256`, edge-tile padding included) so samplers see
/// exactly what the tiles hold.
pub struct Image {
    pub buf: Buf,
    /// Image size.
    pub w: u32,
    pub h: u32,
    /// Tile-area size (the buffer's stride and rows).
    pub tw: u32,
    pub th: u32,
}

struct Entry {
    key: u64,
    /// Keeps the tiles alive (see the module docs).
    _grid: TileGrid,
    levels: Vec<Arc<Image>>,
    bytes: u64,
    used: Instant,
}

#[derive(Default)]
pub struct Cache {
    entries: Vec<Entry>,
}


/// Content identity of a grid on one device (buffers belong to the device
/// that made them: a private context, e.g. in tests, must not see the
/// shared one's entries).
pub fn key(g: &Gpu, grid: &TileGrid) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ ((grid.w as u64) << 32 | grid.h as u64) ^ (g as *const Gpu as usize as u64).rotate_left(17);
    for t in &grid.tiles {
        let p = t.as_ref().map_or(0, |t| Arc::as_ptr(t) as usize as u64);
        h = (h ^ p).wrapping_mul(0x100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

/// Uploads a grid's tile area into a new storage buffer.
pub fn upload(g: &Arc<Gpu>, grid: &TileGrid) -> GResult<Image> {
    let (tw, th) = (grid.cols * TS as u32, grid.rows * TS as u32);
    let n = tw as u64 * th as u64 * 4;
    let mappable = g.caps.mappable_primary;
    let usage = if mappable { wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_WRITE } else { wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC };
    let up = buffer(g, n, usage)?;
    up.map(n, wgpu::MapMode::Write)?;
    {
        let mut view = up.slice(..n).get_mapped_range_mut();
        let mut wo = view.slice(..);
        // SAFETY: see `run::run_filter`.
        let px: &mut [u32] = unsafe { std::slice::from_raw_parts_mut(wo.as_raw_ptr().as_ptr() as *mut u32, n as usize / 4) };
        let strip = 32usize;
        px.par_chunks_mut(tw as usize * strip).enumerate().for_each(|(i, chunk)| {
            let h = chunk.len() / tw as usize;
            grid.read_region(0, (i * strip) as i32, tw as usize, h, chunk);
        });
    }
    up.unmap_now();
    if mappable {
        return Ok(Image { buf: up, w: grid.w, h: grid.h, tw, th });
    }
    let dst = buffer(g, n, STORAGE)?;
    let mut enc = encoder(g);
    enc.copy_buffer_to_buffer(&up, 0, &dst, 0, n);
    let idx = submit(g, enc);
    wait(g, idx)?;
    Ok(Image { buf: dst, w: grid.w, h: grid.h, tw, th })
}

/// Level 0 of `grid` on the GPU, uploading it unless it's resident.
pub fn image(g: &Arc<Gpu>, grid: &TileGrid) -> GResult<Arc<Image>> {
    Ok(levels(g, grid, 0)?.remove(0))
}

/// Levels `0..=max_level` of `grid` (mips built on the GPU as needed).

pub fn levels(g: &Arc<Gpu>, grid: &TileGrid, max_level: u32) -> GResult<Vec<Arc<Image>>> {
    let k = key(g, grid);
    let mut have: Vec<Arc<Image>> = {
        let mut c = g.resident.lock().unwrap();
        match c.entries.iter_mut().find(|e| e.key == k) {
            Some(e) => {
                e.used = Instant::now();
                e.levels.clone()
            }
            None => Vec::new(),
        }
    };
    let fresh = have.is_empty();
    if fresh {
        have.push(Arc::new(upload(g, grid)?));
    }
    let want = max_level.min(crate::engine::mips::mip_count(grid.w, grid.h)) as usize;
    let built = have.len();
    if built <= want {
        let mut enc = encoder(g);
        let scopes = exec::Scopes::push(g);
        for l in built..=want {
            let prev = have[l - 1].clone();
            let (lw, lh) = crate::engine::mips::level_dims(grid.w, grid.h, l as u32);
            let (cols, rows) = (lw.div_ceil(TS as u32).max(1), lh.div_ceil(TS as u32).max(1));
            let (tw, th) = (cols * TS as u32, rows * TS as u32);
            let buf = buffer(g, tw as u64 * th as u64 * 4, STORAGE)?;
            let key = PipeKey { module: "mips", entry: "down2", f16: false, consts: Vec::new() };
            dispatch(g, &mut enc, None, &key, &[prev.tw, prev.th, tw, th], &[(1, &prev.buf), (2, &buf)], [groups(tw, 16), groups(th, 16), 1])?;
            have.push(Arc::new(Image { buf, w: lw, h: lh, tw, th }));
        }
        let idx = submit(g, enc);
        scopes.pop()?;
        wait(g, idx)?;
    }
    if fresh || built <= want {
        let bytes: u64 = have.iter().map(|i| i.buf.size).sum();
        let mut c = g.resident.lock().unwrap();
        c.entries.retain(|e| e.key != k);
        c.entries.push(Entry { key: k, _grid: grid.clone(), levels: have.clone(), bytes, used: Instant::now() });
        let cap = g.budget() / 2;
        while c.entries.iter().map(|e| e.bytes).sum::<u64>() > cap && !c.entries.is_empty() {
            let i = c.entries.iter().enumerate().min_by_key(|(_, e)| e.used).map(|(i, _)| i).unwrap();
            c.entries.remove(i);
        }
    }
    have.truncate(want + 1);
    Ok(have)
}

/// Level 0 of `grid` if it's resident (no upload).
pub fn cached(g: &Gpu, grid: &TileGrid) -> Option<Arc<Image>> {
    let k = key(g, grid);
    let mut c = g.resident.lock().unwrap();
    let e = c.entries.iter_mut().find(|e| e.key == k)?;
    e.used = Instant::now();
    e.levels.first().cloned()
}

/// Registers an image the GPU just produced as the contents of `grid` (the
/// result of a filter), so the next operation on it skips the upload.
pub fn adopt(g: &Gpu, grid: &TileGrid, img: Image) {
    let k = key(g, grid);
    let bytes = img.buf.size;
    let mut c = g.resident.lock().unwrap();
    c.entries.retain(|e| e.key != k);
    c.entries.push(Entry { key: k, _grid: grid.clone(), levels: vec![Arc::new(img)], bytes, used: Instant::now() });
    while c.entries.iter().map(|e| e.bytes).sum::<u64>() > g.budget() / 2 && !c.entries.is_empty() {
        let i = c.entries.iter().enumerate().min_by_key(|(_, e)| e.used).map(|(i, _)| i).unwrap();
        c.entries.swap_remove(i);
    }
}

/// Drops everything (memory pressure, device loss, tests).
pub fn clear(g: &Gpu) {
    g.resident.lock().unwrap().entries.clear();
}

/// Resident bytes and entries, for the Info panel.
pub fn usage(g: &Gpu) -> (u64, usize) {
    let c = g.resident.lock().unwrap();
    (c.entries.iter().map(|e| e.bytes).sum(), c.entries.len())
}

/// Release cached images before declining a GPU allocation. Active operations
/// own their own Arc and remain valid; buffers return to the shared pool.
pub fn trim(g: &Gpu, wanted: u64) {
    let Ok(mut c) = g.resident.try_lock() else { return };
    while !c.entries.is_empty() && crate::engine::memory::global().available() < wanted {
        let i = c.entries.iter().enumerate().min_by_key(|(_, e)| e.used).map(|(i, _)| i).unwrap();
        c.entries.swap_remove(i);
    }
}
