//! A GPU tile pool keyed by tile identity, and per-region tile tables.
//!
//! Tiles are immutable and shared (`Arc`), so a tile's pointer identifies its
//! pixels. The pool keeps uploaded tiles in fixed 256² slots of one storage
//! buffer and remembers which tile each slot holds (keeping the `Arc` alive,
//! so its address can't be reused while cached). Showing a layer after an
//! edit uploads only the tiles that changed; flat tiles are never uploaded
//! (their color lives in the table); tiles shared between layers or levels
//! are uploaded once. Least-recently-used slots are recycled.
//!
//! A [`Table`] describes a tile-aligned span of one grid for one operation:
//! per tile, the slot, or "empty" / "uniform color". Kernels fetch pixels
//! through it (`fetch` in `compose.wgsl`), with `read_region` semantics:
//! outside the grid's tile area reads 0.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use super::device::Gpu;
use super::exec::{Buf, GResult, GpuError, STORAGE, buffer};
use crate::engine::grid::{RectI, TPX, TS_I, Tile, TileGrid};
use crate::engine::selection::{MaskGrid, MaskTile};

pub const EMPTY: u32 = 0xFFFF_FFFF;
pub const UNIFORM: u32 = 0xFFFF_FFFE;

/// What a pool slot holds: a color tile or a mask tile (kept alive).
enum Held {
    Px(Arc<Tile>),
    Mask(Arc<crate::engine::memory::Bytes>),
}

#[derive(Default)]
pub struct Pool {
    buf: Option<Arc<Buf>>,
    slots: usize,
    /// Tile pointer -> slot.
    map: HashMap<usize, u32>,
    /// Per slot: what it holds and when it was last used.
    held: Vec<Option<(Held, u64)>>,
    free: Vec<u32>,
    frame: u64,
    /// Epochs whose GPU work may still read their slots.
    active: Vec<u64>,
}


/// Bytes per slot: one 256² tile of packed pixels (a mask tile uses a
/// quarter of it).
const SLOT_BYTES: u64 = TPX as u64 * 4;

/// The pool buffer, allocated on first use (a sixteenth of the budget,
/// 256 MB to 2 GB).
fn pool_buf(g: &Arc<Gpu>, p: &mut Pool) -> GResult<Arc<Buf>> {
    if let Some(b) = &p.buf {
        return Ok(b.clone());
    }
    let bytes = (g.budget() / 16).clamp(256 << 20, 2 << 30).min(g.max_buffer());
    let slots = (bytes / SLOT_BYTES) as usize;
    let b = Arc::new(buffer(g, slots as u64 * SLOT_BYTES, STORAGE)?);
    p.buf = Some(b.clone());
    p.slots = slots;
    p.held = (0..slots).map(|_| None).collect();
    p.free = (0..slots as u32).rev().collect();
    Ok(b)
}

/// A use epoch (one frame or operation). Slots used by a live epoch are
/// never evicted, so other threads can't recycle them under GPU work that
/// hasn't run yet; drop the guard after waiting for that work.
pub struct Frame(pub u64, Arc<Gpu>);

impl Drop for Frame {
    fn drop(&mut self) {
        let mut p = self.1.tiles.lock().unwrap();
        p.active.retain(|&f| f != self.0);
    }
}

pub fn next_frame(g: &Arc<Gpu>) -> Frame {
    let mut p = g.tiles.lock().unwrap();
    p.frame += 1;
    let f = p.frame;
    p.active.push(f);
    Frame(f, g.clone())
}

fn take_slot(p: &mut Pool, _frame: u64) -> Option<u32> {
    if let Some(s) = p.free.pop() {
        return Some(s);
    }
    // Evict the least recently used slot no live epoch uses.
    let oldest = p.active.iter().copied().min().unwrap_or(u64::MAX);
    let (i, _) = p.held.iter().enumerate().filter_map(|(i, h)| h.as_ref().map(|h| (i, h.1))).filter(|&(_, t)| t < oldest).min_by_key(|&(_, t)| t)?;
    let key = match &p.held[i].as_ref().unwrap().0 {
        Held::Px(t) => Arc::as_ptr(t) as usize,
        Held::Mask(m) => Arc::as_ptr(m) as *const u8 as usize,
    };
    p.map.remove(&key);
    p.held[i] = None;
    Some(i as u32)
}

/// Drops every pooled tile (memory pressure, tests).
pub fn clear(g: &Gpu) {
    let mut p = g.tiles.lock().unwrap();
    let (frame, active) = (p.frame, std::mem::take(&mut p.active));
    *p = Pool { buf: None, slots: 0, map: HashMap::new(), held: Vec::new(), free: Vec::new(), frame, active };
}

/// Pooled tiles and capacity, for the Info panel.
pub fn usage(g: &Gpu) -> (usize, usize) {
    let p = g.tiles.lock().unwrap();
    (p.map.len(), p.slots)
}

/// A tile-aligned span of one grid, as kernels see it.
#[derive(Clone, Debug)]
pub struct Table {
    /// First tile of the span and its size in tiles.
    pub tx0: i32,
    pub ty0: i32,
    pub tw: u32,
    pub th: u32,
    /// Per tile: slot, `EMPTY` or `UNIFORM`, and the uniform color.
    pub entries: Vec<[u32; 2]>,
}

impl Table {
    /// The u32 words a kernel reads: header (tx0, ty0, tw, th) + entries.
    pub fn words(&self) -> Vec<u32> {
        let mut w = Vec::with_capacity(4 + self.entries.len() * 2);
        w.extend_from_slice(&[self.tx0 as u32, self.ty0 as u32, self.tw, self.th]);
        for e in &self.entries {
            w.extend_from_slice(e);
        }
        w
    }
}

/// Tile span of a level-space rect.
pub fn span(r: RectI) -> (i32, i32, i32, i32) {
    (r.x0.div_euclid(TS_I), r.y0.div_euclid(TS_I), (r.x1 - 1).div_euclid(TS_I) + 1, (r.y1 - 1).div_euclid(TS_I) + 1)
}

/// Makes the tiles of `grid` covering `r` resident and returns their table,
/// recording the uploads in `enc`. Tiles outside the grid are empty.
pub fn table(g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, frame: u64, grid: &TileGrid, r: RectI, staging: &mut Vec<Buf>) -> GResult<(Table, Arc<Buf>)> {
    let (tx0, ty0, tx1, ty1) = span(r);
    let (tw, th) = ((tx1 - tx0).max(0) as u32, (ty1 - ty0).max(0) as u32);
    let mut entries = vec![[EMPTY, 0]; (tw * th) as usize];
    let mut uploads: Vec<(u32, Arc<Tile>)> = Vec::new();
    let pool;
    {
        let mut p = g.tiles.lock().unwrap();
        pool = pool_buf(g, &mut p)?;
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                let i = ((ty - ty0) as u32 * tw + (tx - tx0) as u32) as usize;
                let Some(t) = grid.get(tx, ty) else { continue };
                if let Some(c) = t.uniform {
                    entries[i] = [UNIFORM, c];
                    continue;
                }
                let key = Arc::as_ptr(t) as usize;
                let slot = match p.map.get(&key) {
                    Some(&s) => s,
                    None => {
                        let s = take_slot(&mut p, frame).ok_or(GpuError::Vram)?;
                        p.map.insert(key, s);
                        uploads.push((s, t.clone()));
                        s
                    }
                };
                p.held[slot as usize] = Some((Held::Px(t.clone()), frame));
                entries[i] = [slot, 0];
            }
        }
    }
    let pixels: Vec<_> = uploads.iter().map(|(_, t)| t.read()).collect();
    upload_tiles(g, enc, &pool, uploads.iter().zip(&pixels).map(|((s, _), p)| (*s, bytemuck::cast_slice::<u32, u8>(p))).collect(), staging)?;
    Ok((Table { tx0, ty0, tw, th, entries }, pool))
}

/// Like [`table`] for a coverage mask: entries are a slot, `EMPTY` (0) or
/// `UNIFORM` with 255 (fully revealed). Mask slots hold bytes.
pub fn mask_table(g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, frame: u64, m: &MaskGrid, r: RectI, staging: &mut Vec<Buf>) -> GResult<Table> {
    let (tx0, ty0, tx1, ty1) = span(r);
    let (tw, th) = ((tx1 - tx0).max(0) as u32, (ty1 - ty0).max(0) as u32);
    let mut entries = vec![[EMPTY, 0]; (tw * th) as usize];
    let mut uploads: Vec<(u32, Arc<crate::engine::memory::Bytes>)> = Vec::new();
    let pool;
    {
        let mut p = g.tiles.lock().unwrap();
        pool = pool_buf(g, &mut p)?;
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                if tx < 0 || ty < 0 || tx >= m.cols as i32 || ty >= m.rows as i32 {
                    continue;
                }
                let i = ((ty - ty0) as u32 * tw + (tx - tx0) as u32) as usize;
                match &m.tiles[(ty as u32 * m.cols + tx as u32) as usize] {
                    MaskTile::Empty => {}
                    MaskTile::Full => entries[i] = [UNIFORM, 255],
                    MaskTile::Partial(d) => {
                        let key = Arc::as_ptr(d) as *const u8 as usize;
                        let slot = match p.map.get(&key) {
                            Some(&s) => s,
                            None => {
                                let s = take_slot(&mut p, frame).ok_or(GpuError::Vram)?;
                                p.map.insert(key, s);
                                uploads.push((s, d.clone()));
                                s
                            }
                        };
                        p.held[slot as usize] = Some((Held::Mask(d.clone()), frame));
                        entries[i] = [slot, 0];
                    }
                }
            }
        }
    }
    upload_tiles(g, enc, &pool, uploads.iter().map(|(s, d)| (*s, &d[..])).collect(), staging)?;
    Ok(Table { tx0, ty0, tw, th, entries })
}

/// Copies tile bytes into their slots through one mapped staging buffer
/// (filled in parallel), kept alive in `staging` until the work is waited.
fn upload_tiles(g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, pool: &Buf, tiles: Vec<(u32, &[u8])>, staging: &mut Vec<Buf>) -> GResult<()> {
    if tiles.is_empty() {
        return Ok(());
    }
    let offs: Vec<u64> = tiles
        .iter()
        .scan(0u64, |o, (_, d)| {
            let at = *o;
            *o += (d.len() as u64).next_multiple_of(256);
            Some(at)
        })
        .collect();
    let total = offs.last().unwrap() + (tiles.last().unwrap().1.len() as u64).next_multiple_of(256);
    let up = buffer(g, total, wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC)?;
    up.map(total, wgpu::MapMode::Write)?;
    {
        let mut view = up.slice(..total).get_mapped_range_mut();
        let mut wo = view.slice(..);
        let base = wo.as_raw_ptr().as_ptr() as *mut u8 as usize;
        tiles.par_iter().zip(&offs).for_each(|((_, d), &o)| {
            // SAFETY: each tile writes its own disjoint, in-bounds range of
            // the live mapping.
            unsafe { std::ptr::copy_nonoverlapping(d.as_ptr(), (base + o as usize) as *mut u8, d.len()) };
        });
    }
    up.unmap_now();
    for ((slot, d), &o) in tiles.iter().zip(&offs) {
        enc.copy_buffer_to_buffer(&up, o, pool, *slot as u64 * SLOT_BYTES, d.len() as u64);
    }
    staging.push(up);
    Ok(())
}


/// The pool can be replaced only when no submission still references its slots.
pub fn trim_idle(g: &Gpu) {
    let mut p = g.tiles.lock().unwrap();
    if p.active.is_empty() { let frame = p.frame; *p = Pool { frame, ..Default::default() }; }
}
