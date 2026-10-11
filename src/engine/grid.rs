//! Sparse, copy-on-write tile grids.
//!
//! Images are split into 256x256 tiles held behind `Arc`. A missing tile is
//! fully transparent, and a tile flagged `uniform` is one flat color, so a
//! 300 MP white canvas costs one tile of memory. Cloning a grid only copies
//! pointers, which is what makes undo snapshots, background jobs and stroke
//! bases nearly free: writers call [`tile_mut`], which clones a tile only when
//! someone else still holds it.

use std::sync::Arc;

pub const TS: usize = 256;
pub const TS_I: i32 = 256;
pub const TSHIFT: u32 = 8;
pub const TPX: usize = TS * TS;

pub struct Tile {
    data: Arc<super::memory::PixelStore>,
    /// `Some(c)` when every pixel equals `c`. Cleared by [`tile_mut`].
    pub uniform: Option<u32>,
}
impl Clone for Tile {
    fn clone(&self) -> Self { Self { data: self.data.duplicate(), uniform: self.uniform } }
}
impl Tile {
    pub fn solid(c: u32) -> Self { Self { data: super::memory::PixelStore::new(vec![c; TPX].into_boxed_slice(), true), uniform: Some(c) } }
    pub fn from_px(px: Box<[u32]>) -> Self {
        assert_eq!(px.len(), TPX);
        let uniform = px.iter().all(|&p| p == px[0]).then_some(px[0]);
        Self { data: super::memory::PixelStore::new(px, uniform.is_some()), uniform }
    }
    pub fn from_raw(px: Box<[u32]>) -> Self { Self { data: super::memory::PixelStore::new(px, false), uniform: None } }
    #[inline] pub fn read(&self) -> super::memory::Pixels<'_> { self.data.read() }
    #[inline] pub fn write(&mut self) -> super::memory::PixelsMut<'_> { self.uniform = None; self.data.write() }
    pub fn mark_cold(&self) { if self.uniform.is_none() { self.data.mark_cold(); } }
    #[cfg(test)]
    pub fn is_cold(&self) -> bool { self.data.is_cold() }
    #[cfg(test)]
    pub fn detect_uniform(&mut self) { self.uniform = { let px = self.read(); px.iter().all(|&p| p == px[0]).then_some(px[0]) }; }
    pub fn resident_bytes(&self) -> usize { self.data.resident_bytes() }
}

pub type TileRef = Option<Arc<Tile>>;

/// Wraps a freshly computed tile, dropping it entirely when transparent.
pub fn finish_tile(px: Box<[u32]>) -> TileRef {
    let t = Tile::from_px(px);
    if t.uniform == Some(0) { None } else { Some(Arc::new(t)) }
}

/// Mutable access to a tile slot, allocating a transparent tile or cloning a
/// shared one as needed.
#[inline]
pub fn tile_mut(slot: &mut TileRef) -> &mut Tile {
    let arc = slot.get_or_insert_with(|| Arc::new(Tile::solid(0)));
    let t = Arc::make_mut(arc);
    t.uniform = None;
    t
}

pub fn same_tile(a: &TileRef, b: &TileRef) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

/// Half-open integer rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RectI {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl RectI {
    pub const EMPTY: RectI = RectI { x0: 0, y0: 0, x1: 0, y1: 0 };

    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        RectI { x0, y0, x1, y1 }
    }

    pub const fn wh(x: i32, y: i32, w: i32, h: i32) -> Self {
        RectI { x0: x, y0: y, x1: x + w, y1: y + h }
    }

    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }

    pub fn w(&self) -> i32 {
        self.x1 - self.x0
    }

    pub fn h(&self) -> i32 {
        self.y1 - self.y0
    }

    pub fn intersect(&self, o: &RectI) -> RectI {
        RectI {
            x0: self.x0.max(o.x0),
            y0: self.y0.max(o.y0),
            x1: self.x1.min(o.x1),
            y1: self.y1.min(o.y1),
        }
    }

    pub fn union(&self, o: &RectI) -> RectI {
        if self.is_empty() {
            return *o;
        }
        if o.is_empty() {
            return *self;
        }
        RectI {
            x0: self.x0.min(o.x0),
            y0: self.y0.min(o.y0),
            x1: self.x1.max(o.x1),
            y1: self.y1.max(o.y1),
        }
    }

    pub fn intersects(&self, o: &RectI) -> bool {
        !self.intersect(o).is_empty()
    }

    pub fn contains_rect(&self, o: &RectI) -> bool {
        o.x0 >= self.x0 && o.y0 >= self.y0 && o.x1 <= self.x1 && o.y1 <= self.y1
    }

    pub fn translate(&self, dx: i32, dy: i32) -> RectI {
        RectI { x0: self.x0 + dx, y0: self.y0 + dy, x1: self.x1 + dx, y1: self.y1 + dy }
    }

    pub fn inflate(&self, d: i32) -> RectI {
        RectI { x0: self.x0 - d, y0: self.y0 - d, x1: self.x1 + d, y1: self.y1 + d }
    }

    /// Range of tile indices `[t0, t1)` covering this rect, clamped to a grid.
    pub fn tile_span(&self, cols: u32, rows: u32) -> (i32, i32, i32, i32) {
        if self.is_empty() {
            return (0, 0, 0, 0);
        }
        let tx0 = (self.x0 >> TSHIFT).max(0);
        let ty0 = (self.y0 >> TSHIFT).max(0);
        let tx1 = (((self.x1 - 1) >> TSHIFT) + 1).min(cols as i32);
        let ty1 = (((self.y1 - 1) >> TSHIFT) + 1).min(rows as i32);
        (tx0, ty0, tx1.max(tx0), ty1.max(ty0))
    }
}

#[derive(Clone)]
pub struct TileGrid {
    pub w: u32,
    pub h: u32,
    pub cols: u32,
    pub rows: u32,
    pub tiles: Vec<TileRef>,
}

impl TileGrid {
    pub fn new(w: u32, h: u32) -> Self {
        let cols = w.div_ceil(TS as u32).max(1);
        let rows = h.div_ceil(TS as u32).max(1);
        TileGrid { w, h, cols, rows, tiles: vec![None; (cols * rows) as usize] }
    }

    /// A grid where every tile shares one uniform tile.
    pub fn filled(w: u32, h: u32, c: u32) -> Self {
        let mut g = Self::new(w, h);
        if c != 0 {
            let t = Arc::new(Tile::solid(c));
            for slot in &mut g.tiles {
                *slot = Some(t.clone());
            }
        }
        g
    }

    pub fn bounds(&self) -> RectI {
        RectI::new(0, 0, self.w as i32, self.h as i32)
    }

    #[inline(always)]
    pub fn idx(&self, tx: i32, ty: i32) -> usize {
        (ty as u32 * self.cols + tx as u32) as usize
    }

    #[inline]
    pub fn get(&self, tx: i32, ty: i32) -> Option<&Arc<Tile>> {
        if tx < 0 || ty < 0 || tx >= self.cols as i32 || ty >= self.rows as i32 {
            return None;
        }
        self.tiles[self.idx(tx, ty)].as_ref()
    }

    pub fn tile_rect(tx: i32, ty: i32) -> RectI {
        RectI::wh(tx * TS_I, ty * TS_I, TS_I, TS_I)
    }

    #[inline]
    pub fn pixel(&self, x: i32, y: i32) -> u32 {
        match self.get(x >> TSHIFT, y >> TSHIFT) {
            Some(t) => t.uniform.unwrap_or_else(|| t.read()[(((y & 255) as usize) << TSHIFT) | (x & 255) as usize]),
            None => 0,
        }
    }

    /// Whether any tile overlapping the region holds pixels.
    pub fn region_has_content(&self, r: RectI) -> bool {
        let (tx0, ty0, tx1, ty1) = r.tile_span(self.cols, self.rows);
        (ty0..ty1).any(|ty| (tx0..tx1).any(|tx| self.tiles[self.idx(tx, ty)].is_some()))
    }

    /// If every tile overlapping `r` is the same uniform color (or all are
    /// empty), returns that tile so callers can share it instead of copying.
    pub fn region_uniform(&self, r: RectI) -> Option<TileRef> {
        let clipped = r.intersect(&self.bounds());
        if clipped != r {
            // Partly outside the image: only all-empty can be shared.
            let (tx0, ty0, tx1, ty1) = clipped.tile_span(self.cols, self.rows);
            let any = (ty0..ty1).any(|ty| (tx0..tx1).any(|tx| self.tiles[self.idx(tx, ty)].is_some()));
            return if any { None } else { Some(None) };
        }
        let (tx0, ty0, tx1, ty1) = r.tile_span(self.cols, self.rows);
        let first = self.tiles[self.idx(tx0, ty0)].clone();
        let color = match &first {
            None => None,
            Some(t) => Some(t.uniform?),
        };
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                let t = &self.tiles[self.idx(tx, ty)];
                let c = match t {
                    None => None,
                    Some(t) => Some(t.uniform?),
                };
                if c != color {
                    return None;
                }
            }
        }
        Some(first)
    }

    /// Copies `w x h` pixels starting at `(x0, y0)` into `dst` (stride `w`).
    /// Anything outside the grid or in a missing tile reads as transparent.
    pub fn read_region(&self, x0: i32, y0: i32, w: usize, h: usize, dst: &mut [u32]) {
        debug_assert!(dst.len() >= w * h);
        let r = RectI::wh(x0, y0, w as i32, h as i32);
        let (tx0, ty0, tx1, ty1) = r.tile_span(self.cols, self.rows);
        let covered = r.intersect(&RectI::new(0, 0, self.cols as i32 * TS_I, self.rows as i32 * TS_I));
        if covered != r {
            dst[..w * h].fill(0);
        }
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                let tr = Self::tile_rect(tx, ty).intersect(&r);
                if tr.is_empty() {
                    continue;
                }
                let tw = tr.w() as usize;
                match &self.tiles[self.idx(tx, ty)] {
                    None => {
                        for y in tr.y0..tr.y1 {
                            let o = (y - y0) as usize * w + (tr.x0 - x0) as usize;
                            dst[o..o + tw].fill(0);
                        }
                    }
                    Some(t) => {
                        if let Some(c) = t.uniform {
                            for y in tr.y0..tr.y1 {
                                let o = (y - y0) as usize * w + (tr.x0 - x0) as usize;
                                dst[o..o + tw].fill(c);
                            }
                        } else {
                            let sx = (tr.x0 - tx * TS_I) as usize;
                            let pixels = t.read();
                            for y in tr.y0..tr.y1 {
                                let sy = (y - ty * TS_I) as usize;
                                let o = (y - y0) as usize * w + (tr.x0 - x0) as usize;
                                dst[o..o + tw].copy_from_slice(&pixels[sy * TS + sx..sy * TS + sx + tw]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Like [`read_region`], but pixels outside the image replicate the
    /// nearest edge pixel (what blur-style filters expect).
    pub fn read_region_clamped(&self, x0: i32, y0: i32, w: usize, h: usize, dst: &mut [u32]) {
        if w == 0 || h == 0 { return; }
        self.read_region(x0, y0, w, h, dst);
        let img = self.bounds();
        let r = RectI::wh(x0, y0, w as i32, h as i32);
        let inner = r.intersect(&img);
        if inner == r || inner.is_empty() {
            if inner.is_empty() {
                // Entirely outside one axis. Read each edge row once rather
                // than locking the same tile once for every replicated pixel.
                if y0 >= img.y1 || r.y1 <= 0 {
                    self.read_region_clamped(x0, y0.clamp(0, img.y1 - 1), w, 1, &mut dst[..w]);
                    for y in 1..h { let (first, rest) = dst.split_at_mut(y * w); rest[..w].copy_from_slice(&first[..w]); }
                } else {
                    let sx = x0.clamp(0, img.x1 - 1);
                    for y in 0..h { let p = self.pixel(sx, (y0 + y as i32).clamp(0, img.y1 - 1)); dst[y * w..(y + 1) * w].fill(p); }
                }
            }
            return;
        }
        let ix0 = (inner.x0 - x0) as usize;
        let ix1 = (inner.x1 - x0) as usize;
        let iy0 = (inner.y0 - y0) as usize;
        let iy1 = (inner.y1 - y0) as usize;
        for y in iy0..iy1 {
            let row = &mut dst[y * w..(y + 1) * w];
            let l = row[ix0];
            row[..ix0].fill(l);
            let rr = row[ix1 - 1];
            row[ix1..].fill(rr);
        }
        for y in 0..iy0 {
            let (top, rest) = dst.split_at_mut(iy0 * w);
            top[y * w..(y + 1) * w].copy_from_slice(&rest[..w]);
        }
        for y in iy1..h {
            let (head, tail) = dst.split_at_mut(y * w);
            tail[..w].copy_from_slice(&head[(iy1 - 1) * w..iy1 * w]);
        }
    }

    /// Number of distinct tiles actually allocated.
    pub fn unique_tiles(&self, seen: &mut std::collections::HashSet<usize>) -> usize {
        let mut n = 0;
        for t in self.tiles.iter().flatten() {
            if seen.insert(Arc::as_ptr(t) as usize) {
                n += 1;
            }
        }
        n
    }
}

/// Builds a new grid of `w x h` where each destination tile is produced by
/// `f(tile_rect)` in parallel.
pub fn build_grid<F>(w: u32, h: u32, f: F) -> TileGrid
where
    F: Fn(i32, i32) -> TileRef + Sync,
{
    use rayon::prelude::*;
    let mut g = TileGrid::new(w, h);
    let cols = g.cols as usize;
    g.tiles.par_iter_mut().enumerate().for_each(|(i, slot)| {
        *slot = f((i % cols) as i32, (i / cols) as i32);
    });
    g
}

/// Raw pointer wrapper for handing out disjoint `&mut` slots to rayon tasks.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// # Safety
    /// Callers must guarantee that no two live references alias the same index.
    #[inline]
    pub unsafe fn at(&self, i: usize) -> &mut T {
        unsafe { &mut *self.0.add(i) }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn shared_scratch_tiles_keep_copy_on_write_isolation() {
        let budget = super::super::memory::MemoryBudget::new(1024, std::env::temp_dir(), 0);
        let expected: Box<[u32]> = (0..super::TPX).map(|i| (i as u32).wrapping_mul(123456789)).collect();
        let tile = super::Tile {
            data: super::super::memory::PixelStore::with_budget(expected.clone(), false, budget.clone()),
            uniform: None,
        };
        let mut edited = Some(std::sync::Arc::new(tile));
        let original = edited.clone();
        let original_disk = budget.stats().disk;
        assert!(original_disk > 0);
        assert_eq!(budget.stats().disk, original_disk);
        super::tile_mut(&mut edited).write()[300] = 77;
        budget.maintain();
        assert_eq!(edited.as_ref().unwrap().read()[300], 77);
        assert_eq!(&*original.as_ref().unwrap().read(), &*expected);
        drop(edited);
        drop(original);
        assert_eq!((budget.stats().used, budget.stats().disk), (0, 0));
    }

    use super::*;

    #[test]
    fn region_read_crosses_tiles() {
        let mut g = TileGrid::new(600, 300);
        for ty in 0..g.rows as i32 {
            for tx in 0..g.cols as i32 {
                let i = g.idx(tx, ty);
                let t = tile_mut(&mut g.tiles[i]);
                let mut pixels = t.write();
                for y in 0..TS {
                    for x in 0..TS {
                        let gx = tx as u32 * 256 + x as u32;
                        let gy = ty as u32 * 256 + y as u32;
                        pixels[y * TS + x] = gx | gy << 16;
                    }
                }
            }
        }
        let mut buf = vec![0; 20 * 10];
        g.read_region(250, 250, 20, 10, &mut buf);
        for y in 0..10 {
            for x in 0..20 {
                assert_eq!(buf[y * 20 + x], (250 + x as u32) | (250 + y as u32) << 16);
            }
        }
        g.read_region(-5, -5, 10, 10, &mut buf);
        assert_eq!(buf[0], 0);
        assert_eq!(buf[5 * 10 + 5], 0);
        assert_eq!(buf[6 * 10 + 6], 1 | 1 << 16);
    }

    #[test]
    fn clamped_read_replicates_edges() {
        let g = TileGrid::filled(10, 10, 7);
        let mut buf = vec![0; 30 * 30];
        g.read_region_clamped(-10, -10, 30, 30, &mut buf);
        assert!(buf.iter().all(|&p| p == 7));
    }
    #[test]
    fn clamped_reads_match_pixel_reference_outside_each_edge() {
        let g = crate::engine::ops::procedural(17, 13);
        for (x0, y0) in [(-50, -50), (-50, 3), (4, -50), (50, 2), (2, 50), (50, 50), (-3, -2), (10, 7)] {
            let mut pixels = vec![0; 23 * 19];
            g.read_region_clamped(x0, y0, 23, 19, &mut pixels);
            for y in 0..19 { for x in 0..23 {
                assert_eq!(pixels[y * 23 + x], g.pixel((x0 + x as i32).clamp(0, 16), (y0 + y as i32).clamp(0, 12)));
            } }
        }
    }

}
