//! Sparse coverage masks: boolean selection ops, Select ▸ Modify, mask mips.
//!
//! Everything works per 256² tile. `Full` and `Empty` tiles short-circuit:
//! boolean ops on two uniform tiles never touch a pixel, and the
//! neighborhood operations (feather, expand, contract, smooth, border) only
//! compute tiles whose padded neighborhood is not uniform, so modifying an
//! ellipse selection on a 268 MP canvas costs only the tiles along its rim.
//!
//! Expand and contract use an exact Euclidean distance transform
//! (Felzenszwalb–Huttenlocher, linear time, radius-independent) so the grown
//! edge is a true circle offset; feather is the planar Gaussian from
//! `filters::fast`, and smooth is a box-mean majority vote.

use std::sync::Arc;

use rayon::prelude::*;

use super::grid::{RectI, TS, TS_I, TileGrid};
use super::selection::{MaskGrid, MaskTile, Selection, Shape, TileCoverage};

/// How a new selection combines with the existing one.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SelOp {
    #[default]
    New,
    Add,
    Subtract,
    Intersect,
}

impl SelOp {
    pub const ALL: [SelOp; 4] = [SelOp::New, SelOp::Add, SelOp::Subtract, SelOp::Intersect];

    pub fn label(self) -> &'static str {
        match self {
            SelOp::New => "New",
            SelOp::Add => "Add",
            SelOp::Subtract => "Subtract",
            SelOp::Intersect => "Intersect",
        }
    }

    /// Photoshop modifiers: Shift adds, Alt subtracts, Shift+Alt intersects.
    pub fn from_mods(shift: bool, alt: bool, default: SelOp) -> SelOp {
        match (shift, alt) {
            (true, true) => SelOp::Intersect,
            (true, false) => SelOp::Add,
            (false, true) => SelOp::Subtract,
            _ => default,
        }
    }
}

/// Collapses a partial tile that turned out all-0 or all-255.
pub fn normalize(px: Vec<u8>) -> MaskTile {
    let first = px[0];
    if (first == 0 || first == 255) && px.iter().all(|&v| v == first) {
        if first == 0 { MaskTile::Empty } else { MaskTile::Full }
    } else {
        MaskTile::Partial(Arc::new(crate::engine::memory::Bytes::new(px)))
    }
}

#[inline]
fn tile_value(t: &MaskTile, i: usize) -> u8 {
    match t {
        MaskTile::Empty => 0,
        MaskTile::Full => 255,
        MaskTile::Partial(m) => m[i],
    }
}

#[inline]
fn mul255(a: u32, b: u32) -> u32 {
    let v = a * b + 128;
    (v + (v >> 8)) >> 8
}

impl MaskGrid {
    pub fn full(w: u32, h: u32) -> Self {
        let mut m = Self::new(w, h);
        m.tiles.iter_mut().for_each(|t| *t = MaskTile::Full);
        m
    }

    #[inline]
    pub fn idx(&self, tx: i32, ty: i32) -> usize {
        (ty as u32 * self.cols + tx as u32) as usize
    }

    /// Tile `(tx, ty)`, or `Empty` outside the grid.
    #[inline]
    pub fn tile(&self, tx: i32, ty: i32) -> &MaskTile {
        const EMPTY: MaskTile = MaskTile::Empty;
        if tx < 0 || ty < 0 || tx >= self.cols as i32 || ty >= self.rows as i32 {
            return &EMPTY;
        }
        &self.tiles[self.idx(tx, ty)]
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.iter().all(|t| matches!(t, MaskTile::Empty))
    }

    /// Rasterizes any selection into a mask (only edge tiles get pixels).
    pub fn from_selection(sel: &Selection, w: u32, h: u32) -> MaskGrid {
        if let (Shape::Mask(m), false) = (&sel.shape, sel.inverted) {
            if (m.w, m.h) == (w, h) {
                return (**m).clone();
            }
        }
        let mut m = MaskGrid::new(w, h);
        let cols = m.cols as usize;
        m.tiles.par_iter_mut().enumerate().for_each(|(i, t)| {
            let (tx, ty) = ((i % cols) as i32, (i / cols) as i32);
            *t = match sel.tile_coverage(tx, ty) {
                TileCoverage::Full => MaskTile::Full,
                TileCoverage::Empty => MaskTile::Empty,
                TileCoverage::Partial(px) => normalize(px.into_vec()),
            };
        });
        m
    }

    pub fn to_selection(self) -> Option<Selection> {
        (!self.is_empty()).then(|| Selection { shape: Shape::Mask(Arc::new(self)), inverted: false })
    }

    /// Per-tile boolean combination; uniform tiles never touch pixels.
    pub fn combine(&self, b: &MaskGrid, op: SelOp) -> MaskGrid {
        debug_assert_eq!((self.w, self.h), (b.w, b.h));
        let tiles = self
            .tiles
            .par_iter()
            .zip(b.tiles.par_iter())
            .map(|(ta, tb)| combine_tile(ta, tb, op))
            .collect();
        MaskGrid { w: self.w, h: self.h, cols: self.cols, rows: self.rows, tiles }
    }

    pub fn invert(&self) -> MaskGrid {
        let tiles = self
            .tiles
            .par_iter()
            .map(|t| match t {
                MaskTile::Empty => MaskTile::Full,
                MaskTile::Full => MaskTile::Empty,
                MaskTile::Partial(m) => normalize(m.iter().map(|&v| 255 - v).collect()),
            })
            .collect();
        MaskGrid { tiles, ..self.clone_empty() }
    }

    fn clone_empty(&self) -> MaskGrid {
        MaskGrid { w: self.w, h: self.h, cols: self.cols, rows: self.rows, tiles: vec![] }
    }

    /// Hard edges (Anti-alias off): coverage ≥ 50% becomes fully selected.
    pub fn threshold(&self) -> MaskGrid {
        let tiles = self
            .tiles
            .par_iter()
            .map(|t| match t {
                MaskTile::Partial(m) => normalize(m.iter().map(|&v| if v >= 128 { 255 } else { 0 }).collect()),
                t => t.clone(),
            })
            .collect();
        MaskGrid { tiles, ..self.clone_empty() }
    }

    /// Copies `w x h` coverage values at `(x0, y0)` into `dst`. Pixels outside
    /// the mask replicate the nearest edge value when `clamp`, else read 0.
    pub fn read_region(&self, x0: i32, y0: i32, w: usize, h: usize, dst: &mut [u8], clamp: bool) {
        let (mw, mh) = (self.w as i32, self.h as i32);
        for y in 0..h {
            let mut gy = y0 + y as i32;
            let row = &mut dst[y * w..(y + 1) * w];
            if gy < 0 || gy >= mh {
                if !clamp {
                    row.fill(0);
                    continue;
                }
                gy = gy.clamp(0, mh - 1);
            }
            let ty = gy >> 8;
            let ly = (gy & 255) as usize;
            let mut x = 0usize;
            while x < w {
                let gx = x0 + x as i32;
                if gx < 0 || gx >= mw {
                    // Run of outside pixels up to the image edge.
                    let end = if gx < 0 { ((-x0) as usize).min(w) } else { w };
                    let v = if clamp { self.at(gx.clamp(0, mw - 1), gy) } else { 0 };
                    row[x..end].fill(v);
                    x = end;
                    continue;
                }
                let tx = gx >> 8;
                let lx = (gx & 255) as usize;
                let run = (TS - lx).min(w - x).min((mw - gx) as usize);
                match &self.tiles[self.idx(tx, ty)] {
                    MaskTile::Empty => row[x..x + run].fill(0),
                    MaskTile::Full => row[x..x + run].fill(255),
                    MaskTile::Partial(m) => row[x..x + run].copy_from_slice(&m[ly * TS + lx..ly * TS + lx + run]),
                }
                x += run;
            }
        }
    }

    /// Whether every tile overlapping `r` (clamped to the grid) is `Full` or
    /// every one is `Empty`.
    pub(crate) fn region_uniform(&self, r: RectI) -> Option<bool> {
        let r = r.intersect(&RectI::new(0, 0, self.w as i32, self.h as i32));
        let (tx0, ty0, tx1, ty1) = r.tile_span(self.cols, self.rows);
        let mut full = true;
        let mut empty = true;
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                match &self.tiles[self.idx(tx, ty)] {
                    MaskTile::Empty => full = false,
                    MaskTile::Full => empty = false,
                    MaskTile::Partial(_) => return None,
                }
                if !full && !empty {
                    return None;
                }
            }
        }
        if full {
            Some(true)
        } else if empty {
            Some(false)
        } else {
            None
        }
    }

    /// Runs a neighborhood operation over every tile whose padded
    /// neighborhood is not uniform. `f` gets a padded block (clamped reads)
    /// and must leave the result in place; tiles are processed in blocks of
    /// k×k sized to the padding so the padding overhead stays small.
    pub fn map_padded<F>(&self, pad: usize, f: F) -> MaskGrid
    where
        F: Fn(&mut [u8], usize, usize) + Sync,
    {
        let reach = pad.div_ceil(TS) as i32;
        let k: i32 = match pad {
            0..=48 => 1,
            49..=160 => 2,
            161..=480 => 4,
            _ => 8,
        };
        let pad_i = pad as i32;
        let (cols, rows) = (self.cols as i32, self.rows as i32);
        let blocks: Vec<(i32, i32)> =
            (0..rows).step_by(k as usize).flat_map(|by| (0..cols).step_by(k as usize).map(move |bx| (bx, by))).collect();
        let results: Vec<Vec<(usize, MaskTile)>> = blocks
            .par_iter()
            .map(|&(bx, by)| {
                let mut todo = Vec::new();
                for ty in by..(by + k).min(rows) {
                    for tx in bx..(bx + k).min(cols) {
                        let nb = RectI::new((tx - reach) * TS_I, (ty - reach) * TS_I, (tx + reach + 1) * TS_I, (ty + reach + 1) * TS_I);
                        if self.region_uniform(nb).is_none() {
                            todo.push((tx, ty));
                        }
                    }
                }
                if todo.is_empty() {
                    return vec![];
                }
                let rx0 = todo.iter().map(|t| t.0).min().unwrap();
                let ry0 = todo.iter().map(|t| t.1).min().unwrap();
                let rx1 = todo.iter().map(|t| t.0).max().unwrap() + 1;
                let ry1 = todo.iter().map(|t| t.1).max().unwrap() + 1;
                let (bw, bh) = (((rx1 - rx0) * TS_I + 2 * pad_i) as usize, ((ry1 - ry0) * TS_I + 2 * pad_i) as usize);
                let mut buf = vec![0u8; bw * bh];
                self.read_region(rx0 * TS_I - pad_i, ry0 * TS_I - pad_i, bw, bh, &mut buf, true);
                f(&mut buf, bw, bh);
                todo.into_iter()
                    .map(|(tx, ty)| {
                        let ox = ((tx - rx0) * TS_I + pad_i) as usize;
                        let oy = ((ty - ry0) * TS_I + pad_i) as usize;
                        let mut px = vec![0u8; TS * TS];
                        for y in 0..TS {
                            px[y * TS..(y + 1) * TS].copy_from_slice(&buf[(oy + y) * bw + ox..(oy + y) * bw + ox + TS]);
                        }
                        (self.idx(tx, ty), normalize(px))
                    })
                    .collect()
            })
            .collect();
        let mut out = self.clone();
        for (i, t) in results.into_iter().flatten() {
            out.tiles[i] = t;
        }
        out
    }

    /// Select ▸ Modify ▸ Feather: Gaussian blur of the coverage.
    pub fn feather(&self, radius: f32) -> MaskGrid {
        if radius < 0.3 {
            return self.clone();
        }
        let pad = (radius * 3.0).ceil() as usize + 2;
        self.map_padded(pad, |b, w, h| super::filters::fast::gauss_plane(b, w, h, radius))
    }

    /// Select ▸ Modify ▸ Expand by `r` pixels (exact circular offset).
    pub fn expand(&self, r: f32) -> MaskGrid {
        if r <= 0.0 {
            return self.clone();
        }
        let pad = r.ceil() as usize + 2;
        self.map_padded(pad, |b, w, h| offset_plane(b, w, h, r, true))
    }

    /// Select ▸ Modify ▸ Contract by `r` pixels.
    pub fn contract(&self, r: f32) -> MaskGrid {
        if r <= 0.0 {
            return self.clone();
        }
        let pad = r.ceil() as usize + 2;
        self.map_padded(pad, |b, w, h| offset_plane(b, w, h, r, false))
    }

    /// Select ▸ Modify ▸ Border: a band `width` wide centred on the edge,
    /// with softened sides like Photoshop's.
    pub fn border(&self, width: f32) -> MaskGrid {
        let half = (width * 0.5).max(0.5);
        let outer = self.expand(half);
        let inner = self.contract(half);
        outer.combine(&inner, SelOp::Subtract).feather((width * 0.12).clamp(0.5, 8.0))
    }

    /// Select ▸ Modify ▸ Smooth: majority vote over a (2r+1)² window, which
    /// removes specks and rounds jagged corners; the result keeps a 1 px
    /// anti-aliased edge.
    pub fn smooth(&self, r: usize) -> MaskGrid {
        if r == 0 {
            return self.clone();
        }
        let pad = r + 2;
        let k = 2 * r + 1;
        self.map_padded(pad, move |b, w, h| {
            super::filters::fast::box_plane(b, w, h, &[k]);
            for v in b.iter_mut() {
                // Ramp of ±1/2r around 50% keeps edges anti-aliased.
                let t = (*v as i32 - 128) * (k as i32) / 2 + 128;
                *v = t.clamp(0, 255) as u8;
            }
        })
    }

    /// Level-0 rects of tiles that differ between two masks of equal size.
    pub fn diff_rects(&self, other: &MaskGrid) -> Vec<RectI> {
        let cols = self.cols as usize;
        self.tiles
            .iter()
            .zip(&other.tiles)
            .enumerate()
            .filter(|(_, (a, b))| !same_mask_tile(a, b))
            .map(|(i, _)| TileGrid::tile_rect((i % cols) as i32, (i / cols) as i32))
            .collect()
    }

    /// Number of distinct partial tiles (memory estimate).
    #[allow(dead_code)]
    pub fn partial_tiles(&self) -> usize {
        self.tiles.iter().filter(|t| matches!(t, MaskTile::Partial(_))).count()
    }
}

pub fn same_mask_tile(a: &MaskTile, b: &MaskTile) -> bool {
    match (a, b) {
        (MaskTile::Empty, MaskTile::Empty) | (MaskTile::Full, MaskTile::Full) => true,
        (MaskTile::Partial(x), MaskTile::Partial(y)) => Arc::ptr_eq(x, y),
        _ => false,
    }
}

fn combine_tile(a: &MaskTile, b: &MaskTile, op: SelOp) -> MaskTile {
    use MaskTile::*;
    match op {
        SelOp::New => b.clone(),
        SelOp::Add => match (a, b) {
            (Full, _) | (_, Full) => Full,
            (Empty, x) | (x, Empty) => x.clone(),
            (Partial(x), Partial(y)) => normalize(x.iter().zip(y.iter()).map(|(&p, &q)| p.max(q)).collect()),
        },
        SelOp::Subtract => match (a, b) {
            (Empty, _) | (_, Full) => Empty,
            (x, Empty) => x.clone(),
            (Full, Partial(y)) => normalize(y.iter().map(|&q| 255 - q).collect()),
            (Partial(x), Partial(y)) => normalize(x.iter().zip(y.iter()).map(|(&p, &q)| p.min(255 - q)).collect()),
        },
        SelOp::Intersect => match (a, b) {
            (Empty, _) | (_, Empty) => Empty,
            (Full, x) | (x, Full) => x.clone(),
            (Partial(x), Partial(y)) => normalize(x.iter().zip(y.iter()).map(|(&p, &q)| p.min(q)).collect()),
        },
    }
}

/// Combines `new` into the current selection (None = nothing selected).
pub fn combine_selection(cur: Option<&Selection>, new: Option<Selection>, op: SelOp, w: u32, h: u32) -> Option<Selection> {
    match (op, cur, new) {
        (SelOp::New, _, n) => n,
        (SelOp::Add, None, n) => n,
        (SelOp::Add, Some(c), None) => Some(c.clone()),
        (SelOp::Subtract, None, _) | (SelOp::Intersect, None, _) => None,
        (SelOp::Subtract, Some(c), None) => Some(c.clone()),
        (SelOp::Intersect, Some(_), None) => None,
        (op, Some(c), Some(n)) => {
            let a = MaskGrid::from_selection(c, w, h);
            let b = MaskGrid::from_selection(&n, w, h);
            a.combine(&b, op).to_selection()
        }
    }
}

// ----- distance-transform offsets ------------------------------------------------

const INF: f32 = 1e20;

/// 1-D squared Euclidean distance transform of `f` (Felzenszwalb–Huttenlocher).
fn edt_1d(f: &[f32], d: &mut [f32], v: &mut Vec<usize>, z: &mut Vec<f32>) {
    let n = f.len();
    v.clear();
    v.resize(n, 0);
    z.clear();
    z.resize(n + 1, 0.0);
    let mut k = 0usize;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    let isect = |q: usize, p: usize| ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2.0 * (q as f32 - p as f32));
    for q in 1..n {
        let mut s = isect(q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = isect(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f32::INFINITY;
    }
    k = 0;
    for (q, dq) in d.iter_mut().enumerate().take(n) {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        let dx = q as f32 - p as f32;
        *dq = dx * dx + f[p];
    }
}

/// Squared distance from every pixel to the nearest pixel where `feature` holds.
pub fn edt(feature: &[bool], w: usize, h: usize) -> Vec<f32> {
    let mut g: Vec<f32> = feature.iter().map(|&f| if f { 0.0 } else { INF }).collect();
    // Columns: transpose so every pass is over contiguous memory.
    let mut t = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            t[x * h + y] = g[y * w + x];
        }
    }
    t.par_chunks_mut(h).for_each_init(
        || (vec![0f32; h], vec![0f32; h], Vec::new(), Vec::new()),
        |(f, d, v, z), col| {
            f.copy_from_slice(col);
            edt_1d(f, d, v, z);
            col.copy_from_slice(d);
        },
    );
    for y in 0..h {
        for x in 0..w {
            g[y * w + x] = t[x * h + y];
        }
    }
    g.par_chunks_mut(w).for_each_init(
        || (vec![0f32; w], vec![0f32; w], Vec::new(), Vec::new()),
        |(f, d, v, z), row| {
            f.copy_from_slice(row);
            edt_1d(f, d, v, z);
            row.copy_from_slice(d);
        },
    );
    g
}

/// Grows (`expand`) or shrinks the coverage in `b` by a circular offset of
/// `r`, with a one-pixel anti-aliased edge.
fn offset_plane(b: &mut [u8], w: usize, h: usize, r: f32, expand: bool) {
    let feature: Vec<bool> = b.iter().map(|&v| if expand { v >= 128 } else { v < 128 }).collect();
    let d2 = edt(&feature, w, h);
    for (v, &dd) in b.iter_mut().zip(&d2) {
        let d = dd.sqrt();
        if expand {
            let c = ((r + 1.0 - d).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            *v = (*v).max(c);
        } else {
            let c = ((d - r).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            *v = (*v).min(c);
        }
    }
}

// ----- mask pyramids (layer masks at display levels) --------------------------------

fn downsample_mask_tile(src: &MaskGrid, tx: i32, ty: i32) -> MaskTile {
    let kids = [src.tile(2 * tx, 2 * ty), src.tile(2 * tx + 1, 2 * ty), src.tile(2 * tx, 2 * ty + 1), src.tile(2 * tx + 1, 2 * ty + 1)];
    if kids.iter().all(|k| matches!(k, MaskTile::Empty)) {
        return MaskTile::Empty;
    }
    if kids.iter().all(|k| matches!(k, MaskTile::Full)) {
        return MaskTile::Full;
    }
    let mut px = vec![0u8; TS * TS];
    const H: usize = TS / 2;
    for (q, kid) in kids.iter().enumerate() {
        let (ox, oy) = ((q & 1) * H, (q >> 1) * H);
        match kid {
            MaskTile::Empty => {}
            MaskTile::Full => {
                for y in 0..H {
                    px[(oy + y) * TS + ox..(oy + y) * TS + ox + H].fill(255);
                }
            }
            MaskTile::Partial(m) => {
                for y in 0..H {
                    let r0 = &m[2 * y * TS..2 * y * TS + TS];
                    let r1 = &m[(2 * y + 1) * TS..(2 * y + 1) * TS + TS];
                    for x in 0..H {
                        let s = r0[2 * x] as u32 + r0[2 * x + 1] as u32 + r1[2 * x] as u32 + r1[2 * x + 1] as u32;
                        px[(oy + y) * TS + ox + x] = ((s + 2) >> 2) as u8;
                    }
                }
            }
        }
    }
    normalize(px)
}

pub fn build_mask_mips(base: &MaskGrid) -> Vec<MaskGrid> {
    let n = super::mips::mip_count(base.w, base.h);
    let mut levels: Vec<MaskGrid> = Vec::with_capacity(n as usize);
    for l in 1..=n {
        let (lw, lh) = super::mips::level_dims(base.w, base.h, l);
        let src = if l == 1 { base } else { &levels[l as usize - 2] };
        let mut g = MaskGrid::new(lw, lh);
        let cols = g.cols as usize;
        g.tiles.par_iter_mut().enumerate().for_each(|(i, t)| *t = downsample_mask_tile(src, (i % cols) as i32, (i / cols) as i32));
        levels.push(g);
    }
    levels
}

pub fn update_mask_mips(base: &MaskGrid, mips: &mut [MaskGrid], dirty: &[RectI]) {
    for l in 1..=mips.len() {
        let (before, after) = mips.split_at_mut(l - 1);
        let src = if l == 1 { base } else { &before[l - 2] };
        let dst = &mut after[0];
        let shift = super::grid::TSHIFT + l as u32;
        let mut list = Vec::new();
        let mut mark = vec![false; dst.tiles.len()];
        for r in dirty.iter().filter(|r| !r.is_empty()) {
            let tx0 = (r.x0.max(0) >> shift).min(dst.cols as i32 - 1);
            let ty0 = (r.y0.max(0) >> shift).min(dst.rows as i32 - 1);
            let tx1 = ((r.x1 - 1).max(0) >> shift).min(dst.cols as i32 - 1);
            let ty1 = ((r.y1 - 1).max(0) >> shift).min(dst.rows as i32 - 1);
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    let i = dst.idx(tx, ty);
                    if !mark[i] {
                        mark[i] = true;
                        list.push((tx, ty));
                    }
                }
            }
        }
        let fresh: Vec<MaskTile> = list.par_iter().map(|&(tx, ty)| downsample_mask_tile(src, tx, ty)).collect();
        for ((tx, ty), t) in list.into_iter().zip(fresh) {
            let i = dst.idx(tx, ty);
            dst.tiles[i] = t;
        }
    }
}

// ----- Quick Mask ------------------------------------------------------------------------

impl MaskGrid {
    /// Gray pixels (white = selected), sharing one tile per uniform value.
    pub fn to_gray(&self) -> TileGrid {
        let white = Arc::new(super::grid::Tile::solid(super::px::pack(255, 255, 255, 255)));
        let black = Arc::new(super::grid::Tile::solid(super::px::pack(0, 0, 0, 255)));
        let mut g = TileGrid::new(self.w, self.h);
        g.tiles = self
            .tiles
            .par_iter()
            .map(|t| match t {
                MaskTile::Full => Some(white.clone()),
                MaskTile::Empty => Some(black.clone()),
                MaskTile::Partial(m) => {
                    let px: Box<[u32]> = m.iter().map(|&v| super::px::pack(v, v, v, 255)).collect();
                    Some(Arc::new(super::grid::Tile::from_px(px)))
                }
            })
            .collect();
        g
    }

    /// Back from gray pixels (luminance of the red channel = coverage).
    pub fn from_gray(g: &TileGrid) -> MaskGrid {
        let mut m = MaskGrid::new(g.w, g.h);
        m.tiles = g.tiles.par_iter().map(gray_tile_to_mask).collect();
        m
    }
}

fn gray_tile_to_mask(t: &super::grid::TileRef) -> MaskTile {
    match t {
        None => MaskTile::Empty,
        Some(t) => match t.uniform {
            Some(c) => match super::px::unpack(c)[0] {
                0 => MaskTile::Empty,
                255 => MaskTile::Full,
                v => MaskTile::Partial(Arc::new(crate::engine::memory::Bytes::new(vec![v; TS * TS]))),
            },
            None => normalize(t.read().iter().map(|&p| super::px::unpack(p)[0]).collect()),
        },
    }
}

/// Red Quick Mask overlay tile for a gray tile: unselected areas get a 50%
/// red tint (Photoshop's default).
pub fn quick_mask_overlay_tile(t: &super::grid::TileRef) -> super::grid::TileRef {
    let red = super::px::pack(255, 0, 0, 255);
    let tint = |v: u8| super::px::scale(red, (255 - v as u32) / 2);
    match t {
        None => Some(Arc::new(super::grid::Tile::solid(tint(0)))),
        Some(t) => match t.uniform {
            Some(c) => {
                let v = super::px::unpack(c)[0];
                if v == 255 { None } else { Some(Arc::new(super::grid::Tile::solid(tint(v)))) }
            }
            None => super::grid::finish_tile(t.read().iter().map(|&p| tint(super::px::unpack(p)[0])).collect()),
        },
    }
}

/// Classification of a level-space region of a mask.
pub enum RegionCover {
    Full,
    Empty,
    Partial,
}

pub fn classify_region(m: &MaskGrid, r: RectI) -> RegionCover {
    match m.region_uniform(r) {
        Some(true) if r.intersect(&RectI::new(0, 0, m.w as i32, m.h as i32)) == r => RegionCover::Full,
        Some(false) => RegionCover::Empty,
        _ => RegionCover::Partial,
    }
}

/// Multiplies premultiplied pixels by mask coverage (0..255).
#[inline]
pub fn apply_coverage(buf: &mut [u32], cov: &[u8]) {
    for (p, &c) in buf.iter_mut().zip(cov) {
        match c {
            255 => {}
            0 => *p = 0,
            c => *p = super::px::scale(*p, c as u32),
        }
    }
}

#[allow(dead_code)]
pub(crate) fn mask_value(t: &MaskTile, i: usize) -> u8 {
    tile_value(t, i)
}

#[allow(dead_code)]
pub(crate) fn mul_cov(a: u8, b: u8) -> u8 {
    mul255(a as u32, b as u32) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disk(w: u32, h: u32, cx: f32, cy: f32, r: f32) -> MaskGrid {
        MaskGrid::from_selection(&Selection::ellipse(RectI::new((cx - r) as i32, (cy - r) as i32, (cx + r) as i32, (cy + r) as i32)), w, h)
    }

    #[test]
    fn boolean_truth_tables() {
        // Every combination of Empty / Full / Partial for both operands.
        let p = |v: u8| MaskTile::Partial(Arc::new(crate::engine::memory::Bytes::new(vec![v; TS * TS])));
        let tiles = [MaskTile::Empty, MaskTile::Full, p(100), p(200)];
        let val = |t: &MaskTile| tile_value(t, 77) as i32;
        for op in SelOp::ALL {
            for a in &tiles {
                for b in &tiles {
                    let (x, y) = (val(a), val(b));
                    let want = match op {
                        SelOp::New => y,
                        SelOp::Add => x.max(y),
                        SelOp::Subtract => x.min(255 - y),
                        SelOp::Intersect => x.min(y),
                    };
                    let got = val(&combine_tile(a, b, op));
                    assert_eq!(got, want, "{op:?} {x} {y}");
                }
            }
        }
        // Uniform operands stay uniform (no pixel buffers allocated).
        assert!(matches!(combine_tile(&MaskTile::Full, &MaskTile::Empty, SelOp::Add), MaskTile::Full));
        assert!(matches!(combine_tile(&MaskTile::Full, &MaskTile::Full, SelOp::Subtract), MaskTile::Empty));
        assert!(matches!(combine_tile(&p(255), &MaskTile::Full, SelOp::Intersect), MaskTile::Partial(_)));
        assert!(matches!(normalize(vec![255; TS * TS]), MaskTile::Full));
    }

    #[test]
    fn selection_combine_ops() {
        let (w, h) = (1000u32, 800u32);
        let a = Selection::rect(RectI::new(100, 100, 600, 500));
        let b = Selection::rect(RectI::new(400, 300, 900, 700));
        let add = combine_selection(Some(&a), Some(b.clone()), SelOp::Add, w, h).unwrap();
        let sub = combine_selection(Some(&a), Some(b.clone()), SelOp::Subtract, w, h).unwrap();
        let int = combine_selection(Some(&a), Some(b.clone()), SelOp::Intersect, w, h).unwrap();
        let m = |s: &Selection| MaskGrid::from_selection(s, w, h);
        let (ma, ms, mi) = (m(&add), m(&sub), m(&int));
        for &(x, y, ina, inb) in &[(150, 150, true, false), (450, 350, true, true), (800, 650, false, true), (50, 750, false, false)] {
            assert_eq!(ma.at(x, y) == 255, ina || inb, "add {x},{y}");
            assert_eq!(ms.at(x, y) == 255, ina && !inb, "sub {x},{y}");
            assert_eq!(mi.at(x, y) == 255, ina && inb, "int {x},{y}");
        }
        // Inverted selections rasterize too.
        let mut inv = a.clone();
        inv.inverted = true;
        let mi = MaskGrid::from_selection(&inv, w, h);
        assert_eq!(mi.at(10, 10), 255);
        assert_eq!(mi.at(300, 300), 0);
    }

    #[test]
    fn edt_matches_brute_force() {
        let (w, h) = (37usize, 29usize);
        let feat: Vec<bool> = (0..w * h).map(|i| (i * 7919) % 97 == 0).collect();
        let d = edt(&feat, w, h);
        for y in 0..h {
            for x in 0..w {
                let mut best = INF;
                for (j, &f) in feat.iter().enumerate() {
                    if f {
                        let (fx, fy) = ((j % w) as f32, (j / w) as f32);
                        best = best.min((fx - x as f32).powi(2) + (fy - y as f32).powi(2));
                    }
                }
                assert!((d[y * w + x] - best).abs() < 1e-3, "({x},{y}) {} vs {best}", d[y * w + x]);
            }
        }
    }

    #[test]
    fn expand_contract_are_circular_offsets() {
        let m = disk(1200, 1200, 600.0, 600.0, 200.0);
        let e = m.expand(50.0);
        let c = m.contract(50.0);
        for ang in [0.0f32, 0.7, 1.9, 3.3, 5.1] {
            let (s, co) = ang.sin_cos();
            let at = |g: &MaskGrid, r: f32| g.at((600.0 + co * r) as i32, (600.0 + s * r) as i32);
            assert_eq!(at(&e, 245.0), 255, "expand inside at {ang}");
            assert_eq!(at(&e, 255.0), 0, "expand outside at {ang}");
            assert_eq!(at(&c, 145.0), 255, "contract inside at {ang}");
            assert_eq!(at(&c, 155.0), 0, "contract outside at {ang}");
        }
        // Tiles far from the rim were never touched.
        let far = m.idx(0, 0);
        assert!(matches!(e.tiles[far], MaskTile::Empty));
    }

    #[test]
    fn feather_smooth_border() {
        let m = MaskGrid::from_selection(&Selection::rect(RectI::new(300, 300, 900, 900)), 1200, 1200);
        let f = m.feather(10.0);
        let edge = f.at(300, 600) as i32;
        assert!((edge - 128).abs() < 12, "edge {edge}");
        assert_eq!(f.at(600, 600), 255);
        assert_eq!(f.at(250, 600), 0);
        // Smooth removes an isolated speck.
        let mut speck = m.clone();
        let i = speck.idx(0, 0);
        let mut px = vec![0u8; TS * TS];
        px[100 * TS + 100] = 255;
        speck.tiles[i] = MaskTile::Partial(Arc::new(crate::engine::memory::Bytes::new(px)));
        let s = speck.smooth(2);
        assert_eq!(s.at(100, 100), 0);
        assert_eq!(s.at(600, 600), 255);
        // Border is a band around the edge.
        let b = m.border(20.0);
        assert!(b.at(300, 600) > 200);
        assert_eq!(b.at(600, 600), 0);
        assert_eq!(b.at(200, 600), 0);
    }

    #[test]
    fn select_all_survives_modify() {
        let m = MaskGrid::full(700, 500);
        for g in [m.contract(30.0), m.feather(20.0), m.smooth(4)] {
            assert_eq!(g.at(0, 0), 255);
            assert_eq!(g.at(699, 499), 255);
        }
    }

    #[test]
    fn mask_mips_average() {
        let m = disk(2048, 2048, 1024.0, 1024.0, 600.0);
        let mips = build_mask_mips(&m);
        assert_eq!(mips[0].at(512, 512), 255);
        assert_eq!(mips[0].at(5, 5), 0);
        let mut changed = m.combine(&disk(2048, 2048, 300.0, 300.0, 100.0), SelOp::Add);
        let dirty = m.diff_rects(&changed);
        let mut inc = mips.clone();
        update_mask_mips(&changed, &mut inc, &dirty);
        let full = build_mask_mips(&changed);
        for (a, b) in inc.iter().zip(&full) {
            for &(x, y) in &[(150, 150), (100, 100), (20, 30)] {
                assert_eq!(a.at(x >> 1, y >> 1), b.at(x >> 1, y >> 1));
            }
        }
        changed.tiles.clear();
    }
}
