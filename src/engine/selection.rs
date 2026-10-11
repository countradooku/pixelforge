//! Selections stored as shapes (or a sparse tile mask for the magic wand).
//!
//! Operations never rasterize the whole selection. They ask
//! [`Selection::classify`] per tile: fully covered and uncovered tiles take
//! fast paths, and only edge tiles compute per-pixel anti-aliased coverage.

use std::sync::Arc;

use super::grid::{RectI, TS, TS_I, TileGrid};

#[derive(Clone, Debug)]
pub enum MaskTile {
    Empty,
    Full,
    Partial(Arc<crate::engine::memory::Bytes>),
}

/// A sparse per-tile coverage mask (level-0 tiling).
#[derive(Clone, Debug)]
pub struct MaskGrid {
    pub w: u32,
    pub h: u32,
    pub cols: u32,
    pub rows: u32,
    pub tiles: Vec<MaskTile>,
}

impl MaskGrid {
    pub fn new(w: u32, h: u32) -> Self {
        let g = TileGrid::new(w, h);
        MaskGrid { w, h, cols: g.cols, rows: g.rows, tiles: vec![MaskTile::Empty; (g.cols * g.rows) as usize] }
    }

    #[inline]
    pub fn at(&self, x: i32, y: i32) -> u8 {
        if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
            return 0;
        }
        match &self.tiles[((y >> 8) as u32 * self.cols + (x >> 8) as u32) as usize] {
            MaskTile::Empty => 0,
            MaskTile::Full => 255,
            MaskTile::Partial(m) => m[((y & 255) as usize) * TS + (x & 255) as usize],
        }
    }

    pub fn bounds(&self) -> RectI {
        let mut b = RectI::EMPTY;
        for (i, t) in self.tiles.iter().enumerate() {
            if !matches!(t, MaskTile::Empty) {
                let tx = (i as u32 % self.cols) as i32;
                let ty = (i as u32 / self.cols) as i32;
                b = b.union(&TileGrid::tile_rect(tx, ty));
            }
        }
        b.intersect(&RectI::new(0, 0, self.w as i32, self.h as i32))
    }
}

#[derive(Clone, Debug)]
pub enum Shape {
    Rect { x0: f32, y0: f32, x1: f32, y1: f32 },
    Ellipse { cx: f32, cy: f32, rx: f32, ry: f32 },
    Poly(Arc<Vec<[f32; 2]>>),
    Mask(Arc<MaskGrid>),
}

#[derive(Clone, Debug)]
pub struct Selection {
    pub shape: Shape,
    pub inverted: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cover {
    Full,
    Empty,
    Partial,
}

pub enum TileCoverage {
    Full,
    Empty,
    Partial(Box<[u8]>),
}

impl TileCoverage {
    #[inline]
    pub fn at(&self, i: usize) -> u32 {
        match self {
            TileCoverage::Full => 255,
            TileCoverage::Empty => 0,
            TileCoverage::Partial(m) => m[i] as u32,
        }
    }
}

impl Selection {
    pub fn rect(r: RectI) -> Self {
        Selection {
            shape: Shape::Rect { x0: r.x0 as f32, y0: r.y0 as f32, x1: r.x1 as f32, y1: r.y1 as f32 },
            inverted: false,
        }
    }

    pub fn ellipse(r: RectI) -> Self {
        Selection {
            shape: Shape::Ellipse {
                cx: (r.x0 + r.x1) as f32 * 0.5,
                cy: (r.y0 + r.y1) as f32 * 0.5,
                rx: r.w() as f32 * 0.5,
                ry: r.h() as f32 * 0.5,
            },
            inverted: false,
        }
    }

    /// Bounding box of the shape itself (ignores inversion).
    pub fn shape_bounds(&self) -> RectI {
        match &self.shape {
            Shape::Rect { x0, y0, x1, y1 } => RectI::new(*x0 as i32, *y0 as i32, *x1 as i32, *y1 as i32),
            Shape::Ellipse { cx, cy, rx, ry } => RectI::new(
                (cx - rx).floor() as i32,
                (cy - ry).floor() as i32,
                (cx + rx).ceil() as i32,
                (cy + ry).ceil() as i32,
            ),
            Shape::Poly(pts) => {
                let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
                for p in pts.iter() {
                    b[0] = b[0].min(p[0]);
                    b[1] = b[1].min(p[1]);
                    b[2] = b[2].max(p[0]);
                    b[3] = b[3].max(p[1]);
                }
                RectI::new(b[0].floor() as i32, b[1].floor() as i32, b[2].ceil() as i32 + 1, b[3].ceil() as i32 + 1)
            }
            Shape::Mask(m) => m.bounds(),
        }
    }

    /// Region where coverage can be non-zero, clipped to the image.
    pub fn effect_bounds(&self, img: RectI) -> RectI {
        if self.inverted { img } else { self.shape_bounds().intersect(&img) }
    }

    fn classify_shape(&self, r: RectI) -> Cover {
        let b = self.shape_bounds();
        if !b.intersects(&r) {
            return Cover::Empty;
        }
        match &self.shape {
            Shape::Rect { .. } => {
                if b.contains_rect(&r) { Cover::Full } else { Cover::Partial }
            }
            Shape::Ellipse { cx, cy, rx, ry } => {
                let (rx, ry) = ((rx - 1.0).max(0.1), (ry - 1.0).max(0.1));
                let inside = |x: f32, y: f32| {
                    let dx = (x - cx) / rx;
                    let dy = (y - cy) / ry;
                    dx * dx + dy * dy <= 1.0
                };
                let (x0, y0, x1, y1) = (r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32);
                if inside(x0, y0) && inside(x1, y0) && inside(x0, y1) && inside(x1, y1) {
                    Cover::Full
                } else {
                    Cover::Partial
                }
            }
            Shape::Poly(pts) => {
                let (x0, y0, x1, y1) = (r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32);
                let n = pts.len();
                let crosses = (0..n).any(|i| {
                    let a = pts[i];
                    let c = pts[(i + 1) % n];
                    a[0].min(c[0]) <= x1 && a[0].max(c[0]) >= x0 && a[1].min(c[1]) <= y1 && a[1].max(c[1]) >= y0
                });
                if crosses {
                    Cover::Partial
                } else if point_in_poly(pts, (x0 + x1) * 0.5, (y0 + y1) * 0.5) {
                    Cover::Full
                } else {
                    Cover::Empty
                }
            }
            Shape::Mask(m) => {
                let (tx0, ty0, tx1, ty1) = r.tile_span(m.cols, m.rows);
                let aligned = r.x0 % TS_I == 0 && r.y0 % TS_I == 0 && r.w() == TS_I && r.h() == TS_I;
                let mut full = true;
                let mut empty = true;
                for ty in ty0..ty1 {
                    for tx in tx0..tx1 {
                        match &m.tiles[(ty as u32 * m.cols + tx as u32) as usize] {
                            MaskTile::Empty => full = false,
                            MaskTile::Full => empty = false,
                            MaskTile::Partial(_) => {
                                full = false;
                                empty = false;
                            }
                        }
                    }
                }
                if empty {
                    Cover::Empty
                } else if full && aligned {
                    Cover::Full
                } else {
                    Cover::Partial
                }
            }
        }
    }

    pub fn classify(&self, r: RectI) -> Cover {
        let c = self.classify_shape(r);
        if !self.inverted {
            return c;
        }
        match c {
            Cover::Full => Cover::Empty,
            Cover::Empty => Cover::Full,
            Cover::Partial => Cover::Partial,
        }
    }

    /// Coverage of `out.len()` samples along row `yc`, starting at x `xc0`
    /// and advancing by `step` (all in level-0 pixel units, sample centers).
    pub fn row(&self, yc: f32, xc0: f32, step: f32, out: &mut [u8]) {
        match &self.shape {
            Shape::Rect { x0, y0, x1, y1 } => {
                let inside_y = yc >= *y0 && yc < *y1;
                for (i, o) in out.iter_mut().enumerate() {
                    let x = xc0 + i as f32 * step;
                    *o = if inside_y && x >= *x0 && x < *x1 { 255 } else { 0 };
                }
            }
            Shape::Ellipse { cx, cy, rx, ry } => {
                let dy = (yc - cy) / ry;
                if dy.abs() >= 1.0 {
                    out.fill(0);
                } else {
                    let half = rx * (1.0 - dy * dy).sqrt();
                    span_coverage(&[(cx - half, cx + half)], xc0, step, out);
                }
            }
            Shape::Poly(pts) => {
                let mut xs: Vec<f32> = Vec::with_capacity(8);
                let n = pts.len();
                for i in 0..n {
                    let a = pts[i];
                    let b = pts[(i + 1) % n];
                    if (a[1] <= yc) != (b[1] <= yc) {
                        xs.push(a[0] + (yc - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
                    }
                }
                xs.sort_by(|a, b| a.total_cmp(b));
                let spans: Vec<(f32, f32)> = xs.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                span_coverage(&spans, xc0, step, out);
            }
            Shape::Mask(m) => {
                let y = yc.floor() as i32;
                for (i, o) in out.iter_mut().enumerate() {
                    *o = m.at((xc0 + i as f32 * step).floor() as i32, y);
                }
            }
        }
        if self.inverted {
            for o in out.iter_mut() {
                *o = 255 - *o;
            }
        }
    }

    /// Coverage of one pixel (0..255).
    pub fn at(&self, x: i32, y: i32) -> u8 {
        if let Shape::Mask(m) = &self.shape {
            let v = m.at(x, y);
            return if self.inverted { 255 - v } else { v };
        }
        let mut o = [0u8];
        self.row(y as f32 + 0.5, x as f32 + 0.5, 1.0, &mut o);
        o[0]
    }

    /// Level-0 coverage for one 256x256 tile.
    pub fn tile_coverage(&self, tx: i32, ty: i32) -> TileCoverage {
        let r = TileGrid::tile_rect(tx, ty);
        match self.classify(r) {
            Cover::Full => TileCoverage::Full,
            Cover::Empty => TileCoverage::Empty,
            Cover::Partial => {
                let mut m = vec![0u8; TS * TS].into_boxed_slice();
                for y in 0..TS {
                    self.row(r.y0 as f32 + y as f32 + 0.5, r.x0 as f32 + 0.5, 1.0, &mut m[y * TS..(y + 1) * TS]);
                }
                TileCoverage::Partial(m)
            }
        }
    }

    /// Coverage for a `w x h` region at mip `level` (negative levels are not
    /// needed: magnified views composite at level 0).
    pub fn region_coverage(&self, level: u32, x0: i32, y0: i32, w: usize, h: usize) -> Option<Vec<u8>> {
        let s = (1u32 << level) as f32;
        let r0 = RectI::wh(x0 << level, y0 << level, (w as i32) << level, (h as i32) << level);
        match self.classify(r0) {
            Cover::Full => None,
            c => {
                let mut m = vec![0u8; w * h];
                if c == Cover::Partial {
                    for y in 0..h {
                        self.row((y0 as f32 + y as f32 + 0.5) * s, (x0 as f32 + 0.5) * s, s, &mut m[y * w..(y + 1) * w]);
                    }
                }
                Some(m)
            }
        }
    }

    pub fn translate(&mut self, dx: f32, dy: f32) {
        match &mut self.shape {
            Shape::Rect { x0, y0, x1, y1 } => {
                *x0 += dx;
                *x1 += dx;
                *y0 += dy;
                *y1 += dy;
            }
            Shape::Ellipse { cx, cy, .. } => {
                *cx += dx;
                *cy += dy;
            }
            Shape::Poly(pts) => {
                *pts = Arc::new(pts.iter().map(|p| [p[0] + dx, p[1] + dy]).collect());
            }
            Shape::Mask(_) => {}
        }
    }

    /// Closed outlines in image coordinates for drawing marching ants.
    pub fn outlines(&self) -> Vec<Vec<[f32; 2]>> {
        match &self.shape {
            Shape::Rect { x0, y0, x1, y1 } => vec![vec![[*x0, *y0], [*x1, *y0], [*x1, *y1], [*x0, *y1]]],
            Shape::Ellipse { cx, cy, rx, ry } => {
                let n = 96;
                vec![(0..n)
                    .map(|i| {
                        let t = i as f32 / n as f32 * std::f32::consts::TAU;
                        [cx + rx * t.cos(), cy + ry * t.sin()]
                    })
                    .collect()]
            }
            Shape::Poly(pts) => vec![pts.as_ref().clone()],
            Shape::Mask(m) => mask_outline(m),
        }
    }

    /// Identity of the shape for caching derived data (outlines).
    pub fn cache_key(&self) -> (usize, bool) {
        match &self.shape {
            Shape::Mask(m) => (Arc::as_ptr(m) as usize, self.inverted),
            Shape::Poly(p) => (Arc::as_ptr(p) as usize, self.inverted),
            _ => (0, self.inverted),
        }
    }
}

fn point_in_poly(pts: &[[f32; 2]], x: f32, y: f32) -> bool {
    let mut inside = false;
    let n = pts.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (pts[i], pts[j]);
        if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Fractional horizontal coverage of sample cells by a set of spans.
fn span_coverage(spans: &[(f32, f32)], xc0: f32, step: f32, out: &mut [u8]) {
    out.fill(0);
    let n = out.len() as i32;
    let left0 = xc0 - step * 0.5;
    for &(a, b) in spans {
        if b <= a {
            continue;
        }
        let i0 = (((a - left0) / step).floor() as i32).max(0);
        let i1 = (((b - left0) / step).ceil() as i32).min(n);
        for i in i0..i1 {
            let cl = left0 + i as f32 * step;
            let cov = ((b.min(cl + step) - a.max(cl)) / step).clamp(0.0, 1.0);
            let v = out[i as usize] as f32 + cov * 255.0;
            out[i as usize] = v.min(255.0) as u8;
        }
    }
}

/// Boundary segments of a mask, traced on a coarse grid (≤ 1024 cells per
/// side) so huge masks stay cheap. A cell is "on" when any pixel in it is at
/// least half selected (max pooling), so 1-pixel rows and columns still get
/// marching ants. Full and Empty tiles are classified without pixel reads.
fn mask_outline(m: &MaskGrid) -> Vec<Vec<[f32; 2]>> {
    use rayon::prelude::*;
    let longest = m.w.max(m.h) as f32;
    let cell = (longest / 1024.0).ceil().max(1.0) as i32;
    let gw = (m.w as i32 + cell - 1) / cell;
    let gh = (m.h as i32 + cell - 1) / cell;
    let mut grid = vec![false; (gw * gh) as usize];
    // Cells per tile row/column (a cell never spans two tiles when cell
    // divides 256; otherwise fall back to per-pixel lookups).
    let aligned = TS_I % cell == 0;
    grid.par_chunks_mut(gw as usize).enumerate().for_each(|(gy, row)| {
        let y0 = gy as i32 * cell;
        let y1 = (y0 + cell).min(m.h as i32);
        for (gx, v) in row.iter_mut().enumerate() {
            let x0 = gx as i32 * cell;
            let x1 = (x0 + cell).min(m.w as i32);
            if aligned {
                let t = &m.tiles[((y0 >> 8) as u32 * m.cols + (x0 >> 8) as u32) as usize];
                *v = match t {
                    MaskTile::Full => true,
                    MaskTile::Empty => false,
                    MaskTile::Partial(px) => (y0..y1).any(|y| {
                        let r = ((y & 255) as usize) * TS;
                        px[r + (x0 & 255) as usize..r + ((x1 - 1) & 255) as usize + 1].iter().any(|&p| p >= 128)
                    }),
                };
            } else {
                *v = (y0..y1).any(|y| (x0..x1).any(|x| m.at(x, y) >= 128));
            }
        }
    });
    let on = |gx: i32, gy: i32| -> bool { gx >= 0 && gy >= 0 && gx < gw && gy < gh && grid[(gy * gw + gx) as usize] };
    let mut segs: Vec<Vec<[f32; 2]>> = Vec::new();
    let c = cell as f32;
    let (mw, mh) = (m.w as f32, m.h as f32);
    let cx = |g: i32| (g as f32 * c).min(mw);
    let cy = |g: i32| (g as f32 * c).min(mh);
    for gy in 0..=gh {
        let mut run: Option<i32> = None;
        for gx in 0..=gw {
            let edge = gx < gw && on(gx, gy) != on(gx, gy - 1);
            match (edge, run) {
                (true, None) => run = Some(gx),
                (false, Some(s)) => {
                    segs.push(vec![[cx(s), cy(gy)], [cx(gx), cy(gy)]]);
                    run = None;
                }
                _ => {}
            }
        }
    }
    for gx in 0..=gw {
        let mut run: Option<i32> = None;
        for gy in 0..=gh {
            let edge = gy < gh && on(gx, gy) != on(gx - 1, gy);
            match (edge, run) {
                (true, None) => run = Some(gy),
                (false, Some(s)) => {
                    segs.push(vec![[cx(gx), cy(s)], [cx(gx), cy(gy)]]);
                    run = None;
                }
                _ => {}
            }
        }
    }
    segs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ellipse_classification() {
        let s = Selection::ellipse(RectI::new(0, 0, 2048, 2048));
        assert_eq!(s.classify(RectI::wh(768, 768, 256, 256)), Cover::Full);
        assert_eq!(s.classify(RectI::wh(4096, 0, 256, 256)), Cover::Empty);
        assert_eq!(s.classify(RectI::wh(0, 0, 256, 256)), Cover::Partial);
    }

    #[test]
    fn poly_row_coverage() {
        let s = Selection {
            shape: Shape::Poly(Arc::new(vec![[10.0, 0.0], [20.0, 0.0], [20.0, 10.0], [10.0, 10.0]])),
            inverted: false,
        };
        let mut out = [0u8; 30];
        s.row(5.0, 0.5, 1.0, &mut out);
        assert_eq!(out[9], 0);
        assert_eq!(out[10], 255);
        assert_eq!(out[19], 255);
        assert_eq!(out[20], 0);
    }
}
