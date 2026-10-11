//! Layer kinds beyond plain pixels: groups, adjustment layers, smart objects,
//! vector shapes (and text, in `text.rs`), plus layer masks.
//!
//! Every kind that produces pixels keeps a rasterized cache in the layer's
//! ordinary `TileGrid` + mips, so the compositor, display cache and export
//! treat it exactly like a pixel layer. Editing a vector description
//! re-rasterizes only the tiles its old and new bounds touch.

use std::sync::Arc;

use rayon::prelude::*;

use super::filters::Filter;
use super::grid::{RectI, TS, TS_I, TileGrid, TileRef, finish_tile};
use super::masks::{build_mask_mips, update_mask_mips};
use super::px;
use super::selection::MaskGrid;
use super::vector;

#[derive(Clone)]
pub enum LayerKind {
    Pixel,
    /// A folder. Pass-through groups composite their children straight into
    /// the backdrop; isolated ones composite them first and blend the result.
    Group { pass_through: bool, collapsed: bool },
    /// A non-destructive point filter applied to everything below it.
    Adjustment(Filter),
    /// A placed image with a transform (source pixels stay untouched).
    Smart(Arc<SmartObject>),
    /// A vector shape (fill + stroke).
    Shape(Arc<ShapeLayer>),
    /// A text layer (`text.rs`).
    Text(Arc<super::text::TextLayer>),
    /// A frame with optional placed content (`frame.rs`).
    Frame(Arc<super::frame::FrameLayer>),
    /// An artboard: a clipped container with a background (`artboard.rs`).
    Artboard(Arc<super::artboard::ArtboardLayer>),
}

impl LayerKind {
    pub fn name(&self) -> &'static str {
        match self {
            LayerKind::Pixel => "Pixel",
            LayerKind::Group { .. } => "Group",
            LayerKind::Adjustment(_) => "Adjustment",
            LayerKind::Smart(_) => "Smart Object",
            LayerKind::Shape(_) => "Shape",
            LayerKind::Text(_) => "Text",
            LayerKind::Frame(_) => "Frame",
            LayerKind::Artboard(_) => "Artboard",
        }
    }

    /// Kinds whose pixels are derived from a description (re-rasterized).
    pub fn is_vector(&self) -> bool {
        matches!(self, LayerKind::Smart(_) | LayerKind::Shape(_) | LayerKind::Text(_) | LayerKind::Frame(_))
    }

    /// Kinds that hold no pixels of their own.
    pub fn is_container(&self) -> bool {
        matches!(self, LayerKind::Group { .. } | LayerKind::Adjustment(_) | LayerKind::Artboard(_))
    }

    /// Folders in the Layers panel (groups and artboards): `Some(collapsed)`.
    pub fn folder(&self) -> Option<bool> {
        match self {
            LayerKind::Group { collapsed, .. } => Some(*collapsed),
            LayerKind::Artboard(a) => Some(a.collapsed),
            _ => None,
        }
    }

    /// Level-0 bounds of the content the description draws.
    pub fn content_bounds(&self) -> Option<RectI> {
        match self {
            LayerKind::Smart(s) => Some(s.bounds()),
            LayerKind::Shape(s) => Some(s.bounds()),
            LayerKind::Text(t) => Some(t.bounds()),
            LayerKind::Frame(f) => Some(f.bounds()),
            _ => None,
        }
    }

    /// Renders the tiles of `region` into `grid` (vector kinds only).
    pub fn rasterize(&self, grid: &mut TileGrid, region: RectI) {
        match self {
            LayerKind::Smart(s) => s.render(grid, region),
            LayerKind::Shape(s) => s.render(grid, region),
            LayerKind::Text(t) => t.render(grid, region),
            LayerKind::Frame(f) => f.render(grid, region),
            _ => {}
        }
    }
}

/// A layer mask with its own mip pyramid (for display at any zoom).
#[derive(Clone)]
pub struct LayerMask {
    pub grid: MaskGrid,
    pub mips: Vec<MaskGrid>,
    pub enabled: bool,
    /// Bumped on every edit (thumbnail refresh).
    pub version: u64,
}

impl LayerMask {
    pub fn new(grid: MaskGrid) -> Self {
        let mips = build_mask_mips(&grid);
        LayerMask { grid, mips, enabled: true, version: 0 }
    }

    pub fn level(&self, l: u32) -> &MaskGrid {
        if l == 0 || self.mips.is_empty() { &self.grid } else { &self.mips[(l as usize).min(self.mips.len()) - 1] }
    }

    /// Replaces the mask, updating only the mip tiles that changed.
    pub fn replace(&mut self, grid: MaskGrid) -> Vec<RectI> {
        if (grid.w, grid.h) != (self.grid.w, self.grid.h) {
            *self = LayerMask { enabled: self.enabled, version: self.version + 1, ..LayerMask::new(grid) };
            return vec![self.grid.bounds()];
        }
        self.version += 1;
        let rects = self.grid.diff_rects(&grid);
        self.grid = grid;
        update_mask_mips(&self.grid, &mut self.mips, &rects);
        rects
    }
}

// ----- smart objects -------------------------------------------------------------------

/// A placed image. `m` maps source pixel coordinates to document coordinates.
pub struct SmartObject {
    pub name: String,
    pub src: TileGrid,
    pub mips: Vec<TileGrid>,
    pub m: [f32; 9],
}

impl SmartObject {
    pub fn new(name: impl Into<String>, src: TileGrid, m: [f32; 9]) -> Self {
        let mips = super::mips::build_mips(&src);
        SmartObject { name: name.into(), src, mips, m }
    }

    pub fn with_matrix(&self, m: [f32; 9]) -> SmartObject {
        SmartObject { name: self.name.clone(), src: self.src.clone(), mips: self.mips.clone(), m }
    }

    pub fn corners(&self) -> [vector::P; 4] {
        let (w, h) = (self.src.w as f32, self.src.h as f32);
        [vector::p(0.0, 0.0), vector::p(w, 0.0), vector::p(w, h), vector::p(0.0, h)].map(|q| q.xform(&self.m))
    }

    pub fn bounds(&self) -> RectI {
        let c = self.corners();
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for q in c {
            x0 = x0.min(q.x);
            y0 = y0.min(q.y);
            x1 = x1.max(q.x);
            y1 = y1.max(q.y);
        }
        RectI::new(x0.floor() as i32 - 1, y0.floor() as i32 - 1, x1.ceil() as i32 + 1, y1.ceil() as i32 + 1)
    }

    fn level(&self, l: u32) -> &TileGrid {
        if l == 0 || self.mips.is_empty() { &self.src } else { &self.mips[(l as usize).min(self.mips.len()) - 1] }
    }

    /// Local scale (source px per document px) at the centre of output tile
    /// `(tx, ty)` and the source mip level [`SmartObject::render`] samples there.
    pub fn tile_level(inv: &[f32; 9], tx: i32, ty: i32, mips: u32) -> (f32, u32) {
        let tr = TileGrid::tile_rect(tx, ty);
        let c = vector::p(tr.x0 as f32 + 128.0, tr.y0 as f32 + 128.0);
        let s0 = c.xform(inv);
        let sx = (vector::p(c.x + 1.0, c.y).xform(inv) - s0).len();
        let sy = (vector::p(c.x, c.y + 1.0).xform(inv) - s0).len();
        let scale = sx.max(sy);
        let level = if scale >= 2.0 { (scale.log2().floor() as u32).min(mips) } else { 0 };
        (scale, level)
    }

    /// Resamples the tiles of `region` from the source mip that matches the
    /// local scale, bilinearly, with anti-aliased edges.
    pub fn render(&self, grid: &mut TileGrid, region: RectI) {
        let inv = super::filters::distort::mat_inv(&self.m);
        let b = self.bounds();
        let (tx0, ty0, tx1, ty1) = region.intersect(&grid.bounds()).tile_span(grid.cols, grid.rows);
        let cols = grid.cols;
        let list: Vec<(i32, i32)> = (ty0..ty1).flat_map(|ty| (tx0..tx1).map(move |tx| (tx, ty))).collect();
        let (sw, sh) = (self.src.w as f32, self.src.h as f32);
        let tiles: Vec<(usize, TileRef)> = list
            .par_iter()
            .map(|&(tx, ty)| {
                let idx = (ty as u32 * cols + tx as u32) as usize;
                let tr = TileGrid::tile_rect(tx, ty);
                if !tr.intersects(&b) {
                    return (idx, None);
                }
                let (scale, level) = Self::tile_level(&inv, tx, ty, self.mips.len() as u32);
                let k = (1u32 << level) as f32;
                let g = self.level(level);
                // A projective map has its extrema at the corners unless its
                // denominator crosses zero. Prefetch only this tile's footprint.
                let corners = [(tr.x0, tr.y0), (tr.x1, tr.y0), (tr.x1, tr.y1), (tr.x0, tr.y1)];
                let denominators = corners.map(|(x, y)| inv[6] * x as f32 + inv[7] * y as f32 + inv[8]);
                let bbox = if denominators.iter().all(|&d| d > 1e-6) || denominators.iter().all(|&d| d < -1e-6) {
                    let q = corners.map(|(x, y)| vector::p(x as f32, y as f32).xform(&inv));
                    Some(q.iter().fold((f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY), |(x0, y0, x1, y1), p| {
                        (x0.min(p.x / k), y0.min(p.y / k), x1.max(p.x / k), y1.max(p.y / k))
                    }))
                } else { None };
                let sampler = super::filters::distort::Sampler::new(g, bbox, TS * TS, false);
                let mut out = vec![0u32; TS * TS].into_boxed_slice();
                // Edge softness in source units: one destination pixel.
                let soft = scale.max(1e-3);
                for y in 0..TS {
                    for x in 0..TS {
                        let q = vector::p(tr.x0 as f32 + x as f32 + 0.5, tr.y0 as f32 + y as f32 + 0.5).xform(&inv);
                        let ex = ((q.x / soft + 0.5).min((sw - q.x) / soft + 0.5)).clamp(0.0, 1.0);
                        let ey = ((q.y / soft + 0.5).min((sh - q.y) / soft + 0.5)).clamp(0.0, 1.0);
                        let e = ex * ey;
                        if e <= 0.0 {
                            continue;
                        }
                        let v = sampler.sample(q.x / k, q.y / k);
                        out[y * TS + x] = if e >= 1.0 { v } else { px::scale(v, (e * 255.0 + 0.5) as u32) };
                    }
                }
                (idx, finish_tile(out))
            })
            .collect();
        for (i, t) in tiles {
            grid.tiles[i] = t;
        }
    }
}

// ----- shape layers: see `shape.rs` ---------------------------------------------------------

pub use super::shape::{GradPaint, Paint, PatPaint, PathOp, ShapeGeom, ShapeLayer, ShapePart};

/// Re-rasterizes a vector layer after its description changed from `old`
/// to `new`; only tiles inside either bounds are recomputed. Returns the
/// dirty level-0 rect.
pub fn rerender(grid: &mut TileGrid, old: Option<RectI>, new: &LayerKind) -> RectI {
    let nb = new.content_bounds().unwrap_or(RectI::EMPTY);
    let dirty = old.unwrap_or(RectI::EMPTY).union(&nb).intersect(&grid.bounds());
    if !dirty.is_empty() {
        // Snap to tiles: rasterize() clears and fills whole tiles.
        let (tx0, ty0, tx1, ty1) = dirty.tile_span(grid.cols, grid.rows);
        let snapped = RectI::new(tx0 * TS_I, ty0 * TS_I, tx1 * TS_I, ty1 * TS_I);
        new.rasterize(grid, snapped);
    }
    dirty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_object_scales_from_mips() {
        let src = super::super::ops::procedural(2000, 1000);
        // Place at 25%: the 1/4 mip is used, so the sample matches level 2 closely.
        let m = [0.25, 0.0, 100.0, 0.0, 0.25, 50.0, 0.0, 0.0, 1.0];
        let so = SmartObject::new("img", src.clone(), m);
        let mut g = TileGrid::new(1024, 1024);
        let b = g.bounds();
        so.render(&mut g, b);
        assert_eq!(g.pixel(50, 50), 0);
        let want = px::unpack(so.level(2).pixel(100, 60));
        let got = px::unpack(g.pixel(200, 110));
        for c in 0..3 {
            assert!((want[c] as i32 - got[c] as i32).abs() <= 24, "{want:?} {got:?}");
        }
    }
}
