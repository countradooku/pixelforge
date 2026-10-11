//! Frame layers (Photoshop's Frame tool): a frame shape — rectangle,
//! ellipse, or any path (Convert to Frame turns shapes and text into frames)
//! — holding an optional placed image (a smart object) clipped to it. The
//! content moves and scales independently of the frame; the frame itself
//! transforms parametrically like a shape component.

use std::sync::Arc;

use rayon::prelude::*;

use super::grid::{RectI, TS, TileGrid, TileRef, finish_tile};
use super::layers::SmartObject;
use super::px;
use super::selection::MaskTile;
use super::shape::{Affine, ShapeGeom, ShapePart};
use super::vector::{self, FillRule, Poly};

#[derive(Clone)]
pub struct FrameLayer {
    pub shape: ShapePart,
    pub content: Option<Arc<SmartObject>>,
}

impl PartialEq for FrameLayer {
    fn eq(&self, o: &Self) -> bool {
        self.shape == o.shape
            && match (&self.content, &o.content) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b) || (a.name == b.name && a.m == b.m && a.src.w == b.src.w && a.src.h == b.src.h),
                _ => false,
            }
    }
}

impl std::fmt::Debug for FrameLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameLayer").field("shape", &self.shape).field("content", &self.content.as_ref().map(|c| c.name.clone())).finish()
    }
}

impl FrameLayer {
    pub fn new(geom: ShapeGeom) -> Self {
        FrameLayer { shape: ShapePart::new(geom, Default::default()), content: None }
    }

    pub fn outline(&self) -> Vec<Poly> {
        self.shape.doc_path().flatten(0.1)
    }

    /// The frame's box in document coordinates.
    pub fn frame_box(&self) -> Option<(f32, f32, f32, f32)> {
        let mut b = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for q in self.outline().iter().flat_map(|pl| pl.pts.iter()) {
            b = (b.0.min(q.x), b.1.min(q.y), b.2.max(q.x), b.3.max(q.y));
        }
        (b.0 <= b.2).then_some(b)
    }

    /// Content is clipped to the frame, so the frame's box bounds the pixels.
    pub fn bounds(&self) -> RectI {
        match self.frame_box() {
            Some((x0, y0, x1, y1)) => RectI::new(x0.floor() as i32 - 1, y0.floor() as i32 - 1, x1.ceil() as i32 + 1, y1.ceil() as i32 + 1),
            None => RectI::EMPTY,
        }
    }

    /// `so` scaled to cover (`fill`) or fit inside the frame's box, centred.
    pub fn fitted(&self, so: &SmartObject, fill: bool) -> SmartObject {
        let Some((x0, y0, x1, y1)) = self.frame_box() else { return so.with_matrix(so.m) };
        let (fw, fh) = (x1 - x0, y1 - y0);
        let (sw, sh) = (so.src.w as f32, so.src.h as f32);
        let k = if fill { (fw / sw).max(fh / sh) } else { (fw / sw).min(fh / sh) };
        let (ox, oy) = (x0 + (fw - sw * k) * 0.5, y0 + (fh - sh * k) * 0.5);
        so.with_matrix([k, 0.0, ox, 0.0, k, oy, 0.0, 0.0, 1.0])
    }

    /// Content scale (uniform part of its matrix) and top-left position.
    pub fn content_placement(&self) -> Option<(f32, f32, f32)> {
        let c = self.content.as_ref()?;
        Some(((c.m[0] * c.m[4] - c.m[1] * c.m[3]).abs().sqrt(), c.m[2], c.m[5]))
    }

    /// The content moved / scaled (about the frame's centre when scaling).
    pub fn with_content_matrix(&self, m: [f32; 9]) -> FrameLayer {
        FrameLayer { shape: self.shape.clone(), content: self.content.as_ref().map(|c| Arc::new(c.with_matrix(m))) }
    }

    /// The frame and its content under a document transform.
    pub fn transformed(&self, m: &Affine) -> FrameLayer {
        let m9 = [m[0], m[1], m[2], m[3], m[4], m[5], 0.0, 0.0, 1.0];
        FrameLayer { shape: self.shape.transformed(m), content: self.content.as_ref().map(|c| Arc::new(c.with_matrix(super::filters::distort::mat_mul(&m9, &c.m)))) }
    }

    pub fn render(&self, grid: &mut TileGrid, region: RectI) {
        let region = region.intersect(&grid.bounds());
        let (tx0, ty0, tx1, ty1) = region.tile_span(grid.cols, grid.rows);
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                let i = grid.idx(tx, ty);
                grid.tiles[i] = None;
            }
        }
        let Some(c) = &self.content else { return };
        let snapped = RectI::new(tx0 * TS as i32, ty0 * TS as i32, tx1 * TS as i32, ty1 * TS as i32).intersect(&grid.bounds());
        let mut tmp = TileGrid::new(grid.w, grid.h);
        c.render(&mut tmp, snapped);
        let mask = vector::rasterize(&vector::edges(&self.outline()), grid.cols, grid.rows, snapped, FillRule::NonZero);
        let done: Vec<(usize, TileRef)> = mask
            .into_par_iter()
            .map(|(i, mt)| {
                let src = tmp.tiles[i].clone();
                let t = match (mt, src) {
                    (_, None) | (MaskTile::Empty, _) => None,
                    (MaskTile::Full, s) => s,
                    (MaskTile::Partial(m), Some(s)) => finish_tile(s.read().iter().zip(m.iter()).map(|(&p, &a)| px::scale(p, a as u32)).collect()),
                };
                (i, t)
            })
            .collect();
        for (i, t) in done {
            grid.tiles[i] = t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_clips_content_and_fits() {
        let img = super::super::grid::TileGrid::filled(400, 200, px::pack(200, 30, 30, 255));
        let mut f = FrameLayer::new(ShapeGeom::Ellipse { x0: 100.0, y0: 100.0, x1: 300.0, y1: 300.0 });
        let so = SmartObject::new("img", img, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        // Fill covers the 200 × 200 frame: scale 1, centred.
        let filled = f.fitted(&so, true);
        assert_eq!((filled.m[0], filled.m[2], filled.m[5]), (1.0, 0.0, 100.0));
        let fit = f.fitted(&so, false);
        assert!((fit.m[0] - 0.5).abs() < 1e-6 && (fit.m[5] - 150.0).abs() < 1e-3);
        f.content = Some(Arc::new(filled));
        let mut g = TileGrid::new(512, 512);
        let full = g.bounds();
        f.render(&mut g, full);
        // Inside the ellipse: content; its corners and outside: transparent.
        assert_eq!(g.pixel(200, 200), px::pack(200, 30, 30, 255));
        assert_eq!(px::alpha(g.pixel(105, 105)), 0);
        assert_eq!(px::alpha(g.pixel(50, 200)), 0);
        // Ink only inside the bounds.
        let b = f.bounds();
        assert!(b.x0 >= 98 && b.x1 <= 302);
        // Transforming moves frame and content together.
        let t = f.transformed(&[1.0, 0.0, 50.0, 0.0, 1.0, 0.0]);
        assert_eq!(t.content.as_ref().unwrap().m[2], 50.0);
        assert_eq!(t.shape.geom.rect(), Some((150.0, 100.0, 350.0, 300.0)));
        // An empty frame draws nothing.
        let mut g2 = TileGrid::new(512, 512);
        FrameLayer::new(ShapeGeom::Rect { x0: 0.0, y0: 0.0, x1: 300.0, y1: 300.0, radii: [0.0; 4] }).render(&mut g2, full);
        assert!(g2.tiles.iter().all(|t| t.is_none()));
    }
}
