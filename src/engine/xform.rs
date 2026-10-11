//! Geometric transforms of whole layers and of the canvas (crop, rotated /
//! perspective crop, straighten, free transform). Vector kinds keep their
//! descriptions where the transform allows and are re-rendered rather than
//! resampled; layer masks follow their layer.

use std::sync::Arc;

use rayon::prelude::*;

use super::annot::Annotations;
use super::crop::Interp;
use super::document::{Document, Layer, PathItem};
use super::grid::{RectI, TileGrid};
use super::layers::{LayerKind, LayerMask};
use super::selection::MaskGrid;
use super::mips::build_mips;
use super::vector::p;

/// Integer translation `(tx, ty)` when `m` is one.
fn as_translation(m: &[f32; 9]) -> Option<(i32, i32)> {
    let id = (m[0] - 1.0).abs() < 1e-6 && m[1].abs() < 1e-6 && m[3].abs() < 1e-6 && (m[4] - 1.0).abs() < 1e-6 && m[6].abs() < 1e-9 && m[7].abs() < 1e-9;
    let (tx, ty) = (m[2].round(), m[5].round());
    (id && (m[2] - tx).abs() < 1e-4 && (m[5] - ty).abs() < 1e-4).then_some((tx as i32, ty as i32))
}

fn is_affine(m: &[f32; 9]) -> bool {
    m[6].abs() < 1e-9 && m[7].abs() < 1e-9
}

fn apply(m: &[f32; 9], x: f32, y: f32) -> (f32, f32) {
    let q = p(x, y).xform(m);
    (q.x, q.y)
}

/// Pixels of `g` moved through `m` into a `w × h` grid (pointer copies for
/// tile-aligned translations).
pub fn grid_transformed(g: &TileGrid, m: &[f32; 9], w: u32, h: u32) -> TileGrid {
    grid_transformed_with(g, m, w, h, None)
}

/// [`grid_transformed`] with a resampling filter for affine maps (`None`:
/// the mip-aware bilinear sampler, best for strong reductions).
pub fn grid_transformed_with(g: &TileGrid, m: &[f32; 9], w: u32, h: u32, interp: Option<Interp>) -> TileGrid {
    match (as_translation(m), interp) {
        (Some((tx, ty)), _) => super::ops::crop(g, RectI::new(-tx, -ty, -tx + w as i32, -ty + h as i32)),
        (None, Some(i)) if is_affine(m) => super::crop::warp_affine(g, m, w, h, i),
        (None, _) => super::annot::warp_grid(g, *m, w, h),
    }
}

/// How a canvas / layer transform treats pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct XformOpts {
    /// Resampling filter (`None`: mip-aware bilinear).
    pub interp: Option<Interp>,
    /// "Delete Cropped Pixels" off: pixel layers keep what leaves the canvas
    /// (the Background becomes "Layer 0", as in Photoshop).
    pub keep: bool,
    /// Leave plain pixel layers' grids empty: the caller computes them with
    /// a dedicated operation (exact flips, Image Size, Super Zoom).
    pub skip_pixels: bool,
}

fn mask_transformed(mk: &LayerMask, m: &[f32; 9], w: u32, h: u32) -> LayerMask {
    // Outside the old canvas a mask reveals (white), like Photoshop.
    let gray = mk.grid.to_gray();
    let moved = grid_transformed(&gray, m, w, h);
    let mut g = MaskGrid::from_gray(&moved);
    // Areas the transform brought in from beyond the old canvas are white.
    let inside = super::vector::rasterize_mask(
        &[super::vector::Poly { pts: [(0.0, 0.0), (gray.w as f32, 0.0), (gray.w as f32, gray.h as f32), (0.0, gray.h as f32)].iter().map(|&(x, y)| p(x, y).xform(m)).collect(), closed: true }],
        w,
        h,
        super::vector::FillRule::NonZero,
    );
    g = g.combine(&inside.invert(), super::masks::SelOp::Add);
    let mut lm = LayerMask::new(g);
    lm.enabled = mk.enabled;
    lm
}

/// The layer's kind after transform `m`, or `None` when it must become
/// pixels (text under anything but a translation).
pub fn kind_transformed(kind: &LayerKind, m: &[f32; 9]) -> Option<LayerKind> {
    Some(match kind {
        LayerKind::Pixel | LayerKind::Group { .. } | LayerKind::Adjustment(_) => kind.clone(),
        LayerKind::Smart(so) => LayerKind::Smart(Arc::new(so.with_matrix(super::filters::distort::mat_mul(m, &so.m)))),
        LayerKind::Shape(s) => {
            if !is_affine(m) {
                return None;
            }
            // Components stay parametric (rotation, box resize) where they can.
            LayerKind::Shape(Arc::new(s.transformed(&[m[0], m[1], m[2], m[3], m[4], m[5]])))
        }
        LayerKind::Artboard(a) => {
            // The board's rect follows the transform (bounding box of its corners).
            let r = a.rect;
            let c = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)].map(|(x, y)| (m[0] * x as f32 + m[1] * y as f32 + m[2], m[3] * x as f32 + m[4] * y as f32 + m[5]));
            let (x0, y0) = c.iter().fold((f32::MAX, f32::MAX), |a, p| (a.0.min(p.0), a.1.min(p.1)));
            let (x1, y1) = c.iter().fold((f32::MIN, f32::MIN), |a, p| (a.0.max(p.0), a.1.max(p.1)));
            let rect = RectI::new(x0.round() as i32, y0.round() as i32, x1.round() as i32, y1.round() as i32);
            LayerKind::Artboard(Arc::new(super::artboard::ArtboardLayer { rect, ..(**a).clone() }))
        }
        LayerKind::Frame(f) => {
            if !is_affine(m) {
                return None;
            }
            LayerKind::Frame(Arc::new(f.transformed(&[m[0], m[1], m[2], m[3], m[4], m[5]])))
        }
        LayerKind::Text(t) => {
            let (dx, dy) = as_translation(m)?;
            let mut nt = (**t).clone();
            nt.origin = (nt.origin.0 + dx as f32, nt.origin.1 + dy as f32);
            nt.relayout();
            LayerKind::Text(Arc::new(nt))
        }
    })
}

/// `layer` moved through `m` onto a `w × h` canvas.
pub fn layer_transformed(layer: &Layer, m: &[f32; 9], w: u32, h: u32) -> Layer {
    layer_transformed_with(layer, m, w, h, XformOpts::default())
}

/// [`layer_transformed`] with a filter, keeping pixels beyond the canvas
/// when asked (or when the layer already has hidden pixels).
pub fn layer_transformed_with(layer: &Layer, m: &[f32; 9], w: u32, h: u32, o: XformOpts) -> Layer {
    let mut out = layer.clone();
    out.hidden = None;
    if o.keep && out.background {
        out.background = false;
        out.name = "Layer 0".into();
    }
    let keep = (o.keep || layer.hidden.is_some()) && is_affine(m) && matches!(layer.kind, LayerKind::Pixel);
    if o.skip_pixels && !keep && matches!(layer.kind, LayerKind::Pixel) {
        out.grid = TileGrid::new(w, h);
        out.mips = build_mips(&out.grid);
        out.mask = layer.mask.as_ref().map(|mk| mask_transformed(mk, m, w, h));
        out.offset = (0, 0);
        out.version += 1;
        return out;
    }
    if keep {
        let k = super::crop::transformed_keep_masked(&layer.grid, layer.hidden.as_deref(), layer.mask.as_ref().map(|mk| &mk.grid), m, w, h, o.interp.unwrap_or_default());
        out.grid = k.grid;
        out.hidden = k.hidden;
        out.mips = build_mips(&out.grid);
        out.mask = layer.mask.as_ref().zip(k.mask).map(|(old, g)| {
            let mut lm = LayerMask::new(g);
            lm.enabled = old.enabled;
            lm
        });
        out.offset = (0, 0);
        out.version += 1;
        return out;
    }
    match kind_transformed(&layer.kind, m) {
        Some(k) if k.is_vector() => {
            let mut g = TileGrid::new(w, h);
            let b = g.bounds();
            k.rasterize(&mut g, b);
            out.grid = g;
            out.kind = k;
        }
        Some(k) => {
            out.grid = if k.is_container() { TileGrid::new(w, h) } else { grid_transformed_with(&layer.grid, m, w, h, o.interp) };
            out.kind = k;
        }
        None => {
            out.grid = grid_transformed_with(&layer.grid, m, w, h, o.interp);
            out.kind = LayerKind::Pixel;
        }
    }
    out.mips = build_mips(&out.grid);
    out.mask = layer.mask.as_ref().map(|mk| mask_transformed(mk, m, w, h));
    out.offset = (0, 0);
    out.version += 1;
    out
}

fn ann_transformed(a: &Annotations, m: &[f32; 9], w: u32, h: u32) -> Annotations {
    let full = RectI::wh(0, 0, w as i32, h as i32);
    let rect = |r: RectI| {
        let c = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)].map(|(x, y)| apply(m, x as f32, y as f32));
        let (x0, y0) = c.iter().fold((f32::MAX, f32::MAX), |a, q| (a.0.min(q.0), a.1.min(q.1)));
        let (x1, y1) = c.iter().fold((f32::MIN, f32::MIN), |a, q| (a.0.max(q.0), a.1.max(q.1)));
        RectI::new(x0.round() as i32, y0.round() as i32, x1.round() as i32, y1.round() as i32).intersect(&full)
    };
    let inside = |q: (f32, f32)| q.0 >= 0.0 && q.1 >= 0.0 && q.0 < w as f32 && q.1 < h as f32;
    let mut o = a.clone();
    // Samplers that leave the canvas go, with their Info modes.
    let kept: Vec<((f32, f32), super::color::ColorMode)> =
        a.samplers.iter().enumerate().map(|(i, &(x, y))| (apply(m, x, y), a.sampler_modes.get(i).copied().unwrap_or_default())).filter(|&(q, _)| inside(q)).collect();
    o.samplers = kept.iter().map(|k| k.0).collect();
    o.sampler_modes = kept.iter().map(|k| k.1).collect();
    // Planes follow the transform (corners through `m`).
    for pl in &mut o.vp_planes {
        for c in &mut pl.c {
            *c = apply(m, c.0, c.1);
        }
    }
    for n in &mut o.notes {
        let q = apply(m, n.x, n.y);
        (n.x, n.y) = (q.0.clamp(0.0, w as f32 - 1.0), q.1.clamp(0.0, h as f32 - 1.0));
    }
    for g in &mut o.counts {
        g.pts = g.pts.iter().map(|&(x, y)| apply(m, x, y)).filter(|&q| inside(q)).collect();
    }
    for ab in &mut o.artboards {
        ab.rect = rect(ab.rect);
    }
    o.artboards.retain(|ab| !ab.rect.is_empty());
    for s in &mut o.slices {
        s.rect = rect(s.rect);
    }
    o.slices.retain(|s| !s.rect.is_empty());
    o
}

/// The whole document moved through `m` onto a `w × h` canvas.
pub struct CanvasResult {
    pub w: u32,
    pub h: u32,
    pub layers: Vec<Layer>,
    pub paths: Vec<PathItem>,
    pub ann: Annotations,
}

pub fn canvas_transformed(layers: &[Layer], paths: &[PathItem], ann: &Annotations, m: &[f32; 9], w: u32, h: u32) -> CanvasResult {
    canvas_transformed_with(layers, paths, ann, m, w, h, XformOpts::default())
}

pub fn canvas_transformed_with(layers: &[Layer], paths: &[PathItem], ann: &Annotations, m: &[f32; 9], w: u32, h: u32, o: XformOpts) -> CanvasResult {
    let layers = layers.par_iter().map(|l| layer_transformed_with(l, m, w, h, o)).collect();
    let paths = paths.iter().map(|pi| PathItem { name: pi.name.clone(), path: pi.path.transformed(m) }).collect();
    CanvasResult { w, h, layers, paths, ann: ann_transformed(ann, m, w, h) }
}

impl Document {
    /// Installs a canvas transform result (one undo step is the caller's).
    pub fn install_canvas(&mut self, r: CanvasResult) {
        self.w = r.w;
        self.h = r.h;
        self.layers = r.layers;
        self.paths = r.paths;
        self.ann = r.ann;
        self.selection = None;
        self.preview = None;
        self.generation += 1;
        self.dirty.all = true;
        self.modified = true;
        self.active = self.active.min(self.layers.len().saturating_sub(1));
    }
}

/// Tight bounds of the non-transparent pixels of `g`: a tile-level scan,
/// then pixel scans of the boundary tiles only.
pub fn content_rect(g: &TileGrid) -> Option<RectI> {
    use super::grid::{TS, TS_I};
    let cols = g.cols as usize;
    let occupied = |i: usize| g.tiles[i].as_ref().is_some_and(|t| t.uniform.is_none_or(|c| super::px::alpha(c) != 0));
    let tiles: Vec<usize> = (0..g.tiles.len()).filter(|&i| occupied(i)).collect();
    if tiles.is_empty() {
        return None;
    }
    let (mut tx0, mut ty0, mut tx1, mut ty1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for &i in &tiles {
        let (tx, ty) = ((i % cols) as i32, (i / cols) as i32);
        (tx0, ty0, tx1, ty1) = (tx0.min(tx), ty0.min(ty), tx1.max(tx), ty1.max(ty));
    }
    let border: Vec<usize> = tiles.into_iter().filter(|&i| {
        let (tx, ty) = ((i % cols) as i32, (i / cols) as i32);
        tx == tx0 || tx == tx1 || ty == ty0 || ty == ty1
    }).collect();
    let r = border
        .par_iter()
        .filter_map(|&i| {
            let t = g.tiles[i].as_ref()?;
            let (ox, oy) = ((i % cols) as i32 * TS_I, (i / cols) as i32 * TS_I);
            if t.uniform.is_some() {
                return Some(RectI::new(ox, oy, ox + TS_I, oy + TS_I));
            }
            let mut b = RectI::EMPTY;
            let pixels = t.read();
            for y in 0..TS {
                let row = &pixels[y * TS..(y + 1) * TS];
                let Some(first) = row.iter().position(|&p| super::px::alpha(p) != 0) else { continue };
                let last = row.iter().rposition(|&p| super::px::alpha(p) != 0).unwrap();
                b = b.union(&RectI::new(ox + first as i32, oy + y as i32, ox + last as i32 + 1, oy + y as i32 + 1));
            }
            (!b.is_empty()).then_some(b)
        })
        .reduce(|| RectI::EMPTY, |a, b| a.union(&b));
    let r = r.intersect(&g.bounds());
    (!r.is_empty()).then_some(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::layers::{ShapeGeom, ShapeLayer};
    use crate::engine::px;
    use crate::engine::vector::StrokeStyle;

    #[test]
    fn crop_keeps_vector_descriptions_and_masks() {
        let mut doc = Document::from_grid(TileGrid::filled(1000, 800, px::pack(255, 255, 255, 255)), "bg");
        let shape = ShapeLayer::new(ShapeGeom::Rect { x0: 300.0, y0: 300.0, x1: 500.0, y1: 450.0, radii: [0.0; 4] }, crate::engine::layers::Paint::Solid([255, 0, 0]), None);
        let at = doc.insert_layer("shape", TileGrid::new(1000, 800));
        let kind = LayerKind::Shape(Arc::new(shape));
        let mut g = TileGrid::new(1000, 800);
        let b = g.bounds();
        kind.rasterize(&mut g, b);
        doc.layers[at].grid = g;
        doc.layers[at].kind = kind;
        let mut mask = MaskGrid::new(1000, 800);
        mask = mask.invert();
        doc.layers[at].mask = Some(LayerMask::new(mask));
        // Crop to (200, 250)–(700, 650).
        let m = [1.0, 0.0, -200.0, 0.0, 1.0, -250.0, 0.0, 0.0, 1.0];
        let r = canvas_transformed(&doc.layers, &doc.paths, &doc.ann, &m, 500, 400);
        let l = &r.layers[at];
        assert_eq!((l.grid.w, l.grid.h), (500, 400));
        assert_eq!(l.mask.as_ref().unwrap().grid.w, 500);
        let LayerKind::Shape(s) = &l.kind else { panic!("shape lost") };
        assert_eq!(s.geom.rect(), Some((100.0, 50.0, 300.0, 200.0)));
        assert_eq!(l.grid.pixel(150, 100), px::pack(255, 0, 0, 255));
        assert_eq!(l.grid.pixel(50, 20), 0);
    }

    #[test]
    fn rotate_90_moves_shapes_paths_and_hidden_pixels() {
        // Image ▸ Rotate 90° CW goes through the canvas transform: the shape
        // stays a shape, the path and the hidden pixels turn with it, and
        // plain pixel layers are left to the exact pixel op (skip_pixels).
        let (w, h) = (600u32, 400u32);
        let mut doc = Document::from_grid(TileGrid::filled(w, h, px::pack(255, 255, 255, 255)), "Background");
        let at = doc.insert_layer("shape", TileGrid::new(w, h));
        doc.layers[at].kind = LayerKind::Shape(Arc::new(ShapeLayer::new(ShapeGeom::Rect { x0: 100.0, y0: 50.0, x1: 300.0, y1: 150.0, radii: [0.0; 4] }, crate::engine::layers::Paint::Solid([0, 0, 255]), None)));
        let path = crate::engine::vector::Path::polygon(&[p(10.0, 20.0), p(60.0, 20.0), p(60.0, 70.0)]);
        doc.paths.push(PathItem { name: "p".into(), path });
        let keep = layer_transformed_with(&Layer::new(9, "k", crate::engine::ops::procedural(w, h)), &[1.0, 0.0, -100.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], w - 200, h, XformOpts { keep: true, ..Default::default() });
        doc.layers.push(keep);
        let m = [0.0, -1.0, h as f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let r = canvas_transformed_with(&doc.layers, &doc.paths, &doc.ann, &m, h, w, XformOpts { skip_pixels: true, ..Default::default() });
        let LayerKind::Shape(s) = &r.layers[at].kind else { panic!("shape lost") };
        let b = s.bounds();
        // (bounds include a few pixels of antialiasing margin)
        assert!([b.x0 - 250, b.y0 - 100, 350 - b.x1, 300 - b.y1].iter().all(|d| (-4..=0).contains(d)), "{b:?}");
        let (a, _) = r.paths[0].path.bounds().unwrap();
        assert_eq!((a.x.round(), a.y.round()), (330.0, 10.0));
        // Plain pixels are the caller's (empty here); hidden pixels turned too.
        assert_eq!(r.layers[0].grid.pixel(10, 10), 0);
        let hd = r.layers[2].hidden.as_ref().unwrap();
        assert_eq!(hd.rect(), RectI::new(0, -100, 400, 500));
    }

    #[test]
    fn content_rect_is_tight() {
        let mut g = TileGrid::new(2000, 1500);
        let mut st = crate::engine::brush::Stroke::brush(0, &g, crate::engine::brush::BrushSettings { size: 40.0, hardness: 1.0, ..Default::default() }, [255, 0, 0], false, None);
        st.add_point(300.0, 700.0, &mut g);
        st.add_point(1300.0, 900.0, &mut g);
        st.finish(&mut g);
        let r = content_rect(&g).unwrap();
        // Naive scan.
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, 0, 0);
        for y in 0..1500 {
            for x in 0..2000 {
                if px::alpha(g.pixel(x, y)) > 0 {
                    (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
                }
            }
        }
        assert_eq!(r, RectI::new(x0, y0, x1, y1));
        assert!(content_rect(&TileGrid::new(100, 100)).is_none());
    }

    #[test]
    fn rotation_keeps_shapes_live_and_turns_text_into_pixels() {
        use crate::engine::layers::Paint;
        let shape = LayerKind::Shape(Arc::new(ShapeLayer::new(ShapeGeom::Ellipse { x0: 0.0, y0: 0.0, x1: 10.0, y1: 10.0 }, Paint::Solid([0; 3]), Some((Paint::Solid([0; 3]), StrokeStyle::default())))));
        let m = super::super::annot::rotation_about(10.0, 50.0, 50.0);
        assert!(matches!(kind_transformed(&shape, &m), Some(LayerKind::Shape(s)) if matches!(s.geom, ShapeGeom::Ellipse { .. }) && s.rot != 0.0));
        let t = LayerKind::Text(Arc::new(crate::engine::text::TextLayer::new("hi", Default::default(), (10.0, 10.0), None, false)));
        assert!(kind_transformed(&t, &m).is_none());
        assert!(matches!(kind_transformed(&t, &[1.0, 0.0, 5.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0]), Some(LayerKind::Text(_))));
    }
}
