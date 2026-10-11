//! Shape layers: live, parametric vector shapes.
//!
//! - **Components.** A shape layer is one or more components; each is a
//!   parametric geometry (rectangle with per-corner radii, ellipse,
//!   polygon / star, line, custom shape or free path) with its own rotation
//!   and transform, combined with the components before it by a path
//!   operation (Combine, Subtract Front, Intersect, Exclude). The booleans
//!   are live: the geometry stays editable until Merge Shape Components
//!   replaces the layer with the resulting path (`pathops.rs`).
//! - **Rotation** is stored per component (about its box centre), so a
//!   rotated rectangle is still a rectangle with editable size and radii.
//! - **Paint.** Fill and stroke are each none, a solid colour, a gradient
//!   (any of the five gradient kinds, at an angle and scale, aligned with
//!   the shape's bounds) or a pattern (scale and angle). Strokes are
//!   aligned inside / centre / outside, with caps, joins, a dash pattern
//!   and a dash offset.

use std::sync::Arc;

use rayon::prelude::*;

use super::gradient::{GradKind, Gradient, GradientFill};
use super::grid::{RectI, TS, Tile, TileGrid, TileRef, finish_tile};
use super::pathops::{self, BoolOp};
use super::px;
use super::selection::{MaskGrid, MaskTile};
use super::vector::{self, Align, Anchor, FillRule, P, Path, Poly, StrokeStyle, SubPath, p};

/// A 2-D affine map `[a, b, c, d, e, f]`: `x' = a·x + b·y + c`, `y' = d·x + e·y + f`.
pub type Affine = [f32; 6];
pub const IDENTITY: Affine = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

pub fn affine_apply(m: &Affine, q: P) -> P {
    p(m[0] * q.x + m[1] * q.y + m[2], m[3] * q.x + m[4] * q.y + m[5])
}

/// `a ∘ b` (apply `b` first).
pub fn affine_mul(a: &Affine, b: &Affine) -> Affine {
    [
        a[0] * b[0] + a[1] * b[3],
        a[0] * b[1] + a[1] * b[4],
        a[0] * b[2] + a[1] * b[5] + a[2],
        a[3] * b[0] + a[4] * b[3],
        a[3] * b[1] + a[4] * b[4],
        a[3] * b[2] + a[4] * b[5] + a[5],
    ]
}

pub fn affine_inverse(m: &Affine) -> Affine {
    let det = m[0] * m[4] - m[1] * m[3];
    let id = if det.abs() < 1e-12 { 0.0 } else { 1.0 / det };
    let (a, b, d, e) = (m[4] * id, -m[1] * id, -m[3] * id, m[0] * id);
    [a, b, -(a * m[2] + b * m[5]), d, e, -(d * m[2] + e * m[5])]
}

fn to9(m: &Affine) -> [f32; 9] {
    [m[0], m[1], m[2], m[3], m[4], m[5], 0.0, 0.0, 1.0]
}

/// Rotation by `rot` radians (clockwise on screen) about `c`.
pub fn rotation_about(rot: f32, c: P) -> Affine {
    let (s, co) = rot.sin_cos();
    [co, -s, c.x - co * c.x + s * c.y, s, co, c.y - s * c.x - co * c.y]
}

// ----- geometry ------------------------------------------------------------------------------

/// How a shape was drawn, so it can be live-edited parametrically.
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeGeom {
    Rect { x0: f32, y0: f32, x1: f32, y1: f32, radii: [f32; 4] },
    Ellipse { x0: f32, y0: f32, x1: f32, y1: f32 },
    Triangle { x0: f32, y0: f32, x1: f32, y1: f32 },
    Polygon { x0: f32, y0: f32, x1: f32, y1: f32, sides: u32, star: f32 },
    Line { a: (f32, f32), b: (f32, f32), weight: f32, arrow_start: Option<(f32, f32)>, arrow_end: Option<(f32, f32)> },
    Custom { x0: f32, y0: f32, x1: f32, y1: f32, which: usize },
    /// A free path (Pen tool, Convert to Shape, merged components).
    Path(Path),
}

impl ShapeGeom {
    pub fn path(&self) -> Path {
        match *self {
            ShapeGeom::Rect { x0, y0, x1, y1, radii } => Path::rounded_rect(x0, y0, x1, y1, radii),
            ShapeGeom::Ellipse { x0, y0, x1, y1 } => Path::ellipse((x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0).abs() * 0.5, (y1 - y0).abs() * 0.5),
            ShapeGeom::Triangle { x0, y0, x1, y1 } => Path::triangle(x0, y0, x1, y1),
            ShapeGeom::Polygon { x0, y0, x1, y1, sides, star } => Path::star((x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0).abs() * 0.5, (y1 - y0).abs() * 0.5, sides, star, 0.0),
            ShapeGeom::Line { a, b, weight, arrow_start, arrow_end } => Path::line_shape(p(a.0, a.1), p(b.0, b.1), weight, arrow_start, arrow_end),
            ShapeGeom::Custom { x0, y0, x1, y1, which } => super::shapes::custom_shape(which, x0, y0, x1, y1),
            ShapeGeom::Path(ref pa) => pa.clone(),
        }
    }

    /// Bounding box corners, for live resize handles.
    pub fn rect(&self) -> Option<(f32, f32, f32, f32)> {
        match *self {
            ShapeGeom::Rect { x0, y0, x1, y1, .. }
            | ShapeGeom::Ellipse { x0, y0, x1, y1 }
            | ShapeGeom::Triangle { x0, y0, x1, y1 }
            | ShapeGeom::Polygon { x0, y0, x1, y1, .. }
            | ShapeGeom::Custom { x0, y0, x1, y1, .. } => Some((x0, y0, x1, y1)),
            ShapeGeom::Line { a, b, .. } => Some((a.0, a.1, b.0, b.1)),
            ShapeGeom::Path(_) => None,
        }
    }

    /// The box (or path bounds) centre: the pivot of the component's rotation.
    pub fn center(&self) -> P {
        match self.rect() {
            Some((x0, y0, x1, y1)) => p((x0 + x1) * 0.5, (y0 + y1) * 0.5),
            None => self.path().bounds().map_or(p(0.0, 0.0), |(lo, hi)| lo.lerp(hi, 0.5)),
        }
    }

    /// Same geometry stretched to a new bounding box.
    pub fn with_rect(&self, nx0: f32, ny0: f32, nx1: f32, ny1: f32) -> ShapeGeom {
        let mut g = self.clone();
        match &mut g {
            ShapeGeom::Rect { x0, y0, x1, y1, .. }
            | ShapeGeom::Ellipse { x0, y0, x1, y1 }
            | ShapeGeom::Triangle { x0, y0, x1, y1 }
            | ShapeGeom::Polygon { x0, y0, x1, y1, .. }
            | ShapeGeom::Custom { x0, y0, x1, y1, .. } => {
                *x0 = nx0;
                *y0 = ny0;
                *x1 = nx1;
                *y1 = ny1;
            }
            ShapeGeom::Line { a, b, .. } => {
                *a = (nx0, ny0);
                *b = (nx1, ny1);
            }
            ShapeGeom::Path(_) => {}
        }
        g
    }

    /// Same geometry moved by `(dx, dy)`.
    pub fn translated(&self, dx: f32, dy: f32) -> ShapeGeom {
        match (self, self.rect()) {
            (ShapeGeom::Path(pa), _) => ShapeGeom::Path(pa.transformed(&[1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0])),
            (_, Some((x0, y0, x1, y1))) => self.with_rect(x0 + dx, y0 + dy, x1 + dx, y1 + dy),
            _ => self.clone(),
        }
    }
}

/// Photoshop's path operations, applied to a component and everything
/// before it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum PathOp {
    /// Combine Shapes (union).
    #[default]
    Combine,
    /// Subtract Front Shape.
    Subtract,
    /// Intersect Shape Areas.
    Intersect,
    /// Exclude Overlapping Shapes (xor).
    Exclude,
}

impl PathOp {
    pub const ALL: [PathOp; 4] = [PathOp::Combine, PathOp::Subtract, PathOp::Intersect, PathOp::Exclude];
    pub fn label(self) -> &'static str {
        match self {
            PathOp::Combine => "Combine",
            PathOp::Subtract => "Subtract",
            PathOp::Intersect => "Intersect",
            PathOp::Exclude => "Exclude",
        }
    }
    fn bool_op(self) -> BoolOp {
        match self {
            PathOp::Combine => BoolOp::Union,
            PathOp::Subtract => BoolOp::Subtract,
            PathOp::Intersect => BoolOp::Intersect,
            PathOp::Exclude => BoolOp::Xor,
        }
    }
}

/// One component of a shape layer.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapePart {
    pub geom: ShapeGeom,
    /// Rotation about the geometry's centre (radians, clockwise on screen).
    pub rot: f32,
    /// Any further transform (skew, non-uniform scale of rotated shapes).
    pub xf: Affine,
    /// How this component combines with the ones before it.
    pub op: PathOp,
}

impl ShapePart {
    pub fn new(geom: ShapeGeom, op: PathOp) -> Self {
        ShapePart { geom, rot: 0.0, xf: IDENTITY, op }
    }

    /// Geometry space → document.
    pub fn matrix(&self) -> Affine {
        affine_mul(&self.xf, &rotation_about(self.rot, self.geom.center()))
    }

    pub fn doc_path(&self) -> Path {
        let m = self.matrix();
        if m == IDENTITY { self.geom.path() } else { self.geom.path().transformed(&to9(&m)) }
    }

    /// The component under a document transform, kept parametric where
    /// possible: translations move the geometry, rotations add to `rot`,
    /// axis-aligned scales of unrotated boxes resize the box; anything else
    /// composes into `xf`.
    pub fn transformed(&self, m: &Affine) -> ShapePart {
        let mut out = self.clone();
        let (a, b, d, e) = (m[0], m[1], m[3], m[4]);
        let det = a * e - b * d;
        let ortho = (a * a + d * d - 1.0).abs() < 1e-4 && (b * b + e * e - 1.0).abs() < 1e-4 && (a * b + d * e).abs() < 1e-4 && det > 0.0;
        let axis = b.abs() < 1e-6 && d.abs() < 1e-6 && a > 0.0 && e > 0.0;
        if self.xf == IDENTITY && ortho {
            // Rigid motion: the centre moves, the angle adds.
            let c = self.geom.center();
            let nc = affine_apply(m, c);
            out.geom = self.geom.translated(nc.x - c.x, nc.y - c.y);
            out.rot = self.rot + d.atan2(a);
        } else if self.xf == IDENTITY && axis && (self.rot.abs() < 1e-6 || matches!(self.geom, ShapeGeom::Path(_))) {
            out.geom = match (&self.geom, self.geom.rect()) {
                (ShapeGeom::Path(pa), _) => ShapeGeom::Path(pa.transformed(&to9(m))),
                (g, Some((x0, y0, x1, y1))) => {
                    let (q0, q1) = (affine_apply(m, p(x0, y0)), affine_apply(m, p(x1, y1)));
                    let mut g2 = g.with_rect(q0.x, q0.y, q1.x, q1.y);
                    if let ShapeGeom::Rect { radii, .. } = &mut g2 {
                        // Corner radii scale with the box (as Photoshop's live shapes).
                        let k = (a * e).sqrt();
                        *radii = radii.map(|r| r * k);
                    }
                    g2
                }
                _ => self.geom.clone(),
            };
            if let ShapeGeom::Line { weight, .. } = &mut out.geom {
                *weight *= (a * e).sqrt();
            }
        } else {
            out.xf = affine_mul(m, &self.xf);
        }
        out
    }
}

// ----- paint ---------------------------------------------------------------------------------

/// A gradient paint, aligned with the shape's bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct GradPaint {
    pub grad: Gradient,
    pub kind: GradKind,
    /// Direction in degrees (0 = left to right, counter-clockwise like Photoshop).
    pub angle: f32,
    /// 1 = the gradient spans the bounds.
    pub scale: f32,
    pub reverse: bool,
}

/// A pattern paint (one of the built-in patterns).
#[derive(Clone, Debug, PartialEq)]
pub struct PatPaint {
    pub which: usize,
    pub scale: f32,
    /// Degrees.
    pub angle: f32,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub enum Paint {
    #[default]
    None,
    Solid([u8; 3]),
    Gradient(GradPaint),
    Pattern(PatPaint),
}

impl Paint {
    pub fn is_none(&self) -> bool {
        matches!(self, Paint::None)
    }

    /// A representative colour (old file records, swatches).
    pub fn color(&self) -> Option<[u8; 3]> {
        match self {
            Paint::None => None,
            Paint::Solid(c) => Some(*c),
            Paint::Gradient(g) => g.grad.stops.first().map(|s| s.color),
            Paint::Pattern(pp) => super::retouch::patterns().get(pp.which).map(|pt| {
                let [r, g, b, _] = px::unpack(pt.px[0]);
                [r, g, b]
            }),
        }
    }
}

/// Per-pixel paint, resolved against the shape's bounds.
enum Shader {
    None,
    Solid(u32),
    Grad(Box<GradientFill>),
    Pat { which: usize, inv: Affine },
}

impl Shader {
    fn new(paint: &Paint, b: (f32, f32, f32, f32)) -> Shader {
        let (x0, y0, x1, y1) = b;
        let c = p((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        match paint {
            Paint::None => Shader::None,
            Paint::Solid(k) => Shader::Solid(px::pack(k[0], k[1], k[2], 255)),
            Paint::Gradient(g) => {
                let a = -g.angle.to_radians();
                let dir = p(a.cos(), a.sin());
                // Half the bounds' extent along the direction.
                let half = ((x1 - x0) * dir.x.abs() + (y1 - y0) * dir.y.abs()) * 0.5 * g.scale.max(0.01);
                let (p0, p1) = match g.kind {
                    GradKind::Linear => (c - dir * half, c + dir * half),
                    _ => (c, c + dir * half),
                };
                Shader::Grad(Box::new(GradientFill::new(g.grad.clone(), g.kind, (p0.x, p0.y), (p1.x, p1.y), g.reverse, true, 1.0)))
            }
            Paint::Pattern(pp) => {
                // Pattern space → document: scale, rotate, anchored at the bounds' corner.
                let m = affine_mul(&rotation_about(-pp.angle.to_radians(), p(x0, y0)), &[pp.scale.max(0.01), 0.0, x0 * (1.0 - pp.scale.max(0.01)), 0.0, pp.scale.max(0.01), y0 * (1.0 - pp.scale.max(0.01))]);
                let m = affine_mul(&m, &[1.0, 0.0, x0, 0.0, 1.0, y0]);
                Shader::Pat { which: pp.which, inv: affine_inverse(&m) }
            }
        }
    }

    fn solid(&self) -> Option<u32> {
        match self {
            Shader::Solid(c) => Some(*c),
            _ => None,
        }
    }

    #[inline]
    fn at(&self, x: i32, y: i32) -> u32 {
        match self {
            Shader::None => 0,
            Shader::Solid(c) => *c,
            Shader::Grad(g) => g.pixel(x as f32 + 0.5, y as f32 + 0.5, x, y),
            Shader::Pat { which, inv } => {
                let Some(pt) = super::retouch::patterns().get(*which) else { return 0 };
                let q = affine_apply(inv, p(x as f32 + 0.5, y as f32 + 0.5));
                let (u, v) = ((q.x.floor() as i64).rem_euclid(pt.w as i64) as usize, (q.y.floor() as i64).rem_euclid(pt.h as i64) as usize);
                pt.px[v * pt.w + u]
            }
        }
    }
}

// ----- the layer ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct ShapeLayer {
    /// The first component (its op is ignored).
    pub geom: ShapeGeom,
    pub rot: f32,
    pub xf: Affine,
    /// Further components, combined in order by their path operation.
    pub parts: Vec<ShapePart>,
    pub fill: Paint,
    pub stroke: Option<(Paint, StrokeStyle)>,
    pub rule: FillRule,
}

impl ShapeLayer {
    pub fn new(geom: ShapeGeom, fill: Paint, stroke: Option<(Paint, StrokeStyle)>) -> Self {
        ShapeLayer { geom, rot: 0.0, xf: IDENTITY, parts: vec![], fill, stroke, rule: FillRule::NonZero }
    }

    /// The first component as a part.
    pub fn first(&self) -> ShapePart {
        ShapePart { geom: self.geom.clone(), rot: self.rot, xf: self.xf, op: PathOp::Combine }
    }

    pub fn set_first(&mut self, part: ShapePart) {
        self.geom = part.geom;
        self.rot = part.rot;
        self.xf = part.xf;
    }

    /// The visible outline: the components' booleans (live), with the fill
    /// rule to rasterize it by.
    pub fn outline(&self) -> (Vec<Poly>, FillRule) {
        let tol = 0.1;
        let first = self.first().doc_path().flatten(tol);
        if self.parts.is_empty() {
            return (first, self.rule);
        }
        let mut items = vec![(BoolOp::Union, first, self.rule)];
        for pt in &self.parts {
            items.push((pt.op.bool_op(), pt.doc_path().flatten(tol), FillRule::NonZero));
        }
        (pathops::combine_sequence(&items), FillRule::NonZero)
    }

    /// Merge Shape Components: one free path of the combined outline.
    pub fn merged(&self) -> ShapeLayer {
        let (polys, _) = self.outline();
        let path = Path { subs: polys.into_iter().map(|pl| SubPath { anchors: pl.pts.into_iter().map(Anchor::corner).collect(), closed: true }).collect() };
        ShapeLayer { geom: ShapeGeom::Path(path), rot: 0.0, xf: IDENTITY, parts: vec![], rule: FillRule::NonZero, ..self.clone() }
    }

    /// The layer under a document transform (each component kept parametric
    /// where possible, see `ShapePart::transformed`).
    pub fn transformed(&self, m: &Affine) -> ShapeLayer {
        let mut out = self.clone();
        out.set_first(self.first().transformed(m));
        out.parts = self.parts.iter().map(|pt| pt.transformed(m)).collect();
        out
    }

    fn outline_bounds(polys: &[Poly]) -> Option<(f32, f32, f32, f32)> {
        let mut b = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for q in polys.iter().flat_map(|pl| pl.pts.iter()) {
            b = (b.0.min(q.x), b.1.min(q.y), b.2.max(q.x), b.3.max(q.y));
        }
        (b.0 <= b.2).then_some(b)
    }

    pub fn bounds(&self) -> RectI {
        let (polys, _) = self.outline();
        let Some((x0, y0, x1, y1)) = Self::outline_bounds(&polys) else { return RectI::EMPTY };
        let pad = self.stroke.as_ref().map_or(0.0, |(_, s)| {
            let w = if s.align == Align::Center { s.width * 0.5 } else { s.width };
            // Miters can reach miter_limit × half width.
            w * s.miter_limit.max(1.0)
        }) + 2.0;
        RectI::new((x0 - pad).floor() as i32, (y0 - pad).floor() as i32, (x1 + pad).ceil() as i32, (y1 + pad).ceil() as i32)
    }

    /// Coverage tiles of the fill and the (aligned) stroke for `region`.
    fn coverage(&self, polys: &[Poly], rule: FillRule, cols: u32, rows: u32, region: RectI) -> (Vec<(usize, MaskTile)>, Vec<(usize, MaskTile)>) {
        let fill_lines = vector::edges(polys);
        let needs_fill_cov = !self.fill.is_none() || self.stroke.as_ref().is_some_and(|(_, s)| s.align != Align::Center);
        let fill = if needs_fill_cov { vector::rasterize(&fill_lines, cols, rows, region, rule) } else { vec![] };
        let stroke = match &self.stroke {
            None => vec![],
            Some((_, st)) => {
                let mut st2 = st.clone();
                if st.align != Align::Center {
                    st2.width *= 2.0;
                }
                let sp = vector::stroke_polys(polys, &st2);
                let raw = vector::rasterize(&vector::edges(&sp), cols, rows, region, FillRule::NonZero);
                match st.align {
                    Align::Center => raw,
                    Align::Inside | Align::Outside => {
                        let fmap: std::collections::HashMap<usize, &MaskTile> = fill.iter().map(|(i, t)| (*i, t)).collect();
                        raw.into_iter()
                            .filter_map(|(i, t)| {
                                let f = fmap.get(&i).copied().cloned().unwrap_or(MaskTile::Empty);
                                let op = if st.align == Align::Inside { super::masks::SelOp::Intersect } else { super::masks::SelOp::Subtract };
                                let mut a = MaskGrid { w: TS as u32, h: TS as u32, cols: 1, rows: 1, tiles: vec![t] };
                                let b = MaskGrid { w: TS as u32, h: TS as u32, cols: 1, rows: 1, tiles: vec![f] };
                                a = a.combine(&b, op);
                                let t = a.tiles.pop().unwrap();
                                (!matches!(t, MaskTile::Empty)).then_some((i, t))
                            })
                            .collect()
                    }
                }
            }
        };
        let fill = if self.fill.is_none() { vec![] } else { fill };
        (fill, stroke)
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
        let (polys, rule) = self.outline();
        let Some(ob) = Self::outline_bounds(&polys) else { return };
        let (fill, stroke) = self.coverage(&polys, rule, grid.cols, grid.rows, region);
        let fill_sh = Shader::new(&self.fill, ob);
        let stroke_sh = self.stroke.as_ref().map_or(Shader::None, |(pt, _)| Shader::new(pt, ob));
        let solid_fill = fill_sh.solid().map(|c| Arc::new(Tile::solid(c)));
        let solid_stroke = stroke_sh.solid().map(|c| Arc::new(Tile::solid(c)));
        let mut by_tile: std::collections::HashMap<usize, (Option<MaskTile>, Option<MaskTile>)> = Default::default();
        for (i, t) in fill {
            by_tile.entry(i).or_default().0 = Some(t);
        }
        for (i, t) in stroke {
            by_tile.entry(i).or_default().1 = Some(t);
        }
        let cols = grid.cols as usize;
        let work: Vec<(usize, Option<MaskTile>, Option<MaskTile>)> = by_tile.into_iter().map(|(i, (f, s))| (i, f, s)).collect();
        let done: Vec<(usize, TileRef)> = work
            .into_par_iter()
            .map(|(i, f, s)| {
                if matches!(s, Some(MaskTile::Full)) && solid_stroke.is_some() {
                    return (i, solid_stroke.clone());
                }
                if s.is_none() && matches!(f, Some(MaskTile::Full)) && solid_fill.is_some() {
                    return (i, solid_fill.clone());
                }
                let (ox, oy) = ((i % cols * TS) as i32, (i / cols * TS) as i32);
                let mut out = vec![0u32; TS * TS].into_boxed_slice();
                let cov = |t: &Option<MaskTile>, k: usize| -> u32 {
                    match t {
                        None | Some(MaskTile::Empty) => 0,
                        Some(MaskTile::Full) => 255,
                        Some(MaskTile::Partial(m)) => m[k] as u32,
                    }
                };
                for (k, o) in out.iter_mut().enumerate() {
                    let (x, y) = (ox + (k % TS) as i32, oy + (k / TS) as i32);
                    let fc = cov(&f, k);
                    let mut v = if fc > 0 { px::scale(fill_sh.at(x, y), fc) } else { 0 };
                    let sc = cov(&s, k);
                    if sc > 0 {
                        v = px::over(px::scale(stroke_sh.at(x, y), sc), v);
                    }
                    *o = v;
                }
                (i, finish_tile(out))
            })
            .collect();
        for (i, t) in done {
            grid.tiles[i] = t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::layers::{LayerKind, rerender};
    use super::*;

    #[test]
    fn shape_layer_renders_fill_and_stroke() {
        let s = ShapeLayer::new(
            ShapeGeom::Rect { x0: 100.0, y0: 100.0, x1: 900.0, y1: 700.0, radii: [0.0; 4] },
            Paint::Solid([255, 0, 0]),
            Some((Paint::Solid([0, 0, 255]), StrokeStyle { width: 10.0, ..Default::default() })),
        );
        let mut g = TileGrid::new(1024, 1024);
        let b = g.bounds();
        s.render(&mut g, b);
        assert_eq!(g.pixel(500, 400), px::pack(255, 0, 0, 255));
        assert_eq!(g.pixel(100, 400), px::pack(0, 0, 255, 255));
        assert_eq!(g.pixel(50, 50), 0);
        // Interior tiles are shared solid tiles.
        let a = g.tiles[g.idx(1, 1)].as_ref().unwrap();
        let b = g.tiles[g.idx(2, 1)].as_ref().unwrap();
        assert!(Arc::ptr_eq(a, b));
        // Inside alignment keeps the stroke within the fill.
        let inside = ShapeLayer { stroke: Some((Paint::Solid([0, 0, 255]), StrokeStyle { width: 10.0, align: Align::Inside, ..Default::default() })), ..s.clone() };
        let mut g2 = TileGrid::new(1024, 1024);
        let b2 = g2.bounds();
        inside.render(&mut g2, b2);
        assert_eq!(g2.pixel(97, 400), 0);
        assert_eq!(g2.pixel(105, 400), px::pack(0, 0, 255, 255));
    }

    #[test]
    fn rerender_touches_only_dirty_tiles() {
        let mut s = ShapeLayer::new(ShapeGeom::Ellipse { x0: 100.0, y0: 100.0, x1: 400.0, y1: 400.0 }, Paint::Solid([0, 200, 0]), None);
        let mut g = TileGrid::new(4096, 4096);
        let k = LayerKind::Shape(Arc::new(s.clone()));
        rerender(&mut g, None, &k);
        let far_before = g.tiles[g.idx(10, 10)].clone();
        let old = k.content_bounds();
        s.geom = ShapeGeom::Ellipse { x0: 150.0, y0: 100.0, x1: 600.0, y1: 400.0 };
        let dirty = rerender(&mut g, old, &LayerKind::Shape(Arc::new(s)));
        assert!(dirty.x1 < 700 && dirty.y1 < 500);
        assert!(far_before.is_none() && g.tiles[g.idx(10, 10)].is_none());
        assert_eq!(g.pixel(550, 250), px::pack(0, 200, 0, 255));
        assert_eq!(g.pixel(120, 250), 0);
    }

    #[test]
    fn rerender_during_resize_matches_fresh_render() {
        let mk = |r: f32| {
            LayerKind::Shape(Arc::new(ShapeLayer::new(
                ShapeGeom::Polygon { x0: 50.0, y0: 60.0, x1: 50.0 + r, y1: 60.0 + r * 0.8, sides: 6, star: 0.3 },
                Paint::Solid([10, 200, 30]),
                Some((Paint::Solid([0, 0, 0]), StrokeStyle { width: 7.0, ..Default::default() })),
            )))
        };
        let mut g = TileGrid::new(1200, 1000);
        let mut old = None;
        // Grow then shrink: stale pixels outside the final shape must be cleared.
        for r in [100.0, 700.0, 1100.0, 300.0, 520.0] {
            let k = mk(r);
            rerender(&mut g, old, &k);
            old = k.content_bounds();
        }
        let mut fresh = TileGrid::new(1200, 1000);
        let b = fresh.bounds();
        mk(520.0).rasterize(&mut fresh, b);
        for y in (0..1000).step_by(2) {
            for x in 0..1200 {
                assert_eq!(g.pixel(x, y), fresh.pixel(x, y), "({x}, {y})");
            }
        }
    }

    fn ink(g: &TileGrid) -> f64 {
        let b = g.bounds();
        let mut s = 0.0;
        for y in b.y0..b.y1 {
            for x in b.x0..b.x1 {
                s += px::alpha(g.pixel(x, y)) as f64 / 255.0;
            }
        }
        s
    }

    #[test]
    fn live_path_operations_match_analytic_areas() {
        // Two discs of radius r, centres d apart: lens area 2r²acos(d/2r) − (d/2)√(4r² − d²).
        let (r, d) = (200.0f64, 160.0f64);
        let lens = 2.0 * r * r * (d / (2.0 * r)).acos() - d / 2.0 * (4.0 * r * r - d * d).sqrt();
        let disc = std::f64::consts::PI * r * r;
        let a = ShapeGeom::Ellipse { x0: 100.0, y0: 100.0, x1: 500.0, y1: 500.0 };
        let b = ShapeGeom::Ellipse { x0: 260.0, y0: 100.0, x1: 660.0, y1: 500.0 };
        for (op, want) in [(PathOp::Combine, 2.0 * disc - lens), (PathOp::Subtract, disc - lens), (PathOp::Intersect, lens), (PathOp::Exclude, 2.0 * disc - 2.0 * lens)] {
            let mut s = ShapeLayer::new(a.clone(), Paint::Solid([0, 0, 0]), None);
            s.parts.push(ShapePart::new(b.clone(), op));
            let (polys, _) = s.outline();
            let area = pathops::area(&polys);
            assert!((area - want).abs() / want < 0.001, "{op:?}: area {area:.0} vs {want:.0}");
            // Rendered coverage agrees too.
            let mut g = TileGrid::new(768, 768);
            let full = g.bounds();
            s.render(&mut g, full);
            assert!((ink(&g) - want).abs() / want < 0.002, "{op:?}: ink {:.0} vs {want:.0}", ink(&g));
            // Merging keeps the area and drops the components.
            let m = s.merged();
            assert!(m.parts.is_empty() && matches!(m.geom, ShapeGeom::Path(_)));
            assert!((pathops::area(&m.outline().0) - area).abs() < 1.0);
        }
    }

    #[test]
    fn rotation_stays_parametric_and_transforms_compose() {
        let s = ShapeLayer::new(ShapeGeom::Rect { x0: 100.0, y0: 100.0, x1: 300.0, y1: 200.0, radii: [10.0; 4] }, Paint::Solid([1, 2, 3]), None);
        // Rotate 90° about the rect centre: still a Rect, with rot set.
        let r = s.transformed(&rotation_about(std::f32::consts::FRAC_PI_2, p(200.0, 150.0)));
        assert!(matches!(r.geom, ShapeGeom::Rect { .. }) && (r.rot - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        let b = r.bounds();
        assert!((b.w() - 104).abs() <= 2 && (b.h() - 204).abs() <= 2, "{b:?}");
        // Translating moves the geometry; scaling an unrotated rect resizes it.
        let t = s.transformed(&[1.0, 0.0, 50.0, 0.0, 1.0, -20.0]);
        assert_eq!(t.geom.rect(), Some((150.0, 80.0, 350.0, 180.0)));
        let sc = s.transformed(&[2.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
        assert!(matches!(sc.geom, ShapeGeom::Rect { x0: 200.0, x1: 600.0, radii, .. } if (radii[0] - 20.0).abs() < 1e-4));
        // A skew goes into xf; the geometry stays a rect.
        let sk = s.transformed(&[1.0, 0.5, 0.0, 0.0, 1.0, 0.0]);
        assert!(matches!(sk.geom, ShapeGeom::Rect { .. }) && sk.xf != IDENTITY);
    }

    #[test]
    fn gradient_and_pattern_paints() {
        let mut g = super::super::gradient::Gradient::two("bw", [0, 0, 0], [255, 255, 255]);
        g.sort();
        let s = ShapeLayer::new(
            ShapeGeom::Rect { x0: 0.0, y0: 0.0, x1: 512.0, y1: 256.0, radii: [0.0; 4] },
            Paint::Gradient(GradPaint { grad: g, kind: GradKind::Linear, angle: 0.0, scale: 1.0, reverse: false }),
            Some((Paint::Pattern(PatPaint { which: 0, scale: 1.0, angle: 0.0 }), StrokeStyle { width: 40.0, align: Align::Inside, ..Default::default() })),
        );
        let mut t = TileGrid::new(512, 256);
        let full = t.bounds();
        s.render(&mut t, full);
        // Left to right ramp in the interior.
        let l = px::unpack(t.pixel(60, 128))[0];
        let m = px::unpack(t.pixel(256, 128))[0];
        let r = px::unpack(t.pixel(450, 128))[0];
        assert!(l < m && m < r && (m as i32 - 128).abs() < 12, "{l} {m} {r}");
        // The stroke shows the checker pattern: both of its colours appear.
        let row: Vec<u8> = (0..512).map(|x| px::unpack(t.pixel(x, 5))[0]).collect();
        assert!(row.iter().any(|&v| v > 200) && row.iter().any(|&v| v < 90));
    }

    #[test]
    fn dashed_stroke_with_offset() {
        let mk = |off: f32| {
            ShapeLayer::new(
                ShapeGeom::Line { a: (0.0, 0.0), b: (0.0, 0.0), weight: 1.0, arrow_start: None, arrow_end: None },
                Paint::None,
                Some((Paint::Solid([0, 0, 0]), StrokeStyle { width: 10.0, dashes: vec![2.0, 2.0], dash_offset: off, ..Default::default() })),
            )
        };
        // A straight open path along y = 50.
        let path = Path { subs: vec![SubPath { anchors: vec![Anchor::corner(p(20.0, 50.0)), Anchor::corner(p(420.0, 50.0))], closed: false }] };
        let mut a = mk(0.0);
        a.geom = ShapeGeom::Path(path.clone());
        let mut b = mk(2.0);
        b.geom = ShapeGeom::Path(path);
        let (mut ga, mut gb) = (TileGrid::new(512, 128), TileGrid::new(512, 128));
        let full = ga.bounds();
        a.render(&mut ga, full);
        b.render(&mut gb, full);
        // Offset by half the period swaps dashes and gaps.
        for x in [30, 50, 70, 90] {
            assert_ne!(px::alpha(ga.pixel(x, 50)) > 128, px::alpha(gb.pixel(x, 50)) > 128, "x {x}");
        }
    }
}
