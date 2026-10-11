//! Vector tools: the Pen family, Path / Direct Selection, the Paths panel
//! actions, Shape tools (with live resize) and the Frame tool.

use std::sync::Arc;
use std::time::Instant;

use gpui_kit::component::button::Button;
use gpui_kit::component::Sizable as _;
use gpui_kit::prelude::*;
use gpui_kit::{Bounds, Context, Div, MouseDownEvent, SharedString, Window, div, point, px, rgb};

use crate::engine::brush::{BrushSettings, Stroke};
use crate::engine::document::PathItem;
use crate::engine::grid::{RectI, TileGrid};
use crate::engine::layers::{self, LayerKind, Paint, PathOp, ShapeGeom, ShapeLayer, ShapePart};
use crate::engine::frame::FrameLayer;
use crate::engine::shape::{Affine, affine_apply, affine_inverse};
use crate::engine::livewire;
use crate::engine::ops;
use crate::engine::vector::{self, Align, Anchor, Cap, FillRule, Join, P, Path, StrokeStyle, SubPath, p};

use super::editor::{Drag, Editor, Overlay, Tool};
use super::options::{label, segmented, spec, toggle};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PenMode {
    #[default]
    Path,
    Shape,
}

pub struct VectorOptions {
    pub pen_mode: PenMode,
    pub fill: bool,
    pub stroke: bool,
    pub stroke_width: f32,
    pub align: Align,
    pub join: Join,
    pub cap: Cap,
    pub dashed: bool,
    pub radius: f32,
    pub sides: u32,
    pub star: f32,
    pub line_weight: f32,
    pub arrow_start: bool,
    pub arrow_end: bool,
    pub custom: usize,
    pub frame_ellipse: bool,
    /// Path operation for new shapes: `None` = a new layer, else a new
    /// component of the active shape layer.
    pub path_op: Option<PathOp>,
    /// Corner radius widgets change all corners together.
    pub radius_linked: bool,
    /// Last Properties edit (layer, time, what): slider drags coalesce undo.
    pub last_edit: Option<(u64, std::time::Instant, &'static str)>,
    /// Path Selection: the selected subpath of the work path.
    pub sel_sub: Option<usize>,
    /// Content-Aware Tracing preview.
    pub trace_preview: Vec<P>,
}

impl Default for VectorOptions {
    fn default() -> Self {
        VectorOptions {
            pen_mode: PenMode::Path,
            fill: true,
            stroke: false,
            stroke_width: 6.0,
            align: Align::Center,
            join: Join::Miter,
            cap: Cap::Butt,
            dashed: false,
            radius: 0.0,
            sides: 5,
            star: 0.0,
            line_weight: 8.0,
            arrow_start: false,
            arrow_end: true,
            custom: 0,
            frame_ellipse: false,
            path_op: None,
            radius_linked: true,
            last_edit: None,
            sel_sub: None,
            trace_preview: vec![],
        }
    }
}

/// What a vector drag is doing.
pub enum VDrag {
    /// Pulling the handles of the anchor just placed (Pen).
    PenHandle { sub: usize, idx: usize },
    Freeform(Vec<P>),
    /// Moving a whole subpath (Path Selection).
    MoveSub { sub: usize, last: P },
    /// Moving an anchor (0) or one of its handles (1 = in, 2 = out).
    Point { sub: usize, idx: usize, part: u8 },
    /// Drawing a new shape layer, or (`part`) a new component of one.
    Shape { layer_id: u64, start: (f32, f32), old: Option<RectI>, part: Option<usize> },
    /// Resizing a shape layer's first component by a box handle (0..8), in
    /// its own (unrotated) space; `m` maps that space to the document.
    Resize { layer_id: u64, handle: u8, rect: (f32, f32, f32, f32), old: Option<RectI>, m: Affine },
    /// Dragging a rectangle's corner-radius widget (`corner`, or all when linked).
    Radius { layer_id: u64, corner: usize, all: bool, old: Option<RectI>, m: Affine },
    /// Dragging a polygon's star-indent handle.
    StarIndent { layer_id: u64, old: Option<RectI>, m: Affine },
}

const HIT: f32 = 7.0;

impl Editor {
    /// The work path (creating it on demand) and its index.
    pub fn work_path(&mut self) -> usize {
        if let Some(i) = self.doc.active_path.filter(|&i| i < self.doc.paths.len()) {
            return i;
        }
        if let Some(i) = self.doc.paths.iter().position(|p| p.name == "Work Path") {
            self.doc.active_path = Some(i);
            return i;
        }
        self.doc.paths.push(PathItem { name: "Work Path".into(), path: Path::new() });
        let i = self.doc.paths.len() - 1;
        self.doc.active_path = Some(i);
        i
    }

    fn path_mut(&mut self) -> &mut Path {
        let i = self.work_path();
        &mut self.doc.paths[i].path
    }

    fn hit_tol(&self) -> f32 {
        HIT / self.vp.logical_scale(self.sf)
    }

    /// Anchor or handle under the cursor: (subpath, anchor, part).
    fn hit_point(&self, q: P) -> Option<(usize, usize, u8)> {
        let i = self.doc.active_path?;
        let path = &self.doc.paths.get(i)?.path;
        let tol = self.hit_tol();
        for (si, sp) in path.subs.iter().enumerate() {
            for (ai, a) in sp.anchors.iter().enumerate() {
                if a.p.dist(q) <= tol {
                    return Some((si, ai, 0));
                }
                if a.has_in() && a.h_in.dist(q) <= tol {
                    return Some((si, ai, 1));
                }
                if a.has_out() && a.h_out.dist(q) <= tol {
                    return Some((si, ai, 2));
                }
            }
        }
        None
    }

    /// Segment under the cursor: (subpath, segment index, t).
    fn hit_segment(&self, q: P) -> Option<(usize, usize, f32)> {
        let i = self.doc.active_path?;
        let path = &self.doc.paths.get(i)?.path;
        let tol = self.hit_tol();
        let mut best: Option<(usize, usize, f32, f32)> = None;
        for (si, sp) in path.subs.iter().enumerate() {
            for (k, c) in sp.segments().enumerate() {
                let (t, d) = vector::nearest_on_cubic(&c, q);
                if d <= tol && best.is_none_or(|b| d < b.3) {
                    best = Some((si, k, t, d));
                }
            }
        }
        best.map(|b| (b.0, b.1, b.2))
    }

    /// Mouse-down for the vector tools. Returns true when handled.
    pub fn vector_down(&mut self, ix: f32, iy: f32, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let q = p(ix, iy);
        let tool = self.tool;
        match tool {
            Tool::Pen => {
                self.doc.push_undo("Pen");
                let tol = self.hit_tol();
                let open = self.pen_sub;
                let path = self.path_mut();
                match open {
                    Some(si) if si < path.subs.len() => {
                        let sp = &mut path.subs[si];
                        if sp.anchors.len() > 1 && sp.anchors[0].p.dist(q) <= tol {
                            sp.closed = true;
                            self.pen_sub = None;
                            self.finish_pen_subpath(si, cx);
                            return true;
                        }
                        sp.anchors.push(Anchor::corner(q));
                        let idx = sp.anchors.len() - 1;
                        self.vdrag = Some(VDrag::PenHandle { sub: si, idx });
                    }
                    _ => {
                        path.subs.push(SubPath { anchors: vec![Anchor::corner(q)], closed: false });
                        let si = path.subs.len() - 1;
                        self.pen_sub = Some(si);
                        self.vdrag = Some(VDrag::PenHandle { sub: si, idx: 0 });
                    }
                }
            }
            Tool::CurvaturePen => {
                self.doc.push_undo("Curvature Pen");
                let tol = self.hit_tol();
                let open = self.pen_sub;
                let double = ev.click_count >= 2;
                let path = self.path_mut();
                let si = match open {
                    Some(si) if si < path.subs.len() => si,
                    _ => {
                        path.subs.push(SubPath { anchors: vec![], closed: false });
                        path.subs.len() - 1
                    }
                };
                let sp = &mut path.subs[si];
                if sp.anchors.len() > 2 && sp.anchors[0].p.dist(q) <= tol {
                    sp.closed = true;
                    curvature_smooth(sp);
                    self.pen_sub = None;
                    self.finish_pen_subpath(si, cx);
                    return true;
                }
                if double && !sp.anchors.is_empty() {
                    // Double-click: the last point becomes a corner.
                    sp.anchors.last_mut().unwrap().smooth = false;
                } else {
                    sp.anchors.push(Anchor { p: q, h_in: q, h_out: q, smooth: true });
                }
                curvature_smooth(sp);
                self.pen_sub = Some(si);
            }
            Tool::FreeformPen => {
                self.vdrag = Some(VDrag::Freeform(vec![q]));
            }
            Tool::ContentAwareTracing => {
                let pts = std::mem::take(&mut self.vopts.trace_preview);
                if pts.len() > 3 {
                    self.doc.push_undo("Content-Aware Tracing");
                    let cub = vector::fit_curve(&pts, 1.0);
                    let sp = vector::subpath_from_cubics(&cub, pts[0].dist(*pts.last().unwrap()) < 2.0);
                    self.path_mut().subs.push(sp);
                    self.flash(format!("Traced an edge into {} curve segments", cub.len()));
                } else {
                    self.flash("Hover an edge until it highlights, then click");
                }
            }
            Tool::AddAnchor => {
                if let Some((si, k, t)) = self.hit_segment(q) {
                    self.doc.push_undo("Add Anchor Point");
                    let path = self.path_mut();
                    let sp = &mut path.subs[si];
                    let c = sp.segments().nth(k).unwrap();
                    let (l, r) = vector::split_cubic(&c, t);
                    let n = sp.anchors.len();
                    sp.anchors[k].h_out = l[1];
                    sp.anchors[(k + 1) % n].h_in = r[2];
                    sp.anchors.insert(k + 1, Anchor { p: l[3], h_in: l[2], h_out: r[1], smooth: true });
                }
            }
            Tool::DeleteAnchor => {
                if let Some((si, ai, 0)) = self.hit_point(q) {
                    self.doc.push_undo("Delete Anchor Point");
                    let path = self.path_mut();
                    path.subs[si].anchors.remove(ai);
                    if path.subs[si].anchors.is_empty() {
                        path.subs.remove(si);
                    }
                }
            }
            Tool::ConvertPoint => {
                if let Some((si, ai, part)) = self.hit_point(q) {
                    self.doc.push_undo("Convert Point");
                    if part == 0 {
                        let path = self.path_mut();
                        let sp = &mut path.subs[si];
                        let n = sp.anchors.len();
                        let a = &mut sp.anchors[ai];
                        if a.has_in() || a.has_out() {
                            // Smooth → corner: retract both handles.
                            a.h_in = a.p;
                            a.h_out = a.p;
                            a.smooth = false;
                        } else {
                            // Corner → smooth: handles along the neighbours' chord.
                            let prev = sp.anchors[(ai + n - 1) % n].p;
                            let next = sp.anchors[(ai + 1) % n].p;
                            let a = &mut sp.anchors[ai];
                            let d = (next - prev) * (1.0 / 6.0);
                            a.h_in = a.p - d;
                            a.h_out = a.p + d;
                            a.smooth = true;
                        }
                        self.vdrag = Some(VDrag::PenHandle { sub: si, idx: ai });
                    } else {
                        self.vdrag = Some(VDrag::Point { sub: si, idx: ai, part });
                    }
                }
            }
            Tool::PathSelect => {
                // Click a path to select it; drag to move it.
                let hit = self.hit_segment(q).map(|h| h.0).or_else(|| {
                    let i = self.doc.active_path?;
                    self.doc.paths[i].path.subs.iter().position(|sp| {
                        let polys = Path { subs: vec![sp.clone()] }.flatten(0.5);
                        sp.closed && vector::contains(q, &polys, FillRule::NonZero)
                    })
                });
                self.vopts.sel_sub = hit;
                if let Some(si) = hit {
                    self.doc.push_undo("Move Path");
                    self.vdrag = Some(VDrag::MoveSub { sub: si, last: q });
                }
            }
            Tool::DirectSelect => {
                if let Some((si, ai, part)) = self.hit_point(q) {
                    self.doc.push_undo("Edit Path");
                    self.vopts.sel_sub = Some(si);
                    self.vdrag = Some(VDrag::Point { sub: si, idx: ai, part });
                }
            }
            Tool::RectShape | Tool::EllipseShape | Tool::TriangleShape | Tool::PolygonShape | Tool::LineShape | Tool::CustomShape | Tool::Frame => {
                // Grab a live widget or resize handle of the active shape
                // layer, or draw a new shape (layer or component).
                if let Some(vd) = self.shape_widget_hit(q, ev.modifiers.alt) {
                    self.doc.push_undo(match vd {
                        VDrag::Radius { .. } => "Corner Radius",
                        VDrag::StarIndent { .. } => "Star Ratio",
                        _ => "Resize Shape",
                    });
                    self.vdrag = Some(vd);
                } else {
                    let m = ev.modifiers;
                    // Photoshop: Shift adds, Alt subtracts, Shift+Alt intersects.
                    let op = match (m.shift, m.alt) {
                        (true, true) => Some(PathOp::Intersect),
                        (true, false) => Some(PathOp::Combine),
                        (false, true) => Some(PathOp::Subtract),
                        _ => self.vopts.path_op,
                    };
                    self.start_shape(ix, iy, op);
                }
            }
            _ => return false,
        }
        let _ = (window, cx);
        true
    }

    /// Ends an open Pen subpath; in Shape mode it becomes a shape layer.
    fn finish_pen_subpath(&mut self, si: usize, _cx: &mut Context<Self>) {
        if self.vopts.pen_mode != PenMode::Shape {
            return;
        }
        let i = self.work_path();
        if si >= self.doc.paths[i].path.subs.len() {
            return;
        }
        let sp = self.doc.paths[i].path.subs.remove(si);
        let shape = self.shape_style(ShapeGeom::Path(Path { subs: vec![sp] }));
        self.add_shape_layer("Shape", shape);
    }

    pub fn end_pen(&mut self, cx: &mut Context<Self>) {
        if let Some(si) = self.pen_sub.take() {
            self.finish_pen_subpath(si, cx);
        }
    }

    fn shape_style(&self, geom: ShapeGeom) -> ShapeLayer {
        let v = &self.vopts;
        let line = matches!(geom, ShapeGeom::Line { .. });
        let stroke = (v.stroke && !line).then(|| {
            (
                Paint::Solid(self.bg),
                StrokeStyle { width: v.stroke_width, join: v.join, cap: v.cap, miter_limit: 4.0, align: v.align, dashes: if v.dashed { vec![3.0, 2.0] } else { vec![] }, dash_offset: 0.0 },
            )
        });
        ShapeLayer::new(geom, if v.fill || line { Paint::Solid(self.fg) } else { Paint::None }, stroke)
    }

    fn add_shape_layer(&mut self, name: &str, shape: ShapeLayer) -> u64 {
        let (w, h) = (self.doc.w, self.doc.h);
        let at = self.doc.insert_layer(name, TileGrid::new(w, h));
        let l = std::mem::replace(&mut self.doc.layers[at], crate::engine::document::Layer::new(0, "", TileGrid::new(1, 1)));
        self.doc.layers[at] = l.with_kind(LayerKind::Shape(Arc::new(shape)));
        self.doc.mark_all_dirty();
        self.doc.layers[at].id
    }

    fn geom_for(&self, tool: Tool, a: (f32, f32), b: (f32, f32)) -> ShapeGeom {
        let v = &self.vopts;
        let (x0, y0, x1, y1) = (a.0.min(b.0), a.1.min(b.1), a.0.max(b.0), a.1.max(b.1));
        match tool {
            Tool::RectShape => ShapeGeom::Rect { x0, y0, x1, y1, radii: [v.radius; 4] },
            Tool::EllipseShape => ShapeGeom::Ellipse { x0, y0, x1, y1 },
            Tool::TriangleShape => ShapeGeom::Triangle { x0, y0, x1, y1 },
            Tool::PolygonShape => ShapeGeom::Polygon { x0, y0, x1, y1, sides: v.sides, star: v.star },
            Tool::LineShape => {
                let head = (3.0, 3.0);
                ShapeGeom::Line { a, b, weight: v.line_weight, arrow_start: v.arrow_start.then_some(head), arrow_end: v.arrow_end.then_some(head) }
            }
            Tool::CustomShape => ShapeGeom::Custom { x0, y0, x1, y1, which: v.custom },
            _ => {
                if v.frame_ellipse {
                    ShapeGeom::Ellipse { x0, y0, x1, y1 }
                } else {
                    ShapeGeom::Rect { x0, y0, x1, y1, radii: [0.0; 4] }
                }
            }
        }
    }

    fn start_shape(&mut self, ix: f32, iy: f32, op: Option<PathOp>) {
        let tool = self.tool;
        self.doc.push_undo(tool.name());
        let geom = self.geom_for(tool, (ix, iy), (ix + 1.0, iy + 1.0));
        // A new component of the active shape layer.
        if let (Some(op), false) = (op, tool == Tool::Frame) {
            let li = self.doc.active;
            if let Some(LayerKind::Shape(s)) = self.doc.layers.get(li).map(|l| &l.kind) {
                let mut ns = (**s).clone();
                ns.parts.push(ShapePart::new(geom, op));
                let idx = ns.parts.len() - 1;
                let old = self.doc.layers[li].kind.content_bounds();
                let id = self.doc.layers[li].id;
                self.doc.layers[li].kind = LayerKind::Shape(Arc::new(ns));
                self.vdrag = Some(VDrag::Shape { layer_id: id, start: (ix, iy), old, part: Some(idx) });
                return;
            }
        }
        if tool == Tool::Frame {
            // A frame layer: empty (placeholder cross) until an image is placed or dropped in.
            let (w, h) = (self.doc.w, self.doc.h);
            let n = self.doc.layers.iter().filter(|l| matches!(l.kind, LayerKind::Frame(_))).count() + 1;
            let at = self.doc.insert_layer(format!("Frame {n}"), TileGrid::new(w, h));
            let l = std::mem::replace(&mut self.doc.layers[at], crate::engine::document::Layer::new(0, "", TileGrid::new(1, 1)));
            self.doc.layers[at] = l.with_kind(LayerKind::Frame(Arc::new(FrameLayer::new(geom))));
            let id = self.doc.layers[at].id;
            self.vdrag = Some(VDrag::Shape { layer_id: id, start: (ix, iy), old: None, part: None });
            return;
        }
        let shape = self.shape_style(geom);
        let id = self.add_shape_layer(tool.name(), shape);
        let li = self.doc.layers.iter().position(|l| l.id == id).unwrap();
        let old = self.doc.layers[li].kind.content_bounds();
        self.vdrag = Some(VDrag::Shape { layer_id: id, start: (ix, iy), old, part: None });
    }

    /// Re-rasterizes a shape layer with new geometry (only dirty tiles).
    fn reshape(&mut self, id: u64, geom: ShapeGeom, old: Option<RectI>, part: Option<usize>) -> Option<RectI> {
        if part.is_none() {
            return self.reshape_first(id, old, |p| p.geom = geom);
        }
        self.reshape_with(id, old, |ns| match part {
            Some(k) => {
                if let Some(pt) = ns.parts.get_mut(k) {
                    pt.geom = geom;
                }
            }
            None => ns.geom = geom,
        })
    }

    /// Edits the first component of a shape layer, or a frame's shape.
    pub fn reshape_first(&mut self, id: u64, old: Option<RectI>, f: impl FnOnce(&mut ShapePart)) -> Option<RectI> {
        let li = self.doc.layers.iter().position(|l| l.id == id)?;
        let kind = match &self.doc.layers[li].kind {
            LayerKind::Shape(s) => {
                let mut ns = (**s).clone();
                let mut p = ns.first();
                f(&mut p);
                ns.set_first(p);
                LayerKind::Shape(Arc::new(ns))
            }
            LayerKind::Frame(fr) => {
                let mut nf = (**fr).clone();
                f(&mut nf.shape);
                LayerKind::Frame(Arc::new(nf))
            }
            _ => return None,
        };
        let t0 = Instant::now();
        let dirty = layers::rerender(&mut self.doc.layers[li].grid, old, &kind);
        self.doc.layers[li].kind = kind;
        let nb = self.doc.layers[li].kind.content_bounds();
        self.doc.layer_pixels_changed(li, &[dirty]);
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
        nb
    }

    /// Re-rasterizes a shape layer after `f` edits its description.
    pub fn reshape_with(&mut self, id: u64, old: Option<RectI>, f: impl FnOnce(&mut ShapeLayer)) -> Option<RectI> {
        let li = self.doc.layers.iter().position(|l| l.id == id)?;
        let LayerKind::Shape(s) = &self.doc.layers[li].kind else { return None };
        let mut ns = (**s).clone();
        f(&mut ns);
        let kind = LayerKind::Shape(Arc::new(ns));
        let t0 = Instant::now();
        let dirty = layers::rerender(&mut self.doc.layers[li].grid, old, &kind);
        self.doc.layers[li].kind = kind;
        let nb = self.doc.layers[li].kind.content_bounds();
        self.doc.layer_pixels_changed(li, &[dirty]);
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
        nb
    }

    /// The active shape layer's first component: id, box (in its own
    /// space) and the map from that space to the document.
    fn shape_handles(&self) -> Option<(u64, (f32, f32, f32, f32), Affine)> {
        let (id, part) = self.active_first_part()?;
        let r = part.geom.rect()?;
        Some((id, r, part.matrix()))
    }

    /// The active shape layer's first component, or the active frame's shape.
    fn active_first_part(&self) -> Option<(u64, ShapePart)> {
        let l = self.doc.layers.get(self.doc.active)?;
        match &l.kind {
            LayerKind::Shape(s) => Some((l.id, s.first())),
            LayerKind::Frame(f) => Some((l.id, f.shape.clone())),
            _ => None,
        }
    }

    /// Box handles (8), then live widgets: corner radii (Rect) or the star
    /// indent (Polygon), all in the component's own space.
    fn shape_widgets(&self) -> Option<(u64, Vec<P>, Vec<(u8, P)>, Affine)> {
        let (id, (x0, y0, x1, y1), m) = self.shape_handles()?;
        let handles = vec![p(x0, y0), p((x0 + x1) * 0.5, y0), p(x1, y0), p(x1, (y0 + y1) * 0.5), p(x1, y1), p((x0 + x1) * 0.5, y1), p(x0, y1), p(x0, (y0 + y1) * 0.5)];
        let (_, part) = self.active_first_part()?;
        let inset = 14.0 / self.vp.logical_scale(self.sf);
        let (lx0, ly0, lx1, ly1) = (x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1));
        let widgets = match &part.geom {
            ShapeGeom::Rect { radii, .. } => {
                let d = |k: usize| radii[k].max(inset).min((lx1 - lx0).min(ly1 - ly0) * 0.5);
                vec![(0, p(lx0 + d(0), ly0 + d(0))), (1, p(lx1 - d(1), ly0 + d(1))), (2, p(lx1 - d(2), ly1 - d(2))), (3, p(lx0 + d(3), ly1 - d(3)))]
            }
            ShapeGeom::Polygon { sides, star, .. } => {
                let c = p((lx0 + lx1) * 0.5, (ly0 + ly1) * 0.5);
                let a = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI / (*sides).max(3) as f32;
                let k = 1.0 - star;
                vec![(10, p(c.x + (lx1 - lx0) * 0.5 * k * a.cos(), c.y + (ly1 - ly0) * 0.5 * k * a.sin()))]
            }
            _ => vec![],
        };
        Some((id, handles, widgets, m))
    }

    /// The drag a press at `q` starts on the active shape's widgets / handles.
    fn shape_widget_hit(&self, q: P, alt: bool) -> Option<VDrag> {
        let (id, handles, widgets, m) = self.shape_widgets()?;
        let tol = self.hit_tol();
        let li = self.doc.active;
        let old = self.doc.layers[li].kind.content_bounds();
        for (k, w) in widgets {
            if affine_apply(&m, w).dist(q) <= tol {
                return Some(if k == 10 {
                    VDrag::StarIndent { layer_id: id, old, m }
                } else {
                    VDrag::Radius { layer_id: id, corner: k as usize, all: self.vopts.radius_linked && !alt, old, m }
                });
            }
        }
        let (_, rect, _) = self.shape_handles()?;
        handles.iter().position(|&h| affine_apply(&m, h).dist(q) <= tol).map(|h| VDrag::Resize { layer_id: id, handle: h as u8, rect, old, m })
    }

    pub fn vector_move(&mut self, ix: f32, iy: f32, shift: bool, alt: bool) {
        let q = p(ix, iy);
        let Some(vd) = self.vdrag.take() else {
            if self.tool == Tool::ContentAwareTracing && matches!(self.drag, Drag::None) {
                self.update_trace_preview(q);
            }
            return;
        };
        let vd = match vd {
            VDrag::PenHandle { sub, idx } => {
                let path = self.path_mut();
                if let Some(a) = path.subs.get_mut(sub).and_then(|s| s.anchors.get_mut(idx)) {
                    if a.p.dist(q) > 0.5 {
                        a.h_out = q;
                        a.h_in = a.p - (q - a.p);
                        a.smooth = true;
                    }
                }
                VDrag::PenHandle { sub, idx }
            }
            VDrag::Freeform(mut pts) => {
                if pts.last().is_none_or(|l| l.dist(q) * self.vp.logical_scale(self.sf) > 1.5) {
                    pts.push(q);
                }
                VDrag::Freeform(pts)
            }
            VDrag::MoveSub { sub, last } => {
                let d = q - last;
                let path = self.path_mut();
                if let Some(sp) = path.subs.get_mut(sub) {
                    for a in &mut sp.anchors {
                        a.p = a.p + d;
                        a.h_in = a.h_in + d;
                        a.h_out = a.h_out + d;
                    }
                }
                VDrag::MoveSub { sub, last: q }
            }
            VDrag::Point { sub, idx, part } => {
                let path = self.path_mut();
                if let Some(a) = path.subs.get_mut(sub).and_then(|s| s.anchors.get_mut(idx)) {
                    match part {
                        0 => {
                            let d = q - a.p;
                            a.p = q;
                            a.h_in = a.h_in + d;
                            a.h_out = a.h_out + d;
                        }
                        1 => {
                            a.h_in = q;
                            if a.smooth && !alt {
                                let len = (a.h_out - a.p).len();
                                a.h_out = a.p - (q - a.p).norm() * len;
                            }
                        }
                        _ => {
                            a.h_out = q;
                            if a.smooth && !alt {
                                let len = (a.h_in - a.p).len();
                                a.h_in = a.p - (q - a.p).norm() * len;
                            }
                        }
                    }
                }
                VDrag::Point { sub, idx, part }
            }
            VDrag::Shape { layer_id, start, old, part } => {
                let (mut dx, mut dy) = (ix - start.0, iy - start.1);
                if shift && self.tool != Tool::LineShape {
                    let m = dx.abs().max(dy.abs());
                    dx = m * dx.signum();
                    dy = m * dy.signum();
                }
                let (a, b) = if alt { ((start.0 - dx, start.1 - dy), (start.0 + dx, start.1 + dy)) } else { (start, (start.0 + dx, start.1 + dy)) };
                let geom = self.geom_for(self.tool, a, b);
                let nb = self.reshape(layer_id, geom, old, part);
                VDrag::Shape { layer_id, start, old: nb.or(old), part }
            }
            VDrag::Radius { layer_id, corner, all, old, m } => {
                let ql = affine_apply(&affine_inverse(&m), q);
                let nb = self.reshape_first(layer_id, old, |s| {
                    if let ShapeGeom::Rect { x0, y0, x1, y1, radii } = &mut s.geom {
                        let (lx0, ly0, lx1, ly1) = (x0.min(*x1), y0.min(*y1), x0.max(*x1), y0.max(*y1));
                        let corner_pt = [p(lx0, ly0), p(lx1, ly0), p(lx1, ly1), p(lx0, ly1)][corner];
                        // Radius: the drag's inward distance from the corner (mean of both axes).
                        let r = (((ql.x - corner_pt.x).abs() + (ql.y - corner_pt.y).abs()) * 0.5).clamp(0.0, (lx1 - lx0).min(ly1 - ly0) * 0.5);
                        if all {
                            *radii = [r; 4];
                        } else {
                            radii[corner] = r;
                        }
                    }
                });
                VDrag::Radius { layer_id, corner, all, old: nb.or(old), m }
            }
            VDrag::StarIndent { layer_id, old, m } => {
                let ql = affine_apply(&affine_inverse(&m), q);
                let nb = self.reshape_first(layer_id, old, |s| {
                    if let ShapeGeom::Polygon { x0, y0, x1, y1, star, .. } = &mut s.geom {
                        let c = p((*x0 + *x1) * 0.5, (*y0 + *y1) * 0.5);
                        let (rx, ry) = (((*x1 - *x0) * 0.5).abs().max(1e-3), ((*y1 - *y0) * 0.5).abs().max(1e-3));
                        let d = (((ql.x - c.x) / rx).powi(2) + ((ql.y - c.y) / ry).powi(2)).sqrt();
                        *star = (1.0 - d).clamp(0.0, 0.95);
                    }
                });
                VDrag::StarIndent { layer_id, old: nb.or(old), m }
            }
            VDrag::Resize { layer_id, handle, rect, old, m } => {
                let (ix, iy) = {
                    let l = affine_apply(&affine_inverse(&m), q);
                    (l.x, l.y)
                };
                let (mut x0, mut y0, mut x1, mut y1) = rect;
                match handle {
                    0 => (x0, y0) = (ix, iy),
                    1 => y0 = iy,
                    2 => (x1, y0) = (ix, iy),
                    3 => x1 = ix,
                    4 => (x1, y1) = (ix, iy),
                    5 => y1 = iy,
                    6 => (x0, y1) = (ix, iy),
                    _ => x0 = ix,
                }
                // Resize in the component's space, then shift the geometry so the
                // rotation about its new centre still maps it through `m`.
                let nb = self.reshape_first(layer_id, old, |s| {
                    let g = s.geom.with_rect(x0, y0, x1, y1);
                    let c_new = g.center();
                    s.geom = g.clone();
                    let want = affine_apply(&m, c_new);
                    let got = affine_apply(&s.matrix(), c_new);
                    s.geom = g.translated(want.x - got.x, want.y - got.y);
                });
                VDrag::Resize { layer_id, handle, rect, old: nb.or(old), m }
            }
        };
        self.vdrag = Some(vd);
    }

    pub fn vector_up(&mut self, _cx: &mut Context<Self>) {
        let Some(vd) = self.vdrag.take() else { return };
        if let VDrag::Freeform(pts) = vd {
            if pts.len() > 2 {
                self.doc.push_undo("Freeform Pen");
                let tol = (2.0 / self.vp.logical_scale(self.sf)).max(0.5);
                let cub = vector::fit_curve(&pts, tol);
                let sp = vector::subpath_from_cubics(&cub, false);
                if self.vopts.pen_mode == PenMode::Shape {
                    let shape = self.shape_style(ShapeGeom::Path(Path { subs: vec![sp] }));
                    self.add_shape_layer("Shape", shape);
                } else {
                    self.path_mut().subs.push(sp);
                }
            }
        }
    }

    fn update_trace_preview(&mut self, q: P) {
        if self.last_trace.is_some_and(|l| l.dist(q) < 2.0 / self.vp.logical_scale(self.sf)) {
            return;
        }
        self.last_trace = Some(q);
        let layer = &self.doc.layers[self.doc.active];
        let t0 = Instant::now();
        self.vopts.trace_preview = livewire::follow_edge(layer, q, 4000, 0.3);
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    // ----- Paths panel actions -------------------------------------------------------------

    pub fn path_action(&mut self, what: &str, window: &mut Window, cx: &mut Context<Self>) {
        let Some(pi) = self.doc.active_path.filter(|&i| i < self.doc.paths.len()) else {
            self.flash("No path selected");
            return;
        };
        let path = self.doc.paths[pi].path.clone();
        let (w, h) = (self.doc.w, self.doc.h);
        match what {
            "selection" => {
                let m = vector::rasterize_mask(&path.flatten(0.1), w, h, FillRule::NonZero);
                let feather = self.opts.feather;
                let m = if feather > 0.0 { m.feather(feather) } else { m };
                let combined = crate::engine::masks::combine_selection(self.doc.selection.as_ref(), m.to_selection(), self.opts.sel_op, w, h);
                self.doc.push_undo("Make Selection");
                self.doc.selection = combined;
            }
            "fill" => {
                let li = self.doc.active;
                if !self.ensure_pixels(li) {
                    return;
                }
                let m = vector::rasterize_mask(&path.flatten(0.1), w, h, FillRule::NonZero);
                let rgb = self.fg;
                let sel = self.doc.selection.clone();
                let g = ops::fill_mask(&self.doc.layers[li].grid, &m, sel.as_ref(), rgb, 1.0);
                self.doc.push_undo("Fill Path");
                self.doc.set_layer_grid(li, g);
            }
            "stroke" => self.stroke_path(&path),
            "save" => {
                self.doc.push_undo("Save Path");
                let n = self.doc.paths.iter().filter(|p| p.name.starts_with("Path ")).count() + 1;
                self.doc.paths[pi].name = format!("Path {n}");
            }
            "delete" => {
                self.doc.push_undo("Delete Path");
                self.doc.paths.remove(pi);
                self.doc.active_path = None;
                self.pen_sub = None;
            }
            "from_selection" => {
                if let Some(sel) = &self.doc.selection {
                    let mut np = Path::new();
                    for poly in sel.outlines() {
                        let pts: Vec<P> = poly.iter().map(|q| p(q[0], q[1])).collect();
                        if pts.len() > 3 {
                            let cub = vector::fit_curve(&pts, 2.0);
                            np.subs.push(vector::subpath_from_cubics(&cub, true));
                        }
                    }
                    self.doc.push_undo("Make Work Path");
                    self.doc.paths.push(PathItem { name: "Work Path".into(), path: np });
                    self.doc.active_path = Some(self.doc.paths.len() - 1);
                }
            }
            _ => {}
        }
        let _ = (window, cx);
    }

    /// Stroke Path: drives the Brush along the arc-length resampled path.
    pub fn stroke_path(&mut self, path: &Path) {
        let li = self.doc.active;
        if !self.ensure_pixels(li) {
            return;
        }
        // Like Photoshop, Stroke Path follows the path exactly: no smoothing.
        let settings = BrushSettings { smoothing: 0.0, ..*self.brush_of(Tool::Brush) };
        self.doc.push_undo("Stroke Path");
        let t0 = Instant::now();
        let mut all = Vec::new();
        for pl in path.flatten(0.25) {
            let mut st = Stroke::brush(li, &self.doc.layers[li].grid, settings, self.fg, false, self.doc.selection.clone());
            // The dab engine spaces dabs along each segment itself.
            for q in pl.pts.iter().chain(pl.closed.then(|| &pl.pts[0])) {
                all.extend(st.add_point(q.x, q.y, &mut self.doc.layers[li].grid));
            }
            all.extend(st.finish(&mut self.doc.layers[li].grid));
        }
        self.doc.layer_pixels_changed(li, &all);
        self.flash(format!("Stroke Path — {:.1} ms (length {:.0} px)", t0.elapsed().as_secs_f32() * 1000.0, path.length()));
    }

    // ----- overlays --------------------------------------------------------------------------

    pub fn vector_overlays(&self, out: &mut Vec<Overlay>) {
        let blue = gpui_kit::hsla(0.58, 0.9, 0.55, 1.0);
        let white = gpui_kit::hsla(0., 0., 1., 1.);
        let show_points = matches!(
            self.tool,
            Tool::Pen | Tool::FreeformPen | Tool::CurvaturePen | Tool::AddAnchor | Tool::DeleteAnchor | Tool::ConvertPoint | Tool::PathSelect | Tool::DirectSelect | Tool::ContentAwareTracing
        );
        if let Some(pi) = self.doc.active_path.filter(|&i| i < self.doc.paths.len()) {
            let path = &self.doc.paths[pi].path;
            let polys: Vec<(Vec<gpui_kit::Point<gpui_kit::Pixels>>, bool)> = path
                .flatten(0.5 / self.vp.logical_scale(self.sf).max(1e-3))
                .iter()
                .map(|pl| (pl.pts.iter().map(|q| self.vp.to_screen(q.x, q.y, self.canvas, self.sf)).collect(), pl.closed))
                .collect();
            out.push(Overlay::Path { polys, dashed: false, color: blue, width: 1.5 });
            if show_points {
                for (si, sp) in path.subs.iter().enumerate() {
                    let selected = self.vopts.sel_sub == Some(si) || self.pen_sub == Some(si);
                    for a in &sp.anchors {
                        let c = self.vp.to_screen(a.p.x, a.p.y, self.canvas, self.sf);
                        if selected || self.tool != Tool::PathSelect {
                            for h in [a.h_in, a.h_out] {
                                if h != a.p {
                                    let hc = self.vp.to_screen(h.x, h.y, self.canvas, self.sf);
                                    out.push(Overlay::Path { polys: vec![(vec![c, hc], false)], dashed: false, color: blue, width: 1.0 });
                                    out.push(Overlay::Quad { bounds: Bounds::from_corners(point(hc.x - px(3.), hc.y - px(3.)), point(hc.x + px(3.), hc.y + px(3.))), bg: white, border: blue, border_w: 1.0, radius: 3.0 });
                                }
                            }
                        }
                        let bg = if selected { blue } else { white };
                        out.push(Overlay::Quad { bounds: Bounds::from_corners(point(c.x - px(3.5), c.y - px(3.5)), point(c.x + px(3.5), c.y + px(3.5))), bg, border: blue, border_w: 1.0, radius: 0.0 });
                    }
                }
            }
        }
        // Pen rubber band from the last anchor to the cursor.
        if let (Tool::Pen, Some(si), Some(m)) = (self.tool, self.pen_sub, self.mouse_img) {
            if let Some(a) = self.doc.active_path.and_then(|pi| self.doc.paths.get(pi)).and_then(|p| p.path.subs.get(si)).and_then(|s| s.anchors.last()) {
                let c = [a.p, a.h_out, p(m.0, m.1), p(m.0, m.1)];
                let pts: Vec<_> = (0..=24).map(|k| {
                    let q = vector::cubic_at(&c, k as f32 / 24.0);
                    self.vp.to_screen(q.x, q.y, self.canvas, self.sf)
                }).collect();
                out.push(Overlay::Path { polys: vec![(pts, false)], dashed: true, color: blue, width: 1.0 });
            }
        }
        if let Some(VDrag::Freeform(pts)) = &self.vdrag {
            let sp: Vec<_> = pts.iter().map(|q| self.vp.to_screen(q.x, q.y, self.canvas, self.sf)).collect();
            out.push(Overlay::Path { polys: vec![(sp, false)], dashed: false, color: blue, width: 1.5 });
        }
        if self.tool == Tool::ContentAwareTracing && self.vopts.trace_preview.len() > 1 {
            let sp: Vec<_> = self.vopts.trace_preview.iter().map(|q| self.vp.to_screen(q.x, q.y, self.canvas, self.sf)).collect();
            out.push(Overlay::Path { polys: vec![(sp, false)], dashed: false, color: gpui_kit::hsla(0.33, 1.0, 0.5, 1.0), width: 2.0 });
        }
        // Shape resize handles.
        let shape_tool = matches!(self.tool, Tool::RectShape | Tool::EllipseShape | Tool::TriangleShape | Tool::PolygonShape | Tool::LineShape | Tool::CustomShape | Tool::Frame);
        if shape_tool || self.tool == Tool::PathSelect {
            if let Some((_, handles, widgets, m)) = self.shape_widgets() {
                let sc = |q: P| {
                    let d = affine_apply(&m, q);
                    self.vp.to_screen(d.x, d.y, self.canvas, self.sf)
                };
                let corners = [handles[0], handles[2], handles[4], handles[6]].map(sc);
                out.push(Overlay::Path { polys: vec![(corners.to_vec(), true)], dashed: false, color: blue, width: 1.0 });
                for h in &handles {
                    let c = sc(*h);
                    out.push(Overlay::Quad { bounds: Bounds::from_corners(point(c.x - px(4.), c.y - px(4.)), point(c.x + px(4.), c.y + px(4.))), bg: white, border: blue, border_w: 1.0, radius: 0.0 });
                }
                // Live widgets: round dots (corner radius / star ratio).
                for (_, w) in &widgets {
                    let c = sc(*w);
                    out.push(Overlay::Quad { bounds: Bounds::from_corners(point(c.x - px(5.), c.y - px(5.)), point(c.x + px(5.), c.y + px(5.))), bg: white, border: blue, border_w: 1.5, radius: 5.0 });
                }
            }
            // Outlines of the other components, so live booleans stay visible.
            if let Some(LayerKind::Shape(s)) = self.doc.layers.get(self.doc.active).map(|l| &l.kind) {
                for pt in &s.parts {
                    let polys: Vec<_> = pt.doc_path().flatten(0.5).iter().map(|pl| (pl.pts.iter().map(|q| self.vp.to_screen(q.x, q.y, self.canvas, self.sf)).collect(), pl.closed)).collect();
                    out.push(Overlay::Path { polys, dashed: true, color: blue, width: 1.0 });
                }
            }
        }
        // Frames: their outline, and Photoshop's placeholder cross while empty.
        let gray = gpui_kit::hsla(0., 0., 0.45, 0.9);
        for l in self.doc.layers.iter().filter(|l| l.visible) {
            let LayerKind::Frame(fr) = &l.kind else { continue };
            let polys: Vec<_> = fr.outline().iter().map(|pl| (pl.pts.iter().map(|q| self.vp.to_screen(q.x, q.y, self.canvas, self.sf)).collect(), true)).collect();
            out.push(Overlay::Path { polys, dashed: false, color: gray, width: 1.0 });
            if fr.content.is_none() {
                if let Some((x0, y0, x1, y1)) = fr.shape.geom.rect() {
                    let m = fr.shape.matrix();
                    let c = [p(x0, y0), p(x1, y1), p(x1, y0), p(x0, y1)].map(|q| {
                        let d = affine_apply(&m, q);
                        self.vp.to_screen(d.x, d.y, self.canvas, self.sf)
                    });
                    out.push(Overlay::Path { polys: vec![(vec![c[0], c[1]], false), (vec![c[2], c[3]], false)], dashed: false, color: gray, width: 1.0 });
                }
            }
        }
    }

    /// Merge Shape Components: the active shape layer's live booleans become
    /// one path.
    pub fn merge_shape_components(&mut self) {
        let li = self.doc.active;
        let Some(LayerKind::Shape(s)) = self.doc.layers.get(li).map(|l| &l.kind) else { return };
        if s.parts.is_empty() {
            return;
        }
        let t0 = Instant::now();
        let merged = s.merged();
        self.doc.push_undo("Merge Shape Components");
        let old = self.doc.layers[li].kind.content_bounds();
        let kind = LayerKind::Shape(Arc::new(merged));
        let dirty = layers::rerender(&mut self.doc.layers[li].grid, old, &kind);
        self.doc.layers[li].kind = kind;
        self.doc.layer_pixels_changed(li, &[dirty]);
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    // ----- options bar ---------------------------------------------------------------------

    pub fn vector_options(&mut self, mut bar: Div, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let t = self.tool;
        let pen = matches!(t, Tool::Pen | Tool::FreeformPen | Tool::CurvaturePen);
        let shape = matches!(t, Tool::RectShape | Tool::EllipseShape | Tool::TriangleShape | Tool::PolygonShape | Tool::LineShape | Tool::CustomShape);
        if pen {
            bar = bar.child(segmented("penmode", &[(PenMode::Path, "Path"), (PenMode::Shape, "Shape")], self.vopts.pen_mode, cx, |e, v| e.vopts.pen_mode = v));
        }
        if shape || (pen && self.vopts.pen_mode == PenMode::Shape) {
            bar = bar
                .child(toggle("v-fill", "Fill (FG)", self.vopts.fill).on_click(cx.listener(|e, _, _, cx| {
                    e.vopts.fill = !e.vopts.fill;
                    cx.notify();
                })))
                .child(toggle("v-stroke", "Stroke (BG)", self.vopts.stroke).on_click(cx.listener(|e, _, _, cx| {
                    e.vopts.stroke = !e.vopts.stroke;
                    cx.notify();
                })));
            if self.vopts.stroke {
                bar = bar
                    .child(self.opt_slider("Width", "v.sw", spec(0.5, 200.0, 0.5, |e| e.vopts.stroke_width, |e, v| e.vopts.stroke_width = v), |v| format!("{v:.1} px"), 70., window, cx))
                    .child(segmented("v-align", &[(Align::Inside, "Inside"), (Align::Center, "Center"), (Align::Outside, "Outside")], self.vopts.align, cx, |e, v| e.vopts.align = v))
                    .child(segmented("v-join", &[(Join::Miter, "Miter"), (Join::Round, "Round"), (Join::Bevel, "Bevel")], self.vopts.join, cx, |e, v| e.vopts.join = v))
                    .child(toggle("v-dash", "Dashed", self.vopts.dashed).on_click(cx.listener(|e, _, _, cx| {
                        e.vopts.dashed = !e.vopts.dashed;
                        cx.notify();
                    })));
            }
        }
        match t {
            Tool::RectShape => bar = bar.child(self.opt_slider("Radius", "v.r", spec(0.0, 500.0, 1.0, |e| e.vopts.radius, |e, v| e.vopts.radius = v), |v| format!("{v:.0} px"), 80., window, cx)),
            Tool::PolygonShape => {
                bar = bar
                    .child(self.opt_slider("Sides", "v.sides", spec(3.0, 32.0, 1.0, |e| e.vopts.sides as f32, |e, v| e.vopts.sides = v as u32), |v| format!("{v:.0}"), 60., window, cx))
                    .child(self.opt_slider("Star", "v.star", spec(0.0, 95.0, 1.0, |e| e.vopts.star * 100.0, |e, v| e.vopts.star = v / 100.0), |v| format!("{v:.0}%"), 60., window, cx));
            }
            Tool::LineShape => {
                bar = bar
                    .child(self.opt_slider("Weight", "v.lw", spec(1.0, 200.0, 1.0, |e| e.vopts.line_weight, |e, v| e.vopts.line_weight = v), |v| format!("{v:.0} px"), 70., window, cx))
                    .child(toggle("v-as", "Arrow start", self.vopts.arrow_start).on_click(cx.listener(|e, _, _, cx| {
                        e.vopts.arrow_start = !e.vopts.arrow_start;
                        cx.notify();
                    })))
                    .child(toggle("v-ae", "Arrow end", self.vopts.arrow_end).on_click(cx.listener(|e, _, _, cx| {
                        e.vopts.arrow_end = !e.vopts.arrow_end;
                        cx.notify();
                    })));
            }
            Tool::CustomShape => {
                let items: Vec<(usize, &'static str)> = crate::engine::shapes::CUSTOM_SHAPES.iter().enumerate().map(|(i, n)| (i, *n)).collect();
                bar = bar.child(segmented("v-custom", &items, self.vopts.custom, cx, |e, v| e.vopts.custom = v));
            }
            Tool::Frame => {
                bar = bar
                    .child(segmented("frame", &[(false, "Rectangle"), (true, "Ellipse")], self.vopts.frame_ellipse, cx, |e, v| e.vopts.frame_ellipse = v))
                    .child(label("Place an image (File ▸ Place) into the active frame, or drop an image file onto a frame"));
            }
            _ => {}
        }
        if shape {
            let ops: Vec<(Option<PathOp>, &'static str)> = std::iter::once((None, "New Layer")).chain(PathOp::ALL.iter().map(|o| (Some(*o), o.label()))).collect();
            bar = bar.child(segmented("v-pathop", &ops, self.vopts.path_op, cx, |e, v| e.vopts.path_op = v));
            let has_parts = matches!(self.doc.layers.get(self.doc.active).map(|l| &l.kind), Some(LayerKind::Shape(s)) if !s.parts.is_empty());
            if has_parts {
                bar = bar.child(Button::new("v-merge").label("Merge Shape Components").xsmall().on_click(cx.listener(|e, _, _, cx| {
                    e.merge_shape_components();
                    cx.notify();
                })));
            }
            bar = bar.child(label(if self.shape_handles().is_some() { format!("Live: {:.1} ms/frame", self.last_tool_ms) } else { String::new() }));
        }
        bar
    }

    /// The Paths panel (side panel section).
    pub fn render_paths_panel(&mut self, cx: &mut Context<Self>) -> Div {
        let mut list = div().flex().flex_col().p_1();
        for (i, item) in self.doc.paths.iter().enumerate() {
            let active = self.doc.active_path == Some(i);
            let n: usize = item.path.subs.iter().map(|s| s.anchors.len()).sum();
            list = list.child(
                div()
                    .id(SharedString::from(format!("path-{i}")))
                    .px_2()
                    .py_0p5()
                    .rounded_sm()
                    .when(active, |d| d.bg(rgb(0x2f6fde).opacity(0.35)))
                    .text_color(rgb(0xdcdcdc))
                    .child(format!("{} · {} anchors", item.name, n))
                    .on_click(cx.listener(move |e, _, _, cx| {
                        e.doc.active_path = Some(i);
                        cx.notify();
                    })),
            );
        }
        let b = |id: &'static str, l: &'static str, what: &'static str, cx: &mut Context<Editor>| {
            Button::new(id).label(l).xsmall().on_click(cx.listener(move |e, _, w, cx| {
                e.path_action(what, w, cx);
                cx.notify();
            }))
        };
        div()
            .flex()
            .flex_col()
            .border_t_1()
            .border_color(rgb(0x3a3a3d))
            .child(super::render::section_title("Paths"))
            .child(list)
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_1()
                    .p_1()
                    .child(b("p-fill", "Fill", "fill", cx))
                    .child(b("p-stroke", "Stroke", "stroke", cx))
                    .child(b("p-sel", "Selection", "selection", cx))
                    .child(b("p-from", "From Selection", "from_selection", cx))
                    .child(b("p-save", "Save", "save", cx))
                    .child(b("p-del", "Delete", "delete", cx)),
            )
    }
}

/// Curvature Pen: every smooth anchor gets Catmull-Rom tangents so the curve
/// passes smoothly through all clicked points.
pub fn curvature_smooth(sp: &mut SubPath) {
    let n = sp.anchors.len();
    if n < 2 {
        return;
    }
    let pts: Vec<P> = sp.anchors.iter().map(|a| a.p).collect();
    for i in 0..n {
        let a = &mut sp.anchors[i];
        if !a.smooth {
            a.h_in = a.p;
            a.h_out = a.p;
            continue;
        }
        let prev = if i > 0 { pts[i - 1] } else if sp.closed { pts[n - 1] } else { pts[i] };
        let next = if i + 1 < n { pts[i + 1] } else if sp.closed { pts[0] } else { pts[i] };
        let d = (next - prev) * (1.0 / 6.0);
        a.h_in = a.p - d;
        a.h_out = a.p + d;
    }
}
