//! The Properties panel for shape layers: transform (W, H, X, Y, angle —
//! the angle is stored per component so shapes stay parametric), fill and
//! stroke paint (none / solid / gradient / pattern), stroke width,
//! alignment, caps, joins, dashes and dash offset, corner radii (linked or
//! per corner), polygon sides and star ratio, and the live path operation
//! of every component.

use std::sync::Arc;
use std::time::Instant;

use gpui_kit::component::button::Button;
use gpui_kit::component::Sizable as _;
use gpui_kit::prelude::*;
use gpui_kit::{Context, Div, Window, div, rgb};

use crate::engine::frame::FrameLayer;
use crate::engine::gradient::{GradKind, Gradient};
use crate::engine::layers::{self, GradPaint, LayerKind, Paint, PatPaint, PathOp, ShapeGeom, ShapeLayer};
use crate::engine::vector::{Align, Cap, Join, StrokeStyle};

use super::editor::{Editor, Tool};
use super::options::{label, segmented, spec, toggle};

/// Dash presets (in stroke widths), as Photoshop's stroke options.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DashKind {
    Solid,
    Dashed,
    Dotted,
    Custom,
}

impl DashKind {
    fn of(d: &[f32]) -> DashKind {
        match d {
            [] => DashKind::Solid,
            [a, b] if *a == 4.0 && *b == 2.0 => DashKind::Dashed,
            [a, b] if *a == 0.0 && *b == 2.0 => DashKind::Dotted,
            _ => DashKind::Custom,
        }
    }
}

fn paint_kind(p: &Paint) -> u8 {
    match p {
        Paint::None => 0,
        Paint::Solid(_) => 1,
        Paint::Gradient(_) => 2,
        Paint::Pattern(_) => 3,
    }
}

impl Editor {
    pub fn active_shape(&self) -> Option<(usize, Arc<ShapeLayer>)> {
        let li = self.doc.active;
        match &self.doc.layers.get(li)?.kind {
            LayerKind::Shape(s) => Some((li, s.clone())),
            _ => None,
        }
    }

    pub fn active_frame(&self) -> Option<(usize, Arc<FrameLayer>)> {
        let li = self.doc.active;
        match &self.doc.layers.get(li)?.kind {
            LayerKind::Frame(f) => Some((li, f.clone())),
            _ => None,
        }
    }

    pub fn shape_panel_visible(&self) -> bool {
        (self.active_shape().is_some() || self.active_frame().is_some())
            && matches!(self.tool, Tool::Frame | Tool::RectShape | Tool::EllipseShape | Tool::TriangleShape | Tool::PolygonShape | Tool::LineShape | Tool::CustomShape | Tool::PathSelect | Tool::DirectSelect | Tool::Move)
    }

    /// Edits the active shape layer (consecutive edits of the same property
    /// within 1.5 s share one undo step, so slider drags are one step).
    pub fn shape_edit(&mut self, what: &'static str, f: impl FnOnce(&mut ShapeLayer)) {
        let Some((li, s)) = self.active_shape() else { return };
        let id = self.doc.layers[li].id;
        let burst = self.vopts.last_edit.is_some_and(|(l, at, w)| l == id && w == what && at.elapsed().as_secs_f32() < 1.5) && self.doc.undo_label() == Some(what);
        if !burst {
            self.doc.push_undo(what);
        }
        self.vopts.last_edit = Some((id, Instant::now(), what));
        let t0 = Instant::now();
        let mut ns = (*s).clone();
        f(&mut ns);
        let old = s.bounds();
        let kind = LayerKind::Shape(Arc::new(ns));
        let dirty = layers::rerender(&mut self.doc.layers[li].grid, Some(old), &kind);
        self.doc.layers[li].kind = kind;
        self.doc.layer_pixels_changed(li, &[dirty]);
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    fn shape_rect(&self) -> (f32, f32, f32, f32) {
        self.active_shape().and_then(|(_, s)| s.geom.rect()).map_or((0.0, 0.0, 0.0, 0.0), |(x0, y0, x1, y1)| (x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)))
    }

    /// Resizes the first component about its centre (W / H fields).
    fn set_shape_size(&mut self, w: Option<f32>, h: Option<f32>) {
        self.shape_edit("Shape Size", |s| {
            if let Some((x0, y0, x1, y1)) = s.geom.rect() {
                let (cx, cy) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
                let (hw, hh) = (w.map_or((x1 - x0).abs() * 0.5, |v| v.max(1.0) * 0.5), h.map_or((y1 - y0).abs() * 0.5, |v| v.max(1.0) * 0.5));
                s.geom = s.geom.with_rect(cx - hw, cy - hh, cx + hw, cy + hh);
            }
        });
    }

    /// Moves / scales the active frame's content (scale about the frame's centre).
    fn frame_content_edit(&mut self, f: impl FnOnce(&FrameLayer, [f32; 9]) -> [f32; 9]) {
        let Some((li, fr)) = self.active_frame() else { return };
        let Some(c) = &fr.content else { return };
        let id = self.doc.layers[li].id;
        let what = "Frame Content";
        let burst = self.vopts.last_edit.is_some_and(|(l, at, w)| l == id && w == what && at.elapsed().as_secs_f32() < 1.5) && self.doc.undo_label() == Some(what);
        if !burst {
            self.doc.push_undo(what);
        }
        self.vopts.last_edit = Some((id, Instant::now(), what));
        let m = f(&fr, c.m);
        let nf = fr.with_content_matrix(m);
        self.set_frame(li, nf, fr.bounds());
    }

    fn render_frame_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let Some((_, fr)) = self.active_frame() else { return div() };
        let mut col = div().flex().flex_col().gap_1p5().p_2().border_b_1().border_color(rgb(0x3a3a3d)).child(super::render::section_title("Frame"));
        match &fr.content {
            None => col = col.child(label("Empty: File ▸ Place, or drop an image file onto the frame")),
            Some(c) => {
                col = col
                    .child(label(format!("Content: {} ({} × {})", c.name, c.src.w, c.src.h)))
                    .child(self.opt_slider("Scale", "fp.k", spec(1.0, 800.0, 0.5, |e| e.active_frame().and_then(|(_, f)| f.content_placement()).map_or(100.0, |p| p.0 * 100.0), |e, v| e.frame_content_edit(|f, m| {
                        let k0 = (m[0] * m[4] - m[1] * m[3]).abs().sqrt().max(1e-6);
                        let r = v / 100.0 / k0;
                        let (x0, y0, x1, y1) = f.frame_box().unwrap_or((0.0, 0.0, 0.0, 0.0));
                        let (cx_, cy) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
                        [m[0] * r, m[1] * r, cx_ + (m[2] - cx_) * r, m[3] * r, m[4] * r, cy + (m[5] - cy) * r, 0.0, 0.0, 1.0]
                    })), |v| format!("{v:.1}%"), 120., window, cx))
                    .child(self.opt_slider("X", "fp.x", spec(-30000.0, 30000.0, 1.0, |e| e.active_frame().and_then(|(_, f)| f.content_placement()).map_or(0.0, |p| p.1), |e, v| e.frame_content_edit(|_, mut m| {
                        m[2] = v;
                        m
                    })), |v| format!("{v:.0} px"), 120., window, cx))
                    .child(self.opt_slider("Y", "fp.y", spec(-30000.0, 30000.0, 1.0, |e| e.active_frame().and_then(|(_, f)| f.content_placement()).map_or(0.0, |p| p.2), |e, v| e.frame_content_edit(|_, mut m| {
                        m[5] = v;
                        m
                    })), |v| format!("{v:.0} px"), 120., window, cx))
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .child(Button::new("fp-fill").label("Fill Frame").xsmall().on_click(cx.listener(|e, _, _, cx| {
                                e.frame_content_edit(|f, m| f.content.as_ref().map_or(m, |c| f.fitted(c, true).m));
                                cx.notify();
                            })))
                            .child(Button::new("fp-fit").label("Fit Frame").xsmall().on_click(cx.listener(|e, _, _, cx| {
                                e.frame_content_edit(|f, m| f.content.as_ref().map_or(m, |c| f.fitted(c, false).m));
                                cx.notify();
                            })))
                            .child(Button::new("fp-clear").label("Clear Content").xsmall().on_click(cx.listener(|e, _, _, cx| {
                                if let Some((li, fr)) = e.active_frame() {
                                    e.doc.push_undo("Clear Frame");
                                    let nf = FrameLayer { shape: fr.shape.clone(), content: None };
                                    e.set_frame(li, nf, fr.bounds());
                                }
                                cx.notify();
                            }))),
                    );
            }
        }
        col
    }

    pub fn render_shape_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        if self.active_frame().is_some() {
            return self.render_frame_panel(window, cx);
        }
        let Some((_, s)) = self.active_shape() else { return div() };
        let mut col = div().flex().flex_col().gap_1p5().p_2().border_b_1().border_color(rgb(0x3a3a3d)).child(super::render::section_title("Properties"));
        // Components and their live path operations.
        if !s.parts.is_empty() {
            col = col.child(label(format!("Components ({}), combined live", s.parts.len() + 1)));
            for (k, pt) in s.parts.iter().enumerate() {
                let items: Vec<((usize, PathOp), &'static str)> = PathOp::ALL.iter().map(|o| ((k, *o), o.label())).collect();
                let id: &'static str = ["sp-op0", "sp-op1", "sp-op2", "sp-op3", "sp-op4", "sp-op5", "sp-op6", "sp-op7"][k.min(7)];
                col = col.child(div().flex().items_center().gap_1().child(label(format!("{}", k + 2))).child(segmented(id, &items, (k, pt.op), cx, |e, (k, op)| e.shape_edit("Path Operation", |s| {
                    if let Some(p) = s.parts.get_mut(k) {
                        p.op = op;
                    }
                }))));
            }
            col = col.child(Button::new("sp-merge").label("Merge Shape Components").xsmall().on_click(cx.listener(|e, _, _, cx| {
                e.merge_shape_components();
                cx.notify();
            })));
        }
        // Transform.
        if s.geom.rect().is_some() {
            col = col
                .child(self.opt_slider("W", "sp.w", spec(1.0, 30000.0, 1.0, |e| { let r = e.shape_rect(); r.2 - r.0 }, |e, v| e.set_shape_size(Some(v), None)), |v| format!("{v:.0} px"), 120., window, cx))
                .child(self.opt_slider("H", "sp.h", spec(1.0, 30000.0, 1.0, |e| { let r = e.shape_rect(); r.3 - r.1 }, |e, v| e.set_shape_size(None, Some(v))), |v| format!("{v:.0} px"), 120., window, cx))
                .child(self.opt_slider("X", "sp.x", spec(-30000.0, 30000.0, 1.0, |e| e.shape_rect().0, |e, v| {
                    let dx = v - e.shape_rect().0;
                    e.shape_edit("Move Shape", |s| s.geom = s.geom.translated(dx, 0.0));
                }), |v| format!("{v:.0} px"), 120., window, cx))
                .child(self.opt_slider("Y", "sp.y", spec(-30000.0, 30000.0, 1.0, |e| e.shape_rect().1, |e, v| {
                    let dy = v - e.shape_rect().1;
                    e.shape_edit("Move Shape", |s| s.geom = s.geom.translated(0.0, dy));
                }), |v| format!("{v:.0} px"), 120., window, cx));
        }
        col = col.child(self.opt_slider("Angle", "sp.rot", spec(-180.0, 180.0, 0.5, |e| e.active_shape().map_or(0.0, |(_, s)| -s.rot.to_degrees() + 0.0), |e, v| e.shape_edit("Rotate Shape", |s| s.rot = -v.to_radians())), |v| format!("{:.1}°", (v * 10.0).round() / 10.0 + 0.0), 120., window, cx));
        // Fill.
        col = col.child(label("Fill")).child(segmented("sp-fill", &[(0u8, "None"), (1, "Solid"), (2, "Gradient"), (3, "Pattern")], paint_kind(&s.fill), cx, |e, v| {
            let p = e.new_paint(v);
            e.shape_edit("Fill", |s| s.fill = p);
        }));
        col = self.paint_controls(col, &s.fill, false, window, cx);
        // Stroke.
        let stroke_kind = s.stroke.as_ref().map_or(0, |(p, _)| paint_kind(p));
        col = col.child(label("Stroke")).child(segmented("sp-stroke", &[(0u8, "None"), (1, "Solid"), (2, "Gradient"), (3, "Pattern")], stroke_kind, cx, |e, v| {
            let p = e.new_paint(v);
            e.shape_edit("Stroke", |s| {
                if v == 0 {
                    s.stroke = None;
                } else {
                    let st = s.stroke.take().map_or_else(|| StrokeStyle { width: 6.0, ..Default::default() }, |(_, st)| st);
                    s.stroke = Some((p, st));
                }
            });
        }));
        if let Some((paint, st)) = &s.stroke {
            col = self.paint_controls(col, paint, true, window, cx);
            let dk = DashKind::of(&st.dashes);
            col = col
                .child(self.opt_slider("Width", "sp.sw", spec(0.5, 300.0, 0.5, |e| e.active_shape().and_then(|(_, s)| s.stroke.as_ref().map(|(_, st)| st.width)).unwrap_or(1.0), |e, v| e.shape_edit("Stroke Width", |s| {
                    if let Some((_, st)) = &mut s.stroke {
                        st.width = v;
                    }
                })), |v| format!("{v:.1} px"), 120., window, cx))
                .child(segmented("sp-align", &[(Align::Inside, "Inside"), (Align::Center, "Center"), (Align::Outside, "Outside")], st.align, cx, |e, v| e.shape_edit("Stroke Align", |s| {
                    if let Some((_, st)) = &mut s.stroke {
                        st.align = v;
                    }
                })))
                .child(segmented("sp-cap", &[(Cap::Butt, "Butt"), (Cap::Round, "Round"), (Cap::Square, "Square")], st.cap, cx, |e, v| e.shape_edit("Stroke Cap", |s| {
                    if let Some((_, st)) = &mut s.stroke {
                        st.cap = v;
                    }
                })))
                .child(segmented("sp-join", &[(Join::Miter, "Miter"), (Join::Round, "Round"), (Join::Bevel, "Bevel")], st.join, cx, |e, v| e.shape_edit("Stroke Join", |s| {
                    if let Some((_, st)) = &mut s.stroke {
                        st.join = v;
                    }
                })))
                .child(segmented("sp-dash", &[(DashKind::Solid, "Solid"), (DashKind::Dashed, "Dashed"), (DashKind::Dotted, "Dotted"), (DashKind::Custom, "Custom")], dk, cx, |e, v| e.shape_edit("Stroke Dashes", |s| {
                    if let Some((_, st)) = &mut s.stroke {
                        st.dashes = match v {
                            DashKind::Solid => vec![],
                            DashKind::Dashed => vec![4.0, 2.0],
                            DashKind::Dotted => {
                                st.cap = Cap::Round;
                                vec![0.0, 2.0]
                            }
                            DashKind::Custom => vec![3.0, 1.5],
                        };
                    }
                })));
            if dk == DashKind::Custom {
                col = col
                    .child(self.opt_slider("Dash", "sp.dash", spec(0.0, 20.0, 0.1, |e| e.stroke_dash(0), |e, v| e.set_stroke_dash(0, v)), |v| format!("{v:.1}×"), 110., window, cx))
                    .child(self.opt_slider("Gap", "sp.gap", spec(0.1, 20.0, 0.1, |e| e.stroke_dash(1), |e, v| e.set_stroke_dash(1, v)), |v| format!("{v:.1}×"), 110., window, cx));
            }
            if dk != DashKind::Solid {
                col = col.child(self.opt_slider("Offset", "sp.doff", spec(0.0, 40.0, 0.1, |e| e.active_shape().and_then(|(_, s)| s.stroke.as_ref().map(|(_, st)| st.dash_offset)).unwrap_or(0.0), |e, v| e.shape_edit("Dash Offset", |s| {
                    if let Some((_, st)) = &mut s.stroke {
                        st.dash_offset = v;
                    }
                })), |v| format!("{v:.1}×"), 110., window, cx));
            }
        }
        // Geometry-specific controls.
        match &s.geom {
            ShapeGeom::Rect { radii, .. } => {
                let linked = self.vopts.radius_linked;
                col = col.child(div().flex().items_center().gap_1().child(label("Corners")).child(toggle("sp-link", "Linked", linked).on_click(cx.listener(|e, _, _, cx| {
                    e.vopts.radius_linked = !e.vopts.radius_linked;
                    cx.notify();
                }))));
                if linked {
                    col = col.child(self.opt_slider("Radius", "sp.r", spec(0.0, 2000.0, 1.0, |e| e.corner_radius(0), |e, v| e.shape_edit("Corner Radius", |s| {
                        if let ShapeGeom::Rect { radii, .. } = &mut s.geom {
                            *radii = [v; 4];
                        }
                    })), |v| format!("{v:.0} px"), 110., window, cx));
                } else {
                    let _ = radii;
                    col = col
                        .child(self.opt_slider("Top left", "sp.r0", spec(0.0, 2000.0, 1.0, |e| e.corner_radius(0), |e, v| e.set_corner(0, v)), |v| format!("{v:.0} px"), 100., window, cx))
                        .child(self.opt_slider("Top right", "sp.r1", spec(0.0, 2000.0, 1.0, |e| e.corner_radius(1), |e, v| e.set_corner(1, v)), |v| format!("{v:.0} px"), 100., window, cx))
                        .child(self.opt_slider("Bottom right", "sp.r2", spec(0.0, 2000.0, 1.0, |e| e.corner_radius(2), |e, v| e.set_corner(2, v)), |v| format!("{v:.0} px"), 100., window, cx))
                        .child(self.opt_slider("Bottom left", "sp.r3", spec(0.0, 2000.0, 1.0, |e| e.corner_radius(3), |e, v| e.set_corner(3, v)), |v| format!("{v:.0} px"), 100., window, cx));
                }
            }
            ShapeGeom::Polygon { .. } => {
                col = col
                    .child(self.opt_slider("Sides", "sp.sides", spec(3.0, 64.0, 1.0, |e| e.poly_param().0, |e, v| e.shape_edit("Polygon Sides", |s| {
                        if let ShapeGeom::Polygon { sides, .. } = &mut s.geom {
                            *sides = v as u32;
                        }
                    })), |v| format!("{v:.0}"), 110., window, cx))
                    .child(self.opt_slider("Star ratio", "sp.star", spec(0.0, 95.0, 1.0, |e| e.poly_param().1 * 100.0, |e, v| e.shape_edit("Star Ratio", |s| {
                        if let ShapeGeom::Polygon { star, .. } = &mut s.geom {
                            *star = v / 100.0;
                        }
                    })), |v| format!("{v:.0}%"), 110., window, cx));
            }
            _ => {}
        }
        col
    }

    /// Gradient / pattern / solid sub-controls of a paint.
    fn paint_controls(&mut self, mut col: Div, paint: &Paint, stroke: bool, window: &mut Window, cx: &mut Context<Self>) -> Div {
        match paint {
            Paint::None => {}
            Paint::Solid(_) => {
                col = col.child(Button::new(if stroke { "sp-s-fg" } else { "sp-f-fg" }).label("Color = FG").xsmall().on_click(cx.listener(move |e, _, _, cx| {
                    let c = e.fg;
                    e.shape_edit(if stroke { "Stroke" } else { "Fill" }, |s| e_set_paint(s, stroke, Paint::Solid(c)));
                    cx.notify();
                })));
            }
            Paint::Gradient(g) => {
                let kinds: Vec<(GradKind, &'static str)> = GradKind::ALL.iter().map(|k| (*k, k.label())).collect();
                col = col.child(segmented(if stroke { "sp-s-gk" } else { "sp-f-gk" }, &kinds, g.kind, cx, if stroke { |e: &mut Editor, v| e.edit_grad(true, |g| g.kind = v) } else { |e: &mut Editor, v| e.edit_grad(false, |g| g.kind = v) }));
                col = if stroke {
                    col.child(self.opt_slider("Angle", "sp.s.ga", spec(-180.0, 180.0, 1.0, |e| e.grad_of(true).map_or(0.0, |g| g.angle), |e, v| e.edit_grad(true, |g| g.angle = v)), |v| format!("{v:.0}°"), 110., window, cx))
                        .child(self.opt_slider("Scale", "sp.s.gs", spec(10.0, 300.0, 1.0, |e| e.grad_of(true).map_or(100.0, |g| g.scale * 100.0), |e, v| e.edit_grad(true, |g| g.scale = v / 100.0)), |v| format!("{v:.0}%"), 110., window, cx))
                } else {
                    col.child(self.opt_slider("Angle", "sp.f.ga", spec(-180.0, 180.0, 1.0, |e| e.grad_of(false).map_or(0.0, |g| g.angle), |e, v| e.edit_grad(false, |g| g.angle = v)), |v| format!("{v:.0}°"), 110., window, cx))
                        .child(self.opt_slider("Scale", "sp.f.gs", spec(10.0, 300.0, 1.0, |e| e.grad_of(false).map_or(100.0, |g| g.scale * 100.0), |e, v| e.edit_grad(false, |g| g.scale = v / 100.0)), |v| format!("{v:.0}%"), 110., window, cx))
                };
                let rev = g.reverse;
                col = col.child(
                    div()
                        .flex()
                        .gap_1()
                        .child(toggle(if stroke { "sp-s-grev" } else { "sp-f-grev" }, "Reverse", rev).on_click(cx.listener(move |e, _, _, cx| {
                            e.edit_grad(stroke, |g| g.reverse = !rev);
                            cx.notify();
                        })))
                        .child(Button::new(if stroke { "sp-s-gfb" } else { "sp-f-gfb" }).label("FG → BG").xsmall().on_click(cx.listener(move |e, _, _, cx| {
                            let mut gr = Gradient::two("FG to BG", e.fg, e.bg);
                            gr.sort();
                            e.edit_grad(stroke, |g| g.grad = gr);
                            cx.notify();
                        }))),
                );
            }
            Paint::Pattern(pp) => {
                let names: Vec<(usize, &'static str)> = crate::engine::retouch::patterns().iter().enumerate().map(|(i, p)| (i, p.name)).collect();
                col = col.child(div().flex().flex_wrap().child(segmented(if stroke { "sp-s-pat" } else { "sp-f-pat" }, &names, pp.which, cx, if stroke { |e: &mut Editor, v| e.edit_pat(true, |p| p.which = v) } else { |e: &mut Editor, v| e.edit_pat(false, |p| p.which = v) })));
                col = if stroke {
                    col.child(self.opt_slider("Scale", "sp.s.ps", spec(10.0, 800.0, 1.0, |e| e.pat_of(true).map_or(100.0, |p| p.scale * 100.0), |e, v| e.edit_pat(true, |p| p.scale = v / 100.0)), |v| format!("{v:.0}%"), 110., window, cx))
                } else {
                    col.child(self.opt_slider("Scale", "sp.f.ps", spec(10.0, 800.0, 1.0, |e| e.pat_of(false).map_or(100.0, |p| p.scale * 100.0), |e, v| e.edit_pat(false, |p| p.scale = v / 100.0)), |v| format!("{v:.0}%"), 110., window, cx))
                        .child(self.opt_slider("Angle", "sp.f.pa", spec(-180.0, 180.0, 1.0, |e| e.pat_of(false).map_or(0.0, |p| p.angle), |e, v| e.edit_pat(false, |p| p.angle = v)), |v| format!("{v:.0}°"), 110., window, cx))
                };
            }
        }
        col
    }

    fn new_paint(&self, kind: u8) -> Paint {
        match kind {
            0 => Paint::None,
            1 => Paint::Solid(self.fg),
            2 => {
                let mut g = Gradient::two("FG to BG", self.fg, self.bg);
                g.sort();
                Paint::Gradient(GradPaint { grad: g, kind: GradKind::Linear, angle: 0.0, scale: 1.0, reverse: false })
            }
            _ => Paint::Pattern(PatPaint { which: 0, scale: 1.0, angle: 0.0 }),
        }
    }

    fn paint_of(&self, stroke: bool) -> Option<Paint> {
        let (_, s) = self.active_shape()?;
        if stroke { s.stroke.as_ref().map(|(p, _)| p.clone()) } else { Some(s.fill.clone()) }
    }

    fn grad_of(&self, stroke: bool) -> Option<GradPaint> {
        match self.paint_of(stroke)? {
            Paint::Gradient(g) => Some(g),
            _ => None,
        }
    }

    fn pat_of(&self, stroke: bool) -> Option<PatPaint> {
        match self.paint_of(stroke)? {
            Paint::Pattern(p) => Some(p),
            _ => None,
        }
    }

    fn edit_grad(&mut self, stroke: bool, f: impl FnOnce(&mut GradPaint)) {
        let Some(mut g) = self.grad_of(stroke) else { return };
        f(&mut g);
        self.shape_edit(if stroke { "Stroke Gradient" } else { "Fill Gradient" }, |s| e_set_paint(s, stroke, Paint::Gradient(g)));
    }

    fn edit_pat(&mut self, stroke: bool, f: impl FnOnce(&mut PatPaint)) {
        let Some(mut p) = self.pat_of(stroke) else { return };
        f(&mut p);
        self.shape_edit(if stroke { "Stroke Pattern" } else { "Fill Pattern" }, |s| e_set_paint(s, stroke, Paint::Pattern(p)));
    }

    fn stroke_dash(&self, k: usize) -> f32 {
        self.active_shape().and_then(|(_, s)| s.stroke.as_ref().and_then(|(_, st)| st.dashes.get(k).copied())).unwrap_or(1.0)
    }

    fn set_stroke_dash(&mut self, k: usize, v: f32) {
        self.shape_edit("Stroke Dashes", |s| {
            if let Some((_, st)) = &mut s.stroke {
                if st.dashes.len() < 2 {
                    st.dashes = vec![3.0, 1.5];
                }
                st.dashes[k] = v;
            }
        });
    }

    fn corner_radius(&self, k: usize) -> f32 {
        match self.active_shape().map(|(_, s)| s.geom.clone()) {
            Some(ShapeGeom::Rect { radii, .. }) => radii[k],
            _ => 0.0,
        }
    }

    fn set_corner(&mut self, k: usize, v: f32) {
        self.shape_edit("Corner Radius", |s| {
            if let ShapeGeom::Rect { radii, .. } = &mut s.geom {
                radii[k] = v;
            }
        });
    }

    fn poly_param(&self) -> (f32, f32) {
        match self.active_shape().map(|(_, s)| s.geom.clone()) {
            Some(ShapeGeom::Polygon { sides, star, .. }) => (sides as f32, star),
            _ => (5.0, 0.0),
        }
    }
}

fn e_set_paint(s: &mut ShapeLayer, stroke: bool, p: Paint) {
    if stroke {
        if let Some((sp, _)) = &mut s.stroke {
            *sp = p;
        }
    } else {
        s.fill = p;
    }
}
