//! Edit ▸ Perspective Warp: a workspace on the active pixel layer.
//!
//! **Layout** (L): drag on the image to draw a quad over a plane; drag its
//! corners onto the plane's corners. A corner dropped near another quad's
//! corner snaps to it and the quads join (they then share that edge).
//! Delete removes the selected quad. **Warp** (W): drag corners; the image
//! follows (each quad projectively, blended across joined edges) and the
//! preview renders when the drag ends. Straighten Vertical / Horizontal /
//! Both levels the near-vertical / near-horizontal edges. Enter applies
//! (one undo step, in a background job like Free Transform), Esc cancels.

use gpui_kit::component::Sizable as _;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::prelude::*;
use gpui_kit::{Bounds, Context, Div, Hsla, Pixels, Point, SharedString, Window, div, hsla, point, px};

use crate::engine::grid::TileGrid;
use crate::engine::perspwarp::{self, Mesh, Pt, Straighten};

use super::editor::{Editor, Overlay};
use super::options::{label, segmented};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PwPhase {
    Layout,
    Warp,
}

pub enum PwDrag {
    NewQuad { start: Pt, cur: Pt },
    Vertex { v: usize },
}

pub struct PwState {
    pub mesh: Mesh,
    pub phase: PwPhase,
    /// The edited layer (id) and its pixels when the workspace opened.
    pub layer: u64,
    pub orig: TileGrid,
    pub drag: Option<PwDrag>,
    pub sel: Option<usize>,
    /// Bumped by every edit; a preview job only lands if still current.
    pub generation: u64,
}

/// Corner snapping distance (screen pixels).
const SNAP_PX: f32 = 10.0;

impl Editor {
    pub fn pw_active(&self) -> bool {
        self.pww.is_some()
    }

    pub fn open_perspective_warp(&mut self, cx: &mut Context<Self>) {
        if self.input_blocked() || self.vp_active() {
            return;
        }
        let li = self.doc.active;
        if !self.ensure_pixels(li) {
            return;
        }
        let l = &self.doc.layers[li];
        self.pww = Some(PwState { mesh: Mesh::default(), phase: PwPhase::Layout, layer: l.id, orig: l.grid.clone(), drag: None, sel: None, generation: 0 });
        self.flash("Perspective Warp: drag to draw a quad over each plane · W: Warp · Enter applies · Esc cancels");
        cx.notify();
    }

    fn pw_layer(&self) -> Option<usize> {
        let id = self.pww.as_ref()?.layer;
        self.doc.layers.iter().position(|l| l.id == id)
    }

    fn pw_tol(&self) -> f32 {
        SNAP_PX / self.vp.logical_scale(self.sf)
    }

    pub fn pw_set_phase(&mut self, phase: PwPhase, cx: &mut Context<Self>) {
        self.pw_phase(phase);
        cx.notify();
    }

    fn pw_phase(&mut self, phase: PwPhase) {
        let Some(st) = &mut self.pww else { return };
        if st.phase == phase {
            return;
        }
        st.phase = phase;
        st.drag = None;
        st.generation += 1;
        if phase == PwPhase::Layout {
            // Back to the planes as drawn, over the original pixels.
            st.mesh.reset_warp();
            let orig = st.orig.clone();
            if let Some(li) = self.pw_layer() {
                self.doc.set_layer_grid(li, orig);
            }
        } else if st.mesh.is_empty() {
            self.flash("Draw at least one quad in Layout mode first");
        }
    }

    /// Renders the current warp into the layer (background job).
    fn pw_preview(&mut self, cx: &mut Context<Self>) {
        let Some(st) = &mut self.pww else { return };
        st.generation += 1;
        let (generation, id, orig, mesh) = (st.generation, st.layer, st.orig.clone(), st.mesh.clone());
        self.spawn_job("Perspective Warp preview", cx, move |_| {
            let out = perspwarp::apply(&orig, &mesh);
            Box::new(move |ed: &mut Editor, _| {
                if ed.pww.as_ref().is_some_and(|s| s.generation == generation && s.layer == id && s.phase == PwPhase::Warp)
                    && let Some(li) = ed.pw_layer()
                {
                    ed.doc.set_layer_grid(li, out);
                }
            })
        });
    }

    /// Applies the warp as one undo step.
    pub fn pw_apply(&mut self, cx: &mut Context<Self>) {
        if self.busy.is_some() {
            return;
        }
        let Some(st) = self.pww.take() else { return };
        let Some(li) = self.pw_layer_of(st.layer) else { return };
        if st.mesh.is_empty() || st.mesh.warp == st.mesh.layout {
            self.doc.set_layer_grid(li, st.orig);
            cx.notify();
            return;
        }
        let (id, orig, mesh) = (st.layer, st.orig, st.mesh);
        self.spawn_job("Perspective Warp", cx, move |_| {
            let out = perspwarp::apply(&orig, &mesh);
            Box::new(move |ed: &mut Editor, _| {
                if let Some(li) = ed.doc.layers.iter().position(|l| l.id == id) {
                    ed.doc.set_layer_grid(li, orig);
                    ed.doc.push_undo("Perspective Warp");
                    ed.doc.set_layer_grid(li, out);
                    ed.doc.mark_all_dirty();
                }
            })
        });
    }

    fn pw_layer_of(&self, id: u64) -> Option<usize> {
        self.doc.layers.iter().position(|l| l.id == id)
    }

    pub fn pw_cancel(&mut self, cx: &mut Context<Self>) {
        let Some(st) = self.pww.take() else { return };
        if let Some(li) = self.pw_layer_of(st.layer) {
            self.doc.set_layer_grid(li, st.orig);
        }
        cx.notify();
    }

    pub fn pw_straighten(&mut self, s: Straighten, cx: &mut Context<Self>) {
        let Some(st) = &mut self.pww else { return };
        if st.phase != PwPhase::Warp || st.mesh.is_empty() {
            return;
        }
        st.mesh.straighten(s);
        self.pw_preview(cx);
        cx.notify();
    }

    pub fn pw_reset(&mut self, cx: &mut Context<Self>) {
        let Some(st) = &mut self.pww else { return };
        st.mesh.reset_warp();
        st.generation += 1;
        let orig = st.orig.clone();
        if let Some(li) = self.pw_layer() {
            self.doc.set_layer_grid(li, orig);
        }
        cx.notify();
    }

    /// Mouse down inside the workspace (always consumed).
    pub fn pw_down(&mut self, q: Pt) {
        let tol = self.pw_tol();
        let Some(st) = &mut self.pww else { return };
        let pts = if st.phase == PwPhase::Warp { &st.mesh.warp } else { &st.mesh.layout };
        let near = (0..pts.len()).filter(|&v| (pts[v].0 - q.0).hypot(pts[v].1 - q.1) <= tol).min_by(|&a, &b| {
            let d = |v: usize| (pts[v].0 - q.0).hypot(pts[v].1 - q.1);
            d(a).total_cmp(&d(b))
        });
        if let Some(v) = near {
            st.drag = Some(PwDrag::Vertex { v });
            return;
        }
        let inside = (0..st.mesh.quads.len()).rev().find(|&i| {
            let c = if st.phase == PwPhase::Warp { st.mesh.warp_quad(i) } else { st.mesh.layout_quad(i) };
            point_in_quad(&c, q)
        });
        st.sel = inside;
        if inside.is_none() && st.phase == PwPhase::Layout {
            st.drag = Some(PwDrag::NewQuad { start: q, cur: q });
        }
    }

    pub fn pw_move(&mut self, q: Pt) {
        let Some(st) = &mut self.pww else { return };
        match &mut st.drag {
            Some(PwDrag::NewQuad { cur, .. }) => *cur = q,
            Some(PwDrag::Vertex { v }) => {
                let v = *v;
                let warp = st.phase == PwPhase::Warp;
                st.mesh.move_vertex(v, q, warp);
            }
            None => {}
        }
    }

    pub fn pw_up(&mut self, cx: &mut Context<Self>) {
        let tol = self.pw_tol();
        let Some(st) = &mut self.pww else { return };
        match st.drag.take() {
            Some(PwDrag::NewQuad { start, cur }) => {
                if (cur.0 - start.0).abs() >= tol && (cur.1 - start.1).abs() >= tol {
                    let q = st.mesh.add_quad(Mesh::rect_quad(start, cur), tol);
                    st.sel = Some(q);
                }
            }
            Some(PwDrag::Vertex { v }) => {
                if st.phase == PwPhase::Layout {
                    if st.mesh.join(v, tol) {
                        self.flash("Joined: the quads share this edge");
                    }
                } else {
                    self.pw_preview(cx);
                }
            }
            None => {}
        }
    }

    /// Workspace keys: L / W (mode), Delete (selected quad), Enter, Esc.
    pub fn pw_key(&mut self, key: &str, cx: &mut Context<Self>) -> bool {
        if self.pww.is_none() {
            return false;
        }
        match key {
            "enter" => self.pw_apply(cx),
            "escape" => self.pw_cancel(cx),
            "l" => self.pw_set_phase(PwPhase::Layout, cx),
            "w" => self.pw_set_phase(PwPhase::Warp, cx),
            "backspace" | "delete" => {
                if let Some(st) = &mut self.pww
                    && st.phase == PwPhase::Layout
                    && let Some(i) = st.sel.take()
                {
                    st.mesh.remove_quad(i);
                }
            }
            _ => {}
        }
        cx.notify();
        true
    }

    // ----- overlays & options -------------------------------------------------------------

    pub fn pw_overlays(&self, out: &mut Vec<Overlay>) {
        let Some(st) = &self.pww else { return };
        let (c, sf) = (self.canvas, self.sf);
        let sc = |p: Pt| self.vp.to_screen(p.0, p.1, c, sf);
        let white = hsla(0., 0., 1., 1.);
        let black = hsla(0., 0., 0., 1.);
        let col = if st.phase == PwPhase::Layout { hsla(0.58, 0.9, 0.6, 1.0) } else { hsla(0.12, 0.95, 0.55, 1.0) };
        let handle = |p: Point<Pixels>, r: f32, bg: Hsla| Overlay::Quad { bounds: Bounds::from_corners(point(p.x - px(r), p.y - px(r)), point(p.x + px(r), p.y + px(r))), bg, border: black, border_w: 1.0, radius: 0.0 };
        let lerp = |a: Pt, b: Pt, t: f32| (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
        for i in 0..st.mesh.quads.len() {
            let q = if st.phase == PwPhase::Warp { st.mesh.warp_quad(i) } else { st.mesh.layout_quad(i) };
            let ok = perspwarp::quad_valid(&q);
            let qc = if ok { col } else { hsla(0.0, 0.9, 0.55, 1.0) };
            // A 4 × 4 grid inside the quad.
            let mut lines = Vec::new();
            for k in 1..4 {
                let t = k as f32 / 4.0;
                lines.push((vec![sc(lerp(q[0], q[1], t)), sc(lerp(q[3], q[2], t))], false));
                lines.push((vec![sc(lerp(q[0], q[3], t)), sc(lerp(q[1], q[2], t))], false));
            }
            out.push(Overlay::Path { polys: lines, dashed: false, color: Hsla { a: 0.45, ..qc }, width: 1.0 });
            let quad: Vec<Point<Pixels>> = q.iter().map(|&p| sc(p)).collect();
            out.push(Overlay::Path { polys: vec![(quad, true)], dashed: false, color: qc, width: if st.sel == Some(i) { 2.5 } else { 1.5 } });
        }
        let pts = if st.phase == PwPhase::Warp { &st.mesh.warp } else { &st.mesh.layout };
        for &p in pts {
            out.push(handle(sc(p), 4.0, white));
        }
        if let Some(PwDrag::NewQuad { start, cur }) = &st.drag {
            let q: Vec<Point<Pixels>> = Mesh::rect_quad(*start, *cur).iter().map(|&p| sc(p)).collect();
            out.push(Overlay::Path { polys: vec![(q, true)], dashed: true, color: white, width: 1.5 });
        }
    }

    pub fn render_pw_options(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> Div {
        let Some(st) = &self.pww else { return div() };
        let phase = st.phase;
        let n = st.mesh.quads.len();
        let joined = st.mesh.shared_edges();
        let mut bar = div()
            .min_h(px(42.))
            .flex()
            .flex_wrap()
            .items_center()
            .gap_x_3()
            .gap_y_1()
            .px_3()
            .py_1()
            .child(label("Perspective Warp"))
            .child(segmented("pw-phase", &[(PwPhase::Layout, "Layout (L)"), (PwPhase::Warp, "Warp (W)")], phase, cx, |e, p| e.pw_phase(p)));
        if phase == PwPhase::Warp {
            for s in Straighten::ALL {
                let id: SharedString = format!("pw-{s:?}").into();
                bar = bar.child(Button::new(id).label(s.label()).xsmall().ghost().on_click(cx.listener(move |e, _, _, cx| e.pw_straighten(s, cx))));
            }
            bar = bar.child(Button::new("pw-reset").label("Reset Warp").xsmall().ghost().on_click(cx.listener(|e, _, _, cx| e.pw_reset(cx))));
        }
        let hint = match phase {
            PwPhase::Layout => "Drag to draw a quad over each plane · drag corners onto the plane · corners dropped together join · Delete removes",
            PwPhase::Warp => "Drag corners to move the planes · the image follows",
        };
        bar.child(label(format!("{n} quads · {joined} joined edges"))).child(label(hint)).child(
            div()
                .flex()
                .gap_1()
                .child(Button::new("pw-ok").label("OK (Enter)").xsmall().primary().on_click(cx.listener(|e, _, _, cx| e.pw_apply(cx))))
                .child(Button::new("pw-cancel").label("Cancel (Esc)").xsmall().ghost().on_click(cx.listener(|e, _, _, cx| e.pw_cancel(cx)))),
        )
    }
}

fn point_in_quad(q: &[Pt; 4], p: Pt) -> bool {
    let s: Vec<f32> = (0..4).map(|i| {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
    }).collect();
    s.iter().all(|&v| v >= 0.0) || s.iter().all(|&v| v <= 0.0)
}
