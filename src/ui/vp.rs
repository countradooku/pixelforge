//! Filter ▸ Vanishing Point: an interactive workspace on the active layer.
//!
//! Tools: Edit Plane (V: drag corners; ⌘-drag an edge's middle handle to tear off a
//! perpendicular plane), Create Plane (C: click four corners), Marquee (M:
//! a rectangle on a plane; ⌥-drag copies it in perspective, healing when
//! Heal is on), Stamp (S: ⌥-click the source, paint the copy), Brush (B:
//! the foreground colour, as circles on the plane) and Transform (T: paste
//! the clipboard onto a plane, drag to move, corner to scale). Enter / OK
//! applies everything as one undo step and keeps the planes with the
//! document; Esc / Cancel restores the layer.

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::Sizable as _;
use gpui_kit::prelude::*;
use gpui_kit::{Bounds, Context, Div, Hsla, Pixels, Point, SharedString, Window, div, hsla, point, px};

use crate::engine::grid::{RectI, TileGrid};
use crate::engine::vanish::{self, DabSource, Frame, Plane, Pt, Scene};

use super::editor::{Editor, Overlay};
use super::options::{label, segmented, spec, toggle};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VpTool {
    Edit,
    Create,
    Marquee,
    Stamp,
    Brush,
    Transform,
}

impl VpTool {
    pub const ALL: [VpTool; 6] = [VpTool::Edit, VpTool::Create, VpTool::Marquee, VpTool::Stamp, VpTool::Brush, VpTool::Transform];

    pub fn label(self) -> &'static str {
        match self {
            VpTool::Edit => "Edit Plane (V)",
            VpTool::Create => "Create Plane (C)",
            VpTool::Marquee => "Marquee (M)",
            VpTool::Stamp => "Stamp (S)",
            VpTool::Brush => "Brush (B)",
            VpTool::Transform => "Transform (T)",
        }
    }
}

/// A marquee: plane `plane`, rectangle `(s0, t0, s1, t1)` in its frame.
#[derive(Clone, Copy, Debug)]
pub struct Marquee {
    pub plane: usize,
    pub r: (f32, f32, f32, f32),
}

/// Pasted content waiting on a plane (Transform).
pub struct Floating {
    pub content: TileGrid,
    pub plane: usize,
    pub r: (f32, f32, f32, f32),
}

pub enum VpDrag {
    Corner { plane: usize, k: usize },
    TearOff { plane: usize, new: usize },
    NewMarquee { plane: usize, start: Pt },
    MoveMarquee { start: Pt, r0: (f32, f32, f32, f32), copy: bool },
    Stroke { plane: usize, last: Pt, offset: (f32, f32), dirty: RectI },
    FloatMove { start: Pt, r0: (f32, f32, f32, f32) },
    FloatScale { r0: (f32, f32, f32, f32) },
}

pub struct VpState {
    pub scene: Scene,
    pub tool: VpTool,
    pub sel: Option<usize>,
    pub creating: Vec<Pt>,
    /// The edited layer (id) and its pixels when the workspace opened.
    pub layer: u64,
    pub orig: TileGrid,
    /// The layer's pixels at the start of the current stroke.
    pub base: TileGrid,
    pub grid_step: f32,
    pub radius: f32,
    pub hardness: f32,
    pub opacity: f32,
    pub heal: bool,
    pub marquee: Option<Marquee>,
    pub stamp_src: Option<(usize, Pt)>,
    pub float: Option<Floating>,
    pub drag: Option<VpDrag>,
}

const PLANE_BLUE: (f32, f32) = (0.58, 0.9);

impl Editor {
    pub fn vp_active(&self) -> bool {
        self.vpw.is_some()
    }

    /// Opens the workspace on the active pixel layer, with the document's planes.
    pub fn open_vanishing_point(&mut self, cx: &mut Context<Self>) {
        let li = self.doc.active;
        if !self.ensure_pixels(li) {
            return;
        }
        let l = &self.doc.layers[li];
        let mut scene = Scene::new(self.doc.w, self.doc.h);
        scene.planes = self.doc.ann.vp_planes.clone();
        self.vpw = Some(VpState {
            scene,
            tool: if self.doc.ann.vp_planes.is_empty() { VpTool::Create } else { VpTool::Edit },
            sel: None,
            creating: vec![],
            layer: l.id,
            orig: l.grid.clone(),
            base: l.grid.clone(),
            grid_step: 100.0,
            radius: 40.0,
            hardness: 0.5,
            opacity: 1.0,
            heal: true,
            marquee: None,
            stamp_src: None,
            float: None,
            drag: None,
        });
        self.flash("Vanishing Point: click four corners of a plane · Enter applies · Esc cancels");
        cx.notify();
    }

    fn vp_layer(&self) -> Option<usize> {
        let id = self.vpw.as_ref()?.layer;
        self.doc.layers.iter().position(|l| l.id == id)
    }

    /// Replaces the edited layer's pixels (live preview).
    fn vp_set(&mut self, g: TileGrid) {
        if let Some(li) = self.vp_layer() {
            self.doc.set_layer_grid(li, g);
        }
    }

    fn vp_grid(&self) -> Option<TileGrid> {
        self.vp_layer().map(|li| self.doc.layers[li].grid.clone())
    }

    fn vp_tol(&self, screen_px: f32) -> f32 {
        screen_px / self.vp.logical_scale(self.sf)
    }

    /// Applies the edits (one undo step) and keeps the planes.
    pub fn vp_apply(&mut self, cx: &mut Context<Self>) {
        self.vp_commit_float();
        let Some(st) = self.vpw.take() else { return };
        let Some(li) = self.doc.layers.iter().position(|l| l.id == st.layer) else { return };
        let work = self.doc.layers[li].grid.clone();
        self.doc.set_layer_grid(li, st.orig);
        self.doc.push_undo("Vanishing Point");
        self.doc.set_layer_grid(li, work);
        self.doc.ann.vp_planes = st.scene.planes;
        self.doc.mark_all_dirty();
        cx.notify();
    }

    pub fn vp_cancel(&mut self, cx: &mut Context<Self>) {
        let Some(st) = self.vpw.take() else { return };
        if let Some(li) = self.doc.layers.iter().position(|l| l.id == st.layer) {
            self.doc.set_layer_grid(li, st.orig);
        }
        cx.notify();
    }

    /// Transform: drops the clipboard (or the selected pixels) onto the
    /// selected plane.
    pub fn vp_paste(&mut self) {
        // The clipboard is a canvas-sized masked copy: keep its content.
        let clip = self.clipboard_grid().and_then(|g| crate::engine::xform::content_rect(g).map(|r| crate::engine::ops::crop(g, r)));
        let content = clip.or_else(|| {
            let sel = self.doc.selection.as_ref()?;
            let r = sel.shape_bounds().intersect(&self.doc.bounds());
            (!r.is_empty()).then(|| crate::engine::ops::crop(&self.doc.layers[self.doc.active].grid, r))
        });
        let Some(content) = content else {
            self.flash("Copy something first (or make a selection) to paste it onto a plane");
            return;
        };
        let Some(st) = &mut self.vpw else { return };
        let Some(i) = st.sel.or((!st.scene.planes.is_empty()).then_some(0)) else {
            self.flash("Create a plane first");
            return;
        };
        let f = st.scene.frame(i);
        let aspect = content.w as f32 / content.h.max(1) as f32;
        let w = f.sw * 0.4;
        let h = w / aspect;
        let (cs, ct) = (f.sw * 0.5, f.sh * 0.5);
        st.float = Some(Floating { content, plane: i, r: (cs - w / 2.0, ct - h / 2.0, cs + w / 2.0, ct + h / 2.0) });
        st.tool = VpTool::Transform;
    }

    fn vp_commit_float(&mut self) {
        let Some(st) = &mut self.vpw else { return };
        let Some(fl) = st.float.take() else { return };
        let f = st.scene.frame(fl.plane);
        if let Some(g) = self.vp_grid() {
            let (out, _) = vanish::place(&g, &fl.content, &f, fl.r);
            self.vp_set(out);
        }
    }

    /// Handles a mouse-down inside the workspace (always consumes it).
    pub fn vp_down(&mut self, q: Pt, alt: bool, cmd: bool) -> bool {
        let tol = self.vp_tol(8.0);
        let grid = self.vp_grid();
        let fg = self.fg;
        let Some(st) = &mut self.vpw else { return false };
        let near = |a: Pt, b: Pt| (a.0 - b.0).hypot(a.1 - b.1) <= tol;
        match st.tool {
            VpTool::Create => {
                st.creating.push(q);
                if st.creating.len() == 4 {
                    let p = Plane { c: [st.creating[0], st.creating[1], st.creating[2], st.creating[3]] };
                    st.creating.clear();
                    if p.valid() {
                        st.scene.planes.push(p);
                        st.sel = Some(st.scene.planes.len() - 1);
                        st.tool = VpTool::Edit;
                    }
                }
            }
            VpTool::Edit => {
                // Corners, then (⌘) edge handles of the selected plane, then planes.
                let hit = (0..st.scene.planes.len()).rev().find_map(|i| (0..4).find(|&k| near(st.scene.planes[i].c[k], q)).map(|k| (i, k)));
                if let Some((plane, k)) = hit {
                    st.sel = Some(plane);
                    st.drag = Some(VpDrag::Corner { plane, k });
                } else if let Some(i) = st.sel.filter(|_| cmd) {
                    let p = st.scene.planes[i];
                    let edge = (0..4).find(|&e| {
                        let (a, b) = (p.c[e], p.c[(e + 1) % 4]);
                        near(((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5), q)
                    });
                    if let Some(e) = edge {
                        let (a, b) = (p.c[e], p.c[(e + 1) % 4]);
                        let depth = (a.0 - b.0).hypot(a.1 - b.1) * 0.5;
                        if let Some(new) = st.scene.tear_off(i, e, depth) {
                            st.sel = Some(new);
                            st.drag = Some(VpDrag::TearOff { plane: i, new });
                        }
                    }
                } else {
                    st.sel = st.scene.plane_at(q);
                }
            }
            VpTool::Marquee => {
                let Some(plane) = st.scene.plane_at(q) else { return true };
                let f = st.scene.frame(plane);
                let pq = f.to_plane(q);
                match st.marquee.filter(|m| m.plane == plane && pq.0 >= m.r.0 && pq.0 <= m.r.2 && pq.1 >= m.r.1 && pq.1 <= m.r.3) {
                    Some(m) => st.drag = Some(VpDrag::MoveMarquee { start: pq, r0: m.r, copy: alt }),
                    None => {
                        st.marquee = Some(Marquee { plane, r: (pq.0, pq.1, pq.0, pq.1) });
                        st.drag = Some(VpDrag::NewMarquee { plane, start: pq });
                    }
                }
            }
            VpTool::Stamp | VpTool::Brush => {
                let Some(plane) = st.scene.plane_at(q) else { return true };
                let f = st.scene.frame(plane);
                let pq = f.to_plane(q);
                if st.tool == VpTool::Stamp && alt {
                    st.stamp_src = Some((plane, pq));
                    return true;
                }
                let Some(g) = grid else { return true };
                st.base = g.clone();
                let offset = match (st.tool, st.stamp_src) {
                    (VpTool::Stamp, Some((sp, s))) => {
                        // Source and destination in one frame: the source plane's
                        // point carried over by plane coordinates.
                        let _ = sp;
                        (pq.0 - s.0, pq.1 - s.1)
                    }
                    (VpTool::Stamp, None) => {
                        self.flash("⌥-click to set the Stamp source");
                        return true;
                    }
                    _ => (0.0, 0.0),
                };
                let what = if st.tool == VpTool::Stamp { DabSource::Stamp { src: st.scene.frame(st.stamp_src.unwrap().0), offset } } else { DabSource::Color(fg, st.opacity) };
                let (out, r) = vanish::dab(&g, &st.base, &f, pq.0, pq.1, st.radius, st.hardness, what);
                st.drag = Some(VpDrag::Stroke { plane, last: pq, offset, dirty: r });
                self.vp_set(out);
            }
            VpTool::Transform => {
                let Some(fl) = &st.float else { return true };
                let f = st.scene.frame(fl.plane);
                let quad = f.rect_quad(fl.r);
                if near(quad[2], q) {
                    st.drag = Some(VpDrag::FloatScale { r0: fl.r });
                } else {
                    let pq = f.to_plane(q);
                    st.drag = Some(VpDrag::FloatMove { start: pq, r0: fl.r });
                }
            }
        }
        true
    }

    pub fn vp_move(&mut self, q: Pt) {
        let grid = self.vp_grid();
        let fg = self.fg;
        let Some(st) = &mut self.vpw else { return };
        let Some(d) = st.drag.take() else { return };
        let d = match d {
            VpDrag::Corner { plane, k } => {
                let mut p = st.scene.planes[plane];
                p.c[k] = q;
                st.scene.planes[plane] = p;
                VpDrag::Corner { plane, k }
            }
            VpDrag::TearOff { plane, new } => {
                // Drag the far edge: depth follows the pointer along the normal.
                let p = st.scene.planes[plane];
                let w = st.scene.planes[new];
                let (a, a2) = (w.c[3], w.c[0]);
                let dir = (a2.0 - a.0, a2.1 - a.1);
                let l = dir.0.hypot(dir.1).max(1e-3);
                let depth = (((q.0 - a.0) * dir.0 + (q.1 - a.1) * dir.1) / l).max(4.0);
                let edge = (0..4).find(|&e| p.c[e] == a && p.c[(e + 1) % 4] == w.c[2]).unwrap_or(0);
                st.scene.planes.truncate(new);
                let new = st.scene.tear_off(plane, edge, depth).unwrap_or_else(|| {
                    st.scene.planes.push(w);
                    st.scene.planes.len() - 1
                });
                st.sel = Some(new);
                VpDrag::TearOff { plane, new }
            }
            VpDrag::NewMarquee { plane, start } => {
                let pq = st.scene.frame(plane).to_plane(q);
                st.marquee = Some(Marquee { plane, r: (start.0.min(pq.0), start.1.min(pq.1), start.0.max(pq.0), start.1.max(pq.1)) });
                VpDrag::NewMarquee { plane, start }
            }
            VpDrag::MoveMarquee { start, r0, copy } => {
                if let Some(m) = &mut st.marquee {
                    let pq = st.scene.frame(m.plane).to_plane(q);
                    let (dx, dy) = (pq.0 - start.0, pq.1 - start.1);
                    m.r = (r0.0 + dx, r0.1 + dy, r0.2 + dx, r0.3 + dy);
                }
                VpDrag::MoveMarquee { start, r0, copy }
            }
            VpDrag::Stroke { plane, last, offset, mut dirty } => {
                let f = st.scene.frame(plane);
                let pq = f.to_plane(q);
                let Some(mut g) = grid else { return };
                // Dabs every quarter radius along the path, in plane space.
                let step = (st.radius * 0.25).max(0.5);
                let dist = (pq.0 - last.0).hypot(pq.1 - last.1);
                let n = (dist / step).floor() as usize;
                let mut at = last;
                for k in 1..=n {
                    let t = k as f32 * step / dist;
                    at = (last.0 + (pq.0 - last.0) * t, last.1 + (pq.1 - last.1) * t);
                    let what = match st.tool {
                        VpTool::Stamp => DabSource::Stamp { src: st.scene.frame(st.stamp_src.map_or(plane, |s| s.0)), offset },
                        _ => DabSource::Color(fg, st.opacity),
                    };
                    let (out, r) = vanish::dab(&g, &st.base, &f, at.0, at.1, st.radius, st.hardness, what);
                    g = out;
                    dirty = dirty.union(&r);
                }
                if n > 0 {
                    self.vp_set(g);
                }
                let Some(st) = &mut self.vpw else { return };
                st.drag = Some(VpDrag::Stroke { plane, last: if n > 0 { at } else { last }, offset, dirty });
                return;
            }
            VpDrag::FloatMove { start, r0 } => {
                if let Some(fl) = &mut st.float {
                    let pq = st.scene.frame(fl.plane).to_plane(q);
                    let (dx, dy) = (pq.0 - start.0, pq.1 - start.1);
                    fl.r = (r0.0 + dx, r0.1 + dy, r0.2 + dx, r0.3 + dy);
                }
                VpDrag::FloatMove { start, r0 }
            }
            VpDrag::FloatScale { r0 } => {
                if let Some(fl) = &mut st.float {
                    let pq = st.scene.frame(fl.plane).to_plane(q);
                    // Uniform scale about the top-left corner.
                    let k = ((pq.0 - r0.0) / (r0.2 - r0.0)).max((pq.1 - r0.1) / (r0.3 - r0.1)).max(0.05);
                    fl.r = (r0.0, r0.1, r0.0 + (r0.2 - r0.0) * k, r0.1 + (r0.3 - r0.1) * k);
                }
                VpDrag::FloatScale { r0 }
            }
        };
        st.drag = Some(d);
    }

    pub fn vp_up(&mut self) {
        let grid = self.vp_grid();
        let Some(st) = &mut self.vpw else { return };
        match st.drag.take() {
            Some(VpDrag::MoveMarquee { r0, copy: true, .. }) => {
                // ⌥-drag: copy the marquee's original area to where it was dropped.
                let (Some(m), Some(g)) = (st.marquee, grid) else { return };
                let f = st.scene.frame(m.plane);
                let (out, _) = vanish::copy_rect(&g, &g, &f, r0, &f, (m.r.0, m.r.1), st.heal);
                self.vp_set(out);
            }
            Some(VpDrag::Stroke { dirty, .. }) if st.heal && st.tool == VpTool::Stamp => {
                let Some(g) = grid else { return };
                let out = vanish::heal_region(&g, &st.base, dirty, &[]);
                self.vp_set(out);
            }
            _ => {}
        }
    }

    /// Workspace keys: tool letters, Delete (selected plane), Enter, Esc.
    pub fn vp_key(&mut self, key: &str, cx: &mut Context<Self>) -> bool {
        let Some(st) = &mut self.vpw else { return false };
        match key {
            "enter" => self.vp_apply(cx),
            "escape" => self.vp_cancel(cx),
            "v" => st.tool = VpTool::Edit,
            "c" => st.tool = VpTool::Create,
            "m" => st.tool = VpTool::Marquee,
            "s" => st.tool = VpTool::Stamp,
            "b" => st.tool = VpTool::Brush,
            "t" => st.tool = VpTool::Transform,
            "backspace" | "delete" => {
                if let Some(i) = st.sel.take() {
                    st.scene.planes.remove(i);
                    st.marquee = None;
                    st.float = None;
                    st.stamp_src = None;
                }
            }
            "[" => st.radius = (st.radius / 1.2).max(2.0),
            "]" => st.radius = (st.radius * 1.2).min(2000.0),
            _ => {}
        }
        cx.notify();
        true
    }

    // ----- overlays & options -------------------------------------------------------------

    pub fn vp_overlays(&self, out: &mut Vec<Overlay>) {
        let Some(st) = &self.vpw else { return };
        let (c, sf) = (self.canvas, self.sf);
        let sc = |p: Pt| self.vp.to_screen(p.0, p.1, c, sf);
        let white = hsla(0., 0., 1., 1.);
        let black = hsla(0., 0., 0., 1.);
        let handle = |p: Point<Pixels>, r: f32, bg: Hsla| Overlay::Quad { bounds: Bounds::from_corners(point(p.x - px(r), p.y - px(r)), point(p.x + px(r), p.y + px(r))), bg, border: black, border_w: 1.0, radius: 0.0 };
        for (i, p) in st.scene.planes.iter().enumerate() {
            let col = if p.valid() { hsla(PLANE_BLUE.0, PLANE_BLUE.1, 0.6, 1.0) } else { hsla(0.0, 0.9, 0.55, 1.0) };
            if p.valid() {
                let grid: Vec<(Vec<Point<Pixels>>, bool)> = st.scene.grid(i, st.grid_step).into_iter().map(|l| (l.into_iter().map(sc).collect(), false)).collect();
                out.push(Overlay::Path { polys: grid, dashed: false, color: hsla(PLANE_BLUE.0, PLANE_BLUE.1, 0.7, 0.45), width: 1.0 });
            }
            let quad: Vec<Point<Pixels>> = p.c.iter().map(|&q| sc(q)).collect();
            out.push(Overlay::Path { polys: vec![(quad.clone(), true)], dashed: false, color: col, width: if st.sel == Some(i) { 2.5 } else { 1.5 } });
            if st.tool == VpTool::Edit {
                for q in &quad {
                    out.push(handle(*q, 4.0, white));
                }
                if st.sel == Some(i) {
                    for e in 0..4 {
                        let (a, b) = (quad[e], quad[(e + 1) % 4]);
                        out.push(handle(point((a.x + b.x) * 0.5, (a.y + b.y) * 0.5), 3.0, col));
                    }
                }
            }
        }
        if !st.creating.is_empty() {
            let pts: Vec<Point<Pixels>> = st.creating.iter().map(|&q| sc(q)).collect();
            out.push(Overlay::Path { polys: vec![(pts.clone(), false)], dashed: true, color: white, width: 1.5 });
            for q in pts {
                out.push(handle(q, 4.0, white));
            }
        }
        if let Some(m) = st.marquee.filter(|m| m.plane < st.scene.planes.len()) {
            let quad: Vec<Point<Pixels>> = st.scene.frame(m.plane).rect_quad(m.r).iter().map(|&q| sc(q)).collect();
            out.push(Overlay::Path { polys: vec![(quad.clone(), true)], dashed: false, color: black, width: 1.0 });
            out.push(Overlay::Path { polys: vec![(quad, true)], dashed: true, color: white, width: 1.0 });
        }
        if let Some((p, s)) = st.stamp_src.filter(|s| s.0 < st.scene.planes.len()) {
            let q = sc(st.scene.frame(p).to_img(s.0, s.1));
            out.push(Overlay::Path { polys: vec![(vec![q - point(px(9.), px(0.)), q + point(px(9.), px(0.))], false), (vec![q - point(px(0.), px(9.)), q + point(px(0.), px(9.))], false)], dashed: false, color: white, width: 1.5 });
        }
        if let Some(fl) = st.float.as_ref().filter(|f| f.plane < st.scene.planes.len()) {
            let quad: Vec<Point<Pixels>> = st.scene.frame(fl.plane).rect_quad(fl.r).iter().map(|&q| sc(q)).collect();
            out.push(Overlay::Path { polys: vec![(quad.clone(), true)], dashed: false, color: white, width: 1.5 });
            out.push(handle(quad[2], 5.0, white));
        }
        // Brush / Stamp footprint: a circle on the plane under the pointer.
        if let (VpTool::Brush | VpTool::Stamp, Some(m)) = (st.tool, self.mouse_img) {
            if let Some(i) = st.scene.plane_at(m) {
                let f: Frame = st.scene.frame(i);
                let (s, t) = f.to_plane(m);
                let ring: Vec<Point<Pixels>> = (0..=40).map(|k| {
                    let a = k as f32 / 40.0 * std::f32::consts::TAU;
                    sc(f.to_img(s + st.radius * a.cos(), t + st.radius * a.sin()))
                }).collect();
                out.push(Overlay::Path { polys: vec![(ring, true)], dashed: false, color: white, width: 1.0 });
            }
        }
    }

    pub fn render_vp_options(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let Some(st) = &self.vpw else { return div() };
        let tool = st.tool;
        let heal = st.heal;
        let n = st.scene.planes.len();
        let focal = st.scene.focal();
        let mut bar = div()
            .min_h(px(42.))
            .flex()
            .flex_wrap()
            .items_center()
            .gap_x_3()
            .gap_y_1()
            .px_3()
            .py_1()
            .child(label("Vanishing Point"))
            .child(segmented("vp-tool", &VpTool::ALL.map(|t| (t, t.label())), tool, cx, |e, t| {
                e.vp_commit_float();
                if let Some(s) = &mut e.vpw {
                    s.tool = t;
                }
            }))
            .child(self.opt_slider("Grid", "vp.grid", spec(10.0, 500.0, 5.0, |e| e.vpw.as_ref().map_or(100.0, |s| s.grid_step), |e, v| {
                if let Some(s) = &mut e.vpw {
                    s.grid_step = v;
                }
            }), |v| format!("{v:.0}"), 70., window, cx));
        if matches!(tool, VpTool::Stamp | VpTool::Brush) {
            bar = bar
                .child(self.opt_slider("Diameter", "vp.rad", spec(2.0, 1000.0, 1.0, |e| e.vpw.as_ref().map_or(40.0, |s| s.radius * 2.0), |e, v| {
                    if let Some(s) = &mut e.vpw {
                        s.radius = v * 0.5;
                    }
                }), |v| format!("{v:.0}"), 80., window, cx))
                .child(self.opt_slider("Hardness", "vp.hard", spec(0.0, 100.0, 1.0, |e| e.vpw.as_ref().map_or(50.0, |s| s.hardness * 100.0), |e, v| {
                    if let Some(s) = &mut e.vpw {
                        s.hardness = v / 100.0;
                    }
                }), |v| format!("{v:.0}%"), 60., window, cx));
        }
        if tool == VpTool::Brush {
            bar = bar.child(self.opt_slider("Opacity", "vp.op", spec(1.0, 100.0, 1.0, |e| e.vpw.as_ref().map_or(100.0, |s| s.opacity * 100.0), |e, v| {
                if let Some(s) = &mut e.vpw {
                    s.opacity = v / 100.0;
                }
            }), |v| format!("{v:.0}%"), 60., window, cx));
        }
        if matches!(tool, VpTool::Stamp | VpTool::Marquee) {
            bar = bar.child(toggle("vp-heal", "Heal", heal).on_click(cx.listener(|e, _, _, cx| {
                if let Some(s) = &mut e.vpw {
                    s.heal = !s.heal;
                }
                cx.notify();
            })));
        }
        if tool == VpTool::Transform {
            bar = bar.child(Button::new("vp-paste").label("Paste onto Plane").xsmall().ghost().on_click(cx.listener(|e, _, _, cx| {
                e.vp_paste();
                cx.notify();
            })));
        }
        let hint: SharedString = match tool {
            VpTool::Create => "Click the four corners of a rectangle in perspective".into(),
            VpTool::Edit => "Drag corners · ⌘-drag an edge's middle handle to tear off a perpendicular plane · Delete removes a plane".into(),
            VpTool::Marquee => "Drag a rectangle on a plane · ⌥-drag it to copy in perspective".into(),
            VpTool::Stamp => "⌥-click the source, then paint".into(),
            VpTool::Brush => "Paints the foreground colour in perspective".into(),
            VpTool::Transform => "Paste, then drag to move · drag the corner handle to scale".into(),
        };
        bar.child(label(format!("{n} planes · focal {focal:.0} px"))).child(label(hint)).child(
            div()
                .flex()
                .gap_1()
                .child(Button::new("vp-ok").label("OK (Enter)").xsmall().primary().on_click(cx.listener(|e, _, _, cx| e.vp_apply(cx))))
                .child(Button::new("vp-cancel").label("Cancel (Esc)").xsmall().ghost().on_click(cx.listener(|e, _, _, cx| e.vp_cancel(cx)))),
        )
    }
}
