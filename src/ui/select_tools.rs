//! Selection tools: marquees, lassos (freehand, polygonal, magnetic), Object
//! and Quick Selection, Magic Wand, Select ▸ Modify, Select Subject and
//! Quick Mask.

use std::sync::Arc;
use std::time::Instant;

use gpui_kit::{Context, MouseDownEvent, Window};

use crate::engine::document::Layer;
use crate::engine::grid::{RectI, TileGrid};
use crate::engine::livewire::{self, WireParams};
use crate::engine::masks::{self, SelOp, combine_selection};
use crate::engine::ops;
use crate::engine::px;
use crate::engine::segment::{self, QuickSel};
use crate::engine::selection::{MaskGrid, Selection, Shape};
use crate::engine::vector::{P, p};

use super::editor::{Drag, Editor, QuickMask, Tool};

/// Polygonal Lasso in progress (it spans several clicks).
pub struct PolyState {
    pub pts: Vec<[f32; 2]>,
    pub op: SelOp,
}

/// Magnetic Lasso in progress: committed path, the anchor it continues
/// from, the cursor trail since that anchor and the live wire.
pub struct MagState {
    pub committed: Vec<P>,
    pub anchors: Vec<P>,
    pub trail: Vec<P>,
    pub live: Vec<P>,
    pub op: SelOp,
}

pub struct QuickDrag {
    pub qs: QuickSel,
    pub op: SelOp,
    pub base: Option<Selection>,
    pub last_solve: Instant,
    pub dirty: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Modify {
    Border,
    Smooth,
    Expand,
    Contract,
    Feather,
}

impl Modify {
    pub fn title(self) -> &'static str {
        match self {
            Modify::Border => "Border Selection",
            Modify::Smooth => "Smooth Selection",
            Modify::Expand => "Expand Selection",
            Modify::Contract => "Contract Selection",
            Modify::Feather => "Feather Selection",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Modify::Border => "Width",
            Modify::Smooth => "Sample Radius",
            Modify::Feather => "Feather Radius",
            _ => "By",
        }
    }
}

impl Editor {
    /// Source for segmentation: the active layer or the merged image.
    fn seg_layers(&self) -> (Vec<Layer>, usize) {
        if self.opts.sample_all { (self.doc.layers.clone(), usize::MAX) } else { (vec![self.doc.layers[self.doc.active].clone()], 0) }
    }

    /// Feathers / hardens a freshly drawn selection, combines it with the
    /// current one and records one undo step.
    pub fn commit_selection(&mut self, label: &str, new: Option<Selection>, op: SelOp, soft_ok: bool) {
        let (w, h) = (self.doc.w, self.doc.h);
        let mut new = new;
        if let Some(sel) = &new {
            let feather = self.opts.feather;
            let needs_mask = feather > 0.0 || (!self.opts.anti_alias && soft_ok);
            if needs_mask {
                let mut m = MaskGrid::from_selection(sel, w, h);
                if !self.opts.anti_alias && soft_ok {
                    m = m.threshold();
                }
                if feather > 0.0 {
                    m = m.feather(feather);
                }
                new = m.to_selection();
            }
        }
        let combined = combine_selection(self.doc.selection.as_ref(), new, op, w, h);
        self.doc.push_undo(label);
        self.doc.selection = combined;
    }

    pub fn selection_down(&mut self, ix: f32, iy: f32, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let op = SelOp::from_mods(ev.modifiers.shift, ev.modifiers.alt, self.opts.sel_op);
        match self.tool {
            Tool::RectSelect | Tool::EllipseSelect => {
                self.drag = Drag::Marquee { start: (ix, iy), cur: (ix, iy), ellipse: self.tool == Tool::EllipseSelect, center: false, op };
            }
            Tool::RowSelect | Tool::ColumnSelect => {
                let r = if self.tool == Tool::RowSelect {
                    RectI::new(0, iy.floor() as i32, self.doc.w as i32, iy.floor() as i32 + 1)
                } else {
                    RectI::new(ix.floor() as i32, 0, ix.floor() as i32 + 1, self.doc.h as i32)
                };
                self.commit_selection(self.tool.name(), Some(Selection::rect(r)), op, false);
            }
            Tool::Lasso => self.drag = Drag::Lasso(vec![[ix, iy]], op),
            Tool::PolyLasso => self.poly_click(ix, iy, ev.click_count >= 2, op),
            Tool::MagneticLasso => self.magnetic_click(ix, iy, ev.click_count >= 2, op),
            Tool::MagicWand => self.magic_wand(ix, iy, op),
            Tool::QuickSelect => {
                let op = if ev.modifiers.alt { SelOp::Subtract } else if op == SelOp::New && self.doc.selection.is_some() { SelOp::Add } else { op };
                let mut qs = QuickSel::new(self.doc.w, self.doc.h, self.doc.max_level());
                qs.add_dab(ix, iy, self.brush_ref().size * 0.5);
                self.drag = Drag::QuickSelect(Box::new(QuickDrag { qs, op, base: self.doc.selection.clone(), last_solve: Instant::now(), dirty: true }));
                self.quick_select_update(true);
            }
            Tool::ObjectSelect => {
                self.drag = Drag::ObjectSelect { start: (ix, iy), cur: (ix, iy), lasso: vec![[ix, iy]], op };
            }
            _ => {}
        }
        let _ = (window, cx);
    }

    // ----- polygonal lasso -----------------------------------------------------------

    fn poly_click(&mut self, ix: f32, iy: f32, double: bool, op: SelOp) {
        let s = self.vp.logical_scale(self.sf);
        match &mut self.poly {
            None => self.poly = Some(PolyState { pts: vec![[ix, iy]], op }),
            Some(st) => {
                let first = st.pts[0];
                let near_start = ((first[0] - ix).powi(2) + (first[1] - iy).powi(2)).sqrt() * s < 8.0;
                if near_start || double {
                    if !near_start && !double {
                        st.pts.push([ix, iy]);
                    }
                    self.finish_poly();
                } else {
                    st.pts.push([ix, iy]);
                }
            }
        }
    }

    pub fn finish_poly(&mut self) {
        if let Some(st) = self.poly.take() {
            let sel = (st.pts.len() >= 3).then(|| Selection { shape: Shape::Poly(Arc::new(st.pts)), inverted: false });
            self.commit_selection("Polygonal Lasso", sel, st.op, true);
        }
    }

    // ----- magnetic lasso ---------------------------------------------------------------

    fn wire_params(&self) -> WireParams {
        WireParams { width: self.opts.magnetic_width.max(2.0), contrast: self.opts.magnetic_contrast / 100.0 }
    }

    fn magnetic_click(&mut self, ix: f32, iy: f32, double: bool, op: SelOp) {
        let q = p(ix, iy);
        let s = self.vp.logical_scale(self.sf);
        let close = match &self.magnetic {
            Some(st) => st.anchors.first().is_some_and(|a| a.dist(q) * s < 8.0) || double,
            None => false,
        };
        if close {
            self.finish_magnetic();
            return;
        }
        match &mut self.magnetic {
            None => self.magnetic = Some(MagState { committed: vec![q], anchors: vec![q], trail: vec![q], live: vec![], op }),
            Some(st) => {
                // Manual anchor: commit the live wire up to the click.
                let live = std::mem::take(&mut st.live);
                st.committed.extend(live.into_iter().skip(1));
                st.committed.push(q);
                st.anchors.push(q);
                st.trail = vec![q];
            }
        }
    }

    /// Updates the live wire as the cursor moves (no button needed).
    pub fn magnetic_hover(&mut self, ix: f32, iy: f32) {
        let params = self.wire_params();
        let freq = self.opts.magnetic_frequency;
        let li = self.doc.active;
        let Some(st) = &mut self.magnetic else { return };
        let q = p(ix, iy);
        if st.trail.last().is_some_and(|l| l.dist(q) < 1.0) {
            return;
        }
        st.trail.push(q);
        let t0 = Instant::now();
        let layer = &self.doc.layers[li];
        st.live = livewire::trace(layer, &st.trail, &params);
        // Frequency: automatically anchor once the wire is long enough.
        let spacing = 20.0 + (100.0 - freq) * 4.0;
        let len: f32 = st.live.windows(2).map(|w| w[0].dist(w[1])).sum();
        if len > spacing && st.live.len() > 8 {
            let keep = st.live.len() * 2 / 3;
            let anchor = st.live[keep];
            st.committed.extend(st.live[1..=keep].iter().copied());
            st.anchors.push(anchor);
            let rest: Vec<P> = st.trail.iter().copied().skip_while(|t| t.dist(anchor) > params.width * 0.5).collect();
            st.trail = std::iter::once(anchor).chain(rest.into_iter().skip(1)).collect();
            if st.trail.len() < 2 {
                st.trail.push(q);
            }
            st.live = livewire::trace(layer, &st.trail, &params);
        }
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    pub fn finish_magnetic(&mut self) {
        if let Some(mut st) = self.magnetic.take() {
            st.committed.extend(st.live.into_iter().skip(1));
            // Close back to the first anchor through the wire.
            if let (Some(&first), Some(&last)) = (st.anchors.first(), st.committed.last()) {
                let layer = &self.doc.layers[self.doc.active];
                let back = livewire::trace(layer, &[last, first], &WireParams { width: self.opts.magnetic_width.max(2.0), contrast: self.opts.magnetic_contrast / 100.0 });
                st.committed.extend(back.into_iter().skip(1));
            }
            let pts: Vec<[f32; 2]> = st.committed.iter().map(|q| [q.x, q.y]).collect();
            let sel = (pts.len() >= 3).then(|| Selection { shape: Shape::Poly(Arc::new(pts)), inverted: false });
            self.commit_selection("Magnetic Lasso", sel, st.op, true);
        }
    }

    // ----- magic wand ---------------------------------------------------------------------

    pub fn magic_wand(&mut self, x: f32, y: f32, op: SelOp) {
        let t = Instant::now();
        let (xi, yi) = (x.floor() as i32, y.floor() as i32);
        let tol = self.tolerance as u8;
        let mut mask = if self.opts.sample_all {
            let src = ops::CompositeSource::new(&self.doc.layers, self.doc.w, self.doc.h);
            ops::flood_mask_on(&src, xi, yi, tol, self.contiguous)
        } else {
            ops::flood_mask(&self.doc.layers[self.doc.active].grid, xi, yi, tol, self.contiguous)
        };
        if self.opts.anti_alias {
            // A half-pixel blur of the edge only (uniform tiles are skipped).
            mask = mask.feather(0.55);
        }
        let new = mask.to_selection();
        // Wand results are already masks: feather is not applied twice.
        let feather = std::mem::replace(&mut self.opts.feather, 0.0);
        self.commit_selection("Magic Wand", new, op, false);
        self.opts.feather = feather;
        self.last_tool_ms = t.elapsed().as_secs_f32() * 1000.0;
        self.flash(format!("Magic Wand — {:.1} ms", self.last_tool_ms));
    }

    // ----- quick selection -------------------------------------------------------------------

    /// Re-solves the quick selection at the coarse level (throttled).
    pub fn quick_select_update(&mut self, force: bool) {
        let Drag::QuickSelect(q) = &mut self.drag else { return };
        if !force && (q.last_solve.elapsed().as_millis() < 40 || !q.dirty) {
            return;
        }
        let t0 = Instant::now();
        let (w, h) = (self.doc.w, self.doc.h);
        let (layers, _) = if self.opts.sample_all { (self.doc.layers.clone(), 0) } else { (vec![self.doc.layers[self.doc.active].clone()], 0) };
        let src = move |l: u32, x: i32, y: i32, bw: usize, bh: usize, out: &mut [u32]| compose_src(&layers, l, x, y, bw, bh, out);
        let cur = q.base.as_ref().map(|s| MaskGrid::from_selection(s, w, h));
        if let Some(c) = q.qs.solve(&src, w, h, if q.op == SelOp::Add { cur.as_ref() } else { None }) {
            let m = segment::upsample(&c, w, h, None);
            self.doc.selection = combine_selection(q.base.as_ref(), m.to_selection(), q.op, w, h);
        }
        q.last_solve = Instant::now();
        q.dirty = false;
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// Mouse-up: refine at full resolution in the background.
    pub fn quick_select_finish(&mut self, q: Box<QuickDrag>, cx: &mut Context<Self>) {
        let (w, h) = (self.doc.w, self.doc.h);
        let (layers, _) = self.seg_layers();
        let base = q.base.clone();
        let op = q.op;
        let cur = base.as_ref().map(|s| MaskGrid::from_selection(s, w, h));
        let qs = q.qs;
        // Undo restores the selection from before the gesture.
        let live = std::mem::replace(&mut self.doc.selection, base.clone());
        self.doc.push_undo("Quick Selection");
        self.doc.selection = live;
        self.spawn_job("Quick Selection (refine)", cx, move |_| {
            let src = |l: u32, x: i32, y: i32, bw: usize, bh: usize, out: &mut [u32]| compose_src(&layers, l, x, y, bw, bh, out);
            let refined = qs.solve(&src, w, h, if op == SelOp::Add { cur.as_ref() } else { None }).map(|c| segment::upsample(&c, w, h, Some(&src)));
            Box::new(move |ed: &mut Editor, _| {
                if let Some(m) = refined {
                    ed.doc.selection = combine_selection(base.as_ref(), m.to_selection(), op, w, h);
                }
            })
        });
    }

    // ----- object selection ---------------------------------------------------------------------

    pub fn object_select_finish(&mut self, start: (f32, f32), cur: (f32, f32), lasso: Vec<[f32; 2]>, op: SelOp, cx: &mut Context<Self>) {
        let (w, h) = (self.doc.w, self.doc.h);
        let moved = (cur.0 - start.0).abs().max((cur.1 - start.1).abs()) * self.vp.logical_scale(self.sf);
        if moved < 3.0 {
            if self.opts.object_finder && !self.object_click(start.0, start.1, op, cx) {
                self.flash("No object under the cursor (drag a box or lasso instead)");
            }
            return;
        }
        let use_lasso = self.opts.object_lasso && lasso.len() >= 3;
        let pts: Vec<P> = lasso.iter().map(|q| p(q[0], q[1])).collect();
        let rect = if use_lasso {
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for q in &pts {
                x0 = x0.min(q.x);
                y0 = y0.min(q.y);
                x1 = x1.max(q.x);
                y1 = y1.max(q.y);
            }
            RectI::new(x0 as i32, y0 as i32, x1 as i32, y1 as i32)
        } else {
            RectI::new(start.0.min(cur.0) as i32, start.1.min(cur.1) as i32, start.0.max(cur.0) as i32, start.1.max(cur.1) as i32)
        };
        let (layers, _) = self.seg_layers();
        let max_level = self.doc.max_level();
        if !use_lasso && crate::engine::ml::ready(crate::engine::ml::SUBJECT_MODEL) {
            let base = self.doc.selection.clone();
            self.spawn_job("Object Selection (model)", cx, move |_| {
                let src = |l: u32, x: i32, y: i32, bw: usize, bh: usize, out: &mut [u32]| compose_src(&layers, l, x, y, bw, bh, out);
                let m = crate::engine::objects::object_in_box(&src, w, h, max_level, rect);
                Box::new(move |ed: &mut Editor, _| {
                    ed.doc.push_undo("Object Selection");
                    ed.doc.selection = combine_selection(base.as_ref(), m.and_then(|m| m.to_selection()), op, w, h);
                })
            });
            return;
        }
        // First result at ~0.12 MP right away, refined at ~1 MP in the background.
        let t0 = Instant::now();
        let src = |l: u32, x: i32, y: i32, bw: usize, bh: usize, out: &mut [u32]| compose_src(&layers, l, x, y, bw, bh, out);
        let coarse = segment::object_select(&src, w, h, max_level, rect, use_lasso.then_some(&pts[..]), 0.12e6, false);
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let base = self.doc.selection.clone();
        self.doc.push_undo("Object Selection");
        if let Some(m) = coarse {
            self.doc.selection = combine_selection(base.as_ref(), m.to_selection(), op, w, h);
        }
        let first_ms = self.last_tool_ms;
        self.spawn_job("Object Selection (refine)", cx, move |_| {
            let src = |l: u32, x: i32, y: i32, bw: usize, bh: usize, out: &mut [u32]| compose_src(&layers, l, x, y, bw, bh, out);
            let t1 = Instant::now();
            let fine = segment::object_select(&src, w, h, max_level, rect, use_lasso.then_some(&pts[..]), 1.0e6, true);
            let ms = t1.elapsed().as_secs_f32() * 1000.0;
            Box::new(move |ed: &mut Editor, _| {
                if let Some(m) = fine {
                    ed.doc.selection = combine_selection(base.as_ref(), m.to_selection(), op, w, h);
                }
                ed.last_tool_ms = ms;
                eprintln!("object selection: first {first_ms:.0} ms, refined {ms:.0} ms");
            })
        });
    }

    /// Select ▸ Subject: the salient-object model when one is installed,
    /// otherwise GrabCut seeded with the image inset by 4%.
    pub fn select_subject(&mut self, cx: &mut Context<Self>) {
        let (layers, _) = self.seg_layers();
        let max_level = self.doc.max_level();
        let (dw, dh) = (self.doc.w, self.doc.h);
        self.spawn_job("Select Subject", cx, move |_| {
            let src = |l: u32, x: i32, y: i32, bw: usize, bh: usize, out: &mut [u32]| compose_src(&layers, l, x, y, bw, bh, out);
            let (m, backend) = crate::engine::objects::select_subject(&src, dw, dh, max_level);
            Box::new(move |ed: &mut Editor, _| {
                ed.doc.push_undo("Select Subject");
                ed.doc.selection = m.and_then(|m| m.to_selection());
                if backend == crate::engine::objects::Backend::GrabCut {
                    ed.flash(crate::engine::ml::describe(crate::engine::ml::SUBJECT_MODEL, "Select Subject", "classical GrabCut"));
                }
            })
        });
    }

    // ----- object finder ---------------------------------------------------------------------------

    /// Identity of what the finder was built from (pixels + sampling).
    fn finder_key(&self) -> u64 {
        let mut k = self.doc.generation.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.opts.sample_all as u64;
        let ids: Vec<usize> = if self.opts.sample_all { (0..self.doc.layers.len()).collect() } else { vec![self.doc.active] };
        for i in ids {
            let l = &self.doc.layers[i];
            k = (k ^ l.id.wrapping_mul(31) ^ l.version.wrapping_mul(1_000_003)).rotate_left(7);
        }
        k
    }

    /// Starts building the hover object set in the background (no busy HUD:
    /// the tool stays usable meanwhile) unless it is current or on its way.
    pub fn ensure_object_finder(&mut self, cx: &mut Context<Self>) {
        if !self.opts.object_finder {
            return;
        }
        let key = self.finder_key();
        if self.finder.as_ref().is_some_and(|f| f.0 == key) || self.finder_building == Some(key) {
            return;
        }
        self.finder_building = Some(key);
        let (layers, _) = self.seg_layers();
        let max_level = self.doc.max_level();
        let (w, h) = (self.doc.w, self.doc.h);
        cx.spawn(async move |this, cx| {
            let f = cx
                .background_executor()
                .spawn(async move {
                    let src = |l: u32, x: i32, y: i32, bw: usize, bh: usize, out: &mut [u32]| compose_src(&layers, l, x, y, bw, bh, out);
                    crate::engine::objects::ObjectFinder::build(&src, w, h, max_level)
                })
                .await;
            let _ = this.update(cx, |ed, cx| {
                if ed.finder_building == Some(key) {
                    ed.finder_building = None;
                }
                eprintln!("object finder: {} objects in {:.0} ms", f.objects.len(), f.build_ms);
                ed.finder = Some((key, Arc::new(f)));
                cx.notify();
            });
        })
        .detach();
    }

    /// The hovered object: translucent blue tint plus outline (Photoshop's
    /// object finder look). The tint is the finder's label image for that
    /// object, built once and stretched over the document.
    pub fn object_overlays(&self, out: &mut Vec<super::editor::Overlay>) {
        if self.tool != Tool::ObjectSelect || !matches!(self.drag, Drag::None) {
            return;
        }
        let (Some(id), Some((_, f))) = (self.hover_object, &self.finder) else { return };
        let Some(o) = f.objects.get(id) else { return };
        if self.vp.rot == 0.0 {
            let mut cache = self.hover_tint.borrow_mut();
            if cache.as_ref().is_none_or(|(i, _)| *i != id) {
                let px_ = f.tint_bgra(id, [40, 120, 255], 90);
                let img = image::RgbaImage::from_raw(f.w as u32, f.h as u32, px_).expect("tint size");
                *cache = Some((id, crate::ui::render_image(vec![image::Frame::new(img)])));
            }
            let s = (1u32 << f.level) as f32;
            let a = self.vp.to_screen(0.0, 0.0, self.canvas, self.sf);
            let b = self.vp.to_screen(f.w as f32 * s, f.h as f32 * s, self.canvas, self.sf);
            out.push(super::editor::Overlay::Image { bounds: gpui_kit::Bounds::from_corners(a, b), img: cache.as_ref().unwrap().1.clone(), radius: 0.0 });
        }
        let polys: Vec<(Vec<gpui_kit::Point<gpui_kit::Pixels>>, bool)> =
            o.outline.iter().map(|pl| (pl.iter().map(|q| self.vp.to_screen(q[0], q[1], self.canvas, self.sf)).collect(), true)).collect();
        out.push(super::editor::Overlay::Path { polys, dashed: false, color: gpui_kit::hsla(0.6, 0.9, 0.6, 0.95), width: 2.0 });
    }

    /// Hover: the object under the cursor (a label lookup).
    pub fn object_hover(&mut self, ix: f32, iy: f32, cx: &mut Context<Self>) {
        if !self.opts.object_finder {
            return;
        }
        let t0 = Instant::now();
        let key = self.finder_key();
        let hit = match &self.finder {
            Some((k, f)) if *k == key => f.at(ix, iy),
            _ => {
                self.ensure_object_finder(cx);
                None
            }
        };
        if hit != self.hover_object {
            self.hover_object = hit;
            if let (Some(id), Some((_, f))) = (hit, &self.finder) {
                let prior = match f.backend {
                    crate::engine::objects::Backend::Neural => "neural",
                    crate::engine::objects::Backend::GrabCut => "classical",
                };
                self.flash(format!("Object {} · {:.1}% of the image ({prior} finder) · click to select", id + 1, f.objects[id].coverage * 100.0));
            }
            cx.notify();
        }
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// Click (no drag) with the finder on: selects the hovered object.
    fn object_click(&mut self, ix: f32, iy: f32, op: SelOp, cx: &mut Context<Self>) -> bool {
        let Some((_, f)) = self.finder.clone().filter(|(k, _)| *k == self.finder_key()) else { return false };
        let Some(id) = f.at(ix, iy) else { return false };
        let (layers, _) = self.seg_layers();
        let (w, h) = (self.doc.w, self.doc.h);
        let base = self.doc.selection.clone();
        self.spawn_job("Object Selection", cx, move |_| {
            let src = |l: u32, x: i32, y: i32, bw: usize, bh: usize, out: &mut [u32]| compose_src(&layers, l, x, y, bw, bh, out);
            let m = f.mask(id, &src, w, h);
            Box::new(move |ed: &mut Editor, _| {
                ed.doc.push_undo("Object Selection");
                ed.doc.selection = combine_selection(base.as_ref(), m.and_then(|m| m.to_selection()), op, w, h);
            })
        });
        true
    }

    // ----- Select ▸ Modify -------------------------------------------------------------------------

    pub fn apply_modify(&mut self, kind: Modify, amount: f32, cx: &mut Context<Self>) {
        let Some(sel) = self.doc.selection.clone() else {
            self.flash("No selection to modify");
            return;
        };
        let (w, h) = (self.doc.w, self.doc.h);
        self.spawn_job(kind.title(), cx, move |_| {
            let m = MaskGrid::from_selection(&sel, w, h);
            let out = match kind {
                Modify::Border => m.border(amount),
                Modify::Smooth => m.smooth(amount.round() as usize),
                Modify::Expand => m.expand(amount),
                Modify::Contract => m.contract(amount),
                Modify::Feather => m.feather(amount),
            };
            Box::new(move |ed: &mut Editor, _| {
                ed.doc.push_undo(kind.title());
                ed.doc.selection = out.to_selection();
            })
        });
    }

    // ----- Quick Mask ---------------------------------------------------------------------------------

    pub fn toggle_quick_mask(&mut self, cx: &mut Context<Self>) {
        let (w, h) = (self.doc.w, self.doc.h);
        match self.quick_mask.take() {
            None => {
                let m = match &self.doc.selection {
                    Some(s) => MaskGrid::from_selection(s, w, h),
                    None => MaskGrid::full(w, h),
                };
                let grid = m.to_gray();
                let mut ov = TileGrid::new(w, h);
                ov.tiles = grid.tiles.iter().map(masks::quick_mask_overlay_tile).collect();
                self.quick_mask = Some(QuickMask { grid, overlay: Layer::new(0, "Quick Mask", ov) });
                self.flash("Quick Mask: paint black to mask, white to select; press Q to exit");
            }
            Some(qm) => {
                let m = MaskGrid::from_gray(&qm.grid);
                self.doc.push_undo("Quick Mask");
                self.doc.selection = m.to_selection();
                self.flash("Quick Mask converted to selection");
            }
        }
        self.doc.mark_all_dirty();
        cx.notify();
    }

    /// Refreshes the red overlay after the gray mask changed in `rects`.
    pub fn quick_mask_changed(&mut self, rects: &[RectI]) {
        let Some(qm) = &mut self.quick_mask else { return };
        let (cols, rows) = (qm.grid.cols, qm.grid.rows);
        let mut seen = std::collections::HashSet::new();
        for r in rects {
            let (tx0, ty0, tx1, ty1) = r.tile_span(cols, rows);
            for ty in ty0..ty1 {
                for tx in tx0..tx1 {
                    let i = qm.grid.idx(tx, ty);
                    if seen.insert(i) {
                        qm.overlay.grid.tiles[i] = masks::quick_mask_overlay_tile(&qm.grid.tiles[i]);
                    }
                }
            }
        }
        crate::engine::mips::update_mips(&qm.overlay.grid, &mut qm.overlay.mips, rects);
        self.doc.dirty.rects.extend_from_slice(rects);
    }

    /// Gray value painted in Quick Mask mode for a color.
    pub fn quick_mask_gray(c: [u8; 3]) -> [u8; 3] {
        let l = ((c[0] as u32 * 77 + c[1] as u32 * 150 + c[2] as u32 * 29) >> 8) as u8;
        [l, l, l]
    }
}

/// Region reader over a layer stack (one layer = its own pixels).
pub fn compose_src(layers: &[Layer], level: u32, x: i32, y: i32, w: usize, h: usize, out: &mut [u32]) {
    if layers.len() == 1 {
        layers[0].level(level).read_region_clamped(x, y, w, h, out);
        return;
    }
    let input = crate::engine::composite::ComposeInput { layers, preview: None, extra: None };
    let mut s = crate::engine::composite::Scratch::default();
    crate::engine::composite::compose_region(&input, level, x, y, w, h, out, &mut s);
    let _ = px::alpha;
}
