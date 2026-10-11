//! Multiple documents: tabs, 2-up and tiled arrangements, and Zoom All /
//! Scroll All / Match Zoom · Location · Rotation (Photoshop's Window ▸
//! Arrange). The active document's state lives in the editor's own fields
//! (so every tool works unchanged); the other documents wait in
//! [`DocSlot`]s and are swapped in when activated. Background jobs remember
//! the document they started on and apply to it even after a tab switch.

use std::collections::HashMap;
use std::sync::Arc;

use gpui_kit::{Bounds, Context, Pixels, RenderImage, Window, point, px};

use crate::engine::document::Document;
use crate::engine::grid::RectI;

use super::editor::{Drag, Editor};
use super::viewport::{DisplayCache, FramePlan, Viewport};

/// How the open documents share the canvas area.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Arrange {
    /// One document at a time, the others behind tabs.
    #[default]
    Tabs,
    /// The active document and the next one side by side.
    TwoUp,
    /// Every document in a grid.
    Tile,
}

impl Arrange {
    pub const ALL: [Arrange; 3] = [Arrange::Tabs, Arrange::TwoUp, Arrange::Tile];

    pub fn label(self) -> &'static str {
        match self {
            Arrange::Tabs => "Tabs",
            Arrange::TwoUp => "2-up",
            Arrange::Tile => "Tile",
        }
    }
}

/// Window ▸ Arrange ▸ Match ….
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Match {
    Zoom,
    Location,
    Rotation,
    All,
}

/// Per-document Navigation state (crop frame, ruler, selections of
/// annotations).
#[derive(Default)]
pub struct NavDocState {
    pub crop_angle: f32,
    pub ruler: Option<((f32, f32), (f32, f32))>,
    pub ruler2: Option<(f32, f32)>,
    pub pcrop: Vec<(f32, f32)>,
    pub slice_sel: Option<usize>,
    pub artboard_sel: Option<u64>,
    pub note_sel: Option<usize>,
}

type Thumbs = HashMap<u64, (u64, Arc<RenderImage>)>;

/// A document that isn't active, with its view.
pub struct DocSlot {
    pub uid: u64,
    pub doc: Document,
    pub vp: Viewport,
    pub cache: DisplayCache,
    pub canvas: Bounds<Pixels>,
    pub needs_fit: bool,
    pub thumbs: Thumbs,
    pub mask_thumbs: Thumbs,
    pub crop: Option<RectI>,
    pub mask_edit: bool,
    pub history_seen: usize,
    pub nav: NavDocState,
}

impl DocSlot {
    fn new(uid: u64, doc: Document) -> Self {
        DocSlot {
            uid,
            doc,
            vp: Viewport::default(),
            cache: DisplayCache::default(),
            canvas: Bounds::default(),
            needs_fit: true,
            thumbs: HashMap::new(),
            mask_thumbs: HashMap::new(),
            crop: None,
            mask_edit: false,
            history_seen: 0,
            nav: NavDocState::default(),
        }
    }
}

pub struct Docs {
    /// The active document's id.
    pub uid: u64,
    pub slots: Vec<DocSlot>,
    /// Tab order (document ids).
    pub order: Vec<u64>,
    pub arrange: Arrange,
    /// Zoom All Windows / Scroll All Windows.
    pub zoom_all: bool,
    pub pan_all: bool,
    next: u64,
    /// The active view when the other views were last synced.
    last_vp: Option<Viewport>,
    /// A modified document whose close was asked once (the next close discards).
    close_armed: Option<u64>,
}

impl Default for Docs {
    fn default() -> Self {
        Docs { uid: 1, slots: Vec::new(), order: vec![1], arrange: Arrange::Tabs, zoom_all: false, pan_all: false, next: 2, last_vp: None, close_armed: None }
    }
}

impl Docs {
    pub fn count(&self) -> usize {
        self.order.len()
    }

    /// Documents shown in panes, in tab order (the active one included).
    pub fn panes(&self) -> Vec<u64> {
        match self.arrange {
            _ if self.order.len() < 2 => vec![self.uid],
            Arrange::Tabs => vec![self.uid],
            Arrange::Tile => self.order.clone(),
            Arrange::TwoUp => {
                let i = self.order.iter().position(|&u| u == self.uid).unwrap_or(0);
                let j = (i + 1) % self.order.len();
                let (a, b) = (self.order[i.min(j)], self.order[i.max(j)]);
                vec![a, b]
            }
        }
    }

    /// Grid shape for `n` tiled panes: (columns, rows).
    pub fn grid(n: usize) -> (usize, usize) {
        let cols = (n as f32).sqrt().ceil().max(1.0) as usize;
        (cols, n.div_ceil(cols))
    }
}

/// A document's tab title.
pub fn title(doc: &Document, uid: u64) -> String {
    let name = doc.path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| format!("Untitled-{uid}"));
    if doc.modified { format!("{name} •") } else { name }
}

impl Editor {
    /// Exchanges the editor's active-document state with `s`.
    fn swap_slot(&mut self, s: &mut DocSlot) {
        use std::mem::swap;
        swap(&mut self.docs.uid, &mut s.uid);
        swap(&mut self.doc, &mut s.doc);
        swap(&mut self.vp, &mut s.vp);
        swap(&mut self.cache, &mut s.cache);
        swap(&mut self.canvas, &mut s.canvas);
        swap(&mut self.needs_fit, &mut s.needs_fit);
        swap(&mut self.thumbs, &mut s.thumbs);
        swap(&mut self.mask_thumbs, &mut s.mask_thumbs);
        swap(&mut self.crop, &mut s.crop);
        swap(&mut self.mask_edit, &mut s.mask_edit);
        swap(&mut self.history_seen, &mut s.history_seen);
        let n = &mut self.nopts;
        swap(&mut n.crop_angle, &mut s.nav.crop_angle);
        swap(&mut n.ruler, &mut s.nav.ruler);
        swap(&mut n.ruler2, &mut s.nav.ruler2);
        swap(&mut n.pcrop, &mut s.nav.pcrop);
        swap(&mut n.slice_sel, &mut s.nav.slice_sel);
        swap(&mut n.artboard_sel, &mut s.nav.artboard_sel);
        swap(&mut n.note_sel, &mut s.nav.note_sel);
    }

    /// Runs `f` with document `uid` active (swapped in and back out when it
    /// isn't); `None` when it was closed.
    pub fn with_doc<R>(&mut self, uid: u64, f: impl FnOnce(&mut Editor) -> R) -> Option<R> {
        if uid == self.docs.uid {
            return Some(f(self));
        }
        let i = self.docs.slots.iter().position(|s| s.uid == uid)?;
        let mut s = self.docs.slots.remove(i);
        self.swap_slot(&mut s);
        let r = f(self);
        self.swap_slot(&mut s);
        self.docs.slots.insert(i, s);
        Some(r)
    }

    /// Why the active document can't be left right now.
    fn switch_blocker(&self) -> Option<&'static str> {
        if self.busy.is_some() {
            Some("Wait for the current operation to finish")
        } else if self.caf.is_some() {
            Some("Apply or cancel Content-Aware Fill first")
        } else if self.liquify_session.is_some() {
            Some("Apply or cancel Liquify first")
        } else if self.vpw.is_some() {
            Some("Apply or cancel Vanishing Point first")
        } else if self.pww.is_some() {
            Some("Apply or cancel Perspective Warp first")
        } else if self.quick_mask.is_some() {
            Some("Leave Quick Mask mode first")
        } else if self.float.is_some() || self.dialog.is_some() {
            Some("Finish the current edit first")
        } else {
            None
        }
    }

    /// Ends per-document interactions before another document comes in.
    fn leave_document(&mut self) {
        self.text_edit = None;
        self.drag = Drag::None;
        self.ndrag = None;
        self.vdrag = None;
        self.poly = None;
        self.magnetic = None;
        self.pen_sub = None;
        self.web = None;
        self.slice_fields = None;
        self.finder = None;
        self.hover_object = None;
        self.clone_preview = None;
        *self.outline_cache.borrow_mut() = None;
    }

    pub fn activate_doc(&mut self, uid: u64, window: &mut Window, cx: &mut Context<Self>) {
        if uid == self.docs.uid {
            return;
        }
        if let Some(why) = self.switch_blocker() {
            self.flash(why);
            cx.notify();
            return;
        }
        let Some(i) = self.docs.slots.iter().position(|s| s.uid == uid) else { return };
        self.leave_document();
        let mut s = self.docs.slots.remove(i);
        self.swap_slot(&mut s);
        self.docs.slots.insert(i, s);
        self.docs.last_vp = Some(self.vp);
        self.docs.close_armed = None;
        self.sync_layer_controls(window, cx);
        cx.notify();
    }

    /// The next (or previous) tab.
    pub fn cycle_doc(&mut self, back: bool, window: &mut Window, cx: &mut Context<Self>) {
        let o = &self.docs.order;
        let i = o.iter().position(|&u| u == self.docs.uid).unwrap_or(0);
        let j = if back { (i + o.len() - 1) % o.len() } else { (i + 1) % o.len() };
        let uid = o[j];
        self.activate_doc(uid, window, cx);
    }

    /// Opens `doc` in a new tab (or in place of an untouched empty one).
    pub fn open_document(&mut self, doc: Document) {
        let pristine = !self.doc.modified && self.doc.path.is_none() && self.doc.history_len() == 0;
        if pristine || self.switch_blocker().is_some() {
            self.replace_document(doc);
            return;
        }
        self.leave_document();
        let uid = self.docs.next;
        self.docs.next += 1;
        let mut s = DocSlot::new(uid, doc);
        self.swap_slot(&mut s);
        self.docs.slots.push(s);
        self.docs.order.push(uid);
        self.docs.last_vp = None;
        self.needs_fit = true;
    }

    /// Closes document `uid`. A modified document needs a second close
    /// (the first one warns); the last document stays open.
    pub fn close_doc(&mut self, uid: u64, window: &mut Window, cx: &mut Context<Self>) {
        if self.docs.count() < 2 {
            self.flash("The last document stays open");
            return;
        }
        let modified = if uid == self.docs.uid { self.doc.modified } else { self.docs.slots.iter().any(|s| s.uid == uid && s.doc.modified) };
        if modified && self.docs.close_armed != Some(uid) {
            self.docs.close_armed = Some(uid);
            self.flash("Unsaved changes — close again to discard them");
            cx.notify();
            return;
        }
        if uid == self.docs.uid {
            if let Some(why) = self.switch_blocker() {
                self.flash(why);
                return;
            }
            let o = &self.docs.order;
            let i = o.iter().position(|&u| u == uid).unwrap_or(0);
            let next = if i + 1 < o.len() { o[i + 1] } else { o[i - 1] };
            self.activate_doc(next, window, cx);
        }
        if let Some(i) = self.docs.slots.iter().position(|s| s.uid == uid) {
            let mut s = self.docs.slots.remove(i);
            // Its GPU images are released with the active cache's next flush.
            for img in s.cache.take_images() {
                self.cache.queue_drop_tile(img);
            }
            for img in s.thumbs.drain().map(|(_, (_, i))| i).chain(s.mask_thumbs.drain().map(|(_, (_, i))| i)) {
                self.cache.queue_drop(img);
            }
        }
        self.docs.order.retain(|&u| u != uid);
        self.docs.close_armed = None;
        cx.notify();
    }

    /// Titles of the open documents in tab order: (id, title, active).
    pub fn doc_tabs(&self) -> Vec<(u64, String, bool)> {
        self.docs
            .order
            .iter()
            .map(|&u| {
                let t = if u == self.docs.uid { title(&self.doc, u) } else { self.docs.slots.iter().find(|s| s.uid == u).map_or_else(String::new, |s| title(&s.doc, u)) };
                (u, t, u == self.docs.uid)
            })
            .collect()
    }

    pub fn doc_title(&self, uid: u64) -> String {
        self.doc_tabs().into_iter().find(|t| t.0 == uid).map_or_else(String::new, |t| t.1)
    }

    pub fn set_arrange(&mut self, a: Arrange) {
        self.docs.arrange = a;
        // Every pane refits to its new size.
        self.needs_fit = true;
        for s in &mut self.docs.slots {
            s.needs_fit = true;
        }
        self.docs.last_vp = None;
    }

    /// Plans an inactive pane's frame (its own tile cache and view).
    pub fn prepare_slot(&mut self, uid: u64, bounds: Bounds<Pixels>, window: &mut Window) -> Option<FramePlan> {
        let sf = window.scale_factor();
        let s = self.docs.slots.iter_mut().find(|s| s.uid == uid)?;
        s.canvas = bounds;
        if s.needs_fit && bounds.size.width > px(10.) {
            s.vp.fit(s.doc.w, s.doc.h, bounds, sf);
            s.needs_fit = false;
        }
        Some(s.cache.prepare(&mut s.doc, None, &s.vp, bounds, sf, window))
    }

    /// The image point at the centre of the view, as a fraction of the image.
    fn view_center(vp: &Viewport, canvas: Bounds<Pixels>, sf: f32, w: u32, h: u32) -> (f32, f32) {
        let (x, y) = vp.to_image(canvas.center(), canvas, sf);
        (x / w as f32, y / h as f32)
    }

    /// Pans `vp` so the image fraction `c` sits at the view centre.
    fn center_on(vp: &mut Viewport, canvas: Bounds<Pixels>, sf: f32, w: u32, h: u32, c: (f32, f32)) {
        let s = vp.logical_scale(sf);
        let (cw, ch) = (f32::from(canvas.size.width), f32::from(canvas.size.height));
        vp.pan = point(px(cw * 0.5 - c.0 * w as f32 * s), px(ch * 0.5 - c.1 * h as f32 * s));
    }

    /// Window ▸ Arrange ▸ Match …: the other documents take the active
    /// view's zoom, relative location and/or rotation.
    pub fn match_views(&mut self, what: Match) {
        let (vp, canvas, sf) = (self.vp, self.canvas, self.sf);
        let c = Self::view_center(&vp, canvas, sf, self.doc.w, self.doc.h);
        for s in &mut self.docs.slots {
            let keep = Self::view_center(&s.vp, s.canvas, sf, s.doc.w, s.doc.h);
            if matches!(what, Match::Zoom | Match::All) {
                s.vp.zoom = vp.zoom;
            }
            if matches!(what, Match::Rotation | Match::All) {
                s.vp.rot = vp.rot;
            }
            let at = if matches!(what, Match::Location | Match::All) { c } else { keep };
            Self::center_on(&mut s.vp, s.canvas, sf, s.doc.w, s.doc.h, at);
            s.needs_fit = false;
        }
        self.docs.last_vp = Some(vp);
    }

    /// Zoom All / Scroll All: mirrors the active view's zoom factor and
    /// scroll onto the other documents (called every frame).
    pub fn sync_views(&mut self) {
        let vp = self.vp;
        let Some(last) = self.docs.last_vp.replace(vp) else { return };
        if !(self.docs.zoom_all || self.docs.pan_all) || self.docs.slots.is_empty() {
            return;
        }
        if (vp.zoom - last.zoom).abs() > 1e-6 {
            if self.docs.zoom_all {
                let k = vp.zoom / last.zoom;
                for s in &mut self.docs.slots {
                    let c = s.canvas.center();
                    let z = s.vp.zoom * k;
                    s.vp.zoom_about(z, c, s.canvas);
                }
            }
        } else if self.docs.pan_all && vp.pan != last.pan {
            let d = vp.pan - last.pan;
            for s in &mut self.docs.slots {
                s.vp.pan = s.vp.pan + d;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panes_follow_the_arrangement() {
        let mut d = Docs { order: vec![1, 2, 3, 4, 5], uid: 3, ..Default::default() };
        assert_eq!(d.panes(), vec![3]);
        d.arrange = Arrange::TwoUp;
        assert_eq!(d.panes(), vec![3, 4]);
        d.uid = 5;
        assert_eq!(d.panes(), vec![1, 5]);
        d.arrange = Arrange::Tile;
        assert_eq!(d.panes().len(), 5);
        assert_eq!(Docs::grid(5), (3, 2));
        assert_eq!(Docs::grid(4), (2, 2));
        assert_eq!(Docs::grid(2), (2, 1));
        // One document: always a single pane.
        let one = Docs { arrange: Arrange::Tile, ..Default::default() };
        assert_eq!(one.panes(), vec![1]);
    }

    #[test]
    fn titles_mark_unsaved_documents() {
        let mut doc = Document::new(10, 10, None);
        assert_eq!(title(&doc, 4), "Untitled-4");
        doc.path = Some("/tmp/x/dog.jpg".into());
        doc.modified = true;
        assert_eq!(title(&doc, 4), "dog.jpg •");
    }
}
