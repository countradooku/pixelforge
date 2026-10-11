//! Type tools: horizontal / vertical type and type masks, with on-canvas
//! editing (caret, range selection by Shift + arrows, drag, double-click
//! word, triple-click line, ⌘A), type on a path and area type, and the
//! Character / Paragraph / OpenType / Warp panel. The options bar and the
//! panel apply to the selected range while editing, to the typing style at
//! a bare caret, and to the whole layer otherwise.

use std::sync::Arc;
use std::time::Instant;

use gpui_kit::component::button::Button;
use gpui_kit::component::select::{Select, SelectEvent, SelectState};
use gpui_kit::component::{IndexPath, Sizable as _};
use gpui_kit::prelude::*;
use gpui_kit::{Context, Div, Entity, KeyDownEvent, MouseDownEvent, SharedString, Subscription, Window, div, px, rgb};

use crate::engine::grid::TileGrid;
use crate::engine::layers::LayerKind;
use crate::engine::text::{self, Caps, CharStyle, Composer, LastLine, ParaStyle, Position, TextAlign, TextLayer, TextPath, TextStyle, Warp, WarpStyle};
use crate::engine::vector::{self, FillRule, p};

use super::editor::{Drag, Editor, Overlay, Tool};
use super::options::{label, segmented, spec, toggle};

/// On-canvas text editing session.
#[derive(Clone)]
pub struct TextEdit {
    pub layer_id: u64,
    pub caret: usize,
    /// Selection anchor (equal to `caret` when nothing is selected).
    pub anchor: usize,
    /// Type Mask tools: the text becomes a selection when committed.
    pub mask: bool,
    /// Style for the next typed characters (set from the panels at a bare caret).
    pub typing: Option<CharStyle>,
}

impl TextEdit {
    pub fn range(&self) -> (usize, usize) {
        (self.caret.min(self.anchor), self.caret.max(self.anchor))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TypePanel {
    #[default]
    Character,
    Paragraph,
    OpenType,
    Warp,
    Options,
}

impl TypePanel {
    pub const ALL: [TypePanel; 5] = [TypePanel::Character, TypePanel::Paragraph, TypePanel::OpenType, TypePanel::Warp, TypePanel::Options];
    pub fn label(self) -> &'static str {
        match self {
            TypePanel::Character => "Character",
            TypePanel::Paragraph => "Paragraph",
            TypePanel::OpenType => "OpenType",
            TypePanel::Warp => "Warp",
            TypePanel::Options => "Path / Vertical",
        }
    }
}

pub struct TypeOptions {
    /// Defaults for new text layers.
    pub style: TextStyle,
    pub warp: Warp,
    pub panel: TypePanel,
    pub family_select: Option<(Entity<SelectState<Vec<SharedString>>>, Subscription)>,
    /// Layer-wide restyles within a burst (slider drags) share one undo step.
    last_restyle: Option<(u64, Instant)>,
}

impl Default for TypeOptions {
    fn default() -> Self {
        TypeOptions { style: TextStyle::default(), warp: Warp::default(), panel: TypePanel::default(), family_select: None, last_restyle: None }
    }
}

fn prev_char(s: &str, i: usize) -> usize {
    s[..i].char_indices().next_back().map_or(0, |(k, _)| k)
}

fn next_char(s: &str, i: usize) -> usize {
    s[i..].char_indices().nth(1).map_or(s.len(), |(k, _)| i + k)
}

/// Start of the previous word (⌥←) / end of the next word (⌥→).
fn word_left(s: &str, i: usize) -> usize {
    let b: Vec<(usize, char)> = s[..i].char_indices().collect();
    let mut k = b.len();
    while k > 0 && !b[k - 1].1.is_alphanumeric() {
        k -= 1;
    }
    while k > 0 && b[k - 1].1.is_alphanumeric() {
        k -= 1;
    }
    b.get(k).map_or(i.min(if k == 0 { 0 } else { i }), |c| c.0)
}

fn word_right(s: &str, i: usize) -> usize {
    let mut it = s[i..].char_indices().peekable();
    while it.peek().is_some_and(|c| !c.1.is_alphanumeric()) {
        it.next();
    }
    while it.peek().is_some_and(|c| c.1.is_alphanumeric()) {
        it.next();
    }
    it.peek().map_or(s.len(), |c| i + c.0)
}

impl Editor {
    fn is_type_tool(&self) -> bool {
        matches!(self.tool, Tool::TypeH | Tool::TypeV | Tool::TypeMaskH | Tool::TypeMaskV)
    }

    fn active_text(&self) -> Option<(usize, Arc<TextLayer>)> {
        let li = self.doc.active;
        match &self.doc.layers.get(li)?.kind {
            LayerKind::Text(t) => Some((li, t.clone())),
            _ => None,
        }
    }

    pub fn edited_text(&self) -> Option<(usize, Arc<TextLayer>)> {
        let e = self.text_edit.as_ref()?;
        let li = self.doc.layers.iter().position(|l| l.id == e.layer_id)?;
        match &self.doc.layers[li].kind {
            LayerKind::Text(t) => Some((li, t.clone())),
            _ => None,
        }
    }

    /// Mouse-down for the type tools. Returns true when handled.
    pub fn type_down(&mut self, ix: f32, iy: f32, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) -> bool {
        if !self.is_type_tool() {
            return false;
        }
        let _ = window;
        // Type on a path: grab its start / end marker.
        if let Some(end) = self.path_handle_at(ix, iy) {
            if self.text_edit.is_none() {
                self.doc.push_undo("Move Type on Path");
                self.topts.last_restyle = None;
            }
            self.drag = Drag::TypePathHandle { end };
            return true;
        }
        // Clicking inside the edited / active text layer places the caret.
        if let Some((li, t)) = self.edited_text().or_else(|| self.active_text()) {
            if t.contains_point(ix, iy, 6.0 / self.vp.zoom) {
                let at = t.hit(ix, iy);
                let id = self.doc.layers[li].id;
                if self.text_edit.as_ref().is_none_or(|e| e.layer_id != id) {
                    self.commit_text(cx);
                    self.doc.push_undo("Edit Type");
                    self.text_edit = Some(TextEdit { layer_id: id, caret: at, anchor: at, mask: false, typing: None });
                }
                let e = self.text_edit.as_mut().unwrap();
                e.typing = None;
                match ev.click_count {
                    2 => (e.anchor, e.caret) = t.word_at(at),
                    3 => (e.anchor, e.caret) = t.line_at(at),
                    n if n >= 4 => (e.anchor, e.caret) = (0, t.text.len()),
                    _ if ev.modifiers.shift => e.caret = at,
                    _ => {
                        (e.anchor, e.caret) = (at, at);
                        self.drag = Drag::TypeSelect;
                    }
                }
                cx.notify();
                return true;
            }
        }
        self.commit_text(cx);
        // Near the active path: type on the path; inside a closed path: area type.
        let horizontal = matches!(self.tool, Tool::TypeH | Tool::TypeMaskH);
        if horizontal && let Some(path) = self.doc.active_path.and_then(|i| self.doc.paths.get(i)).map(|p| p.path.clone()) {
            let polys = path.flatten(0.5);
            let q = p(ix, iy);
            let tol = 8.0 / self.vp.zoom;
            if vector::dist_to_polys(q, &polys) <= tol {
                let start = polys.first().map_or(0.0, |pl| {
                    let mut best = (0.0, f32::MAX);
                    let mut acc = 0.0;
                    let n = pl.pts.len();
                    let segs = if pl.closed { n } else { n.saturating_sub(1) };
                    for k in 0..segs {
                        let (a, b) = (pl.pts[k], pl.pts[(k + 1) % n]);
                        let d = vector::dist_to_segment(q, a, b);
                        if d < best.1 {
                            let ab = b - a;
                            let t = ((q - a).dot(ab) / ab.dot(ab).max(1e-9)).clamp(0.0, 1.0);
                            best = (acc + t * ab.len(), d);
                        }
                        acc += a.dist(b);
                    }
                    best.0 / vector::poly_length(pl).max(1e-6)
                });
                // A closed path takes the whole loop from the click; an open one runs to its end.
                let end = if path.subs.first().is_some_and(|sp| sp.closed) { start + 1.0 } else { 1.0 };
                self.create_text(None, (ix, iy), None, Some(TextPath { path, start, end, flip: false, area: false }));
                return true;
            }
            if path.subs.iter().all(|s| s.closed) && vector::contains(q, &polys, FillRule::NonZero) {
                self.create_text(None, (ix, iy), None, Some(TextPath { path, start: 0.0, end: 1.0, flip: false, area: true }));
                return true;
            }
        }
        self.drag = Drag::TypeBox { start: (ix, iy), cur: (ix, iy) };
        true
    }

    /// Image positions of the edited / active path text's start and end markers.
    fn path_handles(&self) -> Option<(Arc<TextLayer>, vector::P, vector::P, vector::P, vector::P)> {
        let (_, t) = self.edited_text().or_else(|| self.active_text().filter(|_| self.is_type_tool()))?;
        let tp = t.path.clone().filter(|tp| !tp.area)?;
        let (s, st) = t.path_at(tp.start)?;
        let (e, et) = t.path_at(tp.end)?;
        Some((t, s, st, e, et))
    }

    /// Which marker (false = start, true = end) is under an image point.
    fn path_handle_at(&self, ix: f32, iy: f32) -> Option<bool> {
        let (_, s, _, e, _) = self.path_handles()?;
        let tol = 9.0 / self.vp.zoom;
        let q = p(ix, iy);
        if q.dist(s) <= tol {
            Some(false)
        } else if q.dist(e) <= tol {
            Some(true)
        } else {
            None
        }
    }

    /// Drags a path-text marker along the path; dragging the start marker
    /// across the path flips the text to the other side (as Photoshop).
    pub fn type_path_handle_move(&mut self, ix: f32, iy: f32, end: bool) {
        let Some((t, ..)) = self.path_handles() else { return };
        let Some(tp) = t.path.clone() else { return };
        let Some((f, side)) = t.path_project(p(ix, iy)) else { return };
        let closed = tp.path.subs.first().is_some_and(|s| s.closed);
        let em = t.style_at(0).size;
        let (mut start, mut stop, mut flip) = (tp.start, tp.end, tp.flip);
        if end {
            stop = if closed { start + (f - start).rem_euclid(1.0) } else { f.max(start) };
            if closed && stop - start < 1e-3 {
                stop = start + 1.0;
            }
        } else {
            let span = if closed { (stop - start).clamp(0.0, 1.0) } else { 0.0 };
            start = if closed { f } else { f.min(stop) };
            if closed {
                // The text keeps its span along the loop.
                stop = start + span;
            }
            // Across the path by half an em: flip to that side.
            let want_flip = side < -0.5 * em;
            let want_unflip = side > 0.5 * em;
            if (want_flip && !flip) || (want_unflip && flip) {
                flip = !flip;
            }
        }
        let new = TextPath { path: tp.path.clone(), start, end: stop, flip, area: false };
        if *tp == new {
            return;
        }
        if self.text_edit.is_some() {
            self.update_text(|t, _| t.path = Some(Arc::new(new.clone())));
        } else if let Some((li, t)) = self.active_text() {
            let mut nt = (*t).clone();
            nt.path = Some(Arc::new(new));
            nt.relayout();
            let kind = LayerKind::Text(Arc::new(nt));
            let dirty = crate::engine::layers::rerender(&mut self.doc.layers[li].grid, Some(t.bounds()), &kind);
            self.doc.layers[li].kind = kind;
            self.doc.layer_pixels_changed(li, &[dirty]);
        }
    }

    /// Drag-selecting inside the edited text.
    pub fn type_select_move(&mut self, ix: f32, iy: f32) {
        let Some((_, t)) = self.edited_text() else { return };
        if let Some(e) = &mut self.text_edit {
            e.caret = t.hit(ix, iy);
        }
    }

    fn create_text(&mut self, vertical: Option<bool>, origin: (f32, f32), box_size: Option<(f32, f32)>, path: Option<TextPath>) {
        let vertical = vertical.unwrap_or(matches!(self.tool, Tool::TypeV | Tool::TypeMaskV));
        let mask = matches!(self.tool, Tool::TypeMaskH | Tool::TypeMaskV);
        let mut style = self.topts.style.clone();
        style.color = if mask { [255, 40, 40] } else { self.fg };
        let mut t = TextLayer::new(String::new(), style, origin, box_size, vertical);
        t.warp = self.topts.warp;
        t.path = path.map(Arc::new);
        t.relayout();
        self.doc.push_undo(if mask { "Type Mask" } else { "Type" });
        let (dw, dh) = (self.doc.w, self.doc.h);
        let at = self.doc.insert_layer(if mask { "Type Mask" } else { "Text" }, TileGrid::new(dw, dh));
        self.doc.layers[at].kind = LayerKind::Text(Arc::new(t));
        let id = self.doc.layers[at].id;
        self.text_edit = Some(TextEdit { layer_id: id, caret: 0, anchor: 0, mask, typing: None });
        self.flash("Type, then press Esc or ⌘↩ to commit");
    }

    /// Mouse-up of a type drag: point text (click) or paragraph text (box).
    pub fn type_box_done(&mut self, start: (f32, f32), cur: (f32, f32)) {
        let (w, h) = ((cur.0 - start.0).abs(), (cur.1 - start.1).abs());
        if w > 8.0 && h > 8.0 {
            self.create_text(None, (start.0.min(cur.0), start.1.min(cur.1)), Some((w, h)), None);
        } else {
            self.create_text(None, start, None, None);
        }
    }

    /// Replaces the edited text layer's description, re-rendering only the
    /// tiles whose glyphs changed.
    fn update_text(&mut self, f: impl FnOnce(&mut TextLayer, &mut TextEdit)) {
        let Some(mut e) = self.text_edit.clone() else { return };
        let Some((li, old)) = self.edited_text() else { return };
        let t0 = Instant::now();
        let mut t = (*old).clone();
        f(&mut t, &mut e);
        e.caret = e.caret.min(t.text.len());
        e.anchor = e.anchor.min(t.text.len());
        if t != *old {
            t.relayout();
            let r = t.rerender_edit(&mut self.doc.layers[li].grid, &old);
            self.doc.layers[li].kind = LayerKind::Text(Arc::new(t));
            if !r.is_empty() {
                self.doc.layer_pixels_changed(li, &[r]);
            }
            self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
        }
        self.text_edit = Some(e);
    }

    /// Inserts `s` at the caret (replacing the selection).
    pub fn type_text(&mut self, s: &str) {
        self.update_text(|t, e| {
            let (a, b) = e.range();
            let style = e.typing.clone().or_else(|| (a < b).then(|| t.style_at(a)));
            t.delete(a, b);
            t.insert(a, s, style);
            e.caret = a + s.len();
            e.anchor = e.caret;
        });
    }

    /// Ends the editing session; Type Mask text becomes a selection.
    pub fn commit_text(&mut self, _cx: &mut Context<Self>) {
        let Some(e) = self.text_edit.take() else { return };
        let Some(li) = self.doc.layers.iter().position(|l| l.id == e.layer_id) else { return };
        let LayerKind::Text(t) = &self.doc.layers[li].kind else { return };
        let empty = t.text.trim().is_empty();
        if e.mask {
            let (w, h) = (self.doc.w, self.doc.h);
            let m = vector::rasterize_mask(&t.outlines(), w, h, FillRule::NonZero);
            self.doc.delete_layer(li);
            self.doc.selection = m.to_selection();
            self.doc.mark_all_dirty();
        } else if empty {
            self.doc.delete_layer(li);
            self.doc.mark_all_dirty();
        } else {
            let name: String = t.text.chars().take(24).collect::<String>().replace('\n', " ");
            self.doc.layers[li].name = name;
        }
    }

    /// Keystrokes while editing text. Returns true when consumed.
    pub fn text_key(&mut self, ev: &KeyDownEvent, cx: &mut Context<Self>) -> bool {
        if self.text_edit.is_none() {
            return false;
        }
        let ks = &ev.keystroke;
        let m = &ks.modifiers;
        let key = ks.key.as_str();
        if key == "escape" || (m.platform && key == "enter") {
            self.commit_text(cx);
            cx.notify();
            return true;
        }
        if m.platform && key == "a" {
            self.update_text(|t, e| (e.anchor, e.caret) = (0, t.text.len()));
            cx.notify();
            return true;
        }
        if m.platform || m.control {
            return false;
        }
        let shift = m.shift;
        let alt = m.alt;
        // Caret motion: Shift extends the selection, otherwise it collapses.
        let motion = |e: &mut TextEdit, to: usize| {
            e.caret = to;
            if !shift {
                e.anchor = to;
            }
            e.typing = None;
        };
        match key {
            "backspace" | "delete" => self.update_text(|t, e| {
                let (mut a, mut b) = e.range();
                if a == b {
                    if key == "backspace" {
                        a = if alt { word_left(&t.text, a) } else { prev_char(&t.text, a) };
                    } else if b < t.text.len() {
                        b = if alt { word_right(&t.text, b) } else { next_char(&t.text, b) };
                    }
                }
                t.delete(a, b);
                (e.caret, e.anchor) = (a, a);
            }),
            "left" | "right" => self.update_text(|t, e| {
                let (a, b) = e.range();
                let to = match (key, alt, shift || a == b) {
                    ("left", true, _) => word_left(&t.text, e.caret),
                    ("right", true, _) => word_right(&t.text, e.caret),
                    ("left", false, true) => prev_char(&t.text, e.caret),
                    ("right", false, true) => next_char(&t.text, e.caret),
                    // Collapse a selection to its edge.
                    ("left", ..) => a,
                    _ => b,
                };
                motion(e, to);
            }),
            "up" | "down" => self.update_text(|t, e| {
                let to = t.line_move(e.caret, if key == "down" { 1 } else { -1 });
                motion(e, to);
            }),
            "home" | "end" => self.update_text(|t, e| {
                let (ls, le) = t.line_at(e.caret);
                motion(e, if key == "home" { ls } else { le });
            }),
            "enter" => self.type_text("\n"),
            _ => {
                let Some(s) = ks.key_char.clone().filter(|s| !s.chars().any(|ch| ch.is_control())) else { return true };
                self.type_text(&s);
            }
        }
        cx.notify();
        true
    }

    // ----- styling --------------------------------------------------------------------------

    /// The character style the panels show: the typing style, the start of
    /// the selection (or the character before the caret), or the layer's.
    pub fn shown_char(&self) -> CharStyle {
        if let (Some(e), Some((_, t))) = (&self.text_edit, self.edited_text()) {
            if let Some(s) = &e.typing {
                return s.clone();
            }
            let (a, b) = e.range();
            return t.style_at(if a < b { a } else { a.saturating_sub(1) });
        }
        if let Some((_, t)) = self.active_text().filter(|_| self.is_type_tool()) {
            return t.style_at(0);
        }
        self.topts.style.char_style()
    }

    pub fn shown_para(&self) -> ParaStyle {
        let t = self.edited_text().or_else(|| self.active_text().filter(|_| self.is_type_tool()));
        match (t, &self.text_edit) {
            (Some((_, t)), Some(e)) => t.paras.get(t.para_index(e.caret)).copied().unwrap_or_default(),
            (Some((_, t)), None) => t.paras.first().copied().unwrap_or_default(),
            _ => self.topts.style.para_style(),
        }
    }

    fn shown_layer(&self) -> Option<Arc<TextLayer>> {
        self.edited_text().or_else(|| self.active_text()).map(|(_, t)| t)
    }

    /// Keeps the tool defaults (for new layers) in step with a style change.
    fn sync_defaults(&mut self, f: &impl Fn(&mut CharStyle)) {
        let mut cs = self.topts.style.char_style();
        f(&mut cs);
        let s = &mut self.topts.style;
        (s.family, s.weight, s.italic, s.size, s.leading, s.tracking, s.color) = (cs.family, cs.weight, cs.italic, cs.size, cs.leading, cs.tracking, cs.color);
    }

    /// Replaces the active (not edited) text layer, as one undo step per burst.
    fn restyle_layer(&mut self, label: &str, f: impl FnOnce(&mut TextLayer)) {
        let Some((li, t)) = self.active_text() else { return };
        let id = self.doc.layers[li].id;
        let burst = self.topts.last_restyle.is_some_and(|(l, at)| l == id && at.elapsed().as_secs_f32() < 1.5) && self.doc.undo_label() == Some(label);
        if !burst {
            self.doc.push_undo(label);
        }
        self.topts.last_restyle = Some((id, Instant::now()));
        let t0 = Instant::now();
        let mut nt = (*t).clone();
        f(&mut nt);
        nt.relayout();
        let old = t.bounds();
        let kind = LayerKind::Text(Arc::new(nt));
        let dirty = crate::engine::layers::rerender(&mut self.doc.layers[li].grid, Some(old), &kind);
        self.doc.layers[li].kind = kind;
        self.doc.layer_pixels_changed(li, &[dirty]);
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// Applies a character-style change to the selection / typing style /
    /// whole layer (see the module docs).
    pub fn apply_char(&mut self, f: impl Fn(&mut CharStyle)) {
        self.sync_defaults(&f);
        if let Some(e) = self.text_edit.clone() {
            let (a, b) = e.range();
            if a < b {
                self.update_text(|t, _| t.set_char_style(a, b, &f));
            } else if let Some((_, t)) = self.edited_text() {
                let mut s = e.typing.clone().unwrap_or_else(|| t.style_at(a.saturating_sub(1)));
                f(&mut s);
                if t.text.is_empty() {
                    // An empty layer's typing style is its default style.
                    self.update_text(|t, _| t.set_char_style_all(&f));
                }
                if let Some(e) = &mut self.text_edit {
                    e.typing = Some(s);
                }
            }
        } else if self.is_type_tool() {
            self.restyle_layer("Character Style", |t| t.set_char_style_all(&f));
        }
    }

    /// Applies a paragraph-style change to the paragraphs of the selection
    /// (or the whole layer when not editing).
    pub fn apply_para(&mut self, f: impl Fn(&mut ParaStyle)) {
        let mut d = self.topts.style.para_style();
        f(&mut d);
        self.topts.style.align = d.align;
        if let Some(e) = self.text_edit.clone() {
            let (a, b) = e.range();
            self.update_text(|t, _| t.set_para_style(a, b, &f));
        } else if self.is_type_tool() {
            self.restyle_layer("Paragraph Style", |t| {
                let n = t.text.len();
                t.set_para_style(0, n, &f);
            });
        }
    }

    /// Layer-wide text options (warp, vertical, path).
    pub fn apply_layer(&mut self, f: impl Fn(&mut TextLayer)) {
        if self.text_edit.is_some() {
            self.update_text(|t, _| f(t));
        } else {
            self.restyle_layer("Text Options", f);
        }
    }

    // ----- overlays -------------------------------------------------------------------------

    pub fn type_overlays(&self, out: &mut Vec<Overlay>) {
        let Some(e) = &self.text_edit else {
            self.path_handle_overlays(out);
            return;
        };
        let Some((_, t)) = self.edited_text() else { return };
        let blue = gpui_kit::hsla(0.58, 0.9, 0.55, 1.0);
        let s = |q: vector::P| self.vp.to_screen(q.x, q.y, self.canvas, self.sf);
        let b = t.bounds();
        let c = [(b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1)].map(|(x, y)| s(p(x as f32, y as f32)));
        out.push(Overlay::Path { polys: vec![(c.to_vec(), true)], dashed: true, color: blue, width: 1.0 });
        if let Some((bw, bh)) = t.box_size {
            let (x0, y0) = t.origin;
            let bc = [(x0, y0), (x0 + bw, y0), (x0 + bw, y0 + bh), (x0, y0 + bh)].map(|(x, y)| s(p(x, y)));
            out.push(Overlay::Path { polys: vec![(bc.to_vec(), true)], dashed: false, color: blue, width: 1.0 });
            if t.layout.overflow {
                // Photoshop's overflow mark: a ⊞ at the box's bottom-right handle.
                let q = s(p(x0 + bw, y0 + bh));
                out.push(Overlay::Text { pos: q, text: "⊞".into(), color: gpui_kit::hsla(0.0, 0.85, 0.5, 1.0), size: 14.0, bg: None });
            }
        }
        if let Some(tp) = &t.path {
            let polys: Vec<_> = tp.path.flatten(0.5).iter().map(|pl| (pl.pts.iter().map(|&q| s(q)).collect(), pl.closed)).collect();
            out.push(Overlay::Path { polys, dashed: false, color: blue, width: 1.0 });
        }
        self.path_handle_overlays(out);
        let (a, bsel) = e.range();
        if a < bsel {
            let quads: Vec<Vec<_>> = t.range_quads(a, bsel).into_iter().map(|q| q.iter().map(|&v| s(v)).collect()).collect();
            out.push(Overlay::Fill { polys: quads, color: gpui_kit::hsla(0.58, 0.9, 0.55, 0.35) });
        }
        let (c0, c1) = t.caret_seg(e.caret);
        out.push(Overlay::Path { polys: vec![(vec![s(c0), s(c1)], false)], dashed: false, color: gpui_kit::hsla(0., 0., 0., 1.), width: 2.0 });
    }

    /// Start (bar with a filled square) and end (bar with a hollow circle)
    /// markers of type on a path, across the path like Photoshop's I-beams.
    pub fn path_handle_overlays(&self, out: &mut Vec<Overlay>) {
        let Some((t, s0, st, e0, et)) = self.path_handles() else { return };
        let blue = gpui_kit::hsla(0.58, 0.9, 0.55, 1.0);
        let sc = |q: vector::P| self.vp.to_screen(q.x, q.y, self.canvas, self.sf);
        if self.text_edit.is_none() {
            // The path itself (while editing, type_overlays draws it).
            if let Some(tp) = &t.path {
                let polys: Vec<_> = tp.path.flatten(0.5).iter().map(|pl| (pl.pts.iter().map(|&q| sc(q)).collect(), pl.closed)).collect();
                out.push(Overlay::Path { polys, dashed: false, color: blue, width: 1.0 });
            }
        }
        let r = t.style_at(0).size * 0.6;
        for (q, tn, end) in [(s0, st, false), (e0, et, true)] {
            let up = p(tn.y, -tn.x);
            out.push(Overlay::Path { polys: vec![(vec![sc(q - up * r), sc(q + up * r)], false)], dashed: false, color: blue, width: 2.0 });
            let c = sc(q);
            let k = gpui_kit::px(5.0);
            let ring: Vec<_> = (0..=16)
                .map(|i| {
                    let a = i as f32 / 16.0 * std::f32::consts::TAU;
                    gpui_kit::point(c.x + k * a.cos(), c.y + k * a.sin())
                })
                .collect();
            if end {
                out.push(Overlay::Path { polys: vec![(ring, true)], dashed: false, color: blue, width: 1.5 });
            } else {
                out.push(Overlay::Fill { polys: vec![vec![gpui_kit::point(c.x - k, c.y - k), gpui_kit::point(c.x + k, c.y - k), gpui_kit::point(c.x + k, c.y + k), gpui_kit::point(c.x - k, c.y + k)]], color: blue });
            }
        }
    }

    // ----- options bar ----------------------------------------------------------------------

    pub fn type_options(&mut self, mut bar: Div, window: &mut Window, cx: &mut Context<Self>) -> Div {
        if !self.is_type_tool() {
            return bar;
        }
        if self.topts.family_select.is_none() {
            let fams: Vec<SharedString> = text::families().into_iter().map(SharedString::from).collect();
            let cur = fams.iter().position(|f| f.as_ref() == self.topts.style.family).unwrap_or(0);
            let st = cx.new(|cx| SelectState::new(fams, Some(IndexPath::new(cur)), window, cx).searchable(true));
            let sub = cx.subscribe_in(&st, window, |this: &mut Editor, _, ev: &SelectEvent<Vec<SharedString>>, _, cx| {
                if let SelectEvent::Confirm(Some(f)) = ev {
                    let f = f.to_string();
                    this.apply_char(move |s| s.family = f.clone());
                    cx.notify();
                }
            });
            self.topts.family_select = Some((st, sub));
        }
        let sel = self.topts.family_select.as_ref().unwrap().0.clone();
        let cs = self.shown_char();
        // Show the family at the caret / selection, not just the tool default.
        let fam = SharedString::from(cs.family.clone());
        if sel.read(cx).selected_value() != Some(&fam) {
            sel.update(cx, |st, cx| st.set_selected_value(&fam, window, cx));
        }
        let ps = self.shown_para();
        bar = bar
            .child(div().w(px(180.)).child(Select::new(&sel).small()))
            .child(toggle("t-bold", "Bold", cs.weight >= 600).on_click(cx.listener(move |e, _, _, cx| {
                let w = if cs.weight >= 600 { 400 } else { 700 };
                e.apply_char(|s| s.weight = w);
                cx.notify();
            })))
            .child(toggle("t-ital", "Italic", cs.italic).on_click(cx.listener(move |e, _, _, cx| {
                let it = !cs.italic;
                e.apply_char(|s| s.italic = it);
                cx.notify();
            })))
            .child(self.opt_slider("Size", "t.size", super::options::spec_log(4.0, 2000.0, 1.0, |e| e.shown_char().size, |e, v| e.apply_char(|s| s.size = v)), |v| format!("{v:.0} px"), 90., window, cx))
            .child(segmented("t-align", &TextAlign::ALL.map(|a| (a, a.label())), ps.align, cx, |e, v| e.apply_para(|p| p.align = v)))
            .child(Button::new("t-color").label("Color = FG").xsmall().on_click(cx.listener(|e, _, _, cx| {
                let c = e.fg;
                e.apply_char(|s| s.color = c);
                cx.notify();
            })));
        if self.text_edit.is_some() {
            let (a, b) = self.text_edit.as_ref().unwrap().range();
            bar = bar
                .child(Button::new("t-commit").label("Commit (Esc)").xsmall().on_click(cx.listener(|e, _, _, cx| {
                    e.commit_text(cx);
                    cx.notify();
                })))
                .child(label(if a < b { format!("{} selected", b - a) } else { String::new() }))
                .child(label(format!("{:.2} ms / edit", self.last_tool_ms)));
        }
        bar
    }

    // ----- Character / Paragraph / OpenType / Warp panel ------------------------------------

    pub fn type_panel_visible(&self) -> bool {
        self.is_type_tool() || self.text_edit.is_some()
    }

    pub fn render_type_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let cs = self.shown_char();
        let ps = self.shown_para();
        let mut col = div().flex().flex_col().gap_1p5().p_2().border_b_1().border_color(rgb(0x3a3a3d));
        col = col.child(div().flex().flex_wrap().gap_0p5().child(segmented("tp-tab", &TypePanel::ALL.map(|t| (t, t.label())), self.topts.panel, cx, |e, v| e.topts.panel = v)));
        macro_rules! flag {
            ($id:literal, $name:literal, $on:expr, $set:expr) => {{
                let on = $on;
                toggle($id, $name, on).on_click(cx.listener(move |e, _, _, cx| {
                    let v = !on;
                    #[allow(clippy::redundant_closure_call)]
                    e.apply_char(|s| ($set)(s, v));
                    cx.notify();
                }))
            }};
        }
        match self.topts.panel {
            TypePanel::Character => {
                col = col
                    .child(self.opt_slider("Leading", "tp.lead", spec(0.0, 1000.0, 1.0, |e| e.shown_char().leading, |e, v| e.apply_char(|s| s.leading = v)), |v| if v <= 0.0 { "Auto".into() } else { format!("{v:.0} px") }, 120., window, cx))
                    .child(self.opt_slider("Tracking", "tp.track", spec(-200.0, 1000.0, 5.0, |e| e.shown_char().tracking, |e, v| e.apply_char(|s| s.tracking = v)), |v| format!("{v:.0}"), 120., window, cx))
                    .child(self.opt_slider("Baseline", "tp.shift", spec(-200.0, 200.0, 1.0, |e| e.shown_char().baseline_shift, |e, v| e.apply_char(|s| s.baseline_shift = v)), |v| format!("{v:.0} px"), 120., window, cx))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .gap_0p5()
                            .child(flag!("tp-fb", "Faux Bold", cs.faux_bold, |s: &mut CharStyle, v| s.faux_bold = v))
                            .child(flag!("tp-fi", "Faux Italic", cs.faux_italic, |s: &mut CharStyle, v| s.faux_italic = v))
                            .child(flag!("tp-ac", "All Caps", cs.caps == Caps::All, |s: &mut CharStyle, v| s.caps = if v { Caps::All } else { Caps::Normal }))
                            .child(flag!("tp-sc", "Small Caps", cs.caps == Caps::Small, |s: &mut CharStyle, v| s.caps = if v { Caps::Small } else { Caps::Normal }))
                            .child(flag!("tp-sup", "Superscript", cs.position == Position::Super, |s: &mut CharStyle, v| s.position = if v { Position::Super } else { Position::Normal }))
                            .child(flag!("tp-sub", "Subscript", cs.position == Position::Sub, |s: &mut CharStyle, v| s.position = if v { Position::Sub } else { Position::Normal }))
                            .child(flag!("tp-ul", "Underline", cs.underline, |s: &mut CharStyle, v| s.underline = v))
                            .child(flag!("tp-st", "Strikethrough", cs.strike, |s: &mut CharStyle, v| s.strike = v)),
                    );
            }
            TypePanel::Paragraph => {
                col = col
                    .child(segmented("tp-align", &TextAlign::ALL.map(|a| (a, a.label())), ps.align, cx, |e, v| e.apply_para(|p| p.align = v)))
                    .when(ps.align == TextAlign::Justify, |d| d.child(segmented("tp-last", &LastLine::ALL.map(|a| (a, a.label())), ps.last, cx, |e, v| e.apply_para(|p| p.last = v))))
                    .child(self.opt_slider("Indent Left", "tp.il", spec(-500.0, 1000.0, 1.0, |e| e.shown_para().indent_left, |e, v| e.apply_para(|p| p.indent_left = v)), |v| format!("{v:.0} px"), 110., window, cx))
                    .child(self.opt_slider("Indent Right", "tp.ir", spec(-500.0, 1000.0, 1.0, |e| e.shown_para().indent_right, |e, v| e.apply_para(|p| p.indent_right = v)), |v| format!("{v:.0} px"), 110., window, cx))
                    .child(self.opt_slider("First Line", "tp.if", spec(-500.0, 1000.0, 1.0, |e| e.shown_para().indent_first, |e, v| e.apply_para(|p| p.indent_first = v)), |v| format!("{v:.0} px"), 110., window, cx))
                    .child(self.opt_slider("Space Before", "tp.sb", spec(0.0, 500.0, 1.0, |e| e.shown_para().space_before, |e, v| e.apply_para(|p| p.space_before = v)), |v| format!("{v:.0} px"), 110., window, cx))
                    .child(self.opt_slider("Space After", "tp.sa", spec(0.0, 500.0, 1.0, |e| e.shown_para().space_after, |e, v| e.apply_para(|p| p.space_after = v)), |v| format!("{v:.0} px"), 110., window, cx))
                    .child(toggle("tp-hy", "Hyphenate", ps.hyphenate).on_click(cx.listener(move |e, _, _, cx| {
                        let v = !ps.hyphenate;
                        e.apply_para(|p| p.hyphenate = v);
                        cx.notify();
                    })))
                    .child(label("Composer"))
                    .child(segmented("tp-comp", &Composer::ALL.map(|c| (c, c.label())), ps.composer, cx, |e, v| e.apply_para(|p| p.composer = v)));
            }
            TypePanel::OpenType => {
                let f = cs.features;
                let mut row = div()
                    .flex()
                    .flex_wrap()
                    .gap_0p5()
                    .child(flag!("ot-liga", "Ligatures", f.liga, |s: &mut CharStyle, v| s.features.liga = v))
                    .child(flag!("ot-dlig", "Discretionary", f.dlig, |s: &mut CharStyle, v| s.features.dlig = v))
                    .child(flag!("ot-onum", "Oldstyle", f.onum, |s: &mut CharStyle, v| s.features.onum = v))
                    .child(flag!("ot-swsh", "Swash", f.swsh, |s: &mut CharStyle, v| s.features.swsh = v))
                    .child(flag!("ot-frac", "Fractions", f.frac, |s: &mut CharStyle, v| s.features.frac = v));
                for k in 0..8u32 {
                    let on = f.ss & (1 << k) != 0;
                    row = row.child(toggle(SharedString::from(format!("ot-ss{k}")), SharedString::from(format!("ss0{}", k + 1)), on).on_click(cx.listener(move |e, _, _, cx| {
                        e.apply_char(|s| s.features.ss ^= 1 << k);
                        cx.notify();
                    })));
                }
                col = col.child(label("Stylistic sets apply where the font has them")).child(row);
            }
            TypePanel::Warp => {
                let w = self.shown_layer().map_or(self.topts.warp, |t| t.warp);
                col = col
                    .child(div().flex().flex_wrap().gap_0p5().child(segmented("tp-warp", &WarpStyle::ALL.map(|s| (s, s.label())), w.style, cx, |e, v| {
                        e.topts.warp.style = v;
                        e.apply_layer(|t| t.warp.style = v);
                    })))
                    .child(segmented("tp-wdir", &[(false, "Horizontal"), (true, "Vertical")], w.vertical, cx, |e, v| {
                        e.topts.warp.vertical = v;
                        e.apply_layer(|t| t.warp.vertical = v);
                    }));
                let wget = |e: &Editor| e.shown_layer().map_or(e.topts.warp, |t| t.warp);
                let _ = wget;
                col = col
                    .child(self.opt_slider("Bend", "tp.bend", spec(-100.0, 100.0, 1.0, |e| e.shown_layer().map_or(e.topts.warp, |t| t.warp).bend * 100.0, |e, v| {
                        e.topts.warp.bend = v / 100.0;
                        e.apply_layer(|t| t.warp.bend = v / 100.0);
                    }), |v| format!("{v:.0}%"), 120., window, cx))
                    .child(self.opt_slider("H Distortion", "tp.hd", spec(-100.0, 100.0, 1.0, |e| e.shown_layer().map_or(e.topts.warp, |t| t.warp).h_distort * 100.0, |e, v| {
                        e.topts.warp.h_distort = v / 100.0;
                        e.apply_layer(|t| t.warp.h_distort = v / 100.0);
                    }), |v| format!("{v:.0}%"), 120., window, cx))
                    .child(self.opt_slider("V Distortion", "tp.vd", spec(-100.0, 100.0, 1.0, |e| e.shown_layer().map_or(e.topts.warp, |t| t.warp).v_distort * 100.0, |e, v| {
                        e.topts.warp.v_distort = v / 100.0;
                        e.apply_layer(|t| t.warp.v_distort = v / 100.0);
                    }), |v| format!("{v:.0}%"), 120., window, cx));
            }
            TypePanel::Options => {
                let t = self.shown_layer();
                let upright = t.as_ref().is_some_and(|t| t.upright_roman);
                col = col
                    .child(label("Vertical type"))
                    .child(
                        div()
                            .flex()
                            .flex_wrap()
                            .gap_0p5()
                            .child(toggle("tp-svra", "Standard Vertical Roman", upright).on_click(cx.listener(move |e, _, _, cx| {
                                e.apply_layer(|t| t.upright_roman = !upright);
                                cx.notify();
                            })))
                            .child(Button::new("tp-tcy").label("Tate-Chu-Yoko").xsmall().on_click(cx.listener(|e, _, _, cx| {
                                // Toggles TCY on the selected range.
                                if let Some((a, b)) = e.text_edit.as_ref().map(|x| x.range()).filter(|r| r.0 < r.1) {
                                    e.apply_layer(|t| {
                                        if t.tcy.iter().any(|&(x, y)| x < b && y > a) {
                                            t.tcy.retain(|&(x, y)| !(x < b && y > a));
                                        } else {
                                            t.tcy.push((a, b));
                                        }
                                    });
                                }
                                cx.notify();
                            }))),
                    );
                if let Some(tp) = t.as_ref().and_then(|t| t.path.clone()) {
                    col = col
                        .child(label(if tp.area { "Area type (path)" } else { "Type on a path" }))
                        .child(self.opt_slider("Start", "tp.ps", spec(0.0, 100.0, 0.5, |e| e.shown_layer().and_then(|t| t.path.as_ref().map(|p| p.start * 100.0)).unwrap_or(0.0), |e, v| {
                            e.apply_layer(|t| {
                                if let Some(p) = &mut t.path {
                                    Arc::make_mut(p).start = v / 100.0;
                                }
                            })
                        }), |v| format!("{v:.1}%"), 120., window, cx))
                        .child(self.opt_slider("End", "tp.pe", spec(0.0, 200.0, 0.5, |e| e.shown_layer().and_then(|t| t.path.as_ref().map(|p| p.end * 100.0)).unwrap_or(100.0), |e, v| {
                            e.apply_layer(|t| {
                                if let Some(p) = &mut t.path {
                                    Arc::make_mut(p).end = v / 100.0;
                                }
                            })
                        }), |v| format!("{v:.1}%"), 120., window, cx))
                        .child(toggle("tp-flip", "Flip", tp.flip).on_click(cx.listener(move |e, _, _, cx| {
                            e.apply_layer(|t| {
                                if let Some(p) = &mut t.path {
                                    Arc::make_mut(p).flip = !tp.flip;
                                }
                            });
                            cx.notify();
                        })));
                } else {
                    col = col.child(label("Click the active path with the Type tool for type on a path; inside a closed path for area type"));
                }
            }
        }
        col
    }
}
