//! Navigation, measurement and canvas tools: Rotate View, scrubby Zoom,
//! Eyedropper sample size, Color Sampler, Ruler (Straighten), Note, Count,
//! Artboards, Slices / Slice Select, Crop (handles, rotation, ratios),
//! Perspective Crop and the Move tool's auto-select, transform controls,
//! alignment and floating selections.

use std::time::Instant;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputState};
use gpui_kit::component::Sizable as _;
use gpui_kit::prelude::*;
use gpui_kit::{Bounds, Context, Div, Entity, Hsla, MouseDownEvent, PathPromptOptions, Pixels, Point, SharedString, Window, div, hsla, point, px};

use crate::engine::annot::{self, CountGroup, MAX_SAMPLERS, Note, Slice};
use crate::engine::composite;
use crate::engine::document::Layer;
use crate::engine::grid::RectI;
use crate::engine::layers::LayerKind;
use crate::engine::ops;
use crate::engine::xform;

use super::editor::{Drag, Editor, Overlay, Tool};
use super::options::{label, segmented, spec, toggle};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CropRatio {
    #[default]
    Free,
    Original,
    Square,
    R4x5,
    R3x2,
    R16x9,
}

impl CropRatio {
    pub const ALL: [CropRatio; 6] = [CropRatio::Free, CropRatio::Original, CropRatio::Square, CropRatio::R4x5, CropRatio::R3x2, CropRatio::R16x9];
    pub fn label(self) -> &'static str {
        match self {
            CropRatio::Free => "Ratio",
            CropRatio::Original => "Original",
            CropRatio::Square => "1:1",
            CropRatio::R4x5 => "4:5",
            CropRatio::R3x2 => "3:2",
            CropRatio::R16x9 => "16:9",
        }
    }
    fn value(self, w: u32, h: u32) -> Option<f32> {
        match self {
            CropRatio::Free => None,
            CropRatio::Original => Some(w as f32 / h as f32),
            CropRatio::Square => Some(1.0),
            CropRatio::R4x5 => Some(0.8),
            CropRatio::R3x2 => Some(1.5),
            CropRatio::R16x9 => Some(16.0 / 9.0),
        }
    }
}

/// An axis-aligned float rectangle (x0, y0, x1, y1).
type FRect = (f32, f32, f32, f32);

fn norm(r: FRect) -> FRect {
    (r.0.min(r.2), r.1.min(r.3), r.0.max(r.2), r.1.max(r.3))
}

fn to_rect(r: FRect) -> RectI {
    let r = norm(r);
    RectI::new(r.0.round() as i32, r.1.round() as i32, r.2.round() as i32, r.3.round() as i32)
}

fn of_rect(r: RectI) -> FRect {
    (r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32)
}

/// The 8 handle positions of a rectangle (TL, T, TR, R, BR, B, BL, L).
fn handles(r: FRect) -> [(f32, f32); 8] {
    let (x0, y0, x1, y1) = r;
    let (mx, my) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
    [(x0, y0), (mx, y0), (x1, y0), (x1, my), (x1, y1), (mx, y1), (x0, y1), (x0, my)]
}

/// Drags rectangle `r0` by handle `h` to image point `q`, optionally keeping
/// aspect ratio `ratio` (w / h).
fn drag_handle(r0: FRect, h: u8, q: (f32, f32), ratio: Option<f32>) -> FRect {
    let (x0, y0, x1, y1) = r0;
    let (mx, my) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
    match (h, ratio) {
        // Corners: the opposite corner stays fixed.
        (0 | 2 | 4 | 6, _) => {
            let (fx, fy) = match h {
                0 => (x1, y1),
                2 => (x0, y1),
                4 => (x0, y0),
                _ => (x1, y0),
            };
            let (dx, dy) = (q.0 - fx, q.1 - fy);
            let (ex, ey) = match ratio {
                Some(k) => {
                    let w = dx.abs().max(dy.abs() * k);
                    (fx + w * if dx < 0.0 { -1.0 } else { 1.0 }, fy + w / k * if dy < 0.0 { -1.0 } else { 1.0 })
                }
                None => q,
            };
            (fx.min(ex), fy.min(ey), fx.max(ex), fy.max(ey))
        }
        // Edges: with a ratio the other dimension follows, centred.
        (1 | 5, r) => {
            let (ny0, ny1) = if h == 1 { (q.1, y1) } else { (y0, q.1) };
            let (ny0, ny1) = (ny0.min(ny1), ny0.max(ny1));
            match r {
                Some(k) => {
                    let hw = (ny1 - ny0) * k * 0.5;
                    (mx - hw, ny0, mx + hw, ny1)
                }
                None => (x0, ny0, x1, ny1),
            }
        }
        (_, r) => {
            let (nx0, nx1) = if h == 3 { (x0, q.0) } else { (q.0, x1) };
            let (nx0, nx1) = (nx0.min(nx1), nx0.max(nx1));
            match r {
                Some(k) => {
                    let hh = (nx1 - nx0) / k * 0.5;
                    (nx0, my - hh, nx1, my + hh)
                }
                None => (nx0, y0, nx1, y1),
            }
        }
    }
}

/// Computes a pixel layer's new grid for a canvas transform.
pub type PixelOp = std::sync::Arc<dyn Fn(&Layer) -> crate::engine::grid::TileGrid + Send + Sync>;

pub struct NavOptions {
    pub scrubby: bool,
    pub crop_ratio: CropRatio,
    /// Crop: off keeps pixels outside the frame as hidden pixels.
    pub crop_delete: bool,
    /// Crop: Content-Aware fill of canvas area the crop adds.
    pub crop_fill: bool,
    pub crop_interp: crate::engine::crop::Interp,
    pub crop_overlay: crate::engine::crop::CropOverlay,
    /// Overlay orientation (Shift+O cycles).
    pub crop_orient: u8,
    /// Crop ▸ Straighten: the next drag draws the line to level.
    pub crop_straighten: bool,
    /// Crop frame rotation (degrees, clockwise).
    pub crop_angle: f32,
    pub ruler: Option<((f32, f32), (f32, f32))>,
    /// Protractor: the second leg's end; its vertex is the ruler's first point.
    pub ruler2: Option<(f32, f32)>,
    /// Perspective Crop corners (TL, TR, BR, BL once complete).
    pub pcrop: Vec<(f32, f32)>,
    pub slice_sel: Option<usize>,
    /// Selected artboard (layer id).
    pub artboard_sel: Option<u64>,
    /// Artboard export format: 0 PNG, 1 JPEG, 2 PDF (one page per artboard).
    pub artboard_format: u8,
    /// Divide Slice: columns × rows.
    pub slice_divide: (u32, u32),
    pub note_sel: Option<usize>,
    /// Note being edited: index, its text field and the last text synced.
    pub note_input: Option<(usize, Entity<InputState>, String)>,
    pub count_group: usize,
    pub sample_layer: bool,
    /// Info panel readout mode of the cursor and of each color sampler.
    pub info_mode: crate::engine::color::ColorMode,
    /// Move tool: cached content bounds (layer id, version, rect).
    pub content: Option<(u64, u64, Option<RectI>)>,
    /// Align To: the selected layers' bounds (or a marquee / the canvas),
    /// or the key object (the active layer stays put).
    pub align_key: bool,
}

impl Default for NavOptions {
    fn default() -> Self {
        NavOptions {
            scrubby: true,
            crop_ratio: CropRatio::Free,
            crop_delete: true,
            crop_fill: false,
            crop_interp: Default::default(),
            crop_overlay: Default::default(),
            crop_orient: 0,
            crop_straighten: false,
            crop_angle: 0.0,
            ruler: None,
            ruler2: None,
            pcrop: vec![],
            slice_sel: None,
            artboard_sel: None,
            artboard_format: 0,
            slice_divide: (2, 2),
            note_sel: None,
            note_input: None,
            info_mode: Default::default(),
            count_group: 0,
            sample_layer: false,
            content: None,
            align_key: false,
        }
    }
}

pub enum NavDrag {
    RotateView { a0: f32, rot0: f32 },
    Scrubby { start: Point<Pixels>, zoom0: f32, moved: bool, alt: bool },
    ZoomBox { start: (f32, f32), cur: (f32, f32) },
    Eyedrop,
    Sampler(usize),
    Ruler { end: u8 },
    /// Protractor's second leg.
    RulerLeg2,
    Note(usize),
    Count { group: usize, idx: usize },
    /// New rectangle (crop, slice, artboard).
    NewRect { start: (f32, f32), cur: (f32, f32) },
    CropHandle { h: u8, r0: FRect },
    CropMove { r0: FRect, start: (f32, f32) },
    CropRotate { a0: f32, ang0: f32 },
    /// Crop ▸ Straighten (or ⌘-drag): the line to level.
    CropStraighten { a: (f32, f32), b: (f32, f32) },
    PCorner(usize),
    SliceEdit { idx: usize, h: u8, r0: FRect, start: (f32, f32) },
    ArtboardEdit { idx: usize, h: u8, r0: FRect, start: (f32, f32) },
    /// Move tool transform handle (scale from `r0`).
    Transform { h: u8, r0: FRect, cur: FRect },
    /// Floating selected pixels.
    Float { start: (f32, f32), layer: usize },
}

const NOTE_COLOR: [u8; 3] = [250, 214, 70];
const GROUP_COLORS: [[u8; 3]; 6] = [[230, 40, 40], [40, 140, 240], [40, 190, 80], [240, 160, 20], [170, 70, 220], [20, 200, 200]];

impl Editor {
    fn tol_img(&self, screen_px: f32) -> f32 {
        screen_px / self.vp.logical_scale(self.sf)
    }

    fn near(&self, a: (f32, f32), b: (f32, f32), screen_px: f32) -> bool {
        (a.0 - b.0).hypot(a.1 - b.1) <= self.tol_img(screen_px)
    }

    fn screen_angle(&self, p: Point<Pixels>) -> f32 {
        let c = self.canvas.center();
        f32::from(p.y - c.y).atan2(f32::from(p.x - c.x))
    }

    /// The crop frame's corners (image space), honouring its rotation.
    pub fn crop_corners(&self, r: RectI) -> [(f32, f32); 4] {
        let (cx, cy) = ((r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5);
        let m = annot::rotation_about(self.nopts.crop_angle, cx, cy);
        [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)].map(|(x, y)| {
            let (x, y) = (x as f32, y as f32);
            (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
        })
    }

    /// `q` in the crop frame's unrotated coordinates.
    fn crop_local(&self, r: RectI, q: (f32, f32)) -> (f32, f32) {
        let (cx, cy) = ((r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5);
        let m = annot::rotation_about(-self.nopts.crop_angle, cx, cy);
        (m[0] * q.0 + m[1] * q.1 + m[2], m[3] * q.0 + m[4] * q.1 + m[5])
    }

    /// Sampled colour (straight RGBA) at an image point with the current
    /// Sample Size.
    pub fn sample_at(&self, x: f32, y: f32) -> Option<[u8; 4]> {
        let side = self.opts.sample_size.side();
        let li = self.doc.active;
        let layers: &[Layer] = if self.nopts.sample_layer { std::slice::from_ref(&self.doc.layers[li]) } else { &self.doc.layers };
        annot::box_average(layers, self.doc.w, self.doc.h, x, y, side)
    }

    /// Tight bounds of the active layer's content (cached per version).
    fn content_rect(&mut self) -> Option<RectI> {
        let l = self.doc.layers.get(self.doc.active)?;
        if let Some((id, v, r)) = self.nopts.content {
            if id == l.id && v == l.version {
                return r;
            }
        }
        let r = l.kind.content_bounds().map(|b| b.intersect(&l.grid.bounds())).or_else(|| xform::content_rect(&l.grid));
        self.nopts.content = Some((l.id, l.version, r));
        r
    }

    // ----- mouse -------------------------------------------------------------------------

    pub fn nav_down(&mut self, ix: f32, iy: f32, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let q = (ix, iy);
        let alt = ev.modifiers.alt;
        let nd = match self.tool {
            Tool::RotateView => {
                if ev.click_count >= 2 {
                    self.vp.rot = 0.0;
                    return true;
                }
                NavDrag::RotateView { a0: self.screen_angle(ev.position), rot0: self.vp.rot }
            }
            Tool::Zoom => {
                if self.nopts.scrubby {
                    NavDrag::Scrubby { start: ev.position, zoom0: self.vp.zoom, moved: false, alt }
                } else {
                    NavDrag::ZoomBox { start: q, cur: q }
                }
            }
            Tool::Eyedropper => {
                self.eyedrop(ix, iy, window, cx);
                NavDrag::Eyedrop
            }
            Tool::ColorSampler => {
                let hit = self.doc.ann.samplers.iter().position(|&s| self.near(s, q, 7.0));
                match (hit, alt) {
                    (Some(i), true) => {
                        self.doc.push_undo("Delete Color Sampler");
                        self.doc.ann.samplers.remove(i);
                        if i < self.doc.ann.sampler_modes.len() {
                            self.doc.ann.sampler_modes.remove(i);
                        }
                        return true;
                    }
                    (Some(i), false) => {
                        self.doc.push_undo("Move Color Sampler");
                        NavDrag::Sampler(i)
                    }
                    (None, _) => {
                        if self.doc.ann.samplers.len() >= MAX_SAMPLERS {
                            self.flash(format!("At most {MAX_SAMPLERS} color samplers"));
                            return true;
                        }
                        self.doc.push_undo("Color Sampler");
                        self.doc.ann.samplers.push(q);
                        NavDrag::Sampler(self.doc.ann.samplers.len() - 1)
                    }
                }
            }
            Tool::Ruler => {
                // ⌥-drag from an end point starts the protractor's second leg
                // there (that point becomes the vertex).
                let leg2_start = match self.nopts.ruler {
                    Some((a, b)) if alt && (self.near(a, q, 7.0) || self.near(b, q, 7.0)) => {
                        if !self.near(a, q, 7.0) {
                            self.nopts.ruler = Some((b, a));
                        }
                        self.nopts.ruler2 = Some(q);
                        true
                    }
                    _ => self.nopts.ruler2.is_some_and(|c| self.near(c, q, 7.0)),
                };
                if leg2_start {
                    NavDrag::RulerLeg2
                } else {
                    let end = match self.nopts.ruler {
                        Some((a, _)) if self.near(a, q, 7.0) => 0,
                        Some((_, b)) if self.near(b, q, 7.0) => 1,
                        _ => {
                            self.nopts.ruler = Some((q, q));
                            self.nopts.ruler2 = None;
                            1
                        }
                    };
                    NavDrag::Ruler { end }
                }
            }
            Tool::Note => {
                let hit = self.doc.ann.notes.iter().position(|n| self.near((n.x + 8.0 * self.tol_img(1.0), n.y + 8.0 * self.tol_img(1.0)), q, 11.0));
                let i = match hit {
                    Some(i) => i,
                    None => {
                        self.doc.push_undo("New Note");
                        let author = self.dockst.prefs.author.clone();
                        self.doc.ann.notes.push(Note { x: ix, y: iy, author, text: String::new() });
                        self.doc.ann.notes.len() - 1
                    }
                };
                self.select_note(Some(i), window, cx);
                NavDrag::Note(i)
            }
            Tool::Count => {
                let hit = self.doc.ann.counts.iter().enumerate().find_map(|(gi, g)| g.pts.iter().position(|&p| self.near(p, q, 8.0)).map(|pi| (gi, pi)));
                match (hit, alt) {
                    (Some((g, i)), true) => {
                        self.doc.push_undo("Delete Count");
                        self.doc.ann.counts[g].pts.remove(i);
                        return true;
                    }
                    (Some((g, i)), false) => {
                        self.doc.push_undo("Move Count");
                        NavDrag::Count { group: g, idx: i }
                    }
                    (None, _) => {
                        self.doc.push_undo("Count");
                        if self.doc.ann.counts.is_empty() {
                            self.doc.ann.counts.push(CountGroup::new("Count Group 1", GROUP_COLORS[0]));
                        }
                        let g = self.nopts.count_group.min(self.doc.ann.counts.len() - 1);
                        self.doc.ann.counts[g].pts.push(q);
                        NavDrag::Count { group: g, idx: self.doc.ann.counts[g].pts.len() - 1 }
                    }
                }
            }
            Tool::Crop if self.nopts.crop_straighten || ev.modifiers.platform => NavDrag::CropStraighten { a: q, b: q },
            Tool::Crop => {
                if let Some(r) = self.crop {
                    let local = self.crop_local(r, q);
                    let fr = of_rect(r);
                    if let Some(h) = handles(fr).iter().position(|&hp| self.near(hp, local, 7.0)) {
                        NavDrag::CropHandle { h: h as u8, r0: fr }
                    } else if local.0 >= fr.0 && local.0 <= fr.2 && local.1 >= fr.1 && local.1 <= fr.3 {
                        NavDrag::CropMove { r0: fr, start: q }
                    } else {
                        let c = ((fr.0 + fr.2) * 0.5, (fr.1 + fr.3) * 0.5);
                        NavDrag::CropRotate { a0: (q.1 - c.1).atan2(q.0 - c.0), ang0: self.nopts.crop_angle }
                    }
                } else {
                    self.nopts.crop_angle = 0.0;
                    NavDrag::NewRect { start: q, cur: q }
                }
            }
            Tool::PerspectiveCrop => {
                if let Some(i) = self.nopts.pcrop.iter().position(|&c| self.near(c, q, 8.0)) {
                    NavDrag::PCorner(i)
                } else if self.nopts.pcrop.len() < 4 {
                    self.nopts.pcrop.push(q);
                    if self.nopts.pcrop.len() == 4 {
                        self.flash("Drag the corners to match the plane · Enter crops · Esc cancels");
                    }
                    NavDrag::PCorner(self.nopts.pcrop.len() - 1)
                } else {
                    return true;
                }
            }
            Tool::Slice => NavDrag::NewRect { start: q, cur: q },
            Tool::SliceSelect => {
                let hit = self.slice_hit(q);
                self.nopts.slice_sel = hit.map(|(i, _)| i);
                match hit {
                    Some((i, h)) => {
                        self.doc.push_undo("Edit Slice");
                        NavDrag::SliceEdit { idx: i, h, r0: of_rect(self.doc.ann.slices[i].rect), start: q }
                    }
                    None => return true,
                }
            }
            Tool::Artboard => {
                let hit = self.artboard_hit(q);
                match hit {
                    Some((i, h)) => {
                        let (li, id, r) = self.artboards()[i];
                        self.nopts.artboard_sel = Some(id);
                        self.doc.active = li;
                        self.doc.push_undo("Edit Artboard");
                        NavDrag::ArtboardEdit { idx: i, h, r0: of_rect(r), start: q }
                    }
                    None => NavDrag::NewRect { start: q, cur: q },
                }
            }
            Tool::Move => return self.move_down(ix, iy, ev, cx),
            _ => return false,
        };
        self.ndrag = Some(nd);
        self.drag = Drag::Nav;
        true
    }

    fn move_down(&mut self, ix: f32, iy: f32, ev: &MouseDownEvent, cx: &mut Context<Self>) -> bool {
        // Transform handles of the active layer win over Auto-Select.
        if self.opts.show_transform {
            if let Some(r) = self.content_rect() {
                let fr = of_rect(r);
                if let Some(h) = handles(fr).iter().position(|&hp| self.near(hp, (ix, iy), 7.0)) {
                    self.ndrag = Some(NavDrag::Transform { h: h as u8, r0: fr, cur: fr });
                    self.drag = Drag::Nav;
                    return true;
                }
            }
        }
        // Auto-Select (or ⌘-click): the topmost layer with pixels under the cursor.
        if self.opts.auto_select || ev.modifiers.platform {
            let (xi, yi) = (ix.floor() as i32, iy.floor() as i32);
            if let Some(mut li) = self.layer_at(xi, yi) {
                if self.opts.auto_select_group {
                    // The outermost group around the hit layer.
                    let mut i = li;
                    while let Some(pi) = self.doc.layers[i].parent.and_then(|p| self.doc.layers.iter().position(|l| l.id == p)) {
                        if matches!(self.doc.layers[pi].kind, LayerKind::Group { .. }) {
                            li = pi;
                        }
                        i = pi;
                    }
                }
                if !self.doc.selected.contains(&self.doc.layers[li].id) {
                    self.doc.selected.clear();
                }
                self.doc.active = li;
            }
        }
        let li = self.doc.active;
        // Inside a selection on a pixel layer: float the selected pixels.
        if let Some(sel) = &self.doc.selection {
            if matches!(self.doc.layers[li].kind, LayerKind::Pixel) && sel.at(ix.floor() as i32, iy.floor() as i32) > 0 {
                let sel = sel.clone();
                let g = &self.doc.layers[li].grid;
                let float = ops::copy_masked(g, Some(&sel));
                // The locked Background has no transparency: the hole gets the BG colour.
                let hole = if self.doc.layers[li].background { ops::fill(g, Some(&sel), self.bg, 1.0) } else { ops::clear(g, Some(&sel)) };
                self.doc.push_undo("Move Selection");
                self.doc.set_layer_grid(li, hole);
                let mut fl = Layer::new(u64::MAX - 1, "float", float);
                fl.opacity = self.doc.layers[li].opacity;
                self.float = Some(fl);
                self.ndrag = Some(NavDrag::Float { start: (ix, iy), layer: li });
                self.drag = Drag::Nav;
                return true;
            }
        }
        let _ = cx;
        let ids = self.moving_ids();
        if ids.is_empty() {
            self.flash("Could not move: the Background layer is locked (double-click it in Layers to unlock)");
            return true;
        }
        self.drag = Drag::Move { start: (ix, iy), layer: li, ids };
        true
    }

    /// Topmost visible layer with pixels at a point (Auto-Select).
    fn layer_at(&self, xi: i32, yi: i32) -> Option<usize> {
        (0..self.doc.layers.len()).rev().find(|&i| {
            let l = &self.doc.layers[i];
            l.visible && !l.kind.is_container() && xi >= 0 && yi >= 0 && xi < l.grid.w as i32 && yi < l.grid.h as i32 && crate::engine::px::alpha(l.grid.pixel(xi, yi)) > 0
        })
    }

    /// Ids the Move tool moves: the selected layers and every descendant of a
    /// selected group / artboard, minus the locked Background.
    fn moving_ids(&self) -> Vec<u64> {
        let sel: Vec<u64> = self.selected_layers().iter().map(|&i| self.doc.layers[i].id).collect();
        let mut out: Vec<u64> = Vec::new();
        for (i, l) in self.doc.layers.iter().enumerate() {
            if l.background {
                continue;
            }
            // Selected, or inside a selected container.
            let mut cur = Some(i);
            while let Some(c) = cur {
                if sel.contains(&self.doc.layers[c].id) {
                    out.push(l.id);
                    break;
                }
                cur = self.doc.layers[c].parent.and_then(|p| self.doc.layers.iter().position(|m| m.id == p));
            }
        }
        out
    }

    /// Content bounds of layer `li` (containers: the union of their children).
    fn layer_rect(&self, li: usize) -> Option<RectI> {
        let l = &self.doc.layers[li];
        match &l.kind {
            LayerKind::Artboard(a) => Some(a.rect),
            k if k.folder().is_some() => {
                let kids: Vec<usize> = (0..self.doc.layers.len()).filter(|&i| self.doc.layers[i].parent == Some(l.id)).collect();
                kids.iter().filter_map(|&k| self.layer_rect(k)).reduce(|a, b| a.union(&b))
            }
            k => k.content_bounds().map(|b| b.intersect(&l.grid.bounds())).or_else(|| xform::content_rect(&l.grid)),
        }
    }

    /// Moves layer `li` (and, for containers, its descendants) by `d`.
    fn translate_layer_tree(&mut self, li: usize, d: (i32, i32)) {
        if d == (0, 0) {
            return;
        }
        let (w, h) = (self.doc.w, self.doc.h);
        let m = [1.0, 0.0, d.0 as f32, 0.0, 1.0, d.1 as f32, 0.0, 0.0, 1.0];
        let id = self.doc.layers[li].id;
        let mut todo: Vec<usize> = vec![li];
        let mut k = 0;
        while k < todo.len() {
            let pid = self.doc.layers[todo[k]].id;
            todo.extend((0..self.doc.layers.len()).filter(|&i| self.doc.layers[i].parent == Some(pid)));
            k += 1;
        }
        let _ = id;
        for i in todo {
            if !self.doc.layers[i].background {
                self.doc.layers[i] = xform::layer_transformed(&self.doc.layers[i], &m, w, h);
            }
        }
    }

    pub fn nav_move(&mut self, ix: f32, iy: f32, pos: Point<Pixels>, shift: bool, alt: bool) {
        let Some(nd) = self.ndrag.take() else { return };
        let q = (ix, iy);
        let nd = match nd {
            NavDrag::RotateView { a0, rot0 } => {
                let mut r = rot0 + self.screen_angle(pos) - a0;
                if shift {
                    let step = 15f32.to_radians();
                    r = (r / step).round() * step;
                }
                self.vp.rot = (r + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
                NavDrag::RotateView { a0, rot0 }
            }
            NavDrag::Scrubby { start, zoom0, moved, alt } => {
                let dx = f32::from(pos.x - start.x);
                let moved = moved || dx.abs() > 3.0;
                if moved {
                    let z = zoom0 * (dx * 0.01).exp();
                    self.vp.zoom_about(z, start, self.canvas);
                }
                NavDrag::Scrubby { start, zoom0, moved, alt }
            }
            NavDrag::ZoomBox { start, .. } => NavDrag::ZoomBox { start, cur: q },
            NavDrag::Eyedrop => {
                if let Some(c) = self.sample_at(ix, iy) {
                    self.fg_preview = Some([c[0], c[1], c[2]]);
                }
                NavDrag::Eyedrop
            }
            NavDrag::Sampler(i) => {
                if let Some(s) = self.doc.ann.samplers.get_mut(i) {
                    *s = q;
                }
                NavDrag::Sampler(i)
            }
            NavDrag::RulerLeg2 => {
                if let (Some((v, _)), true) = (self.nopts.ruler, shift) {
                    let (dx, dy) = (q.0 - v.0, q.1 - v.1);
                    let ang = (dy.atan2(dx) / std::f32::consts::FRAC_PI_4).round() * std::f32::consts::FRAC_PI_4;
                    let l = dx.hypot(dy);
                    self.nopts.ruler2 = Some((v.0 + l * ang.cos(), v.1 + l * ang.sin()));
                } else {
                    self.nopts.ruler2 = Some(q);
                }
                NavDrag::RulerLeg2
            }
            NavDrag::Ruler { end } => {
                if let Some((a, b)) = &mut self.nopts.ruler {
                    let anchor = if end == 0 { *b } else { *a };
                    let mut t = q;
                    if shift {
                        // Snap to 45° steps.
                        let (dx, dy) = (q.0 - anchor.0, q.1 - anchor.1);
                        let ang = (dy.atan2(dx) / std::f32::consts::FRAC_PI_4).round() * std::f32::consts::FRAC_PI_4;
                        let l = dx.hypot(dy);
                        t = (anchor.0 + l * ang.cos(), anchor.1 + l * ang.sin());
                    }
                    if end == 0 { *a = t } else { *b = t }
                }
                NavDrag::Ruler { end }
            }
            NavDrag::Note(i) => {
                if let Some(n) = self.doc.ann.notes.get_mut(i) {
                    (n.x, n.y) = q;
                }
                NavDrag::Note(i)
            }
            NavDrag::Count { group, idx } => {
                if let Some(p) = self.doc.ann.counts.get_mut(group).and_then(|g| g.pts.get_mut(idx)) {
                    *p = q;
                }
                NavDrag::Count { group, idx }
            }
            NavDrag::NewRect { start, .. } => {
                let mut cur = q;
                let ratio = if self.tool == Tool::Crop { self.nopts.crop_ratio.value(self.doc.w, self.doc.h) } else { None }.or(shift.then_some(1.0));
                if let Some(k) = ratio {
                    let (dx, dy) = (cur.0 - start.0, cur.1 - start.1);
                    let w = dx.abs().max(dy.abs() * k);
                    cur = (start.0 + w * dx.signum(), start.1 + w / k * if dy == 0.0 { 1.0 } else { dy.signum() });
                }
                if self.tool == Tool::Crop {
                    let r = to_rect((start.0, start.1, cur.0, cur.1));
                    self.crop = (r.w() > 0 && r.h() > 0).then_some(r);
                }
                NavDrag::NewRect { start, cur }
            }
            NavDrag::CropHandle { h, r0 } => {
                if let Some(r) = self.crop {
                    let local = self.crop_local(r, q);
                    let ratio = self.nopts.crop_ratio.value(self.doc.w, self.doc.h).or(shift.then(|| (r0.2 - r0.0) / (r0.3 - r0.1).max(1.0)));
                    let nr = to_rect(drag_handle(r0, h, local, ratio));
                    if nr.w() > 1 && nr.h() > 1 {
                        self.crop = Some(nr);
                    }
                }
                NavDrag::CropHandle { h, r0 }
            }
            NavDrag::CropMove { r0, start } => {
                let (dx, dy) = (q.0 - start.0, q.1 - start.1);
                self.crop = Some(to_rect((r0.0 + dx, r0.1 + dy, r0.2 + dx, r0.3 + dy)));
                NavDrag::CropMove { r0, start }
            }
            NavDrag::CropStraighten { a, .. } => NavDrag::CropStraighten { a, b: q },
            NavDrag::CropRotate { a0, ang0 } => {
                if let Some(r) = self.crop {
                    let c = ((r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5);
                    let mut a = ang0 + ((q.1 - c.1).atan2(q.0 - c.0) - a0).to_degrees();
                    if shift {
                        a = (a / 15.0).round() * 15.0;
                    }
                    self.nopts.crop_angle = (a + 180.0).rem_euclid(360.0) - 180.0;
                }
                NavDrag::CropRotate { a0, ang0 }
            }
            NavDrag::PCorner(i) => {
                if let Some(c) = self.nopts.pcrop.get_mut(i) {
                    *c = q;
                }
                NavDrag::PCorner(i)
            }
            NavDrag::SliceEdit { idx, h, r0, start } => {
                if let Some(s) = self.doc.ann.slices.get_mut(idx) {
                    s.rect = to_rect(edit_rect(r0, h, start, q));
                }
                NavDrag::SliceEdit { idx, h, r0, start }
            }
            NavDrag::ArtboardEdit { idx, h, r0, start } => {
                if let Some(&(_, id, _)) = self.artboards().get(idx) {
                    self.set_artboard_rect(id, to_rect(edit_rect(r0, h, start, q)));
                }
                NavDrag::ArtboardEdit { idx, h, r0, start }
            }
            NavDrag::Transform { h, r0, .. } => {
                // Corners keep proportions unless Shift (Photoshop 2026 default); Alt scales about the centre.
                let ratio = (matches!(h, 0 | 2 | 4 | 6) && !shift).then(|| (r0.2 - r0.0) / (r0.3 - r0.1).max(1.0));
                let mut cur = drag_handle(r0, h, q, ratio);
                if alt {
                    let c = ((r0.0 + r0.2) * 0.5, (r0.1 + r0.3) * 0.5);
                    let n = norm(cur);
                    let (hw, hh) = (((n.2 - n.0) * 0.5).max((c.0 - n.0).abs().max((n.2 - c.0).abs())), ((n.3 - n.1) * 0.5).max((c.1 - n.1).abs().max((n.3 - c.1).abs())));
                    cur = (c.0 - hw, c.1 - hh, c.0 + hw, c.1 + hh);
                }
                NavDrag::Transform { h, r0, cur }
            }
            NavDrag::Float { start, layer } => {
                if let Some(f) = &mut self.float {
                    let off = ((ix - start.0).round() as i32, (iy - start.1).round() as i32);
                    if f.offset != off {
                        f.offset = off;
                        self.doc.mark_all_dirty();
                    }
                }
                NavDrag::Float { start, layer }
            }
        };
        self.ndrag = Some(nd);
    }

    pub fn nav_up(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(nd) = self.ndrag.take() else { return };
        match nd {
            NavDrag::Scrubby { start, moved: false, alt, .. } => {
                let z = if alt { self.vp.zoom / 2.0 } else { self.vp.zoom * 2.0 };
                self.vp.zoom_about(z, start, self.canvas);
            }
            NavDrag::ZoomBox { start, cur } => {
                let r = to_rect((start.0, start.1, cur.0, cur.1));
                if r.w() > 4 && r.h() > 4 {
                    self.zoom_to_rect(r);
                } else {
                    let p = self.vp.to_screen(start.0, start.1, self.canvas, self.sf);
                    self.vp.zoom_about(self.vp.zoom * 2.0, p, self.canvas);
                }
            }
            NavDrag::Eyedrop => {
                if let Some(c) = self.fg_preview.take() {
                    self.set_fg(c, window, cx);
                }
            }
            NavDrag::Ruler { .. } => {
                if let Some((a, b)) = self.nopts.ruler {
                    if self.near(a, b, 2.0) {
                        self.nopts.ruler = None;
                    }
                }
            }
            NavDrag::CropStraighten { a, b } => {
                self.nopts.crop_straighten = false;
                if (a.0 - b.0).hypot(a.1 - b.1) > 4.0 {
                    let ang = annot::straighten_angle(a, b);
                    let (r, fa) = crate::engine::crop::straighten_frame(self.doc.w, self.doc.h, ang);
                    self.crop = Some(r);
                    self.nopts.crop_angle = fa;
                    self.flash(format!("Straightened by {ang:.1}° · Enter crops"));
                }
            }
            NavDrag::NewRect { start, cur } => {
                let r = to_rect((start.0, start.1, cur.0, cur.1));
                let big = r.w() > 2 && r.h() > 2;
                match self.tool {
                    Tool::Crop => {
                        if big {
                            self.flash("Drag handles to resize, inside to move, outside to rotate · Enter crops");
                        } else {
                            self.crop = None;
                        }
                    }
                    Tool::Slice if big => {
                        self.doc.push_undo("Slice");
                        let n = self.doc.ann.slices.len() + 1;
                        self.doc.ann.slices.push(Slice::new(format!("slice_{n:02}"), r.intersect(&self.doc.bounds())));
                        self.nopts.slice_sel = Some(n - 1);
                    }
                    Tool::Artboard if big => {
                        self.doc.push_undo("Artboard");
                        let n = self.artboards().len() + 1;
                        self.new_artboard(format!("Artboard {n}"), r);
                    }
                    _ => {}
                }
            }
            NavDrag::Transform { r0, cur, .. } => self.apply_transform(r0, cur, cx),
            NavDrag::Float { layer, .. } => self.drop_float(layer, cx),
            _ => {}
        }
    }

    fn zoom_to_rect(&mut self, r: RectI) {
        let c = self.canvas;
        let k = (f32::from(c.size.width) / r.w() as f32).min(f32::from(c.size.height) / r.h() as f32) * self.sf;
        let target = self.vp.to_screen((r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5, c, self.sf);
        self.vp.pan = self.vp.pan + self.vp.unrotate_delta(c.center() - target);
        self.vp.zoom_about(k, c.center(), c);
    }

    fn eyedrop(&mut self, ix: f32, iy: f32, window: &mut Window, cx: &mut Context<Self>) {
        let t0 = Instant::now();
        if let Some(c) = self.sample_at(ix, iy) {
            self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
            if c[3] > 0 {
                self.set_fg([c[0], c[1], c[2]], window, cx);
                self.flash(format!("Picked #{:02X}{:02X}{:02X} ({}, {:.2} ms)", c[0], c[1], c[2], super::options::sample_label(self.opts.sample_size), self.last_tool_ms));
            }
        }
    }

    pub fn set_note_selection(&mut self, i: Option<usize>, window: &mut Window, cx: &mut Context<Self>) {
        self.select_note(i, window, cx);
    }

    fn select_note(&mut self, i: Option<usize>, window: &mut Window, cx: &mut Context<Self>) {
        self.nopts.note_sel = i;
        self.nopts.note_input = i.map(|i| {
            let text = self.doc.ann.notes[i].text.clone();
            (i, cx.new(|cx| InputState::new(window, cx).placeholder("Note text").default_value(text.clone())), text)
        });
    }

    /// Sets a note's text (and its field when it is being edited).
    pub fn set_note_text(&mut self, i: usize, text: &str, window: &mut Window, cx: &mut Context<Self>) {
        if let Some(n) = self.doc.ann.notes.get_mut(i) {
            n.text = text.into();
        }
        if let Some((j, input, last)) = &mut self.nopts.note_input {
            if *j == i {
                input.update(cx, |s, cx| s.set_value(text.to_string(), window, cx));
                *last = text.into();
            }
        }
    }

    /// Exports slices / artboards straight into `dir` (no folder prompt).
    pub fn export_regions_now(&self, slices: bool, dir: &std::path::Path) -> anyhow::Result<usize> {
        let regions: Vec<(String, RectI)> = if slices {
            annot::all_slices(&self.resolved_slices(), self.doc.w, self.doc.h).into_iter().enumerate().map(|(i, (r, _))| (format!("slice_{:02}", i + 1), r)).collect()
        } else {
            return export_artboards_to(&self.doc.layers, 0, dir);
        };
        export_regions_to(&self.doc.layers, &regions, dir)
    }

    /// Artboard layers: (layer index, id, rect), bottom to top.
    pub fn artboards(&self) -> Vec<(usize, u64, RectI)> {
        self.doc.layers.iter().enumerate().filter_map(|(i, l)| match &l.kind {
            LayerKind::Artboard(a) => Some((i, l.id, a.rect)),
            _ => None,
        }).collect()
    }

    fn set_artboard_rect(&mut self, id: u64, rect: RectI) {
        if let Some(l) = self.doc.layers.iter_mut().find(|l| l.id == id) {
            if let LayerKind::Artboard(a) = &mut l.kind {
                std::sync::Arc::make_mut(a).rect = rect;
            }
        }
        self.doc.mark_all_dirty();
    }

    /// A new (top-level, white) artboard; the canvas grows to hold it.
    pub fn new_artboard(&mut self, name: String, rect: RectI) -> u64 {
        let rect = self.grow_canvas_to(rect);
        let id = self.doc.next_layer_id();
        let mut l = Layer::new(id, name, crate::engine::grid::TileGrid::new(self.doc.w, self.doc.h));
        l.kind = LayerKind::Artboard(std::sync::Arc::new(crate::engine::artboard::ArtboardLayer { rect, bg: Some([255, 255, 255]), collapsed: false }));
        self.doc.layers.push(l);
        self.doc.active = self.doc.layers.len() - 1;
        self.doc.selected.clear();
        self.nopts.artboard_sel = Some(id);
        self.doc.mark_all_dirty();
        id
    }

    /// Grows the canvas (synchronously, within the caller's undo step) so it
    /// contains `r`; returns `r` in the new canvas's coordinates.
    fn grow_canvas_to(&mut self, r: RectI) -> RectI {
        let b = self.doc.bounds();
        if b.contains_rect(&r) {
            return r;
        }
        let u = b.union(&r);
        let (dx, dy) = (-u.x0.min(0), -u.y0.min(0));
        let (w, h) = ((u.x1 + dx) as u32, (u.y1 + dy) as u32);
        let m = [1.0, 0.0, dx as f32, 0.0, 1.0, dy as f32, 0.0, 0.0, 1.0];
        let mut res = xform::canvas_transformed(&self.doc.layers, &self.doc.paths, &self.doc.ann, &m, w, h);
        for l in res.layers.iter_mut().filter(|l| l.background) {
            let g = ops::matte(&l.grid, self.bg);
            l.replace_grid(g);
        }
        self.doc.install_canvas(res);
        self.needs_fit = true;
        RectI::new(r.x0 + dx, r.y0 + dy, r.x1 + dx, r.y1 + dy)
    }

    /// Deletes an artboard; its layers stay, at the top level.
    pub fn delete_artboard(&mut self, id: u64) {
        let Some(li) = self.doc.layers.iter().position(|l| l.id == id) else { return };
        for l in self.doc.layers.iter_mut().filter(|l| l.parent == Some(id)) {
            l.parent = None;
        }
        self.doc.layers.remove(li);
        self.doc.active = self.doc.active.min(self.doc.layers.len() - 1);
        self.nopts.artboard_sel = None;
        self.doc.mark_all_dirty();
    }

    fn slice_hit(&self, q: (f32, f32)) -> Option<(usize, u8)> {
        rect_hit(self.resolved_slices().iter().map(|s| s.rect), q, self.tol_img(7.0), self.nopts.slice_sel)
    }

    fn artboard_hit(&self, q: (f32, f32)) -> Option<(usize, u8)> {
        let abs = self.artboards();
        let sel = self.nopts.artboard_sel.and_then(|id| abs.iter().position(|a| a.1 == id));
        rect_hit(abs.iter().map(|a| a.2), q, self.tol_img(7.0), sel)
    }

    // ----- commits ----------------------------------------------------------------------

    /// O cycles the crop overlay; Shift+O its orientation.
    pub fn cycle_crop_overlay(&mut self, orientation: bool) {
        use crate::engine::crop::CropOverlay;
        let o = &mut self.nopts;
        if orientation {
            o.crop_orient = (o.crop_orient + 1) % o.crop_overlay.orientations().max(1);
        } else {
            let i = CropOverlay::ALL.iter().position(|&k| k == o.crop_overlay).unwrap_or(0);
            o.crop_overlay = CropOverlay::ALL[(i + 1) % CropOverlay::ALL.len()];
            o.crop_orient = 0;
        }
    }

    /// Crop with the current frame (rotated frames resample).
    pub fn commit_crop(&mut self, cx: &mut Context<Self>) {
        let Some(r) = self.crop.take() else { return };
        if r.w() < 1 || r.h() < 1 {
            return;
        }
        let ang = std::mem::take(&mut self.nopts.crop_angle);
        let m = if ang == 0.0 { [1.0, 0.0, -(r.x0 as f32), 0.0, 1.0, -(r.y0 as f32), 0.0, 0.0, 1.0] } else { annot::rotated_crop(r, ang) };
        let o = xform::XformOpts { interp: Some(self.nopts.crop_interp), keep: !self.nopts.crop_delete, ..Default::default() };
        self.canvas_xform_with("Crop", m, r.w() as u32, r.h() as u32, o, None, self.nopts.crop_fill, cx);
    }

    pub fn commit_perspective_crop(&mut self, cx: &mut Context<Self>) {
        if self.nopts.pcrop.len() != 4 {
            return;
        }
        let c = std::mem::take(&mut self.nopts.pcrop);
        // Order the corners TL, TR, BR, BL by angle around the centroid.
        let (mx, my) = (c.iter().map(|p| p.0).sum::<f32>() / 4.0, c.iter().map(|p| p.1).sum::<f32>() / 4.0);
        let mut c = c;
        c.sort_by(|a, b| (a.1 - my).atan2(a.0 - mx).total_cmp(&(b.1 - my).atan2(b.0 - mx)));
        // atan2 order starts at −π (left); rotate so the top-left comes first.
        let tl = c.iter().enumerate().min_by(|a, b| (a.1.0 + a.1.1).total_cmp(&(b.1.0 + b.1.1))).map(|(i, _)| i).unwrap();
        c.rotate_left(tl);
        let q = [c[0], c[1], c[2], c[3]];
        let (m, w, h) = annot::perspective_crop(q);
        self.canvas_xform("Perspective Crop", m, w, h, cx);
    }

    /// Straighten Layer: rotates the image so the ruler is level, cropping
    /// to the largest same-aspect rectangle (Alt keeps the whole image). On a
    /// layer other than the Background only that layer turns, about its
    /// centre, keeping its corners as hidden pixels.
    pub fn straighten(&mut self, keep_all: bool, cx: &mut Context<Self>) {
        let Some((a, b)) = self.nopts.ruler.take() else {
            self.flash("Draw a ruler along a line that should be level");
            return;
        };
        let ang = annot::straighten_angle(a, b);
        if ang.abs() < 1e-3 {
            return;
        }
        let li = self.doc.active;
        let l = &self.doc.layers[li];
        if !l.background && matches!(l.kind, LayerKind::Pixel) {
            let Some(r) = crate::engine::crop::extent(&l.grid, l.hidden.as_deref()) else { return };
            let m = annot::rotation_about(ang, (r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5);
            let o = xform::XformOpts { interp: Some(self.nopts.crop_interp), keep: true, ..Default::default() };
            let (layer, w, h, id) = (l.clone(), self.doc.w, self.doc.h, l.id);
            self.spawn_job("Straighten Layer", cx, move |_| {
                let out = xform::layer_transformed_with(&layer, &m, w, h, o);
                Box::new(move |ed: &mut Editor, _| {
                    if let Some(li) = ed.doc.layers.iter().position(|l| l.id == id) {
                        ed.doc.push_undo("Straighten Layer");
                        ed.doc.layers[li] = out;
                        ed.doc.mark_all_dirty();
                    }
                })
            });
            return;
        }
        let (w, h) = (self.doc.w as f32, self.doc.h as f32);
        let (s, c) = ang.to_radians().abs().sin_cos();
        let (bw, bh) = (w * c + h * s, w * s + h * c);
        let (ow, oh) = if keep_all {
            (bw, bh)
        } else {
            let k = (w / bw).min(h / bh);
            (w * k, h * k)
        };
        let rot = annot::rotation_about(ang, w * 0.5, h * 0.5);
        let tr = [1.0, 0.0, (ow - w) * 0.5, 0.0, 1.0, (oh - h) * 0.5, 0.0, 0.0, 1.0];
        let m = crate::engine::filters::distort::mat_mul(&tr, &rot);
        let o = xform::XformOpts { interp: Some(self.nopts.crop_interp), ..Default::default() };
        self.canvas_xform_with("Straighten", m, ow.round() as u32, oh.round() as u32, o, None, false, cx);
    }

    /// Image ▸ Reveal All: grows the canvas to show every layer's hidden
    /// pixels (and vector content beyond the canvas).
    pub fn reveal_all(&mut self, cx: &mut Context<Self>) {
        let b = self.doc.bounds();
        let u = self.doc.layers.iter().filter(|l| l.visible || l.hidden.is_some()).fold(b, |u, l| {
            let e = match &l.kind {
                LayerKind::Pixel => crate::engine::crop::extent(&l.grid, l.hidden.as_deref()),
                k if k.is_vector() => k.content_bounds(),
                _ => None,
            };
            e.map_or(u, |e| u.union(&e))
        });
        if u == b {
            self.flash("Nothing to reveal: every layer fits the canvas");
            return;
        }
        let m = [1.0, 0.0, -u.x0 as f32, 0.0, 1.0, -u.y0 as f32, 0.0, 0.0, 1.0];
        self.canvas_xform("Reveal All", m, u.w() as u32, u.h() as u32, cx);
    }

    pub fn canvas_xform(&mut self, label: &'static str, m: [f32; 9], w: u32, h: u32, cx: &mut Context<Self>) {
        self.canvas_xform_with(label, m, w, h, xform::XformOpts::default(), None, false, cx);
    }

    /// Moves the whole document through `m` onto a `w × h` canvas: layers
    /// (masks, vector descriptions, hidden pixels), paths and annotations.
    /// `op` computes plain pixel layers directly (exact flips / rotations,
    /// Image Size, Super Zoom); `fill` content-aware fills canvas area the
    /// transform adds on the bottom pixel layer.
    #[allow(clippy::too_many_arguments)]
    pub fn canvas_xform_with(&mut self, label: &'static str, m: [f32; 9], w: u32, h: u32, o: xform::XformOpts, op: Option<PixelOp>, fill: bool, cx: &mut Context<Self>) {
        let (layers, paths, ann) = (self.doc.layers.clone(), self.doc.paths.clone(), self.doc.ann.clone());
        let (ow, oh) = (self.doc.w, self.doc.h);
        let bg = self.bg;
        self.spawn_job(label, cx, move |_| {
            use rayon::prelude::*;
            let o = xform::XformOpts { skip_pixels: op.is_some(), ..o };
            let mut r = xform::canvas_transformed_with(&layers, &paths, &ann, &m, w, h, o);
            if let Some(op) = &op {
                r.layers.par_iter_mut().zip(&layers).filter(|(nl, l)| matches!(l.kind, LayerKind::Pixel) && l.hidden.is_none() && nl.hidden.is_none()).for_each(|(nl, l)| {
                    nl.replace_grid(op(l));
                });
            }
            // New canvas area: Content-Aware fill on the bottom pixel layer,
            // the background colour on the locked Background.
            let bottom = r.layers.iter().position(|l| matches!(l.kind, LayerKind::Pixel));
            let hole = fill.then(|| crate::engine::crop::new_area(&m, ow, oh, w, h));
            for (i, l) in r.layers.iter_mut().enumerate() {
                if let (Some(hole), true) = (&hole, Some(i) == bottom) {
                    let g = crate::engine::crop::content_aware(&l.grid, hole);
                    l.replace_grid(g);
                }
                if l.background {
                    let g = ops::matte(&l.grid, bg);
                    l.replace_grid(g);
                }
            }
            Box::new(move |ed: &mut Editor, _| {
                ed.doc.push_undo(label);
                ed.doc.install_canvas(r);
                ed.needs_fit = true;
            })
        });
    }

    fn apply_transform(&mut self, r0: FRect, cur: FRect, cx: &mut Context<Self>) {
        let n = norm(cur);
        if (n.2 - n.0) < 1.0 || (n.3 - n.1) < 1.0 || n == r0 {
            return;
        }
        let (sx, sy) = ((cur.2 - cur.0) / (r0.2 - r0.0), (cur.3 - cur.1) / (r0.3 - r0.1));
        let m = [sx, 0.0, cur.0 - r0.0 * sx, 0.0, sy, cur.1 - r0.1 * sy, 0.0, 0.0, 1.0];
        let li = self.doc.active;
        let layer = self.doc.layers[li].clone();
        let (w, h) = (self.doc.w, self.doc.h);
        let id = layer.id;
        self.spawn_job("Free Transform", cx, move |_| {
            let out = xform::layer_transformed(&layer, &m, w, h);
            Box::new(move |ed: &mut Editor, _| {
                if let Some(li) = ed.doc.layers.iter().position(|l| l.id == id) {
                    ed.doc.push_undo("Free Transform");
                    ed.doc.layers[li] = out;
                    ed.doc.mark_all_dirty();
                }
            })
        });
    }

    fn drop_float(&mut self, li: usize, cx: &mut Context<Self>) {
        let Some(fl) = self.float.take() else { return };
        let off = fl.offset;
        let mut base = self.doc.layers[li].clone();
        let (opacity, blend) = (base.opacity, base.blend);
        base.opacity = 1.0;
        base.blend = crate::engine::blend::BlendMode::Normal;
        base.visible = true;
        base.mask = None;
        base.kind = LayerKind::Pixel;
        let (w, h) = (self.doc.w, self.doc.h);
        let id = base.id;
        let mut top = Layer::new(u64::MAX - 1, "float", ops::translate(&fl.grid, off.0, off.1));
        top.opacity = 1.0;
        let _ = (opacity, blend);
        self.doc.mark_all_dirty();
        self.spawn_job("Move Selection", cx, move |_| {
            let merged = composite::flatten_layers(&[base, top], w, h);
            Box::new(move |ed: &mut Editor, _| {
                if let Some(li) = ed.doc.layers.iter().position(|l| l.id == id) {
                    ed.doc.set_layer_grid(li, merged);
                }
                if let Some(sel) = &mut ed.doc.selection {
                    sel.translate(off.0 as f32, off.1 as f32);
                }
                ed.doc.mark_all_dirty();
            })
        });
    }

    /// Aligns the active layer's content to the selection (or the canvas).
    /// Align (Move tool options bar): one layer to the marquee / canvas;
    /// several to their combined bounds (or the marquee), or to the key
    /// object (the active layer) with "Align To: Key Object".
    pub fn align_layer(&mut self, how: &str, cx: &mut Context<Self>) {
        use crate::engine::arrange::{self, Align};
        let how = match how {
            "left" => Align::Left,
            "hcenter" => Align::HCenter,
            "right" => Align::Right,
            "top" => Align::Top,
            "vcenter" => Align::VCenter,
            _ => Align::Bottom,
        };
        let items: Vec<(usize, RectI)> = self.selected_layers().into_iter().filter(|&i| !self.doc.layers[i].background).filter_map(|i| self.layer_rect(i).map(|r| (i, r))).collect();
        if items.is_empty() {
            return;
        }
        let rects: Vec<RectI> = items.iter().map(|x| x.1).collect();
        let marquee = self.doc.selection.as_ref().map(|s| s.shape_bounds().intersect(&self.doc.bounds()));
        let key = (self.nopts.align_key && items.len() > 1).then(|| items.iter().position(|x| x.0 == self.doc.active)).flatten();
        let target = if key.is_some() { None } else if items.len() == 1 { Some(marquee.unwrap_or(self.doc.bounds())) } else { marquee };
        let d = arrange::align(&rects, how, target, key);
        self.apply_arrange("Align", &items, &d);
        let _ = cx;
    }

    /// Distribute (three or more selected layers).
    pub fn distribute_layers(&mut self, how: crate::engine::arrange::Distribute) {
        let items: Vec<(usize, RectI)> = self.selected_layers().into_iter().filter(|&i| !self.doc.layers[i].background).filter_map(|i| self.layer_rect(i).map(|r| (i, r))).collect();
        if items.len() < 3 {
            self.flash("Distribute needs three or more selected layers");
            return;
        }
        let rects: Vec<RectI> = items.iter().map(|x| x.1).collect();
        let d = crate::engine::arrange::distribute(&rects, how);
        self.apply_arrange("Distribute", &items, &d);
    }

    fn apply_arrange(&mut self, what: &str, items: &[(usize, RectI)], d: &[(i32, i32)]) {
        if d.iter().all(|&o| o == (0, 0)) {
            return;
        }
        let t0 = Instant::now();
        self.doc.push_undo(what);
        let ids: Vec<(u64, (i32, i32))> = items.iter().zip(d).map(|(&(i, _), &o)| (self.doc.layers[i].id, o)).collect();
        for (id, o) in ids {
            if let Some(li) = self.doc.layers.iter().position(|l| l.id == id) {
                self.translate_layer_tree(li, o);
            }
        }
        self.doc.mark_all_dirty();
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.flash(format!("{what}: {} layers in {:.1} ms", items.len(), self.last_tool_ms));
    }

    /// Writes each artboard (or each slice) to `dir` as PNG.
    pub fn export_regions(&mut self, slices: bool, cx: &mut Context<Self>) {
        let regions: Vec<(String, RectI)> = if slices {
            annot::all_slices(&self.resolved_slices(), self.doc.w, self.doc.h)
                .into_iter()
                .enumerate()
                .map(|(i, (r, user))| {
                    let name = if user { self.doc.ann.slices.iter().find(|s| s.rect == r).map(|s| s.name.clone()).unwrap_or_default() } else { String::new() };
                    (if name.is_empty() { format!("slice_{:02}", i + 1) } else { name }, r)
                })
                .collect()
        } else {
            vec![]
        };
        if slices && regions.is_empty() {
            self.flash("No slices");
            return;
        }
        if !slices && self.artboards().is_empty() {
            self.flash("No artboards");
            return;
        }
        let format = self.nopts.artboard_format;
        let layers = self.doc.layers.clone();
        let rx = cx.prompt_for_paths(PathPromptOptions { files: false, directories: true, multiple: false, prompt: Some("Export".into()) });
        cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else { return };
            let Some(dir) = paths.into_iter().next() else { return };
            let _ = this.update(cx, |ed, cx| {
                ed.spawn_job("Export", cx, move |_| {
                    let res = if slices { export_regions_to(&layers, &regions, &dir) } else { export_artboards_to(&layers, format, &dir) };
                    Box::new(move |ed: &mut Editor, _| match res {
                        Ok(n) => ed.flash(format!("Exported {n} files")),
                        Err(e) => ed.flash(format!("Error: {e:#}")),
                    })
                });
            });
        })
        .detach();
    }

    // ----- overlays ----------------------------------------------------------------------

    pub fn nav_overlays(&mut self, out: &mut Vec<Overlay>) {
        let (c, sf) = (self.canvas, self.sf);
        let sc = |e: &Editor, x: f32, y: f32| e.vp.to_screen(x, y, c, sf);
        let white = hsla(0., 0., 1., 1.);
        let black = hsla(0., 0., 0., 1.);
        let blue = hsla(0.58, 0.9, 0.55, 1.0);
        let shade = hsla(0., 0., 0., 0.55);
        let quad = |p: Point<Pixels>, r: f32, bg: Hsla, border: Hsla, radius: f32| Overlay::Quad {
            bounds: Bounds::from_corners(point(p.x - px(r), p.y - px(r)), point(p.x + px(r), p.y + px(r))),
            bg,
            border,
            border_w: 1.0,
            radius,
        };
        let canvas_poly = vec![c.origin, c.origin + point(c.size.width, px(0.)), c.origin + point(c.size.width, c.size.height), c.origin + point(px(0.), c.size.height)];
        // Crop frame: shade outside, thirds, handles.
        let crop = match &self.ndrag {
            Some(NavDrag::NewRect { start, cur }) if self.tool == Tool::Crop => Some(to_rect((start.0, start.1, cur.0, cur.1))),
            _ => self.crop,
        };
        if let Some(r) = crop {
            let cs = self.crop_corners(r).map(|(x, y)| sc(self, x, y));
            out.push(Overlay::Fill { polys: vec![canvas_poly.clone(), cs.to_vec()], color: shade });
            // Composition overlay, in frame coordinates mapped onto the
            // (possibly rotated) frame.
            let (fw, fh) = (r.w() as f32, r.h() as f32);
            let grid = 48.0 / self.vp.logical_scale(sf);
            let at = |(u, v): (f32, f32)| {
                let (u, v) = (u / fw, v / fh);
                point(cs[0].x + (cs[1].x - cs[0].x) * u + (cs[3].x - cs[0].x) * v, cs[0].y + (cs[1].y - cs[0].y) * u + (cs[3].y - cs[0].y) * v)
            };
            let lines = crate::engine::crop::overlay_lines(self.nopts.crop_overlay, self.nopts.crop_orient, fw, fh, grid);
            let polys: Vec<(Vec<Point<Pixels>>, bool)> = lines.into_iter().map(|l| (l.into_iter().map(at).collect(), false)).collect();
            out.push(Overlay::Path { polys: polys.clone(), dashed: false, color: hsla(0., 0., 0., 0.35), width: 2.0 });
            out.push(Overlay::Path { polys, dashed: false, color: hsla(0., 0., 1., 0.7), width: 1.0 });
            out.push(Overlay::Path { polys: vec![(cs.to_vec(), true)], dashed: false, color: white, width: 1.0 });
            let fr = of_rect(r);
            let (cx0, cy0) = ((fr.0 + fr.2) * 0.5, (fr.1 + fr.3) * 0.5);
            let m = annot::rotation_about(self.nopts.crop_angle, cx0, cy0);
            for (x, y) in handles(fr) {
                let p = sc(self, m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]);
                out.push(quad(p, 4.0, white, black, 0.0));
            }
            if self.nopts.crop_angle != 0.0 || matches!(self.ndrag, Some(NavDrag::CropRotate { .. })) {
                let p = sc(self, cx0, cy0);
                out.push(Overlay::Text { pos: p, text: format!("{:.1}°", self.nopts.crop_angle).into(), color: white, size: 12.0, bg: Some(hsla(0., 0., 0., 0.6)) });
            }
            out.push(Overlay::Text { pos: cs[2] + point(px(6.), px(6.)), text: format!("W {} px  H {} px", r.w(), r.h()).into(), color: white, size: 11.0, bg: Some(hsla(0., 0., 0., 0.6)) });
        }
        if let Some(NavDrag::CropStraighten { a, b }) = self.ndrag {
            let (pa, pb) = (sc(self, a.0, a.1), sc(self, b.0, b.1));
            out.push(Overlay::Path { polys: vec![(vec![pa, pb], false)], dashed: false, color: black, width: 3.0 });
            out.push(Overlay::Path { polys: vec![(vec![pa, pb], false)], dashed: false, color: white, width: 1.0 });
            out.push(Overlay::Text { pos: pb + point(px(8.), px(8.)), text: format!("{:.1}°", annot::straighten_angle(a, b)).into(), color: white, size: 12.0, bg: Some(hsla(0., 0., 0., 0.6)) });
        }
        // Perspective crop quad + perspective grid.
        if self.tool == Tool::PerspectiveCrop && !self.nopts.pcrop.is_empty() {
            let pts: Vec<Point<Pixels>> = self.nopts.pcrop.iter().map(|&(x, y)| sc(self, x, y)).collect();
            if pts.len() == 4 {
                out.push(Overlay::Fill { polys: vec![canvas_poly.clone(), pts.clone()], color: shade });
                let bil = |u: f32, v: f32| {
                    let top = point(pts[0].x + (pts[1].x - pts[0].x) * u, pts[0].y + (pts[1].y - pts[0].y) * u);
                    let bot = point(pts[3].x + (pts[2].x - pts[3].x) * u, pts[3].y + (pts[2].y - pts[3].y) * u);
                    point(top.x + (bot.x - top.x) * v, top.y + (bot.y - top.y) * v)
                };
                let mut grid = vec![];
                for k in 1..4 {
                    let t = k as f32 / 4.0;
                    grid.push((vec![bil(t, 0.0), bil(t, 1.0)], false));
                    grid.push((vec![bil(0.0, t), bil(1.0, t)], false));
                }
                out.push(Overlay::Path { polys: grid, dashed: false, color: hsla(0., 0., 1., 0.5), width: 1.0 });
            }
            out.push(Overlay::Path { polys: vec![(pts.clone(), pts.len() == 4)], dashed: false, color: white, width: 1.5 });
            for p in pts {
                out.push(quad(p, 4.5, white, black, 0.0));
            }
        }
        // Color samplers.
        for (i, &(x, y)) in self.doc.ann.samplers.iter().enumerate() {
            let p = sc(self, x, y);
            out.push(Overlay::Path { polys: vec![(vec![p - point(px(9.), px(0.)), p + point(px(9.), px(0.))], false), (vec![p - point(px(0.), px(9.)), p + point(px(0.), px(9.))], false)], dashed: false, color: black, width: 1.0 });
            out.push(quad(p, 5.0, hsla(0., 0., 0., 0.), white, 5.0));
            out.push(Overlay::Text { pos: p + point(px(7.), px(4.)), text: format!("{}", i + 1).into(), color: white, size: 11.0, bg: Some(hsla(0., 0., 0., 0.7)) });
        }
        // Ruler.
        if let Some((a, b)) = self.nopts.ruler {
            let (pa, pb) = (sc(self, a.0, a.1), sc(self, b.0, b.1));
            out.push(Overlay::Path { polys: vec![(vec![pa, pb], false)], dashed: false, color: black, width: 2.5 });
            out.push(Overlay::Path { polys: vec![(vec![pa, pb], false)], dashed: false, color: white, width: 1.0 });
            for p in [pa, pb] {
                out.push(Overlay::Path { polys: vec![(vec![p - point(px(6.), px(0.)), p + point(px(6.), px(0.))], false), (vec![p - point(px(0.), px(6.)), p + point(px(0.), px(6.))], false)], dashed: false, color: blue, width: 1.5 });
            }
            let (_, _, ang, len) = annot::measure(a, b);
            if let Some(c) = self.nopts.ruler2 {
                // Protractor: second leg, an arc between the legs, the angle.
                let pc = sc(self, c.0, c.1);
                out.push(Overlay::Path { polys: vec![(vec![pa, pc], false)], dashed: false, color: black, width: 2.5 });
                out.push(Overlay::Path { polys: vec![(vec![pa, pc], false)], dashed: false, color: white, width: 1.0 });
                out.push(Overlay::Path { polys: vec![(vec![pc - point(px(6.), px(0.)), pc + point(px(6.), px(0.))], false), (vec![pc - point(px(0.), px(6.)), pc + point(px(0.), px(6.))], false)], dashed: false, color: blue, width: 1.5 });
                let (a1, a2) = (f32::from(pb.y - pa.y).atan2(f32::from(pb.x - pa.x)), f32::from(pc.y - pa.y).atan2(f32::from(pc.x - pa.x)));
                let mut d = a2 - a1;
                while d > std::f32::consts::PI {
                    d -= std::f32::consts::TAU;
                }
                while d < -std::f32::consts::PI {
                    d += std::f32::consts::TAU;
                }
                let arc: Vec<_> = (0..=24).map(|k| {
                    let t = a1 + d * k as f32 / 24.0;
                    pa + point(px(28.0 * t.cos()), px(28.0 * t.sin()))
                }).collect();
                out.push(Overlay::Path { polys: vec![(arc, false)], dashed: false, color: white, width: 1.0 });
                let pang = crate::engine::measure::protractor_angle(a, b, c);
                out.push(Overlay::Text { pos: pa + point(px(32.), px(-22.)), text: format!("{pang:.1}°").into(), color: white, size: 12.0, bg: Some(hsla(0., 0., 0., 0.7)) });
            } else {
                out.push(Overlay::Text { pos: pb + point(px(8.), px(-18.)), text: format!("{len:.1} px  {ang:.1}°").into(), color: white, size: 12.0, bg: Some(hsla(0., 0., 0., 0.7)) });
            }
        }
        // Notes.
        for (i, n) in self.doc.ann.notes.iter().enumerate() {
            let p = sc(self, n.x, n.y);
            let sel = self.nopts.note_sel == Some(i);
            let [r, g, b] = NOTE_COLOR;
            let col = gpui_kit::Rgba { r: r as f32 / 255., g: g as f32 / 255., b: b as f32 / 255., a: 1.0 }.into();
            out.push(Overlay::Quad { bounds: Bounds::from_corners(p, p + point(px(16.), px(14.))), bg: col, border: if sel { blue } else { black }, border_w: if sel { 2.0 } else { 1.0 }, radius: 2.0 });
            out.push(Overlay::Path { polys: vec![(vec![p + point(px(3.), px(4.)), p + point(px(13.), px(4.))], false), (vec![p + point(px(3.), px(7.)), p + point(px(13.), px(7.))], false), (vec![p + point(px(3.), px(10.)), p + point(px(9.), px(10.))], false)], dashed: false, color: black, width: 1.0 });
            if sel || self.tool == Tool::Note {
                let text = if n.text.is_empty() { format!("{} — (empty note)", n.author) } else { format!("{}: {}", n.author, n.text) };
                out.push(Overlay::Text { pos: p + point(px(20.), px(0.)), text: text.into(), color: black, size: 12.0, bg: Some(col) });
            }
        }
        // Count markers.
        for g in self.doc.ann.counts.iter().filter(|g| g.visible) {
            let col: Hsla = gpui_kit::Rgba { r: g.color[0] as f32 / 255., g: g.color[1] as f32 / 255., b: g.color[2] as f32 / 255., a: 1.0 }.into();
            for (k, &(x, y)) in g.pts.iter().enumerate() {
                let p = sc(self, x, y);
                out.push(quad(p, g.size * 0.5, col, white, g.size * 0.5));
                out.push(Overlay::Text { pos: p + point(px(g.size * 0.6), px(-g.label_size * 0.6)), text: format!("{}", k + 1).into(), color: col, size: g.label_size, bg: None });
            }
        }
        // Artboards: name labels and outlines; the pasteboard outside is dimmed.
        let abs: Vec<(String, u64, RectI)> = self.artboards().into_iter().filter(|a| self.doc.layers[a.0].visible).map(|(i, id, r)| {
            let (ox, oy) = self.doc.layers[i].offset;
            (self.doc.layers[i].name.clone(), id, RectI::new(r.x0 + ox, r.y0 + oy, r.x1 + ox, r.y1 + oy))
        }).collect();
        if !abs.is_empty() {
            let mut polys = vec![canvas_poly.clone()];
            for (_, _, ar) in &abs {
                let r = of_rect(*ar);
                polys.push([(r.0, r.1), (r.2, r.1), (r.2, r.3), (r.0, r.3)].map(|(x, y)| sc(self, x, y)).to_vec());
            }
            out.push(Overlay::Fill { polys, color: hsla(0., 0., 0.16, 0.85) });
            for (name, id, ar) in &abs {
                let r = of_rect(*ar);
                let cs: Vec<_> = [(r.0, r.1), (r.2, r.1), (r.2, r.3), (r.0, r.3)].iter().map(|&(x, y)| sc(self, x, y)).collect();
                let sel = self.nopts.artboard_sel == Some(*id) && self.tool == Tool::Artboard;
                out.push(Overlay::Path { polys: vec![(cs.clone(), true)], dashed: false, color: if sel { blue } else { hsla(0., 0., 0.6, 1.) }, width: 1.0 });
                out.push(Overlay::Text { pos: cs[0] + point(px(0.), px(-18.)), text: format!("{}  {}×{}", name, ar.w(), ar.h()).into(), color: hsla(0., 0., 0.85, 1.), size: 12.0, bg: None });
                if sel {
                    for (x, y) in handles(r) {
                        out.push(quad(sc(self, x, y), 4.0, white, blue, 0.0));
                    }
                }
            }
        }
        // Slices (shown while a slice tool is active).
        if matches!(self.tool, Tool::Slice | Tool::SliceSelect) {
            let all = annot::all_slices(&self.resolved_slices(), self.doc.w, self.doc.h);
            let orange = hsla(0.08, 0.95, 0.55, 1.0);
            for (n, (r, user)) in all.iter().enumerate() {
                let fr = of_rect(*r);
                let cs: Vec<_> = [(fr.0, fr.1), (fr.2, fr.1), (fr.2, fr.3), (fr.0, fr.3)].iter().map(|&(x, y)| sc(self, x, y)).collect();
                out.push(Overlay::Path { polys: vec![(cs.clone(), true)], dashed: !user, color: if *user { orange } else { hsla(0., 0., 0.7, 0.9) }, width: 1.0 });
                out.push(Overlay::Text { pos: cs[0] + point(px(2.), px(2.)), text: format!("{:02}", n + 1).into(), color: white, size: 11.0, bg: Some(if *user { orange } else { hsla(0., 0., 0.45, 0.9) }) });
            }
            if let (Tool::SliceSelect, Some(i)) = (self.tool, self.nopts.slice_sel) {
                if let Some(s) = self.doc.ann.slices.get(i) {
                    for (x, y) in handles(of_rect(s.rect)) {
                        out.push(quad(sc(self, x, y), 4.0, white, orange, 0.0));
                    }
                }
            }
        }
        if let Some(NavDrag::NewRect { start, cur }) = &self.ndrag {
            if matches!(self.tool, Tool::Slice | Tool::Artboard) {
                let r = norm((start.0, start.1, cur.0, cur.1));
                let cs: Vec<_> = [(r.0, r.1), (r.2, r.1), (r.2, r.3), (r.0, r.3)].iter().map(|&(x, y)| sc(self, x, y)).collect();
                out.push(Overlay::Path { polys: vec![(cs, true)], dashed: true, color: white, width: 1.0 });
            }
        }
        if let Some(NavDrag::ZoomBox { start, cur }) = &self.ndrag {
            let r = norm((start.0, start.1, cur.0, cur.1));
            let cs: Vec<_> = [(r.0, r.1), (r.2, r.1), (r.2, r.3), (r.0, r.3)].iter().map(|&(x, y)| sc(self, x, y)).collect();
            out.push(Overlay::Path { polys: vec![(cs, true)], dashed: true, color: white, width: 1.0 });
        }
        // Move tool transform box.
        if self.tool == Tool::Move && self.opts.show_transform && self.float.is_none() {
            let cur = match &self.ndrag {
                Some(NavDrag::Transform { cur, .. }) => Some(*cur),
                _ => self.content_rect().map(of_rect),
            };
            if let Some(r) = cur {
                let cs: Vec<_> = [(r.0, r.1), (r.2, r.1), (r.2, r.3), (r.0, r.3)].iter().map(|&(x, y)| sc(self, x, y)).collect();
                out.push(Overlay::Path { polys: vec![(cs, true)], dashed: false, color: blue, width: 1.0 });
                for (x, y) in handles(r) {
                    out.push(quad(sc(self, x, y), 4.0, white, blue, 0.0));
                }
            }
        }
    }

    // ----- options bar -------------------------------------------------------------------

    pub fn nav_options(&mut self, mut bar: Div, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let btn = |id: &'static str, l: SharedString| Button::new(id).label(l).xsmall();
        match self.tool {
            Tool::RotateView => {
                bar = bar
                    .child(self.opt_slider("Rotation Angle", "rv.ang", spec(-180.0, 180.0, 1.0, |e| e.vp.rot.to_degrees(), |e, v| e.vp.rot = v.to_radians()), |v| format!("{v:.0}°"), 120., window, cx))
                    .child(btn("rv-reset", "Reset View".into()).on_click(cx.listener(|e, _, _, cx| {
                        e.vp.rot = 0.0;
                        cx.notify();
                    })))
                    .child(label("Drag to rotate the view · Shift snaps to 15° · double-click resets"));
            }
            Tool::Zoom => {
                bar = bar
                    .child(toggle("z-scrub", "Scrubby Zoom", self.nopts.scrubby).on_click(cx.listener(|e, _, _, cx| {
                        e.nopts.scrubby = !e.nopts.scrubby;
                        cx.notify();
                    })))
                    .child(btn("z-100", "100%".into()).on_click(cx.listener(|e, _, _, cx| e.act_zoom(3, cx))))
                    .child(btn("z-fit", "Fit Screen".into()).on_click(cx.listener(|e, _, _, cx| e.act_zoom(2, cx))))
                    .child(btn("z-fill", "Fill Screen".into()).on_click(cx.listener(|e, _, _, cx| {
                        let c = e.canvas;
                        let k = (f32::from(c.size.width) / e.doc.w as f32).max(f32::from(c.size.height) / e.doc.h as f32) * e.sf;
                        e.vp.center_at_zoom(k, e.doc.w, e.doc.h, c, e.sf);
                        cx.notify();
                    })))
                    .child(label(if self.nopts.scrubby { "Drag left/right to zoom · click zooms in, ⌥-click out" } else { "Drag a rectangle to zoom into it" }));
            }
            Tool::Eyedropper | Tool::ColorSampler => {
                bar = bar.child(segmented("smp-src", &[(false, "All Layers"), (true, "Current Layer")], self.nopts.sample_layer, cx, |e, v| e.nopts.sample_layer = v));
                if self.tool == Tool::ColorSampler {
                    bar = bar
                        .child(label(format!("{}/{MAX_SAMPLERS} samplers · ⌥-click deletes", self.doc.ann.samplers.len())))
                        .child(btn("cs-clear", "Clear All".into()).on_click(cx.listener(|e, _, _, cx| {
                            e.doc.push_undo("Clear Color Samplers");
                            e.doc.ann.samplers.clear();
                            e.doc.ann.sampler_modes.clear();
                            cx.notify();
                        })));
                }
            }
            Tool::Ruler => {
                let r = self.nopts.ruler;
                let (x, y) = r.map_or((0.0, 0.0), |(a, _)| a);
                let (w, h, a, l) = r.map_or((0.0, 0.0, 0.0, 0.0), |(p, q)| annot::measure(p, q));
                let readout = match (r, self.nopts.ruler2) {
                    // Protractor: the angle between the legs and both lengths.
                    (Some((v, b)), Some(c)) => format!("X: {:.1}  Y: {:.1}  A: {:.1}°  L1: {l:.1}  L2: {:.1}", v.0, v.1, crate::engine::measure::protractor_angle(v, b, c), (c.0 - v.0).hypot(c.1 - v.1)),
                    _ => format!("X: {x:.1}  Y: {y:.1}  W: {w:.1}  H: {h:.1}  A: {a:.1}°  L1: {l:.1}"),
                };
                bar = bar
                    .child(label(readout))
                    .child(btn("ru-str", "Straighten Layer".into()).on_click(cx.listener(|e, ev: &gpui_kit::ClickEvent, _, cx| {
                        e.straighten(ev.modifiers().alt, cx);
                        cx.notify();
                    })))
                    .child(btn("ru-clear", "Clear".into()).on_click(cx.listener(|e, _, _, cx| {
                        e.nopts.ruler = None;
                        e.nopts.ruler2 = None;
                        cx.notify();
                    })))
                    .child(btn("ru-rec", "Record Measurements".into()).on_click(cx.listener(|e, _, _, cx| {
                        e.record_measurements(cx);
                        cx.notify();
                    })))
                    .child(label("⌥-drag from an end: protractor · ⌥-click Straighten keeps the whole image"));
            }
            Tool::Note => {
                if let Some((i, input, last)) = self.nopts.note_input.clone() {
                    // Only edits made in the field flow back (undo can change the note too).
                    let v = input.read(cx).value().to_string();
                    if v != last {
                        if let Some(n) = self.doc.ann.notes.get_mut(i) {
                            n.text = v.clone();
                        }
                        self.nopts.note_input = Some((i, input.clone(), v));
                    }
                    bar = bar.child(div().w(px(320.)).child(Input::new(&input).small())).child(btn("note-del", "Delete Note".into()).on_click(cx.listener(move |e, _, _, cx| {
                        if i < e.doc.ann.notes.len() {
                            e.doc.push_undo("Delete Note");
                            e.doc.ann.notes.remove(i);
                        }
                        e.nopts.note_sel = None;
                        e.nopts.note_input = None;
                        cx.notify();
                    })));
                }
                bar = bar.child(label(format!("{} notes", self.doc.ann.notes.len()))).child(btn("note-clear", "Clear All".into()).on_click(cx.listener(|e, _, _, cx| {
                    e.doc.push_undo("Clear Notes");
                    e.doc.ann.notes.clear();
                    e.nopts.note_sel = None;
                    e.nopts.note_input = None;
                    cx.notify();
                })));
            }
            Tool::Count => {
                let items: Vec<(usize, SharedString)> = self.doc.ann.counts.iter().enumerate().map(|(i, g)| (i, format!("{} ({})", g.name, g.pts.len()).into())).collect();
                if !items.is_empty() {
                    let mut seg = div().flex().gap_0p5();
                    for (i, l) in items {
                        let b = Button::new(SharedString::from(format!("cg-{i}"))).label(l).xsmall().on_click(cx.listener(move |e, _, _, cx| {
                            e.nopts.count_group = i;
                            cx.notify();
                        }));
                        seg = seg.child(if self.nopts.count_group == i { b.primary() } else { b.ghost() });
                    }
                    bar = bar.child(seg);
                }
                bar = bar
                    .child(btn("cg-new", "New Group".into()).on_click(cx.listener(|e, _, _, cx| {
                        e.doc.push_undo("New Count Group");
                        let n = e.doc.ann.counts.len();
                        e.doc.ann.counts.push(CountGroup::new(format!("Count Group {}", n + 1), GROUP_COLORS[n % GROUP_COLORS.len()]));
                        e.nopts.count_group = n;
                        cx.notify();
                    })))
                    .child(self.opt_slider("Marker", "cnt.size", spec(4.0, 40.0, 1.0, |e| e.doc.ann.counts.get(e.nopts.count_group).map_or(10.0, |c| c.size), |e, v| {
                        let g = e.nopts.count_group;
                        if let Some(c) = e.doc.ann.counts.get_mut(g) {
                            c.size = v;
                        }
                    }), |v| format!("{v:.0}"), 60., window, cx))
                    .child(self.opt_slider("Label", "cnt.label", spec(6.0, 48.0, 1.0, |e| e.doc.ann.counts.get(e.nopts.count_group).map_or(12.0, |c| c.label_size), |e, v| {
                        let g = e.nopts.count_group;
                        if let Some(c) = e.doc.ann.counts.get_mut(g) {
                            c.label_size = v;
                        }
                    }), |v| format!("{v:.0}"), 60., window, cx))
                    .child(toggle("cg-vis", "Visible", self.doc.ann.counts.get(self.nopts.count_group).is_none_or(|c| c.visible)).on_click(cx.listener(|e, _, _, cx| {
                        let g = e.nopts.count_group;
                        if let Some(c) = e.doc.ann.counts.get_mut(g) {
                            c.visible = !c.visible;
                        }
                        cx.notify();
                    })))
                    .child(btn("cg-rec", "Record".into()).on_click(cx.listener(|e, _, _, cx| {
                        e.record_measurements(cx);
                        cx.notify();
                    })))
                    .child(btn("cg-clear", "Clear".into()).on_click(cx.listener(|e, _, _, cx| {
                        let g = e.nopts.count_group;
                        if let Some(c) = e.doc.ann.counts.get_mut(g) {
                            c.pts.clear();
                        }
                        cx.notify();
                    })))
                    .child(label(format!("Count: {}", self.doc.ann.count_total())));
            }
            Tool::Crop => {
                use crate::engine::crop::{CropOverlay, Interp};
                let o = &self.nopts;
                bar = bar
                    .child(segmented("crop-ratio", &CropRatio::ALL.map(|r| (r, r.label())), o.crop_ratio, cx, |e, v| e.nopts.crop_ratio = v))
                    .child(toggle("crop-straighten", "Straighten", o.crop_straighten).on_click(cx.listener(|e, _, _, cx| {
                        e.nopts.crop_straighten = !e.nopts.crop_straighten;
                        if e.nopts.crop_straighten {
                            e.flash("Drag along a line that should be level (or ⌘-drag any time)");
                        }
                        cx.notify();
                    })))
                    .child(segmented("crop-ov", &CropOverlay::ALL.map(|k| (k, k.label())), o.crop_overlay, cx, |e, v| {
                        e.nopts.crop_overlay = v;
                        e.nopts.crop_orient = 0;
                    }))
                    .child(toggle("crop-del", "Delete Cropped Pixels", o.crop_delete).on_click(cx.listener(|e, _, _, cx| {
                        e.nopts.crop_delete = !e.nopts.crop_delete;
                        cx.notify();
                    })))
                    .child(toggle("crop-caf", "Content-Aware", o.crop_fill).on_click(cx.listener(|e, _, _, cx| {
                        e.nopts.crop_fill = !e.nopts.crop_fill;
                        cx.notify();
                    })))
                    .child(segmented("crop-interp", &Interp::ALL.map(|k| (k, k.label())), o.crop_interp, cx, |e, v| e.nopts.crop_interp = v))
                    .child(label(format!("Angle {:.1}°", o.crop_angle)));
                if self.crop.is_some() {
                    bar = bar
                        .child(btn("crop-apply", "Commit (Enter)".into()).primary().on_click(cx.listener(|e, _, _, cx| {
                            e.commit_crop(cx);
                            cx.notify();
                        })))
                        .child(btn("crop-cancel", "Cancel (Esc)".into()).on_click(cx.listener(|e, _, _, cx| {
                            e.crop = None;
                            e.nopts.crop_angle = 0.0;
                            cx.notify();
                        })));
                } else {
                    bar = bar.child(label("Drag a crop frame"));
                }
            }
            Tool::PerspectiveCrop => {
                bar = bar.child(label(format!("{} / 4 corners", self.nopts.pcrop.len())));
                if self.nopts.pcrop.len() == 4 {
                    bar = bar.child(btn("pc-apply", "Commit (Enter)".into()).primary().on_click(cx.listener(|e, _, _, cx| {
                        e.commit_perspective_crop(cx);
                        cx.notify();
                    })));
                }
                bar = bar.child(btn("pc-clear", "Clear".into()).on_click(cx.listener(|e, _, _, cx| {
                    e.nopts.pcrop.clear();
                    cx.notify();
                })));
            }
            Tool::Slice | Tool::SliceSelect => {
                let n = annot::all_slices(&self.resolved_slices(), self.doc.w, self.doc.h).len();
                bar = bar
                    .child(label(format!("{} user slices · {n} total", self.doc.ann.slices.len())))
                    .child(btn("sl-export", "Export Slices…".into()).on_click(cx.listener(|e, _, _, cx| e.export_regions(true, cx))))
                    .child(btn("sl-clear", "Delete All".into()).on_click(cx.listener(|e, _, _, cx| {
                        e.doc.push_undo("Delete Slices");
                        e.doc.ann.slices.clear();
                        e.nopts.slice_sel = None;
                        cx.notify();
                    })));
                if let (Tool::SliceSelect, Some(i)) = (self.tool, self.nopts.slice_sel) {
                    bar = bar.child(btn("sl-del", "Delete Slice".into()).on_click(cx.listener(move |e, _, _, cx| {
                        if i < e.doc.ann.slices.len() {
                            e.doc.push_undo("Delete Slice");
                            e.doc.ann.slices.remove(i);
                        }
                        e.nopts.slice_sel = None;
                        cx.notify();
                    })));
                }
            }
            Tool::Artboard => {
                let mut pr = div().flex().flex_wrap().gap_0p5().child(label("Presets:"));
                for (i, &(name, w, h)) in crate::engine::artboard::PRESETS.iter().enumerate() {
                    pr = pr.child(Button::new(SharedString::from(format!("ab-p{i}"))).label(name).xsmall().ghost().on_click(cx.listener(move |e, _, _, cx| {
                        e.add_preset_artboard(name, w, h);
                        cx.notify();
                    })));
                }
                bar = bar
                    .child(pr)
                    .child(segmented("ab-fmt", &[(0u8, "PNG"), (1, "JPEG"), (2, "PDF")], self.nopts.artboard_format, cx, |e, v| e.nopts.artboard_format = v))
                    .child(btn("ab-export", "Export Artboards…".into()).on_click(cx.listener(|e, _, _, cx| e.export_regions(false, cx))));
                if let Some(id) = self.nopts.artboard_sel.filter(|id| self.artboards().iter().any(|a| a.1 == *id)) {
                    bar = bar.child(btn("ab-del", "Delete Artboard".into()).on_click(cx.listener(move |e, _, _, cx| {
                        e.doc.push_undo("Delete Artboard");
                        e.delete_artboard(id);
                        cx.notify();
                    })));
                }
            }
            Tool::Move => {
                let mut row = div().flex().gap_0p5().child(label("Align:"));
                for (id, l, how) in [("al-l", "⇤ Left", "left"), ("al-c", "↔ Center", "hcenter"), ("al-r", "Right ⇥", "right"), ("al-t", "⤒ Top", "top"), ("al-m", "↕ Middle", "vcenter"), ("al-b", "Bottom ⤓", "bottom")] {
                    row = row.child(Button::new(id).label(l).xsmall().ghost().on_click(cx.listener(move |e, _, _, cx| {
                        e.align_layer(how, cx);
                        cx.notify();
                    })));
                }
                let n = self.selected_layers().len();
                bar = bar.child(row).child(segmented("al-to", &[(false, "To Selection"), (true, "To Key Object")], self.nopts.align_key, cx, |e, v| e.nopts.align_key = v));
                if n == 1 {
                    bar = bar.child(label(if self.doc.selection.is_some() { "(marquee)" } else { "(canvas)" }));
                }
                if n >= 3 {
                    use crate::engine::arrange::Distribute as D;
                    let mut drow = div().flex().gap_0p5().child(label("Distribute:"));
                    for (id, l, how) in [("ds-l", "Left", D::Left), ("ds-c", "H Center", D::HCenter), ("ds-r", "Right", D::Right), ("ds-t", "Top", D::Top), ("ds-m", "V Center", D::VCenter), ("ds-b", "Bottom", D::Bottom), ("ds-hs", "H Spacing", D::HSpace), ("ds-vs", "V Spacing", D::VSpace)] {
                        drow = drow.child(Button::new(id).label(l).xsmall().ghost().on_click(cx.listener(move |e, _, _, cx| {
                            e.distribute_layers(how);
                            cx.notify();
                        })));
                    }
                    bar = bar.child(drow);
                }
                bar = bar.child(label(format!("{n} layer{} selected", if n == 1 { "" } else { "s" })));
            }
            _ => {}
        }
        bar
    }

    /// Info panel rows for the colour samplers.
    pub fn sampler_rows(&self) -> Vec<(String, [u8; 4], crate::engine::color::ColorMode)> {
        self.doc
            .ann
            .samplers
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| (format!("#{} ({x:.0}, {y:.0})", i + 1), self.sample_at(x, y).unwrap_or([0; 4]), self.doc.ann.sampler_modes.get(i).copied().unwrap_or_default()))
            .collect()
    }
}

/// Move (h = 8) or resize (h = handle) rectangle `r0` by the drag `start → q`.
fn edit_rect(r0: FRect, h: u8, start: (f32, f32), q: (f32, f32)) -> FRect {
    if h == 8 {
        let (dx, dy) = (q.0 - start.0, q.1 - start.1);
        (r0.0 + dx, r0.1 + dy, r0.2 + dx, r0.3 + dy)
    } else {
        drag_handle(r0, h, q, None)
    }
}

/// Handle (0..8) or interior (8) of the topmost rect containing `q`; the
/// selected rect's handles win.
fn rect_hit(rects: impl Iterator<Item = RectI>, q: (f32, f32), tol: f32, sel: Option<usize>) -> Option<(usize, u8)> {
    let rs: Vec<RectI> = rects.collect();
    if let Some(i) = sel.filter(|&i| i < rs.len()) {
        if let Some(h) = handles(of_rect(rs[i])).iter().position(|&(x, y)| (x - q.0).hypot(y - q.1) <= tol) {
            return Some((i, h as u8));
        }
    }
    rs.iter().enumerate().rev().find(|(_, r)| q.0 >= r.x0 as f32 && q.0 <= r.x1 as f32 && q.1 >= r.y0 as f32 && q.1 <= r.y1 as f32).map(|(i, _)| (i, 8))
}

impl Editor {
    /// A preset-size artboard to the right of the existing ones.
    pub fn add_preset_artboard(&mut self, name: &str, w: i32, h: i32) -> u64 {
        self.doc.push_undo("Artboard");
        let x0 = self.artboards().iter().map(|a| a.2.x1 + 100).max().unwrap_or(0);
        let n = self.artboards().len() + 1;
        self.new_artboard(format!("{name} {n}"), RectI::new(x0, 0, x0 + w, h))
    }
}

/// Exports every visible artboard (its own layers, clipped to it): one PNG
/// or JPEG per artboard, or a single PDF with a page per artboard.
fn export_artboards_to(layers: &[Layer], format: u8, dir: &std::path::Path) -> anyhow::Result<usize> {
    use crate::engine::webexport::{self, WebFormat};
    use rayon::prelude::*;
    let boards: Vec<usize> = (0..layers.len()).filter(|&i| matches!(layers[i].kind, LayerKind::Artboard(_)) && layers[i].visible).collect();
    let pages: Vec<(String, Vec<u8>, u32, u32)> = boards
        .par_iter()
        .filter_map(|&i| crate::engine::artboard::render(layers, i).map(|(rgba, w, h)| (layers[i].name.clone(), rgba, w, h)))
        .collect();
    let safe = |s: &str| -> String { s.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' { c } else { '_' }).collect() };
    if format == 2 {
        let pdf = webexport::pdf(&pages.iter().map(|p| (p.1.clone(), p.2, p.3)).collect::<Vec<_>>(), 72.0, Some(90)).map_err(anyhow::Error::msg)?;
        std::fs::write(dir.join("Artboards.pdf"), pdf)?;
        return Ok(1);
    }
    for (name, rgba, w, h) in &pages {
        let (fmt, ext) = if format == 1 { (WebFormat::Jpeg { quality: 90 }, "jpg") } else { (WebFormat::Png24 { transparency: true }, "png") };
        let bytes = webexport::encode(rgba, *w, *h, fmt, [255, 255, 255]).map_err(anyhow::Error::msg)?;
        std::fs::write(dir.join(format!("{}.{ext}", safe(name))), bytes)?;
    }
    Ok(pages.len())
}

fn export_regions_to(layers: &[Layer], regions: &[(String, RectI)], dir: &std::path::Path) -> anyhow::Result<usize> {
    use image::ImageEncoder;
    let mut n = 0;
    for (name, r) in regions {
        if r.is_empty() {
            continue;
        }
        let rgba = annot::region_rgba(layers, *r);
        let safe: String = name.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' { c } else { '_' }).collect();
        let f = std::io::BufWriter::new(std::fs::File::create(dir.join(format!("{safe}.png")))?);
        image::codecs::png::PngEncoder::new(f).write_image(&rgba, r.w() as u32, r.h() as u32, image::ExtendedColorType::Rgba8)?;
        n += 1;
    }
    Ok(n)
}
