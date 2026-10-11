//! Retouching tools: Clone / Pattern Stamp, History / Art History Brush,
//! Healing Brush, Spot Healing, Patch, Content-Aware Move and Red Eye.

use std::sync::Arc;
use std::time::Instant;

use gpui_kit::{Bounds, Context, MouseDownEvent, Window, point, px};
use image::{Frame, RgbaImage};

use crate::engine::brush::{BrushSettings, Paint, Stroke};
use crate::engine::document::Layer;
use crate::engine::grid::{RectI, TileGrid};
use crate::engine::masks;
use crate::engine::px as pxm;
use crate::engine::retouch::{self, ArtStyle, PatchMode, RegionFn, SpotMode};
use crate::engine::selection::MaskGrid;

use super::editor::{Drag, Editor, Overlay, Tool};
use super::layers_ui::PaintTarget;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SampleFrom {
    #[default]
    Current,
    CurrentBelow,
    All,
}

impl SampleFrom {
    pub const ALL: [SampleFrom; 3] = [SampleFrom::Current, SampleFrom::CurrentBelow, SampleFrom::All];
    pub fn label(self) -> &'static str {
        match self {
            SampleFrom::Current => "Current Layer",
            SampleFrom::CurrentBelow => "Current & Below",
            SampleFrom::All => "All Layers",
        }
    }
}

/// Retouching options (options bar).
pub struct RetouchOptions {
    pub aligned: bool,
    pub sample: SampleFrom,
    /// Alt-clicked source point (level 0).
    pub source: Option<(f32, f32)>,
    /// Offset kept between strokes in aligned mode.
    pub offset: Option<(i32, i32)>,
    pub overlay: bool,
    pub spot_mode: SpotMode,
    pub patch_mode: PatchMode,
    pub cam_extend: bool,
    pub cam_adapt: f32,
    pub pupil: f32,
    pub darken: f32,
    pub pattern: usize,
    /// History position the history brushes paint from (0 = as opened).
    pub history_pos: usize,
    pub art_style: ArtStyle,
    pub art_area: f32,
    /// Healing solve quality (Healing Brush options bar).
    pub heal_quality: crate::engine::poisson::Quality,
}

impl Default for RetouchOptions {
    fn default() -> Self {
        RetouchOptions {
            aligned: true,
            sample: SampleFrom::Current,
            source: None,
            offset: None,
            overlay: true,
            spot_mode: SpotMode::ContentAware,
            patch_mode: PatchMode::Source,
            cam_extend: false,
            cam_adapt: 0.6,
            pupil: 1.0,
            darken: 0.5,
            pattern: 1,
            history_pos: 0,
            art_style: ArtStyle::TightShort,
            art_area: 50.0,
            heal_quality: Default::default(),
        }
    }
}

/// Spot Healing in progress: the painted area as a gray mask.
pub struct SpotState {
    pub mask: TileGrid,
}

impl Editor {
    /// A level-0 region reader for the clone sources (snapshot at stroke start).
    fn clone_source_fn(&self) -> Arc<RegionFn> {
        let li = self.doc.active;
        match self.ropts.sample {
            SampleFrom::Current => {
                let g = self.doc.layers[li].grid.clone();
                Arc::new(move |x, y, w, h, out: &mut [u32]| g.read_region(x, y, w, h, out))
            }
            SampleFrom::CurrentBelow | SampleFrom::All => {
                let layers: Vec<Layer> = if self.ropts.sample == SampleFrom::All { self.doc.layers.clone() } else { self.doc.layers[..=li].to_vec() };
                Arc::new(move |x, y, w, h, out: &mut [u32]| super::select_tools::compose_src(&layers, 0, x, y, w, h, out))
            }
        }
    }

    /// Mouse-down for the retouching tools. Returns true when handled.
    pub fn retouch_down(&mut self, ix: f32, iy: f32, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let tool = self.tool;
        let li = self.doc.active;
        match tool {
            Tool::CloneStamp | Tool::Heal => {
                if ev.modifiers.alt {
                    self.ropts.source = Some((ix, iy));
                    self.ropts.offset = None;
                    self.flash(format!("Source set at {:.0}, {:.0}", ix, iy));
                    return true;
                }
                let Some(src) = self.ropts.source else {
                    self.flash("Alt-click to set the source point first");
                    return true;
                };
                if !self.ensure_pixels(li) {
                    return true;
                }
                let off = match (self.ropts.aligned, self.ropts.offset) {
                    (true, Some(o)) => o,
                    _ => ((src.0 - ix).round() as i32, (src.1 - iy).round() as i32),
                };
                self.ropts.offset = Some(off);
                let f = self.clone_source_fn();
                let kernel: Box<dyn crate::engine::brush::Kernel> = if tool == Tool::CloneStamp {
                    Box::new(retouch::CloneKernel { src: f, dx: off.0, dy: off.1 })
                } else {
                    Box::new(retouch::HealKernel::new(f, off.0, off.1, self.ropts.heal_quality))
                };
                self.start_kernel_stroke(tool.name(), kernel, ix, iy);
            }
            Tool::PatternStamp => {
                if !self.ensure_pixels(li) {
                    return true;
                }
                let k = Box::new(retouch::PatternKernel { pattern: self.ropts.pattern });
                self.start_kernel_stroke("Pattern Stamp", k, ix, iy);
            }
            Tool::HistoryBrush | Tool::ArtHistory => {
                if !self.ensure_pixels(li) {
                    return true;
                }
                let id = self.doc.layers[li].id;
                let pos = self.ropts.history_pos.min(self.doc.history_len());
                let Some(src) = self.doc.history_source(pos, id) else {
                    self.flash("The chosen history state has no matching layer");
                    return true;
                };
                if (src.w, src.h) != (self.doc.w, self.doc.h) {
                    self.flash("The history state has a different canvas size");
                    return true;
                }
                let k: Box<dyn crate::engine::brush::Kernel> = if tool == Tool::HistoryBrush {
                    Box::new(retouch::HistoryKernel { src })
                } else {
                    let op = (self.brush_ref().opacity * 255.0) as u32;
                    Box::new(retouch::ArtHistoryKernel { src, style: self.ropts.art_style, area: self.ropts.art_area, opacity: op, rng: 0x9E37_79B9 })
                };
                self.start_kernel_stroke(tool.name(), k, ix, iy);
            }
            Tool::SpotHeal => {
                if !self.ensure_pixels(li) {
                    return true;
                }
                let mut mask = TileGrid::new(self.doc.w, self.doc.h);
                let settings = BrushSettings { opacity: 1.0, flow: 1.0, ..*self.brush_ref() };
                let mut st = Box::new(Stroke::new(li, &mask, settings, Box::new(Paint { color: pxm::pack(255, 255, 255, 255) }), None));
                st.add_point_p(ix, iy, self.pressure, &mut mask);
                self.spot = Some(SpotState { mask });
                self.drag = Drag::SpotHeal(st);
                self.refresh_spot_overlay();
            }
            Tool::Patch | Tool::ContentAwareMove => {
                let inside = self.doc.selection.as_ref().is_some_and(|s| s.at(ix as i32, iy as i32) > 0);
                if inside {
                    self.drag = Drag::PatchMove { start: (ix, iy), cur: (ix, iy) };
                } else {
                    self.drag = Drag::Lasso(vec![[ix, iy]], crate::engine::masks::SelOp::New);
                }
            }
            Tool::RedEye => {
                if !self.ensure_pixels(li) {
                    return true;
                }
                let t0 = Instant::now();
                let grid = self.doc.layers[li].grid.clone();
                match retouch::red_eye(&grid, ix as i32, iy as i32, self.ropts.pupil, self.ropts.darken) {
                    Some((g, _)) => {
                        self.doc.push_undo("Red Eye");
                        self.doc.set_layer_grid(li, g);
                        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
                        self.flash(format!("Red Eye — {:.1} ms", self.last_tool_ms));
                    }
                    None => self.flash("No red pupil found near the click"),
                }
            }
            _ => return false,
        }
        let _ = (window, cx);
        true
    }

    /// Starts a dab-engine stroke with any kernel on the active layer.
    pub fn start_kernel_stroke(&mut self, label: &str, kernel: Box<dyn crate::engine::brush::Kernel>, ix: f32, iy: f32) {
        let li = self.doc.active;
        let settings = *self.brush_ref();
        self.doc.push_undo(label);
        let t0 = Instant::now();
        let mut stroke = Box::new(Stroke::new(li, &self.doc.layers[li].grid, settings, kernel, self.doc.selection.clone()));
        let rects = stroke.add_point_p(ix, iy, self.pressure, &mut self.doc.layers[li].grid);
        self.doc.layer_pixels_changed(li, &rects);
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
        self.drag = Drag::Paint(stroke, PaintTarget::Layer(li));
    }

    // ----- spot healing -----------------------------------------------------------------

    pub fn spot_move(&mut self, ix: f32, iy: f32) {
        let pr = self.pressure;
        let Drag::SpotHeal(st) = &mut self.drag else { return };
        if let Some(sp) = &mut self.spot {
            st.add_point_p(ix, iy, pr, &mut sp.mask);
        }
        self.refresh_spot_overlay();
    }

    /// Shows the painted spot-heal area as a translucent dark overlay.
    fn refresh_spot_overlay(&mut self) {
        let Some(sp) = &self.spot else { return };
        let mut ov = TileGrid::new(sp.mask.w, sp.mask.h);
        ov.tiles = sp
            .mask
            .tiles
            .iter()
            .map(|t| t.as_ref().and_then(|t| crate::engine::grid::finish_tile(t.read().iter().map(|&p| pxm::scale(pxm::pack(40, 40, 40, 255), pxm::alpha(p) * 2 / 3)).collect())))
            .collect();
        self.liquify_overlay = Some(Layer::new(0, "Spot", ov));
        self.doc.mark_all_dirty();
    }

    pub fn spot_finish(&mut self, st: Box<Stroke>, cx: &mut Context<Self>) {
        let Some(mut sp) = self.spot.take() else { return };
        st.finish(&mut sp.mask);
        self.liquify_overlay = None;
        self.doc.mark_all_dirty();
        let li = self.doc.active;
        let id = self.doc.layers[li].id;
        let grid = self.doc.layers[li].grid.clone();
        let sel = self.doc.selection.clone();
        let mode = self.ropts.spot_mode;
        let cov = alpha_mask(&sp.mask);
        self.spawn_job("Spot Healing", cx, move |_| {
            let t0 = Instant::now();
            let res = retouch::spot_heal(&grid, &cov, sel.as_ref(), mode);
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            Box::new(move |ed: &mut Editor, _| {
                if let (Some((g, _)), Some(li)) = (res, ed.doc.layers.iter().position(|l| l.id == id)) {
                    ed.doc.push_undo("Spot Healing Brush");
                    ed.doc.set_layer_grid(li, g);
                }
                ed.last_tool_ms = ms;
            })
        });
    }

    // ----- patch / content-aware move ------------------------------------------------------

    pub fn patch_finish(&mut self, start: (f32, f32), cur: (f32, f32), cx: &mut Context<Self>) {
        let (dx, dy) = ((cur.0 - start.0).round() as i32, (cur.1 - start.1).round() as i32);
        if dx == 0 && dy == 0 {
            return;
        }
        let Some(sel) = self.doc.selection.clone() else { return };
        let li = self.doc.active;
        if !self.ensure_pixels(li) {
            return;
        }
        let id = self.doc.layers[li].id;
        let grid = self.doc.layers[li].grid.clone();
        let (w, h) = (self.doc.w, self.doc.h);
        let tool = self.tool;
        let (mode, extend, adapt) = (self.ropts.patch_mode, self.ropts.cam_extend, self.ropts.cam_adapt);
        self.spawn_job(tool.name(), cx, move |_| {
            let m = MaskGrid::from_selection(&sel, w, h);
            let t0 = Instant::now();
            let res = if tool == Tool::Patch { retouch::patch(&grid, &m, dx, dy, mode) } else { retouch::content_aware_move(&grid, &m, dx, dy, extend, adapt) };
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            Box::new(move |ed: &mut Editor, _| {
                if let (Some((g, _)), Some(li)) = (res, ed.doc.layers.iter().position(|l| l.id == id)) {
                    ed.doc.push_undo(tool.name());
                    ed.doc.set_layer_grid(li, g);
                    // The selection follows the moved object / patch target.
                    if tool == Tool::ContentAwareMove || mode == PatchMode::Destination {
                        let moved = masks_translate(&m, dx, dy);
                        ed.doc.selection = moved.to_selection();
                    }
                }
                ed.last_tool_ms = ms;
            })
        });
    }

    // ----- overlays ----------------------------------------------------------------------------

    /// Clone / heal source preview under the cursor, plus the source marker.
    pub fn retouch_overlays(&mut self, out: &mut Vec<Overlay>) {
        if !matches!(self.tool, Tool::CloneStamp | Tool::Heal) {
            return;
        }
        let Some(src) = self.ropts.source else { return };
        let s = self.vp.logical_scale(self.sf);
        // Marker at the sampling point (follows the cursor while painting).
        let (mx, my) = match (self.mouse_img, self.ropts.offset) {
            (Some(m), Some(o)) if matches!(self.drag, Drag::Paint(..)) || self.ropts.aligned => (m.0 + o.0 as f32, m.1 + o.1 as f32),
            _ => src,
        };
        let c = self.vp.to_screen(mx, my, self.canvas, self.sf);
        let r = 6.0;
        out.push(Overlay::Path { polys: vec![(vec![point(c.x - px(r), c.y), point(c.x + px(r), c.y)], false), (vec![point(c.x, c.y - px(r)), point(c.x, c.y + px(r))], false)], dashed: false, color: gpui_kit::hsla(0., 0., 1., 0.9), width: 1.5 });
        if !self.ropts.overlay || matches!(self.drag, Drag::Paint(..)) {
            return;
        }
        let (Some(m), Some(mi)) = (self.mouse, self.mouse_img) else { return };
        let off = match (self.ropts.aligned, self.ropts.offset) {
            (true, Some(o)) => o,
            _ => ((src.0 - mi.0).round() as i32, (src.1 - mi.1).round() as i32),
        };
        // Render the source region at screen resolution (bounded by the brush footprint).
        let rad = (self.brush_ref().size * 0.5).max(4.0);
        let screen_r = (rad * s).clamp(4.0, 400.0);
        let n = ((screen_r * 2.0 * self.sf) as usize).clamp(8, 800);
        let key = (mi.0.round() as i32 + off.0, mi.1.round() as i32 + off.1, n, rad as i32);
        if self.clone_preview.as_ref().is_none_or(|p| p.0 != key) || self.doc_dirty_seen != self.doc.layers[self.doc.active].version {
            let level = ((2.0 * rad / n as f32).log2().floor().max(0.0) as u32).min(self.doc.max_level());
            let k = (1u32 << level) as f32;
            let (x0, y0) = ((mi.0 + off.0 as f32 - rad) / k, (mi.1 + off.1 as f32 - rad) / k);
            let side = ((2.0 * rad) / k).ceil().max(1.0) as usize;
            let mut buf = vec![0u32; side * side];
            let layers: Vec<Layer> = match self.ropts.sample {
                SampleFrom::Current => vec![self.doc.layers[self.doc.active].clone()],
                SampleFrom::CurrentBelow => self.doc.layers[..=self.doc.active].to_vec(),
                SampleFrom::All => self.doc.layers.clone(),
            };
            super::select_tools::compose_src(&layers, level, x0.floor() as i32, y0.floor() as i32, side, side, &mut buf);
            let mut rgba = vec![0u8; n * n * 4];
            for y in 0..n {
                for x in 0..n {
                    let (u, v) = (x * side / n, y * side / n);
                    let p = buf[(v.min(side - 1)) * side + u.min(side - 1)];
                    let [r, g, b, a] = pxm::unpremultiply(p);
                    let o = (y * n + x) * 4;
                    // BGRA, 55% opacity like Photoshop's clipped overlay.
                    rgba[o..o + 4].copy_from_slice(&[b, g, r, (a as u32 * 140 / 255) as u8]);
                }
            }
            let img = crate::ui::render_image(vec![Frame::new(RgbaImage::from_raw(n as u32, n as u32, rgba).expect("overlay"))]);
            if let Some(old) = self.clone_preview.replace((key, img)) {
                self.cache.queue_drop(old.1);
            }
            self.doc_dirty_seen = self.doc.layers[self.doc.active].version;
        }
        let img = self.clone_preview.as_ref().unwrap().1.clone();
        let b = Bounds::from_corners(point(m.x - px(screen_r), m.y - px(screen_r)), point(m.x + px(screen_r), m.y + px(screen_r)));
        out.push(Overlay::Image { bounds: b, img, radius: screen_r });
    }
}

/// The spot-heal coverage from the painted gray tiles' alpha.
fn alpha_mask(g: &TileGrid) -> MaskGrid {
    let mut m = MaskGrid::new(g.w, g.h);
    for (i, t) in g.tiles.iter().enumerate() {
        if let Some(t) = t {
            m.tiles[i] = masks::normalize(t.read().iter().map(|&p| pxm::alpha(p) as u8).collect());
        }
    }
    m
}

/// A mask shifted by whole pixels (tile-aligned shifts are pointer moves).
pub fn masks_translate(m: &MaskGrid, dx: i32, dy: i32) -> MaskGrid {
    let gray = m.to_gray();
    let moved = crate::engine::ops::translate(&gray, dx, dy);
    let mut out = MaskGrid::from_gray(&moved);
    // Pixels shifted in from outside the canvas are unselected.
    let edge = RectI::new(0, 0, m.w as i32, m.h as i32).translate(dx, dy).intersect(&RectI::new(0, 0, m.w as i32, m.h as i32));
    let keep = MaskGrid::from_selection(&crate::engine::selection::Selection::rect(edge), m.w, m.h);
    out = out.combine(&keep, crate::engine::masks::SelOp::Intersect);
    out
}
