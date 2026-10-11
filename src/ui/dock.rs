//! The panel dock (a second column beside the side panel): Brushes (preset
//! folders, `.abr` import), Brush Settings (every dynamic), Measurement Log
//! and Notes. Window ▸ … opens a panel; its tab closes or switches it.

use std::collections::HashMap;
use std::sync::Arc;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputState};
use gpui_kit::component::Sizable as _;
use gpui_kit::prelude::*;
use gpui_kit::{Context, Div, Entity, PathPromptOptions, RenderImage, SharedString, Window, div, img, px, rgb};
use image::{Frame, RgbaImage};

use crate::engine::brush::{self, BrushSettings, Control, Stroke, TEXTURES, TexMode};
use crate::engine::grid::TileGrid;
use crate::engine::measure;
use crate::engine::prefs::Prefs;
use crate::engine::presets::{BrushPreset, PresetFolder, PresetLib};
use crate::engine::px as pxm;

use super::editor::{Editor, Tool};
use super::options::{label, segmented, spec, toggle};

const PANEL: u32 = 0x262628;
const PANEL_HI: u32 = 0x2f2f32;
const BORDER: u32 = 0x3a3a3d;
const TEXT: u32 = 0xdcdcdc;
const DIM: u32 = 0x8e8e93;
const ACCENT: u32 = 0x2f6fde;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DockTab {
    Brushes,
    BrushSettings,
    Measurements,
    Notes,
    Channels,
}

impl DockTab {
    pub const ALL: [DockTab; 5] = [DockTab::Brushes, DockTab::BrushSettings, DockTab::Measurements, DockTab::Notes, DockTab::Channels];

    pub fn label(self) -> &'static str {
        match self {
            DockTab::Brushes => "Brushes",
            DockTab::BrushSettings => "Brush Settings",
            DockTab::Measurements => "Measurement Log",
            DockTab::Notes => "Notes",
            DockTab::Channels => "Channels",
        }
    }
}

/// The dock's state (presets are loaded once per session).
pub struct DockState {
    pub tab: Option<DockTab>,
    pub presets: PresetLib,
    /// Selected preset (folder, index).
    pub sel: Option<(usize, usize)>,
    thumbs: HashMap<u64, Arc<RenderImage>>,
    rename: Option<(Entity<InputState>, String)>,
    author: Option<(Entity<InputState>, String)>,
    /// Scale fields and the scale they were filled from.
    scale: Option<([Entity<InputState>; 3], measure::MeasureScale)>,
    pub prefs: Prefs,
    pub report: Option<String>,
    pub log_sel: Option<usize>,
    /// Strokes started (each gets its own dynamics seed).
    pub strokes: u32,
}

impl Default for DockState {
    fn default() -> Self {
        DockState {
            tab: None,
            presets: PresetLib::load_or_default(),
            sel: None,
            thumbs: HashMap::new(),
            rename: None,
            author: None,
            scale: None,
            prefs: Prefs::load(),
            report: None,
            log_sel: None,
            strokes: 0,
        }
    }
}

/// A stable key of a preset's look (its thumbnails are cached by it).
fn settings_key(s: &BrushSettings, kind: u8) -> u64 {
    let t = format!("{kind}{s:?}");
    t.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

/// BGRA render image from premultiplied pixels over the panel colour.
fn to_image(g: &TileGrid) -> Arc<RenderImage> {
    let (w, h) = (g.w as usize, g.h as usize);
    let mut px = vec![0u32; w * h];
    g.read_region(0, 0, w, h, &mut px);
    let bg = pxm::pack(0x2f, 0x2f, 0x32, 255);
    let mut out = vec![0u8; w * h * 4];
    for (o, &p) in out.chunks_exact_mut(4).zip(&px) {
        let [r, gg, b, _] = pxm::unpack(pxm::over(p, bg));
        o.copy_from_slice(&[b, gg, r, 255]);
    }
    crate::ui::render_image(vec![Frame::new(RgbaImage::from_raw(w as u32, h as u32, out).unwrap())])
}

/// The tip on its own (a single dab, fitted to `side`).
fn tip_thumb(s: &BrushSettings, side: u32) -> TileGrid {
    let mut g = TileGrid::new(side, side);
    let t = BrushSettings { size: side as f32 * 0.85, pressure_size: false, shape: Default::default(), scatter: 0.0, dual: Default::default(), color: Default::default(), transfer: Default::default(), wet_edges: false, ..*s };
    let mut st = Stroke::brush(0, &g, t, [230, 230, 230], false, None);
    st.add_point(side as f32 * 0.5, side as f32 * 0.5, &mut g);
    g
}

/// The stroke preview: an S-curve with rising pressure.
fn stroke_thumb(s: &BrushSettings, w: u32, h: u32) -> TileGrid {
    let mut g = TileGrid::new(w, h);
    let k = (h as f32 * 0.6 / s.size.max(1.0)).min(1.0);
    let t = BrushSettings { size: s.size * k, dual: brush::DualBrush { size: s.dual.size * k, ..s.dual }, smoothing: 0.0, build_up: false, color: brush::ColorDynamics { bg: [120, 170, 250], ..s.color }, ..*s };
    let mut st = Stroke::brush(0, &g, t, [230, 230, 230], false, None);
    for i in 0..=40 {
        let u = i as f32 / 40.0;
        st.add_point_p(8.0 + u * (w as f32 - 16.0), h as f32 * 0.5 + (u * std::f32::consts::TAU).sin() * h as f32 * 0.22, 0.25 + 0.75 * u, &mut g);
    }
    st.finish(&mut g);
    g
}

impl Editor {
    pub fn toggle_dock(&mut self, tab: DockTab) {
        self.dockst.tab = if self.dockst.tab == Some(tab) { None } else { Some(tab) };
    }

    fn thumb(&mut self, s: &BrushSettings, stroke: bool) -> Arc<RenderImage> {
        let key = settings_key(s, stroke as u8);
        if let Some(t) = self.dockst.thumbs.get(&key) {
            return t.clone();
        }
        let t = to_image(&if stroke { stroke_thumb(s, 150, 34) } else { tip_thumb(s, 34) });
        self.dockst.thumbs.insert(key, t.clone());
        t
    }

    fn save_presets(&mut self) {
        if let Err(e) = self.dockst.presets.save() {
            self.flash(format!("Could not save brush presets: {e}"));
        }
    }

    /// The settings a new stroke uses: the tool's, with this stroke's seed
    /// and the background colour for Color Dynamics.
    pub fn stroke_settings(&mut self) -> BrushSettings {
        self.dockst.strokes = self.dockst.strokes.wrapping_add(1);
        let mut s = *self.brush_ref();
        s.seed = s.seed.wrapping_mul(31).wrapping_add(self.dockst.strokes);
        s.color.bg = self.bg;
        s
    }

    /// Selects preset `(f, i)` and loads it into the current painting tool.
    pub fn apply_preset(&mut self, f: usize, i: usize) {
        let Some(p) = self.dockst.presets.folders.get(f).and_then(|x| x.presets.get(i)) else { return };
        let mut s = p.settings;
        let name = p.name.clone();
        let cur = *self.brush_ref();
        if cur.protect_texture {
            // Protect Texture: the current texture survives preset changes.
            (s.texture, s.texture_depth, s.tex) = (cur.texture, cur.texture_depth, cur.tex);
        }
        if !self.tool.is_brush() {
            self.tool = Tool::Brush;
        }
        *self.brush_mut() = s;
        self.dockst.sel = Some((f, i));
        self.flash(format!("Brush preset: {name}"));
    }

    /// New Brush Preset from the current settings (into the selected folder).
    pub fn new_preset(&mut self, name: &str) {
        let s = *self.brush_ref();
        let lib = &mut self.dockst.presets;
        if lib.folders.is_empty() {
            lib.folders.push(PresetFolder { name: "My Brushes".into(), open: true, presets: vec![] });
        }
        let f = self.dockst.sel.map_or(0, |(f, _)| f).min(lib.folders.len() - 1);
        lib.folders[f].presets.push(BrushPreset { name: name.into(), settings: s });
        lib.folders[f].open = true;
        self.dockst.sel = Some((f, lib.folders[f].presets.len() - 1));
        self.save_presets();
    }

    pub fn import_abr(&mut self, path: &std::path::Path) {
        match self.dockst.presets.import_abr_file(path) {
            Ok(r) => {
                let msg = format!(
                    "Imported {} sampled brushes; skipped {} computed{}",
                    r.added,
                    r.skipped_computed,
                    if r.skipped_other > 0 { format!(" and {} other", r.skipped_other) } else { String::new() }
                );
                self.flash(msg.clone());
                self.dockst.report = Some(msg);
                self.save_presets();
            }
            Err(e) => self.flash(format!("Could not import {}: {e}", path.display())),
        }
    }

    fn prompt_import_abr(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some("Import Brushes".into()) });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await {
                if let Some(p) = paths.into_iter().next() {
                    let _ = this.update(cx, |ed, cx| {
                        ed.import_abr(&p);
                        cx.notify();
                    });
                }
            }
        })
        .detach();
    }

    // ----- measurements ----------------------------------------------------------------

    /// Image ▸ Analysis ▸ Record Measurements: the protractor or ruler with
    /// the Ruler tool, the count groups with the Count tool, else the
    /// selection (then the ruler, then the counts).
    pub fn record_measurements(&mut self, cx: &mut Context<Self>) {
        let doc = super::docs::title(&self.doc, self.docs.uid);
        let scale = self.doc.ann.measure.scale.clone();
        let ruler = || -> Option<measure::Record> {
            let (a, b) = self.nopts.ruler?;
            Some(match self.nopts.ruler2 {
                Some(c) => measure::protractor(a, b, c, &scale, &doc),
                None => measure::ruler(a, b, &scale, &doc),
            })
        };
        let counts = || measure::counts(&self.doc.ann.counts, &scale, &doc);
        let recs: Vec<measure::Record> = match self.tool {
            Tool::Ruler => ruler().into_iter().collect(),
            Tool::Count => counts(),
            _ if self.doc.selection.is_some() => {
                // The selection's gray values need the composite: off the UI thread.
                let (layers, w, h) = (self.doc.layers.clone(), self.doc.w, self.doc.h);
                let sel = self.doc.selection.clone().unwrap();
                self.spawn_job("Record Measurements", cx, move |_| {
                    let mask = crate::engine::selection::MaskGrid::from_selection(&sel, w, h);
                    let flat = crate::engine::composite::flatten_layers(&layers, w, h);
                    let gray = |x: i32, y: i32| {
                        let [r, g, b, _] = pxm::unpremultiply(flat.pixel(x, y));
                        measure::ps_gray([r, g, b])
                    };
                    let rec = measure::selection(&mask, &gray, &scale, &doc);
                    Box::new(move |ed: &mut Editor, _| match rec {
                        Some(r) => {
                            ed.doc.push_undo("Record Measurements");
                            let i = ed.doc.ann.measure.push(r);
                            ed.dockst.log_sel = Some(i);
                            ed.dockst.tab = Some(DockTab::Measurements);
                        }
                        None => ed.flash("The selection is empty"),
                    })
                });
                return;
            }
            _ => ruler().map(|r| vec![r]).unwrap_or_else(counts),
        };
        if recs.is_empty() {
            self.flash("Nothing to measure: draw a ruler, count, or make a selection");
            return;
        }
        self.doc.push_undo("Record Measurements");
        for r in recs {
            let i = self.doc.ann.measure.push(r);
            self.dockst.log_sel = Some(i);
        }
        self.dockst.tab = Some(DockTab::Measurements);
    }

    pub fn export_measurements_to(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, measure::csv(&self.doc.ann.measure.records))
    }

    fn prompt_export_measurements(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_new_path(&self.default_dir(), Some("Measurements.csv"));
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(path))) = rx.await {
                let _ = this.update(cx, |ed, cx| {
                    match ed.export_measurements_to(&path) {
                        Ok(()) => ed.flash(format!("Exported {} measurements to {}", ed.doc.ann.measure.records.len(), path.display())),
                        Err(e) => ed.flash(format!("Export failed: {e}")),
                    }
                    cx.notify();
                });
            }
        })
        .detach();
    }

    /// Centres the view on note `i` and edits it.
    pub fn jump_to_note(&mut self, i: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(n) = self.doc.ann.notes.get(i) else { return };
        let (x, y) = (n.x, n.y);
        let c = self.canvas.center();
        let p = self.vp.to_screen(x, y, self.canvas, self.sf);
        let d = self.vp.unrotate_delta(c - p);
        self.vp.pan = self.vp.pan + d;
        self.set_note_selection(Some(i), window, cx);
    }

    // ----- rendering ---------------------------------------------------------------------

    pub fn render_dock(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Option<Div> {
        let tab = self.dockst.tab?;
        let mut tabs = div().flex().flex_wrap().gap_0p5().p_1().border_b_1().border_color(rgb(BORDER));
        for t in DockTab::ALL {
            let b = Button::new(SharedString::from(format!("dock-{}", t.label()))).label(t.label()).xsmall();
            tabs = tabs.child(if t == tab { b.primary() } else { b.ghost() }.on_click(cx.listener(move |e, _, _, cx| {
                e.toggle_dock(t);
                cx.notify();
            })));
        }
        let body = match tab {
            DockTab::Brushes => self.render_brushes(window, cx),
            DockTab::BrushSettings => self.render_brush_settings(window, cx),
            DockTab::Measurements => self.render_measurements(window, cx),
            DockTab::Notes => self.render_notes(window, cx),
            DockTab::Channels => self.render_channels(),
        };
        Some(
            div()
                .w(px(310.))
                .flex()
                .flex_col()
                .bg(rgb(PANEL))
                .border_l_1()
                .border_color(rgb(BORDER))
                .child(tabs)
                .child(div().id("dock-body").flex_1().min_h_0().overflow_y_scroll().child(body)),
        )
    }

    fn render_channels(&self) -> Div {
        let mut body = div().flex().flex_col().gap_1().p_2().text_size(px(12.)).text_color(rgb(TEXT));
        let names = crate::engine::psd::mapping::channel_labels(self.doc.psd.as_deref());
        for (index, name) in names.iter().enumerate() {
            body = body.child(div().flex().justify_between().py_1().border_b_1().border_color(rgb(BORDER))
                .child(name.clone()).child(div().text_color(rgb(DIM)).child(if index < 3 { "Colour" } else { "Preserved" })));
        }
        if names.len() > 3 {
            body = body.child(div().pt_2().text_color(rgb(DIM)).child("Extra channels are preserved on save. Channel editing is unavailable."));
        }
        body
    }

    fn render_brushes(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let btn = |id: &str, l: &str| Button::new(SharedString::from(id.to_string())).label(l.to_string()).xsmall().ghost();
        let mut col = div().flex().flex_col().gap_1().p_2().text_size(px(12.));
        // Rename field.
        if let Some((input, last)) = self.dockst.rename.clone() {
            let v = input.read(cx).value().to_string();
            if v != last {
                if let Some((f, i)) = self.dockst.sel {
                    if let Some(p) = self.dockst.presets.folders.get_mut(f).and_then(|x| x.presets.get_mut(i)) {
                        p.name = v.clone();
                    }
                }
                self.dockst.rename = Some((input.clone(), v));
            }
            col = col.child(div().flex().gap_1().child(div().flex_1().child(Input::new(&input).small())).child(btn("br-ren-ok", "Done").on_click(cx.listener(|e, _, _, cx| {
                e.dockst.rename = None;
                e.save_presets();
                cx.notify();
            }))));
        }
        col = col.child(
            div()
                .flex()
                .flex_wrap()
                .gap_0p5()
                .child(btn("br-new", "New Preset").on_click(cx.listener(|e, _, _, cx| {
                    let n = e.dockst.presets.folders.iter().map(|f| f.presets.len()).sum::<usize>() + 1;
                    e.new_preset(&format!("Brush Preset {n}"));
                    cx.notify();
                })))
                .child(btn("br-folder", "New Folder").on_click(cx.listener(|e, _, _, cx| {
                    let n = e.dockst.presets.folders.len() + 1;
                    e.dockst.presets.folders.push(PresetFolder { name: format!("Folder {n}"), open: true, presets: vec![] });
                    e.dockst.sel = Some((e.dockst.presets.folders.len() - 1, 0));
                    e.save_presets();
                    cx.notify();
                })))
                .child(btn("br-import", "Import .abr…").on_click(cx.listener(|e, _, _, cx| e.prompt_import_abr(cx))))
                .child(btn("br-rename", "Rename").on_click(cx.listener(|e, _, window, cx| {
                    if let Some((f, i)) = e.dockst.sel {
                        if let Some(p) = e.dockst.presets.folders.get(f).and_then(|x| x.presets.get(i)) {
                            let name = p.name.clone();
                            e.dockst.rename = Some((cx.new(|cx| InputState::new(window, cx).default_value(name.clone())), name));
                        }
                    }
                    cx.notify();
                })))
                .child(btn("br-up", "↑").on_click(cx.listener(|e, _, _, cx| {
                    if let Some((f, i)) = e.dockst.sel {
                        if i < e.dockst.presets.folders[f].presets.len() {
                            e.dockst.sel = Some(e.dockst.presets.move_preset(f, i, -1));
                            e.save_presets();
                        }
                    }
                    cx.notify();
                })))
                .child(btn("br-down", "↓").on_click(cx.listener(|e, _, _, cx| {
                    if let Some((f, i)) = e.dockst.sel {
                        if i < e.dockst.presets.folders[f].presets.len() {
                            e.dockst.sel = Some(e.dockst.presets.move_preset(f, i, 1));
                            e.save_presets();
                        }
                    }
                    cx.notify();
                })))
                .child(btn("br-del", "Delete").on_click(cx.listener(|e, _, _, cx| {
                    if let Some((f, i)) = e.dockst.sel {
                        let folder = &mut e.dockst.presets.folders[f];
                        if i < folder.presets.len() {
                            folder.presets.remove(i);
                        } else if folder.presets.is_empty() {
                            e.dockst.presets.folders.remove(f);
                        }
                        e.dockst.sel = None;
                        e.save_presets();
                    }
                    cx.notify();
                }))),
        );
        if let Some(r) = &self.dockst.report {
            col = col.child(div().text_color(rgb(DIM)).child(r.clone()));
        }
        let _ = window;
        let folders = self.dockst.presets.folders.clone();
        for (f, folder) in folders.iter().enumerate() {
            let open = folder.open;
            col = col.child(
                div()
                    .id(SharedString::from(format!("br-f-{f}")))
                    .flex()
                    .gap_1()
                    .py_0p5()
                    .text_color(rgb(TEXT))
                    .child(if open { "▾" } else { "▸" })
                    .child(format!("{} ({})", folder.name, folder.presets.len()))
                    .on_click(cx.listener(move |e, _, _, cx| {
                        if let Some(x) = e.dockst.presets.folders.get_mut(f) {
                            x.open = !x.open;
                        }
                        if e.dockst.sel.is_none_or(|s| s.0 != f) {
                            e.dockst.sel = Some((f, usize::MAX));
                        }
                        e.save_presets();
                        cx.notify();
                    })),
            );
            if !open {
                continue;
            }
            for (i, p) in folder.presets.iter().enumerate() {
                let sel = self.dockst.sel == Some((f, i));
                let (tip, stroke) = (self.thumb(&p.settings, false), self.thumb(&p.settings, true));
                col = col.child(
                    div()
                        .id(SharedString::from(format!("br-p-{f}-{i}")))
                        .flex()
                        .items_center()
                        .gap_1p5()
                        .pl_3()
                        .rounded_sm()
                        .bg(rgb(if sel { ACCENT } else { PANEL }))
                        .hover(|s| s.bg(rgb(PANEL_HI)))
                        .child(img(tip).w(px(34.)).h(px(34.)))
                        .child(div().flex().flex_col().min_w_0().child(div().text_color(rgb(TEXT)).child(p.name.clone())).child(img(stroke).w(px(150.)).h(px(34.))))
                        .on_click(cx.listener(move |e, _, _, cx| {
                            e.apply_preset(f, i);
                            cx.notify();
                        })),
                );
            }
        }
        col
    }

    fn ctl_button(&self, id: &'static str, cur: Control, cx: &mut Context<Self>, set: fn(&mut BrushSettings, Control)) -> Button {
        Button::new(id).label(format!("Control: {}", cur.label())).xsmall().ghost().on_click(cx.listener(move |e, _, _, cx| {
            let i = Control::MENU.iter().position(|c| std::mem::discriminant(c) == std::mem::discriminant(&cur)).unwrap_or(0);
            set(e.brush_mut(), Control::MENU[(i + 1) % Control::MENU.len()]);
            cx.notify();
        }))
    }

    fn mode_button(&self, id: &'static str, cur: TexMode, cx: &mut Context<Self>, set: fn(&mut BrushSettings, TexMode)) -> Button {
        Button::new(id).label(format!("Mode: {}", cur.label())).xsmall().ghost().on_click(cx.listener(move |e, _, _, cx| {
            let i = TexMode::ALL.iter().position(|&m| m == cur).unwrap_or(0);
            set(e.brush_mut(), TexMode::ALL[(i + 1) % TexMode::ALL.len()]);
            cx.notify();
        }))
    }

    fn render_brush_settings(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let b = *self.brush_ref();
        let pct = |v: f32| format!("{v:.0}%");
        let sect = |t: &str| div().mt_2().pb_0p5().text_color(rgb(DIM)).border_b_1().border_color(rgb(BORDER)).child(t.to_uppercase());
        let sw = 110.;
        let tip_name = brush::tip(b.tip).map_or("Computed round".to_string(), |t| t.name.clone());
        let mut col = div().flex().flex_col().gap_1().p_2().text_size(px(12.));
        if !self.tool.is_brush() {
            col = col.child(div().text_color(rgb(DIM)).child("Settings of the Brush tool (choose a painting tool to edit its own)."));
        }
        // Brush tip shape.
        col = col
            .child(sect("Brush Tip Shape"))
            .child(div().flex().gap_1().items_center().child(label(format!("Tip: {tip_name}"))).child(Button::new("bs-round").label("Round").xsmall().ghost().on_click(cx.listener(|e, _, _, cx| {
                e.brush_mut().tip = 0;
                cx.notify();
            }))))
            .child(self.opt_slider("Angle", "bs.angle", spec(-180.0, 180.0, 1.0, |e| e.brush_ref().angle, |e, v| e.brush_mut().angle = v), |v| format!("{v:.0}°"), sw, window, cx))
            .child(self.opt_slider("Roundness", "bs.round", spec(1.0, 100.0, 1.0, |e| e.brush_ref().roundness * 100.0, |e, v| e.brush_mut().roundness = v / 100.0), pct, sw, window, cx))
            .child(self.opt_slider("Spacing", "bs.spacing", spec(1.0, 1000.0, 1.0, |e| e.brush_ref().spacing * 100.0, |e, v| e.brush_mut().spacing = v / 100.0), pct, sw, window, cx))
            .child(div().flex().gap_1().child(toggle("bs-fx", "Flip X", b.flip_x).on_click(cx.listener(|e, _, _, cx| {
                e.brush_mut().flip_x ^= true;
                cx.notify();
            }))).child(toggle("bs-fy", "Flip Y", b.flip_y).on_click(cx.listener(|e, _, _, cx| {
                e.brush_mut().flip_y ^= true;
                cx.notify();
            }))));
        // Shape dynamics.
        col = col.child(div().flex().items_center().justify_between().child(sect("Shape Dynamics")).child(toggle("bs-sh", "On", b.shape.on).on_click(cx.listener(|e, _, _, cx| {
            e.brush_mut().shape.on ^= true;
            cx.notify();
        }))));
        if b.shape.on {
            col = col
                .child(self.opt_slider("Size Jitter", "bs.sj", spec(0.0, 100.0, 1.0, |e| e.brush_ref().shape.size_jitter * 100.0, |e, v| e.brush_mut().shape.size_jitter = v / 100.0), pct, sw, window, cx))
                .child(self.ctl_button("bs-sc", b.shape.size_ctl, cx, |s, c| s.shape.size_ctl = c))
                .child(self.opt_slider("Min Diameter", "bs.md", spec(0.0, 100.0, 1.0, |e| e.brush_ref().shape.min_size * 100.0, |e, v| e.brush_mut().shape.min_size = v / 100.0), pct, sw, window, cx))
                .child(self.opt_slider("Angle Jitter", "bs.aj", spec(0.0, 100.0, 1.0, |e| e.brush_ref().shape.angle_jitter * 100.0, |e, v| e.brush_mut().shape.angle_jitter = v / 100.0), pct, sw, window, cx))
                .child(self.ctl_button("bs-ac", b.shape.angle_ctl, cx, |s, c| s.shape.angle_ctl = c))
                .child(self.opt_slider("Round Jitter", "bs.rj", spec(0.0, 100.0, 1.0, |e| e.brush_ref().shape.round_jitter * 100.0, |e, v| e.brush_mut().shape.round_jitter = v / 100.0), pct, sw, window, cx))
                .child(self.ctl_button("bs-rc", b.shape.round_ctl, cx, |s, c| s.shape.round_ctl = c))
                .child(self.opt_slider("Min Round", "bs.mr", spec(1.0, 100.0, 1.0, |e| e.brush_ref().shape.min_round * 100.0, |e, v| e.brush_mut().shape.min_round = v / 100.0), pct, sw, window, cx))
                .child(div().flex().gap_1().child(toggle("bs-fxj", "Flip X Jitter", b.shape.flip_x_jitter).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().shape.flip_x_jitter ^= true;
                    cx.notify();
                }))).child(toggle("bs-fyj", "Flip Y Jitter", b.shape.flip_y_jitter).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().shape.flip_y_jitter ^= true;
                    cx.notify();
                }))));
        }
        // Scattering.
        col = col
            .child(sect("Scattering"))
            .child(self.opt_slider("Scatter", "bs.scat", spec(0.0, 1000.0, 1.0, |e| e.brush_ref().scatter * 100.0, |e, v| e.brush_mut().scatter = v / 100.0), pct, sw, window, cx))
            .child(div().flex().gap_1().child(toggle("bs-both", "Both Axes", b.scattering.both_axes).on_click(cx.listener(|e, _, _, cx| {
                e.brush_mut().scattering.both_axes ^= true;
                cx.notify();
            }))).child(self.ctl_button("bs-scc", b.scattering.ctl, cx, |s, c| s.scattering.ctl = c)))
            .child(self.opt_slider("Count", "bs.cnt", spec(1.0, 16.0, 1.0, |e| e.brush_ref().scattering.count as f32, |e, v| e.brush_mut().scattering.count = v as u8), |v| format!("{v:.0}"), sw, window, cx))
            .child(self.opt_slider("Count Jitter", "bs.cj", spec(0.0, 100.0, 1.0, |e| e.brush_ref().scattering.count_jitter * 100.0, |e, v| e.brush_mut().scattering.count_jitter = v / 100.0), pct, sw, window, cx))
            .child(self.ctl_button("bs-ccc", b.scattering.count_ctl, cx, |s, c| s.scattering.count_ctl = c));
        // Texture.
        col = col.child(sect("Texture")).child(segmented("bs-tex", &[(0u8, TEXTURES[0]), (1, TEXTURES[1]), (2, TEXTURES[2]), (3, TEXTURES[3])], b.texture, cx, |e, v| e.brush_mut().texture = v));
        if b.texture != 0 {
            col = col
                .child(div().flex().gap_1().child(toggle("bs-tinv", "Invert", b.tex.invert).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().tex.invert ^= true;
                    cx.notify();
                }))).child(toggle("bs-teach", "Texture Each Tip", b.tex.each_tip).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().tex.each_tip ^= true;
                    cx.notify();
                }))).child(self.mode_button("bs-tmode", b.tex.mode, cx, |s, m| s.tex.mode = m)))
                .child(self.opt_slider("Scale", "bs.tscale", spec(1.0, 1000.0, 1.0, |e| e.brush_ref().tex.scale * 100.0, |e, v| e.brush_mut().tex.scale = v / 100.0), pct, sw, window, cx))
                .child(self.opt_slider("Brightness", "bs.tbri", spec(-150.0, 150.0, 1.0, |e| e.brush_ref().tex.brightness * 150.0, |e, v| e.brush_mut().tex.brightness = v / 150.0), |v| format!("{v:.0}"), sw, window, cx))
                .child(self.opt_slider("Contrast", "bs.tcon", spec(-50.0, 100.0, 1.0, |e| e.brush_ref().tex.contrast * 100.0, |e, v| e.brush_mut().tex.contrast = v / 100.0), |v| format!("{v:.0}"), sw, window, cx))
                .child(self.opt_slider("Depth", "bs.tdepth", spec(0.0, 100.0, 1.0, |e| e.brush_ref().texture_depth * 100.0, |e, v| e.brush_mut().texture_depth = v / 100.0), pct, sw, window, cx));
            if b.tex.each_tip {
                col = col
                    .child(self.opt_slider("Min Depth", "bs.tmin", spec(0.0, 100.0, 1.0, |e| e.brush_ref().tex.min_depth * 100.0, |e, v| e.brush_mut().tex.min_depth = v / 100.0), pct, sw, window, cx))
                    .child(self.opt_slider("Depth Jitter", "bs.tjit", spec(0.0, 100.0, 1.0, |e| e.brush_ref().tex.depth_jitter * 100.0, |e, v| e.brush_mut().tex.depth_jitter = v / 100.0), pct, sw, window, cx))
                    .child(self.ctl_button("bs-tdc", b.tex.depth_ctl, cx, |s, c| s.tex.depth_ctl = c));
            }
        }
        // Dual brush.
        col = col.child(div().flex().items_center().justify_between().child(sect("Dual Brush")).child(toggle("bs-dual", "On", b.dual.on).on_click(cx.listener(|e, _, _, cx| {
            e.brush_mut().dual.on ^= true;
            cx.notify();
        }))));
        if b.dual.on {
            let dual_tip = brush::tip(b.dual.tip).map_or("Hard round".to_string(), |t| t.name.clone());
            col = col
                .child(div().flex().flex_wrap().gap_1().items_center().child(label(format!("Tip: {dual_tip}"))).child(Button::new("bs-dtip").label("Use selected preset's tip").xsmall().ghost().on_click(cx.listener(|e, _, _, cx| {
                    let t = e.dockst.sel.and_then(|(f, i)| e.dockst.presets.folders.get(f).and_then(|x| x.presets.get(i))).map_or(0, |p| p.settings.tip);
                    e.brush_mut().dual.tip = t;
                    cx.notify();
                }))).child(self.mode_button("bs-dmode", b.dual.mode, cx, |s, m| s.dual.mode = m)))
                .child(self.opt_slider("Size", "bs.dsize", spec(1.0, 1000.0, 1.0, |e| e.brush_ref().dual.size, |e, v| e.brush_mut().dual.size = v), |v| format!("{v:.0} px"), sw, window, cx))
                .child(self.opt_slider("Spacing", "bs.dsp", spec(1.0, 1000.0, 1.0, |e| e.brush_ref().dual.spacing * 100.0, |e, v| e.brush_mut().dual.spacing = v / 100.0), pct, sw, window, cx))
                .child(self.opt_slider("Scatter", "bs.dsc", spec(0.0, 1000.0, 1.0, |e| e.brush_ref().dual.scatter * 100.0, |e, v| e.brush_mut().dual.scatter = v / 100.0), pct, sw, window, cx))
                .child(self.opt_slider("Count", "bs.dcnt", spec(1.0, 16.0, 1.0, |e| e.brush_ref().dual.count as f32, |e, v| e.brush_mut().dual.count = v as u8), |v| format!("{v:.0}"), sw, window, cx))
                .child(div().flex().gap_1().child(toggle("bs-dboth", "Both Axes", b.dual.both_axes).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().dual.both_axes ^= true;
                    cx.notify();
                }))).child(toggle("bs-dflip", "Flip", b.dual.flip).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().dual.flip ^= true;
                    cx.notify();
                }))));
        }
        // Color dynamics.
        col = col.child(div().flex().items_center().justify_between().child(sect("Color Dynamics")).child(toggle("bs-col", "On", b.color.on).on_click(cx.listener(|e, _, _, cx| {
            e.brush_mut().color.on ^= true;
            cx.notify();
        }))));
        if b.color.on {
            col = col
                .child(div().flex().gap_1().child(toggle("bs-pertip", "Apply Per Tip", b.color.per_tip).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().color.per_tip ^= true;
                    cx.notify();
                }))).child(self.ctl_button("bs-fbc", b.color.fg_bg_ctl, cx, |s, c| s.color.fg_bg_ctl = c)))
                .child(self.opt_slider("FG/BG Jitter", "bs.fbj", spec(0.0, 100.0, 1.0, |e| e.brush_ref().color.fg_bg_jitter * 100.0, |e, v| e.brush_mut().color.fg_bg_jitter = v / 100.0), pct, sw, window, cx))
                .child(self.opt_slider("Hue Jitter", "bs.hj", spec(0.0, 100.0, 1.0, |e| e.brush_ref().color.hue_jitter * 100.0, |e, v| e.brush_mut().color.hue_jitter = v / 100.0), pct, sw, window, cx))
                .child(self.opt_slider("Sat. Jitter", "bs.satj", spec(0.0, 100.0, 1.0, |e| e.brush_ref().color.sat_jitter * 100.0, |e, v| e.brush_mut().color.sat_jitter = v / 100.0), pct, sw, window, cx))
                .child(self.opt_slider("Bright. Jitter", "bs.brij", spec(0.0, 100.0, 1.0, |e| e.brush_ref().color.bri_jitter * 100.0, |e, v| e.brush_mut().color.bri_jitter = v / 100.0), pct, sw, window, cx))
                .child(self.opt_slider("Purity", "bs.pur", spec(-100.0, 100.0, 1.0, |e| e.brush_ref().color.purity * 100.0, |e, v| e.brush_mut().color.purity = v / 100.0), pct, sw, window, cx));
        }
        // Transfer.
        col = col.child(div().flex().items_center().justify_between().child(sect("Transfer")).child(toggle("bs-tr", "On", b.transfer.on).on_click(cx.listener(|e, _, _, cx| {
            e.brush_mut().transfer.on ^= true;
            cx.notify();
        }))));
        if b.transfer.on {
            col = col
                .child(self.opt_slider("Opacity Jitter", "bs.oj", spec(0.0, 100.0, 1.0, |e| e.brush_ref().transfer.opacity_jitter * 100.0, |e, v| e.brush_mut().transfer.opacity_jitter = v / 100.0), pct, sw, window, cx))
                .child(self.ctl_button("bs-oc", b.transfer.opacity_ctl, cx, |s, c| s.transfer.opacity_ctl = c))
                .child(self.opt_slider("Flow Jitter", "bs.fj", spec(0.0, 100.0, 1.0, |e| e.brush_ref().transfer.flow_jitter * 100.0, |e, v| e.brush_mut().transfer.flow_jitter = v / 100.0), pct, sw, window, cx))
                .child(self.ctl_button("bs-fc", b.transfer.flow_ctl, cx, |s, c| s.transfer.flow_ctl = c));
        }
        // Other options.
        col.child(sect("Options")).child(
            div()
                .flex()
                .flex_wrap()
                .gap_1()
                .child(toggle("bs-noise", "Noise", b.noise).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().noise ^= true;
                    cx.notify();
                })))
                .child(toggle("bs-wet", "Wet Edges", b.wet_edges).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().wet_edges ^= true;
                    cx.notify();
                })))
                .child(toggle("bs-build", "Build-up", b.build_up).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().build_up ^= true;
                    cx.notify();
                })))
                .child(toggle("bs-smooth", "Smoothing", b.smoothing > 0.0).on_click(cx.listener(|e, _, _, cx| {
                    let s = e.brush_mut();
                    s.smoothing = if s.smoothing > 0.0 { 0.0 } else { 0.1 };
                    cx.notify();
                })))
                .child(toggle("bs-protect", "Protect Texture", b.protect_texture).on_click(cx.listener(|e, _, _, cx| {
                    e.brush_mut().protect_texture ^= true;
                    cx.notify();
                }))),
        )
    }

    fn render_measurements(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let btn = |id: &str, l: &str| Button::new(SharedString::from(id.to_string())).label(l.to_string()).xsmall().ghost();
        // Scale fields.
        // (Re)fill the fields when the document's scale changed elsewhere
        // (another document, undo, a script).
        let s = self.doc.ann.measure.scale.clone();
        if self.dockst.scale.as_ref().is_none_or(|(_, last)| *last != s) {
            let mk = |v: String, window: &mut Window, cx: &mut Context<Self>| cx.new(|cx| InputState::new(window, cx).default_value(v));
            self.dockst.scale = Some(([mk(s.px.to_string(), window, cx), mk(s.length.to_string(), window, cx), mk(s.units.clone(), window, cx)], s));
        }
        let inputs = self.dockst.scale.clone().unwrap().0;
        let mut col = div().flex().flex_col().gap_1().p_2().text_size(px(12.));
        col = col
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_0p5()
                    .child(btn("ml-rec", "Record Measurements").on_click(cx.listener(|e, _, _, cx| {
                        e.record_measurements(cx);
                        cx.notify();
                    })))
                    .child(btn("ml-csv", "Export CSV…").on_click(cx.listener(|e, _, _, cx| e.prompt_export_measurements(cx))))
                    .child(btn("ml-del", "Delete").on_click(cx.listener(|e, _, _, cx| {
                        if let Some(i) = e.dockst.log_sel.filter(|&i| i < e.doc.ann.measure.records.len()) {
                            e.doc.push_undo("Delete Measurement");
                            e.doc.ann.measure.records.remove(i);
                            e.dockst.log_sel = None;
                        }
                        cx.notify();
                    })))
                    .child(btn("ml-clear", "Clear").on_click(cx.listener(|e, _, _, cx| {
                        e.doc.push_undo("Clear Measurements");
                        e.doc.ann.measure.records.clear();
                        cx.notify();
                    }))),
            )
            .child(div().text_color(rgb(DIM)).child("Measurement scale"))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(div().w(px(56.)).child(Input::new(&inputs[0]).small()))
                    .child(label("px ="))
                    .child(div().w(px(56.)).child(Input::new(&inputs[1]).small()))
                    .child(div().w(px(64.)).child(Input::new(&inputs[2]).small()))
                    .child(btn("ml-scale", "Set").on_click(cx.listener(|e, _, _, cx| {
                        let Some((i, _)) = e.dockst.scale.clone() else { return };
                        let (p, l, u) = (i[0].read(cx).value().to_string(), i[1].read(cx).value().to_string(), i[2].read(cx).value().to_string());
                        match (p.trim().parse::<f32>(), l.trim().parse::<f32>()) {
                            (Ok(p), Ok(l)) if p > 0.0 && l > 0.0 => {
                                e.doc.push_undo("Set Measurement Scale");
                                e.doc.ann.measure.scale = measure::MeasureScale { px: p, length: l, units: if u.trim().is_empty() { "pixels".into() } else { u.trim().into() } };
                                e.flash(format!("Measurement scale: {}", e.doc.ann.measure.scale.label()));
                            }
                            _ => e.flash("Enter two positive numbers for the scale"),
                        }
                        cx.notify();
                    }))),
            )
            .child(div().text_color(rgb(DIM)).child(format!("Scale: {}", self.doc.ann.measure.scale.label())));
        let f = |v: Option<f64>| v.map_or("—".to_string(), |v| if v.abs() >= 100.0 { format!("{v:.0}") } else { format!("{v:.2}") });
        for (i, r) in self.doc.ann.measure.records.iter().enumerate().rev() {
            let sel = self.dockst.log_sel == Some(i);
            let units = &r.scale.units;
            let detail = match r.source {
                measure::Source::Selection => format!(
                    "Area {} {units}² · Perim. {} · Circ. {} · Centroid {}, {} · Gray {} · Regions {}",
                    f(r.area),
                    f(r.perimeter),
                    f(r.circularity),
                    f(r.cx),
                    f(r.cy),
                    f(r.gray),
                    r.count.unwrap_or(0)
                ),
                measure::Source::Ruler => format!("Length {} {units} · Angle {}° · W {} H {}", f(r.length), f(r.angle), f(r.width), f(r.height)),
                measure::Source::Protractor => format!("Angle {}° · L1 {} {units}", f(r.angle), f(r.length)),
                measure::Source::Count => format!("Count {}", r.count.unwrap_or(0)),
            };
            col = col.child(
                div()
                    .id(SharedString::from(format!("ml-r-{i}")))
                    .p_1()
                    .rounded_sm()
                    .bg(rgb(if sel { ACCENT } else { PANEL_HI }))
                    .child(div().flex().justify_between().child(div().text_color(rgb(TEXT)).child(format!("{} · {}", r.label, r.source.label()))).child(div().text_color(rgb(DIM)).child(r.when.get(11..16).unwrap_or("").to_string())))
                    .child(div().text_color(rgb(DIM)).child(detail))
                    .on_click(cx.listener(move |e, _, _, cx| {
                        e.dockst.log_sel = Some(i);
                        cx.notify();
                    })),
            );
        }
        col
    }

    fn render_notes(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let btn = |id: &str, l: &str| Button::new(SharedString::from(id.to_string())).label(l.to_string()).xsmall().ghost();
        // Author (a preference).
        if self.dockst.author.is_none() {
            let a = self.dockst.prefs.author.clone();
            self.dockst.author = Some((cx.new(|cx| InputState::new(window, cx).default_value(a.clone())), a));
        }
        let (author_in, last) = self.dockst.author.clone().unwrap();
        let v = author_in.read(cx).value().to_string();
        if v != last {
            self.dockst.prefs.author = v.clone();
            let _ = self.dockst.prefs.save();
            self.dockst.author = Some((author_in.clone(), v));
        }
        // The selected note's text field (shared with the Note tool).
        if let Some((i, input, last)) = self.nopts.note_input.clone() {
            let v = input.read(cx).value().to_string();
            if v != last {
                if let Some(n) = self.doc.ann.notes.get_mut(i) {
                    n.text = v.clone();
                }
                self.nopts.note_input = Some((i, input, v));
            }
        }
        let mut col = div()
            .flex()
            .flex_col()
            .gap_1()
            .p_2()
            .text_size(px(12.))
            .child(div().flex().items_center().gap_1().child(label("Author")).child(div().flex_1().child(Input::new(&author_in).small())))
            .child(div().text_color(rgb(DIM)).child(format!("{} notes · new notes are signed by the author above", self.doc.ann.notes.len())));
        for (i, n) in self.doc.ann.notes.iter().enumerate() {
            let sel = self.nopts.note_sel == Some(i);
            let first = n.text.lines().next().unwrap_or("").to_string();
            col = col.child(
                div()
                    .id(SharedString::from(format!("nt-{i}")))
                    .p_1()
                    .rounded_sm()
                    .bg(rgb(if sel { ACCENT } else { PANEL_HI }))
                    .child(div().flex().justify_between().child(div().text_color(rgb(TEXT)).child(format!("#{} {}", i + 1, n.author))).child(div().text_color(rgb(DIM)).child(format!("{:.0}, {:.0}", n.x, n.y))))
                    .child(div().text_color(rgb(if first.is_empty() { DIM } else { TEXT })).child(if first.is_empty() { "(empty)".to_string() } else { first }))
                    .on_click(cx.listener(move |e, _, window, cx| {
                        e.jump_to_note(i, window, cx);
                        cx.notify();
                    })),
            );
            if sel {
                if let Some((_, input, _)) = &self.nopts.note_input {
                    col = col.child(Input::new(input).small()).child(
                        div()
                            .flex()
                            .gap_1()
                            .child(btn("nt-author", "Sign with my name").on_click(cx.listener(move |e, _, _, cx| {
                                let a = e.dockst.prefs.author.clone();
                                if let Some(n) = e.doc.ann.notes.get_mut(i) {
                                    n.author = a;
                                }
                                cx.notify();
                            })))
                            .child(btn("nt-del", "Delete Note").on_click(cx.listener(move |e, _, _, cx| {
                                if i < e.doc.ann.notes.len() {
                                    e.doc.push_undo("Delete Note");
                                    e.doc.ann.notes.remove(i);
                                }
                                e.nopts.note_sel = None;
                                e.nopts.note_input = None;
                                cx.notify();
                            }))),
                    );
                }
            }
        }
        col
    }
}
