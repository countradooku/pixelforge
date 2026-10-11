//! Painting tools: Pencil, Color Replacement, Mixer Brush, Blur / Sharpen /
//! Smudge, Dodge / Burn / Sponge, Background and Magic Eraser, the live
//! Gradient (with on-canvas handles and an editor) and the pattern bucket.

use std::sync::Arc;
use std::time::Instant;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::Sizable as _;
use gpui_kit::prelude::*;
use gpui_kit::{Bounds, Context, Div, MouseDownEvent, SharedString, Window, div, point, px, rgb};

use crate::engine::brush::Kernel;
use crate::engine::composite::{self, Preview};
use crate::engine::filters::Filter;
use crate::engine::gradient::{AlphaStop, ColorStop, GradKind, Gradient, GradientFill};
use crate::engine::ops;
use crate::engine::paint::{self, Range, ReplaceMode, Sampling, Tone};
use crate::engine::px as pxm;

use super::editor::{Drag, Editor, Overlay, Tool};
use super::options::{label, segmented, spec, toggle};

pub struct PaintOptions {
    pub range: Range,
    pub exposure: f32,
    pub protect_tones: bool,
    pub sponge_saturate: bool,
    pub vibrance: bool,
    pub focus_strength: f32,
    pub smudge_strength: f32,
    pub finger: bool,
    pub wet: f32,
    pub load: f32,
    pub mix: f32,
    pub mixer_flow: f32,
    /// Reservoir colour; None = clean brush.
    pub mixer_color: Option<[u8; 3]>,
    pub load_after: bool,
    pub clean_after: bool,
    pub replace_mode: ReplaceMode,
    pub sampling: Sampling,
    pub paint_tolerance: f32,
    pub protect_fg: bool,
    /// Color Replacement / Background Eraser "Limits".
    pub limits: paint::Limits,
    pub auto_erase: bool,
    pub grad: Gradient,
    pub grad_kind: GradKind,
    pub grad_reverse: bool,
    pub grad_dither: bool,
    pub grad_editor: bool,
    pub grad_stop: usize,
    pub bucket_pattern: bool,
}

impl Default for PaintOptions {
    fn default() -> Self {
        PaintOptions {
            range: Range::Midtones,
            exposure: 0.5,
            protect_tones: true,
            sponge_saturate: false,
            vibrance: true,
            focus_strength: 0.5,
            smudge_strength: 0.5,
            finger: false,
            wet: 0.5,
            load: 0.5,
            mix: 0.5,
            mixer_flow: 1.0,
            mixer_color: None,
            load_after: true,
            clean_after: false,
            replace_mode: ReplaceMode::Color,
            sampling: Sampling::Continuous,
            paint_tolerance: 30.0,
            protect_fg: false,
            limits: paint::Limits::Contiguous,
            auto_erase: false,
            grad: Gradient::two("Foreground to Background", [0, 0, 0], [255, 255, 255]),
            grad_kind: GradKind::Linear,
            grad_reverse: false,
            grad_dither: true,
            grad_editor: false,
            grad_stop: 0,
            bucket_pattern: false,
        }
    }
}

/// A placed gradient still being adjusted with its handles.
pub struct GradLive {
    pub layer_id: u64,
    pub p0: (f32, f32),
    pub p1: (f32, f32),
}

impl Editor {
    fn kernel_for(&self, tool: Tool) -> Option<Box<dyn Kernel>> {
        let po = &self.popts;
        let li = self.doc.active;
        Some(match tool {
            Tool::Dodge => Box::new(paint::ToneKernel::new(Tone::Dodge, po.range, po.exposure, po.protect_tones, false)),
            Tool::Burn => Box::new(paint::ToneKernel::new(Tone::Burn, po.range, po.exposure, po.protect_tones, false)),
            Tool::Sponge => Box::new(paint::ToneKernel::new(if po.sponge_saturate { Tone::Saturate } else { Tone::Desaturate }, po.range, po.exposure, false, po.vibrance)),
            Tool::Blur => Box::new(paint::FocusKernel::new(&self.doc.layers[li].grid, po.focus_strength, false)),
            Tool::Sharpen => Box::new(paint::FocusKernel::new(&self.doc.layers[li].grid, po.focus_strength, true)),
            Tool::Smudge => {
                let f = po.finger.then(|| pxm::pack(self.fg[0], self.fg[1], self.fg[2], 255));
                Box::new(paint::SmudgeKernel::new(po.smudge_strength, f))
            }
            Tool::Mixer => {
                let c = po.mixer_color.map(|c| pxm::pack(c[0], c[1], c[2], 255));
                Box::new(paint::MixerKernel::new(c, po.wet, po.load, po.mix, po.mixer_flow))
            }
            Tool::ColorReplace => Box::new(paint::ReplaceKernel::new(self.fg, self.bg, po.replace_mode, po.sampling, po.paint_tolerance).with_limits(po.limits)),
            Tool::BackgroundEraser => Box::new(paint::BgEraseKernel::new(po.sampling, po.paint_tolerance, self.bg, po.protect_fg.then_some(self.fg)).with_limits(po.limits)),
            _ => return None,
        })
    }

    /// Mouse-down for the painting tools. Returns true when handled.
    pub fn paint_down(&mut self, ix: f32, iy: f32, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let tool = self.tool;
        let li = self.doc.active;
        match tool {
            Tool::Pencil => {
                // Auto Erase: starting on the foreground colour paints the background.
                let mut color = self.fg;
                if self.popts.auto_erase {
                    let p = pxm::unpremultiply(self.doc.layers[li].grid.pixel(ix as i32, iy as i32));
                    if p[3] > 0 && p[..3] == self.fg {
                        color = self.bg;
                    }
                }
                if !self.ensure_pixels(li) {
                    return true;
                }
                let k = Box::new(crate::engine::brush::Paint { color: pxm::pack(color[0], color[1], color[2], 255) });
                self.start_kernel_stroke("Pencil", k, ix, iy);
            }
            Tool::Mixer if ev.modifiers.alt => {
                // Alt-click loads the reservoir with the clicked colour.
                let p = pxm::unpremultiply(composite::sample(&self.doc.layers, ix as i32, iy as i32));
                self.popts.mixer_color = Some([p[0], p[1], p[2]]);
                self.flash("Mixer brush loaded from the canvas");
            }
            Tool::ColorReplace | Tool::Mixer | Tool::Blur | Tool::Sharpen | Tool::Smudge | Tool::Dodge | Tool::Burn | Tool::Sponge | Tool::BackgroundEraser => {
                if !self.ensure_pixels(li) {
                    return true;
                }
                if tool == Tool::BackgroundEraser {
                    self.background_to_layer(li);
                }
                if tool == Tool::Mixer && self.popts.load_after && self.popts.mixer_color.is_none() {
                    self.popts.mixer_color = Some(self.fg);
                }
                let k = self.kernel_for(tool).unwrap();
                self.start_kernel_stroke(tool.name(), k, ix, iy);
            }
            Tool::MagicEraser => {
                if !self.ensure_pixels(li) {
                    return true;
                }
                // Erasing to transparency on the Background unlocks it ("Layer 0").
                self.background_to_layer(li);
                let t0 = Instant::now();
                let (xi, yi) = (ix as i32, iy as i32);
                let mut mask = if self.opts.sample_all {
                    let src = ops::CompositeSource::new(&self.doc.layers, self.doc.w, self.doc.h);
                    ops::flood_mask_on(&src, xi, yi, self.tolerance as u8, self.contiguous)
                } else {
                    ops::flood_mask(&self.doc.layers[li].grid, xi, yi, self.tolerance as u8, self.contiguous)
                };
                if self.opts.anti_alias {
                    mask = mask.feather(0.55);
                }
                let g = paint::erase_mask(&self.doc.layers[li].grid, &mask, self.doc.selection.as_ref(), self.fill_opacity);
                self.doc.push_undo("Magic Eraser");
                self.doc.set_layer_grid(li, g);
                self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
                self.flash(format!("Magic Eraser — {:.1} ms", self.last_tool_ms));
            }
            Tool::Bucket => {
                let (xi, yi) = (ix.floor() as i32, iy.floor() as i32);
                if xi < 0 || yi < 0 || xi >= self.doc.w as i32 || yi >= self.doc.h as i32 || !self.ensure_pixels(li) {
                    return true;
                }
                let (tol, contiguous, rgb, op) = (self.tolerance as u8, self.contiguous, self.fg, self.fill_opacity);
                let pattern = self.popts.bucket_pattern.then_some(self.ropts.pattern);
                let all = self.opts.sample_all.then(|| self.doc.layers.clone());
                let (w, h) = (self.doc.w, self.doc.h);
                self.run_layer_op("Paint Bucket", cx, move |g, sel, _| {
                    let mask = match &all {
                        Some(layers) => {
                            let src = ops::CompositeSource::new(layers, w, h);
                            ops::flood_mask_on(&src, xi, yi, tol, contiguous)
                        }
                        None => ops::flood_mask(g, xi, yi, tol, contiguous),
                    };
                    match pattern {
                        Some(p) => paint::fill_mask_pattern(g, &mask, sel, p, op),
                        None => ops::fill_mask(g, &mask, sel, rgb, op),
                    }
                });
            }
            Tool::Gradient => {
                // Grab an endpoint handle of the live gradient, or start a new one.
                let s = self.vp.logical_scale(self.sf);
                if let Some(gl) = &self.grad_live {
                    for (which, p) in [(0u8, gl.p0), (1, gl.p1)] {
                        if ((p.0 - ix).powi(2) + (p.1 - iy).powi(2)).sqrt() * s < 9.0 {
                            self.drag = Drag::GradHandle(which);
                            return true;
                        }
                    }
                }
                self.commit_gradient(cx);
                if !self.ensure_pixels(li) {
                    return true;
                }
                self.drag = Drag::Gradient { start: (ix, iy), cur: (ix, iy) };
            }
            _ => return false,
        }
        let _ = window;
        true
    }

    pub fn grad_fill(&self, p0: (f32, f32), p1: (f32, f32)) -> GradientFill {
        GradientFill::new(self.popts.grad.clone(), self.popts.grad_kind, p0, p1, self.popts.grad_reverse, self.popts.grad_dither, self.fill_opacity)
    }

    /// Shows (or updates) the live gradient preview on the active layer.
    pub fn set_live_gradient(&mut self, p0: (f32, f32), p1: (f32, f32)) {
        let li = self.doc.active;
        let f = Filter::Gradient(Arc::new(self.grad_fill(p0, p1)));
        self.doc.preview = Some(Preview::new(li, f, self.doc.selection.clone()));
        self.grad_live = Some(GradLive { layer_id: self.doc.layers[li].id, p0, p1 });
        self.doc.mark_all_dirty();
    }

    /// Applies the live gradient at full resolution (one undo step).
    pub fn commit_gradient(&mut self, cx: &mut Context<Self>) {
        let Some(gl) = self.grad_live.take() else { return };
        let Some(li) = self.doc.layers.iter().position(|l| l.id == gl.layer_id) else { return };
        let f = Filter::Gradient(Arc::new(self.grad_fill(gl.p0, gl.p1)));
        self.doc.active = li;
        self.run_layer_op("Gradient", cx, move |g, sel, p| ops::apply_filter(g, &f, sel, p));
    }

    pub fn cancel_gradient(&mut self) {
        if self.grad_live.take().is_some() {
            self.doc.preview = None;
            self.doc.mark_all_dirty();
        }
    }

    /// Refreshes the live preview after an option changed.
    pub fn refresh_live_gradient(&mut self) {
        if let Some(gl) = &self.grad_live {
            let (p0, p1) = (gl.p0, gl.p1);
            self.set_live_gradient(p0, p1);
        }
    }

    /// Handles and stop markers of the live gradient.
    pub fn gradient_overlays(&self, out: &mut Vec<Overlay>) {
        let (p0, p1) = match (&self.drag, &self.grad_live) {
            (Drag::Gradient { start, cur }, _) => (*start, *cur),
            (_, Some(g)) => (g.p0, g.p1),
            _ => return,
        };
        let a = self.vp.to_screen(p0.0, p0.1, self.canvas, self.sf);
        let b = self.vp.to_screen(p1.0, p1.1, self.canvas, self.sf);
        out.push(Overlay::Path { polys: vec![(vec![a, b], false)], dashed: false, color: gpui_kit::hsla(0., 0., 0., 0.8), width: 3.0 });
        out.push(Overlay::Path { polys: vec![(vec![a, b], false)], dashed: false, color: gpui_kit::hsla(0., 0., 1., 1.), width: 1.0 });
        let white = gpui_kit::hsla(0., 0., 1., 1.);
        let black = gpui_kit::hsla(0., 0., 0., 1.);
        for (c, r) in [(a, 7.0), (b, 7.0)] {
            out.push(Overlay::Quad { bounds: Bounds::from_corners(point(c.x - px(r), c.y - px(r)), point(c.x + px(r), c.y + px(r))), bg: white, border: black, border_w: 1.5, radius: r });
        }
        // Colour stops along the line.
        for st in &self.popts.grad.stops {
            let t = if self.popts.grad_reverse { 1.0 - st.pos } else { st.pos };
            let c = point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
            let col = super::editor::rgb_to_hsla(st.color);
            out.push(Overlay::Quad { bounds: Bounds::from_corners(point(c.x - px(5.), c.y - px(5.)), point(c.x + px(5.), c.y + px(5.))), bg: col, border: white, border_w: 1.5, radius: 1.0 });
        }
    }

    // ----- options bar ----------------------------------------------------------------

    pub fn paint_options(&mut self, mut bar: Div, window: &mut Window, cx: &mut Context<Self>) -> Div {
        match self.tool {
            Tool::Pencil => {
                bar = bar.child(toggle("autoerase", "Auto Erase", self.popts.auto_erase).on_click(cx.listener(|e, _, _, cx| {
                    e.popts.auto_erase = !e.popts.auto_erase;
                    cx.notify();
                })));
            }
            Tool::Dodge | Tool::Burn => {
                bar = bar
                    .child(label("Range:"))
                    .child(segmented("range", &Range::ALL.map(|r| (r, r.label())), self.popts.range, cx, |e, v| e.popts.range = v))
                    .child(self.opt_slider("Exposure", "exposure", spec(1.0, 100.0, 1.0, |e| e.popts.exposure * 100.0, |e, v| e.popts.exposure = v / 100.0), |v| format!("{v:.0}%"), 80., window, cx))
                    .child(toggle("protect", "Protect Tones", self.popts.protect_tones).on_click(cx.listener(|e, _, _, cx| {
                        e.popts.protect_tones = !e.popts.protect_tones;
                        cx.notify();
                    })));
            }
            Tool::Sponge => {
                bar = bar
                    .child(segmented("sponge", &[(false, "Desaturate"), (true, "Saturate")], self.popts.sponge_saturate, cx, |e, v| e.popts.sponge_saturate = v))
                    .child(self.opt_slider("Flow", "sponge.flow", spec(1.0, 100.0, 1.0, |e| e.popts.exposure * 100.0, |e, v| e.popts.exposure = v / 100.0), |v| format!("{v:.0}%"), 80., window, cx))
                    .child(toggle("vibrance", "Vibrance", self.popts.vibrance).on_click(cx.listener(|e, _, _, cx| {
                        e.popts.vibrance = !e.popts.vibrance;
                        cx.notify();
                    })));
            }
            Tool::Blur | Tool::Sharpen => {
                bar = bar.child(self.opt_slider("Strength", "focus", spec(1.0, 100.0, 1.0, |e| e.popts.focus_strength * 100.0, |e, v| e.popts.focus_strength = v / 100.0), |v| format!("{v:.0}%"), 90., window, cx));
            }
            Tool::Smudge => {
                bar = bar
                    .child(self.opt_slider("Strength", "smudge", spec(1.0, 100.0, 1.0, |e| e.popts.smudge_strength * 100.0, |e, v| e.popts.smudge_strength = v / 100.0), |v| format!("{v:.0}%"), 90., window, cx))
                    .child(toggle("finger", "Finger Painting", self.popts.finger).on_click(cx.listener(|e, _, _, cx| {
                        e.popts.finger = !e.popts.finger;
                        cx.notify();
                    })));
            }
            Tool::Mixer => {
                let swatch = self.popts.mixer_color.map_or("Clean".to_string(), |c| format!("#{:02X}{:02X}{:02X}", c[0], c[1], c[2]));
                bar = bar
                    .child(label(format!("Reservoir: {swatch}")))
                    .child(Button::new("mix-load").label("Load").xsmall().on_click(cx.listener(|e, _, _, cx| {
                        e.popts.mixer_color = Some(e.fg);
                        cx.notify();
                    })))
                    .child(Button::new("mix-clean").label("Clean").xsmall().on_click(cx.listener(|e, _, _, cx| {
                        e.popts.mixer_color = None;
                        cx.notify();
                    })))
                    .child(self.opt_slider("Wet", "mix.wet", spec(0.0, 100.0, 1.0, |e| e.popts.wet * 100.0, |e, v| e.popts.wet = v / 100.0), |v| format!("{v:.0}%"), 60., window, cx))
                    .child(self.opt_slider("Load", "mix.load", spec(1.0, 100.0, 1.0, |e| e.popts.load * 100.0, |e, v| e.popts.load = v / 100.0), |v| format!("{v:.0}%"), 60., window, cx))
                    .child(self.opt_slider("Mix", "mix.mix", spec(0.0, 100.0, 1.0, |e| e.popts.mix * 100.0, |e, v| e.popts.mix = v / 100.0), |v| format!("{v:.0}%"), 60., window, cx))
                    .child(toggle("mix-la", "Load after stroke", self.popts.load_after).on_click(cx.listener(|e, _, _, cx| {
                        e.popts.load_after = !e.popts.load_after;
                        cx.notify();
                    })))
                    .child(toggle("mix-ca", "Clean after stroke", self.popts.clean_after).on_click(cx.listener(|e, _, _, cx| {
                        e.popts.clean_after = !e.popts.clean_after;
                        cx.notify();
                    })));
            }
            Tool::ColorReplace | Tool::BackgroundEraser => {
                if self.tool == Tool::ColorReplace {
                    bar = bar.child(segmented("rmode", &ReplaceMode::ALL.map(|m| (m, m.label())), self.popts.replace_mode, cx, |e, v| e.popts.replace_mode = v));
                }
                bar = bar
                    .child(label("Sampling:"))
                    .child(segmented("sampling", &Sampling::ALL.map(|m| (m, m.label())), self.popts.sampling, cx, |e, v| e.popts.sampling = v))
                    .child(label("Limits:"))
                    .child(segmented("limits", &paint::Limits::ALL.map(|m| (m, m.label())), self.popts.limits, cx, |e, v| e.popts.limits = v))
                    .child(self.opt_slider("Tolerance", "ptol", spec(1.0, 255.0, 1.0, |e| e.popts.paint_tolerance, |e, v| e.popts.paint_tolerance = v), |v| format!("{v:.0}"), 80., window, cx));
                if self.tool == Tool::BackgroundEraser {
                    bar = bar.child(toggle("pfg", "Protect Foreground", self.popts.protect_fg).on_click(cx.listener(|e, _, _, cx| {
                        e.popts.protect_fg = !e.popts.protect_fg;
                        cx.notify();
                    })));
                }
            }
            Tool::MagicEraser => {
                bar = bar.child(toggle("me-aa", "Anti-alias", self.opts.anti_alias).on_click(cx.listener(|e, _, _, cx| {
                    e.opts.anti_alias = !e.opts.anti_alias;
                    cx.notify();
                })));
            }
            Tool::Bucket => {
                bar = bar.child(segmented("bsrc", &[(false, "Foreground"), (true, "Pattern")], self.popts.bucket_pattern, cx, |e, v| e.popts.bucket_pattern = v));
                if self.popts.bucket_pattern {
                    let pats: Vec<(usize, &'static str)> = crate::engine::retouch::patterns().iter().enumerate().map(|(i, p)| (i, p.name)).collect();
                    bar = bar.child(segmented("bpat", &pats, self.ropts.pattern, cx, |e, v| e.ropts.pattern = v));
                }
            }
            Tool::Gradient => {
                bar = self.gradient_options(bar, window, cx);
            }
            _ => {}
        }
        bar
    }

    fn gradient_options(&mut self, mut bar: Div, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let presets = Gradient::presets(self.fg, self.bg);
        let mut pr = div().flex().gap_0p5();
        for (i, g) in presets.into_iter().enumerate() {
            let name = g.name.clone();
            pr = pr.child(Button::new(SharedString::from(format!("gpre-{i}"))).child(gradient_strip(&g, 54.0, 14.0)).xsmall().ghost().tooltip(name).on_click(cx.listener(move |e, _, _, cx| {
                e.popts.grad = g.clone();
                e.popts.grad_stop = 0;
                e.refresh_live_gradient();
                cx.notify();
            })));
        }
        bar = bar
            .child(pr)
            .child(segmented("gkind", &GradKind::ALL.map(|k| (k, k.label())), self.popts.grad_kind, cx, |e, v| {
                e.popts.grad_kind = v;
                e.refresh_live_gradient();
            }))
            .child(toggle("grev", "Reverse", self.popts.grad_reverse).on_click(cx.listener(|e, _, _, cx| {
                e.popts.grad_reverse = !e.popts.grad_reverse;
                e.refresh_live_gradient();
                cx.notify();
            })))
            .child(toggle("gdith", "Dither", self.popts.grad_dither).on_click(cx.listener(|e, _, _, cx| {
                e.popts.grad_dither = !e.popts.grad_dither;
                e.refresh_live_gradient();
                cx.notify();
            })))
            .child(self.opt_slider("Opacity", "fillop", spec(1.0, 100.0, 1.0, |e| e.fill_opacity * 100.0, |e, v| {
                e.fill_opacity = v / 100.0;
                e.refresh_live_gradient();
            }), |v| format!("{v:.0}%"), 70., window, cx))
            .child(toggle("gedit", "Edit Stops ▾", self.popts.grad_editor).on_click(cx.listener(|e, _, _, cx| {
                e.popts.grad_editor = !e.popts.grad_editor;
                cx.notify();
            })));
        if self.grad_live.is_some() {
            bar = bar
                .child(Button::new("gcommit").label("Apply (Enter)").xsmall().primary().on_click(cx.listener(|e, _, _, cx| e.commit_gradient(cx))))
                .child(Button::new("gcancel").label("Cancel (Esc)").xsmall().on_click(cx.listener(|e, _, _, cx| {
                    e.cancel_gradient();
                    cx.notify();
                })));
        }
        if self.popts.grad_editor {
            bar = bar.child(self.gradient_editor(window, cx));
        }
        bar
    }

    /// Stops editor: the ramp with clickable stops, plus stop controls.
    fn gradient_editor(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let g = self.popts.grad.clone();
        let sel = self.popts.grad_stop.min(g.stops.len().saturating_sub(1));
        let mut markers = div().relative().w(px(300.)).h(px(14.));
        for (i, st) in g.stops.iter().enumerate() {
            let x = st.pos * 292.0;
            markers = markers.child(
                div()
                    .id(SharedString::from(format!("gstop-{i}")))
                    .absolute()
                    .left(px(x))
                    .top(px(0.))
                    .size(px(10.))
                    .border_1()
                    .border_color(rgb(if i == sel { 0xffffff } else { 0x888888 }))
                    .bg(super::editor::rgb_to_hsla(st.color))
                    .on_click(cx.listener(move |e, _, _, cx| {
                        e.popts.grad_stop = i;
                        cx.notify();
                    })),
            );
        }
        div()
            .flex()
            .items_center()
            .gap_2()
            .child(div().flex().flex_col().gap_0p5().child(gradient_strip(&g, 300.0, 16.0)).child(markers))
            .child(Button::new("gs-fg").label("Stop = FG").xsmall().on_click(cx.listener(|e, _, _, cx| {
                let i = e.popts.grad_stop.min(e.popts.grad.stops.len() - 1);
                e.popts.grad.stops[i].color = e.fg;
                e.refresh_live_gradient();
                cx.notify();
            })))
            .child(Button::new("gs-add").label("Add Stop").xsmall().on_click(cx.listener(|e, _, _, cx| {
                let g = &mut e.popts.grad;
                let i = e.popts.grad_stop.min(g.stops.len() - 1);
                let j = (i + 1).min(g.stops.len() - 1);
                let pos = if i == j { (g.stops[i].pos + 1.0) * 0.5 } else { (g.stops[i].pos + g.stops[j].pos) * 0.5 };
                g.stops.push(ColorStop { pos, color: e.fg, mid: 0.5 });
                g.sort();
                e.popts.grad_stop = g.stops.iter().position(|s| s.pos == pos).unwrap_or(0);
                e.refresh_live_gradient();
                cx.notify();
            })))
            .child(Button::new("gs-del").label("Delete").xsmall().on_click(cx.listener(|e, _, _, cx| {
                if e.popts.grad.stops.len() > 2 {
                    let i = e.popts.grad_stop.min(e.popts.grad.stops.len() - 1);
                    e.popts.grad.stops.remove(i);
                    e.popts.grad_stop = i.saturating_sub(1);
                    e.refresh_live_gradient();
                }
                cx.notify();
            })))
            .child(self.opt_slider("Location", "gs.pos", spec(0.0, 100.0, 1.0, |e| {
                let i = e.popts.grad_stop.min(e.popts.grad.stops.len() - 1);
                e.popts.grad.stops[i].pos * 100.0
            }, |e, v| {
                let i = e.popts.grad_stop.min(e.popts.grad.stops.len() - 1);
                e.popts.grad.stops[i].pos = v / 100.0;
                let c = e.popts.grad.stops[i];
                e.popts.grad.sort();
                e.popts.grad_stop = e.popts.grad.stops.iter().position(|s| *s == c).unwrap_or(0);
                e.refresh_live_gradient();
            }), |v| format!("{v:.0}%"), 70., window, cx))
            .child(self.opt_slider("End opacity", "gs.alpha", spec(0.0, 100.0, 1.0, |e| e.popts.grad.alpha.last().map_or(100.0, |a| a.alpha * 100.0), |e, v| {
                let g = &mut e.popts.grad;
                if g.alpha.len() < 2 {
                    g.alpha = vec![AlphaStop { pos: 0.0, alpha: 1.0 }, AlphaStop { pos: 1.0, alpha: 1.0 }];
                }
                let n = g.alpha.len();
                g.alpha[n - 1].alpha = v / 100.0;
                e.refresh_live_gradient();
            }), |v| format!("{v:.0}%"), 70., window, cx))
    }

    /// Mixer: reload / clean after each stroke.
    pub fn after_paint_stroke(&mut self) {
        if self.tool == Tool::Mixer {
            if self.popts.clean_after {
                self.popts.mixer_color = None;
            }
            if self.popts.load_after {
                self.popts.mixer_color = Some(self.fg);
            }
        }
    }
}

/// A horizontal preview of a gradient as a row of coloured cells.
pub fn gradient_strip(g: &Gradient, w: f32, h: f32) -> Div {
    let fill = GradientFill::new(g.clone(), GradKind::Linear, (0.0, 0.0), (64.0, 0.0), false, false, 1.0);
    let mut row = div().flex().w(px(w)).h(px(h)).border_1().border_color(rgb(0x555555));
    for k in 0..32 {
        let c = fill.pixel(k as f32 * 2.0 + 1.0, 0.0, 0, 0);
        let check = if k % 2 == 0 { 0xFFFF_FFFF } else { 0xFFCC_CCCC };
        let [r, gg, b, _] = pxm::unpack(pxm::over(c, check));
        row = row.child(div().flex_1().h_full().bg(gpui_kit::Rgba { r: r as f32 / 255.0, g: gg as f32 / 255.0, b: b as f32 / 255.0, a: 1.0 }));
    }
    row
}
