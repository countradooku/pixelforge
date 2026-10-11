//! Tool options (the data) and the options-bar widgets.
//!
//! Sliders are created on demand and kept in a registry keyed by name; each
//! is bound two-way to a field through a getter/setter pair, so adding an
//! option is one line and keyboard shortcuts that change a value (e.g. `[`)
//! show up in the slider on the next frame.

use std::collections::HashMap;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::slider::{Slider, SliderEvent, SliderScale, SliderState};
use gpui_kit::component::{Selectable as _, Sizable as _};
use gpui_kit::prelude::*;
use gpui_kit::{Context, Div, Entity, SharedString, Subscription, Window, div, px, rgb};

use crate::engine::masks::SelOp;

use super::editor::Editor;

pub const DIM: u32 = 0x8e8e93;
pub const TEXT: u32 = 0xdcdcdc;

pub struct Knob {
    pub state: Entity<SliderState>,
    _sub: Subscription,
}

pub type Getter = fn(&Editor) -> f32;
pub type Setter = fn(&mut Editor, f32);

#[derive(Clone, Copy)]
pub struct KnobSpec {
    pub min: f32,
    pub max: f32,
    pub step: f32,
    pub log: bool,
    pub get: Getter,
    pub set: Setter,
}

pub const fn spec(min: f32, max: f32, step: f32, get: Getter, set: Setter) -> KnobSpec {
    KnobSpec { min, max, step, log: false, get, set }
}

pub const fn spec_log(min: f32, max: f32, step: f32, get: Getter, set: Setter) -> KnobSpec {
    KnobSpec { min, max, step, log: true, get, set }
}

/// Which sample the selection tools combine with (options bar).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SampleSize {
    #[default]
    Point,
    Avg3,
    Avg5,
    Avg11,
    Avg31,
    Avg51,
    Avg101,
}

impl SampleSize {
    pub const ALL: [SampleSize; 7] =
        [SampleSize::Point, SampleSize::Avg3, SampleSize::Avg5, SampleSize::Avg11, SampleSize::Avg31, SampleSize::Avg51, SampleSize::Avg101];
    pub fn side(self) -> u32 {
        match self {
            SampleSize::Point => 1,
            SampleSize::Avg3 => 3,
            SampleSize::Avg5 => 5,
            SampleSize::Avg11 => 11,
            SampleSize::Avg31 => 31,
            SampleSize::Avg51 => 51,
            SampleSize::Avg101 => 101,
        }
    }
}

/// Options for every tool, shown in the options bar.
pub struct ToolOptions {
    pub sel_op: SelOp,
    pub feather: f32,
    pub anti_alias: bool,
    pub sample_all: bool,
    pub sample_size: SampleSize,
    pub auto_select: bool,
    /// Auto-Select picks the layer's outermost group instead of the layer.
    pub auto_select_group: bool,
    pub show_transform: bool,
    pub magnetic_width: f32,
    pub magnetic_contrast: f32,
    pub magnetic_frequency: f32,
    pub object_lasso: bool,
    /// Object Selection: highlight and click-select objects under the cursor.
    pub object_finder: bool,
}

impl Default for ToolOptions {
    fn default() -> Self {
        ToolOptions {
            sel_op: SelOp::New,
            feather: 0.0,
            anti_alias: true,
            sample_all: false,
            sample_size: SampleSize::Point,
            auto_select: false,
            auto_select_group: false,
            show_transform: false,
            magnetic_width: 10.0,
            magnetic_contrast: 10.0,
            magnetic_frequency: 57.0,
            object_lasso: false,
            object_finder: true,
        }
    }
}

pub fn label(t: impl Into<SharedString>) -> Div {
    div().text_color(rgb(DIM)).child(t.into())
}

impl Editor {
    /// The slider bound to `key`, created on first use and re-synced to the
    /// model value every frame.
    pub fn knob(&mut self, key: &'static str, s: KnobSpec, window: &mut Window, cx: &mut Context<Self>) -> Entity<SliderState> {
        let cur = (s.get)(self);
        if let Some(k) = self.knobs.get(key) {
            let st = k.state.clone();
            let v = st.read(cx).value().start();
            if (v - cur).abs() > s.step * 0.25 {
                st.update(cx, |st, cx| st.set_value(cur, window, cx));
            }
            return st;
        }
        let state = cx.new(|_| {
            SliderState::new()
                .min(s.min)
                .max(s.max)
                .step(s.step)
                .scale(if s.log { SliderScale::Logarithmic } else { SliderScale::Linear })
                .default_value(cur.clamp(s.min, s.max))
        });
        let set = s.set;
        let sub = cx.subscribe(&state, move |this: &mut Editor, _, ev: &SliderEvent, cx| {
            let v = match ev {
                SliderEvent::Change(v) | SliderEvent::Release(v) => v.start(),
            };
            set(this, v);
            cx.notify();
        });
        self.knobs.insert(key, Knob { state: state.clone(), _sub: sub });
        state
    }

    /// `Label [slider] value` for the options bar.
    #[allow(clippy::too_many_arguments)]
    pub fn opt_slider(
        &mut self,
        name: &'static str,
        key: &'static str,
        s: KnobSpec,
        fmt: impl Fn(f32) -> String,
        width: f32,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Div {
        let st = self.knob(key, s, window, cx);
        let v = (s.get)(self);
        div()
            .flex()
            .items_center()
            .gap_1p5()
            .child(label(name))
            .child(div().w(px(width)).child(Slider::new(&st)))
            .child(div().min_w(px(44.)).text_color(rgb(TEXT)).child(fmt(v)))
    }
}

/// A small toggle button.
pub fn toggle(id: impl Into<SharedString>, text: impl Into<SharedString>, on: bool) -> Button {
    // Filled when on (like the segmented controls), so the state reads at a glance.
    let b = Button::new(id.into()).label(text.into()).xsmall().selected(on);
    if on { b.primary() } else { b.ghost() }
}

/// Segmented choice: one button per option; `f` receives the chosen index.
pub fn segmented<T: Copy + PartialEq + 'static>(
    id: &'static str,
    items: &[(T, &'static str)],
    cur: T,
    cx: &mut Context<Editor>,
    f: fn(&mut Editor, T),
) -> Div {
    let mut row = div().flex().flex_wrap().gap_0p5();
    for (i, &(v, name)) in items.iter().enumerate() {
        let b = Button::new(SharedString::from(format!("{id}-{i}"))).label(name).xsmall();
        let b = if v == cur { b.primary() } else { b.ghost() };
        row = row.child(b.on_click(cx.listener(move |this, _, _, cx| {
            f(this, v);
            cx.notify();
        })));
    }
    row
}

/// Registry type stored on the editor.
pub type Knobs = HashMap<&'static str, Knob>;

// ----- the options bar ---------------------------------------------------------------

use super::editor::Tool;
use gpui_kit::component::Icon;

const PANEL_HI: u32 = 0x2f2f32;
const BORDER: u32 = 0x3a3a3d;

fn pct(v: f32) -> String {
    format!("{:.0}%", v * 100.0)
}

impl Editor {
    pub fn render_options(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        let tool = self.tool;
        let mut bar = div()
            .min_h(px(42.))
            .flex()
            .flex_wrap()
            .items_center()
            .gap_x_4()
            .gap_y_1()
            .px_3()
            .py_1()
            .bg(rgb(PANEL_HI))
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .min_w(px(170.))
                    .child(Icon::new(tool.icon()).text_color(rgb(TEXT)))
                    .child(div().text_color(rgb(TEXT)).child(tool.name())),
            );
        if tool.is_selection() {
            bar = bar.child(segmented("selop", &SelOp::ALL.map(|o| (o, o.label())), self.opts.sel_op, cx, |e, v| e.opts.sel_op = v));
            if !matches!(tool, Tool::MagicWand | Tool::QuickSelect | Tool::ObjectSelect) {
                bar = bar.child(self.opt_slider("Feather", "feather", spec(0.0, 250.0, 0.5, |e| e.opts.feather, |e, v| e.opts.feather = v), |v| format!("{v:.1} px"), 90., window, cx));
            }
            if !matches!(tool, Tool::RectSelect | Tool::RowSelect | Tool::ColumnSelect) {
                bar = bar.child(toggle("aa", "Anti-alias", self.opts.anti_alias).on_click(cx.listener(|e, _, _, cx| {
                    e.opts.anti_alias = !e.opts.anti_alias;
                    cx.notify();
                })));
            }
        }
        if tool.is_brush() && tool != Tool::Liquify {
            bar = self.brush_options(bar, window, cx);
        }
        bar = self.tool_options(bar, window, cx);
        bar = self.paint_options(bar, window, cx);
        bar = self.vector_options(bar, window, cx);
        bar = self.nav_options(bar, window, cx);
        bar = self.type_options(bar, window, cx);
        bar.child(div().flex_1()).child(div().text_size(px(11.)).text_color(rgb(DIM)).child(tool.help()))
    }

    /// Size / hardness / opacity / flow / spacing and the tip toggle.
    fn brush_options(&mut self, mut bar: Div, window: &mut Window, cx: &mut Context<Self>) -> Div {
        bar = bar
            .child(self.opt_slider("Size", "b.size", spec_log(1.0, 2500.0, 1.0, |e| e.brush_ref().size, |e, v| e.brush_mut().size = v), |v| format!("{v:.0} px"), 120., window, cx))
            .child(self.opt_slider("Hardness", "b.hard", spec(0.0, 100.0, 1.0, |e| e.brush_ref().hardness * 100.0, |e, v| e.brush_mut().hardness = v / 100.0), |v| format!("{v:.0}%"), 70., window, cx));
        if !matches!(self.tool, Tool::QuickSelect | Tool::SpotHeal) {
            bar = bar
                .child(self.opt_slider("Opacity", "b.op", spec(1.0, 100.0, 1.0, |e| e.brush_ref().opacity * 100.0, |e, v| e.brush_mut().opacity = v / 100.0), |v| format!("{v:.0}%"), 70., window, cx))
                .child(self.opt_slider("Flow", "b.flow", spec(1.0, 100.0, 1.0, |e| e.brush_ref().flow * 100.0, |e, v| e.brush_mut().flow = v / 100.0), |v| format!("{v:.0}%"), 70., window, cx));
        }
        bar = bar.child(toggle("tip", "Tip ▾", self.show_tip).on_click(cx.listener(|e, _, _, cx| {
            e.show_tip = !e.show_tip;
            cx.notify();
        })));
        if self.show_tip {
            let b = *self.brush_ref();
            bar = bar
                .child(self.opt_slider("Spacing", "b.sp", spec(1.0, 200.0, 1.0, |e| e.brush_ref().spacing * 100.0, |e, v| e.brush_mut().spacing = v / 100.0), |v| format!("{v:.0}%"), 70., window, cx))
                .child(self.opt_slider("Angle", "b.ang", spec(-180.0, 180.0, 1.0, |e| e.brush_ref().angle, |e, v| e.brush_mut().angle = v), |v| format!("{v:.0}°"), 70., window, cx))
                .child(self.opt_slider("Roundness", "b.rnd", spec(1.0, 100.0, 1.0, |e| e.brush_ref().roundness * 100.0, |e, v| e.brush_mut().roundness = v / 100.0), |v| format!("{v:.0}%"), 60., window, cx))
                .child(self.opt_slider("Scatter", "b.sc", spec(0.0, 100.0, 1.0, |e| e.brush_ref().scatter * 100.0, |e, v| e.brush_mut().scatter = v / 100.0), |v| format!("{v:.0}%"), 60., window, cx))
                .child(self.opt_slider("Smoothing", "b.sm", spec(0.0, 100.0, 1.0, |e| e.brush_ref().smoothing * 100.0, |e, v| e.brush_mut().smoothing = v / 100.0), |v| format!("{v:.0}%"), 60., window, cx))
                .child(segmented(
                    "tex",
                    &[(0u8, crate::engine::brush::TEXTURES[0]), (1, crate::engine::brush::TEXTURES[1]), (2, crate::engine::brush::TEXTURES[2]), (3, crate::engine::brush::TEXTURES[3])],
                    b.texture,
                    cx,
                    |e, v| e.brush_mut().texture = v,
                ));
            if b.texture > 0 {
                bar = bar.child(self.opt_slider("Depth", "b.td", spec(0.0, 100.0, 1.0, |e| e.brush_ref().texture_depth * 100.0, |e, v| e.brush_mut().texture_depth = v / 100.0), |v| format!("{v:.0}%"), 60., window, cx));
            }
            bar = bar
                .child(label("Pressure:"))
                .child(toggle("p-size", "Size", b.pressure_size).on_click(cx.listener(|e, _, _, cx| {
                    let b = e.brush_mut();
                    b.pressure_size = !b.pressure_size;
                    cx.notify();
                })))
                .child(toggle("p-op", "Opacity", b.pressure_opacity).on_click(cx.listener(|e, _, _, cx| {
                    let b = e.brush_mut();
                    b.pressure_opacity = !b.pressure_opacity;
                    cx.notify();
                })))
                .child(toggle("p-flow", "Flow", b.pressure_flow).on_click(cx.listener(|e, _, _, cx| {
                    let b = e.brush_mut();
                    b.pressure_flow = !b.pressure_flow;
                    cx.notify();
                })));
        }
        bar
    }

    /// Options specific to one tool.
    fn tool_options(&mut self, mut bar: Div, window: &mut Window, cx: &mut Context<Self>) -> Div {
        match self.tool {
            Tool::MagicWand | Tool::Bucket | Tool::MagicEraser => {
                bar = bar
                    .child(self.opt_slider("Tolerance", "tol", spec(0.0, 255.0, 1.0, |e| e.tolerance, |e, v| e.tolerance = v), |v| format!("{v:.0}"), 100., window, cx))
                    .child(toggle("contig", "Contiguous", self.contiguous).on_click(cx.listener(|e, _, _, cx| {
                        e.contiguous = !e.contiguous;
                        cx.notify();
                    })))
                    .child(toggle("all-layers", "Sample All Layers", self.opts.sample_all).on_click(cx.listener(|e, _, _, cx| {
                        e.opts.sample_all = !e.opts.sample_all;
                        cx.notify();
                    })));
                if self.tool != Tool::MagicWand {
                    bar = bar.child(self.opt_slider("Opacity", "fillop", spec(1.0, 100.0, 1.0, |e| e.fill_opacity * 100.0, |e, v| e.fill_opacity = v / 100.0), |v| format!("{v:.0}%"), 80., window, cx));
                }
            }
            Tool::MagneticLasso => {
                bar = bar
                    .child(self.opt_slider("Width", "mag.w", spec(1.0, 256.0, 1.0, |e| e.opts.magnetic_width, |e, v| e.opts.magnetic_width = v), |v| format!("{v:.0} px"), 70., window, cx))
                    .child(self.opt_slider("Contrast", "mag.c", spec(1.0, 100.0, 1.0, |e| e.opts.magnetic_contrast, |e, v| e.opts.magnetic_contrast = v), |v| format!("{v:.0}%"), 70., window, cx))
                    .child(self.opt_slider("Frequency", "mag.f", spec(0.0, 100.0, 1.0, |e| e.opts.magnetic_frequency, |e, v| e.opts.magnetic_frequency = v), |v| format!("{v:.0}"), 70., window, cx));
            }
            Tool::ObjectSelect => {
                use crate::engine::ml;
                bar = bar
                    .child(toggle("objfind", "Object Finder", self.opts.object_finder).on_click(cx.listener(|e, _, _, cx| {
                        e.opts.object_finder = !e.opts.object_finder;
                        e.hover_object = None;
                        e.ensure_object_finder(cx);
                        cx.notify();
                    })))
                    .child(segmented("objmode", &[(false, "Rectangle"), (true, "Lasso")], self.opts.object_lasso, cx, |e, v| e.opts.object_lasso = v))
                    .child(label(ml::describe(ml::SUBJECT_MODEL, "Model", "classical GrabCut")));
                if self.opts.object_finder && self.finder_building.is_some() {
                    bar = bar.child(label("Finding objects…"));
                }
            }
            Tool::CloneStamp | Tool::Heal => {
                use super::retouch_tools::SampleFrom;
                bar = bar
                    .child(toggle("aligned", "Aligned", self.ropts.aligned).on_click(cx.listener(|e, _, _, cx| {
                        e.ropts.aligned = !e.ropts.aligned;
                        e.ropts.offset = None;
                        cx.notify();
                    })))
                    .child(label("Sample:"))
                    .child(segmented("sample", &SampleFrom::ALL.map(|s| (s, s.label())), self.ropts.sample, cx, |e, v| e.ropts.sample = v))
                    .child(toggle("ovl", "Show Overlay", self.ropts.overlay).on_click(cx.listener(|e, _, _, cx| {
                        e.ropts.overlay = !e.ropts.overlay;
                        cx.notify();
                    })))
                    .child(label(match self.ropts.source {
                        Some((x, y)) => format!("Source {x:.0}, {y:.0}"),
                        None => "Alt-click to set source".into(),
                    }));
                if self.tool == Tool::Heal {
                    use crate::engine::poisson::Quality;
                    bar = bar.child(label("Quality:")).child(segmented("hq", &[(Quality::Fast, "Fast"), (Quality::Exact, "Exact")], self.ropts.heal_quality, cx, |e, v| e.ropts.heal_quality = v));
                }
            }
            Tool::PatternStamp => {
                let pats: Vec<(usize, &'static str)> = crate::engine::retouch::patterns().iter().enumerate().map(|(i, p)| (i, p.name)).collect();
                bar = bar.child(label("Pattern:")).child(segmented("pat", &pats, self.ropts.pattern, cx, |e, v| e.ropts.pattern = v));
            }
            Tool::SpotHeal => {
                use crate::engine::retouch::SpotMode;
                bar = bar.child(label("Type:")).child(segmented("spot", &[(SpotMode::ContentAware, "Content-Aware"), (SpotMode::CreateTexture, "Create Texture"), (SpotMode::ProximityMatch, "Proximity Match")], self.ropts.spot_mode, cx, |e, v| e.ropts.spot_mode = v));
            }
            Tool::Patch => {
                use crate::engine::retouch::PatchMode;
                bar = bar
                    .child(segmented("patch", &[(PatchMode::Source, "Source"), (PatchMode::Destination, "Destination"), (PatchMode::ContentAware, "Content-Aware")], self.ropts.patch_mode, cx, |e, v| e.ropts.patch_mode = v))
                    .child(label("Lasso the area, then drag inside it"));
            }
            Tool::ContentAwareMove => {
                bar = bar
                    .child(segmented("cam", &[(false, "Move"), (true, "Extend")], self.ropts.cam_extend, cx, |e, v| e.ropts.cam_extend = v))
                    .child(self.opt_slider("Adapt", "cam.adapt", spec(0.0, 100.0, 1.0, |e| e.ropts.cam_adapt * 100.0, |e, v| e.ropts.cam_adapt = v / 100.0), |v| format!("{v:.0}%"), 80., window, cx))
                    .child(label("Lasso the object, then drag it"));
            }
            Tool::RedEye => {
                bar = bar
                    .child(self.opt_slider("Pupil Size", "re.p", spec(30.0, 200.0, 1.0, |e| e.ropts.pupil * 100.0, |e, v| e.ropts.pupil = v / 100.0), |v| format!("{v:.0}%"), 80., window, cx))
                    .child(self.opt_slider("Darken", "re.d", spec(1.0, 100.0, 1.0, |e| e.ropts.darken * 100.0, |e, v| e.ropts.darken = v / 100.0), |v| format!("{v:.0}%"), 80., window, cx));
            }
            Tool::HistoryBrush | Tool::ArtHistory => {
                if self.tool == Tool::ArtHistory {
                    use crate::engine::retouch::ArtStyle;
                    bar = bar
                        .child(segmented("art", &ArtStyle::ALL.map(|a| (a, a.label())), self.ropts.art_style, cx, |e, v| e.ropts.art_style = v))
                        .child(self.opt_slider("Area", "art.area", spec_log(4.0, 500.0, 1.0, |e| e.ropts.art_area, |e, v| e.ropts.art_area = v), |v| format!("{v:.0} px"), 80., window, cx));
                }
                bar = bar.child(label(format!("Source: history state {} (click ○ in History)", self.ropts.history_pos)));
            }
            Tool::Move => {
                bar = bar
                    .child(toggle("autosel", "Auto-Select", self.opts.auto_select).on_click(cx.listener(|e, _, _, cx| {
                        e.opts.auto_select = !e.opts.auto_select;
                        cx.notify();
                    })))
                    .child(segmented("autosel-what", &[(false, "Layer"), (true, "Group")], self.opts.auto_select_group, cx, |e, v| e.opts.auto_select_group = v))
                    .child(toggle("xform", "Show Transform Controls", self.opts.show_transform).on_click(cx.listener(|e, _, _, cx| {
                        e.opts.show_transform = !e.opts.show_transform;
                        cx.notify();
                    })));
            }
            Tool::Eyedropper | Tool::ColorSampler => {
                bar = bar.child(segmented("ssize", &SampleSize::ALL.map(|s| (s, sample_label(s))), self.opts.sample_size, cx, |e, v| e.opts.sample_size = v));
            }
            Tool::Liquify => {
                let mut modes = div().flex().gap_0p5();
                for (i, m) in crate::engine::liquify::Mode::ALL.into_iter().enumerate() {
                    let active = self.liquify_mode == m;
                    let b = Button::new(SharedString::from(format!("liq-{i}")))
                        .label(m.label())
                        .xsmall()
                        .tooltip(m.help())
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.liquify_mode = m;
                            this.flash(m.help());
                            cx.notify();
                        }));
                    modes = modes.child(if active { b.primary() } else { b.ghost() });
                }
                bar = bar
                    .child(modes)
                    .child(self.opt_slider("Size", "liq.size", spec_log(4.0, 8000.0, 1.0, |e| e.liquify.size, |e, v| e.liquify.size = v), |v| format!("{v:.0} px"), 110., window, cx))
                    .child(self.opt_slider("Pressure", "liq.p", spec(1.0, 100.0, 1.0, |e| e.liquify.flow * 100.0, |e, v| e.liquify.flow = v / 100.0), |v| format!("{v:.0}%"), 70., window, cx))
                    .child(self.opt_slider("Rate", "liq.r", spec(1.0, 100.0, 1.0, |e| e.liquify.hardness * 100.0, |e, v| e.liquify.hardness = v / 100.0), |v| format!("{v:.0}"), 70., window, cx));
            }
            Tool::Hand => {
                bar = bar
                    .child(Button::new("z-fit").label("Fit on Screen").small().on_click(cx.listener(|this, _, _, cx| this.act_zoom(2, cx))))
                    .child(Button::new("z-100").label("100%").small().on_click(cx.listener(|this, _, _, cx| this.act_zoom(3, cx))));
            }
            _ => {}
        }
        let _ = pct;
        bar
    }
}

pub fn sample_label(s: SampleSize) -> &'static str {
    match s {
        SampleSize::Point => "Point",
        SampleSize::Avg3 => "3×3",
        SampleSize::Avg5 => "5×5",
        SampleSize::Avg11 => "11×11",
        SampleSize::Avg31 => "31×31",
        SampleSize::Avg51 => "51×51",
        SampleSize::Avg101 => "101×101",
    }
}
