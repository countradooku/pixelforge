//! Layer ▸ Layer Style: the Layer Style dialog (Blending Options and every
//! effect, with live preview: the style is part of the composite, so a
//! change only marks the document dirty), its menu actions, and the "fx"
//! badge / Fill slider helpers for the Layers panel.

use std::sync::{Arc, OnceLock};

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::color_picker::{ColorPicker, ColorPickerEvent, ColorPickerState};
use gpui_kit::component::select::{Select, SelectEvent, SelectState};
use gpui_kit::component::slider::{Slider, SliderEvent, SliderState};
use gpui_kit::component::{Icon, IndexPath, Selectable as _, Sizable as _};
use gpui_kit::prelude::*;
use gpui_kit::{AnyElement, Context, Entity, Rgba, SharedString, Subscription, Window, div, px, rgb};

use crate::engine::blend::BlendMode;
use crate::engine::document::Layer;
use crate::engine::layers::LayerKind;
use crate::engine::styles::{BevelStyle, Effect, LayerStyle, StrokePos};

use super::editor::{Editor, rgb_to_hsla};

const PANEL_HI: u32 = 0x2f2f32;
const BORDER: u32 = 0x3a3a3d;
const TEXT: u32 = 0xdcdcdc;
const DIM: u32 = 0x8e8e93;
const ACCENT: u32 = 0x2f6fde;

/// A page of the dialog.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Section {
    Blending,
    Fx(Effect),
}

impl Section {
    /// Menu keys (`actions::LayerStyleOp`).
    pub fn from_key(k: &str) -> Option<Section> {
        Some(match k {
            "blending" => Section::Blending,
            "bevel" => Section::Fx(Effect::Bevel),
            "stroke" => Section::Fx(Effect::Stroke),
            "inner_shadow" => Section::Fx(Effect::InnerShadow),
            "inner_glow" => Section::Fx(Effect::InnerGlow),
            "satin" => Section::Fx(Effect::Satin),
            "color_overlay" => Section::Fx(Effect::ColorOverlay),
            "gradient_overlay" => Section::Fx(Effect::GradientOverlay),
            "outer_glow" => Section::Fx(Effect::OuterGlow),
            "drop_shadow" => Section::Fx(Effect::DropShadow),
            _ => return None,
        })
    }

    pub fn title(self) -> &'static str {
        match self {
            Section::Blending => "Blending Options",
            Section::Fx(e) => e.name(),
        }
    }
}

/// The menu entries of Layer ▸ Layer Style (key, label).
pub const MENU: [(&str, &str); 10] = [
    ("blending", "Blending Options…"),
    ("bevel", "Bevel & Emboss…"),
    ("stroke", "Stroke…"),
    ("inner_shadow", "Inner Shadow…"),
    ("inner_glow", "Inner Glow…"),
    ("satin", "Satin…"),
    ("color_overlay", "Color Overlay…"),
    ("gradient_overlay", "Gradient Overlay…"),
    ("outer_glow", "Outer Glow…"),
    ("drop_shadow", "Drop Shadow…"),
];

fn sty(l: &Layer) -> &LayerStyle {
    static D: OnceLock<LayerStyle> = OnceLock::new();
    l.style.as_deref().unwrap_or_else(|| D.get_or_init(LayerStyle::default))
}

/// Replaces the layer's style with an edited copy (styles are shared with
/// history, so they are never mutated in place).
fn edit(l: &mut Layer, f: impl FnOnce(&mut LayerStyle)) {
    let mut s = sty(l).clone();
    f(&mut s);
    l.style = Some(Arc::new(s));
}

/// One control. Getters and setters act on the layer.
enum Param {
    Slider { label: &'static str, min: f32, max: f32, unit: &'static str, get: fn(&Layer) -> f32, set: fn(&mut Layer, f32) },
    Color { label: &'static str, get: fn(&Layer) -> [u8; 3], set: fn(&mut Layer, [u8; 3]) },
    Blend { label: &'static str, get: fn(&Layer) -> BlendMode, set: fn(&mut Layer, BlendMode) },
    Choice { label: &'static str, options: &'static [&'static str], get: fn(&Layer) -> usize, set: fn(&mut Layer, usize) },
    Toggle { label: &'static str, get: fn(&Layer) -> bool, set: fn(&mut Layer, bool) },
}

macro_rules! slider {
    ($label:expr, $min:expr, $max:expr, $unit:expr, $($f:tt)+) => {
        Param::Slider { label: $label, min: $min, max: $max, unit: $unit, get: |l| sty(l).$($f)+, set: |l, v| edit(l, |s| s.$($f)+ = v) }
    };
}
macro_rules! color {
    ($label:expr, $($f:tt)+) => {
        Param::Color { label: $label, get: |l| sty(l).$($f)+, set: |l, v| edit(l, |s| s.$($f)+ = v) }
    };
}
macro_rules! blend {
    ($label:expr, $($f:tt)+) => {
        Param::Blend { label: $label, get: |l| sty(l).$($f)+, set: |l, v| edit(l, |s| s.$($f)+ = v) }
    };
}
macro_rules! toggle {
    ($label:expr, $($f:tt)+) => {
        Param::Toggle { label: $label, get: |l| sty(l).$($f)+, set: |l, v| edit(l, |s| s.$($f)+ = v) }
    };
}

const BEVEL_STYLES: [&str; 4] = [BevelStyle::Outer.name(), BevelStyle::Inner.name(), BevelStyle::Emboss.name(), BevelStyle::Pillow.name()];
const POSITIONS: [&str; 3] = [StrokePos::Outside.name(), StrokePos::Inside.name(), StrokePos::Center.name()];

fn params(section: Section) -> Vec<Param> {
    use Param::*;
    match section {
        Section::Blending => vec![
            Blend { label: "Blend Mode", get: |l| l.blend, set: |l, v| l.blend = v },
            Slider { label: "Opacity", min: 0.0, max: 100.0, unit: "%", get: |l| l.opacity * 100.0, set: |l, v| l.opacity = v / 100.0 },
            Slider { label: "Fill Opacity", min: 0.0, max: 100.0, unit: "%", get: |l| l.fill * 100.0, set: |l, v| l.fill = v / 100.0 },
        ],
        Section::Fx(Effect::DropShadow) => vec![
            blend!("Blend Mode", drop_shadow.blend),
            color!("Color", drop_shadow.color),
            slider!("Opacity", 0.0, 100.0, "%", drop_shadow.opacity),
            slider!("Angle", -180.0, 180.0, "°", drop_shadow.angle),
            slider!("Distance", 0.0, 500.0, "px", drop_shadow.distance),
            slider!("Spread", 0.0, 100.0, "%", drop_shadow.spread),
            slider!("Size", 0.0, 250.0, "px", drop_shadow.size),
            slider!("Noise", 0.0, 100.0, "%", drop_shadow.noise),
            toggle!("Layer Knocks Out Drop Shadow", drop_shadow.knockout),
        ],
        Section::Fx(Effect::InnerShadow) => vec![
            blend!("Blend Mode", inner_shadow.blend),
            color!("Color", inner_shadow.color),
            slider!("Opacity", 0.0, 100.0, "%", inner_shadow.opacity),
            slider!("Angle", -180.0, 180.0, "°", inner_shadow.angle),
            slider!("Distance", 0.0, 500.0, "px", inner_shadow.distance),
            slider!("Choke", 0.0, 100.0, "%", inner_shadow.spread),
            slider!("Size", 0.0, 250.0, "px", inner_shadow.size),
        ],
        Section::Fx(Effect::OuterGlow) => vec![
            blend!("Blend Mode", outer_glow.blend),
            color!("Color", outer_glow.color),
            slider!("Opacity", 0.0, 100.0, "%", outer_glow.opacity),
            slider!("Spread", 0.0, 100.0, "%", outer_glow.spread),
            slider!("Size", 0.0, 250.0, "px", outer_glow.size),
            slider!("Range", 1.0, 100.0, "%", outer_glow.range),
        ],
        Section::Fx(Effect::InnerGlow) => vec![
            blend!("Blend Mode", inner_glow.blend),
            color!("Color", inner_glow.color),
            slider!("Opacity", 0.0, 100.0, "%", inner_glow.opacity),
            Choice { label: "Source", options: &["Center", "Edge"], get: |l| (!sty(l).inner_glow.center) as usize, set: |l, i| edit(l, |s| s.inner_glow.center = i == 0) },
            slider!("Choke", 0.0, 100.0, "%", inner_glow.spread),
            slider!("Size", 0.0, 250.0, "px", inner_glow.size),
            slider!("Range", 1.0, 100.0, "%", inner_glow.range),
        ],
        Section::Fx(Effect::Bevel) => vec![
            Choice { label: "Style", options: &BEVEL_STYLES, get: |l| sty(l).bevel.style as usize, set: |l, i| edit(l, |s| s.bevel.style = BevelStyle::ALL[i]) },
            Choice { label: "Direction", options: &["Up", "Down"], get: |l| (!sty(l).bevel.up) as usize, set: |l, i| edit(l, |s| s.bevel.up = i == 0) },
            slider!("Depth", 1.0, 1000.0, "%", bevel.depth),
            slider!("Size", 0.0, 250.0, "px", bevel.size),
            slider!("Soften", 0.0, 16.0, "px", bevel.soften),
            slider!("Angle", -180.0, 180.0, "°", bevel.angle),
            slider!("Altitude", 0.0, 90.0, "°", bevel.altitude),
            blend!("Highlight Mode", bevel.hi_blend),
            color!("Highlight Color", bevel.hi_color),
            slider!("Highlight Opacity", 0.0, 100.0, "%", bevel.hi_opacity),
            blend!("Shadow Mode", bevel.sh_blend),
            color!("Shadow Color", bevel.sh_color),
            slider!("Shadow Opacity", 0.0, 100.0, "%", bevel.sh_opacity),
        ],
        Section::Fx(Effect::Satin) => vec![
            blend!("Blend Mode", satin.blend),
            color!("Color", satin.color),
            slider!("Opacity", 0.0, 100.0, "%", satin.opacity),
            slider!("Angle", -180.0, 180.0, "°", satin.angle),
            slider!("Distance", 1.0, 250.0, "px", satin.distance),
            slider!("Size", 0.0, 250.0, "px", satin.size),
            toggle!("Invert", satin.invert),
        ],
        Section::Fx(Effect::ColorOverlay) => vec![blend!("Blend Mode", color_overlay.blend), color!("Color", color_overlay.color), slider!("Opacity", 0.0, 100.0, "%", color_overlay.opacity)],
        Section::Fx(Effect::GradientOverlay) => vec![
            blend!("Blend Mode", gradient_overlay.blend),
            slider!("Opacity", 0.0, 100.0, "%", gradient_overlay.opacity),
            color!("Start Color", gradient_overlay.from),
            color!("End Color", gradient_overlay.to),
            Choice {
                label: "Style",
                options: &["Linear", "Radial"],
                get: |l| sty(l).gradient_overlay.radial as usize,
                set: |l, i| edit(l, |s| s.gradient_overlay.radial = i == 1),
            },
            slider!("Angle", -180.0, 180.0, "°", gradient_overlay.angle),
            slider!("Scale", 10.0, 150.0, "%", gradient_overlay.scale),
            toggle!("Reverse", gradient_overlay.reverse),
        ],
        Section::Fx(Effect::Stroke) => vec![
            slider!("Size", 1.0, 250.0, "px", stroke.size),
            Choice { label: "Position", options: &POSITIONS, get: |l| sty(l).stroke.position as usize, set: |l, i| edit(l, |s| s.stroke.position = StrokePos::ALL[i]) },
            blend!("Blend Mode", stroke.blend),
            slider!("Opacity", 0.0, 100.0, "%", stroke.opacity),
            color!("Color", stroke.color),
        ],
    }
}

enum Widget {
    None,
    Slider(Entity<SliderState>),
    Color(Entity<ColorPickerState>),
    Blend(Entity<SelectState<Vec<&'static str>>>),
}

/// The open Layer Style dialog.
pub struct StyleDialog {
    /// Document and layer being edited.
    pub doc_uid: u64,
    pub layer: u64,
    pub section: Section,
    params: Vec<Param>,
    widgets: Vec<Widget>,
    /// Whether an undo step was recorded (Cancel undoes it).
    changed: bool,
    _subs: Vec<Subscription>,
}

fn hsla_to_rgb(h: gpui_kit::Hsla) -> [u8; 3] {
    let c: Rgba = h.to_rgb();
    [(c.r * 255.0).round() as u8, (c.g * 255.0).round() as u8, (c.b * 255.0).round() as u8]
}

/// Whether the layer shows an "fx" badge.
pub fn has_fx(l: &Layer) -> bool {
    l.style.as_ref().is_some_and(|s| s.any())
}

impl Editor {
    fn style_layer_index(&self) -> Option<usize> {
        let d = self.lstyle.as_ref()?;
        if d.doc_uid != self.docs.uid {
            return None;
        }
        self.doc.layers.iter().position(|l| l.id == d.layer)
    }

    /// Layer ▸ Layer Style ▸ … (`key` from [`MENU`], or "clear").
    pub fn layer_style_op(&mut self, key: &str, window: &mut Window, cx: &mut Context<Self>) {
        if key == "clear" {
            let li = self.doc.active;
            if self.doc.layers[li].style.is_some() {
                self.doc.push_undo("Clear Layer Style");
                self.doc.layers[li].style = None;
                self.doc.mark_all_dirty();
            }
            self.lstyle = None;
            cx.notify();
            return;
        }
        if let Some(s) = Section::from_key(key) {
            self.open_layer_style(s, window, cx);
        }
    }

    pub fn open_layer_style(&mut self, section: Section, window: &mut Window, cx: &mut Context<Self>) {
        if self.input_blocked() {
            return;
        }
        let li = self.doc.active;
        let l = &self.doc.layers[li];
        if matches!(l.kind, LayerKind::Adjustment(_) | LayerKind::Artboard(_)) {
            self.flash("Layer styles apply to pixel, type, shape, smart object, frame and group layers");
            return;
        }
        if l.background {
            self.flash("The Background layer is locked: double-click it in Layers to unlock it first");
            return;
        }
        self.lstyle = Some(StyleDialog { doc_uid: self.docs.uid, layer: l.id, section, params: vec![], widgets: vec![], changed: false, _subs: vec![] });
        self.select_style_section(section, window, cx);
    }

    /// Shows a page; picking an effect turns it on (like Photoshop).
    pub fn select_style_section(&mut self, section: Section, window: &mut Window, cx: &mut Context<Self>) {
        let Some(li) = self.style_layer_index() else {
            self.lstyle = None;
            return;
        };
        if let Section::Fx(e) = section {
            if !sty(&self.doc.layers[li]).enabled(e) {
                self.style_apply(|l| edit(l, |s| s.set_enabled(e, true)));
            }
        }
        let params = params(section);
        let layer = &self.doc.layers[li];
        let mut widgets = Vec::new();
        let mut subs = Vec::new();
        for (i, p) in params.iter().enumerate() {
            match p {
                Param::Slider { min, max, get, .. } => {
                    let v = get(layer).clamp(*min, *max);
                    let st = cx.new(|_| SliderState::new().min(*min).max(*max).step(1.0).default_value(v));
                    subs.push(cx.subscribe_in(&st, window, move |this, _, ev: &SliderEvent, _, cx| {
                        let v = match ev {
                            SliderEvent::Change(v) | SliderEvent::Release(v) => v.start(),
                        };
                        this.style_param(i, ParamValue::F(v), cx);
                    }));
                    widgets.push(Widget::Slider(st));
                }
                Param::Color { get, .. } => {
                    let c = get(layer);
                    let st = cx.new(|cx| ColorPickerState::new(window, cx).default_value(rgb_to_hsla(c)));
                    subs.push(cx.subscribe_in(&st, window, move |this, _, ev: &ColorPickerEvent, _, cx| {
                        if let ColorPickerEvent::Change(Some(c)) = ev {
                            this.style_param(i, ParamValue::C(hsla_to_rgb(*c)), cx);
                        }
                    }));
                    widgets.push(Widget::Color(st));
                }
                Param::Blend { get, .. } => {
                    let names: Vec<&'static str> = BlendMode::ALL.iter().map(|b| b.name()).collect();
                    let cur = get(layer).index();
                    let st = cx.new(|cx| SelectState::new(names, Some(IndexPath::new(cur)), window, cx));
                    subs.push(cx.subscribe_in(&st, window, move |this, _, ev: &SelectEvent<Vec<&'static str>>, _, cx| {
                        let SelectEvent::Confirm(Some(name)) = ev else { return };
                        let name = name.to_string();
                        if let Some(m) = BlendMode::ALL.iter().copied().find(|m| m.name() == name) {
                            this.style_param(i, ParamValue::B(m), cx);
                        }
                    }));
                    widgets.push(Widget::Blend(st));
                }
                Param::Choice { .. } | Param::Toggle { .. } => widgets.push(Widget::None),
            }
        }
        if let Some(d) = &mut self.lstyle {
            d.section = section;
            d.params = params;
            d.widgets = widgets;
            d._subs = subs;
        }
        cx.notify();
    }

    /// Applies an edit to the styled layer (one undo step per dialog).
    fn style_apply(&mut self, f: impl FnOnce(&mut Layer)) {
        let Some(li) = self.style_layer_index() else { return };
        if let Some(d) = &mut self.lstyle {
            if !d.changed {
                d.changed = true;
                self.doc.push_undo("Layer Style");
            }
        }
        f(&mut self.doc.layers[li]);
        self.doc.mark_all_dirty();
    }

    fn style_param(&mut self, i: usize, v: ParamValue, cx: &mut Context<Self>) {
        let Some(d) = &self.lstyle else { return };
        let Some(li) = self.style_layer_index() else { return };
        let Some(p) = d.params.get(i) else { return };
        let l = &self.doc.layers[li];
        // Skip no-op events (initial values echoed back).
        let (same, apply): (bool, Box<dyn FnOnce(&mut Layer)>) = match (p, v) {
            (Param::Slider { get, set, .. }, ParamValue::F(v)) => ((get(l) - v).abs() < 1e-4, {
                let set = *set;
                Box::new(move |l| set(l, v))
            }),
            (Param::Color { get, set, .. }, ParamValue::C(c)) => (get(l) == c, {
                let set = *set;
                Box::new(move |l| set(l, c))
            }),
            (Param::Blend { get, set, .. }, ParamValue::B(m)) => (get(l) == m, {
                let set = *set;
                Box::new(move |l| set(l, m))
            }),
            (Param::Choice { get, set, .. }, ParamValue::I(k)) => (get(l) == k, {
                let set = *set;
                Box::new(move |l| set(l, k))
            }),
            (Param::Toggle { get, set, .. }, ParamValue::T(b)) => (get(l) == b, {
                let set = *set;
                Box::new(move |l| set(l, b))
            }),
            _ => return,
        };
        if !same {
            self.style_apply(apply);
            cx.notify();
        }
    }

    pub fn close_layer_style(&mut self, ok: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(d) = self.lstyle.take() else { return };
        if !ok && d.changed && d.doc_uid == self.docs.uid && self.doc.undo_label() == Some("Layer Style") {
            self.doc.undo();
        }
        self.doc.mark_all_dirty();
        self.sync_layer_controls(window, cx);
        cx.notify();
    }

    /// The dialog (floating over the canvas, like filter dialogs, so the
    /// document stays visible).
    pub fn render_layer_style(&mut self, cx: &mut Context<Self>) -> Option<AnyElement> {
        let li = self.style_layer_index()?;
        let d = self.lstyle.as_ref()?;
        let layer = &self.doc.layers[li];
        let style = sty(layer);
        // Effect list with checkboxes.
        let mut list = div().flex().flex_col().gap_0p5().w(px(160.)).pr_2().border_r_1().border_color(rgb(BORDER));
        let item = |id: String, sel: bool| {
            div()
                .id(SharedString::from(id))
                .flex_1()
                .px_1p5()
                .py_1()
                .rounded_sm()
                .text_color(rgb(TEXT))
                .when(sel, |d| d.bg(rgb(ACCENT).opacity(0.35)))
                .when(!sel, |d| d.hover(|s| s.bg(rgb(BORDER))))
        };
        list = list.child(
            div().flex().child(
                item("ls-blending".into(), d.section == Section::Blending)
                    .child("Blending Options")
                    .on_click(cx.listener(|t, _, w, cx| t.select_style_section(Section::Blending, w, cx))),
            ),
        );
        for e in Effect::ALL {
            let on = style.enabled(e);
            let check = div()
                .id(SharedString::from(format!("ls-on-{e:?}")))
                .size(px(14.))
                .flex()
                .flex_none()
                .items_center()
                .justify_center()
                .rounded_sm()
                .border_1()
                .border_color(rgb(if on { ACCENT } else { DIM }))
                .when(on, |d| d.bg(rgb(ACCENT)).child(Icon::new(gpui_kit::assets::IconName::Check).size(px(10.)).text_color(rgb(0xffffff))))
                .on_click(cx.listener(move |t, _, _, cx| {
                    let on = t.style_layer_index().is_some_and(|li| sty(&t.doc.layers[li]).enabled(e));
                    t.style_apply(|l| edit(l, |s| s.set_enabled(e, !on)));
                    cx.notify();
                }));
            list = list.child(
                div().flex().items_center().gap_1().pl_1().child(check).child(
                    item(format!("ls-{e:?}"), d.section == Section::Fx(e))
                        .child(e.name())
                        .on_click(cx.listener(move |t, _, w, cx| t.select_style_section(Section::Fx(e), w, cx))),
                ),
            );
        }
        // Parameters of the page.
        let mut body = div().id("ls-body").flex_1().min_w_0().max_h(px(460.)).overflow_y_scroll().flex().flex_col().gap_2p5().pl_3();
        for (i, (p, w)) in d.params.iter().zip(&d.widgets).enumerate() {
            let label_row = |label: &'static str, value: Option<String>| {
                div()
                    .flex()
                    .justify_between()
                    .child(div().text_color(rgb(DIM)).child(label))
                    .when_some(value, |d, v| d.child(div().text_color(rgb(TEXT)).child(v)))
            };
            let el: AnyElement = match (p, w) {
                (Param::Slider { label, unit, get, .. }, Widget::Slider(s)) => {
                    div().flex().flex_col().gap_1().child(label_row(label, Some(format!("{:.0} {unit}", get(layer))))).child(Slider::new(s)).into_any_element()
                }
                (Param::Color { label, .. }, Widget::Color(s)) => {
                    div().flex().items_center().justify_between().child(div().text_color(rgb(DIM)).child(*label)).child(ColorPicker::new(s).small()).into_any_element()
                }
                (Param::Blend { label, .. }, Widget::Blend(s)) => div().flex().flex_col().gap_1().child(label_row(label, None)).child(Select::new(s).small()).into_any_element(),
                (Param::Choice { label, options, get, .. }, _) => {
                    let cur = get(layer);
                    let mut btns = div().flex().flex_wrap().gap_1();
                    for (k, name) in options.iter().enumerate() {
                        btns = btns.child(
                            Button::new(SharedString::from(format!("ls-c{i}-{k}")))
                                .label(*name)
                                .xsmall()
                                .selected(k == cur)
                                .on_click(cx.listener(move |t, _, _, cx| t.style_param(i, ParamValue::I(k), cx))),
                        );
                    }
                    div().flex().flex_col().gap_1().child(label_row(label, None)).child(btns).into_any_element()
                }
                (Param::Toggle { label, get, .. }, _) => {
                    let on = get(layer);
                    Button::new(SharedString::from(format!("ls-t{i}")))
                        .label(*label)
                        .xsmall()
                        .selected(on)
                        .on_click(cx.listener(move |t, _, _, cx| t.style_param(i, ParamValue::T(!on), cx)))
                        .into_any_element()
                }
                _ => div().into_any_element(),
            };
            body = body.child(el);
        }
        if let Section::Fx(e) = d.section {
            if !style.enabled(e) {
                body = body.child(div().text_size(px(11.)).text_color(rgb(DIM)).child("This effect is off: tick its checkbox to turn it on."));
            }
        }
        body = body.child(div().text_size(px(11.)).text_color(rgb(DIM)).child(
            "Effects are live: they're part of the composite, computed from the layer's shape at every zoom level. Fill fades the layer's pixels but not its effects.",
        ));
        let title = format!("Layer Style · {} · {}", layer.name, d.section.title());
        Some(
            div()
                .id("layer-style")
                .occlude()
                .w(px(560.))
                .p_4()
                .rounded_lg()
                .bg(rgb(PANEL_HI))
                .border_1()
                .border_color(rgb(BORDER))
                .shadow_lg()
                .flex()
                .flex_col()
                .gap_3()
                .on_mouse_down(gpui_kit::MouseButton::Left, |_, _, cx| cx.stop_propagation())
                .child(div().text_size(px(14.)).text_color(rgb(TEXT)).truncate().child(title))
                .child(div().flex().flex_row().child(list).child(body))
                .child(
                    div()
                        .flex()
                        .justify_between()
                        .gap_2()
                        .child(Button::new("ls-clear").label("Clear Layer Style").small().ghost().on_click(cx.listener(|t, _, w, cx| {
                            t.style_apply(|l| l.style = None);
                            let s = t.lstyle.as_ref().map(|d| d.section).unwrap_or(Section::Blending);
                            t.select_style_section(if matches!(s, Section::Fx(_)) { Section::Blending } else { s }, w, cx);
                        })))
                        .child(
                            div()
                                .flex()
                                .gap_2()
                                .child(Button::new("ls-cancel").label("Cancel").small().on_click(cx.listener(|t, _, w, cx| t.close_layer_style(false, w, cx))))
                                .child(Button::new("ls-ok").label("OK").small().primary().on_click(cx.listener(|t, _, w, cx| t.close_layer_style(true, w, cx)))),
                        ),
                )
                .into_any_element(),
        )
    }
}

#[derive(Clone, Copy)]
enum ParamValue {
    F(f32),
    C([u8; 3]),
    B(BlendMode),
    I(usize),
    T(bool),
}
