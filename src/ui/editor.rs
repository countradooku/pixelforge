//! Editor state, tools, input handling and commands.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use gpui_kit::component::color_picker::{ColorPickerEvent, ColorPickerState};
use gpui_kit::component::input::InputState;
use gpui_kit::component::select::{SelectEvent, SelectState};
use gpui_kit::component::slider::{SliderEvent, SliderScale, SliderState};
use gpui_kit::component::{IndexPath, WindowExt as _};
use gpui_kit::prelude::*;
use gpui_kit::{
    Bounds, Context, CursorStyle, Entity, FocusHandle, Hsla, KeyDownEvent, KeyUpEvent, MouseButton, MouseDownEvent,
    MouseMoveEvent, MouseUpEvent, PathPromptOptions, PinchEvent, Pixels, Point, RenderImage, Rgba, ScrollDelta,
    ScrollWheelEvent, SharedString, Subscription, Window, point, px,
};

use crate::engine::blend::BlendMode;
use crate::engine::brush::{BrushSettings, Stroke};
use crate::engine::composite::{self, Preview};
use crate::engine::document::{Document, Layer};
use crate::engine::filters::Filter;
use crate::engine::grid::{RectI, TileGrid};
use crate::engine::io;
use crate::engine::ops::{self, Progress};
use crate::engine::px;
use crate::engine::masks::SelOp;
use crate::engine::selection::{Selection, Shape};

use super::viewport::{DisplayCache, FramePlan, Viewport, thumbnail};

pub use super::tools::Tool;
use super::tools::GROUPS;

pub enum Drag {
    None,
    Paint(Box<Stroke>, super::layers_ui::PaintTarget),
    Liquify { layer: usize },
    Pan { start: Point<Pixels>, pan0: Point<Pixels> },
    Marquee { start: (f32, f32), cur: (f32, f32), ellipse: bool, center: bool, op: SelOp },
    Lasso(Vec<[f32; 2]>, SelOp),
    QuickSelect(Box<super::select_tools::QuickDrag>),
    SpotHeal(Box<Stroke>),
    GradHandle(u8),
    /// A vector-tool drag (state lives in `Editor::vdrag`).
    Vector,
    TypeBox { start: (f32, f32), cur: (f32, f32) },
    /// Drag-selecting text while editing.
    TypeSelect,
    /// Dragging a type-on-a-path start / end marker.
    TypePathHandle { end: bool },
    PatchMove { start: (f32, f32), cur: (f32, f32) },
    ObjectSelect { start: (f32, f32), cur: (f32, f32), lasso: Vec<[f32; 2]>, op: SelOp },
    /// Moving the selected layers (`ids`; `layer` is the key layer).
    Move { start: (f32, f32), layer: usize, ids: Vec<u64> },
    Gradient { start: (f32, f32), cur: (f32, f32) },
    /// A navigation / measurement / canvas tool drag (state in `Editor::ndrag`).
    Nav,
    /// Painting the Content-Aware Fill sampling area.
    CafPaint { last: (f32, f32), subtract: bool },
    /// A Vanishing Point workspace drag (state in `Editor::vpw`).
    Vp,
    /// A Perspective Warp workspace drag (state in `Editor::pww`).
    Pw,
}

/// Quick Mask: a gray grid (white = selected) painted with any paint tool,
/// shown as a red overlay through `ComposeInput.extra`.
pub struct QuickMask {
    pub grid: TileGrid,
    pub overlay: Layer,
}

pub struct Busy {
    pub label: SharedString,
    pub progress: Arc<Progress>,
    pub started: Instant,
}

pub use crate::engine::filters::catalog::Kind as FilterKind;

pub enum DialogKind {
    NewDocument,
    ImageSize,
    /// Settings ▸ Performance (GPU).
    Preferences,
    Modify(super::select_tools::Modify),
    Filter(FilterKind),
    SelectMask,
}

pub struct DialogSlider {
    pub label: &'static str,
    pub state: Entity<SliderState>,
    pub decimals: usize,
    pub choices: &'static [&'static str],
}

pub struct Dialog {
    pub kind: DialogKind,
    pub sliders: Vec<DialogSlider>,
    pub inputs: Vec<(&'static str, Entity<InputState>)>,
    pub flag: bool,
    /// Histogram stats for statistics-driven filters (sampled once on open).
    pub stats: Option<Arc<crate::engine::filters::Stats>>,
    pub source: Option<Arc<crate::engine::filters::Stats>>,
    /// Image chosen as the Displace map.
    pub disp_map: Option<Arc<crate::engine::filters::extra::DispMap>>,
    pub(crate) _subs: Vec<Subscription>,
}

pub struct Ui {
    pub layer_opacity: Entity<SliderState>,
    /// Fill opacity (Layer Styles).
    pub layer_fill: Entity<SliderState>,
    pub blend: Entity<SelectState<Vec<&'static str>>>,
    pub fg: Entity<ColorPickerState>,
    pub bg: Entity<ColorPickerState>,
}

pub enum Overlay {
    Quad { bounds: Bounds<Pixels>, bg: Hsla, border: Hsla, border_w: f32, radius: f32 },
    Image { bounds: Bounds<Pixels>, img: Arc<RenderImage>, radius: f32 },
    Path { polys: Vec<(Vec<Point<Pixels>>, bool)>, dashed: bool, color: Hsla, width: f32 },
    /// Even-odd filled polygons (shading with holes).
    Fill { polys: Vec<Vec<Point<Pixels>>>, color: Hsla },
    Text { pos: Point<Pixels>, text: SharedString, color: Hsla, size: f32, bg: Option<Hsla> },
}

pub struct CanvasFrame {
    pub plan: FramePlan,
    pub overlays: Vec<Overlay>,
    pub cursor: CursorStyle,
    /// Rotate View angle: small square handles turn with the view.
    pub rot: f32,
}

pub struct Editor {
    pub doc: Document,
    pub vp: Viewport,
    pub cache: DisplayCache,
    pub canvas: Bounds<Pixels>,
    pub sf: f32,
    pub needs_fit: bool,
    pub tool: Tool,
    /// The tool last used in each toolbar group (what its button shows).
    pub tool_memory: Vec<Tool>,
    /// Open toolbar flyout: group index and its button's top (window px).
    pub flyout: Option<(usize, f32)>,
    /// Pending long-press on a toolbar group (opens its flyout).
    pub press_started: Option<(usize, Instant)>,
    pub toolbar_hidden: std::collections::HashSet<Tool>,
    pub space_hand: bool,
    /// Per-tool brush settings (Photoshop keeps one set per tool).
    pub brushes: HashMap<Tool, BrushSettings>,
    pub liquify: BrushSettings,
    /// Last tablet / Force Touch pressure (1.0 with a mouse).
    pub pressure: f32,
    pub opts: super::options::ToolOptions,
    pub knobs: super::options::Knobs,
    /// Brush tip row of the options bar expanded.
    pub show_tip: bool,
    /// Quick Mask mode: the selection as an editable gray mask.
    pub quick_mask: Option<QuickMask>,
    /// 0 standard, 1 full screen with menus, 2 full screen.
    pub screen_mode: u8,
    pub poly: Option<super::select_tools::PolyState>,
    pub magnetic: Option<super::select_tools::MagState>,
    /// Duration of the last interactive tool step (status / benchmarks).
    pub last_tool_ms: f32,
    /// Result of Settings ▸ Performance ▸ Benchmark GPU.
    pub gpu_bench: Option<String>,
    /// Painting targets the active layer's mask.
    pub mask_edit: bool,
    pub mask_thumbs: HashMap<u64, (u64, Arc<RenderImage>)>,
    pub adjust_menu: bool,
    /// Adjustment layer the open filter dialog edits live.
    pub adjust_target: Option<u64>,
    /// Gray copy of the mask being painted.
    pub mask_paint: Option<TileGrid>,
    pub ropts: super::retouch_tools::RetouchOptions,
    pub popts: super::paint_tools::PaintOptions,
    pub vopts: super::vector_tools::VectorOptions,
    pub vdrag: Option<super::vector_tools::VDrag>,
    /// The Pen's open subpath in the work path.
    pub pen_sub: Option<usize>,
    pub last_trace: Option<crate::engine::vector::P>,
    pub topts: super::type_tools::TypeOptions,
    pub text_edit: Option<super::type_tools::TextEdit>,
    pub nopts: super::nav_tools::NavOptions,
    pub ndrag: Option<super::nav_tools::NavDrag>,
    /// Selected pixels floating while the Move tool drags them.
    pub float: Option<Layer>,
    /// Eyedropper colour while dragging (applied on release).
    pub fg_preview: Option<[u8; 3]>,
    pub grad_live: Option<super::paint_tools::GradLive>,
    pub spot: Option<super::retouch_tools::SpotState>,
    /// Clone-source preview image under the cursor (cache key, image).
    pub clone_preview: Option<((i32, i32, usize, i32), Arc<RenderImage>)>,
    pub doc_dirty_seen: u64,
    /// Marching-ants outlines of the current selection (by shape identity).
    pub outline_cache: std::cell::RefCell<Option<((usize, bool), Vec<Vec<[f32; 2]>>)>>,
    /// Edit ▸ Content-Aware Fill workspace, while open.
    pub caf: Option<super::caf::CafState>,
    /// Select and Mask dialog state (live overlay preview).
    pub select_mask: Option<super::select_mask::SelectMaskState>,
    /// Save for Web workspace and the Slice Options fields.
    pub web: Option<super::web_ui::WebState>,
    pub slice_fields: Option<super::web_ui::SliceFields>,
    /// Object Selection's hover object finder (built in the background).
    pub finder: Option<(u64, Arc<crate::engine::objects::ObjectFinder>)>,
    pub finder_building: Option<u64>,
    pub hover_object: Option<usize>,
    /// Tint image of the hovered object (object id, image).
    pub hover_tint: std::cell::RefCell<Option<(usize, Arc<RenderImage>)>>,
    pub liquify_mode: crate::engine::liquify::Mode,
    /// Image Size ▸ Resample (kept between dialogs, like Photoshop).
    pub size_resample: crate::engine::resample::Resample,
    pub liquify_session: Option<crate::engine::liquify::Session>,
    pub liquify_overlay: Option<Layer>,
    pub fg: [u8; 3],
    pub bg: [u8; 3],
    pub tolerance: f32,
    pub contiguous: bool,
    pub fill_opacity: f32,
    pub gradient_radial: bool,
    pub drag: Drag,
    pub crop: Option<RectI>,
    last_paint_point: Option<(f32, f32)>,
    pub mouse: Option<Point<Pixels>>,
    pub mouse_img: Option<(f32, f32)>,
    pub hover_rgba: Option<[u8; 4]>,
    pub status: SharedString,
    pub busy: Option<Busy>,
    clipboard: Option<TileGrid>,
    pub thumbs: HashMap<u64, (u64, Arc<RenderImage>)>,
    pub dialog: Option<Dialog>,
    pub ui: Ui,
    pub focus: FocusHandle,
    pub history_scroll: gpui_kit::ScrollHandle,
    pub history_seen: usize,
    /// The other open documents, tab order and arrangement.
    pub docs: super::docs::Docs,
    /// Brushes / Brush Settings / Measurement Log / Notes dock.
    pub dockst: super::dock::DockState,
    /// Filter ▸ Vanishing Point workspace, while open.
    pub vpw: Option<super::vp::VpState>,
    /// Edit ▸ Perspective Warp workspace, while open.
    pub pww: Option<super::perspwarp_ui::PwState>,
    /// Layer ▸ Layer Style dialog, while open.
    pub lstyle: Option<super::style_ui::StyleDialog>,
    /// The last background job and how long it took (Info panel).
    pub last_job: Option<(SharedString, f32)>,
    _subs: Vec<Subscription>,
}

/// Photoshop-like defaults per painting tool.
pub fn default_brush_for(t: Tool) -> BrushSettings {
    let d = BrushSettings::default();
    match t {
        Tool::Eraser | Tool::BackgroundEraser => BrushSettings { size: 80.0, hardness: 0.6, ..d },
        Tool::Pencil => BrushSettings { size: 4.0, hardness: 1.0, aliased: true, spacing: 0.25, ..d },
        Tool::CloneStamp | Tool::PatternStamp | Tool::HistoryBrush => BrushSettings { size: 120.0, hardness: 0.5, ..d },
        Tool::ArtHistory => BrushSettings { size: 60.0, hardness: 0.6, ..d },
        Tool::SpotHeal | Tool::Heal => BrushSettings { size: 60.0, hardness: 0.7, spacing: 0.25, ..d },
        Tool::Blur | Tool::Sharpen => BrushSettings { size: 150.0, hardness: 0.4, flow: 0.5, ..d },
        Tool::Smudge => BrushSettings { size: 80.0, hardness: 0.5, flow: 0.5, spacing: 0.08, ..d },
        Tool::Dodge | Tool::Burn | Tool::Sponge => BrushSettings { size: 200.0, hardness: 0.3, flow: 0.5, ..d },
        Tool::Mixer => BrushSettings { size: 80.0, hardness: 0.7, spacing: 0.08, ..d },
        Tool::ColorReplace => BrushSettings { size: 80.0, hardness: 0.8, spacing: 0.25, ..d },
        Tool::QuickSelect => BrushSettings { size: 40.0, hardness: 1.0, ..d },
        _ => d,
    }
}

fn default_brushes(brush: BrushSettings, eraser: BrushSettings) -> HashMap<Tool, BrushSettings> {
    let mut m = HashMap::new();
    m.insert(Tool::Brush, brush);
    m.insert(Tool::Eraser, eraser);
    m
}

pub fn rgb_to_hsla(c: [u8; 3]) -> Hsla {
    Rgba { r: c[0] as f32 / 255.0, g: c[1] as f32 / 255.0, b: c[2] as f32 / 255.0, a: 1.0 }.into()
}

fn hsla_to_rgb(h: Hsla) -> [u8; 3] {
    let c = h.to_rgb();
    [(c.r * 255.0).round() as u8, (c.g * 255.0).round() as u8, (c.b * 255.0).round() as u8]
}

pub(crate) fn slider(cx: &mut Context<Editor>, min: f32, max: f32, step: f32, v: f32, log: bool) -> Entity<SliderState> {
    cx.new(|_| {
        SliderState::new()
            .min(min)
            .max(max)
            .step(step)
            .scale(if log { SliderScale::Logarithmic } else { SliderScale::Linear })
            .default_value(v)
    })
}

impl Editor {
    pub fn new(window: &mut Window, cx: &mut Context<Self>) -> Self {
        let brush = BrushSettings::default();
        let eraser = BrushSettings { size: 80.0, hardness: 0.6, ..BrushSettings::default() };
        let ui = Ui {
            layer_opacity: slider(cx, 0., 100., 1., 100., false),
            layer_fill: slider(cx, 0., 100., 1., 100., false),
            blend: cx.new(|cx| {
                let names: Vec<&'static str> = BlendMode::ALL.iter().map(|b| b.name()).collect();
                SelectState::new(names, Some(IndexPath::new(0)), window, cx)
            }),
            fg: cx.new(|cx| ColorPickerState::new(window, cx).default_value(rgb_to_hsla([20, 20, 24]))),
            bg: cx.new(|cx| ColorPickerState::new(window, cx).default_value(rgb_to_hsla([255, 255, 255]))),
        };

        let mut subs = Vec::new();
        subs.push(cx.subscribe_in(&ui.layer_opacity, window, |this, _, ev: &SliderEvent, _, cx| {
            let v = match ev {
                SliderEvent::Change(v) | SliderEvent::Release(v) => v.start() / 100.0,
            };
            let li = this.doc.active;
            if this.doc.layers[li].background {
                return;
            }
            if (this.doc.layers[li].opacity - v).abs() > 1e-4 {
                if this.doc.undo_label() != Some("Layer Opacity") {
                    this.doc.push_undo("Layer Opacity");
                }
                this.doc.layers[li].opacity = v;
                this.doc.mark_all_dirty();
                cx.notify();
            }
        }));
        subs.push(cx.subscribe_in(&ui.layer_fill, window, |this, _, ev: &SliderEvent, _, cx| {
            let v = match ev {
                SliderEvent::Change(v) | SliderEvent::Release(v) => v.start() / 100.0,
            };
            let li = this.doc.active;
            if this.doc.layers[li].background || matches!(this.doc.layers[li].kind, crate::engine::layers::LayerKind::Adjustment(_)) {
                return;
            }
            if (this.doc.layers[li].fill - v).abs() > 1e-4 {
                if this.doc.undo_label() != Some("Fill Opacity") {
                    this.doc.push_undo("Fill Opacity");
                }
                this.doc.layers[li].fill = v;
                this.doc.mark_all_dirty();
                cx.notify();
            }
        }));
        subs.push(cx.subscribe_in(&ui.blend, window, |this, _, ev: &SelectEvent<Vec<&'static str>>, _, cx| {
            let SelectEvent::Confirm(Some(name)) = ev else { return };
            let name = name.to_string();
            if let Some(mode) = BlendMode::ALL.iter().copied().find(|m| m.name() == name) {
                let li = this.doc.active;
                if this.doc.layers[li].background && mode != BlendMode::Normal {
                    this.flash("The Background layer is locked: double-click it in Layers to unlock");
                    return;
                }
                if this.doc.layers[li].blend != mode {
                    this.doc.push_undo("Blend Mode");
                    this.doc.layers[li].blend = mode;
                    this.doc.mark_all_dirty();
                    cx.notify();
                }
            }
        }));
        subs.push(cx.subscribe_in(&ui.fg, window, |this, _, ev: &ColorPickerEvent, _, cx| {
            if let ColorPickerEvent::Change(Some(c)) = ev {
                this.fg = hsla_to_rgb(*c);
                cx.notify();
            }
        }));
        subs.push(cx.subscribe_in(&ui.bg, window, |this, _, ev: &ColorPickerEvent, _, cx| {
            if let ColorPickerEvent::Change(Some(c)) = ev {
                this.bg = hsla_to_rgb(*c);
                cx.notify();
            }
        }));

        let focus = cx.focus_handle();
        focus.focus(window, cx);
        crate::engine::text::warm_up();
        match std::env::var("PF_DEMO").as_deref() {
            Ok("filters") => cx.defer_in(window, |this, window, cx| this.start_filter_demo(window, cx)),
            Ok("tools") => cx.defer_in(window, |this, window, cx| this.start_tools_demo(window, cx)),
            Ok(_) => cx.defer_in(window, |this, window, cx| this.start_demo(window, cx)),
            Err(_) => {}
        }

        cx.spawn(async move |this, cx| {
            let mut previous = String::new();
            loop {
                cx.background_executor().timer(std::time::Duration::from_secs(1)).await;
                let stats = crate::engine::memory::global().stats();
                let current = format!("{}{:?}", stats.summary(), stats.fault);
                if this.update(cx, |_, cx| { if current != previous { cx.notify(); } }).is_err() { break; }
                previous = current;
            }
        }).detach();
        Editor {
            doc: Document::new(3000, 2000, Some(px::pack(255, 255, 255, 255))),
            vp: Viewport::default(),
            cache: DisplayCache::default(),
            canvas: Bounds::default(),
            sf: window.scale_factor(),
            needs_fit: true,
            tool: Tool::Brush,
            tool_memory: GROUPS.iter().map(|g| g.tools[0]).collect(),
            flyout: None,
            press_started: None,
            toolbar_hidden: Default::default(),
            space_hand: false,
            brushes: default_brushes(brush, eraser),
            pressure: 1.0,
            opts: Default::default(),
            knobs: Default::default(),
            show_tip: false,
            quick_mask: None,
            screen_mode: 0,
            poly: None,
            magnetic: None,
            last_tool_ms: 0.0,
            gpu_bench: None,
            mask_edit: false,
            mask_thumbs: HashMap::new(),
            adjust_menu: false,
            adjust_target: None,
            mask_paint: None,
            outline_cache: Default::default(),
            ropts: Default::default(),
            popts: Default::default(),
            vopts: Default::default(),
            vdrag: None,
            pen_sub: None,
            last_trace: None,
            topts: Default::default(),
            text_edit: None,
            nopts: Default::default(),
            ndrag: None,
            float: None,
            fg_preview: None,
            grad_live: None,
            spot: None,
            clone_preview: None,
            doc_dirty_seen: 0,
            caf: None,
            select_mask: None,
            web: None,
            slice_fields: None,
            finder: None,
            finder_building: None,
            hover_object: None,
            hover_tint: Default::default(),
            liquify: BrushSettings { size: 200.0, hardness: 0.5, opacity: 1.0, flow: 0.5, spacing: 0.1, ..Default::default() },
            liquify_mode: crate::engine::liquify::Mode::ForwardWarp,
            size_resample: Default::default(),
            liquify_session: None,
            liquify_overlay: None,
            fg: [20, 20, 24],
            bg: [255, 255, 255],
            tolerance: 32.0,
            contiguous: true,
            fill_opacity: 1.0,
            gradient_radial: false,
            drag: Drag::None,
            crop: None,
            last_paint_point: None,
            mouse: None,
            mouse_img: None,
            hover_rgba: None,
            status: "Ready".into(),
            busy: None,
            clipboard: None,
            thumbs: HashMap::new(),
            dialog: None,
            ui,
            focus,
            history_scroll: gpui_kit::ScrollHandle::new(),
            history_seen: usize::MAX,
            docs: Default::default(),
            dockst: Default::default(),
            vpw: None,
            pww: None,
            lstyle: None,
            last_job: None,
            _subs: subs,
        }
    }

    pub fn brush_mut(&mut self) -> &mut BrushSettings {
        if self.tool == Tool::Liquify {
            return &mut self.liquify;
        }
        let t = self.tool;
        self.brushes.entry(t).or_insert_with(|| default_brush_for(t))
    }

    pub fn brush_ref(&self) -> &BrushSettings {
        static FALLBACK: std::sync::OnceLock<BrushSettings> = std::sync::OnceLock::new();
        if self.tool == Tool::Liquify {
            return &self.liquify;
        }
        self.brushes.get(&self.tool).unwrap_or_else(|| FALLBACK.get_or_init(BrushSettings::default))
    }

    /// Settings of a specific tool (e.g. the Brush while demoing).
    pub fn brush_of(&mut self, t: Tool) -> &mut BrushSettings {
        self.brushes.entry(t).or_insert_with(|| default_brush_for(t))
    }

    pub fn flash(&mut self, msg: impl Into<SharedString>) {
        self.status = msg.into();
    }

    // ----- sync UI controls from state -------------------------------------------

    /// Options-bar sliders re-sync from the model every frame (see `knob`).
    pub fn sync_brush_sliders(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        cx.notify();
    }

    pub fn sync_layer_controls(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let l = &self.doc.layers[self.doc.active];
        let (op, blend) = (l.opacity * 100.0, l.blend.index());
        self.ui.layer_opacity.update(cx, |s, cx| s.set_value(op, window, cx));
        let fill = self.doc.layers[self.doc.active].fill * 100.0;
        self.ui.layer_fill.update(cx, |s, cx| s.set_value(fill, window, cx));
        self.ui.blend.update(cx, |s, cx| s.set_selected_index(Some(IndexPath::new(blend)), window, cx));
    }

    pub fn set_fg(&mut self, c: [u8; 3], window: &mut Window, cx: &mut Context<Self>) {
        self.fg = c;
        self.ui.fg.update(cx, |s, cx| s.set_value(rgb_to_hsla(c), window, cx));
    }

    pub fn set_bg(&mut self, c: [u8; 3], window: &mut Window, cx: &mut Context<Self>) {
        self.bg = c;
        self.ui.bg.update(cx, |s, cx| s.set_value(rgb_to_hsla(c), window, cx));
    }

    pub fn set_tool(&mut self, tool: Tool, window: &mut Window, cx: &mut Context<Self>) {
        let changed = self.tool != tool;
        if changed && self.grad_live.is_some() {
            self.commit_gradient(cx);
        }
        if changed {
            self.commit_text(cx);
            self.end_pen(cx);
        }
        self.tool = tool;
        self.tool_memory[tool.group()] = tool;
        self.flyout = None;
        if tool.is_brush() || tool == Tool::Liquify {
            self.brush_mut();
        }
        if changed {
            self.sync_brush_sliders(window, cx);
        }
        if tool != Tool::Crop {
            self.crop = None;
        }
        if tool == Tool::ObjectSelect {
            self.ensure_object_finder(cx);
        } else {
            self.hover_object = None;
        }
        if tool != Tool::Liquify && (self.liquify_session.is_some() || self.liquify_overlay.is_some()) {
            self.liquify_session = None;
            self.liquify_overlay = None;
            self.doc.mark_all_dirty();
        }
        cx.notify();
    }

    /// Applies the current Liquify tool at `(x, y)`; `tick` = continuous step.
    fn liquify_tick(&mut self, x: f32, y: f32, tick: bool) {
        let Drag::Liquify { layer: li } = self.drag else { return };
        let Some(s) = &mut self.liquify_session else { return };
        let b = crate::engine::liquify::Brush {
            radius: (self.liquify.size * 0.5).max(2.0),
            pressure: self.liquify.flow,
            rate: self.liquify.hardness,
        };
        let mode = self.liquify_mode;
        let rects = s.stroke_to(mode, x, y, &b, &mut self.doc.layers[li].grid, tick);
        if matches!(mode, crate::engine::liquify::Mode::Freeze | crate::engine::liquify::Mode::Thaw) {
            let g = s.overlay_grid();
            self.liquify_overlay = Some(Layer::new(0, "freeze", g));
            self.doc.mark_all_dirty();
        } else if !rects.is_empty() {
            self.doc.layer_pixels_changed(li, &rects);
            s.version = self.doc.layers[li].version;
        }
    }

    pub fn select_layer(&mut self, li: usize, window: &mut Window, cx: &mut Context<Self>) {
        if li < self.doc.layers.len() {
            self.doc.active = li;
            self.sync_layer_controls(window, cx);
            cx.notify();
        }
    }

    // ----- frame preparation (called from the canvas prepaint) -------------------

    pub fn prepare_frame(&mut self, bounds: Bounds<Pixels>, window: &mut Window, _cx: &mut Context<Self>) -> CanvasFrame {
        self.canvas = bounds;
        self.sf = window.scale_factor();
        if self.needs_fit && bounds.size.width > px(10.) {
            self.vp.fit(self.doc.w, self.doc.h, bounds, self.sf);
            self.needs_fit = false;
        }
        self.sync_views();
        self.paint_tick(window);
        // Continuous Liquify tools keep acting while the button is held.
        if matches!(self.drag, Drag::Liquify { .. }) && self.liquify_mode.continuous() {
            if let Some((ix, iy)) = self.mouse_img {
                self.liquify_tick(ix, iy, true);
            }
            window.request_animation_frame();
        }
        let extra = self.caf.as_ref().map(|c| &c.overlay).or(self.select_mask.as_ref().map(|s| &s.overlay)).or(self.quick_mask.as_ref().map(|q| &q.overlay)).or(self.liquify_overlay.as_ref()).or(self.float.as_ref());
        let plan = self.cache.prepare(&mut self.doc, extra, &self.vp, bounds, self.sf, window);
        if plan.pending > 0 {
            window.request_animation_frame();
        }
        let mut overlays = self.overlays();
        self.retouch_overlays(&mut overlays);
        self.gradient_overlays(&mut overlays);
        self.vector_overlays(&mut overlays);
        self.nav_overlays(&mut overlays);
        self.vp_overlays(&mut overlays);
        self.pw_overlays(&mut overlays);
        self.object_overlays(&mut overlays);
        self.type_overlays(&mut overlays);
        let cursor = match (self.space_hand, self.tool, &self.drag) {
            (_, _, Drag::Pan { .. }) => CursorStyle::ClosedHand,
            (true, _, _) | (_, Tool::Hand, _) => CursorStyle::OpenHand,
            (_, Tool::Move, _) => CursorStyle::Arrow,
            (_, Tool::Zoom, _) => CursorStyle::PointingHand,
            _ => CursorStyle::Crosshair,
        };
        CanvasFrame { plan, overlays, cursor, rot: self.vp.rot }
    }

    fn screen_pts(&self, pts: &[[f32; 2]]) -> Vec<Point<Pixels>> {
        pts.iter().map(|p| self.vp.to_screen(p[0], p[1], self.canvas, self.sf)).collect()
    }

    fn ants(&self, out: &mut Vec<Overlay>, polys: Vec<(Vec<Point<Pixels>>, bool)>) {
        let black = gpui_kit::hsla(0., 0., 0., 1.);
        let white = gpui_kit::hsla(0., 0., 1., 1.);
        out.push(Overlay::Path { polys: polys.clone(), dashed: false, color: black, width: 1.0 });
        out.push(Overlay::Path { polys, dashed: true, color: white, width: 1.0 });
    }

    fn rect_pts(r: (f32, f32, f32, f32)) -> Vec<[f32; 2]> {
        vec![[r.0, r.1], [r.2, r.1], [r.2, r.3], [r.0, r.3]]
    }

    fn overlays(&self) -> Vec<Overlay> {
        let mut out = Vec::new();
        let s = self.vp.logical_scale(self.sf);
        if let Some(sel) = &self.doc.selection {
            let key = sel.cache_key();
            let mut cache = self.outline_cache.borrow_mut();
            let fresh = !matches!(&*cache, Some((k, _)) if *k == key && key.0 != 0);
            if fresh {
                *cache = Some((key, sel.outlines()));
            }
            let polys = cache.as_ref().unwrap().1.iter().map(|p| (self.screen_pts(p), p.len() > 2)).collect();
            drop(cache);
            self.ants(&mut out, polys);
        }
        match &self.drag {
            Drag::Marquee { start, cur, ellipse, center, .. } => {
                let r = self.drag_rect(*start, *cur, *center);
                let pts = if *ellipse {
                    Selection::ellipse(r).outlines().remove(0)
                } else {
                    Self::rect_pts((r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32))
                };
                let pts = self.screen_pts(&pts);
                self.ants(&mut out, vec![(pts, true)]);
            }
            Drag::Lasso(pts, _) => {
                let pts = self.screen_pts(pts);
                self.ants(&mut out, vec![(pts, false)]);
            }
            Drag::PatchMove { start, cur } => {
                if let Some(sel) = &self.doc.selection {
                    let (dx, dy) = (cur.0 - start.0, cur.1 - start.1);
                    let polys = sel.outlines().iter().map(|p| {
                        let moved: Vec<[f32; 2]> = p.iter().map(|q| [q[0] + dx, q[1] + dy]).collect();
                        (self.screen_pts(&moved), p.len() > 2)
                    }).collect();
                    self.ants(&mut out, polys);
                }
            }
            Drag::ObjectSelect { start, cur, lasso, .. } => {
                let pts = if self.opts.object_lasso {
                    lasso.clone()
                } else {
                    Self::rect_pts((start.0.min(cur.0), start.1.min(cur.1), start.0.max(cur.0), start.1.max(cur.1)))
                };
                let pts = self.screen_pts(&pts);
                self.ants(&mut out, vec![(pts, !self.opts.object_lasso)]);
            }
            _ => {}
        }
        if let Some(st) = &self.poly {
            let mut pts = st.pts.clone();
            if let Some(m) = self.mouse_img {
                pts.push([m.0, m.1]);
            }
            let pts = self.screen_pts(&pts);
            self.ants(&mut out, vec![(pts, false)]);
        }
        if let Some(st) = &self.magnetic {
            let mut pts: Vec<[f32; 2]> = st.committed.iter().map(|q| [q.x, q.y]).collect();
            pts.extend(st.live.iter().skip(1).map(|q| [q.x, q.y]));
            let pts = self.screen_pts(&pts);
            self.ants(&mut out, vec![(pts, false)]);
            for a in &st.anchors {
                let c = self.vp.to_screen(a.x, a.y, self.canvas, self.sf);
                out.push(Overlay::Quad {
                    bounds: Bounds::from_corners(point(c.x - px(3.), c.y - px(3.)), point(c.x + px(3.), c.y + px(3.))),
                    bg: gpui_kit::hsla(0., 0., 1., 1.),
                    border: gpui_kit::hsla(0., 0., 0., 1.),
                    border_w: 1.0,
                    radius: 0.0,
                });
            }
        }
        // Brush footprint (also the Content-Aware Fill sampling brush).
        if (self.tool.is_brush() || self.caf.is_some()) && !self.space_hand {
            if let (Some(m), Some(_)) = (self.mouse, self.mouse_img) {
                if self.canvas.contains(&m) {
                    let size = self.caf.as_ref().map_or(self.brush_ref().size, |c| c.brush);
                    let r = (size * 0.5 * s).max(1.5);
                    let b = Bounds::from_corners(point(m.x - px(r), m.y - px(r)), point(m.x + px(r), m.y + px(r)));
                    let b2 = Bounds::from_corners(point(m.x - px(r + 1.), m.y - px(r + 1.)), point(m.x + px(r + 1.), m.y + px(r + 1.)));
                    let clear = gpui_kit::hsla(0., 0., 0., 0.);
                    out.push(Overlay::Quad { bounds: b2, bg: clear, border: gpui_kit::hsla(0., 0., 0., 0.6), border_w: 1.0, radius: r + 1.0 });
                    out.push(Overlay::Quad { bounds: b, bg: clear, border: gpui_kit::hsla(0., 0., 1., 0.9), border_w: 1.0, radius: r });
                }
            }
        }
        out
    }

    // ----- pointer input -----------------------------------------------------------

    fn img_point(&self, p: Point<Pixels>) -> (f32, f32) {
        self.vp.to_image(p, self.canvas, self.sf)
    }

    fn marquee_rect(&self, a: (f32, f32), b: (f32, f32)) -> RectI {
        let r = RectI::new(
            a.0.min(b.0).round() as i32,
            a.1.min(b.1).round() as i32,
            a.0.max(b.0).round() as i32,
            a.1.max(b.1).round() as i32,
        );
        r.intersect(&self.doc.bounds().inflate(1_000_000))
    }

    /// Marquee rect; with `center` the start point is the middle (Alt).
    fn drag_rect(&self, a: (f32, f32), b: (f32, f32), center: bool) -> RectI {
        if center {
            let (dx, dy) = ((b.0 - a.0).abs(), (b.1 - a.1).abs());
            self.marquee_rect((a.0 - dx, a.1 - dy), (a.0 + dx, a.1 + dy))
        } else {
            self.marquee_rect(a, b)
        }
    }

    pub(crate) fn input_blocked(&self) -> bool {
        self.busy.is_some() || self.dialog.is_some()
    }

    pub fn on_mouse_down(&mut self, ev: &MouseDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        self.focus.focus(window, cx);
        if self.input_blocked() {
            return;
        }
        let p = ev.position;
        let (ix, iy) = self.img_point(p);
        if ev.button == MouseButton::Middle || (ev.button == MouseButton::Left && (self.space_hand || self.tool == Tool::Hand)) {
            self.drag = Drag::Pan { start: p, pan0: self.vp.pan };
            cx.notify();
            return;
        }
        if ev.button != MouseButton::Left {
            return;
        }
        let alt = ev.modifiers.alt;
        let shift = ev.modifiers.shift;
        if self.vp_active() {
            self.vp_down((ix, iy), alt, ev.modifiers.platform);
            self.drag = Drag::Vp;
            cx.notify();
            return;
        }
        if self.pw_active() {
            self.pw_down((ix, iy));
            self.drag = Drag::Pw;
            cx.notify();
            return;
        }
        if self.caf_down(ix, iy, alt) {
            cx.notify();
            return;
        }
        if self.tool.is_selection() {
            self.selection_down(ix, iy, ev, window, cx);
            cx.notify();
            return;
        }
        if self.retouch_down(ix, iy, ev, window, cx) || self.paint_down(ix, iy, ev, window, cx) || self.type_down(ix, iy, ev, window, cx) {
            cx.notify();
            return;
        }
        if self.nav_down(ix, iy, ev, window, cx) {
            cx.notify();
            return;
        }
        if self.vector_down(ix, iy, ev, window, cx) {
            if self.vdrag.is_some() {
                self.drag = Drag::Vector;
            }
            cx.notify();
            return;
        }
        match self.tool {
            Tool::Brush | Tool::Eraser => {
                if alt && self.tool == Tool::Brush {
                    self.pick_color(ix, iy, window, cx);
                } else {
                    self.begin_stroke(ix, iy, shift);
                }
            }
            Tool::Liquify => {
                let li = self.doc.active;
                if self.doc.layers[li].visible {
                    let (id, ver) = (self.doc.layers[li].id, self.doc.layers[li].version);
                    let stale = self.liquify_session.as_ref().is_none_or(|s| s.layer_id != id || s.version != ver);
                    if stale {
                        self.liquify_session = Some(crate::engine::liquify::Session::new(id, ver, &self.doc.layers[li].grid));
                        self.liquify_overlay = None;
                    }
                    use crate::engine::liquify::Mode;
                    if !matches!(self.liquify_mode, Mode::Freeze | Mode::Thaw) {
                        self.doc.push_undo(self.liquify_mode.label());
                    }
                    if let Some(s) = &mut self.liquify_session {
                        s.begin_stroke(ix, iy);
                    }
                    self.drag = Drag::Liquify { layer: li };
                    self.liquify_tick(ix, iy, true);
                } else {
                    self.flash("Cannot liquify a hidden layer");
                }
            }
            Tool::Hand => {}
            t => self.flash(format!("{} is not available yet", t.name())),
        }
        cx.notify();
    }

    fn begin_stroke(&mut self, x: f32, y: f32, shift: bool) {
        use super::layers_ui::PaintTarget;
        let erase = self.tool == Tool::Eraser;
        let settings = self.stroke_settings();
        let li = self.doc.active;
        // Quick Mask and layer masks are painted as gray (eraser paints white).
        let gray = if erase { [255, 255, 255] } else { Self::quick_mask_gray(self.fg) };
        if self.quick_mask.is_some() {
            let qm = self.quick_mask.as_mut().unwrap();
            let mut stroke = Box::new(Stroke::brush(usize::MAX, &qm.grid, settings, gray, false, None));
            let rects = stroke.add_point_p(x, y, self.pressure, &mut qm.grid);
            self.quick_mask_changed(&rects);
            self.drag = Drag::Paint(stroke, PaintTarget::QuickMask);
            return;
        }
        if self.mask_edit && self.doc.layers[li].mask.is_some() {
            self.doc.push_undo("Paint Layer Mask");
            let mut g = self.doc.layers[li].mask.as_ref().unwrap().grid.to_gray();
            let mut stroke = Box::new(Stroke::brush(li, &g, settings, gray, false, self.doc.selection.clone()));
            let rects = stroke.add_point_p(x, y, self.pressure, &mut g);
            self.mask_painted(li, &g, &rects);
            self.mask_paint = Some(g);
            self.drag = Drag::Paint(stroke, PaintTarget::Mask(li));
            return;
        }
        if !self.doc.layers[li].visible {
            self.flash("Cannot paint on a hidden layer");
            return;
        }
        if !self.ensure_pixels(li) {
            return;
        }
        self.doc.push_undo(if erase { "Eraser" } else { "Brush" });
        // The Eraser paints the background colour on the locked Background.
        let (color, erase) = if erase && self.doc.layers[li].background { (self.bg, false) } else { (self.fg, erase) };
        let mut stroke = Box::new(Stroke::brush(li, &self.doc.layers[li].grid, settings, color, erase, self.doc.selection.clone()));
        let mut rects = Vec::new();
        if let (true, Some((lx, ly))) = (shift, self.last_paint_point) {
            rects.extend(stroke.add_point(lx, ly, &mut self.doc.layers[li].grid));
        }
        rects.extend(stroke.add_point_p(x, y, self.pressure, &mut self.doc.layers[li].grid));
        self.doc.layer_pixels_changed(li, &rects);
        self.drag = Drag::Paint(stroke, PaintTarget::Layer(li));
    }

    /// Build-up (airbrush): while the button is held, dabs keep landing.
    fn paint_tick(&mut self, window: &mut Window) {
        use super::layers_ui::PaintTarget;
        let Drag::Paint(stroke, target) = &mut self.drag else { return };
        if !self.brushes.get(&self.tool).is_some_and(|b| b.build_up) {
            return;
        }
        match *target {
            PaintTarget::Layer(li) => {
                let rects = stroke.tick(&mut self.doc.layers[li].grid);
                self.doc.layer_pixels_changed(li, &rects);
            }
            PaintTarget::QuickMask => {
                if let Some(qm) = &mut self.quick_mask {
                    let rects = stroke.tick(&mut qm.grid);
                    self.quick_mask_changed(&rects);
                }
            }
            PaintTarget::Mask(li) => {
                if let Some(mut g) = self.mask_paint.take() {
                    let rects = stroke.tick(&mut g);
                    self.mask_painted(li, &g, &rects);
                    self.mask_paint = Some(g);
                }
            }
        }
        window.request_animation_frame();
    }

    /// Feeds one input point to the active stroke.
    fn stroke_to(&mut self, x: f32, y: f32) {
        use super::layers_ui::PaintTarget;
        let pr = self.pressure;
        let Drag::Paint(stroke, target) = &mut self.drag else { return };
        let t0 = Instant::now();
        match *target {
            PaintTarget::QuickMask => {
                if let Some(qm) = &mut self.quick_mask {
                    let rects = stroke.add_point_p(x, y, pr, &mut qm.grid);
                    self.quick_mask_changed(&rects);
                }
            }
            PaintTarget::Mask(li) => {
                if let Some(mut g) = self.mask_paint.take() {
                    let rects = stroke.add_point_p(x, y, pr, &mut g);
                    self.mask_painted(li, &g, &rects);
                    self.mask_paint = Some(g);
                }
            }
            PaintTarget::Layer(li) => {
                let rects = stroke.add_point_p(x, y, pr, &mut self.doc.layers[li].grid);
                self.doc.layer_pixels_changed(li, &rects);
            }
        }
        self.last_tool_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// Ends the active stroke (flushes smoothing, finalizes tiles).
    fn end_stroke(&mut self, stroke: Box<Stroke>, target: super::layers_ui::PaintTarget) {
        use super::layers_ui::PaintTarget;
        self.last_paint_point = stroke.last_point();
        match target {
            PaintTarget::QuickMask => {
                if let Some(qm) = &mut self.quick_mask {
                    let rects = stroke.finish(&mut qm.grid);
                    self.quick_mask_changed(&rects);
                }
            }
            PaintTarget::Mask(li) => {
                if let Some(mut g) = self.mask_paint.take() {
                    let rects = stroke.finish(&mut g);
                    self.mask_painted(li, &g, &rects);
                }
            }
            PaintTarget::Layer(li) => {
                let rects = stroke.finish(&mut self.doc.layers[li].grid);
                self.doc.layer_pixels_changed(li, &rects);
            }
        }
    }

    pub fn on_mouse_move(&mut self, ev: &MouseMoveEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let p = ev.position;
        let inside = self.canvas.contains(&p);
        let was_inside = self.mouse.is_some();
        if inside {
            self.mouse = Some(p);
            let (ix, iy) = self.img_point(p);
            self.mouse_img = Some((ix, iy));
            let (xi, yi) = (ix.floor() as i32, iy.floor() as i32);
            self.hover_rgba = (xi >= 0 && yi >= 0 && xi < self.doc.w as i32 && yi < self.doc.h as i32)
                .then(|| px::unpremultiply(composite::sample(&self.doc.layers, xi, yi)));
        } else {
            self.mouse = None;
            self.mouse_img = None;
        }
        let (ix, iy) = self.img_point(p);
        let s = self.vp.logical_scale(self.sf);
        match &mut self.drag {
            Drag::Vector => self.vector_move(ix, iy, ev.modifiers.shift, ev.modifiers.alt),
            Drag::TypeBox { cur, .. } => *cur = (ix, iy),
            Drag::TypeSelect => self.type_select_move(ix, iy),
            &mut Drag::TypePathHandle { end } => self.type_path_handle_move(ix, iy, end),
            Drag::None => {
                if self.tool == Tool::ObjectSelect {
                    self.object_hover(ix, iy, cx);
                }
                if self.tool == Tool::ContentAwareTracing || self.tool == Tool::Pen {
                    self.vector_move(ix, iy, false, false);
                    cx.notify();
                }
                if self.magnetic.is_some() {
                    self.magnetic_hover(ix, iy);
                    cx.notify();
                } else if inside || was_inside || self.poly.is_some() {
                    cx.notify();
                }
                return;
            }
            Drag::Pan { start, pan0 } => {
                self.vp.pan = *pan0 + self.vp.unrotate_delta(p - *start);
            }
            Drag::Paint(..) => self.stroke_to(ix, iy),
            Drag::SpotHeal(_) => self.spot_move(ix, iy),
            Drag::PatchMove { cur, .. } => *cur = (ix, iy),
            Drag::QuickSelect(q) => {
                let r = self.brushes.get(&Tool::QuickSelect).map_or(20.0, |b| b.size * 0.5);
                q.qs.add_dab(ix, iy, r);
                q.dirty = true;
                self.quick_select_update(false);
            }
            Drag::ObjectSelect { cur, lasso, .. } => {
                *cur = (ix, iy);
                let last = lasso.last().copied().unwrap_or([ix, iy]);
                if ((last[0] - ix).powi(2) + (last[1] - iy).powi(2)).sqrt() * s > 2.0 {
                    lasso.push([ix, iy]);
                }
            }
            Drag::Liquify { .. } => {
                self.liquify_tick(ix, iy, false);
            }
            Drag::Marquee { start, cur, center, .. } => {
                let (mut dx, mut dy) = (ix - start.0, iy - start.1);
                if ev.modifiers.shift {
                    let m = dx.abs().max(dy.abs());
                    dx = m * dx.signum();
                    dy = m * dy.signum();
                }
                *cur = (start.0 + dx, start.1 + dy);
                *center = ev.modifiers.alt;
            }
            Drag::Nav => self.nav_move(ix, iy, p, ev.modifiers.shift, ev.modifiers.alt),
            Drag::Vp => self.vp_move((ix, iy)),
            Drag::Pw => self.pw_move((ix, iy)),
            Drag::CafPaint { last, subtract } => {
                let (a, sub) = (*last, *subtract);
                *last = (ix, iy);
                self.caf_paint(a, (ix, iy), sub);
            }
            Drag::Lasso(pts, _) => {
                let last = pts.last().copied().unwrap_or([ix, iy]);
                if ((last[0] - ix).powi(2) + (last[1] - iy).powi(2)).sqrt() * s > 2.0 {
                    pts.push([ix, iy]);
                }
            }
            Drag::Move { start, ids, .. } => {
                let off = ((ix - start.0).round() as i32, (iy - start.1).round() as i32);
                let mut changed = false;
                for l in self.doc.layers.iter_mut().filter(|l| ids.contains(&l.id)) {
                    if l.offset != off {
                        l.offset = off;
                        changed = true;
                    }
                }
                if changed {
                    self.doc.mark_all_dirty();
                }
            }
            Drag::Gradient { start, cur } => {
                *cur = (ix, iy);
                let (a, b) = (*start, *cur);
                if ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt() > 1.0 {
                    self.set_live_gradient(a, b);
                }
            }
            Drag::GradHandle(which) => {
                let which = *which;
                if let Some(gl) = &self.grad_live {
                    let (p0, p1) = if which == 0 { ((ix, iy), gl.p1) } else { (gl.p0, (ix, iy)) };
                    self.set_live_gradient(p0, p1);
                }
            }
        }
        cx.notify();
    }

    pub fn on_mouse_up(&mut self, _ev: &MouseUpEvent, window: &mut Window, cx: &mut Context<Self>) {
        let drag = std::mem::replace(&mut self.drag, Drag::None);
        match drag {
            Drag::None | Drag::Pan { .. } | Drag::TypeSelect | Drag::TypePathHandle { .. } => {}
            Drag::Liquify { .. } => {
                if let Some(s) = &mut self.liquify_session {
                    s.end_stroke();
                }
            }
            Drag::Paint(stroke, target) => {
                self.end_stroke(stroke, target);
                self.after_paint_stroke();
            }
            Drag::Marquee { start, cur, ellipse, center, op } => {
                let r = self.drag_rect(start, cur, center);
                if r.w() < 1 || r.h() < 1 {
                    if op == SelOp::New && self.doc.selection.is_some() {
                        self.doc.push_undo("Deselect");
                        self.doc.selection = None;
                    }
                } else {
                    let sel = if ellipse { Selection::ellipse(r) } else { Selection::rect(r) };
                    self.commit_selection(if ellipse { "Elliptical Marquee" } else { "Rectangular Marquee" }, Some(sel), op, ellipse);
                }
            }
            Drag::Lasso(pts, op) => {
                let sel = (pts.len() >= 3).then(|| Selection { shape: Shape::Poly(Arc::new(pts)), inverted: false });
                self.commit_selection("Lasso", sel, op, true);
            }
            Drag::QuickSelect(q) => self.quick_select_finish(q, cx),
            Drag::Vector => self.vector_up(cx),
            Drag::TypeBox { start, cur } => self.type_box_done(start, cur),
            Drag::SpotHeal(st) => self.spot_finish(st, cx),
            Drag::PatchMove { start, cur } => self.patch_finish(start, cur, cx),
            Drag::ObjectSelect { start, cur, lasso, op } => self.object_select_finish(start, cur, lasso, op, cx),
            Drag::Move { layer, ids, .. } => {
                let off = self.doc.layers[layer].offset;
                if off != (0, 0) {
                    // Every moved layer goes through the layer transform (vector
                    // descriptions, masks and artboard rects move with the pixels).
                    use rayon::prelude::*;
                    let moving: Vec<Layer> = self.doc.layers.iter().filter(|l| ids.contains(&l.id)).cloned().collect();
                    let (w, h) = (self.doc.w, self.doc.h);
                    let single = moving.len() == 1;
                    self.spawn_job("Move", cx, move |_| {
                        let m = [1.0, 0.0, off.0 as f32, 0.0, 1.0, off.1 as f32, 0.0, 0.0, 1.0];
                        let moved: Vec<Layer> = moving
                            .par_iter()
                            .map(|l| {
                                let mut l = l.clone();
                                l.offset = (0, 0);
                                crate::engine::xform::layer_transformed(&l, &m, w, h)
                            })
                            .collect();
                        Box::new(move |ed: &mut Editor, _| {
                            ed.doc.push_undo("Move");
                            for nl in moved {
                                if let Some(li) = ed.doc.layers.iter().position(|l| l.id == nl.id) {
                                    ed.doc.layers[li] = nl;
                                    ed.doc.layers[li].version += 1;
                                }
                            }
                            if single {
                                if let Some(sel) = &mut ed.doc.selection {
                                    sel.translate(off.0 as f32, off.1 as f32);
                                }
                            }
                            ed.doc.mark_all_dirty();
                        })
                    });
                }
            }
            Drag::Gradient { start, cur } => {
                // The gradient stays live (handles + preview) until applied.
                if ((cur.0 - start.0).powi(2) + (cur.1 - start.1).powi(2)).sqrt() > 1.0 {
                    self.set_live_gradient(start, cur);
                    self.flash("Drag the handles to adjust · Enter applies · Esc cancels");
                }
            }
            Drag::GradHandle(_) => {}
            Drag::Nav => self.nav_up(window, cx),
            Drag::CafPaint { .. } => self.caf_changed(cx),
            Drag::Vp => self.vp_up(),
            Drag::Pw => self.pw_up(cx),
        }
        cx.notify();
    }

    pub fn on_scroll(&mut self, ev: &ScrollWheelEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let d = match ev.delta {
            ScrollDelta::Pixels(p) => point(f32::from(p.x), f32::from(p.y)),
            ScrollDelta::Lines(l) => point(l.x * 40.0, l.y * 40.0),
        };
        if ev.modifiers.platform || ev.modifiers.control || ev.modifiers.alt {
            let z = self.vp.zoom * (d.y * 0.004).exp();
            self.vp.zoom_about(z, ev.position, self.canvas);
        } else {
            self.vp.pan = self.vp.pan + self.vp.unrotate_delta(point(px(d.x), px(d.y)));
        }
        cx.notify();
    }

    pub fn on_pinch(&mut self, ev: &PinchEvent, _window: &mut Window, cx: &mut Context<Self>) {
        let z = self.vp.zoom * (1.0 + ev.delta);
        self.vp.zoom_about(z, ev.position, self.canvas);
        cx.notify();
    }

    pub fn on_key_down(&mut self, ev: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        if window.has_focused_input(cx) {
            return;
        }
        if self.text_key(ev, cx) {
            return;
        }
        let ks = &ev.keystroke;
        let m = &ks.modifiers;
        if m.platform || m.control {
            return;
        }
        let key = ks.key.as_str();
        if self.vp_active() && self.vp_key(key, cx) {
            return;
        }
        if self.pw_active() && self.pw_key(key, cx) {
            return;
        }
        if self.caf.is_some() {
            match key {
                "escape" => self.caf_cancel(cx),
                "enter" => self.caf_apply(cx),
                "[" | "]" => {
                    if let Some(c) = &mut self.caf {
                        c.brush = (c.brush * if key == "]" { 1.2 } else { 1.0 / 1.2 }).clamp(4.0, 4000.0).round();
                    }
                    cx.notify();
                }
                _ => {}
            }
            return;
        }
        if self.dialog.is_some() {
            match key {
                "escape" => self.close_dialog(false, window, cx),
                "enter" => self.close_dialog(true, window, cx),
                _ => {}
            }
            return;
        }
        if self.busy.is_some() {
            return;
        }
        if key == "q" {
            self.toggle_quick_mask(cx);
            return;
        }
        // Crop: O cycles the overlay, Shift+O its orientation.
        if key == "o" && self.tool == Tool::Crop {
            self.cycle_crop_overlay(m.shift);
            cx.notify();
            return;
        }
        if key == "f" {
            self.cycle_screen_mode(window, cx);
            return;
        }
        let mem = self.tool_memory.clone();
        if let Some(t) = super::tools::tool_for_key(key, m.shift, self.tool, &|gi| mem[gi]) {
            self.set_tool(t, window, cx);
            return;
        }
        match key {
            "space" => {
                if !self.space_hand {
                    self.space_hand = true;
                    cx.notify();
                }
            }
            "[" | "]" => {
                let up = key == "]";
                let b = self.brush_mut();
                if m.shift {
                    b.hardness = (b.hardness + if up { 0.25 } else { -0.25 }).clamp(0.0, 1.0);
                } else {
                    b.size = (b.size * if up { 1.2 } else { 1.0 / 1.2 }).clamp(1.0, 2500.0).round();
                }
                self.sync_brush_sliders(window, cx);
                cx.notify();
            }
            "x" => {
                let (f, b) = (self.fg, self.bg);
                self.set_fg(b, window, cx);
                self.set_bg(f, window, cx);
                cx.notify();
            }
            "d" => {
                self.set_fg([0, 0, 0], window, cx);
                self.set_bg([255, 255, 255], window, cx);
                cx.notify();
            }
            "escape" => {
                self.poly = None;
                self.magnetic = None;
                self.cancel_gradient();
                self.end_pen(cx);
                if let Drag::Move { ids, .. } = &self.drag {
                    for l in self.doc.layers.iter_mut().filter(|l| ids.contains(&l.id)) {
                        l.offset = (0, 0);
                    }
                    self.doc.mark_all_dirty();
                }
                self.drag = Drag::None;
                self.crop = None;
                self.nopts.crop_angle = 0.0;
                self.nopts.pcrop.clear();
                cx.notify();
            }
            "enter" => {
                if self.pen_sub.is_some() {
                    self.end_pen(cx);
                } else if self.grad_live.is_some() {
                    self.commit_gradient(cx);
                } else if self.poly.is_some() {
                    self.finish_poly();
                } else if self.magnetic.is_some() {
                    self.finish_magnetic();
                } else if self.crop.is_some() {
                    self.commit_crop(cx);
                } else if self.nopts.pcrop.len() == 4 {
                    self.commit_perspective_crop(cx);
                }
                cx.notify();
            }
            d if d.len() == 1 && d.as_bytes()[0].is_ascii_digit() => {
                let n = (d.as_bytes()[0] - b'0') as f32;
                let v = if n == 0.0 { 1.0 } else { n / 10.0 };
                match self.tool {
                    t if t.is_brush() => self.brush_mut().opacity = v,
                    Tool::Bucket | Tool::Gradient => self.fill_opacity = v,
                    _ => {}
                }
                cx.notify();
            }
            _ => {}
        }
    }

    pub fn on_key_up(&mut self, ev: &KeyUpEvent, _window: &mut Window, cx: &mut Context<Self>) {
        if ev.keystroke.key == "space" && self.space_hand {
            self.space_hand = false;
            cx.notify();
        }
    }

    // ----- tools ---------------------------------------------------------------------

    fn pick_color(&mut self, x: f32, y: f32, window: &mut Window, cx: &mut Context<Self>) {
        let (xi, yi) = (x.floor() as i32, y.floor() as i32);
        if xi < 0 || yi < 0 || xi >= self.doc.w as i32 || yi >= self.doc.h as i32 {
            return;
        }
        let [r, g, b, a] = px::unpremultiply(composite::sample(&self.doc.layers, xi, yi));
        if a > 0 {
            self.set_fg([r, g, b], window, cx);
            self.flash(format!("Picked #{r:02X}{g:02X}{b:02X}"));
        }
    }

    // ----- background jobs -----------------------------------------------------------

    pub fn spawn_job<F>(&mut self, label: impl Into<SharedString>, cx: &mut Context<Self>, work: F)
    where
        F: FnOnce(&Progress) -> Box<dyn FnOnce(&mut Editor, &mut Context<Editor>) + Send> + Send + 'static,
    {
        let progress = Arc::new(Progress::default());
        let label: SharedString = label.into();
        self.busy = Some(Busy { label: label.clone(), progress: progress.clone(), started: Instant::now() });
        // The result belongs to the document the job started on.
        let uid = self.docs.uid;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let started = Instant::now();
            let apply = cx.background_executor().spawn(async move { work(&progress) }).await;
            let _ = this.update(cx, |ed, cx| {
                ed.with_doc(uid, |ed| apply(ed, cx));
                ed.busy = None;
                let ms = started.elapsed().as_secs_f32() * 1000.0;
                ed.last_job = Some((label.clone(), ms));
                if !ed.status.starts_with("Error") {
                    ed.flash(format!("{label} — {ms:.0} ms"));
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Runs `op` on a snapshot of the active layer off the UI thread, then
    /// installs the result as one undo step.
    pub fn run_layer_op<F>(&mut self, label: &'static str, cx: &mut Context<Self>, op: F)
    where
        F: FnOnce(&TileGrid, Option<&Selection>, &Progress) -> TileGrid + Send + 'static,
    {
        let li = self.doc.active;
        let id = self.doc.layers[li].id;
        let grid = self.doc.layers[li].grid.clone();
        let sel = self.doc.selection.clone();
        self.spawn_job(label, cx, move |p| {
            let out = op(&grid, sel.as_ref(), p);
            Box::new(move |ed: &mut Editor, _| {
                if let Some(li) = ed.doc.layers.iter().position(|l| l.id == id) {
                    ed.doc.push_undo(label);
                    ed.doc.set_layer_grid(li, out);
                }
                ed.doc.preview = None;
                ed.doc.mark_all_dirty();
            })
        });
    }

    /// Canvas-level operations (flip, rotate 90°, Image Size, Super Zoom):
    /// `m` moves masks, vector layers, hidden pixels, paths and annotations;
    /// `op` computes plain pixel layers exactly.
    pub fn run_canvas_op<F>(&mut self, label: &'static str, m: [f32; 9], w: u32, h: u32, cx: &mut Context<Self>, op: F)
    where
        F: Fn(&Layer) -> TileGrid + Send + Sync + 'static,
    {
        self.crop = None;
        self.canvas_xform_with(label, m, w, h, Default::default(), Some(std::sync::Arc::new(op)), false, cx);
    }

    fn crop_to(&mut self, r: RectI, cx: &mut Context<Self>) {
        let r = r.intersect(&self.doc.bounds());
        if r.is_empty() {
            return;
        }
        let m = [1.0, 0.0, -(r.x0 as f32), 0.0, 1.0, -(r.y0 as f32), 0.0, 0.0, 1.0];
        self.canvas_xform("Crop", m, r.w() as u32, r.h() as u32, cx);
    }

    pub fn replace_document(&mut self, doc: Document) {
        self.doc = doc;
        self.last_job = None;
        self.status = SharedString::default();
        crate::engine::gpu::clear_last_op();
        if self.canvas.size.width > px(10.) {
            self.vp.fit(self.doc.w, self.doc.h, self.canvas, self.sf);
        }
        self.cache.clear();
        for (_, (_, img)) in self.thumbs.drain() {
            self.cache.queue_drop(img);
        }
        self.needs_fit = true;
        self.drag = Drag::None;
        self.crop = None;
        self.dialog = None;
        self.last_paint_point = None;
        // Per-document navigation state belongs to the old document.
        let n = &mut self.nopts;
        (n.ruler, n.ruler2, n.crop_angle, n.slice_sel, n.artboard_sel, n.note_sel, n.note_input) = (None, None, 0.0, None, None, None, None);
        n.pcrop.clear();
        self.dockst.log_sel = None;
    }

    pub fn open_path(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        self.spawn_job(format!("Opening {name}"), cx, move |_| {
            let result = if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("pforge")) {
                io::load_native(&path).map(|(w, h, a, layers, x)| io::native_to_document(w, h, a, layers, x))
            } else if crate::engine::psd::mapping::is_psd(&path) {
                crate::engine::psd::mapping::open(&path)
            } else {
                io::open_image(&path).map(|g| Document::from_grid(g, "Background"))
            };
            Box::new(move |ed: &mut Editor, _| match result {
                Ok(mut doc) => {
                    doc.path = Some(path);
                    ed.open_document(doc);
                }
                Err(e) => ed.flash(format!("Error: {e:#}")),
            })
        });
    }

    fn save_to(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        let layers = self.doc.layers.clone();
        let extras = io::DocExtras { paths: self.doc.paths.clone(), ann: self.doc.ann.clone(), selected: self.doc.selected.clone() };
        let (w, h, active) = (self.doc.w, self.doc.h, self.doc.active);
        let native = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("pforge"));
        let psd = crate::engine::psd::mapping::is_psd(&path);
        let backing = self.doc.psd.clone();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        self.spawn_job(format!("Saving {name}"), cx, move |_| {
            let result = if native && backing.is_some() {
                Err(anyhow::anyhow!("Save this imported document as PSD/PSB to preserve its original Photoshop data"))
            } else if native {
                io::save_native(&layers, w, h, active, &extras, &path)
            } else if psd {
                crate::engine::psd::mapping::save(&layers, w, h, active, &extras, backing.as_deref(), &path)
            } else {
                io::export(&layers, w, h, &path)
            };
            Box::new(move |ed: &mut Editor, _| match result {
                Ok(()) => {
                    if native || psd {
                        ed.doc.path = Some(path);
                        ed.doc.modified = false;
                    }
                }
                Err(e) => ed.flash(format!("Error: {e:#}")),
            })
        });
    }

    /// The pixels last copied (Edit ▸ Copy / Cut).
    pub fn clipboard_grid(&self) -> Option<&TileGrid> {
        self.clipboard.as_ref()
    }

    pub fn default_dir(&self) -> PathBuf {
        self.doc
            .path
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Pictures")))
            .unwrap_or_else(|| PathBuf::from("."))
    }

    // ----- actions -----------------------------------------------------------------------

    pub fn act_open(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: false,
            prompt: Some("Open".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await {
                if let Some(p) = paths.into_iter().next() {
                    let _ = this.update(cx, |ed, cx| ed.open_path(p, cx));
                }
            }
        })
        .detach();
    }

    pub fn act_save(&mut self, save_as: bool, cx: &mut Context<Self>) {
        if !save_as {
            if let Some(p) = self.doc.path.clone().filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("pforge")) || crate::engine::psd::mapping::is_psd(p)) {
                self.save_to(p, cx);
                return;
            }
        }
        let stem = self
            .doc
            .path
            .as_deref()
            .and_then(Path::file_stem)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled".into());
        let extension = self.doc.path.as_deref().filter(|p| crate::engine::psd::mapping::is_psd(p))
            .and_then(Path::extension).and_then(|s| s.to_str()).unwrap_or("pforge");
        let rx = cx.prompt_for_new_path(&self.default_dir(), Some(&format!("{stem}.{extension}")));
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(path))) = rx.await {
                let _ = this.update(cx, |ed, cx| ed.save_to(path, cx));
            }
        })
        .detach();
    }

    pub fn act_benchmark(&mut self, cx: &mut Context<Self>) {
        self.spawn_job("Generating 16384 × 16384 (268 MP)", cx, |_| {
            let g = ops::procedural(16384, 16384);
            let mut doc = Document::from_grid(g, "Procedural");
            let id = doc.next_layer_id();
            let mut paint = Layer::new(id, "Paint", TileGrid::new(16384, 16384));
            paint.blend = BlendMode::Normal;
            doc.layers.push(paint);
            doc.active = 1;
            Box::new(move |ed: &mut Editor, _| ed.replace_document(doc))
        });
    }

    pub fn act_undo(&mut self, redo: bool, window: &mut Window, cx: &mut Context<Self>) {
        if self.busy.is_some() || !matches!(self.drag, Drag::None) {
            return;
        }
        let dims = (self.doc.w, self.doc.h);
        let label = if redo { self.doc.redo() } else { self.doc.undo() };
        // Undoing or redoing a canvas-size change (Image Size, Canvas Size,
        // crop) refits the view, as opening a document does.
        if (self.doc.w, self.doc.h) != dims && self.canvas.size.width > px(10.) {
            self.vp.fit(self.doc.w, self.doc.h, self.canvas, self.sf);
        }
        if let Some(l) = label {
            self.flash(format!("{} {l}", if redo { "Redo" } else { "Undo" }));
            self.crop = None;
            self.sync_layer_controls(window, cx);
        }
        cx.notify();
    }

    pub fn jump_history(&mut self, undo_steps: isize, window: &mut Window, cx: &mut Context<Self>) {
        let dims = (self.doc.w, self.doc.h);
        for _ in 0..undo_steps.unsigned_abs() {
            if undo_steps > 0 { self.doc.undo() } else { self.doc.redo() };
        }
        if (self.doc.w, self.doc.h) != dims && self.canvas.size.width > px(10.) {
            self.vp.fit(self.doc.w, self.doc.h, self.canvas, self.sf);
        }
        self.sync_layer_controls(window, cx);
        cx.notify();
    }

    pub fn act_copy(&mut self, cut: bool, cx: &mut Context<Self>) {
        let li = self.doc.active;
        let sel = self.doc.selection.clone();
        let copied = ops::copy_masked(&self.doc.layers[li].grid, sel.as_ref());
        self.clipboard = Some(copied);
        if cut {
            self.run_layer_op("Cut", cx, |g, sel, _| ops::clear(g, sel));
        } else {
            self.flash("Copied");
        }
    }

    pub fn act_paste(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(clip) = self.clipboard.clone() else { return };
        let grid = if (clip.w, clip.h) == (self.doc.w, self.doc.h) {
            clip
        } else {
            ops::crop(&clip, RectI::new(0, 0, self.doc.w as i32, self.doc.h as i32))
        };
        self.doc.push_undo("Paste");
        self.doc.insert_layer("Pasted Layer", grid);
        self.sync_layer_controls(window, cx);
        cx.notify();
    }

    pub fn act_fill(&mut self, background: bool, cx: &mut Context<Self>) {
        let rgb = if background { self.bg } else { self.fg };
        self.run_layer_op("Fill", cx, move |g, sel, _| ops::fill(g, sel, rgb, 1.0));
    }

    pub fn act_clear(&mut self, cx: &mut Context<Self>) {
        // Backspace is bound to Clear; while typing it deletes text.
        if self.text_edit.is_some() {
            let ev = KeyDownEvent { keystroke: gpui_kit::Keystroke::parse("backspace").unwrap(), is_held: false, prefer_character_input: false };
            self.text_key(&ev, cx);
            return;
        }
        // Delete on the locked Background fills with the background colour.
        if self.doc.layers.get(self.doc.active).is_some_and(|l| l.background) {
            let bg = self.bg;
            self.run_layer_op("Clear", cx, move |g, sel, _| ops::fill(g, sel, bg, 1.0));
            return;
        }
        self.run_layer_op("Clear", cx, |g, sel, _| ops::clear(g, sel));
    }

    pub fn act_select(&mut self, which: u8, cx: &mut Context<Self>) {
        match which {
            0 => {
                self.doc.push_undo("Select All");
                self.doc.selection = Some(Selection::rect(self.doc.bounds()));
            }
            1 => {
                if self.doc.selection.is_some() {
                    self.doc.push_undo("Deselect");
                    self.doc.selection = None;
                }
            }
            _ => {
                if let Some(s) = &self.doc.selection {
                    let mut s = s.clone();
                    s.inverted = !s.inverted;
                    self.doc.push_undo("Inverse");
                    self.doc.selection = Some(s);
                }
            }
        }
        cx.notify();
    }

    pub fn act_zoom(&mut self, how: u8, cx: &mut Context<Self>) {
        let center = self.canvas.center();
        match how {
            0 => self.vp.zoom_about(self.vp.zoom * 1.5, center, self.canvas),
            1 => self.vp.zoom_about(self.vp.zoom / 1.5, center, self.canvas),
            2 => self.vp.fit(self.doc.w, self.doc.h, self.canvas, self.sf),
            _ => self.vp.center_at_zoom(1.0, self.doc.w, self.doc.h, self.canvas, self.sf),
        }
        cx.notify();
    }

    pub fn act_layer(&mut self, what: &str, window: &mut Window, cx: &mut Context<Self>) {
        let li = self.doc.active;
        match what {
            "new" => {
                self.doc.push_undo("New Layer");
                let name = self.doc.new_layer_name();
                self.doc.insert_layer(name, TileGrid::new(self.doc.w, self.doc.h));
            }
            "duplicate" => {
                self.doc.push_undo("Duplicate Layer");
                let l = &self.doc.layers[li];
                let (name, grid, op, blend) = (format!("{} copy", l.name), l.grid.clone(), l.opacity, l.blend);
                let at = self.doc.insert_layer(name, grid);
                self.doc.layers[at].opacity = op;
                self.doc.layers[at].blend = blend;
            }
            "delete" => {
                if self.doc.layers.len() > 1 {
                    self.doc.push_undo("Delete Layer");
                    self.doc.delete_layer(li);
                }
            }
            "up" | "down" => {
                self.doc.push_undo("Reorder Layer");
                if !self.doc.move_layer(li, what == "up") {
                    self.doc.undo();
                }
            }
            "via_copy" | "via_cut" => {
                let sel = self.doc.selection.clone();
                let copied = ops::copy_masked(&self.doc.layers[li].grid, sel.as_ref());
                self.doc.push_undo(if what == "via_cut" { "Layer via Cut" } else { "Layer via Copy" });
                if what == "via_cut" {
                    let cleared = if self.doc.layers[li].background { ops::fill(&self.doc.layers[li].grid, sel.as_ref(), self.bg, 1.0) } else { ops::clear(&self.doc.layers[li].grid, sel.as_ref()) };
                    self.doc.set_layer_grid(li, cleared);
                }
                let name = self.doc.new_layer_name();
                self.doc.insert_layer(name, copied);
            }
            "merge" => {
                if li == 0 {
                    return;
                }
                if self.doc.layers[li - 1..=li].iter().any(|layer| layer.advanced_blend.as_ref().is_some_and(|settings| !settings.is_identity())) {
                    self.flash("Merge Down cannot yet preserve Blend If or channel restrictions. The layers are unchanged.");
                    cx.notify();
                    return;
                }
                let mut below = self.doc.layers[li - 1].clone();
                below.opacity = 1.0;
                below.blend = BlendMode::Normal;
                below.visible = true;
                let top = self.doc.layers[li].clone();
                let (w, h) = (self.doc.w, self.doc.h);
                let (below_id, top_id) = (below.id, top.id);
                self.spawn_job("Merge Down", cx, move |_| {
                    let merged = composite::flatten_layers(&[below, top], w, h);
                    Box::new(move |ed: &mut Editor, _| {
                        let bi = ed.doc.layers.iter().position(|l| l.id == below_id);
                        let ti = ed.doc.layers.iter().position(|l| l.id == top_id);
                        if let (Some(bi), Some(ti)) = (bi, ti) {
                            ed.doc.push_undo("Merge Down");
                            ed.doc.set_layer_grid(bi, merged);
                            ed.doc.delete_layer(ti);
                            ed.doc.active = bi;
                        }
                    })
                });
            }
            "flatten" => {
                let layers = self.doc.layers.clone();
                let (w, h) = (self.doc.w, self.doc.h);
                self.spawn_job("Flatten Image", cx, move |_| {
                    let flat = composite::flatten_layers(&layers, w, h);
                    let layer = Layer::new(0, "Background", flat);
                    Box::new(move |ed: &mut Editor, _| {
                        ed.doc.push_undo("Flatten Image");
                        let id = ed.doc.next_layer_id();
                        let mut layer = layer;
                        layer.id = id;
                        ed.doc.layers = vec![layer];
                        ed.doc.active = 0;
                        ed.doc.mark_all_dirty();
                    })
                });
            }
            _ => {}
        }
        self.sync_layer_controls(window, cx);
        cx.notify();
    }

    pub fn toggle_visibility(&mut self, li: usize, cx: &mut Context<Self>) {
        self.doc.push_undo("Layer Visibility");
        let l = &mut self.doc.layers[li];
        l.visible = !l.visible;
        self.doc.maintain_history();
        self.doc.mark_all_dirty();
        cx.notify();
    }

    pub fn act_canvas(&mut self, what: &str, cx: &mut Context<Self>) {
        let (wf, hf) = (self.doc.w as f32, self.doc.h as f32);
        match what {
            "flip_h" => self.run_canvas_op("Flip Horizontal", [-1.0, 0.0, wf, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], self.doc.w, self.doc.h, cx, |l| ops::flip_h(&l.grid)),
            "flip_v" => self.run_canvas_op("Flip Vertical", [1.0, 0.0, 0.0, 0.0, -1.0, hf, 0.0, 0.0, 1.0], self.doc.w, self.doc.h, cx, |l| ops::flip_v(&l.grid)),
            "rot_cw" => self.run_canvas_op("Rotate 90° CW", [0.0, -1.0, hf, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], self.doc.h, self.doc.w, cx, |l| ops::rotate_cw(&l.grid)),
            "rot_ccw" => self.run_canvas_op("Rotate 90° CCW", [0.0, 1.0, 0.0, -1.0, 0.0, wf, 0.0, 0.0, 1.0], self.doc.h, self.doc.w, cx, |l| ops::rotate_ccw(&l.grid)),
            "reveal_all" => self.reveal_all(cx),
            "crop_sel" => {
                if let Some(s) = &self.doc.selection {
                    let r = s.shape_bounds();
                    self.crop_to(r, cx);
                }
            }
            _ => {}
        }
    }

    pub fn act_quick_filter(&mut self, f: Filter, cx: &mut Context<Self>) {
        let name = f.name();
        self.run_layer_op(name, cx, move |g, sel, p| ops::apply_filter(g, &f, sel, p));
    }

    // ----- dialogs -----------------------------------------------------------------------

    fn build_ctx<'a>(&self, stats: Option<&'a crate::engine::filters::Stats>) -> crate::engine::filters::catalog::BuildCtx<'a> {
        let img = self.doc.bounds();
        let area = self.doc.selection.as_ref().map_or(img, |s| s.shape_bounds().intersect(&img));
        crate::engine::filters::catalog::BuildCtx {
            fg: self.fg,
            bg: self.bg,
            area: if area.is_empty() { img } else { area },
            stats,
            source: None,
            disp_map: self.dialog.as_ref().and_then(|d| d.disp_map.clone()),
            layer_mask: self.doc.layers.get(self.doc.active).and_then(|l| l.mask.as_ref()).map(|m| Arc::new(crate::engine::filters::blur::DepthMask::new(&m.grid, &m.mips))),
            doc_w: self.doc.w,
            doc_h: self.doc.h,
        }
    }

    /// Opens a filter: instant filters run right away; others get a dialog
    /// with a live preview.
    pub fn open_filter_dialog(&mut self, kind: FilterKind, window: &mut Window, cx: &mut Context<Self>) {
        if self.input_blocked() {
            return;
        }
        if kind == FilterKind::Liquify {
            self.set_tool(Tool::Liquify, window, cx);
            return;
        }
        if kind == FilterKind::VanishingPoint {
            self.open_vanishing_point(cx);
            return;
        }
        let entry = kind.entry();
        let stats = kind.needs_stats().then(|| Arc::new(crate::engine::filters::stats(&self.doc.layers[self.doc.active])));
        let source = (matches!(kind, FilterKind::MatchColor | FilterKind::Harmonization) && self.doc.active > 0)
            .then(|| Arc::new(crate::engine::filters::stats(&self.doc.layers[self.doc.active - 1])));
        if entry.instant() {
            let f = kind.build(&[], false, &self.build_ctx(stats.as_deref()));
            self.act_quick_filter(f, cx);
            return;
        }
        let mut sliders = Vec::new();
        let mut subs = Vec::new();
        for p in &entry.params {
            let state = slider(cx, p.min, p.max, p.step, p.default, p.log);
            subs.push(cx.subscribe_in(&state, window, |this, _, _: &SliderEvent, _, cx| {
                this.update_preview(cx);
            }));
            sliders.push(DialogSlider { label: p.label, state, decimals: p.decimals, choices: p.choices });
        }
        let flag = entry.flag.is_some_and(|f| f.1);
        self.dialog = Some(Dialog { kind: DialogKind::Filter(kind), sliders, inputs: vec![], flag, stats, source, disp_map: None, _subs: subs });
        self.update_preview(cx);
    }

    pub fn update_preview(&mut self, cx: &mut Context<Self>) {
        let Some(d) = &self.dialog else { return };
        let DialogKind::Filter(kind) = d.kind else { return };
        let vals: Vec<f32> = d.sliders.iter().map(|s| s.state.read(cx).value().start()).collect();
        let mut ctx = self.build_ctx(d.stats.as_deref());
        ctx.source = d.source.as_deref();
        let filter = kind.build(&vals, d.flag, &ctx);
        let flag = d.flag;
        if !self.adjustment_dialog_update(filter.clone(), (kind, vals, flag)) {
            self.doc.preview = Some(Preview::new(self.doc.active, filter, self.doc.selection.clone()));
        }
        self.doc.mark_all_dirty();
        cx.notify();
    }

    pub fn open_new_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.input_blocked() {
            return;
        }
        let w = cx.new(|cx| InputState::new(window, cx).default_value("3000"));
        let h = cx.new(|cx| InputState::new(window, cx).default_value("2000"));
        self.dialog = Some(Dialog { kind: DialogKind::NewDocument, sliders: vec![], inputs: vec![("Width", w), ("Height", h)], flag: true, stats: None, source: None, disp_map: None, _subs: vec![] });
        cx.notify();
    }

    pub fn open_preferences(&mut self, cx: &mut Context<Self>) {
        if self.input_blocked() {
            return;
        }
        self.dialog = Some(Dialog { kind: DialogKind::Preferences, sliders: vec![], inputs: vec![], flag: false, stats: None, source: None, disp_map: None, _subs: vec![] });
        cx.notify();
    }

    /// Use Graphics Processor: Auto / Off / Force (saved immediately).
    pub fn set_gpu_mode(&mut self, m: crate::engine::gpu::Mode, cx: &mut Context<Self>) {
        self.dockst.prefs.gpu = m;
        self.save_gpu_prefs();
        cx.notify();
    }

    /// A per-feature override (None: follow the global setting).
    pub fn set_gpu_override(&mut self, f: crate::engine::gpu::Feature, m: Option<crate::engine::gpu::Mode>, cx: &mut Context<Self>) {
        let o = &mut self.dockst.prefs.gpu_overrides;
        o.retain(|x| x.0 != f);
        if let Some(m) = m {
            o.push((f, m));
        }
        self.save_gpu_prefs();
        cx.notify();
    }

    fn save_gpu_prefs(&mut self) {
        self.dockst.prefs.apply_gpu();
        if self.dockst.prefs.save().is_err() {
            self.flash("Couldn't save the settings");
        }
        // Previews follow the new setting.
        self.doc.mark_all_dirty();
    }

    pub fn benchmark_gpu(&mut self, cx: &mut Context<Self>) {
        self.gpu_bench = Some("Benchmarking…".into());
        self.spawn_job("Benchmark GPU", cx, |_| {
            let summary = crate::engine::gpu::benchmark(&crate::bench_gpu::gpu_filters());
            Box::new(move |ed: &mut Editor, _| ed.gpu_bench = Some(summary))
        });
    }

    pub fn open_size_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.input_blocked() {
            return;
        }
        let (dw, dh) = (self.doc.w.to_string(), self.doc.h.to_string());
        let w = cx.new(|cx| InputState::new(window, cx).default_value(dw));
        let h = cx.new(|cx| InputState::new(window, cx).default_value(dh));
        // Preserve Details' Reduce Noise (shown only for those methods).
        let noise = DialogSlider { label: "Reduce Noise (%)", state: slider(cx, 0.0, 100.0, 1.0, 0.0, false), decimals: 0, choices: &[] };
        self.dialog = Some(Dialog { kind: DialogKind::ImageSize, sliders: vec![noise], inputs: vec![("Width", w), ("Height", h)], flag: true, stats: None, source: None, disp_map: None, _subs: vec![] });
        cx.notify();
    }

    pub fn open_modify_dialog(&mut self, kind: super::select_tools::Modify, _window: &mut Window, cx: &mut Context<Self>) {
        if self.input_blocked() {
            return;
        }
        if self.doc.selection.is_none() {
            self.flash("Make a selection first");
            cx.notify();
            return;
        }
        use super::select_tools::Modify;
        let (max, def) = match kind {
            Modify::Smooth => (100.0, 4.0),
            Modify::Feather => (1000.0, 10.0),
            _ => (500.0, 10.0),
        };
        let state = slider(cx, if kind == Modify::Smooth { 1.0 } else { 0.5 }, max, 0.5, def, true);
        self.dialog = Some(Dialog {
            kind: DialogKind::Modify(kind),
            sliders: vec![DialogSlider { label: kind.label(), state, decimals: 1, choices: &[] }],
            inputs: vec![],
            flag: false,
            stats: None,
            source: None,
            disp_map: None,
            _subs: vec![],
        });
        cx.notify();
    }

    /// F: standard → full screen with menus → full screen.
    pub fn cycle_screen_mode(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.screen_mode = (self.screen_mode + 1) % 3;
        let full = window.is_fullscreen();
        if (self.screen_mode == 2) != full {
            window.toggle_fullscreen();
        }
        self.flash(["Standard screen mode", "Full screen mode with menus", "Full screen mode (F to exit)"][self.screen_mode as usize]);
        cx.notify();
    }

    pub fn set_preset(&mut self, w: u32, h: u32, window: &mut Window, cx: &mut Context<Self>) {
        let Some(d) = &self.dialog else { return };
        let inputs: Vec<_> = d.inputs.iter().map(|(_, e)| e.clone()).collect();
        if inputs.len() == 2 {
            inputs[0].update(cx, |s, cx| s.set_value(w.to_string(), window, cx));
            inputs[1].update(cx, |s, cx| s.set_value(h.to_string(), window, cx));
        }
        cx.notify();
    }

    /// Displace: pick an image to use as the displacement map.
    pub fn choose_displace_map(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some("Use as Map".into()) });
        cx.spawn_in(window, async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else { return };
            let Some(path) = paths.into_iter().next() else { return };
            let loaded = cx.background_executor().spawn(async move { crate::engine::filters::extra::DispMap::load(&path) }).await;
            let _ = this.update_in(cx, |ed, w, cx| match loaded {
                Ok(m) => {
                    if let Some(d) = &mut ed.dialog {
                        d.disp_map = Some(Arc::new(m));
                        if let Some(s) = d.sliders.get(2) {
                            let st = s.state.clone();
                            st.update(cx, |s, cx| s.set_value(2.0, w, cx));
                        }
                    }
                    ed.update_preview(cx);
                }
                Err(e) => {
                    ed.flash(format!("Error: {e:#}"));
                    cx.notify();
                }
            });
        })
        .detach();
    }

    pub fn toggle_dialog_flag(&mut self, cx: &mut Context<Self>) {
        if let Some(d) = &mut self.dialog {
            d.flag = !d.flag;
        }
        self.update_preview(cx);
        cx.notify();
    }

    fn dialog_dims(&self, cx: &Context<Self>) -> Option<(u32, u32)> {
        let d = self.dialog.as_ref()?;
        let parse = |i: usize| d.inputs.get(i)?.1.read(cx).value().trim().parse::<u32>().ok();
        let (w, h) = (parse(0)?, parse(1)?);
        (w >= 1 && h >= 1 && w <= 100_000 && h <= 100_000 && (w as u64 * h as u64) <= 2_000_000_000).then_some((w, h))
    }

    pub fn close_dialog(&mut self, ok: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(d) = &self.dialog else { return };
        match (&d.kind, ok) {
            (DialogKind::Filter(FilterKind::SuperZoom), true) => {
                let vals: Vec<f32> = d.sliders.iter().map(|s| s.state.read(cx).value().start()).collect();
                let factor = vals[0].round() as u32 + 2;
                let detail = vals[1];
                let (nw, nh) = (self.doc.w * factor, self.doc.h * factor);
                self.doc.preview = None;
                if nw as u64 * nh as u64 > 2_000_000_000 {
                    self.flash("Super Zoom result would exceed 2 gigapixels");
                } else {
                    let k = factor as f32;
                    self.run_canvas_op("Super Zoom", [k, 0.0, 0.0, 0.0, k, 0.0, 0.0, 0.0, 1.0], nw, nh, cx, move |l| {
                        let g = ops::upscale_bicubic(l, nw, nh);
                        let f = crate::engine::filters::Filter::UnsharpMask { amount: detail * 2.0, radius: 0.6 * factor as f32, threshold: 2.0 };
                        ops::apply_filter(&g, &f, None, &Progress::default())
                    });
                }
            }
            (DialogKind::Filter(_), true) if self.adjust_target.is_some() => {
                self.adjust_target = None;
            }
            (DialogKind::Filter(_), false) if self.adjust_target.is_some() => {
                // Cancel removes the adjustment layer that was just created.
                self.adjust_target = None;
                self.doc.undo();
            }
            (DialogKind::Filter(_), true) => {
                if let Some(p) = self.doc.preview.clone() {
                    let label = p.filter.name();
                    let f = p.filter;
                    self.run_layer_op(label, cx, move |g, sel, prog| ops::apply_filter(g, &f, sel, prog));
                }
            }
            (DialogKind::Modify(kind), true) => {
                let v = d.sliders[0].state.read(cx).value().start();
                let kind = *kind;
                self.apply_modify(kind, v, cx);
            }
            (DialogKind::Filter(_), false) => {
                self.doc.preview = None;
                self.doc.mark_all_dirty();
            }
            (DialogKind::SelectMask, true) => {
                let vals: Vec<f32> = d.sliders.iter().map(|s| s.state.read(cx).value().start()).collect();
                self.apply_select_mask(super::select_mask::params_from(&vals), cx);
            }
            (DialogKind::SelectMask, false) => self.cancel_select_mask(),
            (DialogKind::NewDocument, true) => {
                let Some((w, h)) = self.dialog_dims(cx) else {
                    self.flash("Enter a size between 1 and 100000 px (max 2 gigapixels)");
                    cx.notify();
                    return;
                };
                let white = d.flag;
                self.open_document(Document::new(w, h, white.then_some(px::pack(255, 255, 255, 255))));
                self.flash(format!("New document {w} × {h} ({:.1} MP)", w as f64 * h as f64 / 1e6));
            }
            (DialogKind::ImageSize, true) => {
                let Some((w, h)) = self.dialog_dims(cx) else {
                    self.flash("Enter a size between 1 and 100000 px (max 2 gigapixels)");
                    cx.notify();
                    return;
                };
                if (w, h) != (self.doc.w, self.doc.h) {
                    let m = [w as f32 / self.doc.w as f32, 0.0, 0.0, 0.0, h as f32 / self.doc.h as f32, 0.0, 0.0, 0.0, 1.0];
                    let method = self.size_resample;
                    let noise = d.sliders.first().map_or(0.0, |s| s.state.read(cx).value().start());
                    self.run_canvas_op("Image Size", m, w, h, cx, move |l| ops::resize_with(l, w, h, method, noise));
                }
            }
            (DialogKind::Preferences, _) => {}
            (_, false) => {}
        }
        self.dialog = None;
        self.focus.focus(window, cx);
        cx.notify();
    }

    // ----- misc ---------------------------------------------------------------------------

    pub fn refresh_thumbs(&mut self) {
        if matches!(self.drag, Drag::Paint(..) | Drag::Liquify { .. }) {
            return;
        }
        let (w, h) = (self.doc.w, self.doc.h);
        let side = (40.0 * self.sf) as u32;
        let live: Vec<u64> = self.doc.layers.iter().map(|l| l.id).collect();
        for l in &self.doc.layers {
            let fresh = self.thumbs.get(&l.id).is_none_or(|(v, _)| *v != l.version);
            if fresh {
                let img = thumbnail(l, w, h, side);
                if let Some((_, old)) = self.thumbs.insert(l.id, (l.version, img)) {
                    self.cache.queue_drop(old);
                }
            }
        }
        let stale: Vec<u64> = self.thumbs.keys().copied().filter(|k| !live.contains(k)).collect();
        for k in stale {
            if let Some((_, img)) = self.thumbs.remove(&k) {
                self.cache.queue_drop(img);
            }
        }
    }

    pub fn memory_text(&mut self) -> SharedString {
        let s = crate::engine::memory::global().stats();
        format!("{:.0} / {:.0} MB", s.used as f64 / 1048576., s.limit as f64 / 1048576.).into()
    }
    pub fn set_memory_percent(&mut self, percent: u8, cx: &mut Context<Self>) {
        self.dockst.prefs.memory_percent = percent.clamp(10, 90);
        self.save_memory_prefs(); cx.notify();
    }
    fn save_memory_prefs(&mut self) {
        let p = &self.dockst.prefs;
        crate::engine::memory::global().configure(p.memory_percent, p.scratch_dir.clone(), p.cold_seconds);
        if p.save().is_err() { self.flash("Couldn't save the memory settings"); }
    }
    pub fn choose_scratch_directory(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files: false, directories: true, multiple: false, prompt: Some("Use for scratch".into()) });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await {
                if let Some(path) = paths.into_iter().next() {
                    let _ = this.update(cx, |ed, cx| { ed.dockst.prefs.scratch_dir = path; ed.save_memory_prefs(); cx.notify(); });
                }
            }
        }).detach();
    }
}

