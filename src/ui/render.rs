//! Rendering of the editor window: tool palette, options bar, canvas, panels.

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::color_picker::ColorPicker;
use gpui_kit::component::input::Input;
use gpui_kit::component::slider::Slider;
use gpui_kit::component::{Icon, Selectable as _, Sizable as _};
use gpui_kit::prelude::*;
use gpui_kit::{
    AnyElement, App, BorderStyle, Bounds, ContentMask, Corners, DispatchPhase, Div, Entity, ExternalPaths, Hitbox,
    HitboxBehavior, MouseDownEvent, MouseMoveEvent, MouseUpEvent, PathBuilder, PinchEvent, Pixels,
    Rgba, ScrollWheelEvent, SharedString, Stateful, Window, canvas, div, fill, hsla, point, px, quad, rgb,
};


use super::actions as a;
use super::editor::{CanvasFrame, DialogKind, Drag, Editor, Overlay, Tool};

const BG: u32 = 0x1d1d1f;
const PANEL: u32 = 0x262628;
const PANEL_HI: u32 = 0x2f2f32;
const BORDER: u32 = 0x3a3a3d;
const TEXT: u32 = 0xdcdcdc;
const DIM: u32 = 0x8e8e93;
const ACCENT: u32 = 0x2f6fde;
const PASTEBOARD: u32 = 0x161617;

pub fn section_title(t: &'static str) -> Div {
    div()
        .px_3()
        .py_1p5()
        .text_size(px(11.))
        .text_color(rgb(DIM))
        .border_b_1()
        .border_color(rgb(BORDER))
        .child(t.to_uppercase())
}

fn info_row(k: &'static str, v: impl Into<SharedString>) -> Div {
    div()
        .flex()
        .justify_between()
        .gap_2()
        .child(div().flex_none().text_color(rgb(DIM)).child(k))
        // Long values wrap, right-aligned, instead of running off the panel.
        .child(div().min_w_0().text_right().text_color(rgb(TEXT)).child(v.into()))
}

/// The document's drop shadow and display tiles.
fn paint_plan(plan: super::viewport::FramePlan, window: &mut Window) {
    if let Some(r) = plan.image_rect {
        let shadow = Bounds { origin: r.origin + point(px(0.), px(2.)), size: r.size };
        window.paint_quad(fill(shadow.dilate(px(3.)), hsla(0., 0., 0., 0.35)));
    }
    for d in plan.draws {
        d.img.paint(window, d.bounds, d.image_bounds, d.xf);
    }
}

fn paint_canvas(ent: Entity<Editor>, bounds: Bounds<Pixels>, hitbox: Hitbox, frame: CanvasFrame, window: &mut Window, _cx: &mut App) {
    window.with_content_mask(Some(ContentMask { bounds }), |window| {
        paint_plan(frame.plan, window);
        for o in frame.overlays {
            match o {
                // Square handles (small, unrounded) turn with Rotate View.
                Overlay::Quad { bounds: b, bg, border, border_w, radius } if frame.rot != 0.0 && radius == 0.0 && b.size.width <= px(24.) && b.size.height <= px(24.) => {
                    let c = b.center();
                    let (sn, cs) = frame.rot.sin_cos();
                    let (hw, hh) = (f32::from(b.size.width) * 0.5, f32::from(b.size.height) * 0.5);
                    let pts: Vec<_> = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)].iter().map(|&(x, y)| c + point(px(x * cs - y * sn), px(x * sn + y * cs))).collect();
                    let mut fill_pb = PathBuilder::fill();
                    fill_pb.move_to(pts[0]);
                    for p in &pts[1..] {
                        fill_pb.line_to(*p);
                    }
                    fill_pb.close();
                    if let Ok(path) = fill_pb.build() {
                        window.paint_path(path, bg);
                    }
                    if border_w > 0.0 {
                        let mut pb = PathBuilder::stroke(px(border_w));
                        pb.move_to(pts[0]);
                        for p in pts[1..].iter().chain(std::iter::once(&pts[0])) {
                            pb.line_to(*p);
                        }
                        if let Ok(path) = pb.build() {
                            window.paint_path(path, border);
                        }
                    }
                }
                Overlay::Quad { bounds, bg, border, border_w, radius } => {
                    window.paint_quad(quad(bounds, px(radius), bg, px(border_w), border, BorderStyle::Solid));
                }
                Overlay::Image { bounds: ib, img, radius } => {
                    // Clip to the canvas: document-sized overlays (the object
                    // finder's tint) may extend past it when zoomed in.
                    let _ = window.paint_image(ib.intersect(&bounds), ib, Corners::all(px(radius)), img, 0, false);
                }
                Overlay::Path { polys, dashed, color, width } => {
                    let mut pb = PathBuilder::stroke(px(width));
                    if dashed {
                        pb = pb.dash_array(&[px(4.), px(4.)]);
                    }
                    let mut any = false;
                    for (pts, closed) in polys {
                        if pts.len() < 2 {
                            continue;
                        }
                        pb.move_to(pts[0]);
                        for p in &pts[1..] {
                            pb.line_to(*p);
                        }
                        if closed {
                            pb.line_to(pts[0]);
                        }
                        any = true;
                    }
                    if any {
                        if let Ok(path) = pb.build() {
                            window.paint_path(path, color);
                        }
                    }
                }
                Overlay::Fill { polys, color } => {
                    let mut pb = PathBuilder::fill();
                    let mut any = false;
                    for pts in polys {
                        if pts.len() < 3 {
                            continue;
                        }
                        pb.move_to(pts[0]);
                        for p in &pts[1..] {
                            pb.line_to(*p);
                        }
                        pb.close();
                        any = true;
                    }
                    if any {
                        if let Ok(path) = pb.build() {
                            window.paint_path(path, color);
                        }
                    }
                }
                Overlay::Text { pos, text, color, size, bg } => {
                    let font = window.text_style().font();
                    let run = gpui_kit::TextRun { len: text.len(), font, color, background_color: None, underline: None, strikethrough: None };
                    let line = window.text_system().shape_line(text, px(size), &[run], None);
                    let lh = px(size * 1.3);
                    if let Some(bg) = bg {
                        let b = Bounds::from_corners(pos - point(px(3.), px(1.)), pos + point(line.width + px(3.), lh + px(1.)));
                        window.paint_quad(fill(b, bg).corner_radii(px(3.)));
                    }
                    let _ = line.paint(pos, lh, gpui_kit::TextAlign::Left, None, window, _cx);
                }
            }
        }
    });
    window.set_cursor_style(frame.cursor, &hitbox);

    {
        let ent = ent.clone();
        let hitbox = hitbox.clone();
        window.on_mouse_event(move |ev: &MouseDownEvent, phase, window, cx| {
            if phase == DispatchPhase::Bubble && hitbox.is_hovered(window) {
                ent.update(cx, |ed, cx| ed.on_mouse_down(ev, window, cx));
            }
        });
    }
    {
        let ent = ent.clone();
        window.on_mouse_event(move |ev: &MouseMoveEvent, phase, window, cx| {
            if phase == DispatchPhase::Bubble {
                ent.update(cx, |ed, cx| ed.on_mouse_move(ev, window, cx));
            }
        });
    }
    {
        let ent = ent.clone();
        window.on_mouse_event(move |ev: &MouseUpEvent, phase, window, cx| {
            if phase == DispatchPhase::Bubble && !matches!(ent.read(cx).drag, Drag::None) {
                ent.update(cx, |ed, cx| ed.on_mouse_up(ev, window, cx));
            }
        });
    }
    {
        let ent = ent.clone();
        let hitbox = hitbox.clone();
        window.on_mouse_event(move |ev: &ScrollWheelEvent, phase, window, cx| {
            if phase == DispatchPhase::Bubble && hitbox.is_hovered(window) {
                ent.update(cx, |ed, cx| ed.on_scroll(ev, window, cx));
            }
        });
    }
    {
        let ent = ent.clone();
        window.on_mouse_event(move |ev: &gpui_kit::MousePressureEvent, phase, _window, cx| {
            if phase == DispatchPhase::Bubble {
                ent.update(cx, |ed, _| ed.pressure = if ev.stage == gpui_kit::PressureStage::Zero { 1.0 } else { ev.pressure.clamp(0.05, 1.0) });
            }
        });
    }
    window.on_mouse_event(move |ev: &PinchEvent, phase, window, cx| {
        if phase == DispatchPhase::Bubble && hitbox.is_hovered(window) {
            ent.update(cx, |ed, cx| ed.on_pinch(ev, window, cx));
        }
    });
}

impl Editor {
    fn render_toolbox(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        use super::tools::{GROUPS, SEPARATE_BEFORE};
        let mut col = div()
            .id("toolbox")
            .w(px(46.))
            .flex()
            .flex_col()
            .items_center()
            .gap_0p5()
            .py_2()
            .bg(rgb(PANEL))
            .border_r_1()
            .border_color(rgb(BORDER))
            .overflow_y_scroll();
        for (gi, g) in GROUPS.iter().enumerate() {
            let visible: Vec<Tool> = g.tools.iter().copied().filter(|t| !self.toolbar_hidden.contains(t)).collect();
            if visible.is_empty() {
                continue;
            }
            if SEPARATE_BEFORE.contains(&gi) {
                col = col.child(div().w(px(26.)).h(px(1.)).my_1().bg(rgb(BORDER)));
            }
            let shown = if visible.contains(&self.tool_memory[gi]) { self.tool_memory[gi] } else { visible[0] };
            let active = self.tool.group() == gi;
            let multi = visible.len() > 1;
            let mut cell = div()
                .id(SharedString::from(format!("tg-{gi}")))
                .relative()
                .on_mouse_down(gpui_kit::MouseButton::Right, cx.listener(move |this, _, _, cx| {
                    this.flyout = if this.flyout.map(|f| f.0) == Some(gi) { None } else { Some((gi, 0.0)) };
                    cx.notify();
                }))
                .on_mouse_down(gpui_kit::MouseButton::Left, cx.listener(move |this, _, _, cx| {
                    this.press_started = Some((gi, std::time::Instant::now()));
                    let t0 = this.press_started;
                    cx.spawn(async move |this, cx| {
                        cx.background_executor().timer(std::time::Duration::from_millis(380)).await;
                        let _ = this.update(cx, |ed, cx| {
                            if ed.press_started.is_some() && ed.press_started.map(|p| p.0) == t0.map(|p| p.0) {
                                ed.flyout = Some((gi, 0.0));
                                ed.press_started = None;
                                cx.notify();
                            }
                        });
                    })
                    .detach();
                }))
                .child(
                    Button::new(SharedString::from(format!("tool-{gi}")))
                        .icon(Icon::new(shown.icon()))
                        .ghost()
                        .small()
                        .selected(active)
                        .tooltip(if multi { format!("{} · right-click or hold for more", shown.tooltip()) } else { shown.tooltip() })
                        .on_click(cx.listener(move |this, _, window, cx| {
                            let long = this.press_started.is_none() && this.flyout.is_some();
                            this.press_started = None;
                            if !long {
                                this.set_tool(shown, window, cx);
                            }
                        })),
                );
            if multi {
                // Photoshop's little corner triangle marks a flyout.
                cell = cell.child(div().absolute().right(px(1.)).bottom(px(1.)).size(px(4.)).bg(rgb(DIM)).rounded_sm());
            }
            if self.flyout.map(|f| f.0) == Some(gi) {
                let mut list = div()
                    .id(SharedString::from(format!("flyout-{gi}")))
                    .occlude()
                    .min_w(px(260.))
                    .p_1()
                    .rounded_md()
                    .bg(rgb(PANEL_HI))
                    .border_1()
                    .border_color(rgb(BORDER))
                    .shadow_lg()
                    .flex()
                    .flex_col();
                for t in visible {
                    let key = t.shortcut().map(|k| k.to_uppercase()).unwrap_or_default();
                    list = list.child(
                        div()
                            .id(SharedString::from(format!("fly-{t:?}")))
                            .flex()
                            .items_center()
                            .gap_2()
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .when(self.tool == t, |d| d.bg(rgb(ACCENT).opacity(0.35)))
                            .hover(|s| s.bg(rgb(BORDER)))
                            .on_click(cx.listener(move |this, _, window, cx| this.set_tool(t, window, cx)))
                            .child(Icon::new(t.icon()).text_color(rgb(TEXT)))
                            .child(div().flex_1().text_color(rgb(TEXT)).child(t.name()))
                            .child(div().text_color(rgb(DIM)).child(key)),
                    );
                }
                cell = cell.child(gpui_kit::deferred(div().absolute().left(px(40.)).top(px(0.)).child(list)).with_priority(2));
            }
            col = col.child(cell);
        }
        // Edit Toolbar ("…"): show / hide tools.
        let mut edit = div().relative().child(
            Button::new("edit-toolbar")
                .icon(Icon::new(gpui_kit::assets::IconName::Ellipsis))
                .ghost()
                .small()
                .tooltip("Edit Toolbar — choose which tools are shown")
                .on_click(cx.listener(|this, _, _, cx| {
                    this.flyout = if this.flyout.map(|f| f.0) == Some(usize::MAX) { None } else { Some((usize::MAX, 0.0)) };
                    cx.notify();
                })),
        );
        if self.flyout.map(|f| f.0) == Some(usize::MAX) {
            let mut list = div().id("edit-toolbar-list").occlude().w(px(520.)).max_h(px(560.)).overflow_y_scroll().p_2().rounded_md().bg(rgb(PANEL_HI)).border_1().border_color(rgb(BORDER)).shadow_lg().flex().flex_wrap().gap_1();
            for g in super::tools::GROUPS {
                for &t in g.tools {
                    let hidden = self.toolbar_hidden.contains(&t);
                    list = list.child(
                        Button::new(SharedString::from(format!("et-{t:?}")))
                            .icon(Icon::new(t.icon()))
                            .label(t.name())
                            .xsmall()
                            .selected(!hidden)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                if !this.toolbar_hidden.remove(&t) {
                                    this.toolbar_hidden.insert(t);
                                }
                                cx.notify();
                            })),
                    );
                }
            }
            edit = edit.child(gpui_kit::deferred(div().absolute().left(px(40.)).bottom(px(0.)).child(list)).with_priority(2));
        }
        col = col.child(edit);
        col.child(div().flex_1().min_h(px(8.)))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .items_center()
                    .gap_1()
                    .child(ColorPicker::new(&self.ui.fg).small())
                    .child(ColorPicker::new(&self.ui.bg).small())
                    .child(
                        div()
                            .flex()
                            .child(
                                Button::new("swap-colors")
                                    .icon(Icon::new(gpui_kit::assets::IconName::ArrowLeftRight))
                                    .ghost()
                                    .xsmall()
                                    .tooltip("Swap colors (X)")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        let (f, b) = (this.fg, this.bg);
                                        this.set_fg(b, window, cx);
                                        this.set_bg(f, window, cx);
                                        cx.notify();
                                    })),
                            )
                            .child(
                                Button::new("reset-colors")
                                    .icon(Icon::new(gpui_kit::assets::IconName::Contrast))
                                    .ghost()
                                    .xsmall()
                                    .tooltip("Default colors (D)")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.set_fg([0, 0, 0], window, cx);
                                        this.set_bg([255, 255, 255], window, cx);
                                        cx.notify();
                                    })),
                            ),
                    )
                    .child(
                        Button::new("quick-mask")
                            .icon(Icon::new(gpui_kit::assets::IconName::SquareDashed))
                            .ghost()
                            .small()
                            .selected(self.quick_mask.is_some())
                            .tooltip("Quick Mask mode (Q) — paint the selection as a red overlay")
                            .on_click(cx.listener(|this, _, _, cx| this.toggle_quick_mask(cx))),
                    )
                    .child(
                        Button::new("screen-mode")
                            .icon(Icon::new(gpui_kit::assets::IconName::Monitor))
                            .ghost()
                            .small()
                            .tooltip("Screen mode (F) — standard, full screen with menus, full screen")
                            .on_click(cx.listener(|this, _, window, cx| this.cycle_screen_mode(window, cx))),
                    ),
            )
    }

    /// Document tabs, the arrangement switcher and Match.
    fn render_doc_tabs(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        use super::docs::{Arrange, Match};
        let mut bar = div().h(px(30.)).flex().flex_row().items_end().gap_0p5().px_2().bg(rgb(BG)).border_b_1().border_color(rgb(BORDER));
        for (uid, title, active) in self.doc_tabs() {
            let tab = div()
                .id(SharedString::from(format!("doc-tab-{uid}")))
                .flex()
                .items_center()
                .gap_2()
                .px_3()
                .h(px(26.))
                .rounded_t_md()
                .text_size(px(12.))
                .bg(rgb(if active { PANEL_HI } else { PANEL }))
                .text_color(rgb(if active { TEXT } else { DIM }))
                .border_t_2()
                .border_color(if active { rgb(ACCENT) } else { rgb(PANEL) })
                .child(title)
                .child(
                    div()
                        .id(SharedString::from(format!("doc-close-{uid}")))
                        .px_1()
                        .rounded_sm()
                        .hover(|s| s.bg(rgb(BORDER)))
                        .child("×")
                        .on_click(cx.listener(move |ed, _, window, cx| {
                            cx.stop_propagation();
                            ed.close_doc(uid, window, cx);
                        })),
                )
                .on_click(cx.listener(move |ed, _, window, cx| ed.activate_doc(uid, window, cx)));
            bar = bar.child(tab);
        }
        if self.docs.count() > 1 {
            let mut right = div().ml_auto().flex().items_center().gap_1().pb_0p5();
            right = right.child(super::options::segmented("arrange", &Arrange::ALL.map(|a| (a, a.label())), self.docs.arrange, cx, |e, a| e.set_arrange(a)));
            for (i, (m, name)) in [(Match::Zoom, "Match Zoom"), (Match::Location, "Location"), (Match::Rotation, "Rotation"), (Match::All, "Match All")].into_iter().enumerate() {
                right = right.child(Button::new(SharedString::from(format!("match-{i}"))).label(name).xsmall().ghost().on_click(cx.listener(move |e, _, _, cx| {
                    e.match_views(m);
                    cx.notify();
                })));
            }
            right = right
                .child(super::options::toggle("zoom-all", "Zoom All", self.docs.zoom_all).on_click(cx.listener(|e, _, _, cx| {
                    e.docs.zoom_all = !e.docs.zoom_all;
                    cx.notify();
                })))
                .child(super::options::toggle("pan-all", "Scroll All", self.docs.pan_all).on_click(cx.listener(|e, _, _, cx| {
                    e.docs.pan_all = !e.docs.pan_all;
                    cx.notify();
                })));
            bar = bar.child(right);
        }
        bar
    }

    /// The active document's interactive canvas.
    fn main_canvas(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let ent = cx.entity();
        let prep = ent.clone();
        canvas(
            move |bounds, window, cx| {
                let hitbox = window.insert_hitbox(bounds, HitboxBehavior::Normal);
                let frame = prep.update(cx, |ed, cx| ed.prepare_frame(bounds, window, cx));
                (hitbox, frame)
            },
            move |bounds, (hitbox, frame), window, cx| paint_canvas(ent, bounds, hitbox, frame, window, cx),
        )
        .size_full()
    }

    /// Another document's pane: its own tiles; a click activates it.
    fn slot_canvas(&self, uid: u64, cx: &mut Context<Self>) -> impl IntoElement {
        let ent = cx.entity();
        let prep = ent.clone();
        canvas(
            move |bounds, window, cx| {
                let hitbox = window.insert_hitbox(bounds, HitboxBehavior::Normal);
                let plan = prep.update(cx, |ed, _| ed.prepare_slot(uid, bounds, window));
                (hitbox, plan)
            },
            move |bounds, (hitbox, plan), window, _cx| {
                if let Some(plan) = plan {
                    if plan.pending > 0 {
                        window.request_animation_frame();
                    }
                    window.with_content_mask(Some(ContentMask { bounds }), |window| paint_plan(plan, window));
                }
                window.on_mouse_event(move |ev: &MouseDownEvent, phase, window, cx| {
                    if phase == DispatchPhase::Bubble && hitbox.is_hovered(window) && ev.button == gpui_kit::MouseButton::Left {
                        ent.update(cx, |ed, cx| ed.activate_doc(uid, window, cx));
                    }
                });
            },
        )
        .size_full()
    }

    fn render_canvas_area(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        use super::docs::{Arrange, Docs};
        let busy = self.busy.as_ref().map(|b| (b.label.clone(), b.progress.fraction(), b.started.elapsed().as_secs_f32()));
        let panes = self.docs.panes();
        let mut area = div().flex_1().min_w_0().min_h_0().relative().overflow_hidden().bg(rgb(PASTEBOARD));
        if panes.len() < 2 {
            area = area.child(self.main_canvas(cx));
        } else {
            let (cols, rows) = if self.docs.arrange == Arrange::TwoUp { (2, 1) } else { Docs::grid(panes.len()) };
            let mut grid = div().size_full().flex().flex_col().gap_1();
            for r in 0..rows {
                let mut row = div().flex_1().min_h_0().flex().flex_row().gap_1();
                for c in 0..cols {
                    let Some(&uid) = panes.get(r * cols + c) else {
                        row = row.child(div().flex_1().min_w_0());
                        continue;
                    };
                    let active = uid == self.docs.uid;
                    let body: AnyElement = if active { self.main_canvas(cx).into_any_element() } else { self.slot_canvas(uid, cx).into_any_element() };
                    row = row.child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .relative()
                            .overflow_hidden()
                            .border_1()
                            .border_color(if active { rgb(ACCENT) } else { rgb(BORDER) })
                            .child(body)
                            .child(
                                div()
                                    .absolute()
                                    .top(px(6.))
                                    .left(px(8.))
                                    .px_2()
                                    .rounded_sm()
                                    .text_size(px(11.))
                                    .bg(hsla(0., 0., 0., 0.55))
                                    .text_color(rgb(if active { TEXT } else { DIM }))
                                    .child(self.doc_title(uid)),
                            ),
                    );
                }
                grid = grid.child(row);
            }
            area = area.child(grid);
        }
        if let Some((label, frac, secs)) = busy {
            area = area.child(
                div().absolute().bottom(px(18.)).left_0().right_0().flex().justify_center().child(
                    div()
                        .w(px(360.))
                        .p_3()
                        .rounded_lg()
                        .bg(rgb(PANEL_HI))
                        .border_1()
                        .border_color(rgb(BORDER))
                        .shadow_lg()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div()
                                .flex()
                                .justify_between()
                                .child(div().text_color(rgb(TEXT)).child(label))
                                .child(div().text_color(rgb(DIM)).child(format!("{secs:.1}s"))),
                        )
                        .child(
                            div().h(px(4.)).w_full().rounded_full().bg(rgb(BORDER)).child(
                                div()
                                    .h_full()
                                    .rounded_full()
                                    .bg(rgb(ACCENT))
                                    .w(gpui_kit::relative(if frac > 0.0 { frac } else { ((secs * 1.5).sin() * 0.5 + 0.5) * 0.9 + 0.05 })),
                            ),
                        ),
                ),
            );
        }
        if let Some(d) = &self.dialog {
            if matches!(d.kind, DialogKind::Filter(_)) {
                area = area.child(div().absolute().top(px(14.)).right(px(14.)).child(self.render_dialog(cx)));
            }
        }
        if let Some(ls) = self.render_layer_style(cx) {
            area = area.child(div().absolute().top(px(14.)).left(px(14.)).child(ls));
        }
        area
    }

    fn render_history(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let (undo, redo) = self.doc.history_labels();
        let cur = undo.len();
        if self.history_seen != cur {
            self.history_seen = cur;
            self.history_scroll.scroll_to_item(cur);
        }
        let mut list = div().id("history").track_scroll(&self.history_scroll).h(px(150.)).overflow_y_scroll().flex().flex_col().p_1();
        let entries = std::iter::once("Document".to_string()).chain(undo).chain(redo).enumerate().collect::<Vec<_>>();
        let src_pos = self.ropts.history_pos;
        for (pos, label) in entries {
            let future = pos > cur;
            let marker = div()
                .id(SharedString::from(format!("hsrc-{pos}")))
                .w(px(16.))
                .text_color(rgb(if pos == src_pos { TEXT } else { BORDER }))
                .child(if pos == src_pos { "◉" } else { "○" })
                .on_click(cx.listener(move |this, _, _, cx| {
                    cx.stop_propagation();
                    this.ropts.history_pos = pos;
                    this.flash("History Brush source set");
                    cx.notify();
                }));
            list = list.child(
                div()
                    .id(SharedString::from(format!("hist-{pos}")))
                    .flex()
                    .gap_1()
                    .px_2()
                    .py_0p5()
                    .rounded_sm()
                    .child(marker)
                    .text_color(rgb(if future { DIM } else { TEXT }))
                    .when(pos == cur, |d| d.bg(rgb(ACCENT).opacity(0.35)))
                    .when(pos != cur, |d| d.hover(|s| s.bg(rgb(PANEL_HI))))
                    .on_click(cx.listener(move |this, _, window, cx| this.jump_history(cur as isize - pos as isize, window, cx)))
                    .child(label),
            );
        }
        div().flex().flex_col().border_t_1().border_color(rgb(BORDER)).child(section_title("History")).child(list)
    }

    fn render_info(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let st = self.cache.last;
        let mp = self.doc.w as f64 * self.doc.h as f64 / 1e6;
        let cursor = self.mouse_img.map_or("—".to_string(), |(x, y)| format!("{:.0}, {:.0}", x.floor(), y.floor()));
        let swatch = self.hover_rgba.unwrap_or([0, 0, 0, 0]);
        let mem = self.memory_text();
        let storage = crate::engine::memory::global().stats();
        let text_limits = self.doc.layers.get(self.doc.active).and_then(|layer| {
            if let crate::engine::layers::LayerKind::Text(text) = &layer.kind {
                (!text.imported_text_limitations().is_empty()).then(|| text.imported_text_limitations().join("; "))
            } else { None }
        });
        div()
            .flex()
            .flex_col()
            .border_t_1()
            .border_color(rgb(BORDER))
            .child(section_title("Info"))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_0p5()
                    .p_2()
                    .child(info_row("Document", format!("{} × {} · {mp:.1} MP", self.doc.w, self.doc.h)))
                    .child(info_row("Mode", "RGB · 8 bits/channel"))
                    .child(info_row("Format", self.doc.path.as_ref().and_then(|p| p.extension()).map(|s| s.to_string_lossy().to_uppercase()).unwrap_or_else(|| "Unsaved".into())))
                    .child(info_row("Zoom", format!("{:.1}% · mip level {}", self.vp.zoom * 100.0, st.level)))
                    .child(info_row("Cursor", cursor))
                    .when_some(text_limits, |d, message| d.child(div().text_color(rgb(DIM)).child(message)))
                    .child(
                        div()
                            .flex()
                            .justify_between()
                            .child(
                                div()
                                    .id("info-mode")
                                    .text_color(rgb(DIM))
                                    .hover(|s| s.text_color(rgb(TEXT)))
                                    .child(format!("{} ▾", self.nopts.info_mode.label()))
                                    .on_click(cx.listener(|e, _, _, cx| {
                                        e.nopts.info_mode = e.nopts.info_mode.next();
                                        cx.notify();
                                    })),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(div().size(px(10.)).border_1().border_color(rgb(BORDER)).bg(Rgba {
                                        r: swatch[0] as f32 / 255.,
                                        g: swatch[1] as f32 / 255.,
                                        b: swatch[2] as f32 / 255.,
                                        a: 1.0,
                                    }))
                                    .child(div().text_color(rgb(TEXT)).child(match self.nopts.info_mode {
                                        crate::engine::color::ColorMode::Rgb => format!("{} {} {} {}", swatch[0], swatch[1], swatch[2], swatch[3]),
                                        m => m.format([swatch[0], swatch[1], swatch[2]]),
                                    })),
                            ),
                    )
                    .children(self.sampler_rows().into_iter().enumerate().map(|(i, (name, c, mode))| {
                        div()
                            .flex()
                            .justify_between()
                            .child(
                                div()
                                    .id(SharedString::from(format!("smode-{i}")))
                                    .text_color(rgb(DIM))
                                    .hover(|s| s.text_color(rgb(TEXT)))
                                    .child(format!("{name} {}", mode.label()))
                                    .on_click(cx.listener(move |e, _, _, cx| {
                                        let m = &mut e.doc.ann.sampler_modes;
                                        if m.len() <= i {
                                            m.resize(i + 1, Default::default());
                                        }
                                        m[i] = m[i].next();
                                        cx.notify();
                                    })),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(div().size(px(10.)).border_1().border_color(rgb(BORDER)).bg(Rgba { r: c[0] as f32 / 255., g: c[1] as f32 / 255., b: c[2] as f32 / 255., a: 1.0 }))
                                    .child(div().text_color(rgb(TEXT)).child(mode.format([c[0], c[1], c[2]]))),
                            )
                    }))
                    .when_some(self.nopts.ruler, |d, (a, b)| {
                        let (w, h, ang, l) = crate::engine::annot::measure(a, b);
                        d.child(info_row("Ruler", format!("W {w:.0} H {h:.0} · {ang:.1}° · {l:.1} px")))
                    })
                    .when(self.doc.ann.count_total() > 0, |d| d.child(info_row("Count", format!("{}", self.doc.ann.count_total()))))
                    .child(info_row("Display tiles", format!("{} visible · {} cached", st.visible, st.cached)))
                    .child(info_row("Last compose", format!("{} tiles in {:.1} ms{}", st.composed, st.compose_ms, if st.pending > 0 { format!(" · {} pending", st.pending) } else { String::new() })))
                    .child(info_row("Timing", match &self.last_job {
                        // The tool step only when a tool ran (long labels wrap otherwise).
                        Some((l, ms)) if self.last_tool_ms < 0.005 => format!("{l} {ms:.0} ms"),
                        Some((l, ms)) => format!("{l} {ms:.0} ms · tool {:.2} ms", self.last_tool_ms),
                        None => format!("tool step {:.2} ms", self.last_tool_ms),
                    }))
                    .child(info_row("Path", match crate::engine::gpu::last_op() {
                        Some(op) => format!("{} · {} {:.0} ms", op.op, op.short_path(), op.ms),
                        None => "—".into(),
                    }))
                    .child(info_row("Compose", match crate::engine::gpu::last_compose() {
                        Some(op) => format!("{} · {} {:.0} ms", op.op, op.short_path(), op.ms),
                        None => "CPU".into(),
                    }))
                    .child(info_row("GPU", format!("{} · {}", crate::engine::gpu::describe_short(), crate::engine::gpu::mode().label())))
                    .when_some(crate::engine::gpu::context(false), |d, g| {
                        let ((bytes, n), (pooled, _)) = (crate::engine::gpu::resident::usage(&g), crate::engine::gpu::tiles::usage(&g));
                        d.when(n > 0 || pooled > 0, |d| d.child(info_row("GPU memory", format!("{n} image{}, {} MB · {pooled} tiles", if n == 1 { "" } else { "s" }, bytes >> 20))))
                    })
                    .child(info_row("Memory", mem))
                    .child(info_row("Compressed", format!("{:.1} MB", storage.compressed as f64 / 1048576.)))
                    .child(info_row("Scratch disk", format!("{:.1} MB", storage.disk as f64 / 1048576.)))
                    .when_some(storage.fault, |d, fault| d.child(info_row("Storage", fault)))
                    .child(info_row("Threads", format!("{} rayon workers", rayon::current_num_threads()))),
            )
    }

    /// Settings ▸ Performance: Use Graphics Processor, per-feature overrides,
    /// Benchmark GPU.
    fn render_performance(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        use crate::engine::gpu::{Feature, Mode};
        let prefs = &self.dockst.prefs;
        let env = crate::engine::gpu::env_mode();
        let mut modes = div().flex().items_center().gap_1().child(div().w(px(150.)).text_color(rgb(TEXT)).child("Use Graphics Processor"));
        for m in Mode::ALL {
            let b = Button::new(SharedString::from(format!("gpu-mode-{}", m.label()))).label(m.label()).xsmall();
            modes = modes.child(if prefs.gpu == m { b.primary() } else { b.ghost() }.on_click(cx.listener(move |this, _, _, cx| this.set_gpu_mode(m, cx))));
        }
        let mut grid = div().flex().flex_col().gap_1();
        for f in Feature::ALL {
            let cur = prefs.gpu_overrides.iter().find(|o| o.0 == f).map(|o| o.1);
            let mut row = div().flex().items_center().gap_1().child(div().w(px(150.)).text_color(rgb(DIM)).child(f.label()));
            for (i, (label, m)) in [("Default", None), ("Auto", Some(Mode::Auto)), ("Off", Some(Mode::Off)), ("Force", Some(Mode::Force))].into_iter().enumerate() {
                let b = Button::new(SharedString::from(format!("gpu-ov-{}-{i}", f.key()))).label(label).xsmall();
                row = row.child(if cur == m { b.primary() } else { b.ghost() }.on_click(cx.listener(move |this, _, _, cx| this.set_gpu_override(f, m, cx))));
            }
            grid = grid.child(row);
        }
        let status = match crate::engine::gpu::context(false) {
            Some(g) => format!(
                "{} · {} · {:.1} GB{} · f16 {} · subgroups {} · timestamps {}",
                g.info.name,
                g.info.backend,
                g.info.vram as f64 / (1u64 << 30) as f64,
                if g.caps.unified_memory { " unified" } else { "" },
                if g.caps.f16 { "yes" } else { "no" },
                if g.caps.subgroups { "yes" } else { "no" },
                if g.caps.timestamps { "yes" } else { "no" },
            ),
            None => crate::engine::gpu::describe(),
        };
        div()
            .flex()
            .flex_col()
            .gap_3()
            .w_full()
            .child(div().text_size(px(11.)).text_color(rgb(DIM)).child(status))
            .child(div().flex().items_center().gap_2()
                .child(div().w_40().child("Memory allowance"))
                .child(Button::new("memory-less").label("−10%").small().on_click(cx.listener(|ed, _, _, cx| ed.set_memory_percent(ed.dockst.prefs.memory_percent.saturating_sub(10), cx))))
                .child(format!("{}% of RAM", prefs.memory_percent))
                .child(Button::new("memory-more").label("+10%").small().on_click(cx.listener(|ed, _, _, cx| ed.set_memory_percent(ed.dockst.prefs.memory_percent.saturating_add(10), cx)))))
            .child(div().flex().items_center().gap_2()
                .child(div().w_40().child("Scratch folder"))
                .child(div().flex_1().min_w_0().truncate().child(prefs.scratch_dir.display().to_string()))
                .child(Button::new("scratch-folder").label("Choose…").small().on_click(cx.listener(|ed, _, _, cx| ed.choose_scratch_directory(cx)))))
            .child(modes)
            .child(div().text_size(px(11.)).text_color(rgb(DIM)).child(
                "Auto runs an operation on the GPU when the cost model predicts it's faster (big regions, heavy filters); Off always uses the CPU; Force uses the GPU whenever it has a kernel. The CPU result is the reference: the GPU gives the same pixels, and anything that fails on the GPU finishes on the CPU.",
            ))
            .when_some(env, |d, m| d.child(div().text_size(px(11.)).text_color(rgb(ACCENT)).child(format!("PF_GPU={} is set and overrides these settings.", m.label().to_lowercase()))))
            .child(div().text_color(rgb(TEXT)).child("Advanced: per-feature overrides"))
            .child(grid)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(Button::new("gpu-bench").label("Benchmark GPU").xsmall().primary().on_click(cx.listener(|this, _, _, cx| this.benchmark_gpu(cx))))
                    .child(div().text_size(px(11.)).text_color(rgb(DIM)).child("Re-enables a disabled GPU and recalibrates Auto mode.")),
            )
            .when_some(self.gpu_bench.clone(), |d, t| d.child(div().text_size(px(11.)).text_color(rgb(TEXT)).child(t)))
    }

    fn render_side_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let caf = self.caf.is_some().then(|| self.render_caf_panel(cx));
        let typ = (self.caf.is_none() && self.type_panel_visible()).then(|| self.render_type_panel(window, cx));
        let shp = (self.caf.is_none() && typ.is_none() && self.shape_panel_visible()).then(|| self.render_shape_panel(window, cx));
        let slc = (self.caf.is_none() && typ.is_none() && shp.is_none() && self.slice_panel_visible()).then(|| self.render_slice_panel(window, cx));
        div()
            .w(px(300.))
            .flex()
            .flex_col()
            .bg(rgb(PANEL))
            .border_l_1()
            .border_color(rgb(BORDER))
            .when_some(caf, |d, p| d.child(p))
            .when_some(typ, |d, p| d.child(div().id("type-panel").flex_shrink_0().max_h(px(330.)).overflow_y_scroll().child(p)))
            .when_some(shp, |d, p| d.child(div().id("shape-panel").flex_shrink_0().max_h(px(380.)).overflow_y_scroll().child(p)))
            .when_some(slc, |d, p| d.child(div().id("slice-panel").flex_shrink_0().max_h(px(520.)).overflow_y_scroll().child(p)))
            .when(self.caf.is_none(), |d| d.child(self.render_layers(cx)).child(self.render_paths_panel(cx)))
            .child(self.render_history(cx))
            .child(self.render_info(cx))
    }

    fn render_status(&mut self) -> impl IntoElement {
        let title = self
            .doc
            .path
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Untitled".into());
        div()
            .h(px(24.))
            .flex()
            .items_center()
            .justify_between()
            .px_3()
            .bg(rgb(PANEL_HI))
            .border_t_1()
            .border_color(rgb(BORDER))
            .text_size(px(11.))
            .child(div().text_color(rgb(TEXT)).child(self.status.clone()))
            .child(div().text_color(rgb(DIM)).child(format!(
                "{title}{} · {:.1}% · {} layers",
                if self.doc.modified { " •" } else { "" },
                self.vp.zoom * 100.0,
                self.doc.layers.len()
            )))
    }

    fn render_dialog(&mut self, cx: &mut Context<Self>) -> AnyElement {
        let Some(d) = &self.dialog else { return div().into_any_element() };
        let title = match &d.kind {
            DialogKind::NewDocument => "New Document".to_string(),
            DialogKind::ImageSize => "Image Size".to_string(),
            DialogKind::Preferences => "Settings ▸ Performance".to_string(),
            DialogKind::Modify(m) => m.title().to_string(),
            DialogKind::SelectMask => "Select and Mask".to_string(),
            DialogKind::Filter(k) => k.entry().title.to_string(),
        };
        let mut body = div().id("dialog-body").max_h(px(520.)).overflow_y_scroll().flex().flex_col().gap_3().pr_1();
        for s in &d.sliders {
            if matches!(d.kind, DialogKind::ImageSize) && !self.size_resample.has_noise() {
                continue;
            }
            let val = s.state.read(cx).value().start();
            body = body.child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .flex()
                            .justify_between()
                            .child(div().text_color(rgb(DIM)).child(s.label))
                            .child(div().text_color(rgb(TEXT)).child(if s.choices.is_empty() {
                                format!("{val:.prec$}", prec = s.decimals)
                            } else {
                                s.choices[(val.round() as usize).min(s.choices.len() - 1)].to_string()
                            })),
                    )
                    .child(Slider::new(&s.state)),
            );
        }
        for (label, input) in &d.inputs {
            body = body.child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(div().w(px(56.)).text_color(rgb(DIM)).child(*label))
                    .child(div().flex_1().child(Input::new(input).small()))
                    .child(div().text_color(rgb(DIM)).child("px")),
            );
        }
        match &d.kind {
            DialogKind::NewDocument => {
                let presets: [(&str, u32, u32); 6] = [
                    ("Full HD", 1920, 1080),
                    ("4K", 3840, 2160),
                    ("24 MP", 6000, 4000),
                    ("100 MP", 12500, 8000),
                    ("300 MP", 20000, 15000),
                    ("1 GP", 40000, 25000),
                ];
                let mut row = div().flex().flex_wrap().gap_1();
                for (i, (name, w, h)) in presets.into_iter().enumerate() {
                    row = row.child(
                        Button::new(SharedString::from(format!("preset-{i}")))
                            .label(name)
                            .xsmall()
                            .on_click(cx.listener(move |this, _, window, cx| this.set_preset(w, h, window, cx))),
                    );
                }
                let white = d.flag;
                body = body.child(row).child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .child(div().text_color(rgb(DIM)).child("Background"))
                        .child(Button::new("bg-white").label("White").xsmall().selected(white).on_click(cx.listener(|this, _, _, cx| {
                            if let Some(d) = &mut this.dialog {
                                d.flag = true;
                            }
                            cx.notify();
                        })))
                        .child(Button::new("bg-clear").label("Transparent").xsmall().selected(!white).on_click(cx.listener(|this, _, _, cx| {
                            if let Some(d) = &mut this.dialog {
                                d.flag = false;
                            }
                            cx.notify();
                        }))),
                );
                body = body.child(div().text_size(px(11.)).text_color(rgb(DIM)).child(
                    "Uniform tiles are shared, so even a gigapixel blank canvas allocates a single 256 KB tile until you paint.",
                ));
            }
            DialogKind::ImageSize => {
                let (w, h) = (self.doc.w, self.doc.h);
                let mut row = div().flex().gap_1();
                for (i, f) in [0.25f32, 0.5, 2.0, 4.0].into_iter().enumerate() {
                    let (nw, nh) = (((w as f32 * f).round() as u32).max(1), ((h as f32 * f).round() as u32).max(1));
                    row = row.child(
                        Button::new(SharedString::from(format!("scale-{i}")))
                            .label(format!("{:.0}%", f * 100.))
                            .xsmall()
                            .on_click(cx.listener(move |this, _, window, cx| this.set_preset(nw, nh, window, cx))),
                    );
                }
                use crate::engine::resample::Resample;
                let cur = self.size_resample;
                let note = match cur {
                    Resample::Bilinear => "Bilinear samples the matching mip level, so large reductions stay alias-free.",
                    Resample::Automatic => "Automatic: Bicubic Sharper for reductions, Bicubic Smoother for enlargements.",
                    Resample::PreserveDetails | Resample::PreserveDetails2 => {
                        "Lanczos enlargement with edge-aware detail recovery; Reduce Noise smooths flat areas."
                    }
                    Resample::NearestNeighbor => "Copies the nearest pixel: hard edges, no new colours.",
                    _ => "Separable filter, widened on reductions so they stay alias-free.",
                };
                body = body
                    .child(row)
                    .child(div().text_color(rgb(DIM)).child("Resample"))
                    .child(super::options::segmented("size-resample", &Resample::ALL.map(|k| (k, k.label())), cur, cx, |e, v| e.size_resample = v))
                    .child(div().text_size(px(11.)).text_color(rgb(DIM)).child(note));
            }
            DialogKind::Modify(_) => {
                body = body.child(div().text_size(px(11.)).text_color(rgb(DIM)).child(
                    "Runs per 256 px tile; only tiles near the selection edge are computed.",
                ));
            }
            DialogKind::Preferences => {
                body = body.child(self.render_performance(cx));
            }
            DialogKind::SelectMask => {
                let state = match self.select_mask.as_ref() {
                    Some(s) if s.computing => "Refining…",
                    _ => "Red overlay = unselected (live preview of the refined selection).",
                };
                body = body.child(div().text_size(px(11.)).text_color(rgb(DIM)).child(state)).child(div().text_size(px(11.)).text_color(rgb(DIM)).child(
                    "Radius snaps the edge band to image edges (guided filter); Smart Radius narrows it on hard edges. Decontaminate Colors always outputs a new layer with a layer mask.",
                ));
            }
            DialogKind::Filter(k) => {
                let entry = k.entry();
                if let Some((flag, _)) = entry.flag {
                    body = body.child(Button::new("dlg-flag").label(flag).xsmall().selected(d.flag).on_click(cx.listener(|this, _, _, cx| this.toggle_dialog_flag(cx))));
                }
                if let Some(note) = entry.note {
                    body = body.child(div().text_size(px(11.)).text_color(rgb(DIM)).child(note));
                }
                if *k == crate::engine::filters::catalog::Kind::Displace {
                    let label = if d.disp_map.is_some() { "Map image loaded · Choose another…" } else { "Choose map image…" };
                    body = body.child(Button::new("disp-map").label(label).xsmall().on_click(cx.listener(|this, _, w, cx| this.choose_displace_map(w, cx))));
                }
                body = body.child(div().text_size(px(11.)).text_color(rgb(DIM)).child(
                    "Live preview runs at screen resolution; OK applies it at full resolution across all cores.",
                ));
            }
        }
        div()
            .id("dialog")
            .occlude()
            .w(px(if matches!(self.dialog.as_ref().map(|d| &d.kind), Some(DialogKind::Preferences)) { 560. } else { 360. }))
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
            .child(div().text_size(px(14.)).text_color(rgb(TEXT)).child(title))
            .child(body)
            .child(
                div()
                    .flex()
                    .justify_end()
                    .gap_2()
                    .child(Button::new("dlg-cancel").label("Cancel").small().on_click(cx.listener(|this, _, window, cx| this.close_dialog(false, window, cx))))
                    .child(Button::new("dlg-ok").label("OK").small().primary().on_click(cx.listener(|this, _, window, cx| this.close_dialog(true, window, cx)))),
            )
            .into_any_element()
    }
}

fn root_actions(el: Stateful<Div>, cx: &mut Context<Editor>) -> Stateful<Div> {
    el.on_action(cx.listener(|t, _: &a::NewDocument, w, cx| t.open_new_dialog(w, cx)))
        .on_action(cx.listener(|t, _: &a::NewBenchmark, _, cx| t.act_benchmark(cx)))
        .on_action(cx.listener(|t, _: &a::Open, _, cx| t.act_open(cx)))
        .on_action(cx.listener(|t, _: &a::Save, _, cx| t.act_save(false, cx)))
        .on_action(cx.listener(|t, _: &a::SaveAs, _, cx| t.act_save(true, cx)))
        .on_action(cx.listener(|t, _: &a::EditUndo, w, cx| t.act_undo(false, w, cx)))
        .on_action(cx.listener(|t, _: &a::EditRedo, w, cx| t.act_undo(true, w, cx)))
        .on_action(cx.listener(|t, _: &a::EditCut, _, cx| t.act_copy(true, cx)))
        .on_action(cx.listener(|t, _: &a::EditCopy, _, cx| t.act_copy(false, cx)))
        .on_action(cx.listener(|t, _: &a::EditPaste, w, cx| t.act_paste(w, cx)))
        .on_action(cx.listener(|t, _: &a::FillForeground, _, cx| t.act_fill(false, cx)))
        .on_action(cx.listener(|t, _: &a::FillBackground, _, cx| t.act_fill(true, cx)))
        .on_action(cx.listener(|t, _: &a::ClearPixels, _, cx| t.act_clear(cx)))
        .on_action(cx.listener(|t, _: &a::SelectAll, _, cx| t.act_select(0, cx)))
        .on_action(cx.listener(|t, _: &a::Deselect, _, cx| t.act_select(1, cx)))
        .on_action(cx.listener(|t, _: &a::InvertSelection, _, cx| t.act_select(2, cx)))
        .on_action(cx.listener(|t, act: &a::ModifySelection, w, cx| t.open_modify_dialog(act.0, w, cx)))
        .on_action(cx.listener(|t, _: &a::SelectSubject, _, cx| t.select_subject(cx)))
        .on_action(cx.listener(|t, _: &a::SelectAndMask, w, cx| t.open_select_mask(w, cx)))
        .on_action(cx.listener(|t, _: &a::ContentAwareFill, _, cx| t.open_content_aware_fill(cx)))
        .on_action(cx.listener(|t, _: &a::PerspectiveWarp, _, cx| t.open_perspective_warp(cx)))
        .on_action(cx.listener(|t, _: &a::QuickMaskMode, _, cx| t.toggle_quick_mask(cx)))
        .on_action(cx.listener(|t, _: &a::GroupLayers, w, cx| t.group_layer(w, cx)))
        .on_action(cx.listener(|t, _: &a::UngroupLayers, w, cx| t.ungroup(w, cx)))
        .on_action(cx.listener(|t, _: &a::ClippingMask, _, cx| t.toggle_clip(cx)))
        .on_action(cx.listener(|t, act: &a::LayerMaskOp, _, cx| t.layer_mask_op(act.0, cx)))
        .on_action(cx.listener(|t, act: &a::LayerStyleOp, w, cx| t.layer_style_op(act.0, w, cx)))
        .on_action(cx.listener(|t, act: &a::NewAdjustment, w, cx| t.new_adjustment_layer(act.0, w, cx)))
        .on_action(cx.listener(|t, _: &a::PlaceEmbedded, _, cx| t.place_embedded(cx)))
        .on_action(cx.listener(|t, _: &a::SelectAllLayers, _, cx| {
            t.select_all_layers();
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::ConvertToFrame, _, cx| {
            t.convert_to_frame();
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::MergeShapeComponents, _, cx| {
            t.merge_shape_components();
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::RasterizeLayer, _, cx| {
            let li = t.doc.active;
            t.rasterize_layer(li);
            t.doc.mark_all_dirty();
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::GroupIsolation, _, cx| {
            let li = t.doc.active;
            if let crate::engine::layers::LayerKind::Group { pass_through, .. } = &mut t.doc.layers[li].kind {
                *pass_through = !*pass_through;
                t.doc.mark_all_dirty();
                cx.notify();
            }
        }))
        .on_action(cx.listener(|t, _: &a::ScreenMode, w, cx| t.cycle_screen_mode(w, cx)))
        .on_action(cx.listener(|t, _: &a::ZoomIn, _, cx| t.act_zoom(0, cx)))
        .on_action(cx.listener(|t, _: &a::ZoomOut, _, cx| t.act_zoom(1, cx)))
        .on_action(cx.listener(|t, _: &a::ZoomFit, _, cx| t.act_zoom(2, cx)))
        .on_action(cx.listener(|t, _: &a::ZoomActual, _, cx| t.act_zoom(3, cx)))
        .on_action(cx.listener(|t, _: &a::NewLayer, w, cx| t.act_layer("new", w, cx)))
        .on_action(cx.listener(|t, _: &a::DuplicateLayer, w, cx| t.act_layer("duplicate", w, cx)))
        .on_action(cx.listener(|t, _: &a::DeleteLayer, w, cx| t.act_layer("delete", w, cx)))
        .on_action(cx.listener(|t, _: &a::MergeDown, w, cx| t.act_layer("merge", w, cx)))
        .on_action(cx.listener(|t, _: &a::FlattenImage, w, cx| t.act_layer("flatten", w, cx)))
        .on_action(cx.listener(|t, _: &a::LayerUp, w, cx| t.act_layer("up", w, cx)))
        .on_action(cx.listener(|t, _: &a::LayerDown, w, cx| t.act_layer("down", w, cx)))
        .on_action(cx.listener(|t, _: &a::LayerViaCopy, w, cx| t.act_layer("via_copy", w, cx)))
        .on_action(cx.listener(|t, _: &a::LayerViaCut, w, cx| t.act_layer("via_cut", w, cx)))
        .on_action(cx.listener(|t, act: &a::RunFilter, w, cx| t.open_filter_dialog(act.0, w, cx)))
        .on_action(cx.listener(|t, _: &a::FlipHorizontal, _, cx| t.act_canvas("flip_h", cx)))
        .on_action(cx.listener(|t, _: &a::FlipVertical, _, cx| t.act_canvas("flip_v", cx)))
        .on_action(cx.listener(|t, _: &a::RotateCw, _, cx| t.act_canvas("rot_cw", cx)))
        .on_action(cx.listener(|t, _: &a::RotateCcw, _, cx| t.act_canvas("rot_ccw", cx)))
        .on_action(cx.listener(|t, _: &a::CropToSelection, _, cx| t.act_canvas("crop_sel", cx)))
        .on_action(cx.listener(|t, _: &a::RevealAll, _, cx| t.act_canvas("reveal_all", cx)))
        .on_action(cx.listener(|t, _: &a::CloseDocument, w, cx| t.close_doc(t.docs.uid, w, cx)))
        .on_action(cx.listener(|t, _: &a::ShowBrushes, _, cx| {
            t.toggle_dock(super::dock::DockTab::Brushes);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::ShowBrushSettings, _, cx| {
            t.toggle_dock(super::dock::DockTab::BrushSettings);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::ShowMeasurementLog, _, cx| {
            t.toggle_dock(super::dock::DockTab::Measurements);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::ShowChannels, _, cx| {
            t.toggle_dock(super::dock::DockTab::Channels);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::ShowNotes, _, cx| {
            t.toggle_dock(super::dock::DockTab::Notes);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::RecordMeasurements, _, cx| {
            t.record_measurements(cx);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::NextDocument, w, cx| t.cycle_doc(false, w, cx)))
        .on_action(cx.listener(|t, _: &a::PrevDocument, w, cx| t.cycle_doc(true, w, cx)))
        .on_action(cx.listener(|t, _: &a::ArrangeTabs, _, cx| {
            t.set_arrange(super::docs::Arrange::Tabs);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::ArrangeTwoUp, _, cx| {
            t.set_arrange(super::docs::Arrange::TwoUp);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::ArrangeTile, _, cx| {
            t.set_arrange(super::docs::Arrange::Tile);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::MatchZoom, _, cx| {
            t.match_views(super::docs::Match::Zoom);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::MatchLocation, _, cx| {
            t.match_views(super::docs::Match::Location);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::MatchRotation, _, cx| {
            t.match_views(super::docs::Match::Rotation);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::MatchAll, _, cx| {
            t.match_views(super::docs::Match::All);
            cx.notify();
        }))
        .on_action(cx.listener(|t, _: &a::ImageSize, w, cx| t.open_size_dialog(w, cx)))
        .on_action(cx.listener(|t, _: &a::Preferences, _, cx| t.open_preferences(cx)))
}

impl Render for Editor {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.sf = window.scale_factor();
        self.refresh_thumbs();
        if self.caf.as_ref().is_some_and(|c| c.pending) {
            self.caf_changed(cx);
        }
        if self.busy.is_some() {
            window.request_animation_frame();
        }
        let modal = self.dialog.as_ref().is_some_and(|d| !matches!(d.kind, DialogKind::Filter(_)));
        let root = div()
            .id("pixelforge")
            .key_context(a::CONTEXT)
            .track_focus(&self.focus)
            .size_full()
            .flex()
            .flex_col()
            .bg(rgb(BG))
            .text_color(rgb(TEXT))
            .text_size(px(12.))
            .on_key_down(cx.listener(Self::on_key_down))
            .on_key_up(cx.listener(Self::on_key_up))
            .on_drop(cx.listener(|this, paths: &ExternalPaths, _, cx| {
                if let Some(p) = paths.paths().first() {
                    // An image dropped onto a frame fills it; anything else opens.
                    if !this.drop_into_frame(p.clone(), cx) {
                        this.open_path(p.clone(), cx);
                    }
                }
            }));
        root_actions(root, cx)
            .when(self.screen_mode < 2, |el| {
                let bar = if self.vp_active() {
                    self.render_vp_options(window, cx)
                } else if self.pw_active() {
                    self.render_pw_options(window, cx)
                } else {
                    self.render_options(window, cx)
                };
                el.child(bar)
            })
            .child(
                div()
                    .flex()
                    .flex_row()
                    .flex_1()
                    .min_h_0()
                    .when(self.screen_mode < 2, |el| el.child(self.render_toolbox(cx)))
                    .child(div().flex_1().min_w_0().flex().flex_col().child(self.render_doc_tabs(cx)).child(self.render_canvas_area(cx)))
                    .when(self.screen_mode == 0, |el| {
                        let dock = self.render_dock(window, cx);
                        el.when_some(dock, |el, d| el.child(d)).child(self.render_side_panel(window, cx))
                    }),
            )
            .when(self.screen_mode == 0, |el| el.child(self.render_status()))
            .when(modal, |el| {
                el.child(
                    div()
                        .absolute()
                        .inset_0()
                        .flex()
                        .items_center()
                        .justify_center()
                        .bg(hsla(0., 0., 0., 0.45))
                        .child(self.render_dialog(cx)),
                )
            })
    }
}
