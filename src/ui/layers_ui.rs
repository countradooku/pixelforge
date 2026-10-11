//! The Layers panel and layer-model operations: masks, groups, clipping
//! masks, adjustment layers, smart objects and rasterizing.

use std::sync::Arc;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::slider::Slider;
use gpui_kit::component::select::Select;
use gpui_kit::component::{Icon, Sizable as _};
use gpui_kit::prelude::*;
use gpui_kit::{Context, ObjectFit, PathPromptOptions, SharedString, Window, div, img, px, rgb};

use crate::engine::document::Layer;
use crate::engine::filters::catalog::{Kind, Menu as FilterMenu};
use crate::engine::grid::TileGrid;
use crate::engine::layers::{LayerKind, LayerMask, SmartObject};
use crate::engine::masks;
use crate::engine::selection::MaskGrid;

use super::editor::{DialogKind, Editor};

const PANEL_HI: u32 = 0x2f2f32;
const BORDER: u32 = 0x3a3a3d;
const TEXT: u32 = 0xdcdcdc;
const DIM: u32 = 0x8e8e93;
const ACCENT: u32 = 0x2f6fde;

/// Where paint strokes go.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PaintTarget {
    Layer(usize),
    QuickMask,
    /// The layer mask of layer `li` (painted as gray).
    Mask(usize),
}

/// Adjustment kinds that are point operations (valid as adjustment layers).
pub fn adjustment_kinds() -> Vec<Kind> {
    Kind::all()
        .into_iter()
        .filter(|k| k.entry().menu == FilterMenu::Adjustments && !k.needs_stats())
        .filter(|k| {
            let ctx = crate::engine::filters::catalog::BuildCtx {
                fg: [0, 0, 0],
                bg: [255, 255, 255],
                area: crate::engine::grid::RectI::new(0, 0, 16, 16),
                stats: None,
                source: None,
                disp_map: None,
                layer_mask: None,
                doc_w: 16,
                doc_h: 16,
            };
            k.default_filter(&ctx).class() == crate::engine::filters::Class::Point
        })
        .collect()
}

impl Editor {
    /// Rows of the panel, top to bottom: (layer index, depth).
    fn layer_rows(&self) -> Vec<(usize, usize)> {
        fn walk(ed: &Editor, parent: Option<u64>, depth: usize, out: &mut Vec<(usize, usize)>) {
            let kids = crate::engine::composite::siblings(&ed.doc.layers, parent);
            for &i in kids.iter().rev() {
                out.push((i, depth));
                if ed.doc.layers[i].kind.folder() == Some(false) {
                    walk(ed, Some(ed.doc.layers[i].id), depth + 1, out);
                }
            }
        }
        let mut out = Vec::new();
        walk(self, None, 0, &mut out);
        out
    }

    fn mask_thumb(&mut self, li: usize) -> Option<Arc<gpui_kit::RenderImage>> {
        let l = &self.doc.layers[li];
        let m = l.mask.as_ref()?;
        let key = (l.id, m.version);
        if let Some((k, img)) = self.mask_thumbs.get(&l.id) {
            if *k == key.1 {
                return Some(img.clone());
            }
        }
        let side = (40.0 * self.sf) as u32;
        let top = m.level(m.mips.len() as u32);
        let gray = top.to_gray();
        let tmp = Layer::new(0, "m", gray);
        let img = super::viewport::thumbnail(&tmp, self.doc.w, self.doc.h, side);
        if let Some((_, old)) = self.mask_thumbs.insert(l.id, (key.1, img.clone())) {
            self.cache.queue_drop(old);
        }
        Some(img)
    }

    pub fn render_layers(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = self.layer_rows();
        let mut list = div().id("layer-list").flex_1().min_h_0().overflow_y_scroll().flex().flex_col().p_1().gap_0p5();
        for (li, depth) in rows {
            let mask_img = self.mask_thumb(li);
            let l = &self.doc.layers[li];
            let active = li == self.doc.active;
            let thumb = self.thumbs.get(&l.id).map(|(_, t)| t.clone());
            let kind_badge = match &l.kind {
                LayerKind::Pixel => None,
                LayerKind::Group { .. } => Some(gpui_kit::assets::IconName::Folder),
                LayerKind::Adjustment(_) => Some(gpui_kit::assets::IconName::Contrast),
                LayerKind::Smart(_) => Some(gpui_kit::assets::IconName::Image),
                LayerKind::Shape(_) => Some(gpui_kit::assets::IconName::Shapes),
                LayerKind::Text(_) => Some(gpui_kit::assets::IconName::Type),
                LayerKind::Frame(_) => Some(gpui_kit::assets::IconName::Frame),
                LayerKind::Artboard(_) => Some(gpui_kit::assets::IconName::Monitor),
            };
            let mut meta = Vec::new();
            if l.blend != crate::engine::blend::BlendMode::Normal || l.opacity < 1.0 {
                meta.push(format!("{} · {:.0}%", l.blend.name(), l.opacity * 100.));
            }
            match &l.kind {
                LayerKind::Group { pass_through, .. } => meta.push(if *pass_through { "Pass Through".into() } else { "Isolated".into() }),
                LayerKind::Adjustment(f) => meta.push(f.name().to_string()),
                LayerKind::Smart(s) => meta.push(format!("Smart Object · {}", s.name)),
                LayerKind::Frame(f) => meta.push(f.content.as_ref().map_or_else(|| "Frame · empty".to_string(), |c| format!("Frame · {}", c.name))),
                _ => {}
            }
            let id = l.id;
            let is_group = l.kind.folder().is_some();
            let collapsed = l.kind.folder() == Some(true);
            let selected = !active && self.doc.selected.contains(&id);
            let locked = l.background;
            let mask_active = active && self.mask_edit;
            let mut row = div()
                .id(SharedString::from(format!("layer-{id}")))
                .flex()
                .items_center()
                .gap_1p5()
                .pl(px(4.0 + depth as f32 * 14.0))
                .pr_1p5()
                .py_1()
                .rounded_md()
                .when(active, |d| d.bg(rgb(ACCENT).opacity(0.35)))
                .when(selected, |d| d.bg(rgb(ACCENT).opacity(0.18)))
                .when(!active && !selected, |d| d.hover(|s| s.bg(rgb(PANEL_HI))))
                .on_click(cx.listener(move |this, ev: &gpui_kit::ClickEvent, window, cx| {
                    this.mask_edit = false;
                    let m = ev.modifiers();
                    this.click_layer(li, m.platform, m.shift, window, cx);
                    if ev.click_count() >= 2 {
                        if this.doc.layers[li].background {
                            // Double-click the Background: make it a normal layer.
                            this.background_to_layer(li);
                        } else if matches!(this.doc.layers[li].kind, LayerKind::Adjustment(_)) {
                            // Double-click an adjustment layer to edit its settings.
                            this.edit_adjustment(li, window, cx);
                        } else if !matches!(this.doc.layers[li].kind, LayerKind::Artboard(_)) {
                            // Any other layer: Layer Style (like Photoshop).
                            this.open_layer_style(super::style_ui::Section::Blending, window, cx);
                        }
                    }
                }))
                .child(
                    Button::new(SharedString::from(format!("eye-{id}")))
                        .icon(Icon::new(if l.visible { gpui_kit::assets::IconName::Eye } else { gpui_kit::assets::IconName::EyeOff }))
                        .ghost()
                        .xsmall()
                        .on_click(cx.listener(move |this, _, _, cx| {
                            cx.stop_propagation();
                            this.toggle_visibility(li, cx);
                        })),
                );
            if is_group {
                row = row.child(
                    Button::new(SharedString::from(format!("col-{id}")))
                        .icon(Icon::new(if collapsed { gpui_kit::assets::IconName::ChevronRight } else { gpui_kit::assets::IconName::ChevronDown }))
                        .ghost()
                        .xsmall()
                        .on_click(cx.listener(move |this, _, _, cx| {
                            cx.stop_propagation();
                            match &mut this.doc.layers[li].kind {
                                LayerKind::Group { collapsed, .. } => *collapsed = !*collapsed,
                                LayerKind::Artboard(a) => Arc::make_mut(a).collapsed = !a.collapsed,
                                _ => {}
                            }
                            cx.notify();
                        })),
                );
            }
            if l.clip {
                row = row.child(div().text_color(rgb(DIM)).child("↳"));
            }
            row = row.child(
                div()
                    .size(px(34.))
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(rgb(0x111111))
                    .border_1()
                    .border_color(rgb(if active && !mask_active { ACCENT } else { BORDER }))
                    .when_some(thumb.filter(|_| !l.kind.is_container()), |d, t| d.child(img(t).size(px(32.)).object_fit(ObjectFit::Contain)))
                    .when_some(kind_badge.filter(|_| l.kind.is_container()), |d, b| d.child(Icon::new(b).text_color(rgb(TEXT)))),
            );
            if let Some(mi) = mask_img {
                let enabled = l.mask.as_ref().is_some_and(|m| m.enabled);
                row = row.child(
                    div()
                        .id(SharedString::from(format!("mask-{id}")))
                        .size(px(34.))
                        .flex()
                        .items_center()
                        .justify_center()
                        .border_1()
                        .border_color(rgb(if mask_active { ACCENT } else { BORDER }))
                        .when(!enabled, |d| d.opacity(0.35))
                        .on_click(cx.listener(move |this, ev: &gpui_kit::ClickEvent, window, cx| {
                            cx.stop_propagation();
                            this.select_layer(li, window, cx);
                            if ev.modifiers().shift {
                                this.layer_mask_op("toggle", cx);
                            } else {
                                this.mask_edit = true;
                                this.flash("Painting edits the layer mask (black hides, white reveals)");
                            }
                            cx.notify();
                        }))
                        .child(img(mi).size(px(32.)).object_fit(ObjectFit::Contain)),
                );
            }
            row = row.child(
                div()
                    .flex()
                    .flex_col()
                    .min_w_0()
                    .flex_1()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .when_some(kind_badge.filter(|_| !l.kind.is_container()), |d, b| d.child(Icon::new(b).text_color(rgb(DIM))))
                            .child(div().text_color(rgb(if l.visible { TEXT } else { DIM })).truncate().child(if locked { format!("{}  🔒", l.name) } else { l.name.clone() }))
                            .when(super::style_ui::has_fx(l), |d| {
                                // The "fx" badge: click to edit the layer's style.
                                d.child(
                                    div()
                                        .id(SharedString::from(format!("fx-{id}")))
                                        .flex_none()
                                        .px_1()
                                        .rounded_sm()
                                        .italic()
                                        .text_size(px(10.))
                                        .text_color(rgb(TEXT))
                                        .bg(rgb(BORDER))
                                        .hover(|s| s.bg(rgb(ACCENT)))
                                        .child("fx")
                                        .on_click(cx.listener(move |this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.select_layer(li, window, cx);
                                            this.open_layer_style(super::style_ui::Section::Blending, window, cx);
                                        })),
                                )
                            }),
                    )
                    .when(!meta.is_empty(), |d| d.child(div().text_size(px(10.)).text_color(rgb(DIM)).truncate().child(meta.join(" · ")))),
            );
            list = list.child(row);
        }
        let btn = |id: &'static str, icon: gpui_kit::assets::IconName, tip: &'static str| Button::new(id).icon(Icon::new(icon)).ghost().small().tooltip(tip);
        let op = self.doc.layers[self.doc.active].opacity;
        let fill = self.doc.layers[self.doc.active].fill;
        let mut adj_menu = div().relative().child(btn("l-adj", gpui_kit::assets::IconName::Contrast, "New adjustment layer").on_click(cx.listener(|t, _, _, cx| {
            t.adjust_menu = !t.adjust_menu;
            cx.notify();
        })));
        if self.adjust_menu {
            let mut m = div().id("adj-menu").occlude().w(px(220.)).p_1().rounded_md().bg(rgb(PANEL_HI)).border_1().border_color(rgb(BORDER)).shadow_lg().flex().flex_col();
            for k in adjustment_kinds() {
                m = m.child(
                    div()
                        .id(SharedString::from(format!("adj-{k:?}")))
                        .px_2()
                        .py_1()
                        .rounded_sm()
                        .hover(|s| s.bg(rgb(BORDER)))
                        .text_color(rgb(TEXT))
                        .child(k.entry().title)
                        .on_click(cx.listener(move |t, _, w, cx| {
                            t.adjust_menu = false;
                            t.new_adjustment_layer(k, w, cx);
                        })),
                );
            }
            adj_menu = adj_menu.child(gpui_kit::deferred(div().absolute().bottom(px(30.)).right(px(0.)).child(m)).with_priority(2));
        }
        div()
            .flex_1()
            .min_h(px(230.))
            .flex()
            .flex_col()
            .child(super::render::section_title("Layers"))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_2()
                    .p_2()
                    .border_b_1()
                    .border_color(rgb(BORDER))
                    .child(Select::new(&self.ui.blend).small())
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(div().text_color(rgb(DIM)).child("Opacity"))
                            .child(div().w(px(150.)).child(Slider::new(&self.ui.layer_opacity)))
                            .child(div().w(px(64.)).text_color(rgb(TEXT)).child(format!("{:.0}%", op * 100.))),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(div().text_color(rgb(DIM)).child("Fill"))
                            .child(div().w(px(150.)).ml(px(19.)).child(Slider::new(&self.ui.layer_fill)))
                            .child(div().w(px(64.)).text_color(rgb(TEXT)).child(format!("{:.0}%", fill * 100.))),
                    ),
            )
            .child(list)
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_end()
                    .gap_0p5()
                    .p_1()
                    .border_t_1()
                    .border_color(rgb(BORDER))
                    .child(btn("l-clip", gpui_kit::assets::IconName::CornerDownRight, "Create / release clipping mask (⌥⌘G)").on_click(cx.listener(|t, _, _, cx| t.toggle_clip(cx))))
                    .child(btn("l-mask", gpui_kit::assets::IconName::SquareDot, "Add layer mask (reveals the selection); Shift-click a mask to disable it").on_click(cx.listener(|t, _, _, cx| t.layer_mask_op("add", cx))))
                    .child(adj_menu)
                    .child(btn("l-group", gpui_kit::assets::IconName::FolderOpen, "Group layer (⌘G)").on_click(cx.listener(|t, _, w, cx| t.group_layer(w, cx))))
                    .child(btn("l-up", gpui_kit::assets::IconName::ArrowUp, "Bring forward (⌘])").on_click(cx.listener(|t, _, w, cx| t.act_layer("up", w, cx))))
                    .child(btn("l-down", gpui_kit::assets::IconName::ArrowDown, "Send backward (⌘[)").on_click(cx.listener(|t, _, w, cx| t.act_layer("down", w, cx))))
                    .child(btn("l-merge", gpui_kit::assets::IconName::Merge, "Merge down (⌘E)").on_click(cx.listener(|t, _, w, cx| t.act_layer("merge", w, cx))))
                    .child(btn("l-dup", gpui_kit::assets::IconName::Copy, "Duplicate layer").on_click(cx.listener(|t, _, w, cx| t.act_layer("duplicate", w, cx))))
                    .child(btn("l-new", gpui_kit::assets::IconName::Plus, "New layer (⇧⌘N)").on_click(cx.listener(|t, _, w, cx| t.act_layer("new", w, cx))))
                    .child(btn("l-del", gpui_kit::assets::IconName::Trash, "Delete layer").on_click(cx.listener(|t, _, w, cx| t.act_layer("delete", w, cx)))),
            )
    }

    // ----- operations ---------------------------------------------------------------------

    /// Layer mask operations: add / toggle / apply / delete / invert.
    pub fn layer_mask_op(&mut self, what: &str, cx: &mut Context<Self>) {
        let li = self.doc.active;
        let (w, h) = (self.doc.w, self.doc.h);
        match what {
            "add" => {
                if self.doc.layers[li].mask.is_some() {
                    self.flash("This layer already has a mask");
                    return;
                }
                let m = match &self.doc.selection {
                    Some(s) => MaskGrid::from_selection(s, w, h),
                    None => MaskGrid::full(w, h),
                };
                self.doc.push_undo("Add Layer Mask");
                self.doc.layers[li].mask = Some(LayerMask::new(m));
                self.mask_edit = true;
            }
            "toggle" => {
                if let Some(m) = &self.doc.layers[li].mask {
                    let on = !m.enabled;
                    self.doc.push_undo(if on { "Enable Layer Mask" } else { "Disable Layer Mask" });
                    self.doc.layers[li].mask.as_mut().unwrap().enabled = on;
                }
            }
            "delete" => {
                if self.doc.layers[li].mask.is_some() {
                    self.doc.push_undo("Delete Layer Mask");
                    self.doc.layers[li].mask = None;
                    self.mask_edit = false;
                }
            }
            "invert" => {
                if let Some(m) = &self.doc.layers[li].mask {
                    let inv = m.grid.invert();
                    self.doc.push_undo("Invert Layer Mask");
                    let lm = self.doc.layers[li].mask.as_mut().unwrap();
                    lm.replace(inv);
                }
            }
            "apply" => {
                let Some(m) = self.doc.layers[li].mask.clone() else { return };
                if !matches!(self.doc.layers[li].kind, LayerKind::Pixel) {
                    self.flash("Rasterize the layer before applying its mask");
                    return;
                }
                let sel = crate::engine::selection::Selection { shape: crate::engine::selection::Shape::Mask(Arc::new(m.grid.invert())), inverted: false };
                let grid = crate::engine::ops::clear(&self.doc.layers[li].grid, Some(&sel));
                self.doc.push_undo("Apply Layer Mask");
                self.doc.layers[li].mask = None;
                self.doc.set_layer_grid(li, grid);
                self.mask_edit = false;
            }
            _ => {}
        }
        self.doc.mark_all_dirty();
        cx.notify();
    }

    pub fn toggle_clip(&mut self, cx: &mut Context<Self>) {
        let li = self.doc.active;
        let has_base = crate::engine::composite::siblings(&self.doc.layers, self.doc.layers[li].parent).first() != Some(&li);
        if !has_base && !self.doc.layers[li].clip {
            self.flash("A clipping mask needs a layer below it");
            return;
        }
        let on = !self.doc.layers[li].clip;
        self.doc.push_undo(if on { "Create Clipping Mask" } else { "Release Clipping Mask" });
        self.doc.layers[li].clip = on;
        self.doc.mark_all_dirty();
        cx.notify();
    }

    /// Puts the active layer (with the layers clipped to it) into a new
    /// group (⌘G).
    pub fn group_layer(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let li = self.doc.active;
        // Several selected sibling layers (not folders): they move into one
        // new group placed where the topmost of them was.
        let parent = self.doc.layers[li].parent;
        let sel: Vec<usize> = self.selected_layers().into_iter().filter(|&i| self.doc.layers[i].parent == parent && !self.doc.layers[i].background).collect();
        if sel.len() > 1 && sel.iter().all(|&i| self.doc.layers[i].kind.folder().is_none()) {
            self.doc.push_undo("Group Layers");
            let gid = self.doc.next_layer_id();
            let n_groups = self.doc.layers.iter().filter(|l| matches!(l.kind, LayerKind::Group { .. })).count();
            let top = *sel.iter().max().unwrap();
            let mut moved: Vec<Layer> = Vec::new();
            for &i in sel.iter().rev() {
                moved.insert(0, self.doc.layers.remove(i));
            }
            for l in &mut moved {
                l.parent = Some(gid);
            }
            let at = top + 1 - sel.len();
            let mut g = Layer::new(gid, format!("Group {}", n_groups + 1), TileGrid::new(self.doc.w, self.doc.h));
            g.kind = LayerKind::Group { pass_through: true, collapsed: false };
            g.parent = parent;
            let n = moved.len();
            for (k, l) in moved.into_iter().enumerate() {
                self.doc.layers.insert(at + k, l);
            }
            self.doc.layers.insert(at + n, g);
            self.doc.selected.clear();
            self.doc.mark_all_dirty();
            self.select_layer(at + n, window, cx);
            return;
        }
        self.doc.push_undo("Group Layers");
        let parent = self.doc.layers[li].parent;
        let gid = self.doc.next_layer_id();
        // The block: the layer, everything nested in it (directly below a
        // group) and the clipped layers stacked on it.
        let id = self.doc.layers[li].id;
        let mut start = li;
        while start > 0 && descends_from(&self.doc.layers, start - 1, id) {
            start -= 1;
        }
        let mut end = li + 1;
        while end < self.doc.layers.len() && self.doc.layers[end].clip && self.doc.layers[end].parent == parent {
            end += 1;
        }
        let n_groups = self.doc.layers.iter().filter(|l| matches!(l.kind, LayerKind::Group { .. })).count();
        let mut g = Layer::new(gid, format!("Group {}", n_groups + 1), TileGrid::new(self.doc.w, self.doc.h));
        g.kind = LayerKind::Group { pass_through: true, collapsed: false };
        g.parent = parent;
        for i in start..end {
            if self.doc.layers[i].parent == parent {
                self.doc.layers[i].parent = Some(gid);
            }
        }
        self.doc.layers.insert(end, g);
        self.doc.mark_all_dirty();
        self.select_layer(end, window, cx);
    }

    /// Ungroups the active group (⇧⌘G).
    pub fn ungroup(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let li = self.doc.active;
        let LayerKind::Group { .. } = self.doc.layers[li].kind else { return };
        self.doc.push_undo("Ungroup Layers");
        let gid = self.doc.layers[li].id;
        let parent = self.doc.layers[li].parent;
        for l in &mut self.doc.layers {
            if l.parent == Some(gid) {
                l.parent = parent;
            }
        }
        self.doc.layers.remove(li);
        self.doc.active = li.saturating_sub(1).min(self.doc.layers.len() - 1);
        self.doc.mark_all_dirty();
        self.select_layer(self.doc.active, window, cx);
    }

    /// Inserts an adjustment layer above the active layer and opens its
    /// dialog; the dialog edits the layer live (non-destructively).
    pub fn new_adjustment_layer(&mut self, kind: Kind, window: &mut Window, cx: &mut Context<Self>) {
        let ctx = self.adjust_ctx();
        let f = kind.default_filter(&ctx);
        self.doc.push_undo(format!("New {} Layer", kind.entry().title));
        let at = self.doc.insert_layer(kind.entry().title, TileGrid::new(self.doc.w, self.doc.h));
        self.doc.layers[at].kind = LayerKind::Adjustment(f);
        self.doc.layers[at].parent = self.doc.layers[at.saturating_sub(1)].parent.filter(|_| at > 0);
        if let Some(sel) = &self.doc.selection {
            self.doc.layers[at].mask = Some(LayerMask::new(MaskGrid::from_selection(sel, self.doc.w, self.doc.h)));
        }
        self.adjust_target = Some(self.doc.layers[at].id);
        self.open_filter_dialog(kind, window, cx);
    }

    /// Reopens the dialog of an adjustment layer; it edits the layer live.
    pub fn edit_adjustment(&mut self, li: usize, window: &mut Window, cx: &mut Context<Self>) {
        let LayerKind::Adjustment(f) = &self.doc.layers[li].kind else { return };
        let Some(kind) = adjustment_kind_of(f) else { return };
        self.doc.push_undo(format!("Edit {}", kind.entry().title));
        self.adjust_target = Some(self.doc.layers[li].id);
        let saved = self.doc.layers[li].adj_params.clone();
        self.open_filter_dialog(kind, window, cx);
        // Resume from the layer's current settings.
        if let (Some(p), Some(d)) = (saved, &mut self.dialog) {
            d.flag = p.2;
            let states: Vec<_> = d.sliders.iter().map(|s| s.state.clone()).collect();
            for (st, &v) in states.iter().zip(&p.1) {
                st.update(cx, |s, cx| s.set_value(v, window, cx));
            }
            self.update_preview(cx);
        }
    }

    pub fn adjust_ctx(&self) -> crate::engine::filters::catalog::BuildCtx<'static> {
        crate::engine::filters::catalog::BuildCtx {
            fg: self.fg,
            bg: self.bg,
            area: self.doc.bounds(),
            stats: None,
            source: None,
            disp_map: None,
            layer_mask: None,
            doc_w: self.doc.w,
            doc_h: self.doc.h,
        }
    }

    /// Keeps a dialog's filter on its adjustment layer (live).
    pub fn adjustment_dialog_update(&mut self, filter: crate::engine::filters::Filter, params: (Kind, Vec<f32>, bool)) -> bool {
        let Some(id) = self.adjust_target else { return false };
        if let Some(l) = self.doc.layers.iter_mut().find(|l| l.id == id) {
            l.kind = LayerKind::Adjustment(filter);
            l.adj_params = Some(Arc::new(params));
            l.version += 1;
            self.doc.mark_all_dirty();
            return true;
        }
        false
    }

    /// File ▸ Place Embedded: a smart object fitted inside the canvas.
    pub fn place_embedded(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files: true, directories: false, multiple: false, prompt: Some("Place".into()) });
        cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else { return };
            let Some(path) = paths.into_iter().next() else { return };
            let _ = this.update(cx, |ed, cx| ed.place_path(path, cx));
        })
        .detach();
    }

    pub fn place_path(&mut self, path: std::path::PathBuf, cx: &mut Context<Self>) {
        let (dw, dh) = (self.doc.w as f32, self.doc.h as f32);
        let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "Placed".into());
        self.spawn_job(format!("Placing {name}"), cx, move |_| {
            let res = crate::engine::io::open_image(&path).map(|g| {
                let (sw, sh) = (g.w as f32, g.h as f32);
                let k = (dw * 0.8 / sw).min(dh * 0.8 / sh).min(1.0);
                let m = [k, 0.0, (dw - sw * k) * 0.5, 0.0, k, (dh - sh * k) * 0.5, 0.0, 0.0, 1.0];
                SmartObject::new(name.clone(), g, m)
            });
            Box::new(move |ed: &mut Editor, _| match res {
                Ok(so) => ed.add_smart_layer(so),
                Err(e) => ed.flash(format!("Error: {e:#}")),
            })
        });
    }

    /// Active layer plus the other selected layers, as indices (bottom to top).
    pub fn selected_layers(&self) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.doc.layers.len()).filter(|&i| i == self.doc.active || self.doc.selected.contains(&self.doc.layers[i].id)).collect();
        v.dedup();
        v
    }

    /// A click on a Layers-panel row: plain click selects one layer, ⌘
    /// toggles it in the selection, Shift selects the range from the active row.
    pub fn click_layer(&mut self, li: usize, cmd: bool, shift: bool, window: &mut Window, cx: &mut Context<Self>) {
        let id = self.doc.layers[li].id;
        let active_id = self.doc.layers.get(self.doc.active).map(|l| l.id);
        if cmd {
            if Some(id) == active_id {
                // Deselecting the key layer: another selected one takes over.
                if let Some(next) = self.doc.selected.pop() {
                    if let Some(ni) = self.doc.layers.iter().position(|l| l.id == next) {
                        self.select_layer(ni, window, cx);
                    }
                }
            } else if let Some(k) = self.doc.selected.iter().position(|&s| s == id) {
                self.doc.selected.remove(k);
            } else {
                if let Some(a) = active_id {
                    if !self.doc.selected.contains(&a) {
                        self.doc.selected.push(a);
                    }
                }
                self.select_layer(li, window, cx);
                self.doc.selected.retain(|&s| s != id);
            }
        } else if shift {
            let rows: Vec<usize> = self.layer_rows().into_iter().map(|(i, _)| i).collect();
            let (a, b) = (rows.iter().position(|&i| i == self.doc.active), rows.iter().position(|&i| i == li));
            if let (Some(a), Some(b)) = (a, b) {
                let (lo, hi) = (a.min(b), a.max(b));
                self.doc.selected = rows[lo..=hi].iter().map(|&i| self.doc.layers[i].id).filter(|&x| x != id).collect();
            }
            self.select_layer(li, window, cx);
        } else {
            self.doc.selected.clear();
            self.select_layer(li, window, cx);
        }
        cx.notify();
    }

    /// Select ▸ All Layers (everything but the Background).
    pub fn select_all_layers(&mut self) {
        self.doc.selected = self.doc.layers.iter().filter(|l| !l.background).map(|l| l.id).collect();
        if let Some(top) = (0..self.doc.layers.len()).rev().find(|&i| !self.doc.layers[i].background) {
            self.doc.active = top;
        }
    }

    /// Layer ▸ New ▸ Layer from Background ("Layer 0", unlocked).
    pub fn background_to_layer(&mut self, li: usize) {
        if !self.doc.layers[li].background {
            return;
        }
        self.doc.push_undo("Layer from Background");
        self.doc.layers[li].background = false;
        self.doc.layers[li].name = "Layer 0".into();
        self.doc.mark_all_dirty();
    }

    pub fn add_smart_layer(&mut self, so: SmartObject) {
        // Placing into an active Frame fills it (content scaled to cover).
        if let Some(LayerKind::Frame(_)) = self.doc.layers.get(self.doc.active).map(|l| &l.kind) {
            let li = self.doc.active;
            self.place_into_frame(li, so);
            return;
        }
        self.doc.push_undo("Place Embedded");
        let name = so.name.clone();
        let at = self.doc.insert_layer(name, TileGrid::new(self.doc.w, self.doc.h));
        let l = std::mem::replace(&mut self.doc.layers[at], Layer::new(0, "", TileGrid::new(1, 1)));
        let kind = LayerKind::Smart(Arc::new(so));
        self.doc.layers[at] = l.with_kind(kind);
        self.doc.mark_all_dirty();
    }

    /// Sets frame `li`'s content to `so`, scaled to cover the frame.
    pub fn place_into_frame(&mut self, li: usize, so: SmartObject) {
        let LayerKind::Frame(fr) = &self.doc.layers[li].kind else { return };
        let mut nf = (**fr).clone();
        let old = fr.bounds();
        self.doc.push_undo("Place into Frame");
        nf.content = Some(Arc::new(nf.fitted(&so, true)));
        self.set_frame(li, nf, old);
    }

    pub fn set_frame(&mut self, li: usize, nf: crate::engine::frame::FrameLayer, old: crate::engine::grid::RectI) {
        let kind = LayerKind::Frame(Arc::new(nf));
        let dirty = crate::engine::layers::rerender(&mut self.doc.layers[li].grid, Some(old), &kind);
        self.doc.layers[li].kind = kind;
        self.doc.layer_pixels_changed(li, &[dirty]);
    }

    /// An image file dropped on the canvas: when the cursor is over a frame
    /// (topmost visible), it becomes that frame's content.
    pub fn drop_into_frame(&mut self, path: std::path::PathBuf, cx: &mut Context<Self>) -> bool {
        if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("pforge")) {
            return false;
        }
        let Some((mx, my)) = self.mouse_img else { return false };
        let q = crate::engine::vector::p(mx, my);
        let Some(li) = (0..self.doc.layers.len()).rev().find(|&i| {
            let l = &self.doc.layers[i];
            l.visible && matches!(&l.kind, LayerKind::Frame(f) if crate::engine::vector::contains(q, &f.outline(), crate::engine::vector::FillRule::NonZero))
        }) else {
            return false;
        };
        let id = self.doc.layers[li].id;
        let name = path.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "Image".into());
        self.spawn_job(format!("Placing {name}"), cx, move |_| {
            let result = crate::engine::io::open_image(&path);
            Box::new(move |ed: &mut Editor, _| match result {
                Ok(g) => {
                    if let Some(li) = ed.doc.layers.iter().position(|l| l.id == id) {
                        ed.place_into_frame(li, SmartObject::new(name, g, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
                        ed.doc.active = li;
                    }
                }
                Err(e) => ed.flash(format!("Error: {e:#}")),
            })
        });
        true
    }

    /// Layer ▸ Convert to Frame: a shape layer's outline or a text layer's
    /// glyphs become an (empty) frame of that shape.
    pub fn convert_to_frame(&mut self) {
        use crate::engine::frame::FrameLayer;
        use crate::engine::shape::ShapeGeom;
        use crate::engine::vector::{Anchor, Path, SubPath};
        let li = self.doc.active;
        let frame = match &self.doc.layers[li].kind {
            LayerKind::Shape(s) if s.parts.is_empty() => {
                let mut f = FrameLayer::new(s.geom.clone());
                f.shape.rot = s.rot;
                f.shape.xf = s.xf;
                f
            }
            LayerKind::Shape(s) => FrameLayer::new(s.merged().geom),
            LayerKind::Text(t) => {
                let path = Path { subs: t.outlines().into_iter().map(|pl| SubPath { anchors: pl.pts.into_iter().map(Anchor::corner).collect(), closed: true }).collect() };
                FrameLayer::new(ShapeGeom::Path(path))
            }
            _ => {
                self.flash("Convert to Frame works on shape and text layers");
                return;
            }
        };
        self.doc.push_undo("Convert to Frame");
        let old = self.doc.layers[li].kind.content_bounds().unwrap_or(crate::engine::grid::RectI::EMPTY);
        self.set_frame(li, frame, old);
        let n = self.doc.layers.iter().filter(|l| matches!(l.kind, LayerKind::Frame(_))).count();
        self.doc.layers[li].name = format!("Frame {n}");
    }

    /// Converts a vector / smart layer to plain pixels (its cached raster).
    pub fn rasterize_layer(&mut self, li: usize) {
        if self.doc.layers[li].kind.is_vector() {
            self.doc.push_undo("Rasterize Layer");
            self.doc.layers[li].kind = LayerKind::Pixel;
        }
    }

    /// Makes sure layer `li` can take pixel edits: vector layers are
    /// rasterized (Photoshop asks; we say so in the status bar), containers
    /// refuse.
    pub fn ensure_pixels(&mut self, li: usize) -> bool {
        match &self.doc.layers[li].kind {
            LayerKind::Pixel => true,
            k if k.is_container() => {
                let name = k.name();
                self.flash(format!("Pixel tools need a pixel layer (this is a {name} layer)"));
                false
            }
            _ => {
                self.rasterize_layer(li);
                self.flash("Layer rasterized for painting");
                true
            }
        }
    }

    /// Writes painted gray tiles back into the layer mask.
    pub fn mask_painted(&mut self, li: usize, gray: &TileGrid, rects: &[crate::engine::grid::RectI]) {
        let Some(m) = self.doc.layers[li].mask.as_mut() else { return };
        let mut g = m.grid.clone();
        let fresh = MaskGrid::from_gray(gray);
        for r in rects {
            let (tx0, ty0, tx1, ty1) = r.tile_span(g.cols, g.rows);
            for ty in ty0..ty1 {
                for tx in tx0..tx1 {
                    let i = g.idx(tx, ty);
                    g.tiles[i] = fresh.tiles[i].clone();
                }
            }
        }
        m.replace(g);
        self.doc.dirty.rects.extend_from_slice(rects);
        let _ = masks::SelOp::New;
    }
}

fn descends_from(layers: &[Layer], i: usize, id: u64) -> bool {
    let mut p = layers[i].parent;
    while let Some(pid) = p {
        if pid == id {
            return true;
        }
        p = layers.iter().find(|l| l.id == pid).and_then(|l| l.parent);
    }
    false
}

/// Opens the dialog of an existing adjustment layer (double-click / menu).
pub fn adjustment_kind_of(f: &crate::engine::filters::Filter) -> Option<Kind> {
    adjustment_kinds().into_iter().find(|k| k.entry().title == f.name())
}

#[allow(dead_code)]
fn _dialog_kind_is_used(d: &DialogKind) -> bool {
    matches!(d, DialogKind::Filter(_))
}
