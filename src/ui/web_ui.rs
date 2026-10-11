//! Slices panel (Slice Options: name, URL, target, alt text; layer-based
//! slices; Divide Slice) and the Save for Web workspace: PNG-8 (colours,
//! dither, transparency), PNG-24, JPEG (quality) or WebP, with a live preview
//! of the optimized slice, its size and encode time; Save writes every slice
//! plus an HTML table page.

use std::sync::Arc;
use std::time::Instant;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::input::{Input, InputState};
use gpui_kit::component::Sizable as _;
use gpui_kit::prelude::*;
use gpui_kit::{Context, Div, Entity, PathPromptOptions, RenderImage, Window, div, px, rgb};

use crate::engine::annot::{self, Slice};
use crate::engine::grid::RectI;
use crate::engine::slices;
use crate::engine::webexport::{self, WebFormat};

use super::editor::{Editor, Tool};
use super::options::{label, segmented, spec, toggle};

/// Slice Options fields of the selected slice: (slice index, name, URL, target, alt).
pub struct SliceFields {
    pub index: usize,
    pub inputs: [Entity<InputState>; 4],
    pub last: [String; 4],
}

pub struct WebState {
    /// 0 PNG-8, 1 PNG-24, 2 JPEG, 3 WebP.
    pub kind: u8,
    pub colors: u16,
    pub dither: bool,
    pub transparency: bool,
    pub quality: u8,
    pub preview: Option<(Arc<RenderImage>, usize, f32, RectI)>,
    pub stale: bool,
}

impl Default for WebState {
    fn default() -> Self {
        WebState { kind: 0, colors: 128, dither: true, transparency: true, quality: 60, preview: None, stale: true }
    }
}

impl WebState {
    pub fn format(&self) -> WebFormat {
        match self.kind {
            0 => WebFormat::Png8 { colors: self.colors, dither: self.dither, transparency: self.transparency },
            1 => WebFormat::Png24 { transparency: self.transparency },
            2 => WebFormat::Jpeg { quality: self.quality },
            _ => WebFormat::Webp,
        }
    }
}

/// A BGRA preview image (≤ 280 px) of straight RGBA bytes.
fn preview_of(rgba: &[u8], w: u32, h: u32) -> Arc<RenderImage> {
    let k = (w.max(h) as f32 / 280.0).max(1.0);
    let (tw, th) = (((w as f32 / k) as u32).max(1), ((h as f32 / k) as u32).max(1));
    let mut out = vec![0u8; (tw * th * 4) as usize];
    for y in 0..th {
        for x in 0..tw {
            let (sx, sy) = ((((x as f32 + 0.5) * k) as u32).min(w - 1), (((y as f32 + 0.5) * k) as u32).min(h - 1));
            let i = ((sy * w + sx) * 4) as usize;
            let a = rgba[i + 3] as u32;
            // Over a light checker-free grey so transparency reads.
            let bgc = 230u32;
            let c = |v: u8| ((v as u32 * a + bgc * (255 - a)) / 255) as u8;
            let o = ((y * tw + x) * 4) as usize;
            out[o..o + 4].copy_from_slice(&[c(rgba[i + 2]), c(rgba[i + 1]), c(rgba[i]), 255]);
        }
    }
    crate::ui::render_image(vec![image::Frame::new(image::RgbaImage::from_raw(tw, th, out).expect("preview"))])
}

impl Editor {
    /// User slices with layer-based rects resolved.
    pub fn resolved_slices(&self) -> Vec<Slice> {
        slices::resolved(&self.doc.ann.slices, &self.doc.layers, self.doc.w, self.doc.h)
    }

    pub fn slice_panel_visible(&self) -> bool {
        matches!(self.tool, Tool::Slice | Tool::SliceSelect) || self.web.is_some()
    }

    /// Recreates the Slice Options fields when the selected slice changes.
    fn sync_slice_fields(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let sel = self.nopts.slice_sel.filter(|&i| i < self.doc.ann.slices.len());
        match (sel, &self.slice_fields) {
            (None, _) => self.slice_fields = None,
            (Some(i), Some(f)) if f.index == i => {
                // Edits made in the fields flow back into the slice.
                let vals: [String; 4] = std::array::from_fn(|k| f.inputs[k].read(cx).value().to_string());
                if vals != f.last {
                    let s = &mut self.doc.ann.slices[i];
                    (s.name, s.url, s.target, s.alt) = (vals[0].clone(), vals[1].clone(), vals[2].clone(), vals[3].clone());
                    if let Some(f) = &mut self.slice_fields {
                        f.last = vals;
                    }
                    self.doc.modified = true;
                }
            }
            (Some(i), _) => {
                let s = self.doc.ann.slices[i].clone();
                let vals = [s.name, s.url, s.target, s.alt];
                let ph = ["Name", "URL (https://…)", "Target (_blank, _self…)", "Alt text"];
                let inputs = std::array::from_fn(|k| cx.new(|cx| InputState::new(window, cx).placeholder(ph[k]).default_value(vals[k].clone())));
                self.slice_fields = Some(SliceFields { index: i, inputs, last: vals });
            }
        }
    }

    /// Slice ▸ New Layer-based Slice: a slice that follows the active layer.
    pub fn layer_based_slice(&mut self) {
        let li = self.doc.active;
        let id = self.doc.layers[li].id;
        let Some(r) = slices::layer_bounds(&self.doc.layers, id) else {
            self.flash("The active layer is empty");
            return;
        };
        self.doc.push_undo("Layer-based Slice");
        let name = format!("{}", self.doc.layers[li].name.replace(' ', "_"));
        self.doc.ann.slices.push(Slice { layer: Some(id), ..Slice::new(name, r) });
        self.nopts.slice_sel = Some(self.doc.ann.slices.len() - 1);
    }

    /// Divide Slice: the selected slice becomes `nx × ny` slices.
    pub fn divide_slice(&mut self, nx: u32, ny: u32) {
        let Some(i) = self.nopts.slice_sel.filter(|&i| i < self.doc.ann.slices.len()) else {
            self.flash("Select a user slice to divide");
            return;
        };
        let r = self.resolved_slices()[i].rect;
        self.doc.push_undo("Divide Slice");
        let base = self.doc.ann.slices.remove(i);
        for (k, q) in slices::divide(r, nx, ny).into_iter().enumerate() {
            self.doc.ann.slices.insert(i + k, Slice { name: format!("{}_{:02}", base.name, k + 1), rect: q, layer: None, ..base.clone() });
        }
        self.nopts.slice_sel = Some(i);
        self.slice_fields = None;
    }

    pub fn open_save_for_web(&mut self) {
        self.web = Some(WebState::default());
    }

    /// Re-encodes the preview (the selected slice, else the first user slice,
    /// else a central 512² crop).
    fn refresh_web_preview(&mut self) {
        let Some(st) = &self.web else { return };
        if !st.stale {
            return;
        }
        let fmt = st.format();
        let user = self.resolved_slices();
        let full = self.doc.bounds();
        let r = self
            .nopts
            .slice_sel
            .and_then(|i| user.get(i))
            .or(user.first())
            .map(|s| s.rect)
            .unwrap_or_else(|| {
                let (cx_, cy) = (full.x1 / 2, full.y1 / 2);
                RectI::new(cx_ - 256, cy - 256, cx_ + 256, cy + 256).intersect(&full)
            });
        if r.is_empty() {
            return;
        }
        let rgba = annot::region_rgba(&self.doc.layers, r);
        let t0 = Instant::now();
        let enc = webexport::encode(&rgba, r.w() as u32, r.h() as u32, fmt, self.bg);
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        let st = self.web.as_mut().unwrap();
        st.stale = false;
        match enc.ok().and_then(|bytes| image::load_from_memory(&bytes).ok().map(|img| (img.to_rgba8(), bytes.len()))) {
            Some((img, size)) => st.preview = Some((preview_of(img.as_raw(), img.width(), img.height()), size, ms, r)),
            None => st.preview = None,
        }
    }

    pub fn save_for_web(&mut self, cx: &mut Context<Self>) {
        let Some(st) = &self.web else { return };
        let fmt = st.format();
        let (layers, user, w, h, bg) = (self.doc.layers.clone(), self.doc.ann.slices.clone(), self.doc.w, self.doc.h, self.bg);
        let base = self.doc.path.as_ref().and_then(|p| p.file_stem()).map_or_else(|| "page".to_string(), |s| s.to_string_lossy().into_owned());
        let rx = cx.prompt_for_paths(PathPromptOptions { files: false, directories: true, multiple: false, prompt: Some("Save".into()) });
        cx.spawn(async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else { return };
            let Some(dir) = paths.into_iter().next() else { return };
            let _ = this.update(cx, |ed, cx| {
                ed.spawn_job("Save for Web", cx, move |_| {
                    let t0 = Instant::now();
                    let res = slices::save_for_web(&layers, w, h, &user, fmt, bg, &dir, &base);
                    let ms = t0.elapsed().as_secs_f32() * 1000.0;
                    Box::new(move |ed: &mut Editor, _| match res {
                        Ok(files) => ed.flash(format!("Saved {} files (HTML + images) in {ms:.0} ms", files.len())),
                        Err(e) => ed.flash(format!("Error: {e}")),
                    })
                });
            });
        })
        .detach();
    }

    pub fn render_slice_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        self.sync_slice_fields(window, cx);
        let mut col = div().flex().flex_col().gap_1p5().p_2().border_b_1().border_color(rgb(0x3a3a3d)).child(super::render::section_title("Slices"));
        let n_user = self.doc.ann.slices.len();
        let total = annot::all_slices(&self.resolved_slices(), self.doc.w, self.doc.h).len();
        col = col.child(label(format!("{n_user} user slices · {total} with auto slices")));
        if let Some(f) = &self.slice_fields {
            let i = f.index;
            let based = self.doc.ann.slices[i].layer.and_then(|id| self.doc.layers.iter().find(|l| l.id == id)).map(|l| l.name.clone());
            col = col.child(label(format!("Slice Options — user slice {} of {n_user}{}", i + 1, based.map_or(String::new(), |n| format!(" (layer-based: {n})")))));
            for (k, name) in ["Name", "URL", "Target", "Alt"].iter().enumerate() {
                col = col.child(div().flex().items_center().gap_1().child(div().w(px(46.)).child(label(*name))).child(div().flex_1().child(Input::new(&f.inputs[k]).small())));
            }
        } else {
            col = col.child(label("Select a slice (Slice Select tool) for its options"));
        }
        let (nx, ny) = self.nopts.slice_divide;
        if self.web.is_some() {
            // Save for Web takes the panel's room (Divide Slice comes back on Done).
            return col.child(self.render_web_panel(window, cx));
        }
        col = col
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .gap_1()
                    .child(Button::new("sl-layer").label("New Layer-based Slice").xsmall().on_click(cx.listener(|e, _, _, cx| {
                        e.layer_based_slice();
                        cx.notify();
                    })))
                    .child(Button::new("sl-sfw").label("Save for Web…").xsmall().primary().on_click(cx.listener(|e, _, _, cx| {
                        e.open_save_for_web();
                        cx.notify();
                    }))),
            )
            .child(label(format!("Divide Slice: {nx} × {ny}")))
            .child(self.opt_slider("Columns", "sl.nx", spec(1.0, 12.0, 1.0, |e| e.nopts.slice_divide.0 as f32, |e, v| e.nopts.slice_divide.0 = v as u32), |v| format!("{v:.0}"), 100., window, cx))
            .child(self.opt_slider("Rows", "sl.ny", spec(1.0, 12.0, 1.0, |e| e.nopts.slice_divide.1 as f32, |e, v| e.nopts.slice_divide.1 = v as u32), |v| format!("{v:.0}"), 100., window, cx))
            .child(Button::new("sl-div").label("Divide Slice").xsmall().on_click(cx.listener(move |e, _, _, cx| {
                e.divide_slice(nx, ny);
                cx.notify();
            })));
        col
    }

    fn render_web_panel(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Div {
        self.refresh_web_preview();
        let Some(st) = &self.web else { return div() };
        let kind = st.kind;
        let mut col = div().flex().flex_col().gap_1p5().pt_2().child(super::render::section_title("Save for Web"));
        if let Some((img, size, ms, r)) = &st.preview {
            col = col
                .child(div().flex().justify_center().child(gpui_kit::img(img.clone()).max_w(px(280.)).max_h(px(200.))))
                .child(label(format!("{}×{} slice: {:.1} KB · encoded in {ms:.1} ms", r.w(), r.h(), *size as f32 / 1024.0)));
        }
        col = col.child(segmented("web-fmt", &[(0u8, "PNG-8"), (1, "PNG-24"), (2, "JPEG"), (3, "WebP")], kind, cx, |e, v| {
            if let Some(w) = &mut e.web {
                w.kind = v;
                w.stale = true;
            }
        }));
        match kind {
            0 => {
                let (dither, transp) = (st.dither, st.transparency);
                col = col
                    .child(self.opt_slider("Colors", "web.colors", spec(2.0, 256.0, 1.0, |e| e.web.as_ref().map_or(128.0, |w| w.colors as f32), |e, v| {
                        if let Some(w) = &mut e.web {
                            w.colors = v as u16;
                            w.stale = true;
                        }
                    }), |v| format!("{v:.0}"), 110., window, cx))
                    .child(
                        div()
                            .flex()
                            .gap_1()
                            .child(toggle("web-dither", "Dither", dither).on_click(cx.listener(|e, _, _, cx| {
                                if let Some(w) = &mut e.web {
                                    w.dither = !w.dither;
                                    w.stale = true;
                                }
                                cx.notify();
                            })))
                            .child(toggle("web-transp", "Transparency", transp).on_click(cx.listener(|e, _, _, cx| {
                                if let Some(w) = &mut e.web {
                                    w.transparency = !w.transparency;
                                    w.stale = true;
                                }
                                cx.notify();
                            }))),
                    );
            }
            2 => {
                col = col.child(self.opt_slider("Quality", "web.q", spec(1.0, 100.0, 1.0, |e| e.web.as_ref().map_or(60.0, |w| w.quality as f32), |e, v| {
                    if let Some(w) = &mut e.web {
                        w.quality = v as u8;
                        w.stale = true;
                    }
                }), |v| format!("{v:.0}"), 110., window, cx));
            }
            3 => col = col.child(label("WebP (lossless)")),
            _ => {}
        }
        col.child(
            div()
                .flex()
                .gap_1()
                .child(Button::new("web-save").label("Save…").xsmall().primary().on_click(cx.listener(|e, _, _, cx| {
                    e.save_for_web(cx);
                    cx.notify();
                })))
                .child(Button::new("web-close").label("Done").xsmall().on_click(cx.listener(|e, _, _, cx| {
                    e.web = None;
                    cx.notify();
                }))),
        )
    }
}
