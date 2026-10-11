//! Edit ▸ Content-Aware Fill: a workspace with an editable sampling area
//! (shown in green on the canvas, painted with the brush: drag adds, Alt
//! drag subtracts), fill settings (colour / rotation / scale / mirror
//! adaptation), a live preview — on the canvas and as a panel thumbnail —
//! and output to the current, a new or a duplicate layer.
//!
//! The fill is recomputed off the UI thread whenever the settings or the
//! sampling area change; a newer request supersedes an older one.

use std::sync::Arc;

use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::Sizable as _;
use gpui_kit::prelude::*;
use gpui_kit::{Context, Div, RenderImage, SharedString, div, px, rgb};

use crate::engine::document::Layer;
use crate::engine::grid::{RectI, TS, TS_I, TileGrid, finish_tile};
use crate::engine::masks::SelOp;
use crate::engine::patchmatch::{Adapt, ColorAdapt, RotationAdapt};
use crate::engine::px;
use crate::engine::retouch::{self, CafResult, SamplingArea};
use crate::engine::selection::{MaskGrid, MaskTile, Selection};

use super::editor::{Drag, Editor};
use super::options::{label, segmented, toggle};

/// Where Apply writes the fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CafOutput {
    #[default]
    Current,
    NewLayer,
    Duplicate,
}

pub struct CafState {
    pub hole: Arc<MaskGrid>,
    pub sampling: MaskGrid,
    pub area: SamplingArea,
    pub adapt: Adapt,
    pub output: CafOutput,
    pub brush: f32,
    /// Sampling area (green) plus the current fill, drawn over the canvas.
    pub overlay: Layer,
    pub result: Option<Arc<CafResult>>,
    pub preview: Option<Arc<RenderImage>>,
    /// Bumped on every change; a finished job only lands if still current.
    pub version: u64,
    pub computing: Option<u64>,
    pub last_ms: f32,
    /// Set by panel controls; the next frame recomputes.
    pub pending: bool,
}

/// The green tint of the sampling area (premultiplied).
const GREEN: u32 = px::pack(0, 70, 0, 110);

impl Editor {
    /// Opens the workspace for the current selection.
    pub fn open_content_aware_fill(&mut self, cx: &mut Context<Self>) {
        let Some(sel) = self.doc.selection.clone() else {
            self.flash("Content-Aware Fill needs a selection of the area to fill");
            return;
        };
        let li = self.doc.active;
        if !self.ensure_pixels(li) {
            return;
        }
        let (w, h) = (self.doc.w, self.doc.h);
        let hole = Arc::new(MaskGrid::from_selection(&sel, w, h));
        let sampling = retouch::caf_sampling(&self.doc.layers[li].grid, &hole, SamplingArea::Auto);
        self.caf = Some(CafState {
            hole,
            sampling,
            area: SamplingArea::Auto,
            adapt: Adapt { color: ColorAdapt::Default, ..Default::default() },
            output: CafOutput::Current,
            brush: 120.0,
            overlay: Layer::new(0, "caf", TileGrid::new(w, h)),
            result: None,
            preview: None,
            version: 0,
            computing: None,
            last_ms: 0.0,
            pending: false,
        });
        self.caf_changed(cx);
        self.flash("Content-Aware Fill: paint the sampling area (Alt subtracts) · Enter applies · Esc cancels");
    }

    /// Settings or sampling changed: refresh the overlay and recompute.
    pub fn caf_changed(&mut self, cx: &mut Context<Self>) {
        let li = self.doc.active;
        let Some(c) = &mut self.caf else { return };
        c.pending = false;
        c.version += 1;
        let v = c.version;
        c.computing = Some(v);
        let grid = self.doc.layers[li].grid.clone();
        let (hole, sampling, adapt) = (c.hole.clone(), c.sampling.clone(), c.adapt);
        self.caf_rebuild_overlay();
        cx.spawn(async move |this, cx| {
            let t0 = std::time::Instant::now();
            let res = cx.background_executor().spawn(async move { retouch::content_aware_fill(&grid, &hole, &sampling, adapt) }).await;
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            let _ = this.update(cx, |ed, cx| {
                let Some(c) = &mut ed.caf else { return };
                if c.version != v {
                    return;
                }
                c.computing = None;
                c.last_ms = ms;
                c.preview = res.as_ref().map(preview_image);
                c.result = res.map(Arc::new);
                ed.caf_rebuild_overlay();
                cx.notify();
            });
        })
        .detach();
        cx.notify();
    }

    /// Rebuilds the overlay tiles over the sampling area and the fill.
    fn caf_rebuild_overlay(&mut self) {
        let Some(c) = &mut self.caf else { return };
        let (w, h) = (c.sampling.w, c.sampling.h);
        let mut g = TileGrid::new(w, h);
        let bbox = retouch::mask_bbox(&c.sampling).union(&c.result.as_ref().map_or(RectI::EMPTY, |r| r.region));
        let (tx0, ty0, tx1, ty1) = bbox.tile_span(g.cols, g.rows);
        use rayon::prelude::*;
        let list: Vec<(i32, i32)> = (ty0..ty1).flat_map(|ty| (tx0..tx1).map(move |tx| (tx, ty))).collect();
        let (sampling, result) = (&c.sampling, c.result.as_deref());
        let tiles: Vec<(usize, crate::engine::grid::TileRef)> = list
            .par_iter()
            .filter_map(|&(tx, ty)| {
                let st = sampling.tile(tx, ty);
                let rr = TileGrid::tile_rect(tx, ty);
                let hit = result.filter(|r| r.region.intersects(&rr));
                if matches!(st, MaskTile::Empty) && hit.is_none() {
                    return None;
                }
                let mut out = vec![0u32; TS * TS].into_boxed_slice();
                for y in 0..TS {
                    for x in 0..TS {
                        let k = y * TS + x;
                        let s = match st {
                            MaskTile::Empty => 0,
                            MaskTile::Full => 255,
                            MaskTile::Partial(m) => m[k] as u32,
                        };
                        let mut v = px::scale(GREEN, s);
                        if let Some(r) = hit {
                            let (gx, gy) = (tx * TS_I + x as i32, ty * TS_I + y as i32);
                            if gx >= r.region.x0 && gy >= r.region.y0 && gx < r.region.x1 && gy < r.region.y1 {
                                let i = ((gy - r.region.y0) * r.region.w() + gx - r.region.x0) as usize;
                                if r.cov[i] > 0 {
                                    v = px::lerp(v, r.px[i], r.cov[i] as u32);
                                }
                            }
                        }
                        out[k] = v;
                    }
                }
                Some(((ty as u32 * g.cols + tx as u32) as usize, finish_tile(out)))
            })
            .collect();
        for (i, t) in tiles {
            g.tiles[i] = t;
        }
        c.overlay = Layer::new(0, "caf", g);
        self.doc.mark_all_dirty();
    }

    /// Paints (or with `subtract`, erases) the sampling area from `a` to `b`.
    pub fn caf_paint(&mut self, a: (f32, f32), b: (f32, f32), subtract: bool) {
        let Some(c) = &mut self.caf else { return };
        let r = c.brush * 0.5;
        let n = (((b.0 - a.0).hypot(b.1 - a.1)) / (r * 0.4).max(1.0)).ceil().max(1.0) as usize;
        let (w, h) = (c.sampling.w, c.sampling.h);
        let mut dabs = MaskGrid::new(w, h);
        for k in 0..=n {
            let t = k as f32 / n as f32;
            let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            let e = Selection::ellipse(RectI::new((x - r) as i32, (y - r) as i32, (x + r) as i32, (y + r) as i32));
            dabs = dabs.combine(&MaskGrid::from_selection(&e, w, h), SelOp::Add);
        }
        c.sampling = c.sampling.combine(&dabs, if subtract { SelOp::Subtract } else { SelOp::Add }).combine(&c.hole, SelOp::Subtract);
        c.area = SamplingArea::Custom;
        self.caf_rebuild_overlay();
    }

    /// Writes the fill to the chosen output and closes the workspace.
    pub fn caf_apply(&mut self, cx: &mut Context<Self>) {
        let Some(c) = self.caf.take() else { return };
        let Some(r) = c.result else {
            self.flash("The fill is still being computed");
            self.caf = Some(CafState { result: None, ..c });
            return;
        };
        let li = self.doc.active;
        let (w, h) = (self.doc.w, self.doc.h);
        self.doc.push_undo("Content-Aware Fill");
        match c.output {
            CafOutput::Current => {
                let g = retouch::write_region(&self.doc.layers[li].grid, r.region, &r.px, &r.cov);
                self.doc.set_layer_grid(li, g);
            }
            CafOutput::NewLayer => {
                let g = retouch::write_region(&TileGrid::new(w, h), r.region, &r.px, &r.cov);
                let name = format!("{} (Content-Aware Fill)", self.doc.layers[li].name);
                self.doc.insert_layer(name, g);
            }
            CafOutput::Duplicate => {
                let base = self.doc.layers[li].grid.clone();
                let name = format!("{} copy", self.doc.layers[li].name);
                let g = retouch::write_region(&base, r.region, &r.px, &r.cov);
                self.doc.insert_layer(name, g);
            }
        }
        self.doc.selection = None;
        self.doc.mark_all_dirty();
        self.flash(format!("Content-Aware Fill applied ({:.0} ms)", c.last_ms));
        cx.notify();
    }

    pub fn caf_cancel(&mut self, cx: &mut Context<Self>) {
        if self.caf.take().is_some() {
            self.doc.mark_all_dirty();
            cx.notify();
        }
    }

    /// Mouse-down while the workspace is open: starts painting the area.
    pub fn caf_down(&mut self, ix: f32, iy: f32, alt: bool) -> bool {
        if self.caf.is_none() {
            return false;
        }
        self.caf_paint((ix, iy), (ix, iy), alt);
        self.drag = Drag::CafPaint { last: (ix, iy), subtract: alt };
        true
    }

    /// The workspace panel (top of the side panel while it is open).
    pub fn render_caf_panel(&mut self, cx: &mut Context<Self>) -> Div {
        let Some(c) = &self.caf else { return div() };
        let status: SharedString = match (c.computing.is_some(), &c.result) {
            (true, _) => "Computing fill…".into(),
            (false, Some(r)) => format!("Fill {} × {} in {:.0} ms", r.region.w(), r.region.h(), c.last_ms).into(),
            (false, None) => "No sampling area left".into(),
        };
        let mut col = div().flex().flex_col().gap_1p5().p_2().border_b_1().border_color(rgb(0x3a3a3d)).child(super::render::section_title("Content-Aware Fill"));
        if let Some(img) = &c.preview {
            col = col.child(div().flex().justify_center().child(gpui_kit::img(img.clone()).max_w(px(280.)).max_h(px(200.))));
        }
        let (area, adapt, output) = (c.area, c.adapt, c.output);
        col.child(label(status))
            .child(label("Sampling Area"))
            .child(segmented("caf-area", &SamplingArea::ALL.map(|m| (m, m.label())), area, cx, |e, v| {
                if let Some(c) = &mut e.caf {
                    c.area = v;
                    let li = e.doc.active;
                    c.sampling = retouch::caf_sampling(&e.doc.layers[li].grid, &c.hole, v);
                    c.pending = true;
                }
            }))
            .child(label("Color Adaptation"))
            .child(segmented("caf-color", &ColorAdapt::ALL.map(|m| (m, m.label())), adapt.color, cx, |e, v| {
                if let Some(c) = &mut e.caf {
                    c.adapt.color = v;
                    c.pending = true;
                }
            }))
            .child(label("Rotation Adaptation"))
            .child(segmented("caf-rot", &RotationAdapt::ALL.map(|m| (m, m.label())), adapt.rotation, cx, |e, v| {
                if let Some(c) = &mut e.caf {
                    c.adapt.rotation = v;
                    c.pending = true;
                }
            }))
            .child(
                div()
                    .flex()
                    .gap_1()
                    .child(toggle("caf-scale", "Scale", adapt.scale).on_click(cx.listener(|e, _, _, cx| {
                        if let Some(c) = &mut e.caf {
                            c.adapt.scale = !c.adapt.scale;
                        }
                        e.caf_changed(cx);
                    })))
                    .child(toggle("caf-mirror", "Mirror", adapt.mirror).on_click(cx.listener(|e, _, _, cx| {
                        if let Some(c) = &mut e.caf {
                            c.adapt.mirror = !c.adapt.mirror;
                        }
                        e.caf_changed(cx);
                    }))),
            )
            .child(label("Output To"))
            .child(segmented("caf-out", &[(CafOutput::Current, "Current Layer"), (CafOutput::NewLayer, "New Layer"), (CafOutput::Duplicate, "Duplicate Layer")], output, cx, |e, v| {
                if let Some(c) = &mut e.caf {
                    c.output = v;
                }
            }))
            .child(
                div()
                    .flex()
                    .gap_1()
                    .justify_end()
                    .child(Button::new("caf-cancel").label("Cancel").xsmall().ghost().on_click(cx.listener(|e, _, _, cx| e.caf_cancel(cx))))
                    .child(Button::new("caf-ok").label("Apply").xsmall().primary().on_click(cx.listener(|e, _, _, cx| e.caf_apply(cx)))),
            )
    }
}

/// A ≤ 280 px thumbnail of the filled region (BGRA for GPUI).
fn preview_image(r: &CafResult) -> Arc<RenderImage> {
    let (w, h) = (r.region.w() as u32, r.region.h() as u32);
    let k = (w.max(h) as f32 / 280.0).max(1.0);
    let (tw, th) = (((w as f32 / k) as u32).max(1), ((h as f32 / k) as u32).max(1));
    let mut out = vec![0u8; (tw * th * 4) as usize];
    for y in 0..th {
        for x in 0..tw {
            let (sx, sy) = (((x as f32 + 0.5) * k) as u32, ((y as f32 + 0.5) * k) as u32);
            let i = (sy.min(h - 1) * w + sx.min(w - 1)) as usize;
            let [r_, g, b, _] = px::unpremultiply(r.px[i]);
            let o = ((y * tw + x) * 4) as usize;
            out[o..o + 4].copy_from_slice(&[b, g, r_, 255]);
        }
    }
    let rgba = image::RgbaImage::from_raw(tw, th, out).expect("preview size");
    crate::ui::render_image(vec![image::Frame::new(rgba)])
}
