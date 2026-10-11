//! Viewport math and the GPU display-tile cache.
//!
//! Each frame we pick the mip level that matches the zoom, list the display
//! tiles that intersect the canvas, and composite only the ones missing from
//! the cache, in parallel, under a time budget. Tiles that could not be
//! composited this frame are drawn from a coarser cached level instead, so
//! panning and zooming never stall regardless of image size.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use gpui_kit::{Bounds, Pixels, Point, RenderImage, TransformationMatrix, Window, point, px};
use image::{Frame, RgbaImage};
use rayon::prelude::*;

use crate::engine::composite::{ComposeInput, DT, Scratch, display_tile};
use crate::engine::document::{Document, Layer};
use crate::engine::grid::{RectI, TS};
use crate::engine::px as pxm;

const MAX_MAGNIFY_LEVEL: i32 = 5;
const FRAME_BUDGET_MS: f32 = 26.0;

#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    /// Device pixels per image pixel (1.0 = 100%).
    pub zoom: f32,
    /// Logical position of the image origin relative to the canvas origin
    /// (before the view rotation).
    pub pan: Point<Pixels>,
    /// Rotate View angle in radians, about the canvas centre (display only).
    pub rot: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Viewport { zoom: 1.0, pan: point(px(0.), px(0.)), rot: 0.0 }
    }
}

impl Viewport {
    pub fn logical_scale(&self, sf: f32) -> f32 {
        self.zoom / sf
    }

    /// Mip level for the current zoom; negative levels are nearest-neighbor
    /// magnifications.
    pub fn level(&self, max_level: u32) -> i32 {
        if self.zoom >= 2.0 {
            -(self.zoom.log2().floor() as i32).min(MAX_MAGNIFY_LEVEL)
        } else if self.zoom >= 1.0 {
            0
        } else {
            ((1.0 / self.zoom).log2().floor() as i32).clamp(0, max_level as i32)
        }
    }

    /// Rotates screen point `p` by `ang` about the canvas centre.
    fn spin(p: Point<Pixels>, ang: f32, canvas: Bounds<Pixels>) -> Point<Pixels> {
        if ang == 0.0 {
            return p;
        }
        let c = canvas.center();
        let (dx, dy) = (f32::from(p.x - c.x), f32::from(p.y - c.y));
        let (sn, cs) = ang.sin_cos();
        point(c.x + px(dx * cs - dy * sn), c.y + px(dx * sn + dy * cs))
    }

    /// Screen point → the unrotated view's screen point.
    pub fn unrotate(&self, p: Point<Pixels>, canvas: Bounds<Pixels>) -> Point<Pixels> {
        Self::spin(p, -self.rot, canvas)
    }

    /// A screen-space drag delta expressed in unrotated (pan) space.
    pub fn unrotate_delta(&self, d: Point<Pixels>) -> Point<Pixels> {
        let (sn, cs) = (-self.rot).sin_cos();
        let (dx, dy) = (f32::from(d.x), f32::from(d.y));
        point(px(dx * cs - dy * sn), px(dx * sn + dy * cs))
    }

    pub fn to_image(&self, p: Point<Pixels>, canvas: Bounds<Pixels>, sf: f32) -> (f32, f32) {
        let p = self.unrotate(p, canvas);
        let s = self.logical_scale(sf);
        (
            (f32::from(p.x - canvas.origin.x) - f32::from(self.pan.x)) / s,
            (f32::from(p.y - canvas.origin.y) - f32::from(self.pan.y)) / s,
        )
    }

    pub fn to_screen(&self, x: f32, y: f32, canvas: Bounds<Pixels>, sf: f32) -> Point<Pixels> {
        let s = self.logical_scale(sf);
        let q = point(canvas.origin.x + self.pan.x + px(x * s), canvas.origin.y + self.pan.y + px(y * s));
        Self::spin(q, self.rot, canvas)
    }

    /// Zooms to `zoom`, keeping the image point under `anchor` fixed.
    pub fn zoom_about(&mut self, zoom: f32, anchor: Point<Pixels>, canvas: Bounds<Pixels>) {
        let zoom = zoom.clamp(0.002, 32.0);
        let anchor = self.unrotate(anchor, canvas);
        let ax = f32::from(anchor.x - canvas.origin.x);
        let ay = f32::from(anchor.y - canvas.origin.y);
        let k = zoom / self.zoom;
        self.pan = point(
            px(ax - (ax - f32::from(self.pan.x)) * k),
            px(ay - (ay - f32::from(self.pan.y)) * k),
        );
        self.zoom = zoom;
    }

    pub fn fit(&mut self, w: u32, h: u32, canvas: Bounds<Pixels>, sf: f32) {
        let cw = f32::from(canvas.size.width).max(1.0);
        let ch = f32::from(canvas.size.height).max(1.0);
        let s = ((cw - 40.0) / w as f32).min((ch - 40.0) / h as f32).max(0.0005);
        self.zoom = (s * sf).min(32.0);
        let s = self.logical_scale(sf);
        self.pan = point(px((cw - w as f32 * s) * 0.5), px((ch - h as f32 * s) * 0.5));
    }

    pub fn center_at_zoom(&mut self, zoom: f32, w: u32, h: u32, canvas: Bounds<Pixels>, sf: f32) {
        self.zoom = zoom;
        let s = self.logical_scale(sf);
        let cw = f32::from(canvas.size.width);
        let ch = f32::from(canvas.size.height);
        self.pan = point(px((cw - w as f32 * s) * 0.5), px((ch - h as f32 * s) * 0.5));
    }
}

pub struct TileDraw {
    pub bounds: Bounds<Pixels>,
    pub image_bounds: Bounds<Pixels>,
    pub img: TileImg,
    /// Rotate View: the sprite's device-pixel transformation.
    pub xf: Option<TransformationMatrix>,
}

/// Rotation by `ang` about `center` (logical pixels), in device pixels.
pub fn view_rotation(ang: f32, center: Point<Pixels>, sf: f32) -> TransformationMatrix {
    let (sn, cs) = ang.sin_cos();
    let (cx, cy) = (f32::from(center.x) * sf, f32::from(center.y) * sf);
    TransformationMatrix { rotation_scale: [[cs, -sn], [sn, cs]], translation: [cx - (cs * cx - sn * cy), cy - (sn * cx + cs * cy)] }
}

#[derive(Default)]
pub struct FramePlan {
    pub draws: Vec<TileDraw>,
    pub image_rect: Option<Bounds<Pixels>>,
    pub visible: usize,
    pub composed: usize,
    pub pending: usize,
    pub compose_ms: f32,
}

/// A cached display tile: pixels uploaded to GPUI's atlas, or a texture the
/// GPU compositor rendered (drawn without copying).
#[derive(Clone)]
pub enum TileImg {
    Cpu(Arc<CpuImg>),
    Gpu(Arc<GpuImg>),
}

pub struct CpuImg { image: Arc<RenderImage> }

pub struct GpuImg {
    pub tile: crate::engine::gpu::compose::GpuTile,
    /// Registration with the window's sprite atlas (made when first painted).
    pub atlas: std::sync::Mutex<Option<gpui_kit::AtlasTile>>,
}

impl TileImg {
    fn cpu(image: Arc<RenderImage>) -> Self {
        Self::Cpu(Arc::new(CpuImg { image }))
    }
    /// Paints the tile (registering a GPU texture on first use).
    pub fn paint(&self, window: &mut Window, bounds: Bounds<Pixels>, image_bounds: Bounds<Pixels>, xf: Option<TransformationMatrix>) {
        match self {
            TileImg::Cpu(img) => {
                let _ = match xf {
                    Some(xf) => window.paint_image_transformed(bounds, image_bounds, img.image.clone(), 0, xf),
                    None => window.paint_image(bounds, image_bounds, gpui_kit::Corners::default(), img.image.clone(), 0, false),
                };
            }
            TileImg::Gpu(g) => {
                let mut a = g.atlas.lock().unwrap();
                if a.is_none() {
                    let dt = DT as u32;
                    *a = window.insert_external_texture(gpui_kit::ExternalTexture { raw: g.tile.raw as *mut std::ffi::c_void, width: dt, height: dt });
                }
                if let Some(t) = a.clone() {
                    window.paint_atlas_tile(bounds, image_bounds, t, xf);
                }
            }
        }
    }

    fn release(self, window: &mut Window) {
        match self {
            TileImg::Cpu(img) => {
                let _ = window.drop_image(img.image.clone());
            }
            TileImg::Gpu(g) => {
                if let Some(t) = g.atlas.lock().unwrap().take() {
                    window.remove_external_texture(&t);
                }
            }
        }
    }
}

struct Entry {
    img: TileImg,
    stamp: u64,
}

#[derive(Default)]
pub struct DisplayCache {
    map: HashMap<(i32, i32, i32), Entry>,
    frame: u64,
    drop_queue: Vec<TileImg>,
    generation: u64,
    /// Bumped whenever a tile enters or leaves the cache.
    version: u64,
    pub last: FrameStats,
}

#[derive(Default, Clone, Copy)]
pub struct FrameStats {
    pub level: i32,
    pub visible: usize,
    pub composed: usize,
    pub pending: usize,
    pub compose_ms: f32,
    pub cached: usize,
}

fn snap(v: f32, sf: f32) -> f32 {
    (v * sf).round() / sf
}

fn tile_image_px(level: i32) -> f32 {
    TS as f32 * 2f32.powi(level)
}

/// Level-0 pixel area a display tile reads (including its gutter).
fn tile_l0_rect(l: i32, x: i32, y: i32) -> RectI {
    let t = TS as i32;
    if l >= 0 {
        RectI::new((x * t - 1) << l, (y * t - 1) << l, ((x + 1) * t + 1) << l, ((y + 1) * t + 1) << l)
    } else {
        let k = -l;
        RectI::new((x * t - 1) >> k, (y * t - 1) >> k, (((x + 1) * t + 1) >> k) + 1, (((y + 1) * t + 1) >> k) + 1)
    }
}

/// Screen rect of a tile (snapped to device pixels) and its gutter-extended
/// image bounds.
fn tile_geometry(level: i32, x: i32, y: i32, origin: Point<Pixels>, s: f32, sf: f32) -> (Bounds<Pixels>, Bounds<Pixels>) {
    let tsi = tile_image_px(level);
    let ox = f32::from(origin.x);
    let oy = f32::from(origin.y);
    let x0 = snap(ox + x as f32 * tsi * s, sf);
    let x1 = snap(ox + (x + 1) as f32 * tsi * s, sf);
    let y0 = snap(oy + y as f32 * tsi * s, sf);
    let y1 = snap(oy + (y + 1) as f32 * tsi * s, sf);
    let tw = (x1 - x0) / TS as f32;
    let th = (y1 - y0) / TS as f32;
    (
        Bounds::from_corners(point(px(x0), px(y0)), point(px(x1), px(y1))),
        Bounds::from_corners(point(px(x0 - tw), px(y0 - th)), point(px(x1 + tw), px(y1 + th))),
    )
}

impl DisplayCache {
    pub fn clear(&mut self) {
        for (_, e) in self.map.drain() {
            self.drop_queue.push(e.img);
        }
    }

    /// Every GPU image this cache holds (closing its document).
    pub fn take_images(&mut self) -> Vec<TileImg> {
        self.clear();
        std::mem::take(&mut self.drop_queue)
    }

    pub fn queue_drop(&mut self, img: Arc<RenderImage>) {
        self.drop_queue.push(TileImg::cpu(img));
    }

    pub fn queue_drop_tile(&mut self, img: TileImg) {
        self.drop_queue.push(img);
    }

    pub fn flush_drops(&mut self, window: &mut Window) {
        for img in self.drop_queue.drain(..) {
            img.release(window);
        }
    }

    pub fn prepare(
        &mut self,
        doc: &mut Document,
        extra: Option<&Layer>,
        vp: &Viewport,
        canvas: Bounds<Pixels>,
        sf: f32,
        window: &mut Window,
    ) -> FramePlan {
        self.frame += 1;
        self.version += 1;
        let dirty = doc.take_dirty();
        let dirtied = dirty.all || doc.generation != self.generation || !dirty.rects.is_empty();
        if !dirtied {
            self.version -= 1;
        }
        if dirty.all || doc.generation != self.generation {
            self.generation = doc.generation;
            self.clear();
        } else if !dirty.rects.is_empty() {
            let stale: Vec<_> = self
                .map
                .keys()
                .copied()
                .filter(|&(l, x, y)| {
                    let r = tile_l0_rect(l, x, y);
                    dirty.rects.iter().any(|d| d.intersects(&r))
                })
                .collect();
            for k in stale {
                if let Some(e) = self.map.remove(&k) {
                    self.drop_queue.push(e.img);
                }
            }
        }
        self.flush_drops(window);

        self.plan(doc, extra, vp, canvas, sf)
    }

    /// The frame's draw list (composing missing tiles under the time
    /// budget). Rotate View draws the same cached tiles as GPU-transformed
    /// sprites: the tiles of the rotated canvas footprint are listed and each
    /// draw carries the rotation about the canvas centre.
    pub fn plan(&mut self, doc: &Document, extra: Option<&Layer>, vp: &Viewport, canvas: Bounds<Pixels>, sf: f32) -> FramePlan {
        let max_level = doc.max_level();
        let level = vp.level(max_level);
        let rotated = vp.rot != 0.0;
        let s = vp.logical_scale(sf);
        let origin = canvas.origin + vp.pan;
        let (w, h) = (doc.w as f32, doc.h as f32);
        let img_rect = Bounds::from_corners(
            point(px(snap(f32::from(origin.x), sf)), px(snap(f32::from(origin.y), sf))),
            point(px(snap(f32::from(origin.x) + w * s, sf)), px(snap(f32::from(origin.y) + h * s, sf))),
        );
        let mut plan = FramePlan { image_rect: (!rotated).then_some(img_rect), ..Default::default() };
        // Visible image rect: the canvas mapped back into the (unrotated) image.
        let (ix0, iy0, ix1, iy1) = if rotated {
            let (cw, ch) = (canvas.size.width, canvas.size.height);
            let corners = [canvas.origin, canvas.origin + point(cw, px(0.)), canvas.origin + point(px(0.), ch), canvas.origin + point(cw, ch)];
            let ic: Vec<(f32, f32)> = corners.iter().map(|&c| vp.to_image(c, canvas, sf)).collect();
            (
                ic.iter().map(|c| c.0).fold(f32::MAX, f32::min).max(0.0),
                ic.iter().map(|c| c.1).fold(f32::MAX, f32::min).max(0.0),
                ic.iter().map(|c| c.0).fold(f32::MIN, f32::max).min(w),
                ic.iter().map(|c| c.1).fold(f32::MIN, f32::max).min(h),
            )
        } else {
            let vis = img_rect.intersect(&canvas);
            if vis.size.width <= px(0.) || vis.size.height <= px(0.) {
                return plan;
            }
            (
                (f32::from(vis.origin.x - origin.x) / s).max(0.0),
                (f32::from(vis.origin.y - origin.y) / s).max(0.0),
                (f32::from(vis.origin.x + vis.size.width - origin.x) / s).min(w),
                (f32::from(vis.origin.y + vis.size.height - origin.y) / s).min(h),
            )
        };
        if ix1 <= ix0 || iy1 <= iy0 {
            return plan;
        }
        let xf = rotated.then(|| view_rotation(vp.rot, canvas.center(), sf));
        // Rotated tiles (drawn without edge antialiasing) grow half a device
        // pixel into their gutter, so neighbours overlap and rasterization
        // never leaves a pixel between them uncovered.
        let grow = px(if rotated { 0.5 / sf } else { 0.0 });
        let img_clip = img_rect.dilate(grow);
        let clip = |b: Bounds<Pixels>| if rotated { b.dilate(grow).intersect(&img_clip) } else { b.intersect(&img_rect) };
        let tsi = tile_image_px(level);
        let max_tx = (w / tsi).ceil() as i32;
        let max_ty = (h / tsi).ceil() as i32;
        let tx0 = ((ix0 / tsi).floor() as i32).clamp(0, max_tx);
        let ty0 = ((iy0 / tsi).floor() as i32).clamp(0, max_ty);
        let tx1 = ((ix1 / tsi).ceil() as i32).clamp(0, max_tx);
        let ty1 = ((iy1 / tsi).ceil() as i32).clamp(0, max_ty);
        let (cx0, cy0) = (canvas.origin.x, canvas.origin.y);
        let (cx1, cy1) = (cx0 + canvas.size.width, cy0 + canvas.size.height);
        // Rotated: skip tiles whose square lies entirely outside the canvas.
        let outside = |tx: i32, ty: i32| {
            if !rotated {
                return false;
            }
            let c = vp.to_screen((tx as f32 + 0.5) * tsi, (ty as f32 + 0.5) * tsi, canvas, sf);
            let r = px(tsi * s * std::f32::consts::FRAC_1_SQRT_2);
            c.x + r < cx0 || c.y + r < cy0 || c.x - r > cx1 || c.y - r > cy1
        };

        let mut missing = Vec::new();
        let frame = self.frame;
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                if outside(tx, ty) {
                    continue;
                }
                plan.visible += 1;
                if let Some(e) = self.map.get_mut(&(level, tx, ty)) {
                    e.stamp = frame;
                    let (b, ib) = tile_geometry(level, tx, ty, origin, s, sf);
                    plan.draws.push(TileDraw { bounds: clip(b), image_bounds: ib, img: e.img.clone(), xf });
                } else {
                    missing.push((tx, ty));
                }
            }
        }
        let cx = (tx0 + tx1) as f32 * 0.5;
        let cy = (ty0 + ty1) as f32 * 0.5;
        missing.sort_by(|a, b| {
            let da = (a.0 as f32 + 0.5 - cx).powi(2) + (a.1 as f32 + 0.5 - cy).powi(2);
            let db = (b.0 as f32 + 0.5 - cx).powi(2) + (b.1 as f32 + 0.5 - cy).powi(2);
            da.total_cmp(&db)
        });

        let start = Instant::now();
        // A live filter preview renders the region of every missing tile in
        // one GPU pass (when the GPU is on and worth it); the compositor
        // then copies from it.
        if let Some(p) = doc.preview.as_ref().filter(|_| !missing.is_empty())
            && let Some(layer) = doc.layers.get(p.layer)
        {
            let (lvl, sh) = if level >= 0 { (level as u32, 0) } else { (0, (-level) as u32) };
            let (ox, oy) = (layer.offset.0 >> lvl, layer.offset.1 >> lvl);
            let mut r = RectI::EMPTY;
            for &(tx, ty) in &missing {
                let (gx0, gy0) = (tx * TS as i32 - 1, ty * TS as i32 - 1);
                let t = RectI::new(gx0 >> sh, gy0 >> sh, ((gx0 + DT as i32 - 1) >> sh) + 1, ((gy0 + DT as i32 - 1) >> sh) + 1);
                r = if r.is_empty() { t } else { r.union(&t) };
            }
            crate::engine::gpu::preview::prepare(&p.gpu, &p.filter, layer, lvl, r.translate(-ox, -oy));
        }
        let input = ComposeInput { layers: &doc.layers, preview: doc.preview.as_ref(), extra };
        let chunk = rayon::current_num_threads().max(4) * 2;
        let mut idx = 0;
        // The GPU compositor takes every missing tile in one pass (no time
        // budget needed); otherwise the CPU composes under the budget.
        if let Some(tiles) = crate::engine::gpu::compose::try_display_tiles(&input, level, &missing, Some((tx0, ty0, tx1, ty1)), true) {
            let imgs: Vec<TileImg> = match tiles {
                crate::engine::gpu::compose::Tiles::Bytes(v) => v
                    .into_iter()
                    .map(|buf| TileImg::cpu(crate::ui::render_image(vec![Frame::new(RgbaImage::from_raw(DT as u32, DT as u32, buf).expect("display tile size"))])))
                    .collect(),
                crate::engine::gpu::compose::Tiles::Textures(v) => v.into_iter().map(|tile| TileImg::Gpu(Arc::new(GpuImg { tile, atlas: Default::default() }))).collect(),
            };
            for (&(tx, ty), img) in missing.iter().zip(imgs) {
                let (b, ib) = tile_geometry(level, tx, ty, origin, s, sf);
                plan.draws.push(TileDraw { bounds: clip(b), image_bounds: ib, img: img.clone(), xf });
                self.map.insert((level, tx, ty), Entry { img, stamp: frame });
            }
            plan.composed = missing.len();
            idx = missing.len();
        }
        while idx < missing.len() {
            let end = (idx + chunk).min(missing.len());
            let results: Vec<((i32, i32), Vec<u8>)> = missing[idx..end]
                .par_iter()
                .map_init(Scratch::default, |scratch, &(tx, ty)| {
                    let mut buf = vec![0u8; DT * DT * 4];
                    display_tile(&input, level, tx, ty, &mut buf, scratch);
                    ((tx, ty), buf)
                })
                .collect();
            for ((tx, ty), buf) in results {
                let rgba = RgbaImage::from_raw(DT as u32, DT as u32, buf).expect("display tile size");
                let img = TileImg::cpu(crate::ui::render_image(vec![Frame::new(rgba)]));
                let (b, ib) = tile_geometry(level, tx, ty, origin, s, sf);
                plan.draws.push(TileDraw { bounds: clip(b), image_bounds: ib, img: img.clone(), xf });
                self.map.insert((level, tx, ty), Entry { img, stamp: frame });
            }
            plan.composed += end - idx;
            idx = end;
            if start.elapsed().as_secs_f32() * 1000.0 > FRAME_BUDGET_MS {
                break;
            }
        }
        plan.compose_ms = start.elapsed().as_secs_f32() * 1000.0;
        plan.pending = missing.len() - idx;

        // Stand-ins from coarser levels for tiles still pending.
        let mut fallback = Vec::new();
        for &(tx, ty) in &missing[idx..] {
            for j in 1..=6 {
                let pl = level + j;
                if pl > max_level as i32 {
                    break;
                }
                let (px_, py_) = if level >= 0 { (tx >> j, ty >> j) } else { (tx >> j, ty >> j) };
                if let Some(e) = self.map.get_mut(&(pl, px_, py_)) {
                    e.stamp = frame;
                    let (child, _) = tile_geometry(level, tx, ty, origin, s, sf);
                    let (_, parent_ib) = tile_geometry(pl, px_, py_, origin, s, sf);
                    fallback.push(TileDraw { bounds: clip(child), image_bounds: parent_ib, img: e.img.clone(), xf });
                    break;
                }
            }
        }
        if !fallback.is_empty() {
            fallback.append(&mut plan.draws);
            plan.draws = fallback;
        }

        // LRU eviction, never touching tiles used this frame.
        let normal = (plan.visible * 3).max(768);
        let memory = crate::engine::memory::global();
        let keep = normal.min((memory.available() / (DT * DT * 8) as u64) as usize).max(plan.visible);
        if self.map.len() > keep {
            let mut old: Vec<((i32, i32, i32), u64)> =
                self.map.iter().filter(|(_, e)| e.stamp != frame).map(|(k, e)| (*k, e.stamp)).collect();
            old.sort_by_key(|(_, s)| *s);
            let excess = self.map.len() - keep;
            for (k, _) in old.into_iter().take(excess) {
                if let Some(e) = self.map.remove(&k) {
                    self.drop_queue.push(e.img);
                }
            }
        }

        self.last = FrameStats {
            level,
            visible: plan.visible,
            composed: plan.composed,
            pending: plan.pending,
            compose_ms: plan.compose_ms,
            cached: self.map.len(),
        };
        plan
    }
}

/// A small layer preview drawn from the coarsest mip, over a checkerboard.
pub fn thumbnail(layer: &Layer, w: u32, h: u32, side: u32) -> Arc<RenderImage> {
    let top = layer.mips.len() as u32;
    let g = layer.level(top);
    let (lw, lh) = (g.w.max(1), g.h.max(1));
    let mut src = vec![0u32; (lw * lh) as usize];
    g.read_region(0, 0, lw as usize, lh as usize, &mut src);
    let scale = (side as f32 / w as f32).min(side as f32 / h as f32);
    let tw = ((w as f32 * scale).round() as u32).clamp(1, side);
    let th = ((h as f32 * scale).round() as u32).clamp(1, side);
    let mut out = vec![0u8; (tw * th * 4) as usize];
    for y in 0..th {
        for x in 0..tw {
            // 2x2 supersample in the coarse level.
            let mut acc = [0u32; 4];
            for (sx, sy) in [(0.25f32, 0.25f32), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let u = (((x as f32 + sx) / tw as f32) * lw as f32) as u32;
                let v = (((y as f32 + sy) / th as f32) * lh as f32) as u32;
                let p = pxm::unpack(src[(v.min(lh - 1) * lw + u.min(lw - 1)) as usize]);
                for c in 0..4 {
                    acc[c] += p[c] as u32;
                }
            }
            let p = pxm::pack((acc[0] / 4) as u8, (acc[1] / 4) as u8, (acc[2] / 4) as u8, (acc[3] / 4) as u8);
            let check = if ((x >> 2) ^ (y >> 2)) & 1 == 0 { 0xFFFF_FFFF } else { 0xFFCC_CCCC };
            let [r, g2, b, _] = pxm::unpack(pxm::over(p, check));
            let o = ((y * tw + x) * 4) as usize;
            out[o..o + 4].copy_from_slice(&[b, g2, r, 255]);
        }
    }
    let rgba = RgbaImage::from_raw(tw, th, out).expect("thumb size");
    crate::ui::render_image(vec![Frame::new(rgba)])
}

