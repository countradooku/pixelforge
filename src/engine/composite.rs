//! Layer compositing for display tiles, flattening and export.

use rayon::prelude::*;

use super::document::Layer;
use super::layers::LayerKind;
use super::filters::Filter;
use super::grid::{RectI, TS, TS_I, TileGrid, TileRef, finish_tile};
use super::px::{self, scale};
use super::selection::Selection;
use super::blend::{blend_row, blend_region, BlendMode};

/// Side of a display tile: 256 pixels plus a one-pixel gutter on each side so
/// the GPU's bilinear sampling never bleeds across tile seams.
pub const DT: usize = TS + 2;

/// A live, uncommitted filter shown on one layer.
#[derive(Clone)]
pub struct Preview {
    pub layer: usize,
    pub filter: Filter,
    pub selection: Option<Selection>,
    /// Regions the GPU already rendered (see `gpu::preview`).
    pub gpu: std::sync::Arc<crate::engine::gpu::preview::PreviewGpu>,
}

impl Preview {
    pub fn new(layer: usize, filter: Filter, selection: Option<Selection>) -> Preview {
        Preview { layer, filter, selection, gpu: Default::default() }
    }
}

pub struct ComposeInput<'a> {
    pub layers: &'a [Layer],
    pub preview: Option<&'a Preview>,
    /// A display-only layer drawn on top (e.g. Liquify's freeze mask).
    pub extra: Option<&'a Layer>,
}

#[derive(Default)]
pub struct Scratch {
    layer: Vec<u32>,
    padded: Vec<u32>,
    orig: Vec<u32>,
    src: Vec<u32>,
    acc: Vec<u32>,
    /// Buffers for nested groups / clipping groups (taken and returned).
    pool: Vec<Vec<u32>>,
    masks: Vec<Vec<u8>>,
}

impl Scratch {
    fn take(&mut self, n: usize) -> Vec<u32> {
        let mut v = self.pool.pop().unwrap_or_default();
        v.clear();
        v.resize(n, 0);
        v
    }
    fn give(&mut self, v: Vec<u32>) {
        self.pool.push(v);
    }
    fn take_mask(&mut self, n: usize) -> Vec<u8> {
        let mut v = self.masks.pop().unwrap_or_default();
        v.resize(n, 0);
        v
    }
    fn give_mask(&mut self, v: Vec<u8>) {
        self.masks.push(v);
    }
}

/// Composites `w x h` pixels at `(x0, y0)` of mip `level` into `out`.
pub fn compose_region(
    input: &ComposeInput,
    level: u32,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    out: &mut [u32],
    s: &mut Scratch,
) {
    let n = w * h;
    let out = &mut out[..n];
    if input.layers.iter().all(Layer::is_plain) {
        compose_plain(input, level, x0, y0, w, h, out, s);
    } else {
        out.fill(0);
        let top: Vec<usize> = siblings(input.layers, None);
        let mut empty = true;
        compose_list(input, &top, level, x0, y0, w, h, out, &mut empty, s);
    }
    if let Some(x) = input.extra {
        let g = x.level(level);
        let (ex, ey) = (x0 - (x.offset.0 >> level), y0 - (x.offset.1 >> level));
        if g.region_has_content(RectI::wh(ex, ey, w as i32, h as i32)) {
            s.layer.resize(n, 0);
            g.read_region(ex, ey, w, h, &mut s.layer);
            blend_row(out, &s.layer, (x.opacity.clamp(0.0, 1.0) * 255.0).round() as u32, super::blend::BlendMode::Normal);
        }
    }
}

fn advanced(layer: &Layer) -> bool {
    layer.advanced_blend.as_ref().is_some_and(|settings| !settings.is_identity())
}

fn blend_layer_region(destination: &mut [u32], source: &[u32], opacity: u32, layer: &Layer, width: usize, origin: (i32, i32)) {
    if let Some(settings) = layer.advanced_blend.as_ref().filter(|settings| !settings.is_identity()) {
        settings.blend_region(destination, source, opacity, layer.blend, width, origin);
    } else {
        blend_region(destination, source, opacity, layer.blend, width, origin);
    }
}

/// The fast path: a flat stack of plain pixel layers.
#[allow(clippy::too_many_arguments)]
fn compose_plain(input: &ComposeInput, level: u32, x0: i32, y0: i32, w: usize, h: usize, out: &mut [u32], s: &mut Scratch) {
    let n = w * h;
    let mut acc_empty = true;
    s.layer.resize(n, 0);
    for (li, layer) in input.layers.iter().enumerate() {
        if !layer.visible || layer.opacity <= 0.0 {
            continue;
        }
        let op = (layer.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        let mut buf = std::mem::take(&mut s.layer);
        let has = layer_pixels(input, li, level, x0, y0, w, h, &mut buf, s);
        if has {
            if acc_empty && layer.blend != BlendMode::Dissolve && !advanced(layer) {
                // Any blend mode over transparency is just the (faded) source.
                if op == 255 {
                    out.copy_from_slice(&buf);
                } else {
                    for (o, &l) in out.iter_mut().zip(&buf) {
                        *o = scale(l, op);
                    }
                }
            } else {
                if acc_empty {out.fill(0);}
                blend_layer_region(out, &buf, op, layer,w,(x0,y0));
            }
            acc_empty = false;
        }
        s.layer = buf;
    }
    if acc_empty {
        out.fill(0);
    }
}

/// Indices of the layers whose parent is `parent`, bottom to top.
pub fn siblings(layers: &[Layer], parent: Option<u64>) -> Vec<usize> {
    (0..layers.len()).filter(|&i| layers[i].parent == parent).collect()
}

/// Reads a layer's own pixels for the region (applying a live preview),
/// before its mask. Returns false when the region is empty.
#[allow(clippy::too_many_arguments)]
fn layer_pixels(input: &ComposeInput, li: usize, level: u32, x0: i32, y0: i32, w: usize, h: usize, buf: &mut Vec<u32>, s: &mut Scratch) -> bool {
    let layer = &input.layers[li];
    let n = w * h;
    buf.resize(n, 0);
    let grid = layer.level(level);
    let rx = x0 - (layer.offset.0 >> level);
    let ry = y0 - (layer.offset.1 >> level);
    match input.preview.filter(|p| p.layer == li) {
        Some(p) => {
            if !p.gpu.lookup(level, RectI::wh(rx, ry, w as i32, h as i32), buf) {
                p.filter.render(&crate::engine::filters::Src::layer(layer), level, rx, ry, w, h, buf, &mut s.padded);
            }
            if let Some(sel) = &p.selection {
                if let Some(mask) = sel.region_coverage(level, rx, ry, w, h) {
                    s.orig.resize(n, 0);
                    grid.read_region(rx, ry, w, h, &mut s.orig);
                    for ((l, &o), &m) in buf.iter_mut().zip(&s.orig).zip(&mask) {
                        *l = px::lerp(o, *l, m as u32);
                    }
                }
            }
            true
        }
        None => {
            let r = RectI::wh(rx, ry, w as i32, h as i32);
            // A live Move of a layer with hidden pixels: what slides in from
            // beyond the canvas comes from them (point-sampled at mip levels).
            let canvas = RectI::wh(0, 0, (layer.grid.w as i32 + (1 << level) - 1) >> level, (layer.grid.h as i32 + (1 << level) - 1) >> level);
            if let Some(hd) = layer.hidden.as_deref().filter(|_| layer.offset != (0, 0) && !canvas.contains_rect(&r)) {
                grid.read_region(rx, ry, w, h, buf);
                let k = level as i32;
                for (j, row) in buf.chunks_exact_mut(w).enumerate() {
                    let y = ry + j as i32;
                    for (i, v) in row.iter_mut().enumerate() {
                        let x = rx + i as i32;
                        if x < canvas.x0 || y < canvas.y0 || x >= canvas.x1 || y >= canvas.y1 {
                            let (gx, gy) = ((x << k) - hd.x, (y << k) - hd.y);
                            *v = if gx >= 0 && gy >= 0 && gx < hd.grid.w as i32 && gy < hd.grid.h as i32 { hd.grid.pixel(gx, gy) } else { 0 };
                        }
                    }
                }
                return true;
            }
            if !grid.region_has_content(r) {
                return false;
            }
            grid.read_region(rx, ry, w, h, buf);
            true
        }
    }
}

/// Multiplies `buf` by the layer mask. Returns false if the mask hides the
/// whole region.
#[allow(clippy::too_many_arguments)]
fn apply_layer_mask(layer: &Layer, level: u32, x0: i32, y0: i32, w: usize, h: usize, buf: &mut [u32], s: &mut Scratch) -> bool {
    let Some(m) = layer.mask.as_ref().filter(|m| m.enabled) else { return true };
    let g = m.level(level);
    let rx = x0 - (layer.offset.0 >> level);
    let ry = y0 - (layer.offset.1 >> level);
    match super::masks::classify_region(g, RectI::wh(rx, ry, w as i32, h as i32)) {
        super::masks::RegionCover::Full => true,
        super::masks::RegionCover::Empty => false,
        super::masks::RegionCover::Partial => {
            let mut cov = s.take_mask(w * h);
            g.read_region(rx, ry, w, h, &mut cov, true);
            super::masks::apply_coverage(buf, &cov);
            s.give_mask(cov);
            true
        }
    }
}

/// Mask coverage for the region as a vector (adjustment layers), or None
/// when fully revealed. `Some(empty)` means fully hidden.
#[allow(clippy::too_many_arguments)]
fn mask_cov(layer: &Layer, level: u32, x0: i32, y0: i32, w: usize, h: usize, s: &mut Scratch) -> Option<Vec<u8>> {
    let m = layer.mask.as_ref().filter(|m| m.enabled)?;
    let g = m.level(level);
    match super::masks::classify_region(g, RectI::wh(x0, y0, w as i32, h as i32)) {
        super::masks::RegionCover::Full => None,
        super::masks::RegionCover::Empty => Some(vec![]),
        super::masks::RegionCover::Partial => {
            let mut cov = s.take_mask(w * h);
            g.read_region(x0, y0, w, h, &mut cov, true);
            Some(cov)
        }
    }
}

/// Applies an adjustment layer to `acc` in place (opacity × mask).
#[allow(clippy::too_many_arguments)]
fn apply_adjustment(layer: &Layer, f: &Filter, level: u32, x0: i32, y0: i32, w: usize, h: usize, acc: &mut [u32], s: &mut Scratch) {
    let op = (layer.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let cov = mask_cov(layer, level, x0, y0, w, h, s);
    if cov.as_ref().is_some_and(|c| c.is_empty()) || op == 0 {
        return;
    }
    let mut tmp = s.take(w * h);
    tmp.copy_from_slice(acc);
    f.apply(&mut tmp, w, h, level, (x0, y0));
    match &cov {
        None if op == 255 => acc.copy_from_slice(&tmp),
        None => acc.iter_mut().zip(&tmp).for_each(|(a, &t)| *a = px::lerp(*a, t, op)),
        Some(c) => acc.iter_mut().zip(&tmp).zip(c).for_each(|((a, &t), &m)| *a = px::lerp(*a, t, (m as u32 * op + 127) / 255)),
    }
    s.give(tmp);
    if let Some(c) = cov {
        if !c.is_empty() {
            s.give_mask(c);
        }
    }
}

/// Renders one stack item (a layer or a whole group) into `buf` (zeroed).
/// Returns false when it contributes nothing to the region.
#[allow(clippy::too_many_arguments)]
fn render_item(input: &ComposeInput, li: usize, level: u32, x0: i32, y0: i32, w: usize, h: usize, buf: &mut Vec<u32>, s: &mut Scratch) -> bool {
    // Layer Style / Fill: the masked item over the padded region, styled
    // (`styles::render`), replaces `buf`.
    if let Some(plan) = super::styles::plan_for(&input.layers[li], level) {
        let p = plan.pad;
        let (pw, ph) = (w + 2 * p, h + 2 * p);
        let (px0, py0) = (x0 - p as i32, y0 - p as i32);
        let mut pbuf = s.take(pw * ph);
        let has = render_unstyled(input, li, level, px0, py0, pw, ph, &mut pbuf, s);
        if has {
            let l = &input.layers[li];
            buf.resize(w * h, 0);
            super::styles::render(&plan, &pbuf, pw, ph, (px0 - (l.offset.0 >> level), py0 - (l.offset.1 >> level)), buf);
        }
        s.give(pbuf);
        return has;
    }
    render_unstyled(input, li, level, x0, y0, w, h, buf, s)
}

/// [`render_item`] without the layer's style.
#[allow(clippy::too_many_arguments)]
fn render_unstyled(input: &ComposeInput, li: usize, level: u32, x0: i32, y0: i32, w: usize, h: usize, buf: &mut Vec<u32>, s: &mut Scratch) -> bool {
    let layer = &input.layers[li];
    let has = match &layer.kind {
        LayerKind::Group { .. } => {
            buf.resize(w * h, 0);
            buf.fill(0);
            let kids = siblings(input.layers, Some(layer.id));
            let mut empty = true;
            compose_list(input, &kids, level, x0, y0, w, h, buf, &mut empty, s);
            !empty
        }
        LayerKind::Artboard(ab) => {
            // Background, children, then everything clipped to the board.
            buf.resize(w * h, 0);
            buf.fill(0);
            let k = 1i32 << level;
            // The live Move offset applies to the board's rect too.
            let (ox, oy) = layer.offset;
            let ar = RectI::new(ab.rect.x0 + ox, ab.rect.y0 + oy, ab.rect.x1 + ox, ab.rect.y1 + oy);
            let r = RectI::new(ar.x0.div_euclid(k), ar.y0.div_euclid(k), (ar.x1 + k - 1).div_euclid(k), (ar.y1 + k - 1).div_euclid(k));
            let inside = RectI::wh(x0, y0, w as i32, h as i32).intersect(&r);
            if inside.is_empty() {
                return false;
            }
            if let Some(c) = ab.bg {
                let bg = px::pack(c[0], c[1], c[2], 255);
                for y in inside.y0..inside.y1 {
                    let row = (y - y0) as usize * w;
                    buf[row + (inside.x0 - x0) as usize..row + (inside.x1 - x0) as usize].fill(bg);
                }
            }
            let kids = siblings(input.layers, Some(layer.id));
            let mut empty = ab.bg.is_none();
            compose_list(input, &kids, level, x0, y0, w, h, buf, &mut empty, s);
            for y in 0..h as i32 {
                for x in 0..w as i32 {
                    let (gx, gy) = (x0 + x, y0 + y);
                    if gx < inside.x0 || gx >= inside.x1 || gy < inside.y0 || gy >= inside.y1 {
                        buf[y as usize * w + x as usize] = 0;
                    }
                }
            }
            true
        }
        LayerKind::Adjustment(_) => false,
        _ => layer_pixels(input, li, level, x0, y0, w, h, buf, s),
    };
    has && apply_layer_mask(layer, level, x0, y0, w, h, buf, s)
}

/// Composites a list of sibling layers (bottom to top) onto `acc`.
#[allow(clippy::too_many_arguments)]
fn compose_list(input: &ComposeInput, list: &[usize], level: u32, x0: i32, y0: i32, w: usize, h: usize, acc: &mut [u32], empty: &mut bool, s: &mut Scratch) {
    let n = w * h;
    let layers = input.layers;
    let mut k = 0;
    while k < list.len() {
        let li = list[k];
        // The clipped layers stacked on this base.
        let mut e = k + 1;
        while e < list.len() && layers[list[e]].clip {
            e += 1;
        }
        let clipped = &list[k + 1..e];
        k = e;
        let layer = &layers[li];
        if !layer.visible || layer.opacity <= 0.0 {
            continue;
        }
        let op = (layer.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        match &layer.kind {
            LayerKind::Adjustment(f) => {
                if !*empty {
                    apply_adjustment(layer, f, level, x0, y0, w, h, acc, s);
                }
                continue;
            }
            LayerKind::Group { pass_through: true, .. } if layer.mask.is_none() && op == 255 && clipped.is_empty() && !super::styles::is_styled(layer) && !advanced(layer) => {
                let kids = siblings(layers, Some(layer.id));
                compose_list(input, &kids, level, x0, y0, w, h, acc, empty, s);
                continue;
            }
            _ => {}
        }
        let mut buf = s.take(n);
        if !render_item(input, li, level, x0, y0, w, h, &mut buf, s) {
            s.give(buf);
            continue;
        }
        if clipped.iter().any(|&c| layers[c].visible && layers[c].opacity > 0.0) {
            // Clipping group: clipped layers blend onto the base and are
            // limited to the base's alpha ("atop").
            let bound_alpha: Vec<u8> = buf.iter().map(|&p| px::alpha(p) as u8).collect();
            // A styled base clips to its shape, not to its effects; the
            // styled result's alpha still bounds the group.
            let shape_alpha = super::styles::is_styled(layer).then(|| {
                let mut shape = s.take(n);
                let ok = render_unstyled(input, li, level, x0, y0, w, h, &mut shape, s);
                let a: Vec<u8> = shape.iter().map(|&p| if ok { px::alpha(p) as u8 } else { 0 }).collect();
                s.give(shape);
                a
            });
            let base_alpha: &[u8] = shape_alpha.as_deref().unwrap_or(&bound_alpha);
            let mut cbuf = s.take(n);
            for &c in clipped {
                let cl = &layers[c];
                if !cl.visible || cl.opacity <= 0.0 {
                    continue;
                }
                let cop = (cl.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                if let LayerKind::Adjustment(f) = &cl.kind {
                    apply_adjustment(cl, f, level, x0, y0, w, h, &mut buf, s);
                    continue;
                }
                cbuf.fill(0);
                if !render_item(input, c, level, x0, y0, w, h, &mut cbuf, s) {
                    continue;
                }
                for (p, &a) in cbuf.iter_mut().zip(base_alpha) {
                    *p = px::scale(*p, a as u32);
                }
                blend_layer_region(&mut buf, &cbuf, cop, cl,w,(x0,y0));
                for (p, &a) in buf.iter_mut().zip(&bound_alpha) {
                    let pa = px::alpha(*p);
                    if pa > a as u32 {
                        *p = if a == 0 { 0 } else { px::scale(*p, (a as u32 * 255 + pa / 2) / pa) };
                    }
                }
            }
            s.give(cbuf);
        }
        if *empty && layer.blend != BlendMode::Dissolve && !advanced(layer) {
            if op == 255 {
                acc.copy_from_slice(&buf);
            } else {
                acc.iter_mut().zip(&buf).for_each(|(o, &l)| *o = scale(l, op));
            }
        } else {
            if *empty {acc.fill(0);}
            blend_layer_region(acc, &buf, op, layer,w,(x0,y0));
        }
        *empty = false;
        s.give(buf);
    }
}

const CHECK_A: u32 = 0xFFFF_FFFF;
const CHECK_B: u32 = 0xFFCC_CCCC;

/// Renders display tile `(tx, ty)` of `level` (negative = magnified, nearest
/// neighbor) as opaque BGRA over a checkerboard, `DT x DT` with gutter.
pub fn display_tile(input: &ComposeInput, level: i32, tx: i32, ty: i32, out: &mut [u8], s: &mut Scratch) {
    let gx0 = tx * TS_I - 1;
    let gy0 = ty * TS_I - 1;
    let mut acc = std::mem::take(&mut s.acc);
    acc.resize(DT * DT, 0);
    if level >= 0 {
        compose_region(input, level as u32, gx0, gy0, DT, DT, &mut acc, s);
    } else {
        let k = (-level) as u32;
        let sx0 = gx0 >> k;
        let sy0 = gy0 >> k;
        let sw = (((gx0 + DT as i32 - 1) >> k) - sx0 + 1) as usize;
        let sh = (((gy0 + DT as i32 - 1) >> k) - sy0 + 1) as usize;
        let mut src = std::mem::take(&mut s.src);
        src.resize(sw * sh, 0);
        compose_region(input, 0, sx0, sy0, sw, sh, &mut src, s);
        for v in 0..DT {
            let sy = (((gy0 + v as i32) >> k) - sy0) as usize;
            let row = &src[sy * sw..(sy + 1) * sw];
            for u in 0..DT {
                acc[v * DT + u] = row[(((gx0 + u as i32) >> k) - sx0) as usize];
            }
        }
        s.src = src;
    }
    for v in 0..DT {
        let gy = gy0 + v as i32;
        for u in 0..DT {
            let gx = gx0 + u as i32;
            let check = if ((gx >> 3) ^ (gy >> 3)) & 1 == 0 { CHECK_A } else { CHECK_B };
            let p = px::over(acc[v * DT + u], check);
            let [r, g, b, _] = px::unpack(p);
            let o = (v * DT + u) * 4;
            out[o] = b;
            out[o + 1] = g;
            out[o + 2] = r;
            out[o + 3] = 255;
        }
    }
    s.acc = acc;
}

/// Composites a subset of layers into a single level-0 grid (merge/flatten).
pub fn flatten_layers(layers: &[Layer], w: u32, h: u32) -> TileGrid {
    let input = ComposeInput { layers, preview: None, extra: None };
    super::grid::build_grid(w, h, |tx, ty| compose_tile(&input, tx, ty))
}

fn compose_tile(input: &ComposeInput, tx: i32, ty: i32) -> TileRef {
    // Share a tile untouched when exactly one visible layer has content there
    // and it would be copied verbatim.
    let r = TileGrid::tile_rect(tx, ty);
    if !input.layers.iter().all(Layer::is_plain) {
        let mut s = Scratch::default();
        let mut out = vec![0u32; TS * TS].into_boxed_slice();
        compose_region(input, 0, r.x0, r.y0, TS, TS, &mut out, &mut s);
        return finish_tile(out);
    }
    let mut only: Option<&Layer> = None;
    let mut count = 0;
    for l in input.layers {
        if l.visible && l.opacity > 0.0 && l.grid.region_has_content(r.translate(-l.offset.0, -l.offset.1)) {
            count += 1;
            only = Some(l);
        }
    }
    if count == 0 {
        return None;
    }
    if count == 1 {
        let l = only.unwrap();
        if l.offset == (0, 0) && l.opacity >= 1.0 {
            return l.grid.tiles[l.grid.idx(tx, ty)].clone();
        }
    }
    let mut s = Scratch::default();
    let mut out = vec![0u32; TS * TS].into_boxed_slice();
    compose_region(input, 0, r.x0, r.y0, TS, TS, &mut out, &mut s);
    finish_tile(out)
}

/// Full-resolution straight-alpha RGBA8 export, optionally over a matte color.
pub fn export_rgba(layers: &[Layer], w: u32, h: u32, matte: Option<[u8; 3]>) -> Vec<u8> {
    let input = ComposeInput { layers, preview: None, extra: None };
    let (w, h) = (w as usize, h as usize);
    let mut buf = vec![0u8; w * h * 4];
    let cols = w.div_ceil(TS);
    let matte_px = matte.map(|m| px::pack(m[0], m[1], m[2], 255));
    buf.par_chunks_mut(w * 4 * TS).enumerate().for_each(|(ty, strip)| {
        let mut s = Scratch::default();
        let mut tile = vec![0u32; TS * TS];
        let rows = strip.len() / (w * 4);
        for tx in 0..cols {
            compose_region(&input, 0, (tx * TS) as i32, (ty * TS) as i32, TS, TS, &mut tile, &mut s);
            let cw = TS.min(w - tx * TS);
            for y in 0..rows {
                let dst = &mut strip[(y * w + tx * TS) * 4..(y * w + tx * TS + cw) * 4];
                for (x, d) in dst.chunks_exact_mut(4).enumerate() {
                    let p = tile[y * TS + x];
                    let rgba = match matte_px {
                        Some(m) => px::unpack(px::over(p, m)),
                        None => px::unpremultiply(p),
                    };
                    d.copy_from_slice(&rgba);
                }
            }
        }
    });
    buf
}

/// The composited color at one image pixel.
pub fn sample(layers: &[Layer], x: i32, y: i32) -> u32 {
    let input = ComposeInput { layers, preview: None, extra: None };
    let mut s = Scratch::default();
    let mut out = [0u32; 1];
    compose_region(&input, 0, x, y, 1, 1, &mut out, &mut s);
    out[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::blend::BlendMode;
    use crate::engine::layers::LayerMask;
    use crate::engine::selection::MaskGrid;

    fn solid(id: u64, c: u32) -> Layer {
        Layer::new(id, format!("L{id}"), TileGrid::filled(512, 512, c))
    }

    fn at(layers: &[Layer], x: i32, y: i32) -> [u8; 4] {
        px::unpack(sample(layers, x, y))
    }

    #[test]
    fn advanced_blending_applies_on_empty_stacks_and_nested_groups() {
        use crate::engine::advanced_blend::{AdvancedBlend, Range};
        let settings = std::sync::Arc::new(AdvancedBlend {
            ranges: vec![Range { source: [0, 0, 100, 200], ..Range::default() }],
            restricted_channels: vec![],
        });
        let mut top = solid(1, px::pack(150,150,150,255));
        top.advanced_blend=Some(settings);
        assert_eq!(at(&[top.clone()],0,0),[75,75,75,128]);
        let mut group=Layer::new(2,"Group",TileGrid::new(512,512));
        group.kind=LayerKind::Group{pass_through:false,collapsed:false};
        top.parent=Some(2);
        assert_eq!(at(&[top,group],0,0),[75,75,75,128]);
    }

    #[test]
    fn restricted_channels_apply_in_layer_and_clipping_stacks() {
        use crate::engine::advanced_blend::AdvancedBlend;
        let base=solid(1,px::pack(12,34,56,255));
        let mut top=solid(2,px::pack(90,100,110,255));
        top.advanced_blend=Some(std::sync::Arc::new(AdvancedBlend{ranges:vec![],restricted_channels:vec![0,2]}));
        assert_eq!(at(&[base.clone(),top.clone()],0,0),[12,100,56,255]);
        top.clip=true;
        assert_eq!(at(&[base,top],0,0),[12,100,56,255]);
    }

    #[test]
    fn layer_mask_hides_pixels() {
        let base = solid(1, px::pack(0, 0, 255, 255));
        let mut top = solid(2, px::pack(255, 0, 0, 255));
        top.mask = Some(LayerMask::new(MaskGrid::from_selection(&Selection::rect(RectI::new(0, 0, 256, 512)), 512, 512)));
        let layers = vec![base, top];
        assert_eq!(at(&layers, 100, 100), [255, 0, 0, 255]);
        assert_eq!(at(&layers, 400, 100), [0, 0, 255, 255]);
        // Mip levels agree (display path).
        let input = ComposeInput { layers: &layers, preview: None, extra: None };
        let mut s = Scratch::default();
        let mut out = vec![0u32; 4];
        compose_region(&input, 1, 50, 50, 2, 2, &mut out, &mut s);
        assert_eq!(px::unpack(out[0]), [255, 0, 0, 255]);
        compose_region(&input, 1, 200, 50, 2, 2, &mut out, &mut s);
        assert_eq!(px::unpack(out[0]), [0, 0, 255, 255]);
    }

    #[test]
    fn clipping_mask_limits_to_base_alpha() {
        let bg = solid(1, px::pack(255, 255, 255, 255));
        // Base: opaque only on the left half.
        let mut half = TileGrid::new(512, 512);
        half.tiles[0] = Some(std::sync::Arc::new(crate::engine::grid::Tile::solid(px::pack(0, 255, 0, 255))));
        half.tiles[2] = half.tiles[0].clone();
        let base = Layer::new(2, "base", half);
        let mut clipped = solid(3, px::pack(255, 0, 0, 255));
        clipped.clip = true;
        let layers = vec![bg, base, clipped];
        assert_eq!(at(&layers, 100, 100), [255, 0, 0, 255]);
        assert_eq!(at(&layers, 400, 100), [255, 255, 255, 255]);
        // Hidden base hides its clipped layers.
        let mut l2 = layers.clone();
        l2[1].visible = false;
        assert_eq!(at(&l2, 100, 100), [255, 255, 255, 255]);
    }

    #[test]
    fn groups_and_adjustments() {
        let base = solid(1, px::pack(200, 100, 50, 255));
        let mut grp = Layer::new(10, "Group", TileGrid::new(512, 512));
        grp.kind = LayerKind::Group { pass_through: false, collapsed: false };
        let mut child = solid(2, px::pack(0, 0, 0, 255));
        child.parent = Some(10);
        child.blend = BlendMode::Screen;
        // Isolated group: screen over transparency inside the group = black,
        // then the group (normal, 50%) over base.
        grp.opacity = 0.5;
        let layers = vec![base.clone(), child.clone(), grp.clone()];
        let c = at(&layers, 10, 10);
        assert!((c[0] as i32 - 100).abs() <= 1 && (c[1] as i32 - 50).abs() <= 1, "{c:?}");
        // Pass-through: the child's screen-with-black blends with base = base.
        let mut pt = grp.clone();
        pt.kind = LayerKind::Group { pass_through: true, collapsed: false };
        pt.opacity = 1.0;
        let layers = vec![base.clone(), child, pt];
        assert_eq!(at(&layers, 10, 10), [200, 100, 50, 255]);
        // Adjustment layer inverts everything below it, masked to the left half.
        let mut adj = Layer::new(20, "Invert", TileGrid::new(512, 512));
        adj.kind = LayerKind::Adjustment(Filter::Invert);
        adj.mask = Some(LayerMask::new(MaskGrid::from_selection(&Selection::rect(RectI::new(0, 0, 256, 512)), 512, 512)));
        let layers = vec![base, adj];
        assert_eq!(at(&layers, 10, 10), [55, 155, 205, 255]);
        assert_eq!(at(&layers, 300, 10), [200, 100, 50, 255]);
    }

    #[test]
    fn flatten_matches_display_with_kinds() {
        let base = solid(1, px::pack(10, 20, 30, 255));
        let mut top = solid(2, px::pack(250, 0, 0, 255));
        top.mask = Some(LayerMask::new(MaskGrid::from_selection(&Selection::ellipse(RectI::new(50, 50, 450, 450)), 512, 512)));
        let layers = vec![base, top];
        let flat = flatten_layers(&layers, 512, 512);
        for &(x, y) in &[(256, 256), (5, 5), (60, 256), (52, 256)] {
            assert_eq!(flat.pixel(x, y), sample(&layers, x, y));
        }
    }
}
