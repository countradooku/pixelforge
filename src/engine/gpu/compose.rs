//! The GPU compositor for display tiles.
//!
//! It mirrors `composite::compose_region` decision for decision (the same
//! content checks, groups, clipping, masks and adjustment layers), but runs
//! the pixel work as GPU passes over one region covering every display tile
//! the viewer is missing, reading layers from the tile pool (`tiles`), so
//! after an edit only the changed tiles are uploaded. A plain stack is one
//! fused pass. The result is identical to the CPU compositor: Normal blending
//! is exact integer math; the other modes are float on both sides (within
//! 1/255).
//!
//! Anything it doesn't handle (artboards, live moves revealing hidden
//! pixels, generator adjustments, unsupported preview filters) returns
//! `Unsupported` and the viewer composes on the CPU.

use std::sync::Arc;

use super::device::Gpu;
use super::exec::{self, Buf, GResult, GpuError, STORAGE, buffer, dispatch, encoder, groups, submit, table as table_buf, wait};
use super::shaders::PipeKey;
use super::tiles::{self, Table};
use crate::engine::composite::{ComposeInput, DT, siblings};
use crate::engine::document::Layer;
use crate::engine::grid::{RectI, TS_I};
use crate::engine::layers::LayerKind;

const NO_MASK: u32 = 0xFFFF_FFFF;

/// Everything one compose pass records: tile tables (concatenated into one
/// buffer at the end), buffers, and the dispatches that use table offsets.
struct Frame<'a> {
    g: &'a Arc<Gpu>,
    input: &'a ComposeInput<'a>,
    level: u32,
    rect: RectI,
    enc: wgpu::CommandEncoder,
    words: Vec<u32>,
    staging: Vec<Buf>,
    epoch: u64,
    pool: Option<Arc<Buf>>,
    /// Recorded passes, run once the table buffer exists. They refer to
    /// buffers by index, so buffers are only ever appended.
    ops: Vec<Op>,
    /// The cached composite of the layers below was reused.
    prefix_hit: bool,
    /// Tables of fused adjustment chains (`adjust_chain`), as u32 words;
    /// starts with the unpremultiply table when used.
    lut_words: Vec<u32>,
}

/// Foreground composite caching: the composite of the unchanged layers
/// below the one being edited, over the visible region. Only plain stacks
/// use it (no groups, clipping or masks), and its key holds every
/// contributing layer's tiles and properties, so any change below the
/// edited layer — or to the view — rebuilds it instead of showing a stale
/// backdrop.
/// How often the cached composite below the edited layer was reused.
pub static PREFIX_HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub struct Prefix {
    level: u32,
    view: RectI,
    keys: Vec<u64>,
    len: usize,
    buf: Arc<Buf>,
}

/// Identity of a layer's contribution to `view`: properties plus the tiles
/// it reads there.
fn layer_key(l: &Layer, level: u32, view: RectI) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |v: u64| {
        h = (h ^ v).wrapping_mul(0x100_0000_01b3);
        h ^= h >> 31;
    };
    mix(l.id);
    mix(l.opacity.to_bits() as u64);
    mix(l.blend.index() as u64);
    mix(((l.offset.0 as u32 as u64) << 32) | l.offset.1 as u32 as u64);
    let grid = l.level(level);
    let r = view.translate(-(l.offset.0 >> level), -(l.offset.1 >> level));
    let (tx0, ty0, tx1, ty1) = tiles::span(r);
    for ty in ty0..ty1 {
        for tx in tx0..tx1 {
            mix(grid.get(tx, ty).map_or(0, |t| Arc::as_ptr(t) as usize as u64));
        }
    }
    h
}

enum Op {
    /// (entry, params, acc, aux, aux2)
    Pass(&'static str, [u32; 12], usize, Option<usize>, Option<usize>),
    /// An adjustment's point filter: (filter kernel, src, dst).
    Point(Box<dyn super::kernels::Kernel>, usize, usize),
    /// Copies a sub-rect of a GPU-resident buffer (a live preview region):
    /// (src, src stride, src rows, offset x, offset y, dst).
    CopyIn(Arc<Buf>, u32, u32, i32, i32, usize),
    /// A Layer Style: (plan, padded unstyled src, dst, padded size, pad,
    /// layer-local origin of src[0]).
    Style(Arc<crate::engine::styles::Plan>, usize, usize, (u32, u32), u32, (i32, i32)),
}

impl<'a> Frame<'a> {
    fn n(&self) -> u32 {
        (self.rect.w() * self.rect.h()) as u32
    }

    /// Appends a table; returns its word offset.
    fn push_table(&mut self, t: &Table) -> u32 {
        let off = self.words.len() as u32;
        self.words.extend(t.words());
        off
    }

    fn take(&mut self, bufs: &mut Vec<Arc<Buf>>) -> GResult<usize> {
        bufs.push(Arc::new(buffer(self.g, self.n() as u64 * 4, STORAGE)?));
        Ok(bufs.len() - 1)
    }

    /// The descriptor of a layer's pixels (and mask) for this region:
    /// 10 words, see `compose.wgsl`.
    fn layer_desc(&mut self, layer: &Layer) -> GResult<Vec<u32>> {
        let rect = self.rect;
        self.layer_desc_in(layer, rect)
    }

    fn layer_desc_in(&mut self, layer: &Layer, rect: RectI) -> GResult<Vec<u32>> {
        let lv = self.level;
        let (ox, oy) = (layer.offset.0 >> lv, layer.offset.1 >> lv);
        let r = rect.translate(-ox, -oy);
        let (t, pool) = tiles::table(self.g, &mut self.enc, self.epoch, layer.level(lv), r, &mut self.staging)?;
        self.pool = Some(pool);
        let toff = self.push_table(&t);
        let mut d = vec![toff, r.x0 as u32, r.y0 as u32, 0, 0, NO_MASK, 0, 0, 0, 0];
        if let Some(m) = layer.mask.as_ref().filter(|m| m.enabled) {
            let mg = m.level(lv);
            let mo = self.mask_table(mg, r)?;
            d[5] = mo;
            d[6] = r.x0 as u32;
            d[7] = r.y0 as u32;
            d[8] = mg.w;
            d[9] = mg.h;
        }
        Ok(d)
    }

    /// A mask's table for a region (coordinates clamped to the mask).
    fn mask_table(&mut self, mg: &crate::engine::selection::MaskGrid, r: RectI) -> GResult<u32> {
        let (mw, mh) = (mg.w as i32, mg.h as i32);
        let c = RectI::new(r.x0.clamp(0, mw - 1), r.y0.clamp(0, mh - 1), r.x1.clamp(1, mw), r.y1.clamp(1, mh));
        let t = tiles::mask_table(self.g, &mut self.enc, self.epoch, mg, c, &mut self.staging)?;
        Ok(self.push_table(&t))
    }

    fn pass(&mut self, entry: &'static str, params: [u32; 12], acc: usize, aux: Option<usize>, aux2: Option<usize>) {
        self.ops.push(Op::Pass(entry, params, acc, aux, aux2));
    }

    fn params(&self) -> [u32; 12] {
        let mut p = [0u32; 12];
        p[0] = self.rect.w() as u32;
        p[1] = self.rect.h() as u32;
        p[6] = self.rect.x0 as u32;
        p[7] = self.rect.y0 as u32;
        p
    }
}

/// Whether the GPU compositor can do this document (else the CPU does).
pub fn supported(input: &ComposeInput) -> bool {
    input.layers.iter().all(|l| {
        let ok_kind = match &l.kind {
            LayerKind::Artboard(_) => false,
            LayerKind::Adjustment(f) => !f.generates() && super::kernels::point::supports(f),
            _ => true,
        };
        ok_kind && (l.hidden.is_none() || l.offset == (0, 0))
            && l.advanced_blend.as_ref().is_none_or(|settings| settings.is_identity())
    })
}

/// Composes `rect` (at `level`) into a GPU buffer. Returns the buffer (rect
/// sized, premultiplied) after recording every pass in a new encoder.
fn compose(g: &Arc<Gpu>, input: &ComposeInput, level: u32, rect: RectI, view: Option<RectI>, epoch: u64) -> GResult<(wgpu::CommandEncoder, Vec<Arc<Buf>>, usize, Vec<Buf>, Option<wgpu::Buffer>, Vec<super::kernels::Temps>)> {
    let mut f = Frame { g, input, level, rect, enc: encoder(g), words: Vec::new(), staging: Vec::new(), epoch, pool: None, ops: Vec::new(), prefix_hit: false, lut_words: Vec::new() };
    let mut bufs: Vec<Arc<Buf>> = Vec::new();
    let mut acc = f.take(&mut bufs)?;
    let plain = input.layers.iter().all(Layer::is_plain) && input.preview.is_none();
    if plain {
        // One fused pass over every visible layer with content here,
        // starting from the cached composite of the unchanged layers below
        // when there is one (foreground composite caching).
        let shown: Vec<&Layer> = input.layers.iter().filter(|l| l.visible && l.opacity > 0.0).collect();
        let mut start = 0;
        let mut base: Option<(Arc<Buf>, RectI)> = None;
        if let Some(view) = view.filter(|v| v.contains_rect(&rect)) {
            let keys: Vec<u64> = shown.iter().map(|l| layer_key(l, level, view)).collect();
            let mut cache = g.prefix.lock().unwrap();
            let same = cache.as_ref().filter(|c| c.level == level && c.view == view);
            let first_diff = same.map(|c| c.keys.iter().zip(&keys).take_while(|(a, b)| a == b).count().min(c.keys.len().min(keys.len())));
            match (same, first_diff) {
                (Some(c), Some(d)) if c.len > 0 && d >= c.len => {
                    // The layers below the edit are unchanged: reuse them.
                    start = c.len;
                    base = Some((c.buf.clone(), view));
                    f.prefix_hit = true;
                    PREFIX_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                _ => {
                    // (Re)build the prefix below the first changed layer (the
                    // top layer when nothing is known yet).
                    let len = match first_diff {
                        Some(d) if d < keys.len() => d,
                        _ => keys.len().saturating_sub(1),
                    };
                    if len > 0 {
                        let buf = Arc::new(buffer(g, (view.w() * view.h()) as u64 * 4, STORAGE)?);
                        let mut descs = Vec::new();
                        for layer in &shown[..len] {
                            let r = view.translate(-(layer.offset.0 >> level), -(layer.offset.1 >> level));
                            if !layer.level(level).region_has_content(r) {
                                continue;
                            }
                            let mut d = f.layer_desc_in(layer, view)?;
                            d[3] = (layer.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                            d[4] = layer.blend.index() as u32;
                            descs.push(d);
                        }
                        let off = f.words.len() as u32;
                        let n = descs.len() as u32;
                        f.words.extend(descs.into_iter().flatten());
                        let mut p = [0u32; 12];
                        p[0] = view.w() as u32;
                        p[1] = view.h() as u32;
                        p[2] = n;
                        p[5] = off;
                        p[6] = view.x0 as u32;
                        p[7] = view.y0 as u32;
                        bufs.push(buf.clone());
                        let bi = bufs.len() - 1;
                        f.pass("stack", p, bi, None, None);
                        *cache = Some(Prefix { level, view, keys: keys.clone(), len, buf: buf.clone() });
                        start = len;
                        base = Some((buf, view));
                    } else {
                        *cache = None;
                    }
                }
            }
        }
        let mut descs = Vec::new();
        for layer in &shown[start..] {
            let r = rect.translate(-(layer.offset.0 >> level), -(layer.offset.1 >> level));
            if !layer.level(level).region_has_content(r) {
                continue;
            }
            let mut d = f.layer_desc(layer)?;
            d[3] = (layer.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
            d[4] = layer.blend.index() as u32;
            descs.push(d);
        }
        let off = f.words.len() as u32;
        let n = descs.len() as u32;
        f.words.extend(descs.into_iter().flatten());
        let mut p = f.params();
        p[2] = n;
        p[5] = off;
        let aux = base.map(|(b, v)| {
            p[8] = 1;
            p[9] = v.w() as u32;
            p[10] = (rect.x0 - v.x0) as u32;
            p[11] = (rect.y0 - v.y0) as u32;
            bufs.push(b);
            bufs.len() - 1
        });
        f.pass("stack", p, acc, aux, None);
    } else {
        let mut z = f.params();
        z[2] = 0;
        f.pass("zero", z, acc, None, None);
        let top = siblings(input.layers, None);
        let mut empty = true;
        list(&mut f, &mut bufs, &top, &mut acc, &mut empty)?;
    }
    if let Some(x) = input.extra {
        let lv = level;
        let r = rect.translate(-(x.offset.0 >> lv), -(x.offset.1 >> lv));
        if x.level(lv).region_has_content(r) {
            let mut d = f.layer_desc(x)?;
            d[3] = (x.opacity.clamp(0.0, 1.0) * 255.0).round() as u32;
            d[4] = 0;
            let off = f.words.len() as u32;
            f.words.extend(d);
            let mut p = f.params();
            p[2] = 1;
            p[5] = off;
            p[8] = 1;
            p[9] = rect.w() as u32;
            let out = f.take(&mut bufs)?;
            f.pass("stack", p, out, Some(acc), None);
            acc = out;
        }
    }
    // Run the recorded passes now that every table is known.
    let tab = table_buf(g, bytemuck::cast_slice(&f.words));
    let pool = match f.pool.clone() {
        Some(p) => p,
        None => {
            let (_, p) = tiles::table(g, &mut f.enc, epoch, &crate::engine::grid::TileGrid::new(1, 1), RectI::new(0, 0, 1, 1), &mut f.staging)?;
            p
        }
    };
    let dummy = g.shared("zero16", || vec![0u8; 16]);
    let luts = (!f.lut_words.is_empty()).then(|| table_buf(g, bytemuck::cast_slice(&f.lut_words)));
    // Style scratch buffers live until the work is waited.
    let mut style_temps = Vec::new();
    let ops = std::mem::take(&mut f.ops);
    for op in ops {
        match op {
            Op::Pass(entry, params, a, x, y) => {
                let key = PipeKey { module: "compose", entry, f16: false, consts: Vec::new() };
                let mut binds: Vec<(u32, &wgpu::Buffer)> = Vec::new();
                for &b in bindings(entry) {
                    binds.push((
                        b,
                        match b {
                            1 => &pool,
                            2 => &tab,
                            3 => &bufs[a],
                            4 => x.map_or(&*dummy, |x| &bufs[x]),
                            6 => luts.as_ref().unwrap_or(&*dummy),
                            _ => y.map_or(&*dummy, |y| &bufs[y]),
                        },
                    ));
                }
                dispatch(g, &mut f.enc, None, &key, &params, &binds, [groups(params[0], 16), groups(params[1], 16), 1])?;
            }
            Op::CopyIn(src, stride, rows, ox, oy, dst) => {
                let (w, h) = (rect.w() as u32, rect.h() as u32);
                let key = PipeKey { module: "util", entry: "region", f16: false, consts: Vec::new() };
                let params = [stride, rows, stride, rows, ox as u32, oy as u32, w, h, 0];
                dispatch(g, &mut f.enc, None, &key, &params, &[(1, &src), (2, &bufs[dst])], [groups(w, 16), groups(h, 16), 1])?;
            }
            Op::Style(plan, src, dst, (pw, ph), pad, origin) => {
                let io = super::kernels::Io { src: &bufs[src], dst: &bufs[dst], pw, ph, pad, level, origin, img: (0, 0), source: &[] };
                let mut temps = super::kernels::Temps::default();
                super::styles::render(g, &mut f.enc, &plan, &io, &mut temps, None)?;
                style_temps.push(temps);
            }
            Op::Point(k, src, dst) => {
                let (w, h) = (rect.w() as u32, rect.h() as u32);
                let io = super::kernels::Io { src: &bufs[src], dst: &bufs[dst], pw: w, ph: h, pad: 0, level, origin: (rect.x0, rect.y0), img: (0, 0), source: &[] };
                k.encode(g, &mut f.enc, &io, &mut Default::default(), None)?;
            }
        }
    }
    Ok((f.enc, bufs, acc, f.staging, Some(tab), style_temps))
}

/// The bindings each compose entry point uses (its auto layout).
fn bindings(entry: &str) -> &'static [u32] {
    match entry {
        "stack" | "mix" => &[1, 2, 3, 4],
        "adjust_chain" => &[1, 2, 3, 6],
        "load" => &[1, 2, 3],
        "blend_buf" | "copy_scaled" => &[3, 4],
        "zero" => &[3],
        "by_alpha" | "select_mix" => &[3, 4, 5],
        "clamp_alpha" => &[3, 5],
        _ => &[2, 3, 4],
    }
}

/// `compose_list` on the GPU.
fn list(f: &mut Frame, bufs: &mut Vec<Arc<Buf>>, items: &[usize], acc: &mut usize, empty: &mut bool) -> GResult<()> {
    let layers = f.input.layers;
    // Consecutive adjustment layers, applied together (fused when they're
    // all LUT-type) before the next non-adjustment item.
    let mut chain: Vec<usize> = Vec::new();
    let mut k = 0;
    while k < items.len() {
        if !chain.is_empty() && !matches!(layers[items[k]].kind, LayerKind::Adjustment(_)) {
            *acc = adjust_run(f, bufs, &std::mem::take(&mut chain), *acc)?;
        }
        let li = items[k];
        let mut e = k + 1;
        while e < items.len() && layers[items[e]].clip {
            e += 1;
        }
        let clipped: Vec<usize> = items[k + 1..e].to_vec();
        k = e;
        let layer = &layers[li];
        if !layer.visible || layer.opacity <= 0.0 {
            continue;
        }
        let op = (layer.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        match &layer.kind {
            LayerKind::Adjustment(_) => {
                if !*empty {
                    chain.push(li);
                }
                continue;
            }
            LayerKind::Group { pass_through: true, .. } if layer.mask.is_none() && op == 255 && clipped.is_empty() && !crate::engine::styles::is_styled(layer) => {
                let kids = siblings(layers, Some(layer.id));
                list(f, bufs, &kids, acc, empty)?;
                continue;
            }
            _ => {}
        }
        let Some(mut buf) = item(f, bufs, li)? else { continue };
        if clipped.iter().any(|&c| layers[c].visible && layers[c].opacity > 0.0) {
            // Base alphas, then each clipped layer "atop" the base.
            // Bound: the (styled) base's alpha. A styled base clips its
            // clipped layers to its shape, not its effects.
            let base = f.take(bufs)?;
            let mut p = f.params();
            p[3] = 255;
            f.pass("copy_scaled", p, base, Some(buf), None);
            let shape = if crate::engine::styles::is_styled(layer) {
                match item_unstyled(f, bufs, li)? {
                    Some(sb) => sb,
                    None => {
                        let z = f.take(bufs)?;
                        let mut zp = f.params();
                        zp[2] = 0;
                        f.pass("zero", zp, z, None, None);
                        z
                    }
                }
            } else {
                base
            };
            for &c in &clipped {
                let cl = &layers[c];
                if !cl.visible || cl.opacity <= 0.0 {
                    continue;
                }
                let cop = (cl.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                if let LayerKind::Adjustment(filt) = &cl.kind {
                    buf = adjustment(f, bufs, cl, filt, buf)?;
                    continue;
                }
                let Some(cb) = item(f, bufs, c)? else { continue };
                let scaled = f.take(bufs)?;
                f.pass("by_alpha", f.params(), scaled, Some(cb), Some(shape));
                let mut p = f.params();
                p[3] = cop;
                p[4] = cl.blend.index() as u32;
                f.pass("blend_buf", p, buf, Some(scaled), None);
                f.pass("clamp_alpha", f.params(), buf, None, Some(base));
            }
        }
        let mut p = f.params();
        p[3] = op;
        if *empty && layer.blend != crate::engine::blend::BlendMode::Dissolve {
            f.pass("copy_scaled", p, *acc, Some(buf), None);
        } else {
            if *empty {f.pass("zero",f.params(),*acc,None,None);}
            p[4] = layer.blend.index() as u32;
            f.pass("blend_buf", p, *acc, Some(buf), None);
        }
        *empty = false;
    }
    if !chain.is_empty() {
        *acc = adjust_run(f, bufs, &chain, *acc)?;
    }
    Ok(())
}

/// A filter an `adjust_chain` pass can run: (kind, table bytes).
fn chain_op(filt: &crate::engine::filters::Filter) -> Option<(u32, Vec<u8>)> {
    use crate::engine::filters::Filter;
    if let Some(l) = filt.point_luts() {
        return Some((0, l.concat()));
    }
    match filt {
        Filter::Color { lut, .. } => {
            let mut b = Vec::with_capacity(lut.raw().len() * 8);
            for c in lut.raw() {
                b.extend_from_slice(&(c[0] as u32 | (c[1] as u32) << 16).to_le_bytes());
                b.extend_from_slice(&(c[2] as u32).to_le_bytes());
            }
            Some((1, b))
        }
        Filter::Threshold { level } => Some((2, bytemuck::cast_slice(&crate::engine::filters::threshold_cut(*level)).to_vec())),
        Filter::Desaturate => Some((3, Vec::new())),
        _ => None,
    }
}

/// Applies a run of adjustment layers to `acc`: one fused pass when every
/// one is LUT-type, else one by one.
fn adjust_run(f: &mut Frame, bufs: &mut Vec<Arc<Buf>>, run: &[usize], acc: usize) -> GResult<usize> {
    let layers = f.input.layers;
    let filt = |li: usize| match &layers[li].kind {
        LayerKind::Adjustment(x) => Some(x),
        _ => None,
    };
    let ops: Option<Vec<(u32, Vec<u8>)>> = run.iter().map(|&li| filt(li).and_then(chain_op)).collect();
    let Some(ops) = ops.filter(|_| run.len() > 1) else {
        let mut a = acc;
        for &li in run {
            if let Some(x) = filt(li) {
                a = adjustment(f, bufs, &layers[li], x, a)?;
            }
        }
        return Ok(a);
    };
    if f.lut_words.is_empty() {
        f.lut_words = bytemuck::pod_collect_to_vec::<u8, u32>(&crate::engine::px::unpremul_table());
    }
    let lv = f.level;
    let mut words = vec![0u32];
    for (&li, (kind, tab)) in run.iter().zip(ops) {
        let layer = &layers[li];
        let op = (layer.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        let mut m = [NO_MASK, 0, 0, 0, 0];
        if let Some(mk) = layer.mask.as_ref().filter(|m| m.enabled) {
            let mg = mk.level(lv);
            match crate::engine::masks::classify_region(mg, f.rect) {
                crate::engine::masks::RegionCover::Empty => continue,
                crate::engine::masks::RegionCover::Full => {}
                crate::engine::masks::RegionCover::Partial => m = [f.mask_table(mg, f.rect)?, f.rect.x0 as u32, f.rect.y0 as u32, mg.w, mg.h],
            }
        }
        if op == 0 {
            continue;
        }
        let off = f.lut_words.len() as u32;
        let mut t = tab;
        t.resize(t.len().next_multiple_of(4), 0);
        f.lut_words.extend(bytemuck::pod_collect_to_vec::<u8, u32>(&t));
        words.extend_from_slice(&[kind, off, op, m[0], m[1], m[2], m[3], m[4]]);
        words[0] += 1;
    }
    if words[0] == 0 {
        return Ok(acc);
    }
    let off = f.words.len() as u32;
    f.words.extend(words);
    let mut p = f.params();
    p[5] = off;
    f.pass("adjust_chain", p, acc, None, None);
    Ok(acc)
}

/// `render_item`: a layer or a whole group into a new buffer, or None when
/// it adds nothing to the region. A styled layer (Layer Style / Fill) is
/// rendered unstyled over the region padded by the effects' reach, then
/// styled on the GPU (`gpu::styles`), like `composite::render_item`.
fn item(f: &mut Frame, bufs: &mut Vec<Arc<Buf>>, li: usize) -> GResult<Option<usize>> {
    let layer = &f.input.layers[li];
    let lv = f.level;
    let Some(plan) = crate::engine::styles::plan_for(layer, lv) else { return item_unstyled(f, bufs, li) };
    let p = plan.pad as i32;
    let saved = f.rect;
    f.rect = saved.inflate(p);
    let inner = item_unstyled(f, bufs, li);
    f.rect = saved;
    let Some(src) = inner? else { return Ok(None) };
    let out = f.take(bufs)?;
    let origin = (saved.x0 - p - (layer.offset.0 >> lv), saved.y0 - p - (layer.offset.1 >> lv));
    f.ops.push(Op::Style(plan, src, out, ((saved.w() + 2 * p) as u32, (saved.h() + 2 * p) as u32), p as u32, origin));
    Ok(Some(out))
}

fn item_unstyled(f: &mut Frame, bufs: &mut Vec<Arc<Buf>>, li: usize) -> GResult<Option<usize>> {
    let layers = f.input.layers;
    let layer = &layers[li];
    let lv = f.level;
    let r = f.rect.translate(-(layer.offset.0 >> lv), -(layer.offset.1 >> lv));
    let buf = match &layer.kind {
        LayerKind::Group { .. } => {
            let mut buf = f.take(bufs)?;
            let mut z = f.params();
            z[2] = 0;
            f.pass("zero", z, buf, None, None);
            let kids = siblings(layers, Some(layer.id));
            let mut empty = true;
            list(f, bufs, &kids, &mut buf, &mut empty)?;
            if empty {
                return Ok(None);
            }
            group_mask(f, bufs, layer, buf)?
        }
        LayerKind::Adjustment(_) => return Ok(None),
        _ => match f.input.preview.filter(|p| p.layer == li) {
            Some(p) => {
                let buf = f.take(bufs)?;
                preview_pixels(f, bufs, layer, p, buf)?;
                group_mask(f, bufs, layer, buf)?
            }
            None => {
                if !layer.level(lv).region_has_content(r) {
                    return Ok(None);
                }
                let d = f.layer_desc(layer)?;
                let off = f.words.len() as u32;
                f.words.extend(d);
                let buf = f.take(bufs)?;
                let mut p = f.params();
                p[5] = off;
                // `load` applies the mask too.
                f.pass("load", p, buf, None, None);
                buf
            }
        },
    };
    Ok(mask_reveals(layer, lv, r).then_some(buf))
}

/// Whether a layer's mask leaves anything of the region visible.
fn mask_reveals(layer: &Layer, lv: u32, r: RectI) -> bool {
    match layer.mask.as_ref().filter(|m| m.enabled) {
        Some(m) => !matches!(crate::engine::masks::classify_region(m.level(lv), r), crate::engine::masks::RegionCover::Empty),
        None => true,
    }
}

/// Applies a layer mask to an already rendered buffer (groups, previews);
/// returns the buffer holding the result. Coverage multiply is
/// `lerp(0, px, m)`, which is exactly `scale(px, m)`.
fn group_mask(f: &mut Frame, bufs: &mut Vec<Arc<Buf>>, layer: &Layer, buf: usize) -> GResult<usize> {
    let Some(m) = layer.mask.as_ref().filter(|m| m.enabled) else { return Ok(buf) };
    let lv = f.level;
    let r = f.rect.translate(-(layer.offset.0 >> lv), -(layer.offset.1 >> lv));
    let mg = m.level(lv);
    let mo = f.mask_table(mg, r)?;
    let out = f.take(bufs)?;
    let mut z = f.params();
    z[2] = 0;
    f.pass("zero", z, out, None, None);
    let mut p = f.params();
    p[3] = 255;
    p[5] = mo;
    p[6] = r.x0 as u32;
    p[7] = r.y0 as u32;
    p[9] = mg.w;
    p[10] = mg.h;
    f.pass("mix", p, out, Some(buf), None);
    Ok(out)
}

/// An adjustment layer: point filter of `acc`, mixed by opacity x mask.
fn adjustment(f: &mut Frame, bufs: &mut Vec<Arc<Buf>>, layer: &Layer, filt: &crate::engine::filters::Filter, acc: usize) -> GResult<usize> {
    let op = (layer.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    if op == 0 {
        return Ok(acc);
    }
    let lv = f.level;
    let mut mask = None;
    if let Some(m) = layer.mask.as_ref().filter(|m| m.enabled) {
        let mg = m.level(lv);
        match crate::engine::masks::classify_region(mg, f.rect) {
            crate::engine::masks::RegionCover::Empty => return Ok(acc),
            crate::engine::masks::RegionCover::Full => {}
            crate::engine::masks::RegionCover::Partial => mask = Some((f.mask_table(mg, f.rect)?, mg.w, mg.h)),
        }
    }
    let k = super::kernels::point::plan(f.g, filt).ok_or(GpuError::Unsupported)?;
    let tmp = f.take(bufs)?;
    f.ops.push(Op::Point(k, acc, tmp));
    let mut p = f.params();
    p[3] = op;
    match mask {
        Some((mo, mw, mh)) => {
            p[5] = mo;
            p[6] = f.rect.x0 as u32;
            p[7] = f.rect.y0 as u32;
            p[9] = mw;
            p[10] = mh;
        }
        None => p[5] = NO_MASK,
    }
    f.pass("mix", p, acc, Some(tmp), None);
    Ok(acc)
}

/// A live filter preview's pixels for the region, with its selection: the
/// GPU-resident preview result copied in on the GPU when there is one (no
/// readback), else the CPU's `Filter::render`.
fn preview_pixels(f: &mut Frame, bufs: &mut Vec<Arc<Buf>>, layer: &Layer, p: &crate::engine::composite::Preview, buf: usize) -> GResult<()> {
    let lv = f.level;
    let r = f.rect.translate(-(layer.offset.0 >> lv), -(layer.offset.1 >> lv));
    let (w, h) = (r.w() as usize, r.h() as usize);
    let mask = p.selection.as_ref().and_then(|sel| sel.region_coverage(lv, r.x0, r.y0, w, h));
    if let Some(reg) = p.gpu.gpu_region(lv, r) {
        f.ops.push(Op::CopyIn(reg.buf, reg.rect.w() as u32, reg.rect.h() as u32, r.x0 - reg.rect.x0, r.y0 - reg.rect.y0, buf));
        if let Some(m) = mask {
            // lerp(original, filtered, coverage) on the GPU.
            let mut d = f.layer_desc(layer)?;
            d[5] = NO_MASK;
            let off = f.words.len() as u32;
            f.words.extend(d);
            let orig = f.take(bufs)?;
            let mut lp = f.params();
            lp[5] = off;
            f.pass("load", lp, orig, None, None);
            let mut bytes = m;
            bytes.resize(bytes.len().next_multiple_of(4), 0);
            let cov = Arc::new(buffer(f.g, bytes.len() as u64, STORAGE)?);
            f.g.queue.write_buffer(&cov, 0, &bytes);
            bufs.push(cov);
            let ci = bufs.len() - 1;
            f.pass("select_mix", f.params(), buf, Some(orig), Some(ci));
        }
        return Ok(());
    }
    let mut px = vec![0u32; w * h];
    if !p.gpu.lookup(lv, r, &mut px) {
        match super::preview::render(f.g, &p.filter, layer, lv, r) {
            Ok(v) => px = v,
            Err(_) => p.filter.render(&crate::engine::filters::Src::layer(layer), lv, r.x0, r.y0, w, h, &mut px, &mut Vec::new()),
        }
    }
    if let Some(mask) = mask {
        let mut orig = vec![0u32; w * h];
        layer.level(lv).read_region(r.x0, r.y0, w, h, &mut orig);
        for ((l, &o), &m) in px.iter_mut().zip(&orig).zip(&mask) {
            *l = crate::engine::px::lerp(o, *l, m as u32);
        }
    }
    f.g.queue.write_buffer(&bufs[buf], 0, bytemuck::cast_slice(&px));
    Ok(())
}

/// Where display tiles go: back to the CPU as bytes, or into GPU textures
/// the UI draws directly.
enum Out {
    Bytes,
    Textures,
}

/// A display tile rendered into a GPU texture (BGRA8, `DT x DT`). `raw` is
/// the platform texture handle (`id<MTLTexture>` on Metal) for the UI's
/// renderer; the texture lives as long as this value.
pub struct GpuTile {
    _memory: crate::engine::memory::Reservation,
    /// Owns the texture `raw` points at.
    pub _tex: wgpu::Texture,
    pub raw: usize,
}

/// Composes display tiles `(tx, ty)` of `level` (negative = magnified) and
/// returns each as `DT x DT` BGRA bytes, exactly like `display_tile`.
/// `view`: the visible tile range (foreground caching).
pub fn display_tiles(g: &Arc<Gpu>, input: &ComposeInput, level: i32, tiles_xy: &[(i32, i32)], view: Option<(i32, i32, i32, i32)>) -> GResult<Vec<Vec<u8>>> {
    match run_display(g, input, level, tiles_xy, view, Out::Bytes)? {
        (b, _) => Ok(b),
    }
}

/// [`display_tiles`] into GPU textures (no readback).
pub fn display_textures(g: &Arc<Gpu>, input: &ComposeInput, level: i32, tiles_xy: &[(i32, i32)], view: Option<(i32, i32, i32, i32)>) -> GResult<Vec<GpuTile>> {
    match run_display(g, input, level, tiles_xy, view, Out::Textures)? {
        (_, t) => Ok(t),
    }
}

#[cfg(target_vendor = "apple")]
fn raw_texture(t: &wgpu::Texture) -> Option<usize> {
    // SAFETY: only reads the handle; the texture outlives every use (the
    // returned `GpuTile` owns it).
    unsafe { t.as_hal::<wgpu::hal::api::Metal>() }.map(|h| h.raw_handle() as *const _ as *const std::ffi::c_void as usize)
}

#[cfg(not(target_vendor = "apple"))]
fn raw_texture(_: &wgpu::Texture) -> Option<usize> {
    None
}

fn run_display(g: &Arc<Gpu>, input: &ComposeInput, level: i32, tiles_xy: &[(i32, i32)], view: Option<(i32, i32, i32, i32)>, out_kind: Out) -> GResult<(Vec<Vec<u8>>, Vec<GpuTile>)> {
    if tiles_xy.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if !supported(input) || super::is_broken("compose") {
        return Err(GpuError::Unsupported);
    }
    let (lvl, sh) = if level >= 0 { (level as u32, 0) } else { (0, (-level) as u32) };
    let tile_rect = |tx: i32, ty: i32| {
        let (gx0, gy0) = (tx * TS_I - 1, ty * TS_I - 1);
        RectI::new(gx0 >> sh, gy0 >> sh, ((gx0 + DT as i32 - 1) >> sh) + 1, ((gy0 + DT as i32 - 1) >> sh) + 1)
    };
    let mut r = RectI::EMPTY;
    for &(tx, ty) in tiles_xy {
        let t = tile_rect(tx, ty);
        r = if r.is_empty() { t } else { r.union(&t) };
    }
    // The visible tile range as a region of the compose level.
    let view = view.map(|(tx0, ty0, tx1, ty1)| tile_rect(tx0, ty0).union(&tile_rect(tx1 - 1, ty1 - 1)));
    let _one = g.compose_lock.lock().unwrap_or_else(|e| e.into_inner());
    let frame = tiles::next_frame(g);
    let scopes = exec::Scopes::push(g);
    let (mut enc, bufs, acc, _staging, _tab, _style_temps) = compose(g, input, lvl, r, view, frame.0)?;
    let n = tiles_xy.len();
    // Texture copies need 256-byte rows: 320 px per row.
    let row = match out_kind {
        Out::Bytes => DT as u32,
        Out::Textures => 320,
    };
    let tile_px = row * DT as u32;
    let out_bytes = n as u64 * tile_px as u64 * 4;
    let out = buffer(g, out_bytes, STORAGE)?;
    let list: Vec<u32> = tiles_xy.iter().flat_map(|&(x, y)| [x as u32, y as u32]).collect();
    let lt = table_buf(g, bytemuck::cast_slice(&list));
    let params = [r.w() as u32, r.h() as u32, 0, 0, 0, 0, r.x0 as u32, r.y0 as u32, sh, row, tile_px, 0];
    let key = PipeKey { module: "compose", entry: "display", f16: false, consts: Vec::new() };
    dispatch(g, &mut enc, None, &key, &params, &[(2, &lt), (3, &out), (4, &bufs[acc])], [groups(DT as u32, 16), groups(DT as u32, 16), n as u32])?;
    match out_kind {
        Out::Bytes => {
            let down = buffer(g, out_bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
            enc.copy_buffer_to_buffer(&out, 0, &down, 0, out_bytes);
            let idx = submit(g, enc);
            scopes.pop()?;
            wait(g, idx)?;
            drop(frame);
            down.map(out_bytes, wgpu::MapMode::Read)?;
            let res = {
                let view = down.slice(..out_bytes).get_mapped_range();
                (0..n).map(|i| view[i * DT * DT * 4..(i + 1) * DT * DT * 4].to_vec()).collect()
            };
            down.unmap_now();
            Ok((res, Vec::new()))
        }
        Out::Textures => {
            let mut texs = Vec::with_capacity(n);
            for i in 0..n {
                let memory = crate::engine::memory::global().reserve_gpu((DT * DT * 4) as u64).ok_or(GpuError::Vram)?;
                let tex = g.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("display tile"),
                    size: wgpu::Extent3d { width: DT as u32, height: DT as u32, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Bgra8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                enc.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &out,
                        layout: wgpu::TexelCopyBufferLayout { offset: i as u64 * tile_px as u64 * 4, bytes_per_row: Some(row * 4), rows_per_image: Some(DT as u32) },
                    },
                    tex.as_image_copy(),
                    wgpu::Extent3d { width: DT as u32, height: DT as u32, depth_or_array_layers: 1 },
                );
                texs.push((tex, memory));
            }
            let idx = submit(g, enc);
            scopes.pop()?;
            // The UI's renderer samples these on its own queue: they must
            // be complete before they're handed over.
            wait(g, idx)?;
            drop(frame);
            let mut tiles = Vec::with_capacity(n);
            for (tex, memory) in texs {
                let raw = raw_texture(&tex).ok_or(GpuError::Unsupported)?;
                tiles.push(GpuTile { _memory: memory, _tex: tex, raw });
            }
            Ok((Vec::new(), tiles))
        }
    }
}

/// Display tiles from the GPU compositor: textures the UI draws directly
/// (Metal), or bytes where the platform can't share textures.
pub enum Tiles {
    Bytes(Vec<Vec<u8>>),
    Textures(Vec<GpuTile>),
}

/// The viewer's entry point: display tiles on the GPU when compositing on
/// the GPU is enabled and the document is supported; None means compose on
/// the CPU. `textures`: hand back GPU textures (zero copy) instead of bytes.
/// Records the path for the Info panel.
pub fn try_display_tiles(input: &ComposeInput, level: i32, tiles_xy: &[(i32, i32)], view: Option<(i32, i32, i32, i32)>, textures: bool) -> Option<Tiles> {
    let m = super::feature_mode(super::Feature::Compositing);
    // A couple of tiles are quicker on the CPU than a GPU round trip.
    if m == super::Mode::Off || tiles_xy.is_empty() || (m == super::Mode::Auto && tiles_xy.len() < 3) || !supported(input) {
        return None;
    }
    let g = super::context(m == super::Mode::Force || super::blocking())?;
    let t = std::time::Instant::now();
    let textures = textures && cfg!(target_vendor = "apple");
    let r = if textures { display_textures(&g, input, level, tiles_xy, view).map(Tiles::Textures) } else { display_tiles(&g, input, level, tiles_xy, view).map(Tiles::Bytes) };
    match r {
        Ok(v) => {
            let how = if textures { "GPU textures" } else { "GPU" };
            super::record_compose(&format!("{} tiles ({how})", tiles_xy.len()), super::PathTaken::Gpu, t.elapsed().as_secs_f64() * 1e3);
            Some(v)
        }
        Err(e) => {
            match &e {
                GpuError::Validation(_) => super::mark_broken("compose"),
                e if e.is_fault() => g.fail(e.to_string()),
                _ => {}
            }
            None
        }
    }
}
