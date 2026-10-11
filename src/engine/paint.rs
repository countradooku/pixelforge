//! Painting kernels on the shared dab engine: Dodge / Burn / Sponge, Blur /
//! Sharpen, Smudge, Mixer Brush, Color Replacement and Background Eraser, plus
//! the Magic Eraser and pattern fills.
//!
//! - **Tone tools** bake a 256-entry LUT per channel for the chosen range
//!   (shadows / midtones / highlights) and exposure; the per-tile target is
//!   one LUT lookup per channel, and each dab only lerps toward it.
//! - **Blur / Sharpen** build their target for a tile from the stroke base:
//!   the padded tile is read once and run through the planar kernels of
//!   `filters::fast` (only tiles the stroke touches).
//! - **Smudge / Mixer** are sequential: every dab reads the current canvas
//!   under it and carries a dab-sized pickup buffer along the path.
//! - **Color Replacement / Background Eraser** weigh every stamped pixel by
//!   its colour distance to the sampled colour (continuous, once or the
//!   background swatch).

use rayon::prelude::*;

use super::brush::{BaseView, Dab, DabMask, DabShape, Erase, Kernel, lerp_rows};
use super::filters::util::{hsl_to_rgb, rgb_to_hsl};
use super::grid::{RectI, TS, TS_I, Tile, TileGrid, tile_mut};
use super::px;
use super::selection::{MaskGrid, MaskTile, Selection, TileCoverage};

// ----- dodge / burn / sponge -------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Range {
    Shadows,
    #[default]
    Midtones,
    Highlights,
}

impl Range {
    pub const ALL: [Range; 3] = [Range::Shadows, Range::Midtones, Range::Highlights];
    pub fn label(self) -> &'static str {
        match self {
            Range::Shadows => "Shadows",
            Range::Midtones => "Midtones",
            Range::Highlights => "Highlights",
        }
    }

    /// Weight of a tone value (0..1) in this range.
    fn weight(self, v: f32) -> f32 {
        match self {
            Range::Shadows => (1.0 - v).powi(2),
            Range::Midtones => 1.0 - (2.0 * v - 1.0).powi(2),
            Range::Highlights => v.powi(2),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tone {
    Dodge,
    Burn,
    Saturate,
    Desaturate,
}

/// Dodge / Burn / Sponge: lerps toward a LUT-mapped copy of the base.
pub struct ToneKernel {
    lut: Box<[u8; 256]>,
    mode: Tone,
    protect: bool,
    vibrance: bool,
    amount: f32,
}

impl ToneKernel {
    pub fn new(mode: Tone, range: Range, exposure: f32, protect: bool, vibrance: bool) -> Self {
        let e = exposure.clamp(0.0, 1.0);
        let mut lut = Box::new([0u8; 256]);
        for (i, o) in lut.iter_mut().enumerate() {
            let v = i as f32 / 255.0;
            let w = range.weight(v) * e;
            let out = match mode {
                // Dodge lifts toward white, burn pulls toward black, both
                // weighted by the range curve.
                Tone::Dodge => v + (1.0 - v) * w * 0.85,
                Tone::Burn => v - v * w * 0.85,
                _ => v,
            };
            *o = (out.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
        ToneKernel { lut, mode, protect, vibrance, amount: e }
    }

    #[inline]
    fn map(&self, p: u32) -> u32 {
        let a = px::alpha(p);
        if a == 0 {
            return p;
        }
        let [r, g, b, _] = px::unpremultiply(p);
        let out = match self.mode {
            Tone::Dodge | Tone::Burn => {
                if self.protect {
                    // Protect Tones: map luminance, scale the colour with it,
                    // so hue and saturation stay and channels don't clip apart.
                    let y = (r as f32 * 0.299 + g as f32 * 0.587 + b as f32 * 0.114).max(1.0);
                    let y2 = self.lut[y.min(255.0) as usize] as f32;
                    let k = y2 / y;
                    [r, g, b].map(|c| (c as f32 * k).clamp(0.0, 255.0) as u8)
                } else {
                    [self.lut[r as usize], self.lut[g as usize], self.lut[b as usize]]
                }
            }
            Tone::Saturate | Tone::Desaturate => {
                let c = [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0];
                let gray = c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114;
                let sat = c.iter().fold(0f32, |m, &v| m.max(v)) - c.iter().fold(1f32, |m, &v| m.min(v));
                let mut k = self.amount;
                if self.vibrance && self.mode == Tone::Saturate {
                    // Vibrance: boost muted colours more than saturated ones.
                    k *= (1.0 - sat).powi(2);
                }
                let f = if self.mode == Tone::Saturate { 1.0 + k } else { 1.0 - k };
                c.map(|v| ((gray + (v - gray) * f).clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
            }
        };
        px::premultiply(out[0], out[1], out[2], a as u8)
    }
}

impl Kernel for ToneKernel {
    fn prepare(&self, _tx: i32, _ty: i32, base: Option<&Tile>) -> Option<Box<[u32]>> {
        let base = base?;
        if let Some(u) = base.uniform {
            return Some(vec![self.map(u); TS * TS].into_boxed_slice());
        }
        Some(base.read().iter().map(|&p| self.map(p)).collect())
    }
    fn apply_row(&self, base: &[u32], target: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
        lerp_rows(base, target, alpha, out);
    }
}

// ----- blur / sharpen -------------------------------------------------------------------------

/// Blur / Sharpen: the target of a tile is the planar-filtered stroke base.
pub struct FocusKernel {
    base: TileGrid,
    sigma: f32,
    sharpen: Option<f32>,
}

impl FocusKernel {
    /// `strength` 0..1. Blur maps it to a Gaussian σ of 0.6–4 px; Sharpen to
    /// an unsharp amount at σ = 1.
    pub fn new(base: &TileGrid, strength: f32, sharpen: bool) -> Self {
        let s = strength.clamp(0.0, 1.0);
        if sharpen {
            FocusKernel { base: base.clone(), sigma: 1.0, sharpen: Some(0.4 + 2.2 * s) }
        } else {
            FocusKernel { base: base.clone(), sigma: 0.6 + 3.4 * s, sharpen: None }
        }
    }
}

impl Kernel for FocusKernel {
    fn prepare(&self, tx: i32, ty: i32, base: Option<&Tile>) -> Option<Box<[u32]>> {
        // A flat tile in a flat neighbourhood stays flat.
        if base.is_some_and(|t| t.uniform.is_some()) && self.base.region_uniform(TileGrid::tile_rect(tx, ty).inflate(16)).is_some() {
            return Some(base.unwrap().read().to_vec().into_boxed_slice());
        }
        let pad = (self.sigma * 3.0).ceil() as usize + 2;
        let n = TS + 2 * pad;
        let mut buf = vec![0u32; n * n];
        self.base.read_region_clamped(tx * TS_I - pad as i32, ty * TS_I - pad as i32, n, n, &mut buf);
        let mode = match self.sharpen {
            Some(k) => super::filters::fast::Combine::Unsharp { k: (k * 256.0) as i32, threshold: 0 },
            None => super::filters::fast::Combine::Blur,
        };
        super::filters::fast::blur_combine(&mut buf, n, n, self.sigma, None, mode);
        let mut t = vec![0u32; TS * TS].into_boxed_slice();
        for y in 0..TS {
            t[y * TS..(y + 1) * TS].copy_from_slice(&buf[(y + pad) * n + pad..(y + pad) * n + pad + TS]);
        }
        Some(t)
    }
    fn apply_row(&self, base: &[u32], target: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
        lerp_rows(base, target, alpha, out);
    }
}

// ----- smudge / mixer ---------------------------------------------------------------------------

/// Reads the current pixels of a rect (outside the image = transparent).
fn read(grid: &TileGrid, r: RectI) -> Vec<u32> {
    let mut v = vec![0u32; (r.w() * r.h()) as usize];
    grid.read_region(r.x0, r.y0, r.w() as usize, r.h() as usize, &mut v);
    v
}

/// Writes `px` back into the rect, touching only pixels with non-zero `a`.
fn write(grid: &mut TileGrid, r: RectI, px_: &[u32], a: &[u8]) {
    let img = grid.bounds();
    let rr = r.intersect(&img);
    let w = r.w() as usize;
    let (tx0, ty0, tx1, ty1) = rr.tile_span(grid.cols, grid.rows);
    for ty in ty0..ty1 {
        for tx in tx0..tx1 {
            let tr = TileGrid::tile_rect(tx, ty).intersect(&rr);
            let any = (tr.y0..tr.y1).any(|y| (tr.x0..tr.x1).any(|x| a[((y - r.y0) as usize) * w + (x - r.x0) as usize] > 0));
            if !any {
                continue;
            }
            let i = grid.idx(tx, ty);
            let t = tile_mut(&mut grid.tiles[i]);
            let mut pin = t.write();
            let pixels: &mut [u32] = &mut pin;
            for y in tr.y0..tr.y1 {
                for x in tr.x0..tr.x1 {
                    let k = ((y - r.y0) as usize) * w + (x - r.x0) as usize;
                    if a[k] > 0 {
                        pixels[((y - ty * TS_I) as usize) * TS + (x - tx * TS_I) as usize] = px_[k];
                    }
                }
            }
        }
    }
}

fn sel_cov(sel: Option<&Selection>, x: i32, y: i32) -> u32 {
    sel.map_or(255, |s| s.at(x, y) as u32)
}

/// Smudge: drags a pickup buffer of colour along the stroke.
pub struct SmudgeKernel {
    pub strength: f32,
    /// Finger Painting: start from the foreground colour.
    pub finger: Option<u32>,
    pickup: Vec<u32>,
    side: usize,
}

impl SmudgeKernel {
    pub fn new(strength: f32, finger: Option<u32>) -> Self {
        SmudgeKernel { strength, finger, pickup: vec![], side: 0 }
    }
}

impl Kernel for SmudgeKernel {
    fn apply_row(&self, base: &[u32], _t: Option<&[u32]>, _a: &[u8], out: &mut [u32]) {
        out.copy_from_slice(base);
    }
    fn sequential(&self) -> bool {
        true
    }
    fn seq_dab(&mut self, d: &Dab, shape: &DabShape, grid: &mut TileGrid, sel: Option<&Selection>) -> RectI {
        let r = d.rect();
        let side = r.w().max(r.h()) as usize;
        let cur = read(grid, r);
        let (w, h) = (r.w() as usize, r.h() as usize);
        if self.side != side || self.pickup.len() != w * h {
            self.pickup = match self.finger {
                Some(c) => vec![c; w * h],
                None => cur.clone(),
            };
            self.side = side;
            return r;
        }
        let k = self.strength.clamp(0.0, 1.0);
        let mut out = cur.clone();
        let mut a = vec![0u8; w * h];
        let carry = ((1.0 - k) * 255.0) as u32;
        let ks = (k * 255.0) as u32;
        // Rows are independent: each pixel only touches its own pickup cell.
        out.par_chunks_mut(w)
            .zip(a.par_chunks_mut(w))
            .zip(self.pickup.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, ((orow, arow), prow))| {
                let gy = r.y0 + y as i32;
                for x in 0..w {
                    let gx = r.x0 + x as i32;
                    let f = shape.at(gx, gy);
                    if f == 0 {
                        continue;
                    }
                    let al = ((f >> 8) * ks / 255 * sel_cov(sel, gx, gy) / 255).min(255);
                    if al == 0 {
                        continue;
                    }
                    let c = cur[y * w + x];
                    orow[x] = px::lerp(c, prow[x], al);
                    arow[x] = al as u8;
                    // The pickup takes on some of the canvas it passes over.
                    prow[x] = px::lerp(prow[x], orow[x], carry);
                }
            });
        write(grid, r, &out, &a);
        r
    }
}

/// Mixer Brush: a reservoir of paint that picks up canvas colour.
pub struct MixerKernel {
    /// Loaded colour (premultiplied), or None for a clean brush.
    pub load_color: Option<u32>,
    /// How much canvas paint the reservoir picks up (0..1).
    pub wet: f32,
    /// How much paint is loaded; depletes along the stroke (0..1).
    pub load: f32,
    /// Ratio of canvas paint to reservoir paint in the mix (0..1).
    pub mix: f32,
    pub flow: f32,
    reservoir: Vec<u32>,
    level: f32,
}

impl MixerKernel {
    pub fn new(load_color: Option<u32>, wet: f32, load: f32, mix: f32, flow: f32) -> Self {
        MixerKernel { load_color, wet, load, mix, flow, reservoir: vec![], level: 1.0 }
    }
}

impl Kernel for MixerKernel {
    fn apply_row(&self, base: &[u32], _t: Option<&[u32]>, _a: &[u8], out: &mut [u32]) {
        out.copy_from_slice(base);
    }
    fn sequential(&self) -> bool {
        true
    }
    fn seq_dab(&mut self, d: &Dab, shape: &DabShape, grid: &mut TileGrid, sel: Option<&Selection>) -> RectI {
        let r = d.rect();
        let (w, h) = (r.w() as usize, r.h() as usize);
        let cur = read(grid, r);
        if self.reservoir.len() != w * h {
            self.reservoir = match self.load_color {
                Some(c) => vec![c; w * h],
                None => cur.clone(),
            };
        }
        // The loaded paint runs out over ~ (load × 200) dabs.
        let deplete = 1.0 - 1.0 / (self.load.max(0.01) * 200.0);
        let mut out = cur.clone();
        let mut a = vec![0u8; w * h];
        let mix = (self.mix.clamp(0.0, 1.0) * 255.0) as u32;
        let wet = (self.wet.clamp(0.0, 1.0) * 128.0) as u32;
        let level = self.flow * self.level.max(if self.load_color.is_some() { 0.0 } else { 1.0 });
        out.par_chunks_mut(w)
            .zip(a.par_chunks_mut(w))
            .zip(self.reservoir.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, ((orow, arow), rrow))| {
                let gy = r.y0 + y as i32;
                for x in 0..w {
                    let gx = r.x0 + x as i32;
                    let f = shape.at(gx, gy);
                    if f == 0 {
                        continue;
                    }
                    let c = cur[y * w + x];
                    let paint = px::lerp(rrow[x], c, mix);
                    let dep = (((f >> 8) as f32 * level) as u32 * sel_cov(sel, gx, gy) / 255).min(255);
                    if dep > 0 {
                        orow[x] = px::lerp(c, paint, dep);
                        arow[x] = dep as u8;
                    }
                    // The reservoir is contaminated by the wet canvas.
                    rrow[x] = px::lerp(rrow[x], c, wet);
                }
            });
        self.level *= deplete;
        write(grid, r, &out, &a);
        r
    }
}

// ----- colour replacement / background eraser ------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Sampling {
    #[default]
    Continuous,
    Once,
    BackgroundSwatch,
}

impl Sampling {
    pub const ALL: [Sampling; 3] = [Sampling::Continuous, Sampling::Once, Sampling::BackgroundSwatch];
    pub fn label(self) -> &'static str {
        match self {
            Sampling::Continuous => "Continuous",
            Sampling::Once => "Once",
            Sampling::BackgroundSwatch => "Background Swatch",
        }
    }
}

/// Colour distance → weight 0..255 with a soft edge over the last 25%.
#[inline]
fn match_weight(p: u32, sample: [u8; 3], tol: f32) -> u32 {
    if px::alpha(p) == 0 {
        return 0;
    }
    let c = px::unpremultiply(p);
    let d = (0..3).map(|k| (c[k] as i32 - sample[k] as i32).unsigned_abs()).max().unwrap_or(0) as f32;
    let t = tol.max(0.5);
    if d <= t * 0.75 {
        255
    } else if d >= t {
        0
    } else {
        ((t - d) / (t * 0.25) * 255.0) as u32
    }
}

/// Which similar pixels a dab may change (Color Replacement and Background
/// Eraser "Limits").
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Limits {
    /// Every matching pixel under the tip.
    #[default]
    Discontiguous,
    /// Matching pixels connected to the hotspot (the dab centre).
    Contiguous,
    /// Contiguous, and the fill also stops at colour edges inside the
    /// tolerance, keeping sharp boundaries sharp.
    FindEdges,
}

impl Limits {
    pub const ALL: [Limits; 3] = [Limits::Discontiguous, Limits::Contiguous, Limits::FindEdges];
    pub fn label(self) -> &'static str {
        match self {
            Limits::Discontiguous => "Discontiguous",
            Limits::Contiguous => "Contiguous",
            Limits::FindEdges => "Find Edges",
        }
    }
}

/// The per-dab weight mask for `limits`: colour-match weights inside the
/// dab's circle, flooded from the hotspot (4-connected, scanline). With
/// `FindEdges` the flood also stops at pixels whose local gradient exceeds
/// a third of the tolerance; edge pixels touching the fill keep their own
/// match weight so the boundary stays anti-aliased. `None` for
/// Discontiguous (the plain per-pixel weight applies).
pub fn limits_mask(d: &Dab, base: &BaseView, sample: [u8; 3], tol: f32, limits: Limits) -> Option<DabMask> {
    if limits == Limits::Discontiguous {
        return None;
    }
    let rect = d.rect().intersect(&RectI::new(0, 0, base.w as i32, base.h as i32));
    let (w, h) = (rect.w().max(0) as usize, rect.h().max(0) as usize);
    let mut out = vec![0u8; w * h].into_boxed_slice();
    if w == 0 || h == 0 {
        return Some(DabMask { rect, w: out });
    }
    let mut pxs = vec![0u32; w * h];
    base.read_region(rect.x0, rect.y0, w, h, &mut pxs);
    // Match weights straight from the packed pixels (opaque pixels need no
    // unpremultiply), zero outside the dab's circle.
    let r2 = (d.r + 1.0) * (d.r + 1.0);
    let mut wt = vec![0u8; w * h];
    wt.par_chunks_exact_mut(w).with_min_len(32).enumerate().for_each(|(y, row)| {
        let dy = rect.y0 as f32 + y as f32 + 0.5 - d.y;
        let dy2 = dy * dy;
        if dy2 >= r2 {
            return;
        }
        let half = (r2 - dy2).sqrt();
        let xa = ((d.x - half - rect.x0 as f32).floor().max(0.0) as usize).min(w);
        let xb = ((d.x + half - rect.x0 as f32).ceil().max(0.0) as usize).min(w);
        for x in xa..xb {
            row[x] = match_weight(pxs[y * w + x], sample, tol) as u8;
        }
    });
    // Edge pixels (Find Edges): central differences of the packed colour,
    // max over channels (premultiplied equals straight for opaque pixels).
    let edge: Option<Vec<bool>> = (limits == Limits::FindEdges).then(|| {
        let thr = 2 * (tol / 3.0).max(8.0) as i32;
        let mut e = vec![false; w * h];
        e.par_chunks_exact_mut(w).with_min_len(32).enumerate().for_each(|(y, erow)| {
            let (ya, yb) = (y.saturating_sub(1) * w, (y + 1).min(h - 1) * w);
            for x in 0..w {
                let i = y * w + x;
                if wt[i] == 0 {
                    continue;
                }
                let (l, r) = (px::unpack(pxs[y * w + x.saturating_sub(1)]), px::unpack(pxs[y * w + (x + 1).min(w - 1)]));
                let (tp, b) = (px::unpack(pxs[ya + x]), px::unpack(pxs[yb + x]));
                let g = (0..3).map(|k| (l[k] as i32 - r[k] as i32).abs() + (tp[k] as i32 - b[k] as i32).abs()).max().unwrap_or(0);
                erow[x] = g > thr;
            }
        });
        e
    });
    let pass = |i: usize| wt[i] > 0 && edge.as_ref().is_none_or(|e| !e[i]);
    let (sx, sy) = (((d.x - rect.x0 as f32) as usize).min(w - 1), ((d.y - rect.y0 as f32) as usize).min(h - 1));
    if !pass(sy * w + sx) {
        return Some(DabMask { rect, w: out });
    }
    // Scanline flood.
    let mut stack = vec![(sx, sy)];
    while let Some((x, y)) = stack.pop() {
        let row = y * w;
        if out[row + x] != 0 || !pass(row + x) {
            continue;
        }
        let (mut a, mut b) = (x, x);
        while a > 0 && out[row + a - 1] == 0 && pass(row + a - 1) {
            a -= 1;
        }
        while b + 1 < w && out[row + b + 1] == 0 && pass(row + b + 1) {
            b += 1;
        }
        for k in a..=b {
            out[row + k] = wt[row + k];
        }
        for ny in [y.wrapping_sub(1), y + 1] {
            if ny >= h {
                continue;
            }
            let nrow = ny * w;
            let mut k = a;
            while k <= b {
                if out[nrow + k] == 0 && pass(nrow + k) {
                    stack.push((k, ny));
                    while k <= b && pass(nrow + k) {
                        k += 1;
                    }
                }
                k += 1;
            }
        }
    }
    // Edge pixels bordering the fill keep their match weight.
    if let Some(e) = &edge {
        let filled = out.clone();
        out.par_chunks_exact_mut(w).with_min_len(32).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let i = y * w + x;
                if !e[i] || wt[i] == 0 {
                    continue;
                }
                let near = (x > 0 && filled[i - 1] > 0) || (x + 1 < w && filled[i + 1] > 0) || (y > 0 && filled[i - w] > 0) || (y + 1 < h && filled[i + w] > 0);
                if near {
                    *o = wt[i];
                }
            }
        });
    }
    Some(DabMask { rect, w: out })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ReplaceMode {
    Hue,
    Saturation,
    #[default]
    Color,
    Luminosity,
}

impl ReplaceMode {
    pub const ALL: [ReplaceMode; 4] = [ReplaceMode::Hue, ReplaceMode::Saturation, ReplaceMode::Color, ReplaceMode::Luminosity];
    pub fn label(self) -> &'static str {
        match self {
            ReplaceMode::Hue => "Hue",
            ReplaceMode::Saturation => "Saturation",
            ReplaceMode::Color => "Color",
            ReplaceMode::Luminosity => "Luminosity",
        }
    }
}

/// Color Replacement: recolours pixels similar to the sampled colour.
pub struct ReplaceKernel {
    pub fg: [u8; 3],
    pub mode: ReplaceMode,
    pub sampling: Sampling,
    pub tolerance: f32,
    pub limits: Limits,
    sample: Option<[u8; 3]>,
}

impl ReplaceKernel {
    pub fn new(fg: [u8; 3], bg: [u8; 3], mode: ReplaceMode, sampling: Sampling, tolerance: f32) -> Self {
        ReplaceKernel { fg, mode, sampling, tolerance, limits: Limits::Discontiguous, sample: if sampling == Sampling::BackgroundSwatch { Some(bg) } else { None } }
    }

    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    fn recolor(&self, p: u32) -> u32 {
        let [r, g, b, a] = px::unpremultiply(p);
        if a == 0 {
            return p;
        }
        let (h, s, l) = rgb_to_hsl([r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]);
        let (fh, fs, fl) = rgb_to_hsl(self.fg.map(|v| v as f32 / 255.0));
        let (nh, ns, nl) = match self.mode {
            ReplaceMode::Hue => (fh, s, l),
            ReplaceMode::Saturation => (h, fs, l),
            ReplaceMode::Color => (fh, fs, l),
            ReplaceMode::Luminosity => (h, s, fl),
        };
        let c = hsl_to_rgb(nh, ns, nl);
        px::premultiply((c[0] * 255.0 + 0.5) as u8, (c[1] * 255.0 + 0.5) as u8, (c[2] * 255.0 + 0.5) as u8, a)
    }
}

impl Kernel for ReplaceKernel {
    fn prepare(&self, _tx: i32, _ty: i32, base: Option<&Tile>) -> Option<Box<[u32]>> {
        let base = base?;
        Some(base.read().iter().map(|&p| self.recolor(p)).collect())
    }
    fn apply_row(&self, base: &[u32], target: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
        lerp_rows(base, target, alpha, out);
    }
    fn has_weight(&self) -> bool {
        true
    }
    fn weight(&self, base_px: u32) -> u32 {
        match self.sample {
            Some(s) => match_weight(base_px, s, self.tolerance),
            None => 0,
        }
    }
    fn on_dab(&mut self, d: &Dab, base: &BaseView) {
        let take = match self.sampling {
            Sampling::Continuous => true,
            Sampling::Once => self.sample.is_none(),
            Sampling::BackgroundSwatch => false,
        };
        if take {
            let p = base.pixel(d.x as i32, d.y as i32);
            if px::alpha(p) > 0 {
                let c = px::unpremultiply(p);
                self.sample = Some([c[0], c[1], c[2]]);
            }
        }
    }
    fn dab_mask(&mut self, d: &Dab, base: &BaseView) -> Option<DabMask> {
        limits_mask(d, base, self.sample?, self.tolerance, self.limits)
    }
}

/// Background Eraser: erases pixels similar to the sampled colour, keeping
/// the foreground colour when protected.
pub struct BgEraseKernel {
    pub sampling: Sampling,
    pub tolerance: f32,
    pub protect: Option<[u8; 3]>,
    pub limits: Limits,
    sample: Option<[u8; 3]>,
}

impl BgEraseKernel {
    pub fn new(sampling: Sampling, tolerance: f32, bg: [u8; 3], protect: Option<[u8; 3]>) -> Self {
        BgEraseKernel { sampling, tolerance, protect, limits: Limits::Discontiguous, sample: if sampling == Sampling::BackgroundSwatch { Some(bg) } else { None } }
    }

    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }
}

impl Kernel for BgEraseKernel {
    fn apply_row(&self, base: &[u32], target: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
        Erase.apply_row(base, target, alpha, out);
    }
    fn dab_mask(&mut self, d: &Dab, base: &BaseView) -> Option<DabMask> {
        let mut m = limits_mask(d, base, self.sample?, self.tolerance, self.limits)?;
        if let Some(fg) = self.protect {
            let (w, h) = (m.rect.w() as usize, m.rect.h() as usize);
            let mut pxs = vec![0u32; w * h];
            base.read_region(m.rect.x0, m.rect.y0, w, h, &mut pxs);
            for (v, &p) in m.w.iter_mut().zip(&pxs) {
                if *v > 0 && match_weight(p, fg, self.tolerance * 0.5) > 128 {
                    *v = 0;
                }
            }
        }
        Some(m)
    }
    fn has_weight(&self) -> bool {
        true
    }
    fn weight(&self, base_px: u32) -> u32 {
        let Some(s) = self.sample else { return 0 };
        let w = match_weight(base_px, s, self.tolerance);
        match self.protect {
            Some(fg) if w > 0 && match_weight(base_px, fg, self.tolerance * 0.5) > 128 => 0,
            _ => w,
        }
    }
    fn on_dab(&mut self, d: &Dab, base: &BaseView) {
        let take = match self.sampling {
            Sampling::Continuous => true,
            Sampling::Once => self.sample.is_none(),
            Sampling::BackgroundSwatch => false,
        };
        if take {
            let p = base.pixel(d.x as i32, d.y as i32);
            if px::alpha(p) > 0 {
                let c = px::unpremultiply(p);
                self.sample = Some([c[0], c[1], c[2]]);
            }
        }
    }
}

// ----- magic eraser / pattern fill -----------------------------------------------------------

/// Erases `mask` (soft) from `grid` at `opacity`, sharing untouched tiles
/// (like the bucket): fully covered tiles become empty without pixel work.
pub fn erase_mask(grid: &TileGrid, mask: &MaskGrid, sel: Option<&Selection>, opacity: f32) -> TileGrid {
    use rayon::prelude::*;
    let mut out = grid.clone();
    let a = (opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let cols = grid.cols;
    out.tiles.par_iter_mut().enumerate().for_each(|(i, slot)| {
        if slot.is_none() {
            return;
        }
        let (tx, ty) = ((i as u32 % cols) as i32, (i as u32 / cols) as i32);
        let cov = sel.map_or(TileCoverage::Full, |s| s.tile_coverage(tx, ty));
        if matches!(cov, TileCoverage::Empty) {
            return;
        }
        match &mask.tiles[i] {
            MaskTile::Empty => {}
            MaskTile::Full if a == 255 && matches!(cov, TileCoverage::Full) => *slot = None,
            mt => {
                let t = tile_mut(slot);
                for (k, p) in t.write().iter_mut().enumerate() {
                    let m = match mt {
                        MaskTile::Full => 255,
                        MaskTile::Partial(m) => m[k] as u32,
                        MaskTile::Empty => 0,
                    };
                    let e = m * cov.at(k) / 255 * a / 255;
                    if e > 0 {
                        *p = px::scale(*p, 255 - e);
                    }
                }
            }
        }
    });
    out
}

/// Paint bucket with a pattern through a coverage mask.
pub fn fill_mask_pattern(grid: &TileGrid, mask: &MaskGrid, sel: Option<&Selection>, pattern: usize, opacity: f32) -> TileGrid {
    use rayon::prelude::*;
    let pat: &super::retouch::Pattern = &super::retouch::patterns()[pattern % super::retouch::patterns().len()];
    let mut out = grid.clone();
    let a = (opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let cols = grid.cols;
    out.tiles.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let (tx, ty) = ((i as u32 % cols) as i32, (i as u32 / cols) as i32);
        let mt = &mask.tiles[i];
        if matches!(mt, MaskTile::Empty) {
            return;
        }
        let cov = sel.map_or(TileCoverage::Full, |s| s.tile_coverage(tx, ty));
        if matches!(cov, TileCoverage::Empty) {
            return;
        }
        let t = tile_mut(slot);
        let mut pin = t.write();
        let pixels: &mut [u32] = &mut pin;
        for y in 0..TS {
            let py = (ty as usize * TS + y) % pat.h;
            for x in 0..TS {
                let k = y * TS + x;
                let m = match mt {
                    MaskTile::Full => 255,
                    MaskTile::Partial(m) => m[k] as u32,
                    MaskTile::Empty => 0,
                };
                let e = m * cov.at(k) / 255 * a / 255;
                if e > 0 {
                    let c = pat.px[py * pat.w + (tx as usize * TS + x) % pat.w];
                    pixels[k] = px::lerp(pixels[k], c, e);
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::brush::{BrushSettings, Stroke};

    fn gray(v: u8) -> u32 {
        px::pack(v, v, v, 255)
    }

    #[test]
    fn color_replacement_limits() {
        // Two red blobs separated by a white gap, both under one dab.
        let mut img = TileGrid::filled(300, 300, gray(255));
        let red = px::pack(200, 20, 20, 255);
        for y in 100..200 {
            for x in (40..100).chain(160..220) {
                super::super::grid::tile_mut(&mut img.tiles[0]).write()[y * TS + x] = red;
            }
        }
        let run = |limits: Limits, img: &TileGrid, at: (f32, f32), tol: f32| -> TileGrid {
            let mut g = img.clone();
            let k = ReplaceKernel::new([0, 0, 255], [255; 3], ReplaceMode::Color, Sampling::Once, tol).with_limits(limits);
            let mut st = Stroke::new(0, &g, BrushSettings { size: 260.0, hardness: 1.0, ..Default::default() }, Box::new(k), None);
            st.add_point(at.0, at.1, &mut g);
            g
        };
        let changed = |g: &TileGrid, x: i32, y: i32| {
            let c = px::unpack(g.pixel(x, y));
            c[2] as i32 > c[0] as i32 + 50
        };
        let d = run(Limits::Discontiguous, &img, (70.0, 150.0), 40.0);
        assert!(changed(&d, 70, 150) && changed(&d, 190, 150));
        let c = run(Limits::Contiguous, &img, (70.0, 150.0), 40.0);
        assert!(changed(&c, 70, 150) && !changed(&c, 190, 150), "contiguous leaked across the gap");
        assert!(!changed(&c, 130, 150), "white gap was recoloured");
        // A step edge inside the tolerance: Contiguous crosses it, Find Edges stops.
        let mut step = TileGrid::filled(300, 300, gray(100));
        for y in 0..TS {
            for x in 150..TS {
                super::super::grid::tile_mut(&mut step.tiles[0]).write()[y * TS + x] = gray(128);
            }
        }
        let c = run(Limits::Contiguous, &step, (100.0, 150.0), 40.0);
        let e = run(Limits::FindEdges, &step, (100.0, 150.0), 40.0);
        let tinted = |g: &TileGrid, x: i32| {
            let c = px::unpack(g.pixel(x, 150));
            c[2] as i32 - c[0] as i32 > 10
        };
        assert!(tinted(&c, 100) && tinted(&c, 200), "contiguous should cross a within-tolerance step");
        assert!(tinted(&e, 100) && !tinted(&e, 200), "find edges crossed the edge");
    }

    #[test]
    fn dodge_burn_ranges() {
        let base = TileGrid::filled(256, 256, gray(128));
        for (mode, want_up) in [(Tone::Dodge, true), (Tone::Burn, false)] {
            let mut g = base.clone();
            let mut st = Stroke::new(0, &g, BrushSettings { size: 80.0, hardness: 1.0, ..Default::default() }, Box::new(ToneKernel::new(mode, Range::Midtones, 0.5, true, false)), None);
            st.add_point(128.0, 128.0, &mut g);
            let v = px::unpack(g.pixel(128, 128))[0];
            assert!(if want_up { v > 150 } else { v < 105 }, "{mode:?} gave {v}");
            assert_eq!(px::unpack(g.pixel(10, 10))[0], 128);
        }
        // Shadows range barely touches highlights.
        let mut g = TileGrid::filled(256, 256, gray(240));
        let mut st = Stroke::new(0, &g, BrushSettings { size: 80.0, hardness: 1.0, ..Default::default() }, Box::new(ToneKernel::new(Tone::Dodge, Range::Shadows, 1.0, false, false)), None);
        st.add_point(128.0, 128.0, &mut g);
        assert!(px::unpack(g.pixel(128, 128))[0] <= 242);
        // Sponge desaturates.
        let mut g = TileGrid::filled(256, 256, px::pack(200, 50, 50, 255));
        let mut st = Stroke::new(0, &g, BrushSettings { size: 80.0, hardness: 1.0, ..Default::default() }, Box::new(ToneKernel::new(Tone::Desaturate, Range::Midtones, 1.0, false, false)), None);
        st.add_point(128.0, 128.0, &mut g);
        let c = px::unpack(g.pixel(128, 128));
        assert!((c[0] as i32 - c[1] as i32).abs() < 10, "{c:?}");
    }

    #[test]
    fn blur_and_sharpen_change_edges_only_inside_dab() {
        let mut g = TileGrid::new(512, 512);
        for y in 0..512 {
            for x in 0..512 {
                let i = g.idx(x >> 8, y >> 8);
                tile_mut(&mut g.tiles[i]).write()[((y & 255) as usize) * TS + (x & 255) as usize] = if x < 256 { gray(0) } else { gray(255) };
            }
        }
        let orig = g.clone();
        let mut st = Stroke::new(0, &g, BrushSettings { size: 60.0, hardness: 1.0, ..Default::default() }, Box::new(FocusKernel::new(&orig, 1.0, false)), None);
        st.add_point(256.0, 100.0, &mut g);
        let v = px::unpack(g.pixel(255, 100))[0];
        assert!(v > 40 && v < 215, "blur didn't soften the edge: {v}");
        assert_eq!(g.pixel(255, 400), orig.pixel(255, 400));
        let mut g2 = orig.clone();
        let mut st = Stroke::new(0, &g2, BrushSettings { size: 60.0, hardness: 1.0, ..Default::default() }, Box::new(FocusKernel::new(&orig, 1.0, true)), None);
        st.add_point(256.0, 300.0, &mut g2);
        assert_eq!(g2.pixel(10, 300), orig.pixel(10, 300));
    }

    #[test]
    fn smudge_drags_color() {
        let mut g = TileGrid::filled(512, 256, gray(255));
        for y in 100..156 {
            for x in 100..140 {
                let i = g.idx(x >> 8, y >> 8);
                tile_mut(&mut g.tiles[i]).write()[((y & 255) as usize) * TS + (x & 255) as usize] = px::pack(255, 0, 0, 255);
            }
        }
        let mut st = Stroke::new(0, &g, BrushSettings { size: 40.0, hardness: 0.8, spacing: 0.1, ..Default::default() }, Box::new(SmudgeKernel::new(0.8, None)), None);
        for k in 0..20 {
            st.add_point(120.0 + k as f32 * 6.0, 128.0, &mut g);
        }
        let c = px::unpack(g.pixel(180, 128));
        assert!(c[1] < 230, "no red dragged to x=180: {c:?}");
    }

    #[test]
    fn background_eraser_keeps_other_colors() {
        let mut g = TileGrid::filled(256, 256, px::pack(30, 200, 30, 255));
        for y in 0..256 {
            for x in 128..256 {
                let i = g.idx(0, 0);
                tile_mut(&mut g.tiles[i]).write()[y * TS + x] = px::pack(200, 30, 30, 255);
            }
        }
        let mut st = Stroke::new(0, &g, BrushSettings { size: 100.0, hardness: 1.0, ..Default::default() }, Box::new(BgEraseKernel::new(Sampling::Once, 40.0, [0; 3], None)), None);
        st.add_point(110.0, 128.0, &mut g);
        assert_eq!(px::alpha(g.pixel(100, 128)), 0);
        assert_eq!(px::alpha(g.pixel(140, 128)), 255);
        // Color replacement recolours only the sampled colour.
        let mut g2 = TileGrid::filled(256, 256, px::pack(30, 200, 30, 255));
        let mut st = Stroke::new(0, &g2, BrushSettings { size: 100.0, hardness: 1.0, ..Default::default() }, Box::new(ReplaceKernel::new([0, 0, 255], [255; 3], ReplaceMode::Color, Sampling::Once, 40.0)), None);
        st.add_point(110.0, 128.0, &mut g2);
        let c = px::unpack(g2.pixel(110, 128));
        assert!(c[2] > c[1], "not recoloured: {c:?}");
    }
}

#[cfg(test)]
mod limits_perf {
    use super::*;
    #[test]
    #[ignore]
    fn limits_mask_timing() {
        let g = crate::engine::ops::demo_scene(4096, 4096);
        let base = BaseView { tiles: &g.tiles, cols: g.cols, w: g.w, h: g.h };
        for limits in [Limits::Contiguous, Limits::FindEdges] {
            let mut tot = 0.0;
            for k in 0..40 {
                let d = Dab::new(800.0 + k as f32 * 60.0, 2400.0, 250.0, 65535, 255);
                let c = px::unpremultiply(base.pixel(d.x as i32, d.y as i32));
                let t = std::time::Instant::now();
                let m = limits_mask(&d, &base, [c[0], c[1], c[2]], 40.0, limits);
                tot += t.elapsed().as_secs_f64() * 1e3;
                assert!(m.is_some());
            }
            eprintln!("{limits:?}: {:.3} ms per dab mask", tot / 40.0);
        }
    }
}

