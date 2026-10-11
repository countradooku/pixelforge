//! Remaining Photoshop filters: Render ▸ Lighting Effects, Distort ▸ Displace,
//! Blur ▸ Shape Blur, Blur Gallery (Tilt-Shift, Iris), Stylize ▸ Extrude /
//! Tiles, Other ▸ Custom / De-Interlace / NTSC Colors, Camera Raw, and the
//! color adjustments Channel Mixer, Selective Color, Replace Color and
//! Match Color.

use std::sync::Arc;

use super::util::{Lut3d, hash01, hsl_to_rgb, luma, merge, opaque, rgb_to_hsl, split};
use super::{Filter, Stats};
use crate::engine::px::{self, pack, unpack};

#[inline]
fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Luminance (0..1) of straight 8-bit color (Camera Raw, Lighting Effects).
#[inline]
pub(crate) fn lum_of(r: u8, g: u8, b: u8) -> f32 {
    (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) / 255.0
}

/// The 8-bit quantization `blur_f` blurs in.
#[inline]
pub(crate) fn q8(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn lum_plane(buf: &[u32]) -> Vec<f32> {
    buf.iter()
        .map(|&p| {
            let [r, g, b, _] = px::unpremultiply(p);
            lum_of(r, g, b)
        })
        .collect()
}

fn blur_f(v: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let mut p: Vec<u8> = v.iter().map(|&x| q8(x)).collect();
    super::fast::gauss_plane(&mut p, w, h, sigma);
    p.iter().map(|&x| x as f32 / 255.0).collect()
}

// ----- Lighting Effects -----------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// 0 = point (omni), 1 = spot, 2 = infinite (directional).
    pub kind: u8,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    /// Spot/infinite direction (degrees) and spot cone half-angle.
    pub angle: f32,
    pub cone: f32,
    pub intensity: f32,
    pub ambient: f32,
    pub bump: f32,
    pub color: [u8; 3],
}

/// Lighting Effects: point, spot or infinite light with luminance bump map.
pub fn lighting(buf: &mut [u32], w: usize, h: usize, level: u32, origin: (i32, i32), lt: &Light) {
    let s = (1u32 << level) as f32;
    let l = lum_plane(buf);
    let hgt = blur_f(&l, w, h, 1.0);
    let at = |x: isize, y: isize| hgt[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    let lc = [lt.color[0] as f32 / 255.0, lt.color[1] as f32 / 255.0, lt.color[2] as f32 / 255.0];
    let k = lt.intensity / 50.0;
    let amb = lt.ambient / 100.0;
    let bz = lt.bump / 10.0;
    let dir = lt.angle.to_radians();
    let (ddx, ddy) = (dir.cos(), -dir.sin());
    let cone = lt.cone.to_radians().cos();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let (gx, gy) = ((origin.0 as f32 + x as f32 + 0.5) * s, (origin.1 as f32 + y as f32 + 0.5) * s);
            let nx = -(at(x as isize + 1, y as isize) - at(x as isize - 1, y as isize)) * bz * 4.0;
            let ny = -(at(x as isize, y as isize + 1) - at(x as isize, y as isize - 1)) * bz * 4.0;
            let nl = (nx * nx + ny * ny + 1.0).sqrt();
            let e = match lt.kind {
                2 => {
                    // Infinite: constant direction, elevation 45°.
                    let (lx, ly, lz) = (-ddx * 0.7, -ddy * 0.7, 0.7);
                    amb + k * ((nx * lx + ny * ly + lz) / nl).max(0.0)
                }
                _ => {
                    let (dx, dy) = (lt.x - gx, lt.y - gy);
                    let dist = (dx * dx + dy * dy).sqrt();
                    let fall = (1.0 - (dist / lt.radius).powi(2)).max(0.0);
                    let z = lt.radius * 0.5;
                    let ld = (dx * dx + dy * dy + z * z).sqrt();
                    let lambert = ((nx * dx + ny * dy + z) / (nl * ld)).max(0.0);
                    let spot = if lt.kind == 1 {
                        let c = if dist > 1e-3 { (-dx * ddx - dy * ddy) / dist } else { 1.0 };
                        smooth(cone - 0.05, cone + 0.05, c)
                    } else {
                        1.0
                    };
                    amb + k * fall * lambert * spot
                }
            };
            let [r, g, b, a] = px::unpremultiply(buf[i]);
            let ch = |v: u8, c: f32| ((v as f32 / 255.0) * (amb + (e - amb) * c)).clamp(0.0, 1.0);
            let o = [ch(r, lc[0]), ch(g, lc[1]), ch(b, lc[2])];
            buf[i] = px::premultiply((o[0] * 255.0) as u8, (o[1] * 255.0) as u8, (o[2] * 255.0) as u8, a);
        }
    }
}

// ----- Displace -------------------------------------------------------------------------

/// A displacement map loaded from an image file (red = horizontal, green = vertical).
#[derive(Debug, PartialEq)]
pub struct DispMap {
    pub w: u32,
    pub h: u32,
    pub rg: Vec<[u8; 2]>,
}

impl DispMap {
    pub fn load(path: &std::path::Path) -> anyhow::Result<DispMap> {
        let mut r = image::ImageReader::open(path)?.with_guessed_format()?;
        r.no_limits();
        let img = r.decode()?;
        // Cap the map at 4096 px on its long side; it is sampled smoothly anyway.
        let img = if img.width().max(img.height()) > 4096 { img.thumbnail(4096, 4096) } else { img };
        let rgb = img.into_rgb8();
        let (w, h) = rgb.dimensions();
        Ok(DispMap { w, h, rg: rgb.pixels().map(|p| [p[0], p[1]]).collect() })
    }

    #[inline]
    fn at(&self, u: f32, v: f32) -> (f32, f32) {
        let x = (u as i64).clamp(0, self.w as i64 - 1) as usize;
        let y = (v as i64).clamp(0, self.h as i64 - 1) as usize;
        let p = self.rg[y * self.w as usize + x];
        (p[0] as f32 / 255.0, p[1] as f32 / 255.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MapSource {
    Layer,
    Clouds,
    /// File map, stretched over the image (`tile = false`) or tiled at 1:1.
    File { map: Arc<DispMap>, tile: bool, img_w: f32, img_h: f32 },
}

/// Displace pixels by a map: 128 = no shift, 0/255 = full scale left/right.
#[allow(clippy::too_many_arguments)]
pub fn displace(buf: &mut [u32], w: usize, h: usize, level: u32, origin: (i32, i32), hs: f32, vs: f32, src_map: &MapSource) {
    let s = (1u32 << level) as f32;
    let g = |i: usize| ((origin.0 as f32 + (i % w) as f32 + 0.5) * s, (origin.1 as f32 + (i / w) as f32 + 0.5) * s);
    let (mx, my): (Vec<f32>, Vec<f32>) = match src_map {
        MapSource::Clouds => (0..w * h)
            .map(|i| {
                let (gx, gy) = g(i);
                (super::util::fbm(gx / 180.0, gy / 180.0, 4, 401), super::util::fbm(gx / 180.0 + 9.1, gy / 180.0, 4, 403))
            })
            .unzip(),
        MapSource::File { map, tile, img_w, img_h } => (0..w * h)
            .map(|i| {
                let (gx, gy) = g(i);
                if *tile {
                    map.at(gx.rem_euclid(map.w as f32), gy.rem_euclid(map.h as f32))
                } else {
                    map.at(gx / img_w * map.w as f32, gy / img_h * map.h as f32)
                }
            })
            .unzip(),
        MapSource::Layer => {
            let planes = split(buf);
            let r: Vec<f32> = planes[0].iter().map(|&v| v as f32 / 255.0).collect();
            let gg: Vec<f32> = planes[1].iter().map(|&v| v as f32 / 255.0).collect();
            (blur_f(&r, w, h, 2.0), blur_f(&gg, w, h, 2.0))
        }
    };
    let src = buf.to_vec();
    let sample = |x: f32, y: f32| -> u32 {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (((x - x0) * 255.0) as u32, ((y - y0) * 255.0) as u32);
        let at = |xi: isize, yi: isize| src[(yi.clamp(0, h as isize - 1) as usize) * w + xi.clamp(0, w as isize - 1) as usize];
        let (xi, yi) = (x0 as isize, y0 as isize);
        px::lerp(px::lerp(at(xi, yi), at(xi + 1, yi), fx), px::lerp(at(xi, yi + 1), at(xi + 1, yi + 1), fx), fy)
    };
    for i in 0..w * h {
        let (x, y) = ((i % w) as f32, (i / w) as f32);
        buf[i] = sample(x + (mx[i] - 0.5) * 2.0 * hs / s, y + (my[i] - 0.5) * 2.0 * vs / s);
    }
}

// ----- "Neural" filters done classically -------------------------------------------------

/// Skin Smoothing: edge-preserving blur restricted to skin-tone pixels
/// (YCbCr skin cluster, soft edges), so eyes, hair and background stay sharp.
pub fn skin_smoothing(buf: &mut [u32], w: usize, h: usize, radius: usize, smoothness: f32) {
    let orig = buf.to_vec();
    super::rank::surface(buf, w, h, radius.max(1), 4.0 + smoothness * 0.4, false);
    for i in 0..w * h {
        let [r, g, b, _] = px::unpremultiply(orig[i]);
        buf[i] = px::lerp(orig[i], buf[i], skin_weight(r, g, b));
    }
}

/// Skin Smoothing's skin-tone weight (0..=255) of a straight color: a soft
/// YCbCr skin cluster.
#[inline]
pub fn skin_weight(r: u8, g: u8, b: u8) -> u32 {
    let (r, g, b) = (r as f32, g as f32, b as f32);
    let cb = 128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b;
    let cr = 128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b;
    let skin = smooth(72.0, 82.0, cb) * smooth(132.0, 124.0, cb) * smooth(130.0, 140.0, cr) * smooth(180.0, 170.0, cr);
    (skin.clamp(0.0, 1.0) * 255.0) as u32
}

/// [`skin_weight`] of every straight color, `r | g << 8 | b << 16`, one byte each.
pub fn skin_table() -> Vec<u8> {
    use rayon::prelude::*;
    (0..1u32 << 24).into_par_iter().map(|i| skin_weight(i as u8, (i >> 8) as u8, (i >> 16) as u8) as u8).collect()
}

/// JPEG Artifact Removal: deblocks 8x8 boundaries (aligned to the image grid)
/// in smooth regions, then removes ringing with a light edge-preserving blur.
pub fn jpeg_cleanup(buf: &mut [u32], w: usize, h: usize, origin: (i32, i32), level: u32, strength: f32) {
    let k = (strength / 100.0).clamp(0.0, 1.0);
    if level == 0 {
        let src = buf.to_vec();
        let lum = |p: u32| {
            let [r, g, b, _] = unpack(p);
            (r as i32 * 77 + g as i32 * 150 + b as i32 * 29) >> 8
        };
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let (gx, gy) = (origin.0 + x as i32, origin.1 + y as i32);
                let edge_x = gx.rem_euclid(8) == 0 || gx.rem_euclid(8) == 7;
                let edge_y = gy.rem_euclid(8) == 0 || gy.rem_euclid(8) == 7;
                if !(edge_x || edge_y) {
                    continue;
                }
                let i = y * w + x;
                let (a, b) = if edge_x { (src[i - 1], src[i + 1]) } else { (src[i - w], src[i + w]) };
                // Only smooth genuine block steps, not real edges.
                if (lum(a) - lum(b)).abs() < 24 {
                    let avg = px::lerp(a, b, 128);
                    buf[i] = px::lerp(src[i], avg, (k * 200.0) as u32);
                }
            }
        }
    }
    super::rank::surface(buf, w, h, 2, 3.0 + k * 10.0, false);
}

/// Photo Restoration: dust and scratch removal, noise reduction, contrast
/// recovery and gentle sharpening in one pass.
pub fn restore(buf: &mut [u32], w: usize, h: usize, strength: f32, scratch: f32) {
    let k = strength / 100.0;
    super::rank::dust_scratches(buf, w, h, 1 + (scratch / 50.0) as usize, (30.0 - scratch * 0.25).max(4.0) as u8);
    super::rank::surface(buf, w, h, 2, 4.0 + k * 12.0, false);
    super::fast::blur_combine(buf, w, h, 1.2, None, super::fast::Combine::Unsharp { k: (60.0 * k * 2.56) as i32, threshold: 3 });
    let lut = super::util::lut_from(|v| (v - 0.5) * (1.0 + k * 0.25) + 0.5);
    super::util::apply_lut(buf, &lut);
}

// ----- Shape Blur -----------------------------------------------------------------------

pub const SHAPES: &[&str] = &["Disk", "Square", "Diamond", "Hexagon", "Star", "Heart"];

fn inside(shape: usize, x: f32, y: f32) -> bool {
    match shape {
        1 => x.abs() <= 1.0 && y.abs() <= 1.0,
        2 => x.abs() + y.abs() <= 1.0,
        3 => {
            let (ax, ay) = (x.abs(), y.abs());
            ay <= 0.866 && ax * 0.866 + ay * 0.5 <= 0.866
        }
        4 => {
            let r = (x * x + y * y).sqrt();
            let a = y.atan2(x) + std::f32::consts::FRAC_PI_2;
            let k = (a * 5.0 / std::f32::consts::TAU).fract();
            let edge = 0.45 + 0.55 * (1.0 - (k - 0.5).abs() * 2.0).powf(2.0);
            r <= edge
        }
        5 => {
            let (x, y) = (x * 1.15, -y * 1.15 + 0.25);
            let q = x * x + y * y - 0.6;
            q * q * q - x * x * y * y * y <= 0.0
        }
        _ => x * x + y * y <= 1.0,
    }
}

/// Blur with an arbitrary kernel shape, decomposed into horizontal spans per
/// kernel row and evaluated with row prefix sums (vectorized along x).
/// The kernel of Shape Blur as horizontal spans `[dy, x0, x1]` (inclusive).
pub fn shape_spans(radius: f32, shape: usize) -> Vec<[i32; 3]> {
    let r = radius.round().max(1.0) as isize;
    let mut spans = Vec::new();
    for dy in -r..=r {
        let mut start: Option<isize> = None;
        for dx in -r..=r + 1 {
            let on = dx <= r && inside(shape, dx as f32 / r as f32, dy as f32 / r as f32);
            match (on, start) {
                (true, None) => start = Some(dx),
                (false, Some(s)) => {
                    spans.push([dy as i32, s as i32, dx as i32 - 1]);
                    start = None;
                }
                _ => {}
            }
        }
    }
    spans
}

pub fn shape_blur(buf: &mut [u32], w: usize, h: usize, radius: f32, shape: usize) {
    let r = radius.round().max(1.0) as isize;
    let spans: Vec<(isize, isize, isize)> = shape_spans(radius, shape).iter().map(|s| (s[0] as isize, s[1] as isize, s[2] as isize)).collect();
    let count: f32 = spans.iter().map(|&(_, a, b)| (b - a + 1) as f32).sum();
    if count <= 1.0 {
        return;
    }
    let opaque = opaque(buf);
    let mut planes = split(buf);
    let chans = if opaque { 3 } else { 4 };
    for p in planes.iter_mut().take(chans) {
        let mut pre = vec![0u32; (w + 1) * h];
        for y in 0..h {
            let mut acc = 0u32;
            for x in 0..w {
                acc += p[y * w + x] as u32;
                pre[y * (w + 1) + x + 1] = acc;
            }
        }
        let mut out = vec![0u8; w * h];
        let mut acc = vec![0u32; w];
        let rr = r as usize;
        for y in 0..h {
            acc.iter_mut().for_each(|a| *a = 0);
            for &(dy, a, b) in &spans {
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                let row = &pre[yy * (w + 1)..(yy + 1) * (w + 1)];
                // Interior x where x+a >= 0 and x+b+1 <= w.
                let lo = (-a).max(0) as usize;
                let hi = (w as isize - b - 1).min(w as isize).max(lo as isize) as usize;
                for x in lo..hi {
                    acc[x] += row[(x as isize + b + 1) as usize] - row[(x as isize + a) as usize];
                }
                for x in (0..lo.min(w)).chain(hi..w) {
                    let x0 = (x as isize + a).clamp(0, w as isize) as usize;
                    let x1 = (x as isize + b + 1).clamp(0, w as isize) as usize;
                    let n = (b - a + 1) as u32;
                    let got = (x1 - x0) as u32;
                    acc[x] += if got > 0 { (row[x1] - row[x0]) * n / got } else { 0 };
                }
            }
            for (o, &a) in out[y * w..(y + 1) * w].iter_mut().zip(&acc) {
                *o = (a as f32 / count + 0.5) as u8;
            }
        }
        let _ = rr;
        *p = out;
    }
    merge(&planes, buf);
}

// ----- Variable blur (Blur Gallery) ------------------------------------------------------

/// Gaussian levels of [`variable_blur`]'s stack (level 0 is the source).
pub const VBLUR_LEVELS: usize = 5;

/// Sigma of stack level `k` (`k` in `1..VBLUR_LEVELS`).
pub fn vblur_sigma(max_sigma: f32, k: usize) -> f32 {
    max_sigma * k as f32 / (VBLUR_LEVELS - 1) as f32
}

/// Where a [`variable_blur`] runs, for the Blur Gallery effects: the
/// prepared effects, the level-space coordinate of `buf[0]` and the level's
/// scale (`2^level`).
pub struct FxAt<'a> {
    pub fx: &'a FxPrep,
    pub origin: (i32, i32),
    pub scale: f32,
}

/// Blurs with a per-pixel sigma by interpolating between a small stack of
/// Gaussian levels (each O(1) per pixel). With `fx`, the Blur Gallery
/// effects (bokeh, noise) are applied in proportion to the local blur.
pub fn variable_blur(buf: &mut [u32], w: usize, h: usize, max_sigma: f32, fx: Option<FxAt>, sigma_at: impl Fn(usize, usize) -> f32) {
    const K: usize = VBLUR_LEVELS;
    if max_sigma < 0.4 {
        return;
    }
    let opaque = opaque(buf);
    let planes = split(buf);
    let chans = if opaque { 3 } else { 4 };
    let mut stacks: Vec<Vec<Vec<u8>>> = Vec::new();
    for p in planes.iter().take(chans) {
        let mut st = vec![p.clone()];
        for k in 1..K {
            let mut b = p.clone();
            super::fast::gauss_plane(&mut b, w, h, vblur_sigma(max_sigma, k));
            st.push(b);
        }
        stacks.push(st);
    }
    let mut out = planes.clone();
    let fx = fx.filter(|f| f.fx.active());
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let t = (sigma_at(x, y) / max_sigma).clamp(0.0, 1.0) * (K - 1) as f32;
            let k0 = (t.floor() as usize).min(K - 2);
            let f = t - k0 as f32;
            match &fx {
                None => {
                    for (c, st) in stacks.iter().enumerate() {
                        let v = st[k0][i] as f32 * (1.0 - f) + st[k0 + 1][i] as f32 * f;
                        out[c][i] = (v + 0.5) as u8;
                    }
                }
                Some(at) => {
                    let mut v = [255f32; 4];
                    for (c, st) in stacks.iter().enumerate() {
                        v[c] = st[k0][i] as f32 * (1.0 - f) + st[k0 + 1][i] as f32 * f;
                    }
                    let src = pack(planes[0][i], planes[1][i], planes[2][i], planes[3][i]);
                    let gx = (at.origin.0 as f32 + x as f32 + 0.5) * at.scale;
                    let gy = (at.origin.1 as f32 + y as f32 + 0.5) * at.scale;
                    let [r, g, b, a] = unpack(at.fx.apply(v, src, t * 0.25, gx, gy));
                    out[0][i] = r;
                    out[1][i] = g;
                    out[2][i] = b;
                    out[3][i] = a;
                }
            }
        }
    }
    merge(&out, buf);
}

// ----- Blur Gallery effects (Effects and Noise panels) -------------------------------------

/// Blur Gallery's Effects panel (bokeh) and Noise panel, shared by Field,
/// Iris, Tilt-Shift, Spin and Path Blur. Both act only where the image is
/// blurred, in proportion to the local blur amount; the defaults do nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlurFx {
    /// Light Bokeh, 0–100 %: brightens highlights in the blurred area.
    pub bokeh: f32,
    /// Bokeh Color, 0–100 %: saturates the boosted highlights.
    pub bokeh_color: f32,
    /// Light Range: source luminances (0–255) that get the bokeh boost.
    pub range: [f32; 2],
    /// Noise amount, 0–100 %.
    pub noise: f32,
    /// Gaussian (true) or Uniform noise distribution.
    pub gaussian: bool,
    /// Grain size in image pixels (1–100).
    pub size: f32,
    /// Roughness, 0–100 %: smooth (0) to hard-edged (100) grain.
    pub roughness: f32,
    /// Color, 0–100 %: 0 is monochromatic noise, 100 independent per channel.
    pub color: f32,
    /// Highlights, 0–100 %: how much noise the highlights keep.
    pub highlights: f32,
}

impl Default for BlurFx {
    fn default() -> Self {
        BlurFx { bokeh: 0.0, bokeh_color: 0.0, range: [191.0, 255.0], noise: 0.0, gaussian: false, size: 1.0, roughness: 50.0, color: 0.0, highlights: 100.0 }
    }
}

/// Width (in luminance levels) of the soft edges of the Light Range.
const RANGE_SOFT: f32 = 16.0;
/// Noise amplitude at 100 % (levels).
const NOISE_FULL: f32 = 64.0;
/// Seed of the Blur Gallery grain (`hash01`).
pub const FX_SEED: u32 = 7919;

impl BlurFx {
    /// The per-pixel constants (shared verbatim with the GPU kernels).
    pub fn prep(&self) -> FxPrep {
        let (lo, hi) = (self.range[0].min(self.range[1]), self.range[0].max(self.range[1]));
        let mut wr = [0f32; 256];
        for (l, w) in wr.iter_mut().enumerate() {
            let l = l as f32;
            *w = if l < lo {
                (1.0 - (lo - l) / RANGE_SOFT).max(0.0)
            } else if l > hi {
                (1.0 - (l - hi) / RANGE_SOFT).max(0.0)
            } else {
                1.0
            };
        }
        FxPrep {
            bokeh: (self.bokeh / 100.0).clamp(0.0, 1.0),
            sat: (self.bokeh_color / 100.0).clamp(0.0, 1.0),
            wr,
            amp: (self.noise / 100.0).clamp(0.0, 1.0) * NOISE_FULL,
            gaussian: self.gaussian,
            size: self.size.clamp(1.0, 100.0),
            rough: (self.roughness / 100.0).clamp(0.0, 1.0),
            col: (self.color / 100.0).clamp(0.0, 1.0),
            keep: (self.highlights / 100.0).clamp(0.0, 1.0),
        }
    }
}

/// [`BlurFx`] prepared for per-pixel use.
#[derive(Clone, Debug)]
pub struct FxPrep {
    /// Light Bokeh as a fraction: the boost is `1 + bokeh * blur * range`.
    pub bokeh: f32,
    /// Saturation gain of the boost.
    pub sat: f32,
    /// Light Range weight per source luminance (soft-edged).
    pub wr: [f32; 256],
    /// Noise amplitude in levels.
    pub amp: f32,
    pub gaussian: bool,
    pub size: f32,
    pub rough: f32,
    pub col: f32,
    pub keep: f32,
}

/// Straight-color luminance of a premultiplied pixel, as an integer level
/// (indexes the Light Range table; the GPU computes the same integers).
#[inline]
pub fn fx_luma8(p: u32) -> usize {
    let [r, g, b, a] = unpack(p);
    if a == 0 {
        return 0;
    }
    let lp = (77 * r as u32 + 150 * g as u32 + 29 * b as u32 + 128) >> 8;
    ((lp * 255 + a as u32 / 2) / a as u32).min(255) as usize
}

impl FxPrep {
    pub fn active(&self) -> bool {
        self.bokeh > 0.0 || self.amp > 0.0
    }

    /// One lattice value of the grain (`ch` 0 = monochrome, 1–3 = r, g, b).
    #[inline]
    fn lattice(&self, x: i32, y: i32, ch: u32) -> f32 {
        let s = FX_SEED + ch * 3;
        if self.gaussian {
            // Irwin–Hall(3), scaled to the uniform noise's deviation.
            (hash01(x, y, s) + hash01(x, y, s + 1) + hash01(x, y, s + 2) - 1.5) * 1.1547
        } else {
            hash01(x, y, s) * 2.0 - 1.0
        }
    }

    /// Grain at a level-0 position: value noise on a `size` lattice, blended
    /// from smooth (smoothstep) to hard (nearest lattice point) by roughness,
    /// and from monochrome to per-channel by color.
    pub fn grain(&self, gx: f32, gy: f32) -> [f32; 3] {
        let u = gx / self.size - 0.5;
        let v = gy / self.size - 0.5;
        let (fu, fv) = (u.floor(), v.floor());
        let (fx, fy) = (u - fu, v - fv);
        let (ix, iy) = (fu as i32, fv as i32);
        let sx = fx * fx * (3.0 - 2.0 * fx);
        let sy = fy * fy * (3.0 - 2.0 * fy);
        let (nx, ny) = (ix + (fx >= 0.5) as i32, iy + (fy >= 0.5) as i32);
        let one = |ch: u32| {
            let v00 = self.lattice(ix, iy, ch);
            let v10 = self.lattice(ix + 1, iy, ch);
            let v01 = self.lattice(ix, iy + 1, ch);
            let v11 = self.lattice(ix + 1, iy + 1, ch);
            let top = v00 + (v10 - v00) * sx;
            let bot = v01 + (v11 - v01) * sx;
            let smooth = top + (bot - top) * sy;
            let near = self.lattice(nx, ny, ch);
            smooth + (near - smooth) * self.rough
        };
        let mono = if self.col < 1.0 { one(0) } else { 0.0 };
        let mut out = [mono; 3];
        if self.col > 0.0 {
            for (c, o) in out.iter_mut().enumerate() {
                *o = mono + (one(c as u32 + 1) - mono) * self.col;
            }
        }
        out
    }

    /// Applies the effects to a blurred pixel `v` (premultiplied, unrounded,
    /// alpha in `v[3]`): `src` is the source pixel there, `tb` the local blur
    /// amount (0–1), `(gx, gy)` the level-0 pixel center. Returns the
    /// rounded pixel, color clamped to alpha.
    pub fn apply(&self, v: [f32; 4], src: u32, tb: f32, gx: f32, gy: f32) -> u32 {
        let a8 = (v[3] + 0.5).clamp(0.0, 255.0) as u8;
        let af = a8 as f32;
        let mut c = [v[0], v[1], v[2]];
        if self.bokeh > 0.0 && tb > 0.0 {
            let k = self.bokeh * tb * self.wr[fx_luma8(src)];
            if k > 0.0 {
                let g = 1.0 + k;
                c = [c[0] * g, c[1] * g, c[2] * g];
                if self.sat > 0.0 {
                    let l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
                    let s = 1.0 + self.sat * k;
                    c = [l + (c[0] - l) * s, l + (c[1] - l) * s, l + (c[2] - l) * s];
                }
            }
        }
        if self.amp > 0.0 && tb > 0.0 && a8 > 0 {
            let lum = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) * 255.0 / af;
            let hw = ((lum - 192.0) / 63.0).clamp(0.0, 1.0);
            let att = 1.0 - (1.0 - self.keep) * hw;
            let scale = self.amp * tb * att * (af / 255.0);
            let n = self.grain(gx, gy);
            c = [c[0] + n[0] * scale, c[1] + n[1] * scale, c[2] + n[2] * scale];
        }
        let q = |x: f32| (x + 0.5).clamp(0.0, af) as u8;
        pack(q(c[0]), q(c[1]), q(c[2]), a8)
    }

    /// [`apply`](Self::apply) to a gather's sample sum (`acc` over `n` taps).
    pub fn apply_avg(&self, acc: [u32; 4], n: u32, src: u32, tb: f32, gx: f32, gy: f32) -> u32 {
        let nf = n as f32;
        self.apply([acc[0] as f32 / nf, acc[1] as f32 / nf, acc[2] as f32 / nf, acc[3] as f32 / nf], src, tb, gx, gy)
    }
}

// ----- Extrude / Tiles -------------------------------------------------------------------

/// Blocks or pyramids on a level-0 grid of `size`, shaded per face.
pub fn extrude(buf: &mut [u32], w: usize, h: usize, level: u32, origin: (i32, i32), size: f32, depth: f32, pyramids: bool) {
    let s = (1u32 << level) as f32;
    let size = size.max(2.0);
    // Cell average via a mosaic pass, then face shading.
    let mut avg = buf.to_vec();
    super::stylize::mosaic(&mut avg, w, h, size / s, origin);
    let dk = (depth / 100.0).clamp(0.0, 1.0) * 0.5;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let gx = (origin.0 as f32 + x as f32 + 0.5) * s;
            let gy = (origin.1 as f32 + y as f32 + 0.5) * s;
            let (u, v) = ((gx / size).fract() - 0.5, (gy / size).fract() - 0.5);
            let shade = if pyramids {
                if v.abs() > u.abs() {
                    if v < 0.0 { 1.0 + dk } else { 1.0 - dk }
                } else if u < 0.0 {
                    1.0 + dk * 0.5
                } else {
                    1.0 - dk * 0.5
                }
            } else {
                let face = 0.5 - 0.5 * dk * 1.2;
                if u.abs() <= face && v.abs() <= face {
                    1.0
                } else if v.abs() > u.abs() {
                    if v < 0.0 { 1.0 + dk } else { 1.0 - dk * 1.4 }
                } else if u < 0.0 {
                    1.0 + dk * 0.6
                } else {
                    1.0 - dk
                }
            };
            let [r, g, b, a] = unpack(avg[i]);
            let f = |c: u8| (c as f32 * shade).min(a as f32) as u8;
            buf[i] = pack(f(r), f(g), f(b), a);
        }
    }
}

/// Splits the image into tiles shifted by up to `offset`% of their size;
/// gaps are filled with `fill`.
#[allow(clippy::too_many_arguments)]
pub fn tiles(buf: &mut [u32], w: usize, h: usize, level: u32, origin: (i32, i32), size: f32, offset: f32, fill: u32) {
    let s = (1u32 << level) as f32;
    let src = buf.to_vec();
    let max = size * offset / 100.0;
    for y in 0..h {
        for x in 0..w {
            let gx = (origin.0 as f32 + x as f32 + 0.5) * s;
            let gy = (origin.1 as f32 + y as f32 + 0.5) * s;
            let mut out = fill;
            // A pixel may be covered by a shifted neighbor tile: check 3x3.
            let (cx, cy) = ((gx / size).floor() as i32, (gy / size).floor() as i32);
            for oy in -1..=1 {
                for ox in -1..=1 {
                    let (tx, ty) = (cx + ox, cy + oy);
                    let sx = (hash01(tx, ty, 501) - 0.5) * 2.0 * max;
                    let sy = (hash01(tx, ty, 503) - 0.5) * 2.0 * max;
                    let (lx, ly) = (gx - sx, gy - sy);
                    if (lx / size).floor() as i32 == tx && (ly / size).floor() as i32 == ty {
                        let bx = (lx / s - origin.0 as f32).floor() as isize;
                        let by = (ly / s - origin.1 as f32).floor() as isize;
                        out = src[(by.clamp(0, h as isize - 1) as usize) * w + bx.clamp(0, w as isize - 1) as usize];
                    }
                }
            }
            buf[y * w + x] = out;
        }
    }
}

// ----- Custom / De-Interlace ---------------------------------------------------------------

/// 3x3 convolution with Photoshop's scale and offset.
pub fn custom3(buf: &mut [u32], w: usize, h: usize, k: [f32; 9], scale: f32, offset: f32) {
    let src = buf.to_vec();
    let sc = if scale.abs() < 1e-3 { 1.0 } else { scale };
    for y in 0..h as isize {
        for x in 0..w as isize {
            let mut acc = [0f32; 3];
            for j in 0..9 {
                let (dx, dy) = ((j % 3) as isize - 1, (j / 3) as isize - 1);
                let p = unpack(src[((y + dy).clamp(0, h as isize - 1) as usize) * w + (x + dx).clamp(0, w as isize - 1) as usize]);
                for c in 0..3 {
                    acc[c] += p[c] as f32 * k[j];
                }
            }
            let i = y as usize * w + x as usize;
            let a = unpack(src[i])[3];
            let f = |v: f32| (v / sc + offset).round().clamp(0.0, a as f32) as u8;
            buf[i] = pack(f(acc[0]), f(acc[1]), f(acc[2]), a);
        }
    }
}

/// Replaces odd (or even) lines, by duplication or interpolation.
pub fn deinterlace(buf: &mut [u32], w: usize, h: usize, origin: (i32, i32), odd: bool, interpolate: bool) {
    let src = buf.to_vec();
    for y in 0..h {
        let gy = origin.1 + y as i32;
        if (gy.rem_euclid(2) == 1) != odd {
            continue;
        }
        let up = y.saturating_sub(1);
        let dn = (y + 1).min(h - 1);
        for x in 0..w {
            buf[y * w + x] = if interpolate { px::lerp(src[up * w + x], src[dn * w + x], 128) } else { src[up * w + x] };
        }
    }
}

// ----- Camera Raw ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Raw {
    // Basic
    pub temperature: f32,
    pub tint: f32,
    pub exposure: f32,
    pub contrast: f32,
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
    pub texture: f32,
    pub clarity: f32,
    pub dehaze: f32,
    pub vibrance: f32,
    pub saturation: f32,
    // Parametric tone curve
    pub curve: [f32; 4],
    // HSL: red, orange, yellow, green, aqua, blue, purple, magenta
    pub hue: [f32; 8],
    pub sat: [f32; 8],
    pub lum: [f32; 8],
    // Color grading
    pub shadow_hue: f32,
    pub shadow_sat: f32,
    pub highlight_hue: f32,
    pub highlight_sat: f32,
    pub balance: f32,
    // Detail
    pub sharpen: f32,
    pub noise_reduction: f32,
    // Effects
    pub vignette: f32,
    pub grain: f32,
    // Masks: linear gradient (angle °, position %, exposure EV) and radial (cx %, cy %, radius %, exposure EV)
    pub linear: [f32; 3],
    pub radial: [f32; 4],
    pub img_w: f32,
    pub img_h: f32,
}

pub const RAW_CLARITY_RADIUS: f32 = 30.0;
const HSL_CENTERS: [f32; 8] = [0.0, 30.0, 60.0, 120.0, 180.0, 225.0, 270.0, 315.0];

/// Weight of a hue (degrees) in each HSL band, with smooth overlap.
fn band_weights(h: f32) -> [f32; 8] {
    let mut w = [0f32; 8];
    for (k, &c) in HSL_CENTERS.iter().enumerate() {
        let d = ((h - c + 540.0) % 360.0 - 180.0).abs();
        let width = if k == 0 || k == 1 || k == 2 { 30.0 } else { 45.0 };
        w[k] = smooth(width, 0.0, d);
    }
    w
}

/// Per-image constants of [`camera_raw`] (shared with the GPU kernel so
/// both use the same values).
pub struct RawPre {
    pub wb: [f32; 3],
    pub ex: f32,
    pub hsl_on: bool,
    pub gs: [f32; 3],
    pub gh: [f32; 3],
    pub lin_dx: f32,
    pub lin_dy: f32,
}

pub fn raw_pre(r: &Raw) -> RawPre {
    let wb = [1.0 + r.temperature / 200.0 + r.tint / 400.0, 1.0 - r.tint / 200.0, 1.0 - r.temperature / 200.0 + r.tint / 400.0];
    let ex = 2f32.powf(r.exposure);
    let hsl_on = r.hue.iter().chain(&r.sat).chain(&r.lum).any(|v| *v != 0.0);
    let grade = |hue: f32, sat: f32| {
        let c = hsl_to_rgb(hue / 360.0, 1.0, 0.5);
        [(c[0] - 0.5) * sat / 100.0 * 0.3, (c[1] - 0.5) * sat / 100.0 * 0.3, (c[2] - 0.5) * sat / 100.0 * 0.3]
    };
    let (gs, gh) = (grade(r.shadow_hue, r.shadow_sat), grade(r.highlight_hue, r.highlight_sat));
    let lin_a = r.linear[0].to_radians();
    let (lin_dx, lin_dy) = (lin_a.cos(), lin_a.sin());
    RawPre { wb, ex, hsl_on, gs, gh, lin_dx, lin_dy }
}

/// Camera Raw–style develop. Local terms (texture, clarity, dehaze,
/// sharpening, noise reduction) come from blurred luminance; masks, vignette
/// and grain use global coordinates so the preview matches the commit.
pub fn camera_raw(buf: &mut [u32], w: usize, h: usize, level: u32, origin: (i32, i32), r: &Raw) {
    let s = (1u32 << level) as f32;
    let l = lum_plane(buf);
    let big = if r.clarity != 0.0 || r.dehaze != 0.0 { blur_f(&l, w, h, RAW_CLARITY_RADIUS / s) } else { l.clone() };
    let small = if r.texture != 0.0 || r.sharpen != 0.0 || r.noise_reduction != 0.0 { blur_f(&l, w, h, (2.0 / s).max(0.5)) } else { l.clone() };
    let RawPre { wb, ex, hsl_on, gs, gh, lin_dx, lin_dy } = raw_pre(r);
    for i in 0..w * h {
        let p = buf[i];
        if px::alpha(p) == 0 {
            continue;
        }
        let gx = (origin.0 as f32 + (i % w) as f32 + 0.5) * s;
        let gy = (origin.1 as f32 + (i / w) as f32 + 0.5) * s;
        let [cr, cg, cb, a] = px::unpremultiply(p);
        // Masks: extra exposure where the gradients apply.
        let mut mask_ev = 0.0;
        if r.linear[2] != 0.0 {
            let t = ((gx / r.img_w - 0.5) * lin_dx + (gy / r.img_h - 0.5) * lin_dy) + 0.5;
            mask_ev += r.linear[2] * smooth(r.linear[1] / 100.0 - 0.15, r.linear[1] / 100.0 + 0.15, t);
        }
        if r.radial[3] != 0.0 {
            let (cx, cy) = (r.img_w * r.radial[0] / 100.0, r.img_h * r.radial[1] / 100.0);
            let rad = r.img_w.min(r.img_h) * r.radial[2] / 100.0;
            let d = (gx - cx).hypot(gy - cy) / rad.max(1.0);
            mask_ev += r.radial[3] * smooth(1.0, 0.6, d);
        }
        let ex = ex * 2f32.powf(mask_ev);
        let mut c = [cr as f32 / 255.0 * wb[0] * ex, cg as f32 / 255.0 * wb[1] * ex, cb as f32 / 255.0 * wb[2] * ex];
        let y = luma(c).max(1e-4);
        // Basic tone.
        let mut ny = (y - 0.5) * (1.0 + r.contrast / 100.0) + 0.5;
        ny += r.highlights / 100.0 * 0.35 * smooth(0.45, 1.0, ny) * (if r.highlights < 0.0 { ny } else { 1.0 - ny });
        ny += r.shadows / 100.0 * 0.35 * smooth(0.55, 0.0, ny) * (if r.shadows > 0.0 { 1.0 - ny } else { ny });
        ny += r.whites / 100.0 * 0.25 * smooth(0.7, 1.0, ny);
        ny += r.blacks / 100.0 * 0.25 * smooth(0.3, 0.0, ny);
        // Parametric curve: shadows, darks, lights, highlights regions.
        let regions = [smooth(0.25, 0.0, ny), (1.0 - ((ny - 0.3).abs() / 0.25)).max(0.0), (1.0 - ((ny - 0.7).abs() / 0.25)).max(0.0), smooth(0.75, 1.0, ny)];
        let order = [r.curve[3], r.curve[2], r.curve[1], r.curve[0]];
        for (wt, amt) in regions.iter().zip(order) {
            ny += wt * amt / 100.0 * 0.25;
        }
        // Local contrast, sharpening and noise reduction on luminance.
        let detail = l[i] - small[i];
        ny += detail * (r.texture / 100.0 * 1.5 + r.sharpen / 150.0 * 1.2);
        ny -= detail * r.noise_reduction / 100.0 * 0.9;
        ny += (l[i] - big[i]) * r.clarity / 100.0 * 1.2;
        if r.dehaze != 0.0 {
            let veil = big[i] * 0.6;
            let k = r.dehaze / 100.0;
            ny = (ny - veil * k) / (1.0 - veil * k).max(0.2);
        }
        let ratio = ny.max(0.0) / y;
        for v in &mut c {
            *v *= ratio;
        }
        // HSL panel.
        if hsl_on {
            let (hh, ss, ll) = rgb_to_hsl([c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0)]);
            let bw = band_weights(hh * 360.0);
            let (mut dh, mut ds, mut dl) = (0.0, 0.0, 0.0);
            for k in 0..8 {
                dh += bw[k] * r.hue[k];
                ds += bw[k] * r.sat[k];
                dl += bw[k] * r.lum[k];
            }
            let s2 = (ss * (1.0 + ds / 100.0)).clamp(0.0, 1.0);
            let l2 = (ll + dl / 100.0 * 0.3 * ss).clamp(0.0, 1.0);
            c = hsl_to_rgb((hh + dh / 360.0 * 0.3).rem_euclid(1.0), s2, l2);
        }
        // Color grading (split toning), balance shifts the crossover.
        let yy = luma(c);
        let pivot = 0.5 + r.balance / 200.0;
        let (ws, wh) = (smooth(pivot, 0.0, yy), smooth(pivot, 1.0, yy));
        for k in 0..3 {
            c[k] += gs[k] * ws + gh[k] * wh;
        }
        // Vibrance and saturation.
        let mx = c[0].max(c[1]).max(c[2]);
        let mn = c[0].min(c[1]).min(c[2]);
        let cur = if mx > 0.0 { (mx - mn) / mx } else { 0.0 };
        let k = 1.0 + r.saturation / 100.0 + r.vibrance / 100.0 * (1.0 - cur);
        let yy = luma(c);
        for v in &mut c {
            *v = yy + (*v - yy) * k;
        }
        // Effects.
        if r.vignette != 0.0 {
            let (dx, dy) = (gx / r.img_w - 0.5, gy / r.img_h - 0.5);
            let d2 = (dx * dx + dy * dy) * 2.0;
            let f = 1.0 + r.vignette / 100.0 * d2.min(1.0);
            for v in &mut c {
                *v *= f;
            }
        }
        if r.grain != 0.0 {
            let n = (hash01(gx as i32, gy as i32, 777) - 0.5) * r.grain / 100.0 * 0.25;
            for v in &mut c {
                *v += n;
            }
        }
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        buf[i] = px::premultiply(q(c[0]), q(c[1]), q(c[2]), a);
    }
}

// ----- color adjustments as baked 3D LUTs ----------------------------------------------------

pub fn channel_mixer(m: [f32; 12], mono: bool) -> Filter {
    Filter::Color {
        name: "Channel Mixer",
        lut: Arc::new(Lut3d::bake(move |c| {
            let row = |k: usize| (c[0] * m[k * 4] + c[1] * m[k * 4 + 1] + c[2] * m[k * 4 + 2]) / 100.0 + m[k * 4 + 3] / 100.0;
            if mono {
                let v = row(0);
                [v, v, v]
            } else {
                [row(0), row(1), row(2)]
            }
        })),
    }
}

pub const SELECTIVE_RANGES: &[&str] = &["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas", "Whites", "Neutrals", "Blacks"];

/// Photoshop Selective Color for one color range, CMYK adjustments in %.
pub fn selective_color(range: usize, cmyk: [f32; 4], absolute: bool) -> Filter {
    Filter::Color {
        name: "Selective Color",
        lut: Arc::new(Lut3d::bake(move |c| {
            let mx = c[0].max(c[1]).max(c[2]);
            let mn = c[0].min(c[1]).min(c[2]);
            let (h, _, _) = rgb_to_hsl(c);
            // Weight of this pixel in the chosen range.
            let wgt = match range {
                0..=5 => {
                    let center = range as f32 / 6.0;
                    let d = ((h - center + 1.5).rem_euclid(1.0) - 0.5).abs();
                    (mx - mn) * smooth(1.0 / 6.0, 0.0, d)
                }
                6 => smooth(0.5, 1.0, mn),
                7 => 1.0 - (mx - mn) - (smooth(0.5, 1.0, mn) + smooth(0.5, 0.0, mx)) * 0.5,
                _ => smooth(0.5, 0.0, mx),
            }
            .clamp(0.0, 1.0);
            let mut o = c;
            for k in 0..3 {
                // Ink k reduces channel k; K reduces all.
                let ink = cmyk[k] / 100.0 + cmyk[3] / 100.0;
                let base = if absolute { 1.0 } else { 1.0 - c[k] };
                o[k] = c[k] - ink * base * wgt;
            }
            o
        })),
    }
}

/// Replaces colors near `target` (fuzziness in levels) by shifting HSL.
pub fn replace_color(target: [u8; 3], fuzz: f32, dh: f32, ds: f32, dl: f32) -> Filter {
    let t = [target[0] as f32 / 255.0, target[1] as f32 / 255.0, target[2] as f32 / 255.0];
    let f = (fuzz / 255.0).max(0.01);
    Filter::Color {
        name: "Replace Color",
        lut: Arc::new(Lut3d::bake(move |c| {
            let d = ((c[0] - t[0]).powi(2) + (c[1] - t[1]).powi(2) + (c[2] - t[2]).powi(2)).sqrt();
            let wgt = smooth(f, f * 0.4, d);
            if wgt <= 0.0 {
                return c;
            }
            let (h, s, l) = rgb_to_hsl(c);
            let s2 = if ds >= 0.0 { s + (1.0 - s) * ds / 100.0 } else { s * (1.0 + ds / 100.0) };
            let l2 = if dl >= 0.0 { l + (1.0 - l) * dl / 100.0 } else { l * (1.0 + dl / 100.0) };
            let o = hsl_to_rgb((h + dh / 360.0).rem_euclid(1.0), s2.clamp(0.0, 1.0), l2.clamp(0.0, 1.0));
            [c[0] + (o[0] - c[0]) * wgt, c[1] + (o[1] - c[1]) * wgt, c[2] + (o[2] - c[2]) * wgt]
        })),
    }
}

/// Mean and standard deviation per channel from histogram stats.
fn moments(st: &Stats) -> [(f32, f32); 3] {
    let n = st.count.max(1) as f64;
    [0, 1, 2].map(|k| {
        let mean = st.sum[k] / n;
        let var: f64 = st.hist[k].iter().enumerate().map(|(v, &c)| c as f64 * (v as f64 - mean).powi(2)).sum::<f64>() / n;
        ((mean / 255.0) as f32, (var.sqrt() / 255.0).max(1e-3) as f32)
    })
}

/// Match Color: Reinhard-style mean/std transfer from `source` onto `target`.
pub fn match_color(target: &Stats, source: &Stats, luminance: f32, intensity: f32, fade: f32) -> Filter {
    let (t, s) = (moments(target), moments(source));
    let lum = luminance / 100.0;
    let inten = intensity / 100.0;
    let fade = fade / 100.0;
    Filter::Color {
        name: "Match Color",
        lut: Arc::new(Lut3d::bake(move |c| {
            let mut o = [0f32; 3];
            for k in 0..3 {
                o[k] = (c[k] - t[k].0) * (s[k].1 / t[k].1) + s[k].0;
            }
            // Color intensity scales chroma, luminance scales overall level.
            let ly = luma(o);
            for v in &mut o {
                *v = ly * lum + (*v - ly) * inten;
            }
            [c[0] + (o[0] - c[0]) * (1.0 - fade), c[1] + (o[1] - c[1]) * (1.0 - fade), c[2] + (o[2] - c[2]) * (1.0 - fade)]
        })),
    }
}

/// Limits saturation and luminance to broadcast-safe ranges.
pub fn ntsc_colors() -> Filter {
    Filter::Color {
        name: "NTSC Colors",
        lut: Arc::new(Lut3d::bake(|c| {
            let y = luma(c);
            let chroma = [c[0] - y, c[1] - y, c[2] - y];
            let mag = (chroma[0].powi(2) + chroma[1].powi(2) + chroma[2].powi(2)).sqrt();
            let k = if mag > 0.33 { 0.33 / mag } else { 1.0 };
            let y = y.clamp(16.0 / 255.0, 235.0 / 255.0);
            [y + chroma[0] * k, y + chroma[1] * k, y + chroma[2] * k]
        })),
    }
}

// ----- Lens Correction (geometric, per channel) --------------------------------------------

/// Barrel/pincushion distortion, lateral chromatic aberration and vignette.
#[allow(clippy::too_many_arguments)]
pub fn render_lens_correction(
    distortion: f32,
    ca_rc: f32,
    ca_by: f32,
    vignette: f32,
    w0: f32,
    h0: f32,
    src: &super::Src,
    level: u32,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    out: &mut [u32],
) {
    let s = (1u32 << level) as f32;
    let (cx, cy) = (w0 * 0.5, h0 * 0.5);
    let rn = (cx * cx + cy * cy).sqrt();
    let k = -distortion / 100.0 * 0.5;
    let grid = src.at(level);
    let sampler = super::distort::Sampler::new(grid, None, w * h, false);
    for y in 0..h {
        for x in 0..w {
            let gx = (x0 as f32 + x as f32 + 0.5) * s;
            let gy = (y0 as f32 + y as f32 + 0.5) * s;
            let (dx, dy) = ((gx - cx) / rn, (gy - cy) / rn);
            let r2 = dx * dx + dy * dy;
            let f = 1.0 + k * r2;
            let mut o = [0u8; 4];
            for (c, sc) in [(0usize, 1.0 + ca_rc / 1000.0), (1, 1.0), (2, 1.0 + ca_by / 1000.0)] {
                let (sx, sy) = (cx + dx * rn * f * sc, cy + dy * rn * f * sc);
                o[c] = unpack(sampler.sample(sx / s, sy / s))[c];
                if c == 1 {
                    o[3] = unpack(sampler.sample(sx / s, sy / s))[3];
                }
            }
            let v = 1.0 + vignette / 100.0 * r2.min(1.0) * 1.2;
            let a = o[3];
            let q = |c: u8| (c as f32 * v).clamp(0.0, a as f32) as u8;
            out[y * w + x] = pack(q(o[0]), q(o[1]), q(o[2]), a);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_blur_keeps_flat() {
        let c = pack(30, 60, 90, 255);
        for shape in 0..SHAPES.len() {
            let mut b = vec![c; 40 * 40];
            shape_blur(&mut b, 40, 40, 6.0, shape);
            assert_eq!(unpack(b[20 * 40 + 20]), unpack(c), "{}", SHAPES[shape]);
        }
    }

    #[test]
    fn identity_mixer_is_identity() {
        let f = channel_mixer([100., 0., 0., 0., 0., 100., 0., 0., 0., 0., 100., 0.], false);
        let mut b = vec![pack(10, 200, 77, 255)];
        f.apply(&mut b, 1, 1, 0, (0, 0));
        let o = unpack(b[0]);
        assert!((o[0] as i32 - 10).abs() <= 1 && (o[1] as i32 - 200).abs() <= 1 && (o[2] as i32 - 77).abs() <= 1);
    }
}

#[cfg(test)]
mod fx_tests {
    use super::*;
    use crate::engine::filters::Src;
    use crate::engine::grid::TileGrid;

    /// A gray scene with small bright (and one colored) lights.
    fn scene(w: usize, h: usize) -> Vec<u32> {
        let mut b = vec![pack(60, 70, 80, 255); w * h];
        for y in 0..h {
            for x in 0..w {
                if x % 16 < 3 && y % 16 < 3 {
                    b[y * w + x] = if (x / 16) % 2 == 0 { pack(250, 250, 245, 255) } else { pack(240, 200, 120, 255) };
                }
            }
        }
        b
    }

    fn tilt(fx: BlurFx) -> Filter {
        Filter::TiltShift { cy: 40.0, band: 10.0, transition: 30.0, blur: 8.0, fx }
    }

    fn run(f: &Filter, w: usize, h: usize) -> Vec<u32> {
        let mut b = scene(w, h);
        f.apply(&mut b, w, h, 0, (0, 0));
        b
    }

    fn lum(p: u32) -> f32 {
        let [r, g, b, _] = unpack(p);
        0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32
    }

    fn sat(p: u32) -> i32 {
        let [r, g, b, _] = unpack(p);
        r.max(g).max(b) as i32 - r.min(g).min(b) as i32
    }

    #[test]
    fn defaults_change_nothing() {
        assert!(!BlurFx::default().prep().active());
        let (w, h) = (96, 160);
        let mut a = scene(w, h);
        let mut b = a.clone();
        let sig = |_: usize, y: usize| (y as f32 / 20.0).min(6.0);
        variable_blur(&mut a, w, h, 6.0, None, sig);
        let prep = BlurFx::default().prep();
        variable_blur(&mut b, w, h, 6.0, Some(FxAt { fx: &prep, origin: (0, 0), scale: 1.0 }), sig);
        assert_eq!(a, b);
        // Missing catalog values (older files) take the defaults.
        assert_eq!(super::super::catalog::blur_fx(&[1.0, 2.0], 2), BlurFx::default());
    }

    #[test]
    fn bokeh_brightens_blurred_highlights_only() {
        let (w, h) = (96, 160);
        let plain = run(&tilt(BlurFx::default()), w, h);
        let bok = run(&tilt(BlurFx { bokeh: 100.0, ..Default::default() }), w, h);
        // In focus (rows 30..50): untouched.
        assert_eq!(&plain[30 * w..50 * w], &bok[30 * w..50 * w]);
        // Fully blurred rows: lights brighter, the gray background (outside
        // the Light Range) the same.
        let (mut up, mut same) = (0, 0);
        for y in 120..150 {
            for x in 8..88 {
                let i = y * w + x;
                if x % 16 < 3 && y % 16 < 3 {
                    assert!(lum(bok[i]) >= lum(plain[i]));
                    up += (lum(bok[i]) > lum(plain[i]) + 3.0) as usize;
                } else if x % 16 > 7 && y % 16 > 7 {
                    same += (bok[i] == plain[i]) as usize;
                }
            }
        }
        assert!(up >= 80, "{up}");
        assert!(same > 300, "{same}");
    }

    #[test]
    fn light_range_selects_source_luminance() {
        let (w, h) = (96, 160);
        let plain = run(&tilt(BlurFx::default()), w, h);
        // Only dark sources: the lights are not boosted, the background is.
        let dark = run(&tilt(BlurFx { bokeh: 100.0, range: [0.0, 100.0], ..Default::default() }), w, h);
        let (y, x) = (129, 1);
        assert_eq!(dark[y * w + x], plain[y * w + x], "light boosted");
        let (y, x) = (140, 10);
        assert!(lum(dark[y * w + x]) > lum(plain[y * w + x]) + 5.0);
    }

    #[test]
    fn bokeh_color_saturates_the_boost() {
        let (w, h) = (96, 160);
        let bok = run(&tilt(BlurFx { bokeh: 80.0, ..Default::default() }), w, h);
        let col = run(&tilt(BlurFx { bokeh: 80.0, bokeh_color: 100.0, ..Default::default() }), w, h);
        // The orange lights (odd columns of lights) get more saturated.
        let i = 129 * w + 17;
        assert!(sat(col[i]) > sat(bok[i]), "{:?} vs {:?}", unpack(col[i]), unpack(bok[i]));
    }

    #[test]
    fn noise_matches_blur_amount_and_is_deterministic() {
        let (w, h) = (96, 160);
        let fx = BlurFx { noise: 50.0, ..Default::default() };
        let plain = run(&tilt(BlurFx::default()), w, h);
        let a = run(&tilt(fx), w, h);
        let b = run(&tilt(fx), w, h);
        assert_eq!(a, b);
        assert_eq!(&plain[30 * w..50 * w], &a[30 * w..50 * w], "noise in focus");
        let diff = (120 * w..150 * w).filter(|&i| a[i] != plain[i]).count();
        assert!(diff > 30 * w / 2, "{diff}");
        // Monochromatic by default: equal deltas on every channel.
        for i in 120 * w..150 * w {
            let (p, q) = (unpack(plain[i]), unpack(a[i]));
            let d: Vec<i32> = (0..3).map(|c| q[c] as i32 - p[c] as i32).collect();
            if p[0] > 30 && p[0] < 200 && p[2] < 200 {
                assert!((d[0] - d[1]).abs() <= 1 && (d[1] - d[2]).abs() <= 1, "{d:?}");
            }
        }
        // Color noise: channels differ.
        let c = run(&tilt(BlurFx { color: 100.0, ..fx }), w, h);
        let chroma = (120 * w..150 * w)
            .filter(|&i| {
                let (p, q) = (unpack(plain[i]), unpack(c[i]));
                (q[0] as i32 - p[0] as i32) != (q[2] as i32 - p[2] as i32)
            })
            .count();
        assert!(chroma > 30 * w / 2, "{chroma}");
        // Gaussian is a different grain.
        let g = run(&tilt(BlurFx { gaussian: true, ..fx }), w, h);
        assert_ne!(&g[120 * w..150 * w], &a[120 * w..150 * w]);
    }

    #[test]
    fn noise_size_roughness_and_highlights() {
        let prep = |fx: BlurFx| fx.prep();
        // Size: neighbors on a coarse lattice are correlated.
        let fine = prep(BlurFx { noise: 100.0, ..Default::default() });
        let coarse = prep(BlurFx { noise: 100.0, size: 8.0, roughness: 0.0, ..Default::default() });
        let step = |p: &FxPrep| (0..200).map(|x| (p.grain(x as f32 + 1.5, 7.5)[0] - p.grain(x as f32 + 0.5, 7.5)[0]).abs()).sum::<f32>() / 200.0;
        assert!(step(&coarse) * 3.0 < step(&fine), "{} vs {}", step(&coarse), step(&fine));
        // Roughness 100: hard cells (constant inside a lattice cell, which
        // spans [0, 8) here), changing at the cell edge.
        let hard = prep(BlurFx { noise: 100.0, size: 8.0, roughness: 100.0, ..Default::default() });
        let (p, q, r) = (hard.grain(1.5, 1.5)[0], hard.grain(6.5, 5.5)[0], hard.grain(8.5, 5.5)[0]);
        assert!((p - q).abs() < 1e-6 && (q - r).abs() > 1e-3, "{p} {q} {r}");
        // Highlights 0: no grain on white, still grain on gray.
        let fx = prep(BlurFx { noise: 100.0, highlights: 0.0, ..Default::default() });
        let white = [255.0, 255.0, 255.0, 255.0];
        let w = pack(255, 255, 255, 255);
        assert!((0..50).all(|x| fx.apply(white, w, 1.0, x as f32 + 0.5, 3.5) == w));
        let gray = [100.0, 100.0, 100.0, 255.0];
        assert!((0..50).any(|x| fx.apply(gray, w, 1.0, x as f32 + 0.5, 3.5) != pack(100, 100, 100, 255)));
        // No blur, no grain.
        assert!((0..50).all(|x| fx.apply(gray, w, 0.0, x as f32 + 0.5, 3.5) == pack(100, 100, 100, 255)));
    }

    #[test]
    fn spin_and_path_effects() {
        let (w, h) = (256usize, 256usize);
        let mut g = TileGrid::new(w as u32, h as u32);
        let px = scene(w, h);
        g.tiles = crate::engine::grid::build_grid(w as u32, h as u32, |_, _| crate::engine::grid::finish_tile(px.clone().into_boxed_slice())).tiles;
        let src = Src::single(&g);
        let render = |f: &Filter| {
            let mut o = vec![0u32; w * h];
            f.render(&src, 0, 0, 0, w, h, &mut o, &mut Vec::new());
            o
        };
        let fx = BlurFx { bokeh: 100.0, noise: 30.0, ..Default::default() };
        let spin = |fx| Filter::SpinBlur { cx: 128.0, cy: 128.0, rx: 60.0, ry: 60.0, amount: 30.0, fx };
        let (a, b) = (render(&spin(BlurFx::default())), render(&spin(fx)));
        // Outside the ellipse: the source, effects or not.
        for &(x, y) in &[(5usize, 5usize), (250, 10), (10, 250)] {
            assert_eq!(a[y * w + x], px[y * w + x]);
            assert_eq!(b[y * w + x], px[y * w + x]);
        }
        let inside = (100..156).flat_map(|y| (100..156).map(move |x| y * w + x)).filter(|&i| a[i] != b[i]).count();
        assert!(inside > 1000, "{inside}");
        let path = |fx| Filter::PathBlur { x0: 20.0, y0: 200.0, x1: 230.0, y1: 40.0, bend: 30.0, length: 40.0, fx };
        let (a, b) = (render(&path(BlurFx::default())), render(&path(fx)));
        assert!(a.iter().zip(&b).filter(|(p, q)| p != q).count() > w * h / 2);
    }
}
