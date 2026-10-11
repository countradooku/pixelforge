//! Stylize and Pixelate filters. Buffers are padded; only the center is
//! required to be correct. `origin`/`level` give global coordinates so
//! position-dependent effects (cells, noise) line up across tiles and zoom
//! levels.

use super::util::{hash01, merge, opaque, split};
use crate::engine::px::{self, pack, unpack};

#[inline]
fn lum8(p: u32) -> i32 {
    let [r, g, b, _] = unpack(p);
    (r as i32 * 77 + g as i32 * 150 + b as i32 * 29) >> 8
}

#[inline]
fn at(buf: &[u32], w: usize, h: usize, x: isize, y: isize) -> u32 {
    buf[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize]
}

/// Sobel magnitude per channel, inverted like Photoshop's Find Edges.
pub fn find_edges(buf: &mut [u32], w: usize, h: usize) {
    let src = buf.to_vec();
    for y in 0..h as isize {
        for x in 0..w as isize {
            let g = |dx: isize, dy: isize| unpack(at(&src, w, h, x + dx, y + dy));
            let (tl, t, tr, l, r, bl, b, br) = (g(-1, -1), g(0, -1), g(1, -1), g(-1, 0), g(1, 0), g(-1, 1), g(0, 1), g(1, 1));
            let mut o = [0u8; 4];
            for c in 0..3 {
                let gx = (tr[c] as i32 + 2 * r[c] as i32 + br[c] as i32) - (tl[c] as i32 + 2 * l[c] as i32 + bl[c] as i32);
                let gy = (bl[c] as i32 + 2 * b[c] as i32 + br[c] as i32) - (tl[c] as i32 + 2 * t[c] as i32 + tr[c] as i32);
                let m = (((gx * gx + gy * gy) as f32).sqrt() * 0.5) as i32;
                o[c] = (255 - m).clamp(0, 255) as u8;
            }
            let a = unpack(src[y as usize * w + x as usize])[3];
            o[3] = a;
            buf[y as usize * w + x as usize] = px::premultiply(o[0], o[1], o[2], a);
        }
    }
}

/// Gray relief from the luminance gradient along `angle`.
pub fn emboss(buf: &mut [u32], w: usize, h: usize, angle: f32, height: f32, amount: f32) {
    let src = buf.to_vec();
    let a = angle.to_radians();
    let (dx, dy) = ((a.cos() * height).round() as isize, (-a.sin() * height).round() as isize);
    let k = amount / 100.0;
    for y in 0..h as isize {
        for x in 0..w as isize {
            let d = lum8(at(&src, w, h, x + dx, y + dy)) - lum8(at(&src, w, h, x - dx, y - dy));
            let v = (128.0 + d as f32 * k).clamp(0.0, 255.0) as u8;
            let al = unpack(src[y as usize * w + x as usize])[3];
            buf[y as usize * w + x as usize] = px::premultiply(v, v, v, al);
        }
    }
}

/// Outlines where each channel crosses `level` (Trace Contour).
pub fn trace_contour(buf: &mut [u32], w: usize, h: usize, level: u8) {
    let src = buf.to_vec();
    for y in 0..h as isize {
        for x in 0..w as isize {
            let c = unpack(at(&src, w, h, x, y));
            let r = unpack(at(&src, w, h, x + 1, y));
            let d = unpack(at(&src, w, h, x, y + 1));
            let mut o = [255u8; 3];
            for k in 0..3 {
                let above = c[k] >= level;
                if above != (r[k] >= level) || above != (d[k] >= level) {
                    o[k] = 0;
                }
            }
            buf[y as usize * w + x as usize] = px::premultiply(o[0], o[1], o[2], c[3]);
        }
    }
}

/// Swaps each pixel with a pseudo-random neighbor (deterministic per position).
pub fn diffuse(buf: &mut [u32], w: usize, h: usize, radius: f32, origin: (i32, i32)) {
    let src = buf.to_vec();
    let r = radius.max(1.0);
    for y in 0..h {
        for x in 0..w {
            let gx = origin.0 + x as i32;
            let gy = origin.1 + y as i32;
            let ox = ((hash01(gx, gy, 11) * 2.0 - 1.0) * r).round() as isize;
            let oy = ((hash01(gx, gy, 23) * 2.0 - 1.0) * r).round() as isize;
            buf[y * w + x] = at(&src, w, h, x as isize + ox, y as isize + oy);
        }
    }
}

/// Wind: bright pixels streak with a linear falloff. A decaying running max
/// along each row makes it O(1) per pixel for any streak length.
pub fn wind(buf: &mut [u32], w: usize, h: usize, length: f32, from_left: bool) {
    let decay = 255.0 / length.max(1.0);
    let mut planes = split(buf);
    for p in planes.iter_mut().take(3) {
        for y in 0..h {
            let row = &mut p[y * w..(y + 1) * w];
            let mut run = 0.0f32;
            let mut step = |v: &mut u8| {
                run = (run - decay).max(*v as f32);
                *v = run as u8;
            };
            if from_left {
                row.iter_mut().for_each(&mut step);
            } else {
                row.iter_mut().rev().for_each(&mut step);
            }
        }
    }
    merge(&planes, buf);
}

/// Kuwahara oil-paint: each pixel takes the mean of whichever of its four
/// (r+1)² quadrant windows has the lowest luminance variance. Window sums
/// come from separable running box sums (u32, row-vectorized), so the cost
/// per pixel is constant in r.
pub fn kuwahara(buf: &mut [u32], w: usize, h: usize, r: usize) {
    if r == 0 || w <= 2 * r || h <= 2 * r {
        return;
    }
    if r <= 3 {
        kuwahara_small(buf, w, h, r);
        return;
    }
    let s = r + 1;
    let (bw, bh) = (w - s + 1, h - s + 1);
    // Planes: 4 channels, luma, luma².
    let n = w * h;
    let mut planes: Vec<Vec<u32>> = vec![vec![0u32; n]; 6];
    for (i, &p) in buf.iter().enumerate() {
        let c = unpack(p);
        let l = (c[0] as u32 * 77 + c[1] as u32 * 150 + c[2] as u32 * 29) >> 8;
        planes[0][i] = c[0] as u32;
        planes[1][i] = c[1] as u32;
        planes[2][i] = c[2] as u32;
        planes[3][i] = c[3] as u32;
        planes[4][i] = l;
        planes[5][i] = l * l;
    }
    // Box sums with top-left at (x, y): horizontal then vertical.
    let sums: Vec<Vec<u32>> = planes
        .iter()
        .map(|a| {
            let mut hs = vec![0u32; bw * h];
            for y in 0..h {
                let row = &a[y * w..(y + 1) * w];
                let mut acc: u32 = row[..s].iter().sum();
                let o = &mut hs[y * bw..(y + 1) * bw];
                o[0] = acc;
                for x in 1..bw {
                    acc = acc + row[x + s - 1] - row[x - 1];
                    o[x] = acc;
                }
            }
            let mut vs = vec![0u32; bw * bh];
            let mut acc = vec![0u32; bw];
            for y in 0..s {
                for (a_, &v) in acc.iter_mut().zip(&hs[y * bw..(y + 1) * bw]) {
                    *a_ += v;
                }
            }
            vs[..bw].copy_from_slice(&acc);
            for y in 1..bh {
                let (add, sub) = (&hs[(y + s - 1) * bw..(y + s) * bw], &hs[(y - 1) * bw..y * bw]);
                for x in 0..bw {
                    acc[x] = acc[x] + add[x] - sub[x];
                }
                vs[y * bw..(y + 1) * bw].copy_from_slice(&acc);
            }
            vs
        })
        .collect();
    let cnt = (s * s) as u64;
    if s <= 16 {
        // u32 variance keys, computed per row for all x at once.
        let c32 = cnt as u32;
        let ow = w - 2 * r;
        let mut keys = vec![[0u32; 4]; ow];
        let mut best = vec![0usize; ow];
        for y in r..h - r {
            let offs = [((y - r) * bw, 0usize), ((y - r) * bw, r), (y * bw, 0), (y * bw, r)];
            for (q, &(row, dx)) in offs.iter().enumerate() {
                let l1 = &sums[4][row + dx..row + dx + ow];
                let l2 = &sums[5][row + dx..row + dx + ow];
                for ((k, &a), &b) in keys.iter_mut().zip(l1).zip(l2) {
                    k[q] = c32 * b - a * a;
                }
            }
            for (x, (k, bi)) in keys.iter().zip(best.iter_mut()).enumerate() {
                let mut q = 0;
                let mut m = k[0];
                for (j, &kv) in k.iter().enumerate().skip(1) {
                    if kv < m {
                        m = kv;
                        q = j;
                    }
                }
                let (row, dx) = offs[q];
                *bi = row + dx + x;
            }
            let half = c32 / 2;
            for (x, &i) in best.iter().enumerate() {
                let ch = |k: usize| ((sums[k][i] + half) / c32) as u8;
                let a = ch(3);
                buf[y * w + x + r] = pack(ch(0).min(a), ch(1).min(a), ch(2).min(a), a);
            }
        }
        return;
    }
    for y in r..h - r {
        for x in r..w - r {
            let quads = [(x - r, y - r), (x, y - r), (x - r, y), (x, y)];
            let mut best = (u64::MAX, 0usize);
            for &(qx, qy) in quads.iter() {
                let i = qy * bw + qx;
                let l = sums[4][i] as u64;
                // n² · variance = n·Σl² − (Σl)²
                let v = cnt * sums[5][i] as u64 - l * l;
                if v < best.0 {
                    best = (v, i);
                }
            }
            let i = best.1;
            let half = cnt as u32 / 2;
            let ch = |k: usize| ((sums[k][i] + half) / cnt as u32) as u8;
            let a = ch(3);
            buf[y * w + x] = pack(ch(0).min(a), ch(1).min(a), ch(2).min(a), a);
        }
    }
}

/// Small-window Kuwahara: (r+1)² window sums by shifted-slice adds, all
/// contiguous and 8–16 lanes wide; quadrant choice is branch-light per row.
fn kuwahara_small(buf: &mut [u32], w: usize, h: usize, r: usize) {
    let s = r + 1;
    let (bw, bh) = (w - s + 1, h - s + 1);
    let n = w * h;
    let mut ch = [vec![0u16; n], vec![0u16; n], vec![0u16; n], vec![0u16; n]];
    let mut lum = vec![0u16; n];
    let mut lum2 = vec![0u32; n];
    for (i, &p) in buf.iter().enumerate() {
        let c = unpack(p);
        let l = (c[0] as u32 * 77 + c[1] as u32 * 150 + c[2] as u32 * 29) >> 8;
        ch[0][i] = c[0] as u16;
        ch[1][i] = c[1] as u16;
        ch[2][i] = c[2] as u16;
        ch[3][i] = c[3] as u16;
        lum[i] = l as u16;
        lum2[i] = l * l;
    }
    fn box16(a: &[u16], w: usize, h: usize, s: usize, bw: usize, bh: usize) -> Vec<u16> {
        let mut hs = vec![0u16; bw * h];
        for y in 0..h {
            let row = &a[y * w..(y + 1) * w];
            let o = &mut hs[y * bw..(y + 1) * bw];
            o.copy_from_slice(&row[..bw]);
            for j in 1..s {
                for (d, &v) in o.iter_mut().zip(&row[j..j + bw]) {
                    *d += v;
                }
            }
        }
        let mut vs = vec![0u16; bw * bh];
        for y in 0..bh {
            let o = &mut vs[y * bw..(y + 1) * bw];
            o.copy_from_slice(&hs[y * bw..(y + 1) * bw]);
            for j in 1..s {
                for (d, &v) in o.iter_mut().zip(&hs[(y + j) * bw..(y + j + 1) * bw]) {
                    *d += v;
                }
            }
        }
        vs
    }
    fn box32(a: &[u32], w: usize, h: usize, s: usize, bw: usize, bh: usize) -> Vec<u32> {
        let mut hs = vec![0u32; bw * h];
        for y in 0..h {
            let row = &a[y * w..(y + 1) * w];
            let o = &mut hs[y * bw..(y + 1) * bw];
            o.copy_from_slice(&row[..bw]);
            for j in 1..s {
                for (d, &v) in o.iter_mut().zip(&row[j..j + bw]) {
                    *d += v;
                }
            }
        }
        let mut vs = vec![0u32; bw * bh];
        for y in 0..bh {
            let o = &mut vs[y * bw..(y + 1) * bw];
            o.copy_from_slice(&hs[y * bw..(y + 1) * bw]);
            for j in 1..s {
                for (d, &v) in o.iter_mut().zip(&hs[(y + j) * bw..(y + j + 1) * bw]) {
                    *d += v;
                }
            }
        }
        vs
    }
    let cs: Vec<Vec<u16>> = ch.iter().map(|a| box16(a, w, h, s, bw, bh)).collect();
    let ls = box16(&lum, w, h, s, bw, bh);
    let l2s = box32(&lum2, w, h, s, bw, bh);
    let cnt = (s * s) as u32;
    let half = cnt / 2;
    let ow = w - 2 * r;
    let mut keys = vec![[0u32; 4]; ow];
    for y in r..h - r {
        let offs = [(y - r) * bw, (y - r) * bw + r, y * bw, y * bw + r];
        for (q, &o) in offs.iter().enumerate() {
            let l1 = &ls[o..o + ow];
            let l2 = &l2s[o..o + ow];
            for ((k, &a), &b) in keys.iter_mut().zip(l1).zip(l2) {
                let a = a as u32;
                k[q] = cnt * b - a * a;
            }
        }
        for (x, k) in keys.iter().enumerate() {
            let mut q = 0;
            let mut m = k[0];
            for (j, &kv) in k.iter().enumerate().skip(1) {
                if kv < m {
                    m = kv;
                    q = j;
                }
            }
            let i = offs[q] + x;
            let c = |k: usize| ((cs[k][i] as u32 + half) / cnt) as u8;
            let a = c(3);
            buf[y * w + x + r] = pack(c(0).min(a), c(1).min(a), c(2).min(a), a);
        }
    }
}

// ----- Oil Paint ------------------------------------------------------------------------

/// Oil Paint's controls beyond the brush size, after Photoshop's dialog
/// (Brush: Stylization, Cleanliness, Scale, Bristle Detail; Lighting:
/// Angle, Shine). The defaults leave the paint stage off, so Oil Paint is
/// exactly the plain Kuwahara filter unless one of Stylization, Bristle
/// Detail or Shine is raised.
///
/// The stage after Kuwahara is continuous float math (no thresholds, no
/// quantization until the end), so the GPU reproduces it within 1/255.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Oil {
    /// Stroke length along the image's flow, 0..=10 (0: no oriented strokes).
    pub stylization: f32,
    /// Smoothness of the stroke direction field, 0..=10.
    pub cleanliness: f32,
    /// Size of strokes and bristles, 0.1..=10.
    pub scale: f32,
    /// Bristle texture along the strokes, 0..=10.
    pub bristle: f32,
    /// Direction the light comes from, degrees counter-clockwise from +x.
    pub angle: f32,
    /// Relief lighting of the paint surface, 0..=10 (0: flat).
    pub shine: f32,
}

impl Default for Oil {
    fn default() -> Self {
        Oil { stylization: 0.0, cleanliness: 5.0, scale: 1.0, bristle: 0.0, angle: -60.0, shine: 0.0 }
    }
}

/// The paint stage's geometry at one mip level (lengths in level pixels).
#[derive(Clone, Debug, PartialEq)]
pub struct OilGeom {
    /// Level scale `2^level`.
    pub s: f32,
    /// Stroke samples: half-length, steps per side, step, weights (2n + 1),
    /// reciprocal of the weight sum. `ns == 0`: no oriented strokes.
    pub ns: usize,
    pub step_s: f32,
    pub wts_s: Vec<f32>,
    pub inv_s: f32,
    /// Bristle samples along the stroke (`nb == 0`: no bristles).
    pub nb: usize,
    pub step_b: f32,
    pub wts_b: Vec<f32>,
    pub inv_b: f32,
    /// Reciprocal of the bristle lattice spacing (level-0 px).
    pub inv_lambda: f32,
    /// Structure-tensor window radius (0: no tensor).
    pub rho: usize,
    /// Light direction (image space, y down) and gain (0: no lighting).
    pub light: (f32, f32),
    pub gain: f32,
    /// Bristle gains: color modulation and relief height.
    pub kc: f32,
    pub kh: f32,
}

/// Seed of the bristle noise.
pub const OIL_SEED: u32 = 71;

fn line_weights(n: usize) -> (Vec<f32>, f32) {
    let w: Vec<f32> = (-(n as i32)..=n as i32)
        .map(|k| {
            let t = k as f32 / (n + 1) as f32;
            1.0 - t * t
        })
        .collect();
    let sum: f32 = w.iter().sum();
    (w, 1.0 / sum)
}

impl Oil {
    /// Whether anything runs after the Kuwahara pass.
    pub fn active(&self) -> bool {
        self.stylization > 0.0 || self.bristle > 0.0 || self.shine > 0.0
    }

    pub fn geom(&self, level: u32) -> OilGeom {
        let s = (1u32 << level) as f32;
        let sc = self.scale.clamp(0.1, 10.0);
        let ls = self.stylization.clamp(0.0, 10.0) * 1.5 * sc / s;
        let ns = if self.stylization > 0.0 { (ls.ceil() as usize).max(1) } else { 0 };
        let (wts_s, inv_s) = if ns > 0 { line_weights(ns) } else { (Vec::new(), 0.0) };
        let lb = (self.stylization.clamp(0.0, 10.0) * 1.5).max(3.0) * sc / s;
        let nb = if self.bristle > 0.0 { (lb.ceil() as usize).max(1) } else { 0 };
        let (wts_b, inv_b) = if nb > 0 { line_weights(nb) } else { (Vec::new(), 0.0) };
        let rho = if ns > 0 || nb > 0 { ((1.0 + self.cleanliness.clamp(0.0, 10.0) * 0.5) * sc / s).round().clamp(1.0, 40.0) as usize } else { 0 };
        let a = self.angle.to_radians();
        OilGeom {
            s,
            ns,
            step_s: if ns > 0 { ls / ns as f32 } else { 0.0 },
            wts_s,
            inv_s,
            nb,
            step_b: if nb > 0 { lb / nb as f32 } else { 0.0 },
            wts_b,
            inv_b,
            inv_lambda: 1.0 / (1.2 * sc),
            rho,
            light: (a.cos(), -a.sin()),
            gain: self.shine.clamp(0.0, 10.0) * 2.5,
            kc: self.bristle.clamp(0.0, 10.0) * 0.06,
            kh: self.bristle.clamp(0.0, 10.0) * 0.04,
        }
    }

    /// The apron the whole filter needs for a Kuwahara radius `r`.
    pub fn pad(&self, r: usize, level: u32) -> usize {
        if !self.active() {
            return r;
        }
        let g = self.geom(level);
        let reach = (g.ns as f32 * g.step_s).max(g.nb as f32 * g.step_b).ceil() as usize;
        (r + reach + 3).max(g.rho + 3)
    }
}

/// Unit stroke direction (along edges) and anisotropy from a smoothed
/// structure tensor `(Σgx², Σgx·gy, Σgy²)`.
#[inline]
pub fn oil_tangent(j: [i32; 3]) -> (f32, f32, f32) {
    let (a, b, c) = (j[0] as f32, j[1] as f32, j[2] as f32);
    let d = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let aniso = d / (a + c + 1.0);
    if d > 0.0 {
        let c2 = (a - c) / d;
        let cp = ((1.0 + c2) * 0.5).max(0.0).sqrt();
        let sp = ((1.0 - c2) * 0.5).max(0.0).sqrt();
        let sp = if b < 0.0 { -sp } else { sp };
        (-sp, cp, aniso)
    } else {
        (1.0, 0.0, aniso)
    }
}

/// Smooth value noise (the lattice of `hash01`), for the bristles.
#[inline]
pub fn oil_noise(x: f32, y: f32) -> f32 {
    super::util::value_noise(x, y, OIL_SEED)
}

/// Bilinear sample of premultiplied channels at `(x + dx, y + dy)`, split
/// as integer pixel plus offset so the result doesn't depend on where the
/// buffer starts (clamped to the buffer).
#[inline]
fn bilerp(buf: &[u32], w: usize, h: usize, x: usize, y: usize, dx: f32, dy: f32) -> [f32; 4] {
    let axis = |p: usize, d: f32, n: usize| -> (usize, usize, f32) {
        let f = d.floor();
        let i = p as isize + f as isize;
        if i < 0 {
            (0, 0, 0.0)
        } else if i >= n as isize - 1 {
            (n - 1, n - 1, 0.0)
        } else {
            (i as usize, i as usize + 1, d - f)
        }
    };
    let (xi, xj, fx) = axis(x, dx, w);
    let (yi, yj, fy) = axis(y, dy, h);
    let (p00, p10, p01, p11) = (unpack(buf[yi * w + xi]), unpack(buf[yi * w + xj]), unpack(buf[yj * w + xi]), unpack(buf[yj * w + xj]));
    let mut o = [0f32; 4];
    for c in 0..4 {
        let t = p00[c] as f32 + (p10[c] as f32 - p00[c] as f32) * fx;
        let b = p01[c] as f32 + (p11[c] as f32 - p01[c] as f32) * fx;
        o[c] = t + (b - t) * fy;
    }
    o
}

/// Structure tensor of the luma: Sobel products (`>> 4`), box-summed over
/// `(2 rho + 1)²`. Valid at least `rho + 1` from the buffer edge (zero
/// elsewhere).
pub fn oil_tensor(src: &[u32], w: usize, h: usize, rho: usize) -> Vec<[i32; 3]> {
    let n = w * h;
    let l: Vec<i32> = src.iter().map(|&p| lum8(p)).collect();
    let mut j = vec![[0i32; 3]; n];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w - 1 {
            let at = |dx: isize, dy: isize| l[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
            let gx = (at(1, -1) + 2 * at(1, 0) + at(1, 1)) - (at(-1, -1) + 2 * at(-1, 0) + at(-1, 1));
            let gy = (at(-1, 1) + 2 * at(0, 1) + at(1, 1)) - (at(-1, -1) + 2 * at(0, -1) + at(1, -1));
            j[y * w + x] = [(gx * gx) >> 4, (gx * gy) >> 4, (gy * gy) >> 4];
        }
    }
    let m = rho + 1;
    let mut hs = vec![[0i32; 3]; n];
    for y in 0..h {
        for x in m..w.saturating_sub(m) {
            let mut acc = [0i32; 3];
            for k in x - rho..=x + rho {
                for c in 0..3 {
                    acc[c] += j[y * w + k][c];
                }
            }
            hs[y * w + x] = acc;
        }
    }
    let mut out = vec![[0i32; 3]; n];
    for y in m..h.saturating_sub(m) {
        for x in m..w.saturating_sub(m) {
            let mut acc = [0i32; 3];
            for k in y - rho..=y + rho {
                for c in 0..3 {
                    acc[c] += hs[k * w + x][c];
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Oil Paint: Kuwahara with radius `r`, then (when `oil` is active) strokes
/// along the image's flow, bristle texture and relief lighting.
#[allow(clippy::too_many_arguments)]
pub fn oil_paint(buf: &mut [u32], w: usize, h: usize, r: usize, oil: &Oil, level: u32, origin: (i32, i32)) {
    if !oil.active() {
        kuwahara(buf, w, h, r);
        return;
    }
    let src = buf.to_vec();
    kuwahara(buf, w, h, r);
    let g = oil.geom(level);
    let n = w * h;
    let tens = if g.rho > 0 { oil_tensor(&src, w, h, g.rho) } else { Vec::new() };
    // Paint (F) and bristle (T) planes everywhere but the outermost ring.
    let mut f = vec![[0f32; 4]; n];
    let mut t = vec![0.5f32; n];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let k0 = unpack(buf[i]);
            let (tx, ty, an) = if g.rho > 0 { oil_tangent(tens[i]) } else { (1.0, 0.0, 0.0) };
            let mut fv = [k0[0] as f32, k0[1] as f32, k0[2] as f32, k0[3] as f32];
            if g.ns > 0 {
                let mut acc = [0f32; 4];
                for k in 0..2 * g.ns + 1 {
                    let o = (k as i32 - g.ns as i32) as f32 * g.step_s;
                    let v = bilerp(buf, w, h, x, y, o * tx, o * ty);
                    for c in 0..4 {
                        acc[c] += v[c] * g.wts_s[k];
                    }
                }
                for c in 0..4 {
                    fv[c] += (acc[c] * g.inv_s - fv[c]) * an;
                }
            }
            f[i] = fv;
            if g.nb > 0 {
                let (gx, gy) = ((origin.0 + x as i32) as f32, (origin.1 + y as i32) as f32);
                let mut acc = 0f32;
                for k in 0..2 * g.nb + 1 {
                    let o = (k as i32 - g.nb as i32) as f32 * g.step_b;
                    let u = (gx + o * tx + 0.5) * g.s * g.inv_lambda;
                    let v = (gy + o * ty + 0.5) * g.s * g.inv_lambda;
                    acc += oil_noise(u, v) * g.wts_b[k];
                }
                t[i] = acc * g.inv_b;
            }
        }
    }
    let height = |i: usize| (0.299 * f[i][0] + 0.587 * f[i][1] + 0.114 * f[i][2]) / 255.0 + (t[i] - 0.5) * g.kh;
    let half = 0.5 / g.s;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let fv = f[i];
            let m = 1.0 + (t[i] - 0.5) * g.kc;
            let e = if g.gain > 0.0 {
                let hx = (height(i + 1) - height(i - 1)) * half;
                let hy = (height(i + w) - height(i - w)) * half;
                -(hx * g.light.0 + hy * g.light.1) * g.gain
            } else {
                0.0
            };
            let a8 = (fv[3] + 0.5).clamp(0.0, 255.0) as u8;
            let q = |c: usize| (fv[c] * m + e * fv[3] + 0.5).clamp(0.0, a8 as f32) as u8;
            buf[i] = pack(q(0), q(1), q(2), a8);
        }
    }
}

/// Square cells averaged over their full extent, aligned to global coordinates.
pub fn mosaic(buf: &mut [u32], w: usize, h: usize, cell: f32, origin: (i32, i32)) {
    let cell = cell.max(1.0);
    let src = buf.to_vec();
    let cx = |x: usize| ((origin.0 as f32 + x as f32) / cell).floor() as i64;
    let cy = |y: usize| ((origin.1 as f32 + y as f32) / cell).floor() as i64;
    // Accumulate per-cell sums for every cell touching the buffer.
    let (c0x, c1x) = (cx(0), cx(w - 1));
    let (c0y, c1y) = (cy(0), cy(h - 1));
    let nx = (c1x - c0x + 1) as usize;
    let ny = (c1y - c0y + 1) as usize;
    let mut sums = vec![[0u32; 5]; nx * ny];
    let xcell: Vec<usize> = (0..w).map(|x| (cx(x) - c0x) as usize).collect();
    for y in 0..h {
        let row = (cy(y) - c0y) as usize * nx;
        for x in 0..w {
            let p = unpack(src[y * w + x]);
            let s = &mut sums[row + xcell[x]];
            for c in 0..4 {
                s[c] += p[c] as u32;
            }
            s[4] += 1;
        }
    }
    for y in 0..h {
        let row = (cy(y) - c0y) as usize * nx;
        for x in 0..w {
            let s = sums[row + xcell[x]];
            let n = s[4].max(1);
            buf[y * w + x] = pack(((s[0] + n / 2) / n) as u8, ((s[1] + n / 2) / n) as u8, ((s[2] + n / 2) / n) as u8, ((s[3] + n / 2) / n) as u8);
        }
    }
}

/// Jittered-grid Voronoi: returns (nearest seed, distance to it, distance to the second).
#[inline]
fn voronoi(gx: f32, gy: f32, cell: f32) -> ((f32, f32), f32, f32) {
    let ix = (gx / cell).floor() as i32;
    let iy = (gy / cell).floor() as i32;
    let mut best = ((0.0, 0.0), f32::MAX);
    let mut second = f32::MAX;
    for oy in -1..=1 {
        for ox in -1..=1 {
            let (cxi, cyi) = (ix + ox, iy + oy);
            let sx = (cxi as f32 + 0.1 + 0.8 * hash01(cxi, cyi, 7)) * cell;
            let sy = (cyi as f32 + 0.1 + 0.8 * hash01(cxi, cyi, 13)) * cell;
            let d = (sx - gx).powi(2) + (sy - gy).powi(2);
            if d < best.1 {
                second = best.1;
                best = ((sx, sy), d);
            } else if d < second {
                second = d;
            }
        }
    }
    (best.0, best.1.sqrt(), second.sqrt())
}

/// Crystallize (`dots == None`) or Pointillize (`dots == Some(bg)`).
/// Every Voronoi cell is filled with the average of the pixels it covers
/// (as Photoshop does): pass 1 assigns pixels to cells and accumulates sums,
/// pass 2 writes the means. Cells are defined in level-0 space so they line
/// up across tiles and zoom levels; the buffer padding (two cells) guarantees
/// every cell touching the output is fully inside the buffer.
pub fn cells(buf: &mut [u32], w: usize, h: usize, cell: f32, level: u32, origin: (i32, i32), dots: Option<u32>) {
    let s = (1u32 << level) as f32;
    let c0x = (((origin.0 as f32) * s / cell).floor() as i32) - 1;
    let c0y = (((origin.1 as f32) * s / cell).floor() as i32) - 1;
    let c1x = ((((origin.0 + w as i32) as f32) * s / cell).floor() as i32) + 1;
    let c1y = ((((origin.1 + h as i32) as f32) * s / cell).floor() as i32) + 1;
    let (nx, ny) = ((c1x - c0x + 1) as usize, (c1y - c0y + 1) as usize);
    let mut sums = vec![[0u32; 5]; nx * ny];
    let mut ids = vec![0u32; w * h];
    let mut dist = if dots.is_some() { vec![0f32; w * h] } else { Vec::new() };
    for y in 0..h {
        for x in 0..w {
            let gx = (origin.0 as f32 + x as f32 + 0.5) * s;
            let gy = (origin.1 as f32 + y as f32 + 0.5) * s;
            let ((sx, sy), d, _) = voronoi(gx, gy, cell);
            let cx = ((sx / cell).floor() as i32 - c0x).clamp(0, nx as i32 - 1) as usize;
            let cy = ((sy / cell).floor() as i32 - c0y).clamp(0, ny as i32 - 1) as usize;
            let id = cy * nx + cx;
            ids[y * w + x] = id as u32;
            if dots.is_some() {
                dist[y * w + x] = d;
            }
            let p = unpack(buf[y * w + x]);
            let acc = &mut sums[id];
            for c in 0..4 {
                acc[c] += p[c] as u32;
            }
            acc[4] += 1;
        }
    }
    let means: Vec<u32> = sums
        .iter()
        .map(|a| {
            let n = a[4].max(1);
            pack(((a[0] + n / 2) / n) as u8, ((a[1] + n / 2) / n) as u8, ((a[2] + n / 2) / n) as u8, ((a[3] + n / 2) / n) as u8)
        })
        .collect();
    for i in 0..w * h {
        let c = means[ids[i] as usize];
        buf[i] = match dots {
            None => c,
            Some(bg) => {
                let rad = cell * 0.42;
                let cov = ((rad - dist[i]) / s + 0.5).clamp(0.0, 1.0);
                px::lerp(bg, c, (cov * 255.0) as u32)
            }
        };
    }
}

/// RGB color halftone: per-channel rotated screens with anti-aliased dots.
pub fn halftone(buf: &mut [u32], w: usize, h: usize, radius: f32, level: u32, origin: (i32, i32)) {
    let s = (1u32 << level) as f32;
    let src = buf.to_vec();
    let cell = radius * 2.0;
    let angles = [108f32.to_radians(), 162f32.to_radians(), 90f32.to_radians()];
    for y in 0..h {
        for x in 0..w {
            let gx = (origin.0 as f32 + x as f32 + 0.5) * s;
            let gy = (origin.1 as f32 + y as f32 + 0.5) * s;
            let a8 = unpack(src[y * w + x])[3];
            let mut o = [0u8; 3];
            for (k, &ang) in angles.iter().enumerate() {
                let (sn, cs) = ang.sin_cos();
                let u = gx * cs + gy * sn;
                let v = -gx * sn + gy * cs;
                let cu = ((u / cell).floor() + 0.5) * cell;
                let cv = ((v / cell).floor() + 0.5) * cell;
                // Back to image space to sample the channel at the cell center.
                let ix = cu * cs - cv * sn;
                let iy = cu * sn + cv * cs;
                let val = unpack(at(&src, w, h, (ix / s - origin.0 as f32) as isize, (iy / s - origin.1 as f32) as isize))[k];
                let ink = 1.0 - val as f32 / 255.0;
                let rad = radius * 1.41 * ink.sqrt();
                let d = ((u - cu).powi(2) + (v - cv).powi(2)).sqrt();
                let cov = ((rad - d) / s + 0.5).clamp(0.0, 1.0);
                o[k] = (255.0 * (1.0 - cov)) as u8;
            }
            buf[y * w + x] = px::premultiply(o[0], o[1], o[2], a8);
        }
    }
}

/// Fragment: average of four copies offset diagonally.
pub fn fragment(buf: &mut [u32], w: usize, h: usize, d: f32) {
    let src = buf.to_vec();
    let d = d.round().max(1.0) as isize;
    for y in 0..h as isize {
        for x in 0..w as isize {
            let mut s = [0u32; 4];
            for (ox, oy) in [(-d, -d), (d, -d), (-d, d), (d, d)] {
                let p = unpack(at(&src, w, h, x + ox, y + oy));
                for c in 0..4 {
                    s[c] += p[c] as u32;
                }
            }
            buf[y as usize * w + x as usize] = pack(((s[0] + 2) / 4) as u8, ((s[1] + 2) / 4) as u8, ((s[2] + 2) / 4) as u8, ((s[3] + 2) / 4) as u8);
        }
    }
}

/// Mezzotint: per-channel random threshold dither (fine dots).
pub fn mezzotint(buf: &mut [u32], origin: (i32, i32), w: usize) {
    for (i, p) in buf.iter_mut().enumerate() {
        let a = px::alpha(*p);
        if a == 0 {
            continue;
        }
        let gx = origin.0 + (i % w) as i32;
        let gy = origin.1 + (i / w) as i32;
        let [r, g, b, a8] = px::unpremultiply(*p);
        let t = |v: u8, s: u32| if v as f32 > hash01(gx, gy, s) * 255.0 { 255 } else { 0 };
        *p = px::premultiply(t(r, 1), t(g, 2), t(b, 3), a8);
    }
}

#[allow(dead_code)]
pub fn is_opaque(buf: &[u32]) -> bool {
    opaque(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::filters::{Filter, Src};

    fn noise_buf(w: usize, h: usize, seed: u32) -> Vec<u32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                let v = |s: u32| (hash01(x, y, seed + s) * 255.0) as u8;
                pack(v(0), v(1), v(2), 255)
            })
            .collect()
    }

    #[test]
    fn oil_defaults_are_plain_kuwahara() {
        let oil = Oil::default();
        assert!(!oil.active());
        for r in [1usize, 2, 4, 9, 20] {
            assert_eq!(oil.pad(r, 0), r);
            let src = noise_buf(97, 83, r as u32);
            let mut a = src.clone();
            let mut b = src.clone();
            kuwahara(&mut a, 97, 83, r);
            oil_paint(&mut b, 97, 83, r, &oil, 0, (5, 7));
            assert_eq!(a, b, "r = {r}");
        }
        // Any Cleanliness, Scale or Angle alone changes nothing.
        let quiet = Oil { cleanliness: 9.0, scale: 3.0, angle: 45.0, ..Default::default() };
        assert!(!quiet.active());
    }

    #[test]
    fn oil_parameters_change_the_paint_and_keep_alpha() {
        let src = noise_buf(120, 100, 3);
        let mut plain = src.clone();
        kuwahara(&mut plain, 120, 100, 3);
        for oil in [
            Oil { stylization: 5.0, ..Default::default() },
            Oil { bristle: 6.0, ..Default::default() },
            Oil { shine: 4.0, ..Default::default() },
            Oil { stylization: 10.0, cleanliness: 0.0, scale: 0.5, bristle: 10.0, angle: 90.0, shine: 10.0 },
        ] {
            let pad = oil.pad(3, 0);
            let mut b = src.clone();
            oil_paint(&mut b, 120, 100, 3, &oil, 0, (0, 0));
            let mut differ = 0;
            for y in pad..100 - pad {
                for x in pad..120 - pad {
                    let p = b[y * 120 + x];
                    let c = unpack(p);
                    assert_eq!(c[3], 255);
                    differ += (p != plain[y * 120 + x]) as usize;
                }
            }
            assert!(differ > 100, "{oil:?}: only {differ} pixels changed");
        }
    }

    /// Strokes follow the flow: horizontal stripes with per-pixel noise keep
    /// their stripes while the noise along them is smoothed away.
    #[test]
    fn oil_strokes_follow_the_image_flow() {
        let (w, h) = (160usize, 120usize);
        let src: Vec<u32> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let base = if (y / 6) % 2 == 0 { 60.0 } else { 190.0 };
                let v = (base + (hash01(x as i32, y as i32, 9) - 0.5) * 70.0) as u8;
                pack(v, v, v, 255)
            })
            .collect();
        let row_stats = |b: &[u32]| {
            // Mean |difference| along rows and across stripes (center rows).
            let (mut along, mut across, mut n) = (0f64, 0f64, 0f64);
            for y in 30..90 {
                for x in 30..130 {
                    let v = |x: usize, y: usize| unpack(b[y * w + x])[0] as f64;
                    along += (v(x + 1, y) - v(x, y)).abs();
                    across += (v(x, y + 6) - v(x, y)).abs();
                    n += 1.0;
                }
            }
            (along / n, across / n)
        };
        let mut plain = src.clone();
        kuwahara(&mut plain, w, h, 1);
        let mut styl = src.clone();
        oil_paint(&mut styl, w, h, 1, &Oil { stylization: 8.0, ..Default::default() }, 0, (0, 0));
        let (a0, c0) = row_stats(&plain);
        let (a1, c1) = row_stats(&styl);
        assert!(a1 < a0 * 0.6, "noise along the stripes: {a0:.1} -> {a1:.1}");
        assert!(c1 > c0 * 0.85, "stripe contrast: {c0:.1} -> {c1:.1}");
    }

    /// Shine lights the edge facing the light and shades the other.
    #[test]
    fn oil_relief_follows_the_light_angle() {
        let (w, h) = (80usize, 40usize);
        let src: Vec<u32> = (0..w * h).map(|i| if i % w < 40 { pack(60, 60, 60, 255) } else { pack(200, 200, 200, 255) }).collect();
        let lit = |angle: f32| {
            let mut b = src.clone();
            oil_paint(&mut b, w, h, 2, &Oil { shine: 5.0, angle, ..Default::default() }, 0, (0, 0));
            // The dark side of the step: the height rises to the right.
            unpack(b[20 * w + 39])[0] as i32
        };
        // Light from the right (0°): the rising step faces away from it.
        assert!(lit(0.0) < 60, "{}", lit(0.0));
        assert!(lit(180.0) > 60, "{}", lit(180.0));
        // Light from above or below: no change across a vertical edge.
        assert_eq!(lit(90.0), 60);
    }

    /// The commit (tile blocks) equals one whole-image render: the pad
    /// covers every stage's reach.
    #[test]
    fn oil_paint_is_block_independent() {
        let g = crate::engine::ops::procedural(700, 530);
        let f = Filter::OilPaint { radius: 5.0, oil: Oil { stylization: 6.0, cleanliness: 4.0, scale: 1.5, bristle: 4.0, angle: -60.0, shine: 3.0 } };
        let c = crate::engine::ops::apply_filter_cpu(&g, &f, None, &Default::default(), None);
        let (tw, th) = (g.cols as usize * 256, g.rows as usize * 256);
        let mut out = vec![0u32; tw * th];
        f.render(&Src::single(&g), 0, 0, 0, tw, th, &mut out, &mut Vec::new());
        for y in 0..g.h as usize {
            for x in 0..g.w as usize {
                assert_eq!(c.pixel(x as i32, y as i32), out[y * tw + x], "({x}, {y})");
            }
        }
    }

    #[test]
    fn oil_geometry_scales_with_the_level() {
        let oil = Oil { stylization: 4.0, bristle: 2.0, shine: 1.0, ..Default::default() };
        let (g0, g2) = (oil.geom(0), oil.geom(2));
        assert_eq!(g0.ns, 6);
        assert!(g2.ns <= 2 && g2.ns >= 1);
        assert!(g2.rho <= g0.rho);
        assert!(oil.pad(4, 0) >= 4 + g0.ns + 3 && oil.pad(4, 0) >= g0.rho + 3);
        let s: f32 = g0.wts_s.iter().sum();
        assert!((s * g0.inv_s - 1.0).abs() < 1e-6);
    }
}
