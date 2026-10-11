//! Shared helpers for filters: plane split/merge, LUTs, 3D color LUTs, hashing.

use crate::engine::px::{self, pack, unpack};

/// Splits premultiplied pixels into four channel planes.
pub fn split(buf: &[u32]) -> [Vec<u8>; 4] {
    let n = buf.len();
    let mut p = [vec![0u8; n], vec![0u8; n], vec![0u8; n], vec![0u8; n]];
    let [r, g, b, a] = &mut p;
    for (i, &v) in buf.iter().enumerate() {
        let [c0, c1, c2, c3] = unpack(v);
        r[i] = c0;
        g[i] = c1;
        b[i] = c2;
        a[i] = c3;
    }
    p
}

/// Packs four planes back into pixels, clamping color to alpha.
pub fn merge(p: &[Vec<u8>; 4], out: &mut [u32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let a = p[3][i];
        *o = pack(p[0][i].min(a), p[1][i].min(a), p[2][i].min(a), a);
    }
}

/// Whether every pixel is opaque (lets filters skip the alpha plane).
pub fn opaque(buf: &[u32]) -> bool {
    buf.iter().all(|&p| p >= 0xFF00_0000)
}

pub fn lut_from(f: impl Fn(f32) -> f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = (f(i as f32 / 255.0) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Applies per-channel LUTs to straight color, keeping alpha.
pub fn apply_lut3(buf: &mut [u32], lut: &[[u8; 256]; 3]) {
    for p in buf.iter_mut() {
        let a = px::alpha(*p);
        if a == 0 {
            continue;
        }
        if a == 255 {
            let [r, g, b, _] = unpack(*p);
            *p = pack(lut[0][r as usize], lut[1][g as usize], lut[2][b as usize], 255);
        } else {
            let [r, g, b, a8] = px::unpremultiply(*p);
            *p = px::premultiply(lut[0][r as usize], lut[1][g as usize], lut[2][b as usize], a8);
        }
    }
}

pub fn apply_lut(buf: &mut [u32], lut: &[u8; 256]) {
    apply_lut3(buf, &[*lut, *lut, *lut]);
}

/// Per-pixel straight-color map (for cheap transforms).
pub fn map_straight(buf: &mut [u32], f: impl Fn([u8; 3]) -> [u8; 3]) {
    for p in buf.iter_mut() {
        let a = px::alpha(*p);
        if a == 0 {
            continue;
        }
        let [r, g, b, a8] = if a == 255 { unpack(*p) } else { px::unpremultiply(*p) };
        let [r, g, b] = f([r, g, b]);
        *p = px::premultiply(r, g, b, a8);
    }
}

/// A 33³ color lookup table evaluated with tetrahedral interpolation (four
/// taps instead of trilinear's eight). Any color transform, however
/// expensive, costs the same few multiply-adds per pixel once baked.
#[derive(Debug, PartialEq)]
pub struct Lut3d {
    data: Vec<[u16; 3]>,
}

const N: usize = 33;

impl Lut3d {
    pub fn bake(f: impl Fn([f32; 3]) -> [f32; 3] + Sync) -> Lut3d {
        use rayon::prelude::*;
        let data = (0..N * N * N)
            .into_par_iter()
            .map(|i| {
                let r = (i % N) as f32 / (N - 1) as f32;
                let g = ((i / N) % N) as f32 / (N - 1) as f32;
                let b = (i / (N * N)) as f32 / (N - 1) as f32;
                let o = f([r, g, b]);
                [
                    (o[0].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
                    (o[1].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
                    (o[2].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
                ]
            })
            .collect();
        Lut3d { data }
    }

    /// The 33³ grid (r fastest), 16-bit per channel.
    pub fn raw(&self) -> &[[u16; 3]] {
        &self.data
    }

    #[inline(always)]
    fn at(&self, r: usize, g: usize, b: usize) -> [i32; 3] {
        let v = self.data[r + g * N + b * N * N];
        [v[0] as i32, v[1] as i32, v[2] as i32]
    }

    #[inline]
    pub fn eval(&self, c: [u8; 3]) -> [u8; 3] {
        // 8-bit value v maps to v * 32 / 255 in grid units; keep 8 fractional bits.
        let s = |v: u8| -> (usize, i32) {
            let f = v as u32 * 32 * 256 / 255;
            (((f >> 8) as usize).min(N - 2), (f - ((f >> 8).min(N as u32 - 2) << 8)) as i32)
        };
        let (r0, fr) = s(c[0]);
        let (g0, fg) = s(c[1]);
        let (b0, fb) = s(c[2]);
        let c000 = self.at(r0, g0, b0);
        let c111 = self.at(r0 + 1, g0 + 1, b0 + 1);
        // Tetrahedral interpolation: pick the simplex by ordering the fractions.
        let (w0, w1, w2, w3, p1, p2);
        if fr >= fg {
            if fg >= fb {
                (w0, w1, w2, w3) = (256 - fr, fr - fg, fg - fb, fb);
                p1 = self.at(r0 + 1, g0, b0);
                p2 = self.at(r0 + 1, g0 + 1, b0);
            } else if fr >= fb {
                (w0, w1, w2, w3) = (256 - fr, fr - fb, fb - fg, fg);
                p1 = self.at(r0 + 1, g0, b0);
                p2 = self.at(r0 + 1, g0, b0 + 1);
            } else {
                (w0, w1, w2, w3) = (256 - fb, fb - fr, fr - fg, fg);
                p1 = self.at(r0, g0, b0 + 1);
                p2 = self.at(r0 + 1, g0, b0 + 1);
            }
        } else if fb >= fg {
            (w0, w1, w2, w3) = (256 - fb, fb - fg, fg - fr, fr);
            p1 = self.at(r0, g0, b0 + 1);
            p2 = self.at(r0, g0 + 1, b0 + 1);
        } else if fb >= fr {
            (w0, w1, w2, w3) = (256 - fg, fg - fb, fb - fr, fr);
            p1 = self.at(r0, g0 + 1, b0);
            p2 = self.at(r0, g0 + 1, b0 + 1);
        } else {
            (w0, w1, w2, w3) = (256 - fg, fg - fr, fr - fb, fb);
            p1 = self.at(r0, g0 + 1, b0);
            p2 = self.at(r0 + 1, g0 + 1, b0);
        }
        let mut out = [0u8; 3];
        for k in 0..3 {
            let v = c000[k] * w0 + p1[k] * w1 + p2[k] * w2 + c111[k] * w3;
            // v is 16-bit value << 8; convert to 8-bit with rounding.
            out[k] = (((v as i64 + 128) >> 8) as i32 * 255 / 65535).clamp(0, 255) as u8;
        }
        out
    }

    pub fn apply(&self, buf: &mut [u32]) {
        let mut last_in = u32::MAX;
        let mut last_out = 0u32;
        for p in buf.iter_mut() {
            let v = *p;
            if v == last_in {
                *p = last_out;
                continue;
            }
            let a = px::alpha(v);
            if a == 0 {
                continue;
            }
            let o = if a == 255 {
                let [r, g, b, _] = unpack(v);
                let [r, g, b] = self.eval([r, g, b]);
                pack(r, g, b, 255)
            } else {
                let [r, g, b, a8] = px::unpremultiply(v);
                let [r, g, b] = self.eval([r, g, b]);
                px::premultiply(r, g, b, a8)
            };
            last_in = v;
            last_out = o;
            *p = o;
        }
    }
}

#[inline]
pub fn luma(c: [f32; 3]) -> f32 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

pub fn rgb_to_hsl(c: [f32; 3]) -> (f32, f32, f32) {
    let (r, g, b) = (c[0], c[1], c[2]);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let l = (mx + mn) * 0.5;
    if mx - mn < 1e-6 {
        return (0.0, 0.0, l);
    }
    let d = mx - mn;
    let s = if l > 0.5 { d / (2.0 - mx - mn) } else { d / (mx + mn) };
    let h = if mx == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if mx == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    (h / 6.0, s, l)
}

pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    if s <= 0.0 {
        return [l, l, l];
    }
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let f = |t: f32| {
        let t = t.rem_euclid(1.0);
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    };
    [f(h + 1.0 / 3.0), f(h), f(h - 1.0 / 3.0)]
}

/// Stateless integer hash → [0, 1).
#[inline(always)]
pub fn hash01(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

/// Smooth value noise with fractal Brownian motion (for Clouds / Fibers).
pub fn fbm(x: f32, y: f32, octaves: u32, seed: u32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for o in 0..octaves {
        sum += amp * value_noise(x * freq, y * freq, seed.wrapping_add(o * 977));
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

/// Smoothstep-interpolated lattice noise over `hash01` (one fbm octave).
#[inline]
pub fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let fx = x - xi;
    let fy = y - yi;
    let (xi, yi) = (xi as i32, yi as i32);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (u, v) = (s(fx), s(fy));
    let a = hash01(xi, yi, seed);
    let b = hash01(xi + 1, yi, seed);
    let c = hash01(xi, yi + 1, seed);
    let d = hash01(xi + 1, yi + 1, seed);
    a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut3d_identity_is_exact() {
        let l = Lut3d::bake(|c| c);
        for &c in &[[0u8, 0, 0], [255, 255, 255], [12, 200, 77], [128, 64, 250], [1, 254, 3]] {
            let o = l.eval(c);
            for k in 0..3 {
                assert!((o[k] as i32 - c[k] as i32).abs() <= 1, "{c:?} -> {o:?}");
            }
        }
    }
}
