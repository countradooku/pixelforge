//! Image ▸ Image Size resampling methods (Photoshop's list) on the CPU.
//!
//! - **Bilinear** is the original mip-aware bilinear resize
//!   ([`super::ops::resize`]), unchanged and still the dialog's default.
//! - **Bicubic** (smooth gradients) is Catmull-Rom (Keys a = -0.5),
//!   **Bicubic Sharper** (reduction) Keys a = -0.75, **Bicubic Smoother**
//!   (enlargement) Mitchell-Netravali B = C = 1/3 (the B-spline side of the
//!   cubic family), **Lanczos** a 3-lobe windowed sinc.
//! - **Nearest Neighbor** copies the source pixel under each centre.
//! - **Preserve Details** (enlargement) follows Adobe's description: an
//!   enlargement that keeps edges crisp, with a Reduce Noise slider. Here: a
//!   Lanczos-3 upscale L, a cubic B-spline upscale S of the same source (the
//!   source's low-pass), and the detail band L − S added back with an
//!   edge-aware gain driven by S's local gradient: edges get extra detail,
//!   flat areas are pulled towards S as Reduce Noise rises.
//!   **Preserve Details 2.0** uses a stronger edge gain and clamps every
//!   pixel to the min / max of L's 3 × 3 neighbourhood, so the extra
//!   sharpness can't overshoot into halos.
//! - **Automatic** picks Bicubic Sharper for reductions and Bicubic Smoother
//!   for enlargements, like Photoshop.
//!
//! The filter methods are separable and run in fixed point so the GPU
//! reproduces them bit for bit: per-axis tap tables (start index, 14-bit
//! weights summing to exactly 2^14) are built here and shared with the GPU;
//! the horizontal pass keeps 6 fractional bits in an i16, the vertical pass
//! rounds once. Edge pixels repeat. Reductions by 4× or more start from the
//! mip level that leaves a 2–4× reduction, so the tap count stays bounded.

use super::document::Layer;
use super::grid::{TS, TS_I, TileGrid, build_grid, finish_tile};

/// Image Size ▸ Resample.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Resample {
    Automatic,
    PreserveDetails,
    PreserveDetails2,
    BicubicSmoother,
    BicubicSharper,
    Bicubic,
    NearestNeighbor,
    /// The original mip-aware bilinear resize (default: unchanged pixels).
    #[default]
    Bilinear,
    Lanczos,
}

impl Resample {
    pub const ALL: [Resample; 9] = [
        Resample::Automatic,
        Resample::PreserveDetails,
        Resample::PreserveDetails2,
        Resample::BicubicSmoother,
        Resample::BicubicSharper,
        Resample::Bicubic,
        Resample::NearestNeighbor,
        Resample::Bilinear,
        Resample::Lanczos,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Resample::Automatic => "Automatic",
            Resample::PreserveDetails => "Preserve Details (enlargement)",
            Resample::PreserveDetails2 => "Preserve Details 2.0",
            Resample::BicubicSmoother => "Bicubic Smoother (enlargement)",
            Resample::BicubicSharper => "Bicubic Sharper (reduction)",
            Resample::Bicubic => "Bicubic (smooth gradients)",
            Resample::NearestNeighbor => "Nearest Neighbor (hard edges)",
            Resample::Bilinear => "Bilinear",
            Resample::Lanczos => "Lanczos-3",
        }
    }

    /// Whether the method has the Reduce Noise slider.
    pub fn has_noise(self) -> bool {
        matches!(self, Resample::PreserveDetails | Resample::PreserveDetails2)
    }

    /// The concrete method for a `sw x sh` → `dw x dh` resize (Automatic's pick).
    pub fn resolve(self, sw: u32, sh: u32, dw: u32, dh: u32) -> Resample {
        match self {
            Resample::Automatic => {
                if (dw as u64 * dh as u64) < sw as u64 * sh as u64 {
                    Resample::BicubicSharper
                } else {
                    Resample::BicubicSmoother
                }
            }
            m => m,
        }
    }

    /// Reconstruction kernel of the separable methods.
    pub fn kernel(self) -> Option<Kernel> {
        match self {
            Resample::Bicubic => Some(Kernel::CatmullRom),
            Resample::BicubicSharper => Some(Kernel::Sharper),
            Resample::BicubicSmoother => Some(Kernel::Mitchell),
            Resample::Lanczos => Some(Kernel::Lanczos3),
            _ => None,
        }
    }
}

/// 1-D reconstruction kernels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kernel {
    /// Keys cubic, a = -0.5.
    CatmullRom,
    /// Keys cubic, a = -0.75.
    Sharper,
    /// Mitchell-Netravali, B = C = 1/3.
    Mitchell,
    /// Cubic B-spline (B = 1, C = 0): the smooth base of Preserve Details.
    BSpline,
    /// Lanczos, 3 lobes.
    Lanczos3,
}

impl Kernel {
    pub fn radius(self) -> f64 {
        match self {
            Kernel::Lanczos3 => 3.0,
            _ => 2.0,
        }
    }

    pub fn eval(self, x: f64) -> f64 {
        let (b, c) = match self {
            Kernel::CatmullRom => (0.0, 0.5),
            Kernel::Sharper => (0.0, 0.75),
            Kernel::Mitchell => (1.0 / 3.0, 1.0 / 3.0),
            Kernel::BSpline => (1.0, 0.0),
            Kernel::Lanczos3 => {
                let x = x.abs();
                if x < 1e-9 {
                    return 1.0;
                }
                if x >= 3.0 {
                    return 0.0;
                }
                let p = std::f64::consts::PI * x;
                return 3.0 * p.sin() * (p / 3.0).sin() / (p * p);
            }
        };
        let x = x.abs();
        if x < 1.0 {
            ((12.0 - 9.0 * b - 6.0 * c) * x * x * x + (-18.0 + 12.0 * b + 6.0 * c) * x * x + (6.0 - 2.0 * b)) / 6.0
        } else if x < 2.0 {
            ((-b - 6.0 * c) * x * x * x + (6.0 * b + 30.0 * c) * x * x + (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)) / 6.0
        } else {
            0.0
        }
    }
}

/// Fractional bits of the tap weights.
pub const WBITS: u32 = 14;
/// Fractional bits kept between the horizontal and vertical passes.
pub const HBITS: u32 = 6;

/// Per-output-coordinate taps along one axis: output `i` reads source
/// `clamp(start[i] + k, 0, len - 1)` with weight `w[i * n + k]` (`k < n`).
#[derive(Clone, Debug, PartialEq)]
pub struct Taps {
    pub n: usize,
    pub start: Vec<i32>,
    pub w: Vec<i32>,
    /// Source length (indices clamp to it).
    pub len: u32,
}

impl Taps {
    /// Taps of `kernel` for `count` outputs, `scale` source px per output px
    /// (output `i`'s centre maps to `(i + 0.5) * scale - 0.5`), widened by the
    /// scale when reducing so the result stays alias-free.
    pub fn new(kernel: Kernel, scale: f64, len: u32, count: u32) -> Taps {
        let fs = scale.max(1.0);
        let support = kernel.radius() * fs;
        let n = (2.0 * support).ceil() as usize + 1;
        let mut start = Vec::with_capacity(count as usize);
        let mut w = vec![0i32; count as usize * n];
        let mut f = vec![0f64; n];
        for i in 0..count as usize {
            let c = (i as f64 + 0.5) * scale - 0.5;
            let s0 = (c - support).floor() as i64 + 1;
            let mut sum = 0.0;
            for (k, fk) in f.iter_mut().enumerate() {
                *fk = kernel.eval(((s0 + k as i64) as f64 - c) / fs);
                sum += *fk;
            }
            let row = &mut w[i * n..(i + 1) * n];
            if sum.abs() < 1e-9 {
                // Can't happen for these kernels; fall back to the nearest tap.
                let k = (c.round() as i64 - s0).clamp(0, n as i64 - 1) as usize;
                row[k] = 1 << WBITS;
            } else {
                let one = (1i64 << WBITS) as f64;
                let mut total = 0i32;
                let mut big = 0;
                for k in 0..n {
                    row[k] = (f[k] / sum * one).round() as i32;
                    total += row[k];
                    if f[k] > f[big] {
                        big = k;
                    }
                }
                // Weights sum to exactly 2^14 (flat areas stay exact).
                row[big] += (1 << WBITS) - total;
            }
            start.push(s0.clamp(i32::MIN as i64 / 2, i32::MAX as i64 / 2) as i32);
        }
        Taps { n, start, w, len }
    }

    /// Source index range `[lo, hi]` (clamped) read by outputs `i0..i1`.
    pub fn span(&self, i0: usize, i1: usize) -> (i32, i32) {
        let last = self.len as i32 - 1;
        let lo = self.start[i0..i1].iter().min().copied().unwrap_or(0).clamp(0, last);
        let hi = self.start[i0..i1].iter().max().copied().unwrap_or(0).saturating_add(self.n as i32 - 1).clamp(0, last);
        (lo, hi)
    }

    #[inline]
    pub fn idx(&self, i: usize, k: usize) -> usize {
        (self.start[i] + k as i32).clamp(0, self.len as i32 - 1) as usize
    }
}

/// Everything a resize needs, shared by the CPU and GPU paths.
pub struct Plan {
    pub method: Resample,
    /// Source mip level the taps index.
    pub level: u32,
    /// Output tile-area size (tables cover it: edge tiles' padding too).
    pub tw: u32,
    pub th: u32,
    /// Separable taps (filter methods), or the L taps of Preserve Details.
    pub x: Option<Taps>,
    pub y: Option<Taps>,
    /// Preserve Details: the B-spline taps and the edge gain table (8.8).
    pub sx: Option<Taps>,
    pub sy: Option<Taps>,
    pub gain: Vec<i32>,
    /// Nearest Neighbor: source column / row of every output column / row.
    pub nn: Option<(Vec<i32>, Vec<i32>)>,
}

/// Mip level for a filter resize: the one leaving a 2–4× reduction.
pub fn filter_level(layer: &Layer, nw: u32, nh: u32) -> u32 {
    let s = (layer.grid.w as f32 / nw as f32).min(layer.grid.h as f32 / nh as f32);
    if s >= 4.0 { (s.log2().floor() as u32 - 1).min(layer.mips.len() as u32) } else { 0 }
}

/// Preserve Details' gain per edge strength (0..=255), 8.8 fixed point.
pub fn gain_table(v2: bool, noise: f32) -> Vec<i32> {
    let nr = (noise / 100.0).clamp(0.0, 1.0) as f64;
    let boost = if v2 { 1.0 } else { 0.6 };
    let (lo, hi) = if v2 { (3.0, 28.0) } else { (4.0, 32.0) };
    (0..256)
        .map(|e| {
            let t = ((e as f64 - lo) / (hi - lo)).clamp(0.0, 1.0);
            let edge = t * t * (3.0 - 2.0 * t);
            let g = (1.0 - nr * (1.0 - edge)) + boost * edge;
            (g * 256.0).round() as i32
        })
        .collect()
}

impl Plan {
    /// The plan for `method` (not Bilinear, which is `ops::resize`).
    pub fn new(layer: &Layer, nw: u32, nh: u32, method: Resample, noise: f32) -> Plan {
        let g0 = &layer.grid;
        let method = method.resolve(g0.w, g0.h, nw, nh);
        let tw = nw.div_ceil(TS as u32) * TS as u32;
        let th = nh.div_ceil(TS as u32) * TS as u32;
        let mut p = Plan { method, level: 0, tw, th, x: None, y: None, sx: None, sy: None, gain: Vec::new(), nn: None };
        if method == Resample::NearestNeighbor {
            // Exact integer centres: floor((i + 0.5) * src / dst).
            let idx = |src: u32, dst: u32, count: u32| -> Vec<i32> {
                (0..count as u64).map(|i| (((2 * i + 1) * src as u64) / (2 * dst as u64)).min(src as u64 - 1) as i32).collect()
            };
            p.nn = Some((idx(g0.w, nw, tw), idx(g0.h, nh, th)));
            return p;
        }
        p.level = filter_level(layer, nw, nh);
        let g = layer.level(p.level);
        let f = (1u64 << p.level) as f64;
        let (sx, sy) = (g0.w as f64 / nw as f64 / f, g0.h as f64 / nh as f64 / f);
        match method {
            Resample::PreserveDetails | Resample::PreserveDetails2 => {
                p.x = Some(Taps::new(Kernel::Lanczos3, sx, g.w, tw));
                p.y = Some(Taps::new(Kernel::Lanczos3, sy, g.h, th));
                p.sx = Some(Taps::new(Kernel::BSpline, sx, g.w, tw));
                p.sy = Some(Taps::new(Kernel::BSpline, sy, g.h, th));
                p.gain = gain_table(method == Resample::PreserveDetails2, noise);
            }
            m => {
                let k = m.kernel().unwrap_or(Kernel::CatmullRom);
                p.x = Some(Taps::new(k, sx, g.w, tw));
                p.y = Some(Taps::new(k, sy, g.h, th));
            }
        }
        p
    }
}

#[inline]
fn chans(v: u32) -> [i32; 4] {
    [(v & 255) as i32, ((v >> 8) & 255) as i32, ((v >> 16) & 255) as i32, (v >> 24) as i32]
}

/// Final rounding of a vertical-pass sum (20 fractional bits) to a
/// premultiplied pixel.
#[inline]
pub fn finish(acc: [i32; 4]) -> u32 {
    let r = |v: i32| (v + (1 << (WBITS + HBITS - 1))) >> (WBITS + HBITS);
    let a = r(acc[3]).clamp(0, 255);
    let c = |v: i32| r(v).clamp(0, a) as u32;
    c(acc[0]) | (c(acc[1]) << 8) | (c(acc[2]) << 16) | ((a as u32) << 24)
}

/// Separable resample of output rows `y0..y1`, columns `x0..x1` (tile-area
/// coordinates) into `out` (row stride `x1 - x0`).
#[allow(clippy::too_many_arguments)]
pub fn separable(src: &TileGrid, tx: &Taps, ty: &Taps, x0: usize, x1: usize, y0: usize, y1: usize, out: &mut [u32]) {
    let w = x1 - x0;
    let (r0, r1) = ty.span(y0, y1);
    let (c0, c1) = tx.span(x0, x1);
    let (bw, bh) = ((c1 - c0 + 1) as usize, (r1 - r0 + 1) as usize);
    let mut b = vec![0u32; bw * bh];
    src.read_region(c0, r0, bw, bh, &mut b);
    // Horizontal: 6 fractional bits in i16 lanes.
    let mut hrow = vec![[0i16; 4]; w * bh];
    for r in 0..bh {
        let row = &b[r * bw..(r + 1) * bw];
        for x in 0..w {
            let i = x0 + x;
            let mut acc = [0i32; 4];
            let wt = &tx.w[i * tx.n..(i + 1) * tx.n];
            for (k, &wk) in wt.iter().enumerate() {
                if wk == 0 {
                    continue;
                }
                let p = chans(row[tx.idx(i, k) - c0 as usize]);
                for c in 0..4 {
                    acc[c] += wk * p[c];
                }
            }
            let h = &mut hrow[r * w + x];
            for c in 0..4 {
                h[c] = ((acc[c] + (1 << (WBITS - HBITS - 1))) >> (WBITS - HBITS)).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            }
        }
    }
    // Vertical: one rounding to 8 bits.
    for y in y0..y1 {
        let wt = &ty.w[y * ty.n..(y + 1) * ty.n];
        let o = &mut out[(y - y0) * w..(y - y0 + 1) * w];
        for (x, ov) in o.iter_mut().enumerate() {
            let mut acc = [0i32; 4];
            for (k, &wk) in wt.iter().enumerate() {
                if wk == 0 {
                    continue;
                }
                let h = &hrow[(ty.idx(y, k) - r0 as usize) * w + x];
                for c in 0..4 {
                    acc[c] += wk * h[c] as i32;
                }
            }
            *ov = finish(acc);
        }
    }
}

/// Luma of a premultiplied pixel for the edge measure.
#[inline]
pub fn luma(v: u32) -> i32 {
    let c = chans(v);
    (77 * c[0] + 150 * c[1] + 29 * c[2] + 128) >> 8
}

/// Preserve Details' combine for one pixel: `l`, `s` at the pixel, the
/// edge strength `e` (0..=255), and (2.0) L's 3 × 3 min / max per channel.
#[inline]
pub fn pd_pixel(l: u32, s: u32, e: usize, gain: &[i32], clamp: Option<([i32; 4], [i32; 4])>) -> u32 {
    let (lc, sc) = (chans(l), chans(s));
    let g = gain[e.min(255)];
    let mut o = [0i32; 4];
    for c in 0..4 {
        let d = lc[c] - sc[c];
        o[c] = sc[c] + ((d * g + 128) >> 8);
        if let Some((mn, mx)) = clamp {
            o[c] = o[c].clamp(mn[c], mx[c]);
        }
    }
    let a = o[3].clamp(0, 255);
    let q = |v: i32| v.clamp(0, a) as u32;
    q(o[0]) | (q(o[1]) << 8) | (q(o[2]) << 16) | ((a as u32) << 24)
}

/// Resizes `layer` to `nw x nh` with `method` on the CPU (Reduce Noise
/// `noise` in 0..=100 for the Preserve Details methods).
pub fn resize_cpu(layer: &Layer, nw: u32, nh: u32, method: Resample, noise: f32) -> TileGrid {
    if method == Resample::Bilinear {
        return super::ops::resize_bilinear_cpu(layer, nw, nh);
    }
    let p = Plan::new(layer, nw, nh, method, noise);
    if p.method == Resample::Bilinear {
        return super::ops::resize_bilinear_cpu(layer, nw, nh);
    }
    let g = layer.level(p.level);
    if let Some((cols, rows)) = &p.nn {
        let g0 = &layer.grid;
        return build_grid(nw, nh, |tx, ty| {
            let (x0, y0) = ((tx * TS_I) as usize, (ty * TS_I) as usize);
            let (c0, c1) = (cols[x0..x0 + TS].iter().min().copied().unwrap(), cols[x0..x0 + TS].iter().max().copied().unwrap());
            let (r0, r1) = (rows[y0..y0 + TS].iter().min().copied().unwrap(), rows[y0..y0 + TS].iter().max().copied().unwrap());
            let (bw, bh) = ((c1 - c0 + 1) as usize, (r1 - r0 + 1) as usize);
            let mut b = vec![0u32; bw * bh];
            g0.read_region(c0, r0, bw, bh, &mut b);
            let mut out = vec![0u32; TS * TS].into_boxed_slice();
            for y in 0..TS {
                let row = (rows[y0 + y] - r0) as usize * bw;
                for x in 0..TS {
                    out[y * TS + x] = b[row + (cols[x0 + x] - c0) as usize];
                }
            }
            finish_tile(out)
        });
    }
    let (tx_, ty_) = (p.x.as_ref().unwrap(), p.y.as_ref().unwrap());
    if !p.method.has_noise() {
        return build_grid(nw, nh, |tx, ty| {
            let (x0, y0) = ((tx * TS_I) as usize, (ty * TS_I) as usize);
            let mut out = vec![0u32; TS * TS].into_boxed_slice();
            separable(g, tx_, ty_, x0, x0 + TS, y0, y0 + TS, &mut out);
            finish_tile(out)
        });
    }
    let (sx_, sy_) = (p.sx.as_ref().unwrap(), p.sy.as_ref().unwrap());
    let v2 = p.method == Resample::PreserveDetails2;
    let (tw, th) = (p.tw as usize, p.th as usize);
    let (nwi, nhi) = (nw as i32, nh as i32);
    build_grid(nw, nh, |tx, ty| {
        let (x0, y0) = ((tx * TS_I) as usize, (ty * TS_I) as usize);
        // L and S with a one-pixel apron (neighbours clamp to the image).
        let (ax0, ax1) = (x0.saturating_sub(1), (x0 + TS + 1).min(tw));
        let (ay0, ay1) = (y0.saturating_sub(1), (y0 + TS + 1).min(th));
        let aw = ax1 - ax0;
        let mut l = vec![0u32; aw * (ay1 - ay0)];
        let mut s = vec![0u32; aw * (ay1 - ay0)];
        separable(g, tx_, ty_, ax0, ax1, ay0, ay1, &mut l);
        separable(g, sx_, sy_, ax0, ax1, ay0, ay1, &mut s);
        let at = |b: &[u32], x: usize, y: usize| b[(y - ay0) * aw + x - ax0];
        let nb = |v: usize, d: i32, n: i32| if d == 0 { v } else { (v as i32 + d).clamp(0, n - 1) as usize };
        let mut out = vec![0u32; TS * TS].into_boxed_slice();
        for y in y0..y0 + TS {
            for x in x0..x0 + TS {
                let e = (luma(at(&s, nb(x, 1, nwi), y)) - luma(at(&s, nb(x, -1, nwi), y))).abs()
                    + (luma(at(&s, x, nb(y, 1, nhi))) - luma(at(&s, x, nb(y, -1, nhi)))).abs();
                let clamp = v2.then(|| {
                    let (mut mn, mut mx) = ([255i32; 4], [0i32; 4]);
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let c = chans(at(&l, nb(x, dx, nwi), nb(y, dy, nhi)));
                            for k in 0..4 {
                                mn[k] = mn[k].min(c[k]);
                                mx[k] = mx[k].max(c[k]);
                            }
                        }
                    }
                    (mn, mx)
                });
                out[(y - y0) * TS + x - x0] = pd_pixel(at(&l, x, y), at(&s, x, y), e as usize, &p.gain, clamp);
            }
        }
        finish_tile(out)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ops, px};

    fn layer(w: u32, h: u32) -> Layer {
        Layer::new(1, "t", ops::procedural(w, h))
    }

    fn from_fn(w: u32, h: u32, f: impl Fn(i32, i32) -> u32 + Sync) -> TileGrid {
        build_grid(w, h, |tx, ty| {
            let px: Vec<u32> = (0..TS * TS).map(|i| f(tx * TS_I + (i % TS) as i32, ty * TS_I + (i / TS) as i32)).collect();
            finish_tile(px.into_boxed_slice())
        })
    }

    #[test]
    fn taps_sum_to_one_and_flat_stays_flat() {
        for k in [Kernel::CatmullRom, Kernel::Sharper, Kernel::Mitchell, Kernel::BSpline, Kernel::Lanczos3] {
            for (scale, len, count) in [(0.37, 300u32, 811u32), (1.0, 64, 64), (2.7, 1000, 371), (3.99, 999, 256)] {
                let t = Taps::new(k, scale, len, count);
                for i in 0..count as usize {
                    assert_eq!(t.w[i * t.n..(i + 1) * t.n].iter().sum::<i32>(), 1 << WBITS, "{k:?} {scale} {i}");
                }
            }
        }
        let flat = Layer::new(1, "f", TileGrid::filled(301, 203, px::pack(200, 100, 50, 255)));
        for m in Resample::ALL {
            let o = resize_cpu(&flat, 777, 95, m, 50.0);
            for (x, y) in [(0, 0), (776, 94), (400, 50), (123, 7)] {
                assert_eq!(o.pixel(x, y), px::pack(200, 100, 50, 255), "{m:?} at ({x}, {y})");
            }
        }
    }

    #[test]
    fn bilinear_is_the_original_resize() {
        let l = layer(1531, 1021);
        for (w, h) in [(700, 333), (2000, 1500), (191, 1500)] {
            let a = resize_cpu(&l, w, h, Resample::Bilinear, 0.0);
            let b = ops::resize_bilinear_cpu(&l, w, h);
            assert!(a.tiles.iter().zip(&b.tiles).all(|(x, y)| x.as_ref().map(|t| t.read().to_vec()) == y.as_ref().map(|t| t.read().to_vec())));
        }
    }

    #[test]
    fn nearest_neighbor_copies_source_pixels() {
        let l = layer(300, 200);
        let o = resize_cpu(&l, 600, 400, Resample::NearestNeighbor, 0.0);
        for (x, y) in [(0, 0), (1, 1), (599, 399), (301, 177)] {
            assert_eq!(o.pixel(x, y), l.grid.pixel(x / 2, y / 2));
        }
    }

    #[test]
    fn methods_order_by_sharpness() {
        // A step enlarged 4×: Smoother is softer than Bicubic; Preserve
        // Details is steeper than its Lanczos base.
        let g = from_fn(64, 16, |x, _| {
            let v = if x < 32 { 40 } else { 220 };
            px::pack(v as u8, v as u8, v as u8, 255)
        });
        let l = Layer::new(1, "s", g);
        let slope = |m: Resample| {
            let o = resize_cpu(&l, 256, 64, m, 0.0);
            (100..160).map(|x| px::unpack(o.pixel(x + 1, 30))[0] as i32 - px::unpack(o.pixel(x, 30))[0] as i32).max().unwrap()
        };
        let peak = |m: Resample| {
            let o = resize_cpu(&l, 256, 64, m, 0.0);
            (100..160).map(|x| px::unpack(o.pixel(x, 30))[0]).max().unwrap()
        };
        let (sm, bc, lz, pd) = (slope(Resample::BicubicSmoother), slope(Resample::Bicubic), slope(Resample::Lanczos), slope(Resample::PreserveDetails));
        assert!(sm < bc && pd > lz, "{sm} {bc} {lz} {pd}");
        // Sharper rings more (it is the sharpening kernel), Smoother least.
        let (psm, pbc, psh) = (peak(Resample::BicubicSmoother), peak(Resample::Bicubic), peak(Resample::BicubicSharper));
        assert!(pbc > psm && psh > pbc, "{psm} {pbc} {psh}");
    }

    #[test]
    fn reduce_noise_smooths_flat_areas() {
        let g = from_fn(128, 128, |x, y| {
            let n = (crate::engine::filters::util::hash01(x, y, 7) * 12.0) as u8;
            px::pack(120 + n, 120 + n, 120 + n, 255)
        });
        let l = Layer::new(1, "n", g);
        let var = |o: &TileGrid| {
            let v: Vec<f64> = (0..256).flat_map(|y| (0..256).map(move |x| (x, y))).map(|(x, y)| px::unpack(o.pixel(x, y))[0] as f64).collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / v.len() as f64
        };
        let a = var(&resize_cpu(&l, 256, 256, Resample::PreserveDetails, 0.0));
        let b = var(&resize_cpu(&l, 256, 256, Resample::PreserveDetails, 100.0));
        assert!(b < a * 0.8, "{a} {b}");
    }
}
