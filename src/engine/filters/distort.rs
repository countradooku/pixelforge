//! Geometric filters (Distort menu, Offset, Radial Blur). These map each output
//! pixel to source positions in level-0 image space, so they can't run on a
//! fixed padded window. A bounding-box prefetch turns most tiles into a
//! single contiguous read; far-reaching ones fall back to per-sample tile
//! lookups. Sampling is bilinear with edge repeat.

use std::f32::consts::{PI, TAU};

use crate::engine::grid::TileGrid;
use crate::engine::px;

#[derive(Clone, Debug, PartialEq)]
pub enum Warp {
    Twirl { angle: f32, cx: f32, cy: f32, radius: f32 },
    Pinch { amount: f32, cx: f32, cy: f32, radius: f32 },
    Spherize { amount: f32, cx: f32, cy: f32, radius: f32 },
    ZigZag { amount: f32, ridges: f32, cx: f32, cy: f32, radius: f32 },
    Polar { to_polar: bool, w: f32, h: f32 },
    Ripple { amount: f32, size: f32 },
    Wave { wavelength: f32, amplitude: f32 },
    Offset { dx: f32, dy: f32, w: f32, h: f32 },
    Shear { amount: f32, h: f32 },
    /// Projective map: source = H · output (homogeneous); outside → transparent.
    Homography { m: [f32; 9] },
    /// Fisheye (equidistant) to rectilinear correction around the center.
    WideAngle { strength: f32, scale: f32, cx: f32, cy: f32, diag: f32 },
    /// Edit ▸ Transform ▸ Warp: Warp Text's Bézier patch (`m`, box-relative,
    /// see `warp::invert`) applied to the pixels of the box `(x0, y0, w, h)`;
    /// outside the warped box → transparent.
    /// `nodes`: Newton's start points (`warp::start_nodes(m)`).
    Mesh { m: [f32; 32], nodes: [f32; 50], x0: f32, y0: f32, w: f32, h: f32 },
}

/// 3x3 matrix helpers for homographies.
pub fn mat_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut o = [0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            o[r * 3 + c] = (0..3).map(|k| a[r * 3 + k] * b[k * 3 + c]).sum();
        }
    }
    o
}

pub fn mat_inv(m: &[f32; 9]) -> [f32; 9] {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    let k = if det.abs() < 1e-12 { 0.0 } else { 1.0 / det };
    [
        (e * i - f * h) * k,
        (c * h - b * i) * k,
        (b * f - c * e) * k,
        (f * g - d * i) * k,
        (a * i - c * g) * k,
        (c * d - a * f) * k,
        (d * h - e * g) * k,
        (b * g - a * h) * k,
        (a * e - b * d) * k,
    ]
}

/// Homography mapping the unit square corners to quad `q` (TL, TR, BR, BL).
pub fn square_to_quad(q: [(f32, f32); 4]) -> [f32; 9] {
    let [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = q;
    let (dx1, dy1, dx2, dy2) = (x1 - x2, y1 - y2, x3 - x2, y3 - y2);
    let (sx, sy) = (x0 - x1 + x2 - x3, y0 - y1 + y2 - y3);
    let det = dx1 * dy2 - dx2 * dy1;
    let (g, h) = if det.abs() < 1e-9 { (0.0, 0.0) } else { ((sx * dy2 - dx2 * sy) / det, (dx1 * sy - sx * dy1) / det) };
    [x1 - x0 + g * x1, x3 - x0 + h * x3, x0, y1 - y0 + g * y1, y3 - y0 + h * y3, y0, g, h, 1.0]
}

impl Warp {
    /// Source position (level-0) for output position `(x, y)` (level-0).
    #[inline]
    pub fn map(&self, x: f32, y: f32) -> (f32, f32) {
        match *self {
            Warp::Twirl { angle, cx, cy, radius } => {
                let (dx, dy) = (x - cx, y - cy);
                let r = (dx * dx + dy * dy).sqrt();
                if r >= radius {
                    return (x, y);
                }
                let t = 1.0 - r / radius;
                let a = angle.to_radians() * t * t;
                let (s, c) = a.sin_cos();
                (cx + dx * c - dy * s, cy + dx * s + dy * c)
            }
            Warp::Pinch { amount, cx, cy, radius } | Warp::Spherize { amount, cx, cy, radius } => {
                let (dx, dy) = (x - cx, y - cy);
                let r = (dx * dx + dy * dy).sqrt();
                if r >= radius || r == 0.0 {
                    return (x, y);
                }
                let rn = r / radius;
                let k = match self {
                    // Pinch > 0 pulls the content toward the center.
                    Warp::Pinch { .. } => rn.powf(-amount / 100.0 * 0.5),
                    // Spherize > 0 bulges outward like a sphere.
                    _ => {
                        let a = amount / 100.0;
                        let sph = (2.0 / PI) * rn.asin() / rn.max(1e-6);
                        1.0 + (sph - 1.0) * a
                    }
                };
                (cx + dx * k, cy + dy * k)
            }
            Warp::ZigZag { amount, ridges, cx, cy, radius } => {
                let (dx, dy) = (x - cx, y - cy);
                let r = (dx * dx + dy * dy).sqrt();
                if r >= radius || r == 0.0 {
                    return (x, y);
                }
                let rn = r / radius;
                let d = (amount / 100.0) * radius * 0.1 * (rn * ridges * TAU).sin() * (1.0 - rn);
                let k = (r + d) / r;
                (cx + dx * k, cy + dy * k)
            }
            Warp::Polar { to_polar, w, h } => {
                let (cx, cy) = (w * 0.5, h * 0.5);
                if to_polar {
                    // Output is polar: angle -> source x, radius -> source y.
                    let (dx, dy) = (x - cx, y - cy);
                    let ang = dy.atan2(dx) + PI * 0.5;
                    let r = (dx * dx + dy * dy).sqrt() / (w.min(h) * 0.5);
                    (ang.rem_euclid(TAU) / TAU * w, r * h)
                } else {
                    let ang = x / w * TAU - PI * 0.5;
                    let r = y / h * w.min(h) * 0.5;
                    (cx + r * ang.cos(), cy + r * ang.sin())
                }
            }
            Warp::Ripple { amount, size } => {
                let a = amount / 100.0 * size * 0.25;
                (x + a * (y / size * TAU).sin(), y + a * (x / size * TAU).sin())
            }
            Warp::Wave { wavelength, amplitude } => {
                let f = TAU / wavelength.max(1.0);
                (
                    x + amplitude * (y * f).sin() + amplitude * 0.5 * (y * f * 2.3 + 1.7).sin(),
                    y + amplitude * (x * f).sin() + amplitude * 0.5 * (x * f * 1.9 + 0.4).sin(),
                )
            }
            Warp::Offset { dx, dy, w, h } => ((x - dx).rem_euclid(w), (y - dy).rem_euclid(h)),
            Warp::Shear { amount, h } => {
                let t = y / h;
                (x - amount * (t * PI).sin(), y)
            }
            Warp::Homography { m } => {
                let wq = m[6] * x + m[7] * y + m[8];
                if wq.abs() < 1e-9 {
                    return (f32::NAN, f32::NAN);
                }
                ((m[0] * x + m[1] * y + m[2]) / wq, (m[3] * x + m[4] * y + m[5]) / wq)
            }
            Warp::WideAngle { strength, scale, cx, cy, diag } => {
                let (dx, dy) = (x - cx, y - cy);
                let r = (dx * dx + dy * dy).sqrt();
                if r < 1e-3 {
                    return (x, y);
                }
                // Rectilinear output radius r -> fisheye source radius.
                let f = diag / (1.0 + strength / 25.0).max(0.05);
                let rn = r / (scale / 100.0);
                let src = f * (rn / f).atan();
                let k = 1.0 + (src / r - 1.0) * (strength / 100.0).clamp(0.0, 1.0);
                (cx + dx * k, cy + dy * k)
            }
            Warp::Mesh { ref m, ref nodes, x0, y0, w, h } => match crate::engine::warp::invert_from(m, nodes, x - x0, y - y0) {
                Some((u, v)) => (x0 + u * w, y0 + v * h),
                None => (-1e30, -1e30),
            },
        }
    }

    /// Whether samples outside the image read as transparent (transforms).
    pub fn transparent_outside(&self) -> bool {
        matches!(self, Warp::Homography { .. } | Warp::Mesh { .. })
    }

    /// Whether sampling wraps around the image (Offset).
    pub fn wraps(&self) -> bool {
        matches!(self, Warp::Offset { .. })
    }
}

/// Bilinear sampler over a level grid with optional prefetched window.
pub struct Sampler<'a> {
    grid: &'a TileGrid,
    win: Option<(Vec<u32>, i32, i32, usize, usize)>,
    wrap: bool,
    // Four neighboring source tiles stay pinned for this output tile only.
    // Sampling never holds an entire document resident under memory pressure.
    cache: std::cell::RefCell<[Option<(usize, crate::engine::memory::Pixels<'a>)>; 4]>,
}

impl<'a> Sampler<'a> {
    pub fn new(grid: &'a TileGrid, bbox: Option<(f32, f32, f32, f32)>, out_px: usize, wrap: bool) -> Self {
        let win = bbox.and_then(|(x0, y0, x1, y1)| {
            if wrap {
                return None;
            }
            let gx0 = (x0.floor() as i32 - 1).max(0);
            let gy0 = (y0.floor() as i32 - 1).max(0);
            let gx1 = (x1.ceil() as i32 + 2).min(grid.w as i32);
            let gy1 = (y1.ceil() as i32 + 2).min(grid.h as i32);
            if gx1 <= gx0 || gy1 <= gy0 {
                return None;
            }
            let (w, h) = ((gx1 - gx0) as usize, (gy1 - gy0) as usize);
            if w * h > out_px.max(1) * 24 {
                return None;
            }
            let mut b = vec![0u32; w * h];
            grid.read_region(gx0, gy0, w, h, &mut b);
            Some((b, gx0, gy0, w, h))
        });
        Sampler { grid, win, wrap, cache: std::cell::RefCell::new(std::array::from_fn(|_| None)) }
    }

    #[inline]
    pub(crate) fn px(&self, x: i32, y: i32) -> u32 {
        let (gw, gh) = (self.grid.w as i32, self.grid.h as i32);
        let (x, y) = if self.wrap { (x.rem_euclid(gw), y.rem_euclid(gh)) } else { (x.clamp(0, gw - 1), y.clamp(0, gh - 1)) };
        if let Some((b, x0, y0, w, h)) = &self.win {
            let (lx, ly) = (x - x0, y - y0);
            if lx >= 0 && ly >= 0 && (lx as usize) < *w && (ly as usize) < *h {
                return b[ly as usize * w + lx as usize];
            }
        }
        let (tx, ty) = (x as usize >> 8, y as usize >> 8);
        let idx = ty * self.grid.cols as usize + tx;
        let Some(tile) = &self.grid.tiles[idx] else { return 0 };
        if let Some(c) = tile.uniform { return c; }
        let mut cache = self.cache.borrow_mut();
        let slot = &mut cache[(tx & 1) | ((ty & 1) << 1)];
        if slot.as_ref().is_none_or(|(i, _)| *i != idx) {
            *slot = None;
            *slot = Some((idx, tile.read()));
        }
        slot.as_ref().unwrap().1[((y as usize & 255) << 8) | (x as usize & 255)]
    }

    /// Four adjacent taps. Interior samples need one cache lookup / pin.
    pub(crate) fn quad(&self, x: i32, y: i32) -> [u32; 4] {
        if let Some((b, x0, y0, w, h)) = &self.win {
            let (lx, ly) = (x - x0, y - y0);
            if lx >= 0 && ly >= 0 && (lx as usize + 1) < *w && (ly as usize + 1) < *h {
                let i = ly as usize * w + lx as usize;
                return [b[i], b[i + 1], b[i + w], b[i + w + 1]];
            }
        }
        let (x, y) = if self.wrap { (x.rem_euclid(self.grid.w as i32), y.rem_euclid(self.grid.h as i32)) } else { (x, y) };
        if self.win.is_none() && x >= 0 && y >= 0
            && x + 1 < self.grid.w as i32 && y + 1 < self.grid.h as i32
            && (x & 255) != 255 && (y & 255) != 255 {
            let (tx, ty) = (x as usize >> 8, y as usize >> 8);
            let idx = ty * self.grid.cols as usize + tx;
            let Some(tile) = &self.grid.tiles[idx] else { return [0; 4] };
            if let Some(c) = tile.uniform { return [c; 4]; }
            let mut cache = self.cache.borrow_mut();
            let slot = &mut cache[(tx & 1) | ((ty & 1) << 1)];
            if slot.as_ref().is_none_or(|(i, _)| *i != idx) {
                *slot = None;
                *slot = Some((idx, tile.read()));
            }
            let p: &[u32] = &slot.as_ref().unwrap().1;
            let i = ((y as usize & 255) << 8) | (x as usize & 255);
            [p[i], p[i + 1], p[i + 256], p[i + 257]]
        } else {
            [self.px(x, y), self.px(x + 1, y), self.px(x, y + 1), self.px(x + 1, y + 1)]
        }
    }

    /// Bilinear sample at level coordinates (pixel centers at +0.5).
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> u32 {
        let x = x - 0.5;
        let y = y - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = ((x - x0) * 255.0) as u32;
        let fy = ((y - y0) * 255.0) as u32;
        let (xi, yi) = (x0 as i32, y0 as i32);
        let [a, b, c, d] = self.quad(xi, yi);
        let top = px::lerp(a, b, fx);
        let bot = px::lerp(c, d, fx);
        px::lerp(top, bot, fy)
    }
}

/// Renders a warp into `out` (`w x h` at `(x0, y0)` of `level`).
pub fn render_warp(warp: &Warp, grid: &TileGrid, level: u32, x0: i32, y0: i32, w: usize, h: usize, out: &mut [u32]) {
    let s = (1u32 << level) as f32;
    let mut coords = Vec::with_capacity(w * h);
    let mut bb = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for y in 0..h {
        for x in 0..w {
            let gx = (x0 as f32 + x as f32 + 0.5) * s;
            let gy = (y0 as f32 + y as f32 + 0.5) * s;
            let (sx, sy) = warp.map(gx, gy);
            let (lx, ly) = (sx / s, sy / s);
            bb = (bb.0.min(lx), bb.1.min(ly), bb.2.max(lx), bb.3.max(ly));
            coords.push((lx, ly));
        }
    }
    let bb = if bb.0.is_finite() && bb.2.is_finite() { Some(bb) } else { None };
    let sampler = Sampler::new(grid, bb, w * h, warp.wraps());
    let clip = warp.transparent_outside();
    let (gw, gh) = (grid.w as f32, grid.h as f32);
    for (o, &(lx, ly)) in out.iter_mut().zip(&coords) {
        *o = if clip && !(lx >= 0.0 && ly >= 0.0 && lx < gw && ly < gh) { 0 } else { sampler.sample(lx, ly) };
    }
}

/// Blur along a curved path (Blur Gallery ▸ Path Blur): each pixel averages
/// samples along the local tangent of a quadratic curve P0→P1 bent by
/// `bend`, read from the mip level matching the sample spacing.
#[allow(clippy::too_many_arguments)]
pub fn render_path_blur(
    p0: (f32, f32),
    p1: (f32, f32),
    bend: f32,
    length: f32,
    src: &super::Src,
    level: u32,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    out: &mut [u32],
) {
    render_path_blur_with(p0, p1, bend, length, src, level, x0, y0, w, h, out, |_, _, acc, n| avg_px(acc, n))
}

/// The rounded average of a gather's samples.
#[inline]
pub fn avg_px(acc: [u32; 4], n: u32) -> u32 {
    px::pack(((acc[0] + n / 2) / n) as u8, ((acc[1] + n / 2) / n) as u8, ((acc[2] + n / 2) / n) as u8, ((acc[3] + n / 2) / n) as u8)
}

/// Path Blur's per-pixel tap count (`n`) and mip offset (`m`) for a path
/// `length` (level 0) at `level`, with `extra` usable mip levels above it.
pub fn path_taps(length: f32, level: u32, extra: u32) -> (usize, u32, f32) {
    const TAPS: f32 = 16.0;
    let s = (1u32 << level) as f32;
    let len_l = length / s;
    let n = len_l.ceil().clamp(1.0, TAPS) as usize;
    let spacing = len_l / n as f32;
    let m = if spacing > 1.5 { (spacing.log2().floor() as u32).min(extra) } else { 0 };
    (n, m, len_l)
}

/// [`render_path_blur`] finishing each pixel with `fin(x, y, sample sum, taps)`.
#[allow(clippy::too_many_arguments)]
pub fn render_path_blur_with(
    p0: (f32, f32),
    p1: (f32, f32),
    bend: f32,
    length: f32,
    src: &super::Src,
    level: u32,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    out: &mut [u32],
    fin: impl Fn(usize, usize, [u32; 4], u32) -> u32,
) {
    let s = (1u32 << level) as f32;
    let (mx, my) = ((p0.0 + p1.0) * 0.5, (p0.1 + p1.1) * 0.5);
    let (cx, cy) = (p1.0 - p0.0, p1.1 - p0.1);
    let cl = (cx * cx + cy * cy).sqrt().max(1.0);
    let ctrl = (mx - cy / cl * bend, my + cx / cl * bend);
    let tangent = |t: f32| {
        // d/dt of the quadratic Bezier.
        let tx = 2.0 * (1.0 - t) * (ctrl.0 - p0.0) + 2.0 * t * (p1.0 - ctrl.0);
        let ty = 2.0 * (1.0 - t) * (ctrl.1 - p0.1) + 2.0 * t * (p1.1 - ctrl.1);
        let l = (tx * tx + ty * ty).sqrt().max(1e-6);
        (tx / l, ty / l)
    };
    let extra = src.max_level().saturating_sub(level).min(6);
    let (n, m, len_l) = path_taps(length, level, extra);
    let k = (1u32 << m) as f32;
    let pad = len_l * 0.5 + 2.0;
    let bb = ((x0 as f32 - pad) / k, (y0 as f32 - pad) / k, ((x0 + w as i32) as f32 + pad) / k, ((y0 + h as i32) as f32 + pad) / k);
    let sampler = Sampler::new(src.at(level + m), Some(bb), (w * h) >> (2 * m).min(30), false);
    for y in 0..h {
        for x in 0..w {
            let (px_, py_) = (x0 as f32 + x as f32 + 0.5, y0 as f32 + y as f32 + 0.5);
            let (gx, gy) = (px_ * s, py_ * s);
            let t = (((gx - p0.0) * cx + (gy - p0.1) * cy) / (cl * cl)).clamp(0.0, 1.0);
            let (dx, dy) = tangent(t);
            let mut acc = [0u32; 4];
            for i in 0..n {
                let o = ((i as f32 + 0.5) / n as f32 - 0.5) * len_l;
                let p = px::unpack(sampler.sample((px_ + dx * o) / k, (py_ + dy * o) / k));
                for c in 0..4 {
                    acc[c] += p[c] as u32;
                }
            }
            out[y * w + x] = fin(x, y, acc, n as u32);
        }
    }
}

/// Radial blur: spin (arc) or zoom (ray) average of bilinear samples.
///
/// Long paths are not brute-forced: samples are spaced along the path and
/// each one is read from the mip level whose pixel size matches the spacing,
/// so a 300 px arc costs ~24 prefiltered taps instead of hundreds of raw
/// ones, with equivalent smoothness.
#[allow(clippy::too_many_arguments)]
pub fn render_radial(
    amount: f32,
    zoom: bool,
    cx: f32,
    cy: f32,
    src: &super::Src,
    level: u32,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    out: &mut [u32],
) {
    render_radial_with(amount, zoom, cx, cy, src, level, x0, y0, w, h, out, |_, _, acc, n| avg_px(acc, n))
}

/// Most taps of a radial (spin / zoom) path.
pub const RADIAL_MAX_TAPS: f32 = 24.0;

/// [`render_radial`] finishing each pixel with `fin(x, y, sample sum, taps)`.
#[allow(clippy::too_many_arguments)]
pub fn render_radial_with(
    amount: f32,
    zoom: bool,
    cx: f32,
    cy: f32,
    src: &super::Src,
    level: u32,
    x0: i32,
    y0: i32,
    w: usize,
    h: usize,
    out: &mut [u32],
    fin: impl Fn(usize, usize, [u32; 4], u32) -> u32,
) {
    const MAX_TAPS: f32 = RADIAL_MAX_TAPS;
    let s = (1u32 << level) as f32;
    let (lcx, lcy) = (cx / s, cy / s);
    let extra = (src.max_level().saturating_sub(level)).min(6);
    // One sampler per usable mip offset, each prefetching the region the
    // block's paths can reach at that level.
    let span = (amount / 100.0 * 30.0).to_radians();
    // Tight bounds of every sample the block can take: the block itself plus
    // (zoom) the block pulled toward the center, or (spin) the block's
    // corners rotated by ±span/2, padded by the arc's sagitta.
    let corners = [
        (x0 as f32, y0 as f32),
        ((x0 + w as i32) as f32, y0 as f32),
        (x0 as f32, (y0 + h as i32) as f32),
        ((x0 + w as i32) as f32, (y0 + h as i32) as f32),
    ];
    let mut bb = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    let mut grow = |x: f32, y: f32| bb = (bb.0.min(x), bb.1.min(y), bb.2.max(x), bb.3.max(y));
    let mut rmax = 0f32;
    for &(cx_, cy_) in &corners {
        let (dx, dy) = (cx_ - lcx, cy_ - lcy);
        rmax = rmax.max((dx * dx + dy * dy).sqrt());
        if zoom {
            let t = 1.0 - amount / 200.0;
            grow(cx_, cy_);
            grow(lcx + dx * t, lcy + dy * t);
        } else {
            for a in [-span * 0.5, 0.0, span * 0.5] {
                let (sn, cs) = a.sin_cos();
                grow(lcx + dx * cs - dy * sn, lcy + dx * sn + dy * cs);
            }
        }
    }
    let sag = if zoom { 2.0 } else { rmax * (1.0 - (span * 0.5).cos()) + 2.0 };
    let (bx0, by0, bx1, by1) = (bb.0 - sag, bb.1 - sag, bb.2 + sag, bb.3 + sag);
    let samplers: Vec<Sampler> = (0..=extra)
        .map(|m| {
            let k = (1u32 << m) as f32;
            Sampler::new(src.at(level + m), Some((bx0 / k, by0 / k, bx1 / k, by1 / k)), ((w * h) >> (2 * m)).max(1) * 4, false)
        })
        .collect();
    for y in 0..h {
        for x in 0..w {
            let px_ = x0 as f32 + x as f32 + 0.5;
            let py_ = y0 as f32 + y as f32 + 0.5;
            let (dx, dy) = (px_ - lcx, py_ - lcy);
            let r = (dx * dx + dy * dy).sqrt();
            let len = if zoom { r * amount / 100.0 * 0.5 } else { r * span };
            let n = len.ceil().clamp(1.0, MAX_TAPS) as usize;
            let spacing = len / n as f32;
            let m = if spacing > 1.5 { (spacing.log2().floor() as u32).min(extra) } else { 0 };
            let sm = &samplers[m as usize];
            let k = 1.0 / (1u32 << m) as f32;
            let mut acc = [0u32; 4];
            if zoom {
                for i in 0..n {
                    let t = 1.0 - ((i as f32 + 0.5) / n as f32) * (amount / 100.0 * 0.5);
                    let p = px::unpack(sm.sample((lcx + dx * t) * k, (lcy + dy * t) * k));
                    for c in 0..4 {
                        acc[c] += p[c] as u32;
                    }
                }
            } else {
                // Rotate (dx, dy) by -span/2 + half a step, then step by a
                // fixed rotation: no trig per tap.
                let step = span / n as f32;
                let (s0, c0) = (-span * 0.5 + step * 0.5).sin_cos();
                let (ss, cs) = step.sin_cos();
                let (mut vx, mut vy) = (dx * c0 - dy * s0, dx * s0 + dy * c0);
                for _ in 0..n {
                    let p = px::unpack(sm.sample((lcx + vx) * k, (lcy + vy) * k));
                    for c in 0..4 {
                        acc[c] += p[c] as u32;
                    }
                    (vx, vy) = (vx * cs - vy * ss, vx * ss + vy * cs);
                }
            }
            out[y * w + x] = fin(x, y, acc, n as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_sampling_matches_grid_across_tile_and_document_edges() {
        for (w, h) in [(1, 1), (257, 259), (600, 510)] {
            let mut grid = crate::engine::ops::procedural(w, h);
            if grid.tiles.len() > 3 {
                grid.tiles[1] = None;
                grid.tiles[2] = Some(std::sync::Arc::new(crate::engine::grid::Tile::solid(0xff123456)));
            }
            for wrap in [false, true] {
                for bbox in [None, Some((0., 0., w as f32, h as f32))] {
                    let sampler = Sampler::new(&grid, bbox, (w * h) as usize, wrap);
                    let mut seed = 0xabcde123u32;
                    for n in 0..4000 {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        let x = if n < 260 { n - 2 } else { (seed % (w * 3)) as i32 - w as i32 };
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        let y = if n < 260 { n - 2 } else { (seed % (h * 3)) as i32 - h as i32 };
                        let reference = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)].map(|(x, y)| {
                            let (x, y) = if wrap { (x.rem_euclid(w as i32), y.rem_euclid(h as i32)) }
                                else { (x.clamp(0, w as i32 - 1), y.clamp(0, h as i32 - 1)) };
                            grid.pixel(x, y)
                        });
                        assert_eq!(sampler.quad(x, y), reference, "{w}x{h} ({x}, {y}) wrap={wrap}");
                    }
                }
            }
        }
    }

    #[test]
    fn twirl_is_identity_outside_radius() {
        let w = Warp::Twirl { angle: 90.0, cx: 50.0, cy: 50.0, radius: 10.0 };
        assert_eq!(w.map(80.0, 80.0), (80.0, 80.0));
        let (x, y) = w.map(50.0, 50.0);
        assert!((x - 50.0).abs() < 1e-3 && (y - 50.0).abs() < 1e-3);
    }

    #[test]
    fn polar_roundtrip_center() {
        let p = Warp::Polar { to_polar: false, w: 200.0, h: 100.0 };
        let (x, y) = p.map(0.0, 0.0);
        assert!((x - 100.0).abs() < 1e-3 && (y - 50.0).abs() < 1e-3);
    }
}
