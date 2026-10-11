//! Blur kernels. All run in O(1) per pixel regardless of radius except the
//! lens (disk) blur, which is O(r) via per-row prefix sums.

use super::util::{hash01, merge, opaque, split};

/// Box sizes whose triple convolution approximates a Gaussian of `sigma`.
pub fn boxes_for_gauss(sigma: f32) -> [usize; 3] {
    if sigma < 0.3 {
        return [1, 1, 1];
    }
    let n = 3.0f32;
    let w_ideal = (12.0 * sigma * sigma / n + 1.0).sqrt();
    let mut wl = w_ideal.floor() as i32;
    if wl % 2 == 0 {
        wl -= 1;
    }
    let wl = wl.max(1);
    let wu = wl + 2;
    let wlf = wl as f32;
    let m_ideal = (12.0 * sigma * sigma - n * wlf * wlf - 4.0 * n * wlf - 3.0 * n) / (-4.0 * wlf - 4.0);
    let m = m_ideal.round() as i32;
    let mut out = [0usize; 3];
    for (i, o) in out.iter_mut().enumerate() {
        *o = if (i as i32) < m { wl as usize } else { wu as usize };
    }
    out
}

pub fn gauss_pad(sigma: f32) -> usize {
    if sigma < 0.3 {
        0
    } else if sigma <= super::fast::DIRECT_SIGMA_MAX {
        (sigma * 3.0).ceil() as usize
    } else {
        boxes_for_gauss(sigma).iter().map(|b| b / 2).sum()
    }
}

/// Separable running-sum box passes on packed premultiplied pixels.
pub fn box_passes(buf: &mut [u32], w: usize, h: usize, boxes: &[usize]) {
    if boxes.iter().all(|&b| b <= 1) {
        return;
    }
    let pad: usize = boxes.iter().map(|b| b / 2).sum();
    let mut tmp = vec![0u32; w.max(h)];
    for &b in boxes {
        let r = b / 2;
        if r == 0 {
            continue;
        }
        for y in 0..h {
            box_line(&mut buf[y * w..(y + 1) * w], &mut tmp[..w], r);
        }
    }
    let cx0 = pad.min(w);
    let cx1 = w.saturating_sub(pad).max(cx0);
    let mut sums = vec![[0u32; 4]; cx1 - cx0];
    let mut col_out = vec![0u32; (cx1 - cx0) * h];
    for &b in boxes {
        let r = b / 2;
        if r == 0 {
            continue;
        }
        box_vertical(buf, w, h, cx0, cx1, r, &mut sums, &mut col_out);
    }
}

/// Gaussian blur: direct fixed-point convolution for small sigma, three
/// running-sum box passes (O(1) per pixel) for large sigma.
pub fn gaussian(buf: &mut [u32], w: usize, h: usize, sigma: f32) {
    if sigma < 0.3 {
        return;
    }
    if sigma <= super::fast::DIRECT_SIGMA_MAX {
        super::fast::conv_sep(buf, w, h, &super::fast::gauss_kernel(sigma));
    } else {
        box_passes(buf, w, h, &boxes_for_gauss(sigma));
    }
}

#[inline(always)]
fn add4(s: &mut [u32; 4], p: u32) {
    s[0] += p & 0xFF;
    s[1] += (p >> 8) & 0xFF;
    s[2] += (p >> 16) & 0xFF;
    s[3] += p >> 24;
}

#[inline(always)]
fn sub4(s: &mut [u32; 4], p: u32) {
    s[0] -= p & 0xFF;
    s[1] -= (p >> 8) & 0xFF;
    s[2] -= (p >> 16) & 0xFF;
    s[3] -= p >> 24;
}

#[inline(always)]
fn div4(s: &[u32; 4], inv: u64) -> u32 {
    let d = |v: u32| ((v as u64 * inv + (1 << 23)) >> 24) as u32;
    d(s[0]) | d(s[1]) << 8 | d(s[2]) << 16 | d(s[3]) << 24
}

pub(crate) fn box_line(line: &mut [u32], tmp: &mut [u32], r: usize) {
    let n = line.len();
    if n == 0 {
        return;
    }
    let inv = (1u64 << 24) / (2 * r as u64 + 1);
    let at = |i: isize| line[i.clamp(0, n as isize - 1) as usize];
    let mut s = [0u32; 4];
    for i in -(r as isize)..=(r as isize) {
        add4(&mut s, at(i));
    }
    for (i, t) in tmp.iter_mut().enumerate().take(n) {
        *t = div4(&s, inv);
        add4(&mut s, at(i as isize + r as isize + 1));
        sub4(&mut s, at(i as isize - r as isize));
    }
    line.copy_from_slice(&tmp[..n]);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn box_vertical(
    buf: &mut [u32],
    w: usize,
    h: usize,
    cx0: usize,
    cx1: usize,
    r: usize,
    sums: &mut [[u32; 4]],
    out: &mut [u32],
) {
    let cw = cx1 - cx0;
    if cw == 0 || h == 0 {
        return;
    }
    let inv = (1u64 << 24) / (2 * r as u64 + 1);
    let row = |y: isize| y.clamp(0, h as isize - 1) as usize * w;
    for s in sums.iter_mut() {
        *s = [0; 4];
    }
    for y in -(r as isize)..=(r as isize) {
        let o = row(y);
        for (x, s) in sums.iter_mut().enumerate() {
            add4(s, buf[o + cx0 + x]);
        }
    }
    for y in 0..h {
        let ob = y * cw;
        for (x, s) in sums.iter().enumerate() {
            out[ob + x] = div4(s, inv);
        }
        let oa = row(y as isize + r as isize + 1);
        let os = row(y as isize - r as isize);
        for (x, s) in sums.iter_mut().enumerate() {
            add4(s, buf[oa + cx0 + x]);
            sub4(s, buf[os + cx0 + x]);
        }
    }
    for y in 0..h {
        buf[y * w + cx0..y * w + cx1].copy_from_slice(&out[y * cw..(y + 1) * cw]);
    }
}

/// Motion blur along `angle` (degrees) over `dist` pixels in O(1) per pixel:
/// running sums along sheared sample lines, interpolated across lines.
pub fn motion(buf: &mut [u32], w: usize, h: usize, angle: f32, dist: f32) {
    // Axis-aligned angles are a plain box: running sums, exact and fastest.
    let ang = angle.rem_euclid(180.0);
    if ang.abs() < 1e-3 || (ang - 180.0).abs() < 1e-3 {
        let r = (dist * 0.5).round() as usize;
        if r > 0 {
            let mut tmp = vec![0u32; w];
            for y in 0..h {
                box_line(&mut buf[y * w..(y + 1) * w], &mut tmp, r);
            }
        }
        return;
    }
    if (ang - 90.0).abs() < 1e-3 {
        let r = (dist * 0.5).round() as usize;
        if r > 0 {
            let mut sums = vec![[0u32; 4]; w];
            let mut out = vec![0u32; w * h];
            box_vertical(buf, w, h, 0, w, r, &mut sums, &mut out);
        }
        return;
    }
    let a = angle.to_radians();
    let (c, s) = (a.cos(), -a.sin());
    let opaque = opaque(buf);
    let mut planes = split(buf);
    let chans = if opaque { 3 } else { 4 };
    if c.abs() >= s.abs() {
        for p in planes.iter_mut().take(chans) {
            *p = motion_plane(p, w, h, s / c, (dist * c.abs() * 0.5).round() as usize);
        }
    } else {
        for p in planes.iter_mut().take(chans) {
            let t = transpose(p, w, h);
            let o = motion_plane(&t, h, w, c / s, (dist * s.abs() * 0.5).round() as usize);
            *p = transpose(&o, h, w);
        }
    }
    merge(&planes, buf);
}

/// Averages `2*half+1` samples along lines `y = b + x*slope` (|slope| <= 1).
fn motion_plane(p: &[u8], w: usize, h: usize, slope: f32, half: usize) -> Vec<u8> {
    if half == 0 {
        return p.to_vec();
    }
    let sample = |x: usize, y: f32| -> f32 {
        let y = y.clamp(0.0, (h - 1) as f32);
        let y0 = y.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let f = y - y0 as f32;
        p[y0 * w + x] as f32 * (1.0 - f) + p[y1 * w + x] as f32 * f
    };
    // Line b passes through (x, b + x*slope). Output (x, y) lies between
    // lines floor(y - x*slope) and +1.
    let bmin = (0.0f32).min(-(w as f32) * slope).floor() as i32 - 1;
    let bmax = (h as f32).max(h as f32 - (w as f32) * slope).ceil() as i32 + 1;
    let nb = (bmax - bmin + 1) as usize;
    let n = 2 * half + 1;
    let inv = 1.0 / n as f32;
    let mut v = vec![0f32; nb * w];
    let mut line = vec![0f32; w];
    for bi in 0..nb {
        let b = (bmin + bi as i32) as f32;
        // Skip lines that never cross the buffer.
        let y_at0 = b;
        let y_atw = b + w as f32 * slope;
        if y_at0.max(y_atw) < -1.0 || y_at0.min(y_atw) > h as f32 {
            continue;
        }
        for (x, l) in line.iter_mut().enumerate() {
            *l = sample(x, b + x as f32 * slope);
        }
        let at = |i: isize| line[i.clamp(0, w as isize - 1) as usize];
        let mut sum = 0.0f32;
        for i in -(half as isize)..=(half as isize) {
            sum += at(i);
        }
        let row = &mut v[bi * w..(bi + 1) * w];
        for (x, o) in row.iter_mut().enumerate() {
            *o = sum * inv;
            sum += at(x as isize + half as isize + 1) - at(x as isize - half as isize);
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let u = y as f32 - x as f32 * slope;
            let b = u.floor();
            let f = u - b;
            let bi = ((b as i32 - bmin) as usize).min(nb - 2);
            let a = v[bi * w + x];
            let c = v[(bi + 1) * w + x];
            out[y * w + x] = (a + (c - a) * f + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

pub fn transpose(p: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut o = vec![0u8; w * h];
    const B: usize = 32;
    for by in (0..h).step_by(B) {
        for bx in (0..w).step_by(B) {
            for y in by..(by + B).min(h) {
                for x in bx..(bx + B).min(w) {
                    o[x * h + y] = p[y * w + x];
                }
            }
        }
    }
    o
}

/// Disk ("lens") blur with highlight emphasis: average in a gamma-expanded
/// space so bright points bloom into bokeh discs. O(r) per pixel.
pub fn lens(buf: &mut [u32], w: usize, h: usize, radius: f32, bloom: f32) {
    let r = radius.round().max(1.0) as isize;
    let g = 1.0 + bloom / 25.0;
    let to_lin: Vec<f32> = (0..256).map(|v| (v as f32 / 255.0).powf(g)).collect();
    let from_lin: Vec<u8> = (0..4096).map(|i| ((i as f32 / 4095.0).powf(1.0 / g) * 255.0 + 0.5) as u8).collect();
    let spans: Vec<isize> = (-r..=r).map(|dy| (((r * r - dy * dy) as f32).sqrt()).floor() as isize).collect();
    let count: f32 = spans.iter().map(|&s| (2 * s + 1) as f32).sum();
    let opaque = opaque(buf);
    let mut planes = split(buf);
    let chans = if opaque { 3 } else { 4 };
    for (ci, p) in planes.iter_mut().enumerate().take(chans) {
        // Row prefix sums with one leading zero.
        let mut pre = vec![0f32; (w + 1) * h];
        for y in 0..h {
            let mut acc = 0.0;
            for x in 0..w {
                acc += if ci == 3 { p[y * w + x] as f32 / 255.0 } else { to_lin[p[y * w + x] as usize] };
                pre[y * (w + 1) + x + 1] = acc;
            }
        }
        // Row-major disk sum: for each output row, add one contiguous span
        // difference per disk row across all x at once (vectorizes along x).
        let mut out = vec![0u8; w * h];
        let mut acc = vec![0f32; w];
        let ri = r as usize;
        for y in 0..h {
            acc.iter_mut().for_each(|a| *a = 0.0);
            for (k, &span) in spans.iter().enumerate() {
                let yy = (y as isize + k as isize - r).clamp(0, h as isize - 1) as usize;
                let row = &pre[yy * (w + 1)..(yy + 1) * (w + 1)];
                let sp = span as usize;
                // Interior: both span ends inside the row.
                if w > 2 * sp {
                    let (hi, lo) = (&row[2 * sp + 1..], &row[..w - 2 * sp]);
                    for (a, (h_, l_)) in acc[sp..w - sp].iter_mut().zip(hi.iter().zip(lo)) {
                        *a += h_ - l_;
                    }
                }
                // Borders: clamp the span to the row (only touches ≤ 2r pixels).
                for x in (0..sp.min(w)).chain(w.saturating_sub(sp).max(sp.min(w))..w) {
                    let x0 = x.saturating_sub(sp);
                    let x1 = (x + sp + 1).min(w);
                    acc[x] += (row[x1] - row[x0]) * (2 * sp + 1) as f32 / (x1 - x0) as f32;
                }
            }
            let _ = ri;
            let o = &mut out[y * w..(y + 1) * w];
            for (ov, &a) in o.iter_mut().zip(&acc) {
                let m = (a / count).clamp(0.0, 1.0);
                *ov = if ci == 3 { (m * 255.0 + 0.5) as u8 } else { from_lin[(m * 4095.0) as usize] };
            }
        }
        *p = out;
    }
    merge(&planes, buf);
}

// ----- Lens Blur: iris, depth map, specular highlights, noise -------------------------

/// A layer mask used as Lens Blur's depth map: level 0 and its mips.
#[derive(Clone, Debug)]
pub struct DepthMask {
    pub levels: Vec<crate::engine::selection::MaskGrid>,
}

impl DepthMask {
    pub fn new(base: &crate::engine::selection::MaskGrid, mips: &[crate::engine::selection::MaskGrid]) -> DepthMask {
        DepthMask { levels: std::iter::once(base.clone()).chain(mips.iter().cloned()).collect() }
    }

    /// Mask values of a `w x h` region at `level` (edges replicated).
    pub fn read(&self, level: u32, x0: i32, y0: i32, w: usize, h: usize) -> Vec<u8> {
        let g = &self.levels[(level as usize).min(self.levels.len() - 1)];
        let mut out = vec![0u8; w * h];
        g.read_region(x0, y0, w, h, &mut out, true);
        out
    }
}

/// Where Lens Blur's per-pixel depth comes from (Photoshop's Depth Map
/// Source). With a depth map the blur radius at a pixel is
/// `radius * |depth - focal| / 255`.
#[derive(Clone, Debug, Default)]
pub enum Depth {
    #[default]
    None,
    /// The layer's own alpha.
    Transparency,
    /// A layer mask.
    Mask(std::sync::Arc<DepthMask>),
}

impl PartialEq for Depth {
    fn eq(&self, o: &Depth) -> bool {
        match (self, o) {
            (Depth::None, Depth::None) | (Depth::Transparency, Depth::Transparency) => true,
            (Depth::Mask(a), Depth::Mask(b)) => std::sync::Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// Lens Blur settings beyond the radius and the specular brightness
/// (`Filter::LensBlur { radius, bloom }`). The default is the round iris,
/// no depth map, every level specular, no noise: exactly [`lens`].
#[derive(Clone, Debug, PartialEq)]
pub struct LensOpts {
    /// Iris blades: 0 = round (the disk), 3..=8 = triangle .. octagon.
    pub blades: u8,
    /// Blade curvature, 0 (straight) ..= 100 (round).
    pub curvature: f32,
    /// Iris rotation in degrees.
    pub rotation: f32,
    pub depth: Depth,
    /// Blur focal distance, 0..=255 (the depth that stays sharp).
    pub focal: f32,
    /// Invert the depth map.
    pub invert: bool,
    /// Specular threshold, 0..=255: levels above it are boosted by the
    /// brightness (`bloom`); 0 boosts every level, 255 none.
    pub threshold: f32,
    /// Noise amount in levels, 0..=100.
    pub noise: f32,
    /// Gaussian (else uniform) noise.
    pub gaussian: bool,
    /// Monochromatic noise.
    pub mono: bool,
}

impl Default for LensOpts {
    fn default() -> Self {
        LensOpts {
            blades: 0,
            curvature: 0.0,
            rotation: 0.0,
            depth: Depth::None,
            focal: 0.0,
            invert: false,
            threshold: 0.0,
            noise: 0.0,
            gaussian: false,
            mono: false,
        }
    }
}

impl LensOpts {
    /// Today's disk-with-bloom blur ([`lens`]).
    pub fn is_plain(&self) -> bool {
        self.blades < 3 && self.depth == Depth::None && self.threshold <= 0.0 && self.noise <= 0.0
    }
}

/// The specular transfer pair of Lens Blur: `to_lin` (per 8-bit level) and
/// `from_lin` (per 1/4095 of the averaged value). Averaging through a
/// convex curve lets bright points dominate and bloom into bokeh. Levels up
/// to the threshold stay linear; above it the curve is `x^g`
/// (`g = 1 + bloom / 25`), joined with matching slope and normalized to
/// 0..=1. Threshold 0 is [`lens`]'s pair (`x^g` everywhere), 255 is linear.
pub fn lens_tables(bloom: f32, threshold: f32) -> (Vec<f32>, Vec<u8>) {
    let g = 1.0 + bloom / 25.0;
    if threshold <= 0.0 {
        let to_lin: Vec<f32> = (0..256).map(|v| (v as f32 / 255.0).powf(g)).collect();
        let from_lin: Vec<u8> = (0..4096).map(|i| ((i as f32 / 4095.0).powf(1.0 / g) * 255.0 + 0.5) as u8).collect();
        return (to_lin, from_lin);
    }
    let t = (threshold / 255.0).min(1.0);
    let (tg, gt) = (t.powf(g), g * t.powf(g - 1.0));
    let f = |x: f32| if x <= t { x } else { t + (x.powf(g) - tg) / gt };
    let f1 = f(1.0);
    let to_lin: Vec<f32> = (0..256).map(|v| f(v as f32 / 255.0) / f1).collect();
    let from_lin: Vec<u8> = (0..4096)
        .map(|i| {
            let y = i as f32 / 4095.0 * f1;
            let x = if y <= t { y } else { (tg + (y - t) * gt).powf(1.0 / g) };
            (x.min(1.0) * 255.0 + 0.5) as u8
        })
        .collect();
    (to_lin, from_lin)
}

/// Rows of the iris of radius `rho` as `[dy, a, b]` (pixels `a..=b` of row
/// `dy`). Round: [`lens`]'s disk. Polygons: `blades` straight edges whose
/// radial distance is blended toward the circle by `curvature` (0..=100),
/// rotated by `rotation` degrees (0: a vertex points up); each row is the
/// span from its first to its last inside pixel.
pub fn iris_spans(rho: usize, blades: u8, curvature: f32, rotation: f32) -> Vec<[i32; 3]> {
    let r = rho as isize;
    if rho == 0 {
        return vec![[0, 0, 0]];
    }
    if blades < 3 {
        return (-r..=r)
            .map(|dy| {
                let s = (((r * r - dy * dy) as f32).sqrt()).floor() as i32;
                [dy as i32, -s, s]
            })
            .collect();
    }
    let n = blades as f64;
    let rf = rho as f64;
    let c = (curvature as f64 / 100.0).clamp(0.0, 1.0);
    let rot = std::f64::consts::FRAC_PI_2 + (rotation as f64).to_radians();
    let sector = std::f64::consts::TAU / n;
    let apothem = rf * (std::f64::consts::PI / n).cos();
    let inside = |dx: isize, dy: isize| -> bool {
        let (x, y) = (dx as f64, -(dy as f64));
        let d = x.hypot(y);
        if d == 0.0 {
            return true;
        }
        // Angle from the nearest edge normal (normals sit between vertices).
        let phi = y.atan2(x) - rot - sector * 0.5;
        let t = phi.rem_euclid(sector) - sector * 0.5;
        let rp = apothem / t.cos();
        d <= rp + c * (rf - rp) + 1e-9
    };
    let mut out = Vec::new();
    for dy in -r..=r {
        let Some(a) = (-r..=r).find(|&dx| inside(dx, dy)) else { continue };
        let b = (-r..=r).rev().find(|&dx| inside(dx, dy)).unwrap_or(a);
        out.push([dy as i32, a as i32, b as i32]);
    }
    out
}

/// Per-pixel radii and the iris rows of every radius (shared with the GPU).
pub struct LensPlan {
    /// Blur radius per depth value (all equal without a depth map).
    pub rad: [u32; 256],
    pub uniform: bool,
    /// Iris rows of radius `rho` are `spans[off[rho]..off[rho + 1]]`.
    pub spans: Vec<[i32; 3]>,
    pub off: Vec<u32>,
    /// Pixels in the iris of each radius.
    pub count: Vec<f32>,
}

impl LensPlan {
    /// `r`: the radius at this level (`radius / 2^level`).
    pub fn new(r: f32, o: &LensOpts) -> LensPlan {
        let mut rad = [0u32; 256];
        let uniform = o.depth == Depth::None;
        if uniform {
            let rr = if r >= 0.5 { r.round().max(1.0) as u32 } else { 0 };
            rad = [rr; 256];
        } else {
            let rmax = r.round().max(0.0) as u32;
            for (d, v) in rad.iter_mut().enumerate() {
                let dd = if o.invert { 255.0 - d as f32 } else { d as f32 };
                *v = ((r * (dd - o.focal).abs() / 255.0).round().max(0.0) as u32).min(rmax);
            }
        }
        let max = *rad.iter().max().unwrap() as usize;
        let mut spans = Vec::new();
        let mut off = vec![0u32];
        let mut count = Vec::new();
        for rho in 0..=max {
            // Without a depth map only the one radius is needed.
            let s = if !uniform || rho == max { iris_spans(rho, o.blades, o.curvature, o.rotation) } else { Vec::new() };
            count.push(s.iter().map(|v| (v[2] - v[1] + 1) as f32).sum::<f32>().max(1.0));
            spans.extend_from_slice(&s);
            off.push(spans.len() as u32);
        }
        LensPlan { rad, uniform, spans, off, count }
    }

    pub fn spans_of(&self, rho: usize) -> &[[i32; 3]] {
        &self.spans[self.off[rho] as usize..self.off[rho + 1] as usize]
    }
}

/// Lens Blur noise of one channel at a level-space position.
#[inline]
pub fn lens_noise(gx: i32, gy: i32, c: u32, amount: f32, gaussian: bool) -> f32 {
    if gaussian {
        (hash01(gx, gy, c + 300) + hash01(gx, gy, c + 400) + hash01(gx, gy, c + 500) - 1.5) * (amount * 1.154_700_5)
    } else {
        (hash01(gx, gy, c + 300) * 2.0 - 1.0) * amount
    }
}

/// Adds Lens Blur's noise to premultiplied pixels (like Add Noise).
pub fn add_lens_noise(buf: &mut [u32], w: usize, origin: (i32, i32), amount: f32, gaussian: bool, mono: bool) {
    for (i, p) in buf.iter_mut().enumerate() {
        let [r, g, b, a8] = crate::engine::px::unpack(*p);
        if a8 == 0 {
            continue;
        }
        let (gx, gy) = (origin.0 + (i % w) as i32, origin.1 + (i / w) as i32);
        let n0 = lens_noise(gx, gy, 0, amount, gaussian);
        let (n1, n2) = if mono { (n0, n0) } else { (lens_noise(gx, gy, 1, amount, gaussian), lens_noise(gx, gy, 2, amount, gaussian)) };
        let (a, f) = (a8 as f32, a8 as f32 / 255.0);
        let ch = |v: u8, n: f32| (v as f32 + n * f).round().clamp(0.0, a) as u8;
        *p = crate::engine::px::pack(ch(r, n0), ch(g, n1), ch(b, n2), a8);
    }
}

/// Lens Blur with every Photoshop option. `r` is the radius at this level,
/// `depth` the per-pixel depth map (None: uniform radius), `origin` the
/// level-space position of `buf[0]` (for the noise). With the default
/// options this equals [`lens`].
#[allow(clippy::too_many_arguments)]
pub fn lens_ex(buf: &mut [u32], w: usize, h: usize, r: f32, bloom: f32, o: &LensOpts, depth: Option<&[u8]>, origin: (i32, i32)) {
    let plan = LensPlan::new(r, o);
    let (to_lin, from_lin) = lens_tables(bloom, o.threshold);
    let planes = split(buf);
    let mut out = planes.clone();
    let mut pre = vec![0f32; (w + 1) * h];
    let depth_at = |i: usize| depth.map_or(0, |d| d[i]) as usize;
    // One iris row's contribution at x, clamped to the row (scaled like
    // `lens` at the buffer's borders).
    let span = |row: &[f32], x: usize, a: i32, b: i32| -> f32 {
        let (xa, xb) = (x as isize + a as isize, x as isize + b as isize + 1);
        if xa >= 0 && xb <= w as isize {
            row[xb as usize] - row[xa as usize]
        } else {
            let x0 = xa.clamp(0, w as isize) as usize;
            let x1 = xb.clamp(0, w as isize) as usize;
            if x1 > x0 { (row[x1] - row[x0]) * (b - a + 1) as f32 / (x1 - x0) as f32 } else { 0.0 }
        }
    };
    for (ci, p) in planes.iter().enumerate() {
        for y in 0..h {
            let mut acc = 0.0;
            for x in 0..w {
                acc += if ci == 3 { p[y * w + x] as f32 / 255.0 } else { to_lin[p[y * w + x] as usize] };
                pre[y * (w + 1) + x + 1] = acc;
            }
        }
        let conv = |m: f32| if ci == 3 { (m * 255.0 + 0.5) as u8 } else { from_lin[(m * 4095.0) as usize] };
        let row_of = |y: usize, dy: i32| {
            let yy = (y as isize + dy as isize).clamp(0, h as isize - 1) as usize;
            &pre[yy * (w + 1)..(yy + 1) * (w + 1)]
        };
        let o_ch = &mut out[ci];
        if plan.uniform {
            let rho = plan.rad[0] as usize;
            if rho == 0 {
                continue;
            }
            let spans = plan.spans_of(rho);
            let count = plan.count[rho];
            let mut acc = vec![0f32; w];
            for y in 0..h {
                acc.iter_mut().for_each(|a| *a = 0.0);
                for &[dy, a, b] in spans {
                    let row = row_of(y, dy);
                    // Interior: the whole span inside the row (vectorizes).
                    let lo = ((-a).max(0) as usize).min(w);
                    let hi = ((w as isize - b as isize).max(0) as usize).clamp(lo, w);
                    let (ab, bb) = (a as isize, b as isize + 1);
                    for x in lo..hi {
                        acc[x] += row[(x as isize + bb) as usize] - row[(x as isize + ab) as usize];
                    }
                    for x in (0..lo).chain(hi..w) {
                        acc[x] += span(row, x, a, b);
                    }
                }
                for (x, &a) in acc.iter().enumerate() {
                    o_ch[y * w + x] = conv((a / count).clamp(0.0, 1.0));
                }
            }
        } else {
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let rho = plan.rad[depth_at(i)] as usize;
                    if rho == 0 {
                        continue;
                    }
                    let mut acc = 0.0f32;
                    for &[dy, a, b] in plan.spans_of(rho) {
                        acc += span(row_of(y, dy), x, a, b);
                    }
                    o_ch[i] = conv((acc / plan.count[rho]).clamp(0.0, 1.0));
                }
            }
        }
    }
    merge(&out, buf);
    if o.noise > 0.0 {
        add_lens_noise(buf, w, origin, o.noise, o.gaussian, o.mono);
    }
}

// ----- Smart Sharpen: Remove (Gaussian / Lens / Motion) and tonal fades -----------------

/// The blur Smart Sharpen's mask models (Photoshop's Remove).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Remove {
    #[default]
    Gaussian,
    /// A disk of the radius.
    Lens,
    /// A line of `2 * radius + 1` pixels at the angle.
    Motion,
}

/// Smart Sharpen's Shadows or Highlights fade: sharpening is attenuated by
/// up to `amount` % where the neighborhood (a box of `radius`) is within
/// `width` % of the tonal range from black (shadows) or white (highlights).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fade {
    /// Fade amount, 0..=100 %.
    pub amount: f32,
    /// Tonal width, 0..=100 %.
    pub width: f32,
    /// Neighborhood radius in px, 1..=100.
    pub radius: f32,
}

impl Default for Fade {
    fn default() -> Self {
        Fade { amount: 0.0, width: 50.0, radius: 1.0 }
    }
}

/// Smart Sharpen settings beyond amount, radius and reduce-noise.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SharpenOpts {
    pub remove: Remove,
    /// Motion Blur removal angle in degrees.
    pub angle: f32,
    pub shadows: Fade,
    pub highlights: Fade,
}

impl SharpenOpts {
    /// Plain unsharp masking with a Gaussian (the blur-combine family).
    pub fn is_plain(&self) -> bool {
        self.remove == Remove::Gaussian && self.shadows.amount <= 0.0 && self.highlights.amount <= 0.0
    }
}

/// Attenuation (8.8 fixed point, 256 = none) of the sharpening per local
/// luminance for one fade.
pub fn fade_table(f: &Fade, shadows: bool) -> [i32; 256] {
    let mut t = [256i32; 256];
    if f.amount <= 0.0 {
        return t;
    }
    let tw = f.width.clamp(0.0, 100.0) / 100.0 * 255.0;
    let k = f.amount.clamp(0.0, 100.0) / 100.0;
    for (l, v) in t.iter_mut().enumerate() {
        let l = l as f32;
        let wgt = if tw > 0.0 {
            if shadows { (tw - l) / tw } else { (l - (255.0 - tw)) / tw }.clamp(0.0, 1.0)
        } else {
            0.0
        };
        *v = ((1.0 - k * wgt) * 256.0).round() as i32;
    }
    t
}

/// Integer row prefix sums of a plane (`w + 1` per row, leading zero).
fn prefix_u32(p: &[u8], w: usize, h: usize) -> Vec<u32> {
    let mut pre = vec![0u32; (w + 1) * h];
    for y in 0..h {
        let mut acc = 0u32;
        for x in 0..w {
            acc += p[y * w + x] as u32;
            pre[y * (w + 1) + x + 1] = acc;
        }
    }
    pre
}

/// Disk average of radius `r` (Smart Sharpen's Lens Blur removal), rounded
/// to nearest: integer sums of row spans from prefix sums.
pub fn disk_plane(p: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let spans = iris_spans(r, 0, 0.0, 0.0);
    let count: u32 = spans.iter().map(|s| (s[2] - s[1] + 1) as u32).sum();
    let pre = prefix_u32(p, w, h);
    let mut out = vec![0u8; w * h];
    let mut acc = vec![0u32; w];
    for y in 0..h {
        acc.iter_mut().for_each(|a| *a = 0);
        for &[dy, a, b] in &spans {
            let yy = (y as isize + dy as isize).clamp(0, h as isize - 1) as usize;
            let row = &pre[yy * (w + 1)..(yy + 1) * (w + 1)];
            for (x, ac) in acc.iter_mut().enumerate() {
                let x0 = (x as isize + a as isize).clamp(0, w as isize) as usize;
                let x1 = (x as isize + b as isize + 1).clamp(0, w as isize) as usize;
                let (n, got) = ((b - a + 1) as u32, (x1 - x0) as u32);
                *ac += if got == n { row[x1] - row[x0] } else if got > 0 { (row[x1] - row[x0]) * n / got } else { 0 };
            }
        }
        for (o, &a) in out[y * w..(y + 1) * w].iter_mut().zip(&acc) {
            *o = ((2 * a + count) / (2 * count)) as u8;
        }
    }
    out
}

/// Offsets of the `2r + 1` pixels of a line at `angle` degrees through the
/// center (Smart Sharpen's Motion Blur removal).
pub fn line_taps(r: usize, angle: f32) -> Vec<[i32; 2]> {
    let a = (angle as f64).to_radians();
    let (c, s) = (a.cos(), -a.sin());
    (-(r as i32)..=r as i32).map(|t| [(t as f64 * c).round() as i32, (t as f64 * s).round() as i32]).collect()
}

/// Line average (clamped to the buffer), rounded to nearest.
pub fn line_plane(p: &[u8], w: usize, h: usize, taps: &[[i32; 2]]) -> Vec<u8> {
    let n = taps.len() as u32;
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0u32;
            for &[dx, dy] in taps {
                let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                s += p[yy * w + xx] as u32;
            }
            out[y * w + x] = ((2 * s + n) / (2 * n)) as u8;
        }
    }
    out
}

/// Box mean of radius `r` (clamped to the buffer), one rounding.
pub fn box_mean(p: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return p.to_vec();
    }
    let n = (2 * r + 1) as u32;
    let mut hs = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0;
            for j in -(r as isize)..=r as isize {
                s += p[y * w + (x as isize + j).clamp(0, w as isize - 1) as usize] as u32;
            }
            hs[y * w + x] = s;
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0;
            for j in -(r as isize)..=r as isize {
                s += hs[(y as isize + j).clamp(0, h as isize - 1) as usize * w + x];
            }
            out[y * w + x] = ((s + n * n / 2) / (n * n)) as u8;
        }
    }
    out
}

/// Luma (straight color) of premultiplied pixels, as Shadows/Highlights' table computes it.
pub fn luma8(buf: &[u32]) -> Vec<u8> {
    buf.iter()
        .map(|&p| {
            let [r, g, b, _] = crate::engine::px::unpremultiply(p);
            ((r as u32 * 77 + g as u32 * 150 + b as u32 * 29 + 128) >> 8) as u8
        })
        .collect()
}

/// Smart Sharpen beyond plain unsharp masking: the blur removed is a
/// Gaussian, a disk or a line, and the sharpening fades in the shadows and
/// highlights. `radius` and the fade radii are at this level. The combine is
/// [`super::fast::blur_combine`]'s, with the fades scaling the correction.
#[allow(clippy::too_many_arguments)]
pub fn smart_sharpen(buf: &mut [u32], w: usize, h: usize, amount: f32, radius: f32, threshold: f32, o: &SharpenOpts, fade_r: (usize, usize)) {
    let k = (amount / 100.0 * 256.0).round() as i32;
    let th = threshold.round() as i32;
    let mut planes = split(buf);
    let alpha = planes[3].clone();
    let taps = line_taps(radius.round().max(0.0) as usize, o.angle);
    let att: Option<Vec<i32>> = (o.shadows.amount > 0.0 || o.highlights.amount > 0.0).then(|| {
        let l = luma8(buf);
        let (ts, th_) = (fade_table(&o.shadows, true), fade_table(&o.highlights, false));
        let ls = box_mean(&l, w, h, fade_r.0);
        let lh = box_mean(&l, w, h, fade_r.1);
        ls.iter().zip(&lh).map(|(&s, &hh)| ts[s as usize] * th_[hh as usize]).collect()
    });
    for (ci, p) in planes.iter_mut().enumerate() {
        let r = radius.round().max(0.0) as usize;
        let b = match o.remove {
            Remove::Gaussian => {
                let mut b = p.clone();
                super::fast::gauss_plane(&mut b, w, h, radius);
                b
            }
            Remove::Lens if r > 0 => disk_plane(p, w, h, r),
            Remove::Motion if r > 0 => line_plane(p, w, h, &taps),
            _ => p.clone(),
        };
        for (i, (o_, &bv)) in p.iter_mut().zip(&b).enumerate() {
            let d = *o_ as i32 - bv as i32;
            if d.abs() < th {
                continue;
            }
            let mut delta = (d * k + 128) >> 8;
            if let Some(att) = &att {
                delta = (delta * att[i] + 32768) >> 16;
            }
            let lim = if ci == 3 { 255 } else { alpha[i] as i32 };
            *o_ = (*o_ as i32 + delta).clamp(0, lim) as u8;
        }
    }
    merge(&planes, buf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::px::pack;

    fn noisy(w: usize, h: usize, translucent: bool) -> Vec<u32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                let a = if translucent { (40 + (x * 7 + y * 3) % 216) as u8 } else { 255 };
                let c = |s: u32| (hash01(x, y, s) * 255.0) as u8;
                crate::engine::px::premultiply(c(1), c(2), c(3), a)
            })
            .collect()
    }

    #[test]
    fn lens_ex_defaults_equal_lens() {
        for translucent in [false, true] {
            let (w, h) = (70, 50);
            let src = noisy(w, h, translucent);
            for (r, bloom) in [(3.0, 30.0), (6.4, 0.0), (11.0, 75.0)] {
                let mut a = src.clone();
                lens(&mut a, w, h, r, bloom);
                let mut b = src.clone();
                lens_ex(&mut b, w, h, r, bloom, &LensOpts::default(), None, (0, 0));
                assert_eq!(a, b, "r {r} bloom {bloom} translucent {translucent}");
                // The same radius through the per-pixel (depth map) path.
                let depth = vec![255u8; w * h];
                let o = LensOpts { depth: Depth::Transparency, focal: 0.0, ..Default::default() };
                let mut c = src.clone();
                lens_ex(&mut c, w, h, r, bloom, &o, Some(&depth), (0, 0));
                assert_eq!(a, c, "depth path r {r}");
            }
        }
    }

    #[test]
    fn iris_shapes() {
        // Round: the disk, symmetric.
        let d = iris_spans(5, 0, 0.0, 0.0);
        assert_eq!(d.len(), 11);
        assert_eq!(d[5], [0, -5, 5]);
        for blades in 3..=8u8 {
            let s = iris_spans(20, blades, 0.0, 0.0);
            let area: i32 = s.iter().map(|v| v[2] - v[1] + 1).sum();
            let n = blades as f64;
            let ideal = 0.5 * n * 400.0 * (std::f64::consts::TAU / n).sin();
            assert!((area as f64 - ideal).abs() < ideal * 0.12, "{blades} blades: {area} vs {ideal}");
            // Full curvature is the circle.
            let c = iris_spans(20, blades, 100.0, 0.0);
            let ca: i32 = c.iter().map(|v| v[2] - v[1] + 1).sum();
            assert!((ca as f64 - std::f64::consts::PI * 400.0).abs() < 60.0, "{blades}: {ca}");
        }
        // Square rotated by 45° is a diamond: narrow top row.
        let sq = iris_spans(10, 4, 0.0, 0.0);
        let dia = iris_spans(10, 4, 0.0, 45.0);
        assert!(sq[0][2] - sq[0][1] < dia[0][2] - dia[0][1] || sq.len() != dia.len());
        // Triangle points up: rows near the top are narrower than near the bottom.
        let t = iris_spans(12, 3, 0.0, 0.0);
        let width = |dy: i32| t.iter().find(|v| v[0] == dy).map_or(0, |v| v[2] - v[1] + 1);
        assert!(width(-8) < width(4), "{} vs {}", width(-8), width(4));
    }

    #[test]
    fn lens_depth_map_blurs_by_distance_from_focus() {
        let (w, h) = (60, 60);
        let mut src = noisy(w, h, false);
        // Left half depth 255 (alpha), right half depth 0 after inversion.
        for (i, p) in src.iter_mut().enumerate() {
            if i % w >= 30 {
                let [r, g, b, _] = crate::engine::px::unpack(*p);
                *p = crate::engine::px::premultiply(r, g, b, 254);
            }
        }
        let depth: Vec<u8> = src.iter().map(|&p| (p >> 24) as u8).collect();
        // Focus on the left: it stays exactly sharp, the right isn't blurred
        // either (|254 - 255| * 8 / 255 rounds to 0).
        let o = LensOpts { depth: Depth::Transparency, focal: 255.0, ..Default::default() };
        let mut a = src.clone();
        lens_ex(&mut a, w, h, 8.0, 30.0, &o, Some(&depth), (0, 0));
        assert_eq!(a, src);
        // Focus at 0: everything gets the full radius.
        let o = LensOpts { depth: Depth::Transparency, focal: 0.0, ..Default::default() };
        let mut b = src.clone();
        lens_ex(&mut b, w, h, 8.0, 30.0, &o, Some(&depth), (0, 0));
        assert_ne!(b[30 * w + 10], src[30 * w + 10]);
        // Inverted map, focus 0: the left (depth 255 -> 0) stays sharp.
        let o = LensOpts { depth: Depth::Transparency, focal: 0.0, invert: true, ..Default::default() };
        let mut c = src.clone();
        lens_ex(&mut c, w, h, 8.0, 30.0, &o, Some(&depth), (0, 0));
        assert_eq!(c[30 * w + 10], src[30 * w + 10]);
        let plan = LensPlan::new(8.0, &LensOpts { depth: Depth::Transparency, focal: 100.0, ..Default::default() });
        assert_eq!(plan.rad[100], 0);
        assert_eq!(plan.rad[255], 5);
        assert_eq!(plan.rad[0], 3);
    }

    #[test]
    fn lens_specular_threshold() {
        let (w, h) = (50, 40);
        let src = noisy(w, h, false);
        // Threshold 255: nothing is specular, i.e. no bloom.
        let mut a = src.clone();
        lens_ex(&mut a, w, h, 5.0, 80.0, &LensOpts { threshold: 255.0, ..Default::default() }, None, (0, 0));
        let mut b = src.clone();
        lens(&mut b, w, h, 5.0, 0.0);
        let close = a.iter().zip(&b).all(|(&x, &y)| crate::engine::px::unpack(x).iter().zip(crate::engine::px::unpack(y)).all(|(p, q)| p.abs_diff(q) <= 1));
        assert!(close);
        // A mid threshold blooms less than threshold 0 but more than none.
        let mean = |v: &[u32]| v.iter().map(|&p| (p & 255) as f64).sum::<f64>() / v.len() as f64;
        let mut c = src.clone();
        lens_ex(&mut c, w, h, 5.0, 80.0, &LensOpts { threshold: 128.0, ..Default::default() }, None, (0, 0));
        let mut d = src.clone();
        lens(&mut d, w, h, 5.0, 80.0);
        assert!(mean(&b) < mean(&c) && mean(&c) < mean(&d), "{} {} {}", mean(&b), mean(&c), mean(&d));
        // The tables are monotone and invert each other closely.
        let (to, from) = lens_tables(60.0, 100.0);
        assert!(to.windows(2).all(|p| p[0] <= p[1]));
        for v in 0..256 {
            let back = from[(to[v] * 4095.0) as usize] as i32;
            assert!((back - v as i32).abs() <= 3, "{v} -> {back}");
        }
    }

    #[test]
    fn lens_noise_is_deterministic_and_mono() {
        let (w, h) = (40, 30);
        let c = crate::engine::px::pack(120, 120, 120, 255);
        let o = LensOpts { noise: 20.0, mono: true, ..Default::default() };
        let mut a = vec![c; w * h];
        lens_ex(&mut a, w, h, 3.0, 30.0, &o, None, (5, 7));
        let mut b = vec![c; w * h];
        lens_ex(&mut b, w, h, 3.0, 30.0, &o, None, (5, 7));
        assert_eq!(a, b);
        assert!(a.iter().any(|&p| p != c));
        assert!(a.iter().all(|&p| {
            let [r, g, b, _] = crate::engine::px::unpack(p);
            r == g && g == b && (r as i32 - 120).abs() <= 20
        }));
        let g = LensOpts { noise: 20.0, gaussian: true, ..Default::default() };
        let mut d = vec![c; w * h];
        lens_ex(&mut d, w, h, 3.0, 30.0, &g, None, (5, 7));
        assert!(d.iter().any(|&p| { let [r, gg, _, _] = crate::engine::px::unpack(p); r != gg }));
    }

    #[test]
    fn smart_sharpen_removal_modes() {
        let (w, h) = (48, 40);
        let flat = vec![crate::engine::px::pack(90, 140, 30, 255); w * h];
        for remove in [Remove::Gaussian, Remove::Lens, Remove::Motion] {
            let o = SharpenOpts { remove, angle: 30.0, ..Default::default() };
            let mut b = flat.clone();
            smart_sharpen(&mut b, w, h, 200.0, 3.0, 0.0, &o, (0, 0));
            assert_eq!(b, flat, "{remove:?} keeps flat areas");
        }
        // Lens removal: a disk average.
        let p: Vec<u8> = (0..w * h).map(|i| (i * 37 % 251) as u8).collect();
        let d = disk_plane(&p, w, h, 2);
        let spans = iris_spans(2, 0, 0.0, 0.0);
        let (x, y) = (20usize, 20usize);
        let (mut s, mut n) = (0u32, 0u32);
        for &[dy, a, b] in &spans {
            for dx in a..=b {
                s += p[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize] as u32;
                n += 1;
            }
        }
        assert_eq!(d[y * w + x] as u32, (2 * s + n) / (2 * n));
        // Motion removal at 0° is a horizontal line; at 90° a vertical one.
        assert_eq!(line_taps(2, 0.0), vec![[-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0]]);
        assert!(line_taps(2, 90.0).iter().all(|t| t[0] == 0));
        let l = line_plane(&p, w, h, &line_taps(2, 0.0));
        let s: u32 = (18..=22).map(|x| p[20 * w + x] as u32).sum();
        assert_eq!(l[20 * w + 20] as u32, (2 * s + 5) / 10);
    }

    #[test]
    fn smart_sharpen_fades_protect_shadows_and_highlights() {
        let (w, h) = (60, 20);
        // Dark stripes on the left, bright stripes on the right.
        let src: Vec<u32> = (0..w * h)
            .map(|i| {
                let x = i % w;
                let v = if x < 30 { if x % 2 == 0 { 10 } else { 30 } } else if x % 2 == 0 { 225 } else { 245 };
                crate::engine::px::pack(v, v, v, 255)
            })
            .collect();
        let none = SharpenOpts::default();
        let mut base = src.clone();
        smart_sharpen(&mut base, w, h, 150.0, 1.0, 0.0, &none, (0, 0));
        let full = Fade { amount: 100.0, width: 50.0, radius: 2.0 };
        let o = SharpenOpts { shadows: full, ..Default::default() };
        let mut s = src.clone();
        smart_sharpen(&mut s, w, h, 150.0, 1.0, 0.0, &o, (2, 0));
        let diff = |a: &[u32], b: &[u32], xs: std::ops::Range<usize>| xs.map(|x| (a[10 * w + x] & 255) as i32 - (b[10 * w + x] & 255) as i32).map(i32::abs).sum::<i32>();
        assert!(diff(&s, &src, 5..25) < diff(&base, &src, 5..25), "shadows faded");
        assert_eq!(diff(&s, &base, 35..55), 0, "highlights untouched");
        let o = SharpenOpts { highlights: full, ..Default::default() };
        let mut hl = src.clone();
        smart_sharpen(&mut hl, w, h, 150.0, 1.0, 0.0, &o, (0, 2));
        assert!(diff(&hl, &src, 35..55) < diff(&base, &src, 35..55), "highlights faded");
        assert_eq!(fade_table(&Fade::default(), true), [256; 256]);
        let t = fade_table(&Fade { amount: 50.0, width: 50.0, radius: 1.0 }, true);
        assert_eq!((t[0], t[255]), (128, 256));
    }

    #[test]
    fn plain_options_reproduce_the_old_filters() {
        use super::super::Filter;
        let (w, h) = (90, 70);
        let src = noisy(w, h, true);
        // Lens Blur: the plain options run `lens`.
        let mut a = src.clone();
        Filter::LensBlur { radius: 7.0, bloom: 30.0, opts: LensOpts::default() }.apply(&mut a, w, h, 0, (0, 0));
        let mut b = src.clone();
        lens(&mut b, w, h, 7.0, 30.0);
        assert_eq!(a, b);
        // Smart Sharpen: the plain options are unsharp masking, and the new
        // path with no fades gives the same pixels.
        let f = Filter::SmartSharpen { amount: 180.0, radius: 1.6, noise: 3.0, opts: SharpenOpts::default() };
        let mut c = src.clone();
        f.apply(&mut c, w, h, 0, (0, 0));
        let mut d = src.clone();
        super::super::fast::blur_combine(&mut d, w, h, 1.6, None, super::super::fast::Combine::Unsharp { k: (180.0f32 / 100.0 * 256.0).round() as i32, threshold: 3 });
        assert_eq!(c, d);
        let mut e = src.clone();
        smart_sharpen(&mut e, w, h, 180.0, 1.6, 3.0, &SharpenOpts::default(), (0, 0));
        let inner = |v: &[u32]| (10..h - 10).flat_map(|y| (10..w - 10).map(move |x| y * w + x)).map(|i| v[i]).collect::<Vec<_>>();
        assert_eq!(inner(&c), inner(&e));
    }

    #[test]
    fn motion_keeps_flat() {
        let c = pack(40, 90, 200, 255);
        let mut b = vec![c; 64 * 64];
        motion(&mut b, 64, 64, 30.0, 20.0);
        assert!(b.iter().all(|&p| p == c));
        motion(&mut b, 64, 64, 80.0, 20.0);
        assert!(b.iter().all(|&p| p == c));
    }

    #[test]
    fn horizontal_motion_smears_horizontally() {
        let (w, h) = (64, 16);
        let mut b = vec![pack(0, 0, 0, 255); w * h];
        b[8 * w + 32] = pack(255, 255, 255, 255);
        motion(&mut b, w, h, 0.0, 10.0);
        assert!(crate::engine::px::unpack(b[8 * w + 36])[0] > 0);
        assert_eq!(crate::engine::px::unpack(b[4 * w + 32])[0], 0);
    }
}
