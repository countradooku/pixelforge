//! Select ▸ Select and Mask: refines the current selection against the image.
//!
//! Stages, in Photoshop's order (all on the selection's coverage mask):
//! 1. **Edge Detection** (Radius, Smart Radius): inside a band around the
//!    selection edge the mask is replaced by a guided filter (He, Sun & Tang)
//!    guided by the image's luminance. Its input there is a local luminance
//!    key between the confidently selected and unselected pixels just outside
//!    the band (a trimap from the selection), so the edge snaps to image
//!    edges and soft detail (hair, fur) becomes partial coverage. The band is
//!    where a (2R+1)² window sees both selected and unselected pixels, with a
//!    soft falloff; Smart Radius narrows it where the image has a hard edge
//!    (strong local gradient) and keeps the full radius on soft detail.
//! 2. **Smooth**: a median of the mask (removes the "hills and valleys" of a
//!    jagged outline but keeps straight edges and soft ramps).
//! 3. **Feather**: Gaussian blur of the mask (as Select ▸ Modify ▸ Feather).
//! 4. **Contrast**: steepens soft edges (100 % = hard).
//! 5. **Shift Edge**: moves soft borders out (+) or in (−) by remapping the
//!    coverage with a gamma curve (0 and 255 stay put, so like Photoshop it
//!    acts on soft edges).
//! 6. **Decontaminate Colors** (Amount): edge pixels take the color of the
//!    nearest confidently selected pixels (normalized box averages of the
//!    pixels with coverage >= 250 at growing radii): fully up to 75 %
//!    coverage, fading out toward 100 %, times Amount.
//!
//! Every stage is integer arithmetic (box sums, fixed-point guided filter
//! coefficients, medians, the `gauss_plane` blur, CPU-built LUTs), so the GPU
//! version (`gpu::kernels::refine`) is bit-exact. Work is per tile with a
//! clamped apron covering every stage's reach; tiles whose apron is uniformly
//! selected or unselected come out unchanged and are skipped.

use rayon::prelude::*;

use super::grid::{RectI, TS, TS_I, TileGrid, finish_tile};
use super::selection::{MaskGrid, MaskTile};
use super::px;

/// Guided-filter regularization, in (0..1)² units.
pub const GF_EPS: f32 = 0.005;
/// Largest guided-filter window radius (keeps the fixed-point sums in 32 bits).
pub const GF_MAX_R: usize = 8;
/// Smart Radius: mean Sobel magnitude counted as a fully hard edge.
pub const HARD_GRAD: u32 = 160;
/// Coverage counted as confidently selected (decontamination source).
pub const FG_MIN: u32 = 250;
/// Radii at which decontamination looks for selected colors.
pub const DECON_RADII: [usize; 6] = [2, 4, 8, 16, 32, 64];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Output {
    #[default]
    Selection,
    LayerMask,
    NewLayerWithMask,
}

impl Output {
    pub const ALL: [Output; 3] = [Output::Selection, Output::LayerMask, Output::NewLayerWithMask];
    pub const LABELS: &'static [&'static str] = &["Selection", "Layer Mask", "New Layer with Layer Mask"];
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Edge Detection radius, px (0 = no edge refinement).
    pub radius: f32,
    pub smart: bool,
    /// 0..100.
    pub smooth: f32,
    /// px.
    pub feather: f32,
    /// 0..100 %.
    pub contrast: f32,
    /// −100..100 %.
    pub shift: f32,
    pub decontaminate: bool,
    /// 0..100 %.
    pub amount: f32,
    pub output: Output,
}

impl Default for Params {
    fn default() -> Self {
        Params { radius: 0.0, smart: false, smooth: 0.0, feather: 0.0, contrast: 0.0, shift: 0.0, decontaminate: false, amount: 100.0, output: Output::Selection }
    }
}

impl Params {
    pub fn edge_r(&self) -> usize {
        self.radius.round().clamp(0.0, 250.0) as usize
    }

    /// Radius within which the luminance key looks for confident pixels.
    pub fn key_r(&self) -> usize {
        2 * self.edge_r() + 1
    }

    /// Guided-filter window radius.
    pub fn gf_r(&self) -> usize {
        self.edge_r().clamp(1, GF_MAX_R)
    }

    /// Smart Radius: the narrow band used on hard edges.
    pub fn smart_r(&self) -> usize {
        (self.edge_r() / 4).max(1)
    }

    pub fn smooth_r(&self) -> usize {
        (self.smooth / 20.0).ceil().clamp(0.0, 5.0) as usize
    }

    /// Guided-filter epsilon in the fixed-point units of [`guided`].
    pub fn eps_q(&self) -> i32 {
        let n = ((2 * self.gf_r() + 1) * (2 * self.gf_r() + 1)) as f64;
        (((GF_EPS as f64 * 65025.0 * n * n) as i64) >> 12).max(1) as i32
    }

    pub fn decon(&self) -> bool {
        self.decontaminate && self.amount > 0.0
    }

    pub fn amount255(&self) -> u32 {
        (self.amount * 2.55).round().clamp(0.0, 255.0) as u32
    }

    /// Contrast then Shift Edge as one table (None when it is the identity).
    pub fn lut(&self) -> Option<[u8; 256]> {
        if self.contrast <= 0.0 && self.shift == 0.0 {
            return None;
        }
        let k = 1.0 / (1.0 - self.contrast.clamp(0.0, 100.0) / 100.0 * 0.995);
        let gamma = 2f32.powf(-self.shift.clamp(-100.0, 100.0) / 50.0);
        let mut lut = [0u8; 256];
        for (i, o) in lut.iter_mut().enumerate() {
            let c = ((i as f32 - 127.5) * k + 127.5).clamp(0.0, 255.0) / 255.0;
            *o = (c.powf(gamma) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
        }
        lut[0] = 0;
        lut[255] = 255;
        Some(lut)
    }

    /// How far any output pixel reads (each stage's reach, chained).
    pub fn pad(&self) -> usize {
        let er = self.edge_r();
        let s1 = if er > 0 { (er + self.key_r() + 2 * self.gf_r()).max(if self.smart { self.smart_r() + 1 } else { 0 }) } else { 0 };
        let feather = if self.feather >= 0.3 { super::filters::blur::gauss_pad(self.feather) } else { 0 };
        s1 + self.smooth_r() + feather + if self.decon() { DECON_RADII[DECON_RADII.len() - 1] } else { 0 } + 2
    }

    /// Whether every stage is off (the mask passes through).
    pub fn is_identity(&self) -> bool {
        self.edge_r() == 0 && self.smooth_r() == 0 && self.feather < 0.3 && self.lut().is_none() && !self.decon()
    }
}

// ----- stages on padded planes (the GPU mirrors these) ------------------------------------

/// Luminance guide (0..255) of a premultiplied pixel.
#[inline]
pub fn guide_of(p: u32) -> u32 {
    let [r, g, b, _] = px::unpremultiply(p);
    (r as u32 * 77 + g as u32 * 150 + b as u32 * 29) >> 8
}

/// Box sums over (2r+1)² windows, clamped at the buffer edges, wrapping
/// 32-bit arithmetic (exact whenever the true sum fits, signed or not).
pub fn box_sum(src: &[u32], w: usize, h: usize, r: usize) -> Vec<u32> {
    let mut tmp = vec![0u32; w * h];
    let mut sums = vec![0u32; w];
    let row = |y: isize| (y.clamp(0, h as isize - 1) as usize) * w;
    for j in -(r as isize)..=(r as isize) {
        let o = row(j);
        for (s, &v) in sums.iter_mut().zip(&src[o..o + w]) {
            *s = s.wrapping_add(v);
        }
    }
    for y in 0..h {
        tmp[y * w..(y + 1) * w].copy_from_slice(&sums);
        let (a, b) = (row(y as isize + r as isize + 1), row(y as isize - r as isize));
        for x in 0..w {
            sums[x] = sums[x].wrapping_add(src[a + x]).wrapping_sub(src[b + x]);
        }
    }
    let mut out = vec![0u32; w * h];
    out.par_chunks_mut(w).zip(tmp.par_chunks(w)).for_each(|(o, t)| {
        let at = |x: isize| t[x.clamp(0, w as isize - 1) as usize];
        let mut s = 0u32;
        for j in -(r as isize)..=(r as isize) {
            s = s.wrapping_add(at(j));
        }
        for x in 0..w {
            o[x] = s;
            s = s.wrapping_add(at(x as isize + r as isize + 1)).wrapping_sub(at(x as isize - r as isize));
        }
    });
    out
}

/// Fixed-point guided-filter coefficients from the window sums of I, p, I²
/// and I·p: `a` in Q10 (clamped to ±16), `b` in Q10.
#[inline]
pub fn gf_ab(n: u32, s_i: u32, s_p: u32, s_ii: u32, s_ip: u32, eps_q: i32) -> (i32, i32) {
    let cov = n.wrapping_mul(s_ip).wrapping_sub(s_i.wrapping_mul(s_p)) as i32;
    let var = n.wrapping_mul(s_ii).wrapping_sub(s_i.wrapping_mul(s_i)) as i32;
    let a = (((cov >> 12) * 1024) / ((var >> 12) + eps_q)).clamp(-16384, 16384);
    let b = (s_p as i32 * 1024 - a * s_i as i32) / n as i32;
    (a, b)
}

/// The filtered mask from the window sums of a and b.
#[inline]
pub fn gf_q(n: u32, s_a: u32, s_b: u32, i: u32) -> u32 {
    let qa = (s_a as i32 * i as i32) / n as i32;
    let qb = s_b as i32 / n as i32;
    ((qa + qb + 512) >> 10).clamp(0, 255) as u32
}

/// Guided filter of `p` (0..255) guided by `gi` (0..255), radius r.
pub fn guided(gi: &[u32], p: &[u32], w: usize, h: usize, r: usize, eps_q: i32) -> Vec<u32> {
    let n = ((2 * r + 1) * (2 * r + 1)) as u32;
    let ii: Vec<u32> = gi.iter().map(|&v| v * v).collect();
    let ip: Vec<u32> = gi.iter().zip(p).map(|(&a, &b)| a * b).collect();
    let (s_i, s_p, s_ii, s_ip) = (box_sum(gi, w, h, r), box_sum(p, w, h, r), box_sum(&ii, w, h, r), box_sum(&ip, w, h, r));
    let (mut a, mut b) = (vec![0u32; w * h], vec![0u32; w * h]);
    a.par_iter_mut().zip(b.par_iter_mut()).enumerate().for_each(|(k, (a, b))| {
        let (av, bv) = gf_ab(n, s_i[k], s_p[k], s_ii[k], s_ip[k], eps_q);
        *a = av as u32;
        *b = bv as u32;
    });
    let (s_a, s_b) = (box_sum(&a, w, h, r), box_sum(&b, w, h, r));
    (0..w * h).into_par_iter().map(|k| gf_q(n, s_a[k], s_b[k], gi[k])).collect()
}

/// |Sobel| (|gx| + |gy|, 0..2040) with clamped edges.
pub fn sobel(gi: &[u32], w: usize, h: usize) -> Vec<u32> {
    let at = |x: isize, y: isize| gi[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize] as i32;
    (0..w * h)
        .into_par_iter()
        .map(|k| {
            let (x, y) = ((k % w) as isize, (k / w) as isize);
            let gx = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
            let gy = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
            (gx.abs() + gy.abs()) as u32
        })
        .collect()
}

/// Band weight (0..255) from the count `c` of selected pixels in a window of `n`.
#[inline]
pub fn band_w(c: u32, n: u32) -> u32 {
    (c.min(n - c) * 2040 / n).min(255)
}

/// Smart Radius: hardness (0..255) from a window's summed Sobel magnitude.
#[inline]
pub fn hardness(g: u32, n: u32) -> u32 {
    ((g / n) * 255 / HARD_GRAD).min(255)
}

#[inline]
pub fn lerp8(a: u32, b: u32, t: u32) -> u32 {
    let v = a * (255 - t) + b * t + 128;
    (v + (v >> 8)) >> 8
}

/// Edge Detection's initial alpha estimate: inside the band, a local
/// luminance key between the mean guide value of the confidently selected
/// pixels (`kf`: a radius-R window fully selected) and of the confidently
/// unselected ones (`kb`) within `R2`; elsewhere (or without contrast) the
/// mask itself. Inputs are window sums at radius R2.
#[inline]
pub fn alpha_est(m: u32, i: u32, cf: u32, cb: u32, sf: u32, sb: u32) -> u32 {
    if cf == 0 || cb == 0 {
        return m;
    }
    let f = ((sf + cf / 2) / cf) as i32;
    let b = ((sb + cb / 2) / cb) as i32;
    let d = f - b;
    if d.abs() < 8 {
        return m;
    }
    ((i as i32 - b) * 255 / d).clamp(0, 255) as u32
}

/// Edge Detection: inside the band, the guided filter of the luminance-key
/// estimate (snaps the edge to the image and keeps soft detail partial).
pub fn edge_refine(m: &[u32], gi: &[u32], w: usize, h: usize, p: &Params) -> Vec<u32> {
    let er = p.edge_r();
    if er == 0 {
        return m.to_vec();
    }
    let bin: Vec<u32> = m.iter().map(|&v| (v >= 128) as u32).collect();
    let n_r = ((2 * er + 1) * (2 * er + 1)) as u32;
    let c_r = box_sum(&bin, w, h, er);
    let r2 = p.key_r();
    let kf: Vec<u32> = c_r.iter().map(|&c| (c == n_r) as u32).collect();
    let kb: Vec<u32> = c_r.iter().map(|&c| (c == 0) as u32).collect();
    let fi: Vec<u32> = kf.iter().zip(gi).map(|(&k, &i)| k * i).collect();
    let bi: Vec<u32> = kb.iter().zip(gi).map(|(&k, &i)| k * i).collect();
    let (cf, cb, sf, sb) = (box_sum(&kf, w, h, r2), box_sum(&kb, w, h, r2), box_sum(&fi, w, h, r2), box_sum(&bi, w, h, r2));
    let est: Vec<u32> = (0..w * h).into_par_iter().map(|k| alpha_est(m[k], gi[k], cf[k], cb[k], sf[k], sb[k])).collect();
    let q = guided(gi, &est, w, h, p.gf_r(), p.eps_q());
    let smart = p.smart.then(|| {
        let rs = p.smart_r();
        let n_s = ((2 * rs + 1) * (2 * rs + 1)) as u32;
        (n_s, box_sum(&bin, w, h, rs), box_sum(&sobel(gi, w, h), w, h, rs))
    });
    (0..w * h)
        .into_par_iter()
        .map(|k| {
            let mut wt = band_w(c_r[k], n_r);
            if let Some((n_s, c_s, g_s)) = &smart {
                let hard = hardness(g_s[k], *n_s);
                wt = (band_w(c_s[k], *n_s) * hard + wt * (255 - hard) + 127) / 255;
            }
            lerp8(m[k], q[k], wt)
        })
        .collect()
}

/// Exact (2r+1)² median (the (n/2)-th smallest); pixels within r of the
/// buffer edge are copied.
pub fn median(m: &[u32], w: usize, h: usize, r: usize) -> Vec<u32> {
    if r == 0 || w <= 2 * r || h <= 2 * r {
        return m.to_vec();
    }
    let k = ((2 * r + 1) * (2 * r + 1) / 2) as u32;
    let mut out = m.to_vec();
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y < r || y + r >= h {
            return;
        }
        let mut hist = [0u32; 256];
        for yy in y - r..=y + r {
            for &v in &m[yy * w..yy * w + 2 * r + 1] {
                hist[v as usize] += 1;
            }
        }
        for x in r..w - r {
            if x > r {
                for yy in y - r..=y + r {
                    hist[m[yy * w + x - r - 1] as usize] -= 1;
                    hist[m[yy * w + x + r] as usize] += 1;
                }
            }
            let mut acc = 0;
            for (v, &c) in hist.iter().enumerate() {
                acc += c;
                if acc > k {
                    row[x] = v as u32;
                    break;
                }
            }
        }
    });
    out
}

/// Feather: `gauss_plane` on the mask.
pub fn feather(m: &[u32], w: usize, h: usize, radius: f32) -> Vec<u32> {
    let mut p: Vec<u8> = m.iter().map(|&v| v as u8).collect();
    super::filters::fast::gauss_plane(&mut p, w, h, radius);
    p.into_iter().map(|v| v as u32).collect()
}

/// Decontaminate Colors on a padded buffer: edge pixels (0 < m < 255) move
/// toward the nearest confidently selected color. Returns premultiplied pixels.
pub fn decontaminate(img: &[u32], m: &[u32], w: usize, h: usize, amount255: u32) -> Vec<u32> {
    let straight: Vec<[u8; 4]> = img.par_iter().map(|&p| px::unpremultiply(p)).collect();
    let fg: Vec<u32> = m.iter().map(|&v| (v >= FG_MIN) as u32).collect();
    let lanes: Vec<Vec<u32>> = (0..3).map(|c| straight.iter().zip(&fg).map(|(s, &f)| s[c] as u32 * f).collect()).collect();
    // Packed straight color + 255 once resolved (as the GPU stores it).
    let mut col = vec![0u32; w * h];
    for &r in &DECON_RADII {
        let cnt = box_sum(&fg, w, h, r);
        let sums: Vec<Vec<u32>> = lanes.par_iter().map(|l| box_sum(l, w, h, r)).collect();
        col.par_iter_mut().enumerate().for_each(|(k, c)| {
            if *c >> 24 == 0 && cnt[k] > 0 {
                let n = cnt[k];
                let ch = |s: &Vec<u32>| (s[k] + n / 2) / n;
                *c = ch(&sums[0]) | ch(&sums[1]) << 8 | ch(&sums[2]) << 16 | 255 << 24;
            }
        });
    }
    (0..w * h)
        .into_par_iter()
        .map(|k| decon_px(img[k], m[k], col[k], amount255))
        .collect()
}

/// One decontaminated pixel: `col` is the packed straight fg color (alpha
/// byte 255 when found).
#[inline]
pub fn decon_px(p: u32, m: u32, col: u32, amount255: u32) -> u32 {
    let a = px::alpha(p);
    if m == 0 || m >= 255 || a == 0 || col >> 24 == 0 {
        return p;
    }
    let t = (amount255 * (4 * (255 - m)).min(255) + 127) / 255;
    let [r, g, b, _] = px::unpremultiply(p);
    let s = r as u32 | (g as u32) << 8 | (b as u32) << 16;
    let o = px::lerp(s, col & 0xff_ffff, t);
    let [r, g, b, _] = px::unpack(o);
    px::premultiply(r, g, b, a as u8)
}

/// All stages on one padded buffer: `(mask, colors)`, full buffer size.
pub fn refine_buf(m: &[u32], img: &[u32], w: usize, h: usize, p: &Params) -> (Vec<u32>, Option<Vec<u32>>) {
    let gi: Vec<u32> = img.par_iter().map(|&v| guide_of(v)).collect();
    let mut cur = edge_refine(m, &gi, w, h, p);
    let rs = p.smooth_r();
    if rs > 0 {
        cur = median(&cur, w, h, rs);
    }
    if p.feather >= 0.3 {
        cur = feather(&cur, w, h, p.feather);
    }
    if let Some(lut) = p.lut() {
        cur.iter_mut().for_each(|v| *v = lut[*v as usize] as u32);
    }
    let colors = p.decon().then(|| decontaminate(img, &cur, w, h, p.amount255()));
    (cur, colors)
}

// ----- whole selections ---------------------------------------------------------------------

pub struct Refined {
    pub mask: MaskGrid,
    /// Decontaminated layer pixels (when Decontaminate Colors is on).
    pub colors: Option<TileGrid>,
}

/// Tiles whose padded neighborhood is not uniformly selected or unselected.
pub fn todo_tiles(mask: &MaskGrid, pad: usize) -> Vec<bool> {
    let reach = pad.div_ceil(TS) as i32;
    let cols = mask.cols as usize;
    (0..mask.tiles.len())
        .into_par_iter()
        .map(|i| {
            let (tx, ty) = ((i % cols) as i32, (i / cols) as i32);
            let nb = RectI::new((tx - reach) * TS_I, (ty - reach) * TS_I, (tx + reach + 1) * TS_I, (ty + reach + 1) * TS_I);
            mask.region_uniform(nb).is_none()
        })
        .collect()
}

/// Reads the padded inputs of the tile block `[tx0, tx1) x [ty0, ty1)`.
pub fn gather(mask: &MaskGrid, img: &TileGrid, pad: usize, tx0: i32, ty0: i32, tx1: i32, ty1: i32) -> (Vec<u32>, Vec<u32>, usize, usize) {
    let (bw, bh) = (((tx1 - tx0) * TS_I) as usize + 2 * pad, ((ty1 - ty0) * TS_I) as usize + 2 * pad);
    let (x0, y0) = (tx0 * TS_I - pad as i32, ty0 * TS_I - pad as i32);
    let mut m8 = vec![0u8; bw * bh];
    mask.read_region(x0, y0, bw, bh, &mut m8, true);
    let mut im = vec![0u32; bw * bh];
    img.read_region_clamped(x0, y0, bw, bh, &mut im);
    (m8.into_iter().map(|v| v as u32).collect(), im, bw, bh)
}

/// Writes the results of a block (center buffers, `cw` wide, origin tile
/// `(tx0, ty0)`) for its to-do tiles.
#[allow(clippy::too_many_arguments)]
pub fn scatter(
    todo: &[bool],
    cols: u32,
    tx0: i32,
    ty0: i32,
    tx1: i32,
    ty1: i32,
    cw: usize,
    mask: &[u32],
    colors: Option<&[u32]>,
) -> Vec<(usize, MaskTile, Option<super::grid::TileRef>)> {
    let mut out = Vec::new();
    for ty in ty0..ty1 {
        for tx in tx0..tx1 {
            let i = (ty as u32 * cols + tx as u32) as usize;
            if !todo[i] {
                continue;
            }
            let (ox, oy) = (((tx - tx0) * TS_I) as usize, ((ty - ty0) * TS_I) as usize);
            let mut mt = vec![0u8; TS * TS];
            for y in 0..TS {
                for x in 0..TS {
                    mt[y * TS + x] = mask[(oy + y) * cw + ox + x] as u8;
                }
            }
            let ct = colors.map(|c| {
                let mut t = vec![0u32; TS * TS].into_boxed_slice();
                for y in 0..TS {
                    t[y * TS..(y + 1) * TS].copy_from_slice(&c[(oy + y) * cw + ox..(oy + y) * cw + ox + TS]);
                }
                finish_tile(t)
            });
            out.push((i, super::masks::normalize(mt), ct));
        }
    }
    out
}

/// Assembles the refined mask (and colors) from scattered tiles.
pub fn assemble(mask: &MaskGrid, img: &TileGrid, decon: bool, parts: Vec<(usize, MaskTile, Option<super::grid::TileRef>)>) -> Refined {
    let mut m = mask.clone();
    let mut colors = decon.then(|| img.clone());
    for (i, t, c) in parts {
        m.tiles[i] = t;
        if let (Some(cg), Some(c)) = (colors.as_mut(), c) {
            cg.tiles[i] = c;
        }
    }
    Refined { mask: m, colors }
}

/// The CPU reference: blocks of tiles with the full apron.
pub fn refine_cpu(mask: &MaskGrid, img: &TileGrid, p: &Params) -> Refined {
    if p.is_identity() {
        return Refined { mask: mask.clone(), colors: None };
    }
    let pad = p.pad();
    let todo = todo_tiles(mask, pad);
    let k: i32 = match pad {
        0..=48 => 1,
        49..=160 => 2,
        161..=480 => 4,
        _ => 8,
    };
    let (cols, rows) = (mask.cols as i32, mask.rows as i32);
    let blocks: Vec<(i32, i32)> = (0..rows).step_by(k as usize).flat_map(|by| (0..cols).step_by(k as usize).map(move |bx| (bx, by))).collect();
    let parts: Vec<_> = blocks
        .par_iter()
        .flat_map_iter(|&(bx, by)| {
            let mut span: Option<(i32, i32, i32, i32)> = None;
            for ty in by..(by + k).min(rows) {
                for tx in bx..(bx + k).min(cols) {
                    if todo[(ty * cols + tx) as usize] {
                        span = Some(span.map_or((tx, ty, tx + 1, ty + 1), |s| (s.0.min(tx), s.1.min(ty), s.2.max(tx + 1), s.3.max(ty + 1))));
                    }
                }
            }
            let Some((tx0, ty0, tx1, ty1)) = span else { return Vec::new() };
            let (m, im, bw, bh) = gather(mask, img, pad, tx0, ty0, tx1, ty1);
            let (rm, rc) = refine_buf(&m, &im, bw, bh, p);
            let (cw, ch) = (bw - 2 * pad, bh - 2 * pad);
            let crop = |v: &[u32]| -> Vec<u32> { (0..ch).flat_map(|y| v[(y + pad) * bw + pad..(y + pad) * bw + pad + cw].iter().copied()).collect() };
            let cm = crop(&rm);
            let cc = rc.as_deref().map(crop);
            scatter(&todo, mask.cols, tx0, ty0, tx1, ty1, cw, &cm, cc.as_deref())
        })
        .collect();
    assemble(mask, img, p.decon(), parts)
}

/// Select and Mask: the GPU when it's on and worth it, else the CPU.
pub fn refine(mask: &MaskGrid, img: &TileGrid, p: &Params) -> Refined {
    super::gpu::kernels::refine::apply(mask, img, p)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn square_mask(w: u32, h: u32, r: RectI) -> MaskGrid {
        MaskGrid::from_selection(&super::super::selection::Selection::rect(r), w, h)
    }

    fn plane(m: &MaskGrid) -> Vec<u8> {
        let mut v = vec![0u8; (m.w * m.h) as usize];
        m.read_region(0, 0, m.w as usize, m.h as usize, &mut v, false);
        v
    }

    fn flat(w: u32, h: u32, c: u32) -> TileGrid {
        TileGrid::filled(w, h, c)
    }

    #[test]
    fn feather_of_a_hard_square_is_a_symmetric_ramp() {
        let (w, h) = (300u32, 300u32);
        let m = square_mask(w, h, RectI::new(100, 100, 200, 200));
        let img = flat(w, h, px::pack(128, 128, 128, 255));
        let p = Params { feather: 6.0, ..Default::default() };
        let r = refine_cpu(&m, &img, &p);
        let v = plane(&r.mask);
        let at = |x: usize, y: usize| v[y * w as usize + x] as i32;
        // Deep inside / outside unchanged, the edge at about half coverage.
        assert_eq!(at(150, 150), 255);
        assert_eq!(at(20, 20), 0);
        assert!((at(100, 150) - 128).abs() <= 24, "edge {}", at(100, 150));
        // Monotone ramp across the left edge, symmetric about the edge.
        let row: Vec<i32> = (80..121).map(|x| at(x, 150)).collect();
        assert!(row.windows(2).all(|p| p[0] <= p[1]), "{row:?}");
        for d in 1..15 {
            let (a, b) = (at(100 - d, 150), at(99 + d, 150));
            assert!((a + b - 255).abs() <= 3, "d {d}: {a} + {b}");
        }
        // Same as Select ▸ Modify ▸ Feather.
        assert_eq!(v, plane(&m.feather(6.0)));
    }

    /// Column (along a row through the middle) where coverage crosses 50 %.
    fn crossing(v: &[u8], w: usize, y: usize) -> f32 {
        for x in 1..w {
            let (a, b) = (v[y * w + x - 1] as f32, v[y * w + x] as f32);
            if a < 128.0 && b >= 128.0 {
                return x as f32 - 1.0 + (128.0 - a) / (b - a);
            }
        }
        f32::NAN
    }

    #[test]
    fn shift_edge_moves_the_boundary() {
        let (w, h) = (300u32, 200u32);
        let m = square_mask(w, h, RectI::new(100, 0, 300, 200));
        let img = flat(w, h, px::pack(90, 90, 90, 255));
        let base = Params { feather: 5.0, ..Default::default() };
        let c0 = crossing(&plane(&refine_cpu(&m, &img, &base).mask), w as usize, 100);
        let out = crossing(&plane(&refine_cpu(&m, &img, &Params { shift: 60.0, ..base }).mask), w as usize, 100);
        let inn = crossing(&plane(&refine_cpu(&m, &img, &Params { shift: -60.0, ..base }).mask), w as usize, 100);
        assert!((c0 - 100.0).abs() < 1.0, "{c0}");
        assert!(out < c0 - 2.0, "positive shift moves the edge outward: {c0} -> {out}");
        assert!(inn > c0 + 2.0, "negative shift moves the edge inward: {c0} -> {inn}");
    }

    #[test]
    fn contrast_hardens_and_smooth_removes_specks() {
        let (w, h) = (300u32, 200u32);
        let m = square_mask(w, h, RectI::new(100, 0, 300, 200));
        let img = flat(w, h, px::pack(90, 90, 90, 255));
        let soft = plane(&refine_cpu(&m, &img, &Params { feather: 5.0, ..Default::default() }).mask);
        let hard = plane(&refine_cpu(&m, &img, &Params { feather: 5.0, contrast: 100.0, ..Default::default() }).mask);
        let partial = |v: &[u8]| v.iter().filter(|&&x| x > 0 && x < 255).count();
        assert!(partial(&hard) * 5 < partial(&soft), "{} vs {}", partial(&hard), partial(&soft));
        // A 2x2 speck outside and a 2x2 hole inside vanish with Smooth.
        let mut sel = plane(&m);
        for (x, y) in [(40, 40), (41, 40), (40, 41), (41, 41)] {
            sel[y * w as usize + x] = 255;
        }
        for (x, y) in [(200, 60), (201, 60), (200, 61), (201, 61)] {
            sel[y * w as usize + x] = 0;
        }
        let mut g = TileGrid::new(w, h);
        g.tiles = crate::engine::grid::build_grid(w, h, |tx, ty| {
            let mut t = vec![0u32; TS * TS].into_boxed_slice();
            for y in 0..TS {
                for x in 0..TS {
                    let (gx, gy) = (tx as usize * TS + x, ty as usize * TS + y);
                    if gx < w as usize && gy < h as usize {
                        let v = sel[gy * w as usize + gx];
                        t[y * TS + x] = px::pack(v, v, v, 255);
                    }
                }
            }
            finish_tile(t)
        })
        .tiles;
        let specky = MaskGrid::from_gray(&g);
        let sm = plane(&refine_cpu(&specky, &img, &Params { smooth: 40.0, ..Default::default() }).mask);
        assert_eq!(sm[40 * w as usize + 40], 0);
        assert_eq!(sm[60 * w as usize + 200], 255);
        assert_eq!(sm[100 * w as usize + 150], 255);
    }

    /// A red disk over blue with a soft edge: I = αF + (1 − α)B.
    pub fn fringe_scene(w: u32, h: u32) -> (MaskGrid, TileGrid) {
        let (cx, cy, rad) = (w as f32 * 0.5, h as f32 * 0.5, w.min(h) as f32 * 0.3);
        let alpha = |x: u32, y: u32| (((rad - ((x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy))) / 6.0 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        let mut img = TileGrid::new(w, h);
        img.tiles = crate::engine::grid::build_grid(w, h, |tx, ty| {
            let mut t = vec![0u32; TS * TS].into_boxed_slice();
            for y in 0..TS {
                for x in 0..TS {
                    let (gx, gy) = (tx as u32 * 256 + x as u32, ty as u32 * 256 + y as u32);
                    if gx < w && gy < h {
                        let a = alpha(gx, gy) as u32;
                        let mixc = |f: u32, b: u32| ((f * a + b * (255 - a) + 127) / 255) as u8;
                        t[y * TS + x] = px::pack(mixc(220, 30), mixc(40, 60), mixc(30, 230), 255);
                    }
                }
            }
            finish_tile(t)
        })
        .tiles;
        let mut g = TileGrid::new(w, h);
        g.tiles = crate::engine::grid::build_grid(w, h, |tx, ty| {
            let mut t = vec![0u32; TS * TS].into_boxed_slice();
            for y in 0..TS {
                for x in 0..TS {
                    let (gx, gy) = (tx as u32 * 256 + x as u32, ty as u32 * 256 + y as u32);
                    if gx < w && gy < h {
                        let a = alpha(gx, gy);
                        t[y * TS + x] = px::pack(a, a, a, 255);
                    }
                }
            }
            finish_tile(t)
        })
        .tiles;
        (MaskGrid::from_gray(&g), img)
    }

    #[test]
    fn decontaminate_removes_background_fringes() {
        let (w, h) = (300u32, 280u32);
        let (m, img) = fringe_scene(w, h);
        let p = Params { decontaminate: true, amount: 100.0, ..Default::default() };
        let r = refine_cpu(&m, &img, &p);
        let colors = r.colors.expect("decontaminated layer");
        let mv = plane(&r.mask);
        assert_eq!(mv, plane(&m), "decontamination alone keeps the mask");
        // Blue fringe on edge pixels before and after.
        let (mut before, mut after, mut n) = (0f64, 0f64, 0);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let a = mv[(y as u32 * w + x as u32) as usize];
                if (40..=200).contains(&a) {
                    before += px::unpack(img.pixel(x, y))[2] as f64;
                    after += px::unpack(colors.pixel(x, y))[2] as f64;
                    n += 1;
                }
            }
        }
        let (before, after) = (before / n as f64, after / n as f64);
        assert!(n > 200, "{n} edge pixels");
        assert!(before > 90.0, "fringe is blue before: {before:.1}");
        assert!(after < before * 0.4, "blue fringe removed: {before:.1} -> {after:.1}");
        // Fully selected and unselected pixels are untouched.
        assert_eq!(colors.pixel(150, 140), img.pixel(150, 140));
        assert_eq!(colors.pixel(2, 2), img.pixel(2, 2));
    }

    #[test]
    fn edge_detection_snaps_a_rough_selection_to_the_image_edge() {
        // Image: bright square on dark; the selection is 6 px too large.
        let (w, h) = (256u32, 256u32);
        let mut img = TileGrid::new(w, h);
        img.tiles = crate::engine::grid::build_grid(w, h, |_, _| {
            let mut t = vec![0u32; TS * TS].into_boxed_slice();
            for y in 0..TS {
                for x in 0..TS {
                    let inside = (80..176).contains(&x) && (80..176).contains(&y);
                    let v = if inside { 210 } else { 40 };
                    t[y * TS + x] = px::pack(v, v, v, 255);
                }
            }
            finish_tile(t)
        })
        .tiles;
        let m = square_mask(w, h, RectI::new(74, 74, 182, 182));
        let r = refine_cpu(&m, &img, &Params { radius: 10.0, ..Default::default() });
        let v = plane(&r.mask);
        let c = crossing(&v, w as usize, 128);
        assert!((c - 80.0).abs() < 1.5, "edge moved to the image edge: {c}");
        assert_eq!(v[128 * 256 + 128], 255);
        assert_eq!(v[10 * 256 + 10], 0);
    }

    #[test]
    fn blocks_do_not_change_the_result() {
        // The tile-block result equals one whole-image buffer's center.
        let (m, img) = fringe_scene(600, 520);
        let p = Params { radius: 12.0, smart: true, smooth: 30.0, feather: 2.0, contrast: 20.0, shift: 10.0, decontaminate: true, amount: 80.0, output: Output::Selection };
        let r = refine_cpu(&m, &img, &p);
        let pad = p.pad();
        let (mb, ib, bw, bh) = gather(&m, &img, pad, 0, 0, m.cols as i32, m.rows as i32);
        let (rm, rc) = refine_buf(&mb, &ib, bw, bh, &p);
        let v = plane(&r.mask);
        let rc = rc.unwrap();
        let colors = r.colors.unwrap();
        for y in 0..520usize {
            for x in 0..600usize {
                assert_eq!(v[y * 600 + x] as u32, rm[(y + pad) * bw + x + pad], "mask ({x}, {y})");
                assert_eq!(colors.pixel(x as i32, y as i32), rc[(y + pad) * bw + x + pad], "color ({x}, {y})");
            }
        }
    }
}
