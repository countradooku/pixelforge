//! Filter Gallery ▸ Sketch (part 1) effects.

use super::*;
use rayon::prelude::*;

pub(super) fn run(g: G, p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    use G::*;
    match g {
        BasRelief => bas_relief(p, img, ctx),
        Chrome => chrome(p, img, ctx),
        ChalkCharcoal => chalk_charcoal(p, img, ctx),
        Charcoal => charcoal(p, img, ctx),
        ConteCrayon => conte(p, img, ctx),
        GraphicPen => graphic_pen(p, img, ctx),
        HalftonePattern => halftone(p, img, ctx),
        _ => unreachable!("not in this group"),
    }
}

// ----- helpers ---------------------------------------------------------------------------

/// Exact separable Gaussian (kernel radius 3σ, edge clamped); for the small
/// radii where the three-box approximation of `blur` is too coarse.
fn gauss(v: &[f32], w: usize, h: usize, sx: f32, sy: f32) -> Vec<f32> {
    fn kernel(s: f32) -> Vec<f32> {
        if s < 0.2 {
            return vec![1.0];
        }
        let r = (s * 3.0).ceil() as i32;
        let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * s * s)).exp()).collect();
        let t: f32 = k.iter().sum();
        k.iter().map(|x| x / t).collect()
    }
    let (kx, ky) = (kernel(sx), kernel(sy));
    let (rx, ry) = ((kx.len() / 2) as isize, (ky.len() / 2) as isize);
    let mut t = vec![0f32; w * h];
    t.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = &v[y * w..(y + 1) * w];
        for (x, o) in out.iter_mut().enumerate() {
            let mut s = 0.0;
            for (k, &c) in kx.iter().enumerate() {
                s += c * row[(x as isize + k as isize - rx).clamp(0, w as isize - 1) as usize];
            }
            *o = s;
        }
    });
    if ky.len() == 1 {
        return t;
    }
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, dst)| {
        for (k, &c) in ky.iter().enumerate() {
            let sy = (y as isize + k as isize - ry).clamp(0, h as isize - 1) as usize;
            for (d, &s) in dst.iter_mut().zip(&t[sy * w..(sy + 1) * w]) {
                *d += c * s;
            }
        }
    });
    out
}

/// Bilinear sample of a plane, edge clamped.
#[inline]
fn bilerp(v: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let a = v[y0 * w + x0] + (v[y0 * w + x1] - v[y0 * w + x0]) * fx;
    let b = v[y1 * w + x0] + (v[y1 * w + x1] - v[y1 * w + x0]) * fx;
    a + (b - a) * fy
}

// ----- Bas Relief --------------------------------------------------------------------------

/// Photoshop lights the relief from below by default; our angle parameter
/// (default 135°) is turned so that the default reproduces that (135° →
/// light from the bottom). Dark areas are raised. The relief is the
/// difference of the smoothed luminance a few pixels apart along the light
/// direction, blurred a little more across it, then compressed with a soft
/// saturating curve whose knee is set by Detail.
fn bas_relief(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = img.lum();
    let k = p[1].max(0.5) / 3.0;
    let a = (p[2] + 135.0).to_radians();
    // Unit vector pointing towards the light in buffer coordinates (y down).
    let (ux, uy) = (a.cos(), -a.sin());
    let (vx, vy) = (-uy, ux);
    let (fwd, back) = (ctx.px(1.406 * k), ctx.px(2.39 * k));
    // σ ≈ 1.03 · k along the light and 1.62 · k across it.
    let (s_par, s_perp) = (ctx.px(1.025 * k), ctx.px(1.62 * k));
    let c = (0.036 * (16.0 - p[0].clamp(1.0, 15.0)) / 3.0).max(1e-3);
    let curve = |d: f32| 0.5 + 0.5 * d / (d.abs() + c);
    let mut t = vec![0f32; w * h];
    if ux.abs() < 1e-4 || uy.abs() < 1e-4 {
        // Axis-aligned light (the default): separable anisotropic blur.
        let b = if ux.abs() < 1e-4 { gauss(&l, w, h, s_perp, s_par) } else { gauss(&l, w, h, s_par, s_perp) };
        let (ux, uy) = (ux.round(), uy.round());
        t.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let (fx, fy) = (x as f32, y as f32);
                *o = curve(bilerp(&b, w, h, fx + fwd * ux, fy + fwd * uy) - bilerp(&b, w, h, fx - back * ux, fy - back * uy));
            }
        });
    } else {
        let b = gauss(&l, w, h, s_par, s_par);
        let se = (s_perp * s_perp - s_par * s_par).sqrt();
        let taps: Vec<(f32, f32)> = if se < 0.3 {
            vec![(0.0, 1.0)]
        } else {
            let r = (se * 2.5).ceil() as i32;
            let raw: Vec<(f32, f32)> = (-r..=r).map(|t| (t as f32, (-(t * t) as f32 / (2.0 * se * se)).exp())).collect();
            let s: f32 = raw.iter().map(|t| t.1).sum();
            raw.iter().map(|&(t, g)| (t, g / s)).collect()
        };
        t.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let (fx, fy) = (x as f32, y as f32);
                let mut d = 0.0;
                for &(q, g) in &taps {
                    let (px_, py_) = (fx + q * vx, fy + q * vy);
                    d += g * (bilerp(&b, w, h, px_ + fwd * ux, py_ + fwd * uy) - bilerp(&b, w, h, px_ - back * ux, py_ - back * uy));
                }
                *o = curve(d);
            }
        });
    }
    duotone(img, ctx, |i| t[i]);
}

// ----- Halftone Pattern ----------------------------------------------------------------------

/// Photoshop's screen: luminance smoothed by a 5×5 tent, plus a fixed
/// periodic pattern (45° dot grid of period 6 px at size 1, phase-locked to
/// the image origin), through a convex tone curve. At mip levels the pattern
/// is box-averaged over each preview pixel's footprint.
fn halftone(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = img.lum();
    let lb = if ctx.s <= 1.0 {
        let t = box3(&box3(&l, w, h, true), w, h, true);
        box3(&box3(&t, w, h, false), w, h, false)
    } else {
        gauss(&l, w, h, ctx.px(1.155), ctx.px(1.155))
    };
    let period = 4.0 + 2.0 * p[0].clamp(1.0, 12.0);
    let kind = p[2] as usize;
    let kc = (1.0 + p[1].max(0.0) / 5.0) * 0.5;
    let (beta, v0, ca, cb) = (0.0433f32, 0.2088f32, 1.647f32, 2.151f32);
    // Pivot where the curve crosses 0.5; contrast scales around it.
    let piv = (-ca + (ca * ca + 2.0 * cb).sqrt()) / (2.0 * cb);
    let shape = |c: f32| c.signum() * c.abs().powf(0.726);
    let tau = std::f32::consts::TAU;
    // Footprint average of shape(cos(2π(x − ph)/P)) over the level-0 pixels of a buffer pixel.
    let s = ctx.s.max(1.0) as i32;
    let avg = |lo: i32, ph: f32| -> f32 { (0..s).map(|k| shape((tau * ((lo + k) as f32 - ph) / period).cos())).sum::<f32>() / s as f32 };
    let x0 = |x: usize| (ctx.origin.0 + x as i32) * s;
    let y0 = |y: usize| (ctx.origin.1 + y as i32) * s;
    let (hx, hy): (Vec<f32>, Vec<f32>) = match kind {
        1 => (vec![], vec![]),
        2 => (vec![1.0; w], (0..h).map(|y| avg(y0(y), 1.0)).collect()),
        _ => ((0..w).map(|x| avg(x0(x), 4.0)).collect(), (0..h).map(|y| avg(y0(y), 1.0)).collect()),
    };
    duotone(img, ctx, |i| {
        let (x, y) = (i % w, i / w);
        let q = if kind == 1 {
            let (gx, gy) = ctx.g(i);
            shape((tau * (gx - 4096.0).hypot(gy - 4096.0) / period).cos())
        } else {
            hx[x] * hy[y]
        };
        let d = ((lb[i] + beta * q - v0 - piv) * kc + piv).max(0.0);
        (ca * d + cb * d * d).min(1.0)
    });
}

/// 3-tap box filter along x (`horiz`) or y, edge clamped.
fn box3(v: &[f32], w: usize, h: usize, horiz: bool) -> Vec<f32> {
    let mut o = vec![0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (a, c) = if horiz {
                (v[y * w + x.saturating_sub(1)], v[y * w + (x + 1).min(w - 1)])
            } else {
                (v[y.saturating_sub(1) * w + x], v[(y + 1).min(h - 1) * w + x])
            };
            o[y * w + x] = (a + v[y * w + x] + c) * (1.0 / 3.0);
        }
    }
    o
}

// ----- positional noise ------------------------------------------------------------------

/// Uniform noise on the buffer grid extended by `m` pixels on every side,
/// keyed on level-0 coordinates so tiles and mip levels agree.
fn noise_ext(ctx: &Ctx, m: usize, seed: u32) -> (Vec<f32>, usize, usize) {
    let (ew, eh) = (ctx.w + 2 * m, ctx.h + 2 * m);
    let mut v = vec![0f32; ew * eh];
    v.par_chunks_mut(ew).enumerate().for_each(|(ey, row)| {
        let gy = (((ctx.origin.1 + ey as i32 - m as i32) as f32 + 0.5) * ctx.s).floor() as i32;
        for (ex, o) in row.iter_mut().enumerate() {
            let gx = (((ctx.origin.0 + ex as i32 - m as i32) as f32 + 0.5) * ctx.s).floor() as i32;
            *o = super::super::util::hash01(gx, gy, seed);
        }
    });
    (v, ew, eh)
}

/// Mean of `n` samples centred on each pixel along the integer step (dx, dy), edge clamped.
fn line_mean(v: &[f32], w: usize, h: usize, n: usize, (dx, dy): (i32, i32)) -> Vec<f32> {
    let n = n.max(1) as i32;
    let r0 = -(n / 2);
    let inv = 1.0 / n as f32;
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for k in r0..r0 + n {
                let xx = (x as i32 + k * dx).clamp(0, w as i32 - 1) as usize;
                let yy = (y as i32 + k * dy).clamp(0, h as i32 - 1) as usize;
                s += v[yy * w + xx];
            }
            *o = s * inv;
        }
    });
    out
}

/// Crops the interior of an extended plane back to the buffer.
fn crop(v: &[f32], ew: usize, m: usize, w: usize, h: usize) -> Vec<f32> {
    let mut o = Vec::with_capacity(w * h);
    for y in 0..h {
        o.extend_from_slice(&v[(y + m) * ew + m..(y + m) * ew + m + w]);
    }
    o
}

/// A level-0 stroke length in whole level pixels (at least one).
fn taps(ctx: &Ctx, len: f32) -> usize {
    (ctx.px(len).round() as usize).max(1)
}

// ----- Graphic Pen -----------------------------------------------------------------------

/// Photoshop thresholds the luminance against uniform noise averaged along
/// one-pixel-wide lines (length = Stroke length) in the stroke direction.
fn graphic_pen(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = img.lum();
    let dir = [(1, -1), (1, 0), (1, 1), (0, 1)][(p[2].max(0.0) as usize).min(3)];
    let n = taps(ctx, p[0].clamp(1.0, 15.0));
    let m = n / 2 + 1;
    let (nz, ew, eh) = noise_ext(ctx, m, 131);
    let st = crop(&line_mean(&nz, ew, eh, n, dir), ew, m, w, h);
    // Keep the noise spread constant (σ ≈ 0.11) whatever the stroke length.
    let amp = 0.112 / (0.2887 / (n as f32).sqrt());
    let thr = 0.455 + (p[1] - 50.0) / 100.0;
    duotone(img, ctx, |i| if l[i] + (st[i] - 0.5) * amp > thr { 1.0 } else { 0.0 });
}

// ----- Conté Crayon ----------------------------------------------------------------------

/// Smooth 1-D value noise along x for row `y`.
fn row_noise(x: f32, y: i32, seed: u32) -> f32 {
    let xi = x.floor();
    let t = x - xi;
    let t = t * t * (3.0 - 2.0 * t);
    let a = super::super::util::hash01(xi as i32, y, seed);
    let b = super::super::util::hash01(xi as i32 + 1, y, seed);
    a + (b - a) * t
}

/// Smooth 2-D value noise.
fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (tx, ty) = (x - xi, y - yi);
    let (tx, ty) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let hsh = |a: f32, b: f32| super::super::util::hash01(a as i32, b as i32, seed);
    let a0 = hsh(xi, yi) + (hsh(xi + 1.0, yi) - hsh(xi, yi)) * tx;
    let a1 = hsh(xi, yi + 1.0) + (hsh(xi + 1.0, yi + 1.0) - hsh(xi, yi + 1.0)) * tx;
    a0 + (a1 - a0) * ty
}

/// Conté paper "Canvas": fine horizontal crayon dashes on a 3-row weave whose
/// rows drift slowly; 1 = ridge (crayon skips it), 0 = valley.
fn conte_canvas(x: f32, y: f32) -> f32 {
    let yi = y.floor() as i32;
    let drift = (value_noise(x / 150.0, y / 150.0, 5) * 3.0) as i32;
    let ridge = (yi + drift).rem_euclid(3) == 0;
    let n = row_noise(x / 5.0, yi, 3) * 0.7 + row_noise(x / 1.75, yi, 4) * 0.3;
    (0.45 + 2.0 * (n - 0.5) + if ridge { 0.6 } else { 0.0 }).clamp(0.0, 1.0)
}

/// Photoshop's Conté: a 50 % "paper" tone; foreground crayon darkens it where
/// the image is darker than a texture-dependent threshold, background crayon
/// lightens it where brighter. Ridges of the paper texture resist both.
fn conte(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = img.lum();
    let kind = p[2].max(0.0) as usize;
    let cd = 0.519 + (p[0] - 11.0) * 0.025;
    let cb = 0.5625 - (p[1] - 7.0) * 0.025;
    let tone = |v: f32, tv: f32| {
        let d = ((v - (cd - 0.444 * tv)) * 3.0).clamp(0.0, 1.0);
        let b = ((v - (cb + 0.25 * tv)) * 6.375).clamp(0.0, 1.0);
        0.5 * d + 0.5 * b
    };
    let mut t = vec![0f32; w * h];
    if kind == 0 {
        // Average a few level-0 rows per preview pixel so mip levels do not alias the weave.
        let sub = (ctx.s as usize).clamp(1, 4);
        let step = ctx.s / sub as f32;
        t.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let gx = (ctx.origin.0 + x as i32) as f32 * ctx.s;
                let gy = (ctx.origin.1 + y as i32) as f32 * ctx.s;
                let v = l[y * w + x];
                *o = (0..sub).map(|k| tone(v, conte_canvas(gx + (k as f32 + 0.5) * step, gy + (k as f32 + 0.5) * step))).sum::<f32>() / sub as f32;
            }
        });
    } else {
        let tex = texture(ctx, kind - 1, 1.0);
        for i in 0..w * h {
            t[i] = tone(l[i], (tex[i] * 1.6 - 0.3).clamp(0.0, 1.0));
        }
    }
    duotone(img, ctx, |i| t[i]);
}

// ----- Chalk & Charcoal --------------------------------------------------------------------

/// Each pixel takes `src` at the nearest jittered stroke anchor under an
/// elongated metric (strokes `len` × `wid` level-0 px along `ang`°, y down),
/// so the image breaks into short flat-coloured dashes. Also returns a
/// per-stroke random value.
fn dash_field(src: &[f32], ctx: &Ctx, ang: f32, len: f32, wid: f32, seed: u32) -> (Vec<f32>, Vec<f32>) {
    #[inline]
    fn mix(x: i32, y: i32, seed: u32) -> u32 {
        let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
        h ^= h >> 13;
        h = h.wrapping_mul(0x5bd1_e995);
        h ^ (h >> 15)
    }
    const U16: f32 = 1.0 / 65536.0;
    let (w, h) = (ctx.w, ctx.h);
    let a = ang.to_radians();
    let (c, s) = (a.cos(), a.sin());
    let (cu, cv) = (len * 0.75, wid);
    let (il, iw) = (1.0 / len, 1.0 / wid);
    let mut val = vec![0f32; w * h];
    let mut hv = vec![0f32; w * h];
    val.par_chunks_mut(w).zip(hv.par_chunks_mut(w)).enumerate().for_each(|(y, (vr, hr))| {
        for x in 0..w {
            let (gx, gy) = ctx.g(y * w + x);
            let (u, v) = (gx * c + gy * s, -gx * s + gy * c);
            let (iu, iv) = ((u / cu).floor() as i32, (v / cv).floor() as i32);
            let mut best = (f32::MAX, 0f32, 0f32, 0f32);
            for dv in -1..=1 {
                for du in -1..=1 {
                    let (ku, kv) = (iu + du, iv + dv);
                    // Two 32-bit hashes give the anchor jitter (u, v), a priority and the stroke value.
                    let (h1, h2) = (mix(ku, kv, seed), mix(ku, kv, seed ^ 0x9e37_79b9));
                    let au = (ku as f32 + (h1 >> 16) as f32 * U16) * cu;
                    let av = (kv as f32 + (h1 & 0xffff) as f32 * U16) * cv;
                    let d = ((u - au) * il).powi(2) + ((v - av) * iw).powi(2) + 0.3 * (h2 >> 16) as f32 * U16;
                    if d < best.0 {
                        best = (d, au, av, (h2 & 0xffff) as f32 * U16);
                    }
                }
            }
            let (ax, ay) = (best.1 * c - best.2 * s, best.1 * s + best.2 * c);
            let bx = ax / ctx.s - ctx.origin.0 as f32 - 0.5;
            let by = ay / ctx.s - ctx.origin.1 as f32 - 0.5;
            vr[x] = bilerp(src, w, h, bx, by);
            hr[x] = best.3;
        }
    });
    (val, hv)
}

/// Photoshop's Chalk & Charcoal: the image broken into short flat dashes;
/// charcoal (foreground) dashes run "\" in the shadows, chalk (background)
/// dashes run "/" in the highlights, over a 50 % tone.
fn chalk_charcoal(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = gauss(&img.lum(), w, h, ctx.px(1.0), ctx.px(1.0));
    let src_d = line_mean(&l, w, h, taps(ctx, 5.0), (1, 1));
    let src_l = line_mean(&l, w, h, taps(ctx, 7.0), (1, -1));
    let (dk, hd) = dash_field(&src_d, ctx, 45.0, 7.0, 2.0, 211);
    let (lt, _) = dash_field(&src_l, ctx, -45.0, 10.0, 2.5, 223);
    let k = 0.6 + 0.4 * p[2].clamp(0.0, 5.0);
    let td = 0.48 + (p[0] - 6.0) * 0.02;
    let tb = 0.43 - (p[1] - 6.0) * 0.02;
    duotone(img, ctx, |i| {
        // Some charcoal strokes leave paper showing even in the deepest shadow.
        let d = ((td - dk[i]) * 3.1 * k).clamp(0.0, 1.0).min(if hd[i] > 0.75 { 0.75 } else { 1.0 });
        let c = ((lt[i] - tb) * 4.2 * k).clamp(0.0, 1.0);
        0.5 - 0.5 * d + 0.435 * c
    });
}

// ----- Charcoal ---------------------------------------------------------------------------

/// Hatching noise: blurred noise averaged along `dir`, band-passed across it,
/// for a `w`×`h` buffer at `ctx`'s origin/level; unit variance.
fn hatch(ctx: &Ctx, n: usize, dir: (i32, i32), pre: f32, bp: f32, seed: u32) -> Vec<f32> {
    let sd = hatch_sd(ctx.s, n, dir, pre, bp, seed);
    hatch_field(ctx, n, dir, pre, bp, seed).iter().map(|v| v / sd).collect()
}

/// The un-normalized hatching field of `hatch`.
fn hatch_field(c: &Ctx, n: usize, dir: (i32, i32), pre: f32, bp: f32, seed: u32) -> Vec<f32> {
    let m = n / 2 + (3.0 * pre).ceil() as usize + (3.0 * bp).ceil() as usize + 2;
    let (nz, ew, eh) = noise_ext(c, m, seed);
    let nz = gauss(&nz, ew, eh, pre, pre);
    let st = line_mean(&nz, ew, eh, n, dir);
    let lo = gauss(&st, ew, eh, bp, bp);
    let hp: Vec<f32> = st.iter().zip(&lo).map(|(a, b)| a - b).collect();
    crop(&hp, ew, m, c.w, c.h)
}

/// `hatch`'s normalization: the field's RMS over a fixed reference patch, so
/// the scale never depends on the tile; it only depends on the arguments,
/// so it is cached. (The GPU path divides by the same value.)
pub(crate) fn hatch_sd(s: f32, n: usize, dir: (i32, i32), pre: f32, bp: f32, seed: u32) -> f32 {
    use std::sync::Mutex;
    static CACHE: Mutex<Vec<([u32; 5], f32)>> = Mutex::new(Vec::new());
    let key = [n as u32, (dir.0 + 2 + 4 * (dir.1 + 2)) as u32, pre.to_bits(), bp.to_bits(), s.to_bits() ^ seed.rotate_left(16)];
    let cached = CACHE.lock().ok().and_then(|c| c.iter().find(|e| e.0 == key).map(|e| e.1));
    cached.unwrap_or_else(|| {
        let r = Ctx { w: 64, h: 64, s, origin: (0, 0), fg: [0.0; 3], bg: [0.0; 3], doc: None };
        let refp = hatch_field(&r, n, dir, pre, bp, seed);
        let sd = (refp.iter().map(|v| v * v).sum::<f32>() / refp.len() as f32).sqrt().max(1e-6);
        if let Ok(mut c) = CACHE.lock() {
            if c.len() > 64 {
                c.clear();
            }
            c.push((key, sd));
        }
        sd
    })
}

/// Photoshop's Charcoal: the image smeared along both diagonals, through a
/// steep tone curve that leaves shadows at a charcoal grey, overlaid with
/// long "/" and shorter "\" hatching strokes that are strongest in the darks.
fn charcoal(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = img.lum();
    let ns = taps(ctx, 13.0 + (5.0 - p[1].clamp(0.0, 5.0)) * 3.0);
    let s1 = line_mean(&l, w, h, ns, (1, -1));
    let s2 = line_mean(&l, w, h, ns, (1, 1));
    let th = 0.5 + 0.5 * p[0].clamp(1.0, 7.0);
    let (pre, bp) = (ctx.px(1.0 * th), ctx.px(2.5 * th));
    let f1 = hatch(ctx, taps(ctx, 15.0), (1, -1), pre, bp, 121);
    let f2 = hatch(ctx, taps(ctx, 7.0), (1, 1), pre, bp, 122);
    let q0 = 0.41 + (p[2] - 50.0) / 100.0 * 0.6;
    duotone(img, ctx, |i| {
        let ramp = ((0.5 * (s1[i] + s2[i]) - q0) * 8.0).clamp(0.0, 1.0);
        let amp = 0.02 + 0.25 * (1.0 - ramp);
        0.22 + 0.78 * ramp + amp * (0.7 * f1[i] + 0.4 * f2[i])
    });
}

// ----- Chrome ------------------------------------------------------------------------------

/// Liquid chrome: the smoothed image is treated as a polished surface; a
/// phase built from its fine relief (lit from the top left) and its local
/// height drives a periodic reflection curve that keeps flat areas at a
/// metallic mid grey, opens black pools with bright rims where the surface
/// bends sharply, and adds soft grey ripples in between.
fn chrome(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = img.lum();
    let sig0 = 0.5 + p[1].clamp(0.0, 10.0) * 0.93;
    let sig = ctx.px(sig0);
    let b = gauss(&l, w, h, sig, sig);
    let lo = gauss(&l, w, h, ctx.px(12.0), ctx.px(12.0));
    let det = 0.5 + p[0].clamp(0.0, 10.0) / 8.0;
    // Slopes shrink as the surface gets smoother: scale the relief gain with σ.
    let (k, fr) = (300.0 * det * (sig0 / 7.01) / ctx.s, 50.0 * det);
    let st = |e0: f32, e1: f32, x: f32| {
        let u = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        u * u * (3.0 - 2.0 * u)
    };
    let mut t = vec![0f32; w * h];
    t.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let (yu, yd) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for (x, o) in row.iter_mut().enumerate() {
            let (xl, xr) = (x.saturating_sub(1), (x + 1).min(w - 1));
            let i = y * w + x;
            let gx = b[y * w + xr] - b[y * w + xl];
            let gy = b[yd * w + x] - b[yu * w + x];
            let phi = fr * (b[i] - lo[i]) + k * (-0.6 * gx - 0.8 * gy);
            let c = (1.0 - phi.cos()) * 0.5;
            let base = 0.45 + (lo[i] - 0.45) * 0.7 + 0.10 * phi.sin();
            *o = base * (1.0 - 0.85 * st(0.85, 0.97, c)) + 0.4 * (-((c - 0.73) / 0.05).powi(2)).exp();
        }
    });
    let t = gauss(&t, w, h, ctx.px(0.7), ctx.px(0.7));
    duotone(img, ctx, |i| t[i]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::px;

    const EFFECTS: [G; 7] = [G::BasRelief, G::ChalkCharcoal, G::Charcoal, G::Chrome, G::ConteCrayon, G::GraphicPen, G::HalftonePattern];

    fn defaults(g: G) -> [f32; 3] {
        let mut p = [0f32; 3];
        for (k, d) in g.params().iter().enumerate().take(3) {
            p[k] = d.default;
        }
        p
    }

    /// The reference photo when present, else the procedural test image, at 1024 × 683.
    fn source() -> (Vec<u32>, usize, usize) {
        let (w, h) = (1024usize, 683usize);
        if let Ok(im) = image::open("/tmp/pf_ps_out/gallery_src.png") {
            let im = im.to_rgba8();
            if (im.width() as usize, im.height() as usize) == (w, h) {
                return (im.pixels().map(|p| px::premultiply(p[0], p[1], p[2], p[3])).collect(), w, h);
            }
        }
        let g = crate::engine::ops::procedural(w as u32, h as u32);
        let mut buf = vec![0u32; w * h];
        g.read_region_clamped(0, 0, w, h, &mut buf);
        (buf, w, h)
    }

    #[test]
    fn sketch1_tiles_match_whole_image() {
        let (src, w, h) = source();
        let at = |x: i32, y: i32| src[y.clamp(0, h as i32 - 1) as usize * w + x.clamp(0, w as i32 - 1) as usize];
        for g in EFFECTS {
            let p = defaults(g);
            let mut full = src.clone();
            g.apply(p, [0, 0, 0], [255, 255, 255], &mut full, w, h, 0, (0, 0));
            for &(x0, y0, cw, ch) in &[(300i32, 200i32, 96usize, 80usize), (611, 37, 70, 64), (12, 600, 64, 60)] {
                let pad = g.pad(p, 0);
                let (bw, bh) = (cw + 2 * pad, ch + 2 * pad);
                let (ox, oy) = (x0 - pad as i32, y0 - pad as i32);
                let mut tile: Vec<u32> = (0..bw * bh).map(|i| at(ox + (i % bw) as i32, oy + (i / bw) as i32)).collect();
                g.apply(p, [0, 0, 0], [255, 255, 255], &mut tile, bw, bh, 0, (ox, oy));
                let mut worst = 0i32;
                for y in 0..ch {
                    for x in 0..cw {
                        let a = px::unpack(tile[(y + pad) * bw + x + pad]);
                        let b = px::unpack(full[(y0 as usize + y) * w + x0 as usize + x]);
                        for k in 0..4 {
                            worst = worst.max((a[k] as i32 - b[k] as i32).abs());
                        }
                    }
                }
                assert!(worst <= 2, "{} tile at ({x0},{y0}) differs by {worst}", g.title());
            }
        }
    }

    #[test]
    fn sketch1_extreme_params_stay_finite() {
        let src = crate::engine::ops::procedural(160, 120);
        for g in EFFECTS {
            let ps = g.params();
            for pick in 0..3 {
                let mut p = [0f32; 3];
                for (k, d) in ps.iter().enumerate().take(3) {
                    p[k] = [d.min, d.max, d.default][(pick + k) % 3];
                }
                for level in [0u32, 1, 3] {
                    let (w, h) = (90usize, 70usize);
                    let mut buf = vec![0u32; w * h];
                    src.read_region_clamped(5, 5, w, h, &mut buf);
                    let mut img = Img::load(&buf, w, h);
                    let ctx = Ctx { w, h, s: (1u32 << level) as f32, origin: (5, 5), fg: [0.1, 0.2, 0.3], bg: [1.0, 0.9, 0.8], doc: None };
                    run(g, p, &mut img, &ctx);
                    assert!(img.c.iter().all(|c| c.iter().all(|v| v.is_finite())), "{} {p:?} level {level}", g.title());
                }
            }
        }
    }
}
