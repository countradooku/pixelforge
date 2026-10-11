//! Filter Gallery ▸ Artistic (part 2) effects: Plastic Wrap, Poster Edges,
//! Rough Pastels, Smudge Stick, Sponge, Underpainting and Watercolor.
//!
//! Each effect was rebuilt from measurements of Photoshop 2026's output at
//! the default settings (tone curves, filter scales, stroke angles and
//! lengths); the comments name what each stage reproduces.

use super::super::util::hash01;
use super::*;

pub(super) fn run(g: G, p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    use G::*;
    match g {
        PlasticWrap => plastic_wrap(p, img, ctx),
        PosterEdges => poster_edges(p, img, ctx),
        RoughPastels => rough_pastels(p, img, ctx),
        SmudgeStick => smudge_stick(p, img, ctx),
        Sponge => sponge(p, img, ctx),
        Underpainting => underpainting(p, img, ctx),
        Watercolor => watercolor(p, img, ctx),
        _ => unreachable!("not in this group"),
    }
}

// ----- helpers ------------------------------------------------------------------------

/// Exact separable Gaussian (edge clamped); `blur`'s box approximation is
/// too coarse at the small radii these effects use.
fn gauss(v: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.2 {
        return v.to_vec();
    }
    let r = (sigma * 3.0).ceil() as isize;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|x| *x /= s);
    let wi = w as isize;
    let mut t = vec![0f32; w * h];
    for y in 0..h {
        let row = &v[y * w..(y + 1) * w];
        let out = &mut t[y * w..(y + 1) * w];
        for x in 0..wi {
            let mut a = 0.0;
            if x >= r && x + r < wi {
                let base = (x - r) as usize;
                for (j, kk) in k.iter().enumerate() {
                    a += kk * row[base + j];
                }
            } else {
                for (j, kk) in k.iter().enumerate() {
                    a += kk * row[(x + j as isize - r).clamp(0, wi - 1) as usize];
                }
            }
            out[x as usize] = a;
        }
    }
    let mut o = vec![0f32; w * h];
    for y in 0..h as isize {
        let out = &mut o[y as usize * w..(y as usize + 1) * w];
        for (j, kk) in k.iter().enumerate() {
            let yy = (y + j as isize - r).clamp(0, h as isize - 1) as usize;
            for (o, s) in out.iter_mut().zip(&t[yy * w..(yy + 1) * w]) {
                *o += kk * s;
            }
        }
    }
    o
}

/// `passes` box blurs of radius `r` (edge clamped) of a plane of 8-bit
/// values in 0..1, summed in integers so the result does not depend on
/// where the buffer starts (tiles match the whole image exactly).
fn boxes(v: &[f32], w: usize, h: usize, r: usize, passes: usize) -> Vec<f32> {
    if r == 0 || passes == 0 {
        return v.to_vec();
    }
    let mut a: Vec<u64> = v.iter().map(|&x| (x * 255.0).round().max(0.0) as u64).collect();
    let ri = r as isize;
    let mut t = vec![0u64; w.max(h)];
    for _ in 0..passes {
        for y in 0..h {
            let row = &mut a[y * w..(y + 1) * w];
            let at = |i: isize| row[i.clamp(0, w as isize - 1) as usize];
            let mut s: u64 = (-ri..=ri).map(at).sum();
            for x in 0..w {
                t[x] = s;
                s = s + at(x as isize + ri + 1) - at(x as isize - ri);
            }
            row.copy_from_slice(&t[..w]);
        }
        let src = a.clone();
        let rowi = |y: isize| y.clamp(0, h as isize - 1) as usize * w;
        let mut col = vec![0u64; w];
        for y in -ri..=ri {
            let o = rowi(y);
            for x in 0..w {
                col[x] += src[o + x];
            }
        }
        for y in 0..h {
            a[y * w..(y + 1) * w].copy_from_slice(&col);
            let (ad, sb) = (rowi(y as isize + ri + 1), rowi(y as isize - ri));
            for x in 0..w {
                col[x] = col[x] + src[ad + x] - src[sb + x];
            }
        }
    }
    let n = ((2 * r + 1) as f64).powi(2 * passes as i32) * 255.0;
    a.iter().map(|&x| (x as f64 / n) as f32).collect()
}

/// Box blur of `passes` boxes of radius `r` level-0 pixels: exact at level
/// 0, the equal-variance Gaussian at coarser preview levels.
fn box_blur(v: &[f32], ctx: &Ctx, r: usize, passes: usize) -> Vec<f32> {
    if ctx.s <= 1.0 {
        return boxes(v, ctx.w, ctx.h, r, passes);
    }
    let var = passes as f32 * (((2 * r + 1) * (2 * r + 1)) as f32 - 1.0) / 12.0;
    gauss(v, ctx.w, ctx.h, ctx.px(var.sqrt()))
}

/// Runs `f` on a copy of the image extended by `b` edge-clamped pixels on
/// every side, then copies the interior back: rank and Kuwahara filters
/// leave a border unprocessed, which would otherwise show at image edges.
fn with_border(img: &mut Img, b: usize, f: impl FnOnce(&mut Img)) {
    let (w, h) = (img.w, img.h);
    let (bw, bh) = (w + 2 * b, h + 2 * b);
    let ext = |p: &[f32]| -> Vec<f32> {
        (0..bw * bh)
            .map(|i| {
                let x = (i % bw).saturating_sub(b).min(w - 1);
                let y = (i / bw).saturating_sub(b).min(h - 1);
                p[y * w + x]
            })
            .collect()
    };
    let mut big = Img { w: bw, h: bh, c: [ext(&img.c[0]), ext(&img.c[1]), ext(&img.c[2])], a: ext(&img.a) };
    f(&mut big);
    for k in 0..3 {
        for y in 0..h {
            img.c[k][y * w..(y + 1) * w].copy_from_slice(&big.c[k][(y + b) * bw + b..(y + b) * bw + b + w]);
        }
    }
}

fn vmax(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2])
}

/// Rescales a colour to a new HSV value, keeping hue and saturation.
fn with_value(c: [f32; 3], v: f32) -> [f32; 3] {
    let m = vmax(c);
    if m <= 1e-6 {
        return [v, v, v];
    }
    let k = v / m;
    [c[0] * k, c[1] * k, c[2] * k]
}

fn value_plane(img: &Img) -> Vec<f32> {
    (0..img.w * img.h).map(|i| vmax([img.c[0][i], img.c[1][i], img.c[2][i]])).collect()
}

/// Smooth value noise on the integer lattice (level-0 units).
fn vnoise(x: f32, y: f32, seed: u32) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (fx, fy) = (x - xi, y - yi);
    let (xi, yi) = (xi as i32, yi as i32);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (u, v) = (s(fx), s(fy));
    let a = hash01(xi, yi, seed);
    let b = hash01(xi + 1, yi, seed);
    let c = hash01(xi, yi + 1, seed);
    let d = hash01(xi + 1, yi + 1, seed);
    a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}

/// Mean and minimum of each plane along a segment of half-length `half`
/// buffer pixels at `angle` degrees (y down) through every pixel.
fn line_filter(planes: &[&[f32]], w: usize, h: usize, angle: f32, half: f32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n = half.max(0.0).round() as i32;
    line_filter_range(planes, w, h, angle, -n, n)
}

/// Like `line_filter`, over the samples `lo..=hi` steps along the line.
fn line_filter_range(planes: &[&[f32]], w: usize, h: usize, angle: f32, lo: i32, hi: i32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let a = angle.to_radians();
    let (dx, dy) = (a.cos(), a.sin());
    let offs: Vec<(isize, isize)> = (lo..=hi).map(|t| ((t as f32 * dx).round() as isize, (t as f32 * dy).round() as isize)).collect();
    let deltas: Vec<isize> = offs.iter().map(|&(ox, oy)| oy * w as isize + ox).collect();
    let (x0, x1) = (offs.iter().map(|o| o.0).min().unwrap_or(0), offs.iter().map(|o| o.0).max().unwrap_or(0));
    let (y0, y1) = (offs.iter().map(|o| o.1).min().unwrap_or(0), offs.iter().map(|o| o.1).max().unwrap_or(0));
    let inv = 1.0 / offs.len() as f32;
    let mut means = vec![vec![0f32; w * h]; planes.len()];
    let mut mins = vec![vec![0f32; w * h]; planes.len()];
    for (k, p) in planes.iter().enumerate() {
        let (mean, min) = (&mut means[k], &mut mins[k]);
        for y in 0..h as isize {
            let row_in = y + y0 >= 0 && y + y1 < h as isize;
            for x in 0..w as isize {
                let i = y as usize * w + x as usize;
                let (mut s, mut m) = (0f32, f32::MAX);
                if row_in && x + x0 >= 0 && x + x1 < w as isize {
                    for &d in &deltas {
                        let v = p[(i as isize + d) as usize];
                        s += v;
                        m = m.min(v);
                    }
                } else {
                    for &(ox, oy) in &offs {
                        let v = p[(y + oy).clamp(0, h as isize - 1) as usize * w + (x + ox).clamp(0, w as isize - 1) as usize];
                        s += v;
                        m = m.min(v);
                    }
                }
                mean[i] = s * inv;
                min[i] = m;
            }
        }
    }
    (means, mins)
}

/// Mean of `p` over the `m + 1` pixels (x - k, y + k), k = 0..=m (a
/// stroke reaching down-left), edge clamped; a running sum along each
/// anti-diagonal.
fn diag_mean(p: &[f32], w: usize, h: usize, m: usize) -> Vec<f32> {
    let at = |x: isize, y: isize| p[y.clamp(0, h as isize - 1) as usize * w + x.clamp(0, w as isize - 1) as usize];
    let mi = m as isize;
    let mut sum = vec![0f32; w * h];
    for y in (0..h as isize).rev() {
        for x in 0..w as isize {
            let i = y as usize * w + x as usize;
            sum[i] = if x == 0 || y == h as isize - 1 {
                (0..=mi).map(|k| at(x - k, y + k)).sum()
            } else {
                sum[i + w - 1] + p[i] - at(x - mi - 1, y + mi + 1)
            };
        }
    }
    let inv = 1.0 / (m + 1) as f32;
    sum.iter_mut().for_each(|v| *v *= inv);
    sum
}

/// Procedural texture height at level-0 point (x, y), roughly 0..1, for the
/// Texture choice of Rough Pastels / Underpainting: 0 canvas (diagonal
/// twill crossed by thread rows), 1 burlap, 2 brick, 3 sandstone.
fn weave(x: f32, y: f32, kind: usize) -> f32 {
    use std::f32::consts::PI;
    match kind {
        0 => {
            let r = (y / 3.0).floor() as i32;
            let seg = (x / 8.0 + hash01(r, 0, 3) * 5.0).floor() as i32;
            let jit = hash01(seg, r, 91) - 0.5;
            let rib = (2.0 * PI * (x * 0.577 + y) / 4.0).sin();
            let thread = (PI * (y / 3.0).fract()).sin();
            0.5 + 0.22 * rib + 0.25 * thread + 0.25 * jit
        }
        1 => {
            let (u, v) = (x / 6.0, y / 6.0);
            let (i, j) = (u.floor() as i32, v.floor() as i32);
            let (fu, fv) = (u - i as f32, v - j as f32);
            let prof = if (i + j).rem_euclid(2) == 0 { (PI * fv).sin() } else { (PI * fu).sin() };
            0.3 + 0.5 * prof * (0.7 + 0.3 * hash01(i, j, 91)) + 0.2 * super::super::util::fbm(x * 0.2, y * 0.2, 2, 3)
        }
        2 => {
            let bh = 12.0;
            let row = (y / bh).floor();
            let bx = x + if row as i32 % 2 == 0 { 0.0 } else { bh };
            let mortar = y.rem_euclid(bh) < 1.5 || bx.rem_euclid(2.0 * bh) < 1.5;
            if mortar { 0.15 } else { 0.6 + 0.2 * (hash01((bx / (2.0 * bh)) as i32, row as i32, 9) - 0.5) }
        }
        _ => super::super::util::fbm(x * 0.6, y * 0.6, 4, 11),
    }
}

/// Relief shading (about -1..1, 0 = flat) of the texture lit from `angle`
/// degrees (counter-clockwise from +x, y up): central differences of the
/// texture sampled at the buffer's pixel centres.
fn weave_shade(ctx: &Ctx, kind: usize, angle: f32) -> Vec<f32> {
    let (w, h) = (ctx.w, ctx.h);
    let a = angle.to_radians();
    let (lx, ly) = (a.cos(), -a.sin());
    let t: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = ctx.g(i);
            weave(x, y, kind)
        })
        .collect();
    let k = 1.5 / (2.0 * ctx.s);
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let gx = t[y * w + (x + 1).min(w - 1)] - t[y * w + x.saturating_sub(1)];
            let gy = t[(y + 1).min(h - 1) * w + x] - t[y.saturating_sub(1) * w + x];
            ((gx * lx + gy * ly) * k).clamp(-1.0, 1.0)
        })
        .collect()
}

/// Canvas texturing: thread rows `row` level-0 px tall are cut into
/// segments about `seg` px long, each shifting the image horizontally by up
/// to `disp` px (the jagged row edges of Photoshop's canvas effects), then
/// `shade` (scaled by `amp`) is added.
fn canvas_rows(img: &mut Img, ctx: &Ctx, row: f32, seg: f32, disp: f32, shade: &[f32], amp: f32) {
    let (w, h) = (ctx.w, ctx.h);
    let src = [img.c[0].clone(), img.c[1].clone(), img.c[2].clone()];
    let mut dx = vec![0f32; w];
    for y in 0..h {
        let (_, gy) = ctx.g(y * w);
        let r = (gy / row).floor();
        let jit = hash01(r as i32, 0, 5) * 7.0;
        for (x, d) in dx.iter_mut().enumerate() {
            let (gx, _) = ctx.g(y * w + x);
            *d = (vnoise(gx / seg + jit, r, 71) - 0.5) * 2.0 * disp / ctx.s;
        }
        for (k, p) in src.iter().enumerate() {
            let line = &p[y * w..(y + 1) * w];
            let out = &mut img.c[k][y * w..(y + 1) * w];
            for x in 0..w {
                let sx = (x as f32 + dx[x]).clamp(0.0, (w - 1) as f32);
                let x0 = sx as usize;
                let x1 = (x0 + 1).min(w - 1);
                let f = sx - x0 as f32;
                out[x] = line[x0] + (line[x1] - line[x0]) * f + shade[y * w + x] * amp;
            }
        }
    }
}

// ----- the effects --------------------------------------------------------------------

/// Plastic Wrap: a smoothed luminance height field is lit by a light at the
/// viewer: flat spots (small gradient, normalized by the local gradient
/// energy so textured areas also get highlights) catch a sheen: the pixel
/// is mixed toward the highlight level by 0.75 × its strength.
fn plastic_wrap(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = img.lum();
    let sg = 1.0 + p[2] * 3.0 / 7.0;
    let hf = gauss(&l, w, h, ctx.px(sg));
    let mut g2 = vec![0f32; w * h];
    let inv = 0.25 / (ctx.s * ctx.s);
    for y in 0..h {
        let (yu, yd) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for x in 0..w {
            let (xl, xr) = (x.saturating_sub(1), (x + 1).min(w - 1));
            let gx = hf[y * w + xr] - hf[y * w + xl];
            let gy = hf[yd * w + x] - hf[yu * w + x];
            g2[y * w + x] = (gx * gx + gy * gy) * inv;
        }
    }
    let m = gauss(&g2, w, h, ctx.px(4.0));
    let eps = 0.006;
    let a = (0.75 * p[0] / 15.0).min(1.0);
    let g0 = 0.5 * (0.5 + p[1] / 18.0);
    let k = 0.75;
    img.map(|i, c| {
        let gn = (g2[i] / (m[i] + eps * eps)).sqrt() / g0;
        let hh = a * (-gn * gn).exp();
        let t = k * hh;
        c.map(|v| v * (1.0 - t) + hh * t)
    });
}

/// Poster Edges: the HSV value is posterized (hue and saturation kept) and
/// pixels darker than their surroundings (a box difference of Gaussians on
/// the value) are inked toward black in quantized steps.
fn poster_edges(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let v = value_plane(img);
    let r1 = (p[0] * 0.5 + 1.0).round().max(1.0) as usize;
    let b0 = box_blur(&v, ctx, 1, 1);
    let b1 = box_blur(&v, ctx, r1, 2);
    let gain = 12.0 * (0.5 + p[1] * 0.5);
    let step = 1.0 / (1u32 << (p[2].round().clamp(0.0, 6.0) as u32 + 1)) as f32;
    let (th, ink_step) = (0.5 / 255.0, 9.8 / 255.0);
    img.map(|i, c| {
        // The value is posterized from its 3 × 3 mean, so flat areas keep
        // clean level boundaries.
        let vq = ((b0[i] / step).round() * step).min(1.0);
        let d = (b1[i] - b0[i] - th).max(0.0);
        let ink = ((d * gain) / ink_step).round() * ink_step;
        with_value(c, (vq - ink).max(0.0))
    });
}

/// Rough Pastels: a contrast boost (gain 1.15), colour dragged up and to
/// the right along the strokes, thin pastel streaks along the 45° diagonal
/// (strongest where the image has detail), and the chosen texture in
/// relief with jagged thread rows.
fn rough_pastels(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let (ga, gb) = (1.15, -18.0 / 255.0);
    img.map(|_, c| c.map(|v| v * ga + gb));
    // Each pixel mixes in the mean of the stroke behind it (lower left).
    let m = ctx.px(p[0] * 1.2).round() as usize;
    if m >= 1 {
        let mw = 0.35;
        for k in 0..3 {
            let d = diag_mean(&img.c[k], w, h, m);
            for (o, s) in img.c[k].iter_mut().zip(&d) {
                *o = *o * (1.0 - mw) + s * mw;
            }
        }
    }
    // Texture relief plus thread rows 3 px tall.
    let kind = p[2].round().clamp(0.0, 3.0) as usize;
    let shade: Vec<f32> = weave_shade(ctx, kind, 270.0)
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let (_, y) = ctx.g(i);
            0.5 * t + 0.5 * ((y / 3.0).fract() * std::f32::consts::TAU).cos()
        })
        .collect();
    canvas_rows(img, ctx, 3.0, 10.0, 0.8, &shade, 12.0 / 255.0);
    // Pastel streaks: 2 px wide segments along the "/" diagonal, one random
    // light/dark value each, `stroke length`-dependent length.
    let sa = 0.04 * p[1];
    if sa > 0.0 {
        let sl = 4.0 * p[0] + 1.0;
        let st: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ctx.g(i);
                let u = ((x + y) / 2.0).floor();
                let v = (x - y) / sl + hash01(u as i32, 0, 31) * 13.0;
                hash01(u as i32, v.floor() as i32, 29) - 0.5
            })
            .collect();
        let l = img.lum();
        let lb = blur(&l, w, h, ctx.px(2.0));
        let d: Vec<f32> = l.iter().zip(&lb).map(|(a, b)| (a - b).abs()).collect();
        let d = blur(&d, w, h, ctx.px(2.0));
        img.map(|i, c| {
            let t = st[i].signum() * smooth(0.2, 0.5, st[i].abs()) * sa * (0.3 + (d[i] * 8.0).min(1.0));
            c.map(|v| v + t)
        });
    }
}

/// Smudge Stick: the HSV value is replaced by its minimum along a short
/// stroke at 40° (dark smears), mapped through a curve that darkens the
/// mid-tones and blows the light tones out to full value; each pixel keeps
/// its hue and saturation, so light areas keep their texture.
fn smudge_stick(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let v = value_plane(img);
    let (mean, min) = line_filter(&[&v], ctx.w, ctx.h, 40.0, ctx.px(p[0] * 1.5));
    // Mostly the stroke minimum, a little of its mean.
    let mw = 0.2;
    let min: Vec<f32> = min[0].iter().zip(&mean[0]).map(|(a, b)| a * (1.0 - mw) + b * mw).collect();
    // Highlight area widens the blown-out range; intensity is its strength.
    let t1 = (157.0 - p[1] * 4.0) / 255.0;
    let t0 = t1 - 27.0 / 255.0;
    let k = p[2] / 10.0;
    let lo = |m: f32| 0.65 * m + 30.0 / 255.0;
    img.map(|i, c| {
        let m = min[i];
        let r = ((m - t0) / (t1 - t0)).clamp(0.0, 1.0);
        with_value(c, lo(m) + (1.0 - lo(m)) * r * r * k)
    });
}

/// Sponge: a median-smoothed image with lifted shadows, overlaid with a
/// two-tone blotch texture that raises or lowers the HSV value.
fn sponge(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let r = ctx.px(p[0]).round().max(1.0) as usize;
    with_border(img, r + 1, |img| median(img, r));
    let sc = 1.0 + p[2];
    let d = 15.0 / 255.0 * p[1] / 12.0;
    let (ga, gb) = (0.81, 24.0 / 255.0);
    img.map(|i, c| {
        let (x, y) = ctx.g(i);
        let n = super::super::util::fbm(x / sc, y / sc, 2, 21);
        let t = smooth(0.48, 0.52, n);
        with_value(c, (vmax(c) * ga + gb + d * (1.0 - 2.0 * t)).clamp(0.0, 1.0))
    });
}

/// Underpainting: a broad blur painted onto canvas: horizontal thread rows
/// shift the paint sideways and shade it.
fn underpainting(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let sg = ctx.px(0.75 * p[0] + 0.5);
    // Exact Gaussian for small brushes, the three-box approximation for
    // large ones (speed).
    img.each_channel(|v| if sg < 6.0 { gauss(v, w, h, sg) } else { blur(v, w, h, sg) });
    let kind = p[2].round().clamp(0.0, 3.0) as usize;
    let amp = 2.5 / 255.0 * p[1] / 16.0;
    let shade: Vec<f32> = if kind == 0 {
        (0..w * h)
            .map(|i| {
                let (_, y) = ctx.g(i);
                ((y / 3.0).fract() * std::f32::consts::TAU).cos()
            })
            .collect()
    } else {
        weave_shade(ctx, kind, 90.0)
    };
    canvas_rows(img, ctx, 3.0, 12.0, 2.5 * p[1] / 16.0, &shade, amp);
}

/// Watercolor: detail is sharpened, then flattened into blotches (median,
/// then Kuwahara), and a steep tone curve on luminance sends the shadows
/// to black and saturates the light washes.
fn watercolor(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let ua = 1.5;
    // Unsharp mask with a σ = 2 blur (two radius-2 boxes, exact integer
    // sums so tiles match the whole image).
    img.each_channel(|v| {
        let b = box_blur(v, ctx, 2, 2);
        v.iter().zip(&b).map(|(p, q)| p + ua * (p - q)).collect()
    });
    let detail = (14.0 - p[0]) / 5.0; // 1 at the default 9
    let (mr, kr) = (ctx.px(3.0 * detail).round().max(1.0) as usize, ctx.px(2.0 * detail).round().max(1.0) as usize);
    with_border(img, 2 * mr + kr + 1, |img| {
        // Two passes of the next smaller median: about as flat, much faster.
        if mr >= 3 {
            median(img, mr - 1);
            median(img, mr - 1);
        } else {
            median(img, mr);
        }
        kuwahara(img, kr);
    });
    let l0 = (71.0 + p[1] * 4.0) / 255.0;
    let k = 2.0 + p[2] * 0.5;
    let l = img.lum();
    img.map(|i, c| {
        let li = l[i].max(1e-4);
        let f = (k * (li - l0)).clamp(0.0, 1.0) / li;
        c.map(|v| v * f)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::px;

    const EFFECTS: [G; 7] = [G::PlasticWrap, G::PosterEdges, G::RoughPastels, G::SmudgeStick, G::Sponge, G::Underpainting, G::Watercolor];

    /// The 1024 × 683 Filter Gallery reference photo when present, else a
    /// procedural image of the same size.
    fn source() -> (Vec<u32>, usize, usize) {
        if let Ok(im) = image::open("/tmp/pf_ps_out/gallery_src.png") {
            let im = im.to_rgba8();
            let (w, h) = (im.width() as usize, im.height() as usize);
            return (im.pixels().map(|q| px::premultiply(q[0], q[1], q[2], q[3])).collect(), w, h);
        }
        let (w, h) = (1024, 683);
        let mut buf = vec![0u32; w * h];
        crate::engine::ops::procedural(w as u32, h as u32).read_region_clamped(0, 0, w, h, &mut buf);
        (buf, w, h)
    }

    fn param_sets(g: G) -> Vec<[f32; 3]> {
        let ps = g.params();
        let pick = |f: &dyn Fn(&GParam) -> f32| {
            let mut p = [0f32; 3];
            for (k, q) in ps.iter().enumerate().take(3) {
                p[k] = f(q);
            }
            p
        };
        vec![pick(&|q| q.default), pick(&|q| q.min), pick(&|q| q.max), pick(&|q| (q.min + q.max) * 0.5)]
    }

    #[test]
    fn artistic2_tiles_match_whole_image() {
        let (base, w, h) = source();
        for g in EFFECTS {
            for p in param_sets(g) {
                let mut whole = base.clone();
                g.apply(p, [0, 0, 0], [255, 255, 255], &mut whole, w, h, 0, (0, 0));
                let pad = g.pad(p, 0);
                for &(x0, y0, cw, ch) in &[(300usize, 200usize, 160usize, 120usize), (pad + 7, pad + 3, 97, 61), (w - pad - 130, h - pad - 90, 130, 90)] {
                    let (bw, bh) = (cw + 2 * pad, ch + 2 * pad);
                    let (ox, oy) = (x0 - pad, y0 - pad);
                    let mut buf = vec![0u32; bw * bh];
                    for y in 0..bh {
                        buf[y * bw..(y + 1) * bw].copy_from_slice(&base[(oy + y) * w + ox..(oy + y) * w + ox + bw]);
                    }
                    g.apply(p, [0, 0, 0], [255, 255, 255], &mut buf, bw, bh, 0, (ox as i32, oy as i32));
                    let mut worst = 0i32;
                    for y in 0..ch {
                        for x in 0..cw {
                            let a = px::unpack(buf[(y + pad) * bw + x + pad]);
                            let b = px::unpack(whole[(y0 + y) * w + x0 + x]);
                            for k in 0..4 {
                                worst = worst.max((a[k] as i32 - b[k] as i32).abs());
                            }
                        }
                    }
                    assert!(worst <= 2, "{} {p:?} tile at ({x0}, {y0}): max diff {worst}", g.title());
                }
            }
        }
    }

    #[test]
    fn artistic2_preview_levels_are_finite() {
        let (base, w, h) = source();
        for g in EFFECTS {
            for p in param_sets(g) {
                for level in [1u32, 2, 3] {
                    let s = 1usize << level;
                    let (lw, lh) = (w / s, h / s);
                    let mut buf: Vec<u32> = (0..lw * lh).map(|i| base[(i / lw) * s * w + (i % lw) * s]).collect();
                    g.apply(p, [0, 0, 0], [255, 255, 255], &mut buf, lw, lh, level, (0, 0));
                    assert!(buf.iter().all(|&q| px::alpha(q) == 255), "{} {p:?} level {level}", g.title());
                }
            }
        }
    }
}

