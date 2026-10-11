//! Filter Gallery ▸ Brush Strokes (and Stylize ▸ Glowing Edges) effects.

use super::*;
use rayon::prelude::*;

pub(super) fn run(g: G, p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    use G::*;
    let (w, h) = (ctx.w, ctx.h);
    match g {
        AccentedEdges => {
            // Median-smoothed image whose HSB brightness is pushed by the
            // local brightness range: V' = V + (brightness - 25) / 4 * range(V),
            // hue and saturation kept (RGB scaled by V'/V).
            median_clamped(img, (ctx.px(p[2]) * 0.5).floor() as usize);
            let v: Vec<f32> = (0..w * h).map(|i| img.c[0][i].max(img.c[1][i]).max(img.c[2][i])).collect();
            let e = range_disc(&v, w, h, ctx.px(p[0] * 0.5 + 0.5).max(1.0));
            let gain = (p[1] - 25.0) / 4.0;
            img.map(|i, c| {
                let v0 = v[i];
                let v1 = (v0 + gain * e[i]).clamp(0.0, 1.0);
                if v0 > 1e-4 {
                    let k = v1 / v0;
                    [c[0] * k, c[1] * k, c[2] * k]
                } else {
                    [v1, v1, v1]
                }
            });
        }
        GlowingEdges => {
            // Per channel: brightness / 2 x (max - min) of the median-smoothed image.
            median_clamped(img, (ctx.px(p[2]) * 0.5).floor() as usize);
            let r = ctx.px(p[0] * 0.5 + 0.5).max(1.0);
            let gain = p[1] * 0.5;
            let planes = &img.c;
            let out: Vec<Vec<f32>> = (0..3usize).into_par_iter().map(|k| range_disc(&planes[k], w, h, r).into_iter().map(|v| v * gain).collect()).collect();
            for (k, o) in out.into_iter().enumerate() {
                img.c[k] = o;
            }
        }
        AngledStrokes => {
            // Per-channel medians along both diagonals; light areas take the
            // "\\" strokes, dark areas the "/" strokes, then an unsharp mask.
            let n = odd_len(ctx.px(p[1]));
            let lum = median3(&img.lum(), w, h);
            let th = 136.0 / 255.0 + (50.0 - p[0]) / 50.0 * 0.6;
            let sharp = p[2] / 6.0;
            let sig = ctx.px(2.5);
            let planes = &img.c;
            let st: Vec<Vec<f32>> = (0..6usize).into_par_iter().map(|j| dir_median(&planes[j / 2], w, h, j % 2 == 0, n)).collect();
            let out: Vec<Vec<f32>> = (0..3usize)
                .into_par_iter()
                .map(|k| {
                    let (a, b) = (&st[2 * k], &st[2 * k + 1]);
                    let m: Vec<f32> = (0..w * h).map(|i| if lum[i] > th { a[i] } else { b[i] }).collect();
                    let bl = gblur(&m, w, h, sig);
                    m.iter().zip(&bl).map(|(v, b)| v + sharp * (v - b)).collect()
                })
                .collect();
            for (k, o) in out.into_iter().enumerate() {
                img.c[k] = o;
            }
        }
        Crosshatch => {
            // Hatching from the image itself: "\\" stroke detail everywhere and
            // "/" detail added in the darker areas, each a sharpened directional
            // median over a soft base.
            let n = odd_len(ctx.px(p[0]));
            let sig = ctx.px(2.0);
            let lum = gblur(&img.lum(), w, h, ctx.px(1.0));
            let k = p[1] / 6.0 * (0.5 + 0.5 * p[2]);
            let (ka, kb) = (3.0 * k, 2.0 * k);
            let planes = &img.c;
            let st: Vec<(Vec<f32>, Vec<f32>)> = (0..6usize)
                .into_par_iter()
                .map(|j| {
                    let m = dir_median(&planes[j / 2], w, h, j % 2 == 0, n);
                    let g = gblur(&m, w, h, sig);
                    (m, g)
                })
                .collect();
            let out: Vec<Vec<f32>> = (0..3usize)
                .into_par_iter()
                .map(|c| {
                    let ((a, ga), (b, gb)) = (&st[2 * c], &st[2 * c + 1]);
                    (0..w * h)
                        .map(|i| {
                            let dark = 1.0 - smooth(80.0 / 255.0, 140.0 / 255.0, lum[i]);
                            0.5 * (ga[i] + gb[i]) + ka * (a[i] - ga[i]) + kb * dark * (b[i] - gb[i]) - k * 5.0 / 255.0
                        })
                        .collect()
                })
                .collect();
            for (c, o) in out.into_iter().enumerate() {
                img.c[c] = o;
            }
        }
        DarkStrokes => {
            // Light areas: long "/" medians; dark areas: the softened image;
            // then a steep HSB-brightness curve (darks to black, lights to white).
            let n = odd_len(ctx.px(11.0));
            let sig = ctx.px(1.0);
            let lum = gblur(&img.lum(), w, h, sig);
            let th = 130.0 / 255.0;
            let planes = &img.c;
            let f: Vec<Vec<f32>> = (0..3usize)
                .into_par_iter()
                .map(|k| {
                    let g = gblur(&planes[k], w, h, sig);
                    let m = gblur(&dir_median(&planes[k], w, h, false, n), w, h, sig);
                    // A light unsharp mask keeps the stroke edges visible.
                    let mb = gblur(&m, w, h, ctx.px(3.0));
                    let m: Vec<f32> = m.iter().zip(&mb).map(|(a, b)| a + 0.6 * (a - b)).collect();
                    (0..w * h).map(|i| if lum[i] > th { m[i] } else { g[i] }).collect()
                })
                .collect();
            let shift = (p[0] - 5.0) * 8.0 / 255.0;
            let (bk, wk) = (p[1] / 6.0, p[2] / 2.0);
            const KNOTS: [(f32, f32); 9] = [(0.0, 0.0), (110.0, 0.0), (120.0, 13.0), (130.0, 51.0), (140.0, 99.0), (150.0, 145.0), (160.0, 205.0), (170.0, 245.0), (180.0, 252.0)];
            let curve = move |v: f32| {
                let x = (v - shift) * 255.0;
                let y = pwl(&KNOTS, x) / 255.0;
                let v = v.clamp(0.0, 1.0);
                if y < v { v + (y - v) * bk } else { v + (y - v) * wk }.clamp(0.0, 1.0)
            };
            ratio_tone(img, &f, curve);
        }
        InkOutlines => {
            // Photoshop's result sits half a pixel up-left of the source:
            // average of the pixels at (x-1, y) and (x-1, y-1), optional "\\"
            // stroke smear for long strokes, local-contrast boost, then a steep
            // ink curve on HSB brightness (darks to black, lights lifted).
            let s1 = ctx.px(1.0).round().max(1.0) as usize;
            let n = odd_len(ctx.px(p[0] * 0.25));
            let planes = &img.c;
            let f: Vec<Vec<f32>> = (0..3usize)
                .into_par_iter()
                .map(|k| {
                    let v = &planes[k];
                    let mut o = vec![0f32; w * h];
                    for y in 0..h {
                        let (ya, yb) = (y, y.saturating_sub(s1));
                        for x in 0..w {
                            let xs = x.saturating_sub(s1);
                            o[y * w + x] = 0.5 * (v[ya * w + xs] + v[yb * w + xs]);
                        }
                    }
                    let o = if n > 1 { dir_mean(&o, w, h, n) } else { o };
                    // Strong local contrast: the ink picks up fine detail.
                    let b = gblur(&o, w, h, ctx.px(4.0));
                    o.iter().zip(&b).map(|(a, b)| 2.0 * a - b).collect()
                })
                .collect();
            let (dk, lk) = (p[1] / 20.0, p[2] / 10.0);
            const KNOTS: [(f32, f32); 16] = [
                (0.0, 0.0), (40.0, 1.0), (60.0, 10.0), (70.0, 22.0), (80.0, 31.0), (90.0, 36.0), (100.0, 44.0), (110.0, 58.0),
                (120.0, 79.0), (130.0, 102.0), (140.0, 122.0), (150.0, 139.0), (160.0, 173.0), (170.0, 211.0), (180.0, 232.0), (255.0, 250.0),
            ];
            ratio_tone(img, &f, move |v| {
                let y = pwl(&KNOTS, v * 250.0) / 255.0;
                let v = v.clamp(0.0, 1.0);
                if y < v { v + (y - v) * dk } else { v + (y - v) * lk }.clamp(0.0, 1.0)
            });
        }
        Spatter => {
            // Each small cell of pixels (size set by the smoothness) is fetched
            // from a random spot within the spray radius.
            let (dx, dy) = spray_field(ctx, p[0], 0.5 + p[1] * 0.5, 81);
            warp(img, &dx, &dy);
        }
        SprayedStrokes => {
            // Stroke segments (length = stroke length) along the chosen
            // direction each copy the source shifted along that direction by
            // a random amount within the spray radius.
            let dir = [(1, 1), (1, 0), (1, -1), (0, 1)][p[2] as usize % 4];
            let len = p[0].max(1.0);
            let n = w * h;
            let (mut dx, mut dy) = (vec![0f32; n], vec![0f32; n]);
            for i in 0..n {
                let (gx, gy) = ctx.g(i);
                let (x, y) = (gx.floor() as i32, gy.floor() as i32);
                // Line id across the direction, position along it.
                let (line, along) = match dir {
                    (1, 1) => (x - y, y),
                    (1, -1) => (x + y, y),
                    (1, 0) => (y, x),
                    _ => (x, y),
                };
                // Strokes are three lines wide.
                let band = line.div_euclid(3);
                let off = len * super::super::util::hash01(band, 0, 89);
                let seg = ((along as f32 + off) / len).floor() as i32;
                let t = ((super::super::util::hash01(band, seg, 91) - 0.5) * 1.2 * p[1]).round();
                dx[i] = ctx.px(t * dir.0 as f32);
                dy[i] = ctx.px(t * dir.1 as f32);
            }
            warp(img, &dx, &dy);
        }
        SumiE => {
            // Wide "/" ink strokes (directional median, softened), then a
            // near-threshold HSB-brightness curve.
            let n = odd_len(ctx.px(p[0] * 1.5));
            let sig = ctx.px(1.0);
            let planes = &img.c;
            let f: Vec<Vec<f32>> = (0..3usize).into_par_iter().map(|k| gblur(&dir_median(&planes[k], w, h, false, n), w, h, sig)).collect();
            let c = 135.0 / 255.0 + (p[1] - 2.0) * 0.01;
            let hw = 0.5 / (1.0 + 0.27 * p[2]);
            ratio_tone(img, &f, move |v| smooth(c - hw, c + hw, v));
        }
        _ => unreachable!("not in this group"),
    }
}

// ----- helpers ---------------------------------------------------------------------------

/// Square median of radius `r` with edge-clamped borders (the shared
/// `median` leaves an `r`-pixel frame untouched). Channels run in parallel;
/// radius 2 uses the exact 25-sample selection kernel on an edge-replicated
/// u8 plane.
fn median_clamped(img: &mut Img, r: usize) {
    if r == 0 {
        return;
    }
    let (w, h) = (img.w, img.h);
    if r == 1 || r == 2 {
        let planes = &img.c;
        let out: Vec<Vec<f32>> = (0..3usize)
            .into_par_iter()
            .map(|k| {
                if r == 1 {
                    return median3(&planes[k], w, h);
                }
                let (pw, ph) = (w + 4, h + 4);
                let mut pad = vec![0u8; pw * ph];
                for y in 0..ph {
                    let sy = (y as isize - 2).clamp(0, h as isize - 1) as usize;
                    for x in 0..pw {
                        let sx = (x as isize - 2).clamp(0, w as isize - 1) as usize;
                        pad[y * pw + x] = (planes[k][sy * w + sx].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    }
                }
                let mut o = vec![0u8; w * h];
                super::super::fast::median5(&pad, pw, ph, &mut o);
                o.into_iter().map(|v| v as f32 / 255.0).collect()
            })
            .collect();
        for (k, o) in out.into_iter().enumerate() {
            img.c[k] = o;
        }
        return;
    }
    let (pw, ph) = (w + 2 * r, h + 2 * r);
    let idx = |x: usize, y: usize| (y.clamp(r, r + h - 1) - r) * w + (x.clamp(r, r + w - 1) - r);
    let mut p = Img { w: pw, h: ph, c: [vec![0f32; pw * ph], vec![0f32; pw * ph], vec![0f32; pw * ph]], a: vec![1f32; pw * ph] };
    for y in 0..ph {
        for x in 0..pw {
            let j = idx(x, y);
            for k in 0..3 {
                p.c[k][y * pw + x] = img.c[k][j];
            }
        }
    }
    median(&mut p, r);
    for y in 0..h {
        for k in 0..3 {
            img.c[k][y * w..(y + 1) * w].copy_from_slice(&p.c[k][(y + r) * pw + r..(y + r) * pw + r + w]);
        }
    }
}

/// Max - min over a disc of radius `r` (edge clamped). Horizontal running
/// extrema per half-width, then one pass per disc row.
fn range_disc(v: &[f32], w: usize, h: usize, r: f32) -> Vec<f32> {
    let ri = r.floor() as usize;
    let half: Vec<usize> = (0..=ri).map(|dy| ((r * r - (dy * dy) as f32).max(0.0)).sqrt().floor() as usize).collect();
    let hmax_w = half[0];
    // hx[k] / hn[k]: horizontal max / min over [x - k, x + k].
    let mut hx: Vec<Vec<f32>> = vec![v.to_vec()];
    let mut hn: Vec<Vec<f32>> = vec![v.to_vec()];
    for k in 1..=hmax_w {
        let (px, pn) = (&hx[k - 1], &hn[k - 1]);
        let mut nx = px.clone();
        let mut nn = pn.clone();
        for y in 0..h {
            let row = &v[y * w..(y + 1) * w];
            for x in 0..w {
                let a = row[x.saturating_sub(k)];
                let b = row[(x + k).min(w - 1)];
                let i = y * w + x;
                nx[i] = nx[i].max(a).max(b);
                nn[i] = nn[i].min(a).min(b);
            }
        }
        hx.push(nx);
        hn.push(nn);
    }
    let mut mx = vec![f32::MIN; w * h];
    let mut mn = vec![f32::MAX; w * h];
    for dy in -(ri as isize)..=ri as isize {
        let k = half[dy.unsigned_abs()];
        let (sx, sn) = (&hx[k], &hn[k]);
        for y in 0..h {
            let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
            let (o, s) = (y * w, sy * w);
            for x in 0..w {
                mx[o + x] = mx[o + x].max(sx[s + x]);
                mn[o + x] = mn[o + x].min(sn[s + x]);
            }
        }
    }
    mx.iter().zip(&mn).map(|(a, b)| a - b).collect()
}

/// Odd sample count for a stroke of `len` buffer pixels (at least 1).
fn odd_len(len: f32) -> usize {
    2 * (len * 0.5).floor().max(0.0) as usize + 1
}

/// Median of `n` (odd) samples along each 45° diagonal: `down` = "\\"
/// (x - y constant), otherwise "/" (x + y constant); the ends of each
/// diagonal are replicated. Row-major sweep that keeps one sorted window per
/// diagonal (one removal + one insertion per pixel), so memory access stays
/// sequential.
fn dir_median(v: &[f32], w: usize, h: usize, down: bool, n: usize) -> Vec<f32> {
    let r = n / 2;
    if r == 0 {
        return v.to_vec();
    }
    let q: Vec<u8> = v.iter().map(|&x| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
    let mut wins = vec![0u8; (w + h) * n];
    let mut out = vec![0f32; w * h];
    let (wi, hi, ri) = (w as isize, h as isize, r as isize);
    for y in 0..hi {
        for x in 0..wi {
            // Position t along the diagonal: (x + sx * t, y + t).
            let (sx, d, tmin, tmax) = if down {
                (1, (x - y + hi - 1) as usize, -x.min(y), (wi - 1 - x).min(hi - 1 - y))
            } else {
                (-1, (x + y) as usize, -(wi - 1 - x).min(y), x.min(hi - 1 - y))
            };
            let at = |t: isize| {
                let t = t.clamp(tmin, tmax);
                q[((y + t) * wi + x + sx * t) as usize]
            };
            let win = &mut wins[d * n..(d + 1) * n];
            if tmin == 0 {
                for (k, o) in win.iter_mut().enumerate() {
                    *o = at(k as isize - ri);
                }
                win.sort_unstable();
            } else {
                let (sub, add) = (at(-ri - 1), at(ri));
                if sub != add {
                    let pos = win.binary_search(&sub).unwrap_or_else(|p| p);
                    let ins = win.partition_point(|&e| e < add);
                    if ins > pos {
                        win.copy_within(pos + 1..ins, pos);
                        win[ins - 1] = add;
                    } else {
                        win.copy_within(ins..pos, ins + 1);
                        win[ins] = add;
                    }
                }
            }
            out[(y * wi + x) as usize] = win[r] as f32 / 255.0;
        }
    }
    out
}

/// 3 x 3 median of a plane (edge clamped), min/max selection network.
fn median3(v: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let rows = [y.saturating_sub(1), y, (y + 1).min(h - 1)];
        for (x, o) in row.iter_mut().enumerate() {
            let cols = [x.saturating_sub(1), x, (x + 1).min(w - 1)];
            // Sort each column of three, then med3(max of lows, med of mids, min of highs).
            let mut lo = [0f32; 3];
            let mut mi = [0f32; 3];
            let mut hi = [0f32; 3];
            for (k, &cx) in cols.iter().enumerate() {
                let (a, b, c) = (v[rows[0] * w + cx], v[rows[1] * w + cx], v[rows[2] * w + cx]);
                let (a, b) = (a.min(b), a.max(b));
                let (b, c) = (b.min(c), b.max(c));
                let (a, b) = (a.min(b), a.max(b));
                lo[k] = a;
                mi[k] = b;
                hi[k] = c;
            }
            let l = lo[0].max(lo[1]).max(lo[2]);
            let hh = hi[0].min(hi[1]).min(hi[2]);
            let m = mi[0].min(mi[1]).max(mi[0].max(mi[1]).min(mi[2]));
            *o = l.min(m).max(l.max(m).min(hh));
        }
    });
    out
}

/// Piecewise-linear interpolation through `(x, y)` knots (clamped ends).
fn pwl(k: &[(f32, f32)], x: f32) -> f32 {
    if x <= k[0].0 {
        return k[0].1;
    }
    for s in k.windows(2) {
        if x <= s[1].0 {
            return s[0].1 + (s[1].1 - s[0].1) * (x - s[0].0) / (s[1].0 - s[0].0);
        }
    }
    k[k.len() - 1].1
}

/// Tone-maps the HSB brightness V = max(r, g, b) of the planes `f` through
/// `curve` and scales the colour by V'/V (hue and saturation kept).
fn ratio_tone(img: &mut Img, f: &[Vec<f32>], curve: impl Fn(f32) -> f32) {
    for i in 0..img.w * img.h {
        let c = [f[0][i], f[1][i], f[2][i]];
        let v = c[0].max(c[1]).max(c[2]);
        let v1 = curve(v);
        if v > 1e-4 {
            let k = v1 / v;
            for (ch, &x) in c.iter().enumerate() {
                img.c[ch][i] = x * k;
            }
        } else {
            for ch in 0..3 {
                img.c[ch][i] = v1;
            }
        }
    }
}

/// Random displacement field in buffer pixels: every `cell` x `cell` block
/// (level-0 px, global coordinates) moves by a whole-pixel offset drawn
/// uniformly from the disc of radius `r / 2` (level-0 px).
fn spray_field(ctx: &Ctx, r: f32, cell: f32, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let n = ctx.w * ctx.h;
    let (mut dx, mut dy) = (vec![0f32; n], vec![0f32; n]);
    for i in 0..n {
        let (gx, gy) = ctx.g(i);
        // Cell rows are offset randomly so the blocks do not line up.
        let cy = (gy / cell).floor() as i32;
        let cx = ((gx + cell * super::super::util::hash01(0, cy, seed + 2)) / cell).floor() as i32;
        let a = std::f32::consts::TAU * super::super::util::hash01(cx, cy, seed);
        let d = 0.5 * r * super::super::util::hash01(cx, cy, seed + 1).sqrt();
        dx[i] = ctx.px((d * a.cos()).round());
        dy[i] = ctx.px((d * a.sin()).round());
    }
    (dx, dy)
}

/// Mean of `n` (odd) samples along the "\\" diagonal (edge clamped).
fn dir_mean(v: &[f32], w: usize, h: usize, n: usize) -> Vec<f32> {
    let r = (n / 2) as isize;
    let mut out = vec![0f32; w * h];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let mut s = 0.0;
            for t in -r..=r {
                let (xx, yy) = ((x + t).clamp(0, w as isize - 1), (y + t).clamp(0, h as isize - 1));
                s += v[(yy * w as isize + xx) as usize];
            }
            out[(y * w as isize + x) as usize] = s / n as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EFFECTS: [G; 9] = [
        G::AccentedEdges,
        G::AngledStrokes,
        G::Crosshatch,
        G::DarkStrokes,
        G::InkOutlines,
        G::Spatter,
        G::SprayedStrokes,
        G::SumiE,
        G::GlowingEdges,
    ];

    /// The dog photo at 1024 x 683 (Photoshop reference size); a procedural
    /// image when the test photo is missing.
    fn source() -> (Vec<u32>, usize, usize) {
        let (w, h) = (1024usize, 683usize);
        if let Ok(img) = image::open("testdata/photos/dog_grass.jpg") {
            let img = image::imageops::resize(&img.to_rgba8(), w as u32, h as u32, image::imageops::FilterType::Triangle);
            return (img.pixels().map(|p| px::premultiply(p[0], p[1], p[2], 255)).collect(), w, h);
        }
        let g = crate::engine::ops::procedural(w as u32, h as u32);
        let mut buf = vec![0u32; w * h];
        g.read_region_clamped(0, 0, w, h, &mut buf);
        (buf, w, h)
    }

    fn defaults(g: G) -> [f32; 3] {
        let mut p = [0f32; 3];
        for (k, d) in g.params().iter().enumerate() {
            p[k] = d.default;
        }
        p
    }

    /// Edge-clamped crop of `src` (as the tile renderer reads it).
    fn crop(src: &[u32], w: usize, h: usize, x0: i32, y0: i32, cw: usize, ch: usize) -> Vec<u32> {
        let mut out = vec![0u32; cw * ch];
        for y in 0..ch {
            let sy = (y0 + y as i32).clamp(0, h as i32 - 1) as usize;
            for x in 0..cw {
                let sx = (x0 + x as i32).clamp(0, w as i32 - 1) as usize;
                out[y * cw + x] = src[sy * w + sx];
            }
        }
        out
    }

    fn check_tile(g: G, p: [f32; 3], src: &[u32], w: usize, h: usize, level: u32) {
        let mut whole = src.to_vec();
        g.apply(p, [0, 0, 0], [255, 255, 255], &mut whole, w, h, level, (0, 0));
        // Two tiles away from the image border (at the border the tile's
        // clamped padding and the whole-image per-stage clamping legitimately
        // differ by a few pixels, as for every gallery effect).
        for &(x0, y0, cw, ch) in &[(w as i32 / 3, h as i32 / 3, 160usize, 120usize), (w as i32 / 2 + 37, h as i32 / 2 + 11, 96, 80)] {
            let pad = g.pad(p, level);
            let (tw, th) = (cw + 2 * pad, ch + 2 * pad);
            let (ox, oy) = (x0 - pad as i32, y0 - pad as i32);
            let mut tile = crop(src, w, h, ox, oy, tw, th);
            g.apply(p, [0, 0, 0], [255, 255, 255], &mut tile, tw, th, level, (ox, oy));
            let mut worst = 0i32;
            for y in 0..ch {
                for x in 0..cw {
                    let a = px::unpack(tile[(y + pad) * tw + x + pad]);
                    let b = px::unpack(whole[(y0 as usize + y) * w + x0 as usize + x]);
                    for k in 0..4 {
                        worst = worst.max((a[k] as i32 - b[k] as i32).abs());
                    }
                }
            }
            assert!(worst <= 2, "{} level {level} tile at ({x0},{y0}): max diff {worst}", g.title());
        }
    }

    #[test]
    fn brush_tiles_match_whole_image() {
        let (src, w, h) = source();
        for g in EFFECTS {
            check_tile(g, defaults(g), &src, w, h, 0);
        }
        // Mip level 1 (the live preview) on a half-size image.
        let (hw, hh) = (w / 2, h / 2);
        let half: Vec<u32> = (0..hw * hh).map(|i| src[(i / hw) * 2 * w + (i % hw) * 2]).collect();
        for g in EFFECTS {
            check_tile(g, defaults(g), &half, hw, hh, 1);
        }
    }

    #[test]
    fn brush_non_default_params_stay_finite() {
        let (src, w, h) = source();
        let (cw, ch) = (200usize, 150usize);
        let part = crop(&src, w, h, 300, 200, cw, ch);
        for g in EFFECTS {
            let ps = g.params();
            for t in [0.0f32, 1.0] {
                let mut p = [0f32; 3];
                for (k, d) in ps.iter().enumerate() {
                    p[k] = d.min + (d.max - d.min) * t;
                }
                let mut buf = part.clone();
                g.apply(p, [0, 0, 0], [255, 255, 255], &mut buf, cw, ch, 0, (300, 200));
                assert!(buf.iter().all(|&q| px::alpha(q) == 255), "{} p {p:?}", g.title());
            }
        }
    }
}

/// Three-box Gaussian like the shared `blur` (same box sizes, edge clamped),
/// with one scratch buffer, rows in parallel and a running-sum column pass.
fn gblur(v: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.4 {
        return v.to_vec();
    }
    let mut a = v.to_vec();
    let mut t = vec![0f32; w * h];
    for b in super::super::blur::boxes_for_gauss(sigma) {
        let r = b / 2;
        if r == 0 {
            continue;
        }
        let inv = 1.0 / (2 * r + 1) as f32;
        t.par_chunks_mut(w).zip(a.par_chunks(w)).for_each(|(o, row)| {
            let at = |i: isize| row[i.clamp(0, w as isize - 1) as usize];
            let ri = r as isize;
            let mut s: f32 = (-ri..=ri).map(at).sum();
            for (x, d) in o.iter_mut().enumerate() {
                *d = s * inv;
                s += at(x as isize + ri + 1) - at(x as isize - ri);
            }
        });
        let rowi = |y: isize| y.clamp(0, h as isize - 1) as usize * w;
        let mut acc = vec![0f32; w];
        for y in -(r as isize)..=r as isize {
            let o = rowi(y);
            for (c, &q) in acc.iter_mut().zip(&t[o..o + w]) {
                *c += q;
            }
        }
        for y in 0..h {
            for (d, &c) in a[y * w..(y + 1) * w].iter_mut().zip(&acc) {
                *d = c * inv;
            }
            let (ad, sb) = (rowi(y as isize + r as isize + 1), rowi(y as isize - r as isize));
            for x in 0..w {
                acc[x] += t[ad + x] - t[sb + x];
            }
        }
    }
    a
}
