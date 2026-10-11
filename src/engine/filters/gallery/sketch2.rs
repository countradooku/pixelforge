//! Filter Gallery ▸ Sketch (part 2) effects, matched against Photoshop 2026
//! renders at the default settings (foreground black, background white).
//!
//! Photoshop's Sketch filters work on the mean of the three channels and turn
//! it into foreground/background through a steep linear ramp (a fixed number
//! of output levels per grey level), so most effects here are "grey →
//! smoothing → ramp" plus an effect-specific texture. Grain and fibres are
//! keyed on level-0 coordinates, so previews at mip levels and tiles match
//! the full-resolution result.

use super::*;

pub(super) fn run(g: G, p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    use G::*;
    match g {
        NotePaper => note_paper(p, img, ctx),
        Photocopy => photocopy(p, img, ctx),
        Plaster => plaster(p, img, ctx),
        Reticulation => reticulation(p, img, ctx),
        Stamp => stamp(p, img, ctx),
        TornEdges => torn_edges(p, img, ctx),
        WaterPaper => water_paper(p, img, ctx),
        _ => unreachable!("not in this group"),
    }
}

// ----- effects --------------------------------------------------------------------------

/// Note Paper: the grey thresholded at Image Balance is raised as white paper
/// over grey paper and lit from the top; the paper grain is noise smeared over
/// three pixels horizontally and embossed the same way.
fn note_paper(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = avg_gray(img);
    let t = (126.0 + (p[0] - 25.0) * 4.0) / 255.0;
    let m: Vec<f32> = l.iter().map(|&v| ((v - t) * (255.0 / 4.0) + 0.5).clamp(0.0, 1.0)).collect();
    let mb = blur(&m, w, h, ctx.px(0.5));
    let n3 = hbox3(&white(ctx, 151), w, h, ctx);
    let relief = p[2] * (220.0 / 255.0 / 11.0);
    let grain = p[1] * (0.415 / 10.0);
    duotone(img, ctx, |i| {
        let y = i / w;
        let (up, dn) = (if y > 0 { i - w } else { i }, if y + 1 < h { i + w } else { i });
        (170.0 + 72.0 * m[i]) / 255.0 + (mb[dn] - mb[up]) * relief + (n3[up] - n3[dn]) * grain
    });
}

/// Photocopy: ink where the 3 × 3-smoothed grey falls below its wide Gaussian
/// surround (sigma ≈ 0.91 × Detail); Darkness is the ink gain in output levels
/// per grey level. (Fitted to Photoshop's ramp to within half a level.)
fn photocopy(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = avg_gray(img);
    let fine = box3(&l, w, h, ctx);
    let wide = blur(&l, w, h, ctx.px(p[0] * 0.914));
    let gain = p[1];
    duotone(img, ctx, |i| 1.0 - gain * (wide[i] - fine[i] - 0.9 / 255.0));
}

/// Plaster: the grey thresholded at Image Balance gives raised (dark,
/// foreground) and sunken (light, background) plaster. The smoothed height is
/// lit from the light angle: diffuse light on the albedo plus a white sheen on
/// the slopes facing the light.
fn plaster(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = avg_gray(img);
    let g = blur(&l, w, h, ctx.px(p[1] * 0.3 + 1.0));
    let t = (86.0 + (p[0] - 20.0) * 4.0) / 255.0;
    let mw = 30.0 / 255.0;
    let m: Vec<f32> = g.iter().map(|&v| smooth(t - mw, t + mw, v)).collect();
    let (gx, gy) = sobel(&blur(&m, w, h, ctx.px(3.0)), w, h);
    let k = 1.5 / ctx.s;
    let (a, e) = (p[2].to_radians(), 20f32.to_radians());
    let lv = [a.cos() * e.cos(), -a.sin() * e.cos(), e.sin()];
    let hn = (lv[0] * lv[0] + lv[1] * lv[1] + (lv[2] + 1.0).powi(2)).sqrt();
    let hv = [lv[0] / hn, lv[1] / hn, (lv[2] + 1.0) / hn];
    // Photoshop lights wide sunken plateaus with a ramp across the whole
    // document toward the light, 255·(1 − y/H) for the default top light,
    // while busy texture stays at the mid-image level. Without the document
    // size every plateau sits at that level.
    let plateau = ctx.doc.map(|_| blur(&m, w, h, ctx.px(6.0)));
    let flat = |i: usize| -> f32 {
        let mid = 150.0 / 255.0;
        let level = match (ctx.doc, &plateau) {
            (Some((dw, dh)), Some(pl)) => {
                let (x, y) = ctx.g(i);
                let (c, s) = (a.cos(), a.sin());
                let half = (c.abs() * dw + s.abs() * dh) * 0.5;
                let ramp = (0.5 + 0.5 * ((x - dw * 0.5) * c - (y - dh * 0.5) * s) / half.max(1.0)).clamp(0.0, 1.0);
                mid + (ramp - mid) * smooth(0.85, 0.98, pl[i])
            }
            _ => mid,
        };
        level / lv[2]
    };
    let (shin, spec) = (3.0, 0.35);
    let sp0 = hv[2].powf(shin);
    duotone(img, ctx, |i| {
        // Raised = dark: the height is 1 − m, so the normal follows +∇m.
        let (nx, ny) = (gx[i] * k, gy[i] * k);
        let r = (nx * nx + ny * ny + 1.0).sqrt().recip();
        let n = [nx * r, ny * r, r];
        let diff = (n[0] * lv[0] + n[1] * lv[1] + n[2] * lv[2]).max(0.0);
        let sp = (n[0] * hv[0] + n[1] * hv[1] + n[2] * hv[2]).max(0.0).powf(shin);
        m[i] * diff * flat(i) + spec * (sp - sp0).max(0.0)
    });
}

/// Reticulation: maze-like band-pass grain (the clumped emulsion) over a soft
/// threshold of the grey, dark and dense in the shadows and lighter in the
/// highlights.
fn reticulation(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = avg_gray(img);
    let b = blur(&l, w, h, ctx.px(2.0));
    let n = white(ctx, 161);
    let (n1, n2) = (gauss(&n, w, h, ctx.px(1.0)), gauss(&n, w, h, ctx.px(1.6)));
    let d = p[0] / 12.0 * 25.0;
    // Foreground level darkens the shadows, background level lifts the lights.
    let bd = (0.14 + (40.0 - p[1]) * 0.004).clamp(0.0, 0.5);
    let bh = (0.78 + (p[2] - 5.0) * 0.006).clamp(0.5, 1.0);
    let (ad, ah) = (0.15, 0.26);
    duotone(img, ctx, |i| {
        let t = smooth(0.37, 0.55, b[i]);
        bd + (bh - bd) * t + (n1[i] - n2[i]) * d * (ad + (ah - ad) * t)
    });
}

/// Stamp: smoothed grey through a steep threshold ramp; a little smooth noise
/// gives Photoshop's scattered blobs.
fn stamp(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let n = white(ctx, 131);
    let l: Vec<f32> = avg_gray(img).iter().zip(&n).map(|(v, q)| v + q * (150.0 / 255.0)).collect();
    let b = blur(&l, w, h, ctx.px(p[1] * 0.5));
    let t = (116.0 + (p[0] - 25.0) * 4.0) / 255.0;
    duotone(img, ctx, |i| (b[i] - t) * 80.0 + 0.5);
}

/// Torn Edges: lightly smoothed grey plus clumpy grain through a threshold
/// ramp whose steepness follows Contrast.
fn torn_edges(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let l = avg_gray(img);
    let b = blur(&l, w, h, ctx.px(p[1] * 0.1));
    let n = blur(&white(ctx, 141), w, h, ctx.px(0.7));
    let t = (120.0 + (p[0] - 25.0) * 4.0) / 255.0;
    let gain = p[2] * 1.5;
    let amp = 16.0 / 255.0;
    duotone(img, ctx, |i| (b[i] + n[i] * amp - t) * gain + 0.5);
}

/// Water Paper: each 4 px paper cell soaks up colour along a horizontal or
/// vertical fibre, bright fibres run along the cell borders, and Brightness /
/// Contrast act on luminance and scale the channels together.
fn water_paper(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let fl = ctx.px(p[0] * (4.0 / 15.0));
    let cell_of = |i: usize| {
        let (x, y) = ctx.g(i);
        (((x.floor() as i32) + 1).div_euclid(4), ((y.floor() as i32) + 1).div_euclid(4))
    };
    // Mostly one direction per cell, a little of the other.
    let sel: Vec<f32> = (0..w * h)
        .map(|i| {
            let (cx, cy) = cell_of(i);
            if hash01(cx, cy, 191) < 0.5 { 0.1 } else { 0.9 }
        })
        .collect();
    img.each_channel(|v| {
        let (a, b) = (hblur(v, w, h, fl), vblur(v, w, h, fl));
        (0..w * h).map(|i| a[i] * sel[i] + b[i] * (1.0 - sel[i])).collect()
    });
    if ctx.s < 2.0 {
        img.each_channel(|v| gauss(&cells4(v, w, h, ctx), w, h, 0.5));
    }
    let k = 0.25 + p[2] * 0.0215;
    let b = (p[1] - 50.0) * 3.0 / 255.0;
    let fib = fibers(ctx, p[0]);
    let fa = 0.08 / ctx.s;
    img.map(|i, c| {
        let lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
        let lo = ((lum - 0.5) * k + 0.5 + b).max(0.0);
        let g = lo / lum.max(1e-3) * (1.0 + fib[i] * fa);
        [c[0] * g, c[1] * g, c[2] * g]
    });
}

// ----- helpers --------------------------------------------------------------------------

/// Mean of the three channels (the grey Photoshop's Sketch filters use).
fn avg_gray(img: &Img) -> Vec<f32> {
    (0..img.w * img.h).map(|i| (img.c[0][i] + img.c[1][i] + img.c[2][i]) * (1.0 / 3.0)).collect()
}

/// Per-pixel white noise in −0.5..0.5 keyed on level-0 coordinates.
fn white(ctx: &Ctx, seed: u32) -> Vec<f32> {
    (0..ctx.w * ctx.h).map(|i| ctx.noise(i, seed) - 0.5).collect()
}

/// 3 × 3 box blur at level 0; the equivalent Gaussian at coarser levels.
fn box3(v: &[f32], w: usize, h: usize, ctx: &Ctx) -> Vec<f32> {
    if ctx.s > 1.0 {
        return blur(v, w, h, ctx.px(0.82));
    }
    vbox3(&hbox3(v, w, h, ctx), w, h)
}

/// Horizontal 3-pixel box at level 0 (the equivalent Gaussian above).
fn hbox3(v: &[f32], w: usize, h: usize, ctx: &Ctx) -> Vec<f32> {
    if ctx.s > 1.0 {
        return hblur(v, w, h, ctx.px(0.82));
    }
    let mut o = vec![0f32; w * h];
    for y in 0..h {
        let r = &v[y * w..(y + 1) * w];
        for x in 0..w {
            o[y * w + x] = (r[x.saturating_sub(1)] + r[x] + r[(x + 1).min(w - 1)]) * (1.0 / 3.0);
        }
    }
    o
}

/// Vertical 3-pixel box.
fn vbox3(v: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut o = vec![0f32; w * h];
    for y in 0..h {
        let (a, b, c) = (y.saturating_sub(1) * w, y * w, (y + 1).min(h - 1) * w);
        for x in 0..w {
            o[b + x] = (v[a + x] + v[b + x] + v[c + x]) * (1.0 / 3.0);
        }
    }
    o
}

/// Exact separable Gaussian (edge clamped) for small radii, where the
/// three-box approximation is too coarse.
fn gauss(v: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.2 {
        return v.to_vec();
    }
    let r = (sigma * 3.0).ceil() as usize;
    let k: Vec<f32> = (0..=2 * r).map(|i| (-((i as f32 - r as f32).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let ks: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|x| x / ks).collect();
    let mut t = vec![0f32; w * h];
    let mut row = vec![0f32; w + 2 * r];
    for y in 0..h {
        let src = &v[y * w..(y + 1) * w];
        row[r..r + w].copy_from_slice(src);
        row[..r].fill(src[0]);
        row[r + w..].fill(src[w - 1]);
        let dst = &mut t[y * w..(y + 1) * w];
        for (j, &kv) in k.iter().enumerate() {
            for (d, s) in dst.iter_mut().zip(&row[j..j + w]) {
                *d += kv * s;
            }
        }
    }
    let mut o = vec![0f32; w * h];
    for (j, &kv) in k.iter().enumerate() {
        for y in 0..h {
            let sy = (y + j).saturating_sub(r).min(h - 1);
            let (src, dst) = (&t[sy * w..(sy + 1) * w], &mut o[y * w..(y + 1) * w]);
            for (d, s) in dst.iter_mut().zip(src) {
                *d += kv * s;
            }
        }
    }
    o
}

/// Horizontal-only Gaussian (three box passes, edge clamped).
fn hblur(v: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.4 {
        return v.to_vec();
    }
    let boxes = super::super::blur::boxes_for_gauss(sigma);
    let mut a = v.to_vec();
    let mut t = vec![0f32; w];
    for &bx in &boxes {
        let r = (bx / 2) as isize;
        if r == 0 {
            continue;
        }
        let inv = 1.0 / (2 * r + 1) as f32;
        for y in 0..h {
            let row = &mut a[y * w..(y + 1) * w];
            let at = |i: isize| row[i.clamp(0, w as isize - 1) as usize];
            let mut s: f32 = (-r..=r).map(at).sum();
            for x in 0..w {
                t[x] = s * inv;
                s += at(x as isize + r + 1) - at(x as isize - r);
            }
            row.copy_from_slice(&t[..w]);
        }
    }
    a
}

/// Vertical-only Gaussian (three running-sum box passes down the columns,
/// edge clamped).
fn vblur(v: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.4 {
        return v.to_vec();
    }
    let boxes = super::super::blur::boxes_for_gauss(sigma);
    let mut a = v.to_vec();
    let mut col = vec![0f32; w];
    for &bx in &boxes {
        let r = (bx / 2) as isize;
        if r == 0 {
            continue;
        }
        let inv = 1.0 / (2 * r + 1) as f32;
        let rowi = |y: isize| y.clamp(0, h as isize - 1) as usize * w;
        col.fill(0.0);
        for y in -r..=r {
            let o = rowi(y);
            for (c, s) in col.iter_mut().zip(&a[o..o + w]) {
                *c += s;
            }
        }
        let src = a.clone();
        for y in 0..h {
            for (d, c) in a[y * w..(y + 1) * w].iter_mut().zip(&col) {
                *d = c * inv;
            }
            let (ad, sb) = (rowi(y as isize + r + 1), rowi(y as isize - r));
            for x in 0..w {
                col[x] += src[ad + x] - src[sb + x];
            }
        }
    }
    a
}

/// Averages over the 4 × 4 paper cells [4k − 1, 4k + 2] of level-0 space
/// (Photoshop's Water Paper grid phase).
fn cells4(v: &[f32], w: usize, h: usize, ctx: &Ctx) -> Vec<f32> {
    let cell = |o: i32, n: usize| -> Vec<i32> { (0..n).map(|i| (o + i as i32 + 1).div_euclid(4)).collect() };
    let (cx, cy) = (cell(ctx.origin.0, w), cell(ctx.origin.1, h));
    let mut o = vec![0f32; w * h];
    let mut y0 = 0;
    while y0 < h {
        let mut y1 = y0;
        while y1 < h && cy[y1] == cy[y0] {
            y1 += 1;
        }
        let mut x0 = 0;
        while x0 < w {
            let mut x1 = x0;
            while x1 < w && cx[x1] == cx[x0] {
                x1 += 1;
            }
            let mut s = 0.0;
            for y in y0..y1 {
                s += v[y * w + x0..y * w + x1].iter().sum::<f32>();
            }
            let m = s / ((y1 - y0) * (x1 - x0)) as f32;
            for y in y0..y1 {
                o[y * w + x0..y * w + x1].fill(m);
            }
            x0 = x1;
        }
        y0 = y1;
    }
    o
}

/// Horizontal and vertical fibres of random length on the cell borders
/// (level-0 space), 0..1.
fn fibers(ctx: &Ctx, len: f32) -> Vec<f32> {
    let len = len.max(1.0);
    (0..ctx.w * ctx.h)
        .map(|i| {
            let (x, y) = ctx.g(i);
            let (xi, yi) = (x.floor() as i32, y.floor() as i32);
            let mut v: f32 = 0.0;
            if (yi + 1).rem_euclid(4) == 0 {
                let j = (yi + 1).div_euclid(4);
                let seg = ((x + hash01(j, 0, 171) * len) / len).floor() as i32;
                if hash01(seg, j, 173) < 0.35 {
                    v = v.max(0.5 + 0.5 * hash01(seg, j, 175));
                }
            }
            if (xi + 1).rem_euclid(4) == 0 {
                let j = (xi + 1).div_euclid(4);
                let seg = ((y + hash01(0, j, 177) * len) / len).floor() as i32;
                if hash01(j, seg, 179) < 0.35 {
                    v = v.max(0.5 + 0.5 * hash01(j, seg, 181));
                }
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINE: [G; 7] = [G::NotePaper, G::Photocopy, G::Plaster, G::Reticulation, G::Stamp, G::TornEdges, G::WaterPaper];

    fn defaults(g: G) -> [f32; 3] {
        let mut p = [0f32; 3];
        for (k, d) in g.params().iter().enumerate().take(3) {
            p[k] = d.default;
        }
        p
    }

    /// A padded tile rendered on its own matches the whole-image render.
    #[test]
    fn sketch2_tiles_match_whole_image() {
        let (w, h) = (1024usize, 683usize);
        let src = crate::engine::ops::procedural(w as u32, h as u32);
        let mut full = vec![0u32; w * h];
        src.read_region(0, 0, w, h, &mut full);
        for g in MINE {
            let p = defaults(g);
            let mut whole = full.clone();
            g.apply(p, [0, 0, 0], [255, 255, 255], &mut whole, w, h, 0, (0, 0));
            let pad = g.pad(p, 0);
            let (x0, y0, tw, th) = (301i32, 203i32, 160usize, 120usize);
            let (bw, bh) = (tw + 2 * pad, th + 2 * pad);
            let mut tile = vec![0u32; bw * bh];
            let origin = (x0 - pad as i32, y0 - pad as i32);
            src.read_region_clamped(origin.0, origin.1, bw, bh, &mut tile);
            g.apply(p, [0, 0, 0], [255, 255, 255], &mut tile, bw, bh, 0, origin);
            for y in 0..th {
                for x in 0..tw {
                    let a = px::unpack(tile[(y + pad) * bw + x + pad]);
                    let b = px::unpack(whole[(y0 as usize + y) * w + x0 as usize + x]);
                    for k in 0..4 {
                        assert!((a[k] as i32 - b[k] as i32).abs() <= 2, "{} at ({x}, {y}): {a:?} vs {b:?}", g.title());
                    }
                }
            }
        }
    }

    /// Extreme parameters and mip levels stay finite and keep alpha.
    #[test]
    fn sketch2_extreme_params_are_sane() {
        let src = crate::engine::ops::procedural(120, 90);
        for g in MINE {
            let ps = g.params();
            for pick in [0usize, 1] {
                let mut p = [0f32; 3];
                for (k, d) in ps.iter().enumerate().take(3) {
                    p[k] = if pick == 0 { d.min } else { d.max };
                }
                for level in [0u32, 1, 3] {
                    let pad = g.pad(p, level);
                    let (w, h) = (40 + 2 * pad, 30 + 2 * pad);
                    let mut buf = vec![0u32; w * h];
                    src.read_region_clamped(-(pad as i32), -(pad as i32), w, h, &mut buf);
                    g.apply(p, [20, 30, 200], [250, 240, 200], &mut buf, w, h, level, (-(pad as i32), -(pad as i32)));
                    assert!(buf.iter().all(|&q| px::alpha(q) >= 250), "{} {p:?} level {level}", g.title());
                }
            }
        }
    }
}
