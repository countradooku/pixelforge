//! Filter Gallery ▸ Artistic (part 1) effects.

use super::*;
use rayon::prelude::*;

pub(super) fn run(g: G, p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    use G::*;
    match g {
        ColoredPencil => colored_pencil(p, img, ctx),
        Cutout => cutout(p, img, ctx),
        DryBrush => dry_brush(p, img, ctx),
        FilmGrain => film_grain(p, img, ctx),
        Fresco => fresco(p, img, ctx),
        NeonGlow => neon_glow(p, img, ctx),
        PaintDaubs => paint_daubs(p, img, ctx),
        PaletteKnife => palette_knife(p, img, ctx),
        _ => unreachable!("not in this group"),
    }
}

/// Film Grain works on HSB brightness (max channel): the brightness is mixed
/// with uniform white noise (`Grain` × 5 % opacity), then brightness above a
/// threshold set by `Highlight area` is pushed up steeply (`Intensity`); the
/// colour is rescaled so hue and saturation are kept (until it clips).
fn film_grain(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let k = (p[0] * 0.05).clamp(0.0, 1.0);
    let t = (140.0 - p[1] * 5.0) / 255.0;
    let s = p[2] * 0.64;
    img.map(|i, c| {
        let v = c[0].max(c[1]).max(c[2]);
        let n = ctx.noise(i, 41);
        let vo = (v * (1.0 - k) + k * n + (v - t).max(0.0) * s).clamp(0.0, 1.0);
        if v <= 1e-4 { [vo; 3] } else { let g = vo / v; [c[0] * g, c[1] * g, c[2] * g] }
    });
}

/// Piecewise-linear lookup of `t` in 0..1 over evenly spaced knots.
fn lut(knots: &[f32], t: f32) -> f32 {
    let x = t.clamp(0.0, 1.0) * (knots.len() - 1) as f32;
    let i = (x as usize).min(knots.len() - 2);
    knots[i] + (knots[i + 1] - knots[i]) * (x - i as f32)
}

/// Neon Glow: the image becomes a foreground→background duotone of its
/// luminance, then the glow (the luminance blurred by `Glow size`) darkens
/// the bright areas and tints them with the glow colour (Photoshop's default
/// blue). A negative size glows from the dark areas instead.
fn neon_glow(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    const GLOW: [f32; 3] = [0.0, 0.0, 1.0];
    // Multipliers of the duotone (grey) and of the glow tint over the blurred
    // luminance 0, 1/16, …, 1, measured from Photoshop at the defaults.
    const DARK: [f32; 17] = [0.91, 0.91, 0.91, 0.91, 0.91, 0.87, 0.84, 0.78, 0.72, 0.635, 0.545, 0.448, 0.348, 0.254, 0.18, 0.12, 0.08];
    const TINT: [f32; 17] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.014, 0.089, 0.205, 0.28, 0.316, 0.329, 0.311, 0.253, 0.2, 0.15, 0.1];
    let (w, h) = (ctx.w, ctx.h);
    let l = img.lum();
    let size = p[0];
    let lb = blur(&l, w, h, ctx.px(size.abs() * 0.6 + 0.2));
    let k = p[1] / 15.0;
    let (fg, bg) = (ctx.fg, ctx.bg);
    img.map(|i, _| {
        let key = if size < 0.0 { 1.0 - lb[i] } else { lb[i] };
        let d = (1.0 - (1.0 - lut(&DARK, key)) * k).max(0.0);
        let t = lut(&TINT, key) * k;
        let base = mix(fg, bg, l[i]);
        [base[0] * d + GLOW[0] * t * l[i], base[1] * d + GLOW[1] * t * l[i], base[2] * d + GLOW[2] * t * l[i]]
    });
}

/// Cutout: the smoothed luminance is cut into `levels` bands and every band
/// is filled with the average colour of its own pixels nearby, which gives
/// flat paper-cutout regions with simplified edges.
fn cutout(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let n = w * h;
    let levels = p[0].round().clamp(2.0, 8.0) as usize;
    // Edge simplicity smooths the band outlines; fidelity keeps them closer.
    let l = blur(&img.lum(), w, h, ctx.px(cutout_smooth(p)));
    let band: Vec<u8> = l.iter().map(|&v| (((v - 10.0 / 255.0) / (240.0 / 255.0) * levels as f32).floor().max(0.0) as usize).min(levels - 1) as u8).collect();
    let sig = ctx.px(CUTOUT_REACH);
    let src = &img.c;
    // Per band: Gaussian-weighted sums of its own pixels' colours (planes 0–2)
    // and of its pixel count (plane 3), all blurred in parallel.
    let present: Vec<u8> = (0..levels as u8).filter(|k| band.contains(k)).collect();
    let sums: Vec<Vec<f32>> = present
        .iter()
        .flat_map(|&k| (0..4).map(move |c| (k, c)))
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(k, c)| {
            let v: Vec<f32> = (0..n).map(|i| if band[i] == k { if c < 3 { src[c][i] } else { 1.0 } } else { 0.0 }).collect();
            blur(&v, w, h, sig)
        })
        .collect();
    let mut out = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
    for (j, &k) in present.iter().enumerate() {
        let s = &sums[j * 4..j * 4 + 4];
        for i in 0..n {
            if band[i] == k {
                let d = 1.0 / s[3][i].max(1e-6);
                for c in 0..3 {
                    out[c][i] = s[c][i] * d;
                }
            }
        }
    }
    // Cut edges are slightly soft, as in Photoshop.
    let aa = ctx.px(0.7);
    let soft: Vec<Vec<f32>> = out.par_iter().map(|o| blur(o, w, h, aa)).collect();
    for (c, o) in soft.into_iter().enumerate() {
        img.c[c] = o;
    }
}

/// Level-0 blur of the luminance that sets Cutout's edge simplicity.
fn cutout_smooth(p: [f32; 3]) -> f32 {
    0.5 + 0.8 * p[1] / p[2].max(1.0)
}

/// Level-0 reach (Gaussian sigma) of a Cutout band's colour average.
const CUTOUT_REACH: f32 = 16.0;

/// Unit-variance diagonal stroke noise: white noise (hashed in level-0 space,
/// so tiles agree) averaged along 45° lines over `2·half+1` pixels; `up`
/// selects "/" strokes, otherwise "\\".
fn diag_streaks(ctx: &Ctx, half: usize, up: bool, seed: u32) -> Vec<f32> {
    let (w, h) = (ctx.w as i32, ctx.h as i32);
    let nz = |x: i32, y: i32| {
        let gx = ((ctx.origin.0 + x) as f32 + 0.5) * ctx.s;
        let gy = ((ctx.origin.1 + y) as f32 + 0.5) * ctx.s;
        super::hash01(gx.floor() as i32, gy.floor() as i32, seed)
    };
    let r = half as i32;
    let n = (2 * r + 1) as f32;
    let k = n.sqrt() / 0.288_675;
    let mut out = vec![0f32; (w * h) as usize];
    // "/" lines: x + y = d, step (+1, -1). "\\" lines: x - y = d, step (+1, +1).
    let (dmin, dmax) = if up { (0, w + h - 2) } else { (-(h - 1), w - 1) };
    for d in dmin..=dmax {
        let (x0, x1) = if up { ((d - (h - 1)).max(0), d.min(w - 1)) } else { (d.max(0), (d + h - 1).min(w - 1)) };
        if x0 > x1 {
            continue;
        }
        let yof = |x: i32| if up { d - x } else { x - d };
        let mut sum: f32 = (x0 - r..=x0 + r).map(|t| nz(t, yof(t))).sum();
        for x in x0..=x1 {
            out[(yof(x) * w + x) as usize] = (sum / n - 0.5) * k;
            sum += nz(x + r + 1, yof(x + r + 1)) - nz(x - r, yof(x - r));
        }
    }
    out
}

/// Colored Pencil: the image is redrawn with diagonal pencil hatching (mostly
/// "/" with lighter "\\" cross strokes) over grey paper of `Paper brightness`;
/// pencil does not lighten light paper, so highlights fall back to the paper.
fn colored_pencil(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let width = p[0].max(1.0);
    let paper = ((p[2] + 5.0) / 50.0).clamp(0.0, 1.0);
    let paper = [ctx.bg[0] * paper, ctx.bg[1] * paper, ctx.bg[2] * paper];
    let col: Vec<Vec<f32>> = (0..3).map(|c| blur(&img.c[c], w, h, ctx.px(1.0))).collect();
    let half = ctx.px(width * 2.5).round().max(1.0) as usize;
    let s1 = diag_streaks(ctx, half, true, 3);
    let s2 = diag_streaks(ctx, half, false, 5);
    let cov = 0.5 + 0.0575 * p[1];
    // Pencil is never lighter than light paper (highlights show the paper);
    // on dark paper it still reaches a mid-light grey.
    let cap: [f32; 3] = std::array::from_fn(|c| paper[c].max(0.6 * ctx.bg[c]));
    img.map(|i, _| {
        let a = (cov + 0.2 * s1[i] + 0.125 * s2[i]).clamp(0.0, 1.0);
        let k = [col[0][i].min(cap[0]), col[1][i].min(cap[1]), col[2][i].min(cap[2])];
        mix(paper, k, a)
    });
}

/// Palette Knife: a large square per-channel median (about the knife's
/// stroke size across, finer with more `Stroke detail`) flattens the paint
/// into slabs with stepped edges, and a small Kuwahara pass gives the slabs
/// crisp knife-cut borders; `Softness` blurs them.
fn palette_knife(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let r = ctx.px(p[0] * (0.35 + 0.05 * p[1])).max(1.0) as usize;
    median_clamped(img, r);
    kuwahara(img, ctx.px(p[0] * 0.16).round() as usize);
    if p[2] > 0.0 {
        img.each_channel(|v| blur(v, w, h, ctx.px(p[2] * 0.5)));
    }
}

/// Per-channel square median of radius `r` with edge-clamped borders (the
/// shared `median` leaves an `r`-pixel frame unfiltered, so it runs on a copy
/// padded by `r`).
fn median_clamped(img: &mut Img, r: usize) {
    if r == 0 {
        return;
    }
    let (w, h) = (img.w, img.h);
    let (pw, ph) = (w + 2 * r, h + 2 * r);
    let src = |i: usize| {
        let x = (i % pw).saturating_sub(r).min(w - 1);
        let y = (i / pw).saturating_sub(r).min(h - 1);
        y * w + x
    };
    let mut padded = Img { w: pw, h: ph, c: std::array::from_fn(|k| (0..pw * ph).map(|i| img.c[k][src(i)]).collect()), a: vec![1.0; pw * ph] };
    median(&mut padded, r);
    for k in 0..3 {
        for y in 0..h {
            img.c[k][y * w..(y + 1) * w].copy_from_slice(&padded.c[k][(y + r) * pw + r..(y + r) * pw + r + w]);
        }
    }
}

/// `q`-quantile over a disc of radius `r` (edge-clamped) of an f32 plane,
/// with a sliding 256-bin histogram per row (rows in parallel).
fn disc_quantile(v: &[f32], w: usize, h: usize, r: f32, q: f32) -> Vec<f32> {
    let ri = r.floor().max(0.0) as usize;
    let hw: Vec<usize> = (0..=2 * ri).map(|j| { let dy = j as f32 - ri as f32; ((r * r - dy * dy).max(0.0) + 1e-3).sqrt().floor() as usize }).collect();
    let n: usize = hw.iter().map(|&x| 2 * x + 1).sum();
    // Rank counted from the top: the output is the value with `above` samples above it.
    let above = (n - 1) - ((n - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize;
    // Edge-clamped u8 copy padded by `ri` on every side.
    let pw = w + 2 * ri;
    let b: Vec<u8> = (0..(h + 2 * ri) * pw)
        .map(|i| {
            let x = (i % pw) as isize - ri as isize;
            let y = (i / pw) as isize - ri as isize;
            let s = v[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
            (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        })
        .collect();
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let mut fine = [0u16; 256];
        // Window rows y..=y+2ri of the padded plane; the center column is x+ri.
        for (j, &hx) in hw.iter().enumerate() {
            let line = &b[(y + j) * pw..(y + j + 1) * pw];
            for &v in &line[ri - hx..=ri + hx] {
                fine[v as usize] += 1;
            }
        }
        // Running answer `f` and the number of samples strictly above it.
        let (mut f, mut hi) = (255usize, 0usize);
        for x in 0..w {
            if x > 0 {
                for (j, &hx) in hw.iter().enumerate() {
                    let line = &b[(y + j) * pw..(y + j + 1) * pw];
                    let o = line[x + ri - 1 - hx] as usize;
                    let a = line[x + ri + hx] as usize;
                    fine[o] -= 1;
                    fine[a] += 1;
                    hi = hi + (a > f) as usize - (o > f) as usize;
                }
            }
            loop {
                if hi > above {
                    f += 1;
                    hi -= fine[f] as usize;
                } else if hi + fine[f] as usize <= above {
                    hi += fine[f] as usize;
                    f -= 1;
                } else {
                    break;
                }
            }
            row[x] = f as f32 / 255.0;
        }
    });
    out
}

/// Paint Daubs (simple brush): every channel takes the light 80th percentile
/// of a round brush (`Brush size` across), which lifts the paint and lays it
/// in soft round daubs, then `Sharpness` adds an unsharp mask.
fn paint_daubs(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let r = ctx.px(p[0] * 0.5).max(0.5);
    let amount = p[1] / 7.0;
    let sig = ctx.px(2.0);
    img.each_channel(|v| {
        let d = disc_quantile(v, w, h, r, 0.8);
        if amount <= 0.0 {
            return d;
        }
        let b = blur(&d, w, h, sig);
        d.iter().zip(&b).map(|(&x, &y)| x + (x - y) * amount).collect()
    });
}

/// Fresco's tone curve over luminance 0, 1/16, …, 1 (fitted to Photoshop).
const FRESCO_TONE: [f32; 17] = [0.0, 0.03, 0.046, 0.056, 0.078, 0.142, 0.241, 0.318, 0.484, 0.64, 0.721, 0.804, 0.901, 0.963, 0.988, 1.0, 1.0];

/// Fresco: hard, contrasty dabs. The image is cut into Kuwahara dabs (`Brush
/// size`), the dab brightness goes through a steep tone curve, and pixels that
/// stand out from their dab get dark rims (narrower with more `Brush detail`);
/// `Texture` adds grain. As in Photoshop, every pixel keeps its own colour
/// ratios and is only rescaled to the new brightness.
fn fresco(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let l = img.lum();
    let orig = img.c.clone();
    kuwahara(img, ctx.px(p[0] + 1.0).round().max(1.0) as usize);
    let lk = img.lum();

    let rim = 1.0 + (8.0 - p[1]) * 0.1;
    let (sp, sn) = (0.12 * rim, 0.16 * rim);
    let tex = (p[2] - 1.0).max(0.0) * 0.04;
    img.map(|i, _| {
        let d = l[i] - lk[i];
        let g = if d > 0.0 { (-(d / sp).powi(2)).exp() } else { (-(d / sn).powi(2)).exp() };
        let t = lut(&FRESCO_TONE, lk[i]) * g + if tex > 0.0 { (ctx.noise(i, 77) - 0.5) * tex } else { 0.0 };
        // Photoshop keeps each pixel's own colour ratios and only rescales it.
        let k = t.max(0.0) / l[i].max(1e-3);
        [orig[0][i] * k, orig[1][i] * k, orig[2][i] * k]
    });
}

/// Dry Brush: Kuwahara dabs (`Brush size`, a little larger at low `Brush
/// detail`) with optional grain (`Texture` above 1).
fn dry_brush(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let r = p[0] + 1.0 + (8.0 - p[1]).max(0.0) * 0.25;
    kuwahara(img, ctx.px(r).round().max(1.0) as usize);
    let tex = (p[2] - 1.0).max(0.0) * 0.04;
    if tex > 0.0 {
        img.map(|i, c| {
            let t = (ctx.noise(i, 77) - 0.5) * tex;
            [c[0] + t, c[1] + t, c[2] + t]
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1024 × 683 test photo when it is available, else a procedural image.
    fn source() -> (Vec<u32>, usize, usize) {
        if let Ok(i) = image::open("/tmp/pf_ps_out/gallery_src.png") {
            let i = i.to_rgba8();
            let (w, h) = (i.width() as usize, i.height() as usize);
            return (i.pixels().map(|p| px::premultiply(p[0], p[1], p[2], p[3])).collect(), w, h);
        }
        let (w, h) = (1024, 683);
        let g = crate::engine::ops::procedural(w as u32, h as u32);
        let mut buf = vec![0u32; w * h];
        g.read_region_clamped(0, 0, w, h, &mut buf);
        (buf, w, h)
    }

    #[test]
    fn artistic1_tiles_match_whole_image() {
        let (src, w, h) = source();
        for g in [G::ColoredPencil, G::Cutout, G::DryBrush, G::FilmGrain, G::Fresco, G::NeonGlow, G::PaintDaubs, G::PaletteKnife] {
            let mut p = [0f32; 3];
            for (k, d) in g.params().iter().enumerate() {
                p[k] = d.default;
            }
            let mut full = src.clone();
            g.apply(p, [0, 0, 0], [255, 255, 255], &mut full, w, h, 0, (0, 0));
            let pad = g.pad(p, 0);
            let (x0, y0, cw, ch) = (300usize, 200usize, 256usize, 192usize);
            let (bx, by, bw, bh) = (x0 - pad, y0 - pad, cw + 2 * pad, ch + 2 * pad);
            let mut tile: Vec<u32> = (0..bw * bh).map(|i| src[(by + i / bw) * w + bx + i % bw]).collect();
            g.apply(p, [0, 0, 0], [255, 255, 255], &mut tile, bw, bh, 0, (bx as i32, by as i32));
            let mut worst = 0;
            for y in 0..ch {
                for x in 0..cw {
                    let a = px::unpack(full[(y0 + y) * w + x0 + x]);
                    let b = px::unpack(tile[(y + pad) * bw + x + pad]);
                    for k in 0..4 {
                        worst = worst.max((a[k] as i32 - b[k] as i32).abs());
                    }
                }
            }
            assert!(worst <= 2, "{}: tile differs from whole image by {worst}", g.title());
        }
    }
}
