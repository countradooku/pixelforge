//! Artistic effects (`filters::gallery::artistic1` / `artistic2`).

use super::fx::*;
use super::{B, Pl, Shape};
use crate::engine::filters::gallery::G;

pub fn run(b: &mut B, g: G, p: [f32; 3], img: Img, ctx: &Ctx) -> Option<Img> {
    use G::*;
    Some(match g {
        ColoredPencil => colored_pencil(b, p, img, ctx),
        Cutout => cutout(b, p, img),
        DryBrush => dry_brush(b, p, img),
        FilmGrain => film_grain(b, p, img),
        Fresco => fresco(b, p, img),
        NeonGlow => neon_glow(b, p, img, ctx),
        PaintDaubs => paint_daubs(b, p, img),
        PaletteKnife => palette_knife(b, p, img),
        PlasticWrap => plastic_wrap(b, p, img),
        PosterEdges => poster_edges(b, p, img),
        RoughPastels => rough_pastels(b, p, img),
        SmudgeStick => smudge_stick(b, p, img),
        Sponge => sponge(b, p, img),
        Underpainting => underpainting(b, p, img),
        Watercolor => watercolor(b, p, img),
        _ => return None,
    })
}

/// A pointwise stage `entry` of the artistic module over the color planes
/// (plus `extra` inputs after them) into new color planes.
fn stage(b: &mut B, entry: &'static str, img: Img, extra: &[Pl], f: &[f32]) -> Img {
    let mut ins = img.c.to_vec();
    ins.extend_from_slice(extra);
    let o: [Pl; 3] = b.planes();
    let op = b.op("gallery_a", entry).ins(&ins).outs(&o).f(f);
    // Every stage but `affine` ends in `emit3`.
    if entry == "affine" { op.run() } else { op.emit(img.a).run() }
    with_c(b, img, o)
}

fn film_grain(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let k = (p[0] * 0.05).clamp(0.0, 1.0);
    let t = (140.0 - p[1] * 5.0) / 255.0;
    let s = p[2] * 0.64;
    stage(b, "film_grain", img, &[], &[k, t, s])
}

fn neon_glow(b: &mut B, p: [f32; 3], img: Img, ctx: &Ctx) -> Img {
    const DARK: [f32; 17] = [0.91, 0.91, 0.91, 0.91, 0.91, 0.87, 0.84, 0.78, 0.72, 0.635, 0.545, 0.448, 0.348, 0.254, 0.18, 0.12, 0.08];
    const TINT: [f32; 17] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.014, 0.089, 0.205, 0.28, 0.316, 0.329, 0.311, 0.253, 0.2, 0.15, 0.1];
    let l = lum(b, &img);
    let size = p[0];
    let sg = b.px(size.abs() * 0.6 + 0.2);
    let lb = blur(b, l, sg);
    let k = p[1] / 15.0;
    let mut t = DARK.to_vec();
    t.extend_from_slice(&TINT);
    let tab = b.tab_f32(&t);
    let o: [Pl; 3] = b.planes();
    b.op("gallery_a", "neon")
        .ins(&[l, lb])
        .outs(&o)
        .tabs(&[tab])
        .f(&[k, (size < 0.0) as u32 as f32, 0.0, 0.0, ctx.fg[0], ctx.fg[1], ctx.fg[2], 0.0, ctx.bg[0], ctx.bg[1], ctx.bg[2]])
        .emit(img.a)
        .run();
    b.free(&[l, lb]);
    with_c(b, img, o)
}

fn colored_pencil(b: &mut B, p: [f32; 3], img: Img, ctx: &Ctx) -> Img {
    let width = p[0].max(1.0);
    let paper = ((p[2] + 5.0) / 50.0).clamp(0.0, 1.0);
    let paper = [ctx.bg[0] * paper, ctx.bg[1] * paper, ctx.bg[2] * paper];
    let sig = b.px(1.0);
    let colb = blur3(b, &img.c, sig);
    let half = b.px(width * 2.5).round().max(1.0) as usize;
    let n = (2 * half + 1) as f32;
    let k = n.sqrt() / 0.288_675;
    let s1 = b.plane();
    b.op("gallery_a", "diag_streaks").outs(&[s1]).i(&[half as i32, 1, 3]).f(&[n, k]).shape(Shape::Diag).run();
    let s2 = b.plane();
    b.op("gallery_a", "diag_streaks").outs(&[s2]).i(&[half as i32, 0, 5]).f(&[n, k]).shape(Shape::Diag).run();
    let cov = 0.5 + 0.0575 * p[1];
    let cap: [f32; 3] = std::array::from_fn(|c| paper[c].max(0.6 * ctx.bg[c]));
    let o: [Pl; 3] = b.planes();
    b.op("gallery_a", "pencil")
        .ins(&[colb[0], colb[1], colb[2], s1, s2])
        .outs(&o)
        .f(&[cov, 0.0, 0.0, 0.0, paper[0], paper[1], paper[2], 0.0, cap[0], cap[1], cap[2]])
        .emit(img.a)
        .run();
    b.free(&colb);
    b.free(&[s1, s2]);
    with_c(b, img, o)
}

fn cutout(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let levels = p[0].round().clamp(2.0, 8.0) as usize;
    let smooth = 0.5 + 0.8 * p[1] / p[2].max(1.0);
    let l0 = lum(b, &img);
    let sg = b.px(smooth);
    let l = blur(b, l0, sg);
    b.free(&[l0]);
    let band = b.plane();
    b.op("gallery_a", "cutout_band").ins(&[l]).outs(&[band]).f(&[10.0 / 255.0, 240.0 / 255.0, levels as f32]).i(&[levels as i32]).run();
    b.free(&[l]);
    let sig = b.px(16.0);
    let out: [Pl; 3] = b.planes();
    for k in 0..levels {
        // The band's (r, g, b, count) planes blurred together.
        let m = b.plane4();
        b.op("gallery_a", "cutout_mask4").ins(&[band, img.c[0], img.c[1], img.c[2]]).wout(m).i(&[k as i32]).run();
        let s = if sig < 0.4 { m } else { blur4(b, m, sig) };
        b.op("gallery_a", "cutout_pick4").ins(&[band]).qins(&[s]).outs(&out).i(&[k as i32]).f(&[1e-6]).run();
        if s != m {
            b.free(&[s]);
        }
        b.free(&[m]);
    }
    b.free(&[band]);
    let aa = b.px(0.7);
    let soft = blur3(b, &out, aa);
    b.free(&out);
    with_c(b, img, soft)
}

fn dry_brush(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let r = p[0] + 1.0 + (8.0 - p[1]).max(0.0) * 0.25;
    let kr = b.px(r).round().max(1.0) as usize;
    let img = kuwahara(b, img, kr, 0);
    let tex = (p[2] - 1.0).max(0.0) * 0.04;
    if tex > 0.0 { stage(b, "tex_noise", img, &[], &[tex]) } else { img }
}

const FRESCO_TONE: [f32; 17] = [0.0, 0.03, 0.046, 0.056, 0.078, 0.142, 0.241, 0.318, 0.484, 0.64, 0.721, 0.804, 0.901, 0.963, 0.988, 1.0, 1.0];

fn fresco(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let l = lum(b, &img);
    let r = b.px(p[0] + 1.0).round().max(1.0) as usize;
    // Kuwahara of a copy: the original colors are rescaled at the end.
    let pk = pack(b, &img, 0);
    let kw = kuwahara_packed(b, pk, r, 0);
    b.free(&[pk]);
    let k_img = unpack(b, kw, 0);
    b.free(&[kw]);
    let lk = lum(b, &k_img);
    b.free(&k_img.c);
    let rim = 1.0 + (8.0 - p[1]) * 0.1;
    let (sp, sn) = (0.12 * rim, 0.16 * rim);
    let tex = (p[2] - 1.0).max(0.0) * 0.04;
    let tab = b.tab_f32(&FRESCO_TONE);
    let o: [Pl; 3] = b.planes();
    b.op("gallery_a", "fresco").ins(&[img.c[0], img.c[1], img.c[2], l, lk]).outs(&o).tabs(&[tab]).f(&[sp, sn, tex]).emit(k_img.a).run();
    b.free(&[l, lk]);
    b.free(&img.c);
    b.free(&[img.a]);
    Img { c: o, a: k_img.a }
}

fn paint_daubs(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let r = b.px(p[0] * 0.5).max(0.5);
    let amount = p[1] / 7.0;
    let sig = b.px(2.0);
    // `disc_quantile`'s disc and rank, as the CPU computes them.
    let q = 0.8f32;
    let ri = r.floor().max(0.0) as usize;
    let hw: Vec<u32> = (0..=2 * ri)
        .map(|j| {
            let dy = j as f32 - ri as f32;
            ((r * r - dy * dy).max(0.0) + 1e-3).sqrt().floor() as u32
        })
        .collect();
    let n: usize = hw.iter().map(|&x| 2 * x as usize + 1).sum();
    let above = (n - 1) - ((n - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize;
    // Column half-heights of the same disc, for the sliding histogram.
    let vh: Vec<u32> = (-(ri as i32)..=ri as i32).map(|dx| (0..=ri).filter(|&dy| hw[ri + dy] as i32 >= dx.abs()).max().unwrap_or(0) as u32).collect();
    let d: [Pl; 3] = b.planes();
    let q = b.plane();
    b.op("gallery", "quant3").ins(&img.c).vouts(&[q]).run();
    rank_slide(b, q, &d, ri, 3, 1, n - 1 - above, true, Some(&vh), 0);
    b.free(&[q]);
    let img = with_c(b, img, d);
    if amount <= 0.0 {
        return img;
    }
    let bl = blur3(b, &img.c, sig);
    let o: Vec<Pl> = (0..3)
        .map(|k| {
            let o = b.plane();
            b.op("gallery_a", "unsharp").ins(&[img.c[k], bl[k]]).outs(&[o]).f(&[amount]).run();
            o
        })
        .collect();
    b.free(&bl);
    with_c(b, img, [o[0], o[1], o[2]])
}

fn palette_knife(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let r = b.px(p[0] * (0.35 + 0.05 * p[1])).max(1.0) as usize;
    let m = median_q(b, &img.c, r, 0);
    let img = with_c(b, img, m);
    let kr = b.px(p[0] * 0.16).round() as usize;
    let img = kuwahara(b, img, kr, 0);
    if p[2] > 0.0 {
        let sg = b.px(p[2] * 0.5);
        let o = blur3(b, &img.c, sg);
        return with_c(b, img, o);
    }
    img
}

fn plastic_wrap(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let l = lum(b, &img);
    let sg = 1.0 + p[2] * 3.0 / 7.0;
    let s = b.px(sg);
    let hf = gauss(b, l, s);
    b.free(&[l]);
    let inv = 0.25 / (b.s * b.s);
    let g2 = b.plane();
    b.op("gallery_a", "grad2").ins(&[hf]).outs(&[g2]).f(&[inv]).run();
    b.free(&[hf]);
    let s4 = b.px(4.0);
    let m = gauss(b, g2, s4);
    let eps = 0.006f32;
    let a = (0.75 * p[0] / 15.0).min(1.0);
    let g0 = 0.5 * (0.5 + p[1] / 18.0);
    let k = 0.75;
    let out = stage(b, "plastic", img, &[g2, m], &[eps * eps, a, g0, k]);
    b.free(&[g2, m]);
    out
}

fn poster_edges(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let v = vmax(b, &img);
    let r1 = (p[0] * 0.5 + 1.0).round().max(1.0) as usize;
    let b0 = box_blur(b, v, 1, 1);
    let b1 = box_blur(b, v, r1, 2);
    b.free(&[v]);
    let gain = 12.0 * (0.5 + p[1] * 0.5);
    let step = 1.0 / (1u32 << (p[2].round().clamp(0.0, 6.0) as u32 + 1)) as f32;
    let (th, ink_step) = (0.5 / 255.0, 9.8 / 255.0);
    let out = stage(b, "poster_edges", img, &[b0, b1], &[step, th, gain, ink_step]);
    b.free(&[b0, b1]);
    out
}

/// `artistic2::weave_shade`: relief (about -1..1) of the procedural texture
/// lit from `angle` degrees.
fn weave_shade(b: &mut B, kind: usize, angle: f32) -> Pl {
    let t = b.plane();
    b.op("gallery_a", "weave").outs(&[t]).i(&[kind as i32]).run();
    let a = angle.to_radians();
    let (lx, ly) = (a.cos(), -a.sin());
    let k = 1.5 / (2.0 * b.s);
    let o = b.plane();
    b.op("gallery_a", "weave_grad").ins(&[t]).outs(&[o]).f(&[lx, ly, k]).run();
    b.free(&[t]);
    o
}

/// `artistic2::canvas_rows`.
fn canvas_rows(b: &mut B, img: Img, row: f32, seg: f32, disp: f32, shade: Pl, amp: f32) -> Img {
    let out = stage(b, "canvas_rows", img, &[shade], &[row, seg, disp, amp]);
    b.free(&[shade]);
    out
}

fn rough_pastels(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let (ga, gb) = (1.15f32, -18.0f32 / 255.0);
    let mut img = stage(b, "affine", img, &[], &[ga, gb]);
    let m = b.px(p[0] * 1.2).round() as usize;
    if m >= 1 {
        let mw = 0.35f32;
        let o: Vec<Pl> = img
            .c
            .iter()
            .map(|&c| {
                let d = b.plane();
                b.op("gallery_a", "diag_mean").ins(&[c]).outs(&[d]).i(&[m as i32]).f(&[1.0 / (m + 1) as f32]).shape(Shape::Diag).run();
                let o = b.plane();
                b.op("gallery_a", "mixw").ins(&[c, d]).outs(&[o]).f(&[1.0 - mw, mw]).run();
                b.free(&[d]);
                o
            })
            .collect();
        img = with_c(b, img, [o[0], o[1], o[2]]);
    }
    let kind = p[2].round().clamp(0.0, 3.0) as usize;
    let ws = weave_shade(b, kind, 270.0);
    let shade = b.plane();
    b.op("gallery_a", "pastel_shade").ins(&[ws]).outs(&[shade]).run();
    b.free(&[ws]);
    let mut img = canvas_rows(b, img, 3.0, 10.0, 0.8, shade, 12.0 / 255.0);
    let sa = 0.04 * p[1];
    if sa > 0.0 {
        let sl = 4.0 * p[0] + 1.0;
        let l = lum(b, &img);
        let s2 = b.px(2.0);
        let lb = blur(b, l, s2);
        let d0 = b.plane();
        b.op("gallery_a", "absdiff").ins(&[l, lb]).outs(&[d0]).run();
        b.free(&[l, lb]);
        let d = blur(b, d0, s2);
        b.free(&[d0]);
        img = stage(b, "pastel_streaks", img, &[d], &[sl, sa]);
        b.free(&[d]);
    }
    img
}

fn smudge_stick(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let v = vmax(b, &img);
    let half = b.px(p[0] * 1.5);
    let n = half.max(0.0).round() as i32;
    let a = 40f32.to_radians();
    let (dx, dy) = (a.cos(), a.sin());
    let offs: Vec<i32> = (-n..=n).flat_map(|t| [(t as f32 * dx).round() as i32, (t as f32 * dy).round() as i32]).collect();
    let cnt = (2 * n + 1) as usize;
    let inv = 1.0 / cnt as f32;
    let t = b.tab_i32(&offs);
    let (mean, min) = (b.plane(), b.plane());
    b.op("gallery_a", "line_filter").ins(&[v]).outs(&[mean, min]).tabs(&[t]).i(&[cnt as i32]).f(&[inv]).run();
    b.free(&[v]);
    let mw = 0.2f32;
    let t1 = (157.0 - p[1] * 4.0) / 255.0;
    let t0 = t1 - 27.0 / 255.0;
    let k = p[2] / 10.0;
    let out = stage(b, "smudge", img, &[mean, min], &[1.0 - mw, mw, t0, t1, k, 30.0 / 255.0]);
    b.free(&[mean, min]);
    out
}

fn sponge(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let r = b.px(p[0]).round().max(1.0) as usize;
    let e = r as u32 + 1;
    let img = with_border(b, img, e, |b, big| median_img(b, big, r, e));
    let sc = 1.0 + p[2];
    let d = 15.0 / 255.0 * p[1] / 12.0;
    stage(b, "sponge", img, &[], &[sc, d, 0.81, 24.0 / 255.0])
}

fn underpainting(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let sg = b.px(0.75 * p[0] + 0.5);
    let o: [Pl; 3] = if sg < 6.0 { [gauss(b, img.c[0], sg), gauss(b, img.c[1], sg), gauss(b, img.c[2], sg)] } else { blur3(b, &img.c, sg) };
    let img = with_c(b, img, o);
    let kind = p[2].round().clamp(0.0, 3.0) as usize;
    let amp = 2.5 / 255.0 * p[1] / 16.0;
    let shade = if kind == 0 {
        let s = b.plane();
        b.op("gallery_a", "row_cos").outs(&[s]).run();
        s
    } else {
        weave_shade(b, kind, 90.0)
    };
    canvas_rows(b, img, 3.0, 12.0, 2.5 * p[1] / 16.0, shade, amp)
}

fn watercolor(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let ua = 1.5f32;
    let o: Vec<Pl> = img
        .c
        .iter()
        .map(|&c| {
            let bl = box_blur(b, c, 2, 2);
            let o = b.plane();
            b.op("gallery_a", "unsharp").ins(&[c, bl]).outs(&[o]).f(&[ua]).run();
            b.free(&[bl]);
            o
        })
        .collect();
    let img = with_c(b, img, [o[0], o[1], o[2]]);
    let detail = (14.0 - p[0]) / 5.0;
    let (mr, kr) = (b.px(3.0 * detail).round().max(1.0) as usize, b.px(2.0 * detail).round().max(1.0) as usize);
    let e = (2 * mr + kr + 1) as u32;
    let img = with_border(b, img, e, |b, big| {
        let big = if mr >= 3 {
            let m = median_img(b, big, mr - 1, e);
            median_img(b, m, mr - 1, e)
        } else {
            median_img(b, big, mr, e)
        };
        kuwahara(b, big, kr, e)
    });
    let l0 = (71.0 + p[1] * 4.0) / 255.0;
    let k = 2.0 + p[2] * 0.5;
    let l = lum(b, &img);
    let out = stage(b, "watercolor", img, &[l], &[l0, k]);
    b.free(&[l]);
    out
}
