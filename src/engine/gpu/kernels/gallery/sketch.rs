//! Sketch effects (`filters::gallery::sketch1` / `sketch2`).

use super::fx::*;
use super::{B, Pl};
use crate::engine::filters::gallery::{G, hatch_sd};

/// Water Paper (the only Sketch effect that keeps the image's color).
pub fn run(b: &mut B, g: G, p: [f32; 3], img: Img, _ctx: &Ctx) -> Option<Img> {
    match g {
        G::WaterPaper => Some(water_paper(b, p, img)),
        _ => None,
    }
}

/// The gray Sketch effects: they only read the luminance (or the channel
/// mean) and alpha, so the load produces just those.
pub fn run_gray(b: &mut B, g: G, p: [f32; 3], ctx: &Ctx) -> Option<Img> {
    use G::*;
    let avg = matches!(g, NotePaper | Photocopy | Plaster | Reticulation | Stamp | TornEdges);
    let (l, a) = (b.plane(), b.plane());
    let t = b.by255();
    let un = b.unpremul();
    let f: &[f32] = if avg { &[1.0 / 3.0] } else { &[0.299, 0.587, 0.114] };
    b.op("gallery", "load_gray").src().outs(&[l, a]).tabs(&[un, t]).i(&[avg as i32]).f(f).run();
    Some(match g {
        BasRelief => bas_relief(b, p, l, a, ctx),
        ChalkCharcoal => chalk_charcoal(b, p, l, a, ctx),
        Charcoal => charcoal(b, p, l, a, ctx),
        Chrome => chrome(b, p, l, a, ctx),
        ConteCrayon => conte(b, p, l, a, ctx),
        GraphicPen => graphic_pen(b, p, l, a, ctx),
        HalftonePattern => halftone(b, p, l, a, ctx),
        NotePaper => note_paper(b, p, l, a, ctx),
        Photocopy => photocopy(b, p, l, a, ctx),
        Plaster => plaster(b, p, l, a, ctx),
        Reticulation => reticulation(b, p, l, a, ctx),
        Stamp => stamp(b, p, l, a, ctx),
        TornEdges => torn_edges(b, p, l, a, ctx),
        _ => return None,
    })
}

/// `duotone` of the gray effects: fg -> bg by t, alpha `a` (into the band
/// output when it is the last pass).
fn duo(b: &mut B, a: Pl, t: Pl, ctx: &Ctx) -> Img {
    let c: [Pl; 3] = b.planes();
    b.op("gallery_a", "duotone")
        .ins(&[t])
        .outs(&c)
        .f(&[ctx.fg[0], ctx.fg[1], ctx.fg[2], 0.0, ctx.bg[0], ctx.bg[1], ctx.bg[2]])
        .emit(a)
        .run();
    b.free(&[t]);
    Img { c, a }
}

/// A pointwise stage of the sketch module: `ins` -> one plane.
fn st1(b: &mut B, entry: &'static str, ins: &[Pl], f: &[f32]) -> Pl {
    let o = b.plane();
    b.op("gallery_s", entry).ins(ins).outs(&[o]).f(f).run();
    o
}

/// `sketch1::taps`: a level-0 stroke length in whole level pixels.
fn taps(b: &B, len: f32) -> usize {
    (b.px(len).round() as usize).max(1)
}

/// `sketch1::line_mean` (on a plane extended by `ext`).
fn line_mean(b: &mut B, v: Pl, n: usize, d: (i32, i32), ext: u32) -> Pl {
    let o = b.plane();
    let n = n.max(1);
    b.op("gallery_a", "line_mean").ins(&[v]).outs(&[o]).i(&[n as i32, d.0, d.1]).f(&[1.0 / n as f32]).ext(ext).run();
    o
}

fn graphic_pen(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let dir = [(1, -1), (1, 0), (1, 1), (0, 1)][(p[2].max(0.0) as usize).min(3)];
    let n = taps(b, p[0].clamp(1.0, 15.0));
    let m = (n / 2 + 1) as u32;
    let nz = noise(b, 131, 0.0, m);
    let lm = line_mean(b, nz, n, dir, m);
    b.free(&[nz]);
    let st = crop(b, lm, m);
    b.free(&[lm]);
    let amp = 0.112 / (0.2887 / (n as f32).sqrt());
    let thr = 0.455 + (p[1] - 50.0) / 100.0;
    let t = b.plane();
    b.op("gallery_a", "pen_threshold").ins(&[l, st]).outs(&[t]).f(&[amp, thr]).run();
    b.free(&[l, st]);
    duo(b, a, t, ctx)
}

fn bas_relief(b: &mut B, p: [f32; 3], l: Pl, alpha: Pl, ctx: &Ctx) -> Img {
    let k = p[1].max(0.5) / 3.0;
    let a = (p[2] + 135.0).to_radians();
    let (ux, uy) = (a.cos(), -a.sin());
    let (vx, vy) = (-uy, ux);
    let (fwd, back) = (b.px(1.406 * k), b.px(2.39 * k));
    let (s_par, s_perp) = (b.px(1.025 * k), b.px(1.62 * k));
    let c = (0.036 * (16.0 - p[0].clamp(1.0, 15.0)) / 3.0).max(1e-3);
    let (bl, u, taps): (Pl, (f32, f32), Vec<f32>) = if ux.abs() < 1e-4 || uy.abs() < 1e-4 {
        let bl = if ux.abs() < 1e-4 { gauss_xy(b, l, s_perp, s_par, 0) } else { gauss_xy(b, l, s_par, s_perp, 0) };
        (bl, (ux.round(), uy.round()), vec![0.0, 1.0])
    } else {
        let bl = gauss_xy(b, l, s_par, s_par, 0);
        let se = (s_perp * s_perp - s_par * s_par).sqrt();
        let t: Vec<(f32, f32)> = if se < 0.3 {
            vec![(0.0, 1.0)]
        } else {
            let r = (se * 2.5).ceil() as i32;
            let raw: Vec<(f32, f32)> = (-r..=r).map(|t| (t as f32, (-(t * t) as f32 / (2.0 * se * se)).exp())).collect();
            let s: f32 = raw.iter().map(|t| t.1).sum();
            raw.iter().map(|&(t, g)| (t, g / s)).collect()
        };
        (bl, (ux, uy), t.iter().flat_map(|t| [t.0, t.1]).collect())
    };
    b.free(&[l]);
    let tab = b.tab_f32(&taps);
    let t = b.plane();
    b.op("gallery_s", "bas_relief")
        .ins(&[bl])
        .outs(&[t])
        .tabs(&[tab])
        .i(&[(taps.len() / 2) as i32])
        .f(&[u.0, u.1, vx, vy, fwd, back, c])
        .run();
    b.free(&[bl]);
    duo(b, alpha, t, ctx)
}

/// `sketch1::dash_field`: (value, per-stroke random).
fn dash_field(b: &mut B, src: Pl, ang: f32, len: f32, wid: f32, seed: u32) -> (Pl, Pl) {
    let a = ang.to_radians();
    let (c, s) = (a.cos(), a.sin());
    let (cu, cv) = (len * 0.75, wid);
    let (il, iw) = (1.0 / len, 1.0 / wid);
    let (v, h) = (b.plane(), b.plane());
    b.op("gallery_s", "dash").ins(&[src]).outs(&[v, h]).i(&[seed as i32]).f(&[c, s, cu, cv, il, iw]).run();
    (v, h)
}

fn chalk_charcoal(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let l0 = l;
    let s1 = b.px(1.0);
    let l = gauss_xy(b, l0, s1, s1, 0);
    b.free(&[l0]);
    let (t5, t7) = (taps(b, 5.0), taps(b, 7.0));
    let src_d = line_mean(b, l, t5, (1, 1), 0);
    let src_l = line_mean(b, l, t7, (1, -1), 0);
    b.free(&[l]);
    let (dk, hd) = dash_field(b, src_d, 45.0, 7.0, 2.0, 211);
    let (lt, h2) = dash_field(b, src_l, -45.0, 10.0, 2.5, 223);
    b.free(&[src_d, src_l, h2]);
    let k = 0.6 + 0.4 * p[2].clamp(0.0, 5.0);
    let td = 0.48 + (p[0] - 6.0) * 0.02;
    let tb = 0.43 - (p[1] - 6.0) * 0.02;
    let t = st1(b, "chalk", &[dk, hd, lt], &[k, td, tb]);
    b.free(&[dk, hd, lt]);
    duo(b, a, t, ctx)
}

/// `sketch1::hatch`'s field (un-normalized).
fn hatch_field(b: &mut B, n: usize, dir: (i32, i32), pre: f32, bp: f32, seed: u32) -> Pl {
    let m = (n / 2 + (3.0 * pre).ceil() as usize + (3.0 * bp).ceil() as usize + 2) as u32;
    let nz0 = noise(b, seed, 0.0, m);
    let nz = gauss_xy(b, nz0, pre, pre, m);
    b.free(&[nz0]);
    let st = line_mean(b, nz, n, dir, m);
    b.free(&[nz]);
    let lo = gauss_xy(b, st, bp, bp, m);
    let hp = b.plane();
    b.op("gallery_s", "sub").ins(&[st, lo]).outs(&[hp]).ext(m).run();
    b.free(&[st, lo]);
    let o = crop(b, hp, m);
    b.free(&[hp]);
    o
}

fn charcoal(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let ns = taps(b, 13.0 + (5.0 - p[1].clamp(0.0, 5.0)) * 3.0);
    let s1 = line_mean(b, l, ns, (1, -1), 0);
    let s2 = line_mean(b, l, ns, (1, 1), 0);
    b.free(&[l]);
    let th = 0.5 + 0.5 * p[0].clamp(1.0, 7.0);
    let (pre, bp) = (b.px(th), b.px(2.5 * th));
    let (n1, n2) = (taps(b, 15.0), taps(b, 7.0));
    let f1 = hatch_field(b, n1, (1, -1), pre, bp, 121);
    let f2 = hatch_field(b, n2, (1, 1), pre, bp, 122);
    let sd1 = hatch_sd(b.s, n1, (1, -1), pre, bp, 121);
    let sd2 = hatch_sd(b.s, n2, (1, 1), pre, bp, 122);
    let q0 = 0.41 + (p[2] - 50.0) / 100.0 * 0.6;
    let t = st1(b, "charcoal", &[s1, s2, f1, f2], &[q0, sd1, sd2]);
    b.free(&[s1, s2, f1, f2]);
    duo(b, a, t, ctx)
}

fn chrome(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let sig0 = 0.5 + p[1].clamp(0.0, 10.0) * 0.93;
    let sig = b.px(sig0);
    let bl = gauss_xy(b, l, sig, sig, 0);
    let s12 = b.px(12.0);
    let lo = gauss_xy(b, l, s12, s12, 0);
    b.free(&[l]);
    let det = 0.5 + p[0].clamp(0.0, 10.0) / 8.0;
    let (k, fr) = (300.0 * det * (sig0 / 7.01) / b.s, 50.0 * det);
    let t0 = st1(b, "chrome", &[bl, lo], &[k, fr]);
    b.free(&[bl, lo]);
    let s7 = b.px(0.7);
    let t = gauss_xy(b, t0, s7, s7, 0);
    b.free(&[t0]);
    duo(b, a, t, ctx)
}

fn conte(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let kind = p[2].max(0.0) as usize;
    let cd = 0.519 + (p[0] - 11.0) * 0.025;
    let cb = 0.5625 - (p[1] - 7.0) * 0.025;
    let sub = (b.s as usize).clamp(1, 4);
    let step = b.s / sub as f32;
    let t = b.plane();
    b.op("gallery_s", "conte").ins(&[l]).outs(&[t]).i(&[kind as i32, sub as i32]).f(&[cd, cb, step]).run();
    b.free(&[l]);
    duo(b, a, t, ctx)
}

fn halftone(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let lb = if b.s <= 1.0 {
        let mut cur = l;
        for horiz in [1, 1, 0, 0] {
            let o = st1(b, "box3", &[cur], &[1.0 / 3.0]);
            b.ops.last_mut().unwrap().i[0] = horiz;
            if cur != l {
                b.free(&[cur]);
            }
            cur = o;
        }
        cur
    } else {
        let s = b.px(1.155);
        gauss_xy(b, l, s, s, 0)
    };
    b.free(&[l]);
    let period = 4.0 + 2.0 * p[0].clamp(1.0, 12.0);
    let kind = p[2] as usize;
    let kc = (1.0 + p[1].max(0.0) / 5.0) * 0.5;
    let (beta, v0, ca, cb) = (0.0433f32, 0.2088f32, 1.647f32, 2.151f32);
    let piv = (-ca + (ca * ca + 2.0 * cb).sqrt()) / (2.0 * cb);
    let t = b.plane();
    b.op("gallery_s", "halftone").ins(&[lb]).outs(&[t]).i(&[kind as i32]).f(&[period, kc, beta, v0, ca, cb, piv]).run();
    b.free(&[lb]);
    duo(b, a, t, ctx)
}

/// `sketch2::hbox3`: 3-tap horizontal box at level 0, else a horizontal blur.
fn hbox3(b: &mut B, v: Pl) -> Pl {
    if b.s > 1.0 {
        let s = b.px(0.82);
        return blur_axes(b, v, s, true, false, 0);
    }
    let o = st1(b, "box3", &[v], &[1.0 / 3.0]);
    b.ops.last_mut().unwrap().i[0] = 1;
    o
}

/// `sketch2::box3`.
fn box3(b: &mut B, v: Pl) -> Pl {
    if b.s > 1.0 {
        let s = b.px(0.82);
        return blur(b, v, s);
    }
    let h = hbox3(b, v);
    let o = st1(b, "box3", &[h], &[1.0 / 3.0]);
    b.free(&[h]);
    o
}

fn note_paper(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let t = (126.0 + (p[0] - 25.0) * 4.0) / 255.0;
    let m = st1(b, "note_mask", &[l], &[t, 255.0 / 4.0]);
    b.free(&[l]);
    let s = b.px(0.5);
    let mb = blur(b, m, s);
    let w = noise(b, 151, 0.5, 0);
    let n3 = hbox3(b, w);
    b.free(&[w]);
    let relief = p[2] * (220.0 / 255.0 / 11.0);
    let grain = p[1] * (0.415 / 10.0);
    let tt = st1(b, "note_paper", &[m, mb, n3], &[relief, grain]);
    b.free(&[m, mb, n3]);
    duo(b, a, tt, ctx)
}

fn photocopy(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let fine = box3(b, l);
    let s = b.px(p[0] * 0.914);
    let wide = blur(b, l, s);
    b.free(&[l]);
    let t = st1(b, "photocopy", &[wide, fine], &[p[1], 0.9 / 255.0]);
    b.free(&[wide, fine]);
    duo(b, a, t, ctx)
}

fn plaster(b: &mut B, p: [f32; 3], l: Pl, alpha: Pl, ctx: &Ctx) -> Img {
    let s = b.px(p[1] * 0.3 + 1.0);
    let g = blur(b, l, s);
    b.free(&[l]);
    let t = (86.0 + (p[0] - 20.0) * 4.0) / 255.0;
    let mw = 30.0 / 255.0;
    let m = st1(b, "smooth_th", &[g], &[t - mw, t + mw]);
    b.free(&[g]);
    let s3 = b.px(3.0);
    let mb = blur(b, m, s3);
    let (gx, gy) = (b.plane(), b.plane());
    b.op("gallery_s", "sobel").ins(&[mb]).outs(&[gx, gy]).run();
    b.free(&[mb]);
    let s6 = b.px(6.0);
    let pl = blur(b, m, s6);
    let k = 1.5 / b.s;
    let (a, e) = (p[2].to_radians(), 20f32.to_radians());
    let lv = [a.cos() * e.cos(), -a.sin() * e.cos(), e.sin()];
    let hn = (lv[0] * lv[0] + lv[1] * lv[1] + (lv[2] + 1.0).powi(2)).sqrt();
    let hv = [lv[0] / hn, lv[1] / hn, (lv[2] + 1.0) / hn];
    let (shin, spec) = (3.0f32, 0.35f32);
    let sp0 = hv[2].powf(shin);
    let (c, sn) = (a.cos(), a.sin());
    let (dw, dh) = ctx.doc;
    let half = (c.abs() * dw + sn.abs() * dh) * 0.5;
    let tt = st1(
        b,
        "plaster",
        &[m, gx, gy, pl],
        &[k, lv[0], lv[1], lv[2], hv[0], hv[1], hv[2], sp0, spec, c, sn, half.max(1.0), dw, dh, 150.0 / 255.0],
    );
    b.free(&[m, gx, gy, pl]);
    duo(b, alpha, tt, ctx)
}

fn reticulation(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let s2 = b.px(2.0);
    let bl = blur(b, l, s2);
    b.free(&[l]);
    let n = noise(b, 161, 0.5, 0);
    let (sa, sb) = (b.px(1.0), b.px(1.6));
    let n1 = gauss(b, n, sa);
    let n2 = gauss(b, n, sb);
    b.free(&[n]);
    let d = p[0] / 12.0 * 25.0;
    let bd = (0.14 + (40.0 - p[1]) * 0.004).clamp(0.0, 0.5);
    let bh = (0.78 + (p[2] - 5.0) * 0.006).clamp(0.5, 1.0);
    let t = st1(b, "reticulation", &[bl, n1, n2], &[d, bd, bh, 0.15, 0.26]);
    b.free(&[bl, n1, n2]);
    duo(b, a, t, ctx)
}

fn stamp(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let n = noise(b, 131, 0.5, 0);
    let g = l;
    let l = st1(b, "add_scaled", &[g, n], &[150.0 / 255.0]);
    b.free(&[n, g]);
    let s = b.px(p[1] * 0.5);
    let bl = blur(b, l, s);
    b.free(&[l]);
    let t = (116.0 + (p[0] - 25.0) * 4.0) / 255.0;
    let tt = st1(b, "ramp", &[bl, bl], &[t, 80.0, 0.0]);
    b.free(&[bl]);
    duo(b, a, tt, ctx)
}

fn torn_edges(b: &mut B, p: [f32; 3], l: Pl, a: Pl, ctx: &Ctx) -> Img {
    let s = b.px(p[1] * 0.1);
    let bl = blur(b, l, s);
    b.free(&[l]);
    let w = noise(b, 141, 0.5, 0);
    let s7 = b.px(0.7);
    let n = blur(b, w, s7);
    b.free(&[w]);
    let t = (120.0 + (p[0] - 25.0) * 4.0) / 255.0;
    let gain = p[2] * 1.5;
    let amp = 16.0 / 255.0;
    let tt = st1(b, "ramp", &[bl, n], &[t, gain, amp]);
    b.free(&[bl, n]);
    duo(b, a, tt, ctx)
}

fn water_paper(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let fl = b.px(p[0] * (4.0 / 15.0));
    let sel = b.plane();
    b.op("gallery_s", "wp_sel").outs(&[sel]).run();
    let o: Vec<Pl> = img
        .c
        .iter()
        .map(|&c| {
            let a = blur_axes(b, c, fl, true, false, 0);
            let v = blur_axes(b, c, fl, false, true, 0);
            let o = st1(b, "wp_mix", &[a, v, sel], &[]);
            b.free(&[a, v]);
            o
        })
        .collect();
    b.free(&[sel]);
    let mut img = with_c(b, img, [o[0], o[1], o[2]]);
    if b.s < 2.0 {
        let o: Vec<Pl> = img
            .c
            .iter()
            .map(|&c| {
                let m = st1(b, "cells4", &[c], &[]);
                let o = gauss(b, m, 0.5);
                b.free(&[m]);
                o
            })
            .collect();
        img = with_c(b, img, [o[0], o[1], o[2]]);
    }
    let k = 0.25 + p[2] * 0.0215;
    let bb = (p[1] - 50.0) * 3.0 / 255.0;
    let fa = 0.08 / b.s;
    let o: [Pl; 3] = b.planes();
    b.op("gallery_s", "water_paper").ins(&img.c).outs(&o).f(&[k, bb, fa, p[0].max(1.0)]).emit(img.a).run();
    with_c(b, img, o)
}
