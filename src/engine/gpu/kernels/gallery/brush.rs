//! Brush Strokes effects and Stylize ▸ Glowing Edges (`filters::gallery::brush`).

use super::fx::*;
use super::{B, Pl};
use crate::engine::filters::gallery::G;

pub fn run(b: &mut B, g: G, p: [f32; 3], img: Img, _ctx: &Ctx) -> Option<Img> {
    use G::*;
    Some(match g {
        AccentedEdges => accented_edges(b, p, img),
        GlowingEdges => glowing_edges(b, p, img),
        AngledStrokes => angled_strokes(b, p, img),
        Crosshatch => crosshatch(b, p, img),
        DarkStrokes => dark_strokes(b, p, img),
        InkOutlines => ink_outlines(b, p, img),
        Spatter => spatter(b, p, img),
        SprayedStrokes => sprayed_strokes(b, p, img),
        SumiE => sumi_e(b, p, img),
        _ => return None,
    })
}

/// `brush::median_clamped`: float 3 x 3 median, else the 8-bit median.
fn median_clamped(b: &mut B, img: Img, r: usize) -> Img {
    match r {
        0 => img,
        1 => {
            let o: Vec<Pl> = img
                .c
                .iter()
                .map(|&c| {
                    let o = b.plane();
                    b.op("gallery", "median3f").ins(&[c]).outs(&[o]).run();
                    o
                })
                .collect();
            with_c(b, img, [o[0], o[1], o[2]])
        }
        _ => {
            let o = median_q(b, &img.c, r, 0);
            with_c(b, img, o)
        }
    }
}

fn odd_len(len: f32) -> usize {
    2 * (len * 0.5).floor().max(0.0) as usize + 1
}

/// `brush::range_disc` times `gain`.
fn range_disc(b: &mut B, v: Pl, r: f32, gain: f32) -> Pl {
    let ri = r.floor() as usize;
    let half: Vec<u32> = (0..=ri).map(|dy| ((r * r - (dy * dy) as f32).max(0.0)).sqrt().floor() as u32).collect();
    let t = b.tab_u32(&half);
    let o = b.plane();
    b.op("gallery", "range_disc").ins(&[v]).outs(&[o]).tabs(&[t]).i(&[ri as i32]).f(&[gain]).run();
    o
}

fn accented_edges(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let r = (b.px(p[2]) * 0.5).floor() as usize;
    let img = median_clamped(b, img, r);
    let v = vmax(b, &img);
    let rr = b.px(p[0] * 0.5 + 0.5).max(1.0);
    let e = range_disc(b, v, rr, 1.0);
    let gain = (p[1] - 25.0) / 4.0;
    let o: [Pl; 3] = b.planes();
    b.op("gallery_b", "accent").ins(&[img.c[0], img.c[1], img.c[2], v, e]).outs(&o).f(&[gain]).emit(img.a).run();
    b.free(&[v, e]);
    with_c(b, img, o)
}

fn glowing_edges(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let r = (b.px(p[2]) * 0.5).floor() as usize;
    let img = median_clamped(b, img, r);
    let rr = b.px(p[0] * 0.5 + 0.5).max(1.0);
    let gain = p[1] * 0.5;
    let o: Vec<Pl> = img.c.iter().map(|&c| range_disc(b, c, rr, gain)).collect();
    with_c(b, img, [o[0], o[1], o[2]])
}

/// `a0 > th ? a1 : a2`.
fn pick(b: &mut B, l: Pl, a: Pl, c: Pl, th: f32) -> Pl {
    let o = b.plane();
    b.op("gallery_b", "pick_th").ins(&[l, a, c]).outs(&[o]).f(&[th]).run();
    o
}

/// v + k (v - w), exactly.
fn sharpen(b: &mut B, v: Pl, w: Pl, k: f32) -> Pl {
    let o = b.plane();
    b.op("gallery_a", "unsharp").ins(&[v, w]).outs(&[o]).f(&[k]).run();
    o
}

fn angled_strokes(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let n = odd_len(b.px(p[1]));
    let l0 = lum(b, &img);
    let l = b.plane();
    b.op("gallery", "median3f").ins(&[l0]).outs(&[l]).run();
    b.free(&[l0]);
    let th = 136.0 / 255.0 + (50.0 - p[0]) / 50.0 * 0.6;
    let sharp = p[2] / 6.0;
    let sig = b.px(2.5);
    let da = dir_median3(b, &img.c, true, n);
    let dd = dir_median3(b, &img.c, false, n);
    let m: Vec<Pl> = (0..3).map(|k| pick(b, l, da[k], dd[k], th)).collect();
    b.free(&da);
    b.free(&dd);
    let m = [m[0], m[1], m[2]];
    let bl = blur3(b, &m, sig);
    let o: Vec<Pl> = (0..3).map(|k| sharpen(b, m[k], bl[k], sharp)).collect();
    b.free(&m);
    b.free(&bl);
    b.free(&[l]);
    with_c(b, img, [o[0], o[1], o[2]])
}

fn crosshatch(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let n = odd_len(b.px(p[0]));
    let sig = b.px(2.0);
    let l0 = lum(b, &img);
    let s1 = b.px(1.0);
    let l = blur(b, l0, s1);
    b.free(&[l0]);
    let k = p[1] / 6.0 * (0.5 + 0.5 * p[2]);
    let (ka, kb) = (3.0 * k, 2.0 * k);
    let da = dir_median3(b, &img.c, true, n);
    let dd = dir_median3(b, &img.c, false, n);
    let gas = blur3(b, &da, sig);
    let gds = blur3(b, &dd, sig);
    let o: Vec<Pl> = (0..3)
        .map(|ch| {
            let (a, d, ga, gd) = (da[ch], dd[ch], gas[ch], gds[ch]);
            let o = b.plane();
            b.op("gallery_b", "crosshatch").ins(&[a, ga, d, gd, l]).outs(&[o]).f(&[ka, kb, k * 5.0 / 255.0, 80.0 / 255.0, 140.0 / 255.0]).run();
            b.free(&[a, ga, d, gd]);
            o
        })
        .collect();
    b.free(&[l]);
    with_c(b, img, [o[0], o[1], o[2]])
}

/// `brush::ratio_tone` with one of the three curves (0 Dark Strokes, 1 Ink
/// Outlines, 2 Sumi-e).
fn ratio_tone(b: &mut B, img: Img, f: [Pl; 3], mode: i32, knots: &[(f32, f32)], fp: &[f32]) -> Img {
    let flat: Vec<f32> = knots.iter().flat_map(|k| [k.0, k.1]).collect();
    let t = b.tab_f32(if flat.is_empty() { &[0.0, 0.0] } else { &flat });
    let o: [Pl; 3] = b.planes();
    b.op("gallery_b", "ratio_tone").ins(&f).outs(&o).tabs(&[t]).i(&[mode, knots.len() as i32]).f(fp).emit(img.a).run();
    b.free(&f);
    with_c(b, img, o)
}

fn dark_strokes(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let n = odd_len(b.px(11.0));
    let sig = b.px(1.0);
    let l0 = lum(b, &img);
    let l = blur(b, l0, sig);
    b.free(&[l0]);
    let th = 130.0 / 255.0;
    let s3 = b.px(3.0);
    let dms = dir_median3(b, &img.c, false, n);
    let gs = blur3(b, &img.c, sig);
    let ms = blur3(b, &dms, sig);
    b.free(&dms);
    let mbs = blur3(b, &ms, s3);
    let f: Vec<Pl> = (0..3)
        .map(|k| {
            let m2 = sharpen(b, ms[k], mbs[k], 0.6);
            let o = pick(b, l, m2, gs[k], th);
            b.free(&[m2]);
            o
        })
        .collect();
    b.free(&gs);
    b.free(&ms);
    b.free(&mbs);
    b.free(&[l]);
    let shift = (p[0] - 5.0) * 8.0 / 255.0;
    let (bk, wk) = (p[1] / 6.0, p[2] / 2.0);
    const KNOTS: [(f32, f32); 9] = [(0.0, 0.0), (110.0, 0.0), (120.0, 13.0), (130.0, 51.0), (140.0, 99.0), (150.0, 145.0), (160.0, 205.0), (170.0, 245.0), (180.0, 252.0)];
    ratio_tone(b, img, [f[0], f[1], f[2]], 0, &KNOTS, &[shift, bk, wk])
}

fn ink_outlines(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let s1 = b.px(1.0).round().max(1.0) as usize;
    let n = odd_len(b.px(p[0] * 0.25));
    let s4 = b.px(4.0);
    let os: Vec<Pl> = img
        .c
        .iter()
        .map(|&c| {
            let o = b.plane();
            b.op("gallery_b", "ink_shift").ins(&[c]).outs(&[o]).i(&[s1 as i32]).run();
            if n > 1 {
                let m = b.plane();
                b.op("gallery_b", "dir_mean").ins(&[o]).outs(&[m]).i(&[n as i32]).f(&[n as f32]).run();
                b.free(&[o]);
                m
            } else {
                o
            }
        })
        .collect();
    let os = [os[0], os[1], os[2]];
    let bl = blur3(b, &os, s4);
    let f: Vec<Pl> = (0..3)
        .map(|k| {
            let r = b.plane();
            b.op("gallery_b", "lin2").ins(&[os[k], bl[k]]).outs(&[r]).run();
            r
        })
        .collect();
    b.free(&os);
    b.free(&bl);
    let (dk, lk) = (p[1] / 20.0, p[2] / 10.0);
    const KNOTS: [(f32, f32); 16] = [
        (0.0, 0.0),
        (40.0, 1.0),
        (60.0, 10.0),
        (70.0, 22.0),
        (80.0, 31.0),
        (90.0, 36.0),
        (100.0, 44.0),
        (110.0, 58.0),
        (120.0, 79.0),
        (130.0, 102.0),
        (140.0, 122.0),
        (150.0, 139.0),
        (160.0, 173.0),
        (170.0, 211.0),
        (180.0, 232.0),
        (255.0, 250.0),
    ];
    ratio_tone(b, img, [f[0], f[1], f[2]], 1, &KNOTS, &[0.0, dk, lk])
}

fn spatter(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let (dx, dy) = (b.plane(), b.plane());
    let cell = 0.5 + p[1] * 0.5;
    b.op("gallery_b", "spray").outs(&[dx, dy]).i(&[81]).f(&[0.5 * p[0], cell]).run();
    warp(b, img, dx, dy)
}

fn sprayed_strokes(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let dir = [(1, 1), (1, 0), (1, -1), (0, 1)][p[2] as usize % 4];
    let len = p[0].max(1.0);
    let (dx, dy) = (b.plane(), b.plane());
    b.op("gallery_b", "sprayed").outs(&[dx, dy]).i(&[dir.0, dir.1]).f(&[len, p[1]]).run();
    warp(b, img, dx, dy)
}

fn sumi_e(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let n = odd_len(b.px(p[0] * 1.5));
    let sig = b.px(1.0);
    let ds = dir_median3(b, &img.c, false, n);
    let f = blur3(b, &ds, sig);
    b.free(&ds);
    let c = 135.0 / 255.0 + (p[1] - 2.0) * 0.01;
    let hw = 0.5 / (1.0 + 0.27 * p[2]);
    ratio_tone(b, img, f, 2, &[], &[c - hw, c + hw])
}
