//! Distort and Texture effects (`filters::gallery::texture`).

use super::fx::*;
use super::{B, Pl};
use crate::engine::filters::gallery::G;

pub fn run(b: &mut B, g: G, p: [f32; 3], img: Img, ctx: &Ctx) -> Option<Img> {
    use G::*;
    Some(match g {
        DiffuseGlow => diffuse_glow(b, p, img, ctx),
        Glass | OceanRipple => glass_ripple(b, g, p, img),
        Craquelure => craquelure(b, p, img),
        Grain => grain(b, p, img),
        MosaicTiles => mosaic_tiles(b, p, img),
        Patchwork => patchwork(b, p, img)?,
        StainedGlass => stained_glass(b, p, img, ctx),
        Texturizer => texturizer(b, p, img),
        _ => return None,
    })
}

fn grain(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let k = p[0] / 100.0;
    let ct = p[1] / 100.0;
    let kind = p[2] as usize;
    let gain = 0.5 + ct;
    let o: [Pl; 3] = b.planes();
    b.op("gallery_a", "grain").ins(&img.c).outs(&o).i(&[kind as i32]).f(&[k, gain, k * 0.98]).emit(img.a).run();
    with_c(b, img, o)
}

fn diffuse_glow(b: &mut B, p: [f32; 3], img: Img, ctx: &Ctx) -> Img {
    let grain = p[0] * 0.0227;
    let amount = p[1] / 10.0;
    let thr = 0.15 + p[2] * 0.02;
    let l = lum(b, &img);
    let s = b.px(9.0);
    let lb = blur(b, l, s);
    b.free(&[l]);
    let o: [Pl; 3] = b.planes();
    b.op("gallery_a", "diffuse_glow")
        .ins(&[img.c[0], img.c[1], img.c[2], lb])
        .outs(&o)
        .f(&[grain, amount, thr, 0.0, ctx.bg[0], ctx.bg[1], ctx.bg[2]])
        .emit(img.a)
        .run();
    b.free(&[lb]);
    with_c(b, img, o)
}

fn glass_ripple(b: &mut B, g: G, p: [f32; 3], img: Img) -> Img {
    let glass = g == G::Glass;
    let (cell, mag, off, oct) = if glass {
        ((3.0 + p[1] * 1.2) * p[2] / 100.0, p[0] * 1.8, [0.0, 0.0], 2)
    } else {
        (3.0 + p[0] * 1.8, p[1] * 2.6, [-0.75 * p[1], -0.21 * p[1]], 3)
    };
    let cell = cell.max(0.5);
    let bump = b.plane();
    b.op("gallery_t", "bump").outs(&[bump]).i(&[oct, glass as i32]).f(&[cell]).run();
    let k = mag * cell / b.s;
    let (dx, dy) = (b.plane(), b.plane());
    b.op("gallery_t", "bump_grad").ins(&[bump]).outs(&[dx, dy]).f(&[k, off[0], off[1]]).run();
    b.free(&[bump]);
    warp(b, img, dx, dy)
}

/// `texture::Grooves::apply` with the distance field `dist`.
#[allow(clippy::too_many_arguments)]
fn grooves(b: &mut B, img: Img, dist: Pl, r: f32, side: f32, img_relief: f32, img_blur: f32, dark: f32, bright: f32, floor: f32, gain: f32) -> Img {
    let l0 = lum(b, &img);
    let s = b.px(img_blur);
    let l = blur(b, l0, s);
    b.free(&[l0]);
    let h = b.plane();
    b.op("gallery_t", "groove_height").ins(&[dist, l]).outs(&[h]).f(&[r, img_relief]).run();
    b.free(&[l]);
    let k = 0.5 / b.s;
    let o: [Pl; 3] = b.planes();
    b.op("gallery_t", "grooves").ins(&[img.c[0], img.c[1], img.c[2], h, dist]).outs(&o).f(&[k, side, r, floor, dark, bright, gain]).emit(img.a).run();
    b.free(&[h, dist]);
    with_c(b, img, o)
}

fn mosaic_tiles(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let tile = (p[0] * 1.25).max(2.0);
    let wl = tile * 0.7;
    let wob = 0.3 * (15.0 / tile).min(1.0);
    let dist = b.plane();
    b.op("gallery_t", "mosaic_dist").outs(&[dist]).f(&[tile, wl, wob]).run();
    let lg = p[2] / 10.0;
    grooves(b, img, dist, (p[1] * 0.9).max(0.75), 0.05, 13.5, 1.0, 1.3, 1.4, 0.25 * (1.0 - 0.3 * lg), 0.92)
}

fn craquelure(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let cell = (p[0] * 0.75).max(2.0);
    let wl = cell * 0.8;
    let m = cell * 0.5;
    let dist = b.plane();
    b.op("gallery_t", "craquelure_dist").outs(&[dist]).i(&[409]).f(&[cell, 0.9, wl, m]).run();
    let k = p[1] / 6.0;
    grooves(b, img, dist, 2.0, 0.1, 30.0 * k, 1.5, 1.5 * k, 1.2 * k * (0.5 + 0.05 * p[2]), 0.15, 1.0)
}

fn patchwork(b: &mut B, p: [f32; 3], img: Img) -> Option<Img> {
    const T: [[f32; 5]; 15] = [
        [0.699, -0.732, 0.104, 0.532, 0.021],
        [1.143, -0.841, 0.091, 0.202, 0.026],
        [1.645, -1.061, 0.045, 0.020, 0.027],
        [0.865, -0.654, 0.072, 0.792, -0.020],
        [1.634, -0.878, 0.040, 0.267, -0.012],
        [2.049, -0.956, 0.031, -0.022, 0.000],
        [0.560, -0.106, 0.031, 0.656, -0.013],
        [1.084, -0.152, 0.010, 0.195, -0.011],
        [1.349, -0.195, 0.012, 0.009, 0.008],
        [0.364, 0.019, 0.027, 0.505, -0.000],
        [0.772, 0.013, 0.011, 0.118, 0.001],
        [0.981, 0.001, 0.014, 0.005, -0.005],
        [0.098, 0.036, 0.223, 0.298, 0.024],
        [0.367, 0.025, 0.236, 0.026, 0.024],
        [0.417, 0.007, 0.288, 0.015, 0.019],
    ];
    let size = (p[0] * 2.0 + 1.0).max(3.0);
    // Cells smaller than a preview pixel would need more cells than the
    // window has pixels: those previews stay on the CPU.
    if size < b.s {
        return None;
    }
    let r = p[1] / 8.0;
    let relief = if r <= 1.0 { r } else { 1.0 + (r - 1.0) * 0.5 };
    let flat = T[11];
    let tab: Vec<f32> = T.iter().flat_map(|t| std::array::from_fn::<f32, 5, _>(|k| flat[k] + (t[k] - flat[k]) * relief)).collect();
    let means: [Pl; 3] = b.planes();
    b.op("gallery_t", "pw_means").ins(&img.c).outs(&means).f(&[size]).ext(1).run();
    let n = (b.s as usize).clamp(1, 4);
    let t = b.tab_f32(&tab);
    let o: [Pl; 3] = b.planes();
    b.op("gallery_t", "patchwork").ins(&means).outs(&o).tabs(&[t]).i(&[n as i32, size as i32 - 1]).f(&[size]).emit(img.a).run();
    b.free(&means);
    Some(with_c(b, img, o))
}

fn stained_glass(b: &mut B, p: [f32; 3], img: Img, ctx: &Ctx) -> Img {
    let cell = (p[0] * 2.15).max(2.0);
    let half = (p[1] * 0.35).max(0.3);
    let lift = p[2] * 0.01;
    let aa = 0.5 * b.s.max(1.0);
    let (dw, dh) = ctx.doc;
    let o: [Pl; 3] = b.planes();
    b.op("gallery_t", "stained_glass")
        .ins(&img.c)
        .outs(&o)
        .i(&[211])
        .f(&[cell, 0.75, half, lift, aa, p[2], dw, dh, ctx.fg[0], ctx.fg[1], ctx.fg[2], (dw.min(dh) * 0.5).max(1.0)])
        .emit(img.a)
        .run();
    with_c(b, img, o)
}

fn texturizer(b: &mut B, p: [f32; 3], img: Img) -> Img {
    let scale = (p[1] / 100.0).max(0.1);
    let amp = p[2] * 0.1;
    let kind = p[0] as usize;
    let shade = if kind == 0 && b.s <= 1.0 {
        let step = (8.0 * scale).floor().clamp(1.0, 8.0) as i32;
        let hgt = b.plane();
        b.op("gallery_t", "canvas_hgt").outs(&[hgt]).i(&[step]).f(&[scale, 1.0 / step as f32]).run();
        let k = amp * 0.75 * scale;
        let s = b.plane();
        b.op("gallery_t", "canvas_shade").ins(&[hgt]).outs(&[s]).f(&[k]).run();
        b.free(&[hgt]);
        s
    } else if kind == 0 {
        let n = (b.s as usize).clamp(1, 4);
        let step = b.s / n as f32;
        let k = amp * 0.75 * scale;
        let s = b.plane();
        b.op("gallery_t", "canvas_sampled").outs(&[s]).i(&[n as i32]).f(&[scale, step, k]).run();
        s
    } else {
        let t = b.plane();
        b.op("gallery_t", "texture").outs(&[t]).i(&[kind as i32]).f(&[(scale * 2.0).max(0.1)]).run();
        let (gx, gy) = (b.plane(), b.plane());
        b.op("gallery_s", "sobel").ins(&[t]).outs(&[gx, gy]).run();
        b.free(&[t]);
        let a = 135f32.to_radians();
        let (lx, ly) = (a.cos(), -a.sin());
        let s = b.plane();
        b.op("gallery_t", "relief_shade").ins(&[gx, gy]).outs(&[s]).f(&[lx, ly, p[2] * 0.12]).run();
        b.free(&[gx, gy]);
        s
    };
    let o: [Pl; 3] = b.planes();
    b.op("gallery_t", "add_shade").ins(&[img.c[0], img.c[1], img.c[2], shade]).outs(&o).emit(img.a).run();
    b.free(&[shade]);
    with_c(b, img, o)
}
