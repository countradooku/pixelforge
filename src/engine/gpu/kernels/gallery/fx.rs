//! The gallery's shared building blocks as GPU passes (`filters::gallery`
//! helpers), and the effect dispatch.

use super::{B, Pl, Shape};
use crate::engine::filters::blur::boxes_for_gauss;
use crate::engine::filters::gallery::G;

/// Straight-alpha planes of one batch of windows.
#[derive(Clone, Copy)]
pub struct Img {
    pub c: [Pl; 3],
    pub a: Pl,
}

pub struct Ctx {
    pub fg: [f32; 3],
    pub bg: [f32; 3],
    pub doc: (f32, f32),
}

fn col(c: [u8; 3]) -> [f32; 3] {
    [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]
}

/// Effects whose GPU program meets the equality bar.
pub fn supported(_g: G) -> bool {
    true
}

pub fn build(b: &mut B, g: G, p: [f32; 3], fg: [u8; 3], bg: [u8; 3], doc: (u32, u32)) -> Option<()> {
    use G::*;
    let ctx = Ctx { fg: col(fg), bg: col(bg), doc: (doc.0 as f32, doc.1 as f32) };
    if !matches!(g, WaterPaper) && g.group() == "Sketch" {
        let out = super::sketch::run_gray(b, g, p, &ctx)?;
        store(b, out);
        return Some(());
    }
    let img = load(b);
    let out = match g {
        ColoredPencil | Cutout | DryBrush | FilmGrain | Fresco | NeonGlow | PaintDaubs | PaletteKnife | PlasticWrap | PosterEdges
        | RoughPastels | SmudgeStick | Sponge | Underpainting | Watercolor => super::art::run(b, g, p, img, &ctx)?,
        AccentedEdges | AngledStrokes | Crosshatch | DarkStrokes | InkOutlines | Spatter | SprayedStrokes | SumiE | GlowingEdges => {
            super::brush::run(b, g, p, img, &ctx)?
        }
        BasRelief | ChalkCharcoal | Charcoal | Chrome | ConteCrayon | GraphicPen | HalftonePattern | NotePaper | Photocopy | Plaster
        | Reticulation | Stamp | TornEdges | WaterPaper => super::sketch::run(b, g, p, img, &ctx)?,
        DiffuseGlow | Glass | OceanRipple | Craquelure | Grain | MosaicTiles | Patchwork | StainedGlass | Texturizer => {
            super::texture::run(b, g, p, img, &ctx)?
        }
    };
    store(b, out);
    Some(())
}

// ----- load / store / packing ------------------------------------------------------------

pub fn load(b: &mut B) -> Img {
    let [r, g, bl, a] = b.planes();
    let t = [b.unpremul(), b.by255()];
    b.op("gallery", "load").src().outs(&[r, g, bl, a]).tabs(&t).run();
    Img { c: [r, g, bl], a }
}

pub fn store(b: &mut B, img: Img) {
    // Fold the store into the last pass when that pass produced exactly
    // these color planes through `emit3`.
    if let Some(last) = b.ops.last_mut() {
        let outs: Vec<usize> = (10..13).filter_map(|k| last.bufs.iter().find(|(bd, _)| *bd == k).and_then(|(_, r)| if let super::Res::Plane(i) = r { Some(*i) } else { None })).collect();
        if last.emit == Some(img.a.0) && outs == [img.c[0].0, img.c[1].0, img.c[2].0] {
            last.f[15] = 1.0;
            return;
        }
    }
    b.op("gallery", "store").ins(&[img.c[0], img.c[1], img.c[2], img.a]).dst().shape(Shape::Center).run();
}

/// `Img::store` into a packed plane (on planes extended by `e`).
pub fn pack(b: &mut B, img: &Img, e: u32) -> Pl {
    let v = b.plane();
    b.op("gallery", "pack").ins(&[img.c[0], img.c[1], img.c[2], img.a]).vouts(&[v]).ext(e).run();
    v
}

/// `Img::load` of a packed plane.
pub fn unpack(b: &mut B, v: Pl, e: u32) -> Img {
    let [r, g, bl, a] = b.planes();
    let t = [b.unpremul(), b.by255()];
    b.op("gallery", "unpack").uins(&[v]).outs(&[r, g, bl, a]).tabs(&t).ext(e).run();
    Img { c: [r, g, bl], a }
}

pub fn extend(b: &mut B, v: Pl, e: u32) -> Pl {
    let o = b.plane();
    b.op("gallery", "extend").ins(&[v]).outs(&[o]).ext(e).run();
    o
}

pub fn crop(b: &mut B, v: Pl, e: u32) -> Pl {
    let o = b.plane();
    b.op("gallery", "crop").ins(&[v]).outs(&[o]).ext(e).run();
    o
}

pub fn copy(b: &mut B, v: Pl) -> Pl {
    let o = b.plane();
    b.op("gallery", "copy").ins(&[v]).outs(&[o]).run();
    o
}

/// Replaces an image's color planes (frees the old ones).
pub fn with_c(b: &mut B, img: Img, c: [Pl; 3]) -> Img {
    b.free(&img.c);
    Img { c, a: img.a }
}

// ----- point helpers ------------------------------------------------------------------------

pub fn lum(b: &mut B, img: &Img) -> Pl {
    lum_e(b, &img.c, 0)
}

pub fn lum_e(b: &mut B, c: &[Pl; 3], e: u32) -> Pl {
    let l = b.plane();
    b.op("gallery", "lum").ins(c).outs(&[l]).f(&[0.299, 0.587, 0.114]).ext(e).run();
    l
}

pub fn vmax(b: &mut B, img: &Img) -> Pl {
    let v = b.plane();
    b.op("gallery", "vmax").ins(&img.c).outs(&[v]).run();
    v
}

/// Uniform noise keyed on level-0 coordinates (`ctx.noise(i, seed) - off`),
/// on a plane extended by `ext`.
pub fn noise(b: &mut B, seed: u32, off: f32, ext: u32) -> Pl {
    let o = b.plane();
    b.op("gallery_a", "noise").outs(&[o]).i(&[seed as i32]).f(&[off]).ext(ext).run();
    o
}

// ----- blurs ------------------------------------------------------------------------------

fn box_pass(b: &mut B, v: Pl, r: usize, horiz: bool, e: u32) -> Pl {
    let inv = 1.0 / (2 * r + 1) as f32;
    let t = b.plane();
    let (entry, shape) = if horiz { ("box_h", Shape::Rows) } else { ("box_v", Shape::Cols) };
    b.op("gallery", entry).ins(&[v]).outs(&[t]).i(&[r as i32]).f(&[inv]).shape(shape).ext(e).run();
    t
}

/// `gallery::blur` (and `brush::gblur`): three boxes, each horizontal then
/// vertical, f32 running sums. Returns a new plane.
pub fn blur(b: &mut B, v: Pl, sigma: f32) -> Pl {
    blur_axes(b, v, sigma, true, true, 0)
}

/// `blur` on chosen axes (`sketch2::hblur` / `vblur`), planes extended by `e`.
pub fn blur_axes(b: &mut B, v: Pl, sigma: f32, horiz: bool, vert: bool, e: u32) -> Pl {
    if sigma < 0.4 {
        return copy_e(b, v, e);
    }
    let mut cur = v;
    for bx in boxes_for_gauss(sigma) {
        let r = bx / 2;
        if r == 0 {
            continue;
        }
        for (on, horizontal) in [(horiz, true), (vert, false)] {
            if !on {
                continue;
            }
            let n = box_pass(b, cur, r, horizontal, e);
            if cur != v {
                b.free(&[cur]);
            }
            cur = n;
        }
    }
    if cur == v { copy_e(b, v, e) } else { cur }
}

/// `blur` of three planes with one sigma. New planes.
pub fn blur3(b: &mut B, c: &[Pl; 3], sigma: f32) -> [Pl; 3] {
    let o: [Pl; 3] = b.planes();
    if sigma < 0.4 {
        for k in 0..3 {
            b.op("gallery", "copy").ins(&[c[k]]).outs(&[o[k]]).run();
        }
        return o;
    }
    // Three scalar chains: a vec4 box pass costs about four scalar ones, so
    // packing only pays off for four channels (Cutout's sums).
    b.free(&o);
    [blur(b, c[0], sigma), blur(b, c[1], sigma), blur(b, c[2], sigma)]
}

/// `blur` of a vec4 plane (sigma >= 0.4); new vec4 plane.
pub fn blur4(b: &mut B, v: Pl, sigma: f32) -> Pl {
    let mut cur = v;
    for bx in boxes_for_gauss(sigma) {
        let r = bx / 2;
        if r == 0 {
            continue;
        }
        let inv = 1.0 / (2 * r + 1) as f32;
        for (entry, shape) in [("box_h4", Shape::Rows), ("box_v4", Shape::Cols)] {
            let t = b.plane4();
            b.op("gallery", entry).qins(&[cur]).wout(t).i(&[r as i32]).f(&[inv]).shape(shape).run();
            if cur != v {
                b.free(&[cur]);
            }
            cur = t;
        }
    }
    if cur == v {
        let t = b.plane4();
        b.op("gallery", "copy4").qins(&[v]).wout(t).run();
        return t;
    }
    cur
}

pub fn copy_e(b: &mut B, v: Pl, e: u32) -> Pl {
    let o = b.plane();
    b.op("gallery", "copy").ins(&[v]).outs(&[o]).ext(e).run();
    o
}

/// The normalized Gaussian taps of the CPU's exact separable Gaussians
/// (`artistic2::gauss`, `sketch1::gauss`, `sketch2::gauss` build the same
/// kernel with the same f32 expressions).
pub fn gauss_kernel(sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil() as isize;
    let mut k: Vec<f32> = (-r..=r).map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|x| *x /= s);
    k
}

/// Exact separable Gaussian with the given taps per axis (None = skip).
pub fn gauss_taps(b: &mut B, v: Pl, kx: Option<&[f32]>, ky: Option<&[f32]>, e: u32) -> Pl {
    let mut cur = v;
    for (k, entry) in [(kx, "gauss_h"), (ky, "gauss_v")] {
        let Some(k) = k else { continue };
        let t = b.tab_f32(k);
        let o = b.plane();
        b.op("gallery", entry).ins(&[cur]).outs(&[o]).tabs(&[t]).i(&[k.len() as i32]).ext(e).run();
        if cur != v {
            b.free(&[cur]);
        }
        cur = o;
    }
    if cur == v { copy_e(b, v, e) } else { cur }
}

/// `artistic2::gauss` / `sketch2::gauss` (copy below sigma 0.2).
pub fn gauss(b: &mut B, v: Pl, sigma: f32) -> Pl {
    if sigma < 0.2 {
        return copy(b, v);
    }
    let k = gauss_kernel(sigma);
    gauss_taps(b, v, Some(&k), Some(&k), 0)
}

/// `sketch1::gauss` (anisotropic; a 1-tap axis below 0.2, the vertical pass
/// skipped then), on planes extended by `e`.
pub fn gauss_xy(b: &mut B, v: Pl, sx: f32, sy: f32, e: u32) -> Pl {
    let kern = |s: f32| if s < 0.2 { vec![1.0f32] } else { gauss_kernel(s) };
    let (kx, ky) = (kern(sx), kern(sy));
    let ky = if ky.len() == 1 { None } else { Some(&ky[..]) };
    gauss_taps(b, v, Some(&kx), ky, e)
}

/// `artistic2::box_blur`: `passes` integer boxes of radius r at level 0,
/// the equal-variance exact Gaussian at preview levels.
pub fn box_blur(b: &mut B, v: Pl, r: usize, passes: usize) -> Pl {
    if b.s > 1.0 {
        let var = passes as f32 * (((2 * r + 1) * (2 * r + 1)) as f32 - 1.0) / 12.0;
        let s = b.px(var.sqrt());
        return gauss(b, v, s);
    }
    if r == 0 || passes == 0 {
        return copy(b, v);
    }
    let q = b.plane();
    b.op("gallery", "iquant").ins(&[v]).vouts(&[q]).run();
    let mut cur = q;
    for _ in 0..passes {
        for (entry, shape) in [("ibox_h", Shape::Rows), ("ibox_v", Shape::Cols)] {
            let o = b.plane();
            b.op("gallery", entry).uins(&[cur]).vouts(&[o]).i(&[r as i32]).shape(shape).run();
            b.free(&[cur]);
            cur = o;
        }
    }
    let n = ((2 * r + 1) as f64).powi(2 * passes as i32) * 255.0;
    let o = b.plane();
    b.op("gallery", "idiv").uins(&[cur]).outs(&[o]).f(&[n as f32]).run();
    b.free(&[cur]);
    o
}

// ----- rank filters ----------------------------------------------------------------------

/// `rank::median` of a packed plane (extended by `e`): premultiplied
/// channels, alpha too unless the whole (extended) window is opaque, the
/// r-pixel frame unfiltered.
pub fn median_packed(b: &mut B, v: Pl, r: usize, e: u32) -> Pl {
    let flags = b.plane();
    b.clear(flags);
    b.op("gallery", "not_opaque").uins(&[v]).bind(20, flags).shape(Shape::Rows).ext(e).run();
    let o = b.plane();
    if r <= 2 {
        b.op("gallery_m", "median_fs_px").uins(&[v, flags]).vouts(&[o]).cst("MR", r as u32).ext(e).run();
    } else if r == 3 {
        b.op("gallery", "median_reg_px").uins(&[v, flags]).vouts(&[o]).cst("MR", r as u32).ext(e).run();
    } else {
        let ch: [Pl; 4] = b.planes();
        let n = (2 * r + 1) * (2 * r + 1);
        rank_slide(b, v, &ch, r, 4, 0, n / 2, false, None, e);
        b.op("gallery", "median_pack").uins(&[v, flags]).ins(&ch).vouts(&[o]).i(&[r as i32]).ext(e).run();
        b.free(&ch);
    }
    b.free(&[flags]);
    o
}

/// Rows per thread of `rank_slide`.
const SLIDE_ROWS: u32 = 64;

/// The sliding-histogram rank filter: the rank-`k` (0-based, ascending)
/// sample of each channel of a packed 8-bit plane over a square of radius
/// `ri` (or a disc with column half-heights `vh`).
#[allow(clippy::too_many_arguments)]
pub fn rank_slide(b: &mut B, v: Pl, outs: &[Pl], ri: usize, nch: u32, mode: i32, k: usize, by255: bool, vh: Option<&[u32]>, e: u32) {
    let cpg = 48 / nch;
    let t0 = b.tab_u32(vh.unwrap_or(&[0]));
    let t1 = b.by255();
    // The shader binds four outputs; a spare one takes an unused fourth channel.
    let spare = (outs.len() < 4).then(|| b.plane());
    let mut o = outs.to_vec();
    o.extend(spare);
    b.op("gallery", "rank_slide")
        .uins(&[v])
        .outs(&o)
        .tabs(&[t0, t1])
        .i(&[ri as i32, nch as i32, mode, k as i32, SLIDE_ROWS as i32, by255 as i32])
        .shape(Shape::Groups(cpg, SLIDE_ROWS))
        .ext(e)
        .run();
    if let Some(sp) = spare {
        b.free(&[sp]);
    }
}
/// `gallery::median`: store, `rank::median`, load (all four planes).
pub fn median_img(b: &mut B, img: Img, r: usize, e: u32) -> Img {
    if r == 0 {
        return img;
    }
    let pk = pack(b, &img, e);
    let m = median_packed(b, pk, r, e);
    b.free(&[pk]);
    let out = unpack(b, m, e);
    b.free(&[m]);
    b.free(&img.c);
    b.free(&[img.a]);
    out
}

/// Square median of 8-bit-quantized color planes, edge clamped
/// (`median_clamped`, `fast::median5`); new planes.
pub fn median_q(b: &mut B, c: &[Pl; 3], r: usize, e: u32) -> [Pl; 3] {
    let o: [Pl; 3] = b.planes();
    let t = b.by255();
    let q = b.plane();
    b.op("gallery", "quant3").ins(c).vouts(&[q]).ext(e).run();
    if r <= 2 {
        b.op("gallery_m", "median_fs_q").uins(&[q]).outs(&o).tab(1, t).cst("MR", r as u32).ext(e).run();
    } else if r == 3 {
        b.op("gallery", "median_reg_q").uins(&[q]).outs(&o).tab(1, t).cst("MR", r as u32).ext(e).run();
    } else {
        let n = (2 * r + 1) * (2 * r + 1);
        rank_slide(b, q, &o, r, 3, 0, n / 2, true, None, e);
    }
    b.free(&[q]);
    o
}

/// `stylize::kuwahara` of a packed plane.
pub fn kuwahara_packed(b: &mut B, v: Pl, r: usize, e: u32) -> Pl {
    let o = b.plane();
    b.op("gallery", "kuwahara").uins(&[v]).vouts(&[o]).i(&[r as i32]).ext(e).run();
    o
}

/// `gallery::kuwahara`: store, Kuwahara, load (all four planes).
pub fn kuwahara(b: &mut B, img: Img, r: usize, e: u32) -> Img {
    if r == 0 {
        return img;
    }
    let pk = pack(b, &img, e);
    let k = kuwahara_packed(b, pk, r, e);
    b.free(&[pk]);
    let out = unpack(b, k, e);
    b.free(&[k]);
    b.free(&img.c);
    b.free(&[img.a]);
    out
}

/// `artistic2::with_border`: runs `f` on the image extended by `bw`
/// clamped pixels and keeps the color of its interior.
pub fn with_border(b: &mut B, img: Img, bw: u32, f: impl FnOnce(&mut B, Img) -> Img) -> Img {
    let ext = Img { c: [extend(b, img.c[0], bw), extend(b, img.c[1], bw), extend(b, img.c[2], bw)], a: extend(b, img.a, bw) };
    let big = f(b, ext);
    let c = [crop(b, big.c[0], bw), crop(b, big.c[1], bw), crop(b, big.c[2], bw)];
    b.free(&big.c);
    b.free(&[big.a]);
    with_c(b, img, c)
}

/// `brush::dir_median` of the three color planes (n samples, "\\" when
/// `down`).
pub fn dir_median3(b: &mut B, c: &[Pl; 3], down: bool, n: usize) -> [Pl; 3] {
    let o: [Pl; 3] = b.planes();
    if n / 2 == 0 {
        for k in 0..3 {
            b.op("gallery", "copy").ins(&[c[k]]).outs(&[o[k]]).run();
        }
        return o;
    }
    let q = b.plane();
    b.op("gallery", "quant3").ins(c).vouts(&[q]).run();
    let t = b.by255();
    if n <= 25 {
        b.op("gallery_m", "dir_median_fs").uins(&[q]).outs(&o).tab(1, t).i(&[n as i32, down as i32]).cst("DN", n as u32).run();
    } else if n <= 64 {
        b.op("gallery", "dir_median3").uins(&[q]).outs(&o).tab(1, t).i(&[n as i32, down as i32]).cst("DN", n as u32).run();
    } else {
        for k in 0..3 {
            b.op("gallery", "dir_median").ins(&[c[k]]).outs(&[o[k]]).tab(1, t.clone()).i(&[n as i32, down as i32]).run();
        }
    }
    b.free(&[q]);
    o
}

/// Bilinear self-warp of all four planes by (dx, dy).
pub fn warp(b: &mut B, img: Img, dx: Pl, dy: Pl) -> Img {
    let [r, g, bl, a] = b.planes();
    b.op("gallery", "warp4").ins(&[img.c[0], img.c[1], img.c[2], img.a, dx, dy]).outs(&[r, g, bl, a]).run();
    b.free(&img.c);
    b.free(&[img.a, dx, dy]);
    Img { c: [r, g, bl], a }
}
