//! Filter Gallery ▸ Distort and Texture effects.

use super::*;

pub(super) fn run(g: G, p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    use G::*;
    match g {
        DiffuseGlow => diffuse_glow(p, img, ctx),
        Glass | OceanRipple => glass_ripple(g, p, img, ctx),
        Craquelure => craquelure(p, img, ctx),
        Grain => grain(p, img, ctx),
        MosaicTiles => mosaic_tiles(p, img, ctx),
        Patchwork => patchwork(p, img, ctx),
        StainedGlass => stained_glass(p, img, ctx),
        Texturizer => texturizer(p, img, ctx),
        _ => unreachable!("not in this group"),
    }
}

/// Grain: Regular adds independent uniform noise to each channel
/// (±0.49 × Intensity, like Photoshop's coloured speckle); the other types are
/// monochrome patterns. Contrast 50 leaves the tones alone.
fn grain(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let k = p[0] / 100.0;
    let ct = p[1] / 100.0;
    let kind = p[2] as usize;
    let gain = 0.5 + ct;
    if kind == 0 {
        let amp = k * 0.98;
        img.map(|i, c| {
            let (x, y) = ctx.g(i);
            let (xi, yi) = (x.floor() as i32, y.floor() as i32);
            std::array::from_fn(|ch| (c[ch] - 0.5) * gain + 0.5 + (hash01(xi, yi, 199 + ch as u32 * 7) - 0.5) * amp)
        });
        return;
    }
    img.map(|i, c| {
        let (x, y) = ctx.g(i);
        let n = match kind {
            1 => super::super::util::fbm(x * 0.6, y * 0.6, 2, 181) - 0.5,
            2 => if hash01(x as i32, y as i32, 183) > 0.96 { 0.5 } else { 0.0 },
            3 => super::super::util::fbm(x * 0.25, y * 0.25, 3, 185) - 0.5,
            4 => (hash01(x as i32, y as i32, 187) - 0.5).signum() * 0.5,
            5 => hash01(x as i32 / 3, y as i32 / 3, 189) - 0.5,
            6 => if hash01(x as i32, y as i32, 191) > 0.7 { -0.5 } else { 0.0 },
            7 => hash01(0, y as i32, 193) - 0.5,
            8 => hash01(x as i32, 0, 195) - 0.5,
            9 => if hash01(x as i32, y as i32, 197) > 0.92 { -0.6 } else { 0.0 },
            _ => hash01(x as i32, y as i32, 199) - 0.5,
        };
        let t = n * k * 0.8;
        let a = |v: f32| (v - 0.5) * gain + 0.5 + t;
        [a(c[0]), a(c[1]), a(c[2])]
    });
}

/// Mean colours of the `size`-square cells of the global level-0 grid that
/// the buffer covers. Cells smaller than a buffer pixel that no pixel centre
/// falls in take the colour of the pixel under their centre.
struct GridMeans {
    size: f32,
    bx0: i32,
    by0: i32,
    nbx: usize,
    nby: usize,
    means: Vec<[f32; 3]>,
}

impl GridMeans {
    fn new(img: &Img, ctx: &Ctx, size: f32) -> GridMeans {
        let (w, h) = (ctx.w, ctx.h);
        let cell = |v: f32| (v / size).floor() as i32;
        let (ex0, ey0) = (ctx.origin.0 as f32 * ctx.s, ctx.origin.1 as f32 * ctx.s);
        let (ex1, ey1) = ((ctx.origin.0 + w as i32) as f32 * ctx.s - 1e-3, (ctx.origin.1 + h as i32) as f32 * ctx.s - 1e-3);
        let (bx0, by0) = (cell(ex0), cell(ey0));
        let (nbx, nby) = ((cell(ex1) - bx0 + 1) as usize, (cell(ey1) - by0 + 1) as usize);
        let mut acc = vec![[0f32; 4]; nbx * nby];
        let cx: Vec<usize> = (0..w).map(|x| (cell(ctx.g(x).0) - bx0) as usize).collect();
        for y in 0..h {
            let by = (cell(ctx.g(y * w).1) - by0) as usize;
            for x in 0..w {
                let i = y * w + x;
                let e = &mut acc[by * nbx + cx[x]];
                e[0] += img.c[0][i];
                e[1] += img.c[1][i];
                e[2] += img.c[2][i];
                e[3] += 1.0;
            }
        }
        let means = acc
            .iter()
            .enumerate()
            .map(|(k, e)| {
                if e[3] > 0.0 {
                    return [e[0] / e[3], e[1] / e[3], e[2] / e[3]];
                }
                let (i, j) = ((k % nbx) as i32 + bx0, (k / nbx) as i32 + by0);
                let px = (((i as f32 + 0.5) * size / ctx.s) as i32 - ctx.origin.0).clamp(0, w as i32 - 1) as usize;
                let py = (((j as f32 + 0.5) * size / ctx.s) as i32 - ctx.origin.1).clamp(0, h as i32 - 1) as usize;
                let q = py * w + px;
                [img.c[0][q], img.c[1][q], img.c[2][q]]
            })
            .collect();
        GridMeans { size, bx0, by0, nbx, nby, means }
    }

    /// Mean of the cell `(di, dj)` cells away from the one holding level-0 point (x, y).
    #[inline]
    fn at(&self, x: f32, y: f32, di: i32, dj: i32) -> [f32; 3] {
        let i = ((x / self.size).floor() as i32 - self.bx0 + di).clamp(0, self.nbx as i32 - 1) as usize;
        let j = ((y / self.size).floor() as i32 - self.by0 + dj).clamp(0, self.nby as i32 - 1) as usize;
        self.means[j * self.nbx + i]
    }
}

/// Patchwork: square cells filled with their mean colour, brightened by
/// screening its brightness (× (2 − max channel), which keeps the hue), each
/// cell bevelled with a bright top lip, a dark groove along
/// the bottom and the left neighbour bleeding into its first columns; the
/// top lip embosses against the cell above (light from the top).
fn patchwork(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    // Per position (rows 0, 1, 2, interior, last) × (columns 0, 1, interior):
    // weights of [own cell, cell above, cell below, cell to the left, constant].
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
    // Relief 8 is Photoshop's default; stronger relief deepens the bevel more gently.
    let r = p[1] / 8.0;
    let relief = if r <= 1.0 { r } else { 1.0 + (r - 1.0) * 0.5 };
    let flat = T[11];
    let tab: Vec<[f32; 5]> = T.iter().map(|t| std::array::from_fn(|k| flat[k] + (t[k] - flat[k]) * relief)).collect();
    let mut grid = GridMeans::new(img, ctx, size);
    for m in grid.means.iter_mut() {
        let k = 2.0 - m[0].max(m[1]).max(m[2]);
        for v in m.iter_mut() {
            *v *= k;
        }
    }
    let last = size as i32 - 1;
    let eval = |gx: f32, gy: f32| -> [f32; 3] {
        let r = (gy - (gy / size).floor() * size) as i32;
        let c = (gx - (gx / size).floor() * size) as i32;
        let row = match r { 0..=2 => r as usize, _ if r >= last => 4, _ => 3 };
        let t = &tab[row * 3 + if c < 2 { c as usize } else { 2 }];
        let (s, u, d, l) = (grid.at(gx, gy, 0, 0), grid.at(gx, gy, 0, -1), grid.at(gx, gy, 0, 1), grid.at(gx, gy, -1, 0));
        std::array::from_fn(|k| t[0] * s[k] + t[1] * u[k] + t[2] * d[k] + t[3] * l[k] + t[4])
    };
    // At preview levels each pixel averages a few level-0 samples so the
    // bevel lines blend instead of aliasing.
    let n = (ctx.s as usize).clamp(1, 4);
    let out = par_pixels(ctx, |i| {
        let x0 = (ctx.origin.0 + (i % ctx.w) as i32) as f32 * ctx.s;
        let y0 = (ctx.origin.1 + (i / ctx.w) as i32) as f32 * ctx.s;
        let step = ctx.s / n as f32;
        let mut acc = [0f32; 3];
        for sy in 0..n {
            for sx in 0..n {
                let o = eval(x0 + (sx as f32 + 0.5) * step, y0 + (sy as f32 + 0.5) * step);
                for k in 0..3 {
                    acc[k] += o[k];
                }
            }
        }
        acc.map(|v| v / (n * n) as f32)
    });
    img.set(&out);
}

/// Diffuse Glow: the background colour glows into the bright parts of the
/// image (a soft mask of the blurred luminance above a threshold set by
/// Clear amount), and every pixel's glow is jittered by uniform grain, so dark
/// areas get sparse light speckles and half the pixels stay untouched.
fn diffuse_glow(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let grain = p[0] * 0.0227;
    let amount = p[1] / 10.0;
    let thr = 0.15 + p[2] * 0.02;
    let glow: Vec<f32> = blur(&img.lum(), w, h, ctx.px(9.0)).iter().map(|l| ((l - thr) / 0.32).clamp(0.0, 1.0).powf(0.75) * amount).collect();
    let bg = ctx.bg;
    img.map(|i, c| {
        let t = (glow[i].min(1.0) + grain * (2.0 * ctx.noise(i, 91) - 1.0)).clamp(0.0, 1.0);
        mix(c, bg, t)
    });
}

/// Glass and Ocean Ripple: the image is refracted through a bumpy surface,
/// displaced by the gradient of a value-noise height field in level-0 space.
/// Glass uses a ridged ("frosted") surface whose creases break the image into
/// small chips; Ocean Ripple uses larger three-octave waves and, like
/// Photoshop, pushes the image right and down on average.
fn glass_ripple(g: G, p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let glass = g == G::Glass;
    let (cell, mag, off, oct) = if glass {
        ((3.0 + p[1] * 1.2) * p[2] / 100.0, p[0] * 1.8, [0.0, 0.0], 2)
    } else {
        (3.0 + p[0] * 1.8, p[1] * 2.6, [-0.75 * p[1], -0.21 * p[1]], 3)
    };
    let cell = cell.max(0.5);
    let bump = par_map(ctx, |i| {
        let (x, y) = ctx.g(i);
        let n = super::super::util::fbm(x / cell, y / cell, oct, 101);
        if glass { (2.0 * n - 1.0).abs() } else { n }
    });
    // Central differences in level-0 units (one-sided at the buffer edge).
    let k = mag * cell / ctx.s;
    let (mut dx, mut dy) = (vec![0f32; w * h], vec![0f32; w * h]);
    for y in 0..h {
        let (ya, yb) = (y.saturating_sub(1), (y + 1).min(h - 1));
        for x in 0..w {
            let (xa, xb) = (x.saturating_sub(1), (x + 1).min(w - 1));
            let i = y * w + x;
            let gx = (bump[y * w + xb] - bump[y * w + xa]) / (xb - xa).max(1) as f32;
            let gy = (bump[yb * w + x] - bump[ya * w + x]) / (yb - ya).max(1) as f32;
            dx[i] = ctx.px(gx * k + off[0]);
            dy[i] = ctx.px(gy * k + off[1]);
        }
    }
    warp(img, &dx, &dy);
}

/// Seeds of a jittered square lattice (spacing `cell`, jitter `jit` of the
/// spacing, level-0 space) for every lattice cell touching the buffer plus a
/// margin of two cells: (first cell x, first cell y, cells across, seeds).
struct Lattice {
    cx0: i32,
    cy0: i32,
    nx: usize,
    cell: f32,
    seeds: Vec<(f32, f32)>,
}

impl Lattice {
    fn new(ctx: &Ctx, cell: f32, jit: f32, seed: u32) -> Lattice {
        let (x0, y0) = ctx.g(0);
        let (x1, y1) = ctx.g(ctx.w * ctx.h - 1);
        let (cx0, cy0) = ((x0 / cell).floor() as i32 - 2, (y0 / cell).floor() as i32 - 2);
        let (cx1, cy1) = ((x1 / cell).floor() as i32 + 2, (y1 / cell).floor() as i32 + 2);
        let nx = (cx1 - cx0 + 1) as usize;
        let mut seeds = Vec::with_capacity(nx * (cy1 - cy0 + 1) as usize);
        for cy in cy0..=cy1 {
            for cx in cx0..=cx1 {
                seeds.push((
                    (cx as f32 + 0.5 + jit * (hash01(cx, cy, seed) - 0.5)) * cell,
                    (cy as f32 + 0.5 + jit * (hash01(cx, cy, seed + 1) - 0.5)) * cell,
                ));
            }
        }
        Lattice { cx0, cy0, nx, cell, seeds }
    }

    /// Nearest seed of level-0 point (x, y) and the distance to the nearest
    /// cell border (closest bisector with a seed of the surrounding 3 × 3 cells).
    #[inline]
    fn nearest(&self, x: f32, y: f32) -> ((f32, f32), f32) {
        let ix = (x / self.cell).floor() as i32 - self.cx0;
        let iy = (y / self.cell).floor() as i32 - self.cy0;
        let mut pts = [(0f32, 0f32); 9];
        let (mut best, mut bd) = (0, f32::MAX);
        for k in 0..9 {
            let (ox, oy) = (k as i32 % 3 - 1, k as i32 / 3 - 1);
            let q = self.seeds[(iy + oy) as usize * self.nx + (ix + ox) as usize];
            let d = (q.0 - x).powi(2) + (q.1 - y).powi(2);
            if d < bd {
                bd = d;
                best = k;
            }
            pts[k] = q;
        }
        let (ax, ay) = pts[best];
        let mut edge = f32::MAX;
        for (k, &(bx, by)) in pts.iter().enumerate() {
            if k == best {
                continue;
            }
            let (vx, vy) = (bx - ax, by - ay);
            let len = (vx * vx + vy * vy).sqrt().max(1e-6);
            edge = edge.min((((ax + bx) * 0.5 - x) * vx + ((ay + by) * 0.5 - y) * vy) / len);
        }
        ((ax, ay), edge)
    }
}

/// Stained Glass: Voronoi cells filled with the colour of the image at each
/// cell's seed, separated by foreground-colour leading; Light intensity lifts
/// the glass a little. (Photoshop lights a spot at the document centre; the
/// gallery does not know the document size, so the light is uniform.)
fn stained_glass(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let cell = (p[0] * 2.15).max(2.0);
    let half = (p[1] * 0.35).max(0.3);
    let lift = p[2] * 0.01;
    let lat = Lattice::new(ctx, cell, 0.75, 211);
    let lead = ctx.fg;
    let aa = 0.5 * ctx.s.max(1.0);
    let src = &*img;
    let out = par_pixels(ctx, |i| {
        let (x, y) = ctx.g(i);
        let ((sx, sy), edge) = lat.nearest(x, y);
        // Seed position in buffer pixels (clamped into the padded buffer).
        let bx = ((sx / ctx.s - ctx.origin.0 as f32).floor() as isize).clamp(0, w as isize - 1) as usize;
        let by = ((sy / ctx.s - ctx.origin.1 as f32).floor() as isize).clamp(0, h as isize - 1) as usize;
        let j = by * w + bx;
        // Light intensity: Photoshop shines a spot on the document centre that
        // brightens cells by up to 1 + 0.233·intensity, fading out at half the
        // shorter side, and leads a ~1.6 px frame around the document.
        let (gain, frame) = match ctx.doc {
            Some((dw, dh)) => {
                let rr = ((sx - dw * 0.5).powi(2) + (sy - dh * 0.5).powi(2)) / (dw.min(dh) * 0.5).max(1.0).powi(2);
                let gain = if rr < 1.0 { 1.0 + 0.233 * p[2] * (1.0 - rr.sqrt()).powf(1.2) } else { 1.0 };
                let edge = x.min(y).min(dw - x).min(dh - y);
                (gain, if edge < 3.0 * aa.max(0.5) + 1.14 { ((edge - 1.14) / aa.max(0.5) * 0.5).clamp(0.0, 1.0) } else { 1.0 })
            }
            None => (1.0, 1.0),
        };
        let c: [f32; 3] = std::array::from_fn(|k| ((src.c[k][j] * 0.935 + lift) * gain).min(1.0));
        mix(lead, c, smooth(half - aa, half + aa, edge) * frame)
    });
    img.set(&out);
}

/// Evaluates `f` for every buffer pixel, rows in parallel.
fn par_pixels(ctx: &Ctx, f: impl Fn(usize) -> [f32; 3] + Sync) -> Vec<[f32; 3]> {
    use rayon::prelude::*;
    let w = ctx.w;
    let mut out = vec![[0f32; 3]; ctx.w * ctx.h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = f(y * w + x);
        }
    });
    out
}

impl Img {
    fn set(&mut self, v: &[[f32; 3]]) {
        for (i, o) in v.iter().enumerate() {
            self.c[0][i] = o[0];
            self.c[1][i] = o[1];
            self.c[2][i] = o[2];
        }
    }
}

/// Relief of V-shaped grooves plus the image's own relief, lit from the
/// top. `dist` is each pixel's distance (level-0 px) to the nearest groove
/// centre line. Faces turned away from the light darken towards black, lit
/// faces brighten the colour, and the groove floor is darkened.
struct Grooves {
    /// Groove half-width (level-0 px).
    r: f32,
    /// Weight of the x slope (light slightly from the side).
    side: f32,
    /// Relief of the (slightly blurred) image luminance, per level-0 px.
    img_relief: f32,
    /// Blur (level-0 px) of the luminance used for the image relief.
    img_blur: f32,
    dark: f32,
    bright: f32,
    floor: f32,
    gain: f32,
}

impl Grooves {
    fn apply(&self, img: &mut Img, ctx: &Ctx, dist: &[f32]) {
        let (w, h) = (ctx.w, ctx.h);
        let lum = blur(&img.lum(), w, h, ctx.px(self.img_blur));
        // Height in units where a groove wall has slope 1 per level-0 px.
        let height: Vec<f32> = dist.iter().zip(&lum).map(|(&d, &l)| -(self.r - d).max(0.0) + l * self.img_relief).collect();
        let k = 0.5 / ctx.s;
        let src = &*img;
        let out = par_pixels(ctx, |i| {
            let (x, y) = (i % w, i / w);
            let (xa, xb) = (x.saturating_sub(1), (x + 1).min(w - 1));
            let (ya, yb) = (y.saturating_sub(1), (y + 1).min(h - 1));
            // Height rising going down (a lower wall) faces the light.
            let gy = (height[yb * w + x] - height[ya * w + x]) * k;
            let gx = (height[y * w + xb] - height[y * w + xa]) * k;
            let s = (gy + self.side * gx).clamp(-1.0, 1.0);
            let f = ((1.0 - dist[i] / self.r) * 2.0).clamp(0.0, 1.0);
            let m = self.gain * (1.0 - f * self.floor) * if s < 0.0 { 1.0 + s * self.dark } else { 1.0 + s * self.bright };
            [src.c[0][i] * m, src.c[1][i] * m, src.c[2][i] * m]
        });
        img.set(&out);
    }
}

/// Mosaic Tiles: rows of irregular tiles (Tile size apart) in a brick-like
/// layout with wobbly joints, separated by grout grooves (Grout width) lit
/// from the top: a black upper wall, a darkened grout floor (less so with
/// Lighten grout) and a bright lower lip. The tiles keep the image, embossed
/// by its own relief like Photoshop's.
fn mosaic_tiles(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let tile = (p[0] * 1.25).max(2.0);
    let wl = tile * 0.7;
    // Joints wobble by up to ~4.5 px (less, relative to the tile, for big tiles).
    let wob = 0.3 * (15.0 / tile).min(1.0);
    let fbm = super::super::util::fbm;
    let dist = par_map(ctx, |i| {
        let (x, y) = ctx.g(i);
        let yy = y / tile + wob * (2.0 * fbm(x / wl, y / wl, 2, 301) - 1.0);
        let row = yy.floor();
        let xx = x / tile + hash01(row as i32, 0, 307) + 0.5 * wob * (2.0 * fbm(x / wl + 31.7, y / wl - 7.3, 2, 303) - 1.0);
        let dy = (yy - row - 0.5).abs();
        let dx = (xx - xx.floor() - 0.5).abs();
        (0.5 - dy.max(dx)) * tile
    });
    let lg = p[2] / 10.0;
    Grooves { r: (p[1] * 0.9).max(0.75), side: 0.05, img_relief: 13.5, img_blur: 1.0, dark: 1.3, bright: 1.4, floor: 0.25 * (1.0 - 0.3 * lg), gain: 0.92 }.apply(img, ctx, &dist);
}

/// Craquelure: a network of fine cracks along the borders of a domain-warped
/// jittered Voronoi tiling (Crack spacing apart), some fading out, embossed as
/// V-grooves lit from the top together with the image's own relief (Crack
/// depth); Crack brightness scales the lit lips.
fn craquelure(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let cell = (p[0] * 0.75).max(2.0);
    let wl = cell * 0.8;
    let lat = Lattice::new(ctx, cell, 0.9, 409);
    let fbm = super::super::util::fbm;
    // The warp stays below one cell so lookups stay inside the lattice margin.
    let m = cell * 0.5;
    let dist = par_map(ctx, |i| {
        let (x, y) = ctx.g(i);
        let xx = x + m * (2.0 * fbm(x / wl, y / wl, 2, 411) - 1.0);
        let yy = y + m * (2.0 * fbm(x / wl + 13.1, y / wl + 7.7, 2, 413) - 1.0);
        let fade = (fbm(x / (2.0 * wl), y / (2.0 * wl), 2, 417) - 0.45) * 1.2;
        lat.nearest(xx, yy).1.max(0.0) + fade.max(0.0) * cell
    });
    let k = p[1] / 6.0;
    Grooves { r: 2.0, side: 0.1, img_relief: 30.0 * k, img_blur: 1.5, dark: 1.5 * k, bright: 1.2 * k * (0.5 + 0.05 * p[2]), floor: 0.15, gain: 1.0 }.apply(img, ctx, &dist);
}

/// Evaluates `f` for every buffer pixel, rows in parallel.
fn par_map(ctx: &Ctx, f: impl Fn(usize) -> f32 + Sync) -> Vec<f32> {
    use rayon::prelude::*;
    let w = ctx.w;
    let mut out = vec![0f32; ctx.w * ctx.h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            *o = f(y * w + x);
        }
    });
    out
}

/// Height of a woven canvas in level-0 px (threads ~3 px apart at 100 %):
/// wobbling horizontal weft threads broken into irregular dashes over a faint
/// vertical warp, roughly 0..1.
fn canvas_height(x: f32, y: f32) -> f32 {
    let [wob, wav] = canvas_slow(x, y);
    canvas_height_with(x, y, wob, wav)
}

/// The weave's two slow wobbles (row bend and warp phase).
fn canvas_slow(x: f32, y: f32) -> [f32; 2] {
    let fbm = super::super::util::fbm;
    [fbm(x / 41.0, y / 29.0, 2, 501), fbm(x / 23.0, y / 31.0, 1, 505)]
}

fn canvas_height_with(x: f32, y: f32, wob: f32, wav: f32) -> f32 {
    let fbm = super::super::util::fbm;
    let v = y / 3.0 + 0.6 * (wob - 0.5);
    let row = v.floor();
    let f = v - row;
    let profile = 1.0 - (2.0 * f - 1.0).powi(2);
    let dash = fbm(x / 4.5 + row * 7.31, row * 0.37, 2, 503);
    let warp = 0.5 + 0.5 * (std::f32::consts::TAU * (x / 4.0 + 0.3 * wav)).sin();
    profile * (0.6 + 1.4 * (dash - 0.5)).max(0.2) + 0.1 * warp + 0.2 * hash01(x.floor() as i32, y.floor() as i32, 507)
}

/// Canvas heights for a full-resolution buffer. The slow wobbles barely change
/// over a few pixels, so they are sampled on a lattice fixed to level-0
/// coordinates (tiles agree) and interpolated bilinearly.
fn canvas_field(ctx: &Ctx, scale: f32) -> Vec<f32> {
    let step = (8.0 * scale).floor().clamp(1.0, 8.0) as i32;
    let (ox, oy) = (ctx.origin.0.div_euclid(step), ctx.origin.1.div_euclid(step));
    let gw = ((ctx.origin.0 + ctx.w as i32).div_euclid(step) - ox + 2) as usize;
    let gh = ((ctx.origin.1 + ctx.h as i32).div_euclid(step) - oy + 2) as usize;
    let mut lat = vec![[0f32; 2]; gw * gh];
    {
        use rayon::prelude::*;
        lat.par_chunks_mut(gw).enumerate().for_each(|(j, row)| {
            for (i, o) in row.iter_mut().enumerate() {
                let (x, y) = (((ox + i as i32) * step) as f32 + 0.5, ((oy + j as i32) * step) as f32 + 0.5);
                *o = canvas_slow(x / scale, y / scale);
            }
        });
    }
    let inv = 1.0 / step as f32;
    par_map(ctx, |i| {
        let (px, py) = (ctx.origin.0 + (i % ctx.w) as i32, ctx.origin.1 + (i / ctx.w) as i32);
        let (cx, cy) = (px.div_euclid(step), py.div_euclid(step));
        let (fx, fy) = (px.rem_euclid(step) as f32 * inv, py.rem_euclid(step) as f32 * inv);
        let k = (cy - oy) as usize * gw + (cx - ox) as usize;
        let s = |c: usize| {
            let t = lat[k][c] + (lat[k + 1][c] - lat[k][c]) * fx;
            let b = lat[k + gw][c] + (lat[k + gw + 1][c] - lat[k + gw][c]) * fx;
            t + (b - t) * fy
        };
        let (x, y) = (px as f32 + 0.5, py as f32 + 0.5);
        canvas_height_with(x / scale, y / scale, s(0), s(1))
    })
}

/// Texturizer: the texture's relief lit from the top is added to the image
/// (Photoshop adds it rather than multiplying, so shadows keep their weave).
fn texturizer(p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    let (w, h) = (ctx.w, ctx.h);
    let scale = (p[1] / 100.0).max(0.1);
    let amp = p[2] * 0.1;
    let shade: Vec<f32> = if p[0] as usize == 0 && ctx.s <= 1.0 {
        let hgt = canvas_field(ctx, scale);
        // Height difference between the rows above and below.
        let k = amp * 0.75 * scale;
        par_map(ctx, |i| {
            let y = i / w;
            let (ya, yb) = (y.saturating_sub(1), (y + 1).min(h - 1));
            (hgt[ya * w + i % w] - hgt[yb * w + i % w]) * k * 2.0 / (yb - ya).max(1) as f32
        })
    } else if p[0] as usize == 0 {
        // Preview levels: average the level-0 shading over a few samples per
        // pixel instead of aliasing the 3 px weave.
        let n = (ctx.s as usize).clamp(1, 4);
        let step = ctx.s / n as f32;
        let k = amp * 0.75 * scale;
        par_map(ctx, |i| {
            let x0 = (ctx.origin.0 + (i % w) as i32) as f32 * ctx.s;
            let y0 = (ctx.origin.1 + (i / w) as i32) as f32 * ctx.s;
            let mut acc = 0.0;
            for sy in 0..n {
                for sx in 0..n {
                    let (x, y) = ((x0 + (sx as f32 + 0.5) * step) / scale, (y0 + (sy as f32 + 0.5) * step) / scale);
                    acc += canvas_height(x, y - 1.0 / scale) - canvas_height(x, y + 1.0 / scale);
                }
            }
            acc * k / (n * n) as f32
        })
    } else {
        let t = texture(ctx, p[0] as usize, scale * 2.0);
        relief(&t, w, h, p[2] * 0.12, 135.0).iter().map(|v| (v - 0.5) * 0.5).collect()
    };
    {
        use rayon::prelude::*;
        img.c.par_iter_mut().for_each(|ch| ch.par_chunks_mut(w).zip(shade.par_chunks(w)).for_each(|(r, s)| r.iter_mut().zip(s).for_each(|(v, d)| *v += d)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GROUP: [G; 9] = [G::DiffuseGlow, G::Glass, G::OceanRipple, G::Craquelure, G::Grain, G::MosaicTiles, G::Patchwork, G::StainedGlass, G::Texturizer];

    fn defaults(g: G) -> [f32; 3] {
        let mut p = [0f32; 3];
        for (k, d) in g.params().iter().enumerate().take(3) {
            p[k] = d.default;
        }
        p
    }

    /// The 1024 × 683 dog photo from the Photoshop references when present,
    /// otherwise a procedural image of the same size.
    fn source() -> (Vec<u32>, usize, usize) {
        if let Ok(img) = image::open("/tmp/pf_ps_out/gallery_src.png") {
            let img = img.to_rgba8();
            let px = img.pixels().map(|p| crate::engine::px::premultiply(p[0], p[1], p[2], p[3])).collect();
            return (px, img.width() as usize, img.height() as usize);
        }
        let (w, h) = (1024usize, 683usize);
        let grid = crate::engine::ops::procedural(w as u32, h as u32);
        let mut buf = vec![0u32; w * h];
        grid.read_region_clamped(0, 0, w, h, &mut buf);
        (buf, w, h)
    }

    #[test]
    fn texture_effects_tiled_match_whole_image() {
        let (src, w, h) = source();
        let (fg, bg) = ([0, 0, 0], [255, 255, 255]);
        for g in GROUP {
            let mut params = vec![defaults(g)];
            // A couple of non-default settings too.
            let alt: Vec<f32> = g.params().iter().map(|q| q.min + (q.max - q.min) * 0.7).collect();
            params.push([alt[0], alt.get(1).copied().unwrap_or(0.0), alt.get(2).copied().unwrap_or(0.0)].map(|v| v.round()));
            for p in params {
                let mut whole = src.clone();
                g.apply(p, fg, bg, &mut whole, w, h, 0, (0, 0));
                for &(x0, y0, cw, ch) in &[(300usize, 200usize, 160usize, 120usize), (611, 397, 97, 133)] {
                    let pad = g.pad(p, 0);
                    let (bx, by) = (x0 as i32 - pad as i32, y0 as i32 - pad as i32);
                    let (bw, bh) = (cw + 2 * pad, ch + 2 * pad);
                    let mut buf = vec![0u32; bw * bh];
                    for y in 0..bh {
                        for x in 0..bw {
                            let sx = (bx + x as i32).clamp(0, w as i32 - 1) as usize;
                            let sy = (by + y as i32).clamp(0, h as i32 - 1) as usize;
                            buf[y * bw + x] = src[sy * w + sx];
                        }
                    }
                    g.apply(p, fg, bg, &mut buf, bw, bh, 0, (bx, by));
                    let mut worst = 0i32;
                    for y in 0..ch {
                        for x in 0..cw {
                            let a = crate::engine::px::unpremultiply(whole[(y0 + y) * w + x0 + x]);
                            let b = crate::engine::px::unpremultiply(buf[(pad + y) * bw + pad + x]);
                            for k in 0..4 {
                                worst = worst.max((a[k] as i32 - b[k] as i32).abs());
                            }
                        }
                    }
                    assert!(worst <= 2, "{} {p:?}: tiled differs from whole image by {worst} at crop ({x0}, {y0})", g.title());
                }
            }
        }
    }

    #[test]
    fn texture_effects_handle_extreme_parameters() {
        let (src, w, h) = source();
        let (cw, ch) = (180usize, 140usize);
        let crop: Vec<u32> = (0..cw * ch).map(|i| src[((250 + i / cw) % h) * w + 300 + i % cw]).collect();
        for g in GROUP {
            let ps = g.params();
            for f in [0.0f32, 1.0] {
                let mut p = [0f32; 3];
                for (k, q) in ps.iter().enumerate().take(3) {
                    p[k] = q.min + (q.max - q.min) * f;
                }
                for level in [0u32, 2] {
                    let mut img = Img::load(&crop, cw, ch);
                    let ctx = Ctx { w: cw, h: ch, s: (1u32 << level) as f32, origin: (40, 30), fg: col([10, 10, 10]), bg: col([250, 250, 240]), doc: None };
                    run(g, p, &mut img, &ctx);
                    let finite = img.c.iter().chain(std::iter::once(&img.a)).all(|c| c.iter().all(|v| v.is_finite()));
                    assert!(finite, "{} {p:?} level {level}: non-finite output", g.title());
                }
            }
        }
    }
}
