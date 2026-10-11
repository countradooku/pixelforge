//! Object Selection (GrabCut) and Quick Selection (seeded graph cut), both
//! solved on a coarse mip level and upsampled with an edge-aware guided
//! filter that runs only on level-0 tiles along the object boundary.
//!
//! - **GrabCut** (Rother et al. 2004): 5-component full-covariance GMMs for
//!   foreground and background, initialized by k-means, alternating with
//!   min cuts (Boykov–Kolmogorov, `graphcut.rs`) over a contrast-sensitive
//!   8-connected grid.
//! - **Quick Selection**: brushed pixels are hard foreground; foreground and
//!   background colour histograms (from the strokes and from the ring
//!   around the region of interest) give the data term; the region of
//!   interest grows with the strokes, so cost follows the brushed area.
//! - **Refinement**: the coarse binary mask is bilinearly upsampled and then
//!   guided-filtered with the full-resolution luminance (He et al.), which
//!   snaps the soft edge to image edges. Tiles whose footprint is entirely
//!   inside or outside are emitted as `Full` / `Empty` without pixel work.

use rayon::prelude::*;

use super::graphcut::cut_grid;
use super::grid::{RectI, TS, TS_I};
use super::px;
use super::selection::{MaskGrid, MaskTile};
use super::vector::{P, Poly, contains};

/// Reads `w x h` premultiplied pixels at `(x0, y0)` of a mip level.
pub type Source<'a> = dyn Fn(u32, i32, i32, usize, usize, &mut [u32]) + Sync + 'a;

fn rgb_f(c: u32) -> [f32; 3] {
    let [r, g, b, a] = px::unpack(c);
    // Over white so transparency reads as background-ish but stable.
    let k = 255 - a as u32;
    [(r as u32 + k) as f32, (g as u32 + k) as f32, (b as u32 + k) as f32]
}

// ----- Gaussian mixture models ----------------------------------------------------------

const K: usize = 5;

#[derive(Clone, Copy, Default)]
struct Gauss {
    w: f32,
    mean: [f32; 3],
    inv: [[f32; 3]; 3],
    log_norm: f32,
}

struct Gmm {
    c: [Gauss; K],
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inv3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let d = det3(m);
    let k = 1.0 / d;
    [
        [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * k, (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * k, (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * k],
        [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * k, (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * k, (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * k],
        [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * k, (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * k, (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * k],
    ]
}

impl Gmm {
    /// Fits the model to samples given their component assignment.
    fn fit(samples: &[[f32; 3]], comp: &[u8]) -> Gmm {
        let mut sum = [[0f64; 3]; K];
        let mut prod = [[[0f64; 3]; 3]; K];
        let mut cnt = [0usize; K];
        for (s, &k) in samples.iter().zip(comp) {
            let k = k as usize;
            cnt[k] += 1;
            for a in 0..3 {
                sum[k][a] += s[a] as f64;
                for b in 0..3 {
                    prod[k][a][b] += s[a] as f64 * s[b] as f64;
                }
            }
        }
        let total = samples.len().max(1) as f64;
        let mut c = [Gauss::default(); K];
        for k in 0..K {
            if cnt[k] == 0 {
                c[k].w = 0.0;
                continue;
            }
            let n = cnt[k] as f64;
            let mean = [sum[k][0] / n, sum[k][1] / n, sum[k][2] / n];
            let mut cov = [[0f64; 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    cov[a][b] = prod[k][a][b] / n - mean[a] * mean[b];
                }
                cov[a][a] += 4.0; // regularization (≈ 2 levels of noise)
            }
            let d = det3(&cov).max(1e-6);
            let inv = inv3(&cov);
            c[k] = Gauss {
                w: (n / total) as f32,
                mean: mean.map(|v| v as f32),
                inv: inv.map(|r| r.map(|v| v as f32)),
                log_norm: (-0.5 * d.ln()) as f32,
            };
        }
        Gmm { c }
    }

    #[inline]
    fn comp_log(&self, k: usize, z: &[f32; 3]) -> f32 {
        let g = &self.c[k];
        let d = [z[0] - g.mean[0], z[1] - g.mean[1], z[2] - g.mean[2]];
        let mut q = 0.0;
        for a in 0..3 {
            for b in 0..3 {
                q += d[a] * g.inv[a][b] * d[b];
            }
        }
        g.log_norm - 0.5 * q
    }

    /// −log likelihood of `z`.
    fn nll(&self, z: &[f32; 3]) -> f32 {
        let mut s = 0f32;
        let mut best = f32::MIN;
        for k in 0..K {
            if self.c[k].w > 0.0 {
                best = best.max(self.comp_log(k, z) + self.c[k].w.ln());
            }
        }
        for k in 0..K {
            if self.c[k].w > 0.0 {
                s += (self.comp_log(k, z) + self.c[k].w.ln() - best).exp();
            }
        }
        -(best + s.ln())
    }

    fn assign(&self, z: &[f32; 3]) -> u8 {
        let mut best = (0u8, f32::MIN);
        for k in 0..K {
            if self.c[k].w > 0.0 {
                let v = self.comp_log(k, z) + self.c[k].w.ln();
                if v > best.1 {
                    best = (k as u8, v);
                }
            }
        }
        best.0
    }
}

/// k-means (k-means++ seeding) assignment used to initialize a GMM.
fn kmeans(samples: &[[f32; 3]], iters: usize) -> Vec<u8> {
    let n = samples.len();
    if n == 0 {
        return vec![];
    }
    let d2 = |a: &[f32; 3], b: &[f32; 3]| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    let mut centers: Vec<[f32; 3]> = vec![samples[n / 2]];
    while centers.len() < K {
        // Farthest-point seeding (deterministic k-means++ variant).
        let far = (0..n).step_by((n / 4096).max(1)).max_by(|&a, &b| {
            let da = centers.iter().map(|c| d2(&samples[a], c)).fold(f32::MAX, f32::min);
            let db = centers.iter().map(|c| d2(&samples[b], c)).fold(f32::MAX, f32::min);
            da.total_cmp(&db)
        });
        centers.push(samples[far.unwrap_or(0)]);
    }
    let mut assign = vec![0u8; n];
    for _ in 0..iters {
        for (i, s) in samples.iter().enumerate() {
            let mut best = (0u8, f32::MAX);
            for (k, c) in centers.iter().enumerate() {
                let d = d2(s, c);
                if d < best.1 {
                    best = (k as u8, d);
                }
            }
            assign[i] = best.0;
        }
        let mut sum = [[0f64; 3]; K];
        let mut cnt = [0usize; K];
        for (s, &k) in samples.iter().zip(&assign) {
            cnt[k as usize] += 1;
            for a in 0..3 {
                sum[k as usize][a] += s[a] as f64;
            }
        }
        for k in 0..K {
            if cnt[k] > 0 {
                centers[k] = [sum[k][0] / cnt[k] as f64, sum[k][1] / cnt[k] as f64, sum[k][2] / cnt[k] as f64].map(|v| v as f32);
            }
        }
    }
    assign
}

/// Contrast-sensitive smoothness: γ · exp(−β‖zᵢ − zⱼ‖²), β from the mean.
fn beta(img: &[[f32; 3]], w: usize, h: usize) -> f32 {
    let mut s = 0f64;
    let mut n = 0usize;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let d = |j: usize| (img[i][0] - img[j][0]).powi(2) + (img[i][1] - img[j][1]).powi(2) + (img[i][2] - img[j][2]).powi(2);
            if x + 1 < w {
                s += d(i + 1) as f64;
                n += 1;
            }
            if y + 1 < h {
                s += d(i + w) as f64;
                n += 1;
            }
        }
    }
    let mean = (s / n.max(1) as f64).max(1e-3);
    (1.0 / (2.0 * mean)) as f32
}

/// Fits a GMM: k-means on a subsample (≤ 40k) seeds the components, then
/// every sample is assigned to its most likely component and the model refit.
fn init_gmm(samples: &[[f32; 3]]) -> Gmm {
    let step = (samples.len() / 40_000).max(1);
    let sub: Vec<[f32; 3]> = samples.iter().step_by(step).copied().collect();
    let g0 = Gmm::fit(&sub, &kmeans(&sub, 5));
    let comp: Vec<u8> = samples.par_iter().map(|z| g0.assign(z)).collect();
    Gmm::fit(samples, &comp)
}

/// Hard labels of the region: 0 = background (fixed), 1 = unknown.
pub fn grabcut(img: &[[f32; 3]], w: usize, h: usize, unknown: &[bool], iters: usize) -> Vec<bool> {
    grabcut_init(img, w, h, unknown, None, iters)
}

/// GrabCut with an optional initial foreground estimate (coarse-to-fine:
/// the coarse level's result seeds the colour models at the fine level).
pub fn grabcut_init(img: &[[f32; 3]], w: usize, h: usize, unknown: &[bool], init: Option<Vec<bool>>, iters: usize) -> Vec<bool> {
    let n = w * h;
    let mut fg: Vec<bool> = init.map(|v| v.iter().zip(unknown).map(|(&a, &u)| a && u).collect()).unwrap_or_else(|| unknown.to_vec());
    let gamma = 50.0f32;
    let b = beta(img, w, h);
    // Pairwise weights don't change between iterations.
    let right: Vec<f32> = (0..n).into_par_iter().map(|i| if i % w + 1 < w { pair(img, i, i + 1, b, gamma) } else { 0.0 }).collect();
    let down: Vec<f32> = (0..n).into_par_iter().map(|i| if i + w < n { pair(img, i, i + w, b, gamma) } else { 0.0 }).collect();
    let mut models: Option<(Gmm, Gmm)> = None;
    for _ in 0..iters.max(1) {
        let fg_s: Vec<[f32; 3]> = (0..n).filter(|&i| fg[i]).map(|i| img[i]).collect();
        let bg_s: Vec<[f32; 3]> = (0..n).filter(|&i| !fg[i]).map(|i| img[i]).collect();
        if fg_s.is_empty() || bg_s.is_empty() {
            break;
        }
        let (gf, gb) = match models.take() {
            None => (init_gmm(&fg_s), init_gmm(&bg_s)),
            Some((gf, gb)) => {
                // GrabCut step: reassign components under the current models, refit.
                let cf: Vec<u8> = fg_s.par_iter().map(|z| gf.assign(z)).collect();
                let cb: Vec<u8> = bg_s.par_iter().map(|z| gb.assign(z)).collect();
                (Gmm::fit(&fg_s, &cf), Gmm::fit(&bg_s, &cb))
            }
        };
        let d: Vec<(f32, f32)> = img.par_iter().map(|z| (gf.nll(z), gb.nll(z))).collect();
        let big = 1e9f32;
        let tcut = std::time::Instant::now();
        fg = cut_grid(
            w,
            h,
            true,
            |i| if !unknown[i] { (0.0, big) } else { (d[i].1.max(0.0), d[i].0.max(0.0)) },
            |i, j| {
                if j == i + 1 {
                    right[i]
                } else if j == i + w {
                    down[i]
                } else {
                    pair(img, i, j, b, gamma)
                }
            },
        );
        if std::env::var("PF_TIMING").is_ok() {
            eprintln!("  grabcut iter: cut {:.1} ms ({}x{})", tcut.elapsed().as_secs_f64() * 1000.0, w, h);
        }
        models = Some((gf, gb));
    }
    fg
}

#[inline]
fn pair(img: &[[f32; 3]], i: usize, j: usize, b: f32, gamma: f32) -> f32 {
    let d = (img[i][0] - img[j][0]).powi(2) + (img[i][1] - img[j][1]).powi(2) + (img[i][2] - img[j][2]).powi(2);
    gamma * (-b * d).exp()
}

/// Removes small islands: keeps components whose area is at least `frac`
/// of the largest one, and fills holes smaller than that.
pub fn clean(mask: &mut [bool], w: usize, h: usize, frac: f32) {
    for pass in [true, false] {
        let mut label = vec![u32::MAX; w * h];
        let mut sizes: Vec<usize> = vec![];
        for s in 0..w * h {
            if mask[s] != pass || label[s] != u32::MAX {
                continue;
            }
            let id = sizes.len() as u32;
            let mut stack = vec![s];
            label[s] = id;
            let mut size = 0;
            while let Some(i) = stack.pop() {
                size += 1;
                let (x, y) = (i % w, i / w);
                let mut push = |j: usize| {
                    if mask[j] == pass && label[j] == u32::MAX {
                        label[j] = id;
                        stack.push(j);
                    }
                };
                if x > 0 {
                    push(i - 1);
                }
                if x + 1 < w {
                    push(i + 1);
                }
                if y > 0 {
                    push(i - w);
                }
                if y + 1 < h {
                    push(i + w);
                }
            }
            sizes.push(size);
        }
        let Some(&biggest) = sizes.iter().max() else { continue };
        let keep = (biggest as f32 * frac) as usize;
        for i in 0..w * h {
            if mask[i] == pass && sizes[label[i] as usize] < keep {
                mask[i] = !pass;
            }
        }
    }
}

// ----- quick selection ---------------------------------------------------------------------

const HB: usize = 16;

fn hist_bin(z: &[f32; 3]) -> usize {
    let q = |v: f32| ((v / 256.0 * HB as f32) as usize).min(HB - 1);
    (q(z[0]) * HB + q(z[1])) * HB + q(z[2])
}

/// Smoothed, normalized 16³ colour histogram.
fn histogram(samples: impl Iterator<Item = [f32; 3]>) -> Vec<f32> {
    let mut hist = vec![0f32; HB * HB * HB];
    let mut n = 0f32;
    for z in samples {
        hist[hist_bin(&z)] += 1.0;
        n += 1.0;
    }
    // Box smoothing across neighbouring bins avoids zero-probability holes.
    let mut sm = vec![0f32; hist.len()];
    for r in 0..HB {
        for g in 0..HB {
            for b in 0..HB {
                let mut s = 0.0;
                for dr in -1i32..=1 {
                    for dg in -1i32..=1 {
                        for db in -1i32..=1 {
                            let (rr, gg, bb) = (r as i32 + dr, g as i32 + dg, b as i32 + db);
                            if rr >= 0 && gg >= 0 && bb >= 0 && rr < HB as i32 && gg < HB as i32 && bb < HB as i32 {
                                s += hist[((rr as usize) * HB + gg as usize) * HB + bb as usize];
                            }
                        }
                    }
                }
                sm[(r * HB + g) * HB + b] = s;
            }
        }
    }
    let total = n * 27.0 + 1e-3;
    sm.iter().map(|&v| (v + 0.01) / total).collect()
}

/// One quick-selection solve inside a region: `seed` = brushed pixels
/// (hard foreground), `sel` = already selected pixels (soft prior), `bg_h`
/// the background colour model. Returns the foreground labels.
pub fn quick_cut(img: &[[f32; 3]], w: usize, h: usize, seed: &[bool], sel: &[bool], bg_h: &[f32]) -> Vec<bool> {
    let n = w * h;
    let fg_h = histogram((0..n).filter(|&i| seed[i] || sel[i]).map(|i| img[i]));
    let b = beta(img, w, h);
    let lambda = 50.0f32;
    cut_grid(
        w,
        h,
        true,
        |i| {
            if seed[i] {
                return (1e9, 0.0);
            }
            let k = hist_bin(&img[i]);
            let (pf, pb) = (fg_h[k], bg_h[k]);
            let df = -(pf / (pf + pb)).ln();
            let db = -(pb / (pf + pb)).ln();
            (db * 4.0, df * 4.0)
        },
        |i, j| {
            let d = (img[i][0] - img[j][0]).powi(2) + (img[i][1] - img[j][1]).powi(2) + (img[i][2] - img[j][2]).powi(2);
            lambda * (-b * d).exp()
        },
    )
}

// ----- coarse mask → level-0 MaskGrid -------------------------------------------------------

/// A binary mask on a region of mip level `level`.
pub struct Coarse {
    pub level: u32,
    pub x0: i32,
    pub y0: i32,
    pub w: usize,
    pub h: usize,
    pub m: Vec<u8>,
}

impl Coarse {
    #[inline]
    fn at(&self, x: i32, y: i32) -> u8 {
        let (lx, ly) = (x - self.x0, y - self.y0);
        if lx < 0 || ly < 0 || lx >= self.w as i32 || ly >= self.h as i32 {
            return 0;
        }
        self.m[ly as usize * self.w + lx as usize]
    }

    /// Bilinear coverage at level-0 pixel center (x, y).
    #[inline]
    fn sample(&self, x: f32, y: f32) -> f32 {
        let s = (1u32 << self.level) as f32;
        let (u, v) = (x / s - 0.5, y / s - 0.5);
        let (x0, y0) = (u.floor(), v.floor());
        let (fx, fy) = (u - x0, v - y0);
        let (xi, yi) = (x0 as i32, y0 as i32);
        let a = self.at(xi, yi) as f32;
        let b = self.at(xi + 1, yi) as f32;
        let c = self.at(xi, yi + 1) as f32;
        let d = self.at(xi + 1, yi + 1) as f32;
        ((a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy) / 255.0
    }
}

/// Running box mean of radius r over a w×h f32 plane (clamped edges).
#[cfg(test)]
fn box_mean(src: &[f32], w: usize, h: usize, r: usize, out: &mut [f32], tmp: &mut Vec<f32>) {
    tmp.resize(w * h, 0.0);
    let k = (2 * r + 1) as f32;
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let mut acc = 0.0f32;
        for i in 0..=2 * r {
            acc += row[(i as i32 - r as i32).clamp(0, w as i32 - 1) as usize];
        }
        for x in 0..w {
            tmp[y * w + x] = acc / k;
            let add = row[(x + r + 1).min(w - 1)];
            let sub = row[(x as i32 - r as i32).clamp(0, w as i32 - 1) as usize];
            acc += add - sub;
        }
    }
    let mut acc = vec![0f32; w];
    for i in 0..=2 * r {
        let yy = (i as i32 - r as i32).clamp(0, h as i32 - 1) as usize;
        for x in 0..w {
            acc[x] += tmp[yy * w + x];
        }
    }
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = acc[x] / k;
        }
        let ya = (y + r + 1).min(h - 1);
        let ys = (y as i32 - r as i32).clamp(0, h as i32 - 1) as usize;
        for x in 0..w {
            acc[x] += tmp[ya * w + x] - tmp[ys * w + x];
        }
    }
}

/// Guided filter (He et al.) of `p` with gray guide `g`.
#[cfg(test)]
fn guided_reference(g: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let n = w * h;
    let mut tmp = Vec::new();
    let mut mi = vec![0f32; n];
    let mut mp = vec![0f32; n];
    let mut mii = vec![0f32; n];
    let mut mip = vec![0f32; n];
    box_mean(g, w, h, r, &mut mi, &mut tmp);
    box_mean(p, w, h, r, &mut mp, &mut tmp);
    let gg: Vec<f32> = g.iter().map(|v| v * v).collect();
    let gp: Vec<f32> = g.iter().zip(p).map(|(a, b)| a * b).collect();
    box_mean(&gg, w, h, r, &mut mii, &mut tmp);
    box_mean(&gp, w, h, r, &mut mip, &mut tmp);
    let mut a = vec![0f32; n];
    let mut b = vec![0f32; n];
    for i in 0..n {
        let var = mii[i] - mi[i] * mi[i];
        let cov = mip[i] - mi[i] * mp[i];
        a[i] = cov / (var + eps);
        b[i] = mp[i] - a[i] * mi[i];
    }
    let mut ma = vec![0f32; n];
    let mut mb = vec![0f32; n];
    box_mean(&a, w, h, r, &mut ma, &mut tmp);
    box_mean(&b, w, h, r, &mut mb, &mut tmp);
    (0..n).map(|i| ma[i] * g[i] + mb[i]).collect()
}

/// Independent channels share indexing and use SIMD lanes, while preserving
/// each channel's original addition/subtraction/division order exactly.
fn box_mean_channels<const C: usize>(src: &[[f32; C]], w: usize, h: usize, r: usize) -> Vec<[f32; C]> {
    let mut tmp = vec![[0.0; C]; w * h];
    let mut out = vec![[0.0; C]; w * h];
    let k = (2 * r + 1) as f32;
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let mut acc = [0.0; C];
        for i in 0..=2 * r {
            let value = row[(i as i32 - r as i32).clamp(0, w as i32 - 1) as usize];
            for c in 0..C {
                acc[c] += value[c];
            }
        }
        for x in 0..w {
            tmp[y * w + x] = acc.map(|v| v / k);
            let add = row[(x + r + 1).min(w - 1)];
            let sub = row[(x as i32 - r as i32).clamp(0, w as i32 - 1) as usize];
            for c in 0..C {
                acc[c] += add[c] - sub[c];
            }
        }
    }
    let mut acc = vec![[0.0; C]; w];
    for i in 0..=2 * r {
        let yy = (i as i32 - r as i32).clamp(0, h as i32 - 1) as usize;
        for x in 0..w {
            for c in 0..C {
                acc[x][c] += tmp[yy * w + x][c];
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = acc[x].map(|v| v / k);
        }
        let ya = (y + r + 1).min(h - 1);
        let ys = (y as i32 - r as i32).clamp(0, h as i32 - 1) as usize;
        for x in 0..w {
            for c in 0..C {
                acc[x][c] += tmp[ya * w + x][c] - tmp[ys * w + x][c];
            }
        }
    }
    out
}

/// Guided filter (He et al.) of `p` with gray guide `g`.
pub fn guided(g: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let moments: Vec<_> = g.iter().zip(p).map(|(&g, &p)| [g, p, g * g, g * p]).collect();
    let means = box_mean_channels(&moments, w, h, r);
    drop(moments);
    let coefficients: Vec<_> = means.iter().map(|&[mi, mp, mii, mip]| {
        let var = mii - mi * mi;
        let cov = mip - mi * mp;
        let a = cov / (var + eps);
        [a, mp - a * mi]
    }).collect();
    drop(means);
    box_mean_channels(&coefficients, w, h, r).iter().zip(g).map(|(&[ma, mb], &g)| ma * g + mb).collect()
}

/// Upsamples a coarse mask to a level-0 `MaskGrid` of `doc_w x doc_h`.
/// With a guide source, boundary tiles are guided-filtered against the
/// full-resolution luminance; interior and exterior tiles cost nothing.
pub fn upsample(c: &Coarse, doc_w: u32, doc_h: u32, guide: Option<&Source>) -> MaskGrid {
    let mut out = MaskGrid::new(doc_w, doc_h);
    let s = 1i32 << c.level;
    let bounds = RectI::new(c.x0 * s, c.y0 * s, (c.x0 + c.w as i32) * s, (c.y0 + c.h as i32) * s).intersect(&RectI::new(0, 0, doc_w as i32, doc_h as i32));
    let (tx0, ty0, tx1, ty1) = bounds.tile_span(out.cols, out.rows);
    let r = (s as usize * 2).clamp(2, 16);
    let tiles: Vec<(i32, i32)> = (ty0..ty1).flat_map(|ty| (tx0..tx1).map(move |tx| (tx, ty))).collect();
    let done: Vec<(usize, MaskTile)> = tiles
        .par_iter()
        .map(|&(tx, ty)| {
            let idx = (ty as u32 * out.cols + tx as u32) as usize;
            // Footprint on the coarse grid (+1 for bilinear, + margin).
            let m = (r as i32 / s) + 2;
            let (cx0, cy0) = ((tx * TS_I) / s - m, (ty * TS_I) / s - m);
            let (cx1, cy1) = (((tx + 1) * TS_I) / s + m, ((ty + 1) * TS_I) / s + m);
            let (mut any, mut all) = (false, true);
            for y in cy0..cy1 {
                for x in cx0..cx1 {
                    let v = c.at(x, y) >= 128;
                    any |= v;
                    all &= v;
                }
            }
            if !any {
                return (idx, MaskTile::Empty);
            }
            if all {
                return (idx, MaskTile::Full);
            }
            let pad = if guide.is_some() { r } else { 0 };
            let (pw, ph) = (TS + 2 * pad, TS + 2 * pad);
            let (gx0, gy0) = (tx * TS_I - pad as i32, ty * TS_I - pad as i32);
            let mut p = vec![0f32; pw * ph];
            for y in 0..ph {
                for x in 0..pw {
                    p[y * pw + x] = c.sample((gx0 + x as i32) as f32 + 0.5, (gy0 + y as i32) as f32 + 0.5);
                }
            }
            let vals: Vec<f32> = match guide {
                Some(src) => {
                    let mut buf = vec![0u32; pw * ph];
                    src(0, gx0, gy0, pw, ph, &mut buf);
                    let g: Vec<f32> = buf
                        .iter()
                        .map(|&q| {
                            let z = rgb_f(q);
                            (0.299 * z[0] + 0.587 * z[1] + 0.114 * z[2]) / 255.0
                        })
                        .collect();
                    let q = guided(&g, &p, pw, ph, r, 1e-3);
                    // Sharpen the guided result back toward a crisp edge.
                    q.iter().map(|&v| ((v - 0.5) * 1.6 + 0.5).clamp(0.0, 1.0)).collect()
                }
                None => p,
            };
            let mut px = vec![0u8; TS * TS];
            for y in 0..TS {
                for x in 0..TS {
                    px[y * TS + x] = (vals[(y + pad) * pw + x + pad] * 255.0 + 0.5) as u8;
                }
            }
            (idx, super::masks::normalize(px))
        })
        .collect();
    for (i, t) in done {
        out.tiles[i] = t;
    }
    out
}

/// Reads a region at a level as float RGB.
fn read_rgb(src: &Source, level: u32, x0: i32, y0: i32, w: usize, h: usize) -> Vec<[f32; 3]> {
    let mut buf = vec![0u32; w * h];
    src(level, x0, y0, w, h, &mut buf);
    buf.iter().map(|&c| rgb_f(c)).collect()
}

/// Picks the level where a level-0 region of `area` pixels has at most
/// `budget` pixels.
pub fn level_for(area: f64, budget: f64, max_level: u32) -> u32 {
    let mut l = 0;
    while area / 4f64.powi(l as i32) > budget && l < max_level {
        l += 1;
    }
    l
}

/// Object Selection: GrabCut inside `rect` (or a lasso polygon) at a level
/// where the region is about `budget` pixels.
#[allow(clippy::too_many_arguments)]
pub fn object_select(src: &Source, doc_w: u32, doc_h: u32, max_level: u32, rect: RectI, lasso: Option<&[P]>, budget: f64, refine: bool) -> Option<MaskGrid> {
    let doc = RectI::new(0, 0, doc_w as i32, doc_h as i32);
    let rect = rect.intersect(&doc);
    if rect.w() < 4 || rect.h() < 4 {
        return None;
    }
    let margin = (rect.w().max(rect.h()) / 10).max(8);
    let region = rect.inflate(margin).intersect(&doc);
    let level = level_for(region.w() as f64 * region.h() as f64, budget, max_level);
    let s = 1i32 << level;
    let (x0, y0) = (region.x0 / s, region.y0 / s);
    let (w, h) = (((region.x1 + s - 1) / s - x0) as usize, ((region.y1 + s - 1) / s - y0) as usize);
    let img = read_rgb(src, level, x0, y0, w, h);
    let poly = lasso.map(|pts| vec![Poly { pts: pts.iter().map(|q| super::vector::p(q.x / s as f32, q.y / s as f32)).collect(), closed: true }]);
    let unknown: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = ((x0 + (i % w) as i32) as f32 + 0.5, (y0 + (i / w) as i32) as f32 + 0.5);
            match &poly {
                Some(pl) => contains(super::vector::p(x, y), pl, super::vector::FillRule::NonZero),
                None => x * s as f32 >= rect.x0 as f32 && x * s as f32 <= rect.x1 as f32 && y * s as f32 >= rect.y0 as f32 && y * s as f32 <= rect.y1 as f32,
            }
        })
        .collect();
    let t0 = std::time::Instant::now();
    // Coarse-to-fine: solve at ~0.1 MP first and use it to seed the models.
    let init = if refine && level > 0 {
        let cl = level_for(region.w() as f64 * region.h() as f64, 0.12e6, max_level).max(level + 1);
        let cs = 1i32 << cl;
        let (cx0, cy0) = (region.x0 / cs, region.y0 / cs);
        let (cw, ch) = (((region.x1 + cs - 1) / cs - cx0) as usize, ((region.y1 + cs - 1) / cs - cy0) as usize);
        let cimg = read_rgb(src, cl, cx0, cy0, cw, ch);
        let cunk: Vec<bool> = (0..cw * ch)
            .map(|i| {
                let (x, y) = ((cx0 + (i % cw) as i32) * cs + cs / 2, (cy0 + (i / cw) as i32) * cs + cs / 2);
                let (xf, yf) = (x as f32 / s as f32 - x0 as f32, y as f32 / s as f32 - y0 as f32);
                let (xi, yi) = (xf as usize, yf as usize);
                xi < w && yi < h && unknown[yi * w + xi]
            })
            .collect();
        let mut cfg = grabcut(&cimg, cw, ch, &cunk, 3);
        clean(&mut cfg, cw, ch, 0.08);
        let k = 1usize << (cl - level);
        Some(
            (0..w * h)
                .map(|i| {
                    let (x, y) = ((x0 as usize + i % w) / k, (y0 as usize + i / w) / k);
                    let (lx, ly) = (x as i64 - cx0 as i64, y as i64 - cy0 as i64);
                    lx >= 0 && ly >= 0 && (lx as usize) < cw && (ly as usize) < ch && cfg[ly as usize * cw + lx as usize]
                })
                .collect(),
        )
    } else {
        None
    };
    let mut fg = grabcut_init(&img, w, h, &unknown, init, if refine { 3 } else { 2 });
    clean(&mut fg, w, h, 0.08);
    let t1 = std::time::Instant::now();
    let c = Coarse { level, x0, y0, w, h, m: fg.iter().map(|&b| if b { 255 } else { 0 }).collect() };
    let m = upsample(&c, doc_w, doc_h, if refine { Some(src) } else { None });
    if std::env::var("PF_TIMING").is_ok() {
        eprintln!("  object_select: grabcut {:.1} ms, upsample {:.1} ms", (t1 - t0).as_secs_f64() * 1000.0, t1.elapsed().as_secs_f64() * 1000.0);
    }
    (!m.is_empty()).then_some(m)
}

/// Quick Selection state for one brushing gesture.
pub struct QuickSel {
    pub level: u32,
    seeds: std::collections::HashSet<(i32, i32)>,
    lo: (i32, i32),
    hi: (i32, i32),
    radius: f32,
    /// Background colour model: the whole working-level image.
    bg: std::sync::OnceLock<Vec<f32>>,
}

impl QuickSel {
    pub fn new(doc_w: u32, doc_h: u32, max_level: u32) -> Self {
        // Work where the whole document is about 1 MP.
        let level = level_for(doc_w as f64 * doc_h as f64, 1.0e6, max_level);
        QuickSel { level, seeds: Default::default(), lo: (i32::MAX, i32::MAX), hi: (i32::MIN, i32::MIN), radius: 1.0, bg: Default::default() }
    }

    /// Adds a brush dab (level-0 center and radius) to the seeds.
    pub fn add_dab(&mut self, x: f32, y: f32, r: f32) {
        let s = (1u32 << self.level) as f32;
        let (cx, cy, rr) = (x / s, y / s, (r / s).max(0.75));
        self.radius = self.radius.max(rr);
        let ri = rr.ceil() as i32;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                if (dx * dx + dy * dy) as f32 <= rr * rr {
                    let q = (cx as i32 + dx, cy as i32 + dy);
                    self.seeds.insert(q);
                    self.lo = (self.lo.0.min(q.0), self.lo.1.min(q.1));
                    self.hi = (self.hi.0.max(q.0), self.hi.1.max(q.1));
                }
            }
        }
    }

    fn bg_model(&self, src: &Source, lw: i32, lh: i32) -> &Vec<f32> {
        self.bg.get_or_init(|| {
            // Subsample the working level to ≤ 64k pixels.
            let step = (((lw as f64 * lh as f64) / 65536.0).sqrt().ceil() as i32).max(1);
            let img = read_rgb(src, self.level, 0, 0, lw as usize, lh as usize);
            let it = (0..lh).step_by(step as usize).flat_map(|y| (0..lw).step_by(step as usize).map(move |x| (x, y)));
            histogram(it.map(|(x, y)| img[(y * lw + x) as usize]).collect::<Vec<_>>().into_iter())
        })
    }

    /// Solves at the coarse level; `cur` is the existing selection. The
    /// region of interest starts around the strokes and doubles while the
    /// result touches its border, so it grows to the object's edges.
    pub fn solve(&self, src: &Source, doc_w: u32, doc_h: u32, cur: Option<&MaskGrid>) -> Option<Coarse> {
        if self.seeds.is_empty() {
            return None;
        }
        let s = 1i32 << self.level;
        let (lw, lh) = (((doc_w as i32) + s - 1) / s, ((doc_h as i32) + s - 1) / s);
        let bg = self.bg_model(src, lw, lh);
        let mut grow = (self.radius * 4.0) as i32 + 24;
        loop {
            let x0 = (self.lo.0 - grow).max(0);
            let y0 = (self.lo.1 - grow).max(0);
            let x1 = (self.hi.0 + grow + 1).min(lw);
            let y1 = (self.hi.1 + grow + 1).min(lh);
            let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
            if w < 3 || h < 3 {
                return None;
            }
            let img = read_rgb(src, self.level, x0, y0, w, h);
            let mut seed = vec![false; w * h];
            for &(x, y) in &self.seeds {
                if x >= x0 && y >= y0 && x < x1 && y < y1 {
                    seed[(y - y0) as usize * w + (x - x0) as usize] = true;
                }
            }
            let sel: Vec<bool> = (0..w * h)
                .map(|i| {
                    cur.is_some_and(|m| {
                        let (x, y) = ((x0 + (i % w) as i32) * s + s / 2, (y0 + (i / w) as i32) * s + s / 2);
                        m.at(x, y) >= 128
                    })
                })
                .collect();
            let mut fg = quick_cut(&img, w, h, &seed, &sel, bg);
            // Keep only what is connected to the strokes.
            keep_seeded(&mut fg, &seed, w, h);
            let at_edge = |x: i32, y: i32| (x == 0 && x0 > 0) || (y == 0 && y0 > 0) || (x == w as i32 - 1 && x1 < lw) || (y == h as i32 - 1 && y1 < lh);
            let touches = (0..w * h).any(|i| fg[i] && at_edge((i % w) as i32, (i / w) as i32));
            let whole = x0 == 0 && y0 == 0 && x1 == lw && y1 == lh;
            if touches && !whole {
                grow *= 2;
                continue;
            }
            return Some(Coarse { level: self.level, x0, y0, w, h, m: fg.iter().map(|&b| if b { 255 } else { 0 }).collect() });
        }
    }
}

/// Clears foreground components that contain no seed pixel.
fn keep_seeded(fg: &mut [bool], seed: &[bool], w: usize, h: usize) {
    let mut keep = vec![false; w * h];
    let mut stack: Vec<usize> = (0..w * h).filter(|&i| seed[i] && fg[i]).collect();
    for &i in &stack {
        keep[i] = true;
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        let mut push = |j: usize| {
            if fg[j] && !keep[j] {
                keep[j] = true;
                stack.push(j);
            }
        };
        if x > 0 {
            push(i - 1);
        }
        if x + 1 < w {
            push(i + 1);
        }
        if y > 0 {
            push(i - w);
        }
        if y + 1 < h {
            push(i + w);
        }
    }
    fg.copy_from_slice(&keep);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::grid::{TileGrid, tile_mut};

    #[test]
    fn fused_guided_filter_is_bit_exact_with_the_separate_passes() {
        for (w, h) in [(1, 1), (1, 29), (31, 1), (17, 23), (288, 288)] {
            let g: Vec<_> = (0..w * h).map(|i| ((i * 31 + i / w * 17) % 257) as f32 / 256.0).collect();
            let p: Vec<_> = (0..w * h).map(|i| if i % 11 < 6 { 0.0 } else { 1.0 }).collect();
            for radius in [0, 1, 3, 16] {
                let reference = guided_reference(&g, &p, w, h, radius, 1e-3);
                let fused = guided(&g, &p, w, h, radius, 1e-3);
                for (i, (a, b)) in fused.iter().zip(&reference).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "{w}x{h} r{radius}, pixel {i}");
                }
            }
        }
    }

    /// A textured red disk on a textured green background.
    fn scene(n: u32) -> TileGrid {
        let mut g = TileGrid::new(n, n);
        let c = n as f32 * 0.5;
        for y in 0..n as i32 {
            for x in 0..n as i32 {
                let d = ((x as f32 + 0.5 - c).powi(2) + (y as f32 + 0.5 - c).powi(2)).sqrt();
                let noise = (crate::engine::filters::util::hash01(x, y, 3) * 40.0) as u8;
                let col = if d < n as f32 * 0.25 { px::pack(200 + noise / 2, 40 + noise, 30, 255) } else { px::pack(40 + noise, 150 + noise / 2, 60, 255) };
                let i = g.idx(x >> 8, y >> 8);
                tile_mut(&mut g.tiles[i]).write()[((y & 255) as usize) * TS + (x & 255) as usize] = col;
            }
        }
        g
    }

    fn accuracy(m: &MaskGrid, n: u32) -> f32 {
        let c = n as f32 * 0.5;
        let mut ok = 0;
        let mut tot = 0;
        for y in (0..n as i32).step_by(7) {
            for x in (0..n as i32).step_by(7) {
                let d = ((x as f32 + 0.5 - c).powi(2) + (y as f32 + 0.5 - c).powi(2)).sqrt();
                if (d - n as f32 * 0.25).abs() < 3.0 {
                    continue;
                }
                let want = d < n as f32 * 0.25;
                if (m.at(x, y) >= 128) == want {
                    ok += 1;
                }
                tot += 1;
            }
        }
        ok as f32 / tot as f32
    }

    #[test]
    fn grabcut_finds_the_disk() {
        let n = 1024u32;
        let g = scene(n);
        let layer = crate::engine::document::Layer::new(1, "s", g);
        let src = |l: u32, x: i32, y: i32, w: usize, h: usize, out: &mut [u32]| layer.level(l).read_region_clamped(x, y, w, h, out);
        let rect = RectI::new(220, 220, 804, 804);
        let coarse = object_select(&src, n, n, layer.mips.len() as u32, rect, None, 0.12e6, false).unwrap();
        assert!(accuracy(&coarse, n) > 0.98, "coarse accuracy {}", accuracy(&coarse, n));
        let fine = object_select(&src, n, n, layer.mips.len() as u32, rect, None, 1.0e6, true).unwrap();
        assert!(accuracy(&fine, n) > 0.99, "refined accuracy {}", accuracy(&fine, n));
    }

    #[test]
    fn quick_selection_grows_to_edges() {
        let n = 1024u32;
        let layer = crate::engine::document::Layer::new(1, "s", scene(n));
        let src = |l: u32, x: i32, y: i32, w: usize, h: usize, out: &mut [u32]| layer.level(l).read_region_clamped(x, y, w, h, out);
        let mut q = QuickSel::new(n, n, layer.mips.len() as u32);
        for k in 0..20 {
            q.add_dab(420.0 + k as f32 * 9.0, 500.0, 12.0);
        }
        let c = q.solve(&src, n, n, None).unwrap();
        let m = upsample(&c, n, n, Some(&src));
        assert!(accuracy(&m, n) > 0.98, "accuracy {}", accuracy(&m, n));
    }

    #[test]
    fn guided_filter_preserves_constant() {
        let (w, h) = (40, 30);
        let g: Vec<f32> = (0..w * h).map(|i| (i % 7) as f32 / 7.0).collect();
        let p = vec![0.6f32; w * h];
        let q = guided(&g, &p, w, h, 3, 1e-3);
        assert!(q.iter().all(|&v| (v - 0.6).abs() < 1e-3));
    }
}
