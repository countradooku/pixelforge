//! Select Subject, model-backed Object Selection and the Object Selection
//! tool's hover **object finder**.
//!
//! - With `models/select_subject.onnx` and the `ml` feature, Select Subject
//!   runs the salient-object net at ≤ 1024 px and the probability map is
//!   upsampled by `segment::upsample` (guided filter on boundary tiles
//!   only). Object Selection runs the same net on the box's crop.
//!   Without a model both fall back to GrabCut (`segment::object_select`).
//! - The **object finder** is precomputed in the background when the tool
//!   is activated: the subject's connected components plus a
//!   Felzenszwalb–Huttenlocher segmentation at ≈ 0.25 MP whose regions are
//!   merged (region adjacency graph, by colour distance weighted by size)
//!   into object-sized regions. Each candidate is kept when it is not mostly
//!   on the image border and covers 0.15–40 % of the frame. Hover is then a
//!   label lookup plus cached outlines (no per-move work), and a click
//!   upsamples that candidate with the guided filter.

use rayon::prelude::*;

use super::grid::RectI;
use super::ml;
use super::segment::{self, Coarse, Source};
use super::selection::{MaskGrid, MaskTile, Selection};

/// Straight RGB of a whole mip level (over white).
fn level_rgb(src: &Source, level: u32, w: usize, h: usize) -> Vec<[u8; 3]> {
    let mut buf = vec![0u32; w * h];
    src(level, 0, 0, w, h, &mut buf);
    buf.par_iter()
        .map(|&c| {
            let [r, g, b, a] = super::px::unpack(c);
            let k = 255 - a;
            [r.saturating_add(k), g.saturating_add(k), b.saturating_add(k)]
        })
        .collect()
}

fn level_dims(doc_w: u32, doc_h: u32, l: u32) -> (usize, usize) {
    ((doc_w as usize).div_ceil(1 << l), (doc_h as usize).div_ceil(1 << l))
}

/// The coarsest level whose long side is still ≥ `side` (or 0).
fn level_for_side(doc_w: u32, doc_h: u32, side: u32, max_level: u32) -> u32 {
    let mut l = 0;
    while l < max_level && (doc_w.max(doc_h) >> (l + 1)) >= side {
        l += 1;
    }
    l
}

/// Bilinear resample of a `sw x sh` plane to `w x h`.
fn resample(p: &[f32], sw: usize, sh: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = ((y as f32 + 0.5) * sh as f32 / h as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
        let (y0, ty) = (fy as usize, fy.fract());
        let y1 = (y0 + 1).min(sh - 1);
        for (x, o) in row.iter_mut().enumerate() {
            let fx = ((x as f32 + 0.5) * sw as f32 / w as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
            let (x0, tx) = (fx as usize, fx.fract());
            let x1 = (x0 + 1).min(sw - 1);
            let a = p[y0 * sw + x0] * (1.0 - tx) + p[y0 * sw + x1] * tx;
            let b = p[y1 * sw + x0] * (1.0 - tx) + p[y1 * sw + x1] * tx;
            *o = a * (1.0 - ty) + b * ty;
        }
    });
    out
}

/// Which algorithm produced a subject / object mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Neural,
    GrabCut,
}

/// Select ▸ Subject. Neural when `select_subject.onnx` is available,
/// otherwise GrabCut with a boundary-connectivity prior (inset fallback).
pub fn select_subject(src: &Source, doc_w: u32, doc_h: u32, max_level: u32) -> (Option<MaskGrid>, Backend) {
    if ml::ready(ml::SUBJECT_MODEL) {
        let l = level_for_side(doc_w, doc_h, 1024, max_level);
        let (w, h) = level_dims(doc_w, doc_h, l);
        let rgb = level_rgb(src, l, w, h);
        if let Some((p, pw, ph)) = ml::subject_probability(&rgb, w, h) {
            let prob = resample(&p, pw, ph, w, h);
            let c = Coarse { level: l, x0: 0, y0: 0, w, h, m: prob.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect() };
            let m = segment::upsample(&c, doc_w, doc_h, Some(src));
            return ((!m.is_empty()).then_some(m), Backend::Neural);
        }
    }
    (classical_subject(src, doc_w, doc_h, max_level), Backend::GrabCut)
}

/// Background probability per pixel from **boundary connectivity** (Zhu,
/// Liang, Wei & Sun, "Saliency Optimization from Robust Background
/// Detection", CVPR 2014): superpixels (Felzenszwalb–Huttenlocher), a
/// region adjacency graph weighted by Lab distance, geodesic distances by
/// Dijkstra, then for each region the soft area it spans and the soft
/// length of image border it reaches: `BndCon = len / sqrt(area)`, and
/// `P(bg) = 1 − exp(−BndCon² / 2)`. A subject that touches an edge (a
/// portrait's shoulders) reaches the border with a small part of a large
/// spanning area, so it stays foreground; a backdrop does not.
pub fn background_prob(lab: &[[f32; 3]], w: usize, h: usize) -> Vec<f32> {
    let (labels, n) = fh_segment(lab, w, h, 60.0, ((w * h) / 1500).max(8) as u32);
    let mut mean = vec![[0f64; 3]; n];
    let mut cnt = vec![0f64; n];
    let mut border = vec![false; n];
    for (i, &l) in labels.iter().enumerate() {
        for c in 0..3 {
            mean[l as usize][c] += lab[i][c] as f64;
        }
        cnt[l as usize] += 1.0;
        let (x, y) = (i % w, i / w);
        if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
            border[l as usize] = true;
        }
    }
    for (m, c) in mean.iter_mut().zip(&cnt) {
        m.iter_mut().for_each(|v| *v /= c.max(1.0));
    }
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for y in 0..h {
        for x in 0..w {
            let a = labels[y * w + x] as usize;
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx < w && ny < h {
                    let b = labels[ny * w + nx] as usize;
                    if a != b && !adj[a].contains(&b) {
                        adj[a].push(b);
                        adj[b].push(a);
                    }
                }
            }
        }
    }
    let dc = |a: usize, b: usize| ((mean[a][0] - mean[b][0]).powi(2) + (mean[a][1] - mean[b][1]).powi(2) + (mean[a][2] - mean[b][2]).powi(2)).sqrt();
    let sigma = 10.0f64;
    let bnd: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|p| {
            // Dijkstra over the region graph.
            let mut d = vec![f64::MAX; n];
            let mut heap = std::collections::BinaryHeap::new();
            d[p] = 0.0;
            heap.push(std::cmp::Reverse((0u64, p)));
            while let Some(std::cmp::Reverse((dk, u))) = heap.pop() {
                let du = f64::from_bits(dk);
                if du > d[u] {
                    continue;
                }
                for &v in &adj[u] {
                    let nd = du + dc(u, v);
                    if nd < d[v] {
                        d[v] = nd;
                        heap.push(std::cmp::Reverse((nd.to_bits(), v)));
                    }
                }
            }
            let (mut area, mut len) = (0f64, 0f64);
            for q in 0..n {
                if d[q] == f64::MAX {
                    continue;
                }
                let s = (-(d[q] * d[q]) / (2.0 * sigma * sigma)).exp();
                area += s;
                if border[q] {
                    len += s;
                }
            }
            len / area.max(1e-9).sqrt()
        })
        .collect();
    let pbg: Vec<f32> = bnd.iter().map(|b| (1.0 - (-(b * b) / 2.0).exp()) as f32).collect();
    // Normalize per image (textured backdrops split into many superpixels
    // and never reach the absolute maximum).
    let top = pbg.iter().copied().fold(0f32, f32::max).max(1e-3);
    labels.iter().map(|&l| pbg[l as usize] / top).collect()
}

/// Share the classical subject prior with Object Finder so both use the
/// same colour-aware background constraints, including subjects at an edge.
fn classical_prior(rgb8: &[[u8; 3]], w: usize, h: usize, prgb: &[[u8; 3]], pw: usize, ph: usize) -> Option<(Vec<bool>, Vec<bool>)> {
    let plab: Vec<[f32; 3]> = prgb.iter().map(|c| super::color::srgb8_to_lab(*c)).collect();
    let pbg = background_prob(&plab, pw, ph);
    let at = |x: usize, y: usize| pbg[(y * ph / h).min(ph - 1) * pw + (x * pw / w).min(pw - 1)];
    let (tu, ti) = (0.85f32, 0.4f32);
    let init: Vec<bool> = (0..w * h).map(|i| at(i % w, i / w) < ti).collect();
    // Hard background only where the prior is sure *and* the colour is
    // rare in the likely subject: hair that runs off the frame keeps the
    // subject's colours, so it stays undecided for GrabCut.
    let lab: Vec<[f32; 3]> = rgb8.iter().map(|c| super::color::srgb8_to_lab(*c)).collect();
    let bin = |c: [f32; 3]| {
        let b = |v: f32, lo: f32, hi: f32| (((v - lo) / (hi - lo) * 8.0) as i32).clamp(0, 7) as usize;
        b(c[0], 0.0, 100.0) * 64 + b(c[1], -80.0, 80.0) * 8 + b(c[2], -80.0, 80.0)
    };
    let (mut hf, mut hb) = (vec![0f32; 512], vec![0f32; 512]);
    for i in 0..w * h {
        let p = at(i % w, i / w);
        if p < ti {
            hf[bin(lab[i])] += 1.0;
        } else if p >= tu {
            hb[bin(lab[i])] += 1.0;
        }
    }
    let (nf, nb) = (hf.iter().sum::<f32>().max(1.0), hb.iter().sum::<f32>().max(1.0));
    let unknown: Vec<bool> = (0..w * h)
        .map(|i| {
            let k = bin(lab[i]);
            at(i % w, i / w) < tu || hf[k] / nf > 0.5 * hb[k] / nb
        })
        .collect();
    if std::env::var("PF_TIMING").is_ok() {
        let f = |v: &[bool]| v.iter().filter(|&&b| b).count() as f32 / v.len() as f32;
        let mut hist = [0usize; 10];
        for &p in &pbg {
            hist[((p * 10.0) as usize).min(9)] += 1;
        }
        eprintln!("  subject prior: init {:.2}, unknown {:.2}, pbg hist {hist:?}", f(&init), f(&unknown));
    }
    if !init.iter().any(|&b| b) || unknown.iter().all(|&b| b) {
        None
    } else {
        Some((unknown, init))
    }
}

/// Classical Select Subject: a boundary-connectivity background prior
/// fixes clear background, seeds GrabCut's foreground with likely subject
/// regions, and GrabCut + the guided-filter edge do the rest.
pub fn classical_subject(src: &Source, doc_w: u32, doc_h: u32, max_level: u32) -> Option<MaskGrid> {
    let l = segment::level_for(doc_w as f64 * doc_h as f64, 0.6e6, max_level);
    let (w, h) = level_dims(doc_w, doc_h, l);
    let rgb8 = level_rgb(src, l, w, h);
    // The prior at ≈ 0.15 MP (two levels coarser at most), mapped up.
    let pl = (l + 1).min(max_level);
    let (pw, ph) = level_dims(doc_w, doc_h, pl);
    let prgb = level_rgb(src, pl, pw, ph);
    let Some((unknown, init)) = classical_prior(&rgb8, w, h, &prgb, pw, ph) else {
        // No usable prior: fall back to the inset rectangle.
        let r = RectI::new(doc_w as i32 / 25, doc_h as i32 / 25, doc_w as i32 * 24 / 25, doc_h as i32 * 24 / 25);
        return segment::object_select(src, doc_w, doc_h, max_level, r, None, 1.0e6, true);
    };
    let img: Vec<[f32; 3]> = rgb8.iter().map(|c| c.map(|v| v as f32)).collect();
    let mut fg = segment::grabcut_init(&img, w, h, &unknown, Some(init), 4);
    segment::clean(&mut fg, w, h, 0.08);
    let c = Coarse { level: l, x0: 0, y0: 0, w, h, m: fg.iter().map(|&b| if b { 255 } else { 0 }).collect() };
    let m = segment::upsample(&c, doc_w, doc_h, Some(src));
    (!m.is_empty()).then_some(m)
}

/// Object Selection inside `rect` with the salient-object net on the crop
/// (the prompt box limits the result). `None` when no model is available.
pub fn object_in_box(src: &Source, doc_w: u32, doc_h: u32, max_level: u32, rect: RectI) -> Option<MaskGrid> {
    if !ml::ready(ml::SUBJECT_MODEL) {
        return None;
    }
    let doc = RectI::new(0, 0, doc_w as i32, doc_h as i32);
    let rect = rect.intersect(&doc);
    let region = rect.inflate((rect.w().max(rect.h()) / 10).max(8)).intersect(&doc);
    let mut l = 0;
    while l < max_level && (region.w().max(region.h()) >> (l + 1)) >= 1024 {
        l += 1;
    }
    let s = 1i32 << l;
    let (x0, y0) = (region.x0 / s, region.y0 / s);
    let (w, h) = (((region.x1 + s - 1) / s - x0) as usize, ((region.y1 + s - 1) / s - y0) as usize);
    let mut buf = vec![0u32; w * h];
    src(l, x0, y0, w, h, &mut buf);
    let rgb: Vec<[u8; 3]> = buf.iter().map(|&c| {
        let [r, g, b, _] = super::px::unpremultiply(c);
        [r, g, b]
    }).collect();
    let (p, pw, ph) = ml::subject_probability(&rgb, w, h)?;
    let prob = resample(&p, pw, ph, w, h);
    let inside = |i: usize| {
        let (x, y) = ((x0 + (i % w) as i32) * s, (y0 + (i / w) as i32) * s);
        x >= rect.x0 && x < rect.x1 && y >= rect.y0 && y < rect.y1
    };
    let m: Vec<u8> = (0..w * h).map(|i| if inside(i) { (prob[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8 } else { 0 }).collect();
    let mg = segment::upsample(&Coarse { level: l, x0, y0, w, h, m }, doc_w, doc_h, Some(src));
    (!mg.is_empty()).then_some(mg)
}

// ----- object finder -----------------------------------------------------------------------

/// Union-find with region sizes.
struct Dsu {
    p: Vec<u32>,
    size: Vec<u32>,
    thr: Vec<f32>,
}

impl Dsu {
    fn new(n: usize, k: f32) -> Self {
        Dsu { p: (0..n as u32).collect(), size: vec![1; n], thr: vec![k; n] }
    }
    fn find(&mut self, mut x: u32) -> u32 {
        while self.p[x as usize] != x {
            let g = self.p[self.p[x as usize] as usize];
            self.p[x as usize] = g;
            x = g;
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) -> u32 {
        let (a, b) = if self.size[a as usize] < self.size[b as usize] { (b, a) } else { (a, b) };
        self.p[b as usize] = a;
        self.size[a as usize] += self.size[b as usize];
        a
    }
}

/// Felzenszwalb–Huttenlocher graph segmentation (8-connected, colour
/// distance weights, scale `k`, minimum region `min`). Returns dense labels.
pub fn fh_segment(rgb: &[[f32; 3]], w: usize, h: usize, k: f32, min: u32) -> (Vec<u32>, usize) {
    let n = w * h;
    let d = |a: usize, b: usize| {
        let (p, q) = (rgb[a], rgb[b]);
        ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
    };
    let mut edges: Vec<(f32, u32, u32)> = (0..h)
        .into_par_iter()
        .flat_map_iter(|y| {
            let mut v = Vec::with_capacity(w * 4);
            for x in 0..w {
                let i = y * w + x;
                if x + 1 < w {
                    v.push((d(i, i + 1), i as u32, (i + 1) as u32));
                }
                if y + 1 < h {
                    v.push((d(i, i + w), i as u32, (i + w) as u32));
                    if x + 1 < w {
                        v.push((d(i, i + w + 1), i as u32, (i + w + 1) as u32));
                    }
                    if x > 0 {
                        v.push((d(i, i + w - 1), i as u32, (i + w - 1) as u32));
                    }
                }
            }
            v
        })
        .collect();
    edges.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut u = Dsu::new(n, k);
    for &(wt, a, b) in &edges {
        let (ra, rb) = (u.find(a), u.find(b));
        if ra != rb && wt <= u.thr[ra as usize] && wt <= u.thr[rb as usize] {
            let r = u.union(ra, rb);
            u.thr[r as usize] = wt + k / u.size[r as usize] as f32;
        }
    }
    for &(_, a, b) in &edges {
        let (ra, rb) = (u.find(a), u.find(b));
        if ra != rb && (u.size[ra as usize] < min || u.size[rb as usize] < min) {
            u.union(ra, rb);
        }
    }
    let mut map = std::collections::HashMap::new();
    let labels: Vec<u32> = (0..n as u32)
        .map(|i| {
            let r = u.find(i);
            let next = map.len() as u32;
            *map.entry(r).or_insert(next)
        })
        .collect();
    (labels, map.len())
}

/// Greedy region-adjacency merging: repeatedly merges the adjacent pair
/// with the smallest mean-colour distance (ΔE in Lab, ties toward small
/// regions) while it is below `max_de` or more than `max_regions` remain.
fn merge_regions(rgb: &[[f32; 3]], labels: &mut [u32], count: usize, w: usize, h: usize, max_de: f64, max_regions: usize) -> usize {
    let mut sum = vec![[0f64; 3]; count];
    let mut size = vec![0f64; count];
    for (i, &l) in labels.iter().enumerate() {
        for c in 0..3 {
            sum[l as usize][c] += rgb[i][c] as f64;
        }
        size[l as usize] += 1.0;
    }
    let mut adj: Vec<std::collections::HashSet<u32>> = vec![Default::default(); count];
    for y in 0..h {
        for x in 0..w {
            let a = labels[y * w + x];
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx < w && ny < h {
                    let b = labels[ny * w + nx];
                    if a != b {
                        adj[a as usize].insert(b);
                        adj[b as usize].insert(a);
                    }
                }
            }
        }
    }
    let mut parent: Vec<u32> = (0..count as u32).collect();
    let mut alive = count;
    let total: f64 = size.iter().sum();
    let cost = |sum: &[[f64; 3]], size: &[f64], a: usize, b: usize| {
        let (ma, mb) = (sum[a].map(|v| v / size[a]), sum[b].map(|v| v / size[b]));
        let dc = ((ma[0] - mb[0]).powi(2) + (ma[1] - mb[1]).powi(2) + (ma[2] - mb[2]).powi(2)).sqrt();
        // Tiny fragments merge first: their mean colour is noisy.
        dc * (0.5 + (size[a].min(size[b]) / total * 200.0).min(1.0))
    };
    loop {
        let mut best = (f64::MAX, 0usize, 0usize);
        for a in 0..count {
            if parent[a] != a as u32 {
                continue;
            }
            for &b in &adj[a] {
                let b = b as usize;
                if b > a {
                    let c = cost(&sum, &size, a, b);
                    if c < best.0 {
                        best = (c, a, b);
                    }
                }
            }
        }
        let (_, a, b) = best;
        if best.0 == f64::MAX || (best.0 > max_de && alive <= max_regions) {
            break;
        }
        parent[b] = a as u32;
        for c in 0..3 {
            sum[a][c] += sum[b][c];
        }
        size[a] += size[b];
        let nb: Vec<u32> = adj[b].drain().collect();
        for c in nb {
            let c = c as usize;
            adj[c].remove(&(b as u32));
            if c != a {
                adj[c].insert(a as u32);
                adj[a].insert(c as u32);
            }
        }
        adj[a].remove(&(a as u32));
        adj[a].remove(&(b as u32));
        alive -= 1;
    }
    let root = |mut x: u32| {
        while parent[x as usize] != x {
            x = parent[x as usize];
        }
        x
    };
    let mut map = std::collections::HashMap::new();
    for l in labels.iter_mut() {
        let r = root(*l);
        let next = map.len() as u32;
        *l = *map.entry(r).or_insert(next);
    }
    map.len()
}

/// One hover candidate.
pub struct FoundObject {
    /// Coarse bounds, used to avoid expanding empty parts of a click mask.
    bounds: (usize, usize, usize, usize),
    /// Outline polylines in level-0 coordinates (cached for hover).
    pub outline: Vec<Vec<[f32; 2]>>,
    pub coverage: f32,
}

/// Precomputed object set for the Object Selection tool's hover finder.
pub struct ObjectFinder {
    pub level: u32,
    pub w: usize,
    pub h: usize,
    /// Object id + 1 per coarse pixel (0 = none).
    labels: Vec<u16>,
    pub objects: Vec<FoundObject>,
    pub backend: Backend,
    pub build_ms: f32,
}

/// A `w x h` u8 coverage as a mask grid (for outlines).
fn to_mask(m: &[u8], w: usize, h: usize) -> MaskGrid {
    let mut g = MaskGrid::new(w as u32, h as u32);
    let cols = g.cols as usize;
    let ts = super::grid::TS;
    g.tiles.par_iter_mut().enumerate().for_each(|(i, t)| {
        let (tx, ty) = (i % cols, i / cols);
        let mut px = vec![0u8; ts * ts];
        for y in 0..ts {
            let gy = ty * ts + y;
            if gy >= h {
                break;
            }
            for x in 0..ts {
                let gx = tx * ts + x;
                if gx < w {
                    px[y * ts + x] = m[gy * w + gx];
                }
            }
        }
        *t = super::masks::normalize(px);
    });
    g
}

impl ObjectFinder {
    /// Builds the candidate set at ≈ 0.25 MP. `subject` is the subject
    /// mask, if already known (its components come first).
    pub fn build(src: &Source, doc_w: u32, doc_h: u32, max_level: u32) -> ObjectFinder {
        let t0 = std::time::Instant::now();
        let l = level_for_side(doc_w, doc_h, 512, max_level);
        let (w, h) = level_dims(doc_w, doc_h, l);
        let rgb8 = level_rgb(src, l, w, h);
        // Subject first (neural when available, else a quick GrabCut).
        let (subj, backend) = if ml::ready(ml::SUBJECT_MODEL) {
            let p = ml::subject_probability(&rgb8, w, h).map(|(p, pw, ph)| resample(&p, pw, ph, w, h));
            (p.map(|p| p.iter().map(|&v| v > 0.5).collect::<Vec<bool>>()), Backend::Neural)
        } else {
            let img: Vec<[f32; 3]> = rgb8.iter().map(|c| c.map(|v| v as f32)).collect();
            // An all-inset foreground model learns backdrop colours too,
            // making the first cut both ambiguous and expensive. Reuse Select
            // Subject's boundary/colour prior without reducing resolution or
            // iterations; keep the rectangle fallback for unusable priors.
            let pl = (l + 1).min(max_level);
            let (pw, ph) = level_dims(doc_w, doc_h, pl);
            let prgb = level_rgb(src, pl, pw, ph);
            let (unknown, init) = match classical_prior(&rgb8, w, h, &prgb, pw, ph) {
                Some((unknown, init)) => (unknown, Some(init)),
                None => {
                    let (mx, my) = ((w / 25).max(1), (h / 25).max(1));
                    let unknown = (0..w * h).map(|i| (mx..w.saturating_sub(mx)).contains(&(i % w)) && (my..h.saturating_sub(my)).contains(&(i / w))).collect();
                    (unknown, None)
                }
            };
            let mut fg = segment::grabcut_init(&img, w, h, &unknown, init, 2);
            segment::clean(&mut fg, w, h, 0.08);
            (Some(fg), Backend::GrabCut)
        };
        // Lab-ish working colour for the region segmentation.
        let lab: Vec<[f32; 3]> = rgb8.iter().map(|c| super::color::srgb8_to_lab(*c)).collect();
        let (mut lbl, n) = fh_segment(&lab, w, h, 220.0, ((w * h) / 2000).max(20) as u32);
        let n = merge_regions(&lab, &mut lbl, n, w, h, 9.0, 48);
        let total = (w * h) as f32;
        let mut labels = vec![0u16; w * h];
        let mut cands: Vec<Vec<u8>> = Vec::new();
        let mut rejected_subject = false;
        // Subject components.
        if let Some(fg) = &subj {
            let (comp, nc) = components(fg, w, h);
            for c in 1..=nc {
                let m: Vec<u8> = comp.iter().map(|&v| if v == c { 255 } else { 0 }).collect();
                let area = m.iter().filter(|&&v| v > 0).count() as f32 / total;
                // A broad saliency component can include textured ground.
                // Reject frame-dominated backdrops before they suppress the
                // smaller region candidates inside them (e.g. a ball).
                if area > 0.0015 {
                    if frame_dominated(&m, w, h) {
                        rejected_subject = true;
                    } else {
                        cands.push(m);
                    }
                }
            }
        }
        // Merged regions that look like objects.
        for r in 0..n as u32 {
            let m: Vec<u8> = lbl.iter().map(|&v| if v == r { 255 } else { 0 }).collect();
            let area = m.iter().filter(|&&v| v > 0).count() as f32 / total;
            if !(0.0015..=0.40).contains(&area) {
                continue;
            }
            // Backdrops: mostly on the frame edge, or spanning the frame.
            let (mut bx0, mut bx1, mut by0, mut by1) = (w, 0, h, 0);
            for (i, &v) in m.iter().enumerate() {
                if v > 0 {
                    bx0 = bx0.min(i % w);
                    bx1 = bx1.max(i % w + 1);
                    by0 = by0.min(i / w);
                    by1 = by1.max(i / w + 1);
                }
            }
            if frame_dominated(&m, w, h) || (bx1 - bx0) * 10 > w * 9 || (by1 - by0) * 10 > h * 9 {
                continue;
            }
            // Skip regions mostly inside an existing candidate.
            let dup = cands.iter().any(|c| {
                let inter = c.iter().zip(&m).filter(|(a, b)| **a > 0 && **b > 0).count();
                inter as f32 > 0.6 * (area * total)
            });
            if !dup {
                // A rejected global prior leaves colour regions as the only
                // candidates. A local cut recovers shaded parts of each
                // object without revisiting the ambiguous whole-image cut.
                cands.push(if rejected_subject { refine_region(m, &rgb8, w, h, (bx0, by0, bx1, by1)) } else { m });
            }
        }
        // Smaller candidates win where they overlap (hover the finest object).
        let mut order: Vec<usize> = (0..cands.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(cands[i].iter().filter(|&&v| v > 0).count()));
        for &i in &order {
            for (lab_, &v) in labels.iter_mut().zip(&cands[i]) {
                if v > 0 {
                    *lab_ = i as u16 + 1;
                }
            }
        }
        let s = (1u32 << l) as f32;
        let objects = cands
            .par_iter()
            .map(|m| {
                let (mut cnt, mut x0, mut y0, mut x1, mut y1) = (0, w, h, 0, 0);
                for (i, &v) in m.iter().enumerate() {
                    if v != 0 {
                        cnt += 1;
                        x0 = x0.min(i % w);
                        y0 = y0.min(i / w);
                        x1 = x1.max(i % w + 1);
                        y1 = y1.max(i / w + 1);
                    }
                }
                let outline = Selection { shape: super::selection::Shape::Mask(std::sync::Arc::new(to_mask(m, w, h))), inverted: false }
                    .outlines()
                    .into_iter()
                    .map(|pl| pl.into_iter().map(|q| [q[0] * s, q[1] * s]).collect())
                    .collect();
                FoundObject { bounds: (x0, y0, x1, y1), outline, coverage: cnt as f32 / total }
            })
            .collect();
        ObjectFinder { level: l, w, h, labels, objects, backend, build_ms: t0.elapsed().as_secs_f32() * 1000.0 }
    }

    /// The object under level-0 point `(x, y)`.
    pub fn at(&self, x: f32, y: f32) -> Option<usize> {
        let s = (1u32 << self.level) as f32;
        let (cx, cy) = ((x / s) as i64, (y / s) as i64);
        if cx < 0 || cy < 0 || cx >= self.w as i64 || cy >= self.h as i64 {
            return None;
        }
        let v = self.labels[cy as usize * self.w + cx as usize];
        (v > 0).then(|| v as usize - 1)
    }

    /// A BGRA tint image of object `id` at the finder's resolution (for the
    /// hover highlight, stretched over the document on screen).
    pub fn tint_bgra(&self, id: usize, rgb: [u8; 3], alpha: u8) -> Vec<u8> {
        let mut out = vec![0u8; self.w * self.h * 4];
        for (o, &v) in out.chunks_exact_mut(4).zip(&self.labels) {
            if v as usize == id + 1 {
                o.copy_from_slice(&[rgb[2], rgb[1], rgb[0], alpha]);
            }
        }
        out
    }

    /// The level-0 selection mask of object `id`, guided-filter upsampled.
    pub fn mask(&self, id: usize, src: &Source, doc_w: u32, doc_h: u32) -> Option<MaskGrid> {
        let (x0, y0, x1, y1) = self.objects.get(id)?.bounds;
        // Include the bilinear footprint and both guided-filter box passes.
        // Beyond this halo the original full-image mask is uniformly empty.
        let scale = 1usize << self.level;
        let radius = (scale * 2).clamp(2, 16);
        let margin = (2 * radius).div_ceil(scale) + 2;
        let (x0, y0, x1, y1) = (x0.saturating_sub(margin), y0.saturating_sub(margin), (x1 + margin).min(self.w), (y1 + margin).min(self.h));
        let m = (y0..y1).flat_map(|y| self.labels[y * self.w + x0..y * self.w + x1].iter().map(|&v| if v as usize == id + 1 { 255 } else { 0 })).collect();
        let c = Coarse { level: self.level, x0: x0 as i32, y0: y0 as i32, w: x1 - x0, h: y1 - y0, m };
        let g = segment::upsample(&c, doc_w, doc_h, Some(src));
        (!g.tiles.iter().all(|t| matches!(t, MaskTile::Empty))).then_some(g)
    }
}

fn refine_region(mut mask: Vec<u8>, rgb: &[[u8; 3]], w: usize, h: usize, bounds: (usize, usize, usize, usize)) -> Vec<u8> {
    let (x0, y0, x1, y1) = bounds;
    let margin = ((x1 - x0).max(y1 - y0) / 5).max(2);
    let (x0, y0, x1, y1) = (x0.saturating_sub(margin), y0.saturating_sub(margin), (x1 + margin).min(w), (y1 + margin).min(h));
    let (cw, ch) = (x1 - x0, y1 - y0);
    let index = |i: usize| (y0 + i / cw) * w + x0 + i % cw;
    let img: Vec<_> = (0..cw * ch).map(|i| rgb[index(i)].map(f32::from)).collect();
    let init: Vec<_> = (0..cw * ch).map(|i| mask[index(i)] != 0).collect();
    let unknown: Vec<_> = (0..cw * ch).map(|i| init[i] || (i % cw > 0 && i % cw + 1 < cw && i / cw > 0 && i / cw + 1 < ch)).collect();
    let mut fg = segment::grabcut_init(&img, cw, ch, &unknown, Some(init), 2);
    segment::clean(&mut fg, cw, ch, 0.08);
    if fg.iter().any(|&v| v) {
        mask.fill(0);
        for (i, selected) in fg.into_iter().enumerate() {
            mask[index(i)] = if selected { 255 } else { 0 };
        }
    }
    mask
}

/// Backdrops often run along the image frame for most of their perimeter.
/// Small frame contacts, such as a portrait's shoulders, remain eligible.
fn frame_dominated(mask: &[u8], w: usize, h: usize) -> bool {
    let (mut border, mut perimeter) = (0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] == 0 {
                continue;
            }
            let edge = x == 0 || y == 0 || x + 1 == w || y + 1 == h;
            if edge || mask[y * w + x - 1] == 0 || mask[y * w + x + 1] == 0 || mask[(y - 1) * w + x] == 0 || mask[(y + 1) * w + x] == 0 {
                perimeter += 1;
                border += usize::from(edge);
            }
        }
    }
    border as f32 > 0.3 * perimeter as f32
}

/// 4-connected components of a boolean mask (labels 1..=n, 0 = off).
fn components(fg: &[bool], w: usize, h: usize) -> (Vec<u32>, u32) {
    let mut lab = vec![0u32; w * h];
    let mut n = 0;
    let mut stack = Vec::new();
    for s in 0..w * h {
        if !fg[s] || lab[s] != 0 {
            continue;
        }
        n += 1;
        lab[s] = n;
        stack.push(s);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            for (nx, ny) in [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)] {
                if nx < w && ny < h {
                    let j = ny * w + nx;
                    if fg[j] && lab[j] == 0 {
                        lab[j] = n;
                        stack.push(j);
                    }
                }
            }
        }
    }
    (lab, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_uncropped(finder: &ObjectFinder, id: usize, src: &Source, w: u32, h: u32, cropped: &MaskGrid) {
        let m = finder.labels.iter().map(|&v| if v as usize == id + 1 { 255 } else { 0 }).collect();
        let coarse = Coarse { level: finder.level, x0: 0, y0: 0, w: finder.w, h: finder.h, m };
        let reference = segment::upsample(&coarse, w, h, Some(src));
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                assert_eq!(cropped.at(x, y), reference.at(x, y), "crop changed coverage at {x},{y}");
            }
        }
    }

    #[test]
    fn fh_separates_flat_regions() {
        let (w, h) = (64usize, 48usize);
        let rgb: Vec<[f32; 3]> = (0..w * h).map(|i| if i % w < 30 { [10.0, 10.0, 10.0] } else { [90.0, 20.0, 20.0] }).collect();
        let (lbl, n) = fh_segment(&rgb, w, h, 50.0, 10);
        assert_eq!(n, 2);
        assert_ne!(lbl[0], lbl[w - 1]);
    }

    #[test]
    fn finder_finds_objects_on_a_backdrop() {
        // Two discs on a gradient backdrop; hovering each finds a different object.
        let (w, h) = (1024u32, 768u32);
        let g = super::super::grid::build_grid(w, h, |tx, ty| {
            let ts = super::super::grid::TS;
            let mut t = vec![0u32; ts * ts].into_boxed_slice();
            for y in 0..ts {
                for x in 0..ts {
                    let (gx, gy) = ((tx as usize * ts + x) as f32, (ty as usize * ts + y) as f32);
                    let d1 = ((gx - 300.0).powi(2) + (gy - 380.0).powi(2)).sqrt();
                    let d2 = ((gx - 720.0).powi(2) + (gy - 400.0).powi(2)).sqrt();
                    let bg = (120.0 + gy * 0.08) as u8;
                    t[y * ts + x] = if d1 < 120.0 {
                        super::super::px::pack(200, 40, 30, 255)
                    } else if d2 < 90.0 {
                        super::super::px::pack(30, 60, 210, 255)
                    } else {
                        super::super::px::pack(bg, bg, bg.saturating_add(10), 255)
                    };
                }
            }
            super::super::grid::finish_tile(t)
        });
        let mips = super::super::mips::build_mips(&g);
        let src = |l: u32, x: i32, y: i32, bw: usize, bh: usize, out: &mut [u32]| {
            let lv = if l == 0 { &g } else { &mips[(l as usize).min(mips.len()) - 1] };
            lv.read_region(x, y, bw, bh, out)
        };
        let f = ObjectFinder::build(&src, w, h, mips.len() as u32);
        let a = f.at(300.0, 380.0).expect("red disc found");
        let b = f.at(720.0, 400.0).expect("blue disc found");
        assert_ne!(a, b);
        assert_eq!(f.at(30.0, 30.0), None, "backdrop corner is not an object");
        let m = f.mask(a, &src, w, h).unwrap();
        check_uncropped(&f, a, &src, w, h, &m);
        assert_eq!(m.at(300, 380), 255);
        assert_eq!(m.at(720, 400), 0);
        // Edge accuracy: the upsampled mask follows the disc boundary.
        assert!(m.at(300 + 115, 380) > 128 && m.at(300 + 125, 380) < 128);
        let (mut intersection, mut union) = (0u32, 0u32);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let expected = (x - 300).pow(2) + (y - 380).pow(2) < 120 * 120;
                let selected = m.at(x, y) > 128;
                intersection += u32::from(expected && selected);
                union += u32::from(expected || selected);
            }
        }
        assert!(intersection as f64 / union as f64 > 0.97, "whole-mask IoU: {intersection}/{union}");
    }

    #[test]
    fn finder_keeps_subjects_that_touch_the_frame() {
        use super::super::{grid, px};
        let (w, h) = (384, 256);
        let inside = |x: i32, y: i32| (x - 192).pow(2) + (y - 125).pow(2) < 75 * 75 || ((175..210).contains(&x) && y >= 125);
        let g = grid::build_grid(w, h, |tx, ty| {
            let mut tile = vec![0; grid::TS * grid::TS].into_boxed_slice();
            for y in 0..grid::TS {
                for x in 0..grid::TS {
                    let (gx, gy) = ((tx as usize * grid::TS + x) as i32, (ty as usize * grid::TS + y) as i32);
                    let bg = 110 + (gy.clamp(0, 255) / 4) as u8;
                    tile[y * grid::TS + x] = if inside(gx, gy) { px::pack(210, 45, 35, 255) } else { px::pack(bg, bg, bg + 10, 255) };
                }
            }
            grid::finish_tile(tile)
        });
        let src = |_: u32, x, y, bw, bh, out: &mut [u32]| g.read_region(x, y, bw, bh, out);
        let finder = ObjectFinder::build(&src, w, h, 0);
        let id = finder.at(192.0, 125.0).expect("subject found");
        let mask = finder.mask(id, &src, w, h).unwrap();
        check_uncropped(&finder, id, &src, w, h, &mask);
        assert!(mask.at(192, 254) > 128, "the frame edge must not force the subject to background");
        assert_eq!(finder.at(20.0, 20.0), None);
        let (mut intersection, mut union) = (0u32, 0u32);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let (expected, selected) = (inside(x, y), mask.at(x, y) > 128);
                intersection += u32::from(expected && selected);
                union += u32::from(expected || selected);
            }
        }
        assert!(intersection as f64 / union as f64 > 0.97, "whole-mask IoU: {intersection}/{union}");
    }

    #[test]
    fn finder_handles_a_flat_image_without_a_usable_prior() {
        let src = |_: u32, _: i32, _: i32, _: usize, _: usize, out: &mut [u32]| out.fill(super::super::px::pack(80, 80, 80, 255));
        for (w, h) in [(1, 1), (128, 96)] {
            let finder = ObjectFinder::build(&src, w, h, 0);
            assert!(finder.objects.is_empty(), "flat {w}x{h} image should not invent objects");
        }
    }

    #[test]
    fn finder_separates_a_shaded_object_from_textured_ground() {
        // Use the actual regression image: averaging its textured ground
        // into mips produces a different prior than generating a small copy.
        let (w, h) = (14204u32, 10652u32);
        let grid = super::super::ops::demo_scene(w, h);
        let mips = super::super::mips::build_mips(&grid);
        let src = |level: u32, x, y, bw, bh, out: &mut [u32]| {
            let image = if level == 0 { &grid } else { &mips[level as usize - 1] };
            image.read_region(x, y, bw, bh, out);
        };
        let finder = ObjectFinder::build(&src, w, h, mips.len() as u32);
        let (cx, cy, radius) = (0.30 * w as f32, 0.62 * h as f32, 0.14 * h as f32);
        let id = finder.at(cx, cy).expect("shaded ball found");
        let mask = finder.mask(id, &src, w, h).unwrap();
        let (mut intersection, mut union) = (0u32, 0u32);
        for y in (0..h as i32).step_by(16) {
            for x in (0..w as i32).step_by(16) {
                let expected = (x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2) < radius * radius;
                let selected = mask.at(x, y) > 128;
                intersection += u32::from(expected && selected);
                union += u32::from(expected || selected);
            }
        }
        assert!(intersection as f64 / union as f64 > 0.90, "whole-mask IoU: {intersection}/{union}");
    }
}
