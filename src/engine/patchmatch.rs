//! PatchMatch (Barnes et al. 2009) and multi-scale image completion
//! (Wexler et al. 2007) for Spot Healing, Patch (content-aware) and
//! Content-Aware Move.
//!
//! Only the region the caller passes (the hole plus a margin) is processed,
//! never the whole image. The region is downsampled into a pyramid; the hole
//! is filled at the coarsest level first and each finer level starts from the
//! upsampled nearest-neighbour field (NNF). At every level an EM loop alternates
//! a PatchMatch pass (propagation from neighbours plus a randomized search
//! with exponentially shrinking radius) with a vote that rebuilds the hole
//! pixels from the patches that cover them.
//!
//! The PatchMatch pass is parallelized by scanline parity: all even rows run
//! in parallel (each row propagates left→right and from the row above, which
//! the previous phase finished), then all odd rows; the next iteration runs
//! right→left and bottom-up. Patch distances are SSD over 7×7 RGBA bytes with
//! early termination.
//!
//! **Adaptation** (Content-Aware Fill): every NNF entry also carries a
//! transform index into a table of rotations (up to ±180° in 15° steps),
//! scales (0.8–1.25) and a horizontal mirror; the table holds each
//! transform's 49 rounded source offsets, so a transformed distance is the
//! same 49 lookups as an identity one. Propagation composes the neighbour's
//! transform with the step, and random search also tries random
//! transforms. **Colour adaptation** matches patches up to a per-channel
//! brightness offset (clamped to ±10 / 25 / 50 levels): both patch means
//! come from box-filtered mean maps, and the vote adds the same offset.

use rayon::prelude::*;

use super::grid::SyncPtr;
use super::px;

pub const R: i32 = 3; // patch radius (7×7)

/// One pyramid level of the region.
pub(crate) struct Lvl {
    pub(crate) w: i32,
    pub(crate) h: i32,
    /// RGBA bytes (straight color, alpha 255 inside the image).
    pub(crate) img: Vec<[u8; 4]>,
    pub(crate) hole: Vec<u8>,
    /// Source centers whose whole patch is known (identity transform).
    pub(crate) valid: Vec<u8>,
    pub(crate) valid_list: Vec<u32>,
    /// Same for transformed patches, whose samples reach `xf.reach`.
    pub(crate) valid_wide: Vec<u8>,
    pub(crate) xf: std::sync::Arc<Xfs>,
    /// 7×7 box means (colour adaptation only).
    pub(crate) mean: Vec<[i16; 3]>,
    /// Per-source-centre SSD penalty (the focus prior); empty when off.
    pub(crate) prox: Vec<u32>,
}

/// Patch offset index of `(dx, dy)`.
#[inline]
fn oi(dx: i32, dy: i32) -> usize {
    ((dy + R) * (2 * R + 1) + dx + R) as usize
}

const NP: usize = ((2 * R + 1) * (2 * R + 1)) as usize;

/// The transform table: per transform, the rounded source offset of every
/// patch offset. Entry 0 is the identity.
pub struct Xfs {
    off: Vec<[(i8, i8); NP]>,
    /// Exact (unrounded) offsets, for sub-pixel source centres.
    pub(crate) offf: Vec<[(f32, f32); NP]>,
    /// Each transform as a matrix `(u, v) = (a dx + b dy, c dx + d dy)`
    /// (the GPU's long propagation jumps, beyond the patch table).
    pub(crate) mat: Vec<[f32; 4]>,
    /// Chebyshev reach of any non-identity sample.
    pub(crate) reach: i32,
    /// Max |colour offset| a match may absorb (0 = off).
    pub(crate) bias: i32,
}

impl Xfs {
    fn new(a: &Adapt) -> Xfs {
        let max_deg = match a.rotation {
            RotationAdapt::None => 0,
            RotationAdapt::Low => 15,
            RotationAdapt::Medium => 45,
            RotationAdapt::High => 90,
            RotationAdapt::Full => 180,
        };
        let mut angles: Vec<i32> = vec![0];
        let mut d = 15;
        while d <= max_deg {
            angles.push(d);
            if d < 180 {
                angles.push(-d);
            }
            d += 15;
        }
        let scales: &[f32] = if a.scale { &[1.0, 0.8, 0.9, 1.12, 1.25] } else { &[1.0] };
        let mirrors: &[bool] = if a.mirror { &[false, true] } else { &[false] };
        let mut off = Vec::new();
        let mut offf = Vec::new();
        let mut mat = Vec::new();
        let mut reach = R;
        for &m in mirrors {
            for &sc in scales {
                for &deg in &angles {
                    let (sn, cs) = (deg as f32).to_radians().sin_cos();
                    let fl = if m { -1.0 } else { 1.0 };
                    mat.push([fl * cs * sc, -sn * sc, fl * sn * sc, cs * sc]);
                    let mut t = [(0i8, 0i8); NP];
                    let mut tf = [(0f32, 0f32); NP];
                    for dy in -R..=R {
                        for dx in -R..=R {
                            let fx = if m { -dx } else { dx } as f32;
                            let (u, v) = ((fx * cs - dy as f32 * sn) * sc, (fx * sn + dy as f32 * cs) * sc);
                            tf[oi(dx, dy)] = (u, v);
                            // +1: a sub-pixel centre can round one further.
                            reach = reach.max(u.abs().ceil() as i32 + 1).max(v.abs().ceil() as i32 + 1);
                            t[oi(dx, dy)] = (u.round() as i8, v.round() as i8);
                        }
                    }
                    off.push(t);
                    offf.push(tf);
                }
            }
        }
        let bias = match a.color {
            ColorAdapt::None => 0,
            ColorAdapt::Default => 10,
            ColorAdapt::High => 25,
            ColorAdapt::VeryHigh => 50,
        };
        Xfs { off, offf, mat, reach, bias }
    }

    pub(crate) fn len(&self) -> usize {
        self.off.len()
    }
}

/// Sub-pixel source centres are stored as an integer centre plus a
/// fraction in 1/64 px.
const SUB: f32 = 64.0;

/// Splits an exact position into (integer centre, 1/64 px remainder).
#[inline]
fn split(e: f32) -> (i32, i8) {
    let r = e.round();
    (r as i32, ((e - r) * SUB).round().clamp(-32.0, 32.0) as i8)
}

impl Lvl {
    fn from(img: &[[u8; 4]], hole: &[u8], w: i32, h: i32, allowed: Option<&[u8]>, xf: std::sync::Arc<Xfs>) -> Lvl {
        let n = (w * h) as usize;
        let mut l = Lvl { w, h, img: img.to_vec(), hole: hole.to_vec(), valid: vec![0; n], valid_list: vec![], valid_wide: vec![], xf, mean: vec![], prox: vec![] };
        l.compute_valid(allowed);
        l.update_mean();
        l
    }

    /// Box dilation of the hole by `r` (horizontal then vertical running OR).
    /// Box dilation of the hole by `r` with sliding-window counts (O(N)
    /// for any radius): horizontal pass per row, vertical pass per column
    /// strip.
    fn near_hole(&self, r: i32) -> Vec<u8> {
        let (w, h) = (self.w as usize, self.h as usize);
        let r = r.max(0) as usize;
        let mut tmp = vec![0u8; w * h];
        tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let src = &self.hole[y * w..(y + 1) * w];
            let mut cnt = src[..r.min(w)].iter().filter(|&&v| v != 0).count();
            for (x, o) in row.iter_mut().enumerate() {
                if x + r < w && src[x + r] != 0 {
                    cnt += 1;
                }
                if x > r && src[x - r - 1] != 0 {
                    cnt -= 1;
                }
                *o = (cnt > 0) as u8;
            }
        });
        let mut near = vec![0u8; w * h];
        // Columns in strips of 64 so each task walks rows contiguously.
        let strips: Vec<usize> = (0..w).step_by(64).collect();
        let out = super::grid::SyncPtr(near.as_mut_ptr());
        strips.par_iter().for_each(|&x0| {
            let x1 = (x0 + 64).min(w);
            let mut cnt = vec![0usize; x1 - x0];
            for y in 0..r.min(h) {
                for x in x0..x1 {
                    cnt[x - x0] += tmp[y * w + x] as usize;
                }
            }
            for y in 0..h {
                for x in x0..x1 {
                    if y + r < h {
                        cnt[x - x0] += tmp[(y + r) * w + x] as usize;
                    }
                    if y > r {
                        cnt[x - x0] -= tmp[(y - r - 1) * w + x] as usize;
                    }
                    // SAFETY: strips are disjoint column ranges.
                    unsafe { *out.at(y * w + x) = (cnt[x - x0] > 0) as u8 };
                }
            }
        });
        near
    }

    fn compute_valid(&mut self, allowed: Option<&[u8]>) {
        let (w, h) = (self.w, self.h);
        // A center is valid when no hole pixel lies within R (box dilation).
        let mask = |near: &[u8], r: i32| -> Vec<u8> {
            let mut v = vec![0u8; (w * h) as usize];
            v.par_chunks_mut(w as usize).enumerate().for_each(|(y, row)| {
                let y = y as i32;
                if y < r || y >= h - r {
                    return;
                }
                for x in r..w - r {
                    let i = (y * w + x) as usize;
                    row[x as usize] = (near[i] == 0 && allowed.is_none_or(|a| a[i] != 0)) as u8;
                }
            });
            v
        };
        let near = self.near_hole(R);
        self.valid = mask(&near, R);
        self.valid_list = self.valid.par_iter().enumerate().filter(|(_, v)| **v != 0).map(|(i, _)| i as u32).collect();
        if self.xf.len() > 1 {
            let rw = self.xf.reach;
            let near = if rw == R { near } else { self.near_hole(rw) };
            self.valid_wide = mask(&near, rw);
        }
    }

    /// Focus prior: penalizes source centres whose local detail (7×7 mean
    /// gradient magnitude of the luma) is below the hole's immediate ring,
    /// so a fill matches the sharpness of its surroundings instead of
    /// copying defocused areas (which match a smooth coarse estimate best).
    /// The penalty is `strength² · (1 − g/g_ring)² · NP · 3 · 100`: at
    /// strength 1 a source with no detail costs as much as a 10-level error
    /// per channel.
    fn add_focus(&mut self, strength: f32) {
        let (w, h) = (self.w as usize, self.h as usize);
        let lum: Vec<f32> = self.img.iter().map(|c| 0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32).collect();
        let mut g = vec![0f32; w * h];
        g.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let (ya, yb) = (y.saturating_sub(1), (y + 1).min(h - 1));
            for (x, o) in row.iter_mut().enumerate() {
                let (xa, xb) = (x.saturating_sub(1), (x + 1).min(w - 1));
                *o = (lum[y * w + xb] - lum[y * w + xa]).abs() + (lum[yb * w + x] - lum[ya * w + x]).abs();
            }
        });
        // 7×7 box mean (separable).
        let r = R as usize;
        let mut t = vec![0f32; w * h];
        t.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let (a, b) = (x.saturating_sub(r), (x + r + 1).min(w));
                *o = g[y * w + a..y * w + b].iter().sum::<f32>() / (b - a) as f32;
            }
        });
        let mut m = vec![0f32; w * h];
        m.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let (a, b) = (y.saturating_sub(r), (y + r + 1).min(h));
            for (x, o) in row.iter_mut().enumerate() {
                *o = (a..b).map(|yy| t[yy * w + x]).sum::<f32>() / (b - a) as f32;
            }
        });
        // Ring: known pixels 2R..6R from the hole.
        let (near2, near6) = (self.near_hole(2 * R), self.near_hole(6 * R));
        let mut ring: Vec<f32> = (0..w * h).filter(|&i| near6[i] != 0 && near2[i] == 0).map(|i| m[i]).collect();
        if ring.is_empty() {
            return;
        }
        let k = ring.len() / 2;
        let (_, med, _) = ring.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
        let gr = med.max(1.0);
        let kk = strength * strength * (NP * 3 * 100) as f32;
        if self.prox.is_empty() {
            self.prox = vec![0; w * h];
        }
        self.prox.par_iter_mut().zip(&m).for_each(|(p, &v)| {
            let d = (1.0 - v / gr).max(0.0);
            *p += (kk * d * d) as u32;
        });
    }

    /// Recomputes the 7×7 mean map (colour adaptation needs current means).
    fn update_mean(&mut self) {
        if self.xf.bias == 0 {
            return;
        }
        let (w, h) = (self.w as usize, self.h as usize);
        let r = R as usize;
        let img = &self.img;
        // Horizontal then vertical box sums, rows in parallel.
        let mut tmp = vec![[0i32; 3]; w * h];
        tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let mut a = [0i32; 3];
                for xx in x.saturating_sub(r)..(x + r + 1).min(w) {
                    for c in 0..3 {
                        a[c] += img[y * w + xx][c] as i32;
                    }
                }
                *o = a;
            }
        });
        let mut mean = vec![[0i16; 3]; w * h];
        mean.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let (y0, y1) = (y.saturating_sub(r), (y + r + 1).min(h));
            for (x, o) in row.iter_mut().enumerate() {
                let n = ((y1 - y0) * ((x + r + 1).min(w) - x.saturating_sub(r))) as i32;
                let mut a = [0i32; 3];
                for yy in y0..y1 {
                    for c in 0..3 {
                        a[c] += tmp[yy * w + x][c];
                    }
                }
                *o = a.map(|v| (v / n) as i16);
            }
        });
        self.mean = mean;
    }

    #[inline]
    fn is_valid(&self, x: i32, y: i32, k: usize) -> bool {
        if k == 0 {
            x >= R && y >= R && x < self.w - R && y < self.h - R && self.valid[(y * self.w + x) as usize] != 0
        } else {
            let rw = self.xf.reach;
            x >= rw && y >= rw && x < self.w - rw && y < self.h - rw && self.valid_wide[(y * self.w + x) as usize] != 0
        }
    }

    /// Colour offset (target − source means, clamped) for a match.
    #[inline]
    fn bias(&self, t: usize, s: usize) -> [i32; 3] {
        let b = self.xf.bias;
        if b == 0 {
            return [0; 3];
        }
        let (mt, ms) = (self.mean[t], self.mean[s]);
        [0, 1, 2].map(|c| (mt[c] as i32 - ms[c] as i32).clamp(-b, b))
    }

    /// SSD between the patch centered at `t` and source patch `s` under
    /// transform `k` (both inside their reach of the region border), giving
    /// up past `best`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn dist(&self, tx: i32, ty: i32, sx: i32, sy: i32, sub: [i8; 2], k: usize, best: u32) -> u32 {
        let w = self.w;
        let mut d = if self.prox.is_empty() { 0 } else { self.prox[(sy * w + sx) as usize] };
        let n = (2 * R + 1) as usize;
        let bias = self.bias((ty * w + tx) as usize, (sy * w + sx) as usize);
        if k == 0 && bias == [0; 3] {
            for dy in -R..=R {
                let (t0, s0) = (((ty + dy) * w + tx - R) as usize, ((sy + dy) * w + sx - R) as usize);
                let tr = &self.img[t0..t0 + n];
                let sr = &self.img[s0..s0 + n];
                let mut acc = 0u32;
                for (a, b) in tr.iter().zip(sr) {
                    for c in 0..3 {
                        let e = a[c] as i32 - b[c] as i32;
                        acc += (e * e) as u32;
                    }
                }
                d += acc;
                if d >= best {
                    return d;
                }
            }
            return d;
        }
        let off = &self.xf.offf[k];
        let (fx, fy) = (sub[0] as f32 / SUB, sub[1] as f32 / SUB);
        for dy in -R..=R {
            let t0 = ((ty + dy) * w + tx - R) as usize;
            let mut acc = 0u32;
            for dx in -R..=R {
                let a = self.img[t0 + (dx + R) as usize];
                let (u, v) = off[oi(dx, dy)];
                let (u, v) = ((u + fx).round() as i32, (v + fy).round() as i32);
                let b = self.img[((sy + v) * w + sx + u) as usize];
                for c in 0..3 {
                    let e = a[c] as i32 - b[c] as i32 - bias[c];
                    acc += (e * e) as u32;
                }
            }
            d += acc;
            if d >= best {
                return d;
            }
        }
        d
    }
}

#[inline]
fn xorshift(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

/// Nearest-neighbour field for the target centers (hole dilated by R).
pub(crate) struct Nnf {
    /// Source center per target pixel (index), u32::MAX = not a target.
    pub(crate) src: Vec<u32>,
    /// Transform index per target pixel.
    pub(crate) xf: Vec<u16>,
    /// Sub-pixel remainder of the source centre (1/64 px).
    pub(crate) sub: Vec<[i8; 2]>,
    pub(crate) dist: Vec<u32>,
    pub(crate) targets: Vec<u8>,
}

/// Target centers: every pixel within R of the hole, kept R away from the
/// region border (callers give the hole a margin).
pub(crate) fn targets_of(l: &Lvl) -> Vec<u8> {
    let (w, h) = (l.w, l.h);
    let mut t = l.near_hole(R);
    t.par_chunks_mut(w as usize).enumerate().for_each(|(y, row)| {
        let y = y as i32;
        for (x, v) in row.iter_mut().enumerate() {
            let x = x as i32;
            if !(x >= R && y >= R && x < w - R && y < h - R) {
                *v = 0;
            }
        }
    });
    t
}

fn random_nnf(l: &Lvl, seed: u32) -> Nnf {
    let n = (l.w * l.h) as usize;
    let targets = targets_of(l);
    let mut src = vec![u32::MAX; n];
    let xf = vec![0u16; n];
    let sub = vec![[0i8; 2]; n];
    let mut dist = vec![u32::MAX; n];
    if l.valid_list.is_empty() {
        return Nnf { src, xf, sub, dist, targets };
    }
    let mut rng = seed | 1;
    for i in 0..n {
        if targets[i] != 0 {
            let s = l.valid_list[(xorshift(&mut rng) as usize) % l.valid_list.len()];
            src[i] = s;
            let (tx, ty) = ((i as i32) % l.w, (i as i32) / l.w);
            dist[i] = l.dist(tx, ty, s as i32 % l.w, s as i32 / l.w, [0, 0], 0, u32::MAX);
        }
    }
    Nnf { src, xf, sub, dist, targets }
}

/// The best match found so far for one target.
struct Best {
    s: u32,
    k: usize,
    sub: [i8; 2],
    d: u32,
}

impl Best {
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn try_c(&mut self, l: &Lvl, x: i32, y: i32, sx: i32, sy: i32, sub: [i8; 2], k: usize) {
        if !l.is_valid(sx, sy, k) {
            return;
        }
        let si = (sy * l.w + sx) as u32;
        if si == self.s && k == self.k && sub == self.sub {
            return;
        }
        let d = l.dist(x, y, sx, sy, sub, k, self.d);
        if d < self.d {
            *self = Best { s: si, k, sub, d };
        }
    }
}

/// One PatchMatch iteration (propagation + random search), parallel by
/// scanline parity. `reverse` flips the scan direction.
fn pm_pass(l: &Lvl, f: &mut Nnf, reverse: bool, seed: u32) {
    let (w, h) = (l.w, l.h);
    let maxr = w.max(h);
    let nxf = l.xf.len();
    for parity in 0..2 {
        let src_ptr = SyncPtr(f.src.as_mut_ptr());
        let xf_ptr = SyncPtr(f.xf.as_mut_ptr());
        let sub_ptr = SyncPtr(f.sub.as_mut_ptr());
        let dist_ptr = SyncPtr(f.dist.as_mut_ptr());
        let targets = &f.targets;
        let rows: Vec<i32> = (0..h).filter(|y| y % 2 == parity).collect();
        rows.par_iter().for_each(|&y| {
            let y = if reverse { h - 1 - y } else { y };
            let mut rng = seed ^ (y as u32).wrapping_mul(0x9E37_79B9) ^ 0x1234_5678;
            let dir: i32 = if reverse { -1 } else { 1 };
            let xs: Box<dyn Iterator<Item = i32>> = if reverse { Box::new((0..w).rev()) } else { Box::new(0..w) };
            for x in xs {
                let i = (y * w + x) as usize;
                if targets[i] == 0 {
                    continue;
                }
                // SAFETY: rows of one parity are disjoint; neighbours read are
                // either in this row (same thread) or the other parity (idle).
                let (bs, bk, bsub, bd) = unsafe { (src_ptr.at(i), xf_ptr.at(i), sub_ptr.at(i), dist_ptr.at(i)) };
                let mut b = Best { s: *bs, k: *bk as usize, sub: *bsub, d: *bd };
                let try_c = |b: &mut Best, sx: i32, sy: i32, sub: [i8; 2], k: usize| b.try_c(l, x, y, sx, sy, sub, k);
                // Propagation from the previous pixel in scan order and the
                // previous row: the neighbour's source stepped by its own
                // transform, so the patches stay aligned.
                for (nx, ny) in [(x - dir, y), (x, y - dir)] {
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let j = (ny * w + nx) as usize;
                    let (s, k, sb) = unsafe { (*src_ptr.at(j), *xf_ptr.at(j) as usize, *sub_ptr.at(j)) };
                    if s == u32::MAX {
                        continue;
                    }
                    let (ddx, ddy) = (x - nx, y - ny);
                    if k == 0 {
                        try_c(&mut b, s as i32 % w + ddx, s as i32 / w + ddy, [0, 0], 0);
                    } else {
                        let o = l.xf.offf[k][oi(ddx, ddy)];
                        let (sx, fx) = split(s as i32 as f32 % w as f32 + sb[0] as f32 / SUB + o.0);
                        let (sy, fy) = split((s as i32 / w) as f32 + sb[1] as f32 / SUB + o.1);
                        try_c(&mut b, sx, sy, [fx, fy], k);
                    }
                }
                // Random search around the current best (and, with
                // adaptation on, under random transforms).
                if b.s != u32::MAX {
                    let (bx, by, bsb) = (b.s as i32 % w, b.s as i32 / w, b.sub);
                    let mut r = maxr;
                    while r >= 1 {
                        let rx = bx + (xorshift(&mut rng) % (2 * r as u32 + 1)) as i32 - r;
                        let ry = by + (xorshift(&mut rng) % (2 * r as u32 + 1)) as i32 - r;
                        let k = if nxf > 1 && xorshift(&mut rng) & 1 == 0 { xorshift(&mut rng) as usize % nxf } else { b.k };
                        try_c(&mut b, rx, ry, [0, 0], k);
                        r /= 2;
                    }
                    if nxf > 1 {
                        // Re-test the best position under another transform.
                        let k = xorshift(&mut rng) as usize % nxf;
                        try_c(&mut b, bx, by, bsb, k);
                    }
                }
                *bs = b.s;
                *bk = b.k as u16;
                *bsub = b.sub;
                *bd = b.d;
            }
        });
    }
}

/// Rebuilds hole pixels from the NNF: weighted average of the patches that
/// cover each pixel (or the matched center pixel when `center`).
fn vote(lm: &mut Lvl, f: &Nnf, center: bool) {
    let l = &*lm;
    let (w, h) = (l.w, l.h);
    let img = &l.img;
    let hole = &l.hole;
    // Distance-based weights: exp(-d / (2σ²)), σ² from the 75th percentile.
    let mut ds: Vec<u32> = f.dist.iter().zip(&f.targets).filter(|(d, t)| **t != 0 && **d != u32::MAX).map(|(d, _)| *d).collect();
    let sigma2 = if ds.is_empty() {
        1.0
    } else {
        let k = ds.len() * 3 / 4;
        let (_, v, _) = ds.select_nth_unstable(k);
        (*v as f32).max(1.0)
    };
    // One weight per target (not per covered pixel × patch).
    let wts: Vec<f32> = f
        .dist
        .par_iter()
        .zip(&f.targets)
        // Unscored matches (right after upsampling) vote uniformly.
        .map(|(&d, &t)| if t == 0 { 0.0 } else { (-(d as f32) / (2.0 * sigma2)).exp().max(1e-6) })
        .collect();
    let out: Vec<(usize, [u8; 4])> = (0..h)
        .into_par_iter()
        .flat_map_iter(|y| {
            let mut res = Vec::new();
            for x in 0..w {
                let i = (y * w + x) as usize;
                if hole[i] == 0 {
                    continue;
                }
                if center {
                    let s = f.src[i];
                    if s != u32::MAX {
                        let b = l.bias(i, s as usize);
                        let c = img[s as usize];
                        res.push((i, [0, 1, 2, 3].map(|k| if k == 3 { 255 } else { (c[k] as i32 + b[k]).clamp(0, 255) as u8 })));
                    }
                    continue;
                }
                let mut acc = [0f32; 3];
                let mut wsum = 0f32;
                for dy in -R..=R {
                    for dx in -R..=R {
                        let (tx, ty) = (x + dx, y + dy);
                        if tx < 0 || ty < 0 || tx >= w || ty >= h {
                            continue;
                        }
                        let t = (ty * w + tx) as usize;
                        let s = f.src[t];
                        if s == u32::MAX {
                            continue;
                        }
                        let k = f.xf[t] as usize;
                        let (u, v) = if k == 0 {
                            (-dx, -dy)
                        } else {
                            let o = l.xf.offf[k][oi(-dx, -dy)];
                            let sb = f.sub[t];
                            ((o.0 + sb[0] as f32 / SUB).round() as i32, (o.1 + sb[1] as f32 / SUB).round() as i32)
                        };
                        let (sx, sy) = (s as i32 % w + u, s as i32 / w + v);
                        let c = img[(sy * w + sx) as usize];
                        let b = l.bias(t, s as usize);
                        let wt = wts[t];
                        for k in 0..3 {
                            acc[k] += (c[k] as f32 + b[k] as f32) * wt;
                        }
                        wsum += wt;
                    }
                }
                if wsum > 0.0 {
                    res.push((i, [(acc[0] / wsum + 0.5) as u8, (acc[1] / wsum + 0.5) as u8, (acc[2] / wsum + 0.5) as u8, 255]));
                }
            }
            res
        })
        .collect();
    for (i, c) in out {
        lm.img[i] = c;
    }
    lm.update_mean();
}

fn downsample(l: &Lvl) -> (Vec<[u8; 4]>, Vec<u8>, i32, i32) {
    let (cw, ch) = ((l.w + 1) / 2, (l.h + 1) / 2);
    let mut img = vec![[0u8; 4]; (cw * ch) as usize];
    let mut hole = vec![0u8; (cw * ch) as usize];
    for y in 0..ch {
        for x in 0..cw {
            let mut acc = [0u32; 3];
            let mut n = 0;
            let mut hh = 0;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (fx, fy) = (2 * x + dx, 2 * y + dy);
                if fx < l.w && fy < l.h {
                    let i = (fy * l.w + fx) as usize;
                    hh |= l.hole[i];
                    if l.hole[i] == 0 {
                        for c in 0..3 {
                            acc[c] += l.img[i][c] as u32;
                        }
                        n += 1;
                    }
                }
            }
            let i = (y * cw + x) as usize;
            hole[i] = hh;
            if n > 0 {
                img[i] = [(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8, 255];
            }
        }
    }
    (img, hole, cw, ch)
}

/// Fills hole pixels by repeated averaging of known neighbours (the coarse
/// initial guess, like a fast diffusion inpainting).
fn diffuse(l: &mut Lvl) {
    let (w, h) = (l.w, l.h);
    let mut known: Vec<bool> = l.hole.iter().map(|&v| v == 0).collect();
    let mut changed = true;
    while changed {
        changed = false;
        let mut next = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                if known[i] {
                    continue;
                }
                let mut acc = [0u32; 3];
                let mut n = 0;
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx >= 0 && yy >= 0 && xx < w && yy < h && known[(yy * w + xx) as usize] {
                        let c = l.img[(yy * w + xx) as usize];
                        for k in 0..3 {
                            acc[k] += c[k] as u32;
                        }
                        n += 1;
                    }
                }
                if n > 0 {
                    next.push((i, [(acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8, 255]));
                }
            }
        }
        for (i, c) in next {
            l.img[i] = c;
            known[i] = true;
            changed = true;
        }
    }
}

/// Rotation Adaptation (Content-Aware Fill): the largest source rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RotationAdapt {
    #[default]
    None,
    /// ±15°
    Low,
    /// ±45°
    Medium,
    /// ±90°
    High,
    /// ±180°
    Full,
}

impl RotationAdapt {
    pub const ALL: [RotationAdapt; 5] = [RotationAdapt::None, RotationAdapt::Low, RotationAdapt::Medium, RotationAdapt::High, RotationAdapt::Full];
    pub fn label(self) -> &'static str {
        match self {
            RotationAdapt::None => "None",
            RotationAdapt::Low => "Low",
            RotationAdapt::Medium => "Medium",
            RotationAdapt::High => "High",
            RotationAdapt::Full => "Full",
        }
    }
}

/// Color Adaptation (Content-Aware Fill): how far a source patch's
/// brightness may be shifted to match.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorAdapt {
    #[default]
    None,
    Default,
    High,
    VeryHigh,
}

impl ColorAdapt {
    pub const ALL: [ColorAdapt; 4] = [ColorAdapt::None, ColorAdapt::Default, ColorAdapt::High, ColorAdapt::VeryHigh];
    pub fn label(self) -> &'static str {
        match self {
            ColorAdapt::None => "None",
            ColorAdapt::Default => "Default",
            ColorAdapt::High => "High",
            ColorAdapt::VeryHigh => "Very High",
        }
    }
}

/// Source-patch adaptation for completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Adapt {
    pub rotation: RotationAdapt,
    pub scale: bool,
    pub mirror: bool,
    pub color: ColorAdapt,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// EM iterations at the coarsest and at the finest level.
    pub iters_coarse: usize,
    pub iters_fine: usize,
    /// Stop coarsening when the hole is this small (pixels on its longest side).
    pub min_hole: i32,
    pub seed: u32,
    pub adapt: Adapt,
    /// Final pixels from the best match instead of the patch average.
    pub sharp: bool,
    /// Strength of the focus prior (0 = off), see `Lvl::add_focus`.
    pub focus: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params { iters_coarse: 8, iters_fine: 2, min_hole: 6, seed: 7, adapt: Adapt::default(), sharp: true, focus: 1.0 }
    }
}

/// Completes the `hole` (1 = missing) of a `w x h` RGBA region. `allowed`
/// optionally restricts where source patches may come from. Returns the
/// region with hole pixels replaced.
pub fn complete(img: &[u32], hole: &[u8], w: usize, h: usize, allowed: Option<&[u8]>, p: &Params) -> Vec<u32> {
    // Small regions run on a few performance cores: their many short
    // parallel steps cost more in scheduling than they gain on all cores.
    let (out, st) = if w * h < 2_000_000 {
        super::poisson::dab_pool().install(|| complete_stats(img, hole, w, h, allowed, p))
    } else {
        complete_stats(img, hole, w, h, allowed, p)
    };
    if std::env::var("PF_TIMING").is_ok() {
        eprintln!("    pm transforms {:?}, mean patch SSD {:.0}", st.xf_hist, st.mean_dist);
    }
    out
}

/// Final-level match statistics: transform histogram and mean patch SSD.
pub struct MatchStats {
    pub xf_hist: Vec<usize>,
    pub mean_dist: f64,
}

/// Who runs the EM iterations of the fine levels.
#[derive(Clone)]
pub enum Engine {
    /// The GPU when `gpu::retouch_gate` allows it (feature mode, cost model).
    Auto,
    Cpu,
    /// The GPU unconditionally (tests, benchmarks).
    Gpu(std::sync::Arc<super::gpu::Gpu>),
}

/// [`complete`] plus the final NNF's statistics.
pub fn complete_stats(img: &[u32], hole: &[u8], w: usize, h: usize, allowed: Option<&[u8]>, p: &Params) -> (Vec<u32>, MatchStats) {
    let (out, st, _) = complete_with(img, hole, w, h, allowed, p, Engine::Auto);
    (out, st)
}

/// Pyramid levels at least this large run on the GPU (coarser ones are
/// dominated by per-dispatch costs and stay on the CPU).
pub const GPU_MIN_LEVEL_PX: i64 = 1 << 17;

/// [`complete_stats`] on a chosen engine, with the path it took. The CPU
/// always builds the pyramid and solves the coarse levels; from the
/// coarsest level of at least [`GPU_MIN_LEVEL_PX`] pixels on, the GPU
/// runs the same EM steps (see `gpu::patchmatch`). If the GPU fails, the
/// CPU continues from that level with its own state.
#[allow(clippy::too_many_arguments)]
pub fn complete_with(img: &[u32], hole: &[u8], w: usize, h: usize, allowed: Option<&[u8]>, p: &Params, engine: Engine) -> (Vec<u32>, MatchStats, super::gpu::PathTaken) {
    use super::gpu::{self, PathTaken};
    let t_all = std::time::Instant::now();
    let op = gpu::cost::OP_PATCHMATCH;
    let units = (w * h) as u64;
    let (wi, hi) = (w as i32, h as i32);
    let tm = std::time::Instant::now();
    let timing = std::env::var("PF_TIMING").is_ok();
    let straight: Vec<[u8; 4]> = img.iter().map(|&c| px::unpremultiply(c)).collect();
    let xf = std::sync::Arc::new(Xfs::new(&p.adapt));
    let mut levels = vec![Lvl::from(&straight, hole, wi, hi, allowed, xf.clone())];
    if timing {
        eprintln!("    pm level 0 setup {:.0} ms", tm.elapsed().as_secs_f64() * 1e3);
    }
    let hole_extent = |l: &Lvl| -> i32 {
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for y in 0..l.h {
            for x in 0..l.w {
                if l.hole[(y * l.w + x) as usize] != 0 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        if x1 < x0 { 0 } else { (x1 - x0 + 1).max(y1 - y0 + 1) }
    };
    // Allowed-source masks downsample with "all children allowed".
    let mut allowed_lv: Vec<Option<Vec<u8>>> = vec![allowed.map(|a| a.to_vec())];
    loop {
        let l = levels.last().unwrap();
        if hole_extent(l) <= p.min_hole || l.w < 4 * R + 8 || l.h < 4 * R + 8 {
            break;
        }
        let (img2, hole2, cw, ch) = downsample(l);
        let al = allowed_lv.last().unwrap().as_ref().map(|a| {
            let mut v = vec![0u8; (cw * ch) as usize];
            for y in 0..ch {
                for x in 0..cw {
                    let mut all = 1u8;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (fx, fy) = (2 * x + dx, 2 * y + dy);
                        if fx < l.w && fy < l.h {
                            all &= a[(fy * l.w + fx) as usize];
                        }
                    }
                    v[(y * cw + x) as usize] = all;
                }
            }
            v
        });
        let nl = Lvl::from(&img2, &hole2, cw, ch, al.as_deref(), xf.clone());
        if nl.valid_list.len() < 4 {
            break;
        }
        allowed_lv.push(al);
        levels.push(nl);
    }
    if p.focus > 0.0 {
        let tf = std::time::Instant::now();
        levels.par_iter_mut().for_each(|l| l.add_focus(p.focus));
        if timing {
            eprintln!("    pm focus prior {:.1} ms", tf.elapsed().as_secs_f64() * 1e3);
        }
    }
    let top = levels.len() - 1;
    if timing {
        eprintln!("    pm pyramid ({} levels) {:.0} ms", levels.len(), tm.elapsed().as_secs_f64() * 1e3);
    }
    // The GPU takes over from the coarsest large level below the top.
    let gpu_from = (0..top).rev().find(|&li| levels[li].w as i64 * levels[li].h as i64 >= GPU_MIN_LEVEL_PX);
    let mut why = String::from("no large pyramid level");
    let gpu_ctx = match (&engine, gpu_from) {
        (_, None) | (Engine::Cpu, _) => {
            if matches!(engine, Engine::Cpu) {
                why = "CPU requested".into();
            }
            None
        }
        (Engine::Gpu(g), _) => Some(g.clone()),
        (Engine::Auto, _) => match gpu::retouch_gate(op, units) {
            Ok(g) => Some(g),
            Err(e) => {
                why = e;
                None
            }
        },
    };
    diffuse(&mut levels[top]);
    let mut nnf = random_nnf(&levels[top], p.seed);
    for li in (0..=top).rev() {
        if let (Some(g), Some(from)) = (&gpu_ctx, gpu_from)
            && li == from
        {
            let want_stats = timing || !matches!(engine, Engine::Auto);
            match gpu::patchmatch::run(g, &levels, &nnf, from, p, want_stats) {
                Ok((fill, st)) => {
                    let out = img
                        .iter()
                        .zip(&fill)
                        .zip(hole)
                        .map(|((&orig, &c), &hh)| if hh != 0 { c | 0xFF00_0000 } else { orig })
                        .collect();
                    let ms = t_all.elapsed().as_secs_f64() * 1e3;
                    if matches!(engine, Engine::Auto) {
                        gpu::cost::observe_op(op, units, ms, true);
                    }
                    gpu::record(op, PathTaken::Gpu, ms);
                    if timing {
                        eprintln!("    pm done on the GPU from level {from}: {ms:.0} ms");
                    }
                    let st = st.unwrap_or(MatchStats { xf_hist: vec![0; xf.len()], mean_dist: f64::NAN });
                    return (out, st, PathTaken::Gpu);
                }
                Err(e) => {
                    if e.is_fault() {
                        g.fail(e.to_string());
                    }
                    why = format!("GPU stopped at level {from}: {e}");
                }
            }
        }
        let tl = std::time::Instant::now();
        if li < top {
            // Seed the hole from the coarser result (so patch means and the
            // first distances never see the placeholder colour), then
            // upsample the NNF from the coarser level and re-vote.
            let (cw, _) = (levels[li + 1].w, levels[li + 1].h);
            {
                let (fine, coarse) = levels.split_at_mut(li + 1);
                let (f, c) = (&mut fine[li], &coarse[0]);
                for y in 0..f.h {
                    for x in 0..f.w {
                        let i = (y * f.w + x) as usize;
                        if f.hole[i] != 0 {
                            f.img[i] = c.img[((y / 2).min(c.h - 1) * c.w + (x / 2).min(c.w - 1)) as usize];
                        }
                    }
                }
                f.update_mean();
            }
            let l = &levels[li];
            let n = (l.w * l.h) as usize;
            let targets = targets_of(l);
            let lw = l.w;
            let up: Vec<(u32, [i8; 2], u16)> = (0..n)
                .into_par_iter()
                .map(|i| {
                    if targets[i] == 0 {
                        return (u32::MAX, [0, 0], 0);
                    }
                    let (x, y) = (i as i32 % lw, i as i32 / lw);
                    let ci = ((y / 2) * cw + x / 2) as usize;
                    let s = nnf.src.get(ci).copied().unwrap_or(u32::MAX);
                    let k = nnf.xf.get(ci).copied().unwrap_or(0) as usize;
                    let cand = if s != u32::MAX {
                        let (ox, oy) = (x % 2, y % 2);
                        let (sx, sy, sub) = if k == 0 {
                            ((s as i32 % cw) * 2 + ox, (s as i32 / cw) * 2 + oy, [0, 0])
                        } else {
                            let o = l.xf.offf[k][oi(ox, oy)];
                            let sb = nnf.sub[ci];
                            let (sx, fx) = split(((s as i32 % cw) as f32 + sb[0] as f32 / SUB) * 2.0 + o.0);
                            let (sy, fy) = split(((s as i32 / cw) as f32 + sb[1] as f32 / SUB) * 2.0 + o.1);
                            (sx, sy, [fx, fy])
                        };
                        if l.is_valid(sx, sy, k) { Some(((sy * lw + sx) as u32, sub)) } else { None }
                    } else {
                        None
                    };
                    match cand {
                        Some((s, sub)) => (s, sub, k as u16),
                        None => (l.valid_list[(i * 2654435761) % l.valid_list.len().max(1)], [0, 0], 0),
                    }
                })
                .collect();
            let f = Nnf {
                src: up.iter().map(|u| u.0).collect(),
                sub: up.iter().map(|u| u.1).collect(),
                xf: up.iter().map(|u| u.2).collect(),
                dist: vec![u32::MAX; n],
                targets,
            };
            nnf = f;
            // Re-vote from the upsampled field: the fine sources carry the
            // detail the coarse seed lacks (optimizing against the blurry
            // seed instead prefers smooth sources: visibly soft fills).
            vote(&mut levels[li], &nnf, false);
            let l = &levels[li];
            let nf = &nnf;
            let d: Vec<u32> = (0..nf.src.len())
                .into_par_iter()
                .map(|i| {
                    if nf.targets[i] != 0 && nf.src[i] != u32::MAX {
                        let s = nf.src[i] as i32;
                        l.dist(i as i32 % l.w, i as i32 / l.w, s % l.w, s / l.w, nf.sub[i], nf.xf[i] as usize, u32::MAX)
                    } else {
                        u32::MAX
                    }
                })
                .collect();
            nnf.dist = d;
        }
        if timing {
            eprintln!("      level {li}: upsample+vote+rescore {:.1} ms", tl.elapsed().as_secs_f64() * 1e3);
        }
        let iters = if li == top { p.iters_coarse } else if li == 0 { p.iters_fine } else { (p.iters_fine + 2).min(p.iters_coarse) };
        for it in 0..iters {
            let ta = std::time::Instant::now();
            pm_pass(&levels[li], &mut nnf, it % 2 == 1, p.seed.wrapping_add((li * 977 + it * 131) as u32));
            let tb = std::time::Instant::now();
            vote(&mut levels[li], &nnf, false);
            if timing {
                eprintln!("      level {li} ({}x{}) it {it}: pass {:.1} ms, vote {:.1} ms", levels[li].w, levels[li].h, (tb - ta).as_secs_f64() * 1e3, tb.elapsed().as_secs_f64() * 1e3);
            }
            // Distances refer to the updated image.
            let l = &levels[li];
            let upd: Vec<(usize, u32)> = (0..nnf.src.len())
                .into_par_iter()
                .filter(|&i| nnf.targets[i] != 0 && nnf.src[i] != u32::MAX)
                .map(|i| {
                    let (tx, ty) = (i as i32 % l.w, i as i32 / l.w);
                    let s = nnf.src[i] as i32;
                    (i, l.dist(tx, ty, s % l.w, s / l.w, nnf.sub[i], nnf.xf[i] as usize, u32::MAX))
                })
                .collect();
            for (i, d) in upd {
                nnf.dist[i] = d;
            }
        }
    }
    // Sharp final reconstruction: averaging overlapping patches (the EM
    // vote) smooths texture away, so the finest level's last step takes
    // every hole pixel from its own best match after one more search.
    if p.sharp {
        pm_pass(&levels[0], &mut nnf, true, p.seed ^ 0x5eed);
        vote(&mut levels[0], &nnf, true);
    }
    if timing {
        eprintln!("    pm done {:.0} ms", tm.elapsed().as_secs_f64() * 1e3);
    }
    let mut hist = vec![0usize; xf.len()];
    let (mut dsum, mut n) = (0u64, 0u64);
    for i in 0..nnf.src.len() {
        if nnf.targets[i] != 0 && nnf.src[i] != u32::MAX {
            hist[nnf.xf[i] as usize] += 1;
            dsum += nnf.dist[i] as u64;
            n += 1;
        }
    }
    let l0 = &levels[0];
    let out = img
        .iter()
        .zip(&l0.img)
        .zip(hole)
        .map(|((&orig, c), &hh)| if hh != 0 { px::pack(c[0], c[1], c[2], 255) } else { orig })
        .collect();
    let ms = t_all.elapsed().as_secs_f64() * 1e3;
    if matches!(engine, Engine::Auto) {
        gpu::cost::observe_op(op, units, ms, false);
    }
    let path = if gpu_ctx.is_some() { PathTaken::GpuThenCpu(why) } else { PathTaken::Cpu(why) };
    gpu::record(op, path.clone(), ms);
    (out, MatchStats { xf_hist: hist, mean_dist: dsum as f64 / n.max(1) as f64 }, path)
}

/// Raw PatchMatch NNF between a target and a source image of equal size.
#[cfg(test)]
pub fn nnf_offsets(target: &[u32], source: &[u32], w: usize, h: usize, iters: usize) -> (Vec<(i32, i32)>, Vec<u32>) {
    // Stack target | source side by side; the target half is the "hole".
    let (wi, hi) = (w as i32 * 2, h as i32);
    let mut img = vec![[0u8; 4]; (wi * hi) as usize];
    let mut hole = vec![0u8; (wi * hi) as usize];
    for y in 0..h {
        for x in 0..w {
            img[y * 2 * w + x] = px::unpremultiply(target[y * w + x]);
            img[y * 2 * w + w + x] = px::unpremultiply(source[y * w + x]);
        }
    }
    // Only source-half centers are valid sources; target-half centers are targets.
    let mut l = Lvl { w: wi, h: hi, img, hole: hole.clone(), valid: vec![0; (wi * hi) as usize], valid_list: vec![], valid_wide: vec![], xf: std::sync::Arc::new(Xfs::new(&Adapt::default())), mean: vec![], prox: vec![] };
    for y in R..hi - R {
        for x in w as i32 + R..wi - R {
            let i = (y * wi + x) as usize;
            l.valid[i] = 1;
            l.valid_list.push(i as u32);
        }
    }
    for y in 0..h {
        for x in 0..w {
            hole[y * 2 * w + x] = 1;
        }
    }
    let mut targets = vec![0u8; (wi * hi) as usize];
    for y in R..hi - R {
        for x in R..w as i32 - R {
            targets[(y * wi + x) as usize] = 1;
        }
    }
    let mut f = Nnf { src: vec![u32::MAX; (wi * hi) as usize], xf: vec![0; (wi * hi) as usize], sub: vec![[0; 2]; (wi * hi) as usize], dist: vec![u32::MAX; (wi * hi) as usize], targets };
    let mut rng = 99u32;
    for i in 0..f.src.len() {
        if f.targets[i] != 0 {
            let s = l.valid_list[(xorshift(&mut rng) as usize) % l.valid_list.len()];
            f.src[i] = s;
            f.dist[i] = l.dist(i as i32 % wi, i as i32 / wi, s as i32 % wi, s as i32 / wi, [0, 0], 0, u32::MAX);
        }
    }
    let _ = &mut l.hole;
    for it in 0..iters {
        pm_pass(&l, &mut f, it % 2 == 1, it as u32 * 7 + 3);
    }
    let offs = (0..w * h)
        .map(|k| {
            let (x, y) = (k % w, k / w);
            let i = y * 2 * w + x;
            let s = f.src[i];
            if s == u32::MAX { (0, 0) } else { ((s as i32 % wi) - w as i32 - x as i32, s as i32 / wi - y as i32) }
        })
        .collect();
    let d = (0..w * h).map(|k| f.dist[(k / w) * 2 * w + k % w]).collect();
    (offs, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(w: usize, h: usize, shift: (i32, i32)) -> Vec<u32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as i32 + shift.0, (i / w) as i32 + shift.1);
                let v = |a: i32, b: i32| (((a * 37 + b * 101) ^ (a * b)) & 255) as u8;
                px::pack(v(x, y), v(y, x), v(x + y, x - y), 255)
            })
            .collect()
    }

    #[test]
    fn nnf_converges_to_true_offset() {
        // The target is the source shifted by (5, -3): every patch has an
        // exact match, and PatchMatch should find nearly all of them.
        let (w, h) = (120usize, 90usize);
        let src = texture(w, h, (0, 0));
        let tgt = texture(w, h, (5, -3));
        let (offs, d) = nnf_offsets(&tgt, &src, w, h, 6);
        let mut exact = 0;
        let mut total = 0;
        for y in (R as usize + 4)..h - R as usize - 4 {
            for x in (R as usize)..w - R as usize - 6 {
                total += 1;
                let k = y * w + x;
                if offs[k] == (5, -3) && d[k] == 0 {
                    exact += 1;
                }
            }
        }
        assert!(exact as f32 / total as f32 > 0.97, "{exact}/{total} exact matches");
    }

    /// Mean abs error in the hole of completing `img` (ground truth) when
    /// sources may only come from `allowed`, whose content is `src_img`.
    fn adapt_err(img: &[u32], src_img: &[u32], allowed: &[u8], hole: &[u8], w: usize, h: usize, adapt: Adapt) -> f64 {
        let mut damaged: Vec<u32> = (0..w * h).map(|i| if allowed[i] != 0 { src_img[i] } else { img[i] }).collect();
        for i in 0..w * h {
            if hole[i] != 0 {
                damaged[i] = px::pack(0, 255, 0, 255);
            }
        }
        let out = complete(&damaged, hole, w, h, Some(allowed), &Params { adapt, iters_fine: 4, ..Params::default() });
        let (mut e, mut n) = (0f64, 0);
        for i in 0..w * h {
            if hole[i] != 0 {
                let (a, b) = (px::unpack(out[i]), px::unpack(img[i]));
                e += (0..3).map(|c| (a[c] as f64 - b[c] as f64).abs()).sum::<f64>() / 3.0;
                n += 1;
            }
        }
        e / n as f64
    }

    fn disk_hole(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Vec<u8> {
        (0..w * h).map(|i| (((i % w) as f32 - cx).powi(2) + ((i / w) as f32 - cy).powi(2) < r * r) as u8).collect()
    }

    #[test]
    fn rotation_adaptation_uses_rotated_sources() {
        // Ground truth: stripes at 0°; the only allowed source shows them at 45°.
        let (w, h) = (200usize, 160usize);
        let stripes = |deg: f32| -> Vec<u32> {
            let (s, c) = deg.to_radians().sin_cos();
            (0..w * h).map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let v = (((x * c + y * s) / 5.0).floor() as i32).rem_euclid(2);
                if v == 0 { px::pack(230, 200, 40, 255) } else { px::pack(30, 40, 120, 255) }
            }).collect()
        };
        let truth = stripes(0.0);
        let src = stripes(45.0);
        let allowed: Vec<u8> = (0..w * h).map(|i| (i % w >= 120) as u8).collect();
        let hole = disk_hole(w, h, 55.0, 80.0, 16.0);
        let plain = adapt_err(&truth, &src, &allowed, &hole, w, h, Adapt::default());
        let rot = adapt_err(&truth, &src, &allowed, &hole, w, h, Adapt { rotation: RotationAdapt::Medium, ..Default::default() });
        assert!(rot < plain * 0.6, "rotation {rot:.1} vs none {plain:.1}");
    }

    #[test]
    fn scale_adaptation_uses_scaled_sources() {
        // Truth: dots of radius 4 on a 12 px lattice; the only allowed source
        // has the same pattern ×0.83 (radius 3.3, 10 px). Unlike stripes,
        // a dot can't be stitched to a bigger one from unscaled patches.
        let (w, h) = (200usize, 120usize);
        let dots = |pitch: f32, r: f32| -> Vec<u32> {
            (0..w * h).map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let (dx, dy) = (x - (x / pitch).round() * pitch, y - (y / pitch).round() * pitch);
                if dx * dx + dy * dy < r * r { px::pack(230, 200, 40, 255) } else { px::pack(30, 40, 120, 255) }
            }).collect()
        };
        let truth = dots(12.0, 4.0);
        let src = dots(10.0, 3.3);
        let allowed: Vec<u8> = (0..w * h).map(|i| (i % w >= 120) as u8).collect();
        let hole = disk_hole(w, h, 55.0, 60.0, 22.0);
        let run = |adapt: Adapt| {
            let mut damaged: Vec<u32> = (0..w * h).map(|i| if allowed[i] != 0 { src[i] } else { truth[i] }).collect();
            for i in 0..w * h {
                if hole[i] != 0 {
                    damaged[i] = px::pack(0, 255, 0, 255);
                }
            }
            complete_stats(&damaged, &hole, w, h, Some(&allowed), &Params { adapt, iters_fine: 4, ..Params::default() }).1
        };
        let (plain, sc) = (run(Adapt::default()), run(Adapt { scale: true, ..Default::default() }));
        // Transforms 1 and 2 are the 0.8× and 0.9× scales. Flat background
        // patches match under any transform, so only the dots need them.
        let shrink = sc.xf_hist[1] + sc.xf_hist[2];
        assert!(shrink > sc.xf_hist[3] + sc.xf_hist[4], "scaled matches {:?}", sc.xf_hist);
        assert!(sc.mean_dist < 0.8 * plain.mean_dist, "patch SSD: scale {:.0} vs none {:.0}", sc.mean_dist, plain.mean_dist);
    }

    #[test]
    fn mirror_adaptation_uses_flipped_sources() {
        // A horizontal sawtooth ramp; the allowed source is its mirror image.
        let (w, h) = (200usize, 120usize);
        let saw = |flip: bool| -> Vec<u32> {
            (0..w * h).map(|i| {
                let x = (i % w) as i32;
                let t = (x.rem_euclid(16)) as u8 * 15;
                let v = if flip { 225 - t } else { t };
                px::pack(v, 255 - v, 128, 255)
            }).collect()
        };
        let truth = saw(false);
        let src = saw(true);
        let allowed: Vec<u8> = (0..w * h).map(|i| (i % w >= 120) as u8).collect();
        let hole = disk_hole(w, h, 55.0, 60.0, 14.0);
        // Fraction of hole pixels whose red rises to the right (the truth's
        // ramps rise; the unmirrored source's fall). Phase-independent.
        let rising = |adapt: Adapt| {
            let mut damaged: Vec<u32> = (0..w * h).map(|i| if allowed[i] != 0 { src[i] } else { truth[i] }).collect();
            for i in 0..w * h {
                if hole[i] != 0 {
                    damaged[i] = px::pack(0, 255, 0, 255);
                }
            }
            let out = complete(&damaged, &hole, w, h, Some(&allowed), &Params { adapt, iters_fine: 4, ..Params::default() });
            let (mut up, mut n) = (0, 0);
            for i in 0..w * h - 1 {
                if hole[i] != 0 && hole[i + 1] != 0 {
                    let d = px::unpack(out[i + 1])[0] as i32 - px::unpack(out[i])[0] as i32;
                    if d.abs() < 100 {
                        up += (d > 0) as usize;
                        n += 1;
                    }
                }
            }
            up as f32 / n as f32
        };
        let (plain, mir) = (rising(Adapt::default()), rising(Adapt { mirror: true, ..Default::default() }));
        assert!(mir > 0.8 && plain < 0.3, "rising fraction: mirror {mir:.2} vs none {plain:.2}");
    }

    #[test]
    fn color_adaptation_matches_brightness() {
        // The allowed source is the same texture 40 levels darker.
        let (w, h) = (200usize, 120usize);
        let tex = |off: i32| -> Vec<u32> {
            (0..w * h).map(|i| {
                let (x, y) = ((i % w) as i32, (i / w) as i32);
                let v = (140 + ((x * 37 + y * 101) ^ (x * y)) % 40 - off).clamp(0, 255) as u8;
                px::pack(v, v, v, 255)
            }).collect()
        };
        let truth = tex(0);
        let src = tex(40);
        let allowed: Vec<u8> = (0..w * h).map(|i| (i % w >= 120) as u8).collect();
        let hole = disk_hole(w, h, 55.0, 60.0, 14.0);
        let plain = adapt_err(&truth, &src, &allowed, &hole, w, h, Adapt::default());
        let col = adapt_err(&truth, &src, &allowed, &hole, w, h, Adapt { color: ColorAdapt::VeryHigh, ..Default::default() });
        assert!(plain > 25.0 && col < 16.0, "color adaptation {col:.1} vs none {plain:.1}");
    }

    #[test]
    fn completion_continues_stripes() {
        // Vertical stripes with a square hole: the fill should continue them.
        let (w, h) = (160usize, 120usize);
        let stripe = |x: usize| if (x / 6) % 2 == 0 { px::pack(220, 40, 40, 255) } else { px::pack(30, 30, 200, 255) };
        let img: Vec<u32> = (0..w * h).map(|i| stripe(i % w)).collect();
        let mut hole = vec![0u8; w * h];
        for y in 40..80 {
            for x in 60..100 {
                hole[y * w + x] = 1;
            }
        }
        let mut damaged = img.clone();
        for i in 0..w * h {
            if hole[i] != 0 {
                damaged[i] = px::pack(0, 255, 0, 255);
            }
        }
        let out = complete(&damaged, &hole, w, h, None, &Params::default());
        let mut err = 0f64;
        let mut n = 0;
        for i in 0..w * h {
            if hole[i] != 0 {
                let (a, b) = (px::unpack(out[i]), px::unpack(img[i]));
                err += (0..3).map(|c| (a[c] as f64 - b[c] as f64).abs()).sum::<f64>() / 3.0;
                n += 1;
            }
        }
        // Stripes may be phase-shifted by a few pixels but must not be green
        // or a blur: the per-pixel colors must be stripe colors.
        let green = (0..w * h).filter(|&i| hole[i] != 0 && px::unpack(out[i])[1] > 150).count();
        assert_eq!(green, 0, "hole color leaked");
        let pure = (0..w * h).filter(|&i| hole[i] != 0).filter(|&i| {
            let c = px::unpack(out[i]);
            (c[0] > 180 && c[2] < 80) || (c[2] > 160 && c[0] < 70)
        }).count();
        assert!(pure as f32 / n as f32 > 0.85, "only {pure}/{n} pixels are clean stripe colors (mean err {})", err / n as f64);
    }
}

#[cfg(test)]
mod sharp_fill {
    use super::*;
    #[test]
    fn fill_keeps_noise_variance() {
        let (w, h) = (300usize, 240usize);
        let mut rng = 12345u32;
        let img: Vec<u32> = (0..w * h).map(|_| { let v = (xorshift(&mut rng) % 200) as u8 + 20; px::pack(v, v, v, 255) }).collect();
        let hole: Vec<u8> = (0..w * h).map(|i| ((((i % w) as f32 - 150.0) / 60.0).powi(2) + (((i / w) as f32 - 120.0) / 40.0).powi(2) < 1.0) as u8).collect();
        let out = complete(&img, &hole, w, h, None, &Params::default());
        let var = |sel: &dyn Fn(usize) -> bool| {
            let v: Vec<f64> = (0..w * h).filter(|&i| sel(i)).map(|i| px::unpack(out[i])[0] as f64).collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let (hs, os) = (var(&|i| hole[i] != 0), var(&|i| hole[i] == 0));
        assert!(hs > 0.8 * os, "fill smoothed the noise: {hs:.1} vs {os:.1}");
    }
}
