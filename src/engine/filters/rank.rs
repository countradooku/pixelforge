//! Rank and edge-preserving filters built on constant-time sliding histograms.
//!
//! Perreault & Hébert's algorithm: one 256-bin histogram per column (updated
//! by one add and one remove per pixel as the window slides down) plus a
//! kernel histogram split into 16 coarse bins and 16 lazily-synchronized fine
//! segments. Per pixel the work is O(1) in the radius, so a radius-100
//! median costs about the same as radius 3.
//!
//! Maximum / Minimum use van Herk / Gil-Werman: three comparisons per pixel
//! per axis for any radius.

use super::util::{merge, opaque, split};

struct Columns {
    fine: Vec<u16>,
    coarse: Vec<u16>,
}

pub struct Kernel<'a> {
    cols: &'a Columns,
    r: usize,
    x: usize,
    coarse: [u32; 16],
    fine: [[u32; 16]; 16],
    last: [isize; 16],
    pub count: u32,
}

impl<'a> Kernel<'a> {
    fn reset(&mut self, x: usize) {
        self.x = x;
        self.coarse = [0; 16];
        for c in x..=x + 2 * self.r {
            let s = &self.cols.coarse[c * 16..c * 16 + 16];
            for (k, v) in self.coarse.iter_mut().zip(s) {
                *k += *v as u32;
            }
        }
        self.last = [-1; 16];
    }

    fn step(&mut self) {
        let add = self.x + 2 * self.r + 1;
        let sub = self.x;
        let a = &self.cols.coarse[add * 16..add * 16 + 16];
        let s = &self.cols.coarse[sub * 16..sub * 16 + 16];
        for k in 0..16 {
            self.coarse[k] = self.coarse[k] + a[k] as u32 - s[k] as u32;
        }
        self.x += 1;
    }

    /// Brings fine segment `seg` up to date with the current window.
    #[inline]
    fn sync(&mut self, seg: usize) -> &[u32; 16] {
        let span = 2 * self.r + 1;
        let x = self.x as isize;
        let last = self.last[seg];
        let f = &mut self.fine[seg];
        if last < 0 || x - last > span as isize {
            *f = [0; 16];
            for c in self.x..self.x + span {
                let col = &self.cols.fine[c * 256 + seg * 16..c * 256 + seg * 16 + 16];
                for (k, v) in f.iter_mut().zip(col) {
                    *k += *v as u32;
                }
            }
        } else {
            for j in (last + 1)..=x {
                let j = j as usize;
                let a = &self.cols.fine[(j + 2 * self.r) * 256 + seg * 16..][..16];
                let s = &self.cols.fine[(j - 1) * 256 + seg * 16..][..16];
                for k in 0..16 {
                    f[k] = f[k] + a[k] as u32 - s[k] as u32;
                }
            }
        }
        self.last[seg] = x;
        &self.fine[seg]
    }

    /// Value of rank `k` (0-based) in the window.
    #[inline]
    pub fn rank(&mut self, k: u32) -> u8 {
        let mut acc = 0;
        let mut seg = 15;
        for (s, &c) in self.coarse.iter().enumerate() {
            if acc + c > k {
                seg = s;
                break;
            }
            acc += c;
        }
        let f = *self.sync(seg);
        for (i, &c) in f.iter().enumerate() {
            if acc + c > k {
                return (seg * 16 + i) as u8;
            }
            acc += c;
        }
        255
    }

    /// Count of pixels with value `v` in the window.
    #[inline]
    pub fn bin_segment(&mut self, seg: usize) -> [u32; 16] {
        *self.sync(seg)
    }
}

/// Runs `f(kernel, center_value)` for every output pixel of a plane padded
/// by `r` on each side.
fn sliding(p: &[u8], pw: usize, ph: usize, r: usize, out: &mut [u8], mut f: impl FnMut(&mut Kernel, u8) -> u8) {
    let ow = pw - 2 * r;
    let oh = ph - 2 * r;
    let mut cols = Columns { fine: vec![0u16; pw * 256], coarse: vec![0u16; pw * 16] };
    let add = |cols: &mut Columns, y: usize, d: i32| {
        let row = &p[y * pw..(y + 1) * pw];
        for (x, &v) in row.iter().enumerate() {
            let fi = x * 256 + v as usize;
            let ci = x * 16 + (v >> 4) as usize;
            cols.fine[fi] = (cols.fine[fi] as i32 + d) as u16;
            cols.coarse[ci] = (cols.coarse[ci] as i32 + d) as u16;
        }
    };
    for y in 0..=2 * r {
        add(&mut cols, y, 1);
    }
    let count = ((2 * r + 1) * (2 * r + 1)) as u32;
    for y in 0..oh {
        if y > 0 {
            add(&mut cols, y - 1, -1);
            add(&mut cols, y + 2 * r, 1);
        }
        let mut k = Kernel { cols: &cols, r, x: 0, coarse: [0; 16], fine: [[0; 16]; 16], last: [-1; 16], count };
        k.reset(0);
        for x in 0..ow {
            if x > 0 {
                k.step();
            }
            let c = p[(y + r) * pw + x + r];
            out[y * ow + x] = f(&mut k, c);
        }
    }
}

fn per_plane(buf: &mut [u32], w: usize, h: usize, r: usize, f: impl Fn(&[u8], &mut [u8]) + Sync) {
    // Output is the un-padded center, written back into the padded buffer.
    let opaque = opaque(buf);
    let mut planes = split(buf);
    let chans = if opaque { 3 } else { 4 };
    let (ow, oh) = (w - 2 * r, h - 2 * r);
    use rayon::prelude::*;
    let outs: Vec<Vec<u8>> = planes[..chans]
        .par_iter()
        .map(|p| {
            let mut o = vec![0u8; ow * oh];
            f(p, &mut o);
            o
        })
        .collect();
    for (ci, o) in outs.into_iter().enumerate() {
        let p = &mut planes[ci];
        for y in 0..oh {
            p[(y + r) * w + r..(y + r) * w + r + ow].copy_from_slice(&o[y * ow..(y + 1) * ow]);
        }
    }
    merge(&planes, buf);
}

/// 3x3 median with a column-sort network: sort each column of three, then
/// median = med3(max of lows, med3 of mids, min of highs). Pure u8 min/max
/// over contiguous rows, so it vectorizes 16 lanes wide.
fn median3(p: &[u8], w: usize, h: usize, out: &mut [u8]) {
    let ow = w - 2;
    let (mut lo, mut mi, mut hi) = (vec![0u8; w], vec![0u8; w], vec![0u8; w]);
    for y in 0..h - 2 {
        let (a, b, c) = (&p[y * w..(y + 1) * w], &p[(y + 1) * w..(y + 2) * w], &p[(y + 2) * w..(y + 3) * w]);
        for x in 0..w {
            let (l1, h1) = (a[x].min(b[x]), a[x].max(b[x]));
            let l = l1.min(c[x]);
            let hh = h1.max(c[x]);
            lo[x] = l;
            hi[x] = hh;
            mi[x] = h1.min(l1.max(c[x]));
        }
        let o = &mut out[y * ow..(y + 1) * ow];
        for x in 0..ow {
            let mxl = lo[x].max(lo[x + 1]).max(lo[x + 2]);
            let mnh = hi[x].min(hi[x + 1]).min(hi[x + 2]);
            let (m0, m1, m2) = (mi[x], mi[x + 1], mi[x + 2]);
            let mm = m0.min(m1).max(m0.max(m1).min(m2));
            o[x] = mxl.min(mm).max(mxl.max(mm).min(mnh));
        }
    }
}

pub fn median(buf: &mut [u32], w: usize, h: usize, r: usize) {
    if r == 0 {
        return;
    }
    if r == 1 {
        per_plane(buf, w, h, 1, |p, o| median3(p, w, h, o));
        return;
    }
    if r == 2 {
        per_plane(buf, w, h, 2, |p, o| super::fast::median5(p, w, h, o));
        return;
    }
    per_plane(buf, w, h, r, |p, o| sliding(p, w, h, r, o, |k, _| k.rank(k.count / 2)));
}

/// Median applied only where it differs from the original by more than `threshold`.
pub fn dust_scratches(buf: &mut [u32], w: usize, h: usize, r: usize, threshold: u8) {
    if r == 0 {
        return;
    }
    if r <= 2 {
        // Network median, then keep originals that are within the threshold.
        per_plane(buf, w, h, r, |p, o| {
            if r == 1 { median3(p, w, h, o) } else { super::fast::median5(p, w, h, o) }
            let ow = w - 2 * r;
            for (i, m) in o.iter_mut().enumerate() {
                let c = p[(i / ow + r) * w + i % ow + r];
                if (*m as i32 - c as i32).unsigned_abs() <= threshold as u32 {
                    *m = c;
                }
            }
        });
        return;
    }
    per_plane(buf, w, h, r, |p, o| {
        sliding(p, w, h, r, o, |k, c| {
            let m = k.rank(k.count / 2);
            if (m as i32 - c as i32).unsigned_abs() > threshold as u32 { m } else { c }
        })
    });
}

/// Edge-preserving box blur (Photoshop "Surface Blur"): each neighbor is
/// weighted by `1 - |v - c| / (2.5 * threshold)`; with `hard` the weight is a
/// step (Photoshop "Smart Blur").
///
/// Implemented as Yang's constant-time bilateral filter: for K intensity
/// levels c_k, box-filter `w(I - c_k)` and `w(I - c_k) * I` (running sums,
/// O(1) per pixel for any radius), then interpolate between the two levels
/// that bracket each pixel's own value. Everything is contiguous f32 row
/// math that vectorizes.
pub fn surface(buf: &mut [u32], w: usize, h: usize, r: usize, threshold: f32, hard: bool) {
    if r == 0 {
        return;
    }
    let t = threshold.max(1.0);
    if hard {
        smart_hist(buf, w, h, r, t);
        return;
    }
    let band = 2.5 * t;
    // Level spacing well under the kernel width keeps interpolation accurate.
    let k = ((255.0 / (band * 0.4)).ceil() as usize + 1).clamp(3, 96);
    per_plane(buf, w, h, r, |p, o| pbfic(p, w, h, r, k, band, o));
}

/// Step-weight (Smart Blur) variant on the sliding histogram: only the
/// segments inside ±threshold are visited.
fn smart_hist(buf: &mut [u32], w: usize, h: usize, r: usize, t: f32) {
    let ti = t.floor() as i32;
    per_plane(buf, w, h, r, |p, o| {
        sliding(p, w, h, r, o, |k, c| {
            let lo = (c as i32 - ti).max(0) as usize;
            let hi = (c as i32 + ti).min(255) as usize;
            let (mut n, mut sv) = (0u32, 0u32);
            for seg in lo >> 4..=hi >> 4 {
                let bins = k.bin_segment(seg);
                for i in 0..16 {
                    let v = seg * 16 + i;
                    let inside = (v >= lo && v <= hi) as u32;
                    n += bins[i] * inside;
                    sv += bins[i] * inside * v as u32;
                }
            }
            if n > 0 { ((sv + n / 2) / n) as u8 } else { c }
        })
    });
}

fn pbfic(p: &[u8], w: usize, h: usize, r: usize, levels: usize, band: f32, out: &mut [u8]) {
    let (ow, oh) = (w - 2 * r, h - 2 * r);
    let n = 2 * r + 1;
    let step = 255.0 / (levels - 1) as f32;
    let inv_band = 1.0 / band;
    let fp: Vec<f32> = p.iter().map(|&v| v as f32).collect();
    // Only levels bracketing values that actually occur in the output matter.
    let (mut cmin, mut cmax) = (255u8, 0u8);
    for y in 0..oh {
        for &v in &p[(y + r) * w + r..(y + r) * w + r + ow] {
            cmin = cmin.min(v);
            cmax = cmax.max(v);
        }
    }
    let l0 = ((cmin as f32 / step).floor() as usize).min(levels - 1);
    let l1 = ((cmax as f32 / step).ceil() as usize).min(levels - 1).max(l0);
    if l0 == l1 {
        // Flat plane: the weighted mean of a constant is the constant.
        for y in 0..oh {
            out[y * ow..(y + 1) * ow].copy_from_slice(&p[(y + r) * w + r..(y + r) * w + r + ow]);
        }
        return;
    }
    let mut res = vec![0f32; ow * oh];
    let mut prev = vec![0f32; ow * oh];
    let mut cur = vec![0f32; ow * oh];
    let mut wpl = vec![0f32; w * h];
    let mut ipl = vec![0f32; w * h];
    let mut hw = vec![0f32; ow * h];
    let mut hi = vec![0f32; ow * h];
    let mut vw = vec![0f32; ow];
    let mut vi = vec![0f32; ow];
    for li in l0..=l1 {
        let ck = li as f32 * step;
        // Tent weights computed arithmetically (vectorizes; no table gather).
        for (i, (wv, iv)) in wpl.iter_mut().zip(ipl.iter_mut()).enumerate() {
            let v = fp[i];
            let wt = (1.0 - (v - ck).abs() * inv_band).max(0.0);
            *wv = wt;
            *iv = wt * v;
        }
        // Horizontal running sums, four rows at a time for independent chains.
        let mut y = 0;
        while y < h {
            let rows = (h - y).min(4);
            let mut sw = [0f32; 4];
            let mut si = [0f32; 4];
            for k in 0..rows {
                let b = (y + k) * w;
                for x in 0..n {
                    sw[k] += wpl[b + x];
                    si[k] += ipl[b + x];
                }
                hw[(y + k) * ow] = sw[k];
                hi[(y + k) * ow] = si[k];
            }
            for x in 1..ow {
                for k in 0..rows {
                    let b = (y + k) * w;
                    sw[k] += wpl[b + x + n - 1] - wpl[b + x - 1];
                    si[k] += ipl[b + x + n - 1] - ipl[b + x - 1];
                    hw[(y + k) * ow + x] = sw[k];
                    hi[(y + k) * ow + x] = si[k];
                }
            }
            y += rows;
        }
        // Vertical running sums, row-major across all columns at once.
        vw.iter_mut().for_each(|v| *v = 0.0);
        vi.iter_mut().for_each(|v| *v = 0.0);
        for y in 0..n {
            for x in 0..ow {
                vw[x] += hw[y * ow + x];
                vi[x] += hi[y * ow + x];
            }
        }
        for y in 0..oh {
            if y > 0 {
                let (a, s) = ((y + n - 1) * ow, (y - 1) * ow);
                for x in 0..ow {
                    vw[x] += hw[a + x] - hw[s + x];
                    vi[x] += hi[a + x] - hi[s + x];
                }
            }
            let o = &mut cur[y * ow..(y + 1) * ow];
            for x in 0..ow {
                o[x] = if vw[x] > 1e-6 { vi[x] / vw[x] } else { ck };
            }
        }
        if li > l0 {
            let c_lo = ck - step;
            for y in 0..oh {
                let crow = &fp[(y + r) * w + r..(y + r) * w + r + ow];
                for x in 0..ow {
                    let c = crow[x];
                    if c >= c_lo && c <= ck {
                        let f = (c - c_lo) / step;
                        let i = y * ow + x;
                        res[i] = prev[i] + (cur[i] - prev[i]) * f;
                    }
                }
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    for (o, &v) in out.iter_mut().zip(&res) {
        *o = (v + 0.5).clamp(0.0, 255.0) as u8;
    }
}

/// Despeckle: 3x3 median except across strong edges.
pub fn despeckle(buf: &mut [u32], w: usize, h: usize) {
    let orig = buf.to_vec();
    median(buf, w, h, 1);
    let lum = |p: u32| {
        let [r, g, b, _] = p.to_le_bytes();
        (r as i32 * 77 + g as i32 * 150 + b as i32 * 29) >> 8
    };
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            let gx = lum(orig[i + 1]) - lum(orig[i - 1]);
            let gy = lum(orig[i + w]) - lum(orig[i - w]);
            if gx.abs() + gy.abs() > 48 {
                buf[i] = orig[i];
            }
        }
    }
}

// ----- van Herk / Gil-Werman max & min -------------------------------------------

fn van_herk(line: &mut [u8], r: usize, max: bool, g: &mut Vec<u8>, hbuf: &mut Vec<u8>) {
    let n = line.len();
    let k = 2 * r + 1;
    let op = |a: u8, b: u8| if max { a.max(b) } else { a.min(b) };
    let ident = if max { 0u8 } else { 255u8 };
    // Pad by r on each side with the edge-neutral value.
    let m = n + 2 * r;
    let padded = |i: usize| -> u8 { if i < r || i >= r + n { ident } else { line[i - r] } };
    g.clear();
    g.resize(m, ident);
    hbuf.clear();
    hbuf.resize(m, ident);
    for i in 0..m {
        g[i] = if i % k == 0 { padded(i) } else { op(g[i - 1], padded(i)) };
    }
    for i in (0..m).rev() {
        hbuf[i] = if i == m - 1 || (i + 1) % k == 0 { padded(i) } else { op(hbuf[i + 1], padded(i)) };
    }
    for i in 0..n {
        // Window [i, i + 2r] in padded coordinates.
        line[i] = op(hbuf[i], g[(i + 2 * r).min(m - 1)]);
    }
}

pub fn max_min(buf: &mut [u32], w: usize, h: usize, r: usize, max: bool) {
    if r == 0 {
        return;
    }
    let opaque = opaque(buf);
    let mut planes = split(buf);
    let chans = if opaque { 3 } else { 4 };
    use rayon::prelude::*;
    planes[..chans].par_iter_mut().for_each(|p| {
        let (mut g, mut hb) = (Vec::new(), Vec::new());
        for y in 0..h {
            van_herk(&mut p[y * w..(y + 1) * w], r, max, &mut g, &mut hb);
        }
        let mut t = super::blur::transpose(p, w, h);
        for x in 0..w {
            van_herk(&mut t[x * h..(x + 1) * h], r, max, &mut g, &mut hb);
        }
        *p = super::blur::transpose(&t, h, w);
    });
    merge(&planes, buf);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::px::{pack, unpack};

    fn naive_median(p: &[u8], w: usize, r: usize, x: usize, y: usize) -> u8 {
        let mut v = Vec::new();
        for dy in 0..=2 * r {
            for dx in 0..=2 * r {
                v.push(p[(y + dy) * w + x + dx]);
            }
        }
        v.sort();
        v[v.len() / 2]
    }

    #[test]
    fn median_matches_naive() {
        let (w, h, r) = (40, 30, 4);
        let buf: Vec<u32> = (0..w * h)
            .map(|i| {
                let v = ((i * 7919 + (i / w) * 104_729) % 251) as u8;
                pack(v, v / 2, 255 - v, 255)
            })
            .collect();
        let planes = split(&buf);
        let mut m = buf.clone();
        median(&mut m, w, h, r);
        for &(x, y) in &[(0, 0), (5, 7), (31, 21), (17, 3)] {
            let want = naive_median(&planes[0], w, r, x, y);
            assert_eq!(unpack(m[(y + r) * w + x + r])[0], want, "({x},{y})");
            let want_b = naive_median(&planes[2], w, r, x, y);
            assert_eq!(unpack(m[(y + r) * w + x + r])[2], want_b);
        }
    }

    #[test]
    fn median3_network_matches_naive() {
        let (w, h) = (37, 23);
        let buf: Vec<u32> = (0..w * h).map(|i| { let v = ((i * 7919 + 13) % 253) as u8; pack(v, 255 - v, v / 3, 255) }).collect();
        let planes = split(&buf);
        let mut m = buf.clone();
        median(&mut m, w, h, 1);
        for y in 0..h - 2 {
            for x in 0..w - 2 {
                assert_eq!(unpack(m[(y + 1) * w + x + 1])[0], naive_median(&planes[0], w, 1, x, y));
                assert_eq!(unpack(m[(y + 1) * w + x + 1])[1], naive_median(&planes[1], w, 1, x, y));
            }
        }
    }

    #[test]
    fn max_is_dilation() {
        let (w, h) = (20, 20);
        let mut b = vec![pack(0, 0, 0, 255); w * h];
        b[10 * w + 10] = pack(200, 200, 200, 255);
        max_min(&mut b, w, h, 2, true);
        assert_eq!(unpack(b[12 * w + 12])[0], 200);
        assert_eq!(unpack(b[13 * w + 10])[0], 0);
        assert_eq!(unpack(b[8 * w + 8])[0], 200);
    }

    #[test]
    fn surface_blur_preserves_edges() {
        let (w, h) = (30, 10);
        let mut b: Vec<u32> = (0..w * h).map(|i| if i % w < 15 { pack(20, 20, 20, 255) } else { pack(220, 220, 220, 255) }).collect();
        surface(&mut b, w, h, 3, 15.0, false);
        assert_eq!(unpack(b[5 * w + 14])[0], 20);
        assert_eq!(unpack(b[5 * w + 15])[0], 220);
    }
}
