//! Planar fast paths for small kernels.
//!
//! Every inner loop here is "one tap = one contiguous slice operation", so
//! LLVM turns them into 8–16-lane NEON arithmetic with no per-pixel branches,
//! clamps or gathers. Buffers are padded, so interior-only loops are enough.

use super::util::{merge, opaque, split};

/// Normalized Gaussian taps in 8.8 fixed point (sum = 256).
pub fn gauss_kernel(sigma: f32) -> Vec<u16> {
    let r = (sigma * 3.0).ceil().max(1.0) as i32;
    let w: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = w.iter().sum();
    let mut k: Vec<u16> = w.iter().map(|v| (v / s * 256.0).round() as u16).collect();
    // Fix rounding so the taps sum to exactly 256 (keeps flat areas exact).
    let total: i32 = k.iter().map(|&v| v as i32).sum();
    let mid = k.len() / 2;
    k[mid] = (k[mid] as i32 + 256 - total) as u16;
    k
}

/// Largest sigma for which direct convolution beats running box sums.
pub const DIRECT_SIGMA_MAX: f32 = 2.6;

/// Separable convolution of the padded buffer with a symmetric kernel;
/// result valid in the center (pad = kernel radius).
pub fn conv_sep(buf: &mut [u32], w: usize, h: usize, k: &[u16]) {
    let r = k.len() / 2;
    if r == 0 || w <= 2 * r || h <= 2 * r {
        return;
    }
    let opaque = opaque(buf);
    let mut planes = split(buf);
    let chans = if opaque { 3 } else { 4 };
    let ow = w - 2 * r;
    let mut hrow = vec![0u16; w * h];
    let mut acc = vec![0u32; ow];
    for p in planes.iter_mut().take(chans) {
        // Horizontal: hrow[y][x] = Σ k[j] * p[y][x + j] (x in center columns).
        for y in 0..h {
            let src = &p[y * w..(y + 1) * w];
            let dst = &mut hrow[y * w + r..y * w + r + ow];
            dst.iter_mut().for_each(|d| *d = 0);
            for (j, &kw) in k.iter().enumerate() {
                let s = &src[j..j + ow];
                for (d, &v) in dst.iter_mut().zip(s) {
                    *d += v as u16 * kw;
                }
            }
        }
        // Vertical over the horizontal result, center rows only.
        for y in r..h - r {
            acc.iter_mut().for_each(|a| *a = 0);
            for (j, &kw) in k.iter().enumerate() {
                let s = &hrow[(y + j - r) * w + r..(y + j - r) * w + r + ow];
                for (a, &v) in acc.iter_mut().zip(s) {
                    *a += v as u32 * kw as u32;
                }
            }
            let dst = &mut p[y * w + r..y * w + r + ow];
            for (d, &a) in dst.iter_mut().zip(&acc) {
                *d = ((a + 32768) >> 16) as u8;
            }
        }
    }
    merge(&planes, buf);
}

/// Square max/min of radius r by shifted-slice comparisons (O(r), but every
/// comparison is 16 lanes wide; beats van Herk for small radii).
pub fn maxmin_small(buf: &mut [u32], w: usize, h: usize, r: usize, max: bool) {
    let opaque = opaque(buf);
    let mut planes = split(buf);
    let chans = if opaque { 3 } else { 4 };
    let ow = w - 2 * r;
    let mut tmp = vec![0u8; w * h];
    for p in planes.iter_mut().take(chans) {
        for y in 0..h {
            let src = &p[y * w..(y + 1) * w];
            let dst = &mut tmp[y * w + r..y * w + r + ow];
            dst.copy_from_slice(&src[..ow]);
            for j in 1..=2 * r {
                let s = &src[j..j + ow];
                if max {
                    dst.iter_mut().zip(s).for_each(|(d, &v)| *d = (*d).max(v));
                } else {
                    dst.iter_mut().zip(s).for_each(|(d, &v)| *d = (*d).min(v));
                }
            }
        }
        for y in r..h - r {
            let mut row: Vec<u8> = tmp[(y - r) * w + r..(y - r) * w + r + ow].to_vec();
            for j in 1..=2 * r {
                let s = &tmp[(y - r + j) * w + r..(y - r + j) * w + r + ow];
                if max {
                    row.iter_mut().zip(s).for_each(|(d, &v)| *d = (*d).max(v));
                } else {
                    row.iter_mut().zip(s).for_each(|(d, &v)| *d = (*d).min(v));
                }
            }
            p[y * w + r..y * w + r + ow].copy_from_slice(&row);
        }
    }
    merge(&planes, buf);
}

/// Exact median of 25 by forgetful selection over whole rows: keep 14
/// candidates, repeatedly drop the current min and max (neither can be the
/// median) and pull in the next value. All operations are elementwise min/max
/// on contiguous slices.
pub fn median5(p: &[u8], w: usize, h: usize, out: &mut [u8]) {
    let ow = w - 4;
    let oh = h - 4;
    let mut v: Vec<Vec<u8>> = (0..14).map(|_| vec![0u8; ow]).collect();
    for y in 0..oh {
        let src = |i: usize| -> &[u8] {
            let (dy, dx) = (i / 5, i % 5);
            &p[(y + dy) * w + dx..(y + dy) * w + dx + ow]
        };
        for (i, vi) in v.iter_mut().enumerate() {
            vi.copy_from_slice(src(i));
        }
        let mut n = 14;
        let mut next = 14;
        loop {
            // Move the min of v[0..n] into v[0] and the max into v[n-1].
            for i in 1..n {
                let (a, b) = v.split_at_mut(i);
                for (x, y_) in a[0].iter_mut().zip(b[0].iter_mut()) {
                    let (lo, hi) = ((*x).min(*y_), (*x).max(*y_));
                    *x = lo;
                    *y_ = hi;
                }
            }
            for i in 1..n - 1 {
                let (a, b) = v.split_at_mut(n - 1);
                for (x, y_) in a[i].iter_mut().zip(b[0].iter_mut()) {
                    let (lo, hi) = ((*x).min(*y_), (*x).max(*y_));
                    *x = lo;
                    *y_ = hi;
                }
            }
            if next == 25 {
                // Remaining set is {min, median, max} once n == 3.
                break;
            }
            // Drop min (v[0]) and max (v[n-1]); bring in the next sample.
            let s = src(next);
            v[0].copy_from_slice(s);
            n -= 1;
            next += 1;
        }
        // After the last pass the median is the middle element of the 3 left.
        out[y * ow..(y + 1) * ow].copy_from_slice(&v[1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::px::{pack, unpack};

    #[test]
    fn kernel_sums_to_256() {
        for s in [0.5f32, 1.0, 2.0, 2.6] {
            assert_eq!(gauss_kernel(s).iter().map(|&v| v as u32).sum::<u32>(), 256);
        }
    }

    #[test]
    fn median5_is_exact() {
        let (w, h) = (40, 20);
        let p: Vec<u8> = (0..w * h).map(|i| ((i * 7919 + 31) % 251) as u8).collect();
        let mut o = vec![0u8; (w - 4) * (h - 4)];
        median5(&p, w, h, &mut o);
        for y in 0..h - 4 {
            for x in 0..w - 4 {
                let mut v: Vec<u8> = (0..25).map(|i| p[(y + i / 5) * w + x + i % 5]).collect();
                v.sort();
                assert_eq!(o[y * (w - 4) + x], v[12], "({x},{y})");
            }
        }
    }

    #[test]
    fn conv_keeps_flat() {
        let c = pack(17, 128, 250, 255);
        let mut b = vec![c; 30 * 30];
        conv_sep(&mut b, 30, 30, &gauss_kernel(1.5));
        assert_eq!(unpack(b[15 * 30 + 15]), unpack(c));
    }
}

/// One running-sum box pass down columns, row-major (vectorizes across x).
fn box_cols(p: &mut [u8], w: usize, h: usize, r: usize, tmp: &mut [u8], sums: &mut [u32]) {
    let n = 2 * r + 1;
    let inv = ((1u64 << 24) / n as u64) as u32;
    let row = |y: isize| y.clamp(0, h as isize - 1) as usize * w;
    sums.iter_mut().for_each(|s| *s = 0);
    for y in -(r as isize)..=(r as isize) {
        let o = row(y);
        for (s, &v) in sums.iter_mut().zip(&p[o..o + w]) {
            *s += v as u32;
        }
    }
    for y in 0..h {
        let out = &mut tmp[y * w..(y + 1) * w];
        for (o, &s) in out.iter_mut().zip(sums.iter()) {
            *o = ((s * inv + (1 << 23)) >> 24) as u8;
        }
        let (a, b) = (row(y as isize + r as isize + 1), row(y as isize - r as isize));
        let (add, sub) = (&p[a..a + w], &p[b..b + w]);
        for ((s, &ad), &sb) in sums.iter_mut().zip(add).zip(sub) {
            *s = *s + ad as u32 - sb as u32;
        }
    }
    p.copy_from_slice(tmp);
}

/// Cache-blocked 8x8 transpose of a u8 plane (w x h -> h x w).
pub fn transpose_u8(p: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut o = vec![0u8; w * h];
    let (bw, bh) = (w / 8 * 8, h / 8 * 8);
    for by in (0..bh).step_by(8) {
        for bx in (0..bw).step_by(8) {
            let mut blk = [[0u8; 8]; 8];
            for (k, row) in blk.iter_mut().enumerate() {
                row.copy_from_slice(&p[(by + k) * w + bx..(by + k) * w + bx + 8]);
            }
            for j in 0..8 {
                let dst = &mut o[(bx + j) * h + by..(bx + j) * h + by + 8];
                for k in 0..8 {
                    dst[k] = blk[k][j];
                }
            }
        }
    }
    for y in 0..h {
        for x in bw..w {
            o[x * h + y] = p[y * w + x];
        }
    }
    for y in bh..h {
        for x in 0..bw {
            o[x * h + y] = p[y * w + x];
        }
    }
    o
}

/// Box passes on a single plane. Horizontal passes run as column passes on
/// the transposed plane: column passes vectorize (0.1 ns/px) while row
/// running sums are a serial chain (1.1 ns/px), and two 8x8-blocked
/// transposes cost only 0.3 ns/px.
pub fn box_plane(p: &mut [u8], w: usize, h: usize, boxes: &[usize]) {
    let active: Vec<usize> = boxes.iter().map(|b| b / 2).filter(|&r| r > 0).collect();
    if active.is_empty() {
        return;
    }
    let mut tmp = vec![0u8; w * h];
    let mut sums = vec![0u32; w.max(h)];
    for &r in &active {
        box_cols(p, w, h, r, &mut tmp, &mut sums[..w]);
    }
    let mut t = transpose_u8(p, w, h);
    for &r in &active {
        box_cols(&mut t, h, w, r, &mut tmp, &mut sums[..h]);
    }
    p.copy_from_slice(&transpose_u8(&t, h, w));
}

/// Box passes along one axis on packed pixels, done planar (vectorized
/// column passes; horizontal via transpose). Result valid away from edges.
pub fn box_axis(buf: &mut [u32], w: usize, h: usize, boxes: &[usize], horizontal: bool) {
    let active: Vec<usize> = boxes.iter().map(|b| b / 2).filter(|&r| r > 0).collect();
    if active.is_empty() {
        return;
    }
    let opaque = opaque(buf);
    let mut planes = split(buf);
    let chans = if opaque { 3 } else { 4 };
    use rayon::prelude::*;
    planes[..chans].par_iter_mut().for_each(|p| {
        let mut tmp = vec![0u8; w * h];
        let mut sums = vec![0u32; w.max(h)];
        if horizontal {
            let mut t = transpose_u8(p, w, h);
            for &r in &active {
                box_cols(&mut t, h, w, r, &mut tmp, &mut sums[..h]);
            }
            p.copy_from_slice(&transpose_u8(&t, h, w));
        } else {
            for &r in &active {
                box_cols(p, w, h, r, &mut tmp, &mut sums[..w]);
            }
        }
    });
    merge(&planes, buf);
}

/// Gaussian blur of one plane: direct taps for small sigma, boxes otherwise.
pub fn gauss_plane(p: &mut [u8], w: usize, h: usize, sigma: f32) {
    if sigma < 0.3 {
        return;
    }
    if sigma <= DIRECT_SIGMA_MAX {
        let k = gauss_kernel(sigma);
        let r = k.len() / 2;
        if w <= 2 * r || h <= 2 * r {
            return;
        }
        let ow = w - 2 * r;
        // Horizontal taps in u16, rounded back to u8 so the vertical taps can
        // also run in 16-bit lanes (twice as wide as u32).
        let mut hrow = vec![0u8; w * h];
        let mut acc = vec![0u16; ow];
        for y in 0..h {
            let src = &p[y * w..(y + 1) * w];
            acc.iter_mut().for_each(|a| *a = 0);
            for (j, &kw) in k.iter().enumerate() {
                for (a, &v) in acc.iter_mut().zip(&src[j..j + ow]) {
                    *a += v as u16 * kw;
                }
            }
            for (d, &a) in hrow[y * w + r..y * w + r + ow].iter_mut().zip(&acc) {
                *d = ((a as u32 + 128) >> 8) as u8;
            }
        }
        for y in r..h - r {
            acc.iter_mut().for_each(|a| *a = 0);
            for (j, &kw) in k.iter().enumerate() {
                let s = &hrow[(y + j - r) * w + r..(y + j - r) * w + r + ow];
                for (a, &v) in acc.iter_mut().zip(s) {
                    *a += v as u16 * kw;
                }
            }
            for (d, &a) in p[y * w + r..y * w + r + ow].iter_mut().zip(&acc) {
                *d = ((a as u32 + 128) >> 8) as u8;
            }
        }
    } else {
        box_plane(p, w, h, &super::blur::boxes_for_gauss(sigma));
    }
}

#[derive(Clone, Copy)]
pub enum Combine {
    /// o + k·(o − b) where |o − b| ≥ threshold (k in 8.8 fixed point).
    Unsharp { k: i32, threshold: i32 },
    /// o − b + a/2.
    HighPass,
    /// b.
    Blur,
}

/// Blur-and-combine on planes: one split, one merge, every step vectorized.
pub fn blur_combine(buf: &mut [u32], w: usize, h: usize, sigma: f32, boxes: Option<&[usize]>, mode: Combine) {
    let opaque = opaque(buf);
    let mut planes = split(buf);
    let chans = if opaque { 3 } else { 4 };
    let (color, alpha_plane) = planes.split_at_mut(3);
    let alpha: &[u8] = &alpha_plane[0];
    let mut alpha_out = if chans == 4 { Some(alpha.to_vec()) } else { None };
    use rayon::prelude::*;
    let mut work: Vec<(usize, &mut Vec<u8>)> = color.iter_mut().enumerate().collect();
    if let Some(a) = alpha_out.as_mut() {
        work.push((3, a));
    }
    work.into_par_iter().for_each(|(ci, p)| {
        let mut b = p.clone();
        match boxes {
            Some(bx) => box_plane(&mut b, w, h, bx),
            None => gauss_plane(&mut b, w, h, sigma),
        }
        match mode {
            Combine::Blur => *p = b,
            Combine::Unsharp { k, threshold } => {
                for ((o, &bv), &a) in p.iter_mut().zip(&b).zip(alpha) {
                    let d = *o as i32 - bv as i32;
                    let keep = d.abs() < threshold;
                    let v = *o as i32 + ((d * k + 128) >> 8);
                    let lim = if ci == 3 { 255 } else { a as i32 };
                    *o = if keep { *o } else { v.clamp(0, lim) as u8 };
                }
            }
            Combine::HighPass => {
                for ((o, &bv), &a) in p.iter_mut().zip(&b).zip(alpha) {
                    if ci == 3 {
                        continue;
                    }
                    let v = *o as i32 - bv as i32 + (a as i32 >> 1);
                    *o = v.clamp(0, a as i32) as u8;
                }
            }
        }
    });
    if let Some(a) = alpha_out {
        planes[3] = a;
    }
    merge(&planes, buf);
}

#[cfg(test)]
mod micro {
    use super::*;
    #[test]
    #[ignore]
    fn micro_box() {
        let (w, h) = (286usize, 286usize);
        let mut p: Vec<u8> = (0..w * h).map(|i| (i * 31 % 251) as u8).collect();
        let mut tmp = vec![0u8; w * h];
        let mut sums = vec![0u32; w];
        let n = 2000;
        let rows = 0.0;
        let t = std::time::Instant::now();
        for _ in 0..n {
            box_cols(&mut p, w, h, 4, &mut tmp, &mut sums);
        }
        let cols = t.elapsed().as_secs_f64() / n as f64 / (w * h) as f64 * 1e9;
        let t = std::time::Instant::now();
        for _ in 0..n {
            std::hint::black_box(transpose_u8(&p, w, h));
        }
        let tr = t.elapsed().as_secs_f64() / n as f64 / (w * h) as f64 * 1e9;
        println!("transpose {tr:.2} ns/px");
        let t = std::time::Instant::now();
        for _ in 0..n {
            let s = super::super::util::split(&vec![0x12345678u32; w * h]);
            std::hint::black_box(s);
        }
        let split = t.elapsed().as_secs_f64() / n as f64 / (w * h) as f64 * 1e9;
        println!("rows {rows:.2} ns/px  cols {cols:.2} ns/px  split {split:.2} ns/px");
    }
}
