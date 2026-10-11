//! Image-quality metrics for comparisons against Photoshop and ground
//! truth: SSIM, mean CIEDE2000 colour difference and mask IoU.

use rayon::prelude::*;

use super::color::srgb_to_lab;

/// Rec. 601 luma of straight RGB8.
fn luma(c: [u8; 3]) -> f32 {
    0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32
}

/// Mean SSIM (Wang et al. 2004) of the luma of two `w x h` straight-RGB
/// images: 11×11 Gaussian window (σ = 1.5), K1 = 0.01, K2 = 0.03, L = 255.
/// With `mask`, the mean is taken over pixels where it is non-zero.
pub fn ssim(a: &[[u8; 3]], b: &[[u8; 3]], w: usize, h: usize, mask: Option<&[u8]>) -> f64 {
    let (x, y): (Vec<f32>, Vec<f32>) = (a.iter().map(|&c| luma(c)).collect(), b.iter().map(|&c| luma(c)).collect());
    let g: Vec<f32> = {
        let k: Vec<f32> = (-5..=5).map(|i| (-(i * i) as f32 / (2.0 * 1.5 * 1.5)).exp()).collect();
        let s: f32 = k.iter().sum();
        k.iter().map(|v| v / s).collect()
    };
    // Separable Gaussian with clamped borders.
    let blur = |p: &[f32]| -> Vec<f32> {
        let mut t = vec![0f32; w * h];
        t.par_chunks_mut(w).enumerate().for_each(|(yy, row)| {
            for (xx, o) in row.iter_mut().enumerate() {
                *o = (0..11).map(|i| g[i] * p[yy * w + (xx as i64 + i as i64 - 5).clamp(0, w as i64 - 1) as usize]).sum();
            }
        });
        let mut out = vec![0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(yy, row)| {
            for (xx, o) in row.iter_mut().enumerate() {
                *o = (0..11).map(|i| g[i] * t[(yy as i64 + i as i64 - 5).clamp(0, h as i64 - 1) as usize * w + xx]).sum();
            }
        });
        out
    };
    let xx: Vec<f32> = x.iter().map(|v| v * v).collect();
    let yy: Vec<f32> = y.iter().map(|v| v * v).collect();
    let xy: Vec<f32> = x.iter().zip(&y).map(|(p, q)| p * q).collect();
    let (mx, my, sxx, syy, sxy) = (blur(&x), blur(&y), blur(&xx), blur(&yy), blur(&xy));
    let (c1, c2) = ((0.01f32 * 255.0).powi(2), (0.03f32 * 255.0).powi(2));
    let (mut sum, mut n) = (0f64, 0f64);
    for i in 0..w * h {
        if mask.is_some_and(|m| m[i] == 0) {
            continue;
        }
        let (vx, vy, cov) = (sxx[i] - mx[i] * mx[i], syy[i] - my[i] * my[i], sxy[i] - mx[i] * my[i]);
        let v = ((2.0 * mx[i] * my[i] + c1) * (2.0 * cov + c2)) / ((mx[i] * mx[i] + my[i] * my[i] + c1) * (vx + vy + c2));
        sum += v as f64;
        n += 1.0;
    }
    if n == 0.0 { 1.0 } else { sum / n }
}

/// CIEDE2000 colour difference between two L\*a\*b\* colours (kL = kC = kH = 1).
pub fn delta_e2000(l1: [f64; 3], l2: [f64; 3]) -> f64 {
    use std::f64::consts::PI;
    let (lp1, a1, b1) = (l1[0], l1[1], l1[2]);
    let (lp2, a2, b2) = (l2[0], l2[1], l2[2]);
    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let cb = (c1 + c2) / 2.0;
    let g = 0.5 * (1.0 - (cb.powi(7) / (cb.powi(7) + 25f64.powi(7))).sqrt());
    let (ap1, ap2) = ((1.0 + g) * a1, (1.0 + g) * a2);
    let (cp1, cp2) = ((ap1 * ap1 + b1 * b1).sqrt(), (ap2 * ap2 + b2 * b2).sqrt());
    let hue = |b: f64, a: f64| {
        if a == 0.0 && b == 0.0 {
            0.0
        } else {
            let h = b.atan2(a).to_degrees();
            if h < 0.0 { h + 360.0 } else { h }
        }
    };
    let (hp1, hp2) = (hue(b1, ap1), hue(b2, ap2));
    let dl = lp2 - lp1;
    let dc = cp2 - cp1;
    let dhp = if cp1 * cp2 == 0.0 {
        0.0
    } else if (hp2 - hp1).abs() <= 180.0 {
        hp2 - hp1
    } else if hp2 - hp1 > 180.0 {
        hp2 - hp1 - 360.0
    } else {
        hp2 - hp1 + 360.0
    };
    let dh = 2.0 * (cp1 * cp2).sqrt() * (dhp.to_radians() / 2.0).sin();
    let lb = (lp1 + lp2) / 2.0;
    let cbp = (cp1 + cp2) / 2.0;
    let hb = if cp1 * cp2 == 0.0 {
        hp1 + hp2
    } else if (hp1 - hp2).abs() <= 180.0 {
        (hp1 + hp2) / 2.0
    } else if hp1 + hp2 < 360.0 {
        (hp1 + hp2 + 360.0) / 2.0
    } else {
        (hp1 + hp2 - 360.0) / 2.0
    };
    let t = 1.0 - 0.17 * ((hb - 30.0) * PI / 180.0).cos() + 0.24 * ((2.0 * hb) * PI / 180.0).cos() + 0.32 * ((3.0 * hb + 6.0) * PI / 180.0).cos()
        - 0.20 * ((4.0 * hb - 63.0) * PI / 180.0).cos();
    let dth = 30.0 * (-((hb - 275.0) / 25.0).powi(2)).exp();
    let rc = 2.0 * (cbp.powi(7) / (cbp.powi(7) + 25f64.powi(7))).sqrt();
    let sl = 1.0 + 0.015 * (lb - 50.0).powi(2) / (20.0 + (lb - 50.0).powi(2)).sqrt();
    let sc = 1.0 + 0.045 * cbp;
    let sh = 1.0 + 0.015 * cbp * t;
    let rt = -(2.0 * dth * PI / 180.0).sin() * rc;
    ((dl / sl).powi(2) + (dc / sc).powi(2) + (dh / sh).powi(2) + rt * (dc / sc) * (dh / sh)).sqrt()
}

/// Mean CIEDE2000 between two straight-RGB images (optionally masked).
pub fn mean_delta_e(a: &[[u8; 3]], b: &[[u8; 3]], mask: Option<&[u8]>) -> f64 {
    let (s, n) = a
        .par_iter()
        .zip(b.par_iter())
        .enumerate()
        .filter(|(i, _)| mask.is_none_or(|m| m[*i] != 0))
        .map(|(_, (p, q))| (delta_e2000(srgb_to_lab(*p), srgb_to_lab(*q)), 1.0))
        .reduce(|| (0.0, 0.0), |x, y| (x.0 + y.0, x.1 + y.1));
    if n == 0.0 { 0.0 } else { s / n }
}

/// Intersection over union of two masks thresholded at 128.
pub fn iou(a: &[u8], b: &[u8]) -> f64 {
    let (i, u) = a.par_iter().zip(b.par_iter()).map(|(&p, &q)| ((p >= 128 && q >= 128) as u64, (p >= 128 || q >= 128) as u64)).reduce(|| (0, 0), |x, y| (x.0 + y.0, x.1 + y.1));
    if u == 0 { 1.0 } else { i as f64 / u as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ciede2000_reference_pairs() {
        // Sharma, Wu & Dalal (2005), test data pairs 1, 7, 13, 19, 25, 31.
        let cases = [
            ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0425),
            ([50.0, 0.0, 0.0], [50.0, -1.0, 2.0], 2.3669),
            ([50.0, 2.49, -0.001], [50.0, -2.49, 0.0011], 7.2195),
            ([50.0, 2.5, 0.0], [50.0, 0.0, -2.5], 4.3065),
            ([60.2574, -34.0099, 36.2677], [60.4626, -34.1751, 39.4387], 1.2644),
            ([22.7233, 20.0904, -46.694], [23.0331, 14.973, -42.5619], 2.0373),
        ];
        for (a, b, want) in cases {
            let got = delta_e2000(a, b);
            assert!((got - want).abs() < 1e-3, "{a:?} {b:?}: {got} vs {want}");
        }
    }

    #[test]
    fn ssim_and_iou_basics() {
        let (w, h) = (64usize, 48usize);
        let img: Vec<[u8; 3]> = (0..w * h).map(|i| [((i * 7) % 255) as u8, ((i * 3) % 200) as u8, 90]).collect();
        assert!((ssim(&img, &img, w, h, None) - 1.0).abs() < 1e-6);
        let noisy: Vec<[u8; 3]> = img.iter().enumerate().map(|(i, c)| c.map(|v| v.saturating_add(((i * 2654435761) % 40) as u8))).collect();
        let s = ssim(&img, &noisy, w, h, None);
        assert!(s < 0.95 && s > 0.0, "{s}");
        assert_eq!(mean_delta_e(&img, &img, None), 0.0);
        let a = vec![255u8, 255, 0, 0];
        let b = vec![255u8, 0, 255, 0];
        assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-9);
    }
}
