//! Colour-space conversions for readouts (Info panel, samplers) and the
//! segmentation working space. ICC-free and exact for sRGB:
//!
//! - **Lab** is CIE L\*a\*b\* relative to **D50** (Photoshop's Lab white):
//!   sRGB is linearized with the piecewise IEC 61966-2-1 curve, converted
//!   to XYZ (D65) with the sRGB primaries matrix, chromatically adapted to
//!   D50 with the Bradford transform, then mapped to Lab with the CIE
//!   ε / κ constants.
//! - **HSB** is the hexcone model (hue in degrees, S and B in percent).
//! - **CMYK** follows Photoshop's default working CMYK, "U.S. Web Coated
//!   (SWOP) v2": a 17³ table of Photoshop's own readouts
//!   (`assets/cmyk/swop_v2_17.bin`), interpolated trilinearly. The naive
//!   device formula ([`srgb_to_cmyk_naive`]) is kept for comparison.

/// sRGB 0..255 → linear 0..1.
#[inline]
pub fn srgb_to_linear(v: u8) -> f64 {
    let c = v as f64 / 255.0;
    if c <= 0.040_45 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// sRGB → XYZ (D65), Y of white = 1.
pub fn srgb_to_xyz_d65(c: [u8; 3]) -> [f64; 3] {
    let [r, g, b] = c.map(srgb_to_linear);
    [
        0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b,
        0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b,
        0.019_333_9 * r + 0.119_192_0 * g + 0.950_304_1 * b,
    ]
}

/// Bradford chromatic adaptation D65 → D50.
pub fn d65_to_d50(x: [f64; 3]) -> [f64; 3] {
    const M: [[f64; 3]; 3] = [[1.047_811_2, 0.022_886_6, -0.050_127_0], [0.029_542_4, 0.990_484_5, -0.017_049_1], [-0.009_234_5, 0.015_043_6, 0.752_131_6]];
    [0, 1, 2].map(|i| M[i][0] * x[0] + M[i][1] * x[1] + M[i][2] * x[2])
}

/// D50 reference white.
pub const D50: [f64; 3] = [0.964_22, 1.0, 0.825_21];

/// XYZ (relative to `white`) → L\*a\*b\*.
pub fn xyz_to_lab(x: [f64; 3], white: [f64; 3]) -> [f64; 3] {
    const E: f64 = 216.0 / 24389.0;
    const K: f64 = 24389.0 / 27.0;
    let f = |t: f64| if t > E { t.cbrt() } else { (K * t + 16.0) / 116.0 };
    let (fx, fy, fz) = (f(x[0] / white[0]), f(x[1] / white[1]), f(x[2] / white[2]));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// sRGB → Lab (D50), as Photoshop's Lab readout.
pub fn srgb_to_lab(c: [u8; 3]) -> [f64; 3] {
    xyz_to_lab(d65_to_d50(srgb_to_xyz_d65(c)), D50)
}

/// Fast f32 Lab for segmentation (same maths, LUT linearization).
pub fn srgb8_to_lab(c: [u8; 3]) -> [f32; 3] {
    static LIN: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    let lin = LIN.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as u8) as f32));
    let [r, g, b] = c.map(|v| lin[v as usize]);
    let x = 0.436_074 * r + 0.385_064 * g + 0.143_080 * b;
    let y = 0.222_504 * r + 0.716_878 * g + 0.060_617 * b;
    let z = 0.013_932 * r + 0.097_103 * g + 0.714_173 * b;
    let f = |t: f32| if t > 216.0 / 24389.0 { t.cbrt() } else { (24389.0 / 27.0 * t + 16.0) / 116.0 };
    let (fx, fy, fz) = (f(x / 0.964_22), f(y), f(z / 0.825_21));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// sRGB → HSB (hue 0..360°, saturation and brightness 0..100 %).
pub fn srgb_to_hsb(c: [u8; 3]) -> [f64; 3] {
    let [r, g, b] = c.map(|v| v as f64 / 255.0);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / d).rem_euclid(6.0))
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    [h, if max == 0.0 { 0.0 } else { d / max * 100.0 }, max * 100.0]
}

/// sRGB → SWOP v2 CMYK in percent, as Photoshop's Info panel reads it.
pub fn srgb_to_cmyk(c: [u8; 3]) -> [f64; 4] {
    static LUT: &[u8] = include_bytes!("../../assets/cmyk/swop_v2_17.bin");
    // Nodes at 0, 16, …, 240, 255.
    let cell = |v: u8| -> (usize, f64) {
        if v >= 240 {
            (15, (v - 240) as f64 / 15.0)
        } else {
            ((v / 16) as usize, (v % 16) as f64 / 16.0)
        }
    };
    let ((ri, rf), (gi, gf), (bi, bf)) = (cell(c[0]), cell(c[1]), cell(c[2]));
    let at = |r: usize, g: usize, b: usize, ch: usize| {
        let o = (((r * 17 + g) * 17 + b) * 4 + ch) * 2;
        u16::from_le_bytes([LUT[o], LUT[o + 1]]) as f64 / 65535.0 * 100.0
    };
    std::array::from_fn(|ch| {
        let mut v = 0.0;
        for (dr, wr) in [(0, 1.0 - rf), (1, rf)] {
            for (dg, wg) in [(0, 1.0 - gf), (1, gf)] {
                for (db, wb) in [(0, 1.0 - bf), (1, bf)] {
                    let w = wr * wg * wb;
                    if w > 0.0 {
                        v += w * at(ri + dr, gi + dg, bi + db, ch);
                    }
                }
            }
        }
        v
    })
}

/// sRGB → naive device CMYK in percent: `K = 1 − max(R, G, B)` and no
/// profile, ink limit or dot gain.
pub fn srgb_to_cmyk_naive(c: [u8; 3]) -> [f64; 4] {
    let [r, g, b] = c.map(|v| v as f64 / 255.0);
    let k = 1.0 - r.max(g).max(b);
    if k >= 1.0 {
        return [0.0, 0.0, 0.0, 100.0];
    }
    let f = |v: f64| (1.0 - v - k) / (1.0 - k) * 100.0;
    [f(r), f(g), f(b), k * 100.0]
}

/// An Info panel readout mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    Rgb,
    Hsb,
    Lab,
    Cmyk,
}

impl ColorMode {
    pub const ALL: [ColorMode; 4] = [ColorMode::Rgb, ColorMode::Hsb, ColorMode::Lab, ColorMode::Cmyk];

    pub fn label(self) -> &'static str {
        match self {
            ColorMode::Rgb => "RGB",
            ColorMode::Hsb => "HSB",
            ColorMode::Lab => "Lab",
            ColorMode::Cmyk => "CMYK",
        }
    }

    pub fn next(self) -> ColorMode {
        Self::ALL[(Self::ALL.iter().position(|&m| m == self).unwrap_or(0) + 1) % Self::ALL.len()]
    }

    /// The readout text for an sRGB colour.
    pub fn format(self, c: [u8; 3]) -> String {
        // Round first, then add 0.0 so tiny negatives don't print as "-0".
        let r = |v: f64| v.round() + 0.0;
        match self {
            ColorMode::Rgb => format!("R {} G {} B {}", c[0], c[1], c[2]),
            ColorMode::Hsb => {
                let [h, s, b] = srgb_to_hsb(c).map(r);
                format!("H {h:.0}° S {s:.0}% B {b:.0}%")
            }
            ColorMode::Lab => {
                let [l, a, b] = srgb_to_lab(c).map(r);
                format!("L {l:.0} a {a:.0} b {b:.0}")
            }
            ColorMode::Cmyk => {
                let [c_, m, y, k] = srgb_to_cmyk(c).map(r);
                format!("C {c_:.0} M {m:.0} Y {y:.0} K {k:.0}%")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
    }

    #[test]
    fn lab_d50_reference_values() {
        // Bruce Lindbloom's calculator (sRGB, D50 reference white, Bradford).
        assert!(close(srgb_to_lab([255, 255, 255]), [100.0, 0.0, 0.0], 0.01));
        assert!(close(srgb_to_lab([0, 0, 0]), [0.0, 0.0, 0.0], 0.01));
        assert!(close(srgb_to_lab([255, 0, 0]), [54.29, 80.80, 69.89], 0.05), "{:?}", srgb_to_lab([255, 0, 0]));
        assert!(close(srgb_to_lab([0, 255, 0]), [87.82, -79.29, 80.99], 0.05), "{:?}", srgb_to_lab([0, 255, 0]));
        assert!(close(srgb_to_lab([0, 0, 255]), [29.57, 68.30, -112.03], 0.05), "{:?}", srgb_to_lab([0, 0, 255]));
        // Neutral grey stays neutral; L of 50 % grey.
        let g = srgb_to_lab([128, 128, 128]);
        assert!((g[0] - 53.59).abs() < 0.02 && g[1].abs() < 0.01 && g[2].abs() < 0.01, "{g:?}");
        // The fast f32 path agrees.
        for c in [[255u8, 0, 0], [12, 200, 90], [128, 128, 128], [3, 7, 250]] {
            let (a, b) = (srgb_to_lab(c), srgb8_to_lab(c));
            assert!((0..3).all(|i| (a[i] - b[i] as f64).abs() < 0.05), "{c:?}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn hsb_and_cmyk() {
        assert!(close(srgb_to_hsb([255, 0, 0]), [0.0, 100.0, 100.0], 1e-9));
        assert!(close(srgb_to_hsb([0, 128, 255]), [209.88, 100.0, 100.0], 0.01));
        assert!(close(srgb_to_hsb([128, 128, 128]), [0.0, 0.0, 50.196], 0.01));
        let k = srgb_to_cmyk_naive([255, 0, 0]);
        assert_eq!(k.map(|v| v.round()), [0.0, 100.0, 100.0, 0.0]);
        let k = srgb_to_cmyk_naive([0, 0, 0]);
        assert_eq!(k, [0.0, 0.0, 0.0, 100.0]);
        assert_eq!(ColorMode::Lab.format([255, 255, 255]), "L 100 a 0 b 0");
        assert_eq!(ColorMode::Rgb.next(), ColorMode::Hsb);
        // SWOP v2 (Photoshop's readouts): rich black, and white is paper.
        let k = srgb_to_cmyk([0, 0, 0]);
        assert!((k[0] - 74.97).abs() < 0.01 && (k[3] - 90.15).abs() < 0.01, "{k:?}");
        assert!(srgb_to_cmyk([255, 255, 255]).iter().all(|&v| v < 0.01));
        // Between nodes it interpolates (monotone in K along a grey ramp).
        let ks: Vec<f64> = (0..=255).step_by(5).map(|v| srgb_to_cmyk([v as u8; 3])[3]).collect();
        assert!(ks.windows(2).all(|w| w[1] <= w[0] + 1e-9));
        let k = srgb_to_cmyk_naive([128, 64, 32]);
        assert!((k[3] - 49.8).abs() < 0.1 && (k[0]).abs() < 1e-9 && (k[1] - 50.0).abs() < 0.1, "{k:?}");
    }
}
