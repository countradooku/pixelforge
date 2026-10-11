//! Layer blend modes (W3C compositing formulas on premultiplied pixels).

use super::px::{self, alpha, scale};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[repr(u8)]
pub enum BlendMode {
    #[default]
    Normal,
    Darken,
    Multiply,
    ColorBurn,
    LinearBurn,
    Lighten,
    Screen,
    ColorDodge,
    LinearDodge,
    Overlay,
    SoftLight,
    HardLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
    // Append only: these discriminants are stored in migration .pforge files.
    Dissolve,
    DarkerColor,
    LighterColor,
    VividLight,
    LinearLight,
    PinLight,
    HardMix,
    Subtract,
    Divide,
}

impl BlendMode {
    pub const ALL: [BlendMode; 27] = [
        BlendMode::Normal,
        BlendMode::Darken,
        BlendMode::Multiply,
        BlendMode::ColorBurn,
        BlendMode::LinearBurn,
        BlendMode::Lighten,
        BlendMode::Screen,
        BlendMode::ColorDodge,
        BlendMode::LinearDodge,
        BlendMode::Overlay,
        BlendMode::SoftLight,
        BlendMode::HardLight,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
        BlendMode::Dissolve,
        BlendMode::DarkerColor,
        BlendMode::LighterColor,
        BlendMode::VividLight,
        BlendMode::LinearLight,
        BlendMode::PinLight,
        BlendMode::HardMix,
        BlendMode::Subtract,
        BlendMode::Divide,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Darken => "Darken",
            BlendMode::Multiply => "Multiply",
            BlendMode::ColorBurn => "Color Burn",
            BlendMode::LinearBurn => "Linear Burn",
            BlendMode::Lighten => "Lighten",
            BlendMode::Screen => "Screen",
            BlendMode::ColorDodge => "Color Dodge",
            BlendMode::LinearDodge => "Linear Dodge (Add)",
            BlendMode::Overlay => "Overlay",
            BlendMode::SoftLight => "Soft Light",
            BlendMode::HardLight => "Hard Light",
            BlendMode::Difference => "Difference",
            BlendMode::Exclusion => "Exclusion",
            BlendMode::Hue => "Hue",
            BlendMode::Saturation => "Saturation",
            BlendMode::Color => "Color",
            BlendMode::Luminosity => "Luminosity",
            BlendMode::Dissolve => "Dissolve",
            BlendMode::DarkerColor => "Darker Color",
            BlendMode::LighterColor => "Lighter Color",
            BlendMode::VividLight => "Vivid Light",
            BlendMode::LinearLight => "Linear Light",
            BlendMode::PinLight => "Pin Light",
            BlendMode::HardMix => "Hard Mix",
            BlendMode::Subtract => "Subtract",
            BlendMode::Divide => "Divide",
        }
    }

    pub fn from_u8(v: u8) -> BlendMode {
        Self::ALL.get(v as usize).copied().unwrap_or_default()
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

#[inline(always)]
fn hard_light(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 {
        cb * 2.0 * cs
    } else {
        let s = 2.0 * cs - 1.0;
        cb + s - cb * s
    }
}

#[inline(always)]
fn separable(mode: BlendMode, cb: f32, cs: f32) -> f32 {
    match mode {
        BlendMode::Normal => cs,
        BlendMode::Multiply => cb * cs,
        BlendMode::Screen => cb + cs - cb * cs,
        BlendMode::Overlay => hard_light(cs, cb),
        BlendMode::Darken => cb.min(cs),
        BlendMode::Lighten => cb.max(cs),
        BlendMode::ColorDodge => {
            if cb <= 0.0 {
                0.0
            } else if cs >= 1.0 {
                1.0
            } else {
                (cb / (1.0 - cs)).min(1.0)
            }
        }
        BlendMode::ColorBurn => {
            if cb >= 1.0 {
                1.0
            } else if cs <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - cb) / cs).min(1.0)
            }
        }
        BlendMode::HardLight => hard_light(cb, cs),
        BlendMode::SoftLight => {
            if cs <= 0.5 {
                cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
            } else {
                let d = if cb <= 0.25 {
                    ((16.0 * cb - 12.0) * cb + 4.0) * cb
                } else {
                    cb.sqrt()
                };
                cb + (2.0 * cs - 1.0) * (d - cb)
            }
        }
        BlendMode::Difference => (cb - cs).abs(),
        BlendMode::Exclusion => cb + cs - 2.0 * cb * cs,
        BlendMode::LinearDodge => (cb + cs).min(1.0),
        BlendMode::LinearBurn => (cb + cs - 1.0).max(0.0),
        _ => cs,
    }
}

#[inline(always)]
fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn clip_color(c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut o = c;
    if n < 0.0 {
        let d = l - n;
        for v in &mut o {
            *v = l + (*v - l) * l / d.max(1e-6);
        }
    }
    if x > 1.0 {
        let d = x - l;
        for v in &mut o {
            *v = l + (*v - l) * (1.0 - l) / d.max(1e-6);
        }
    }
    o
}

fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_color([c[0] + d, c[1] + d, c[2] + d])
}

fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    let mx = c[0].max(c[1]).max(c[2]);
    let mn = c[0].min(c[1]).min(c[2]);
    if mx <= mn {
        return [0.0; 3];
    }
    let mut o = [0.0; 3];
    for i in 0..3 {
        o[i] = (c[i] - mn) * s / (mx - mn);
    }
    o
}

#[inline]
fn blend_color(mode: BlendMode, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    match mode {
        BlendMode::Hue => set_lum(set_sat(cs, sat(cb)), lum(cb)),
        BlendMode::Saturation => set_lum(set_sat(cb, sat(cs)), lum(cb)),
        BlendMode::Color => set_lum(cs, lum(cb)),
        BlendMode::Luminosity => set_lum(cb, lum(cs)),
        m => [
            separable(m, cb[0], cs[0]),
            separable(m, cb[1], cs[1]),
            separable(m, cb[2], cs[2]),
        ],
    }
}

/// Composites one row of `src` onto `dst` with layer opacity (`0..=255`).
pub fn blend_row(dst: &mut [u32], src: &[u32], opacity: u32, mode: BlendMode) {
    if opacity == 0 {
        return;
    }
    if mode == BlendMode::Dissolve {
        for (x, (d, &s)) in dst.iter_mut().zip(src).enumerate() {
            *d = dissolve_px(*d, s, opacity, x as i32, 0);
        }
        return;
    }
    if mode == BlendMode::Normal {
        if opacity == 255 {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = px::over(s, *d);
            }
        } else {
            for (d, &s) in dst.iter_mut().zip(src) {
                if s != 0 {
                    *d = px::over(scale(s, opacity), *d);
                }
            }
        }
        return;
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        if s == 0 {
            continue;
        }
        let s = if opacity == 255 { s } else { scale(s, opacity) };
        let db = *d;
        if alpha(db) == 0 {
            *d = s;
            continue;
        }
        *d = blend_px(db, s, mode);
    }
}

#[inline]
pub fn blend_px(db: u32, s: u32, mode: BlendMode) -> u32 {
    if mode.index() >= BlendMode::Dissolve.index() {
        return if mode == BlendMode::Dissolve {
            dissolve_px(db, s, 255, 0, 0)
        } else {
            blend_integer(db, s, mode)
        };
    }
    let (cs, sa) = px::to_f32(s);
    let (cb, da) = px::to_f32(db);
    if sa <= 0.0 {
        return db;
    }
    let b = blend_color(mode, cb, cs);
    let ao = sa + da - sa * da;
    let mut co = [0.0f32; 3];
    for i in 0..3 {
        // Premultiplied result, then back to straight for packing.
        let pre = cs[i] * sa * (1.0 - da) + cb[i] * da * (1.0 - sa) + sa * da * b[i];
        co[i] = if ao > 0.0 { pre / ao } else { 0.0 };
    }
    px::from_f32(co, ao)
}

/// Blends a region with a stable document-coordinate stochastic pattern.
/// Existing modes retain the same row implementation and arithmetic.
pub fn blend_region(
    dst: &mut [u32],
    src: &[u32],
    opacity: u32,
    mode: BlendMode,
    width: usize,
    origin: (i32, i32),
) {
    if mode != BlendMode::Dissolve {
        blend_row(dst, src, opacity, mode);
        return;
    }
    if width == 0 || opacity == 0 {
        return;
    }
    for (i, (d, &s)) in dst.iter_mut().zip(src).enumerate() {
        *d = dissolve_px(
            *d,
            s,
            opacity,
            origin.0.wrapping_add((i % width) as i32),
            origin.1.wrapping_add((i / width) as i32),
        );
    }
}

/// PixelForge's stable dissolve pattern. Adobe does not publish Photoshop's
/// sequence, so this is deliberately not advertised as bit-exact with it.
#[inline]
pub fn dissolve_px(destination: u32, source: u32, opacity: u32, x: i32, y: i32) -> u32 {
    let source_alpha = alpha(source);
    let coverage = (source_alpha * opacity.min(255) + 127) / 255;
    if coverage == 0 {
        return destination;
    }
    let mut noise =
        (x as u32).wrapping_mul(0x1f12_3bb5) ^ (y as u32).wrapping_mul(0x5f35_6495) ^ 0xa511_e9b3;
    noise = (noise ^ (noise >> 16)).wrapping_mul(0x7feb_352d);
    noise = (noise ^ (noise >> 15)).wrapping_mul(0x846c_a68b);
    noise ^= noise >> 16;
    if coverage < 255 && noise % 255 >= coverage {
        return destination;
    }
    let c = straight_integer(source);
    px::pack(c[0] as u8, c[1] as u8, c[2] as u8, 255)
}

fn straight_integer(pixel: u32) -> [u32; 3] {
    let c = px::unpack(pixel);
    let a = u32::from(c[3]);
    if a == 0 {
        return [0; 3];
    }
    [0, 1, 2].map(|i| ((u32::from(c[i]) * 255 + a / 2) / a).min(255))
}

fn integer_channel(mode: BlendMode, base: u32, source: u32) -> u32 {
    match mode {
        BlendMode::VividLight => if source < 128 {
            if base == 255 {
                255
            } else if source == 0 {
                0
            } else {
                255 - (((255 - base) * 255 + source) / (2 * source)).min(255)
            }
        } else if base == 0 {
            0
        } else if source == 255 {
            255
        } else {
            (base * 255 + (255 - source)) / (2 * (255 - source)).max(1)
        }
        .min(255),
        BlendMode::LinearLight => (base as i32 + 2 * source as i32 - 255).clamp(0, 255) as u32,
        BlendMode::PinLight => {
            if source < 128 {
                base.min(2 * source)
            } else {
                base.max(2 * source - 255)
            }
        }
        BlendMode::HardMix => {
            if base + source >= 255 {
                255
            } else {
                0
            }
        }
        BlendMode::Subtract => base.saturating_sub(source),
        BlendMode::Divide => {
            if source == 0 {
                255
            } else {
                ((base * 255 + source / 2) / source).min(255)
            }
        }
        _ => source,
    }
}

fn blend_integer(destination: u32, source: u32, mode: BlendMode) -> u32 {
    let sa = alpha(source);
    let da = alpha(destination);
    if sa == 0 {
        return destination;
    }
    if da == 0 {
        return source;
    }
    let cs = straight_integer(source);
    let cb = straight_integer(destination);
    let blend = match mode {
        BlendMode::DarkerColor => {
            if cs.iter().sum::<u32>() < cb.iter().sum::<u32>() {
                cs
            } else {
                cb
            }
        }
        BlendMode::LighterColor => {
            if cs.iter().sum::<u32>() > cb.iter().sum::<u32>() {
                cs
            } else {
                cb
            }
        }
        _ => [0, 1, 2].map(|i| integer_channel(mode, cb[i], cs[i])),
    };
    let s = px::unpack(source);
    let d = px::unpack(destination);
    let alpha = (sa * 255 + da * 255 - sa * da + 127) / 255;
    let output = [0, 1, 2].map(|i| {
        ((u32::from(s[i]) * (255 - da) * 255
            + u32::from(d[i]) * (255 - sa) * 255
            + blend[i] * sa * da
            + 32512)
            / 65025)
            .min(alpha) as u8
    });
    px::pack(output[0], output[1], output[2], alpha as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::px::pack;

    #[test]
    fn multiply_white_is_identity() {
        let base = pack(120, 60, 30, 255);
        let mut row = [base];
        blend_row(
            &mut row,
            &[pack(255, 255, 255, 255)],
            255,
            BlendMode::Multiply,
        );
        let [r, g, b, a] = px::unpack(row[0]);
        assert_eq!(a, 255);
        assert!(
            (r as i32 - 120).abs() <= 1 && (g as i32 - 60).abs() <= 1 && (b as i32 - 30).abs() <= 1
        );
    }

    #[test]
    fn screen_black_is_identity() {
        let base = pack(10, 200, 90, 255);
        let mut row = [base];
        blend_row(&mut row, &[pack(0, 0, 0, 255)], 255, BlendMode::Screen);
        assert_eq!(row[0], base);
    }

    #[test]
    fn migration_blend_indices_are_stable_and_all_modes_are_distinct() {
        assert_eq!(BlendMode::Normal.index(), 0);
        assert_eq!(BlendMode::Luminosity.index(), 17);
        assert_eq!(BlendMode::Dissolve.index(), 18);
        assert_eq!(BlendMode::Divide.index(), 26);
        for (index, mode) in BlendMode::ALL.into_iter().enumerate() {
            assert_eq!(BlendMode::from_u8(index as u8), mode);
        }
        let mut names: Vec<_> = BlendMode::ALL.into_iter().map(BlendMode::name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 27);
    }

    #[test]
    fn integer_blend_modes_match_independent_opaque_channel_examples() {
        for (mode, base, source, want) in [
            (BlendMode::Subtract, 200, 80, 120),
            (BlendMode::Subtract, 20, 80, 0),
            (BlendMode::Divide, 64, 128, 128),
            (BlendMode::Divide, 0, 0, 255),
            (BlendMode::Divide, 100, 255, 100),
            (BlendMode::HardMix, 127, 128, 255),
            (BlendMode::HardMix, 127, 127, 0),
            (BlendMode::LinearLight, 100, 50, 0),
            (BlendMode::LinearLight, 100, 200, 245),
            (BlendMode::PinLight, 100, 25, 50),
            (BlendMode::PinLight, 100, 200, 145),
            (BlendMode::VividLight, 128, 0, 0),
            (BlendMode::VividLight, 128, 255, 255),
            (BlendMode::VividLight, 128, 64, 2),
            (BlendMode::VividLight, 64, 192, 130),
        ] {
            assert_eq!(
                blend_px(
                    pack(base, base, base, 255),
                    pack(source, source, source, 255),
                    mode
                ),
                pack(want, want, want, 255),
                "{mode:?} {base}/{source}"
            );
        }
    }

    #[test]
    fn darker_and_lighter_color_choose_whole_rgb_by_channel_sum() {
        let base = pack(240, 20, 20, 255);
        let source = pack(0, 180, 80, 255);
        assert_eq!(blend_px(base, source, BlendMode::DarkerColor), source);
        assert_eq!(blend_px(base, source, BlendMode::LighterColor), base);
        // Distinguishes RGB total from luminance and componentwise min/max.
        let equal = pack(0, 200, 80, 255);
        assert_eq!(blend_px(base, equal, BlendMode::DarkerColor), base);
    }

    #[test]
    fn new_modes_preserve_alpha_and_premultiplication_over_domain() {
        for mode in &BlendMode::ALL[19..] {
            for sa in (0..=255u8).step_by(17) {
                for da in (0..=255u8).step_by(15) {
                    let source = px::premultiply(250, 125, 5, sa);
                    let destination = px::premultiply(17, 140, 230, da);
                    let out = blend_px(destination, source, *mode);
                    let [r, g, b, a] = px::unpack(out);
                    assert!(r <= a && g <= a && b <= a, "{mode:?} {sa}/{da}");
                    assert_eq!(
                        a as u32,
                        (u32::from(sa) * 255 + u32::from(da) * 255 - u32::from(sa) * u32::from(da)
                            + 127)
                            / 255
                    );
                }
            }
        }
    }

    #[test]
    fn dissolve_is_region_partition_invariant_and_has_opaque_samples() {
        let width = 529;
        let height = 7;
        let source = vec![px::premultiply(80, 130, 240, 170); width * height];
        let mut whole = vec![0; source.len()];
        blend_region(
            &mut whole,
            &source,
            190,
            BlendMode::Dissolve,
            width,
            (-101, 37),
        );
        let mut partitioned = vec![0; source.len()];
        for y in 0..height {
            for (start, end) in [(0, 127), (127, 256), (256, width)] {
                let offset = y * width + start;
                let length = end - start;
                blend_region(
                    &mut partitioned[offset..offset + length],
                    &source[offset..offset + length],
                    190,
                    BlendMode::Dissolve,
                    length,
                    (-101 + start as i32, 37 + y as i32),
                );
            }
        }
        assert_eq!(whole, partitioned);
        assert!(whole.iter().all(|&p| alpha(p) == 0 || alpha(p) == 255));
        let count = whole.iter().filter(|&&p| p != 0).count();
        assert!((whole.len() * 40 / 100..whole.len() * 60 / 100).contains(&count));
    }

    #[test]
    fn dissolve_zero_and_full_coverage_are_exact() {
        let destination = pack(22, 33, 44, 255);
        let source = pack(200, 100, 40, 255);
        for x in -100..100 {
            assert_eq!(dissolve_px(destination, source, 0, x, 3), destination);
            assert_eq!(dissolve_px(destination, source, 255, x, 3), source);
        }
    }
}
