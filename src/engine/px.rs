//! Packed premultiplied RGBA8 pixel math.
//!
//! A pixel is a `u32` laid out little-endian as `r | g << 8 | b << 16 | a << 24`,
//! so a `&[u32]` reinterpreted as bytes is plain RGBA. Every color channel is
//! premultiplied (`c <= a`), which makes "over" a single multiply-add and lets
//! the mip pyramid average pixels without fringing.
//!
//! The hot helpers work two channels at a time in 16-bit lanes (SWAR), so a
//! full 4-channel scale costs two integer multiplies.

pub const LO: u32 = 0x00FF_00FF;

#[inline(always)]
pub const fn pack(r: u8, g: u8, b: u8, a: u8) -> u32 {
    r as u32 | (g as u32) << 8 | (b as u32) << 16 | (a as u32) << 24
}

#[inline(always)]
pub const fn unpack(p: u32) -> [u8; 4] {
    p.to_le_bytes()
}

#[inline(always)]
pub const fn alpha(p: u32) -> u32 {
    p >> 24
}

/// Exact-rounding `x * f / 255` for one 16-bit lane pair.
#[inline(always)]
const fn div255_lanes(v: u32) -> u32 {
    ((v + ((v >> 8) & LO)) >> 8) & LO
}

/// Multiplies all four channels by `f / 255` (`f` in `0..=255`).
#[inline(always)]
pub const fn scale(p: u32, f: u32) -> u32 {
    let lo = div255_lanes((p & LO) * f + 0x0080_0080);
    let hi = div255_lanes(((p >> 8) & LO) * f + 0x0080_0080);
    lo | (hi << 8)
}

/// Linear interpolation `a * (1 - t) + b * t` with a single rounding step.
#[inline(always)]
pub const fn lerp(a: u32, b: u32, t: u32) -> u32 {
    let it = 255 - t;
    let lo = div255_lanes((a & LO) * it + (b & LO) * t + 0x0080_0080);
    let hi = div255_lanes(((a >> 8) & LO) * it + ((b >> 8) & LO) * t + 0x0080_0080);
    lo | (hi << 8)
}

/// Porter-Duff source-over for premultiplied pixels.
#[inline(always)]
pub const fn over(src: u32, dst: u32) -> u32 {
    let sa = alpha(src);
    if sa == 255 {
        src
    } else if sa == 0 {
        dst
    } else {
        src + scale(dst, 255 - sa)
    }
}

/// Rounded average of four pixels (box filter for the mip pyramid).
#[inline(always)]
pub const fn avg4(a: u32, b: u32, c: u32, d: u32) -> u32 {
    let lo = (a & LO) + (b & LO) + (c & LO) + (d & LO) + 0x0002_0002;
    let hi = ((a >> 8) & LO) + ((b >> 8) & LO) + ((c >> 8) & LO) + ((d >> 8) & LO) + 0x0002_0002;
    ((lo >> 2) & LO) | (((hi >> 2) & LO) << 8)
}

#[inline]
pub fn premultiply(r: u8, g: u8, b: u8, a: u8) -> u32 {
    if a == 255 {
        pack(r, g, b, 255)
    } else {
        scale(pack(r, g, b, 0), a as u32) | (a as u32) << 24
    }
}

/// Straight (non-premultiplied) RGBA from a premultiplied pixel.
#[inline]
pub fn unpremultiply(p: u32) -> [u8; 4] {
    let [r, g, b, a] = unpack(p);
    match a {
        0 => [0, 0, 0, 0],
        255 => [r, g, b, 255],
        _ => {
            let inv = 255.0 / a as f32;
            [
                (r as f32 * inv + 0.5).min(255.0) as u8,
                (g as f32 * inv + 0.5).min(255.0) as u8,
                (b as f32 * inv + 0.5).min(255.0) as u8,
                a,
            ]
        }
    }
}

/// `unpremultiply` for every (alpha, value): entry `a * 256 + v`.
pub fn unpremul_table() -> Vec<u8> {
    (0..65536u32).map(|i| unpremultiply(pack((i & 255) as u8, 0, 0, (i >> 8) as u8))[0]).collect()
}

/// Float straight color (0..1) and alpha from a premultiplied pixel.
#[inline]
pub fn to_f32(p: u32) -> ([f32; 3], f32) {
    let [r, g, b, a] = unpack(p);
    if a == 0 {
        return ([0.0; 3], 0.0);
    }
    let inv = 1.0 / a as f32;
    ([r as f32 * inv, g as f32 * inv, b as f32 * inv], a as f32 / 255.0)
}

/// Packs straight float color + alpha into a premultiplied pixel.
#[inline]
pub fn from_f32(c: [f32; 3], a: f32) -> u32 {
    let a = a.clamp(0.0, 1.0);
    let a8 = (a * 255.0 + 0.5) as u32;
    if a8 == 0 {
        return 0;
    }
    let ch = |v: f32| ((v.clamp(0.0, 1.0) * a8 as f32 + 0.5) as u32).min(a8);
    ch(c[0]) | ch(c[1]) << 8 | ch(c[2]) << 16 | a8 << 24
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_is_exact() {
        for v in [0u32, 1, 127, 128, 254, 255] {
            for f in [0u32, 1, 128, 254, 255] {
                let p = pack(v as u8, v as u8, v as u8, v as u8);
                let want = ((v * f) as f32 / 255.0).round() as u8;
                assert_eq!(unpack(scale(p, f)), [want; 4], "v={v} f={f}");
            }
        }
    }

    #[test]
    fn over_preserves_premultiplied_invariant() {
        for sa in (0..=255).step_by(17) {
            for da in (0..=255).step_by(15) {
                let s = pack(sa as u8, 0, sa as u8, sa as u8);
                let d = pack(da as u8, da as u8, 0, da as u8);
                let [r, g, b, a] = unpack(over(s, d));
                assert!(r <= a && g <= a && b <= a);
            }
        }
    }

    #[test]
    fn premultiply_roundtrip() {
        let p = premultiply(200, 100, 50, 128);
        let [r, g, b, a] = unpremultiply(p);
        assert_eq!(a, 128);
        assert!((r as i32 - 200).abs() <= 2 && (g as i32 - 100).abs() <= 2 && (b as i32 - 50).abs() <= 2);
    }
}
