//! Blend If split sliders and RGB channel restrictions.
//!
//! The ordinary compositor never enters this path without explicit advanced
//! settings. Integer ramps are a CPU reference; Photoshop corpus comparisons,
//! especially overlapping ramps and translucent backgrounds, remain the fidelity
//! authority. GPU composition explicitly falls back for this state.

use super::{
    blend::{self, BlendMode},
    px,
};
use anyhow::{Result, ensure};

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Range {
    pub source: [u8; 4],
    pub destination: [u8; 4],
}
impl Default for Range {
    fn default() -> Self {
        Self {
            source: [0, 0, 255, 255],
            destination: [0, 0, 255, 255],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdvancedBlend {
    /// Gray composite first, then R, G, B and any preserved extra channels.
    pub ranges: Vec<Range>,
    /// Adobe's ordered brst list. Duplicates retain their wire meaning/order.
    pub restricted_channels: Vec<u32>,
}
impl AdvancedBlend {
    pub fn from_psd(ranges: &[u8], restrictions: &[u8]) -> Result<Self> {
        ensure!(
            ranges.len() % 8 == 0 && ranges.len() <= 57 * 8,
            "Invalid Blend If range count"
        );
        ensure!(
            restrictions.len() % 4 == 0 && restrictions.len() <= 56 * 4,
            "Invalid restricted-channel list"
        );
        let ranges = ranges
            .chunks_exact(8)
            .map(|r| Range {
                source: [r[0], r[1], r[2], r[3]],
                destination: [r[4], r[5], r[6], r[7]],
            })
            .collect();
        let restricted_channels = restrictions
            .chunks_exact(4)
            .map(|r| u32::from_be_bytes([r[0], r[1], r[2], r[3]]))
            .collect();
        let value = Self {
            ranges,
            restricted_channels,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.ranges.len() <= 57 && self.restricted_channels.len() <= 56,
            "Advanced blend record limit exceeded"
        );
        ensure!(
            self.restricted_channels.iter().all(|&c| c < 3),
            "Restrictions of non-RGB channels are not implemented"
        );
        for range in &self.ranges {
            for stops in [range.source, range.destination] {
                ensure!(
                    stops[0] <= stops[1] && stops[2] <= stops[3],
                    "Reversed Blend If split handles"
                );
            }
        }
        ensure!(
            self.ranges.iter().skip(4).all(|r| *r == Range::default()),
            "Blend If on extra channels is not implemented"
        );
        Ok(())
    }
    pub fn is_identity(&self) -> bool {
        self.restricted_channels.is_empty() && self.ranges.iter().all(|r| *r == Range::default())
    }
    pub fn psd_ranges(&self) -> Vec<u8> {
        self.ranges
            .iter()
            .flat_map(|r| r.source.into_iter().chain(r.destination))
            .collect()
    }
    pub fn psd_restrictions(&self) -> Vec<u8> {
        self.restricted_channels
            .iter()
            .flat_map(|&c| c.to_be_bytes())
            .collect()
    }
    /// Versioned PFLX payload. The PSD representation stays big-endian so one
    /// parser owns its bounds and validation in both file formats.
    pub fn to_bytes(&self) -> Vec<u8> {
        let ranges = self.psd_ranges();
        let restrictions = self.psd_restrictions();
        let mut output = vec![
            1,
            self.ranges.len() as u8,
            self.restricted_channels.len() as u8,
        ];
        output.extend_from_slice(&ranges);
        output.extend_from_slice(&restrictions);
        output
    }
    pub fn from_bytes(input: &[u8]) -> Result<Self> {
        ensure!(
            input.len() >= 3 && input[0] == 1,
            "Unsupported advanced blend extension"
        );
        let n = usize::from(input[1]) * 8;
        let m = usize::from(input[2]) * 4;
        ensure!(
            input.len() == 3 + n + m,
            "Truncated or trailing advanced blend extension"
        );
        Self::from_psd(&input[3..3 + n], &input[3 + n..])
    }
    pub fn blend_region(
        &self,
        destination: &mut [u32],
        source: &[u32],
        opacity: u32,
        mode: BlendMode,
        width: usize,
        origin: (i32, i32),
    ) {
        if width == 0 {
            return;
        }
        let mut restricted = [false; 3];
        for &channel in &self.restricted_channels {
            if channel < 3 {
                restricted[channel as usize] = true;
            }
        }
        for (index, (d, &s)) in destination.iter_mut().zip(source).enumerate() {
            if s == 0 || opacity == 0 {
                continue;
            }
            let cs = px::unpremultiply(s);
            let cb = px::unpremultiply(*d);
            let mut coverage = 255;
            for (channel, range) in self.ranges.iter().take(4).enumerate() {
                let src = if channel == 0 {
                    gray(cs)
                } else {
                    cs[channel - 1]
                };
                let dst = if channel == 0 {
                    gray(cb)
                } else {
                    cb[channel - 1]
                };
                coverage = mul255(coverage, ramp(src, range.source));
                coverage = mul255(coverage, ramp(dst, range.destination));
            }
            let opacity = mul255(opacity.min(255), coverage);
            let old = *d;
            let scaled = px::scale(s, opacity);
            let mut output = if mode == BlendMode::Dissolve {
                blend::dissolve_px(
                    old,
                    s,
                    opacity,
                    origin.0.wrapping_add((index % width) as i32),
                    origin.1.wrapping_add((index / width) as i32),
                )
            } else if mode == BlendMode::Normal {
                px::over(scaled, old)
            } else {
                blend::blend_px(old, scaled, mode)
            };
            if restricted.iter().any(|&value| value) {
                if restricted.iter().all(|&value| value) {
                    *d = old;
                    continue;
                }
                let alpha = px::alpha(output);
                let mut bytes = px::unpack(output);
                for channel in 0..3 {
                    if restricted[channel] {
                        bytes[channel] = ((u32::from(cb[channel]) * alpha + 127) / 255) as u8;
                    }
                }
                output = u32::from_le_bytes(bytes);
            }
            *d = output;
        }
    }
}
fn mul255(a: u32, b: u32) -> u32 {
    (a * b + 127) / 255
}
fn gray(c: [u8; 4]) -> u8 {
    ((u32::from(c[0]) * 30 + u32::from(c[1]) * 59 + u32::from(c[2]) * 11 + 50) / 100) as u8
}
fn ramp(value: u8, stops: [u8; 4]) -> u32 {
    let [black0, black1, white0, white1] = stops.map(u32::from);
    let value = u32::from(value);
    if value < black0 || value > white1 {
        return 0;
    }
    let black = if value < black1 && black1 > black0 {
        ((value - black0) * 255 + (black1 - black0) / 2) / (black1 - black0)
    } else {
        255
    };
    let white = if value > white0 && white1 > white0 {
        ((white1 - value) * 255 + (white1 - white0) / 2) / (white1 - white0)
    } else {
        255
    };
    mul255(black, white)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_ramps_preserve_every_8bit_value() {
        for value in 0..=255 {
            assert_eq!(ramp(value, [0, 0, 255, 255]), 255);
        }
    }
    #[test]
    fn split_ramps_have_exact_endpoints_and_midpoints() {
        for (value, want) in [
            (0, 0),
            (20, 0),
            (30, 128),
            (40, 255),
            (100, 255),
            (200, 255),
            (210, 128),
            (220, 0),
            (255, 0),
        ] {
            assert_eq!(ramp(value, [20, 40, 200, 220]), want);
        }
    }
    #[test]
    fn source_and_destination_ranges_change_coverage() {
        let advanced = AdvancedBlend {
            ranges: vec![Range {
                source: [0, 0, 100, 200],
                ..Range::default()
            }],
            restricted_channels: vec![],
        };
        let mut output = [px::pack(0, 0, 0, 255); 3];
        let source = [
            px::pack(100, 100, 100, 255),
            px::pack(150, 150, 150, 255),
            px::pack(200, 200, 200, 255),
        ];
        advanced.blend_region(&mut output, &source, 255, BlendMode::Normal, 3, (0, 0));
        assert_eq!(
            output,
            [
                px::pack(100, 100, 100, 255),
                px::pack(75, 75, 75, 255),
                px::pack(0, 0, 0, 255)
            ]
        );
        let mut dest = advanced.clone();
        dest.ranges[0] = Range {
            source: [0, 0, 255, 255],
            destination: [0, 100, 255, 255],
        };
        let mut output = [px::pack(0, 0, 0, 255)];
        dest.blend_region(
            &mut output,
            &[px::pack(255, 0, 0, 255)],
            255,
            BlendMode::Normal,
            1,
            (0, 0),
        );
        assert_eq!(output, [px::pack(0, 0, 0, 255)]);
    }
    #[test]
    fn channel_restrictions_preserve_disabled_destination_channels() {
        let advanced = AdvancedBlend {
            ranges: vec![],
            restricted_channels: vec![0, 2, 0],
        };
        let mut output = [px::pack(12, 34, 56, 255)];
        advanced.blend_region(
            &mut output,
            &[px::pack(90, 100, 110, 255)],
            255,
            BlendMode::Normal,
            1,
            (0, 0),
        );
        assert_eq!(output, [px::pack(12, 100, 56, 255)]);
    }
    #[test]
    fn wire_and_native_roundtrips_preserve_order_duplicates_and_identity_ranges() {
        let ranges = [
            0, 0, 255, 255, 0, 0, 255, 255, 1, 2, 253, 254, 3, 4, 250, 251,
        ];
        let restrictions = [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2];
        let original = AdvancedBlend::from_psd(&ranges, &restrictions).unwrap();
        assert_eq!(original.psd_ranges(), ranges);
        assert_eq!(original.psd_restrictions(), restrictions);
        assert_eq!(
            AdvancedBlend::from_bytes(&original.to_bytes()).unwrap(),
            original
        );
        for length in 0..original.to_bytes().len() {
            assert!(AdvancedBlend::from_bytes(&original.to_bytes()[..length]).is_err());
        }
    }
    #[test]
    fn invalid_limits_and_unsupported_channels_are_explicit_errors() {
        assert!(AdvancedBlend::from_psd(&[0; 7], &[]).is_err());
        assert!(AdvancedBlend::from_psd(&[0; 58 * 8], &[]).is_err());
        assert!(AdvancedBlend::from_psd(&[], &3u32.to_be_bytes()).is_err());
        assert!(AdvancedBlend::from_psd(&[20, 10, 200, 255, 0, 0, 255, 255], &[]).is_err());
    }
}
