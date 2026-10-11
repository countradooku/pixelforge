//! Lossless wire structures for raster/vector masks, blend ranges and groups.
//!
//! A combined -2 channel is distinct from the real user mask (-3). Density,
//! feather and vector parameters remain editable metadata; callers must not
//! bake them into mask samples and discard the original representation.

use super::{Error, Limits, RawBytes, Reader, Result, Writer};

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct RasterMask {
    pub(crate) bounds: [i32; 4],
    pub(crate) default: u8,
    pub(crate) flags: u8,
}
impl RasterMask {
    pub(crate) fn absolute_bounds(&self, layer_bounds: [i32; 4]) -> Result<[i32; 4]> {
        let mut bounds = self.bounds;
        if self.flags & 1 != 0 {
            for index in [0, 2] {
                bounds[index] = bounds[index]
                    .checked_add(layer_bounds[0])
                    .ok_or(Error::Limit("relative mask y"))?;
            }
            for index in [1, 3] {
                bounds[index] = bounds[index]
                    .checked_add(layer_bounds[1])
                    .ok_or(Error::Limit("relative mask x"))?;
            }
        }
        if bounds[2] < bounds[0] || bounds[3] < bounds[1] {
            return Err(Error::Invalid("mask bounds"));
        }
        Ok(bounds)
    }
    pub(crate) fn enabled(&self) -> bool {
        self.flags & 2 == 0
    }
    pub(crate) fn inverted(&self) -> bool {
        self.flags & 4 != 0
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct MaskParameters {
    pub(crate) user_density: Option<u8>,
    pub(crate) user_feather_bits: Option<u64>,
    pub(crate) vector_density: Option<u8>,
    pub(crate) vector_feather_bits: Option<u64>,
}
impl MaskParameters {
    /// Numerical defaults only; absent and explicitly encoded defaults still
    /// remain different in the raw representation.
    pub(crate) fn is_neutral(&self) -> bool {
        self.user_density.is_none_or(|x| x == 255)
            && self.vector_density.is_none_or(|x| x == 255)
            && self
                .user_feather_bits
                .is_none_or(|x| f64::from_bits(x) == 0.0)
            && self
                .vector_feather_bits
                .is_none_or(|x| f64::from_bits(x) == 0.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct MaskMetadata {
    pub(crate) user: RasterMask,
    pub(crate) parameters: Option<MaskParameters>,
    pub(crate) real: Option<RasterMask>,
    pub(crate) tail: RawBytes,
}
impl MaskMetadata {
    pub(crate) fn parse(bytes: &RawBytes, has_real_channel: bool) -> Result<Option<Self>> {
        if bytes.is_empty() {
            return Ok(None);
        }
        let mut reader = Reader::blob(bytes);
        let user = RasterMask {
            bounds: read_bounds(&mut reader)?,
            default: reader.array::<1>()?[0],
            flags: reader.array::<1>()?[0],
        };
        user.absolute_bounds([0; 4])?;
        let parameters = if user.flags & 16 != 0 {
            let flags = reader.array::<1>()?[0];
            if flags & !15 != 0 {
                // Unknown parameter types have no universal framing. The raw
                // enclosing mask stays available through the PSD block tree.
                return Err(Error::Unsupported("mask parameter flags"));
            }
            Some(MaskParameters {
                user_density: if flags & 1 != 0 {
                    Some(reader.array::<1>()?[0])
                } else {
                    None
                },
                user_feather_bits: if flags & 2 != 0 {
                    Some(u64::from_be_bytes(reader.array()?))
                } else {
                    None
                },
                vector_density: if flags & 4 != 0 {
                    Some(reader.array::<1>()?[0])
                } else {
                    None
                },
                vector_feather_bits: if flags & 8 != 0 {
                    Some(u64::from_be_bytes(reader.array()?))
                } else {
                    None
                },
            })
        } else {
            None
        };
        let real = if has_real_channel || reader.remaining() >= 18 {
            let flags = reader.array::<1>()?[0];
            let default = reader.array::<1>()?[0];
            let real = RasterMask {
                bounds: read_bounds(&mut reader)?,
                default,
                flags,
            };
            real.absolute_bounds([0; 4])?;
            Some(real)
        } else {
            None
        };
        let tail = reader.raw(reader.remaining())?;
        Ok(Some(Self {
            user,
            parameters,
            real,
            tail,
        }))
    }

    pub(crate) fn channel(&self, id: i16) -> Result<&RasterMask> {
        match id {
            -2 => Ok(&self.user),
            -3 => self
                .real
                .as_ref()
                .ok_or(Error::Invalid("real user mask channel without real bounds")),
            _ => Err(Error::Invalid("mask channel ID")),
        }
    }

    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        if (self.user.flags & 16 != 0) != self.parameters.is_some() {
            return Err(Error::Invalid("mask parameter flag/fields mismatch"));
        }
        let mut writer = Writer::new(16 << 20);
        write_bounds(&mut writer, self.user.bounds)?;
        writer.bytes(&[self.user.default, self.user.flags])?;
        if let Some(p) = &self.parameters {
            let flags = u8::from(p.user_density.is_some())
                | (u8::from(p.user_feather_bits.is_some()) << 1)
                | (u8::from(p.vector_density.is_some()) << 2)
                | (u8::from(p.vector_feather_bits.is_some()) << 3);
            writer.bytes(&[flags])?;
            if let Some(v) = p.user_density {
                writer.bytes(&[v])?;
            }
            if let Some(v) = p.user_feather_bits {
                writer.bytes(&v.to_be_bytes())?;
            }
            if let Some(v) = p.vector_density {
                writer.bytes(&[v])?;
            }
            if let Some(v) = p.vector_feather_bits {
                writer.bytes(&v.to_be_bytes())?;
            }
        }
        if let Some(real) = &self.real {
            writer.bytes(&[real.flags, real.default])?;
            write_bounds(&mut writer, real.bounds)?;
        }
        writer.bytes(self.tail.as_slice())?;
        Ok(writer.bytes)
    }
}

fn read_bounds(reader: &mut Reader<'_>) -> Result<[i32; 4]> {
    Ok([reader.i32()?, reader.i32()?, reader.i32()?, reader.i32()?])
}
fn write_bounds(writer: &mut Writer, bounds: [i32; 4]) -> Result<()> {
    if bounds[2] < bounds[0] || bounds[3] < bounds[1] {
        return Err(Error::Invalid("mask bounds"));
    }
    for value in bounds {
        writer.bytes(&value.to_be_bytes())?;
    }
    Ok(())
}

/// A path record carries its complete 24-byte payload, including Adobe-private
/// path operations/origination metadata inside a subpath-length record.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct PathRecord {
    pub(crate) selector: u16,
    pub(crate) data: [u8; 24],
}
impl PathRecord {
    pub(crate) fn knot_count(&self) -> Option<u16> {
        matches!(self.selector, 0 | 3).then(|| u16::from_be_bytes([self.data[0], self.data[1]]))
    }
    /// Independently documented extension absent from Adobe's published spec:
    /// 0 exclude, 1 combine, 2 subtract, 3 intersect. Keep other values opaque.
    pub(crate) fn operation(&self) -> Option<i16> {
        matches!(self.selector, 0 | 3).then(|| i16::from_be_bytes([self.data[2], self.data[3]]))
    }
    /// Preceding, anchor and leaving points, each (y,x), signed 8.24 fixed.
    pub(crate) fn knot(&self) -> Option<[[i32; 2]; 3]> {
        if !matches!(self.selector, 1 | 2 | 4 | 5) {
            return None;
        }
        Some(std::array::from_fn(|point| {
            std::array::from_fn(|axis| {
                let i = point * 8 + axis * 4;
                i32::from_be_bytes([
                    self.data[i],
                    self.data[i + 1],
                    self.data[i + 2],
                    self.data[i + 3],
                ])
            })
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct VectorMask {
    pub(crate) flags: u32,
    pub(crate) records: Vec<PathRecord>,
    pub(crate) tail: RawBytes,
}
impl VectorMask {
    pub(crate) fn parse(bytes: &RawBytes, limits: &Limits) -> Result<Self> {
        if bytes.len() > limits.max_file_bytes {
            return Err(Error::Limit("vector mask bytes"));
        }
        let mut reader = Reader::blob(bytes);
        if reader.u32()? != 3 {
            return Err(Error::Unsupported("vector mask version"));
        }
        let flags = reader.u32()?;
        let count = reader.remaining() / 26;
        if count > limits.max_blocks {
            return Err(Error::Limit("vector path records"));
        }
        let mut records = Vec::new();
        records
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("vector path allocation"))?;
        for _ in 0..count {
            records.push(PathRecord {
                selector: reader.u16()?,
                data: reader.array()?,
            });
        }
        let tail = reader.raw(reader.remaining())?;
        // Photoshop aligns this block to four bytes. A longer partial record
        // is truncation, not padding, and must not become a silent missing knot.
        if tail.len() > 3 {
            return Err(Error::Truncated("vector path record"));
        }
        let mask = Self {
            flags,
            records,
            tail,
        };
        mask.validate_subpaths()?;
        Ok(mask)
    }
    pub(crate) fn validate_subpaths(&self) -> Result<()> {
        let mut index = 0;
        while index < self.records.len() {
            let record = &self.records[index];
            if let Some(count) = record.knot_count() {
                let end = index
                    .checked_add(1 + count as usize)
                    .filter(|&end| end <= self.records.len())
                    .ok_or(Error::Truncated("vector subpath knots"))?;
                let closed = record.selector == 0;
                for knot in &self.records[index + 1..end] {
                    if if closed {
                        !matches!(knot.selector, 1 | 2)
                    } else {
                        !matches!(knot.selector, 4 | 5)
                    } {
                        return Err(Error::Invalid("vector subpath knot selector"));
                    }
                }
                index = end;
            } else {
                if matches!(record.selector, 1 | 2 | 4 | 5) {
                    return Err(Error::Invalid("vector knot without subpath"));
                }
                index += 1;
            }
        }
        Ok(())
    }
    pub(crate) fn encode(&self, limits: &Limits) -> Result<Vec<u8>> {
        self.validate_subpaths()?;
        if self.records.len() > limits.max_blocks {
            return Err(Error::Limit("vector path records"));
        }
        let mut writer = Writer::new(limits.max_file_bytes);
        writer.bytes(&3u32.to_be_bytes())?;
        writer.bytes(&self.flags.to_be_bytes())?;
        for record in &self.records {
            writer.bytes(&record.selector.to_be_bytes())?;
            writer.bytes(&record.data)?;
        }
        writer.bytes(self.tail.as_slice())?;
        Ok(writer.bytes)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct BlendRange {
    pub(crate) source: [u8; 4],
    pub(crate) destination: [u8; 4],
}
impl BlendRange {
    pub(crate) fn parse_all(bytes: &[u8]) -> Result<Vec<Self>> {
        if bytes.len() % 8 != 0 {
            return Err(Error::Invalid("blend range length"));
        }
        let mut result = Vec::new();
        result
            .try_reserve_exact(bytes.len() / 8)
            .map_err(|_| Error::Limit("blend ranges allocation"))?;
        for pair in bytes.chunks_exact(8) {
            result.push(Self {
                source: [pair[0], pair[1], pair[2], pair[3]],
                destination: [pair[4], pair[5], pair[6], pair[7]],
            });
        }
        Ok(result)
    }
    pub(crate) fn is_identity(&self) -> bool {
        self.source == [0, 0, 255, 255] && self.destination == [0, 0, 255, 255]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct SectionDivider {
    pub(crate) kind: u32,
    pub(crate) blend_mode: Option<[u8; 4]>,
    pub(crate) subtype: Option<u32>,
    pub(crate) tail: RawBytes,
}
impl SectionDivider {
    pub(crate) fn parse(bytes: &RawBytes) -> Result<Self> {
        let mut reader = Reader::blob(bytes);
        let kind = reader.u32()?;
        if kind > 3 {
            return Err(Error::Unsupported("group divider kind"));
        }
        if reader.remaining() < 8 && reader.peek4() == Some(b"8BIM") {
            return Err(Error::Truncated("group blend mode"));
        }
        let blend_mode = if reader.remaining() >= 8 {
            if reader.array::<4>()? != *b"8BIM" {
                return Err(Error::Invalid("group blend signature"));
            }
            Some(reader.array()?)
        } else {
            None
        };
        let subtype = if blend_mode.is_some() && reader.remaining() >= 4 {
            Some(reader.u32()?)
        } else {
            None
        };
        let tail = reader.raw(reader.remaining())?;
        Ok(Self {
            kind,
            blend_mode,
            subtype,
            tail,
        })
    }
    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        if self.kind > 3 {
            return Err(Error::Unsupported("group divider kind"));
        }
        if self.subtype.is_some() && self.blend_mode.is_none() {
            return Err(Error::Invalid("divider subtype without blend mode"));
        }
        let mut writer = Writer::new(16 << 20);
        writer.bytes(&self.kind.to_be_bytes())?;
        if let Some(mode) = self.blend_mode {
            writer.bytes(b"8BIM")?;
            writer.bytes(&mode)?;
        }
        if let Some(subtype) = self.subtype {
            writer.bytes(&subtype.to_be_bytes())?;
        }
        writer.bytes(self.tail.as_slice())?;
        Ok(writer.bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mask_prefix(flags: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        for n in [-1i32, 2, 7, 9] {
            bytes.extend_from_slice(&n.to_be_bytes());
        }
        bytes.extend_from_slice(&[255, flags]);
        bytes
    }
    #[test]
    fn mask_20byte_and_36byte_variants_preserve_padding_and_distinct_bounds() {
        let mut raw = mask_prefix(0);
        raw.extend_from_slice(&[0xab, 0xcd]);
        let parsed = MaskMetadata::parse(&RawBytes::from_vec(raw.clone()), false)
            .unwrap()
            .unwrap();
        assert_eq!(parsed.encode().unwrap(), raw);
        assert!(parsed.real.is_none());
        let mut raw = mask_prefix(0);
        raw.extend_from_slice(&[2, 0]);
        for n in [10i32, 20, 30, 40] {
            raw.extend_from_slice(&n.to_be_bytes());
        }
        let parsed = MaskMetadata::parse(&RawBytes::from_vec(raw.clone()), true)
            .unwrap()
            .unwrap();
        assert_eq!(parsed.encode().unwrap(), raw);
        assert_eq!(parsed.channel(-2).unwrap().bounds, [-1, 2, 7, 9]);
        assert_eq!(parsed.channel(-3).unwrap().bounds, [10, 20, 30, 40]);
        assert!(!parsed.channel(-3).unwrap().enabled());
    }
    #[test]
    fn all_mask_parameters_and_noncanonical_double_bits_survive() {
        let mut raw = mask_prefix(16);
        raw.extend_from_slice(&[15, 137]);
        raw.extend_from_slice(&0x7ff8_1234_5678_9abcu64.to_be_bytes());
        raw.push(219);
        raw.extend_from_slice(&(-0.0f64).to_bits().to_be_bytes());
        raw.extend_from_slice(&[1, 255]);
        for n in [0i32, 1, 2, 3] {
            raw.extend_from_slice(&n.to_be_bytes());
        }
        let parsed = MaskMetadata::parse(&RawBytes::from_vec(raw.clone()), true)
            .unwrap()
            .unwrap();
        assert_eq!(parsed.encode().unwrap(), raw);
        assert!(!parsed.parameters.as_ref().unwrap().is_neutral());
        let mut changed = parsed;
        changed.user.flags |= 2;
        let edited = changed.encode().unwrap();
        assert_eq!(&edited[18..], &raw[18..]);
    }
    #[test]
    fn truncated_parameter_and_real_mask_are_rejected() {
        let mut raw = mask_prefix(16);
        raw.push(2);
        raw.extend_from_slice(&[0; 7]);
        assert!(MaskMetadata::parse(&RawBytes::from_vec(raw), false).is_err());
        assert!(MaskMetadata::parse(&RawBytes::from_vec(mask_prefix(0)), true).is_err());
        let mut raw = mask_prefix(16);
        raw.push(128);
        assert!(matches!(
            MaskMetadata::parse(&RawBytes::from_vec(raw), false),
            Err(Error::Unsupported(_))
        ));
    }
    #[test]
    fn relative_mask_bounds_use_layer_origin_and_reject_overflow() {
        let mask = RasterMask {
            bounds: [-1, 2, 7, 9],
            default: 255,
            flags: 5,
        };
        assert_eq!(
            mask.absolute_bounds([100, 200, 300, 400]).unwrap(),
            [99, 202, 107, 209]
        );
        assert!(mask.inverted());
        assert!(mask.absolute_bounds([0, i32::MAX, 0, 0]).is_err());
    }
    fn vector_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3u32.to_be_bytes());
        bytes.extend_from_slice(&5u32.to_be_bytes());
        bytes.extend_from_slice(&0u16.to_be_bytes());
        let mut header = [0u8; 24];
        header[..2].copy_from_slice(&1u16.to_be_bytes());
        header[2..4].copy_from_slice(&2i16.to_be_bytes());
        header[22..].copy_from_slice(&[0xab, 0xcd]);
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&1u16.to_be_bytes());
        for value in [
            -8_388_608i32,
            0,
            8_388_608,
            16_777_216,
            25_165_824,
            33_554_432,
        ] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes
    }
    #[test]
    fn vector_mask_preserves_signed_knots_operations_and_unknown_fields() {
        let mut raw = vector_bytes();
        raw.extend_from_slice(&0x1234u16.to_be_bytes());
        raw.extend_from_slice(&[0xa5; 24]);
        raw.extend_from_slice(&[0xbc, 0xde]);
        let parsed =
            VectorMask::parse(&RawBytes::from_vec(raw.clone()), &Limits::default()).unwrap();
        assert_eq!(parsed.encode(&Limits::default()).unwrap(), raw);
        assert_eq!(parsed.records[0].operation(), Some(2));
        assert_eq!(parsed.records[1].knot().unwrap()[0], [-8_388_608, 0]);
    }
    #[test]
    fn vector_knot_count_and_partial_record_are_checked() {
        let raw = vector_bytes();
        // An eight-byte header with up to three alignment bytes is a valid
        // empty vector mask; other prefixes cut a declared knot/record.
        for len in (0..8).chain(12..raw.len()) {
            assert!(
                VectorMask::parse(&RawBytes::from_vec(raw[..len].to_vec()), &Limits::default())
                    .is_err()
            );
        }
        let mut bad = raw.clone();
        bad[10..12].copy_from_slice(&2u16.to_be_bytes());
        assert!(VectorMask::parse(&RawBytes::from_vec(bad), &Limits::default()).is_err());
        assert!(
            VectorMask::parse(
                &RawBytes::from_vec(raw),
                &Limits {
                    max_blocks: 1,
                    ..Limits::default()
                }
            )
            .is_err()
        );
    }
    #[test]
    fn group_divider_preserves_subtype_tail_and_checks_signature() {
        let mut raw = [
            1u32.to_be_bytes().as_slice(),
            b"8BIMpass",
            0u32.to_be_bytes().as_slice(),
            &[0xab, 0xcd],
        ]
        .concat();
        let divider = SectionDivider::parse(&RawBytes::from_vec(raw.clone())).unwrap();
        assert_eq!(divider.blend_mode, Some(*b"pass"));
        assert_eq!(divider.encode().unwrap(), raw);
        raw[4] = 0;
        assert!(SectionDivider::parse(&RawBytes::from_vec(raw)).is_err());
    }
    #[test]
    fn blend_ranges_retain_split_sliders_without_silent_identity() {
        let ranges = BlendRange::parse_all(&[
            0, 0, 255, 255, 0, 0, 255, 255, 5, 20, 230, 250, 0, 1, 254, 255,
        ])
        .unwrap();
        assert!(ranges[0].is_identity());
        assert!(!ranges[1].is_identity());
        assert_eq!(ranges[1].source, [5, 20, 230, 250]);
        assert!(BlendRange::parse_all(&[0; 7]).is_err());
    }
}
