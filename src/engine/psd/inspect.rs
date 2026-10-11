//! Structural descriptor checks for a parsed PSD, independent of rendering.
//!
//! A typed verdict proves descriptor and EngineData re-encoding, not Photoshop
//! rendering fidelity or editable document mapping. Unknown versions and types
//! leave the bounded block opaque; malformed known layouts are reported without
//! discarding the raw document.

use std::fmt;

use anyhow::{Context, Result, ensure};

use super::descriptor::{self, Block, Descriptor, Limits, Value};
use super::engine_data::EngineData;
use super::{AdditionalInfo, PsdDocument};

#[derive(Clone, Debug)]
#[non_exhaustive]
pub(crate) struct Inspection {
    pub(crate) location: String,
    pub(crate) key: String,
    pub(crate) status: &'static str,
    pub(crate) descriptors: usize,
    pub(crate) engine_data: usize,
    pub(crate) message: String,
}

#[derive(Default)]
struct Counts {
    descriptors: usize,
    engine_data: usize,
    tail_bytes: usize,
    nonidentity_ranges: usize,
}

#[derive(Debug)]
struct Opaque(String);
impl fmt::Display for Opaque {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for Opaque {}

/// Inspects each additional-information block and image resource in file order.
pub(crate) fn inspect(document: &PsdDocument) -> Vec<Inspection> {
    let mut results = Vec::new();
    for (layer_ix, layer) in document.layer_mask.layers.iter().enumerate() {
        if !layer.mask.is_empty() {
            let result = (|| -> Result<Counts> {
                let parsed = super::masks::MaskMetadata::parse(
                    &layer.mask,
                    layer.channels.iter().any(|c| c.id == -3),
                )?
                .context("Missing parsed raster mask")?;
                ensure!(
                    parsed.encode()? == layer.mask.as_slice(),
                    "Raster mask changed wire bytes"
                );
                Ok(Counts {
                    tail_bytes: parsed.tail.len(),
                    ..Counts::default()
                })
            })();
            results.push(inspection(
                format!("layer[{layer_ix}].mask"),
                "raster-mask".into(),
                result,
            ));
        }
        if !layer.blending_ranges.is_empty() {
            let result = super::masks::BlendRange::parse_all(layer.blending_ranges.as_slice())
                .map(|ranges| Counts {
                    nonidentity_ranges: ranges.iter().filter(|range| !range.is_identity()).count(),
                    ..Counts::default()
                })
                .map_err(Into::into);
            results.push(inspection(
                format!("layer[{layer_ix}].blending_ranges"),
                "blend-ranges".into(),
                result,
            ));
        }
        for (block_ix, block) in layer.additional_info.iter().enumerate() {
            results.push(inspect_additional(
                block,
                format!("layer[{layer_ix}].additional[{block_ix}]"),
            ));
        }
    }
    for (block_ix, block) in document.layer_mask.additional_info.iter().enumerate() {
        results.push(inspect_additional(
            block,
            format!("global.additional[{block_ix}]"),
        ));
    }
    for (resource_ix, resource) in document.resources.iter().enumerate() {
        let bytes = resource.data.as_slice();
        let result = if matches!(
            resource.id,
            1065 | 1074..=1076 | 1078 | 1080 | 1082 | 1083 | 1088 | 3000
        ) {
            single_descriptor(bytes, 0, &mut Counts::default())
        } else {
            Err(Opaque("image resource payload retained without interpretation".into()).into())
        };
        results.push(inspection(
            format!("resource[{resource_ix}]"),
            resource.id.to_string(),
            result,
        ));
    }
    results
}

fn inspection(location: String, key: String, result: Result<Counts>) -> Inspection {
    match result {
        Ok(counts) => Inspection {
            location,
            key,
            status: "typed",
            descriptors: counts.descriptors,
            engine_data: counts.engine_data,
            message: format!(
                "Typed structure checked; {} EngineData trees exact; {} suffix bytes retained; {} nondefault blend ranges (not a rendering verdict)",
                counts.engine_data, counts.tail_bytes, counts.nonidentity_ranges
            ),
        },
        Err(error) => Inspection {
            location,
            key,
            status: if error.downcast_ref::<Opaque>().is_some()
                || matches!(
                    error.downcast_ref::<super::Error>(),
                    Some(super::Error::Unsupported(_))
                ) {
                "opaque"
            } else {
                "malformed"
            },
            descriptors: 0,
            engine_data: 0,
            message: format!("{error:#}"),
        },
    }
}

fn inspect_additional(block: &AdditionalInfo, location: String) -> Inspection {
    let bytes = block.data.as_slice();
    let key = String::from_utf8_lossy(&block.key).into_owned();
    let result = match &block.key {
        b"vmsk" | b"vsms" => (|| -> Result<Counts> {
            let parsed = super::masks::VectorMask::parse(&block.data, &super::Limits::default())?;
            ensure!(
                parsed.encode(&super::Limits::default())? == bytes,
                "Vector mask changed wire bytes"
            );
            // Exercise interpreted record views without pretending unknown
            // path-operation/origination fields have been rendered.
            for record in &parsed.records {
                let _ = (record.operation(), record.knot());
            }
            Ok(Counts {
                tail_bytes: parsed.tail.len(),
                ..Counts::default()
            })
        })(),
        b"lsct" | b"lsdk" => (|| -> Result<Counts> {
            let parsed = super::masks::SectionDivider::parse(&block.data)?;
            ensure!(
                parsed.encode()? == bytes,
                "Group divider changed wire bytes"
            );
            Ok(Counts {
                tail_bytes: parsed.tail.len(),
                ..Counts::default()
            })
        })(),
        b"brst" => {
            if bytes.len() % 4 == 0 {
                Ok(Counts::default())
            } else {
                Err(anyhow::anyhow!(
                    "Restricted-channel list is not a sequence of u32 values"
                ))
            }
        }
        _ => additional_payload(&block.key, bytes),
    };
    inspection(location, key, result)
}

fn additional_payload(key: &[u8; 4], bytes: &[u8]) -> Result<Counts> {
    let mut counts = Counts::default();
    match key {
        // These layouts begin with the descriptor version itself.
        b"SoCo" | b"GdFl" | b"PtFl" | b"vstk" | b"cinf" | b"PxSc" | b"artb" | b"artd" | b"abdd"
        | b"blwh" | b"CgEd" | b"vibA" | b"pths" | b"anFX" => {
            single_descriptor(bytes, 0, &mut counts)
        }
        b"lfx2" | b"lmfx" => {
            known_u32(bytes, 0, &[0], "layer effects")?;
            single_descriptor(bytes, 4, &mut counts)
        }
        b"vogk" => {
            known_u32(bytes, 0, &[1], "vector origination")?;
            single_descriptor(bytes, 4, &mut counts)
        }
        b"vscg" => {
            take(bytes, 0, 4)?;
            single_descriptor(bytes, 4, &mut counts)
        }
        b"clrL" => {
            known_u16(bytes, 0, 1, "color lookup")?;
            single_descriptor(bytes, 2, &mut counts)
        }
        b"SoLd" | b"SoLE" => {
            ensure!(
                take(bytes, 0, 4)? == b"soLD",
                "invalid smart-object signature"
            );
            known_u32(bytes, 4, &[4, 5], "smart object")?;
            // Adobe's SoLE table omits this version word; actual Photoshop
            // layouts use the same DescriptorBlock as SoLd.
            single_descriptor(bytes, 8, &mut counts)
        }
        b"TySh" => type_tool(bytes),
        b"plLd" | b"PlLd" => {
            ensure!(
                take(bytes, 0, 4)? == b"plcL",
                "invalid placed-layer signature"
            );
            known_u32(bytes, 4, &[3], "placed layer")?;
            let name_len = usize::from(
                *take(bytes, 8, 1)?
                    .first()
                    .context("placed-layer ID length")?,
            );
            // Pascal ID has no padding, then four longs and eight doubles.
            let warp_offset = 9_usize
                .checked_add(name_len)
                .and_then(|value| value.checked_add(16 + 64))
                .context("placed-layer length overflow")?;
            take(bytes, 9, name_len + 16 + 64)?;
            known_u32(bytes, warp_offset, &[0], "placed-layer warp")?;
            single_descriptor(bytes, warp_offset + 4, &mut counts)
        }
        // Txt2 is a separate text-engine representation; the layer's TySh
        // EngineData is checked above. Its private layout remains raw here.
        _ => Err(
            Opaque("additional-information payload retained without interpretation".into()).into(),
        ),
    }
}

fn type_tool(bytes: &[u8]) -> Result<Counts> {
    known_u16(bytes, 0, 1, "type tool")?;
    take(bytes, 2, 48)?;
    known_u16(bytes, 50, 50, "text")?;
    known_u32(bytes, 52, &[16], "text descriptor")?;
    let mut counts = Counts::default();
    let text_end = descriptor_at(bytes, 56, &mut counts)?;
    known_u16(bytes, text_end, 1, "text warp")?;
    known_u32(bytes, text_end + 2, &[16], "warp descriptor")?;
    let warp_end = descriptor_at(bytes, text_end + 6, &mut counts)?;
    // Photoshop writes four 32-bit bounds. Adobe's public table says 4*8;
    // retaining the complete suffix avoids changing either representation.
    take(bytes, warp_end, 16)?;
    counts.tail_bytes += bytes.len() - warp_end;
    Ok(counts)
}

fn single_descriptor(bytes: &[u8], version_offset: usize, counts: &mut Counts) -> Result<Counts> {
    known_u32(bytes, version_offset, &[16], "descriptor")?;
    let end = descriptor_at(bytes, version_offset + 4, counts)?;
    counts.tail_bytes += bytes.len() - end;
    Ok(std::mem::take(counts))
}

fn descriptor_at(bytes: &[u8], offset: usize, counts: &mut Counts) -> Result<usize> {
    let bounded = bytes.get(offset..).context("truncated descriptor offset")?;
    let block = descriptor::decode_preserving(bounded, Limits::default())?;
    match &block {
        Block::Parsed {
            descriptor: parsed,
            trailing,
        } => {
            let used = bounded.len() - trailing.len();
            let encoded = descriptor::encode(parsed, Limits::default())?;
            ensure!(
                encoded == bounded[..used],
                "descriptor changed wire bytes after re-encoding"
            );
            ensure!(
                descriptor::decode(&encoded, Limits::default())? == *parsed,
                "descriptor changed semantic values after re-encoding"
            );
            // Also check the API that preserves opaque suffixes, which may
            // contain a second descriptor or newer private extension.
            ensure!(
                descriptor::encode_block(&block, Limits::default())? == bounded,
                "descriptor suffix changed after re-encoding"
            );
            counts.descriptors += 1;
            inspect_engine_data(parsed, counts)?;
            Ok(offset + used)
        }
        Block::Opaque { unknown_type, .. } => {
            ensure!(
                descriptor::encode_block(&block, Limits::default())? == bounded,
                "opaque descriptor bytes changed"
            );
            Err(Opaque(format!("unknown descriptor type {unknown_type:?}; entire bounded parent retained byte-exactly")).into())
        }
    }
}

fn inspect_engine_data(descriptor: &Descriptor, counts: &mut Counts) -> Result<()> {
    for entry in &descriptor.entries {
        if entry.key.bytes() == b"EngineData" {
            if let Value::Raw(bytes) = &entry.value {
                let tree = EngineData::parse(bytes, Limits::default())
                    .context("parsing text EngineData")?;
                ensure!(
                    tree.to_bytes(Limits::default())? == *bytes,
                    "EngineData lexical bytes changed"
                );
                counts.engine_data += 1;
            } else {
                return Err(Opaque(
                    "EngineData uses a value type other than tdta; payload retained".into(),
                )
                .into());
            }
        }
        inspect_nested_value(&entry.value, counts)?;
    }
    Ok(())
}

fn inspect_nested_value(value: &Value, counts: &mut Counts) -> Result<()> {
    match value {
        Value::Object(descriptor)
        | Value::GlobalObject(descriptor)
        | Value::ObjectArray { descriptor, .. } => inspect_engine_data(descriptor, counts)?,
        Value::List(values) => {
            for value in values {
                inspect_nested_value(value, counts)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn take(bytes: &[u8], offset: usize, length: usize) -> Result<&[u8]> {
    let end = offset
        .checked_add(length)
        .context("descriptor block offset overflow")?;
    bytes.get(offset..end).with_context(|| {
        format!("truncated descriptor block at byte {offset}, need {length} bytes")
    })
}
fn known_u16(bytes: &[u8], offset: usize, expected: u16, name: &str) -> Result<()> {
    let bytes = take(bytes, offset, 2)?;
    let version = u16::from_be_bytes([bytes[0], bytes[1]]);
    if version != expected {
        return Err(Opaque(format!(
            "unsupported {name} version {version}; bounded payload retained"
        ))
        .into());
    }
    Ok(())
}
fn known_u32(bytes: &[u8], offset: usize, expected: &[u32], name: &str) -> Result<()> {
    let bytes = take(bytes, offset, 4)?;
    let version = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if !expected.contains(&version) {
        return Err(Opaque(format!(
            "unsupported {name} version {version}; bounded payload retained"
        ))
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::descriptor::{Class, Entry, Id};
    use super::super::{Header, ImageResource, LayerRecord, Version};
    use super::*;

    fn descriptor_bytes() -> Vec<u8> {
        let mut descriptor = Descriptor::new(Class::new("", Id::Four(*b"null")));
        descriptor.entries.push(Entry::new(
            Id::String(b"EngineData".to_vec()),
            Value::Raw(b"<< /Mystery [+0001 (a\\051) unknown] >>".to_vec()),
        ));
        descriptor
            .entries
            .push(Entry::new(Id::Four(*b"????"), Value::Boolean(23)));
        descriptor::encode(&descriptor, Limits::default()).unwrap()
    }
    fn versioned() -> Vec<u8> {
        let mut bytes = 16_u32.to_be_bytes().to_vec();
        bytes.extend(descriptor_bytes());
        bytes
    }
    fn tysh() -> Vec<u8> {
        let mut bytes = 1_u16.to_be_bytes().to_vec();
        for value in [1.0_f64, 0.0, 0.0, 1.0, 10.0, 20.0] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes.extend_from_slice(&50_u16.to_be_bytes());
        bytes.extend(versioned());
        bytes.extend_from_slice(&1_u16.to_be_bytes());
        bytes.extend(versioned());
        bytes.extend_from_slice(&[0; 16]);
        bytes
    }

    #[test]
    fn future_mask_and_group_variants_are_opaque_not_malformed() {
        let mut layer = LayerRecord::new([0; 4], "Future mask");
        let mut mask = vec![0; 18];
        mask[17] = 16;
        mask.push(128);
        layer.mask = super::super::RawBytes::from_vec(mask);
        layer
            .additional_info
            .push(AdditionalInfo::new(*b"lsct", 4u32.to_be_bytes().to_vec()));
        let mut document = PsdDocument::new(Header::rgb(1, 1, 3, Version::Psd), vec![]);
        document.layer_mask.layers.push(layer);
        let results = inspect(&document);
        assert_eq!(
            results
                .iter()
                .find(|item| item.key == "raster-mask")
                .unwrap()
                .status,
            "opaque"
        );
        assert_eq!(
            results
                .iter()
                .find(|item| item.key == "lsct")
                .unwrap()
                .status,
            "opaque"
        );
    }

    #[test]
    fn every_known_descriptor_envelope_is_checked() {
        let mut cases = vec![(b"TySh", tysh())];
        for key in [
            b"SoCo", b"GdFl", b"PtFl", b"vstk", b"cinf", b"PxSc", b"artb", b"artd", b"abdd",
            b"blwh", b"CgEd", b"vibA", b"pths", b"anFX",
        ] {
            cases.push((key, versioned()));
        }
        for key in [b"lfx2", b"lmfx"] {
            let mut bytes = 0_u32.to_be_bytes().to_vec();
            bytes.extend(versioned());
            cases.push((key, bytes));
        }
        let mut bytes = 1_u32.to_be_bytes().to_vec();
        bytes.extend(versioned());
        cases.push((b"vogk", bytes));
        let mut bytes = b"SoCo".to_vec();
        bytes.extend(versioned());
        cases.push((b"vscg", bytes));
        let mut bytes = 1_u16.to_be_bytes().to_vec();
        bytes.extend(versioned());
        cases.push((b"clrL", bytes));
        for key in [b"SoLd", b"SoLE"] {
            let mut bytes = b"soLD".to_vec();
            bytes.extend_from_slice(&4_u32.to_be_bytes());
            bytes.extend(versioned());
            cases.push((key, bytes));
        }
        for (key, bytes) in cases {
            let row = inspect_additional(&AdditionalInfo::new(*key, bytes), "test".into());
            assert_eq!(row.status, "typed", "{key:?}: {}", row.message);
            assert!(row.descriptors > 0 && row.engine_data > 0);
        }
    }

    #[test]
    fn text_envelope_checks_both_descriptors_and_engine_data() {
        let counts = type_tool(&tysh()).unwrap();
        assert_eq!(
            (counts.descriptors, counts.engine_data, counts.tail_bytes),
            (2, 2, 16)
        );
    }

    #[test]
    fn unknown_type_and_future_version_remain_opaque() {
        let mut bytes = 16_u32.to_be_bytes().to_vec();
        bytes.extend_from_slice(b"\0\0\0\0\0\0\0\0null\0\0\0\x01\0\0\0\0testZZZZ\0\xff\xaa");
        for bytes in [bytes, 17_u32.to_be_bytes().to_vec()] {
            let row = inspect_additional(&AdditionalInfo::new(*b"SoCo", bytes), "test".into());
            assert_eq!(row.status, "opaque", "{}", row.message);
        }
    }

    #[test]
    fn malformed_known_envelope_is_reported_without_mutating_raw_document() {
        let block = AdditionalInfo::new(*b"TySh", vec![0, 1]);
        let before = block.clone();
        let row = inspect_additional(&block, "test".into());
        assert_eq!(row.status, "malformed");
        assert_eq!(block, before);
    }

    #[test]
    fn invalid_engine_data_is_a_visible_validation_failure() {
        let mut descriptor = Descriptor::new(Class::new("", Id::Four(*b"null")));
        descriptor.set(
            Id::String(b"EngineData".to_vec()),
            Value::Raw(b"<< /MissingValue >>".to_vec()),
        );
        let mut bytes = 16_u32.to_be_bytes().to_vec();
        bytes.extend(descriptor::encode(&descriptor, Limits::default()).unwrap());
        let row = inspect_additional(&AdditionalInfo::new(*b"SoCo", bytes), "test".into());
        assert_eq!(row.status, "malformed");
        assert!(row.message.contains("EngineData"));
    }

    #[test]
    fn inspection_preserves_duplicate_layer_blocks_and_resource_order() {
        let mut document =
            PsdDocument::new(Header::rgb(1, 1, 3, Version::Psd), vec![0, 0, 1, 2, 3]);
        let mut layer = LayerRecord::new([0, 0, 1, 1], "test");
        layer.additional_info.clear();
        layer
            .additional_info
            .push(AdditionalInfo::new(*b"SoCo", versioned()));
        layer
            .additional_info
            .push(AdditionalInfo::new(*b"SoCo", versioned()));
        document.layer_mask.layers.push(layer);
        document
            .resources
            .push(ImageResource::new(1065, versioned()));
        document
            .resources
            .push(ImageResource::new(4999, vec![1, 2, 3]));
        let rows = inspect(&document);
        assert_eq!(rows.len(), 4);
        assert_eq!(
            rows.iter().map(|row| row.status).collect::<Vec<_>>(),
            ["typed", "typed", "typed", "opaque"]
        );
        assert_eq!(rows[1].location, "layer[0].additional[1]");
        assert_eq!(rows[2].key, "1065");
    }
}
