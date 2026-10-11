//! Bounded Photoshop channel compression. Rows are independent for PackBits and
//! prediction; the ZIP stream spans all rows (including composite planes).

use std::io::Write;

use flate2::{Compression, Decompress, FlushDecompress, Status, write::ZlibEncoder};

use super::{Error, Limits, Result, Version};

struct BoundedOutput {
    bytes: Vec<u8>,
    limit: usize,
    memory: Vec<super::super::memory::Reservation>,
}
impl BoundedOutput {
    fn grow(&mut self, end: usize) -> std::io::Result<()> {
        if end > self.limit {
            return Err(std::io::Error::other(
                "PSD encoded channel exceeds output limit",
            ));
        }
        let capacity = self.bytes.capacity();
        if end > capacity {
            // Geometric growth is bounded by the configured output ceiling;
            // reserve_exact avoids Vec doubling past that ceiling.
            let target = end.max(capacity.saturating_mul(2)).min(self.limit);
            self.bytes
                .try_reserve_exact(target - self.bytes.len())
                .map_err(|_| std::io::Error::other("PSD encoded channel allocation failed"))?;
            self.memory.push(
                super::super::memory::global().track_cpu((self.bytes.capacity() - capacity) as u64),
            );
        }
        Ok(())
    }
}
impl Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let end = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| std::io::Error::other("PSD encoded channel size overflow"))?;
        self.grow(end)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    Raw = 0,
    PackBits = 1,
    Zip = 2,
    ZipPrediction = 3,
}

impl TryFrom<u16> for Method {
    type Error = Error;
    fn try_from(value: u16) -> Result<Self> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::PackBits),
            2 => Ok(Self::Zip),
            3 => Ok(Self::ZipPrediction),
            _ => Err(Error::Unsupported("channel compression")),
        }
    }
}

fn dimensions(width: u32, rows: u32, depth: u16, limits: &Limits) -> Result<(usize, usize)> {
    let row = match depth {
        1 => (width as usize).div_ceil(8),
        8 => width as usize,
        _ => {
            return Err(Error::Unsupported(
                "channel depth (only Bitmap and 8-bit supported)",
            ));
        }
    };
    let size = row
        .checked_mul(rows as usize)
        .ok_or(Error::Limit("channel dimensions"))?;
    if size > limits.max_decoded_bytes {
        return Err(Error::Limit("decoded channel bytes"));
    }
    Ok((row, size))
}

fn zeroed(size: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| Error::Limit("channel allocation"))?;
    bytes.resize(size, 0);
    Ok(bytes)
}

pub(crate) fn decode(
    encoded: &[u8],
    width: u32,
    rows: u32,
    depth: u16,
    version: Version,
    limits: &Limits,
) -> Result<Vec<u8>> {
    if encoded.len() > limits.max_file_bytes {
        return Err(Error::Limit("encoded channel bytes"));
    }
    let (row_bytes, expected) = dimensions(width, rows, depth, limits)?;
    let _decoded_memory =
        super::super::memory::global().track_cpu(expected.saturating_add(1) as u64);
    let (method, input) = encoded
        .split_at_checked(2)
        .ok_or(Error::Truncated("compression marker"))?;
    let method = Method::try_from(u16::from_be_bytes([method[0], method[1]]))?;
    if method == Method::Raw {
        if input.len() != expected {
            return Err(Error::Invalid("raw channel size"));
        }
        let mut out = zeroed(expected)?;
        out.copy_from_slice(input);
        return Ok(out);
    }
    if method == Method::PackBits {
        let entry = if version == Version::Psb { 4 } else { 2 };
        let table_size = (rows as usize)
            .checked_mul(entry)
            .ok_or(Error::Limit("RLE table"))?;
        let (table, mut payload) = input
            .split_at_checked(table_size)
            .ok_or(Error::Truncated("RLE row counts"))?;
        let mut out = zeroed(expected)?;
        for (row, count) in table.chunks_exact(entry).enumerate() {
            let length = if entry == 4 {
                u32::from_be_bytes([count[0], count[1], count[2], count[3]]) as usize
            } else {
                u16::from_be_bytes([count[0], count[1]]) as usize
            };
            let (encoded_row, rest) = payload
                .split_at_checked(length)
                .ok_or(Error::Truncated("RLE row"))?;
            unpack_row(
                encoded_row,
                &mut out[row * row_bytes..(row + 1) * row_bytes],
            )?;
            payload = rest;
        }
        if !payload.is_empty() {
            return Err(Error::Invalid("trailing RLE channel bytes"));
        }
        return Ok(out);
    }
    if method == Method::ZipPrediction && depth != 8 {
        return Err(Error::Unsupported("Bitmap ZIP prediction"));
    }
    // One extra byte detects a ZIP bomb even when the advertised dimensions are
    // tiny. StreamEnd and total_in distinguish truncation and appended streams.
    let mut out = zeroed(expected.checked_add(1).ok_or(Error::Limit("ZIP output"))?)?;
    let mut decoder = Decompress::new(true);
    let status = decoder
        .decompress(input, &mut out, FlushDecompress::Finish)
        .map_err(|_| Error::Invalid("ZIP stream or checksum"))?;
    if status != Status::StreamEnd
        || decoder.total_out() != expected as u64
        || decoder.total_in() != input.len() as u64
    {
        return Err(Error::Invalid("ZIP stream length"));
    }
    out.truncate(expected);
    if method == Method::ZipPrediction && row_bytes > 0 {
        for row in out.chunks_exact_mut(row_bytes) {
            for x in 1..row.len() {
                row[x] = row[x].wrapping_add(row[x - 1]);
            }
        }
    }
    Ok(out)
}

fn unpack_row(mut input: &[u8], output: &mut [u8]) -> Result<()> {
    let mut position = 0usize;
    while let Some((&control, rest)) = input.split_first() {
        input = rest;
        let control = control as i8;
        match control {
            -128 => {}
            0..=127 => {
                let length = control as usize + 1;
                let (literal, rest) = input
                    .split_at_checked(length)
                    .ok_or(Error::Truncated("PackBits literal"))?;
                let end = position
                    .checked_add(length)
                    .ok_or(Error::Limit("PackBits output"))?;
                output
                    .get_mut(position..end)
                    .ok_or(Error::Invalid("PackBits row overrun"))?
                    .copy_from_slice(literal);
                input = rest;
                position = end;
            }
            _ => {
                let length = (1i16 - control as i16) as usize;
                let (&value, rest) = input
                    .split_first()
                    .ok_or(Error::Truncated("PackBits repeat"))?;
                let end = position
                    .checked_add(length)
                    .ok_or(Error::Limit("PackBits output"))?;
                output
                    .get_mut(position..end)
                    .ok_or(Error::Invalid("PackBits row overrun"))?
                    .fill(value);
                input = rest;
                position = end;
            }
        }
    }
    if position != output.len() {
        return Err(Error::Invalid("PackBits row underrun"));
    }
    Ok(())
}

fn pack_row(input: &[u8], output: &mut BoundedOutput) -> Result<()> {
    let mut position = 0;
    while position < input.len() {
        let mut repeated = 1;
        while repeated < 128
            && position + repeated < input.len()
            && input[position] == input[position + repeated]
        {
            repeated += 1;
        }
        if repeated >= 3 {
            output.write_all(&[(1i16 - repeated as i16) as u8, input[position]])?;
            position += repeated;
            continue;
        }
        let start = position;
        position += repeated;
        while position < input.len() && position - start < 128 {
            if position + 2 < input.len()
                && input[position] == input[position + 1]
                && input[position] == input[position + 2]
            {
                break;
            }
            position += 1;
        }
        output.write_all(&[(position - start - 1) as u8])?;
        output.write_all(&input[start..position])?;
    }
    Ok(())
}

pub(crate) fn encode(
    input: &[u8],
    width: u32,
    rows: u32,
    depth: u16,
    version: Version,
    method: Method,
    limits: &Limits,
) -> Result<Vec<u8>> {
    let (row_bytes, expected) = dimensions(width, rows, depth, limits)?;
    if input.len() != expected {
        return Err(Error::Invalid("input channel size"));
    }
    let mut output = BoundedOutput {
        bytes: Vec::new(),
        limit: limits.max_file_bytes,
        memory: Vec::new(),
    };
    output.write_all(&(method as u16).to_be_bytes())?;
    match method {
        Method::Raw => output.write_all(input)?,
        Method::PackBits => {
            let entry = if version == Version::Psb { 4 } else { 2 };
            let table_size = (rows as usize)
                .checked_mul(entry)
                .and_then(|x| x.checked_add(2))
                .ok_or(Error::Limit("RLE table"))?;
            if table_size > limits.max_file_bytes {
                return Err(Error::Limit("RLE table"));
            }
            output.grow(table_size)?;
            output.bytes.resize(table_size, 0);
            for row in 0..rows as usize {
                let start = output.bytes.len();
                pack_row(&input[row * row_bytes..(row + 1) * row_bytes], &mut output)?;
                let length = output.bytes.len() - start;
                let offset = 2 + row * entry;
                if entry == 4 {
                    let length = u32::try_from(length).map_err(|_| Error::Limit("PSB RLE row"))?;
                    output.bytes[offset..offset + 4].copy_from_slice(&length.to_be_bytes());
                } else {
                    let length = u16::try_from(length)
                        .map_err(|_| Error::Limit("PSD RLE row; use ZIP or PSB"))?;
                    output.bytes[offset..offset + 2].copy_from_slice(&length.to_be_bytes());
                }
            }
        }
        Method::Zip | Method::ZipPrediction => {
            if method == Method::ZipPrediction && depth != 8 {
                return Err(Error::Unsupported("Bitmap ZIP prediction"));
            }
            let mut encoder = ZlibEncoder::new(output, Compression::fast());
            if method == Method::ZipPrediction && row_bytes > 0 {
                let _prediction_memory = super::super::memory::global().track_cpu(row_bytes as u64);
                let mut predicted = zeroed(row_bytes)?;
                for row in input.chunks_exact(row_bytes) {
                    predicted[0] = row[0];
                    for x in 1..row.len() {
                        predicted[x] = row[x].wrapping_sub(row[x - 1]);
                    }
                    encoder.write_all(&predicted)?;
                }
            } else {
                encoder.write_all(input)?;
            }
            output = encoder.finish()?;
        }
    }
    Ok(output.bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_methods_round_trip_boundary_runs_and_prediction_wrap() {
        let limits = Limits::default();
        for width in [0, 1, 2, 3, 127, 128, 129, 255, 256, 257] {
            for version in [Version::Psd, Version::Psb] {
                for method in [
                    Method::Raw,
                    Method::PackBits,
                    Method::Zip,
                    Method::ZipPrediction,
                ] {
                    let mut bytes: Vec<u8> = (0..width * 3).map(|x| (x * 37) as u8).collect();
                    if bytes.len() >= 150 {
                        bytes[10..150].fill(211);
                    }
                    let encoded = encode(&bytes, width, 3, 8, version, method, &limits).unwrap();
                    assert_eq!(
                        decode(&encoded, width, 3, 8, version, &limits).unwrap(),
                        bytes,
                        "{width} {version:?} {method:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn packbits_rejects_truncation_overrun_and_underrun() {
        for (input, size) in [
            (&[2, 1, 2][..], 3),
            (&[255][..], 2),
            (&[253, 7][..], 3),
            (&[0, 7][..], 2),
        ] {
            assert!(unpack_row(input, &mut vec![0; size]).is_err());
        }
        let mut out = [0; 2];
        unpack_row(&[128, 255, 42, 128], &mut out).unwrap();
        assert_eq!(out, [42, 42]);
    }

    #[test]
    fn zip_rejects_truncation_appended_stream_and_expansion_beyond_dimensions() {
        let limits = Limits::default();
        let bytes = vec![42; 1024];
        let encoded = encode(&bytes, 1024, 1, 8, Version::Psd, Method::Zip, &limits).unwrap();
        for length in 0..encoded.len() {
            assert!(decode(&encoded[..length], 1024, 1, 8, Version::Psd, &limits).is_err());
        }
        assert!(decode(&encoded, 1, 1, 8, Version::Psd, &limits).is_err());
        let mut appended = encoded.clone();
        appended.extend_from_slice(&encoded[2..]);
        assert!(decode(&appended, 1024, 1, 8, Version::Psd, &limits).is_err());
        let mut corrupt = encoded;
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        assert!(decode(&corrupt, 1024, 1, 8, Version::Psd, &limits).is_err());
    }

    #[test]
    fn allocation_limit_precedes_decoding() {
        let limits = Limits {
            max_decoded_bytes: 8,
            ..Limits::default()
        };
        assert!(matches!(
            decode(&[0, 0], u32::MAX, u32::MAX, 8, Version::Psb, &limits),
            Err(Error::Limit(_))
        ));
    }

    #[test]
    fn bitmap_rows_use_ceil_byte_width() {
        let input = [0xa5, 0x80, 0x5a, 0];
        for method in [Method::Raw, Method::PackBits, Method::Zip] {
            let bytes = encode(&input, 9, 2, 1, Version::Psd, method, &Limits::default()).unwrap();
            assert_eq!(
                decode(&bytes, 9, 2, 1, Version::Psd, &Limits::default()).unwrap(),
                input
            );
        }
    }

    #[test]
    fn specification_packbits_vectors_decode_without_encoder() {
        // Literal ABC followed by four D bytes and a legal no-op. The row
        // count includes compressed bytes, not decoded pixels.
        let psd = [0, 1, 0, 7, 2, b'A', b'B', b'C', 253, b'D', 128];
        assert_eq!(
            decode(&psd, 7, 1, 8, Version::Psd, &Limits::default()).unwrap(),
            b"ABCDDDD"
        );
        let psb = [0, 1, 0, 0, 0, 7, 2, b'A', b'B', b'C', 253, b'D', 128];
        assert_eq!(
            decode(&psb, 7, 1, 8, Version::Psb, &Limits::default()).unwrap(),
            b"ABCDDDD"
        );
    }

    #[test]
    fn encoded_output_limit_applies_during_all_compressors() {
        let limits = Limits {
            max_file_bytes: 2,
            ..Limits::default()
        };
        for method in [
            Method::Raw,
            Method::PackBits,
            Method::Zip,
            Method::ZipPrediction,
        ] {
            assert!(encode(&[0; 4], 4, 1, 8, Version::Psd, method, &limits).is_err());
        }
    }
}
