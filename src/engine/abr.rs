//! Parser for Adobe Photoshop brush files (`.abr`), extracting **sampled**
//! (image-based) brush tips.
//!
//! Supported layouts:
//!
//! * **Versions 1 and 2** (Photoshop ≤ 6): a `u16` brush count followed by
//!   brush records (`1` = computed, `2` = sampled).
//! * **Versions 6, 7 and 10** (Photoshop 7+): a `u16` subversion followed by
//!   `8BIM` sections. `samp` holds the brush images, `desc` an Action
//!   Descriptor listing the presets (name, diameter, spacing, angle, ...).
//!   `patt` (patterns) and `phry` (preset hierarchy) are ignored.
//!
//! The parser never panics: every read is bounds-checked and malformed input
//! yields `Err(String)`. A single sampled image that fails to decode is
//! counted in [`AbrFile::skipped_other`] instead of failing the whole file.
//!
//! Format notes verified against Photoshop 2026's bundled files:
//!
//! * Section lengths and `samp` entry lengths are padded to a multiple of 4.
//! * A subversion-2 `samp` entry is: `u8`-length UUID, 4 unknown bytes
//!   (`00 01 00 00`), then a PSD "virtual memory array list" (version 3,
//!   length, bounding rect, channel-slot count) whose slots are mostly empty
//!   (`u32` 0); the first written slot holds the gray image (depth `u32`,
//!   rect, depth `u16`, compression `u8`, data).
//! * Preset names may be localisation keys like
//!   `$$$/Presets/Brushes/SoftRound=Soft Round`; the text after `=` is used.

use std::collections::HashMap;

/// Maximum width/height accepted for a brush image.
const MAX_DIM: u32 = 30_000;
/// Maximum pixel count accepted for a brush image (guards allocations).
const MAX_PIXELS: u64 = 64 * 1024 * 1024;
/// Maximum nesting depth of Action Descriptors (guards the stack).
const MAX_DEPTH: usize = 64;
/// Spacing used when the file does not specify one.
const DEFAULT_SPACING: f32 = 0.25;

/// A sampled (bitmap) brush tip ready for use as a brush mask.
#[derive(Debug, Clone, PartialEq)]
pub struct SampledTip {
    /// Preset name from the descriptor, else `"Sampled Brush N"`.
    pub name: String,
    /// `sampledData` UUID (v6+); empty for v1/v2 files.
    pub uuid: String,
    /// Image width in pixels.
    pub w: u32,
    /// Image height in pixels.
    pub h: u32,
    /// `w * h` coverage values, row-major: 0 = no paint, 255 = full paint.
    pub alpha: Vec<u8>,
    /// Spacing as a fraction of the diameter (e.g. `0.25`).
    pub spacing: f32,
    /// Preset diameter in pixels, else `max(w, h)`.
    pub diameter: f32,
    /// Tip angle in degrees (0 when unknown).
    pub angle: f32,
    /// Tip roundness in `0..=1` (1 when unknown).
    pub roundness: f32,
}

/// The result of parsing an `.abr` file.
#[derive(Debug, Clone, PartialEq)]
pub struct AbrFile {
    /// File format version (1, 2, 6, 7 or 10).
    pub version: u16,
    /// Subversion for v6+ files, 0 for v1/v2.
    pub subversion: u16,
    /// All sampled tips, in preset order, followed by unreferenced images.
    pub tips: Vec<SampledTip>,
    /// Number of computed (parametric round) brushes skipped.
    pub skipped_computed: usize,
    /// Number of other entries skipped (unknown types, missing or
    /// undecodable images).
    pub skipped_other: usize,
}

/// Parses an `.abr` file and returns its sampled brush tips.
pub fn parse(bytes: &[u8]) -> Result<AbrFile, String> {
    let mut r = Reader::new(bytes);
    let version = r.u16().map_err(|e| format!("header: {e}"))?;
    match version {
        1 | 2 => parse_v12(&mut r, version),
        6 | 7 | 10 => parse_v6(&mut r, version),
        v => Err(format!("unsupported ABR version {v}")),
    }
}

// ---------------------------------------------------------------------------
// Byte reader
// ---------------------------------------------------------------------------

/// A bounds-checked big-endian cursor over a byte slice.
#[derive(Clone)]
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.remaining() {
            return Err(format!(
                "unexpected end of data at offset {} (need {n} bytes, have {})",
                self.pos,
                self.remaining()
            ));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn skip(&mut self, n: usize) -> Result<(), String> {
        self.take(n).map(|_| ())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.array::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_be_bytes(self.array()?))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_be_bytes(self.array()?))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_be_bytes(self.array()?))
    }
    fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_be_bytes(self.array()?))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_be_bytes(self.array()?))
    }

    /// Reads a `u32` length and returns that many bytes.
    fn len_prefixed(&mut self) -> Result<&'a [u8], String> {
        let n = self.u32()? as usize;
        self.take(n)
    }

    /// Unicode string: `u32` code-unit count + UTF-16BE, trailing NULs removed.
    fn unicode(&mut self) -> Result<String, String> {
        let n = self.u32()? as usize;
        let bytes = self.take(n.checked_mul(2).ok_or("string length overflow")?)?;
        let units = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_be_bytes(c));
        let s: String = char::decode_utf16(units)
            .map(|c| c.unwrap_or(char::REPLACEMENT_CHARACTER))
            .collect();
        Ok(s.trim_end_matches('\0').to_string())
    }

    /// Descriptor key / class ID: `u32` length, 0 meaning a 4-byte code.
    fn key(&mut self) -> Result<String, String> {
        let n = self.u32()? as usize;
        let bytes = self.take(if n == 0 { 4 } else { n })?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

// ---------------------------------------------------------------------------
// Image decoding (shared by all versions)
// ---------------------------------------------------------------------------

/// Validates bounds and returns `(w, h)`.
fn dims(top: i32, left: i32, bottom: i32, right: i32) -> Result<(u32, u32), String> {
    let w = i64::from(right) - i64::from(left);
    let h = i64::from(bottom) - i64::from(top);
    if w <= 0 || h <= 0 || w > i64::from(MAX_DIM) || h > i64::from(MAX_DIM) {
        return Err(format!("invalid brush bounds {w}x{h}"));
    }
    if (w as u64) * (h as u64) > MAX_PIXELS {
        return Err(format!("brush too large ({w}x{h})"));
    }
    Ok((w as u32, h as u32))
}

/// Decodes PackBits data into exactly `out_len` bytes (short rows are
/// zero-filled, excess output is discarded).
fn packbits(src: &[u8], out: &mut Vec<u8>, out_len: usize) {
    let start = out.len();
    let end = start + out_len;
    let mut i = 0;
    while i < src.len() && out.len() < end {
        let n = src[i] as i8;
        i += 1;
        if n >= 0 {
            let cnt = n as usize + 1;
            let lit = &src[i..(i + cnt).min(src.len())];
            let room = end - out.len();
            out.extend_from_slice(&lit[..lit.len().min(room)]);
            i += cnt;
        } else if n != -128 {
            let Some(&b) = src.get(i) else { break };
            i += 1;
            let cnt = (1 - n as isize) as usize;
            let room = end - out.len();
            out.extend(std::iter::repeat_n(b, cnt.min(room)));
        }
    }
    out.resize(end, 0);
}

/// Reads `w x h` pixels of the given bit depth and compression and converts
/// them to 8-bit coverage.
fn read_pixels(
    r: &mut Reader,
    w: u32,
    h: u32,
    depth: u16,
    compression: u8,
) -> Result<Vec<u8>, String> {
    let bpp = match depth {
        8 => 1usize,
        16 => 2,
        32 => 4,
        d => return Err(format!("unsupported sample depth {d}")),
    };
    let (w, h) = (w as usize, h as usize);
    let row_bytes = w * bpp;
    let raw: Vec<u8> = match compression {
        0 => r.take(row_bytes * h)?.to_vec(),
        1 => {
            // Row-length table (u16 per row), then PackBits rows.
            let table = r.take(h * 2)?;
            let mut out = Vec::new();
            for &c in table.as_chunks::<2>().0 {
                let n = u16::from_be_bytes(c) as usize;
                packbits(r.take(n)?, &mut out, row_bytes);
            }
            out
        }
        c => return Err(format!("unsupported compression {c}")),
    };
    Ok(match bpp {
        1 => raw,
        2 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| ((u32::from(u16::from_be_bytes(c)) * 255 + 32767) / 65535) as u8)
            .collect(),
        _ => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| {
                let f = f32::from_be_bytes(c);
                if f.is_nan() {
                    0
                } else {
                    (f.clamp(0.0, 1.0) * 255.0).round() as u8
                }
            })
            .collect(),
    })
}

/// Reads bounds (`top, left, bottom, right` as i32), depth `u16`,
/// compression `u8` and the pixel data.
fn read_image_block(r: &mut Reader) -> Result<(u32, u32, Vec<u8>), String> {
    let (top, left, bottom, right) = (r.i32()?, r.i32()?, r.i32()?, r.i32()?);
    let depth = r.u16()?;
    let compression = r.u8()?;
    let (w, h) = dims(top, left, bottom, right)?;
    let alpha = read_pixels(r, w, h, depth, compression)?;
    Ok((w, h, alpha))
}

fn fallback_name(n: usize) -> String {
    format!("Sampled Brush {n}")
}

// ---------------------------------------------------------------------------
// Versions 1 and 2
// ---------------------------------------------------------------------------

fn parse_v12(r: &mut Reader, version: u16) -> Result<AbrFile, String> {
    let count = r.u16()?;
    let mut file = AbrFile {
        version,
        subversion: 0,
        tips: Vec::new(),
        skipped_computed: 0,
        skipped_other: 0,
    };
    for i in 0..count {
        let kind = r.u16().map_err(|e| format!("brush {i}: {e}"))?;
        let size = r.u32().map_err(|e| format!("brush {i}: {e}"))? as usize;
        let data = r.take(size).map_err(|e| format!("brush {i}: {e}"))?;
        match kind {
            1 => file.skipped_computed += 1,
            2 => match parse_v12_sampled(&mut Reader::new(data), version) {
                Ok(mut tip) => {
                    if tip.name.is_empty() {
                        tip.name = fallback_name(file.tips.len() + 1);
                    }
                    file.tips.push(tip);
                }
                Err(_) => file.skipped_other += 1,
            },
            _ => file.skipped_other += 1,
        }
    }
    Ok(file)
}

fn parse_v12_sampled(r: &mut Reader, version: u16) -> Result<SampledTip, String> {
    let _misc = r.u32()?;
    let spacing = r.u16()?;
    let name = if version == 2 {
        r.unicode()?
    } else {
        String::new()
    };
    let _antialias = r.u8()?;
    r.skip(8)?; // short bounds (superseded by the long bounds)
    let (w, h, alpha) = read_image_block(r)?;
    Ok(SampledTip {
        name,
        uuid: String::new(),
        w,
        h,
        alpha,
        spacing: if spacing == 0 {
            DEFAULT_SPACING
        } else {
            f32::from(spacing) / 100.0
        },
        diameter: w.max(h) as f32,
        angle: 0.0,
        roundness: 1.0,
    })
}

// ---------------------------------------------------------------------------
// Versions 6, 7, 10
// ---------------------------------------------------------------------------

/// A decoded image from a `samp` section.
struct SampImage {
    uuid: String,
    w: u32,
    h: u32,
    alpha: Vec<u8>,
}

fn parse_v6(r: &mut Reader, version: u16) -> Result<AbrFile, String> {
    let subversion = r.u16()?;
    let mut file = AbrFile {
        version,
        subversion,
        tips: Vec::new(),
        skipped_computed: 0,
        skipped_other: 0,
    };
    let mut images: Vec<SampImage> = Vec::new();
    let mut presets: Vec<Preset> = Vec::new();

    while r.remaining() > 0 {
        // Tolerate trailing zero padding after the last section.
        if r.buf[r.pos..].iter().all(|&b| b == 0) {
            break;
        }
        let at = r.pos;
        let sig = r.array::<4>().map_err(|e| format!("section header: {e}"))?;
        if &sig != b"8BIM" {
            return Err(format!("bad section signature {sig:02x?} at offset {at}"));
        }
        let key = r.array::<4>()?;
        let len = r.u32()? as usize;
        let data = r
            .take(len)
            .map_err(|e| format!("section {}: {e}", String::from_utf8_lossy(&key)))?;
        // Sections are padded to a multiple of 4 (the pad may be cut at EOF).
        let pad = (4 - len % 4) % 4;
        r.skip(pad.min(r.remaining()))?;

        match &key {
            b"samp" => parse_samp(data, subversion, &mut images, &mut file.skipped_other)?,
            b"desc" => presets = parse_desc(data)?,
            _ => {} // patt, phry, ...
        }
    }

    let mut used = vec![false; images.len()];
    let index: HashMap<&str, usize> = images
        .iter()
        .enumerate()
        .map(|(i, s)| (s.uuid.as_str(), i))
        .collect();

    for p in presets {
        match p.kind {
            PresetKind::Computed => file.skipped_computed += 1,
            PresetKind::Other => file.skipped_other += 1,
            PresetKind::Sampled {
                uuid,
                diameter,
                spacing,
                angle,
                roundness,
            } => {
                let Some(&i) = index.get(uuid.as_str()) else {
                    file.skipped_other += 1;
                    continue;
                };
                used[i] = true;
                let img = &images[i];
                let n = file.tips.len() + 1;
                file.tips.push(SampledTip {
                    name: if p.name.is_empty() {
                        fallback_name(n)
                    } else {
                        p.name
                    },
                    uuid,
                    w: img.w,
                    h: img.h,
                    alpha: img.alpha.clone(),
                    spacing: spacing.unwrap_or(DEFAULT_SPACING),
                    diameter: diameter.unwrap_or(img.w.max(img.h) as f32),
                    angle: angle.unwrap_or(0.0),
                    roundness: roundness.unwrap_or(1.0),
                });
            }
        }
    }

    // Images not referenced by any preset still become tips.
    for (img, used) in images.into_iter().zip(used) {
        if used {
            continue;
        }
        let n = file.tips.len() + 1;
        file.tips.push(SampledTip {
            name: fallback_name(n),
            uuid: img.uuid,
            diameter: img.w.max(img.h) as f32,
            w: img.w,
            h: img.h,
            alpha: img.alpha,
            spacing: DEFAULT_SPACING,
            angle: 0.0,
            roundness: 1.0,
        });
    }
    Ok(file)
}

/// Walks the entries of a `samp` section.
fn parse_samp(
    data: &[u8],
    subversion: u16,
    out: &mut Vec<SampImage>,
    skipped: &mut usize,
) -> Result<(), String> {
    let mut r = Reader::new(data);
    while r.remaining() >= 4 {
        let len = r.u32()? as usize;
        let entry = r
            .take(len)
            .map_err(|e| format!("samp entry {}: {e}", out.len()))?;
        r.skip(((4 - len % 4) % 4).min(r.remaining()))?;
        match parse_samp_entry(entry, subversion) {
            Ok(img) => out.push(img),
            Err(_) => *skipped += 1,
        }
    }
    Ok(())
}

/// Decodes one `samp` entry. Subversion 1 uses a fixed 10-byte gap before the
/// bounds (as GIMP does); subversion 2+ wraps the image in a virtual memory
/// array list. Each layout is tried as a fallback for the other.
fn parse_samp_entry(entry: &[u8], subversion: u16) -> Result<SampImage, String> {
    let mut r = Reader::new(entry);
    let uuid_len = r.u8()? as usize;
    let uuid = String::from_utf8_lossy(r.take(uuid_len)?).into_owned();
    let body = r;

    let fixed = |mut r: Reader| -> Result<(u32, u32, Vec<u8>), String> {
        r.skip(10)?;
        read_image_block(&mut r)
    };
    let vmal = |mut r: Reader| -> Result<(u32, u32, Vec<u8>), String> {
        r.skip(4)?; // unknown, observed 00 01 00 00
        read_vmal(&mut r)
    };
    let (w, h, alpha) = if subversion == 1 {
        fixed(body.clone()).or_else(|_| vmal(body))?
    } else {
        vmal(body.clone()).or_else(|_| fixed(body))?
    };
    Ok(SampImage { uuid, w, h, alpha })
}

/// Reads a PSD "virtual memory array list" and returns its first written
/// channel as an 8-bit image.
fn read_vmal(r: &mut Reader) -> Result<(u32, u32, Vec<u8>), String> {
    let version = r.u32()?;
    if version != 3 {
        return Err(format!("unexpected VMAL version {version}"));
    }
    let len = r.u32()? as usize;
    let mut r = Reader::new(r.take(len)?);
    r.skip(16)?; // overall bounding rect
    let slots = r.u32()?;
    // `slots` channels plus user and sheet masks.
    for _ in 0..slots.saturating_add(2) {
        if r.u32()? == 0 {
            continue; // slot not written
        }
        let clen = r.u32()? as usize;
        if clen == 0 {
            continue;
        }
        let mut c = Reader::new(r.take(clen)?);
        let _depth32 = c.u32()?;
        return read_image_block(&mut c);
    }
    Err("VMAL has no written channel".into())
}

// ---------------------------------------------------------------------------
// Action Descriptors
// ---------------------------------------------------------------------------

/// A parsed Action Descriptor value (only what preset extraction needs is
/// retained; opaque payloads are skipped).
#[derive(Debug, Clone)]
enum Value {
    Object(Descriptor),
    List(Vec<Value>),
    Number(f64),
    Text(String),
    Other,
}

#[derive(Debug, Clone, Default)]
struct Descriptor {
    class_id: String,
    items: Vec<(String, Value)>,
}

impl Descriptor {
    fn get(&self, key: &str) -> Option<&Value> {
        self.items.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    fn number(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            Value::Number(n) if n.is_finite() => Some(*n),
            _ => None,
        }
    }
    fn text(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }
}

fn read_descriptor(r: &mut Reader, depth: usize) -> Result<Descriptor, String> {
    if depth > MAX_DEPTH {
        return Err("descriptor nesting too deep".into());
    }
    let _name = r.unicode()?;
    let class_id = r.key()?;
    let count = r.u32()?;
    let mut items = Vec::new();
    for _ in 0..count {
        let key = r.key()?;
        let ty = r.array::<4>()?;
        let value = read_value(r, &ty, depth + 1).map_err(|e| format!("{class_id}.{key}: {e}"))?;
        items.push((key, value));
    }
    Ok(Descriptor { class_id, items })
}

fn read_value(r: &mut Reader, ty: &[u8; 4], depth: usize) -> Result<Value, String> {
    if depth > MAX_DEPTH {
        return Err("descriptor nesting too deep".into());
    }
    Ok(match ty {
        b"Objc" | b"GlbO" => Value::Object(read_descriptor(r, depth)?),
        b"VlLs" => {
            let n = r.u32()?;
            let mut items = Vec::new();
            for _ in 0..n {
                let t = r.array::<4>()?;
                items.push(read_value(r, &t, depth + 1)?);
            }
            Value::List(items)
        }
        b"doub" => Value::Number(r.f64()?),
        b"UntF" => {
            r.skip(4)?; // unit, e.g. #Pxl, #Prc, #Ang
            Value::Number(r.f64()?)
        }
        b"UnFl" => {
            r.skip(4)?;
            let n = r.u32()? as usize;
            r.skip(n.checked_mul(8).ok_or("UnFl count overflow")?)?;
            Value::Other
        }
        b"long" => Value::Number(f64::from(r.i32()?)),
        b"comp" => Value::Number(r.i64()? as f64),
        b"bool" => {
            r.u8()?;
            Value::Other
        }
        b"TEXT" => Value::Text(r.unicode()?),
        b"enum" => {
            r.key()?;
            r.key()?;
            Value::Other
        }
        b"type" | b"GlbC" => {
            r.unicode()?;
            r.key()?;
            Value::Other
        }
        b"tdta" | b"alis" | b"Pth " => {
            r.len_prefixed()?;
            Value::Other
        }
        b"obj " => {
            read_reference(r)?;
            Value::Other
        }
        b"ObAr" => {
            r.u32()?; // item count
            read_descriptor(r, depth)?;
            Value::Other
        }
        t => {
            return Err(format!(
                "unknown descriptor type {:?}",
                String::from_utf8_lossy(t)
            ));
        }
    })
}

/// Skips a reference (`obj `) value.
fn read_reference(r: &mut Reader) -> Result<(), String> {
    let n = r.u32()?;
    for _ in 0..n {
        let t = r.array::<4>()?;
        match &t {
            b"prop" => {
                r.unicode()?;
                r.key()?;
                r.key()?;
            }
            b"Clss" => {
                r.unicode()?;
                r.key()?;
            }
            b"Enmr" => {
                r.unicode()?;
                r.key()?;
                r.key()?;
                r.key()?;
            }
            b"rele" => {
                r.unicode()?;
                r.key()?;
                r.u32()?;
            }
            b"Idnt" | b"indx" => {
                r.u32()?;
            }
            b"name" => {
                r.unicode()?;
                r.key()?;
                r.unicode()?;
            }
            t => {
                return Err(format!(
                    "unknown reference type {:?}",
                    String::from_utf8_lossy(t)
                ));
            }
        }
    }
    Ok(())
}

enum PresetKind {
    Sampled {
        uuid: String,
        diameter: Option<f32>,
        spacing: Option<f32>,
        angle: Option<f32>,
        roundness: Option<f32>,
    },
    Computed,
    Other,
}

struct Preset {
    name: String,
    kind: PresetKind,
}

/// Turns a localisation key (`$$$/path/Key=Display Name`) into its display name.
fn display_name(raw: &str) -> String {
    match raw.strip_prefix("$$$/").and_then(|s| s.split_once('=')) {
        Some((_, name)) => name.trim().to_string(),
        None => raw.trim().to_string(),
    }
}

/// Parses the `desc` section into a flat preset list.
fn parse_desc(data: &[u8]) -> Result<Vec<Preset>, String> {
    let mut r = Reader::new(data);
    let _version = r.u32()?; // observed 16
    let root = read_descriptor(&mut r, 0).map_err(|e| format!("desc: {e}"))?;
    let Some(Value::List(list)) = root.get("Brsh") else {
        return Ok(Vec::new());
    };
    let presets = list
        .iter()
        .map(|v| {
            let Value::Object(preset) = v else {
                return Preset {
                    name: String::new(),
                    kind: PresetKind::Other,
                };
            };
            let brush = match preset.get("Brsh") {
                Some(Value::Object(b)) => Some(b),
                _ => None,
            };
            // Preset name, falling back to the tip's own name.
            let name = preset
                .text("Nm  ")
                .or_else(|| brush.and_then(|b| b.text("Nm  ")))
                .map(display_name)
                .unwrap_or_default();
            let kind = match brush {
                Some(b) if b.class_id == "sampledBrush" => match b.text("sampledData") {
                    Some(uuid) => PresetKind::Sampled {
                        uuid: uuid.to_string(),
                        diameter: b.number("Dmtr").filter(|d| *d > 0.0).map(|d| d as f32),
                        spacing: b
                            .number("Spcn")
                            .filter(|s| *s > 0.0)
                            .map(|s| (s / 100.0) as f32),
                        angle: b.number("Angl").map(|a| a as f32),
                        roundness: b.number("Rndn").map(|v| (v / 100.0).clamp(0.0, 1.0) as f32),
                    },
                    None => PresetKind::Other,
                },
                Some(b) if b.class_id == "computedBrush" => PresetKind::Computed,
                // Bristle (`dBrush`), erodible (`dTips`) and unknown tips.
                _ => PresetKind::Other,
            };
            Preset { name, kind }
        })
        .collect();
    Ok(presets)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_FILES: [&str; 3] = [
        "/Applications/Adobe Photoshop 2026/Presets/Brushes/Legacy Brushes.abr",
        "/Applications/Adobe Photoshop 2026/Presets/Brushes/Converted Legacy Tool Presets.abr",
        "/Applications/Adobe Photoshop 2026/Adobe Photoshop 2026.app/Contents/Required/Default Brushes.abr",
    ];

    fn load(path: &str) -> Option<Vec<u8>> {
        match std::fs::read(path) {
            Ok(b) => Some(b),
            Err(e) => {
                println!("SKIP {path}: {e}");
                None
            }
        }
    }

    #[test]
    fn parse_real_files() {
        let mut total_tips = 0;
        for path in REAL_FILES {
            let Some(bytes) = load(path) else { continue };
            let f = parse(&bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
            println!(
                "{path}\n  version {}.{} tips {} skipped_computed {} skipped_other {}",
                f.version,
                f.subversion,
                f.tips.len(),
                f.skipped_computed,
                f.skipped_other
            );
            for t in f.tips.iter().take(5) {
                println!(
                    "  {:?} {}x{} diam {} spacing {} angle {} round {} uuid {}",
                    t.name, t.w, t.h, t.diameter, t.spacing, t.angle, t.roundness, t.uuid
                );
            }
            let mut nontrivial = 0;
            for t in &f.tips {
                assert_eq!(t.alpha.len(), (t.w * t.h) as usize, "{}", t.name);
                assert!(
                    (1..=5000).contains(&t.w) && (1..=5000).contains(&t.h),
                    "{}",
                    t.name
                );
                assert!(!t.name.is_empty());
                assert!(t.spacing > 0.0 && t.diameter > 0.0);
                if t.alpha.contains(&0) && t.alpha.iter().any(|&v| v > 200) {
                    nontrivial += 1;
                }
            }
            println!("  tips with both 0 and >200 values: {nontrivial}");
            assert!(f.tips.is_empty() || nontrivial > 0);
            total_tips += f.tips.len();
        }
        let any_present = REAL_FILES.iter().any(|p| std::path::Path::new(p).exists());
        if any_present {
            assert!(total_tips > 0);
        }
    }

    #[test]
    fn truncated_real_files_never_panic() {
        for path in REAL_FILES {
            let Some(bytes) = load(path) else { continue };
            // Simple LCG so the test is deterministic without dependencies.
            let mut seed: u64 = 0x2545_f491_4f6c_dd1d ^ bytes.len() as u64;
            let mut cuts: Vec<usize> = (0..64).collect();
            for _ in 0..200 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                cuts.push((seed >> 33) as usize % bytes.len());
            }
            let (mut ok, mut err) = (0, 0);
            for cut in cuts {
                match parse(&bytes[..cut]) {
                    Ok(_) => ok += 1,
                    Err(_) => err += 1,
                }
            }
            // Also flip some bytes in place.
            let mut corrupt = bytes.clone();
            for _ in 0..200 {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let i = (seed >> 33) as usize % corrupt.len();
                corrupt[i] = (seed >> 13) as u8;
                let _ = parse(&corrupt);
            }
            println!("{path}: truncated ok {ok} err {err}");
        }
    }

    /// Builds a v1 or v2 file with one sampled brush and one computed brush.
    fn synth_v12(
        version: u16,
        w: u16,
        h: u16,
        pixels: &[u8],
        rle_rows: Option<&[Vec<u8>]>,
    ) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&0u32.to_be_bytes()); // misc
        body.extend_from_slice(&50u16.to_be_bytes()); // spacing 50%
        if version == 2 {
            let name: Vec<u16> = "Test Tip\0".encode_utf16().collect();
            body.extend_from_slice(&(name.len() as u32).to_be_bytes());
            for u in name {
                body.extend_from_slice(&u.to_be_bytes());
            }
        }
        body.push(1); // antialias
        for v in [0i16, 0, h as i16, w as i16] {
            body.extend_from_slice(&v.to_be_bytes());
        }
        for v in [0i32, 0, i32::from(h), i32::from(w)] {
            body.extend_from_slice(&v.to_be_bytes());
        }
        body.extend_from_slice(&8u16.to_be_bytes());
        match rle_rows {
            None => {
                body.push(0);
                body.extend_from_slice(pixels);
            }
            Some(rows) => {
                body.push(1);
                for row in rows {
                    body.extend_from_slice(&(row.len() as u16).to_be_bytes());
                }
                for row in rows {
                    body.extend_from_slice(row);
                }
            }
        }
        let mut f = Vec::new();
        f.extend_from_slice(&version.to_be_bytes());
        f.extend_from_slice(&2u16.to_be_bytes());
        // computed brush
        f.extend_from_slice(&1u16.to_be_bytes());
        f.extend_from_slice(&14u32.to_be_bytes());
        f.extend_from_slice(&[0u8; 14]);
        // sampled brush
        f.extend_from_slice(&2u16.to_be_bytes());
        f.extend_from_slice(&(body.len() as u32).to_be_bytes());
        f.extend_from_slice(&body);
        f
    }

    #[test]
    fn synthetic_v1_raw_round_trip() {
        let pixels: Vec<u8> = (0..12).map(|i| (i * 20) as u8).collect();
        let f = parse(&synth_v12(1, 4, 3, &pixels, None)).unwrap();
        assert_eq!(
            (f.version, f.subversion, f.skipped_computed, f.skipped_other),
            (1, 0, 1, 0)
        );
        assert_eq!(f.tips.len(), 1);
        let t = &f.tips[0];
        assert_eq!((t.w, t.h), (4, 3));
        assert_eq!(t.alpha, pixels);
        assert_eq!(t.name, "Sampled Brush 1");
        assert!(t.uuid.is_empty());
        assert_eq!(
            (t.spacing, t.diameter, t.angle, t.roundness),
            (0.5, 4.0, 0.0, 1.0)
        );
    }

    #[test]
    fn synthetic_v2_rle_round_trip() {
        // 5x2: row 0 = run of 5 x 0xFF, row 1 = literal [1,2,3] + run of 2 x 9.
        let rows = vec![vec![0xFC, 0xFF], vec![0x02, 1, 2, 3, 0xFF, 9, 0x80]];
        let f = parse(&synth_v12(2, 5, 2, &[], Some(&rows))).unwrap();
        assert_eq!(f.tips.len(), 1);
        let t = &f.tips[0];
        assert_eq!(t.name, "Test Tip");
        assert_eq!((t.w, t.h), (5, 2));
        assert_eq!(t.alpha, vec![255, 255, 255, 255, 255, 1, 2, 3, 9, 9]);
        // Every truncation of the synthetic file must also be handled.
        let bytes = synth_v12(2, 5, 2, &[], Some(&rows));
        for cut in 0..bytes.len() {
            let _ = parse(&bytes[..cut]);
        }
    }

    #[test]
    fn display_names() {
        assert_eq!(
            display_name("$$$/Presets/Brushes/SoftRound=Soft Round"),
            "Soft Round"
        );
        assert_eq!(display_name("Chalk 23"), "Chalk 23");
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse(&[]).is_err());
        assert!(parse(&[0, 6, 0, 2, b'X']).is_err());
        assert!(parse(&[0, 99]).is_err());
    }
}
