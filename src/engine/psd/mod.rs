//! Photoshop PSD/PSB structural codec, limited to Bitmap and 8-bit channels.
//!
//! Length-delimited unknown payloads are immutable views into one shared source
//! buffer. Re-encoding preserves their order, contents and padding, including
//! payloads inside blocks the document mapper does not understand.
//! Format: <https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/>.

pub(crate) mod compression;
pub(crate) mod descriptor;
pub(crate) mod engine_data;
pub(crate) mod inspect;
pub(crate) mod mapping;
pub(crate) mod text;
pub(crate) mod masks;

use std::{
    fmt,
    fs::File,
    io::{Read, Write},
    ops::Range,
    path::Path,
    sync::Arc,
};

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub(crate) enum Error {
    Truncated(&'static str),
    Invalid(&'static str),
    Unsupported(&'static str),
    Limit(&'static str),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated(what) => write!(f, "Truncated PSD {what}"),
            Self::Invalid(what) => write!(f, "Invalid PSD {what}"),
            Self::Unsupported(what) => write!(f, "Unsupported PSD {what}"),
            Self::Limit(what) => write!(f, "PSD exceeds limit: {what}"),
            Self::Io(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}
impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub(crate) struct Limits {
    pub(crate) max_file_bytes: usize,
    pub(crate) max_decoded_bytes: usize,
    pub(crate) max_layers: usize,
    pub(crate) max_blocks: usize,
}
impl Default for Limits {
    fn default() -> Self {
        // The 268 MP RGBA8 benchmark needs just over 1 GiB for four planes.
        // Callers can lower these bounds for untrusted/batch workloads.
        Self {
            max_file_bytes: 8usize << 30,
            max_decoded_bytes: 2usize << 30,
            max_layers: 32767,
            max_blocks: 100_000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Version {
    Psd,
    Psb,
}
impl Version {
    fn wide(self) -> bool {
        self == Self::Psb
    }
    fn number(self) -> u16 {
        if self.wide() { 2 } else { 1 }
    }
}

/// A cheap immutable slice of the source file; replacing one payload does not
/// duplicate or mutate the bytes owned by another layer or history snapshot.
#[derive(Clone, Default)]
pub(crate) struct RawBytes {
    source: Arc<Vec<u8>>,
    range: Range<usize>,
}
impl RawBytes {
    pub(crate) fn from_vec(bytes: Vec<u8>) -> Self {
        let end = bytes.len();
        let capacity = bytes.capacity();
        let source = Arc::new(bytes);
        let source = if capacity == 0 {
            source
        } else {
            super::memory::global().track_shared(source, capacity as u64)
        };
        Self {
            source,
            range: 0..end,
        }
    }
    pub(crate) fn empty() -> Self {
        Self::default()
    }
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.source[self.range.clone()]
    }
    pub(crate) fn len(&self) -> usize {
        self.range.len()
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
}
impl fmt::Debug for RawBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawBytes")
            .field("length", &self.len())
            .finish()
    }
}
impl PartialEq for RawBytes {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}
impl Eq for RawBytes {}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct Header {
    pub(crate) version: Version,
    pub(crate) channels: u16,
    pub(crate) height: u32,
    pub(crate) width: u32,
    pub(crate) depth: u16,
    pub(crate) color_mode: u16,
}
impl Header {
    pub(crate) fn rgb(width: u32, height: u32, channels: u16, version: Version) -> Self {
        Self {
            version,
            channels,
            height,
            width,
            depth: 8,
            color_mode: 3,
        }
    }
    fn validate(&self) -> Result<()> {
        let max_side = if self.version.wide() { 300_000 } else { 30_000 };
        if self.width == 0 || self.height == 0 || self.width > max_side || self.height > max_side {
            return Err(Error::Invalid("header dimensions"));
        }
        if !(1..=56).contains(&self.channels) {
            return Err(Error::Invalid("header channel count"));
        }
        if !matches!(self.color_mode,0..=4|7..=9) {
            return Err(Error::Unsupported("color mode"));
        }
        if !matches!((self.color_mode,self.depth),(0,1)|(1..=4|7..=9,8)) {
            return Err(Error::Unsupported(
                "depth or color-mode/depth combination (Phase 2 is 8-bit)",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct ImageResource {
    pub(crate) signature: [u8; 4],
    pub(crate) id: u16,
    /// Pascal string including its length byte and even-byte padding.
    pub(crate) name: RawBytes,
    pub(crate) data: RawBytes,
    pub(crate) padding: RawBytes,
}
impl ImageResource {
    pub(crate) fn new(id: u16, data: Vec<u8>) -> Self {
        let padding = RawBytes::from_vec(vec![0; data.len() % 2]);
        Self {
            signature: *b"8BIM",
            id,
            name: RawBytes::from_vec(vec![0, 0]),
            data: RawBytes::from_vec(data),
            padding,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct AdditionalInfo {
    pub(crate) signature: [u8; 4],
    pub(crate) key: [u8; 4],
    pub(crate) data: RawBytes,
    /// Global blocks have external alignment; layer blocks include their
    /// even-byte padding in the advertised data length.
    pub(crate) padding: RawBytes,
}
impl AdditionalInfo {
    pub(crate) fn new(key: [u8; 4], mut data: Vec<u8>) -> Self {
        if data.len() % 2 != 0 {
            data.push(0);
        }
        Self {
            signature: *b"8BIM",
            key,
            data: RawBytes::from_vec(data),
            padding: RawBytes::empty(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct ChannelData {
    pub(crate) id: i16,
    pub(crate) encoded: RawBytes,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct LayerRecord {
    /// Top, left, bottom, right, in document coordinates.
    pub(crate) bounds: [i32; 4],
    pub(crate) channels: Vec<ChannelData>,
    pub(crate) blend_mode: [u8; 4],
    pub(crate) opacity: u8,
    pub(crate) clipping: u8,
    pub(crate) flags: u8,
    pub(crate) filler: u8,
    pub(crate) mask: RawBytes,
    pub(crate) blending_ranges: RawBytes,
    /// Pascal string including its length byte and four-byte padding.
    pub(crate) name: RawBytes,
    pub(crate) additional_info: Vec<AdditionalInfo>,
    pub(crate) extra_tail: RawBytes,
}
impl LayerRecord {
    pub(crate) fn new(bounds: [i32; 4], name: &str) -> Self {
        let mut layer = Self {
            bounds,
            channels: Vec::new(),
            blend_mode: *b"norm",
            opacity: 255,
            clipping: 0,
            flags: 8,
            filler: 0,
            mask: RawBytes::empty(),
            blending_ranges: RawBytes::empty(),
            name: RawBytes::empty(),
            additional_info: Vec::new(),
            extra_tail: RawBytes::empty(),
        };
        layer.set_name(name);
        layer
    }
    pub(crate) fn name(&self) -> String {
        if let Some(block) = self
            .additional_info
            .iter()
            .find(|block| block.key == *b"luni")
        {
            let bytes = block.data.as_slice();
            if bytes.len() >= 4 {
                let count = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
                if let Some(end) = count.checked_mul(2).and_then(|x| x.checked_add(4)) {
                    if let Some(text) = bytes.get(4..end) {
                        let units: Vec<u16> = text
                            .chunks_exact(2)
                            .map(|p| u16::from_be_bytes([p[0], p[1]]))
                            .collect();
                        return String::from_utf16_lossy(&units)
                            .trim_end_matches('\0')
                            .to_owned();
                    }
                }
            }
        }
        let bytes = self.name.as_slice();
        let length = bytes.first().copied().unwrap_or(0) as usize;
        String::from_utf8_lossy(bytes.get(1..1 + length).unwrap_or(&[])).into_owned()
    }
    pub(crate) fn set_name(&mut self, name: &str) {
        let legacy: Vec<u8> = name
            .chars()
            .take(255)
            .map(|c| if c.is_ascii() { c as u8 } else { b'?' })
            .collect();
        let mut pascal = vec![legacy.len() as u8];
        pascal.extend_from_slice(&legacy);
        pascal.resize(pascal.len().next_multiple_of(4), 0);
        self.name = RawBytes::from_vec(pascal);
        let units: Vec<u16> = name.encode_utf16().collect();
        let mut unicode = Vec::with_capacity(4 + units.len() * 2);
        unicode.extend_from_slice(&(units.len() as u32).to_be_bytes());
        for unit in units {
            unicode.extend_from_slice(&unit.to_be_bytes());
        }
        let new = AdditionalInfo::new(*b"luni", unicode);
        if let Some(block) = self
            .additional_info
            .iter_mut()
            .find(|block| block.key == *b"luni")
        {
            *block = new;
        } else {
            self.additional_info.push(new);
        }
    }
    pub(crate) fn dimensions(&self) -> Result<(u32, u32)> {
        let width = i64::from(self.bounds[3]) - i64::from(self.bounds[1]);
        let height = i64::from(self.bounds[2]) - i64::from(self.bounds[0]);
        Ok((
            u32::try_from(width).map_err(|_| Error::Invalid("layer bounds"))?,
            u32::try_from(height).map_err(|_| Error::Invalid("layer bounds"))?,
        ))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct LayerMaskInfo {
    pub(crate) present: bool,
    pub(crate) layer_info_present: bool,
    pub(crate) merged_alpha: bool,
    pub(crate) layers: Vec<LayerRecord>,
    pub(crate) layer_info_tail: RawBytes,
    pub(crate) global_mask: Option<RawBytes>,
    pub(crate) additional_info: Vec<AdditionalInfo>,
    pub(crate) tail: RawBytes,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct PsdDocument {
    pub(crate) header: Header,
    pub(crate) color_mode_data: RawBytes,
    pub(crate) resources: Vec<ImageResource>,
    pub(crate) layer_mask: LayerMaskInfo,
    pub(crate) image_data: RawBytes,
}
impl PsdDocument {
    pub(crate) fn new(header: Header, image_data: Vec<u8>) -> Self {
        Self {
            header,
            color_mode_data: RawBytes::empty(),
            resources: Vec::new(),
            layer_mask: LayerMaskInfo::default(),
            image_data: RawBytes::from_vec(image_data),
        }
    }
    pub(crate) fn read(path: &Path, limits: &Limits) -> Result<Self> {
        let mut file = File::open(path)?;
        let size =
            usize::try_from(file.metadata()?.len()).map_err(|_| Error::Limit("file size"))?;
        if size > limits.max_file_bytes {
            return Err(Error::Limit("file size"));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| Error::Limit("source allocation"))?;
        Read::by_ref(&mut file)
            .take(limits.max_file_bytes as u64 + 1)
            .read_to_end(&mut bytes)?;
        Self::parse(bytes, limits)
    }
    pub(crate) fn parse(bytes: Vec<u8>, limits: &Limits) -> Result<Self> {
        if bytes.len() > limits.max_file_bytes {
            return Err(Error::Limit("file size"));
        }
        let capacity = bytes.capacity();
        let source = super::memory::global().track_shared(Arc::new(bytes), capacity as u64);
        let mut reader = Reader::new(&source, 0..source.len());
        if reader.array::<4>()? != *b"8BPS" {
            return Err(Error::Invalid("signature"));
        }
        let version = match reader.u16()? {
            1 => Version::Psd,
            2 => Version::Psb,
            _ => return Err(Error::Unsupported("version")),
        };
        if reader.array::<6>()? != [0; 6] {
            return Err(Error::Invalid("reserved header bytes"));
        }
        let header = Header {
            version,
            channels: reader.u16()?,
            height: reader.u32()?,
            width: reader.u32()?,
            depth: reader.u16()?,
            color_mode: reader.u16()?,
        };
        header.validate()?;
        if version == Version::Psd && source.len() > 2usize << 30 {
            return Err(Error::Limit("PSD 2 GB size; use PSB"));
        }
        let color_mode_data = reader.sized(false)?;
        if header.color_mode == 2 && color_mode_data.len() != 768 {
            return Err(Error::Invalid("indexed palette length"));
        }
        let resource_bytes = reader.sized(false)?;
        let mut resource_reader = Reader::blob(&resource_bytes);
        let mut resources = Vec::new();
        let mut blocks = 0usize;
        while resource_reader.remaining() > 0 {
            bump_blocks(&mut blocks, limits)?;
            let signature = resource_reader.array()?;
            let id = resource_reader.u16()?;
            let name = resource_reader.pascal(2)?;
            let data = resource_reader.sized(false)?;
            let padding = resource_reader.raw(data.len() % 2)?;
            resources.try_reserve(1).map_err(|_|Error::Limit("resource table allocation"))?;
            resources.push(ImageResource {
                signature,
                id,
                name,
                data,
                padding,
            });
        }
        let mask_bytes = reader.sized(version.wide())?;
        let layer_mask = parse_layer_mask(&mask_bytes, version, limits, &mut blocks)?;
        let image_data = reader.raw(reader.remaining())?;
        if image_data.len() < 2 {
            return Err(Error::Truncated("composite compression marker"));
        }
        let compression = u16::from_be_bytes([image_data.as_slice()[0], image_data.as_slice()[1]]);
        if compression > 3 {
            return Err(Error::Unsupported("composite compression"));
        }
        Ok(Self {
            header,
            color_mode_data,
            resources,
            layer_mask,
            image_data,
        })
    }
    pub(crate) fn decode_composite(&self, limits: &Limits) -> Result<Vec<u8>> {
        let rows = self
            .header
            .height
            .checked_mul(u32::from(self.header.channels))
            .ok_or(Error::Limit("composite rows"))?;
        compression::decode(
            self.image_data.as_slice(),
            self.header.width,
            rows,
            self.header.depth,
            self.header.version,
            limits,
        )
    }
    pub(crate) fn encode(&self, limits: &Limits) -> Result<Vec<u8>> {
        self.header.validate()?;
        let mut writer = Writer::new(limits.max_file_bytes.min(
            if self.header.version == Version::Psd {
                2usize << 30
            } else {
                usize::MAX
            },
        ));
        writer.bytes(b"8BPS")?;
        writer.bytes(&self.header.version.number().to_be_bytes())?;
        writer.bytes(&[0; 6])?;
        writer.bytes(&self.header.channels.to_be_bytes())?;
        writer.bytes(&self.header.height.to_be_bytes())?;
        writer.bytes(&self.header.width.to_be_bytes())?;
        writer.bytes(&self.header.depth.to_be_bytes())?;
        writer.bytes(&self.header.color_mode.to_be_bytes())?;
        writer.sized(false, |w| w.bytes(self.color_mode_data.as_slice()))?;
        writer.sized(false, |w| {
            for resource in &self.resources {
                w.bytes(&resource.signature)?;
                w.bytes(&resource.id.to_be_bytes())?;
                validate_pascal(&resource.name, 2)?;
                w.bytes(resource.name.as_slice())?;
                w.sized(false, |w| w.bytes(resource.data.as_slice()))?;
                write_padding(w, &resource.padding, resource.data.len() % 2)?;
            }
            Ok(())
        })?;
        writer.sized(self.header.version.wide(), |w| {
            write_layer_mask(w, &self.layer_mask, self.header.version, limits)
        })?;
        writer.bytes(self.image_data.as_slice())?;
        Ok(writer.bytes)
    }
    /// Same-directory atomic replacement. On error the old destination remains
    /// intact; only this operation's private temporary file is removed.
    pub(crate) fn write(&self, path: &Path, limits: &Limits) -> Result<()> {
        let bytes = self.encode(limits)?;
        let _output_memory = super::memory::global().track_cpu(bytes.capacity() as u64);
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let name = path
            .file_name()
            .ok_or(Error::Invalid("output filename"))?
            .to_string_lossy();
        let mut temporary = None;
        for nonce in 0..100 {
            let candidate = parent.join(format!(".{name}.pf-{}-{nonce}.tmp", std::process::id()));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(file) => {
                    temporary = Some((candidate, file));
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        let (temporary_path, mut file) =
            temporary.ok_or(Error::Invalid("temporary filename exhaustion"))?;
        let result = (|| {
            file.write_all(&bytes)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&temporary_path, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary_path);
        }
        result
    }
}

fn bump_blocks(count: &mut usize, limits: &Limits) -> Result<()> {
    *count += 1;
    if *count > limits.max_blocks {
        return Err(Error::Limit("block count"));
    }
    Ok(())
}

fn parse_layer_mask(
    bytes: &RawBytes,
    version: Version,
    limits: &Limits,
    blocks: &mut usize,
) -> Result<LayerMaskInfo> {
    if bytes.is_empty() {
        return Ok(LayerMaskInfo::default());
    }
    let mut reader = Reader::blob(bytes);
    let info = reader.sized(version.wide())?;
    // The declared length includes even-byte padding. Accepting an odd length
    // would make the writer add a byte and change an ostensibly preserved file.
    if info.len() % 2 != 0 {
        return Err(Error::Invalid("odd Layer Info length"));
    }
    let mut result = LayerMaskInfo {
        present: true,
        layer_info_present: !info.is_empty(),
        ..LayerMaskInfo::default()
    };
    if !info.is_empty() {
        let mut r = Reader::blob(&info);
        let signed_count = r.i16()?;
        let count = signed_count.unsigned_abs() as usize;
        if count > limits.max_layers {
            return Err(Error::Limit("layer count"));
        }
        // A short layer record is at least 34 bytes before extra fields.
        if count > r.remaining() / 34 {
            return Err(Error::Truncated("layer records"));
        }
        result.merged_alpha = signed_count < 0;
        let mut channel_lengths = Vec::new();
        channel_lengths.try_reserve_exact(count).map_err(|_|Error::Limit("channel length table allocation"))?;
        result.layers.try_reserve_exact(count).map_err(|_|Error::Limit("layer table allocation"))?;
        for _ in 0..count {
            let bounds = [r.i32()?, r.i32()?, r.i32()?, r.i32()?];
            let channel_count = r.u16()? as usize;
            if channel_count > 56 {
                return Err(Error::Limit("layer channels"));
            }
            let mut channels = Vec::new();
            let mut lengths = Vec::new();
            channels.try_reserve_exact(channel_count).map_err(|_|Error::Limit("channel table allocation"))?;
            lengths.try_reserve_exact(channel_count).map_err(|_|Error::Limit("channel length allocation"))?;
            for _ in 0..channel_count {
                let id = r.i16()?;
                let length = r.length(version.wide())?;
                channels.push(ChannelData {
                    id,
                    encoded: RawBytes::empty(),
                });
                lengths.push(length);
            }
            if r.array::<4>()? != *b"8BIM" {
                return Err(Error::Invalid("blend mode signature"));
            }
            let blend_mode = r.array()?;
            let [opacity, clipping, flags, filler] = r.array()?;
            let extra = r.sized(false)?;
            let mut e = Reader::blob(&extra);
            let mask = e.sized(false)?;
            if !mask.is_empty() && mask.len() < 18 {
                return Err(Error::Invalid("layer mask length"));
            }
            let blending_ranges = e.sized(false)?;
            if blending_ranges.len() % 8 != 0 {
                return Err(Error::Invalid("blending ranges length"));
            }
            let name = e.pascal(4)?;
            let (additional_info, extra_tail) =
                parse_additional(&mut e, version, 1, limits, blocks)?;
            let layer = LayerRecord {
                bounds,
                channels,
                blend_mode,
                opacity,
                clipping,
                flags,
                filler,
                mask,
                blending_ranges,
                name,
                additional_info,
                extra_tail,
            };
            layer.dimensions()?;
            result.layers.push(layer);
            channel_lengths.push(lengths);
        }
        for (layer, lengths) in result.layers.iter_mut().zip(channel_lengths) {
            for (channel, length) in layer.channels.iter_mut().zip(lengths) {
                channel.encoded = r.raw(length)?;
                // Empty channels are legal for divider and adjustment layers.
                if length == 1 {
                    return Err(Error::Truncated("channel compression marker"));
                }
                if length >= 2 {
                    let code = u16::from_be_bytes([
                        channel.encoded.as_slice()[0],
                        channel.encoded.as_slice()[1],
                    ]);
                    if code > 3 {
                        return Err(Error::Unsupported("layer compression"));
                    }
                }
            }
        }
        result.layer_info_tail = r.raw(r.remaining())?;
    }
    // Photoshop sometimes omits the empty global-mask length before tags.
    if reader.remaining() >= 4 && !matches!(reader.peek4(), Some(b"8BIM" | b"8B64")) {
        result.global_mask = Some(reader.sized(false)?);
    }
    (result.additional_info, result.tail) =
        parse_additional(&mut reader, version, 4, limits, blocks)?;
    Ok(result)
}

fn wide_tag(version: Version, key: [u8; 4], signature: [u8; 4]) -> bool {
    // Current Photoshop also uses 8B64 for keys beyond Adobe's published
    // PSB list (for example cinf). The signature explicitly selects u64;
    // retain the key fallback for wide PSB records carrying 8BIM instead.
    signature == *b"8B64" || version.wide()
        && matches!(
            &key,
            b"LMsk"
                | b"Lr16"
                | b"Lr32"
                | b"Layr"
                | b"Mt16"
                | b"Mt32"
                | b"Mtrn"
                | b"Alph"
                | b"FMsk"
                | b"lnk2"
                | b"FEid"
                | b"FXid"
                | b"PxSD"
        )
}

fn parse_additional(
    reader: &mut Reader<'_>,
    version: Version,
    alignment: usize,
    limits: &Limits,
    count: &mut usize,
) -> Result<(Vec<AdditionalInfo>, RawBytes)> {
    let mut blocks = Vec::new();
    while reader.remaining() >= 4 && matches!(reader.peek4(), Some(b"8BIM" | b"8B64")) {
        bump_blocks(count, limits)?;
        let signature = reader.array()?;
        let key = reader.array()?;
        let data = reader.sized(wide_tag(version, key, signature))?;
        let padding = reader.raw((alignment - data.len() % alignment) % alignment)?;
        blocks.try_reserve(1).map_err(|_|Error::Limit("additional block table allocation"))?;
        blocks.push(AdditionalInfo {
            signature,
            key,
            data,
            padding,
        });
    }
    // Unrecognised section tails stay opaque. Never scan them for a coincidental
    // signature: without a framing contract their internal boundaries are unknown.
    let tail = reader.raw(reader.remaining())?;
    Ok((blocks, tail))
}

fn write_layer_mask(
    writer: &mut Writer,
    info: &LayerMaskInfo,
    version: Version,
    limits: &Limits,
) -> Result<()> {
    if !info.present {
        if !info.layers.is_empty() {
            return Err(Error::Invalid("layers in absent layer section"));
        }
        return Ok(());
    }
    if info.layers.len() > limits.max_layers {
        return Err(Error::Limit("layer count"));
    }
    writer.sized(version.wide(), |w| {
        let body_start = w.bytes.len();
        if !info.layer_info_present {
            if !info.layers.is_empty() {
                return Err(Error::Invalid("layers in absent layer info"));
            }
            return Ok(());
        }
        let count =
            i16::try_from(info.layers.len()).map_err(|_| Error::Limit("signed layer count"))?;
        w.bytes(&(if info.merged_alpha { -count } else { count }).to_be_bytes())?;
        for layer in &info.layers {
            layer.dimensions()?;
            for coord in layer.bounds {
                w.bytes(&coord.to_be_bytes())?;
            }
            let count = u16::try_from(layer.channels.len())
                .map_err(|_| Error::Limit("layer channel count"))?;
            if count > 56 {
                return Err(Error::Limit("layer channel count"));
            }
            w.bytes(&count.to_be_bytes())?;
            for channel in &layer.channels {
                w.bytes(&channel.id.to_be_bytes())?;
                w.length(channel.encoded.len(), version.wide())?;
            }
            w.bytes(b"8BIM")?;
            w.bytes(&layer.blend_mode)?;
            w.bytes(&[layer.opacity, layer.clipping, layer.flags, layer.filler])?;
            w.sized(false, |w| {
                w.sized(false, |w| w.bytes(layer.mask.as_slice()))?;
                w.sized(false, |w| w.bytes(layer.blending_ranges.as_slice()))?;
                validate_pascal(&layer.name, 4)?;
                w.bytes(layer.name.as_slice())?;
                write_additional(w, &layer.additional_info, version, 1)?;
                w.bytes(layer.extra_tail.as_slice())
            })?;
        }
        for layer in &info.layers {
            for channel in &layer.channels {
                w.bytes(channel.encoded.as_slice())?;
            }
        }
        w.bytes(info.layer_info_tail.as_slice())?;
        if (w.bytes.len() - body_start) % 2 != 0 {
            w.bytes(&[0])?;
        }
        Ok(())
    })?;
    if let Some(mask) = &info.global_mask {
        writer.sized(false, |w| w.bytes(mask.as_slice()))?;
    }
    write_additional(writer, &info.additional_info, version, 4)?;
    writer.bytes(info.tail.as_slice())
}

fn write_additional(
    writer: &mut Writer,
    blocks: &[AdditionalInfo],
    version: Version,
    alignment: usize,
) -> Result<()> {
    for block in blocks {
        writer.bytes(&block.signature)?;
        writer.bytes(&block.key)?;
        writer.sized(wide_tag(version, block.key, block.signature), |w| {
            w.bytes(block.data.as_slice())
        })?;
        write_padding(
            writer,
            &block.padding,
            (alignment - block.data.len() % alignment) % alignment,
        )?;
    }
    Ok(())
}
fn write_padding(writer: &mut Writer, padding: &RawBytes, required: usize) -> Result<()> {
    if padding.len() == required {
        writer.bytes(padding.as_slice())
    } else if padding.is_empty() {
        writer.bytes(&[0; 4][..required])
    } else {
        Err(Error::Invalid("block padding length"))
    }
}
fn validate_pascal(bytes: &RawBytes, alignment: usize) -> Result<()> {
    let len = bytes
        .as_slice()
        .first()
        .copied()
        .ok_or(Error::Invalid("Pascal name"))? as usize
        + 1;
    if len.next_multiple_of(alignment) != bytes.len() {
        return Err(Error::Invalid("Pascal name length"));
    }
    Ok(())
}

struct Reader<'a> {
    source: &'a Arc<Vec<u8>>,
    position: usize,
    end: usize,
}
impl<'a> Reader<'a> {
    fn new(source: &'a Arc<Vec<u8>>, range: Range<usize>) -> Self {
        Self {
            source,
            position: range.start,
            end: range.end,
        }
    }
    fn blob(bytes: &'a RawBytes) -> Self {
        Self::new(&bytes.source, bytes.range.clone())
    }
    fn remaining(&self) -> usize {
        self.end - self.position
    }
    fn peek4(&self) -> Option<&[u8]> {
        self.source
            .get(self.position..self.position.checked_add(4)?)
            .filter(|_| self.remaining() >= 4)
    }
    fn raw(&mut self, count: usize) -> Result<RawBytes> {
        let end = self
            .position
            .checked_add(count)
            .filter(|&x| x <= self.end)
            .ok_or(Error::Truncated("section payload"))?;
        let bytes = RawBytes {
            source: Arc::clone(self.source),
            range: self.position..end,
        };
        self.position = end;
        Ok(bytes)
    }
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let bytes = self.raw(N)?;
        let mut array = [0; N];
        array.copy_from_slice(bytes.as_slice());
        Ok(array)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.array()?))
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_be_bytes(self.array()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.array()?))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.array()?))
    }
    fn length(&mut self, wide: bool) -> Result<usize> {
        let len = if wide {
            u64::from_be_bytes(self.array()?)
        } else {
            u64::from(self.u32()?)
        };
        usize::try_from(len).map_err(|_| Error::Limit("section length"))
    }
    fn sized(&mut self, wide: bool) -> Result<RawBytes> {
        let len = self.length(wide)?;
        self.raw(len)
    }
    fn pascal(&mut self, alignment: usize) -> Result<RawBytes> {
        let len = self
            .source
            .get(self.position)
            .copied()
            .ok_or(Error::Truncated("Pascal name"))? as usize
            + 1;
        self.raw(len.next_multiple_of(alignment))
    }
}

struct Writer {
    bytes: Vec<u8>,
    limit: usize,
    memory: Vec<super::memory::Reservation>,
}
impl Writer {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            memory: Vec::new(),
        }
    }
    fn bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let end = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .filter(|&x| x <= self.limit)
            .ok_or(Error::Limit("encoded file size"))?;
        let capacity = self.bytes.capacity();
        if end > capacity {
            let target = end.max(capacity.saturating_mul(2)).min(self.limit);
            self.bytes
                .try_reserve_exact(target - self.bytes.len())
                .map_err(|_| Error::Limit("output allocation"))?;
        }
        if self.bytes.capacity() > capacity {
            self.memory
                .push(super::memory::global().track_cpu((self.bytes.capacity() - capacity) as u64));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn length(&mut self, len: usize, wide: bool) -> Result<()> {
        if wide {
            self.bytes(&(len as u64).to_be_bytes())
        } else {
            self.bytes(
                &u32::try_from(len)
                    .map_err(|_| Error::Limit("32-bit section length; use PSB"))?
                    .to_be_bytes(),
            )
        }
    }
    fn sized(&mut self, wide: bool, write: impl FnOnce(&mut Self) -> Result<()>) -> Result<()> {
        let marker = self.bytes.len();
        self.length(0, wide)?;
        let start = self.bytes.len();
        write(self)?;
        let length = self.bytes.len() - start;
        if wide {
            self.bytes[marker..start].copy_from_slice(&(length as u64).to_be_bytes());
        } else {
            self.bytes[marker..start].copy_from_slice(
                &u32::try_from(length)
                    .map_err(|_| Error::Limit("32-bit section length; use PSB"))?
                    .to_be_bytes(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compression::Method;

    fn example(version: Version) -> PsdDocument {
        let limits = Limits::default();
        let composite = compression::encode(
            &[10, 20, 30, 40, 50, 60],
            2,
            3,
            8,
            version,
            Method::Raw,
            &limits,
        )
        .unwrap();
        let mut document = PsdDocument::new(Header::rgb(2, 1, 3, version), composite);
        let mut layer = LayerRecord::new([0, 0, 1, 2], "Original α");
        for (id, bytes) in [
            (0, [10, 20]),
            (1, [30, 40]),
            (2, [50, 60]),
            (-1, [255, 255]),
        ] {
            layer.channels.push(ChannelData {
                id,
                encoded: RawBytes::from_vec(
                    compression::encode(&bytes, 2, 1, 8, version, Method::PackBits, &limits)
                        .unwrap(),
                ),
            });
        }
        layer.additional_info.push(AdditionalInfo {
            signature: *b"8BIM",
            key: *b"zZZz",
            data: RawBytes::from_vec(vec![1, 2, 3, 4, 0xfe, 0xfd]),
            padding: RawBytes::empty(),
        });
        document.resources.push(ImageResource {
            signature: *b"8BIM",
            id: 4999,
            name: RawBytes::from_vec(vec![2, b'a', b'b', 0xaa]),
            data: RawBytes::from_vec(vec![23, 42, 57]),
            padding: RawBytes::from_vec(vec![0xcc]),
        });
        document.layer_mask = LayerMaskInfo {
            present: true,
            layer_info_present: true,
            merged_alpha: false,
            layers: vec![layer],
            global_mask: Some(RawBytes::empty()),
            ..LayerMaskInfo::default()
        };
        document.layer_mask.additional_info.push(AdditionalInfo {
            signature: *b"8BIM",
            key: *b"test",
            data: RawBytes::from_vec(vec![6, 7]),
            padding: RawBytes::from_vec(vec![0xab, 0xcd]),
        });
        document
    }

    #[test]
    fn psd_and_psb_preserve_all_structural_bytes_including_padding() {
        for version in [Version::Psd, Version::Psb] {
            let bytes = example(version).encode(&Limits::default()).unwrap();
            let parsed = PsdDocument::parse(bytes.clone(), &Limits::default()).unwrap();
            assert_eq!(parsed.encode(&Limits::default()).unwrap(), bytes);
            assert_eq!(
                parsed.decode_composite(&Limits::default()).unwrap(),
                [10, 20, 30, 40, 50, 60]
            );
            assert_eq!(parsed.layer_mask.layers[0].name(), "Original α");
        }
    }

    #[test]
    fn odd_layer_info_length_is_rejected_without_reframing_following_sections() {
        let limits = Limits::default();
        for version in [Version::Psd, Version::Psb] {
            let mut document = PsdDocument::new(
                Header::rgb(1, 1, 3, version),
                vec![0, 0, 10, 20, 30],
            );
            document.layer_mask = LayerMaskInfo {
                present: true,
                layer_info_present: true,
                global_mask: Some(RawBytes::empty()),
                ..LayerMaskInfo::default()
            };
            let valid = document.encode(&limits).unwrap();
            assert_eq!(
                PsdDocument::parse(valid.clone(), &limits)
                    .unwrap()
                    .encode(&limits)
                    .unwrap(),
                valid
            );

            // Empty colour/resource sections put the enclosing length at 34.
            // Keep its framing correct while changing Layer Info from a valid
            // two-byte zero layer count to an invalid three-byte section.
            let width = if version.wide() { 8 } else { 4 };
            let length_offset = 34 + width;
            let body_offset = length_offset + width;
            let mut malformed = valid;
            malformed.insert(body_offset + 2, 0xa7);
            let outer_length = (width + 3 + 4) as u64;
            malformed[34..34 + width]
                .copy_from_slice(&outer_length.to_be_bytes()[8 - width..]);
            malformed[length_offset..body_offset]
                .copy_from_slice(&3_u64.to_be_bytes()[8 - width..]);
            assert!(matches!(
                PsdDocument::parse(malformed, &limits),
                Err(Error::Invalid("odd Layer Info length"))
            ));
        }
    }

    #[test]
    fn unrelated_name_and_pixel_edit_keep_unknown_blocks_exact() {
        let limits = Limits::default();
        let mut original = example(Version::Psd);
        let mut untouched = original.layer_mask.layers[0].clone();
        untouched.set_name("Untouched");
        untouched
            .additional_info
            .push(AdditionalInfo::new(*b"TySh", vec![0, 1, 4, 7, 0, 0, 0, 6]));
        original.layer_mask.layers.push(untouched);
        let original = PsdDocument::parse(original.encode(&limits).unwrap(), &limits).unwrap();
        let resources = original.resources.clone();
        let unknown = original.layer_mask.layers[0].additional_info[1].clone();
        let second = original.layer_mask.layers[1].clone();
        let mut edited = original.clone();
        edited.layer_mask.layers[0].set_name("Edited 😀");
        edited.layer_mask.layers[0].channels[0].encoded = RawBytes::from_vec(
            compression::encode(
                &[200, 201],
                2,
                1,
                8,
                Version::Psd,
                Method::ZipPrediction,
                &limits,
            )
            .unwrap(),
        );
        let parsed = PsdDocument::parse(edited.encode(&limits).unwrap(), &limits).unwrap();
        assert_eq!(parsed.resources, resources);
        assert_eq!(parsed.layer_mask.layers[0].additional_info[1], unknown);
        assert_eq!(parsed.layer_mask.layers[1], second);
        assert_eq!(parsed.layer_mask.layers[0].name(), "Edited 😀");
        assert_eq!(
            compression::decode(
                parsed.layer_mask.layers[0].channels[0].encoded.as_slice(),
                2,
                1,
                8,
                Version::Psd,
                &limits
            )
            .unwrap(),
            [200, 201]
        );
    }

    #[test]
    fn every_truncated_prefix_fails_parse_or_composite_decode() {
        for version in [Version::Psd, Version::Psb] {
            let bytes = example(version).encode(&Limits::default()).unwrap();
            for end in 0..bytes.len() {
                let result = PsdDocument::parse(bytes[..end].to_vec(), &Limits::default())
                    .and_then(|doc| doc.decode_composite(&Limits::default()));
                assert!(
                    result.is_err(),
                    "accepted truncated prefix at {end}, {version:?}"
                );
            }
        }
    }

    #[test]
    fn malformed_section_length_is_rejected_without_allocation() {
        let mut bytes = example(Version::Psd).encode(&Limits::default()).unwrap();
        bytes[26..30].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(
            PsdDocument::parse(bytes, &Limits::default()),
            Err(Error::Truncated(_))
        ));
    }

    #[test]
    fn limits_reject_file_layer_and_block_counts() {
        let bytes = example(Version::Psd).encode(&Limits::default()).unwrap();
        for limits in [
            Limits {
                max_file_bytes: bytes.len() - 1,
                ..Limits::default()
            },
            Limits {
                max_layers: 0,
                ..Limits::default()
            },
            Limits {
                max_blocks: 0,
                ..Limits::default()
            },
        ] {
            assert!(matches!(
                PsdDocument::parse(bytes.clone(), &limits),
                Err(Error::Limit(_))
            ));
        }
        assert!(matches!(
            example(Version::Psd).encode(&Limits {
                max_file_bytes: 30,
                ..Limits::default()
            }),
            Err(Error::Limit(_))
        ));
    }

    #[test]
    fn unknown_duplicate_keys_keep_order_and_opaque_tail() {
        let mut document = example(Version::Psd);
        document.layer_mask.layers[0].additional_info.extend([
            AdditionalInfo::new(*b"same", vec![1, 2]),
            AdditionalInfo::new(*b"same", vec![3, 4]),
        ]);
        document.layer_mask.layers[0].extra_tail = RawBytes::from_vec(vec![0xa1, 0xb2, 0xc3]);
        let bytes = document.encode(&Limits::default()).unwrap();
        let parsed = PsdDocument::parse(bytes.clone(), &Limits::default()).unwrap();
        assert_eq!(parsed.encode(&Limits::default()).unwrap(), bytes);
        let keys: Vec<[u8; 4]> = parsed.layer_mask.layers[0]
            .additional_info
            .iter()
            .map(|b| b.key)
            .collect();
        assert_eq!(keys, vec![*b"luni", *b"zZZz", *b"same", *b"same"]);
    }

    #[test]
    fn psb_8bim_uses_eight_byte_lengths_only_for_specified_tag_keys() {
        let mut document = example(Version::Psb);
        for key in [
            *b"LMsk", *b"Lr16", *b"Lr32", *b"Layr", *b"Mt16", *b"Mt32", *b"Mtrn", *b"Alph",
            *b"FMsk", *b"lnk2", *b"FEid", *b"FXid", *b"PxSD", *b"zZZz",
        ] {
            document
                .layer_mask
                .additional_info
                .push(AdditionalInfo::new(key, vec![0, 1, 2, 3]));
        }
        let bytes = document.encode(&Limits::default()).unwrap();
        let parsed = PsdDocument::parse(bytes.clone(), &Limits::default()).unwrap();
        assert_eq!(parsed.encode(&Limits::default()).unwrap(), bytes);
        for key in [*b"LMsk", *b"Lr16", *b"lnk2"] {
            let marker = [
                b"8BIM".as_slice(),
                key.as_slice(),
                &[0, 0, 0, 0, 0, 0, 0, 4],
            ]
            .concat();
            assert!(bytes.windows(marker.len()).any(|window| window == marker));
        }
    }

    #[test]
    fn authored_cinf_8b64_and_8bim_lengths_preserve_descriptors_and_following_blocks() {
        // Identical 410-byte cinf payload captured from locally authored
        // Photoshop 2026 large-layered-24.psd, -150.psb and -268.psb. The
        // PSD uses 8BIM/u32; both PSBs use 8B64/u64. Construct wire headers
        // independently of our encoder: an encode/decode-only test missed
        // the old parser retaining the descriptor as an opaque section tail.
        const CINF: &[u8; 410] = &[
            0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6e, 0x75,
            0x6c, 0x6c, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x56, 0x72, 0x73, 0x6e, 0x4f, 0x62,
            0x6a, 0x63, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6e, 0x75, 0x6c, 0x6c,
            0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x05, 0x6d, 0x61, 0x6a, 0x6f, 0x72, 0x6c, 0x6f, 0x6e,
            0x67, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x05, 0x6d, 0x69, 0x6e, 0x6f, 0x72, 0x6c, 0x6f,
            0x6e, 0x67, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x66, 0x69, 0x78, 0x6c, 0x6f, 0x6e,
            0x67, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x70, 0x73, 0x56, 0x65, 0x72, 0x73, 0x69,
            0x6f, 0x6e, 0x4f, 0x62, 0x6a, 0x63, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x6e, 0x75, 0x6c, 0x6c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x05, 0x6d, 0x61, 0x6a, 0x6f,
            0x72, 0x6c, 0x6f, 0x6e, 0x67, 0x00, 0x00, 0x00, 0x1b, 0x00, 0x00, 0x00, 0x05, 0x6d, 0x69, 0x6e,
            0x6f, 0x72, 0x6c, 0x6f, 0x6e, 0x67, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x03, 0x66, 0x69,
            0x78, 0x6c, 0x6f, 0x6e, 0x67, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0b, 0x64, 0x65, 0x73,
            0x63, 0x72, 0x69, 0x70, 0x74, 0x69, 0x6f, 0x6e, 0x54, 0x45, 0x58, 0x54, 0x00, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x54, 0x45, 0x58, 0x54,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x45, 0x6e, 0x67, 0x6e, 0x65, 0x6e,
            0x75, 0x6d, 0x00, 0x00, 0x00, 0x00, 0x45, 0x6e, 0x67, 0x6e, 0x00, 0x00, 0x00, 0x0b, 0x63, 0x6f,
            0x6d, 0x70, 0x43, 0x6f, 0x72, 0x65, 0x47, 0x50, 0x55, 0x00, 0x00, 0x00, 0x11, 0x65, 0x6e, 0x61,
            0x62, 0x6c, 0x65, 0x43, 0x6f, 0x6d, 0x70, 0x43, 0x6f, 0x72, 0x65, 0x47, 0x50, 0x55, 0x65, 0x6e,
            0x75, 0x6d, 0x00, 0x00, 0x00, 0x06, 0x65, 0x6e, 0x61, 0x62, 0x6c, 0x65, 0x00, 0x00, 0x00, 0x07,
            0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x00, 0x00, 0x00, 0x15, 0x65, 0x6e, 0x61, 0x62, 0x6c,
            0x65, 0x43, 0x6f, 0x6d, 0x70, 0x43, 0x6f, 0x72, 0x65, 0x54, 0x68, 0x72, 0x65, 0x61, 0x64, 0x73,
            0x65, 0x6e, 0x75, 0x6d, 0x00, 0x00, 0x00, 0x06, 0x65, 0x6e, 0x61, 0x62, 0x6c, 0x65, 0x00, 0x00,
            0x00, 0x07, 0x66, 0x65, 0x61, 0x74, 0x75, 0x72, 0x65, 0x00, 0x00, 0x00, 0x12, 0x63, 0x6f, 0x6d,
            0x70, 0x43, 0x6f, 0x72, 0x65, 0x47, 0x50, 0x55, 0x53, 0x75, 0x70, 0x70, 0x6f, 0x72, 0x74, 0x65,
            0x6e, 0x75, 0x6d, 0x00, 0x00, 0x00, 0x06, 0x72, 0x65, 0x61, 0x73, 0x6f, 0x6e, 0x00, 0x00, 0x00,
            0x09, 0x73, 0x75, 0x70, 0x70, 0x6f, 0x72, 0x74, 0x65, 0x64,
        ];
        let limits = Limits::default();
        for (version, signature, length) in [
            (Version::Psd, *b"8BIM", 410_u32.to_be_bytes().to_vec()),
            (Version::Psb, *b"8B64", 410_u64.to_be_bytes().to_vec()),
            (Version::Psb, *b"8BIM", 410_u32.to_be_bytes().to_vec()),
        ] {
            let mut wire = signature.to_vec();
            wire.extend_from_slice(b"cinf");
            wire.extend(length);
            wire.extend_from_slice(CINF);
            wire.extend_from_slice(&[0, 0]);
            // A following unknown block and nonzero padding must not be
            // swallowed into the descriptor or mistaken for section tails.
            wire.extend_from_slice(b"8BIMnext\0\0\0\x03\x11\x22\x33\x99");
            wire.extend_from_slice(&[0xa1, 0xb2, 0xc3]);
            let bounded_end = wire.len();
            wire.extend_from_slice(b"8BIMoutside-the-section");
            let source = Arc::new(wire);
            let mut reader = Reader::new(&source, 0..bounded_end);
            let mut count = 0;
            let (blocks, tail) = parse_additional(&mut reader, version, 4, &limits, &mut count).unwrap();
            assert_eq!(count, 2);
            assert_eq!(blocks[0].signature, signature);
            assert_eq!(blocks[0].key, *b"cinf");
            assert_eq!(blocks[0].data.as_slice(), CINF);
            assert_eq!(blocks[0].padding.as_slice(), &[0, 0]);
            assert_eq!(blocks[1].key, *b"next");
            assert_eq!(blocks[1].data.as_slice(), &[0x11, 0x22, 0x33]);
            assert_eq!(blocks[1].padding.as_slice(), &[0x99]);
            assert_eq!(tail.as_slice(), &[0xa1, 0xb2, 0xc3]);
            assert_eq!(reader.remaining(), 0);
            let mut document = example(version);
            document.layer_mask.additional_info = blocks.clone();
            let rows = inspect::inspect(&document);
            let row = rows.iter().find(|row| row.key == "cinf").unwrap();
            assert_eq!(row.status, "typed", "{}", row.message);
            assert_eq!(row.descriptors, 1);
            let mut writer = Writer::new(limits.max_file_bytes);
            write_additional(&mut writer, &blocks, version, 4).unwrap();
            writer.bytes(tail.as_slice()).unwrap();
            assert_eq!(writer.bytes, source[..bounded_end]);
        }
    }

    #[test]
    fn explicit_8b64_lengths_cannot_escape_the_bounded_section() {
        for bytes in [
            b"8B64cinf\0\0\0\0".to_vec(),
            b"8B64cinf\0\0\0\0\0\0\x01\x9a\0\0\0\x10".to_vec(),
            [b"8B64new!".as_slice(), &u64::MAX.to_be_bytes()].concat(),
        ] {
            let source = Arc::new(bytes);
            let mut reader = Reader::new(&source, 0..source.len());
            assert!(parse_additional(&mut reader, Version::Psb, 4, &Limits::default(), &mut 0).is_err());
        }
    }

    #[test]
    fn color_modes_and_palette_survive_raw_round_trip() {
        for mode in [0, 1, 2, 3, 4, 7, 8, 9] {
            let mut document = PsdDocument::new(Header::rgb(1, 1, 1, Version::Psd), vec![0, 0, 42]);
            document.header.color_mode = mode;
            document.header.depth = if mode == 0 { 1 } else { 8 };
            if mode == 2 {
                document.color_mode_data = RawBytes::from_vec((0..768).map(|x| x as u8).collect());
            }
            if mode == 8 {
                document.color_mode_data = RawBytes::from_vec(vec![0x12, 0xab, 0xff, 0]);
            }
            let bytes = document.encode(&Limits::default()).unwrap();
            let parsed = PsdDocument::parse(bytes.clone(), &Limits::default()).unwrap();
            assert_eq!(parsed.encode(&Limits::default()).unwrap(), bytes);
        }
    }

    #[test]
    fn later_phase_depths_and_invalid_header_values_are_explicit_errors() {
        let original = example(Version::Psd).encode(&Limits::default()).unwrap();
        for depth in [0u16, 16, 32] {
            let mut bytes = original.clone();
            bytes[22..24].copy_from_slice(&depth.to_be_bytes());
            assert!(matches!(
                PsdDocument::parse(bytes, &Limits::default()),
                Err(Error::Unsupported(_))
            ));
        }
        for offset in [12usize, 14, 18] {
            let mut bytes = original.clone();
            let end = offset + if offset == 12 { 2 } else { 4 };
            bytes[offset..end].fill(0);
            assert!(PsdDocument::parse(bytes, &Limits::default()).is_err());
        }
    }

    #[test]
    fn malformed_additional_block_length_is_not_silently_preserved() {
        let mut bytes = example(Version::Psb).encode(&Limits::default()).unwrap();
        let position = bytes.windows(8).position(|b| b == b"8BIMzZZz").unwrap();
        bytes[position + 8..position + 12].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(PsdDocument::parse(bytes, &Limits::default()).is_err());
    }

    #[test]
    fn signed_layer_count_preserves_merged_transparency() {
        let mut document = example(Version::Psd);
        document.layer_mask.merged_alpha = true;
        document.header.channels = 4;
        document.image_data = RawBytes::from_vec(vec![0, 0, 10, 20, 30, 40, 50, 60, 255, 254]);
        let parsed = PsdDocument::parse(
            document.encode(&Limits::default()).unwrap(),
            &Limits::default(),
        )
        .unwrap();
        assert!(parsed.layer_mask.merged_alpha);
        assert_eq!(
            parsed.decode_composite(&Limits::default()).unwrap(),
            [10, 20, 30, 40, 50, 60, 255, 254]
        );
    }
}
