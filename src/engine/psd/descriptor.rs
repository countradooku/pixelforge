//! Ordered Photoshop ActionDescriptor codec.
//!
//! All wire distinctions are retained, including duplicate keys, compact versus
//! string IDs, invalid UTF-16, NaN payloads and noncanonical Boolean bytes. An
//! unknown OSType has no universal length field: only a caller-provided bounded
//! block can be preserved opaquely in that case; guessing its size is unsafe.
//!
//! Format: Adobe Photoshop File Formats, Actions / Descriptor structure:
//! <https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/>.
//! Adobe omits ObAr, UnFl and Pth layouts; those extensions follow the wire
//! structure independently documented by psd-tools and need Photoshop corpus
//! coverage in addition to the codec tests.

use std::fmt;

use anyhow::{Context, Result, ensure};

/// Limits apply to the complete descriptor, not independently to each child.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Limits {
    pub max_bytes: usize,
    pub max_allocated_bytes: usize,
    pub max_items: usize,
    pub max_depth: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_bytes: 64 << 20,
            max_allocated_bytes: 128 << 20,
            max_items: 1_000_000,
            max_depth: 64,
        }
    }
}

/// Both encodings of a four-byte ID are legal and must remain distinct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Id {
    Four([u8; 4]),
    String(Vec<u8>),
}

impl Id {
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::Four(value) => value,
            Self::String(value) => value,
        }
    }
}

/// Names retain UTF-16 units so unpaired surrogates are not replaced on save.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Class {
    pub name: Vec<u16>,
    pub id: Id,
}

impl Class {
    #[cfg(test)]
    pub fn new(name: &str, id: Id) -> Self {
        Self {
            name: name.encode_utf16().collect(),
            id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Entry {
    pub key: Id,
    pub value: Value,
}

impl Entry {
    pub fn new(key: Id, value: Value) -> Self {
        Self { key, value }
    }
}

/// Descriptor order and repeated keys carry through every edit.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Descriptor {
    pub class: Class,
    pub entries: Vec<Entry>,
}

impl Descriptor {
    #[cfg(test)]
    pub fn new(class: Class) -> Self {
        Self {
            class,
            entries: Vec::new(),
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<&Value> {
        self.entries
            .iter()
            .find(|entry| entry.key.bytes() == key)
            .map(|entry| &entry.value)
    }

    pub fn get_mut(&mut self, key: &[u8]) -> Option<&mut Value> {
        self.entries
            .iter_mut()
            .find(|entry| entry.key.bytes() == key)
            .map(|entry| &mut entry.value)
    }

    /// Updates the first occurrence, preserving the original key encoding/order.
    pub fn set(&mut self, key: Id, value: Value) {
        if let Some(existing) = self.get_mut(key.bytes()) {
            *existing = value;
        } else {
            self.entries.push(Entry::new(key, value));
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reference {
    Property {
        class: Class,
        key: Id,
    },
    Class(Class),
    Enumerated {
        class: Class,
        type_id: Id,
        value: Id,
    },
    Offset {
        class: Class,
        value: i32,
    },
    Identifier(i32),
    Index(i32),
    Name {
        class: Class,
        value: Vec<u16>,
    },
}

/// Floating-point payloads are stored as bits, avoiding NaN canonicalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Reference(Vec<Reference>),
    Object(Box<Descriptor>),
    GlobalObject(Box<Descriptor>),
    List(Vec<Value>),
    Double(u64),
    UnitFloat {
        unit: [u8; 4],
        bits: u64,
    },
    UnitFloats {
        unit: [u8; 4],
        bits: Vec<u64>,
    },
    Text(Vec<u16>),
    Enumerated {
        type_id: Id,
        value: Id,
    },
    Integer(i32),
    LargeInteger(i64),
    Boolean(u8),
    Class(Class),
    GlobalClass(Class),
    Alias(Vec<u8>),
    Raw(Vec<u8>),
    Path(Vec<u8>),
    ObjectArray {
        count: u32,
        descriptor: Box<Descriptor>,
    },
}

impl Value {
    pub fn double(value: f64) -> Self {
        Self::Double(value.to_bits())
    }
    pub fn text(value: &str) -> Self {
        Self::Text(value.encode_utf16().collect())
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Double(bits) | Self::UnitFloat { bits, .. } => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }
    pub fn type_code(&self) -> [u8; 4] {
        *match self {
            Self::Reference(_) => b"obj ",
            Self::Object(_) => b"Objc",
            Self::GlobalObject(_) => b"GlbO",
            Self::List(_) => b"VlLs",
            Self::Double(_) => b"doub",
            Self::UnitFloat { .. } => b"UntF",
            Self::UnitFloats { .. } => b"UnFl",
            Self::Text(_) => b"TEXT",
            Self::Enumerated { .. } => b"enum",
            Self::Integer(_) => b"long",
            Self::LargeInteger(_) => b"comp",
            Self::Boolean(_) => b"bool",
            Self::Class(_) => b"type",
            Self::GlobalClass(_) => b"GlbC",
            Self::Alias(_) => b"alis",
            Self::Raw(_) => b"tdta",
            Self::Path(_) => b"Pth ",
            Self::ObjectArray { .. } => b"ObAr",
        }
    }
}

/// A bounded descriptor can carry an uninterpreted suffix or an unknown type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Block {
    Parsed {
        descriptor: Descriptor,
        trailing: Vec<u8>,
    },
    Opaque {
        bytes: Vec<u8>,
        unknown_type: [u8; 4],
    },
}

#[derive(Debug)]
struct UnknownType([u8; 4]);
impl fmt::Display for UnknownType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown Photoshop descriptor type {:?}; preserve enclosing block",
            self.0
        )
    }
}
impl std::error::Error for UnknownType {}

pub fn decode(bytes: &[u8], limits: Limits) -> Result<Descriptor> {
    let (descriptor, used) = decode_prefix(bytes, limits)?;
    ensure!(
        used == bytes.len(),
        "{} trailing descriptor bytes",
        bytes.len() - used
    );
    Ok(descriptor)
}

/// Reads one descriptor from a larger typed block, reporting consumed bytes.
pub fn decode_prefix(bytes: &[u8], limits: Limits) -> Result<(Descriptor, usize)> {
    ensure!(
        bytes.len() <= limits.max_bytes,
        "descriptor exceeds byte limit"
    );
    let mut reader = Reader {
        bytes,
        position: 0,
        limits,
        allocated: 0,
        items: 0,
    };
    let descriptor = reader.descriptor(0)?;
    Ok((descriptor, reader.position))
}

/// Only unknown types trigger preservation; corruption and limit failures fail.
pub fn decode_preserving(bytes: &[u8], limits: Limits) -> Result<Block> {
    ensure!(
        bytes.len() <= limits.max_bytes,
        "descriptor exceeds byte limit"
    );
    let mut reader = Reader {
        bytes,
        position: 0,
        limits,
        allocated: 0,
        items: 0,
    };
    match reader.descriptor(0) {
        Ok(descriptor) => {
            reader.charge(bytes.len() - reader.position)?;
            let trailing = copy_bytes(&bytes[reader.position..])?;
            Ok(Block::Parsed {
                descriptor,
                trailing,
            })
        }
        Err(error) => {
            if let Some(unknown) = error.downcast_ref::<UnknownType>() {
                ensure!(
                    bytes.len() <= limits.max_allocated_bytes,
                    "opaque descriptor exceeds allocation limit"
                );
                Ok(Block::Opaque {
                    bytes: copy_bytes(bytes)?,
                    unknown_type: unknown.0,
                })
            } else {
                Err(error)
            }
        }
    }
}

pub fn encode(descriptor: &Descriptor, limits: Limits) -> Result<Vec<u8>> {
    let mut writer = Writer {
        bytes: Vec::new(),
        limits,
        items: 0,
    };
    writer.descriptor(descriptor, 0)?;
    Ok(writer.bytes)
}

pub fn encode_block(block: &Block, limits: Limits) -> Result<Vec<u8>> {
    match block {
        Block::Parsed {
            descriptor,
            trailing,
        } => {
            let mut writer = Writer {
                bytes: encode(descriptor, limits)?,
                limits,
                items: 0,
            };
            writer.write(trailing)?;
            Ok(writer.bytes)
        }
        Block::Opaque { bytes, .. } => {
            ensure!(
                bytes.len() <= limits.max_bytes && bytes.len() <= limits.max_allocated_bytes,
                "opaque descriptor exceeds byte limit"
            );
            copy_bytes(bytes)
        }
    }
}

fn copy_bytes(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(bytes.len())
        .context("allocating descriptor bytes")?;
    result.extend_from_slice(bytes);
    Ok(result)
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
    limits: Limits,
    allocated: usize,
    items: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .context("descriptor length overflow")?;
        let result = self.bytes.get(self.position..end).with_context(|| {
            format!(
                "truncated descriptor at byte {}: need {len} bytes",
                self.position
            )
        })?;
        self.position = end;
        Ok(result)
    }
    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut result = [0; N];
        result.copy_from_slice(self.take(N)?);
        Ok(result)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.fixed()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.fixed()?))
    }
    fn allocate<T>(&mut self, count: usize) -> Result<Vec<T>> {
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .context("descriptor allocation overflow")?;
        self.charge(bytes)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .context("allocating descriptor values")?;
        Ok(result)
    }
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.allocated = self
            .allocated
            .checked_add(bytes)
            .context("descriptor allocation overflow")?;
        ensure!(
            self.allocated <= self.limits.max_allocated_bytes,
            "descriptor exceeds allocation limit"
        );
        Ok(())
    }
    fn count(&mut self, minimum_size: usize) -> Result<usize> {
        let count = self.u32()? as usize;
        self.items = self
            .items
            .checked_add(count)
            .context("descriptor item overflow")?;
        ensure!(
            self.items <= self.limits.max_items,
            "descriptor exceeds item limit"
        );
        ensure!(
            count <= (self.bytes.len() - self.position) / minimum_size,
            "truncated descriptor item array"
        );
        Ok(count)
    }
    fn blob(&mut self) -> Result<Vec<u8>> {
        let count = self.u32()? as usize;
        let bytes = self.take(count)?;
        let mut result = self.allocate(count)?;
        result.extend_from_slice(bytes);
        Ok(result)
    }
    fn id(&mut self) -> Result<Id> {
        let count = self.u32()? as usize;
        if count == 0 {
            return Ok(Id::Four(self.fixed()?));
        }
        let bytes = self.take(count)?;
        let mut result = self.allocate(count)?;
        result.extend_from_slice(bytes);
        Ok(Id::String(result))
    }
    fn unicode(&mut self) -> Result<Vec<u16>> {
        let count = self.u32()? as usize;
        let bytes = self.take(count.checked_mul(2).context("UTF-16 length overflow")?)?;
        let mut result = self.allocate(count)?;
        result.extend(
            bytes
                .chunks_exact(2)
                .map(|unit| u16::from_be_bytes([unit[0], unit[1]])),
        );
        Ok(result)
    }
    fn class(&mut self) -> Result<Class> {
        Ok(Class {
            name: self.unicode()?,
            id: self.id()?,
        })
    }
    fn descriptor(&mut self, depth: usize) -> Result<Descriptor> {
        ensure!(
            depth <= self.limits.max_depth,
            "descriptor exceeds nesting limit"
        );
        let class = self.class()?;
        let count = self.count(9)?;
        let mut entries = self.allocate(count)?;
        for _ in 0..count {
            let key = self.id()?;
            let type_code = self.fixed()?;
            entries.push(Entry {
                key,
                value: self.value(type_code, depth + 1)?,
            });
        }
        Ok(Descriptor { class, entries })
    }
    fn boxed_descriptor(&mut self, depth: usize) -> Result<Box<Descriptor>> {
        self.charge(std::mem::size_of::<Descriptor>())?;
        Ok(Box::new(self.descriptor(depth)?))
    }
    fn value(&mut self, type_code: [u8; 4], depth: usize) -> Result<Value> {
        ensure!(
            depth <= self.limits.max_depth,
            "descriptor exceeds nesting limit"
        );
        Ok(match &type_code {
            b"obj " => {
                let count = self.count(4)?;
                let mut result = self.allocate(count)?;
                for _ in 0..count {
                    result.push(self.reference()?);
                }
                Value::Reference(result)
            }
            b"Objc" => Value::Object(self.boxed_descriptor(depth)?),
            b"GlbO" => Value::GlobalObject(self.boxed_descriptor(depth)?),
            b"ObAr" => {
                let count = self.u32()?;
                Value::ObjectArray {
                    count,
                    descriptor: self.boxed_descriptor(depth)?,
                }
            }
            b"VlLs" => {
                let count = self.count(4)?;
                let mut result = self.allocate(count)?;
                for _ in 0..count {
                    let code = self.fixed()?;
                    result.push(self.value(code, depth + 1)?);
                }
                Value::List(result)
            }
            b"doub" => Value::Double(self.u64()?),
            b"UntF" => Value::UnitFloat {
                unit: self.fixed()?,
                bits: self.u64()?,
            },
            b"UnFl" => {
                let unit = self.fixed()?;
                let count = self.count(8)?;
                let mut bits = self.allocate(count)?;
                for _ in 0..count {
                    bits.push(self.u64()?);
                }
                Value::UnitFloats { unit, bits }
            }
            b"TEXT" => Value::Text(self.unicode()?),
            b"enum" => Value::Enumerated {
                type_id: self.id()?,
                value: self.id()?,
            },
            b"long" => Value::Integer(self.u32()? as i32),
            b"comp" => Value::LargeInteger(self.u64()? as i64),
            b"bool" => Value::Boolean(self.take(1)?[0]),
            b"type" => Value::Class(self.class()?),
            b"GlbC" => Value::GlobalClass(self.class()?),
            b"alis" => Value::Alias(self.blob()?),
            b"tdta" => Value::Raw(self.blob()?),
            b"Pth " => Value::Path(self.blob()?),
            _ => return Err(UnknownType(type_code).into()),
        })
    }
    fn reference(&mut self) -> Result<Reference> {
        let code = self.fixed()?;
        Ok(match &code {
            b"prop" => Reference::Property {
                class: self.class()?,
                key: self.id()?,
            },
            b"Clss" => Reference::Class(self.class()?),
            b"Enmr" => Reference::Enumerated {
                class: self.class()?,
                type_id: self.id()?,
                value: self.id()?,
            },
            b"rele" => Reference::Offset {
                class: self.class()?,
                value: self.u32()? as i32,
            },
            b"Idnt" => Reference::Identifier(self.u32()? as i32),
            b"indx" => Reference::Index(self.u32()? as i32),
            b"name" => Reference::Name {
                class: self.class()?,
                value: self.unicode()?,
            },
            _ => return Err(UnknownType(code).into()),
        })
    }
}

struct Writer {
    bytes: Vec<u8>,
    limits: Limits,
    items: usize,
}
impl Writer {
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        let end = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .context("descriptor output length overflow")?;
        ensure!(
            end <= self.limits.max_bytes && end <= self.limits.max_allocated_bytes,
            "descriptor output exceeds byte limit"
        );
        if end > self.bytes.capacity() {
            let limit = self.limits.max_bytes.min(self.limits.max_allocated_bytes);
            let capacity = end.checked_next_power_of_two().unwrap_or(end).min(limit);
            self.bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .context("allocating descriptor output")?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn u32(&mut self, value: u32) -> Result<()> {
        self.write(&value.to_be_bytes())
    }
    fn len(&mut self, value: usize) -> Result<()> {
        self.u32(u32::try_from(value).context("descriptor length exceeds u32")?)
    }
    fn count(&mut self, count: usize) -> Result<()> {
        self.items = self
            .items
            .checked_add(count)
            .context("descriptor item overflow")?;
        ensure!(
            self.items <= self.limits.max_items,
            "descriptor exceeds item limit"
        );
        self.len(count)
    }
    fn blob(&mut self, value: &[u8]) -> Result<()> {
        self.len(value.len())?;
        self.write(value)
    }
    fn id(&mut self, value: &Id) -> Result<()> {
        match value {
            Id::Four(bytes) => {
                self.u32(0)?;
                self.write(bytes)
            }
            Id::String(bytes) => {
                ensure!(!bytes.is_empty(), "empty descriptor string ID is ambiguous");
                self.blob(bytes)
            }
        }
    }
    fn unicode(&mut self, value: &[u16]) -> Result<()> {
        self.len(value.len())?;
        for unit in value {
            self.write(&unit.to_be_bytes())?;
        }
        Ok(())
    }
    fn class(&mut self, value: &Class) -> Result<()> {
        self.unicode(&value.name)?;
        self.id(&value.id)
    }
    fn descriptor(&mut self, value: &Descriptor, depth: usize) -> Result<()> {
        ensure!(
            depth <= self.limits.max_depth,
            "descriptor exceeds nesting limit"
        );
        self.class(&value.class)?;
        self.count(value.entries.len())?;
        for entry in &value.entries {
            self.id(&entry.key)?;
            self.value(&entry.value, depth + 1)?;
        }
        Ok(())
    }
    fn value(&mut self, value: &Value, depth: usize) -> Result<()> {
        ensure!(
            depth <= self.limits.max_depth,
            "descriptor exceeds nesting limit"
        );
        self.write(&value.type_code())?;
        match value {
            Value::Reference(values) => {
                self.count(values.len())?;
                for value in values {
                    self.reference(value)?;
                }
            }
            Value::Object(value) | Value::GlobalObject(value) => self.descriptor(value, depth)?,
            Value::ObjectArray { count, descriptor } => {
                self.u32(*count)?;
                self.descriptor(descriptor, depth)?;
            }
            Value::List(values) => {
                self.count(values.len())?;
                for value in values {
                    self.value(value, depth + 1)?;
                }
            }
            Value::Double(bits) => self.write(&bits.to_be_bytes())?,
            Value::UnitFloat { unit, bits } => {
                self.write(unit)?;
                self.write(&bits.to_be_bytes())?;
            }
            Value::UnitFloats { unit, bits } => {
                self.write(unit)?;
                self.count(bits.len())?;
                for value in bits {
                    self.write(&value.to_be_bytes())?;
                }
            }
            Value::Text(value) => self.unicode(value)?,
            Value::Enumerated { type_id, value } => {
                self.id(type_id)?;
                self.id(value)?;
            }
            Value::Integer(value) => self.write(&value.to_be_bytes())?,
            Value::LargeInteger(value) => self.write(&value.to_be_bytes())?,
            Value::Boolean(value) => self.write(&[*value])?,
            Value::Class(value) | Value::GlobalClass(value) => self.class(value)?,
            Value::Alias(value) | Value::Raw(value) | Value::Path(value) => self.blob(value)?,
        }
        Ok(())
    }
    fn reference(&mut self, value: &Reference) -> Result<()> {
        match value {
            Reference::Property { class, key } => {
                self.write(b"prop")?;
                self.class(class)?;
                self.id(key)?;
            }
            Reference::Class(class) => {
                self.write(b"Clss")?;
                self.class(class)?;
            }
            Reference::Enumerated {
                class,
                type_id,
                value,
            } => {
                self.write(b"Enmr")?;
                self.class(class)?;
                self.id(type_id)?;
                self.id(value)?;
            }
            Reference::Offset { class, value } => {
                self.write(b"rele")?;
                self.class(class)?;
                self.write(&value.to_be_bytes())?;
            }
            Reference::Identifier(value) => {
                self.write(b"Idnt")?;
                self.write(&value.to_be_bytes())?;
            }
            Reference::Index(value) => {
                self.write(b"indx")?;
                self.write(&value.to_be_bytes())?;
            }
            Reference::Name { class, value } => {
                self.write(b"name")?;
                self.class(class)?;
                self.unicode(value)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Descriptor {
        let class = Class {
            name: vec![0x61, 0xd800, 0],
            id: Id::String(b"test".to_vec()),
        };
        let nested = Descriptor {
            class: class.clone(),
            entries: vec![Entry::new(Id::Four(*b"????"), Value::Boolean(0x7f))],
        };
        let references = vec![
            Reference::Property {
                class: class.clone(),
                key: Id::Four(*b"Prpr"),
            },
            Reference::Class(class.clone()),
            Reference::Enumerated {
                class: class.clone(),
                type_id: Id::Four(*b"Ordn"),
                value: Id::Four(*b"Trgt"),
            },
            Reference::Offset {
                class: class.clone(),
                value: -4,
            },
            Reference::Identifier(i32::MIN),
            Reference::Index(100),
            Reference::Name {
                class: class.clone(),
                value: vec![0x61, 0, 0xdfff],
            },
        ];
        let values = vec![
            Value::Reference(references),
            Value::Object(Box::new(nested.clone())),
            Value::GlobalObject(Box::new(nested.clone())),
            Value::List(vec![
                Value::Boolean(255),
                Value::List(vec![Value::Integer(4)]),
            ]),
            Value::Double(0xfff8_0123_4567_89ab),
            Value::UnitFloat {
                unit: *b"????",
                bits: (-0.0_f64).to_bits(),
            },
            Value::UnitFloats {
                unit: *b"#Pxl",
                bits: vec![0, f64::INFINITY.to_bits(), 1],
            },
            Value::Text(vec![0xd800, 0, 0xdc00]),
            Value::Enumerated {
                type_id: Id::String(b"test".to_vec()),
                value: Id::Four(*b"test"),
            },
            Value::Integer(i32::MIN),
            Value::LargeInteger(i64::MIN),
            Value::Boolean(2),
            Value::Class(class.clone()),
            Value::GlobalClass(class.clone()),
            Value::Alias(vec![0, 255, 128]),
            Value::Raw(vec![255, 0, 1]),
            Value::Path(vec![12, 0, 0, 0]),
            Value::ObjectArray {
                count: 100,
                descriptor: Box::new(nested),
            },
        ];
        Descriptor {
            class,
            entries: values
                .into_iter()
                .map(|value| Entry::new(Id::Four(*b"same"), value))
                .collect(),
        }
    }

    #[test]
    fn all_types_and_wire_distinctions_round_trip() {
        let descriptor = fixture();
        let bytes = encode(&descriptor, Limits::default()).unwrap();
        let parsed = decode(&bytes, Limits::default()).unwrap();
        assert_eq!(parsed, descriptor);
        assert_eq!(encode(&parsed, Limits::default()).unwrap(), bytes);
    }

    #[test]
    fn literal_integer_descriptor_matches_adobe_layout() {
        let bytes = b"\0\0\0\0\0\0\0\0null\0\0\0\x01\0\0\0\0testlong\xff\xff\xff\xfe";
        let descriptor = decode(bytes, Limits::default()).unwrap();
        assert_eq!(descriptor.get(b"test"), Some(&Value::Integer(-2)));
        assert_eq!(encode(&descriptor, Limits::default()).unwrap(), bytes);
    }

    #[test]
    fn editing_known_key_keeps_unknown_keys_payloads_and_order_exact() {
        let mut descriptor = fixture();
        descriptor.set(Id::Four(*b"edit"), Value::Integer(10));
        let before = encode(&descriptor, Limits::default()).unwrap();
        let mut parsed = decode(&before, Limits::default()).unwrap();
        parsed.set(Id::String(b"edit".to_vec()), Value::Integer(20));
        let after = encode(&parsed, Limits::default()).unwrap();
        assert_eq!(&before[..before.len() - 4], &after[..after.len() - 4]);
        assert_eq!(parsed.entries.last().unwrap().key, Id::Four(*b"edit"));
    }

    #[test]
    fn unknown_type_preserves_entire_bounded_parent() {
        let bytes = b"\0\0\0\0\0\0\0\0null\0\0\0\x01\0\0\0\0testZZZZ\0\xff\xaa";
        let block = decode_preserving(bytes, Limits::default()).unwrap();
        assert!(matches!(block, Block::Opaque { unknown_type: code, .. } if code == *b"ZZZZ"));
        assert_eq!(encode_block(&block, Limits::default()).unwrap(), bytes);
    }

    #[test]
    fn nested_unknown_type_preserves_parent_without_guessing_its_length() {
        let mut bytes = encode(&fixture(), Limits::default()).unwrap();
        let position = bytes.windows(4).position(|bytes| bytes == b"bool").unwrap();
        bytes[position..position + 4].copy_from_slice(b"futr");
        let block = decode_preserving(&bytes, Limits::default()).unwrap();
        assert!(matches!(block, Block::Opaque { .. }));
        assert_eq!(encode_block(&block, Limits::default()).unwrap(), bytes);
    }

    #[test]
    fn unknown_list_type_does_not_assume_a_payload_minimum() {
        let bytes = b"\0\0\0\0\0\0\0\0null\0\0\0\x01\0\0\0\0testVlLs\0\0\0\x01ZZZZ";
        assert!(matches!(
            decode_preserving(bytes, Limits::default()).unwrap(),
            Block::Opaque { .. }
        ));
    }

    #[test]
    fn truncated_input_never_becomes_opaque_success() {
        let bytes = encode(&fixture(), Limits::default()).unwrap();
        for len in 0..bytes.len() {
            assert!(
                decode_preserving(&bytes[..len], Limits::default()).is_err(),
                "accepted prefix {len}"
            );
        }
    }

    #[test]
    fn malicious_lengths_fail_before_allocation() {
        for bytes in [
            b"\xff\xff\xff\xff".as_slice(),
            b"\0\0\0\0\0\0\0\0null\xff\xff\xff\xff".as_slice(),
        ] {
            assert!(decode(bytes, Limits::default()).is_err());
        }
    }

    #[test]
    fn input_output_allocation_item_and_depth_limits_are_enforced() {
        let descriptor = fixture();
        let bytes = encode(&descriptor, Limits::default()).unwrap();
        for limits in [
            Limits {
                max_bytes: 10,
                ..Limits::default()
            },
            Limits {
                max_allocated_bytes: 8,
                ..Limits::default()
            },
            Limits {
                max_items: 1,
                ..Limits::default()
            },
            Limits {
                max_depth: 1,
                ..Limits::default()
            },
        ] {
            assert!(decode(&bytes, limits).is_err());
            assert!(encode(&descriptor, limits).is_err());
        }
    }

    #[test]
    fn bounded_suffix_and_prefix_consumption_are_preserved() {
        let mut bytes = encode(&fixture(), Limits::default()).unwrap();
        let used = bytes.len();
        bytes.extend_from_slice(&[0xab, 0xcd]);
        assert_eq!(decode_prefix(&bytes, Limits::default()).unwrap().1, used);
        assert!(decode(&bytes, Limits::default()).is_err());
        assert_eq!(
            encode_block(
                &decode_preserving(&bytes, Limits::default()).unwrap(),
                Limits::default()
            )
            .unwrap(),
            bytes
        );
    }

    #[test]
    fn empty_string_identifier_cannot_emit_an_ambiguous_length() {
        let descriptor = Descriptor::new(Class::new("", Id::String(Vec::new())));
        assert!(encode(&descriptor, Limits::default()).is_err());
    }
}
