//! Photoshop text EngineData with a lossless lexical tree.
//!
//! EngineData is an undocumented PostScript-like grammar, not executable
//! PostScript. Parsing never evaluates names/operators. Original whitespace,
//! number spelling, escaped strings, unknown atoms and key order survive; an
//! edit replaces only the selected value's span. UTF-16 text remains separate
//! from the byte grammar, which can also contain binary strings.

use std::ops::Range;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use super::descriptor::Limits;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Dictionary(Vec<Entry>),
    Array(Vec<Node>),
    String(Vec<u8>),
    Name(Vec<u8>),
    Integer(i64),
    Real(f64),
    Boolean(bool),
    Null,
    /// An unknown scalar token. It is retained, never evaluated.
    Atom(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    key: Vec<u8>,
    value: Node,
}

impl Entry {
    pub fn new(key: Vec<u8>, value: Node) -> Self {
        Self { key, value }
    }
    pub fn key(&self) -> &[u8] {
        &self.key
    }
    pub fn value(&self) -> &Node {
        &self.value
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    value: Value,
    span: Option<Range<usize>>,
    source_identity: Option<Arc<()>>,
    replaced: bool,
    implicit_dictionary: bool,
}

impl Node {
    pub fn new(value: Value) -> Self {
        Self {
            value,
            span: None,
            source_identity: None,
            replaced: true,
            implicit_dictionary: false,
        }
    }
    pub fn value(&self) -> &Value {
        &self.value
    }
    pub fn replace(&mut self, value: Value) {
        self.value = value;
        self.replaced = true;
    }
    pub fn get(&self, key: &[u8]) -> Option<&Node> {
        let Value::Dictionary(entries) = &self.value else {
            return None;
        };
        entries
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.value)
    }
    pub fn get_mut(&mut self, key: &[u8]) -> Option<&mut Node> {
        let Value::Dictionary(entries) = &mut self.value else {
            return None;
        };
        entries
            .iter_mut()
            .find(|entry| entry.key == key)
            .map(|entry| &mut entry.value)
    }
    pub fn get_index(&self, index: usize) -> Option<&Node> {
        let Value::Array(values) = &self.value else {
            return None;
        };
        values.get(index)
    }
    pub fn get_index_mut(&mut self, index: usize) -> Option<&mut Node> {
        let Value::Array(values) = &mut self.value else {
            return None;
        };
        values.get_mut(index)
    }
    /// Existing entries retain their lexical key; appending retains all old values.
    pub fn set(&mut self, key: &[u8], value: Value) -> Result<()> {
        let Value::Dictionary(entries) = &mut self.value else {
            bail!("EngineData value is not a dictionary");
        };
        if let Some(entry) = entries.iter_mut().find(|entry| entry.key == key) {
            entry.value.replace(value);
        } else {
            entries
                .try_reserve(1)
                .context("allocating EngineData entry")?;
            entries.push(Entry::new(key.to_vec(), Node::new(value)));
            self.replaced = true;
        }
        Ok(())
    }
    pub fn text(value: &str) -> Self {
        let mut bytes = vec![0xfe, 0xff];
        for unit in value.encode_utf16() {
            bytes.extend_from_slice(&unit.to_be_bytes());
        }
        Self::new(Value::String(bytes))
    }
    /// Decodes BOM-marked UTF-16, rejecting malformed units instead of replacing.
    pub fn as_text(&self) -> Result<String> {
        let Value::String(bytes) = &self.value else {
            bail!("EngineData value is not a string");
        };
        let (little_endian, bytes) = if bytes.starts_with(&[0xfe, 0xff]) {
            (false, &bytes[2..])
        } else if bytes.starts_with(&[0xff, 0xfe]) {
            (true, &bytes[2..])
        } else {
            bail!("EngineData string has no UTF-16 byte order mark");
        };
        ensure!(
            bytes.len() % 2 == 0,
            "EngineData UTF-16 string has odd byte length"
        );
        let units = bytes.chunks_exact(2).map(|value| {
            if little_endian {
                u16::from_le_bytes([value[0], value[1]])
            } else {
                u16::from_be_bytes([value[0], value[1]])
            }
        });
        char::decode_utf16(units)
            .collect::<std::result::Result<String, _>>()
            .context("invalid EngineData UTF-16")
    }
}

#[derive(Clone, Debug)]
pub struct EngineData {
    source: Arc<[u8]>,
    source_identity: Arc<()>,
    root: Node,
}

impl EngineData {
    #[cfg(test)]
    pub fn new(root: Node) -> Self {
        Self {
            source: Arc::from([]),
            source_identity: Arc::new(()),
            root,
        }
    }
    pub fn root(&self) -> &Node {
        &self.root
    }
    pub fn root_mut(&mut self) -> &mut Node {
        &mut self.root
    }

    pub fn parse(bytes: &[u8], limits: Limits) -> Result<Self> {
        ensure!(
            bytes.len() <= limits.max_bytes,
            "EngineData exceeds byte limit"
        );
        let mut parser = Parser {
            bytes,
            source_identity: Arc::new(()),
            position: 0,
            allocated: bytes.len(),
            items: 0,
            limits,
        };
        ensure!(
            parser.allocated <= limits.max_allocated_bytes,
            "EngineData exceeds allocation limit"
        );
        parser.trivia();
        let root = if parser.peek() == Some(b'/') {
            parser.count_node()?;
            parser.dictionary(0, true)?
        } else {
            parser.node(0)?
        };
        parser.trivia();
        ensure!(
            parser.position == bytes.len(),
            "trailing EngineData token at byte {}",
            parser.position
        );
        Ok(Self {
            source: Arc::from(bytes),
            source_identity: parser.source_identity,
            root,
        })
    }

    pub fn to_bytes(&self, limits: Limits) -> Result<Vec<u8>> {
        let mut writer = Writer {
            source: &self.source,
            source_identity: &self.source_identity,
            bytes: Vec::new(),
            limits,
            items: 0,
        };
        if let Some(span) = &self.root.span {
            writer.write(
                self.source
                    .get(..span.start)
                    .context("EngineData root span outside source")?,
            )?;
        }
        writer.node(&self.root, 0)?;
        if let Some(span) = &self.root.span {
            writer.write(
                self.source
                    .get(span.end..)
                    .context("EngineData root span outside source")?,
            )?;
        }
        Ok(writer.bytes)
    }
}

fn whitespace(byte: u8) -> bool {
    matches!(byte, 0 | b'\t' | b'\n' | 12 | b'\r' | b' ')
}
fn delimiter(byte: u8) -> bool {
    whitespace(byte)
        || matches!(
            byte,
            b'/' | b'[' | b']' | b'<' | b'>' | b'(' | b')' | b'%' | b'{' | b'}'
        )
}
fn hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    source_identity: Arc<()>,
    position: usize,
    allocated: usize,
    items: usize,
    limits: Limits,
}
impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }
    fn trivia(&mut self) {
        loop {
            while self.peek().is_some_and(whitespace) {
                self.position += 1;
            }
            if self.peek() != Some(b'%') {
                break;
            }
            while self
                .peek()
                .is_some_and(|byte| !matches!(byte, b'\r' | b'\n'))
            {
                self.position += 1;
            }
        }
    }
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.allocated = self
            .allocated
            .checked_add(bytes)
            .context("EngineData allocation overflow")?;
        ensure!(
            self.allocated <= self.limits.max_allocated_bytes,
            "EngineData exceeds allocation limit"
        );
        Ok(())
    }
    fn copy(&mut self, span: Range<usize>) -> Result<Vec<u8>> {
        self.charge(span.len())?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(span.len())
            .context("allocating EngineData token")?;
        bytes.extend_from_slice(&self.bytes[span]);
        Ok(bytes)
    }
    fn push_byte(&mut self, output: &mut Vec<u8>, byte: u8) -> Result<()> {
        self.reserve(output)?;
        output.push(byte);
        Ok(())
    }
    fn reserve<T>(&mut self, values: &mut Vec<T>) -> Result<()> {
        if values.len() < values.capacity() {
            return Ok(());
        }
        let size = std::mem::size_of::<T>();
        let available = self
            .limits
            .max_allocated_bytes
            .saturating_sub(self.allocated)
            / size;
        ensure!(available > 0, "EngineData exceeds allocation limit");
        let extra = values.capacity().max(4).min(available);
        self.charge(
            extra
                .checked_mul(size)
                .context("EngineData allocation overflow")?,
        )?;
        values
            .try_reserve_exact(extra)
            .context("allocating EngineData values")?;
        Ok(())
    }
    fn node(&mut self, depth: usize) -> Result<Node> {
        ensure!(
            depth <= self.limits.max_depth,
            "EngineData exceeds nesting limit"
        );
        self.count_node()?;
        self.trivia();
        let start = self.position;
        let value = match self.peek().context("truncated EngineData value")? {
            b'<' if self.bytes.get(self.position + 1) == Some(&b'<') => {
                return self.dictionary(depth, false);
            }
            b'[' => {
                self.position += 1;
                let mut values = Vec::new();
                loop {
                    self.trivia();
                    if self.peek() == Some(b']') {
                        self.position += 1;
                        break;
                    }
                    ensure!(self.peek().is_some(), "unterminated EngineData array");
                    self.reserve(&mut values)?;
                    values.push(self.node(depth + 1)?);
                }
                Value::Array(values)
            }
            b'(' => Value::String(self.string()?),
            b'<' => Value::String(self.hex_string()?),
            b'/' => Value::Name(self.name()?),
            b']' | b'>' | b')' | b'{' | b'}' => {
                bail!("unexpected EngineData delimiter at byte {}", self.position)
            }
            _ => self.atom()?,
        };
        Ok(Node {
            value,
            span: Some(start..self.position),
            source_identity: Some(Arc::clone(&self.source_identity)),
            replaced: false,
            implicit_dictionary: false,
        })
    }
    fn count_node(&mut self) -> Result<()> {
        self.items = self
            .items
            .checked_add(1)
            .context("EngineData item overflow")?;
        ensure!(
            self.items <= self.limits.max_items,
            "EngineData exceeds item limit"
        );
        Ok(())
    }
    fn dictionary(&mut self, depth: usize, implicit: bool) -> Result<Node> {
        ensure!(
            depth <= self.limits.max_depth,
            "EngineData exceeds nesting limit"
        );
        let start = self.position;
        if !implicit {
            self.position += 2;
        }
        let mut entries = Vec::new();
        loop {
            self.trivia();
            if implicit && self.peek().is_none() {
                break;
            }
            if !implicit && self.bytes.get(self.position..self.position + 2) == Some(b">>") {
                self.position += 2;
                break;
            }
            ensure!(
                self.peek() == Some(b'/'),
                "expected EngineData dictionary key at byte {}",
                self.position
            );
            let key = self.name()?;
            let value = self.node(depth + 1)?;
            self.reserve(&mut entries)?;
            entries.push(Entry { key, value });
        }
        Ok(Node {
            value: Value::Dictionary(entries),
            span: Some(start..self.position),
            source_identity: Some(Arc::clone(&self.source_identity)),
            replaced: false,
            implicit_dictionary: implicit,
        })
    }
    fn name(&mut self) -> Result<Vec<u8>> {
        self.position += 1;
        let mut value = Vec::new();
        while let Some(byte) = self.peek() {
            if delimiter(byte) {
                break;
            }
            self.position += 1;
            if byte == b'#' {
                let a = self
                    .peek()
                    .and_then(hex)
                    .context("invalid EngineData name escape")?;
                self.position += 1;
                let b = self
                    .peek()
                    .and_then(hex)
                    .context("invalid EngineData name escape")?;
                self.position += 1;
                self.push_byte(&mut value, a * 16 + b)?;
            } else {
                self.push_byte(&mut value, byte)?;
            }
        }
        Ok(value)
    }
    fn atom(&mut self) -> Result<Value> {
        let start = self.position;
        while self.peek().is_some_and(|byte| !delimiter(byte)) {
            self.position += 1;
        }
        ensure!(start < self.position, "empty EngineData token");
        let bytes = &self.bytes[start..self.position];
        Ok(match bytes {
            b"true" => Value::Boolean(true),
            b"false" => Value::Boolean(false),
            b"null" | b"nil" => Value::Null,
            _ => {
                if let Ok(text) = std::str::from_utf8(bytes) {
                    if let Ok(value) = text.parse::<i64>() {
                        return Ok(Value::Integer(value));
                    }
                    if text.bytes().any(|byte| matches!(byte, b'.' | b'e' | b'E')) {
                        if let Ok(value) = text.parse::<f64>() {
                            if value.is_finite() {
                                return Ok(Value::Real(value));
                            }
                        }
                    }
                }
                Value::Atom(self.copy(start..self.position)?)
            }
        })
    }
    fn string(&mut self) -> Result<Vec<u8>> {
        self.position += 1;
        let mut nesting = 1_usize;
        let mut value = Vec::new();
        loop {
            let byte = self.peek().context("unterminated EngineData string")?;
            self.position += 1;
            match byte {
                b'(' => {
                    nesting += 1;
                    ensure!(
                        nesting <= self.limits.max_depth,
                        "EngineData string exceeds nesting limit"
                    );
                    self.push_byte(&mut value, byte)?;
                }
                b')' => {
                    nesting -= 1;
                    if nesting == 0 {
                        break;
                    }
                    self.push_byte(&mut value, byte)?;
                }
                b'\\' => {
                    let escaped = self.peek().context("truncated EngineData escape")?;
                    self.position += 1;
                    match escaped {
                        b'n' => self.push_byte(&mut value, b'\n')?,
                        b'r' => self.push_byte(&mut value, b'\r')?,
                        b't' => self.push_byte(&mut value, b'\t')?,
                        b'b' => self.push_byte(&mut value, 8)?,
                        b'f' => self.push_byte(&mut value, 12)?,
                        b'\r' => {
                            if self.peek() == Some(b'\n') {
                                self.position += 1;
                            }
                        }
                        b'\n' => {}
                        b'0'..=b'7' => {
                            let mut code = u16::from(escaped - b'0');
                            for _ in 0..2 {
                                if let Some(next @ b'0'..=b'7') = self.peek() {
                                    self.position += 1;
                                    code = code * 8 + u16::from(next - b'0');
                                } else {
                                    break;
                                }
                            }
                            self.push_byte(&mut value, code as u8)?;
                        }
                        _ => self.push_byte(&mut value, escaped)?,
                    }
                }
                _ => self.push_byte(&mut value, byte)?,
            }
        }
        Ok(value)
    }
    fn hex_string(&mut self) -> Result<Vec<u8>> {
        self.position += 1;
        let mut value = Vec::new();
        let mut high = None;
        loop {
            let byte = self.peek().context("unterminated EngineData hex string")?;
            self.position += 1;
            if byte == b'>' {
                if let Some(high) = high {
                    self.push_byte(&mut value, high * 16)?;
                }
                break;
            }
            if whitespace(byte) {
                continue;
            }
            let digit = hex(byte).context("invalid EngineData hex string")?;
            if let Some(high) = high.take() {
                self.push_byte(&mut value, high * 16 + digit)?;
            } else {
                high = Some(digit);
            }
        }
        Ok(value)
    }
}

struct Writer<'a> {
    source: &'a [u8],
    source_identity: &'a Arc<()>,
    bytes: Vec<u8>,
    limits: Limits,
    items: usize,
}
impl Writer<'_> {
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        let end = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .context("EngineData output length overflow")?;
        ensure!(
            end <= self.limits.max_bytes && end <= self.limits.max_allocated_bytes,
            "EngineData output exceeds byte limit"
        );
        if end > self.bytes.capacity() {
            let limit = self.limits.max_bytes.min(self.limits.max_allocated_bytes);
            let capacity = end.checked_next_power_of_two().unwrap_or(end).min(limit);
            self.bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .context("allocating EngineData output")?;
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn node(&mut self, node: &Node, depth: usize) -> Result<()> {
        if let Some(identity) = &node.source_identity {
            ensure!(
                Arc::ptr_eq(identity, self.source_identity),
                "EngineData node belongs to a different source; replace its value instead"
            );
        }
        ensure!(
            depth <= self.limits.max_depth,
            "EngineData exceeds nesting limit"
        );
        self.items = self
            .items
            .checked_add(1)
            .context("EngineData item overflow")?;
        ensure!(
            self.items <= self.limits.max_items,
            "EngineData exceeds item limit"
        );
        if !node.replaced {
            if let Some(span) = &node.span {
                let mut position = span.start;
                match &node.value {
                    Value::Dictionary(entries) => {
                        for entry in entries {
                            position = self.child(&entry.value, position, depth + 1)?;
                        }
                    }
                    Value::Array(values) => {
                        for value in values {
                            position = self.child(value, position, depth + 1)?;
                        }
                    }
                    _ => {}
                }
                let bytes = self
                    .source
                    .get(position..span.end)
                    .context("EngineData span outside source")?;
                return self.write(bytes);
            }
        }
        self.canonical(&node.value, depth, node.implicit_dictionary)
    }
    fn child(&mut self, node: &Node, position: usize, depth: usize) -> Result<usize> {
        let span = node
            .span
            .as_ref()
            .context("new EngineData child needs a replaced parent")?;
        let bytes = self
            .source
            .get(position..span.start)
            .context("overlapping EngineData spans")?;
        self.write(bytes)?;
        self.node(node, depth)?;
        Ok(span.end)
    }
    fn name(&mut self, name: &[u8]) -> Result<()> {
        self.write(b"/")?;
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        for &byte in name {
            if delimiter(byte) || byte == b'#' || !(33..=126).contains(&byte) {
                self.write(&[b'#', HEX[(byte >> 4) as usize], HEX[(byte & 15) as usize]])?;
            } else {
                self.write(&[byte])?;
            }
        }
        Ok(())
    }
    fn canonical(&mut self, value: &Value, depth: usize, implicit: bool) -> Result<()> {
        match value {
            Value::Dictionary(entries) => {
                if !implicit {
                    self.write(b"<<")?;
                }
                for entry in entries {
                    self.write(b"\n")?;
                    self.name(&entry.key)?;
                    self.write(b" ")?;
                    self.node(&entry.value, depth + 1)?;
                }
                self.write(if implicit { b"\n" } else { b"\n>>" })?;
            }
            Value::Array(values) => {
                self.write(b"[")?;
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        self.write(b" ")?;
                    }
                    self.node(value, depth + 1)?;
                }
                self.write(b"]")?;
            }
            Value::String(bytes) => {
                self.write(b"(")?;
                for &byte in bytes {
                    if matches!(byte, b'(' | b')' | b'\\') {
                        self.write(b"\\")?;
                    }
                    self.write(&[byte])?;
                }
                self.write(b")")?;
            }
            Value::Name(name) => self.name(name)?,
            Value::Integer(value) => self.write(value.to_string().as_bytes())?,
            Value::Real(value) => {
                ensure!(value.is_finite(), "cannot write nonfinite EngineData real");
                let mut spelling = value.to_string();
                if !spelling.contains(['.', 'e', 'E']) {
                    spelling.push_str(".0");
                }
                self.write(spelling.as_bytes())?;
            }
            Value::Boolean(value) => self.write(if *value { b"true" } else { b"false" })?,
            Value::Null => self.write(b"null")?,
            Value::Atom(bytes) => {
                ensure!(
                    !bytes.is_empty() && !bytes.iter().copied().any(delimiter),
                    "invalid EngineData atom"
                );
                self.write(bytes)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untouched_tree_retains_every_lexical_byte() {
        let bytes = b"\n% preserve me\r\n<< /EngineDict << /Editor << /Text (\xfe\xff\0H\0i\0\r) >> >> /Odd [+01 -.500e+2 true nil /N#61me <aBC> (a\\(b\\)\\\\c\\101)] /Mystery @@future >> \0";
        let data = EngineData::parse(bytes, Limits::default()).unwrap();
        assert_eq!(data.to_bytes(Limits::default()).unwrap(), bytes);
        assert_eq!(
            data.root()
                .get(b"EngineDict")
                .unwrap()
                .get(b"Editor")
                .unwrap()
                .get(b"Text")
                .unwrap()
                .as_text()
                .unwrap(),
            "Hi\r"
        );
    }

    #[test]
    fn leaf_edit_preserves_unknown_sibling_bytes_and_whitespace() {
        let bytes = b"\n<< /Known  +0007 % odd spelling\r\n /Unknown [ /who#3F <FEff> -0.00e+5 (a\\051) opaque ] >>\n";
        let mut data = EngineData::parse(bytes, Limits::default()).unwrap();
        data.root_mut()
            .get_mut(b"Known")
            .unwrap()
            .replace(Value::Integer(9));
        let expected = b"\n<< /Known  9 % odd spelling\r\n /Unknown [ /who#3F <FEff> -0.00e+5 (a\\051) opaque ] >>\n";
        assert_eq!(data.to_bytes(Limits::default()).unwrap(), expected);
    }

    #[test]
    fn implicit_root_and_duplicate_keys_are_lossless() {
        let bytes = b"/EngineDict << /Text (one) /Text (two) >>\n/ResourceDict <<>>\n";
        let data = EngineData::parse(bytes, Limits::default()).unwrap();
        assert_eq!(data.to_bytes(Limits::default()).unwrap(), bytes);
    }

    #[test]
    fn binary_strings_escape_parentheses_backslashes_and_all_byte_values() {
        let expected: Vec<u8> = (0..=255).collect();
        let data = EngineData::new(Node::new(Value::String(expected.clone())));
        let bytes = data.to_bytes(Limits::default()).unwrap();
        let parsed = EngineData::parse(&bytes, Limits::default()).unwrap();
        assert_eq!(parsed.root().value(), &Value::String(expected));
    }

    #[test]
    fn utf16_supplementary_characters_round_trip() {
        let data = EngineData::new(Node::text("A🦊𝄞\rB()\\"));
        let encoded = data.to_bytes(Limits::default()).unwrap();
        let parsed = EngineData::parse(&encoded, Limits::default()).unwrap();
        assert_eq!(parsed.root().as_text().unwrap(), "A🦊𝄞\rB()\\");
    }

    #[test]
    fn invalid_utf16_is_preserved_but_not_silently_decoded() {
        let data = EngineData::parse(b"(\xfe\xff\xd8\0)", Limits::default()).unwrap();
        assert!(data.root().as_text().is_err());
        assert_eq!(
            data.to_bytes(Limits::default()).unwrap(),
            b"(\xfe\xff\xd8\0)"
        );
    }

    #[test]
    fn adding_key_keeps_unknown_value_lexemes() {
        let mut data =
            EngineData::parse(b"<< /Unknown [ +0001 (a\\051) ] >>", Limits::default()).unwrap();
        data.root_mut().set(b"New", Value::Boolean(true)).unwrap();
        assert_eq!(
            data.to_bytes(Limits::default()).unwrap(),
            b"<<\n/Unknown [ +0001 (a\\051) ]\n/New true\n>>"
        );
    }

    #[test]
    fn foreign_source_nodes_fail_instead_of_using_wrong_raw_bytes() {
        let mut first = EngineData::parse(b"<< /A [1] >>", Limits::default()).unwrap();
        let second = EngineData::parse(b"<< /B [2] >>", Limits::default()).unwrap();
        *first.root_mut().get_mut(b"A").unwrap() = second.root().get(b"B").unwrap().clone();
        assert!(
            first
                .to_bytes(Limits::default())
                .unwrap_err()
                .to_string()
                .contains("different source")
        );
    }

    #[test]
    fn copied_document_remains_independently_editable() {
        let first = EngineData::parse(b"<< /A +0001 >>", Limits::default()).unwrap();
        let mut second = first.clone();
        second
            .root_mut()
            .get_mut(b"A")
            .unwrap()
            .replace(Value::Integer(2));
        assert_eq!(
            first.to_bytes(Limits::default()).unwrap(),
            b"<< /A +0001 >>"
        );
        assert_eq!(second.to_bytes(Limits::default()).unwrap(), b"<< /A 2 >>");
    }

    #[test]
    fn malformed_delimiters_escapes_and_incomplete_values_fail() {
        for bytes in [
            b"".as_slice(),
            b"<< /A >>",
            b"<< /A 1",
            b"[1",
            b"(abc",
            b"(abc\\",
            b"<xz>",
            b"/X#q1 2",
            b"1 2",
            b"]",
            b"{ foo }",
        ] {
            assert!(
                EngineData::parse(bytes, Limits::default()).is_err(),
                "accepted {bytes:?}"
            );
        }
    }

    #[test]
    fn depth_count_input_and_allocation_limits_are_enforced() {
        let bytes = b"<< /A [[[1 2 3]]] >>";
        for limits in [
            Limits {
                max_bytes: 8,
                ..Limits::default()
            },
            Limits {
                max_allocated_bytes: 8,
                ..Limits::default()
            },
            Limits {
                max_items: 2,
                ..Limits::default()
            },
            Limits {
                max_depth: 2,
                ..Limits::default()
            },
        ] {
            assert!(EngineData::parse(bytes, limits).is_err());
            let data = EngineData::parse(bytes, Limits::default()).unwrap();
            assert!(data.to_bytes(limits).is_err());
        }
    }

    #[test]
    fn cannot_emit_nonfinite_real_or_operator_injection_atom() {
        for value in [
            Value::Real(f64::NAN),
            Value::Atom(b"1 /Injected true".to_vec()),
        ] {
            assert!(
                EngineData::new(Node::new(value))
                    .to_bytes(Limits::default())
                    .is_err()
            );
        }
    }

    #[test]
    fn real_type_and_negative_zero_survive_reencoding() {
        for value in [0.0, -0.0, 1.0, f64::MIN_POSITIVE, f64::MAX] {
            let data = EngineData::new(Node::new(Value::Real(value)));
            let bytes = data.to_bytes(Limits::default()).unwrap();
            let parsed = EngineData::parse(&bytes, Limits::default()).unwrap();
            let Value::Real(actual) = parsed.root().value() else {
                panic!("lost real type");
            };
            assert_eq!(actual.to_bits(), value.to_bits());
        }
    }
}
