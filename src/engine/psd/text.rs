//! Live Type Tool mapping. The original TySh descriptor and EngineData tree
//! are the write template: unmodified values, unknown keys, resources, path
//! records, warp records and lexical bytes survive an edit to another field.
//!
//! Adobe documents the TySh envelope, but not EngineData. The run schema is
//! independently documented by ag-psd's text.ts. Photoshop comparison remains
//! necessary: a successful descriptor round trip does not prove text layout.

use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use super::descriptor::{self, Descriptor, Id, Limits, Value as D};
use super::engine_data::{EngineData, Node, Value as E};
use super::{LayerRecord, RawBytes};
use crate::engine::grid::RectI;
use crate::engine::text::{
    self, Caps, CharStyle, Composer, ImportedRun, ImportedText, LastLine, ParaStyle, Position,
    StyleRun, TextAlign, TextLayer, TextStyle, Warp, WarpStyle,
};

const MAX_TEXT_BYTES: usize = 1 << 20;
const STYLE_PATH: &[&[u8]] = &[b"StyleSheet", b"StyleSheetData"];
const PARA_PATH: &[&[u8]] = &[b"ParagraphSheet", b"Properties"];

struct TypeTool {
    header: Vec<u8>,
    transform: [f64; 6],
    text: Descriptor,
    warp: Descriptor,
    tail: Vec<u8>,
}

impl TypeTool {
    fn parse(bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() >= 56, "Truncated TySh header");
        ensure!(bytes[..2] == 1u16.to_be_bytes(), "Unsupported TySh version");
        ensure!(
            bytes[50..52] == 50u16.to_be_bytes(),
            "Unsupported TySh text version"
        );
        ensure!(
            bytes[52..56] == 16u32.to_be_bytes(),
            "Unsupported TySh descriptor version"
        );
        let mut transform = [0.0; 6];
        for (i, value) in transform.iter_mut().enumerate() {
            *value = f64::from_be_bytes(bytes[2 + i * 8..10 + i * 8].try_into()?);
            ensure!(
                value.is_finite() && value.abs() < 1.0e9,
                "Invalid TySh transform"
            );
        }
        let (text, used) = descriptor::decode_prefix(&bytes[56..], Limits::default())
            .context("TySh text descriptor")?;
        let warp_start = 56 + used;
        let warp_header = bytes
            .get(warp_start..warp_start + 6)
            .context("Truncated TySh warp header")?;
        ensure!(
            warp_header[..2] == 1u16.to_be_bytes() && warp_header[2..] == 16u32.to_be_bytes(),
            "Unsupported TySh warp version"
        );
        let (warp, used) = descriptor::decode_prefix(&bytes[warp_start + 6..], Limits::default())
            .context("TySh warp descriptor")?;
        let tail = bytes[warp_start + 6 + used..].to_vec();
        ensure!(tail.len() >= 16, "Truncated TySh bounds");
        Ok(Self {
            header: bytes[..56].to_vec(),
            transform,
            text,
            warp,
            tail,
        })
    }

    fn bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = self.header.clone();
        for (i, value) in self.transform.iter().enumerate() {
            bytes[2 + i * 8..10 + i * 8].copy_from_slice(&value.to_be_bytes());
        }
        bytes.extend(descriptor::encode(&self.text, Limits::default())?);
        bytes.extend(1u16.to_be_bytes());
        bytes.extend(16u32.to_be_bytes());
        bytes.extend(descriptor::encode(&self.warp, Limits::default())?);
        bytes.extend(&self.tail);
        Ok(bytes)
    }

    fn engine_data(&self) -> Result<EngineData> {
        let Some(D::Raw(bytes)) = self.text.get(b"EngineData") else {
            bail!("TySh has no supported EngineData text record")
        };
        EngineData::parse(bytes, Limits::default()).context("TySh EngineData")
    }
}

fn at<'a>(node: &'a Node, path: &[&[u8]]) -> Option<&'a Node> {
    path.iter().try_fold(node, |node, key| node.get(key))
}

fn at_mut<'a>(node: &'a mut Node, path: &[&[u8]]) -> Option<&'a mut Node> {
    path.iter().try_fold(node, |node, key| node.get_mut(key))
}

fn array(node: &Node) -> Result<&[Node]> {
    match node.value() {
        E::Array(values) => Ok(values),
        _ => bail!("Expected EngineData array"),
    }
}

fn number(node: &Node) -> Result<f32> {
    let value = match node.value() {
        E::Integer(v) => *v as f64,
        E::Real(v) => *v,
        _ => bail!("Expected EngineData number"),
    };
    ensure!(
        value.is_finite() && value.abs() < 1.0e7,
        "Out-of-range EngineData number"
    );
    Ok(value as f32)
}

fn integer(node: &Node) -> Result<i64> {
    match node.value() {
        E::Integer(value) => Ok(*value),
        _ => bail!("Expected EngineData integer"),
    }
}

fn num(node: &Node, key: &[u8], default: f32) -> Result<f32> {
    node.get(key).map_or(Ok(default), number)
}

fn int(node: &Node, key: &[u8], default: i64) -> Result<i64> {
    node.get(key).map_or(Ok(default), integer)
}

fn flag(node: &Node, key: &[u8], default: bool) -> Result<bool> {
    match node.get(key).map(Node::value) {
        None => Ok(default),
        Some(E::Boolean(value)) => Ok(*value),
        _ => bail!("Expected EngineData boolean"),
    }
}

fn overlay(target: &mut Node, source: Option<&Node>) -> Result<()> {
    if let Some(source) = source {
        let E::Dictionary(entries) = source.value() else {
            bail!("Expected EngineData style dictionary")
        };
        for entry in entries {
            target.set(entry.key(), entry.value().value().clone())?;
        }
    }
    Ok(())
}

fn merged_style(root: &Node, run: &Node, paragraph: bool) -> Result<Node> {
    let (group, path, normal, set) = if paragraph {
        (
            b"ParagraphRun".as_slice(),
            PARA_PATH,
            b"TheNormalParagraphSheet".as_slice(),
            b"ParagraphSheetSet".as_slice(),
        )
    } else {
        (
            b"StyleRun".as_slice(),
            STYLE_PATH,
            b"TheNormalStyleSheet".as_slice(),
            b"StyleSheetSet".as_slice(),
        )
    };
    let mut merged = Node::new(E::Dictionary(vec![]));
    if let Some(resources) = root.get(b"ResourceDict") {
        let index = usize::try_from(int(resources, normal, 0)?)
            .context("Negative default text style index")?;
        let normal_style = resources
            .get(set)
            .and_then(|n| n.get_index(index))
            .and_then(|n| {
                n.get(if paragraph {
                    b"Properties"
                } else {
                    b"StyleSheetData"
                })
            });
        overlay(&mut merged, normal_style)?;
    }
    overlay(
        &mut merged,
        at(root, &[b"EngineDict", group, b"DefaultRunData"]).and_then(|n| at(n, path)),
    )?;
    overlay(&mut merged, at(run, path))?;
    Ok(merged)
}

fn font_names(root: &Node) -> Result<Vec<String>> {
    let fonts = at(root, &[b"ResourceDict", b"FontSet"]).context("EngineData missing FontSet")?;
    array(fonts)?
        .iter()
        .map(|font| {
            font.get(b"Name")
                .context("FontSet entry missing Name")?
                .as_text()
        })
        .collect()
}

fn limitation(out: &mut Vec<String>, message: impl Into<String>) {
    let message = message.into();
    if !out.contains(&message) {
        out.push(message);
    }
}

fn read_char(node: &Node, fonts: &[String], limitations: &mut Vec<String>) -> Result<CharStyle> {
    let mut style = TextStyle {
        size: 12.0,
        ..TextStyle::default()
    }
    .char_style();
    let index = usize::try_from(int(node, b"Font", 0)?).context("Negative text font index")?;
    let name = fonts
        .get(index)
        .context("Text font index outside FontSet")?;
    if let Some((family, weight, italic)) = text::font_from_postscript(name) {
        style.family = family;
        style.weight = weight;
        style.italic = italic;
    } else {
        style.family = name.clone();
        limitation(limitations, format!("Font {name} is not installed"));
    }
    style.size = num(node, b"FontSize", 12.0)?;
    ensure!(
        style.size > 0.0 && style.size <= 16384.0,
        "Unsupported text font size"
    );
    style.leading = if flag(node, b"AutoLeading", true)? {
        0.0
    } else {
        num(node, b"Leading", 0.0)?
    };
    style.tracking = num(node, b"Tracking", 0.0)?;
    style.baseline_shift = num(node, b"BaselineShift", 0.0)?;
    style.faux_bold = flag(node, b"FauxBold", false)?;
    style.faux_italic = flag(node, b"FauxItalic", false)?;
    style.underline = flag(node, b"Underline", false)?;
    style.strike = flag(node, b"Strikethrough", false)?;
    style.features.liga = flag(node, b"Ligatures", true)?;
    style.features.dlig = flag(node, b"DLigatures", false)?;
    style.caps = match int(node, b"FontCaps", 0)? {
        0 => Caps::Normal,
        1 => Caps::Small,
        2 => Caps::All,
        _ => {
            limitation(limitations, "Unknown Photoshop capitalization mode");
            Caps::Normal
        }
    };
    style.position = match int(node, b"FontBaseline", 0)? {
        0 => Position::Normal,
        1 => Position::Super,
        2 => Position::Sub,
        _ => {
            limitation(limitations, "Unknown Photoshop baseline mode");
            Position::Normal
        }
    };
    if let Some(color) = node.get(b"FillColor") {
        let values = array(color.get(b"Values").context("Text color missing Values")?)?;
        if int(color, b"Type", 1)? == 1 && values.len() == 4 {
            let channels: Vec<f32> = values.iter().map(number).collect::<Result<_>>()?;
            ensure!(
                channels.iter().all(|v| (0.0..=1.0).contains(v)),
                "Invalid text color component"
            );
            if channels[0] != 1.0 {
                limitation(limitations, "Per-character fill transparency");
            }
            style.color = std::array::from_fn(|i| (channels[i + 1] * 255.0).round() as u8);
        } else {
            limitation(limitations, "Non-RGB character fill color");
        }
    }
    for (key, normal, label) in [
        (
            b"HorizontalScale".as_slice(),
            1.0,
            "Horizontal character scaling",
        ),
        (
            b"VerticalScale".as_slice(),
            1.0,
            "Vertical character scaling",
        ),
        (b"Kerning".as_slice(), 0.0, "Manual character kerning"),
        (b"Tsume".as_slice(), 0.0, "Tsume character compression"),
    ] {
        if num(node, key, normal)? != normal {
            limitation(limitations, label);
        }
    }
    if flag(node, b"StrokeFlag", false)? {
        limitation(limitations, "Stroked character outlines");
    }
    if !flag(node, b"FillFlag", true)? {
        limitation(limitations, "Characters without fill");
    }
    if flag(node, b"NoBreak", false)? {
        limitation(limitations, "Character no-break ranges");
    }
    Ok(style)
}

fn read_para(node: &Node, limitations: &mut Vec<String>) -> Result<ParaStyle> {
    let (align, last) = match int(node, b"Justification", 0)? {
        0 => (TextAlign::Left, LastLine::Left),
        1 => (TextAlign::Right, LastLine::Left),
        2 => (TextAlign::Center, LastLine::Left),
        3 => (TextAlign::Justify, LastLine::Left),
        4 => (TextAlign::Justify, LastLine::Right),
        5 => (TextAlign::Justify, LastLine::Center),
        6 => (TextAlign::Justify, LastLine::All),
        _ => {
            limitation(limitations, "Unknown paragraph justification");
            (TextAlign::Left, LastLine::Left)
        }
    };
    if num(node, b"AutoLeading", 1.2)? != 1.2 {
        limitation(limitations, "Custom automatic line spacing");
    }
    for key in [b"Hanging".as_slice(), b"Burasagari"] {
        if flag(node, key, false)? {
            limitation(limitations, "Hanging punctuation");
        }
    }
    Ok(ParaStyle {
        align,
        last,
        indent_left: num(node, b"StartIndent", 0.0)?,
        indent_right: num(node, b"EndIndent", 0.0)?,
        indent_first: num(node, b"FirstLineIndent", 0.0)?,
        space_before: num(node, b"SpaceBefore", 0.0)?,
        space_after: num(node, b"SpaceAfter", 0.0)?,
        hyphenate: flag(node, b"AutoHyphenate", true)?,
        composer: if flag(node, b"EveryLineComposer", false)? {
            Composer::Every
        } else {
            Composer::Single
        },
    })
}

/// Validate UTF-16 run lengths, including the editor's terminal CR, and map
/// them to UTF-8 byte offsets without rounding across a surrogate pair.
fn ranges<'a>(
    root: &'a Node,
    group: &[u8],
    editor: &str,
    visible_len: usize,
) -> Result<Vec<(usize, usize, &'a Node)>> {
    let runs = at(root, &[b"EngineDict", group]).context("Missing EngineData runs")?;
    let nodes = array(runs.get(b"RunArray").context("Missing text RunArray")?)?;
    let lengths = array(
        runs.get(b"RunLengthArray")
            .context("Missing text RunLengthArray")?,
    )?;
    ensure!(
        nodes.len() == lengths.len(),
        "Text run arrays have different counts"
    );
    let mut boundaries = vec![None; editor.encode_utf16().count() + 1];
    let mut units = 0;
    for (byte, ch) in editor.char_indices() {
        boundaries[units] = Some(byte);
        units += ch.len_utf16();
    }
    boundaries[units] = Some(editor.len());
    let mut start_units = 0usize;
    let mut out = Vec::with_capacity(nodes.len());
    for (node, length) in nodes.iter().zip(lengths) {
        let length = usize::try_from(integer(length)?).context("Negative text run length")?;
        ensure!(length > 0, "Empty Photoshop text run");
        let end_units = start_units
            .checked_add(length)
            .context("Text run length overflow")?;
        let start = boundaries
            .get(start_units)
            .copied()
            .flatten()
            .context("Text run splits a UTF-16 surrogate pair")?;
        let end = boundaries
            .get(end_units)
            .copied()
            .flatten()
            .context("Text run exceeds text or splits a UTF-16 surrogate pair")?;
        out.push((start.min(visible_len), end.min(visible_len), node));
        start_units = end_units;
    }
    ensure!(
        start_units == units,
        "Text run lengths do not cover Editor.Text"
    );
    Ok(out)
}

const WARPS: &[(&[u8], WarpStyle)] = &[
    (b"warpNone", WarpStyle::None),
    (b"warpArc", WarpStyle::Arc),
    (b"warpArcLower", WarpStyle::ArcLower),
    (b"warpArcUpper", WarpStyle::ArcUpper),
    (b"warpArch", WarpStyle::Arch),
    (b"warpBulge", WarpStyle::Bulge),
    (b"warpShellLower", WarpStyle::ShellLower),
    (b"warpShellUpper", WarpStyle::ShellUpper),
    (b"warpFlag", WarpStyle::Flag),
    (b"warpWave", WarpStyle::Wave),
    (b"warpFish", WarpStyle::Fish),
    (b"warpRise", WarpStyle::Rise),
    (b"warpFisheye", WarpStyle::Fisheye),
    (b"warpInflate", WarpStyle::Inflate),
    (b"warpSqueeze", WarpStyle::Squeeze),
    (b"warpTwist", WarpStyle::Twist),
];

fn enum_value<'a>(descriptor: &'a Descriptor, key: &[u8]) -> Option<&'a [u8]> {
    match descriptor.get(key) {
        Some(D::Enumerated { value, .. }) => Some(value.bytes()),
        _ => None,
    }
}

fn read_warp(descriptor: &Descriptor, limitations: &mut Vec<String>) -> Result<Warp> {
    let name = enum_value(descriptor, b"warpStyle").unwrap_or(b"warpNone");
    let style = WARPS
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| *value)
        .unwrap_or_else(|| {
            limitation(limitations, "Custom Photoshop text warp mesh");
            WarpStyle::None
        });
    let value = |key| -> Result<f32> {
        let v = descriptor.get(key).and_then(D::as_f64).unwrap_or(0.0);
        ensure!(
            v.is_finite() && (-100.0..=100.0).contains(&v),
            "Invalid text warp amount"
        );
        Ok(v as f32 / 100.0)
    };
    Ok(Warp {
        style,
        bend: value(b"warpValue")?,
        h_distort: value(b"warpPerspective")?,
        v_distort: value(b"warpPerspectiveOther")?,
        vertical: enum_value(descriptor, b"warpRotate") == Some(b"Vrtc"),
    })
}

fn shape(root: &Node) -> Option<&Node> {
    at(root, &[b"EngineDict", b"Rendered", b"Shapes", b"Children"])?
        .get_index(0)
        .and_then(|n| at(n, &[b"Cookie", b"Photoshop"]))
}

/// Txt2 is document-level data, so the document mapper supplies this context.
/// Current Photoshop writes both logical runs and a glyph cache there; changing
/// TySh alone would leave contradictory copies of the same text in the file.
pub(crate) fn mark_global_cache(layer: &mut TextLayer) {
    if let Some(source) = &mut layer.imported {
        limitation(
            &mut Arc::make_mut(source).limitations,
            "Photoshop text cache cannot yet be updated; text edits cannot be saved",
        );
    }
}

/// Import logical text while the caller keeps the layer's stored pixel grid.
pub(crate) fn decode(record: &LayerRecord) -> Result<TextLayer> {
    let block = record
        .additional_info
        .iter()
        .find(|block| block.key == *b"TySh")
        .context("Layer has no TySh record")?;
    let tool = TypeTool::parse(block.data.as_slice())?;
    let engine = tool.engine_data()?;
    let root = engine.root();
    let mut editor = at(root, &[b"EngineDict", b"Editor", b"Text"])
        .context("Missing Editor.Text")?
        .as_text()?;
    ensure!(
        editor.len() <= MAX_TEXT_BYTES,
        "Text layer exceeds editable text size limit"
    );
    // Photoshop appends one paragraph sentinel. Preserve intentional empty
    // user paragraphs by removing exactly one CR, never all trailing CRs.
    ensure!(
        editor.ends_with('\r'),
        "Editor.Text is missing its terminal paragraph sentinel"
    );
    editor = editor.replace('\r', "\n");
    let visible = &editor[..editor.len() - 1];
    let fonts = font_names(root)?;
    let mut limitations = Vec::new();
    let mut runs = Vec::new();
    let mut character_origins = Vec::new();
    let mut empty_character_origin = 0;
    for (index, (start, end, node)) in ranges(root, b"StyleRun", &editor, visible.len())?
        .into_iter()
        .enumerate()
    {
        let style = read_char(&merged_style(root, node, false)?, &fonts, &mut limitations)?;
        empty_character_origin = index;
        if end > start {
            character_origins.push(ImportedRun {
                start,
                end,
                source: index,
            });
            runs.push(StyleRun { start, end, style });
        }
    }
    let default = if let Some(run) = runs.first() {
        run.style.clone()
    } else {
        let node = at(root, &[b"EngineDict", b"StyleRun", b"DefaultRunData"])
            .context("Empty text missing default style")?;
        read_char(&merged_style(root, node, false)?, &fonts, &mut limitations)?
    };
    let paragraph_runs = ranges(root, b"ParagraphRun", &editor, visible.len())?;
    let mut paras = Vec::new();
    let mut paragraph_origins = Vec::new();
    let mut position = 0;
    for paragraph in visible.split('\n') {
        let (index, (_, _, node)) = paragraph_runs
            .iter()
            .enumerate()
            .find(|(_, (start, end, _))| {
                *start <= position && (*end > position || position == visible.len())
            })
            .context("Paragraph style missing at text position")?;
        paragraph_origins.push(index);
        paras.push(read_para(
            &merged_style(root, node, true)?,
            &mut limitations,
        )?);
        position += paragraph.len() + 1;
    }
    let [a, b, c, d, tx, ty] = tool.transform;
    if [a, b, c, d] != [1.0, 0.0, 0.0, 1.0] {
        limitation(
            &mut limitations,
            "Affine transformed text (scale, rotation, shear or reflection)",
        );
    }
    let vertical = enum_value(&tool.text, b"Ornt") == Some(b"Vrtc");
    let mut origin = (tx as f32, ty as f32);
    let mut box_size = None;
    if let Some(shape) = shape(root) {
        match int(shape, b"ShapeType", 0)? {
            0 => {
                if let Some(point) = shape.get(b"PointBase") {
                    let point = array(point)?;
                    ensure!(point.len() == 2, "Invalid point-text origin");
                    origin.0 += number(&point[0])?;
                    origin.1 += number(&point[1])?;
                }
            }
            1 => {
                let bounds = array(
                    shape
                        .get(b"BoxBounds")
                        .context("Paragraph text missing BoxBounds")?,
                )?;
                ensure!(bounds.len() == 4, "Invalid paragraph-text bounds");
                let bounds: Vec<f32> = bounds.iter().map(number).collect::<Result<_>>()?;
                ensure!(
                    bounds[2] >= bounds[0] && bounds[3] >= bounds[1],
                    "Inverted paragraph-text bounds"
                );
                origin.0 += bounds[0];
                origin.1 += bounds[1];
                box_size = Some((bounds[2] - bounds[0], bounds[3] - bounds[1]));
            }
            _ => limitation(&mut limitations, "Photoshop path or area text frame"),
        }
    }
    if at(root, &[b"ResourceDict", b"TextFrameSet"]).is_some()
        || tool.text.get(b"textShape").is_some()
    {
        // Newer Photoshop path frames require document-level Txt2 indices.
        // Retain them verbatim; an approximate rectangular conversion is lossy.
        limitation(
            &mut limitations,
            "Photoshop textShape/path frame descriptors",
        );
    }
    if record
        .additional_info
        .iter()
        .any(|block| block.key == *b"Txt2")
    {
        limitation(
            &mut limitations,
            "Additional Photoshop Txt2 text engine state",
        );
    }
    let style = TextStyle {
        family: default.family.clone(),
        weight: default.weight,
        italic: default.italic,
        size: default.size,
        leading: default.leading,
        tracking: default.tracking,
        color: default.color,
        align: paras.first().map_or(TextAlign::Left, |p| p.align),
    };
    let mut layer = TextLayer::new(visible, style, origin, box_size, vertical);
    layer.runs = runs;
    layer.paras = paras;
    layer.warp = read_warp(&tool.warp, &mut limitations)?;
    layer.relayout();
    layer.imported = Some(Arc::new(ImportedText {
        raster_bounds: RectI::new(
            record.bounds[1],
            record.bounds[0],
            record.bounds[3],
            record.bounds[2],
        ),
        limitations,
        character_origins,
        paragraph_origins,
        empty_character_origin,
        logical_text: layer.text.clone(),
        identity: Arc::new(()),
    }));
    Ok(layer)
}

fn set_num(node: &mut Node, key: &[u8], value: f32) -> Result<()> {
    ensure!(value.is_finite(), "Cannot save a non-finite text attribute");
    node.set(key, E::Real(f64::from(value)))
}

fn write_char(
    node: &mut Node,
    old: &CharStyle,
    new: &CharStyle,
    font_index: Option<usize>,
) -> Result<()> {
    if let Some(index) = font_index {
        node.set(b"Font", E::Integer(i64::try_from(index)?))?;
    }
    for (key, before, after) in [
        (b"FontSize".as_slice(), old.size, new.size),
        (b"Leading", old.leading, new.leading),
        (b"Tracking", old.tracking, new.tracking),
        (b"BaselineShift", old.baseline_shift, new.baseline_shift),
    ] {
        if before != after {
            set_num(node, key, after)?;
        }
    }
    if old.leading != new.leading {
        node.set(b"AutoLeading", E::Boolean(new.leading <= 0.0))?;
    }
    for (key, before, after) in [
        (b"FauxBold".as_slice(), old.faux_bold, new.faux_bold),
        (b"FauxItalic", old.faux_italic, new.faux_italic),
        (b"Underline", old.underline, new.underline),
        (b"Strikethrough", old.strike, new.strike),
        (b"Ligatures", old.features.liga, new.features.liga),
        (b"DLigatures", old.features.dlig, new.features.dlig),
    ] {
        if before != after {
            node.set(key, E::Boolean(after))?;
        }
    }
    if old.caps != new.caps {
        node.set(
            b"FontCaps",
            E::Integer(match new.caps {
                Caps::Normal => 0,
                Caps::Small => 1,
                Caps::All => 2,
            }),
        )?;
    }
    if old.position != new.position {
        node.set(
            b"FontBaseline",
            E::Integer(match new.position {
                Position::Normal => 0,
                Position::Super => 1,
                Position::Sub => 2,
            }),
        )?;
    }
    if old.color != new.color {
        if node.get(b"FillColor").is_none() {
            node.set(b"FillColor", E::Dictionary(vec![]))?;
        }
        let color = node
            .get_mut(b"FillColor")
            .context("Cannot edit text color without source FillColor")?;
        color.set(b"Type", E::Integer(1))?;
        color.set(
            b"Values",
            E::Array(
                std::iter::once(1.0)
                    .chain(new.color.map(|c| f64::from(c) / 255.0))
                    .map(|c| Node::new(E::Real(c)))
                    .collect(),
            ),
        )?;
    }
    ensure!(
        new.size > 0.0 && new.size <= 16384.0,
        "Unsupported edited text size"
    );
    ensure!(
        (
            old.features.onum,
            old.features.swsh,
            old.features.frac,
            old.features.ss
        ) == (
            new.features.onum,
            new.features.swsh,
            new.features.frac,
            new.features.ss
        ),
        "Writing these OpenType feature changes to Photoshop EngineData is not supported"
    );
    Ok(())
}

fn write_para(node: &mut Node, old: &ParaStyle, new: &ParaStyle) -> Result<()> {
    if (old.align, old.last) != (new.align, new.last) {
        let value = match new.align {
            TextAlign::Left => 0,
            TextAlign::Right => 1,
            TextAlign::Center => 2,
            TextAlign::Justify => match new.last {
                LastLine::Left => 3,
                LastLine::Right => 4,
                LastLine::Center => 5,
                LastLine::All => 6,
            },
        };
        node.set(b"Justification", E::Integer(value))?;
    }
    for (key, before, after) in [
        (b"StartIndent".as_slice(), old.indent_left, new.indent_left),
        (b"EndIndent", old.indent_right, new.indent_right),
        (b"FirstLineIndent", old.indent_first, new.indent_first),
        (b"SpaceBefore", old.space_before, new.space_before),
        (b"SpaceAfter", old.space_after, new.space_after),
    ] {
        if before != after {
            set_num(node, key, after)?;
        }
    }
    if old.hyphenate != new.hyphenate {
        node.set(b"AutoHyphenate", E::Boolean(new.hyphenate))?;
    }
    if old.composer != new.composer {
        node.set(
            b"EveryLineComposer",
            E::Boolean(new.composer == Composer::Every),
        )?;
    }
    Ok(())
}

fn write_warp(descriptor: &mut Descriptor, old: Warp, new: Warp) {
    if old.style != new.style {
        let name = WARPS
            .iter()
            .find(|(_, value)| *value == new.style)
            .map_or(b"warpNone".as_slice(), |(name, _)| *name);
        descriptor.set(
            Id::String(b"warpStyle".to_vec()),
            D::Enumerated {
                type_id: Id::String(b"warpStyle".to_vec()),
                value: Id::String(name.to_vec()),
            },
        );
    }
    for (key, before, after) in [
        (b"warpValue".as_slice(), old.bend, new.bend),
        (b"warpPerspective", old.h_distort, new.h_distort),
        (b"warpPerspectiveOther", old.v_distort, new.v_distort),
    ] {
        if before != after {
            descriptor.set(
                Id::String(key.to_vec()),
                D::double(f64::from(after) * 100.0),
            );
        }
    }
    if old.vertical != new.vertical {
        descriptor.set(
            Id::String(b"warpRotate".to_vec()),
            D::Enumerated {
                type_id: Id::Four(*b"Ornt"),
                value: Id::Four(if new.vertical { *b"Vrtc" } else { *b"Hrzn" }),
            },
        );
    }
}

/// Update only the fields represented by an actual user edit. Unsupported
/// source features and unsupported new edits fail before mutating the record.
pub(crate) fn encode(record: &mut LayerRecord, old: &TextLayer, new: &TextLayer) -> Result<()> {
    if old == new {
        return Ok(());
    }
    ensure!(
        old.imported_text_limitations().is_empty(),
        "Cannot safely edit this Photoshop text layer: {}",
        old.imported_text_limitations().join("; ")
    );
    if let Some(source) = &old.imported {
        let current = new
            .imported
            .as_ref()
            .context("Imported text lost its source provenance")?;
        ensure!(
            Arc::ptr_eq(&source.identity, &current.identity) && current.logical_text == new.text,
            "Text was replaced without retaining Photoshop run provenance"
        );
    }
    ensure!(
        old.path == new.path,
        "Writing Photoshop text path/area changes is not supported"
    );
    ensure!(
        old.upright_roman == new.upright_roman && old.tcy == new.tcy,
        "Writing vertical Roman alignment or tate-chu-yoko changes is not supported"
    );
    ensure!(
        old.vertical == new.vertical,
        "Changing imported Photoshop text orientation is not supported"
    );
    ensure!(
        old.box_size.is_some() == new.box_size.is_some(),
        "Converting imported Photoshop point/paragraph text is not supported"
    );
    ensure!(
        new.text.len() <= MAX_TEXT_BYTES && !new.text.contains('\r'),
        "Invalid or oversized edited text"
    );
    ensure!(
        new.paras.len() == new.text.split('\n').count(),
        "Paragraph styles do not cover edited text"
    );
    let mut position = 0;
    for run in &new.runs {
        ensure!(
            run.start == position
                && run.end > run.start
                && new.text.is_char_boundary(run.start)
                && new.text.is_char_boundary(run.end),
            "Invalid edited text style run"
        );
        position = run.end;
    }
    ensure!(
        position == new.text.len(),
        "Character styles do not cover edited text"
    );
    let index = record
        .additional_info
        .iter()
        .position(|block| block.key == *b"TySh")
        .context("Layer has no TySh record")?;
    let mut tool = TypeTool::parse(record.additional_info[index].data.as_slice())?;
    let mut engine = tool.engine_data()?;
    update_engine(&mut engine, old, new)?;
    if old.text != new.text {
        tool.text
            .set(Id::Four(*b"Txt "), D::text(&new.text.replace('\n', "\r")));
    }
    tool.text.set(
        Id::String(b"EngineData".to_vec()),
        D::Raw(engine.to_bytes(Limits::default())?),
    );
    tool.transform[4] += f64::from(new.origin.0 - old.origin.0);
    tool.transform[5] += f64::from(new.origin.1 - old.origin.1);
    ensure!(
        tool.transform.iter().all(|value| value.is_finite()),
        "Invalid edited text transform"
    );
    write_warp(&mut tool.warp, old.warp, new.warp);
    let mut bytes = tool.bytes()?;
    if bytes.len() % 2 != 0 {
        bytes.push(0);
    }
    record.additional_info[index].data = RawBytes::from_vec(bytes);
    Ok(())
}

/// Map a single replaced span using equal Unicode prefixes and suffixes. This
/// keeps opaque character metadata on untouched text when insertion shifts all
/// subsequent UTF-8 offsets. Multiple disjoint edits conservatively share the
/// replaced span; source metadata within that span cannot be assigned safely.
struct TextChange {
    prefix: usize,
    old_tail: usize,
    new_tail: usize,
}

impl TextChange {
    fn between(old: &str, new: &str) -> Self {
        let prefix = old
            .chars()
            .zip(new.chars())
            .take_while(|(a, b)| a == b)
            .map(|(c, _)| c.len_utf8())
            .sum();
        let suffix: usize = old[prefix..]
            .chars()
            .rev()
            .zip(new[prefix..].chars().rev())
            .take_while(|(a, b)| a == b)
            .map(|(c, _)| c.len_utf8())
            .sum();
        Self {
            prefix,
            old_tail: old.len() - suffix,
            new_tail: new.len() - suffix,
        }
    }

    fn source_position(&self, byte: usize) -> usize {
        if byte < self.prefix {
            byte
        } else if byte >= self.new_tail {
            self.old_tail + byte - self.new_tail
        } else {
            self.prefix
        }
    }

    fn destination_boundary(&self, byte: usize) -> Option<usize> {
        if byte <= self.prefix {
            Some(byte)
        } else if byte >= self.old_tail {
            Some(self.new_tail + byte - self.old_tail)
        } else {
            None
        }
    }
}

fn update_engine(engine: &mut EngineData, old: &TextLayer, new: &TextLayer) -> Result<()> {
    let root = engine.root();
    let editor = at(root, &[b"EngineDict", b"Editor", b"Text"])
        .context("Missing Editor.Text")?
        .as_text()?
        .replace('\r', "\n");
    let source_runs = ranges(root, b"StyleRun", &editor, old.text.len())?;
    let source_paras = ranges(root, b"ParagraphRun", &editor, old.text.len())?;
    let mut fonts = font_names(root)?;
    let original_font_count = fonts.len();
    let change = TextChange::between(&old.text, &new.text);
    let mut new_runs = Vec::new();
    let mut run_lengths = Vec::new();
    let effective_runs = if new.text.is_empty() {
        vec![StyleRun {
            start: 0,
            end: 0,
            style: new.style.char_style(),
        }]
    } else {
        // Do not merge source runs whose mapped styles happen to be equal:
        // their unknown character attributes may differ.
        let mut split_runs = Vec::new();
        let mut boundaries: Vec<usize> = if let Some(source) = &new.imported {
            source.character_origins.iter().map(|run| run.end).collect()
        } else {
            let mut boundaries: Vec<_> = source_runs
                .iter()
                .filter_map(|(_, end, _)| change.destination_boundary(*end))
                .collect();
            boundaries.extend([change.prefix, change.new_tail]);
            boundaries
        };
        boundaries.sort_unstable();
        boundaries.dedup();
        for run in &new.runs {
            let mut start = run.start;
            for end in boundaries
                .iter()
                .copied()
                .filter(|end| *end > run.start && *end < run.end)
                .chain(std::iter::once(run.end))
            {
                split_runs.push(StyleRun {
                    start,
                    end,
                    style: run.style.clone(),
                });
                start = end;
            }
        }
        split_runs
    };
    for (index, run) in effective_runs.iter().enumerate() {
        let (_, _, source) = if let Some(provenance) = &new.imported {
            let source_index = provenance
                .character_origins
                .iter()
                .find(|candidate| candidate.start <= run.start && candidate.end > run.start)
                .map_or(provenance.empty_character_origin, |run| run.source);
            source_runs.get(source_index)
        } else {
            let source_at = change.source_position(run.start).min(old.text.len());
            source_runs.iter().find(|(start, end, _)| {
                *start <= source_at && (*end > source_at || source_at == old.text.len())
            })
        }
        .context("Missing source character run")?;
        let old_style = read_char(&merged_style(root, source, false)?, &fonts, &mut Vec::new())?;
        let mut node = (*source).clone();
        let font_changed = (
            old_style.family.as_str(),
            old_style.weight,
            old_style.italic,
        ) != (
            run.style.family.as_str(),
            run.style.weight,
            run.style.italic,
        );
        let font = if font_changed {
            let name =
                text::postscript_from_font(&run.style.family, run.style.weight, run.style.italic)
                    .context("Edited text font is not installed")?;
            Some(
                if let Some(index) = fonts.iter().position(|font| *font == name) {
                    index
                } else {
                    fonts.push(name);
                    fonts.len() - 1
                },
            )
        } else {
            None
        };
        write_char(
            at_mut(&mut node, STYLE_PATH).context("Missing source StyleSheetData")?,
            &old_style,
            &run.style,
            font,
        )?;
        new_runs.push(node);
        let count = new.text[run.start..run.end].encode_utf16().count()
            + usize::from(index + 1 == effective_runs.len());
        run_lengths.push(Node::new(E::Integer(i64::try_from(count)?)));
    }
    let mut new_paras = Vec::new();
    let mut para_lengths = Vec::new();
    let mut new_position = 0;
    for (index, paragraph) in new.text.split('\n').enumerate() {
        let (_, _, source) = if let Some(provenance) = &new.imported {
            provenance
                .paragraph_origins
                .get(index)
                .and_then(|index| source_paras.get(*index))
        } else {
            let old_position = change.source_position(new_position).min(old.text.len());
            source_paras
                .iter()
                .find(|(start, end, _)| {
                    *start <= old_position
                        && (*end > old_position || old_position == old.text.len())
                })
                .or(source_paras.last())
        }
        .context("Missing source paragraph run")?;
        let old_style = read_para(&merged_style(root, source, true)?, &mut Vec::new())?;
        let mut node = (*source).clone();
        write_para(
            at_mut(&mut node, PARA_PATH).context("Missing paragraph Properties")?,
            &old_style,
            &new.paras[index],
        )?;
        new_paras.push(node);
        para_lengths.push(Node::new(E::Integer(i64::try_from(
            paragraph.encode_utf16().count() + 1,
        )?)));
        new_position += paragraph.len() + 1;
    }
    let root = engine.root_mut();
    if old.text != new.text {
        at_mut(root, &[b"EngineDict", b"Editor", b"Text"])
            .context("Missing Editor.Text")?
            .replace(
                Node::text(&(new.text.replace('\n', "\r") + "\r"))
                    .value()
                    .clone(),
            );
    }
    if old.runs != new.runs
        || old.text != new.text
        || (old.text.is_empty() && old.style != new.style)
    {
        let group = at_mut(root, &[b"EngineDict", b"StyleRun"]).context("Missing StyleRun")?;
        group.set(b"RunArray", E::Array(new_runs))?;
        group.set(b"RunLengthArray", E::Array(run_lengths))?;
    }
    if old.paras != new.paras || old.text != new.text {
        let group =
            at_mut(root, &[b"EngineDict", b"ParagraphRun"]).context("Missing ParagraphRun")?;
        group.set(b"RunArray", E::Array(new_paras))?;
        group.set(b"RunLengthArray", E::Array(para_lengths))?;
    }
    if fonts.len() != original_font_count {
        for resources in [b"ResourceDict".as_slice(), b"DocumentResources"] {
            let Some(set) = at_mut(root, &[resources, b"FontSet"]) else {
                continue;
            };
            let mut values = array(set)?.to_vec();
            ensure!(
                values.len() == original_font_count,
                "Mismatching Photoshop document font resources"
            );
            for name in &fonts[original_font_count..] {
                let mut font = Node::new(E::Dictionary(vec![]));
                font.set(b"Name", Node::text(name).value().clone())?;
                for key in [b"Script".as_slice(), b"FontType", b"Synthetic"] {
                    font.set(key, E::Integer(0))?;
                }
                values.push(font);
            }
            set.replace(E::Array(values));
        }
    }
    if old.box_size != new.box_size {
        let shape = at_mut(root, &[b"EngineDict", b"Rendered", b"Shapes", b"Children"])
            .and_then(|n| n.get_index_mut(0))
            .and_then(|n| at_mut(n, &[b"Cookie", b"Photoshop"]))
            .context("Missing editable text box")?;
        let bounds = array(shape.get(b"BoxBounds").context("Missing BoxBounds")?)?;
        let x = number(&bounds[0])?;
        let y = number(&bounds[1])?;
        let (width, height) = new.box_size.context("Missing edited text box")?;
        ensure!(
            width.is_finite() && height.is_finite() && width >= 0.0 && height >= 0.0,
            "Invalid edited text box"
        );
        shape.set(
            b"BoxBounds",
            E::Array(
                [x, y, x + width, y + height]
                    .map(|v| Node::new(E::Real(f64::from(v))))
                    .to_vec(),
            ),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::{AdditionalInfo, descriptor::Class};
    use super::*;

    fn fixture(text: &str) -> LayerRecord {
        let length = text.encode_utf16().count() + 1;
        let mut engine = EngineData::parse(format!(r#"<<
 /EngineDict <<
  /Editor << /Text <FEFF000D> >>
  /StyleRun <<
   /DefaultRunData << /StyleSheet << /StyleSheetData << /Font 0 /FontSize 12 >> >> >>
   /RunArray [ << /StyleSheet << /StyleSheetData << /Font 0 /FontSize 12 /Tracking 15 /FontCaps 1 /FontBaseline 2 /Underline true /FillColor << /Type 1 /Values [1 0.2 0.4 0.6] >> /UnknownStyle +01.2300 >> >> /UnknownRun <123456> >> ]
   /RunLengthArray [{length}]
  >>
  /ParagraphRun <<
   /RunArray [ << /ParagraphSheet << /Properties << /Justification 5 /StartIndent 3 /EveryLineComposer true /UnknownParagraph (keep\\me) >> >> >> ]
   /RunLengthArray [{length}]
  >>
  /Rendered << /Shapes << /Children [ << /Cookie << /Photoshop << /ShapeType 1 /BoxBounds [2 4 202 104] /UnknownShape /keep >> >> >> ] >> >>
 >>
 /ResourceDict << /FontSet [ << /Name <FEFF004600690078007400750072006500500053> /FontType 0 /UnknownFont 999 >> ] >>
 /UnknownRoot +0007
>>
"#).as_bytes(), Limits::default()).unwrap();
        at_mut(engine.root_mut(), &[b"EngineDict", b"Editor", b"Text"])
            .unwrap()
            .replace(
                Node::text(&(text.replace('\n', "\r") + "\r"))
                    .value()
                    .clone(),
            );
        let mut descriptor = Descriptor::new(Class::new("", Id::Four(*b"TxLr")));
        descriptor.set(Id::Four(*b"Txt "), D::text(&text.replace('\n', "\r")));
        descriptor.set(
            Id::String(b"EngineData".to_vec()),
            D::Raw(engine.to_bytes(Limits::default()).unwrap()),
        );
        descriptor.set(Id::String(b"private".to_vec()), D::Raw(vec![0, 255, 4, 0]));
        let mut warp = Descriptor::new(Class::new("", Id::String(b"warp".to_vec())));
        warp.set(
            Id::String(b"warpStyle".to_vec()),
            D::Enumerated {
                type_id: Id::String(b"warpStyle".to_vec()),
                value: Id::String(b"warpNone".to_vec()),
            },
        );
        warp.set(Id::Four(*b"priv"), D::Boolean(3));
        let mut header = 1u16.to_be_bytes().to_vec();
        header.resize(50, 0);
        header.extend(50u16.to_be_bytes());
        header.extend(16u32.to_be_bytes());
        let tool = TypeTool {
            header,
            transform: [1.0, 0.0, 0.0, 1.0, 10.0, 20.0],
            text: descriptor,
            warp,
            tail: vec![0; 16],
        };
        let mut record = LayerRecord::new([1, 2, 80, 240], "type");
        record
            .additional_info
            .push(AdditionalInfo::new(*b"TySh", tool.bytes().unwrap()));
        record
    }

    /// Font availability is deliberately outside these wire-format tests.
    /// The missing-font test below separately verifies the edit refusal.
    fn editable(record: &LayerRecord) -> TextLayer {
        let mut layer = decode(record).unwrap();
        Arc::make_mut(layer.imported.as_mut().unwrap())
            .limitations
            .clear();
        layer
    }

    fn tysh(record: &LayerRecord) -> &AdditionalInfo {
        record.additional_info.iter().find(|block| block.key == *b"TySh").unwrap()
    }

    fn replace_tysh(record: &mut LayerRecord, bytes: Vec<u8>) {
        let block = record.additional_info.iter_mut().find(|block| block.key == *b"TySh").unwrap();
        *block = AdditionalInfo::new(*b"TySh", bytes);
    }

    #[test]
    fn maps_utf16_text_character_and_paragraph_styles_and_box_without_rasterizing() {
        let record = fixture("A😀\n\n");
        let layer = decode(&record).unwrap();
        assert_eq!(layer.text, "A😀\n\n");
        assert_eq!(layer.runs[0].end, layer.text.len());
        assert_eq!(layer.runs[0].style.caps, Caps::Small);
        assert_eq!(layer.runs[0].style.position, Position::Sub);
        assert_eq!(layer.runs[0].style.color, [51, 102, 153]);
        assert!(layer.runs[0].style.underline);
        assert_eq!(layer.paras.len(), 3);
        assert_eq!(layer.paras[0].align, TextAlign::Justify);
        assert_eq!(layer.paras[0].last, LastLine::Center);
        assert_eq!(layer.paras[0].composer, Composer::Every);
        assert_eq!(layer.origin, (12.0, 24.0));
        assert_eq!(layer.box_size, Some((200.0, 100.0)));
        assert_eq!(
            layer.imported.as_ref().unwrap().raster_bounds,
            RectI::new(2, 1, 240, 80)
        );
    }

    #[test]
    fn no_op_retains_every_byte_even_with_unsupported_font() {
        let mut record = fixture("A😀");
        let original = record.clone();
        let layer = decode(&record).unwrap();
        assert!(!layer.imported_text_limitations().is_empty());
        encode(&mut record, &layer, &layer).unwrap();
        assert_eq!(record, original);
        let source = &tysh(&record).data;
        assert_eq!(
            TypeTool::parse(source.as_slice()).unwrap().bytes().unwrap(),
            source.as_slice()
        );
    }

    #[test]
    fn changed_known_field_preserves_unknown_descriptor_and_engine_lexemes() {
        let mut record = fixture("A😀BC");
        let old = editable(&record);
        let before = TypeTool::parse(tysh(&record).data.as_slice()).unwrap();
        let other_blocks: Vec<_> = record.additional_info.iter().filter(|block| block.key != *b"TySh").cloned().collect();
        let mut new = old.clone();
        new.set_char_style(1, 5, |style| style.tracking = 101.0);
        encode(&mut record, &old, &new).unwrap();
        let after = TypeTool::parse(tysh(&record).data.as_slice()).unwrap();
        assert_eq!(record.additional_info.iter().filter(|block| block.key != *b"TySh").cloned().collect::<Vec<_>>(), other_blocks);
        assert_eq!(before.text.get(b"private"), after.text.get(b"private"));
        assert_eq!(before.warp, after.warp);
        let data = after
            .engine_data()
            .unwrap()
            .to_bytes(Limits::default())
            .unwrap();
        for untouched in [
            b"/UnknownRoot +0007".as_slice(),
            b"/UnknownStyle +01.2300",
            b"/UnknownRun <123456>",
            b"/UnknownFont 999",
            b"/UnknownShape /keep",
        ] {
            assert!(
                data.windows(untouched.len())
                    .any(|window| window == untouched)
            );
        }
        let mapped = decode(&record).unwrap();
        assert_eq!(
            mapped
                .runs
                .iter()
                .map(|run| (run.start, run.end, run.style.tracking as i32))
                .collect::<Vec<_>>(),
            [(0, 1, 15), (1, 5, 101), (5, 7, 15)]
        );
        assert_eq!(tysh(&record).data.len() % 2, 0);
    }

    #[test]
    fn separate_edits_keep_unknown_run_origins_and_valid_utf16_lengths() {
        let mut record = fixture("ABCD");
        let mut tool = TypeTool::parse(tysh(&record).data.as_slice()).unwrap();
        let mut engine = tool.engine_data().unwrap();
        let group = at_mut(engine.root_mut(), &[b"EngineDict", b"StyleRun"]).unwrap();
        let first = array(group.get(b"RunArray").unwrap()).unwrap()[0].clone();
        let mut second = first.clone();
        at_mut(&mut second, STYLE_PATH)
            .unwrap()
            .set(b"UnknownStyle", E::Integer(999))
            .unwrap();
        group
            .set(b"RunArray", E::Array(vec![first, second]))
            .unwrap();
        group
            .set(
                b"RunLengthArray",
                E::Array(vec![Node::new(E::Integer(2)), Node::new(E::Integer(3))]),
            )
            .unwrap();
        tool.text.set(
            Id::String(b"EngineData".to_vec()),
            D::Raw(engine.to_bytes(Limits::default()).unwrap()),
        );
        replace_tysh(&mut record, tool.bytes().unwrap());
        let old = editable(&record);
        let mut new = old.clone();
        new.insert(1, "😀", None);
        new.insert(new.text.len() - 1, "é", None);
        new.set_char_style_all(|style| style.tracking = 30.0);
        encode(&mut record, &old, &new).unwrap();
        let updated = TypeTool::parse(tysh(&record).data.as_slice())
            .unwrap()
            .engine_data()
            .unwrap();
        let editor = at(updated.root(), &[b"EngineDict", b"Editor", b"Text"])
            .unwrap()
            .as_text()
            .unwrap()
            .replace('\r', "\n");
        assert_eq!(editor, "A😀BCéD\n");
        let runs = ranges(updated.root(), b"StyleRun", &editor, new.text.len()).unwrap();
        let c = new.text.find('C').unwrap();
        let (_, _, source) = runs
            .iter()
            .find(|(start, end, _)| *start <= c && *end > c)
            .unwrap();
        assert_eq!(
            int(at(source, STYLE_PATH).unwrap(), b"UnknownStyle", 0).unwrap(),
            999
        );
        assert_eq!(new.imported.as_ref().unwrap().logical_text, new.text);
    }

    #[test]
    fn malformed_utf16_run_split_and_untracked_edit_are_rejected() {
        let mut record = fixture("😀");
        let mut tool = TypeTool::parse(tysh(&record).data.as_slice()).unwrap();
        let mut engine = tool.engine_data().unwrap();
        let group = at_mut(engine.root_mut(), &[b"EngineDict", b"StyleRun"]).unwrap();
        let node = array(group.get(b"RunArray").unwrap()).unwrap()[0].clone();
        group
            .set(b"RunArray", E::Array(vec![node.clone(), node]))
            .unwrap();
        group
            .set(
                b"RunLengthArray",
                E::Array(vec![Node::new(E::Integer(1)), Node::new(E::Integer(2))]),
            )
            .unwrap();
        tool.text.set(
            Id::String(b"EngineData".to_vec()),
            D::Raw(engine.to_bytes(Limits::default()).unwrap()),
        );
        replace_tysh(&mut record, tool.bytes().unwrap());
        assert!(
            decode(&record)
                .unwrap_err()
                .to_string()
                .contains("surrogate")
        );
        let mut record = fixture("A");
        let old = editable(&record);
        let mut new = old.clone();
        new.text = "B".into();
        let original = record.clone();
        assert!(
            encode(&mut record, &old, &new)
                .unwrap_err()
                .to_string()
                .contains("provenance")
        );
        assert_eq!(record, original);
    }

    #[test]
    fn unsupported_edit_is_atomic_and_stored_raster_bounds_are_cleared() {
        let mut record = fixture("A");
        let old = decode(&record).unwrap();
        let original = record.clone();
        let mut new = old.clone();
        new.set_char_style_all(|style| style.tracking += 1.0);
        assert!(
            encode(&mut record, &old, &new)
                .unwrap_err()
                .to_string()
                .contains("not installed")
        );
        assert_eq!(record, original);
        let rect = new.changed_rect(&old);
        let stored = old.imported.as_ref().unwrap().raster_bounds;
        assert_eq!(rect.union(&stored), rect);
        new.delete(0, new.text.len());
        assert!(new.imported.as_ref().unwrap().character_origins.is_empty());
        new.insert(0, "new\nparagraph", None);
        assert_eq!(new.imported.as_ref().unwrap().paragraph_origins.len(), 2);
    }

    #[test]
    fn global_text_cache_is_visible_and_prevents_inconsistent_edits() {
        let mut record = fixture("A");
        let mut old = editable(&record);
        mark_global_cache(&mut old);
        mark_global_cache(&mut old);
        assert_eq!(old.imported_text_limitations().len(), 1);
        assert!(old.imported_text_limitations()[0].contains("text edits cannot be saved"));
        let original = record.clone();
        encode(&mut record, &old, &old).unwrap();
        assert_eq!(record, original);
        let mut edited = old.clone();
        edited.insert(1, " changed", None);
        let error = encode(&mut record, &old, &edited).unwrap_err();
        assert!(error.to_string().contains("Photoshop text cache"));
        assert_eq!(record, original);
    }

    #[test]
    #[ignore = "Requires the explicitly supplied Photoshop-authored text corpus; never silently skipped"]
    fn authored_photoshop_text_files_decode_preserve_and_refuse_stale_global_cache() {
        use super::super::{PsdDocument, mapping};
        use crate::engine::{io::DocExtras, layers::LayerKind};
        let root = std::env::var_os("PIXELFORGE_PSD_TEXT_CORPUS")
            .expect("Set PIXELFORGE_PSD_TEXT_CORPUS to the Photoshop corpus originals directory");
        let root = std::path::PathBuf::from(root);
        let mut files: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("text-")
                    && path.extension().is_some_and(|ext| ext == "psd")
            })
            .collect();
        files.sort();
        assert!(
            files.len() >= 5,
            "The authoring corpus must include point, paragraph, runs, Unicode and vertical files"
        );
        let mut count = 0;
        for path in files {
            let document = PsdDocument::read(&path, &super::super::Limits::default()).unwrap();
            let mut text_count = 0;
            for record in &document.layer_mask.layers {
                if !record
                    .additional_info
                    .iter()
                    .any(|block| block.key == *b"TySh")
                {
                    continue;
                }
                let text =
                    decode(record).unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
                let mut roundtrip = record.clone();
                encode(&mut roundtrip, &text, &text).unwrap();
                assert_eq!(&roundtrip, record, "{}", path.display());
                text_count += 1;
            }
            assert!(
                text_count > 0,
                "{} is not a live text fixture",
                path.display()
            );
            count += text_count;
        }
        let path = root.join("text-point-default.psd");
        let tree = PsdDocument::read(&path, &super::super::Limits::default()).unwrap();
        assert!(
            tree.layer_mask
                .additional_info
                .iter()
                .any(|block| block.key == *b"Txt2")
        );
        let version = tree.header.version;
        let mut document = mapping::from_tree(tree).unwrap();
        let text = document
            .layers
            .iter_mut()
            .find_map(|layer| {
                if let LayerKind::Text(text) = &mut layer.kind {
                    Some(Arc::make_mut(text))
                } else {
                    None
                }
            })
            .unwrap();
        assert!(
            text.imported_text_limitations()
                .iter()
                .any(|message| message.contains("Photoshop text cache"))
        );
        text.insert(0, "Edited ", None);
        let extras = DocExtras {
            selected: document.selected.clone(),
            ..DocExtras::default()
        };
        let error = mapping::to_tree(
            &document.layers,
            document.w,
            document.h,
            document.active,
            &extras,
            document.psd.as_deref(),
            version,
        )
        .unwrap_err();
        assert!(error.to_string().contains("Txt2"), "{error:#}");
        eprintln!(
            "Verified {count} authored text layers decode and retain bytes; actual Txt2 text edit refusal verified separately"
        );
    }

    #[test]
    fn all_warp_styles_reencode_through_descriptor_fields() {
        for (_, style) in WARPS {
            let mut descriptor = Descriptor::new(Class::new("", Id::String(b"warp".to_vec())));
            let old = Warp::default();
            let warp = Warp {
                style: *style,
                bend: -0.2,
                h_distort: 0.3,
                v_distort: -0.4,
                vertical: true,
            };
            write_warp(&mut descriptor, old, warp);
            assert_eq!(read_warp(&descriptor, &mut Vec::new()).unwrap(), warp);
        }
    }
}
