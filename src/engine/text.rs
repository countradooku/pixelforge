//! Text layers: vector layers re-rasterized when edited.
//!
//! - **Model.** The text carries contiguous **style runs** (font, weight,
//!   italic, size, leading, tracking, colour, baseline shift, underline /
//!   strikethrough, faux bold / italic, all / small caps, super / subscript,
//!   OpenType features) covering every byte, and a **paragraph style** per
//!   line of source text (alignment with last-line justification, indents,
//!   space before / after, hyphenation). Edits keep both in step.
//! - **Shaping** uses `cosmic-text` (HarfRust, font fallback) one paragraph
//!   at a time as a single line of rich-text spans; each glyph carries its
//!   run through the span metadata. Caps and super / subscript are applied
//!   to the shaped string, with a byte map back to the source.
//! - **Line breaking** is PixelForge's own: greedy breaks at spaces and
//!   hyphens over per-line widths (box text, first-line indent, area type
//!   inside a closed path), optional hyphenation of overlong words, and
//!   justification (last line left / centre / right / all).
//! - **Glyph outlines** come from `swash` at the exact size and are
//!   rasterized with the exact-area rasterizer of `vector.rs`; coverage is
//!   cached per (font, glyph, size, weight, faux flags, ¼-pixel phase).
//! - **Rendering** touches only the tiles the text covers, in parallel, with
//!   each glyph composited in its run's colour; an edit re-renders only the
//!   tiles whose glyphs or decorations changed.
//! - **Vertical type** stacks upright CJK and rotates Latin 90° (unless
//!   Standard Vertical Roman Alignment keeps it upright); tate-chu-yoko
//!   ranges sit horizontally across the column. **Type on a path** places
//!   glyphs along a path (start / end / flip); **area type** flows lines
//!   inside a closed path. **Warp** uses Photoshop's 15 Bézier-patch styles
//!   (`warp.rs`) over the text's line box.
//!
//! Shaping and layout stay on the CPU (no GPU path): they are branchy,
//! per-glyph and sequential (line breaking depends on every earlier
//! glyph's advance), and work on a few kilobytes of text; only the
//! rasterized result reaches the compositor, which may be on the GPU.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cosmic_text::{Attrs, Buffer, Family, FeatureTag, FontFeatures, FontSystem, Metrics, Shaping, Style, Weight, Wrap};
use rayon::prelude::*;

use super::grid::{RectI, TS, TS_I, TileGrid, TileRef, finish_tile};
use super::px;
use super::vector::{self, FillRule, P, Poly, p};
pub use super::warp::{Warp, WarpStyle};

fn fonts() -> &'static Mutex<FontSystem> {
    static F: OnceLock<Mutex<FontSystem>> = OnceLock::new();
    F.get_or_init(|| Mutex::new(FontSystem::new()))
}

/// Loads the system font database in the background (it takes a moment).
pub fn warm_up() {
    std::thread::spawn(|| {
        fonts();
    });
}

/// Installed font families, sorted.
pub fn families() -> Vec<String> {
    static FAM: OnceLock<Vec<String>> = OnceLock::new();
    FAM.get_or_init(|| {
        let fs = fonts().lock().unwrap();
        let mut v: Vec<String> = fs.db().faces().filter_map(|f| f.families.first().map(|(n, _)| n.clone())).collect();
        v.sort();
        v.dedup();
        v
    })
    .clone()
}

/// Resolve Photoshop's PostScript font identifier without guessing a family
/// from its spelling (hyphens are valid family-name characters).
pub(crate) fn font_from_postscript(name: &str) -> Option<(String, u16, bool)> {
    let fs = fonts().lock().ok()?;
    let face = fs.db().faces().find(|face| face.post_script_name == name)?;
    Some((
        face.families.first()?.0.clone(),
        face.weight.0,
        face.style != Style::Normal,
    ))
}

/// The actual installed font identifier used when writing an edited PSD run.
pub(crate) fn postscript_from_font(family: &str, weight: u16, italic: bool) -> Option<String> {
    let fs = fonts().lock().ok()?;
    let id = fs.db().query(&cosmic_text::fontdb::Query {
        families: &[Family::Name(family)],
        weight: Weight(weight),
        stretch: cosmic_text::fontdb::Stretch::Normal,
        style: if italic { Style::Italic } else { Style::Normal },
    })?;
    Some(fs.db().face(id)?.post_script_name.clone())
}

// ----- styles --------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
    Justify,
}

impl TextAlign {
    pub const ALL: [TextAlign; 4] = [TextAlign::Left, TextAlign::Center, TextAlign::Right, TextAlign::Justify];
    pub fn label(self) -> &'static str {
        match self {
            TextAlign::Left => "Left",
            TextAlign::Center => "Center",
            TextAlign::Right => "Right",
            TextAlign::Justify => "Justify",
        }
    }
}

/// How the last line of a justified paragraph is set.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum LastLine {
    #[default]
    Left,
    Center,
    Right,
    /// Justify all lines, the last included.
    All,
}

impl LastLine {
    pub const ALL: [LastLine; 4] = [LastLine::Left, LastLine::Center, LastLine::Right, LastLine::All];
    pub fn label(self) -> &'static str {
        match self {
            LastLine::Left => "Last Left",
            LastLine::Center => "Last Center",
            LastLine::Right => "Last Right",
            LastLine::All => "Justify All",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Caps {
    #[default]
    Normal,
    All,
    Small,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Position {
    #[default]
    Normal,
    Super,
    Sub,
}

/// OpenType features, passed to the shaper per run.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct OtFeatures {
    /// Standard ligatures (`liga`, `clig`), on by default.
    pub liga: bool,
    /// Discretionary ligatures (`dlig`).
    pub dlig: bool,
    /// Old-style figures (`onum`).
    pub onum: bool,
    /// Swashes (`swsh`).
    pub swsh: bool,
    /// Fractions (`frac`).
    pub frac: bool,
    /// Stylistic sets `ss01`… `ss20` as bits 0… 19.
    pub ss: u32,
}

impl Default for OtFeatures {
    fn default() -> Self {
        OtFeatures { liga: true, dlig: false, onum: false, swsh: false, frac: false, ss: 0 }
    }
}

impl OtFeatures {
    fn to_cosmic(self) -> FontFeatures {
        let mut f = FontFeatures::new();
        if !self.liga {
            f.disable(FeatureTag::STANDARD_LIGATURES);
            f.disable(FeatureTag::CONTEXTUAL_LIGATURES);
        }
        for (on, tag) in [(self.dlig, b"dlig"), (self.onum, b"onum"), (self.swsh, b"swsh"), (self.frac, b"frac")] {
            if on {
                f.enable(FeatureTag::new(tag));
            }
        }
        for k in 0..20u32 {
            if self.ss & (1 << k) != 0 {
                let t = format!("ss{:02}", k + 1);
                let b = t.as_bytes();
                f.enable(FeatureTag::new(&[b[0], b[1], b[2], b[3]]));
            }
        }
        f
    }
}

/// Character attributes of one run.
#[derive(Clone, Debug, PartialEq)]
pub struct CharStyle {
    pub family: String,
    pub weight: u16,
    pub italic: bool,
    /// Font size in pixels.
    pub size: f32,
    /// Line height in pixels (0 = auto, 1.2 × size).
    pub leading: f32,
    /// Tracking in 1/1000 em (Photoshop's units).
    pub tracking: f32,
    pub color: [u8; 3],
    /// Baseline shift in pixels (up is positive).
    pub baseline_shift: f32,
    pub underline: bool,
    pub strike: bool,
    pub faux_bold: bool,
    pub faux_italic: bool,
    pub caps: Caps,
    pub position: Position,
    pub features: OtFeatures,
}

impl CharStyle {
    pub fn line_height(&self) -> f32 {
        if self.leading > 0.0 { self.leading } else { self.size * 1.2 }
    }

    /// The size the glyphs are shaped at (super / subscript are 58.3 %).
    fn shaped_size(&self) -> f32 {
        if self.position == Position::Normal { self.size } else { self.size * 0.583 }
    }
}

/// Line breaking: greedy line by line, or optimal over the whole paragraph
/// (Knuth–Plass; Photoshop's default "Every-line Composer").
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Composer {
    Single,
    #[default]
    Every,
}

impl Composer {
    pub const ALL: [Composer; 2] = [Composer::Single, Composer::Every];
    pub fn label(self) -> &'static str {
        match self {
            Composer::Single => "Single-line",
            Composer::Every => "Every-line",
        }
    }
}

/// Paragraph attributes.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ParaStyle {
    pub align: TextAlign,
    pub last: LastLine,
    pub indent_left: f32,
    pub indent_right: f32,
    /// First-line indent (may be negative: hanging).
    pub indent_first: f32,
    pub space_before: f32,
    pub space_after: f32,
    pub hyphenate: bool,
    pub composer: Composer,
}

/// The layer's default character style plus default alignment (what the
/// Type tool's options bar edits when nothing is selected).
#[derive(Clone, Debug, PartialEq)]
pub struct TextStyle {
    pub family: String,
    pub weight: u16,
    pub italic: bool,
    pub size: f32,
    pub leading: f32,
    pub tracking: f32,
    pub color: [u8; 3],
    pub align: TextAlign,
}

impl Default for TextStyle {
    fn default() -> Self {
        TextStyle { family: "Helvetica".into(), weight: 400, italic: false, size: 72.0, leading: 0.0, tracking: 0.0, color: [0, 0, 0], align: TextAlign::Left }
    }
}

impl TextStyle {
    pub fn line_height(&self) -> f32 {
        if self.leading > 0.0 { self.leading } else { self.size * 1.2 }
    }

    pub fn char_style(&self) -> CharStyle {
        CharStyle {
            family: self.family.clone(),
            weight: self.weight,
            italic: self.italic,
            size: self.size,
            leading: self.leading,
            tracking: self.tracking,
            color: self.color,
            baseline_shift: 0.0,
            underline: false,
            strike: false,
            faux_bold: false,
            faux_italic: false,
            caps: Caps::Normal,
            position: Position::Normal,
            features: OtFeatures::default(),
        }
    }

    pub fn para_style(&self) -> ParaStyle {
        ParaStyle { align: self.align, ..Default::default() }
    }
}

/// A run of characters `start..end` (bytes) sharing a style.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleRun {
    pub start: usize,
    pub end: usize,
    pub style: CharStyle,
}

/// Type on a path (or inside a closed path: area type).
#[derive(Clone, Debug, PartialEq)]
pub struct TextPath {
    pub path: vector::Path,
    /// Start and end of the text along the path, as fractions of its
    /// length; on a closed path `end` may pass 1 (wrapping, up to start + 1).
    pub start: f32,
    pub end: f32,
    /// Text runs the other way, on the other side of the path.
    pub flip: bool,
    /// Area type: lines flow inside the (closed) path instead of along it.
    pub area: bool,
}

// ----- layout output ------------------------------------------------------------------------

/// Flags of a placed glyph: faux bold / italic, and generated hyphens.
pub const SYNTH_BOLD: u8 = 1;
pub const SYNTH_ITALIC: u8 = 2;
pub const SYNTH_HYPHEN: u8 = 4;

/// One glyph placed in layout coordinates (pen position on the baseline;
/// the warp, if any, maps layout to document coordinates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedGlyph {
    pub font_id: cosmic_text::fontdb::ID,
    pub glyph: u16,
    pub size: f32,
    pub weight: u16,
    pub x: f32,
    pub y: f32,
    pub advance: f32,
    /// Byte range of the source text.
    pub start: usize,
    pub end: usize,
    pub color: [u8; 3],
    pub synth: u8,
    /// Rotation about the pen position (radians, clockwise in image space).
    pub rot: f32,
    /// The glyph's cell: leading edge `cell[0]`→`cell[1]`, trailing edge
    /// `cell[3]`→`cell[2]` (carets and selection highlights).
    pub cell: [P; 4],
}

impl PlacedGlyph {
    fn soft(&self) -> bool {
        self.synth & SYNTH_HYPHEN != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineInfo {
    pub top: f32,
    pub height: f32,
    pub baseline: f32,
    pub start: usize,
    pub end: usize,
    pub x0: f32,
    pub x1: f32,
    /// Glyph index range in `TextLayout::glyphs`.
    pub g0: usize,
    pub g1: usize,
}

/// An underline or strikethrough: a quad in layout coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decoration {
    pub quad: [P; 4],
    pub color: [u8; 3],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextLayout {
    pub glyphs: Vec<PlacedGlyph>,
    pub lines: Vec<LineInfo>,
    pub decos: Vec<Decoration>,
    /// The line box (what Warp Text bends): x0, y0, x1, y1.
    pub line_box: (f32, f32, f32, f32),
    /// Paragraph or area text that does not fit (Photoshop's ⊞ indicator).
    pub overflow: bool,
}

#[derive(Clone, Debug)]
pub struct TextLayer {
    pub text: String,
    /// Defaults: the typing style for an empty layer, and the options bar.
    pub style: TextStyle,
    /// Contiguous runs covering `0..text.len()`.
    pub runs: Vec<StyleRun>,
    /// One paragraph style per `\n`-separated paragraph.
    pub paras: Vec<ParaStyle>,
    /// Point text: start of the first baseline (aligned per paragraph).
    /// Paragraph text: top-left of the text box.
    pub origin: (f32, f32),
    pub box_size: Option<(f32, f32)>,
    pub vertical: bool,
    /// Vertical type: keep Latin upright (Standard Vertical Roman Alignment).
    pub upright_roman: bool,
    /// Vertical type: byte ranges set horizontally across the column.
    pub tcy: Vec<(usize, usize)>,
    pub warp: Warp,
    pub path: Option<Arc<TextPath>>,
    pub layout: Arc<TextLayout>,
    /// Source raster bounds and explicit limitations of editing imported type.
    pub(crate) imported: Option<Arc<ImportedText>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ImportedText {
    pub(crate) raster_bounds: RectI,
    pub(crate) limitations: Vec<String>,
    /// The original EngineData run to inherit for each current text span.
    /// Keeping this separate from CharStyle prevents normalization from
    /// merging away unknown Photoshop attributes that look identical here.
    pub(crate) character_origins: Vec<ImportedRun>,
    pub(crate) paragraph_origins: Vec<usize>,
    pub(crate) empty_character_origin: usize,
    pub(crate) logical_text: String,
    pub(crate) identity: Arc<()>,
}

#[derive(Clone, Debug)]
pub(crate) struct ImportedRun {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) source: usize,
}

impl ImportedText {
    fn insert(&mut self, at: usize, text: &str, paragraph: usize) {
        let inherited = self.character_origins.iter().find(|run| run.start <= at && run.end >= at)
            .map_or(self.empty_character_origin, |run| run.source);
        let mut ranges = Vec::with_capacity(self.character_origins.len() + 2);
        for run in self.character_origins.drain(..) {
            if run.end <= at {
                ranges.push(run);
            } else if run.start >= at {
                ranges.push(ImportedRun { start: run.start + text.len(), end: run.end + text.len(), source: run.source });
            } else {
                ranges.push(ImportedRun { start: run.start, end: at, source: run.source });
                ranges.push(ImportedRun { start: at + text.len(), end: run.end + text.len(), source: run.source });
            }
        }
        ranges.push(ImportedRun { start: at, end: at + text.len(), source: inherited });
        ranges.sort_by_key(|run| run.start);
        self.character_origins = ranges;
        let inherited_paragraph = self.paragraph_origins.get(paragraph).copied().unwrap_or(0);
        for _ in 0..text.matches('\n').count() { self.paragraph_origins.insert(paragraph + 1, inherited_paragraph); }
        self.logical_text.insert_str(at, text);
    }

    fn delete(&mut self, start: usize, end: usize, paragraph: usize, gone: usize) {
        self.empty_character_origin = self.character_origins.iter().find(|run| run.start <= start && run.end >= start)
            .map_or(self.empty_character_origin, |run| run.source);
        let map = |byte| if byte <= start { byte } else if byte >= end { byte - (end - start) } else { start };
        for run in &mut self.character_origins { run.start = map(run.start); run.end = map(run.end); }
        self.character_origins.retain(|run| run.end > run.start);
        let last = (paragraph + 1 + gone).min(self.paragraph_origins.len());
        self.paragraph_origins.drain(paragraph + 1..last);
        self.logical_text.replace_range(start..end, "");
    }
}

impl PartialEq for TextLayer {
    fn eq(&self, o: &Self) -> bool {
        self.text == o.text
            && self.style == o.style
            && self.runs == o.runs
            && self.paras == o.paras
            && self.origin == o.origin
            && self.box_size == o.box_size
            && self.vertical == o.vertical
            && self.upright_roman == o.upright_roman
            && self.tcy == o.tcy
            && self.warp == o.warp
            && self.path == o.path
    }
}

// ----- editing --------------------------------------------------------------------------------

impl TextLayer {
    pub fn new(text: impl Into<String>, style: TextStyle, origin: (f32, f32), box_size: Option<(f32, f32)>, vertical: bool) -> Self {
        let text = text.into();
        let runs = if text.is_empty() { vec![] } else { vec![StyleRun { start: 0, end: text.len(), style: style.char_style() }] };
        let paras = vec![style.para_style(); text.matches('\n').count() + 1];
        let mut t = TextLayer {
            text,
            style,
            runs,
            paras,
            origin,
            box_size,
            vertical,
            upright_roman: false,
            tcy: vec![],
            warp: Warp::default(),
            path: None,
            layout: Arc::new(TextLayout::default()),
            imported: None,
        };
        t.relayout();
        t
    }

    /// Features retained in the source PSD that cannot yet be edited safely.
    /// Opening and saving an unchanged text layer retains their original bytes.
    pub fn imported_text_limitations(&self) -> &[String] {
        self.imported
            .as_ref()
            .map_or(&[], |source| source.limitations.as_slice())
    }

    /// The style of the character at byte `i` (the typing style past the end).
    pub fn style_at(&self, i: usize) -> CharStyle {
        let k = self.runs.partition_point(|r| r.end <= i);
        self.runs.get(k).or(self.runs.last()).map_or_else(|| self.style.char_style(), |r| r.style.clone())
    }

    /// Index of the paragraph containing byte `i`.
    pub fn para_index(&self, i: usize) -> usize {
        self.text[..i.min(self.text.len())].matches('\n').count()
    }

    /// Merges equal neighbours and drops empty runs.
    fn normalize(&mut self) {
        self.runs.retain(|r| r.end > r.start);
        let mut out: Vec<StyleRun> = Vec::with_capacity(self.runs.len());
        for r in self.runs.drain(..) {
            match out.last_mut() {
                Some(l) if l.style == r.style && l.end == r.start => l.end = r.end,
                _ => out.push(r),
            }
        }
        self.runs = out;
        let n = self.text.matches('\n').count() + 1;
        let fill = self.paras.last().copied().unwrap_or_else(|| self.style.para_style());
        self.paras.resize(n, fill);
    }

    /// Inserts `s` at byte `at`, in `style` (or the style of the character
    /// before the caret, as Photoshop does).
    pub fn insert(&mut self, at: usize, s: &str, style: Option<CharStyle>) {
        if s.is_empty() {
            return;
        }
        let n = s.len();
        let st = style.unwrap_or_else(|| self.style_at(at.saturating_sub(1)));
        // Paragraph styles: a new paragraph inherits the one it splits.
        let pi = self.para_index(at);
        if let Some(source) = &mut self.imported { Arc::make_mut(source).insert(at, s, pi); }
        let new_paras = s.matches('\n').count();
        if new_paras > 0 {
            let ps = self.paras.get(pi).copied().unwrap_or_else(|| self.style.para_style());
            for _ in 0..new_paras {
                self.paras.insert(pi + 1, ps);
            }
        }
        self.text.insert_str(at, s);
        let mut out = Vec::with_capacity(self.runs.len() + 2);
        let mut placed = false;
        for r in self.runs.drain(..) {
            if r.end <= at {
                if r.end == at && !placed && r.style == st {
                    out.push(StyleRun { end: r.end + n, ..r });
                    placed = true;
                } else {
                    out.push(r);
                }
            } else if r.start >= at {
                if !placed {
                    out.push(StyleRun { start: at, end: at + n, style: st.clone() });
                    placed = true;
                }
                out.push(StyleRun { start: r.start + n, end: r.end + n, style: r.style });
            } else {
                // `at` falls inside this run.
                if r.style == st {
                    out.push(StyleRun { end: r.end + n, ..r });
                } else {
                    out.push(StyleRun { start: r.start, end: at, style: r.style.clone() });
                    out.push(StyleRun { start: at, end: at + n, style: st.clone() });
                    out.push(StyleRun { start: at + n, end: r.end + n, style: r.style });
                }
                placed = true;
            }
        }
        if !placed {
            out.push(StyleRun { start: at, end: at + n, style: st });
        }
        self.runs = out;
        self.normalize();
    }

    /// Deletes the bytes `a..b`.
    pub fn delete(&mut self, a: usize, b: usize) {
        let (a, b) = (a.min(b), a.max(b).min(self.text.len()));
        if a >= b {
            return;
        }
        let gone = self.text[a..b].matches('\n').count();
        let pi = self.para_index(a);
        if let Some(source) = &mut self.imported { Arc::make_mut(source).delete(a, b, pi, gone); }
        if gone > 0 {
            let end = (pi + 1 + gone).min(self.paras.len());
            self.paras.drain(pi + 1..end);
        }
        self.text.replace_range(a..b, "");
        let n = b - a;
        let map = |x: usize| if x <= a { x } else if x >= b { x - n } else { a };
        for r in &mut self.runs {
            r.start = map(r.start);
            r.end = map(r.end);
        }
        self.normalize();
    }

    /// Applies `f` to the character style of bytes `a..b` (splitting runs).
    pub fn set_char_style(&mut self, a: usize, b: usize, f: impl Fn(&mut CharStyle)) {
        let (a, b) = (a.min(b), a.max(b).min(self.text.len()));
        if a >= b {
            return;
        }
        let mut out = Vec::with_capacity(self.runs.len() + 2);
        for r in self.runs.drain(..) {
            if r.end <= a || r.start >= b {
                out.push(r);
                continue;
            }
            if r.start < a {
                out.push(StyleRun { start: r.start, end: a, style: r.style.clone() });
            }
            let mut st = r.style.clone();
            f(&mut st);
            out.push(StyleRun { start: r.start.max(a), end: r.end.min(b), style: st });
            if r.end > b {
                out.push(StyleRun { start: b, end: r.end, style: r.style });
            }
        }
        self.runs = out;
        self.normalize();
    }

    /// Applies `f` to every character and to the typing style.
    pub fn set_char_style_all(&mut self, f: impl Fn(&mut CharStyle)) {
        let n = self.text.len();
        self.set_char_style(0, n, &f);
        let mut cs = self.style.char_style();
        f(&mut cs);
        self.style.family = cs.family;
        self.style.weight = cs.weight;
        self.style.italic = cs.italic;
        self.style.size = cs.size;
        self.style.leading = cs.leading;
        self.style.tracking = cs.tracking;
        self.style.color = cs.color;
    }

    /// Applies `f` to the paragraphs touched by bytes `a..b`.
    pub fn set_para_style(&mut self, a: usize, b: usize, f: impl Fn(&mut ParaStyle)) {
        let (p0, p1) = (self.para_index(a.min(b)), self.para_index(a.max(b)));
        for ps in self.paras.iter_mut().take(p1 + 1).skip(p0) {
            f(ps);
        }
        if p0 == 0 {
            let mut d = self.style.para_style();
            f(&mut d);
            self.style.align = d.align;
        }
    }

    /// Byte range of the word around byte `i` (double-click).
    pub fn word_at(&self, i: usize) -> (usize, usize) {
        let i = i.min(self.text.len());
        let is_word = |c: char| c.is_alphanumeric() || c == '\'' || c == '_';
        let a = self.text[..i].char_indices().rev().take_while(|&(_, c)| is_word(c)).last().map_or(i, |(k, _)| k);
        let b = self.text[i..].char_indices().find(|&(_, c)| !is_word(c)).map_or(self.text.len(), |(k, _)| i + k);
        (a, b)
    }

    /// Byte range of the laid-out line containing byte `i` (triple-click).
    pub fn line_at(&self, i: usize) -> (usize, usize) {
        for l in &self.layout.lines {
            if i >= l.start && i <= l.end {
                return (l.start, l.end);
            }
        }
        (0, self.text.len())
    }
}

// ----- shaping --------------------------------------------------------------------------------

/// Per-em font metrics (y down: `asc` above, `desc` below the baseline).
#[derive(Clone, Copy, Debug)]
struct FontMetrics {
    asc: f32,
    desc: f32,
    /// Underline centre below the baseline, and thickness.
    ul: f32,
    thick: f32,
    /// Strikethrough centre above the baseline.
    strike: f32,
}

/// Cached; takes the font-system lock only on a miss (callers must not hold it).
fn font_metrics(id: cosmic_text::fontdb::ID, weight: u16) -> FontMetrics {
    static C: OnceLock<Mutex<HashMap<(cosmic_text::fontdb::ID, u16), FontMetrics>>> = OnceLock::new();
    let c = C.get_or_init(Default::default);
    if let Some(m) = c.lock().unwrap().get(&(id, weight)) {
        return *m;
    }
    let m = fonts()
        .lock()
        .unwrap()
        .get_font(id, Weight(weight))
        .map(|f| {
            let m = f.as_swash().metrics(&[]);
            let u = m.units_per_em.max(1) as f32;
            let thick = if m.stroke_size > 0.0 { m.stroke_size / u } else { 0.06 };
            FontMetrics {
                asc: m.ascent / u,
                desc: m.descent.abs() / u,
                ul: if m.underline_offset != 0.0 { -m.underline_offset / u + thick * 0.5 } else { 0.12 },
                thick,
                strike: if m.strikeout_offset > 0.0 { m.strikeout_offset / u } else { 0.28 },
            }
        })
        .unwrap_or(FontMetrics { asc: 0.8, desc: 0.2, ul: 0.12, thick: 0.06, strike: 0.28 });
    c.lock().unwrap().insert((id, weight), m);
    m
}

/// A shaped glyph of one paragraph; byte offsets and the run index are
/// relative to the paragraph (so unchanged paragraphs hit the cache).
#[derive(Clone, Copy, Debug)]
struct SGlyph {
    font_id: cosmic_text::fontdb::ID,
    glyph: u16,
    size: f32,
    weight: u16,
    adv: f32,
    xo: f32,
    yo: f32,
    start: usize,
    end: usize,
    run: usize,
}

struct Shaped {
    glyphs: Vec<SGlyph>,
    /// A hyphen per run (relative index), when the paragraph hyphenates.
    hyphens: HashMap<usize, SGlyph>,
}

fn hash_style(h: &mut impl std::hash::Hasher, s: &CharStyle) {
    use std::hash::Hash;
    s.family.hash(h);
    s.weight.hash(h);
    s.italic.hash(h);
    for v in [s.size, s.leading, s.tracking] {
        v.to_bits().hash(h);
    }
    s.caps.hash(h);
    s.position.hash(h);
    s.features.hash(h);
}

fn attrs_for(st: &CharStyle, small: bool, meta: usize, vertical: bool) -> Attrs<'_> {
    let size = st.shaped_size() * if small { 0.7 } else { 1.0 };
    Attrs::new()
        .family(Family::Name(&st.family))
        .weight(Weight(st.weight))
        .style(if st.italic { Style::Italic } else { Style::Normal })
        .metrics(Metrics::new(size, st.line_height()))
        .letter_spacing(st.tracking / 1000.0)
        .font_features({
            let mut f = st.features.to_cosmic();
            if vertical {
                // Vertical alternates (punctuation, brackets, ー) for upright CJK.
                f.enable(FeatureTag::new(b"vert"));
            }
            f
        })
        .metadata(meta)
}

/// Shapes `text` (one paragraph) as a single line of styled spans.
/// `runs` are the paragraph's runs, offsets relative to `text`.
fn shape_para(text: &str, runs: &[StyleRun], hyphenate: bool, vertical: bool) -> Arc<Shaped> {
    use std::hash::{Hash, Hasher};
    static CACHE: OnceLock<Mutex<HashMap<u64, Arc<Shaped>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    hyphenate.hash(&mut h);
    vertical.hash(&mut h);
    for r in runs {
        (r.start, r.end).hash(&mut h);
        hash_style(&mut h, &r.style);
    }
    let key = h.finish();
    if let Some(s) = cache.lock().unwrap().get(&key) {
        return s.clone();
    }
    // Display string (caps applied) with a byte map back to the source, and
    // spans of (display range, run, small caps).
    let mut disp = String::with_capacity(text.len());
    let mut map: Vec<usize> = Vec::with_capacity(text.len() + 1);
    let mut spans: Vec<(usize, usize, usize, bool)> = Vec::new();
    for (k, r) in runs.iter().enumerate() {
        for (i, c) in text[r.start..r.end].char_indices() {
            let src = r.start + i;
            let (small, s): (bool, String) = match r.style.caps {
                Caps::Normal => (false, c.to_string()),
                Caps::All => (false, c.to_uppercase().collect()),
                Caps::Small if c.is_lowercase() => (true, c.to_uppercase().collect()),
                Caps::Small => (false, c.to_string()),
            };
            let d0 = disp.len();
            disp.push_str(&s);
            map.extend(std::iter::repeat_n(src, s.len()));
            match spans.last_mut() {
                Some(l) if l.2 == k && l.3 == small => l.1 = disp.len(),
                _ => spans.push((d0, disp.len(), k, small)),
            }
        }
    }
    map.push(text.len());
    let mut fs = fonts().lock().unwrap();
    let mut out = Shaped { glyphs: Vec::with_capacity(disp.len()), hyphens: HashMap::new() };
    let to_sg = |g: &cosmic_text::LayoutGlyph, start: usize, end: usize, run: usize| SGlyph {
        font_id: g.font_id,
        glyph: g.glyph_id,
        size: g.font_size,
        weight: g.font_weight.0,
        adv: g.w,
        xo: g.font_size * g.x_offset,
        yo: -g.font_size * g.y_offset,
        start,
        end,
        run,
    };
    if !spans.is_empty() {
        let st0 = &runs[spans[0].2].style;
        let mut buf = Buffer::new(&mut fs, Metrics::new(st0.shaped_size(), st0.line_height()));
        buf.set_size(None, None);
        buf.set_wrap(Wrap::None);
        let items: Vec<(&str, Attrs)> = spans.iter().enumerate().map(|(si, &(a, b, k, small))| (&disp[a..b], attrs_for(&runs[k].style, small, si, vertical))).collect();
        buf.set_rich_text(items, &attrs_for(st0, false, 0, vertical), Shaping::Advanced, None);
        buf.shape_until_scroll(&mut fs, false);
        for lr in buf.layout_runs() {
            for g in lr.glyphs {
                let k = spans[g.metadata.min(spans.len() - 1)].2;
                out.glyphs.push(to_sg(g, map[g.start.min(map.len() - 1)], map[g.end.min(map.len() - 1)], k));
            }
        }
    }
    if hyphenate && !runs.is_empty() {
        let st0 = &runs[0].style;
        let mut buf = Buffer::new(&mut fs, Metrics::new(st0.shaped_size(), st0.line_height()));
        buf.set_size(None, None);
        buf.set_wrap(Wrap::None);
        let hy = "-".repeat(runs.len());
        let items: Vec<(&str, Attrs)> = (0..runs.len()).map(|k| (&hy[k..k + 1], attrs_for(&runs[k].style, false, k, vertical))).collect();
        buf.set_rich_text(items, &attrs_for(st0, false, 0, vertical), Shaping::Advanced, None);
        buf.shape_until_scroll(&mut fs, false);
        for lr in buf.layout_runs() {
            for g in lr.glyphs {
                out.hyphens.insert(g.metadata, to_sg(g, 0, 0, g.metadata));
            }
        }
    }
    let s = Arc::new(out);
    let mut c = cache.lock().unwrap();
    if c.len() > 512 {
        c.clear();
    }
    c.insert(key, s.clone());
    s
}

fn is_vowel(c: char) -> bool {
    "aeiouyàáâäèéêëìíîïòóôöùúûü".contains(c)
}

/// Consonant pairs that stay together when hyphenating.
const DIGRAPHS: [&str; 10] = ["ch", "sh", "th", "ph", "wh", "ck", "ng", "qu", "gh", "gn"];
/// Consonant clusters that may begin a syllable.
const ONSETS: [&str; 24] = ["bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sc", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr", "tw", "th", "ch", "ph"];

/// Syllable-heuristic break points (before letter index) for words the
/// patterns don't cover: VC|CV (not inside a digraph), V|CV, V|onset-V.
fn heuristic_breaks(word: &str) -> Vec<usize> {
    let ch: Vec<char> = word.chars().flat_map(|c| c.to_lowercase()).collect();
    let n = ch.len();
    let at = |i: usize| ch.get(i).copied().unwrap_or(' ');
    if n < 6 {
        return vec![];
    }
    (2..=n - 3)
        .filter(|&k| {
            let (a, b) = (at(k - 1), at(k));
            let pair: String = [a, b].iter().collect();
            match (is_vowel(a), is_vowel(b)) {
                (false, false) => is_vowel(at(k.wrapping_sub(2))) && is_vowel(at(k + 1)) && !DIGRAPHS.contains(&pair.as_str()),
                (true, false) => is_vowel(at(k + 1)) || (ONSETS.contains(&[b, at(k + 1)].iter().collect::<String>().as_str()) && is_vowel(at(k + 2))),
                _ => false,
            }
        })
        .collect()
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x1100..=0x11FF | 0x2E80..=0x2FDF | 0x3000..=0x30FF | 0x3100..=0x31FF | 0x3200..=0x9FFF
        | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0xFE30..=0xFE4F | 0xFF00..=0xFFEF | 0x20000..=0x3FFFF)
}

/// Greedy line breaking over unit advances. `avail(line)` gives the width
/// of each line; whitespace never overflows (it hangs past the edge).
/// A negative `avail` leaves that line empty. `hyphenate(s, e, avail)` may return a break inside the overflowing word
/// (then the line ends with a hyphen). Returns `(start, end, hyphen)`.
fn greedy_breaks(
    adv: &[f32],
    space: &[bool],
    brk_after: &[bool],
    mut avail: impl FnMut(usize) -> f32,
    mut hyphenate: impl FnMut(usize, usize, usize, f32) -> Option<usize>,
) -> Vec<(usize, usize, bool)> {
    let n = adv.len();
    let mut out = Vec::new();
    let mut s = 0;
    while s < n {
        let a = avail(out.len());
        // A negative width marks a line to leave empty (area type).
        if a < 0.0 && out.len() < 100_000 {
            out.push((s, s, false));
            continue;
        }
        let mut w = 0.0;
        let mut last_brk = None;
        let mut e = s;
        while e < n {
            if !space[e] && e > s && w + adv[e] > a + 0.01 {
                break;
            }
            w += adv[e];
            if brk_after[e] {
                last_brk = Some(e + 1);
            }
            e += 1;
        }
        if e >= n {
            out.push((s, n, false));
            break;
        }
        let word0 = last_brk.unwrap_or(s);
        if let Some(k) = hyphenate(s, word0, e, a) {
            out.push((s, k, true));
            s = k;
            continue;
        }
        let end = match last_brk {
            Some(b) if b > s => b,
            _ => e.max(s + 1),
        };
        out.push((s, end, false));
        s = end;
    }
    out
}

/// Knuth–Plass optimal line breaking (Photoshop's "Every-line Composer").
/// Boxes are glyph advances, glue the spaces (word spacing 80–133 %, as
/// Photoshop's justification defaults), penalties the hyphen points.
/// `widths[l]` is line `l`'s measure. Ragged alignments use a rag cost
/// instead of glue stretch. Returns `None` when no set of breaks fits.
fn kp_breaks(adv: &[f32], space: &[bool], brk: &[bool], hy: &[Option<f32>], widths: &[f32], justify: bool, last_all: bool) -> Option<Vec<(usize, usize, bool)>> {
    struct Node {
        pos: usize,
        line: usize,
        dem: f64,
        prev: Option<usize>,
        hyphen: bool,
        fit: u8,
    }
    let n = adv.len();
    let mut pre = vec![0f64; n + 1];
    let mut sp_w = vec![0f64; n + 1];
    for i in 0..n {
        pre[i + 1] = pre[i] + adv[i] as f64;
        sp_w[i + 1] = sp_w[i] + if space[i] { adv[i] as f64 } else { 0.0 };
    }
    let width = |l: usize| widths[l.min(widths.len() - 1)] as f64;
    for tol in [1.0f64, 4.0] {
        let mut nodes = vec![Node { pos: 0, line: 0, dem: 0.0, prev: None, hyphen: false, fit: 1 }];
        let mut active = vec![0usize];
        for b in 1..=n {
            let is_space_brk = brk[b - 1];
            let hyw = if b < n && !is_space_brk { hy[b] } else { None };
            if b < n && !is_space_brk && hyw.is_none() {
                continue;
            }
            let last = b == n;
            // Best new node per line number.
            let mut best: Vec<(usize, f64, usize, u8)> = Vec::new();
            let mut keep = Vec::with_capacity(active.len());
            for &a in &active {
                let (s, line) = (nodes[a].pos, nodes[a].line);
                let w = width(line);
                let mut e = b;
                while e > s && space[e - 1] {
                    e -= 1;
                }
                let nat = pre[e] - pre[s] + hyw.map_or(0.0, |h| h as f64);
                let glue = sp_w[e] - sp_w[s];
                let (stretch, shrink) = if justify { (glue * 0.33, glue * 0.2) } else { (w * 0.4, 0.0) };
                if nat - shrink > w + 0.01 {
                    // Too long now and for every later break: drop the node.
                    continue;
                }
                keep.push(a);
                let r = if nat > w {
                    (w - nat) / shrink.max(1e-9)
                } else if last && !(justify && last_all) {
                    0.0
                } else if stretch > 0.0 {
                    (w - nat) / stretch
                } else {
                    f64::INFINITY
                };
                if r > tol {
                    continue;
                }
                let bad = (100.0 * r.abs().powi(3)).min(10_000.0);
                let pen = if hyw.is_some() { 50.0 } else { 0.0 };
                let mut d = (1.0 + bad + pen).powi(2);
                if hyw.is_some() && nodes[a].hyphen {
                    d += 3000.0;
                }
                let fit = if r < -0.5 { 0 } else if r <= 0.5 { 1 } else if r <= 1.0 { 2 } else { 3 };
                if (fit as i32 - nodes[a].fit as i32).abs() > 1 {
                    d += 100.0;
                }
                let total = nodes[a].dem + d;
                match best.iter_mut().find(|x| x.0 == line + 1) {
                    Some(x) if x.1 <= total => {}
                    Some(x) => *x = (line + 1, total, a, fit),
                    None => best.push((line + 1, total, a, fit)),
                }
            }
            active = keep;
            for (line, dem, from, fit) in best {
                nodes.push(Node { pos: b, line, dem, prev: Some(from), hyphen: hyw.is_some(), fit });
                active.push(nodes.len() - 1);
            }
            if active.is_empty() {
                break;
            }
        }
        let Some(end) = (0..nodes.len()).filter(|&k| nodes[k].pos == n && n > 0).min_by(|&x, &y| nodes[x].dem.total_cmp(&nodes[y].dem)) else { continue };
        let mut out = Vec::new();
        let mut k = end;
        while let Some(p) = nodes[k].prev {
            out.push((nodes[p].pos, nodes[k].pos, nodes[k].hyphen));
            k = p;
        }
        out.reverse();
        return Some(out);
    }
    None
}

// ----- layout ---------------------------------------------------------------------------------

/// The rect `x0..x1 × y0..y1` around `pen`, rotated by `rot`: leading edge
/// first (as `PlacedGlyph::cell`).
fn local_quad(pen: P, rot: f32, x0: f32, x1: f32, y0: f32, y1: f32) -> [P; 4] {
    let (s, c) = rot.sin_cos();
    let t = |x: f32, y: f32| p(pen.x + x * c - y * s, pen.y + x * s + y * c);
    [t(x0, y0), t(x0, y1), t(x1, y1), t(x1, y0)]
}

/// Horizontal extent of a closed polygon set along the band `y0..y1`:
/// the widest interval at the band's middle, narrowed by its edges.
fn band_interval(polys: &[Poly], y0: f32, y1: f32) -> Option<(f32, f32)> {
    let cross = |y: f32| {
        let mut xs = Vec::new();
        for pl in polys {
            let n = pl.pts.len();
            for i in 0..n {
                let (a, b) = (pl.pts[i], pl.pts[(i + 1) % n]);
                if (a.y <= y) != (b.y <= y) {
                    xs.push(a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x));
                }
            }
        }
        xs.sort_by(f32::total_cmp);
        xs.chunks_exact(2).map(|c| (c[0], c[1])).collect::<Vec<_>>()
    };
    let mid = cross((y0 + y1) * 0.5);
    let (mut a, mut b) = mid.into_iter().max_by(|x, y| (x.1 - x.0).total_cmp(&(y.1 - y.0)))?;
    for y in [y0 + (y1 - y0) * 0.1, y1 - (y1 - y0) * 0.1] {
        let best = cross(y).into_iter().filter(|&(c, d)| d > a && c < b).max_by(|x, y| (x.1.min(b) - x.0.max(a)).total_cmp(&(y.1.min(b) - y.0.max(a))));
        let (c, d) = best?;
        a = a.max(c);
        b = b.min(d);
    }
    (b > a).then_some((a, b))
}

/// What a laid-out paragraph needs from its shaped glyphs.
struct ParaCtx {
    runs: Vec<StyleRun>,
    shaped: Arc<Shaped>,
    chars: Vec<char>,
    pre: Vec<f32>,
}

impl ParaCtx {
    fn adv_range(&self, s: usize, e: usize) -> f32 {
        self.pre[e] - self.pre[s]
    }
}

impl TextLayer {
    pub fn relayout(&mut self) {
        let mut l = if self.vertical {
            self.layout_vertical()
        } else if let Some(tp) = self.path.clone().filter(|t| !t.area) {
            self.layout_on_path(&tp)
        } else {
            self.layout_horizontal()
        };
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for li in &l.lines {
            x0 = x0.min(li.x0);
            x1 = x1.max(li.x1);
            y0 = y0.min(li.top);
            y1 = y1.max(li.top + li.height);
        }
        l.line_box = if x0 > x1 { (self.origin.0, self.origin.1, self.origin.0 + 1.0, self.origin.1 + 1.0) } else { (x0, y0, x1.max(x0 + 1.0), y1.max(y0 + 1.0)) };
        self.layout = Arc::new(l);
    }

    /// Paragraph byte ranges (without the `\n`).
    fn para_ranges(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(self.paras.len());
        let mut a = 0;
        for (i, c) in self.text.char_indices() {
            if c == '\n' {
                out.push((a, i));
                a = i + 1;
            }
        }
        out.push((a, self.text.len()));
        out
    }

    fn para_ctx(&self, a: usize, b: usize, hyphenate: bool) -> ParaCtx {
        let k0 = self.runs.partition_point(|r| r.end <= a);
        let mut runs: Vec<StyleRun> =
            self.runs[k0..].iter().take_while(|r| r.start < b).map(|r| StyleRun { start: r.start.max(a) - a, end: r.end.min(b) - a, style: r.style.clone() }).collect();
        if runs.is_empty() {
            // An empty paragraph keeps the style of the text around it.
            runs.push(StyleRun { start: 0, end: 0, style: self.style_at(a.saturating_sub(1)) });
        }
        let shaped = shape_para(&self.text[a..b], if a == b { &[] } else { &runs }, hyphenate, self.vertical);
        let chars = shaped.glyphs.iter().map(|g| self.text[a + g.start..].chars().next().unwrap_or(' ')).collect();
        let mut pre = Vec::with_capacity(shaped.glyphs.len() + 1);
        pre.push(0.0);
        for g in &shaped.glyphs {
            pre.push(pre.last().unwrap() + g.adv);
        }
        ParaCtx { runs, shaped, chars, pre }
    }

    fn synth_of(st: &CharStyle) -> u8 {
        (if st.faux_bold { SYNTH_BOLD } else { 0 }) | (if st.faux_italic { SYNTH_ITALIC } else { 0 })
    }

    /// Vertical offset of a run's glyphs from the baseline (y down).
    fn shift_of(st: &CharStyle) -> f32 {
        -st.baseline_shift
            + match st.position {
                Position::Normal => 0.0,
                Position::Super => -st.size * 0.333,
                Position::Sub => st.size * 0.333,
            }
    }

    /// Hyphen advance before glyph `k` of a paragraph where Liang's patterns
    /// (or, for non-ASCII words, the syllable heuristic) allow a break.
    fn hyphen_points(&self, cx: &ParaCtx, a: usize) -> Vec<Option<f32>> {
        let gl = &cx.shaped.glyphs;
        let n = gl.len();
        let mut out = vec![None; n + 1];
        let mut w0 = 0;
        while w0 < n {
            if !cx.chars[w0].is_alphabetic() {
                w0 += 1;
                continue;
            }
            let mut we = w0;
            while we < n && cx.chars[we].is_alphabetic() {
                we += 1;
            }
            let (b0, b1) = (a + gl[w0].start, a + gl[we - 1].end);
            let word = &self.text[b0..b1];
            let letters: Vec<usize> = match super::hyphen::breaks(word) {
                Some(v) => v,
                None => heuristic_breaks(word),
            };
            for li in letters {
                // Letter index → byte offset → the glyph starting there.
                let off = word.char_indices().nth(li).map_or(word.len(), |(o, _)| o);
                if let Some(k) = (w0 + 1..we).find(|&k| a + gl[k].start == b0 + off && gl[k].start != gl[k - 1].start) {
                    out[k] = Some(cx.shaped.hyphens.get(&gl[k - 1].run).map_or(0.0, |h| h.adv));
                }
            }
            w0 = we;
        }
        out
    }

    fn layout_horizontal(&self) -> TextLayout {
        let area = self.path.as_ref().filter(|t| t.area).map(|t| t.path.flatten(0.25)).filter(|p| !p.is_empty());
        let wrapping = self.box_size.is_some() || area.is_some();
        let (bx0, by0, bx1, by1) = match (&area, self.box_size) {
            (Some(polys), _) => {
                let (mut lo, mut hi) = (p(f32::MAX, f32::MAX), p(f32::MIN, f32::MIN));
                for q in polys.iter().flat_map(|pl| pl.pts.iter()) {
                    lo = p(lo.x.min(q.x), lo.y.min(q.y));
                    hi = p(hi.x.max(q.x), hi.y.max(q.y));
                }
                (lo.x, lo.y, hi.x, hi.y)
            }
            (None, Some((w, h))) => (self.origin.0, self.origin.1, self.origin.0 + w, self.origin.1 + h),
            (None, None) => (self.origin.0, f32::MIN, self.origin.0, f32::MAX),
        };
        let mut out = TextLayout::default();
        let mut prev_base: Option<f32> = None;
        let mut prev_after = 0.0f32;
        let mut hidden = false;
        for (pi, (a, b)) in self.para_ranges().into_iter().enumerate() {
            let ps = self.paras.get(pi).copied().unwrap_or_else(|| self.style.para_style());
            let cx = self.para_ctx(a, b, ps.hyphenate && wrapping);
            let gl = &cx.shaped.glyphs;
            let n = gl.len();
            let lh_est = cx.runs.iter().map(|r| r.style.line_height()).fold(0.0f32, f32::max);
            let gap = if pi > 0 { prev_after + ps.space_before } else { 0.0 };
            prev_after = ps.space_after;
            // Where this paragraph's first baseline would sit (area type
            // needs it before the lines exist).
            let est_base0 = prev_base.map_or(by0 + lh_est * 0.8, |bl| bl + lh_est + gap);
            let space: Vec<bool> = cx.chars.iter().map(|c| c.is_whitespace()).collect();
            let brk: Vec<bool> = (0..n)
                .map(|i| {
                    let c = cx.chars[i];
                    let next_same = i + 1 < n && gl[i + 1].start == gl[i].start;
                    !next_same && (c.is_whitespace() || matches!(c, '-' | '\u{2013}' | '\u{2014}' | '/') || is_cjk(c) || (i + 1 < n && is_cjk(cx.chars[i + 1])))
                })
                .collect();
            let indent = |line: usize| ps.indent_left + if line == 0 { ps.indent_first } else { 0.0 };
            let mut intervals: Vec<(f32, f32)> = Vec::new();
            // Called once per line, in order (area type records each band).
            let mut avail = |line: usize| -> f32 {
                let w = match &area {
                    Some(polys) => {
                        let bl = est_base0 + line as f32 * lh_est;
                        let iv = band_interval(polys, bl - lh_est * 0.8, bl + lh_est * 0.2).unwrap_or((bx0, bx0));
                        if intervals.len() == line {
                            intervals.push(iv);
                        }
                        // Bands narrower than a few ems stay empty (Photoshop
                        // skips them rather than setting a letter per line).
                        if iv.1 - iv.0 < lh_est * 3.0 && bl < by1 {
                            return -1.0;
                        }
                        iv.1 - iv.0
                    }
                    None if wrapping => bx1 - bx0,
                    None => return f32::INFINITY,
                };
                if w < 0.0 { w } else { (w - indent(line) - ps.indent_right).max(0.0) }
            };
            // Hyphen width before glyph k where a break is allowed there.
            let hy_at: Vec<Option<f32>> = if ps.hyphenate && wrapping { self.hyphen_points(&cx, a) } else { vec![None; n + 1] };
            let hyph = |s: usize, w0: usize, e: usize, av: f32| -> Option<usize> {
                // The last allowed point of the overflowing word that still fits.
                let mut we = e;
                while we < n && cx.chars[we].is_alphabetic() {
                    we += 1;
                }
                (w0 + 1..we).rev().find(|&k| k > s && hy_at[k].is_some_and(|hw| cx.adv_range(s, k) + hw <= av))
            };
            let adv: Vec<f32> = cx.pre[1..].iter().zip(&cx.pre).map(|(b, a)| b - a).collect();
            let justify_para = ps.align == TextAlign::Justify;
            let breaks = if n == 0 {
                vec![(0, 0, false)]
            } else if ps.composer == Composer::Every && wrapping && area.is_none() {
                let widths: Vec<f32> = (0..n + 1).map(&mut avail).collect();
                kp_breaks(&adv, &space, &brk, &hy_at, &widths, justify_para, ps.last == LastLine::All)
                    .unwrap_or_else(|| greedy_breaks(&adv, &space, &brk, |l| widths[l.min(n)], hyph))
            } else {
                greedy_breaks(&adv, &space, &brk, &mut avail, hyph)
            };
            let nl = breaks.len();
            for (li, &(s, e, hyphen)) in breaks.iter().enumerate() {
                let last = li + 1 == nl;
                let st0 = &cx.runs[gl.get(s).map_or(0, |g| g.run)].style;
                // Line metrics: the largest leading, ascent and descent.
                let (mut lh, mut asc, mut desc) = (0.0f32, 0.0f32, 0.0f32);
                for g in &gl[s..e] {
                    let st = &cx.runs[g.run].style;
                    let fm = font_metrics(g.font_id, g.weight);
                    lh = lh.max(st.line_height());
                    asc = asc.max(fm.asc * st.size);
                    desc = desc.max(fm.desc * st.size);
                }
                if s == e {
                    lh = st0.line_height();
                    asc = st0.size * 0.8;
                    desc = st0.size * 0.2;
                }
                let baseline = match prev_base {
                    None if self.box_size.is_none() && area.is_none() => self.origin.1,
                    None => by0 + asc,
                    Some(bl) => bl + lh + if li == 0 { gap } else { 0.0 },
                };
                prev_base = Some(baseline);
                let top = baseline - asc - (lh - asc - desc) * 0.5;
                if wrapping && baseline + desc > by1 + 0.5 {
                    hidden = true;
                }
                let lstart = a + if li == 0 { 0 } else { gl[s].start };
                let lend = breaks.get(li + 1).map_or(b, |nb| a + gl[nb.0].start);
                if hidden {
                    out.overflow = true;
                    let g0 = out.glyphs.len();
                    out.lines.push(LineInfo { top, height: lh, baseline, start: lstart, end: lend, x0: bx0, x1: bx0, g0, g1: g0 });
                    continue;
                }
                let mut e_trim = e;
                while e_trim > s && space[e_trim - 1] {
                    e_trim -= 1;
                }
                let hy = if hyphen { cx.shaped.hyphens.get(&gl[e - 1].run).copied() } else { None };
                let natural = cx.adv_range(s, e_trim) + hy.map_or(0.0, |h| h.adv);
                let (lx0, lw) = match &area {
                    Some(_) => {
                        let iv = intervals.get(li).copied().unwrap_or((bx0, bx1));
                        (iv.0 + indent(li), (iv.1 - iv.0 - indent(li) - ps.indent_right).max(0.0))
                    }
                    None => (bx0 + indent(li), (bx1 - bx0 - indent(li) - ps.indent_right).max(0.0)),
                };
                let extra = lw - natural;
                // Justified lines stretch (or, from the composer, shrink) their spaces.
                let justify = wrapping && ps.align == TextAlign::Justify && (!last || ps.last == LastLine::All) && extra.abs() > 0.01;
                let nsp = (s..e_trim).filter(|&i| space[i]).count();
                // Word spacing only (Photoshop's default letter-spacing limit
                // is 0 %): a line without spaces stays flush with the start.
                let per_space = if justify && nsp > 0 { extra / nsp as f32 } else { 0.0 };
                let al = match ps.align {
                    TextAlign::Justify => match ps.last {
                        LastLine::Center => TextAlign::Center,
                        LastLine::Right => TextAlign::Right,
                        _ => TextAlign::Left,
                    },
                    a => a,
                };
                let mut pen = if wrapping {
                    lx0 + if justify {
                        0.0
                    } else {
                        match al {
                            TextAlign::Center => extra * 0.5,
                            TextAlign::Right => extra,
                            _ => 0.0,
                        }
                    }
                } else {
                    match al {
                        TextAlign::Center => self.origin.0 - natural * 0.5 + (ps.indent_left - ps.indent_right) * 0.5,
                        TextAlign::Right => self.origin.0 - natural - ps.indent_right,
                        _ => lx0,
                    }
                };
                let g0 = out.glyphs.len();
                let lx = pen;
                let placed: Vec<(SGlyph, bool)> = gl[s..e].iter().map(|g| (*g, false)).chain(hy.map(|h| (SGlyph { start: gl[e - 1].end, end: gl[e - 1].end, ..h }, true))).collect();
                for (j, (g, soft)) in placed.iter().enumerate() {
                    let i = s + j;
                    let st = &cx.runs[g.run].style;
                    let mut adv = g.adv;
                    if !soft && i < e_trim && space[i] {
                        adv += per_space;
                    }
                    let x0 = pen;
                    out.glyphs.push(PlacedGlyph {
                        font_id: g.font_id,
                        glyph: g.glyph,
                        size: g.size,
                        weight: g.weight,
                        x: pen + g.xo,
                        y: baseline + g.yo + Self::shift_of(st),
                        advance: adv,
                        start: a + g.start,
                        end: a + g.end,
                        color: st.color,
                        synth: Self::synth_of(st) | if *soft { SYNTH_HYPHEN } else { 0 },
                        rot: 0.0,
                        cell: [p(x0, top), p(x0, top + lh), p(x0 + adv, top + lh), p(x0 + adv, top)],
                    });
                    pen += adv;
                }
                // Decorations: one quad per run segment, trailing spaces excluded.
                let vis: Vec<usize> = (0..placed.len()).filter(|&j| placed[j].1 || s + j < e_trim).collect();
                let mut j = 0;
                while j < vis.len() {
                    let run = placed[vis[j]].0.run;
                    let mut k = j;
                    while k + 1 < vis.len() && placed[vis[k + 1]].0.run == run {
                        k += 1;
                    }
                    let st = &cx.runs[run].style;
                    if st.underline || st.strike {
                        let g = &placed[vis[j]].0;
                        let fm = font_metrics(g.font_id, g.weight);
                        let (dx0, dx1) = (out.glyphs[g0 + vis[j]].cell[0].x, out.glyphs[g0 + vis[k]].cell[2].x);
                        let th = (fm.thick * st.size).max(1.0);
                        let by = baseline - st.baseline_shift;
                        for (on, y) in [(st.underline, by + fm.ul * st.size), (st.strike, by - fm.strike * st.size)] {
                            if on {
                                out.decos.push(Decoration { quad: local_quad(p(0.0, 0.0), 0.0, dx0, dx1, y - th * 0.5, y + th * 0.5), color: st.color });
                            }
                        }
                    }
                    j = k + 1;
                }
                out.lines.push(LineInfo { top, height: lh, baseline, start: lstart, end: lend, x0: lx, x1: pen, g0, g1: out.glyphs.len() });
            }
        }
        out
    }
}

impl TextLayer {
    fn in_tcy(&self, i: usize) -> Option<(usize, usize)> {
        self.tcy.iter().copied().find(|&(a, b)| i >= a && i < b)
    }

    /// Vertical type: columns right to left. CJK stays upright, Latin is
    /// rotated 90° clockwise (or upright with Standard Vertical Roman
    /// Alignment), tate-chu-yoko ranges sit across the column.
    fn layout_vertical(&self) -> TextLayout {
        let mut out = TextLayout::default();
        let col_len = self.box_size.map(|(_, h)| h);
        let first_lh = self.runs.first().map_or(self.style.line_height(), |r| r.style.line_height());
        // Point text: the origin is the top of the first column's centre;
        // box text starts at the box's top-right corner.
        let mut cx_col = match self.box_size {
            Some((w, _)) => self.origin.0 + w - first_lh * 0.5,
            None => self.origin.0,
        };
        let top0 = self.origin.1;
        let mut prev_after = 0.0;
        let mut first_col = true;
        for (pi, (a, b)) in self.para_ranges().into_iter().enumerate() {
            let ps = self.paras.get(pi).copied().unwrap_or_else(|| self.style.para_style());
            let cx = self.para_ctx(a, b, false);
            let gl = &cx.shaped.glyphs;
            let n = gl.len();
            // Unit advances along the column.
            let mut adv = vec![0.0f32; n];
            let mut upright = vec![false; n];
            let mut tcy_of = vec![None; n];
            for i in 0..n {
                let st = &cx.runs[gl[i].run].style;
                let em = st.size * (1.0 + st.tracking / 1000.0);
                let c = cx.chars[i];
                tcy_of[i] = self.in_tcy(a + gl[i].start);
                if let Some(t) = tcy_of[i] {
                    let first = i == 0 || tcy_of[i - 1] != Some(t);
                    adv[i] = if first { em } else { 0.0 };
                    upright[i] = true;
                } else if is_cjk(c) || self.upright_roman {
                    adv[i] = if c.is_whitespace() && !is_cjk(c) { st.size * 0.5 } else { em };
                    upright[i] = true;
                } else {
                    adv[i] = gl[i].adv;
                }
            }
            let space: Vec<bool> = cx.chars.iter().map(|c| c.is_whitespace()).collect();
            let brk: Vec<bool> = (0..n)
                .map(|i| {
                    let inside_tcy = tcy_of[i].is_some() && i + 1 < n && tcy_of[i + 1] == tcy_of[i];
                    !inside_tcy && (i + 1 >= n || gl[i + 1].start != gl[i].start) && (space[i] || upright[i] || (i + 1 < n && upright[i + 1]))
                })
                .collect();
            let indent = |line: usize| ps.indent_left + if line == 0 { ps.indent_first } else { 0.0 };
            let breaks = if n == 0 {
                vec![(0, 0, false)]
            } else {
                greedy_breaks(&adv, &space, &brk, |line| col_len.map_or(f32::INFINITY, |h| (h - indent(line) - ps.indent_right).max(0.0)), |_, _, _, _| None)
            };
            let gap = if pi > 0 { prev_after + ps.space_before } else { 0.0 };
            prev_after = ps.space_after;
            for (li, &(s, e, _)) in breaks.iter().enumerate() {
                let st0 = &cx.runs[gl.get(s).map_or(0, |g| g.run)].style;
                let lh = gl[s..e].iter().map(|g| cx.runs[g.run].style.line_height()).fold(if s == e { st0.line_height() } else { 0.0 }, f32::max);
                if !first_col {
                    cx_col -= lh + if li == 0 { gap } else { 0.0 };
                }
                first_col = false;
                let natural: f32 = adv[s..e].iter().zip(&space[s..e]).rev().skip_while(|(_, sp)| **sp).map(|(a, _)| a).sum();
                let along = match (col_len, ps.align) {
                    (Some(h), TextAlign::Center) => (h - indent(li) - ps.indent_right - natural) * 0.5,
                    (Some(h), TextAlign::Right) => h - indent(li) - ps.indent_right - natural,
                    (None, TextAlign::Center) => -natural * 0.5,
                    (None, TextAlign::Right) => -natural,
                    _ => 0.0,
                };
                let col_top = top0 + indent(li) + along;
                let mut y = col_top;
                let g0 = out.glyphs.len();
                let mut i = s;
                while i < e {
                    let g = &gl[i];
                    let st = &cx.runs[g.run].style;
                    let fm = font_metrics(g.font_id, g.weight);
                    let shift = Self::shift_of(st);
                    let track = st.tracking / 1000.0 * g.size;
                    if let Some(t) = tcy_of[i] {
                        // The whole tate-chu-yoko group, centred across the column.
                        let mut j = i;
                        while j < e && tcy_of[j] == Some(t) {
                            j += 1;
                        }
                        let w: f32 = gl[i..j].iter().map(|g| g.adv).sum();
                        let mut x = cx_col - w * 0.5;
                        let base = y + st.size * fm.asc / (fm.asc + fm.desc);
                        for k in i..j {
                            let gk = &gl[k];
                            let sk = &cx.runs[gk.run].style;
                            out.glyphs.push(PlacedGlyph {
                                font_id: gk.font_id,
                                glyph: gk.glyph,
                                size: gk.size,
                                weight: gk.weight,
                                x: x + gk.xo,
                                y: base + gk.yo + Self::shift_of(sk),
                                advance: gk.adv,
                                start: a + gk.start,
                                end: a + gk.end,
                                color: sk.color,
                                synth: Self::synth_of(sk),
                                rot: 0.0,
                                cell: [p(x, y), p(x, y + adv[i]), p(x + gk.adv, y + adv[i]), p(x + gk.adv, y)],
                            });
                            x += gk.adv;
                        }
                        y += adv[i];
                        i = j;
                        continue;
                    }
                    let (pen, rot) = if upright[i] {
                        // Centred on the column, baseline at the ideographic ascent.
                        (p(cx_col - (g.adv - track) * 0.5 + g.xo, y + st.size * fm.asc / (fm.asc + fm.desc) + g.yo + shift), 0.0)
                    } else {
                        // Rotated: the glyph's top faces right; centre its body.
                        let off = (fm.asc - fm.desc) * g.size * 0.5;
                        (p(cx_col - off - g.yo - shift, y + g.xo), std::f32::consts::FRAC_PI_2)
                    };
                    let half = lh * 0.5;
                    out.glyphs.push(PlacedGlyph {
                        font_id: g.font_id,
                        glyph: g.glyph,
                        size: g.size,
                        weight: g.weight,
                        x: pen.x,
                        y: pen.y,
                        advance: adv[i],
                        start: a + g.start,
                        end: a + g.end,
                        color: st.color,
                        synth: Self::synth_of(st),
                        rot,
                        cell: [p(cx_col + half, y), p(cx_col - half, y), p(cx_col - half, y + adv[i]), p(cx_col + half, y + adv[i])],
                    });
                    if st.underline || st.strike {
                        let th = (fm.thick * st.size).max(1.0);
                        // Underline on the right of the column (Photoshop's
                        // default for vertical type); strike through the middle.
                        for (on, x) in [(st.underline, cx_col + st.size * 0.55), (st.strike, cx_col)] {
                            if on {
                                out.decos.push(Decoration { quad: local_quad(p(0.0, 0.0), 0.0, x - th * 0.5, x + th * 0.5, y, y + adv[i]), color: st.color });
                            }
                        }
                    }
                    y += adv[i];
                    i += 1;
                }
                let lstart = a + if li == 0 { 0 } else { gl[s].start };
                let lend = breaks.get(li + 1).map_or(b, |nb| a + gl[nb.0].start);
                out.lines.push(LineInfo { top: col_top, height: (y - col_top).max(1.0), baseline: cx_col, start: lstart, end: lend, x0: cx_col - lh * 0.5, x1: cx_col + lh * 0.5, g0, g1: out.glyphs.len() });
                if col_len.is_some_and(|h| y - top0 > h + 0.5) {
                    out.overflow = true;
                }
            }
        }
        out
    }

    /// Type on a path: glyphs follow the path's first subpath between the
    /// start and end markers; Flip runs them the other way, on the other side.
    fn layout_on_path(&self, tp: &TextPath) -> TextLayout {
        let mut out = TextLayout::default();
        let Some(pl) = tp.path.flatten(0.2).into_iter().find(|pl| pl.pts.len() > 1) else { return out };
        let mut pts = pl.pts.clone();
        if pl.closed {
            pts.push(pts[0]);
        }
        let s0 = tp.start.clamp(0.0, 1.0);
        let e0 = if pl.closed { tp.end.clamp(s0, s0 + 1.0) } else { tp.end.clamp(s0, 1.0) };
        let (s, e) = if tp.flip {
            // Reversed parameter: the same stretch of path, run backwards.
            pts.reverse();
            let (a, b) = (1.0 - e0, 1.0 - s0);
            if a < 0.0 { (a + 1.0, b + 1.0) } else { (a, b) }
        } else {
            (s0, e0)
        };
        let mut cum = vec![0.0f32];
        for w in pts.windows(2) {
            cum.push(cum.last().unwrap() + w[0].dist(w[1]));
        }
        let total = *cum.last().unwrap();
        if total <= 0.0 {
            return out;
        }
        let (s, e) = (s * total, e * total);
        let at = |d: f32| -> (P, P) {
            let d = if pl.closed { d.rem_euclid(total) } else { d.clamp(0.0, total) };
            let k = cum.partition_point(|&c| c < d).clamp(1, pts.len() - 1);
            let (a, b) = (pts[k - 1], pts[k]);
            let seg = cum[k] - cum[k - 1];
            let t = if seg > 0.0 { (d - cum[k - 1]) / seg } else { 0.0 };
            (a.lerp(b, t), (b - a).norm())
        };
        let ctxs: Vec<ParaCtx> = self.para_ranges().into_iter().map(|(a, b)| self.para_ctx(a, b, false)).collect();
        let width: f32 = ctxs.iter().map(|c| *c.pre.last().unwrap()).sum();
        let ps = self.paras.first().copied().unwrap_or_default();
        let mut d = match ps.align {
            TextAlign::Center => (s + e) * 0.5 - width * 0.5,
            TextAlign::Right => e - width,
            _ => s,
        };
        let ranges = self.para_ranges();
        for (cx, &(a, b)) in ctxs.iter().zip(&ranges) {
            let g0 = out.glyphs.len();
            for g in &cx.shaped.glyphs {
                let st = &cx.runs[g.run].style;
                let mid = d + g.adv * 0.5;
                d += g.adv;
                let outside = mid < s || mid > e || (!pl.closed && (mid < 0.0 || mid > total));
                if outside {
                    out.overflow = true;
                    continue;
                }
                let (q, t) = at(mid);
                let up = p(t.y, -t.x);
                let rot = t.y.atan2(t.x);
                let lift = -(Self::shift_of(st) + g.yo);
                let pen = q - t * (g.adv * 0.5 - g.xo) + up * lift;
                let fm = font_metrics(g.font_id, g.weight);
                let cell_pen = q - t * (g.adv * 0.5);
                out.glyphs.push(PlacedGlyph {
                    font_id: g.font_id,
                    glyph: g.glyph,
                    size: g.size,
                    weight: g.weight,
                    x: pen.x,
                    y: pen.y,
                    advance: g.adv,
                    start: a + g.start,
                    end: a + g.end,
                    color: st.color,
                    synth: Self::synth_of(st),
                    rot,
                    cell: local_quad(cell_pen, rot, 0.0, g.adv, -fm.asc * st.size, fm.desc * st.size),
                });
                let th = (fm.thick * st.size).max(1.0);
                for (on, y) in [(st.underline, fm.ul * st.size), (st.strike, -fm.strike * st.size)] {
                    if on {
                        out.decos.push(Decoration { quad: local_quad(cell_pen, rot, 0.0, g.adv, y - th * 0.5, y + th * 0.5), color: st.color });
                    }
                }
            }
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for g in &out.glyphs[g0..] {
                for q in g.cell {
                    x0 = x0.min(q.x);
                    y0 = y0.min(q.y);
                    x1 = x1.max(q.x);
                    y1 = y1.max(q.y);
                }
            }
            if x0 > x1 {
                let (q, _) = at(d);
                (x0, y0, x1, y1) = (q.x, q.y, q.x, q.y);
            }
            out.lines.push(LineInfo { top: y0, height: y1 - y0, baseline: y1, start: a, end: b, x0, x1, g0, g1: out.glyphs.len() });
        }
        out
    }
}

// ----- glyph outlines and coverage cache ------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GlyphKey {
    font: cosmic_text::fontdb::ID,
    glyph: u16,
    size_bits: u32,
    weight: u16,
    synth: u8,
    phase: u8,
}

/// A coverage bitmap, positioned relative to its integer pen position.
pub struct GlyphBitmap {
    pub x0: i32,
    pub y0: i32,
    pub w: usize,
    pub h: usize,
    pub cov: Vec<u8>,
}

fn glyph_cache() -> &'static Mutex<HashMap<GlyphKey, Arc<GlyphBitmap>>> {
    static C: OnceLock<Mutex<HashMap<GlyphKey, Arc<GlyphBitmap>>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// Faux italic shear (Photoshop slants by about 12°) and faux bold offset.
const FAUX_SHEAR: f32 = 0.2126;
const FAUX_BOLD: f32 = 0.02;

/// The glyph outline at `size` as polygons, y down, origin on the pen, with
/// faux bold (union of offset copies) and faux italic (shear) applied.
fn glyph_polys(fs: &mut FontSystem, font: cosmic_text::fontdb::ID, glyph: u16, size: f32, weight: u16, synth: u8) -> Vec<Poly> {
    use swash::zeno::{Command, PathData as _};
    let Some(f) = fs.get_font(font, Weight(weight)) else { return vec![] };
    let mut ctx = swash::scale::ScaleContext::new();
    let mut scaler = ctx.builder(f.as_swash()).size(size).hint(false).build();
    let Some(outline) = scaler.scale_outline(glyph).or_else(|| scaler.scale_color_outline(glyph)) else { return vec![] };
    let mut polys: Vec<Poly> = Vec::new();
    let mut cur: Vec<P> = Vec::new();
    let mut last = p(0.0, 0.0);
    let tol = 0.05;
    let shear = if synth & SYNTH_ITALIC != 0 { FAUX_SHEAR } else { 0.0 };
    let flip = |x: f32, y: f32| p(x + y * shear, -y);
    for cmd in outline.path().commands() {
        match cmd {
            Command::MoveTo(a) => {
                if cur.len() > 2 {
                    polys.push(Poly { pts: std::mem::take(&mut cur), closed: true });
                }
                cur.clear();
                last = flip(a.x, a.y);
                cur.push(last);
            }
            Command::LineTo(a) => {
                last = flip(a.x, a.y);
                cur.push(last);
            }
            Command::QuadTo(c, a) => {
                let (c, a) = (flip(c.x, c.y), flip(a.x, a.y));
                // Degree-elevate to a cubic and flatten.
                let cub = [last, last + (c - last) * (2.0 / 3.0), a + (c - a) * (2.0 / 3.0), a];
                let n = vector::cubic_steps(&cub, tol);
                for k in 1..=n {
                    cur.push(vector::cubic_at(&cub, k as f32 / n as f32));
                }
                last = a;
            }
            Command::CurveTo(c1, c2, a) => {
                let cub = [last, flip(c1.x, c1.y), flip(c2.x, c2.y), flip(a.x, a.y)];
                let n = vector::cubic_steps(&cub, tol);
                for k in 1..=n {
                    cur.push(vector::cubic_at(&cub, k as f32 / n as f32));
                }
                last = cub[3];
            }
            Command::Close => {
                if cur.len() > 2 {
                    polys.push(Poly { pts: std::mem::take(&mut cur), closed: true });
                }
                cur.clear();
            }
        }
    }
    if cur.len() > 2 {
        polys.push(Poly { pts: cur, closed: true });
    }
    if synth & SYNTH_BOLD != 0 {
        // Non-zero union of four offset copies thickens every stroke.
        let d = size * FAUX_BOLD;
        let base = std::mem::take(&mut polys);
        for (dx, dy) in [(-d, -d), (d, -d), (-d, d), (d, d)] {
            polys.extend(base.iter().map(|pl| Poly { pts: pl.pts.iter().map(|q| p(q.x + dx * 0.5, q.y + dy * 0.5)).collect(), closed: true }));
        }
    }
    polys
}

fn bitmap_of(polys: &[Poly], dx: f32) -> GlyphBitmap {
    let (mut lo, mut hi) = (p(f32::MAX, f32::MAX), p(f32::MIN, f32::MIN));
    for pl in polys {
        for q in &pl.pts {
            lo = p(lo.x.min(q.x + dx), lo.y.min(q.y));
            hi = p(hi.x.max(q.x + dx), hi.y.max(q.y));
        }
    }
    if polys.is_empty() || lo.x > hi.x {
        return GlyphBitmap { x0: 0, y0: 0, w: 0, h: 0, cov: vec![] };
    }
    let (x0, y0) = (lo.x.floor() as i32, lo.y.floor() as i32);
    let (w, h) = ((hi.x.ceil() as i32 - x0 + 1) as usize, (hi.y.ceil() as i32 - y0 + 1) as usize);
    let shifted: Vec<Poly> = polys.iter().map(|pl| Poly { pts: pl.pts.iter().map(|q| p(q.x + dx - x0 as f32, q.y - y0 as f32)).collect(), closed: true }).collect();
    let cov = vector::raster_buffer(&vector::edges(&shifted), w, h, FillRule::NonZero);
    GlyphBitmap { x0, y0, w, h, cov }
}

fn glyph_bitmap(fs: &mut FontSystem, g: &PlacedGlyph, phase: u8) -> Arc<GlyphBitmap> {
    let synth = g.synth & (SYNTH_BOLD | SYNTH_ITALIC);
    let key = GlyphKey { font: g.font_id, glyph: g.glyph, size_bits: g.size.to_bits(), weight: g.weight, synth, phase };
    if let Some(b) = glyph_cache().lock().unwrap().get(&key) {
        return b.clone();
    }
    let polys = glyph_polys(fs, g.font_id, g.glyph, g.size, g.weight, synth);
    let b = Arc::new(bitmap_of(&polys, phase as f32 * 0.25));
    glyph_cache().lock().unwrap().insert(key, b.clone());
    b
}

/// Composites coverage `cov` in `color` over the premultiplied `dst`.
fn flush_cov(dst: &mut [u32], cov: &mut [u8], color: [u8; 3], first: bool) {
    let c = px::pack(color[0], color[1], color[2], 255);
    if first {
        for (d, v) in dst.iter_mut().zip(cov.iter()) {
            *d = px::scale(c, *v as u32);
        }
    } else {
        for (d, v) in dst.iter_mut().zip(cov.iter()) {
            if *v > 0 {
                *d = px::over(px::scale(c, *v as u32), *d);
            }
        }
    }
    cov.fill(0);
}

// ----- rendering ------------------------------------------------------------------------------

impl TextLayer {
    fn warper(&self) -> Option<super::warp::BoxWarp> {
        (self.warp.style != WarpStyle::None).then(|| super::warp::BoxWarp::new(&self.warp, self.layout.line_box))
    }

    /// Rotated glyphs or a warp: render through vector outlines.
    fn needs_outline(&self) -> bool {
        self.warp.style != WarpStyle::None || self.layout.glyphs.iter().any(|g| g.rot != 0.0)
    }

    fn glyph_doc_polys(fs: &mut FontSystem, g: &PlacedGlyph, w: Option<&super::warp::BoxWarp>) -> Vec<Poly> {
        let (s, c) = g.rot.sin_cos();
        glyph_polys(fs, g.font_id, g.glyph, g.size, g.weight, g.synth & (SYNTH_BOLD | SYNTH_ITALIC))
            .into_iter()
            .map(|pl| {
                let pts = pl
                    .pts
                    .iter()
                    .map(|q| {
                        let d = p(g.x + q.x * c - q.y * s, g.y + q.x * s + q.y * c);
                        w.map_or(d, |w| w.map(d))
                    })
                    .collect();
                Poly { pts, closed: true }
            })
            .collect()
    }

    fn deco_doc_poly(d: &Decoration, w: Option<&super::warp::BoxWarp>) -> Poly {
        let Some(w) = w else { return Poly { pts: d.quad.to_vec(), closed: true } };
        // Subdivide the long edges so the bar follows the warp.
        let mut pts = Vec::with_capacity(4 * 24);
        for k in 0..4 {
            let (a, b) = (d.quad[k], d.quad[(k + 1) % 4]);
            let n = ((a.dist(b) / 4.0).ceil() as usize).clamp(1, 64);
            for j in 0..n {
                pts.push(w.map(a.lerp(b, j as f32 / n as f32)));
            }
        }
        Poly { pts, closed: true }
    }

    /// All glyph and decoration outlines in document coordinates (warped),
    /// for Type Mask selections and warped rendering.
    pub fn outlines(&self) -> Vec<Poly> {
        let mut fs = fonts().lock().unwrap();
        let w = self.warper();
        let mut out = Vec::new();
        for g in &self.layout.glyphs {
            out.extend(Self::glyph_doc_polys(&mut fs, g, w.as_ref()));
        }
        out.extend(self.layout.decos.iter().map(|d| Self::deco_doc_poly(d, w.as_ref())));
        out
    }

    /// Ink bounds (with a margin for antialiasing and faux styles).
    pub fn bounds(&self) -> RectI {
        let l = &self.layout;
        let mut bb = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        let grow = |bb: &mut (f32, f32, f32, f32), a: f32, b: f32, c: f32, d: f32| *bb = (bb.0.min(a), bb.1.min(b), bb.2.max(c), bb.3.max(d));
        for g in &l.glyphs {
            let s = g.size;
            if g.rot == 0.0 {
                grow(&mut bb, g.x - s * 0.3, g.y - s * 1.1, g.x + g.advance.max(s * 0.6) + s * 0.4, g.y + s * 0.4);
            } else {
                // Rotated: the cell (pen to advance, ascent to descent) plus
                // the overhang a glyph can have past it.
                let m = s * 0.35;
                for q in g.cell {
                    grow(&mut bb, q.x - m, q.y - m, q.x + m, q.y + m);
                }
            }
        }
        for d in &l.decos {
            for q in d.quad {
                grow(&mut bb, q.x - 1.0, q.y - 1.0, q.x + 1.0, q.y + 1.0);
            }
        }
        if let Some(w) = self.warper() {
            // Sample the warp densely over the line box grown by the glyphs'
            // overhang (ink can poke out of the line box by a fraction of an
            // em), then pad by the largest gap between samples.
            let m = l.glyphs.iter().map(|g| g.size).fold(0.0f32, f32::max) * 0.35;
            let (x0, y0, x1, y1) = l.line_box;
            let (x0, y0, x1, y1) = (x0.min(bb.0) - m, y0.min(bb.1) - m, x1.max(bb.2) + m, y1.max(bb.3) + m);
            const N: usize = 32;
            let mut wb = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            let mut gap = 0.0f32;
            for j in 0..=N {
                let mut prev: Option<P> = None;
                for i in 0..=N {
                    let q = w.map(p(x0 + (x1 - x0) * i as f32 / N as f32, y0 + (y1 - y0) * j as f32 / N as f32));
                    grow(&mut wb, q.x, q.y, q.x, q.y);
                    if let Some(pq) = prev {
                        gap = gap.max(pq.dist(q));
                    }
                    prev = Some(q);
                }
            }
            bb = (wb.0 - gap * 0.25 - 2.0, wb.1 - gap * 0.25 - 2.0, wb.2 + gap * 0.25 + 2.0, wb.3 + gap * 0.25 + 2.0);
        }
        if let Some((w, h)) = self.box_size {
            grow(&mut bb, self.origin.0, self.origin.1, self.origin.0 + w, self.origin.1 + h);
        }
        let (x0, y0, x1, y1) = bb;
        if x0 > x1 {
            return RectI::new(self.origin.0 as i32, self.origin.1 as i32, self.origin.0 as i32 + 1, self.origin.1 as i32 + 1);
        }
        RectI::new((x0 - 2.0).floor() as i32, (y0 - 2.0).floor() as i32, (x1 + 2.0).ceil() as i32, (y1 + 2.0).ceil() as i32)
    }

    /// Renders the tiles of `region` (whole tiles; others untouched).
    pub fn render(&self, grid: &mut TileGrid, region: RectI) {
        let region = region.intersect(&grid.bounds());
        let (tx0, ty0, tx1, ty1) = region.tile_span(grid.cols, grid.rows);
        if self.needs_outline() {
            self.render_outline(grid, region);
            return;
        }
        // Coverage blobs in paint order: glyphs, then decorations.
        let blobs: Vec<(Arc<GlyphBitmap>, i32, i32, [u8; 3])> = {
            let mut fs = fonts().lock().unwrap();
            let mut v: Vec<_> = self
                .layout
                .glyphs
                .iter()
                .map(|g| {
                    let fx = g.x.floor();
                    let phase = (((g.x - fx) * 4.0) as u8).min(3);
                    (glyph_bitmap(&mut fs, g, phase), fx as i32, g.y.round() as i32, g.color)
                })
                .collect();
            v.extend(self.layout.decos.iter().map(|d| (Arc::new(bitmap_of(&[Poly { pts: d.quad.to_vec(), closed: true }], 0.0)), 0, 0, d.color)));
            v
        };
        let mut by_tile: HashMap<usize, Vec<usize>> = HashMap::new();
        for (k, (b, x, y, _)) in blobs.iter().enumerate() {
            if b.w == 0 {
                continue;
            }
            let r = RectI::wh(x + b.x0, y + b.y0, b.w as i32, b.h as i32).intersect(&region);
            let (a0, b0, a1, b1) = r.tile_span(grid.cols, grid.rows);
            for ty in b0..b1 {
                for tx in a0..a1 {
                    by_tile.entry(grid.idx(tx, ty)).or_default().push(k);
                }
            }
        }
        let cols = grid.cols;
        let work: Vec<(i32, i32)> = (ty0..ty1).flat_map(|ty| (tx0..tx1).map(move |tx| (tx, ty))).collect();
        let done: Vec<(usize, TileRef)> = work
            .par_iter()
            .map(|&(tx, ty)| {
                let i = (ty as u32 * cols + tx as u32) as usize;
                let Some(list) = by_tile.get(&i) else { return (i, None) };
                let mut cov = vec![0u8; TS * TS];
                let mut dst = vec![0u32; TS * TS];
                let (ox, oy) = (tx * TS_I, ty * TS_I);
                let mut cur: Option<[u8; 3]> = None;
                let mut first = true;
                for &k in list {
                    let (b, x, y, color) = &blobs[k];
                    // Same-colour neighbours add coverage (no seams between
                    // touching glyphs); a colour change composites over.
                    if cur.is_some_and(|c| c != *color) {
                        flush_cov(&mut dst, &mut cov, cur.unwrap(), first);
                        first = false;
                    }
                    cur = Some(*color);
                    let (gx, gy) = (x + b.x0, y + b.y0);
                    for row in 0..b.h as i32 {
                        let ly = gy + row - oy;
                        if !(0..TS_I).contains(&ly) {
                            continue;
                        }
                        let (c0, c1) = ((ox - gx).max(0), (ox + TS_I - gx).min(b.w as i32));
                        for col in c0..c1 {
                            let v = b.cov[row as usize * b.w + col as usize];
                            if v > 0 {
                                let d = &mut cov[ly as usize * TS + (gx + col - ox) as usize];
                                *d = d.saturating_add(v);
                            }
                        }
                    }
                }
                if let Some(c) = cur {
                    flush_cov(&mut dst, &mut cov, c, first);
                }
                (i, finish_tile(dst.into_boxed_slice()))
            })
            .collect();
        for (i, t) in done {
            grid.tiles[i] = t;
        }
    }

    fn render_outline(&self, grid: &mut TileGrid, region: RectI) {
        let (tx0, ty0, tx1, ty1) = region.tile_span(grid.cols, grid.rows);
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                let i = grid.idx(tx, ty);
                grid.tiles[i] = None;
            }
        }
        let snapped = RectI::new(tx0 * TS_I, ty0 * TS_I, tx1 * TS_I, ty1 * TS_I);
        // Consecutive same-colour items rasterize together.
        let mut groups: Vec<([u8; 3], Vec<Poly>)> = Vec::new();
        {
            let mut fs = fonts().lock().unwrap();
            let w = self.warper();
            let mut push = |c: [u8; 3], polys: Vec<Poly>| match groups.last_mut() {
                Some(g) if g.0 == c => g.1.extend(polys),
                _ => groups.push((c, polys)),
            };
            for g in &self.layout.glyphs {
                push(g.color, Self::glyph_doc_polys(&mut fs, g, w.as_ref()));
            }
            for d in &self.layout.decos {
                push(d.color, vec![Self::deco_doc_poly(d, w.as_ref())]);
            }
        }
        for (gi, (color, polys)) in groups.iter().enumerate() {
            let c = px::pack(color[0], color[1], color[2], 255);
            let tiles = vector::rasterize(&vector::edges(polys), grid.cols, grid.rows, snapped, FillRule::NonZero);
            let olds: Vec<TileRef> = tiles.iter().map(|(i, _)| grid.tiles[*i].clone()).collect();
            let done: Vec<(usize, TileRef)> = tiles
                .into_par_iter()
                .zip(olds)
                .map(|((i, t), old)| {
                    let src: Box<[u32]> = match t {
                        super::selection::MaskTile::Full => vec![c; TS * TS].into_boxed_slice(),
                        super::selection::MaskTile::Partial(m) => m.iter().map(|&v| px::scale(c, v as u32)).collect(),
                        super::selection::MaskTile::Empty => return (i, old),
                    };
                    match old.filter(|_| gi > 0) {
                        None => (i, finish_tile(src)),
                        Some(o) => (i, finish_tile(src.iter().zip(o.read().iter()).map(|(&s, &d)| px::over(s, d)).collect())),
                    }
                })
                .collect();
            for (i, t) in done {
                grid.tiles[i] = t;
            }
        }
    }

    /// Level-0 rect covering glyphs that differ from `old` (for re-rendering
    /// only what an edit changed).
    pub fn changed_rect(&self, old: &TextLayer) -> RectI {
        // Photoshop's stored glyphs need not match our shaper's bounds. Once
        // text changes, clear the complete imported raster, including glyphs
        // absent from our layout, before drawing the new live text.
        if self != old
            && let Some(source) = &old.imported
        {
            return self
                .bounds()
                .union(&old.bounds())
                .union(&source.raster_bounds);
        }
        if self.needs_outline() || old.needs_outline() || self.box_size != old.box_size || self.layout.decos != old.layout.decos {
            return self.bounds().union(&old.bounds());
        }
        let same = |x: &PlacedGlyph, y: &PlacedGlyph| x.glyph == y.glyph && x.font_id == y.font_id && x.x == y.x && x.y == y.y && x.size == y.size && x.color == y.color && x.synth == y.synth && x.weight == y.weight;
        let (a, b) = (&self.layout.glyphs, &old.layout.glyphs);
        let pre = a.iter().zip(b.iter()).take_while(|(x, y)| same(x, y)).count();
        let suf = a[pre..].iter().rev().zip(b[pre..].iter().rev()).take_while(|(x, y)| same(x, y)).count();
        let mut r = RectI::EMPTY;
        for g in a[pre..a.len() - suf].iter().chain(b[pre..b.len() - suf].iter()) {
            let s = g.size;
            r = r.union(&RectI::new((g.x - s * 0.3) as i32 - 2, (g.y - s * 1.1) as i32 - 2, (g.x + g.advance.max(s * 0.6) + s * 0.4) as i32 + 3, (g.y + s * 0.4) as i32 + 3));
        }
        r
    }

    /// Re-renders `grid` (holding `old`'s pixels) to show `self`, touching
    /// only the tiles under changed glyphs. Returns the tile-snapped rect.
    pub fn rerender_edit(&self, grid: &mut TileGrid, old: &TextLayer) -> RectI {
        let r = self.changed_rect(old).intersect(&grid.bounds());
        if r.is_empty() {
            return r;
        }
        let (tx0, ty0, tx1, ty1) = r.tile_span(grid.cols, grid.rows);
        let snapped = RectI::new(tx0 * TS_I, ty0 * TS_I, tx1 * TS_I, ty1 * TS_I);
        self.render(grid, snapped);
        snapped
    }
}

// ----- caret, hit testing, selection (on-canvas editing) --------------------------------------

fn in_quad(q: P, c: &[P; 4]) -> bool {
    let s: Vec<f32> = (0..4).map(|k| (c[(k + 1) % 4] - c[k]).cross(q - c[k])).collect();
    s.iter().all(|&v| v >= 0.0) || s.iter().all(|&v| v <= 0.0)
}

impl TextLayer {
    fn line_of(&self, i: usize) -> Option<usize> {
        let ls = &self.layout.lines;
        ls.iter()
            .position(|l| i >= l.start && i < l.end)
            .or_else(|| ls.iter().rposition(|l| i >= l.start && i <= l.end))
            .or_else(|| ls.len().checked_sub(1))
    }

    fn empty_line_caret(&self, l: &LineInfo) -> (P, P) {
        if self.vertical { (p(l.x1, l.top), p(l.x0, l.top)) } else { (p(l.x0, l.top), p(l.x0, l.top + l.height)) }
    }

    /// Caret for byte `i` in layout coordinates.
    fn caret_layout(&self, i: usize) -> (P, P) {
        let l = &self.layout;
        let Some(k) = self.line_of(i) else {
            let top = self.origin.1 - if self.box_size.is_some() { 0.0 } else { self.style.size * 0.8 };
            return (p(self.origin.0, top), p(self.origin.0, top + self.style.line_height()));
        };
        let line = &l.lines[k];
        let gs = &l.glyphs[line.g0..line.g1];
        if let Some(g) = gs.iter().find(|g| !g.soft() && g.start >= i) {
            return (g.cell[0], g.cell[1]);
        }
        if let Some(g) = gs.iter().rev().find(|g| !g.soft()) {
            return (g.cell[3], g.cell[2]);
        }
        self.empty_line_caret(line)
    }

    /// Caret for byte `i`: a segment (top, bottom) in document coordinates.
    pub fn caret_seg(&self, i: usize) -> (P, P) {
        let (a, b) = self.caret_layout(i);
        match self.warper() {
            Some(w) => (w.map(a), w.map(b)),
            None => (a, b),
        }
    }

    /// Nearest caret position within line `k` along the flow axis.
    fn hit_line(&self, k: usize, q: P) -> usize {
        let line = &self.layout.lines[k];
        let axis = |v: P| if self.vertical { v.y } else { v.x };
        let mut best = (line.start, f32::MAX);
        for g in self.layout.glyphs[line.g0..line.g1].iter().filter(|g| !g.soft()) {
            for (pos, e) in [(g.start, g.cell[0].lerp(g.cell[1], 0.5)), (g.end, g.cell[3].lerp(g.cell[2], 0.5))] {
                let d = (axis(e) - axis(q)).abs();
                if d < best.1 {
                    best = (pos, d);
                }
            }
        }
        best.0.min(self.text.len())
    }

    /// Byte index nearest to a document point.
    pub fn hit(&self, x: f32, y: f32) -> usize {
        let q = p(x, y);
        let l = &self.layout;
        if l.lines.is_empty() {
            return self.text.len();
        }
        let path_text = self.path.as_ref().is_some_and(|t| !t.area) && !self.vertical;
        if let (None, false) = (self.warper(), path_text) {
            // Pick the line (column) whose band is nearest, then the edge.
            let k = (0..l.lines.len())
                .min_by(|&a, &b| {
                    let c = |li: &LineInfo| if self.vertical { (q.x - li.baseline).abs() } else { (q.y - (li.top + li.height * 0.5)).abs() };
                    c(&l.lines[a]).total_cmp(&c(&l.lines[b]))
                })
                .unwrap();
            return self.hit_line(k, q);
        }
        // Warped or path text: the cell under the point, else the nearest edge.
        let w = self.warper();
        let m = |v: P| w.as_ref().map_or(v, |w| w.map(v));
        let mut best = (self.text.len(), f32::MAX);
        for line in &l.lines {
            let gs = &l.glyphs[line.g0..line.g1];
            if gs.iter().all(|g| g.soft()) {
                let (a, b) = self.empty_line_caret(line);
                let d = vector::dist_to_segment(q, m(a), m(b));
                if d < best.1 {
                    best = (line.start, d);
                }
                continue;
            }
            for g in gs.iter().filter(|g| !g.soft()) {
                let c = g.cell.map(m);
                let (d0, d1) = (vector::dist_to_segment(q, c[0], c[1]), vector::dist_to_segment(q, c[3], c[2]));
                if in_quad(q, &c) {
                    return if d0 <= d1 { g.start } else { g.end };
                }
                for (pos, d) in [(g.start, d0), (g.end, d1)] {
                    if d < best.1 {
                        best = (pos, d);
                    }
                }
            }
        }
        best.0.min(self.text.len())
    }

    /// Whether a document point is on the text: inside its box (paragraph
    /// text), or within `tol` of a glyph cell or an empty line's caret.
    pub fn contains_point(&self, x: f32, y: f32, tol: f32) -> bool {
        let q = p(x, y);
        if let Some((w, h)) = self.box_size {
            let (x0, y0) = self.origin;
            if x >= x0 - tol && x <= x0 + w + tol && y >= y0 - tol && y <= y0 + h + tol {
                return true;
            }
        }
        let w = self.warper();
        let m = |v: P| w.as_ref().map_or(v, |w| w.map(v));
        let near = |c: [P; 4]| in_quad(q, &c) || (0..4).any(|k| vector::dist_to_segment(q, c[k], c[(k + 1) % 4]) <= tol);
        if self.layout.glyphs.iter().any(|g| near(g.cell.map(m))) {
            return true;
        }
        if self.text.is_empty() || self.layout.glyphs.is_empty() {
            let (a, b) = self.caret_seg(0);
            return vector::dist_to_segment(q, a, b) <= tol.max(self.style.size * 0.5);
        }
        false
    }

    /// The type-on-a-path polyline (first subpath, path direction) with its
    /// cumulative lengths; `None` for area type or no path.
    fn path_poly(&self) -> Option<(Vec<P>, Vec<f32>, bool)> {
        let tp = self.path.as_ref().filter(|t| !t.area)?;
        let pl = tp.path.flatten(0.2).into_iter().find(|pl| pl.pts.len() > 1)?;
        let mut pts = pl.pts.clone();
        if pl.closed {
            pts.push(pts[0]);
        }
        let mut cum = vec![0.0f32];
        for w in pts.windows(2) {
            cum.push(cum.last().unwrap() + w[0].dist(w[1]));
        }
        (*cum.last().unwrap() > 0.0).then_some((pts, cum, pl.closed))
    }

    /// Point and unit tangent of the type path at fraction `f` of its length
    /// (wrapping on closed paths).
    pub fn path_at(&self, f: f32) -> Option<(P, P)> {
        let (pts, cum, closed) = self.path_poly()?;
        let total = *cum.last().unwrap();
        let d = if closed { (f * total).rem_euclid(total) } else { (f * total).clamp(0.0, total) };
        let k = cum.partition_point(|&c| c < d).clamp(1, pts.len() - 1);
        let seg = cum[k] - cum[k - 1];
        let t = if seg > 0.0 { (d - cum[k - 1]) / seg } else { 0.0 };
        Some((pts[k - 1].lerp(pts[k], t), (pts[k] - pts[k - 1]).norm()))
    }

    /// Nearest point on the type path to `q`: its fraction, and the signed
    /// distance (positive on the side unflipped glyphs stand on).
    pub fn path_project(&self, q: P) -> Option<(f32, f32)> {
        let (pts, cum, _) = self.path_poly()?;
        let total = *cum.last().unwrap();
        let mut best = (0.0f32, f32::MAX, 0.0f32);
        for k in 1..pts.len() {
            let (a, b) = (pts[k - 1], pts[k]);
            let ab = b - a;
            let t = ((q - a).dot(ab) / ab.dot(ab).max(1e-9)).clamp(0.0, 1.0);
            let c = a + ab * t;
            let d = c.dist(q);
            if d < best.1 {
                let tn = ab.norm();
                let up = p(tn.y, -tn.x);
                best = ((cum[k - 1] + t * ab.len()) / total, d, (q - c).dot(up));
            }
        }
        Some((best.0, best.2))
    }

    /// The caret `delta` lines (columns) away from byte `i`, keeping its
    /// position along the line (Up / Down arrows).
    pub fn line_move(&self, i: usize, delta: i32) -> usize {
        let Some(k) = self.line_of(i) else { return i };
        let t = k as i32 + delta;
        if t < 0 {
            return 0;
        }
        if t as usize >= self.layout.lines.len() {
            return self.text.len();
        }
        let (a, b) = self.caret_layout(i);
        self.hit_line(t as usize, a.lerp(b, 0.5))
    }

    /// Highlight quads (document coordinates) for the byte range `a..b`;
    /// adjacent cells on a line merge.
    pub fn range_quads(&self, a: usize, b: usize) -> Vec<[P; 4]> {
        let (a, b) = (a.min(b), a.max(b));
        let w = self.warper();
        let mut out: Vec<[P; 4]> = Vec::new();
        for g in self.layout.glyphs.iter().filter(|g| !g.soft() && g.start >= a && g.end <= b && g.start < g.end) {
            let c = g.cell;
            if w.is_none() && g.rot == 0.0 {
                if let Some(last) = out.last_mut() {
                    if last[3].dist(c[0]) < 0.01 && last[2].dist(c[1]) < 0.01 {
                        last[3] = c[3];
                        last[2] = c[2];
                        continue;
                    }
                }
            }
            out.push(match &w {
                Some(w) => c.map(|v| w.map(v)),
                None => c,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(text: &str) -> TextLayer {
        TextLayer::new(text, TextStyle { size: 48.0, ..Default::default() }, (100.0, 200.0), None, false)
    }

    fn ink(g: &TileGrid, b: RectI) -> usize {
        (b.y0..b.y1).flat_map(|y| (b.x0..b.x1).map(move |x| (x, y))).filter(|&(x, y)| px::alpha(g.pixel(x, y)) > 128).count()
    }

    #[test]
    fn layout_snapshot() {
        let t = layer("Hello\nWorld");
        let l = &t.layout;
        assert_eq!(l.lines.len(), 2, "two lines");
        assert_eq!(l.glyphs.len(), 10);
        // First baseline hangs on the origin; the second is one line lower.
        assert!((l.lines[0].baseline - 200.0).abs() < 0.5, "{:?}", l.lines[0]);
        assert!((l.lines[1].baseline - l.lines[0].baseline - 48.0 * 1.2).abs() < 0.5);
        assert!((l.glyphs[0].x - 100.0).abs() < 1.0);
        for w in l.glyphs[..5].windows(2) {
            assert!(w[1].x > w[0].x);
        }
        // Tracking widens the line.
        let mut wide = t.clone();
        wide.set_char_style_all(|s| s.tracking = 200.0);
        wide.relayout();
        assert!(wide.layout.lines[0].x1 - wide.layout.lines[0].x0 > l.lines[0].x1 - l.lines[0].x0 + 30.0);
        // Paragraph text wraps inside its box (trailing spaces hang).
        let para = TextLayer::new("one two three four five six seven eight nine ten", TextStyle { size: 30.0, ..Default::default() }, (0.0, 0.0), Some((200.0, 400.0)), false);
        assert!(para.layout.lines.len() >= 3, "{} lines", para.layout.lines.len());
        assert!(para.layout.glyphs.iter().filter(|g| !para.text[g.start..g.end].trim().is_empty()).all(|g| g.x + g.advance <= 201.0));
    }

    #[test]
    fn runs_follow_edits() {
        let mut t = layer("Hello World");
        t.set_char_style(6, 11, |s| s.color = [255, 0, 0]);
        assert_eq!(t.runs.len(), 2);
        // Typing at the end of the red run extends it; typing at the start
        // of the text uses the first run's style.
        t.insert(11, "!", None);
        assert_eq!(t.runs[1], StyleRun { start: 6, end: 12, style: CharStyle { color: [255, 0, 0], ..t.style.char_style() } });
        t.insert(0, ">", None);
        assert_eq!((t.runs[0].start, t.runs[0].end, t.runs[1].start), (0, 7, 7));
        // Deleting across the boundary keeps both styles; deleting a whole
        // run removes it.
        t.delete(5, 9);
        assert_eq!(t.text, ">Hellrld!");
        assert_eq!((t.runs[0].end, t.runs[1].start, t.runs[1].end), (5, 5, 9));
        t.delete(5, 9);
        assert_eq!(t.runs.len(), 1);
        // A newline adds a paragraph with the split paragraph's style.
        t.set_para_style(0, 0, |p| p.align = TextAlign::Center);
        t.insert(2, "\n", None);
        assert_eq!(t.paras.len(), 2);
        assert_eq!(t.paras[1].align, TextAlign::Center);
        t.delete(2, 3);
        assert_eq!(t.paras.len(), 1);
        // Restyling back to the default merges runs.
        let mut u = layer("abcdef");
        u.set_char_style(2, 4, |s| s.size = 60.0);
        assert_eq!(u.runs.len(), 3);
        u.set_char_style(2, 4, |s| s.size = 48.0);
        assert_eq!(u.runs.len(), 1);
        assert_eq!(u.word_at(3), (0, 6));
    }

    #[test]
    fn styled_runs_render_their_colours_and_sizes() {
        let mut t = layer("AAAA BBBB");
        t.set_char_style(5, 9, |s| {
            s.color = [255, 0, 0];
            s.size = 96.0;
        });
        t.relayout();
        let big = t.layout.glyphs.iter().find(|g| g.start == 5).unwrap();
        let small = t.layout.glyphs[0];
        assert!((big.size - 96.0).abs() < 0.01 && (small.size - 48.0).abs() < 0.01);
        // The larger run raises the line's leading.
        assert!((t.layout.lines[0].height - 96.0 * 1.2).abs() < 0.01);
        let mut g = TileGrid::new(1024, 512);
        let b = t.bounds();
        t.render(&mut g, b);
        let at = |x0: f32, x1: f32| {
            let mut c = [0u64; 3];
            for y in 100..260 {
                for x in x0 as i32..x1 as i32 {
                    let [r, gg, bb, a] = px::unpack(g.pixel(x, y));
                    if a > 200 {
                        c[0] += r as u64;
                        c[1] += gg as u64;
                        c[2] += bb as u64;
                    }
                }
            }
            c
        };
        let left = at(100.0, t.layout.glyphs[3].x + 20.0);
        let right = at(big.x, big.x + 200.0);
        assert!(left[0] == 0 && left[1] == 0, "{left:?}");
        assert!(right[0] > 1000 && right[1] == 0, "{right:?}");
    }

    #[test]
    fn decorations_caps_and_positions() {
        let mut t = layer("Hello x2");
        t.set_char_style(0, 5, |s| {
            s.underline = true;
            s.caps = Caps::All;
        });
        t.set_char_style(7, 8, |s| s.position = Position::Super);
        t.relayout();
        assert_eq!(t.layout.decos.len(), 1);
        let d = t.layout.decos[0];
        // The underline sits below the baseline across "HELLO".
        assert!(d.quad[0].y > 200.0 && d.quad[0].y < 215.0, "{d:?}");
        assert!((d.quad[0].x - 100.0).abs() < 1.0 && d.quad[3].x > 200.0);
        // All caps shapes capitals but keeps the source offsets.
        let caps = t.layout.glyphs[1];
        assert_eq!((caps.start, caps.end), (1, 2));
        let plain = layer("HELLO");
        assert_eq!(caps.glyph, plain.layout.glyphs[1].glyph);
        // Superscript: smaller and raised.
        let sup = t.layout.glyphs.iter().find(|g| g.start == 7).unwrap();
        assert!((sup.size - 48.0 * 0.583).abs() < 0.01 && sup.y < 200.0 - 10.0, "{sup:?}");
        // Faux bold inks more than regular.
        let mut a = layer("Faux");
        let mut b = a.clone();
        b.set_char_style_all(|s| s.faux_bold = true);
        b.relayout();
        a.relayout();
        let (mut ga, mut gb) = (TileGrid::new(512, 512), TileGrid::new(512, 512));
        let (ba, bb) = (a.bounds(), b.bounds());
        a.render(&mut ga, ba);
        b.render(&mut gb, bb);
        assert!(ink(&gb, b.bounds()) as f32 > ink(&ga, a.bounds()) as f32 * 1.08);
    }

    #[test]
    fn paragraph_justify_indents_and_hyphenation() {
        let text = "The quick brown fox jumps over the lazy dog and keeps running far away";
        let st = TextStyle { size: 24.0, align: TextAlign::Justify, ..Default::default() };
        let mut t = TextLayer::new(text, st, (0.0, 0.0), Some((300.0, 600.0)), false);
        let l = &t.layout;
        assert!(l.lines.len() >= 3);
        // Every line but the last fills the box exactly; the last is left-set.
        for li in &l.lines[..l.lines.len() - 1] {
            let last = l.glyphs[li.g0..li.g1].iter().rev().find(|g| !t.text[g.start..g.end].trim().is_empty()).unwrap();
            assert!((last.cell[2].x - 300.0).abs() < 0.5, "{li:?} ends at {}", last.cell[2].x);
        }
        let ll = l.lines.last().unwrap();
        assert!(ll.x1 < 299.0);
        // Justify All sets the last line full too; Last Right ends at the edge.
        t.set_para_style(0, 0, |p| p.last = LastLine::Right);
        t.relayout();
        let ll = *t.layout.lines.last().unwrap();
        let end = t.layout.glyphs[ll.g0..ll.g1].iter().rev().find(|g| !t.text[g.start..g.end].trim().is_empty()).unwrap();
        assert!((end.cell[2].x - 300.0).abs() < 0.5);
        // First-line and left indents.
        t.set_para_style(0, 0, |p| {
            p.align = TextAlign::Left;
            p.indent_left = 20.0;
            p.indent_first = 30.0;
        });
        t.relayout();
        assert!((t.layout.lines[0].x0 - 50.0).abs() < 0.5 && (t.layout.lines[1].x0 - 20.0).abs() < 0.5);
        // Hyphenation splits a long word at the line end with a hyphen.
        let mut h = TextLayer::new("internationalization internationalization", TextStyle { size: 30.0, ..Default::default() }, (0.0, 0.0), Some((160.0, 600.0)), false);
        let plain_lines = h.layout.lines.len();
        h.set_para_style(0, 0, |p| p.hyphenate = true);
        h.relayout();
        assert!(h.layout.glyphs.iter().any(|g| g.synth & SYNTH_HYPHEN != 0), "no hyphen");
        assert!(h.layout.lines.len() <= plain_lines);
        // Each hyphenated line still fits the box, hyphen included.
        assert!(h.layout.glyphs.iter().all(|g| g.cell[2].x <= 160.5 || h.text[g.start..g.end].trim().is_empty()));
        // Break points follow syllables: never "counte-rrev…" or "counterr-ev…".
        for w in (140..=420).step_by(20) {
            let mut h = TextLayer::new("counterrevolutionaries", TextStyle { size: 30.0, ..Default::default() }, (0.0, 0.0), Some((w as f32, 900.0)), false);
            h.set_para_style(0, 0, |p| p.hyphenate = true);
            h.relayout();
            for g in h.layout.glyphs.iter().filter(|g| g.synth & SYNTH_HYPHEN != 0) {
                let pre = &h.text[..g.start];
                assert!(!["counte", "counterr", "c", "co", "cou"].contains(&pre), "bad break {pre}- at width {w}");
            }
        }
        // Space before/after separate paragraphs.
        let mut s = layer("a\nb");
        let gap0 = s.layout.lines[1].baseline - s.layout.lines[0].baseline;
        s.set_para_style(0, 0, |p| p.space_after = 10.0);
        s.set_para_style(2, 2, |p| p.space_before = 5.0);
        s.relayout();
        assert!((s.layout.lines[1].baseline - s.layout.lines[0].baseline - gap0 - 15.0).abs() < 0.01);
    }

    #[test]
    fn every_line_composer_evens_lines_and_uses_liang_points() {
        let text = "In olden times when wishing still helped one, there lived a king whose daughters were all beautiful, but the youngest was so beautiful that the sun itself, which has seen so much, was astonished whenever it shone in her face. Close by the king's castle lay a great dark forest, and under an old lime-tree in the forest was a well.";
        // Slack² of every line but the last (natural width from the cells).
        let slack = |t: &TextLayer, w: f32| -> f32 {
            let l = &t.layout;
            l.lines[..l.lines.len() - 1]
                .iter()
                .map(|li| {
                    let gs: Vec<_> = l.glyphs[li.g0..li.g1].iter().filter(|g| !t.text[g.start..g.end].trim().is_empty() || g.synth & SYNTH_HYPHEN != 0).collect();
                    let nat = gs.last().map_or(0.0, |g| g.cell[2].x) - li.x0;
                    (w - nat).powi(2)
                })
                .sum()
        };
        for w in [260.0f32, 340.0, 420.0] {
            for hyph in [false, true] {
                let mut t = TextLayer::new(text, TextStyle { size: 24.0, ..Default::default() }, (0.0, 0.0), Some((w, 3000.0)), false);
                t.set_para_style(0, 0, |p| {
                    p.hyphenate = hyph;
                    p.composer = Composer::Single;
                });
                t.relayout();
                let hyphens = |t: &TextLayer| t.layout.glyphs.iter().filter(|g| g.synth & SYNTH_HYPHEN != 0).count();
                let (greedy, greedy_hy) = (slack(&t, w), hyphens(&t));
                t.set_para_style(0, 0, |p| p.composer = Composer::Every);
                t.relayout();
                let (every, every_hy) = (slack(&t, w), hyphens(&t));
                if hyph {
                    // Hyphens cost a penalty: the composer uses no more of them.
                    assert!(every_hy <= greedy_hy, "width {w}: {every_hy} hyphens vs {greedy_hy}");
                } else {
                    assert!(every <= greedy + 1.0, "width {w}: every-line {every:.0} vs single-line {greedy:.0}");
                }
                // No line overflows the measure.
                assert!(t.layout.glyphs.iter().filter(|g| !t.text[g.start..g.end].trim().is_empty()).all(|g| g.cell[2].x <= w + 0.5));
                // Hyphens sit only at Liang break points of their word.
                for g in t.layout.glyphs.iter().filter(|g| g.synth & SYNTH_HYPHEN != 0) {
                    let ws = t.text[..g.start].rfind(|c: char| !c.is_alphabetic()).map_or(0, |i| i + 1);
                    let we = t.text[g.start..].find(|c: char| !c.is_alphabetic()).map_or(t.text.len(), |i| g.start + i);
                    let word = &t.text[ws..we];
                    let at = t.text[ws..g.start].chars().count();
                    assert!(crate::engine::hyphen::breaks(word).unwrap_or_default().contains(&at), "{word} broken at {at}");
                }
            }
        }
        // Justified: the composer may shrink spaces (≥ 80 %), never below.
        let mut j = TextLayer::new(text, TextStyle { size: 24.0, align: TextAlign::Justify, ..Default::default() }, (0.0, 0.0), Some((300.0, 3000.0)), false);
        j.relayout();
        for li in &j.layout.lines[..j.layout.lines.len() - 1] {
            let gs = &j.layout.glyphs[li.g0..li.g1];
            let last = gs.iter().rev().find(|g| !j.text[g.start..g.end].trim().is_empty()).unwrap();
            assert!((last.cell[2].x - 300.0).abs() < 0.5, "justified line ends at {}", last.cell[2].x);
        }
    }

    #[test]
    fn opentype_features_change_shaping() {
        // Fonts that ship with macOS; elsewhere the test only checks that
        // the features shape without changing the source mapping.
        let ids = |t: &TextLayer| t.layout.glyphs.iter().map(|g| g.glyph).collect::<Vec<_>>();
        let have = |f: &str| families().iter().any(|x| x == f);
        let st = TextStyle { family: "Baskerville".into(), size: 48.0, ..Default::default() };
        let base = TextLayer::new("Quick fjords sty ct 0123456789", st, (0.0, 100.0), None, false);
        for (name, f) in [("onum", (|o: &mut OtFeatures| o.onum = true) as fn(&mut OtFeatures)), ("dlig", |o| o.dlig = true)] {
            let mut t = base.clone();
            t.set_char_style_all(|s| f(&mut s.features));
            t.relayout();
            assert!(t.layout.glyphs.iter().all(|g| g.end <= t.text.len()));
            if have("Baskerville") {
                assert_ne!(ids(&t), ids(&base), "{name} changed nothing");
            }
        }
        // Fractions and standard ligatures (Seravek).
        let st = TextStyle { family: "Seravek".into(), size: 48.0, ..Default::default() };
        let mut t = TextLayer::new("office 1/2", st, (0.0, 100.0), None, false);
        let lig = ids(&t);
        t.set_char_style_all(|s| s.features.liga = false);
        t.relayout();
        let nolig = ids(&t);
        t.set_char_style_all(|s| {
            s.features.liga = true;
            s.features.frac = true;
        });
        t.relayout();
        if have("Seravek") {
            assert!(nolig.len() > lig.len(), "ffi ligature: {lig:?} vs {nolig:?}");
            assert_ne!(ids(&t), lig, "frac changed nothing");
        }
    }

    #[test]
    fn renders_coverage_and_caches_glyphs() {
        let t = layer("Ag");
        let mut g = TileGrid::new(1024, 512);
        let b = t.bounds();
        t.render(&mut g, b);
        let n = ink(&g, b);
        assert!(n > 300, "only {n} inked pixels");
        assert!(g.tiles[g.idx(3, 1)].is_none(), "far tile touched");
        // Type mask outlines cover the same area as the rendered glyphs.
        let m = vector::rasterize_mask(&t.outlines(), 1024, 512, FillRule::NonZero);
        let mask_ink = (b.y0..b.y1).flat_map(|y| (b.x0..b.x1).map(move |x| (x, y))).filter(|&(x, y)| m.at(x, y) > 128).count();
        assert!((mask_ink as i32 - n as i32).abs() < n as i32 / 20, "{mask_ink} vs {n}");
        // Editing re-renders only the changed glyphs' area.
        let mut t2 = t.clone();
        t2.insert(2, "x", None);
        t2.relayout();
        let r = t2.changed_rect(&t);
        assert!(r.x0 > t.layout.glyphs[1].x as i32 - 60, "{r:?}");
        // Every warp style renders, inside its bounds.
        for (style, bend, hd, vd) in WarpStyle::ALL.iter().flat_map(|&st| [(st, 0.5, 0.2, -0.1), (st, 1.0, 0.8, 0.8), (st, -1.0, -0.8, 0.8)]) {
            let mut w = t.clone();
            w.warp = Warp { style, bend, h_distort: hd, v_distort: vd, vertical: false };
            let mut g2 = TileGrid::new(1024, 512);
            let b2 = w.bounds();
            let full = g2.bounds();
            w.render(&mut g2, full);
            let all = ink(&g2, g2.bounds());
            assert!(all > 200, "{style:?} {bend} {hd} {vd}");
            assert_eq!(ink(&g2, b2), all, "{style:?} {bend} {hd} {vd}: ink outside bounds");
        }
    }

    #[test]
    fn caret_and_hit_roundtrip() {
        let mut t = layer("Hello wide world\nsecond");
        for i in [0usize, 2, 5, 16, 17, 20, 23] {
            let (a, b) = t.caret_seg(i);
            let m = a.lerp(b, 0.5);
            assert_eq!(t.hit(m.x + 0.1, m.y), i, "byte {i}");
        }
        // Up / down keep the horizontal position.
        assert_eq!(t.line_move(2, 1), 19);
        assert_eq!(t.line_move(19, -1), 2);
        // Range highlight merges into one quad per line.
        assert_eq!(t.range_quads(0, 20).len(), 2);
        // Warped text: carets follow the warp and still round-trip.
        t.warp = Warp { style: WarpStyle::Arc, ..Default::default() };
        for i in [1usize, 8, 19] {
            let (a, b) = t.caret_seg(i);
            let m = a.lerp(b, 0.5);
            assert_eq!(t.hit(m.x + 0.5, m.y), i, "warped byte {i}");
        }
    }

    #[test]
    fn vertical_type_rotates_roman_and_keeps_cjk_upright() {
        let mut t = TextLayer::new("日本Ab12", TextStyle { size: 40.0, ..Default::default() }, (500.0, 100.0), None, true);
        let g = &t.layout.glyphs;
        assert_eq!(g[0].rot, 0.0);
        assert!(g[2].rot > 1.5, "Latin rotates");
        // Glyphs descend the column.
        assert!(g[1].cell[0].y > g[0].cell[0].y && g[3].cell[0].y > g[2].cell[0].y);
        // Vertical punctuation uses the font's vertical alternates ('vert').
        let v = TextLayer::new("あ。", TextStyle { size: 40.0, ..Default::default() }, (0.0, 0.0), None, true);
        let h = TextLayer::new("あ。", TextStyle { size: 40.0, ..Default::default() }, (0.0, 0.0), None, false);
        if v.layout.glyphs[1].font_id == h.layout.glyphs[1].font_id && cfg!(target_os = "macos") {
            assert_ne!(v.layout.glyphs[1].glyph, h.layout.glyphs[1].glyph, "no vertical alternate for 。");
        }
        // Standard Vertical Roman Alignment keeps Latin upright.
        t.upright_roman = true;
        t.relayout();
        assert!(t.layout.glyphs.iter().all(|g| g.rot == 0.0));
        // Tate-chu-yoko: "12" sits side by side across the column.
        t.upright_roman = false;
        t.tcy = vec![(8, 10)];
        t.relayout();
        let (a, b) = (t.layout.glyphs[4], t.layout.glyphs[5]);
        assert_eq!((a.rot, b.rot), (0.0, 0.0));
        assert!((a.y - b.y).abs() < 0.01 && b.x > a.x);
        // Renders (outline path for the rotated glyphs).
        let mut grid = TileGrid::new(1024, 512);
        let full = grid.bounds();
        t.render(&mut grid, full);
        let tb = t.bounds();
        assert!(ink(&grid, tb) > 200);
        assert_eq!(ink(&grid, tb), ink(&grid, full), "vertical ink outside bounds");
        // Caret round-trip along the column.
        for i in [0usize, 3, 7] {
            let (a, b) = t.caret_seg(i);
            let m = a.lerp(b, 0.5);
            assert_eq!(t.hit(m.x, m.y + 0.1), i, "byte {i}");
        }
    }

    #[test]
    fn type_on_path_and_area_type() {
        let path = vector::Path::ellipse(400.0, 400.0, 300.0, 300.0);
        let mut t = TextLayer::new("Around the circle we go", TextStyle { size: 40.0, ..Default::default() }, (0.0, 0.0), None, false);
        t.path = Some(Arc::new(TextPath { path: path.clone(), start: 0.0, end: 1.0, flip: false, area: false }));
        t.relayout();
        // Every pen sits on the circle (baseline on the path).
        for g in t.layout.glyphs.iter().filter(|g| g.advance > 0.0) {
            let r = ((g.x - 400.0).powi(2) + (g.y - 400.0).powi(2)).sqrt();
            assert!((r - 300.0).abs() < 2.0, "pen radius {r}");
        }
        assert!(t.layout.glyphs.windows(2).any(|w| (w[0].rot - w[1].rot).abs() > 0.01), "glyphs turn with the path");
        // Flip puts the glyphs on the other side: their tops face the centre.
        let up_dist = |t: &TextLayer| {
            let g = t.layout.glyphs[3];
            let top = g.cell[0];
            ((top.x - 400.0).powi(2) + (top.y - 400.0).powi(2)).sqrt()
        };
        let out = up_dist(&t);
        t.path = Some(Arc::new(TextPath { path: path.clone(), start: 0.0, end: 1.0, flip: true, area: false }));
        t.relayout();
        assert!(up_dist(&t) < out - 30.0);
        // Markers: path_at / path_project round-trip, and the glyph side is
        // positive (outside the circle for unflipped text on Path::ellipse).
        t.path = Some(Arc::new(TextPath { path: path.clone(), start: 0.25, end: 1.25, flip: false, area: false }));
        t.relayout();
        let (q, tn) = t.path_at(0.25).unwrap();
        assert!(((q.x - 400.0).powi(2) + (q.y - 400.0).powi(2)).sqrt() - 300.0 < 1.0 && (tn.len() - 1.0).abs() < 1e-3);
        let (f, side) = t.path_project(q).unwrap();
        assert!((f - 0.25).abs() < 1e-3 && side.abs() < 0.5, "{f} {side}");
        let g = t.layout.glyphs[0];
        let top = g.cell[0];
        assert!(t.path_project(top).unwrap().1 > 10.0, "glyph tops on the positive side");
        // An end marker short of the text hides the rest (overflow).
        t.path = Some(Arc::new(TextPath { path: path.clone(), start: 0.0, end: 0.05, flip: false, area: false }));
        t.relayout();
        assert!(t.layout.overflow && t.layout.glyphs.len() < 10, "{} glyphs", t.layout.glyphs.len());
        // Path text ink stays inside its bounds (flipped and not).
        for flip in [false, true] {
            t.path = Some(Arc::new(TextPath { path: path.clone(), start: 0.1, end: 1.1, flip, area: false }));
            t.relayout();
            let mut g = TileGrid::new(1024, 1024);
            let full = g.bounds();
            t.render(&mut g, full);
            assert_eq!(ink(&g, t.bounds()), ink(&g, full), "path text (flip {flip}) ink outside bounds");
        }
        // Area type: lines stay inside the circle.
        let mut a = TextLayer::new("lorem ipsum dolor sit amet ".repeat(12), TextStyle { size: 24.0, ..Default::default() }, (0.0, 0.0), None, false);
        a.path = Some(Arc::new(TextPath { path, start: 0.0, end: 1.0, flip: false, area: true }));
        a.relayout();
        assert!(a.layout.lines.len() > 6);
        for g in a.layout.glyphs.iter().filter(|g| !a.text[g.start..g.end].trim().is_empty()) {
            for q in [p(g.x, g.y), p(g.x + g.advance, g.y)] {
                let r = ((q.x - 400.0).powi(2) + (q.y - 400.0).powi(2)).sqrt();
                assert!(r < 305.0, "glyph at {q:?} outside the circle (r {r})");
            }
        }
    }

    #[test]
    fn incremental_edit_matches_full_render() {
        let style = TextStyle { size: 28.0, ..Default::default() };
        let text = "Pack my box with five dozen liquor jugs. ".repeat(12);
        let mut t = TextLayer::new(text, style, (40.0, 40.0), Some((700.0, 900.0)), false);
        t.set_char_style(50, 90, |s| s.color = [200, 30, 30]);
        t.set_char_style(70, 120, |s| s.size = 34.0);
        t.relayout();
        let mut g = TileGrid::new(1024, 1024);
        let b = t.bounds();
        t.render(&mut g, b);
        for (k, edit) in ["insert", "delete", "newline", "insert", "style"].iter().enumerate() {
            let mut n = t.clone();
            let at = 100 + k * 37;
            match *edit {
                "insert" => n.insert(at, "WW", None),
                "delete" => n.delete(at, at + 3),
                "newline" => n.insert(at, "\n", None),
                _ => n.set_char_style(at, at + 10, |s| s.color = [0, 0, 255]),
            }
            n.relayout();
            n.rerender_edit(&mut g, &t);
            t = n;
            let mut full = TileGrid::new(1024, 1024);
            let fb = full.bounds();
            t.render(&mut full, fb);
            for y in (0..1024).step_by(3) {
                for x in 0..1024 {
                    assert_eq!(g.pixel(x, y), full.pixel(x, y), "{edit}: pixel ({x}, {y})");
                }
            }
        }
    }
}
