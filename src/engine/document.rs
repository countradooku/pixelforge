//! Layers, the document, and undo history.
//!
//! History entries are full document snapshots, but a snapshot is only a
//! vector of `Arc` tile pointers per layer, so each step costs memory only for
//! the tiles that step actually changed.

use std::path::PathBuf;

use super::blend::BlendMode;
use super::composite::Preview;
use super::layers::{LayerKind, LayerMask};
use super::grid::{RectI, TileGrid, same_tile};
use super::mips::{build_mips, diff_rects, mip_count, update_mips};
use super::selection::Selection;

const MAX_HISTORY: usize = 80;
const TILE_BYTES: usize = super::grid::TPX * 4;

#[derive(Clone)]
pub struct Layer {
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    pub grid: TileGrid,
    pub mips: Vec<TileGrid>,
    /// Uncommitted translation while the move tool drags (level-0 pixels).
    pub offset: (i32, i32),
    /// Bumped on every pixel change (drives thumbnail refresh).
    pub version: u64,
    pub kind: LayerKind,
    pub mask: Option<LayerMask>,
    /// Clipped to the nearest non-clipped layer below (clipping mask).
    pub clip: bool,
    /// Id of the group this layer belongs to (children sit directly below it).
    pub parent: Option<u64>,
    /// Dialog settings an adjustment layer was built from (for re-editing).
    pub adj_params: Option<std::sync::Arc<(super::filters::catalog::Kind, Vec<f32>, bool)>>,
    /// Photoshop's locked Background: opaque, bottom-most, can't move or be
    /// reordered; erasing paints the background colour.
    pub background: bool,
    /// Pixels beyond the canvas (crop with "Delete Cropped Pixels" off).
    pub hidden: Option<std::sync::Arc<super::crop::Hidden>>,
    /// Layer Style (effects); `None` or all effects off = no effects.
    pub style: Option<std::sync::Arc<super::styles::LayerStyle>>,
    /// Fill opacity (0..1): scales the layer's own pixels, not its effects.
    pub fill: f32,
    /// Blend If ranges and RGB channel restrictions; absent is the fast path.
    pub advanced_blend: Option<std::sync::Arc<super::advanced_blend::AdvancedBlend>>,
}

impl Layer {
    pub fn new(id: u64, name: impl Into<String>, grid: TileGrid) -> Self {
        let mips = build_mips(&grid);
        Self::from_mips(id, name, grid, mips)
    }

    fn from_mips(id: u64, name: impl Into<String>, grid: TileGrid, mips: Vec<TileGrid>) -> Self {
        Layer {
            id,
            name: name.into(),
            visible: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
            grid,
            mips,
            offset: (0, 0),
            version: 0,
            kind: LayerKind::Pixel,
            mask: None,
            clip: false,
            parent: None,
            adj_params: None,
            style: None,
            fill: 1.0,
            advanced_blend: None,
            background: false,
            hidden: None,
        }
    }

    pub fn with_kind(mut self, kind: LayerKind) -> Self {
        if kind.is_vector() {
            let region = self.grid.bounds().intersect(&kind.content_bounds().unwrap_or(RectI::EMPTY));
            kind.rasterize(&mut self.grid, region);
            self.mips = build_mips(&self.grid);
        }
        self.kind = kind;
        self
    }

    /// Whether the compositor can treat this as a plain pixel layer.
    pub fn is_plain(&self) -> bool {
        matches!(self.kind, LayerKind::Pixel) && self.mask.is_none() && !self.clip && self.parent.is_none() && !super::styles::is_styled(self)
    }

    pub fn level(&self, l: u32) -> &TileGrid {
        if l == 0 || self.mips.is_empty() {
            &self.grid
        } else {
            &self.mips[(l as usize).min(self.mips.len()) - 1]
        }
    }

    /// Swaps in a new level-0 grid of the same size, updating only the mip
    /// tiles that changed. Returns the changed level-0 rects.
    pub fn replace_grid(&mut self, grid: TileGrid) -> Vec<RectI> {
        if grid.w != self.grid.w || grid.h != self.grid.h {
            self.grid = grid;
            self.mips = build_mips(&self.grid);
            self.version += 1;
            return vec![self.grid.bounds()];
        }
        let rects = diff_rects(&self.grid, &grid);
        self.grid = grid;
        update_mips(&self.grid, &mut self.mips, &rects);
        if !rects.is_empty() {
            self.version += 1;
        }
        rects
    }
}

#[derive(Clone)]
struct LayerState {
    id: u64,
    name: String,
    visible: bool,
    opacity: f32,
    blend: BlendMode,
    grid: TileGrid,
    mips: Vec<TileGrid>,
    meta: LayerMeta,
}

/// Non-pixel layer state. Small (vector descriptions are `Arc`s and mask
/// tiles are shared), so history keeps it uncompressed.
#[derive(Clone)]
struct LayerMeta {
    kind: LayerKind,
    mask: Option<LayerMask>,
    clip: bool,
    parent: Option<u64>,
    adj_params: Option<std::sync::Arc<(super::filters::catalog::Kind, Vec<f32>, bool)>>,
    background: bool,
    hidden: Option<std::sync::Arc<super::crop::Hidden>>,
    style: Option<std::sync::Arc<super::styles::LayerStyle>>,
    fill: f32,
    advanced_blend: Option<std::sync::Arc<super::advanced_blend::AdvancedBlend>>,
}

impl LayerMeta {
    fn of(l: &Layer) -> Self {
        LayerMeta {
            kind: l.kind.clone(),
            mask: l.mask.clone(),
            clip: l.clip,
            parent: l.parent,
            adj_params: l.adj_params.clone(),
            background: l.background,
            hidden: l.hidden.clone(),
            style: l.style.clone(),
            fill: l.fill,
            advanced_blend: l.advanced_blend.clone(),
        }
    }

    fn apply(self, l: &mut Layer) {
        l.kind = self.kind;
        l.mask = self.mask;
        l.clip = self.clip;
        l.parent = self.parent;
        l.adj_params = self.adj_params;
        l.background = self.background;
        l.hidden = self.hidden;
        l.style = self.style;
        l.fill = self.fill;
        l.advanced_blend = self.advanced_blend;
    }
}

#[derive(Clone)]
struct DocState {
    w: u32,
    h: u32,
    layers: Vec<LayerState>,
    active: usize,
    selected: Vec<u64>,
    selection: Option<Selection>,
    paths: Vec<PathItem>,
    ann: super::annot::Annotations,
}

/// A saved path in the Paths panel ("Work Path" is the unsaved one).
#[derive(Clone, Debug, PartialEq)]
pub struct PathItem {
    pub name: String,
    pub path: super::vector::Path,
}

/// Snapshots share the same managed tile objects as the document. Compression
/// never duplicates a tile across history steps and undo does not unpack a grid.
struct HistoryEntry { label: String, state: DocState }
impl HistoryEntry {
    fn new(label: String, state: DocState) -> Self { Self { label, state } }
    fn take_state(self) -> DocState { self.state }
}
#[derive(Clone)]
pub struct HistorySource { pub w: u32, pub h: u32, pub cols: u32, pub tiles: Vec<super::grid::TileRef> }
impl HistorySource {
    pub fn tile(&self, i: usize) -> super::grid::TileRef { self.tiles.get(i)?.clone() }
    #[cfg(test)] pub fn packed_tiles(&self) -> usize { self.tiles.iter().flatten().filter(|t| t.is_cold()).count() }
}

#[derive(Default)]
pub struct Dirty { pub all: bool, pub rects: Vec<RectI> }

pub struct Document {
    pub w: u32,
    pub h: u32,
    pub layers: Vec<Layer>,
    /// The active (key) layer.
    pub active: usize,
    /// Further selected layers (ids) for multi-layer Move / Align /
    /// Distribute; the active layer is always selected too.
    pub selected: Vec<u64>,
    pub selection: Option<Selection>,
    pub preview: Option<Preview>,
    /// Paths panel (document-level vector paths).
    pub paths: Vec<PathItem>,
    pub active_path: Option<usize>,
    /// Color samplers, notes, counts, artboards and slices.
    pub ann: super::annot::Annotations,
    pub dirty: Dirty,
    pub path: Option<PathBuf>,
    /// Original PSD blocks and channel bytes, shared across edits and history.
    pub psd: Option<std::sync::Arc<super::psd::mapping::Backing>>,
    pub modified: bool,
    /// Bumped whenever the canvas size changes.
    pub generation: u64,
    undo: Vec<HistoryEntry>,
    redo: Vec<HistoryEntry>,
    next_id: u64,
}

impl Document {
    pub fn new(w: u32, h: u32, background: Option<u32>) -> Self {
        let grid = match background {
            Some(c) => TileGrid::filled(w, h, c),
            None => TileGrid::new(w, h),
        };
        Self::from_grid(grid, if background.is_some() { "Background" } else { "Layer 1" })
    }

    pub fn from_grid(grid: TileGrid, name: &str) -> Self {
        let (w, h) = (grid.w, grid.h);
        Document {
            w,
            h,
            layers: vec![{
                let mut l = Layer::new(1, name, grid);
                // An opaque "Background" is Photoshop's locked Background layer.
                l.background = name == "Background";
                l
            }],
            active: 0,
            selected: vec![],
            selection: None,
            preview: None,
            paths: vec![],
            active_path: None,
            ann: Default::default(),
            dirty: Dirty { all: true, rects: vec![] },
            path: None,
            psd: None,
            modified: false,
            generation: 0,
            undo: vec![],
            redo: vec![],
            next_id: 2,
        }
    }

    /// Rebuilds a document from loaded layers (native format).
    pub fn from_layers(w: u32, h: u32, layers: Vec<Layer>, active: usize) -> Self {
        let next_id = layers.iter().map(|l| l.id).max().unwrap_or(0) + 1;
        Document {
            w,
            h,
            active: active.min(layers.len().saturating_sub(1)),
            layers,
            selected: vec![],
            selection: None,
            preview: None,
            paths: vec![],
            active_path: None,
            ann: Default::default(),
            dirty: Dirty { all: true, rects: vec![] },
            path: None,
            psd: None,
            modified: false,
            generation: 0,
            undo: vec![],
            redo: vec![],
            next_id,
        }
    }

    pub fn bounds(&self) -> RectI {
        RectI::new(0, 0, self.w as i32, self.h as i32)
    }

    pub fn max_level(&self) -> u32 {
        mip_count(self.w, self.h)
    }

    pub fn next_layer_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }



    // ----- dirty tracking -------------------------------------------------

    pub fn mark_all_dirty(&mut self) {
        self.dirty.all = true;
    }

    /// Records that `rects` of layer `li` changed: updates its mips and
    /// schedules the display tiles over them.
    pub fn layer_pixels_changed(&mut self, li: usize, rects: &[RectI]) {
        if rects.is_empty() {
            return;
        }
        let layer = &mut self.layers[li];
        update_mips(&layer.grid, &mut layer.mips, rects);
        layer.version += 1;
        self.dirty.rects.extend_from_slice(rects);
        self.modified = true;
    }

    /// Replaces a layer's pixels and propagates the minimal dirty area.
    pub fn set_layer_grid(&mut self, li: usize, grid: TileGrid) {
        let rects = self.layers[li].replace_grid(grid);
        self.dirty.rects.extend(rects);
        self.modified = true;
    }

    pub fn take_dirty(&mut self) -> Dirty {
        let mut d = std::mem::take(&mut self.dirty);
        // Layer effects reach beyond the changed pixels.
        if !d.all && !d.rects.is_empty() {
            let pad = self.layers.iter().map(super::styles::dirty_pad).max().unwrap_or(0);
            if pad > 0 {
                d.rects.iter_mut().for_each(|r| *r = r.inflate(pad));
            }
        }
        d
    }

    // ----- history --------------------------------------------------------

    fn capture(&self) -> DocState {
        DocState {
            w: self.w,
            h: self.h,
            layers: self
                .layers
                .iter()
                .map(|l| LayerState {
                    id: l.id,
                    name: l.name.clone(),
                    visible: l.visible,
                    opacity: l.opacity,
                    blend: l.blend,
                    grid: l.grid.clone(),
                    mips: l.mips.clone(),
                    meta: LayerMeta::of(l),
                })
                .collect(),
            active: self.active,
            selected: self.selected.clone(),
            selection: self.selection.clone(),
            paths: self.paths.clone(),
            ann: self.ann.clone(),
        }
    }

    pub fn push_undo(&mut self, label: impl Into<String>) {
        let state = self.capture();
        self.undo.push(HistoryEntry::new(label.into(), state));
        if self.undo.len() > MAX_HISTORY {
            self.undo.remove(0);
        }
        self.redo.clear();
        self.modified = true;
        self.maintain_history();
    }

    /// Hint old history and hidden layers to the background reclaimer. Recent
    /// visible pixels stay hot unless the global budget is under pressure.
    pub fn maintain_history(&self) {
        // A hidden folder also makes all descendants cold. Bound the walk so
        // malformed imported parent chains cannot loop forever.
        let visible_layer = |layer: &Layer| {
            let mut current = layer;
            for _ in 0..=self.layers.len() {
                if !current.visible { return false; }
                let Some(id) = current.parent else { return true };
                let Some(parent) = self.layers.iter().find(|l| l.id == id) else { return true };
                current = parent;
            }
            false
        };
        let mut keep = std::collections::HashSet::new();
        for l in self.layers.iter().filter(|l| visible_layer(l)) {
            for t in std::iter::once(&l.grid).chain(&l.mips).flat_map(|g| g.tiles.iter().flatten()) { keep.insert(std::sync::Arc::as_ptr(t) as usize); }
        }
        let visible = keep.clone();
        for e in self.undo.iter().rev().take(2).chain(self.redo.iter().rev().take(2)) {
            for l in &e.state.layers { for t in std::iter::once(&l.grid).chain(&l.mips).flat_map(|g| g.tiles.iter().flatten()) { keep.insert(std::sync::Arc::as_ptr(t) as usize); } }
        }
        for e in self.undo.iter().rev().skip(2).chain(self.redo.iter().rev().skip(2)) {
            for l in &e.state.layers { for t in std::iter::once(&l.grid).chain(&l.mips).flat_map(|g| g.tiles.iter().flatten()) {
                if !keep.contains(&(std::sync::Arc::as_ptr(t) as usize)) { t.mark_cold(); }
            } }
        }
        for l in self.layers.iter().filter(|l| !visible_layer(l)) {
            for g in std::iter::once(&l.grid).chain(&l.mips) { for t in g.tiles.iter().flatten() {
                if !visible.contains(&(std::sync::Arc::as_ptr(t) as usize)) { t.mark_cold(); }
            } }
        }
    }
    pub fn history_bytes(&self) -> usize {
        let mut seen = std::collections::HashSet::new();
        for l in &self.layers { for t in std::iter::once(&l.grid).chain(&l.mips).flat_map(|g| g.tiles.iter().flatten()) { seen.insert(std::sync::Arc::as_ptr(t) as usize); } }
        let mut bytes = 0;
        for e in self.undo.iter().chain(&self.redo) { for l in &e.state.layers { for t in std::iter::once(&l.grid).chain(&l.mips).flat_map(|g| g.tiles.iter().flatten()) {
            if seen.insert(std::sync::Arc::as_ptr(t) as usize) { bytes += t.resident_bytes(); }
        } } }
        bytes
    }
    /// History sources retain compressed/scratch tiles and read only on demand.
    pub fn history_source(&self, pos: usize, layer_id: u64) -> Option<HistorySource> {
        let g = if pos >= self.undo.len() { &self.layers.iter().find(|l| l.id == layer_id)?.grid }
        else { &self.undo[pos].state.layers.iter().find(|l| l.id == layer_id)?.grid };
        Some(HistorySource { w: g.w, h: g.h, cols: g.cols, tiles: g.tiles.clone() })
    }

    pub fn history_len(&self) -> usize {
        self.undo.len()
    }

    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|e| e.label.as_str())
    }


    pub fn history_labels(&self) -> (Vec<String>, Vec<String>) {
        (
            self.undo.iter().map(|e| e.label.clone()).collect(),
            self.redo.iter().rev().map(|e| e.label.clone()).collect(),
        )
    }

    pub fn undo(&mut self) -> Option<String> {
        let e = self.undo.pop()?;
        let label = e.label.clone();
        let cur = self.capture();
        self.redo.push(HistoryEntry::new(label.clone(), cur));
        self.restore(e.take_state());
        self.maintain_history();
        Some(label)
    }

    pub fn redo(&mut self) -> Option<String> {
        let e = self.redo.pop()?;
        let label = e.label.clone();
        let cur = self.capture();
        self.undo.push(HistoryEntry::new(label.clone(), cur));
        self.restore(e.take_state());
        self.maintain_history();
        Some(label)
    }

    fn restore(&mut self, s: DocState) {
        let resized = s.w != self.w || s.h != self.h;
        let mut old: Vec<Layer> = std::mem::take(&mut self.layers);
        let mut layers = Vec::with_capacity(s.layers.len());
        for ls in s.layers {
            let layer = match old.iter().position(|l| l.id == ls.id) {
                Some(i) if !resized => {
                    let mut l = old.swap_remove(i);
                    let changed = l.grid.tiles.len() != ls.grid.tiles.len()
                        || l.grid.tiles.iter().zip(&ls.grid.tiles).any(|(a, b)| !same_tile(a, b));
                    if changed {
                        l.grid = ls.grid;
                        l.mips = ls.mips;
                        l.version += 1;
                    }
                    l.name = ls.name;
                    l.visible = ls.visible;
                    l.opacity = ls.opacity;
                    l.blend = ls.blend;
                    l.offset = (0, 0);
                    ls.meta.apply(&mut l);
                    l
                }
                _ => {
                    let mut l = Layer::from_mips(ls.id, ls.name, ls.grid, ls.mips);
                    l.visible = ls.visible;
                    l.opacity = ls.opacity;
                    l.blend = ls.blend;
                    ls.meta.apply(&mut l);
                    l
                }
            };
            layers.push(layer);
        }
        self.layers = layers;
        self.w = s.w;
        self.h = s.h;
        if resized {
            self.generation += 1;
        }
        self.active = s.active.min(self.layers.len().saturating_sub(1));
        self.selected = s.selected;
        self.selection = s.selection;
        self.paths = s.paths;
        self.ann = s.ann;
        if self.active_path.is_some_and(|i| i >= self.paths.len()) {
            self.active_path = None;
        }
        self.preview = None;
        self.dirty.all = true;
        self.modified = true;
    }

    // ----- layer structure --------------------------------------------------

    /// Inserts a layer above the active one and makes it active.
    /// A new layer above the active one, in its group / artboard (or, when
    /// the active layer is a group or artboard, inside it at the top).
    pub fn insert_layer(&mut self, name: impl Into<String>, grid: TileGrid) -> usize {
        let id = self.next_layer_id();
        let (at, parent) = match self.layers.get(self.active) {
            Some(a) if a.kind.folder().is_some() => (self.active, Some(a.id)),
            Some(a) => ((self.active + 1).min(self.layers.len()), a.parent),
            None => (0, None),
        };
        let mut l = Layer::new(id, name, grid);
        l.parent = parent;
        self.layers.insert(at, l);
        self.active = at;
        self.dirty.all = true;
        self.modified = true;
        at
    }

    pub fn new_layer_name(&self) -> String {
        let mut n = self.layers.len() + 1;
        loop {
            let name = format!("Layer {n}");
            if !self.layers.iter().any(|l| l.name == name) {
                return name;
            }
            n += 1;
        }
    }

    pub fn delete_layer(&mut self, li: usize) {
        if self.layers.len() <= 1 {
            return;
        }
        self.layers.remove(li);
        self.active = self.active.min(self.layers.len() - 1);
        self.dirty.all = true;
    }

    pub fn move_layer(&mut self, li: usize, up: bool) -> bool {
        let to = if up { li + 1 } else { li.wrapping_sub(1) };
        // The locked Background stays at the bottom of the stack.
        if to >= self.layers.len() || self.layers[li].background || self.layers[to].background {
            return false;
        }
        self.layers.swap(li, to);
        if self.active == li {
            self.active = to;
        } else if self.active == to {
            self.active = li;
        }
        self.dirty.all = true;
        true
    }

    /// Distinct tile count across layers, mips and history (memory estimate).
    pub fn memory_tiles(&self) -> (usize, usize) {
        let mut seen = std::collections::HashSet::new();
        let mut live = 0;
        for l in &self.layers {
            live += l.grid.unique_tiles(&mut seen);
            for m in &l.mips {
                live += m.unique_tiles(&mut seen);
            }
        }
        let _ = &seen;
        (live, self.history_bytes().div_ceil(TILE_BYTES))
    }
}

#[cfg(test)]
mod history_tests {
    use super::*;
    use crate::engine::ops;

    #[test]
    fn undo_and_redo_restore_advanced_blending() {
        use crate::engine::advanced_blend::AdvancedBlend;
        let mut doc = Document::new(2, 2, None);
        let settings = std::sync::Arc::new(AdvancedBlend {
            ranges: vec![], restricted_channels: vec![1],
        });
        doc.push_undo("Restrict green");
        doc.layers[0].advanced_blend = Some(settings.clone());
        doc.undo().unwrap();
        assert!(doc.layers[0].advanced_blend.is_none());
        doc.redo().unwrap();
        assert_eq!(doc.layers[0].advanced_blend.as_deref(), Some(settings.as_ref()));
    }

    #[test]
    fn background_is_locked_and_new_layers_join_the_active_folder() {
        let mut d = Document::new(200, 100, Some(super::super::px::pack(255, 255, 255, 255)));
        assert!(d.layers[0].background && d.layers[0].name == "Background");
        let a = d.insert_layer("A", TileGrid::new(200, 100));
        // The Background can't move up, and nothing moves below it.
        assert!(!d.move_layer(0, true));
        assert!(!d.move_layer(a, false));
        // An artboard as the active layer: new layers go inside it, at its top.
        let ab_id = d.next_layer_id();
        let mut ab = Layer::new(ab_id, "Artboard 1", TileGrid::new(200, 100));
        ab.kind = LayerKind::Artboard(std::sync::Arc::new(super::super::artboard::ArtboardLayer { rect: RectI::new(0, 0, 100, 100), bg: None, collapsed: false }));
        d.layers.push(ab);
        d.active = d.layers.len() - 1;
        let k = d.insert_layer("in board", TileGrid::new(200, 100));
        assert_eq!(d.layers[k].parent, Some(ab_id));
        assert!(matches!(d.layers[k + 1].kind, LayerKind::Artboard(_)));
        // And a layer added above a child joins the same artboard.
        let k2 = d.insert_layer("also in board", TileGrid::new(200, 100));
        assert_eq!(d.layers[k2].parent, Some(ab_id));
        // Matte: transparency becomes the colour (the Background stays opaque).
        let m = ops::matte(&TileGrid::new(300, 300), [10, 20, 30]);
        assert_eq!(m.pixel(299, 299), super::super::px::pack(10, 20, 30, 255));
    }

    #[test]
    fn packed_history_roundtrips_exactly() {
        let g = ops::procedural(1024, 1024);
        let mut doc = Document::from_grid(g.clone(), "x");
        let mut snapshots = vec![g.clone()];
        for i in 0..10u8 {
            doc.push_undo(format!("step {i}"));
            let f = ops::fill(&doc.layers[0].grid, None, [i * 20, 255 - i * 20, 77], 0.35);
            let f = ops::apply_filter(&f, &crate::engine::filters::Filter::Invert, None, &ops::Progress::default());
            doc.set_layer_grid(0, f.clone());
            snapshots.push(f);
        }
        // Let background packing finish, then settle entries.
        doc.maintain_history();
        super::super::memory::global().maintain();
        let packed = doc.undo.iter().filter(|e| e.state.layers.iter().any(|l| l.grid.tiles.iter().flatten().any(|t| t.is_cold()))).count();
        assert!(packed >= 7, "only {packed} entries packed");
        let before = doc.history_bytes();
        assert!(before < 9 * 16 * TILE_BYTES, "history not compressed: {before}");
        for k in (0..10).rev() {
            doc.undo().unwrap();
            let want = &snapshots[k];
            for &(x, y) in &[(0, 0), (511, 300), (1023, 1023), (77, 900)] {
                assert_eq!(doc.layers[0].grid.pixel(x, y), want.pixel(x, y), "undo to {k} at ({x},{y})");
            }
        }
        for k in 1..=10 {
            doc.redo().unwrap();
            assert_eq!(doc.layers[0].grid.pixel(511, 300), snapshots[k].pixel(511, 300), "redo to {k}");
        }
    }

    #[test]
    fn history_source_decompresses_per_tile() {
        let g = ops::procedural(1024, 1024);
        let mut doc = Document::from_grid(g.clone(), "x");
        let id = doc.layers[0].id;
        for i in 0..6u8 {
            doc.push_undo(format!("step {i}"));
            let f = ops::apply_filter(&doc.layers[0].grid, &crate::engine::filters::Filter::Invert, None, &ops::Progress::default());
            doc.set_layer_grid(0, f);
        }
        doc.maintain_history();
        super::super::memory::global().maintain();
        // Step 0 is packed: its tiles stay compressed until asked for.
        let src = doc.history_source(0, id).unwrap();
        assert!(src.packed_tiles() > 0, "state 0 was not packed");
        let t = src.tile(5).unwrap();
        assert_eq!(t.read()[100], g.tiles[5].as_ref().unwrap().read()[100]);
        assert!(src.packed_tiles() > 0, "asking for one tile unpacked the state");
    }
}
