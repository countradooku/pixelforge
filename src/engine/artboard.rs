//! Artboards (Photoshop's artboard layers): a top-level container with a
//! rectangle, an optional background colour and its own child layers. The
//! compositor fills the background, composites the children and clips them
//! to the rectangle; export renders one artboard's subtree only (layers
//! outside every artboard are not part of any artboard file).

use super::document::Layer;
use super::grid::RectI;
use super::layers::LayerKind;

#[derive(Clone, Debug, PartialEq)]
pub struct ArtboardLayer {
    pub rect: RectI,
    /// `None` = transparent.
    pub bg: Option<[u8; 3]>,
    pub collapsed: bool,
}

/// Photoshop's artboard presets (pixels).
pub const PRESETS: &[(&str, i32, i32)] = &[
    ("iPhone 16", 1179, 2556),
    ("iPhone 16 Pro Max", 1320, 2868),
    ("iPad Pro 11-inch", 1668, 2420),
    ("Android 1080p", 1080, 1920),
    ("Web 1920", 1920, 1080),
    ("Web 1366", 1366, 768),
    ("MacBook Pro 14-inch", 3024, 1964),
    ("Apple Watch 45mm", 396, 484),
    ("Instagram Post", 1080, 1080),
    ("Instagram Story", 1080, 1920),
    ("A4 @ 150 ppi", 1240, 1754),
    ("Letter @ 150 ppi", 1275, 1650),
];


/// The layer at `li` and all its descendants, in stack order.
pub fn subtree(layers: &[Layer], li: usize) -> Vec<Layer> {
    let root = layers[li].id;
    let mut keep = vec![false; layers.len()];
    keep[li] = true;
    // Parents sit above their children; iterate until no change (shallow trees).
    let mut changed = true;
    while changed {
        changed = false;
        for (i, l) in layers.iter().enumerate() {
            if !keep[i] && l.parent.is_some_and(|p| p == root || layers.iter().enumerate().any(|(j, m)| keep[j] && m.id == p)) {
                keep[i] = true;
                changed = true;
            }
        }
    }
    layers.iter().enumerate().filter(|(i, _)| keep[*i]).map(|(_, l)| l.clone()).collect()
}

/// Straight-RGBA pixels of artboard `li` (its own subtree, clipped to its rect).
pub fn render(layers: &[Layer], li: usize) -> Option<(Vec<u8>, u32, u32)> {
    let LayerKind::Artboard(ab) = &layers[li].kind else { return None };
    let r = ab.rect;
    if r.is_empty() {
        return None;
    }
    let mut sub = subtree(layers, li);
    // Exported on its own, even if hidden in the document.
    let id = layers[li].id;
    if let Some(root) = sub.iter_mut().find(|l| l.id == id) {
        root.visible = true;
        root.parent = None;
    }
    Some((super::annot::region_rgba(&sub, r), r.w() as u32, r.h() as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::grid::TileGrid;
    use crate::engine::px;
    use std::sync::Arc;

    #[test]
    fn artboard_clips_children_and_fills_background() {
        let (w, h) = (1200u32, 800u32);
        let mut ab = Layer::new(10, "Artboard 1", TileGrid::new(w, h));
        ab.kind = LayerKind::Artboard(Arc::new(ArtboardLayer { rect: RectI::new(100, 100, 600, 500), bg: Some([255, 255, 255]), collapsed: false }));
        // A red child covering the whole canvas, and a blue layer outside any artboard.
        let mut child = Layer::new(11, "red", TileGrid::filled(w, h, px::pack(220, 0, 0, 255)));
        child.parent = Some(10);
        child.opacity = 0.5;
        let blue = Layer::new(12, "blue", TileGrid::filled(w, h, px::pack(0, 0, 255, 255)));
        // Stack bottom → top: blue, child, artboard (parents sit above children).
        let layers = vec![blue, child, ab];
        let comp = crate::engine::composite::sample(&layers, 300, 300);
        // Inside: 50 % red over the white artboard background (over blue below).
        let [r, g, b, _] = px::unpack(comp);
        assert!(r > 200 && g > 100 && g < 140 && b > 100 && b < 140, "{r} {g} {b}");
        // Outside the artboard: the child is clipped away, blue shows.
        assert_eq!(crate::engine::composite::sample(&layers, 800, 300), px::pack(0, 0, 255, 255));
        // Export: only the artboard subtree (no blue), at the rect's size.
        let (rgba, ew, eh) = render(&layers, 2).unwrap();
        assert_eq!((ew, eh), (500, 400));
        let p = &rgba[(200 * 500 + 200) * 4..][..4];
        assert!(p[0] > 230 && p[2] > 100 && p[2] < 140 && p[3] == 255, "{p:?}");
        assert_eq!(subtree(&layers, 2).len(), 2);
    }
}
