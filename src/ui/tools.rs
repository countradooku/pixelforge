//! The Photoshop 2026 toolbar: every tool, its group (flyout), shortcut and
//! one-line help.

use gpui_kit::assets::IconName as I;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Tool {
    // Move
    Move,
    Artboard,
    // Marquee
    RectSelect,
    EllipseSelect,
    RowSelect,
    ColumnSelect,
    // Lasso
    Lasso,
    PolyLasso,
    MagneticLasso,
    // Selection
    ObjectSelect,
    QuickSelect,
    MagicWand,
    // Crop
    Crop,
    PerspectiveCrop,
    Slice,
    SliceSelect,
    // Frame
    Frame,
    // Eyedropper
    Eyedropper,
    ColorSampler,
    Ruler,
    Note,
    Count,
    // Healing
    SpotHeal,
    Heal,
    Patch,
    ContentAwareMove,
    RedEye,
    // Brush
    Brush,
    Pencil,
    ColorReplace,
    Mixer,
    // Stamp
    CloneStamp,
    PatternStamp,
    // History
    HistoryBrush,
    ArtHistory,
    // Eraser
    Eraser,
    BackgroundEraser,
    MagicEraser,
    // Gradient
    Gradient,
    Bucket,
    // Blur
    Blur,
    Sharpen,
    Smudge,
    // Dodge
    Dodge,
    Burn,
    Sponge,
    // Pen
    Pen,
    FreeformPen,
    ContentAwareTracing,
    CurvaturePen,
    AddAnchor,
    DeleteAnchor,
    ConvertPoint,
    // Type
    TypeH,
    TypeV,
    TypeMaskV,
    TypeMaskH,
    // Path selection
    PathSelect,
    DirectSelect,
    // Shapes
    RectShape,
    EllipseShape,
    TriangleShape,
    PolygonShape,
    LineShape,
    CustomShape,
    // Navigation
    Hand,
    RotateView,
    Zoom,
    // Filters as tools
    Liquify,
}

/// A toolbar slot: the tools that share one button and flyout.
pub struct Group {
    pub key: Option<&'static str>,
    pub tools: &'static [Tool],
}

pub const GROUPS: &[Group] = &[
    Group { key: Some("v"), tools: &[Tool::Move, Tool::Artboard] },
    Group { key: Some("m"), tools: &[Tool::RectSelect, Tool::EllipseSelect, Tool::RowSelect, Tool::ColumnSelect] },
    Group { key: Some("l"), tools: &[Tool::Lasso, Tool::PolyLasso, Tool::MagneticLasso] },
    Group { key: Some("w"), tools: &[Tool::ObjectSelect, Tool::QuickSelect, Tool::MagicWand] },
    Group { key: Some("c"), tools: &[Tool::Crop, Tool::PerspectiveCrop, Tool::Slice, Tool::SliceSelect] },
    Group { key: Some("k"), tools: &[Tool::Frame] },
    Group { key: Some("i"), tools: &[Tool::Eyedropper, Tool::ColorSampler, Tool::Ruler, Tool::Note, Tool::Count] },
    Group { key: Some("j"), tools: &[Tool::SpotHeal, Tool::Heal, Tool::Patch, Tool::ContentAwareMove, Tool::RedEye] },
    Group { key: Some("b"), tools: &[Tool::Brush, Tool::Pencil, Tool::ColorReplace, Tool::Mixer] },
    Group { key: Some("s"), tools: &[Tool::CloneStamp, Tool::PatternStamp] },
    Group { key: Some("y"), tools: &[Tool::HistoryBrush, Tool::ArtHistory] },
    Group { key: Some("e"), tools: &[Tool::Eraser, Tool::BackgroundEraser, Tool::MagicEraser] },
    Group { key: Some("g"), tools: &[Tool::Gradient, Tool::Bucket] },
    Group { key: None, tools: &[Tool::Blur, Tool::Sharpen, Tool::Smudge] },
    Group { key: Some("o"), tools: &[Tool::Dodge, Tool::Burn, Tool::Sponge] },
    Group {
        key: Some("p"),
        tools: &[Tool::Pen, Tool::FreeformPen, Tool::ContentAwareTracing, Tool::CurvaturePen, Tool::AddAnchor, Tool::DeleteAnchor, Tool::ConvertPoint],
    },
    Group { key: Some("t"), tools: &[Tool::TypeH, Tool::TypeV, Tool::TypeMaskV, Tool::TypeMaskH] },
    Group { key: Some("a"), tools: &[Tool::PathSelect, Tool::DirectSelect] },
    Group { key: Some("u"), tools: &[Tool::RectShape, Tool::EllipseShape, Tool::TriangleShape, Tool::PolygonShape, Tool::LineShape, Tool::CustomShape] },
    Group { key: Some("h"), tools: &[Tool::Hand, Tool::RotateView] },
    Group { key: Some("z"), tools: &[Tool::Zoom] },
    Group { key: None, tools: &[Tool::Liquify] },
];

/// Separators are drawn before these groups (as in Photoshop's layout).
pub const SEPARATE_BEFORE: &[usize] = &[4, 7, 15, 19];

impl Tool {
    pub fn group(self) -> usize {
        GROUPS.iter().position(|g| g.tools.contains(&self)).unwrap_or(0)
    }

    /// Tools whose Shift+key cycling is skipped (Photoshop's Rotate View
    /// has its own key R).
    pub fn shortcut(self) -> Option<&'static str> {
        match self {
            Tool::RotateView => Some("r"),
            Tool::Liquify => None,
            _ => GROUPS[self.group()].key,
        }
    }

    pub fn name(self) -> &'static str {
        use Tool::*;
        match self {
            Move => "Move",
            Artboard => "Artboard",
            RectSelect => "Rectangular Marquee",
            EllipseSelect => "Elliptical Marquee",
            RowSelect => "Single Row Marquee",
            ColumnSelect => "Single Column Marquee",
            Lasso => "Lasso",
            PolyLasso => "Polygonal Lasso",
            MagneticLasso => "Magnetic Lasso",
            ObjectSelect => "Object Selection",
            QuickSelect => "Quick Selection",
            MagicWand => "Magic Wand",
            Crop => "Crop",
            PerspectiveCrop => "Perspective Crop",
            Slice => "Slice",
            SliceSelect => "Slice Select",
            Frame => "Frame",
            Eyedropper => "Eyedropper",
            ColorSampler => "Color Sampler",
            Ruler => "Ruler",
            Note => "Note",
            Count => "Count",
            SpotHeal => "Spot Healing Brush",
            Heal => "Healing Brush",
            Patch => "Patch",
            ContentAwareMove => "Content-Aware Move",
            RedEye => "Red Eye",
            Brush => "Brush",
            Pencil => "Pencil",
            ColorReplace => "Color Replacement",
            Mixer => "Mixer Brush",
            CloneStamp => "Clone Stamp",
            PatternStamp => "Pattern Stamp",
            HistoryBrush => "History Brush",
            ArtHistory => "Art History Brush",
            Eraser => "Eraser",
            BackgroundEraser => "Background Eraser",
            MagicEraser => "Magic Eraser",
            Gradient => "Gradient",
            Bucket => "Paint Bucket",
            Blur => "Blur",
            Sharpen => "Sharpen",
            Smudge => "Smudge",
            Dodge => "Dodge",
            Burn => "Burn",
            Sponge => "Sponge",
            Pen => "Pen",
            FreeformPen => "Freeform Pen",
            ContentAwareTracing => "Content-Aware Tracing",
            CurvaturePen => "Curvature Pen",
            AddAnchor => "Add Anchor Point",
            DeleteAnchor => "Delete Anchor Point",
            ConvertPoint => "Convert Point",
            TypeH => "Horizontal Type",
            TypeV => "Vertical Type",
            TypeMaskV => "Vertical Type Mask",
            TypeMaskH => "Horizontal Type Mask",
            PathSelect => "Path Selection",
            DirectSelect => "Direct Selection",
            RectShape => "Rectangle",
            EllipseShape => "Ellipse",
            TriangleShape => "Triangle",
            PolygonShape => "Polygon",
            LineShape => "Line",
            CustomShape => "Custom Shape",
            Hand => "Hand",
            RotateView => "Rotate View",
            Zoom => "Zoom",
            Liquify => "Liquify",
        }
    }

    /// One-line explanation shown in tooltips and the options bar.
    pub fn help(self) -> &'static str {
        use Tool::*;
        match self {
            Move => "Drag to move the layer or selected pixels; Auto-Select picks the layer under the cursor",
            Artboard => "Drag to create a named artboard; each one exports as its own image",
            RectSelect => "Drag a rectangular selection · Shift add, Alt subtract, Shift+Alt intersect",
            EllipseSelect => "Drag an elliptical selection · Shift add, Alt subtract, Shift+Alt intersect",
            RowSelect => "Click to select one full-width row of pixels",
            ColumnSelect => "Click to select one full-height column of pixels",
            Lasso => "Drag a freehand selection",
            PolyLasso => "Click to place straight-edged points; double-click or click the start to close",
            MagneticLasso => "Click, then move along an edge; the path snaps to it (live-wire)",
            ObjectSelect => "Drag a box around an object to select it (GrabCut)",
            QuickSelect => "Paint over a region; the selection grows to its edges",
            MagicWand => "Click to select similar colors",
            Crop => "Drag or adjust the crop box; drag outside it to rotate; Enter applies",
            PerspectiveCrop => "Click four corners of a plane; Enter rectifies it",
            Slice => "Drag to define a slice for export",
            SliceSelect => "Click a slice to select it; drag to move it",
            Frame => "Drag a rectangular or elliptical frame that clips an image placed into it",
            Eyedropper => "Click to sample a color (sample size in the options bar)",
            ColorSampler => "Click to place up to 10 persistent color readouts (Info panel)",
            Ruler => "Drag to measure distance and angle; Straighten rotates the layer level",
            Note => "Click to add a note to the document",
            Count => "Click to place numbered count markers",
            SpotHeal => "Paint over blemishes; they are filled from surrounding texture",
            Heal => "Alt-click a source, then paint; texture is blended to match the surroundings",
            Patch => "Draw around an area, then drag it onto a source patch",
            ContentAwareMove => "Draw around an object, then drag it; the hole is filled",
            RedEye => "Click a red pupil to neutralize it",
            Brush => "Paint with the foreground color",
            Pencil => "Paint hard-edged, aliased strokes; Auto Erase paints background over foreground",
            ColorReplace => "Replace colors under the brush with the foreground color",
            Mixer => "Paint that mixes with the canvas (wet, load, mix)",
            CloneStamp => "Alt-click a source, then paint to copy it",
            PatternStamp => "Paint with a pattern",
            HistoryBrush => "Paint pixels back from the chosen history state",
            ArtHistory => "Paint stylized strokes sampled from the chosen history state",
            Eraser => "Erase to transparency",
            BackgroundEraser => "Erase pixels similar to the color under the hotspot",
            MagicEraser => "Click to erase similar colors",
            Gradient => "Drag to draw a gradient; edit stops in the options bar",
            Bucket => "Click to fill similar colors with the foreground or a pattern",
            Blur => "Paint to soften detail",
            Sharpen => "Paint to sharpen detail",
            Smudge => "Paint to smear pixels along the stroke",
            Dodge => "Paint to lighten (shadows, midtones or highlights)",
            Burn => "Paint to darken (shadows, midtones or highlights)",
            Sponge => "Paint to saturate or desaturate",
            Pen => "Click for corner points, drag for smooth points; click the start to close",
            FreeformPen => "Drag to draw a path freehand",
            ContentAwareTracing => "Hover an edge and click to trace it into a path",
            CurvaturePen => "Click points; curves pass smoothly through them",
            AddAnchor => "Click a path segment to add an anchor",
            DeleteAnchor => "Click an anchor to remove it",
            ConvertPoint => "Click an anchor to toggle corner / smooth; drag to pull handles",
            TypeH => "Click to type, or drag a box for paragraph text",
            TypeV => "Click to type vertical text",
            TypeMaskV => "Type to create a vertical text-shaped selection",
            TypeMaskH => "Type to create a text-shaped selection",
            PathSelect => "Click a path to select and move it",
            DirectSelect => "Drag anchors and handles of a path",
            RectShape => "Drag a rectangle shape (corner radius in the options bar)",
            EllipseShape => "Drag an ellipse shape",
            TriangleShape => "Drag a triangle shape",
            PolygonShape => "Drag a polygon or star shape",
            LineShape => "Drag a line, optionally with arrowheads",
            CustomShape => "Drag a shape from the custom shape library",
            Hand => "Drag to pan (or hold Space)",
            RotateView => "Drag to rotate the view (the image is not changed)",
            Zoom => "Click to zoom in, Alt-click out; drag left/right for scrubby zoom",
            Liquify => "Push, twirl, pucker and bloat pixels",
        }
    }

    pub fn icon(self) -> I {
        use Tool::*;
        match self {
            Move => I::Move,
            Artboard => I::LayoutTemplate,
            RectSelect => I::SquareDashed,
            EllipseSelect => I::CircleDashed,
            RowSelect => I::Rows3,
            ColumnSelect => I::Columns3,
            Lasso => I::Lasso,
            PolyLasso => I::LassoSelect,
            MagneticLasso => I::Magnet,
            ObjectSelect => I::SquareDashedMousePointer,
            QuickSelect => I::Wand,
            MagicWand => I::WandSparkles,
            Crop => I::Crop,
            PerspectiveCrop => I::Scan,
            Slice => I::Slice,
            SliceSelect => I::SquareMousePointer,
            Frame => I::Frame,
            Eyedropper => I::Pipette,
            ColorSampler => I::Crosshair,
            Ruler => I::Ruler,
            Note => I::StickyNote,
            Count => I::Hash,
            SpotHeal => I::Bandage,
            Heal => I::Sparkles,
            Patch => I::Sticker,
            ContentAwareMove => I::MoveDiagonal,
            RedEye => I::ScanFace,
            Brush => I::Paintbrush,
            Pencil => I::Pencil,
            ColorReplace => I::PaintbrushVertical,
            Mixer => I::Palette,
            CloneStamp => I::Stamp,
            PatternStamp => I::Grid3x3,
            HistoryBrush => I::Undo2,
            ArtHistory => I::Feather,
            Eraser => I::Eraser,
            BackgroundEraser => I::BrushCleaning,
            MagicEraser => I::Sparkles,
            Gradient => I::Blend,
            Bucket => I::PaintBucket,
            Blur => I::Droplet,
            Sharpen => I::Triangle,
            Smudge => I::Pointer,
            Dodge => I::SunDim,
            Burn => I::Eclipse,
            Sponge => I::Droplets,
            Pen => I::PenTool,
            FreeformPen => I::PenLine,
            ContentAwareTracing => I::Signature,
            CurvaturePen => I::Spline,
            AddAnchor => I::CirclePlus,
            DeleteAnchor => I::CircleMinus,
            ConvertPoint => I::ChevronsUpDown,
            TypeH => I::Type,
            TypeV => I::Baseline,
            TypeMaskV => I::SquareDashedText,
            TypeMaskH => I::TypeOutline,
            PathSelect => I::MousePointer2,
            DirectSelect => I::SplinePointer,
            RectShape => I::RectangleHorizontal,
            EllipseShape => I::Circle,
            TriangleShape => I::Triangle,
            PolygonShape => I::Hexagon,
            LineShape => I::Slash,
            CustomShape => I::Shapes,
            Hand => I::Hand,
            RotateView => I::RotateCw,
            Zoom => I::ZoomIn,
            Liquify => I::Tornado,
        }
    }

    pub fn label(self) -> String {
        match self.shortcut() {
            Some(k) => format!("{} ({})", self.name(), k.to_uppercase()),
            None => self.name().to_string(),
        }
    }

    pub fn tooltip(self) -> String {
        format!("{} — {}", self.label(), self.help())
    }

    /// Tools that paint with the dab engine (brush cursor, size keys).
    pub fn is_brush(self) -> bool {
        use Tool::*;
        matches!(
            self,
            Brush | Pencil | ColorReplace | Mixer | CloneStamp | PatternStamp | HistoryBrush | ArtHistory | Eraser | BackgroundEraser
                | Blur | Sharpen | Smudge | Dodge | Burn | Sponge | SpotHeal | Heal | QuickSelect | Liquify
        )
    }

    pub fn is_selection(self) -> bool {
        use Tool::*;
        matches!(self, RectSelect | EllipseSelect | RowSelect | ColumnSelect | Lasso | PolyLasso | MagneticLasso | ObjectSelect | QuickSelect | MagicWand)
    }
}

/// The tool a shortcut key selects: the group's current tool, or with Shift
/// the next tool in the group (Photoshop's "Use Shift key for tool switch").
pub fn tool_for_key(key: &str, shift: bool, current: Tool, remembered: &dyn Fn(usize) -> Tool) -> Option<Tool> {
    if key == "r" {
        return Some(Tool::RotateView);
    }
    let gi = GROUPS.iter().position(|g| g.key == Some(key))?;
    let g = &GROUPS[gi];
    let cur = if current.group() == gi { current } else { remembered(gi) };
    if !shift {
        return Some(cur);
    }
    let i = g.tools.iter().position(|&t| t == cur).unwrap_or(0);
    // Rotate View cycles with Hand only through R.
    let mut j = (i + 1) % g.tools.len();
    if g.tools[j] == Tool::RotateView {
        j = (j + 1) % g.tools.len();
    }
    Some(g.tools[j])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_in_exactly_one_group_and_cycling() {
        let mut seen = std::collections::HashSet::new();
        for g in GROUPS {
            for t in g.tools {
                assert!(seen.insert(*t), "{t:?} twice");
            }
        }
        assert_eq!(seen.len(), 69);
        let rem = |gi: usize| GROUPS[gi].tools[0];
        assert_eq!(tool_for_key("m", false, Tool::Brush, &rem), Some(Tool::RectSelect));
        assert_eq!(tool_for_key("m", true, Tool::RectSelect, &rem), Some(Tool::EllipseSelect));
        assert_eq!(tool_for_key("m", true, Tool::ColumnSelect, &rem), Some(Tool::RectSelect));
        assert_eq!(tool_for_key("j", true, Tool::RedEye, &rem), Some(Tool::SpotHeal));
        assert_eq!(tool_for_key("h", true, Tool::Hand, &rem), Some(Tool::Hand));
        assert_eq!(tool_for_key("r", false, Tool::Hand, &rem), Some(Tool::RotateView));
    }
}
