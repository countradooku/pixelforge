//! The filter catalog: menu placement, dialog parameters and builders.

use super::*;
use crate::engine::grid::RectI;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Menu {
    Adjustments,
    Auto,
    Blur,
    Distort,
    Noise,
    Pixelate,
    Render,
    Sharpen,
    Stylize,
    Video,
    Other,
    /// Filter ▸ Blur Gallery.
    BlurGallery,
    /// Top of the Filter menu (Camera Raw, Lens Correction, Liquify).
    Top,
    /// Filter ▸ Filter Gallery (grouped by [`gallery::G::group`]).
    Gallery,
    /// Filter ▸ Neural Filters (classical implementations).
    Neural,
    /// Edit ▸ Transform.
    Transform,
}

impl Menu {
    pub const FILTER_MENUS: [Menu; 10] = [
        Menu::BlurGallery,
        Menu::Blur,
        Menu::Distort,
        Menu::Noise,
        Menu::Pixelate,
        Menu::Render,
        Menu::Sharpen,
        Menu::Stylize,
        Menu::Video,
        Menu::Other,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Menu::Adjustments => "Adjustments",
            Menu::Auto => "Auto",
            Menu::Blur => "Blur",
            Menu::Distort => "Distort",
            Menu::Noise => "Noise",
            Menu::Pixelate => "Pixelate",
            Menu::Render => "Render",
            Menu::Sharpen => "Sharpen",
            Menu::Stylize => "Stylize",
            Menu::Video => "Video",
            Menu::Other => "Other",
            Menu::BlurGallery => "Blur Gallery",
            Menu::Top => "Filter",
            Menu::Gallery => "Filter Gallery",
            Menu::Neural => "Neural Filters",
            Menu::Transform => "Transform",
        }
    }
}

macro_rules! kinds {
    ($($k:ident),* $(,)?) => {
        #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
        pub enum Kind { $($k,)* Gallery(gallery::G) }
        impl Kind {
            const BASE: &'static [Kind] = &[$(Kind::$k),*];
            /// Every filter, including all Filter Gallery effects.
            pub fn all() -> Vec<Kind> {
                Self::BASE.iter().copied().chain(gallery::G::ALL.iter().map(|&g| Kind::Gallery(g))).collect()
            }
        }
    };
}

kinds!(
    // Adjustments
    BrightnessContrast, Levels, Curves, Exposure, Vibrance, HueSaturation, ColorBalance, BlackWhite,
    PhotoFilter, GradientMap, Invert, Desaturate, Posterize, Threshold, Equalize, ShadowsHighlights,
    AutoTone, AutoContrast,
    // Blur
    Average, Blur, BlurMore, BoxBlur, GaussianBlur, LensBlur, MotionBlur, RadialBlur, SmartBlur, SurfaceBlur,
    // Distort
    Pinch, PolarCoordinates, Ripple, Shear, Spherize, Twirl, Wave, ZigZag,
    // Noise
    AddNoise, Despeckle, DustScratches, Median, ReduceNoise,
    // Pixelate
    ColorHalftone, Crystallize, Facet, Fragment, Mezzotint, Mosaic, Pointillize,
    // Render
    Clouds, DifferenceClouds, Fibers, LensFlare,
    // Sharpen
    Sharpen, SharpenEdges, SharpenMore, SmartSharpen, UnsharpMask,
    // Stylize
    Diffuse, Emboss, FindEdges, OilPaint, Solarize, TraceContour, Wind,
    // Other
    HighPass, HsbHsl, Maximum, Minimum, Offset, Custom,
    // Remaining Photoshop filters & adjustments
    ChannelMixer, SelectiveColor, ReplaceColor, MatchColor,
    LightingEffects, Displace, ShapeBlur, TiltShift, IrisBlur, Extrude, Tiles,
    DeInterlace, NtscColors, CameraRaw, LensCorrection, Liquify,
    FieldBlur, SpinBlur, PathBlur, SkinSmoothing, JpegCleanup, PhotoRestoration, SuperZoom,
    FreeTransform, VanishingPoint, AdaptiveWideAngle, Harmonization, TransformWarp,
);

#[derive(Clone, Copy)]
pub struct Param {
    pub label: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub step: f32,
    pub log: bool,
    pub decimals: usize,
    /// Non-empty: the value is an index into these names (shown instead of a number).
    pub choices: &'static [&'static str],
}

const fn p(label: &'static str, min: f32, max: f32, default: f32, step: f32, log: bool, decimals: usize) -> Param {
    Param { label, min, max, default, step, log, decimals, choices: &[] }
}

/// Edit ▸ Transform ▸ Warp's styles (Warp Text's, without "No Warp").
const WARP_STYLES: &[&str] = &[
    "Arc", "Arc Lower", "Arc Upper", "Arch", "Bulge", "Shell Lower", "Shell Upper", "Flag", "Wave", "Fish", "Rise", "Fisheye", "Inflate", "Squeeze", "Twist",
];

const fn ch(label: &'static str, choices: &'static [&'static str], default: f32) -> Param {
    Param { label, min: 0.0, max: (choices.len() - 1) as f32, default, step: 1.0, log: false, decimals: 0, choices }
}

/// Blur Gallery's Effects (bokeh) and Noise panels, appended to the
/// parameters of Field, Iris, Tilt-Shift, Spin and Path Blur. The defaults
/// leave the blur unchanged.
pub const BLUR_FX_PARAMS: [Param; 10] = [
    p("Light Bokeh (%)", 0., 100., 0., 1., false, 0),
    p("Bokeh Color (%)", 0., 100., 0., 1., false, 0),
    p("Light Range low", 0., 255., 191., 1., false, 0),
    p("Light Range high", 0., 255., 255., 1., false, 0),
    ch("Noise type", &["Uniform", "Gaussian"], 0.),
    p("Noise amount (%)", 0., 100., 0., 1., false, 0),
    p("Noise size (px)", 1., 100., 1., 1., true, 0),
    p("Noise roughness (%)", 0., 100., 50., 1., false, 0),
    p("Noise color (%)", 0., 100., 0., 1., false, 0),
    p("Noise highlights (%)", 0., 100., 100., 1., false, 0),
];

/// The Blur Gallery effects from values `v[at..]` (missing values, e.g.
/// from files saved before these panels existed, take their defaults).
pub fn blur_fx(v: &[f32], at: usize) -> BlurFx {
    let g = |i: usize| v.get(at + i).copied().unwrap_or(BLUR_FX_PARAMS[i].default);
    BlurFx {
        bokeh: g(0),
        bokeh_color: g(1),
        range: [g(2), g(3)],
        gaussian: g(4) >= 0.5,
        noise: g(5),
        size: g(6),
        roughness: g(7),
        color: g(8),
        highlights: g(9),
    }
}

fn with_fx(mut params: Vec<Param>) -> Vec<Param> {
    params.extend_from_slice(&BLUR_FX_PARAMS);
    params
}

pub struct Entry {
    pub title: &'static str,
    pub menu: Menu,
    pub params: Vec<Param>,
    /// Optional toggle (label, default).
    pub flag: Option<(&'static str, bool)>,
    pub note: Option<&'static str>,
}

impl Entry {
    /// Runs immediately, without a dialog.
    pub fn instant(&self) -> bool {
        self.params.is_empty() && self.flag.is_none()
    }
}

pub struct BuildCtx<'a> {
    pub fg: [u8; 3],
    pub bg: [u8; 3],
    /// Selection bounds (or the whole image): centers distortions.
    pub area: RectI,
    pub stats: Option<&'a Stats>,
    /// Stats of the Match Color source (the layer below the active one).
    pub source: Option<&'a Stats>,
    /// Image chosen as the Displace map.
    pub disp_map: Option<std::sync::Arc<extra::DispMap>>,
    /// The active layer's mask (Lens Blur's Layer Mask depth map).
    pub layer_mask: Option<std::sync::Arc<blur::DepthMask>>,
    pub doc_w: u32,
    pub doc_h: u32,
}

impl Kind {
    pub fn needs_stats(self) -> bool {
        matches!(self, Kind::Equalize | Kind::AutoTone | Kind::AutoContrast | Kind::Average | Kind::MatchColor | Kind::Harmonization)
    }

    pub fn entry(self) -> Entry {
        use Kind::*;
        let e = |title, menu, params: Vec<Param>| Entry { title, menu, params, flag: None, note: None };
        let lin = |l, a, b, d| p(l, a, b, d, 1.0, false, 0);
        let radius = |a, b, d| p("Radius (px)", a, b, d, 0.1, true, 1);
        match self {
            BrightnessContrast => e("Brightness/Contrast", Menu::Adjustments, vec![lin("Brightness", -150., 150., 0.), lin("Contrast", -50., 100., 0.)]),
            Levels => e(
                "Levels",
                Menu::Adjustments,
                vec![lin("Black point", 0., 253., 0.), lin("White point", 2., 255., 255.), p("Gamma", 0.1, 9.99, 1.0, 0.01, true, 2)],
            ),
            Curves => e(
                "Curves",
                Menu::Adjustments,
                vec![
                    lin("Output at 0%", 0., 255., 0.),
                    lin("Output at 25%", 0., 255., 64.),
                    lin("Output at 50%", 0., 255., 128.),
                    lin("Output at 75%", 0., 255., 191.),
                    lin("Output at 100%", 0., 255., 255.),
                ],
            ),
            Exposure => e(
                "Exposure",
                Menu::Adjustments,
                vec![p("Exposure (EV)", -5., 5., 0., 0.01, false, 2), p("Offset", -0.5, 0.5, 0., 0.001, false, 3), p("Gamma", 0.01, 9.99, 1., 0.01, true, 2)],
            ),
            Vibrance => e("Vibrance", Menu::Adjustments, vec![lin("Vibrance", -100., 100., 0.), lin("Saturation", -100., 100., 0.)]),
            HueSaturation => e(
                "Hue/Saturation",
                Menu::Adjustments,
                vec![lin("Hue", -180., 180., 0.), lin("Saturation", -100., 100., 0.), lin("Lightness", -100., 100., 0.)],
            ),
            ColorBalance => Entry {
                flag: Some(("Preserve Luminosity", true)),
                ..e(
                    "Color Balance",
                    Menu::Adjustments,
                    vec![lin("Cyan ↔ Red", -100., 100., 0.), lin("Magenta ↔ Green", -100., 100., 0.), lin("Yellow ↔ Blue", -100., 100., 0.)],
                )
            },
            BlackWhite => e(
                "Black & White",
                Menu::Adjustments,
                vec![
                    lin("Reds", -200., 300., 40.),
                    lin("Yellows", -200., 300., 60.),
                    lin("Greens", -200., 300., 40.),
                    lin("Cyans", -200., 300., 60.),
                    lin("Blues", -200., 300., 20.),
                    lin("Magentas", -200., 300., 80.),
                ],
            ),
            PhotoFilter => Entry {
                flag: Some(("Preserve Luminosity", true)),
                note: Some("Filters with the foreground color."),
                ..e("Photo Filter", Menu::Adjustments, vec![lin("Density (%)", 1., 100., 25.)])
            },
            GradientMap => Entry { note: Some("Maps luminance from the foreground to the background color."), ..e("Gradient Map", Menu::Adjustments, vec![]) },
            Invert => e("Invert", Menu::Adjustments, vec![]),
            Desaturate => e("Desaturate", Menu::Adjustments, vec![]),
            Posterize => e("Posterize", Menu::Adjustments, vec![lin("Levels", 2., 64., 4.)]),
            Threshold => e("Threshold", Menu::Adjustments, vec![lin("Level", 1., 255., 128.)]),
            Equalize => e("Equalize", Menu::Adjustments, vec![]),
            ShadowsHighlights => e(
                "Shadows/Highlights",
                Menu::Adjustments,
                vec![lin("Shadows (%)", 0., 100., 35.), lin("Highlights (%)", 0., 100., 0.), radius(1., 500., 30.)],
            ),
            AutoTone => e("Auto Tone", Menu::Auto, vec![]),
            AutoContrast => e("Auto Contrast", Menu::Auto, vec![]),

            Average => e("Average", Menu::Blur, vec![]),
            Blur => e("Blur", Menu::Blur, vec![]),
            BlurMore => e("Blur More", Menu::Blur, vec![]),
            BoxBlur => e("Box Blur", Menu::Blur, vec![p("Radius (px)", 1., 999., 10., 1., true, 0)]),
            GaussianBlur => e("Gaussian Blur", Menu::Blur, vec![radius(0.1, 1000., 5.)]),
            LensBlur => Entry {
                flag: Some(("Monochromatic noise", false)),
                note: Some("Depth map: blur radius = Radius × |depth − focal distance| / 255."),
                ..e(
                    "Lens Blur",
                    Menu::Blur,
                    vec![
                        p("Radius (px)", 1., 100., 15., 1., false, 0),
                        lin("Specular brightness", 0., 100., 30.),
                        lin("Specular threshold", 0., 255., 0.),
                        ch("Iris shape", &["Round", "Triangle", "Square", "Pentagon", "Hexagon", "Heptagon", "Octagon"], 0.),
                        lin("Blade curvature", 0., 100., 0.),
                        lin("Iris rotation (°)", 0., 360., 0.),
                        ch("Depth map", &["None", "Transparency", "Layer Mask"], 0.),
                        lin("Blur focal distance", 0., 255., 0.),
                        ch("Invert depth map", &["Off", "On"], 0.),
                        lin("Noise amount", 0., 100., 0.),
                        ch("Noise distribution", &["Uniform", "Gaussian"], 0.),
                    ],
                )
            },
            MotionBlur => e("Motion Blur", Menu::Blur, vec![lin("Angle (°)", -90., 90., 0.), p("Distance (px)", 1., 2000., 40., 1., true, 0)]),
            RadialBlur => Entry {
                flag: Some(("Zoom (off = Spin)", false)),
                note: Some("Centered on the selection, or on the image."),
                ..e("Radial Blur", Menu::Blur, vec![lin("Amount", 1., 100., 10.)])
            },
            SmartBlur => e("Smart Blur", Menu::Blur, vec![radius(0.1, 100., 3.), p("Threshold", 0.1, 100., 25., 0.1, false, 1)]),
            SurfaceBlur => e("Surface Blur", Menu::Blur, vec![p("Radius (px)", 1., 100., 5., 1., false, 0), lin("Threshold (levels)", 2., 255., 15.)]),

            Pinch => e("Pinch", Menu::Distort, vec![lin("Amount (%)", -100., 100., 50.)]),
            PolarCoordinates => Entry { flag: Some(("Rectangular to Polar", true)), ..e("Polar Coordinates", Menu::Distort, vec![]) },
            Ripple => e("Ripple", Menu::Distort, vec![lin("Amount (%)", -999., 999., 100.), lin("Size (px)", 2., 400., 40.)]),
            Shear => e("Shear", Menu::Distort, vec![lin("Bend (px)", -500., 500., 100.)]),
            Spherize => e("Spherize", Menu::Distort, vec![lin("Amount (%)", -100., 100., 100.)]),
            Twirl => e("Twirl", Menu::Distort, vec![lin("Angle (°)", -999., 999., 50.)]),
            Wave => e("Wave", Menu::Distort, vec![p("Wavelength (px)", 10., 999., 120., 1., true, 0), p("Amplitude (px)", 1., 500., 25., 1., true, 0)]),
            ZigZag => e("ZigZag", Menu::Distort, vec![lin("Amount", -100., 100., 30.), lin("Ridges", 1., 20., 5.)]),

            AddNoise => Entry { flag: Some(("Monochromatic", false)), ..e("Add Noise", Menu::Noise, vec![p("Amount (%)", 0.1, 400., 12.5, 0.1, true, 1)]) },
            Despeckle => e("Despeckle", Menu::Noise, vec![]),
            DustScratches => e("Dust & Scratches", Menu::Noise, vec![p("Radius (px)", 1., 100., 1., 1., true, 0), lin("Threshold (levels)", 0., 255., 0.)]),
            Median => e("Median", Menu::Noise, vec![p("Radius (px)", 1., 100., 2., 1., true, 0)]),
            ReduceNoise => e("Reduce Noise", Menu::Noise, vec![lin("Strength", 0., 10., 6.)]),

            ColorHalftone => e("Color Halftone", Menu::Pixelate, vec![lin("Max radius (px)", 4., 127., 8.)]),
            Crystallize => e("Crystallize", Menu::Pixelate, vec![p("Cell size", 3., 300., 10., 1., true, 0)]),
            Facet => e("Facet", Menu::Pixelate, vec![]),
            Fragment => e("Fragment", Menu::Pixelate, vec![]),
            Mezzotint => e("Mezzotint", Menu::Pixelate, vec![]),
            Mosaic => e("Mosaic", Menu::Pixelate, vec![p("Cell size", 2., 200., 10., 1., true, 0)]),
            Pointillize => Entry {
                note: Some("Dots over the background color."),
                ..e("Pointillize", Menu::Pixelate, vec![p("Cell size", 3., 300., 10., 1., true, 0)])
            },

            Clouds => Entry { note: Some("Foreground → background color noise."), ..e("Clouds", Menu::Render, vec![]) },
            DifferenceClouds => e("Difference Clouds", Menu::Render, vec![]),
            Fibers => e("Fibers", Menu::Render, vec![lin("Variance", 1., 64., 16.), lin("Strength", 1., 64., 4.)]),
            LensFlare => e("Lens Flare", Menu::Render, vec![lin("Brightness (%)", 10., 300., 100.), lin("Center X (%)", 0., 100., 30.), lin("Center Y (%)", 0., 100., 30.)]),

            Sharpen => e("Sharpen", Menu::Sharpen, vec![]),
            SharpenEdges => e("Sharpen Edges", Menu::Sharpen, vec![]),
            SharpenMore => e("Sharpen More", Menu::Sharpen, vec![]),
            SmartSharpen => e(
                "Smart Sharpen",
                Menu::Sharpen,
                vec![
                    lin("Amount (%)", 1., 500., 200.),
                    p("Radius (px)", 0.1, 64., 1.0, 0.1, true, 1),
                    lin("Reduce noise (%)", 0., 100., 10.),
                    ch("Remove", &["Gaussian Blur", "Lens Blur", "Motion Blur"], 0.),
                    lin("Motion angle (°)", -180., 180., 0.),
                    lin("Shadows fade (%)", 0., 100., 0.),
                    lin("Shadows tonal width (%)", 0., 100., 50.),
                    p("Shadows radius (px)", 1., 100., 1., 1., false, 0),
                    lin("Highlights fade (%)", 0., 100., 0.),
                    lin("Highlights tonal width (%)", 0., 100., 50.),
                    p("Highlights radius (px)", 1., 100., 1., 1., false, 0),
                ],
            ),
            UnsharpMask => e(
                "Unsharp Mask",
                Menu::Sharpen,
                vec![lin("Amount (%)", 1., 500., 100.), radius(0.1, 1000., 2.), lin("Threshold (levels)", 0., 255., 0.)],
            ),

            Diffuse => e("Diffuse", Menu::Stylize, vec![p("Radius (px)", 1., 20., 2., 0.5, false, 1)]),
            Emboss => e("Emboss", Menu::Stylize, vec![lin("Angle (°)", -180., 180., 135.), lin("Height (px)", 1., 100., 3.), lin("Amount (%)", 1., 500., 100.)]),
            FindEdges => e("Find Edges", Menu::Stylize, vec![]),
            OilPaint => Entry {
                note: Some("Stylization, Bristle Detail and Shine at 0 give the plain brush-size filter."),
                ..e(
                    "Oil Paint",
                    Menu::Stylize,
                    vec![
                        lin("Brush size (px)", 1., 30., 4.),
                        p("Stylization", 0., 10., 0., 0.1, false, 1),
                        p("Cleanliness", 0., 10., 5., 0.1, false, 1),
                        p("Scale", 0.1, 10., 1., 0.1, false, 1),
                        p("Bristle detail", 0., 10., 0., 0.1, false, 1),
                        lin("Lighting angle (°)", -180., 180., -60.),
                        p("Shine", 0., 10., 0., 0.1, false, 1),
                    ],
                )
            },
            Solarize => e("Solarize", Menu::Stylize, vec![]),
            TraceContour => e("Trace Contour", Menu::Stylize, vec![lin("Level", 0., 255., 128.)]),
            Wind => Entry { flag: Some(("From the left", true)), ..e("Wind", Menu::Stylize, vec![p("Length (px)", 5., 1000., 60., 1., true, 0)]) },

            Gallery(g) => Entry {
                note: matches!(g.group(), "Sketch").then_some("Uses the foreground (ink) and background (paper) colors."),
                ..e(
                    g.title(),
                    Menu::Gallery,
                    g.params()
                        .into_iter()
                        .map(|gp| Param {
                            label: gp.label,
                            min: gp.min,
                            max: gp.max,
                            default: gp.default,
                            step: 1.0,
                            log: false,
                            decimals: 0,
                            choices: gp.choices,
                        })
                        .collect(),
                )
            },
            Custom => Entry {
                note: Some("3×3 kernel, row by row; result = Σ / scale + offset."),
                ..e(
                    "Custom",
                    Menu::Other,
                    vec![
                        lin("k[0,0]", -20., 20., 0.),
                        lin("k[0,1]", -20., 20., -1.),
                        lin("k[0,2]", -20., 20., 0.),
                        lin("k[1,0]", -20., 20., -1.),
                        lin("k[1,1]", -20., 20., 5.),
                        lin("k[1,2]", -20., 20., -1.),
                        lin("k[2,0]", -20., 20., 0.),
                        lin("k[2,1]", -20., 20., -1.),
                        lin("k[2,2]", -20., 20., 0.),
                        lin("Scale", 1., 99., 1.),
                        lin("Offset", -255., 255., 0.),
                    ],
                )
            },
            ChannelMixer => Entry {
                flag: Some(("Monochrome", false)),
                ..e(
                    "Channel Mixer",
                    Menu::Adjustments,
                    vec![
                        lin("Red ← Red %", -200., 200., 100.),
                        lin("Red ← Green %", -200., 200., 0.),
                        lin("Red ← Blue %", -200., 200., 0.),
                        lin("Red constant", -200., 200., 0.),
                        lin("Green ← Red %", -200., 200., 0.),
                        lin("Green ← Green %", -200., 200., 100.),
                        lin("Green ← Blue %", -200., 200., 0.),
                        lin("Green constant", -200., 200., 0.),
                        lin("Blue ← Red %", -200., 200., 0.),
                        lin("Blue ← Green %", -200., 200., 0.),
                        lin("Blue ← Blue %", -200., 200., 100.),
                        lin("Blue constant", -200., 200., 0.),
                    ],
                )
            },
            SelectiveColor => Entry {
                flag: Some(("Absolute (off = Relative)", false)),
                ..e(
                    "Selective Color",
                    Menu::Adjustments,
                    vec![
                        ch("Colors", extra::SELECTIVE_RANGES, 0.),
                        lin("Cyan %", -100., 100., 0.),
                        lin("Magenta %", -100., 100., 0.),
                        lin("Yellow %", -100., 100., 0.),
                        lin("Black %", -100., 100., 0.),
                    ],
                )
            },
            ReplaceColor => Entry {
                note: Some("Replaces colors near the foreground color."),
                ..e(
                    "Replace Color",
                    Menu::Adjustments,
                    vec![lin("Fuzziness", 0., 200., 40.), lin("Hue", -180., 180., 0.), lin("Saturation", -100., 100., 0.), lin("Lightness", -100., 100., 0.)],
                )
            },
            MatchColor => Entry {
                note: Some("Source: the layer below the active layer."),
                ..e("Match Color", Menu::Adjustments, vec![lin("Luminance", 1., 200., 100.), lin("Color intensity", 1., 200., 100.), lin("Fade", 0., 100., 0.)])
            },
            LightingEffects => Entry {
                flag: Some(("Light uses foreground color", false)),
                ..e(
                    "Lighting Effects",
                    Menu::Render,
                    vec![
                        ch("Light type", &["Point", "Spot", "Infinite"], 1.),
                        lin("Light X (%)", 0., 100., 35.),
                        lin("Light Y (%)", 0., 100., 35.),
                        lin("Radius (%)", 10., 200., 75.),
                        lin("Direction (°)", -180., 180., -45.),
                        lin("Spot cone (°)", 5., 90., 35.),
                        lin("Intensity", 0., 100., 40.),
                        lin("Ambience", 0., 100., 25.),
                        lin("Bump height", 0., 50., 10.),
                    ],
                )
            },
            Displace => Entry {
                note: Some("Map: red shifts horizontally, green vertically (128 = none). Use “Choose map image…” for a file."),
                ..e(
                    "Displace",
                    Menu::Distort,
                    vec![
                        lin("Horizontal scale (px)", -500., 500., 20.),
                        lin("Vertical scale (px)", -500., 500., 20.),
                        ch("Map", &["Layer (red/green)", "Clouds", "Image file"], 1.),
                        ch("Map placement", &["Stretch to fit", "Tile"], 0.),
                    ],
                )
            },
            ShapeBlur => e("Shape Blur", Menu::Blur, vec![p("Radius (px)", 1., 200., 15., 1., true, 0), ch("Shape", extra::SHAPES, 3.)]),
            TiltShift => e(
                "Tilt-Shift",
                Menu::BlurGallery,
                with_fx(vec![lin("Focus center Y (%)", 0., 100., 50.), lin("Focus band (%)", 0., 50., 8.), lin("Transition (%)", 1., 60., 20.), p("Blur (px)", 1., 500., 15., 1., true, 0)]),
            ),
            IrisBlur => e(
                "Iris Blur",
                Menu::BlurGallery,
                with_fx(vec![lin("Center X (%)", 0., 100., 50.), lin("Center Y (%)", 0., 100., 50.), lin("Radius (%)", 5., 100., 35.), p("Blur (px)", 1., 500., 15., 1., true, 0)]),
            ),
            Extrude => e("Extrude", Menu::Stylize, vec![lin("Size (px)", 2., 255., 30.), lin("Depth", 1., 100., 30.), ch("Type", &["Blocks", "Pyramids"], 0.)]),
            Tiles => Entry {
                note: Some("Gaps are filled with the background color."),
                ..e("Tiles", Menu::Stylize, vec![lin("Number of tiles", 1., 99., 10.), lin("Maximum offset (%)", 1., 99., 10.)])
            },
            DeInterlace => e("De-Interlace", Menu::Video, vec![ch("Eliminate", &["Odd fields", "Even fields"], 0.), ch("Create new fields by", &["Duplication", "Interpolation"], 1.)]),
            NtscColors => e("NTSC Colors", Menu::Video, vec![]),
            CameraRaw => Entry {
                note: Some("Basic · Curve · HSL · Color Grading · Detail · Effects · Masks."),
                ..e(
                    "Camera Raw Filter",
                    Menu::Top,
                    vec![
                        lin("Temperature", -100., 100., 0.),
                        lin("Tint", -100., 100., 0.),
                        p("Exposure (EV)", -5., 5., 0., 0.05, false, 2),
                        lin("Contrast", -100., 100., 0.),
                        lin("Highlights", -100., 100., 0.),
                        lin("Shadows", -100., 100., 0.),
                        lin("Whites", -100., 100., 0.),
                        lin("Blacks", -100., 100., 0.),
                        lin("Texture", -100., 100., 0.),
                        lin("Clarity", -100., 100., 0.),
                        lin("Dehaze", -100., 100., 0.),
                        lin("Vibrance", -100., 100., 0.),
                        lin("Saturation", -100., 100., 0.),
                        lin("Curve · Highlights", -100., 100., 0.),
                        lin("Curve · Lights", -100., 100., 0.),
                        lin("Curve · Darks", -100., 100., 0.),
                        lin("Curve · Shadows", -100., 100., 0.),
                        lin("HSL · Red hue", -100., 100., 0.),
                        lin("HSL · Orange hue", -100., 100., 0.),
                        lin("HSL · Yellow hue", -100., 100., 0.),
                        lin("HSL · Green hue", -100., 100., 0.),
                        lin("HSL · Aqua hue", -100., 100., 0.),
                        lin("HSL · Blue hue", -100., 100., 0.),
                        lin("HSL · Purple hue", -100., 100., 0.),
                        lin("HSL · Magenta hue", -100., 100., 0.),
                        lin("HSL · Red saturation", -100., 100., 0.),
                        lin("HSL · Orange saturation", -100., 100., 0.),
                        lin("HSL · Yellow saturation", -100., 100., 0.),
                        lin("HSL · Green saturation", -100., 100., 0.),
                        lin("HSL · Aqua saturation", -100., 100., 0.),
                        lin("HSL · Blue saturation", -100., 100., 0.),
                        lin("HSL · Purple saturation", -100., 100., 0.),
                        lin("HSL · Magenta saturation", -100., 100., 0.),
                        lin("HSL · Red luminance", -100., 100., 0.),
                        lin("HSL · Orange luminance", -100., 100., 0.),
                        lin("HSL · Yellow luminance", -100., 100., 0.),
                        lin("HSL · Green luminance", -100., 100., 0.),
                        lin("HSL · Aqua luminance", -100., 100., 0.),
                        lin("HSL · Blue luminance", -100., 100., 0.),
                        lin("HSL · Purple luminance", -100., 100., 0.),
                        lin("HSL · Magenta luminance", -100., 100., 0.),
                        lin("Grading · Shadow hue", 0., 360., 220.),
                        lin("Grading · Shadow saturation", 0., 100., 0.),
                        lin("Grading · Highlight hue", 0., 360., 40.),
                        lin("Grading · Highlight saturation", 0., 100., 0.),
                        lin("Grading · Balance", -100., 100., 0.),
                        lin("Detail · Sharpening", 0., 150., 0.),
                        lin("Detail · Noise reduction", 0., 100., 0.),
                        lin("Effects · Vignette", -100., 100., 0.),
                        lin("Effects · Grain", 0., 100., 0.),
                        lin("Mask · Linear angle (°)", -180., 180., 90.),
                        lin("Mask · Linear position (%)", 0., 100., 50.),
                        p("Mask · Linear exposure (EV)", -4., 4., 0., 0.05, false, 2),
                        lin("Mask · Radial center X (%)", 0., 100., 50.),
                        lin("Mask · Radial center Y (%)", 0., 100., 50.),
                        lin("Mask · Radial radius (%)", 5., 150., 40.),
                        p("Mask · Radial exposure (EV)", -4., 4., 0., 0.05, false, 2),
                    ],
                )
            },
            LensCorrection => e(
                "Lens Correction",
                Menu::Top,
                vec![
                    lin("Remove distortion", -100., 100., 0.),
                    lin("Fix red/cyan fringe", -100., 100., 0.),
                    lin("Fix blue/yellow fringe", -100., 100., 0.),
                    lin("Vignette", -100., 100., 0.),
                ],
            ),
            Liquify => Entry { note: Some("Opens the Liquify (Forward Warp) tool."), ..e("Liquify", Menu::Top, vec![]) },
            FieldBlur => e(
                "Field Blur",
                Menu::BlurGallery,
                with_fx(vec![
                    lin("Pin 1 X (%)", 0., 100., 25.),
                    lin("Pin 1 Y (%)", 0., 100., 30.),
                    p("Pin 1 blur (px)", 0., 500., 30., 1., false, 0),
                    lin("Pin 2 X (%)", 0., 100., 50.),
                    lin("Pin 2 Y (%)", 0., 100., 60.),
                    p("Pin 2 blur (px)", 0., 500., 0., 1., false, 0),
                    lin("Pin 3 X (%)", 0., 100., 80.),
                    lin("Pin 3 Y (%)", 0., 100., 30.),
                    p("Pin 3 blur (px)", 0., 500., 15., 1., false, 0),
                ]),
            ),
            SpinBlur => e(
                "Spin Blur",
                Menu::BlurGallery,
                with_fx(vec![
                    lin("Center X (%)", 0., 100., 50.),
                    lin("Center Y (%)", 0., 100., 50.),
                    lin("Radius X (%)", 1., 100., 25.),
                    lin("Radius Y (%)", 1., 100., 25.),
                    lin("Blur angle", 1., 100., 15.),
                ]),
            ),
            PathBlur => e(
                "Path Blur",
                Menu::BlurGallery,
                with_fx(vec![
                    lin("Start X (%)", 0., 100., 15.),
                    lin("Start Y (%)", 0., 100., 70.),
                    lin("End X (%)", 0., 100., 85.),
                    lin("End Y (%)", 0., 100., 30.),
                    lin("Bend (%)", -100., 100., 30.),
                    p("Speed / length (px)", 1., 1000., 60., 1., true, 0),
                ]),
            ),
            SkinSmoothing => Entry {
                note: Some("Classical, not ML: edge-preserving blur limited to skin tones."),
                ..e("Skin Smoothing", Menu::Neural, vec![lin("Blur", 1., 50., 8.), lin("Smoothness", 0., 100., 40.)])
            },
            JpegCleanup => Entry {
                note: Some("Classical, not ML: 8×8 deblocking plus de-ringing."),
                ..e("JPEG Artifacts Removal", Menu::Neural, vec![lin("Strength", 0., 100., 60.)])
            },
            PhotoRestoration => Entry {
                note: Some("Classical, not ML: dust/scratch removal, denoise, contrast, sharpening."),
                ..e("Photo Restoration", Menu::Neural, vec![lin("Strength", 0., 100., 50.), lin("Scratch reduction", 0., 100., 40.)])
            },
            SuperZoom => Entry {
                note: Some("Classical, not ML: bicubic upscaling of the whole document with detail sharpening."),
                ..e("Super Zoom", Menu::Neural, vec![ch("Scale", &["2×", "3×", "4×"], 0.), lin("Enhance detail", 0., 100., 40.)])
            },
            FreeTransform => Entry {
                note: Some("Transforms the active layer about the selection (or image) center."),
                ..e(
                    "Free Transform",
                    Menu::Transform,
                    vec![
                        lin("Move X (px)", -10000., 10000., 0.),
                        lin("Move Y (px)", -10000., 10000., 0.),
                        p("Scale X (%)", -400., 400., 100., 0.5, false, 1),
                        p("Scale Y (%)", -400., 400., 100., 0.5, false, 1),
                        p("Rotate (°)", -180., 180., 0., 0.1, false, 1),
                        p("Skew X (°)", -80., 80., 0., 0.1, false, 1),
                        p("Skew Y (°)", -80., 80., 0., 0.1, false, 1),
                        lin("Perspective X", -100., 100., 0.),
                        lin("Perspective Y", -100., 100., 0.),
                    ],
                )
            },
            TransformWarp => Entry {
                flag: Some(("Vertical", false)),
                note: Some("Edit ▸ Transform ▸ Warp: the Warp Text styles applied to the pixels of the selection (or image) box."),
                ..e(
                    "Warp",
                    Menu::Transform,
                    vec![
                        ch("Style", WARP_STYLES, 0.),
                        lin("Bend (%)", -100., 100., 50.),
                        lin("Horizontal distortion (%)", -100., 100., 0.),
                        lin("Vertical distortion (%)", -100., 100., 0.),
                    ],
                )
            },
            VanishingPoint => Entry {
                note: Some("Pick the four corners of a plane in perspective; it is rectified to the full canvas."),
                ..e(
                    "Vanishing Point",
                    Menu::Top,
                    vec![
                        lin("Top-left X (%)", 0., 100., 10.),
                        lin("Top-left Y (%)", 0., 100., 10.),
                        lin("Top-right X (%)", 0., 100., 90.),
                        lin("Top-right Y (%)", 0., 100., 20.),
                        lin("Bottom-right X (%)", 0., 100., 90.),
                        lin("Bottom-right Y (%)", 0., 100., 80.),
                        lin("Bottom-left X (%)", 0., 100., 10.),
                        lin("Bottom-left Y (%)", 0., 100., 90.),
                    ],
                )
            },
            Harmonization => Entry {
                note: Some("Classical, not ML: matches this layer's color statistics to the layer below."),
                ..e("Harmonization", Menu::Neural, vec![lin("Strength (%)", 0., 100., 60.), lin("Saturation", 1., 200., 100.)])
            },
            AdaptiveWideAngle => e(
                "Adaptive Wide Angle",
                Menu::Top,
                vec![lin("Fisheye correction", 0., 100., 50.), lin("Scale (%)", 50., 150., 100.)],
            ),
            HighPass => e("High Pass", Menu::Other, vec![radius(0.1, 1000., 10.)]),
            HsbHsl => e("HSB/HSL", Menu::Other, vec![]),
            Maximum => e("Maximum", Menu::Other, vec![p("Radius (px)", 1., 500., 2., 1., true, 0)]),
            Minimum => e("Minimum", Menu::Other, vec![p("Radius (px)", 1., 500., 2., 1., true, 0)]),
            Offset => Entry {
                note: Some("Wraps around the edges."),
                ..e("Offset", Menu::Other, vec![lin("Horizontal (px)", -5000., 5000., 0.), lin("Vertical (px)", -5000., 5000., 0.)])
            },
        }
    }

    pub fn build(self, v: &[f32], flag: bool, ctx: &BuildCtx) -> Filter {
        use Kind::*;
        let a = ctx.area;
        let (cx, cy) = ((a.x0 + a.x1) as f32 * 0.5, (a.y0 + a.y1) as f32 * 0.5);
        let rad = (a.w().min(a.h()) as f32 * 0.5).max(1.0);
        let (dw, dh) = (ctx.doc_w as f32, ctx.doc_h as f32);
        let warp = |name, warp| Filter::Warp { name, warp };
        let st = || ctx.stats.expect("stats");
        match self {
            BrightnessContrast => Filter::BrightnessContrast { brightness: v[0], contrast: v[1] },
            Levels => Filter::Levels { black: v[0], white: v[1], gamma: v[2] },
            Curves => Filter::Curves { pts: [v[0], v[1], v[2], v[3], v[4]] },
            Exposure => Filter::Exposure { exposure: v[0], offset: v[1], gamma: v[2] },
            Vibrance => vibrance(v[0], v[1]),
            HueSaturation => hue_saturation(v[0], v[1], v[2]),
            ColorBalance => color_balance(v[0], v[1], v[2], flag),
            BlackWhite => black_white([v[0], v[1], v[2], v[3], v[4], v[5]]),
            PhotoFilter => photo_filter(ctx.fg, v[0], flag),
            GradientMap => gradient_map(ctx.fg, ctx.bg),
            Invert => Filter::Invert,
            Desaturate => Filter::Desaturate,
            Posterize => Filter::Posterize { levels: v[0] },
            Threshold => Filter::Threshold { level: v[0] },
            Equalize => equalize(st()),
            ShadowsHighlights => Filter::ShadowsHighlights { shadows: v[0], highlights: v[1], radius: v[2] },
            AutoTone => auto_tone(st()),
            AutoContrast => auto_contrast(st()),
            Average => average(st()),
            Blur => Filter::Blur,
            BlurMore => Filter::BlurMore,
            BoxBlur => Filter::BoxBlur { radius: v[0] },
            GaussianBlur => Filter::GaussianBlur { radius: v[0] },
            LensBlur => {
                let g = |i: usize, d: f32| v.get(i).copied().unwrap_or(d);
                let shape = g(3, 0.0).round() as u8;
                Filter::LensBlur {
                    radius: v[0],
                    bloom: v[1],
                    opts: blur::LensOpts {
                        blades: if shape == 0 { 0 } else { shape + 2 },
                        curvature: g(4, 0.0),
                        rotation: g(5, 0.0),
                        depth: match (g(6, 0.0).round() as u8, &ctx.layer_mask) {
                            (1, _) => blur::Depth::Transparency,
                            (2, Some(m)) => blur::Depth::Mask(m.clone()),
                            _ => blur::Depth::None,
                        },
                        focal: g(7, 0.0),
                        invert: g(8, 0.0) >= 0.5,
                        threshold: g(2, 0.0),
                        noise: g(9, 0.0),
                        gaussian: g(10, 0.0) >= 0.5,
                        mono: flag,
                    },
                }
            }
            MotionBlur => Filter::MotionBlur { angle: v[0], distance: v[1] },
            RadialBlur => Filter::RadialBlur { amount: v[0], zoom: flag, cx, cy },
            SmartBlur => Filter::SmartBlur { radius: v[0], threshold: v[1] },
            SurfaceBlur => Filter::SurfaceBlur { radius: v[0], threshold: v[1] },
            Pinch => warp("Pinch", Warp::Pinch { amount: v[0], cx, cy, radius: rad }),
            PolarCoordinates => warp("Polar Coordinates", Warp::Polar { to_polar: flag, w: dw, h: dh }),
            Ripple => warp("Ripple", Warp::Ripple { amount: v[0], size: v[1] }),
            Shear => warp("Shear", Warp::Shear { amount: v[0], h: dh }),
            Spherize => warp("Spherize", Warp::Spherize { amount: v[0], cx, cy, radius: rad }),
            Twirl => warp("Twirl", Warp::Twirl { angle: v[0], cx, cy, radius: rad }),
            Wave => warp("Wave", Warp::Wave { wavelength: v[0], amplitude: v[1] }),
            ZigZag => warp("ZigZag", Warp::ZigZag { amount: v[0], ridges: v[1], cx, cy, radius: rad }),
            AddNoise => Filter::AddNoise { amount: v[0], mono: flag },
            Despeckle => Filter::Despeckle,
            DustScratches => Filter::DustScratches { radius: v[0], threshold: v[1] },
            Median => Filter::Median { radius: v[0] },
            ReduceNoise => Filter::ReduceNoise { strength: v[0] },
            ColorHalftone => Filter::ColorHalftone { radius: v[0] },
            Crystallize => Filter::Crystallize { cell: v[0] },
            Facet => Filter::Facet,
            Fragment => Filter::Fragment,
            Mezzotint => Filter::Mezzotint,
            Mosaic => Filter::Mosaic { cell: v[0] },
            Pointillize => Filter::Pointillize { cell: v[0], bg: ctx.bg },
            Clouds => Filter::Clouds { c0: ctx.fg, c1: ctx.bg, scale: 220.0, difference: false },
            DifferenceClouds => Filter::Clouds { c0: ctx.fg, c1: ctx.bg, scale: 220.0, difference: true },
            Fibers => Filter::Fibers { c0: ctx.fg, c1: ctx.bg, variance: v[0], strength: v[1] },
            LensFlare => Filter::LensFlare { x: dw * v[1] / 100.0, y: dh * v[2] / 100.0, brightness: v[0], size: dw.min(dh) * 0.5 },
            Sharpen => Filter::Sharpen,
            SharpenEdges => Filter::SharpenEdges,
            SharpenMore => Filter::SharpenMore,
            SmartSharpen => {
                let g = |i: usize, d: f32| v.get(i).copied().unwrap_or(d);
                let fade = |o: usize| blur::Fade { amount: g(o, 0.0), width: g(o + 1, 50.0), radius: g(o + 2, 1.0) };
                Filter::SmartSharpen {
                    amount: v[0],
                    radius: v[1],
                    noise: v[2] * 0.3,
                    opts: blur::SharpenOpts {
                        remove: match g(3, 0.0).round() as u8 {
                            1 => blur::Remove::Lens,
                            2 => blur::Remove::Motion,
                            _ => blur::Remove::Gaussian,
                        },
                        angle: g(4, 0.0),
                        shadows: fade(5),
                        highlights: fade(8),
                    },
                }
            }
            UnsharpMask => Filter::UnsharpMask { amount: v[0], radius: v[1], threshold: v[2] },
            Diffuse => Filter::Diffuse { radius: v[0] },
            Emboss => Filter::Emboss { angle: v[0], height: v[1], amount: v[2] },
            FindEdges => Filter::FindEdges,
            OilPaint => Filter::OilPaint {
                radius: v[0],
                oil: stylize::Oil { stylization: v[1], cleanliness: v[2], scale: v[3], bristle: v[4], angle: v[5], shine: v[6] },
            },
            Solarize => Filter::Solarize,
            TraceContour => Filter::TraceContour { level: v[0] },
            Wind => Filter::Wind { length: v[0], from_left: flag },
            Gallery(g) => {
                let mut pp = [0f32; 3];
                for (i, x) in v.iter().take(3).enumerate() {
                    pp[i] = *x;
                }
                Filter::Gallery { g, p: pp, fg: ctx.fg, bg: ctx.bg, doc: (ctx.doc_w, ctx.doc_h) }
            }
            Custom => Filter::Custom { k: [v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]], scale: v[9], offset: v[10] },
            ChannelMixer => extra::channel_mixer([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]], flag),
            SelectiveColor => extra::selective_color(v[0].round() as usize, [v[1], v[2], v[3], v[4]], flag),
            ReplaceColor => extra::replace_color(ctx.fg, v[0], v[1], v[2], v[3]),
            MatchColor => extra::match_color(st(), ctx.source.unwrap_or(st()), v[0], v[1], v[2]),
            LightingEffects => Filter::Lighting(extra::Light {
                kind: v[0].round() as u8,
                x: dw * v[1] / 100.0,
                y: dh * v[2] / 100.0,
                radius: dw.max(dh) * v[3] / 100.0,
                angle: v[4],
                cone: v[5],
                intensity: v[6],
                ambient: v[7],
                bump: v[8],
                color: if flag { ctx.fg } else { [255, 255, 255] },
            }),
            Displace => Filter::Displace {
                hs: v[0],
                vs: v[1],
                map: match (v[2].round() as u8, &ctx.disp_map) {
                    (0, _) => extra::MapSource::Layer,
                    (2, Some(m)) => extra::MapSource::File { map: m.clone(), tile: v[3] >= 0.5, img_w: dw, img_h: dh },
                    _ => extra::MapSource::Clouds,
                },
            },
            ShapeBlur => Filter::ShapeBlur { radius: v[0], shape: v[1].round() as usize },
            TiltShift => Filter::TiltShift { cy: dh * v[0] / 100.0, band: dh * v[1] / 100.0, transition: dh * v[2] / 100.0, blur: v[3], fx: blur_fx(v, 4) },
            IrisBlur => Filter::IrisBlur { cx: dw * v[0] / 100.0, cy: dh * v[1] / 100.0, radius: dw.min(dh) * v[2] / 100.0, blur: v[3], fx: blur_fx(v, 4) },
            Extrude => Filter::Extrude { size: v[0], depth: v[1], pyramids: v[2] >= 0.5 },
            Tiles => Filter::Tiles { size: dw.max(1.0) / v[0].max(1.0), offset: v[1], fill: crate::engine::px::pack(ctx.bg[0], ctx.bg[1], ctx.bg[2], 255) },
            DeInterlace => Filter::DeInterlace { odd: v[0] < 0.5, interpolate: v[1] >= 0.5 },
            NtscColors => extra::ntsc_colors(),
            CameraRaw => {
                let a8 = |o: usize| [v[o], v[o + 1], v[o + 2], v[o + 3], v[o + 4], v[o + 5], v[o + 6], v[o + 7]];
                Filter::CameraRaw(extra::Raw {
                    temperature: v[0],
                    tint: v[1],
                    exposure: v[2],
                    contrast: v[3],
                    highlights: v[4],
                    shadows: v[5],
                    whites: v[6],
                    blacks: v[7],
                    texture: v[8],
                    clarity: v[9],
                    dehaze: v[10],
                    vibrance: v[11],
                    saturation: v[12],
                    curve: [v[13], v[14], v[15], v[16]],
                    hue: a8(17),
                    sat: a8(25),
                    lum: a8(33),
                    shadow_hue: v[41],
                    shadow_sat: v[42],
                    highlight_hue: v[43],
                    highlight_sat: v[44],
                    balance: v[45],
                    sharpen: v[46],
                    noise_reduction: v[47],
                    vignette: v[48],
                    grain: v[49],
                    linear: [v[50], v[51], v[52]],
                    radial: [v[53], v[54], v[55], v[56]],
                    img_w: dw,
                    img_h: dh,
                })
            }
            LensCorrection => Filter::LensCorrection { distortion: v[0], ca_rc: v[1], ca_by: v[2], vignette: v[3], w: dw, h: dh },
            // Liquify is an interactive tool; the editor intercepts it.
            Liquify => Filter::Invert,
            FieldBlur => Filter::FieldBlur {
                pins: [
                    [dw * v[0] / 100.0, dh * v[1] / 100.0, v[2]],
                    [dw * v[3] / 100.0, dh * v[4] / 100.0, v[5]],
                    [dw * v[6] / 100.0, dh * v[7] / 100.0, v[8]],
                ],
                fx: blur_fx(v, 9),
            },
            SpinBlur => Filter::SpinBlur { cx: dw * v[0] / 100.0, cy: dh * v[1] / 100.0, rx: dw * v[2] / 100.0, ry: dh * v[3] / 100.0, amount: v[4], fx: blur_fx(v, 5) },
            PathBlur => Filter::PathBlur {
                x0: dw * v[0] / 100.0,
                y0: dh * v[1] / 100.0,
                x1: dw * v[2] / 100.0,
                y1: dh * v[3] / 100.0,
                bend: dw.max(dh) * v[4] / 200.0,
                length: v[5],
                fx: blur_fx(v, 6),
            },
            SkinSmoothing => Filter::SkinSmoothing { radius: v[0], smoothness: v[1] },
            JpegCleanup => Filter::JpegCleanup { strength: v[0] },
            PhotoRestoration => Filter::PhotoRestoration { strength: v[0], scratch: v[1] },
            // The canvas resample runs on commit; the preview shows the detail sharpening.
            SuperZoom => Filter::UnsharpMask { amount: v[1] * 2.0, radius: 1.0 + v[0], threshold: 2.0 },
            FreeTransform => {
                let (cx, cy) = ((a.x0 + a.x1) as f32 * 0.5, (a.y0 + a.y1) as f32 * 0.5);
                let t = |tx: f32, ty: f32| [1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0];
                let (sx, sy) = (v[2] / 100.0, v[3] / 100.0);
                let (sn, cs) = v[4].to_radians().sin_cos();
                let (kx, ky) = (v[5].to_radians().tan(), v[6].to_radians().tan());
                let scale = [sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0];
                let rot = [cs, -sn, 0.0, sn, cs, 0.0, 0.0, 0.0, 1.0];
                let skew = [1.0, kx, 0.0, ky, 1.0, 0.0, 0.0, 0.0, 1.0];
                let span = a.w().max(a.h()).max(1) as f32;
                let persp = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, v[7] / 100.0 / span, v[8] / 100.0 / span, 1.0];
                use distort::{mat_inv, mat_mul};
                let fwd = mat_mul(&t(cx + v[0], cy + v[1]), &mat_mul(&persp, &mat_mul(&rot, &mat_mul(&skew, &mat_mul(&scale, &t(-cx, -cy))))));
                warp("Free Transform", Warp::Homography { m: mat_inv(&fwd) })
            }
            TransformWarp => {
                use crate::engine::warp::{BoxWarp, Warp as TextWarp, WarpStyle};
                let style = WarpStyle::ALL[1 + (v[0].round().max(0.0) as usize).min(WarpStyle::ALL.len() - 2)];
                let tw = TextWarp { style, bend: v[1] / 100.0, h_distort: v[2] / 100.0, v_distort: v[3] / 100.0, vertical: flag };
                let bw = BoxWarp::new(&tw, (a.x0 as f32, a.y0 as f32, a.x1 as f32, a.y1 as f32));
                warp("Warp", bw.pixel_warp())
            }
            VanishingPoint => {
                let q = [
                    (dw * v[0] / 100.0, dh * v[1] / 100.0),
                    (dw * v[2] / 100.0, dh * v[3] / 100.0),
                    (dw * v[4] / 100.0, dh * v[5] / 100.0),
                    (dw * v[6] / 100.0, dh * v[7] / 100.0),
                ];
                let s2q = distort::square_to_quad(q);
                let norm = [1.0 / dw, 0.0, 0.0, 0.0, 1.0 / dh, 0.0, 0.0, 0.0, 1.0];
                warp("Vanishing Point", Warp::Homography { m: distort::mat_mul(&s2q, &norm) })
            }
            Harmonization => extra::match_color(st(), ctx.source.unwrap_or(st()), 100.0, v[1], 100.0 - v[0]),
            AdaptiveWideAngle => warp(
                "Adaptive Wide Angle",
                Warp::WideAngle { strength: v[0], scale: v[1], cx: dw * 0.5, cy: dh * 0.5, diag: (dw * dw + dh * dh).sqrt() * 0.5 },
            ),
            HighPass => Filter::HighPass { radius: v[0] },
            HsbHsl => channel_swap_hsl(),
            Maximum => Filter::Maximum { radius: v[0] },
            Minimum => Filter::Minimum { radius: v[0] },
            Offset => warp("Offset", Warp::Offset { dx: v[0], dy: v[1], w: dw, h: dh }),
        }
    }

    /// Default-parameter instance (benchmarks, tests).
    pub fn default_filter(self, ctx: &BuildCtx) -> Filter {
        let e = self.entry();
        let v: Vec<f32> = e.params.iter().map(|p| p.default).collect();
        self.build(&v, e.flag.is_some_and(|f| f.1), ctx)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::document::Layer;
    use crate::engine::ops::{Progress, apply_filter, procedural};

    #[test]
    fn every_filter_builds_renders_and_commits() {
        let g = procedural(600, 400);
        let layer = Layer::new(1, "t", g.clone());
        let st = stats(&layer);
        let ctx = BuildCtx { fg: [10, 20, 30], bg: [250, 240, 230], area: g.bounds(), stats: Some(&st), source: None, disp_map: None, layer_mask: None, doc_w: 600, doc_h: 400 };
        let p = Progress::default();
        for k in Kind::all() {
            if k == Kind::Liquify {
                continue;
            }
            let f = k.default_filter(&ctx);
            let out = apply_filter(&g, &f, None, &p);
            assert_eq!((out.w, out.h), (600, 400), "{:?}", k);
            // Preview path at a coarse level must not panic either.
            let mut buf = vec![0u32; 64 * 64];
            let mut scratch = Vec::new();
            f.render(&Src::layer(&layer), 1, 0, 0, 64, 64, &mut buf, &mut scratch);
        }
    }
}

#[cfg(test)]
mod consistency {
    use super::*;
    use crate::engine::document::Document;
    use crate::engine::ops::{Progress, apply_filter, procedural};

    #[test]
    fn warp_commit_keeps_mips_consistent() {
        let g = procedural(2048, 2048);
        let mut doc = Document::from_grid(g.clone(), "x");
        let st = stats(&doc.layers[0]);
        let ctx = BuildCtx { fg: [0; 3], bg: [255; 3], area: g.bounds(), stats: Some(&st), source: None, disp_map: None, layer_mask: None, doc_w: 2048, doc_h: 2048 };
        let f = Kind::Twirl.build(&[400.0], false, &ctx);
        let out = apply_filter(&doc.layers[0].grid, &f, None, &Progress::default());
        let differs = (0..2048).step_by(97).any(|i| out.pixel(i, 700) != g.pixel(i, 700));
        assert!(differs, "twirl changed nothing");
        doc.set_layer_grid(0, out);
        let fresh = crate::engine::mips::build_mips(&doc.layers[0].grid);
        for (l, (a, b)) in doc.layers[0].mips.iter().zip(&fresh).enumerate() {
            for &(x, y) in &[(10, 10), (a.w as i32 / 2, a.h as i32 / 3), (a.w as i32 - 3, a.h as i32 - 3)] {
                assert_eq!(a.pixel(x, y), b.pixel(x, y), "mip level {} stale at ({x},{y})", l + 1);
            }
        }
    }

    #[test]
    fn crystallize_has_no_holes() {
        let g = procedural(1024, 1024);
        let ctx = BuildCtx { fg: [0; 3], bg: [255; 3], area: g.bounds(), stats: None, source: None, disp_map: None, layer_mask: None, doc_w: 1024, doc_h: 1024 };
        let f = Kind::Crystallize.build(&[120.0], false, &ctx);
        let out = apply_filter(&g, &f, None, &Progress::default());
        for y in (0..1024).step_by(7) {
            for x in (0..1024).step_by(7) {
                assert_eq!(crate::engine::px::alpha(out.pixel(x, y)), 255, "hole at ({x},{y})");
            }
        }
    }
}

#[cfg(test)]
mod chain {
    use super::*;
    use crate::engine::ops::{Progress, apply_filter, procedural};

    #[test]
    fn crystallize_uses_its_input() {
        let g = procedural(4096, 4096);
        let ctx = BuildCtx { fg: [0; 3], bg: [255; 3], area: g.bounds(), stats: None, source: None, disp_map: None, layer_mask: None, doc_w: 4096, doc_h: 4096 };
        let p = Progress::default();
        let tw = apply_filter(&g, &Kind::Twirl.build(&[400.0], false, &ctx), None, &p);
        let c1 = apply_filter(&tw, &Kind::Crystallize.build(&[120.0], false, &ctx), None, &p);
        let c0 = apply_filter(&g, &Kind::Crystallize.build(&[120.0], false, &ctx), None, &p);
        let mut same = 0;
        let mut white = 0;
        let mut total = 0;
        for y in (0..4096).step_by(37) {
            for x in (0..4096).step_by(37) {
                total += 1;
                if c1.pixel(x, y) == c0.pixel(x, y) {
                    same += 1;
                }
                let [r, gg, b, _] = crate::engine::px::unpack(c1.pixel(x, y));
                if r > 245 && gg > 245 && b > 245 {
                    white += 1;
                }
            }
        }
        eprintln!("same={same}/{total} white={white}");
        assert!(same < total / 2);
        assert!(white < total / 100);
    }
}

#[cfg(test)]
mod preview_chain {
    use super::*;
    use crate::engine::document::Layer;
    use crate::engine::ops::{Progress, apply_filter, procedural};

    #[test]
    #[ignore]
    fn preview_reads_current_pixels_16k() {
        check(16384);
    }

    #[test]
    fn preview_reads_current_pixels() {
        check(4096);
    }

    fn check(n: u32) {
        let g = procedural(n, n);
        let ctx = BuildCtx { fg: [0; 3], bg: [255; 3], area: g.bounds(), stats: None, source: None, disp_map: None, layer_mask: None, doc_w: n, doc_h: n };
        let tw = apply_filter(&g, &Kind::Twirl.build(&[400.0], false, &ctx), None, &Progress::default());
        let lt = Layer::new(1, "t", tw);
        let lo = Layer::new(2, "o", g);
        let f = Kind::Crystallize.build(&[120.0], false, &ctx);
        let (mut a, mut b) = (vec![0u32; 256 * 256], vec![0u32; 256 * 256]);
        let mut s = Vec::new();
        let c = (n as i32 >> 4) - 128;
        f.render(&Src::layer(&lt), 3, c, c, 256, 256, &mut a, &mut s);
        f.render(&Src::layer(&lo), 3, c, c, 256, 256, &mut b, &mut s);
        let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
        eprintln!("same {same} / {}", a.len());
        // raw levels differ?
        let raw_same = (0..256).filter(|&i| lt.level(3).pixel(c + i, c + 100) == lo.level(3).pixel(c + i, c + 100)).count();
        eprintln!("raw same {raw_same}/256");
        // Crystallize should stay close to its input colors.
        let near = |x: &[u32], l: &Layer| {
            let mut d = 0u64;
            for y in (0..256).step_by(9) {
                for xx in (0..256).step_by(9) {
                    let p = crate::engine::px::unpack(x[y * 256 + xx]);
                    let q = crate::engine::px::unpack(l.level(3).pixel(c + xx as i32, c + y as i32));
                    d += (0..3).map(|c| (p[c] as i64 - q[c] as i64).unsigned_abs()).sum::<u64>();
                }
            }
            d
        };
        eprintln!("dist to twirled {}  dist to original {}", near(&a, &lt), near(&a, &lo));
        assert!(same < a.len() / 2);
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use crate::engine::ops::{Progress, apply_filter, procedural, upscale_bicubic};

    fn ctx(w: u32, h: u32) -> BuildCtx<'static> {
        BuildCtx { fg: [0; 3], bg: [255; 3], area: RectI::new(0, 0, w as i32, h as i32), stats: None, source: None, disp_map: None, layer_mask: None, doc_w: w, doc_h: h }
    }

    #[test]
    fn free_transform_identity_and_scale() {
        let g = procedural(600, 400);
        let c = ctx(600, 400);
        let id = Kind::FreeTransform.default_filter(&c);
        let out = apply_filter(&g, &id, None, &Progress::default());
        for &(x, y) in &[(10, 10), (300, 200), (599, 399)] {
            assert_eq!(out.pixel(x, y), g.pixel(x, y));
        }
        // 200% about the center: the pixel at center+d comes from center+d/2.
        let f = Kind::FreeTransform.build(&[0., 0., 200., 200., 0., 0., 0., 0., 0.], false, &c);
        let out = apply_filter(&g, &f, None, &Progress::default());
        let a = crate::engine::px::unpack(out.pixel(400, 300));
        let b = crate::engine::px::unpack(g.pixel(350, 250));
        for k in 0..3 {
            assert!((a[k] as i32 - b[k] as i32).abs() <= 6, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn vanishing_point_maps_corners() {
        let c = ctx(1000, 800);
        let f = Kind::VanishingPoint.build(&[10., 10., 90., 20., 90., 80., 10., 90.], false, &c);
        let Filter::Warp { warp, .. } = f else { panic!() };
        let (x, y) = warp.map(0.0, 0.0);
        assert!((x - 100.0).abs() < 0.5 && (y - 80.0).abs() < 0.5, "{x},{y}");
        let (x, y) = warp.map(1000.0, 800.0);
        assert!((x - 900.0).abs() < 0.5 && (y - 640.0).abs() < 0.5, "{x},{y}");
    }

    #[test]
    fn bicubic_upscale_keeps_flat_and_gradients() {
        let g = procedural(300, 200);
        let l = crate::engine::document::Layer::new(1, "x", g.clone());
        let up = upscale_bicubic(&l, 600, 400);
        assert_eq!((up.w, up.h), (600, 400));
        let a = crate::engine::px::unpack(up.pixel(301, 201));
        let b = crate::engine::px::unpack(g.pixel(150, 100));
        for k in 0..3 {
            assert!((a[k] as i32 - b[k] as i32).abs() <= 8, "{a:?} vs {b:?}");
        }
    }
}
