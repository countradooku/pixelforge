//! Filters and adjustments.
//!
//! Every filter belongs to one class:
//! - **Point**: per-pixel (LUTs, 3D color LUTs, generators). Pad 0.
//! - **Local**: reads a padded neighborhood. The same code renders the live
//!   preview at the display mip level (spatial parameters scaled by 2^-level)
//!   and the full-resolution commit.
//! - **Warp**: geometric remap, sampled from the source grid.
//! - **Gather**: multi-sample along paths (Radial Blur).
//!
//! Blur-derived filters (Gaussian, Box, Unsharp, High Pass, Smart Sharpen,
//! Shadows/Highlights) additionally expose [`Filter::strip_blur`], letting the
//! commit path use radius-independent full-width/full-height strips.

pub mod blur;
pub mod catalog;
pub mod distort;
pub mod extra;
pub mod fast;
pub mod gallery;
pub mod rank;
pub mod stylize;
pub mod util;

use std::sync::Arc;

use crate::engine::document::Layer;
use crate::engine::px::{self, pack, unpack};

pub use distort::Warp;
pub use extra::BlurFx;
use util::{Lut3d, apply_lut3, fbm, hash01, hsl_to_rgb, lut_from, luma, map_straight, rgb_to_hsl};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Point,
    Local,
    Warp,
    Gather,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    // ----- point adjustments -----
    BrightnessContrast { brightness: f32, contrast: f32 },
    Levels { black: f32, white: f32, gamma: f32 },
    Curves { pts: [f32; 5] },
    Exposure { exposure: f32, offset: f32, gamma: f32 },
    Invert,
    Desaturate,
    Threshold { level: f32 },
    Posterize { levels: f32 },
    Solarize,
    Lut { name: &'static str, lut: Arc<[[u8; 256]; 3]> },
    Color { name: &'static str, lut: Arc<Lut3d> },
    Average { color: u32 },
    // ----- point generators -----
    AddNoise { amount: f32, mono: bool },
    Mezzotint,
    Clouds { c0: [u8; 3], c1: [u8; 3], scale: f32, difference: bool },
    Fibers { c0: [u8; 3], c1: [u8; 3], variance: f32, strength: f32 },
    LensFlare { x: f32, y: f32, brightness: f32, size: f32 },
    /// A placed gradient (Gradient tool), composited over the layer.
    Gradient(Arc<crate::engine::gradient::GradientFill>),
    // ----- local -----
    GaussianBlur { radius: f32 },
    BoxBlur { radius: f32 },
    Blur,
    BlurMore,
    MotionBlur { angle: f32, distance: f32 },
    /// `bloom`: specular highlights brightness; the rest in `opts`.
    LensBlur { radius: f32, bloom: f32, opts: blur::LensOpts },
    SurfaceBlur { radius: f32, threshold: f32 },
    SmartBlur { radius: f32, threshold: f32 },
    Median { radius: f32 },
    DustScratches { radius: f32, threshold: f32 },
    Despeckle,
    ReduceNoise { strength: f32 },
    Maximum { radius: f32 },
    Minimum { radius: f32 },
    UnsharpMask { amount: f32, radius: f32, threshold: f32 },
    Sharpen,
    SharpenMore,
    SharpenEdges,
    /// `noise`: Reduce Noise (a threshold); Remove and the fades in `opts`.
    SmartSharpen { amount: f32, radius: f32, noise: f32, opts: blur::SharpenOpts },
    HighPass { radius: f32 },
    ShadowsHighlights { shadows: f32, highlights: f32, radius: f32 },
    FindEdges,
    Emboss { angle: f32, height: f32, amount: f32 },
    TraceContour { level: f32 },
    Diffuse { radius: f32 },
    Wind { length: f32, from_left: bool },
    /// `oil`: Photoshop's brush and lighting controls (defaults: plain Kuwahara).
    OilPaint { radius: f32, oil: stylize::Oil },
    Facet,
    Mosaic { cell: f32 },
    Crystallize { cell: f32 },
    Pointillize { cell: f32, bg: [u8; 3] },
    ColorHalftone { radius: f32 },
    Fragment,
    // ----- filter gallery & remaining filters -----
    /// `doc`: document size in level-0 px (some effects light or frame the whole document).
    Gallery { g: gallery::G, p: [f32; 3], fg: [u8; 3], bg: [u8; 3], doc: (u32, u32) },
    Lighting(extra::Light),
    Displace { hs: f32, vs: f32, map: extra::MapSource },
    FieldBlur { pins: [[f32; 3]; 3], fx: BlurFx },
    SpinBlur { cx: f32, cy: f32, rx: f32, ry: f32, amount: f32, fx: BlurFx },
    PathBlur { x0: f32, y0: f32, x1: f32, y1: f32, bend: f32, length: f32, fx: BlurFx },
    SkinSmoothing { radius: f32, smoothness: f32 },
    JpegCleanup { strength: f32 },
    PhotoRestoration { strength: f32, scratch: f32 },
    ShapeBlur { radius: f32, shape: usize },
    TiltShift { cy: f32, band: f32, transition: f32, blur: f32, fx: BlurFx },
    IrisBlur { cx: f32, cy: f32, radius: f32, blur: f32, fx: BlurFx },
    Extrude { size: f32, depth: f32, pyramids: bool },
    Tiles { size: f32, offset: f32, fill: u32 },
    Custom { k: [f32; 9], scale: f32, offset: f32 },
    DeInterlace { odd: bool, interpolate: bool },
    CameraRaw(extra::Raw),
    // ----- geometric -----
    Warp { name: &'static str, warp: Warp },
    RadialBlur { amount: f32, zoom: bool, cx: f32, cy: f32 },
    LensCorrection { distortion: f32, ca_rc: f32, ca_by: f32, vignette: f32, w: f32, h: f32 },
}

/// Applies a filter to a whole layer grid at full resolution. This is the
/// one switch between the GPU and CPU paths: the GPU runs it when enabled,
/// supported and predicted faster (see [`crate::engine::gpu`]); otherwise,
/// or for whatever the GPU couldn't finish, the CPU reference does.
pub fn apply(grid: &crate::engine::grid::TileGrid, filter: &Filter, sel: Option<&crate::engine::selection::Selection>, progress: &crate::engine::ops::Progress) -> crate::engine::grid::TileGrid {
    match crate::engine::gpu::apply_filter(grid, filter, sel, progress) {
        Some(g) => g,
        None => crate::engine::ops::apply_filter_cpu(grid, filter, sel, progress, None),
    }
}

/// A layer's pixels at every mip level (index = level).
pub struct Src<'a> {
    pub levels: Vec<&'a crate::engine::grid::TileGrid>,
}

impl<'a> Src<'a> {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn single(g: &'a crate::engine::grid::TileGrid) -> Self {
        Src { levels: vec![g] }
    }

    pub fn layer(l: &'a Layer) -> Self {
        Src { levels: std::iter::once(&l.grid).chain(l.mips.iter()).collect() }
    }

    #[inline]
    pub fn at(&self, l: u32) -> &'a crate::engine::grid::TileGrid {
        self.levels[(l as usize).min(self.levels.len() - 1)]
    }

    pub fn max_level(&self) -> u32 {
        self.levels.len() as u32 - 1
    }
}

/// What a strip-blurred commit combines: `combine(original, blurred)`.
pub struct StripBlur {
    pub boxes: Vec<usize>,
}

#[inline]
fn sc(v: f32, level: u32) -> f32 {
    v / (1u32 << level) as f32
}

#[inline]
fn r_at(v: f32, level: u32) -> usize {
    sc(v, level).round().max(0.0) as usize
}

impl Filter {
    pub fn name(&self) -> &'static str {
        use Filter::*;
        match self {
            BrightnessContrast { .. } => "Brightness/Contrast",
            Levels { .. } => "Levels",
            Curves { .. } => "Curves",
            Exposure { .. } => "Exposure",
            Invert => "Invert",
            Desaturate => "Desaturate",
            Threshold { .. } => "Threshold",
            Posterize { .. } => "Posterize",
            Solarize => "Solarize",
            Lut { name, .. } | Color { name, .. } | Warp { name, .. } => name,
            Average { .. } => "Average",
            AddNoise { .. } => "Add Noise",
            Mezzotint => "Mezzotint",
            Clouds { difference: false, .. } => "Clouds",
            Clouds { difference: true, .. } => "Difference Clouds",
            Fibers { .. } => "Fibers",
            LensFlare { .. } => "Lens Flare",
            Gradient(_) => "Gradient",
            GaussianBlur { .. } => "Gaussian Blur",
            BoxBlur { .. } => "Box Blur",
            Blur => "Blur",
            BlurMore => "Blur More",
            MotionBlur { .. } => "Motion Blur",
            LensBlur { .. } => "Lens Blur",
            SurfaceBlur { .. } => "Surface Blur",
            SmartBlur { .. } => "Smart Blur",
            Median { .. } => "Median",
            DustScratches { .. } => "Dust & Scratches",
            Despeckle => "Despeckle",
            ReduceNoise { .. } => "Reduce Noise",
            Maximum { .. } => "Maximum",
            Minimum { .. } => "Minimum",
            UnsharpMask { .. } => "Unsharp Mask",
            Sharpen => "Sharpen",
            SharpenMore => "Sharpen More",
            SharpenEdges => "Sharpen Edges",
            SmartSharpen { .. } => "Smart Sharpen",
            HighPass { .. } => "High Pass",
            ShadowsHighlights { .. } => "Shadows/Highlights",
            FindEdges => "Find Edges",
            Emboss { .. } => "Emboss",
            TraceContour { .. } => "Trace Contour",
            Diffuse { .. } => "Diffuse",
            Wind { .. } => "Wind",
            OilPaint { .. } => "Oil Paint",
            Facet => "Facet",
            Mosaic { .. } => "Mosaic",
            Crystallize { .. } => "Crystallize",
            Pointillize { .. } => "Pointillize",
            ColorHalftone { .. } => "Color Halftone",
            Fragment => "Fragment",
            RadialBlur { .. } => "Radial Blur",
            Gallery { g, .. } => g.title(),
            Lighting(_) => "Lighting Effects",
            FieldBlur { .. } => "Field Blur",
            SpinBlur { .. } => "Spin Blur",
            PathBlur { .. } => "Path Blur",
            SkinSmoothing { .. } => "Skin Smoothing",
            JpegCleanup { .. } => "JPEG Artifacts Removal",
            PhotoRestoration { .. } => "Photo Restoration",
            Displace { .. } => "Displace",
            ShapeBlur { .. } => "Shape Blur",
            TiltShift { .. } => "Tilt-Shift",
            IrisBlur { .. } => "Iris Blur",
            Extrude { .. } => "Extrude",
            Tiles { .. } => "Tiles",
            Custom { .. } => "Custom",
            DeInterlace { .. } => "De-Interlace",
            CameraRaw(_) => "Camera Raw Filter",
            LensCorrection { .. } => "Lens Correction",
        }
    }

    pub fn class(&self) -> Class {
        use Filter::*;
        match self {
            BrightnessContrast { .. } | Levels { .. } | Curves { .. } | Exposure { .. } | Invert | Desaturate
            | Threshold { .. } | Posterize { .. } | Solarize | Lut { .. } | Color { .. } | Average { .. }
            | AddNoise { .. } | Mezzotint | Clouds { .. } | Fibers { .. } | LensFlare { .. } | Gradient(_) => Class::Point,
            Warp { .. } => Class::Warp,
            RadialBlur { .. } | LensCorrection { .. } | SpinBlur { .. } | PathBlur { .. } => Class::Gather,
            _ => Class::Local,
        }
    }

    /// Transparent pixels stay transparent, so empty tiles can be skipped.
    pub fn is_pointwise(&self) -> bool {
        self.class() == Class::Point && !self.generates()
    }

    /// Generators produce pixels even where the layer is empty.
    pub fn generates(&self) -> bool {
        matches!(self, Filter::Clouds { difference: false, .. } | Filter::Fibers { .. } | Filter::Average { .. } | Filter::Gradient(_))
    }

    /// Output depends on absolute position (disables flat-tile shortcuts).
    pub fn position_dependent(&self) -> bool {
        use Filter::*;
        matches!(
            self,
            AddNoise { .. }
                | Mezzotint
                | Gradient(_)
                | Clouds { .. }
                | Fibers { .. }
                | LensFlare { .. }
                | Diffuse { .. }
                | Mosaic { .. }
                | Crystallize { .. }
                | Pointillize { .. }
                | ColorHalftone { .. }
                | Warp { .. }
                | RadialBlur { .. }
                | Gallery { .. }
                | Lighting(_)
                | Displace { .. }
                | FieldBlur { .. }
                | SpinBlur { .. }
                | PathBlur { .. }
                | JpegCleanup { .. }
                | TiltShift { .. }
                | IrisBlur { .. }
                | Extrude { .. }
                | Tiles { .. }
                | DeInterlace { .. }
                | LensCorrection { .. }
                | CameraRaw(_)
        ) || matches!(self, LensBlur { opts, .. } if opts.noise > 0.0)
    }

    pub(crate) fn unsharp_params(&self) -> Option<(f32, f32, f32)> {
        match *self {
            Filter::UnsharpMask { amount, radius, threshold } => Some((amount, radius, threshold)),
            Filter::Sharpen => Some((60.0, 0.6, 0.0)),
            Filter::SharpenMore => Some((140.0, 0.7, 0.0)),
            Filter::SharpenEdges => Some((100.0, 1.0, 10.0)),
            Filter::SmartSharpen { amount, radius, noise, opts } if opts.is_plain() => Some((amount, radius, noise)),
            _ => None,
        }
    }

    /// Gaussian sigma for the blur-combine family.
    fn blur_sigma(&self) -> Option<f32> {
        match *self {
            Filter::GaussianBlur { radius } | Filter::HighPass { radius } | Filter::ShadowsHighlights { radius, .. } => Some(radius),
            _ => self.unsharp_params().map(|p| p.1),
        }
    }

    /// Commit-time strip blur parameters (radius-independent cost).
    pub fn strip_blur(&self) -> Option<StripBlur> {
        if let Filter::BoxBlur { radius } = *self {
            return Some(StripBlur { boxes: vec![2 * radius.round().max(0.0) as usize + 1] });
        }
        let sigma = self.blur_sigma()?;
        Some(StripBlur { boxes: blur::boxes_for_gauss(sigma).to_vec() })
    }

    /// Prepares the per-pixel merge for the blur-combine family, baking any
    /// expensive math (Shadows/Highlights' power curves) into a table.
    pub fn combiner(&self) -> Combiner<'_> {
        let table = match *self {
            Filter::ShadowsHighlights { shadows, highlights, .. } => {
                let mut t = vec![0u8; 256 * 256].into_boxed_slice();
                for m8 in 0..256 {
                    let m = m8 as f32 / 255.0;
                    let lift = shadows / 100.0 * (1.0 - m).powi(2);
                    let damp = highlights / 100.0 * m.powi(2);
                    let gs = 1.0 / (1.0 + 2.0 * lift);
                    let gh = 1.0 / (1.0 + 2.0 * damp);
                    for v in 0..256 {
                        let c = (v as f32 / 255.0).powf(gs);
                        t[m8 * 256 + v] = ((1.0 - (1.0 - c).powf(gh)) * 255.0 + 0.5) as u8;
                    }
                }
                Some(t)
            }
            _ => None,
        };
        Combiner { f: self, table }
    }

    /// Merges an original pixel with its blurred version (blur-combine family).
    #[inline]
    fn combine(&self, o: u32, b: u32) -> u32 {
        match *self {
            Filter::GaussianBlur { .. } | Filter::BoxBlur { .. } => b,
            Filter::HighPass { .. } => {
                let (ob, bb) = (unpack(o), unpack(b));
                let a = ob[3] as i32;
                let ch = |c: usize| (ob[c] as i32 - bb[c] as i32 + a / 2).clamp(0, a) as u8;
                pack(ch(0), ch(1), ch(2), ob[3])
            }
            Filter::ShadowsHighlights { shadows, highlights, .. } => {
                if px::alpha(o) == 0 {
                    return o;
                }
                let [r, g, bl, a] = px::unpremultiply(o);
                let [br, bg, bb, _] = px::unpremultiply(b);
                let m = luma([br as f32, bg as f32, bb as f32]) / 255.0;
                let lift = shadows / 100.0 * (1.0 - m).powi(2);
                let damp = highlights / 100.0 * m.powi(2);
                let gs = 1.0 / (1.0 + 2.0 * lift);
                let gh = 1.0 / (1.0 + 2.0 * damp);
                let f = |v: u8| {
                    let c = (v as f32 / 255.0).powf(gs);
                    ((1.0 - (1.0 - c).powf(gh)) * 255.0 + 0.5) as u8
                };
                px::premultiply(f(r), f(g), f(bl), a)
            }
            _ => match self.unsharp_params() {
                Some((amount, _, th)) => unsharp_px(o, b, amount, th),
                None => b,
            },
        }
    }

    pub fn pad(&self, level: u32) -> usize {
        use Filter::*;
        if self.class() != Class::Local {
            return 0;
        }
        match *self {
            BoxBlur { radius } => r_at(radius, level),
            Blur => 1,
            BlurMore => 2,
            MotionBlur { distance, .. } => (sc(distance, level) * 0.5).ceil() as usize + 2,
            LensBlur { radius, .. } => r_at(radius, level) + 1,
            SmartSharpen { radius, opts, .. } if !opts.is_plain() => {
                let blur = match opts.remove {
                    blur::Remove::Gaussian => blur::gauss_pad(sc(radius, level)),
                    blur::Remove::Lens => r_at(radius, level) + 1,
                    blur::Remove::Motion => r_at(radius, level),
                };
                let fade = |f: &blur::Fade| if f.amount > 0.0 { r_at(f.radius, level) } else { 0 };
                blur.max(fade(&opts.shadows)).max(fade(&opts.highlights))
            }
            SurfaceBlur { radius, .. } | SmartBlur { radius, .. } | Median { radius } | DustScratches { radius, .. } => r_at(radius, level),
            Despeckle => 1,
            ReduceNoise { .. } => 3,
            Maximum { radius } | Minimum { radius } => r_at(radius, level),
            FindEdges | TraceContour { .. } => 1,
            Emboss { height, .. } => sc(height, level).ceil() as usize + 1,
            Diffuse { radius } => sc(radius, level).ceil() as usize + 1,
            Wind { length, .. } => sc(length, level).ceil() as usize,
            OilPaint { radius, oil } => oil.pad(r_at(radius, level), level),
            Facet => 2,
            Mosaic { cell } => sc(cell, level).ceil() as usize + 1,
            Crystallize { cell } | Pointillize { cell, .. } => (sc(cell, level) * 2.0).ceil() as usize + 2,
            ColorHalftone { radius } => (sc(radius, level) * 3.0).ceil() as usize + 2,
            Fragment => sc(4.0, level).ceil() as usize + 1,
            Gallery { g, p, .. } => g.pad(p, level),
            Lighting(_) => 3,
            Displace { hs, vs, .. } => (sc(hs.abs().max(vs.abs()), level)).ceil() as usize + 6,
            FieldBlur { pins, .. } => blur::gauss_pad(sc(pins.iter().map(|p| p[2]).fold(0.0, f32::max), level)) + 1,
            SkinSmoothing { radius, .. } => r_at(radius, level).max(1) + 1,
            JpegCleanup { .. } => 4,
            PhotoRestoration { .. } => 10,
            ShapeBlur { radius, .. } => r_at(radius, level) + 1,
            TiltShift { blur, .. } | IrisBlur { blur, .. } => blur::gauss_pad(sc(blur, level)) + 1,
            Extrude { size, .. } => sc(size, level).ceil() as usize + 2,
            Tiles { size, offset, .. } => (sc(size * offset / 100.0, level)).ceil() as usize + 2,
            Custom { .. } | DeInterlace { .. } => 1,
            CameraRaw(r) => {
                if r.clarity != 0.0 || r.dehaze != 0.0 {
                    blur::gauss_pad(sc(extra::RAW_CLARITY_RADIUS, level)) + 1
                } else {
                    blur::gauss_pad(sc(2.0, level).max(0.5)) + 2
                }
            }
            _ => self.blur_sigma().map_or(0, |s| blur::gauss_pad(sc(s, level))),
        }
    }

    /// Per-channel LUTs of the LUT-defined adjustments. The CPU and GPU paths
    /// both apply exactly these tables.
    pub fn point_luts(&self) -> Option<[[u8; 256]; 3]> {
        use Filter::*;
        let one = |l: [u8; 256]| Some([l, l, l]);
        match self {
            BrightnessContrast { brightness, contrast } => {
                let (b, c) = (brightness / 100.0, contrast / 100.0);
                let k = if c >= 0.0 { 1.0 / (1.0 - c * 0.99) } else { 1.0 + c };
                one(lut_from(|v| (v + b * 0.5 - 0.5) * k + 0.5))
            }
            Levels { black, white, gamma } => {
                let (b, wt) = (black / 255.0, (white / 255.0).max(black / 255.0 + 1.0 / 255.0));
                let g = gamma.max(0.01);
                one(lut_from(|v| ((v - b) / (wt - b)).clamp(0.0, 1.0).powf(1.0 / g)))
            }
            Curves { pts } => one(curve_lut(pts)),
            Exposure { exposure, offset, gamma } => {
                let k = 2f32.powf(*exposure);
                let g = gamma.max(0.01);
                one(lut_from(|v| {
                    let lin = srgb_to_lin(v) * k + offset;
                    lin_to_srgb(lin.max(0.0).powf(1.0 / g))
                }))
            }
            Invert => one(lut_from(|v| 1.0 - v)),
            Solarize => one(lut_from(|v| if v > 0.5 { 1.0 - v } else { v })),
            Posterize { levels } => {
                let n = levels.round().clamp(2.0, 255.0) - 1.0;
                one(lut_from(|v| (v * n).round() / n))
            }
            Lut { lut, .. } => Some(**lut),
            _ => None,
        }
    }

    /// Applies a Point or Local filter in place. For Local filters `buf` is
    /// padded by `pad(level)` and only its center must come out right.
    /// `origin` is the level-space coordinate of `buf[0]`.
    pub fn apply(&self, buf: &mut [u32], w: usize, h: usize, level: u32, origin: (i32, i32)) {
        use Filter::*;
        let s = (1u32 << level) as f32;
        match self {
            BrightnessContrast { .. } | Levels { .. } | Curves { .. } | Exposure { .. } | Invert | Solarize | Posterize { .. } | Lut { .. } => {
                apply_lut3(buf, &self.point_luts().expect("LUT filter"));
            }
            Threshold { level } => {
                let t = *level;
                map_straight(buf, |c| {
                    let v = if threshold_luma(c) >= t { 255 } else { 0 };
                    [v, v, v]
                });
            }
            Desaturate => map_straight(buf, |c| {
                let l = ((c[0] as u32 * 77 + c[1] as u32 * 150 + c[2] as u32 * 29 + 128) >> 8) as u8;
                [l, l, l]
            }),
            Color { lut, .. } => lut.apply(buf),
            Average { color } => {
                for p in buf.iter_mut() {
                    *p = *color;
                }
            }
            AddNoise { amount, mono } => {
                let amt = amount / 100.0 * 255.0;
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let p = buf[i];
                        let a = px::alpha(p) as f32;
                        if a == 0.0 {
                            continue;
                        }
                        let (gx, gy) = (origin.0 + x as i32, origin.1 + y as i32);
                        let n = |c: u32| (hash01(gx, gy, c) + hash01(gx, gy, c + 100) - 1.0) * amt;
                        let n0 = n(0);
                        let (n1, n2) = if *mono { (n0, n0) } else { (n(1), n(2)) };
                        let [r, g, b, a8] = unpack(p);
                        let f = a / 255.0;
                        let ch = |v: u8, n: f32| (v as f32 + n * f).round().clamp(0.0, a) as u8;
                        buf[i] = pack(ch(r, n0), ch(g, n1), ch(b, n2), a8);
                    }
                }
            }
            Mezzotint => stylize::mezzotint(buf, origin, w),
            Gradient(g) => g.apply(buf, w, h, level, origin),
            Clouds { c0, c1, scale, difference } => {
                let c0p = pack(c0[0], c0[1], c0[2], 255);
                let c1p = pack(c1[0], c1[1], c1[2], 255);
                for y in 0..h {
                    for x in 0..w {
                        let gx = (origin.0 as f32 + x as f32 + 0.5) * s;
                        let gy = (origin.1 as f32 + y as f32 + 0.5) * s;
                        let v = (fbm(gx / scale, gy / scale, 7, 1) - 0.5) * 1.6 + 0.5;
                        let c = px::lerp(c0p, c1p, (v.clamp(0.0, 1.0) * 255.0) as u32);
                        let i = y * w + x;
                        buf[i] = if *difference {
                            let (o, cc) = (unpack(buf[i]), unpack(c));
                            let a = o[3];
                            let d = |k: usize| (o[k] as i32 - cc[k] as i32).unsigned_abs().min(a as u32) as u8;
                            pack(d(0), d(1), d(2), a)
                        } else {
                            c
                        };
                    }
                }
            }
            Fibers { c0, c1, variance, strength } => {
                let c0p = pack(c0[0], c0[1], c0[2], 255);
                let c1p = pack(c1[0], c1[1], c1[2], 255);
                let sx = 2.0 + variance * 0.2;
                let sy = 40.0 + strength * 4.0;
                for y in 0..h {
                    for x in 0..w {
                        let gx = (origin.0 as f32 + x as f32) * s;
                        let gy = (origin.1 as f32 + y as f32) * s;
                        let v = fbm(gx / sx, gy / sy, 4, 9);
                        let v = ((v - 0.5) * 2.2 + 0.5).clamp(0.0, 1.0);
                        buf[y * w + x] = px::lerp(c0p, c1p, (v * 255.0) as u32);
                    }
                }
            }
            LensFlare { x: fx, y: fy, brightness, size } => {
                let k = brightness / 100.0;
                for y in 0..h {
                    for x in 0..w {
                        let i = y * w + x;
                        let a = px::alpha(buf[i]);
                        if a == 0 {
                            continue;
                        }
                        let gx = (origin.0 as f32 + x as f32 + 0.5) * s;
                        let gy = (origin.1 as f32 + y as f32 + 0.5) * s;
                        let d = ((gx - fx).powi(2) + (gy - fy).powi(2)).sqrt() / size;
                        let glow = (-d * d * 18.0).exp() * 1.4 + (-d * 3.0).exp() * 0.35;
                        let ring = (-((d - 0.18) * 60.0).powi(2)).exp() * 0.25;
                        let add = [(glow + ring * 0.8) * k, (glow * 0.92 + ring) * k, (glow * 0.8 + ring * 1.2) * k];
                        let [r, g, b, a8] = unpack(buf[i]);
                        let ch = |v: u8, ad: f32| (v as f32 + ad * a as f32).min(a as f32) as u8;
                        buf[i] = pack(ch(r, add[0]), ch(g, add[1]), ch(b, add[2]), a8);
                    }
                }
            }
            // ----- local -----
            GaussianBlur { radius } => fast::blur_combine(buf, w, h, sc(*radius, level), None, fast::Combine::Blur),
            BoxBlur { radius } => {
                let r = r_at(*radius, level);
                fast::blur_combine(buf, w, h, 0.0, Some(&[2 * r + 1]), fast::Combine::Blur);
            }
            Blur => fast::conv_sep(buf, w, h, &[64, 128, 64]),
            BlurMore => fast::conv_sep(buf, w, h, &[16, 64, 96, 64, 16]),
            MotionBlur { angle, distance } => blur::motion(buf, w, h, *angle, sc(*distance, level)),
            LensBlur { radius, bloom, opts } => {
                let r = sc(*radius, level);
                if opts.is_plain() {
                    if r >= 0.5 {
                        blur::lens(buf, w, h, r, *bloom);
                    }
                } else {
                    let depth = match &opts.depth {
                        blur::Depth::None => None,
                        blur::Depth::Transparency => Some(buf.iter().map(|&p| (p >> 24) as u8).collect::<Vec<u8>>()),
                        blur::Depth::Mask(m) => Some(m.read(level, origin.0, origin.1, w, h)),
                    };
                    blur::lens_ex(buf, w, h, r, *bloom, opts, depth.as_deref(), origin);
                }
            }
            SurfaceBlur { radius, threshold } => rank::surface(buf, w, h, r_at(*radius, level), *threshold, false),
            SmartBlur { radius, threshold } => rank::surface(buf, w, h, r_at(*radius, level), *threshold, true),
            Median { radius } => rank::median(buf, w, h, r_at(*radius, level)),
            DustScratches { radius, threshold } => rank::dust_scratches(buf, w, h, r_at(*radius, level), *threshold as u8),
            Despeckle => rank::despeckle(buf, w, h),
            ReduceNoise { strength } => rank::surface(buf, w, h, 3, 4.0 + strength * 2.5, false),
            Maximum { radius } | Minimum { radius } => {
                let r = r_at(*radius, level);
                let max = matches!(self, Maximum { .. });
                if r == 0 {
                } else if r <= 4 {
                    fast::maxmin_small(buf, w, h, r, max);
                } else {
                    rank::max_min(buf, w, h, r, max);
                }
            }
            HighPass { radius } => fast::blur_combine(buf, w, h, sc(*radius, level), None, fast::Combine::HighPass),
            SmartSharpen { amount, radius, noise, opts } if !opts.is_plain() => {
                let fr = (r_at(opts.shadows.radius, level), r_at(opts.highlights.radius, level));
                blur::smart_sharpen(buf, w, h, *amount, sc(*radius, level), *noise, opts, fr);
            }
            UnsharpMask { .. } | Sharpen | SharpenMore | SharpenEdges | SmartSharpen { .. } => {
                let (amount, radius, th) = self.unsharp_params().unwrap();
                let mode = fast::Combine::Unsharp { k: (amount / 100.0 * 256.0).round() as i32, threshold: th.round() as i32 };
                fast::blur_combine(buf, w, h, sc(radius, level), None, mode);
            }
            ShadowsHighlights { .. } => {
                let orig = buf.to_vec();
                if let Some(sigma) = self.blur_sigma() {
                    blur::gaussian(buf, w, h, sc(sigma, level));
                }
                let comb = self.combiner();
                for (b, &o) in buf.iter_mut().zip(&orig) {
                    *b = comb.apply(o, *b);
                }
            }
            FindEdges => stylize::find_edges(buf, w, h),
            Emboss { angle, height, amount } => stylize::emboss(buf, w, h, *angle, sc(*height, level).max(1.0), *amount),
            TraceContour { level: l } => stylize::trace_contour(buf, w, h, *l as u8),
            Diffuse { radius } => stylize::diffuse(buf, w, h, sc(*radius, level), origin),
            Wind { length, from_left } => stylize::wind(buf, w, h, sc(*length, level), *from_left),
            OilPaint { radius, oil } => stylize::oil_paint(buf, w, h, r_at(*radius, level), oil, level, origin),
            Facet => stylize::kuwahara(buf, w, h, 2),
            Mosaic { cell } => stylize::mosaic(buf, w, h, sc(*cell, level), origin),
            Crystallize { cell } => stylize::cells(buf, w, h, *cell, level, origin, None),
            Pointillize { cell, bg } => stylize::cells(buf, w, h, *cell, level, origin, Some(pack(bg[0], bg[1], bg[2], 255))),
            ColorHalftone { radius } => stylize::halftone(buf, w, h, *radius, level, origin),
            Fragment => stylize::fragment(buf, w, h, sc(4.0, level)),
            Gallery { g, p, fg, bg, doc } => g.apply_in(*p, *fg, *bg, buf, w, h, level, origin, Some(*doc)),
            Lighting(lt) => extra::lighting(buf, w, h, level, origin, lt),
            Displace { hs, vs, map } => extra::displace(buf, w, h, level, origin, *hs, *vs, map),
            FieldBlur { pins, fx } => {
                let mx = sc(pins.iter().map(|p| p[2]).fold(0.0, f32::max), level);
                let pins = *pins;
                let prep = fx.prep();
                let at = extra::FxAt { fx: &prep, origin, scale: s };
                extra::variable_blur(buf, w, h, mx, Some(at), |x, y| {
                    let gx = (origin.0 as f32 + x as f32 + 0.5) * s;
                    let gy = (origin.1 as f32 + y as f32 + 0.5) * s;
                    // Inverse-distance blend of the pins' blur amounts.
                    let (mut num, mut den) = (0.0, 0.0);
                    for p in &pins {
                        let d2 = (gx - p[0]).powi(2) + (gy - p[1]).powi(2) + 1.0;
                        let wt = 1.0 / d2;
                        num += wt * p[2];
                        den += wt;
                    }
                    sc(num / den, level)
                });
            }
            SkinSmoothing { radius, smoothness } => extra::skin_smoothing(buf, w, h, r_at(*radius, level).max(1), *smoothness),
            JpegCleanup { strength } => extra::jpeg_cleanup(buf, w, h, origin, level, *strength),
            PhotoRestoration { strength, scratch } => extra::restore(buf, w, h, *strength, *scratch),
            SpinBlur { .. } | PathBlur { .. } => {}
            ShapeBlur { radius, shape } => {
                let r = sc(*radius, level);
                if r >= 0.5 {
                    extra::shape_blur(buf, w, h, r, *shape);
                }
            }
            TiltShift { cy, band, transition, blur, fx } => {
                let (cy, band, tr) = (*cy, *band, transition.max(1.0));
                let mx = sc(*blur, level);
                let prep = fx.prep();
                let at = extra::FxAt { fx: &prep, origin, scale: s };
                extra::variable_blur(buf, w, h, mx, Some(at), |_, y| {
                    let gy = (origin.1 as f32 + y as f32 + 0.5) * s;
                    let d = ((gy - cy).abs() - band).max(0.0);
                    mx * (d / tr).min(1.0)
                });
            }
            IrisBlur { cx, cy, radius, blur, fx } => {
                let mx = sc(*blur, level);
                let (cx, cy, rad) = (*cx, *cy, radius.max(1.0));
                let prep = fx.prep();
                let at = extra::FxAt { fx: &prep, origin, scale: s };
                extra::variable_blur(buf, w, h, mx, Some(at), |x, y| {
                    let gx = (origin.0 as f32 + x as f32 + 0.5) * s;
                    let gy = (origin.1 as f32 + y as f32 + 0.5) * s;
                    let d = ((gx - cx).hypot(gy - cy) / rad - 0.6).max(0.0) / 0.6;
                    mx * d.min(1.0)
                });
            }
            Extrude { size, depth, pyramids } => extra::extrude(buf, w, h, level, origin, *size, *depth, *pyramids),
            Tiles { size, offset, fill } => extra::tiles(buf, w, h, level, origin, *size, *offset, *fill),
            Custom { k, scale, offset } => extra::custom3(buf, w, h, *k, *scale, *offset),
            DeInterlace { odd, interpolate } => extra::deinterlace(buf, w, h, origin, *odd, *interpolate),
            CameraRaw(r) => extra::camera_raw(buf, w, h, level, origin, r),
            Warp { .. } | RadialBlur { .. } | LensCorrection { .. } => {}
        }
    }

    /// Renders any filter class into `out` for a region of a level grid.
    #[allow(clippy::too_many_arguments)]
    pub fn render(&self, src: &Src, level: u32, x0: i32, y0: i32, w: usize, h: usize, out: &mut [u32], scratch: &mut Vec<u32>) {
        let grid = src.at(level);
        match self.class() {
            Class::Warp => {
                if let Filter::Warp { warp, .. } = self {
                    distort::render_warp(warp, grid, level, x0, y0, w, h, out);
                }
            }
            Class::Gather => match *self {
                Filter::RadialBlur { amount, zoom, cx, cy } => distort::render_radial(amount, zoom, cx, cy, src, level, x0, y0, w, h, out),
                Filter::LensCorrection { distortion, ca_rc, ca_by, vignette, w: iw, h: ih } => {
                    extra::render_lens_correction(distortion, ca_rc, ca_by, vignette, iw, ih, src, level, x0, y0, w, h, out)
                }
                Filter::PathBlur { x0: px0, y0: py0, x1: px1, y1: py1, bend, length, ref fx } => {
                    let prep = fx.prep();
                    if prep.active() {
                        // Effects: the whole image is blurred (blur amount 1).
                        let s = (1u32 << level) as f32;
                        let mut orig = vec![0u32; w * h];
                        grid.read_region(x0, y0, w, h, &mut orig);
                        distort::render_path_blur_with((px0, py0), (px1, py1), bend, length, src, level, x0, y0, w, h, out, |x, y, acc, n| {
                            let (gx, gy) = ((x0 as f32 + x as f32 + 0.5) * s, (y0 as f32 + y as f32 + 0.5) * s);
                            prep.apply_avg(acc, n, orig[y * w + x], 1.0, gx, gy)
                        })
                    } else {
                        distort::render_path_blur((px0, py0), (px1, py1), bend, length, src, level, x0, y0, w, h, out)
                    }
                }
                Filter::SpinBlur { cx, cy, rx, ry, amount, ref fx } => {
                    // Spin inside an ellipse, fading out at its rim; blocks
                    // entirely outside the ellipse are a plain copy.
                    let s = (1u32 << level) as f32;
                    let (bx0, by0) = (x0 as f32 * s, y0 as f32 * s);
                    let (bx1, by1) = ((x0 + w as i32) as f32 * s, (y0 + h as i32) as f32 * s);
                    let mut orig = vec![0u32; w * h];
                    grid.read_region(x0, y0, w, h, &mut orig);
                    if bx1 < cx - rx || bx0 > cx + rx || by1 < cy - ry || by0 > cy + ry {
                        out.copy_from_slice(&orig);
                        return;
                    }
                    let prep = fx.prep();
                    if prep.active() {
                        // Effects on the spun pixels (blur amount 1), which
                        // the rim mask below then fades like the blur.
                        distort::render_radial_with(amount, false, cx, cy, src, level, x0, y0, w, h, out, |x, y, acc, n| {
                            let (gx, gy) = ((x0 as f32 + x as f32 + 0.5) * s, (y0 as f32 + y as f32 + 0.5) * s);
                            prep.apply_avg(acc, n, orig[y * w + x], 1.0, gx, gy)
                        });
                    } else {
                        distort::render_radial(amount, false, cx, cy, src, level, x0, y0, w, h, out);
                    }
                    for yy in 0..h {
                        for xx in 0..w {
                            let gx = (x0 as f32 + xx as f32 + 0.5) * s;
                            let gy = (y0 as f32 + yy as f32 + 0.5) * s;
                            let d = ((gx - cx) / rx.max(1.0)).hypot((gy - cy) / ry.max(1.0));
                            let m = (1.0 - ((d - 0.85) / 0.15).clamp(0.0, 1.0)) * 255.0;
                            let i = yy * w + xx;
                            out[i] = px::lerp(orig[i], out[i], m as u32);
                        }
                    }
                }
                _ => {}
            },
            Class::Point => {
                grid.read_region(x0, y0, w, h, out);
                self.apply(out, w, h, level, (x0, y0));
            }
            Class::Local => {
                let pad = self.pad(level);
                let (pw, ph) = (w + 2 * pad, h + 2 * pad);
                scratch.resize(pw * ph, 0);
                let (px0, py0) = (x0 - pad as i32, y0 - pad as i32);
                grid.read_region_clamped(px0, py0, pw, ph, scratch);
                self.apply(scratch, pw, ph, level, (px0, py0));
                for y in 0..h {
                    out[y * w..(y + 1) * w].copy_from_slice(&scratch[(y + pad) * pw + pad..(y + pad) * pw + pad + w]);
                }
            }
        }
    }
}

pub struct Combiner<'a> {
    f: &'a Filter,
    table: Option<Box<[u8]>>,
}

impl Combiner<'_> {
    /// The baked 256 x 256 table (Shadows/Highlights), indexed by
    /// `blurred luma * 256 + straight channel`.
    pub fn table(&self) -> Option<&[u8]> {
        self.table.as_deref()
    }

    #[inline]
    pub fn apply(&self, o: u32, b: u32) -> u32 {
        match &self.table {
            Some(t) => {
                if px::alpha(o) == 0 {
                    return o;
                }
                let [r, g, bl, a] = px::unpremultiply(o);
                let [br, bg, bb, _] = px::unpremultiply(b);
                let m = ((br as u32 * 77 + bg as u32 * 150 + bb as u32 * 29 + 128) >> 8) as usize * 256;
                px::premultiply(t[m + r as usize], t[m + g as usize], t[m + bl as usize], a)
            }
            None => self.f.combine(o, b),
        }
    }
}

/// Unsharp-mask combine of an original and a blurred premultiplied pixel.
#[inline]
pub fn unsharp_px(o: u32, b: u32, amount: f32, threshold: f32) -> u32 {
    let ob = unpack(o);
    let bb = unpack(b);
    let a = ob[3] as i32;
    let k = amount / 100.0;
    let th = threshold as i32;
    let mut out = [0u8; 4];
    for c in 0..3 {
        let d = ob[c] as i32 - bb[c] as i32;
        let v = if d.abs() < th { ob[c] as i32 } else { ob[c] as i32 + (d as f32 * k).round() as i32 };
        out[c] = v.clamp(0, a) as u8;
    }
    out[3] = ob[3];
    u32::from_le_bytes(out)
}

/// Threshold's luma (float, as the CPU has always computed it).
#[inline]
pub fn threshold_luma(c: [u8; 3]) -> f32 {
    0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32
}

/// For every (r, g): the smallest b with `threshold_luma >= t` (256 if
/// none). Luma is monotone in b, so this tabulates Threshold exactly.
pub fn threshold_cut(t: f32) -> Vec<u16> {
    (0..65536u32)
        .map(|k| {
            let (r, g) = ((k >> 8) as u8, (k & 255) as u8);
            let (mut lo, mut hi) = (0u32, 256u32);
            while lo < hi {
                let m = (lo + hi) / 2;
                if threshold_luma([r, g, m as u8]) >= t { hi = m } else { lo = m + 1 }
            }
            lo as u16
        })
        .collect()
}

fn srgb_to_lin(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn lin_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Monotone cubic (Fritsch-Carlson) through five evenly spaced points.
pub fn curve_lut(pts: &[f32; 5]) -> [u8; 256] {
    let xs = [0.0f32, 0.25, 0.5, 0.75, 1.0];
    let ys: Vec<f32> = pts.iter().map(|v| v / 255.0).collect();
    let n = 5;
    let mut d = [0f32; 4];
    for i in 0..4 {
        d[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    }
    let mut m = [0f32; 5];
    m[0] = d[0];
    m[4] = d[3];
    for i in 1..n - 1 {
        m[i] = if d[i - 1] * d[i] <= 0.0 { 0.0 } else { (d[i - 1] + d[i]) * 0.5 };
    }
    for i in 0..4 {
        if d[i] == 0.0 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let a = m[i] / d[i];
            let b = m[i + 1] / d[i];
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                m[i] = t * a * d[i];
                m[i + 1] = t * b * d[i];
            }
        }
    }
    lut_from(|x| {
        let i = ((x * 4.0) as usize).min(3);
        let hh = xs[i + 1] - xs[i];
        let t = (x - xs[i]) / hh;
        let (t2, t3) = (t * t, t * t * t);
        ys[i] * (2.0 * t3 - 3.0 * t2 + 1.0) + hh * m[i] * (t3 - 2.0 * t2 + t) + ys[i + 1] * (-2.0 * t3 + 3.0 * t2) + hh * m[i + 1] * (t3 - t2)
    })
}

// ----- builders for baked color transforms --------------------------------------

pub fn hue_saturation(hue: f32, sat: f32, light: f32) -> Filter {
    let (dh, ds, dl) = (hue / 360.0, sat / 100.0, light / 100.0);
    Filter::Color {
        name: "Hue/Saturation",
        lut: Arc::new(Lut3d::bake(|c| {
            let (h, s, l) = rgb_to_hsl(c);
            let s = if ds >= 0.0 { s + (1.0 - s) * ds } else { s * (1.0 + ds) };
            let l = if dl >= 0.0 { l + (1.0 - l) * dl } else { l * (1.0 + dl) };
            hsl_to_rgb((h + dh).rem_euclid(1.0), s.clamp(0.0, 1.0), l.clamp(0.0, 1.0))
        })),
    }
}

pub fn vibrance(vib: f32, sat: f32) -> Filter {
    let (v, s) = (vib / 100.0, sat / 100.0);
    Filter::Color {
        name: "Vibrance",
        lut: Arc::new(Lut3d::bake(|c| {
            let l = luma(c);
            let mx = c[0].max(c[1]).max(c[2]);
            let mn = c[0].min(c[1]).min(c[2]);
            let cur = mx - mn;
            // Vibrance boosts muted colors more than saturated ones.
            let k = 1.0 + s + v * (1.0 - cur) * 1.5;
            [l + (c[0] - l) * k, l + (c[1] - l) * k, l + (c[2] - l) * k]
        })),
    }
}

pub fn color_balance(cr: f32, mg: f32, yb: f32, preserve: bool) -> Filter {
    let d = [cr / 100.0, mg / 100.0, yb / 100.0];
    Filter::Color {
        name: "Color Balance",
        lut: Arc::new(Lut3d::bake(|c| {
            let mut o = [0f32; 3];
            for k in 0..3 {
                let mid = 4.0 * c[k] * (1.0 - c[k]);
                o[k] = c[k] + d[k] * 0.35 * mid;
            }
            if preserve {
                let (l0, l1) = (luma(c), luma(o).max(1e-5));
                for v in &mut o {
                    *v *= l0 / l1;
                }
            }
            o
        })),
    }
}

/// Photoshop-style Black & White: gray = min + (mid-min)*w(secondary) + (max-mid)*w(primary).
pub fn black_white(w: [f32; 6]) -> Filter {
    // Weights: reds, yellows, greens, cyans, blues, magentas (percent).
    Filter::Color {
        name: "Black & White",
        lut: Arc::new(Lut3d::bake(move |c| {
            let mut idx = [0usize, 1, 2];
            idx.sort_by(|&a, &b| c[b].total_cmp(&c[a]));
            let (mx, md, mn) = (c[idx[0]], c[idx[1]], c[idx[2]]);
            let primary = [0usize, 2, 4][idx[0]];
            let secondary = match (idx[0], idx[1]) {
                (0, 1) | (1, 0) => 1,
                (1, 2) | (2, 1) => 3,
                _ => 5,
            };
            let g = mn + (md - mn) * w[secondary] / 100.0 + (mx - md) * w[primary] / 100.0;
            [g, g, g]
        })),
    }
}

pub fn photo_filter(color: [u8; 3], density: f32, preserve: bool) -> Filter {
    let f = [color[0] as f32 / 255.0, color[1] as f32 / 255.0, color[2] as f32 / 255.0];
    let k = density / 100.0;
    Filter::Color {
        name: "Photo Filter",
        lut: Arc::new(Lut3d::bake(move |c| {
            let mut o = [0f32; 3];
            for i in 0..3 {
                o[i] = c[i] + (c[i] * f[i] - c[i]) * k;
            }
            if preserve {
                let (l0, l1) = (luma(c), luma(o).max(1e-5));
                for v in &mut o {
                    *v *= l0 / l1;
                }
            }
            o
        })),
    }
}

pub fn gradient_map(c0: [u8; 3], c1: [u8; 3]) -> Filter {
    Filter::Color {
        name: "Gradient Map",
        lut: Arc::new(Lut3d::bake(move |c| {
            let l = luma(c);
            [0, 1, 2].map(|k| (c0[k] as f32 + (c1[k] as f32 - c0[k] as f32) * l) / 255.0)
        })),
    }
}

pub fn channel_swap_hsl() -> Filter {
    Filter::Color {
        name: "HSB/HSL",
        lut: Arc::new(Lut3d::bake(|c| {
            let (h, s, l) = rgb_to_hsl(c);
            [h, s, l]
        })),
    }
}

// ----- statistics-driven adjustments -------------------------------------------------

/// Channel histograms sampled from the coarsest mip with ≥1 MP (or level 0).
pub struct Stats {
    pub hist: [[u64; 256]; 3],
    pub sum: [f64; 4],
    pub count: u64,
}

pub fn stats(layer: &Layer) -> Stats {
    let mut l = layer.mips.len() as u32;
    while l > 0 {
        let g = layer.level(l);
        if g.w as u64 * g.h as u64 >= 1_000_000 {
            break;
        }
        l -= 1;
    }
    let g = layer.level(l);
    use rayon::prelude::*;
    let cols = g.cols as usize;
    g.tiles
        .par_iter()
        .enumerate()
        .fold(
            || Stats { hist: [[0; 256]; 3], sum: [0.0; 4], count: 0 },
            |mut st, (i, t)| {
                let Some(t) = t else { return st };
                let (tx, ty) = ((i % cols) as u32 * 256, (i / cols) as u32 * 256);
                let (cw, ch) = ((g.w - tx).min(256) as usize, (g.h - ty).min(256) as usize);
                for y in 0..ch {
                    for &p in &t.read()[y * 256..y * 256 + cw] {
                        if px::alpha(p) < 8 {
                            continue;
                        }
                        let c = px::unpremultiply(p);
                        for k in 0..3 {
                            st.hist[k][c[k] as usize] += 1;
                            st.sum[k] += c[k] as f64;
                        }
                        st.sum[3] += c[3] as f64;
                        st.count += 1;
                    }
                }
                st
            },
        )
        .reduce(
            || Stats { hist: [[0; 256]; 3], sum: [0.0; 4], count: 0 },
            |mut a, b| {
                for k in 0..3 {
                    for v in 0..256 {
                        a.hist[k][v] += b.hist[k][v];
                    }
                }
                for k in 0..4 {
                    a.sum[k] += b.sum[k];
                }
                a.count += b.count;
                a
            },
        )
}

fn clip_points(h: &[u64; 256], total: u64, clip: f64) -> (u8, u8) {
    let lim = (total as f64 * clip) as u64;
    let mut acc = 0;
    let mut lo = 0u8;
    for (v, &n) in h.iter().enumerate() {
        acc += n;
        if acc > lim {
            lo = v as u8;
            break;
        }
    }
    acc = 0;
    let mut hi = 255u8;
    for (v, &n) in h.iter().enumerate().rev() {
        acc += n;
        if acc > lim {
            hi = v as u8;
            break;
        }
    }
    (lo, hi.max(lo.saturating_add(1)))
}

fn stretch(lo: u8, hi: u8) -> [u8; 256] {
    lut_from(|v| (v * 255.0 - lo as f32) / (hi as f32 - lo as f32))
}

pub fn equalize(st: &Stats) -> Filter {
    let mut all = [0u64; 256];
    for k in 0..3 {
        for v in 0..256 {
            all[v] += st.hist[k][v];
        }
    }
    let total: u64 = all.iter().sum::<u64>().max(1);
    let mut lut = [0u8; 256];
    let mut acc = 0;
    for v in 0..256 {
        acc += all[v];
        lut[v] = ((acc as f64 / total as f64) * 255.0).round() as u8;
    }
    Filter::Lut { name: "Equalize", lut: Arc::new([lut; 3]) }
}

pub fn auto_tone(st: &Stats) -> Filter {
    let n = st.count.max(1);
    let luts = [0, 1, 2].map(|k| {
        let (lo, hi) = clip_points(&st.hist[k], n, 0.001);
        stretch(lo, hi)
    });
    Filter::Lut { name: "Auto Tone", lut: Arc::new(luts) }
}

pub fn auto_contrast(st: &Stats) -> Filter {
    let mut all = [0u64; 256];
    for k in 0..3 {
        for v in 0..256 {
            all[v] += st.hist[k][v];
        }
    }
    let (lo, hi) = clip_points(&all, st.count.max(1) * 3, 0.001);
    let l = stretch(lo, hi);
    Filter::Lut { name: "Auto Contrast", lut: Arc::new([l; 3]) }
}

pub fn average(st: &Stats) -> Filter {
    let n = st.count.max(1) as f64;
    let c = [0, 1, 2].map(|k| (st.sum[k] / n).round() as u8);
    Filter::Average { color: pack(c[0], c[1], c[2], 255) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_curve() {
        let l = curve_lut(&[0.0, 63.75, 127.5, 191.25, 255.0]);
        for v in 0..256 {
            assert!((l[v] as i32 - v as i32).abs() <= 1);
        }
    }

    #[test]
    fn every_local_filter_keeps_size_and_alpha() {
        // Smoke-test all local filters on a small random buffer at levels 0 and 2.
        let list = vec![
            Filter::GaussianBlur { radius: 3.0 },
            Filter::BoxBlur { radius: 3.0 },
            Filter::Blur,
            Filter::BlurMore,
            Filter::MotionBlur { angle: 37.0, distance: 12.0 },
            Filter::LensBlur { radius: 4.0, bloom: 20.0, opts: Default::default() },
            Filter::SurfaceBlur { radius: 3.0, threshold: 20.0 },
            Filter::SmartBlur { radius: 3.0, threshold: 20.0 },
            Filter::Median { radius: 2.0 },
            Filter::DustScratches { radius: 2.0, threshold: 10.0 },
            Filter::Despeckle,
            Filter::ReduceNoise { strength: 5.0 },
            Filter::Maximum { radius: 2.0 },
            Filter::Minimum { radius: 2.0 },
            Filter::UnsharpMask { amount: 100.0, radius: 2.0, threshold: 0.0 },
            Filter::Sharpen,
            Filter::SharpenMore,
            Filter::SharpenEdges,
            Filter::SmartSharpen { amount: 100.0, radius: 1.5, noise: 5.0, opts: Default::default() },
            Filter::HighPass { radius: 4.0 },
            Filter::ShadowsHighlights { shadows: 35.0, highlights: 20.0, radius: 10.0 },
            Filter::FindEdges,
            Filter::Emboss { angle: 135.0, height: 3.0, amount: 100.0 },
            Filter::TraceContour { level: 128.0 },
            Filter::Diffuse { radius: 2.0 },
            Filter::Wind { length: 20.0, from_left: true },
            Filter::OilPaint { radius: 3.0, oil: Default::default() },
            Filter::OilPaint { radius: 3.0, oil: stylize::Oil { stylization: 4.0, bristle: 3.0, shine: 2.0, ..Default::default() } },
            Filter::Facet,
            Filter::Mosaic { cell: 8.0 },
            Filter::Crystallize { cell: 10.0 },
            Filter::Pointillize { cell: 10.0, bg: [255, 255, 255] },
            Filter::ColorHalftone { radius: 4.0 },
            Filter::Fragment,
        ];
        let src = super::super::ops::procedural(300, 200);
        for f in &list {
            assert_eq!(f.class(), Class::Local, "{}", f.name());
            for level in [0u32, 2] {
                let mut out = vec![0u32; 64 * 64];
                let mut scratch = Vec::new();
                f.render(&Src::single(&src), level, 10, 10, 64, 64, &mut out, &mut scratch);
                assert!(out.iter().all(|&p| px::alpha(p) == 255), "{} level {level}", f.name());
            }
        }
    }
}
