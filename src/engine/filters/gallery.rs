//! Filter Gallery: Artistic, Brush Strokes, Distort, Sketch, Stylize and
//! Texture effects.
//!
//! Each effect is a short composition of shared building blocks that work on
//! straight-alpha f32 planes of the padded buffer: Gaussian blur, Sobel
//! gradients, relief shading, posterization, streak noise and procedural
//! textures. All spatial parameters are in level-0 pixels and are scaled by
//! the mip level, and every random/texture lookup uses global coordinates,
//! so the live preview matches the full-resolution result.

use super::util::hash01;
use crate::engine::px;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum G {
    // Artistic
    ColoredPencil,
    Cutout,
    DryBrush,
    FilmGrain,
    Fresco,
    NeonGlow,
    PaintDaubs,
    PaletteKnife,
    PlasticWrap,
    PosterEdges,
    RoughPastels,
    SmudgeStick,
    Sponge,
    Underpainting,
    Watercolor,
    // Brush Strokes
    AccentedEdges,
    AngledStrokes,
    Crosshatch,
    DarkStrokes,
    InkOutlines,
    Spatter,
    SprayedStrokes,
    SumiE,
    // Distort
    DiffuseGlow,
    Glass,
    OceanRipple,
    // Sketch
    BasRelief,
    ChalkCharcoal,
    Charcoal,
    Chrome,
    ConteCrayon,
    GraphicPen,
    HalftonePattern,
    NotePaper,
    Photocopy,
    Plaster,
    Reticulation,
    Stamp,
    TornEdges,
    WaterPaper,
    // Stylize
    GlowingEdges,
    // Texture
    Craquelure,
    Grain,
    MosaicTiles,
    Patchwork,
    StainedGlass,
    Texturizer,
}

pub struct GParam {
    pub label: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub choices: &'static [&'static str],
}

const fn gp(label: &'static str, min: f32, max: f32, default: f32) -> GParam {
    GParam { label, min, max, default, choices: &[] }
}

const fn gc(label: &'static str, choices: &'static [&'static str], default: f32) -> GParam {
    GParam { label, min: 0.0, max: (choices.len() - 1) as f32, default, choices }
}

const TEXTURES: &[&str] = &["Canvas", "Burlap", "Brick", "Sandstone"];

impl G {
    pub const ALL: [G; 47] = [
        G::ColoredPencil,
        G::Cutout,
        G::DryBrush,
        G::FilmGrain,
        G::Fresco,
        G::NeonGlow,
        G::PaintDaubs,
        G::PaletteKnife,
        G::PlasticWrap,
        G::PosterEdges,
        G::RoughPastels,
        G::SmudgeStick,
        G::Sponge,
        G::Underpainting,
        G::Watercolor,
        G::AccentedEdges,
        G::AngledStrokes,
        G::Crosshatch,
        G::DarkStrokes,
        G::InkOutlines,
        G::Spatter,
        G::SprayedStrokes,
        G::SumiE,
        G::DiffuseGlow,
        G::Glass,
        G::OceanRipple,
        G::BasRelief,
        G::ChalkCharcoal,
        G::Charcoal,
        G::Chrome,
        G::ConteCrayon,
        G::GraphicPen,
        G::HalftonePattern,
        G::NotePaper,
        G::Photocopy,
        G::Plaster,
        G::Reticulation,
        G::Stamp,
        G::TornEdges,
        G::WaterPaper,
        G::GlowingEdges,
        G::Craquelure,
        G::Grain,
        G::MosaicTiles,
        G::Patchwork,
        G::StainedGlass,
        G::Texturizer,
    ];

    pub fn group(self) -> &'static str {
        use G::*;
        match self {
            ColoredPencil | Cutout | DryBrush | FilmGrain | Fresco | NeonGlow | PaintDaubs | PaletteKnife | PlasticWrap
            | PosterEdges | RoughPastels | SmudgeStick | Sponge | Underpainting | Watercolor => "Artistic",
            AccentedEdges | AngledStrokes | Crosshatch | DarkStrokes | InkOutlines | Spatter | SprayedStrokes | SumiE => {
                "Brush Strokes"
            }
            DiffuseGlow | Glass | OceanRipple => "Distort",
            BasRelief | ChalkCharcoal | Charcoal | Chrome | ConteCrayon | GraphicPen | HalftonePattern | NotePaper
            | Photocopy | Plaster | Reticulation | Stamp | TornEdges | WaterPaper => "Sketch",
            GlowingEdges => "Stylize",
            Craquelure | Grain | MosaicTiles | Patchwork | StainedGlass | Texturizer => "Texture",
        }
    }

    pub const GROUPS: [&'static str; 6] = ["Artistic", "Brush Strokes", "Distort", "Sketch", "Stylize", "Texture"];

    pub fn title(self) -> &'static str {
        use G::*;
        match self {
            ColoredPencil => "Colored Pencil",
            Cutout => "Cutout",
            DryBrush => "Dry Brush",
            FilmGrain => "Film Grain",
            Fresco => "Fresco",
            NeonGlow => "Neon Glow",
            PaintDaubs => "Paint Daubs",
            PaletteKnife => "Palette Knife",
            PlasticWrap => "Plastic Wrap",
            PosterEdges => "Poster Edges",
            RoughPastels => "Rough Pastels",
            SmudgeStick => "Smudge Stick",
            Sponge => "Sponge",
            Underpainting => "Underpainting",
            Watercolor => "Watercolor",
            AccentedEdges => "Accented Edges",
            AngledStrokes => "Angled Strokes",
            Crosshatch => "Crosshatch",
            DarkStrokes => "Dark Strokes",
            InkOutlines => "Ink Outlines",
            Spatter => "Spatter",
            SprayedStrokes => "Sprayed Strokes",
            SumiE => "Sumi-e",
            DiffuseGlow => "Diffuse Glow",
            Glass => "Glass",
            OceanRipple => "Ocean Ripple",
            BasRelief => "Bas Relief",
            ChalkCharcoal => "Chalk & Charcoal",
            Charcoal => "Charcoal",
            Chrome => "Chrome",
            ConteCrayon => "Conté Crayon",
            GraphicPen => "Graphic Pen",
            HalftonePattern => "Halftone Pattern",
            NotePaper => "Note Paper",
            Photocopy => "Photocopy",
            Plaster => "Plaster",
            Reticulation => "Reticulation",
            Stamp => "Stamp",
            TornEdges => "Torn Edges",
            WaterPaper => "Water Paper",
            GlowingEdges => "Glowing Edges",
            Craquelure => "Craquelure",
            Grain => "Grain",
            MosaicTiles => "Mosaic Tiles",
            Patchwork => "Patchwork",
            StainedGlass => "Stained Glass",
            Texturizer => "Texturizer",
        }
    }

    /// Up to three parameters, Photoshop-style ranges.
    pub fn params(self) -> Vec<GParam> {
        use G::*;
        match self {
            ColoredPencil => vec![gp("Pencil width", 1., 24., 4.), gp("Stroke pressure", 0., 15., 8.), gp("Paper brightness", 0., 50., 25.)],
            Cutout => vec![gp("Number of levels", 2., 8., 4.), gp("Edge simplicity", 0., 10., 4.), gp("Edge fidelity", 1., 3., 2.)],
            DryBrush => vec![gp("Brush size", 0., 10., 2.), gp("Brush detail", 0., 10., 8.), gp("Texture", 1., 3., 1.)],
            FilmGrain => vec![gp("Grain", 0., 20., 4.), gp("Highlight area", 0., 20., 0.), gp("Intensity", 0., 10., 10.)],
            Fresco => vec![gp("Brush size", 0., 10., 2.), gp("Brush detail", 0., 10., 8.), gp("Texture", 1., 3., 1.)],
            NeonGlow => vec![gp("Glow size", -24., 24., 5.), gp("Glow brightness", 0., 50., 15.)],
            PaintDaubs => vec![gp("Brush size", 1., 50., 8.), gp("Sharpness", 0., 40., 7.)],
            PaletteKnife => vec![gp("Stroke size", 1., 50., 25.), gp("Stroke detail", 1., 3., 3.), gp("Softness", 0., 10., 0.)],
            PlasticWrap => vec![gp("Highlight strength", 0., 20., 15.), gp("Detail", 1., 15., 9.), gp("Smoothness", 1., 15., 7.)],
            PosterEdges => vec![gp("Edge thickness", 0., 10., 2.), gp("Edge intensity", 0., 10., 1.), gp("Posterization", 0., 6., 2.)],
            RoughPastels => vec![gp("Stroke length", 0., 40., 6.), gp("Stroke detail", 1., 20., 4.), gc("Texture", TEXTURES, 0.)],
            SmudgeStick => vec![gp("Stroke length", 0., 10., 2.), gp("Highlight area", 0., 20., 0.), gp("Intensity", 0., 10., 10.)],
            Sponge => vec![gp("Brush size", 0., 10., 2.), gp("Definition", 0., 25., 12.), gp("Smoothness", 1., 15., 5.)],
            Underpainting => vec![gp("Brush size", 0., 40., 6.), gp("Texture coverage", 0., 40., 16.), gc("Texture", TEXTURES, 0.)],
            Watercolor => vec![gp("Brush detail", 1., 14., 9.), gp("Shadow intensity", 0., 10., 1.), gp("Texture", 1., 3., 1.)],
            AccentedEdges => vec![gp("Edge width", 1., 14., 2.), gp("Edge brightness", 0., 50., 38.), gp("Smoothness", 1., 15., 5.)],
            AngledStrokes => vec![gp("Direction balance", 0., 100., 50.), gp("Stroke length", 3., 50., 15.), gp("Sharpness", 0., 10., 3.)],
            Crosshatch => vec![gp("Stroke length", 3., 50., 9.), gp("Sharpness", 0., 20., 6.), gp("Strength", 1., 3., 1.)],
            DarkStrokes => vec![gp("Balance", 0., 10., 5.), gp("Black intensity", 0., 10., 6.), gp("White intensity", 0., 10., 2.)],
            InkOutlines => vec![gp("Stroke length", 1., 50., 4.), gp("Dark intensity", 0., 50., 20.), gp("Light intensity", 0., 50., 10.)],
            Spatter => vec![gp("Spray radius", 0., 25., 10.), gp("Smoothness", 1., 15., 5.)],
            SprayedStrokes => vec![gp("Stroke length", 0., 20., 12.), gp("Spray radius", 0., 25., 7.), gc("Direction", &["Right diagonal", "Horizontal", "Left diagonal", "Vertical"], 0.)],
            SumiE => vec![gp("Stroke width", 3., 15., 10.), gp("Stroke pressure", 0., 15., 2.), gp("Contrast", 0., 40., 16.)],
            DiffuseGlow => vec![gp("Graininess", 0., 10., 6.), gp("Glow amount", 0., 20., 10.), gp("Clear amount", 0., 20., 15.)],
            Glass => vec![gp("Distortion", 0., 20., 5.), gp("Smoothness", 1., 15., 3.), gp("Scaling (%)", 50., 200., 100.)],
            OceanRipple => vec![gp("Ripple size", 1., 15., 9.), gp("Ripple magnitude", 0., 20., 9.)],
            BasRelief => vec![gp("Detail", 1., 15., 13.), gp("Smoothness", 1., 15., 3.), gp("Light angle (°)", 0., 360., 135.)],
            ChalkCharcoal => vec![gp("Charcoal area", 0., 20., 6.), gp("Chalk area", 0., 20., 6.), gp("Stroke pressure", 0., 5., 1.)],
            Charcoal => vec![gp("Charcoal thickness", 1., 7., 1.), gp("Detail", 0., 5., 5.), gp("Light/dark balance", 0., 100., 50.)],
            Chrome => vec![gp("Detail", 0., 10., 4.), gp("Smoothness", 0., 10., 7.)],
            ConteCrayon => vec![gp("Foreground level", 1., 15., 11.), gp("Background level", 1., 15., 7.), gc("Texture", TEXTURES, 0.)],
            GraphicPen => vec![gp("Stroke length", 1., 15., 15.), gp("Light/dark balance", 0., 100., 50.), gc("Direction", &["Right diagonal", "Horizontal", "Left diagonal", "Vertical"], 0.)],
            HalftonePattern => vec![gp("Size", 1., 12., 1.), gp("Contrast", 0., 50., 5.), gc("Pattern", &["Dot", "Circle", "Line"], 0.)],
            NotePaper => vec![gp("Image balance", 0., 50., 25.), gp("Graininess", 0., 20., 10.), gp("Relief", 0., 25., 11.)],
            Photocopy => vec![gp("Detail", 1., 24., 7.), gp("Darkness", 1., 50., 8.)],
            Plaster => vec![gp("Image balance", 0., 50., 20.), gp("Smoothness", 1., 15., 2.), gp("Light angle (°)", 0., 360., 90.)],
            Reticulation => vec![gp("Density", 0., 50., 12.), gp("Foreground level", 0., 50., 40.), gp("Background level", 0., 50., 5.)],
            Stamp => vec![gp("Light/dark balance", 0., 50., 25.), gp("Smoothness", 1., 50., 5.)],
            TornEdges => vec![gp("Image balance", 0., 50., 25.), gp("Smoothness", 1., 15., 11.), gp("Contrast", 1., 25., 17.)],
            WaterPaper => vec![gp("Fiber length", 3., 50., 15.), gp("Brightness", 0., 100., 60.), gp("Contrast", 0., 100., 80.)],
            GlowingEdges => vec![gp("Edge width", 1., 14., 2.), gp("Edge brightness", 0., 20., 6.), gp("Smoothness", 1., 15., 5.)],
            Craquelure => vec![gp("Crack spacing", 2., 100., 15.), gp("Crack depth", 0., 10., 6.), gp("Crack brightness", 0., 10., 9.)],
            Grain => vec![gp("Intensity", 0., 100., 40.), gp("Contrast", 0., 100., 50.), gc("Grain type", &["Regular", "Soft", "Sprinkles", "Clumped", "Contrasty", "Enlarged", "Stippled", "Horizontal", "Vertical", "Speckle"], 0.)],
            MosaicTiles => vec![gp("Tile size", 2., 100., 12.), gp("Grout width", 1., 15., 3.), gp("Lighten grout", 0., 10., 9.)],
            Patchwork => vec![gp("Square size", 0., 10., 4.), gp("Relief", 0., 25., 8.)],
            StainedGlass => vec![gp("Cell size", 2., 50., 10.), gp("Border thickness", 1., 20., 4.), gp("Light intensity", 0., 10., 3.)],
            Texturizer => vec![gc("Texture", TEXTURES, 0.), gp("Scaling (%)", 50., 200., 100.), gp("Relief", 0., 50., 4.)],
        }
    }

    /// Neighborhood padding in level pixels.
    pub fn pad(self, p: [f32; 3], level: u32) -> usize {
        use G::*;
        let s = (1u32 << level) as f32;
        let px = match self {
            Glass => p[0] * 6.0 + 6.0,
            OceanRipple => p[1] * 6.0 + 6.0,
            DiffuseGlow => 30.0,
            Spatter => p[0] * 1.5 + 12.0,
            SprayedStrokes => p[0] * 3.0 + p[1] * 1.5 + 12.0,
            AngledStrokes => p[1] * 0.5 + 14.0,
            Crosshatch | InkOutlines | RoughPastels | GraphicPen => p[0].max(10.0) * 1.5 + 12.0,
            PaintDaubs => p[0] * 0.5 + 12.0,
            PaletteKnife => p[0] * 0.7 + p[2] * 1.5 + 8.0,
            Cutout => 3.0 * (0.5 + 0.8 * p[1] / p[2].max(1.0)) + 56.0,
            NeonGlow => p[0].abs() * 1.8 + 6.0,
            StainedGlass => p[0] * 4.0 + 8.0,
            Patchwork => (p[0] * 2.0 + 1.0).max(3.0) * 2.0 + 2.0,
            MosaicTiles => p[0] * 2.0 + 6.0,
            Craquelure => p[0] * 2.5 + 6.0,
            WaterPaper => p[0] * 1.5 + 8.0,
            BasRelief => (p[1] * 2.5 + 4.0).max(24.0),
            Chrome => 40.0,
            PlasticWrap => p[2] * 1.3 + 18.0,
            Underpainting => (p[0] * 2.25 + p[1] * 0.16 + 6.0).max(24.0),
            Photocopy => p[0] * 2.8 + 6.0,
            Stamp => (p[1] * 1.5 + 6.0).max(24.0),
            Plaster => (p[1] * 0.9 + 20.0).max(24.0),
            _ => 24.0,
        };
        (px / s).ceil() as usize + 2
    }

    /// Renders outside any document (tests and the effect-only harnesses).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn apply(self, p: [f32; 3], fg: [u8; 3], bg: [u8; 3], buf: &mut [u32], w: usize, h: usize, level: u32, origin: (i32, i32)) {
        self.apply_in(p, fg, bg, buf, w, h, level, origin, None);
    }

    /// [`G::apply`] inside a document of `doc` level-0 px, for the effects
    /// that light or frame the whole document as Photoshop does (Plaster,
    /// Stained Glass). Without it those effects use document-independent
    /// fallbacks.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_in(self, p: [f32; 3], fg: [u8; 3], bg: [u8; 3], buf: &mut [u32], w: usize, h: usize, level: u32, origin: (i32, i32), doc: Option<(u32, u32)>) {
        let s = (1u32 << level) as f32;
        let mut img = Img::load(buf, w, h);
        let doc = doc.map(|(dw, dh)| (dw as f32, dh as f32));
        let ctx = Ctx { w, h, s, origin, fg: col(fg), bg: col(bg), doc };
        run(self, p, &mut img, &ctx);
        img.store(buf);
    }
}

// ----- building blocks ------------------------------------------------------------------

pub struct Img {
    pub w: usize,
    pub h: usize,
    pub c: [Vec<f32>; 3],
    pub a: Vec<f32>,
}

impl Img {
    fn load(buf: &[u32], w: usize, h: usize) -> Img {
        let n = w * h;
        let mut c = [vec![0f32; n], vec![0f32; n], vec![0f32; n]];
        let mut a = vec![0f32; n];
        for (i, &p) in buf.iter().enumerate() {
            let [r, g, b, al] = px::unpremultiply(p);
            c[0][i] = r as f32 / 255.0;
            c[1][i] = g as f32 / 255.0;
            c[2][i] = b as f32 / 255.0;
            a[i] = al as f32 / 255.0;
        }
        Img { w, h, c, a }
    }

    fn store(&self, buf: &mut [u32]) {
        for (i, o) in buf.iter_mut().enumerate() {
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            *o = px::premultiply(q(self.c[0][i]), q(self.c[1][i]), q(self.c[2][i]), q(self.a[i]));
        }
    }

    fn lum(&self) -> Vec<f32> {
        (0..self.w * self.h).map(|i| 0.299 * self.c[0][i] + 0.587 * self.c[1][i] + 0.114 * self.c[2][i]).collect()
    }

    fn map(&mut self, f: impl Fn(usize, [f32; 3]) -> [f32; 3]) {
        for i in 0..self.w * self.h {
            let o = f(i, [self.c[0][i], self.c[1][i], self.c[2][i]]);
            self.c[0][i] = o[0];
            self.c[1][i] = o[1];
            self.c[2][i] = o[2];
        }
    }

    fn each_channel(&mut self, f: impl Fn(&[f32]) -> Vec<f32>) {
        for k in 0..3 {
            self.c[k] = f(&self.c[k]);
        }
    }
}

struct Ctx {
    w: usize,
    h: usize,
    /// Level scale (level-0 pixels per buffer pixel).
    s: f32,
    origin: (i32, i32),
    fg: [f32; 3],
    bg: [f32; 3],
    /// Document size in level-0 px, when known.
    doc: Option<(f32, f32)>,
}

impl Ctx {
    /// Level-0 coordinate of buffer pixel `i`.
    #[inline]
    fn g(&self, i: usize) -> (f32, f32) {
        (((self.origin.0 + (i % self.w) as i32) as f32 + 0.5) * self.s, ((self.origin.1 + (i / self.w) as i32) as f32 + 0.5) * self.s)
    }

    /// Converts a level-0 length into buffer pixels.
    #[inline]
    fn px(&self, v: f32) -> f32 {
        v / self.s
    }

    fn noise(&self, i: usize, seed: u32) -> f32 {
        let (x, y) = self.g(i);
        hash01(x.floor() as i32, y.floor() as i32, seed)
    }
}

fn col(c: [u8; 3]) -> [f32; 3] {
    [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Three-box Gaussian on an f32 plane (edge clamped).
fn blur(v: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.4 {
        return v.to_vec();
    }
    let boxes = super::blur::boxes_for_gauss(sigma);
    let mut a = v.to_vec();
    let mut t = vec![0f32; w.max(h)];
    for &b in &boxes {
        let r = (b / 2) as isize;
        if r == 0 {
            continue;
        }
        let inv = 1.0 / (2 * r + 1) as f32;
        for y in 0..h {
            let row = &mut a[y * w..(y + 1) * w];
            let at = |i: isize| row[i.clamp(0, w as isize - 1) as usize];
            let mut s: f32 = (-r..=r).map(at).sum();
            for x in 0..w {
                t[x] = s * inv;
                s += at(x as isize + r + 1) - at(x as isize - r);
            }
            row.copy_from_slice(&t[..w]);
        }
        let mut col = vec![0f32; w];
        let rowi = |y: isize| y.clamp(0, h as isize - 1) as usize * w;
        for y in -r..=r {
            let o = rowi(y);
            for x in 0..w {
                col[x] += a[o + x];
            }
        }
        let src = a.clone();
        for y in 0..h {
            for x in 0..w {
                a[y * w + x] = col[x] * inv;
            }
            let (ad, sb) = (rowi(y as isize + r + 1), rowi(y as isize - r));
            for x in 0..w {
                col[x] += src[ad + x] - src[sb + x];
            }
        }
    }
    a
}

fn sobel(v: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let at = |x: isize, y: isize| v[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    let mut gx = vec![0f32; w * h];
    let mut gy = vec![0f32; w * h];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let i = y as usize * w + x as usize;
            gx[i] = (at(x + 1, y - 1) + 2.0 * at(x + 1, y) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x - 1, y) + at(x - 1, y + 1));
            gy[i] = (at(x - 1, y + 1) + 2.0 * at(x, y + 1) + at(x + 1, y + 1)) - (at(x - 1, y - 1) + 2.0 * at(x, y - 1) + at(x + 1, y - 1));
        }
    }
    (gx, gy)
}

/// Lambertian relief of a height field lit from `angle` degrees, 0.5 = flat.
fn relief(height: &[f32], w: usize, h: usize, strength: f32, angle: f32) -> Vec<f32> {
    let (gx, gy) = sobel(height, w, h);
    let a = angle.to_radians();
    let (lx, ly) = (a.cos(), -a.sin());
    gx.iter().zip(&gy).map(|(x, y)| (0.5 + (x * lx + y * ly) * strength).clamp(0.0, 1.0)).collect()
}

/// Procedural paper/canvas textures in level-0 space, roughly 0..1.
fn texture(ctx: &Ctx, kind: usize, scale: f32) -> Vec<f32> {
    let sc = scale.max(0.1);
    (0..ctx.w * ctx.h)
        .map(|i| {
            let (x, y) = ctx.g(i);
            let (x, y) = (x / sc, y / sc);
            match kind {
                // Canvas: woven grid.
                0 => 0.5 + 0.25 * ((x * 1.1).sin() * (y * 1.1).cos()) + 0.1 * (hash01(x as i32 / 3, y as i32 / 3, 5) - 0.5),
                // Burlap: coarse irregular weave.
                1 => {
                    let wx = ((x * 0.5).sin() * 0.5 + 0.5).powf(3.0);
                    let wy = ((y * 0.5).sin() * 0.5 + 0.5).powf(3.0);
                    0.35 + 0.4 * wx.max(wy) + 0.2 * super::util::fbm(x * 0.2, y * 0.2, 3, 3)
                }
                // Brick: offset rows with mortar.
                2 => {
                    let bh = 12.0;
                    let row = (y / bh).floor();
                    let bx = x + if row as i32 % 2 == 0 { 0.0 } else { 12.0 };
                    let mortar = (y % bh) < 1.5 || (bx % 24.0) < 1.5;
                    if mortar { 0.15 } else { 0.6 + 0.2 * (hash01(bx as i32 / 24, row as i32, 9) - 0.5) }
                }
                // Sandstone: fine fractal noise.
                _ => super::util::fbm(x * 0.6, y * 0.6, 4, 11),
            }
        })
        .collect()
}

fn kuwahara(img: &mut Img, r: usize) {
    if r == 0 {
        return;
    }
    let mut b = vec![0u32; img.w * img.h];
    img.store(&mut b);
    super::stylize::kuwahara(&mut b, img.w, img.h, r);
    *img = Img::load(&b, img.w, img.h);
}

fn median(img: &mut Img, r: usize) {
    if r == 0 {
        return;
    }
    let mut b = vec![0u32; img.w * img.h];
    img.store(&mut b);
    super::rank::median(&mut b, img.w, img.h, r);
    *img = Img::load(&b, img.w, img.h);
}

/// Bilinear self-warp inside the padded buffer.
fn warp(img: &mut Img, dx: &[f32], dy: &[f32]) {
    let (w, h) = (img.w, img.h);
    let src = [img.c[0].clone(), img.c[1].clone(), img.c[2].clone(), img.a.clone()];
    let at = |p: &[f32], x: isize, y: isize| p[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let sx = x as f32 + dx[i];
            let sy = y as f32 + dy[i];
            let (x0, y0) = (sx.floor(), sy.floor());
            let (fx, fy) = (sx - x0, sy - y0);
            let (xi, yi) = (x0 as isize, y0 as isize);
            for (k, p) in src.iter().enumerate() {
                let v = at(p, xi, yi) * (1.0 - fx) * (1.0 - fy)
                    + at(p, xi + 1, yi) * fx * (1.0 - fy)
                    + at(p, xi, yi + 1) * (1.0 - fx) * fy
                    + at(p, xi + 1, yi + 1) * fx * fy;
                if k < 3 {
                    img.c[k][i] = v;
                } else {
                    img.a[i] = v;
                }
            }
        }
    }
}

fn duotone(img: &mut Img, ctx: &Ctx, t: impl Fn(usize) -> f32) {
    // t = 0 → foreground (ink), 1 → background (paper)
    let (fg, bg) = (ctx.fg, ctx.bg);
    img.map(|i, _| mix(fg, bg, t(i).clamp(0.0, 1.0)));
}

// ----- the effects ----------------------------------------------------------------------

fn run(g: G, p: [f32; 3], img: &mut Img, ctx: &Ctx) {
    use G::*;
    match g {
        ColoredPencil | Cutout | DryBrush | FilmGrain | Fresco | NeonGlow | PaintDaubs | PaletteKnife => artistic1::run(g, p, img, ctx),
        PlasticWrap | PosterEdges | RoughPastels | SmudgeStick | Sponge | Underpainting | Watercolor => artistic2::run(g, p, img, ctx),
        AccentedEdges | AngledStrokes | Crosshatch | DarkStrokes | InkOutlines | Spatter | SprayedStrokes | SumiE | GlowingEdges => brush::run(g, p, img, ctx),
        BasRelief | ChalkCharcoal | Charcoal | Chrome | ConteCrayon | GraphicPen | HalftonePattern => sketch1::run(g, p, img, ctx),
        NotePaper | Photocopy | Plaster | Reticulation | Stamp | TornEdges | WaterPaper => sketch2::run(g, p, img, ctx),
        DiffuseGlow | Glass | OceanRipple | Craquelure | Grain | MosaicTiles | Patchwork | StainedGlass | Texturizer => texture::run(g, p, img, ctx),
    }
}

mod artistic1;
mod artistic2;
mod brush;
mod sketch1;
pub(crate) use sketch1::hatch_sd;
mod sketch2;
mod texture;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_gallery_effect_runs_and_keeps_alpha() {
        let src = crate::engine::ops::procedural(200, 160);
        for g in G::ALL {
            let defaults: Vec<f32> = g.params().iter().map(|p| p.default).collect();
            let mut p = [0f32; 3];
            for (i, v) in defaults.iter().enumerate() {
                p[i] = *v;
            }
            for level in [0u32, 2] {
                let pad = g.pad(p, level);
                let (w, h) = (48 + 2 * pad, 48 + 2 * pad);
                let mut buf = vec![0u32; w * h];
                src.read_region_clamped(10 - pad as i32, 10 - pad as i32, w, h, &mut buf);
                g.apply(p, [10, 10, 10], [250, 250, 240], &mut buf, w, h, level, (10 - pad as i32, 10 - pad as i32));
                assert!(buf.iter().all(|&q| px::alpha(q) >= 250), "{} at level {level}", g.title());
            }
        }
    }

    /// Effects that light or frame the whole document render the same in
    /// padded tiles as over the whole document, edges included.
    #[test]
    fn document_lit_effects_tile_like_the_whole_document() {
        let (dw, dh) = (300usize, 200usize);
        let src = crate::engine::ops::procedural(dw as u32, dh as u32);
        for g in [G::Plaster, G::StainedGlass] {
            let p: [f32; 3] = std::array::from_fn(|i| g.params().get(i).map_or(0.0, |d| d.default));
            let mut whole = vec![0u32; dw * dh];
            src.read_region_clamped(0, 0, dw, dh, &mut whole);
            g.apply_in(p, [0, 0, 0], [255, 255, 255], &mut whole, dw, dh, 0, (0, 0), Some((dw as u32, dh as u32)));
            let pad = g.pad(p, 0) as i32;
            for (tx, ty, tw, th) in [(0, 0, 80, 60), (150, 90, 70, 60), (230, 140, 70, 60)] {
                let (bw, bh) = (tw + 2 * pad as usize, th + 2 * pad as usize);
                let mut buf = vec![0u32; bw * bh];
                src.read_region_clamped(tx - pad, ty - pad, bw, bh, &mut buf);
                g.apply_in(p, [0, 0, 0], [255, 255, 255], &mut buf, bw, bh, 0, (tx - pad, ty - pad), Some((dw as u32, dh as u32)));
                for y in 0..th {
                    for x in 0..tw {
                        let (gx, gy) = (tx as usize + x, ty as usize + y);
                        // The image's own border is clamped differently by a
                        // whole render and a tile (as for every gallery
                        // effect); compare beyond the padding's reach.
                        let m = pad as usize;
                        if gx < m || gy < m || gx >= dw - m || gy >= dh - m {
                            continue;
                        }
                        let a = px::unpack(whole[gy * dw + gx]);
                        let b = px::unpack(buf[(y + pad as usize) * bw + x + pad as usize]);
                        for k in 0..3 {
                            assert!((a[k] as i32 - b[k] as i32).abs() <= 2, "{} at ({gx}, {gy}): {a:?} vs {b:?}", g.title());
                        }
                    }
                }
            }
            // Stained Glass leads the document's edge, in tiles too.
            if g == G::StainedGlass {
                let mut buf = vec![0u32; (40 + 2 * pad as usize).pow(2)];
                let bw = 40 + 2 * pad as usize;
                src.read_region_clamped(-pad, -pad, bw, bw, &mut buf);
                g.apply_in(p, [0, 0, 0], [255, 255, 255], &mut buf, bw, bw, 0, (-pad, -pad), Some((dw as u32, dh as u32)));
                assert_eq!(px::unpack(whole[0])[..3], [0, 0, 0]);
                assert_eq!(px::unpack(buf[pad as usize * bw + pad as usize])[..3], [0, 0, 0]);
            }
        }
    }
}
