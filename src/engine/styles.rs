//! Layer Styles (Photoshop's layer effects) and Fill opacity.
//!
//! The model is Photoshop's. Every effect is computed from the layer's
//! *shape* `A` (its alpha after the layer mask) inside a padded region
//! (pad = the largest reach of any effect), with fast integer primitives:
//! - spread / choke / stroke width: a disc dilation (`Dil`), the exact
//!   Euclidean distance from every pixel to the nearest covered pixel
//!   (separable min-plus transform, alpha of the nearest pixel as sub-pixel
//!   offset), turned into anti-aliased coverage through a table;
//! - size: three running-sum box passes (`fast::box_plane`, the blur code
//!   the filters use) whose radii add up to the size;
//! - bevel: a relief of the blurred shape, shaded through a CPU-built table
//!   indexed by the height gradient.
//!
//! Effects stack in Photoshop's order, bottom to top: Drop Shadow, Outer
//! Glow, the layer content (scaled by Fill), Gradient Overlay, Color
//! Overlay, Satin, Inner Glow, Inner Shadow, Stroke, Bevel & Emboss. The
//! interior effects stack on the content in "shape space" (as if the shape
//! were opaque, so Fill 0 keeps them) and are then multiplied by `A`; the
//! outer stroke ring and outer bevel live in `1 - A`; shadow and glow sit
//! beneath (knocked out by the shape). The styled result then blends into
//! the backdrop with the layer's blend mode and opacity, like any layer.
//!
//! Every plane is integer-defined, so the GPU twin (`gpu::styles`) computes
//! the same planes bit for bit; only the per-pixel blend modes are float.
//! A [`Plan`] holds every derived parameter and table, shared by both.

use std::sync::{Arc, Mutex};

use super::blend::{BlendMode, blend_px};
use super::document::Layer;
use super::grid::RectI;
use super::layers::LayerKind;
use super::px;

// ----- the style ------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BevelStyle {
    Outer,
    Inner,
    Emboss,
    Pillow,
}

impl BevelStyle {
    pub const ALL: [BevelStyle; 4] = [BevelStyle::Outer, BevelStyle::Inner, BevelStyle::Emboss, BevelStyle::Pillow];
    pub const fn name(self) -> &'static str {
        match self {
            BevelStyle::Outer => "Outer Bevel",
            BevelStyle::Inner => "Inner Bevel",
            BevelStyle::Emboss => "Emboss",
            BevelStyle::Pillow => "Pillow Emboss",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StrokePos {
    Outside,
    Inside,
    Center,
}

impl StrokePos {
    pub const ALL: [StrokePos; 3] = [StrokePos::Outside, StrokePos::Inside, StrokePos::Center];
    pub const fn name(self) -> &'static str {
        match self {
            StrokePos::Outside => "Outside",
            StrokePos::Inside => "Inside",
            StrokePos::Center => "Center",
        }
    }
}

/// Drop Shadow and Inner Shadow (`spread` is Choke for the inner one;
/// `noise` and `knockout` only apply to the drop shadow).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Shadow {
    pub enabled: bool,
    pub blend: BlendMode,
    pub color: [u8; 3],
    /// Percent.
    pub opacity: f32,
    /// Degrees; the light comes from this direction.
    pub angle: f32,
    pub distance: f32,
    /// Percent of `size` turned into a hard dilation (spread / choke).
    pub spread: f32,
    pub size: f32,
    /// Percent.
    pub noise: f32,
    /// Layer Knocks Out Drop Shadow.
    pub knockout: bool,
}

impl Shadow {
    pub fn drop() -> Shadow {
        Shadow { enabled: false, blend: BlendMode::Multiply, color: [0, 0, 0], opacity: 75.0, angle: 120.0, distance: 5.0, spread: 0.0, size: 5.0, noise: 0.0, knockout: true }
    }
    pub fn inner() -> Shadow {
        Shadow { knockout: false, ..Shadow::drop() }
    }
}

/// Outer Glow and Inner Glow (technique Softer; `spread` is Choke for the
/// inner one; `center` is the inner glow's Source: Center vs Edge).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Glow {
    pub enabled: bool,
    pub blend: BlendMode,
    pub color: [u8; 3],
    pub opacity: f32,
    pub spread: f32,
    pub size: f32,
    /// Percent (50 = linear contour over the whole glow).
    pub range: f32,
    pub center: bool,
}

impl Glow {
    pub fn outer() -> Glow {
        Glow { enabled: false, blend: BlendMode::Screen, color: [255, 255, 190], opacity: 75.0, spread: 0.0, size: 5.0, range: 50.0, center: false }
    }
    pub fn inner() -> Glow {
        Glow::outer()
    }
}

/// Bevel & Emboss (technique Smooth).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Bevel {
    pub enabled: bool,
    pub style: BevelStyle,
    /// Percent.
    pub depth: f32,
    pub up: bool,
    pub size: f32,
    pub soften: f32,
    pub angle: f32,
    pub altitude: f32,
    pub hi_blend: BlendMode,
    pub hi_color: [u8; 3],
    pub hi_opacity: f32,
    pub sh_blend: BlendMode,
    pub sh_color: [u8; 3],
    pub sh_opacity: f32,
}

impl Default for Bevel {
    fn default() -> Bevel {
        Bevel {
            enabled: false,
            style: BevelStyle::Inner,
            depth: 100.0,
            up: true,
            size: 5.0,
            soften: 0.0,
            angle: 120.0,
            altitude: 30.0,
            hi_blend: BlendMode::Screen,
            hi_color: [255, 255, 255],
            hi_opacity: 75.0,
            sh_blend: BlendMode::Multiply,
            sh_color: [0, 0, 0],
            sh_opacity: 75.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Satin {
    pub enabled: bool,
    pub blend: BlendMode,
    pub color: [u8; 3],
    pub opacity: f32,
    pub angle: f32,
    pub distance: f32,
    pub size: f32,
    pub invert: bool,
}

impl Default for Satin {
    fn default() -> Satin {
        Satin { enabled: false, blend: BlendMode::Multiply, color: [0, 0, 0], opacity: 50.0, angle: 19.0, distance: 11.0, size: 14.0, invert: true }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ColorOverlay {
    pub enabled: bool,
    pub blend: BlendMode,
    pub color: [u8; 3],
    pub opacity: f32,
}

impl Default for ColorOverlay {
    fn default() -> ColorOverlay {
        ColorOverlay { enabled: false, blend: BlendMode::Normal, color: [255, 0, 0], opacity: 100.0 }
    }
}

/// Two-colour gradient, aligned with the layer's content bounds.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GradientOverlay {
    pub enabled: bool,
    pub blend: BlendMode,
    pub opacity: f32,
    pub from: [u8; 3],
    pub to: [u8; 3],
    pub radial: bool,
    pub angle: f32,
    /// Percent.
    pub scale: f32,
    pub reverse: bool,
}

impl Default for GradientOverlay {
    fn default() -> GradientOverlay {
        GradientOverlay { enabled: false, blend: BlendMode::Normal, opacity: 100.0, from: [0, 0, 0], to: [255, 255, 255], radial: false, angle: 90.0, scale: 100.0, reverse: false }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct StrokeFx {
    pub enabled: bool,
    pub size: f32,
    pub position: StrokePos,
    pub blend: BlendMode,
    pub opacity: f32,
    pub color: [u8; 3],
}

impl Default for StrokeFx {
    fn default() -> StrokeFx {
        StrokeFx { enabled: false, size: 3.0, position: StrokePos::Outside, blend: BlendMode::Normal, opacity: 100.0, color: [255, 0, 0] }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct LayerStyle {
    pub drop_shadow: Shadow,
    pub inner_shadow: Shadow,
    pub outer_glow: Glow,
    pub inner_glow: Glow,
    pub bevel: Bevel,
    pub satin: Satin,
    pub color_overlay: ColorOverlay,
    pub gradient_overlay: GradientOverlay,
    pub stroke: StrokeFx,
}

impl Default for LayerStyle {
    fn default() -> LayerStyle {
        LayerStyle {
            drop_shadow: Shadow::drop(),
            inner_shadow: Shadow::inner(),
            outer_glow: Glow::outer(),
            inner_glow: Glow::inner(),
            bevel: Bevel::default(),
            satin: Satin::default(),
            color_overlay: ColorOverlay::default(),
            gradient_overlay: GradientOverlay::default(),
            stroke: StrokeFx::default(),
        }
    }
}

/// The effects, in the Layer Style dialog's order (top of the stack first).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Effect {
    Bevel,
    Stroke,
    InnerShadow,
    InnerGlow,
    Satin,
    ColorOverlay,
    GradientOverlay,
    OuterGlow,
    DropShadow,
}

impl Effect {
    pub const ALL: [Effect; 9] = [
        Effect::Bevel,
        Effect::Stroke,
        Effect::InnerShadow,
        Effect::InnerGlow,
        Effect::Satin,
        Effect::ColorOverlay,
        Effect::GradientOverlay,
        Effect::OuterGlow,
        Effect::DropShadow,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Effect::Bevel => "Bevel & Emboss",
            Effect::Stroke => "Stroke",
            Effect::InnerShadow => "Inner Shadow",
            Effect::InnerGlow => "Inner Glow",
            Effect::Satin => "Satin",
            Effect::ColorOverlay => "Color Overlay",
            Effect::GradientOverlay => "Gradient Overlay",
            Effect::OuterGlow => "Outer Glow",
            Effect::DropShadow => "Drop Shadow",
        }
    }
}

impl LayerStyle {
    pub fn enabled(&self, e: Effect) -> bool {
        match e {
            Effect::Bevel => self.bevel.enabled,
            Effect::Stroke => self.stroke.enabled,
            Effect::InnerShadow => self.inner_shadow.enabled,
            Effect::InnerGlow => self.inner_glow.enabled,
            Effect::Satin => self.satin.enabled,
            Effect::ColorOverlay => self.color_overlay.enabled,
            Effect::GradientOverlay => self.gradient_overlay.enabled,
            Effect::OuterGlow => self.outer_glow.enabled,
            Effect::DropShadow => self.drop_shadow.enabled,
        }
    }

    pub fn set_enabled(&mut self, e: Effect, on: bool) {
        match e {
            Effect::Bevel => self.bevel.enabled = on,
            Effect::Stroke => self.stroke.enabled = on,
            Effect::InnerShadow => self.inner_shadow.enabled = on,
            Effect::InnerGlow => self.inner_glow.enabled = on,
            Effect::Satin => self.satin.enabled = on,
            Effect::ColorOverlay => self.color_overlay.enabled = on,
            Effect::GradientOverlay => self.gradient_overlay.enabled = on,
            Effect::OuterGlow => self.outer_glow.enabled = on,
            Effect::DropShadow => self.drop_shadow.enabled = on,
        }
    }

    /// Whether any effect is on.
    pub fn any(&self) -> bool {
        Effect::ALL.iter().any(|&e| self.enabled(e))
    }
}

/// Whether the compositor must run the style path for this layer.
pub fn is_styled(layer: &Layer) -> bool {
    (layer.fill < 1.0 || layer.style.as_ref().is_some_and(|s| s.any())) && !matches!(layer.kind, LayerKind::Adjustment(_) | LayerKind::Artboard(_))
}

// ----- persistence ----------------------------------------------------------------------

struct Wr(Vec<u8>);
impl Wr {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn f(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn b(&mut self, v: bool) {
        self.u8(v as u8);
    }
    fn c(&mut self, c: [u8; 3]) {
        self.0.extend_from_slice(&c);
    }
    fn m(&mut self, m: BlendMode) {
        self.u8(m as u8);
    }
}

struct Rd<'a>(&'a [u8]);
impl Rd<'_> {
    fn u8(&mut self) -> Option<u8> {
        let (&v, rest) = self.0.split_first()?;
        self.0 = rest;
        Some(v)
    }
    fn f(&mut self) -> Option<f32> {
        if self.0.len() < 4 {
            return None;
        }
        let v = f32::from_le_bytes(self.0[..4].try_into().ok()?);
        self.0 = &self.0[4..];
        Some(if v.is_finite() { v } else { 0.0 })
    }
    fn b(&mut self) -> Option<bool> {
        Some(self.u8()? != 0)
    }
    fn c(&mut self) -> Option<[u8; 3]> {
        Some([self.u8()?, self.u8()?, self.u8()?])
    }
    fn m(&mut self) -> Option<BlendMode> {
        Some(BlendMode::from_u8(self.u8()?))
    }
}

const FMT: u8 = 1;

impl LayerStyle {
    /// Versioned binary form (`.pforge` "PFLX" tag 6): a format byte, then
    /// records `effect id u8 | len u32 | payload`; unknown ids are skipped.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = vec![FMT];
        let mut rec = |id: u8, f: &dyn Fn(&mut Wr)| {
            let mut w = Wr(Vec::new());
            f(&mut w);
            out.push(id);
            out.extend_from_slice(&(w.0.len() as u32).to_le_bytes());
            out.extend_from_slice(&w.0);
        };
        let shadow = |s: &Shadow| {
            let s = *s;
            move |w: &mut Wr| {
                w.b(s.enabled);
                w.m(s.blend);
                w.c(s.color);
                for v in [s.opacity, s.angle, s.distance, s.spread, s.size, s.noise] {
                    w.f(v);
                }
                w.b(s.knockout);
            }
        };
        let glow = |g: &Glow| {
            let g = *g;
            move |w: &mut Wr| {
                w.b(g.enabled);
                w.m(g.blend);
                w.c(g.color);
                for v in [g.opacity, g.spread, g.size, g.range] {
                    w.f(v);
                }
                w.b(g.center);
            }
        };
        rec(1, &shadow(&self.drop_shadow));
        rec(2, &shadow(&self.inner_shadow));
        rec(3, &glow(&self.outer_glow));
        rec(4, &glow(&self.inner_glow));
        let b = self.bevel;
        rec(5, &move |w: &mut Wr| {
            w.b(b.enabled);
            w.u8(b.style as u8);
            w.b(b.up);
            for v in [b.depth, b.size, b.soften, b.angle, b.altitude, b.hi_opacity, b.sh_opacity] {
                w.f(v);
            }
            w.m(b.hi_blend);
            w.c(b.hi_color);
            w.m(b.sh_blend);
            w.c(b.sh_color);
        });
        let s = self.satin;
        rec(6, &move |w: &mut Wr| {
            w.b(s.enabled);
            w.m(s.blend);
            w.c(s.color);
            for v in [s.opacity, s.angle, s.distance, s.size] {
                w.f(v);
            }
            w.b(s.invert);
        });
        let c = self.color_overlay;
        rec(7, &move |w: &mut Wr| {
            w.b(c.enabled);
            w.m(c.blend);
            w.c(c.color);
            w.f(c.opacity);
        });
        let g = self.gradient_overlay;
        rec(8, &move |w: &mut Wr| {
            w.b(g.enabled);
            w.m(g.blend);
            w.f(g.opacity);
            w.c(g.from);
            w.c(g.to);
            w.b(g.radial);
            w.f(g.angle);
            w.f(g.scale);
            w.b(g.reverse);
        });
        let k = self.stroke;
        rec(9, &move |w: &mut Wr| {
            w.b(k.enabled);
            w.f(k.size);
            w.u8(k.position as u8);
            w.m(k.blend);
            w.f(k.opacity);
            w.c(k.color);
        });
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<LayerStyle> {
        let mut r = Rd(b);
        if r.u8()? != FMT {
            return None;
        }
        let mut st = LayerStyle::default();
        while !r.0.is_empty() {
            let id = r.u8()?;
            let len = u32::from_le_bytes(r.0.get(..4)?.try_into().ok()?) as usize;
            r.0 = &r.0[4..];
            let body = r.0.get(..len)?;
            r.0 = &r.0[len..];
            let mut p = Rd(body);
            let shadow = |p: &mut Rd, mut s: Shadow| -> Option<Shadow> {
                s.enabled = p.b()?;
                s.blend = p.m()?;
                s.color = p.c()?;
                s.opacity = p.f()?;
                s.angle = p.f()?;
                s.distance = p.f()?;
                s.spread = p.f()?;
                s.size = p.f()?;
                s.noise = p.f()?;
                s.knockout = p.b()?;
                Some(s)
            };
            let glow = |p: &mut Rd, mut g: Glow| -> Option<Glow> {
                g.enabled = p.b()?;
                g.blend = p.m()?;
                g.color = p.c()?;
                g.opacity = p.f()?;
                g.spread = p.f()?;
                g.size = p.f()?;
                g.range = p.f()?;
                g.center = p.b()?;
                Some(g)
            };
            match id {
                1 => st.drop_shadow = shadow(&mut p, st.drop_shadow)?,
                2 => st.inner_shadow = shadow(&mut p, st.inner_shadow)?,
                3 => st.outer_glow = glow(&mut p, st.outer_glow)?,
                4 => st.inner_glow = glow(&mut p, st.inner_glow)?,
                5 => {
                    let b = &mut st.bevel;
                    b.enabled = p.b()?;
                    b.style = *BevelStyle::ALL.get(p.u8()? as usize)?;
                    b.up = p.b()?;
                    b.depth = p.f()?;
                    b.size = p.f()?;
                    b.soften = p.f()?;
                    b.angle = p.f()?;
                    b.altitude = p.f()?;
                    b.hi_opacity = p.f()?;
                    b.sh_opacity = p.f()?;
                    b.hi_blend = p.m()?;
                    b.hi_color = p.c()?;
                    b.sh_blend = p.m()?;
                    b.sh_color = p.c()?;
                }
                6 => {
                    let s = &mut st.satin;
                    s.enabled = p.b()?;
                    s.blend = p.m()?;
                    s.color = p.c()?;
                    s.opacity = p.f()?;
                    s.angle = p.f()?;
                    s.distance = p.f()?;
                    s.size = p.f()?;
                    s.invert = p.b()?;
                }
                7 => {
                    let c = &mut st.color_overlay;
                    c.enabled = p.b()?;
                    c.blend = p.m()?;
                    c.color = p.c()?;
                    c.opacity = p.f()?;
                }
                8 => {
                    let g = &mut st.gradient_overlay;
                    g.enabled = p.b()?;
                    g.blend = p.m()?;
                    g.opacity = p.f()?;
                    g.from = p.c()?;
                    g.to = p.c()?;
                    g.radial = p.b()?;
                    g.angle = p.f()?;
                    g.scale = p.f()?;
                    g.reverse = p.b()?;
                }
                9 => {
                    let k = &mut st.stroke;
                    k.enabled = p.b()?;
                    k.size = p.f()?;
                    k.position = *StrokePos::ALL.get(p.u8()? as usize)?;
                    k.blend = p.m()?;
                    k.opacity = p.f()?;
                    k.color = p.c()?;
                }
                _ => {}
            }
        }
        Some(st)
    }
}

// ----- the plan: every derived parameter and table ------------------------------------------

/// Largest size / distance the renderer accepts (px at level 0).
pub const MAX_SIZE: f32 = 250.0;
pub const MAX_DISTANCE: f32 = 500.0;

pub const INF: u32 = u32::MAX;
/// Fixed-point scale of coverage sums (255 * 16).
const ONE16: f64 = 4080.0;

/// Disc dilation of a coverage plane by radius `r` (see the module doc).
///
/// `D(p) = min over covered q of |p - q|^2 * 256 + (255 - a_q)` (the low
/// byte picks the most covered of the nearest pixels), then coverage
/// `clamp(255 r + a - 255 sqrt(d2))`, in 1/16 steps through `sqrt`. Only
/// pixels within `w` (> r + 1) can contribute, so any window of at least
/// `w` gives the same planes.
#[derive(Clone, Debug)]
pub struct Dil {
    pub r16: u32,
    /// Window (`floor(r + 1) + 1`); `d2 >= w * w` means no coverage.
    pub w: u32,
    /// `round(sqrt(d2) * 4080)` for `d2 < w * w`.
    pub sqrt: Arc<Vec<u32>>,
}

fn sqrt_table(w: u32) -> Arc<Vec<u32>> {
    static CACHE: Mutex<Vec<(u32, Arc<Vec<u32>>)>> = Mutex::new(Vec::new());
    let mut c = CACHE.lock().unwrap();
    if let Some((_, t)) = c.iter().find(|(k, _)| *k == w) {
        return t.clone();
    }
    let t: Arc<Vec<u32>> = Arc::new((0..w * w).map(|d2| ((d2 as f64).sqrt() * ONE16).round() as u32).collect());
    if c.len() >= 16 {
        c.remove(0);
    }
    c.push((w, t.clone()));
    t
}

impl Dil {
    pub fn new(r: f32) -> Option<Dil> {
        if r.is_nan() || r < 0.02 {
            return None;
        }
        let r = r.min(MAX_SIZE + 1.0);
        let w = (r + 1.0).floor() as u32 + 1;
        Some(Dil { r16: (r as f64 * ONE16).round() as u32, w, sqrt: sqrt_table(w) })
    }

    /// Coverage (0..255) for a transform value `d`.
    #[inline]
    pub fn coverage(&self, d: u32) -> u8 {
        if d == INF {
            return 0;
        }
        let d2 = d >> 8;
        if d2 >= self.w * self.w {
            return 0;
        }
        let a = 255 - (d & 255);
        let c = self.r16 as i64 + a as i64 * 16 - self.sqrt[d2 as usize] as i64;
        if c <= 0 { 0 } else { ((c + 8) >> 4).min(255) as u8 }
    }
}

/// Where a shape operation reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// The layer's alpha, or `255 - alpha`.
    Alpha { invert: bool },
    /// A finished lane (soften).
    Lane(u8),
}

/// Source plane, optional dilation (then optional inversion: an erosion),
/// then box blur passes (vertical passes, then horizontal, like
/// `fast::box_plane`).
#[derive(Clone, Debug)]
pub struct ShapeOp {
    pub src: Src,
    pub dil: Option<Dil>,
    pub invert_after: bool,
    /// Active box radii, in pass order.
    pub boxes: Vec<u32>,
}

impl ShapeOp {
    fn reach(&self) -> u32 {
        self.dil.as_ref().map_or(0, |d| d.w) + self.boxes.iter().sum::<u32>()
    }
}

/// How a shape plane becomes an effect lane.
#[derive(Clone, Debug)]
pub enum Fin {
    /// `v = plane(x - dx, y - dy)` (`fill` outside the buffer), then the
    /// contour `lut`, then `invert`, then noise (`noise8`, the CPU hash at
    /// layer-local level coordinates).
    Map { dx: i32, dy: i32, fill: u8, lut: Option<Arc<Vec<u8>>>, invert: bool, noise8: u32 },
    /// `|plane(x - o) - plane(x + o)|`, optionally inverted.
    Satin { dx: i32, dy: i32, invert: bool },
}

#[derive(Clone, Debug)]
pub struct Job {
    pub shape: ShapeOp,
    pub fin: Fin,
    /// Destination lane: plane `lane / 4`, byte `lane % 4`.
    pub lane: u8,
}

impl Job {
    fn reach(&self) -> u32 {
        let shift = match &self.fin {
            Fin::Map { dx, dy, .. } | Fin::Satin { dx, dy, .. } => dx.unsigned_abs().max(dy.unsigned_abs()),
        };
        self.shape.reach() + shift
    }
}

pub const L_DS: u8 = 0;
pub const L_OG: u8 = 1;
pub const L_IS: u8 = 2;
pub const L_IG: u8 = 3;
pub const L_SAT: u8 = 4;
pub const L_RING: u8 = 5;
/// The dilated inverse of the shape for the stroke's inside part.
pub const L_EIN: u8 = 6;
pub const L_HI_IN: u8 = 8;
pub const L_SH_IN: u8 = 9;
pub const L_HI_OUT: u8 = 10;
pub const L_SH_OUT: u8 = 11;
pub const LANES: usize = 12;

/// The bevel: relief of the shaped height map through `lut`
/// (`(gx + 255) * 511 + gy + 255` -> highlight | shadow << 8, gradients by
/// central differences, clamped at the buffer edge).
#[derive(Clone, Debug)]
pub struct BevelJob {
    pub shape: ShapeOp,
    pub lut: Arc<Vec<u16>>,
    pub interior: bool,
    pub exterior: bool,
    /// Pillow Emboss: the exterior reads the flipped gradient.
    pub pillow: bool,
}

/// Blend parameters of one colour layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fx {
    pub mode: BlendMode,
    pub op8: u32,
    pub color: [u8; 3],
}

#[derive(Clone, Debug)]
pub struct GradFx {
    pub mode: BlendMode,
    pub op8: u32,
    /// 1024 opaque colours (`px::pack(r, g, b, 255)`).
    pub table: Arc<Vec<u32>>,
    pub radial: bool,
    pub reverse: bool,
    /// Level-space, layer-local centre, direction and 1 / length.
    pub cx: f32,
    pub cy: f32,
    pub ux: f32,
    pub uy: f32,
    pub inv_len: f32,
}

impl GradFx {
    /// Colour at layer-local level pixel `(x, y)`.
    #[inline]
    pub fn at(&self, x: i32, y: i32) -> u32 {
        let px = x as f32 + 0.5 - self.cx;
        let py = y as f32 + 0.5 - self.cy;
        let mut t = if self.radial { (px * px + py * py).sqrt() * self.inv_len } else { (px * self.ux + py * self.uy) * self.inv_len + 0.5 };
        if self.reverse {
            t = 1.0 - t;
        }
        let i = (t.clamp(0.0, 1.0) * 1023.0 + 0.5) as usize;
        self.table[i.min(1023)]
    }
}

/// Everything needed to render a style at one level, CPU or GPU.
#[derive(Clone, Debug)]
pub struct Plan {
    pub level: u32,
    pub pad: usize,
    pub fill8: u32,
    /// Lane jobs in order (shadows, glows, satin, stroke), then the bevel,
    /// then `soften` jobs on the bevel lanes.
    pub jobs: Vec<Job>,
    pub bevel: Option<BevelJob>,
    pub soften: Vec<Job>,
    pub ds: Option<(Fx, bool)>,
    pub og: Option<Fx>,
    pub is: Option<Fx>,
    pub ig: Option<Fx>,
    pub satin: Option<Fx>,
    pub co: Option<Fx>,
    pub go: Option<GradFx>,
    /// Stroke: colour, then whether it has an outside ring / an inside part.
    pub stroke: Option<(Fx, bool, bool)>,
    /// Highlight, shadow.
    pub bevel_fx: Option<(Fx, Fx)>,
}

impl Plan {
    /// Whether anything is stacked on the content inside the shape.
    pub fn interior(&self) -> bool {
        self.go.is_some()
            || self.co.is_some()
            || self.satin.is_some()
            || self.ig.is_some()
            || self.is.is_some()
            || self.stroke.is_some_and(|s| s.2)
            || self.bevel.as_ref().is_some_and(|b| b.interior)
    }

    pub fn exterior(&self) -> bool {
        self.stroke.is_some_and(|s| s.1) || self.bevel.as_ref().is_some_and(|b| b.exterior)
    }

    /// Number of packed lane planes (4 lanes each) the jobs write.
    pub fn planes(&self) -> usize {
        let mut m = 0;
        for j in self.jobs.iter().chain(&self.soften) {
            m = m.max(j.lane as usize / 4 + 1);
        }
        if self.bevel.is_some() {
            m = m.max(3);
        }
        m
    }
}

fn op8(pct: f32) -> u32 {
    ((pct / 100.0).clamp(0.0, 1.0) * 255.0).round() as u32
}

/// Box radii whose sum is `b` (three passes approximate a Gaussian).
fn boxes(b: u32) -> Vec<u32> {
    [b / 3, (b + 1) / 3, (b + 2) / 3].into_iter().filter(|&r| r > 0).collect()
}

fn offset(angle: f32, dist: f32) -> (i32, i32) {
    let t = (angle as f64).to_radians();
    ((-t.cos() * dist as f64).round() as i32, (t.sin() * dist as f64).round() as i32)
}

/// Glow contour for `range` (percent): 50 is linear; lower ranges move the
/// glow towards the edge, higher ones spread it out.
fn range_lut(range: f32) -> Option<Arc<Vec<u8>>> {
    let r = (range / 100.0).clamp(0.01, 1.0) as f64;
    if (r - 0.5).abs() < 1e-6 {
        return None;
    }
    Some(Arc::new(
        (0..256)
            .map(|v| {
                let t = v as f64 / 255.0;
                (((t - (1.0 - 2.0 * r)) / (2.0 * r)).clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect(),
    ))
}

fn bevel_lut(depth: f32, size: f32, angle: f32, altitude: f32, up: bool) -> Arc<Vec<u16>> {
    type Key = (u32, u32, u32, u32, bool);
    static CACHE: Mutex<Vec<(Key, Arc<Vec<u16>>)>> = Mutex::new(Vec::new());
    let key = (depth.to_bits(), size.to_bits(), angle.to_bits(), altitude.to_bits(), up);
    if let Some((_, t)) = CACHE.lock().unwrap().iter().find(|(k, _)| *k == key) {
        return t.clone();
    }
    let k = 0.5 / 255.0 * size as f64 * depth as f64 / 100.0 * if up { 1.0 } else { -1.0 };
    let (alt, th) = ((altitude as f64).to_radians(), (angle as f64).to_radians());
    let l = [alt.cos() * th.cos(), -alt.cos() * th.sin(), alt.sin()];
    let flat = l[2];
    let mut t = vec![0u16; 511 * 511];
    for gx in -255i32..=255 {
        for gy in -255i32..=255 {
            let (sx, sy) = (gx as f64 * k, gy as f64 * k);
            let n = (sx * sx + sy * sy + 1.0).sqrt();
            let s = (-sx * l[0] - sy * l[1] + l[2]) / n;
            let hi = if flat < 0.9999 { ((s - flat) / (1.0 - flat)).clamp(0.0, 1.0) } else { 0.0 };
            let sh = if flat > 1e-4 { ((flat - s) / flat).clamp(0.0, 1.0) } else { 0.0 };
            t[((gx + 255) * 511 + gy + 255) as usize] = ((hi * 255.0).round() as u16) | ((sh * 255.0).round() as u16) << 8;
        }
    }
    let t = Arc::new(t);
    let mut c = CACHE.lock().unwrap();
    if c.len() >= 8 {
        c.remove(0);
    }
    c.push((key, t.clone()));
    t
}

fn grad_table(from: [u8; 3], to: [u8; 3]) -> Arc<Vec<u32>> {
    Arc::new(
        (0..1024)
            .map(|i| {
                let t = i as f32 / 1023.0;
                let ch = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
                px::pack(ch(from[0], to[0]), ch(from[1], to[1]), ch(from[2], to[2]), 255)
            })
            .collect(),
    )
}

impl Plan {
    /// The plan for `style` (may be None: Fill only) at `level`; `bounds`
    /// are the layer's level-0, layer-local content bounds (Gradient
    /// Overlay alignment).
    pub fn new(style: Option<&LayerStyle>, fill: f32, level: u32, bounds: RectI) -> Plan {
        let k = (1u32 << level) as f32;
        let sz = |v: f32| v.clamp(0.0, MAX_SIZE) / k;
        let dist = |v: f32| v.clamp(0.0, MAX_DISTANCE) / k;
        let reach = |v: f32| v.round().max(0.0) as u32;
        let mut p = Plan {
            level,
            pad: 0,
            fill8: op8(fill * 100.0),
            jobs: vec![],
            bevel: None,
            soften: vec![],
            ds: None,
            og: None,
            is: None,
            ig: None,
            satin: None,
            co: None,
            go: None,
            stroke: None,
            bevel_fx: None,
        };
        let Some(st) = style else { return p };
        let fx = |mode, op: f32, color| Fx { mode, op8: op8(op), color };
        // Spread / choke: a hard dilation of size * spread, then the rest blurred.
        let spread_op = |src: Src, size: f32, spread: f32| {
            let s = sz(size);
            let r = s * (spread / 100.0).clamp(0.0, 1.0);
            ShapeOp { src, dil: Dil::new(r), invert_after: false, boxes: boxes(reach(s - r)) }
        };
        let ds = &st.drop_shadow;
        if ds.enabled {
            let (dx, dy) = offset(ds.angle, dist(ds.distance));
            let noise8 = op8(ds.noise);
            p.jobs.push(Job { shape: spread_op(Src::Alpha { invert: false }, ds.size, ds.spread), fin: Fin::Map { dx, dy, fill: 0, lut: None, invert: false, noise8 }, lane: L_DS });
            p.ds = Some((fx(ds.blend, ds.opacity, ds.color), ds.knockout));
        }
        let og = &st.outer_glow;
        if og.enabled {
            p.jobs.push(Job { shape: spread_op(Src::Alpha { invert: false }, og.size, og.spread), fin: Fin::Map { dx: 0, dy: 0, fill: 0, lut: range_lut(og.range), invert: false, noise8: 0 }, lane: L_OG });
            p.og = Some(fx(og.blend, og.opacity, og.color));
        }
        let is = &st.inner_shadow;
        if is.enabled {
            let (dx, dy) = offset(is.angle, dist(is.distance));
            p.jobs.push(Job { shape: spread_op(Src::Alpha { invert: true }, is.size, is.spread), fin: Fin::Map { dx, dy, fill: 255, lut: None, invert: false, noise8: 0 }, lane: L_IS });
            p.is = Some(fx(is.blend, is.opacity, is.color));
        }
        let ig = &st.inner_glow;
        if ig.enabled {
            p.jobs.push(Job {
                shape: spread_op(Src::Alpha { invert: true }, ig.size, ig.spread),
                fin: Fin::Map { dx: 0, dy: 0, fill: 255, lut: range_lut(ig.range), invert: ig.center, noise8: 0 },
                lane: L_IG,
            });
            p.ig = Some(fx(ig.blend, ig.opacity, ig.color));
        }
        let sa = &st.satin;
        if sa.enabled {
            let t = (sa.angle as f64).to_radians();
            let d = dist(sa.distance) as f64;
            let (dx, dy) = ((t.cos() * d).round() as i32, (-t.sin() * d).round() as i32);
            p.jobs.push(Job { shape: ShapeOp { src: Src::Alpha { invert: false }, dil: None, invert_after: false, boxes: boxes(reach(sz(sa.size))) }, fin: Fin::Satin { dx, dy, invert: sa.invert }, lane: L_SAT });
            p.satin = Some(fx(sa.blend, sa.opacity, sa.color));
        }
        let co = &st.color_overlay;
        if co.enabled {
            p.co = Some(fx(co.blend, co.opacity, co.color));
        }
        let go = &st.gradient_overlay;
        if go.enabled {
            let b = if bounds.is_empty() { RectI::new(0, 0, 1, 1) } else { bounds };
            let (w, h) = (b.w() as f32 / k, b.h() as f32 / k);
            let t = go.angle.to_radians();
            let (ux, uy) = (t.cos(), -t.sin());
            let scale = (go.scale / 100.0).clamp(0.1, 1.5);
            let len = if go.radial { w.max(h) * 0.5 * scale } else { ((w * ux).abs() + (h * uy).abs()) * scale };
            p.go = Some(GradFx {
                mode: go.blend,
                op8: op8(go.opacity),
                table: grad_table(go.from, go.to),
                radial: go.radial,
                reverse: go.reverse,
                cx: (b.x0 + b.x1) as f32 * 0.5 / k,
                cy: (b.y0 + b.y1) as f32 * 0.5 / k,
                ux,
                uy,
                inv_len: 1.0 / len.max(1e-3),
            });
        }
        let sk = &st.stroke;
        if sk.enabled && sk.size > 0.0 {
            let s = sz(sk.size);
            let (out_r, in_r) = match sk.position {
                StrokePos::Outside => (s, 0.0),
                StrokePos::Inside => (0.0, s),
                StrokePos::Center => (s * 0.5, s * 0.5),
            };
            let ring = Dil::new(out_r);
            let inner = Dil::new(in_r);
            let (has_ring, has_in) = (ring.is_some(), inner.is_some());
            let map = || Fin::Map { dx: 0, dy: 0, fill: 0, lut: None, invert: false, noise8: 0 };
            if let Some(d) = ring {
                p.jobs.push(Job { shape: ShapeOp { src: Src::Alpha { invert: false }, dil: Some(d), invert_after: false, boxes: vec![] }, fin: map(), lane: L_RING });
            }
            if let Some(d) = inner {
                p.jobs.push(Job { shape: ShapeOp { src: Src::Alpha { invert: true }, dil: Some(d), invert_after: false, boxes: vec![] }, fin: map(), lane: L_EIN });
            }
            if has_ring || has_in {
                p.stroke = Some((fx(sk.blend, sk.opacity, sk.color), has_ring, has_in));
            }
        }
        let bv = &st.bevel;
        if bv.enabled && bv.size > 0.0 {
            let s = sz(bv.size);
            let half = reach(s * 0.5).max(1);
            let shape = match bv.style {
                BevelStyle::Inner => ShapeOp { src: Src::Alpha { invert: true }, dil: Dil::new(s * 0.5), invert_after: true, boxes: boxes(half) },
                BevelStyle::Outer => ShapeOp { src: Src::Alpha { invert: false }, dil: Dil::new(s * 0.5), invert_after: false, boxes: boxes(half) },
                BevelStyle::Emboss | BevelStyle::Pillow => ShapeOp { src: Src::Alpha { invert: false }, dil: None, invert_after: false, boxes: boxes(half) },
            };
            let lut = bevel_lut(bv.depth.clamp(1.0, 1000.0), s, bv.angle, bv.altitude.clamp(0.0, 90.0), bv.up);
            let interior = bv.style != BevelStyle::Outer;
            let exterior = bv.style != BevelStyle::Inner;
            p.bevel = Some(BevelJob { shape, lut, interior, exterior, pillow: bv.style == BevelStyle::Pillow });
            let soft = reach(sz(bv.soften));
            if soft > 0 {
                let lanes: &[u8] = match (interior, exterior) {
                    (true, true) => &[L_HI_IN, L_SH_IN, L_HI_OUT, L_SH_OUT],
                    (true, false) => &[L_HI_IN, L_SH_IN],
                    _ => &[L_HI_OUT, L_SH_OUT],
                };
                for &l in lanes {
                    p.soften.push(Job {
                        shape: ShapeOp { src: Src::Lane(l), dil: None, invert_after: false, boxes: boxes(soft) },
                        fin: Fin::Map { dx: 0, dy: 0, fill: 0, lut: None, invert: false, noise8: 0 },
                        lane: l,
                    });
                }
            }
            p.bevel_fx = Some((fx(bv.hi_blend, bv.hi_opacity, bv.hi_color), fx(bv.sh_blend, bv.sh_opacity, bv.sh_color)));
        }
        let mut pad = p.jobs.iter().map(Job::reach).max().unwrap_or(0);
        if let Some(b) = &p.bevel {
            pad = pad.max(b.shape.reach() + 1 + p.soften.iter().map(Job::reach).max().unwrap_or(0));
        }
        p.pad = if pad > 0 { pad as usize + 1 } else { 0 };
        p
    }
}

// ----- plans for layers (cached) --------------------------------------------------------

// Keys hold a copy of the style (compared by value), so a style edited in
// place can never be served a stale plan.
type PlanKey = (Option<LayerStyle>, u32, u32, RectI);

/// The plan for a layer at `level`, or None when the layer isn't styled.
pub fn plan_for(layer: &Layer, level: u32) -> Option<Arc<Plan>> {
    if !is_styled(layer) {
        return None;
    }
    let style = layer.style.as_deref().filter(|s| s.any());
    let bounds = if style.as_ref().is_some_and(|s| s.gradient_overlay.enabled) { layer_bounds(layer) } else { RectI::EMPTY };
    static CACHE: Mutex<Vec<(PlanKey, Arc<Plan>)>> = Mutex::new(Vec::new());
    let same = |k: &PlanKey| {
        let st = match (&k.0, &style) {
            (None, None) => true,
            (Some(a), Some(b)) => a == *b,
            _ => false,
        };
        st && k.1 == layer.fill.to_bits() && k.2 == level && k.3 == bounds
    };
    if let Some((_, p)) = CACHE.lock().unwrap().iter().find(|(k, _)| same(k)) {
        return Some(p.clone());
    }
    let plan = Arc::new(Plan::new(style, layer.fill, level, bounds));
    let mut c = CACHE.lock().unwrap();
    if c.len() >= 48 {
        c.remove(0);
    }
    c.push(((style.cloned(), layer.fill.to_bits(), level, bounds), plan.clone()));
    Some(plan)
}

/// Level-0 padding a pixel change needs on screen: the most any level's
/// effects reach, in level-0 pixels.
pub fn dirty_pad(layer: &Layer) -> i32 {
    if !layer.visible || !is_styled(layer) {
        return 0;
    }
    (0..8u32).filter_map(|l| plan_for(layer, l).map(|p| (p.pad as i32 + 1) << l)).max().unwrap_or(0)
}

/// Layer-local, level-0 bounds of the pixels with any alpha (cached by the
/// grid's tiles). Groups and containers use the canvas.
pub fn layer_bounds(layer: &Layer) -> RectI {
    let g = &layer.grid;
    if layer.kind.is_container() {
        return g.bounds();
    }
    let mut key: u64 = (g.w as u64) << 32 | g.h as u64;
    for t in &g.tiles {
        let p = t.as_ref().map_or(0, |t| Arc::as_ptr(t) as u64);
        key = (key ^ p).wrapping_mul(0x100_0000_01b3).rotate_left(7);
    }
    static CACHE: Mutex<Vec<(u64, RectI)>> = Mutex::new(Vec::new());
    if let Some((_, r)) = CACHE.lock().unwrap().iter().find(|(k, _)| *k == key) {
        return *r;
    }
    use super::grid::{TS, TS_I};
    let mut b = RectI::EMPTY;
    for ty in 0..g.rows as i32 {
        for tx in 0..g.cols as i32 {
            let Some(t) = g.get(tx, ty) else { continue };
            let tr = super::grid::TileGrid::tile_rect(tx, ty).intersect(&g.bounds());
            if !b.is_empty() && b.contains_rect(&tr) {
                continue;
            }
            let r = if let Some(c) = t.uniform {
                if px::alpha(c) == 0 { RectI::EMPTY } else { tr }
            } else {
                let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
                let pixels = t.read();
                for y in 0..tr.h() as usize {
                    for x in 0..tr.w() as usize {
                        if px::alpha(pixels[y * TS + x]) != 0 {
                            x0 = x0.min(x as i32);
                            x1 = x1.max(x as i32 + 1);
                            y0 = y0.min(y as i32);
                            y1 = y1.max(y as i32 + 1);
                        }
                    }
                }
                if x0 > x1 { RectI::EMPTY } else { RectI::new(tx * TS_I + x0, ty * TS_I + y0, tx * TS_I + x1, ty * TS_I + y1) }
            };
            if !r.is_empty() {
                b = if b.is_empty() { r } else { b.union(&r) };
            }
        }
    }
    let mut c = CACHE.lock().unwrap();
    if c.len() >= 16 {
        c.remove(0);
    }
    c.push((key, b));
    b
}

// ----- CPU planes -----------------------------------------------------------------------

#[inline]
pub fn mul255(a: u32, b: u32) -> u32 {
    let v = a * b + 128;
    (v + (v >> 8)) >> 8
}

/// The CPU's `hash01` bits (`filters::util::hash01` before the float).
#[inline]
pub fn hash_u(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    h >> 8
}

pub const NOISE_SEED: u32 = 0x5eed;

/// Exact 1-D min-plus transform with `256 (x - q)^2` (Felzenszwalb's lower
/// envelope with integer comparisons): `out[x] = min_q 256 (x - q)^2 + f[q]`
/// over finite `f[q]`; values at or above `cap` become INF.
fn dt1(f: &[u32], out: &mut [u32], v: &mut Vec<usize>, z: &mut Vec<(i64, i64)>, cap: u64) {
    v.clear();
    z.clear();
    let val = |q: usize| f[q] as i64 + 256 * (q * q) as i64;
    for q in 0..f.len() {
        if f[q] == INF {
            continue;
        }
        loop {
            match v.last() {
                None => {
                    v.push(q);
                    z.push((i64::MIN, 1));
                    break;
                }
                Some(&p) => {
                    // s = (val(q) - val(p)) / (512 (q - p)); pop while s <= z.last.
                    let num = val(q) - val(p);
                    let den = 512 * (q - p) as i64;
                    let (zn, zd) = *z.last().unwrap();
                    let le = zn != i64::MIN && (num as i128) * (zd as i128) <= (zn as i128) * (den as i128);
                    if le {
                        v.pop();
                        z.pop();
                        continue;
                    }
                    v.push(q);
                    z.push((num, den));
                    break;
                }
            }
        }
    }
    if v.is_empty() {
        out.fill(INF);
        return;
    }
    let mut k = 0;
    for (x, o) in out.iter_mut().enumerate() {
        // Advance while the next parabola's left boundary is below x.
        while k + 1 < v.len() && (z[k + 1].0 as i128) < (x as i128) * (z[k + 1].1 as i128) {
            k += 1;
        }
        let q = v[k];
        let d = x as i64 - q as i64;
        let r = (256 * d * d + f[q] as i64) as u64;
        *o = if r >= cap { INF } else { r as u32 };
    }
}

/// Disc dilation of plane `p` in place (`out = max(p, coverage)`).
pub fn dilate(p: &mut [u8], w: usize, h: usize, d: &Dil) {
    let cap = (d.w as u64 * d.w as u64) << 8;
    let mut g = vec![INF; w * h];
    let (mut v, mut z) = (Vec::new(), Vec::new());
    let mut f = vec![0u32; w.max(h)];
    for y in 0..h {
        let row = &p[y * w..(y + 1) * w];
        if row.iter().all(|&a| a == 0) {
            continue;
        }
        for (fx, &a) in f[..w].iter_mut().zip(row) {
            *fx = if a > 0 { 255 - a as u32 } else { INF };
        }
        dt1(&f[..w], &mut g[y * w..(y + 1) * w], &mut v, &mut z, cap);
    }
    let mut col = vec![0u32; h];
    let mut out = vec![0u32; h];
    for x in 0..w {
        let mut any = false;
        for y in 0..h {
            col[y] = g[y * w + x];
            any |= col[y] != INF;
        }
        if !any {
            continue;
        }
        dt1(&col, &mut out, &mut v, &mut z, cap);
        for y in 0..h {
            let c = d.coverage(out[y]);
            let o = &mut p[y * w + x];
            *o = (*o).max(c);
        }
    }
}

/// Box passes with these radii (vertical passes, then horizontal).
pub fn blur(p: &mut [u8], w: usize, h: usize, radii: &[u32]) {
    if radii.is_empty() {
        return;
    }
    let widths: Vec<usize> = radii.iter().map(|&r| 2 * r as usize + 1).collect();
    super::filters::fast::box_plane(p, w, h, &widths);
}

fn shape_plane(op: &ShapeOp, alpha: &[u8], lanes: &[Option<Vec<u8>>], w: usize, h: usize) -> Vec<u8> {
    let mut p: Vec<u8> = match op.src {
        Src::Alpha { invert: false } => alpha.to_vec(),
        Src::Alpha { invert: true } => alpha.iter().map(|&a| 255 - a).collect(),
        Src::Lane(l) => lanes[l as usize].clone().unwrap_or_else(|| vec![0; w * h]),
    };
    if let Some(d) = &op.dil {
        dilate(&mut p, w, h, d);
    }
    if op.invert_after {
        p.iter_mut().for_each(|v| *v = 255 - *v);
    }
    blur(&mut p, w, h, &op.boxes);
    p
}

fn finish(fin: &Fin, p: &[u8], w: usize, h: usize, origin: (i32, i32)) -> Vec<u8> {
    let at = |x: i32, y: i32, fill: u8| if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 { fill } else { p[y as usize * w + x as usize] };
    let mut out = vec![0u8; w * h];
    match fin {
        Fin::Map { dx, dy, fill, lut, invert, noise8 } => {
            for y in 0..h {
                for x in 0..w {
                    let mut v = at(x as i32 - dx, y as i32 - dy, *fill) as u32;
                    if let Some(l) = lut {
                        v = l[v as usize] as u32;
                    }
                    if *invert {
                        v = 255 - v;
                    }
                    if *noise8 > 0 {
                        let h8 = hash_u(origin.0 + x as i32, origin.1 + y as i32, NOISE_SEED) >> 16;
                        let j = (noise8 * h8 + 127) / 255;
                        v = mul255(v, 255 - j);
                    }
                    out[y * w + x] = v as u8;
                }
            }
        }
        Fin::Satin { dx, dy, invert } => {
            for y in 0..h {
                for x in 0..w {
                    let (x, y) = (x as i32, y as i32);
                    let a = at(x - dx, y - dy, 0) as i32;
                    let b = at(x + dx, y + dy, 0) as i32;
                    let v = (a - b).unsigned_abs() as u8;
                    out[y as usize * w + x as usize] = if *invert { 255 - v } else { v };
                }
            }
        }
    }
    out
}

/// The bevel lanes from the height plane `hgt`.
fn bevel_lanes(b: &BevelJob, hgt: &[u8], w: usize, h: usize, lanes: &mut [Option<Vec<u8>>]) {
    let n = w * h;
    let mut hi_in = vec![0u8; if b.interior { n } else { 0 }];
    let mut sh_in = vec![0u8; if b.interior { n } else { 0 }];
    let mut hi_out = vec![0u8; if b.exterior { n } else { 0 }];
    let mut sh_out = vec![0u8; if b.exterior { n } else { 0 }];
    let at = |x: i32, y: i32| hgt[y.clamp(0, h as i32 - 1) as usize * w + x.clamp(0, w as i32 - 1) as usize] as i32;
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let gx = at(x + 1, y) - at(x - 1, y);
            let gy = at(x, y + 1) - at(x, y - 1);
            let i = y as usize * w + x as usize;
            let e = b.lut[((gx + 255) * 511 + gy + 255) as usize];
            if b.interior {
                hi_in[i] = e as u8;
                sh_in[i] = (e >> 8) as u8;
            }
            if b.exterior {
                let e = if b.pillow { b.lut[((255 - gx) * 511 + 255 - gy) as usize] } else { e };
                hi_out[i] = e as u8;
                sh_out[i] = (e >> 8) as u8;
            }
        }
    }
    if b.interior {
        lanes[L_HI_IN as usize] = Some(hi_in);
        lanes[L_SH_IN as usize] = Some(sh_in);
    }
    if b.exterior {
        lanes[L_HI_OUT as usize] = Some(hi_out);
        lanes[L_SH_OUT as usize] = Some(sh_out);
    }
}

/// One pixel of `blend::blend_row` (same arithmetic).
#[inline]
pub fn blend_one(d: u32, s: u32, op: u32, mode: BlendMode) -> u32 {
    if op == 0 {
        return d;
    }
    if mode == BlendMode::Normal {
        if op == 255 {
            return px::over(s, d);
        }
        return if s != 0 { px::over(px::scale(s, op), d) } else { d };
    }
    if s == 0 {
        return d;
    }
    let s = if op == 255 { s } else { px::scale(s, op) };
    if px::alpha(d) == 0 {
        return s;
    }
    blend_px(d, s, mode)
}

#[inline]
fn premul(c: [u8; 3], a: u32) -> u32 {
    px::premultiply(c[0], c[1], c[2], a as u8)
}

/// One styled pixel from the content `c` and the lanes `l` at that pixel;
/// `(x, y)` are layer-local level coordinates (Gradient Overlay).
#[inline]
pub fn compose_px(p: &Plan, c: u32, l: &[u8; LANES], x: i32, y: i32) -> u32 {
    let blend_at = |destination, source, opacity, mode| {
        if mode == BlendMode::Dissolve {
            super::blend::dissolve_px(destination, source, opacity, x, y)
        } else {
            blend_one(destination, source, opacity, mode)
        }
    };
    let a = px::alpha(c);
    let ia = 255 - a;
    // Beneath: drop shadow (knocked out by the shape), outer glow.
    let mut b = 0u32;
    if let Some((fx, knock)) = &p.ds {
        let v = if *knock { mul255(l[L_DS as usize] as u32, ia) } else { l[L_DS as usize] as u32 };
        b = blend_at(b, premul(fx.color, v), fx.op8, fx.mode);
    }
    if let Some(fx) = &p.og {
        b = blend_at(b, premul(fx.color, mul255(l[L_OG as usize] as u32, ia)), fx.op8, fx.mode);
    }
    // The shape's interior.
    let i = if a == 0 {
        0
    } else if p.interior() {
        let [r, g, bb, _] = px::unpremultiply(c);
        let mut n = px::premultiply(r, g, bb, p.fill8 as u8);
        if let Some(go) = &p.go {
            n = blend_at(n, go.at(x, y), go.op8, go.mode);
        }
        if let Some(fx) = &p.co {
            n = blend_at(n, premul(fx.color, 255), fx.op8, fx.mode);
        }
        if let Some(fx) = &p.satin {
            n = blend_at(n, premul(fx.color, l[L_SAT as usize] as u32), fx.op8, fx.mode);
        }
        if let Some(fx) = &p.ig {
            n = blend_at(n, premul(fx.color, l[L_IG as usize] as u32), fx.op8, fx.mode);
        }
        if let Some(fx) = &p.is {
            n = blend_at(n, premul(fx.color, l[L_IS as usize] as u32), fx.op8, fx.mode);
        }
        if let Some((fx, _, true)) = &p.stroke {
            // Inside part: the shape minus its erosion, relative to the shape.
            let e = (a + l[L_EIN as usize] as u32).saturating_sub(255);
            let cov = (e * 255 + a / 2) / a;
            n = blend_at(n, premul(fx.color, cov), fx.op8, fx.mode);
        }
        if let (Some(bj), Some((hi, sh))) = (&p.bevel, &p.bevel_fx) {
            if bj.interior {
                n = blend_at(n, premul(hi.color, l[L_HI_IN as usize] as u32), hi.op8, hi.mode);
                n = blend_at(n, premul(sh.color, l[L_SH_IN as usize] as u32), sh.op8, sh.mode);
            }
        }
        px::scale(n, a)
    } else if p.fill8 == 255 {
        c
    } else {
        px::scale(c, p.fill8)
    };
    // Outside-only effects: the stroke ring and the outer bevel.
    let r = if ia == 0 || !p.exterior() {
        0
    } else {
        let mut o = 0u32;
        if let Some((fx, true, _)) = &p.stroke {
            o = blend_at(o, premul(fx.color, l[L_RING as usize] as u32), fx.op8, fx.mode);
        }
        if let (Some(bj), Some((hi, sh))) = (&p.bevel, &p.bevel_fx) {
            if bj.exterior {
                o = blend_at(o, premul(hi.color, l[L_HI_OUT as usize] as u32), hi.op8, hi.mode);
                o = blend_at(o, premul(sh.color, l[L_SH_OUT as usize] as u32), sh.op8, sh.mode);
            }
        }
        px::scale(o, ia)
    };
    px::over(i + r, b)
}

/// Computes every lane of the plan over the padded buffer.
pub fn lanes(plan: &Plan, src: &[u32], pw: usize, ph: usize, origin: (i32, i32)) -> Vec<Option<Vec<u8>>> {
    let mut lanes: Vec<Option<Vec<u8>>> = vec![None; LANES];
    if plan.jobs.is_empty() && plan.bevel.is_none() {
        return lanes;
    }
    let alpha: Vec<u8> = src.iter().map(|&p| px::alpha(p) as u8).collect();
    for j in &plan.jobs {
        let p = shape_plane(&j.shape, &alpha, &lanes, pw, ph);
        lanes[j.lane as usize] = Some(finish(&j.fin, &p, pw, ph, origin));
    }
    if let Some(b) = &plan.bevel {
        let hgt = shape_plane(&b.shape, &alpha, &lanes, pw, ph);
        bevel_lanes(b, &hgt, pw, ph, &mut lanes);
    }
    for j in &plan.soften {
        let p = shape_plane(&j.shape, &alpha, &lanes, pw, ph);
        lanes[j.lane as usize] = Some(finish(&j.fin, &p, pw, ph, origin));
    }
    lanes
}

/// Renders the styled layer: `src` is the layer's padded pixels (mask
/// applied), `pw x ph` with `plan.pad` on every side, `origin` the
/// layer-local level coordinate of `src[0]`. Writes the center to `out`.
pub fn render(plan: &Plan, src: &[u32], pw: usize, ph: usize, origin: (i32, i32), out: &mut [u32]) {
    let pad = plan.pad;
    let (cw, ch) = (pw - 2 * pad, ph - 2 * pad);
    let lanes = lanes(plan, src, pw, ph, origin);
    let mut l = [0u8; LANES];
    for y in 0..ch {
        for x in 0..cw {
            let i = (y + pad) * pw + x + pad;
            for (k, v) in l.iter_mut().enumerate() {
                if let Some(p) = &lanes[k] {
                    *v = p[i];
                }
            }
            out[y * cw + x] = compose_px(plan, src[i], &l, origin.0 + (x + pad) as i32, origin.1 + (y + pad) as i32);
        }
    }
}

#[cfg(test)]
mod tests;
