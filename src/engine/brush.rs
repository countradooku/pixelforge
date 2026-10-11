//! The dab engine shared by every painting tool.
//!
//! Like Photoshop, a stroke accumulates dabs into a per-stroke coverage mask
//! (flow builds up inside it), and the layer is recomputed as
//! `kernel(base, target, mask × opacity)`, so opacity caps the whole stroke
//! instead of stacking per dab. `base` is the layer's tile vector at stroke
//! start, which costs nothing to keep thanks to copy-on-write tiles.
//!
//! What a tool *does* is a [`Kernel`]:
//!
//! | kernel family | per-tile target (built once, on first touch) | apply |
//! |---|---|---|
//! | brush, pencil | — (constant color) | lerp toward color |
//! | eraser | — | scale alpha down |
//! | clone / pattern / history | the source pixels (row copies) | lerp toward target |
//! | dodge / burn / sponge | LUT(base) | lerp toward target |
//! | blur / sharpen | planar blur of the padded base tile | lerp toward target |
//! | color replacement, background eraser | recolor(base) / — | lerp, with a per-pixel stamp weight |
//! | healing | source + Poisson membrane, written per dab | lerp toward target |
//! | smudge, mixer | — (sequential: each dab reads the canvas) | direct write |
//!
//! Each input event's dabs are bucketed by tile and tiles are processed in
//! parallel; only the pixels inside each dab's bounding box are touched, so
//! the per-event cost is proportional to the dab area, never the image.
//! Input smoothing (lazy mouse + Catmull-Rom resampling) and pressure live
//! here once for every tool.
//!
//! Dabs stay on the CPU (no GPU path): a dab touches a few thousand pixels
//! of one or two tiles and the next dab depends on its result, so each GPU
//! submit (tens of µs, plus a readback for the display and history) would
//! cost more than the dab. The GPU still composites the stroke's layer.

use std::sync::OnceLock;

use rayon::prelude::*;

use super::grid::{RectI, SyncPtr, TS, TS_I, TSHIFT, Tile, TileGrid, TileRef, tile_mut};
use super::px;
use super::selection::{Selection, TileCoverage};

/// Where a dynamic takes its value from (Photoshop's "Control" menus).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Control {
    #[default]
    Off,
    Pressure,
    /// Pen tilt (1 with a mouse).
    Tilt,
    /// From full to the minimum over this many dabs.
    Fade(u16),
    /// The stroke's direction (angle only).
    Direction,
    /// The stroke's initial direction (angle only).
    Initial,
}

impl Control {
    pub const MENU: [Control; 6] = [Control::Off, Control::Pressure, Control::Tilt, Control::Fade(25), Control::Direction, Control::Initial];

    pub fn label(self) -> &'static str {
        match self {
            Control::Off => "Off",
            Control::Pressure => "Pressure",
            Control::Tilt => "Tilt",
            Control::Fade(_) => "Fade",
            Control::Direction => "Direction",
            Control::Initial => "Initial Dir.",
        }
    }

    /// The control's 0..1 factor for dab `idx` (direction controls give 1).
    fn factor(self, pressure: f32, tilt: f32, idx: u32) -> f32 {
        match self {
            Control::Pressure => pressure,
            Control::Tilt => tilt,
            Control::Fade(n) => (1.0 - idx as f32 / n.max(1) as f32).max(0.0),
            _ => 1.0,
        }
    }
}

/// How a texture (or the dual brush) combines with the tip's coverage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TexMode {
    #[default]
    Multiply,
    Subtract,
    Darken,
    Overlay,
    ColorDodge,
    ColorBurn,
    LinearBurn,
    HardMix,
    LinearHeight,
    Height,
}

impl TexMode {
    pub const ALL: [TexMode; 10] = [
        TexMode::Multiply,
        TexMode::Subtract,
        TexMode::Darken,
        TexMode::Overlay,
        TexMode::ColorDodge,
        TexMode::ColorBurn,
        TexMode::LinearBurn,
        TexMode::HardMix,
        TexMode::LinearHeight,
        TexMode::Height,
    ];

    pub fn label(self) -> &'static str {
        match self {
            TexMode::Multiply => "Multiply",
            TexMode::Subtract => "Subtract",
            TexMode::Darken => "Darken",
            TexMode::Overlay => "Overlay",
            TexMode::ColorDodge => "Color Dodge",
            TexMode::ColorBurn => "Color Burn",
            TexMode::LinearBurn => "Linear Burn",
            TexMode::HardMix => "Hard Mix",
            TexMode::LinearHeight => "Linear Height",
            TexMode::Height => "Height",
        }
    }

    /// Coverage `a` (0..1) combined with texture value `t` (0..1) at depth `d`.
    #[inline]
    pub fn apply(self, a: f32, t: f32, d: f32) -> f32 {
        let full = match self {
            TexMode::Multiply => a * t,
            TexMode::Subtract => (a - (1.0 - t)).max(0.0),
            TexMode::Darken => a.min(t),
            TexMode::Overlay => {
                if a < 0.5 {
                    2.0 * a * t
                } else {
                    1.0 - 2.0 * (1.0 - a) * (1.0 - t)
                }
            }
            TexMode::ColorDodge => (a / (1.0 - t * 0.999)).min(1.0) * if t > 0.0 { 1.0 } else { a.min(1.0) },
            TexMode::ColorBurn => (1.0 - (1.0 - a) / t.max(1e-3)).max(0.0),
            TexMode::LinearBurn => (a + t - 1.0).max(0.0),
            TexMode::HardMix => {
                if a + t >= 1.0 {
                    a
                } else {
                    0.0
                }
            }
            // Paint lands where the texture's height reaches the tip's pressure.
            TexMode::LinearHeight => a * (2.0 * (a + t) - 1.5).clamp(0.0, 1.0),
            TexMode::Height => {
                if t >= 1.0 - a {
                    a
                } else {
                    0.0
                }
            }
        };
        (a + (full - a) * d).clamp(0.0, 1.0)
    }
}

/// Shape Dynamics: jitter amounts are 0..1 (angle jitter 1 = ±180°).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ShapeDynamics {
    pub on: bool,
    pub size_jitter: f32,
    pub size_ctl: Control,
    pub min_size: f32,
    pub angle_jitter: f32,
    pub angle_ctl: Control,
    pub round_jitter: f32,
    pub round_ctl: Control,
    pub min_round: f32,
    pub flip_x_jitter: bool,
    pub flip_y_jitter: bool,
}

/// Scattering (the amount is [`BrushSettings::scatter`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scattering {
    pub both_axes: bool,
    pub ctl: Control,
    /// Dabs per spacing step (1..16).
    pub count: u8,
    pub count_jitter: f32,
    pub count_ctl: Control,
}

impl Default for Scattering {
    fn default() -> Self {
        Scattering { both_axes: false, ctl: Control::Off, count: 1, count_jitter: 0.0, count_ctl: Control::Off }
    }
}

/// Texture options (the pattern and depth are [`BrushSettings::texture`] /
/// [`BrushSettings::texture_depth`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureOpts {
    /// Pattern scale (1 = 100 %).
    pub scale: f32,
    pub mode: TexMode,
    pub invert: bool,
    /// -1..1.
    pub brightness: f32,
    pub contrast: f32,
    /// Texture each tip: per-dab depth with jitter and a minimum.
    pub each_tip: bool,
    pub min_depth: f32,
    pub depth_jitter: f32,
    pub depth_ctl: Control,
}

impl Default for TextureOpts {
    fn default() -> Self {
        TextureOpts { scale: 1.0, mode: TexMode::Multiply, invert: false, brightness: 0.0, contrast: 0.0, each_tip: false, min_depth: 0.0, depth_jitter: 0.0, depth_ctl: Control::Off }
    }
}

/// Dual Brush: a second tip's dabs, with their own size, spacing, scatter and
/// count, limit the primary stroke through `mode`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualBrush {
    pub on: bool,
    /// Registered sampled tip (0 = hard round).
    pub tip: u64,
    pub size: f32,
    pub spacing: f32,
    pub scatter: f32,
    pub both_axes: bool,
    pub count: u8,
    pub mode: TexMode,
    pub flip: bool,
}

impl Default for DualBrush {
    fn default() -> Self {
        DualBrush { on: false, tip: 0, size: 25.0, spacing: 0.25, scatter: 0.0, both_axes: false, count: 1, mode: TexMode::Multiply, flip: false }
    }
}

/// Color Dynamics: jitter between foreground and background and in HSB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorDynamics {
    pub on: bool,
    /// A new colour per dab (else once per stroke).
    pub per_tip: bool,
    pub fg_bg_jitter: f32,
    pub fg_bg_ctl: Control,
    pub hue_jitter: f32,
    pub sat_jitter: f32,
    pub bri_jitter: f32,
    /// -1..1: saturation shift.
    pub purity: f32,
    /// The background colour (foreground is the stroke's colour).
    pub bg: [u8; 3],
}

impl Default for ColorDynamics {
    fn default() -> Self {
        ColorDynamics { on: false, per_tip: true, fg_bg_jitter: 0.0, fg_bg_ctl: Control::Off, hue_jitter: 0.0, sat_jitter: 0.0, bri_jitter: 0.0, purity: 0.0, bg: [255, 255, 255] }
    }
}

/// Transfer: per-dab opacity and flow.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Transfer {
    pub on: bool,
    pub opacity_jitter: f32,
    pub opacity_ctl: Control,
    pub flow_jitter: f32,
    pub flow_ctl: Control,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushSettings {
    /// Diameter in image pixels.
    pub size: f32,
    pub hardness: f32,
    pub opacity: f32,
    pub flow: f32,
    /// Dab spacing as a fraction of the diameter.
    pub spacing: f32,
    /// Tip angle in degrees and roundness (1 = circle).
    pub angle: f32,
    pub roundness: f32,
    /// Random offset as a fraction of the diameter.
    pub scatter: f32,
    /// Built-in texture (0 = none) and its depth.
    pub texture: u8,
    pub texture_depth: f32,
    /// Lazy-mouse string length as a fraction of the diameter (0 = off).
    pub smoothing: f32,
    pub pressure_size: bool,
    pub pressure_opacity: bool,
    pub pressure_flow: bool,
    /// Hard aliased edges (Pencil).
    pub aliased: bool,
    /// Registered sampled tip (0 = the computed round tip).
    pub tip: u64,
    pub flip_x: bool,
    pub flip_y: bool,
    pub shape: ShapeDynamics,
    pub scattering: Scattering,
    pub tex: TextureOpts,
    pub dual: DualBrush,
    pub color: ColorDynamics,
    pub transfer: Transfer,
    /// Wet Edges: paint pools at the stroke's edges.
    pub wet_edges: bool,
    /// Build-up (airbrush): dabs keep landing while the pointer rests.
    pub build_up: bool,
    /// Noise on the soft edges of the tip.
    pub noise: bool,
    /// Keep the current texture when another preset is chosen.
    pub protect_texture: bool,
    /// Seed of the stroke's random dynamics (deterministic per stroke).
    pub seed: u32,
}

impl Default for BrushSettings {
    fn default() -> Self {
        BrushSettings {
            size: 40.0,
            hardness: 0.8,
            opacity: 1.0,
            flow: 1.0,
            spacing: 0.12,
            angle: 0.0,
            roundness: 1.0,
            scatter: 0.0,
            texture: 0,
            texture_depth: 0.5,
            smoothing: 0.0,
            pressure_size: true,
            pressure_opacity: false,
            pressure_flow: false,
            aliased: false,
            tip: 0,
            flip_x: false,
            flip_y: false,
            shape: ShapeDynamics::default(),
            scattering: Scattering::default(),
            tex: TextureOpts::default(),
            dual: DualBrush::default(),
            color: ColorDynamics::default(),
            transfer: Transfer::default(),
            wet_edges: false,
            build_up: false,
            noise: false,
            protect_texture: false,
            seed: 1,
        }
    }
}

/// A sampled brush tip: coverage 0..255 (255 = paint).
#[derive(Debug, PartialEq)]
pub struct TipImage {
    pub name: String,
    pub w: u32,
    pub h: u32,
    pub alpha: Vec<u8>,
}

fn tip_store() -> &'static std::sync::RwLock<std::collections::HashMap<u64, std::sync::Arc<TipImage>>> {
    static S: OnceLock<std::sync::RwLock<std::collections::HashMap<u64, std::sync::Arc<TipImage>>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

/// Registers a sampled tip; its id is a hash of its pixels (stable across
/// sessions, so presets can refer to it).
pub fn register_tip(t: TipImage) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in t.w.to_le_bytes().iter().chain(&t.h.to_le_bytes()).chain(&t.alpha) {
        h = (h ^ *b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    let id = h.max(1);
    tip_store().write().unwrap().entry(id).or_insert_with(|| std::sync::Arc::new(t));
    id
}

/// A registered tip.
pub fn tip(id: u64) -> Option<std::sync::Arc<TipImage>> {
    if id == 0 {
        return None;
    }
    tip_store().read().unwrap().get(&id).cloned()
}

pub const TEXTURES: [&str; 4] = ["None", "Canvas", "Noise", "Burlap"];

/// A tileable 256² texture (u8), generated once.
pub fn texture(i: u8) -> Option<&'static [u8]> {
    static TEX: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    if i == 0 {
        return None;
    }
    let all = TEX.get_or_init(|| {
        use super::filters::util::{fbm, hash01};
        let make = |f: &dyn Fn(usize, usize) -> f32| -> Vec<u8> {
            (0..TS * TS).map(|k| (f(k % TS, k / TS).clamp(0.0, 1.0) * 255.0) as u8).collect()
        };
        vec![
            // Canvas weave.
            make(&|x, y| {
                let wx = ((x as f32 * std::f32::consts::TAU / 8.0).sin() * 0.5 + 0.5).powf(0.6);
                let wy = ((y as f32 * std::f32::consts::TAU / 8.0).sin() * 0.5 + 0.5).powf(0.6);
                let over = if ((x / 8) + (y / 8)) % 2 == 0 { wx } else { wy };
                0.35 + 0.65 * over * (0.85 + 0.15 * hash01(x as i32, y as i32, 3))
            }),
            // Fine noise, tileable through a periodic hash.
            make(&|x, y| hash01(x as i32, y as i32, 11) * 0.7 + 0.3),
            // Burlap: coarse fibrous noise (fbm sampled on a torus).
            make(&|x, y| {
                let (u, v) = (x as f32 / TS as f32 * std::f32::consts::TAU, y as f32 / TS as f32 * std::f32::consts::TAU);
                let n = fbm(u.cos() * 6.0 + 20.0, v.cos() * 6.0 + u.sin() * 6.0 + 20.0 + v.sin() * 3.0, 4, 7);
                (n * 1.6).clamp(0.0, 1.0)
            }),
        ]
    });
    all.get(i as usize - 1).map(|v| v.as_slice())
}

const LUT_N: usize = 4096;

/// One dab: where, how big, and how strong.
#[derive(Clone, Copy, Debug)]
pub struct Dab {
    pub x: f32,
    pub y: f32,
    pub r: f32,
    /// Flow in 0..=65535.
    pub flow: u32,
    /// Opacity cap in 0..=255.
    pub opacity: u32,
    /// Tip angle (radians), roundness and flips of this dab.
    pub angle: f32,
    pub round: f32,
    pub flip: (bool, bool),
    /// Colour of this dab (Color Dynamics per tip), opaque premultiplied.
    pub color: Option<u32>,
    /// Texture depth of this dab (Texture Each Tip), 0..1.
    pub depth: f32,
    /// A Dual Brush dab (stamps the second mask).
    pub dual: bool,
}

impl Dab {
    /// A plain round dab.
    #[cfg(test)]
    pub fn new(x: f32, y: f32, r: f32, flow: u32, opacity: u32) -> Self {
        Dab { x, y, r, flow, opacity, angle: 0.0, round: 1.0, flip: (false, false), color: None, depth: 0.0, dual: false }
    }

    pub fn rect(&self) -> RectI {
        let r = self.r + 1.0;
        RectI::new((self.x - r).floor() as i32, (self.y - r).floor() as i32, (self.x + r).ceil() as i32, (self.y + r).ceil() as i32)
    }
}

/// The stroke-start pixels, readable anywhere.
pub struct BaseView<'a> {
    pub tiles: &'a [TileRef],
    pub cols: u32,
    pub w: u32,
    pub h: u32,
}

impl BaseView<'_> {
    #[inline]
    pub fn pixel(&self, x: i32, y: i32) -> u32 {
        if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
            return 0;
        }
        let i = ((y >> TSHIFT) as u32 * self.cols + (x >> TSHIFT) as u32) as usize;
        match &self.tiles[i] {
            Some(t) => t.read()[(((y & 255) as usize) << TSHIFT) | (x & 255) as usize],
            None => 0,
        }
    }

    /// Copies the `w x h` rect at `(x0, y0)` into `dst` (0 outside the image),
    /// one row slice per tile.
    pub fn read_region(&self, x0: i32, y0: i32, w: usize, h: usize, dst: &mut [u32]) {
        for (ry, row) in dst[..w * h].chunks_exact_mut(w).enumerate() {
            let y = y0 + ry as i32;
            if y < 0 || y >= self.h as i32 {
                row.fill(0);
                continue;
            }
            let mut x = x0;
            let x_end = x0 + w as i32;
            while x < x_end {
                let o = (x - x0) as usize;
                if x < 0 || x >= self.w as i32 {
                    let stop = if x < 0 { 0.min(x_end) } else { x_end };
                    row[o..(stop - x0) as usize].fill(0);
                    x = stop;
                    continue;
                }
                let tx = x >> TSHIFT;
                let run = ((tx + 1) * TS_I).min(x_end).min(self.w as i32) - x;
                let i = ((y >> TSHIFT) as u32 * self.cols + tx as u32) as usize;
                let dst = &mut row[o..o + run as usize];
                match &self.tiles[i] {
                    Some(t) => {
                        let s = (((y & 255) as usize) << TSHIFT) | (x & 255) as usize;
                        dst.copy_from_slice(&t.read()[s..s + run as usize]);
                    }
                    None => dst.fill(0),
                }
                x += run;
            }
        }
    }

}

/// Falloff of a dab at a pixel, shared by the engine and sequential kernels.
#[derive(Clone, Copy)]
pub struct DabShape<'a> {
    cx: f32,
    cy: f32,
    inv_r2: f32,
    cos: f32,
    sin: f32,
    inv_round: f32,
    round: bool,
    aliased: bool,
    lut: &'a [u16],
    /// Sampled tip, its pixels per unit radius and the flip signs.
    tip: Option<&'a TipImage>,
    tip_k: f32,
    fx: f32,
    fy: f32,
}

impl DabShape<'_> {
    /// Falloff in 0..=65535 at pixel `(x, y)` (its center).
    #[inline(always)]
    pub fn at(&self, x: i32, y: i32) -> u32 {
        let dx = x as f32 + 0.5 - self.cx;
        let dy = y as f32 + 0.5 - self.cy;
        if let Some(t) = self.tip {
            let u = (dx * self.cos + dy * self.sin) * self.fx;
            let v = (dy * self.cos - dx * self.sin) * self.inv_round * self.fy;
            let sx = u * self.tip_k + t.w as f32 * 0.5 - 0.5;
            let sy = v * self.tip_k + t.h as f32 * 0.5 - 0.5;
            return sample_tip(t, sx, sy);
        }
        let q = if self.round {
            (dx * dx + dy * dy) * self.inv_r2
        } else {
            let u = dx * self.cos + dy * self.sin;
            let v = (dy * self.cos - dx * self.sin) * self.inv_round;
            (u * u + v * v) * self.inv_r2
        };
        if q >= 1.0 {
            return 0;
        }
        if self.aliased {
            return 65535;
        }
        self.lut[(q * (LUT_N - 1) as f32) as usize] as u32
    }
}

/// Bilinear coverage (0..=65535) of a sampled tip at tip pixel `(sx, sy)`.
#[inline]
fn sample_tip(t: &TipImage, sx: f32, sy: f32) -> u32 {
    let (w, h) = (t.w as i32, t.h as i32);
    if sx <= -1.0 || sy <= -1.0 || sx >= w as f32 || sy >= h as f32 {
        return 0;
    }
    let (x0, y0) = (sx.floor() as i32, sy.floor() as i32);
    let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
    let at = |x: i32, y: i32| if x < 0 || y < 0 || x >= w || y >= h { 0.0 } else { t.alpha[(y * w + x) as usize] as f32 };
    let top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * fx;
    let bot = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * fx;
    ((top + (bot - top) * fy) * 257.0) as u32
}

/// The shape of dab `d`: its angle, roundness and flips, with a sampled
/// `tip` when given.
pub fn dab_shape_tip<'a>(s: &BrushSettings, d: &Dab, lut: &'a [u16], tip: Option<&'a TipImage>) -> DabShape<'a> {
    let (sin, cos) = d.angle.sin_cos();
    let round = d.round.clamp(0.01, 1.0);
    DabShape {
        cx: d.x,
        cy: d.y,
        inv_r2: 1.0 / (d.r * d.r),
        cos,
        sin,
        inv_round: 1.0 / round,
        round: round >= 0.999,
        aliased: s.aliased,
        lut,
        tip_k: tip.map_or(1.0, |t| t.w.max(t.h) as f32 * 0.5 / d.r),
        tip,
        fx: if d.flip.0 { -1.0 } else { 1.0 },
        fy: if d.flip.1 { -1.0 } else { 1.0 },
    }
}

/// Writes per-dab targets into stroke tiles (healing).
pub struct TargetWriter<'a> {
    tiles: &'a mut [Option<StrokeTile>],
    cols: u32,
    selection: Option<&'a Selection>,
}

impl TargetWriter<'_> {
    /// Copies `src` (`w x h` at `(x0, y0)`) into the target buffers.
    pub fn write(&mut self, x0: i32, y0: i32, w: usize, h: usize, src: &[u32], rows: u32) {
        let r = RectI::wh(x0, y0, w as i32, h as i32);
        let (tx0, ty0, tx1, ty1) = r.tile_span(self.cols, rows);
        for ty in ty0..ty1 {
            for tx in tx0..tx1 {
                let i = (ty as u32 * self.cols + tx as u32) as usize;
                let sel = self.selection;
                let st = self.tiles[i].get_or_insert_with(|| StrokeTile::new(sel, tx, ty));
                let t = st.target.get_or_insert_with(|| vec![0u32; TS * TS].into_boxed_slice());
                let tr = TileGrid::tile_rect(tx, ty).intersect(&r);
                for y in tr.y0..tr.y1 {
                    let ly = (y - ty * TS_I) as usize;
                    let lx = (tr.x0 - tx * TS_I) as usize;
                    let sy = (y - y0) as usize;
                    let sx = (tr.x0 - x0) as usize;
                    let n = tr.w() as usize;
                    t[ly * TS + lx..ly * TS + lx + n].copy_from_slice(&src[sy * w + sx..sy * w + sx + n]);
                }
            }
        }
    }
}

/// A per-dab weight mask (see [`Kernel::dab_mask`]).
pub struct DabMask {
    pub rect: RectI,
    pub w: Box<[u8]>,
}

impl DabMask {
    #[inline]
    fn at(&self, x: i32, y: i32) -> u32 {
        if x < self.rect.x0 || y < self.rect.y0 || x >= self.rect.x1 || y >= self.rect.y1 {
            return 0;
        }
        self.w[((y - self.rect.y0) * self.rect.w() + (x - self.rect.x0)) as usize] as u32
    }
}

/// What a painting tool does to pixels. See the module docs.
pub trait Kernel: Send + Sync {
    /// Builds the per-tile target on first touch (None = no target).
    fn prepare(&self, _tx: i32, _ty: i32, _base: Option<&Tile>) -> Option<Box<[u32]>> {
        None
    }
    /// Writes one row of the stroke result: `alpha` is the final per-pixel
    /// strength (mask × opacity × selection × texture), `target` the
    /// prepared target row if any.
    fn apply_row(&self, base: &[u32], target: Option<&[u32]>, alpha: &[u8], out: &mut [u32]);
    /// Per-pixel stamp weight from the stroke-start pixel (0..=255).
    fn weight(&self, _base_px: u32) -> u32 {
        255
    }
    fn has_weight(&self) -> bool {
        false
    }
    /// Called before each dab is stamped (sampling, state).
    fn on_dab(&mut self, _d: &Dab, _base: &BaseView) {}
    /// Optional position-dependent stamp weight for the dab just sampled in
    /// `on_dab` (Color Replacement limits): a 0..=255 mask over the rect,
    /// used instead of [`Kernel::weight`].
    fn dab_mask(&mut self, _d: &Dab, _base: &BaseView) -> Option<DabMask> {
        None
    }
    /// Kernels that write a per-dab target (healing); dabs are then
    /// composed one at a time so each sees the previous result.
    fn dynamic(&self) -> bool {
        false
    }
    fn dab_target(&mut self, _d: &Dab, _shape: &DabShape, _base: &BaseView, _cur: &TileGrid, _t: &mut TargetWriter) {}
    /// Sequential kernels write straight into the layer per dab.
    fn sequential(&self) -> bool {
        false
    }
    fn seq_dab(&mut self, _d: &Dab, _shape: &DabShape, _grid: &mut TileGrid, _sel: Option<&Selection>) -> RectI {
        RectI::EMPTY
    }
}

/// Paint with a flat color (Brush, Pencil, Quick Mask painting).
pub struct Paint {
    pub color: u32,
}

impl Kernel for Paint {
    fn apply_row(&self, base: &[u32], _t: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
        for ((o, &b), &a) in out.iter_mut().zip(base).zip(alpha) {
            *o = if a == 0 { b } else { px::lerp(b, self.color, a as u32) };
        }
    }
}

/// Erase to transparency.
pub struct Erase;

impl Kernel for Erase {
    fn apply_row(&self, base: &[u32], _t: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
        for ((o, &b), &a) in out.iter_mut().zip(base).zip(alpha) {
            *o = if a == 0 { b } else { px::scale(b, 255 - a as u32) };
        }
    }
}

/// Lerp toward a prepared target (shared by the target-based kernels).
#[inline]
pub fn lerp_rows(base: &[u32], target: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
    match target {
        None => out.copy_from_slice(base),
        Some(t) => {
            for (((o, &b), &tv), &a) in out.iter_mut().zip(base).zip(t).zip(alpha) {
                *o = match a {
                    0 => b,
                    255 => tv,
                    a => px::lerp(b, tv, a as u32),
                };
            }
        }
    }
}

struct StrokeTile {
    mask: Box<[u16]>,
    cov: TileCoverage,
    target: Option<Box<[u32]>>,
    prepared: bool,
    /// Per-pixel opacity cap (pressure → opacity).
    cap: Option<Box<[u8]>>,
    /// The Dual Brush's coverage.
    dual: Option<Box<[u16]>>,
}

impl StrokeTile {
    fn new(sel: Option<&Selection>, tx: i32, ty: i32) -> Self {
        StrokeTile {
            mask: vec![0u16; TS * TS].into_boxed_slice(),
            cov: match sel {
                Some(s) => s.tile_coverage(tx, ty),
                None => TileCoverage::Full,
            },
            target: None,
            prepared: false,
            cap: None,
            dual: None,
        }
    }
}

/// Lazy mouse + centripetal Catmull-Rom input smoothing.
struct Smoother {
    string: f32,
    pos: Option<(f32, f32)>,
    ctrl: Vec<(f32, f32, f32)>,
}

impl Smoother {
    /// Points (x, y, pressure) the brush should walk through for one input.
    fn push(&mut self, x: f32, y: f32, pr: f32) -> Vec<(f32, f32, f32)> {
        if self.string <= 0.0 {
            return vec![(x, y, pr)];
        }
        let (bx, by) = match self.pos {
            None => {
                self.pos = Some((x, y));
                self.ctrl.push((x, y, pr));
                return vec![(x, y, pr)];
            }
            Some(b) => b,
        };
        let (dx, dy) = (x - bx, y - by);
        let d = dx.hypot(dy);
        if d <= self.string {
            return vec![];
        }
        let k = (d - self.string) / d;
        let np = (bx + dx * k, by + dy * k);
        self.pos = Some(np);
        self.ctrl.push((np.0, np.1, pr));
        self.emit(false)
    }

    fn flush(&mut self) -> Vec<(f32, f32, f32)> {
        if self.string <= 0.0 || self.ctrl.len() < 2 {
            return vec![];
        }
        self.emit(true)
    }

    /// Emits the Catmull-Rom segment that just became fully determined.
    fn emit(&mut self, last: bool) -> Vec<(f32, f32, f32)> {
        let n = self.ctrl.len();
        let (i1, i2) = if last { (n - 2, n - 1) } else if n >= 3 { (n - 3, n - 2) } else { return vec![] };
        let c = |i: isize| self.ctrl[i.clamp(0, n as isize - 1) as usize];
        let (p0, p1, p2, p3) = (c(i1 as isize - 1), c(i1 as isize), c(i2 as isize), c(i2 as isize + 1));
        let len = (p2.0 - p1.0).hypot(p2.1 - p1.1);
        let steps = (len / 2.0).ceil().clamp(1.0, 256.0) as usize;
        let cr = |a: f32, b: f32, cc: f32, d: f32, t: f32| {
            0.5 * ((2.0 * b) + (-a + cc) * t + (2.0 * a - 5.0 * b + 4.0 * cc - d) * t * t + (-a + 3.0 * b - 3.0 * cc + d) * t * t * t)
        };
        let out = (1..=steps)
            .map(|k| {
                let t = k as f32 / steps as f32;
                (cr(p0.0, p1.0, p2.0, p3.0, t), cr(p0.1, p1.1, p2.1, p3.1, t), p1.2 + (p2.2 - p1.2) * t)
            })
            .collect();
        if self.ctrl.len() > 8 {
            self.ctrl.drain(..self.ctrl.len() - 4);
        }
        out
    }
}

pub struct Stroke {
    kernel: Box<dyn Kernel>,
    s: BrushSettings,
    radius: f32,
    lut: Vec<u16>,
    /// The Dual Brush's falloff (a hard round tip).
    dual_lut: Vec<u16>,
    tip: Option<std::sync::Arc<TipImage>>,
    dual_tip: Option<std::sync::Arc<TipImage>>,
    base: Vec<TileRef>,
    tiles: Vec<Option<StrokeTile>>,
    selection: Option<Selection>,
    cols: u32,
    rows: u32,
    w: u32,
    h: u32,
    img: RectI,
    last: Option<(f32, f32, f32)>,
    last_dir: (f32, f32),
    init_dir: Option<(f32, f32)>,
    since_dab: f32,
    since_dual: f32,
    dabs: u32,
    dual_dabs: u32,
    /// Pen tilt (0 = flat, 1 = upright). GPUI reports no tilt, so it stays
    /// 1, as with a mouse.
    tilt: f32,
    /// The stroke's colour (Color Dynamics' foreground).
    fg: [u8; 3],
    /// Dabs carry their own colour (Color Dynamics per tip).
    per_tip_color: bool,
    smoother: Smoother,
    pub dirty: RectI,
}

/// The falloff LUT of a round tip of `radius` and `hardness`, indexed by
/// normalized squared distance (no sqrt per pixel).
fn falloff_lut(radius: f32, hardness: f32) -> Vec<u16> {
    let hard = hardness.clamp(0.0, 1.0);
    (0..LUT_N)
        .map(|i| {
            let q = i as f32 / (LUT_N - 1) as f32;
            let d = q.sqrt();
            let soft = if d <= hard || hard >= 1.0 {
                1.0
            } else {
                let u = ((d - hard) / (1.0 - hard)).clamp(0.0, 1.0);
                let s = 1.0 - u * u * (3.0 - 2.0 * u);
                s * s.sqrt()
            };
            // One-pixel anti-aliased rim for hard brushes.
            let aa = (radius - d * radius + 0.5).clamp(0.0, 1.0);
            ((soft * aa) * 65535.0 + 0.5) as u16
        })
        .collect()
}

fn hsb_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m].map(|t| (t * 255.0).round().clamp(0.0, 255.0) as u8)
}

/// A texture's value (0..1) at image pixel `(x, y)` with the texture options.
#[inline]
fn tex_value(t: &[u8], o: &TextureOpts, x: i32, y: i32) -> f32 {
    let k = 1.0 / o.scale.max(0.05);
    let (u, v) = (((x as f32 * k) as i32) & 255, ((y as f32 * k) as i32) & 255);
    let mut val = t[v as usize * TS + u as usize] as f32 / 255.0;
    if o.invert {
        val = 1.0 - val;
    }
    ((val - 0.5) * (1.0 + o.contrast) + 0.5 + o.brightness).clamp(0.0, 1.0)
}

impl Stroke {
    /// `_layer` documents which layer the stroke targets (the caller keeps it).
    pub fn new(_layer: usize, grid: &TileGrid, settings: BrushSettings, kernel: Box<dyn Kernel>, selection: Option<Selection>) -> Self {
        let radius = (settings.size * 0.5).max(0.5);
        Stroke {
            kernel,
            s: settings,
            radius,
            lut: falloff_lut(radius, settings.hardness),
            dual_lut: falloff_lut((settings.dual.size * 0.5).max(0.5), 1.0),
            tip: tip(settings.tip),
            dual_tip: tip(settings.dual.tip),
            base: grid.tiles.clone(),
            tiles: (0..grid.tiles.len()).map(|_| None).collect(),
            selection,
            cols: grid.cols,
            rows: grid.rows,
            w: grid.w,
            h: grid.h,
            img: grid.bounds(),
            last: None,
            last_dir: (1.0, 0.0),
            init_dir: None,
            since_dab: 0.0,
            since_dual: 0.0,
            dabs: 0,
            dual_dabs: 0,
            tilt: 1.0,
            fg: [0, 0, 0],
            per_tip_color: false,
            smoother: Smoother { string: settings.smoothing.clamp(0.0, 1.0) * settings.size, pos: None, ctrl: vec![] },
            dirty: RectI::EMPTY,
        }
    }

    /// The classic brush / eraser stroke. Color Dynamics jitter `rgb`
    /// toward their background colour, once per stroke or per tip.
    pub fn brush(layer: usize, grid: &TileGrid, settings: BrushSettings, rgb: [u8; 3], erase: bool, selection: Option<Selection>) -> Self {
        let mut st = Self::new(layer, grid, settings, Box::new(Erase), selection);
        st.fg = rgb;
        if !erase {
            let c = &settings.color;
            let color = if c.on && !c.per_tip { st.dyn_color(0, 1.0) } else { rgb };
            st.kernel = Box::new(Paint { color: px::pack(color[0], color[1], color[2], 255) });
            st.per_tip_color = c.on && c.per_tip;
        }
        st
    }

    pub fn last_point(&self) -> Option<(f32, f32)> {
        self.last.map(|l| (l.0, l.1))
    }

    /// Dabs placed so far.
    pub fn dab_count(&self) -> u32 {
        self.dabs
    }

    /// Extends the stroke to `(x, y)` with full pressure.
    pub fn add_point(&mut self, x: f32, y: f32, grid: &mut TileGrid) -> Vec<RectI> {
        self.add_point_p(x, y, 1.0, grid)
    }

    /// Extends the stroke to `(x, y)` at `pressure` (0..1), painting into
    /// `grid`. Returns the level-0 rects that changed.
    pub fn add_point_p(&mut self, x: f32, y: f32, pressure: f32, grid: &mut TileGrid) -> Vec<RectI> {
        let pts = self.smoother.push(x, y, pressure.clamp(0.0, 1.0));
        self.walk(&pts, grid)
    }

    /// Build-up (airbrush): one more dab where the pointer rests. Does
    /// nothing unless the settings ask for build-up.
    pub fn tick(&mut self, grid: &mut TileGrid) -> Vec<RectI> {
        let Some((x, y, pr)) = self.last.filter(|_| self.s.build_up) else { return vec![] };
        let mut dabs = Vec::new();
        self.emit(x, y, pr, self.last_dir, &mut dabs);
        self.stamp_dabs(dabs, grid)
    }

    /// Ends the stroke: flushes the smoothing tail and drops tiles the stroke
    /// left fully transparent. Returns the rects the flush changed.
    pub fn finish(mut self, grid: &mut TileGrid) -> Vec<RectI> {
        let pts = self.smoother.flush();
        let rects = self.walk(&pts, grid);
        for (i, st) in self.tiles.iter().enumerate() {
            if st.is_some() {
                if let Some(t) = &grid.tiles[i] {
                    if t.read().iter().all(|&p| p == 0) {
                        grid.tiles[i] = None;
                    }
                }
            }
        }
        rects
    }

    /// Deterministic 0..1 random number `k` of dab `idx` (per stroke seed).
    #[inline]
    fn rnd(&self, idx: u32, k: i32) -> f32 {
        super::filters::util::hash01(idx as i32, k, self.s.seed.wrapping_mul(0x9e37_79b9) ^ 0x51ed)
    }

    /// Color Dynamics: the colour of dab `idx` at pressure `pr`.
    fn dyn_color(&self, idx: u32, pr: f32) -> [u8; 3] {
        let c = &self.s.color;
        let t = match c.fg_bg_ctl {
            Control::Off => self.rnd(idx, 20) * c.fg_bg_jitter,
            ctl => (1.0 - ctl.factor(pr, self.tilt, idx)) * c.fg_bg_jitter.max(1e-3).min(1.0) / c.fg_bg_jitter.max(1e-3),
        };
        let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t.clamp(0.0, 1.0)).round() as u8;
        let rgb = [mix(self.fg[0], c.bg[0]), mix(self.fg[1], c.bg[1]), mix(self.fg[2], c.bg[2])];
        let [h, s, b] = super::color::srgb_to_hsb(rgb);
        let h = h as f32 + (self.rnd(idx, 21) * 2.0 - 1.0) * c.hue_jitter * 180.0;
        let s = (s as f32 / 100.0 + c.purity + (self.rnd(idx, 22) * 2.0 - 1.0) * c.sat_jitter).clamp(0.0, 1.0);
        let b = (b as f32 / 100.0 + (self.rnd(idx, 23) * 2.0 - 1.0) * c.bri_jitter).clamp(0.0, 1.0);
        hsb_to_rgb(h, s, b)
    }

    /// The dabs of one spacing step at `(x, y)`: shape dynamics, scattering
    /// (count, both axes), transfer, colour and per-tip texture depth.
    fn emit(&mut self, x: f32, y: f32, pr: f32, dir: (f32, f32), out: &mut Vec<Dab>) {
        let s = self.s;
        let tilt = self.tilt;
        let idx = self.dabs;
        let sh = &s.shape;
        // Size.
        let mut r = if s.pressure_size { (self.radius * pr).max(0.5) } else { self.radius };
        if sh.on {
            let c = sh.size_ctl.factor(pr, tilt, idx);
            r *= (sh.min_size + (1.0 - sh.min_size) * c) * (1.0 - sh.size_jitter * self.rnd(idx, 1));
        }
        let r = r.max(0.5);
        // Angle.
        let mut angle = s.angle.to_radians();
        if sh.on {
            match sh.angle_ctl {
                Control::Direction => angle += dir.1.atan2(dir.0),
                Control::Initial => angle += self.init_dir.map_or(0.0, |d| d.1.atan2(d.0)),
                Control::Off => {}
                ctl => angle += (1.0 - ctl.factor(pr, tilt, idx)) * std::f32::consts::PI,
            }
            angle += (self.rnd(idx, 2) * 2.0 - 1.0) * sh.angle_jitter * std::f32::consts::PI;
        }
        // Roundness and flips.
        let mut round = s.roundness;
        if sh.on {
            let c = sh.round_ctl.factor(pr, tilt, idx);
            round *= (sh.min_round + (1.0 - sh.min_round) * c) * (1.0 - sh.round_jitter * self.rnd(idx, 3));
            round = round.max(sh.min_round.max(0.01));
        }
        let flip = (s.flip_x ^ (sh.on && sh.flip_x_jitter && self.rnd(idx, 4) < 0.5), s.flip_y ^ (sh.on && sh.flip_y_jitter && self.rnd(idx, 5) < 0.5));
        // Transfer.
        let mut flow = s.flow.clamp(0.0, 1.0) * if s.pressure_flow { pr } else { 1.0 };
        let mut op = s.opacity.clamp(0.0, 1.0) * if s.pressure_opacity { pr } else { 1.0 };
        let tr = &s.transfer;
        if tr.on {
            op *= tr.opacity_ctl.factor(pr, tilt, idx) * (1.0 - tr.opacity_jitter * self.rnd(idx, 6));
            flow *= tr.flow_ctl.factor(pr, tilt, idx) * (1.0 - tr.flow_jitter * self.rnd(idx, 7));
        }
        let color = self.per_tip_color.then(|| {
            let c = self.dyn_color(idx, pr);
            px::pack(c[0], c[1], c[2], 255)
        });
        // Texture each tip.
        let tx = &s.tex;
        let depth = if tx.each_tip {
            let c = tx.depth_ctl.factor(pr, tilt, idx);
            s.texture_depth * (tx.min_depth + (1.0 - tx.min_depth) * c) * (1.0 - tx.depth_jitter * self.rnd(idx, 8))
        } else {
            s.texture_depth
        };
        // Scattering: `count` dabs, each offset across (and along) the path.
        let sc = &s.scattering;
        let mut n = sc.count.clamp(1, 16) as u32;
        if n > 1 && sc.count_jitter > 0.0 {
            let c = sc.count_ctl.factor(pr, tilt, idx);
            n -= ((n - 1) as f32 * sc.count_jitter * self.rnd(idx, 9) * c).round() as u32;
        }
        let amount = s.scatter * 2.0 * r * sc.ctl.factor(pr, tilt, idx);
        for k in 0..n {
            let (mut cx, mut cy) = (x, y);
            if amount > 0.0 {
                let j = (self.rnd(idx, 30 + 2 * k as i32) * 2.0 - 1.0) * amount;
                cx += -dir.1 * j;
                cy += dir.0 * j;
                if sc.both_axes {
                    let a = (self.rnd(idx, 31 + 2 * k as i32) * 2.0 - 1.0) * amount;
                    cx += dir.0 * a;
                    cy += dir.1 * a;
                }
            }
            out.push(Dab { x: cx, y: cy, r, flow: (flow * 65535.0 + 0.5) as u32, opacity: (op * 255.0 + 0.5) as u32, angle, round, flip, color, depth, dual: false });
        }
        self.dabs += 1;
    }

    /// The Dual Brush's dabs of one of its spacing steps.
    fn emit_dual(&mut self, x: f32, y: f32, dir: (f32, f32), out: &mut Vec<Dab>) {
        let d = self.s.dual;
        let idx = self.dual_dabs;
        self.dual_dabs += 1;
        let r = (d.size * 0.5).max(0.5);
        for k in 0..d.count.clamp(1, 16) as i32 {
            let (mut cx, mut cy) = (x, y);
            if d.scatter > 0.0 {
                let amount = d.scatter * 2.0 * r;
                let j = (self.rnd(idx, 60 + 2 * k) * 2.0 - 1.0) * amount;
                cx += -dir.1 * j;
                cy += dir.0 * j;
                if d.both_axes {
                    let a = (self.rnd(idx, 61 + 2 * k) * 2.0 - 1.0) * amount;
                    cx += dir.0 * a;
                    cy += dir.1 * a;
                }
            }
            let flip = d.flip && self.rnd(idx, 59) < 0.5;
            out.push(Dab { x: cx, y: cy, r, flow: 65535, opacity: 255, angle: 0.0, round: 1.0, flip: (flip, false), color: None, depth: 0.0, dual: true });
        }
    }

    /// Places dabs every `step` along `(lx, ly) → (x, y)`, carrying the
    /// distance since the last dab in `since`; returns (position, t).
    fn steps(len: f32, since: &mut f32, step_at: impl Fn(f32) -> f32) -> Vec<f32> {
        let mut out = Vec::new();
        let mut t = 0.0f32;
        loop {
            let step = step_at(t);
            let next = t + (step - *since);
            if next > len {
                *since += len - t;
                break;
            }
            t = next;
            *since = 0.0;
            out.push(t);
        }
        out
    }

    fn walk(&mut self, pts: &[(f32, f32, f32)], grid: &mut TileGrid) -> Vec<RectI> {
        let mut dabs: Vec<Dab> = Vec::new();
        let dual_on = self.s.dual.on;
        for &(x, y, pr) in pts {
            match self.last {
                None => {
                    self.emit(x, y, pr, (1.0, 0.0), &mut dabs);
                    if dual_on {
                        self.emit_dual(x, y, (1.0, 0.0), &mut dabs);
                    }
                    self.since_dab = 0.0;
                    self.since_dual = 0.0;
                }
                Some((lx, ly, lp)) => {
                    let (dx, dy) = (x - lx, y - ly);
                    let len = (dx * dx + dy * dy).sqrt();
                    if len > 0.0 {
                        let dir = (dx / len, dy / len);
                        self.last_dir = dir;
                        self.init_dir.get_or_insert(dir);
                        // Spacing follows the (pressure-scaled) size at each point.
                        let (size, spacing, psize) = (self.s.size, self.s.spacing.max(0.01), self.s.pressure_size);
                        let mut since = self.since_dab;
                        let ts = Self::steps(len, &mut since, |t| {
                            let prt = lp + (pr - lp) * (t / len).min(1.0);
                            ((if psize { size * prt } else { size }) * spacing).max(0.5)
                        });
                        self.since_dab = since;
                        for t in ts {
                            let prd = lp + (pr - lp) * (t / len);
                            self.emit(lx + dx * t / len, ly + dy * t / len, prd, dir, &mut dabs);
                        }
                        if dual_on {
                            let step = (self.s.dual.size * self.s.dual.spacing.max(0.01)).max(0.5);
                            let mut since = self.since_dual;
                            let ts = Self::steps(len, &mut since, |_| step);
                            self.since_dual = since;
                            for t in ts {
                                self.emit_dual(lx + dx * t / len, ly + dy * t / len, dir, &mut dabs);
                            }
                        }
                    }
                }
            }
            self.last = Some((x, y, pr));
        }
        self.stamp_dabs(dabs, grid)
    }

    fn stamp_dabs(&mut self, dabs: Vec<Dab>, grid: &mut TileGrid) -> Vec<RectI> {
        if dabs.is_empty() {
            return vec![];
        }
        let rects = if self.kernel.sequential() {
            let mut v = Vec::new();
            let lut = std::mem::take(&mut self.lut);
            let tipc = self.tip.clone();
            for d in dabs.iter().filter(|d| !d.dual) {
                let shape = dab_shape_tip(&self.s, d, &lut, tipc.as_deref());
                let r = self.kernel.seq_dab(d, &shape, grid, self.selection.as_ref());
                if !r.is_empty() {
                    v.push(r);
                }
            }
            self.lut = lut;
            v
        } else if self.kernel.dynamic() {
            let mut v = Vec::new();
            for d in &dabs {
                v.extend(self.apply_dabs(std::slice::from_ref(d), grid));
            }
            v
        } else {
            self.apply_dabs(&dabs, grid)
        };
        for r in &rects {
            self.dirty = self.dirty.union(r);
        }
        rects
    }

    fn apply_dabs(&mut self, dabs: &[Dab], grid: &mut TileGrid) -> Vec<RectI> {
        let masks: Vec<Option<DabMask>> = {
            let base = BaseView { tiles: &self.base, cols: self.cols, w: self.w, h: self.h };
            dabs.iter()
                .map(|d| {
                    if d.dual {
                        return None;
                    }
                    self.kernel.on_dab(d, &base);
                    self.kernel.dab_mask(d, &base)
                })
                .collect()
        };
        if self.kernel.dynamic() {
            let lut = std::mem::take(&mut self.lut);
            let tipc = self.tip.clone();
            for d in dabs.iter().filter(|d| !d.dual) {
                let base = BaseView { tiles: &self.base, cols: self.cols, w: self.w, h: self.h };
                let shape = dab_shape_tip(&self.s, d, &lut, tipc.as_deref());
                let mut tw = TargetWriter { tiles: &mut self.tiles, cols: self.cols, selection: self.selection.as_ref() };
                self.kernel.dab_target(d, &shape, &base, grid, &mut tw);
            }
            self.lut = lut;
        }
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for (di, d) in dabs.iter().enumerate() {
            let b = d.rect().intersect(&self.img);
            let (tx0, ty0, tx1, ty1) = b.tile_span(self.cols, self.rows);
            for ty in ty0..ty1 {
                for tx in tx0..tx1 {
                    pairs.push(((ty as u32) * self.cols + tx as u32, di as u32));
                }
            }
        }
        pairs.sort_unstable();
        let mut groups: Vec<(u32, usize, usize)> = Vec::new();
        let mut i = 0;
        while i < pairs.len() {
            let t = pairs[i].0;
            let s = i;
            while i < pairs.len() && pairs[i].0 == t {
                i += 1;
            }
            groups.push((t, s, i));
        }

        let mut stroke_tiles = std::mem::take(&mut self.tiles);
        let tiles_ptr = SyncPtr(grid.tiles.as_mut_ptr());
        let stroke_ptr = SyncPtr(stroke_tiles.as_mut_ptr());
        let me = &*self;
        let cols = me.cols;
        let tex = texture(me.s.texture);
        let cap_buf = me.s.pressure_opacity || me.s.transfer.on;
        let dirty = groups
            .par_iter()
            .filter_map(|&(ti, s, e)| {
                // SAFETY: each tile index appears in exactly one group.
                let slot = unsafe { tiles_ptr.at(ti as usize) };
                let st = unsafe { stroke_ptr.at(ti as usize) };
                let tx = (ti % cols) as i32;
                let ty = (ti / cols) as i32;
                let st = st.get_or_insert_with(|| StrokeTile::new(me.selection.as_ref(), tx, ty));
                if matches!(st.cov, TileCoverage::Empty) {
                    return None;
                }
                let base_tile = me.base[ti as usize].as_deref();
                if !st.prepared {
                    st.prepared = true;
                    if st.target.is_none() {
                        st.target = me.kernel.prepare(tx, ty, base_tile);
                    }
                }
                if cap_buf && st.cap.is_none() {
                    st.cap = Some(vec![0u8; TS * TS].into_boxed_slice());
                }
                let trect = TileGrid::tile_rect(tx, ty).intersect(&me.img);
                let mut u = RectI::EMPTY;
                for &(_, di) in &pairs[s..e] {
                    let d = &dabs[di as usize];
                    let b = d.rect().intersect(&trect);
                    if b.is_empty() {
                        continue;
                    }
                    u = u.union(&b);
                    me.stamp(st, base_tile, tx, ty, b, d, masks[di as usize].as_ref(), tex);
                }
                if u.is_empty() {
                    return None;
                }
                me.compose(slot, base_tile, st, tx, ty, u, tex);
                Some(u)
            })
            .collect();
        self.tiles = stroke_tiles;
        dirty
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn stamp(&self, st: &mut StrokeTile, base: Option<&Tile>, tx: i32, ty: i32, b: RectI, d: &Dab, dmask: Option<&DabMask>, tex: Option<&[u8]>) {
        let (lut, tipr) = if d.dual { (&self.dual_lut, self.dual_tip.as_deref()) } else { (&self.lut, self.tip.as_deref()) };
        let shape = dab_shape_tip(&self.s, d, lut, tipr);
        let weighted = !d.dual && self.kernel.has_weight() && dmask.is_none();
        let base_pixels = if weighted { base.map(Tile::read) } else { None };
        let flow = d.flow;
        let op = d.opacity as u8;
        let tex_tip = if !d.dual && self.s.tex.each_tip { tex } else { None };
        let noise = !d.dual && self.s.noise;
        let color = if d.dual { None } else { d.color };
        if color.is_some() && st.target.is_none() {
            st.target = Some(vec![0u32; TS * TS].into_boxed_slice());
        }
        let mask = if d.dual { st.dual.get_or_insert_with(|| vec![0u16; TS * TS].into_boxed_slice()) } else { &mut st.mask };
        for y in b.y0..b.y1 {
            let row = ((y - (ty << TSHIFT)) as usize) * TS;
            for x in b.x0..b.x1 {
                let mut a = shape.at(x, y);
                if a == 0 {
                    continue;
                }
                if noise && a < 65535 {
                    // Grain on the soft rim only.
                    let n = super::filters::util::hash01(x, y, 0x0b5e) * 1.2 + 0.4;
                    a = ((a as f32 * n) as u32).min(65535);
                }
                a = (a * flow + 32767) >> 16;
                if let Some(t) = tex_tip {
                    let v = self.s.tex.mode.apply(a as f32 / 65535.0, tex_value(t, &self.s.tex, x, y), d.depth);
                    a = (v * 65535.0) as u32;
                }
                let i = row + (x - (tx << TSHIFT)) as usize;
                if let Some(m) = dmask {
                    a = (a * m.at(x, y) + 127) / 255;
                } else if weighted {
                    let bp = base_pixels.as_ref().map_or(0, |p| p[i]);
                    a = (a * self.kernel.weight(bp) + 127) / 255;
                }
                if a == 0 {
                    continue;
                }
                let m = mask[i] as u32;
                let add = ((65535 - m) * a + 32767) >> 16;
                let nm = m + add;
                mask[i] = nm as u16;
                if let (Some(c), Some(t)) = (color, st.target.as_mut()) {
                    // The new dab's share of the accumulated coverage.
                    let w = if m == 0 { 255 } else { (add * 255 / nm.max(1)).min(255) };
                    t[i] = if m == 0 { c } else { px::lerp(t[i], c, w) };
                }
                if !d.dual {
                    if let Some(c) = &mut st.cap {
                        c[i] = c[i].max(op);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compose(&self, slot: &mut TileRef, base: Option<&Tile>, st: &StrokeTile, tx: i32, ty: i32, u: RectI, tex: Option<&[u8]>) {
        static ZERO: [u32; TS] = [0; TS];
        let dst = tile_mut(slot);
        let mut pin = dst.write();
        let pixels: &mut [u32] = &mut pin;
        let source_pin = base.map(|t| t.read());
        let source = source_pin.as_deref();
        let ox = tx * TS_I;
        let oy = ty * TS_I;
        let opacity = (self.s.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        let (lx0, lx1) = ((u.x0 - ox) as usize, (u.x1 - ox) as usize);
        let n = lx1 - lx0;
        let mut alpha = [0u8; TS];
        let mut out = [0u32; TS];
        // Stroke-level texture (unless applied per tip), dual brush, wet edges.
        let tex = if self.s.tex.each_tip { None } else { tex };
        let tex_depth = self.s.texture_depth.clamp(0.0, 1.0);
        let dual = if self.s.dual.on { Some(st.dual.as_deref()) } else { None };
        let wet = self.s.wet_edges;
        let fancy = dual.is_some() || wet || (tex.is_some() && self.s.tex != TextureOpts::default());
        let tex_lut = (255.0 * tex_depth) as u32;
        for y in u.y0..u.y1 {
            let ly = (y - oy) as usize;
            let row = ly * TS;
            for (k, a) in alpha[..n].iter_mut().enumerate() {
                let i = row + lx0 + k;
                let m = st.mask[i] as u32;
                if m == 0 {
                    *a = 0;
                    continue;
                }
                let cap = st.cap.as_ref().map_or(opacity, |c| c[i] as u32);
                let mut v = if fancy {
                    let x = ox + (lx0 + k) as i32;
                    let mut f = m as f32 / 65535.0;
                    if let Some(dm) = dual {
                        let dv = dm.map_or(0.0, |d| d[i] as f32 / 65535.0);
                        f = self.s.dual.mode.apply(f, dv, 1.0);
                    }
                    if let Some(t) = tex {
                        f = self.s.tex.mode.apply(f, tex_value(t, &self.s.tex, x, y), tex_depth);
                    }
                    if wet {
                        // Paint pools at the rim: the interior thins out.
                        f *= 1.0 - 0.55 * f * f * f;
                    }
                    ((f * 255.0) as u32 * cap + 127) / 255
                } else {
                    let mut v = ((m >> 8) * cap + 127) / 255;
                    if let Some(t) = tex {
                        let tv = t[((y & 255) as usize) * TS + ((lx0 + k + ox as usize) & 255)] as u32;
                        let f = 255 - (tex_lut * (255 - tv) + 127) / 255;
                        v = (v * f + 127) / 255;
                    }
                    v
                };
                let c = st.cov.at(i);
                if c != 255 {
                    v = (v * c + 127) / 255;
                }
                *a = v as u8;
            }
            let base_row: &[u32] = match &source {
                Some(t) => &t[row + lx0..row + lx1],
                None => &ZERO[..n],
            };
            let target_row = st.target.as_ref().map(|t| &t[row + lx0..row + lx1]);
            if self.per_tip_color {
                lerp_rows(base_row, target_row, &alpha[..n], &mut out[..n]);
            } else {
                self.kernel.apply_row(base_row, target_row, &alpha[..n], &mut out[..n]);
            }
            pixels[row + lx0..row + lx1].copy_from_slice(&out[..n]);
        }
    }
}

/// Every dynamic on: sampled tip, shape, scattering, texture per tip,
/// dual brush, colour per tip, transfer, wet edges and noise.
pub fn all_dynamics(size: f32, seed: u32) -> BrushSettings {
    let star = TipImage {
        name: "test star".into(),
        w: 33,
        h: 21,
        alpha: (0..33 * 21).map(|k| {
            let (x, y) = ((k % 33) as f32 - 16.0, (k / 33) as f32 - 10.0);
            if x.abs() < 3.0 || y.abs() < 2.0 || (x / 16.0).powi(2) + (y / 10.0).powi(2) < 0.3 { 255 } else { 0 }
        }).collect(),
    };
    let tip = register_tip(star);
    BrushSettings {
        size,
        spacing: 0.25,
        tip,
        scatter: 0.6,
        texture: 3,
        texture_depth: 0.8,
        shape: ShapeDynamics { on: true, size_jitter: 0.5, size_ctl: Control::Pressure, min_size: 0.2, angle_jitter: 0.3, angle_ctl: Control::Direction, round_jitter: 0.4, min_round: 0.3, flip_x_jitter: true, ..Default::default() },
        scattering: Scattering { both_axes: true, count: 3, count_jitter: 0.5, ..Default::default() },
        tex: TextureOpts { mode: TexMode::Subtract, scale: 1.5, each_tip: true, depth_jitter: 0.5, ..Default::default() },
        dual: DualBrush { on: true, size: size * 0.6, spacing: 0.15, scatter: 0.4, count: 2, ..Default::default() },
        color: ColorDynamics { on: true, per_tip: true, fg_bg_jitter: 0.5, hue_jitter: 0.2, sat_jitter: 0.2, bri_jitter: 0.2, bg: [250, 220, 40], ..Default::default() },
        transfer: Transfer { on: true, opacity_jitter: 0.5, flow_jitter: 0.5, ..Default::default() },
        wet_edges: true,
        noise: true,
        seed,
        ..Default::default()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_stroke_paints_color() {
        let mut g = TileGrid::new(512, 512);
        let mut s = Stroke::brush(0, &g, BrushSettings { size: 20.0, hardness: 1.0, ..Default::default() }, [255, 0, 0], false, None);
        let d = s.add_point(256.0, 256.0, &mut g);
        assert!(!d.is_empty());
        assert_eq!(g.pixel(256, 256), px::pack(255, 0, 0, 255));
        assert_eq!(g.pixel(300, 300), 0);
        // Crossing tile boundaries.
        s.add_point(200.0, 200.0, &mut g);
        assert_eq!(g.pixel(230, 230), px::pack(255, 0, 0, 255));
    }

    #[test]
    fn opacity_caps_overlapping_dabs() {
        let mut g = TileGrid::filled(256, 256, px::pack(255, 255, 255, 255));
        let mut s = Stroke::brush(
            0,
            &g,
            BrushSettings { size: 30.0, hardness: 1.0, opacity: 0.5, flow: 1.0, spacing: 0.05, ..Default::default() },
            [0, 0, 0],
            false,
            None,
        );
        s.add_point(100.0, 100.0, &mut g);
        s.add_point(110.0, 100.0, &mut g);
        s.add_point(100.0, 100.0, &mut g);
        let [r, _, _, _] = px::unpack(g.pixel(105, 100));
        assert!((r as i32 - 128).abs() <= 2, "r={r}");
    }

    #[test]
    fn pencil_is_aliased_and_roundness_squashes() {
        let mut g = TileGrid::new(256, 256);
        let st = BrushSettings { size: 21.0, hardness: 0.0, aliased: true, ..Default::default() };
        let mut s = Stroke::brush(0, &g, st, [0, 0, 255], false, None);
        s.add_point(100.0, 100.0, &mut g);
        for y in 85..115 {
            for x in 85..115 {
                let a = px::alpha(g.pixel(x, y));
                assert!(a == 0 || a == 255, "aliased pencil produced alpha {a}");
            }
        }
        // A flat (roundness 0.25) horizontal tip is wide but short.
        let mut g = TileGrid::new(256, 256);
        let st = BrushSettings { size: 40.0, hardness: 1.0, roundness: 0.25, ..Default::default() };
        let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, None);
        s.add_point(128.0, 128.0, &mut g);
        assert_eq!(px::alpha(g.pixel(145, 128)), 255);
        assert_eq!(px::alpha(g.pixel(128, 140)), 0);
    }

    #[test]
    fn pressure_scales_size_and_smoothing_follows_path() {
        let mut g = TileGrid::new(512, 512);
        let st = BrushSettings { size: 40.0, hardness: 1.0, pressure_size: true, ..Default::default() };
        let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, None);
        s.add_point_p(100.0, 100.0, 0.25, &mut g);
        assert_eq!(px::alpha(g.pixel(100, 100)), 255);
        assert_eq!(px::alpha(g.pixel(100, 110)), 0, "quarter-pressure dab too big");
        // With smoothing the stroke lags behind the cursor until flushed.
        let mut g = TileGrid::new(512, 512);
        let st = BrushSettings { size: 10.0, hardness: 1.0, smoothing: 1.0, ..Default::default() };
        let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, None);
        for k in 0..=40 {
            s.add_point(50.0 + k as f32 * 10.0, 200.0, &mut g);
        }
        assert_eq!(px::alpha(g.pixel(445, 200)), 0, "smoothed stroke reached the cursor");
        assert_eq!(px::alpha(g.pixel(200, 200)), 255);
        s.finish(&mut g);
    }

    fn render(st: BrushSettings) -> TileGrid {
        let mut g = TileGrid::new(512, 512);
        let mut s = Stroke::brush(0, &g, st, [20, 60, 200], false, None);
        for k in 0..=30 {
            let t = k as f32 / 30.0;
            s.add_point_p(60.0 + 390.0 * t, 256.0 + 120.0 * (t * 6.0).sin(), 0.4 + 0.6 * t, &mut g);
        }
        s.finish(&mut g);
        g
    }

    fn digest(g: &TileGrid) -> u64 {
        let mut v = vec![0u32; 512 * 512];
        g.read_region(0, 0, 512, 512, &mut v);
        v.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &p| (h ^ p as u64).wrapping_mul(0x0100_0000_01b3))
    }

    #[test]
    fn dynamics_are_deterministic_per_seed() {
        let (a, b, c) = (render(all_dynamics(60.0, 7)), render(all_dynamics(60.0, 7)), render(all_dynamics(60.0, 8)));
        assert_eq!(digest(&a), digest(&b), "same seed must repeat exactly");
        assert_ne!(digest(&a), digest(&c), "another seed must differ");
        // Reference render of the all-dynamics stroke (seed 7).
        assert_eq!(digest(&a), REFERENCE_ALL_DYNAMICS, "all-dynamics reference render changed");
    }

    const REFERENCE_ALL_DYNAMICS: u64 = 14526122168325237123;

    #[test]
    #[ignore]
    fn export_all_dynamics_reference() {
        let g = render(all_dynamics(60.0, 7));
        let l = crate::engine::document::Layer::new(1, "r", crate::engine::ops::matte(&g, [255, 255, 255]));
        crate::engine::io::export(&[l], 512, 512, std::path::Path::new("/tmp/pf_all_dynamics.png")).unwrap();
    }

    #[test]
    fn colour_dynamics_vary_per_tip_and_transfer_varies_alpha() {
        let g = render(all_dynamics(60.0, 3));
        let mut colors = std::collections::HashSet::new();
        let mut alphas = std::collections::HashSet::new();
        for y in (100..420).step_by(3) {
            for x in (40..470).step_by(3) {
                let p = px::unpremultiply(g.pixel(x, y));
                if p[3] > 40 {
                    colors.insert((p[0] / 16, p[1] / 16, p[2] / 16));
                }
                if p[3] > 0 {
                    alphas.insert(p[3] / 32);
                }
            }
        }
        assert!(colors.len() >= 6, "per-tip colours: {colors:?}");
        assert!(alphas.len() >= 5, "alpha levels: {alphas:?}");
    }

    #[test]
    fn sampled_tip_flip_fade_and_direction() {
        // An "L"-shaped tip: one dab shows the shape, flipped mirrors it.
        let l = TipImage { name: "L".into(), w: 20, h: 20, alpha: (0..400).map(|k| if k % 20 < 5 || k / 20 >= 15 { 255 } else { 0 }).collect() };
        let id = register_tip(l);
        let one = |flip_x: bool| {
            let mut g = TileGrid::new(256, 256);
            let st = BrushSettings { size: 40.0, tip: id, flip_x, pressure_size: false, ..Default::default() };
            let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, None);
            s.add_point(128.0, 128.0, &mut g);
            g
        };
        let (g, f) = (one(false), one(true));
        // Left column of the L is painted, the right part of its top isn't.
        assert_eq!(px::alpha(g.pixel(112, 112)), 255);
        assert_eq!(px::alpha(g.pixel(140, 112)), 0);
        assert_eq!(px::alpha(f.pixel(144, 112)), 255);
        assert_eq!(px::alpha(f.pixel(112, 112)), 0);
        // Fade: size shrinks to the minimum over 10 dabs.
        let mut g = TileGrid::new(1024, 256);
        let st = BrushSettings { size: 60.0, hardness: 1.0, spacing: 1.0, pressure_size: false, shape: ShapeDynamics { on: true, size_ctl: Control::Fade(10), min_size: 0.1, ..Default::default() }, ..Default::default() };
        let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, None);
        s.add_point(40.0, 128.0, &mut g);
        s.add_point(1000.0, 128.0, &mut g);
        assert_eq!(px::alpha(g.pixel(40, 100)), 255, "first dab at full size");
        assert_eq!(px::alpha(g.pixel(640, 110)), 0, "faded dab still large");
        // Direction: a flat tip turns with the stroke.
        let mut g = TileGrid::new(512, 512);
        let st = BrushSettings { size: 80.0, hardness: 1.0, roundness: 0.15, spacing: 0.05, pressure_size: false, shape: ShapeDynamics { on: true, angle_ctl: Control::Direction, ..Default::default() }, ..Default::default() };
        let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, None);
        s.add_point(256.0, 60.0, &mut g);
        s.add_point(256.0, 450.0, &mut g);
        // Vertical stroke with the tip along it: thin across, so x ± 20 is empty.
        assert_eq!(px::alpha(g.pixel(256, 250)), 255);
        assert_eq!(px::alpha(g.pixel(280, 250)), 0);
    }

    #[test]
    fn dual_brush_limits_wet_edges_and_build_up() {
        let line = |st: BrushSettings| {
            let mut g = TileGrid::new(512, 256);
            let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, None);
            s.add_point(60.0, 128.0, &mut g);
            s.add_point(450.0, 128.0, &mut g);
            g
        };
        let base = BrushSettings { size: 60.0, hardness: 1.0, pressure_size: false, ..Default::default() };
        let plain = line(base);
        let dual = line(BrushSettings { dual: DualBrush { on: true, size: 10.0, spacing: 3.0, ..Default::default() }, ..base });
        let cov = |g: &TileGrid| (0..512).flat_map(|x| (98..158).map(move |y| (x, y))).filter(|&(x, y)| px::alpha(g.pixel(x, y)) > 0).count();
        let (cp, cd) = (cov(&plain), cov(&dual));
        assert!(cd * 3 < cp, "dual brush should leave gaps: {cd} vs {cp}");
        // Wet edges: the rim is darker than the middle.
        let wet = line(BrushSettings { hardness: 0.0, wet_edges: true, ..base });
        let (mid, rim) = (px::alpha(wet.pixel(250, 128)), px::alpha(wet.pixel(250, 128 - 18)));
        assert!(rim > mid, "wet edges: rim {rim} vs middle {mid}");
        // Build-up keeps adding while the pointer rests.
        let mut g = TileGrid::new(256, 256);
        let st = BrushSettings { size: 40.0, flow: 0.2, build_up: true, pressure_size: false, ..Default::default() };
        let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, None);
        s.add_point(128.0, 128.0, &mut g);
        let a0 = px::alpha(g.pixel(128, 128));
        for _ in 0..10 {
            s.tick(&mut g);
        }
        assert!(px::alpha(g.pixel(128, 128)) > a0 + 60, "build-up: {a0} → {}", px::alpha(g.pixel(128, 128)));
    }

    #[test]
    fn texture_modes_differ_and_scattering_spreads() {
        let one = |mode: TexMode| {
            let mut g = TileGrid::new(256, 256);
            let st = BrushSettings { size: 120.0, hardness: 1.0, texture: 2, texture_depth: 0.7, tex: TextureOpts { mode, ..Default::default() }, ..Default::default() };
            let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, None);
            s.add_point(128.0, 128.0, &mut g);
            (100..156).map(|x| px::alpha(g.pixel(x, 128)) as u32).sum::<u32>()
        };
        let (m, sub, h) = (one(TexMode::Multiply), one(TexMode::Subtract), one(TexMode::Height));
        assert!(m != sub && sub != h, "{m} {sub} {h}");
        // Scattering on both axes with count 4 spreads dabs off the path.
        let mut g = TileGrid::new(512, 512);
        let st = BrushSettings { size: 20.0, hardness: 1.0, spacing: 0.5, scatter: 2.0, scattering: Scattering { both_axes: true, count: 4, ..Default::default() }, ..Default::default() };
        let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, None);
        s.add_point(100.0, 256.0, &mut g);
        s.add_point(400.0, 256.0, &mut g);
        let off = (0..512i32).flat_map(|x| (0..512i32).map(move |y| (x, y))).filter(|&(x, y)| (y - 256).abs() > 25 && px::alpha(g.pixel(x, y)) > 0).count();
        assert!(off > 500, "scattered coverage away from the path: {off}");
    }

    #[test]
    fn texture_modulates_and_selection_clips() {
        let mut g = TileGrid::filled(256, 256, px::pack(255, 255, 255, 255));
        let sel = Selection::rect(RectI::new(0, 0, 128, 256));
        let st = BrushSettings { size: 100.0, hardness: 1.0, texture: 2, texture_depth: 1.0, ..Default::default() };
        let mut s = Stroke::brush(0, &g, st, [0, 0, 0], false, Some(sel));
        s.add_point(128.0, 128.0, &mut g);
        assert_eq!(g.pixel(150, 128), px::pack(255, 255, 255, 255));
        let vals: Vec<u8> = (100..120).map(|x| px::unpack(g.pixel(x, 128))[0]).collect();
        assert!(vals.iter().any(|&v| v != vals[0]), "texture had no effect");
    }
}
