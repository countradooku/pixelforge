//! Gradients: multi-stop colour and opacity ramps, five shapes (linear,
//! radial, angle, reflected, diamond), ordered dither and reverse.
//!
//! The ramp is baked into a 1024-entry table of 8.8 fixed-point
//! premultiplied colours, so a pixel costs one parametric distance, one
//! lookup and (with dither) one ordered-dither add. A gradient is applied as
//! a position-dependent point filter (`Filter::Gradient`), so the live
//! on-canvas preview renders at the display mip level like every filter.

use super::px;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GradKind {
    #[default]
    Linear,
    Radial,
    Angle,
    Reflected,
    Diamond,
}

impl GradKind {
    pub const ALL: [GradKind; 5] = [GradKind::Linear, GradKind::Radial, GradKind::Angle, GradKind::Reflected, GradKind::Diamond];
    pub fn label(self) -> &'static str {
        match self {
            GradKind::Linear => "Linear",
            GradKind::Radial => "Radial",
            GradKind::Angle => "Angle",
            GradKind::Reflected => "Reflected",
            GradKind::Diamond => "Diamond",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorStop {
    pub pos: f32,
    pub color: [u8; 3],
    /// Midpoint to the next stop (0..1, 0.5 = linear).
    pub mid: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaStop {
    pub pos: f32,
    pub alpha: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    pub name: String,
    pub stops: Vec<ColorStop>,
    pub alpha: Vec<AlphaStop>,
}

impl Gradient {
    pub fn two(name: &str, a: [u8; 3], b: [u8; 3]) -> Gradient {
        Gradient {
            name: name.into(),
            stops: vec![ColorStop { pos: 0.0, color: a, mid: 0.5 }, ColorStop { pos: 1.0, color: b, mid: 0.5 }],
            alpha: vec![AlphaStop { pos: 0.0, alpha: 1.0 }, AlphaStop { pos: 1.0, alpha: 1.0 }],
        }
    }

    /// Photoshop's default presets (FG/BG ones use the current colours).
    pub fn presets(fg: [u8; 3], bg: [u8; 3]) -> Vec<Gradient> {
        let mut fg_t = Gradient::two("Foreground to Transparent", fg, fg);
        fg_t.alpha = vec![AlphaStop { pos: 0.0, alpha: 1.0 }, AlphaStop { pos: 1.0, alpha: 0.0 }];
        let spectrum = Gradient {
            name: "Spectrum".into(),
            stops: [[255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255], [255, 0, 255], [255, 0, 0]]
                .iter()
                .enumerate()
                .map(|(i, &c)| ColorStop { pos: i as f32 / 6.0, color: c, mid: 0.5 })
                .collect(),
            alpha: vec![AlphaStop { pos: 0.0, alpha: 1.0 }, AlphaStop { pos: 1.0, alpha: 1.0 }],
        };
        let copper = Gradient {
            name: "Copper".into(),
            stops: vec![
                ColorStop { pos: 0.0, color: [151, 70, 26], mid: 0.5 },
                ColorStop { pos: 0.3, color: [251, 216, 197], mid: 0.5 },
                ColorStop { pos: 0.83, color: [108, 46, 22], mid: 0.5 },
                ColorStop { pos: 1.0, color: [239, 219, 205], mid: 0.5 },
            ],
            alpha: vec![AlphaStop { pos: 0.0, alpha: 1.0 }, AlphaStop { pos: 1.0, alpha: 1.0 }],
        };
        let chrome = Gradient {
            name: "Chrome".into(),
            stops: vec![
                ColorStop { pos: 0.0, color: [41, 137, 204], mid: 0.5 },
                ColorStop { pos: 0.5, color: [255, 255, 255], mid: 0.5 },
                ColorStop { pos: 0.52, color: [144, 106, 0], mid: 0.5 },
                ColorStop { pos: 0.64, color: [217, 159, 0], mid: 0.5 },
                ColorStop { pos: 1.0, color: [255, 255, 255], mid: 0.5 },
            ],
            alpha: vec![AlphaStop { pos: 0.0, alpha: 1.0 }, AlphaStop { pos: 1.0, alpha: 1.0 }],
        };
        vec![Gradient::two("Foreground to Background", fg, bg), fg_t, Gradient::two("Black, White", [0, 0, 0], [255, 255, 255]), spectrum, copper, chrome]
    }

    fn color_at(&self, t: f32) -> [f32; 3] {
        let s = &self.stops;
        if s.is_empty() {
            return [0.0; 3];
        }
        if t <= s[0].pos {
            return s[0].color.map(|v| v as f32);
        }
        for w in s.windows(2) {
            let (a, b) = (w[0], w[1]);
            if t <= b.pos {
                let span = (b.pos - a.pos).max(1e-6);
                let mut u = (t - a.pos) / span;
                // Midpoint: remap so u = mid gives 50%.
                let m = a.mid.clamp(0.01, 0.99);
                u = if u < m { 0.5 * u / m } else { 0.5 + 0.5 * (u - m) / (1.0 - m) };
                return [0, 1, 2].map(|k| a.color[k] as f32 + (b.color[k] as f32 - a.color[k] as f32) * u);
            }
        }
        s[s.len() - 1].color.map(|v| v as f32)
    }

    fn alpha_at(&self, t: f32) -> f32 {
        let s = &self.alpha;
        if s.is_empty() {
            return 1.0;
        }
        if t <= s[0].pos {
            return s[0].alpha;
        }
        for w in s.windows(2) {
            if t <= w[1].pos {
                let u = (t - w[0].pos) / (w[1].pos - w[0].pos).max(1e-6);
                return w[0].alpha + (w[1].alpha - w[0].alpha) * u;
            }
        }
        s[s.len() - 1].alpha
    }

    /// Keeps stops sorted by position.
    pub fn sort(&mut self) {
        self.stops.sort_by(|a, b| a.pos.total_cmp(&b.pos));
        self.alpha.sort_by(|a, b| a.pos.total_cmp(&b.pos));
    }
}

const LUT: usize = 1024;

/// atan2 with max error ≈ 0.0015 rad (well under one of the 1024 LUT steps).
#[inline]
fn fast_atan2(y: f32, x: f32) -> f32 {
    use std::f32::consts::{FRAC_PI_2, PI};
    let (ax, ay) = (x.abs(), y.abs());
    let (mn, mx) = if ax < ay { (ax, ay) } else { (ay, ax) };
    if mx == 0.0 {
        return 0.0;
    }
    let a = mn / mx;
    let s = a * a;
    let mut r = ((-0.046_496_47 * s + 0.159_314_22) * s - 0.327_622_76) * s * a + a;
    if ay > ax {
        r = FRAC_PI_2 - r;
    }
    if x < 0.0 {
        r = PI - r;
    }
    if y < 0.0 { -r } else { r }
}

/// A gradient placed on the canvas.
#[derive(Clone, Debug, PartialEq)]
pub struct GradientFill {
    pub grad: Gradient,
    pub kind: GradKind,
    pub p0: (f32, f32),
    pub p1: (f32, f32),
    pub reverse: bool,
    pub dither: bool,
    pub opacity: f32,
    /// Premultiplied RGBA in 8.8 fixed point.
    lut: Vec<[u16; 4]>,
}

impl GradientFill {
    pub fn new(grad: Gradient, kind: GradKind, p0: (f32, f32), p1: (f32, f32), reverse: bool, dither: bool, opacity: f32) -> Self {
        let lut = (0..LUT)
            .map(|i| {
                let mut t = i as f32 / (LUT - 1) as f32;
                if reverse {
                    t = 1.0 - t;
                }
                let c = grad.color_at(t);
                let a = grad.alpha_at(t).clamp(0.0, 1.0);
                [c[0] * a, c[1] * a, c[2] * a, 255.0 * a].map(|v| (v * 256.0).round().clamp(0.0, 65535.0) as u16)
            })
            .collect();
        GradientFill { grad, kind, p0, p1, reverse, dither, opacity, lut }
    }

    /// The baked ramp: 1024 premultiplied RGBA entries in 8.8 fixed point.
    pub fn lut(&self) -> &[[u16; 4]] {
        &self.lut
    }

    /// Ramp parameter at a level-0 position.
    #[inline]
    pub fn t_at(&self, x: f32, y: f32) -> f32 {
        let (dx, dy) = (self.p1.0 - self.p0.0, self.p1.1 - self.p0.1);
        let len2 = (dx * dx + dy * dy).max(1e-6);
        let len = len2.sqrt();
        let (vx, vy) = (x - self.p0.0, y - self.p0.1);
        match self.kind {
            GradKind::Linear => ((vx * dx + vy * dy) / len2).clamp(0.0, 1.0),
            GradKind::Reflected => ((vx * dx + vy * dy) / len2).abs().min(1.0),
            GradKind::Radial => ((vx * vx + vy * vy).sqrt() / len).min(1.0),
            GradKind::Diamond => {
                // Distance in the frame rotated to the drag direction (L∞ norm).
                let (ux, uy) = (dx / len, dy / len);
                let a = (vx * ux + vy * uy).abs();
                let b = (-vx * uy + vy * ux).abs();
                (a.max(b) / len).min(1.0)
            }
            GradKind::Angle => {
                let base = dy.atan2(dx);
                let a = fast_atan2(vy, vx) - base;
                // Photoshop's angle gradient sweeps counter-clockwise from the drag.
                let t = (-a).rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU;
                t.clamp(0.0, 1.0)
            }
        }
    }

    /// The premultiplied gradient pixel at a position (dither by pixel).
    #[inline]
    pub fn pixel(&self, x: f32, y: f32, ix: i32, iy: i32) -> u32 {
        let t = self.t_at(x, y);
        let e = self.lut[(t * (LUT - 1) as f32 + 0.5) as usize];
        let d = if self.dither {
            const B: [u16; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
            B[((iy & 3) * 4 + (ix & 3)) as usize] * 16 + 8
        } else {
            128
        };
        let ch = |v: u16| ((v as u32 + d as u32) >> 8).min(255);
        let a = ch(e[3]);
        let c = |v: u16| ch(v).min(a);
        px::pack(c(e[0]) as u8, c(e[1]) as u8, c(e[2]) as u8, a as u8)
    }

    /// Ramp parameters for one row (kind dispatch hoisted out of the loop
    /// so each loop body is straight-line and vectorizes).
    fn row_t(&self, y: f32, x0: f32, step: f32, t: &mut [f32]) {
        let (dx, dy) = (self.p1.0 - self.p0.0, self.p1.1 - self.p0.1);
        let len2 = (dx * dx + dy * dy).max(1e-6);
        let len = len2.sqrt();
        let inv_len = 1.0 / len;
        let vy = y - self.p0.1;
        let n = t.len();
        match self.kind {
            GradKind::Linear | GradKind::Reflected => {
                let (kx, k0) = (dx / len2 * step, ((x0 - self.p0.0) * dx + vy * dy) / len2);
                let refl = self.kind == GradKind::Reflected;
                for (i, o) in t.iter_mut().enumerate().take(n) {
                    let v = k0 + kx * i as f32;
                    *o = if refl { v.abs().min(1.0) } else { v.clamp(0.0, 1.0) };
                }
            }
            GradKind::Radial => {
                for (i, o) in t.iter_mut().enumerate() {
                    let vx = x0 + step * i as f32 - self.p0.0;
                    *o = ((vx * vx + vy * vy).sqrt() * inv_len).min(1.0);
                }
            }
            GradKind::Diamond => {
                let (ux, uy) = (dx * inv_len, dy * inv_len);
                for (i, o) in t.iter_mut().enumerate() {
                    let vx = x0 + step * i as f32 - self.p0.0;
                    let a = (vx * ux + vy * uy).abs();
                    let b = (vy * ux - vx * uy).abs();
                    *o = (a.max(b) * inv_len).min(1.0);
                }
            }
            GradKind::Angle => {
                let base = dy.atan2(dx);
                const TAU: f32 = std::f32::consts::TAU;
                for (i, o) in t.iter_mut().enumerate() {
                    let vx = x0 + step * i as f32 - self.p0.0;
                    let a = -(fast_atan2(vy, vx) - base);
                    let a = a - TAU * (a / TAU).floor();
                    *o = (a / TAU).clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Composites the gradient over a `w x h` block at `origin` of `level`.
    pub fn apply(&self, buf: &mut [u32], w: usize, h: usize, level: u32, origin: (i32, i32)) {
        const B: [u16; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
        let s = (1u32 << level) as f32;
        let op = (self.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
        let mut t = vec![0f32; w];
        for y in 0..h {
            let iy = origin.1 + y as i32;
            let gy = (iy as f32 + 0.5) * s;
            self.row_t(gy, (origin.0 as f32 + 0.5) * s, s, &mut t);
            let row = &mut buf[y * w..(y + 1) * w];
            let drow = &B[((iy & 3) * 4) as usize..((iy & 3) * 4 + 4) as usize];
            for (x, (o, &tv)) in row.iter_mut().zip(&t).enumerate() {
                let e = self.lut[(tv * (LUT - 1) as f32 + 0.5) as usize];
                let d = if self.dither { drow[(origin.0 as usize + x) & 3] as u32 * 16 + 8 } else { 128 };
                let ch = |v: u16| ((v as u32 + d) >> 8).min(255);
                let a = ch(e[3]);
                let g = px::pack(ch(e[0]).min(a) as u8, ch(e[1]).min(a) as u8, ch(e[2]).min(a) as u8, a as u8);
                let g = if op == 255 { g } else { px::scale(g, op) };
                *o = px::over(g, *o);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_and_stops() {
        let g = Gradient::two("bw", [0, 0, 0], [255, 255, 255]);
        let lin = GradientFill::new(g.clone(), GradKind::Linear, (0.0, 0.0), (100.0, 0.0), false, false, 1.0);
        assert_eq!(px::unpack(lin.pixel(0.5, 5.0, 0, 5))[0], 1);
        assert!((px::unpack(lin.pixel(50.0, 5.0, 50, 5))[0] as i32 - 128).abs() <= 1);
        assert_eq!(px::unpack(lin.pixel(150.0, 5.0, 150, 5))[0], 255);
        let refl = GradientFill::new(g.clone(), GradKind::Reflected, (50.0, 0.0), (100.0, 0.0), false, false, 1.0);
        assert_eq!(refl.pixel(25.0, 0.0, 0, 0), refl.pixel(75.0, 0.0, 0, 0));
        let rad = GradientFill::new(g.clone(), GradKind::Radial, (0.0, 0.0), (100.0, 0.0), false, false, 1.0);
        assert_eq!(rad.pixel(0.0, 60.0, 0, 0), rad.pixel(60.0, 0.0, 0, 0));
        let dia = GradientFill::new(g.clone(), GradKind::Diamond, (0.0, 0.0), (100.0, 0.0), false, false, 1.0);
        assert_eq!(dia.pixel(50.0, 0.0, 0, 0), dia.pixel(50.0, 50.0, 0, 0));
        let ang = GradientFill::new(g, GradKind::Angle, (0.0, 0.0), (100.0, 0.0), false, false, 1.0);
        let q = px::unpack(ang.pixel(0.0, -50.0, 0, 0))[0] as i32;
        assert!((q - 64).abs() <= 2, "{q}");
        // Opacity stops: foreground to transparent.
        let mut ft = Gradient::two("ft", [255, 0, 0], [255, 0, 0]);
        ft.alpha = vec![AlphaStop { pos: 0.0, alpha: 1.0 }, AlphaStop { pos: 1.0, alpha: 0.0 }];
        let f = GradientFill::new(ft, GradKind::Linear, (0.0, 0.0), (100.0, 0.0), false, false, 1.0);
        assert!((px::alpha(f.pixel(50.0, 0.0, 0, 0)) as i32 - 128).abs() <= 1);
        // Multi-stop spectrum hits green at 1/3.
        let sp = &Gradient::presets([0; 3], [0; 3])[3];
        let s = GradientFill::new(sp.clone(), GradKind::Linear, (0.0, 0.0), (600.0, 0.0), false, false, 1.0);
        assert_eq!(px::unpack(s.pixel(200.0, 0.0, 0, 0)), [0, 255, 0, 255]);
    }

    #[test]
    fn fast_atan2_is_accurate() {
        for k in 0..1000 {
            let a = k as f32 * 0.0127;
            let (y, x) = (a.sin() * 3.0, a.cos() * 3.0);
            assert!((fast_atan2(y, x) - y.atan2(x)).abs() < 2e-3, "{a}");
        }
    }

    #[test]
    fn dither_breaks_banding_but_keeps_mean() {
        let g = Gradient::two("bw", [100, 100, 100], [101, 101, 101]);
        let f = GradientFill::new(g, GradKind::Linear, (0.0, 0.0), (1000.0, 0.0), false, true, 1.0);
        let vals: Vec<u8> = (0..16).map(|k| px::unpack(f.pixel(500.0, 0.0, k % 4, k / 4))[0]).collect();
        assert!(vals.contains(&100) && vals.contains(&101), "{vals:?}");
    }
}
