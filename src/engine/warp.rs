//! Warp Text: Photoshop's 15 warp styles.
//!
//! Every Photoshop warp is a **bicubic Bézier patch** over the text's box:
//! a 4 × 4 grid of control points. Fitting patches to Photoshop's own
//! warped outlines (exported through `tools/photoshop_gaps.jsx`, phase B)
//! leaves a residual of 0.01 px, which pins down each style's grid:
//!
//! - **Arch / Arc Upper / Arc Lower / Bulge** move the inner control points
//!   of each row by the standard cubic approximation of a circular arc over
//!   the box width, with angle `bend × 180°` (Arch: every row; Upper /
//!   Lower: blended from the top / bottom edge; Bulge: opposite on the two
//!   edges).
//! - **Arc** bends the box into an annulus sector: the bottom edge is that
//!   arc, the top edge the concentric arc `H` further out, corners pushed
//!   along the radii, rows interpolated. **Shell Upper / Lower** take Arc's
//!   outer rows on one side only.
//! - **Flag, Wave, Fish, Rise, Fisheye, Inflate, Squeeze, Twist** offset
//!   specific control points by `bend × W` or `bend × H` multiples.
//! - **Horizontal / vertical distortion** scale the base grid about the box
//!   centre per column / row (`1 ± h`), applied in that order, and scale the
//!   style's offsets per style as Photoshop does (measured, see the tests).

use super::vector::{P, p};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum WarpStyle {
    #[default]
    None,
    Arc,
    Bulge,
    Flag,
    Wave,
    Rise,
    Twist,
    ArcLower,
    ArcUpper,
    Arch,
    ShellLower,
    ShellUpper,
    Fish,
    Fisheye,
    Inflate,
    Squeeze,
}

impl WarpStyle {
    /// Storage order (`.pforge` stores the index; the first seven are the
    /// original PixelForge styles, so old files keep their style).
    pub const STORED: [WarpStyle; 16] = [
        WarpStyle::None,
        WarpStyle::Arc,
        WarpStyle::Bulge,
        WarpStyle::Flag,
        WarpStyle::Wave,
        WarpStyle::Rise,
        WarpStyle::Twist,
        WarpStyle::ArcLower,
        WarpStyle::ArcUpper,
        WarpStyle::Arch,
        WarpStyle::ShellLower,
        WarpStyle::ShellUpper,
        WarpStyle::Fish,
        WarpStyle::Fisheye,
        WarpStyle::Inflate,
        WarpStyle::Squeeze,
    ];
    /// Photoshop's menu order.
    pub const ALL: [WarpStyle; 16] = [
        WarpStyle::None,
        WarpStyle::Arc,
        WarpStyle::ArcLower,
        WarpStyle::ArcUpper,
        WarpStyle::Arch,
        WarpStyle::Bulge,
        WarpStyle::ShellLower,
        WarpStyle::ShellUpper,
        WarpStyle::Flag,
        WarpStyle::Wave,
        WarpStyle::Fish,
        WarpStyle::Rise,
        WarpStyle::Fisheye,
        WarpStyle::Inflate,
        WarpStyle::Squeeze,
        WarpStyle::Twist,
    ];

    pub fn label(self) -> &'static str {
        match self {
            WarpStyle::None => "No Warp",
            WarpStyle::Arc => "Arc",
            WarpStyle::ArcLower => "Arc Lower",
            WarpStyle::ArcUpper => "Arc Upper",
            WarpStyle::Arch => "Arch",
            WarpStyle::Bulge => "Bulge",
            WarpStyle::ShellLower => "Shell Lower",
            WarpStyle::ShellUpper => "Shell Upper",
            WarpStyle::Flag => "Flag",
            WarpStyle::Wave => "Wave",
            WarpStyle::Fish => "Fish",
            WarpStyle::Rise => "Rise",
            WarpStyle::Fisheye => "Fisheye",
            WarpStyle::Inflate => "Inflate",
            WarpStyle::Squeeze => "Squeeze",
            WarpStyle::Twist => "Twist",
        }
    }

    pub fn index(self) -> u8 {
        Self::STORED.iter().position(|&s| s == self).unwrap_or(0) as u8
    }

    pub fn from_index(i: u8) -> WarpStyle {
        Self::STORED.get(i as usize).copied().unwrap_or_default()
    }
}

/// A warp: style, bend and distortions in −1..1, and direction.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Warp {
    pub style: WarpStyle,
    pub bend: f32,
    pub h_distort: f32,
    pub v_distort: f32,
    /// Warp Direction: Vertical (the mesh is built transposed).
    pub vertical: bool,
}

impl Default for Warp {
    fn default() -> Self {
        Warp { style: WarpStyle::None, bend: 0.5, h_distort: 0.0, v_distort: 0.0, vertical: false }
    }
}

/// How a style's offsets respond to distortion (measured against Photoshop):
/// per component, scaled by nothing (`One`), the column factor (`Col`), the
/// row factor (`Row`) or the row factor at the displaced point (`RowAt`);
/// `lerp_rows` rebuilds the inner rows from the distorted edge rows. The `h`
/// laws only matter for Inflate and Squeeze (see `mesh`).
#[derive(Clone, Copy)]
enum S {
    One,
    Col,
    Row,
    RowAt,
    /// The factor of the nearer edge row (top half: 1 − v, bottom: 1 + v).
    Edge,
    /// The factor of the nearer edge column (left half: 1 − h, right: 1 + h).
    EdgeCol,
}

struct Laws {
    hx: S,
    hy: S,
    vx: S,
    vy: S,
    lerp_rows: bool,
}

fn laws(s: WarpStyle) -> Laws {
    use S::*;
    let l = |hx, hy, vx, vy, lerp_rows| Laws { hx, hy, vx, vy, lerp_rows };
    match s {
        WarpStyle::ArcLower => l(Col, One, Row, Row, true),
        WarpStyle::ArcUpper => l(One, One, Row, Row, true),
        WarpStyle::Bulge => l(One, Col, Row, Row, true),
        WarpStyle::Fish => l(One, Col, One, Row, true),
        WarpStyle::Fisheye => l(Col, Col, Row, Row, false),
        WarpStyle::Flag | WarpStyle::Rise => l(One, One, One, Row, false),
        WarpStyle::Inflate | WarpStyle::Squeeze => l(EdgeCol, One, One, Edge, false),
        WarpStyle::Twist => l(Col, Col, RowAt, Row, false),
        WarpStyle::Wave => l(One, Col, One, One, false),
        _ => l(One, One, Row, Row, false),
    }
}

/// Offsets of the first inner control point of a cubic arc over `chord`
/// (relative to its straight position `chord / 3`), bulging up for θ > 0.
fn arc_ctrl(theta: f32, chord: f32) -> (f32, f32) {
    if theta.abs() < 1e-6 {
        return (0.0, 0.0);
    }
    let a = theta.abs();
    let r = chord / (2.0 * (a / 2.0).sin());
    let k = 4.0 / 3.0 * (a / 4.0).tan() * r;
    (k * (a / 2.0).cos() - chord / 3.0, -k * (a / 2.0).sin() * theta.signum())
}

/// Arc's offset rows for a bend-up angle `a`: outer (top) and inner (bottom).
fn arc_rows(a: f32, w: f32, h: f32) -> ([(f32, f32); 4], [(f32, f32); 4]) {
    if a < 1e-6 {
        return ([(0.0, 0.0); 4], [(0.0, 0.0); 4]);
    }
    let (ax, ay) = arc_ctrl(a, w);
    let inner = [(0.0, 0.0), (ax, ay), (-ax, ay), (0.0, 0.0)];
    let r = w / (2.0 * (a / 2.0).sin());
    let rr = r + h;
    let tl = (-h * (a / 2.0).sin(), h - h * (a / 2.0).cos());
    let k = 4.0 / 3.0 * (a / 4.0).tan() * rr;
    let c1 = (tl.0 + k * (a / 2.0).cos(), tl.1 - k * (a / 2.0).sin());
    let outer = [tl, (c1.0 - w / 3.0, c1.1), (-(c1.0 - w / 3.0), c1.1), (-tl.0, tl.1)];
    (outer, inner)
}

/// The style's control-point offsets from the identity grid (undistorted),
/// indexed `[row][col]`, for a `w × h` box.
fn offsets(style: WarpStyle, b: f32, w: f32, h: f32) -> [[(f32, f32); 4]; 4] {
    let mut d = [[(0.0f32, 0.0f32); 4]; 4];
    let th = b * std::f32::consts::PI;
    let (ax, ay) = arc_ctrl(th, w);
    let lerp = |a: (f32, f32), c: (f32, f32), t: f32| (a.0 + (c.0 - a.0) * t, a.1 + (c.1 - a.1) * t);
    match style {
        WarpStyle::None => {}
        WarpStyle::Arch => {
            for row in &mut d {
                row[1] = (ax, ay);
                row[2] = (-ax, ay);
            }
        }
        WarpStyle::ArcLower => {
            for (j, row) in d.iter_mut().enumerate() {
                let f = j as f32 / 3.0;
                row[1] = (ax * f, -ay * f);
                row[2] = (-ax * f, -ay * f);
            }
        }
        WarpStyle::ArcUpper => {
            for (j, row) in d.iter_mut().enumerate() {
                let f = (3 - j) as f32 / 3.0;
                row[1] = (ax * f, ay * f);
                row[2] = (-ax * f, ay * f);
            }
        }
        WarpStyle::Bulge => {
            for (j, row) in d.iter_mut().enumerate() {
                let f = (3.0 - 2.0 * j as f32) / 3.0;
                row[1] = (ax, ay * f);
                row[2] = (-ax, ay * f);
            }
        }
        WarpStyle::Arc | WarpStyle::ShellLower | WarpStyle::ShellUpper => {
            let (outer, inner) = arc_rows(th.abs(), w, h);
            let mut arc = [[(0.0f32, 0.0f32); 4]; 4];
            for (j, row) in arc.iter_mut().enumerate() {
                for i in 0..4 {
                    row[i] = lerp(outer[i], inner[i], j as f32 / 3.0);
                }
            }
            if th < 0.0 {
                // Bend down: the vertical mirror of the bend-up mesh.
                let up = arc;
                for j in 0..4 {
                    for i in 0..4 {
                        arc[j][i] = (up[3 - j][i].0, -up[3 - j][i].1);
                    }
                }
            }
            let neg = |q: (f32, f32)| (q.0, -q.1);
            match style {
                WarpStyle::Arc => d = arc,
                WarpStyle::ShellUpper => {
                    d[0] = arc[0];
                    d[1][0] = arc[1][0];
                    d[1][3] = arc[1][3];
                }
                _ => {
                    d[3] = arc[0].map(neg);
                    d[2][0] = neg(arc[1][0]);
                    d[2][3] = neg(arc[1][3]);
                }
            }
        }
        WarpStyle::Flag => {
            for row in &mut d {
                row[1] = (0.0, -2.0 * b * h);
                row[2] = (0.0, 2.0 * b * h);
            }
        }
        WarpStyle::Wave => {
            for row in &mut d[1..3] {
                row[1] = (0.0, 4.0 / 3.0 * b * h);
                row[2] = (0.0, -4.0 / 3.0 * b * h);
            }
        }
        WarpStyle::Fish => {
            for (j, row) in d.iter_mut().enumerate() {
                let f = (3.0 - 2.0 * j as f32) / 3.0;
                row[1] = (0.0, -2.0 * b * h * f);
                row[2] = (0.0, 2.0 * b * h * f);
            }
        }
        WarpStyle::Rise => {
            for row in &mut d {
                row[0] = (0.0, 2.0 * b * h);
                row[1] = (0.0, 2.0 * b * h);
            }
        }
        WarpStyle::Fisheye => {
            for (j, sy) in [(1usize, -1.0f32), (2, 1.0)] {
                d[j][1] = (-2.0 / 3.0 * b * w, sy * 2.0 / 3.0 * b * h);
                d[j][2] = (2.0 / 3.0 * b * w, sy * 2.0 / 3.0 * b * h);
            }
        }
        WarpStyle::Inflate | WarpStyle::Squeeze => {
            let sx = if style == WarpStyle::Inflate { 1.0 } else { -1.0 };
            d[0][1] = (0.0, -b * h / 3.0);
            d[0][2] = (0.0, -b * h / 3.0);
            d[3][1] = (0.0, b * h / 3.0);
            d[3][2] = (0.0, b * h / 3.0);
            for (j, sy) in [(1usize, -1.0f32), (2, 1.0)] {
                d[j][0] = (-sx * b * w / 3.0, 0.0);
                d[j][3] = (sx * b * w / 3.0, 0.0);
                d[j][1] = (-sx * b * w / 9.0, sy * b * h / 9.0);
                d[j][2] = (sx * b * w / 9.0, sy * b * h / 9.0);
            }
        }
        WarpStyle::Twist => {
            if b >= 0.0 {
                d[1][1] = (b * w, 0.0);
                d[1][2] = (0.0, b * h);
                d[2][1] = (0.0, -b * h);
                d[2][2] = (-b * w, 0.0);
            } else {
                let a = -b;
                d[1][1] = (0.0, a * h);
                d[1][2] = (-a * w, 0.0);
                d[2][1] = (a * w, 0.0);
                d[2][2] = (0.0, -a * h);
            }
        }
    }
    d
}

/// The 4 × 4 control mesh for `warp` over a `w × h` box with its top-left
/// at the origin, `[row][col]`.
pub fn mesh(warp: &Warp, w: f32, h: f32) -> [[P; 4]; 4] {
    if warp.vertical {
        // Vertical direction: build for the transposed box, then transpose.
        let t = mesh(&Warp { vertical: false, ..*warp }, h, w);
        let mut m = [[p(0.0, 0.0); 4]; 4];
        for j in 0..4 {
            for i in 0..4 {
                m[j][i] = p(t[i][j].y, t[i][j].x);
            }
        }
        return m;
    }
    let (hd, vd) = (warp.h_distort.clamp(-1.0, 1.0), warp.v_distort.clamp(-1.0, 1.0));
    // Measured against Photoshop: horizontal distortion applies after the
    // vertical one and scales each column of the mesh uniformly about the
    // midpoint of its end control points by the column factor 1 + h(2i/3 − 1).
    // (Inflate and Squeeze fit their own column law better; see `laws`.)
    if hd != 0.0 && !matches!(warp.style, WarpStyle::Inflate | WarpStyle::Squeeze) {
        let v = mesh(&Warp { h_distort: 0.0, ..*warp }, w, h);
        let mut m = v;
        for i in 0..4 {
            let mid = v[0][i].lerp(v[3][i], 0.5);
            let si = 1.0 + hd * (2.0 * i as f32 / 3.0 - 1.0);
            for j in 0..4 {
                m[j][i] = mid + (v[j][i] - mid) * si;
            }
        }
        return m;
    }
    if vd != 0.0 && matches!(warp.style, WarpStyle::ShellLower | WarpStyle::ShellUpper) {
        // Shells: vertical distortion scales each row uniformly about the
        // midpoint of its end points by the row factor 1 + v(2j/3 − 1).
        let b = mesh(&Warp { v_distort: 0.0, ..*warp }, w, h);
        let mut m = b;
        for j in 0..4 {
            let mid = b[j][0].lerp(b[j][3], 0.5);
            let sj = 1.0 + vd * (2.0 * j as f32 / 3.0 - 1.0);
            for i in 0..4 {
                m[j][i] = mid + (b[j][i] - mid) * sj;
            }
        }
        return m;
    }
    let d = offsets(warp.style, warp.bend.clamp(-1.0, 1.0), w, h);
    let lw = laws(warp.style);
    let mut m = [[p(0.0, 0.0); 4]; 4];
    for j in 0..4 {
        for i in 0..4 {
            let (gx, gy) = (i as f32 * w / 3.0, j as f32 * h / 3.0);
            let (dx, dy) = d[j][i];
            // Horizontal distortion first (column factor on y), then vertical
            // (row factor taken at the new y, on x).
            let si_h = 1.0 + hd * (2.0 * i as f32 / 3.0 - 1.0);
            let sj_h = 1.0 + hd * (2.0 * j as f32 / 3.0 - 1.0);
            let y1 = h / 2.0 + (gy - h / 2.0) * si_h;
            let si_v = 1.0 + vd * (2.0 * i as f32 / 3.0 - 1.0);
            let sj_v = 1.0 + vd * (2.0 * y1 / h - 1.0);
            let uy = (gy + dy) / h;
            let f = |s: S, si: f32, sj: f32, a: f32| match s {
                S::One => 1.0,
                S::Col => si,
                S::Row => sj,
                S::RowAt => 1.0 + a * (2.0 * uy - 1.0),
                S::Edge => 1.0 + a * if j < 2 { -1.0 } else { 1.0 },
                S::EdgeCol => 1.0 + a * if i < 2 { -1.0 } else { 1.0 },
            };
            let x = w / 2.0 + (gx - w / 2.0) * sj_v + dx * f(lw.hx, si_h, sj_h, hd) * f(lw.vx, si_v, sj_v, vd);
            let mut y = y1 + dy * f(lw.hy, si_h, sj_h, hd) * f(lw.vy, si_v, sj_v, vd);
            // Rise also scales heights about the centre under vertical
            // distortion (measured: 2·bend·v, at the h-distorted height).
            if warp.style == WarpStyle::Rise {
                y += 2.0 * warp.bend * vd * (h / 2.0 - y1);
            }
            m[j][i] = p(x, y);
        }
    }
    // Arc under vertical distortion: each edge arc is scaled uniformly about
    // its chord's midpoint by its row factor (the arc keeps its shape on the
    // narrower / wider row); inner rows are interpolated.
    if vd != 0.0 && warp.style == WarpStyle::Arc {
        let undistorted = mesh(&Warp { v_distort: 0.0, ..*warp }, w, h);
        for j in [0usize, 3] {
            let row = undistorted[j];
            let mid = row[0].lerp(row[3], 0.5);
            let s = 1.0 + vd * (2.0 * j as f32 / 3.0 - 1.0);
            for i in 0..4 {
                m[j][i] = mid + (row[i] - mid) * s;
            }
        }
        for j in 1..3 {
            for i in 0..4 {
                m[j][i] = m[0][i].lerp(m[3][i], j as f32 / 3.0);
            }
        }
    }
    if lw.lerp_rows && (hd != 0.0 || vd != 0.0) {
        for j in 1..3 {
            for i in 0..4 {
                m[j][i] = m[0][i].lerp(m[3][i], j as f32 / 3.0);
            }
        }
    }
    m
}

#[inline]
fn bern(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    [s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t]
}

/// Evaluates the patch at box-relative parameters `(u, v)` in 0..1.
pub fn eval(m: &[[P; 4]; 4], u: f32, v: f32) -> P {
    let (bu, bv) = (bern(u), bern(v));
    let mut q = p(0.0, 0.0);
    for j in 0..4 {
        for i in 0..4 {
            let k = bu[i] * bv[j];
            q = q + m[j][i] * k;
        }
    }
    q
}

// ----- pixel warps (Edit ▸ Transform ▸ Warp) ------------------------------------------------
//
// The same patch moves pixels: each output pixel finds the patch parameters
// (u, v) whose image it is by Newton's method, started from the closest
// node of a 5 × 5 grid of patch points, and samples the source at
// `(x0 + u w, y0 + v h)`. The GPU runs the identical float sequence.

/// The 16 control points as `[x0, y0, x1, y1, ...]`, row-major.
pub fn flat(m: &[[P; 4]; 4]) -> [f32; 32] {
    let mut o = [0f32; 32];
    for j in 0..4 {
        for i in 0..4 {
            o[(j * 4 + i) * 2] = m[j][i].x;
            o[(j * 4 + i) * 2 + 1] = m[j][i].y;
        }
    }
    o
}

#[inline]
fn bern_d(t: f32) -> [f32; 4] {
    let s = 1.0 - t;
    [-3.0 * s * s, 3.0 * s * s - 6.0 * t * s, 6.0 * t * s - 3.0 * t * t, 3.0 * t * t]
}

/// Patch point and partial derivatives at `(u, v)`: `[x, y, dx/du, dy/du, dx/dv, dy/dv]`.
pub fn patch(m: &[f32; 32], u: f32, v: f32) -> [f32; 6] {
    let (bu, bv, du, dv) = (bern(u), bern(v), bern_d(u), bern_d(v));
    let mut o = [0f32; 6];
    for j in 0..4 {
        for i in 0..4 {
            let (px, py) = (m[(j * 4 + i) * 2], m[(j * 4 + i) * 2 + 1]);
            let a = bu[i] * bv[j];
            let b = du[i] * bv[j];
            let c = bu[i] * dv[j];
            o[0] += a * px;
            o[1] += a * py;
            o[2] += b * px;
            o[3] += b * py;
            o[4] += c * px;
            o[5] += c * py;
        }
    }
    o
}

/// Patch point at `(u, v)` (the first two values of [`patch`], same float steps).
pub fn patch_p(m: &[f32; 32], u: f32, v: f32) -> (f32, f32) {
    let (bu, bv) = (bern(u), bern(v));
    let (mut x, mut y) = (0f32, 0f32);
    for j in 0..4 {
        for i in 0..4 {
            let a = bu[i] * bv[j];
            x += a * m[(j * 4 + i) * 2];
            y += a * m[(j * 4 + i) * 2 + 1];
        }
    }
    (x, y)
}

/// The 5 x 5 patch points Newton starts from (u, v = 0, 0.25, .., 1).
pub fn start_nodes(m: &[f32; 32]) -> [f32; 50] {
    let mut o = [0f32; 50];
    for j in 0..5 {
        for i in 0..5 {
            let (x, y) = patch_p(m, i as f32 * 0.25, j as f32 * 0.25);
            o[(j * 5 + i) * 2] = x;
            o[(j * 5 + i) * 2 + 1] = y;
        }
    }
    o
}

/// Newton steps of [`invert_from`] (at most: it stops at a fixed point).
pub const NEWTON_STEPS: usize = 10;

/// Patch parameters `(u, v)` in [0, 1]² whose image is the box-relative
/// point `(tx, ty)`, or None (outside the warped box).
#[cfg(test)]
pub fn invert(m: &[f32; 32], tx: f32, ty: f32) -> Option<(f32, f32)> {
    invert_from(m, &start_nodes(m), tx, ty)
}

/// [`invert`] with the start nodes precomputed.
pub fn invert_from(m: &[f32; 32], nodes: &[f32; 50], tx: f32, ty: f32) -> Option<(f32, f32)> {
    // Start from the closest node (first one on ties).
    let (mut u, mut v, mut best) = (0.0f32, 0.0f32, f32::MAX);
    for j in 0..5 {
        for i in 0..5 {
            let (dx, dy) = (nodes[(j * 5 + i) * 2] - tx, nodes[(j * 5 + i) * 2 + 1] - ty);
            let d = dx * dx + dy * dy;
            if d < best {
                (u, v, best) = (i as f32 * 0.25, j as f32 * 0.25, d);
            }
        }
    }
    for _ in 0..NEWTON_STEPS {
        let p = patch(m, u, v);
        let (rx, ry) = (p[0] - tx, p[1] - ty);
        let det = p[2] * p[5] - p[3] * p[4];
        if det.abs() < 1e-6 {
            return None;
        }
        let nu = (u - (rx * p[5] - ry * p[4]) / det).clamp(-0.5, 1.5);
        let nv = (v - (p[2] * ry - p[3] * rx) / det).clamp(-0.5, 1.5);
        // A fixed point stays one: stopping changes nothing.
        let done = nu == u && nv == v;
        (u, v) = (nu, nv);
        if done {
            break;
        }
    }
    let (px, py) = patch_p(m, u, v);
    let ok = (px - tx).abs() <= 0.01 && (py - ty).abs() <= 0.01 && (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v);
    ok.then_some((u, v))
}

/// A ready-to-use warp of a box `(x0, y0, x1, y1)`: maps document points.
pub struct BoxWarp {
    m: [[P; 4]; 4],
    x0: f32,
    y0: f32,
    w: f32,
    h: f32,
}

impl BoxWarp {
    pub fn new(warp: &Warp, b: (f32, f32, f32, f32)) -> BoxWarp {
        let (w, h) = ((b.2 - b.0).max(1.0), (b.3 - b.1).max(1.0));
        BoxWarp { m: mesh(warp, w, h), x0: b.0, y0: b.1, w, h }
    }

    pub fn map(&self, q: P) -> P {
        let r = eval(&self.m, (q.x - self.x0) / self.w, (q.y - self.y0) / self.h);
        p(r.x + self.x0, r.y + self.y0)
    }

    /// The pixel warp of this box as a filter (Edit ▸ Transform ▸ Warp).
    pub fn pixel_warp(&self) -> crate::engine::filters::distort::Warp {
        let m = flat(&self.m);
        crate::engine::filters::distort::Warp::Mesh { m, nodes: start_nodes(&m), x0: self.x0, y0: self.y0, w: self.w, h: self.h }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Control-point offsets Photoshop 2026 produced (fitted from its warped
    /// outlines, 0.01 px residual) for a 724.8 × 436.51 box at bend 50 %,
    /// as (style, row, col, dx, dy).
    const PS_50: &[(WarpStyle, usize, usize, f32, f32)] = &[
        (WarpStyle::Arch, 0, 1, -41.5, -200.2),
        (WarpStyle::ArcLower, 3, 1, -41.4, 200.1),
        (WarpStyle::ArcUpper, 0, 2, 41.4, -200.1),
        (WarpStyle::Bulge, 1, 1, -41.5, -66.7),
        (WarpStyle::Arc, 0, 0, -308.7, 127.9),
        (WarpStyle::Arc, 0, 1, -179.7, -242.8),
        (WarpStyle::Arc, 1, 2, 133.6, -228.5),
        (WarpStyle::ShellLower, 3, 0, -308.7, -127.9),
        (WarpStyle::ShellLower, 2, 0, -205.8, -85.2),
        (WarpStyle::ShellUpper, 0, 1, -179.6, -242.7),
        (WarpStyle::Flag, 2, 1, 0.0, -436.7),
        (WarpStyle::Wave, 1, 2, 0.0, -291.1),
        (WarpStyle::Fish, 1, 1, 0.0, -145.6),
        (WarpStyle::Rise, 3, 0, 0.0, 436.7),
        (WarpStyle::Fisheye, 1, 2, 241.6, -145.4),
        (WarpStyle::Inflate, 1, 0, -120.8, 0.0),
        (WarpStyle::Inflate, 1, 1, -40.3, -24.3),
        (WarpStyle::Squeeze, 2, 3, -120.7, 0.0),
        (WarpStyle::Twist, 1, 1, 362.4, 0.0),
        (WarpStyle::Twist, 2, 1, 0.0, -218.3),
    ];

    #[test]
    fn meshes_match_photoshop_control_points() {
        let (w, h) = (724.8, 436.51);
        for &(style, j, i, dx, dy) in PS_50 {
            let m = mesh(&Warp { style, bend: 0.5, ..Default::default() }, w, h);
            let (gx, gy) = (i as f32 * w / 3.0, j as f32 * h / 3.0);
            let (ex, ey) = (m[j][i].x - gx - dx, m[j][i].y - gy - dy);
            assert!(ex.abs() < 1.0 && ey.abs() < 1.0, "{style:?} [{j}][{i}]: off by ({ex:.2}, {ey:.2})");
        }
        // Distortion: Arch at bend 50 %, horizontal / vertical distortion 40 %.
        let m = mesh(&Warp { style: WarpStyle::Arch, bend: 0.5, h_distort: 0.4, ..Default::default() }, w, h);
        assert!((m[0][0].y - 87.3).abs() < 1.0, "{:?}", m[0][0]);
        let m = mesh(&Warp { style: WarpStyle::Arch, bend: 0.5, v_distort: 0.4, ..Default::default() }, w, h);
        assert!((m[0][0].x - 144.9).abs() < 1.0 && (m[0][1].y - -120.1).abs() < 1.5, "{:?} {:?}", m[0][0], m[0][1]);
    }

    /// Photoshop 2026 meshes with both distortions at 40 % (bend 50 %),
    /// fitted from its outlines (0.01 px residual), in box-relative units:
    /// (style, row, col, x, y).
    const PS_HV: &[(WarpStyle, usize, usize, f32, f32)] = &[
        (WarpStyle::Arc, 0, 0, -0.085, 0.434),
        (WarpStyle::Arc, 0, 2, 0.744, -0.255),
        (WarpStyle::Arc, 3, 3, 1.229, 1.141),
        (WarpStyle::ShellLower, 0, 0, 0.001, 0.141),
        (WarpStyle::ShellLower, 3, 1, -0.050, 1.769),
        (WarpStyle::ShellLower, 3, 3, 1.992, 0.849),
        (WarpStyle::Fish, 0, 1, 0.391, -0.400),
        (WarpStyle::Fish, 3, 2, 0.742, -0.466),
        (WarpStyle::Rise, 0, 0, 0.120, 1.080),
        (WarpStyle::Rise, 3, 1, 0.276, 2.107),
        (WarpStyle::Bulge, 0, 2, 0.622, -0.403),
        (WarpStyle::Bulge, 3, 1, 0.199, 1.514),
        (WarpStyle::Inflate, 1, 0, -0.060, 0.400),
    ];

    /// Inflate with one distortion at 40 % (bend 50 %): (h, v, row, col, x, y).
    const PS_INFLATE: &[(f32, f32, usize, usize, f32, f32)] = &[
        (0.0, 0.4, 1, 1, 0.300, 0.300),
        (0.0, 0.4, 2, 0, -0.233, 0.667),
        (0.0, 0.4, 3, 1, 0.267, 1.233),
        (0.4, 0.0, 1, 1, 0.300, 0.300),
        (0.4, 0.0, 1, 2, 0.744, 0.255),
        (0.4, 0.0, 2, 3, 1.233, 0.733),
    ];

    #[test]
    fn combined_distortions_match_photoshop() {
        let (w, h) = (724.8, 436.51);
        for &(style, j, i, x, y) in PS_HV {
            let m = mesh(&Warp { style, bend: 0.5, h_distort: 0.4, v_distort: 0.4, vertical: false }, w, h);
            let (mx, my) = (m[j][i].x / w, m[j][i].y / h);
            assert!((mx - x).abs() < 0.006 && (my - y).abs() < 0.006, "{style:?} [{j}][{i}]: ({mx:.3}, {my:.3}) vs PS ({x}, {y})");
        }
        for &(hd, vd, j, i, x, y) in PS_INFLATE {
            let m = mesh(&Warp { style: WarpStyle::Inflate, bend: 0.5, h_distort: hd, v_distort: vd, vertical: false }, w, h);
            let (mx, my) = (m[j][i].x / w, m[j][i].y / h);
            assert!((mx - x).abs() < 0.004 && (my - y).abs() < 0.004, "Inflate h {hd} v {vd} [{j}][{i}]: ({mx:.3}, {my:.3}) vs PS ({x}, {y})");
        }
    }

    #[test]
    fn pixel_warp_inverts_the_patch() {
        for style in [WarpStyle::Arc, WarpStyle::Flag, WarpStyle::Bulge, WarpStyle::Fisheye] {
            let w = Warp { style, bend: 0.5, h_distort: 0.1, v_distort: 0.0, vertical: false };
            let m = flat(&mesh(&w, 800.0, 500.0));
            for (u, v) in [(0.1f32, 0.2f32), (0.5, 0.5), (0.93, 0.71), (0.0, 1.0), (0.37, 0.02)] {
                let q = patch(&m, u, v);
                let (iu, iv) = invert(&m, q[0], q[1]).unwrap_or_else(|| panic!("{style:?} ({u}, {v}) not found"));
                assert!((iu - u).abs() < 1e-3 && (iv - v).abs() < 1e-3, "{style:?}: ({u}, {v}) -> ({iu}, {iv})");
            }
            // Far outside the warped box: transparent.
            assert!(invert(&m, -5000.0, -5000.0).is_none());
        }
    }

    #[test]
    fn identity_and_corners() {
        let m = mesh(&Warp { style: WarpStyle::None, ..Default::default() }, 100.0, 50.0);
        let q = eval(&m, 0.3, 0.7);
        assert!((q.x - 30.0).abs() < 1e-3 && (q.y - 35.0).abs() < 1e-3);
        // Arch keeps the corners; Arc at full bend makes a half annulus.
        let bw = BoxWarp::new(&Warp { style: WarpStyle::Arch, bend: 0.7, ..Default::default() }, (10.0, 20.0, 210.0, 120.0));
        let c = bw.map(p(210.0, 120.0));
        assert!((c.x - 210.0).abs() < 1e-3 && (c.y - 120.0).abs() < 1e-3);
        let m = mesh(&Warp { style: WarpStyle::Arc, bend: 1.0, ..Default::default() }, 400.0, 100.0);
        let (tl, bl) = (eval(&m, 0.0, 0.0), eval(&m, 0.0, 1.0));
        assert!((tl.y - bl.y).abs() < 0.5, "half annulus: left edge horizontal ({tl:?}, {bl:?})");
    }
}

