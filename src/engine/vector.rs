//! Vector core: Bézier paths, flattening, exact-area coverage rasterization,
//! stroking and hit testing.
//!
//! **Rasterizer.** Edges deposit signed area into an accumulation buffer
//! (the font-rs / libart scheme): a prefix sum along each scanline turns the
//! deposits into exact winding coverage, so every pixel's value is the true
//! fraction of its square covered by the polygon (nonzero or even-odd).
//! The canvas is processed per 256² tile, with edges split at tile borders.
//! An edge to the left of a tile contributes a per-scanline *carry* of
//! winding instead of being drawn, so:
//!
//! - tiles no edge crosses are classified `Full` / `Empty` from the carry
//!   alone (no pixel work, no allocation),
//! - only tiles an edge crosses run the accumulation, in parallel,
//! - a caller can rasterize just a dirty region: carries from edges left of
//!   it are still summed (cheap), so the result is identical to a full pass.
//!
//! Each pixel belongs to exactly one tile and sees exactly the same edges as
//! a whole-image pass would, so there are no seams at tile borders.

use std::sync::Arc;

use rayon::prelude::*;

use super::grid::{RectI, TS};
#[cfg(test)]
use super::grid::TS_I;
use super::selection::{MaskGrid, MaskTile};

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct P {
    pub x: f32,
    pub y: f32,
}

#[inline]
pub const fn p(x: f32, y: f32) -> P {
    P { x, y }
}

impl std::ops::Add for P {
    type Output = P;
    fn add(self, o: P) -> P {
        p(self.x + o.x, self.y + o.y)
    }
}
impl std::ops::Sub for P {
    type Output = P;
    fn sub(self, o: P) -> P {
        p(self.x - o.x, self.y - o.y)
    }
}
impl std::ops::Mul<f32> for P {
    type Output = P;
    fn mul(self, k: f32) -> P {
        p(self.x * k, self.y * k)
    }
}

impl P {
    pub fn len(self) -> f32 {
        self.x.hypot(self.y)
    }
    pub fn dot(self, o: P) -> f32 {
        self.x * o.x + self.y * o.y
    }
    pub fn cross(self, o: P) -> f32 {
        self.x * o.y - self.y * o.x
    }
    pub fn norm(self) -> P {
        let l = self.len();
        if l > 1e-12 { self * (1.0 / l) } else { p(0.0, 0.0) }
    }
    /// Left-hand perpendicular (CCW in y-down image space is clockwise on screen).
    pub fn perp(self) -> P {
        p(-self.y, self.x)
    }
    pub fn lerp(self, o: P, t: f32) -> P {
        self + (o - self) * t
    }
    pub fn dist(self, o: P) -> f32 {
        (self - o).len()
    }
    /// Applies a 3x3 homography (row-major).
    pub fn xform(self, m: &[f32; 9]) -> P {
        let w = m[6] * self.x + m[7] * self.y + m[8];
        let w = if w.abs() < 1e-12 { 1e-12 } else { w };
        p((m[0] * self.x + m[1] * self.y + m[2]) / w, (m[3] * self.x + m[4] * self.y + m[5]) / w)
    }
}

// ----- path model ------------------------------------------------------------------

/// An anchor with absolute handle positions; a handle equal to `p` is
/// retracted (corner).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub p: P,
    pub h_in: P,
    pub h_out: P,
    /// Smooth anchors keep their handles collinear when edited.
    pub smooth: bool,
}

impl Anchor {
    pub fn corner(pt: P) -> Self {
        Anchor { p: pt, h_in: pt, h_out: pt, smooth: false }
    }
    pub fn smooth(pt: P, h_in: P, h_out: P) -> Self {
        Anchor { p: pt, h_in, h_out, smooth: true }
    }
    pub fn has_in(&self) -> bool {
        self.h_in != self.p
    }
    pub fn has_out(&self) -> bool {
        self.h_out != self.p
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct SubPath {
    pub anchors: Vec<Anchor>,
    pub closed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FillRule {
    #[default]
    NonZero,
    EvenOdd,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Path {
    pub subs: Vec<SubPath>,
}

/// One cubic segment (p0, c0, c1, p1).
pub type Cubic = [P; 4];

impl SubPath {
    pub fn segments(&self) -> impl Iterator<Item = Cubic> + '_ {
        let n = self.anchors.len();
        let count = if self.closed && n > 1 { n } else { n.saturating_sub(1) };
        (0..count).map(move |i| {
            let a = &self.anchors[i];
            let b = &self.anchors[(i + 1) % n];
            [a.p, a.h_out, b.h_in, b.p]
        })
    }
}

#[inline]
pub fn cubic_at(c: &Cubic, t: f32) -> P {
    let mt = 1.0 - t;
    let a = mt * mt * mt;
    let b = 3.0 * mt * mt * t;
    let cc = 3.0 * mt * t * t;
    let d = t * t * t;
    p(a * c[0].x + b * c[1].x + cc * c[2].x + d * c[3].x, a * c[0].y + b * c[1].y + cc * c[2].y + d * c[3].y)
}

/// Splits a cubic at `t` (de Casteljau).
pub fn split_cubic(c: &Cubic, t: f32) -> (Cubic, Cubic) {
    let ab = c[0].lerp(c[1], t);
    let bc = c[1].lerp(c[2], t);
    let cd = c[2].lerp(c[3], t);
    let abc = ab.lerp(bc, t);
    let bcd = bc.lerp(cd, t);
    let m = abc.lerp(bcd, t);
    ([c[0], ab, abc, m], [m, bcd, cd, c[3]])
}

/// Number of line segments that keep a cubic within `tol` (Wang's formula).
pub fn cubic_steps(c: &Cubic, tol: f32) -> usize {
    let d1 = (c[0] - c[1] * 2.0 + c[2]).len();
    let d2 = (c[1] - c[2] * 2.0 + c[3]).len();
    let m = d1.max(d2);
    if m < 1e-6 {
        return 1;
    }
    ((0.75 * m / tol).sqrt().ceil() as usize).clamp(1, 4096)
}

/// A flattened subpath.
#[derive(Clone, Debug, Default)]
pub struct Poly {
    pub pts: Vec<P>,
    pub closed: bool,
}

impl Path {
    pub fn new() -> Self {
        Path { subs: vec![] }
    }

    /// Polygonal approximation within `tol` pixels.
    pub fn flatten(&self, tol: f32) -> Vec<Poly> {
        self.subs
            .iter()
            .filter(|s| !s.anchors.is_empty())
            .map(|s| {
                let mut pts = vec![s.anchors[0].p];
                for c in s.segments() {
                    let line = c[1] == c[0] && c[2] == c[3];
                    if line {
                        pts.push(c[3]);
                        continue;
                    }
                    let n = cubic_steps(&c, tol);
                    for k in 1..=n {
                        pts.push(cubic_at(&c, k as f32 / n as f32));
                    }
                }
                if s.closed && pts.len() > 1 && pts[0] == *pts.last().unwrap() {
                    pts.pop();
                }
                Poly { pts, closed: s.closed }
            })
            .collect()
    }

    pub fn bounds(&self) -> Option<(P, P)> {
        let mut lo = p(f32::MAX, f32::MAX);
        let mut hi = p(f32::MIN, f32::MIN);
        let mut any = false;
        for s in &self.subs {
            for a in &s.anchors {
                for q in [a.p, a.h_in, a.h_out] {
                    lo = p(lo.x.min(q.x), lo.y.min(q.y));
                    hi = p(hi.x.max(q.x), hi.y.max(q.y));
                    any = true;
                }
            }
        }
        any.then_some((lo, hi))
    }

    pub fn transformed(&self, m: &[f32; 9]) -> Path {
        Path {
            subs: self
                .subs
                .iter()
                .map(|s| SubPath {
                    closed: s.closed,
                    anchors: s
                        .anchors
                        .iter()
                        .map(|a| Anchor { p: a.p.xform(m), h_in: a.h_in.xform(m), h_out: a.h_out.xform(m), smooth: a.smooth })
                        .collect(),
                })
                .collect(),
        }
    }

    /// Total arc length (flattened).
    pub fn length(&self) -> f32 {
        self.flatten(0.1).iter().map(poly_length).sum()
    }

    // ----- primitive builders ------------------------------------------------------

    pub fn polygon(pts: &[P]) -> Path {
        Path { subs: vec![SubPath { anchors: pts.iter().map(|&q| Anchor::corner(q)).collect(), closed: true }] }
    }

    /// Rectangle with per-corner radii (TL, TR, BR, BL), clamped to fit.
    pub fn rounded_rect(x0: f32, y0: f32, x1: f32, y1: f32, radii: [f32; 4]) -> Path {
        let (x0, x1) = (x0.min(x1), x0.max(x1));
        let (y0, y1) = (y0.min(y1), y0.max(y1));
        let lim = ((x1 - x0) * 0.5).min((y1 - y0) * 0.5).max(0.0);
        let r = radii.map(|v| v.clamp(0.0, lim));
        if r.iter().all(|&v| v <= 0.0) {
            return Path::polygon(&[p(x0, y0), p(x1, y0), p(x1, y1), p(x0, y1)]);
        }
        const K: f32 = 0.552_284_8;
        let mut a = Vec::new();
        // Each rounded corner is two anchors joined by a quarter-ellipse cubic.
        let corner = |a: &mut Vec<Anchor>, start: P, end: P, corner: P, rad: f32| {
            if rad <= 0.0 {
                a.push(Anchor::corner(corner));
                return;
            }
            let hs = start + (corner - start) * K;
            let he = end + (corner - end) * K;
            a.push(Anchor { p: start, h_in: start, h_out: hs, smooth: false });
            a.push(Anchor { p: end, h_in: he, h_out: end, smooth: false });
        };
        corner(&mut a, p(x0, y0 + r[0]), p(x0 + r[0], y0), p(x0, y0), r[0]);
        corner(&mut a, p(x1 - r[1], y0), p(x1, y0 + r[1]), p(x1, y0), r[1]);
        corner(&mut a, p(x1, y1 - r[2]), p(x1 - r[2], y1), p(x1, y1), r[2]);
        corner(&mut a, p(x0 + r[3], y1), p(x0, y1 - r[3]), p(x0, y1), r[3]);
        Path { subs: vec![SubPath { anchors: a, closed: true }] }
    }

    pub fn ellipse(cx: f32, cy: f32, rx: f32, ry: f32) -> Path {
        const K: f32 = 0.552_284_8;
        let (kx, ky) = (rx * K, ry * K);
        let a = vec![
            Anchor::smooth(p(cx + rx, cy), p(cx + rx, cy - ky), p(cx + rx, cy + ky)),
            Anchor::smooth(p(cx, cy + ry), p(cx + kx, cy + ry), p(cx - kx, cy + ry)),
            Anchor::smooth(p(cx - rx, cy), p(cx - rx, cy + ky), p(cx - rx, cy - ky)),
            Anchor::smooth(p(cx, cy - ry), p(cx - kx, cy - ry), p(cx + kx, cy - ry)),
        ];
        Path { subs: vec![SubPath { anchors: a, closed: true }] }
    }

    /// Regular polygon or star inscribed in the ellipse (cx, cy, rx, ry).
    /// `indent` in 0..1 turns it into a star; `rotation` in radians.
    pub fn star(cx: f32, cy: f32, rx: f32, ry: f32, sides: u32, indent: f32, rotation: f32) -> Path {
        let n = sides.max(3) as usize;
        let mut pts = Vec::with_capacity(n * 2);
        let star = indent > 0.0;
        let steps = if star { 2 * n } else { n };
        for k in 0..steps {
            let t = rotation - std::f32::consts::FRAC_PI_2 + k as f32 * std::f32::consts::TAU / steps as f32;
            let s = if star && k % 2 == 1 { 1.0 - indent.clamp(0.0, 0.99) } else { 1.0 };
            pts.push(p(cx + rx * s * t.cos(), cy + ry * s * t.sin()));
        }
        Path::polygon(&pts)
    }

    pub fn triangle(x0: f32, y0: f32, x1: f32, y1: f32) -> Path {
        Path::polygon(&[p((x0 + x1) * 0.5, y0.min(y1)), p(x0.max(x1), y0.max(y1)), p(x0.min(x1), y0.max(y1))])
    }

    /// A thick line with optional arrowheads, as a closed outline.
    pub fn line_shape(a: P, b: P, weight: f32, arrow_start: Option<(f32, f32)>, arrow_end: Option<(f32, f32)>) -> Path {
        let d = (b - a).norm();
        if d.len() == 0.0 {
            return Path::new();
        }
        let n = d.perp() * (weight * 0.5);
        let mut subs = vec![];
        // Arrowhead (width, length) are multiples of the line weight, as in Photoshop.
        let head = |tip: P, dir: P, (wm, lm): (f32, f32)| -> (SubPath, P) {
            let len = weight * lm;
            let base = tip - dir * len;
            let half = dir.perp() * (weight * wm * 0.5);
            (SubPath { anchors: vec![Anchor::corner(tip), Anchor::corner(base + half), Anchor::corner(base - half)], closed: true }, base)
        };
        let (mut s, mut e) = (a, b);
        if let Some(sz) = arrow_start {
            let (h, base) = head(a, d * -1.0, sz);
            subs.push(h);
            s = base;
        }
        if let Some(sz) = arrow_end {
            let (h, base) = head(b, d, sz);
            subs.push(h);
            e = base;
        }
        subs.push(SubPath { anchors: vec![Anchor::corner(s + n), Anchor::corner(e + n), Anchor::corner(e - n), Anchor::corner(s - n)], closed: true });
        Path { subs }
    }
}

/// A smooth closed 500-anchor test path (Catmull-Rom tangents through
/// hash-jittered points on a wavy circle); `tools/photoshop_gaps.jsx` builds
/// the identical path in Photoshop for the path-operation comparison.
pub fn noisy_loop(cx: f32, cy: f32, r: f32, seed: u32) -> Path {
    let n = 500;
    let pts: Vec<P> = (0..n)
        .map(|i| {
            let t = std::f32::consts::TAU * i as f32 / n as f32;
            let h = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) as f32 / (1u32 << 24) as f32;
            let rr = r + 60.0 * (h - 0.5) + 40.0 * (7.0 * t).sin();
            p(cx + rr * t.cos(), cy + rr * t.sin())
        })
        .collect();
    let anchors = (0..n)
        .map(|i| {
            let d = (pts[(i + 1) % n] - pts[(i + n - 1) % n]) * (1.0 / 6.0);
            Anchor { p: pts[i], h_in: pts[i] - d, h_out: pts[i] + d, smooth: true }
        })
        .collect();
    Path { subs: vec![SubPath { anchors, closed: true }] }
}

pub fn poly_length(pl: &Poly) -> f32 {
    let mut l: f32 = pl.pts.windows(2).map(|w| w[0].dist(w[1])).sum();
    if pl.closed && pl.pts.len() > 2 {
        l += pl.pts[0].dist(*pl.pts.last().unwrap());
    }
    l
}

/// Points spaced `step` apart along the polyline (arc-length resampling),
/// e.g. for Stroke Path driving the brush engine.
#[cfg(test)]
pub fn resample(pl: &Poly, step: f32) -> Vec<P> {
    let mut out = Vec::new();
    if pl.pts.is_empty() {
        return out;
    }
    let mut pts = pl.pts.clone();
    if pl.closed {
        pts.push(pl.pts[0]);
    }
    out.push(pts[0]);
    let mut carry = 0.0f32;
    for w in pts.windows(2) {
        let seg = w[1] - w[0];
        let len = seg.len();
        if len <= 0.0 {
            continue;
        }
        let mut t = step - carry;
        while t <= len {
            out.push(w[0] + seg * (t / len));
            t += step;
        }
        carry = len - (t - step);
    }
    out
}

// ----- curve fitting (Schneider 1990) -----------------------------------------------------

/// Fits a sequence of points with cubic Béziers whose distance to every
/// point is at most `error` (Freeform Pen, Content-Aware Tracing).
pub fn fit_curve(pts: &[P], error: f32) -> Vec<Cubic> {
    let mut v: Vec<P> = Vec::with_capacity(pts.len());
    for &q in pts {
        if v.last().is_none_or(|l: &P| l.dist(q) > 1e-3) {
            v.push(q);
        }
    }
    if v.len() < 2 {
        return vec![];
    }
    if v.len() == 2 {
        let (a, b) = (v[0], v[1]);
        return vec![[a, a.lerp(b, 1.0 / 3.0), a.lerp(b, 2.0 / 3.0), b]];
    }
    let n = v.len();
    let t1 = (v[1] - v[0]).norm();
    let t2 = (v[n - 2] - v[n - 1]).norm();
    let mut out = Vec::new();
    fit_cubic(&v, 0, n - 1, t1, t2, error * error, &mut out, 0);
    out
}

fn chord_params(v: &[P], a: usize, b: usize) -> Vec<f32> {
    let mut u = vec![0f32; b - a + 1];
    for i in a + 1..=b {
        u[i - a] = u[i - a - 1] + v[i].dist(v[i - 1]);
    }
    let total = u[b - a].max(1e-9);
    for x in &mut u {
        *x /= total;
    }
    u
}

fn generate_bezier(v: &[P], a: usize, b: usize, u: &[f32], t1: P, t2: P) -> Cubic {
    let (p0, p3) = (v[a], v[b]);
    let mut c = [[0f32; 2]; 2];
    let mut x = [0f32; 2];
    for (k, &t) in u.iter().enumerate() {
        let mt = 1.0 - t;
        let b1 = 3.0 * t * mt * mt;
        let b2 = 3.0 * t * t * mt;
        let a1 = t1 * b1;
        let a2 = t2 * b2;
        c[0][0] += a1.dot(a1);
        c[0][1] += a1.dot(a2);
        c[1][1] += a2.dot(a2);
        let b0 = mt * mt * mt;
        let b3 = t * t * t;
        let tmp = v[a + k] - (p0 * (b0 + b1) + p3 * (b2 + b3));
        x[0] += a1.dot(tmp);
        x[1] += a2.dot(tmp);
    }
    c[1][0] = c[0][1];
    let det = c[0][0] * c[1][1] - c[1][0] * c[0][1];
    let (mut al, mut ar) = if det.abs() > 1e-12 {
        ((x[0] * c[1][1] - c[0][1] * x[1]) / det, (c[0][0] * x[1] - c[1][0] * x[0]) / det)
    } else {
        (0.0, 0.0)
    };
    let seg = p0.dist(p3);
    let eps = 1e-6 * seg;
    if al < eps || ar < eps {
        al = seg / 3.0;
        ar = seg / 3.0;
    }
    [p0, p0 + t1 * al, p3 + t2 * ar, p3]
}

fn max_error(v: &[P], a: usize, b: usize, bez: &Cubic, u: &[f32]) -> (f32, usize) {
    let mut worst = (0f32, (a + b) / 2);
    for i in a + 1..b {
        let d = (cubic_at(bez, u[i - a]) - v[i]).dot(cubic_at(bez, u[i - a]) - v[i]);
        if d >= worst.0 {
            worst = (d, i);
        }
    }
    worst
}

fn reparameterize(v: &[P], a: usize, bez: &Cubic, u: &mut [f32]) {
    for (k, t) in u.iter_mut().enumerate() {
        // One Newton–Raphson step on (B(t) − P)·B'(t) = 0.
        let q = v[a + k];
        let tt = *t;
        let mt = 1.0 - tt;
        let p = cubic_at(bez, tt);
        let d1 = (bez[1] - bez[0]) * (3.0 * mt * mt) + (bez[2] - bez[1]) * (6.0 * mt * tt) + (bez[3] - bez[2]) * (3.0 * tt * tt);
        let d2 = (bez[2] - bez[1] * 2.0 + bez[0]) * (6.0 * mt) + (bez[3] - bez[2] * 2.0 + bez[1]) * (6.0 * tt);
        let num = (p - q).dot(d1);
        let den = d1.dot(d1) + (p - q).dot(d2);
        if den.abs() > 1e-12 {
            *t = (tt - num / den).clamp(0.0, 1.0);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_cubic(v: &[P], a: usize, b: usize, t1: P, t2: P, err2: f32, out: &mut Vec<Cubic>, depth: u32) {
    if b - a == 1 || depth > 30 {
        let d = v[a].dist(v[b]) / 3.0;
        out.push([v[a], v[a] + t1 * d, v[b] + t2 * d, v[b]]);
        return;
    }
    let mut u = chord_params(v, a, b);
    let mut bez = generate_bezier(v, a, b, &u, t1, t2);
    let (mut e, mut split) = max_error(v, a, b, &bez, &u);
    if e < err2 {
        out.push(bez);
        return;
    }
    // Reparameterize (Newton) unless the fit is hopeless, before splitting.
    if e < (err2 * 400.0).max(64.0) {
        for _ in 0..12 {
            reparameterize(v, a, &bez, &mut u);
            bez = generate_bezier(v, a, b, &u, t1, t2);
            let r = max_error(v, a, b, &bez, &u);
            e = r.0;
            split = r.1;
            if e < err2 {
                out.push(bez);
                return;
            }
        }
    }
    let split = split.clamp(a + 1, b - 1);
    let tc = (v[split - 1] - v[split + 1]).norm();
    fit_cubic(v, a, split, t1, tc, err2, out, depth + 1);
    fit_cubic(v, split, b, tc * -1.0, t2, err2, out, depth + 1);
}

/// Builds a subpath from fitted cubics (smooth anchors where tangents agree).
pub fn subpath_from_cubics(cubics: &[Cubic], closed: bool) -> SubPath {
    let mut anchors: Vec<Anchor> = Vec::with_capacity(cubics.len() + 1);
    for (i, c) in cubics.iter().enumerate() {
        if i == 0 {
            anchors.push(Anchor { p: c[0], h_in: c[0], h_out: c[1], smooth: false });
        } else {
            anchors.last_mut().unwrap().h_out = c[1];
        }
        anchors.push(Anchor { p: c[3], h_in: c[2], h_out: c[3], smooth: false });
    }
    for a in &mut anchors {
        let (din, dout) = (a.p - a.h_in, a.h_out - a.p);
        a.smooth = din.len() > 1e-3 && dout.len() > 1e-3 && din.norm().dot(dout.norm()) > 0.99;
    }
    if closed && anchors.len() > 2 && anchors[0].p.dist(anchors.last().unwrap().p) < 0.5 {
        let last = anchors.pop().unwrap();
        anchors[0].h_in = last.h_in;
    }
    SubPath { anchors, closed }
}

// ----- rasterizer ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub a: P,
    pub b: P,
}

/// Collects polygon edges (closing every polygon).
pub fn edges(polys: &[Poly]) -> Vec<Line> {
    let mut v = Vec::new();
    for pl in polys {
        let n = pl.pts.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            let a = pl.pts[i];
            let b = pl.pts[(i + 1) % n];
            if a.y != b.y {
                v.push(Line { a, b });
            }
        }
    }
    v
}

/// Accumulates one line (tile-relative, x within [0, TS]) into `acc`
/// (rows of `AW` cells). Exact signed area per cell.
const AW: usize = TS + 3;

#[inline]
fn draw_line(acc: &mut [f32], a: P, b: P) {
    if a.y == b.y {
        return;
    }
    let (dir, p0, p1) = if a.y < b.y { (1.0f32, a, b) } else { (-1.0f32, b, a) };
    let dxdy = (p1.x - p0.x) / (p1.y - p0.y);
    let mut x = p0.x;
    let y_start = p0.y.max(0.0);
    if p0.y < 0.0 {
        x -= p0.y * dxdy;
    }
    let y_end = p1.y.min(TS as f32);
    if y_start >= y_end {
        return;
    }
    let mut y = y_start.floor() as usize;
    let yl = y_end.ceil() as usize;
    while y < yl {
        let row = &mut acc[y * AW..(y + 1) * AW];
        let dy = ((y + 1) as f32).min(p1.y) - (y as f32).max(p0.y);
        let xnext = x + dxdy * dy;
        let d = dy * dir;
        let (x0, x1) = if x < xnext { (x, xnext) } else { (xnext, x) };
        let x0floor = x0.floor();
        let x0i = x0floor as i32;
        let x1ceil = x1.ceil();
        let x1i = x1ceil as i32;
        let at = |i: i32| (i.clamp(0, AW as i32 - 1)) as usize;
        if x1i <= x0i + 1 {
            let xmf = 0.5 * (x + xnext) - x0floor;
            row[at(x0i)] += d - d * xmf;
            row[at(x0i + 1)] += d * xmf;
        } else {
            let s = 1.0 / (x1 - x0);
            let x0f = x0 - x0floor;
            let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
            let x1f = x1 - x1ceil + 1.0;
            let am = 0.5 * s * x1f * x1f;
            row[at(x0i)] += d * a0;
            if x1i == x0i + 2 {
                row[at(x0i + 1)] += d * (1.0 - a0 - am);
            } else {
                let a1 = s * (1.5 - x0f);
                row[at(x0i + 1)] += d * (a1 - a0);
                for xi in x0i + 2..x1i - 1 {
                    row[at(xi)] += d * s;
                }
                let a2 = a1 + (x1i - x0i - 3) as f32 * s;
                row[at(x1i - 1)] += d * (1.0 - a2 - am);
            }
            row[at(x1i)] += d * am;
        }
        x = xnext;
        y += 1;
    }
}

#[inline]
fn cov_of(w: f32, rule: FillRule) -> f32 {
    match rule {
        FillRule::NonZero => w.abs().min(1.0),
        FillRule::EvenOdd => {
            let a = w.abs() % 2.0;
            if a > 1.0 { 2.0 - a } else { a }
        }
    }
}

/// A line piece inside one tile row, x already in tile-local coordinates of
/// its column (clamped into [0, TS]); y relative to the tile row.
#[derive(Clone, Copy)]
struct Piece {
    col: i32,
    a: P,
    b: P,
}

/// Splits `l` (image coordinates) into tile-row/tile-column pieces.
/// Pieces left of column 0 go to column -1 (pure carry) and pieces right of
/// the last column are dropped.
fn bucket(lines: &[Line], cols: i32, rows: i32, out: &mut [Vec<Piece>]) {
    let tsf = TS as f32;
    for l in lines {
        let (top, bot) = if l.a.y < l.b.y { (l.a, l.b) } else { (l.b, l.a) };
        let ty0 = ((top.y / tsf).floor() as i32).max(0);
        let ty1 = (((bot.y / tsf).ceil() as i32).min(rows)).max(ty0);
        for ty in ty0..ty1 {
            let (y0, y1) = (ty as f32 * tsf, (ty + 1) as f32 * tsf);
            // Clip the segment to this row (keep original direction).
            let clip = |q: P, r: P| -> Option<(P, P)> {
                let (qy, ry) = (q.y.clamp(y0, y1), r.y.clamp(y0, y1));
                if qy == ry {
                    return None;
                }
                let fx = |y: f32| if r.y == q.y { q.x } else { q.x + (r.x - q.x) * (y - q.y) / (r.y - q.y) };
                Some((p(fx(qy), qy - y0), p(fx(ry), ry - y0)))
            };
            let Some((a, b)) = clip(l.a, l.b) else { continue };
            // Split at tile column boundaries.
            let (xa, xb) = (a.x.min(b.x), a.x.max(b.x));
            let c0 = (xa / tsf).floor() as i32;
            let c1 = (xb / tsf).floor() as i32;
            let mut cuts = vec![0.0f32, 1.0];
            for c in (c0 + 1)..=c1 {
                let bx = c as f32 * tsf;
                if b.x != a.x {
                    let t = (bx - a.x) / (b.x - a.x);
                    if t > 0.0 && t < 1.0 {
                        cuts.push(t);
                    }
                }
            }
            cuts.sort_by(|u, v| u.total_cmp(v));
            for w in cuts.windows(2) {
                let (u, v) = (w[0], w[1]);
                if v <= u {
                    continue;
                }
                let pa = a.lerp(b, u);
                let pb = a.lerp(b, v);
                let mx = (pa.x + pb.x) * 0.5;
                let col = (mx / tsf).floor() as i32;
                if col >= cols {
                    continue;
                }
                let col = col.max(-1);
                let ox = col as f32 * tsf;
                let loc = |q: P| p((q.x - ox).clamp(0.0, tsf), q.y);
                out[ty as usize].push(Piece { col, a: loc(pa), b: loc(pb) });
            }
        }
    }
}

/// Per-scanline winding deposited by a piece (its signed dy per row).
fn add_carry(carry: &mut [f32; TS], pc: &Piece) {
    let (dir, p0, p1) = if pc.a.y < pc.b.y { (1.0f32, pc.a, pc.b) } else { (-1.0f32, pc.b, pc.a) };
    let mut y = p0.y.floor().max(0.0) as usize;
    while (y as f32) < p1.y && y < TS {
        let dy = ((y + 1) as f32).min(p1.y) - (y as f32).max(p0.y);
        carry[y] += dy * dir;
        y += 1;
    }
}

/// Rasterizes `lines` into coverage tiles over the `cols x rows` tile grid,
/// limited to the tiles of `region` (level-0 pixels). Returns
/// `(tile index, tile)` for every tile in the region that is not `Empty`.
pub fn rasterize(lines: &[Line], cols: u32, rows: u32, region: RectI, rule: FillRule) -> Vec<(usize, MaskTile)> {
    let (cols, rows) = (cols as i32, rows as i32);
    let (rtx0, rty0, rtx1, rty1) = region.tile_span(cols as u32, rows as u32);
    if rtx0 >= rtx1 || rty0 >= rty1 || lines.is_empty() {
        return vec![];
    }
    let mut buckets: Vec<Vec<Piece>> = (0..rows).map(|_| Vec::new()).collect();
    bucket(lines, cols, rows, &mut buckets);

    /// Work for one tile row: tiles crossed by edges (with their carry-in)
    /// and runs of tiles no edge crosses (uniform per scanline).
    struct RowPlan {
        ty: i32,
        jobs: Vec<(i32, Box<[f32; TS]>, Vec<Piece>)>,
        runs: Vec<(i32, i32, Box<[f32; TS]>)>,
    }
    let plans: Vec<RowPlan> = (rty0..rty1)
        .into_par_iter()
        .map(|ty| {
            let mut pieces = buckets[ty as usize].clone();
            pieces.sort_by_key(|pc| pc.col);
            let mut carry = Box::new([0f32; TS]);
            let mut plan = RowPlan { ty, jobs: vec![], runs: vec![] };
            let mut next_free = rtx0;
            let mut i = 0;
            while i < pieces.len() {
                let c = pieces[i].col;
                let s = i;
                while i < pieces.len() && pieces[i].col == c {
                    i += 1;
                }
                if c >= rtx1 {
                    break;
                }
                if c >= rtx0 {
                    if next_free < c {
                        plan.runs.push((next_free, c, carry.clone()));
                    }
                    plan.jobs.push((c, carry.clone(), pieces[s..i].to_vec()));
                    next_free = c + 1;
                }
                for pc in &pieces[s..i] {
                    add_carry(&mut carry, pc);
                }
            }
            if next_free < rtx1 {
                plan.runs.push((next_free, rtx1, carry));
            }
            plan
        })
        .collect();

    let carry_tile = |carry: &[f32; TS]| -> MaskTile {
        // Interior runs are usually empty or full. Decide from their row
        // coverage before allocating and filling a 64 KiB partial mask.
        let rows = carry.map(|c| (cov_of(c, rule) * 255.0 + 0.5) as u8);
        let first = rows[0];
        if (first == 0 || first == 255) && rows.iter().all(|&v| v == first) {
            if first == 0 { MaskTile::Empty } else { MaskTile::Full }
        } else {
            let mut px = vec![0u8; TS * TS];
            for (row, v) in px.chunks_exact_mut(TS).zip(rows) {
                row.fill(v);
            }
            MaskTile::Partial(Arc::new(crate::engine::memory::Bytes::new(px)))
        }
    };

    let mut out: Vec<(usize, MaskTile)> = Vec::new();
    let mut jobs: Vec<(i32, i32, &[f32; TS], &[Piece])> = Vec::new();
    for pl in &plans {
        for (a, b, carry) in &pl.runs {
            let t = carry_tile(carry);
            if !matches!(t, MaskTile::Empty) {
                for c in *a..*b {
                    out.push(((pl.ty * cols + c) as usize, t.clone()));
                }
            }
        }
        for (c, carry, pieces) in &pl.jobs {
            jobs.push((pl.ty, *c, carry, pieces));
        }
    }
    let drawn: Vec<(usize, MaskTile)> = jobs
        .par_iter()
        .map_init(
            || vec![0f32; AW * TS],
            |acc, &(ty, c, carry, pieces)| {
                acc.fill(0.0);
                for pc in pieces {
                    draw_line(acc, pc.a, pc.b);
                }
                let mut px = vec![0u8; TS * TS];
                for y in 0..TS {
                    let mut w = carry[y];
                    let row = &acc[y * AW..y * AW + TS];
                    let dst = &mut px[y * TS..(y + 1) * TS];
                    for (d, &a) in dst.iter_mut().zip(row) {
                        w += a;
                        *d = (cov_of(w, rule) * 255.0 + 0.5) as u8;
                    }
                }
                ((ty * cols + c) as usize, super::masks::normalize(px))
            },
        )
        .collect();
    out.extend(drawn.into_iter().filter(|(_, t)| !matches!(t, MaskTile::Empty)));
    out
}

/// Exact-area coverage of `lines` into a `w x h` buffer (small rasters such
/// as glyphs). Same signed-area accumulation as the tiled path.
pub fn raster_buffer(lines: &[Line], w: usize, h: usize, rule: FillRule) -> Vec<u8> {
    let stride = w + 3;
    let mut acc = vec![0f32; stride * h];
    let (wf, hf) = (w as f32, h as f32);
    for l in lines {
        // Clamp to the buffer: x into [0, w] (left of it carries winding), y
        // clipped by the scanline loop.
        let clip = |q: P| p(q.x.clamp(0.0, wf), q.y);
        let (a, b) = (l.a, l.b);
        // Split at x = 0 and x = w so clamped parts become vertical.
        let mut cuts = vec![0.0f32, 1.0];
        for bx in [0.0, wf] {
            if (a.x - bx) * (b.x - bx) < 0.0 {
                cuts.push((bx - a.x) / (b.x - a.x));
            }
        }
        cuts.sort_by(|u, v| u.total_cmp(v));
        for c in cuts.windows(2) {
            let (pa, pb) = (clip(a.lerp(b, c[0])), clip(a.lerp(b, c[1])));
            draw_line_stride(&mut acc, stride, hf, pa, pb);
        }
    }
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let mut wsum = 0.0;
        for x in 0..w {
            wsum += acc[y * stride + x];
            out[y * w + x] = (cov_of(wsum, rule) * 255.0 + 0.5) as u8;
        }
    }
    out
}

fn draw_line_stride(acc: &mut [f32], stride: usize, h: f32, a: P, b: P) {
    if a.y == b.y {
        return;
    }
    let (dir, p0, p1) = if a.y < b.y { (1.0f32, a, b) } else { (-1.0f32, b, a) };
    let dxdy = (p1.x - p0.x) / (p1.y - p0.y);
    let mut x = p0.x;
    let y_start = p0.y.max(0.0);
    if p0.y < 0.0 {
        x -= p0.y * dxdy;
    }
    let y_end = p1.y.min(h);
    if y_start >= y_end {
        return;
    }
    let mut y = y_start.floor() as usize;
    let yl = y_end.ceil() as usize;
    while y < yl {
        let row = &mut acc[y * stride..(y + 1) * stride];
        let dy = ((y + 1) as f32).min(p1.y) - (y as f32).max(p0.y);
        let xnext = x + dxdy * dy;
        let d = dy * dir;
        let (x0, x1) = if x < xnext { (x, xnext) } else { (xnext, x) };
        let x0floor = x0.floor();
        let x0i = x0floor as i32;
        let x1ceil = x1.ceil();
        let x1i = x1ceil as i32;
        let at = |i: i32| (i.clamp(0, stride as i32 - 1)) as usize;
        if x1i <= x0i + 1 {
            let xmf = 0.5 * (x + xnext) - x0floor;
            row[at(x0i)] += d - d * xmf;
            row[at(x0i + 1)] += d * xmf;
        } else {
            let s = 1.0 / (x1 - x0);
            let x0f = x0 - x0floor;
            let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
            let x1f = x1 - x1ceil + 1.0;
            let am = 0.5 * s * x1f * x1f;
            row[at(x0i)] += d * a0;
            if x1i == x0i + 2 {
                row[at(x0i + 1)] += d * (1.0 - a0 - am);
            } else {
                let a1 = s * (1.5 - x0f);
                row[at(x0i + 1)] += d * (a1 - a0);
                for xi in x0i + 2..x1i - 1 {
                    row[at(xi)] += d * s;
                }
                let a2 = a1 + (x1i - x0i - 3) as f32 * s;
                row[at(x1i - 1)] += d * (1.0 - a2 - am);
            }
            row[at(x1i)] += d * am;
        }
        x = xnext;
        y += 1;
    }
}

/// Rasterizes polygons into a full mask (Make Selection, shape coverage).
pub fn rasterize_mask(polys: &[Poly], w: u32, h: u32, rule: FillRule) -> MaskGrid {
    let mut m = MaskGrid::new(w, h);
    let lines = edges(polys);
    for (i, t) in rasterize(&lines, m.cols, m.rows, RectI::new(0, 0, w as i32, h as i32), rule) {
        m.tiles[i] = t;
    }
    m
}

// ----- stroking --------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Join {
    #[default]
    Miter,
    Round,
    Bevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Cap {
    #[default]
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Align {
    #[default]
    Center,
    Inside,
    Outside,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StrokeStyle {
    pub width: f32,
    pub join: Join,
    pub cap: Cap,
    pub miter_limit: f32,
    pub align: Align,
    /// Dash pattern in multiples of the stroke width (on, off, on, off, …).
    pub dashes: Vec<f32>,
    /// Where along the pattern the path starts, in stroke widths.
    pub dash_offset: f32,
}

impl Default for StrokeStyle {
    fn default() -> Self {
        StrokeStyle { width: 3.0, join: Join::Miter, cap: Cap::Butt, miter_limit: 4.0, align: Align::Center, dashes: vec![], dash_offset: 0.0 }
    }
}

fn signed_area(pts: &[P]) -> f32 {
    let n = pts.len();
    (0..n).map(|i| pts[i].cross(pts[(i + 1) % n])).sum::<f32>() * 0.5
}

/// Pushes a polygon with positive orientation so overlaps union under nonzero.
fn push_oriented(out: &mut Vec<Poly>, mut pts: Vec<P>) {
    if pts.len() < 3 {
        return;
    }
    if signed_area(&pts) < 0.0 {
        pts.reverse();
    }
    out.push(Poly { pts, closed: true });
}

fn circle_poly(c: P, r: f32) -> Vec<P> {
    let n = ((r * std::f32::consts::TAU / 1.5).ceil() as usize).clamp(8, 256);
    (0..n)
        .map(|k| {
            let t = k as f32 * std::f32::consts::TAU / n as f32;
            p(c.x + r * t.cos(), c.y + r * t.sin())
        })
        .collect()
}

/// Splits polylines into dash pieces (pattern lengths in pixels).
/// Splits polylines into dashes; `offset` shifts the pattern's phase along
/// each subpath (the pattern starts `offset` into its cycle).
pub fn dash(polys: &[Poly], pattern: &[f32], offset: f32) -> Vec<Poly> {
    let total: f32 = pattern.iter().sum();
    if pattern.len() < 2 || total <= 0.0 {
        return polys.to_vec();
    }
    // Phase: which element of the pattern, and how much of it is left.
    let mut phase = offset.rem_euclid(total);
    let mut k0 = 0usize;
    while phase >= pattern[k0].max(1e-3) {
        phase -= pattern[k0].max(1e-3);
        k0 = (k0 + 1) % pattern.len();
    }
    let mut out = Vec::new();
    for pl in polys {
        let mut pts = pl.pts.clone();
        if pl.closed && !pts.is_empty() {
            pts.push(pts[0]);
        }
        let mut k = k0;
        let mut left = pattern[k0].max(1e-3) - phase;
        let mut on = k0 % 2 == 0;
        let mut cur: Vec<P> = if pts.is_empty() { vec![] } else { vec![pts[0]] };
        for w in pts.windows(2) {
            let (mut a, b) = (w[0], w[1]);
            let mut seg = b.dist(a);
            while seg > 0.0 {
                if seg < left {
                    left -= seg;
                    if on {
                        cur.push(b);
                    }
                    break;
                }
                let m = a + (b - a) * (left / seg);
                if on {
                    cur.push(m);
                    if cur.len() > 1 {
                        out.push(Poly { pts: std::mem::take(&mut cur), closed: false });
                    }
                    cur.clear();
                } else {
                    cur = vec![m];
                }
                seg -= left;
                a = m;
                on = !on;
                k = (k + 1) % pattern.len();
                left = pattern[k].max(1e-3);
            }
        }
        if on && cur.len() > 1 {
            out.push(Poly { pts: cur, closed: false });
        }
    }
    out
}

/// Outline polygons of a stroke (centered); union them with nonzero fill.
pub fn stroke_polys(polys: &[Poly], st: &StrokeStyle) -> Vec<Poly> {
    let hw = (st.width * 0.5).max(0.05);
    let src = if st.dashes.len() >= 2 {
        let pat: Vec<f32> = st.dashes.iter().map(|d| d * st.width).collect();
        dash(polys, &pat, st.dash_offset * st.width)
    } else {
        polys.to_vec()
    };
    let mut out = Vec::new();
    for pl in &src {
        let mut pts: Vec<P> = Vec::with_capacity(pl.pts.len());
        for &q in &pl.pts {
            if pts.last().is_none_or(|l: &P| l.dist(q) > 1e-4) {
                pts.push(q);
            }
        }
        if pl.closed && pts.len() > 2 && pts[0].dist(*pts.last().unwrap()) <= 1e-4 {
            pts.pop();
        }
        let n = pts.len();
        if n == 1 {
            if st.cap == Cap::Round {
                push_oriented(&mut out, circle_poly(pts[0], hw));
            } else if st.cap == Cap::Square {
                let c = pts[0];
                push_oriented(&mut out, vec![p(c.x - hw, c.y - hw), p(c.x + hw, c.y - hw), p(c.x + hw, c.y + hw), p(c.x - hw, c.y + hw)]);
            }
            continue;
        }
        if n < 2 {
            continue;
        }
        let closed = pl.closed && n > 2;
        let segs = if closed { n } else { n - 1 };
        for i in 0..segs {
            let (mut a, mut b) = (pts[i], pts[(i + 1) % n]);
            let d = (b - a).norm();
            if !closed && st.cap == Cap::Square {
                if i == 0 {
                    a = a - d * hw;
                }
                if i == segs - 1 {
                    b = b + d * hw;
                }
            }
            let nrm = d.perp() * hw;
            push_oriented(&mut out, vec![a + nrm, b + nrm, b - nrm, a - nrm]);
        }
        // Joins at interior vertices.
        let joints: Vec<usize> = if closed { (0..n).collect() } else { (1..n - 1).collect() };
        for i in joints {
            let prev = pts[(i + n - 1) % n];
            let cur = pts[i];
            let next = pts[(i + 1) % n];
            let d0 = (cur - prev).norm();
            let d1 = (next - cur).norm();
            let turn = d0.cross(d1);
            let cosang = d0.dot(d1).clamp(-1.0, 1.0);
            if turn.abs() < 1e-6 && cosang > 0.0 {
                continue;
            }
            // Outer side: offsets on the side away from the turn.
            let side = if turn > 0.0 { -1.0 } else { 1.0 };
            let o0 = cur + d0.perp() * (hw * side);
            let o1 = cur + d1.perp() * (hw * side);
            let theta = cosang.acos();
            let gap = hw * (1.0 - (theta * 0.5).cos());
            match st.join {
                Join::Round if gap > 0.05 => push_oriented(&mut out, circle_poly(cur, hw)),
                Join::Miter => {
                    let half = (std::f32::consts::PI - theta) * 0.5;
                    let ratio = 1.0 / half.sin().max(1e-6);
                    if ratio <= st.miter_limit {
                        let bis = (d0.perp() * side + d1.perp() * side).norm();
                        let tip = cur + bis * (hw * ratio);
                        push_oriented(&mut out, vec![cur, o0, tip, o1]);
                    } else {
                        push_oriented(&mut out, vec![cur, o0, o1]);
                    }
                }
                _ => push_oriented(&mut out, vec![cur, o0, o1]),
            }
        }
        if !closed && st.cap == Cap::Round {
            push_oriented(&mut out, circle_poly(pts[0], hw));
            push_oriented(&mut out, circle_poly(pts[n - 1], hw));
        }
    }
    out
}

// ----- hit testing ---------------------------------------------------------------------

pub fn dist_to_segment(q: P, a: P, b: P) -> f32 {
    let ab = b - a;
    let l2 = ab.dot(ab);
    if l2 <= 1e-12 {
        return q.dist(a);
    }
    let t = ((q - a).dot(ab) / l2).clamp(0.0, 1.0);
    q.dist(a + ab * t)
}

/// Distance from `q` to the outline of the flattened path.
pub fn dist_to_polys(q: P, polys: &[Poly]) -> f32 {
    let mut best = f32::MAX;
    for pl in polys {
        let n = pl.pts.len();
        if n == 1 {
            best = best.min(q.dist(pl.pts[0]));
        }
        let segs = if pl.closed { n } else { n.saturating_sub(1) };
        for i in 0..segs {
            best = best.min(dist_to_segment(q, pl.pts[i], pl.pts[(i + 1) % n]));
        }
    }
    best
}

/// Winding number of the polygons around `q`.
pub fn winding(q: P, polys: &[Poly]) -> i32 {
    let mut w = 0;
    for pl in polys {
        let n = pl.pts.len();
        for i in 0..n {
            let (a, b) = (pl.pts[i], pl.pts[(i + 1) % n]);
            if a.y <= q.y {
                if b.y > q.y && (b - a).cross(q - a) > 0.0 {
                    w += 1;
                }
            } else if b.y <= q.y && (b - a).cross(q - a) < 0.0 {
                w -= 1;
            }
        }
    }
    w
}

pub fn contains(q: P, polys: &[Poly], rule: FillRule) -> bool {
    let w = winding(q, polys);
    match rule {
        FillRule::NonZero => w != 0,
        FillRule::EvenOdd => w % 2 != 0,
    }
}

/// Nearest point on a cubic to `q` (sampled then refined); returns (t, dist).
pub fn nearest_on_cubic(c: &Cubic, q: P) -> (f32, f32) {
    let n = 32;
    let mut best = (0.0f32, f32::MAX);
    for k in 0..=n {
        let t = k as f32 / n as f32;
        let d = cubic_at(c, t).dist(q);
        if d < best.1 {
            best = (t, d);
        }
    }
    let mut step = 0.5 / n as f32;
    for _ in 0..12 {
        for t in [best.0 - step, best.0 + step] {
            let t = t.clamp(0.0, 1.0);
            let d = cubic_at(c, t).dist(q);
            if d < best.1 {
                best = (t, d);
            }
        }
        step *= 0.5;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area_of(m: &MaskGrid) -> f64 {
        let mut s = 0f64;
        for (i, t) in m.tiles.iter().enumerate() {
            let tx = (i as u32 % m.cols) as i32;
            let ty = (i as u32 / m.cols) as i32;
            for y in 0..TS as i32 {
                for x in 0..TS as i32 {
                    let (gx, gy) = (tx * TS_I + x, ty * TS_I + y);
                    if gx >= m.w as i32 || gy >= m.h as i32 {
                        continue;
                    }
                    s += match t {
                        MaskTile::Empty => 0.0,
                        MaskTile::Full => 1.0,
                        MaskTile::Partial(px) => px[(y as usize) * TS + x as usize] as f64 / 255.0,
                    };
                }
            }
        }
        s
    }

    fn poly_area(pts: &[P]) -> f64 {
        signed_area(pts).abs() as f64
    }

    #[test]
    fn coverage_equals_exact_area() {
        // Arbitrary triangle and a non-convex polygon spanning several tiles.
        let shapes: Vec<Vec<P>> = vec![
            vec![p(10.3, 20.7), p(700.9, 90.2), p(300.4, 610.6)],
            vec![p(50.5, 50.5), p(900.25, 80.0), p(500.0, 300.0), p(880.0, 700.0), p(60.0, 650.0), p(250.0, 340.0)],
        ];
        for pts in shapes {
            let m = rasterize_mask(&[Poly { pts: pts.clone(), closed: true }], 1024, 768, FillRule::NonZero);
            let got = area_of(&m);
            let want = poly_area(&pts);
            // u8 quantization of edge pixels only: within 0.5/255 per edge pixel.
            assert!((got - want).abs() / want < 2e-3, "area {got} vs {want}");
        }
    }

    #[test]
    fn exact_per_pixel_coverage_of_axis_rect() {
        let m = rasterize_mask(&[Poly { pts: vec![p(10.25, 5.5), p(300.75, 5.5), p(300.75, 270.0), p(10.25, 270.0)], closed: true }], 512, 512, FillRule::NonZero);
        assert_eq!(m.at(10, 100), 191); // 0.75 covered
        assert_eq!(m.at(300, 100), 191);
        assert_eq!(m.at(100, 5), 128); // 0.5 covered
        assert_eq!(m.at(100, 100), 255);
        assert_eq!(m.at(301, 100), 0);
        assert_eq!(m.at(100, 270), 0);
    }

    #[test]
    fn tiles_match_single_buffer_reference() {
        // Reference: one big accumulation buffer for the whole image.
        let (w, h) = (700usize, 600usize);
        let path = Path::ellipse(350.3, 290.7, 300.2, 250.9);
        let polys = path.flatten(0.1);
        let lines = edges(&polys);
        // Simple exact reference: per-pixel 16x16 supersampling of winding.
        let m = rasterize_mask(&polys, w as u32, h as u32, FillRule::NonZero);
        let mut maxdiff = 0i32;
        for &(x, y) in &[(255, 100), (256, 100), (511, 290), (512, 290), (50, 290), (350, 40), (350, 41), (649, 290)] {
            let mut inside = 0;
            for sy in 0..16 {
                for sx in 0..16 {
                    let q = p(x as f32 + (sx as f32 + 0.5) / 16.0, y as f32 + (sy as f32 + 0.5) / 16.0);
                    if contains(q, &polys, FillRule::NonZero) {
                        inside += 1;
                    }
                }
            }
            let want = (inside as f32 / 256.0 * 255.0).round() as i32;
            maxdiff = maxdiff.max((m.at(x, y) as i32 - want).abs());
        }
        assert!(maxdiff <= 8, "maxdiff {maxdiff}");
        // Seam check: pixels on both sides of every tile border agree with a
        // re-rasterization restricted to a single tile region.
        for (tx, ty) in [(0, 0), (1, 0), (1, 1), (2, 1)] {
            let r = RectI::wh(tx * TS_I, ty * TS_I, TS_I, TS_I);
            let part = rasterize(&lines, m.cols, m.rows, r, FillRule::NonZero);
            let idx = (ty as u32 * m.cols + tx as u32) as usize;
            let got = part.iter().find(|(i, _)| *i == idx).map(|(_, t)| t.clone()).unwrap_or(MaskTile::Empty);
            for y in 0..TS {
                for x in 0..TS {
                    let a = match &got {
                        MaskTile::Empty => 0,
                        MaskTile::Full => 255,
                        MaskTile::Partial(px) => px[y * TS + x],
                    };
                    let b = match &m.tiles[idx] {
                        MaskTile::Empty => 0,
                        MaskTile::Full => 255,
                        MaskTile::Partial(px) => px[y * TS + x],
                    };
                    assert_eq!(a, b, "region raster differs in tile {tx},{ty} at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn interior_tiles_are_uniform() {
        let polys = Path::rounded_rect(100.0, 100.0, 3900.0, 3900.0, [200.0; 4]).flatten(0.1);
        let m = rasterize_mask(&polys, 4096, 4096, FillRule::NonZero);
        let full = m.tiles.iter().filter(|t| matches!(t, MaskTile::Full)).count();
        let partial = m.tiles.iter().filter(|t| matches!(t, MaskTile::Partial(_))).count();
        assert!(full >= 13 * 13, "full {full}");
        assert!(partial <= 64, "partial {partial}");
        let want = (3800.0f64 * 3800.0) - 4.0 * (200.0f64 * 200.0) * (1.0 - std::f64::consts::PI / 4.0);
        let got = area_of(&m);
        assert!((got - want).abs() / want < 1e-4, "{got} vs {want}");
    }

    #[test]
    fn even_odd_and_nonzero() {
        let outer = Poly { pts: vec![p(10.0, 10.0), p(200.0, 10.0), p(200.0, 200.0), p(10.0, 200.0)], closed: true };
        let inner = Poly { pts: vec![p(50.0, 50.0), p(150.0, 50.0), p(150.0, 150.0), p(50.0, 150.0)], closed: true };
        let nz = rasterize_mask(&[outer.clone(), inner.clone()], 256, 256, FillRule::NonZero);
        let eo = rasterize_mask(&[outer, inner], 256, 256, FillRule::EvenOdd);
        assert_eq!(nz.at(100, 100), 255);
        assert_eq!(eo.at(100, 100), 0);
        assert_eq!(eo.at(20, 20), 255);
    }

    #[test]
    fn stroke_width_area() {
        // A long horizontal line stroked 10 px wide with butt caps: area = len × w.
        let pl = Poly { pts: vec![p(100.0, 300.0), p(900.0, 300.0)], closed: false };
        let st = StrokeStyle { width: 10.0, ..Default::default() };
        let polys = stroke_polys(&[pl.clone()], &st);
        let m = rasterize_mask(&polys, 1024, 512, FillRule::NonZero);
        assert!((area_of(&m) - 8000.0).abs() < 2.0, "{}", area_of(&m));
        // Square caps add w/2 at each end; round caps add a full circle.
        let sq = rasterize_mask(&stroke_polys(&[pl.clone()], &StrokeStyle { cap: Cap::Square, ..st.clone() }), 1024, 512, FillRule::NonZero);
        assert!((area_of(&sq) - 8100.0).abs() < 2.0);
        let rd = rasterize_mask(&stroke_polys(&[pl.clone()], &StrokeStyle { cap: Cap::Round, ..st.clone() }), 1024, 512, FillRule::NonZero);
        assert!((area_of(&rd) - (8000.0 + std::f64::consts::PI * 25.0)).abs() < 3.0);
        // Dashes: 50% duty.
        let ds = rasterize_mask(&stroke_polys(&[pl], &StrokeStyle { dashes: vec![2.0, 2.0], ..st }), 1024, 512, FillRule::NonZero);
        assert!((area_of(&ds) - 4000.0).abs() < 40.0, "{}", area_of(&ds));
        // Closed square outline with miter joins has an exact area.
        let sqr = Poly { pts: vec![p(100.0, 100.0), p(300.0, 100.0), p(300.0, 300.0), p(100.0, 300.0)], closed: true };
        let ring = rasterize_mask(&stroke_polys(&[sqr], &StrokeStyle { width: 20.0, ..Default::default() }), 512, 512, FillRule::NonZero);
        let want = 220.0f64 * 220.0 - 180.0 * 180.0;
        assert!((area_of(&ring) - want).abs() < 4.0, "{} vs {want}", area_of(&ring));
    }

    #[test]
    fn buffer_raster_matches_tiled_raster() {
        let polys = Path::star(60.3, 50.7, 48.0, 40.0, 5, 0.5, 0.2).flatten(0.05);
        let lines = edges(&polys);
        let buf = raster_buffer(&lines, 130, 110, FillRule::NonZero);
        let m = rasterize_mask(&polys, 130, 110, FillRule::NonZero);
        for y in 0..110 {
            for x in 0..130 {
                assert!((buf[y * 130 + x] as i32 - m.at(x as i32, y as i32) as i32).abs() <= 1);
            }
        }
    }

    #[test]
    fn curve_fit_stays_within_error() {
        // Points on a circle and an S-curve, fitted to 0.5 px.
        let pts: Vec<P> = (0..=200).map(|k| {
            let t = k as f32 / 200.0 * std::f32::consts::TAU * 0.9;
            p(200.0 + 150.0 * t.cos(), 200.0 + 150.0 * t.sin())
        }).collect();
        let fit = fit_curve(&pts, 0.5);
        assert!(fit.len() <= 6, "{} segments for an arc", fit.len());
        for q in &pts {
            let d = fit.iter().map(|c| nearest_on_cubic(c, *q).1).fold(f32::MAX, f32::min);
            assert!(d < 0.6, "point {q:?} is {d} px from the fit");
        }
        let sp = subpath_from_cubics(&fit, false);
        assert_eq!(sp.anchors.len(), fit.len() + 1);
        assert!(sp.anchors[1..sp.anchors.len() - 1].iter().all(|a| a.smooth));
    }

    #[test]
    fn hit_testing_and_resampling() {
        let polys = Path::ellipse(100.0, 100.0, 50.0, 30.0).flatten(0.05);
        assert!(contains(p(100.0, 100.0), &polys, FillRule::NonZero));
        assert!(!contains(p(160.0, 100.0), &polys, FillRule::NonZero));
        assert!(dist_to_polys(p(150.0, 100.0), &polys) < 0.1);
        let pl = Poly { pts: vec![p(0.0, 0.0), p(100.0, 0.0)], closed: false };
        let r = resample(&pl, 10.0);
        assert_eq!(r.len(), 11);
        let c: Cubic = [p(0.0, 0.0), p(0.0, 100.0), p(100.0, 100.0), p(100.0, 0.0)];
        let (t, d) = nearest_on_cubic(&c, cubic_at(&c, 0.37));
        assert!((t - 0.37).abs() < 1e-3 && d < 1e-2);
    }
}
