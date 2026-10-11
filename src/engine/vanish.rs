//! Vanishing Point: perspective planes and editing on them.
//!
//! A plane is a quadrilateral in the image (corners TL, TR, BR, BL) seen as
//! a rectangle in 3D. Its homography maps the unit square onto the quad. The
//! camera's focal length comes from the first plane's two vanishing points
//! (the plane's sides are perpendicular in 3D; principal point at the image
//! centre). With it, each plane gets a metric `(s, t)` space whose aspect is
//! the true 3D aspect, scaled so its area matches the quad's area in pixels.
//!
//! On the planes:
//! - **Tear-off planes:** perpendicular planes share an edge and extend
//!   toward the vanishing point of the first plane's normal.
//! - **Marquee copies** and the **Stamp** read source pixels through the
//!   source plane and write them through the destination plane, with
//!   optional Poisson healing.
//! - **Brush dabs** are circles in plane space (ellipses in the image).
//! - **Placing** maps pasted content onto a rectangle of a plane.

use super::filters::distort::{mat_inv, square_to_quad};
use super::grid::{RectI, TileGrid};
use super::px;

pub type Pt = (f32, f32);

/// Homogeneous helpers (f64 for vanishing points far away).
type V3 = [f64; 3];

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn hom(p: Pt) -> V3 {
    [p.0 as f64, p.1 as f64, 1.0]
}

/// Intersection of two homogeneous lines, as a finite point if it is one.
fn meet(l1: V3, l2: V3) -> Option<Pt> {
    let p = cross(l1, l2);
    (p[2].abs() > 1e-9).then(|| ((p[0] / p[2]) as f32, (p[1] / p[2]) as f32))
}

/// A perspective plane: corners TL, TR, BR, BL in image pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub c: [Pt; 4],
}

impl Plane {
    /// Unit square → image.
    pub fn h(&self) -> [f32; 9] {
        square_to_quad(self.c)
    }

    /// A plane is usable when its quad is convex and not degenerate.
    pub fn valid(&self) -> bool {
        let mut sign = 0.0f32;
        for i in 0..4 {
            let (a, b, c) = (self.c[i], self.c[(i + 1) % 4], self.c[(i + 2) % 4]);
            let z = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
            if z.abs() < 1e-3 || (sign != 0.0 && z.signum() != sign) {
                return false;
            }
            sign = z.signum();
        }
        true
    }

    /// Area of the quad in pixels.
    pub fn area(&self) -> f32 {
        let mut a = 0.0;
        for i in 0..4 {
            let (p, q) = (self.c[i], self.c[(i + 1) % 4]);
            a += p.0 * q.1 - q.0 * p.1;
        }
        (a * 0.5).abs()
    }

    /// Unit-square point → image.
    pub fn uv_to_img(&self, u: f32, v: f32) -> Pt {
        let m = self.h();
        let w = m[6] * u + m[7] * v + m[8];
        ((m[0] * u + m[1] * v + m[2]) / w, (m[3] * u + m[4] * v + m[5]) / w)
    }

    /// Image → unit-square point.
    pub fn img_to_uv(&self, p: Pt) -> Pt {
        let m = mat_inv(&self.h());
        let w = m[6] * p.0 + m[7] * p.1 + m[8];
        ((m[0] * p.0 + m[1] * p.1 + m[2]) / w, (m[3] * p.0 + m[4] * p.1 + m[5]) / w)
    }

    /// Whether image point `p` lies on the plane.
    pub fn contains(&self, p: Pt) -> bool {
        let (u, v) = self.img_to_uv(p);
        (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v)
    }

    /// Vanishing points of the u and v directions and of the normal
    /// (homogeneous), for focal length `f` and principal point `pp`.
    fn vanishing(&self, f: f64, pp: Pt) -> (V3, V3, V3) {
        let m = self.h().map(|v| v as f64);
        let (h1, h2) = ([m[0], m[3], m[6]], [m[1], m[4], m[7]]);
        let (cx, cy) = (pp.0 as f64, pp.1 as f64);
        // K⁻¹ h: camera-space directions of the plane's axes.
        let kinv = |h: V3| [(h[0] - cx * h[2]) / f, (h[1] - cy * h[2]) / f, h[2]];
        let n = cross(kinv(h1), kinv(h2));
        // K n: the normal's vanishing point.
        let vn = [f * n[0] + cx * n[2], f * n[1] + cy * n[2], n[2]];
        (h1, h2, vn)
    }

    /// 3D aspect (width / height) under focal length `f`.
    pub fn aspect(&self, f: f64, pp: Pt) -> f64 {
        let m = self.h().map(|v| v as f64);
        let (cx, cy) = (pp.0 as f64, pp.1 as f64);
        let len = |h: V3| {
            let d = [(h[0] - cx * h[2]) / f, (h[1] - cy * h[2]) / f, h[2]];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        len([m[0], m[3], m[6]]) / len([m[1], m[4], m[7]]).max(1e-12)
    }
}

/// Focal length (pixels) implied by a plane whose sides are perpendicular
/// in 3D, with the principal point `pp`; `None` when the plane is
/// (nearly) fronto-parallel or the estimate is invalid.
pub fn focal_from(plane: &Plane, pp: Pt) -> Option<f64> {
    let m = plane.h().map(|v| v as f64);
    let (cx, cy) = (pp.0 as f64, pp.1 as f64);
    let (h31, h32) = (m[6], m[7]);
    if (h31 * h32).abs() < 1e-14 {
        return None;
    }
    let num = (m[0] - cx * h31) * (m[1] - cx * h32) + (m[3] - cy * h31) * (m[4] - cy * h32);
    let f2 = -num / (h31 * h32);
    (f2 > 1.0 && f2.is_finite()).then(|| f2.sqrt())
}

/// The planes of a document and the camera they share.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Scene {
    pub planes: Vec<Plane>,
    /// Image size (principal point = centre; fallback focal = diagonal).
    pub w: u32,
    pub h: u32,
}

/// A plane's metric frame: `(s, t)` = unit-square coordinates times
/// (`sw`, `sh`), its true 3D size in "pixels".
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub plane: Plane,
    pub sw: f32,
    pub sh: f32,
}

impl Frame {
    pub fn to_img(&self, s: f32, t: f32) -> Pt {
        self.plane.uv_to_img(s / self.sw, t / self.sh)
    }

    pub fn to_plane(&self, p: Pt) -> Pt {
        let (u, v) = self.plane.img_to_uv(p);
        (u * self.sw, v * self.sh)
    }

    /// Image-space bounding box of the plane rectangle `(s0, t0)–(s1, t1)`.
    pub fn rect_bounds(&self, r: (f32, f32, f32, f32)) -> RectI {
        let q = self.rect_quad(r);
        let (x0, y0) = q.iter().fold((f32::MAX, f32::MAX), |a, p| (a.0.min(p.0), a.1.min(p.1)));
        let (x1, y1) = q.iter().fold((f32::MIN, f32::MIN), |a, p| (a.0.max(p.0), a.1.max(p.1)));
        RectI::new(x0.floor() as i32 - 1, y0.floor() as i32 - 1, x1.ceil() as i32 + 1, y1.ceil() as i32 + 1)
    }

    /// The image quad of plane rectangle `(s0, t0, s1, t1)`.
    pub fn rect_quad(&self, r: (f32, f32, f32, f32)) -> [Pt; 4] {
        [self.to_img(r.0, r.1), self.to_img(r.2, r.1), self.to_img(r.2, r.3), self.to_img(r.0, r.3)]
    }
}

impl Scene {
    pub fn new(w: u32, h: u32) -> Scene {
        Scene { planes: vec![], w, h }
    }

    fn pp(&self) -> Pt {
        (self.w as f32 * 0.5, self.h as f32 * 0.5)
    }

    /// The camera's focal length: from the first plane that yields one,
    /// else the image diagonal (a normal lens).
    pub fn focal(&self) -> f64 {
        let pp = self.pp();
        self.planes.iter().filter(|p| p.valid()).find_map(|p| focal_from(p, pp)).unwrap_or_else(|| (self.w as f64).hypot(self.h as f64))
    }

    pub fn frame(&self, i: usize) -> Frame {
        let p = self.planes[i];
        let aspect = p.aspect(self.focal(), self.pp()).clamp(0.02, 50.0) as f32;
        let sh = (p.area() / aspect).sqrt().max(1.0);
        Frame { plane: p, sw: aspect * sh, sh }
    }

    /// The topmost plane (last created) under image point `p`.
    pub fn plane_at(&self, p: Pt) -> Option<usize> {
        (0..self.planes.len()).rev().find(|&i| self.planes[i].valid() && self.planes[i].contains(p))
    }

    /// Tears off a plane perpendicular to plane `i` along its edge `edge`
    /// (0 top, 1 right, 2 bottom, 3 left), extending `depth` pixels in the
    /// image away from the plane. Returns the new plane's index.
    pub fn tear_off(&mut self, i: usize, edge: usize, depth: f32) -> Option<usize> {
        let p = self.planes[i];
        let (vu, vv, vn) = p.vanishing(self.focal(), self.pp());
        let (a, b) = (p.c[edge], p.c[(edge + 1) % 4]);
        // The shared edge's own vanishing direction.
        let ve = if edge % 2 == 0 { vu } else { vv };
        // Direction from `a` toward the normal's vanishing point (or along it, if at infinity).
        let dir = |q: Pt| -> (f32, f32) {
            let (dx, dy) = if vn[2].abs() > 1e-9 { ((vn[0] / vn[2]) as f32 - q.0, (vn[1] / vn[2]) as f32 - q.1) } else { (vn[0] as f32, vn[1] as f32) };
            let l = dx.hypot(dy).max(1e-6);
            (dx / l, dy / l)
        };
        let d = dir(a);
        // Away from the plane's interior.
        let centre = p.uv_to_img(0.5, 0.5);
        let mid = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
        let s = if (d.0 * (centre.0 - mid.0) + d.1 * (centre.1 - mid.1)) > 0.0 { -1.0 } else { 1.0 };
        let a2 = (a.0 + s * d.0 * depth, a.1 + s * d.1 * depth);
        // b' on the line b–vn, on the far edge through a' and the edge's vanishing point.
        let line_b = cross(hom(b), vn);
        let far = cross(hom(a2), ve);
        let b2 = meet(line_b, far)?;
        let np = Plane { c: [a2, b2, b, a] };
        if !np.valid() {
            return None;
        }
        self.planes.push(np);
        Some(self.planes.len() - 1)
    }

    /// Plane grid lines (image polylines) every `step` plane units.
    pub fn grid(&self, i: usize, step: f32) -> Vec<Vec<Pt>> {
        let f = self.frame(i);
        let mut out = Vec::new();
        let (nu, nv) = ((f.sw / step.max(1.0)).round().clamp(1.0, 200.0) as usize, (f.sh / step.max(1.0)).round().clamp(1.0, 200.0) as usize);
        for k in 0..=nu {
            let u = k as f32 / nu as f32;
            out.push(vec![f.plane.uv_to_img(u, 0.0), f.plane.uv_to_img(u, 1.0)]);
        }
        for k in 0..=nv {
            let v = k as f32 / nv as f32;
            out.push(vec![f.plane.uv_to_img(0.0, v), f.plane.uv_to_img(1.0, v)]);
        }
        out
    }
}

/// Bilinear sample (premultiplied) of `g` at continuous `(x, y)`.
fn sample(g: &super::filters::distort::Sampler, x: f32, y: f32) -> u32 {
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (fx.floor() as i32, fy.floor() as i32);
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let [a, b, c, d] = g.quad(x0, y0).map(px::unpack);
    let mut o = [0u8; 4];
    for k in 0..4 {
        let top = a[k] as f32 + (b[k] as f32 - a[k] as f32) * tx;
        let bot = c[k] as f32 + (d[k] as f32 - c[k] as f32) * tx;
        o[k] = (top + (bot - top) * ty).round() as u8;
    }
    px::pack(o[0], o[1], o[2], o[3])
}

/// Writes `px` (the `r`-sized region) into `g` through coverage `cov`
/// (0..255, 0 = keep), optionally Poisson-healing it into its surroundings.
fn write(g: &TileGrid, r: RectI, pxs: &[u32], cov: &[u8], heal: bool) -> TileGrid {
    let (w, h) = (r.w() as usize, r.h() as usize);
    let out = if heal {
        let mut dst = vec![0u32; w * h];
        g.read_region(r.x0, r.y0, w, h, &mut dst);
        let mask: Vec<u8> = cov.iter().enumerate().map(|(i, &c)| {
            let (x, y) = (i % w, i / w);
            (c > 0 && x > 0 && y > 0 && x + 1 < w && y + 1 < h) as u8
        }).collect();
        let mut solver = super::poisson::Solver::new();
        super::poisson::blend_rgba(&mut solver, pxs, &dst, &mask, w, h, 5)
    } else {
        pxs.to_vec()
    };
    super::retouch::write_region(g, r, &out, cov)
}

/// Marquee copy: plane rectangle `src_r` on `src` → the same-size rectangle
/// at `dst_r` (top-left) on `dst`, read from `base`, written into `g`.
pub fn copy_rect(g: &TileGrid, base: &TileGrid, src: &Frame, src_r: (f32, f32, f32, f32), dst: &Frame, dst_at: (f32, f32), heal: bool) -> (TileGrid, RectI) {
    let (sw, sh) = (src_r.2 - src_r.0, src_r.3 - src_r.1);
    let dst_r = (dst_at.0, dst_at.1, dst_at.0 + sw, dst_at.1 + sh);
    let r = dst.rect_bounds(dst_r).intersect(&g.bounds());
    if r.is_empty() {
        return (g.clone(), r);
    }
    let (w, h) = (r.w() as usize, r.h() as usize);
    let mut out = vec![0u32; w * h];
    let mut cov = vec![0u8; w * h];
    let sampler = super::filters::distort::Sampler::new(base, None, w * h, false);
    for y in 0..h {
        for x in 0..w {
            let p = ((r.x0 + x as i32) as f32 + 0.5, (r.y0 + y as i32) as f32 + 0.5);
            let (s, t) = dst.to_plane(p);
            // The source extends past the rectangle so healing sees the
            // true destination − source difference on its boundary ring.
            let q = src.to_img(s - dst_r.0 + src_r.0, t - dst_r.1 + src_r.1);
            out[y * w + x] = sample(&sampler, q.0, q.1);
            if s >= dst_r.0 && t >= dst_r.1 && s <= dst_r.2 && t <= dst_r.3 {
                cov[y * w + x] = 255;
            }
        }
    }
    (write(g, r, &out, &cov, heal), r)
}

/// What a plane dab paints.
#[derive(Clone, Copy, Debug)]
pub enum DabSource {
    /// A colour (straight RGB) at an opacity.
    Color([u8; 3], f32),
    /// Stamp: the source plane frame and the plane offset `dst − src`.
    Stamp { src: Frame, offset: (f32, f32) },
}

/// One dab of radius `r` (plane units) at plane point `(s, t)` of `frame`:
/// a circle on the plane, so an ellipse in perspective. `hardness` 0..1.
pub fn dab(g: &TileGrid, base: &TileGrid, frame: &Frame, s: f32, t: f32, r: f32, hardness: f32, what: DabSource) -> (TileGrid, RectI) {
    let rect = frame.rect_bounds((s - r, t - r, s + r, t + r)).intersect(&g.bounds());
    if rect.is_empty() {
        return (g.clone(), rect);
    }
    let (w, h) = (rect.w() as usize, rect.h() as usize);
    let mut cur = vec![0u32; w * h];
    g.read_region(rect.x0, rect.y0, w, h, &mut cur);
    let mut cov = vec![0u8; w * h];
    let hard = hardness.clamp(0.0, 0.999);
    let sampler = super::filters::distort::Sampler::new(base, None, w * h, false);
    for y in 0..h {
        for x in 0..w {
            let p = ((rect.x0 + x as i32) as f32 + 0.5, (rect.y0 + y as i32) as f32 + 0.5);
            let (ps, pt) = frame.to_plane(p);
            let d = ((ps - s).powi(2) + (pt - t).powi(2)).sqrt() / r;
            if d >= 1.0 {
                continue;
            }
            let fall = if d <= hard { 1.0 } else { 1.0 - (d - hard) / (1.0 - hard) };
            let fall = fall * fall * (3.0 - 2.0 * fall);
            let i = y * w + x;
            let (src, a) = match what {
                DabSource::Color(c, op) => (px::pack(c[0], c[1], c[2], 255), fall * op),
                DabSource::Stamp { src, offset } => {
                    let q = src.to_img(ps - offset.0, pt - offset.1);
                    (sample(&sampler, q.0, q.1), fall)
                }
            };
            let k = (a * 255.0).round() as u32;
            if k > 0 {
                cur[i] = px::lerp(cur[i], src, k);
                cov[i] = 255;
            }
        }
    }
    (super::retouch::write_region(g, rect, &cur, &cov), rect)
}

/// Heals region `r` of `g` (the pixels changed by a stroke, `mask` 255)
/// into the surrounding `base` pixels: Stamp with Heal on.
pub fn heal_region(g: &TileGrid, base: &TileGrid, r: RectI, mask: &[u8]) -> TileGrid {
    let r = r.inflate(2).intersect(&g.bounds());
    let (w, h) = (r.w() as usize, r.h() as usize);
    let mut src = vec![0u32; w * h];
    g.read_region(r.x0, r.y0, w, h, &mut src);
    let mut dst = vec![0u32; w * h];
    base.read_region(r.x0, r.y0, w, h, &mut dst);
    let _ = mask;
    let m: Vec<u8> = src.iter().zip(&dst).enumerate().map(|(i, (a, b))| {
        let (x, y) = (i % w, i / w);
        (a != b && x > 0 && y > 0 && x + 1 < w && y + 1 < h) as u8
    }).collect();
    let mut solver = super::poisson::Solver::new();
    let out = super::poisson::blend_rgba(&mut solver, &src, &dst, &m, w, h, 5);
    let cov: Vec<u8> = m.iter().map(|&v| v * 255).collect();
    super::retouch::write_region(g, r, &out, &cov)
}

/// Places `content` onto plane rectangle `rect` (s0, t0, s1, t1) of `frame`,
/// composited over `g`.
pub fn place(g: &TileGrid, content: &TileGrid, frame: &Frame, rect: (f32, f32, f32, f32)) -> (TileGrid, RectI) {
    let r = frame.rect_bounds(rect).intersect(&g.bounds());
    if r.is_empty() {
        return (g.clone(), r);
    }
    let (w, h) = (r.w() as usize, r.h() as usize);
    let mut cur = vec![0u32; w * h];
    g.read_region(r.x0, r.y0, w, h, &mut cur);
    let mut cov = vec![0u8; w * h];
    let (cw, ch) = (content.w as f32, content.h as f32);
    let sampler = super::filters::distort::Sampler::new(content, None, w * h, false);
    for y in 0..h {
        for x in 0..w {
            let p = ((r.x0 + x as i32) as f32 + 0.5, (r.y0 + y as i32) as f32 + 0.5);
            let (s, t) = frame.to_plane(p);
            let (u, v) = ((s - rect.0) / (rect.2 - rect.0), (t - rect.1) / (rect.3 - rect.1));
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                continue;
            }
            let src = sample(&sampler, u * cw, v * ch);
            if px::alpha(src) == 0 {
                continue;
            }
            let i = y * w + x;
            cur[i] = px::over(src, cur[i]);
            cov[i] = 255;
        }
    }
    (super::retouch::write_region(g, r, &cur, &cov), r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ops;

    /// A floor seen from above at an angle: the projection of a 3D
    /// rectangle through a pinhole camera with focal length `f`.
    fn floor(f: f64, w: u32, h: u32) -> (Plane, f64) {
        let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
        // Points on the plane y = 1 (camera looking down 30°).
        let (sin, cos) = (30f64.to_radians().sin(), 30f64.to_radians().cos());
        let (ys, yc) = (25f64.to_radians().sin(), 25f64.to_radians().cos());
        let proj = |x: f64, z: f64| {
            // Turn the floor point (x, 1.5, z) about the vertical (yaw 25°,
            // around the rectangle's middle), then about the x axis.
            let (x, z) = (x * yc - (z - 4.5) * ys, x * ys + (z - 4.5) * yc + 4.5);
            let (y0, z0) = (1.5, z);
            let (y, zz) = (y0 * cos - z0 * sin, y0 * sin + z0 * cos);
            ((cx + f * x / zz) as f32, (cy + f * y / zz) as f32)
        };
        // A 4 × 3 rectangle (aspect 4 / 3).
        (Plane { c: [proj(-2.0, 6.0), proj(2.0, 6.0), proj(2.0, 3.0), proj(-2.0, 3.0)] }, 4.0 / 3.0)
    }

    #[test]
    fn focal_and_aspect_are_recovered() {
        let (p, aspect) = floor(1200.0, 2000, 1500);
        assert!(p.valid());
        let f = focal_from(&p, (1000.0, 750.0)).unwrap();
        assert!((f / 1200.0 - 1.0).abs() < 0.01, "focal {f}");
        assert!((p.aspect(f, (1000.0, 750.0)) / aspect - 1.0).abs() < 0.01);
        // Fronto-parallel planes give no focal estimate.
        let flat = Plane { c: [(100.0, 100.0), (300.0, 100.0), (300.0, 200.0), (100.0, 200.0)] };
        assert!(focal_from(&flat, (1000.0, 750.0)).is_none());
        // Non-convex quads are invalid.
        assert!(!Plane { c: [(0.0, 0.0), (100.0, 0.0), (10.0, 10.0), (0.0, 100.0)] }.valid());
    }

    #[test]
    fn tear_off_is_perpendicular() {
        let (p, _) = floor(1200.0, 2000, 1500);
        let mut sc = Scene::new(2000, 1500);
        sc.planes.push(p);
        let f0 = sc.focal();
        let k = sc.tear_off(0, 0, 250.0).expect("tear-off");
        let wall = sc.planes[k];
        // The wall shares the floor's far edge…
        assert_eq!((wall.c[2], wall.c[3]), (p.c[1], p.c[0]));
        // …rises above it (smaller y)…
        assert!(wall.c[0].1 < p.c[0].1);
        // …and is perpendicular: the two planes' normals (camera space) are orthogonal.
        let pp = (1000.0, 750.0);
        let (_, _, n1) = p.vanishing(f0, pp);
        let (_, _, n2) = wall.vanishing(f0, pp);
        let back = |v: V3| [(v[0] - pp.0 as f64 * v[2]) / f0, (v[1] - pp.1 as f64 * v[2]) / f0, v[2]];
        let (a, b) = (back(n1), back(n2));
        let cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / ((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt() * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt());
        assert!(cos.abs() < 0.02, "normals not perpendicular: cos {cos}");
        assert!(!sc.grid(0, 50.0).is_empty());
    }

    #[test]
    fn copy_and_dabs_follow_the_plane() {
        let (p, _) = floor(1200.0, 2000, 1500);
        let mut sc = Scene::new(2000, 1500);
        sc.planes.push(p);
        let fr = sc.frame(0);
        // A checkerboard on the plane (in plane space) and a copy of one square.
        let base = ops::fill(&TileGrid::new(2000, 1500), None, [255, 255, 255], 1.0);
        let (g, _) = dab(&base, &base, &fr, fr.sw * 0.25, fr.sh * 0.5, fr.sh * 0.1, 1.0, DabSource::Color([200, 0, 0], 1.0));
        // The dab is a circle on the plane: its plane-space extent is equal in s and t.
        let centre = fr.to_img(fr.sw * 0.25, fr.sh * 0.5);
        assert_eq!(px::unpack(g.pixel(centre.0 as i32, centre.1 as i32))[1], 0);
        let edge_s = fr.to_img(fr.sw * 0.25 + fr.sh * 0.09, fr.sh * 0.5);
        let edge_t = fr.to_img(fr.sw * 0.25, fr.sh * 0.5 + fr.sh * 0.09);
        assert!(px::unpack(g.pixel(edge_s.0 as i32, edge_s.1 as i32))[1] < 30);
        assert!(px::unpack(g.pixel(edge_t.0 as i32, edge_t.1 as i32))[1] < 30);
        // Copy that area to the right half: the red disc appears there, in perspective.
        let r = (fr.sw * 0.1, fr.sh * 0.35, fr.sw * 0.4, fr.sh * 0.65);
        let (c, _) = copy_rect(&g, &g, &fr, r, &fr, (fr.sw * 0.55, fr.sh * 0.35), false);
        let moved = fr.to_img(fr.sw * 0.7, fr.sh * 0.5);
        assert_eq!(px::unpack(c.pixel(moved.0 as i32, moved.1 as i32))[1], 0);
        // Stamp: the same disc cloned through the plane.
        let (s, _) = dab(&g, &g, &fr, fr.sw * 0.75, fr.sh * 0.5, fr.sh * 0.05, 1.0, DabSource::Stamp { src: fr, offset: (fr.sw * 0.5, 0.0) });
        let st = fr.to_img(fr.sw * 0.75, fr.sh * 0.5);
        assert_eq!(px::unpack(s.pixel(st.0 as i32, st.1 as i32))[1], 0);
        // Placing a square image fills the plane rectangle (corners map).
        let content = ops::fill(&TileGrid::new(64, 64), None, [0, 0, 255], 1.0);
        let (pl, _) = place(&base, &content, &fr, (0.0, 0.0, fr.sw * 0.5, fr.sh * 0.5));
        let q = fr.to_img(fr.sw * 0.25, fr.sh * 0.25);
        assert_eq!(px::unpack(pl.pixel(q.0 as i32, q.1 as i32)), [0, 0, 255, 255]);
        let out = fr.to_img(fr.sw * 0.75, fr.sh * 0.75);
        assert_eq!(px::unpack(pl.pixel(out.0 as i32, out.1 as i32)), [255, 255, 255, 255]);
    }

    #[test]
    fn healing_blends_into_the_surroundings() {
        // A grey patch copied onto a gradient: healed, it takes the gradient's level.
        let grad = crate::engine::grid::build_grid(400, 300, |tx, ty| {
            let mut t = vec![0u32; 256 * 256];
            for y in 0..256 {
                for x in 0..256 {
                    let v = ((tx * 256 + x as i32) as f32 / 400.0 * 200.0) as u8 + 30;
                    let _ = ty;
                    t[y * 256 + x] = px::pack(v, v, v, 255);
                }
            }
            crate::engine::grid::finish_tile(t.into_boxed_slice())
        });
        let flat = Plane { c: [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)] };
        let mut sc = Scene::new(400, 300);
        sc.planes.push(flat);
        let fr = sc.frame(0);
        let r = (fr.sw * 0.05, fr.sh * 0.3, fr.sw * 0.25, fr.sh * 0.6);
        let (plain, _) = copy_rect(&grad, &grad, &fr, r, &fr, (fr.sw * 0.7, fr.sh * 0.3), false);
        let (healed, _) = copy_rect(&grad, &grad, &fr, r, &fr, (fr.sw * 0.7, fr.sh * 0.3), true);
        let at = |g: &TileGrid| px::unpack(g.pixel(340, 135))[0] as i32;
        let target = at(&grad);
        assert!((at(&healed) - target).abs() < (at(&plain) - target).abs() / 3, "plain {} healed {} target {target}", at(&plain), at(&healed));
    }
}
