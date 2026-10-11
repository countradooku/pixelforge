//! Document annotations and measurement helpers for the navigation tools:
//! Color Sampler points, Notes, Count groups, Artboards and Slices, plus the
//! geometry behind the Ruler (Straighten), Perspective Crop, rotated Crop
//! and Eyedropper sample-size averaging.

use rayon::prelude::*;

use super::composite::{ComposeInput, Scratch, compose_region};
use super::document::Layer;
use super::filters::distort::{mat_inv, mat_mul, square_to_quad};
use super::grid::{RectI, TileGrid};
use super::layers::SmartObject;
use super::px;

pub const MAX_SAMPLERS: usize = 10;

#[derive(Clone, Debug, PartialEq)]
pub struct Note {
    pub x: f32,
    pub y: f32,
    pub author: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CountGroup {
    pub name: String,
    pub color: [u8; 3],
    /// Marker radius (screen px).
    pub size: f32,
    /// Label font size (screen px).
    pub label_size: f32,
    pub visible: bool,
    pub pts: Vec<(f32, f32)>,
}

impl CountGroup {
    pub fn new(name: impl Into<String>, color: [u8; 3]) -> Self {
        CountGroup { name: name.into(), color, size: 10.0, label_size: 12.0, visible: true, pts: vec![] }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Artboard {
    pub name: String,
    pub rect: RectI,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Slice {
    pub name: String,
    pub rect: RectI,
    /// Slice Options (Save for Web's HTML): link, link target, alt text.
    pub url: String,
    pub target: String,
    pub alt: String,
    /// Layer-based slice: the rect follows this layer's content bounds.
    pub layer: Option<u64>,
}

impl Slice {
    pub fn new(name: impl Into<String>, rect: RectI) -> Self {
        Slice { name: name.into(), rect, url: String::new(), target: String::new(), alt: String::new(), layer: None }
    }
}

/// Everything the navigation / measurement tools leave on a document.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Annotations {
    pub samplers: Vec<(f32, f32)>,
    pub notes: Vec<Note>,
    pub counts: Vec<CountGroup>,
    pub artboards: Vec<Artboard>,
    pub slices: Vec<Slice>,
    /// Info panel readout mode of each color sampler (same order).
    pub sampler_modes: Vec<super::color::ColorMode>,
    /// Measurement Log and scale.
    pub measure: super::measure::Measurements,
    /// Vanishing Point planes.
    pub vp_planes: Vec<super::vanish::Plane>,
}

impl Annotations {
    /// Total Count markers.
    pub fn count_total(&self) -> usize {
        self.counts.iter().map(|g| g.pts.len()).sum()
    }
}

// ----- sampling ----------------------------------------------------------------------------

/// Average un-premultiplied RGBA of the `side × side` box centred on
/// `(x, y)` (clipped to the image) of the composite (`layers`) — the
/// Eyedropper / Color Sampler "Sample Size". Averaging is done on
/// premultiplied values, like Photoshop.
pub fn box_average(layers: &[Layer], w: u32, h: u32, x: f32, y: f32, side: u32) -> Option<[u8; 4]> {
    let (xi, yi) = (x.floor() as i32, y.floor() as i32);
    if xi < 0 || yi < 0 || xi >= w as i32 || yi >= h as i32 {
        return None;
    }
    let r = (side.max(1) as i32 - 1) / 2;
    let rect = RectI::new(xi - r, yi - r, xi + r + 1, yi + r + 1).intersect(&RectI::wh(0, 0, w as i32, h as i32));
    let (bw, bh) = (rect.w() as usize, rect.h() as usize);
    let input = ComposeInput { layers, preview: None, extra: None };
    let mut out = vec![0u32; bw * bh];
    compose_region(&input, 0, rect.x0, rect.y0, bw, bh, &mut out, &mut Scratch::default());
    let mut acc = [0u64; 4];
    for &p in &out {
        let [r, g, b, a] = px::unpack(p);
        acc[0] += r as u64;
        acc[1] += g as u64;
        acc[2] += b as u64;
        acc[3] += a as u64;
    }
    let n = out.len() as u64;
    let prem = px::pack(((acc[0] + n / 2) / n) as u8, ((acc[1] + n / 2) / n) as u8, ((acc[2] + n / 2) / n) as u8, ((acc[3] + n / 2) / n) as u8);
    Some(px::unpremultiply(prem))
}

// ----- ruler -------------------------------------------------------------------------------

/// Ruler readout: (W, H, angle in degrees counter-clockwise from +x, length).
pub fn measure(a: (f32, f32), b: (f32, f32)) -> (f32, f32, f32, f32) {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    (dx, dy, (-dy).atan2(dx).to_degrees(), dx.hypot(dy))
}

/// Rotation (degrees, positive = clockwise on screen) that makes the ruler
/// line horizontal, or vertical when it is closer to vertical — what
/// Photoshop's Straighten Layer applies.
pub fn straighten_angle(a: (f32, f32), b: (f32, f32)) -> f32 {
    let ang = measure(a, b).2;
    // Fold into (-45, 45] around the nearest axis.
    let mut d = ang % 90.0;
    if d > 45.0 {
        d -= 90.0;
    } else if d <= -45.0 {
        d += 90.0;
    }
    d
}

/// Matrix rotating by `deg` (clockwise on screen) about `(cx, cy)`.
pub fn rotation_about(deg: f32, cx: f32, cy: f32) -> [f32; 9] {
    let (s, c) = deg.to_radians().sin_cos();
    [c, -s, cx - c * cx + s * cy, s, c, cy - s * cx - c * cy, 0.0, 0.0, 1.0]
}

/// Resamples `g` through `m` (source → destination) into a `w × h` grid:
/// mip-aware bilinear with anti-aliased edges (`SmartObject::render`). Runs
/// on the GPU when that's on and wins.
pub fn warp_grid(g: &TileGrid, m: [f32; 9], w: u32, h: u32) -> TileGrid {
    let px = w as u64 * h as u64;
    let bytes = 4 * (px + g.w as u64 * g.h as u64);
    crate::engine::gpu::resample::switch("Transform (mip bilinear)", px, bytes, |gpu| crate::engine::gpu::resample::warp_grid(gpu, g, &m, w, h), || warp_grid_cpu(g, m, w, h))
}

/// [`warp_grid`] on the CPU (the reference).
pub fn warp_grid_cpu(g: &TileGrid, m: [f32; 9], w: u32, h: u32) -> TileGrid {
    let so = SmartObject::new("warp", g.clone(), m);
    let mut out = TileGrid::new(w, h);
    let b = out.bounds();
    so.render(&mut out, b);
    out
}

/// Homography mapping source quad `q` (TL, TR, BR, BL) onto the
/// `w × h` output rectangle, and that output size (from the quad's mean
/// edge lengths, like Photoshop's Perspective Crop with no fixed size).
pub fn perspective_crop(q: [(f32, f32); 4]) -> ([f32; 9], u32, u32) {
    let d = |a: (f32, f32), b: (f32, f32)| (a.0 - b.0).hypot(a.1 - b.1);
    let w = ((d(q[0], q[1]) + d(q[3], q[2])) * 0.5).round().max(1.0);
    let h = ((d(q[0], q[3]) + d(q[1], q[2])) * 0.5).round().max(1.0);
    // unit square → quad, so quad → unit square is its inverse; then scale.
    let sq = square_to_quad(q);
    let to_unit = mat_inv(&sq);
    let scale = [w, 0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0, 1.0];
    (mat_mul(&scale, &to_unit), w as u32, h as u32)
}

/// Rotated crop: the rectangle `r` (document px) turned by `deg` about its
/// centre becomes the new canvas. Returns the source → output matrix.
pub fn rotated_crop(r: RectI, deg: f32) -> [f32; 9] {
    let (cx, cy) = ((r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5);
    // Undo the frame's rotation, then move its corner to the origin.
    let rot = rotation_about(-deg, cx, cy);
    let tr = [1.0, 0.0, -(r.x0 as f32), 0.0, 1.0, -(r.y0 as f32), 0.0, 0.0, 1.0];
    mat_mul(&tr, &rot)
}

// ----- slices ------------------------------------------------------------------------------

/// User slices plus the auto slices that tile the rest of the canvas
/// (Photoshop's slicing): rows split at every user-slice edge, merged across
/// columns where possible. Returns (rect, is_user).
pub fn all_slices(user: &[Slice], w: u32, h: u32) -> Vec<(RectI, bool)> {
    let full = RectI::wh(0, 0, w as i32, h as i32);
    let rects: Vec<RectI> = user.iter().map(|s| s.rect.intersect(&full)).filter(|r| !r.is_empty()).collect();
    let mut ys: Vec<i32> = vec![0, h as i32];
    let mut xs: Vec<i32> = vec![0, w as i32];
    for r in &rects {
        ys.extend([r.y0, r.y1]);
        xs.extend([r.x0, r.x1]);
    }
    ys.sort_unstable();
    ys.dedup();
    xs.sort_unstable();
    xs.dedup();
    let mut out: Vec<(RectI, bool)> = rects.iter().map(|r| (*r, true)).collect();
    for wy in ys.windows(2) {
        // Free horizontal runs of this band, merged.
        let mut run: Option<i32> = None;
        for wx in xs.windows(2) {
            let cell = RectI::new(wx[0], wy[0], wx[1], wy[1]);
            let covered = rects.iter().any(|r| r.intersects(&cell));
            match (covered, run) {
                (false, None) => run = Some(wx[0]),
                (true, Some(x0)) => {
                    out.push((RectI::new(x0, wy[0], wx[0], wy[1]), false));
                    run = None;
                }
                _ => {}
            }
        }
        if let Some(x0) = run {
            out.push((RectI::new(x0, wy[0], w as i32, wy[1]), false));
        }
    }
    // Photoshop numbers slices left to right, top to bottom.
    out.sort_by_key(|(r, _)| (r.y0, r.x0));
    out
}

/// Composite pixels of `r` as straight RGBA bytes (for slice / artboard export).
pub fn region_rgba(layers: &[Layer], r: RectI) -> Vec<u8> {
    let (w, h) = (r.w() as usize, r.h() as usize);
    let input = ComposeInput { layers, preview: None, extra: None };
    let mut rgba = vec![0u8; w * h * 4];
    if w == 0 || h == 0 { return rgba; }
    // Compose strips so each source tile is pinned once for many rows. This
    // also avoids one temporary allocation and composition setup per scanline.
    rgba.par_chunks_mut(w * 4 * 16).enumerate().for_each_init(Scratch::default, |s, (strip, bytes)| {
        let rows = bytes.len() / (w * 4);
        let mut pixels = vec![0u32; w * rows];
        compose_region(&input, 0, r.x0, r.y0 + (strip * 16) as i32, w, rows, &mut pixels, s);
        for (dst, p) in bytes.chunks_exact_mut(4).zip(pixels) {
            dst.copy_from_slice(&px::unpremultiply(p));
        }
    });
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::document::Layer;

    #[test]
    fn strip_export_matches_scanline_composition_at_partial_edges() {
        let mut a = Layer::new(1, "base", crate::engine::ops::procedural(513, 391));
        a.opacity = 0.7;
        let mut b = Layer::new(2, "top", crate::engine::ops::procedural(513, 391));
        b.blend = crate::engine::blend::BlendMode::Multiply;
        b.opacity = 0.3;
        let layers = [a, b];
        for r in [RectI::new(3, 7, 509, 390), RectI::new(-7, -3, 300, 270)] {
            let got = region_rgba(&layers, r);
            let input = ComposeInput { layers: &layers, preview: None, extra: None };
            let mut scratch = Scratch::default();
            let mut expected = Vec::new();
            for y in r.y0..r.y1 {
                let mut pixels = vec![0; r.w() as usize];
                compose_region(&input, 0, r.x0, y, r.w() as usize, 1, &mut pixels, &mut scratch);
                expected.extend(pixels.into_iter().flat_map(px::unpremultiply));
            }
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn box_average_matches_naive_mean() {
        use crate::engine::grid::{TS, finish_tile};
        let f = |x: i32, y: i32| px::pack((x % 256) as u8, (y % 200) as u8, ((x * y) % 255) as u8, 255);
        let g = crate::engine::grid::build_grid(300, 300, |tx, ty| {
            let px: Vec<u32> = (0..TS * TS).map(|i| f(tx * TS as i32 + (i % TS) as i32, ty * TS as i32 + (i / TS) as i32)).collect();
            finish_tile(px.into_boxed_slice())
        });
        let layers = vec![Layer::new(1, "a", g.clone())];
        for (x, y, side) in [(10.0, 10.0, 1), (150.5, 120.2, 11), (2.0, 298.0, 31), (100.0, 100.0, 101)] {
            let got = box_average(&layers, 300, 300, x, y, side).unwrap();
            let r = (side as i32 - 1) / 2;
            let (xi, yi) = (x as i32, y as i32);
            let mut acc = [0f64; 3];
            let mut n = 0.0;
            for yy in (yi - r).max(0)..=(yi + r).min(299) {
                for xx in (xi - r).max(0)..=(xi + r).min(299) {
                    let [a, b, c, _] = px::unpack(g.pixel(xx, yy));
                    acc[0] += a as f64;
                    acc[1] += b as f64;
                    acc[2] += c as f64;
                    n += 1.0;
                }
            }
            for c in 0..3 {
                assert!((got[c] as f64 - acc[c] / n).abs() <= 1.0, "{side}: {got:?} vs {:?}", acc.map(|v| v / n));
            }
        }
        assert!(box_average(&layers, 300, 300, -1.0, 5.0, 5).is_none());
    }

    #[test]
    fn ruler_and_straighten() {
        let (w, h, a, l) = measure((0.0, 0.0), (100.0, -100.0));
        assert_eq!((w, h), (100.0, -100.0));
        assert!((a - 45.0).abs() < 1e-4 && (l - 141.421).abs() < 0.01);
        // A line tilted 5° up needs +5° (clockwise) to become level.
        let t = 5f32.to_radians();
        let s = straighten_angle((0.0, 0.0), (100.0 * t.cos(), -100.0 * t.sin()));
        assert!((s - 5.0).abs() < 1e-3, "{s}");
        // A near-vertical line is straightened to vertical.
        let s = straighten_angle((0.0, 0.0), (5.0, -100.0));
        assert!(s.abs() < 3.0 && s < 0.0, "{s}");
        // rotation_about by the straighten angle levels the line.
        let (a, b) = ((10.0f32, 50.0f32), (110.0 * 1.0, 50.0 - 100.0 * t.tan()));
        let m = rotation_about(straighten_angle(a, b), 0.0, 0.0);
        let f = |p: (f32, f32)| (m[0] * p.0 + m[1] * p.1 + m[2], m[3] * p.0 + m[4] * p.1 + m[5]);
        assert!((f(a).1 - f(b).1).abs() < 1e-3, "{:?} {:?}", f(a), f(b));
    }

    #[test]
    fn perspective_crop_maps_quad_to_rect() {
        let q = [(100.0, 80.0), (520.0, 120.0), (480.0, 400.0), (60.0, 360.0)];
        let (m, w, h) = perspective_crop(q);
        let f = |p: (f32, f32)| {
            let z = m[6] * p.0 + m[7] * p.1 + m[8];
            ((m[0] * p.0 + m[1] * p.1 + m[2]) / z, (m[3] * p.0 + m[4] * p.1 + m[5]) / z)
        };
        let want = [(0.0, 0.0), (w as f32, 0.0), (w as f32, h as f32), (0.0, h as f32)];
        for (a, b) in q.iter().zip(want) {
            let r = f(*a);
            assert!((r.0 - b.0).abs() < 0.05 && (r.1 - b.1).abs() < 0.05, "{a:?} -> {r:?}, want {b:?}");
        }
        // Rendering: a solid quad region becomes a solid output.
        let g = TileGrid::filled(600, 500, px::pack(10, 200, 30, 255));
        let out = warp_grid(&g, m, w, h);
        assert_eq!(out.pixel(w as i32 / 2, h as i32 / 2), px::pack(10, 200, 30, 255));
    }

    #[test]
    fn slices_partition_the_canvas() {
        let user = vec![Slice::new("a", RectI::new(100, 50, 300, 200)), Slice::new("b", RectI::new(250, 300, 500, 380))];
        let all = all_slices(&user, 640, 480);
        let area: i64 = all.iter().map(|(r, _)| r.w() as i64 * r.h() as i64).sum();
        assert_eq!(area, 640 * 480);
        for (i, (a, _)) in all.iter().enumerate() {
            for (b, _) in &all[i + 1..] {
                assert!(!a.intersects(b), "{a:?} overlaps {b:?}");
            }
        }
        assert_eq!(all.iter().filter(|s| s.1).count(), 2);
    }
}
