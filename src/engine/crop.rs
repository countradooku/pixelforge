//! Crop beyond the rectangle: rotated / straightened crops resampled with
//! bilinear, bicubic (Catmull-Rom) or Lanczos-3 filters; "Delete Cropped
//! Pixels" off, where pixel layers keep what falls outside the canvas as
//! hidden pixels (brought back by Move, a larger crop or Reveal All);
//! Content-Aware fill of canvas area a crop adds; and the crop overlays
//! (thirds, grid, diagonal, triangle, golden ratio, golden spiral).

use std::sync::Arc;

use super::filters::distort::mat_inv;
use super::grid::{RectI, TS, TS_I, TileGrid, build_grid, finish_tile};
use super::ops;
use super::px;
use super::selection::MaskGrid;
use super::vector::{FillRule, Poly, p, rasterize_mask};

/// Resampling filter for rotated crops, Straighten and canvas rotation.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Interp {
    Bilinear,
    /// Catmull-Rom (Photoshop's "Bicubic").
    #[default]
    Bicubic,
    /// Lanczos, 3 lobes.
    Lanczos,
}

impl Interp {
    pub const ALL: [Interp; 3] = [Interp::Bilinear, Interp::Bicubic, Interp::Lanczos];

    pub fn label(self) -> &'static str {
        match self {
            Interp::Bilinear => "Bilinear",
            Interp::Bicubic => "Bicubic",
            Interp::Lanczos => "Lanczos",
        }
    }

    /// Taps on each side of the sample point.
    pub fn radius(self) -> i32 {
        match self {
            Interp::Bilinear => 1,
            Interp::Bicubic => 2,
            Interp::Lanczos => 3,
        }
    }

    /// Weights of the `2 * radius` taps starting at `floor(t) - radius + 1`
    /// for fractional offset `f` in [0, 1).
    #[inline]
    fn weights(self, f: f32, out: &mut [f32; 6]) {
        match self {
            Interp::Bilinear => {
                out[0] = 1.0 - f;
                out[1] = f;
            }
            Interp::Bicubic => {
                let (t2, t3) = (f * f, f * f * f);
                out[0] = (-t3 + 2.0 * t2 - f) * 0.5;
                out[1] = (3.0 * t3 - 5.0 * t2 + 2.0) * 0.5;
                out[2] = (-3.0 * t3 + 4.0 * t2 + f) * 0.5;
                out[3] = (t3 - t2) * 0.5;
            }
            Interp::Lanczos => {
                let mut s = 0.0;
                for (k, o) in out.iter_mut().enumerate() {
                    let x = f - (k as f32 - 2.0);
                    *o = lanczos3(x);
                    s += *o;
                }
                for o in out.iter_mut() {
                    *o /= s;
                }
            }
        }
    }
}

#[inline]
fn lanczos3(x: f32) -> f32 {
    let x = x.abs();
    if x < 1e-5 {
        return 1.0;
    }
    if x >= 3.0 {
        return 0.0;
    }
    let px_ = std::f32::consts::PI * x;
    3.0 * px_.sin() * (px_ / 3.0).sin() / (px_ * px_)
}

/// Resamples `g` through the affine source → destination matrix `m` into a
/// `w × h` grid (transparent outside the source). Runs on the GPU when
/// that's on and wins, within 1/255 of the CPU.
pub fn warp_affine(g: &TileGrid, m: &[f32; 9], w: u32, h: u32, interp: Interp) -> TileGrid {
    let px = w as u64 * h as u64;
    let bytes = 4 * (px + g.w as u64 * g.h as u64);
    let op = format!("Transform ({})", interp.label());
    super::gpu::resample::switch(&op, px, bytes, |gpu| super::gpu::resample::warp_affine(gpu, g, m, w, h, interp), || warp_affine_cpu(g, m, w, h, interp))
}

/// The source footprint of output tile `(tx, ty)` for [`warp_affine`]:
/// the rect of source pixels its filter taps can reach.
pub fn affine_footprint(inv: &[f32; 9], tx: i32, ty: i32, rad: i32) -> RectI {
    let (x0, y0) = (tx * TS_I, ty * TS_I);
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for (x, y) in [(x0, y0), (x0 + TS_I, y0), (x0, y0 + TS_I), (x0 + TS_I, y0 + TS_I)] {
        let (sx, sy) = (inv[0] * x as f32 + inv[1] * y as f32 + inv[2], inv[3] * x as f32 + inv[4] * y as f32 + inv[5]);
        lo = [lo[0].min(sx), lo[1].min(sy)];
        hi = [hi[0].max(sx), hi[1].max(sy)];
    }
    RectI::new(lo[0].floor() as i32 - rad - 1, lo[1].floor() as i32 - rad - 1, hi[0].ceil() as i32 + rad + 1, hi[1].ceil() as i32 + rad + 1)
}

/// [`warp_affine`] on the CPU (the reference).
pub fn warp_affine_cpu(g: &TileGrid, m: &[f32; 9], w: u32, h: u32, interp: Interp) -> TileGrid {
    let inv = mat_inv(m);
    let rad = interp.radius();
    let n = (2 * rad) as usize;
    let src = g.bounds();
    build_grid(w, h, |tx, ty| {
        let (x0, y0) = (tx * TS_I, ty * TS_I);
        // Source footprint of the tile.
        let fr = affine_footprint(&inv, tx, ty, rad);
        if !fr.intersects(&src) {
            return None;
        }
        if src.contains_rect(&fr) {
            if let Some(u) = g.region_uniform(fr) {
                return u;
            }
        }
        let (bw, bh) = (fr.w() as usize, fr.h() as usize);
        let mut b = vec![0u32; bw * bh];
        g.read_region(fr.x0, fr.y0, bw, bh, &mut b);
        // Edge tiles hold padding past the image: outside it is transparent.
        if !src.contains_rect(&fr) {
            for (y, row) in b.chunks_exact_mut(bw).enumerate() {
                let gy = fr.y0 + y as i32;
                if gy < src.y0 || gy >= src.y1 {
                    row.fill(0);
                    continue;
                }
                for (x, v) in row.iter_mut().enumerate() {
                    let gx = fr.x0 + x as i32;
                    if gx < src.x0 || gx >= src.x1 {
                        *v = 0;
                    }
                }
            }
        }
        let mut out = vec![0u32; TS * TS].into_boxed_slice();
        let (mut wx, mut wy) = ([0f32; 6], [0f32; 6]);
        for y in 0..TS {
            let yc = (y0 + y as i32) as f32 + 0.5;
            for x in 0..TS {
                let xc = (x0 + x as i32) as f32 + 0.5;
                // Positions within 1/1000 px of a pixel centre snap to it
                // (exact right-angle turns and flips).
                let snap = |v: f32| if (v - v.round()).abs() < 1e-3 { v.round() } else { v };
                let sx = snap(inv[0] * xc + inv[1] * yc + inv[2] - 0.5 - fr.x0 as f32);
                let sy = snap(inv[3] * xc + inv[4] * yc + inv[5] - 0.5 - fr.y0 as f32);
                let (ix, iy) = (sx.floor(), sy.floor());
                interp.weights(sx - ix, &mut wx);
                interp.weights(sy - iy, &mut wy);
                let (bx, by) = (ix as isize - rad as isize + 1, iy as isize - rad as isize + 1);
                let mut acc = [0f32; 4];
                for (j, &wyj) in wy.iter().take(n).enumerate() {
                    let row = (by + j as isize).clamp(0, bh as isize - 1) as usize * bw;
                    for (i, &wxi) in wx.iter().take(n).enumerate() {
                        let v = b[row + (bx + i as isize).clamp(0, bw as isize - 1) as usize];
                        if v == 0 {
                            continue;
                        }
                        let q = px::unpack(v);
                        let k = wxi * wyj;
                        acc[0] += q[0] as f32 * k;
                        acc[1] += q[1] as f32 * k;
                        acc[2] += q[2] as f32 * k;
                        acc[3] += q[3] as f32 * k;
                    }
                }
                let a = acc[3].round().clamp(0.0, 255.0);
                if a > 0.0 {
                    let c = |v: f32| v.round().clamp(0.0, a) as u8;
                    out[y * TS + x] = px::pack(c(acc[0]), c(acc[1]), c(acc[2]), a as u8);
                }
            }
        }
        finish_tile(out)
    })
}

// ----- hidden pixels ---------------------------------------------------------------------

/// A pixel layer's pixels beyond the canvas ("Delete Cropped Pixels" off):
/// the layer's full extent, its pixel (0, 0) at document `(x, y)`. Inside the
/// canvas the layer's own grid is authoritative.
#[derive(Clone)]
pub struct Hidden {
    pub grid: TileGrid,
    pub x: i32,
    pub y: i32,
    /// The layer mask over the same rect (opaque gray), when the layer has one.
    pub mask: Option<TileGrid>,
}

impl Hidden {
    pub fn rect(&self) -> RectI {
        RectI::wh(self.x, self.y, self.grid.w as i32, self.grid.h as i32)
    }
}

/// `canvas` (at the document origin) merged over `beyond` (its pixel (0, 0)
/// at document `(bx, by)`): the union and its document origin.
fn merge(canvas: &TileGrid, beyond: &TileGrid, bx: i32, by: i32) -> (TileGrid, i32, i32) {
    let cb = canvas.bounds();
    let br = RectI::wh(bx, by, beyond.w as i32, beyond.h as i32);
    let u = br.union(&cb);
    let out = build_grid(u.w() as u32, u.h() as u32, |tx, ty| {
        let r = TileGrid::tile_rect(tx, ty).translate(u.x0, u.y0);
        let mut a = vec![0u32; TPX];
        if cb.contains_rect(&r) {
            canvas.read_region(r.x0, r.y0, TS, TS, &mut a);
            return finish_tile(a.into_boxed_slice());
        }
        beyond.read_region(r.x0 - bx, r.y0 - by, TS, TS, &mut a);
        let c = r.intersect(&cb);
        if !c.is_empty() {
            let mut b = vec![0u32; TPX];
            canvas.read_region(r.x0, r.y0, TS, TS, &mut b);
            for y in c.y0..c.y1 {
                let o = (y - r.y0) as usize * TS;
                let (s, e) = (o + (c.x0 - r.x0) as usize, o + (c.x1 - r.x0) as usize);
                a[s..e].copy_from_slice(&b[s..e]);
            }
        }
        // `beyond`'s edge tiles hold padding past its image: keep only what
        // lies inside it (or on the canvas, copied above).
        if !br.contains_rect(&r) {
            for y in r.y0..r.y1 {
                for x in r.x0..r.x1 {
                    let on = |q: &RectI| x >= q.x0 && x < q.x1 && y >= q.y0 && y < q.y1;
                    if !on(&br) && !on(&cb) {
                        a[(y - r.y0) as usize * TS + (x - r.x0) as usize] = 0;
                    }
                }
            }
        }
        finish_tile(a.into_boxed_slice())
    });
    (out, u.x0, u.y0)
}

/// The layer's whole pixels (canvas grid merged over its hidden pixels) and
/// the document position of their origin.
pub fn full_pixels(grid: &TileGrid, hidden: Option<&Hidden>) -> (TileGrid, i32, i32) {
    match hidden {
        Some(hd) => merge(grid, &hd.grid, hd.x, hd.y),
        None => (grid.clone(), 0, 0),
    }
}

const TPX: usize = TS * TS;

/// Splits full pixels placed at document `(x, y)` into the `w × h` canvas
/// grid and, when some content lies beyond the canvas, the content rect
/// (in `full`'s coordinates) to keep hidden.
fn split(full: &TileGrid, x: i32, y: i32, w: u32, h: u32) -> (TileGrid, Option<RectI>) {
    let canvas = ops::crop(full, RectI::wh(-x, -y, w as i32, h as i32));
    let Some(c) = super::xform::content_rect(full) else { return (canvas, None) };
    let inside = RectI::wh(0, 0, w as i32, h as i32).contains_rect(&c.translate(x, y));
    (canvas, (!inside).then_some(c))
}

/// A layer moved through a transform with nothing lost beyond the canvas.
pub struct Kept {
    pub grid: TileGrid,
    pub hidden: Option<Arc<Hidden>>,
    /// The layer mask on the new canvas (when the layer has one).
    pub mask: Option<MaskGrid>,
}

/// The layer's pixels (with hidden ones) through `m`, re-split onto a
/// `w × h` canvas: nothing is lost outside it.
#[cfg(test)]
pub fn transformed_keep(grid: &TileGrid, hidden: Option<&Hidden>, m: &[f32; 9], w: u32, h: u32, interp: Interp) -> (TileGrid, Option<Arc<Hidden>>) {
    let k = transformed_keep_masked(grid, hidden, None, m, w, h, interp);
    (k.grid, k.hidden)
}

/// [`transformed_keep`] with the layer mask: its hidden part travels with
/// the pixels (areas it never covered reveal, like Photoshop's masks).
pub fn transformed_keep_masked(grid: &TileGrid, hidden: Option<&Hidden>, mask: Option<&MaskGrid>, m: &[f32; 9], w: u32, h: u32, interp: Interp) -> Kept {
    let white = px::pack(255, 255, 255, 255);
    let (full, ox, oy) = full_pixels(grid, hidden);
    let full_mask = mask.map(|mk| {
        let cm = mk.to_gray();
        match hidden {
            Some(hd) => merge(&cm, &hd.mask.clone().unwrap_or_else(|| TileGrid::filled(hd.grid.w, hd.grid.h, white)), hd.x, hd.y).0,
            None => cm,
        }
    });
    // Mask values outside everything it covered reveal.
    let to_mask = |g: &TileGrid| MaskGrid::from_gray(&ops::matte(g, [255, 255, 255]));
    let Some(c) = super::xform::content_rect(&full) else {
        return Kept { grid: TileGrid::new(w, h), hidden: None, mask: mask.map(|_| MaskGrid::full(w, h)) };
    };
    // Full pixels → document → destination.
    let to_doc = [1.0, 0.0, ox as f32, 0.0, 1.0, oy as f32, 0.0, 0.0, 1.0];
    let mm = super::filters::distort::mat_mul(m, &to_doc);
    let (moved, moved_mask, x, y) = match translation(&mm) {
        Some((dx, dy)) => (full, full_mask, dx, dy),
        None => {
            // Destination bounding box of the content.
            let cs = [(c.x0, c.y0), (c.x1, c.y0), (c.x1, c.y1), (c.x0, c.y1)].map(|(x, y)| p(x as f32, y as f32).xform(&mm));
            let x0 = cs.iter().map(|q| q.x).fold(f32::MAX, f32::min).floor() as i32 - 1;
            let y0 = cs.iter().map(|q| q.y).fold(f32::MAX, f32::min).floor() as i32 - 1;
            let x1 = cs.iter().map(|q| q.x).fold(f32::MIN, f32::max).ceil() as i32 + 1;
            let y1 = cs.iter().map(|q| q.y).fold(f32::MIN, f32::max).ceil() as i32 + 1;
            let shift = [1.0, 0.0, -x0 as f32, 0.0, 1.0, -y0 as f32, 0.0, 0.0, 1.0];
            let mw = super::filters::distort::mat_mul(&shift, &mm);
            let (ww, wh) = ((x1 - x0) as u32, (y1 - y0) as u32);
            let warped = warp_affine(&full, &mw, ww, wh, interp);
            let warped_mask = full_mask.map(|fm| warp_affine(&fm, &mw, ww, wh, Interp::Bilinear));
            (warped, warped_mask, x0, y0)
        }
    };
    let (canvas, r) = split(&moved, x, y, w, h);
    let canvas_mask = moved_mask.as_ref().map(|mg| to_mask(&ops::crop(mg, RectI::wh(-x, -y, w as i32, h as i32))));
    let hidden = r.map(|r| {
        let mask = moved_mask.as_ref().map(|mg| ops::matte(&ops::crop(mg, r), [255, 255, 255]));
        Arc::new(Hidden { grid: ops::crop(&moved, r), x: x + r.x0, y: y + r.y0, mask })
    });
    Kept { grid: canvas, hidden, mask: canvas_mask }
}

/// Integer translation `(tx, ty)` when `m` is one.
pub fn translation(m: &[f32; 9]) -> Option<(i32, i32)> {
    let id = (m[0] - 1.0).abs() < 1e-6 && m[1].abs() < 1e-6 && m[3].abs() < 1e-6 && (m[4] - 1.0).abs() < 1e-6 && m[6].abs() < 1e-9 && m[7].abs() < 1e-9;
    let (tx, ty) = (m[2].round(), m[5].round());
    (id && (m[2] - tx).abs() < 1e-4 && (m[5] - ty).abs() < 1e-4).then_some((tx as i32, ty as i32))
}

/// Document rect covered by a layer's pixels, hidden ones included.
pub fn extent(grid: &TileGrid, hidden: Option<&Hidden>) -> Option<RectI> {
    let own = super::xform::content_rect(grid);
    match (own, hidden) {
        (o, None) => o,
        (None, Some(h)) => Some(h.rect()),
        (Some(o), Some(h)) => Some(o.union(&h.rect())),
    }
}

// ----- straighten / content-aware --------------------------------------------------------

/// Crop frame that straightens a `w × h` image by `ang` degrees (the
/// rotation that levels the line): the largest same-aspect rectangle inside
/// the rotated image, and the frame angle (the opposite rotation).
pub fn straighten_frame(w: u32, h: u32, ang: f32) -> (RectI, f32) {
    let (wf, hf) = (w as f32, h as f32);
    let (s, c) = ang.to_radians().abs().sin_cos();
    let k = (wf / (wf * c + hf * s)).min(hf / (wf * s + hf * c));
    let (cw, ch) = (wf * k, hf * k);
    let x0 = ((wf - cw) * 0.5).round() as i32;
    let y0 = ((hf - ch) * 0.5).round() as i32;
    (RectI::new(x0, y0, x0 + cw.round() as i32, y0 + ch.round() as i32), -ang)
}

/// Canvas area of the `w × h` result that the old `ow × oh` canvas (moved
/// through `m`) doesn't cover: what Content-Aware crop fills. Grown by 2 px
/// so the antialiased edge is refilled too.
pub fn new_area(m: &[f32; 9], ow: u32, oh: u32, w: u32, h: u32) -> MaskGrid {
    let pts = [(0.0, 0.0), (ow as f32, 0.0), (ow as f32, oh as f32), (0.0, oh as f32)].iter().map(|&(x, y)| p(x, y).xform(m)).collect();
    let inside = rasterize_mask(&[Poly { pts, closed: true }], w, h, FillRule::NonZero);
    inside.contract(2.0).invert().threshold()
}

/// Fills `hole` of `g` with Content-Aware Fill, sampling a band around it.
/// Holes on the image border (what crops add) are filled on a padded copy,
/// so PatchMatch's patches never reach past the edge.
pub fn content_aware(g: &TileGrid, hole: &MaskGrid) -> TileGrid {
    if super::retouch::mask_bbox(hole).is_empty() {
        return g.clone();
    }
    const M: i32 = 24;
    let (w, h) = (g.w as i32, g.h as i32);
    let padded = ops::crop(g, RectI::new(-M, -M, w + M, h + M));
    let (pw, ph) = (padded.w, padded.h);
    let moved = ops::crop(&hole.to_gray(), RectI::new(-M, -M, w + M, h + M));
    let frame = MaskGrid::from_selection(&super::selection::Selection::rect(RectI::new(M, M, w + M, h + M)), pw, ph).invert();
    let hole_p = MaskGrid::from_gray(&moved).combine(&frame, super::masks::SelOp::Add);
    // Long border holes converge poorly in one PatchMatch run: fill them in
    // windows, in two checkerboard passes (the second sees the first's fill).
    use rayon::prelude::*;
    const WIN: i32 = 640;
    // Windows evenly over the hole itself; the outermost ones also take the
    // padding margin beside them.
    let hb = super::retouch::mask_bbox(hole).translate(M, M);
    let (nx, ny) = ((hb.w() + WIN - 1) / WIN, (hb.h() + WIN - 1) / WIN);
    let (sx, sy) = ((hb.w() + nx - 1) / nx, (hb.h() + ny - 1) / ny);
    let padded_bounds = RectI::wh(0, 0, pw as i32, ph as i32);
    let mut cur = padded;
    for pass in 0..2 {
        let wins: Vec<RectI> = (0..ny)
            .flat_map(|j| (0..nx).map(move |i| (i, j)))
            .filter(|(i, j)| (i + j) % 2 == pass)
            .map(|(i, j)| {
                let mut r = RectI::wh(hb.x0 + i * sx, hb.y0 + j * sy, sx, sy).intersect(&hb);
                if i == 0 {
                    r.x0 -= M;
                }
                if j == 0 {
                    r.y0 -= M;
                }
                if i == nx - 1 {
                    r.x1 += M;
                }
                if j == ny - 1 {
                    r.y1 += M;
                }
                // Second-pass windows overlap the first pass's edges, whose
                // blend ring saw the still-empty neighbours.
                r.inflate(16 * pass).intersect(&padded_bounds)
            })
            .collect();
        let src = cur.clone();
        let fills: Vec<super::retouch::CafResult> = wins
            .par_iter()
            .filter_map(|&r| {
                let sel = super::selection::Selection::rect(r);
                let hw = hole_p.combine(&MaskGrid::from_selection(&sel, pw, ph), super::masks::SelOp::Intersect);
                if super::retouch::mask_bbox(&hw).is_empty() {
                    return None;
                }
                let sampling = super::retouch::caf_sampling(&src, &hw, super::retouch::SamplingArea::Auto).combine(&hole_p, super::masks::SelOp::Subtract);
                let adapt = super::patchmatch::Adapt::default();
                super::retouch::content_aware_fill(&src, &hw, &sampling, adapt).or_else(|| {
                    // A corner window whose band lies in the hole: sample wider.
                    let wide = MaskGrid::from_selection(&super::selection::Selection::rect(r.inflate(WIN)), pw, ph).combine(&hole_p, super::masks::SelOp::Subtract);
                    super::retouch::content_aware_fill(&src, &hw, &wide, adapt)
                })
            })
            .collect();
        for r in fills {
            cur = super::retouch::write_region(&cur, r.region, &r.px, &r.cov);
        }
    }
    ops::crop(&cur, RectI::new(M, M, w + M, h + M))
}

// ----- overlays --------------------------------------------------------------------------

/// Composition guides drawn inside the crop frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CropOverlay {
    #[default]
    Thirds,
    Grid,
    Diagonal,
    Triangle,
    GoldenRatio,
    GoldenSpiral,
    None,
}

impl CropOverlay {
    pub const ALL: [CropOverlay; 7] =
        [CropOverlay::Thirds, CropOverlay::Grid, CropOverlay::Diagonal, CropOverlay::Triangle, CropOverlay::GoldenRatio, CropOverlay::GoldenSpiral, CropOverlay::None];

    pub fn label(self) -> &'static str {
        match self {
            CropOverlay::Thirds => "Thirds",
            CropOverlay::Grid => "Grid",
            CropOverlay::Diagonal => "Diagonal",
            CropOverlay::Triangle => "Triangle",
            CropOverlay::GoldenRatio => "Golden Ratio",
            CropOverlay::GoldenSpiral => "Golden Spiral",
            CropOverlay::None => "None",
        }
    }

    /// Orientations Shift+O cycles through.
    pub fn orientations(self) -> u8 {
        match self {
            CropOverlay::Triangle => 2,
            CropOverlay::GoldenSpiral => 4,
            _ => 1,
        }
    }
}

const PHI: f32 = 1.618_034;

/// Overlay polylines in frame coordinates (`0..w × 0..h`). `orient` mirrors
/// (bit 0: horizontally, bit 1: vertically); `grid` is the Grid spacing.
pub fn overlay_lines(kind: CropOverlay, orient: u8, w: f32, h: f32, grid: f32) -> Vec<Vec<(f32, f32)>> {
    let mut out: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut line = |a: (f32, f32), b: (f32, f32)| out.push(vec![a, b]);
    let fracs = |f: &[f32], line: &mut dyn FnMut((f32, f32), (f32, f32))| {
        for &t in f {
            line((w * t, 0.0), (w * t, h));
            line((0.0, h * t), (w, h * t));
        }
    };
    match kind {
        CropOverlay::None => {}
        CropOverlay::Thirds => fracs(&[1.0 / 3.0, 2.0 / 3.0], &mut line),
        CropOverlay::GoldenRatio => fracs(&[1.0 - 1.0 / PHI, 1.0 / PHI], &mut line),
        CropOverlay::Grid => {
            let step = grid.max(1.0);
            let (nx, ny) = ((w / step).round().max(2.0) as usize, (h / step).round().max(2.0) as usize);
            for i in 1..nx {
                let x = w * i as f32 / nx as f32;
                line((x, 0.0), (x, h));
            }
            for j in 1..ny {
                let y = h * j as f32 / ny as f32;
                line((0.0, y), (w, y));
            }
        }
        CropOverlay::Diagonal => {
            // 45° from each corner to the opposite long side.
            let s = w.min(h);
            if w >= h {
                line((0.0, 0.0), (s, s));
                line((w, 0.0), (w - s, s));
                line((0.0, h), (s, h - s));
                line((w, h), (w - s, h - s));
            } else {
                line((0.0, 0.0), (s, s));
                line((0.0, h), (s, h - s));
                line((w, 0.0), (w - s, s));
                line((w, h), (w - s, h - s));
            }
        }
        CropOverlay::Triangle => {
            // A diagonal and the perpendiculars to it from the other corners.
            let (a, b, c, d) = if orient & 1 == 0 { ((0.0, 0.0), (w, h), (w, 0.0), (0.0, h)) } else { ((w, 0.0), (0.0, h), (0.0, 0.0), (w, h)) };
            line(a, b);
            let foot = |q: (f32, f32)| {
                let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                let t = ((q.0 - a.0) * dx + (q.1 - a.1) * dy) / (dx * dx + dy * dy);
                (a.0 + dx * t, a.1 + dy * t)
            };
            line(c, foot(c));
            line(d, foot(d));
        }
        CropOverlay::GoldenSpiral => {
            // Built in a golden rectangle (landscape or portrait), then
            // stretched to the frame and mirrored per `orient`.
            let portrait = h > w;
            let (gw, gh) = if portrait { (1.0, PHI) } else { (PHI, 1.0) };
            let (mut x0, mut y0, mut x1, mut y1) = (0.0f32, 0.0f32, gw, gh);
            let mut spiral = Vec::new();
            let mut squares = Vec::new();
            for i in 0..12 {
                let step = if portrait { (i + 1) % 4 } else { i % 4 };
                let (cx, cy, a0, s) = match step {
                    0 => {
                        let s = y1 - y0;
                        squares.push(((x0 + s, y0), (x0 + s, y1)));
                        let r = (x0 + s, y1, std::f32::consts::PI, s);
                        x0 += s;
                        r
                    }
                    1 => {
                        let s = x1 - x0;
                        squares.push(((x0, y0 + s), (x1, y0 + s)));
                        let r = (x0, y0 + s, 1.5 * std::f32::consts::PI, s);
                        y0 += s;
                        r
                    }
                    2 => {
                        let s = y1 - y0;
                        squares.push(((x1 - s, y0), (x1 - s, y1)));
                        let r = (x1 - s, y0, 0.0, s);
                        x1 -= s;
                        r
                    }
                    _ => {
                        let s = x1 - x0;
                        squares.push(((x0, y1 - s), (x1, y1 - s)));
                        let r = (x1, y1 - s, 0.5 * std::f32::consts::PI, s);
                        y1 -= s;
                        r
                    }
                };
                for k in 0..=12 {
                    let a = a0 + std::f32::consts::FRAC_PI_2 * k as f32 / 12.0;
                    spiral.push((cx + s * a.cos(), cy + s * a.sin()));
                }
            }
            let map = |q: (f32, f32)| {
                let (mut u, mut v) = (q.0 / gw, q.1 / gh);
                if orient & 1 != 0 {
                    u = 1.0 - u;
                }
                if orient & 2 != 0 {
                    v = 1.0 - v;
                }
                (u * w, v * h)
            };
            out.push(spiral.into_iter().map(map).collect());
            for (a, b) in squares.into_iter().take(6) {
                out.push(vec![map(a), map(b)]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::annot::rotation_about;

    #[test]
    fn filters_reproduce_a_translation_and_keep_premultiplied() {
        let g = ops::procedural(300, 200);
        let m = [1.0, 0.0, 7.0, 0.0, 1.0, -3.0, 0.0, 0.0, 1.0];
        for interp in Interp::ALL {
            let o = warp_affine(&g, &m, 300, 200, interp);
            for (x, y) in [(10, 10), (150, 100), (290, 190)] {
                let a = px::unpack(o.pixel(x, y));
                let b = px::unpack(g.pixel(x - 7, y + 3));
                assert!(a.iter().zip(b).all(|(p, q)| (*p as i32 - q as i32).abs() <= 1), "{interp:?} {a:?} {b:?}");
                assert!(a[0] <= a[3] && a[1] <= a[3] && a[2] <= a[3]);
            }
        }
    }

    #[test]
    fn lanczos_is_sharper_than_bilinear_on_a_rotation() {
        // A fine checkerboard rotated 30° and back: Lanczos keeps more contrast.
        let g = build_grid(256, 256, |_, _| {
            let mut t = vec![0u32; TPX];
            for y in 0..TS {
                for x in 0..TS {
                    let v = if ((x / 3) + (y / 3)) % 2 == 0 { 230 } else { 25 };
                    t[y * TS + x] = px::pack(v, v, v, 255);
                }
            }
            finish_tile(t.into_boxed_slice())
        });
        let there = rotation_about(30.0, 128.0, 128.0);
        let back = rotation_about(-30.0, 128.0, 128.0);
        let err = |i: Interp| {
            let o = warp_affine(&warp_affine(&g, &there, 256, 256, i), &back, 256, 256, i);
            let mut e = 0.0;
            for y in 96..160 {
                for x in 96..160 {
                    e += (px::unpack(o.pixel(x, y))[0] as f32 - px::unpack(g.pixel(x, y))[0] as f32).abs();
                }
            }
            e / 4096.0
        };
        let (bl, bc, lz) = (err(Interp::Bilinear), err(Interp::Bicubic), err(Interp::Lanczos));
        assert!(lz < bc && bc < bl, "bilinear {bl} bicubic {bc} lanczos {lz}");
    }

    #[test]
    fn hidden_pixels_survive_a_crop_and_come_back() {
        let g = ops::procedural(400, 300);
        // Crop to (100, 50)–(300, 250) keeping the rest.
        let m = [1.0, 0.0, -100.0, 0.0, 1.0, -50.0, 0.0, 0.0, 1.0];
        let (c, hd) = transformed_keep(&g, None, &m, 200, 200, Interp::Bicubic);
        assert_eq!((c.w, c.h), (200, 200));
        assert_eq!(c.pixel(0, 0), g.pixel(100, 50));
        let hd = hd.expect("pixels outside the crop are kept");
        assert_eq!(hd.rect(), RectI::new(-100, -50, 300, 250));
        // Paint on the cropped canvas, then undo the crop by a translation:
        // the painted pixel and the hidden ones are both there.
        let mut c2 = c.clone();
        c2 = ops::fill(&c2, Some(&crate::engine::selection::Selection::rect(RectI::new(0, 0, 10, 10))), [1, 2, 3], 1.0);
        let back = [1.0, 0.0, 100.0, 0.0, 1.0, 50.0, 0.0, 0.0, 1.0];
        let (full, hd2) = transformed_keep(&c2, Some(&hd), &back, 400, 300, Interp::Bicubic);
        assert!(hd2.is_none());
        assert_eq!(full.pixel(5, 5), g.pixel(5, 5));
        assert_eq!(px::unpack(full.pixel(105, 55))[..3], [1, 2, 3]);
        assert_eq!(full.pixel(399, 299), g.pixel(399, 299));
        assert_eq!(extent(&c2, Some(&hd)), Some(RectI::new(-100, -50, 300, 250)));
    }

    #[test]
    fn masks_keep_their_hidden_part() {
        // A mask hiding the left 100 px; crop that strip away keeping pixels,
        // then bring it back: the mask still hides it, and area the mask never
        // covered reveals.
        let g = ops::procedural(400, 300);
        let hide_left = crate::engine::selection::Selection::rect(RectI::new(100, 0, 400, 300));
        let mask = MaskGrid::from_selection(&hide_left, 400, 300);
        let m = [1.0, 0.0, -150.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let k = transformed_keep_masked(&g, None, Some(&mask), &m, 250, 300, Interp::Bicubic);
        assert_eq!(k.mask.as_ref().unwrap().at(10, 10), 255);
        let hd = k.hidden.clone().unwrap();
        assert!(hd.mask.is_some());
        let back = [1.0, 0.0, 150.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let r = transformed_keep_masked(&k.grid, Some(&hd), k.mask.as_ref(), &back, 400, 300, Interp::Bicubic);
        let mk = r.mask.unwrap();
        assert_eq!((mk.at(50, 150), mk.at(250, 150)), (0, 255));
        assert_eq!(r.grid.pixel(50, 150), g.pixel(50, 150));
    }

    #[test]
    fn straighten_frame_fits_inside_the_rotated_image() {
        let (r, a) = straighten_frame(4000, 3000, 5.0);
        assert_eq!(a, -5.0);
        // All four frame corners, rotated by the frame angle, lie inside the image.
        let (cx, cy) = ((r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5);
        let m = rotation_about(a, cx, cy);
        for (x, y) in [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)] {
            let q = p(x as f32, y as f32).xform(&m);
            assert!(q.x >= -1.0 && q.y >= -1.0 && q.x <= 4001.0 && q.y <= 3001.0, "{q:?}");
        }
        assert!((r.w() as f32 / r.h() as f32 - 4.0 / 3.0).abs() < 0.01 && r.w() > 3500);
    }

    #[test]
    fn overlays_have_the_expected_geometry() {
        let l = overlay_lines(CropOverlay::GoldenRatio, 0, 1000.0, 500.0, 50.0);
        assert!(l.iter().any(|s| (s[0].0 - 618.0).abs() < 1.0));
        assert_eq!(overlay_lines(CropOverlay::Diagonal, 0, 1000.0, 500.0, 50.0)[0], vec![(0.0, 0.0), (500.0, 500.0)]);
        // Triangle: the perpendicular from (w, 0) meets the diagonal at a right angle.
        let t = overlay_lines(CropOverlay::Triangle, 0, 1000.0, 500.0, 50.0);
        let (a, b) = (t[1][0], t[1][1]);
        assert!(((b.0 - a.0) * 1000.0 + (b.1 - a.1) * 500.0).abs() < 1.0);
        // Golden spiral: starts at a frame corner and converges near the
        // golden-ratio point.
        let s = &overlay_lines(CropOverlay::GoldenSpiral, 0, 1618.0, 1000.0, 50.0)[0];
        assert!(s[0].0.abs() < 1.0 && (s[0].1 - 1000.0).abs() < 1.0);
        let e = s[s.len() - 1];
        assert!((e.0 - 1618.0 / PHI / PHI * 1.894).abs() < 120.0 && (e.1 - 1000.0 / PHI / PHI * 1.894).abs() < 120.0, "{e:?}");
        let g = overlay_lines(CropOverlay::Grid, 0, 1000.0, 500.0, 100.0);
        assert_eq!(g.len(), 9 + 4);
        // Orientation mirrors the spiral.
        let m = &overlay_lines(CropOverlay::GoldenSpiral, 1, 1618.0, 1000.0, 50.0)[0];
        assert!((m[0].0 - 1618.0).abs() < 1.0);
    }

    #[test]
    fn content_aware_fills_the_corners_a_rotation_adds() {
        let g = ops::procedural(512, 384);
        let m = rotation_about(8.0, 256.0, 192.0);
        let rot = warp_affine(&g, &m, 512, 384, Interp::Bicubic);
        let hole = new_area(&m, 512, 384, 512, 384);
        assert!(hole.at(0, 0) > 0 && hole.at(256, 192) == 0);
        let filled = content_aware(&rot, &hole);
        // The transparent corners are now opaque.
        for (x, y) in [(1, 1), (510, 1), (1, 382), (510, 382)] {
            assert_eq!(px::unpack(filled.pixel(x, y))[3], 255, "({x}, {y})");
        }
    }
}
