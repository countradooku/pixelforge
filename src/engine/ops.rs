//! Whole-layer operations. Each is a pure function from an input grid to a new
//! grid (sharing every tile it did not touch), so long ones can run on a
//! background thread against a snapshot while the UI stays live.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use super::filters::Filter;
use super::grid::{RectI, SyncPtr, TS, TS_I, Tile, TileGrid, TileRef, build_grid, finish_tile, tile_mut};
use super::document::Layer;
use super::px;
use super::selection::{MaskGrid, MaskTile, Selection, TileCoverage};

/// Progress counter shared with the UI for long jobs.
#[derive(Default)]
pub struct Progress {
    pub done: AtomicUsize,
    pub total: AtomicUsize,
}

impl Progress {
    pub fn fraction(&self) -> f32 {
        let t = self.total.load(Ordering::Relaxed);
        if t == 0 { 0.0 } else { self.done.load(Ordering::Relaxed) as f32 / t as f32 }
    }
}

fn coverage(sel: Option<&Selection>, tx: i32, ty: i32) -> TileCoverage {
    match sel {
        Some(s) => s.tile_coverage(tx, ty),
        None => TileCoverage::Full,
    }
}

/// Runs `f` on every tile of `grid` within `region` (respecting the
/// selection), in parallel, on a copy-on-write clone of the grid.
fn map_tiles<F>(grid: &TileGrid, sel: Option<&Selection>, region: RectI, f: F) -> TileGrid
where
    F: Fn(&mut TileRef, &TileCoverage, i32, i32) + Sync,
{
    let mut out = grid.clone();
    let (tx0, ty0, tx1, ty1) = region.tile_span(out.cols, out.rows);
    let cols = out.cols;
    let ptr = SyncPtr(out.tiles.as_mut_ptr());
    let idx: Vec<(i32, i32)> = (ty0..ty1).flat_map(|ty| (tx0..tx1).map(move |tx| (tx, ty))).collect();
    idx.par_iter().for_each(|&(tx, ty)| {
        let cov = coverage(sel, tx, ty);
        if matches!(cov, TileCoverage::Empty) {
            return;
        }
        // SAFETY: every (tx, ty) is visited exactly once.
        let slot = unsafe { ptr.at((ty as u32 * cols + tx as u32) as usize) };
        f(slot, &cov, tx, ty);
    });
    out
}

/// Fills the selection (or everything) with an opaque color at `opacity`.
pub fn fill(grid: &TileGrid, sel: Option<&Selection>, rgb: [u8; 3], opacity: f32) -> TileGrid {
    let c = px::pack(rgb[0], rgb[1], rgb[2], 255);
    let a = (opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let shared = Arc::new(Tile::solid(c));
    let region = sel.map_or(grid.bounds(), |s| s.effect_bounds(grid.bounds()));
    map_tiles(grid, sel, region, |slot, cov, _, _| {
        if a == 255 && matches!(cov, TileCoverage::Full) {
            *slot = Some(shared.clone());
            return;
        }
        if let (Some(t), TileCoverage::Full) = (slot.as_ref(), cov) {
            if let Some(u) = t.uniform {
                let v = px::lerp(u, c, a);
                *slot = Some(Arc::new(Tile::solid(v)));
                return;
            }
        }
        let t = tile_mut(slot);
        for (i, p) in t.write().iter_mut().enumerate() {
            let k = cov.at(i) * a / 255;
            if k > 0 {
                *p = px::lerp(*p, c, k);
            }
        }
    })
}

/// Erases the selection (or everything) to transparency.
/// The grid composited over a solid colour (the locked Background stays
/// opaque: transparency from a canvas change becomes the background colour).
pub fn matte(grid: &TileGrid, rgb: [u8; 3]) -> TileGrid {
    let solid = px::pack(rgb[0], rgb[1], rgb[2], 255);
    let mut out = grid.clone();
    out.tiles = grid
        .tiles
        .par_iter()
        .map(|t| match t {
            None => Some(Arc::new(Tile::solid(solid))),
            Some(t) if t.uniform.is_some_and(|c| px::alpha(c) == 255) => Some(t.clone()),
            Some(t) => finish_tile(t.read().iter().map(|&p| px::over(p, solid)).collect()),
        })
        .collect();
    out
}

pub fn clear(grid: &TileGrid, sel: Option<&Selection>) -> TileGrid {
    let region = sel.map_or(grid.bounds(), |s| s.effect_bounds(grid.bounds()));
    map_tiles(grid, sel, region, |slot, cov, _, _| {
        if slot.is_none() {
            return;
        }
        if matches!(cov, TileCoverage::Full) {
            *slot = None;
            return;
        }
        let t = tile_mut(slot);
        for (i, p) in t.write().iter_mut().enumerate() {
            let k = cov.at(i);
            if k > 0 {
                *p = px::scale(*p, 255 - k);
            }
        }
    })
}

/// Copies the selected pixels into a new grid (for copy / layer via copy).
pub fn copy_masked(grid: &TileGrid, sel: Option<&Selection>) -> TileGrid {
    let Some(sel) = sel else { return grid.clone() };
    let mut out = TileGrid::new(grid.w, grid.h);
    let cols = out.cols as usize;
    out.tiles.par_iter_mut().enumerate().for_each(|(i, slot)| {
        let (tx, ty) = ((i % cols) as i32, (i / cols) as i32);
        let Some(src) = grid.tiles[i].as_ref() else { return };
        match sel.tile_coverage(tx, ty) {
            TileCoverage::Empty => {}
            TileCoverage::Full => *slot = Some(src.clone()),
            TileCoverage::Partial(m) => {
                let px: Box<[u32]> = src.read().iter().zip(m.iter()).map(|(&p, &k)| px::scale(p, k as u32)).collect();
                *slot = finish_tile(px);
            }
        }
    });
    out
}

/// Separable Gaussian over a whole region with radius-independent cost:
/// three box passes along full-width row strips, then along full-height
/// column strips. Returns a grid holding the blurred pixels for `region`'s tiles.
fn blur_grid(src: &TileGrid, boxes: &[usize], region: RectI, progress: &Progress) -> TileGrid {
    let pad: i32 = boxes.iter().map(|b| (b / 2) as i32).sum();
    let (tx0, ty0, tx1, ty1) = region.tile_span(src.cols, src.rows);
    let reach = (pad + TS_I - 1) / TS_I;
    let (hy0, hy1) = ((ty0 - reach).max(0), (ty1 + reach).min(src.rows as i32));
    let x_lo = tx0 * TS_I;
    let x_hi = tx1 * TS_I;
    let strip_w = (x_hi - x_lo + 2 * pad) as usize;
    let total = ((hy1 - hy0) + (tx1 - tx0)).max(1) as usize;
    progress.total.store(total, Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);

    // Horizontal passes: one task per tile row.
    let mut hgrid = TileGrid::new(src.w, src.h);
    let rows: Vec<(i32, Vec<TileRef>)> = (hy0..hy1)
        .into_par_iter()
        .map(|ty| {
            let mut buf = vec![0u32; strip_w * TS];
            src.read_region_clamped(x_lo - pad, ty * TS_I, strip_w, TS, &mut buf);
            super::filters::fast::box_axis(&mut buf, strip_w, TS, boxes, true);
            let tiles = (tx0..tx1)
                .map(|tx| {
                    let ox = ((tx * TS_I - x_lo) + pad) as usize;
                    let mut out = vec![0u32; TS * TS].into_boxed_slice();
                    for y in 0..TS {
                        out[y * TS..(y + 1) * TS].copy_from_slice(&buf[y * strip_w + ox..y * strip_w + ox + TS]);
                    }
                    finish_tile(out)
                })
                .collect();
            progress.done.fetch_add(1, Ordering::Relaxed);
            (ty, tiles)
        })
        .collect();
    for (ty, tiles) in rows {
        for (i, t) in tiles.into_iter().enumerate() {
            let idx = hgrid.idx(tx0 + i as i32, ty);
            hgrid.tiles[idx] = t;
        }
    }

    // Vertical passes: one task per tile column, reading the horizontal result.
    let y_lo = ty0 * TS_I;
    let y_hi = ty1 * TS_I;
    let strip_h = (y_hi - y_lo + 2 * pad) as usize;
    let mut out = TileGrid::new(src.w, src.h);
    let cols: Vec<(i32, Vec<TileRef>)> = (tx0..tx1)
        .into_par_iter()
        .map(|tx| {
            let mut buf = vec![0u32; TS * strip_h];
            hgrid.read_region_clamped(tx * TS_I, y_lo - pad, TS, strip_h, &mut buf);
            super::filters::fast::box_axis(&mut buf, TS, strip_h, boxes, false);
            let tiles = (ty0..ty1)
                .map(|ty| {
                    let oy = ((ty * TS_I - y_lo) + pad) as usize;
                    finish_tile(buf[oy * TS..(oy + TS) * TS].to_vec().into_boxed_slice())
                })
                .collect();
            progress.done.fetch_add(1, Ordering::Relaxed);
            (tx, tiles)
        })
        .collect();
    for (tx, tiles) in cols {
        for (i, t) in tiles.into_iter().enumerate() {
            let idx = out.idx(tx, ty0 + i as i32);
            out.tiles[idx] = t;
        }
    }
    out
}

/// What a filter does to one tile.
pub enum TileJob {
    /// Unchanged: unselected, or empty under a pointwise filter.
    Keep,
    /// A flat tile in a flat neighborhood: the shared result tile.
    Flat(TileRef),
    /// Filter the tile, blending with the original by the coverage.
    Compute(TileCoverage),
}

/// Decides per tile what a full-resolution filter run must do. The CPU and
/// GPU paths share it, so both skip and share exactly the same tiles.
pub struct TilePlanner<'a> {
    grid: &'a TileGrid,
    filter: &'a Filter,
    sel: Option<&'a Selection>,
    pointwise: bool,
    flat_ok: bool,
    pad: usize,
    reach: i32,
    flat: std::sync::Mutex<std::collections::HashMap<u32, TileRef>>,
}

impl<'a> TilePlanner<'a> {
    pub fn new(grid: &'a TileGrid, filter: &'a Filter, sel: Option<&'a Selection>) -> Self {
        use super::filters::Class;
        let pad = filter.pad(0);
        TilePlanner {
            grid,
            filter,
            sel,
            pointwise: filter.is_pointwise(),
            flat_ok: !filter.position_dependent() && matches!(filter.class(), Class::Point | Class::Local),
            pad,
            reach: pad.div_ceil(TS) as i32,
            flat: Default::default(),
        }
    }

    fn flat_result(&self, u: u32) -> TileRef {
        if let Some(t) = self.flat.lock().unwrap().get(&u) {
            return t.clone();
        }
        let side = 2 * self.pad + 1;
        let mut b = vec![u; side * side];
        self.filter.apply(&mut b, side, side, 0, (0, 0));
        let c = b[self.pad * side + self.pad];
        let t = if c == 0 { None } else { Some(Arc::new(Tile::solid(c))) };
        self.flat.lock().unwrap().insert(u, t.clone());
        t
    }

    pub fn job(&self, tx: i32, ty: i32) -> TileJob {
        let grid = self.grid;
        let cov = match self.sel {
            Some(s) => s.tile_coverage(tx, ty),
            None => TileCoverage::Full,
        };
        if matches!(cov, TileCoverage::Empty) {
            return TileJob::Keep;
        }
        let slot = &grid.tiles[grid.idx(tx, ty)];
        if self.pointwise && slot.is_none() {
            return TileJob::Keep;
        }
        if self.flat_ok && matches!(cov, TileCoverage::Full) {
            let u = match slot {
                None => Some(0),
                Some(t) => t.uniform,
            };
            if let Some(u) = u {
                let reach = self.reach;
                let flat = reach == 0
                    || grid
                        .region_uniform(RectI::new(
                            (tx - reach).max(0) * TS_I,
                            (ty - reach).max(0) * TS_I,
                            (tx + reach + 1).min(grid.cols as i32) * TS_I,
                            (ty + reach + 1).min(grid.rows as i32) * TS_I,
                        ))
                        .is_some();
                if flat {
                    return TileJob::Flat(self.flat_result(u));
                }
            }
        }
        TileJob::Compute(cov)
    }
}

/// Applies any filter at full resolution, on the GPU when that's available
/// and faster, else on the CPU (see [`crate::engine::filters::apply`]).
pub fn apply_filter(grid: &TileGrid, filter: &Filter, sel: Option<&Selection>, progress: &Progress) -> TileGrid {
    super::filters::apply(grid, filter, sel, progress)
}

/// The CPU path of [`apply_filter`] (the reference), for tile rows
/// `rows.0..rows.1` of the region (all when `None`).
///
/// - Blur-derived filters with a large radius use radius-independent strips.
/// - Everything else renders in parallel blocks whose size grows with the
///   filter's padding (256 px tiles up to 2048 px blocks), so the padding
///   overhead stays small even for huge radii.
/// - Flat tiles in flat neighborhoods are computed once per color and shared;
///   empty tiles are skipped by point filters; unselected tiles are untouched.
/// Side, in tiles, of the blocks [`apply_filter_cpu`] renders a filter with
/// apron `pad` in (blocks start at the region's first tile row / column).
pub fn block_tiles(pad: usize) -> i32 {
    match pad {
        0..=32 => 1,
        33..=128 => 2,
        129..=384 => 4,
        _ => 8,
    }
}

pub fn apply_filter_cpu(grid: &TileGrid, filter: &Filter, sel: Option<&Selection>, progress: &Progress, rows: Option<(i32, i32)>) -> TileGrid {
    let img = grid.bounds();
    let mut region = sel.map_or(img, |s| s.effect_bounds(img));
    if let Some((r0, r1)) = rows {
        region = region.intersect(&RectI::new(img.x0, r0 * TS_I, img.x1, r1 * TS_I));
    }
    if region.is_empty() {
        return grid.clone();
    }
    if let Some(sb) = filter.strip_blur() {
        let pad: usize = sb.boxes.iter().map(|b| b / 2).sum();
        if pad > 160 {
            let blurred = blur_grid(grid, &sb.boxes, region, progress);
            let comb = filter.combiner();
            return map_tiles(grid, sel, region, |slot, cov, tx, ty| {
                let original = slot.as_ref().map(|t| t.read());
                let b = blurred.get(tx, ty).map(|t| t.read());
                let mut out = vec![0u32; TS * TS].into_boxed_slice();
                for (i, o) in out.iter_mut().enumerate() {
                    let orig = original.as_ref().map_or(0, |t| t[i]);
                    let bl = b.as_ref().map_or(0, |t| t[i]);
                    *o = px::lerp(orig, comb.apply(orig, bl), cov.at(i));
                }
                drop(original);
                *slot = finish_tile(out);
            });
        }
    }

    use super::filters::Class;
    let class = filter.class();
    let pad = filter.pad(0);
    let k = block_tiles(pad);
    let (tx0, ty0, tx1, ty1) = region.tile_span(grid.cols, grid.rows);
    progress.total.store(((tx1 - tx0) * (ty1 - ty0)).max(1) as usize, Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    let planner = TilePlanner::new(grid, filter, sel);

    // Gather filters sample prefiltered mips for long sample paths.
    let mips = if class == Class::Gather { super::mips::build_mips(grid) } else { Vec::new() };
    let src = super::filters::Src { levels: std::iter::once(grid).chain(mips.iter()).collect() };
    let blocks: Vec<(i32, i32)> = (ty0..ty1)
        .step_by(k as usize)
        .flat_map(|by| (tx0..tx1).step_by(k as usize).map(move |bx| (bx, by)))
        .collect();
    let results: Vec<Vec<(usize, TileRef)>> = blocks
        .par_iter()
        .map_init(Vec::new, |scratch, &(bx, by)| {
            let mut done = Vec::new();
            let mut todo: Vec<(i32, i32, TileCoverage)> = Vec::new();
            for ty in by..(by + k).min(ty1) {
                for tx in bx..(bx + k).min(tx1) {
                    progress.done.fetch_add(1, Ordering::Relaxed);
                    match planner.job(tx, ty) {
                        TileJob::Keep => {}
                        TileJob::Flat(t) => done.push((grid.idx(tx, ty), t)),
                        TileJob::Compute(cov) => todo.push((tx, ty, cov)),
                    }
                }
            }
            if todo.is_empty() {
                return done;
            }
            let (rx0, ry0) = (todo.iter().map(|t| t.0).min().unwrap(), todo.iter().map(|t| t.1).min().unwrap());
            let (rx1, ry1) = (todo.iter().map(|t| t.0).max().unwrap() + 1, todo.iter().map(|t| t.1).max().unwrap() + 1);
            let (bw, bh) = (((rx1 - rx0) * TS_I) as usize, ((ry1 - ry0) * TS_I) as usize);
            let mut buf = vec![0u32; bw * bh];
            filter.render(&src, 0, rx0 * TS_I, ry0 * TS_I, bw, bh, &mut buf, scratch);
            for (tx, ty, cov) in todo {
                let idx = grid.idx(tx, ty);
                let orig = grid.tiles[idx].as_ref().map(|t| t.read());
                let (ox, oy) = (((tx - rx0) * TS_I) as usize, ((ty - ry0) * TS_I) as usize);
                let mut out = vec![0u32; TS * TS].into_boxed_slice();
                for y in 0..TS {
                    let src = &buf[(oy + y) * bw + ox..(oy + y) * bw + ox + TS];
                    let dst = &mut out[y * TS..(y + 1) * TS];
                    if matches!(cov, TileCoverage::Full) {
                        dst.copy_from_slice(src);
                    } else {
                        for x in 0..TS {
                            let i = y * TS + x;
                            dst[x] = px::lerp(orig.as_ref().map_or(0, |t| t[i]), src[x], cov.at(i));
                        }
                    }
                }
                done.push((idx, finish_tile(out)));
            }
            done
        })
        .collect();
    let mut out = grid.clone();
    for (idx, t) in results.into_iter().flatten() {
        out.tiles[idx] = t;
    }
    out
}

#[derive(Clone, Copy)]
enum Xform {
    Copy,
    FlipH,
    FlipV,
    RotCw,
    RotCcw,
}

/// Generic tile remapper: destination tile `(tx, ty)` reads the 256x256
/// source block at `src_origin(tx, ty)` and transforms it.
fn remap(grid: &TileGrid, nw: u32, nh: u32, xf: Xform, src_origin: impl Fn(i32, i32) -> (i32, i32) + Sync) -> TileGrid {
    build_grid(nw, nh, |tx, ty| {
        let (sx, sy) = src_origin(tx, ty);
        let r = RectI::wh(sx, sy, TS_I, TS_I);
        if let Some(shared) = grid.region_uniform(r) {
            return shared;
        }
        if matches!(xf, Xform::Copy) {
            if sx % TS_I == 0 && sy % TS_I == 0 {
                return grid.get(sx / TS_I, sy / TS_I).cloned();
            }
            let mut out = vec![0u32; TS * TS].into_boxed_slice();
            grid.read_region(sx, sy, TS, TS, &mut out);
            return finish_tile(out);
        }
        // Full aligned source tiles can be read in place. Edge tiles still
        // use read_region so pixels outside the document remain transparent.
        let pinned = if sx >= 0 && sy >= 0 && sx % TS_I == 0 && sy % TS_I == 0
            && sx + TS_I <= grid.w as i32 && sy + TS_I <= grid.h as i32 {
            grid.get(sx / TS_I, sy / TS_I).map(|t| t.read())
        } else { None };
        let mut scratch = Vec::new();
        let b: &[u32] = if let Some(p) = &pinned { p } else {
            scratch.resize(TS * TS, 0);
            grid.read_region(sx, sy, TS, TS, &mut scratch);
            &scratch
        };
        let mut out = vec![0u32; TS * TS].into_boxed_slice();
        if matches!(xf, Xform::RotCw | Xform::RotCcw) {
            // A small transpose block keeps both the source columns and output
            // rows in L1 instead of walking a full tile once per column.
            for by in (0..TS).step_by(16) {
                for bx in (0..TS).step_by(16) {
                    for dy in by..by + 16 {
                        for dx in bx..bx + 16 {
                            out[dy * TS + dx] = if matches!(xf, Xform::RotCw) {
                                b[(255 - dx) * TS + dy]
                            } else { b[dx * TS + 255 - dy] };
                        }
                    }
                }
            }
        } else {
            for dy in 0..TS {
                for dx in 0..TS {
                    out[dy * TS + dx] = match xf {
                        Xform::FlipH => b[dy * TS + 255 - dx],
                        Xform::FlipV => b[(255 - dy) * TS + dx],
                        _ => b[dy * TS + dx],
                    };
                }
            }
        }
        finish_tile(out)
    })
}

pub fn flip_h(g: &TileGrid) -> TileGrid {
    let w = g.w as i32;
    remap(g, g.w, g.h, Xform::FlipH, |tx, ty| (w - tx * TS_I - TS_I, ty * TS_I))
}

pub fn flip_v(g: &TileGrid) -> TileGrid {
    let h = g.h as i32;
    remap(g, g.w, g.h, Xform::FlipV, |tx, ty| (tx * TS_I, h - ty * TS_I - TS_I))
}

pub fn rotate_cw(g: &TileGrid) -> TileGrid {
    let h = g.h as i32;
    remap(g, g.h, g.w, Xform::RotCw, |tx, ty| (ty * TS_I, h - TS_I - tx * TS_I))
}

pub fn rotate_ccw(g: &TileGrid) -> TileGrid {
    let w = g.w as i32;
    remap(g, g.h, g.w, Xform::RotCcw, |tx, ty| (w - TS_I - ty * TS_I, tx * TS_I))
}

pub fn crop(g: &TileGrid, r: RectI) -> TileGrid {
    remap(g, r.w() as u32, r.h() as u32, Xform::Copy, |tx, ty| (r.x0 + tx * TS_I, r.y0 + ty * TS_I))
}

/// Shifts pixels by `(dx, dy)`; tile-aligned shifts just move pointers.
pub fn translate(g: &TileGrid, dx: i32, dy: i32) -> TileGrid {
    remap(g, g.w, g.h, Xform::Copy, |tx, ty| (tx * TS_I - dx, ty * TS_I - dy))
}

/// Resamples a layer to `nw x nh` with bilinear filtering from the mip level
/// closest to the scale factor (so large downscales stay alias-free).
pub fn resize(layer: &Layer, nw: u32, nh: u32) -> TileGrid {
    resize_with(layer, nw, nh, super::resample::Resample::Bilinear, 0.0)
}

/// Image Size with a resampling method (see [`super::resample`]); Reduce
/// Noise `noise` (0..=100) applies to the Preserve Details methods. Runs on
/// the GPU when that's on and wins, with identical pixels.
pub fn resize_with(layer: &Layer, nw: u32, nh: u32, method: super::resample::Resample, noise: f32) -> TileGrid {
    let g = &layer.grid;
    let op = format!("Image Size ({})", method.resolve(g.w, g.h, nw, nh).label());
    let px = nw as u64 * nh as u64;
    let bytes = 4 * (px + g.w as u64 * g.h as u64);
    super::gpu::resample::switch(&op, px, bytes, |gpu| super::gpu::resample::resize(gpu, layer, nw, nh, method, noise), || {
        super::resample::resize_cpu(layer, nw, nh, method, noise)
    })
}

/// [`resize`] on the CPU (the reference).
pub fn resize_bilinear_cpu(layer: &Layer, nw: u32, nh: u32) -> TileGrid {
    let g0 = &layer.grid;
    let sx = g0.w as f32 / nw as f32;
    let sy = g0.h as f32 / nh as f32;
    let s = sx.min(sy);
    let level = if s >= 2.0 { (s.log2().floor() as u32).min(layer.mips.len() as u32) } else { 0 };
    let g = layer.level(level);
    let f = (1u32 << level) as f32;
    let (fx, fy) = (sx / f, sy / f);
    build_grid(nw, nh, |tx, ty| {
        let x0 = tx * TS_I;
        let y0 = ty * TS_I;
        let u0 = ((x0 as f32 + 0.5) * fx - 0.5).floor() as i32 - 1;
        let v0 = ((y0 as f32 + 0.5) * fy - 0.5).floor() as i32 - 1;
        let u1 = ((x0 as f32 + TS as f32 + 0.5) * fx - 0.5).ceil() as i32 + 2;
        let v1 = ((y0 as f32 + TS as f32 + 0.5) * fy - 0.5).ceil() as i32 + 2;
        let (bw, bh) = ((u1 - u0) as usize, (v1 - v0) as usize);
        if let Some(shared) = g.region_uniform(RectI::new(u0, v0, u1, v1).intersect(&g.bounds())) {
            if shared.is_none() {
                return None;
            }
        }
        let mut b = vec![0u32; bw * bh];
        g.read_region_clamped(u0, v0, bw, bh, &mut b);
        let mut out = vec![0u32; TS * TS].into_boxed_slice();
        for y in 0..TS {
            let v = (y0 as f32 + y as f32 + 0.5) * fy - 0.5 - v0 as f32;
            let vi = (v.floor() as usize).min(bh - 2);
            let wy = ((v - vi as f32).clamp(0.0, 1.0) * 255.0) as u32;
            for x in 0..TS {
                let u = (x0 as f32 + x as f32 + 0.5) * fx - 0.5 - u0 as f32;
                let ui = (u.floor() as usize).min(bw - 2);
                let wx = ((u - ui as f32).clamp(0.0, 1.0) * 255.0) as u32;
                let top = px::lerp(b[vi * bw + ui], b[vi * bw + ui + 1], wx);
                let bot = px::lerp(b[(vi + 1) * bw + ui], b[(vi + 1) * bw + ui + 1], wx);
                out[y * TS + x] = px::lerp(top, bot, wy);
            }
        }
        finish_tile(out)
    })
}

// ----- flood fill / magic wand ------------------------------------------------

#[inline]
fn close(a: u32, b: u32, tol: i32) -> bool {
    let a = px::unpack(a);
    let b = px::unpack(b);
    (0..4).all(|i| (a[i] as i32 - b[i] as i32).abs() <= tol)
}

/// Where the flood fill reads pixels: a layer grid, or the composite of all
/// layers computed lazily per tile ("Sample All Layers").
pub trait PixelSource: Sync {
    fn size(&self) -> (u32, u32);
    /// `Some(c)` when tile `i` is one flat color (transparent = 0).
    fn tile_uniform(&self, i: usize) -> Option<u32>;
    fn pixel(&self, x: i32, y: i32) -> u32;
    fn tile(&self, i: usize) -> TileRef;
}

impl PixelSource for TileGrid {
    fn size(&self) -> (u32, u32) {
        (self.w, self.h)
    }
    fn tile_uniform(&self, i: usize) -> Option<u32> {
        self.tiles[i].as_ref().map_or(Some(0), |t| t.uniform)
    }
    fn pixel(&self, x: i32, y: i32) -> u32 {
        TileGrid::pixel(self, x, y)
    }
    fn tile(&self, i: usize) -> TileRef {
        self.tiles[i].clone()
    }
}

/// The merged image, composited one tile at a time on first access.
pub struct CompositeSource<'a> {
    layers: &'a [Layer],
    w: u32,
    h: u32,
    cols: u32,
    cache: Vec<std::sync::OnceLock<TileRef>>,
    plain: bool,
}

impl<'a> CompositeSource<'a> {
    pub fn new(layers: &'a [Layer], w: u32, h: u32) -> Self {
        let g = TileGrid::new(w, h);
        CompositeSource { layers, w, h, cols: g.cols, cache: (0..g.tiles.len()).map(|_| Default::default()).collect(), plain: layers.iter().all(Layer::is_plain) }
    }

    fn get(&self, i: usize) -> &TileRef {
        self.cache[i].get_or_init(|| {
            let (tx, ty) = ((i as u32 % self.cols) as i32, (i as u32 / self.cols) as i32);
            let input = super::composite::ComposeInput { layers: self.layers, preview: None, extra: None };
            let mut s = super::composite::Scratch::default();
            let mut out = vec![0u32; TS * TS].into_boxed_slice();
            super::composite::compose_region(&input, 0, tx * TS_I, ty * TS_I, TS, TS, &mut out, &mut s);
            finish_tile(out)
        })
    }
}

impl PixelSource for CompositeSource<'_> {
    fn size(&self) -> (u32, u32) {
        (self.w, self.h)
    }
    fn tile_uniform(&self, i: usize) -> Option<u32> {
        // Flat when every visible plain layer is flat there (one pixel composite).
        if self.plain {
            let flat = self.layers.iter().filter(|l| l.visible && l.opacity > 0.0).all(|l| l.offset == (0, 0) && l.grid.tiles[i].as_ref().is_none_or(|t| t.uniform.is_some()));
            if flat {
                let (tx, ty) = ((i as u32 % self.cols) as i32, (i as u32 / self.cols) as i32);
                return Some(super::composite::sample(self.layers, tx * TS_I, ty * TS_I));
            }
        }
        if let Some(t) = self.cache[i].get() {
            return t.as_ref().map_or(Some(0), |t| t.uniform);
        }
        None
    }
    fn pixel(&self, x: i32, y: i32) -> u32 {
        if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
            return 0;
        }
        let i = ((y >> 8) as u32 * self.cols + (x >> 8) as u32) as usize;
        match self.get(i) {
            Some(t) => t.read()[((y & 255) as usize) * TS + (x & 255) as usize],
            None => 0,
        }
    }
    fn tile(&self, i: usize) -> TileRef {
        self.get(i).clone()
    }
}

/// Pixels matching the color at `(x, y)` within `tol`, either the connected
/// region (scanline fill) or every match in the image (parallel per tile).
///
/// Stays on the CPU (no GPU path): a contiguous fill is a serial frontier
/// walk whose extent is unknown until it finishes, so a GPU version would
/// iterate whole-image passes until nothing changes; the scanline fill
/// visits each pixel once, and the tile-parallel global match is already
/// bandwidth-bound on the CPU.
pub fn flood_mask(grid: &TileGrid, x: i32, y: i32, tol: u8, contiguous: bool) -> MaskGrid {
    flood_mask_on(grid, x, y, tol, contiguous)
}

pub fn flood_mask_on<S: PixelSource>(src: &S, x: i32, y: i32, tol: u8, contiguous: bool) -> MaskGrid {
    let (w, h) = src.size();
    let (wi, hi) = (w as i32, h as i32);
    let mut m = MaskGrid::new(w, h);
    if x < 0 || y < 0 || x >= wi || y >= hi {
        return m;
    }
    let seed = src.pixel(x, y);
    let tol = tol as i32;
    let cols = m.cols as usize;
    // The entire grid matches: avoid constructing and walking a tile frontier.
    if (0..m.tiles.len()).all(|i| src.tile_uniform(i).is_some_and(|c| close(c, seed, tol))) {
        m.tiles.fill(MaskTile::Full);
        return m;
    }
    if !contiguous {
        m.tiles.par_iter_mut().enumerate().for_each(|(i, mt)| {
            if let Some(u) = src.tile_uniform(i) {
                *mt = if close(u, seed, tol) { MaskTile::Full } else { MaskTile::Empty };
                return;
            }
            let _ = cols;
            match src.tile(i) {
                None => *mt = if close(0, seed, tol) { MaskTile::Full } else { MaskTile::Empty },
                Some(t) => {
                    let v: Vec<u8> = t.read().iter().map(|&p| if close(p, seed, tol) { 255 } else { 0 }).collect();
                    *mt = super::masks::normalize(v);
                }
            }
        });
        return m;
    }
    flood_parallel(src, x, y, seed, tol)
}

/// Per-tile state of the parallel flood.
#[derive(Default)]
struct FillTile {
    full: bool,
    filled: Option<Box<[u8]>>,
    matches: Option<Box<[u8]>>,
}

/// Tile-parallel contiguous fill. Work proceeds in waves: every tile with
/// pending seed pixels runs a local scanline fill (over a per-tile match
/// bitmap computed in one vectorizable pass), and pixels it fills on its
/// border become seeds of the neighbouring tiles for the next wave. Uniform
/// matching tiles are flooded whole without touching pixels.
fn flood_parallel<S: PixelSource>(src: &S, sx: i32, sy: i32, seed: u32, tol: i32) -> MaskGrid {
    let (w, h) = src.size();
    let mut m = MaskGrid::new(w, h);
    let (cols, rows) = (m.cols as i32, m.rows as i32);
    let n = (cols * rows) as usize;
    let class: Vec<TileClass> = (0..n)
        .into_par_iter()
        .map(|i| match src.tile_uniform(i) {
            Some(u) if close(u, seed, tol) => TileClass::Match,
            Some(_) => TileClass::NoMatch,
            None => TileClass::Mixed,
        })
        .collect();
    let mut tiles: Vec<FillTile> = (0..n).map(|_| FillTile::default()).collect();
    let mut pending: std::collections::HashMap<usize, Vec<u16>> = Default::default();
    pending.entry(((sy >> 8) * cols + (sx >> 8)) as usize).or_default().push((((sy & 255) as u16) << 8) | (sx & 255) as u16);
    let tile_w = |tx: i32| ((w as i32 - tx * TS_I).min(TS_I)) as usize;
    let tile_h = |ty: i32| ((h as i32 - ty * TS_I).min(TS_I)) as usize;
    while !pending.is_empty() {
        // 1. Uniform matching tiles: a sequential tile-level BFS (no pixels).
        let mut queue: Vec<usize> = pending.keys().copied().filter(|&ti| class[ti] == TileClass::Match).collect();
        for ti in &queue {
            pending.remove(ti);
        }
        while let Some(ti) = queue.pop() {
            if tiles[ti].full {
                continue;
            }
            tiles[ti].full = true;
            let (tx, ty) = ((ti as i32) % cols, (ti as i32) / cols);
            let (tw, th) = (tile_w(tx), tile_h(ty));
            let mut nb = |n: usize, edge: &mut dyn Iterator<Item = u16>| match class[n] {
                TileClass::NoMatch => {}
                TileClass::Match => {
                    if !tiles[n].full {
                        queue.push(n);
                    }
                }
                TileClass::Mixed => pending.entry(n).or_default().extend(edge),
            };
            if tx > 0 {
                nb(ti - 1, &mut (0..th).map(|y| ((y as u16) << 8) | 255));
            }
            if tx + 1 < cols && tw == TS {
                nb(ti + 1, &mut (0..th).map(|y| (y as u16) << 8));
            }
            if ty > 0 {
                nb(ti - cols as usize, &mut (0..tw).map(|x| (255u16 << 8) | x as u16));
            }
            if ty + 1 < rows && th == TS {
                nb(ti + cols as usize, &mut (0..tw).map(|x| x as u16));
            }
        }
        // 2. Mixed tiles with seeds, in parallel.
        let batch: Vec<(usize, Vec<u16>)> = pending.drain().collect();
        if batch.is_empty() {
            break;
        }
        let ptr = SyncPtr(tiles.as_mut_ptr());
        let emitted: Vec<Vec<(usize, u16)>> = batch
            .par_iter()
            .map(|(ti, seeds)| {
                let ti = *ti;
                // SAFETY: each tile index appears once per wave.
                let st = unsafe { ptr.at(ti) };
                let (tx, ty) = ((ti as i32) % cols, (ti as i32) / cols);
                let (tw, th) = (tile_w(tx), tile_h(ty));
                let mut out = Vec::new();
                // Seeds for neighbours: uniform matching neighbours need one
                // seed (they flood whole); mixed ones get the filled edge pixels.
                let mut emit_edges = |filled_edge: &dyn Fn(usize, usize) -> bool| {
                    let mut side = |nb: usize, pts: &mut dyn Iterator<Item = u16>| match class[nb] {
                        TileClass::NoMatch => {}
                        TileClass::Match => {
                            if pts.next().is_some() {
                                out.push((nb, 0));
                            }
                        }
                        TileClass::Mixed => out.extend(pts.map(|p| (nb, p))),
                    };
                    if tx > 0 {
                        side(ti - 1, &mut (0..th).filter(|&y| filled_edge(0, y)).map(|y| ((y as u16) << 8) | 255));
                    }
                    if tx + 1 < cols && tw == TS {
                        side(ti + 1, &mut (0..th).filter(|&y| filled_edge(TS - 1, y)).map(|y| (y as u16) << 8));
                    }
                    if ty > 0 {
                        side(ti - cols as usize, &mut (0..tw).filter(|&x| filled_edge(x, 0)).map(|x| (255u16 << 8) | x as u16));
                    }
                    if ty + 1 < rows && th == TS {
                        side(ti + cols as usize, &mut (0..tw).filter(|&x| filled_edge(x, TS - 1)).map(|x| x as u16));
                    }
                };
                match class[ti] {
                    TileClass::NoMatch => {}
                    TileClass::Match => {
                        if !st.full {
                            st.full = true;
                            emit_edges(&|_, _| true);
                        }
                    }
                    TileClass::Mixed => {
                        let tile = src.tile(ti);
                        let matches = st.matches.get_or_insert_with(|| {
                            let mut mm = vec![0u8; TS * TS].into_boxed_slice();
                            match &tile {
                                None => {
                                    let v = if close(0, seed, tol) { 1 } else { 0 };
                                    for y in 0..th {
                                        mm[y * TS..y * TS + tw].fill(v);
                                    }
                                }
                                Some(t) => {
                                    let pin = t.read();
                                    let pixels: &[u32] = &pin;
                                    for y in 0..th {
                                        let row = &pixels[y * TS..y * TS + tw];
                                        for (d, &p) in mm[y * TS..y * TS + tw].iter_mut().zip(row) {
                                            *d = close(p, seed, tol) as u8;
                                        }
                                    }
                                }
                            }
                            mm
                        });
                        let filled = st.filled.get_or_insert_with(|| vec![0u8; TS * TS].into_boxed_slice());
                        let mut newly = vec![0u8; 0];
                        let mut stack: Vec<(usize, usize)> = seeds.iter().map(|&s| ((s & 255) as usize, (s >> 8) as usize)).collect();
                        let mut touched_edge = false;
                        while let Some((x, y)) = stack.pop() {
                            let i = y * TS + x;
                            if x >= tw || y >= th || matches[i] == 0 || filled[i] != 0 {
                                continue;
                            }
                            let mut l = x;
                            while l > 0 && matches[y * TS + l - 1] != 0 && filled[y * TS + l - 1] == 0 {
                                l -= 1;
                            }
                            let mut r = x;
                            while r + 1 < tw && matches[y * TS + r + 1] != 0 && filled[y * TS + r + 1] == 0 {
                                r += 1;
                            }
                            filled[y * TS + l..=y * TS + r].fill(255);
                            if l == 0 || r == tw - 1 || y == 0 || y == th - 1 {
                                touched_edge = true;
                            }
                            for ny in [y.wrapping_sub(1), y + 1] {
                                if ny >= th {
                                    continue;
                                }
                                let mut i = l;
                                while i <= r {
                                    if matches[ny * TS + i] != 0 && filled[ny * TS + i] == 0 {
                                        stack.push((i, ny));
                                        while i <= r && matches[ny * TS + i] != 0 && filled[ny * TS + i] == 0 {
                                            i += 1;
                                        }
                                    } else {
                                        i += 1;
                                    }
                                }
                            }
                        }
                        let _ = &mut newly;
                        if touched_edge {
                            let f: &[u8] = filled;
                            emit_edges(&|x, y| f[y * TS + x] != 0);
                        }
                    }
                }
                out
            })
            .collect();
        for (ti, local) in emitted.into_iter().flatten() {
            // Skip seeds into tiles that cannot take them.
            match class[ti] {
                TileClass::NoMatch => continue,
                TileClass::Match if tiles[ti].full => continue,
                TileClass::Mixed => {
                    if let Some(f) = &tiles[ti].filled {
                        if f[((local >> 8) as usize) * TS + (local & 255) as usize] != 0 {
                            continue;
                        }
                    }
                }
                _ => {}
            }
            pending.entry(ti).or_default().push(local);
        }
    }
    for (mt, st) in m.tiles.iter_mut().zip(tiles) {
        *mt = if st.full {
            MaskTile::Full
        } else {
            match st.filled {
                Some(f) => super::masks::normalize(f.into_vec()),
                None => MaskTile::Empty,
            }
        };
    }
    m
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TileClass {
    Match,
    NoMatch,
    Mixed,
}

/// Paints `rgb` through a coverage mask (paint bucket).
pub fn fill_mask(grid: &TileGrid, mask: &MaskGrid, sel: Option<&Selection>, rgb: [u8; 3], opacity: f32) -> TileGrid {
    let c = px::pack(rgb[0], rgb[1], rgb[2], 255);
    let a = (opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let shared = Arc::new(Tile::solid(c));
    map_tiles(grid, sel, grid.bounds(), |slot, cov, tx, ty| {
        let mt = &mask.tiles[(ty as u32 * mask.cols + tx as u32) as usize];
        match mt {
            MaskTile::Empty => {}
            MaskTile::Full if a == 255 && matches!(cov, TileCoverage::Full) => *slot = Some(shared.clone()),
            _ => {
                let t = tile_mut(slot);
                for (i, p) in t.write().iter_mut().enumerate() {
                    let mk = match mt {
                        MaskTile::Full => 255,
                        MaskTile::Partial(m) => m[i] as u32,
                        MaskTile::Empty => 0,
                    };
                    let k = mk * cov.at(i) / 255 * a / 255;
                    if k > 0 {
                        *p = px::lerp(*p, c, k);
                    }
                }
            }
        }
    })
}

/// Linear or radial gradient from `c0` at `p0` to `c1` at `p1`.
#[allow(clippy::too_many_arguments)]
pub fn gradient(
    grid: &TileGrid,
    sel: Option<&Selection>,
    p0: (f32, f32),
    p1: (f32, f32),
    c0: [u8; 3],
    c1: [u8; 3],
    radial: bool,
    opacity: f32,
) -> TileGrid {
    let (dx, dy) = (p1.0 - p0.0, p1.1 - p0.1);
    let len2 = (dx * dx + dy * dy).max(1e-6);
    let len = len2.sqrt();
    let a = (opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let img = grid.bounds();
    let region = sel.map_or(img, |s| s.effect_bounds(img));
    map_tiles(grid, sel, region, |slot, cov, tx, ty| {
        let t = tile_mut(slot);
        let mut pixels = t.write();
        for y in 0..TS {
            let gy = (ty * TS_I + y as i32) as f32 + 0.5;
            for x in 0..TS {
                let i = y * TS + x;
                let k = cov.at(i) * a / 255;
                if k == 0 {
                    continue;
                }
                let gx = (tx * TS_I + x as i32) as f32 + 0.5;
                let t01 = if radial {
                    (((gx - p0.0).powi(2) + (gy - p0.1).powi(2)).sqrt() / len).min(1.0)
                } else {
                    (((gx - p0.0) * dx + (gy - p0.1) * dy) / len2).clamp(0.0, 1.0)
                };
                // 4x4 ordered dither kills banding on long, subtle ramps.
                let bayer = [0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0][(y & 3) * 4 + (x & 3)];
                let d = (bayer + 0.5) / 16.0;
                let ch = |i: usize| (c0[i] as f32 + (c1[i] as f32 - c0[i] as f32) * t01 + d).clamp(0.0, 255.0) as u8;
                let c = px::pack(ch(0), ch(1), ch(2), 255);
                pixels[i] = px::lerp(pixels[i], c, k);
            }
        }
    })
}

/// A procedural test image, generated in parallel (benchmarking huge canvases).
pub fn procedural(w: u32, h: u32) -> TileGrid {
    build_grid(w, h, |tx, ty| {
        let mut out = vec![0u32; TS * TS].into_boxed_slice();
        let (fw, fh) = (w as f32, h as f32);
        for y in 0..TS {
            let gy = (ty * TS_I + y as i32) as f32;
            for x in 0..TS {
                let gx = (tx * TS_I + x as i32) as f32;
                let u = gx / fw;
                let v = gy / fh;
                let rings = ((((u - 0.5).powi(2) + (v - 0.5).powi(2)).sqrt() * 120.0).sin() * 0.5 + 0.5) * 60.0;
                let grid_line = if (gx as i32 % 512) < 3 || (gy as i32 % 512) < 3 { 80.0 } else { 0.0 };
                let r = (u * 200.0 + rings + grid_line).min(255.0) as u8;
                let g = (v * 200.0 + rings * 0.5 + grid_line).min(255.0) as u8;
                let b = ((1.0 - u) * 180.0 + (gx * 0.05).sin() * 30.0 + 40.0 + grid_line).clamp(0.0, 255.0) as u8;
                out[y * TS + x] = px::pack(r, g, b, 255);
            }
        }
        Some(Arc::new(Tile::from_raw(out)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flip_twice_is_identity() {
        let g = procedural(700, 500);
        let f = flip_h(&flip_h(&g));
        for &(x, y) in &[(0, 0), (699, 499), (300, 260), (511, 3)] {
            assert_eq!(f.pixel(x, y), g.pixel(x, y));
        }
        let r = rotate_ccw(&rotate_cw(&g));
        assert_eq!((r.w, r.h), (700, 500));
        for &(x, y) in &[(0, 0), (699, 499), (300, 260)] {
            assert_eq!(r.pixel(x, y), g.pixel(x, y));
        }
        let fv = flip_v(&g);
        assert_eq!(fv.pixel(10, 0), g.pixel(10, 499));
        let cw = rotate_cw(&g);
        assert_eq!(cw.pixel(499, 0), g.pixel(0, 0));
    }

    #[test]
    fn contiguous_fill_stops_at_boundary() {
        let mut g = TileGrid::filled(600, 300, px::pack(255, 255, 255, 255));
        // Vertical wall at x = 300.
        for y in 0..300 {
            let (tx, ty) = (300 >> 8, y >> 8);
            let i = g.idx(tx, ty);
            tile_mut(&mut g.tiles[i]).write()[((y & 255) as usize) * TS + (300 & 255)] = px::pack(0, 0, 0, 255);
        }
        let m = flood_mask(&g, 10, 10, 10, true);
        assert_eq!(m.at(299, 100), 255);
        assert_eq!(m.at(300, 100), 0);
        assert_eq!(m.at(301, 100), 0);
    }

    #[test]
    fn hierarchical_fill_crosses_mixed_tiles() {
        // White canvas with a black ring of mixed tiles; the fill must flood the
        // uniform inside, stop at the ring, and leave the outside untouched.
        let mut g = TileGrid::filled(2048, 2048, px::pack(255, 255, 255, 255));
        for y in 0..2048i32 {
            for x in 0..2048i32 {
                let d = (((x - 1024) * (x - 1024) + (y - 1024) * (y - 1024)) as f32).sqrt();
                if (d - 700.0).abs() < 3.0 {
                    let i = g.idx(x >> 8, y >> 8);
                    tile_mut(&mut g.tiles[i]).write()[((y & 255) as usize) * TS + (x & 255) as usize] = px::pack(0, 0, 0, 255);
                }
            }
        }
        for t in g.tiles.iter_mut().flatten() {
            Arc::make_mut(t).detect_uniform();
        }
        let m = flood_mask(&g, 1024, 1024, 10, true);
        assert_eq!(m.at(1024, 1024), 255);
        assert_eq!(m.at(1024 + 690, 1024), 255);
        assert_eq!(m.at(1024 + 700, 1024), 0);
        assert_eq!(m.at(5, 5), 0);
        let outside = flood_mask(&g, 5, 5, 10, true);
        assert_eq!(outside.at(2040, 2040), 255);
        assert_eq!(outside.at(1024, 1024), 0);
    }

    #[test]
    fn strip_blur_matches_reference() {
        let g = procedural(700, 520);
        let p = Progress::default();
        let out = apply_filter(&g, &Filter::GaussianBlur { radius: 9.0 }, None, &p);
        // Reference: blur the edge-extended image in one buffer.
        let f = Filter::GaussianBlur { radius: 9.0 };
        let pad = f.pad(0) as i32;
        let (fw, fh) = (700 + 2 * pad as usize, 520 + 2 * pad as usize);
        let mut full = vec![0u32; fw * fh];
        g.read_region_clamped(-pad, -pad, fw, fh, &mut full);
        f.apply(&mut full, fw, fh, 0, (0, 0));
        for &(x, y) in &[(0, 0), (350, 260), (255, 256), (699, 519), (512, 100)] {
            let a = px::unpack(out.pixel(x, y));
            let b = px::unpack(full[((y + pad) as usize) * fw + (x + pad) as usize]);
            for c in 0..4 {
                assert!((a[c] as i32 - b[c] as i32).abs() <= 1, "({x},{y}) {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn white_canvas_fill_shares_tiles() {
        let g = TileGrid::filled(10_000, 10_000, px::pack(255, 255, 255, 255));
        let f = fill(&g, None, [255, 0, 0], 1.0);
        let a = f.tiles[0].as_ref().unwrap();
        let b = f.tiles[100].as_ref().unwrap();
        assert!(Arc::ptr_eq(a, b));
    }
}

/// Catmull-Rom bicubic resample of the layer (Super Zoom's upscaler), on
/// the GPU when that's on and wins.
pub fn upscale_bicubic(layer: &Layer, nw: u32, nh: u32) -> TileGrid {
    let g = &layer.grid;
    let px = nw as u64 * nh as u64;
    let bytes = 4 * (px + g.w as u64 * g.h as u64);
    super::gpu::resample::switch("Super Zoom upscale", px, bytes, |gpu| super::gpu::resample::upscale_bicubic(gpu, layer, nw, nh), || upscale_bicubic_cpu(layer, nw, nh))
}

/// [`upscale_bicubic`] on the CPU (the reference).
pub fn upscale_bicubic_cpu(layer: &Layer, nw: u32, nh: u32) -> TileGrid {
    let g = &layer.grid;
    let (fx, fy) = (g.w as f32 / nw as f32, g.h as f32 / nh as f32);
    #[inline]
    fn cr(t: f32) -> [f32; 4] {
        let (t2, t3) = (t * t, t * t * t);
        [(-t3 + 2.0 * t2 - t) * 0.5, (3.0 * t3 - 5.0 * t2 + 2.0) * 0.5, (-3.0 * t3 + 4.0 * t2 + t) * 0.5, (t3 - t2) * 0.5]
    }
    build_grid(nw, nh, |tx, ty| {
        let (x0, y0) = (tx * TS_I, ty * TS_I);
        let u0 = ((x0 as f32 + 0.5) * fx - 0.5).floor() as i32 - 2;
        let v0 = ((y0 as f32 + 0.5) * fy - 0.5).floor() as i32 - 2;
        let u1 = ((x0 as f32 + TS as f32 + 0.5) * fx).ceil() as i32 + 3;
        let v1 = ((y0 as f32 + TS as f32 + 0.5) * fy).ceil() as i32 + 3;
        let (bw, bh) = ((u1 - u0) as usize, (v1 - v0) as usize);
        if let Some(None) = g.region_uniform(RectI::new(u0, v0, u1, v1).intersect(&g.bounds())) {
            return None;
        }
        let mut b = vec![0u32; bw * bh];
        g.read_region_clamped(u0, v0, bw, bh, &mut b);
        let mut out = vec![0u32; TS * TS].into_boxed_slice();
        for y in 0..TS {
            let v = (y0 as f32 + y as f32 + 0.5) * fy - 0.5 - v0 as f32;
            let vi = v.floor() as isize;
            let wy = cr(v - vi as f32);
            for x in 0..TS {
                let u = (x0 as f32 + x as f32 + 0.5) * fx - 0.5 - u0 as f32;
                let ui = u.floor() as isize;
                let wx = cr(u - ui as f32);
                let mut acc = [0f32; 4];
                for j in 0..4 {
                    let row = ((vi + j as isize - 1).clamp(0, bh as isize - 1) as usize) * bw;
                    for i in 0..4 {
                        let p = px::unpack(b[row + (ui + i as isize - 1).clamp(0, bw as isize - 1) as usize]);
                        let wgt = wx[i] * wy[j];
                        for c in 0..4 {
                            acc[c] += p[c] as f32 * wgt;
                        }
                    }
                }
                let a = acc[3].round().clamp(0.0, 255.0);
                let q = |v: f32| v.round().clamp(0.0, a) as u8;
                out[y * TS + x] = px::pack(q(acc[0]), q(acc[1]), q(acc[2]), a as u8);
            }
        }
        finish_tile(out)
    })
}

/// Centers and radii of the dark blemishes in [`demo_scene`].
pub fn demo_blemishes(w: u32, h: u32) -> Vec<(f32, f32, f32)> {
    use super::filters::util::hash01;
    let (fw, fh) = (w as f32, h as f32);
    (0..24)
        .map(|k| {
            let u = hash01(k, 1, 77);
            let v = hash01(k, 2, 77);
            (fw * (0.05 + 0.9 * u), fh * (0.66 + 0.3 * v), fh * (0.004 + 0.006 * hash01(k, 3, 77)))
        })
        .collect()
}

/// A photo-like test scene (sky, textured ground, a shaded red ball, a blue
/// box, a sun and a few dark blemishes) for demos and tool benchmarks.
pub fn demo_scene(w: u32, h: u32) -> TileGrid {
    use super::filters::util::{fbm, hash01};
    let (fw, fh) = (w as f32, h as f32);
    let s = fh;
    let ball = (0.30 * fw, 0.62 * fh, 0.14 * s);
    let sun = (0.83 * fw, 0.17 * fh, 0.07 * s);
    let bx = (0.55 * fw, 0.42 * fh, 0.76 * fw, 0.80 * fh);
    let blemishes = demo_blemishes(w, h);
    let aa = |d: f32| (0.5 - d).clamp(0.0, 1.0);
    build_grid(w, h, |tx, ty| {
        let mut out = vec![0u32; TS * TS].into_boxed_slice();
        for y in 0..TS {
            let gy = (ty * TS_I + y as i32) as f32 + 0.5;
            for x in 0..TS {
                let gx = (tx * TS_I + x as i32) as f32 + 0.5;
                let v = gy / fh;
                let horizon = 0.6 + 0.02 * (gx / fw * 9.0).sin();
                let mut c: [f32; 3] = if v < horizon {
                    let t = v / horizon;
                    [110.0 + 120.0 * t, 160.0 + 80.0 * t, 235.0 + 15.0 * t]
                } else {
                    let n = fbm(gx / s * 18.0, gy / s * 18.0, 4, 5);
                    let g = hash01(gx as i32, gy as i32, 9) * 18.0;
                    [55.0 + 60.0 * n + g, 110.0 + 70.0 * n + g, 45.0 + 30.0 * n + g * 0.5]
                };
                let mut over = |col: [f32; 3], a: f32| {
                    for i in 0..3 {
                        c[i] = c[i] * (1.0 - a) + col[i] * a;
                    }
                };
                // Sun.
                let ds = ((gx - sun.0).powi(2) + (gy - sun.1).powi(2)).sqrt() - sun.2;
                over([255.0, 214.0, 60.0], aa(ds));
                // Box with a vertical shade and a dark outline.
                let dbx = (bx.0 - gx).max(gx - bx.2).max(bx.1 - gy).max(gy - bx.3);
                let t = (gy - bx.1) / (bx.3 - bx.1);
                over([40.0 + 30.0 * t, 70.0 + 40.0 * t, 190.0 - 50.0 * t], aa(dbx));
                // Ball with a highlight.
                let dbl = ((gx - ball.0).powi(2) + (gy - ball.1).powi(2)).sqrt();
                let l = (1.0 - ((gx - ball.0 + ball.2 * 0.35).powi(2) + (gy - ball.1 + ball.2 * 0.35).powi(2)).sqrt() / (ball.2 * 1.6)).clamp(0.0, 1.0);
                over([150.0 + 105.0 * l, 25.0 + 70.0 * l * l, 30.0 + 60.0 * l * l], aa(dbl - ball.2));
                for &(bx0, by0, br) in &blemishes {
                    let d = ((gx - bx0).powi(2) + (gy - by0).powi(2)).sqrt() - br;
                    if d < 1.0 {
                        over([30.0, 25.0, 20.0], aa(d) * 0.85);
                    }
                }
                out[y * TS + x] = px::pack(c[0].clamp(0.0, 255.0) as u8, c[1].clamp(0.0, 255.0) as u8, c[2].clamp(0.0, 255.0) as u8, 255);
            }
        }
        finish_tile(out)
    })
}
