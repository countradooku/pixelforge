//! Retouching: Clone / Pattern Stamp, History / Art History Brush, Healing
//! Brush, Spot Healing, Patch, Content-Aware Move and Red Eye.
//!
//! The brush-driven tools are [`Kernel`]s on the shared dab engine
//! (`brush.rs`): clone, pattern and history build a per-tile *target* on the
//! stroke's first touch of a tile (an integer clone offset is a plain row
//! copy through `TileGrid::read_region`, a history state decompresses only
//! that tile), healing writes a per-dab target of `source + membrane` solved
//! by the multigrid solver, and Art History paints its own little strokes
//! sequentially. The region tools (spot healing, patch, content-aware move,
//! red eye) read only the affected region plus a margin, run PatchMatch and
//! the Poisson blend there, and write back only the tiles they changed.

use std::sync::Arc;

use rayon::prelude::*;

use super::brush::{BaseView, Dab, DabShape, Kernel, TargetWriter, lerp_rows};
use super::document::HistorySource;
use super::grid::{RectI, TS, TS_I, Tile, TileGrid, TileRef, finish_tile, tile_mut};
use super::patchmatch;
use super::poisson;
use super::px;
use super::selection::{MaskGrid, MaskTile, Selection};

/// Reads `w x h` level-0 source pixels at `(x, y)` (clone sources).
pub type RegionFn = dyn Fn(i32, i32, usize, usize, &mut [u32]) + Send + Sync;

// ----- clone / pattern / history kernels -------------------------------------------------

/// Clone Stamp: the target is the source image shifted by the clone offset.
pub struct CloneKernel {
    pub src: Arc<RegionFn>,
    pub dx: i32,
    pub dy: i32,
}

impl Kernel for CloneKernel {
    fn prepare(&self, tx: i32, ty: i32, _base: Option<&Tile>) -> Option<Box<[u32]>> {
        let mut t = vec![0u32; TS * TS].into_boxed_slice();
        (self.src)(tx * TS_I + self.dx, ty * TS_I + self.dy, TS, TS, &mut t);
        Some(t)
    }
    fn apply_row(&self, base: &[u32], target: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
        lerp_rows(base, target, alpha, out);
    }
}

/// A tileable pattern (straight RGBA → premultiplied).
pub struct Pattern {
    pub name: &'static str,
    pub w: usize,
    pub h: usize,
    pub px: Vec<u32>,
}

/// Built-in patterns (procedural, generated once).
pub fn patterns() -> &'static [Pattern] {
    static P: std::sync::OnceLock<Vec<Pattern>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        use super::filters::util::{fbm, hash01};
        let make = |name: &'static str, w: usize, h: usize, f: &dyn Fn(usize, usize) -> [u8; 3]| Pattern {
            name,
            w,
            h,
            px: (0..w * h).map(|i| {
                let c = f(i % w, i / w);
                px::pack(c[0], c[1], c[2], 255)
            }).collect(),
        };
        vec![
            make("Checker", 64, 64, &|x, y| if ((x / 32) + (y / 32)) % 2 == 0 { [235, 235, 235] } else { [60, 60, 70] }),
            make("Bricks", 128, 64, &|x, y| {
                let row = y / 32;
                let xx = (x + if row % 2 == 1 { 32 } else { 0 }) % 128;
                let mortar = y % 32 < 3 || xx % 64 < 3;
                let n = (hash01((xx / 64) as i32 + row as i32 * 7, row as i32, 3) * 40.0) as u8;
                if mortar { [200, 195, 185] } else { [150 + n, 60 + n / 2, 45 + n / 3] }
            }),
            make("Dots", 48, 48, &|x, y| {
                let (dx, dy) = (x as f32 - 24.0, y as f32 - 24.0);
                if dx * dx + dy * dy < 100.0 { [30, 90, 200] } else { [245, 240, 225] }
            }),
            make("Canvas", 128, 128, &|x, y| {
                let n = fbm(x as f32 / 6.0, y as f32 / 6.0, 3, 2);
                let v = (200.0 + 40.0 * n + 12.0 * ((x % 4) as f32 - 1.5).abs()) as u8;
                [v, v.saturating_sub(8), v.saturating_sub(20)]
            }),
            make("Wood", 256, 64, &|x, y| {
                let g = ((y as f32 * 0.4 + fbm(x as f32 / 40.0, y as f32 / 10.0, 3, 9) * 6.0).sin() * 0.5 + 0.5) * 50.0;
                [(130.0 + g) as u8, (85.0 + g * 0.6) as u8, (45.0 + g * 0.3) as u8]
            }),
        ]
    })
}

pub struct PatternKernel {
    pub pattern: usize,
}

impl Kernel for PatternKernel {
    fn prepare(&self, tx: i32, ty: i32, _base: Option<&Tile>) -> Option<Box<[u32]>> {
        let p = &patterns()[self.pattern % patterns().len()];
        let mut t = vec![0u32; TS * TS].into_boxed_slice();
        for y in 0..TS {
            let py = (ty as usize * TS + y) % p.h;
            for x in 0..TS {
                t[y * TS + x] = p.px[py * p.w + (tx as usize * TS + x) % p.w];
            }
        }
        Some(t)
    }
    fn apply_row(&self, base: &[u32], target: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
        lerp_rows(base, target, alpha, out);
    }
}

/// History Brush: the target is the chosen history state's tile.
pub struct HistoryKernel {
    pub src: HistorySource,
}

impl Kernel for HistoryKernel {
    fn prepare(&self, tx: i32, ty: i32, _base: Option<&Tile>) -> Option<Box<[u32]>> {
        let i = (ty as u32 * self.src.cols + tx as u32) as usize;
        Some(match self.src.tile(i) {
            Some(t) => t.read().to_vec().into_boxed_slice(),
            None => vec![0u32; TS * TS].into_boxed_slice(),
        })
    }
    fn apply_row(&self, base: &[u32], target: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
        lerp_rows(base, target, alpha, out);
    }
}

// ----- healing brush ---------------------------------------------------------------------

/// Healing Brush: per dab, the cloned source is blended into the current
/// pixels with a Poisson membrane solved over the dab's bounding box.
pub struct HealKernel {
    pub src: Arc<RegionFn>,
    pub dx: i32,
    pub dy: i32,
    pub quality: poisson::Quality,
    /// Reused across dabs (no per-dab allocation of the hierarchy).
    solver: poisson::Solver,
}

impl HealKernel {
    pub fn new(src: Arc<RegionFn>, dx: i32, dy: i32, quality: poisson::Quality) -> Self {
        HealKernel { src, dx, dy, quality, solver: poisson::Solver::new() }
    }
}

impl Kernel for HealKernel {
    fn apply_row(&self, base: &[u32], target: Option<&[u32]>, alpha: &[u8], out: &mut [u32]) {
        lerp_rows(base, target, alpha, out);
    }
    fn dynamic(&self) -> bool {
        true
    }
    fn dab_target(&mut self, d: &Dab, shape: &DabShape, _base: &BaseView, cur: &TileGrid, t: &mut TargetWriter) {
        let r = d.rect().inflate(2).intersect(&cur.bounds().inflate(0));
        if r.w() < 3 || r.h() < 3 {
            return;
        }
        let (w, h) = (r.w() as usize, r.h() as usize);
        let mut dst = vec![0u32; w * h];
        cur.read_region_clamped(r.x0, r.y0, w, h, &mut dst);
        let mut src = vec![0u32; w * h];
        (self.src)(r.x0 + self.dx, r.y0 + self.dy, w, h, &mut src);
        let mut mask = vec![0u8; w * h];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                if shape.at(r.x0 + x as i32, r.y0 + y as i32) > 0 {
                    mask[y * w + x] = 1;
                }
            }
        }
        let out = poisson::blend_rgba_q(&mut self.solver, &src, &dst, &mask, w, h, 4, self.quality);
        t.write(r.x0, r.y0, w, h, &out, (cur.h as usize).div_ceil(TS) as u32);
    }
}

// ----- art history brush --------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArtStyle {
    TightShort,
    TightLong,
    LooseCurl,
    Dab,
}

impl ArtStyle {
    pub const ALL: [ArtStyle; 4] = [ArtStyle::TightShort, ArtStyle::TightLong, ArtStyle::LooseCurl, ArtStyle::Dab];
    pub fn label(self) -> &'static str {
        match self {
            ArtStyle::TightShort => "Tight Short",
            ArtStyle::TightLong => "Tight Long",
            ArtStyle::LooseCurl => "Loose Curl",
            ArtStyle::Dab => "Dab",
        }
    }
}

/// Art History Brush: each dab spawns small stylized strokes whose colour
/// comes from the history state (sequential kernel).
pub struct ArtHistoryKernel {
    pub src: HistorySource,
    pub style: ArtStyle,
    /// Area (radius, px) the strokes spread over.
    pub area: f32,
    pub opacity: u32,
    pub rng: u32,
}

impl ArtHistoryKernel {
    fn color_at(&self, x: i32, y: i32, cache: &mut Option<(usize, TileRef)>) -> u32 {
        if x < 0 || y < 0 || x >= self.src.w as i32 || y >= self.src.h as i32 {
            return 0;
        }
        let i = ((y >> 8) as u32 * self.src.cols + (x >> 8) as u32) as usize;
        if cache.as_ref().is_none_or(|c| c.0 != i) {
            *cache = Some((i, self.src.tile(i)));
        }
        match &cache.as_ref().unwrap().1 {
            Some(t) => t.read()[((y & 255) as usize) * TS + (x & 255) as usize],
            None => 0,
        }
    }
}

fn rnd(s: &mut u32) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    (*s >> 8) as f32 / (1u32 << 24) as f32
}

impl Kernel for ArtHistoryKernel {
    fn apply_row(&self, base: &[u32], _t: Option<&[u32]>, _a: &[u8], out: &mut [u32]) {
        out.copy_from_slice(base);
    }
    fn sequential(&self) -> bool {
        true
    }
    fn seq_dab(&mut self, d: &Dab, _shape: &DabShape, grid: &mut TileGrid, sel: Option<&Selection>) -> RectI {
        let tip = (d.r * 0.18).max(1.0);
        let (len, curl, count) = match self.style {
            ArtStyle::TightShort => (tip * 3.0, 0.15, 10),
            ArtStyle::TightLong => (tip * 8.0, 0.08, 6),
            ArtStyle::LooseCurl => (tip * 6.0, 0.6, 7),
            ArtStyle::Dab => (tip * 0.5, 0.0, 14),
        };
        let mut dirty = RectI::EMPTY;
        let mut cache = None;
        let img = grid.bounds();
        for _ in 0..count {
            let ang = rnd(&mut self.rng) * std::f32::consts::TAU;
            let rad = self.area.max(d.r) * rnd(&mut self.rng).sqrt();
            let (mut x, mut y) = (d.x + rad * ang.cos(), d.y + rad * ang.sin());
            let c = self.color_at(x as i32, y as i32, &mut cache);
            if px::alpha(c) == 0 {
                continue;
            }
            let mut dir = rnd(&mut self.rng) * std::f32::consts::TAU;
            let steps = (len / (tip * 0.5)).ceil().max(1.0) as usize;
            for _ in 0..steps {
                let r = RectI::new((x - tip) as i32, (y - tip) as i32, (x + tip) as i32 + 2, (y + tip) as i32 + 2).intersect(&img);
                for py in r.y0..r.y1 {
                    for pxx in r.x0..r.x1 {
                        let dd = ((pxx as f32 + 0.5 - x).powi(2) + (py as f32 + 0.5 - y).powi(2)).sqrt();
                        let mut a = ((tip - dd + 0.5).clamp(0.0, 1.0) * self.opacity as f32) as u32;
                        if let Some(s) = sel {
                            a = a * s.at(pxx, py) as u32 / 255;
                        }
                        if a == 0 {
                            continue;
                        }
                        let ti = grid.idx(pxx >> 8, py >> 8);
                        let t = tile_mut(&mut grid.tiles[ti]);
                        let k = ((py & 255) as usize) * TS + (pxx & 255) as usize;
                        { let mut pixels = t.write(); pixels[k] = px::lerp(pixels[k], c, a); }
                    }
                }
                dirty = dirty.union(&r);
                dir += (rnd(&mut self.rng) - 0.5) * curl * 2.0;
                x += dir.cos() * tip * 0.5;
                y += dir.sin() * tip * 0.5;
            }
        }
        dirty
    }
}

// ----- region tools -----------------------------------------------------------------------------

/// Bounding box of the non-empty pixels of a mask.
pub fn mask_bbox(m: &MaskGrid) -> RectI {
    let mut b = RectI::EMPTY;
    for (i, t) in m.tiles.iter().enumerate() {
        let tx = (i as u32 % m.cols) as i32;
        let ty = (i as u32 / m.cols) as i32;
        match t {
            MaskTile::Empty => {}
            MaskTile::Full => b = b.union(&TileGrid::tile_rect(tx, ty)),
            MaskTile::Partial(px) => {
                let (mut x0, mut y0, mut x1, mut y1) = (TS, TS, 0, 0);
                for y in 0..TS {
                    for x in 0..TS {
                        if px[y * TS + x] > 0 {
                            x0 = x0.min(x);
                            y0 = y0.min(y);
                            x1 = x1.max(x + 1);
                            y1 = y1.max(y + 1);
                        }
                    }
                }
                if x1 > x0 {
                    b = b.union(&RectI::new(tx * TS_I + x0 as i32, ty * TS_I + y0 as i32, tx * TS_I + x1 as i32, ty * TS_I + y1 as i32));
                }
            }
        }
    }
    b.intersect(&RectI::new(0, 0, m.w as i32, m.h as i32))
}

/// Writes a `w x h` region back into a copy of `grid`, touching only tiles
/// where `cov` is non-zero; pixels are blended `lerp(old, new, cov)`.
pub fn write_region(grid: &TileGrid, r: RectI, new: &[u32], cov: &[u8]) -> TileGrid {
    let mut out = grid.clone();
    let (w, _) = (r.w() as usize, r.h() as usize);
    let (tx0, ty0, tx1, ty1) = r.intersect(&grid.bounds()).tile_span(grid.cols, grid.rows);
    let tiles: Vec<(i32, i32)> = (ty0..ty1).flat_map(|ty| (tx0..tx1).map(move |tx| (tx, ty))).collect();
    let fresh: Vec<(usize, TileRef)> = tiles
        .par_iter()
        .filter_map(|&(tx, ty)| {
            let tr = TileGrid::tile_rect(tx, ty).intersect(&r).intersect(&grid.bounds());
            let any = (tr.y0..tr.y1).any(|y| (tr.x0..tr.x1).any(|x| cov[((y - r.y0) as usize) * w + (x - r.x0) as usize] > 0));
            if !any {
                return None;
            }
            let idx = grid.idx(tx, ty);
            let mut px_ = match &grid.tiles[idx] {
                Some(t) => t.read().to_vec().into_boxed_slice(),
                None => vec![0u32; TS * TS].into_boxed_slice(),
            };
            for y in tr.y0..tr.y1 {
                for x in tr.x0..tr.x1 {
                    let k = ((y - r.y0) as usize) * w + (x - r.x0) as usize;
                    let c = cov[k];
                    if c == 0 {
                        continue;
                    }
                    let i = ((y - ty * TS_I) as usize) * TS + (x - tx * TS_I) as usize;
                    px_[i] = if c == 255 { new[k] } else { px::lerp(px_[i], new[k], c as u32) };
                }
            }
            Some((idx, finish_tile(px_)))
        })
        .collect();
    for (i, t) in fresh {
        out.tiles[i] = t;
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpotMode {
    ContentAware,
    /// Synthesizes a fresh texture from the ring around the spot (no
    /// structure is copied), then blends it seamlessly.
    CreateTexture,
    ProximityMatch,
}

/// Spot Healing: fills the stroke area (`cov`, soft coverage) from the
/// surrounding texture and blends it seamlessly. Returns the new grid and
/// the changed rect.
pub fn spot_heal(grid: &TileGrid, cov: &MaskGrid, sel: Option<&Selection>, mode: SpotMode) -> Option<(TileGrid, RectI)> {
    let hb = mask_bbox(cov);
    if hb.is_empty() {
        return None;
    }
    let margin = (hb.w().max(hb.h()) as f32 * 0.9) as i32 + 12;
    let r = hb.inflate(margin).intersect(&grid.bounds());
    let (w, h) = (r.w() as usize, r.h() as usize);
    let mut img = vec![0u32; w * h];
    grid.read_region(r.x0, r.y0, w, h, &mut img);
    let mut c = vec![0u8; w * h];
    cov.read_region(r.x0, r.y0, w, h, &mut c, false);
    if let Some(s) = sel {
        for y in 0..h {
            for x in 0..w {
                let k = y * w + x;
                if c[k] > 0 {
                    c[k] = (c[k] as u32 * s.at(r.x0 + x as i32, r.y0 + y as i32) as u32 / 255) as u8;
                }
            }
        }
    }
    // The hole is everything the brush touched, grown by 1 px so the
    // blemish's own soft edge is replaced too.
    let mut hole = vec![0u8; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let k = y * w + x;
            if c[k] > 0 || c[k - 1] > 0 || c[k + 1] > 0 || c[k - w] > 0 || c[k + w] > 0 {
                hole[k] = 1;
            }
        }
    }
    let fill = match mode {
        SpotMode::ContentAware => patchmatch::complete(&img, &hole, w, h, None, &patchmatch::Params::default()),
        SpotMode::ProximityMatch => proximity_fill(&img, &hole, w, h),
        SpotMode::CreateTexture => texture_fill(&img, &hole, w, h, (r.x0 as u32).wrapping_mul(73_856_093) ^ (r.y0 as u32).wrapping_mul(19_349_663)),
    };
    let mut solver = poisson::Solver::new();
    let blended = poisson::blend_rgba(&mut solver, &fill, &img, &hole, w, h, 5);
    // Soft brush edge: inside the hole use full strength; its rim fades.
    let a: Vec<u8> = c.iter().zip(&hole).map(|(&cv, &hh)| if hh == 0 { 0 } else if cv > 0 { 255 } else { 128 }).collect();
    Some((write_region(grid, r, &blended, &a), r))
}

/// Create Texture: image quilting (Efros & Freeman) seeded from the ring
/// around the hole. The hole's bounding box is covered in raster order by
/// overlapping `p × p` blocks; each block is the best of `K` random source
/// blocks from the ring (fully outside the hole) by SSD over the pixels it
/// overlaps that are already known: the original surround or blocks laid
/// before it. Overlaps are feathered. Only texture statistics carry over,
/// never a recognizable structure; the Poisson blend afterwards removes any
/// low-frequency mismatch at the boundary. Deterministic for a given `seed`.
pub fn texture_fill(img: &[u32], hole: &[u8], w: usize, h: usize, seed: u32) -> Vec<u32> {
    use rayon::prelude::*;
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        let row = &hole[y * w..(y + 1) * w];
        if let Some(a) = row.iter().position(|&m| m != 0) {
            x0 = x0.min(a);
            x1 = x1.max(row.iter().rposition(|&m| m != 0).unwrap() + 1);
            y0 = y0.min(y);
            y1 = y1.max(y + 1);
        }
    }
    let mut out = img.to_vec();
    if x1 <= x0 {
        return out;
    }
    let size = (x1 - x0).max(y1 - y0);
    let p = (size / 5).clamp(6, 24);
    let ov = (p / 3).max(2);
    let step = p - ov;
    // Candidate sources: block corners in the ring whose block avoids the
    // hole (summed-area table of the hole mask).
    let mut sat = vec![0u32; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut run = 0;
        for x in 0..w {
            run += (hole[y * w + x] != 0) as u32;
            sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + run;
        }
    }
    let holes_in = |x: usize, y: usize| sat[(y + p) * (w + 1) + x + p] + sat[y * (w + 1) + x] - sat[y * (w + 1) + x + p] - sat[(y + p) * (w + 1) + x];
    let ring = (size * 3 / 4).max(2 * p);
    let (rx0, ry0) = (x0.saturating_sub(ring), y0.saturating_sub(ring));
    let (rx1, ry1) = ((x1 + ring).min(w).saturating_sub(p), (y1 + ring).min(h).saturating_sub(p));
    let cands: Vec<(usize, usize)> = (ry0..ry1.max(ry0))
        .step_by(2)
        .flat_map(|y| (rx0..rx1.max(rx0)).step_by(2).map(move |x| (x, y)))
        .filter(|&(x, y)| holes_in(x, y) == 0)
        .collect();
    if cands.is_empty() {
        return out;
    }
    const K: usize = 48;
    let mut rng = seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng
    };
    // `known` = original surround or already synthesized.
    let mut known: Vec<bool> = hole.iter().map(|&m| m == 0).collect();
    let rgb = |c: u32| {
        let [r, g, b, _] = px::unpack(c);
        [r as i32, g as i32, b as i32]
    };
    let mut by = y0.saturating_sub(ov);
    while by < y1 {
        let mut bx = x0.saturating_sub(ov);
        while bx < x1 {
            let (bw, bh) = (p.min(w - bx), p.min(h - by));
            let any_hole = (by..by + bh).any(|y| hole[y * w + bx..y * w + bx + bw].iter().any(|&m| m != 0));
            if any_hole {
                let picks: Vec<(usize, usize)> = (0..K).map(|_| cands[next() as usize % cands.len()]).collect();
                let (out_r, known_r) = (&out, &known);
                let cost = |&(sx, sy): &(usize, usize)| -> u64 {
                    let mut c = 0u64;
                    for y in 0..bh {
                        for x in 0..bw {
                            let i = (by + y) * w + bx + x;
                            if known_r[i] {
                                let (a, b) = (rgb(out_r[i]), rgb(img[(sy + y) * w + sx + x]));
                                c += ((a[0] - b[0]).pow(2) + (a[1] - b[1]).pow(2) + (a[2] - b[2]).pow(2)) as u64;
                            }
                        }
                    }
                    c
                };
                let costs: Vec<u64> = picks.par_iter().map(cost).collect();
                let best = *costs.iter().min().unwrap();
                // Random choice among near-best candidates keeps it varied.
                let near: Vec<usize> = (0..K).filter(|&k| costs[k] <= best + best / 10).collect();
                let (sx, sy) = picks[near[next() as usize % near.len()]];
                for y in 0..bh {
                    for x in 0..bw {
                        let i = (by + y) * w + bx + x;
                        if hole[i] == 0 {
                            continue;
                        }
                        let src = img[(sy + y) * w + sx + x];
                        if known[i] {
                            // Feather across the overlap with earlier blocks.
                            let t = ((x.min(y) + 1) as f32 / (ov + 1) as f32).min(1.0);
                            out[i] = px::lerp(out[i], src, (t * 255.0) as u32);
                        } else {
                            out[i] = src;
                            known[i] = true;
                        }
                    }
                }
            }
            bx += step;
        }
        by += step;
    }
    out
}

/// Proximity Match: the hole copies the neighbouring offset whose ring
/// around the hole matches best.
fn proximity_fill(img: &[u32], hole: &[u8], w: usize, h: usize) -> Vec<u32> {
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if hole[y * w + x] != 0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    let size = ((x1 - x0).max(y1 - y0) as i32).max(4);
    let mut best = ((0i32, 0i32), u64::MAX);
    for k in 0..16 {
        let a = k as f32 / 16.0 * std::f32::consts::TAU;
        let (dx, dy) = ((a.cos() * size as f32 * 1.1) as i32, (a.sin() * size as f32 * 1.1) as i32);
        let mut d = 0u64;
        let mut ok = true;
        for y in y0.saturating_sub(3)..(y1 + 3).min(h) {
            for x in x0.saturating_sub(3)..(x1 + 3).min(w) {
                let (sx, sy) = (x as i32 + dx, y as i32 + dy);
                if sx < 0 || sy < 0 || sx >= w as i32 || sy >= h as i32 || hole[sy as usize * w + sx as usize] != 0 {
                    ok = false;
                    break;
                }
                if hole[y * w + x] == 0 {
                    let (p, q) = (px::unpack(img[y * w + x]), px::unpack(img[sy as usize * w + sx as usize]));
                    d += (0..3).map(|c| (p[c] as i64 - q[c] as i64).pow(2) as u64).sum::<u64>();
                }
            }
            if !ok {
                break;
            }
        }
        if ok && d < best.1 {
            best = ((dx, dy), d);
        }
    }
    let (dx, dy) = best.0;
    (0..w * h)
        .map(|i| {
            if hole[i] == 0 {
                return img[i];
            }
            let (x, y) = ((i % w) as i32 + dx, (i / w) as i32 + dy);
            img[(y.clamp(0, h as i32 - 1) as usize) * w + x.clamp(0, w as i32 - 1) as usize]
        })
        .collect()
}

// ----- Content-Aware Fill workspace ------------------------------------------------------------

/// Where Content-Aware Fill may take source patches from.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SamplingArea {
    /// A band around the hole about as wide as the hole.
    #[default]
    Auto,
    /// The hole's bounding box, enlarged by the same band.
    Rectangular,
    /// Painted by the user (starts from Auto).
    Custom,
}

impl SamplingArea {
    pub const ALL: [SamplingArea; 3] = [SamplingArea::Auto, SamplingArea::Rectangular, SamplingArea::Custom];
    pub fn label(self) -> &'static str {
        match self {
            SamplingArea::Auto => "Auto",
            SamplingArea::Rectangular => "Rectangular",
            SamplingArea::Custom => "Custom",
        }
    }
}

/// Band width of the automatic sampling area for a hole.
fn caf_band(hole: &MaskGrid) -> f32 {
    let b = mask_bbox(hole);
    (b.w().max(b.h()) as f32 * 0.8).clamp(48.0, 2000.0)
}

/// The initial sampling area for `mode` (Custom starts from Auto). Auto
/// keeps, inside the band around the hole, only colours that occur in the
/// hole's immediate surroundings (an 8³-bin Lab histogram of a thin ring),
/// so unrelated objects next to the hole are not sampled.
pub fn caf_sampling(grid: &TileGrid, hole: &MaskGrid, mode: SamplingArea) -> MaskGrid {
    let band = caf_band(hole);
    let sub = |m: MaskGrid| m.combine(hole, super::masks::SelOp::Subtract);
    match mode {
        SamplingArea::Rectangular => {
            let r = mask_bbox(hole).inflate(band as i32);
            sub(MaskGrid::from_selection(&Selection::rect(r), hole.w, hole.h))
        }
        SamplingArea::Auto | SamplingArea::Custom => {
            let area = sub(hole.expand(band));
            let ring = sub(hole.expand((band * 0.15).max(12.0)));
            let bin = |c: u32| {
                let [r, g, b, _] = px::unpremultiply(c);
                let l = super::color::srgb8_to_lab([r, g, b]);
                let q = |v: f32, lo: f32, hi: f32| (((v - lo) / (hi - lo) * 8.0) as i32).clamp(0, 7) as usize;
                q(l[0], 0.0, 100.0) * 64 + q(l[1], -80.0, 80.0) * 8 + q(l[2], -80.0, 80.0)
            };
            // Ring histogram (tile-parallel, then summed).
            let cols = ring.cols as usize;
            let hist: Vec<f64> = ring
                .tiles
                .par_iter()
                .enumerate()
                .filter(|(_, t)| !matches!(t, MaskTile::Empty))
                .map(|(i, t)| {
                    let mut hh = vec![0f64; 512];
                    let (tx, ty) = ((i % cols) as i32, (i / cols) as i32);
                    let Some(tile) = grid.get(tx, ty) else { return hh };
                    let pixels = tile.read();
                    for k in 0..TS * TS {
                        let wgt = match t {
                            MaskTile::Full => 255,
                            MaskTile::Partial(m) => m[k],
                            MaskTile::Empty => 0,
                        };
                        if wgt > 127 {
                            hh[bin(pixels[k])] += 1.0;
                        }
                    }
                    hh
                })
                .reduce(|| vec![0f64; 512], |a, b| a.iter().zip(&b).map(|(x, y)| x + y).collect());
            let total: f64 = hist.iter().sum::<f64>().max(1.0);
            let keep: Vec<bool> = hist.iter().map(|&v| v / total > 0.002).collect();
            let mut out = area.clone();
            out.tiles.par_iter_mut().enumerate().for_each(|(i, t)| {
                if matches!(t, MaskTile::Empty) {
                    return;
                }
                let (tx, ty) = ((i % cols) as i32, (i / cols) as i32);
                let Some(tile) = grid.get(tx, ty) else { return };
                let pixels = tile.read();
                let mut m = vec![0u8; TS * TS];
                for (k, v) in m.iter_mut().enumerate() {
                    let a = match &*t {
                        MaskTile::Full => 255,
                        MaskTile::Partial(p) => p[k],
                        MaskTile::Empty => 0,
                    };
                    *v = if keep[bin(pixels[k])] { a } else { 0 };
                }
                *t = super::masks::normalize(m);
            });
            // Close small gaps (texture noise) and drop speckles.
            out.smooth(3)
        }
    }
}

/// A Content-Aware Fill result: the processed region and its new pixels
/// with per-pixel coverage (the hole, soft at its edge).
pub struct CafResult {
    pub region: RectI,
    pub px: Vec<u32>,
    pub cov: Vec<u8>,
}

/// Content-Aware Fill: completes `hole` by PatchMatch from `sampling`
/// (rotation / scale / mirror / colour adaptation per `adapt`), then
/// Poisson-blends the fill into its surroundings.
pub fn content_aware_fill(grid: &TileGrid, hole: &MaskGrid, sampling: &MaskGrid, adapt: patchmatch::Adapt) -> Option<CafResult> {
    let hb = mask_bbox(hole);
    if hb.is_empty() {
        return None;
    }
    let sb = mask_bbox(sampling);
    let r = hb.union(&sb).inflate(2 * patchmatch::R + 2).intersect(&grid.bounds());
    let (w, h) = (r.w() as usize, r.h() as usize);
    let mut img = vec![0u32; w * h];
    grid.read_region(r.x0, r.y0, w, h, &mut img);
    let mut c = vec![0u8; w * h];
    hole.read_region(r.x0, r.y0, w, h, &mut c, false);
    let mut allowed = vec![0u8; w * h];
    sampling.read_region(r.x0, r.y0, w, h, &mut allowed, false);
    let mut hl = vec![0u8; w * h];
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let k = y * w + x;
            if c[k] > 0 || c[k - 1] > 0 || c[k + 1] > 0 || c[k - w] > 0 || c[k + w] > 0 {
                hl[k] = 1;
            }
        }
    }
    for (a, &hh) in allowed.iter_mut().zip(&hl) {
        *a = (*a > 127 && hh == 0) as u8;
    }
    if !allowed.iter().any(|&a| a != 0) {
        return None;
    }
    let p = patchmatch::Params { adapt, ..patchmatch::Params::default() };
    let t0 = std::time::Instant::now();
    let fill = patchmatch::complete(&img, &hl, w, h, Some(&allowed), &p);
    let t1 = std::time::Instant::now();
    let mut solver = poisson::Solver::new();
    let blended = poisson::blend_rgba(&mut solver, &fill, &img, &hl, w, h, 5);
    if std::env::var("PF_TIMING").is_ok() {
        eprintln!("  caf {w}x{h}: patchmatch {:.0} ms, poisson {:.0} ms", (t1 - t0).as_secs_f64() * 1e3, t1.elapsed().as_secs_f64() * 1e3);
    }
    let cov: Vec<u8> = c.iter().zip(&hl).map(|(&cv, &hh)| if hh == 0 { 0 } else if cv > 0 { cv.max(128) } else { 96 }).collect();
    Some(CafResult { region: r, px: blended, cov })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PatchMode {
    /// Replace the selection with the texture from where it is dragged.
    Source,
    /// Copy the selection to where it is dragged.
    Destination,
    /// Fill the selection with PatchMatch, sampling around the drag target.
    ContentAware,
}

/// The Patch tool. `sel` is the patch outline, `(dx, dy)` the drag.
pub fn patch(grid: &TileGrid, sel: &MaskGrid, dx: i32, dy: i32, mode: PatchMode) -> Option<(TileGrid, RectI)> {
    let sb = mask_bbox(sel);
    if sb.is_empty() {
        return None;
    }
    let (dest_box, src_shift) = match mode {
        PatchMode::Source | PatchMode::ContentAware => (sb, (dx, dy)),
        PatchMode::Destination => (sb.translate(dx, dy), (-dx, -dy)),
    };
    let margin = 6 + patchmatch::R * 2;
    let span = if mode == PatchMode::ContentAware { dest_box.union(&sb.translate(dx, dy)) } else { dest_box };
    let r = span.inflate(margin + if mode == PatchMode::ContentAware { (sb.w().max(sb.h()) / 2).max(16) } else { 0 }).intersect(&grid.bounds());
    let (w, h) = (r.w() as usize, r.h() as usize);
    let mut dst = vec![0u32; w * h];
    grid.read_region(r.x0, r.y0, w, h, &mut dst);
    // Mask of the patched area in region coordinates.
    let mut cov = vec![0u8; w * h];
    let (mx, my) = if mode == PatchMode::Destination { (dx, dy) } else { (0, 0) };
    sel.read_region(r.x0 - mx, r.y0 - my, w, h, &mut cov, false);
    let hole: Vec<u8> = cov.iter().enumerate().map(|(k, &c)| {
        let (x, y) = (k % w, k / w);
        (c > 0 && x > 0 && y > 0 && x + 1 < w && y + 1 < h) as u8
    }).collect();
    let src = match mode {
        PatchMode::Source | PatchMode::Destination => {
            let mut s = vec![0u32; w * h];
            grid.read_region_clamped(r.x0 + src_shift.0, r.y0 + src_shift.1, w, h, &mut s);
            s
        }
        PatchMode::ContentAware => {
            // Sources only from around where the patch was dragged to.
            let mut allowed = vec![0u8; w * h];
            let target = sb.translate(dx, dy).inflate((sb.w().max(sb.h()) / 2).max(16));
            for y in 0..h {
                for x in 0..w {
                    let (gx, gy) = (r.x0 + x as i32, r.y0 + y as i32);
                    allowed[y * w + x] = (gx >= target.x0 && gx < target.x1 && gy >= target.y0 && gy < target.y1) as u8;
                }
            }
            patchmatch::complete(&dst, &hole, w, h, Some(&allowed), &patchmatch::Params::default())
        }
    };
    let mut solver = poisson::Solver::new();
    let blended = poisson::blend_rgba(&mut solver, &src, &dst, &hole, w, h, 5);
    let a: Vec<u8> = cov.iter().zip(&hole).map(|(&c, &hh)| if hh != 0 { c } else { 0 }).collect();
    Some((write_region(grid, r, &blended, &a), r))
}

/// Content-Aware Move: moves (or extends) the selected object by `(dx, dy)`,
/// fills the hole it leaves with PatchMatch and blends the object into its
/// new surroundings. `adapt` (0..1) is how strongly the object's colours
/// adapt to the destination (Photoshop's Structure / Color controls).
pub fn content_aware_move(grid: &TileGrid, sel: &MaskGrid, dx: i32, dy: i32, extend: bool, adapt: f32) -> Option<(TileGrid, RectI)> {
    let sb = mask_bbox(sel);
    if sb.is_empty() {
        return None;
    }
    let pad = (sb.w().max(sb.h()) / 2).max(24) + 2 * patchmatch::R;
    let r = sb.union(&sb.translate(dx, dy)).inflate(pad).intersect(&grid.bounds());
    let (w, h) = (r.w() as usize, r.h() as usize);
    let mut img = vec![0u32; w * h];
    grid.read_region(r.x0, r.y0, w, h, &mut img);
    let mut cov = vec![0u8; w * h];
    sel.read_region(r.x0, r.y0, w, h, &mut cov, false);
    let mut moved_cov = vec![0u8; w * h];
    sel.read_region(r.x0 - dx, r.y0 - dy, w, h, &mut moved_cov, false);
    let mut object = vec![0u32; w * h];
    grid.read_region_clamped(r.x0 - dx, r.y0 - dy, w, h, &mut object);
    let inner = |k: usize| {
        let (x, y) = (k % w, k / w);
        x > 0 && y > 0 && x + 1 < w && y + 1 < h
    };
    // 1. Fill the hole the object leaves (sources exclude both object spots).
    let filled = if extend {
        img.clone()
    } else {
        let hole: Vec<u8> = (0..w * h).map(|k| (cov[k] > 0 && inner(k)) as u8).collect();
        let allowed: Vec<u8> = (0..w * h).map(|k| (moved_cov[k] == 0) as u8).collect();
        let mut f = patchmatch::complete(&img, &hole, w, h, Some(&allowed), &patchmatch::Params::default());
        let mut solver = poisson::Solver::new();
        f = poisson::blend_rgba(&mut solver, &f, &img, &hole, w, h, 5);
        f
    };
    // 2. Paste the object, blended into the destination.
    let dmask: Vec<u8> = (0..w * h).map(|k| (moved_cov[k] > 0 && inner(k)) as u8).collect();
    let mut solver = poisson::Solver::new();
    let pasted = poisson::blend_rgba(&mut solver, &object, &filled, &dmask, w, h, 5);
    let a = adapt.clamp(0.0, 1.0);
    let out: Vec<u32> = (0..w * h)
        .map(|k| {
            if dmask[k] != 0 {
                let mixed = px::lerp(object[k], pasted[k], (a * 255.0) as u32);
                px::lerp(filled[k], mixed, moved_cov[k] as u32)
            } else {
                filled[k]
            }
        })
        .collect();
    let all = vec![255u8; w * h];
    Some((write_region(grid, r, &out, &all), r))
}

/// Red Eye: finds the red pupil around the click and neutralizes it with a
/// luminance-preserving desaturation. `pupil` scales the detected pupil
/// (0.5..1.5), `darken` (0..1) how dark it becomes.
pub fn red_eye(grid: &TileGrid, cx: i32, cy: i32, pupil: f32, darken: f32) -> Option<(TileGrid, RectI)> {
    let redness = |c: u32| -> f32 {
        let [r, g, b, a] = px::unpack(c);
        if a < 128 || r < 40 {
            return 0.0;
        }
        let gb = (g as f32 + b as f32) * 0.5 + 8.0;
        (r as f32 / gb - 1.0).max(0.0)
    };
    // Grow the search window until the red region no longer touches it.
    let mut half = 48;
    loop {
        let r = RectI::new(cx - half, cy - half, cx + half, cy + half).intersect(&grid.bounds());
        let (w, h) = (r.w() as usize, r.h() as usize);
        if w < 4 || h < 4 {
            return None;
        }
        let mut img = vec![0u32; w * h];
        grid.read_region(r.x0, r.y0, w, h, &mut img);
        let red: Vec<f32> = img.iter().map(|&c| redness(c)).collect();
        // Seed: the reddest pixel within 12 px of the click.
        let (lx, ly) = ((cx - r.x0) as usize, (cy - r.y0) as usize);
        let mut seed = None;
        let mut best = 0.35f32;
        for y in ly.saturating_sub(12)..(ly + 13).min(h) {
            for x in lx.saturating_sub(12)..(lx + 13).min(w) {
                if red[y * w + x] > best {
                    best = red[y * w + x];
                    seed = Some(y * w + x);
                }
            }
        }
        let seed = seed?;
        // Region of strong redness connected to the seed.
        let thr = (best * 0.35).max(0.3);
        let mut m = vec![false; w * h];
        let mut stack = vec![seed];
        m[seed] = true;
        let mut touches = false;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                touches = true;
            }
            for j in [i.wrapping_sub(1), i + 1, i.wrapping_sub(w), i + w] {
                if j < w * h && !m[j] && red[j] > thr && ((j % w) as i32 - x as i32).abs() <= 1 {
                    m[j] = true;
                    stack.push(j);
                }
            }
        }
        if touches && half < 512 {
            half *= 2;
            continue;
        }
        // Fill holes (the catch-light inside the pupil): anything not reachable
        // from the window border without crossing the red region.
        let mut outside = vec![false; w * h];
        let mut stack: Vec<usize> = (0..w * h).filter(|&i| { let (x, y) = (i % w, i / w); (x == 0 || y == 0 || x + 1 == w || y + 1 == h) && !m[i] }).collect();
        for &i in &stack {
            outside[i] = true;
        }
        while let Some(i) = stack.pop() {
            let x = i % w;
            for j in [i.wrapping_sub(1), i + 1, i.wrapping_sub(w), i + w] {
                if j < w * h && !outside[j] && !m[j] && ((j % w) as i32 - x as i32).abs() <= 1 {
                    outside[j] = true;
                    stack.push(j);
                }
            }
        }
        let pupil_mask: Vec<bool> = (0..w * h).map(|i| !outside[i]).collect();
        // Pupil size: grow / shrink with the distance transform, then feather.
        let feat: Vec<bool> = pupil_mask.iter().map(|&b| !b).collect();
        let d_in = super::masks::edt(&feat, w, h);
        let feat2: Vec<bool> = pupil_mask.clone();
        let d_out = super::masks::edt(&feat2, w, h);
        let area = pupil_mask.iter().filter(|&&b| b).count() as f32;
        let radius = (area / std::f32::consts::PI).sqrt();
        let delta = (pupil.clamp(0.3, 2.0) - 1.0) * radius;
        let cov: Vec<u8> = (0..w * h)
            .map(|i| {
                // Signed distance to the pupil edge (positive inside).
                let sd = if pupil_mask[i] { d_in[i].sqrt() } else { -d_out[i].sqrt() };
                (((sd + delta) / 1.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8
            })
            .collect();
        let k = darken.clamp(0.0, 1.0);
        let fixed: Vec<u32> = img
            .iter()
            .map(|&c| {
                let [r_, g, b, a] = px::unpremultiply(c);
                // Luminance from the non-red channels (the pupil's true
                // brightness), desaturated, then darkened.
                let l = (g as f32 * 0.6 + b as f32 * 0.4).min(r_ as f32) * (1.0 - 0.75 * k);
                let v = l.clamp(0.0, 255.0) as u8;
                px::premultiply(v, (g as f32).min(l + 6.0) as u8, (b as f32).min(l + 8.0) as u8, a)
            })
            .collect();
        return Some((write_region(grid, r, &fixed, &cov), r));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::brush::{BrushSettings, Stroke};

    fn textured(w: u32, h: u32) -> TileGrid {
        super::super::ops::demo_scene(w, h)
    }

    #[test]
    fn clone_copies_with_offset() {
        let g = super::super::ops::procedural(1024, 1024);
        let src = g.clone();
        let f: Arc<RegionFn> = Arc::new(move |x, y, w, h, out: &mut [u32]| src.read_region(x, y, w, h, out));
        let mut grid = g.clone();
        let mut st = Stroke::new(0, &grid, BrushSettings { size: 60.0, hardness: 1.0, ..Default::default() }, Box::new(CloneKernel { src: f, dx: 300, dy: -100 }), None);
        st.add_point(500.0, 500.0, &mut grid);
        assert_eq!(grid.pixel(500, 500), g.pixel(800, 400));
        assert_eq!(grid.pixel(510, 495), g.pixel(810, 395));
        assert_eq!(grid.pixel(600, 600), g.pixel(600, 600));
    }

    #[test]
    fn healing_blends_into_destination_tone() {
        // Destination: a dark flat area; source: bright texture. The healed
        // dab keeps the texture's detail but takes the dark tone.
        let mut grid = TileGrid::filled(512, 512, px::pack(40, 40, 40, 255));
        let tex = super::super::ops::procedural(512, 512);
        let f: Arc<RegionFn> = Arc::new(move |x, y, w, h, out: &mut [u32]| tex.read_region_clamped(x, y, w, h, out));
        let mut st = Stroke::new(0, &grid, BrushSettings { size: 80.0, hardness: 1.0, ..Default::default() }, Box::new(HealKernel::new(f, 0, 0, Default::default())), None);
        st.add_point(256.0, 256.0, &mut grid);
        let c = px::unpack(grid.pixel(256, 256));
        assert!(c[0] < 120 && c[1] < 120, "healed pixel kept the bright source tone: {c:?}");
        let edge = px::unpack(grid.pixel(256 + 36, 256));
        assert!((edge[0] as i32 - 40).abs() < 25, "seam at the dab edge: {edge:?}");
    }

    #[test]
    fn spot_heal_removes_blemish() {
        let g = textured(1024, 768);
        // Paint a dark blemish onto the sky, then heal it.
        let mut dirty = g.clone();
        let mut st = Stroke::brush(0, &dirty, BrushSettings { size: 30.0, hardness: 1.0, ..Default::default() }, [10, 10, 10], false, None);
        st.add_point(300.0, 200.0, &mut dirty);
        let mut cov = MaskGrid::new(1024, 768);
        let sel = Selection::ellipse(RectI::new(280, 180, 320, 220));
        cov = cov.combine(&MaskGrid::from_selection(&sel, 1024, 768), super::super::masks::SelOp::Add);
        let (healed, r) = spot_heal(&dirty, &cov, None, SpotMode::ContentAware).unwrap();
        assert!(r.contains_rect(&RectI::new(280, 180, 320, 220)));
        let a = px::unpack(healed.pixel(300, 200));
        let b = px::unpack(g.pixel(300, 200));
        for c in 0..3 {
            assert!((a[c] as i32 - b[c] as i32).abs() < 20, "healed {a:?} vs clean {b:?}");
        }
        // Untouched tiles stay shared.
        assert!(super::super::grid::same_tile(&healed.tiles[healed.idx(3, 2)], &dirty.tiles[dirty.idx(3, 2)]));
    }

    #[test]
    fn create_texture_synthesizes_matching_texture() {
        // Grainy noise over a horizontal ramp, with a dark blob in the middle.
        let (w, h) = (512u32, 512u32);
        let noise = |x: i32, y: i32| (super::super::filters::util::hash01(x, y, 9) * 60.0) as i32;
        let base = super::super::grid::build_grid(w, h, |tx, ty| {
            let mut t = vec![0u32; TS * TS].into_boxed_slice();
            for y in 0..TS {
                for x in 0..TS {
                    let (gx, gy) = (tx as i32 * 256 + x as i32, ty as i32 * 256 + y as i32);
                    let v = (60 + gx / 8 + noise(gx, gy)).clamp(0, 255) as u8;
                    t[y * TS + x] = px::pack(v, v, (v as u32 * 9 / 10) as u8, 255);
                }
            }
            super::super::grid::finish_tile(t)
        });
        let mut dirty = base.clone();
        let mut st = Stroke::brush(0, &dirty, BrushSettings { size: 100.0, hardness: 1.0, ..Default::default() }, [0, 0, 0], false, None);
        st.add_point(256.0, 256.0, &mut dirty);
        let spot = RectI::new(200, 200, 312, 312);
        let cov = MaskGrid::from_selection(&Selection::ellipse(spot), w, h);
        let (healed, _) = spot_heal(&dirty, &cov, None, SpotMode::CreateTexture).unwrap();
        let stats = |g: &TileGrid, inside: bool| {
            let (mut s, mut s2, mut n) = (0f64, 0f64, 0f64);
            for y in 170..342 {
                for x in 170..342 {
                    let (dx, dy) = (x as f32 - 256.0, y as f32 - 256.0);
                    let d = (dx * dx + dy * dy).sqrt();
                    if (inside && d < 45.0) || (!inside && d > 62.0 && d < 86.0) {
                        let v = px::unpack(g.pixel(x, y))[0] as f64 - (60.0 + x as f64 / 8.0);
                        s += v;
                        s2 += v * v;
                        n += 1.0;
                    }
                }
            }
            let m = s / n;
            (m, (s2 / n - m * m).sqrt())
        };
        let (mi, si) = stats(&healed, true);
        let (mr, sr) = stats(&healed, false);
        // Same tone (ramp-relative) and the same grain strength as the ring.
        assert!((mi - mr).abs() < 6.0, "mean {mi:.1} vs ring {mr:.1}");
        assert!(si > 0.7 * sr && si < 1.3 * sr, "grain {si:.1} vs ring {sr:.1}");
        // Deterministic.
        let (again, _) = spot_heal(&dirty, &cov, None, SpotMode::CreateTexture).unwrap();
        assert_eq!(again.pixel(256, 256), healed.pixel(256, 256));
    }

    #[test]
    fn red_eye_desaturates_pupil() {
        let mut g = TileGrid::filled(256, 256, px::pack(220, 180, 160, 255));
        for y in 0..256 {
            for x in 0..256 {
                let d = ((x as f32 - 128.0).powi(2) + (y as f32 - 128.0).powi(2)).sqrt();
                if d < 20.0 {
                    let c = if d < 4.0 { px::pack(255, 255, 255, 255) } else { px::pack(200, 30, 35, 255) };
                    let i = g.idx(0, 0);
                    tile_mut(&mut g.tiles[i]).write()[y * TS + x] = c;
                }
            }
        }
        let (out, _) = red_eye(&g, 128, 136, 1.0, 0.5).unwrap();
        let p = px::unpack(out.pixel(140, 128));
        assert!(p[0] < 60, "pupil still red: {p:?}");
        let skin = px::unpack(out.pixel(170, 128));
        assert_eq!(skin, [220, 180, 160, 255]);
    }
}
