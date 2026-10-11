//! Per-layer mip pyramids.
//!
//! Level `l` is the layer box-filtered down by `2^l`, tiled exactly like level
//! 0. The viewer composites only the level that matches the zoom, so the cost
//! of drawing the screen is bounded by the screen's pixel count no matter how
//! large the image is. Mips are updated incrementally from dirty rectangles,
//! and four identical uniform children collapse into a shared parent tile.

use std::sync::Arc;

use rayon::prelude::*;

use super::grid::{RectI, TS, TS_I, TSHIFT, Tile, TileGrid, TileRef};
use super::px::avg4;

pub fn level_dims(w: u32, h: u32, l: u32) -> (u32, u32) {
    (((w as u64 + (1u64 << l) - 1) >> l).max(1) as u32, ((h as u64 + (1u64 << l) - 1) >> l).max(1) as u32)
}

/// Number of levels above 0 needed until one tile holds the whole image.
pub fn mip_count(w: u32, h: u32) -> u32 {
    let mut l = 0;
    loop {
        let (lw, lh) = level_dims(w, h, l);
        if lw <= TS as u32 && lh <= TS as u32 {
            return l;
        }
        l += 1;
    }
}

fn downsample_tile(src: &TileGrid, tx: i32, ty: i32) -> TileRef {
    let kids = [
        src.get(2 * tx, 2 * ty),
        src.get(2 * tx + 1, 2 * ty),
        src.get(2 * tx, 2 * ty + 1),
        src.get(2 * tx + 1, 2 * ty + 1),
    ];
    if kids.iter().all(|k| k.is_none()) {
        return None;
    }
    if let Some(first) = kids[0] {
        if let Some(c) = first.uniform {
            if kids.iter().all(|k| k.is_some_and(|t| t.uniform == Some(c))) {
                return Some(first.clone());
            }
        }
    }
    let mut px = vec![0u32; TS * TS].into_boxed_slice();
    const H: usize = TS / 2;
    for (q, kid) in kids.iter().enumerate() {
        let Some(kid) = kid else { continue };
        let ox = (q & 1) * H;
        let oy = (q >> 1) * H;
        if let Some(c) = kid.uniform {
            for y in 0..H {
                px[(oy + y) * TS + ox..(oy + y) * TS + ox + H].fill(c);
            }
            continue;
        }
        let pinned = kid.read();
        let pixels: &[u32] = &pinned;
        for y in 0..H {
            let r0 = &pixels[2 * y * TS..(2 * y + 1) * TS];
            let r1 = &pixels[(2 * y + 1) * TS..(2 * y + 2) * TS];
            let out = &mut px[(oy + y) * TS + ox..(oy + y) * TS + ox + H];
            for (x, o) in out.iter_mut().enumerate() {
                *o = avg4(r0[2 * x], r0[2 * x + 1], r1[2 * x], r1[2 * x + 1]);
            }
        }
    }
    Some(Arc::new(Tile::from_raw(px)))
}

/// Builds all mip levels from scratch.
pub fn build_mips(base: &TileGrid) -> Vec<TileGrid> {
    let n = mip_count(base.w, base.h);
    let mut levels: Vec<TileGrid> = Vec::with_capacity(n as usize);
    if base.w == base.h && base.w.is_power_of_two() {
        let shared = base.tiles.first().cloned().flatten();
        if shared.as_ref().is_none_or(|t| t.uniform.is_some()) && base.tiles.iter().all(|t| super::grid::same_tile(t, &shared)) {
            for l in 1..=n { let (w, h) = level_dims(base.w, base.h, l); let mut g = TileGrid::new(w, h); g.tiles.fill(shared.clone()); levels.push(g); }
            return levels;
        }
    }

    for l in 1..=n {
        let (lw, lh) = level_dims(base.w, base.h, l);
        let src = if l == 1 { base } else { &levels[l as usize - 2] };
        let mut g = TileGrid::new(lw, lh);
        let cols = g.cols as usize;
        g.tiles
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| *slot = downsample_tile(src, (i % cols) as i32, (i / cols) as i32));
        levels.push(g);
    }
    levels
}

/// Recomputes the mip tiles touched by `dirty` (level-0 pixel rects).
pub fn update_mips(base: &TileGrid, mips: &mut [TileGrid], dirty: &[RectI]) {
    if dirty.is_empty() {
        return;
    }
    let mut parents: Vec<(i32, i32)> = Vec::new();
    for l in 1..=mips.len() {
        let (before, after) = mips.split_at_mut(l - 1);
        let src = if l == 1 { base } else { &before[l - 2] };
        let dst = &mut after[0];
        let mut mark = vec![false; dst.tiles.len()];
        let mut list = Vec::new();
        let mut add = |tx, ty| {
            let i = dst.idx(tx, ty);
            if !mark[i] { mark[i] = true; list.push((tx, ty)); }
        };
        if l == 1 {
            let shift = TSHIFT + 1;
            for r in dirty {
                if r.is_empty() { continue; }
                let tx0 = (r.x0.max(0) >> shift).min(dst.cols as i32 - 1);
                let ty0 = (r.y0.max(0) >> shift).min(dst.rows as i32 - 1);
                let tx1 = ((r.x1 - 1).max(0) >> shift).min(dst.cols as i32 - 1);
                let ty1 = ((r.y1 - 1).max(0) >> shift).min(dst.rows as i32 - 1);
                for ty in ty0..=ty1 { for tx in tx0..=tx1 { add(tx, ty); } }
            }
        } else {
            // Every higher-level dirty tile is a parent of a changed child.
            // Long strokes may repeat thousands of rectangles; deduplicate once.
            for &(tx, ty) in &parents { add(tx >> 1, ty >> 1); }
        }
        let fresh: Vec<TileRef> = if list.len() <= 1 { list.iter().map(|&(tx, ty)| downsample_tile(src, tx, ty)).collect() } else { list.par_iter().map(|&(tx, ty)| downsample_tile(src, tx, ty)).collect() };
        for (&(tx, ty), t) in list.iter().zip(fresh) {
            let i = dst.idx(tx, ty);
            dst.tiles[i] = t;
        }
        parents = list;
    }
}

/// Level-0 pixel rects of the tiles that differ between two grids of equal size.
pub fn diff_rects(a: &TileGrid, b: &TileGrid) -> Vec<RectI> {
    let cols = a.cols as usize;
    a.tiles
        .par_iter()
        .zip(b.tiles.par_iter())
        .enumerate()
        .filter(|(_, (x, y))| !super::grid::same_tile(x, y))
        .map(|(i, _)| TileGrid::tile_rect((i % cols) as i32, (i / cols) as i32))
        .collect()
}

#[allow(dead_code)]
pub fn level_tile_px(l: i32) -> i32 {
    TS_I << l.max(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_parents_match_full_rebuild_for_repeated_edge_rectangles() {
        let mut grid = crate::engine::ops::procedural(1301, 779);
        let mut mips = build_mips(&grid);
        let mut dirty = Vec::new();
        for &(tx, ty) in &[(0, 0), (2, 1), (5, 3), (2, 1)] {
            let i = grid.idx(tx, ty);
            grid.tiles[i] = Some(Arc::new(Tile::solid(0xff123456)));
            for _ in 0..50 { dirty.push(TileGrid::tile_rect(tx, ty)); }
        }
        update_mips(&grid, &mut mips, &dirty);
        let reference = build_mips(&grid);
        for (got, want) in mips.iter().zip(reference) {
            for (a, b) in got.tiles.iter().zip(want.tiles) {
                assert_eq!(&*a.as_ref().unwrap().read(), &*b.as_ref().unwrap().read());
            }
        }
    }

    #[test]
    fn uniform_canvas_mips_share_tiles() {
        let g = TileGrid::filled(20_000, 15_000, 0xFFFF_FFFF);
        let mips = build_mips(&g);
        assert!(!mips.is_empty());
        assert_eq!(mips.last().unwrap().cols, 1);
        let a = g.tiles[0].as_ref().unwrap();
        let b = mips[2].tiles[0].as_ref().unwrap();
        assert!(Arc::ptr_eq(a, b));
    }
}
