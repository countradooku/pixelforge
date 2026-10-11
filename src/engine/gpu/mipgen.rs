//! Mip pyramids built on the GPU (`wgsl/mips.wgsl`, exact `avg4`) and read
//! back as tile grids identical to `mips::build_mips`, including its tile
//! sharing (a parent of four equal uniform tiles is the same `Arc`, a parent
//! of four missing tiles is missing).
//!
//! Measured against the CPU (`--bench-gpu … mips`), this does not pay off
//! (see the module's numbers in `bench_gpu`): the CPU pyramid is one
//! bandwidth-bound pass over the image that reads each level once, while the
//! GPU path must also read every level back and copy it into new tiles —
//! the read-back alone costs about as much as the CPU's whole build. So
//! `Layer`/`Document` keep `mips::build_mips` / `update_mips`; this stays
//! for the measurement and for callers that only need the pyramid on the GPU
//! (`resident::levels`).

use std::sync::Arc;

use rayon::prelude::*;

use super::device::Gpu;
use super::exec::{GResult, buffer, encoder, submit, wait};
use crate::engine::grid::{TS, Tile, TileGrid, TileRef};

/// `mips::build_mips(grid)`, computed on the GPU (uploading the grid unless
/// it's resident).
pub fn build_mips(g: &Arc<Gpu>, grid: &TileGrid) -> GResult<Vec<TileGrid>> {
    let n = crate::engine::mips::mip_count(grid.w, grid.h);
    if n == 0 {
        return Ok(Vec::new());
    }
    let imgs = super::resident::levels(g, grid, n)?;
    // Read every level back in one submission.
    let sizes: Vec<u64> = imgs[1..].iter().map(|im| im.tw as u64 * im.th as u64 * 4).collect();
    let total: u64 = sizes.iter().sum();
    let down = buffer(g, total, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
    let mut enc = encoder(g);
    let mut off = 0;
    for (im, &sz) in imgs[1..].iter().zip(&sizes) {
        enc.copy_buffer_to_buffer(&im.buf, 0, &down, off, sz);
        off += sz;
    }
    let idx = submit(g, enc);
    wait(g, idx)?;
    down.map(total, wgpu::MapMode::Read)?;
    let mut levels: Vec<TileGrid> = Vec::with_capacity(n as usize);
    {
        let view = down.slice(..total).get_mapped_range();
        let all: &[u32] = bytemuck::cast_slice(&view);
        let mut off = 0usize;
        for (li, im) in imgs[1..].iter().enumerate() {
            let px = &all[off..off + (im.tw * im.th) as usize];
            off += (im.tw * im.th) as usize;
            let src = if li == 0 { grid } else { &levels[li - 1] };
            let mut lg = TileGrid::new(im.w, im.h);
            let cols = lg.cols as usize;
            let stride = im.tw as usize;
            lg.tiles.par_iter_mut().enumerate().for_each(|(i, slot)| {
                let (tx, ty) = ((i % cols) as i32, (i / cols) as i32);
                *slot = tile(src, tx, ty, px, stride);
            });
            levels.push(lg);
        }
    }
    down.unmap_now();
    Ok(levels)
}

/// The parent tile at (tx, ty): `downsample_tile`'s sharing rules, pixels
/// from the GPU level.
fn tile(src: &TileGrid, tx: i32, ty: i32, px: &[u32], stride: usize) -> TileRef {
    let kids = [src.get(2 * tx, 2 * ty), src.get(2 * tx + 1, 2 * ty), src.get(2 * tx, 2 * ty + 1), src.get(2 * tx + 1, 2 * ty + 1)];
    if kids.iter().all(|k| k.is_none()) {
        return None;
    }
    if let Some(first) = kids[0]
        && let Some(c) = first.uniform
        && kids.iter().all(|k| k.is_some_and(|t| t.uniform == Some(c)))
    {
        return Some(first.clone());
    }
    let mut out = vec![0u32; TS * TS].into_boxed_slice();
    let (x0, y0) = (tx as usize * TS, ty as usize * TS);
    for y in 0..TS {
        out[y * TS..(y + 1) * TS].copy_from_slice(&px[(y0 + y) * stride + x0..(y0 + y) * stride + x0 + TS]);
    }
    Some(Arc::new(Tile::from_raw(out)))
}
