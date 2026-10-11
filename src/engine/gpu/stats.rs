//! Histogram statistics on the GPU (`filters::stats`: Equalize, Auto Tone,
//! Auto Contrast, Average, Match Color), from the same mip level the CPU
//! samples. Integer-exact: same histograms, sums and count.
//!
//! The CPU reads at most a few megapixels here (it samples the coarsest mip
//! with at least 1 MP), so the GPU only pays off when that level is already
//! resident; `--bench-gpu stats` measures both.

use std::sync::Arc;

use super::device::Gpu;
use super::exec::{self, GResult, buffer, dispatch, encoder, groups, submit, wait};
use super::shaders::PipeKey;
use crate::engine::document::Layer;
use crate::engine::filters::Stats;

/// The level `filters::stats` samples.
pub fn level_of(layer: &Layer) -> u32 {
    let mut l = layer.mips.len() as u32;
    while l > 0 {
        let g = layer.level(l);
        if g.w as u64 * g.h as u64 >= 1_000_000 {
            break;
        }
        l -= 1;
    }
    l
}

pub fn stats(g: &Arc<Gpu>, layer: &Layer) -> GResult<Stats> {
    let grid = layer.level(level_of(layer));
    let img = super::resident::image(g, grid)?;
    let un = super::kernels::point::unpremul(g);
    let words = 768 + 10;
    let out = buffer(g, words as u64 * 4, exec::STORAGE)?;
    let down = buffer(g, words as u64 * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
    let scopes = exec::Scopes::push(g);
    let mut enc = encoder(g);
    enc.clear_buffer(&out, 0, None);
    let entry = if g.caps.subgroups && g.caps.subgroup_min > 1 { "stats_sg" } else { "stats_wg" };
    let module = if entry == "stats_sg" { "stats_sg" } else { "stats" };
    let key = PipeKey { module, entry, f16: false, consts: Vec::new() };
    dispatch(g, &mut enc, None, &key, &[img.tw, img.w, img.h, 0], &[(1, &img.buf), (2, &out), (3, &un)], [groups(img.w, 16), groups(img.h, 16), 1])?;
    enc.copy_buffer_to_buffer(&out, 0, &down, 0, words as u64 * 4);
    let idx = submit(g, enc);
    scopes.pop()?;
    wait(g, idx)?;
    down.map(words as u64 * 4, wgpu::MapMode::Read)?;
    let v: Vec<u32> = bytemuck::cast_slice(&down.slice(..words as u64 * 4).get_mapped_range()).to_vec();
    down.unmap_now();
    let mut st = Stats { hist: [[0; 256]; 3], sum: [0.0; 4], count: 0 };
    for k in 0..3 {
        for b in 0..256 {
            st.hist[k][b] = v[k * 256 + b] as u64;
        }
    }
    let wide = |i: usize| v[768 + 2 * i] as u64 | (v[769 + 2 * i] as u64) << 32;
    for k in 0..4 {
        st.sum[k] = wide(k) as f64;
    }
    st.count = wide(4);
    Ok(st)
}
