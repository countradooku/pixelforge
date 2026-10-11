//! Rank and edge-preserving filters (`filters::rank`, `filters::fast`):
//! Median, Dust & Scratches, Despeckle, Maximum / Minimum, Smart Blur,
//! Surface Blur and Reduce Noise.
//!
//! - Maximum / Minimum: the exact window max / min (what both
//!   `fast::maxmin_small` and van Herk compute): a shared-memory tile for
//!   r <= 8, else separable passes with doubling in shared memory.
//! - 3x3 / 5x5 medians (Median, Dust & Scratches r <= 2, Despeckle):
//!   sorting networks on a shared-memory tile.
//! - Medians and Dust & Scratches with r >= 3, Smart Blur with r > 6:
//!   sliding column histograms per strip of columns, combined across the
//!   strip with prefix sums (see `wgsl/rank_hist.wgsl`). Radius <= 127
//!   (UI maximum 100).
//! - Smart Blur r <= 6: the whole window per output from a shared tile.
//! - Surface Blur / Reduce Noise: Yang's PBFIC as the CPU runs it (same
//!   levels, same tent weights), per output tile over the levels its values
//!   span. Float: within 1/255 of the CPU (its running sums drift
//!   differently).
//!
//! All but Surface Blur / Reduce Noise are integer-defined and bit-exact.
//! Metal note: private arrays indexed in loops spill to memory (a 5x5
//! median loop was 4.5x slower than its unrolled form), so the kernels keep
//! state in vectors and unrolled code.

use std::sync::Arc;

use super::{Io, Kernel, Temps};
use crate::engine::filters::Filter;
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, Timer, dispatch, groups};
use crate::engine::gpu::shaders::PipeKey;

pub fn supports(f: &Filter) -> bool {
    matches!(
        f,
        Filter::Median { .. }
            | Filter::DustScratches { .. }
            | Filter::Despeckle
            | Filter::Maximum { .. }
            | Filter::Minimum { .. }
            | Filter::SmartBlur { .. }
            | Filter::SurfaceBlur { .. }
            | Filter::ReduceNoise { .. }
    )
}

/// `r_at` of `filters`: the radius at a mip level.
fn r_at(v: f32, level: u32) -> usize {
    (v / (1u32 << level) as f32).round().max(0.0) as usize
}

enum Op {
    Copy,
    MaxMin { max: bool },
    /// 3x3 / 5x5 medians: mode 0 median, 1 dust (threshold), 2 despeckle.
    Small { mode: u32, thr: u32 },
    /// Sliding column histograms: mode 0 median, 1 dust (threshold), 2
    /// smart blur (threshold).
    Hist { mode: u32, thr: u32 },
    /// Smart Blur: mean of the window values within `ti` of the center.
    Smart { ti: u32 },
    /// Surface Blur / Reduce Noise: PBFIC with `levels` levels.
    Surface { levels: u32, step: f32, inv_band: f32 },
}

/// The CPU's PBFIC parameters (`rank::surface`): level count, level step
/// and 1 / band, computed with the same f32 operations.
fn pbfic_params(threshold: f32) -> (u32, f32, f32) {
    let t = threshold.max(1.0);
    let band = 2.5 * t;
    let k = ((255.0 / (band * 0.4)).ceil() as usize + 1).clamp(3, 96);
    (k as u32, 255.0 / (k - 1) as f32, 1.0 / band)
}

pub struct RankK {
    op: Op,
    r: u32,
}

pub fn plan(_g: &Arc<Gpu>, f: &Filter, level: u32) -> Option<Box<dyn Kernel>> {
    if !supports(f) {
        return None;
    }
    let k = |op, r: usize| Some(Box::new(RankK { op, r: r as u32 }) as Box<dyn Kernel>);
    match *f {
        Filter::Maximum { radius } | Filter::Minimum { radius } => {
            let r = r_at(radius, level);
            if r == 0 {
                return k(Op::Copy, 0);
            }
            if r > 990 {
                return None;
            }
            k(Op::MaxMin { max: matches!(f, Filter::Maximum { .. }) }, r)
        }
        Filter::Despeckle => k(Op::Small { mode: 2, thr: 0 }, 1),
        Filter::Median { radius } => {
            let r = r_at(radius, level);
            match r {
                0 => k(Op::Copy, 0),
                1 | 2 => k(Op::Small { mode: 0, thr: 0 }, r),
                3..=127 => k(Op::Hist { mode: 0, thr: 0 }, r),
                _ => None,
            }
        }
        Filter::DustScratches { radius, threshold } => {
            let r = r_at(radius, level);
            let thr = threshold as u8 as u32;
            match r {
                0 => k(Op::Copy, 0),
                1 | 2 => k(Op::Small { mode: 1, thr }, r),
                3..=127 => k(Op::Hist { mode: 1, thr }, r),
                _ => None,
            }
        }
        Filter::SmartBlur { radius, threshold } => {
            let r = r_at(radius, level);
            let ti = (threshold.max(1.0).floor() as i32).clamp(0, 255) as u32;
            match r {
                0 => k(Op::Copy, 0),
                1..=SMART_DIRECT_R => k(Op::Smart { ti }, r),
                _ if r <= 120 => k(Op::Hist { mode: 2, thr: ti }, r),
                _ => None,
            }
        }
        Filter::SurfaceBlur { radius, threshold } => {
            let r = r_at(radius, level);
            let (levels, step, inv_band) = pbfic_params(threshold);
            match r {
                0 => k(Op::Copy, 0),
                1..=120 if surface_ms(r as u32, levels, step) <= 250.0 => k(Op::Surface { levels, step, inv_band }, r),
                _ => None,
            }
        }
        Filter::ReduceNoise { strength } => {
            let (levels, step, inv_band) = pbfic_params(4.0 + strength * 2.5);
            k(Op::Surface { levels, step, inv_band }, 3)
        }
        _ => None,
    }
}

/// Up to this radius Smart Blur reads the whole window per output (a
/// shared-memory tile); beyond it, sliding column histograms.
const SMART_DIRECT_R: usize = 6;

/// Predicted Surface Blur kernel time per 16.7 MP (M5 Pro; measured 41 ms
/// at r 20 / t 15, 137 at r 60 / t 15, 149 at r 20 / t 2): the tile running
/// sums times the levels a tile spans (~40 values of a photo). Above 250 ms
/// (big radii with low thresholds) the CPU, whose cost doesn't grow with
/// the radius, is faster, and a 64 MP band would near the GPU watchdog.
fn surface_ms(r: u32, levels: u32, step: f32) -> f32 {
    let (tw, th, per) = surface_tile(r);
    let n = (2 * r + 1) as f32;
    let cost = (th + 2 * r) as f32 * (n + 2.0 * tw as f32) / (tw * th) as f32 + (n + 2.0 * per as f32) / per as f32;
    let span = (40.0 / step + 2.0).min(levels as f32);
    0.45 * span * cost
}

/// Surface Blur tile (width, height, outputs per vertical walker): the
/// cheapest per output of the running sums, within the shared arrays
/// (rows x width <= 2560, width x height <= 1024).
fn surface_tile(r: u32) -> (u32, u32, u32) {
    let n = 2 * r + 1;
    let mut best = (f64::MAX, (8, 8, 1));
    for tw in [64u32, 32, 16, 8, 4] {
        let th = (1024 / tw).min((2560 / tw).saturating_sub(2 * r));
        if th < 4 {
            continue;
        }
        let per = (tw * th).div_ceil(256);
        let h = (th + 2 * r) as f64 * (n + 2 * tw) as f64 / (tw * th) as f64;
        let v = (n + 2 * per) as f64 / per as f64;
        if h + v < best.0 {
            best = (h + v, (tw, th, per));
        }
    }
    best.1
}

fn key(entry: &'static str) -> PipeKey {
    PipeKey { module: "rank", entry, f16: false, consts: Vec::new() }
}

impl Kernel for RankK {
    fn temps(&self) -> usize {
        match self.op {
            Op::MaxMin { .. } if self.r > 8 => 1,
            _ => 0,
        }
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        debug_assert_eq!(io.pad, self.r);
        let (cw, ch) = (io.cw(), io.ch());
        let r = self.r;
        let params = |mode: u32, thr: u32, seg: u32, tw: u32| [io.pw, io.ph, cw, ch, r, mode, thr, seg, tw, 0, 0, 0];
        match self.op {
            Op::Copy => dispatch(g, enc, timer, &key("copy_center"), &params(0, 0, 0, 0), &[(1, io.src), (2, io.dst)], [groups(cw, 16), groups(ch, 16), 1]),
            Op::Small { mode, thr } => {
                dispatch(g, enc, timer, &key("median_small"), &params(mode, thr, 0, 0), &[(1, io.src), (2, io.dst)], [groups(cw, 16), groups(ch, 16), 1])
            }
            Op::Hist { mode, thr } => {
                // One column per thread, a strip of 256 (or 512 for medians
                // with r > 63; Smart Blur's sums need the shared memory).
                let s = if r <= 63 || mode == 2 { 256 } else { 512 };
                let tw = s - 2 * r;
                let n = 2 * r + 1;
                let nstrips = cw.div_ceil(tw);
                // Enough workgroups to fill the GPU; chunks long enough to
                // amortize the 2r + 1 rows each one starts with.
                let min_th = (4 * n).max(64);
                let mut nchunks = 1024u32.div_ceil(nstrips * 4).min(ch.div_ceil(min_th)).max(1);
                let th = ch.div_ceil(nchunks);
                nchunks = ch.div_ceil(th);
                let scratch = nstrips as u64 * nchunks as u64 * 4 * 64 * s as u64 * 4;
                tmp.ensure_sizes(g, &[scratch])?;
                let module = if g.caps.subgroups { "rank_hist" } else { "rank_hist_wg" };
                let hkey = |entry| PipeKey { module, entry, f16: false, consts: vec![("WG", s)] };
                let hp = [io.pw, io.ph, cw, ch, r, mode, thr, s, tw, th, nstrips, nchunks];
                let npx = cw * ch;
                let gx = groups(npx, 256).min(65535);
                let gy = groups(npx, gx * 256);
                dispatch(g, enc, timer.as_deref_mut(), &hkey("zero_out"), &hp, &[(2, io.dst)], [gx, gy, 1])?;
                dispatch(g, enc, timer.as_deref_mut(), &hkey(if mode == 2 { "hist_smart" } else { "hist_median" }), &hp, &[(1, io.src), (2, io.dst), (3, tmp.at(0))], [nstrips, nchunks, 4])?;
                dispatch(g, enc, timer, &hkey("clamp_out"), &hp, &[(2, io.dst)], [gx, gy, 1])
            }
            Op::Smart { ti } => {
                let sp = [io.pw, io.ph, cw, ch, r, ti, 0, 0, 0, 0, 0, 0];
                let key = PipeKey { module: "rank_surf", entry: "smart_direct", f16: false, consts: Vec::new() };
                dispatch(g, enc, timer, &key, &sp, &[(1, io.src), (2, io.dst)], [groups(cw, 16), groups(ch, 16), 1])
            }
            Op::Surface { levels, step, inv_band } => {
                let (tw, th, per) = surface_tile(r);
                let sp = [io.pw, io.ph, cw, ch, r, levels, tw, th, step.to_bits(), inv_band.to_bits(), per, 0];
                let hk = |entry| PipeKey { module: if g.caps.subgroups { "rank_hist" } else { "rank_hist_wg" }, entry, f16: false, consts: Vec::new() };
                let key = PipeKey { module: "rank_surf", entry: "surface", f16: false, consts: Vec::new() };
                let npx = cw * ch;
                let gx = groups(npx, 256).min(65535);
                let gy = groups(npx, gx * 256);
                dispatch(g, enc, timer.as_deref_mut(), &hk("zero_out"), &sp, &[(2, io.dst)], [gx, gy, 1])?;
                dispatch(g, enc, timer.as_deref_mut(), &key, &sp, &[(1, io.src), (4, io.dst)], [groups(cw, tw), groups(ch, th), 4])?;
                dispatch(g, enc, timer, &hk("clamp_out"), &sp, &[(2, io.dst)], [gx, gy, 1])
            }
            Op::MaxMin { max } => {
                let mode = max as u32;
                if r <= 8 {
                    return dispatch(g, enc, timer, &key("maxmin_tile"), &params(mode, 0, 0, 0), &[(1, io.src), (2, io.dst)], [groups(cw, 32), groups(ch, 32), 1]);
                }
                let n = 2 * r + 1;
                // Horizontal: up to 2048 values per row run (two buffers).
                let seg = 2048 - (n - 1);
                let seg = seg.min(1024);
                let t = tmp.at(0);
                dispatch(g, enc, timer.as_deref_mut(), &key("maxmin_h"), &params(mode, 0, seg, 0), &[(1, io.src), (2, t)], [groups(cw, seg), io.ph, 1])?;
                // Vertical: strips as wide as fit with >= 64 outputs per column.
                let mut tw = 64;
                while tw > 1 && 2048 / tw < n - 1 + 64 {
                    tw /= 2;
                }
                let seg = 2048 / tw - (n - 1);
                dispatch(g, enc, timer, &key("maxmin_v"), &params(mode, 0, seg, tw), &[(2, io.dst), (3, t)], [groups(cw, tw), groups(ch, seg), 1])
            }
        }
    }
}
