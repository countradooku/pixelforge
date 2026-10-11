//! PatchMatch completion (`engine::patchmatch::complete`) on the GPU.
//!
//! The CPU builds the pyramid (valid-source masks, means, focus penalty) and
//! solves the coarse levels; from a large level on, this runs the same EM
//! steps per level as the CPU — seed the hole from the coarser result,
//! upsample the NNF, vote, rescore, then `iters` x (PatchMatch pass, vote,
//! rescore), and the sharp final pass — recorded as one [`Prog`] and
//! submitted once. See `wgsl/patchmatch.wgsl` for the kernels.
//!
//! The CPU's PatchMatch pass propagates along scanlines, which is inherently
//! sequential, so the GPU propagates by jump flooding (log-step passes over
//! the four axis neighbours) followed by the CPU's random search. The NNF
//! therefore differs in detail from the CPU's; the quality bar (tests) is
//! the final mean patch distance and the fill's SSIM against the CPU fill.

use std::sync::Arc;

use rayon::prelude::*;

use super::device::Gpu;
use super::exec::{self, GResult, GpuError, STORAGE, buffer, encoder, groups, submit, wait};
use super::prog::{Groups, Prog};
use crate::engine::patchmatch::{Lvl, MatchStats, Nnf, Params, targets_of};

const NONE: u32 = u32::MAX;
const CTRL_WORDS: u64 = 131_076;

/// One level's places in the pools (the shader's `L`).
#[derive(Clone, Copy, Debug, Default)]
struct Lv {
    w: u32,
    h: u32,
    n: u32,
    flg: u32,
    mean: u32,
    prox: u32,
    vl: u32,
    vln: u32,
    /// Image planes: current, other.
    img: [u32; 2],
    /// NNF planes (source, meta, distance at +0, +n, +2n): current, other.
    ns: [u32; 2],
    wt: u32,
    /// Target box and hole box grown by R: (x0, y0, x1, y1).
    tb: [u32; 4],
    mb: [u32; 4],
}

impl Lv {
    fn words(&self) -> [u32; 24] {
        let (t, m) = (self.tb, self.mb);
        [self.w, self.h, self.n, self.flg, self.mean, self.prox, self.vl, self.vln, self.img[0], self.img[1], self.ns[0], self.ns[1], self.wt, 0, 0, 0, t[0], t[1], t[2], t[3], m[0], m[1], m[2], m[3]]
    }
    fn swap_img(&mut self) {
        self.img.swap(0, 1);
    }
    fn swap_nn(&mut self) {
        self.ns.swap(0, 1);
    }
    fn grid(&self) -> Groups {
        Groups::Direct([groups(self.w, 16), groups(self.h, 16), 1])
    }
    fn tgrid(&self) -> Groups {
        Groups::Direct([groups(self.tb[2] - self.tb[0], 16), groups(self.tb[3] - self.tb[1], 16), 1])
    }
    fn mgrid(&self) -> Groups {
        Groups::Direct([groups(self.mb[2] - self.mb[0], 16), groups(self.mb[3] - self.mb[1], 16), 1])
    }
}

struct Ctx {
    prog: Prog,
    nxf: u32,
    bias: i32,
    reach: i32,
}

impl Ctx {
    #[allow(clippy::too_many_arguments)]
    fn op(&mut self, entry: &'static str, a: &Lv, b: &Lv, seed: u32, step: u32, maxr: u32, center: bool, gr: Groups) {
        let mut v = Vec::with_capacity(56);
        v.extend(a.words());
        v.extend(b.words());
        v.extend([seed, step, self.nxf, self.bias as u32, self.reach as u32, maxr, center as u32, 0]);
        self.prog.op(entry, &v, gr);
    }

    /// `vote(l, nnf, center)` with its sigma^2 and the mean update.
    fn vote(&mut self, a: &mut Lv, center: bool) {
        let z = Lv::default();
        if !center {
            self.op("clear_bins", a, &z, 0, 0, 0, false, Groups::Direct([groups(CTRL_WORDS as u32, 256), 1, 1]));
            self.op("hist1", a, &z, 0, 0, 0, false, a.tgrid());
            self.op("scan1", a, &z, 0, 0, 0, false, Groups::Direct([1, 1, 1]));
            self.op("hist2", a, &z, 0, 0, 0, false, a.tgrid());
            self.op("scan2", a, &z, 0, 0, 0, false, Groups::Direct([1, 1, 1]));
            self.op("weights", a, &z, 0, 0, 0, false, a.tgrid());
        }
        self.op("vote", a, &z, 0, 0, 0, center, a.mgrid());
        a.swap_img();
        if self.bias != 0 {
            self.op("box_mean", a, &z, 0, 0, 0, false, a.mgrid());
        }
    }

    fn rescore(&mut self, a: &Lv) {
        self.op("rescore", a, &Lv::default(), 0, 0, 0, false, a.tgrid());
    }

    /// One PatchMatch pass: jump-flooding propagation, then random search.
    fn pm_pass(&mut self, a: &mut Lv, seed: u32) {
        let z = Lv::default();
        let m = a.w.max(a.h);
        // Jumps never need to exceed the targets' extent.
        let ext = (a.tb[2] - a.tb[0]).max(a.tb[3] - a.tb[1]).max(2);
        let mut step = 1u32 << (31 - (ext / 2).max(1).leading_zeros());
        let min_step = jfa_min_step();
        while step >= 1 {
            if step >= min_step {
                self.op("prop", a, &z, 0, step, 0, false, a.tgrid());
                a.swap_nn();
            }
            step /= 2;
        }
        self.op("search", a, &z, seed, 0, m, false, a.tgrid());
    }
}

/// `PF_PM_MIN_STEP` (tuning): the smallest jump-flooding step.
fn jfa_min_step() -> u32 {
    std::env::var("PF_PM_MIN_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
}

fn pack(c: [u8; 4]) -> u32 {
    c[0] as u32 | (c[1] as u32) << 8 | (c[2] as u32) << 16 | 0xFF00_0000
}

/// Runs levels `from..=0` of a completion whose CPU state (pyramid, and the
/// NNF and image of level `from + 1`) is given. Returns level 0's final
/// image (packed straight RGB, alpha 255) and, with `want_stats`, the final
/// NNF statistics.
pub fn run(g: &Arc<Gpu>, levels: &[Lvl], coarse_nnf: &Nnf, from: usize, p: &Params, want_stats: bool) -> GResult<(Vec<u32>, Option<MatchStats>)> {
    if g.is_failed() {
        return Err(GpuError::Lost(g.failure().unwrap_or_default()));
    }
    let t0 = std::time::Instant::now();
    let xf = levels[0].xf.clone();
    let nxf = xf.len() as u32;
    // Pool layout.
    let (mut img_w, mut lv_w, mut nn_w, mut wt_w) = (0u32, 0u32, 0u32, 0u32);
    let take = |c: &mut u32, n: u32| {
        let o = *c;
        *c += n;
        o
    };
    let mut lvs: Vec<Lv> = Vec::with_capacity(from + 2);
    for l in &levels[..=from] {
        let n = (l.w * l.h) as u32;
        let mut v = Lv { w: l.w as u32, h: l.h as u32, n, ..Default::default() };
        v.img = [take(&mut img_w, n), take(&mut img_w, n)];
        v.flg = take(&mut lv_w, n);
        v.mean = if xf.bias != 0 { take(&mut lv_w, n) } else { 0 };
        v.prox = if l.prox.is_empty() { NONE } else { take(&mut lv_w, n) };
        v.vln = l.valid_list.len() as u32;
        v.vl = take(&mut lv_w, v.vln.max(1));
        v.ns = [take(&mut nn_w, 3 * n), take(&mut nn_w, 3 * n)];
        v.wt = take(&mut wt_w, n);
        lvs.push(v);
    }
    let cl = &levels[from + 1];
    let cn = (cl.w * cl.h) as u32;
    let mut cv = Lv { w: cl.w as u32, h: cl.h as u32, n: cn, ..Default::default() };
    cv.img = [take(&mut img_w, cn), 0];
    cv.ns = [take(&mut nn_w, 3 * cn), 0];

    // Host data.
    let mut img_h = vec![0u32; img_w as usize];
    let mut lv_h = vec![0u32; lv_w.max(4) as usize];
    let mut nn_h = vec![0u32; 3 * cn as usize];
    for (l, v) in levels[..=from].iter().zip(lvs.iter_mut()) {
        let n = v.n as usize;
        let t = targets_of(l);
        let (w, h) = (v.w as usize, v.h as usize);
        let bbox = |m: &[u8]| -> [u32; 4] {
            let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
            for y in 0..h {
                let row = &m[y * w..(y + 1) * w];
                if let Some(a) = row.iter().position(|&c| c != 0) {
                    x0 = x0.min(a);
                    x1 = x1.max(row.iter().rposition(|&c| c != 0).unwrap() + 1);
                    y0 = y0.min(y);
                    y1 = y1.max(y + 1);
                }
            }
            if x1 <= x0 { [0; 4] } else { [x0 as u32, y0 as u32, x1 as u32, y1 as u32] }
        };
        v.tb = bbox(&t);
        let hb = bbox(&l.hole);
        let r = crate::engine::patchmatch::R as u32;
        v.mb = if hb[2] > hb[0] { [hb[0].saturating_sub(r), hb[1].saturating_sub(r), (hb[2] + r).min(v.w), (hb[3] + r).min(v.h)] } else { [0; 4] };
        img_h[v.img[0] as usize..v.img[0] as usize + n].par_iter_mut().zip(l.img.par_iter()).for_each(|(o, &c)| *o = pack(c));
        let wide = !l.valid_wide.is_empty();
        lv_h[v.flg as usize..v.flg as usize + n].par_iter_mut().enumerate().for_each(|(i, o)| {
            *o = (l.hole[i] != 0) as u32 | ((l.valid[i] != 0) as u32) << 1 | ((wide && l.valid_wide[i] != 0) as u32) << 2 | ((t[i] != 0) as u32) << 3;
        });
        if xf.bias != 0 {
            for (i, m) in l.mean.iter().enumerate() {
                lv_h[v.mean as usize + i] = m[0] as u32 | (m[1] as u32) << 8 | (m[2] as u32) << 16;
            }
        }
        if v.prox != NONE {
            lv_h[v.prox as usize..v.prox as usize + n].copy_from_slice(&l.prox);
        }
        lv_h[v.vl as usize..v.vl as usize + l.valid_list.len()].copy_from_slice(&l.valid_list);
    }
    for i in 0..cn as usize {
        img_h[cv.img[0] as usize + i] = pack(cl.img[i]);
        nn_h[i] = coarse_nnf.src[i];
        let s = coarse_nnf.sub[i];
        nn_h[cn as usize + i] = coarse_nnf.xf[i] as u32 | ((s[0] as u8) as u32) << 16 | ((s[1] as u8) as u32) << 24;
        nn_h[2 * cn as usize + i] = coarse_nnf.dist[i];
    }
    let mut xf_h: Vec<f32> = Vec::with_capacity(xf.len() * 102);
    for k in 0..xf.len() {
        for o in xf.offf[k].iter() {
            xf_h.push(o.0);
            xf_h.push(o.1);
        }
        xf_h.extend_from_slice(&xf.mat[k]);
    }

    // The program.
    let mut cx = Ctx { prog: Prog::new("patchmatch"), nxf, bias: xf.bias, reach: xf.reach };
    let mut coarse = cv;
    for li in (0..=from).rev() {
        let mut a = lvs[li];
        let z = Lv::default();
        cx.op("seed_hole", &a, &coarse, 0, 0, 0, false, a.grid());
        if xf.bias != 0 {
            cx.op("box_mean", &a, &z, 0, 0, 0, false, a.grid());
        }
        cx.op("upsample", &a, &coarse, 0, 0, 0, false, a.grid());
        let gx = a.n.div_ceil(256);
        cx.op("copy_img", &a, &z, 0, 0, 0, false, Groups::Direct([gx.min(65535), gx.div_ceil(65535), 1]));
        cx.vote(&mut a, false);
        cx.rescore(&a);
        let top = levels.len() - 1;
        let iters = if li == top { p.iters_coarse } else if li == 0 { p.iters_fine } else { (p.iters_fine + 2).min(p.iters_coarse) };
        for it in 0..iters {
            cx.pm_pass(&mut a, p.seed.wrapping_add((li * 977 + it * 131) as u32));
            cx.vote(&mut a, false);
            cx.rescore(&a);
        }
        if li == 0 && p.sharp {
            cx.pm_pass(&mut a, p.seed ^ 0x5eed);
            cx.vote(&mut a, true);
        }
        lvs[li] = a;
        coarse = a;
    }
    let l0 = lvs[0];
    let t_plan = t0.elapsed().as_secs_f64() * 1e3;

    // Buffers, upload, run, read back.
    let words = |n: u32| (n.max(4) as u64) * 4;
    let b_img = buffer(g, words(img_w), STORAGE)?;
    let b_lv = buffer(g, words(lv_w), STORAGE)?;
    let b_nn = buffer(g, words(nn_w), STORAGE)?;
    let b_wt = buffer(g, words(wt_w), STORAGE)?;
    let b_xf = buffer(g, (xf_h.len() * 4) as u64, STORAGE)?;
    let b_ctrl = buffer(g, CTRL_WORDS * 4, STORAGE)?;
    let n0 = l0.n as u64;
    let read_words = if want_stats { 4 * n0 } else { n0 };
    let down = buffer(g, read_words * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
    let scopes = exec::Scopes::push(g);
    g.queue.write_buffer(&b_img, 0, bytemuck::cast_slice(&img_h));
    g.queue.write_buffer(&b_lv, 0, bytemuck::cast_slice(&lv_h));
    g.queue.write_buffer(&b_nn, cv.ns[0] as u64 * 4, bytemuck::cast_slice(&nn_h));
    g.queue.write_buffer(&b_xf, 0, bytemuck::cast_slice(&xf_h));
    let mut enc = encoder(g);
    let mut timer = super::prog::profile_timer(g, &cx.prog);
    cx.prog.encode(g, &mut enc, &[(1, &b_img), (2, &b_lv), (3, &b_nn), (4, &b_wt), (5, &b_xf), (6, &b_ctrl)], None, timer.as_mut())?;
    if let Some(t) = timer.as_mut() {
        t.resolve(g, &mut enc);
    }
    enc.copy_buffer_to_buffer(&b_img, l0.img[0] as u64 * 4, &down, 0, n0 * 4);
    if want_stats {
        enc.copy_buffer_to_buffer(&b_nn, l0.ns[0] as u64 * 4, &down, n0 * 4, 3 * n0 * 4);
    }
    let t_enc = t0.elapsed().as_secs_f64() * 1e3;
    let idx = submit(g, enc);
    scopes.pop()?;
    wait(g, idx)?;
    let t_wait = t0.elapsed().as_secs_f64() * 1e3;
    super::prog::report(g, &format!("patchmatch {}x{}", l0.w, l0.h), timer.as_ref());
    down.map(read_words * 4, wgpu::MapMode::Read)?;
    let (fill, stats) = {
        let view = down.slice(..read_words * 4).get_mapped_range();
        let v: &[u32] = bytemuck::cast_slice(&view);
        let n = n0 as usize;
        let fill = v[..n].to_vec();
        let stats = want_stats.then(|| {
            let t = targets_of(&levels[0]);
            let (src, meta, dist) = (&v[n..2 * n], &v[2 * n..3 * n], &v[3 * n..4 * n]);
            let mut hist = vec![0usize; xf.len()];
            let (mut dsum, mut cnt) = (0u64, 0u64);
            for i in 0..n {
                if t[i] != 0 && src[i] != NONE {
                    hist[(meta[i] & 0xFFFF) as usize] += 1;
                    dsum += dist[i] as u64;
                    cnt += 1;
                }
            }
            MatchStats { xf_hist: hist, mean_dist: dsum as f64 / cnt.max(1) as f64 }
        });
        (fill, stats)
    };
    down.unmap_now();
    if std::env::var_os("PF_GPU_TRACE").is_some() || std::env::var_os("PF_TIMING").is_some() {
        eprintln!(
            "gpu patchmatch levels {from}..0 ({}x{}): {} ops; plan+upload prep {t_plan:.1} ms, encode {:.1} ms, gpu {:.1} ms, read {:.1} ms",
            l0.w,
            l0.h,
            cx.prog.len(),
            t_enc - t_plan,
            t_wait - t_enc,
            t0.elapsed().as_secs_f64() * 1e3 - t_wait
        );
    }
    Ok((fill, stats))
}
