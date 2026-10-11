//! The healing membrane solve (`engine::poisson::blend_rgba_q`) on the GPU.
//!
//! The CPU solver's control flow is replayed on the host: the same nested
//! iteration, multigrid hierarchy (masks coarsened by the same integer rule),
//! V-cycles, Fast/Exact branches and boundary band. The masks are computed
//! here exactly as the CPU computes them and uploaded; the float work runs in
//! `wgsl/poisson.wgsl`, recorded as one [`Prog`]. The only data-dependent
//! branch, "stop once the max residual is below `tol`", is decided on the GPU
//! (guard slots, see the shader), so the whole solve is one submission with
//! no read-back in between.
//!
//! Red-black Gauss–Seidel is order-independent within a colour, so the GPU
//! iterates the CPU's exact recurrence; results differ only by float rounding
//! (fast-math contraction), far below the 8-bit output's resolution.

use std::sync::Arc;

use super::device::Gpu;
use super::exec::{self, GResult, GpuError, STORAGE, buffer, encoder, groups, submit, wait};
use super::prog::{Groups, Prog};
use crate::engine::poisson::{EXACT_TOL, Quality};

/// One level (or plane) in the pools: cell offset, size, mask byte offset.
#[derive(Clone, Copy, Debug)]
struct Lv {
    off: u32,
    w: u32,
    h: u32,
    moff: u32,
}

impl Lv {
    fn words(&self) -> [u32; 4] {
        [self.off, self.w, self.h, self.moff]
    }
    fn cells(&self) -> u32 {
        self.w * self.h
    }
}

const NONE: Lv = Lv { off: 0, w: 0, h: 0, moff: 0 };
/// Band width and sweeps of the fast extension (`poisson::BAND`, `SWEEPS`).
const BAND: u32 = 4;
const SWEEPS: usize = 12;

/// Levels at most this many cells run whole V-cycles in one workgroup.
fn small_cells() -> u32 {
    static S: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("PF_POISSON_SMALL").ok().and_then(|v| v.parse().ok()).unwrap_or(128 * 128))
}

#[allow(clippy::too_many_arguments)]
fn params(guard: u32, tol: f32, slot: u32, color: u32, a: Lv, b: Lv, k: [u32; 4], f: [f32; 4], lv: &[Lv]) -> Vec<u32> {
    let mut v = vec![guard, tol.to_bits(), slot, color];
    v.extend(a.words());
    v.extend(b.words());
    v.extend(k);
    v.extend(f.map(f32::to_bits));
    for l in lv {
        v.extend(l.words());
    }
    v
}

/// `Level::guard_border`.
fn guard_border(m: &mut [u8], w: usize, h: usize) {
    for x in 0..w {
        m[x] = 0;
        m[(h - 1) * w + x] = 0;
    }
    for y in 0..h {
        m[y * w] = 0;
        m[y * w + w - 1] = 0;
    }
}

/// The CPU's coarse mask: a cell is unknown when >= 3 of its children are
/// (u8 sums, as the CPU adds them).
fn coarse_mask(m: &[u8], w: usize, h: usize) -> Vec<u8> {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut cm = vec![0u8; cw * ch];
    for y in 0..ch {
        for x in 0..cw {
            let mut k = 0u8;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (fx, fy) = (2 * x + dx, 2 * y + dy);
                if fx < w && fy < h {
                    k = k.wrapping_add(m[fy * w + fx]);
                }
            }
            cm[y * cw + x] = (k >= 3) as u8;
        }
    }
    cm
}

fn count(m: &[u8]) -> usize {
    m.iter().filter(|&&v| v != 0).count()
}

/// Host-side replay of the CPU solver, recording GPU ops.
struct Planner {
    cells: u32,
    masks: Vec<u8>,
    /// Guard / residual slots in `ctrl` (slot 0 = +inf).
    slots: u32,
    prog: Prog,
    small: u32,
}

impl Planner {
    fn alloc(&mut self, n: u32) -> u32 {
        let o = self.cells;
        self.cells += n;
        o
    }

    fn mask(&mut self, m: &[u8]) -> u32 {
        let o = self.masks.len() as u32;
        self.masks.extend_from_slice(m);
        self.masks.resize(self.masks.len().next_multiple_of(4), 0);
        o
    }

    fn slots(&mut self, n: u32) -> u32 {
        let o = self.slots;
        self.slots += n;
        o
    }

    fn grid(n: Lv) -> Groups {
        Groups::Direct([groups(n.w, 16), groups(n.h, 16), 1])
    }

    fn smooth(&mut self, l: Lv, sweeps: usize, guard: u32, tol: f32) {
        let gr = Groups::Direct([groups(l.w.div_ceil(2), 16), groups(l.h, 16), 1]);
        for _ in 0..sweeps {
            for c in 0..2 {
                self.prog.op("mg_smooth", &params(guard, tol, 0, c, l, NONE, [0; 4], [0.0; 4], &[]), gr);
            }
        }
    }

    fn residual(&mut self, l: Lv, slot: u32, guard: u32, tol: f32) {
        self.prog.op("mg_residual", &params(guard, tol, slot, 0, l, NONE, [0; 4], [0.0; 4], &[]), Self::grid(l));
    }

    /// `Solver::vcycle(l)`.
    fn vcycle(&mut self, levels: &[Lv], l: usize, guard: u32, tol: f32) {
        let n = levels.len();
        if levels[l].cells() <= self.small {
            let lv = &levels[l..];
            self.prog.op("vsmall", &params(guard, tol, 0, 0, NONE, NONE, [lv.len() as u32, 0, 0, 0], [0.0; 4], lv), Groups::Direct([1, 1, 1]));
            return;
        }
        if l == n - 1 {
            self.smooth(levels[l], 40, guard, tol);
            return;
        }
        let (f, c) = (levels[l], levels[l + 1]);
        self.smooth(f, 3, guard, tol);
        self.residual(f, 0, guard, tol);
        self.prog.op("mg_restrict", &params(guard, tol, 0, 0, f, c, [0; 4], [0.0; 4], &[]), Self::grid(c));
        self.vcycle(levels, l + 1, guard, tol);
        self.prog.op("mg_prolong", &params(guard, tol, 0, 0, f, c, [0; 4], [0.0; 4], &[]), Self::grid(f));
        self.smooth(f, 3, guard, tol);
    }

    /// `Solver::solve_v4` on the plane at `off` (its U and F), whose raw
    /// mask is `raw` (stored at `raw_moff`).
    #[allow(clippy::too_many_arguments)]
    fn solve(&mut self, off: u32, w: usize, h: usize, raw: &[u8], raw_moff: u32, cycles: usize, tol: f32) {
        let plane = Lv { off, w: w as u32, h: h as u32, moff: raw_moff };
        if count(raw) > 4096 && w > 16 && h > 16 {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            let mc = coarse_mask(raw, w, h);
            let coarse = Lv { off: self.alloc((cw * ch) as u32), w: cw as u32, h: ch as u32, moff: self.mask(&mc) };
            self.prog.op("nested_down", &params(0, 0.0, 0, 0, plane, coarse, [1, 0, 0, 0], [0.0; 4], &[]), Self::grid(coarse));
            self.solve(coarse.off, cw, ch, &mc, coarse.moff, 1, tol);
            self.prog.op("prolong_set", &params(0, 0.0, 0, 0, plane, coarse, [0; 4], [0.0; 4], &[]), Self::grid(plane));
        }
        // `Solver::setup`.
        let mut m = raw.to_vec();
        guard_border(&mut m, w, h);
        let mut levels = vec![Lv { off, w: w as u32, h: h as u32, moff: self.mask(&m) }];
        let mut cnt = count(&m);
        loop {
            let prev = *levels.last().unwrap();
            if prev.w <= 4 || prev.h <= 4 || cnt <= 16 {
                break;
            }
            let (pw, ph) = (prev.w as usize, prev.h as usize);
            let mut cm = coarse_mask(&m, pw, ph);
            let (cw, ch) = (pw.div_ceil(2), ph.div_ceil(2));
            guard_border(&mut cm, cw, ch);
            cnt = count(&cm);
            let lv = Lv { off: self.alloc((cw * ch) as u32), w: cw as u32, h: ch as u32, moff: self.mask(&cm) };
            levels.push(lv);
            m = cm;
        }
        let n = cycles.max(1) as u32;
        let base = self.slots(n + 1);
        self.residual(levels[0], base, 0, 0.0);
        for c in 0..n {
            let guard = base + c;
            self.vcycle(&levels, 0, guard, tol);
            self.residual(levels[0], base + c + 1, guard, tol);
        }
    }
}

/// Where the GPU data lives for one blend.
struct Layout {
    src: u32,
    dst: u32,
    out: u32,
}

/// The blended crop (`cw x ch`, packed premultiplied), the GPU twin of the
/// CPU's crop solve in `poisson::blend_rgba_q`. `s`, `d`: the crop of src and
/// dst; `cm`: the crop of the mask.
#[allow(clippy::too_many_arguments)]
pub fn blend_crop(g: &Arc<Gpu>, s: &[u32], d: &[u32], cm: &[u8], cw: usize, ch: usize, cycles: usize, q: Quality) -> GResult<Vec<u32>> {
    if g.is_failed() {
        return Err(GpuError::Lost(g.failure().unwrap_or_default()));
    }
    let t0 = std::time::Instant::now();
    let n = cw * ch;
    // Initial guess: the mean boundary difference. Integer sums (exact, like
    // the CPU's f32 sums of integers), then the CPU's `sum * (1 / n)`.
    let (sum, nb) = {
        let mut sum = [0i64; 3];
        let mut nb = 0usize;
        for y in 0..ch {
            for x in 0..cw {
                let i = y * cw + x;
                if cm[i] != 0 {
                    continue;
                }
                let nbr = (x > 0 && cm[i - 1] != 0) || (x + 1 < cw && cm[i + 1] != 0) || (y > 0 && cm[i - cw] != 0) || (y + 1 < ch && cm[i + cw] != 0);
                if nbr {
                    for c in 0..3 {
                        sum[c] += ((d[i] >> (8 * c)) & 255) as i64 - ((s[i] >> (8 * c)) & 255) as i64;
                    }
                    nb += 1;
                }
            }
        }
        (sum, nb)
    };
    let mean = if nb > 0 {
        let k = 1.0 / nb as f32;
        [sum[0] as f32 * k, sum[1] as f32 * k, sum[2] as f32 * k, 0.0]
    } else {
        [0.0; 4]
    };
    let unknowns = count(cm);
    let mut prog = Prog::new("poisson");
    if g.limits.max_compute_invocations_per_workgroup >= 1024 && g.limits.max_compute_workgroup_size_x >= 1024 {
        prog.consts.push(("WG", 1024));
    }
    let mut pl = Planner { cells: 0, masks: Vec::new(), slots: 1, prog, small: small_cells() };
    let top = Lv { off: pl.alloc(n as u32), w: cw as u32, h: ch as u32, moff: pl.mask(cm) };
    let lay = Layout { src: 0, dst: n as u32, out: 2 * n as u32 };
    pl.prog.op("init", &params(0, 0.0, 0, 0, top, NONE, [lay.src, lay.dst, 0, 0], mean, &[]), Planner::grid(top));
    // Band lists (fast extension only): hk plane, then one list per colour.
    let mut aux_words = 4u64;
    if q == Quality::Exact {
        pl.solve(top.off, cw, ch, cm, top.moff, cycles.max(12), EXACT_TOL);
    } else if unknowns > 16_384 && cw > 32 && ch > 32 {
        // `harmonic_fast`.
        let (hw, hh) = (cw.div_ceil(2), ch.div_ceil(2));
        let mc = coarse_mask(cm, cw, ch);
        let coarse = Lv { off: pl.alloc((hw * hh) as u32), w: hw as u32, h: hh as u32, moff: pl.mask(&mc) };
        pl.prog.op("nested_down", &params(0, 0.0, 0, 0, top, coarse, [0; 4], [0.0; 4], &[]), Planner::grid(coarse));
        pl.solve(coarse.off, hw, hh, &mc, coarse.moff, cycles, 0.25);
        pl.prog.op("prolong_set", &params(0, 0.0, 0, 0, top, coarse, [0; 4], [0.0; 4], &[]), Planner::grid(top));
        let (l0, l1) = (n as u32, (n + unknowns) as u32);
        aux_words = (n + 2 * unknowns) as u64;
        let cnt = pl.slots(2);
        let indirect_at = pl.slots(6);
        pl.prog.op("band_h", &params(0, 0.0, 0, 0, top, NONE, [0, BAND, 0, 0], [0.0; 4], &[]), Planner::grid(top));
        pl.prog.op("band_list", &params(0, 0.0, cnt, 0, top, NONE, [0, BAND, l0, l1], [0.0; 4], &[]), Planner::grid(top));
        pl.prog.op("band_args", &params(0, 0.0, cnt, 0, NONE, NONE, [indirect_at, 0, 0, 0], [0.0; 4], &[]), Groups::Direct([1, 1, 1]));
        pl.prog.copy_indirect(5, indirect_at as u64 * 4, 0, 24);
        for _ in 0..SWEEPS {
            for c in 0..2u32 {
                pl.prog.op("band_sweep", &params(0, 0.0, cnt, c, top, NONE, [0, 0, l0, l1], [0.0; 4], &[]), Groups::Indirect(3 * c as u64 * 4));
            }
        }
    } else {
        pl.solve(top.off, cw, ch, cm, top.moff, cycles, 0.25);
    }
    pl.prog.op("finish", &params(0, 0.0, 0, 0, top, NONE, [lay.src, lay.dst, lay.out, 0], [0.0; 4], &[]), Planner::grid(top));

    let t_plan = t0.elapsed().as_secs_f64() * 1e3;
    // Buffers.
    let cells = pl.cells as u64;
    let plane = |words: u64| buffer(g, words.max(4) * 16, STORAGE);
    let (bu, bf, br) = (plane(cells)?, plane(cells)?, plane(cells)?);
    let bm = buffer(g, pl.masks.len().max(16) as u64, STORAGE)?;
    let ctrl_words = (pl.slots as u64).max(4);
    let bc = buffer(g, ctrl_words * 4, STORAGE)?;
    let bi = buffer(g, 3 * n as u64 * 4, STORAGE)?;
    let ba = buffer(g, aux_words * 4, STORAGE)?;
    let bind = buffer(g, 32, STORAGE | wgpu::BufferUsages::INDIRECT)?;
    let down = buffer(g, n as u64 * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST)?;
    let scopes = exec::Scopes::push(g);
    g.queue.write_buffer(&bm, 0, &pl.masks);
    let mut ctrl = vec![0u32; ctrl_words as usize];
    ctrl[0] = f32::INFINITY.to_bits();
    g.queue.write_buffer(&bc, 0, bytemuck::cast_slice(&ctrl));
    g.queue.write_buffer(&bi, lay.src as u64 * 4, bytemuck::cast_slice(s));
    g.queue.write_buffer(&bi, lay.dst as u64 * 4, bytemuck::cast_slice(d));
    let mut enc = encoder(g);
    let mut timer = super::prog::profile_timer(g, &pl.prog);
    pl.prog.encode(g, &mut enc, &[(1, &bu), (2, &bf), (3, &br), (4, &bm), (5, &bc), (6, &bi), (7, &ba)], Some(&bind), timer.as_mut())?;
    if let Some(t) = timer.as_mut() {
        t.resolve(g, &mut enc);
    }
    enc.copy_buffer_to_buffer(&bi, lay.out as u64 * 4, &down, 0, n as u64 * 4);
    let t_enc = t0.elapsed().as_secs_f64() * 1e3;
    let idx = submit(g, enc);
    scopes.pop()?;
    wait(g, idx)?;
    let t_wait = t0.elapsed().as_secs_f64() * 1e3;
    super::prog::report(g, &format!("poisson {cw}x{ch}"), timer.as_ref());
    down.map(n as u64 * 4, wgpu::MapMode::Read)?;
    let out = {
        let view = down.slice(..n as u64 * 4).get_mapped_range();
        bytemuck::cast_slice::<u8, u32>(&view).to_vec()
    };
    down.unmap_now();
    if std::env::var_os("PF_GPU_TRACE").is_some() {
        eprintln!(
            "gpu poisson {cw}x{ch}: {} ops, {} cells, {} slots; plan {t_plan:.2} ms, encode {:.2} ms, gpu {:.2} ms, read {:.2} ms",
            pl.prog.len(),
            pl.cells,
            pl.slots,
            t_enc - t_plan,
            t_wait - t_enc,
            t0.elapsed().as_secs_f64() * 1e3 - t_wait
        );
    }
    Ok(out)
}
