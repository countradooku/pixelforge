//! Multigrid Poisson solver for seamless blending (healing, patch,
//! content-aware fill and move).
//!
//! Healing pastes source texture `s` and adds a smooth *membrane* `m` so the
//! result `s + m` matches the destination `d` on the region's boundary:
//! `Δm = 0` inside Ω with `m = d − s` outside (Pérez et al., "Poisson Image
//! Editing"). The solver works on any rectangle with a mask of unknown
//! pixels (Ω); everything else is a Dirichlet boundary value.
//!
//! It is a geometric multigrid solver on cell-centered grids: V-cycles with
//! red-black Gauss–Seidel smoothing, full-weighting restriction of the
//! residual (the domain mask coarsens with it) and bilinear prolongation of
//! the correction, down to a level of a few pixels that is solved directly.
//! Each V-cycle cuts the error by ~4–10×, independent of the region size (a
//! Jacobi or plain Gauss–Seidel solver needs O(n²) sweeps for an n-px
//! region). Nested iteration solves the half-resolution problem first, so
//! the full-resolution level starts close to the answer.
//!
//! All colour channels are solved together: every cell is a 4-lane f32
//! vector, so one smoothing update is one SIMD operation, rows of one colour
//! run in parallel, and the hierarchy's buffers are reused between solves.

use rayon::prelude::*;

use super::gpu;

pub type V4 = [f32; 4];

#[inline(always)]
fn add(a: V4, b: V4) -> V4 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}
#[inline(always)]
fn sub(a: V4, b: V4) -> V4 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}
#[inline(always)]
fn mul(a: V4, k: f32) -> V4 {
    [a[0] * k, a[1] * k, a[2] * k, a[3] * k]
}
#[inline(always)]
fn maxabs(a: V4) -> f32 {
    a[0].abs().max(a[1].abs()).max(a[2].abs()).max(a[3].abs())
}
const Z: V4 = [0.0; 4];

/// One multigrid level: unknowns `u`, right-hand side `f`, residual `r`,
/// the mask (1 = unknown) and per-row runs of unknowns.
#[derive(Default)]
struct Level {
    w: usize,
    h: usize,
    u: Vec<V4>,
    f: Vec<V4>,
    r: Vec<V4>,
    mask: Vec<u8>,
    runs: Vec<Vec<(u32, u32)>>,
    count: usize,
}

impl Level {
    /// Clears any unknown on the grid edge (the solver needs a known border).
    fn guard_border(&mut self) {
        let (w, h) = (self.w, self.h);
        for x in 0..w {
            self.mask[x] = 0;
            self.mask[(h - 1) * w + x] = 0;
        }
        for y in 0..h {
            self.mask[y * w] = 0;
            self.mask[y * w + w - 1] = 0;
        }
    }

    fn reset(&mut self, w: usize, h: usize) {
        self.w = w;
        self.h = h;
        for v in [&mut self.u, &mut self.f, &mut self.r] {
            v.clear();
            v.resize(w * h, Z);
        }
        self.mask.clear();
        self.mask.resize(w * h, 0);
    }

    fn build_runs(&mut self) {
        let w = self.w;
        self.runs.resize(self.h, Vec::new());
        self.runs.truncate(self.h);
        let mut count = 0;
        for y in 0..self.h {
            let row = &self.mask[y * w..(y + 1) * w];
            let v = &mut self.runs[y];
            v.clear();
            let mut x = 0;
            while x < w {
                if row[x] != 0 {
                    let s = x;
                    while x < w && row[x] != 0 {
                        x += 1;
                    }
                    v.push((s as u32, x as u32));
                    count += x - s;
                } else {
                    x += 1;
                }
            }
        }
        self.count = count;
    }

    /// Red-black Gauss–Seidel sweeps: u = (Σ neighbours − f) / 4. Outside
    /// the rectangle the value is 0: at level 0 callers give the region a
    /// known border, and on coarse levels the unknown is an error that
    /// vanishes outside the domain. Cells of one colour only read the other
    /// colour, so rows run in parallel.
    fn smooth(&mut self, sweeps: usize) {
        let (w, h) = (self.w, self.h);
        let parallel = self.count >= 160 * 160;
        for _ in 0..sweeps {
            for color in 0..2usize {
                let up = super::grid::SyncPtr(self.u.as_mut_ptr());
                let f = &self.f;
                let runs = &self.runs;
                let row = |y: usize| {
                    for &(x0, x1) in &runs[y] {
                        let mut x = x0 as usize;
                        if (x + y) % 2 != color {
                            x += 1;
                        }
                        while x < x1 as usize {
                            let i = y * w + x;
                            // SAFETY: this colour's cells are written only by
                            // their own row; neighbours are the other colour.
                            // Unknowns never touch the grid edge (see `setup`),
                            // so all four neighbours exist.
                            unsafe {
                                let at = |j: usize| *up.at(j);
                                let (l, r, t, b) = (at(i - 1), at(i + 1), at(i - w), at(i + w));
                                *up.at(i) = mul(sub(add(add(l, r), add(t, b)), *f.get_unchecked(i)), 0.25);
                            }
                            x += 2;
                        }
                    }
                };
                if parallel {
                    (0..h).into_par_iter().with_min_len(16).for_each(row);
                } else {
                    (0..h).for_each(row);
                }
            }
        }
    }

    /// r = f − L u on unknowns (L = 5-point Laplacian), 0 elsewhere.
    /// Returns the max |r| over all lanes.
    fn residual(&mut self) -> f32 {
        let w = self.w;
        let u = &self.u;
        let f = &self.f;
        let runs = &self.runs;
        let parallel = self.count >= 160 * 160;
        let rows = |(y, rr): (usize, &mut [V4])| -> f32 {
            rr.fill(Z);
            let mut worst = 0f32;
            for &(x0, x1) in &runs[y] {
                for x in x0 as usize..x1 as usize {
                    let i = y * w + x;
                    let (l, r, t, b) = (u[i - 1], u[i + 1], u[i - w], u[i + w]);
                    let lu = sub(add(add(l, r), add(t, b)), mul(u[i], 4.0));
                    let res = sub(f[i], lu);
                    rr[x] = res;
                    worst = worst.max(maxabs(res));
                }
            }
            worst
        };
        if parallel {
            self.r.par_chunks_mut(w).with_min_len(16).enumerate().map(rows).reduce(|| 0.0, f32::max)
        } else {
            self.r.chunks_mut(w).enumerate().map(rows).fold(0.0, f32::max)
        }
    }
}

/// A small pool for interactive-size solves: a few threads finish a dab's
/// red-black sweeps sooner than the whole machine, whose barriers wait on
/// the slowest (efficiency) cores.
pub(crate) fn dab_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new().num_threads(4).thread_name(|i| format!("poisson-{i}")).build().expect("poisson pool")
    })
}

/// Reusable multigrid hierarchy (keep one per tool session or thread).
#[derive(Default)]
pub struct Solver {
    levels: Vec<Level>,
    used: usize,
    coarse: Option<Box<Solver>>,
}

impl Solver {
    pub fn new() -> Self {
        Solver::default()
    }

    fn level(&mut self, i: usize) -> &mut Level {
        while self.levels.len() <= i {
            self.levels.push(Level::default());
        }
        &mut self.levels[i]
    }

    /// Builds the hierarchy for a `w x h` problem with `mask` (1 = unknown).
    fn setup(&mut self, w: usize, h: usize, mask: &[u8]) {
        let l0 = self.level(0);
        l0.reset(w, h);
        l0.mask.copy_from_slice(mask);
        l0.guard_border();
        l0.build_runs();
        self.used = 1;
        loop {
            let prev = &self.levels[self.used - 1];
            if prev.w <= 4 || prev.h <= 4 || prev.count <= 16 {
                break;
            }
            let (pw, ph) = (prev.w, prev.h);
            let (cw, ch) = (pw.div_ceil(2), ph.div_ceil(2));
            let mut cm = vec![0u8; cw * ch];
            for y in 0..ch {
                for x in 0..cw {
                    let mut k = 0u8;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (fx, fy) = (2 * x + dx, 2 * y + dy);
                        if fx < pw && fy < ph {
                            k += prev.mask[fy * pw + fx];
                        }
                    }
                    // A coarse cell is unknown when ≥ 3 of its 4 children
                    // are: the coarse domain must not outgrow the fine one,
                    // or coarse corrections overshoot near the boundary.
                    cm[y * cw + x] = (k >= 3) as u8;
                }
            }
            let i = self.used;
            let c = self.level(i);
            c.reset(cw, ch);
            c.mask = cm;
            c.guard_border();
            c.build_runs();
            self.used += 1;
        }
    }

    /// Restricts the residual of level `l` into the RHS of level `l + 1`.
    fn restrict(&mut self, l: usize) {
        let (fine, coarse) = self.levels.split_at_mut(l + 1);
        let f = &fine[l];
        let c = &mut coarse[0];
        let (fw, fh, cw) = (f.w, f.h, c.w);
        let cmask = &c.mask;
        let rows = |(y, row): (usize, &mut [V4])| {
            for (x, out) in row.iter_mut().enumerate() {
                if cmask[y * cw + x] == 0 {
                    *out = Z;
                    continue;
                }
                let mut s = Z;
                let mut n = 0.0;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (fx, fy) = (2 * x + dx, 2 * y + dy);
                    if fx < fw && fy < fh {
                        s = add(s, f.r[fy * fw + fx]);
                        n += 1.0;
                    }
                }
                // Coarse spacing is 2h: the unit-spacing operator needs ×4.
                *out = mul(s, 4.0 / n);
            }
        };
        if c.count >= 160 * 160 {
            c.f.par_chunks_mut(cw).with_min_len(16).enumerate().for_each(rows);
        } else {
            c.f.chunks_mut(cw).enumerate().for_each(rows);
        }
        c.u.fill(Z);
    }

    /// Adds the prolonged correction of level `l + 1` to level `l`
    /// (bilinear, cell-centered).
    fn prolong(&mut self, l: usize) {
        let (fine, coarse) = self.levels.split_at_mut(l + 1);
        let f = &mut fine[l];
        let c = &coarse[0];
        let (cw, ch, fw) = (c.w, c.h, f.w);
        let cu = |x: i64, y: i64| -> V4 { c.u[(y.clamp(0, ch as i64 - 1) as usize) * cw + x.clamp(0, cw as i64 - 1) as usize] };
        let runs = &f.runs;
        let rows = |(y, row): (usize, &mut [V4])| {
            let (cy, oy) = ((y / 2) as i64, if y % 2 == 0 { -1 } else { 1 });
            for &(x0, x1) in &runs[y] {
                for x in x0 as usize..x1 as usize {
                    let (cx, ox) = ((x / 2) as i64, if x % 2 == 0 { -1 } else { 1 });
                    let v = add(add(mul(cu(cx, cy), 0.5625), mul(add(cu(cx + ox, cy), cu(cx, cy + oy)), 0.1875)), mul(cu(cx + ox, cy + oy), 0.0625));
                    row[x] = add(row[x], v);
                }
            }
        };
        if f.count >= 160 * 160 {
            f.u.par_chunks_mut(fw).with_min_len(16).enumerate().for_each(rows);
        } else {
            f.u.chunks_mut(fw).enumerate().for_each(rows);
        }
    }

    fn vcycle(&mut self, l: usize) {
        if l == self.used - 1 {
            self.levels[l].smooth(40);
            return;
        }
        self.levels[l].smooth(3);
        self.levels[l].residual();
        self.restrict(l);
        self.vcycle(l + 1);
        self.prolong(l);
        self.levels[l].smooth(3);
    }

    /// Solves `L u = f` on the unknowns of `mask` (stops once the max
    /// residual is below `tol`, or after `cycles`). `u` holds the initial
    /// guess and the boundary values on entry and the solution on exit.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_v4(&mut self, u: &mut [V4], f: &[V4], mask: &[u8], w: usize, h: usize, cycles: usize, tol: f32) -> f32 {
        // Nested iteration: solve the half-resolution problem first and use
        // it as the initial guess.
        let unknowns = mask.iter().filter(|&&m| m != 0).count();
        if unknowns > 4096 && w > 16 && h > 16 {
            let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
            let mut uc = vec![Z; cw * ch];
            let mut fc = vec![Z; cw * ch];
            let mut mc = vec![0u8; cw * ch];
            for y in 0..ch {
                for x in 0..cw {
                    let (mut su, mut sf, mut n, mut k) = (Z, Z, 0f32, 0u8);
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (fx, fy) = (2 * x + dx, 2 * y + dy);
                        if fx < w && fy < h {
                            let i = fy * w + fx;
                            su = add(su, u[i]);
                            sf = add(sf, f[i]);
                            n += 1.0;
                            k += mask[i];
                        }
                    }
                    let i = y * cw + x;
                    uc[i] = mul(su, 1.0 / n);
                    fc[i] = mul(sf, 4.0 / n);
                    mc[i] = (k >= 3) as u8;
                }
            }
            // The nested solve is only an initial guess: one V-cycle is enough.
            let coarse = self.coarse.get_or_insert_with(Default::default);
            coarse.solve_v4(&mut uc, &fc, &mc, cw, ch, 1, tol);
            let at = |x: i64, y: i64| uc[(y.clamp(0, ch as i64 - 1) as usize) * cw + x.clamp(0, cw as i64 - 1) as usize];
            for y in 0..h {
                let (cy, oy) = ((y / 2) as i64, if y % 2 == 0 { -1 } else { 1 });
                for x in 0..w {
                    if mask[y * w + x] == 0 {
                        continue;
                    }
                    let (cx, ox) = ((x / 2) as i64, if x % 2 == 0 { -1 } else { 1 });
                    u[y * w + x] = add(add(mul(at(cx, cy), 0.5625), mul(add(at(cx + ox, cy), at(cx, cy + oy)), 0.1875)), mul(at(cx + ox, cy + oy), 0.0625));
                }
            }
        }
        self.setup(w, h, mask);
        {
            let l0 = &mut self.levels[0];
            l0.u.copy_from_slice(u);
            l0.f.copy_from_slice(f);
        }
        let mut res = self.levels[0].residual();
        let mut done = 0;
        for _ in 0..cycles.max(1) {
            if res < tol {
                break;
            }
            self.vcycle(0);
            done += 1;
            res = self.levels[0].residual();
        }
        if std::env::var("PF_TIMING").is_ok() {
            eprintln!("  solve {w}x{h}: {done} cycles, residual {res:.3}");
        }
        u.copy_from_slice(&self.levels[0].u);
        res
    }

    /// Scalar convenience wrapper (one lane).
    #[cfg(test)]
    pub fn solve(&mut self, u: &mut [f32], f: &[f32], mask: &[u8], w: usize, h: usize, cycles: usize) -> f32 {
        let mut uv: Vec<V4> = u.iter().map(|&v| [v, 0.0, 0.0, 0.0]).collect();
        let fv: Vec<V4> = f.iter().map(|&v| [v, 0.0, 0.0, 0.0]).collect();
        let r = self.solve_v4(&mut uv, &fv, mask, w, h, cycles, 1e-3);
        for (o, v) in u.iter_mut().zip(&uv) {
            *o = v[0];
        }
        r
    }
}

/// Harmonic extension for large regions: the membrane's interior is smooth
/// and only boundary detail is high-frequency, and that detail decays within
/// a few pixels of the boundary. So the region is solved at half resolution
/// (recursively, by the multigrid solver), prolonged, and then corrected by
/// Gauss–Seidel sweeps restricted to a band `band` pixels wide along the
/// boundary at full resolution.
#[allow(clippy::too_many_arguments)]
fn harmonic_fast(solver: &mut Solver, u: &mut [V4], mask: &[u8], w: usize, h: usize, cycles: usize, tol: f32, band: usize, sweeps: usize) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let mut uc = vec![Z; cw * ch];
    let mut mc = vec![0u8; cw * ch];
    for y in 0..ch {
        for x in 0..cw {
            let (mut su, mut n, mut k) = (Z, 0f32, 0u8);
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let (fx, fy) = (2 * x + dx, 2 * y + dy);
                if fx < w && fy < h {
                    let i = fy * w + fx;
                    su = add(su, u[i]);
                    n += 1.0;
                    k += mask[i];
                }
            }
            uc[y * cw + x] = mul(su, 1.0 / n);
            mc[y * cw + x] = (k >= 3) as u8;
        }
    }
    let zero = vec![Z; cw * ch];
    let coarse = solver.coarse.get_or_insert_with(Default::default);
    coarse.solve_v4(&mut uc, &zero, &mc, cw, ch, cycles, tol);
    let at = |x: i64, y: i64| uc[(y.clamp(0, ch as i64 - 1) as usize) * cw + x.clamp(0, cw as i64 - 1) as usize];
    u.par_chunks_mut(w).with_min_len(16).enumerate().for_each(|(y, row)| {
        let (cy, oy) = ((y / 2) as i64, if y % 2 == 0 { -1 } else { 1 });
        for (x, v) in row.iter_mut().enumerate() {
            if mask[y * w + x] == 0 {
                continue;
            }
            let (cx, ox) = ((x / 2) as i64, if x % 2 == 0 { -1 } else { 1 });
            *v = add(add(mul(at(cx, cy), 0.5625), mul(add(at(cx + ox, cy), at(cx, cy + oy)), 0.1875)), mul(at(cx + ox, cy + oy), 0.0625));
        }
    });
    // Band: unknowns within `band` (Chebyshev) of a known pixel. Only rows
    // near the boundary are scanned: for each unknown run, the cells within
    // `band` of its ends or of a row with different coverage above/below.
    let near_known = |x: usize, y: usize| -> bool {
        let (y0, y1) = (y.saturating_sub(band), (y + band + 1).min(h));
        let (x0, x1) = (x.saturating_sub(band), (x + band + 1).min(w));
        (y0..y1).any(|yy| mask[yy * w + x0..yy * w + x1].iter().any(|&m| m == 0))
    };
    let cells: Vec<u32> = (1..h - 1)
        .into_par_iter()
        .with_min_len(8)
        .flat_map_iter(|y| {
            (1..w - 1).filter_map(move |x| {
                let i = y * w + x;
                if mask[i] == 0 {
                    return None;
                }
                // Cheap reject: interior cells far from any known pixel.
                let far = x >= band + 1 && x + band + 1 < w && y >= band + 1 && y + band + 1 < h
                    && mask[i - band] != 0 && mask[i + band] != 0 && mask[i - band * w] != 0 && mask[i + band * w] != 0
                    && mask[i - band * w - band] != 0 && mask[i + band * w + band] != 0 && mask[i - band * w + band] != 0 && mask[i + band * w - band] != 0;
                (!far || near_known(x, y)).then_some(i as u32)
            })
        })
        .collect();
    // Red-black Gauss–Seidel on the band (f = 0), the interior held fixed.
    for _ in 0..sweeps {
        for color in 0..2 {
            for &c in &cells {
                let i = c as usize;
                if ((i % w) + (i / w)) % 2 != color {
                    continue;
                }
                let v = add(add(u[i - 1], u[i + 1]), add(u[i - w], u[i + w]));
                u[i] = mul(v, 0.25);
            }
        }
    }
}

const BAND: usize = 4;
const SWEEPS: usize = 12;

/// Healing quality. `Fast` solves large regions' interiors at half
/// resolution (mean deviation < 0.5 of a level); `Exact` runs full-resolution
/// multigrid until the max Laplacian residual is below [`EXACT_TOL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Quality {
    #[default]
    Fast,
    Exact,
}

/// Residual bound of [`Quality::Exact`].
pub const EXACT_TOL: f32 = 0.05;

/// Seamless blend of 4-lane planes: returns `src + membrane` where the
/// membrane matches `dst − src` outside `mask`.
#[cfg(test)]
pub fn membrane_blend_v4(solver: &mut Solver, src: &[V4], dst: &[V4], mask: &[u8], w: usize, h: usize, cycles: usize) -> Vec<V4> {
    membrane_blend_q(solver, src, dst, mask, w, h, cycles, Quality::Fast)
}

/// [`membrane_blend_v4`] at a chosen quality.
#[allow(clippy::too_many_arguments)]
pub fn membrane_blend_q(solver: &mut Solver, src: &[V4], dst: &[V4], mask: &[u8], w: usize, h: usize, cycles: usize, q: Quality) -> Vec<V4> {
    let mut m: Vec<V4> = dst.par_iter().with_min_len(4096).zip(src).map(|(d, s)| sub(*d, *s)).collect();
    // Initial guess inside: the mean boundary difference.
    let (sum, n) = (0..h)
        .into_par_iter()
        .with_min_len(16)
        .map(|y| {
            let (mut sum, mut n) = (Z, 0usize);
            for x in 0..w {
                let i = y * w + x;
                if mask[i] == 0 {
                    let nb = (x > 0 && mask[i - 1] != 0) || (x + 1 < w && mask[i + 1] != 0) || (y > 0 && mask[i - w] != 0) || (y + 1 < h && mask[i + w] != 0);
                    if nb {
                        sum = add(sum, m[i]);
                        n += 1;
                    }
                }
            }
            (sum, n)
        })
        .reduce(|| (Z, 0), |a, b| (add(a.0, b.0), a.1 + b.1));
    let mean = if n > 0 { mul(sum, 1.0 / n as f32) } else { Z };
    m.par_iter_mut().with_min_len(4096).zip(mask).for_each(|(v, &k)| {
        if k != 0 {
            *v = mean;
        }
    });
    // Max Laplacian residual 0.25 ≈ 0.06 of a level of error: invisible
    // after rounding to 8 bits.
    let unknowns = mask.iter().filter(|&&k| k != 0).count();
    let mut run = || {
        if q == Quality::Exact {
            // Full resolution everywhere; the cycle cap only guards against
            // a pathological mask (each V-cycle cuts the residual ~5×).
            let zero = vec![Z; w * h];
            solver.solve_v4(&mut m, &zero, mask, w, h, cycles.max(12), EXACT_TOL);
        } else if unknowns > 16_384 && w > 32 && h > 32 {
            harmonic_fast(solver, &mut m, mask, w, h, cycles, 0.25, BAND, SWEEPS);
        } else {
            let zero = vec![Z; w * h];
            solver.solve_v4(&mut m, &zero, mask, w, h, cycles, 0.25);
        }
    };
    if unknowns < 600 * 600 {
        dab_pool().install(run);
    } else {
        run();
    }
    m.par_iter().with_min_len(4096).zip(src).map(|(mv, s)| add(*s, *mv)).collect()
}

/// Scalar version (tests).
#[cfg(test)]
pub fn membrane_blend(solver: &mut Solver, src: &[f32], dst: &[f32], mask: &[u8], w: usize, h: usize, cycles: usize) -> Vec<f32> {
    let s: Vec<V4> = src.iter().map(|&v| [v, 0.0, 0.0, 0.0]).collect();
    let d: Vec<V4> = dst.iter().map(|&v| [v, 0.0, 0.0, 0.0]).collect();
    membrane_blend_v4(solver, &s, &d, mask, w, h, cycles).iter().map(|v| v[0]).collect()
}

/// Same for packed premultiplied RGBA: every pixel inside the mask becomes
/// `src + membrane` per color channel; alpha comes from `dst`. The solve is
/// cropped to the mask's bounding box (+1 px of boundary).
pub fn blend_rgba(solver: &mut Solver, src: &[u32], dst: &[u32], mask: &[u8], w: usize, h: usize, cycles: usize) -> Vec<u32> {
    blend_rgba_q(solver, src, dst, mask, w, h, cycles, Quality::Fast)
}

/// The solve's crop: the mask's bounding box plus 1 px of boundary
/// `(x0, y0, w, h)`, or None when the mask is empty.
fn crop_of(mask: &[u8], w: usize, h: usize) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
    for y in 0..h {
        let row = &mask[y * w..(y + 1) * w];
        if let Some(a) = row.iter().position(|&m| m != 0) {
            let b = row.iter().rposition(|&m| m != 0).unwrap();
            x0 = x0.min(a);
            x1 = x1.max(b + 1);
            y0 = y0.min(y);
            y1 = y1.max(y + 1);
        }
    }
    if x1 <= x0 {
        return None;
    }
    let (cx0, cy0, cx1, cy1) = (x0.saturating_sub(1), y0.saturating_sub(1), (x1 + 1).min(w), (y1 + 1).min(h));
    Some((cx0, cy0, cx1 - cx0, cy1 - cy0))
}

/// [`blend_rgba`] at a chosen quality. The GPU solves the same crop when
/// it's enabled and predicted faster (`gpu::retouch_gate`); any GPU failure
/// falls back to [`blend_rgba_cpu`].
#[allow(clippy::too_many_arguments)]
pub fn blend_rgba_q(solver: &mut Solver, src: &[u32], dst: &[u32], mask: &[u8], w: usize, h: usize, cycles: usize, q: Quality) -> Vec<u32> {
    let Some((_, _, cw, ch)) = crop_of(mask, w, h) else {
        return dst.to_vec();
    };
    let t0 = std::time::Instant::now();
    let op = if q == Quality::Exact { gpu::cost::OP_HEAL_EXACT } else { gpu::cost::OP_HEAL };
    let units = (cw * ch) as u64;
    let why = match gpu::retouch_gate(op, units) {
        Ok(g) => match blend_rgba_gpu(&g, src, dst, mask, w, h, cycles, q) {
            Ok(out) => {
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                gpu::record(op, gpu::PathTaken::Gpu, ms);
                gpu::cost::observe_op(op, units, ms, true);
                return out;
            }
            Err(e) => {
                if e.is_fault() {
                    g.fail(e.to_string());
                }
                e.to_string()
            }
        },
        Err(why) => why,
    };
    let out = blend_rgba_cpu(solver, src, dst, mask, w, h, cycles, q);
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    gpu::record(op, gpu::PathTaken::Cpu(why), ms);
    gpu::cost::observe_op(op, units, ms, false);
    out
}

/// [`blend_rgba_q`] on the GPU (no fallback): the same crop and result.
#[allow(clippy::too_many_arguments)]
pub fn blend_rgba_gpu(g: &std::sync::Arc<gpu::Gpu>, src: &[u32], dst: &[u32], mask: &[u8], w: usize, h: usize, cycles: usize, q: Quality) -> gpu::exec::GResult<Vec<u32>> {
    let Some((cx0, cy0, cw, ch)) = crop_of(mask, w, h) else {
        return Ok(dst.to_vec());
    };
    let crop = |plane: &[u8]| -> Vec<u8> { (cy0..cy0 + ch).flat_map(|y| plane[y * w + cx0..y * w + cx0 + cw].iter().copied()).collect() };
    let crop_px = |plane: &[u32]| -> Vec<u32> { (cy0..cy0 + ch).flat_map(|y| plane[y * w + cx0..y * w + cx0 + cw].iter().copied()).collect() };
    let c = gpu::poisson::blend_crop(g, &crop_px(src), &crop_px(dst), &crop(mask), cw, ch, cycles, q)?;
    let mut out = dst.to_vec();
    out[cy0 * w..(cy0 + ch) * w].par_chunks_mut(w).with_min_len(16).enumerate().for_each(|(y, row)| row[cx0..cx0 + cw].copy_from_slice(&c[y * cw..(y + 1) * cw]));
    Ok(out)
}

/// [`blend_rgba_q`] on the CPU (the reference).
#[allow(clippy::too_many_arguments)]
pub fn blend_rgba_cpu(solver: &mut Solver, src: &[u32], dst: &[u32], mask: &[u8], w: usize, h: usize, cycles: usize, q: Quality) -> Vec<u32> {
    use super::px;
    let Some((cx0, cy0, cw, ch)) = crop_of(mask, w, h) else {
        return dst.to_vec();
    };
    let (cx1, cy1) = (cx0 + cw, cy0 + ch);
    let to_v = |c: u32| -> V4 {
        let [r, g, b, _] = px::unpack(c);
        [r as f32, g as f32, b as f32, 0.0]
    };
    let crop_rows = |plane: &[u32]| -> Vec<V4> { (cy0..cy1).into_par_iter().with_min_len(16).flat_map_iter(|y| plane[y * w + cx0..y * w + cx1].iter().map(|&c| to_v(c))).collect() };
    let (s, d) = rayon::join(|| crop_rows(src), || crop_rows(dst));
    let cm: Vec<u8> = (cy0..cy1).flat_map(|y| mask[y * w + cx0..y * w + cx1].iter().copied()).collect();
    let solved = membrane_blend_q(solver, &s, &d, &cm, cw, ch, cycles, q);
    let mut out = dst.to_vec();
    out[cy0 * w..cy1 * w].par_chunks_mut(w).with_min_len(16).enumerate().for_each(|(y, row)| {
        for x in 0..cw {
            let k = y * cw + x;
            if cm[k] == 0 {
                continue;
            }
            let i = (cy0 + y) * w + cx0 + x;
            let a = px::alpha(dst[i]).max(px::alpha(src[i]));
            let ch_ = |c: usize| solved[k][c].round().clamp(0.0, a as f32) as u8;
            row[cx0 + x] = px::pack(ch_(0), ch_(1), ch_(2), a as u8);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Max |L u − f| over the unknowns.
    fn residual(u: &[f32], f: &[f32], mask: &[u8], w: usize, h: usize) -> f32 {
        let mut worst = 0f32;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = y * w + x;
                if mask[i] == 0 {
                    continue;
                }
                let lu = u[i - 1] + u[i + 1] + u[i - w] + u[i + w] - 4.0 * u[i];
                worst = worst.max((lu - f[i]).abs());
            }
        }
        worst
    }

    #[test]
    fn multigrid_reaches_tolerance_fast() {
        // A disk of unknowns with a non-trivial boundary and source term.
        let (w, h) = (300usize, 260usize);
        let mut mask = vec![0u8; w * h];
        let mut u = vec![0f32; w * h];
        let mut f = vec![0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let (dx, dy) = (x as f32 - 150.0, y as f32 - 130.0);
                if dx * dx + dy * dy < 110.0 * 110.0 {
                    mask[i] = 1;
                    f[i] = ((x as f32 * 0.05).sin() + (y as f32 * 0.07).cos()) * 0.5;
                } else {
                    u[i] = 100.0 + 50.0 * (x as f32 * 0.02).sin() + 30.0 * (y as f32 * 0.03).cos();
                }
            }
        }
        // Loose: guards against O(n²) behaviour while other tests run in
        // parallel; real timings live in --bench-tools. A private pool keeps
        // the solver's jobs out of the global queue the GPU-equality tests
        // fill, and the best of three runs drops scheduling stalls.
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let u0 = u.clone();
        let mut ms = f32::MAX;
        for _ in 0..3 {
            u.copy_from_slice(&u0);
            let mut s = Solver::new();
            let t = std::time::Instant::now();
            let res = pool.install(|| s.solve(&mut u, &f, &mask, w, h, 12));
            ms = ms.min(t.elapsed().as_secs_f32() * 1000.0);
            let r = residual(&u, &f, &mask, w, h);
            assert!(r < 0.01, "residual {r} (solver reported {res})");
            if ms < 400.0 {
                break;
            }
        }
        assert!(ms < 400.0, "{ms} ms");
    }

    #[test]
    fn membrane_matches_boundary_and_is_harmonic() {
        let (w, h) = (64usize, 64usize);
        let mut mask = vec![0u8; w * h];
        for y in 10..54 {
            for x in 10..54 {
                mask[y * w + x] = 1;
            }
        }
        // Source is a texture; destination differs by a smooth ramp.
        let src: Vec<f32> = (0..w * h).map(|i| 100.0 + ((i * 37) % 17) as f32).collect();
        let dst: Vec<f32> = (0..w * h).map(|i| src[i] + 20.0 + (i % w) as f32 * 0.5).collect();
        let mut s = Solver::new();
        let out = membrane_blend(&mut s, &src, &dst, &mask, w, h, 10);
        // A linear ramp is harmonic: the blend reproduces the destination.
        let worst = (0..w * h).filter(|&i| mask[i] != 0).map(|i| (out[i] - dst[i]).abs()).fold(0f32, f32::max);
        // Below a quarter of an 8-bit level: invisible after rounding.
        assert!(worst < 0.25, "worst {worst}");
    }

    #[test]
    fn fast_harmonic_matches_exact_solve() {
        let (w, h) = (420usize, 400usize);
        let mut mask = vec![0u8; w * h];
        let mut b = vec![Z; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let (dx, dy) = (x as f32 - 210.0, y as f32 - 200.0);
                if dx * dx + dy * dy < 190.0 * 190.0 {
                    mask[i] = 1;
                }
                // Boundary data with pixel-scale texture.
                let t = (((x * 31) ^ (y * 17)) % 29) as f32;
                b[i] = [40.0 + t, (x as f32 * 0.07).sin() * 30.0, (y as f32 * 0.5).cos() * 10.0, 0.0];
            }
        }
        let mut exact = b.clone();
        Solver::new().solve_v4(&mut exact, &vec![Z; w * h], &mask, w, h, 30, 1e-3);
        let mut fast = b.clone();
        let mut s = Solver::new();
        harmonic_fast(&mut s, &mut fast, &mask, w, h, 8, 0.25, BAND, SWEEPS);
        let worst = (0..w * h).filter(|&i| mask[i] != 0).map(|i| maxabs(sub(exact[i], fast[i]))).fold(0f32, f32::max);
        let mean = (0..w * h).filter(|&i| mask[i] != 0).map(|i| maxabs(sub(exact[i], fast[i]))).sum::<f32>() / mask.iter().filter(|&&m| m != 0).count() as f32;
        // Sub-level mean deviation; the worst pixels sit on the rim, where
        // the healing brush's soft edge blends anyway.
        assert!(mean < 0.5 && worst < 4.5, "mean {mean} worst {worst}");
    }

    /// Direct solve of the 5-point Dirichlet problem (f = 0) by banded
    /// Cholesky in f64: the unknowns are numbered row-major, so the
    /// bandwidth is one image row.
    fn direct_harmonic(b: &[f32], mask: &[u8], w: usize, h: usize) -> Vec<f64> {
        let idx: Vec<usize> = {
            let mut n = 0;
            mask.iter().map(|&m| if m != 0 { n += 1; n - 1 } else { usize::MAX }).collect()
        };
        let cells: Vec<usize> = (0..w * h).filter(|&i| mask[i] != 0).collect();
        let n = cells.len();
        let bw = w; // max |row(i) - row(j)| over coupled unknowns
        // Lower band storage: a[i][k] = A[i][i - k], k in 0..=bw.
        let mut a = vec![0f64; n * (bw + 1)];
        let mut rhs = vec![0f64; n];
        for (r, &i) in cells.iter().enumerate() {
            a[r * (bw + 1)] = 4.0;
            for j in [i - 1, i + 1, i - w, i + w] {
                if mask[j] != 0 {
                    let c = idx[j];
                    if c < r {
                        a[r * (bw + 1) + (r - c)] = -1.0;
                    }
                } else {
                    rhs[r] += b[j] as f64;
                }
            }
        }
        // In-place banded Cholesky: A = L Lᵀ.
        for i in 0..n {
            let j0 = i.saturating_sub(bw);
            for j in j0..=i {
                let mut s = a[i * (bw + 1) + (i - j)];
                for k in j0.max(j.saturating_sub(bw))..j {
                    s -= a[i * (bw + 1) + (i - k)] * a[j * (bw + 1) + (j - k)];
                }
                a[i * (bw + 1) + (i - j)] = if i == j { s.sqrt() } else { s / a[j * (bw + 1)] };
            }
        }
        // Forward then back substitution.
        let mut y = rhs;
        for i in 0..n {
            let mut s = y[i];
            for k in i.saturating_sub(bw)..i {
                s -= a[i * (bw + 1) + (i - k)] * y[k];
            }
            y[i] = s / a[i * (bw + 1)];
        }
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in i + 1..(i + bw + 1).min(n) {
                s -= a[k * (bw + 1) + (k - i)] * y[k];
            }
            y[i] = s / a[i * (bw + 1)];
        }
        let mut out = vec![0f64; w * h];
        for (r, &i) in cells.iter().enumerate() {
            out[i] = y[r];
        }
        out
    }

    #[test]
    fn exact_quality_matches_direct_solve() {
        // A 150 px healing dab (≈ 17 700 unknowns) with textured boundary
        // values, solved by Exact quality and by a direct factorization.
        let (w, h) = (156usize, 156usize);
        let mut mask = vec![0u8; w * h];
        let mut src = vec![Z; w * h];
        let mut dst = vec![Z; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let (dx, dy) = (x as f32 - 78.0, y as f32 - 78.0);
                if dx * dx + dy * dy < 75.0 * 75.0 {
                    mask[i] = 1;
                }
                let t = (((x * 31) ^ (y * 17)) % 29) as f32;
                src[i] = [100.0 + t, 90.0, 80.0 + (x as f32 * 0.3).sin() * 9.0, 0.0];
                dst[i] = [60.0 + (x as f32 * 0.1).sin() * 20.0, 70.0 + t, 50.0 + (y as f32 * 0.2).cos() * 25.0, 0.0];
            }
        }
        let out = membrane_blend_q(&mut Solver::new(), &src, &dst, &mask, w, h, 4, Quality::Exact);
        let mut worst = 0f64;
        for c in 0..3 {
            let b: Vec<f32> = (0..w * h).map(|i| dst[i][c] - src[i][c]).collect();
            let m = direct_harmonic(&b, &mask, w, h);
            for i in (0..w * h).filter(|&i| mask[i] != 0) {
                worst = worst.max((out[i][c] as f64 - (src[i][c] as f64 + m[i])).abs());
            }
        }
        assert!(worst < 0.1, "Exact deviates {worst:.3} levels from the direct solve");
        // Fast is allowed to deviate more (half-resolution interior).
        let fast = membrane_blend_q(&mut Solver::new(), &src, &dst, &mask, w, h, 4, Quality::Fast);
        let fw = (0..w * h).filter(|&i| mask[i] != 0).map(|i| maxabs(sub(fast[i], out[i]))).fold(0f32, f32::max);
        eprintln!("exact vs direct: {worst:.4} levels; fast vs exact: {fw:.3} levels");
    }

    #[test]
    fn rgba_lanes_are_independent() {
        let (w, h) = (80usize, 60usize);
        let mut mask = vec![0u8; w * h];
        for y in 10..50 {
            for x in 10..70 {
                mask[y * w + x] = 1;
            }
        }
        let src = vec![super::super::px::pack(100, 100, 100, 255); w * h];
        let dst: Vec<u32> = (0..w * h).map(|i| super::super::px::pack(150, 60, (i % w) as u8 * 2, 255)).collect();
        let mut s = Solver::new();
        let out = blend_rgba(&mut s, &src, &dst, &mask, w, h, 10);
        let c = super::super::px::unpack(out[30 * w + 40]);
        assert!((c[0] as i32 - 150).abs() <= 1 && (c[1] as i32 - 60).abs() <= 1 && (c[2] as i32 - 80).abs() <= 1, "{c:?}");
    }
}

#[cfg(test)]
mod perf {
    use super::*;
    #[test]
    #[ignore]
    fn heal_sized_solve() {
        for d in [100usize, 300, 500] {
            let (w, h) = (d + 6, d + 6);
            let mut mask = vec![0u8; w * h];
            let mut src = vec![Z; w * h];
            let mut dst = vec![Z; w * h];
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let (dx, dy) = (x as f32 - w as f32 / 2.0, y as f32 - h as f32 / 2.0);
                    if dx * dx + dy * dy < (d as f32 / 2.0).powi(2) {
                        mask[i] = 1;
                    }
                    let t = ((x * 31 + y * 17) % 23) as f32;
                    src[i] = [100.0 + t, 90.0 + t, 80.0, 0.0];
                    dst[i] = [60.0 + (x as f32 * 0.1).sin() * 20.0, 70.0, 50.0 + t, 0.0];
                }
            }
            let mut s = Solver::new();
            for _ in 0..3 {
                membrane_blend_v4(&mut s, &src, &dst, &mask, w, h, 4);
            }
            let t = std::time::Instant::now();
            let n = 20;
            for _ in 0..n {
                membrane_blend_v4(&mut s, &src, &dst, &mask, w, h, 4);
            }
            eprintln!("{d} px: {:.2} ms per solve", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
            let t = std::time::Instant::now();
            for _ in 0..n {
                membrane_blend_q(&mut s, &src, &dst, &mask, w, h, 4, Quality::Exact);
            }
            eprintln!("{d} px exact: {:.2} ms per solve", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
        }
    }
}
