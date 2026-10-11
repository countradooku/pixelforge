// The healing membrane solver (`engine::poisson`) on the GPU: the same
// cell-centered multigrid V-cycles with red-black Gauss-Seidel smoothing,
// full-weighting restriction and bilinear prolongation, the same nested
// iteration and the same half-resolution "fast" harmonic extension with its
// boundary-band sweeps. Red-black sweeps don't depend on the order cells of
// one colour are visited in, so a parallel sweep computes what the CPU's row
// loops compute; only float rounding (fast-math contraction) differs.
//
// Every level of every (nested) solve lives in shared pools at a cell offset:
// U (unknowns), F (right-hand side), R (residual), all 4-lane f32, and M (byte
// masks, 1 = unknown). The CPU's early exit (`residual < tol`) is a guard
// slot in `ctrl`: the residual pass after each V-cycle atomically maxes the
// residual into its slot, and every op of the next cycle returns at once when
// that slot is below `tol` (slot 0 holds +inf: never stops).

struct Lv {
    off: u32,
    w: u32,
    h: u32,
    moff: u32,
}

struct P {
    guard: u32,
    tol: f32,
    slot: u32,
    color: u32,
    a: Lv,
    b: Lv,
    k: vec4<u32>,
    f: vec4<f32>,
    lv: array<Lv, 24>,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> U: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> F: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> R: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> M: array<u32>;
@group(0) @binding(5) var<storage, read_write> ctrl: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> img: array<u32>;
@group(0) @binding(7) var<storage, read_write> aux: array<u32>;

const ZERO4 = vec4<f32>(0.0);

// Every entry binds the whole buffer set (see `gpu::prog`).
fn touch() {
    _ = p.a.w;
    _ = U[0];
    _ = F[0];
    _ = R[0];
    _ = M[0];
    _ = atomicLoad(&ctrl[0]);
    _ = img[0];
    _ = aux[0];
}

fn stopped() -> bool {
    return bitcast<f32>(atomicLoad(&ctrl[p.guard])) < p.tol;
}

// Byte `j` of the mask at byte offset `moff`.
fn mk(moff: u32, j: u32) -> u32 {
    let b = moff + j;
    return (M[b >> 2u] >> ((b & 3u) * 8u)) & 255u;
}

fn rgb(c: u32) -> vec4<f32> {
    return vec4<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), 0.0);
}

// 4 / n and 1 / n for n in {1, 2, 4} without a (fast-math) division.
fn four_over(n: u32) -> f32 {
    return select(select(4.0, 2.0, n == 2u), 1.0, n == 4u);
}
fn one_over(n: u32) -> f32 {
    return select(select(1.0, 0.5, n == 2u), 0.25, n == 4u);
}

fn maxabs(v: vec4<f32>) -> f32 {
    let a = abs(v);
    return max(max(a.x, a.y), max(a.z, a.w));
}

// ----- per-cell operations (shared by the grid kernels and `vsmall`) --------

fn smooth_cell(L: Lv, x: u32, y: u32) {
    let j = y * L.w + x;
    if (mk(L.moff, j) == 0u) {
        return;
    }
    let i = L.off + j;
    let l = U[i - 1u];
    let r = U[i + 1u];
    let t = U[i - L.w];
    let b = U[i + L.w];
    U[i] = (((l + r) + (t + b)) - F[i]) * 0.25;
}

fn residual_cell(L: Lv, x: u32, y: u32) -> f32 {
    let j = y * L.w + x;
    let i = L.off + j;
    if (mk(L.moff, j) == 0u) {
        R[i] = ZERO4;
        return 0.0;
    }
    let lu = ((U[i - 1u] + U[i + 1u]) + (U[i - L.w] + U[i + L.w])) - U[i] * 4.0;
    let res = F[i] - lu;
    R[i] = res;
    return maxabs(res);
}

fn restrict_cell(Fl: Lv, C: Lv, x: u32, y: u32) {
    let ci = C.off + y * C.w + x;
    U[ci] = ZERO4;
    if (mk(C.moff, y * C.w + x) == 0u) {
        F[ci] = ZERO4;
        return;
    }
    var s = ZERO4;
    var n = 0u;
    for (var q = 0u; q < 4u; q++) {
        let fx = 2u * x + (q & 1u);
        let fy = 2u * y + (q >> 1u);
        if (fx < Fl.w && fy < Fl.h) {
            s = s + R[Fl.off + fy * Fl.w + fx];
            n += 1u;
        }
    }
    F[ci] = s * four_over(n);
}

fn cu(C: Lv, x: i32, y: i32) -> vec4<f32> {
    let cx = u32(clamp(x, 0, i32(C.w) - 1));
    let cy = u32(clamp(y, 0, i32(C.h) - 1));
    return U[C.off + cy * C.w + cx];
}

fn bilinear(C: Lv, x: u32, y: u32) -> vec4<f32> {
    let cx = i32(x / 2u);
    let cy = i32(y / 2u);
    let ox = select(1, -1, (x & 1u) == 0u);
    let oy = select(1, -1, (y & 1u) == 0u);
    return (cu(C, cx, cy) * 0.5625 + (cu(C, cx + ox, cy) + cu(C, cx, cy + oy)) * 0.1875) + cu(C, cx + ox, cy + oy) * 0.0625;
}

fn prolong_cell(C: Lv, Fl: Lv, x: u32, y: u32) {
    if (mk(Fl.moff, y * Fl.w + x) == 0u) {
        return;
    }
    let i = Fl.off + y * Fl.w + x;
    U[i] = U[i] + bilinear(C, x, y);
}

// ----- grid kernels -----------------------------------------------------------

// The crop's initial membrane: dst - src on known cells, the mean boundary
// difference (computed on the CPU) on unknown ones; f = 0.
@compute @workgroup_size(16, 16)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    let L = p.a;
    if (gid.x >= L.w || gid.y >= L.h) {
        return;
    }
    let j = gid.y * L.w + gid.x;
    let i = L.off + j;
    if (mk(L.moff, j) != 0u) {
        U[i] = vec4<f32>(p.f.xyz, 0.0);
    } else {
        U[i] = rgb(img[p.k.y + j]) - rgb(img[p.k.x + j]);
    }
    F[i] = ZERO4;
}

// One colour of a red-black Gauss-Seidel sweep: x covers half a row.
@compute @workgroup_size(16, 16)
fn mg_smooth(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    if (stopped()) {
        return;
    }
    let L = p.a;
    let y = gid.y;
    let x = 2u * gid.x + ((y + p.color) & 1u);
    if (x >= L.w || y >= L.h) {
        return;
    }
    smooth_cell(L, x, y);
}

var<workgroup> w_stop: u32;
var<workgroup> w_max: atomic<u32>;

// r = f - L u on unknowns (0 elsewhere); with `slot`, also the max |r|.
@compute @workgroup_size(16, 16)
fn mg_residual(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    touch();
    if (li == 0u) {
        w_stop = u32(stopped());
        atomicStore(&w_max, 0u);
    }
    if (workgroupUniformLoad(&w_stop) != 0u) {
        return;
    }
    let L = p.a;
    if (gid.x < L.w && gid.y < L.h) {
        let m = residual_cell(L, gid.x, gid.y);
        if (p.slot != 0u && m > 0.0) {
            atomicMax(&w_max, bitcast<u32>(m));
        }
    }
    workgroupBarrier();
    if (li == 0u && p.slot != 0u) {
        atomicMax(&ctrl[p.slot], atomicLoad(&w_max));
    }
}

// Residual of fine level `a` into the RHS of coarse level `b` (and b.u = 0).
@compute @workgroup_size(16, 16)
fn mg_restrict(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    if (stopped()) {
        return;
    }
    if (gid.x >= p.b.w || gid.y >= p.b.h) {
        return;
    }
    restrict_cell(p.a, p.b, gid.x, gid.y);
}

// Adds the bilinear correction of coarse level `b` to fine level `a`.
@compute @workgroup_size(16, 16)
fn mg_prolong(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    if (stopped()) {
        return;
    }
    if (gid.x >= p.a.w || gid.y >= p.a.h) {
        return;
    }
    prolong_cell(p.b, p.a, gid.x, gid.y);
}

// Nested iteration / fast harmonic extension: the half-resolution problem
// (b) from the fine one (a): u averaged, f summed x 4 / n (k.x = 1) or 0.
@compute @workgroup_size(16, 16)
fn nested_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    let Fl = p.a;
    let C = p.b;
    if (gid.x >= C.w || gid.y >= C.h) {
        return;
    }
    var su = ZERO4;
    var sf = ZERO4;
    var n = 0u;
    for (var q = 0u; q < 4u; q++) {
        let fx = 2u * gid.x + (q & 1u);
        let fy = 2u * gid.y + (q >> 1u);
        if (fx < Fl.w && fy < Fl.h) {
            let i = Fl.off + fy * Fl.w + fx;
            su = su + U[i];
            sf = sf + F[i];
            n += 1u;
        }
    }
    let ci = C.off + gid.y * C.w + gid.x;
    U[ci] = su * one_over(n);
    F[ci] = select(ZERO4, sf * four_over(n), p.k.x != 0u);
}

// Fine unknowns (a, raw mask) = bilinear of the coarse solution (b).
@compute @workgroup_size(16, 16)
fn prolong_set(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    let Fl = p.a;
    if (gid.x >= Fl.w || gid.y >= Fl.h) {
        return;
    }
    if (mk(Fl.moff, gid.y * Fl.w + gid.x) == 0u) {
        return;
    }
    U[Fl.off + gid.y * Fl.w + gid.x] = bilinear(p.b, gid.x, gid.y);
}

// ----- the boundary band of the fast extension ---------------------------------

// aux[k.x + i] = 1 when a known cell lies in the row window x +- band (k.y).
@compute @workgroup_size(16, 16)
fn band_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    let L = p.a;
    if (gid.x >= L.w || gid.y >= L.h) {
        return;
    }
    let band = p.k.y;
    let x0 = select(0u, gid.x - band, gid.x >= band);
    let x1 = min(gid.x + band + 1u, L.w);
    var hit = 0u;
    for (var x = x0; x < x1; x++) {
        if (mk(L.moff, gid.y * L.w + x) == 0u) {
            hit = 1u;
            break;
        }
    }
    aux[p.k.x + gid.y * L.w + gid.x] = hit;
}

// The CPU's band cells (unknowns within `band` of a known cell, or failing
// its cheap interior test), appended to one list per colour: lists at
// aux[k.z] / aux[k.w], counters at ctrl[f-slot] (p.slot, p.slot + 1).
@compute @workgroup_size(16, 16)
fn band_list(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    let L = p.a;
    let x = gid.x;
    let y = gid.y;
    let w = L.w;
    let h = L.h;
    if (x < 1u || y < 1u || x + 1u >= w || y + 1u >= h) {
        return;
    }
    let i = y * w + x;
    if (mk(L.moff, i) == 0u) {
        return;
    }
    let band = p.k.y;
    let far = x >= band + 1u && x + band + 1u < w && y >= band + 1u && y + band + 1u < h
        && mk(L.moff, i - band) != 0u && mk(L.moff, i + band) != 0u && mk(L.moff, i - band * w) != 0u && mk(L.moff, i + band * w) != 0u
        && mk(L.moff, i - band * w - band) != 0u && mk(L.moff, i + band * w + band) != 0u && mk(L.moff, i - band * w + band) != 0u && mk(L.moff, i + band * w - band) != 0u;
    var keep = !far;
    if (far) {
        let y0 = select(0u, y - band, y >= band);
        let y1 = min(y + band + 1u, h);
        for (var yy = y0; yy < y1; yy++) {
            if (aux[p.k.x + yy * w + x] != 0u) {
                keep = true;
                break;
            }
        }
    }
    if (!keep) {
        return;
    }
    let c = (x + y) & 1u;
    let n = atomicAdd(&ctrl[p.slot + c], 1u);
    aux[select(p.k.z, p.k.w, c == 1u) + n] = L.off + i;
}

// Indirect dispatch sizes (256 cells per group) for both band lists.
@compute @workgroup_size(1)
fn band_args() {
    touch();
    for (var c = 0u; c < 2u; c++) {
        let n = atomicLoad(&ctrl[p.slot + c]);
        atomicStore(&ctrl[p.k.x + 3u * c], (n + 255u) / 256u);
        atomicStore(&ctrl[p.k.x + 3u * c + 1u], 1u);
        atomicStore(&ctrl[p.k.x + 3u * c + 2u], 1u);
    }
}

// One colour of the band's Gauss-Seidel sweep (f = 0); a.w is the stride.
@compute @workgroup_size(256)
fn band_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    let n = atomicLoad(&ctrl[p.slot + p.color]);
    if (gid.x >= n) {
        return;
    }
    let i = aux[select(p.k.z, p.k.w, p.color == 1u) + gid.x];
    let w = p.a.w;
    U[i] = ((U[i - 1u] + U[i + 1u]) + (U[i - w] + U[i + w])) * 0.25;
}

// ----- output -----------------------------------------------------------------------

// out = src + membrane, rounded and clamped to the pixel's alpha, on unknown
// cells; dst elsewhere. img: src at k.x, dst at k.y, out at k.z.
@compute @workgroup_size(16, 16)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    let L = p.a;
    if (gid.x >= L.w || gid.y >= L.h) {
        return;
    }
    let j = gid.y * L.w + gid.x;
    let d = img[p.k.y + j];
    if (mk(L.moff, j) == 0u) {
        img[p.k.z + j] = d;
        return;
    }
    let s = img[p.k.x + j];
    let a = max(d >> 24u, s >> 24u);
    let v = rgb(s) + U[L.off + j];
    let fa = f32(a);
    let c = vec3<u32>(clamp(floor(v.xyz + vec3<f32>(0.5)), vec3<f32>(0.0), vec3<f32>(fa)));
    img[p.k.z + j] = c.x | (c.y << 8u) | (c.z << 16u) | (a << 24u);
}

// ----- small levels: a whole V-cycle in one workgroup ---------------------------

// Workgroup size of `vsmall` (1024 where the device allows it).
override WG: u32 = 256u;

// The coarsest level is swept in workgroup memory when it fits.
const SH: u32 = 640u;
var<workgroup> su: array<vec4<f32>, SH>;
var<workgroup> sf: array<vec4<f32>, SH>;
var<workgroup> sm: array<u32, SH>;

fn s_coarsest(L: Lv, t: u32) {
    let n = L.w * L.h;
    if (n > SH) {
        s_smooth(L, 40u, t);
        return;
    }
    for (var k = t; k < n; k += WG) {
        su[k] = U[L.off + k];
        sf[k] = F[L.off + k];
        sm[k] = mk(L.moff, k);
    }
    workgroupBarrier();
    let w = L.w;
    for (var s = 0u; s < 40u; s++) {
        for (var c = 0u; c < 2u; c++) {
            for (var k = t; k < n; k += WG) {
                let y = k / w;
                let x = k - y * w;
                if (((x + y) & 1u) == c && sm[k] != 0u) {
                    su[k] = (((su[k - 1u] + su[k + 1u]) + (su[k - w] + su[k + w])) - sf[k]) * 0.25;
                }
            }
            workgroupBarrier();
        }
    }
    for (var k = t; k < n; k += WG) {
        U[L.off + k] = su[k];
    }
    storageBarrier();
}

fn s_smooth(L: Lv, sweeps: u32, t: u32) {
    let hw = (L.w + 1u) / 2u;
    let n = hw * L.h;
    for (var s = 0u; s < sweeps; s++) {
        for (var c = 0u; c < 2u; c++) {
            for (var k = t; k < n; k += WG) {
                let y = k / hw;
                let x = 2u * (k - y * hw) + ((y + c) & 1u);
                if (x < L.w) {
                    smooth_cell(L, x, y);
                }
            }
            storageBarrier();
        }
    }
}

// Levels p.lv[0 .. k.x] (finest first): smooth, residual, restrict down to
// the coarsest, 40 sweeps there, then prolong and smooth back up; exactly
// `Solver::vcycle` from the first of these levels.
@compute @workgroup_size(WG)
fn vsmall(@builtin(local_invocation_index) t: u32) {
    touch();
    if (t == 0u) {
        w_stop = u32(stopped());
    }
    if (workgroupUniformLoad(&w_stop) != 0u) {
        return;
    }
    let nl = p.k.x;
    for (var l = 0u; l + 1u < nl; l++) {
        let L = p.lv[l];
        s_smooth(L, 3u, t);
        for (var k = t; k < L.w * L.h; k += WG) {
            _ = residual_cell(L, k % L.w, k / L.w);
        }
        storageBarrier();
        let C = p.lv[l + 1u];
        for (var k = t; k < C.w * C.h; k += WG) {
            restrict_cell(L, C, k % C.w, k / C.w);
        }
        storageBarrier();
    }
    s_coarsest(p.lv[nl - 1u], t);
    for (var l = nl - 1u; l > 0u; l--) {
        let Fl = p.lv[l - 1u];
        let C = p.lv[l];
        for (var k = t; k < Fl.w * Fl.h; k += WG) {
            prolong_cell(C, Fl, k % Fl.w, k / Fl.w);
        }
        storageBarrier();
        s_smooth(Fl, 3u, t);
    }
}
