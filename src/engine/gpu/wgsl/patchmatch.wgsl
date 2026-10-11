// PatchMatch completion (`engine::patchmatch`) on the GPU, for the fine
// pyramid levels: NNF upsampling, the EM loop (search, vote, rescore) and the
// sharp final pass, with the CPU's patch distance (7 x 7 SSD over straight RGB,
// early exit, transform table with 1/64 px source centres, colour bias, focus
// penalty), vote weights and box means.
//
// The CPU's scanline propagation is sequential; here propagation is jump
// flooding: log-step passes in which every target tries its four neighbours
// at distance `step` (their source shifted by the step under their own
// transform), each pass reading one NNF plane and writing the other. The
// random search (the CPU's shrinking-radius candidates and random
// transforms) follows, with each pixel's own xorshift stream seeded by the
// CPU's `hash01` hash.
//
// Pools (offsets in `L`): img (straight RGBA as packed u32, two planes per
// level), lv (flags, 7 x 7 means, focus penalty, valid source list), nn (NNF:
// source index, meta = transform | sub-x << 16 | sub-y << 24, distance; two
// planes), wt (vote weights).

struct L {
    w: u32,
    h: u32,
    n: u32,
    flg: u32,
    mean: u32,
    prox: u32,
    vl: u32,
    vln: u32,
    img: u32,
    img2: u32,
    ns: u32,
    ns2: u32,
    wt: u32,
    p13: u32,
    p14: u32,
    p15: u32,
    // Bounding boxes (x0, y0, x1, y1) of the targets, and of the hole grown
    // by R (where votes and means change); kernels run only over these.
    tb: vec4<u32>,
    mb: vec4<u32>,
}

struct P {
    a: L,
    b: L,
    seed: u32,
    step: u32,
    nxf: u32,
    bias: i32,
    reach: i32,
    maxr: u32,
    center: u32,
    p7: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> img: array<u32>;
@group(0) @binding(2) var<storage, read_write> lv: array<u32>;
@group(0) @binding(3) var<storage, read_write> nn: array<u32>;
@group(0) @binding(4) var<storage, read_write> wt: array<f32>;
@group(0) @binding(5) var<storage, read> xf: array<f32>;
@group(0) @binding(6) var<storage, read_write> ctrl: array<atomic<u32>>;

const R: i32 = 3;
const NONE: u32 = 0xFFFFFFFFu;
const F_HOLE: u32 = 1u;
const F_VALID: u32 = 2u;
const F_WIDE: u32 = 4u;
const F_TGT: u32 = 8u;
// Transform table stride: 49 (u, v) offsets, then the matrix a b c d.
const XS: u32 = 102u;
const BINS2: u32 = 65536u;
const CNT: u32 = 131072u;
const SELHI: u32 = 131073u;
const REM: u32 = 131074u;
const SIG: u32 = 131075u;

fn touch() {
    _ = p.a.w;
    _ = img[0];
    _ = lv[0];
    _ = nn[0];
    _ = wt[0];
    _ = xf[0];
    _ = atomicLoad(&ctrl[0]);
}

fn flags(A: L, i: u32) -> u32 {
    return lv[A.flg + i];
}

fn is_valid(A: L, x: i32, y: i32, k: u32) -> bool {
    let w = i32(A.w);
    let h = i32(A.h);
    if (k == 0u) {
        if (x < R || y < R || x >= w - R || y >= h - R) {
            return false;
        }
        return (flags(A, u32(y * w + x)) & F_VALID) != 0u;
    }
    let rw = p.reach;
    if (x < rw || y < rw || x >= w - rw || y >= h - rw) {
        return false;
    }
    return (flags(A, u32(y * w + x)) & F_WIDE) != 0u;
}

fn rgb(c: u32) -> vec3<i32> {
    return vec3<i32>(i32(c & 255u), i32((c >> 8u) & 255u), i32((c >> 16u) & 255u));
}

fn mean3(A: L, i: u32) -> vec3<i32> {
    return rgb(lv[A.mean + i]);
}

// `Lvl::bias`: target minus source means, clamped.
fn bias(A: L, t: u32, s: u32) -> vec3<i32> {
    if (p.bias == 0) {
        return vec3<i32>(0);
    }
    return clamp(mean3(A, t) - mean3(A, s), vec3<i32>(-p.bias), vec3<i32>(p.bias));
}

// f32::round (half away from zero).
fn rnd(v: f32) -> i32 {
    return i32(select(floor(v + 0.5), ceil(v - 0.5), v < 0.0));
}

fn offf(k: u32, o: u32) -> vec2<f32> {
    return vec2<f32>(xf[k * XS + 2u * o], xf[k * XS + 2u * o + 1u]);
}

// Source offset of patch offset (dx, dy) under transform k: the CPU's table
// within the patch, the transform's matrix beyond it.
fn xoff(k: u32, dx: i32, dy: i32) -> vec2<f32> {
    if (abs(dx) <= R && abs(dy) <= R) {
        return offf(k, u32((dy + R) * 7 + dx + R));
    }
    let b = k * XS + 98u;
    let fx = f32(dx);
    let fy = f32(dy);
    return vec2<f32>(xf[b] * fx + xf[b + 1u] * fy, xf[b + 2u] * fx + xf[b + 3u] * fy);
}

fn sub_of(mt: u32) -> vec2<i32> {
    return vec2<i32>(bitcast<i32>(mt << 8u) >> 24u, bitcast<i32>(mt) >> 24u);
}

fn meta_of(k: u32, sub: vec2<i32>) -> u32 {
    return k | ((bitcast<u32>(sub.x) & 255u) << 16u) | ((bitcast<u32>(sub.y) & 255u) << 24u);
}

// `Lvl::dist`: SSD of the target patch at (tx, ty) and source (sx, sy) under
// transform k, giving up past `best`.
fn dist(A: L, tx: i32, ty: i32, sx: i32, sy: i32, sub: vec2<i32>, k: u32, best: u32) -> u32 {
    let w = i32(A.w);
    let si = u32(sy * w + sx);
    var d = 0u;
    if (A.prox != NONE) {
        d = lv[A.prox + si];
    }
    let bs = bias(A, u32(ty * w + tx), si);
    if (k == 0u && all(bs == vec3<i32>(0))) {
        for (var dy = -R; dy <= R; dy++) {
            let t0 = u32((ty + dy) * w + tx - R);
            let s0 = u32((sy + dy) * w + sx - R);
            var acc = 0u;
            for (var dx = 0u; dx < 7u; dx++) {
                let e = rgb(img[A.img + t0 + dx]) - rgb(img[A.img + s0 + dx]);
                acc += u32(dot(e, e));
            }
            d += acc;
            if (d >= best) {
                return d;
            }
        }
        return d;
    }
    let f = vec2<f32>(f32(sub.x), f32(sub.y)) / 64.0;
    var o = 0u;
    for (var dy = -R; dy <= R; dy++) {
        let t0 = u32((ty + dy) * w + tx - R);
        var acc = 0u;
        for (var dx = 0; dx < 7; dx++) {
            let uv = offf(k, o);
            o += 1u;
            let u = rnd(uv.x + f.x);
            let v = rnd(uv.y + f.y);
            let e = rgb(img[A.img + t0 + u32(dx)]) - rgb(img[A.img + u32((sy + v) * w + sx + u)]) - bs;
            acc += u32(dot(e, e));
        }
        d += acc;
        if (d >= best) {
            return d;
        }
    }
    return d;
}

// The CPU's `split`: integer centre and 1/64 px remainder.
fn split(e: f32) -> vec2<i32> {
    let r = rnd(e);
    return vec2<i32>(r, clamp(rnd((e - f32(r)) * 64.0), -32, 32));
}

struct Best {
    s: u32,
    k: u32,
    sub: vec2<i32>,
    d: u32,
}

fn try_c(A: L, b: ptr<function, Best>, x: i32, y: i32, sx: i32, sy: i32, sub: vec2<i32>, k: u32) {
    if (!is_valid(A, sx, sy, k)) {
        return;
    }
    let si = u32(sy * i32(A.w) + sx);
    if (si == (*b).s && k == (*b).k && all(sub == (*b).sub)) {
        return;
    }
    let d = dist(A, x, y, sx, sy, sub, k, (*b).d);
    if (d < (*b).d) {
        *b = Best(si, k, sub, d);
    }
}

fn load_best(base: u32, n: u32, i: u32) -> Best {
    let m = nn[base + n + i];
    return Best(nn[base + i], m & 0xFFFFu, sub_of(m), nn[base + 2u * n + i]);
}

fn store_best(base: u32, n: u32, i: u32, b: Best) {
    nn[base + i] = b.s;
    nn[base + n + i] = meta_of(b.k, b.sub);
    nn[base + 2u * n + i] = b.d;
}

// One jump-flooding propagation pass (plane ns -> ns2).
@compute @workgroup_size(16, 16)
fn prop(@builtin(global_invocation_id) g0: vec3<u32>) {
    touch();
    let A = p.a;
    let gid = vec3<u32>(g0.x + A.tb.x, g0.y + A.tb.y, 0u);
    if (gid.x >= A.tb.z || gid.y >= A.tb.w) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = gid.y * A.w + gid.x;
    var b = load_best(A.ns, A.n, i);
    if ((flags(A, i) & F_TGT) != 0u) {
        let s = i32(p.step);
        let w = i32(A.w);
        let h = i32(A.h);
        for (var q = 0; q < 4; q++) {
            let nx = x + select(select(0, s, q == 1), -s, q == 0);
            let ny = y + select(select(0, s, q == 3), -s, q == 2);
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) {
                continue;
            }
            let j = u32(ny * w + nx);
            let sj = nn[A.ns + j];
            if (sj == NONE) {
                continue;
            }
            let mj = nn[A.ns + A.n + j];
            let k = mj & 0xFFFFu;
            let ddx = x - nx;
            let ddy = y - ny;
            let jx = i32(sj % A.w);
            let jy = i32(sj / A.w);
            if (k == 0u) {
                try_c(A, &b, x, y, jx + ddx, jy + ddy, vec2<i32>(0), 0u);
            } else {
                let o = xoff(k, ddx, ddy);
                let sb = vec2<f32>(sub_of(mj)) / 64.0;
                let ex = split(f32(jx) + sb.x + o.x);
                let ey = split(f32(jy) + sb.y + o.y);
                try_c(A, &b, x, y, ex.x, ey.x, vec2<i32>(ex.y, ey.y), k);
            }
        }
    }
    store_best(A.ns2, A.n, i, b);
}

var<private> rng: u32;

fn xs() -> u32 {
    var x = rng;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    rng = x;
    return x;
}

// Random search around the current best (in place on plane ns), and with
// adaptation a retest of the best position under a random transform.
@compute @workgroup_size(16, 16)
fn search(@builtin(global_invocation_id) g0: vec3<u32>) {
    touch();
    let A = p.a;
    let gid = vec3<u32>(g0.x + A.tb.x, g0.y + A.tb.y, 0u);
    if (gid.x >= A.tb.z || gid.y >= A.tb.w) {
        return;
    }
    let i = gid.y * A.w + gid.x;
    if ((flags(A, i) & F_TGT) == 0u) {
        return;
    }
    var b = load_best(A.ns, A.n, i);
    if (b.s == NONE) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    rng = hash_u(x, y, p.seed) | 1u;
    let bx = i32(b.s % A.w);
    let by = i32(b.s / A.w);
    let bsb = b.sub;
    var r = p.maxr;
    while (r >= 1u) {
        let rx = bx + i32(xs() % (2u * r + 1u)) - i32(r);
        let ry = by + i32(xs() % (2u * r + 1u)) - i32(r);
        var k = b.k;
        if (p.nxf > 1u && (xs() & 1u) == 0u) {
            k = xs() % p.nxf;
        }
        try_c(A, &b, x, y, rx, ry, vec2<i32>(0), k);
        r /= 2u;
    }
    if (p.nxf > 1u) {
        let k = xs() % p.nxf;
        try_c(A, &b, x, y, bx, by, bsb, k);
    }
    store_best(A.ns, A.n, i, b);
}

// Distances of every target's match against the current image.
@compute @workgroup_size(16, 16)
fn rescore(@builtin(global_invocation_id) g0: vec3<u32>) {
    touch();
    let A = p.a;
    let gid = vec3<u32>(g0.x + A.tb.x, g0.y + A.tb.y, 0u);
    if (gid.x >= A.tb.z || gid.y >= A.tb.w) {
        return;
    }
    let i = gid.y * A.w + gid.x;
    let s = nn[A.ns + i];
    if ((flags(A, i) & F_TGT) == 0u || s == NONE) {
        return;
    }
    let m = nn[A.ns + A.n + i];
    nn[A.ns + 2u * A.n + i] = dist(A, i32(gid.x), i32(gid.y), i32(s % A.w), i32(s / A.w), sub_of(m), m & 0xFFFFu, NONE);
}

// ----- the vote's sigma^2: the 75th percentile of the distances (radix select) -----

@compute @workgroup_size(256)
fn clear_bins(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    if (gid.x < SIG + 1u) {
        atomicStore(&ctrl[gid.x], 0u);
    }
}

@compute @workgroup_size(16, 16)
fn hist1(@builtin(global_invocation_id) g0: vec3<u32>) {
    touch();
    let A = p.a;
    let gid = vec3<u32>(g0.x + A.tb.x, g0.y + A.tb.y, 0u);
    if (gid.x >= A.tb.z || gid.y >= A.tb.w) {
        return;
    }
    let i = gid.y * A.w + gid.x;
    let d = nn[A.ns + 2u * A.n + i];
    if ((flags(A, i) & F_TGT) == 0u || d == NONE) {
        return;
    }
    atomicAdd(&ctrl[d >> 16u], 1u);
    atomicAdd(&ctrl[CNT], 1u);
}

@compute @workgroup_size(16, 16)
fn hist2(@builtin(global_invocation_id) g0: vec3<u32>) {
    touch();
    let A = p.a;
    let gid = vec3<u32>(g0.x + A.tb.x, g0.y + A.tb.y, 0u);
    if (gid.x >= A.tb.z || gid.y >= A.tb.w) {
        return;
    }
    let i = gid.y * A.w + gid.x;
    let d = nn[A.ns + 2u * A.n + i];
    if ((flags(A, i) & F_TGT) == 0u || d == NONE || (d >> 16u) != atomicLoad(&ctrl[SELHI])) {
        return;
    }
    atomicAdd(&ctrl[BINS2 + (d & 0xFFFFu)], 1u);
}

var<workgroup> part: array<u32, 256>;

// Finds the bin of rank `k` among 65536 bins at `base`; returns (bin, rank
// left within it) through thread 0.
fn select_bin(base: u32, k: u32, t: u32) -> vec2<u32> {
    var s = 0u;
    for (var j = 0u; j < 256u; j++) {
        s += atomicLoad(&ctrl[base + t * 256u + j]);
    }
    part[t] = s;
    workgroupBarrier();
    var res = vec2<u32>(0u);
    if (t == 0u) {
        var acc = 0u;
        var c = 0u;
        for (; c < 255u; c++) {
            if (acc + part[c] > k) {
                break;
            }
            acc += part[c];
        }
        var b = c * 256u;
        for (; b < c * 256u + 255u; b++) {
            let v = atomicLoad(&ctrl[base + b]);
            if (acc + v > k) {
                break;
            }
            acc += v;
        }
        res = vec2<u32>(b, k - acc);
    }
    return res;
}

@compute @workgroup_size(256)
fn scan1(@builtin(local_invocation_index) t: u32) {
    touch();
    let n = atomicLoad(&ctrl[CNT]);
    let r = select_bin(0u, n * 3u / 4u, t);
    if (t == 0u) {
        atomicStore(&ctrl[SELHI], r.x);
        atomicStore(&ctrl[REM], r.y);
    }
}

@compute @workgroup_size(256)
fn scan2(@builtin(local_invocation_index) t: u32) {
    touch();
    let r = select_bin(BINS2, atomicLoad(&ctrl[REM]), t);
    if (t == 0u) {
        var sig = 1.0;
        if (atomicLoad(&ctrl[CNT]) > 0u) {
            sig = max(f32((atomicLoad(&ctrl[SELHI]) << 16u) | r.x), 1.0);
        }
        atomicStore(&ctrl[SIG], bitcast<u32>(sig));
    }
}

// One vote weight per target: exp(-d / (2 sigma^2)), at least 1e-6.
@compute @workgroup_size(16, 16)
fn weights(@builtin(global_invocation_id) g0: vec3<u32>) {
    touch();
    let A = p.a;
    let gid = vec3<u32>(g0.x + A.tb.x, g0.y + A.tb.y, 0u);
    if (gid.x >= A.tb.z || gid.y >= A.tb.w) {
        return;
    }
    let i = gid.y * A.w + gid.x;
    var v = 0.0;
    if ((flags(A, i) & F_TGT) != 0u) {
        let s2 = bitcast<f32>(atomicLoad(&ctrl[SIG]));
        v = max(p_exp(p_div(-f32(nn[A.ns + 2u * A.n + i]), 2.0 * s2)), 1e-6);
    }
    wt[A.wt + i] = v;
}

// `vote`: hole pixels from the patches covering them (or, with `center`,
// from their own match), plane img -> img2; other pixels copied.
@compute @workgroup_size(16, 16)
fn vote(@builtin(global_invocation_id) g0: vec3<u32>) {
    touch();
    let A = p.a;
    let gid = vec3<u32>(g0.x + A.mb.x, g0.y + A.mb.y, 0u);
    if (gid.x >= A.mb.z || gid.y >= A.mb.w) {
        return;
    }
    let i = gid.y * A.w + gid.x;
    let old = img[A.img + i];
    var out = old;
    if ((flags(A, i) & F_HOLE) != 0u) {
        if (p.center != 0u) {
            let s = nn[A.ns + i];
            if (s != NONE) {
                let c = clamp(rgb(img[A.img + s]) + bias(A, i, s), vec3<i32>(0), vec3<i32>(255));
                out = u32(c.x) | (u32(c.y) << 8u) | (u32(c.z) << 16u) | 0xFF000000u;
            }
        } else {
            let x = i32(gid.x);
            let y = i32(gid.y);
            let w = i32(A.w);
            let h = i32(A.h);
            var acc = vec3<f32>(0.0);
            var wsum = 0.0;
            for (var dy = -R; dy <= R; dy++) {
                for (var dx = -R; dx <= R; dx++) {
                    let tx = x + dx;
                    let ty = y + dy;
                    if (tx < 0 || ty < 0 || tx >= w || ty >= h) {
                        continue;
                    }
                    let t = u32(ty * w + tx);
                    let s = nn[A.ns + t];
                    if (s == NONE) {
                        continue;
                    }
                    let m = nn[A.ns + A.n + t];
                    let k = m & 0xFFFFu;
                    var uv = vec2<i32>(-dx, -dy);
                    if (k != 0u) {
                        let o = offf(k, u32((-dy + R) * 7 - dx + R));
                        let sb = vec2<f32>(sub_of(m)) / 64.0;
                        uv = vec2<i32>(rnd(o.x + sb.x), rnd(o.y + sb.y));
                    }
                    let sx = i32(s % A.w) + uv.x;
                    let sy = i32(s / A.w) + uv.y;
                    let c = rgb(img[A.img + u32(sy * w + sx)]) + bias(A, t, s);
                    let wv = wt[A.wt + t];
                    acc += vec3<f32>(c) * wv;
                    wsum += wv;
                }
            }
            if (wsum > 0.0) {
                let c = vec3<u32>(clamp(acc / wsum + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(255.0)));
                out = c.x | (c.y << 8u) | (c.z << 16u) | 0xFF000000u;
            }
        }
    }
    img[A.img2 + i] = out;
}

// `Lvl::update_mean`: 7 x 7 box means (clamped window, integer division).
@compute @workgroup_size(16, 16)
fn box_mean(@builtin(global_invocation_id) g0: vec3<u32>) {
    touch();
    let A = p.a;
    let gid = vec3<u32>(g0.x + A.mb.x, g0.y + A.mb.y, 0u);
    if (gid.x >= A.mb.z || gid.y >= A.mb.w) {
        return;
    }
    let x0 = max(i32(gid.x) - R, 0);
    let x1 = min(i32(gid.x) + R + 1, i32(A.w));
    let y0 = max(i32(gid.y) - R, 0);
    let y1 = min(i32(gid.y) + R + 1, i32(A.h));
    var a = vec3<i32>(0);
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            a += rgb(img[A.img + u32(y) * A.w + u32(x)]);
        }
    }
    let n = (y1 - y0) * (x1 - x0);
    let m = vec3<u32>(a / n);
    lv[A.mean + gid.y * A.w + gid.x] = m.x | (m.y << 8u) | (m.z << 16u);
}

// Hole pixels of the fine level (a) from the coarse result (b).
@compute @workgroup_size(16, 16)
fn seed_hole(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    let A = p.a;
    let C = p.b;
    if (gid.x >= A.w || gid.y >= A.h) {
        return;
    }
    let i = gid.y * A.w + gid.x;
    if ((flags(A, i) & F_HOLE) == 0u) {
        return;
    }
    img[A.img + i] = img[C.img + min(gid.y / 2u, C.h - 1u) * C.w + min(gid.x / 2u, C.w - 1u)];
}

// (a * b) mod m for 32-bit a, b (the CPU computes it in 64 bits).
fn mulmod(a: u32, b: u32, m: u32) -> u32 {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let lo0 = a0 * b0;
    let mid1 = a1 * b0;
    let mid2 = a0 * b1;
    let hi0 = a1 * b1;
    let t = (lo0 >> 16u) + (mid1 & 0xFFFFu) + (mid2 & 0xFFFFu);
    let lo = (lo0 & 0xFFFFu) | (t << 16u);
    let hi = hi0 + (mid1 >> 16u) + (mid2 >> 16u) + (t >> 16u);
    // (hi * 2^32 + lo) mod m by doubling.
    var r = hi % m;
    for (var j = 0u; j < 32u; j++) {
        r = select(r + r, r - (m - r), r >= m - r);
    }
    let l = lo % m;
    return select(r + l, r - (m - l), r >= m - l);
}

// The NNF of the fine level (a, plane ns) from the coarse one (b, plane ns).
@compute @workgroup_size(16, 16)
fn upsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    let A = p.a;
    let C = p.b;
    if (gid.x >= A.w || gid.y >= A.h) {
        return;
    }
    let i = gid.y * A.w + gid.x;
    var out = Best(NONE, 0u, vec2<i32>(0), NONE);
    if ((flags(A, i) & F_TGT) != 0u) {
        let x = i32(gid.x);
        let y = i32(gid.y);
        let ci = (gid.y / 2u) * C.w + gid.x / 2u;
        var s = NONE;
        var cm = 0u;
        if (ci < C.n) {
            s = nn[C.ns + ci];
            cm = nn[C.ns + C.n + ci];
        }
        let k = cm & 0xFFFFu;
        var ok = false;
        if (s != NONE) {
            let ox = x % 2;
            let oy = y % 2;
            let cx = i32(s % C.w);
            let cy = i32(s / C.w);
            var sx = cx * 2 + ox;
            var sy = cy * 2 + oy;
            var sub = vec2<i32>(0);
            if (k != 0u) {
                let o = offf(k, u32((oy + R) * 7 + ox + R));
                let sb = vec2<f32>(sub_of(cm)) / 64.0;
                let ex = split((f32(cx) + sb.x) * 2.0 + o.x);
                let ey = split((f32(cy) + sb.y) * 2.0 + o.y);
                sx = ex.x;
                sy = ey.x;
                sub = vec2<i32>(ex.y, ey.y);
            }
            if (is_valid(A, sx, sy, k)) {
                out = Best(u32(sy * i32(A.w) + sx), k, sub, NONE);
                ok = true;
            }
        }
        if (!ok) {
            out = Best(lv[A.vl + mulmod(i, 2654435761u, max(A.vln, 1u))], 0u, vec2<i32>(0), NONE);
        }
    }
    store_best(A.ns, A.n, i, out);
    store_best(A.ns2, A.n, i, out);
}

// Level start: the other image plane = the current one (votes then only
// rewrite the hole's box).
@compute @workgroup_size(256)
fn copy_img(@builtin(global_invocation_id) gid: vec3<u32>) {
    touch();
    let A = p.a;
    let i = gid.y * 65535u * 256u + gid.x;
    if (i < A.n) {
        img[A.img2 + i] = img[A.img + i];
    }
}
