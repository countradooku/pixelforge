// Rank filters with small or separable windows, bit-exact with the CPU
// (`filters::rank`, `filters::fast`): Maximum / Minimum (exact window max /
// min, separable), the 3x3 and 5x5 medians (Median, Dust & Scratches,
// Despeckle), and the copy / channel-merge helpers of the histogram kernels.
//
// Every result goes through the CPU's `merge`: color never exceeds alpha.

struct P {
    // Padded input size, output (center) size.
    pw: u32,
    ph: u32,
    cw: u32,
    ch: u32,
    // Window radius (the input is padded by it).
    r: u32,
    // Entry-specific: 1 = maximum (max/min); dust threshold; ...
    mode: u32,
    thr: u32,
    // Segment length (outputs per workgroup) / tile width of a pass.
    seg: u32,
    tw: u32,
    a0: u32,
    a1: u32,
    a2: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> aux: array<u32>;

// `merge`: color clamped to alpha.
fn merged(c: vec4<u32>) -> u32 {
    return pack4(vec4<u32>(min(c.xyz, vec3<u32>(c.w)), c.w));
}

// Copies the center (radius 0: the CPU leaves the buffer untouched).
@compute @workgroup_size(16, 16)
fn copy_center(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    dst[gid.y * p.cw + gid.x] = src[(gid.y + p.r) * p.pw + gid.x + p.r];
}

// Per-channel results assembled in `aux` (center-sized), clamped like
// `merge` into `dst`.
@compute @workgroup_size(16, 16)
fn merge_center(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let i = gid.y * p.cw + gid.x;
    dst[i] = merged(unpack4(aux[i]));
}

// ----- Maximum / Minimum ------------------------------------------------------------
// Minimum is the maximum of the complement (255 - v), so every pass is a max.
// p.mode: 1 = maximum, 0 = minimum.

fn mm_in(v: u32) -> u32 {
    return select(~v, v, p.mode == 1u);
}

fn max_px(a: u32, b: u32) -> u32 {
    return pack4(max(unpack4(a), unpack4(b)));
}

// Small radii (r <= 8): a 32 x 32 output tile, its apron in shared memory,
// direct horizontal then vertical maxima.
var<workgroup> mt_in: array<u32, 2304>;
var<workgroup> mt_h: array<u32, 1536>;

@compute @workgroup_size(256)
fn maxmin_tile(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r = p.r;
    let n = 32u + 2u * r;
    let bx = wid.x * 32u;
    let by = wid.y * 32u;
    for (var t = li; t < n * n; t += 256u) {
        let x = min(bx + t % n, p.pw - 1u);
        let y = min(by + t / n, p.ph - 1u);
        mt_in[t] = mm_in(src[y * p.pw + x]);
    }
    workgroupBarrier();
    for (var t = li; t < n * 32u; t += 256u) {
        let row = t / 32u;
        let col = t % 32u;
        var m = unpack4(mt_in[row * n + col]);
        for (var j = 1u; j <= 2u * r; j++) {
            m = max(m, unpack4(mt_in[row * n + col + j]));
        }
        mt_h[t] = pack4(m);
    }
    workgroupBarrier();
    for (var t = li; t < 1024u; t += 256u) {
        let x = bx + t % 32u;
        let y = by + t / 32u;
        var m = unpack4(mt_h[t]);
        for (var j = 1u; j <= 2u * r; j++) {
            m = max(m, unpack4(mt_h[t + j * 32u]));
        }
        if (x < p.cw && y < p.ch) {
            dst[y * p.cw + x] = merged(unpack4(mm_in(pack4(m))));
        }
    }
}

// Larger radii: separable passes. A run of `len` values sits in shared
// memory (stride `lanes`); after the doubling steps element i is the max of
// [i, i + pw2), and a window of n = 2r + 1 is max(m[i], m[i + n - pw2]).
// Two buffers ping-pong (no private arrays: they spill on Metal).
var<workgroup> ms: array<u32, 4096>;

// Doubling over runs of `len` elements, `lanes` interleaved runs (element
// e of run l at e * lanes + l) in ms[0..2048). Returns (pw2, offset of the
// buffer holding the result). Uniform control flow.
fn doubling(len: u32, lanes: u32, n: u32, li: u32) -> vec2<u32> {
    var s = 1u;
    var cur = 0u;
    loop {
        if (2u * s > n) {
            break;
        }
        let nxt = 2048u - cur;
        for (var t = li; t < len * lanes; t += 256u) {
            var v = ms[cur + t];
            if (t / lanes + s < len) {
                v = max_px(v, ms[cur + t + s * lanes]);
            }
            ms[nxt + t] = v;
        }
        workgroupBarrier();
        cur = nxt;
        s *= 2u;
    }
    return vec2<u32>(s, cur);
}

// Horizontal: one row segment of p.seg outputs per workgroup, into aux-sized
// `dst` (p.cw x p.ph).
@compute @workgroup_size(256)
fn maxmin_h(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = 2u * p.r + 1u;
    let x0 = wid.x * p.seg;
    let y = wid.y;
    let len = min(p.seg + n - 1u, p.pw - x0);
    for (var t = li; t < len; t += 256u) {
        ms[t] = mm_in(src[y * p.pw + x0 + t]);
    }
    workgroupBarrier();
    let d = doubling(len, 1u, n, li);
    for (var t = li; t < p.seg; t += 256u) {
        let x = x0 + t;
        if (x < p.cw) {
            dst[y * p.cw + x] = max_px(ms[d.y + t], ms[d.y + t + n - d.x]);
        }
    }
}

// Vertical over the horizontal result (`aux`, p.cw x p.ph): a strip of
// p.tw columns and p.seg output rows per workgroup.
@compute @workgroup_size(256)
fn maxmin_v(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let n = 2u * p.r + 1u;
    let tw = p.tw;
    let x0 = wid.x * tw;
    let y0 = wid.y * p.seg;
    let len = min(p.seg + n - 1u, p.ph - y0);
    for (var t = li; t < len * tw; t += 256u) {
        let x = min(x0 + t % tw, p.cw - 1u);
        ms[t] = aux[(y0 + t / tw) * p.cw + x];
    }
    workgroupBarrier();
    let d = doubling(len, tw, n, li);
    for (var t = li; t < p.seg * tw; t += 256u) {
        let x = x0 + t % tw;
        let y = y0 + t / tw;
        if (x < p.cw && y < p.ch) {
            dst[y * p.cw + x] = merged(unpack4(mm_in(max_px(ms[d.y + t], ms[d.y + t + (n - d.x) * tw]))));
        }
    }
}

// ----- 3x3 and 5x5 medians --------------------------------------------------------
// A 16 x 16 output tile with its apron in shared memory. Exact medians by
// sorting networks (any exact network gives the CPU's value).

var<workgroup> nt: array<u32, 400>;

fn load_tile(wid: vec3<u32>, li: u32, r: u32) {
    let n = 16u + 2u * r;
    let bx = wid.x * 16u;
    let by = wid.y * 16u;
    for (var t = li; t < n * n; t += 256u) {
        let x = min(bx + t % n, p.pw - 1u);
        let y = min(by + t / n, p.ph - 1u);
        nt[t] = src[y * p.pw + x];
    }
    workgroupBarrier();
}

fn tv(lx: u32, ly: u32, n: u32) -> vec4<u32> {
    return unpack4(nt[ly * n + lx]);
}

// Median of 9: sort the three columns, then med3(max of lows, med3 of
// mids, min of highs).
fn med3(a: vec4<u32>, b: vec4<u32>, c: vec4<u32>) -> vec4<u32> {
    return max(min(a, b), min(max(a, b), c));
}

fn sort3(a: vec4<u32>, b: vec4<u32>, c: vec4<u32>) -> array<vec4<u32>, 3> {
    let l1 = min(a, b);
    let h1 = max(a, b);
    return array<vec4<u32>, 3>(min(l1, c), min(h1, max(l1, c)), max(h1, c));
}

fn median9_at(lx: u32, ly: u32) -> vec4<u32> {
    let n = 18u;
    let c0 = sort3(tv(lx, ly, n), tv(lx, ly + 1u, n), tv(lx, ly + 2u, n));
    let c1 = sort3(tv(lx + 1u, ly, n), tv(lx + 1u, ly + 1u, n), tv(lx + 1u, ly + 2u, n));
    let c2 = sort3(tv(lx + 2u, ly, n), tv(lx + 2u, ly + 1u, n), tv(lx + 2u, ly + 2u, n));
    let mxl = max(max(c0[0], c1[0]), c2[0]);
    let mnh = min(min(c0[2], c1[2]), c2[2]);
    let mm = med3(c0[1], c1[1], c2[1]);
    return med3(mxl, mm, mnh);
}

// Median of 25 by forgetful selection (as `fast::median5`), unrolled so
// every value stays in a register (generated; 168 compare-exchanges).
fn median25_at(lx: u32, ly: u32) -> vec4<u32> {
    let n = 20u;
    var v0 = tv(lx + 0u, ly + 0u, n);
    var v1 = tv(lx + 1u, ly + 0u, n);
    var v2 = tv(lx + 2u, ly + 0u, n);
    var v3 = tv(lx + 3u, ly + 0u, n);
    var v4 = tv(lx + 4u, ly + 0u, n);
    var v5 = tv(lx + 0u, ly + 1u, n);
    var v6 = tv(lx + 1u, ly + 1u, n);
    var v7 = tv(lx + 2u, ly + 1u, n);
    var v8 = tv(lx + 3u, ly + 1u, n);
    var v9 = tv(lx + 4u, ly + 1u, n);
    var v10 = tv(lx + 0u, ly + 2u, n);
    var v11 = tv(lx + 1u, ly + 2u, n);
    var v12 = tv(lx + 2u, ly + 2u, n);
    var v13 = tv(lx + 3u, ly + 2u, n);
    var t: vec4<u32>;
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v0, v4); v4 = max(v0, v4); v0 = t;
    t = min(v0, v5); v5 = max(v0, v5); v0 = t;
    t = min(v0, v6); v6 = max(v0, v6); v0 = t;
    t = min(v0, v7); v7 = max(v0, v7); v0 = t;
    t = min(v0, v8); v8 = max(v0, v8); v0 = t;
    t = min(v0, v9); v9 = max(v0, v9); v0 = t;
    t = min(v0, v10); v10 = max(v0, v10); v0 = t;
    t = min(v0, v11); v11 = max(v0, v11); v0 = t;
    t = min(v0, v12); v12 = max(v0, v12); v0 = t;
    t = min(v0, v13); v13 = max(v0, v13); v0 = t;
    t = min(v1, v13); v13 = max(v1, v13); v1 = t;
    t = min(v2, v13); v13 = max(v2, v13); v2 = t;
    t = min(v3, v13); v13 = max(v3, v13); v3 = t;
    t = min(v4, v13); v13 = max(v4, v13); v4 = t;
    t = min(v5, v13); v13 = max(v5, v13); v5 = t;
    t = min(v6, v13); v13 = max(v6, v13); v6 = t;
    t = min(v7, v13); v13 = max(v7, v13); v7 = t;
    t = min(v8, v13); v13 = max(v8, v13); v8 = t;
    t = min(v9, v13); v13 = max(v9, v13); v9 = t;
    t = min(v10, v13); v13 = max(v10, v13); v10 = t;
    t = min(v11, v13); v13 = max(v11, v13); v11 = t;
    t = min(v12, v13); v13 = max(v12, v13); v12 = t;
    v0 = tv(lx + 4u, ly + 2u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v0, v4); v4 = max(v0, v4); v0 = t;
    t = min(v0, v5); v5 = max(v0, v5); v0 = t;
    t = min(v0, v6); v6 = max(v0, v6); v0 = t;
    t = min(v0, v7); v7 = max(v0, v7); v0 = t;
    t = min(v0, v8); v8 = max(v0, v8); v0 = t;
    t = min(v0, v9); v9 = max(v0, v9); v0 = t;
    t = min(v0, v10); v10 = max(v0, v10); v0 = t;
    t = min(v0, v11); v11 = max(v0, v11); v0 = t;
    t = min(v0, v12); v12 = max(v0, v12); v0 = t;
    t = min(v1, v12); v12 = max(v1, v12); v1 = t;
    t = min(v2, v12); v12 = max(v2, v12); v2 = t;
    t = min(v3, v12); v12 = max(v3, v12); v3 = t;
    t = min(v4, v12); v12 = max(v4, v12); v4 = t;
    t = min(v5, v12); v12 = max(v5, v12); v5 = t;
    t = min(v6, v12); v12 = max(v6, v12); v6 = t;
    t = min(v7, v12); v12 = max(v7, v12); v7 = t;
    t = min(v8, v12); v12 = max(v8, v12); v8 = t;
    t = min(v9, v12); v12 = max(v9, v12); v9 = t;
    t = min(v10, v12); v12 = max(v10, v12); v10 = t;
    t = min(v11, v12); v12 = max(v11, v12); v11 = t;
    v0 = tv(lx + 0u, ly + 3u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v0, v4); v4 = max(v0, v4); v0 = t;
    t = min(v0, v5); v5 = max(v0, v5); v0 = t;
    t = min(v0, v6); v6 = max(v0, v6); v0 = t;
    t = min(v0, v7); v7 = max(v0, v7); v0 = t;
    t = min(v0, v8); v8 = max(v0, v8); v0 = t;
    t = min(v0, v9); v9 = max(v0, v9); v0 = t;
    t = min(v0, v10); v10 = max(v0, v10); v0 = t;
    t = min(v0, v11); v11 = max(v0, v11); v0 = t;
    t = min(v1, v11); v11 = max(v1, v11); v1 = t;
    t = min(v2, v11); v11 = max(v2, v11); v2 = t;
    t = min(v3, v11); v11 = max(v3, v11); v3 = t;
    t = min(v4, v11); v11 = max(v4, v11); v4 = t;
    t = min(v5, v11); v11 = max(v5, v11); v5 = t;
    t = min(v6, v11); v11 = max(v6, v11); v6 = t;
    t = min(v7, v11); v11 = max(v7, v11); v7 = t;
    t = min(v8, v11); v11 = max(v8, v11); v8 = t;
    t = min(v9, v11); v11 = max(v9, v11); v9 = t;
    t = min(v10, v11); v11 = max(v10, v11); v10 = t;
    v0 = tv(lx + 1u, ly + 3u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v0, v4); v4 = max(v0, v4); v0 = t;
    t = min(v0, v5); v5 = max(v0, v5); v0 = t;
    t = min(v0, v6); v6 = max(v0, v6); v0 = t;
    t = min(v0, v7); v7 = max(v0, v7); v0 = t;
    t = min(v0, v8); v8 = max(v0, v8); v0 = t;
    t = min(v0, v9); v9 = max(v0, v9); v0 = t;
    t = min(v0, v10); v10 = max(v0, v10); v0 = t;
    t = min(v1, v10); v10 = max(v1, v10); v1 = t;
    t = min(v2, v10); v10 = max(v2, v10); v2 = t;
    t = min(v3, v10); v10 = max(v3, v10); v3 = t;
    t = min(v4, v10); v10 = max(v4, v10); v4 = t;
    t = min(v5, v10); v10 = max(v5, v10); v5 = t;
    t = min(v6, v10); v10 = max(v6, v10); v6 = t;
    t = min(v7, v10); v10 = max(v7, v10); v7 = t;
    t = min(v8, v10); v10 = max(v8, v10); v8 = t;
    t = min(v9, v10); v10 = max(v9, v10); v9 = t;
    v0 = tv(lx + 2u, ly + 3u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v0, v4); v4 = max(v0, v4); v0 = t;
    t = min(v0, v5); v5 = max(v0, v5); v0 = t;
    t = min(v0, v6); v6 = max(v0, v6); v0 = t;
    t = min(v0, v7); v7 = max(v0, v7); v0 = t;
    t = min(v0, v8); v8 = max(v0, v8); v0 = t;
    t = min(v0, v9); v9 = max(v0, v9); v0 = t;
    t = min(v1, v9); v9 = max(v1, v9); v1 = t;
    t = min(v2, v9); v9 = max(v2, v9); v2 = t;
    t = min(v3, v9); v9 = max(v3, v9); v3 = t;
    t = min(v4, v9); v9 = max(v4, v9); v4 = t;
    t = min(v5, v9); v9 = max(v5, v9); v5 = t;
    t = min(v6, v9); v9 = max(v6, v9); v6 = t;
    t = min(v7, v9); v9 = max(v7, v9); v7 = t;
    t = min(v8, v9); v9 = max(v8, v9); v8 = t;
    v0 = tv(lx + 3u, ly + 3u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v0, v4); v4 = max(v0, v4); v0 = t;
    t = min(v0, v5); v5 = max(v0, v5); v0 = t;
    t = min(v0, v6); v6 = max(v0, v6); v0 = t;
    t = min(v0, v7); v7 = max(v0, v7); v0 = t;
    t = min(v0, v8); v8 = max(v0, v8); v0 = t;
    t = min(v1, v8); v8 = max(v1, v8); v1 = t;
    t = min(v2, v8); v8 = max(v2, v8); v2 = t;
    t = min(v3, v8); v8 = max(v3, v8); v3 = t;
    t = min(v4, v8); v8 = max(v4, v8); v4 = t;
    t = min(v5, v8); v8 = max(v5, v8); v5 = t;
    t = min(v6, v8); v8 = max(v6, v8); v6 = t;
    t = min(v7, v8); v8 = max(v7, v8); v7 = t;
    v0 = tv(lx + 4u, ly + 3u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v0, v4); v4 = max(v0, v4); v0 = t;
    t = min(v0, v5); v5 = max(v0, v5); v0 = t;
    t = min(v0, v6); v6 = max(v0, v6); v0 = t;
    t = min(v0, v7); v7 = max(v0, v7); v0 = t;
    t = min(v1, v7); v7 = max(v1, v7); v1 = t;
    t = min(v2, v7); v7 = max(v2, v7); v2 = t;
    t = min(v3, v7); v7 = max(v3, v7); v3 = t;
    t = min(v4, v7); v7 = max(v4, v7); v4 = t;
    t = min(v5, v7); v7 = max(v5, v7); v5 = t;
    t = min(v6, v7); v7 = max(v6, v7); v6 = t;
    v0 = tv(lx + 0u, ly + 4u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v0, v4); v4 = max(v0, v4); v0 = t;
    t = min(v0, v5); v5 = max(v0, v5); v0 = t;
    t = min(v0, v6); v6 = max(v0, v6); v0 = t;
    t = min(v1, v6); v6 = max(v1, v6); v1 = t;
    t = min(v2, v6); v6 = max(v2, v6); v2 = t;
    t = min(v3, v6); v6 = max(v3, v6); v3 = t;
    t = min(v4, v6); v6 = max(v4, v6); v4 = t;
    t = min(v5, v6); v6 = max(v5, v6); v5 = t;
    v0 = tv(lx + 1u, ly + 4u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v0, v4); v4 = max(v0, v4); v0 = t;
    t = min(v0, v5); v5 = max(v0, v5); v0 = t;
    t = min(v1, v5); v5 = max(v1, v5); v1 = t;
    t = min(v2, v5); v5 = max(v2, v5); v2 = t;
    t = min(v3, v5); v5 = max(v3, v5); v3 = t;
    t = min(v4, v5); v5 = max(v4, v5); v4 = t;
    v0 = tv(lx + 2u, ly + 4u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v0, v4); v4 = max(v0, v4); v0 = t;
    t = min(v1, v4); v4 = max(v1, v4); v1 = t;
    t = min(v2, v4); v4 = max(v2, v4); v2 = t;
    t = min(v3, v4); v4 = max(v3, v4); v3 = t;
    v0 = tv(lx + 3u, ly + 4u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v0, v3); v3 = max(v0, v3); v0 = t;
    t = min(v1, v3); v3 = max(v1, v3); v1 = t;
    t = min(v2, v3); v3 = max(v2, v3); v2 = t;
    v0 = tv(lx + 4u, ly + 4u, n);
    t = min(v0, v1); v1 = max(v0, v1); v0 = t;
    t = min(v0, v2); v2 = max(v0, v2); v0 = t;
    t = min(v1, v2); v2 = max(v1, v2); v1 = t;
    return v1;
}

// p.r = 1 or 2; p.mode: 0 = median, 1 = dust & scratches (p.thr),
// 2 = despeckle.
@compute @workgroup_size(16, 16)
fn median_small(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r = p.r;
    load_tile(wid, li, r);
    let n = 16u + 2u * r;
    let x = wid.x * 16u + lid.x;
    let y = wid.y * 16u + lid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    var m: vec4<u32>;
    if (r == 1u) {
        m = median9_at(lid.x, lid.y);
    } else {
        m = median25_at(lid.x, lid.y);
    }
    let c = tv(lid.x + r, lid.y + r, n);
    if (p.mode == 1u) {
        let d = max(m, c) - min(m, c);
        m = select(m, c, d <= vec4<u32>(p.thr));
    }
    var out = merged(m);
    if (p.mode == 2u) {
        // `despeckle`: keep the original across strong luminance edges.
        let gx = lum(tv(lid.x + 2u, lid.y + 1u, n)) - lum(tv(lid.x, lid.y + 1u, n));
        let gy = lum(tv(lid.x + 1u, lid.y + 2u, n)) - lum(tv(lid.x + 1u, lid.y, n));
        if (abs(gx) + abs(gy) > 48) {
            out = pack4(c);
        }
    }
    dst[y * p.cw + x] = out;
}

fn lum(c: vec4<u32>) -> i32 {
    return i32(c.x * 77u + c.y * 150u + c.z * 29u) >> 8u;
}
