
// ----- generic passes of the Filter Gallery (`filters::gallery` building blocks)

// Dimensions of window z for this pass (`p.i.w` = border extension of
// `with_border` / `median_clamped` copies).
fn dims(z: u32) -> vec2<u32> {
    let e = u32(max(p.i.w, 0));
    return vec2<u32>(win[z].w + 2u * e, win[z].h + 2u * e);
}

fn at0(z: u32, x: i32, y: i32, d: vec2<u32>) -> u32 {
    let cx = u32(clamp(x, 0, i32(d.x) - 1));
    let cy = u32(clamp(y, 0, i32(d.y) - 1));
    return wbase(z) + cy * p.ww + cx;
}

fn unpremul_ch(v: u32, a: u32) -> u32 {
    let i = a * 256u + v;
    return (t0[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn straight4(px: u32) -> vec4<u32> {
    let c = unpack4(px);
    if (c.w == 0u) {
        return vec4<u32>(0u);
    }
    if (c.w == 255u) {
        return c;
    }
    return vec4<u32>(unpremul_ch(c.x, c.w), unpremul_ch(c.y, c.w), unpremul_ch(c.z, c.w), c.w);
}

// `Img::load` of the band input: straight f32 planes b0..b3.
// t0 = unpremultiply table, t1 = v / 255 table.
@compute @workgroup_size(16, 16)
fn load(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let c = straight4(src[(w.sy + id.y) * p.srcw + w.sx + id.x]);
    let i = widx(z, id.x, id.y);
    b0[i] = tabf(t1[c.x]);
    b1[i] = tabf(t1[c.y]);
    b2[i] = tabf(t1[c.z]);
    b3[i] = tabf(t1[c.w]);
}

// `Img::store` of the window center into the band output.
@compute @workgroup_size(16, 16)
fn store(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.cw || id.y >= w.ch) {
        return;
    }
    let i = widx(z, w.cx + id.x, w.cy + id.y);
    let c = vec3<u32>(q8(a0[i]), q8(a1[i]), q8(a2[i]));
    dst[(w.dy + id.y) * p.dstw + w.dx + id.x] = premul(c, q8(a3[i]));
}

// `Img::store` into a packed window plane (v0).
@compute @workgroup_size(16, 16)
fn pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    v0[i] = premul(vec3<u32>(q8(a0[i]), q8(a1[i]), q8(a2[i])), q8(a3[i]));
}

// `Img::load` of a packed window plane (u0) into b0..b3.
@compute @workgroup_size(16, 16)
fn unpack(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let c = straight4(u0[i]);
    b0[i] = tabf(t1[c.x]);
    b1[i] = tabf(t1[c.y]);
    b2[i] = tabf(t1[c.z]);
    b3[i] = tabf(t1[c.w]);
}

// Copies plane a0 (window size) into b0 extended by e = p.i.w edge-clamped
// pixels on every side.
@compute @workgroup_size(16, 16)
fn extend(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let e = i32(p.i.w);
    let w = win[z];
    let sx = clamp(i32(id.x) - e, 0, i32(w.w) - 1);
    let sy = clamp(i32(id.y) - e, 0, i32(w.h) - 1);
    b0[widx(z, id.x, id.y)] = a0[widx(z, u32(sx), u32(sy))];
}

// The inverse: the interior of an extended plane a0 back to window size.
@compute @workgroup_size(16, 16)
fn crop(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let e = u32(p.i.w);
    b0[widx(z, id.x, id.y)] = a0[widx(z, id.x + e, id.y + e)];
}

@compute @workgroup_size(16, 16)
fn copy(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    b0[i] = a0[i];
}

// 0.299 r + 0.587 g + 0.114 b, the CPU's rounding order (p.f = weights).
@compute @workgroup_size(16, 16)
fn lum(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    b0[i] = fadd(fadd(fmul(p.f.x, a0[i]), fmul(p.f.y, a1[i])), fmul(p.f.z, a2[i]));
}

// max(r, g, b) (HSB brightness).
@compute @workgroup_size(16, 16)
fn vmax(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    b0[i] = max(max(a0[i], a1[i]), a2[i]);
}

// ----- three-box Gaussian (`blur`), f32 running sums ----------------------------

// One horizontal box of radius p.i.x (p.f.x = 1 / (2r + 1)): thread per row.
@compute @workgroup_size(64)
fn box_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    let y = id.x;
    if (y >= d.y) {
        return;
    }
    let r = p.i.x;
    let inv = p.f.x;
    let base = wbase(z) + y * p.ww;
    let wm = i32(d.x) - 1;
    var s = 0.0;
    for (var t = -r; t <= r; t++) {
        s = fadd(s, a0[base + u32(clamp(t, 0, wm))]);
    }
    for (var x = 0; x <= wm; x++) {
        b0[base + u32(x)] = fmul(s, inv);
        s = fadd(s, fsub(a0[base + u32(clamp(x + r + 1, 0, wm))], a0[base + u32(clamp(x - r, 0, wm))]));
    }
}

// One vertical box: thread per column.
@compute @workgroup_size(64)
fn box_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    let x = id.x;
    if (x >= d.x) {
        return;
    }
    let r = p.i.x;
    let inv = p.f.x;
    let base = wbase(z) + x;
    let hm = i32(d.y) - 1;
    var c = 0.0;
    for (var t = -r; t <= r; t++) {
        c = fadd(c, a0[base + u32(clamp(t, 0, hm)) * p.ww]);
    }
    for (var y = 0; y <= hm; y++) {
        b0[base + u32(y) * p.ww] = fmul(c, inv);
        c = fadd(c, fsub(a0[base + u32(clamp(y + r + 1, 0, hm)) * p.ww], a0[base + u32(clamp(y - r, 0, hm)) * p.ww]));
    }
}

// ----- exact separable Gaussian (direct taps, t0 = kernel, p.i.x = taps) -------

@compute @workgroup_size(16, 16)
fn gauss_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let n = p.i.x;
    let r = n / 2;
    let base = wbase(z) + id.y * p.ww;
    let wm = i32(d.x) - 1;
    var acc = 0.0;
    for (var j = 0; j < n; j++) {
        acc = fadd(acc, fmul(tabf(t0[j]), a0[base + u32(clamp(i32(id.x) + j - r, 0, wm))]));
    }
    b0[base + id.x] = acc;
}

@compute @workgroup_size(16, 16)
fn gauss_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let n = p.i.x;
    let r = n / 2;
    let base = wbase(z) + id.x;
    let hm = i32(d.y) - 1;
    var acc = 0.0;
    for (var j = 0; j < n; j++) {
        acc = fadd(acc, fmul(tabf(t0[j]), a0[base + u32(clamp(i32(id.y) + j - r, 0, hm)) * p.ww]));
    }
    b0[base + id.y * p.ww] = acc;
}

// ----- integer box sums (`artistic2::boxes`) ----------------------------------------

// v0 = round(a0 * 255) (`(x * 255.0).round().max(0.0) as u64`).
@compute @workgroup_size(16, 16)
fn iquant(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    v0[i] = u32(max(rround(fmul(a0[i], 255.0)), 0.0));
}

@compute @workgroup_size(64)
fn ibox_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    let y = id.x;
    if (y >= d.y) {
        return;
    }
    let r = p.i.x;
    let base = wbase(z) + y * p.ww;
    let wm = i32(d.x) - 1;
    var s = 0u;
    for (var t = -r; t <= r; t++) {
        s += u0[base + u32(clamp(t, 0, wm))];
    }
    for (var x = 0; x <= wm; x++) {
        v0[base + u32(x)] = s;
        s = s + u0[base + u32(clamp(x + r + 1, 0, wm))] - u0[base + u32(clamp(x - r, 0, wm))];
    }
}

@compute @workgroup_size(64)
fn ibox_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    let x = id.x;
    if (x >= d.x) {
        return;
    }
    let r = p.i.x;
    let base = wbase(z) + x;
    let hm = i32(d.y) - 1;
    var c = 0u;
    for (var t = -r; t <= r; t++) {
        c += u0[base + u32(clamp(t, 0, hm)) * p.ww];
    }
    for (var y = 0; y <= hm; y++) {
        v0[base + u32(y) * p.ww] = c;
        c = c + u0[base + u32(clamp(y + r + 1, 0, hm)) * p.ww] - u0[base + u32(clamp(y - r, 0, hm)) * p.ww];
    }
}

// b0 = u0 / p.f.x, correctly rounded (`(x as f64 / n) as f32`).
@compute @workgroup_size(16, 16)
fn idiv(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    b0[i] = fdiv(f32(u0[i]), p.f.x);
}

// ----- 3 x 3 float median (edge clamped) -------------------------------------------

fn sort2(a: ptr<function, f32>, b: ptr<function, f32>) {
    let lo = min(*a, *b);
    let hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

@compute @workgroup_size(16, 16)
fn median3f(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    var v0_ = a0[at0(z, x - 1, y - 1, d)];
    var v1_ = a0[at0(z, x, y - 1, d)];
    var v2_ = a0[at0(z, x + 1, y - 1, d)];
    var v3_ = a0[at0(z, x - 1, y, d)];
    var v4_ = a0[at0(z, x, y, d)];
    var v5_ = a0[at0(z, x + 1, y, d)];
    var v6_ = a0[at0(z, x - 1, y + 1, d)];
    var v7_ = a0[at0(z, x, y + 1, d)];
    var v8_ = a0[at0(z, x + 1, y + 1, d)];
    // Paeth's 19-exchange median-of-9 network.
    sort2(&v1_, &v2_);
    sort2(&v4_, &v5_);
    sort2(&v7_, &v8_);
    sort2(&v0_, &v1_);
    sort2(&v3_, &v4_);
    sort2(&v6_, &v7_);
    sort2(&v1_, &v2_);
    sort2(&v4_, &v5_);
    sort2(&v7_, &v8_);
    sort2(&v0_, &v3_);
    sort2(&v5_, &v8_);
    sort2(&v4_, &v7_);
    sort2(&v3_, &v6_);
    sort2(&v1_, &v4_);
    sort2(&v2_, &v5_);
    sort2(&v4_, &v7_);
    sort2(&v4_, &v2_);
    sort2(&v6_, &v4_);
    sort2(&v4_, &v2_);
    b0[widx(z, id.x, id.y)] = v4_;
}

// ----- rank filters on 8-bit values -------------------------------------------------

// Square median of radius r = p.i.x over each channel of a packed plane (u0),
// as `rank::median`: the r-pixel frame is copied, and alpha is filtered too
// unless every pixel of the (extended) window is opaque (flag in u1[z]).
var<workgroup> tile: array<u32, 2304>;

// u1-style flag per window: 1 if any pixel of the (extended) window is not
// opaque. Thread per row; v1 must be cleared before.
@compute @workgroup_size(64)
fn not_opaque(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    let y = id.x;
    if (y >= d.y) {
        return;
    }
    var any = false;
    let base = wbase(z) + y * p.ww;
    for (var x = 0u; x < d.x; x++) {
        any = any || (u0[base + x] >> 24u) != 255u;
    }
    if (any) {
        v1[z] = 1u;
    }
}

// Kuwahara of radius r = p.i.x on a packed plane (u0 -> v0), as
// `stylize::kuwahara`: quadrant of least luma variance (first on ties),
// rounded means, color clamped to alpha; the r-pixel frame is copied.
@compute @workgroup_size(16, 16)
fn kuwahara(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    let r = p.i.x;
    let n = 16 + 2 * r;
    let bx = i32(wid.x * 16u) - r;
    let by = i32(wid.y * 16u) - r;
    for (var t = i32(lid.y * 16u + lid.x); t < n * n; t += 256) {
        tile[t] = u0[at0(z, bx + t % n, by + t / n, d)];
    }
    workgroupBarrier();
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let x = i32(id.x);
    let y = i32(id.y);
    if (x < r || y < r || x >= i32(d.x) - r || y >= i32(d.y) - r) {
        v0[i] = u0[i];
        return;
    }
    if (i32(d.x) <= 2 * r || i32(d.y) <= 2 * r) {
        v0[i] = u0[i];
        return;
    }
    let s = r + 1;
    let cnt = u32(s * s);
    var best_key = 0xffffffffu;
    var best = vec4<u32>(0u);
    for (var q = 0; q < 4; q++) {
        let qx = i32(lid.x) + select(0, r, (q & 1) == 1);
        let qy = i32(lid.y) + select(0, r, q >= 2);
        var sc = vec4<u32>(0u);
        var sl = 0u;
        var sl2 = 0u;
        for (var dy = 0; dy < s; dy++) {
            let row = (qy + dy) * n + qx;
            for (var dx = 0; dx < s; dx++) {
                let c = unpack4(tile[row + dx]);
                let l = (c.x * 77u + c.y * 150u + c.z * 29u) >> 8u;
                sc += c;
                sl += l;
                sl2 += l * l;
            }
        }
        let key = cnt * sl2 - sl * sl;
        if (q == 0 || key < best_key) {
            best_key = key;
            best = sc;
        }
    }
    let m = (best + vec4<u32>(cnt / 2u)) / vec4<u32>(cnt);
    v0[i] = pack4(vec4<u32>(min(m.xyz, vec3<u32>(m.w)), m.w));
}


// `brush::range_disc` (max - min over a disc, edge clamped) times p.f.x;
// ri = p.i.x, t0 = half widths per |dy|.
@compute @workgroup_size(16, 16)
fn range_disc(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let ri = p.i.x;
    var mx = -3.4e38;
    var mn = 3.4e38;
    for (var dy = -ri; dy <= ri; dy++) {
        let hx = i32(t0[abs(dy)]);
        for (var dx = -hx; dx <= hx; dx++) {
            let v = a0[at0(z, i32(id.x) + dx, i32(id.y) + dy, d)];
            mx = max(mx, v);
            mn = min(mn, v);
        }
    }
    b0[widx(z, id.x, id.y)] = fmul(fsub(mx, mn), p.f.x);
}

// `brush::dir_median`: median of n = p.i.x 8-bit samples along a 45°
// diagonal ("\" when p.i.y = 1, else "/"), ends of the diagonal replicated.
@compute @workgroup_size(16, 16)
fn dir_median(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let n = p.i.x;
    let r = n / 2;
    let x = i32(id.x);
    let y = i32(id.y);
    let W = i32(d.x);
    let H = i32(d.y);
    var sx: i32;
    var tmin: i32;
    var tmax: i32;
    if (p.i.y == 1) {
        sx = 1;
        tmin = -min(x, y);
        tmax = min(W - 1 - x, H - 1 - y);
    } else {
        sx = -1;
        tmin = -min(W - 1 - x, y);
        tmax = min(x, H - 1 - y);
    }
    let k = u32(r);
    var lo = 0u;
    for (var bit = 128u; bit > 0u; bit >>= 1u) {
        let cand = lo + bit;
        var below = 0u;
        for (var t = -r; t <= r; t++) {
            let tc = clamp(t, tmin, tmax);
            let v = q8(a0[widx(z, u32(x + sx * tc), u32(y + tc))]);
            below += select(0u, 1u, v < cand);
        }
        lo = select(lo, cand, below <= k);
    }
    b0[widx(z, id.x, id.y)] = tabf(t1[lo]);
}

// `gallery::warp`: bilinear self-warp of four planes (a0..a3) by (a4, a5).
@compute @workgroup_size(16, 16)
fn warp4(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let sx = fadd(f32(id.x), a4[i]);
    let sy = fadd(f32(id.y), a5[i]);
    let x0 = floor(sx);
    let y0 = floor(sy);
    let fx = fsub(sx, x0);
    let fy = fsub(sy, y0);
    let xi = i32(x0);
    let yi = i32(y0);
    let j00 = at0(z, xi, yi, d);
    let j10 = at0(z, xi + 1, yi, d);
    let j01 = at0(z, xi, yi + 1, d);
    let j11 = at0(z, xi + 1, yi + 1, d);
    b0[i] = bil(a0[j00], a0[j10], a0[j01], a0[j11], fx, fy);
    b1[i] = bil(a1[j00], a1[j10], a1[j01], a1[j11], fx, fy);
    b2[i] = bil(a2[j00], a2[j10], a2[j01], a2[j11], fx, fy);
    b3[i] = bil(a3[j00], a3[j10], a3[j01], a3[j11], fx, fy);
}

// a (1 - fx)(1 - fy) + b fx (1 - fy) + c (1 - fx) fy + d fx fy, in the CPU's order.
fn bil(a: f32, b: f32, c: f32, d: f32, fx: f32, fy: f32) -> f32 {
    let gx = fsub(1.0, fx);
    let gy = fsub(1.0, fy);
    let s0 = fmul(fmul(a, gx), gy);
    let s1 = fmul(fmul(b, fx), gy);
    let s2 = fmul(fmul(c, gx), fy);
    let s3 = fmul(fmul(d, fx), fy);
    return fadd(fadd(fadd(s0, s1), s2), s3);
}

// ----- sliding-histogram rank filter ----------------------------------------------------
//
// Thread per (column, channel) walking down p.j.y rows with a 256-bin
// histogram (u16 pairs) plus 16 coarse bins in workgroup memory: two column
// strips of the window change per row. Exact rank selection, so it equals
// any other median / quantile of the same samples.
// u0 = packed 8-bit samples; p.i = (ri, channels, mode), p.j = (rank k, rows
// per thread, write v / 255 floats (1) or byte values (0)); mode 0 = square
// of radius ri, 1 = disc with column half-heights t0[dx + ri]. Output b0..b3.
const SLIDE_T: u32 = 48u;
var<workgroup> hist: array<u32, 6528>;

fn h_add(base: u32, v: u32, d: u32) {
    hist[base + (v >> 1u)] += d << ((v & 1u) * 16u);
    hist[base + 128u + (v >> 5u)] += d << (((v >> 4u) & 1u) * 16u);
}

fn h_sub(base: u32, v: u32) {
    hist[base + (v >> 1u)] -= 1u << ((v & 1u) * 16u);
    hist[base + 128u + (v >> 5u)] -= 1u << (((v >> 4u) & 1u) * 16u);
}

// Smallest v with #(samples <= v) > k.
fn h_select(base: u32, k: u32) -> u32 {
    var acc = 0u;
    var bin = 15u;
    for (var cw = 0u; cw < 8u; cw++) {
        let wd = hist[base + 128u + cw];
        let c0 = wd & 0xffffu;
        if (acc + c0 > k) {
            bin = 2u * cw;
            break;
        }
        acc += c0;
        let c1 = wd >> 16u;
        if (acc + c1 > k) {
            bin = 2u * cw + 1u;
            break;
        }
        acc += c1;
    }
    var v = bin * 16u + 15u;
    for (var f = 0u; f < 8u; f++) {
        let wd = hist[base + bin * 8u + f];
        let c0 = wd & 0xffffu;
        if (acc + c0 > k) {
            v = bin * 16u + 2u * f;
            break;
        }
        acc += c0;
        let c1 = wd >> 16u;
        if (acc + c1 > k) {
            v = bin * 16u + 2u * f + 1u;
            break;
        }
        acc += c1;
    }
    return v;
}

fn s_byte(z: u32, x: i32, y: i32, d: vec2<u32>, sh: u32) -> u32 {
    return (u0[at0(z, x, y, d)] >> sh) & 255u;
}

fn s_half(dx: i32, ri: i32) -> i32 {
    if (p.i.z == 0) {
        return ri;
    }
    return i32(t0[dx + ri]);
}

@compute @workgroup_size(48)
fn rank_slide(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let z = wid.z;
    let d = dims(z);
    let nch = u32(p.i.y);
    let cpg = SLIDE_T / nch;
    let base = lid.x * 136u;
    for (var k = 0u; k < 136u; k++) {
        hist[base + k] = 0u;
    }
    let col = wid.x * cpg + lid.x / nch;
    let ch = lid.x % nch;
    if (lid.x >= cpg * nch || col >= d.x) {
        return;
    }
    let sh = ch * 8u;
    let ri = p.i.x;
    let x = i32(col);
    let y0 = i32(wid.y) * p.j.y;
    let y1 = min(y0 + p.j.y, i32(d.y));
    if (y0 >= y1) {
        return;
    }
    for (var dx = -ri; dx <= ri; dx++) {
        let vh = s_half(dx, ri);
        for (var dy = -vh; dy <= vh; dy++) {
            h_add(base, s_byte(z, x + dx, y0 + dy, d, sh), 1u);
        }
    }
    let k = u32(p.j.x);
    for (var y = y0; y < y1; y++) {
        if (y > y0) {
            for (var dx = -ri; dx <= ri; dx++) {
                let vh = s_half(dx, ri);
                h_sub(base, s_byte(z, x + dx, y - 1 - vh, d, sh));
                h_add(base, s_byte(z, x + dx, y + vh, d, sh), 1u);
            }
        }
        let v = h_select(base, k);
        let i = widx(z, col, u32(y));
        let o = select(f32(v), tabf(t1[v]), p.j.z == 1);
        switch ch {
            case 0u: {
                b0[i] = o;
            }
            case 1u: {
                b1[i] = o;
            }
            case 2u: {
                b2[i] = o;
            }
            default: {
                b3[i] = o;
            }
        }
    }
}

// Packs three planes' 8-bit quantization (v0).
@compute @workgroup_size(16, 16)
fn quant3(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    v0[i] = q8(a0[i]) | (q8(a1[i]) << 8u) | (q8(a2[i]) << 16u);
}

// `rank::median`'s output from per-channel byte planes (a0..a3): the r-pixel
// frame keeps u0, alpha only when the window isn't opaque (u1[z] != 0).
@compute @workgroup_size(16, 16)
fn median_pack(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let r = p.i.x;
    let x = i32(id.x);
    let y = i32(id.y);
    if (x < r || y < r || x >= i32(d.x) - r || y >= i32(d.y) - r) {
        v0[i] = u0[i];
        return;
    }
    let a = select(u0[i] >> 24u, u32(a3[i]), u1[z] != 0u);
    v0[i] = pack4(vec4<u32>(u32(a0[i]), u32(a1[i]), u32(a2[i]), a));
}

// ----- register rank filters (small windows; sizes are pipeline constants) -------

override MR: i32 = 2;
override DN: i32 = 15;

// Per channel of packed samples s[0..n]: the largest v with #(< v) <= k.
fn bisect4(n: i32, k: u32) -> vec4<u32> {
    var lo = vec4<u32>(0u);
    for (var bit = 128u; bit > 0u; bit >>= 1u) {
        let cand = lo + vec4<u32>(bit);
        var below = vec4<u32>(0u);
        for (var t = 0; t < n; t++) {
            below += select(vec4<u32>(0u), vec4<u32>(1u), unpack4(rs[t]) < cand);
        }
        lo = select(lo, cand, below <= vec4<u32>(k));
    }
    return lo;
}

var<private> rs: array<u32, 64>;

// Square median of radius MR of packed 8-bit planes (u0, from `quant3`),
// edge clamped; b0..b2 = v / 255.
@compute @workgroup_size(16, 16)
fn median_reg_q(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    let side = 2 * MR + 1;
    for (var t = 0; t < side * side; t++) {
        rs[t] = u0[at0(z, x + t % side - MR, y + t / side - MR, d)];
    }
    let m = bisect4(side * side, u32(side * side / 2));
    let i = widx(z, id.x, id.y);
    b0[i] = tabf(t1[m.x]);
    b1[i] = tabf(t1[m.y]);
    b2[i] = tabf(t1[m.z]);
}

// `rank::median` (radius MR) of a packed premultiplied plane: frame copied,
// alpha filtered only when the window isn't opaque (u1[z] != 0).
@compute @workgroup_size(16, 16)
fn median_reg_px(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let x = i32(id.x);
    let y = i32(id.y);
    if (x < MR || y < MR || x >= i32(d.x) - MR || y >= i32(d.y) - MR) {
        v0[i] = u0[i];
        return;
    }
    let side = 2 * MR + 1;
    for (var t = 0; t < side * side; t++) {
        rs[t] = u0[widx(z, u32(x + t % side - MR), u32(y + t / side - MR))];
    }
    let m = bisect4(side * side, u32(side * side / 2));
    let a = select(u0[i] >> 24u, m.w, u1[z] != 0u);
    v0[i] = pack4(vec4<u32>(m.xyz, a));
}

// `brush::dir_median` of DN samples on three channels at once (u0 packed
// 8-bit, from `quant3`); p.i.y = 1 for "\"; b0..b2 = v / 255.
@compute @workgroup_size(16, 16)
fn dir_median3(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let r = DN / 2;
    let x = i32(id.x);
    let y = i32(id.y);
    let W = i32(d.x);
    let H = i32(d.y);
    var sx: i32;
    var tmin: i32;
    var tmax: i32;
    if (p.i.y == 1) {
        sx = 1;
        tmin = -min(x, y);
        tmax = min(W - 1 - x, H - 1 - y);
    } else {
        sx = -1;
        tmin = -min(W - 1 - x, y);
        tmax = min(x, H - 1 - y);
    }
    for (var t = 0; t < DN; t++) {
        let tc = clamp(t - r, tmin, tmax);
        rs[t] = u0[widx(z, u32(x + sx * tc), u32(y + tc))];
    }
    let m = bisect4(DN, u32(r));
    let i = widx(z, id.x, id.y);
    b0[i] = tabf(t1[m.x]);
    b1[i] = tabf(t1[m.y]);
    b2[i] = tabf(t1[m.z]);
}

// ----- vec4-packed planes: up to four same-sigma channels per blur --------------

@compute @workgroup_size(64)
fn box_h4(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    let y = id.x;
    if (y >= d.y) {
        return;
    }
    let r = p.i.x;
    let inv = vec4<f32>(p.f.x);
    let base = wbase(z) + y * p.ww;
    let wm = i32(d.x) - 1;
    var s = vec4<f32>(0.0);
    for (var t = -r; t <= r; t++) {
        s = o4(s + q0[base + u32(clamp(t, 0, wm))]);
    }
    for (var x = 0; x <= wm; x++) {
        w0[base + u32(x)] = o4(s * inv);
        s = o4(s + o4(q0[base + u32(clamp(x + r + 1, 0, wm))] - q0[base + u32(clamp(x - r, 0, wm))]));
    }
}

@compute @workgroup_size(64)
fn box_v4(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    let x = id.x;
    if (x >= d.x) {
        return;
    }
    let r = p.i.x;
    let inv = vec4<f32>(p.f.x);
    let base = wbase(z) + x;
    let hm = i32(d.y) - 1;
    var c = vec4<f32>(0.0);
    for (var t = -r; t <= r; t++) {
        c = o4(c + q0[base + u32(clamp(t, 0, hm)) * p.ww]);
    }
    for (var y = 0; y <= hm; y++) {
        w0[base + u32(y) * p.ww] = o4(c * inv);
        c = o4(c + o4(q0[base + u32(clamp(y + r + 1, 0, hm)) * p.ww] - q0[base + u32(clamp(y - r, 0, hm)) * p.ww]));
    }
}

@compute @workgroup_size(16, 16)
fn copy4(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dims(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(z, id.x, id.y);
    w0[i] = q0[i];
}

// `Img::load` reduced to what the gray Sketch effects use: b0 = luminance
// (p.i.x = 0, weights p.f.xyz) or channel mean (p.i.x = 1, p.f.x = 1 / 3),
// b1 = alpha; same roundings as `load` then `lum` / `avg3`.
@compute @workgroup_size(16, 16)
fn load_gray(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let c = straight4(src[(w.sy + id.y) * p.srcw + w.sx + id.x]);
    let r = tabf(t1[c.x]);
    let g = tabf(t1[c.y]);
    let b = tabf(t1[c.z]);
    let i = widx(z, id.x, id.y);
    if (p.i.x == 0) {
        b0[i] = fadd(fadd(fmul(p.f.x, r), fmul(p.f.y, g)), fmul(p.f.z, b));
    } else {
        b0[i] = fmul(fadd(fadd(r, g), b), p.f.x);
    }
    b1[i] = tabf(t1[c.w]);
}
