// Select and Mask (`engine::refine`), bit-exact: every stage is integer
// math. Planes are one u32 per pixel; window sums are vec4<u32> with
// wrapping arithmetic (exact whenever the true value fits, signed or not).
//
// Edge Detection: prep_bin -> box R (C_R) [and box Rs (C_s, G_s)] ->
// prep_key -> box R2 (key sums) -> prep_gf -> box gf -> gf_ab -> box gf ->
// edge_out. Then median (Smooth), the `blur` module (Feather), lut
// (Contrast + Shift Edge), crop; Decontaminate: dc_prep, per radius box +
// dc_pick, dc_out.

struct P {
    w: u32,
    h: u32,
    r: u32,
    pad: u32,
    cw: u32,
    ch: u32,
    // Window sizes: guided filter / band (n), key or smart (n2).
    n: u32,
    n2: u32,
    eps: i32,
    smart: u32,
    amount: u32,
    k: u32,
}

const HARD_GRAD: u32 = 160u;
const FG_MIN: u32 = 250u;

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pa: array<u32>;
@group(0) @binding(2) var<storage, read> pb: array<u32>;
@group(0) @binding(3) var<storage, read> va: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> vb: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> vc: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read_write> out_p: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_v: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> tab: array<u32>;
@group(0) @binding(9) var<storage, read> pc: array<u32>;

fn tab_byte(i: u32) -> u32 {
    return (tab[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

// `px::unpremultiply` through the shared table (in `tab`).
fn straight(c: vec4<u32>) -> vec3<u32> {
    if (c.w == 255u) {
        return c.xyz;
    }
    if (c.w == 0u) {
        return vec3<u32>(0u);
    }
    return vec3<u32>(tab_byte(c.w * 256u + c.x), tab_byte(c.w * 256u + c.y), tab_byte(c.w * 256u + c.z));
}

fn at_g(x: i32, y: i32) -> i32 {
    return i32(pb[cidx(x, y, p.w, p.h)]);
}

// pa: image -> out_p: luminance guide.
@compute @workgroup_size(16, 16)
fn guide(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    let s = straight(unpack4(pa[i]));
    out_p[i] = (s.x * 77u + s.y * 150u + s.z * 29u) >> 8u;
}

// pa: mask, pb: guide -> out_v: (selected, |Sobel|, 0, 0).
@compute @workgroup_size(16, 16)
fn prep_bin(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let i = gid.y * p.w + gid.x;
    var sob = 0u;
    if (p.smart != 0u) {
        let gx = at_g(x + 1, y - 1) + 2 * at_g(x + 1, y) + at_g(x + 1, y + 1) - at_g(x - 1, y - 1) - 2 * at_g(x - 1, y) - at_g(x - 1, y + 1);
        let gy = at_g(x - 1, y + 1) + 2 * at_g(x, y + 1) + at_g(x + 1, y + 1) - at_g(x - 1, y - 1) - 2 * at_g(x, y - 1) - at_g(x + 1, y - 1);
        sob = u32(abs(gx) + abs(gy));
    }
    out_v[i] = vec4<u32>(select(0u, 1u, pa[i] >= 128u), sob, 0u, 0u);
}

// va: C_R sums, pb: guide -> out_v: (kf, kb, kf I, kb I). p.n = n_R.
@compute @workgroup_size(16, 16)
fn prep_key(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    let c = va[i].x;
    let kf = select(0u, 1u, c == p.n);
    let kb = select(0u, 1u, c == 0u);
    let g = pb[i];
    out_v[i] = vec4<u32>(kf, kb, kf * g, kb * g);
}

// `refine::alpha_est`; pa: mask, pb: guide, va: key sums -> out_v: (I, est, I^2, I est).
@compute @workgroup_size(16, 16)
fn prep_gf(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    let s = va[i];
    let g = pb[i];
    var est = pa[i];
    if (s.x != 0u && s.y != 0u) {
        let f = i32((s.z + s.x / 2u) / s.x);
        let b = i32((s.w + s.y / 2u) / s.y);
        let d = f - b;
        if (abs(d) >= 8) {
            est = u32(clamp((i32(g) - b) * 255 / d, 0, 255));
        }
    }
    out_v[i] = vec4<u32>(g, est, g * g, g * est);
}

// `refine::gf_ab`; va: (sI, sp, sII, sIp) -> out_v: (a, b) bits. p.n = n_gf.
@compute @workgroup_size(16, 16)
fn gf_ab(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    let s = va[i];
    let n = p.n;
    let cov = bitcast<i32>(n * s.w - s.x * s.y);
    let vr = bitcast<i32>(n * s.z - s.x * s.x);
    let a = clamp(((cov >> 12u) * 1024) / ((vr >> 12u) + p.eps), -16384, 16384);
    let b = (i32(s.y) * 1024 - a * i32(s.x)) / i32(n);
    out_v[i] = vec4<u32>(bitcast<u32>(a), bitcast<u32>(b), 0u, 0u);
}

fn band_w(c: u32, n: u32) -> u32 {
    return min(min(c, n - c) * 2040u / n, 255u);
}

// The guided filter inside the band; pa: mask, pb: guide, va: (Sa, Sb),
// vb: C_R sums, vc: (C_s, G_s) sums -> out_p. p.n = n_gf, p.k = n_R, p.n2 = n_s.
@compute @workgroup_size(16, 16)
fn edge_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    let n = i32(p.n);
    let s = va[i];
    let qa = (bitcast<i32>(s.x) * i32(pb[i])) / n;
    let qb = bitcast<i32>(s.y) / n;
    let q = u32(clamp((qa + qb + 512) >> 10u, 0, 255));
    var wt = band_w(vb[i].x, p.k);
    if (p.smart != 0u) {
        let c = vc[i];
        let hard = min((c.y / p.n2) * 255u / HARD_GRAD, 255u);
        wt = (band_w(c.x, p.n2) * hard + wt * (255u - hard) + 127u) / 255u;
    }
    let m = pa[i];
    let v = m * (255u - wt) + q * wt + 128u;
    out_p[i] = (v + (v >> 8u)) >> 8u;
}

// Window sums, vertical: thread (x, chunk of p.k rows), clamped rows.
@compute @workgroup_size(256, 1)
fn bsum_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    if (x >= p.w) {
        return;
    }
    let y0 = i32(gid.y * p.k);
    if (y0 >= i32(p.h)) {
        return;
    }
    let y1 = min(y0 + i32(p.k), i32(p.h));
    let r = i32(p.r);
    let hm = i32(p.h) - 1;
    var s = vec4<u32>(0u);
    for (var j = -r; j <= r; j++) {
        s += va[u32(clamp(y0 + j, 0, hm)) * p.w + x];
    }
    for (var y = y0; y < y1; y++) {
        out_v[u32(y) * p.w + x] = s;
        s += va[u32(clamp(y + r + 1, 0, hm)) * p.w + x];
        s -= va[u32(clamp(y - r, 0, hm)) * p.w + x];
    }
}

// Window sums, horizontal: thread (chunk of p.k columns, y), clamped columns.
@compute @workgroup_size(1, 64)
fn bsum_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    if (y >= p.h) {
        return;
    }
    let x0 = i32(gid.x * p.k);
    if (x0 >= i32(p.w)) {
        return;
    }
    let x1 = min(x0 + i32(p.k), i32(p.w));
    let r = i32(p.r);
    let wm = i32(p.w) - 1;
    let row = y * p.w;
    var s = vec4<u32>(0u);
    for (var j = -r; j <= r; j++) {
        s += va[row + u32(clamp(x0 + j, 0, wm))];
    }
    for (var x = x0; x < x1; x++) {
        out_v[row + u32(x)] = s;
        s += va[row + u32(clamp(x + r + 1, 0, wm))];
        s -= va[row + u32(clamp(x - r, 0, wm))];
    }
}

fn pick(i: u32, s: vec4<u32>) {
    if ((out_p[i] >> 24u) == 0u && s.w > 0u) {
        let n = s.w;
        let c = (s.xyz + vec3<u32>(n / 2u)) / n;
        out_p[i] = c.x | (c.y << 8u) | (c.z << 16u) | (255u << 24u);
    }
}

// `bsum_h` fused with `dc_pick`: the window sums go straight into the
// fg color plane (out_p) instead of memory.
@compute @workgroup_size(1, 64)
fn bsum_h_pick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    if (y >= p.h) {
        return;
    }
    let x0 = i32(gid.x * p.k);
    if (x0 >= i32(p.w)) {
        return;
    }
    let x1 = min(x0 + i32(p.k), i32(p.w));
    let r = i32(p.r);
    let wm = i32(p.w) - 1;
    let row = y * p.w;
    var s = vec4<u32>(0u);
    for (var j = -r; j <= r; j++) {
        s += va[row + u32(clamp(x0 + j, 0, wm))];
    }
    for (var x = x0; x < x1; x++) {
        pick(row + u32(x), s);
        s += va[row + u32(clamp(x + r + 1, 0, wm))];
        s -= va[row + u32(clamp(x - r, 0, wm))];
    }
}

// Exact (2r+1)^2 median (r <= 5); pixels within r of the edge copied.
@compute @workgroup_size(16, 16)
fn median(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.w || y >= p.h) {
        return;
    }
    let i = y * p.w + x;
    let r = p.r;
    if (x < r || y < r || x + r >= p.w || y + r >= p.h) {
        out_p[i] = pa[i];
        return;
    }
    var win: array<u32, 121>;
    var n = 0u;
    for (var dy = 0u; dy <= 2u * r; dy++) {
        let row = (y + dy - r) * p.w + x - r;
        for (var dx = 0u; dx <= 2u * r; dx++) {
            win[n] = pa[row + dx];
            n++;
        }
    }
    let k = n / 2u;
    var m = 0u;
    for (var b = 8u; b > 0u; b--) {
        let t = m | (1u << (b - 1u));
        var cnt = 0u;
        for (var j = 0u; j < n; j++) {
            cnt += select(0u, 1u, win[j] < t);
        }
        if (cnt <= k) {
            m = t;
        }
    }
    out_p[i] = m;
}

// Contrast + Shift Edge table (bytes in `tab`), lane 0 only.
@compute @workgroup_size(16, 16)
fn lut(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    out_p[i] = tab_byte(pa[i] & 255u);
}

// The center of a padded plane.
@compute @workgroup_size(16, 16)
fn crop(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    out_p[gid.y * p.cw + gid.x] = pa[(gid.y + p.pad) * p.w + gid.x + p.pad] & 255u;
}

// pa: image, pb: mask -> out_v: straight color of confident pixels and
// their count; out_p (the fg color plane) cleared.
@compute @workgroup_size(16, 16)
fn dc_prep(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    let f = select(0u, 1u, (pb[i] & 255u) >= FG_MIN);
    out_v[i] = vec4<u32>(straight(unpack4(pa[i])) * f, f);
    out_p[i] = 0u;
}

// va: window sums at this radius -> out_p: fg color where still unresolved.
@compute @workgroup_size(16, 16)
fn dc_pick(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    pick(i, va[i]);
}

// `refine::decon_px` over the center; pa: image, pb: mask, pc: fg colors,
// tab: unpremultiply.
@compute @workgroup_size(16, 16)
fn dc_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let i = (gid.y + p.pad) * p.w + gid.x + p.pad;
    let px = pa[i];
    let m = pb[i] & 255u;
    let col = pc[i];
    let c = unpack4(px);
    var o = px;
    if (m != 0u && m < 255u && c.w != 0u && (col >> 24u) != 0u) {
        let t = (p.amount * min(4u * (255u - m), 255u) + 127u) / 255u;
        let s = straight(c);
        let mixed = unpack4(lerp4(pack4(vec4<u32>(s, 0u)), col & 0xffffffu, t));
        o = premul(mixed.xyz, c.w);
    }
    out_p[gid.y * p.cw + gid.x] = o;
}
