// JPEG Artifacts Removal (`extra::jpeg_cleanup`): 8x8 deblocking (integer,
// exact) and the surface blur (`rank::surface`, a float bilateral evaluated
// directly: the two intensity levels bracketing each value come from a
// CPU-built table, as `pbfic` interpolates them).

struct P {
    w: u32,
    h: u32,
    r: u32,
    // Output: 0 = the whole buffer, else the center past this many pixels.
    out_pad: u32,
    ox: i32,
    oy: i32,
    k: u32,
    threshold: u32,
    inv_band: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> tab: array<u32>;

fn out_index(x: u32, y: u32) -> u32 {
    if (p.out_pad == 0u) {
        return y * p.w + x;
    }
    return (y - p.out_pad) * (p.w - 2u * p.out_pad) + x - p.out_pad;
}

// Whether (x, y) is written: anywhere, or in the center.
fn in_out(x: u32, y: u32) -> bool {
    if (x >= p.w || y >= p.h) {
        return false;
    }
    if (p.out_pad == 0u) {
        return true;
    }
    return x >= p.out_pad && y >= p.out_pad && x + p.out_pad < p.w && y + p.out_pad < p.h;
}

fn lum8(v: u32) -> i32 {
    let c = unpack4(v);
    return (i32(c.x) * 77 + i32(c.y) * 150 + i32(c.z) * 29) >> 8u;
}

// `jpeg_cleanup`'s deblocking at level 0: steps across the 8x8 grid in
// smooth areas move toward the average of the two sides.
@compute @workgroup_size(16, 16)
fn deblock(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.w || y >= p.h) {
        return;
    }
    let i = y * p.w + x;
    var v = src[i];
    if (x >= 1u && y >= 1u && x + 1u < p.w && y + 1u < p.h) {
        let gx = p.ox + i32(x);
        let gy = p.oy + i32(y);
        let mx = ((gx % 8) + 8) % 8;
        let my = ((gy % 8) + 8) % 8;
        let edge_x = mx == 0 || mx == 7;
        let edge_y = my == 0 || my == 7;
        if (edge_x || edge_y) {
            var a: u32;
            var b: u32;
            if (edge_x) {
                a = src[i - 1u];
                b = src[i + 1u];
            } else {
                a = src[i - p.w];
                b = src[i + p.w];
            }
            if (abs(lum8(a) - lum8(b)) < 24) {
                v = lerp4(src[i], lerp4(a, b, 128u), p.k);
            }
        }
    }
    dst[i] = v;
}

// Weighted mean of the window at intensity level `ck` (tent weights).
fn level_mean(win: ptr<function, array<vec4<f32>, 25>>, n: u32, ck: vec4<f32>) -> vec4<f32> {
    var sw = vec4<f32>(0.0);
    var si = vec4<f32>(0.0);
    for (var j = 0u; j < n; j++) {
        let v = (*win)[j];
        let wt = max(vec4<f32>(1.0) - abs(v - ck) * p.inv_band, vec4<f32>(0.0));
        sw += wt;
        si += wt * v;
    }
    return select(ck, si / sw, sw > vec4<f32>(1e-6));
}

// `rank::surface` (soft weights), radius p.r <= 2. tab[c * 3 ..]: the
// lower and upper level and the interpolation fraction for value c.
@compute @workgroup_size(16, 16)
fn surface(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (!in_out(x, y)) {
        return;
    }
    let i = y * p.w + x;
    let r = p.r;
    if (x < r || y < r || x + r >= p.w || y + r >= p.h) {
        dst[out_index(x, y)] = src[i];
        return;
    }
    var win: array<vec4<f32>, 25>;
    var n = 0u;
    for (var dy = 0u; dy <= 2u * r; dy++) {
        for (var dx = 0u; dx <= 2u * r; dx++) {
            win[n] = vec4<f32>(unpack4(src[(y + dy - r) * p.w + x + dx - r]));
            n++;
        }
    }
    let c = unpack4(src[i]);
    let lo = vec4<f32>(bitcast<f32>(tab[c.x * 3u]), bitcast<f32>(tab[c.y * 3u]), bitcast<f32>(tab[c.z * 3u]), bitcast<f32>(tab[c.w * 3u]));
    let hi = vec4<f32>(bitcast<f32>(tab[c.x * 3u + 1u]), bitcast<f32>(tab[c.y * 3u + 1u]), bitcast<f32>(tab[c.z * 3u + 1u]), bitcast<f32>(tab[c.w * 3u + 1u]));
    let f = vec4<f32>(bitcast<f32>(tab[c.x * 3u + 2u]), bitcast<f32>(tab[c.y * 3u + 2u]), bitcast<f32>(tab[c.z * 3u + 2u]), bitcast<f32>(tab[c.w * 3u + 2u]));
    let prev = level_mean(&win, n, lo);
    let cur = level_mean(&win, n, hi);
    let res = prev + (cur - prev) * f;
    let o = vec4<u32>(clamp(res + vec4<f32>(0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
    // `merge`: color never exceeds alpha.
    dst[out_index(x, y)] = pack4(vec4<u32>(min(o.xyz, vec3<u32>(o.w)), o.w));
}
