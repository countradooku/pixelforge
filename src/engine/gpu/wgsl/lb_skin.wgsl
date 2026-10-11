// Skin Smoothing (`extra::skin_smoothing`): the surface blur's piecewise-
// linear bilateral filter (`rank::pbfic`: per intensity level, box sums of
// tent weights and weighted values, interpolated between the two levels
// bracketing each pixel), then a blend with the original by the skin-tone
// weight (a CPU-built table of every straight color). Box sums are f32
// running sums like the CPU's, over chunks instead of whole rows, so the
// result matches up to float rounding.

struct P {
    w: u32,
    h: u32,
    pad: u32,
    cw: u32,
    ch: u32,
    r: u32,
    level: u32,
    ck: f32,
    c_lo: f32,
    step: f32,
    inv_band: f32,
    zero: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
// Horizontal sums of weights / weighted values: padded rows x center columns.
@group(0) @binding(3) var<storage, read_write> hw: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> hi: array<vec4<f32>>;
// Previous level's result and the interpolated result (center).
@group(0) @binding(5) var<storage, read_write> prev: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> res: array<vec4<f32>>;
// skin[r | g << 8 | b << 16] (bytes); unpremul[a * 256 + v] (bytes).
@group(0) @binding(7) var<storage, read> skin: array<u32>;
@group(0) @binding(8) var<storage, read> unpremul: array<u32>;

fn opq4(v: vec4<f32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(v) ^ vec4<u32>(p.zero));
}

fn weight(v: vec4<f32>) -> vec4<f32> {
    return max(vec4<f32>(1.0) - abs(v - vec4<f32>(p.ck)) * p.inv_band, vec4<f32>(0.0));
}

// Thread (64 center columns, padded row).
@compute @workgroup_size(1, 64)
fn pb_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    let x0 = gid.x * 64u;
    if (y >= p.h || x0 >= p.cw) {
        return;
    }
    let x1 = min(x0 + 64u, p.cw);
    let r = i32(p.r);
    let row = y * p.w;
    var sw = vec4<f32>(0.0);
    var si = vec4<f32>(0.0);
    for (var j = -r; j <= r; j++) {
        let v = vec4<f32>(unpack4(src[row + u32(i32(x0 + p.pad) + j)]));
        let wt = weight(v);
        sw += wt;
        si += wt * v;
    }
    for (var x = x0; x < x1; x++) {
        hw[y * p.cw + x] = sw;
        hi[y * p.cw + x] = si;
        let c = i32(x + p.pad);
        let va = vec4<f32>(unpack4(src[row + u32(c + r + 1)]));
        let vs = vec4<f32>(unpack4(src[row + u32(c - r)]));
        let wa = weight(va);
        let ws = weight(vs);
        sw += opq4(wa - ws);
        si += opq4(wa * va - ws * vs);
    }
}

// Thread (center column, 16 rows): vertical sums, this level's mean, and
// the interpolation for pixels between the previous level and this one.
@compute @workgroup_size(256, 1)
fn pb_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y0 = gid.y * 16u;
    if (x >= p.cw || y0 >= p.ch) {
        return;
    }
    let y1 = min(y0 + 16u, p.ch);
    let r = i32(p.r);
    var vw = vec4<f32>(0.0);
    var vi = vec4<f32>(0.0);
    for (var j = -r; j <= r; j++) {
        let i = u32(i32(y0 + p.pad) + j) * p.cw + x;
        vw += hw[i];
        vi += hi[i];
    }
    for (var y = y0; y < y1; y++) {
        var cur: vec4<f32>;
        for (var k = 0; k < 4; k++) {
            cur[k] = select(p.ck, p_div(vi[k], vw[k]), vw[k] > 1e-6);
        }
        let i = y * p.cw + x;
        if (p.level > 0u) {
            let c = vec4<f32>(unpack4(src[(y + p.pad) * p.w + x + p.pad]));
            let pv = prev[i];
            var rv = res[i];
            for (var k = 0; k < 4; k++) {
                if (c[k] >= p.c_lo && c[k] <= p.ck) {
                    let f = p_div(c[k] - p.c_lo, p.step);
                    rv[k] = pv[k] + (cur[k] - pv[k]) * f;
                }
            }
            res[i] = rv;
        }
        prev[i] = cur;
        let c = i32(y + p.pad);
        let ia = u32(c + r + 1) * p.cw + x;
        let is = u32(c - r) * p.cw + x;
        vw += opq4(hw[ia] - hw[is]);
        vi += opq4(hi[ia] - hi[is]);
    }
}

fn unpremul_ch(v: u32, a: u32) -> u32 {
    let i = a * 256u + v;
    return (unpremul[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

// Rounds the result, clamps color to alpha and blends with the original by
// the skin weight.
@compute @workgroup_size(16, 16)
fn pb_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    let o = src[(y + p.pad) * p.w + x + p.pad];
    let v = vec4<u32>(clamp(res[y * p.cw + x] + vec4<f32>(0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
    let b = pack4(vec4<u32>(min(v.xyz, vec3<u32>(v.w)), v.w));
    let c = unpack4(o);
    var s = c.xyz;
    if (c.w != 255u) {
        s = vec3<u32>(unpremul_ch(c.x, c.w), unpremul_ch(c.y, c.w), unpremul_ch(c.z, c.w));
    }
    let si = s.x | (s.y << 8u) | (s.z << 16u);
    let t = (skin[si >> 2u] >> ((si & 3u) * 8u)) & 255u;
    dst[y * p.cw + x] = lerp4(o, b, t);
}
