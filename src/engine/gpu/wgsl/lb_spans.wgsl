// Lens Blur, Shape Blur and Smart Sharpen's disk (Lens Blur removal): sums
// of iris / shape rows from per-row prefix sums, like `blur::lens_ex`,
// `extra::shape_blur` and `blur::disk_plane`.
//
// Prefix sums run left to right over segments that mirror the CPU's padded
// blocks (`p.seg` center columns plus the apron), so Lens Blur's f32 sums
// add the same values in the same order as the CPU and its pixels come out
// identical. Shape Blur and the disk use integer sums (exact in any order).
// The prefix planes hold one channel each: `pre0..pre3[row * pitch +
// segment * (seg + 2 pad + 1) + i]` = sum of the segment's first `i` pixels.

struct P {
    w: u32,
    h: u32,
    pad: u32,
    cw: u32,
    ch: u32,
    seg: u32,
    pitch: u32,
    nseg: u32,
    // 0 lens, 1 shape (float rounding), 2 disk (integer rounding, raw output), 3 copy
    mode: u32,
    uniform: u32,
    // 0 none, 1 alpha, 2 mask (`mask`, one byte per padded pixel)
    depth: u32,
    // bit 0 on, bit 1 gaussian, bit 2 mono
    noise: u32,
    amount: f32,
    ox: i32,
    oy: i32,
    countf: f32,
    count: u32,
    nspans: u32,
    // Runtime zero: keeps the compiler from fusing a multiply into an add.
    zero: u32,
    // Lens table offsets (words into `tab`).
    off_rad: u32,
    off_off: u32,
    off_cnt: u32,
    off_spans: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read_write> pre0: array<u32>;
@group(0) @binding(4) var<storage, read_write> pre1: array<u32>;
@group(0) @binding(5) var<storage, read_write> pre2: array<u32>;
@group(0) @binding(6) var<storage, read_write> pre3: array<u32>;
// Lens: to_lin (256 f32), alpha (256 f32), from_lin (4096), radius per
// depth (256), off (max + 2), count (max + 1, f32), spans (4 words each).
// Shape / disk: spans from `off_spans`.
@group(0) @binding(7) var<storage, read> tab: array<u32>;
@group(0) @binding(8) var<storage, read> mask: array<u32>;

fn nofma(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ p.zero);
}

// Hides a value from the optimizer so `acc + (hi - lo)` isn't reassociated.
fn opq4(v: vec4<f32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(v) ^ vec4<u32>(p.zero));
}

fn tabf(i: u32) -> f32 {
    return bitcast<f32>(tab[i]);
}

fn span_at(k: u32) -> vec3<i32> {
    let i = p.off_spans + k * 4u;
    return vec3<i32>(bitcast<i32>(tab[i]), bitcast<i32>(tab[i + 1u]), bitcast<i32>(tab[i + 2u]));
}

fn seg_stride() -> u32 {
    return p.seg + 2u * p.pad + 1u;
}

// Padded columns of segment `sg`.
fn seg_len(sg: u32) -> u32 {
    return min(p.seg, p.cw - sg * p.seg) + 2u * p.pad;
}

// Thread (row, segment): f32 prefix sums of to_lin (color) and v / 255 (alpha).
@compute @workgroup_size(64)
fn prefix_f(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let sg = gid.y;
    if (row >= p.h || sg >= p.nseg) {
        return;
    }
    let len = seg_len(sg);
    let base = row * p.pitch + sg * seg_stride();
    let sbase = row * p.w + sg * p.seg;
    var acc = vec4<f32>(0.0);
    pre0[base] = 0u;
    pre1[base] = 0u;
    pre2[base] = 0u;
    pre3[base] = 0u;
    for (var i = 0u; i < len; i++) {
        let c = unpack4(src[sbase + i]);
        acc += vec4<f32>(tabf(c.x), tabf(c.y), tabf(c.z), tabf(256u + c.w));
        let o = base + i + 1u;
        pre0[o] = bitcast<u32>(acc.x);
        pre1[o] = bitcast<u32>(acc.y);
        pre2[o] = bitcast<u32>(acc.z);
        pre3[o] = bitcast<u32>(acc.w);
    }
}

// Thread (row, segment): integer prefix sums.
@compute @workgroup_size(64)
fn prefix_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let sg = gid.y;
    if (row >= p.h || sg >= p.nseg) {
        return;
    }
    let len = seg_len(sg);
    let base = row * p.pitch + sg * seg_stride();
    let sbase = row * p.w + sg * p.seg;
    var acc = vec4<u32>(0u);
    pre0[base] = 0u;
    pre1[base] = 0u;
    pre2[base] = 0u;
    pre3[base] = 0u;
    for (var i = 0u; i < len; i++) {
        acc += unpack4(src[sbase + i]);
        let o = base + i + 1u;
        pre0[o] = acc.x;
        pre1[o] = acc.y;
        pre2[o] = acc.z;
        pre3[o] = acc.w;
    }
}

fn pref(i: u32) -> vec4<f32> {
    return vec4<f32>(bitcast<f32>(pre0[i]), bitcast<f32>(pre1[i]), bitcast<f32>(pre2[i]), bitcast<f32>(pre3[i]));
}

fn prefu(i: u32) -> vec4<u32> {
    return vec4<u32>(pre0[i], pre1[i], pre2[i], pre3[i]);
}

fn depth_at(pi: u32, o: u32) -> u32 {
    if (p.depth == 1u) {
        return o >> 24u;
    }
    return (mask[pi >> 2u] >> ((pi & 3u) * 8u)) & 255u;
}

// Rust's f32::round (half away from zero).
fn round_away(v: f32) -> f32 {
    return sign(v) * floor(abs(v) + 0.5);
}

fn lens_noise(gx: i32, gy: i32, c: u32) -> f32 {
    if ((p.noise & 2u) != 0u) {
        return (hash01(gx, gy, c + 300u) + hash01(gx, gy, c + 400u) + hash01(gx, gy, c + 500u) - 1.5) * (p.amount * 1.1547005);
    }
    return (hash01(gx, gy, c + 300u) * 2.0 - 1.0) * p.amount;
}

// `blur::add_lens_noise` for one pixel.
fn add_noise(px: u32, gx: i32, gy: i32) -> u32 {
    let c = unpack4(px);
    if (c.w == 0u) {
        return px;
    }
    let n0 = lens_noise(gx, gy, 0u);
    var n = vec3<f32>(n0);
    if ((p.noise & 4u) == 0u) {
        n = vec3<f32>(n0, lens_noise(gx, gy, 1u), lens_noise(gx, gy, 2u));
    }
    let a = f32(c.w);
    let f = a / 255.0;
    let v = vec3<f32>(c.xyz) + n * f;
    let r = clamp(vec3<f32>(round_away(v.x), round_away(v.y), round_away(v.z)), vec3<f32>(0.0), vec3<f32>(a));
    return pack4(vec4<u32>(vec3<u32>(r), c.w));
}

// Thread per center pixel: Lens Blur.
@compute @workgroup_size(16, 16)
fn lens_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    let px = x + p.pad;
    let py = y + p.pad;
    let pi = py * p.w + px;
    let o = src[pi];
    var rho = tab[p.off_rad];
    if (p.uniform == 0u) {
        rho = tab[p.off_rad + depth_at(pi, o)];
    }
    var out = o;
    if (rho > 0u) {
        let sg = x / p.seg;
        let lx = i32(px - sg * p.seg);
        let sbase = sg * seg_stride();
        let slen = i32(seg_len(sg));
        var acc = vec4<f32>(0.0);
        let k1 = tab[p.off_off + rho + 1u];
        for (var k = tab[p.off_off + rho]; k < k1; k++) {
            let s = span_at(k);
            let yy = u32(clamp(i32(py) + s.x, 0, i32(p.h) - 1));
            let rb = yy * p.pitch + sbase;
            let xa = lx + s.y;
            let xb = lx + s.z + 1;
            if (xa >= 0 && xb <= slen) {
                acc += opq4(pref(rb + u32(xb)) - pref(rb + u32(xa)));
            } else {
                // Buffer borders (apron pixels only): scale the clamped span.
                let x0 = clamp(xa, 0, slen);
                let x1 = clamp(xb, 0, slen);
                if (x1 > x0) {
                    let d = (pref(rb + u32(x1)) - pref(rb + u32(x0))) * f32(s.z - s.y + 1);
                    let g = f32(x1 - x0);
                    acc += vec4<f32>(p_div(d.x, g), p_div(d.y, g), p_div(d.z, g), p_div(d.w, g));
                }
            }
        }
        let cnt = tabf(p.off_cnt + rho);
        let m = clamp(vec4<f32>(p_div(acc.x, cnt), p_div(acc.y, cnt), p_div(acc.z, cnt), p_div(acc.w, cnt)), vec4<f32>(0.0), vec4<f32>(1.0));
        let ci = vec3<u32>(m.xyz * 4095.0);
        let col = vec3<u32>(tab[512u + ci.x], tab[512u + ci.y], tab[512u + ci.z]);
        let a = u32(nofma(m.w * 255.0) + 0.5);
        out = pack4(vec4<u32>(min(col, vec3<u32>(a)), a));
    }
    if ((p.noise & 1u) != 0u) {
        out = add_noise(out, p.ox + i32(px), p.oy + i32(py));
    }
    dst[y * p.cw + x] = out;
}

// Thread per center pixel: Shape Blur (mode 1, merged) or the disk average
// (mode 2, raw blur for Smart Sharpen).
@compute @workgroup_size(16, 16)
fn shape_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    let px = x + p.pad;
    let py = y + p.pad;
    let sg = x / p.seg;
    let lx = i32(px - sg * p.seg);
    let sbase = sg * seg_stride();
    let slen = i32(seg_len(sg));
    var acc = vec4<u32>(0u);
    for (var k = 0u; k < p.nspans; k++) {
        let s = span_at(k);
        let yy = u32(clamp(i32(py) + s.x, 0, i32(p.h) - 1));
        let rb = yy * p.pitch + sbase;
        let xa = lx + s.y;
        let xb = lx + s.z + 1;
        if (xa >= 0 && xb <= slen) {
            acc += prefu(rb + u32(xb)) - prefu(rb + u32(xa));
        } else {
            let x0 = clamp(xa, 0, slen);
            let x1 = clamp(xb, 0, slen);
            if (x1 > x0) {
                acc += (prefu(rb + u32(x1)) - prefu(rb + u32(x0))) * u32(s.z - s.y + 1) / u32(x1 - x0);
            }
        }
    }
    if (p.mode == 2u) {
        dst[y * p.cw + x] = pack4((2u * acc + vec4<u32>(p.count)) / vec4<u32>(2u * p.count));
        return;
    }
    let fa = vec4<f32>(acc);
    let q = vec4<f32>(p_div(fa.x, p.countf), p_div(fa.y, p.countf), p_div(fa.z, p.countf), p_div(fa.w, p.countf));
    let v = min(vec4<u32>(q + vec4<f32>(0.5)), vec4<u32>(255u));
    dst[y * p.cw + x] = pack4(vec4<u32>(min(v.xyz, vec3<u32>(v.w)), v.w));
}

// Copies the center (Shape Blur with a kernel of one pixel).
@compute @workgroup_size(16, 16)
fn copy_center(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    dst[y * p.cw + x] = src[(y + p.pad) * p.w + x + p.pad];
}
