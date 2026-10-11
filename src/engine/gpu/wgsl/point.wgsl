// Per-pixel adjustments. Each reproduces the CPU code bit for bit: tables
// (LUTs, the unpremultiply table, Threshold's per-(r, g) cut) are computed by
// the CPU with the CPU's own float math and only looked up here.

struct P {
    n: u32,
    ox: i32,
    oy: i32,
    w: u32,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
// Op table: 3 x 256 LUT bytes, the 33^3 LUT, or Threshold's cut table.
@group(0) @binding(3) var<storage, read> tab: array<u32>;
// unpremul[a * 256 + v] = round(v * 255 / a), as the CPU computes it.
@group(0) @binding(4) var<storage, read> unpremul: array<u32>;

fn tab_byte(i: u32) -> u32 {
    return (tab[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn unpremul_ch(v: u32, a: u32) -> u32 {
    let i = a * 256u + v;
    return (unpremul[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

// Straight color of a premultiplied pixel with 0 < a.
fn straight(c: vec4<u32>) -> vec3<u32> {
    if (c.w == 255u) {
        return c.xyz;
    }
    return vec3<u32>(unpremul_ch(c.x, c.w), unpremul_ch(c.y, c.w), unpremul_ch(c.z, c.w));
}

fn index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 256u;
}

// `apply_lut3`: per-channel LUTs on straight color.
@compute @workgroup_size(256)
fn lut3(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= p.n) {
        return;
    }
    let v = src[i];
    let c = unpack4(v);
    if (c.w == 0u) {
        dst[i] = v;
        return;
    }
    let s = straight(c);
    let o = vec3<u32>(tab_byte(s.x), tab_byte(256u + s.y), tab_byte(512u + s.z));
    dst[i] = premul(o, c.w);
}

fn lut3d_at(r: u32, g: u32, b: u32) -> vec3<u32> {
    let e = (r + g * 33u + b * 1089u) * 2u;
    let rg = tab[e];
    return vec3<u32>(rg & 65535u, rg >> 16u, tab[e + 1u] & 65535u);
}

fn split33(v: u32) -> vec2<u32> {
    let f = v * 32u * 256u / 255u;
    let i = min(f >> 8u, 31u);
    return vec2<u32>(i, f - (i << 8u));
}

// `Lut3d::eval`: tetrahedral interpolation in 8.8 fixed point.
fn lut3d_eval(c: vec3<u32>) -> vec3<u32> {
    let rr = split33(c.x);
    let gg = split33(c.y);
    let bb = split33(c.z);
    let r0 = rr.x;
    let g0 = gg.x;
    let b0 = bb.x;
    let fr = i32(rr.y);
    let fg = i32(gg.y);
    let fb = i32(bb.y);
    let c000 = vec3<i32>(lut3d_at(r0, g0, b0));
    let c111 = vec3<i32>(lut3d_at(r0 + 1u, g0 + 1u, b0 + 1u));
    var w: vec4<i32>;
    var p1: vec3<u32>;
    var p2: vec3<u32>;
    if (fr >= fg) {
        if (fg >= fb) {
            w = vec4<i32>(256 - fr, fr - fg, fg - fb, fb);
            p1 = lut3d_at(r0 + 1u, g0, b0);
            p2 = lut3d_at(r0 + 1u, g0 + 1u, b0);
        } else if (fr >= fb) {
            w = vec4<i32>(256 - fr, fr - fb, fb - fg, fg);
            p1 = lut3d_at(r0 + 1u, g0, b0);
            p2 = lut3d_at(r0 + 1u, g0, b0 + 1u);
        } else {
            w = vec4<i32>(256 - fb, fb - fr, fr - fg, fg);
            p1 = lut3d_at(r0, g0, b0 + 1u);
            p2 = lut3d_at(r0 + 1u, g0, b0 + 1u);
        }
    } else if (fb >= fg) {
        w = vec4<i32>(256 - fb, fb - fg, fg - fr, fr);
        p1 = lut3d_at(r0, g0, b0 + 1u);
        p2 = lut3d_at(r0, g0 + 1u, b0 + 1u);
    } else if (fb >= fr) {
        w = vec4<i32>(256 - fg, fg - fb, fb - fr, fr);
        p1 = lut3d_at(r0, g0 + 1u, b0);
        p2 = lut3d_at(r0, g0 + 1u, b0 + 1u);
    } else {
        w = vec4<i32>(256 - fg, fg - fr, fr - fb, fb);
        p1 = lut3d_at(r0, g0 + 1u, b0);
        p2 = lut3d_at(r0 + 1u, g0 + 1u, b0);
    }
    let v = c000 * w.x + vec3<i32>(p1) * w.y + vec3<i32>(p2) * w.z + c111 * w.w;
    let o = (((v + vec3<i32>(128)) >> vec3<u32>(8u)) * 255) / 65535;
    return vec3<u32>(clamp(o, vec3<i32>(0), vec3<i32>(255)));
}

@compute @workgroup_size(256)
fn lut3d(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= p.n) {
        return;
    }
    let v = src[i];
    let c = unpack4(v);
    if (c.w == 0u) {
        dst[i] = v;
        return;
    }
    dst[i] = premul(lut3d_eval(straight(c)), c.w);
}

// Threshold: white where b >= cut[r][g] (the CPU's float luma, tabulated).
@compute @workgroup_size(256)
fn threshold(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= p.n) {
        return;
    }
    let v = src[i];
    let c = unpack4(v);
    if (c.w == 0u) {
        dst[i] = v;
        return;
    }
    let s = straight(c);
    let k = s.x * 256u + s.y;
    let cut = (tab[k >> 1u] >> ((k & 1u) * 16u)) & 65535u;
    let o = select(0u, 255u, s.z >= cut);
    dst[i] = premul(vec3<u32>(o), c.w);
}

// Desaturate: integer Rec. 601 luma.
@compute @workgroup_size(256)
fn desaturate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= p.n) {
        return;
    }
    let v = src[i];
    let c = unpack4(v);
    if (c.w == 0u) {
        dst[i] = v;
        return;
    }
    let s = straight(c);
    let l = (s.x * 77u + s.y * 150u + s.z * 29u + 128u) >> 8u;
    dst[i] = premul(vec3<u32>(l), c.w);
}

// Average: one flat color (p.a).
@compute @workgroup_size(256)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= p.n) {
        return;
    }
    dst[i] = p.a;
}

// Add Noise: the CPU's hash, so the grain is the same; p.a = amount (f32),
// p.b = mono.
@compute @workgroup_size(256)
fn add_noise(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= p.n) {
        return;
    }
    let v = src[i];
    let c = unpack4(v);
    if (c.w == 0u) {
        dst[i] = v;
        return;
    }
    let gx = p.ox + i32(i % p.w);
    let gy = p.oy + i32(i / p.w);
    let amt = bitcast<f32>(p.a);
    let n0 = (hash01(gx, gy, 0u) + hash01(gx, gy, 100u) - 1.0) * amt;
    var n1 = n0;
    var n2 = n0;
    if (p.b == 0u) {
        n1 = (hash01(gx, gy, 1u) + hash01(gx, gy, 101u) - 1.0) * amt;
        n2 = (hash01(gx, gy, 2u) + hash01(gx, gy, 102u) - 1.0) * amt;
    }
    let a = f32(c.w);
    let f = a / 255.0;
    // Rust rounds half away from zero; WGSL's round() is half-to-even.
    let o = clamp(floor(vec3<f32>(c.xyz) + vec3<f32>(n0, n1, n2) * f + 0.5), vec3<f32>(0.0), vec3<f32>(a));
    dst[i] = pack4(vec4<u32>(vec3<u32>(o), c.w));
}
