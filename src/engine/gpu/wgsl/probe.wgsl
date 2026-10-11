// Calibration kernels: a copy (dispatch overhead and bandwidth) and the same
// 5 x 5 box filter read from a packed u32 storage buffer and from an
// rgba8unorm texture, to pick the storage layout.

struct P {
    w: u32,
    h: u32,
    n: u32,
    _0: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var tex: texture_2d<f32>;

@compute @workgroup_size(256)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i < p.n) {
        dst[i] = src[i];
    }
}

@compute @workgroup_size(16, 16)
fn box5_buf(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    var s = vec4<u32>(0u);
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            s += unpack4(src[cidx(i32(gid.x) + dx, i32(gid.y) + dy, p.w, p.h)]);
        }
    }
    dst[gid.y * p.w + gid.x] = pack4((s + vec4<u32>(12u)) / 25u);
}

@compute @workgroup_size(16, 16)
fn box5_tex(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    var s = vec4<f32>(0.0);
    let m = vec2<i32>(i32(p.w) - 1, i32(p.h) - 1);
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            s += textureLoad(tex, clamp(vec2<i32>(gid.xy) + vec2<i32>(dx, dy), vec2<i32>(0), m), 0);
        }
    }
    let c = vec4<u32>(s * (255.0 / 25.0) + 0.5);
    dst[gid.y * p.w + gid.x] = pack4(c);
}
