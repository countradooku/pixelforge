// Blur Gallery ▸ Field Blur, Iris Blur, Tilt-Shift (`extra::variable_blur`):
// a stack of Gaussian levels of the padded band, each bit-exact with the
// CPU's `fast::gauss_plane` (direct 8.8 taps rounded per axis for sigma <=
// 2.6, else three box passes per axis, vertical first, clamped at the
// buffer edge), then a per-pixel interpolation between the two levels
// around the local sigma, plus the Blur Gallery effects (`bgfx`).

struct P {
    // Padded band size.
    w: u32,
    h: u32,
    // Box / direct pass: radius, 2^24 / (2r + 1), rows or columns per
    // thread, tap count.
    r: u32,
    inv: u32,
    k: u32,
    taps: u32,
    // Mix: apron, center size, level coordinate of src[0].
    pad: u32,
    cw: u32,
    ch: u32,
    ox: i32,
    oy: i32,
    // 0 Field, 1 Tilt-Shift, 2 Iris.
    kind: u32,
    // Largest sigma (level px), 2^level, effects on.
    mx: f32,
    scale: f32,
    fxon: u32,
    _0: u32,
    // Field: pins (x, y, blur) in a, b, c. Tilt: (cy, band, transition).
    // Iris: (cx, cy, radius).
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    fx: Fx,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> s1: array<u32>;
@group(0) @binding(4) var<storage, read> s2: array<u32>;
@group(0) @binding(5) var<storage, read> s3: array<u32>;
@group(0) @binding(6) var<storage, read> s4: array<u32>;
@group(0) @binding(7) var<storage, read> fxw: array<f32>;
// Direct taps (8.8 fixed point).
@group(0) @binding(8) var<storage, read> taps: array<u32>;

fn box_round(s: vec4<u32>) -> u32 {
    return pack4((s * p.inv + vec4<u32>(8388608u)) >> vec4<u32>(24u));
}

// One vertical box pass: thread (x, chunk of p.k rows).
@compute @workgroup_size(256, 1)
fn box_v(@builtin(global_invocation_id) gid: vec3<u32>) {
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
        s += unpack4(src[u32(clamp(y0 + j, 0, hm)) * p.w + x]);
    }
    for (var y = y0; y < y1; y++) {
        dst[u32(y) * p.w + x] = box_round(s);
        s += unpack4(src[u32(clamp(y + r + 1, 0, hm)) * p.w + x]);
        s -= unpack4(src[u32(clamp(y - r, 0, hm)) * p.w + x]);
    }
}

// One horizontal box pass: thread (chunk of p.k columns, y).
@compute @workgroup_size(1, 64)
fn box_h(@builtin(global_invocation_id) gid: vec3<u32>) {
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
        s += unpack4(src[row + u32(clamp(x0 + j, 0, wm))]);
    }
    for (var x = x0; x < x1; x++) {
        dst[row + u32(x)] = box_round(s);
        s += unpack4(src[row + u32(clamp(x + r + 1, 0, wm))]);
        s -= unpack4(src[row + u32(clamp(x - r, 0, wm))]);
    }
}

// Direct separable taps (radius <= 8), rounded after each axis, for a
// 16 x 16 output block; pixels closer than r to the buffer edge keep the
// source, as `gauss_plane` leaves them.
var<workgroup> hrow: array<vec4<u32>, 512>;

@compute @workgroup_size(16, 16)
fn direct(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = p.r;
    let bx = wid.x * 16u;
    let by = wid.y * 16u;
    let rows = 16u + 2u * r;
    for (var t = lid.y * 16u + lid.x; t < rows * 16u; t += 256u) {
        let ly = t / 16u;
        let lx = t % 16u;
        let x = bx + lx;
        let y = i32(by + ly) - i32(r);
        var acc = vec4<u32>(0u);
        if (x >= r && x + r < p.w && y >= 0 && y < i32(p.h)) {
            let base = u32(y) * p.w + x - r;
            for (var j = 0u; j < p.taps; j++) {
                acc += unpack4(src[base + j]) * taps[j];
            }
            acc = (acc + vec4<u32>(128u)) >> vec4<u32>(8u);
        }
        hrow[t] = acc;
    }
    workgroupBarrier();
    let x = bx + lid.x;
    let y = by + lid.y;
    if (x < r || x + r >= p.w || y < r || y + r >= p.h) {
        if (x < p.w && y < p.h) {
            dst[y * p.w + x] = src[y * p.w + x];
        }
        return;
    }
    var acc = vec4<u32>(0u);
    for (var j = 0u; j < p.taps; j++) {
        acc += hrow[(lid.y + j) * 16u + lid.x] * taps[j];
    }
    dst[y * p.w + x] = pack4((acc + vec4<u32>(128u)) >> vec4<u32>(8u));
}

// Stack level `k` at `i` (level 0 is the source).
fn st(k: u32, i: u32) -> vec4<f32> {
    switch k {
        case 0u: {
            return vec4<f32>(unpack4(src[i]));
        }
        case 1u: {
            return vec4<f32>(unpack4(s1[i]));
        }
        case 2u: {
            return vec4<f32>(unpack4(s2[i]));
        }
        case 3u: {
            return vec4<f32>(unpack4(s3[i]));
        }
        default: {
            return vec4<f32>(unpack4(s4[i]));
        }
    }
}

fn pin(q: vec4<f32>, gx: f32, gy: f32, acc: vec2<f32>) -> vec2<f32> {
    let dx = gx - q.x;
    let dy = gy - q.y;
    let d2 = dx * dx + dy * dy + 1.0;
    let wt = p_div(1.0, d2);
    return vec2<f32>(acc.x + wt * q.z, acc.y + wt);
}

// The filters' `sigma_at` closures (level-0 position -> level sigma).
fn sigma_at(gx: f32, gy: f32) -> f32 {
    switch p.kind {
        case 0u: {
            var nd = pin(p.a, gx, gy, vec2<f32>(0.0));
            nd = pin(p.b, gx, gy, nd);
            nd = pin(p.c, gx, gy, nd);
            return p_div(nd.x, nd.y) / p.scale;
        }
        case 1u: {
            let d = max(abs(gy - p.a.x) - p.a.y, 0.0);
            return p.mx * min(p_div(d, p.a.z), 1.0);
        }
        default: {
            let dx = gx - p.a.x;
            let dy = gy - p.a.y;
            let d = p_div(max(p_div(sqrt(dx * dx + dy * dy), p.a.z) - 0.6, 0.0), 0.6);
            return p.mx * min(d, 1.0);
        }
    }
}

@compute @workgroup_size(16, 16)
fn mix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    let bx = x + p.pad;
    let by = y + p.pad;
    let i = by * p.w + bx;
    let gx = (f32(p.ox + i32(bx)) + 0.5) * p.scale;
    let gy = (f32(p.oy + i32(by)) + 0.5) * p.scale;
    let t = clamp(p_div(sigma_at(gx, gy), p.mx), 0.0, 1.0) * 4.0;
    let k0 = min(u32(floor(t)), 3u);
    let f = t - f32(k0);
    let v = st(k0, i) * (1.0 - f) + st(k0 + 1u, i) * f;
    var o: u32;
    if (p.fxon == 0u) {
        let q = vec4<u32>(v + vec4<f32>(0.5));
        // `merge`: color never exceeds alpha.
        o = pack4(vec4<u32>(min(q.xyz, vec3<u32>(q.w)), q.w));
    } else {
        o = fx_apply(p.fx, v, src[i], t * 0.25, gx, gy);
    }
    dst[y * p.cw + x] = o;
}

// Copies the center of the padded band (no blur: max sigma < 0.4).
@compute @workgroup_size(16, 16)
fn crop(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    dst[y * p.cw + x] = src[(y + p.pad) * p.w + x + p.pad];
}

