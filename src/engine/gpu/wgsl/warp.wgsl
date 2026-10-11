// Distort ▸ warps (`distort::Warp` + `render_warp`): each output pixel maps
// to a level-0 source position and is sampled bilinearly from the whole
// source level (edge repeat, wrap for Offset, transparent outside for
// transforms), with the CPU sampler's 8-bit fractions and integer lerps.

struct P {
    // Output region: origin (level coords) and size.
    ox: i32,
    oy: i32,
    w: u32,
    h: u32,
    // Source level: stride (tile-area width), image size.
    stride: u32,
    iw: u32,
    ih: u32,
    level: u32,
    kind: u32,
    wrap: u32,
    clip: u32,
    _0: u32,
    // Warp parameters (meaning per kind, see `kernels::warp`).
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

fn texel(x: i32, y: i32) -> u32 {
    var xx = x;
    var yy = y;
    let gw = i32(p.iw);
    let gh = i32(p.ih);
    if (p.wrap != 0u) {
        xx = ((x % gw) + gw) % gw;
        yy = ((y % gh) + gh) % gh;
    } else {
        xx = clamp(x, 0, gw - 1);
        yy = clamp(y, 0, gh - 1);
    }
    return src[u32(yy) * p.stride + u32(xx)];
}

// `Sampler::sample`.
fn bilinear(xs: f32, ys: f32) -> u32 {
    let x = xs - 0.5;
    let y = ys - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = u32((x - x0) * 255.0);
    let fy = u32((y - y0) * 255.0);
    let xi = i32(x0);
    let yi = i32(y0);
    let top = lerp4(texel(xi, yi), texel(xi + 1, yi), fx);
    let bot = lerp4(texel(xi, yi + 1), texel(xi + 1, yi + 1), fx);
    return lerp4(top, bot, fy);
}

fn rem_euclid(a: f32, b: f32) -> f32 {
    let r = a - b * trunc(a / b);
    return select(r, r + abs(b), r < 0.0);
}

// `Warp::map` (level-0 coordinates).
fn map(x: f32, y: f32) -> vec2<f32> {
    let a = p.a;
    let b = p.b;
    switch p.kind {
        // Twirl { angle, cx, cy, radius }
        case 0u: {
            let d = vec2<f32>(x - a.y, y - a.z);
            let r = sqrt(d.x * d.x + d.y * d.y);
            if (r >= a.w) {
                return vec2<f32>(x, y);
            }
            let t = 1.0 - r / a.w;
            let ang = a.x * (PI / 180.0) * t * t;
            let sc = p_sincos(ang);
            let s = sc.x;
            let c = sc.y;
            return vec2<f32>(a.y + d.x * c - d.y * s, a.z + d.x * s + d.y * c);
        }
        // Pinch / Spherize { amount, cx, cy, radius }
        case 1u, 2u: {
            let d = vec2<f32>(x - a.y, y - a.z);
            let r = sqrt(d.x * d.x + d.y * d.y);
            if (r >= a.w || r == 0.0) {
                return vec2<f32>(x, y);
            }
            let rn = r / a.w;
            var k: f32;
            if (p.kind == 1u) {
                k = p_pow(rn, -a.x / 100.0 * 0.5);
            } else {
                let am = a.x / 100.0;
                let sph = (2.0 / PI) * p_asin(rn) / max(rn, 1e-6);
                k = 1.0 + (sph - 1.0) * am;
            }
            return vec2<f32>(a.y + d.x * k, a.z + d.y * k);
        }
        // ZigZag { amount, ridges, cx, cy } + b.x = radius
        case 3u: {
            let d = vec2<f32>(x - a.z, y - a.w);
            let r = sqrt(d.x * d.x + d.y * d.y);
            let radius = b.x;
            if (r >= radius || r == 0.0) {
                return vec2<f32>(x, y);
            }
            let rn = r / radius;
            let dd = (a.x / 100.0) * radius * 0.1 * p_sin(rn * a.y * TAU) * (1.0 - rn);
            let k = (r + dd) / r;
            return vec2<f32>(a.z + d.x * k, a.w + d.y * k);
        }
        // Polar { to_polar (a.x), w, h }
        case 4u: {
            let w = a.y;
            let h = a.z;
            let c = vec2<f32>(w * 0.5, h * 0.5);
            if (a.x != 0.0) {
                let d = vec2<f32>(x - c.x, y - c.y);
                let ang = p_atan2(d.y, d.x) + PI * 0.5;
                let r = sqrt(d.x * d.x + d.y * d.y) / (min(w, h) * 0.5);
                return vec2<f32>(rem_euclid(ang, TAU) / TAU * w, r * h);
            }
            let ang = p_div(x, w) * TAU - PI * 0.5;
            let r = p_div(y, h) * min(w, h) * 0.5;
            return vec2<f32>(c.x + r * p_cos(ang), c.y + r * p_sin(ang));
        }
        // Ripple { amount, size }
        case 5u: {
            let am = a.x / 100.0 * a.y * 0.25;
            return vec2<f32>(x + am * p_sin(y / a.y * TAU), y + am * p_sin(x / a.y * TAU));
        }
        // Wave { wavelength, amplitude }
        case 6u: {
            let f = TAU / max(a.x, 1.0);
            let am = a.y;
            return vec2<f32>(
                x + am * p_sin(y * f) + am * 0.5 * p_sin(y * f * 2.3 + 1.7),
                y + am * p_sin(x * f) + am * 0.5 * p_sin(x * f * 1.9 + 0.4),
            );
        }
        // Offset { dx, dy, w, h }
        case 7u: {
            return vec2<f32>(rem_euclid(x - a.x, a.z), rem_euclid(y - a.y, a.w));
        }
        // Shear { amount, h }
        case 8u: {
            let t = y / a.y;
            return vec2<f32>(x - a.x * p_sin(t * PI), y);
        }
        // Homography: a = m0..m3, b = m4..m7, c.x = m8
        case 9u: {
            let wq = b.z * x + b.w * y + p.c.x;
            if (abs(wq) < 1e-9) {
                return vec2<f32>(-1e30, -1e30);
            }
            return vec2<f32>((a.x * x + a.y * y + a.z) / wq, (a.w * x + b.x * y + b.y) / wq);
        }
        // WideAngle { strength, scale, cx, cy } + b.x = diag
        default: {
            let d = vec2<f32>(x - a.z, y - a.w);
            let r = sqrt(d.x * d.x + d.y * d.y);
            if (r < 1e-3) {
                return vec2<f32>(x, y);
            }
            let f = b.x / max(1.0 + a.x / 25.0, 0.05);
            let rn = r / (a.y / 100.0);
            let s = f * p_atan(rn / f);
            let k = 1.0 + (s / r - 1.0) * clamp(a.x / 100.0, 0.0, 1.0);
            return vec2<f32>(a.z + d.x * k, a.w + d.y * k);
        }
    }
}

@compute @workgroup_size(16, 16)
fn warp(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let s = f32(1u << p.level);
    let gx = (f32(p.ox + i32(gid.x)) + 0.5) * s;
    let gy = (f32(p.oy + i32(gid.y)) + 0.5) * s;
    let m = map(gx, gy);
    let l = m / s;
    var v = 0u;
    if (p.clip != 0u && !(l.x >= 0.0 && l.y >= 0.0 && l.x < f32(p.iw) && l.y < f32(p.ih))) {
        v = 0u;
    } else {
        v = bilinear(l.x, l.y);
    }
    dst[gid.y * p.w + gid.x] = v;
}

// ----- Edit ▸ Transform ▸ Warp (`Warp::Mesh`, `warp::invert`) ----------------
// The CPU's float sequence, every product and sum rounded on its own (`nf`)
// so fast math can't fuse or reassociate it.

// Control points (x, y) x 16 row-major, the box x0, y0, w, h, then the
// 5 x 5 start nodes (x, y) computed on the CPU (`warp::start_nodes`).
@group(0) @binding(3) var<storage, read> mesh: array<f32>;

fn nf(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | p._0);
}

fn bern(t: f32) -> vec4<f32> {
    let s = 1.0 - t;
    return vec4<f32>(nf(nf(s * s) * s), nf(nf(nf(3.0 * t) * s) * s), nf(nf(nf(3.0 * t) * t) * s), nf(nf(t * t) * t));
}

fn bern_d(t: f32) -> vec4<f32> {
    let s = 1.0 - t;
    return vec4<f32>(
        nf(nf(-3.0 * s) * s),
        nf(nf(3.0 * s) * s) - nf(nf(6.0 * t) * s),
        nf(nf(6.0 * t) * s) - nf(nf(3.0 * t) * t),
        nf(nf(3.0 * t) * t),
    );
}

struct Patch {
    p: vec2<f32>,
    du: vec2<f32>,
    dv: vec2<f32>,
}

fn patch_at(u: f32, v: f32) -> Patch {
    let bu = bern(u);
    let bv = bern(v);
    let du = bern_d(u);
    let dv = bern_d(v);
    var o = Patch(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
    for (var j = 0u; j < 4u; j++) {
        for (var i = 0u; i < 4u; i++) {
            let q = vec2<f32>(mesh[(j * 4u + i) * 2u], mesh[(j * 4u + i) * 2u + 1u]);
            let a = nf(bu[i] * bv[j]);
            let b = nf(du[i] * bv[j]);
            let c = nf(bu[i] * dv[j]);
            o.p = vec2<f32>(nf(o.p.x + nf(a * q.x)), nf(o.p.y + nf(a * q.y)));
            o.du = vec2<f32>(nf(o.du.x + nf(b * q.x)), nf(o.du.y + nf(b * q.y)));
            o.dv = vec2<f32>(nf(o.dv.x + nf(c * q.x)), nf(o.dv.y + nf(c * q.y)));
        }
    }
    return o;
}

const NEWTON_STEPS: u32 = 10u;

// x / y rounded like IEEE division (two residual corrections of the fast
// quotient).
fn div_ieee(x: f32, y: f32) -> f32 {
    let q0 = x / y;
    let q1 = q0 + fma(-q0, y, x) / y;
    let r = fma(-q1, y, x);
    // q1 is within an ulp: step to the neighbour if it is closer.
    let up = bitcast<f32>(bitcast<u32>(q1) + 1u);
    let dn = bitcast<f32>(bitcast<u32>(q1) - 1u);
    let ru = fma(-up, y, x);
    let rd = fma(-dn, y, x);
    var q = q1;
    var best = abs(r);
    if (abs(ru) < best) {
        q = up;
        best = abs(ru);
    }
    if (abs(rd) < best) {
        q = dn;
    }
    return q;
}

fn patch_p(u: f32, v: f32) -> vec2<f32> {
    let bu = bern(u);
    let bv = bern(v);
    var o = vec2<f32>(0.0);
    for (var j = 0u; j < 4u; j++) {
        for (var i = 0u; i < 4u; i++) {
            let a = nf(bu[i] * bv[j]);
            o = vec2<f32>(nf(o.x + nf(a * mesh[(j * 4u + i) * 2u])), nf(o.y + nf(a * mesh[(j * 4u + i) * 2u + 1u])));
        }
    }
    return o;
}

fn mesh_map(x: f32, y: f32) -> vec2<f32> {
    let x0 = mesh[32];
    let y0 = mesh[33];
    let tx = x - x0;
    let ty = y - y0;
    var u = 0.0;
    var v = 0.0;
    var best = 3.40282347e38;
    for (var j = 0u; j < 5u; j++) {
        for (var i = 0u; i < 5u; i++) {
            let dx = mesh[36u + (j * 5u + i) * 2u] - tx;
            let dy = mesh[37u + (j * 5u + i) * 2u] - ty;
            let d = nf(nf(dx * dx) + nf(dy * dy));
            if (d < best) {
                u = f32(i) * 0.25;
                v = f32(j) * 0.25;
                best = d;
            }
        }
    }
    for (var k = 0u; k < NEWTON_STEPS; k++) {
        let q = patch_at(u, v);
        let rx = q.p.x - tx;
        let ry = q.p.y - ty;
        let det = nf(q.du.x * q.dv.y) - nf(q.du.y * q.dv.x);
        if (abs(det) < 1e-6) {
            return vec2<f32>(-1e30, -1e30);
        }
        let nu = clamp(u - div_ieee(nf(rx * q.dv.y) - nf(ry * q.dv.x), det), -0.5, 1.5);
        let nv = clamp(v - div_ieee(nf(q.du.x * ry) - nf(q.du.y * rx), det), -0.5, 1.5);
        let done = nu == u && nv == v;
        u = nu;
        v = nv;
        if (done) {
            break;
        }
    }
    let q = patch_p(u, v);
    let ok = abs(q.x - tx) <= 0.01 && abs(q.y - ty) <= 0.01 && u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0;
    if (!ok) {
        return vec2<f32>(-1e30, -1e30);
    }
    return vec2<f32>(x0 + nf(u * mesh[34]), y0 + nf(v * mesh[35]));
}

@compute @workgroup_size(16, 16)
fn warp_mesh(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let s = f32(1u << p.level);
    let gx = (f32(p.ox + i32(gid.x)) + 0.5) * s;
    let gy = (f32(p.oy + i32(gid.y)) + 0.5) * s;
    let l = mesh_map(gx, gy) / s;
    var v = 0u;
    if (l.x >= 0.0 && l.y >= 0.0 && l.x < f32(p.iw) && l.y < f32(p.ih)) {
        v = bilinear(l.x, l.y);
    }
    dst[gid.y * p.w + gid.x] = v;
}
