// Filter Gallery: shared declarations of every gallery module.
//
// The CPU renders a gallery effect per padded window (one block of tiles
// plus its apron, `Filter::render`), on straight-alpha f32 planes. The GPU
// runs the same pipeline on a batch of such windows at once: plane `k` of
// window `z` is `z * ww * wh + y * ww + x`, and `win[z]` says where the
// window sits. Float operations whose rounding matters (anything that feeds a
// threshold, a quantization or a rank) go through `o()`: Metal compiles with
// fast math, which may fuse a * b + c or reorder sums; an opaque OR with a
// uniform zero pins every rounding to the CPU's order.

struct P {
    ww: u32,
    wh: u32,
    zero: u32,
    s: u32,
    nw: u32,
    pad: u32,
    srcw: u32,
    dstw: u32,
    i: vec4<i32>,
    j: vec4<i32>,
    f: vec4<f32>,
    g: vec4<f32>,
    h: vec4<f32>,
    k: vec4<f32>,
}

struct Win {
    w: u32,
    h: u32,
    // Level-space coordinate of the window's pixel (0, 0).
    ox: i32,
    oy: i32,
    // The window's offset in the band input.
    sx: u32,
    sy: u32,
    // The center (output) rect inside the window, and its place in the output.
    cx: u32,
    cy: u32,
    cw: u32,
    ch: u32,
    dx: u32,
    dy: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> win: array<Win>;
@group(0) @binding(2) var<storage, read> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;
@group(0) @binding(4) var<storage, read> a0: array<f32>;
@group(0) @binding(5) var<storage, read> a1: array<f32>;
@group(0) @binding(6) var<storage, read> a2: array<f32>;
@group(0) @binding(7) var<storage, read> a3: array<f32>;
@group(0) @binding(8) var<storage, read> a4: array<f32>;
@group(0) @binding(9) var<storage, read> a5: array<f32>;
@group(0) @binding(10) var<storage, read_write> b0: array<f32>;
@group(0) @binding(11) var<storage, read_write> b1: array<f32>;
@group(0) @binding(12) var<storage, read_write> b2: array<f32>;
@group(0) @binding(13) var<storage, read_write> b3: array<f32>;
@group(0) @binding(14) var<storage, read> u0: array<u32>;
@group(0) @binding(15) var<storage, read> u1: array<u32>;
@group(0) @binding(16) var<storage, read_write> v0: array<u32>;
@group(0) @binding(17) var<storage, read> t0: array<u32>;
@group(0) @binding(18) var<storage, read> t1: array<u32>;
@group(0) @binding(19) var<storage, read> t2: array<u32>;
@group(0) @binding(20) var<storage, read_write> v1: array<u32>;
@group(0) @binding(21) var<storage, read_write> v2: array<u32>;
@group(0) @binding(22) var<storage, read> q0: array<vec4<f32>>;
@group(0) @binding(23) var<storage, read> q1: array<vec4<f32>>;
@group(0) @binding(24) var<storage, read_write> w0: array<vec4<f32>>;

// ----- exact float arithmetic ------------------------------------------------

fn o(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | p.zero);
}

fn o4(x: vec4<f32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(x) | vec4<u32>(p.zero));
}

fn fadd(a: f32, b: f32) -> f32 {
    return o(a + b);
}

fn fsub(a: f32, b: f32) -> f32 {
    return o(a - b);
}

fn fmul(a: f32, b: f32) -> f32 {
    return o(a * b);
}

// Rust's a * b + c (two roundings).
fn fmad(a: f32, b: f32, c: f32) -> f32 {
    return o(o(a * b) + c);
}

fn next_up_mag(q: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(q) + 1u);
}

fn next_down_mag(q: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(q) - 1u);
}

// Correctly rounded x / y: refine the hardware quotient, then keep the
// neighbor with the smallest exact residual x - q y.
fn fdiv(x: f32, y: f32) -> f32 {
    var q = o(x / y);
    if (x == 0.0 || y == 0.0) {
        return q;
    }
    q = o(q + o(fma(-q, y, x) / y));
    if ((bitcast<u32>(q) & 0x7fffffffu) == 0u || (bitcast<u32>(q) & 0x7f800000u) == 0x7f800000u) {
        return q;
    }
    let up = next_up_mag(q);
    let dn = next_down_mag(q);
    let r0 = abs(fma(-q, y, x));
    let r1 = abs(fma(-up, y, x));
    let r2 = abs(fma(-dn, y, x));
    var best = q;
    var rb = r0;
    if (r1 < rb || (r1 == rb && (bitcast<u32>(up) & 1u) == 0u)) {
        best = up;
        rb = r1;
    }
    if (r2 < rb || (r2 == rb && (bitcast<u32>(dn) & 1u) == 0u)) {
        best = dn;
    }
    return best;
}

// Correctly rounded sqrt (x >= 0).
fn fsqrt(x: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    var s = o(sqrt(x));
    let up = next_up_mag(s);
    let dn = next_down_mag(s);
    // Compare (c + c') / 2 against sqrt(x) through the residual of the midpoint.
    let r0 = abs(fma(-s, s, x));
    let r1 = abs(fma(-up, up, x));
    let r2 = abs(fma(-dn, dn, x));
    var best = s;
    var rb = r0;
    if (r1 < rb) {
        best = up;
        rb = r1;
    }
    if (r2 < rb) {
        best = dn;
    }
    return best;
}

// Rust's f32::round (half away from zero), exactly.
fn rround(x: f32) -> f32 {
    let f = floor(x);
    let d = o(x - f);
    if (x >= 0.0) {
        return select(f, f + 1.0, d >= 0.5);
    }
    return select(f, f + 1.0, d > 0.5);
}

// `(v.clamp(0, 1) * 255 + 0.5) as u8`.
fn q8(v: f32) -> u32 {
    let c = clamp(v, 0.0, 1.0);
    return min(u32(o(o(c * 255.0) + 0.5)), 255u);
}

// u8 / 255.0 as the CPU computes it (table t1 when bound; this is exact for
// all 256 inputs because the quotient is correctly rounded).
fn by255(v: u32) -> f32 {
    return fdiv(f32(v), 255.0);
}

// ----- windows -------------------------------------------------------------

fn wbase(z: u32) -> u32 {
    return z * p.ww * p.wh;
}

fn widx(z: u32, x: u32, y: u32) -> u32 {
    return z * p.ww * p.wh + y * p.ww + x;
}

// Clamped index inside window z.
fn wcl(z: u32, x: i32, y: i32) -> u32 {
    let w = win[z];
    let cx = u32(clamp(x, 0, i32(w.w) - 1));
    let cy = u32(clamp(y, 0, i32(w.h) - 1));
    return z * p.ww * p.wh + cy * p.ww + cx;
}

// Level-0 integer coordinate of window pixel x along an axis (`floor` of
// `ctx.g`): ((o + x) + 0.5) * s, floored.
fn g0(o_: i32, x: i32) -> i32 {
    let s = i32(p.s);
    return (o_ + x) * s + (s >> 1u);
}

// `ctx.g` (level-0 float coordinate of the pixel center).
fn gf(o_: i32, x: i32) -> f32 {
    return (f32(o_ + x) + 0.5) * f32(p.s);
}

fn sf() -> f32 {
    return f32(p.s);
}

fn tabf(t: u32) -> f32 {
    return bitcast<f32>(t);
}

fn sstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn mix3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

// The CPU's `util::value_noise` and `util::fbm` (smooth value noise).
fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let xi = floor(x);
    let yi = floor(y);
    let fx = x - xi;
    let fy = y - yi;
    let ix = i32(xi);
    let iy = i32(yi);
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);
    let a = hash01(ix, iy, seed);
    let b = hash01(ix + 1, iy, seed);
    let c = hash01(ix, iy + 1, seed);
    let d = hash01(ix + 1, iy + 1, seed);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

fn fbm(x: f32, y: f32, octaves: u32, seed: u32) -> f32 {
    var amp = 0.5;
    var freq = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    for (var k = 0u; k < octaves; k++) {
        sum += amp * value_noise(x * freq, y * freq, seed + k * 977u);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum / norm;
}

// Rust's `%` on f32 (truncated, sign of the dividend), positive divisor.
fn rmod(x: f32, m: f32) -> f32 {
    let q = trunc(fdiv(x, m));
    return fsub(x, fmul(q, m));
}

// `gallery::texture` height at level-0 point (x, y) already divided by the
// scale: 0 canvas, 1 burlap, 2 brick, 3 sandstone.
fn texture_h(x: f32, y: f32, kind: i32) -> f32 {
    switch kind {
        case 0: {
            return 0.5 + 0.25 * (p_sin(x * 1.1) * p_cos(y * 1.1)) + 0.1 * (hash01(i32(x) / 3, i32(y) / 3, 5u) - 0.5);
        }
        case 1: {
            let sx = p_sin(x * 0.5) * 0.5 + 0.5;
            let sy = p_sin(y * 0.5) * 0.5 + 0.5;
            let wx = sx * sx * sx;
            let wy = sy * sy * sy;
            return 0.35 + 0.4 * max(wx, wy) + 0.2 * fbm(x * 0.2, y * 0.2, 3u, 3u);
        }
        case 2: {
            let bh = 12.0;
            let row = floor(fdiv(y, bh));
            let bx = fadd(x, select(12.0, 0.0, i32(row) % 2 == 0));
            let mortar = rmod(y, bh) < 1.5 || rmod(bx, 24.0) < 1.5;
            if (mortar) {
                return 0.15;
            }
            return 0.6 + 0.2 * (hash01(i32(bx) / 24, i32(row), 9u) - 0.5);
        }
        default: {
            return fbm(x * 0.6, y * 0.6, 4u, 11u);
        }
    }
}

// `util::value_noise` / `util::fbm` with the CPU's exact rounding (for
// displacement fields, where noise errors move samples).
fn value_noise_x(x: f32, y: f32, seed: u32) -> f32 {
    let xi = floor(x);
    let yi = floor(y);
    let fx = fsub(x, xi);
    let fy = fsub(y, yi);
    let ix = i32(xi);
    let iy = i32(yi);
    let u = fmul(fmul(fx, fx), fsub(3.0, fmul(2.0, fx)));
    let v = fmul(fmul(fy, fy), fsub(3.0, fmul(2.0, fy)));
    let a = hash01(ix, iy, seed);
    let b = hash01(ix + 1, iy, seed);
    let c = hash01(ix, iy + 1, seed);
    let d = hash01(ix + 1, iy + 1, seed);
    let t1 = fadd(a, fmul(fsub(b, a), u));
    let t2 = fadd(t1, fmul(fsub(c, a), v));
    return fadd(t2, fmul(fmul(fadd(fsub(fsub(a, b), c), d), u), v));
}

fn fbm_x(x: f32, y: f32, octaves: u32, seed: u32) -> f32 {
    var amp = 0.5;
    var freq = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    for (var k = 0u; k < octaves; k++) {
        sum = fadd(sum, fmul(amp, value_noise_x(fmul(x, freq), fmul(y, freq), seed + k * 977u)));
        norm = fadd(norm, amp);
        amp = fmul(amp, 0.5);
        freq = fmul(freq, 2.0);
    }
    return fdiv(sum, norm);
}

// Writes a pass's color result: into b0..b2, or, when the pass is the
// effect's last (p.k.w = 1), straight into the band output like `store`
// (alpha from a5), skipping three plane writes and the store pass.
fn emit3(id: vec3<u32>, i: u32, c: vec3<f32>) {
    if (p.k.w > 0.5) {
        let w = win[id.z];
        if (id.x < w.cx || id.y < w.cy || id.x >= w.cx + w.cw || id.y >= w.cy + w.ch) {
            return;
        }
        dst[(w.dy + id.y - w.cy) * p.dstw + w.dx + id.x - w.cx] = premul(vec3<u32>(q8(c.x), q8(c.y), q8(c.z)), q8(a5[i]));
        return;
    }
    b0[i] = c.x;
    b1[i] = c.y;
    b2[i] = c.z;
}
