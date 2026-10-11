// Render ▸ Clouds, Difference Clouds, Fibers, Lens Flare and the Gradient
// tool's fill: point filters over the raw band (pad 0), positioned by the
// band origin and the mip level like `Filter::apply`.
//
// The noise is the CPU's value noise over `hash01` (identical lattice
// values); its smooth interpolation keeps float differences continuous.
// The Gradient's ramp parameter selects one of 1024 LUT entries, so it is
// computed with exactly the CPU's roundings: `opq` stops products from
// being fused into adds, and divisions / square roots are correctly rounded.

struct P {
    n: u32,
    pw: u32,
    ox: i32,
    oy: i32,
    zero: u32,
    s: f32,
    kind: u32,
    flag: u32,
    // Commit: CPU block width (256, the tile); 0: the whole band (preview).
    blk: u32,
    c0: u32,
    c1: u32,
    _0: u32,
    f: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
// Gradient: 1024 entries of premultiplied 8.8 fixed point (rg, ba).
@group(0) @binding(3) var<storage, read> lut: array<u32>;

fn pf(k: u32) -> f32 {
    return p.f[k >> 2u][k & 3u];
}

fn opq(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ p.zero);
}

// Correctly rounded x / y. Like the prelude's `p_div`, with the quotient and
// the result opaque so fast math can neither fold the residual `x - q * y`
// to zero nor reassociate the correction into a following sum.
fn div_rn(x: f32, y: f32) -> f32 {
    let q = opq(x / y);
    return opq(q + opq(fma(-q, y, x)) / y);
}

fn index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 256u;
}

// `util::value_noise`.
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

// `util::fbm`.
fn fbm(x: f32, y: f32, octaves: u32, seed: u32) -> f32 {
    var amp = 0.5;
    var freq = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    for (var o = 0u; o < octaves; o++) {
        sum += amp * value_noise(x * freq, y * freq, seed + o * 977u);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    return sum / norm;
}

// Clouds (flag 0) / Difference Clouds (flag 1): f0 = scale.
@compute @workgroup_size(256)
fn clouds(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= p.n) {
        return;
    }
    let gx = (f32(p.ox + i32(i % p.pw)) + 0.5) * p.s;
    let gy = (f32(p.oy + i32(i / p.pw)) + 0.5) * p.s;
    let scale = pf(0u);
    let v = (fbm(div_rn(gx, scale), div_rn(gy, scale), 7u, 1u) - 0.5) * 1.6 + 0.5;
    let c = lerp4(p.c0, p.c1, u32(clamp(v, 0.0, 1.0) * 255.0));
    if (p.flag == 0u) {
        dst[i] = c;
        return;
    }
    let o = vec4<i32>(unpack4(src[i]));
    let cc = vec4<i32>(unpack4(c));
    let d = min(vec3<u32>(abs(o.xyz - cc.xyz)), vec3<u32>(u32(o.w)));
    dst[i] = pack4(vec4<u32>(d, u32(o.w)));
}

// Fibers: f0 = horizontal scale, f1 = vertical scale.
@compute @workgroup_size(256)
fn fibers(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= p.n) {
        return;
    }
    let gx = f32(p.ox + i32(i % p.pw)) * p.s;
    let gy = f32(p.oy + i32(i / p.pw)) * p.s;
    let v0 = fbm(div_rn(gx, pf(0u)), div_rn(gy, pf(1u)), 4u, 9u);
    let v = clamp((v0 - 0.5) * 2.2 + 0.5, 0.0, 1.0);
    dst[i] = lerp4(p.c0, p.c1, u32(v * 255.0));
}

// Lens Flare: f0, f1 = center, f2 = brightness / 100, f3 = size.
@compute @workgroup_size(256)
fn lens_flare(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= p.n) {
        return;
    }
    let px = src[i];
    let a = px >> 24u;
    if (a == 0u) {
        dst[i] = px;
        return;
    }
    let gx = (f32(p.ox + i32(i % p.pw)) + 0.5) * p.s;
    let gy = (f32(p.oy + i32(i / p.pw)) + 0.5) * p.s;
    let dx = gx - pf(0u);
    let dy = gy - pf(1u);
    let d = sqrt(dx * dx + dy * dy) / pf(3u);
    let glow = p_exp(-d * d * 18.0) * 1.4 + p_exp(-d * 3.0) * 0.35;
    let q = (d - 0.18) * 60.0;
    let ring = p_exp(-(q * q)) * 0.25;
    let k = pf(2u);
    let add = vec3<f32>((glow + ring * 0.8) * k, (glow * 0.92 + ring) * k, (glow * 0.8 + ring * 1.2) * k);
    let c = vec3<f32>(unpack4(px).xyz);
    let fa = f32(a);
    let o = vec3<u32>(min(c + add * fa, vec3<f32>(fa)));
    dst[i] = pack4(vec4<u32>(o, a));
}

// ----- Gradient -----------------------------------------------------------------------------

// Correctly rounded square root.
fn p_sqrt(x: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    let r = sqrt(x);
    let lo = bitcast<f32>(bitcast<u32>(r) - 1u);
    let hi = bitcast<f32>(bitcast<u32>(r) + 1u);
    var best = r;
    var err = abs(fma(-r, r, x));
    let el = abs(fma(-lo, lo, x));
    if (el < err) {
        best = lo;
        err = el;
    }
    let eh = abs(fma(-hi, hi, x));
    if (eh < err) {
        best = hi;
    }
    return best;
}

// `gradient::fast_atan2`, with the CPU's roundings.
fn fast_atan2(y: f32, x: f32) -> f32 {
    let ax = abs(x);
    let ay = abs(y);
    let mn = min(ax, ay);
    let mx = max(ax, ay);
    if (mx == 0.0) {
        return 0.0;
    }
    let a = div_rn(mn, mx);
    let s = opq(a * a);
    var r = opq(opq(opq(opq(-0.04649647 * s) + 0.15931422) * s) - 0.32762276) * s;
    r = opq(opq(r) * a) + a;
    if (ay > ax) {
        r = 1.5707964 - r;
    }
    if (x < 0.0) {
        r = 3.1415927 - r;
    }
    return select(r, -r, y < 0.0);
}

// Ramp parameter of `GradientFill::row_t` for pixel `i` of a CPU block
// starting at `x0` (level-0 x of its first pixel center), row at level-0 `y`.
// f0, f1 = p0; f2, f3 = dx, dy; f4 = len2; f5 = 1 / len; f6, f7 = unit
// direction; f8 = dx / len2 * step; f9 = atan2(dy, dx).
fn ramp(y: f32, x0: f32, i: u32) -> f32 {
    let fi = f32(i);
    let step = p.s;
    let vy = y - pf(1u);
    switch p.kind {
        // Linear, Reflected
        case 0u, 3u: {
            let k0 = div_rn(opq(opq((x0 - pf(0u)) * pf(2u)) + opq(vy * pf(3u))), pf(4u));
            let v = k0 + opq(pf(8u) * fi);
            if (p.kind == 3u) {
                return min(abs(v), 1.0);
            }
            return clamp(v, 0.0, 1.0);
        }
        // Radial
        case 1u: {
            let vx = opq(x0 + opq(step * fi)) - pf(0u);
            return min(opq(p_sqrt(opq(vx * vx) + opq(vy * vy)) * pf(5u)), 1.0);
        }
        // Diamond
        case 4u: {
            let vx = opq(x0 + opq(step * fi)) - pf(0u);
            let a = abs(opq(vx * pf(6u)) + opq(vy * pf(7u)));
            let b = abs(opq(vy * pf(6u)) - opq(vx * pf(7u)));
            return min(opq(max(a, b) * pf(5u)), 1.0);
        }
        // Angle
        default: {
            let tau = 6.2831855;
            let vx = opq(x0 + opq(step * fi)) - pf(0u);
            let a0 = -(fast_atan2(vy, vx) - pf(9u));
            let a = a0 - opq(tau * floor(div_rn(a0, tau)));
            return clamp(div_rn(a, tau), 0.0, 1.0);
        }
    }
}

const BAYER: array<u32, 16> = array<u32, 16>(0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);

// `GradientFill::apply`: flag = dither, c0 = opacity (0..=255), over the band.
@compute @workgroup_size(256)
fn gradient(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = index(gid, nwg);
    if (i >= p.n) {
        return;
    }
    let bx = i % p.pw;
    let by = i / p.pw;
    // The CPU block holding this pixel: tiles (commit) or the whole band.
    var b0 = 0u;
    if (p.blk != 0u) {
        b0 = (bx / p.blk) * p.blk;
    }
    let xo = p.ox + i32(b0);
    let iy = p.oy + i32(by);
    let s = p.s;
    let gy = (f32(iy) + 0.5) * s;
    let t = ramp(gy, (f32(xo) + 0.5) * s, bx - b0);
    let k = u32(opq(t * 1023.0) + 0.5);
    let e0 = lut[k * 2u];
    let e1 = lut[k * 2u + 1u];
    let e = vec4<u32>(e0 & 65535u, e0 >> 16u, e1 & 65535u, e1 >> 16u);
    var d = 128u;
    if (p.flag != 0u) {
        let gx = p.ox + i32(bx);
        d = BAYER[u32((iy & 3) * 4 + (gx & 3))] * 16u + 8u;
    }
    let c = min((e + vec4<u32>(d)) >> vec4<u32>(8u), vec4<u32>(255u));
    var g = pack4(vec4<u32>(min(c.xyz, vec3<u32>(c.w)), c.w));
    if (p.c0 != 255u) {
        g = pack4(mul255(unpack4(g), p.c0));
    }
    let o = src[i];
    let sa = g >> 24u;
    if (sa == 255u) {
        dst[i] = g;
    } else if (sa == 0u) {
        dst[i] = o;
    } else {
        dst[i] = pack4(unpack4(g) + mul255(unpack4(o), 255u - sa));
    }
}
