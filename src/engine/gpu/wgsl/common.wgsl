// Shared helpers, prepended to every module.
//
// Pixels are packed premultiplied RGBA8 (`r | g << 8 | b << 16 | a << 24`),
// exactly like the CPU tiles. Integer-defined operations use only integer
// math here (or tables computed on the CPU), so they are bit-exact: Metal
// compiles with fast math, which contracts and approximates float math.

fn unpack4(p: u32) -> vec4<u32> {
    return vec4<u32>(p & 255u, (p >> 8u) & 255u, (p >> 16u) & 255u, p >> 24u);
}

fn pack4(c: vec4<u32>) -> u32 {
    return c.x | (c.y << 8u) | (c.z << 16u) | (c.w << 24u);
}

// Exact-rounding x * f / 255 for x, f in 0..=255 (the CPU's `div255_lanes`).
fn mul255(x: vec4<u32>, f: u32) -> vec4<u32> {
    let v = x * f + vec4<u32>(128u);
    return (v + (v >> vec4<u32>(8u))) >> vec4<u32>(8u);
}

// `px::lerp`: a * (255 - t) + b * t over 255, one rounding.
fn lerp4(a: u32, b: u32, t: u32) -> u32 {
    let v = unpack4(a) * (255u - t) + unpack4(b) * t + vec4<u32>(128u);
    return pack4((v + (v >> vec4<u32>(8u))) >> vec4<u32>(8u));
}

// `px::premultiply`.
fn premul(c: vec3<u32>, a: u32) -> u32 {
    if (a == 255u) {
        return pack4(vec4<u32>(c, 255u));
    }
    let s = mul255(vec4<u32>(c, 0u), a);
    return pack4(vec4<u32>(s.xyz, a));
}

// Clamped 2D index into a w x h buffer.
fn cidx(x: i32, y: i32, w: u32, h: u32) -> u32 {
    let cx = u32(clamp(x, 0, i32(w) - 1));
    let cy = u32(clamp(y, 0, i32(h) - 1));
    return cy * w + cx;
}

// The CPU's `hash01` (stateless integer hash -> [0, 1)).
fn hash_u(x: i32, y: i32, seed: u32) -> u32 {
    var h = (bitcast<u32>(x) * 0x8da6b343u) ^ (bitcast<u32>(y) * 0xd8163841u) ^ (seed * 0xcb1ab31fu);
    h = h ^ (h >> 13u);
    h = h * 0x5bd1e995u;
    h = h ^ (h >> 15u);
    return h >> 8u;
}

fn hash01(x: i32, y: i32, seed: u32) -> f32 {
    return f32(hash_u(x, y, seed)) / 16777216.0;
}

// ----- precise math --------------------------------------------------------
// Metal's fast-math transcendentals can be off by more than float rounding,
// which moves sample positions and thresholds. These single-precision
// Cephes approximations (1-2 ulp) keep float kernels within 1/255 of the CPU.

fn p_sincos(xin: f32) -> vec2<f32> {
    // Returns (sin, cos).
    var x = abs(xin);
    var sgn_s = select(1.0, -1.0, xin < 0.0);
    var sgn_c = 1.0;
    var j = u32(x * 1.27323954473516);
    var y = f32(j);
    if ((j & 1u) != 0u) {
        j += 1u;
        y += 1.0;
    }
    j = j & 7u;
    if (j > 3u) {
        sgn_s = -sgn_s;
        sgn_c = -sgn_c;
        j -= 4u;
    }
    if (j > 1u) {
        sgn_c = -sgn_c;
    }
    let z = ((x - y * 0.78515625) - y * 2.4187564849853515625e-4) - y * 3.77489497744594108e-8;
    let zz = z * z;
    let c = ((2.443315711809948e-5 * zz - 1.388731625493765e-3) * zz + 4.166664568298827e-2) * zz * zz - 0.5 * zz + 1.0;
    let s = ((-1.9515295891e-4 * zz + 8.3321608736e-3) * zz - 1.6666654611e-1) * zz * z + z;
    if (j == 1u || j == 2u) {
        return vec2<f32>(sgn_s * c, sgn_c * s);
    }
    return vec2<f32>(sgn_s * s, sgn_c * c);
}

// Correctly rounded x / y (fast math may use an approximate reciprocal).
fn p_div(x: f32, y: f32) -> f32 {
    let q = x / y;
    return q + fma(-q, y, x) / y;
}

fn p_sin(x: f32) -> f32 {
    return p_sincos(x).x;
}

fn p_cos(x: f32) -> f32 {
    return p_sincos(x).y;
}

fn p_atan(xin: f32) -> f32 {
    var x = abs(xin);
    var y = 0.0;
    if (x > 2.414213562373095) {
        y = 1.5707963267948966;
        x = -1.0 / x;
    } else if (x > 0.4142135623730950) {
        y = 0.7853981633974483;
        x = (x - 1.0) / (x + 1.0);
    }
    let z = x * x;
    y += (((8.05374449538e-2 * z - 1.38776856032e-1) * z + 1.99777106478e-1) * z - 3.33329491539e-1) * z * x + x;
    return select(y, -y, xin < 0.0);
}

fn p_atan2(y: f32, x: f32) -> f32 {
    if (x == 0.0) {
        if (y > 0.0) {
            return 1.5707963267948966;
        }
        if (y < 0.0) {
            return -1.5707963267948966;
        }
        return 0.0;
    }
    let a = p_atan(y / x);
    if (x < 0.0) {
        return select(a - 3.141592653589793, a + 3.141592653589793, y >= 0.0);
    }
    return a;
}

fn p_asin(xin: f32) -> f32 {
    var a = abs(xin);
    if (a > 1.0) {
        return 0.0;
    }
    var z: f32;
    var x: f32;
    var flag = false;
    if (a > 0.5) {
        z = 0.5 * (1.0 - a);
        x = sqrt(z);
        flag = true;
    } else {
        x = a;
        z = x * x;
    }
    var r = ((((4.2163199048e-2 * z + 2.4181311049e-2) * z + 4.5470025998e-2) * z + 7.4953002686e-2) * z + 1.6666752422e-1) * z * x + x;
    if (flag) {
        r = 1.5707963267948966 - (r + r);
    }
    return select(r, -r, xin < 0.0);
}

fn p_exp(xin: f32) -> f32 {
    let x0 = clamp(xin, -87.3, 88.7);
    let z = floor(1.44269504088896341 * x0 + 0.5);
    var x = x0 - z * 0.693359375;
    x = x - z * -2.12194440e-4;
    let xx = x * x;
    let p = (((((1.9875691500e-4 * x + 1.3981999507e-3) * x + 8.3334519073e-3) * x + 4.1665795894e-2) * x + 1.6666665459e-1) * x + 5.0000001201e-1) * xx + x + 1.0;
    return ldexp(p, i32(z));
}

fn p_log(xin: f32) -> f32 {
    if (xin <= 0.0) {
        return -3.4e38;
    }
    let fr = frexp(xin);
    var e = f32(fr.exp);
    var x = fr.fract;
    if (x < 0.707106781186547524) {
        e -= 1.0;
        x = x + x - 1.0;
    } else {
        x = x - 1.0;
    }
    let z = x * x;
    var y = ((((((((7.0376836292e-2 * x - 1.1514610310e-1) * x + 1.1676998740e-1) * x - 1.2420140846e-1) * x + 1.4249322787e-1) * x - 1.6668057665e-1) * x + 2.0000714765e-1) * x - 2.4999993993e-1) * x + 3.3333331174e-1) * x * z;
    y += -2.12194440e-4 * e;
    y += -0.5 * z;
    return x + y + 0.693359375 * e;
}

fn p_pow(x: f32, y: f32) -> f32 {
    if (x <= 0.0) {
        return select(0.0, 1.0, y == 0.0);
    }
    return p_exp(y * p_log(x));
}
