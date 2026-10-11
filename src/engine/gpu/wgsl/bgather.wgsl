// Blur ▸ Radial Blur and Blur Gallery ▸ Spin Blur / Path Blur
// (`distort::render_radial`, `distort::render_path_blur`): each output pixel
// averages bilinear samples along an arc, a ray or a curve's tangent, read
// from the mip level whose pixel size matches the sample spacing, with the
// CPU sampler's 8-bit fractions. Per-tap constants (rotation steps, zoom
// factors, path offsets) and the mip thresholds come from the CPU, so no
// transcendental runs here.

struct P {
    // Output size and level coordinate of its first pixel.
    w: u32,
    h: u32,
    ox: i32,
    oy: i32,
    // 0 spin, 1 zoom, 2 path.
    kind: u32,
    // Radial: usable mip levels above this one. Path: the mip offset.
    extra: u32,
    // Spin Blur's ellipse mask; effects on.
    spin: u32,
    fxon: u32,
    // 2^level, radial center (level px), zoom amount, spin span (radians).
    scale: f32,
    lcx: f32,
    lcy: f32,
    amount: f32,
    span: f32,
    // Path: taps, 2^-m.
    n: u32,
    kinv: f32,
    _0: u32,
    // Spin: cx, cy, max(rx, 1), max(ry, 1) (level 0).
    a: vec4<f32>,
    // Path: p0, (p1 - p0) (level 0).
    b: vec4<f32>,
    // Path: ctrl - p0, p1 - ctrl.
    c: vec4<f32>,
    // Path: |p1 - p0|^2.
    d: vec4<f32>,
    // Per level: image size and tile-area size (stride, rows).
    dims: array<vec4<u32>, 7>,
    fx: Fx,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> l0: array<u32>;
@group(0) @binding(2) var<storage, read> l1: array<u32>;
@group(0) @binding(3) var<storage, read> l2: array<u32>;
@group(0) @binding(4) var<storage, read> l3: array<u32>;
@group(0) @binding(5) var<storage, read> l4: array<u32>;
@group(0) @binding(6) var<storage, read> l5: array<u32>;
@group(0) @binding(7) var<storage, read> l6: array<u32>;
@group(0) @binding(8) var<storage, read_write> dst: array<u32>;
// [spin: (c0, s0, cs, ss) per n = 1..24][zoom: t per (n, i)][path offsets
// (16)][mip thresholds (7)].
@group(0) @binding(9) var<storage, read> tab: array<f32>;
@group(0) @binding(10) var<storage, read> fxw: array<f32>;

const SPIN_TAB: u32 = 0u;
const ZOOM_TAB: u32 = 96u;
const PATH_TAB: u32 = 396u;
const MIP_TAB: u32 = 412u;

fn texel(m: u32, x: i32, y: i32) -> u32 {
    let d = p.dims[m];
    let xx = u32(clamp(x, 0, i32(d.x) - 1));
    let yy = u32(clamp(y, 0, i32(d.y) - 1));
    let i = yy * d.z + xx;
    switch m {
        case 0u: {
            return l0[i];
        }
        case 1u: {
            return l1[i];
        }
        case 2u: {
            return l2[i];
        }
        case 3u: {
            return l3[i];
        }
        case 4u: {
            return l4[i];
        }
        case 5u: {
            return l5[i];
        }
        default: {
            return l6[i];
        }
    }
}

// `Sampler::sample` on level offset `m`.
fn sample(m: u32, xs: f32, ys: f32) -> vec4<u32> {
    let x = xs - 0.5;
    let y = ys - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = u32((x - x0) * 255.0);
    let fy = u32((y - y0) * 255.0);
    let xi = i32(x0);
    let yi = i32(y0);
    let top = lerp4(texel(m, xi, yi), texel(m, xi + 1, yi), fx);
    let bot = lerp4(texel(m, xi, yi + 1), texel(m, xi + 1, yi + 1), fx);
    return unpack4(lerp4(top, bot, fy));
}

// The source pixel as `read_region` returns it (raw tile area, 0 outside).
fn orig(x: i32, y: i32) -> u32 {
    let d = p.dims[0];
    if (x < 0 || y < 0 || x >= i32(d.z) || y >= i32(d.w)) {
        return 0u;
    }
    return l0[u32(y) * d.z + u32(x)];
}

fn finish(acc: vec4<u32>, n: u32, src: u32, gx: f32, gy: f32) -> u32 {
    if (p.fxon == 0u) {
        return pack4((acc + vec4<u32>(n / 2u)) / vec4<u32>(n));
    }
    let nf = f32(n);
    let a = vec4<f32>(acc);
    let v = vec4<f32>(p_div(a.x, nf), p_div(a.y, nf), p_div(a.z, nf), p_div(a.w, nf));
    return fx_apply(p.fx, v, src, 1.0, gx, gy);
}

@compute @workgroup_size(16, 16)
fn radial(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let lx = p.ox + i32(gid.x);
    let ly = p.oy + i32(gid.y);
    let px_ = f32(lx) + 0.5;
    let py_ = f32(ly) + 0.5;
    let gx = px_ * p.scale;
    let gy = py_ * p.scale;
    let src = orig(lx, ly);
    let out_i = gid.y * p.w + gid.x;
    // Spin Blur: the ellipse mask (outside it, the source).
    var mask = 255u;
    if (p.spin != 0u) {
        let ex = p_div(gx - p.a.x, p.a.z);
        let ey = p_div(gy - p.a.y, p.a.w);
        let d = sqrt(ex * ex + ey * ey);
        mask = u32((1.0 - clamp(p_div(d - 0.85, 0.15), 0.0, 1.0)) * 255.0);
        if (mask == 0u) {
            dst[out_i] = src;
            return;
        }
    }
    let dx = px_ - p.lcx;
    let dy = py_ - p.lcy;
    let r = sqrt(dx * dx + dy * dy);
    var len: f32;
    if (p.kind == 1u) {
        len = p_div(r * p.amount, 100.0) * 0.5;
    } else {
        len = r * p.span;
    }
    let n = u32(clamp(ceil(len), 1.0, 24.0));
    let spacing = p_div(len, f32(n));
    var m = 0u;
    if (spacing > 1.5) {
        for (var j = 1u; j <= p.extra; j++) {
            if (spacing >= tab[MIP_TAB + j]) {
                m = j;
            }
        }
    }
    let k = ldexp(1.0, -i32(m));
    var acc = vec4<u32>(0u);
    if (p.kind == 1u) {
        let base = ZOOM_TAB + n * (n - 1u) / 2u;
        for (var i = 0u; i < n; i++) {
            let t = tab[base + i];
            acc += sample(m, (p.lcx + dx * t) * k, (p.lcy + dy * t) * k);
        }
    } else {
        let q = SPIN_TAB + (n - 1u) * 4u;
        let c0 = tab[q];
        let s0 = tab[q + 1u];
        let cs = tab[q + 2u];
        let ss = tab[q + 3u];
        var vx = dx * c0 - dy * s0;
        var vy = dx * s0 + dy * c0;
        for (var i = 0u; i < n; i++) {
            acc += sample(m, (p.lcx + vx) * k, (p.lcy + vy) * k);
            let nx = vx * cs - vy * ss;
            let ny = vx * ss + vy * cs;
            vx = nx;
            vy = ny;
        }
    }
    var o = finish(acc, n, src, gx, gy);
    if (p.spin != 0u) {
        o = lerp4(src, o, mask);
    }
    dst[out_i] = o;
}

@compute @workgroup_size(16, 16)
fn path(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let lx = p.ox + i32(gid.x);
    let ly = p.oy + i32(gid.y);
    let px_ = f32(lx) + 0.5;
    let py_ = f32(ly) + 0.5;
    let gx = px_ * p.scale;
    let gy = py_ * p.scale;
    let t = clamp(p_div((gx - p.b.x) * p.b.z + (gy - p.b.y) * p.b.w, p.d.x), 0.0, 1.0);
    let tx = 2.0 * (1.0 - t) * p.c.x + 2.0 * t * p.c.z;
    let ty = 2.0 * (1.0 - t) * p.c.y + 2.0 * t * p.c.w;
    let l = max(sqrt(tx * tx + ty * ty), 1e-6);
    let dx = p_div(tx, l);
    let dy = p_div(ty, l);
    let m = p.extra;
    var acc = vec4<u32>(0u);
    for (var i = 0u; i < p.n; i++) {
        let o = tab[PATH_TAB + i];
        acc += sample(m, (px_ + dx * o) * p.kinv, (py_ + dy * o) * p.kinv);
    }
    var src = 0u;
    if (p.fxon != 0u) {
        src = orig(lx, ly);
    }
    dst[gid.y * p.w + gid.x] = finish(acc, p.n, src, gx, gy);
}
