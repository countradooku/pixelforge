// Stylize ▸ Oil Paint and Pixelate ▸ Facet (`stylize::oil_paint`,
// `stylize::kuwahara`).
//
// Kuwahara, bit-exact: running box sums of (r, g, b, a, luma, luma²) over
// (r + 1)² windows, horizontal (`kuw_h`) then vertical (`kuw_v`, which also
// forms each window's variance key n·Σl² − (Σl)² in 64 bits and its rounded
// mean), then each pixel takes the mean of its lowest-variance quadrant
// (`kuw_sel`, first minimum wins as on the CPU).
//
// The paint stage (Stylization, Bristle Detail, Shine) is the CPU's float
// math in the same order: structure tensor (integer), stroke direction,
// line-integral smoothing of the Kuwahara result, bristle noise, relief.

struct P {
    pw: u32,
    ph: u32,
    pad: u32,
    cw: u32,
    ch: u32,
    ox: i32,
    oy: i32,
    r: u32,
    // Window side r + 1, window positions per row / column.
    sw: u32,
    bw: u32,
    bh: u32,
    // Outputs per thread of the running-sum passes.
    k: u32,
    zero: u32,
    rho: u32,
    ns: u32,
    nb: u32,
    // `kuw_sel` region (buffer coordinates) and output layout.
    rx0: u32,
    ry0: u32,
    rw: u32,
    rh: u32,
    ow: u32,
    oxo: u32,
    oyo: u32,
    full: u32,
    // step_s, inv_s, step_b, inv_b, inv_lambda, s, lx, ly, gain, kc, kh, half
    f: array<vec4<f32>, 3>,
    // The CPU's line weights: strokes at 0.., bristles at 2 ns + 1...
    wts: array<vec4<f32>, 152>,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> in0: array<u32>;
@group(0) @binding(2) var<storage, read> in1: array<u32>;
@group(0) @binding(3) var<storage, read> in2: array<u32>;
@group(0) @binding(4) var<storage, read> in3: array<u32>;
@group(0) @binding(5) var<storage, read_write> out0: array<u32>;
@group(0) @binding(6) var<storage, read_write> out1: array<u32>;
@group(0) @binding(7) var<storage, read_write> out2: array<u32>;
@group(0) @binding(8) var<storage, read_write> out3: array<u32>;

fn pf(k: u32) -> f32 {
    return p.f[k >> 2u][k & 3u];
}

fn opq(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ p.zero);
}

fn lum(c: vec4<u32>) -> u32 {
    return (c.x * 77u + c.y * 150u + c.z * 29u) >> 8u;
}

// ----- Kuwahara -----------------------------------------------------------------------------

// Horizontal window sums -> out0 = r | g << 16, out1 = b | a << 16,
// out2 = Σl, out3 = Σl² (index y * pw + x, window [x, x + sw)).
@compute @workgroup_size(1, 64)
fn kuw_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    let x0 = gid.x * p.k;
    if (y >= p.ph || x0 >= p.bw) {
        return;
    }
    let x1 = min(x0 + p.k, p.bw);
    let row = y * p.pw;
    var c = vec4<u32>(0u);
    var l = 0u;
    var l2 = 0u;
    for (var j = 0u; j < p.sw; j++) {
        let v = unpack4(in0[row + x0 + j]);
        let lv = lum(v);
        c += v;
        l += lv;
        l2 += lv * lv;
    }
    for (var x = x0; x < x1; x++) {
        let i = row + x;
        out0[i] = c.x | (c.y << 16u);
        out1[i] = c.z | (c.w << 16u);
        out2[i] = l;
        out3[i] = l2;
        if (x + 1u < x1) {
            let a = unpack4(in0[row + x + p.sw]);
            let b = unpack4(in0[row + x]);
            let la = lum(a);
            let lb = lum(b);
            c = c + a - b;
            l = l + la - lb;
            l2 = l2 + la * la - lb * lb;
        }
    }
}

// `kuw_h` for windows up to 32 wide: a 256-column row segment and its
// apron are loaded into shared memory once (coalesced), then each thread
// sums its window there and writes its column (coalesced).
const SEG: u32 = 256u;
var<workgroup> seg_px: array<vec4<u32>, 288>;
var<workgroup> seg_l: array<u32, 288>;

@compute @workgroup_size(256)
fn kuw_h_seg(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let y = wid.y;
    let x0 = wid.x * SEG;
    let n = SEG + p.sw - 1u;
    for (var k = lid.x; k < n; k += SEG) {
        let x = min(x0 + k, p.pw - 1u);
        let v = unpack4(in0[y * p.pw + x]);
        seg_px[k] = v;
        seg_l[k] = lum(v);
    }
    workgroupBarrier();
    let x = x0 + lid.x;
    if (x >= p.bw) {
        return;
    }
    var c = vec4<u32>(0u);
    var l = 0u;
    var l2 = 0u;
    for (var j = 0u; j < p.sw; j++) {
        c += seg_px[lid.x + j];
        let lv = seg_l[lid.x + j];
        l += lv;
        l2 += lv * lv;
    }
    let i = y * p.pw + x;
    out0[i] = c.x | (c.y << 16u);
    out1[i] = c.z | (c.w << 16u);
    out2[i] = l;
    out3[i] = l2;
}

fn mul64(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 65535u;
    let a1 = a >> 16u;
    let b0 = b & 65535u;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 65535u) + (p10 & 65535u);
    return vec2<u32>((p00 & 65535u) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}

// Vertical window sums -> out0, out1 = variance key (lo, hi), out2 = mean.
@compute @workgroup_size(256, 1)
fn kuw_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y0 = gid.y * p.k;
    if (x >= p.bw || y0 >= p.bh) {
        return;
    }
    let y1 = min(y0 + p.k, p.bh);
    var rg = vec2<u32>(0u);
    var ba = vec2<u32>(0u);
    var l = 0u;
    var l2 = 0u;
    for (var j = 0u; j < p.sw; j++) {
        let i = (y0 + j) * p.pw + x;
        rg += vec2<u32>(in0[i] & 65535u, in0[i] >> 16u);
        ba += vec2<u32>(in1[i] & 65535u, in1[i] >> 16u);
        l += in2[i];
        l2 += in3[i];
    }
    let cnt = p.sw * p.sw;
    let half = cnt / 2u;
    for (var y = y0; y < y1; y++) {
        let o = y * p.pw + x;
        let a = mul64(cnt, l2);
        let b = mul64(l, l);
        let lo = a.x - b.x;
        let hi = a.y - b.y - select(0u, 1u, a.x < b.x);
        out0[o] = lo;
        out1[o] = hi;
        let m = (vec4<u32>(rg, ba) + vec4<u32>(half)) / vec4<u32>(cnt);
        out2[o] = pack4(vec4<u32>(min(m.xyz, vec3<u32>(m.w)), m.w));
        if (y + 1u < y1) {
            let ia = (y + p.sw) * p.pw + x;
            let ib = y * p.pw + x;
            rg = rg + vec2<u32>(in0[ia] & 65535u, in0[ia] >> 16u) - vec2<u32>(in0[ib] & 65535u, in0[ib] >> 16u);
            ba = ba + vec2<u32>(in1[ia] & 65535u, in1[ia] >> 16u) - vec2<u32>(in1[ib] & 65535u, in1[ib] >> 16u);
            l = l + in2[ia] - in2[ib];
            l2 = l2 + in3[ia] - in3[ib];
        }
    }
}

// Quadrant choice for the region (rx0, ry0, rw, rh) of the buffer.
// in0 = band, in1..in3 = key lo, key hi, mean. Pixels closer than r to the
// buffer edge keep the input (`full`). out0[(y - oyo) * ow + x - oxo].
@compute @workgroup_size(16, 16)
fn kuw_sel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.rw || gid.y >= p.rh) {
        return;
    }
    let x = p.rx0 + gid.x;
    let y = p.ry0 + gid.y;
    let o = (y - p.oyo) * p.ow + (x - p.oxo);
    let r = p.r;
    if (r == 0u || x < r || y < r || x + r >= p.pw || y + r >= p.ph) {
        out0[o] = in0[y * p.pw + x];
        return;
    }
    var best = (y - r) * p.pw + (x - r);
    var bhi = in2[best];
    var blo = in1[best];
    let qs = array<u32, 3>((y - r) * p.pw + x, y * p.pw + (x - r), y * p.pw + x);
    for (var q = 0u; q < 3u; q++) {
        let i = qs[q];
        let hi = in2[i];
        let lo = in1[i];
        if (hi < bhi || (hi == bhi && lo < blo)) {
            best = i;
            bhi = hi;
            blo = lo;
        }
    }
    out0[o] = in3[best];
}

// ----- paint stage --------------------------------------------------------------------------

// Sobel products of the luma >> 4 -> out0..out2 (as i32), zero on the rim.
@compute @workgroup_size(16, 16)
fn oil_tensor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.pw || y >= p.ph) {
        return;
    }
    let i = y * p.pw + x;
    if (x < 1u || y < 1u || x + 1u >= p.pw || y + 1u >= p.ph) {
        out0[i] = 0u;
        out1[i] = 0u;
        out2[i] = 0u;
        return;
    }
    let w = p.pw;
    let tl = i32(lum(unpack4(in0[i - w - 1u])));
    let t = i32(lum(unpack4(in0[i - w])));
    let tr = i32(lum(unpack4(in0[i - w + 1u])));
    let l = i32(lum(unpack4(in0[i - 1u])));
    let r = i32(lum(unpack4(in0[i + 1u])));
    let bl = i32(lum(unpack4(in0[i + w - 1u])));
    let b = i32(lum(unpack4(in0[i + w])));
    let br = i32(lum(unpack4(in0[i + w + 1u])));
    let gx = (tr + 2 * r + br) - (tl + 2 * l + bl);
    let gy = (bl + 2 * b + br) - (tl + 2 * t + tr);
    out0[i] = bitcast<u32>((gx * gx) >> 4u);
    out1[i] = bitcast<u32>((gx * gy) >> 4u);
    out2[i] = bitcast<u32>((gy * gy) >> 4u);
}

// Horizontal (2 rho + 1) sums of in0..in2 -> out0..out2, at least rho + 1
// from the left / right edge (zero elsewhere). Thread: p.k columns of a row.
@compute @workgroup_size(1, 64)
fn oil_box_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    let x0 = gid.x * p.k;
    if (y >= p.ph || x0 >= p.pw) {
        return;
    }
    let x1 = min(x0 + p.k, p.pw);
    let m = p.rho + 1u;
    let row = y * p.pw;
    var s = vec3<i32>(0);
    var started = false;
    for (var x = x0; x < x1; x++) {
        let i = row + x;
        if (x < m || x + m >= p.pw) {
            out0[i] = 0u;
            out1[i] = 0u;
            out2[i] = 0u;
            continue;
        }
        if (!started) {
            for (var k = x - p.rho; k <= x + p.rho; k++) {
                s += vec3<i32>(bitcast<i32>(in0[row + k]), bitcast<i32>(in1[row + k]), bitcast<i32>(in2[row + k]));
            }
            started = true;
        } else {
            let a = row + x + p.rho;
            let b = row + x - p.rho - 1u;
            s += vec3<i32>(bitcast<i32>(in0[a]), bitcast<i32>(in1[a]), bitcast<i32>(in2[a]));
            s -= vec3<i32>(bitcast<i32>(in0[b]), bitcast<i32>(in1[b]), bitcast<i32>(in2[b]));
        }
        out0[i] = bitcast<u32>(s.x);
        out1[i] = bitcast<u32>(s.y);
        out2[i] = bitcast<u32>(s.z);
    }
}

// `stylize::oil_tangent` -> (tx, ty, anisotropy).
fn tangent(j: vec3<i32>) -> vec3<f32> {
    let a = f32(j.x);
    let b = f32(j.y);
    let c = f32(j.z);
    let d = sqrt((a - c) * (a - c) + 4.0 * b * b);
    let an = d / (a + c + 1.0);
    if (d > 0.0) {
        let c2 = (a - c) / d;
        let cp = sqrt(max((1.0 + c2) * 0.5, 0.0));
        var sp = sqrt(max((1.0 - c2) * 0.5, 0.0));
        if (b < 0.0) {
            sp = -sp;
        }
        return vec3<f32>(-sp, cp, an);
    }
    return vec3<f32>(1.0, 0.0, an);
}

// Vertical sums of the horizontal sums, then the stroke direction ->
// out0..out2 = tx, ty, anisotropy (f32). Thread: p.k rows of a column.
@compute @workgroup_size(256, 1)
fn oil_box_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y0 = gid.y * p.k;
    if (x >= p.pw || y0 >= p.ph) {
        return;
    }
    let y1 = min(y0 + p.k, p.ph);
    let m = p.rho + 1u;
    let inx = x >= m && x + m < p.pw;
    var s = vec3<i32>(0);
    var started = false;
    for (var y = y0; y < y1; y++) {
        let i = y * p.pw + x;
        var t = vec3<f32>(1.0, 0.0, 0.0);
        if (inx && y >= m && y + m < p.ph) {
            if (!started) {
                for (var k = y - p.rho; k <= y + p.rho; k++) {
                    let q = k * p.pw + x;
                    s += vec3<i32>(bitcast<i32>(in0[q]), bitcast<i32>(in1[q]), bitcast<i32>(in2[q]));
                }
                started = true;
            } else {
                let a = (y + p.rho) * p.pw + x;
                let b = (y - p.rho - 1u) * p.pw + x;
                s += vec3<i32>(bitcast<i32>(in0[a]), bitcast<i32>(in1[a]), bitcast<i32>(in2[a]));
                s -= vec3<i32>(bitcast<i32>(in0[b]), bitcast<i32>(in1[b]), bitcast<i32>(in2[b]));
            }
            t = tangent(s);
        }
        out0[i] = bitcast<u32>(t.x);
        out1[i] = bitcast<u32>(t.y);
        out2[i] = bitcast<u32>(t.z);
    }
}

fn kpx(x: u32, y: u32) -> vec4<f32> {
    return vec4<f32>(unpack4(in0[y * p.pw + x]));
}

// One axis of `bilerp`: pixel p plus offset d -> (i, j, fraction).
fn axis(pp: u32, d: f32, n: u32) -> vec3<f32> {
    let f = floor(d);
    let i = i32(pp) + i32(f);
    if (i < 0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    if (i >= i32(n) - 1) {
        return vec3<f32>(f32(n - 1u), f32(n - 1u), 0.0);
    }
    return vec3<f32>(f32(i), f32(i + 1), d - f);
}

// `stylize::bilerp`: the Kuwahara result (in0) at (x + dx, y + dy).
fn bilerp(x: u32, y: u32, dx: f32, dy: f32) -> vec4<f32> {
    let ax = axis(x, dx, p.pw);
    let ay = axis(y, dy, p.ph);
    let p00 = kpx(u32(ax.x), u32(ay.x));
    let p10 = kpx(u32(ax.y), u32(ay.x));
    let p01 = kpx(u32(ax.x), u32(ay.y));
    let p11 = kpx(u32(ax.y), u32(ay.y));
    let t = p00 + (p10 - p00) * ax.z;
    let b = p01 + (p11 - p01) * ax.z;
    return t + (b - t) * ay.z;
}

fn wt(i: u32) -> f32 {
    return p.wts[i >> 2u][i & 3u];
}

// The paint region: the center and a 1-pixel ring (the relief's neighbors).
fn region(gid: vec3<u32>) -> vec2<u32> {
    return vec2<u32>(gid.x + p.pad - 1u, gid.y + p.pad - 1u);
}

// Strokes: in0 = Kuwahara, in1..in3 = direction -> out0..out3 = paint (f32).
@compute @workgroup_size(16, 16)
fn oil_paint(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw + 2u || gid.y >= p.ch + 2u) {
        return;
    }
    let q = region(gid);
    let i = q.y * p.pw + q.x;
    var fv = kpx(q.x, q.y);
    let ns = p.ns;
    if (ns > 0u) {
        let tx = bitcast<f32>(in1[i]);
        let ty = bitcast<f32>(in2[i]);
        let an = bitcast<f32>(in3[i]);
        var acc = vec4<f32>(0.0);
        let n = i32(ns);
        for (var k = -n; k <= n; k++) {
            let o = f32(k) * pf(0u);
            let v = bilerp(q.x, q.y, o * tx, o * ty);
            acc += v * wt(u32(k + n));
        }
        fv += (acc * pf(1u) - fv) * an;
    }
    out0[i] = bitcast<u32>(fv.x);
    out1[i] = bitcast<u32>(fv.y);
    out2[i] = bitcast<u32>(fv.z);
    out3[i] = bitcast<u32>(fv.w);
}

// `util::value_noise` with the bristle seed.
fn value_noise(x: f32, y: f32) -> f32 {
    let xi = floor(x);
    let yi = floor(y);
    let fx = x - xi;
    let fy = y - yi;
    let ix = i32(xi);
    let iy = i32(yi);
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);
    let a = hash01(ix, iy, 71u);
    let b = hash01(ix + 1, iy, 71u);
    let c = hash01(ix, iy + 1, 71u);
    let d = hash01(ix + 1, iy + 1, 71u);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

// Bristles: in1..in3 = direction -> out0 = noise averaged along the stroke.
@compute @workgroup_size(16, 16)
fn oil_bristle(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw + 2u || gid.y >= p.ch + 2u) {
        return;
    }
    let q = region(gid);
    let i = q.y * p.pw + q.x;
    let tx = bitcast<f32>(in1[i]);
    let ty = bitcast<f32>(in2[i]);
    let gx = f32(p.ox + i32(q.x));
    let gy = f32(p.oy + i32(q.y));
    let s = pf(5u);
    let il = pf(4u);
    let n = i32(p.nb);
    var acc = 0.0;
    for (var k = -n; k <= n; k++) {
        let o = f32(k) * pf(2u);
        acc += value_noise((gx + o * tx + 0.5) * s * il, (gy + o * ty + 0.5) * s * il) * wt(select(0u, 2u * p.ns + 1u, p.ns > 0u) + u32(k + n));
    }
    out0[i] = bitcast<u32>(acc * pf(3u));
}

fn paint_at(i: u32) -> vec4<f32> {
    return vec4<f32>(bitcast<f32>(in0[i]), bitcast<f32>(in1[i]), bitcast<f32>(in2[i]), bitcast<f32>(in3[i]));
}

fn bristle_at(i: u32) -> f32 {
    if (p.nb == 0u) {
        return 0.5;
    }
    return bitcast<f32>(out1[i]);
}

fn height(i: u32) -> f32 {
    let f = paint_at(i);
    return (0.299 * f.x + 0.587 * f.y + 0.114 * f.z) / 255.0 + (bristle_at(i) - 0.5) * pf(10u);
}

// Bristle color, relief and quantization: in0..in3 = paint, out1 = bristles
// -> out0 = the center.
@compute @workgroup_size(16, 16)
fn oil_final(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = gid.x + p.pad;
    let y = gid.y + p.pad;
    let i = y * p.pw + x;
    let fv = paint_at(i);
    let m = 1.0 + (bristle_at(i) - 0.5) * pf(9u);
    var e = 0.0;
    let gain = pf(8u);
    if (gain > 0.0) {
        let half = pf(11u);
        let hx = (height(i + 1u) - height(i - 1u)) * half;
        let hy = (height(i + p.pw) - height(i - p.pw)) * half;
        e = -(hx * pf(6u) + hy * pf(7u)) * gain;
    }
    let a8 = u32(clamp(fv.w + 0.5, 0.0, 255.0));
    let fa = f32(a8);
    let c = clamp(fv.xyz * m + vec3<f32>(e * fv.w) + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(fa));
    out0[gid.y * p.cw + gid.x] = pack4(vec4<u32>(vec3<u32>(c), a8));
}
