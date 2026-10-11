// Layer Styles (`engine::styles`): every effect plane is integer-defined
// and bit-exact with the CPU (disc dilation by a windowed min-plus
// transform, box blurs with the CPU's rounding, shifts, contour tables,
// the bevel relief through a CPU-built table); the final per-pixel stack
// ports `styles::compose_px` (integer for Normal, float for other modes).
//
// Planes are u32 per pixel holding 0..255; finished effects are packed four
// per pixel into "lane" planes (lane = byte).

// Parameters: 64 words (see `gpu::styles::Words`).
@group(0) @binding(0) var<uniform> pw: array<vec4<u32>, 16>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> aux: array<u32>;
@group(0) @binding(4) var<storage, read> lane0: array<u32>;
@group(0) @binding(5) var<storage, read> lane1: array<u32>;
@group(0) @binding(6) var<storage, read> lane2: array<u32>;
@group(0) @binding(7) var<storage, read> unpre: array<u32>;

fn pu(i: u32) -> u32 {
    return pw[i >> 2u][i & 3u];
}
fn pi(i: u32) -> i32 {
    return bitcast<i32>(pu(i));
}
fn pf(i: u32) -> f32 {
    return bitcast<f32>(pu(i));
}

const W_: u32 = 0u;
const H_: u32 = 1u;
const PAD: u32 = 2u;
const CW: u32 = 3u;
const CH: u32 = 4u;
const MODE: u32 = 5u;
const R: u32 = 6u;
const INV: u32 = 7u;
const K: u32 = 8u;
const R16: u32 = 9u;
const DX: u32 = 10u;
const DY: u32 = 11u;
const FILL: u32 = 12u;
const INVERT: u32 = 13u;
const NOISE: u32 = 14u;
const HAS_LUT: u32 = 15u;
const SHIFT: u32 = 16u;
const OX: u32 = 17u;
const OY: u32 = 18u;
const NSQ: u32 = 19u;
const FILL8: u32 = 20u;
const FLAGS: u32 = 21u;

const INF: u32 = 0xffffffffu;

fn mul1(a: u32, b: u32) -> u32 {
    let v = a * b + 128u;
    return (v + (v >> 8u)) >> 8u;
}

// ----- planes ---------------------------------------------------------------------------

// mode 0: alpha, 1: 255 - alpha (src = pixels); 2: lane SHIFT of src.
@compute @workgroup_size(16, 16)
fn extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = pu(W_);
    if (gid.x >= w || gid.y >= pu(H_)) {
        return;
    }
    let i = gid.y * w + gid.x;
    let m = pu(MODE);
    var v: u32;
    if (m == 2u) {
        v = (src[i] >> pu(SHIFT)) & 255u;
    } else {
        v = src[i] >> 24u;
        if (m == 1u) {
            v = 255u - v;
        }
    }
    dst[i] = v;
}

// Min-plus row pass: dst = min over |dx| <= R of 256 dx^2 + f(src), with
// f(v) = 255 - v for v > 0 (INF otherwise); values >= 256 R^2 -> INF.
// Scanning outwards stops once 256 dx^2 can't win.
var<workgroup> row: array<u32, 1024>;

@compute @workgroup_size(256, 1)
fn edt_row(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let w = pu(W_);
    let y = wid.y;
    let win = pu(R);
    let x0 = i32(wid.x * 256u) - i32(win);
    let n = 256u + 2u * win;
    for (var t = lid.x; t < n; t += 256u) {
        let x = x0 + i32(t);
        var f = INF;
        if (x >= 0 && x < i32(w)) {
            let v = src[y * w + u32(x)];
            if (v > 0u) {
                f = 255u - v;
            }
        }
        row[t] = f;
    }
    workgroupBarrier();
    let x = wid.x * 256u + lid.x;
    if (x >= w) {
        return;
    }
    let c = lid.x + win;
    var best = row[c];
    for (var k = 1u; k <= win; k++) {
        let kk = k * k * 256u;
        if (kk >= best) {
            break;
        }
        let a = min(row[c - k], row[c + k]);
        if (a != INF) {
            best = min(best, kk + a);
        }
    }
    if (best != INF && best >= win * win * 256u) {
        best = INF;
    }
    dst[y * w + x] = best;
}

// Column pass + coverage: D = min over |dy| <= R of 256 dy^2 + src (the
// row pass), coverage through the sqrt table (aux) and R16, max with the
// plane in dst; MODE 1 inverts the result (erosion).
@compute @workgroup_size(64, 4)
fn edt_col(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = pu(W_);
    let h = pu(H_);
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) {
        return;
    }
    let win = pu(R);
    var best = src[y * w + x];
    for (var k = 1u; k <= win; k++) {
        let kk = k * k * 256u;
        if (kk >= best) {
            break;
        }
        var a = INF;
        if (y >= k) {
            a = src[(y - k) * w + x];
        }
        if (y + k < h) {
            a = min(a, src[(y + k) * w + x]);
        }
        if (a != INF) {
            best = min(best, kk + a);
        }
    }
    var cov = 0u;
    if (best != INF) {
        let d2 = best >> 8u;
        if (d2 < pu(NSQ)) {
            let a = 255u - (best & 255u);
            let c = i32(pu(R16)) + i32(a) * 16 - i32(aux[d2]);
            if (c > 0) {
                cov = min(255u, u32(c + 8) >> 4u);
            }
        }
    }
    let i = y * w + x;
    var v = max(dst[i], cov);
    if (pu(MODE) == 1u) {
        v = 255u - v;
    }
    dst[i] = v;
}

fn box_round(s: u32) -> u32 {
    return (s * pu(INV) + 8388608u) >> 24u;
}

// One vertical box pass (radius R, clamped at the buffer edge), K rows per thread.
@compute @workgroup_size(256, 1)
fn box_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = pu(W_);
    let h = pu(H_);
    let x = gid.x;
    if (x >= w) {
        return;
    }
    let y0 = i32(gid.y * pu(K));
    if (y0 >= i32(h)) {
        return;
    }
    let y1 = min(y0 + i32(pu(K)), i32(h));
    let r = i32(pu(R));
    let hm = i32(h) - 1;
    var s = 0u;
    for (var j = -r; j <= r; j++) {
        s += src[u32(clamp(y0 + j, 0, hm)) * w + x];
    }
    for (var y = y0; y < y1; y++) {
        dst[u32(y) * w + x] = box_round(s);
        s += src[u32(clamp(y + r + 1, 0, hm)) * w + x];
        s -= src[u32(clamp(y - r, 0, hm)) * w + x];
    }
}

// One horizontal box pass, K columns per thread.
@compute @workgroup_size(1, 64)
fn box_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = pu(W_);
    let h = pu(H_);
    let y = gid.y;
    if (y >= h) {
        return;
    }
    let x0 = i32(gid.x * pu(K));
    if (x0 >= i32(w)) {
        return;
    }
    let x1 = min(x0 + i32(pu(K)), i32(w));
    let r = i32(pu(R));
    let wm = i32(w) - 1;
    let rw = y * w;
    var s = 0u;
    for (var j = -r; j <= r; j++) {
        s += src[rw + u32(clamp(x0 + j, 0, wm))];
    }
    for (var x = x0; x < x1; x++) {
        dst[rw + u32(x)] = box_round(s);
        s += src[rw + u32(clamp(x + r + 1, 0, wm))];
        s -= src[rw + u32(clamp(x - r, 0, wm))];
    }
}

fn plane_at(x: i32, y: i32, fill: u32) -> u32 {
    if (x < 0 || y < 0 || x >= i32(pu(W_)) || y >= i32(pu(H_))) {
        return fill;
    }
    return src[u32(y) * pu(W_) + u32(x)];
}

fn put_lane(i: u32, v: u32) {
    let s = pu(SHIFT);
    dst[i] = (dst[i] & ~(255u << s)) | (v << s);
}

// `Fin::Map`: shift (fill outside), contour table (aux), invert, noise.
@compute @workgroup_size(16, 16)
fn finish_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = pu(W_);
    if (gid.x >= w || gid.y >= pu(H_)) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    var v = plane_at(x - pi(DX), y - pi(DY), pu(FILL));
    if (pu(HAS_LUT) != 0u) {
        v = aux[v];
    }
    if (pu(INVERT) != 0u) {
        v = 255u - v;
    }
    let nz = pu(NOISE);
    if (nz > 0u) {
        let h8 = hash_u(pi(OX) + x, pi(OY) + y, 0x5eedu) >> 16u;
        let j = (nz * h8 + 127u) / 255u;
        v = mul1(v, 255u - j);
    }
    put_lane(gid.y * w + gid.x, v);
}

// `Fin::Satin`: |plane(x - o) - plane(x + o)|, optionally inverted.
@compute @workgroup_size(16, 16)
fn finish_satin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = pu(W_);
    if (gid.x >= w || gid.y >= pu(H_)) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let a = i32(plane_at(x - pi(DX), y - pi(DY), 0u));
    let b = i32(plane_at(x + pi(DX), y + pi(DY), 0u));
    var v = u32(abs(a - b));
    if (pu(INVERT) != 0u) {
        v = 255u - v;
    }
    put_lane(gid.y * w + gid.x, v);
}

// Bevel relief: central differences of the height plane (clamped), the
// shading table in aux; MODE bits: 1 interior, 2 exterior, 4 pillow.
// Writes the whole bevel plane (lanes hi_in, sh_in, hi_out, sh_out).
@compute @workgroup_size(16, 16)
fn bevel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = pu(W_);
    let h = pu(H_);
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let gx = i32(src[cidx(x + 1, y, w, h)]) - i32(src[cidx(x - 1, y, w, h)]);
    let gy = i32(src[cidx(x, y + 1, w, h)]) - i32(src[cidx(x, y - 1, w, h)]);
    let m = pu(MODE);
    let e = aux[u32((gx + 255) * 511 + gy + 255)];
    var out = 0u;
    if ((m & 1u) != 0u) {
        out = e & 0xffffu;
    }
    if ((m & 2u) != 0u) {
        var o = e;
        if ((m & 4u) != 0u) {
            o = aux[u32((255 - gx) * 511 + 255 - gy)];
        }
        out = out | ((o & 0xffffu) << 16u);
    }
    dst[gid.y * w + gid.x] = out;
}

// ----- the stack ------------------------------------------------------------------------

fn alpha_of(p: u32) -> u32 {
    return p >> 24u;
}

fn scale_px(p: u32, f: u32) -> u32 {
    return pack4(mul255(unpack4(p), f));
}

fn over_px(s: u32, d: u32) -> u32 {
    let sa = alpha_of(s);
    if (sa == 255u) {
        return s;
    }
    if (sa == 0u) {
        return d;
    }
    return s + scale_px(d, 255u - sa);
}

// The float stack below reproduces the CPU's separately rounded f32
// arithmetic: Metal's fast math would fuse a product into a following add
// (FMA) and approximate divisions and square roots, so every product that
// feeds an add goes through `nf` (an OR with a uniform word that is always
// 0, which the compiler can't see through), divisions use `p_div` and
// square roots `p_sqrt`.
fn nf(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | pu(62u));
}

fn nf3(v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(nf(v.x), nf(v.y), nf(v.z));
}

// Correctly rounded sqrt (one Newton step on the hardware estimate).
fn p_sqrt(x: f32) -> f32 {
    if (x <= 0.0) {
        return 0.0;
    }
    let s = sqrt(x);
    return s + p_div(fma(-s, s, x), 2.0 * s);
}

// `px::to_f32`.
fn straight(p: u32) -> vec4<f32> {
    let c = unpack4(p);
    if (c.w == 0u) {
        return vec4<f32>(0.0);
    }
    let inv = p_div(1.0, f32(c.w));
    return vec4<f32>(nf(f32(c.x) * inv), nf(f32(c.y) * inv), nf(f32(c.z) * inv), p_div(f32(c.w), 255.0));
}

fn hard_light(cb: f32, cs: f32) -> f32 {
    if (cs <= 0.5) {
        return nf(nf(cb * 2.0) * cs);
    }
    let s = nf(2.0 * cs) - 1.0;
    return cb + s - nf(cb * s);
}

// `blend::separable` (BlendMode index order).
fn separable(mode: u32, cb: f32, cs: f32) -> f32 {
    switch mode {
        case 1u: { return min(cb, cs); }
        case 2u: { return nf(cb * cs); }
        case 3u: {
            if (cb >= 1.0) { return 1.0; }
            if (cs <= 0.0) { return 0.0; }
            return 1.0 - min(p_div(1.0 - cb, cs), 1.0);
        }
        case 4u: { return max(cb + cs - 1.0, 0.0); }
        case 5u: { return max(cb, cs); }
        case 6u: { return cb + cs - nf(cb * cs); }
        case 7u: {
            if (cb <= 0.0) { return 0.0; }
            if (cs >= 1.0) { return 1.0; }
            return min(p_div(cb, 1.0 - cs), 1.0);
        }
        case 8u: { return min(cb + cs, 1.0); }
        case 9u: { return hard_light(cs, cb); }
        case 10u: {
            if (cs <= 0.5) {
                return cb - nf(nf((1.0 - nf(2.0 * cs)) * cb) * (1.0 - cb));
            }
            var d: f32;
            if (cb <= 0.25) {
                d = nf(nf(nf(nf(16.0 * cb) - 12.0) * cb + 4.0) * cb);
            } else {
                d = p_sqrt(cb);
            }
            return cb + nf((nf(2.0 * cs) - 1.0) * (d - cb));
        }
        case 11u: { return hard_light(cb, cs); }
        case 12u: { return abs(cb - cs); }
        case 13u: { return cb + cs - nf(nf(2.0 * cb) * cs); }
        default: { return cs; }
    }
}

fn lum(c: vec3<f32>) -> f32 {
    return nf(0.3 * c.x) + nf(0.59 * c.y) + nf(0.11 * c.z);
}

fn clip_color(c: vec3<f32>) -> vec3<f32> {
    let l = lum(c);
    let n = min(min(c.x, c.y), c.z);
    let x = max(max(c.x, c.y), c.z);
    var o = c;
    if (n < 0.0) {
        let d = max(l - n, 1e-6);
        o = vec3<f32>(l + p_div(nf((o.x - l) * l), d), l + p_div(nf((o.y - l) * l), d), l + p_div(nf((o.z - l) * l), d));
    }
    if (x > 1.0) {
        let d = max(x - l, 1e-6);
        let k = 1.0 - l;
        o = vec3<f32>(l + p_div(nf((o.x - l) * k), d), l + p_div(nf((o.y - l) * k), d), l + p_div(nf((o.z - l) * k), d));
    }
    return o;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    let d = l - lum(c);
    return clip_color(c + vec3<f32>(d));
}

fn sat(c: vec3<f32>) -> f32 {
    return max(max(c.x, c.y), c.z) - min(min(c.x, c.y), c.z);
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let mx = max(max(c.x, c.y), c.z);
    let mn = min(min(c.x, c.y), c.z);
    if (mx <= mn) {
        return vec3<f32>(0.0);
    }
    return vec3<f32>(p_div(nf((c.x - mn) * s), mx - mn), p_div(nf((c.y - mn) * s), mx - mn), p_div(nf((c.z - mn) * s), mx - mn));
}

fn blend_color(mode: u32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 14u: { return set_lum(set_sat(cs, sat(cb)), lum(cb)); }
        case 15u: { return set_lum(set_sat(cb, sat(cs)), lum(cb)); }
        case 16u: { return set_lum(cs, lum(cb)); }
        case 17u: { return set_lum(cb, lum(cs)); }
        default: { return vec3<f32>(separable(mode, cb.x, cs.x), separable(mode, cb.y, cs.y), separable(mode, cb.z, cs.z)); }
    }
}

// `px::from_f32`.
fn from_f(c: vec3<f32>, a: f32) -> u32 {
    let a8 = u32(nf(clamp(a, 0.0, 1.0) * 255.0) + 0.5);
    if (a8 == 0u) {
        return 0u;
    }
    let af = f32(a8);
    let r = min(u32(nf(clamp(c.x, 0.0, 1.0) * af) + 0.5), a8);
    let g = min(u32(nf(clamp(c.y, 0.0, 1.0) * af) + 0.5), a8);
    let b = min(u32(nf(clamp(c.z, 0.0, 1.0) * af) + 0.5), a8);
    return r | (g << 8u) | (b << 16u) | (a8 << 24u);
}

// New PSD modes use identical integer arithmetic on CPU and GPU. Existing
// W3C float modes keep their established output. Indices are migration-stable.
fn straight_integer(pixel: u32) -> vec3<u32> {
    let a = pixel >> 24u;
    if (a == 0u) { return vec3<u32>(0u); }
    let rgb = vec3<u32>(pixel & 255u, (pixel >> 8u) & 255u, (pixel >> 16u) & 255u);
    return min((rgb * 255u + a / 2u) / a, vec3<u32>(255u));
}
fn dissolve_integer(d: u32, s: u32, op: u32, x: i32, y: i32) -> u32 {
    let coverage = ((s >> 24u) * min(op, 255u) + 127u) / 255u;
    if (coverage == 0u) { return d; }
    var noise = bitcast<u32>(x) * 0x1f123bb5u ^ bitcast<u32>(y) * 0x5f356495u ^ 0xa511e9b3u;
    noise = (noise ^ (noise >> 16u)) * 0x7feb352du;
    noise = (noise ^ (noise >> 15u)) * 0x846ca68bu;
    noise = noise ^ (noise >> 16u);
    if (coverage < 255u && noise % 255u >= coverage) { return d; }
    let color = straight_integer(s);
    return color.x | (color.y << 8u) | (color.z << 16u) | 0xff000000u;
}
fn integer_channel(mode: u32, base: u32, source: u32) -> u32 {
    switch mode {
        case 21u: {
            if (source < 128u) {
                if (base == 255u) { return 255u; }
                if (source == 0u) { return 0u; }
                return 255u - min(((255u - base) * 255u + source) / (2u * source), 255u);
            }
            if (base == 0u) { return 0u; }
            if (source == 255u) { return 255u; }
            return min((base * 255u + (255u - source)) / (2u * (255u - source)), 255u);
        }
        case 22u: { return u32(clamp(i32(base) + 2 * i32(source) - 255, 0, 255)); }
        case 23u: {
            if (source < 128u) { return min(base, 2u * source); }
            return max(base, 2u * source - 255u);
        }
        case 24u: { return select(0u, 255u, base + source >= 255u); }
        case 25u: { return max(base, source) - source; }
        case 26u: {
            if (source == 0u) { return 255u; }
            return min((base * 255u + source / 2u) / source, 255u);
        }
        default: { return source; }
    }
}
fn blend_integer(d: u32, s: u32, mode: u32) -> u32 {
    let sa = s >> 24u;
    let da = d >> 24u;
    if (sa == 0u) { return d; }
    if (da == 0u) { return s; }
    let cs = straight_integer(s);
    let cb = straight_integer(d);
    var color: vec3<u32>;
    if (mode == 19u) {
        color = select(cb, cs, cs.x + cs.y + cs.z < cb.x + cb.y + cb.z);
    } else if (mode == 20u) {
        color = select(cb, cs, cs.x + cs.y + cs.z > cb.x + cb.y + cb.z);
    } else {
        color = vec3<u32>(integer_channel(mode, cb.x, cs.x), integer_channel(mode, cb.y, cs.y), integer_channel(mode, cb.z, cs.z));
    }
    let sc = vec3<u32>(s & 255u, (s >> 8u) & 255u, (s >> 16u) & 255u);
    let dc = vec3<u32>(d & 255u, (d >> 8u) & 255u, (d >> 16u) & 255u);
    let a = (sa * 255u + da * 255u - sa * da + 127u) / 255u;
    let c = min((sc * (255u - da) * 255u + dc * (255u - sa) * 255u + color * sa * da + 32512u) / 65025u, vec3<u32>(a));
    return c.x | (c.y << 8u) | (c.z << 16u) | (a << 24u);
}

// `blend::blend_px`.
fn blend_px(db: u32, s: u32, mode: u32) -> u32 {
    let cs = straight(s);
    let cb = straight(db);
    let sa = cs.w;
    let da = cb.w;
    if (sa <= 0.0) {
        return db;
    }
    let b = blend_color(mode, cb.xyz, cs.xyz);
    let ao = sa + da - nf(sa * da);
    var co = vec3<f32>(0.0);
    if (ao > 0.0) {
        let pre = nf3(nf3(cs.xyz * sa) * (1.0 - da)) + nf3(nf3(cb.xyz * da) * (1.0 - sa)) + nf3(nf(sa * da) * b);
        co = vec3<f32>(p_div(pre.x, ao), p_div(pre.y, ao), p_div(pre.z, ao));
    }
    return from_f(co, ao);
}

// `styles::blend_one`.
fn blend_one(d: u32, s: u32, op: u32, mode: u32, x: i32, y: i32) -> u32 {
    if (mode == 18u) { return dissolve_integer(d, s, op, x, y); }
    if (op == 0u) {
        return d;
    }
    if (mode == 0u) {
        if (op == 255u) {
            return over_px(s, d);
        }
        if (s != 0u) {
            return over_px(scale_px(s, op), d);
        }
        return d;
    }
    if (s == 0u) {
        return d;
    }
    var s2 = s;
    if (op != 255u) {
        s2 = scale_px(s, op);
    }
    if (alpha_of(d) == 0u) {
        return s2;
    }
    if (mode >= 19u) { return blend_integer(d, s2, mode); }
    return blend_px(d, s2, mode);
}

fn unpre8(a: u32, v: u32) -> u32 {
    let i = a * 256u + v;
    return (unpre[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

const E_DS: u32 = 0u;
const E_OG: u32 = 1u;
const E_IS: u32 = 2u;
const E_IG: u32 = 3u;
const E_SAT: u32 = 4u;
const E_CO: u32 = 5u;
const E_STROKE: u32 = 6u;
const E_HI: u32 = 7u;
const E_SH: u32 = 8u;
const E_GO: u32 = 9u;

fn has(bit: u32) -> bool {
    return (pu(FLAGS) & (1u << bit)) != 0u;
}

fn grad_at(x: i32, y: i32) -> u32 {
    let px = f32(x) + 0.5 - pf(53u);
    let py = f32(y) + 0.5 - pf(54u);
    var t: f32;
    if (pu(58u) != 0u) {
        t = nf(p_sqrt(nf(px * px) + nf(py * py)) * pf(57u));
    } else {
        t = nf((nf(px * pf(55u)) + nf(py * pf(56u))) * pf(57u)) + 0.5;
    }
    if (pu(59u) != 0u) {
        t = 1.0 - t;
    }
    let i = min(u32(nf(clamp(t, 0.0, 1.0) * 1023.0) + 0.5), 1023u);
    return aux[i];
}

// `styles::compose_px` for the center (dst is cw x ch).
@compute @workgroup_size(16, 16)
fn compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cw = pu(CW);
    if (gid.x >= cw || gid.y >= pu(CH)) {
        return;
    }
    let pad = pu(PAD);
    let bx = gid.x + pad;
    let by = gid.y + pad;
    let i = by * pu(W_) + bx;
    let c = src[i];
    let a = c >> 24u;
    let ia = 255u - a;
    var l0 = 0u;
    var l1 = 0u;
    var l2 = 0u;
    if (has(16u)) {
        l0 = lane0[i];
    }
    if (has(17u)) {
        l1 = lane1[i];
    }
    if (has(18u)) {
        l2 = lane2[i];
    }
    let inner = a != 0u && has(14u);
    let outer = ia != 0u && has(15u);
    // Three stacks: beneath (b), the interior in shape space (n), the
    // outside-only effects (o). One blend call site in a loop whose bound
    // is a uniform (word 63 = 13), so the float blend code isn't inlined
    // once per effect.
    var b = 0u;
    var n = 0u;
    var o = 0u;
    if (inner) {
        let cc = unpack4(c);
        n = premul(vec3<u32>(unpre8(a, cc.x), unpre8(a, cc.y), unpre8(a, cc.z)), pu(FILL8));
    }
    let steps = pu(63u);
    for (var k = 0u; k < steps; k++) {
        var on = false;
        var tgt = 1u;
        var e = 0u;
        var cov = 0u;
        switch k {
            case 0u: {
                on = has(0u);
                tgt = 0u;
                e = E_DS;
                cov = l0 & 255u;
                if (has(1u)) {
                    cov = mul1(cov, ia);
                }
            }
            case 1u: {
                on = has(2u);
                tgt = 0u;
                e = E_OG;
                cov = mul1((l0 >> 8u) & 255u, ia);
            }
            case 2u: {
                on = inner && has(7u);
                e = E_GO;
            }
            case 3u: {
                on = inner && has(6u);
                e = E_CO;
                cov = 255u;
            }
            case 4u: {
                on = inner && has(5u);
                e = E_SAT;
                cov = l1 & 255u;
            }
            case 5u: {
                on = inner && has(4u);
                e = E_IG;
                cov = (l0 >> 24u) & 255u;
            }
            case 6u: {
                on = inner && has(3u);
                e = E_IS;
                cov = (l0 >> 16u) & 255u;
            }
            case 7u: {
                on = inner && has(10u);
                e = E_STROKE;
                let ed = max(a + ((l1 >> 16u) & 255u), 255u) - 255u;
                cov = (ed * 255u + a / 2u) / max(a, 1u);
            }
            case 8u: {
                on = inner && has(12u);
                e = E_HI;
                cov = l2 & 255u;
            }
            case 9u: {
                on = inner && has(12u);
                e = E_SH;
                cov = (l2 >> 8u) & 255u;
            }
            case 10u: {
                on = outer && has(9u);
                tgt = 2u;
                e = E_STROKE;
                cov = (l1 >> 8u) & 255u;
            }
            case 11u: {
                on = outer && has(13u);
                tgt = 2u;
                e = E_HI;
                cov = (l2 >> 16u) & 255u;
            }
            default: {
                on = outer && has(13u);
                tgt = 2u;
                e = E_SH;
                cov = l2 >> 24u;
            }
        }
        if (!on) {
            continue;
        }
        let base = 24u + 3u * e;
        var s: u32;
        if (e == E_GO) {
            s = grad_at(pi(OX) + i32(bx), pi(OY) + i32(by));
        } else {
            s = premul(unpack4(pu(base + 2u)).xyz, cov);
        }
        var d = b;
        if (tgt == 1u) {
            d = n;
        } else if (tgt == 2u) {
            d = o;
        }
        let r = blend_one(d, s, pu(base + 1u), pu(base), pi(OX) + i32(bx), pi(OY) + i32(by));
        if (tgt == 0u) {
            b = r;
        } else if (tgt == 1u) {
            n = r;
        } else {
            o = r;
        }
    }
    var ip = 0u;
    if (inner) {
        ip = scale_px(n, a);
    } else if (a != 0u) {
        if (pu(FILL8) == 255u) {
            ip = c;
        } else {
            ip = scale_px(c, pu(FILL8));
        }
    }
    var rp = 0u;
    if (outer) {
        rp = scale_px(o, ia);
    }
    dst[gid.y * cw + gid.x] = over_px(ip + rp, b);
}
