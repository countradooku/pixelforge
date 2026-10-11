// The GPU compositor (see `gpu::compose`): layers are read from the tile
// pool through per-region tile tables, blended with `blend::blend_row`'s
// exact semantics (integer `over` / `scale` for Normal, the W3C formulas in
// float for the other modes), and turned into display tiles.

struct P {
    // Region size and the number of layer descriptors (stack kernel).
    w: u32,
    h: u32,
    n: u32,
    op: u32,
    mode: u32,
    // Generic offsets / flags per entry point.
    a: u32,
    b: i32,
    c: i32,
    d: u32,
    e: u32,
    f: u32,
    g: u32,
}

@group(0) @binding(0) var<uniform> p: P;
// The tile pool: 256² u32 slots (mask slots hold bytes).
@group(0) @binding(1) var<storage, read> pool: array<u32>;
// Tile tables and layer descriptors.
@group(0) @binding(2) var<storage, read> tabs: array<u32>;
@group(0) @binding(3) var<storage, read_write> acc: array<u32>;
@group(0) @binding(4) var<storage, read> aux: array<u32>;
@group(0) @binding(5) var<storage, read> aux2: array<u32>;

const EMPTY: u32 = 0xffffffffu;
const UNIFORM: u32 = 0xfffffffeu;
const NO_MASK: u32 = 0xffffffffu;

// A pixel of the grid whose table starts at `off`, at level coordinates
// (x, y) (`read_region` semantics).
fn fetch(off: u32, x: i32, y: i32) -> u32 {
    let tx = (x >> 8u) - bitcast<i32>(tabs[off]);
    let ty = (y >> 8u) - bitcast<i32>(tabs[off + 1u]);
    let tw = tabs[off + 2u];
    if (tx < 0 || ty < 0 || u32(tx) >= tw || u32(ty) >= tabs[off + 3u]) {
        return 0u;
    }
    let e = off + 4u + (u32(ty) * tw + u32(tx)) * 2u;
    let code = tabs[e];
    if (code == EMPTY) {
        return 0u;
    }
    if (code == UNIFORM) {
        return tabs[e + 1u];
    }
    return pool[code * 65536u + u32(y & 255) * 256u + u32(x & 255)];
}

// Coverage of the mask whose table starts at `off` (size mw x mh), with
// coordinates clamped to the mask (`MaskGrid::read_region(.., clamp)`).
fn fetch_mask(off: u32, xin: i32, yin: i32, mw: u32, mh: u32) -> u32 {
    let x = clamp(xin, 0, i32(mw) - 1);
    let y = clamp(yin, 0, i32(mh) - 1);
    let tx = (x >> 8u) - bitcast<i32>(tabs[off]);
    let ty = (y >> 8u) - bitcast<i32>(tabs[off + 1u]);
    let tw = tabs[off + 2u];
    if (tx < 0 || ty < 0 || u32(tx) >= tw || u32(ty) >= tabs[off + 3u]) {
        return 0u;
    }
    let e = off + 4u + (u32(ty) * tw + u32(tx)) * 2u;
    let code = tabs[e];
    if (code == EMPTY) {
        return 0u;
    }
    if (code == UNIFORM) {
        return tabs[e + 1u];
    }
    let i = u32(y & 255) * 256u + u32(x & 255);
    return (pool[code * 65536u + (i >> 2u)] >> ((i & 3u) * 8u)) & 255u;
}

fn scale(px: u32, f: u32) -> u32 {
    return pack4(mul255(unpack4(px), f));
}

// `px::over`.
fn over(s: u32, d: u32) -> u32 {
    let sa = s >> 24u;
    if (sa == 255u) {
        return s;
    }
    if (sa == 0u) {
        return d;
    }
    return s + scale(d, 255u - sa);
}

// `apply_coverage`.
fn cover(px: u32, c: u32) -> u32 {
    if (c == 255u) {
        return px;
    }
    if (c == 0u) {
        return 0u;
    }
    return scale(px, c);
}

// ----- blend modes (`blend::blend_px`) ----------------------------------------

fn hard_light(cb: f32, cs: f32) -> f32 {
    if (cs <= 0.5) {
        return cb * 2.0 * cs;
    }
    let s = 2.0 * cs - 1.0;
    return cb + s - cb * s;
}

fn separable(mode: u32, cb: f32, cs: f32) -> f32 {
    switch mode {
        case 2u: { return cb * cs; }
        case 6u: { return cb + cs - cb * cs; }
        case 9u: { return hard_light(cs, cb); }
        case 1u: { return min(cb, cs); }
        case 5u: { return max(cb, cs); }
        case 7u: {
            if (cb <= 0.0) { return 0.0; }
            if (cs >= 1.0) { return 1.0; }
            return min(p_div(cb, 1.0 - cs), 1.0);
        }
        case 3u: {
            if (cb >= 1.0) { return 1.0; }
            if (cs <= 0.0) { return 0.0; }
            return 1.0 - min(p_div(1.0 - cb, cs), 1.0);
        }
        case 11u: { return hard_light(cb, cs); }
        case 10u: {
            if (cs <= 0.5) {
                return cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb);
            }
            var d: f32;
            if (cb <= 0.25) {
                d = ((16.0 * cb - 12.0) * cb + 4.0) * cb;
            } else {
                d = sqrt(cb);
            }
            return cb + (2.0 * cs - 1.0) * (d - cb);
        }
        case 12u: { return abs(cb - cs); }
        case 13u: { return cb + cs - 2.0 * cb * cs; }
        case 8u: { return min(cb + cs, 1.0); }
        case 4u: { return max(cb + cs - 1.0, 0.0); }
        default: { return cs; }
    }
}

fn blum(c: vec3<f32>) -> f32 {
    return 0.3 * c.x + 0.59 * c.y + 0.11 * c.z;
}

fn clip_color(c: vec3<f32>) -> vec3<f32> {
    let l = blum(c);
    let n = min(min(c.x, c.y), c.z);
    let x = max(max(c.x, c.y), c.z);
    var o = c;
    if (n < 0.0) {
        let d = l - n;
        o = l + (o - l) * l / max(d, 1e-6);
    }
    if (x > 1.0) {
        let d = x - l;
        o = l + (o - l) * (1.0 - l) / max(d, 1e-6);
    }
    return o;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    let d = l - blum(c);
    return clip_color(c + d);
}

fn bsat(c: vec3<f32>) -> f32 {
    return max(max(c.x, c.y), c.z) - min(min(c.x, c.y), c.z);
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let mx = max(max(c.x, c.y), c.z);
    let mn = min(min(c.x, c.y), c.z);
    if (mx <= mn) {
        return vec3<f32>(0.0);
    }
    return (c - mn) * s / (mx - mn);
}

fn blend_color(mode: u32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
    switch mode {
        case 14u: { return set_lum(set_sat(cs, bsat(cb)), blum(cb)); }
        case 15u: { return set_lum(set_sat(cb, bsat(cs)), blum(cb)); }
        case 16u: { return set_lum(cs, blum(cb)); }
        case 17u: { return set_lum(cb, blum(cs)); }
        default: {
            return vec3<f32>(separable(mode, cb.x, cs.x), separable(mode, cb.y, cs.y), separable(mode, cb.z, cs.z));
        }
    }
}

// `px::to_f32` color part (alpha handled by the caller).
fn straight_f(c: vec4<u32>) -> vec3<f32> {
    let inv = 1.0 / f32(c.w);
    return vec3<f32>(f32(c.x) * inv, f32(c.y) * inv, f32(c.z) * inv);
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

// `blend_px` (float W3C compositing, both pixels non-transparent).
fn blend_float(db: u32, s: u32, mode: u32) -> u32 {
    let sc = unpack4(s);
    let dc = unpack4(db);
    let cs = straight_f(sc);
    let sa = f32(sc.w) / 255.0;
    let cb = straight_f(dc);
    let da = f32(dc.w) / 255.0;
    let b = blend_color(mode, cb, cs);
    let ao = sa + da - sa * da;
    let pre = cs * sa * (1.0 - da) + cb * da * (1.0 - sa) + sa * da * b;
    var co = vec3<f32>(0.0);
    if (ao > 0.0) {
        co = pre / ao;
    }
    // `px::from_f32`.
    let a8 = u32(clamp(ao, 0.0, 1.0) * 255.0 + 0.5);
    if (a8 == 0u) {
        return 0u;
    }
    let ch = min(vec3<u32>(clamp(co, vec3<f32>(0.0), vec3<f32>(1.0)) * f32(a8) + 0.5), vec3<u32>(a8));
    return pack4(vec4<u32>(ch, a8));
}

// One pixel of `blend_row`.
fn blend(d: u32, s: u32, op: u32, mode: u32, x: i32, y: i32) -> u32 {
    if (mode == 18u) { return dissolve_integer(d, s, op, x, y); }
    if (op == 0u) {
        return d;
    }
    if (mode == 0u) {
        if (op == 255u) {
            return over(s, d);
        }
        if (s != 0u) {
            return over(scale(s, op), d);
        }
        return d;
    }
    if (s == 0u) {
        return d;
    }
    var ss = s;
    if (op != 255u) {
        ss = scale(s, op);
    }
    if ((d >> 24u) == 0u) {
        return ss;
    }
    if ((ss >> 24u) == 0u) {
        return d;
    }
    if (mode >= 19u) { return blend_integer(d, ss, mode); }
    return blend_float(d, ss, mode);
}

// ----- kernels ------------------------------------------------------------------

// A plain stack in one pass: per pixel, every layer in order. Descriptor k
// (10 words at p.a + 10 k): table offset, layer-local x and y of the region
// origin, opacity, mode, mask table offset (or NO_MASK), mask-local x / y of
// the origin, mask size.
@compute @workgroup_size(16, 16)
fn stack(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    // p.d: start from `aux` (stride p.e, offset (p.f, p.g)) instead of 0:
    // the cached composite of the layers below (foreground caching).
    var a = 0u;
    if (p.d != 0u) {
        a = aux[(gid.y + p.g) * p.e + gid.x + p.f];
    }
    for (var k = 0u; k < p.n; k++) {
        let o = p.a + k * 10u;
        let lx = bitcast<i32>(tabs[o + 1u]) + i32(gid.x);
        let ly = bitcast<i32>(tabs[o + 2u]) + i32(gid.y);
        var s = fetch(tabs[o], lx, ly);
        let mo = tabs[o + 5u];
        if (mo != NO_MASK) {
            s = cover(s, fetch_mask(mo, bitcast<i32>(tabs[o + 6u]) + i32(gid.x), bitcast<i32>(tabs[o + 7u]) + i32(gid.y), tabs[o + 8u], tabs[o + 9u]));
        }
        a = blend(a, s, tabs[o + 3u], tabs[o + 4u], p.b + i32(gid.x), p.c + i32(gid.y));
    }
    acc[gid.y * p.w + gid.x] = a;
}

// acc = layer pixels (descriptor at p.a, as in `stack`), masked.
@compute @workgroup_size(16, 16)
fn load(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let o = p.a;
    var s = fetch(tabs[o], bitcast<i32>(tabs[o + 1u]) + i32(gid.x), bitcast<i32>(tabs[o + 2u]) + i32(gid.y));
    let mo = tabs[o + 5u];
    if (mo != NO_MASK) {
        s = cover(s, fetch_mask(mo, bitcast<i32>(tabs[o + 6u]) + i32(gid.x), bitcast<i32>(tabs[o + 7u]) + i32(gid.y), tabs[o + 8u], tabs[o + 9u]));
    }
    acc[gid.y * p.w + gid.x] = s;
}

// acc = blend(acc, aux, p.op, p.mode).
@compute @workgroup_size(16, 16)
fn blend_buf(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    acc[i] = blend(acc[i], aux[i], p.op, p.mode, p.b + i32(gid.x), p.c + i32(gid.y));
}

// acc = aux (p.op = 255) or aux scaled by the opacity.
@compute @workgroup_size(16, 16)
fn copy_scaled(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    if (p.op == 255u) {
        acc[i] = aux[i];
    } else {
        acc[i] = scale(aux[i], p.op);
    }
}

@compute @workgroup_size(16, 16)
fn zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    acc[gid.y * p.w + gid.x] = 0u;
}

// Clipping: acc = clipped layer (aux) scaled by the base alpha (aux2).
@compute @workgroup_size(16, 16)
fn by_alpha(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    acc[i] = scale(aux[i], aux2[i] >> 24u);
}

// Clipping: limits acc's alpha to the base alpha (aux2), as `compose_list`.
@compute @workgroup_size(16, 16)
fn clamp_alpha(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    let px = acc[i];
    let a = aux2[i] >> 24u;
    let pa = px >> 24u;
    if (pa > a) {
        if (a == 0u) {
            acc[i] = 0u;
        } else {
            acc[i] = scale(px, (a * 255u + pa / 2u) / pa);
        }
    }
}

// Adjustment layer mix: acc = lerp(acc, aux, op), or with mask coverage
// (table at p.a, origin p.b / p.c, size p.e x p.f) (m * op + 127) / 255.
@compute @workgroup_size(16, 16)
fn mix(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    var t = p.op;
    if (p.a != NO_MASK) {
        let m = fetch_mask(p.a, p.b + i32(gid.x), p.c + i32(gid.y), p.e, p.f);
        t = (m * p.op + 127u) / 255u;
    }
    acc[i] = lerp4(acc[i], aux[i], t);
}

// Preview selection: acc = lerp(aux (original), acc (filtered), coverage
// bytes in aux2).
@compute @workgroup_size(16, 16)
fn select_mix(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    let m = (aux2[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
    acc[i] = lerp4(aux[i], acc[i], m);
}

// Display tiles: `display_tile`'s checkerboard composite as BGRA, DT x DT
// per tile with its 1 px gutter. Tile list (tx, ty) in tabs[p.a..]; the
// composed region (p.w x p.h) starts at (p.b, p.c) of the compose level;
// p.d = magnification shift for negative display levels; output row stride
// p.e and tile stride p.f (in pixels; padded for texture copies).
const DT: u32 = 258u;

@compute @workgroup_size(16, 16)
fn display(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= DT || gid.y >= DT) {
        return;
    }
    let t = gid.z;
    let tx = bitcast<i32>(tabs[p.a + 2u * t]);
    let ty = bitcast<i32>(tabs[p.a + 2u * t + 1u]);
    let gx = tx * 256 - 1 + i32(gid.x);
    let gy = ty * 256 - 1 + i32(gid.y);
    let sx = (gx >> p.d) - p.b;
    let sy = (gy >> p.d) - p.c;
    var v = 0u;
    if (sx >= 0 && sy >= 0 && u32(sx) < p.w && u32(sy) < p.h) {
        v = aux[u32(sy) * p.w + u32(sx)];
    }
    var check = 0xffccccccu;
    if ((((gx >> 3u) ^ (gy >> 3u)) & 1) == 0) {
        check = 0xffffffffu;
    }
    let c = unpack4(over(v, check));
    acc[t * p.f + gid.y * p.e + gid.x] = c.z | (c.y << 8u) | (c.x << 16u) | (255u << 24u);
}

// ----- fused adjustment chains -----------------------------------------------------
// Consecutive LUT-type adjustment layers in one pass: per pixel, each
// adjustment's point op (exactly as `point.wgsl` / `Filter::apply`) mixed
// by opacity x mask, in stack order. Tables in `luts`: the unpremultiply
// table at word 0, then each op's table at its offset. Chain in
// tabs[p.a..]: count, then 8 words per op: kind (0 lut3, 1 lut3d,
// 2 threshold, 3 desaturate), table word offset, opacity, mask table offset
// (or NO_MASK), mask origin x / y, mask size.

@group(0) @binding(6) var<storage, read> luts: array<u32>;

fn lbyte(i: u32) -> u32 {
    return (luts[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

fn l_straight(c: vec4<u32>) -> vec3<u32> {
    if (c.w == 255u) {
        return c.xyz;
    }
    let b = c.w * 256u;
    return vec3<u32>(lbyte(b + c.x), lbyte(b + c.y), lbyte(b + c.z));
}

fn l3d_at(off: u32, r: u32, g: u32, b: u32) -> vec3<u32> {
    let e = off + (r + g * 33u + b * 1089u) * 2u;
    let rg = luts[e];
    return vec3<u32>(rg & 65535u, rg >> 16u, luts[e + 1u] & 65535u);
}

fn l_split33(v: u32) -> vec2<u32> {
    let f = v * 32u * 256u / 255u;
    let i = min(f >> 8u, 31u);
    return vec2<u32>(i, f - (i << 8u));
}

fn l3d_eval(off: u32, c: vec3<u32>) -> vec3<u32> {
    let rr = l_split33(c.x);
    let gg = l_split33(c.y);
    let bb = l_split33(c.z);
    let r0 = rr.x;
    let g0 = gg.x;
    let b0 = bb.x;
    let fr = i32(rr.y);
    let fg = i32(gg.y);
    let fb = i32(bb.y);
    let c000 = vec3<i32>(l3d_at(off, r0, g0, b0));
    let c111 = vec3<i32>(l3d_at(off, r0 + 1u, g0 + 1u, b0 + 1u));
    var w: vec4<i32>;
    var p1: vec3<u32>;
    var p2: vec3<u32>;
    if (fr >= fg) {
        if (fg >= fb) {
            w = vec4<i32>(256 - fr, fr - fg, fg - fb, fb);
            p1 = l3d_at(off, r0 + 1u, g0, b0);
            p2 = l3d_at(off, r0 + 1u, g0 + 1u, b0);
        } else if (fr >= fb) {
            w = vec4<i32>(256 - fr, fr - fb, fb - fg, fg);
            p1 = l3d_at(off, r0 + 1u, g0, b0);
            p2 = l3d_at(off, r0 + 1u, g0, b0 + 1u);
        } else {
            w = vec4<i32>(256 - fb, fb - fr, fr - fg, fg);
            p1 = l3d_at(off, r0, g0, b0 + 1u);
            p2 = l3d_at(off, r0 + 1u, g0, b0 + 1u);
        }
    } else if (fb >= fg) {
        w = vec4<i32>(256 - fb, fb - fg, fg - fr, fr);
        p1 = l3d_at(off, r0, g0, b0 + 1u);
        p2 = l3d_at(off, r0, g0 + 1u, b0 + 1u);
    } else if (fb >= fr) {
        w = vec4<i32>(256 - fg, fg - fb, fb - fr, fr);
        p1 = l3d_at(off, r0, g0 + 1u, b0);
        p2 = l3d_at(off, r0, g0 + 1u, b0 + 1u);
    } else {
        w = vec4<i32>(256 - fg, fg - fr, fr - fb, fb);
        p1 = l3d_at(off, r0, g0 + 1u, b0);
        p2 = l3d_at(off, r0 + 1u, g0 + 1u, b0);
    }
    let v = c000 * w.x + vec3<i32>(p1) * w.y + vec3<i32>(p2) * w.z + c111 * w.w;
    let o = (((v + vec3<i32>(128)) >> vec3<u32>(8u)) * 255) / 65535;
    return vec3<u32>(clamp(o, vec3<i32>(0), vec3<i32>(255)));
}

fn point_op(kind: u32, off: u32, v: u32) -> u32 {
    let c = unpack4(v);
    if (c.w == 0u) {
        return v;
    }
    let s = l_straight(c);
    var o: vec3<u32>;
    switch kind {
        case 0u: {
            let b = off * 4u;
            o = vec3<u32>(lbyte(b + s.x), lbyte(b + 256u + s.y), lbyte(b + 512u + s.z));
        }
        case 1u: {
            o = l3d_eval(off, s);
        }
        case 2u: {
            let k = s.x * 256u + s.y;
            let cut = (luts[off + (k >> 1u)] >> ((k & 1u) * 16u)) & 65535u;
            o = vec3<u32>(select(0u, 255u, s.z >= cut));
        }
        default: {
            o = vec3<u32>((s.x * 77u + s.y * 150u + s.z * 29u + 128u) >> 8u);
        }
    }
    return premul(o, c.w);
}

@compute @workgroup_size(16, 16)
fn adjust_chain(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    var a = acc[i];
    let n = tabs[p.a];
    for (var k = 0u; k < n; k++) {
        let o = p.a + 1u + k * 8u;
        let t = point_op(tabs[o], tabs[o + 1u], a);
        var f = tabs[o + 2u];
        let mo = tabs[o + 3u];
        if (mo != NO_MASK) {
            let m = fetch_mask(mo, bitcast<i32>(tabs[o + 4u]) + i32(gid.x), bitcast<i32>(tabs[o + 5u]) + i32(gid.y), tabs[o + 6u], tabs[o + 7u]);
            f = (m * f + 127u) / 255u;
        }
        a = lerp4(a, t, f);
    }
    acc[i] = a;
}
