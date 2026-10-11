// Image Size, Super Zoom and transform resamplers (`gpu::resample`). Each
// entry mirrors its CPU twin: integer passes over tables built on the CPU
// (bit-exact), or the CPU's float formulas in the same order.
//
// Output is a band of the destination tile area: rows y0..y0 + rows of a
// `tw`-wide buffer, written from row 0.

@group(0) @binding(0) var<uniform> pp: array<vec4<u32>, 6>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> ta: array<u32>;
@group(0) @binding(4) var<storage, read> tb: array<u32>;
@group(0) @binding(5) var<storage, read> src2: array<u32>;

const PI: f32 = 3.14159265358979;

fn pu(i: u32) -> u32 {
    return pp[i >> 2u][i & 3u];
}

fn pi(i: u32) -> i32 {
    return bitcast<i32>(pu(i));
}

fn pf(i: u32) -> f32 {
    return bitcast<f32>(pu(i));
}

// The value of x, opaque to the compiler (word 23 is always 0): a product
// feeding an addition rounds on its own, as on the CPU, instead of fusing
// into an FMA or being reassociated by fast math.
fn nf(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | pu(23u));
}

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

fn nf4(x: vec4<f32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(x) | vec4<u32>(pu(23u)));
}

// ----- Image Size: bilinear (`ops::resize_bilinear_cpu`) ---------------------
// params: stride, sw, sh, tw, y0, rows. ta / tb: (source index, 8-bit
// fraction) per output column / row.

@compute @workgroup_size(16, 16)
fn bilin(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = pu(3u);
    if (gid.x >= tw || gid.y >= pu(5u)) {
        return;
    }
    let y = pu(4u) + gid.y;
    let stride = pu(0u);
    let sw = pi(1u);
    let sh = pi(2u);
    let cx = bitcast<i32>(ta[2u * gid.x]);
    let wx = ta[2u * gid.x + 1u];
    let cy = bitcast<i32>(tb[2u * y]);
    let wy = tb[2u * y + 1u];
    let x0 = u32(clamp(cx, 0, sw - 1));
    let x1 = u32(clamp(cx + 1, 0, sw - 1));
    let r0 = u32(clamp(cy, 0, sh - 1)) * stride;
    let r1 = u32(clamp(cy + 1, 0, sh - 1)) * stride;
    let top = lerp4(src[r0 + x0], src[r0 + x1], wx);
    let bot = lerp4(src[r1 + x0], src[r1 + x1], wx);
    dst[gid.y * tw + gid.x] = lerp4(top, bot, wy);
}

// ----- Image Size: nearest neighbour ------------------------------------------
// params: stride, -, -, tw, y0, rows. ta / tb: source column / row.

@compute @workgroup_size(16, 16)
fn nearest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = pu(3u);
    if (gid.x >= tw || gid.y >= pu(5u)) {
        return;
    }
    let y = pu(4u) + gid.y;
    dst[gid.y * tw + gid.x] = src[tb[y] * pu(0u) + ta[gid.x]];
}

// ----- Image Size: separable fixed point (`resample::separable`) --------------

// Horizontal pass. params: stride, sw, n, tw, r0, nr. Writes source rows
// r0..r0 + nr x tw output columns as 4 x i16 (6 fractional bits) in 2 words.
// ta: first tap per column (i32), tb: n weights per column (i32, 14 bits).
@compute @workgroup_size(64, 4)
fn sep_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = pu(3u);
    if (gid.x >= tw || gid.y >= pu(5u)) {
        return;
    }
    let n = pu(2u);
    let sw = pi(1u);
    let row = (pu(4u) + gid.y) * pu(0u);
    let s0 = bitcast<i32>(ta[gid.x]);
    let wb = gid.x * n;
    var acc = vec4<i32>(0);
    for (var k = 0u; k < n; k++) {
        let wk = bitcast<i32>(tb[wb + k]);
        if (wk == 0) {
            continue;
        }
        let c = u32(clamp(s0 + i32(k), 0, sw - 1));
        acc += wk * vec4<i32>(unpack4(src[row + c]));
    }
    let h = clamp((acc + vec4<i32>(128)) >> vec4<u32>(8u), vec4<i32>(-32768), vec4<i32>(32767));
    let o = (gid.y * tw + gid.x) * 2u;
    dst[o] = (bitcast<u32>(h.x) & 0xffffu) | (bitcast<u32>(h.y) << 16u);
    dst[o + 1u] = (bitcast<u32>(h.z) & 0xffffu) | (bitcast<u32>(h.w) << 16u);
}

// `resample::finish`: 20 fractional bits to a premultiplied pixel.
fn finish(acc: vec4<i32>) -> u32 {
    let r = (acc + vec4<i32>(1 << 19u)) >> vec4<u32>(20u);
    let a = clamp(r.w, 0, 255);
    let c = clamp(r.xyz, vec3<i32>(0), vec3<i32>(a));
    return pack4(vec4<u32>(vec3<u32>(c), u32(a)));
}

// Vertical pass. params: -, sh, n, tw, r0 (first row of the intermediate),
// rows, y0 (first output row), d0 (dst row of y0). src: the intermediate.
@compute @workgroup_size(64, 4)
fn sep_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = pu(3u);
    if (gid.x >= tw || gid.y >= pu(5u)) {
        return;
    }
    let y = pu(6u) + gid.y;
    let n = pu(2u);
    let sh = pi(1u);
    let r0 = pi(4u);
    let s0 = bitcast<i32>(ta[y]);
    let wb = y * n;
    var acc = vec4<i32>(0);
    for (var k = 0u; k < n; k++) {
        let wk = bitcast<i32>(tb[wb + k]);
        if (wk == 0) {
            continue;
        }
        let r = u32(clamp(s0 + i32(k), 0, sh - 1) - r0);
        let o = (r * tw + gid.x) * 2u;
        let lo = src[o];
        let hi = src[o + 1u];
        let h = vec4<i32>(bitcast<i32>(lo << 16u) >> 16u, bitcast<i32>(lo) >> 16u, bitcast<i32>(hi << 16u) >> 16u, bitcast<i32>(hi) >> 16u);
        acc += wk * h;
    }
    dst[(gid.y + pu(7u)) * tw + gid.x] = finish(acc);
}

// ----- Image Size: Preserve Details (`resample::pd_pixel`) --------------------
// params: nw, nh, tw, ry0 (first row of L / S), y0, rows, v2.
// src: L, src2: S, ta: gain per edge strength (8.8).

fn ch(v: u32) -> vec4<i32> {
    return vec4<i32>(unpack4(v));
}

fn luma(v: u32) -> i32 {
    let c = ch(v);
    return (77 * c.x + 150 * c.y + 29 * c.z + 128) >> 8u;
}

fn nb(v: u32, d: i32, n: u32) -> u32 {
    if (d == 0) {
        return v;
    }
    return u32(clamp(i32(v) + d, 0, i32(n) - 1));
}

@compute @workgroup_size(16, 16)
fn pd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = pu(2u);
    if (gid.x >= tw || gid.y >= pu(5u)) {
        return;
    }
    let nw = pu(0u);
    let nh = pu(1u);
    let ry0 = pu(3u);
    let x = gid.x;
    let y = pu(4u) + gid.y;
    let sxp = src2[(y - ry0) * tw + nb(x, 1, nw)];
    let sxm = src2[(y - ry0) * tw + nb(x, -1, nw)];
    let syp = src2[(nb(y, 1, nh) - ry0) * tw + x];
    let sym = src2[(nb(y, -1, nh) - ry0) * tw + x];
    let e = abs(luma(sxp) - luma(sxm)) + abs(luma(syp) - luma(sym));
    let gn = bitcast<i32>(ta[u32(min(e, 255))]);
    let lc = ch(src[(y - ry0) * tw + x]);
    let sc = ch(src2[(y - ry0) * tw + x]);
    var o = sc + (((lc - sc) * gn + vec4<i32>(128)) >> vec4<u32>(8u));
    if (pu(6u) != 0u) {
        var mn = vec4<i32>(255);
        var mx = vec4<i32>(0);
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let c = ch(src[(nb(y, dy, nh) - ry0) * tw + nb(x, dx, nw)]);
                mn = min(mn, c);
                mx = max(mx, c);
            }
        }
        o = clamp(o, mn, mx);
    }
    let a = clamp(o.w, 0, 255);
    let c = clamp(o.xyz, vec3<i32>(0), vec3<i32>(a));
    dst[gid.y * tw + x] = pack4(vec4<u32>(vec3<u32>(c), u32(a)));
}

// ----- Super Zoom: Catmull-Rom (`ops::upscale_bicubic_cpu`) -------------------
// params: stride, -, -, tw, y0, rows. ta / tb: per column / row 4 source
// indices then 4 f32 weights.

@compute @workgroup_size(16, 16)
fn cubic4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = pu(3u);
    if (gid.x >= tw || gid.y >= pu(5u)) {
        return;
    }
    let y = pu(4u) + gid.y;
    let stride = pu(0u);
    let cb = gid.x * 8u;
    let rb = y * 8u;
    var acc = vec4<f32>(0.0);
    for (var j = 0u; j < 4u; j++) {
        let row = tb[rb + j] * stride;
        let wyj = bitcast<f32>(tb[rb + 4u + j]);
        for (var i = 0u; i < 4u; i++) {
            let p = vec4<f32>(unpack4(src[row + ta[cb + i]]));
            let wgt = nf(bitcast<f32>(ta[cb + 4u + i]) * wyj);
            acc = nf4(acc + nf4(p * wgt));
        }
    }
    let a = clamp(floor(acc.w + 0.5), 0.0, 255.0);
    let c = clamp(floor(acc.xyz + vec3<f32>(0.5)), vec3<f32>(0.0), vec3<f32>(a));
    dst[gid.y * tw + gid.x] = pack4(vec4<u32>(vec3<u32>(c), u32(a)));
}

// ----- transforms: affine bilinear / Catmull-Rom / Lanczos-3 (`crop::warp_affine_cpu`)
// params: inv[0..6], stride, sw, sh, kind (0 bilinear, 1 bicubic, 2 Lanczos),
// tw, y0, rows, cols. ta: per output tile (footprint x0, y0, w, h), w = 0
// when the tile is empty.

fn lanczos3(xin: f32) -> f32 {
    let x = abs(xin);
    if (x < 1e-5) {
        return 1.0;
    }
    if (x >= 3.0) {
        return 0.0;
    }
    let p = PI * x;
    return div_ieee(3.0 * p_sin(p) * p_sin(div_ieee(p, 3.0)), p * p);
}

fn weights(kind: u32, f: f32) -> array<f32, 6> {
    var o = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    if (kind == 0u) {
        o[0] = 1.0 - f;
        o[1] = f;
    } else if (kind == 1u) {
        let t2 = nf(f * f);
        let t3 = nf(t2 * f);
        o[0] = nf(nf(-t3 + 2.0 * t2) - f) * 0.5;
        o[1] = nf(nf(nf(3.0 * t3) - nf(5.0 * t2)) + 2.0) * 0.5;
        o[2] = nf(nf(nf(-3.0 * t3) + 4.0 * t2) + f) * 0.5;
        o[3] = (t3 - t2) * 0.5;
    } else {
        var s = 0.0;
        for (var k = 0; k < 6; k++) {
            o[k] = lanczos3(f - (f32(k) - 2.0));
            s += o[k];
        }
        for (var k = 0; k < 6; k++) {
            o[k] = div_ieee(o[k], s);
        }
    }
    return o;
}

fn snap(v: f32) -> f32 {
    let r = round(v);
    return select(v, r, abs(v - r) < 1e-3);
}

@compute @workgroup_size(16, 16)
fn affine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = pu(10u);
    if (gid.x >= tw || gid.y >= pu(12u)) {
        return;
    }
    let x = gid.x;
    let y = pu(11u) + gid.y;
    let o = gid.y * tw + x;
    let t = ((y >> 8u) * pu(13u) + (x >> 8u)) * 4u;
    let fx0 = bitcast<i32>(ta[t]);
    let fy0 = bitcast<i32>(ta[t + 1u]);
    let bw = bitcast<i32>(ta[t + 2u]);
    let bh = bitcast<i32>(ta[t + 3u]);
    if (bw == 0) {
        dst[o] = 0u;
        return;
    }
    let kind = pu(9u);
    let stride = pu(6u);
    let sw = pi(7u);
    let sh = pi(8u);
    let xc = f32(x) + 0.5;
    let yc = f32(y) + 0.5;
    let sx = snap(nf(nf(nf(nf(pf(0u) * xc) + nf(pf(1u) * yc)) + pf(2u)) - 0.5) - f32(fx0));
    let sy = snap(nf(nf(nf(nf(pf(3u) * xc) + nf(pf(4u) * yc)) + pf(5u)) - 0.5) - f32(fy0));
    let ix = floor(sx);
    let iy = floor(sy);
    let wx = weights(kind, sx - ix);
    let wy = weights(kind, sy - iy);
    let rad = i32(kind) + 1;
    let n = 2 * rad;
    let bx = i32(ix) - rad + 1;
    let by = i32(iy) - rad + 1;
    var acc = vec4<f32>(0.0);
    for (var j = 0; j < n; j++) {
        let wyj = wy[j];
        let gy = fy0 + clamp(by + j, 0, bh - 1);
        if (gy < 0 || gy >= sh) {
            continue;
        }
        let row = u32(gy) * stride;
        for (var i = 0; i < n; i++) {
            let gx = fx0 + clamp(bx + i, 0, bw - 1);
            if (gx < 0 || gx >= sw) {
                continue;
            }
            let v = src[row + u32(gx)];
            if (v == 0u) {
                continue;
            }
            let k = nf(wx[i] * wyj);
            acc = nf4(acc + nf4(vec4<f32>(unpack4(v)) * k));
        }
    }
    let a = clamp(floor(acc.w + 0.5), 0.0, 255.0);
    if (a > 0.0) {
        let c = clamp(floor(acc.xyz + vec3<f32>(0.5)), vec3<f32>(0.0), vec3<f32>(a));
        dst[o] = pack4(vec4<u32>(vec3<u32>(c), u32(a)));
    } else {
        dst[o] = 0u;
    }
}

// ----- transforms: mip-aware bilinear with soft edges (`SmartObject::render`)
// params: inv[0..9], sw, sh (f32 source size), stride, lw, lh (this level),
// level, first, 1 / 2^level, tw, y0, rows, cols. ta: per output tile (level
// or 0xffffffff when empty, edge softness f32). One dispatch per level used;
// the first also writes the empty tiles.

fn texel(x: i32, y: i32) -> u32 {
    let xx = clamp(x, 0, pi(12u) - 1);
    let yy = clamp(y, 0, pi(13u) - 1);
    return src[u32(yy) * pu(11u) + u32(xx)];
}

// `Sampler::sample`.
fn bilinear(xs: f32, ys: f32) -> u32 {
    let x = xs - 0.5;
    let y = ys - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = u32(nf((x - x0) * 255.0));
    let fy = u32(nf((y - y0) * 255.0));
    let xi = i32(x0);
    let yi = i32(y0);
    let top = lerp4(texel(xi, yi), texel(xi + 1, yi), fx);
    let bot = lerp4(texel(xi, yi + 1), texel(xi + 1, yi + 1), fx);
    return lerp4(top, bot, fy);
}

@compute @workgroup_size(16, 16)
fn smart(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tw = pu(17u);
    if (gid.x >= tw || gid.y >= pu(19u)) {
        return;
    }
    let x = gid.x;
    let y = pu(18u) + gid.y;
    let o = gid.y * tw + x;
    let t = ((y >> 8u) * pu(20u) + (x >> 8u)) * 2u;
    let lv = ta[t];
    if (lv != pu(14u)) {
        if (lv == 0xffffffffu && pu(15u) != 0u) {
            dst[o] = 0u;
        }
        return;
    }
    let soft = bitcast<f32>(ta[t + 1u]);
    let xx = f32(x) + 0.5;
    let yy = f32(y) + 0.5;
    var w = nf(nf(pf(6u) * xx) + nf(pf(7u) * yy)) + pf(8u);
    if (abs(w) < 1e-12) {
        w = 1e-12;
    }
    let qx = div_ieee(nf(nf(pf(0u) * xx) + nf(pf(1u) * yy)) + pf(2u), w);
    let qy = div_ieee(nf(nf(pf(3u) * xx) + nf(pf(4u) * yy)) + pf(5u), w);
    let ex = clamp(min(div_ieee(qx, soft) + 0.5, div_ieee(pf(9u) - qx, soft) + 0.5), 0.0, 1.0);
    let ey = clamp(min(div_ieee(qy, soft) + 0.5, div_ieee(pf(10u) - qy, soft) + 0.5), 0.0, 1.0);
    let e = ex * ey;
    if (e <= 0.0) {
        dst[o] = 0u;
        return;
    }
    let ik = pf(16u);
    let v = bilinear(qx * ik, qy * ik);
    if (e >= 1.0) {
        dst[o] = v;
    } else {
        dst[o] = pack4(mul255(unpack4(v), u32(nf(e * 255.0) + 0.5)));
    }
}
