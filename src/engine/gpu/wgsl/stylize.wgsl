// Stylize ▸ Find Edges, Emboss, Trace Contour, Diffuse, Wind, Tiles,
// Lighting Effects; Pixelate ▸ Fragment, Mezzotint, Color Halftone;
// Other ▸ Custom, De-Interlace (`filters::stylize`, `filters::extra`).
//
// Every kernel reads the padded band `src` (`pw x ph`, clamped like
// `read_region_clamped`; raw tiles for the point filter Mezzotint) and
// writes the `cw x ch` center. Integer filters use integer math only. Float
// math that decides a sample position or a rounding goes through `opq`
// (blocks FMA contraction, so every product rounds as on the CPU) and the
// prelude's correctly rounded `p_div`.
//
// Some CPU results depend on the CPU's render block, not only on the pixel:
// the CPU computes `(x / s - origin) as isize` in float relative to the
// block's buffer (Halftone, Tiles), and Lighting's bump map is blurred only
// inside the block. `p.blk` is the block width in center pixels (256: tile
// blocks of the commit; huge: the preview, whose block is the whole band).

struct P {
    pw: u32,
    ph: u32,
    pad: u32,
    cw: u32,
    ch: u32,
    ox: i32,
    oy: i32,
    blk: u32,
    // A runtime zero (see `opq`).
    zero: u32,
    s: f32,
    inv_s: f32,
    n: u32,
    i: array<vec4<i32>, 2>,
    f: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> tab: array<u32>;
@group(0) @binding(4) var<storage, read> tabf: array<f32>;
// unpremul[a * 256 + v] (bytes) = `px::unpremultiply` channel.
@group(0) @binding(5) var<storage, read> unpremul: array<u32>;
@group(0) @binding(6) var<storage, read> aux: array<u32>;

fn pi(k: u32) -> i32 {
    return p.i[k >> 2u][k & 3u];
}

fn pf(k: u32) -> f32 {
    return p.f[k >> 2u][k & 3u];
}

// The value of `x`, opaque to the compiler: a product passed through it is
// rounded on its own instead of being fused into a following add.
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

fn opq2(x: vec2<f32>) -> vec2<f32> {
    return bitcast<vec2<f32>>(bitcast<vec2<u32>>(x) ^ vec2<u32>(p.zero));
}

// Rust's `f32::round` (half away from zero).
fn round_away(x: f32) -> f32 {
    let t = trunc(x);
    if (abs(x - t) >= 0.5) {
        return t + sign(x);
    }
    return t;
}

fn bidx(x: i32, y: i32) -> u32 {
    let cx = u32(clamp(x, 0, i32(p.pw) - 1));
    let cy = u32(clamp(y, 0, i32(p.ph) - 1));
    return cy * p.pw + cx;
}

fn unpremul_ch(v: u32, a: u32) -> u32 {
    let i = a * 256u + v;
    return (unpremul[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

// `px::unpremultiply`.
fn straight(c: vec4<u32>) -> vec4<u32> {
    if (c.w == 0u) {
        return vec4<u32>(0u);
    }
    if (c.w == 255u) {
        return c;
    }
    return vec4<u32>(unpremul_ch(c.x, c.w), unpremul_ch(c.y, c.w), unpremul_ch(c.z, c.w), c.w);
}

fn lum8(c: vec4<u32>) -> i32 {
    return i32((c.x * 77u + c.y * 150u + c.z * 29u) >> 8u);
}

// floor(sqrt(v)) for v < 2^24 (f32(v) is exact; one correction covers an
// approximate sqrt).
fn isqrt(v: u32) -> u32 {
    var q = u32(sqrt(f32(v)));
    if (q * q > v) {
        q -= 1u;
    } else if ((q + 1u) * (q + 1u) <= v) {
        q += 1u;
    }
    return q;
}

// The CPU block (in band center coordinates) of center column `x`: start
// and end, clipped to the band.
fn block_x(x: u32) -> vec2<u32> {
    let b0 = (x / p.blk) * p.blk;
    return vec2<u32>(b0, min(b0 + p.blk, p.cw));
}

fn block_y(y: u32) -> vec2<u32> {
    let b0 = (y / p.blk) * p.blk;
    return vec2<u32>(b0, min(b0 + p.blk, p.ch));
}

// ----- 3x3 and offset filters ---------------------------------------------------------

// `stylize::find_edges`: Sobel magnitude per channel, inverted.
@compute @workgroup_size(16, 16)
fn find_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = i32(gid.x + p.pad);
    let y = i32(gid.y + p.pad);
    let tl = vec4<i32>(unpack4(src[bidx(x - 1, y - 1)]));
    let t = vec4<i32>(unpack4(src[bidx(x, y - 1)]));
    let tr = vec4<i32>(unpack4(src[bidx(x + 1, y - 1)]));
    let l = vec4<i32>(unpack4(src[bidx(x - 1, y)]));
    let r = vec4<i32>(unpack4(src[bidx(x + 1, y)]));
    let bl = vec4<i32>(unpack4(src[bidx(x - 1, y + 1)]));
    let b = vec4<i32>(unpack4(src[bidx(x, y + 1)]));
    let br = vec4<i32>(unpack4(src[bidx(x + 1, y + 1)]));
    let gx = (tr + 2 * r + br) - (tl + 2 * l + bl);
    let gy = (bl + 2 * b + br) - (tl + 2 * t + tr);
    let s2 = vec3<u32>(gx.xyz * gx.xyz + gy.xyz * gy.xyz);
    let m = vec3<i32>(i32(isqrt(s2.x) >> 1u), i32(isqrt(s2.y) >> 1u), i32(isqrt(s2.z) >> 1u));
    let o = vec3<u32>(clamp(vec3<i32>(255) - m, vec3<i32>(0), vec3<i32>(255)));
    let a = src[bidx(x, y)] >> 24u;
    dst[gid.y * p.cw + gid.x] = premul(o, a);
}

// `stylize::emboss`: i0, i1 = luminance offset; tab[d + 255] = gray level.
@compute @workgroup_size(16, 16)
fn emboss(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = i32(gid.x + p.pad);
    let y = i32(gid.y + p.pad);
    let dx = pi(0u);
    let dy = pi(1u);
    let d = lum8(unpack4(src[bidx(x + dx, y + dy)])) - lum8(unpack4(src[bidx(x - dx, y - dy)]));
    let v = tab[u32(d + 255)];
    let a = src[bidx(x, y)] >> 24u;
    dst[gid.y * p.cw + gid.x] = premul(vec3<u32>(v), a);
}

// `stylize::trace_contour`: i0 = level.
@compute @workgroup_size(16, 16)
fn trace_contour(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = i32(gid.x + p.pad);
    let y = i32(gid.y + p.pad);
    let lv = u32(pi(0u));
    let c = unpack4(src[bidx(x, y)]);
    let r = unpack4(src[bidx(x + 1, y)]);
    let d = unpack4(src[bidx(x, y + 1)]);
    let ab = c.xyz >= vec3<u32>(lv);
    let edge = (ab != (r.xyz >= vec3<u32>(lv))) | (ab != (d.xyz >= vec3<u32>(lv)));
    let o = select(vec3<u32>(255u), vec3<u32>(0u), edge);
    dst[gid.y * p.cw + gid.x] = premul(o, c.w);
}

// `stylize::diffuse`: f0 = radius (level px, >= 1).
@compute @workgroup_size(16, 16)
fn diffuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = i32(gid.x + p.pad);
    let y = i32(gid.y + p.pad);
    let gx = p.ox + x;
    let gy = p.oy + y;
    let r = pf(0u);
    let ox = i32(round_away((hash01(gx, gy, 11u) * 2.0 - 1.0) * r));
    let oy = i32(round_away((hash01(gx, gy, 23u) * 2.0 - 1.0) * r));
    dst[gid.y * p.cw + gid.x] = src[bidx(x + ox, y + oy)];
}

// `stylize::fragment`: i0 = offset.
@compute @workgroup_size(16, 16)
fn fragment(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = i32(gid.x + p.pad);
    let y = i32(gid.y + p.pad);
    let d = pi(0u);
    let s = unpack4(src[bidx(x - d, y - d)]) + unpack4(src[bidx(x + d, y - d)]) + unpack4(src[bidx(x - d, y + d)]) + unpack4(src[bidx(x + d, y + d)]);
    dst[gid.y * p.cw + gid.x] = pack4((s + vec4<u32>(2u)) / vec4<u32>(4u));
}

// `extra::custom3`: f0..f8 = kernel, f9 = scale, f10 = offset.
@compute @workgroup_size(16, 16)
fn custom3(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = i32(gid.x + p.pad);
    let y = i32(gid.y + p.pad);
    var acc = vec3<f32>(0.0);
    for (var j = 0u; j < 9u; j++) {
        let dx = i32(j % 3u) - 1;
        let dy = i32(j / 3u) - 1;
        let c = vec3<f32>(unpack4(src[bidx(x + dx, y + dy)]).xyz);
        let k = pf(j);
        acc = vec3<f32>(opq(acc.x + opq(c.x * k)), opq(acc.y + opq(c.y * k)), opq(acc.z + opq(c.z * k)));
    }
    let a = src[bidx(x, y)] >> 24u;
    let sc = pf(9u);
    let off = pf(10u);
    let fa = f32(a);
    let o = vec3<u32>(
        u32(clamp(round_away(opq(div_rn(acc.x, sc)) + off), 0.0, fa)),
        u32(clamp(round_away(opq(div_rn(acc.y, sc)) + off), 0.0, fa)),
        u32(clamp(round_away(opq(div_rn(acc.z, sc)) + off), 0.0, fa)),
    );
    dst[gid.y * p.cw + gid.x] = pack4(vec4<u32>(o, a));
}

// `extra::deinterlace`: i0 = odd, i1 = interpolate.
@compute @workgroup_size(16, 16)
fn deinterlace(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = i32(gid.x + p.pad);
    let y = i32(gid.y + p.pad);
    let gy = p.oy + y;
    let odd = pi(0u) != 0;
    var v = src[bidx(x, y)];
    if ((((gy % 2) + 2) % 2 == 1) == odd) {
        let up = src[bidx(x, max(y - 1, 0))];
        let dn = src[bidx(x, y + 1)];
        v = select(up, lerp4(up, dn, 128u), pi(1u) != 0);
    }
    dst[gid.y * p.cw + gid.x] = v;
}

// `stylize::wind`: a decaying running max along each row. Thread: one row,
// p.n center columns, starting p.pad columns early (the streak reach), so
// every column sees exactly the CPU's run. i0 = from the left, f0 = decay.
@compute @workgroup_size(1, 64)
fn wind(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    if (y >= p.ch) {
        return;
    }
    let c0 = gid.x * p.n;
    if (c0 >= p.cw) {
        return;
    }
    let c1 = min(c0 + p.n, p.cw);
    let row = (y + p.pad) * p.pw;
    let decay = pf(0u);
    var run = vec3<f32>(0.0);
    if (pi(0u) != 0) {
        for (var b = c0; b < c1 + p.pad; b++) {
            let v = src[row + b];
            run = max(run - vec3<f32>(decay), vec3<f32>(unpack4(v).xyz));
            if (b >= c0 + p.pad) {
                let a = v >> 24u;
                dst[y * p.cw + b - p.pad] = pack4(vec4<u32>(min(vec3<u32>(run), vec3<u32>(a)), a));
            }
        }
    } else {
        for (var b = i32(c1 + 2u * p.pad) - 1; b >= i32(c0 + p.pad); b--) {
            let v = src[row + u32(b)];
            run = max(run - vec3<f32>(decay), vec3<f32>(unpack4(v).xyz));
            if (u32(b) < c1 + p.pad) {
                let a = v >> 24u;
                dst[y * p.cw + u32(b) - p.pad] = pack4(vec4<u32>(min(vec3<u32>(run), vec3<u32>(a)), a));
            }
        }
    }
}

// ----- Mezzotint (a point filter: raw tiles, pad 0) ---------------------------------------

@compute @workgroup_size(256)
fn mezzotint(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.pw * p.ph) {
        return;
    }
    let v = src[i];
    let c = unpack4(v);
    if (c.w == 0u) {
        dst[i] = v;
        return;
    }
    let gx = p.ox + i32(i % p.pw);
    let gy = p.oy + i32(i / p.pw);
    let s = straight(c);
    // `v as f32 > hash01 * 255.0`: hash / 2^24 * 255 is one rounding.
    let k = 255.0 / 16777216.0;
    let o = vec3<u32>(
        select(0u, 255u, f32(s.x) > f32(hash_u(gx, gy, 1u)) * k),
        select(0u, 255u, f32(s.y) > f32(hash_u(gx, gy, 2u)) * k),
        select(0u, 255u, f32(s.z) > f32(hash_u(gx, gy, 3u)) * k),
    );
    dst[i] = premul(o, c.w);
}

// ----- Color Halftone -------------------------------------------------------------------

// The CPU's `(pos / s - origin) as isize` (or `.floor()`), relative to the
// CPU block's buffer origin, clamped into that buffer; returns band buffer
// coordinates. `b`: block start (center coordinates), `bn`: block length.
fn block_px(pos: f32, origin: i32, b: u32, bn: u32, floor_it: bool) -> i32 {
    let bo = origin + i32(b);
    let rel = pos * p.inv_s - f32(bo);
    var k: i32;
    if (floor_it) {
        k = i32(floor(rel));
    } else {
        k = i32(rel);
    }
    return clamp(k, 0, i32(bn + 2u * p.pad) - 1) + i32(b);
}

// `stylize::halftone`: f0 = radius, f1 = cell, f2..f7 = (sin, cos) of the
// three screen angles; tabf[v] = dot radius for channel value v.
@compute @workgroup_size(16, 16)
fn halftone(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = gid.x + p.pad;
    let y = gid.y + p.pad;
    let s = p.s;
    let gx = (f32(p.ox + i32(x)) + 0.5) * s;
    let gy = (f32(p.oy + i32(y)) + 0.5) * s;
    let cell = pf(1u);
    let bx = block_x(gid.x);
    let by = block_y(gid.y);
    var o = vec3<u32>(0u);
    for (var k = 0u; k < 3u; k++) {
        let sn = pf(2u + 2u * k);
        let cs = pf(3u + 2u * k);
        let u = opq(gx * cs) + opq(gy * sn);
        let v = opq(-gx * sn) + opq(gy * cs);
        let cu = opq((floor(div_rn(u, cell)) + 0.5) * cell);
        let cv = opq((floor(div_rn(v, cell)) + 0.5) * cell);
        let ix = opq(cu * cs) - opq(cv * sn);
        let iy = opq(cu * sn) + opq(cv * cs);
        let sx = block_px(ix, p.ox, bx.x, bx.y - bx.x, false);
        let sy = block_px(iy, p.oy, by.x, by.y - by.x, false);
        let val = unpack4(src[u32(sy) * p.pw + u32(sx)])[k];
        let rad = tabf[val];
        let du = u - cu;
        let dv = v - cv;
        let d = sqrt(du * du + dv * dv);
        let cov = clamp((rad - d) / s + 0.5, 0.0, 1.0);
        o[k] = u32(255.0 * (1.0 - cov));
    }
    let a = src[y * p.pw + x] >> 24u;
    dst[gid.y * p.cw + gid.x] = premul(o, a);
}

// ----- Tiles ------------------------------------------------------------------------------

// `extra::tiles`: f0 = tile size, f1 = max shift, i0 = fill pixel.
@compute @workgroup_size(16, 16)
fn tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = gid.x + p.pad;
    let y = gid.y + p.pad;
    let s = p.s;
    let size = pf(0u);
    let mx = pf(1u);
    let gx = (f32(p.ox + i32(x)) + 0.5) * s;
    let gy = (f32(p.oy + i32(y)) + 0.5) * s;
    let bx = block_x(gid.x);
    let by = block_y(gid.y);
    var out = bitcast<u32>(pi(0u));
    let cx = i32(floor(div_rn(gx, size)));
    let cy = i32(floor(div_rn(gy, size)));
    for (var oy = -1; oy <= 1; oy++) {
        for (var ox = -1; ox <= 1; ox++) {
            let tx = cx + ox;
            let ty = cy + oy;
            let sx = opq((hash01(tx, ty, 501u) - 0.5) * 2.0 * mx);
            let sy = opq((hash01(tx, ty, 503u) - 0.5) * 2.0 * mx);
            let lx = gx - sx;
            let ly = gy - sy;
            if (i32(floor(div_rn(lx, size))) == tx && i32(floor(div_rn(ly, size))) == ty) {
                let px = block_px(lx, p.ox, bx.x, bx.y - bx.x, true);
                let py = block_px(ly, p.oy, by.x, by.y - by.x, true);
                out = src[u32(py) * p.pw + u32(px)];
            }
        }
    }
    dst[gid.y * p.cw + gid.x] = out;
}

// ----- Lighting Effects -------------------------------------------------------------------
// `extra::lighting`: the bump map is the luma quantized to 8 bits, blurred
// with `gauss_plane` (sigma 1, 7 taps) inside the CPU block and raw at the
// block's 3-pixel rim.

// Pass 1: quantized luma of every band pixel.
@compute @workgroup_size(16, 16)
fn light_lum(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.pw || gid.y >= p.ph) {
        return;
    }
    let i = gid.y * p.pw + gid.x;
    let c = vec3<f32>(straight(unpack4(src[i])).xyz);
    let l = div_rn(opq(opq(0.299 * c.x) + opq(0.587 * c.y)) + opq(0.114 * c.z), 255.0);
    dst[i] = u32(opq(clamp(l, 0.0, 1.0) * 255.0) + 0.5);
}

// Pass 2 / 3: the 7 taps (tab) along x (i0 = 0) or y (i0 = 1), rounded
// after each axis like `gauss_plane`; only where the taps fit the band.
@compute @workgroup_size(16, 16)
fn light_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.pw || gid.y >= p.ph) {
        return;
    }
    let i = gid.y * p.pw + gid.x;
    let vert = pi(0u) != 0;
    let at = select(gid.x, gid.y, vert);
    let len = select(p.pw, p.ph, vert);
    if (at < 3u || at + 3u >= len) {
        dst[i] = aux[i];
        return;
    }
    var acc = 0u;
    for (var j = 0u; j < 7u; j++) {
        var k: u32;
        if (vert) {
            k = (gid.y + j - 3u) * p.pw + gid.x;
        } else {
            k = i + j - 3u;
        }
        acc += aux[k] * tab[j];
    }
    dst[i] = (acc + 128u) >> 8u;
}

fn smooth_step(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Pass 4: shading. src = the band, aux = blurred luma (valid inside the
// band's 3-pixel rim), tab = raw quantized luma, tabf[v] = v / 255.
// i0 = kind; f0 = x, f1 = y, f2 = radius, f3, f4 = direction (cos, -sin),
// f5 = cos(cone), f6 = intensity / 50, f7 = ambient, f8 = bump / 10,
// f9..f11 = light color / 255.
@group(0) @binding(7) var<storage, read> raw_lum: array<u32>;

fn bump(x: u32, y: u32, bx: vec2<u32>, by: vec2<u32>) -> f32 {
    // Center coordinates of the neighbor (may be -1 or one past the end).
    let cx = i32(x) - i32(p.pad);
    let cy = i32(y) - i32(p.pad);
    let i = y * p.pw + x;
    if (cx >= i32(bx.x) && cx < i32(bx.y) && cy >= i32(by.x) && cy < i32(by.y)) {
        return tabf[aux[i]];
    }
    return tabf[raw_lum[i]];
}

@compute @workgroup_size(16, 16)
fn lighting(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = gid.x + p.pad;
    let y = gid.y + p.pad;
    let s = p.s;
    let bx = block_x(gid.x);
    let by = block_y(gid.y);
    let bz = pf(8u);
    let nx = -(bump(x + 1u, y, bx, by) - bump(x - 1u, y, bx, by)) * bz * 4.0;
    let ny = -(bump(x, y + 1u, bx, by) - bump(x, y - 1u, bx, by)) * bz * 4.0;
    let nl = sqrt(nx * nx + ny * ny + 1.0);
    let k = pf(6u);
    let amb = pf(7u);
    let ddx = pf(3u);
    let ddy = pf(4u);
    var e: f32;
    if (pi(0u) == 2) {
        let lx = -ddx * 0.7;
        let ly = -ddy * 0.7;
        e = amb + k * max((nx * lx + ny * ly + 0.7) / nl, 0.0);
    } else {
        let gx = (f32(p.ox + i32(x)) + 0.5) * s;
        let gy = (f32(p.oy + i32(y)) + 0.5) * s;
        let radius = pf(2u);
        let dx = pf(0u) - gx;
        let dy = pf(1u) - gy;
        let dist = sqrt(dx * dx + dy * dy);
        let q = dist / radius;
        let fall = max(1.0 - q * q, 0.0);
        let z = radius * 0.5;
        let ld = sqrt(dx * dx + dy * dy + z * z);
        let lambert = max((nx * dx + ny * dy + z) / (nl * ld), 0.0);
        var spot = 1.0;
        if (pi(0u) == 1) {
            var c = 1.0;
            if (dist > 1e-3) {
                c = (-dx * ddx - dy * ddy) / dist;
            }
            let cone = pf(5u);
            spot = smooth_step(cone - 0.05, cone + 0.05, c);
        }
        e = amb + k * fall * lambert * spot;
    }
    let sv = straight(unpack4(src[y * p.pw + x]));
    let lc = vec3<f32>(pf(9u), pf(10u), pf(11u));
    let f = vec3<f32>(tabf[sv.x], tabf[sv.y], tabf[sv.z]) * (vec3<f32>(amb) + (e - amb) * lc);
    let o = vec3<u32>(clamp(f, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0);
    dst[gid.y * p.cw + gid.x] = premul(o, sv.w);
}
