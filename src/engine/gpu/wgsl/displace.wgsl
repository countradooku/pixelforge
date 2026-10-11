// Distort ▸ Displace (`extra::displace`) and Lens Correction
// (`extra::render_lens_correction`); see `kernels::displace`.

@group(0) @binding(0) var<uniform> pp: array<vec4<u32>, 8>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
// Displace: the horizontally blurred red / green planes (or `src`).
@group(0) @binding(3) var<storage, read> t0: array<u32>;
// `fast::gauss_kernel(2.0)` taps.
@group(0) @binding(4) var<storage, read> taps: array<u32>;
// Map file: red, green bytes per texel, two texels per word.
@group(0) @binding(5) var<storage, read> fmap: array<u32>;
// The CPU's shift `(v / 255 - 0.5) * 2 * scale / s` per map value v:
// horizontal at [v], vertical at [256 + v].
@group(0) @binding(6) var<storage, read> dtab: array<f32>;
@group(0) @binding(7) var<storage, read_write> hrow: array<u32>;

fn pu(i: u32) -> u32 {
    return pp[i >> 2u][i & 3u];
}

fn pi(i: u32) -> i32 {
    return bitcast<i32>(pu(i));
}

fn pf(i: u32) -> f32 {
    return bitcast<f32>(pu(i));
}

// The value of x, opaque to the compiler (word 31 is always 0), so a product
// feeding an addition rounds on its own as on the CPU instead of fusing
// into an FMA.
fn nf(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | pu(31u));
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

// ----- Displace ---------------------------------------------------------------
// params: pw, ph, pad, cw, ch, ox, oy, s, hs, vs, kind (0 layer, 1 clouds,
// 2 file), r, blur, mw, mh, tile, img_w, img_h, level, 1 / s, commit, k,
// tx0, -, ty0. At commit the CPU renders blocks of k x k tiles from the
// region's first tile (tx0, ty0) and positions are block-local floats: the
// kernel computes them in the same local coordinates.

// `gauss_plane`'s horizontal taps on the red and green planes (interior
// columns; rounded to 8 bits like the CPU's u16 lanes).
@compute @workgroup_size(16, 16)
fn disp_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pw = pu(0u);
    let r = pu(11u);
    if (gid.x >= pw || gid.y >= pu(1u) || gid.x < r || gid.x >= pw - r) {
        return;
    }
    let row = gid.y * pw + gid.x - r;
    var acc = vec2<u32>(0u);
    for (var k = 0u; k <= 2u * r; k++) {
        let v = src[row + k];
        acc += vec2<u32>(v & 255u, (v >> 8u) & 255u) * taps[k];
    }
    let h = (acc + vec2<u32>(128u)) >> vec2<u32>(8u);
    hrow[gid.y * pw + gid.x] = h.x | (h.y << 8u);
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let xi = floor(x);
    let yi = floor(y);
    let fx = x - xi;
    let fy = y - yi;
    let u = nf(fx * fx) * (3.0 - nf(2.0 * fx));
    let v = nf(fy * fy) * (3.0 - nf(2.0 * fy));
    let ix = i32(xi);
    let iy = i32(yi);
    let a = hash01(ix, iy, seed);
    let b = hash01(ix + 1, iy, seed);
    let c = hash01(ix, iy + 1, seed);
    let d = hash01(ix + 1, iy + 1, seed);
    let e = nf(nf(nf(a - b) - c) + d);
    return nf(nf(a + nf((b - a) * u)) + nf((c - a) * v)) + nf(nf(e * u) * v);
}

// `util::fbm` with 4 octaves (every product rounded on its own, as on the CPU).
fn fbm4(x: f32, y: f32, seed: u32) -> f32 {
    var amp = 0.5;
    var freq = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    for (var o = 0u; o < 4u; o++) {
        sum = nf(sum + nf(amp * value_noise(nf(x * freq), nf(y * freq), seed + o * 977u)));
        norm = nf(norm + amp);
        amp *= 0.5;
        freq *= 2.0;
    }
    return div_ieee(sum, norm);
}

fn map_texel(x: u32, y: u32) -> vec2<u32> {
    let i = y * pu(13u) + x;
    let w = fmap[i >> 1u] >> ((i & 1u) * 16u);
    return vec2<u32>(w & 255u, (w >> 8u) & 255u);
}

// Map index along one axis for a tiled file map: `gx.rem_euclid(m)` floored,
// in exact integers (gx = (o + 0.5) * 2^level).
fn tiled(o: i32, m: u32) -> u32 {
    let g = (2 * o + 1) << pu(18u);
    let mm = i32(2u * m);
    return u32(((g % mm) + mm) % mm) >> 1u;
}

fn stretched(u: f32, m: u32) -> u32 {
    return u32(clamp(i32(clamp(u, -1e9, 1e9)), 0, i32(m) - 1));
}

@compute @workgroup_size(16, 16)
fn displace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cw = pu(3u);
    if (gid.x >= cw || gid.y >= pu(4u)) {
        return;
    }
    let pw = pu(0u);
    let ph = pu(1u);
    let pad = pu(2u);
    let bx = gid.x + pad;
    let by = gid.y + pad;
    let ox = pi(5u) + i32(bx);
    let oy = pi(6u) + i32(by);
    let s = pf(7u);
    // Shift in band pixels.
    var d: vec2<f32>;
    let kind = pu(10u);
    if (kind == 1u) {
        let gx = (f32(ox) + 0.5) * s;
        let gy = (f32(oy) + 0.5) * s;
        let u = div_ieee(gx, 180.0);
        let v = div_ieee(gy, 180.0);
        let m = vec2<f32>(fbm4(u, v, 401u), fbm4(u + 9.1, v, 403u));
        let is = pf(19u);
        d = vec2<f32>(nf(nf(nf((m.x - 0.5) * 2.0) * pf(8u)) * is), nf(nf(nf((m.y - 0.5) * 2.0) * pf(9u)) * is));
    } else {
        var c: vec2<u32>;
        if (kind == 0u) {
            if (pu(12u) != 0u) {
                let r = pu(11u);
                var acc = vec2<u32>(0u);
                for (var k = 0u; k <= 2u * r; k++) {
                    let v = t0[(by - r + k) * pw + bx];
                    acc += vec2<u32>(v & 255u, (v >> 8u) & 255u) * taps[k];
                }
                c = (acc + vec2<u32>(128u)) >> vec2<u32>(8u);
            } else {
                let v = src[by * pw + bx];
                c = vec2<u32>(v & 255u, (v >> 8u) & 255u);
            }
        } else {
            let gx = (f32(ox) + 0.5) * s;
            let gy = (f32(oy) + 0.5) * s;
            let mw = pu(13u);
            let mh = pu(14u);
            if (pu(15u) != 0u) {
                c = map_texel(tiled(ox, mw), tiled(oy, mh));
            } else {
                c = map_texel(stretched(div_ieee(gx, pf(16u)) * f32(mw), mw), stretched(div_ieee(gy, pf(17u)) * f32(mh), mh));
            }
        }
        d = vec2<f32>(dtab[c.x], dtab[256u + c.y]);
    }
    // Band index of the CPU buffer's origin (0 outside commits).
    var base = vec2<i32>(0);
    if (pu(20u) != 0u) {
        let k = pi(21u);
        let pad = i32(pad);
        let t = vec2<i32>(ox, oy) >> vec2<u32>(8u);
        let t0 = vec2<i32>(pi(22u), pi(24u));
        let bt = t0 + ((t - t0) / k) * k;
        base = bt * 256 - pad - vec2<i32>(pi(5u), pi(6u));
    }
    let x = f32(i32(bx) - base.x) + d.x;
    let y = f32(i32(by) - base.y) + d.y;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = u32((x - x0) * 255.0);
    let fy = u32((y - y0) * 255.0);
    let xi = i32(x0) + base.x;
    let yi = i32(y0) + base.y;
    let xa = u32(clamp(xi, 0, i32(pw) - 1));
    let xb = u32(clamp(xi + 1, 0, i32(pw) - 1));
    let ya = u32(clamp(yi, 0, i32(ph) - 1)) * pw;
    let yb = u32(clamp(yi + 1, 0, i32(ph) - 1)) * pw;
    dst[gid.y * cw + gid.x] = lerp4(lerp4(src[ya + xa], src[ya + xb], fx), lerp4(src[yb + xa], src[yb + xb], fx), fy);
}

// ----- Lens Correction ----------------------------------------------------------
// params: ox, oy, w, h, stride, iw, ih, s, cx, cy, rn, k, ca_r, ca_b, vignette / 100.

fn texel(x: i32, y: i32) -> u32 {
    let xx = clamp(x, 0, pi(5u) - 1);
    let yy = clamp(y, 0, pi(6u) - 1);
    return src[u32(yy) * pu(4u) + u32(xx)];
}

// `Sampler::sample`.
fn bilinear(xs: f32, ys: f32) -> vec4<u32> {
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
    return unpack4(lerp4(top, bot, fy));
}

@compute @workgroup_size(16, 16)
fn lens(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = pu(2u);
    if (gid.x >= w || gid.y >= pu(3u)) {
        return;
    }
    let s = pf(7u);
    let cx = pf(8u);
    let cy = pf(9u);
    let rn = pf(10u);
    let gx = (f32(pi(0u) + i32(gid.x)) + 0.5) * s;
    let gy = (f32(pi(1u) + i32(gid.y)) + 0.5) * s;
    let dx = div_ieee(gx - cx, rn);
    let dy = div_ieee(gy - cy, rn);
    let r2 = nf(dx * dx) + nf(dy * dy);
    let f = 1.0 + nf(pf(11u) * r2);
    let scs = vec3<f32>(pf(12u), 1.0, pf(13u));
    var o = vec4<u32>(0u);
    for (var c = 0u; c < 3u; c++) {
        // Left to right like the CPU (fast math may otherwise reassociate).
        let sx = cx + nf(nf(nf(dx * rn) * f) * scs[c]);
        let sy = cy + nf(nf(nf(dy * rn) * f) * scs[c]);
        let v = bilinear(div_ieee(sx, s), div_ieee(sy, s));
        o[c] = v[c];
        if (c == 1u) {
            o.w = v.w;
        }
    }
    let vg = 1.0 + nf(nf(pf(14u) * min(r2, 1.0)) * 1.2);
    let a = f32(o.w);
    let q = vec3<u32>(clamp(vec3<f32>(o.xyz) * vg, vec3<f32>(0.0), vec3<f32>(a)));
    dst[gid.y * w + gid.x] = pack4(vec4<u32>(q, o.w));
}
