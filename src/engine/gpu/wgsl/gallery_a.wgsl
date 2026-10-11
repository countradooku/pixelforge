
// ----- Filter Gallery: effect stages (pointwise finishes, noise, strokes) -----

fn dimsx(z: u32) -> vec2<u32> {
    let e = u32(max(p.i.w, 0));
    return vec2<u32>(win[z].w + 2u * e, win[z].h + 2u * e);
}

// Uniform noise hashed at level-0 coordinates (`ctx.noise(i, seed) - off`),
// on a plane extended by e = p.i.w.
@compute @workgroup_size(16, 16)
fn noise(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dimsx(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let w = win[z];
    let e = p.i.w;
    let h = hash01(g0(w.ox - e, i32(id.x)), g0(w.oy - e, i32(id.y)), u32(p.i.x));
    b0[widx(z, id.x, id.y)] = o(h - p.f.x);
}

// `duotone`: fg + (bg - fg) * clamp(t).
@compute @workgroup_size(16, 16)
fn duotone(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let t = clamp(a0[i], 0.0, 1.0);
    let c = mix3(p.f.xyz, p.g.xyz, t);
    emit3(id, i, c);
}

// `sketch1::line_mean`: mean of p.i.x samples along (p.i.y, p.i.z),
// centred, edge clamped (on a plane extended by p.i.w).
@compute @workgroup_size(16, 16)
fn line_mean(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let d = dimsx(z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let n = p.i.x;
    let r0 = -(n / 2);
    var s = 0.0;
    for (var k = r0; k < r0 + n; k++) {
        let xx = clamp(i32(id.x) + k * p.i.y, 0, i32(d.x) - 1);
        let yy = clamp(i32(id.y) + k * p.i.z, 0, i32(d.y) - 1);
        s = fadd(s, a0[widx(z, u32(xx), u32(yy))]);
    }
    b0[widx(z, id.x, id.y)] = fmul(s, p.f.x);
}

// ----- Artistic -------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn film_grain(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let c = vec3<f32>(a0[i], a1[i], a2[i]);
    let k = p.f.x;
    let v = max(max(c.x, c.y), c.z);
    let n = hash01(g0(w.ox, i32(id.x)), g0(w.oy, i32(id.y)), 41u);
    let vo = clamp(v * (1.0 - k) + k * n + max(v - p.f.y, 0.0) * p.f.z, 0.0, 1.0);
    var out = vec3<f32>(vo);
    if (v > 1e-4) {
        out = c * (vo / v);
    }
    emit3(id, i, out);
}

fn lut17(off: u32, t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0) * 16.0;
    let k = min(u32(x), 15u);
    let a = tabf(t0[off + k]);
    let b = tabf(t0[off + k + 1u]);
    return a + (b - a) * (x - f32(k));
}

@compute @workgroup_size(16, 16)
fn neon(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let l = a0[i];
    let lb = a1[i];
    let k = p.f.x;
    let key = select(lb, 1.0 - lb, p.f.y > 0.5);
    let dk = max(1.0 - (1.0 - lut17(0u, key)) * k, 0.0);
    let tt = lut17(17u, key) * k;
    let base = mix3(p.g.xyz, p.h.xyz, l);
    let out = base * dk + vec3<f32>(0.0, 0.0, 1.0) * tt * l;
    emit3(id, i, out);
}

fn nzi(ox: i32, oy: i32, x: i32, y: i32, seed: u32) -> f32 {
    return hash01(g0(ox, x), g0(oy, y), seed);
}

// `artistic1::diag_streaks`: running mean of noise along 45° lines; thread
// per line. p.i = (half, up, seed), p.f = (n, k).
@compute @workgroup_size(64)
fn diag_streaks(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    let W = i32(w.w);
    let H = i32(w.h);
    let t = i32(id.x);
    if (t >= W + H - 1) {
        return;
    }
    let up = p.i.y == 1;
    let r = p.i.x;
    let seed = u32(p.i.z);
    var d: i32;
    var x0: i32;
    var x1: i32;
    if (up) {
        d = t;
        x0 = max(d - (H - 1), 0);
        x1 = min(d, W - 1);
    } else {
        d = t - (H - 1);
        x0 = max(d, 0);
        x1 = min(d + H - 1, W - 1);
    }
    if (x0 > x1) {
        return;
    }
    let sy = select(-1, 1, !up);
    let y0 = select(-d, d, up);
    // y(x) = up ? d - x : x - d
    var sum = 0.0;
    for (var q = x0 - r; q <= x0 + r; q++) {
        sum = fadd(sum, nzi(w.ox, w.oy, q, y0 + sy * q, seed));
    }
    for (var x = x0; x <= x1; x++) {
        let y = y0 + sy * x;
        b0[widx(z, u32(x), u32(y))] = fmul(fsub(fdiv(sum, p.f.x), 0.5), p.f.y);
        let xa = x + r + 1;
        let xs = x - r;
        sum = fadd(sum, fsub(nzi(w.ox, w.oy, xa, y0 + sy * xa, seed), nzi(w.ox, w.oy, xs, y0 + sy * xs, seed)));
    }
}

@compute @workgroup_size(16, 16)
fn pencil(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let a = clamp(p.f.x + 0.2 * a3[i] + 0.125 * a4[i], 0.0, 1.0);
    let kc = min(vec3<f32>(a0[i], a1[i], a2[i]), p.h.xyz);
    let out = mix3(p.g.xyz, kc, a);
    emit3(id, i, out);
}

// Cutout's luminance bands (exact: they are thresholds).
@compute @workgroup_size(16, 16)
fn cutout_band(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let t = fmul(fdiv(fsub(a0[i], p.f.x), p.f.y), p.f.z);
    let b = min(u32(max(floor(t), 0.0)), u32(p.i.x) - 1u);
    b0[i] = f32(b);
}

// Graphic Pen: ink where luminance plus stroke noise stays under the threshold.
@compute @workgroup_size(16, 16)
fn pen_threshold(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let v = fadd(a0[i], fmul(fsub(a1[i], 0.5), p.f.x));
    b0[i] = select(0.0, 1.0, v > p.f.y);
}

// ----- Texture ---------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn grain(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let c = vec3<f32>(a0[i], a1[i], a2[i]);
    let kind = p.i.x;
    let k = p.f.x;
    let gain = p.f.y;
    if (kind == 0) {
        let xi = g0(w.ox, i32(id.x));
        let yi = g0(w.oy, i32(id.y));
        let amp = p.f.z;
        let nz = vec3<f32>(hash01(xi, yi, 199u), hash01(xi, yi, 206u), hash01(xi, yi, 213u));
        let out = (c - 0.5) * gain + 0.5 + (nz - 0.5) * amp;
        emit3(id, i, out);
        return;
    }
    let x = gf(w.ox, i32(id.x));
    let y = gf(w.oy, i32(id.y));
    let xt = i32(x);
    let yt = i32(y);
    var n = 0.0;
    switch kind {
        case 1: {
            n = fbm(x * 0.6, y * 0.6, 2u, 181u) - 0.5;
        }
        case 2: {
            n = select(0.0, 0.5, hash01(xt, yt, 183u) > 0.96);
        }
        case 3: {
            n = fbm(x * 0.25, y * 0.25, 3u, 185u) - 0.5;
        }
        case 4: {
            n = select(-0.5, 0.5, hash01(xt, yt, 187u) - 0.5 >= 0.0);
        }
        case 5: {
            n = hash01(xt / 3, yt / 3, 189u) - 0.5;
        }
        case 6: {
            n = select(0.0, -0.5, hash01(xt, yt, 191u) > 0.7);
        }
        case 7: {
            n = hash01(0, yt, 193u) - 0.5;
        }
        case 8: {
            n = hash01(xt, 0, 195u) - 0.5;
        }
        case 9: {
            n = select(0.0, -0.6, hash01(xt, yt, 197u) > 0.92);
        }
        default: {
            n = hash01(xt, yt, 199u) - 0.5;
        }
    }
    let t = n * k * 0.8;
    let out = (c - 0.5) * gain + 0.5 + t;
    emit3(id, i, out);
}

@compute @workgroup_size(16, 16)
fn diffuse_glow(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let c = vec3<f32>(a0[i], a1[i], a2[i]);
    let glow = p_pow(clamp((a3[i] - p.f.z) / 0.32, 0.0, 1.0), 0.75) * p.f.y;
    let nz = hash01(g0(w.ox, i32(id.x)), g0(w.oy, i32(id.y)), 91u);
    let t = clamp(min(glow, 1.0) + p.f.x * (2.0 * nz - 1.0), 0.0, 1.0);
    let out = mix3(c, p.g.xyz, t);
    emit3(id, i, out);
}

// ----- Artistic (more) --------------------------------------------------------------

const TAU_: f32 = 0x1.921fb6p+2f;
const PI_: f32 = 0x1.921fb6p+1f;

// Rust's f32::fract (x - trunc(x), negative for negative x).
fn rfract(x: f32) -> f32 {
    return x - trunc(x);
}

// Rust's f32::rem_euclid for a positive divisor.
fn rem_e(x: f32, m: f32) -> f32 {
    let q = trunc(fdiv(x, m));
    var r = fsub(x, fmul(q, m));
    if (r < 0.0) {
        r = fadd(r, m);
    }
    return r;
}

fn rsignum(x: f32) -> f32 {
    return select(1.0, -1.0, x < 0.0);
}

fn with_value(c: vec3<f32>, v: f32) -> vec3<f32> {
    let m = max(max(c.x, c.y), c.z);
    if (m <= 1e-6) {
        return vec3<f32>(v);
    }
    return c * (v / m);
}

struct Px {
    i: u32,
    c: vec3<f32>,
    x: i32,
    y: i32,
    ok: bool,
}

fn px_at(id: vec3<u32>) -> Px {
    let w = win[id.z];
    var r: Px;
    r.ok = id.x < w.w && id.y < w.h;
    if (!r.ok) {
        return r;
    }
    r.i = widx(id.z, id.x, id.y);
    r.c = vec3<f32>(a0[r.i], a1[r.i], a2[r.i]);
    r.x = i32(id.x);
    r.y = i32(id.y);
    return r;
}

fn put3(i: u32, c: vec3<f32>) {
    b0[i] = c.x;
    b1[i] = c.y;
    b2[i] = c.z;
}

@compute @workgroup_size(16, 16)
fn tex_noise(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = px_at(id);
    if (!q.ok) {
        return;
    }
    let w = win[id.z];
    let t = (hash01(g0(w.ox, q.x), g0(w.oy, q.y), 77u) - 0.5) * p.f.x;
    emit3(id, q.i, q.c + vec3<f32>(t));
}

@compute @workgroup_size(16, 16)
fn fresco(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = px_at(id);
    if (!q.ok) {
        return;
    }
    let w = win[id.z];
    let l = a3[q.i];
    let lk = a4[q.i];
    let d = l - lk;
    let sd = select(p.f.y, p.f.x, d > 0.0);
    let g = p_exp(-((d / sd) * (d / sd)));
    var t = lut17(0u, lk) * g;
    if (p.f.z > 0.0) {
        t += (hash01(g0(w.ox, q.x), g0(w.oy, q.y), 77u) - 0.5) * p.f.z;
    }
    let k = max(t, 0.0) / max(l, 1e-3);
    emit3(id, q.i, q.c * k);
}

// a0 + (a0 - a1) * k, exactly (unsharp masks feeding quantization).
@compute @workgroup_size(16, 16)
fn unsharp(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = fadd(a0[i], fmul(fsub(a0[i], a1[i]), p.f.x));
}

// Plastic Wrap's squared central-difference gradient.
@compute @workgroup_size(16, 16)
fn grad2(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    let gx = a0[wcl(z, x + 1, y)] - a0[wcl(z, x - 1, y)];
    let gy = a0[wcl(z, x, y + 1)] - a0[wcl(z, x, y - 1)];
    b0[widx(z, id.x, id.y)] = (gx * gx + gy * gy) * p.f.x;
}

@compute @workgroup_size(16, 16)
fn plastic(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = px_at(id);
    if (!q.ok) {
        return;
    }
    let gn = sqrt(a3[q.i] / (a4[q.i] + p.f.x)) / p.f.z;
    let hh = p.f.y * p_exp(-gn * gn);
    let t = p.f.w * hh;
    emit3(id, q.i, q.c * (1.0 - t) + vec3<f32>(hh * t));
}

@compute @workgroup_size(16, 16)
fn poster_edges(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = px_at(id);
    if (!q.ok) {
        return;
    }
    let bb0 = a3[q.i];
    let bb1 = a4[q.i];
    let step = p.f.x;
    let vq = min(fmul(rround(fdiv(bb0, step)), step), 1.0);
    let d = max(fsub(fsub(bb1, bb0), p.f.y), 0.0);
    let ink = fmul(rround(fdiv(fmul(d, p.f.z), p.f.w)), p.f.w);
    emit3(id, q.i, with_value(q.c, max(fsub(vq, ink), 0.0)));
}

// `artistic2::weave`: the Rough Pastels / Underpainting texture height.
fn weave_h(x: f32, y: f32, kind: i32) -> f32 {
    switch kind {
        case 0: {
            let r = i32(floor(fdiv(y, 3.0)));
            let seg = i32(floor(fadd(fdiv(x, 8.0), fmul(hash01(r, 0, 3u), 5.0))));
            let jit = hash01(seg, r, 91u) - 0.5;
            let rib = p_sin(2.0 * PI_ * (x * 0.577 + y) / 4.0);
            let thread = p_sin(PI_ * rfract(fdiv(y, 3.0)));
            return 0.5 + 0.22 * rib + 0.25 * thread + 0.25 * jit;
        }
        case 1: {
            let u = fdiv(x, 6.0);
            let v = fdiv(y, 6.0);
            let fi = floor(u);
            let fj = floor(v);
            let i = i32(fi);
            let j = i32(fj);
            let fu = u - fi;
            let fv = v - fj;
            let prof = select(p_sin(PI_ * fu), p_sin(PI_ * fv), ((i + j) % 2 + 2) % 2 == 0);
            return 0.3 + 0.5 * prof * (0.7 + 0.3 * hash01(i, j, 91u)) + 0.2 * fbm(x * 0.2, y * 0.2, 2u, 3u);
        }
        case 2: {
            let bh = 12.0;
            let row = floor(fdiv(y, bh));
            let bx = x + select(bh, 0.0, i32(row) % 2 == 0);
            let mortar = rem_e(y, bh) < 1.5 || rem_e(bx, 2.0 * bh) < 1.5;
            if (mortar) {
                return 0.15;
            }
            return 0.6 + 0.2 * (hash01(i32(fdiv(bx, 2.0 * bh)), i32(row), 9u) - 0.5);
        }
        default: {
            return fbm(x * 0.6, y * 0.6, 4u, 11u);
        }
    }
}

@compute @workgroup_size(16, 16)
fn weave(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    b0[widx(id.z, id.x, id.y)] = weave_h(gf(w.ox, i32(id.x)), gf(w.oy, i32(id.y)), p.i.x);
}

@compute @workgroup_size(16, 16)
fn weave_grad(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    let gx = a0[wcl(z, x + 1, y)] - a0[wcl(z, x - 1, y)];
    let gy = a0[wcl(z, x, y + 1)] - a0[wcl(z, x, y - 1)];
    b0[widx(z, id.x, id.y)] = clamp((gx * p.f.x + gy * p.f.y) * p.f.z, -1.0, 1.0);
}

@compute @workgroup_size(16, 16)
fn pastel_shade(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let y = gf(w.oy, i32(id.y));
    b0[i] = 0.5 * a0[i] + 0.5 * p_cos(rfract(y / 3.0) * TAU_);
}

@compute @workgroup_size(16, 16)
fn row_cos(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let y = gf(w.oy, i32(id.y));
    b0[widx(id.z, id.x, id.y)] = p_cos(rfract(y / 3.0) * TAU_);
}

// `artistic2::canvas_rows`: thread rows shift the paint sideways (a3 = shade).
@compute @workgroup_size(16, 16)
fn canvas_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = px_at(id);
    if (!q.ok) {
        return;
    }
    let z = id.z;
    let w = win[z];
    let gy = gf(w.oy, q.y);
    let r = floor(fdiv(gy, p.f.x));
    let jit = hash01(i32(r), 0, 5u) * 7.0;
    let gx = gf(w.ox, q.x);
    let dd = (value_noise(gx / p.f.y + jit, r, 71u) - 0.5) * 2.0 * p.f.z / sf();
    let sx = clamp(f32(q.x) + dd, 0.0, f32(w.w - 1u));
    let x0 = u32(sx);
    let x1 = min(x0 + 1u, w.w - 1u);
    let f = sx - f32(x0);
    let j0 = widx(z, x0, id.y);
    let j1 = widx(z, x1, id.y);
    let sh = a3[q.i] * p.f.w;
    emit3(id, q.i, vec3<f32>(a0[j0] + (a0[j1] - a0[j0]) * f + sh, a1[j0] + (a1[j1] - a1[j0]) * f + sh, a2[j0] + (a2[j1] - a2[j0]) * f + sh));
}

@compute @workgroup_size(16, 16)
fn affine(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = px_at(id);
    if (!q.ok) {
        return;
    }
    put3(q.i, vec3<f32>(fmad(q.c.x, p.f.x, p.f.y), fmad(q.c.y, p.f.x, p.f.y), fmad(q.c.z, p.f.x, p.f.y)));
}

// `artistic2::diag_mean`: mean of (x - k, y + k), k = 0..=m, as the CPU's
// running sum along each anti-diagonal (thread per anti-diagonal).
@compute @workgroup_size(64)
fn diag_mean(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    let W = i32(w.w);
    let H = i32(w.h);
    let d = i32(id.x);
    if (d > W + H - 2) {
        return;
    }
    let m = p.i.x;
    var x = max(0, d - (H - 1));
    var y = d - x;
    var sum = 0.0;
    for (var k = 0; k <= m; k++) {
        sum = fadd(sum, a0[wcl(z, x - k, y + k)]);
    }
    loop {
        b0[widx(z, u32(x), u32(y))] = fmul(sum, p.f.x);
        x += 1;
        y -= 1;
        if (x >= W || y < 0) {
            break;
        }
        sum = fsub(fadd(sum, a0[widx(z, u32(x), u32(y))]), a0[wcl(z, x - m - 1, y + m + 1)]);
    }
}

@compute @workgroup_size(16, 16)
fn mixw(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = a0[i] * p.f.x + a1[i] * p.f.y;
}

@compute @workgroup_size(16, 16)
fn absdiff(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = abs(a0[i] - a1[i]);
}

@compute @workgroup_size(16, 16)
fn pastel_streaks(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = px_at(id);
    if (!q.ok) {
        return;
    }
    let w = win[id.z];
    let x = gf(w.ox, q.x);
    let y = gf(w.oy, q.y);
    let u = i32(floor(fdiv(fadd(x, y), 2.0)));
    let v = fadd(fdiv(fsub(x, y), p.f.x), fmul(hash01(u, 0, 31u), 13.0));
    let st = hash01(u, i32(floor(v)), 29u) - 0.5;
    let t = rsignum(st) * sstep(0.2, 0.5, abs(st)) * p.f.y * (0.3 + min(a3[q.i] * 8.0, 1.0));
    emit3(id, q.i, q.c + vec3<f32>(t));
}

// Mean (b0) and min (b1) along the offsets t0 (pairs), p.i.x of them.
@compute @workgroup_size(16, 16)
fn line_filter(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    var s = 0.0;
    var m = 3.4028235e38;
    for (var k = 0; k < p.i.x; k++) {
        let v = a0[wcl(z, i32(id.x) + bitcast<i32>(t0[2 * k]), i32(id.y) + bitcast<i32>(t0[2 * k + 1]))];
        s = fadd(s, v);
        m = min(m, v);
    }
    let i = widx(z, id.x, id.y);
    b0[i] = fmul(s, p.f.x);
    b1[i] = m;
}

@compute @workgroup_size(16, 16)
fn smudge(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = px_at(id);
    if (!q.ok) {
        return;
    }
    let m = a4[q.i] * p.f.x + a3[q.i] * p.f.y;
    let lo = 0.65 * m + p.g.y;
    let r = clamp((m - p.f.z) / (p.f.w - p.f.z), 0.0, 1.0);
    emit3(id, q.i, with_value(q.c, lo + (1.0 - lo) * r * r * p.g.x));
}

@compute @workgroup_size(16, 16)
fn sponge(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = px_at(id);
    if (!q.ok) {
        return;
    }
    let w = win[id.z];
    let x = gf(w.ox, q.x);
    let y = gf(w.oy, q.y);
    let n = fbm(x / p.f.x, y / p.f.x, 2u, 21u);
    let t = sstep(0.48, 0.52, n);
    let v = max(max(q.c.x, q.c.y), q.c.z);
    emit3(id, q.i, with_value(q.c, clamp(v * p.f.z + p.f.w + p.f.y * (1.0 - 2.0 * t), 0.0, 1.0)));
}

@compute @workgroup_size(16, 16)
fn watercolor(@builtin(global_invocation_id) id: vec3<u32>) {
    let q = px_at(id);
    if (!q.ok) {
        return;
    }
    let li = max(a3[q.i], 1e-4);
    let f = clamp(p.f.y * (li - p.f.x), 0.0, 1.0) / li;
    emit3(id, q.i, q.c * f);
}

// Cutout band k's (r, g, b, count) in one vec4 plane: a0 = band, a1..a3 = color.
@compute @workgroup_size(16, 16)
fn cutout_mask4(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    var v = vec4<f32>(0.0);
    if (u32(a0[i]) == u32(p.i.x)) {
        v = vec4<f32>(a1[i], a2[i], a3[i], 1.0);
    }
    w0[i] = v;
}

@compute @workgroup_size(16, 16)
fn cutout_pick4(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    if (u32(a0[i]) != u32(p.i.x)) {
        return;
    }
    let s = q0[i];
    let dd = fdiv(1.0, max(s.w, p.f.x));
    b0[i] = fmul(s.x, dd);
    b1[i] = fmul(s.y, dd);
    b2[i] = fmul(s.z, dd);
}
