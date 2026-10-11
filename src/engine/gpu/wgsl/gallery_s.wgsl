
// ----- Filter Gallery: Sketch stages -------------------------------------------------

const TAU_S: f32 = 0x1.921fb6p+2f;

fn dims_s(z: u32) -> vec2<u32> {
    let e = u32(max(p.i.w, 0));
    return vec2<u32>(win[z].w + 2u * e, win[z].h + 2u * e);
}

// `sketch1::bilerp` of plane a0 in window z (edge clamped).
fn bilerp(z: u32, xin: f32, yin: f32) -> f32 {
    let w = win[z];
    let x = clamp(xin, 0.0, f32(w.w - 1u));
    let y = clamp(yin, 0.0, f32(w.h - 1u));
    let x0 = u32(x);
    let y0 = u32(y);
    let x1 = min(x0 + 1u, w.w - 1u);
    let y1 = min(y0 + 1u, w.h - 1u);
    let fx = fsub(x, f32(x0));
    let fy = fsub(y, f32(y0));
    let v00 = a0[widx(z, x0, y0)];
    let v10 = a0[widx(z, x1, y0)];
    let v01 = a0[widx(z, x0, y1)];
    let v11 = a0[widx(z, x1, y1)];
    let a = fadd(v00, fmul(fsub(v10, v00), fx));
    let b = fadd(v01, fmul(fsub(v11, v01), fx));
    return fadd(a, fmul(fsub(b, a), fy));
}

// Bas Relief: smoothed-luminance difference along the light, over taps
// t0 = (offset across, weight) x p.i.x; p.f = (ux, uy, vx, vy), p.g = (fwd, back, c).
@compute @workgroup_size(16, 16)
fn bas_relief(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let fx = f32(id.x);
    let fy = f32(id.y);
    let ux = p.f.x;
    let uy = p.f.y;
    var d = 0.0;
    for (var k = 0; k < p.i.x; k++) {
        let q = tabf(t0[2 * k]);
        let g = tabf(t0[2 * k + 1]);
        let px_ = fadd(fx, fmul(q, p.f.z));
        let py_ = fadd(fy, fmul(q, p.f.w));
        let fa = bilerp(z, fadd(px_, fmul(p.g.x, ux)), fadd(py_, fmul(p.g.x, uy)));
        let fb = bilerp(z, fsub(px_, fmul(p.g.y, ux)), fsub(py_, fmul(p.g.y, uy)));
        d = fadd(d, fmul(g, fsub(fa, fb)));
    }
    b0[widx(z, id.x, id.y)] = fadd(0.5, fdiv(fmul(0.5, d), fadd(abs(d), p.g.z)));
}

fn mix32(x: i32, y: i32, seed: u32) -> u32 {
    var h = (bitcast<u32>(x) * 0x8da6b343u) ^ (bitcast<u32>(y) * 0xd8163841u) ^ (seed * 0xcb1ab31fu);
    h = h ^ (h >> 13u);
    h = h * 0x5bd1e995u;
    return h ^ (h >> 15u);
}

// `sketch1::dash_field`: a0 sampled at the nearest jittered stroke anchor
// under an elongated metric (exact: the anchor choice is discontinuous).
// p.i.x = seed, p.f = (c, s, cu, cv), p.g = (il, iw). b0 = value, b1 = random.
@compute @workgroup_size(16, 16)
fn dash(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let seed = u32(p.i.x);
    let c = p.f.x;
    let s = p.f.y;
    let cu = p.f.z;
    let cv = p.f.w;
    let il = p.g.x;
    let iw = p.g.y;
    let u16_ = 1.0 / 65536.0;
    let gx = gf(w.ox, i32(id.x));
    let gy = gf(w.oy, i32(id.y));
    let u = fadd(fmul(gx, c), fmul(gy, s));
    let v = fadd(fmul(-gx, s), fmul(gy, c));
    let iu = i32(floor(fdiv(u, cu)));
    let iv = i32(floor(fdiv(v, cv)));
    var bd = 3.4028235e38;
    var bu = 0.0;
    var bv = 0.0;
    var bh = 0.0;
    for (var dv = -1; dv <= 1; dv++) {
        for (var du = -1; du <= 1; du++) {
            let ku = iu + du;
            let kv = iv + dv;
            let h1 = mix32(ku, kv, seed);
            let h2 = mix32(ku, kv, seed ^ 0x9e3779b9u);
            let au = fmul(fadd(f32(ku), f32(h1 >> 16u) * u16_), cu);
            let av = fmul(fadd(f32(kv), f32(h1 & 0xffffu) * u16_), cv);
            let eu = fmul(fsub(u, au), il);
            let ev = fmul(fsub(v, av), iw);
            let d = fadd(fadd(fmul(eu, eu), fmul(ev, ev)), fmul(fmul(0.3, f32(h2 >> 16u)), u16_));
            if (d < bd) {
                bd = d;
                bu = au;
                bv = av;
                bh = f32(h2 & 0xffffu) * u16_;
            }
        }
    }
    let ax = bu * c - bv * s;
    let ay = bu * s + bv * c;
    let bx = ax / sf() - f32(w.ox) - 0.5;
    let by = ay / sf() - f32(w.oy) - 0.5;
    let i = widx(z, id.x, id.y);
    b0[i] = bilerp(z, bx, by);
    b1[i] = bh;
}

@compute @workgroup_size(16, 16)
fn chalk(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let k = p.f.x;
    let d = min(clamp((p.f.y - a0[i]) * 3.1 * k, 0.0, 1.0), select(1.0, 0.75, a1[i] > 0.75));
    let c = clamp((a2[i] - p.f.z) * 4.2 * k, 0.0, 1.0);
    b0[i] = 0.5 - 0.5 * d + 0.435 * c;
}

// a0 - a1 on (extended) planes.
@compute @workgroup_size(16, 16)
fn sub(@builtin(global_invocation_id) id: vec3<u32>) {
    let d = dims_s(id.z);
    if (id.x >= d.x || id.y >= d.y) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = a0[i] - a1[i];
}

@compute @workgroup_size(16, 16)
fn charcoal(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let ramp = clamp((0.5 * (a0[i] + a1[i]) - p.f.x) * 8.0, 0.0, 1.0);
    let amp = 0.02 + 0.25 * (1.0 - ramp);
    b0[i] = 0.22 + 0.78 * ramp + amp * (0.7 * (a2[i] / p.f.y) + 0.4 * (a3[i] / p.f.z));
}

@compute @workgroup_size(16, 16)
fn chrome(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    let i = widx(z, id.x, id.y);
    let gx = a0[wcl(z, x + 1, y)] - a0[wcl(z, x - 1, y)];
    let gy = a0[wcl(z, x, y + 1)] - a0[wcl(z, x, y - 1)];
    let lo = a1[i];
    let phi = p.f.y * (a0[i] - lo) + p.f.x * (-0.6 * gx - 0.8 * gy);
    let sc = p_sincos(phi);
    let c = (1.0 - sc.y) * 0.5;
    let base = 0.45 + (lo - 0.45) * 0.7 + 0.10 * sc.x;
    let e = (c - 0.73) / 0.05;
    b0[i] = base * (1.0 - 0.85 * sstep(0.85, 0.97, c)) + 0.4 * p_exp(-(e * e));
}

// `sketch1::value_noise`, exactly (Conté's drift is truncated).
fn vn_s(x: f32, y: f32, seed: u32) -> f32 {
    let xi = floor(x);
    let yi = floor(y);
    var tx = fsub(x, xi);
    var ty = fsub(y, yi);
    tx = fmul(fmul(tx, tx), fsub(3.0, fmul(2.0, tx)));
    ty = fmul(fmul(ty, ty), fsub(3.0, fmul(2.0, ty)));
    let h00 = hash01(i32(xi), i32(yi), seed);
    let h10 = hash01(i32(xi + 1.0), i32(yi), seed);
    let h01 = hash01(i32(xi), i32(yi + 1.0), seed);
    let h11 = hash01(i32(xi + 1.0), i32(yi + 1.0), seed);
    let a0_ = fadd(h00, fmul(fsub(h10, h00), tx));
    let a1_ = fadd(h01, fmul(fsub(h11, h01), tx));
    return fadd(a0_, fmul(fsub(a1_, a0_), ty));
}

fn row_noise(x: f32, y: i32, seed: u32) -> f32 {
    let xi = floor(x);
    var t = x - xi;
    t = t * t * (3.0 - 2.0 * t);
    let a = hash01(i32(xi), y, seed);
    let b = hash01(i32(xi) + 1, y, seed);
    return a + (b - a) * t;
}

fn conte_canvas(x: f32, y: f32) -> f32 {
    let yi = i32(floor(y));
    let drift = i32(fmul(vn_s(fdiv(x, 150.0), fdiv(y, 150.0), 5u), 3.0));
    let ridge = ((yi + drift) % 3 + 3) % 3 == 0;
    let n = row_noise(x / 5.0, yi, 3u) * 0.7 + row_noise(x / 1.75, yi, 4u) * 0.3;
    return clamp(0.45 + 2.0 * (n - 0.5) + select(0.0, 0.6, ridge), 0.0, 1.0);
}

fn conte_tone(v: f32, tv: f32) -> f32 {
    let d = clamp((v - (p.f.x - 0.444 * tv)) * 3.0, 0.0, 1.0);
    let b = clamp((v - (p.f.y + 0.25 * tv)) * 6.375, 0.0, 1.0);
    return 0.5 * d + 0.5 * b;
}

// Conté Crayon: p.i = (kind, sub), p.f = (cd, cb, step).
@compute @workgroup_size(16, 16)
fn conte(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let v = a0[i];
    if (p.i.x == 0) {
        let gx = f32(w.ox + i32(id.x)) * sf();
        let gy = f32(w.oy + i32(id.y)) * sf();
        var acc = 0.0;
        for (var k = 0; k < p.i.y; k++) {
            let o_ = (f32(k) + 0.5) * p.f.z;
            acc += conte_tone(v, conte_canvas(gx + o_, gy + o_));
        }
        b0[i] = acc / f32(p.i.y);
    } else {
        let tex = texture_h(gf(w.ox, i32(id.x)), gf(w.oy, i32(id.y)), p.i.x - 1);
        b0[i] = conte_tone(v, clamp(tex * 1.6 - 0.3, 0.0, 1.0));
    }
}

// 3-tap box (horizontal if p.i.x == 1), edge clamped, times p.f.x.
@compute @workgroup_size(16, 16)
fn box3(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    var a: f32;
    var c: f32;
    if (p.i.x == 1) {
        a = a0[wcl(z, x - 1, y)];
        c = a0[wcl(z, x + 1, y)];
    } else {
        a = a0[wcl(z, x, y - 1)];
        c = a0[wcl(z, x, y + 1)];
    }
    let i = widx(z, id.x, id.y);
    b0[i] = fmul(fadd(fadd(a, a0[i]), c), p.f.x);
}

fn shape(c: f32) -> f32 {
    let m = p_pow(abs(c), 0.726);
    return select(m, -m, c < 0.0);
}

fn hpavg(lo: i32, ph: f32, period: f32) -> f32 {
    let s = i32(max(p.s, 1u));
    var acc = 0.0;
    for (var k = 0; k < s; k++) {
        acc += shape(p_cos(TAU_S * (f32(lo + k) - ph) / period));
    }
    return acc / f32(s);
}

// Halftone Pattern: p.i.x = kind, p.f = (period, kc, beta, v0), p.g = (ca, cb, piv).
@compute @workgroup_size(16, 16)
fn halftone(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let period = p.f.x;
    let s = i32(max(p.s, 1u));
    let x0 = (w.ox + i32(id.x)) * s;
    let y0 = (w.oy + i32(id.y)) * s;
    var q: f32;
    if (p.i.x == 1) {
        let gx = gf(w.ox, i32(id.x)) - 4096.0;
        let gy = gf(w.oy, i32(id.y)) - 4096.0;
        q = shape(p_cos(TAU_S * sqrt(gx * gx + gy * gy) / period));
    } else if (p.i.x == 2) {
        q = hpavg(y0, 1.0, period);
    } else {
        q = hpavg(x0, 4.0, period) * hpavg(y0, 1.0, period);
    }
    let piv = p.g.z;
    let d = max((a0[i] + p.f.z * q - p.f.w - piv) * p.f.y + piv, 0.0);
    b0[i] = min(p.g.x * d + p.g.y * d * d, 1.0);
}

@compute @workgroup_size(16, 16)
fn note_mask(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = clamp((a0[i] - p.f.x) * p.f.y + 0.5, 0.0, 1.0);
}

@compute @workgroup_size(16, 16)
fn note_paper(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let up = widx(z, id.x, select(id.y, id.y - 1u, id.y > 0u));
    let dn = widx(z, id.x, select(id.y, id.y + 1u, id.y + 1u < w.h));
    b0[i] = (170.0 + 72.0 * a0[i]) / 255.0 + (a1[dn] - a1[up]) * p.f.x + (a2[up] - a2[dn]) * p.f.y;
}

@compute @workgroup_size(16, 16)
fn photocopy(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = 1.0 - p.f.x * (a0[i] - a1[i] - p.f.y);
}

@compute @workgroup_size(16, 16)
fn smooth_th(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = sstep(p.f.x, p.f.y, a0[i]);
}

// `gallery::sobel` (b0 = gx, b1 = gy).
@compute @workgroup_size(16, 16)
fn sobel(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = i32(id.x);
    let y = i32(id.y);
    let at = a0[wcl(z, x - 1, y - 1)];
    let bt = a0[wcl(z, x, y - 1)];
    let ct = a0[wcl(z, x + 1, y - 1)];
    let al = a0[wcl(z, x - 1, y)];
    let cr = a0[wcl(z, x + 1, y)];
    let ab = a0[wcl(z, x - 1, y + 1)];
    let bb = a0[wcl(z, x, y + 1)];
    let cb = a0[wcl(z, x + 1, y + 1)];
    let i = widx(z, id.x, id.y);
    b0[i] = (ct + 2.0 * cr + cb) - (at + 2.0 * al + ab);
    b1[i] = (ab + 2.0 * bb + cb) - (at + 2.0 * bt + ct);
}

// Plaster: a0 = m, a1 = gx, a2 = gy, a3 = plateau. p.f = (k, lv), p.g = (hv,
// sp0), p.h = (spec, cos, sin, half), p.k = (doc w, doc h, mid).
@compute @workgroup_size(16, 16)
fn plaster(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let lv = p.f.yzw;
    let hv = p.g.xyz;
    let nx = a1[i] * p.f.x;
    let ny = a2[i] * p.f.x;
    let r = 1.0 / sqrt(nx * nx + ny * ny + 1.0);
    let n = vec3<f32>(nx * r, ny * r, r);
    let diff = max(dot(n, lv), 0.0);
    let sd = max(dot(n, hv), 0.0);
    let sp = sd * sd * sd;
    let x = gf(w.ox, i32(id.x));
    let y = gf(w.oy, i32(id.y));
    let dw = p.k.x;
    let dh = p.k.y;
    let mid = p.k.z;
    let ramp = clamp(0.5 + 0.5 * ((x - dw * 0.5) * p.h.y - (y - dh * 0.5) * p.h.z) / p.h.w, 0.0, 1.0);
    let level = mid + (ramp - mid) * sstep(0.85, 0.98, a3[i]);
    let flat = level / lv.z;
    b0[i] = a0[i] * diff * flat + p.h.x * max(sp - p.g.w, 0.0);
}

@compute @workgroup_size(16, 16)
fn reticulation(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let t = sstep(0.37, 0.55, a0[i]);
    let bd = p.f.y;
    let bh = p.f.z;
    b0[i] = bd + (bh - bd) * t + (a1[i] - a2[i]) * p.f.x * (p.f.w + (p.g.x - p.f.w) * t);
}

@compute @workgroup_size(16, 16)
fn add_scaled(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = a0[i] + a1[i] * p.f.x;
}

// (a0 + a1 * amp - t) * gain + 0.5; p.f = (t, gain, amp).
@compute @workgroup_size(16, 16)
fn ramp(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = (a0[i] + a1[i] * p.f.z - p.f.x) * p.f.y + 0.5;
}

// Water Paper: 0.1 / 0.9 fibre direction per 4 px cell.
@compute @workgroup_size(16, 16)
fn wp_sel(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let cx = (g0(w.ox, i32(id.x)) + 1) >> 2u;
    let cy = (g0(w.oy, i32(id.y)) + 1) >> 2u;
    b0[widx(id.z, id.x, id.y)] = select(0.9, 0.1, hash01(cx, cy, 191u) < 0.5);
}

@compute @workgroup_size(16, 16)
fn wp_mix(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = a0[i] * a2[i] + a1[i] * (1.0 - a2[i]);
}

// `sketch2::cells4`: mean over the 4 x 4 paper cell (level 0, clipped to the window).
@compute @workgroup_size(16, 16)
fn cells4(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let cx = (w.ox + i32(id.x) + 1) >> 2u;
    let cy = (w.oy + i32(id.y) + 1) >> 2u;
    let x0 = max(0, 4 * cx - 1 - w.ox);
    let x1 = min(i32(w.w), 4 * cx + 3 - w.ox);
    let y0 = max(0, 4 * cy - 1 - w.oy);
    let y1 = min(i32(w.h), 4 * cy + 3 - w.oy);
    var s = 0.0;
    for (var y = y0; y < y1; y++) {
        var r = 0.0;
        for (var x = x0; x < x1; x++) {
            r += a0[widx(z, u32(x), u32(y))];
        }
        s += r;
    }
    b0[widx(z, id.x, id.y)] = s / f32((y1 - y0) * (x1 - x0));
}

fn fibers(x: f32, y: f32, len: f32) -> f32 {
    let xi = i32(floor(x));
    let yi = i32(floor(y));
    var v = 0.0;
    if (((yi + 1) & 3) == 0) {
        let j = (yi + 1) >> 2u;
        let seg = i32(floor(fdiv(fadd(x, fmul(hash01(j, 0, 171u), len)), len)));
        if (hash01(seg, j, 173u) < 0.35) {
            v = max(v, 0.5 + 0.5 * hash01(seg, j, 175u));
        }
    }
    if (((xi + 1) & 3) == 0) {
        let j = (xi + 1) >> 2u;
        let seg = i32(floor(fdiv(fadd(y, fmul(hash01(0, j, 177u), len)), len)));
        if (hash01(j, seg, 179u) < 0.35) {
            v = max(v, 0.5 + 0.5 * hash01(j, seg, 181u));
        }
    }
    return v;
}

// Water Paper finish: p.f = (k, brightness, fibre amount, fibre length).
@compute @workgroup_size(16, 16)
fn water_paper(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let c = vec3<f32>(a0[i], a1[i], a2[i]);
    let l = 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
    let lo = max((l - 0.5) * p.f.x + 0.5 + p.f.y, 0.0);
    let fib = fibers(gf(w.ox, i32(id.x)), gf(w.oy, i32(id.y)), p.f.w);
    let g = lo / max(l, 1e-3) * (1.0 + fib * p.f.z);
    emit3(id, i, c * g);
}
