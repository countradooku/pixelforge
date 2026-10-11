
// ----- Filter Gallery: Brush Strokes and Glowing Edges stages ---------------------

const TAU_B: f32 = 0x1.921fb6p+2f;

fn put3b(i: u32, c: vec3<f32>) {
    b0[i] = c.x;
    b1[i] = c.y;
    b2[i] = c.z;
}

// Accented Edges: V' = V + gain * range, hue and saturation kept.
@compute @workgroup_size(16, 16)
fn accent(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let c = vec3<f32>(a0[i], a1[i], a2[i]);
    let v0 = a3[i];
    let v1 = clamp(v0 + p.f.x * a4[i], 0.0, 1.0);
    if (v0 > 1e-4) {
        emit3(id, i, c * (v1 / v0));
    } else {
        emit3(id, i, vec3<f32>(v1));
    }
}

@compute @workgroup_size(16, 16)
fn pick_th(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = select(a2[i], a1[i], a0[i] > p.f.x);
}

@compute @workgroup_size(16, 16)
fn crosshatch(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let dark = 1.0 - sstep(p.f.w, p.g.x, a4[i]);
    b0[i] = 0.5 * (a1[i] + a3[i]) + p.f.x * (a0[i] - a1[i]) + p.f.y * dark * (a2[i] - a3[i]) - p.f.z;
}

fn pwl(n: i32, x: f32) -> f32 {
    if (x <= tabf(t0[0])) {
        return tabf(t0[1]);
    }
    for (var k = 0; k < n - 1; k++) {
        let x0 = tabf(t0[2 * k]);
        let y0 = tabf(t0[2 * k + 1]);
        let x1 = tabf(t0[2 * k + 2]);
        let y1 = tabf(t0[2 * k + 3]);
        if (x <= x1) {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    return tabf(t0[2 * n - 1]);
}

// `brush::ratio_tone`: the HSB brightness through a curve, color rescaled.
@compute @workgroup_size(16, 16)
fn ratio_tone(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let c = vec3<f32>(a0[i], a1[i], a2[i]);
    let v = max(max(c.x, c.y), c.z);
    var v1: f32;
    if (p.i.x == 2) {
        v1 = sstep(p.f.x, p.f.y, v);
    } else {
        var y: f32;
        var lo_k: f32;
        var hi_k: f32;
        if (p.i.x == 0) {
            y = pwl(p.i.y, (v - p.f.x) * 255.0) / 255.0;
        } else {
            y = pwl(p.i.y, v * 250.0) / 255.0;
        }
        let vc = clamp(v, 0.0, 1.0);
        if (y < vc) {
            v1 = vc + (y - vc) * p.f.y;
        } else {
            v1 = vc + (y - vc) * p.f.z;
        }
        v1 = clamp(v1, 0.0, 1.0);
    }
    if (v > 1e-4) {
        emit3(id, i, c * (v1 / v));
    } else {
        emit3(id, i, vec3<f32>(v1));
    }
}

// Ink Outlines: mean of the pixels s up-left (x - s, y) and (x - s, y - s).
@compute @workgroup_size(16, 16)
fn ink_shift(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let s = u32(p.i.x);
    let xs = select(0u, id.x - s, id.x >= s);
    let yb = select(0u, id.y - s, id.y >= s);
    b0[widx(z, id.x, id.y)] = 0.5 * (a0[widx(z, xs, id.y)] + a0[widx(z, xs, yb)]);
}

// Mean of n samples along "\" (edge clamped per axis).
@compute @workgroup_size(16, 16)
fn dir_mean(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let r = p.i.x / 2;
    var s = 0.0;
    for (var t = -r; t <= r; t++) {
        s += a0[wcl(z, i32(id.x) + t, i32(id.y) + t)];
    }
    b0[widx(z, id.x, id.y)] = s / p.f.x;
}

@compute @workgroup_size(16, 16)
fn lin2(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = 2.0 * a0[i] - a1[i];
}

// Spatter: every cell (offset rows) moves by a whole level-0 pixel offset
// drawn from the disc of radius p.f.x (= r / 2).
@compute @workgroup_size(16, 16)
fn spray(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let seed = u32(p.i.x);
    let cell = p.f.y;
    let gx = gf(w.ox, i32(id.x));
    let gy = gf(w.oy, i32(id.y));
    let cy = i32(floor(fdiv(gy, cell)));
    let cx = i32(floor(fdiv(fadd(gx, fmul(cell, hash01(0, cy, seed + 2u))), cell)));
    let a = fmul(TAU_B, hash01(cx, cy, seed));
    let d = fmul(p.f.x, fsqrt(hash01(cx, cy, seed + 1u)));
    let sc = p_sincos(a);
    let i = widx(z, id.x, id.y);
    b0[i] = rround(fmul(d, sc.y)) / sf();
    b1[i] = rround(fmul(d, sc.x)) / sf();
}

// Sprayed Strokes: stroke segments along (p.i.x, p.i.y) shifted by a random
// whole number of level-0 pixels.
@compute @workgroup_size(16, 16)
fn sprayed(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = g0(w.ox, i32(id.x));
    let y = g0(w.oy, i32(id.y));
    let dx = p.i.x;
    let dy = p.i.y;
    var line: i32;
    var along: i32;
    if (dx == 1 && dy == 1) {
        line = x - y;
        along = y;
    } else if (dx == 1 && dy == -1) {
        line = x + y;
        along = y;
    } else if (dx == 1 && dy == 0) {
        line = y;
        along = x;
    } else {
        line = x;
        along = y;
    }
    let band = select(line / 3, (line - 2) / 3, line < 0);
    let len = p.f.x;
    let off = fmul(len, hash01(band, 0, 89u));
    let seg = i32(floor(fdiv(fadd(f32(along), off), len)));
    let t = rround(fmul(fmul(fsub(hash01(band, seg, 91u), 0.5), 1.2), p.f.y));
    let i = widx(z, id.x, id.y);
    b0[i] = t * f32(dx) / sf();
    b1[i] = t * f32(dy) / sf();
}
