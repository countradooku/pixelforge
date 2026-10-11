
// ----- Filter Gallery: Distort and Texture stages --------------------------------------

const TAU_T: f32 = 0x1.921fb6p+2f;

fn put3t(i: u32, c: vec3<f32>) {
    b0[i] = c.x;
    b1[i] = c.y;
    b2[i] = c.z;
}

// Glass / Ocean Ripple height field: fbm of level-0 coordinates / cell
// (ridged |2n - 1| for Glass). p.i = (octaves, glass), p.f.x = cell.
@compute @workgroup_size(16, 16)
fn bump(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = gf(w.ox, i32(id.x));
    let y = gf(w.oy, i32(id.y));
    let n = fbm_x(fdiv(x, p.f.x), fdiv(y, p.f.x), u32(p.i.x), 101u);
    b0[widx(id.z, id.x, id.y)] = select(n, abs(fsub(fmul(2.0, n), 1.0)), p.i.y == 1);
}

// Displacement from the height gradient (one-sided at the window edge).
@compute @workgroup_size(16, 16)
fn bump_grad(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let xa = select(0u, id.x - 1u, id.x > 0u);
    let xb = min(id.x + 1u, w.w - 1u);
    let ya = select(0u, id.y - 1u, id.y > 0u);
    let yb = min(id.y + 1u, w.h - 1u);
    let gx = fdiv(fsub(a0[widx(z, xb, id.y)], a0[widx(z, xa, id.y)]), f32(max(xb - xa, 1u)));
    let gy = fdiv(fsub(a0[widx(z, id.x, yb)], a0[widx(z, id.x, ya)]), f32(max(yb - ya, 1u)));
    let i = widx(z, id.x, id.y);
    b0[i] = fdiv(fadd(fmul(gx, p.f.x), p.f.y), sf());
    b1[i] = fdiv(fadd(fmul(gy, p.f.x), p.f.z), sf());
}

// ----- jittered lattice (`texture::Lattice`) -----------------------------------------

fn seed_pt(cx: i32, cy: i32, cell: f32, jit: f32, seed: u32) -> vec2<f32> {
    let hx = hash01(cx, cy, seed);
    let hy = hash01(cx, cy, seed + 1u);
    return vec2<f32>(fmul(fadd(fadd(f32(cx), 0.5), fmul(jit, fsub(hx, 0.5))), cell), fmul(fadd(fadd(f32(cy), 0.5), fmul(jit, fsub(hy, 0.5))), cell));
}

struct Near {
    a: vec2<f32>,
    edge: f32,
}

// Nearest seed (exact: the choice is discontinuous) and the distance to the
// nearest cell border.
fn nearest(x: f32, y: f32, cell: f32, jit: f32, seed: u32) -> Near {
    let ix = i32(floor(fdiv(x, cell)));
    let iy = i32(floor(fdiv(y, cell)));
    var pts: array<vec2<f32>, 9>;
    var best = 0;
    var bd = 3.4028235e38;
    for (var k = 0; k < 9; k++) {
        let q = seed_pt(ix + k % 3 - 1, iy + k / 3 - 1, cell, jit, seed);
        let dx = fsub(q.x, x);
        let dy = fsub(q.y, y);
        let d = fadd(fmul(dx, dx), fmul(dy, dy));
        if (d < bd) {
            bd = d;
            best = k;
        }
        pts[k] = q;
    }
    let a = pts[best];
    var edge = 3.4028235e38;
    for (var k = 0; k < 9; k++) {
        if (k == best) {
            continue;
        }
        let b = pts[k];
        let v = b - a;
        let len = max(sqrt(v.x * v.x + v.y * v.y), 1e-6);
        edge = min(edge, (((a.x + b.x) * 0.5 - x) * v.x + ((a.y + b.y) * 0.5 - y) * v.y) / len);
    }
    var r: Near;
    r.a = a;
    r.edge = edge;
    return r;
}

// Craquelure's crack distance: p.i.x = seed, p.f = (cell, jitter, wl, warp).
@compute @workgroup_size(16, 16)
fn craquelure_dist(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = gf(w.ox, i32(id.x));
    let y = gf(w.oy, i32(id.y));
    let cell = p.f.x;
    let wl = p.f.z;
    let m = p.f.w;
    let xx = x + m * (2.0 * fbm(x / wl, y / wl, 2u, 411u) - 1.0);
    let yy = y + m * (2.0 * fbm(x / wl + 13.1, y / wl + 7.7, 2u, 413u) - 1.0);
    let fade = (fbm(x / (2.0 * wl), y / (2.0 * wl), 2u, 417u) - 0.45) * 1.2;
    let n = nearest(xx, yy, cell, p.f.y, u32(p.i.x));
    b0[widx(id.z, id.x, id.y)] = max(n.edge, 0.0) + max(fade, 0.0) * cell;
}

// Mosaic Tiles' grout distance: p.f = (tile, wl, wobble).
@compute @workgroup_size(16, 16)
fn mosaic_dist(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = gf(w.ox, i32(id.x));
    let y = gf(w.oy, i32(id.y));
    let tile = p.f.x;
    let wl = p.f.y;
    let wob = p.f.z;
    let yy = y / tile + wob * (2.0 * fbm(x / wl, y / wl, 2u, 301u) - 1.0);
    let row = floor(yy);
    let xx = x / tile + hash01(i32(row), 0, 307u) + 0.5 * wob * (2.0 * fbm(x / wl + 31.7, y / wl - 7.3, 2u, 303u) - 1.0);
    let dy = abs(yy - row - 0.5);
    let dx = abs(xx - floor(xx) - 0.5);
    b0[widx(id.z, id.x, id.y)] = (0.5 - max(dy, dx)) * tile;
}

@compute @workgroup_size(16, 16)
fn groove_height(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    b0[i] = -max(p.f.x - a0[i], 0.0) + a1[i] * p.f.y;
}

// `Grooves::apply`: a3 = height, a4 = distance; p.f = (k, side, r, floor),
// p.g = (dark, bright, gain).
@compute @workgroup_size(16, 16)
fn grooves(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(z, id.x, id.y);
    let xa = select(0u, id.x - 1u, id.x > 0u);
    let xb = min(id.x + 1u, w.w - 1u);
    let ya = select(0u, id.y - 1u, id.y > 0u);
    let yb = min(id.y + 1u, w.h - 1u);
    let gy = (a3[widx(z, id.x, yb)] - a3[widx(z, id.x, ya)]) * p.f.x;
    let gx = (a3[widx(z, xb, id.y)] - a3[widx(z, xa, id.y)]) * p.f.x;
    let s = clamp(gy + p.f.y * gx, -1.0, 1.0);
    let f = clamp((1.0 - a4[i] / p.f.z) * 2.0, 0.0, 1.0);
    let lit = select(1.0 + s * p.g.y, 1.0 + s * p.g.x, s < 0.0);
    let m = p.g.z * (1.0 - f * p.f.w) * lit;
    emit3(id, i, vec3<f32>(a0[i], a1[i], a2[i]) * m);
}

// ----- Patchwork ---------------------------------------------------------------

struct Grid {
    bx0: i32,
    by0: i32,
    nbx: i32,
    nby: i32,
}

fn pw_grid(w: Win, size: f32) -> Grid {
    let s = sf();
    var g: Grid;
    let ex0 = f32(w.ox) * s;
    let ey0 = f32(w.oy) * s;
    let ex1 = f32(w.ox + i32(w.w)) * s - 1e-3;
    let ey1 = f32(w.oy + i32(w.h)) * s - 1e-3;
    g.bx0 = i32(floor(fdiv(ex0, size)));
    g.by0 = i32(floor(fdiv(ey0, size)));
    g.nbx = i32(floor(fdiv(ex1, size))) - g.bx0 + 1;
    g.nby = i32(floor(fdiv(ey1, size))) - g.by0 + 1;
    return g;
}

fn pw_cell(o_: i32, x: i32, size: f32) -> i32 {
    return i32(floor(fdiv(gf(o_, x), size)));
}

// Mean color of each grid cell (thread per cell), screened by (2 - max).
// Stored at cell (i, j) of the window's plane (planes are extended by 1).
@compute @workgroup_size(16, 16)
fn pw_means(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    let size = p.f.x;
    let g = pw_grid(w, size);
    let ci = i32(id.x);
    let cj = i32(id.y);
    if (ci >= g.nbx || cj >= g.nby) {
        return;
    }
    let cx = g.bx0 + ci;
    let cy = g.by0 + cj;
    let s = sf();
    // Pixel spans whose centers fall in the cell.
    var x0 = max(i32(floor(f32(cx) * size / s)) - w.ox - 2, 0);
    while (x0 < i32(w.w) && pw_cell(w.ox, x0, size) < cx) {
        x0++;
    }
    var x1 = x0;
    while (x1 < i32(w.w) && pw_cell(w.ox, x1, size) == cx) {
        x1++;
    }
    var y0 = max(i32(floor(f32(cy) * size / s)) - w.oy - 2, 0);
    while (y0 < i32(w.h) && pw_cell(w.oy, y0, size) < cy) {
        y0++;
    }
    var y1 = y0;
    while (y1 < i32(w.h) && pw_cell(w.oy, y1, size) == cy) {
        y1++;
    }
    var e = vec3<f32>(0.0);
    var cnt = 0.0;
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            let j = widx(z, u32(x), u32(y));
            e = vec3<f32>(fadd(e.x, a0[j]), fadd(e.y, a1[j]), fadd(e.z, a2[j]));
            cnt += 1.0;
        }
    }
    var m: vec3<f32>;
    if (cnt > 0.0) {
        m = e / cnt;
    } else {
        let px = clamp(i32((f32(cx) + 0.5) * size / s) - w.ox, 0, i32(w.w) - 1);
        let py = clamp(i32((f32(cy) + 0.5) * size / s) - w.oy, 0, i32(w.h) - 1);
        let j = widx(z, u32(px), u32(py));
        m = vec3<f32>(a0[j], a1[j], a2[j]);
    }
    let k = 2.0 - max(max(m.x, m.y), m.z);
    put3t(widx(z, id.x, id.y), m * k);
}

fn pw_at(z: u32, g: Grid, x: f32, y: f32, di: i32, dj: i32, size: f32) -> vec3<f32> {
    let i = clamp(i32(floor(fdiv(x, size))) - g.bx0 + di, 0, g.nbx - 1);
    let j = clamp(i32(floor(fdiv(y, size))) - g.by0 + dj, 0, g.nby - 1);
    let k = widx(z, u32(i), u32(j));
    return vec3<f32>(a0[k], a1[k], a2[k]);
}

fn pw_eval(z: u32, g: Grid, gx: f32, gy: f32, size: f32) -> vec3<f32> {
    let r = i32(fsub(gy, fmul(floor(fdiv(gy, size)), size)));
    let c = i32(fsub(gx, fmul(floor(fdiv(gx, size)), size)));
    var row = 3;
    if (r >= 0 && r <= 2) {
        row = r;
    } else if (r >= p.i.y) {
        row = 4;
    }
    let col = select(2, c, c < 2);
    let t = u32(row * 3 + col) * 5u;
    let s = pw_at(z, g, gx, gy, 0, 0, size);
    let u = pw_at(z, g, gx, gy, 0, -1, size);
    let d = pw_at(z, g, gx, gy, 0, 1, size);
    let l = pw_at(z, g, gx, gy, -1, 0, size);
    return tabf(t0[t]) * s + tabf(t0[t + 1u]) * u + tabf(t0[t + 2u]) * d + tabf(t0[t + 3u]) * l + vec3<f32>(tabf(t0[t + 4u]));
}

// Patchwork: p.i = (samples per axis, last row), p.f.x = size; a0..a2 = means.
@compute @workgroup_size(16, 16)
fn patchwork(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let size = p.f.x;
    let g = pw_grid(w, size);
    let n = p.i.x;
    let s = sf();
    let x0 = f32(w.ox + i32(id.x)) * s;
    let y0 = f32(w.oy + i32(id.y)) * s;
    let step = s / f32(n);
    var acc = vec3<f32>(0.0);
    for (var sy = 0; sy < n; sy++) {
        for (var sx = 0; sx < n; sx++) {
            acc += pw_eval(z, g, x0 + (f32(sx) + 0.5) * step, y0 + (f32(sy) + 0.5) * step, size);
        }
    }
    emit3(id, widx(z, id.x, id.y), acc / f32(n * n));
}

// Stained Glass: p.i.x = seed; p.f = (cell, jitter, lead half width, lift),
// p.g = (aa, light, doc w, doc h), p.h = (lead rgb, min side / 2).
@compute @workgroup_size(16, 16)
fn stained_glass(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = gf(w.ox, i32(id.x));
    let y = gf(w.oy, i32(id.y));
    let n = nearest(x, y, p.f.x, p.f.y, u32(p.i.x));
    let sx = n.a.x;
    let sy = n.a.y;
    let bx = u32(clamp(i32(floor(fsub(sx / sf(), f32(w.ox)))), 0, i32(w.w) - 1));
    let by = u32(clamp(i32(floor(fsub(sy / sf(), f32(w.oy)))), 0, i32(w.h) - 1));
    let j = widx(z, bx, by);
    let aa = p.g.x;
    let dw = p.g.z;
    let dh = p.g.w;
    let ms = p.h.w;
    let rr = ((sx - dw * 0.5) * (sx - dw * 0.5) + (sy - dh * 0.5) * (sy - dh * 0.5)) / (ms * ms);
    var gain = 1.0;
    if (rr < 1.0) {
        gain = 1.0 + 0.233 * p.g.y * p_pow(1.0 - sqrt(rr), 1.2);
    }
    let edge = min(min(x, y), min(dw - x, dh - y));
    let a2_ = max(aa, 0.5);
    var frame = 1.0;
    if (edge < 3.0 * a2_ + 1.14) {
        frame = clamp((edge - 1.14) / a2_ * 0.5, 0.0, 1.0);
    }
    let c = min((vec3<f32>(a0[j], a1[j], a2[j]) * 0.935 + p.f.w) * gain, vec3<f32>(1.0));
    let t = sstep(p.f.z - aa, p.f.z + aa, n.edge) * frame;
    emit3(id, widx(z, id.x, id.y), mix3(p.h.xyz, c, t));
}

// ----- Texturizer --------------------------------------------------------------

fn canvas_slow(x: f32, y: f32) -> vec2<f32> {
    return vec2<f32>(fbm(x / 41.0, y / 29.0, 2u, 501u), fbm(x / 23.0, y / 31.0, 1u, 505u));
}

fn canvas_height_with(x: f32, y: f32, wob: f32, wav: f32) -> f32 {
    let v = y / 3.0 + 0.6 * (wob - 0.5);
    let row = floor(v);
    let f = v - row;
    let q = 2.0 * f - 1.0;
    let profile = 1.0 - q * q;
    let dash = fbm(x / 4.5 + row * 7.31, row * 0.37, 2u, 503u);
    let warp = 0.5 + 0.5 * p_sin(TAU_T * (x / 4.0 + 0.3 * wav));
    return profile * max(0.6 + 1.4 * (dash - 0.5), 0.2) + 0.1 * warp + 0.2 * hash01(i32(floor(x)), i32(floor(y)), 507u);
}

// Canvas heights at level 0 through the slow-wobble lattice: p.i.x = step,
// p.f = (scale, 1 / step).
@compute @workgroup_size(16, 16)
fn canvas_hgt(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let step = p.i.x;
    let scale = p.f.x;
    let px = w.ox + i32(id.x);
    let py = w.oy + i32(id.y);
    let cx = select((px - step + 1) / step, px / step, px >= 0);
    let cy = select((py - step + 1) / step, py / step, py >= 0);
    let fx = f32(px - cx * step) * p.f.y;
    let fy = f32(py - cy * step) * p.f.y;
    let l00 = canvas_slow((f32(cx * step) + 0.5) / scale, (f32(cy * step) + 0.5) / scale);
    let l10 = canvas_slow((f32((cx + 1) * step) + 0.5) / scale, (f32(cy * step) + 0.5) / scale);
    let l01 = canvas_slow((f32(cx * step) + 0.5) / scale, (f32((cy + 1) * step) + 0.5) / scale);
    let l11 = canvas_slow((f32((cx + 1) * step) + 0.5) / scale, (f32((cy + 1) * step) + 0.5) / scale);
    let t = l00 + (l10 - l00) * fx;
    let b = l01 + (l11 - l01) * fx;
    let s = t + (b - t) * fy;
    b0[widx(id.z, id.x, id.y)] = canvas_height_with(fdiv(f32(px) + 0.5, scale), fdiv(f32(py) + 0.5, scale), s.x, s.y);
}

@compute @workgroup_size(16, 16)
fn canvas_shade(@builtin(global_invocation_id) id: vec3<u32>) {
    let z = id.z;
    let w = win[z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let ya = select(0u, id.y - 1u, id.y > 0u);
    let yb = min(id.y + 1u, w.h - 1u);
    b0[widx(z, id.x, id.y)] = (a0[widx(z, id.x, ya)] - a0[widx(z, id.x, yb)]) * p.f.x * 2.0 / f32(max(yb - ya, 1u));
}

fn canvas_height(x: f32, y: f32) -> f32 {
    let s = canvas_slow(x, y);
    return canvas_height_with(x, y, s.x, s.y);
}

// Preview levels: the level-0 shading averaged over n x n samples.
// p.i.x = n, p.f = (scale, step, k).
@compute @workgroup_size(16, 16)
fn canvas_sampled(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let n = p.i.x;
    let scale = p.f.x;
    let step = p.f.y;
    let x0 = f32(w.ox + i32(id.x)) * sf();
    let y0 = f32(w.oy + i32(id.y)) * sf();
    var acc = 0.0;
    for (var sy = 0; sy < n; sy++) {
        for (var sx = 0; sx < n; sx++) {
            let x = fdiv(x0 + (f32(sx) + 0.5) * step, scale);
            let y = fdiv(y0 + (f32(sy) + 0.5) * step, scale);
            acc += canvas_height(x, y - 1.0 / scale) - canvas_height(x, y + 1.0 / scale);
        }
    }
    b0[widx(id.z, id.x, id.y)] = acc * p.f.z / f32(n * n);
}

// `gallery::texture` at level-0 coordinates / p.f.x.
@compute @workgroup_size(16, 16)
fn texture(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let x = fdiv(gf(w.ox, i32(id.x)), p.f.x);
    let y = fdiv(gf(w.oy, i32(id.y)), p.f.x);
    b0[widx(id.z, id.x, id.y)] = texture_h(x, y, p.i.x);
}

// (relief - 0.5) / 2 of a Sobel gradient (a0, a1) lit along (p.f.x, p.f.y).
@compute @workgroup_size(16, 16)
fn relief_shade(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    let v = clamp(0.5 + (a0[i] * p.f.x + a1[i] * p.f.y) * p.f.z, 0.0, 1.0);
    b0[i] = (v - 0.5) * 0.5;
}

@compute @workgroup_size(16, 16)
fn add_shade(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = win[id.z];
    if (id.x >= w.w || id.y >= w.h) {
        return;
    }
    let i = widx(id.z, id.x, id.y);
    emit3(id, i, vec3<f32>(a0[i], a1[i], a2[i]) + vec3<f32>(a3[i]));
}
