// Edit ▸ Perspective Warp (`engine::perspwarp::render_cpu`): each output
// pixel interpolates its mesh cell's source positions (a node table built
// on the CPU) and samples the source bicubically (Catmull-Rom, transparent
// outside the source). Every float step is the CPU's expression with `nb`
// barriers (Metal's fast math can't fuse or reassociate across them), so
// the result is bit-identical.

struct P {
    // Band origin (output pixels) and size.
    ox: u32,
    oy: u32,
    w: u32,
    h: u32,
    // Mesh nodes per row.
    nx: u32,
    // Source: stride (tile-area width) and image size.
    stride: u32,
    iw: u32,
    ih: u32,
    // Always 0 (opaque to the compiler).
    zero: u32,
    _0: u32,
    _1: u32,
    _2: u32,
}

const INVALID: f32 = -1.0e30;

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read> nodes: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

fn nb(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ p.zero);
}

fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    return nb(a + nb(nb(b - a) * t));
}

fn texel(x: i32, y: i32) -> u32 {
    if (x < 0 || y < 0 || x >= i32(p.iw) || y >= i32(p.ih)) {
        return 0u;
    }
    return src[u32(y) * p.stride + u32(x)];
}

// `perspwarp::cubic_weights`.
fn cubic(f: f32) -> vec4<f32> {
    let t2 = nb(f * f);
    let t3 = nb(t2 * f);
    let w0 = nb(nb(nb(-t3 + nb(2.0 * t2)) - f) * 0.5);
    let w1 = nb(nb(nb(nb(3.0 * t3) - nb(5.0 * t2)) + 2.0) * 0.5);
    let w2 = nb(nb(nb(nb(-3.0 * t3) + nb(4.0 * t2)) + f) * 0.5);
    let w3 = nb(nb(t3 - t2) * 0.5);
    return vec4<f32>(w0, w1, w2, w3);
}

fn snap(v: f32) -> f32 {
    let r = rnd(v);
    if (abs(nb(v - r)) < 1e-3) {
        return r;
    }
    return v;
}

// Rust's `f32::round` (half away from zero) for v >= 0; any v < 0 gives a
// value <= 0, which callers clamp to 0 like the CPU.
fn rnd(v: f32) -> f32 {
    let r = floor(v);
    return select(r, r + 1.0, nb(v - r) >= 0.5);
}

@compute @workgroup_size(16, 16)
fn warp(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let x = p.ox + gid.x;
    let y = p.oy + gid.y;
    let o = gid.y * p.w + gid.x;
    // `Table::position`.
    let i = x >> 3u;
    let j = y >> 3u;
    let a = nodes[j * p.nx + i];
    let b = nodes[j * p.nx + i + 1u];
    let c = nodes[(j + 1u) * p.nx + i];
    let d = nodes[(j + 1u) * p.nx + i + 1u];
    if (a.x == INVALID || b.x == INVALID || c.x == INVALID || d.x == INVALID) {
        dst[o] = 0u;
        return;
    }
    let fx = f32(x & 7u) * 0.125;
    let fy = f32(y & 7u) * 0.125;
    let top = vec2<f32>(lerpf(a.x, b.x, fx), lerpf(a.y, b.y, fx));
    let bot = vec2<f32>(lerpf(c.x, d.x, fx), lerpf(c.y, d.y, fx));
    let px = lerpf(top.x, bot.x, fy);
    let py = lerpf(top.y, bot.y, fy);
    // `bicubic`.
    let sx = snap(nb(px - 0.5));
    let sy = snap(nb(py - 0.5));
    let ix = floor(sx);
    let iy = floor(sy);
    let wx = cubic(nb(sx - ix));
    let wy = cubic(nb(sy - iy));
    let bx = i32(ix) - 1;
    let by = i32(iy) - 1;
    var acc = vec4<f32>(0.0);
    for (var jj = 0; jj < 4; jj++) {
        for (var ii = 0; ii < 4; ii++) {
            let v = texel(bx + ii, by + jj);
            if (v == 0u) {
                continue;
            }
            let q = vec4<f32>(unpack4(v));
            let k = nb(wx[ii] * wy[jj]);
            acc = vec4<f32>(nb(acc.x + nb(q.x * k)), nb(acc.y + nb(q.y * k)), nb(acc.z + nb(q.z * k)), nb(acc.w + nb(q.w * k)));
        }
    }
    let al = clamp(rnd(acc.w), 0.0, 255.0);
    if (al <= 0.0) {
        dst[o] = 0u;
        return;
    }
    let cr = u32(clamp(rnd(acc.x), 0.0, al));
    let cg = u32(clamp(rnd(acc.y), 0.0, al));
    let cb = u32(clamp(rnd(acc.z), 0.0, al));
    dst[o] = cr | (cg << 8u) | (cb << 16u) | (u32(al) << 24u);
}
