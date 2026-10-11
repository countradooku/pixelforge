// Liquify (`engine::liquify`): a sparse displacement field D on the GPU and
// the session's source image resampled through it (`out(x) = base(x - D(x))`,
// the CPU sampler's bilinear 8-bit fractions).
//
// The field lives in tile slots (`slot_of[tile]`, 256 x 256 vec2<f32> per
// slot) like the CPU's per-tile boxes; the freeze mask in the same slots,
// four bytes per u32. Every brush edit is the CPU's float expression, with
// each intermediate passed through `nb` (an opaque integer round trip on a
// runtime zero) so Metal's fast math can't fuse or reassociate it: given the
// same field, the GPU's field update and resample match the CPU bit for bit
// (Twirl's sin/cos are the precise `p_sincos`, within an ulp of the CPU's).

struct P {
    // Dab rectangle (image pixels).
    x0: i32,
    y0: i32,
    w: u32,
    h: u32,
    cols: u32,
    iw: u32,
    ih: u32,
    mode: u32,
    cx: f32,
    cy: f32,
    dx: f32,
    dy: f32,
    inv_r2: f32,
    pressure: f32,
    rate: f32,
    // Smooth's neighbor offset.
    k: i32,
    // Source stride (tile-area width).
    stride: u32,
    // Always 0 (opaque to the compiler).
    zero: u32,
    _0: u32,
    _1: u32,
}

const NONE: u32 = 0xffffffffu;
const TPX: u32 = 65536u;

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> slot_of: array<u32>;
@group(0) @binding(2) var<storage, read_write> field: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> frz: array<u32>;
@group(0) @binding(4) var<storage, read_write> avg: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> src: array<u32>;
@group(0) @binding(6) var<storage, read> tiles: array<u32>;
@group(0) @binding(7) var<storage, read_write> out: array<u32>;

// A float the compiler can't see through: blocks FMA contraction and
// reassociation across it.
fn nb(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ p.zero);
}

// `Session::d_at`: the field at a clamped pixel, 0 where no slot exists.
fn d_at(x: i32, y: i32) -> vec2<f32> {
    let cx = u32(clamp(x, 0, i32(p.iw) - 1));
    let cy = u32(clamp(y, 0, i32(p.ih) - 1));
    let s = slot_of[(cy >> 8u) * p.cols + (cx >> 8u)];
    if (s == NONE) {
        return vec2<f32>(0.0, 0.0);
    }
    return field[s * TPX + (cy & 255u) * 256u + (cx & 255u)];
}

// Smooth: the 9-tap average of the field before this dab.
@compute @workgroup_size(16, 16)
fn smooth_avg(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let x = p.x0 + i32(gid.x);
    let y = p.y0 + i32(gid.y);
    let k = p.k;
    var offs = array<vec2<i32>, 9>(
        vec2<i32>(-k, 0), vec2<i32>(k, 0), vec2<i32>(0, -k), vec2<i32>(0, k), vec2<i32>(0, 0),
        vec2<i32>(-k, -k), vec2<i32>(k, k), vec2<i32>(-k, k), vec2<i32>(k, -k),
    );
    var acc = vec2<f32>(0.0, 0.0);
    for (var i = 0u; i < 9u; i++) {
        let d = d_at(x + offs[i].x, y + offs[i].y);
        acc = vec2<f32>(nb(acc.x + d.x), nb(acc.y + d.y));
    }
    avg[gid.y * p.w + gid.x] = vec2<f32>(p_div(acc.x, 9.0), p_div(acc.y, 9.0));
}

// One dab's edit of D (`Session::dab`, modes as in `kernels::liquify`).
@compute @workgroup_size(16, 16)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let x = p.x0 + i32(gid.x);
    let y = p.y0 + i32(gid.y);
    let fx = nb(nb(f32(x) + 0.5) - p.cx);
    let fy = nb(nb(f32(y) + 0.5) - p.cy);
    let q = nb(nb(nb(fx * fx) + nb(fy * fy)) * p.inv_r2);
    if (q >= 1.0) {
        return;
    }
    let ux = u32(x);
    let uy = u32(y);
    let s = slot_of[(uy >> 8u) * p.cols + (ux >> 8u)];
    if (s == NONE) {
        return;
    }
    let i = (uy & 255u) * 256u + (ux & 255u);
    let fb = (frz[s * (TPX / 4u) + (i >> 2u)] >> ((i & 3u) * 8u)) & 255u;
    let frozen = p_div(f32(fb), 255.0);
    let omq = nb(1.0 - q);
    let f = nb(nb(nb(nb(omq * omq) * p.pressure)) * nb(1.0 - frozen));
    if (f <= 0.0) {
        return;
    }
    let vx = fx;
    let vy = fy;
    let fi = s * TPX + i;
    var d = field[fi];
    switch p.mode {
        // Forward Warp
        case 0u: {
            d = vec2<f32>(nb(d.x + nb(p.dx * f)), nb(d.y + nb(p.dy * f)));
        }
        // Push Left
        case 1u: {
            d = vec2<f32>(nb(d.x + nb(p.dy * f)), nb(d.y - nb(p.dx * f)));
        }
        // Twirl clockwise / counter-clockwise
        case 2u, 3u: {
            let sg = select(-1.0, 1.0, p.mode == 2u);
            let th = nb(nb(nb(0.06 * p.rate) * f) * sg);
            let sc = p_sincos(th);
            let sn = sc.x;
            let co = sc.y;
            let rx = nb(nb(vx * co) - nb(vy * sn));
            let ry = nb(nb(vx * sn) + nb(vy * co));
            d = vec2<f32>(nb(d.x + nb(vx - rx)), nb(d.y + nb(vy - ry)));
        }
        // Pucker / Bloat
        case 4u, 5u: {
            let sg = select(-1.0, 1.0, p.mode == 5u);
            let k = nb(nb(nb(0.04 * p.rate) * f) * sg);
            d = vec2<f32>(nb(d.x + nb(vx * k)), nb(d.y + nb(vy * k)));
        }
        // Reconstruct
        case 6u: {
            let k = max(nb(1.0 - nb(nb(0.15 * p.rate) * f)), 0.0);
            d = vec2<f32>(nb(d.x * k), nb(d.y * k));
        }
        // Smooth
        default: {
            let a = avg[gid.y * p.w + gid.x];
            let k = min(nb(nb(0.3 * p.rate) * f), 1.0);
            d = vec2<f32>(nb(d.x + nb(nb(a.x - d.x) * k)), nb(d.y + nb(nb(a.y - d.y) * k)));
        }
    }
    field[fi] = d;
}

fn texel(x: i32, y: i32) -> u32 {
    let xx = clamp(x, 0, i32(p.iw) - 1);
    let yy = clamp(y, 0, i32(p.ih) - 1);
    return src[u32(yy) * p.stride + u32(xx)];
}

// `Session::sample` at `(x, y)` (pixel-center coordinates).
fn sample(xs: f32, ys: f32) -> u32 {
    let x = nb(xs - 0.5);
    let y = nb(ys - 0.5);
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = u32(nb(x - x0) * 255.0);
    let fy = u32(nb(y - y0) * 255.0);
    let xi = i32(x0);
    let yi = i32(y0);
    return lerp4(lerp4(texel(xi, yi), texel(xi + 1, yi), fx), lerp4(texel(xi, yi + 1), texel(xi + 1, yi + 1), fx), fy);
}

// Whole tiles of the result (`tiles[z]` = tx | ty << 16), tile-major:
// image pixels resampled through D, edge-tile padding copied from the
// source (Liquify never writes it).
@compute @workgroup_size(16, 16)
fn resample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = tiles[gid.z];
    let tx = t & 0xffffu;
    let ty = t >> 16u;
    let gx = tx * 256u + gid.x;
    let gy = ty * 256u + gid.y;
    let o = gid.z * TPX + gid.y * 256u + gid.x;
    if (gx >= p.iw || gy >= p.ih) {
        out[o] = src[gy * p.stride + gx];
        return;
    }
    let s = slot_of[ty * p.cols + tx];
    var d = vec2<f32>(0.0, 0.0);
    if (s != NONE) {
        d = field[s * TPX + gid.y * 256u + gid.x];
    }
    out[o] = sample(nb(nb(f32(gx) + 0.5) - d.x), nb(nb(f32(gy) + 0.5) - d.y));
}
