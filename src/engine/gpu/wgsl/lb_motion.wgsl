// Motion Blur (`blur::motion`).
// - 0° / 90°: integer running box sums, bit-exact (`box_line`,
//   `box_vertical`).
// - Other angles: the CPU's sheared lines. Each output interpolates the
//   window means of the two lines `b = floor(Y - X * slope)` and `b + 1`
//   around it. Lines are laid out relative to the CPU block's buffer origin
//   (the same lattice as the CPU), and their window means are running sums
//   along chunks of `p.l` outputs (pass `lines`), so the result is the
//   CPU's up to float rounding.

struct P {
    w: u32,
    h: u32,
    pad: u32,
    cw: u32,
    ch: u32,
    r: u32,
    inv: u32,
    k: u32,
    half: u32,
    l: u32,
    transposed: u32,
    slope: f32,
    invn: f32,
    // Chunks per part, block size in center px (0: one part), parts per row.
    cpp: u32,
    s: u32,
    nbc: u32,
    // Level-space row of the band's first center row, first block row.
    oyc: i32,
    br0: i32,
    zero: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
// Per chunk: X0, X1, bstart, nlines, voff, Bx, By, 0.
@group(0) @binding(3) var<storage, read> desc: array<i32>;
@group(0) @binding(4) var<storage, read_write> lines_v: array<vec4<f32>>;

fn box_round(s: vec4<u32>) -> u32 {
    return pack4((s * p.inv + vec4<u32>(8388608u)) >> vec4<u32>(24u));
}

// Horizontal box of radius p.r over padded rows, center written: thread (p.k columns, row).
@compute @workgroup_size(1, 64)
fn mbox_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    let x0 = gid.x * p.k;
    if (y >= p.ch || x0 >= p.cw) {
        return;
    }
    let x1 = min(x0 + p.k, p.cw);
    let r = i32(p.r);
    let wm = i32(p.w) - 1;
    let row = (y + p.pad) * p.w;
    var s = vec4<u32>(0u);
    for (var j = -r; j <= r; j++) {
        s += unpack4(src[row + u32(clamp(i32(x0 + p.pad) + j, 0, wm))]);
    }
    for (var x = x0; x < x1; x++) {
        dst[y * p.cw + x] = box_round(s);
        let c = i32(x + p.pad);
        s += unpack4(src[row + u32(clamp(c + r + 1, 0, wm))]);
        s -= unpack4(src[row + u32(clamp(c - r, 0, wm))]);
    }
}

// Vertical box: thread (column, p.k rows).
@compute @workgroup_size(256, 1)
fn mbox_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y0 = gid.y * p.k;
    if (x >= p.cw || y0 >= p.ch) {
        return;
    }
    let y1 = min(y0 + p.k, p.ch);
    let r = i32(p.r);
    let hm = i32(p.h) - 1;
    let col = x + p.pad;
    var s = vec4<u32>(0u);
    for (var j = -r; j <= r; j++) {
        s += unpack4(src[u32(clamp(i32(y0 + p.pad) + j, 0, hm)) * p.w + col]);
    }
    for (var y = y0; y < y1; y++) {
        dst[y * p.cw + x] = box_round(s);
        let c = i32(y + p.pad);
        s += unpack4(src[u32(clamp(c + r + 1, 0, hm)) * p.w + col]);
        s -= unpack4(src[u32(clamp(c - r, 0, hm)) * p.w + col]);
    }
}

fn opq4(v: vec4<f32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(v) ^ vec4<u32>(p.zero));
}

// Pixel at plane-local (X, Y) of the block whose buffer starts at (bx, by)
// in band coordinates.
fn plane_px(xp: i32, yp: i32, bx: i32, by: i32) -> vec4<f32> {
    var xl = xp;
    var yl = yp;
    if (p.transposed != 0u) {
        xl = yp;
        yl = xp;
    }
    let x = u32(clamp(bx + xl, 0, i32(p.w) - 1));
    let y = u32(clamp(by + yl, 0, i32(p.h) - 1));
    return vec4<f32>(unpack4(src[y * p.w + x]));
}

// `motion_plane`'s sample: line `b` at X, interpolated across rows.
fn samp(xp: i32, b: f32, bx: i32, by: i32) -> vec4<f32> {
    let y = b + f32(xp) * p.slope;
    let y0 = floor(y);
    let f = y - y0;
    let yi = i32(y0);
    return plane_px(xp, yi, bx, by) * (1.0 - f) + plane_px(xp, yi + 1, bx, by) * f;
}

// Thread (line j, chunk): window means of line `bstart + j` at the chunk's X.
@compute @workgroup_size(64)
fn lines(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = i32(gid.x);
    let d = gid.y * 8u;
    let nlines = desc[d + 3u];
    if (j >= nlines) {
        return;
    }
    let x0 = desc[d];
    let x1 = desc[d + 1u];
    let b = f32(desc[d + 2u] + j);
    let voff = u32(desc[d + 4u]) + u32(j) * p.l;
    let bx = desc[d + 5u];
    let by = desc[d + 6u];
    let half = i32(p.half);
    var sum = vec4<f32>(0.0);
    for (var i = -half; i <= half; i++) {
        sum += samp(x0 + i, b, bx, by);
    }
    for (var x = x0; x < x1; x++) {
        lines_v[voff + u32(x - x0)] = sum * p.invn;
        sum += opq4(samp(x + half + 1, b, bx, by) - samp(x - half, b, bx, by));
    }
}

// Thread per center pixel: interpolate the two lines around it.
@compute @workgroup_size(16, 16)
fn lines_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    var part = 0u;
    if (p.s != 0u) {
        let br = (p.oyc + i32(y)) / i32(p.s) - p.br0;
        part = u32(br) * p.nbc + x / p.s;
    }
    let first = part * p.cpp * 8u;
    let bx = desc[first + 5u];
    let by = desc[first + 6u];
    let xl = i32(x + p.pad) - bx;
    let yl = i32(y + p.pad) - by;
    var xp = xl;
    var yp = yl;
    if (p.transposed != 0u) {
        xp = yl;
        yp = xl;
    }
    let c = part * p.cpp + u32(xp - desc[first]) / p.l;
    let d = c * 8u;
    let u = f32(yp) - f32(xp) * p.slope;
    let bf = floor(u);
    let f = u - bf;
    let j = clamp(i32(bf) - desc[d + 2u], 0, desc[d + 3u] - 2);
    let base = u32(desc[d + 4u]) + u32(xp - desc[d]);
    let a = lines_v[base + u32(j) * p.l];
    let cc = lines_v[base + u32(j + 1) * p.l];
    let v = vec4<u32>(clamp(a + (cc - a) * f + vec4<f32>(0.5), vec4<f32>(0.0), vec4<f32>(255.0)));
    dst[y * p.cw + x] = pack4(vec4<u32>(min(v.xyz, vec3<u32>(v.w)), v.w));
}
