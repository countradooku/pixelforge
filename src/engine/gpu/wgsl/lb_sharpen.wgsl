// Smart Sharpen beyond plain unsharp masking (`blur::smart_sharpen`) and
// Shadows/Highlights' table combine. All integer, bit-exact with the CPU.

struct P {
    w: u32,
    h: u32,
    pad: u32,
    cw: u32,
    ch: u32,
    r: u32,
    k: i32,
    th: i32,
    fades: u32,
    // Where the blur's center pixel (x, y) is: aux[(y + boff) * bpitch + x + boff].
    boff: u32,
    bpitch: u32,
    ntaps: u32,
    raw_edge: u32,
    ox: i32,
    oy: i32,
    img_w: i32,
    img_h: i32,
    zero: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> aux: array<u32>;
// unpremul[a * 256 + v] (bytes), as the CPU computes it.
@group(0) @binding(4) var<storage, read> unpremul: array<u32>;
// Line taps (dx, dy), fade attenuations (2 x 256 i32), or the S/H table (bytes).
@group(0) @binding(5) var<storage, read> tab: array<i32>;
@group(0) @binding(6) var<storage, read> ls: array<u32>;
@group(0) @binding(7) var<storage, read> lh: array<u32>;

fn unpremul_ch(v: u32, a: u32) -> u32 {
    let i = a * 256u + v;
    return (unpremul[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

// `px::unpremultiply` (color only).
fn straight(c: vec4<u32>) -> vec3<u32> {
    if (c.w == 255u) {
        return c.xyz;
    }
    return vec3<u32>(unpremul_ch(c.x, c.w), unpremul_ch(c.y, c.w), unpremul_ch(c.z, c.w));
}

fn luma8(c: vec3<u32>) -> u32 {
    return (c.x * 77u + c.y * 150u + c.z * 29u + 128u) >> 8u;
}

// Luma of straight color, per padded pixel (`blur::luma8`).
@compute @workgroup_size(256)
fn luma(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.w * p.h) {
        return;
    }
    dst[i] = luma8(straight(unpack4(src[i])));
}

// Horizontal box sums of radius p.r (clamped to the row) at the center
// columns of every padded row: thread (64 columns, row).
@compute @workgroup_size(1, 64)
fn lbox_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    let x0 = gid.x * 64u;
    if (y >= p.h || x0 >= p.cw) {
        return;
    }
    let x1 = min(x0 + 64u, p.cw);
    let r = i32(p.r);
    let wm = i32(p.w) - 1;
    let row = y * p.w;
    var s = 0u;
    for (var j = -r; j <= r; j++) {
        s += src[row + u32(clamp(i32(x0 + p.pad) + j, 0, wm))];
    }
    for (var x = x0; x < x1; x++) {
        dst[y * p.cw + x] = s;
        let c = i32(x + p.pad);
        s += src[row + u32(clamp(c + r + 1, 0, wm))];
        s -= src[row + u32(clamp(c - r, 0, wm))];
    }
}

// Vertical sums of the horizontal sums: the box mean at center pixels
// (one rounding, `blur::box_mean`). Thread (column, 16 rows).
@compute @workgroup_size(256, 1)
fn lbox_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y0 = gid.y * 16u;
    if (x >= p.cw || y0 >= p.ch) {
        return;
    }
    let y1 = min(y0 + 16u, p.ch);
    let r = i32(p.r);
    let hm = i32(p.h) - 1;
    let n = 2u * p.r + 1u;
    let nn = n * n;
    var s = 0u;
    for (var j = -r; j <= r; j++) {
        s += src[u32(clamp(i32(y0 + p.pad) + j, 0, hm)) * p.cw + x];
    }
    for (var y = y0; y < y1; y++) {
        dst[y * p.cw + x] = (s + nn / 2u) / nn;
        let c = i32(y + p.pad);
        s += src[u32(clamp(c + r + 1, 0, hm)) * p.cw + x];
        s -= src[u32(clamp(c - r, 0, hm)) * p.cw + x];
    }
}

// Line average at center pixels (Motion Blur removal, `blur::line_plane`).
@compute @workgroup_size(16, 16)
fn line_taps(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    let px = i32(x + p.pad);
    let py = i32(y + p.pad);
    var s = vec4<u32>(0u);
    for (var t = 0u; t < p.ntaps; t++) {
        let xx = clamp(px + tab[2u * t], 0, i32(p.w) - 1);
        let yy = clamp(py + tab[2u * t + 1u], 0, i32(p.h) - 1);
        s += unpack4(src[u32(yy) * p.w + u32(xx)]);
    }
    let n = p.ntaps;
    dst[y * p.cw + x] = pack4((2u * s + vec4<u32>(n)) / vec4<u32>(2u * n));
}

// The unsharp combine with the fades (`blur::smart_sharpen`).
@compute @workgroup_size(16, 16)
fn ss_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    let o = vec4<i32>(unpack4(src[(y + p.pad) * p.w + x + p.pad]));
    let b = vec4<i32>(unpack4(aux[(y + p.boff) * p.bpitch + x + p.boff]));
    let d = o - b;
    var delta = (d * p.k + vec4<i32>(128)) >> vec4<u32>(8u);
    if (p.fades != 0u) {
        let i = y * p.cw + x;
        let att = tab[ls[i]] * tab[256u + lh[i]];
        delta = (delta * att + vec4<i32>(32768)) >> vec4<u32>(16u);
    }
    let lim = vec4<i32>(o.w, o.w, o.w, 255);
    let keep = abs(d) < vec4<i32>(p.th);
    let v = vec4<u32>(select(clamp(o + delta, vec4<i32>(0), lim), o, keep));
    dst[y * p.cw + x] = pack4(vec4<u32>(min(v.xyz, vec3<u32>(v.w)), v.w));
}

fn tab_byte(i: u32) -> u32 {
    return (bitcast<u32>(tab[i >> 2u]) >> ((i & 3u) * 8u)) & 255u;
}

// Shadows/Highlights: `Combiner::apply` with the 256 x 256 table.
@compute @workgroup_size(16, 16)
fn sh_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    let si = (y + p.pad) * p.w + x + p.pad;
    let ov = src[si];
    let bv = aux[si];
    if (p.raw_edge != 0u && (p.ox + i32(x) >= p.img_w || p.oy + i32(y) >= p.img_h)) {
        dst[y * p.cw + x] = bv;
        return;
    }
    let o = unpack4(ov);
    if (o.w == 0u) {
        dst[y * p.cw + x] = ov;
        return;
    }
    let c = straight(o);
    let m = luma8(straight(unpack4(bv))) * 256u;
    dst[y * p.cw + x] = premul(vec3<u32>(tab_byte(m + c.x), tab_byte(m + c.y), tab_byte(m + c.z)), o.w);
}
