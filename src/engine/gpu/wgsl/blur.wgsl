// Gaussian / box blur and the blur-combine family (Unsharp Mask, Sharpen,
// High Pass), bit-exact with the CPU's integer passes:
// - box passes: running sums clamped at the buffer edge, rounded with
//   `(s * floor(2^24 / n) + 2^23) >> 24` per pass;
// - direct taps (sigma <= 2.6): 8.8 fixed-point kernels, rounded after each
//   axis (`gauss_plane`) or once at the end (`conv_sep`).

struct P {
    w: u32,
    h: u32,
    r: u32,
    inv: u32,
    k: u32,
    mode: u32,
    amount: i32,
    threshold: i32,
    pad: u32,
    cw: u32,
    ch: u32,
    taps: u32,
    // Columns past xmax read column xmax (vertical passes of the strip path).
    xmax: u32,
    // Strip path: outside the image (edge-tile padding) the CPU combines
    // with the raw tile, which only the CPU has, so output the blur there
    // and let the scatter combine it (`Kernel::raw_edge_combine`).
    raw_edge: u32,
    ox: i32,
    oy: i32,
    img_w: i32,
    img_h: i32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> aux: array<u32>;
// Strip-path unsharp: tab[d + 255] = round(d * amount / 100) as the CPU rounds it.
@group(0) @binding(4) var<storage, read> tab: array<i32>;

fn box_round(s: vec4<u32>) -> u32 {
    return pack4((s * p.inv + vec4<u32>(8388608u)) >> vec4<u32>(24u));
}

// One vertical box pass: thread (x, chunk of p.k rows).
@compute @workgroup_size(256, 1)
fn box_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    if (x >= p.w) {
        return;
    }
    let sx = min(x, p.xmax);
    let y0 = i32(gid.y * p.k);
    if (y0 >= i32(p.h)) {
        return;
    }
    let y1 = min(y0 + i32(p.k), i32(p.h));
    let r = i32(p.r);
    let hm = i32(p.h) - 1;
    var s = vec4<u32>(0u);
    for (var j = -r; j <= r; j++) {
        s += unpack4(src[u32(clamp(y0 + j, 0, hm)) * p.w + sx]);
    }
    for (var y = y0; y < y1; y++) {
        dst[u32(y) * p.w + x] = box_round(s);
        s += unpack4(src[u32(clamp(y + r + 1, 0, hm)) * p.w + sx]);
        s -= unpack4(src[u32(clamp(y - r, 0, hm)) * p.w + sx]);
    }
}

// One horizontal box pass: thread (chunk of p.k columns, y).
@compute @workgroup_size(1, 64)
fn box_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let y = gid.y;
    if (y >= p.h) {
        return;
    }
    let x0 = i32(gid.x * p.k);
    if (x0 >= i32(p.w)) {
        return;
    }
    let x1 = min(x0 + i32(p.k), i32(p.w));
    let r = i32(p.r);
    let wm = i32(p.w) - 1;
    let row = y * p.w;
    var s = vec4<u32>(0u);
    for (var j = -r; j <= r; j++) {
        s += unpack4(src[row + u32(clamp(x0 + j, 0, wm))]);
    }
    for (var x = x0; x < x1; x++) {
        dst[row + u32(x)] = box_round(s);
        s += unpack4(src[row + u32(clamp(x + r + 1, 0, wm))]);
        s -= unpack4(src[row + u32(clamp(x - r, 0, wm))]);
    }
}

// Direct separable taps (radius <= 8) for a 16 x 16 output block. The
// block's input (with apron) is staged in shared memory once, the
// horizontal pass goes to a second shared array, the vertical pass reads it.
// p.mode: 0 = round after each axis (`gauss_plane`), 1 = round once
// (`conv_sep`). Taps are in `aux` (8.8 fixed point, sum 256). Only pixels at
// least r from the buffer edge are written from the taps; the rest copy the
// input, as on the CPU.
//
// The taps sum to 256, so a weighted sum of 8-bit values fits 16 bits: two
// channels share a u32 (SWAR, like the CPU's 16-bit lanes): `lo` holds r and
// b, `hi` holds g and a.
var<workgroup> tile: array<u32, 1024>;
var<workgroup> hlo: array<u32, 512>;
var<workgroup> hhi: array<u32, 512>;
var<workgroup> taps: array<u32, 17>;

const LO: u32 = 0x00ff00ffu;

@compute @workgroup_size(16, 16)
fn gauss_direct(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = p.r;
    let bx = wid.x * 16u;
    let by = wid.y * 16u;
    let side = 16u + 2u * r;
    let li = lid.y * 16u + lid.x;
    if (li < p.taps) {
        taps[li] = aux[li];
    }
    // Stage rows by - r .. by + 16 + r, columns bx - r .. bx + 16 + r.
    for (var t = li; t < side * side; t += 256u) {
        let ty = t / side;
        let tx = t % side;
        let x = i32(bx + tx) - i32(r);
        let y = i32(by + ty) - i32(r);
        var v = 0u;
        if (x >= 0 && y >= 0 && x < i32(p.w) && y < i32(p.h)) {
            v = src[u32(y) * p.w + u32(x)];
        }
        tile[t] = v;
    }
    workgroupBarrier();
    // Horizontal taps for every staged row, the block's 16 columns. In
    // mode 1 the sums stay unrounded (16 bits per lane).
    for (var t = li; t < side * 16u; t += 256u) {
        let ly = t / 16u;
        let lx = t % 16u;
        let x = bx + lx;
        let y = i32(by + ly) - i32(r);
        var lo = 0u;
        var hi = 0u;
        if (x >= r && x + r < p.w && y >= 0 && y < i32(p.h)) {
            let base = ly * side + lx;
            for (var j = 0u; j < p.taps; j++) {
                let v = tile[base + j];
                let k = taps[j];
                lo += (v & LO) * k;
                hi += ((v >> 8u) & LO) * k;
            }
            if (p.mode == 0u) {
                lo = ((lo + 0x00800080u) >> 8u) & LO;
                hi = ((hi + 0x00800080u) >> 8u) & LO;
            }
        }
        hlo[t] = lo;
        hhi[t] = hi;
    }
    workgroupBarrier();
    let x = bx + lid.x;
    let y = by + lid.y;
    if (x >= p.w || y >= p.h) {
        return;
    }
    if (x < r || x + r >= p.w || y < r || y + r >= p.h) {
        dst[y * p.w + x] = tile[(lid.y + r) * side + lid.x + r];
        return;
    }
    if (p.mode == 0u) {
        var lo = 0u;
        var hi = 0u;
        for (var j = 0u; j < p.taps; j++) {
            let i = (lid.y + j) * 16u + lid.x;
            let k = taps[j];
            lo += hlo[i] * k;
            hi += hhi[i] * k;
        }
        lo = ((lo + 0x00800080u) >> 8u) & LO;
        hi = ((hi + 0x00800080u) >> 8u) & LO;
        dst[y * p.w + x] = lo | (hi << 8u);
    } else {
        // 16-bit sums times 8-bit taps need 24 bits: one channel per u32.
        var acc = vec4<u32>(0u);
        for (var j = 0u; j < p.taps; j++) {
            let i = (lid.y + j) * 16u + lid.x;
            let k = taps[j];
            let lo = hlo[i];
            let hi = hhi[i];
            acc += vec4<u32>(lo & 0xffffu, hi & 0xffffu, lo >> 16u, hi >> 16u) * k;
        }
        acc = (acc + vec4<u32>(32768u)) >> vec4<u32>(16u);
        dst[y * p.w + x] = pack4(acc);
    }
}

// Merges the original (`src`, padded) with its blur (`aux`, padded) into
// the center (`dst`, p.cw x p.ch): p.mode 0 = blur, 1 = unsharp (p.amount
// in 8.8, p.threshold; `fast::blur_combine`), 2 = high pass, 3 = unsharp as
// the strip path does it (`unsharp_px`: color only, `tab` rounding).
@compute @workgroup_size(16, 16)
fn combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    let si = (y + p.pad) * p.w + x + p.pad;
    let o = unpack4(src[si]);
    let b = unpack4(aux[si]);
    var out: vec4<u32>;
    if (p.raw_edge != 0u && (p.ox + i32(x) >= p.img_w || p.oy + i32(y) >= p.img_h)) {
        dst[y * p.cw + x] = pack4(b);
        return;
    }
    if (p.mode == 0u) {
        out = b;
    } else if (p.mode == 1u) {
        let oi = vec4<i32>(o);
        let d = oi - vec4<i32>(b);
        let v = oi + ((d * p.amount + vec4<i32>(128)) >> vec4<u32>(8u));
        let lim = vec4<i32>(i32(o.w), i32(o.w), i32(o.w), 255);
        let keep = abs(d) < vec4<i32>(p.threshold);
        out = vec4<u32>(select(clamp(v, vec4<i32>(0), lim), oi, keep));
    } else if (p.mode == 2u) {
        let a = i32(o.w);
        let v = vec4<i32>(o) - vec4<i32>(b) + vec4<i32>(a >> 1u);
        out = vec4<u32>(vec3<u32>(clamp(v.xyz, vec3<i32>(0), vec3<i32>(a))), o.w);
    } else {
        let oi = vec3<i32>(o.xyz);
        let d = oi - vec3<i32>(b.xyz);
        let v = oi + vec3<i32>(tab[d.x + 255], tab[d.y + 255], tab[d.z + 255]);
        let keep = abs(d) < vec3<i32>(p.threshold);
        out = vec4<u32>(vec3<u32>(clamp(select(v, oi, keep), vec3<i32>(0), vec3<i32>(i32(o.w)))), o.w);
    }
    // `merge`: color never exceeds alpha.
    out = vec4<u32>(min(out.xyz, vec3<u32>(out.w)), out.w);
    dst[y * p.cw + x] = pack4(out);
}

// Copies the center of a padded buffer.
@compute @workgroup_size(16, 16)
fn crop(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.cw || y >= p.ch) {
        return;
    }
    dst[y * p.cw + x] = src[(y + p.pad) * p.w + x + p.pad];
}
