// Pixelate ▸ Mosaic, Crystallize, Pointillize and Stylize ▸ Extrude: every
// cell is filled with the rounded mean of the band pixels it covers, like
// `stylize::mosaic` / `stylize::cells` over the CPU's padded buffer.
//
// Mosaic cells come from CPU-built tables (the cell of every band column and
// row, with the CPU's float formula). Voronoi cells: each 16 x 16 workgroup
// computes the jittered seeds of the grid cells around its pixels once into
// shared memory (its candidate list), every pixel searches its 3 x 3 grid
// neighborhood there exactly as the CPU does, and the per-cell sums are
// gathered with shared-memory atomics before one global atomic per cell.

struct P {
    pw: u32,
    ph: u32,
    pad: u32,
    cw: u32,
    ch: u32,
    ox: i32,
    oy: i32,
    zero: u32,
    s: f32,
    // Mosaic: cells per row of cells. Voronoi: cell grid origin and size.
    nx: u32,
    ny: u32,
    c0x: i32,
    c0y: i32,
    cell: f32,
    mode: u32,
    // Commit: CPU block side in center pixels; 0: the whole band (previews).
    blk: u32,
    // Pointillize background pixel.
    bg: u32,
    _0: u32,
    _1: u32,
    _2: u32,
    f: array<vec4<f32>, 3>,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
// Mosaic: xcell[pw], ycell[ph], xs[nx + 1], ys[ny + 1] (cell id per column /
// row, first column / row of each cell). Voronoi: ix[pw], iy[ph] (grid cell
// of each band column / row).
@group(0) @binding(3) var<storage, read> tab: array<i32>;
@group(0) @binding(4) var<storage, read> sums: array<u32>;
@group(0) @binding(5) var<storage, read_write> acc: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> dist: array<f32>;

fn opq(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ p.zero);
}

// Correctly rounded x / y. Like the prelude's `p_div`, with the quotient and
// the result opaque so fast math can neither fold the residual `x - q * y`
// to zero nor reassociate the correction into a following sum.
fn div_rn(x: f32, y: f32) -> f32 {
    let q = opq(x / y);
    return opq(q + opq(fma(-q, y, x)) / y);
}

fn pf(k: u32) -> f32 {
    return p.f[k >> 2u][k & 3u];
}

fn mean(s: vec4<u32>, n: u32) -> u32 {
    let m = max(n, 1u);
    return pack4((s + vec4<u32>(m / 2u)) / vec4<u32>(m));
}

// ----- Mosaic -------------------------------------------------------------------------------

// Pass 1: sums of each cell column over each band row -> dst[(y * nx + c) * 4 + k].
@compute @workgroup_size(64, 4)
fn mosaic_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let y = gid.y;
    if (c >= p.nx || y >= p.ph) {
        return;
    }
    let off = p.pw + p.ph;
    let x0 = u32(tab[off + c]);
    let x1 = u32(tab[off + c + 1u]);
    var s = vec4<u32>(0u);
    for (var x = x0; x < x1; x++) {
        s += unpack4(src[y * p.pw + x]);
    }
    let o = (y * p.nx + c) * 4u;
    dst[o] = s.x;
    dst[o + 1u] = s.y;
    dst[o + 2u] = s.z;
    dst[o + 3u] = s.w;
}

// Pass 2: cell means -> dst[r * nx + c].
@compute @workgroup_size(64, 4)
fn mosaic_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let r = gid.y;
    if (c >= p.nx || r >= p.ny) {
        return;
    }
    let off = p.pw + p.ph;
    let y0 = u32(tab[off + p.nx + 1u + r]);
    let y1 = u32(tab[off + p.nx + 1u + r + 1u]);
    var s = vec4<u32>(0u);
    for (var y = y0; y < y1; y++) {
        let o = (y * p.nx + c) * 4u;
        s += vec4<u32>(sums[o], sums[o + 1u], sums[o + 2u], sums[o + 3u]);
    }
    let n = (u32(tab[off + c + 1u]) - u32(tab[off + c])) * (y1 - y0);
    dst[r * p.nx + c] = mean(s, n);
}

// Pass 3: every center pixel takes its cell's mean (mode 0), or the
// Extrude face shading of it (mode 1: f0 = size, f1..f8 = shades,
// f9 = pyramids).
@compute @workgroup_size(16, 16)
fn mosaic_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = gid.x + p.pad;
    let y = gid.y + p.pad;
    let m = sums[u32(tab[p.pw + y]) * p.nx + u32(tab[x])];
    if (p.mode == 0u) {
        dst[gid.y * p.cw + gid.x] = m;
        return;
    }
    let size = pf(0u);
    let gx = (f32(p.ox + i32(x)) + 0.5) * p.s;
    let gy = (f32(p.oy + i32(y)) + 0.5) * p.s;
    let qx = div_rn(gx, size);
    let qy = div_rn(gy, size);
    let u = opq(qx - trunc(qx)) - 0.5;
    let v = opq(qy - trunc(qy)) - 0.5;
    var shade: f32;
    if (pf(9u) != 0.0) {
        if (abs(v) > abs(u)) {
            shade = select(pf(2u), pf(1u), v < 0.0);
        } else if (u < 0.0) {
            shade = pf(3u);
        } else {
            shade = pf(4u);
        }
    } else {
        let face = pf(5u);
        if (abs(u) <= face && abs(v) <= face) {
            shade = 1.0;
        } else if (abs(v) > abs(u)) {
            shade = select(pf(6u), pf(1u), v < 0.0);
        } else if (u < 0.0) {
            shade = pf(7u);
        } else {
            shade = pf(8u);
        }
    }
    let c = unpack4(m);
    let fa = f32(c.w);
    let o = vec3<u32>(min(vec3<f32>(c.xyz) * shade, vec3<f32>(fa)));
    dst[gid.y * p.cw + gid.x] = pack4(vec4<u32>(o, c.w));
}

// ----- Crystallize / Pointillize -------------------------------------------------------------

const MAXC: u32 = 256u;
var<workgroup> seeds: array<vec2<f32>, 256>;
var<workgroup> wsum: array<array<atomic<u32>, 4>, 256>;
var<workgroup> wcnt: array<atomic<u32>, 256>;
var<workgroup> wbox: array<array<atomic<u32>, 4>, 256>;

// The jittered seed of grid cell (cx, cy), level-0 coordinates.
fn seed(cx: i32, cy: i32) -> vec2<f32> {
    let cell = p.cell;
    let sx = opq((opq(f32(cx) + 0.1) + opq(0.8 * hash01(cx, cy, 7u))) * cell);
    let sy = opq((opq(f32(cy) + 0.1) + opq(0.8 * hash01(cx, cy, 13u))) * cell);
    return vec2<f32>(sx, sy);
}

fn d2(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = a - b;
    return opq(d.x * d.x) + opq(d.y * d.y);
}

// Pass 1: nearest seed of every band pixel (the CPU's 3 x 3 search, same
// order and ties) -> ids (cell index relative to (c0x, c0y)) and distance;
// per-cell sums, counts and bounding boxes into `acc`
// (cell * 9: r, g, b, a, n, x0, y0, x1, y1, as min/max over band coords).
@compute @workgroup_size(16, 16)
fn vor_assign(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.y * 16u + lid.x;
    let bx = wid.x * 16u;
    let by = wid.y * 16u;
    let ex = min(bx + 15u, p.pw - 1u);
    let ey = min(by + 15u, p.ph - 1u);
    let ix0 = tab[bx] - 1;
    let iy0 = tab[p.pw + by] - 1;
    let nw = u32(tab[ex] + 1 - ix0 + 1);
    let nh = u32(tab[p.pw + ey] + 1 - iy0 + 1);
    let shared_ok = nw * nh <= MAXC;
    if (shared_ok) {
        for (var k = t; k < nw * nh; k += 256u) {
            seeds[k] = seed(ix0 + i32(k % nw), iy0 + i32(k / nw));
            for (var j = 0u; j < 4u; j++) {
                atomicStore(&wsum[k][j], 0u);
            }
            atomicStore(&wcnt[k], 0u);
            atomicStore(&wbox[k][0], 0xffffffffu);
            atomicStore(&wbox[k][1], 0xffffffffu);
            atomicStore(&wbox[k][2], 0u);
            atomicStore(&wbox[k][3], 0u);
        }
    }
    workgroupBarrier();
    let x = bx + lid.x;
    let y = by + lid.y;
    let inside = x < p.pw && y < p.ph;
    var slot = 0u;
    if (inside) {
        let s = p.s;
        let g = vec2<f32>((f32(p.ox + i32(x)) + 0.5) * s, (f32(p.oy + i32(y)) + 0.5) * s);
        let ix = tab[x];
        let iy = tab[p.pw + y];
        var best = 3.4028235e38;
        var bs = vec2<f32>(0.0);
        var bk = 0u;
        for (var oy = -1; oy <= 1; oy++) {
            for (var ox = -1; ox <= 1; ox++) {
                var sd: vec2<f32>;
                var k = 0u;
                if (shared_ok) {
                    k = u32(iy + oy - iy0) * nw + u32(ix + ox - ix0);
                    sd = seeds[k];
                } else {
                    sd = seed(ix + ox, iy + oy);
                }
                let d = d2(sd, g);
                if (d < best) {
                    best = d;
                    bs = sd;
                    bk = k;
                }
            }
        }
        let cx = clamp(i32(floor(div_rn(bs.x, p.cell))) - p.c0x, 0, i32(p.nx) - 1);
        let cy = clamp(i32(floor(div_rn(bs.y, p.cell))) - p.c0y, 0, i32(p.ny) - 1);
        let id = u32(cy) * p.nx + u32(cx);
        let i = y * p.pw + x;
        ids[i] = id;
        if (p.mode == 1u) {
            dist[i] = sqrt(best);
        }
        let c = unpack4(src[i]);
        if (shared_ok) {
            slot = bk;
            atomicAdd(&wsum[bk][0], c.x);
            atomicAdd(&wsum[bk][1], c.y);
            atomicAdd(&wsum[bk][2], c.z);
            atomicAdd(&wsum[bk][3], c.w);
            atomicAdd(&wcnt[bk], 1u);
            atomicMin(&wbox[bk][0], x);
            atomicMin(&wbox[bk][1], y);
            atomicMax(&wbox[bk][2], x);
            atomicMax(&wbox[bk][3], y);
        } else {
            let o = id * 9u;
            atomicAdd(&acc[o], c.x);
            atomicAdd(&acc[o + 1u], c.y);
            atomicAdd(&acc[o + 2u], c.z);
            atomicAdd(&acc[o + 3u], c.w);
            atomicAdd(&acc[o + 4u], 1u);
            atomicMin(&acc[o + 5u], x);
            atomicMin(&acc[o + 6u], y);
            atomicMax(&acc[o + 7u], x);
            atomicMax(&acc[o + 8u], y);
        }
    }
    workgroupBarrier();
    for (var k = t; shared_ok && k < nw * nh; k += 256u) {
        let n = atomicLoad(&wcnt[k]);
        if (n == 0u) {
            continue;
        }
        let sd = seeds[k];
        let cx = clamp(i32(floor(div_rn(sd.x, p.cell))) - p.c0x, 0, i32(p.nx) - 1);
        let cy = clamp(i32(floor(div_rn(sd.y, p.cell))) - p.c0y, 0, i32(p.ny) - 1);
        let o = (u32(cy) * p.nx + u32(cx)) * 9u;
        atomicAdd(&acc[o], atomicLoad(&wsum[k][0]));
        atomicAdd(&acc[o + 1u], atomicLoad(&wsum[k][1]));
        atomicAdd(&acc[o + 2u], atomicLoad(&wsum[k][2]));
        atomicAdd(&acc[o + 3u], atomicLoad(&wsum[k][3]));
        atomicAdd(&acc[o + 4u], n);
        atomicMin(&acc[o + 5u], atomicLoad(&wbox[k][0]));
        atomicMin(&acc[o + 6u], atomicLoad(&wbox[k][1]));
        atomicMax(&acc[o + 7u], atomicLoad(&wbox[k][2]));
        atomicMax(&acc[o + 8u], atomicLoad(&wbox[k][3]));
    }
}

// Clears the accumulators (bounding boxes start empty).
@compute @workgroup_size(256)
fn vor_clear(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= p.nx * p.ny * 9u) {
        return;
    }
    let k = i % 9u;
    atomicStore(&acc[i], select(0u, 0xffffffffu, k == 5u || k == 6u));
}

// The CPU block's buffer (band buffer coordinates) around center pixel
// (x, y): commit blocks are p.blk-pixel squares of tiles from the band's
// first column and from image row 0; previews use the whole band.
fn block_buf(cx: u32, cy: u32) -> vec4<i32> {
    if (p.blk == 0u) {
        return vec4<i32>(0, 0, i32(p.pw), i32(p.ph));
    }
    let b = i32(p.blk);
    let x0 = (i32(cx) / b) * b;
    // Center row of the band in image coordinates.
    let gy = p.oy + i32(p.pad) + i32(cy);
    let y0 = (gy / b) * b - (p.oy + i32(p.pad));
    let x1 = min(x0 + b, i32(p.cw)) + 2 * i32(p.pad);
    let y1 = min(y0 + b, i32(p.ch)) + 2 * i32(p.pad);
    return vec4<i32>(max(x0, 0), max(y0, 0), x1, y1);
}

// Pass 2: each center pixel takes its cell's mean over the CPU block's
// buffer: the band-wide sums when the cell lies inside that buffer (almost
// always), else the cell's pixels inside it, recounted from `ids`.
// mode 1 (Pointillize): f0 = dot radius (level-0 px), i.e. `cell * 0.42`.
@compute @workgroup_size(16, 16)
fn vor_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let x = gid.x + p.pad;
    let y = gid.y + p.pad;
    let i = y * p.pw + x;
    let id = ids[i];
    let o = id * 9u;
    let bb = block_buf(gid.x, gid.y);
    let box = vec4<i32>(i32(sums[o + 5u]), i32(sums[o + 6u]), i32(sums[o + 7u]), i32(sums[o + 8u]));
    var c: u32;
    if (box.x >= bb.x && box.y >= bb.y && box.z < bb.z && box.w < bb.w) {
        c = mean(vec4<u32>(sums[o], sums[o + 1u], sums[o + 2u], sums[o + 3u]), sums[o + 4u]);
    } else {
        var s = vec4<u32>(0u);
        var n = 0u;
        for (var yy = max(box.y, bb.y); yy <= min(box.w, bb.w - 1); yy++) {
            for (var xx = max(box.x, bb.x); xx <= min(box.z, bb.z - 1); xx++) {
                let j = u32(yy) * p.pw + u32(xx);
                if (ids[j] == id) {
                    s += unpack4(src[j]);
                    n += 1u;
                }
            }
        }
        c = mean(s, n);
    }
    if (p.mode == 1u) {
        let cov = clamp((pf(0u) - dist[i]) / p.s + 0.5, 0.0, 1.0);
        c = lerp4(p.bg, c, u32(cov * 255.0));
    }
    dst[gid.y * p.cw + gid.x] = c;
}
