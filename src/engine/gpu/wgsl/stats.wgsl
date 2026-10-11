// Histogram statistics (`filters::stats`): per-channel 256-bin histograms
// of straight color plus channel sums and the pixel count, over pixels with
// alpha >= 8. Workgroups accumulate in shared atomics and flush once; the
// sums are reduced across the subgroup first (`stats_sg`), or through
// shared memory (`stats_wg`) where subgroups are missing.

struct P {
    // Tile-area stride and image size of the source level.
    stride: u32,
    w: u32,
    h: u32,
    _0: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
// 768 histogram bins, then sums r, g, b, a and the count (low/high words).
@group(0) @binding(2) var<storage, read_write> out: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> unpremul: array<u32>;

var<workgroup> hist: array<atomic<u32>, 768>;
var<workgroup> sums: array<atomic<u32>, 5>;

fn unp(v: u32, a: u32) -> u32 {
    let i = a * 256u + v;
    return (unpremul[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

// Straight color of the pixel (and whether it counts).
fn sample(gid: vec3<u32>) -> vec4<u32> {
    if (gid.x >= p.w || gid.y >= p.h) {
        return vec4<u32>(0u);
    }
    let c = unpack4(src[gid.y * p.stride + gid.x]);
    if (c.w < 8u) {
        return vec4<u32>(0u);
    }
    if (c.w == 255u) {
        return c;
    }
    return vec4<u32>(unp(c.x, c.w), unp(c.y, c.w), unp(c.z, c.w), c.w);
}

fn clear(li: u32) {
    for (var i = li; i < 768u; i += 256u) {
        atomicStore(&hist[i], 0u);
    }
    if (li < 5u) {
        atomicStore(&sums[li], 0u);
    }
}

fn flush(li: u32) {
    for (var i = li; i < 768u; i += 256u) {
        let v = atomicLoad(&hist[i]);
        if (v != 0u) {
            atomicAdd(&out[i], v);
        }
    }
    if (li < 5u) {
        // 64-bit sums: low word, then carry into the high word.
        let v = atomicLoad(&sums[li]);
        let old = atomicAdd(&out[768u + li * 2u], v);
        if (old + v < old) {
            atomicAdd(&out[769u + li * 2u], 1u);
        }
    }
}

fn bin(c: vec4<u32>) {
    atomicAdd(&hist[c.x], 1u);
    atomicAdd(&hist[256u + c.y], 1u);
    atomicAdd(&hist[512u + c.z], 1u);
}

var<workgroup> red: array<vec4<u32>, 256>;
var<workgroup> redn: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn stats_wg(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    clear(li);
    let c = sample(gid);
    let counted = c.w != 0u;
    red[li] = c;
    redn[li] = select(0u, 1u, counted);
    workgroupBarrier();
    if (counted) {
        bin(c);
    }
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            red[li] += red[li + s];
            redn[li] += redn[li + s];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let t = red[0];
        atomicStore(&sums[0], t.x);
        atomicStore(&sums[1], t.y);
        atomicStore(&sums[2], t.z);
        atomicStore(&sums[3], t.w);
        atomicStore(&sums[4], redn[0]);
    }
    workgroupBarrier();
    flush(li);
}
