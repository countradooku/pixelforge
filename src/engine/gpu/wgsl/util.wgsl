// Moving pixels between resident images and band buffers on the GPU.

struct P {
    // Source image: stride (tile-area width), tile-area height, image size.
    stride: u32,
    th: u32,
    w: u32,
    h: u32,
    // Region origin in the image, and the region size.
    x0: i32,
    y0: i32,
    rw: u32,
    rh: u32,
    // 1: clamp to the image (`read_region_clamped`); 0: tile area as is,
    // transparent outside it (`read_region`).
    clamped: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(16, 16)
fn region(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.rw || gid.y >= p.rh) {
        return;
    }
    var x = p.x0 + i32(gid.x);
    var y = p.y0 + i32(gid.y);
    var v = 0u;
    if (p.clamped != 0u) {
        x = clamp(x, 0, i32(p.w) - 1);
        y = clamp(y, 0, i32(p.h) - 1);
        v = src[u32(y) * p.stride + u32(x)];
    } else if (x >= 0 && y >= 0 && u32(x) < p.stride && u32(y) < p.th) {
        v = src[u32(y) * p.stride + u32(x)];
    }
    dst[gid.y * p.rw + gid.x] = v;
}
