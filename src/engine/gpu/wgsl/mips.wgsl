// Mip levels on the GPU, identical to `mips::build_mips`: each level is the
// 2 x 2 rounded average (`px::avg4`) of the tile area of the level below,
// with missing tiles reading as transparent.

struct P {
    // Source tile-area size (stride = sw).
    sw: u32,
    sh: u32,
    // Destination tile-area size.
    dw: u32,
    dh: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn at(x: u32, y: u32) -> vec4<u32> {
    if (x >= p.sw || y >= p.sh) {
        return vec4<u32>(0u);
    }
    return unpack4(src[y * p.sw + x]);
}

@compute @workgroup_size(16, 16)
fn down2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= p.dw || y >= p.dh) {
        return;
    }
    let s = at(2u * x, 2u * y) + at(2u * x + 1u, 2u * y) + at(2u * x, 2u * y + 1u) + at(2u * x + 1u, 2u * y + 1u);
    dst[y * p.dw + x] = pack4((s + vec4<u32>(2u)) >> vec4<u32>(2u));
}
