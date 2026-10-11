// Workgroup-wide exclusive prefix sums in shared memory (fallback for
// adapters without subgroups; same interface as `rank_scan_sg.wgsl`).
// `WG` (threads per workgroup) is declared by the including module.

var<workgroup> hs: array<vec4<u32>, 1024>;

fn scan_index(li: u32) -> u32 {
    return li;
}

fn wg_excl(xa: vec4<u32>, xb: vec4<u32>, L: u32) -> array<vec4<u32>, 2> {
    hs[2u * L] = xa;
    hs[2u * L + 1u] = xb;
    workgroupBarrier();
    for (var d = 1u; d < WG; d *= 2u) {
        let j = max(L, d) - d;
        let ta = select(vec4<u32>(0u), hs[2u * j], L >= d);
        let tb = select(vec4<u32>(0u), hs[2u * j + 1u], L >= d);
        workgroupBarrier();
        hs[2u * L] += ta;
        hs[2u * L + 1u] += tb;
        workgroupBarrier();
    }
    let ea = hs[2u * L] - xa;
    let eb = hs[2u * L + 1u] - xb;
    workgroupBarrier();
    return array<vec4<u32>, 2>(ea, eb);
}
