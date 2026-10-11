// Workgroup-wide exclusive prefix sums with subgroup operations (the
// `rank_hist` module; `rank_scan_wg.wgsl` is the shared-memory fallback).
// Threads are ordered by (subgroup, lane): subgroups of a 1D workgroup are
// contiguous blocks of local_invocation_index on every backend we target.

var<workgroup> sg_tot: array<vec4<u32>, 128>;

// The thread's position in scan order.
fn scan_index(li: u32) -> u32 {
    return subgroupMin(li) + subgroupExclusiveAdd(1u);
}

// Exclusive prefix over the workgroup of two vec4 words at once.
fn wg_excl(xa: vec4<u32>, xb: vec4<u32>, L: u32) -> array<vec4<u32>, 2> {
    let size = subgroupAdd(1u);
    let sg = L / size;
    var ea = subgroupExclusiveAdd(xa);
    var eb = subgroupExclusiveAdd(xb);
    if (L % size == size - 1u) {
        sg_tot[2u * sg] = ea + xa;
        sg_tot[2u * sg + 1u] = eb + xb;
    }
    workgroupBarrier();
    for (var s = 0u; s < sg; s++) {
        ea += sg_tot[2u * s];
        eb += sg_tot[2u * s + 1u];
    }
    workgroupBarrier();
    return array<vec4<u32>, 2>(ea, eb);
}
