// Subgroup entry of `stats.wgsl` (needs the SUBGROUP feature).

// One-dimensional workgroups (subgroup builtins require it), each covering
// a 16 x 16 block like `stats_wg`.
@compute @workgroup_size(256)
fn stats_sg(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_invocation_id) sl: u32) {
    clear(li);
    workgroupBarrier();
    let c = sample(vec3<u32>(wid.x * 16u + li % 16u, wid.y * 16u + li / 16u, 0u));
    let counted = c.w != 0u;
    if (counted) {
        bin(c);
    }
    // One atomic per subgroup for the sums.
    let s = subgroupAdd(vec4<u32>(c.xyz, c.w));
    let n = subgroupAdd(select(0u, 1u, counted));
    if (sl == 0u) {
        atomicAdd(&sums[0], s.x);
        atomicAdd(&sums[1], s.y);
        atomicAdd(&sums[2], s.z);
        atomicAdd(&sums[3], s.w);
        atomicAdd(&sums[4], n);
    }
    workgroupBarrier();
    flush(li);
}

