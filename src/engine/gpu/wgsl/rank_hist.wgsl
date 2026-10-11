// Window histograms for large-radius rank filters (Median, Dust &
// Scratches), exact for any radius up to 127.
//
// One workgroup owns a strip of S = WG input columns (TW = S - 2r output
// columns), a chunk of output rows and one channel. Each thread owns one
// column and keeps its histograms over the 2r + 1 window rows, sliding
// them down one row per step (one add, one remove) as in Perreault & Hebert:
// - coarse histograms (16 bins of 16 values, byte counts) in registers;
// - fine histograms (256 byte counts) in a scratch buffer, bin-major so a
//   strip's columns are contiguous.
// The window histogram of an output column is the sum of 2r + 1 column
// histograms: a prefix sum across the strip (subgroup scan, or the
// shared-memory fallback) and one difference. 16-bit counts are packed two
// per u32; prefix sums wrap, but window counts stay below 65536, so the
// packed difference splits back into exact counts.
// The coarse histogram gives the segment of the median and its rank there.
// The 16 fine counts of that segment come either from one more prefix sum
// per segment occurring in the row (few, for the smooth medians of large
// windows) or from summing the 2r + 1 columns directly.

override WG: u32 = 256u;

struct H {
    pw: u32,
    ph: u32,
    cw: u32,
    ch: u32,
    r: u32,
    // 0 = median, 1 = dust & scratches (threshold `thr`).
    mode: u32,
    thr: u32,
    // Strip width (= WG), output columns per strip, rows per chunk.
    s: u32,
    tw: u32,
    th: u32,
    nstrips: u32,
    nchunks: u32,
}

@group(0) @binding(0) var<uniform> h: H;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> outp: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> fine: array<u32>;

// Prefix rows: two vec4 (16 packed 16-bit counts) per entry, WG + 1 entries.
var<workgroup> ps: array<vec4<u32>, 2u * WG + 2u>;
var<workgroup> jm_acc: atomic<u32>;
var<workgroup> jm: u32;
var<workgroup> lo_acc: atomic<u32>;
var<workgroup> hi_acc: atomic<u32>;
var<workgroup> flat: u32;

// Note: no private arrays indexed in loops anywhere here (Metal spills them
// to memory); 16 counts travel as two vec4 of packed 16-bit lanes.

fn chv(x: u32, y: u32, z: u32) -> u32 {
    return (src[y * h.pw + x] >> (8u * z)) & 255u;
}

fn coarse_inc(v: u32) -> vec4<u32> {
    let bin = v >> 4u;
    return select(vec4<u32>(0u), vec4<u32>(1u << ((bin & 3u) * 8u)), vec4<u32>(bin >> 2u) == vec4<u32>(0u, 1u, 2u, 3u));
}

// Byte counts (bin 4w + b in byte b of lane w) to 16-bit lanes: `a` holds
// bins 4w (low) and 4w + 2 (high), `b` bins 4w + 1 and 4w + 3.
fn lo_half(v: vec4<u32>) -> vec4<u32> {
    return v & vec4<u32>(0x00ff00ffu);
}

fn hi_half(v: vec4<u32>) -> vec4<u32> {
    return (v >> vec4<u32>(8u)) & vec4<u32>(0x00ff00ffu);
}

// (bin, rank inside the bin) of rank k (< total) among 16 packed counts.
fn select16(a: vec4<u32>, b: vec4<u32>, k: u32) -> vec2<u32> {
    let m = vec4<u32>(0xffffu);
    let c0 = a & m;
    let c1 = b & m;
    let c2 = a >> vec4<u32>(16u);
    let c3 = b >> vec4<u32>(16u);
    let s = c0 + c1 + c2 + c3;
    let i1 = s.x;
    let i2 = i1 + s.y;
    let i3 = i2 + s.z;
    let g = u32(i1 <= k) + u32(i2 <= k) + u32(i3 <= k);
    let before = select(select(select(0u, i1, g == 1u), i2, g == 2u), i3, g == 3u);
    let oh = select(vec4<u32>(0u), vec4<u32>(1u), vec4<u32>(g) == vec4<u32>(0u, 1u, 2u, 3u));
    let q0 = dot(c0, oh);
    let q1 = dot(c1, oh);
    let q2 = dot(c2, oh);
    let kk = k - before;
    let j1 = q0;
    let j2 = j1 + q1;
    let j3 = j2 + q2;
    let j = u32(j1 <= kk) + u32(j2 <= kk) + u32(j3 <= kk);
    let jb = select(select(select(0u, j1, j == 1u), j2, j == 2u), j3, j == 3u);
    return vec2<u32>(4u * g + j, kk - jb);
}

fn put(x: u32, y: u32, z: u32, v: u32) {
    atomicOr(&outp[y * h.cw + x], v << (8u * z));
}

// Inclusive prefix sums of the strip's column histograms (byte counts `v`)
// into `ps` (entry c + 1 = sum of columns 0..=c, entry 0 = 0).
fn strip_prefix(v: vec4<u32>, L: u32) {
    let xa = lo_half(v);
    let xb = hi_half(v);
    let ex = wg_excl(xa, xb, L);
    ps[2u * L + 2u] = ex[0] + xa;
    ps[2u * L + 3u] = ex[1] + xb;
    if (L == 0u) {
        ps[0] = vec4<u32>(0u);
        ps[1] = vec4<u32>(0u);
    }
    workgroupBarrier();
}

fn fine_seg(o: u32, J: u32, c: u32) -> vec4<u32> {
    let i = o + 4u * J * h.s + c;
    return vec4<u32>(fine[i], fine[i + h.s], fine[i + 2u * h.s], fine[i + 3u * h.s]);
}

@compute @workgroup_size(256)
fn zero_out(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.y * nw.x * 256u + gid.x;
    if (i < h.cw * h.ch) {
        atomicStore(&outp[i], 0u);
    }
}

// `merge` (color clamped to alpha) in place.
@compute @workgroup_size(256)
fn clamp_out(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.y * nw.x * 256u + gid.x;
    if (i < h.cw * h.ch) {
        let c = unpack4(atomicLoad(&outp[i]));
        atomicStore(&outp[i], pack4(vec4<u32>(min(c.xyz, vec3<u32>(c.w)), c.w)));
    }
}

// (entering value, leaving value, their fine words) for output row y.
fn prefetch(gx: u32, y: u32, ok: bool, oc: u32, z: u32, r: u32) -> vec4<u32> {
    if (!ok) {
        return vec4<u32>(0u);
    }
    let vi = chv(gx, y + 2u * r, z);
    let vo = chv(gx, y - 1u, z);
    return vec4<u32>(vi, vo, fine[oc + (vi >> 2u) * h.s], fine[oc + (vo >> 2u) * h.s]);
}

@compute @workgroup_size(WG)
fn hist_median(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let L = scan_index(li);
    let z = wid.z;
    let S = h.s;
    let r = h.r;
    let n = 2u * r + 1u;
    let x0 = wid.x * h.tw;
    let oy0 = wid.y * h.th;
    let rows = min(h.th, h.ch - oy0);
    let c = L;
    let gx = x0 + c;
    let inside = gx < h.pw;
    let is_out = c < h.tw && gx < h.cw;
    // Scratch of this workgroup (column c at o + bin_group * S + c).
    let o = ((z * h.nchunks + wid.y) * h.nstrips + wid.x) * 64u * S;
    let kth = (n * n) / 2u;

    // A constant region (the alpha of an opaque layer, flat fills): the
    // median and every rank is that value.
    if (li == 0u) {
        atomicStore(&lo_acc, 255u);
        atomicStore(&hi_acc, 0u);
    }
    workgroupBarrier();
    var lo = 255u;
    var hi = 0u;
    if (inside) {
        for (var y = oy0; y < oy0 + rows + 2u * r; y++) {
            let v = chv(gx, y, z);
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    atomicMin(&lo_acc, lo);
    atomicMax(&hi_acc, hi);
    workgroupBarrier();
    if (li == 0u) {
        let a = atomicLoad(&lo_acc);
        flat = select(256u, a, a == atomicLoad(&hi_acc));
    }
    let fv = workgroupUniformLoad(&flat);
    if (fv < 256u) {
        if (is_out) {
            for (var i = 0u; i < rows; i++) {
                put(gx, oy0 + i, z, fv);
            }
        }
        return;
    }

    // Column histograms over rows oy0 ..= oy0 + 2r.
    var col = vec4<u32>(0u);
    for (var b = 0u; b < 64u; b++) {
        fine[o + b * S + c] = 0u;
    }
    if (inside) {
        for (var y = oy0; y <= oy0 + 2u * r; y++) {
            let v = chv(gx, y, z);
            col += coarse_inc(v);
            fine[o + (v >> 2u) * S + c] += 1u << ((v & 3u) * 8u);
        }
    }

    // The next row's entering / leaving values and their fine words are
    // loaded a row ahead, so their latency hides behind this row's work.
    var nx = prefetch(gx, oy0 + 1u, rows > 1u && inside, o + c, z, r);
    for (var i = 0u; i < rows; i++) {
        let y = oy0 + i;
        if (i > 0u && inside) {
            let vi = nx.x;
            let vo = nx.y;
            col = col + coarse_inc(vi) - coarse_inc(vo);
            let ii = o + (vi >> 2u) * S + c;
            let io = o + (vo >> 2u) * S + c;
            let di = 1u << ((vi & 3u) * 8u);
            let dd = 1u << ((vo & 3u) * 8u);
            if (ii == io) {
                fine[ii] = nx.z + di - dd;
            } else {
                fine[ii] = nx.z + di;
                fine[io] = nx.w - dd;
            }
            nx = prefetch(gx, y + 1u, i + 1u < rows, o + c, z, r);
        }
        storageBarrier();

        // Coarse window histogram: segment of the median and rank inside it.
        strip_prefix(col, L);
        var jj = vec2<u32>(0u);
        if (is_out) {
            jj = select16(ps[2u * (c + n)] - ps[2u * c], ps[2u * (c + n) + 1u] - ps[2u * c + 1u], kth);
            if (n > 17u) {
                atomicOr(&jm_acc, 1u << jj.x);
            }
        }
        // Small windows always sum their few columns directly (no need to
        // gather the row's segments).
        var mask = 0xffffu;
        if (n > 17u) {
            workgroupBarrier();
            if (li == 0u) {
                jm = atomicLoad(&jm_acc);
                atomicStore(&jm_acc, 0u);
            }
            mask = workgroupUniformLoad(&jm);
        }

        var res = 0u;
        if (countOneBits(mask) * 8u < n) {
            // One prefix sum per segment occurring in the row.
            var mm = mask;
            loop {
                if (mm == 0u) {
                    break;
                }
                let J = firstTrailingBit(mm);
                mm &= mm - 1u;
                strip_prefix(fine_seg(o, J, c), L);
                if (is_out && jj.x == J) {
                    res = 16u * J + select16(ps[2u * (c + n)] - ps[2u * c], ps[2u * (c + n) + 1u] - ps[2u * c + 1u], jj.y).x;
                }
                workgroupBarrier();
            }
        } else if (is_out) {
            // The 2r + 1 columns' fine counts of the segment, summed directly.
            var aa = vec4<u32>(0u);
            var ab = vec4<u32>(0u);
            for (var cc = c; cc < c + n; cc++) {
                let x = fine_seg(o, jj.x, cc);
                aa += lo_half(x);
                ab += hi_half(x);
            }
            res = 16u * jj.x + select16(aa, ab, jj.y).x;
        }
        if (is_out) {
            var m = res;
            if (h.mode == 1u) {
                let cv = chv(gx + r, y + r, z);
                m = select(cv, m, max(m, cv) - min(m, cv) > h.thr);
            }
            put(gx, y, z, m);
        }
        storageBarrier();
    }
}

// ----- Smart Blur (`smart_hist`) ----------------------------------------------------
// Mean of the window values within h.thr of the center: count and sum over
// [lo, hi]. Whole coarse segments strictly inside come from coarse counts
// and per-segment sums (also slid per column, prefix-summed in 32 bits);
// the two segments holding lo and hi from the fine counts.

// Coarse window sums, 32 bits per bin: entry c at 4c .. 4c + 3 holding
// bins 4w, 4w + 1, 4w + 2, 4w + 3 (lane w) of the inclusive prefix.
var<workgroup> pss: array<vec4<u32>, 4u * WG + 4u>;

// Count and sum of the values in [lo, hi] among 16 packed counts of the
// values base .. base + 15.
fn part16(a: vec4<u32>, b: vec4<u32>, base: u32, lo: u32, hi: u32) -> vec2<u32> {
    let m = vec4<u32>(0xffffu);
    let v0 = vec4<u32>(base) + vec4<u32>(0u, 4u, 8u, 12u);
    let v1 = v0 + vec4<u32>(1u);
    let v2 = v0 + vec4<u32>(2u);
    let v3 = v0 + vec4<u32>(3u);
    let l = vec4<u32>(lo);
    let hh = vec4<u32>(hi);
    let c0 = (a & m) * vec4<u32>((v0 >= l) & (v0 <= hh));
    let c1 = (b & m) * vec4<u32>((v1 >= l) & (v1 <= hh));
    let c2 = (a >> vec4<u32>(16u)) * vec4<u32>((v2 >= l) & (v2 <= hh));
    let c3 = (b >> vec4<u32>(16u)) * vec4<u32>((v3 >= l) & (v3 <= hh));
    let one = vec4<u32>(1u);
    return vec2<u32>(dot(c0 + c1 + c2 + c3, one), dot(c0 * v0 + c1 * v1 + c2 * v2 + c3 * v3, one));
}

// Bins of the 16-bit-pair sums (bin 4w + j: j even in `a`, odd in `b`,
// j >= 2 in the high half).
fn sum_inc(v: u32) -> array<vec4<u32>, 2> {
    let bin = v >> 4u;
    let j = bin & 3u;
    let d = select(vec4<u32>(0u), vec4<u32>(v << ((j >> 1u) * 16u)), vec4<u32>(bin >> 2u) == vec4<u32>(0u, 1u, 2u, 3u));
    let z = vec4<u32>(0u);
    return array<vec4<u32>, 2>(select(z, d, (j & 1u) == 0u), select(z, d, (j & 1u) == 1u));
}

@compute @workgroup_size(WG)
fn hist_smart(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let L = scan_index(li);
    let z = wid.z;
    let S = h.s;
    let r = h.r;
    let n = 2u * r + 1u;
    let x0 = wid.x * h.tw;
    let oy0 = wid.y * h.th;
    let rows = min(h.th, h.ch - oy0);
    let c = L;
    let gx = x0 + c;
    let inside = gx < h.pw;
    let is_out = c < h.tw && gx < h.cw;
    let o = ((z * h.nchunks + wid.y) * h.nstrips + wid.x) * 64u * S;

    if (li == 0u) {
        atomicStore(&lo_acc, 255u);
        atomicStore(&hi_acc, 0u);
    }
    workgroupBarrier();
    var lo = 255u;
    var hi = 0u;
    if (inside) {
        for (var y = oy0; y < oy0 + rows + 2u * r; y++) {
            let v = chv(gx, y, z);
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    atomicMin(&lo_acc, lo);
    atomicMax(&hi_acc, hi);
    workgroupBarrier();
    if (li == 0u) {
        let a = atomicLoad(&lo_acc);
        flat = select(256u, a, a == atomicLoad(&hi_acc));
    }
    let fv = workgroupUniformLoad(&flat);
    if (fv < 256u) {
        if (is_out) {
            for (var i = 0u; i < rows; i++) {
                put(gx, oy0 + i, z, fv);
            }
        }
        return;
    }

    var col = vec4<u32>(0u);
    var csa = vec4<u32>(0u);
    var csb = vec4<u32>(0u);
    for (var b = 0u; b < 64u; b++) {
        fine[o + b * S + c] = 0u;
    }
    if (inside) {
        for (var y = oy0; y <= oy0 + 2u * r; y++) {
            let v = chv(gx, y, z);
            col += coarse_inc(v);
            let d = sum_inc(v);
            csa += d[0];
            csb += d[1];
            fine[o + (v >> 2u) * S + c] += 1u << ((v & 3u) * 8u);
        }
    }

    for (var i = 0u; i < rows; i++) {
        let y = oy0 + i;
        if (i > 0u && inside) {
            let vi = chv(gx, y + 2u * r, z);
            let vo = chv(gx, y - 1u, z);
            col = col + coarse_inc(vi) - coarse_inc(vo);
            let di = sum_inc(vi);
            let dd = sum_inc(vo);
            csa = csa + di[0] - dd[0];
            csb = csb + di[1] - dd[1];
            fine[o + (vi >> 2u) * S + c] += 1u << ((vi & 3u) * 8u);
            fine[o + (vo >> 2u) * S + c] -= 1u << ((vo & 3u) * 8u);
        }
        storageBarrier();

        // Coarse counts (16-bit pairs) and sums (32-bit) across the strip.
        let m16 = vec4<u32>(0xffffu);
        let s0 = csa & m16;
        let s1 = csb & m16;
        let s2 = csa >> vec4<u32>(16u);
        let s3 = csb >> vec4<u32>(16u);
        let xa = lo_half(col);
        let xb = hi_half(col);
        let e1 = wg_excl(xa, xb, L);
        let e2 = wg_excl(s0, s1, L);
        let e3 = wg_excl(s2, s3, L);
        ps[2u * L + 2u] = e1[0] + xa;
        ps[2u * L + 3u] = e1[1] + xb;
        pss[4u * L + 4u] = e2[0] + s0;
        pss[4u * L + 5u] = e2[1] + s1;
        pss[4u * L + 6u] = e3[0] + s2;
        pss[4u * L + 7u] = e3[1] + s3;
        if (L == 0u) {
            ps[0] = vec4<u32>(0u);
            ps[1] = vec4<u32>(0u);
            pss[0] = vec4<u32>(0u);
            pss[1] = vec4<u32>(0u);
            pss[2] = vec4<u32>(0u);
            pss[3] = vec4<u32>(0u);
        }
        workgroupBarrier();

        // Whole segments strictly between those of lo and hi.
        var cnt = 0u;
        var sum = 0u;
        var vlo = 0u;
        var vhi = 0u;
        var jl = 0u;
        var jh = 0u;
        if (is_out) {
            let cv = chv(gx + r, y + r, z);
            vlo = cv - min(cv, h.thr);
            vhi = min(cv + h.thr, 255u);
            jl = vlo >> 4u;
            jh = vhi >> 4u;
            let ka = ps[2u * (c + n)] - ps[2u * c];
            let kb = ps[2u * (c + n) + 1u] - ps[2u * c + 1u];
            let b0 = vec4<u32>(0u, 4u, 8u, 12u);
            let m0 = vec4<u32>((b0 > vec4<u32>(jl)) & (b0 < vec4<u32>(jh)));
            let m1 = vec4<u32>((b0 + 1u > vec4<u32>(jl)) & (b0 + 1u < vec4<u32>(jh)));
            let m2 = vec4<u32>((b0 + 2u > vec4<u32>(jl)) & (b0 + 2u < vec4<u32>(jh)));
            let m3 = vec4<u32>((b0 + 3u > vec4<u32>(jl)) & (b0 + 3u < vec4<u32>(jh)));
            let one = vec4<u32>(1u);
            cnt = dot((ka & m16) * m0 + (kb & m16) * m1 + (ka >> vec4<u32>(16u)) * m2 + (kb >> vec4<u32>(16u)) * m3, one);
            let q0 = pss[4u * (c + n)] - pss[4u * c];
            let q1 = pss[4u * (c + n) + 1u] - pss[4u * c + 1u];
            let q2 = pss[4u * (c + n) + 2u] - pss[4u * c + 2u];
            let q3 = pss[4u * (c + n) + 3u] - pss[4u * c + 3u];
            sum = dot(q0 * m0 + q1 * m1 + q2 * m2 + q3 * m3, one);
            atomicOr(&jm_acc, (1u << jl) | (1u << jh));
        }
        workgroupBarrier();
        if (li == 0u) {
            jm = atomicLoad(&jm_acc);
            atomicStore(&jm_acc, 0u);
        }
        let mask = workgroupUniformLoad(&jm);

        if (countOneBits(mask) * 8u < n) {
            var mm = mask;
            loop {
                if (mm == 0u) {
                    break;
                }
                let J = firstTrailingBit(mm);
                mm &= mm - 1u;
                strip_prefix(fine_seg(o, J, c), L);
                if (is_out && (jl == J || jh == J)) {
                    let p = part16(ps[2u * (c + n)] - ps[2u * c], ps[2u * (c + n) + 1u] - ps[2u * c + 1u], 16u * J, vlo, vhi);
                    cnt += p.x;
                    sum += p.y;
                }
                workgroupBarrier();
            }
        } else if (is_out) {
            var aa = vec4<u32>(0u);
            var ab = vec4<u32>(0u);
            var ha = vec4<u32>(0u);
            var hb = vec4<u32>(0u);
            for (var cc = c; cc < c + n; cc++) {
                let x = fine_seg(o, jl, cc);
                aa += lo_half(x);
                ab += hi_half(x);
                if (jh != jl) {
                    let x2 = fine_seg(o, jh, cc);
                    ha += lo_half(x2);
                    hb += hi_half(x2);
                }
            }
            let p = part16(aa, ab, 16u * jl, vlo, vhi);
            let p2 = part16(ha, hb, 16u * jh, vlo, vhi);
            cnt += p.x + p2.x;
            sum += p.y + p2.y;
        }
        if (is_out) {
            put(gx, y, z, (sum + cnt / 2u) / cnt);
        }
        storageBarrier();
    }
}
