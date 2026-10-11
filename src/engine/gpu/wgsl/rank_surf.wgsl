// Edge-preserving blurs of `filters::rank::surface`:
// - Smart Blur (`smart_hist`): mean of the window values within the
//   threshold of the center, (sum + n / 2) / n per channel. Integer, exact.
// - Surface Blur / Reduce Noise (`pbfic`): Yang's constant-time bilateral
//   filter as the CPU runs it, with the same intensity levels and tent
//   weights 1 - |v - c_k| / band. Each output interpolates the box-filtered
//   ratios sum(w v) / sum(w) of the two levels bracketing its value, so a
//   tile only needs the levels its own values span. Float (sums in another
//   order than the CPU): within 1/255.

struct Q {
    pw: u32,
    ph: u32,
    cw: u32,
    ch: u32,
    r: u32,
    // Smart Blur: integer threshold. Surface: level count.
    ti: u32,
    // Surface: tile size.
    tw: u32,
    th: u32,
    // Surface: f32 bits of the level step and 1 / band.
    step: u32,
    inv_band: u32,
    a0: u32,
    a1: u32,
}

@group(0) @binding(0) var<uniform> q: Q;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(4) var<storage, read_write> outp: array<atomic<u32>>;

// ----- Smart Blur ------------------------------------------------------------------

var<workgroup> tile: array<u32, 4096>;

// 16 x 16 outputs; the apron sits in shared memory for r <= 24.
@compute @workgroup_size(16, 16)
fn smart_direct(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let r = q.r;
    let n = 16u + 2u * r;
    let bx = wid.x * 16u;
    let by = wid.y * 16u;
    let shared_tile = r <= 24u;
    if (shared_tile) {
        for (var t = li; t < n * n; t += 256u) {
            let x = min(bx + t % n, q.pw - 1u);
            let y = min(by + t / n, q.ph - 1u);
            tile[t] = src[y * q.pw + x];
        }
    }
    workgroupBarrier();
    let x = bx + lid.x;
    let y = by + lid.y;
    if (x >= q.cw || y >= q.ch) {
        return;
    }
    let c = vec4<i32>(unpack4(src[(y + r) * q.pw + x + r]));
    let ti = vec4<i32>(i32(q.ti));
    var cnt = vec4<u32>(0u);
    var sum = vec4<u32>(0u);
    let k = 2u * r + 1u;
    for (var dy = 0u; dy < k; dy++) {
        for (var dx = 0u; dx < k; dx++) {
            var p: u32;
            if (shared_tile) {
                p = tile[(lid.y + dy) * n + lid.x + dx];
            } else {
                p = src[(y + dy) * q.pw + x + dx];
            }
            let v = unpack4(p);
            let inside = abs(vec4<i32>(v) - c) <= ti;
            cnt += select(vec4<u32>(0u), vec4<u32>(1u), inside);
            sum += select(vec4<u32>(0u), v, inside);
        }
    }
    let m = (sum + cnt / vec4<u32>(2u)) / cnt;
    dst[y * q.cw + x] = pack4(vec4<u32>(min(m.xyz, vec3<u32>(m.w)), m.w));
}

// ----- Surface Blur (PBFIC) ------------------------------------------------------------
// One channel per workgroup and a q.tw x q.th output tile (sized per radius
// on the CPU: rows x tw <= 2560, tw x th <= 1024). Per level: horizontal
// running sums of w and w v for the tile rows plus apron, one row per
// thread; then vertical running sums by walkers of q.a0 outputs of a
// column. Each output keeps the previous level's ratio and its result in
// shared memory.

var<workgroup> hw: array<f32, 2560>;
var<workgroup> hv: array<f32, 2560>;
var<workgroup> rp: array<f32, 1024>;
var<workgroup> rs: array<f32, 1024>;
var<workgroup> lvl_lo: atomic<u32>;
var<workgroup> lvl_hi: atomic<u32>;
var<workgroup> a_lo: atomic<u32>;
var<workgroup> a_hi: atomic<u32>;
var<workgroup> lv: vec2<u32>;

fn chan(x: u32, y: u32, z: u32) -> u32 {
    return (src[min(y, q.ph - 1u) * q.pw + min(x, q.pw - 1u)] >> (8u * z)) & 255u;
}

fn wt(v: f32, ck: f32, inv_band: f32) -> f32 {
    return max(1.0 - abs(v - ck) * inv_band, 0.0);
}

@compute @workgroup_size(256)
fn surface(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let z = wid.z;
    let r = q.r;
    let k = 2u * r + 1u;
    let tw = q.tw;
    let th = q.th;
    let rows = th + 2u * r;
    let ox0 = wid.x * tw;
    let oy0 = wid.y * th;
    let step = bitcast<f32>(q.step);
    let inv_band = bitcast<f32>(q.inv_band);
    let levels = q.ti;
    let per = q.a0;
    let nseg = (th + per - 1u) / per;

    if (li == 0u) {
        atomicStore(&lvl_lo, 255u);
        atomicStore(&lvl_hi, 0u);
        atomicStore(&a_lo, 255u);
        atomicStore(&a_hi, 0u);
    }
    workgroupBarrier();
    // Values of the outputs (levels needed) and of the whole apron (flat?).
    var lo = 255u;
    var hi = 0u;
    for (var t = li; t < tw * th; t += 256u) {
        let x = ox0 + t % tw;
        let y = oy0 + t / tw;
        if (x < q.cw && y < q.ch) {
            let v = chan(x + r, y + r, z);
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    atomicMin(&lvl_lo, lo);
    atomicMax(&lvl_hi, hi);
    lo = 255u;
    hi = 0u;
    let aw = tw + 2u * r;
    for (var t = li; t < rows * aw; t += 256u) {
        let v = chan(ox0 + t % aw, oy0 + t / aw, z);
        lo = min(lo, v);
        hi = max(hi, v);
    }
    atomicMin(&a_lo, lo);
    atomicMax(&a_hi, hi);
    workgroupBarrier();
    if (li == 0u) {
        let alo = atomicLoad(&a_lo);
        if (alo == atomicLoad(&a_hi)) {
            // Constant apron: every ratio is that value.
            lv = vec2<u32>(alo, 0xffffffffu);
        } else {
            let cmin = f32(atomicLoad(&lvl_lo));
            let cmax = f32(atomicLoad(&lvl_hi));
            var l0 = min(u32(floor(cmin / step)), levels - 1u);
            var l1 = max(min(u32(ceil(cmax / step)), levels - 1u), l0);
            // At least one interval (a center exactly on a level).
            if (l1 == l0) {
                if (l1 + 1u < levels) {
                    l1 += 1u;
                } else {
                    l0 -= 1u;
                }
            }
            lv = vec2<u32>(l0, l1);
        }
    }
    let lr = workgroupUniformLoad(&lv);
    if (lr.y == 0xffffffffu) {
        for (var t = li; t < tw * th; t += 256u) {
            let x = ox0 + t % tw;
            let y = oy0 + t / tw;
            if (x < q.cw && y < q.ch) {
                atomicOr(&outp[y * q.cw + x], lr.x << (8u * z));
            }
        }
        return;
    }

    for (var l = lr.x; l <= lr.y; l++) {
        let ck = f32(l) * step;
        // Horizontal running sums, one row per thread.
        for (var row = li; row < rows; row += 256u) {
            let y = oy0 + row;
            var sw = 0.0;
            var sv = 0.0;
            for (var t = 0u; t < k; t++) {
                let v = f32(chan(ox0 + t, y, z));
                let w = wt(v, ck, inv_band);
                sw += w;
                sv += w * v;
            }
            hw[row * tw] = sw;
            hv[row * tw] = sv;
            for (var x = 1u; x < tw; x++) {
                let va = f32(chan(ox0 + x + k - 1u, y, z));
                let vs = f32(chan(ox0 + x - 1u, y, z));
                let wa = wt(va, ck, inv_band);
                let ws = wt(vs, ck, inv_band);
                sw += wa - ws;
                sv += wa * va - ws * vs;
                hw[row * tw + x] = sw;
                hv[row * tw + x] = sv;
            }
        }
        workgroupBarrier();
        // Vertical running sums: walker li owns column j, rows i0 .. i1.
        if (li < tw * nseg) {
            let j = li % tw;
            let i0 = (li / tw) * per;
            let i1 = min(i0 + per, th);
            var vw = 0.0;
            var vv = 0.0;
            for (var t = 0u; t < k; t++) {
                vw += hw[(i0 + t) * tw + j];
                vv += hv[(i0 + t) * tw + j];
            }
            for (var i = i0; i < i1; i++) {
                if (i > i0) {
                    vw += hw[(i + k - 1u) * tw + j] - hw[(i - 1u) * tw + j];
                    vv += hv[(i + k - 1u) * tw + j] - hv[(i - 1u) * tw + j];
                }
                let cur = select(ck, vv / vw, vw > 1e-6);
                let c = f32(chan(ox0 + j + r, oy0 + i + r, z));
                let o = i * tw + j;
                if (l > lr.x && c >= ck - step && c <= ck) {
                    let f = (c - (ck - step)) / step;
                    rs[o] = rp[o] + (cur - rp[o]) * f;
                }
                rp[o] = cur;
            }
        }
        workgroupBarrier();
    }
    for (var t = li; t < tw * th; t += 256u) {
        let x = ox0 + t % tw;
        let y = oy0 + t / tw;
        if (x < q.cw && y < q.ch) {
            let o = u32(clamp(rs[t] + 0.5, 0.0, 255.0));
            atomicOr(&outp[y * q.cw + x], o << (8u * z));
        }
    }
}
