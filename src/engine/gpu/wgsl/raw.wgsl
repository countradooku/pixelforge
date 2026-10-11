// Camera Raw Filter (`extra::camera_raw`), a float pipeline:
// - `raw_lum`: the 8-bit luminance plane the CPU blurs (`blur_f`), made
//   exact: the CPU's float luma rounds like (299 r + 587 g + 114 b + 500) /
//   1000 except at exact ties, which a CPU-built table lists per (r, g);
// - the blurs run through the `blur` module (bit-exact `gauss_plane`);
// - `raw_main`: the per-pixel develop, the CPU's f32 expressions in the
//   CPU's order, `p_exp` for powers of two, the CPU hash for grain.

struct P {
    w: u32,
    h: u32,
    pad: u32,
    cw: u32,
    ch: u32,
    ox: i32,
    oy: i32,
    flags: u32,
    s: f32,
    img_w: f32,
    img_h: f32,
    ex: f32,
    wb: vec4<f32>,
    // contrast, highlights, shadows, whites
    t0: vec4<f32>,
    // blacks, texture factor, clarity, dehaze
    t1: vec4<f32>,
    // noise reduction, vibrance, saturation, balance pivot
    t2: vec4<f32>,
    // curve amounts in region order (curve[3], curve[2], curve[1], curve[0])
    curve: vec4<f32>,
    hue0: vec4<f32>,
    hue1: vec4<f32>,
    sat0: vec4<f32>,
    sat1: vec4<f32>,
    lum0: vec4<f32>,
    lum1: vec4<f32>,
    gs: vec4<f32>,
    gh: vec4<f32>,
    // vignette, grain, linear e0, linear e1
    fx: vec4<f32>,
    // linear dx, dy, ev, radial ev
    lin: vec4<f32>,
    // radial cx, cy, radius (>= 1)
    rad: vec4<f32>,
}

const F_HSL: u32 = 1u;
const F_DEHAZE: u32 = 2u;
const F_LINEAR: u32 = 4u;
const F_RADIAL: u32 = 8u;
const F_VIGNETTE: u32 = 16u;
const F_GRAIN: u32 = 32u;

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
// raw_lum: tie table; raw_main: the blurred planes (lane 0).
@group(0) @binding(3) var<storage, read> tab: array<u32>;
@group(0) @binding(4) var<storage, read> unpremul: array<u32>;
@group(0) @binding(5) var<storage, read> small: array<u32>;
@group(0) @binding(6) var<storage, read> big: array<u32>;

fn unpremul_ch(v: u32, a: u32) -> u32 {
    let i = a * 256u + v;
    return (unpremul[i >> 2u] >> ((i & 3u) * 8u)) & 255u;
}

// `px::unpremultiply` (0 for transparent pixels).
fn straight(c: vec4<u32>) -> vec3<u32> {
    if (c.w == 255u) {
        return c.xyz;
    }
    if (c.w == 0u) {
        return vec3<u32>(0u);
    }
    return vec3<u32>(unpremul_ch(c.x, c.w), unpremul_ch(c.y, c.w), unpremul_ch(c.z, c.w));
}

@compute @workgroup_size(16, 16)
fn raw_lum(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.w || gid.y >= p.h) {
        return;
    }
    let i = gid.y * p.w + gid.x;
    let c = straight(unpack4(src[i]));
    var q = (c.x * 299u + c.y * 587u + c.z * 114u + 500u) / 1000u;
    let t = tab[c.x * 256u + c.y];
    if (t != 0u && (t & 65535u) - 1u == c.z) {
        q = t >> 16u;
    }
    dst[i] = q;
}

fn sstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn luma(c: vec3<f32>) -> f32 {
    return 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
}

fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let r = c.x;
    let g = c.y;
    let b = c.z;
    let mx = max(max(r, g), b);
    let mn = min(min(r, g), b);
    let l = (mx + mn) * 0.5;
    if (mx - mn < 1e-6) {
        return vec3<f32>(0.0, 0.0, l);
    }
    let d = mx - mn;
    var s: f32;
    if (l > 0.5) {
        s = d / (2.0 - mx - mn);
    } else {
        s = d / (mx + mn);
    }
    var h: f32;
    if (mx == r) {
        h = (g - b) / d + select(0.0, 6.0, g < b);
    } else if (mx == g) {
        h = (b - r) / d + 2.0;
    } else {
        h = (r - g) / d + 4.0;
    }
    return vec3<f32>(h / 6.0, s, l);
}

fn fract_euclid(t: f32) -> f32 {
    return t - floor(t);
}

fn hue_ch(pp: f32, q: f32, tin: f32) -> f32 {
    let t = fract_euclid(tin);
    if (t < 1.0 / 6.0) {
        return pp + (q - pp) * 6.0 * t;
    } else if (t < 0.5) {
        return q;
    } else if (t < 2.0 / 3.0) {
        return pp + (q - pp) * (2.0 / 3.0 - t) * 6.0;
    }
    return pp;
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    if (s <= 0.0) {
        return vec3<f32>(l);
    }
    var q: f32;
    if (l < 0.5) {
        q = l * (1.0 + s);
    } else {
        q = l + s - l * s;
    }
    let pp = 2.0 * l - q;
    return vec3<f32>(hue_ch(pp, q, h + 1.0 / 3.0), hue_ch(pp, q, h), hue_ch(pp, q, h - 1.0 / 3.0));
}


fn band_amount(k: u32) -> vec3<f32> {
    if (k < 4u) {
        return vec3<f32>(p.hue0[k], p.sat0[k], p.lum0[k]);
    }
    return vec3<f32>(p.hue1[k - 4u], p.sat1[k - 4u], p.lum1[k - 4u]);
}

fn q8(v: f32) -> u32 {
    return u32(floor(clamp(v, 0.0, 1.0) * 255.0 + 0.5));
}

@compute @workgroup_size(16, 16)
fn raw_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.cw || gid.y >= p.ch) {
        return;
    }
    let bx = gid.x + p.pad;
    let by = gid.y + p.pad;
    let i = by * p.w + bx;
    let pix = src[i];
    let o = gid.y * p.cw + gid.x;
    let a = pix >> 24u;
    if (a == 0u) {
        dst[o] = pix;
        return;
    }
    let cs = straight(unpack4(pix));
    let gx = (f32(p.ox + i32(bx)) + 0.5) * p.s;
    let gy = (f32(p.oy + i32(by)) + 0.5) * p.s;
    var mask_ev = 0.0;
    if ((p.flags & F_LINEAR) != 0u) {
        let t = ((gx / p.img_w - 0.5) * p.lin.x + (gy / p.img_h - 0.5) * p.lin.y) + 0.5;
        mask_ev += p.lin.z * sstep(p.fx.z, p.fx.w, t);
    }
    if ((p.flags & F_RADIAL) != 0u) {
        let dx = gx - p.rad.x;
        let dy = gy - p.rad.y;
        let d = sqrt(dx * dx + dy * dy) / p.rad.z;
        mask_ev += p.lin.w * sstep(1.0, 0.6, d);
    }
    var ex = p.ex;
    if (mask_ev != 0.0) {
        ex = ex * p_exp(mask_ev * 0.6931471805599453);
    }
    var c = vec3<f32>(f32(cs.x) / 255.0 * p.wb.x * ex, f32(cs.y) / 255.0 * p.wb.y * ex, f32(cs.z) / 255.0 * p.wb.z * ex);
    let y = max(luma(c), 1e-4);
    // Basic tone.
    var ny = (y - 0.5) * (1.0 + p.t0.x / 100.0) + 0.5;
    let hl = p.t0.y;
    ny += hl / 100.0 * 0.35 * sstep(0.45, 1.0, ny) * select(1.0 - ny, ny, hl < 0.0);
    let sh = p.t0.z;
    ny += sh / 100.0 * 0.35 * sstep(0.55, 0.0, ny) * select(ny, 1.0 - ny, sh > 0.0);
    ny += p.t0.w / 100.0 * 0.25 * sstep(0.7, 1.0, ny);
    ny += p.t1.x / 100.0 * 0.25 * sstep(0.3, 0.0, ny);
    // Parametric curve.
    let r0 = sstep(0.25, 0.0, ny);
    let r1 = max(1.0 - (abs(ny - 0.3) / 0.25), 0.0);
    let r2 = max(1.0 - (abs(ny - 0.7) / 0.25), 0.0);
    let r3 = sstep(0.75, 1.0, ny);
    ny += r0 * p.curve.x / 100.0 * 0.25;
    ny += r1 * p.curve.y / 100.0 * 0.25;
    ny += r2 * p.curve.z / 100.0 * 0.25;
    ny += r3 * p.curve.w / 100.0 * 0.25;
    // Local contrast, sharpening and noise reduction.
    let l = (0.299 * f32(cs.x) + 0.587 * f32(cs.y) + 0.114 * f32(cs.z)) / 255.0;
    let sm = f32(small[i] & 255u) / 255.0;
    let bg = f32(big[i] & 255u) / 255.0;
    let detail = l - sm;
    ny += detail * p.t1.y;
    ny -= detail * p.t2.x / 100.0 * 0.9;
    ny += (l - bg) * p.t1.z / 100.0 * 1.2;
    if ((p.flags & F_DEHAZE) != 0u) {
        let veil = bg * 0.6;
        let k = p.t1.w / 100.0;
        ny = (ny - veil * k) / max(1.0 - veil * k, 0.2);
    }
    let ratio = max(ny, 0.0) / y;
    c = c * ratio;
    // HSL panel.
    if ((p.flags & F_HSL) != 0u) {
        let hsl = rgb_to_hsl(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
        let hd = hsl.x * 360.0;
        var centers = array<f32, 8>(0.0, 30.0, 60.0, 120.0, 180.0, 225.0, 270.0, 315.0);
        var dd = vec3<f32>(0.0);
        for (var k = 0u; k < 8u; k++) {
            let d = abs((hd - centers[k] + 540.0) % 360.0 - 180.0);
            let width = select(45.0, 30.0, k < 3u);
            dd += sstep(width, 0.0, d) * band_amount(k);
        }
        let s2 = clamp(hsl.y * (1.0 + dd.y / 100.0), 0.0, 1.0);
        let l2 = clamp(hsl.z + dd.z / 100.0 * 0.3 * hsl.y, 0.0, 1.0);
        c = hsl_to_rgb(fract_euclid(hsl.x + dd.x / 360.0 * 0.3), s2, l2);
    }
    // Color grading.
    let yy = luma(c);
    let pivot = p.t2.w;
    let ws = sstep(pivot, 0.0, yy);
    let wh = sstep(pivot, 1.0, yy);
    c = c + p.gs.xyz * ws + p.gh.xyz * wh;
    // Vibrance and saturation.
    let mx = max(max(c.x, c.y), c.z);
    let mn = min(min(c.x, c.y), c.z);
    var cur = 0.0;
    if (mx > 0.0) {
        cur = (mx - mn) / mx;
    }
    let kk = 1.0 + p.t2.z / 100.0 + p.t2.y / 100.0 * (1.0 - cur);
    let y2 = luma(c);
    c = vec3<f32>(y2) + (c - vec3<f32>(y2)) * kk;
    // Effects.
    if ((p.flags & F_VIGNETTE) != 0u) {
        let dx = gx / p.img_w - 0.5;
        let dy = gy / p.img_h - 0.5;
        let d2 = (dx * dx + dy * dy) * 2.0;
        c = c * (1.0 + p.fx.x / 100.0 * min(d2, 1.0));
    }
    if ((p.flags & F_GRAIN) != 0u) {
        let n = (hash01(i32(gx), i32(gy), 777u) - 0.5) * p.fx.y / 100.0 * 0.25;
        c = c + vec3<f32>(n);
    }
    dst[o] = premul(vec3<u32>(q8(c.x), q8(c.y), q8(c.z)), a);
}
