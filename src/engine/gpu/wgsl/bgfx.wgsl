// Blur Gallery effects (`extra::FxPrep::apply`): Light Bokeh, Bokeh Color,
// Light Range, and the Noise panel. Included by the `vblur` and `bgather`
// modules, which declare `fxw` (the Light Range weights, `FxPrep::wr`).

struct Fx {
    bokeh: f32,
    sat: f32,
    amp: f32,
    size: f32,
    rough: f32,
    col: f32,
    keep: f32,
    gaussian: u32,
}

// `extra::fx_luma8`: integer straight-color luminance.
fn fx_luma8(p: u32) -> u32 {
    let c = unpack4(p);
    if (c.w == 0u) {
        return 0u;
    }
    let lp = (77u * c.x + 150u * c.y + 29u * c.z + 128u) >> 8u;
    return min((lp * 255u + c.w / 2u) / c.w, 255u);
}

fn fx_lattice(fx: Fx, x: i32, y: i32, ch: u32) -> f32 {
    let s = 7919u + ch * 3u;
    if (fx.gaussian != 0u) {
        return (hash01(x, y, s) + hash01(x, y, s + 1u) + hash01(x, y, s + 2u) - 1.5) * 1.1547;
    }
    return hash01(x, y, s) * 2.0 - 1.0;
}

fn fx_one(fx: Fx, ix: i32, iy: i32, nx: i32, ny: i32, sx: f32, sy: f32, ch: u32) -> f32 {
    let v00 = fx_lattice(fx, ix, iy, ch);
    let v10 = fx_lattice(fx, ix + 1, iy, ch);
    let v01 = fx_lattice(fx, ix, iy + 1, ch);
    let v11 = fx_lattice(fx, ix + 1, iy + 1, ch);
    let top = v00 + (v10 - v00) * sx;
    let bot = v01 + (v11 - v01) * sx;
    let sm = top + (bot - top) * sy;
    let near = fx_lattice(fx, nx, ny, ch);
    return sm + (near - sm) * fx.rough;
}

// `FxPrep::grain`.
fn fx_grain(fx: Fx, gx: f32, gy: f32) -> vec3<f32> {
    let u = p_div(gx, fx.size) - 0.5;
    let v = p_div(gy, fx.size) - 0.5;
    let fu = floor(u);
    let fv = floor(v);
    let fx_ = u - fu;
    let fy_ = v - fv;
    let ix = i32(fu);
    let iy = i32(fv);
    let sx = fx_ * fx_ * (3.0 - 2.0 * fx_);
    let sy = fy_ * fy_ * (3.0 - 2.0 * fy_);
    let nx = ix + select(0, 1, fx_ >= 0.5);
    let ny = iy + select(0, 1, fy_ >= 0.5);
    var mono = 0.0;
    if (fx.col < 1.0) {
        mono = fx_one(fx, ix, iy, nx, ny, sx, sy, 0u);
    }
    var out = vec3<f32>(mono);
    if (fx.col > 0.0) {
        out = vec3<f32>(
            mono + (fx_one(fx, ix, iy, nx, ny, sx, sy, 1u) - mono) * fx.col,
            mono + (fx_one(fx, ix, iy, nx, ny, sx, sy, 2u) - mono) * fx.col,
            mono + (fx_one(fx, ix, iy, nx, ny, sx, sy, 3u) - mono) * fx.col,
        );
    }
    return out;
}

// `FxPrep::apply`: `v` is the unrounded blurred pixel, `src` the source
// pixel, `tb` the local blur amount, `(gx, gy)` the level-0 pixel center.
fn fx_apply(fx: Fx, v: vec4<f32>, src: u32, tb: f32, gx: f32, gy: f32) -> u32 {
    let a8 = u32(clamp(v.w + 0.5, 0.0, 255.0));
    let af = f32(a8);
    var c = v.xyz;
    if (fx.bokeh > 0.0 && tb > 0.0) {
        let k = fx.bokeh * tb * fxw[fx_luma8(src)];
        if (k > 0.0) {
            c = c * (1.0 + k);
            if (fx.sat > 0.0) {
                let l = 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
                let s = 1.0 + fx.sat * k;
                c = vec3<f32>(l) + (c - vec3<f32>(l)) * s;
            }
        }
    }
    if (fx.amp > 0.0 && tb > 0.0 && a8 > 0u) {
        let lum = p_div((0.299 * c.x + 0.587 * c.y + 0.114 * c.z) * 255.0, af);
        let hw = clamp((lum - 192.0) / 63.0, 0.0, 1.0);
        let att = 1.0 - (1.0 - fx.keep) * hw;
        let scale = fx.amp * tb * att * (af / 255.0);
        c = c + fx_grain(fx, gx, gy) * scale;
    }
    let q = vec3<u32>(clamp(c + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(af)));
    return pack4(vec4<u32>(q, a8));
}
