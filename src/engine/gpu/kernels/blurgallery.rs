//! Blur Gallery (Field Blur, Iris Blur, Tilt-Shift, Spin Blur, Path Blur,
//! with the Effects and Noise panels) and Blur ▸ Radial Blur.
//!
//! - Field / Iris / Tilt-Shift (`extra::variable_blur`): a padded kernel
//!   that builds the four Gaussian levels of the CPU's stack with the same
//!   integer passes (bit-exact), then interpolates per pixel and applies the
//!   effects in floats like the CPU.
//! - Spin / Radial / Path (`distort::render_radial` / `render_path_blur`): a
//!   gather from the resident source and its mips, with every per-tap
//!   constant (rotation steps, zoom factors, path offsets, mip thresholds)
//!   computed on the CPU with the CPU's own expressions.

use std::sync::Arc;

use super::{Input, Io, Kernel, Target, Temps};
use crate::engine::filters::extra::{FxPrep, VBLUR_LEVELS, vblur_sigma};
use crate::engine::filters::{BlurFx, Filter, blur, distort, fast};
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, Timer, dispatch, groups, table};
use crate::engine::gpu::shaders::PipeKey;

pub fn supports(f: &Filter) -> bool {
    matches!(f, Filter::FieldBlur { .. } | Filter::IrisBlur { .. } | Filter::TiltShift { .. } | Filter::SpinBlur { .. } | Filter::PathBlur { .. } | Filter::RadialBlur { .. })
}

pub fn plan(g: &Arc<Gpu>, f: &Filter, level: u32, _target: Target) -> Option<Box<dyn Kernel>> {
    let s = (1u32 << level) as f32;
    let sc = |v: f32| v / s;
    let z = [0f32; 4];
    match *f {
        Filter::FieldBlur { pins, fx } => {
            let mx = sc(pins.iter().map(|p| p[2]).fold(0.0, f32::max));
            let q = |i: usize| [pins[i][0], pins[i][1], pins[i][2], 0.0];
            Some(Box::new(VBlurK::new(g, 0, mx, s, [q(0), q(1), q(2)], &fx)))
        }
        Filter::TiltShift { cy, band, transition, blur, fx } => {
            let mx = sc(blur);
            Some(Box::new(VBlurK::new(g, 1, mx, s, [[cy, band, transition.max(1.0), 0.0], z, z], &fx)))
        }
        Filter::IrisBlur { cx, cy, radius, blur, fx } => {
            let mx = sc(blur);
            Some(Box::new(VBlurK::new(g, 2, mx, s, [[cx, cy, radius.max(1.0), 0.0], z, z], &fx)))
        }
        Filter::RadialBlur { amount, zoom, cx, cy } => Some(Box::new(GatherK::radial(g, amount, zoom, cx, cy, level, None, &BlurFx::default()))),
        Filter::SpinBlur { cx, cy, rx, ry, amount, fx } => {
            // The CPU copies blocks outside the ellipse's bounds; that equals
            // the per-pixel mask only when the radii are at least 1.
            if rx < 1.0 || ry < 1.0 {
                return None;
            }
            Some(Box::new(GatherK::radial(g, amount, false, cx, cy, level, Some([cx, cy, rx.max(1.0), ry.max(1.0)]), &fx)))
        }
        Filter::PathBlur { x0, y0, x1, y1, bend, length, fx } => Some(Box::new(GatherK::path(g, (x0, y0), (x1, y1), bend, length, level, &fx))),
        _ => None,
    }
}

/// `FxPrep` as uniform words (`bgfx.wgsl`'s `Fx`) and its weight table.
fn fx_words(g: &Gpu, fx: &BlurFx) -> (bool, [u32; 8], wgpu::Buffer) {
    let p: FxPrep = fx.prep();
    let words = [p.bokeh.to_bits(), p.sat.to_bits(), p.amp.to_bits(), p.size.to_bits(), p.rough.to_bits(), p.col.to_bits(), p.keep.to_bits(), p.gaussian as u32];
    (p.active(), words, table(g, bytemuck::cast_slice(&p.wr)))
}

// ----- Field / Iris / Tilt-Shift -------------------------------------------------------------

/// How one stack level is computed (`fast::gauss_plane`).
enum Level {
    /// sigma < 0.3: the source itself.
    Source,
    Direct(Vec<u32>),
    /// Box radii (vertical passes, then horizontal).
    Boxes(Vec<u32>),
}

pub struct VBlurK {
    kind: u32,
    mx: f32,
    scale: f32,
    pins: [[f32; 4]; 3],
    levels: Vec<Level>,
    fxon: bool,
    fx: [u32; 8],
    fxw: wgpu::Buffer,
}

impl VBlurK {
    fn new(g: &Gpu, kind: u32, mx: f32, scale: f32, pins: [[f32; 4]; 3], fx: &BlurFx) -> VBlurK {
        let levels = (1..VBLUR_LEVELS)
            .map(|k| {
                let sigma = vblur_sigma(mx, k);
                if sigma < 0.3 {
                    Level::Source
                } else if sigma <= fast::DIRECT_SIGMA_MAX {
                    Level::Direct(fast::gauss_kernel(sigma).iter().map(|&v| v as u32).collect())
                } else {
                    Level::Boxes(blur::boxes_for_gauss(sigma).iter().map(|&b| (b / 2) as u32).filter(|&r| r > 0).collect())
                }
            })
            .collect();
        let (fxon, fxw_words, fxw) = fx_words(g, fx);
        VBlurK { kind, mx, scale, pins, levels, fxon, fx: fxw_words, fxw }
    }

    fn params(&self, io: &Io, r: u32, k: u32, taps: u32) -> Vec<u32> {
        let inv = if r > 0 { ((1u64 << 24) / (2 * r as u64 + 1)) as u32 } else { 0 };
        let mut v = vec![
            io.pw,
            io.ph,
            r,
            inv,
            k,
            taps,
            io.pad,
            io.cw(),
            io.ch(),
            io.origin.0 as u32,
            io.origin.1 as u32,
            self.kind,
            self.mx.to_bits(),
            self.scale.to_bits(),
            self.fxon as u32,
            0,
        ];
        for q in &self.pins {
            v.extend(q.iter().map(|f| f.to_bits()));
        }
        v.extend_from_slice(&self.fx);
        v
    }
}

impl Kernel for VBlurK {
    fn temps(&self) -> usize {
        // One per stack level plus a ping-pong scratch.
        VBLUR_LEVELS
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let key = |entry| PipeKey { module: "vblur", entry, f16: false, consts: Vec::new() };
        let (w, h) = (io.pw, io.ph);
        if self.mx < 0.4 {
            return dispatch(g, enc, timer, &key("crop"), &self.params(io, 0, 0, 0), &[(1, io.src), (2, io.dst)], [groups(io.cw(), 16), groups(io.ch(), 16), 1]);
        }
        let scratch = VBLUR_LEVELS - 1;
        let mut out: Vec<&wgpu::Buffer> = Vec::new();
        for (li, lev) in self.levels.iter().enumerate() {
            match lev {
                Level::Source => out.push(io.src),
                Level::Direct(taps) => {
                    let t = table(g, bytemuck::cast_slice(taps));
                    let r = (taps.len() / 2) as u32;
                    let o = tmp.at(li);
                    dispatch(g, enc, timer.as_deref_mut(), &key("direct"), &self.params(io, r, 0, taps.len() as u32), &[(1, io.src), (2, o), (8, &t)], [groups(w, 16), groups(h, 16), 1])?;
                    out.push(o);
                }
                Level::Boxes(radii) => {
                    let passes: Vec<(u32, bool)> = radii.iter().map(|&r| (r, false)).chain(radii.iter().map(|&r| (r, true))).collect();
                    // Alternate so the last pass lands in this level's buffer.
                    let mut to_level = passes.len() % 2 == 1;
                    let mut input: &wgpu::Buffer = io.src;
                    // Rows / columns per thread (measured at 16.7 MP; 32-128
                    // were within noise of these).
                    let (kv, kh) = (16, 64);
                    for (r, horizontal) in passes {
                        let o = if to_level { tmp.at(li) } else { tmp.at(scratch) };
                        if horizontal {
                            dispatch(g, enc, timer.as_deref_mut(), &key("box_h"), &self.params(io, r, kh, 0), &[(1, input), (2, o)], [groups(w, kh), groups(h, 64), 1])?;
                        } else {
                            dispatch(g, enc, timer.as_deref_mut(), &key("box_v"), &self.params(io, r, kv, 0), &[(1, input), (2, o)], [groups(w, 256), groups(h, kv), 1])?;
                        }
                        input = o;
                        to_level = !to_level;
                    }
                    out.push(input);
                }
            }
        }
        dispatch(
            g,
            enc,
            timer,
            &key("mix"),
            &self.params(io, 0, 0, 0),
            &[(1, io.src), (2, io.dst), (3, out[0]), (4, out[1]), (5, out[2]), (6, out[3]), (7, &self.fxw)],
            [groups(io.cw(), 16), groups(io.ch(), 16), 1],
        )
    }
}

// ----- Spin / Radial / Path --------------------------------------------------------------------

const SPIN_TAB: usize = 0;
const ZOOM_TAB: usize = 96;
const PATH_TAB: usize = 396;
const MIP_TAB: usize = 412;
const TAB_LEN: usize = MIP_TAB + 7;

/// The smallest `x` with `x.log2().floor() >= j` (the CPU's mip choice).
fn log2_threshold(j: u32) -> f32 {
    let mut x = (1u32 << j) as f32;
    loop {
        let below = f32::from_bits(x.to_bits() - 1);
        if (below.log2().floor() as u32) < j {
            return x;
        }
        x = below;
    }
}

pub struct GatherK {
    path: bool,
    words: Vec<u32>,
    tab: wgpu::Buffer,
    fxw: wgpu::Buffer,
    /// Path: tap count and length at this level (the mip offset depends on
    /// the levels available, known at encode time).
    path_len: f32,
}

impl GatherK {
    fn base_table() -> Vec<f32> {
        let mut t = vec![0f32; TAB_LEN];
        for j in 1..=6 {
            t[MIP_TAB + j as usize] = log2_threshold(j);
        }
        t
    }

    #[allow(clippy::too_many_arguments)]
    fn radial(g: &Gpu, amount: f32, zoom: bool, cx: f32, cy: f32, level: u32, spin: Option<[f32; 4]>, fx: &BlurFx) -> GatherK {
        let s = (1u32 << level) as f32;
        let span = (amount / 100.0 * 30.0).to_radians();
        let mut t = Self::base_table();
        // `render_radial`'s per-n constants, with its own expressions.
        let max_taps = distort::RADIAL_MAX_TAPS as usize;
        for n in 1..=max_taps {
            let step = span / n as f32;
            let (s0, c0) = (-span * 0.5 + step * 0.5).sin_cos();
            let (ss, cs) = step.sin_cos();
            t[SPIN_TAB + (n - 1) * 4..SPIN_TAB + n * 4].copy_from_slice(&[c0, s0, cs, ss]);
            let base = ZOOM_TAB + n * (n - 1) / 2;
            for i in 0..n {
                t[base + i] = 1.0 - ((i as f32 + 0.5) / n as f32) * (amount / 100.0 * 0.5);
            }
        }
        let (fxon, fxw_words, fxw) = fx_words(g, fx);
        let words = Self::words(zoom as u32, spin.is_some(), fxon, s, [cx / s, cy / s, amount, span], 0, 1.0, [spin.unwrap_or([0.0; 4]), [0.0; 4], [0.0; 4], [0.0; 4]], &fxw_words);
        GatherK { path: false, words, tab: table(g, bytemuck::cast_slice(&t)), fxw, path_len: 0.0 }
    }

    fn path(g: &Gpu, p0: (f32, f32), p1: (f32, f32), bend: f32, length: f32, level: u32, fx: &BlurFx) -> GatherK {
        let s = (1u32 << level) as f32;
        // `render_path_blur`'s geometry.
        let (mx, my) = ((p0.0 + p1.0) * 0.5, (p0.1 + p1.1) * 0.5);
        let (cx, cy) = (p1.0 - p0.0, p1.1 - p0.1);
        let cl = (cx * cx + cy * cy).sqrt().max(1.0);
        let ctrl = (mx - cy / cl * bend, my + cx / cl * bend);
        let (n, _, len_l) = distort::path_taps(length, level, 0);
        let mut t = Self::base_table();
        for i in 0..n {
            t[PATH_TAB + i] = ((i as f32 + 0.5) / n as f32 - 0.5) * len_l;
        }
        let (fxon, fxw_words, fxw) = fx_words(g, fx);
        let geo = [[0.0; 4], [p0.0, p0.1, cx, cy], [ctrl.0 - p0.0, ctrl.1 - p0.1, p1.0 - ctrl.0, p1.1 - ctrl.1], [cl * cl, 0.0, 0.0, 0.0]];
        let words = Self::words(2, false, fxon, s, [0.0; 4], n as u32, 1.0, geo, &fxw_words);
        GatherK { path: true, words, tab: table(g, bytemuck::cast_slice(&t)), fxw, path_len: length }
    }

    /// The uniform without the per-band fields (output rect, mips, dims).
    #[allow(clippy::too_many_arguments)]
    fn words(kind: u32, spin: bool, fxon: bool, scale: f32, radial: [f32; 4], n: u32, kinv: f32, geo: [[f32; 4]; 4], fx: &[u32; 8]) -> Vec<u32> {
        let mut v = vec![0, 0, 0, 0, kind, 0, spin as u32, fxon as u32, scale.to_bits()];
        v.extend(radial.iter().map(|f| f.to_bits()));
        v.extend_from_slice(&[n, kinv.to_bits(), 0]);
        for q in geo {
            v.extend(q.iter().map(|f| f.to_bits()));
        }
        v.extend(std::iter::repeat_n(0, 28));
        v.extend_from_slice(fx);
        v
    }
}

impl Kernel for GatherK {
    fn input(&self) -> Input {
        Input::Source { mips: 6 }
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, _: &mut Temps, timer: Option<&mut Timer>) -> GResult<()> {
        let (w, h) = (io.cw(), io.ch());
        let extra = (io.source.len() as u32 - 1).min(6);
        let mut v = self.words.clone();
        v[..4].copy_from_slice(&[w, h, io.origin.0 as u32, io.origin.1 as u32]);
        if self.path {
            let level = io.level;
            let (_, m, _) = distort::path_taps(self.path_len, level, extra);
            v[5] = m;
            v[14] = (1.0 / (1u32 << m) as f32).to_bits();
        } else {
            v[5] = extra;
        }
        for (l, im) in io.source.iter().take(7).enumerate() {
            v[32 + l * 4..32 + l * 4 + 4].copy_from_slice(&[im.w, im.h, im.tw, im.th]);
        }
        let lv = |l: usize| -> &wgpu::Buffer { &io.source[l.min(io.source.len() - 1)].buf };
        let entry = if self.path { "path" } else { "radial" };
        let key = PipeKey { module: "bgather", entry, f16: false, consts: Vec::new() };
        dispatch(
            g,
            enc,
            timer,
            &key,
            &v,
            &[(1, lv(0)), (2, lv(1)), (3, lv(2)), (4, lv(3)), (5, lv(4)), (6, lv(5)), (7, lv(6)), (8, io.dst), (9, &self.tab), (10, &self.fxw)],
            [groups(w, 16), groups(h, 16), 1],
        )
    }
}
