//! Gaussian, Box, Blur / Blur More and the blur-combine family (Unsharp
//! Mask, Sharpen, Sharpen More, Sharpen Edges, Smart Sharpen, High Pass),
//! mirroring the CPU's choice of algorithm for every radius and target:
//! - display previews and block commits run `fast::blur_combine` (direct taps
//!   for sigma <= 2.6, else box passes vertical then horizontal);
//! - commits with a box pad over 160 px run the strip path (`blur_grid`:
//!   horizontal then vertical) and the per-pixel `Filter::combine`.

use std::sync::Arc;

use super::{Io, Kernel, Target, Temps};
use crate::engine::filters::{Filter, blur, fast};
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, Timer, dispatch, groups, table};
use crate::engine::gpu::shaders::PipeKey;

pub(crate) enum Pass {
    None,
    /// Direct taps; `once`: round only at the end (`conv_sep`).
    Direct { taps: Vec<u32>, once: bool },
    /// Box radii; `hv`: horizontal passes first.
    Boxes { radii: Vec<u32>, hv: bool },
}

#[derive(Clone, Copy)]
enum Mode {
    Blur = 0,
    Unsharp = 1,
    HighPass = 2,
    UnsharpStrip = 3,
}

pub struct BlurK {
    pass: Pass,
    mode: Mode,
    amount: i32,
    threshold: i32,
    tab: Option<wgpu::Buffer>,
}

pub fn supports(f: &Filter) -> bool {
    matches!(
        f,
        Filter::GaussianBlur { .. }
            | Filter::BoxBlur { .. }
            | Filter::Blur
            | Filter::BlurMore
            | Filter::HighPass { .. }
            | Filter::UnsharpMask { .. }
            | Filter::Sharpen
            | Filter::SharpenMore
            | Filter::SharpenEdges
            | Filter::SmartSharpen { .. }
    )
}

pub(crate) fn gauss_pass(sigma: f32) -> Pass {
    if sigma < 0.3 {
        Pass::None
    } else if sigma <= fast::DIRECT_SIGMA_MAX {
        Pass::Direct { taps: fast::gauss_kernel(sigma).iter().map(|&v| v as u32).collect(), once: false }
    } else {
        Pass::Boxes { radii: blur::boxes_for_gauss(sigma).iter().map(|&b| (b / 2) as u32).collect(), hv: false }
    }
}

pub fn plan(g: &Arc<Gpu>, f: &Filter, level: u32, target: Target) -> Option<Box<dyn Kernel>> {
    if !supports(f) {
        return None;
    }
    let sc = |v: f32| v / (1u32 << level) as f32;
    let k = |pass, mode, amount, threshold, tab: Option<Vec<i32>>| {
        Some(Box::new(BlurK { pass, mode, amount, threshold, tab: tab.map(|t| table(g, bytemuck::cast_slice(&t))) }) as Box<dyn Kernel>)
    };
    // The commit's strip path (see `ops::apply_filter`).
    if target == Target::Commit && level == 0 {
        if let Some(sb) = f.strip_blur() {
            let pad: usize = sb.boxes.iter().map(|b| b / 2).sum();
            if pad > 160 {
                let pass = Pass::Boxes { radii: sb.boxes.iter().map(|&b| (b / 2) as u32).collect(), hv: true };
                return match *f {
                    Filter::GaussianBlur { .. } | Filter::BoxBlur { .. } => k(pass, Mode::Blur, 0, 0, None),
                    Filter::HighPass { .. } => k(pass, Mode::HighPass, 0, 0, None),
                    _ => {
                        let (amount, _, th) = f.unsharp_params()?;
                        let kk = amount / 100.0;
                        let tab: Vec<i32> = (-255..=255).map(|d: i32| (d as f32 * kk).round() as i32).collect();
                        k(pass, Mode::UnsharpStrip, 0, th as i32, Some(tab))
                    }
                };
            }
        }
    }
    match *f {
        Filter::GaussianBlur { radius } => k(gauss_pass(sc(radius)), Mode::Blur, 0, 0, None),
        Filter::BoxBlur { radius } => {
            let r = sc(radius).round().max(0.0) as u32;
            k(Pass::Boxes { radii: vec![r], hv: false }, Mode::Blur, 0, 0, None)
        }
        Filter::Blur => k(Pass::Direct { taps: vec![64, 128, 64], once: true }, Mode::Blur, 0, 0, None),
        Filter::BlurMore => k(Pass::Direct { taps: vec![16, 64, 96, 64, 16], once: true }, Mode::Blur, 0, 0, None),
        Filter::HighPass { radius } => k(gauss_pass(sc(radius)), Mode::HighPass, 0, 0, None),
        _ => {
            let (amount, radius, th) = f.unsharp_params()?;
            k(gauss_pass(sc(radius)), Mode::Unsharp, (amount / 100.0 * 256.0).round() as i32, th.round() as i32, None)
        }
    }
}

/// Encodes `pass` over the padded band (`io.src`) and returns the temporary
/// holding the blur (None: no blur, `io.src` itself). Uses temporaries 0
/// and 1. `xmax`: columns past it read column `xmax` in vertical passes.
pub(crate) fn encode_pass(g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>, pass: &Pass, xmax: u32) -> GResult<Option<usize>> {
    let (w, h) = (io.pw, io.ph);
    let key = |entry| PipeKey { module: "blur", entry, f16: false, consts: Vec::new() };
    let params = |r: u32, k: u32, mode: u32, taps: u32| -> [u32; 18] {
        let inv = if r > 0 { ((1u64 << 24) / (2 * r as u64 + 1)) as u32 } else { 0 };
        [w, h, r, inv, k, mode, 0, 0, io.pad, io.cw(), io.ch(), taps, xmax, 0, 0, 0, io.img.0, io.img.1]
    };
    let mut blurred: Option<usize> = None;
    match pass {
        Pass::None => {}
        Pass::Direct { taps, once } => {
            let r = (taps.len() / 2) as u32;
            let t = table(g, bytemuck::cast_slice(taps));
            dispatch(
                g,
                enc,
                timer.as_deref_mut(),
                &key("gauss_direct"),
                &params(r, 0, *once as u32, taps.len() as u32),
                &[(1, io.src), (2, tmp.at(0)), (3, &t)],
                [groups(w, 16), groups(h, 16), 1],
            )?;
            blurred = Some(0);
        }
        Pass::Boxes { radii, hv } => {
            let active: Vec<u32> = radii.iter().copied().filter(|&r| r > 0).collect();
            let mut order: Vec<(u32, bool)> = Vec::new();
            for horizontal in if *hv { [true, false] } else { [false, true] } {
                order.extend(active.iter().map(|&r| (r, horizontal)));
            }
            let mut which = 0;
            for (r, horizontal) in order {
                let input = blurred.map_or(io.src, |i| tmp.at(i));
                let out = tmp.at(which);
                // Outputs per thread: amortizes the 2r + 1 tap initial sum
                // while keeping enough threads in flight (measured on an
                // M5 Pro at 268 MP: 16 rows for vertical passes in
                // 256-wide groups, 64 columns for horizontal ones).
                if horizontal {
                    let k = 64;
                    dispatch(g, enc, timer.as_deref_mut(), &key("box_h"), &params(r, k, 0, 0), &[(1, input), (2, out)], [groups(w, k), groups(h, 64), 1])?;
                } else {
                    let k = 16;
                    dispatch(g, enc, timer.as_deref_mut(), &key("box_v"), &params(r, k, 0, 0), &[(1, input), (2, out)], [groups(w, 256), groups(h, k), 1])?;
                }
                blurred = Some(which);
                which ^= 1;
            }
        }
    }
    Ok(blurred)
}

impl Kernel for BlurK {
    fn raw_edge_combine(&self) -> bool {
        matches!(self.mode, Mode::HighPass | Mode::UnsharpStrip) && matches!(self.pass, Pass::Boxes { hv: true, .. })
    }

    fn temps(&self) -> usize {
        match &self.pass {
            Pass::None => 0,
            Pass::Direct { .. } => 1,
            Pass::Boxes { .. } => 2,
        }
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let (w, h) = (io.pw, io.ph);
        let key = |entry| PipeKey { module: "blur", entry, f16: false, consts: Vec::new() };
        // The strip path's vertical passes read the horizontal result
        // clamped at the image's right edge (`read_region_clamped` on the
        // intermediate grid), not its clamped extension.
        let xmax = match &self.pass {
            Pass::Boxes { hv: true, .. } => ((io.img.0 as i32 - 1 - io.origin.0).clamp(0, w as i32 - 1)) as u32,
            _ => w - 1,
        };
        let raw_edge = self.raw_edge_combine() as u32;
        let (ox, oy) = (io.origin.0 + io.pad as i32, io.origin.1 + io.pad as i32);
        let params = |r: u32, k: u32, mode: u32, taps: u32| -> [u32; 18] {
            let inv = if r > 0 { ((1u64 << 24) / (2 * r as u64 + 1)) as u32 } else { 0 };
            [w, h, r, inv, k, mode, self.amount as u32, self.threshold as u32, io.pad, io.cw(), io.ch(), taps, xmax, raw_edge, ox as u32, oy as u32, io.img.0, io.img.1]
        };
        let blurred: &wgpu::Buffer = match encode_pass(g, enc, io, tmp, timer.as_deref_mut(), &self.pass, xmax)? {
            Some(i) => tmp.at(i),
            None => io.src,
        };
        let dummy;
        let tab = match &self.tab {
            Some(t) => t,
            None => {
                dummy = g.shared("zero16", || vec![0u8; 16]);
                &dummy
            }
        };
        dispatch(
            g,
            enc,
            timer,
            &key("combine"),
            &params(0, 0, self.mode as u32, 0),
            &[(1, io.src), (2, io.dst), (3, blurred), (4, tab)],
            [groups(io.cw(), 16), groups(io.ch(), 16), 1],
        )
    }
}
