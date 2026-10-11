//! Camera Raw Filter and JPEG Artifacts Removal.
//!
//! Camera Raw: the luminance plane the CPU blurs is 8-bit (`blur_f`), so it is
//! made exact (integer luma plus a CPU-built table of the float ties) and
//! blurred with the `blur` module's bit-exact passes (`gauss_plane`'s choice
//! of direct taps or boxes for the level's sigma); the per-pixel develop is
//! the CPU's f32 math (float bar: max 1/255, mean ΔE2000 <= 0.2).
//!
//! JPEG cleanup: exact integer deblocking, then the surface blur evaluated
//! directly (float bar). Photo Restoration stays on the CPU: it runs the same
//! surface blur, whose CPU result depends on its running-sum order (±1 at
//! rounding ties, like here), and then amplifies it with an unsharp mask
//! (gain up to 1.6) and a contrast LUT, so about 0.0025 % of pixels end up
//! 2-4/255 off, over the 1/255 bar.

use std::sync::{Arc, OnceLock};

use super::{Io, Kernel, Temps};
use crate::engine::filters::{Filter, blur, extra, fast};
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, Timer, dispatch, groups, table};
use crate::engine::gpu::shaders::PipeKey;

pub fn supports(f: &Filter) -> bool {
    matches!(f, Filter::CameraRaw(_) | Filter::JpegCleanup { .. })
}

pub fn plan(g: &Arc<Gpu>, f: &Filter, level: u32) -> Option<Box<dyn Kernel>> {
    match f {
        Filter::CameraRaw(r) => {
            let ties = lum_ties()?;
            let s = (1u32 << level) as f32;
            let small = r.texture != 0.0 || r.sharpen != 0.0 || r.noise_reduction != 0.0;
            let big = r.clarity != 0.0 || r.dehaze != 0.0;
            Some(Box::new(RawK {
                r: *r,
                level,
                small: small.then(|| PlaneBlur::for_sigma((2.0 / s).max(0.5))),
                big: big.then(|| PlaneBlur::for_sigma(extra::RAW_CLARITY_RADIUS / s)),
                ties: g.shared("raw_ties", || bytemuck::cast_slice(ties).to_vec()),
            }))
        }
        Filter::JpegCleanup { strength } => {
            let k = (strength / 100.0).clamp(0.0, 1.0);
            Some(Box::new(DenoiseK { deblock: (level == 0).then_some((k * 200.0) as u32), surface: Surface::new(g, 3.0 + k * 10.0) }))
        }
        _ => None,
    }
}

// ----- shared: 8-bit plane blurs through the `blur` module -------------------------------

/// `fast::gauss_plane`'s algorithm for one sigma.
#[derive(Clone, Debug, PartialEq)]
pub enum PlaneBlur {
    None,
    Direct(Vec<u32>),
    /// Box radii (vertical passes first, then horizontal, like `box_plane`).
    Boxes(Vec<u32>),
}

impl PlaneBlur {
    pub fn for_sigma(sigma: f32) -> PlaneBlur {
        if sigma < 0.3 {
            PlaneBlur::None
        } else if sigma <= fast::DIRECT_SIGMA_MAX {
            PlaneBlur::Direct(fast::gauss_kernel(sigma).iter().map(|&v| v as u32).collect())
        } else {
            PlaneBlur::Boxes(blur::boxes_for_gauss(sigma).iter().map(|&b| (b / 2) as u32).filter(|&r| r > 0).collect())
        }
    }

    /// Blurs `w x h` packed pixels (each lane an 8-bit plane) from `input`;
    /// returns the index into `out` holding the result, or None if the
    /// result is `input` itself (no blur).
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, mut timer: Option<&mut Timer>, w: u32, h: u32, input: &wgpu::Buffer, out: [&wgpu::Buffer; 2]) -> GResult<Option<usize>> {
        let key = |entry| PipeKey { module: "blur", entry, f16: false, consts: Vec::new() };
        let params = |r: u32, k: u32, mode: u32, taps: u32| -> [u32; 18] {
            let inv = if r > 0 { ((1u64 << 24) / (2 * r as u64 + 1)) as u32 } else { 0 };
            [w, h, r, inv, k, mode, 0, 0, 0, w, h, taps, w - 1, 0, 0, 0, w, h]
        };
        match self {
            PlaneBlur::None => Ok(None),
            PlaneBlur::Direct(taps) => {
                let r = (taps.len() / 2) as u32;
                if w <= 2 * r || h <= 2 * r {
                    return Ok(None);
                }
                let t = table(g, bytemuck::cast_slice(taps));
                dispatch(g, enc, timer, &key("gauss_direct"), &params(r, 0, 0, taps.len() as u32), &[(1, input), (2, out[0]), (3, &t)], [groups(w, 16), groups(h, 16), 1])?;
                Ok(Some(0))
            }
            PlaneBlur::Boxes(radii) => {
                if radii.is_empty() {
                    return Ok(None);
                }
                let mut cur: Option<usize> = None;
                for horizontal in [false, true] {
                    for &r in radii {
                        let from = cur.map_or(input, |i| out[i]);
                        let to = cur.map_or(0, |i| i ^ 1);
                        if horizontal {
                            dispatch(g, enc, timer.as_deref_mut(), &key("box_h"), &params(r, 64, 0, 0), &[(1, from), (2, out[to])], [groups(w, 64), groups(h, 64), 1])?;
                        } else {
                            dispatch(g, enc, timer.as_deref_mut(), &key("box_v"), &params(r, 16, 0, 0), &[(1, from), (2, out[to])], [groups(w, 256), groups(h, 16), 1])?;
                        }
                        cur = Some(to);
                    }
                }
                Ok(cur)
            }
        }
    }
}

// ----- Camera Raw ----------------------------------------------------------------------------

/// Per (r, g): `(b + 1) | q << 16` for the one blue value (if any) where the
/// CPU's float luma quantizes differently from (299 r + 587 g + 114 b +
/// 500) / 1000 (exact ties), else 0. None if some (r, g) had two.
pub fn lum_ties() -> Option<&'static [u32]> {
    static T: OnceLock<Option<Vec<u32>>> = OnceLock::new();
    T.get_or_init(|| {
        use rayon::prelude::*;
        let rows: Vec<Option<Vec<u32>>> = (0..256u32)
            .into_par_iter()
            .map(|r| {
                let mut row = vec![0u32; 256];
                for g in 0..256u32 {
                    for b in 0..256u32 {
                        let cpu = extra::q8(extra::lum_of(r as u8, g as u8, b as u8)) as u32;
                        let int = (r * 299 + g * 587 + b * 114 + 500) / 1000;
                        if cpu != int {
                            if row[g as usize] != 0 {
                                return None;
                            }
                            row[g as usize] = (b + 1) | cpu << 16;
                        }
                    }
                }
                Some(row)
            })
            .collect();
        let mut t = Vec::with_capacity(65536);
        for r in rows {
            t.extend(r?);
        }
        Some(t)
    })
    .as_deref()
}

struct RawK {
    r: extra::Raw,
    level: u32,
    small: Option<PlaneBlur>,
    big: Option<PlaneBlur>,
    ties: Arc<wgpu::Buffer>,
}

impl RawK {
    fn params(&self, io: &Io) -> Vec<u32> {
        let r = &self.r;
        let pre = extra::raw_pre(r);
        let flags = pre.hsl_on as u32
            | ((r.dehaze != 0.0) as u32) << 1
            | ((r.linear[2] != 0.0) as u32) << 2
            | ((r.radial[3] != 0.0) as u32) << 3
            | ((r.vignette != 0.0) as u32) << 4
            | ((r.grain != 0.0) as u32) << 5;
        let mut v: Vec<u32> = vec![io.pw, io.ph, io.pad, io.cw(), io.ch(), io.origin.0 as u32, io.origin.1 as u32, flags];
        let rad = r.img_w.min(r.img_h) * r.radial[2] / 100.0;
        let fl: Vec<f32> = vec![
            (1u32 << self.level) as f32,
            r.img_w,
            r.img_h,
            pre.ex,
            // wb
            pre.wb[0],
            pre.wb[1],
            pre.wb[2],
            0.0,
            // t0
            r.contrast,
            r.highlights,
            r.shadows,
            r.whites,
            // t1
            r.blacks,
            r.texture / 100.0 * 1.5 + r.sharpen / 150.0 * 1.2,
            r.clarity,
            r.dehaze,
            // t2
            r.noise_reduction,
            r.vibrance,
            r.saturation,
            0.5 + r.balance / 200.0,
            // curve, in region order
            r.curve[3],
            r.curve[2],
            r.curve[1],
            r.curve[0],
            r.hue[0],
            r.hue[1],
            r.hue[2],
            r.hue[3],
            r.hue[4],
            r.hue[5],
            r.hue[6],
            r.hue[7],
            r.sat[0],
            r.sat[1],
            r.sat[2],
            r.sat[3],
            r.sat[4],
            r.sat[5],
            r.sat[6],
            r.sat[7],
            r.lum[0],
            r.lum[1],
            r.lum[2],
            r.lum[3],
            r.lum[4],
            r.lum[5],
            r.lum[6],
            r.lum[7],
            pre.gs[0],
            pre.gs[1],
            pre.gs[2],
            0.0,
            pre.gh[0],
            pre.gh[1],
            pre.gh[2],
            0.0,
            // fx
            r.vignette,
            r.grain,
            r.linear[1] / 100.0 - 0.15,
            r.linear[1] / 100.0 + 0.15,
            // lin
            pre.lin_dx,
            pre.lin_dy,
            r.linear[2],
            r.radial[3],
            // rad
            r.img_w * r.radial[0] / 100.0,
            r.img_h * r.radial[1] / 100.0,
            rad.max(1.0),
            0.0,
        ];
        v.extend(fl.iter().map(|f| f.to_bits()));
        v
    }
}

impl Kernel for RawK {
    fn temps(&self) -> usize {
        // Luma, two ping-pong buffers for the large blur, the small blur.
        4
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let (w, h) = (io.pw, io.ph);
        let params = self.params(io);
        let key = |entry| PipeKey { module: "raw", entry, f16: false, consts: Vec::new() };
        let un = super::point::unpremul(g);
        let lum = tmp.at(0);
        dispatch(g, enc, timer.as_deref_mut(), &key("raw_lum"), &params, &[(1, io.src), (2, lum), (3, &self.ties), (4, &un)], [groups(w, 16), groups(h, 16), 1])?;
        let big = match &self.big {
            Some(b) => match b.encode(g, enc, timer.as_deref_mut(), w, h, lum, [tmp.at(1), tmp.at(2)])? {
                Some(i) => tmp.at(1 + i),
                None => lum,
            },
            None => lum,
        };
        let small = match &self.small {
            Some(b) => match b.encode(g, enc, timer.as_deref_mut(), w, h, lum, [tmp.at(3), tmp.at(3)])? {
                Some(_) => tmp.at(3),
                None => lum,
            },
            None => lum,
        };
        dispatch(g, enc, timer, &key("raw_main"), &params, &[(1, io.src), (2, io.dst), (4, &un), (5, small), (6, big)], [groups(io.cw(), 16), groups(io.ch(), 16), 1])
    }
}

// ----- JPEG cleanup / Photo Restoration ------------------------------------------------------

/// `rank::surface` constants: per value c, the bracketing levels and the
/// interpolation fraction, as `pbfic` computes them.
struct Surface {
    tab: wgpu::Buffer,
    inv_band: f32,
}

impl Surface {
    fn new(g: &Arc<Gpu>, threshold: f32) -> Surface {
        let (tab, inv_band) = surface_table(threshold);
        Surface { tab: table(g, bytemuck::cast_slice(&tab)), inv_band }
    }
}

/// `(table, inv_band)`: table[c * 3 ..] = (lower level, upper level, f) bits.
pub fn surface_table(threshold: f32) -> (Vec<u32>, f32) {
    let t = threshold.max(1.0);
    let band = 2.5 * t;
    let levels = ((255.0 / (band * 0.4)).ceil() as usize + 1).clamp(3, 96);
    let step = 255.0 / (levels - 1) as f32;
    let mut tab = vec![0u32; 256 * 3];
    for c in 0..256usize {
        let cf = c as f32;
        for li in 1..levels {
            let ck = li as f32 * step;
            let c_lo = ck - step;
            if cf >= c_lo && cf <= ck {
                let prev = (li - 1) as f32 * step;
                tab[c * 3] = prev.to_bits();
                tab[c * 3 + 1] = ck.to_bits();
                tab[c * 3 + 2] = ((cf - c_lo) / step).to_bits();
            }
        }
    }
    (tab, 1.0 / band)
}

struct DenoiseK {
    /// At level 0: the deblocking lerp weight.
    deblock: Option<u32>,
    surface: Surface,
}

impl Kernel for DenoiseK {
    fn temps(&self) -> usize {
        1
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let (w, h) = (io.pw, io.ph);
        let key = |entry| PipeKey { module: "denoise", entry, f16: false, consts: Vec::new() };
        let params = |out_pad: u32, k: u32, inv: f32| -> [u32; 12] { [w, h, 2, out_pad, io.origin.0 as u32, io.origin.1 as u32, k, 0, inv.to_bits(), 0, 0, 0] };
        let gr = [groups(w, 16), groups(h, 16), 1];
        let mut cur: &wgpu::Buffer = io.src;
        if let Some(k) = self.deblock {
            dispatch(g, enc, timer.as_deref_mut(), &key("deblock"), &params(0, k, 0.0), &[(1, cur), (2, tmp.at(0))], gr)?;
            cur = tmp.at(0);
        }
        dispatch(g, enc, timer, &key("surface"), &params(io.pad, 0, self.surface.inv_band), &[(1, cur), (2, io.dst), (3, &self.surface.tab)], gr)
    }
}
