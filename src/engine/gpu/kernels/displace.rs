//! Distort ▸ Displace (`extra::displace`, every map source) and Filter ▸
//! Lens Correction (`extra::render_lens_correction`).
//!
//! - Displace is a Local filter: the padded band, the map per centre pixel
//!   (the layer's red / green planes through `fast::gauss_plane`'s exact
//!   integer Gaussian, the CPU's value-noise Clouds, or the map file looked
//!   up with the CPU's exact index arithmetic), then the CPU's bilinear with
//!   8-bit fractions inside the band.
//! - Lens Correction gathers from the whole source level with the CPU
//!   sampler, one sample per channel.
//!
//! Map values / 255 come from a table, divisions are precise; float
//! positions may differ in the last bits (FMA), so these meet the resampled
//! bar (`check_resampled`).

use std::sync::{Arc, Mutex};

use super::{Input, Io, Kernel, Target, Temps};
use crate::engine::filters::Filter;
use crate::engine::filters::extra::{DispMap, MapSource};
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, Timer, dispatch, groups, table};
use crate::engine::gpu::shaders::PipeKey;

pub fn supports(f: &Filter) -> bool {
    matches!(f, Filter::Displace { .. } | Filter::LensCorrection { .. })
}

pub struct DisplaceK {
    hs: f32,
    vs: f32,
    kind: u32,
    /// File map: (buffer, w, h, tile, img_w, img_h).
    map: Option<(Arc<wgpu::Buffer>, u32, u32, bool, f32, f32)>,
    taps: wgpu::Buffer,
    r: u32,
    /// Commit: the CPU's block size in tiles (it computes positions in
    /// block-local coordinates, whose float rounding the kernel mirrors).
    block: Option<i32>,
    /// First tile row of the region (the first band's), for the blocks.
    ty0: std::sync::atomic::AtomicI32,
}

pub struct LensK {
    params: [f32; 7],
}

/// The last uploaded map file (keyed by device and map identity).
static MAP: Mutex<Option<(usize, usize, Arc<DispMap>, Arc<wgpu::Buffer>)>> = Mutex::new(None);

fn map_buffer(g: &Arc<Gpu>, m: &Arc<DispMap>) -> Arc<wgpu::Buffer> {
    let key = (Arc::as_ptr(g) as usize, Arc::as_ptr(m) as usize);
    let mut c = MAP.lock().unwrap();
    if let Some((a, b, _, buf)) = c.as_ref()
        && (*a, *b) == key
    {
        return buf.clone();
    }
    // Two bytes per texel (red, green), two texels per word.
    let buf = Arc::new(table(g, bytemuck::cast_slice(&m.rg)));
    *c = Some((key.0, key.1, m.clone(), buf.clone()));
    buf
}

pub fn plan(g: &Arc<Gpu>, f: &Filter, target: Target) -> Option<Box<dyn Kernel>> {
    match f {
        Filter::Displace { hs, vs, map } => {
            let k = crate::engine::filters::fast::gauss_kernel(2.0);
            let taps: Vec<u32> = k.iter().map(|&v| v as u32).collect();
            let (kind, map) = match map {
                MapSource::Layer => (0, None),
                MapSource::Clouds => (1, None),
                MapSource::File { map, tile, img_w, img_h } => (2, Some((map_buffer(g, map), map.w, map.h, *tile, *img_w, *img_h))),
            };
            Some(Box::new(DisplaceK { hs: *hs, vs: *vs, kind, map, taps: table(g, bytemuck::cast_slice(&taps)), r: (taps.len() / 2) as u32, block: (target == Target::Commit).then(|| crate::engine::ops::block_tiles(f.pad(0))), ty0: i32::MAX.into() }))
        }
        Filter::LensCorrection { distortion, ca_rc, ca_by, vignette, w, h } => {
            // The CPU's derived constants, computed as it computes them.
            let (cx, cy) = (w * 0.5, h * 0.5);
            let rn = (cx * cx + cy * cy).sqrt();
            let k = -distortion / 100.0 * 0.5;
            Some(Box::new(LensK { params: [cx, cy, rn, k, 1.0 + ca_rc / 1000.0, 1.0 + ca_by / 1000.0, vignette / 100.0] }))
        }
        _ => None,
    }
}

fn key(entry: &'static str) -> PipeKey {
    PipeKey { module: "displace", entry, f16: false, consts: Vec::new() }
}

const PARAMS: usize = 32;

impl Kernel for DisplaceK {
    fn temps(&self) -> usize {
        if self.kind == 0 { 1 } else { 0 }
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, timer: Option<&mut Timer>) -> GResult<()> {
        let (pw, ph) = (io.pw, io.ph);
        let s = (1u32 << io.level) as f32;
        let blur = pw > 2 * self.r && ph > 2 * self.r;
        let (mw, mh, tile, iw, ih) = self.map.as_ref().map_or((1, 1, false, 1.0, 1.0), |m| (m.1, m.2, m.3, m.4, m.5));
        let mut p: Vec<u32> = vec![
            pw,
            ph,
            io.pad,
            io.cw(),
            io.ch(),
            io.origin.0 as u32,
            io.origin.1 as u32,
            s.to_bits(),
            self.hs.to_bits(),
            self.vs.to_bits(),
            self.kind,
            self.r,
            blur as u32,
            mw,
            mh,
            tile as u32,
            iw.to_bits(),
            ih.to_bits(),
            io.level,
            (1.0 / s).to_bits(),
        ];
        p.resize(PARAMS, 0);
        if let Some(k) = self.block
            && io.level == 0
        {
            // Bands run in order: the first one starts at the region's first row.
            let pad = io.pad as i32;
            let ty0 = self.ty0.fetch_min((io.origin.1 + pad).div_euclid(256), std::sync::atomic::Ordering::Relaxed).min((io.origin.1 + pad).div_euclid(256));
            p[20] = 1;
            p[21] = k as u32;
            p[22] = (io.origin.0 + pad).div_euclid(256) as u32;
            p[24] = ty0 as u32;
        }
        let mut timer = timer;
        let dummy;
        let map: &wgpu::Buffer = match &self.map {
            Some(m) => &m.0,
            None => {
                dummy = table(g, &[0u8; 16]);
                &dummy
            }
        };
        if self.kind == 0 && blur {
            tmp.ensure(g, 1, pw as u64 * ph as u64 * 4)?;
            dispatch(g, enc, timer.as_deref_mut(), &key("disp_h"), &p, &[(1, io.src), (7, tmp.at(0)), (4, &self.taps)], [groups(pw, 16), groups(ph, 16), 1])?;
        }
        let t0 = if self.kind == 0 && blur { tmp.at(0) } else { io.src };
        // Shift per 8-bit map value, as the CPU computes it (exact: the
        // shader only adds it to the integer position).
        let shift = |sc: f32| (0..256).map(move |v| (v as f32 / 255.0 - 0.5) * 2.0 * sc / s);
        let dtab: Vec<f32> = shift(self.hs).chain(shift(self.vs)).collect();
        let dtab = table(g, bytemuck::cast_slice(&dtab));
        dispatch(
            g,
            enc,
            timer,
            &key("displace"),
            &p,
            &[(1, io.src), (2, io.dst), (3, t0), (4, &self.taps), (5, map), (6, &dtab)],
            [groups(io.cw(), 16), groups(io.ch(), 16), 1],
        )
    }
}

impl Kernel for LensK {
    fn input(&self) -> Input {
        Input::Source { mips: 0 }
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, _: &mut Temps, timer: Option<&mut Timer>) -> GResult<()> {
        let s = &io.source[0];
        let (w, h) = (io.cw(), io.ch());
        let mut p: Vec<u32> = vec![io.origin.0 as u32, io.origin.1 as u32, w, h, s.tw, s.w, s.h, ((1u32 << io.level) as f32).to_bits()];
        p.extend(self.params.iter().map(|v| v.to_bits()));
        p.resize(PARAMS, 0);
        dispatch(g, enc, timer, &key("lens"), &p, &[(1, &s.buf), (2, io.dst)], [groups(w, 16), groups(h, 16), 1])
    }
}
