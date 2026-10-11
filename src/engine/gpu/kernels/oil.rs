//! Stylize ▸ Oil Paint (Kuwahara plus the paint stage) and Pixelate ▸ Facet.
//!
//! Scratch planes (one u32 per padded pixel each): Kuwahara's horizontal
//! sums use 0..4, the vertical keys and means 4..7, the Kuwahara result
//! lands in 0; the paint stage then uses 1..4 (tensor, then stroke
//! direction), 4..8 (horizontal tensor sums, then the paint) and 8 (bristles).

use std::sync::Arc;

use super::{Io, Kernel, Temps};
use crate::engine::filters::Filter;
use crate::engine::filters::stylize::{Oil, OilGeom};
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, Timer, dispatch, groups};
use crate::engine::gpu::shaders::PipeKey;

pub struct OilK {
    r: u32,
    /// The paint stage, when active.
    geom: Option<OilGeom>,
}

pub fn supports(f: &Filter) -> bool {
    matches!(f, Filter::OilPaint { .. } | Filter::Facet)
}

/// Largest Kuwahara radius whose 16-bit horizontal sums can't overflow.
const MAX_R: usize = 200;

pub fn plan(_g: &Arc<Gpu>, f: &Filter, level: u32) -> Option<Box<dyn Kernel>> {
    let (r, oil) = match f {
        Filter::OilPaint { radius, oil } => ((radius / (1u32 << level) as f32).round().max(0.0) as usize, *oil),
        Filter::Facet => (2, Oil::default()),
        _ => return None,
    };
    if r > MAX_R {
        return None;
    }
    let geom = oil.active().then(|| oil.geom(level));
    if geom.as_ref().is_some_and(|g| g.wts_s.len() + g.wts_b.len() > 152 * 4) {
        return None;
    }
    Some(Box::new(OilK { r: r as u32, geom }))
}

impl OilK {
    fn params(&self, io: &Io, k: u32, region: [u32; 4], out: [u32; 3], full: bool) -> Vec<u32> {
        let sw = self.r + 1;
        let (bw, bh) = ((io.pw + 1).saturating_sub(sw), (io.ph + 1).saturating_sub(sw));
        let (rho, ns, nb, f) = match &self.geom {
            Some(g) => (
                g.rho as u32,
                g.ns as u32,
                g.nb as u32,
                [g.step_s, g.inv_s, g.step_b, g.inv_b, g.inv_lambda, g.s, g.light.0, g.light.1, g.gain, g.kc, g.kh, 0.5 / g.s],
            ),
            None => (0, 0, 0, [0.0; 12]),
        };
        let mut v = vec![
            io.pw,
            io.ph,
            io.pad,
            io.cw(),
            io.ch(),
            io.origin.0 as u32,
            io.origin.1 as u32,
            self.r,
            sw,
            bw,
            bh,
            k,
            0,
            rho,
            ns,
            nb,
            region[0],
            region[1],
            region[2],
            region[3],
            out[0],
            out[1],
            out[2],
            full as u32,
        ];
        v.extend(f.iter().map(|x| x.to_bits()));
        let mut w = [0f32; 152 * 4];
        if let Some(g) = &self.geom {
            w[..g.wts_s.len()].copy_from_slice(&g.wts_s);
            w[g.wts_s.len()..g.wts_s.len() + g.wts_b.len()].copy_from_slice(&g.wts_b);
        }
        v.extend(w.iter().map(|x| x.to_bits()));
        v
    }
}

impl Kernel for OilK {
    fn temps(&self) -> usize {
        if self.geom.is_some() { 9 } else { 7 }
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let key = |entry| PipeKey { module: "oil", entry, f16: false, consts: Vec::new() };
        let (pw, ph, cw, ch) = (io.pw, io.ph, io.cw(), io.ch());
        let t = |i: usize| tmp.at(i);
        let none = [0u32; 4];
        let sw = self.r + 1;
        let (bw, bh) = ((pw + 1).saturating_sub(sw), (ph + 1).saturating_sub(sw));
        if self.r > 0 {
            let pr = self.params(io, 64, none, [0; 3], false);
            let hb: [(u32, &wgpu::Buffer); 5] = [(1, io.src), (5, t(0)), (6, t(1)), (7, t(2)), (8, t(3))];
            if sw <= 33 {
                dispatch(g, enc, timer.as_deref_mut(), &key("kuw_h_seg"), &pr, &hb, [groups(bw, 256), ph, 1])?;
            } else {
                dispatch(g, enc, timer.as_deref_mut(), &key("kuw_h"), &pr, &hb, [groups(bw, 64), groups(ph, 64), 1])?;
            }
            let pr = self.params(io, 16, none, [0; 3], false);
            dispatch(
                g,
                enc,
                timer.as_deref_mut(),
                &key("kuw_v"),
                &pr,
                &[(1, t(0)), (2, t(1)), (3, t(2)), (4, t(3)), (5, t(4)), (6, t(5)), (7, t(6))],
                [groups(bw, 256), groups(bh, 16), 1],
            )?;
        }
        let sel = |region: [u32; 4], out: [u32; 3], dst: &wgpu::Buffer, timer: Option<&mut Timer>, enc: &mut wgpu::CommandEncoder| {
            let pr = self.params(io, 0, region, out, true);
            // With r == 0 the key planes are never read (every pixel keeps
            // the input), but they are still bound.
            dispatch(g, enc, timer, &key("kuw_sel"), &pr, &[(1, io.src), (2, t(4)), (3, t(5)), (4, t(6)), (5, dst)], [groups(region[2], 16), groups(region[3], 16), 1])
        };
        let Some(geo) = &self.geom else {
            return sel([io.pad, io.pad, cw, ch], [cw, io.pad, io.pad], io.dst, timer, enc);
        };
        // Kuwahara over the whole band (the rim keeps the input).
        sel([0, 0, pw, ph], [pw, 0, 0], t(0), timer.as_deref_mut(), enc)?;
        let pr = self.params(io, 64, none, [0; 3], false);
        if geo.rho > 0 {
            dispatch(g, enc, timer.as_deref_mut(), &key("oil_tensor"), &pr, &[(1, io.src), (5, t(1)), (6, t(2)), (7, t(3))], [groups(pw, 16), groups(ph, 16), 1])?;
            dispatch(g, enc, timer.as_deref_mut(), &key("oil_box_h"), &pr, &[(1, t(1)), (2, t(2)), (3, t(3)), (5, t(4)), (6, t(5)), (7, t(6))], [groups(pw, 64), groups(ph, 64), 1])?;
            let pr = self.params(io, 16, none, [0; 3], false);
            dispatch(g, enc, timer.as_deref_mut(), &key("oil_box_v"), &pr, &[(1, t(4)), (2, t(5)), (3, t(6)), (5, t(1)), (6, t(2)), (7, t(3))], [groups(pw, 256), groups(ph, 16), 1])?;
        }
        let ring = [groups(cw + 2, 16), groups(ch + 2, 16), 1];
        dispatch(
            g,
            enc,
            timer.as_deref_mut(),
            &key("oil_paint"),
            &pr,
            &[(1, t(0)), (2, t(1)), (3, t(2)), (4, t(3)), (5, t(4)), (6, t(5)), (7, t(6)), (8, t(7))],
            ring,
        )?;
        if geo.nb > 0 {
            dispatch(g, enc, timer.as_deref_mut(), &key("oil_bristle"), &pr, &[(2, t(1)), (3, t(2)), (5, t(8))], ring)?;
        }
        dispatch(g, enc, timer, &key("oil_final"), &pr, &[(1, t(4)), (2, t(5)), (3, t(6)), (4, t(7)), (5, io.dst), (6, t(8))], [groups(cw, 16), groups(ch, 16), 1])
    }
}
