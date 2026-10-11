//! Per-pixel adjustments: LUTs, the 33³ color LUT, Threshold, Desaturate,
//! Average and Add Noise.

use std::sync::Arc;

use super::{Io, Kernel, Temps};
use crate::engine::filters::Filter;
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, Timer, dispatch, groups, table};
use crate::engine::gpu::shaders::PipeKey;

pub struct Point {
    entry: &'static str,
    tab: Option<wgpu::Buffer>,
    a: u32,
    b: u32,
}

pub fn supports(f: &Filter) -> bool {
    f.point_luts().is_some() || matches!(f, Filter::Color { .. } | Filter::Threshold { .. } | Filter::Desaturate | Filter::Average { .. } | Filter::AddNoise { .. })
}

pub fn plan(g: &Arc<Gpu>, f: &Filter) -> Option<Box<dyn Kernel>> {
    let k = |entry, tab: Option<Vec<u8>>, a, b| Some(Box::new(Point { entry, tab: tab.map(|t| table(g, &t)), a, b }) as Box<dyn Kernel>);
    if let Some(l) = f.point_luts() {
        return k("lut3", Some(l.concat()), 0, 0);
    }
    match f {
        Filter::Color { lut, .. } => {
            let mut words = Vec::with_capacity(lut.raw().len() * 8);
            for c in lut.raw() {
                words.extend_from_slice(&(c[0] as u32 | (c[1] as u32) << 16).to_le_bytes());
                words.extend_from_slice(&(c[2] as u32).to_le_bytes());
            }
            k("lut3d", Some(words), 0, 0)
        }
        Filter::Threshold { level } => {
            let cut = crate::engine::filters::threshold_cut(*level);
            k("threshold", Some(bytemuck::cast_slice(&cut).to_vec()), 0, 0)
        }
        Filter::Desaturate => k("desaturate", None, 0, 0),
        Filter::Average { color } => k("fill", None, *color, 0),
        Filter::AddNoise { amount, mono } => k("add_noise", None, (amount / 100.0 * 255.0).to_bits(), *mono as u32),
        _ => None,
    }
}

/// The shared `unpremultiply` table.
pub fn unpremul(g: &Arc<Gpu>) -> Arc<wgpu::Buffer> {
    g.shared("unpremul", || crate::engine::px::unpremul_table())
}

impl Kernel for Point {
    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, _: &mut Temps, timer: Option<&mut Timer>) -> GResult<()> {
        debug_assert_eq!(io.pad, 0);
        let n = io.pw * io.ph;
        let un = unpremul(g);
        let dummy;
        let tab = match &self.tab {
            Some(t) => t,
            None => {
                dummy = g.shared("zero16", || vec![0u8; 16]);
                &dummy
            }
        };
        let gx = groups(n, 256).min(32768);
        let gy = groups(n, gx * 256);
        let params = [n, io.origin.0 as u32, io.origin.1 as u32, io.pw, self.a, self.b, 0, 0];
        let key = PipeKey { module: "point", entry: self.entry, f16: false, consts: Vec::new() };
        let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![(1, io.src), (2, io.dst)];
        if matches!(self.entry, "lut3" | "lut3d" | "threshold") {
            binds.push((3, tab));
        }
        if matches!(self.entry, "lut3" | "lut3d" | "threshold" | "desaturate") {
            binds.push((4, &un));
        }
        if self.entry == "fill" {
            binds.retain(|b| b.0 != 1);
        }
        dispatch(g, enc, timer, &key, &params, &binds, [gx, gy, 1])
    }
}
