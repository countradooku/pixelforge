//! Distort ▸ Twirl, Pinch, Spherize, ZigZag, Polar Coordinates, Ripple, Wave,
//! Shear, Offset, and the projective / wide-angle transforms: a gather from
//! the whole source level with the CPU sampler's bilinear integer math.

use std::sync::Arc;

use super::{Input, Io, Kernel, Temps};
use crate::engine::filters::{Filter, Warp};
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, Timer, dispatch, groups};
use crate::engine::gpu::shaders::PipeKey;

pub struct WarpK {
    kind: u32,
    wrap: bool,
    clip: bool,
    v: [f32; 12],
    /// Mesh warp: the 16 control points, the box (x0, y0, w, h), then
    /// Newton's 5 x 5 start nodes.
    mesh: Option<wgpu::Buffer>,
}

pub fn supports(f: &Filter) -> bool {
    matches!(f, Filter::Warp { .. })
}

pub fn plan(g: &Arc<Gpu>, f: &Filter) -> Option<Box<dyn Kernel>> {
    let Filter::Warp { warp, .. } = f else { return None };
    let mut v = [0f32; 12];
    let mut mesh = None;
    let kind = match *warp {
        Warp::Twirl { angle, cx, cy, radius } => {
            v[..4].copy_from_slice(&[angle, cx, cy, radius]);
            0
        }
        Warp::Pinch { amount, cx, cy, radius } => {
            v[..4].copy_from_slice(&[amount, cx, cy, radius]);
            1
        }
        Warp::Spherize { amount, cx, cy, radius } => {
            v[..4].copy_from_slice(&[amount, cx, cy, radius]);
            2
        }
        Warp::ZigZag { amount, ridges, cx, cy, radius } => {
            v[..5].copy_from_slice(&[amount, ridges, cx, cy, radius]);
            3
        }
        Warp::Polar { to_polar, w, h } => {
            v[..3].copy_from_slice(&[to_polar as u32 as f32, w, h]);
            4
        }
        Warp::Ripple { amount, size } => {
            v[..2].copy_from_slice(&[amount, size]);
            5
        }
        Warp::Wave { wavelength, amplitude } => {
            v[..2].copy_from_slice(&[wavelength, amplitude]);
            6
        }
        Warp::Offset { dx, dy, w, h } => {
            v[..4].copy_from_slice(&[dx, dy, w, h]);
            7
        }
        Warp::Shear { amount, h } => {
            v[..2].copy_from_slice(&[amount, h]);
            8
        }
        Warp::Homography { m } => {
            v[..9].copy_from_slice(&m);
            9
        }
        Warp::WideAngle { strength, scale, cx, cy, diag } => {
            v[..5].copy_from_slice(&[strength, scale, cx, cy, diag]);
            10
        }
        Warp::Mesh { ref m, ref nodes, x0, y0, w, h } => {
            let mut t = m.to_vec();
            t.extend([x0, y0, w, h]);
            t.extend(nodes);
            mesh = Some(crate::engine::gpu::exec::table(g, bytemuck::cast_slice(&t)));
            11
        }
    };
    Some(Box::new(WarpK { kind, wrap: warp.wraps(), clip: warp.transparent_outside(), v, mesh }))
}

impl Kernel for WarpK {
    fn input(&self) -> Input {
        Input::Source { mips: 0 }
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, _: &mut Temps, timer: Option<&mut Timer>) -> GResult<()> {
        let s = &io.source[0];
        let (w, h) = (io.cw(), io.ch());
        let mut params: Vec<u32> = vec![io.origin.0 as u32, io.origin.1 as u32, w, h, s.tw, s.w, s.h, io.level, self.kind, self.wrap as u32, self.clip as u32, 0];
        params.extend(self.v.iter().map(|f| f.to_bits()));
        if let Some(m) = &self.mesh {
            let key = PipeKey { module: "warp", entry: "warp_mesh", f16: false, consts: Vec::new() };
            return dispatch(g, enc, timer, &key, &params, &[(1, &s.buf), (2, io.dst), (3, m)], [groups(w, 16), groups(h, 16), 1]);
        }
        let key = PipeKey { module: "warp", entry: "warp", f16: false, consts: Vec::new() };
        dispatch(g, enc, timer, &key, &params, &[(1, &s.buf), (2, io.dst)], [groups(w, 16), groups(h, 16), 1])
    }
}
