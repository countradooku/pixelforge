//! Filter Gallery (Artistic, Brush Strokes, Distort, Sketch, Stylize and
//! Texture effects) on the GPU.
//!
//! The CPU renders an effect per padded window (`Filter::render` of one
//! block of tiles plus its apron) on straight-alpha f32 planes, and its
//! running-sum blurs, edge clamps and border frames depend on where that
//! window starts. So the GPU replays the CPU's windows exactly: a commit band
//! is cut into the CPU's blocks (`ops::apply_filter_cpu`: k x k tiles, k from
//! the pad), every block is a window of the batch, and every pass runs on all
//! windows at once (`z` = window). A preview is one window, like
//! `Filter::render` of the requested rect.
//!
//! Each effect is a program of passes recorded once per filter (see
//! [`fx`]): the same building blocks as `filters::gallery` (load / store,
//! luminance, box and exact Gaussians, integer boxes, medians, Kuwahara,
//! rank and line filters, noise...) in the same order, with every rounding
//! that feeds a threshold, a quantization or a rank pinned to the CPU's
//! (`gallery_lib.wgsl`). Parameters derived from the effect settings are
//! computed here with the CPU's own f32 expressions.

use std::sync::Arc;

use super::{Io, Kernel, Target, Temps};
use crate::engine::filters::Filter;
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, GpuError, Timer, dispatch, groups, table};
use crate::engine::gpu::shaders::PipeKey;
use crate::engine::grid::TS;

pub mod art;
pub mod brush;
pub mod fx;
pub mod sketch;
pub mod texture;

/// A plane (f32 or packed u32 per pixel, window-batched).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Pl(pub usize);

pub enum Res {
    Plane(usize),
    Tab(Arc<wgpu::Buffer>),
    Src,
    Dst,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    /// One thread per window pixel (16 x 16 groups).
    Pix,
    /// One thread per row.
    Rows,
    /// One thread per column.
    Cols,
    /// One thread per diagonal (w + h of them).
    Diag,
    /// One thread per output (center) pixel.
    Center,
    /// Workgroups covering (ww / a) x (wh / b) per window.
    Groups(u32, u32),
    /// Clears the plane (`bufs[0]`) instead of dispatching.
    Clear,
}

pub struct Op {
    pub module: &'static str,
    pub entry: &'static str,
    pub shape: Shape,
    pub i: [i32; 8],
    pub f: [f32; 16],
    pub bufs: Vec<(u32, Res)>,
    /// Pipeline-overridable constants (window sizes of unrolled loops).
    pub consts: Vec<(&'static str, u32)>,
    /// For passes whose shader ends in `emit3`: the alpha plane, so the pass
    /// can write the band output itself when it is the effect's last.
    pub emit: Option<usize>,
}

/// Records an effect's passes.
pub struct B {
    pub g: Arc<Gpu>,
    pub ops: Vec<Op>,
    /// Level scale (level-0 pixels per window pixel).
    pub s: f32,
    n: usize,
    /// Per plane: 1 (f32 / packed u32) or 4 (vec4) words per pixel.
    words: Vec<u32>,
    free: Vec<usize>,
    /// Largest border extension any pass uses.
    pub ext: u32,
}

/// One pass being recorded.
pub struct OpB<'a> {
    b: &'a mut B,
    op: Op,
}

impl<'a> OpB<'a> {
    /// Input f32 planes a0, a1, ...
    pub fn ins(mut self, p: &[Pl]) -> Self {
        for (k, q) in p.iter().enumerate() {
            self.op.bufs.push((4 + k as u32, Res::Plane(q.0)));
        }
        self
    }
    /// Output f32 planes b0, b1, ...
    pub fn outs(mut self, p: &[Pl]) -> Self {
        for (k, q) in p.iter().enumerate() {
            self.op.bufs.push((10 + k as u32, Res::Plane(q.0)));
        }
        self
    }
    /// Input vec4 planes q0, q1.
    pub fn qins(mut self, p: &[Pl]) -> Self {
        for (k, q) in p.iter().enumerate() {
            self.op.bufs.push((22 + k as u32, Res::Plane(q.0)));
        }
        self
    }
    /// Output vec4 plane w0.
    pub fn wout(mut self, p: Pl) -> Self {
        self.op.bufs.push((24, Res::Plane(p.0)));
        self
    }
    /// Input packed planes u0, u1.
    pub fn uins(mut self, p: &[Pl]) -> Self {
        for (k, q) in p.iter().enumerate() {
            self.op.bufs.push((14 + k as u32, Res::Plane(q.0)));
        }
        self
    }
    /// Output packed planes v0, v1, v2.
    pub fn vouts(mut self, p: &[Pl]) -> Self {
        for (k, q) in p.iter().enumerate() {
            self.op.bufs.push(([16, 20, 21][k], Res::Plane(q.0)));
        }
        self
    }
    /// Tables t0, t1, t2.
    pub fn tabs(mut self, t: &[Arc<wgpu::Buffer>]) -> Self {
        for (k, q) in t.iter().enumerate() {
            self.op.bufs.push((17 + k as u32, Res::Tab(q.clone())));
        }
        self
    }
    /// Any plane at an explicit binding.
    pub fn bind(mut self, binding: u32, p: Pl) -> Self {
        self.op.bufs.push((binding, Res::Plane(p.0)));
        self
    }
    /// Table at slot k (t0, t1, t2).
    pub fn tab(mut self, k: u32, t: Arc<wgpu::Buffer>) -> Self {
        self.op.bufs.push((17 + k, Res::Tab(t)));
        self
    }
    pub fn src(mut self) -> Self {
        self.op.bufs.push((2, Res::Src));
        self
    }
    pub fn dst(mut self) -> Self {
        self.op.bufs.push((3, Res::Dst));
        self
    }
    /// Integer parameters `p.i` (0..3) and `p.j` (4..7); i[3] is the border
    /// extension (`ext`).
    pub fn i(mut self, v: &[i32]) -> Self {
        for (k, &x) in v.iter().enumerate() {
            let k = if k >= 3 { k + 1 } else { k };
            self.op.i[k] = x;
        }
        self
    }
    /// Float parameters `p.f`, `p.g`, `p.h`, `p.k`.
    pub fn f(mut self, v: &[f32]) -> Self {
        self.op.f[..v.len()].copy_from_slice(v);
        self
    }
    pub fn ext(mut self, e: u32) -> Self {
        self.op.i[3] = e as i32;
        self.b.ext = self.b.ext.max(e);
        self
    }
    /// The shader ends in `emit3`: binds the band output and the alpha plane.
    pub fn emit(mut self, alpha: Pl) -> Self {
        self.op.bufs.push((3, Res::Dst));
        self.op.bufs.push((9, Res::Plane(alpha.0)));
        self.op.emit = Some(alpha.0);
        self
    }
    pub fn cst(mut self, name: &'static str, v: u32) -> Self {
        self.op.consts.push((name, v));
        self
    }
    pub fn shape(mut self, s: Shape) -> Self {
        self.op.shape = s;
        self
    }
    pub fn run(self) {
        self.b.ops.push(self.op);
    }
}

impl B {
    pub fn new(g: &Arc<Gpu>, level: u32) -> B {
        B { g: g.clone(), ops: Vec::new(), s: (1u32 << level) as f32, n: 0, words: Vec::new(), free: Vec::new(), ext: 0 }
    }

    pub fn plane(&mut self) -> Pl {
        self.plane_of(1)
    }

    /// A vec4 plane (four f32 channels per pixel).
    pub fn plane4(&mut self) -> Pl {
        self.plane_of(4)
    }

    fn plane_of(&mut self, words: u32) -> Pl {
        if let Some(k) = self.free.iter().position(|&i| self.words[i] == words) {
            return Pl(self.free.swap_remove(k));
        }
        self.n += 1;
        self.words.push(words);
        Pl(self.n - 1)
    }

    pub fn planes<const N: usize>(&mut self) -> [Pl; N] {
        std::array::from_fn(|_| self.plane())
    }

    /// Returns planes to the pool (later passes may reuse them).
    pub fn free(&mut self, p: &[Pl]) {
        for q in p {
            debug_assert!(!self.free.contains(&q.0));
            self.free.push(q.0);
        }
    }

    /// Zeroes a plane.
    pub fn clear(&mut self, p: Pl) {
        self.ops.push(Op { module: "", entry: "", shape: Shape::Clear, i: [0; 8], f: [0.0; 16], bufs: vec![(0, Res::Plane(p.0))], consts: Vec::new(), emit: None });
    }

    pub fn op(&mut self, module: &'static str, entry: &'static str) -> OpB<'_> {
        OpB { b: self, op: Op { module, entry, shape: Shape::Pix, i: [0; 8], f: [0.0; 16], bufs: Vec::new(), consts: Vec::new(), emit: None } }
    }

    /// `ctx.px`: a level-0 length in window pixels.
    pub fn px(&self, v: f32) -> f32 {
        v / self.s
    }

    pub fn tab_f32(&self, v: &[f32]) -> Arc<wgpu::Buffer> {
        Arc::new(table(&self.g, bytemuck::cast_slice(v)))
    }

    pub fn tab_u32(&self, v: &[u32]) -> Arc<wgpu::Buffer> {
        Arc::new(table(&self.g, bytemuck::cast_slice(v)))
    }

    pub fn tab_i32(&self, v: &[i32]) -> Arc<wgpu::Buffer> {
        Arc::new(table(&self.g, bytemuck::cast_slice(v)))
    }

    /// `px::unpremultiply` table.
    pub fn unpremul(&self) -> Arc<wgpu::Buffer> {
        super::point::unpremul(&self.g)
    }

    /// `v as f32 / 255.0` for v in 0..=255.
    pub fn by255(&self) -> Arc<wgpu::Buffer> {
        self.g.shared("gallery_by255", || {
            let v: Vec<f32> = (0..256).map(|v| v as f32 / 255.0).collect();
            bytemuck::cast_slice(&v).to_vec()
        })
    }
}

pub struct GalleryK {
    ops: Vec<Op>,
    /// Words per pixel of each plane.
    planes: Vec<u32>,
    ext: u32,
    /// CPU block size in tiles (commit), or 0: one window (preview).
    block: u32,
    temps: usize,
}

/// Effects with a GPU program that meets the equality bar.
pub fn supports(f: &Filter) -> bool {
    matches!(f, Filter::Gallery { g, .. } if fx::supported(*g))
}

pub fn plan(g: &Arc<Gpu>, f: &Filter, level: u32, target: Target) -> Option<Box<dyn Kernel>> {
    let Filter::Gallery { g: e, p, fg, bg, doc } = *f else { return None };
    if !fx::supported(e) {
        return None;
    }
    let mut b = B::new(g, level);
    fx::build(&mut b, e, p, fg, bg, doc)?;
    let pad = f.pad(level) as u32;
    // `ops::apply_filter_cpu` renders blocks of k x k tiles.
    let block = if target == Target::Commit && level == 0 {
        match f.pad(0) {
            0..=32 => 1,
            33..=128 => 2,
            129..=384 => 4,
            _ => 8,
        }
    } else {
        0
    };
    // Planes are window-batched: a band's windows overlap by their aprons.
    let side = (block.max(1) * TS as u32) as f32;
    let infl = ((side + 2.0 * (pad + b.ext) as f32) / side).powi(2);
    let words: u32 = b.words.iter().sum();
    let temps = (words as f32 * infl).ceil() as usize + 1;
    Some(Box::new(GalleryK { ops: b.ops, planes: b.words, ext: b.ext, block, temps }))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Win {
    w: u32,
    h: u32,
    ox: i32,
    oy: i32,
    sx: u32,
    sy: u32,
    cx: u32,
    cy: u32,
    cw: u32,
    ch: u32,
    dx: u32,
    dy: u32,
}

impl GalleryK {
    fn windows(&self, io: &Io) -> Vec<Win> {
        let (cw, ch, pad) = (io.cw(), io.ch(), io.pad);
        if self.block == 0 {
            return vec![Win { w: io.pw, h: io.ph, ox: io.origin.0, oy: io.origin.1, sx: 0, sy: 0, cx: pad, cy: pad, cw, ch, dx: 0, dy: 0 }];
        }
        let step = self.block * TS as u32;
        let mut v = Vec::new();
        for by in (0..ch).step_by(step as usize) {
            for bx in (0..cw).step_by(step as usize) {
                let (bw, bh) = (step.min(cw - bx), step.min(ch - by));
                v.push(Win {
                    w: bw + 2 * pad,
                    h: bh + 2 * pad,
                    ox: io.origin.0 + bx as i32,
                    oy: io.origin.1 + by as i32,
                    sx: bx,
                    sy: by,
                    cx: pad,
                    cy: pad,
                    cw: bw,
                    ch: bh,
                    dx: bx,
                    dy: by,
                });
            }
        }
        v
    }
}

impl Kernel for GalleryK {
    fn band_multiple(&self) -> u32 {
        self.block.max(1)
    }

    fn temps(&self) -> usize {
        self.temps
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let wins = self.windows(io);
        let nw = wins.len() as u32;
        let ww = wins.iter().map(|w| w.w).max().unwrap_or(1) + 2 * self.ext;
        let wh = wins.iter().map(|w| w.h).max().unwrap_or(1) + 2 * self.ext;
        let (mcw, mch) = (wins.iter().map(|w| w.cw).max().unwrap_or(1), wins.iter().map(|w| w.ch).max().unwrap_or(1));
        let plane = nw as u64 * ww as u64 * wh as u64 * 4;
        if plane * 4 > g.max_buffer() {
            return Err(GpuError::Vram);
        }
        let sizes: Vec<u64> = self.planes.iter().map(|&w| plane * w as u64).collect();
        tmp.ensure_sizes(g, &sizes)?;
        let wt = table(g, bytemuck::cast_slice(&wins));
        for op in &self.ops {
            if op.shape == Shape::Clear {
                if let Some((_, Res::Plane(i))) = op.bufs.first() {
                    enc.clear_buffer(tmp.at(*i), 0, None);
                }
                continue;
            }
            let mut params: Vec<u32> = vec![ww, wh, 0, 1u32 << io.level.min(31), nw, io.pad, io.pw, io.cw()];
            params.extend(op.i.iter().map(|&v| v as u32));
            params.extend(op.f.iter().map(|v| v.to_bits()));
            let mut bufs: Vec<(u32, &wgpu::Buffer)> = vec![(1, &wt)];
            for (b, r) in &op.bufs {
                let buf: &wgpu::Buffer = match r {
                    Res::Plane(i) => tmp.at(*i),
                    Res::Tab(t) => t,
                    Res::Src => io.src,
                    Res::Dst => io.dst,
                };
                bufs.push((*b, buf));
            }
            let grid = match op.shape {
                Shape::Pix => [groups(ww, 16), groups(wh, 16), nw],
                Shape::Rows => [groups(wh, 64), 1, nw],
                Shape::Cols => [groups(ww, 64), 1, nw],
                Shape::Diag => [groups(ww + wh, 64), 1, nw],
                Shape::Center => [groups(mcw, 16), groups(mch, 16), nw],
                Shape::Groups(a, b) => [groups(ww, a), groups(wh, b), nw],
                Shape::Clear => unreachable!(),
            };
            let key = PipeKey { module: op.module, entry: op.entry, f16: false, consts: op.consts.clone() };
            dispatch(g, enc, timer.as_deref_mut(), &key, &params, &bufs, grid)?;
        }
        Ok(())
    }
}
