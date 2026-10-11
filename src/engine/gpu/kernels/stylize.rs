//! Stylize, Pixelate, Render and the Gradient fill: Find Edges, Emboss,
//! Trace Contour, Diffuse, Wind, Fragment, Custom, De-Interlace, Mezzotint,
//! Color Halftone, Tiles, Lighting Effects, Mosaic, Extrude, Crystallize,
//! Pointillize, Clouds, Difference Clouds, Fibers, Lens Flare, Gradient.
//! (Oil Paint and Facet are in [`super::oil`].)
//!
//! Some CPU results depend on the CPU's render block (see `wgsl/stylize.wgsl`);
//! those kernels emulate it: tiles for commits whose pad is at most 32 (the
//! CPU's blocks are single tiles then), the whole band for previews.
//! Halftone and Tiles commits with a larger pad stay on the CPU (blocks of
//! several tiles start where the selection's tile span starts, which a band
//! doesn't know).

use std::sync::Arc;

use super::{Io, Kernel, Target, Temps};
use crate::engine::filters::{Filter, fast};
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, Timer, dispatch, groups, table};
use crate::engine::gpu::shaders::PipeKey;
use crate::engine::px;

pub fn supports(f: &Filter) -> bool {
    use Filter::*;
    matches!(
        f,
        FindEdges
            | Emboss { .. }
            | TraceContour { .. }
            | Diffuse { .. }
            | Wind { .. }
            | Fragment
            | Custom { .. }
            | DeInterlace { .. }
            | Mezzotint
            | ColorHalftone { .. }
            | Tiles { .. }
            | Lighting(_)
            | Mosaic { .. }
            | Extrude { .. }
            | Crystallize { .. }
            | Pointillize { .. }
            | Clouds { .. }
            | Fibers { .. }
            | LensFlare { .. }
            | Gradient(_)
    )
}

#[inline]
fn sc(v: f32, level: u32) -> f32 {
    v / (1u32 << level) as f32
}

/// The CPU's commit block side (in tiles) for a pad (`ops::apply_filter_cpu`).
fn block_tiles(pad: usize) -> u32 {
    match pad {
        0..=32 => 1,
        33..=128 => 2,
        129..=384 => 4,
        _ => 8,
    }
}

const WHOLE: u32 = 1 << 30;

fn fbits(v: &[f32]) -> impl Iterator<Item = u32> + '_ {
    v.iter().map(|x| x.to_bits())
}

fn shared_zero(g: &Arc<Gpu>) -> Arc<wgpu::Buffer> {
    g.shared("zero16", || vec![0u8; 16])
}

pub fn plan(g: &Arc<Gpu>, f: &Filter, level: u32, target: Target) -> Option<Box<dyn Kernel>> {
    use Filter::*;
    if !supports(f) {
        return None;
    }
    let s = (1u32 << level) as f32;
    let pad = f.pad(level);
    let commit = target == Target::Commit;
    let simple = |entry: &'static str, i: &[i32], fl: &[f32]| {
        let mut k = Simple { entry, i: [0; 8], f: [0.0; 16], tab: None, tabf: None, unpremul: false, blk: WHOLE };
        k.i[..i.len()].copy_from_slice(i);
        k.f[..fl.len()].copy_from_slice(fl);
        k
    };
    let b = |k: Simple| Some(Box::new(k) as Box<dyn Kernel>);
    match f {
        FindEdges => b(simple("find_edges", &[], &[])),
        Emboss { angle, height, amount } => {
            let height = sc(*height, level).max(1.0);
            let a = angle.to_radians();
            let (dx, dy) = ((a.cos() * height).round() as i32, (-a.sin() * height).round() as i32);
            let k = amount / 100.0;
            let t: Vec<u32> = (-255..=255).map(|d: i32| (128.0 + d as f32 * k).clamp(0.0, 255.0) as u8 as u32).collect();
            let mut kk = simple("emboss", &[dx, dy], &[]);
            kk.tab = Some(table(g, bytemuck::cast_slice(&t)));
            b(kk)
        }
        TraceContour { level: l } => b(simple("trace_contour", &[*l as u8 as i32], &[])),
        Diffuse { radius } => b(simple("diffuse", &[], &[sc(*radius, level).max(1.0)])),
        Fragment => b(simple("fragment", &[sc(4.0, level).round().max(1.0) as i32], &[])),
        Custom { k, scale, offset } => {
            let scl = if scale.abs() < 1e-3 { 1.0 } else { *scale };
            let mut fl = k.to_vec();
            fl.extend([scl, *offset]);
            b(simple("custom3", &[], &fl))
        }
        DeInterlace { odd, interpolate } => b(simple("deinterlace", &[*odd as i32, *interpolate as i32], &[])),
        Wind { length, from_left } => {
            let decay = 255.0 / sc(*length, level).max(1.0);
            b(simple("wind", &[*from_left as i32], &[decay]))
        }
        Mezzotint => {
            let mut k = simple("mezzotint", &[], &[]);
            k.unpremul = true;
            b(k)
        }
        ColorHalftone { radius } => {
            if commit && pad > 32 {
                return None;
            }
            let mut fl = vec![*radius, radius * 2.0];
            for deg in [108f32, 162.0, 90.0] {
                let (sn, cs) = deg.to_radians().sin_cos();
                fl.extend([sn, cs]);
            }
            let rad: Vec<f32> = (0..256)
                .map(|v| {
                    let ink = 1.0 - v as f32 / 255.0;
                    radius * 1.41 * ink.sqrt()
                })
                .collect();
            let mut k = simple("halftone", &[], &fl);
            k.tabf = Some(table(g, bytemuck::cast_slice(&rad)));
            k.blk = if commit { 256 } else { WHOLE };
            b(k)
        }
        Tiles { size, offset, fill } => {
            if commit && pad > 32 {
                return None;
            }
            let mut k = simple("tiles", &[*fill as i32], &[*size, size * offset / 100.0]);
            k.blk = if commit { 256 } else { WHOLE };
            b(k)
        }
        Lighting(lt) => {
            let dir = lt.angle.to_radians();
            let lc = [lt.color[0] as f32 / 255.0, lt.color[1] as f32 / 255.0, lt.color[2] as f32 / 255.0];
            let fl = [
                lt.x,
                lt.y,
                lt.radius,
                dir.cos(),
                -dir.sin(),
                lt.cone.to_radians().cos(),
                lt.intensity / 50.0,
                lt.ambient / 100.0,
                lt.bump / 10.0,
                lc[0],
                lc[1],
                lc[2],
            ];
            let mut k = simple("lighting", &[lt.kind as i32], &fl);
            k.blk = if commit { 256 } else { WHOLE };
            let taps: Vec<u32> = fast::gauss_kernel(1.0).iter().map(|&v| v as u32).collect();
            if taps.len() != 7 {
                return None;
            }
            k.tab = Some(table(g, bytemuck::cast_slice(&taps)));
            let inv: Vec<f32> = (0..256).map(|v| v as f32 / 255.0).collect();
            k.tabf = Some(table(g, bytemuck::cast_slice(&inv)));
            Some(Box::new(Light(k)))
        }
        Mosaic { cell } => Some(Box::new(MosaicK { cell: sc(*cell, level).max(1.0), shade: None })),
        Extrude { size, depth, pyramids } => {
            let size = size.max(2.0);
            let dk = (depth / 100.0).clamp(0.0, 1.0) * 0.5;
            let shade = [
                size,
                1.0 + dk,
                1.0 - dk,
                1.0 + dk * 0.5,
                1.0 - dk * 0.5,
                0.5 - 0.5 * dk * 1.2,
                1.0 - dk * 1.4,
                1.0 + dk * 0.6,
                1.0 - dk,
                *pyramids as u32 as f32,
                0.0,
                0.0,
            ];
            Some(Box::new(MosaicK { cell: (size / s).max(1.0), shade: Some(shade) }))
        }
        Crystallize { cell } | Pointillize { cell, .. } => {
            let bg = match f {
                Pointillize { bg, .. } => Some(px::pack(bg[0], bg[1], bg[2], 255)),
                _ => None,
            };
            let blk = if commit { block_tiles(pad) * 256 } else { 0 };
            Some(Box::new(Voronoi { cell: *cell, bg, blk }))
        }
        Clouds { c0, c1, scale, difference } => {
            Some(Box::new(Render { entry: "clouds", flag: *difference as u32, c0: px::pack(c0[0], c0[1], c0[2], 255), c1: px::pack(c1[0], c1[1], c1[2], 255), f: vec![*scale], kind: 0, lut: None, blk: 0 }))
        }
        Fibers { c0, c1, variance, strength } => Some(Box::new(Render {
            entry: "fibers",
            flag: 0,
            c0: px::pack(c0[0], c0[1], c0[2], 255),
            c1: px::pack(c1[0], c1[1], c1[2], 255),
            f: vec![2.0 + variance * 0.2, 40.0 + strength * 4.0],
            kind: 0,
            lut: None,
            blk: 0,
        })),
        LensFlare { x, y, brightness, size } => {
            Some(Box::new(Render { entry: "lens_flare", flag: 0, c0: 0, c1: 0, f: vec![*x, *y, brightness / 100.0, *size], kind: 0, lut: None, blk: 0 }))
        }
        Gradient(gf) => {
            use crate::engine::gradient::GradKind;
            let (dx, dy) = (gf.p1.0 - gf.p0.0, gf.p1.1 - gf.p0.1);
            let len2 = (dx * dx + dy * dy).max(1e-6);
            let len = len2.sqrt();
            let inv_len = 1.0 / len;
            let fl = vec![gf.p0.0, gf.p0.1, dx, dy, len2, inv_len, dx * inv_len, dy * inv_len, dx / len2 * s, dy.atan2(dx)];
            let kind = match gf.kind {
                GradKind::Linear => 0,
                GradKind::Radial => 1,
                GradKind::Angle => 2,
                GradKind::Reflected => 3,
                GradKind::Diamond => 4,
            };
            let words: Vec<u32> = gf.lut().iter().flat_map(|e| [e[0] as u32 | (e[1] as u32) << 16, e[2] as u32 | (e[3] as u32) << 16]).collect();
            let op = (gf.opacity.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
            Some(Box::new(Render {
                entry: "gradient",
                flag: gf.dither as u32,
                c0: op,
                c1: 0,
                f: fl,
                kind,
                lut: Some(table(g, bytemuck::cast_slice(&words))),
                blk: if commit { 256 } else { 0 },
            }))
        }
        _ => None,
    }
}

// ----- one-pass kernels of the `stylize` module ----------------------------------------------

pub struct Simple {
    entry: &'static str,
    i: [i32; 8],
    f: [f32; 16],
    tab: Option<wgpu::Buffer>,
    tabf: Option<wgpu::Buffer>,
    unpremul: bool,
    /// CPU block width in center pixels (see the module docs).
    blk: u32,
}

impl Simple {
    fn params(&self, io: &Io, n: u32) -> Vec<u32> {
        let s = (1u32 << io.level) as f32;
        let mut v = vec![io.pw, io.ph, io.pad, io.cw(), io.ch(), io.origin.0 as u32, io.origin.1 as u32, self.blk, 0, s.to_bits(), (1.0 / s).to_bits(), n];
        v.extend(self.i.iter().map(|&x| x as u32));
        v.extend(fbits(&self.f));
        v
    }

    fn key(&self, entry: &'static str) -> PipeKey {
        PipeKey { module: "stylize", entry, f16: false, consts: Vec::new() }
    }
}

impl Kernel for Simple {
    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, _: &mut Temps, timer: Option<&mut Timer>) -> GResult<()> {
        let (cw, ch) = (io.cw(), io.ch());
        let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![(1, io.src), (2, io.dst)];
        if let Some(t) = &self.tab {
            binds.push((3, t));
        }
        if let Some(t) = &self.tabf {
            binds.push((4, t));
        }
        let un;
        if self.unpremul {
            un = super::point::unpremul(g);
            binds.push((5, &un));
        }
        match self.entry {
            "wind" => {
                let n = 256u32.max(2 * io.pad);
                dispatch(g, enc, timer, &self.key("wind"), &self.params(io, n), &binds, [groups(cw, n), groups(ch, 64), 1])
            }
            "mezzotint" => {
                let n = io.pw * io.ph;
                let gx = groups(n, 256).min(32768);
                dispatch(g, enc, timer, &self.key("mezzotint"), &self.params(io, 0), &binds, [gx, groups(n, gx * 256), 1])
            }
            e => dispatch(g, enc, timer, &self.key(e), &self.params(io, 0), &binds, [groups(cw, 16), groups(ch, 16), 1]),
        }
    }
}

// ----- Lighting Effects ----------------------------------------------------------------------

pub struct Light(Simple);

impl Kernel for Light {
    fn temps(&self) -> usize {
        3
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let k = &self.0;
        let un = super::point::unpremul(g);
        let (taps, inv) = (k.tab.as_ref().unwrap(), k.tabf.as_ref().unwrap());
        let (t0, t1, t2) = (tmp.at(0), tmp.at(1), tmp.at(2));
        let full = [groups(io.pw, 16), groups(io.ph, 16), 1];
        let mut pr = k.params(io, 0);
        dispatch(g, enc, timer.as_deref_mut(), &k.key("light_lum"), &pr, &[(1, io.src), (2, t0), (5, &un)], full)?;
        pr[12] = 0;
        dispatch(g, enc, timer.as_deref_mut(), &k.key("light_blur"), &pr, &[(2, t1), (3, taps), (6, t0)], full)?;
        pr[12] = 1;
        dispatch(g, enc, timer.as_deref_mut(), &k.key("light_blur"), &pr, &[(2, t2), (3, taps), (6, t1)], full)?;
        let pr = k.params(io, 0);
        dispatch(
            g,
            enc,
            timer,
            &k.key("lighting"),
            &pr,
            &[(1, io.src), (2, io.dst), (4, inv), (5, &un), (6, t2), (7, t0)],
            [groups(io.cw(), 16), groups(io.ch(), 16), 1],
        )
    }
}

// ----- Mosaic / Extrude ----------------------------------------------------------------------

pub struct MosaicK {
    /// Cell size in level pixels (`stylize::mosaic`'s `cell.max(1.0)`).
    cell: f32,
    /// Extrude: size (level 0), the shades and pyramids (see `cells.wgsl`).
    shade: Option<[f32; 12]>,
}

/// Cell index of every band column (or row) by the CPU's formula, and the
/// first column of every cell (plus the end).
fn mosaic_axis(origin: i32, n: u32, cell: f32) -> (Vec<i32>, Vec<i32>) {
    let c = |x: u32| ((origin as f32 + x as f32) / cell).floor() as i64;
    let c0 = c(0);
    let ids: Vec<i32> = (0..n).map(|x| (c(x) - c0) as i32).collect();
    let cells = *ids.last().unwrap() as usize + 1;
    let mut starts = vec![0i32; cells + 1];
    for x in (0..n as usize).rev() {
        starts[ids[x] as usize] = x as i32;
    }
    starts[cells] = n as i32;
    (ids, starts)
}

fn cells_params(io: &Io, nx: u32, ny: u32, c0: (i32, i32), cell: f32, mode: u32, blk: u32, bg: u32, f: &[f32]) -> Vec<u32> {
    let s = (1u32 << io.level) as f32;
    let mut v = vec![
        io.pw,
        io.ph,
        io.pad,
        io.cw(),
        io.ch(),
        io.origin.0 as u32,
        io.origin.1 as u32,
        0,
        s.to_bits(),
        nx,
        ny,
        c0.0 as u32,
        c0.1 as u32,
        cell.to_bits(),
        mode,
        blk,
        bg,
        0,
        0,
        0,
    ];
    let mut ff = [0f32; 12];
    ff[..f.len()].copy_from_slice(f);
    v.extend(fbits(&ff));
    v
}

impl Kernel for MosaicK {
    fn temps(&self) -> usize {
        2
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let (xc, xs) = mosaic_axis(io.origin.0, io.pw, self.cell);
        let (yc, ys) = mosaic_axis(io.origin.1, io.ph, self.cell);
        let (nx, ny) = ((xs.len() - 1) as u32, (ys.len() - 1) as u32);
        let mut t = xc;
        t.extend(yc);
        t.extend(xs);
        t.extend(ys);
        let tab = table(g, bytemuck::cast_slice(&t));
        tmp.ensure_sizes(g, &[io.ph as u64 * nx as u64 * 16, nx as u64 * ny as u64 * 4])?;
        let key = |entry| PipeKey { module: "cells", entry, f16: false, consts: Vec::new() };
        let (mode, f): (u32, &[f32]) = match &self.shade {
            Some(sh) => (1, sh),
            None => (0, &[]),
        };
        let pr = cells_params(io, nx, ny, (0, 0), self.cell, mode, 0, 0, f);
        let (rows, means) = (tmp.at(0), tmp.at(1));
        dispatch(g, enc, timer.as_deref_mut(), &key("mosaic_rows"), &pr, &[(1, io.src), (2, rows), (3, &tab)], [groups(nx, 64), groups(io.ph, 4), 1])?;
        dispatch(g, enc, timer.as_deref_mut(), &key("mosaic_cells"), &pr, &[(2, means), (3, &tab), (4, rows)], [groups(nx, 64), groups(ny, 4), 1])?;
        dispatch(g, enc, timer, &key("mosaic_out"), &pr, &[(2, io.dst), (3, &tab), (4, means)], [groups(io.cw(), 16), groups(io.ch(), 16), 1])
    }
}

// ----- Crystallize / Pointillize -------------------------------------------------------------

pub struct Voronoi {
    /// Cell size, level-0 pixels.
    cell: f32,
    bg: Option<u32>,
    /// CPU block side for commits (0: previews, the whole band).
    blk: u32,
}

impl Kernel for Voronoi {
    fn temps(&self) -> usize {
        3
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let s = (1u32 << io.level) as f32;
        let cell = self.cell;
        // Grid cell of every band column / row, as the CPU's `voronoi`.
        let axis = |o: i32, n: u32| -> Vec<i32> { (0..n).map(|x| (((o as f32 + x as f32 + 0.5) * s) / cell).floor() as i32).collect() };
        let ix = axis(io.origin.0, io.pw);
        let iy = axis(io.origin.1, io.ph);
        let c0 = (ix[0] - 1, iy[0] - 1);
        let nx = (ix[io.pw as usize - 1] + 1 - c0.0 + 1) as u32;
        let ny = (iy[io.ph as usize - 1] + 1 - c0.1 + 1) as u32;
        let mut t = ix;
        t.extend(iy);
        let tab = table(g, bytemuck::cast_slice(&t));
        let n = io.pw as u64 * io.ph as u64 * 4;
        let cells = nx as u64 * ny as u64;
        tmp.ensure_sizes(g, &[n, n, cells * 36])?;
        let (ids, dist, acc) = (tmp.at(0), tmp.at(1), tmp.at(2));
        let mode = self.bg.is_some() as u32;
        let pr = cells_params(io, nx, ny, c0, cell, mode, self.blk, self.bg.unwrap_or(0), &[cell * 0.42]);
        let key = |entry| PipeKey { module: "cells", entry, f16: false, consts: Vec::new() };
        let nacc = (cells * 9) as u32;
        let gx = groups(nacc, 256).min(32768);
        dispatch(g, enc, timer.as_deref_mut(), &key("vor_clear"), &pr, &[(5, acc)], [gx, groups(nacc, gx * 256), 1])?;
        let binds: [(u32, &wgpu::Buffer); 5] = [(1, io.src), (3, &tab), (5, acc), (6, ids), (7, dist)];
        dispatch(g, enc, timer.as_deref_mut(), &key("vor_assign"), &pr, &binds, [groups(io.pw, 16), groups(io.ph, 16), 1])?;
        let binds: [(u32, &wgpu::Buffer); 5] = [(1, io.src), (2, io.dst), (4, acc), (6, ids), (7, dist)];
        dispatch(g, enc, timer, &key("vor_out"), &pr, &binds, [groups(io.cw(), 16), groups(io.ch(), 16), 1])
    }
}

// ----- Render ----------------------------------------------------------------------------------

pub struct Render {
    entry: &'static str,
    flag: u32,
    c0: u32,
    c1: u32,
    f: Vec<f32>,
    kind: u32,
    lut: Option<wgpu::Buffer>,
    /// Gradient: CPU block width for commits (the tile), 0 for previews.
    blk: u32,
}

impl Kernel for Render {
    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, _: &mut Temps, timer: Option<&mut Timer>) -> GResult<()> {
        debug_assert_eq!(io.pad, 0);
        let n = io.pw * io.ph;
        let s = (1u32 << io.level) as f32;
        let mut pr = vec![n, io.pw, io.origin.0 as u32, io.origin.1 as u32, 0, s.to_bits(), self.kind, self.flag, self.blk, self.c0, self.c1, 0];
        let mut ff = [0f32; 16];
        ff[..self.f.len()].copy_from_slice(&self.f);
        pr.extend(fbits(&ff));
        let key = PipeKey { module: "render", entry: self.entry, f16: false, consts: Vec::new() };
        let zero = shared_zero(g);
        let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![(2, io.dst)];
        if self.entry != "fibers" {
            binds.push((1, io.src));
        }
        if self.entry == "gradient" {
            binds.push((3, self.lut.as_ref().unwrap_or(&zero)));
        }
        let gx = groups(n, 256).min(32768);
        dispatch(g, enc, timer, &key, &pr, &binds, [gx, groups(n, gx * 256), 1])
    }
}
