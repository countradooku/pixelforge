//! Layer Styles on the GPU: the twin of `engine::styles::render`.
//!
//! The compositor reads a styled layer's padded region (pad =
//! `Plan::pad`, the layer mask already applied, layer-local level origin)
//! into `io.src` and calls [`render`], which records every effect pass and
//! the final stack into `enc`, writing the `cw x ch` center to `io.dst`.
//! Every effect plane is bit-exact with the CPU; the stack's non-Normal
//! blend modes are float (within 1/255 of the CPU).
//!
//! ```ignore
//! let plan = engine::styles::plan_for(layer, level)?;      // None: unstyled
//! let io = Io { src, dst, pw, ph, pad: plan.pad as u32, level, origin, .. };
//! gpu::styles::render(g, &mut enc, &plan, &io, &mut temps, None)?;
//! ```

use std::any::Any;
use std::sync::{Arc, Mutex};

use super::device::Gpu;
use super::exec::{self, GResult, GpuError, STORAGE, Timer, buffer, dispatch, encoder, groups, submit, wait};
use super::kernels::{Io, Temps};
use super::shaders::PipeKey;
use crate::engine::styles::{self as st, Fin, Plan, ShapeOp, Src};

const W0: usize = 0;
const W1: usize = 1;
const P0: usize = 2;

fn key(entry: &'static str) -> PipeKey {
    PipeKey { module: "styles", entry, f16: false, consts: Vec::new() }
}

/// Temporaries [`render`] needs (`ph * pw * 4` bytes each).
pub fn temps(plan: &Plan) -> usize {
    match plan.planes() {
        0 => 0,
        n => 2 + n,
    }
}

/// The 64 parameter words (`pu(i)` in `styles.wgsl`).
#[derive(Clone)]
struct Words([u32; 64]);

impl Words {
    fn new(io: &Io) -> Words {
        let mut w = [0u32; 64];
        w[0] = io.pw;
        w[1] = io.ph;
        w[2] = io.pad;
        w[3] = io.cw();
        w[4] = io.ch();
        w[17] = io.origin.0 as u32;
        w[18] = io.origin.1 as u32;
        Words(w)
    }
    fn with(&self, kv: &[(usize, u32)]) -> [u32; 64] {
        let mut w = self.0;
        for &(k, v) in kv {
            w[k] = v;
        }
        w
    }
}

/// Device copies of the plan's tables, cached by the host data's identity.
fn device_table<T: Send + Sync + 'static>(g: &Arc<Gpu>, data: &Arc<Vec<T>>, words: impl FnOnce(&[T]) -> Vec<u32>) -> Arc<wgpu::Buffer> {
    type Entry = (usize, usize, Arc<dyn Any + Send + Sync>, Arc<wgpu::Buffer>);
    static CACHE: Mutex<Vec<Entry>> = Mutex::new(Vec::new());
    let (gk, dk) = (Arc::as_ptr(g) as usize, Arc::as_ptr(data) as usize);
    if let Some(e) = CACHE.lock().unwrap().iter().find(|e| e.0 == gk && e.1 == dk) {
        return e.3.clone();
    }
    let b = Arc::new(exec::table(g, bytemuck::cast_slice(&words(data))));
    let mut c = CACHE.lock().unwrap();
    if c.len() >= 32 {
        c.remove(0);
    }
    c.push((gk, dk, data.clone() as Arc<dyn Any + Send + Sync>, b.clone()));
    b
}

fn zero(g: &Arc<Gpu>) -> Arc<wgpu::Buffer> {
    g.shared("styles_zero", || vec![0u8; 4096])
}

/// Runs a shape operation into W0 or W1; returns which.
#[allow(clippy::too_many_arguments)]
fn shape(g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &Temps, op: &ShapeOp, words: &Words, timer: &mut Option<&mut Timer>) -> GResult<usize> {
    let (w, h) = (io.pw, io.ph);
    let px = [groups(w, 16), groups(h, 16), 1];
    let (mode, input, shift) = match op.src {
        Src::Alpha { invert } => (invert as u32, io.src, 0),
        Src::Lane(l) => (2, tmp.at(P0 + l as usize / 4), (l as u32 % 4) * 8),
    };
    dispatch(g, enc, timer.as_deref_mut(), &key("extract"), &words.with(&[(5, mode), (16, shift)]), &[(1, input), (2, tmp.at(W0))], px)?;
    if let Some(d) = &op.dil {
        dispatch(g, enc, timer.as_deref_mut(), &key("edt_row"), &words.with(&[(6, d.w)]), &[(1, tmp.at(W0)), (2, tmp.at(W1))], [groups(w, 256), h, 1])?;
        let sq = device_table(g, &d.sqrt, |s| s.to_vec());
        dispatch(
            g,
            enc,
            timer.as_deref_mut(),
            &key("edt_col"),
            &words.with(&[(5, op.invert_after as u32), (6, d.w), (9, d.r16), (19, d.w * d.w)]),
            &[(1, tmp.at(W1)), (2, tmp.at(W0)), (3, &sq)],
            [groups(w, 64), groups(h, 4), 1],
        )?;
    } else if op.invert_after {
        return Err(GpuError::Unsupported);
    }
    let mut cur = W0;
    for horizontal in [false, true] {
        for &r in &op.boxes {
            let inv = ((1u64 << 24) / (2 * r as u64 + 1)) as u32;
            let (entry, k, grid) = if horizontal { ("box_h", 64, [groups(w, 64), groups(h, 64), 1]) } else { ("box_v", 16, [groups(w, 256), groups(h, 16), 1]) };
            dispatch(g, enc, timer.as_deref_mut(), &key(entry), &words.with(&[(6, r), (7, inv), (8, k)]), &[(1, tmp.at(cur)), (2, tmp.at(cur ^ 1))], grid)?;
            cur ^= 1;
        }
    }
    Ok(cur)
}

#[allow(clippy::too_many_arguments)]
fn finish(g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &Temps, plane: usize, fin: &Fin, lane: u8, words: &Words, timer: &mut Option<&mut Timer>) -> GResult<()> {
    let px = [groups(io.pw, 16), groups(io.ph, 16), 1];
    let out = tmp.at(P0 + lane as usize / 4);
    let shift = (lane as u32 % 4) * 8;
    match fin {
        Fin::Map { dx, dy, fill, lut, invert, noise8 } => {
            let tab = match lut {
                Some(l) => device_table(g, l, |l| l.iter().map(|&v| v as u32).collect()),
                None => zero(g),
            };
            let w = words.with(&[(10, *dx as u32), (11, *dy as u32), (12, *fill as u32), (13, *invert as u32), (14, *noise8), (15, lut.is_some() as u32), (16, shift)]);
            dispatch(g, enc, timer.as_deref_mut(), &key("finish_map"), &w, &[(1, tmp.at(plane)), (2, out), (3, &tab)], px)
        }
        Fin::Satin { dx, dy, invert } => {
            let w = words.with(&[(10, *dx as u32), (11, *dy as u32), (13, *invert as u32), (16, shift)]);
            dispatch(g, enc, timer.as_deref_mut(), &key("finish_satin"), &w, &[(1, tmp.at(plane)), (2, out)], px)
        }
    }
}

/// Records the styled layer for one padded region: `io.src` holds the
/// layer's pixels (mask applied, `io.pad == plan.pad`), `io.origin` is the
/// layer-local level coordinate of `src[0]`; writes the center to `io.dst`.
pub fn render(g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, plan: &Plan, io: &Io, tmp: &mut Temps, timer: Option<&mut Timer>) -> GResult<()> {
    let mut timer = timer;
    if io.pad as usize != plan.pad || io.pw < 2 * io.pad || io.ph < 2 * io.pad {
        return Err(GpuError::Unsupported);
    }
    let words = Words::new(io);
    let n = io.pw as u64 * io.ph as u64 * 4;
    let planes = plan.planes();
    tmp.ensure(g, temps(plan), n)?;
    for j in &plan.jobs {
        let p = shape(g, enc, io, tmp, &j.shape, &words, &mut timer)?;
        finish(g, enc, io, tmp, p, &j.fin, j.lane, &words, &mut timer)?;
    }
    if let Some(b) = &plan.bevel {
        let p = shape(g, enc, io, tmp, &b.shape, &words, &mut timer)?;
        let lut = device_table(g, &b.lut, |l| l.iter().map(|&v| v as u32).collect());
        let mode = b.interior as u32 | (b.exterior as u32) << 1 | (b.pillow as u32) << 2;
        dispatch(g, enc, timer.as_deref_mut(), &key("bevel"), &words.with(&[(5, mode)]), &[(1, tmp.at(p)), (2, tmp.at(P0 + 2)), (3, &lut)], [groups(io.pw, 16), groups(io.ph, 16), 1])?;
    }
    for j in &plan.soften {
        let p = shape(g, enc, io, tmp, &j.shape, &words, &mut timer)?;
        finish(g, enc, io, tmp, p, &j.fin, j.lane, &words, &mut timer)?;
    }
    // The stack.
    let mut w = words.0;
    w[20] = plan.fill8;
    let mut flags = 0u32;
    let mut set = |bit: u32, on: bool| flags |= (on as u32) << bit;
    let mut put = |e: usize, fx: &st::Fx| {
        w[24 + 3 * e] = fx.mode as u32;
        w[24 + 3 * e + 1] = fx.op8;
        w[24 + 3 * e + 2] = crate::engine::px::pack(fx.color[0], fx.color[1], fx.color[2], 0);
    };
    if let Some((fx, knock)) = &plan.ds {
        put(0, fx);
        set(0, true);
        set(1, *knock);
    }
    if let Some(fx) = &plan.og {
        put(1, fx);
        set(2, true);
    }
    if let Some(fx) = &plan.is {
        put(2, fx);
        set(3, true);
    }
    if let Some(fx) = &plan.ig {
        put(3, fx);
        set(4, true);
    }
    if let Some(fx) = &plan.satin {
        put(4, fx);
        set(5, true);
    }
    if let Some(fx) = &plan.co {
        put(5, fx);
        set(6, true);
    }
    if let Some((fx, ring, inner)) = &plan.stroke {
        put(6, fx);
        set(8, true);
        set(9, *ring);
        set(10, *inner);
    }
    if let (Some(b), Some((hi, sh))) = (&plan.bevel, &plan.bevel_fx) {
        put(7, hi);
        put(8, sh);
        set(11, true);
        set(12, b.interior);
        set(13, b.exterior);
    }
    let grad = match &plan.go {
        Some(go) => {
            w[24 + 27] = go.mode as u32;
            w[24 + 28] = go.op8;
            w[53] = go.cx.to_bits();
            w[54] = go.cy.to_bits();
            w[55] = go.ux.to_bits();
            w[56] = go.uy.to_bits();
            w[57] = go.inv_len.to_bits();
            w[58] = go.radial as u32;
            w[59] = go.reverse as u32;
            set(7, true);
            device_table(g, &go.table, |t| t.to_vec())
        }
        None => zero(g),
    };
    set(14, plan.interior());
    set(15, plan.exterior());
    for k in 0..3 {
        set(16 + k as u32, k < planes);
    }
    w[21] = flags;
    // Steps of the stack loop (a uniform, so the loop isn't unrolled).
    w[63] = 13;
    let unpre = g.shared("styles_unpremul", crate::engine::px::unpremul_table);
    let lane = |k: usize| if k < planes { tmp.at(P0 + k) } else { io.src };
    dispatch(
        g,
        enc,
        timer,
        &key("compose"),
        &w,
        &[(1, io.src), (2, io.dst), (3, &grad), (4, lane(0)), (5, lane(1)), (6, lane(2)), (7, &unpre)],
        [groups(io.cw(), 16), groups(io.ch(), 16), 1],
    )
}

/// Buffers of one standalone run (tests, benchmarks).
struct Bufs {
    src: exec::Buf,
    dst: exec::Buf,
    tmp: Temps,
}

fn upload(g: &Arc<Gpu>, plan: &Plan, src: &[u32], pw: u32, ph: u32, with_tmp: bool) -> GResult<Bufs> {
    let pad = plan.pad as u32;
    let (cw, ch) = (pw - 2 * pad, ph - 2 * pad);
    let s = buffer(g, (src.len() as u64 * 4).max(256), STORAGE)?;
    g.queue.write_buffer(&s, 0, bytemuck::cast_slice(src));
    let d = buffer(g, (cw as u64 * ch as u64 * 4).max(256), STORAGE)?;
    let mut tmp = Temps::default();
    if with_tmp {
        tmp.ensure(g, temps(plan), pw as u64 * ph as u64 * 4)?;
    }
    Ok(Bufs { src: s, dst: d, tmp })
}

/// Styles a padded host buffer on the GPU (upload, passes, readback):
/// the GPU twin of `styles::render`.
/// Large regions run in row bands (the styled pixels don't depend on
/// where a region starts), one submission each.
pub fn run(g: &Arc<Gpu>, plan: &Plan, src: &[u32], pw: u32, ph: u32, origin: (i32, i32)) -> GResult<Vec<u32>> {
    let pad = plan.pad as u32;
    if ph < 2 * pad + 1 || src.len() != (pw * ph) as usize {
        return Err(GpuError::Unsupported);
    }
    let mut out = Vec::with_capacity(((pw - 2 * pad) * (ph - 2 * pad)) as usize);
    for (y0, rows) in bands(pw, ph - 2 * pad) {
        let s = &src[(y0 * pw) as usize..((y0 + rows + 2 * pad) * pw) as usize];
        out.extend(run_with_lanes(g, plan, s, pw, rows + 2 * pad, (origin.0, origin.1 + y0 as i32), false)?.0);
    }
    Ok(out)
}

/// `(first center row, rows)` of bands of about 32 MP.
fn bands(pw: u32, ch: u32) -> Vec<(u32, u32)> {
    let rows = ((32u32 << 20) / pw.max(1)).clamp(64, ch.max(1));
    (0..ch).step_by(rows as usize).map(|y| (y, rows.min(ch - y))).collect()
}

/// [`run`], also reading back the packed lane planes (`pw x ph` each).
pub fn run_with_lanes(g: &Arc<Gpu>, plan: &Plan, src: &[u32], pw: u32, ph: u32, origin: (i32, i32), lanes: bool) -> GResult<(Vec<u32>, Vec<Vec<u32>>)> {
    let pad = plan.pad as u32;
    if pw < 2 * pad + 1 || ph < 2 * pad + 1 || src.len() != (pw * ph) as usize {
        return Err(GpuError::Unsupported);
    }
    let (cw, ch) = (pw - 2 * pad, ph - 2 * pad);
    let scopes = exec::Scopes::push(g);
    let mut b = upload(g, plan, src, pw, ph, true)?;
    let bytes = cw as u64 * ch as u64 * 4;
    let pbytes = pw as u64 * ph as u64 * 4;
    let planes = if lanes { plan.planes() } else { 0 };
    let read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
    let down = buffer(g, bytes.max(256), read)?;
    let downs = (0..planes).map(|_| buffer(g, pbytes, read)).collect::<GResult<Vec<_>>>()?;
    let mut enc = encoder(g);
    let io = Io { src: &b.src, dst: &b.dst, pw, ph, pad, level: plan.level, origin, img: (cw, ch), source: &[] };
    render(g, &mut enc, plan, &io, &mut b.tmp, None)?;
    enc.copy_buffer_to_buffer(&b.dst, 0, &down, 0, bytes);
    for (k, d) in downs.iter().enumerate() {
        enc.copy_buffer_to_buffer(b.tmp.at(P0 + k), 0, d, 0, pbytes);
    }
    let idx = submit(g, enc);
    scopes.pop()?;
    wait(g, idx)?;
    let get = |buf: &exec::Buf, n: u64| -> GResult<Vec<u32>> {
        buf.map(n, wgpu::MapMode::Read)?;
        let v: Vec<u32> = bytemuck::cast_slice(&buf.slice(..n).get_mapped_range()).to_vec();
        buf.unmap_now();
        Ok(v)
    };
    let out = get(&down, bytes)?;
    let lanes = downs.iter().map(|d| get(d, pbytes)).collect::<GResult<Vec<_>>>()?;
    Ok((out, lanes))
}

/// Best-of-`reps` time of the passes alone with the region resident:
/// (wall ms, timestamp-query kernel ms).
pub fn resident_ms(g: &Arc<Gpu>, plan: &Plan, src: &[u32], pw: u32, ph: u32, reps: usize) -> GResult<(f64, Option<f64>)> {
    let pad = plan.pad as u32;
    // Every band uploaded first (one set of temporaries); the timed part
    // is the passes alone.
    let mut tmp = Temps::default();
    let mut bufs = Vec::new();
    for (y0, rows) in bands(pw, ph - 2 * pad) {
        let s = &src[(y0 * pw) as usize..((y0 + rows + 2 * pad) * pw) as usize];
        bufs.push((upload(g, plan, s, pw, rows + 2 * pad, false)?, rows + 2 * pad));
    }
    let (mut best, mut best_k) = (f64::MAX, None::<f64>);
    for _ in 0..reps + 1 {
        let t = std::time::Instant::now();
        let mut kms = Some(0.0);
        for (b, bh) in &bufs {
            let io = Io { src: &b.src, dst: &b.dst, pw, ph: *bh, pad, level: plan.level, origin: (0, 0), img: (pw - 2 * pad, *bh - 2 * pad), source: &[] };
            let mut enc = encoder(g);
            let mut timer = Timer::new(g, 64);
            render(g, &mut enc, plan, &io, &mut tmp, timer.as_mut())?;
            if let Some(t) = timer.as_mut() {
                t.resolve(g, &mut enc);
            }
            let idx = submit(g, enc);
            wait(g, idx)?;
            match timer.as_ref().and_then(|t| t.read_passes(g)) {
                Some(passes) => {
                    if std::env::var_os("PF_GPU_TRACE").is_some() {
                        eprintln!("  passes: {}", passes.iter().map(|(l, v)| format!("{l} {v:.2}")).collect::<Vec<_>>().join(", "));
                    }
                    kms = kms.map(|k: f64| k + passes.iter().map(|p| p.1).sum::<f64>());
                }
                None => kms = None,
            }
        }
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
        if let Some(ms) = kms {
            best_k = Some(best_k.map_or(ms, |b: f64| b.min(ms)));
        }
    }
    Ok((best, best_k))
}
