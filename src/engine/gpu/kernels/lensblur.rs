//! Lens Blur (iris, depth map, specular highlights, noise), Shape Blur,
//! Smart Sharpen (Gaussian / Lens / Motion removal, tonal fades), Motion
//! Blur, Shadows/Highlights and Skin Smoothing.
//!
//! - Lens Blur and Shape Blur sum iris / shape rows from per-row prefix
//!   sums (`lb_spans.wgsl`). Lens Blur's sums are f32, so the prefix sums
//!   run over segments that mirror the CPU's padded blocks (the commit's
//!   `k x 256` px blocks, or the whole preview region) and add the same
//!   values in the same order: the pixels match the CPU's.
//! - Smart Sharpen reuses the blur family's Gaussian passes, the disk sums
//!   and a line-tap pass, then combines with the fades' luminance boxes
//!   (`lb_sharpen.wgsl`), all integer.
//! - Motion Blur: exact box passes at 0° / 90°, else the CPU's sheared-line
//!   running sums in the CPU blocks' line lattice (`lb_motion.wgsl`).
//! - Shadows/Highlights: the blur family's passes plus the CPU's 256 x 256
//!   table (`lb_sharpen.wgsl`).
//! - Skin Smoothing: the surface blur's piecewise-linear bilateral levels
//!   plus the skin-tone weight from a CPU-built 2^24 table (`lb_skin.wgsl`).

use std::sync::Arc;

use super::{Io, Kernel, Target, Temps};
use crate::engine::filters::{Filter, blur};
use crate::engine::gpu::device::Gpu;
use crate::engine::gpu::exec::{GResult, GpuError, Timer, dispatch, groups, table};
use crate::engine::gpu::shaders::PipeKey;

pub fn supports(f: &Filter) -> bool {
    matches!(
        f,
        Filter::LensBlur { .. } | Filter::ShapeBlur { .. } | Filter::SmartSharpen { .. } | Filter::ShadowsHighlights { .. } | Filter::MotionBlur { .. } | Filter::SkinSmoothing { .. }
    )
}

/// The CPU commit renders blocks of `k x k` tiles (`ops::apply_filter_cpu`).
fn block_tiles(pad: usize) -> u32 {
    match pad {
        0..=32 => 1,
        33..=128 => 2,
        129..=384 => 4,
        _ => 8,
    }
}

/// Center columns per CPU block for this target (None: the whole region).
fn seg_for(f: &Filter, level: u32, target: Target) -> Option<u32> {
    (target == Target::Commit && level == 0).then(|| block_tiles(f.pad(0)) * 256)
}

pub fn plan(g: &Arc<Gpu>, f: &Filter, level: u32, target: Target) -> Option<Box<dyn Kernel>> {
    let sc = |v: f32| v / (1u32 << level) as f32;
    let r_at = |v: f32| sc(v).round().max(0.0) as usize;
    let seg = seg_for(f, level, target);
    match f {
        Filter::LensBlur { radius, bloom, opts } => Some(Box::new(LensK::new(g, sc(*radius), *bloom, opts, seg))),
        Filter::ShapeBlur { radius, shape } => {
            let r = sc(*radius);
            if r < 0.5 {
                return Some(Box::new(SpansK::copy(g)));
            }
            Some(Box::new(SpansK::shape(g, r, *shape, seg)))
        }
        Filter::SmartSharpen { amount, radius, noise, opts } if !opts.is_plain() => {
            Some(Box::new(SharpenK::new(g, *amount, sc(*radius), *noise, opts, (r_at(opts.shadows.radius), r_at(opts.highlights.radius)))))
        }
        Filter::ShadowsHighlights { .. } => ShK::plan(g, f, level, target).map(|k| Box::new(k) as Box<dyn Kernel>),
        Filter::MotionBlur { angle, distance } => Some(Box::new(MotionK::new(*angle, sc(*distance), seg))),
        Filter::SkinSmoothing { radius, smoothness } => Some(Box::new(SkinK::new(r_at(*radius).max(1), *smoothness))),
        _ => None,
    }
}

fn key(module: &'static str, entry: &'static str) -> PipeKey {
    PipeKey { module, entry, f16: false, consts: Vec::new() }
}

/// Sizes scratch buffers, or says (without touching the device) that they
/// can't be bound, so the runner falls back to the CPU without counting a
/// device fault.
fn ensure(g: &Arc<Gpu>, tmp: &mut Temps, sizes: &[u64]) -> GResult<()> {
    if sizes.iter().any(|&s| s > g.max_buffer()) {
        return Err(GpuError::Vram);
    }
    tmp.ensure_sizes(g, sizes)
}

fn zero16(g: &Arc<Gpu>) -> Arc<wgpu::Buffer> {
    g.shared("zero16", || vec![0u8; 16])
}

// ----- prefix-sum spans (Lens Blur, Shape Blur, disk) ---------------------------------------

/// Prefix layout of one band.
struct Layout {
    seg: u32,
    nseg: u32,
    pitch: u32,
}

impl Layout {
    fn new(io: &Io, seg: Option<u32>) -> Layout {
        let cw = io.cw();
        let seg = seg.unwrap_or(cw).clamp(1, cw.max(1));
        let nseg = cw.div_ceil(seg);
        Layout { seg, nseg, pitch: nseg * (seg + 2 * io.pad + 1) }
    }

    fn plane_bytes(&self, io: &Io) -> u64 {
        self.pitch as u64 * io.ph as u64 * 4
    }
}

#[derive(Default)]
struct SpanParams {
    mode: u32,
    uniform: u32,
    depth: u32,
    noise: u32,
    amount: f32,
    countf: f32,
    count: u32,
    nspans: u32,
    off_rad: u32,
    off_off: u32,
    off_cnt: u32,
    off_spans: u32,
}

fn span_params(io: &Io, l: &Layout, s: &SpanParams) -> Vec<u32> {
    vec![
        io.pw,
        io.ph,
        io.pad,
        io.cw(),
        io.ch(),
        l.seg,
        l.pitch,
        l.nseg,
        s.mode,
        s.uniform,
        s.depth,
        s.noise,
        s.amount.to_bits(),
        (io.origin.0) as u32,
        (io.origin.1) as u32,
        s.countf.to_bits(),
        s.count,
        s.nspans,
        0,
        s.off_rad,
        s.off_off,
        s.off_cnt,
        s.off_spans,
    ]
}

/// Records the prefix pass and the span sums of one band into `out`
/// (center layout: `io.dst`, or temporary `out`). Uses temporaries
/// `t0..t0 + 4`.
#[allow(clippy::too_many_arguments)]
fn encode_spans(
    g: &Arc<Gpu>,
    enc: &mut wgpu::CommandEncoder,
    io: &Io,
    tmp: &mut Temps,
    mut timer: Option<&mut Timer>,
    seg: Option<u32>,
    s: &SpanParams,
    tab: &wgpu::Buffer,
    mask: Option<&wgpu::Buffer>,
    out: Option<usize>,
    t0: usize,
) -> GResult<()> {
    let l = Layout::new(io, seg);
    let params = span_params(io, &l, s);
    let entry = match s.mode {
        0 => "lens_sum",
        3 => "copy_center",
        _ => "shape_sum",
    };
    if s.mode == 3 {
        let dst = io.dst;
        return dispatch(g, enc, timer, &key("lb_spans", entry), &params, &[(1, io.src), (2, dst)], [groups(io.cw(), 16), groups(io.ch(), 16), 1]);
    }
    let pb = l.plane_bytes(io);
    let mut sizes = vec![0u64; t0];
    sizes.extend([pb; 4]);
    ensure(g, tmp, &sizes)?;
    let dst = out.map_or(io.dst, |i| tmp.at(i));
    let planes = [(3, tmp.at(t0)), (4, tmp.at(t0 + 1)), (5, tmp.at(t0 + 2)), (6, tmp.at(t0 + 3))];
    let prefix = if s.mode == 0 { "prefix_f" } else { "prefix_u" };
    let mut b: Vec<(u32, &wgpu::Buffer)> = vec![(1, io.src)];
    b.extend_from_slice(&planes);
    if s.mode == 0 {
        b.push((7, tab));
    }
    dispatch(g, enc, timer.as_deref_mut(), &key("lb_spans", prefix), &params, &b, [groups(io.ph, 64), l.nseg, 1])?;
    let z = zero16(g);
    let mut b: Vec<(u32, &wgpu::Buffer)> = if s.mode == 0 { vec![(1, io.src), (2, dst)] } else { vec![(2, dst)] };
    b.extend_from_slice(&planes);
    b.push((7, tab));
    if s.mode == 0 {
        b.push((8, mask.unwrap_or(&z)));
    }
    dispatch(g, enc, timer, &key("lb_spans", entry), &params, &b, [groups(io.cw(), 16), groups(io.ch(), 16), 1])
}

fn span_words(spans: &[[i32; 3]]) -> Vec<u32> {
    spans.iter().flat_map(|s| [s[0] as u32, s[1] as u32, s[2] as u32, 0]).collect()
}

/// Lens Blur.
pub struct LensK {
    tab: wgpu::Buffer,
    s: SpanParams,
    seg: Option<u32>,
    mask: Option<Arc<blur::DepthMask>>,
}

impl LensK {
    fn new(g: &Arc<Gpu>, r: f32, bloom: f32, o: &blur::LensOpts, seg: Option<u32>) -> LensK {
        let plan = blur::LensPlan::new(r, o);
        let (to_lin, from_lin) = blur::lens_tables(bloom, o.threshold);
        let mut t: Vec<u32> = to_lin.iter().map(|v| v.to_bits()).collect();
        t.extend((0..256).map(|v| (v as f32 / 255.0).to_bits()));
        t.extend(from_lin.iter().map(|&v| v as u32));
        let off_rad = t.len() as u32;
        t.extend_from_slice(&plan.rad);
        let off_off = t.len() as u32;
        t.extend_from_slice(&plan.off);
        let off_cnt = t.len() as u32;
        t.extend(plan.count.iter().map(|c| c.to_bits()));
        let off_spans = t.len() as u32;
        t.extend(span_words(&plan.spans));
        let depth = match &o.depth {
            blur::Depth::None => 0,
            blur::Depth::Transparency => 1,
            blur::Depth::Mask(_) => 2,
        };
        let noise = if o.noise > 0.0 { 1 | (o.gaussian as u32) << 1 | (o.mono as u32) << 2 } else { 0 };
        let s = SpanParams { mode: 0, uniform: plan.uniform as u32, depth, noise, amount: o.noise, off_rad, off_off, off_cnt, off_spans, ..Default::default() };
        let mask = match &o.depth {
            blur::Depth::Mask(m) => Some(m.clone()),
            _ => None,
        };
        LensK { tab: table(g, bytemuck::cast_slice(&t)), s, seg, mask }
    }
}

impl Kernel for LensK {
    fn temps(&self) -> usize {
        4
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, timer: Option<&mut Timer>) -> GResult<()> {
        let mask = self.mask.as_ref().map(|m| {
            let mut v = m.read(io.level, io.origin.0, io.origin.1, io.pw as usize, io.ph as usize);
            v.resize(v.len().next_multiple_of(4), 0);
            table(g, &v)
        });
        encode_spans(g, enc, io, tmp, timer, self.seg, &self.s, &self.tab, mask.as_ref(), None, 0)
    }
}

/// Shape Blur (and a plain copy when the radius rounds to nothing).
pub struct SpansK {
    tab: wgpu::Buffer,
    s: SpanParams,
    seg: Option<u32>,
}

impl SpansK {
    fn copy(g: &Arc<Gpu>) -> SpansK {
        SpansK { tab: table(g, &[0u8; 16]), s: SpanParams { mode: 3, ..Default::default() }, seg: None }
    }

    fn shape(g: &Arc<Gpu>, radius: f32, shape: usize, seg: Option<u32>) -> SpansK {
        let spans = crate::engine::filters::extra::shape_spans(radius, shape);
        let count: f32 = spans.iter().map(|s| (s[2] - s[1] + 1) as f32).sum();
        if count <= 1.0 {
            return SpansK::copy(g);
        }
        let t = span_words(&spans);
        SpansK { tab: table(g, bytemuck::cast_slice(&t)), s: SpanParams { mode: 1, countf: count, nspans: spans.len() as u32, ..Default::default() }, seg }
    }
}

impl Kernel for SpansK {
    fn temps(&self) -> usize {
        if self.s.mode == 3 { 0 } else { 4 }
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, timer: Option<&mut Timer>) -> GResult<()> {
        encode_spans(g, enc, io, tmp, timer, self.seg, &self.s, &self.tab, None, None, 0)
    }
}

// ----- Smart Sharpen and Shadows/Highlights ------------------------------------------------

fn sharpen_params(io: &Io, r: u32, k: i32, th: i32, fades: u32, boff: u32, bpitch: u32, ntaps: u32, raw_edge: u32) -> Vec<u32> {
    let (ox, oy) = (io.origin.0 + io.pad as i32, io.origin.1 + io.pad as i32);
    vec![io.pw, io.ph, io.pad, io.cw(), io.ch(), r, k as u32, th as u32, fades, boff, bpitch, ntaps, raw_edge, ox as u32, oy as u32, io.img.0, io.img.1, 0]
}

/// Smart Sharpen with a Lens / Motion removal or fades.
pub struct SharpenK {
    k: i32,
    th: i32,
    remove: blur::Remove,
    sigma: f32,
    r: usize,
    taps: Option<wgpu::Buffer>,
    ntaps: u32,
    disk: Option<(wgpu::Buffer, u32)>,
    att: Option<wgpu::Buffer>,
    fade_r: (usize, usize),
}

impl SharpenK {
    fn new(g: &Arc<Gpu>, amount: f32, radius: f32, noise: f32, o: &blur::SharpenOpts, fade_r: (usize, usize)) -> SharpenK {
        let r = radius.round().max(0.0) as usize;
        let taps = blur::line_taps(r, o.angle);
        let att = (o.shadows.amount > 0.0 || o.highlights.amount > 0.0).then(|| {
            let mut t = blur::fade_table(&o.shadows, true).to_vec();
            t.extend_from_slice(&blur::fade_table(&o.highlights, false));
            table(g, bytemuck::cast_slice(&t))
        });
        let disk = (o.remove == blur::Remove::Lens && r > 0).then(|| {
            let spans = blur::iris_spans(r, 0, 0.0, 0.0);
            let count: u32 = spans.iter().map(|s| (s[2] - s[1] + 1) as u32).sum();
            (table(g, bytemuck::cast_slice(&span_words(&spans))), count)
        });
        let words: Vec<i32> = taps.iter().flatten().copied().collect();
        SharpenK {
            k: (amount / 100.0 * 256.0).round() as i32,
            th: noise.round() as i32,
            remove: o.remove,
            sigma: radius,
            r,
            taps: (o.remove == blur::Remove::Motion && r > 0).then(|| table(g, bytemuck::cast_slice(&words))),
            ntaps: taps.len() as u32,
            disk,
            att,
            fade_r,
        }
    }
}

impl Kernel for SharpenK {
    fn temps(&self) -> usize {
        6
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let n = io.pw as u64 * io.ph as u64 * 4;
        let c = io.cw() as u64 * io.ch() as u64 * 4;
        let rows = io.ph as u64 * io.cw() as u64 * 4;
        // 0, 1: Gaussian passes; 2..6: disk prefix planes; 6: disk / line
        // blur; 7: luma; 8: horizontal sums; 9, 10: shadow / highlight means.
        ensure(g, tmp, &[n, n, 0, 0, 0, 0, c, n, rows, c, c])?;
        let module = "lb_sharpen";
        // The blur, and where its center pixels are.
        let (blurred, boff, bpitch): (&wgpu::Buffer, u32, u32) = match (self.remove, &self.disk, &self.taps) {
            (blur::Remove::Lens, Some((tab, count)), _) => {
                let s = SpanParams { mode: 2, count: *count, nspans: tab.size() as u32 / 16, ..Default::default() };
                encode_spans(g, enc, io, tmp, timer.as_deref_mut(), None, &s, tab, None, Some(6), 2)?;
                (tmp.at(6), 0, io.cw())
            }
            (blur::Remove::Motion, _, Some(taps)) => {
                let params = sharpen_params(io, 0, 0, 0, 0, 0, 0, self.ntaps, 0);
                dispatch(g, enc, timer.as_deref_mut(), &key(module, "line_taps"), &params, &[(1, io.src), (2, tmp.at(6)), (5, taps)], [groups(io.cw(), 16), groups(io.ch(), 16), 1])?;
                (tmp.at(6), 0, io.cw())
            }
            (blur::Remove::Gaussian, _, _) => {
                let pass = super::blur::gauss_pass(self.sigma);
                match super::blur::encode_pass(g, enc, io, tmp, timer.as_deref_mut(), &pass, io.pw - 1)? {
                    Some(i) => (tmp.at(i), io.pad, io.pw),
                    None => (io.src, io.pad, io.pw),
                }
            }
            // Radius 0: nothing to remove.
            _ => (io.src, io.pad, io.pw),
        };
        let z = zero16(g);
        let att = match &self.att {
            Some(att) => {
                let un = super::point::unpremul(g);
                let total = io.pw * io.ph;
                let gx = groups(total, 256).min(32768);
                let params = sharpen_params(io, 0, 0, 0, 0, 0, 0, 0, 0);
                dispatch(g, enc, timer.as_deref_mut(), &key(module, "luma"), &params, &[(1, io.src), (2, tmp.at(7)), (4, &un)], [gx, groups(total, gx * 256), 1])?;
                for (r, out) in [(self.fade_r.0, 9), (self.fade_r.1, 10)] {
                    let params = sharpen_params(io, r as u32, 0, 0, 0, 0, 0, 0, 0);
                    dispatch(g, enc, timer.as_deref_mut(), &key(module, "lbox_h"), &params, &[(1, tmp.at(7)), (2, tmp.at(8))], [groups(io.cw(), 64), groups(io.ph, 64), 1])?;
                    dispatch(g, enc, timer.as_deref_mut(), &key(module, "lbox_v"), &params, &[(1, tmp.at(8)), (2, tmp.at(out))], [groups(io.cw(), 256), groups(io.ch(), 16), 1])?;
                }
                Some(att)
            }
            None => None,
        };
        let params = sharpen_params(io, 0, self.k, self.th, att.is_some() as u32, boff, bpitch, 0, 0);
        let (tab, ls, lh) = match att {
            Some(a) => (a, tmp.at(9), tmp.at(10)),
            None => (&*z, &*z, &*z),
        };
        let _ = self.r;
        dispatch(
            g,
            enc,
            timer,
            &key(module, "ss_combine"),
            &params,
            &[(1, io.src), (2, io.dst), (3, blurred), (5, tab), (6, ls), (7, lh)],
            [groups(io.cw(), 16), groups(io.ch(), 16), 1],
        )
    }
}

/// Shadows/Highlights: the Gaussian as `blur::gaussian` (or the commit's
/// strip boxes), then the table combine.
pub struct ShK {
    pass: super::blur::Pass,
    strip: bool,
    tab: wgpu::Buffer,
}

impl ShK {
    fn plan(g: &Arc<Gpu>, f: &Filter, level: u32, target: Target) -> Option<ShK> {
        let Filter::ShadowsHighlights { radius, .. } = *f else { return None };
        let comb = f.combiner();
        let tab = table(g, comb.table()?);
        if target == Target::Commit && level == 0 {
            let sb = f.strip_blur()?;
            if sb.boxes.iter().map(|b| b / 2).sum::<usize>() > 160 {
                let pass = super::blur::Pass::Boxes { radii: sb.boxes.iter().map(|&b| (b / 2) as u32).collect(), hv: true };
                return Some(ShK { pass, strip: true, tab });
            }
        }
        let sigma = radius / (1u32 << level) as f32;
        let pass = if sigma < 0.3 {
            super::blur::Pass::None
        } else if sigma <= crate::engine::filters::fast::DIRECT_SIGMA_MAX {
            super::blur::Pass::Direct { taps: crate::engine::filters::fast::gauss_kernel(sigma).iter().map(|&v| v as u32).collect(), once: true }
        } else {
            super::blur::Pass::Boxes { radii: blur::boxes_for_gauss(sigma).iter().map(|&b| (b / 2) as u32).collect(), hv: true }
        };
        Some(ShK { pass, strip: false, tab })
    }
}

impl Kernel for ShK {
    fn raw_edge_combine(&self) -> bool {
        self.strip
    }

    fn temps(&self) -> usize {
        2
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let xmax = if self.strip { (io.img.0 as i32 - 1 - io.origin.0).clamp(0, io.pw as i32 - 1) as u32 } else { io.pw - 1 };
        let blurred = match super::blur::encode_pass(g, enc, io, tmp, timer.as_deref_mut(), &self.pass, xmax)? {
            Some(i) => tmp.at(i),
            None => io.src,
        };
        let un = super::point::unpremul(g);
        let params = sharpen_params(io, 0, 0, 0, 0, 0, 0, 0, self.strip as u32);
        dispatch(
            g,
            enc,
            timer,
            &key("lb_sharpen", "sh_combine"),
            &params,
            &[(1, io.src), (2, io.dst), (3, blurred), (4, &un), (5, &self.tab)],
            [groups(io.cw(), 16), groups(io.ch(), 16), 1],
        )
    }
}

// ----- Motion Blur ----------------------------------------------------------------------------

enum Motion {
    Copy,
    BoxH(u32),
    BoxV(u32),
    Lines { transposed: bool, slope: f32, half: u32 },
}

pub struct MotionK {
    m: Motion,
    seg: Option<u32>,
}

impl MotionK {
    /// `blur::motion`'s choice of algorithm (`dist` at this level).
    fn new(angle: f32, dist: f32, seg: Option<u32>) -> MotionK {
        let ang = angle.rem_euclid(180.0);
        let r = (dist * 0.5).round() as u32;
        let m = if ang.abs() < 1e-3 || (ang - 180.0).abs() < 1e-3 {
            if r > 0 { Motion::BoxH(r) } else { Motion::Copy }
        } else if (ang - 90.0).abs() < 1e-3 {
            if r > 0 { Motion::BoxV(r) } else { Motion::Copy }
        } else {
            let a = angle.to_radians();
            let (c, s) = (a.cos(), -a.sin());
            let (transposed, slope, half) = if c.abs() >= s.abs() { (false, s / c, (dist * c.abs() * 0.5).round()) } else { (true, c / s, (dist * s.abs() * 0.5).round()) };
            if half as u32 == 0 { Motion::Copy } else { Motion::Lines { transposed, slope, half: half as u32 } }
        };
        MotionK { m, seg }
    }
}

impl Kernel for MotionK {
    fn temps(&self) -> usize {
        if matches!(self.m, Motion::Lines { .. }) { 5 } else { 0 }
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let (cw, ch, pad) = (io.cw(), io.ch(), io.pad);
        let oyc = io.origin.1 + pad as i32;
        let params = |r: u32, k: u32, half: u32, l: u32, transposed: bool, slope: f32, cpp: u32, s: u32, nbc: u32, br0: i32| -> Vec<u32> {
            let inv = if r > 0 { ((1u64 << 24) / (2 * r as u64 + 1)) as u32 } else { 0 };
            let invn = 1.0 / (2 * half + 1) as f32;
            vec![io.pw, io.ph, pad, cw, ch, r, inv, k, half, l, transposed as u32, slope.to_bits(), invn.to_bits(), cpp, s, nbc, oyc as u32, br0 as u32, 0]
        };
        let module = "lb_motion";
        match self.m {
            Motion::Copy => {
                let p = span_params(io, &Layout::new(io, None), &SpanParams::default());
                dispatch(g, enc, timer, &key("lb_spans", "copy_center"), &p, &[(1, io.src), (2, io.dst)], [groups(cw, 16), groups(ch, 16), 1])
            }
            Motion::BoxH(r) => {
                let k = 64;
                dispatch(g, enc, timer, &key(module, "mbox_h"), &params(r, k, 0, 1, false, 0.0, 0, 0, 0, 0), &[(1, io.src), (2, io.dst)], [groups(cw, k), groups(ch, 64), 1])
            }
            Motion::BoxV(r) => {
                let k = 16;
                dispatch(g, enc, timer, &key(module, "mbox_v"), &params(r, k, 0, 1, false, 0.0, 0, 0, 0, 0), &[(1, io.src), (2, io.dst)], [groups(cw, 256), groups(ch, k), 1])
            }
            Motion::Lines { transposed, slope, half } => {
                // Chunk length: amortizes each line's 2 half + 1 sample warm-up.
                let l = (2 * half + 1).next_power_of_two().clamp(64, 512);
                let (s, nbc, br0, nbr) = match self.seg {
                    Some(s) => {
                        let br0 = oyc.div_euclid(s as i32);
                        (s, cw.div_ceil(s), br0, (oyc + ch as i32 - 1).div_euclid(s as i32) - br0 + 1)
                    }
                    None => (0, 1, 0, 1),
                };
                let extent = if s > 0 { s } else { cw.max(ch) };
                let cpp = extent.div_ceil(l);
                let sl = slope as f64;
                let (p_, l_) = (pad as i64, l as i64);
                let mut desc: Vec<i32> = Vec::new();
                let mut voff: i64 = 0;
                let mut max_lines = 0i64;
                for prow in 0..nbr as i64 {
                    for bc in 0..nbc as i64 {
                        let (bx, by, partw, yr) = if s > 0 {
                            let s = s as i64;
                            let by = (br0 as i64 + prow) * s - oyc as i64;
                            let yr = (p_.max(p_ - by), (p_ + s).min(p_ + ch as i64 - by));
                            (bc * s, by, s.min(cw as i64 - bc * s), yr)
                        } else {
                            (0, 0, cw as i64, (p_, p_ + ch as i64))
                        };
                        let xr = (p_, p_ + partw);
                        let (xs, ys) = if transposed { (yr, xr) } else { (xr, yr) };
                        for c in 0..cpp as i64 {
                            let x0 = xs.0 + c * l_;
                            let x1 = (x0 + l_).min(xs.1);
                            let (bstart, nlines) = if x0 < x1 {
                                let (a, b) = (x0 as f64 * sl, (x1 - 1) as f64 * sl);
                                let bstart = (ys.0 as f64 - a.max(b)).floor() as i64 - 2;
                                let bend = ((ys.1 - 1) as f64 - a.min(b)).floor() as i64 + 3;
                                (bstart, bend - bstart + 1)
                            } else {
                                (0, 0)
                            };
                            desc.extend([x0, x1, bstart, nlines, voff, bx, by, 0].map(|v| v as i32));
                            voff += nlines * l_;
                            max_lines = max_lines.max(nlines);
                        }
                    }
                }
                let nchunks = desc.len() as u32 / 8;
                ensure(g, tmp, &[(voff.max(1) as u64) * 16])?;
                let dt = table(g, bytemuck::cast_slice(&desc));
                let p = params(0, 0, half, l, transposed, slope, cpp, s, nbc, br0);
                dispatch(g, enc, timer.as_deref_mut(), &key(module, "lines"), &p, &[(1, io.src), (3, &dt), (4, tmp.at(0))], [groups(max_lines as u32, 64), nchunks, 1])?;
                dispatch(g, enc, timer, &key(module, "lines_out"), &p, &[(2, io.dst), (3, &dt), (4, tmp.at(0))], [groups(cw, 16), groups(ch, 16), 1])
            }
        }
    }
}

// ----- Skin Smoothing --------------------------------------------------------------------------

pub struct SkinK {
    r: u32,
    levels: u32,
    step: f32,
    inv_band: f32,
}

impl SkinK {
    /// `rank::surface(radius, 4 + smoothness * 0.4)`'s level setup.
    fn new(r: usize, smoothness: f32) -> SkinK {
        let t = (4.0 + smoothness * 0.4f32).max(1.0);
        let band = 2.5 * t;
        let levels = ((255.0 / (band * 0.4)).ceil() as usize + 1).clamp(3, 96);
        SkinK { r: r as u32, levels: levels as u32, step: 255.0 / (levels - 1) as f32, inv_band: 1.0 / band }
    }
}

impl Kernel for SkinK {
    fn temps(&self) -> usize {
        4
    }

    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, mut timer: Option<&mut Timer>) -> GResult<()> {
        let (cw, ch) = (io.cw(), io.ch());
        let rows = io.ph as u64 * cw as u64 * 16;
        let center = cw as u64 * ch as u64 * 16;
        ensure(g, tmp, &[rows, rows, center, center])?;
        enc.clear_buffer(tmp.at(3), 0, Some(center));
        let module = "lb_skin";
        let params = |level: u32| -> Vec<u32> {
            let ck = level as f32 * self.step;
            let c_lo = ck - self.step;
            vec![io.pw, io.ph, io.pad, cw, ch, self.r, level, ck.to_bits(), c_lo.to_bits(), self.step.to_bits(), self.inv_band.to_bits(), 0]
        };
        for level in 0..self.levels {
            let p = params(level);
            dispatch(g, enc, timer.as_deref_mut(), &key(module, "pb_h"), &p, &[(1, io.src), (3, tmp.at(0)), (4, tmp.at(1))], [groups(cw, 64), groups(io.ph, 64), 1])?;
            dispatch(
                g,
                enc,
                timer.as_deref_mut(),
                &key(module, "pb_v"),
                &p,
                &[(1, io.src), (3, tmp.at(0)), (4, tmp.at(1)), (5, tmp.at(2)), (6, tmp.at(3))],
                [groups(cw, 256), groups(ch, 16), 1],
            )?;
        }
        let skin = g.shared("skin_lut", crate::engine::filters::extra::skin_table);
        let un = super::point::unpremul(g);
        dispatch(g, enc, timer, &key(module, "pb_out"), &params(0), &[(1, io.src), (2, io.dst), (6, tmp.at(3)), (7, &skin), (8, &un)], [groups(cw, 16), groups(ch, 16), 1])
    }
}
