//! Recorded GPU programs: long sequences of small dispatches (multigrid
//! V-cycles, PatchMatch iterations) encoded in one compute pass.
//!
//! [`exec::dispatch`] makes a uniform buffer, a bind group and a pass per
//! dispatch, which is fine for a few passes per band but dominates a solver
//! that issues hundreds of tiny dispatches. A [`Prog`] collects the
//! dispatches first, puts every op's parameters into one uniform buffer
//! (one 512-byte slot per op, bound at a static offset) and records them
//! into a single pass. Dispatches inside a pass still run in order, with
//! storage writes of one visible to the next.
//!
//! Every entry point of a module used this way binds the module's whole
//! buffer set (each touches all of its bindings), so all ops take the same
//! buffers.

use std::num::NonZeroU64;
use std::sync::Arc;

use super::device::Gpu;
use super::exec::{GResult, GpuError, Timer};
use super::shaders::PipeKey;

/// Bytes of parameters per op (the uniform slot).
pub const SLOT: u64 = 512;
pub const SLOT_WORDS: usize = (SLOT / 4) as usize;

#[derive(Clone, Copy, Debug)]
pub enum Groups {
    Direct([u32; 3]),
    /// Workgroup counts read from the indirect buffer at this byte offset.
    Indirect(u64),
}

struct Op {
    entry: &'static str,
    slot: usize,
    groups: Groups,
    /// A buffer copy into the indirect buffer instead of a dispatch:
    /// (source binding, source byte offset, destination byte offset, bytes).
    copy: Option<(u32, u64, u64, u64)>,
}

/// `PF_PROG_PROFILE`: every op in its own timed pass (see [`report`]).
pub fn profiling() -> bool {
    static P: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *P.get_or_init(|| std::env::var_os("PF_PROG_PROFILE").is_some())
}

/// A timer for a profiled program (None unless profiling).
pub fn profile_timer(g: &Gpu, prog: &Prog) -> Option<Timer> {
    if profiling() { Timer::new(g, prog.len() as u32 + 2) } else { None }
}

/// Prints the profiled per-entry GPU times (after the submission ran).
pub fn report(g: &Gpu, what: &str, t: Option<&Timer>) {
    let Some(passes) = t.and_then(|t| t.read_passes(g)) else { return };
    let mut by: Vec<(String, f64, usize)> = Vec::new();
    for (l, ms) in passes {
        match by.iter_mut().find(|e| e.0 == l) {
            Some(e) => {
                e.1 += ms;
                e.2 += 1;
            }
            None => by.push((l, ms, 1)),
        }
    }
    let total: f64 = by.iter().map(|e| e.1).sum();
    eprintln!("{what}: {total:.2} ms GPU: {}", by.iter().map(|(l, ms, n)| format!("{l} {ms:.2}/{n}")).collect::<Vec<_>>().join(", "));
}

/// A recorded sequence of dispatches of one WGSL module.
pub struct Prog {
    module: &'static str,
    /// Pipeline-overridable constants of every op's pipeline.
    pub consts: Vec<(&'static str, u32)>,
    params: Vec<u32>,
    ops: Vec<Op>,
}

impl Prog {
    pub fn new(module: &'static str) -> Prog {
        Prog { module, consts: Vec::new(), params: Vec::new(), ops: Vec::new() }
    }

    /// Appends one dispatch. `params` (at most 128 words) is the op's uniform.
    pub fn op(&mut self, entry: &'static str, params: &[u32], groups: Groups) {
        assert!(params.len() <= SLOT_WORDS, "op params too long");
        if let Groups::Direct(gr) = groups
            && gr.contains(&0)
        {
            return;
        }
        let slot = self.params.len() / SLOT_WORDS;
        self.params.extend_from_slice(params);
        self.params.resize((slot + 1) * SLOT_WORDS, 0);
        self.ops.push(Op { entry, slot, groups, copy: None });
    }

    /// Copies `size` bytes of binding `src` (at `src_off`) into the indirect
    /// buffer at `dst_off`. A buffer can't be both a dispatch's storage
    /// binding and its indirect argument, so arguments computed by one op
    /// are copied out (between passes) before an indirect dispatch.
    pub fn copy_indirect(&mut self, src: u32, src_off: u64, dst_off: u64, size: u64) {
        self.ops.push(Op { entry: "", slot: 0, groups: Groups::Direct([0; 3]), copy: Some((src, src_off, dst_off, size)) });
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Records the program into `enc`. `bufs` are the module's storage
    /// bindings (every op binds all of them); `indirect` holds indirect
    /// dispatch arguments.
    pub fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, bufs: &[(u32, &wgpu::Buffer)], indirect: Option<&wgpu::Buffer>, timer: Option<&mut Timer>) -> GResult<()> {
        if self.ops.is_empty() {
            return Ok(());
        }
        let lim = g.limits.max_compute_workgroups_per_dimension;
        for op in &self.ops {
            if let Groups::Direct(gr) = op.groups
                && gr.iter().any(|&n| n > lim)
            {
                return Err(GpuError::Unsupported);
            }
        }
        let ubuf = g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prog params"),
            size: (self.params.len() * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        g.queue.write_buffer(&ubuf, 0, bytemuck::cast_slice(&self.params));
        // Pipelines and bind groups first (the pass borrows them).
        let mut pipes: Vec<Option<Arc<wgpu::ComputePipeline>>> = Vec::with_capacity(self.ops.len());
        let mut groups: Vec<Option<wgpu::BindGroup>> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            if op.copy.is_some() {
                pipes.push(None);
                groups.push(None);
                continue;
            }
            let key = PipeKey { module: self.module, entry: op.entry, f16: false, consts: self.consts.clone() };
            let pipe = g.pipeline(&key).map_err(GpuError::Validation)?;
            let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(bufs.len() + 1);
            entries.push(wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &ubuf, offset: op.slot as u64 * SLOT, size: NonZeroU64::new(SLOT) }),
            });
            for &(b, buf) in bufs {
                entries.push(wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() });
            }
            groups.push(Some(g.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe.get_bind_group_layout(0), entries: &entries })));
            pipes.push(Some(pipe));
        }
        let mut tw = timer;
        let mut i = 0;
        while i < self.ops.len() {
            if let Some((src, so, d, n)) = self.ops[i].copy {
                let sb = bufs.iter().find(|b| b.0 == src).map(|b| b.1).ok_or(GpuError::Unsupported)?;
                let db = indirect.ok_or(GpuError::Unsupported)?;
                enc.copy_buffer_to_buffer(sb, so, db, d, n);
                i += 1;
                continue;
            }
            let per_op = profiling();
            let label = if per_op { self.ops[i].entry } else { self.module };
            let writes = tw.as_deref_mut().and_then(|t| t.pass_writes(label));
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: writes });
            let first = i;
            while i < self.ops.len() && self.ops[i].copy.is_none() && !(per_op && i > first) {
                let op = &self.ops[i];
                pass.set_pipeline(pipes[i].as_ref().unwrap());
                pass.set_bind_group(0, groups[i].as_ref().unwrap(), &[]);
                match op.groups {
                    Groups::Direct(gr) => pass.dispatch_workgroups(gr[0], gr[1], gr[2]),
                    Groups::Indirect(off) => match indirect {
                        Some(b) => pass.dispatch_workgroups_indirect(b, off),
                        None => return Err(GpuError::Unsupported),
                    },
                }
                i += 1;
            }
        }
        Ok(())
    }
}
