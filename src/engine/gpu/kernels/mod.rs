//! GPU implementations of filters.
//!
//! A kernel is the GPU twin of [`Filter::apply`] / [`Filter::render`]:
//! - [`Input::Padded`] kernels (Point / Local filters) read a padded buffer
//!   (`pad = Filter::pad(level)` pixels of apron, clamped like the CPU's
//!   `read_region_clamped`; point filters read raw tiles like
//!   `read_region`) and write the unpadded center. The CPU's padded results
//!   don't depend on where a block starts, so any band of tiles gives the
//!   same pixels as the CPU's blocks and strips.
//! - [`Input::Source`] kernels (Warp / Gather filters) read the whole source
//!   level (and mips above it) from the resident cache and write the band.
//!
//! Adding a kernel:
//! 1. WGSL in `wgsl/<module>.wgsl`, registered in `shaders::MODULES`. The
//!    prelude (`common.wgsl`) has pixel packing, exact 8-bit math, the CPU's
//!    hash, and precise transcendentals (`p_sin`, `p_exp`, ...): Metal
//!    compiles with fast math, so never use the built-in transcendentals
//!    where the result feeds a threshold or a sample position.
//! 2. A `Kernel` here, returned by `plan` and counted by `supports`.
//! 3. Equality tests against the CPU (`gpu::tests` helpers): bit-exact for
//!    integer-defined filters, else at most 1/255 and mean ΔE2000 <= 0.2,
//!    plus previews at mip levels.
//! 4. A row in `bench_gpu::gpu_filters`.

pub mod blur;
pub mod liquify;
pub mod perspwarp;
pub mod blurgallery;
#[cfg(test)]
mod blurgallery_tests;
pub mod lensblur;
#[cfg(test)]
mod lensblur_tests;
pub mod displace;
pub mod gallery;
#[cfg(test)]
mod gallery_tests;
pub mod point;
pub mod raw;
#[cfg(test)]
mod raw_tests;
pub mod refine;
#[cfg(test)]
mod refine_tests;
pub mod rank;
#[cfg(test)]
mod rank_tests;
pub mod oil;
pub mod stylize;
pub mod warp;
#[cfg(test)]
mod interactive_tests;

#[cfg(test)]
mod stylize_tests;
#[cfg(test)]
mod resample_tests;

use std::sync::Arc;

use super::device::Gpu;
use super::exec::{Buf, GResult, STORAGE, Timer, buffer};
use crate::engine::filters::Filter;

/// Buffers of one band.
pub struct Io<'a> {
    /// Padded input, `pw x ph`.
    pub src: &'a wgpu::Buffer,
    /// Output center, `cw x ch` (`pw - 2 pad` x `ph - 2 pad`).
    pub dst: &'a wgpu::Buffer,
    pub pw: u32,
    pub ph: u32,
    pub pad: u32,
    pub level: u32,
    /// Level-space coordinate of `src[0]`.
    pub origin: (i32, i32),
    /// Size of the image at this level.
    pub img: (u32, u32),
    /// For [`Input::Source`] kernels: the whole source image at this level
    /// and the levels above it (`source[0]` is level `level`).
    pub source: &'a [Arc<super::resident::Image>],
}

/// What a kernel reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Input {
    /// A padded band (`Io::src`), like `Filter::render` for Point / Local.
    Padded,
    /// The whole source image plus `mips` levels above it (Warp / Gather
    /// filters, which may sample anywhere). `Io::src` is unused, `pad` is 0.
    Source { mips: u32 },
}

impl Io<'_> {
    pub fn cw(&self) -> u32 {
        self.pw - 2 * self.pad
    }
    pub fn ch(&self) -> u32 {
        self.ph - 2 * self.pad
    }
}

/// Scratch buffers of one band slot, kept across bands.
#[derive(Default)]
pub struct Temps {
    bufs: Vec<Buf>,
}

impl Temps {
    /// Makes sure `n` temporaries of at least `size` bytes exist.
    pub fn ensure(&mut self, g: &Arc<Gpu>, n: usize, size: u64) -> GResult<()> {
        for i in 0..n {
            if i >= self.bufs.len() {
                self.bufs.push(buffer(g, size, STORAGE)?);
            } else if self.bufs[i].size < size {
                self.bufs[i] = buffer(g, size, STORAGE)?;
            }
        }
        Ok(())
    }

    /// Makes sure temporary `i` holds at least `sizes[i]` bytes (for
    /// kernels whose scratch buffers differ in size, e.g. f32 planes).
    pub fn ensure_sizes(&mut self, g: &Arc<Gpu>, sizes: &[u64]) -> GResult<()> {
        for (i, &size) in sizes.iter().enumerate() {
            if i >= self.bufs.len() {
                self.bufs.push(buffer(g, size, STORAGE)?);
            } else if self.bufs[i].size < size {
                self.bufs[i] = buffer(g, size, STORAGE)?;
            }
        }
        Ok(())
    }

    pub fn at(&self, i: usize) -> &wgpu::Buffer {
        &self.bufs[i]
    }
}

pub trait Kernel: Send + Sync {
    fn input(&self) -> Input {
        Input::Padded
    }

    /// Outside the image (edge-tile padding), the output holds the blur and
    /// the scatter must apply `Filter::combiner` with the raw tile pixel, as
    /// the CPU's strip path does.
    fn raw_edge_combine(&self) -> bool {
        false
    }

    /// Band heights (in tile rows) should be a multiple of this, when the
    /// band is taller (kernels that replay the CPU's blocks of tiles).
    fn band_multiple(&self) -> u32 {
        1
    }

    /// Padded-size temporaries the kernel needs per band.
    fn temps(&self) -> usize {
        0
    }

    /// Records the kernel for one band.
    fn encode(&self, g: &Arc<Gpu>, enc: &mut wgpu::CommandEncoder, io: &Io, tmp: &mut Temps, timer: Option<&mut Timer>) -> GResult<()>;
}

/// Where the filter is being run: the full-resolution commit (whose blur
/// pass order follows `ops::apply_filter`), or a display-level preview.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    Commit,
    Preview,
}

/// The GPU program for `filter` at `level`, if there is one.
pub fn plan(g: &Arc<Gpu>, filter: &Filter, level: u32, target: Target) -> Option<Box<dyn Kernel>> {
    if let Some(k) = point::plan(g, filter) {
        return Some(k);
    }
    if let Some(k) = blur::plan(g, filter, level, target) {
        return Some(k);
    }
    if let Some(k) = warp::plan(g, filter) {
        return Some(k);
    }
    if let Some(k) = blurgallery::plan(g, filter, level, target) {
        return Some(k);
    }
    if let Some(k) = raw::plan(g, filter, level) {
        return Some(k);
    }
    if let Some(k) = lensblur::plan(g, filter, level, target) {
        return Some(k);
    }
    if let Some(k) = rank::plan(g, filter, level) {
        return Some(k);
    }
    if let Some(k) = stylize::plan(g, filter, level, target) {
        return Some(k);
    }
    if let Some(k) = oil::plan(g, filter, level) {
        return Some(k);
    }
    if let Some(k) = displace::plan(g, filter, target) {
        return Some(k);
    }
    if let Some(k) = gallery::plan(g, filter, level, target) {
        return Some(k);
    }
    None
}

/// Whether some kernel exists for this kind of filter (any parameters).
pub fn supports(filter: &Filter) -> bool {
    point::supports(filter) || blur::supports(filter) || warp::supports(filter) || raw::supports(filter) || blurgallery::supports(filter) || lensblur::supports(filter) || rank::supports(filter) || stylize::supports(filter) || oil::supports(filter) || displace::supports(filter) || gallery::supports(filter)
}

