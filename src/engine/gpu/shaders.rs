//! WGSL modules and the compute-pipeline cache.
//!
//! Every module is the shared prelude (`common.wgsl`) plus its own source.
//! Modules are written against `half` / `half4`, which alias `f32` unless the
//! adapter has `shader-f16`, where the f16 build is used for kernels that pass
//! the CPU-equality tests with it. Pipelines are keyed by module, entry point,
//! precision and overridable constants (workgroup sizes), created on first use
//! and precompiled in the background at startup.

use std::sync::Arc;

use super::device::Gpu;

pub const PRELUDE: &str = include_str!("wgsl/common.wgsl");

/// `(name, source)` of every module.
pub const MODULES: &[(&str, &str)] = &[
    ("probe", include_str!("wgsl/probe.wgsl")),
    ("point", include_str!("wgsl/point.wgsl")),
    ("blur", include_str!("wgsl/blur.wgsl")),
    ("mips", include_str!("wgsl/mips.wgsl")),
    ("util", include_str!("wgsl/util.wgsl")),
    ("warp", include_str!("wgsl/warp.wgsl")),
    ("raw", include_str!("wgsl/raw.wgsl")),
    ("denoise", include_str!("wgsl/denoise.wgsl")),
    ("refine", include_str!("wgsl/refine.wgsl")),
    ("liquify", include_str!("wgsl/liquify.wgsl")),
    ("perspwarp", include_str!("wgsl/perspwarp.wgsl")),
    ("compose", include_str!("wgsl/compose.wgsl")),
    ("stats", include_str!("wgsl/stats.wgsl")),
    ("stats_sg", concat!(include_str!("wgsl/stats.wgsl"), include_str!("wgsl/stats_sg.wgsl"))),
    ("vblur", concat!(include_str!("wgsl/bgfx.wgsl"), include_str!("wgsl/vblur.wgsl"))),
    ("bgather", concat!(include_str!("wgsl/bgfx.wgsl"), include_str!("wgsl/bgather.wgsl"))),
    ("poisson", include_str!("wgsl/poisson.wgsl")),
    ("patchmatch", include_str!("wgsl/patchmatch.wgsl")),
    ("lb_spans", include_str!("wgsl/lb_spans.wgsl")),
    ("lb_sharpen", include_str!("wgsl/lb_sharpen.wgsl")),
    ("lb_motion", include_str!("wgsl/lb_motion.wgsl")),
    ("lb_skin", include_str!("wgsl/lb_skin.wgsl")),
    ("styles", include_str!("wgsl/styles.wgsl")),
    ("rank", include_str!("wgsl/rank.wgsl")),
    ("rank_hist", concat!(include_str!("wgsl/rank_scan_sg.wgsl"), include_str!("wgsl/rank_hist.wgsl"))),
    ("rank_hist_wg", concat!(include_str!("wgsl/rank_scan_wg.wgsl"), include_str!("wgsl/rank_hist.wgsl"))),
    ("rank_surf", include_str!("wgsl/rank_surf.wgsl")),
    ("stylize", include_str!("wgsl/stylize.wgsl")),
    ("cells", include_str!("wgsl/cells.wgsl")),
    ("oil", include_str!("wgsl/oil.wgsl")),
    ("render", include_str!("wgsl/render.wgsl")),
    ("resample", include_str!("wgsl/resample.wgsl")),
    ("displace", include_str!("wgsl/displace.wgsl")),
    ("gallery", concat!(include_str!("wgsl/gallery_lib.wgsl"), include_str!("wgsl/gallery.wgsl"))),
    ("gallery_a", concat!(include_str!("wgsl/gallery_lib.wgsl"), include_str!("wgsl/gallery_a.wgsl"))),
    ("gallery_b", concat!(include_str!("wgsl/gallery_lib.wgsl"), include_str!("wgsl/gallery_b.wgsl"))),
    ("gallery_s", concat!(include_str!("wgsl/gallery_lib.wgsl"), include_str!("wgsl/gallery_s.wgsl"))),
    ("gallery_t", concat!(include_str!("wgsl/gallery_lib.wgsl"), include_str!("wgsl/gallery_t.wgsl"))),
    ("gallery_m", concat!(include_str!("wgsl/gallery_lib.wgsl"), include_str!("wgsl/gallery_m.wgsl"))),
];

/// Modules that need the `subgroups` feature (they have shared-memory
/// fallbacks under another name).
pub const SUBGROUP_MODULES: &[&str] = &["rank_hist", "stats_sg"];

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipeKey {
    pub module: &'static str,
    pub entry: &'static str,
    pub f16: bool,
    pub consts: Vec<(&'static str, u32)>,
}

pub fn source(name: &str) -> Option<&'static str> {
    MODULES.iter().find(|m| m.0 == name).map(|m| m.1)
}

/// The full text compiled for a module.
pub fn full_source(name: &str, f16: bool) -> Option<String> {
    let body = source(name)?;
    let mut s = String::new();
    if f16 {
        s.push_str("enable f16;\n");
    }
    // Naga 29 rejects `enable subgroups;` (subgroup operations are gated by
    // the device's SUBGROUP feature instead), so subgroup modules carry no
    // directive.
    if f16 {
        s.push_str("alias half = f16;\nalias half4 = vec4<f16>;\n");
    } else {
        s.push_str("alias half = f32;\nalias half4 = vec4<f32>;\n");
    }
    s.push_str(PRELUDE);
    s.push('\n');
    s.push_str(body);
    Some(s)
}

/// Entry points declared with `@compute` in a module.
pub fn entry_points(src: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut compute = false;
    for line in src.lines() {
        let l = line.trim();
        if l.starts_with("@compute") {
            compute = true;
        }
        if compute && let Some(rest) = l.split("fn ").nth(1) {
            if let Some(name) = rest.split('(').next() {
                out.push(name.trim());
            }
            compute = false;
        }
    }
    out
}

impl Gpu {
    fn module(&self, name: &'static str, f16: bool) -> Result<Arc<wgpu::ShaderModule>, String> {
        if let Some(m) = self.modules.lock().unwrap().get(&(name, f16)) {
            return Ok(m.clone());
        }
        let src = full_source(name, f16).ok_or_else(|| format!("unknown shader module {name}"))?;
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let m = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(name), source: wgpu::ShaderSource::Wgsl(src.into()) });
        if let Some(e) = super::device::block_on(scope.pop()) {
            return Err(format!("shader {name}: {e}"));
        }
        let m = Arc::new(m);
        self.modules.lock().unwrap().insert((name, f16), m.clone());
        Ok(m)
    }

    /// The pipeline for `key`, compiling it on first use.
    pub fn pipeline(&self, key: &PipeKey) -> Result<Arc<wgpu::ComputePipeline>, String> {
        if let Some(p) = self.pipelines.lock().unwrap().get(key) {
            return Ok(p.clone());
        }
        let module = self.module(key.module, key.f16 && self.caps.f16)?;
        let consts: Vec<(&str, f64)> = key.consts.iter().map(|&(k, v)| (k, v as f64)).collect();
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key.entry),
            layout: None,
            module: &module,
            entry_point: Some(key.entry),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &consts, zero_initialize_workgroup_memory: false },
            cache: self.cache.as_ref(),
        });
        if let Some(e) = super::device::block_on(scope.pop()) {
            return Err(format!("pipeline {}::{}: {e}", key.module, key.entry));
        }
        let p = Arc::new(p);
        self.pipelines.lock().unwrap().insert(key.clone(), p.clone());
        Ok(p)
    }

    /// Compiles every entry point of every module (default constants), so
    /// the first filter doesn't pay for shader compilation. Returns the
    /// number of pipelines and the failures.
    pub fn precompile(&self) -> (usize, Vec<String>) {
        let mut n = 0;
        let mut errs = Vec::new();
        for &(name, src) in MODULES {
            if SUBGROUP_MODULES.contains(&name) && !self.caps.subgroups {
                continue;
            }
            for f16 in [false, true] {
                if f16 && (!self.caps.f16 || !src.contains("half")) {
                    continue;
                }
                for e in entry_points(src) {
                    // Entry names come from the static sources.
                    let e: &'static str = src.match_indices(e).next().map(|(i, _)| &src[i..i + e.len()]).unwrap();
                    match self.pipeline(&PipeKey { module: name, entry: e, f16, consts: Vec::new() }) {
                        Ok(_) => n += 1,
                        Err(er) => errs.push(er),
                    }
                }
            }
        }
        self.save_pipeline_cache();
        (n, errs)
    }
}
