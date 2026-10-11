//! GPU-against-CPU equality, shader validation for every backend, the
//! fallback paths and the cost model.
//!
//! GPU tests skip themselves when no adapter exists or `PF_GPU=off`.

use std::sync::{Arc, OnceLock};

use super::run::RunOpts;
use super::*;
use crate::engine::filters::{self, Filter};
use crate::engine::grid::{RectI, TS, TileGrid};
use crate::engine::px;
use crate::engine::selection::Selection;

pub fn gpu() -> Option<Arc<Gpu>> {
    if env_mode() == Some(Mode::Off) {
        return None;
    }
    let g = context(true);
    if g.is_none() {
        eprintln!("no GPU adapter ({}); skipping", describe());
    }
    g
}

/// The dog photo (1024 x 683 after downscale in the loader, or as stored).
pub fn photo(name: &str) -> TileGrid {
    static CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<String, TileGrid>>> = OnceLock::new();
    let c = CACHE.get_or_init(Default::default);
    if let Some(g) = c.lock().unwrap().get(name) {
        return g.clone();
    }
    let path = format!("testdata/photos/{name}.jpg");
    let g = crate::engine::io::open_image(std::path::Path::new(&path)).unwrap_or_else(|_| crate::engine::ops::procedural(1531, 1021));
    c.lock().unwrap().insert(name.to_string(), g.clone());
    g
}

/// Edge cases in one odd-sized grid: a photo crop with an alpha ramp, a
/// fully transparent band, a flat opaque block and a flat translucent block.
pub fn edge_case() -> TileGrid {
    let src = photo("butterfly");
    let (w, h) = (1013u32, 777u32);
    let mut buf = vec![0u32; (w * h) as usize];
    src.read_region(37, 51, w as usize, h as usize, &mut buf);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let a = if x < 300 {
                255
            } else if x < 600 {
                ((x - 300) * 255 / 300) as u8
            } else if x < 700 {
                0
            } else {
                200
            };
            let [r, g, b, _] = px::unpack(buf[i]);
            buf[i] = px::premultiply(r, g, b, a as u8);
            if (300..560).contains(&y) && x >= 800 {
                buf[i] = px::premultiply(40, 160, 220, 255);
            }
        }
    }
    let mut g = TileGrid::new(w, h);
    g.tiles = crate::engine::grid::build_grid(w, h, |tx, ty| {
        let mut t = vec![0u32; TS * TS].into_boxed_slice();
        for y in 0..TS {
            for x in 0..TS {
                let (gx, gy) = (tx as u32 * 256 + x as u32, ty as u32 * 256 + y as u32);
                if gx < w && gy < h {
                    t[y * TS + x] = buf[(gy * w + gx) as usize];
                }
            }
        }
        crate::engine::grid::finish_tile(t)
    })
    .tiles;
    g
}

/// First differing pixel over the whole tile area (padding included).
pub fn diff_exact(a: &TileGrid, b: &TileGrid) -> Option<String> {
    if (a.w, a.h) != (b.w, b.h) {
        return Some(format!("size {}x{} vs {}x{}", a.w, a.h, b.w, b.h));
    }
    for (i, (ta, tb)) in a.tiles.iter().zip(&b.tiles).enumerate() {
        let a_pixels = ta.as_ref().map(|t| t.read());
        let b_pixels = tb.as_ref().map(|t| t.read());
        let pa = |k: usize| a_pixels.as_ref().map_or(0, |t| t[k]);
        let pb = |k: usize| b_pixels.as_ref().map_or(0, |t| t[k]);
        for k in 0..TS * TS {
            if pa(k) != pb(k) {
                let (tx, ty) = (i as u32 % a.cols, i as u32 / a.cols);
                return Some(format!(
                    "pixel ({}, {}): {:?} vs {:?}",
                    tx * 256 + (k % TS) as u32,
                    ty * 256 + (k / TS) as u32,
                    px::unpack(pa(k)),
                    px::unpack(pb(k))
                ));
            }
        }
    }
    None
}

/// Max per-channel difference and mean ΔE2000 over the image area.
pub fn diff_stats(a: &TileGrid, b: &TileGrid) -> (u8, f64) {
    let (w, h) = (a.w as usize, a.h as usize);
    let mut pa = vec![0u32; w * h];
    let mut pb = vec![0u32; w * h];
    a.read_region(0, 0, w, h, &mut pa);
    b.read_region(0, 0, w, h, &mut pb);
    let mut mx = 0u8;
    for (&x, &y) in pa.iter().zip(&pb) {
        let (ux, uy) = (px::unpack(x), px::unpack(y));
        for c in 0..4 {
            mx = mx.max(ux[c].abs_diff(uy[c]));
        }
    }
    let rgb = |v: &[u32]| -> Vec<[u8; 3]> {
        v.iter()
            .map(|&p| {
                let c = px::unpremultiply(p);
                [c[0], c[1], c[2]]
            })
            .collect()
    };
    (mx, crate::engine::metrics::mean_delta_e(&rgb(&pa), &rgb(&pb), None))
}

pub fn cpu(g: &TileGrid, f: &Filter, sel: Option<&Selection>) -> TileGrid {
    crate::engine::ops::apply_filter_cpu(g, f, sel, &Default::default(), None)
}

pub fn on_gpu(gpu: &Arc<Gpu>, g: &TileGrid, f: &Filter, sel: Option<&Selection>, opts: &RunOpts) -> (TileGrid, PathTaken) {
    apply_filter_on(gpu, g, f, sel, &Default::default(), opts)
}

/// Bit-exact GPU = CPU on the photos, the edge-case grid, a selection, and
/// one-tile-row bands (every band seam).
pub fn check_exact(gpu: &Arc<Gpu>, f: &Filter) {
    let ell = Selection::ellipse(RectI::new(150, 90, 900, 640));
    let cases: Vec<(&str, TileGrid, Option<&Selection>, RunOpts)> = vec![
        ("dog", photo("dog_grass"), None, RunOpts::default()),
        ("edge", edge_case(), None, RunOpts::default()),
        ("edge+sel", edge_case(), Some(&ell), RunOpts::default()),
        ("butterfly 1-row bands", photo("butterfly"), None, RunOpts { max_band_rows: Some(1), ..Default::default() }),
    ];
    for (name, g, sel, opts) in cases {
        let c = cpu(&g, f, sel);
        let (o, path) = on_gpu(gpu, &g, f, sel, &opts);
        assert_eq!(path, PathTaken::Gpu, "{} on {name}: {path:?}", f.name());
        if let Some(d) = diff_exact(&o, &c) {
            panic!("{f:?} on {name}: GPU != CPU at {d}");
        }
    }
}

/// Float pipelines: at most 1/255 per channel, mean ΔE2000 <= 0.2.
pub fn check_close(gpu: &Arc<Gpu>, f: &Filter) {
    for (name, g) in [("dog", photo("dog_grass")), ("edge", edge_case())] {
        let c = cpu(&g, f, None);
        let (o, path) = on_gpu(gpu, &g, f, None, &RunOpts::default());
        assert_eq!(path, PathTaken::Gpu, "{} on {name}", f.name());
        let (mx, de) = diff_stats(&o, &c);
        assert!(mx <= 1 && de <= 0.2, "{} on {name}: max diff {mx}, mean ΔE {de:.4}", f.name());
    }
}

#[test]
#[ignore]
fn debug_one() {
    let Some(g) = gpu() else { return };
    let img = edge_case();
    let f = Filter::GaussianBlur { radius: 1.3 };
    let c = cpu(&img, &f, None);
    let (o, _) = on_gpu(&g, &img, &f, None, &RunOpts::default());
    for y in 0..3 {
        for x in 1008..1024 {
            println!("({x},{y}) in {:?} cpu {:?} gpu {:?}", px::unpack(img.pixel(x.min(1012), y)), px::unpack(c.pixel(x, y)), px::unpack(o.pixel(x, y)));
        }
    }
}

#[test]
fn naga_validates_every_module_for_all_backends() {
    use naga::back::{hlsl, msl, spv};
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    for &(name, src) in shaders::MODULES {
        for f16 in [false, true] {
            if f16 && !src.contains("half") {
                continue;
            }
            let full = shaders::full_source(name, f16).unwrap();
            let module = naga::front::wgsl::parse_str(&full).unwrap_or_else(|e| panic!("{name} (f16 {f16}): {}", e.emit_to_string(&full)));
            let info = Validator::new(ValidationFlags::all(), Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name} (f16 {f16}): {}", e.emit_to_string(&full)));
            for ep in &module.entry_points {
                let (m, i) = naga::back::pipeline_constants::process_overrides(&module, &info, Some((naga::ShaderStage::Compute, &ep.name)), &Default::default())
                    .unwrap_or_else(|e| panic!("{name}::{}: overrides: {e}", ep.name));
                // Metal.
                let mo = msl::Options { lang_version: (2, 4), ..Default::default() };
                let mp = msl::PipelineOptions { entry_point: Some((naga::ShaderStage::Compute, ep.name.clone())), ..Default::default() };
                msl::write_string(&m, &i, &mo, &mp).unwrap_or_else(|e| panic!("{name}::{} MSL: {e}", ep.name));
                // DirectX 12.
                let ho = hlsl::Options { shader_model: hlsl::ShaderModel::V6_2, ..Default::default() };
                let hp = hlsl::PipelineOptions { entry_point: Some((naga::ShaderStage::Compute, ep.name.clone())) };
                let mut out = String::new();
                hlsl::Writer::new(&mut out, &ho, &hp).write(&m, &i, None).unwrap_or_else(|e| panic!("{name}::{} HLSL: {e}", ep.name));
                // Vulkan.
                let so = spv::Options { lang_version: (1, 3), ..Default::default() };
                let sp = spv::PipelineOptions { shader_stage: naga::ShaderStage::Compute, entry_point: ep.name.clone() };
                spv::write_vec(&m, &i, &so, Some(&sp)).unwrap_or_else(|e| panic!("{name}::{} SPIR-V: {e}", ep.name));
            }
        }
    }
}

#[test]
fn every_pipeline_compiles_on_the_device() {
    let Some(g) = gpu() else { return };
    let (n, errs) = g.precompile();
    assert!(errs.is_empty(), "{errs:#?}");
    assert!(n > 5);
}

#[test]
fn point_adjustments_are_bit_exact() {
    let Some(g) = gpu() else { return };
    let list = vec![
        Filter::BrightnessContrast { brightness: 40.0, contrast: 30.0 },
        Filter::Levels { black: 12.0, white: 230.0, gamma: 1.3 },
        Filter::Curves { pts: [0.0, 80.0, 128.0, 180.0, 255.0] },
        Filter::Exposure { exposure: 0.7, offset: 0.01, gamma: 1.1 },
        Filter::Invert,
        Filter::Solarize,
        Filter::Posterize { levels: 5.0 },
        Filter::Threshold { level: 128.0 },
        Filter::Threshold { level: 77.5 },
        Filter::Desaturate,
        filters::hue_saturation(25.0, 30.0, -10.0),
        filters::vibrance(40.0, 10.0),
        filters::black_white([40.0, 60.0, 40.0, 60.0, 20.0, 80.0]),
        Filter::Average { color: px::pack(10, 20, 30, 255) },
    ];
    for f in &list {
        check_exact(&g, f);
    }
}

#[test]
fn add_noise_matches_the_cpu_grain() {
    let Some(g) = gpu() else { return };
    check_close(&g, &Filter::AddNoise { amount: 25.0, mono: false });
    check_close(&g, &Filter::AddNoise { amount: 12.0, mono: true });
}

#[test]
fn blurs_and_sharpens_are_bit_exact() {
    let Some(g) = gpu() else { return };
    let list = vec![
        Filter::GaussianBlur { radius: 0.2 },
        Filter::GaussianBlur { radius: 1.3 },
        Filter::GaussianBlur { radius: 2.6 },
        Filter::GaussianBlur { radius: 9.0 },
        Filter::GaussianBlur { radius: 120.0 },
        Filter::BoxBlur { radius: 7.0 },
        Filter::BoxBlur { radius: 200.0 },
        Filter::Blur,
        Filter::BlurMore,
        Filter::HighPass { radius: 4.0 },
        Filter::HighPass { radius: 150.0 },
        Filter::UnsharpMask { amount: 150.0, radius: 2.0, threshold: 3.0 },
        Filter::UnsharpMask { amount: 80.0, radius: 30.0, threshold: 0.0 },
        Filter::UnsharpMask { amount: 120.0, radius: 140.0, threshold: 4.6 },
        Filter::Sharpen,
        Filter::SharpenMore,
        Filter::SharpenEdges,
        Filter::SmartSharpen { amount: 100.0, radius: 1.5, noise: 5.0, opts: Default::default() },
    ];
    for f in &list {
        check_exact(&g, f);
    }
}

#[test]
fn simulated_device_loss_mid_filter_falls_back_identically() {
    if env_mode() == Some(Mode::Off) {
        return;
    }
    // Fresh tiles: the resident cache is global and keyed by tile identity,
    // so the shared photo may be resident from the shared context (e.g.
    // `warps_match_the_cpu`), whose buffers don't exist on this device.
    let shared = photo("dog_grass");
    let mut img = shared.clone();
    img.tiles = crate::engine::grid::build_grid(shared.w, shared.h, |tx, ty| {
        let mut t = vec![0u32; TS * TS].into_boxed_slice();
        shared.read_region(tx * TS as i32, ty * TS as i32, TS, TS, &mut t);
        crate::engine::grid::finish_tile(t)
    })
    .tiles;
    for f in [Filter::GaussianBlur { radius: 6.0 }, Filter::Levels { black: 20.0, white: 220.0, gamma: 0.8 }] {
        // A private context, so destroying it doesn't disturb other tests.
        let Ok(g) = Gpu::new() else { return };
        let g = Arc::new(g);
        let c = cpu(&img, &f, None);
        let opts = RunOpts { lose_device_at_band: Some(2), max_band_rows: Some(1), ..Default::default() };
        let (o, path) = on_gpu(&g, &img, &f, None, &opts);
        assert!(matches!(path, PathTaken::GpuThenCpu(_)), "{path:?}");
        assert!(diff_exact(&o, &c).is_none(), "{}", f.name());
        assert!(g.is_failed());
        // A dead context goes straight to the CPU, still identical.
        let (o2, path2) = on_gpu(&g, &img, &f, None, &RunOpts::default());
        assert!(matches!(path2, PathTaken::Cpu(_)), "{path2:?}");
        assert!(diff_exact(&o2, &c).is_none());
    }
}

#[test]
fn off_and_force_give_identical_pixels() {
    let Some(g) = gpu() else { return };
    let img = edge_case();
    let f = Filter::UnsharpMask { amount: 90.0, radius: 3.0, threshold: 2.0 };
    let off = cpu(&img, &f, None);
    let (force, path) = on_gpu(&g, &img, &f, None, &RunOpts::default());
    assert_eq!(path, PathTaken::Gpu);
    assert!(diff_exact(&off, &force).is_none());
    // And through the public switch, whatever PF_GPU says.
    let auto = filters::apply(&img, &f, None, &Default::default());
    assert!(diff_exact(&off, &auto).is_none());
}

#[test]
fn cost_model_round_trips_and_prefers_the_cpu_for_small_regions() {
    let mut m = cost::Model { key: "k".into(), ..Default::default() };
    let f = Filter::GaussianBlur { radius: 10.0 };
    m.ops.insert(cost::op_key(&f, 0), cost::Rates { gpu_ns_px: 0.2, cpu_ns_px: 1.5 });
    m.ops.insert(cost::area_key(&f, 0, 16 << 20), cost::Rates { gpu_ns_px: 0.2, cpu_ns_px: 1.5 });
    assert_eq!(cost::Model::parse(&m.to_text()), m);
    let (g_small, c_small) = m.predict(&f, 64 * 64, 0).unwrap();
    assert!(g_small > c_small, "fixed GPU cost dominates tiny regions");
    let (g_big, c_big) = m.predict(&f, 16 << 20, 0).unwrap();
    assert!(g_big < c_big, "{g_big} vs {c_big}");
    // Different radii fall in different buckets.
    assert_ne!(cost::op_key(&f, 0), cost::op_key(&Filter::GaussianBlur { radius: 200.0 }, 0));
    // A preview at a coarser level costs like a smaller radius.
    assert_eq!(cost::op_key(&Filter::GaussianBlur { radius: 40.0 }, 2), cost::op_key(&Filter::GaussianBlur { radius: 10.0 }, 0));
}

#[test]
fn cost_model_keeps_end_to_end_rates_separate_by_working_set() {
    let f = Filter::GaussianBlur { radius: 20.0 };
    let mut m = cost::Model { fixed_ms: 2.0, transfer_ns_b: 10.0, ..Default::default() };
    let small = 24_000_000;
    let large = 150_000_000;
    m.ops.insert(cost::area_key(&f, 0, small), cost::Rates { gpu_ns_px: 1.0, cpu_ns_px: 2.0 });
    m.ops.insert(cost::area_key(&f, 0, large), cost::Rates { gpu_ns_px: 3.0, cpu_ns_px: 1.0 });
    assert_eq!(m.predict(&f, small, 0), Some((26.0, 48.0)), "area rates already include transfers");
    assert_eq!(m.predict(&f, large, 0), Some((452.0, 150.0)));
    assert_eq!(cost::Model::parse(&m.to_text()), m);
    m.observe(&f, small, 0, 50.0, true);
    assert_eq!(m.predict(&f, large, 0), Some((452.0, 150.0)), "a smaller run must not change large-document rates");
    assert!(m.predict(&f, small, 0).unwrap().0 > 26.0);
}

#[test]
fn cost_model_area_boundaries_and_partial_observations() {
    let f = Filter::GaussianBlur { radius: 20.0 };
    assert_ne!(cost::area_key(&f, 0, (1 << 25) - 1), cost::area_key(&f, 0, 1 << 25));
    assert_eq!(cost::area_key(&f, 0, 1 << 25), cost::area_key(&f, 0, (1 << 25) + 1));
    assert_ne!(cost::area_key(&f, 0, 14204 * 10652), cost::area_key(&f, 0, 16384 * 16384));
    let mut m = cost::Model::default();
    let px = 1 << 22;
    m.ops.insert(cost::op_key(&f, 0), cost::Rates { gpu_ns_px: 1.0, cpu_ns_px: 2.0 });
    let before = m.predict(&f, px, 0).unwrap();
    m.observe(&f, px, 0, 20.0, false);
    let after = m.predict(&f, px, 0).unwrap();
    assert!((before.0 - after.0).abs() < 1e-9, "learning CPU must preserve the GPU estimate");
    assert!(after.1 > before.1);
    let saved = m.clone();
    m.observe(&f, px, 0, f64::NAN, true);
    assert_eq!(m, saved);
}

#[test]
fn cost_model_distinguishes_gradient_algorithms() {
    use crate::engine::gradient::{GradKind, Gradient, GradientFill};
    let keys: std::collections::HashSet<_> = GradKind::ALL.into_iter().map(|kind| {
        let fill = GradientFill::new(Gradient::two("bw", [0; 3], [255; 3]), kind, (0.0, 0.0), (1.0, 1.0), false, true, 1.0);
        cost::op_key(&Filter::Gradient(Arc::new(fill)), 0)
    }).collect();
    assert_eq!(keys.len(), GradKind::ALL.len(), "angle/radial math must not change linear-gradient estimates");
}

/// Bandwidth ceiling: a plain copy kernel and buffer-to-buffer copy.
#[test]
#[ignore]
fn probe_bandwidth() {
    let Some(g) = gpu() else { return };
    let n: u64 = 16384 * 2048;
    let a = exec::buffer(&g, n * 4, exec::STORAGE).unwrap();
    let b = exec::buffer(&g, n * 4, exec::STORAGE).unwrap();
    for _ in 0..3 {
        let mut enc = exec::encoder(&g);
        let mut timer = exec::Timer::new(&g, 16);
        let gx = exec::groups(n as u32, 256).min(32768);
        let gy = exec::groups(n as u32, gx * 256);
        let key = shaders::PipeKey { module: "probe", entry: "copy", f16: false, consts: Vec::new() };
        for _ in 0..10 {
            exec::dispatch(&g, &mut enc, timer.as_mut(), &key, &[0, 0, n as u32, 0], &[(1, &a), (2, &b)], [gx, gy, 1]).unwrap();
        }
        let t = std::time::Instant::now();
        if let Some(t) = timer.as_mut() {
            t.resolve(&g, &mut enc);
        }
        let idx = exec::submit(&g, enc);
        exec::wait(&g, idx).unwrap();
        let wall = t.elapsed().as_secs_f64() * 1e3;
        let passes = timer.unwrap().read_passes(&g).unwrap();
        println!("copy {} MB x10: wall {wall:.2} ms ({:.0} GB/s), passes {:?}", n * 4 >> 20, (n * 80) as f64 / wall / 1e6, passes.iter().map(|p| (p.1 * 100.0).round() / 100.0).collect::<Vec<_>>());
    }
}

/// GPU previews at display mip levels equal `Filter::render` there.
pub fn check_preview(gpu: &Arc<Gpu>, f: &Filter) {
    let mut layer = crate::engine::document::Layer::new(1, "t", edge_case());
    layer.mips = crate::engine::mips::build_mips(&layer.grid);
    for (level, rect) in [(1u32, RectI::new(-3, 5, 400, 300)), (2, RectI::new(37, 0, 260, 197)), (3, RectI::new(0, 0, 127, 98))] {
        let (w, h) = (rect.w() as usize, rect.h() as usize);
        let mut c = vec![0u32; w * h];
        f.render(&filters::Src::layer(&layer), level, rect.x0, rect.y0, w, h, &mut c, &mut Vec::new());
        let o = preview::render(gpu, f, &layer, level, rect).unwrap_or_else(|e| panic!("{f:?} level {level}: {e}"));
        if let Some(i) = (0..w * h).find(|&i| c[i] != o[i]) {
            panic!("{f:?} preview level {level}: ({}, {}) GPU {:?} CPU {:?}", rect.x0 + (i % w) as i32, rect.y0 + (i / w) as i32, px::unpack(o[i]), px::unpack(c[i]));
        }
    }
}

#[test]
fn previews_match_the_cpu_at_mip_levels() {
    let Some(g) = gpu() else { return };
    for f in [
        Filter::Levels { black: 12.0, white: 230.0, gamma: 1.3 },
        filters::hue_saturation(25.0, 30.0, -10.0),
        Filter::GaussianBlur { radius: 9.0 },
        Filter::GaussianBlur { radius: 60.0 },
        Filter::BoxBlur { radius: 12.0 },
        Filter::UnsharpMask { amount: 150.0, radius: 6.0, threshold: 2.0 },
        Filter::HighPass { radius: 20.0 },
        Filter::Blur,
    ] {
        check_preview(&g, &f);
    }
}

/// Max diff, mean ΔE and the count of pixels off by more than 1.
pub fn report(name: &str, a: &TileGrid, b: &TileGrid) -> (u8, f64, usize) {
    let (w, h) = (a.w as usize, a.h as usize);
    let mut pa = vec![0u32; w * h];
    let mut pb = vec![0u32; w * h];
    a.read_region(0, 0, w, h, &mut pa);
    b.read_region(0, 0, w, h, &mut pb);
    let over = pa.iter().zip(&pb).filter(|(x, y)| px::unpack(**x).iter().zip(px::unpack(**y)).any(|(p, q)| p.abs_diff(q) > 1)).count();
    let (mx, de) = diff_stats(a, b);
    println!("{name}: max {mx}, mean ΔE {de:.4}, {over} px over 1 ({:.4} %)", over as f64 * 100.0 / (w * h) as f64);
    (mx, de, over)
}

pub fn warps() -> Vec<Filter> {
    let (cx, cy, rad) = (512.0, 341.0, 341.0);
    let w = |name, warp| Filter::Warp { name, warp };
    use crate::engine::filters::Warp as W;
    vec![
        w("Twirl", W::Twirl { angle: 200.0, cx, cy, radius: rad }),
        w("Pinch", W::Pinch { amount: 60.0, cx, cy, radius: rad }),
        w("Spherize", W::Spherize { amount: 80.0, cx, cy, radius: rad }),
        w("ZigZag", W::ZigZag { amount: 40.0, ridges: 6.0, cx, cy, radius: rad }),
        w("Polar", W::Polar { to_polar: true, w: 1024.0, h: 683.0 }),
        w("Polar back", W::Polar { to_polar: false, w: 1024.0, h: 683.0 }),
        w("Ripple", W::Ripple { amount: 100.0, size: 40.0 }),
        w("Wave", W::Wave { wavelength: 120.0, amplitude: 9.0 }),
        w("Offset", W::Offset { dx: 300.5, dy: -77.0, w: 1024.0, h: 683.0 }),
        w("Shear", W::Shear { amount: 40.0, h: 683.0 }),
        w("Homography", W::Homography { m: crate::engine::filters::distort::square_to_quad([(-20.0, 10.0), (1100.0, -30.0), (990.0, 700.0), (40.0, 650.0)]) }),
        w("Wide angle", W::WideAngle { strength: 60.0, scale: 100.0, cx, cy, diag: 1200.0 }),
    ]
}

#[test]
fn warps_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let img = photo("dog_grass");
    for f in warps() {
        let c = cpu(&img, &f, None);
        let (o, path) = on_gpu(&g, &img, &f, None, &RunOpts::default());
        assert_eq!(path, PathTaken::Gpu);
        check_resampled(f.name(), &o, &c);
    }
}

/// Tolerance for kernels that resample through the CPU sampler's two
/// cascaded 8-bit bilinear fractions: a coordinate that differs in its last
/// float bits can flip both fractions of one pixel (2/255 there), so allow
/// that on at most 0.01 % of pixels; everything else within 1/255, and mean
/// ΔE2000 <= 0.2.
pub fn check_resampled(name: &str, o: &TileGrid, c: &TileGrid) {
    let (mx, de, over) = report(name, o, c);
    let n = (o.w * o.h) as usize;
    assert!(mx <= 2 && over * 10_000 <= n && de <= 0.2, "{name}: max {mx}, {over} px over 1, ΔE {de:.4}");
}

#[test]
#[ignore]
fn probe_fixed_overhead() {
    let Some(g) = gpu() else { return };
    let small = crate::engine::ops::procedural(256, 256);
    for _ in 0..10 {
        let t = std::time::Instant::now();
        let _ = on_gpu(&g, &small, &Filter::Invert, None, &RunOpts::default());
        println!("invert 256²: {:.3} ms", t.elapsed().as_secs_f64() * 1e3);
    }
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let enc = exec::encoder(&g);
        let idx = exec::submit(&g, enc);
        let t1 = t.elapsed().as_secs_f64() * 1e3;
        exec::wait(&g, idx).unwrap();
        println!("empty submit {t1:.3} ms + wait {:.3} ms", t.elapsed().as_secs_f64() * 1e3 - t1);
    }
}

/// GPU histogram statistics equal `filters::stats` (subgroup and
/// shared-memory variants).
#[test]
fn histogram_stats_match_the_cpu() {
    let Some(g) = gpu() else { return };
    for grid in [photo("dog_grass"), edge_case(), crate::engine::ops::procedural(3000, 2100)] {
        let layer = crate::engine::document::Layer::new(1, "s", grid);
        let c = crate::engine::filters::stats(&layer);
        let o = stats::stats(&g, &layer).unwrap();
        assert_eq!(c.hist, o.hist);
        assert_eq!(c.count, o.count);
        assert_eq!(c.sum, o.sum);
    }
}
