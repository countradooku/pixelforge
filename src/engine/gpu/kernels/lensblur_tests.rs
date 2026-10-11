//! GPU = CPU for Lens Blur, Shape Blur, Smart Sharpen, Motion Blur,
//! Shadows/Highlights and Skin Smoothing (commits, band seams, selections
//! and previews at mip levels).

use std::sync::Arc;

use crate::engine::filters::{Filter, blur};
use crate::engine::gpu::Gpu;
use crate::engine::gpu::run::RunOpts;
use crate::engine::gpu::tests::*;
use crate::engine::gpu::{PathTaken, preview};
use crate::engine::grid::RectI;
use crate::engine::px;
use crate::engine::selection::Selection;

/// Float bar on the commit cases of `check_exact` (photo, edge cases, a
/// selection, one-tile-row bands): max 1/255, mean ΔE2000 <= 0.2.
fn check_float(gpu: &Arc<Gpu>, f: &Filter) {
    let ell = Selection::ellipse(RectI::new(150, 90, 900, 640));
    let cases: Vec<(&str, crate::engine::grid::TileGrid, Option<&Selection>, RunOpts)> = vec![
        ("dog", photo("dog_grass"), None, RunOpts::default()),
        ("edge", edge_case(), None, RunOpts::default()),
        ("edge+sel", edge_case(), Some(&ell), RunOpts::default()),
        ("butterfly 1-row bands", photo("butterfly"), None, RunOpts { max_band_rows: Some(1), ..Default::default() }),
    ];
    for (name, g, sel, opts) in cases {
        let c = cpu(&g, f, sel);
        let (o, path) = on_gpu(gpu, &g, f, sel, &opts);
        assert_eq!(path, PathTaken::Gpu, "{} on {name}: {path:?}", f.name());
        let (mx, de, _) = report(&format!("{f:?} on {name}"), &o, &c);
        assert!(mx <= 1 && de <= 0.2, "{f:?} on {name}: max diff {mx}, mean ΔE {de:.4}");
    }
}

/// Previews at levels 1-3 against `Filter::render`: exact, or the float bar.
fn check_preview_bar(gpu: &Arc<Gpu>, f: &Filter, exact: bool) {
    if exact {
        check_preview(gpu, f);
        return;
    }
    let mut layer = crate::engine::document::Layer::new(1, "t", edge_case());
    layer.mips = crate::engine::mips::build_mips(&layer.grid);
    for (level, rect) in [(1u32, RectI::new(-3, 5, 400, 300)), (2, RectI::new(37, 0, 260, 197)), (3, RectI::new(0, 0, 127, 98))] {
        let (w, h) = (rect.w() as usize, rect.h() as usize);
        let mut c = vec![0u32; w * h];
        f.render(&crate::engine::filters::Src::layer(&layer), level, rect.x0, rect.y0, w, h, &mut c, &mut Vec::new());
        let o = preview::render(gpu, f, &layer, level, rect).unwrap_or_else(|e| panic!("{f:?} level {level}: {e}"));
        let mut mx = 0u8;
        for (a, b) in o.iter().zip(&c) {
            for (p, q) in px::unpack(*a).iter().zip(px::unpack(*b)) {
                mx = mx.max(p.abs_diff(q));
            }
        }
        let rgb = |v: &[u32]| -> Vec<[u8; 3]> { v.iter().map(|&p| { let c = px::unpremultiply(p); [c[0], c[1], c[2]] }).collect() };
        let de = crate::engine::metrics::mean_delta_e(&rgb(&o), &rgb(&c), None);
        assert!(mx <= 1 && de <= 0.2, "{f:?} preview level {level}: max {mx}, ΔE {de:.4}");
    }
}

fn lens(radius: f32, bloom: f32, opts: blur::LensOpts) -> Filter {
    Filter::LensBlur { radius, bloom, opts }
}

#[test]
fn lens_blur_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let d = blur::LensOpts::default;
    // Plain (today's disk with bloom), radii in each CPU block size.
    for f in [lens(15.0, 30.0, d()), lens(3.0, 0.0, d()), lens(40.0, 60.0, d())] {
        check_exact(&g, &f);
        check_preview_bar(&g, &f, true);
    }
    // Iris, specular threshold and the depth map: still exact.
    let hex = blur::LensOpts { blades: 6, curvature: 30.0, rotation: 15.0, threshold: 180.0, ..d() };
    let depth = blur::LensOpts { blades: 5, depth: blur::Depth::Transparency, focal: 40.0, invert: true, ..d() };
    for f in [lens(12.0, 50.0, hex), lens(20.0, 40.0, depth)] {
        check_exact(&g, &f);
        check_preview_bar(&g, &f, true);
    }
    // Noise rounds a float sum: the float bar.
    let noisy = blur::LensOpts { noise: 12.0, gaussian: true, ..d() };
    let mono = blur::LensOpts { noise: 6.0, mono: true, blades: 3, ..d() };
    for f in [lens(9.0, 30.0, noisy), lens(9.0, 30.0, mono)] {
        check_float(&g, &f);
        check_preview_bar(&g, &f, false);
    }
}

#[test]
fn shape_blur_is_bit_exact() {
    let Some(g) = gpu() else { return };
    for shape in 0..crate::engine::filters::extra::SHAPES.len() {
        let f = Filter::ShapeBlur { radius: 9.0, shape };
        check_exact(&g, &f);
        check_preview_bar(&g, &f, true);
    }
    for f in [Filter::ShapeBlur { radius: 60.0, shape: 3 }, Filter::ShapeBlur { radius: 1.0, shape: 4 }] {
        check_exact(&g, &f);
        check_preview_bar(&g, &f, true);
    }
}

/// Prints how far one filter is from the CPU (diagnostics).
#[test]
#[ignore]
fn debug_one_filter() {
    let Some(g) = gpu() else { return };
    let img = photo("dog_grass");
    let f = lens(15.0, 30.0, blur::LensOpts::default());
    let c = cpu(&img, &f, None);
    let (o, _) = on_gpu(&g, &img, &f, None, &RunOpts::default());
    report("lens dog", &o, &c);
    let n = (0..img.h as i32).flat_map(|y| (0..img.w as i32).map(move |x| (x, y))).filter(|&(x, y)| o.pixel(x, y) != c.pixel(x, y)).count();
    println!("differing pixels: {n}");
}

fn smart(amount: f32, radius: f32, noise: f32, opts: blur::SharpenOpts) -> Filter {
    Filter::SmartSharpen { amount, radius, noise, opts }
}

#[test]
fn smart_sharpen_is_bit_exact() {
    let Some(g) = gpu() else { return };
    use blur::{Fade, Remove, SharpenOpts};
    let fs = Fade { amount: 70.0, width: 40.0, radius: 6.0 };
    let fh = Fade { amount: 50.0, width: 30.0, radius: 3.0 };
    let list = vec![
        smart(200.0, 3.0, 3.0, SharpenOpts { remove: Remove::Lens, ..Default::default() }),
        smart(150.0, 6.0, 0.0, SharpenOpts { remove: Remove::Motion, angle: 30.0, ..Default::default() }),
        smart(150.0, 4.0, 0.0, SharpenOpts { remove: Remove::Motion, angle: -90.0, ..Default::default() }),
        smart(250.0, 1.5, 2.0, SharpenOpts { shadows: fs, ..Default::default() }),
        smart(250.0, 9.0, 0.0, SharpenOpts { highlights: fh, shadows: fs, ..Default::default() }),
        smart(120.0, 2.0, 1.0, SharpenOpts { remove: Remove::Lens, shadows: fs, highlights: fh, ..Default::default() }),
        smart(300.0, 40.0, 0.0, SharpenOpts { remove: Remove::Motion, angle: 10.0, highlights: fh, ..Default::default() }),
    ];
    for f in &list {
        check_exact(&g, f);
        check_preview_bar(&g, f, true);
    }
    // The plain options stay with the blur family (and its tests).
    let plain = smart(100.0, 1.5, 5.0, SharpenOpts::default());
    assert!(crate::engine::gpu::kernels::lensblur::plan(&g, &plain, 0, crate::engine::gpu::kernels::Target::Commit).is_none());
    check_exact(&g, &plain);
}

#[test]
fn shadows_highlights_is_bit_exact() {
    let Some(g) = gpu() else { return };
    for f in [
        Filter::ShadowsHighlights { shadows: 35.0, highlights: 0.0, radius: 30.0 },
        Filter::ShadowsHighlights { shadows: 60.0, highlights: 40.0, radius: 2.0 },
        Filter::ShadowsHighlights { shadows: 20.0, highlights: 70.0, radius: 300.0 },
    ] {
        check_exact(&g, &f);
        check_preview_bar(&g, &f, true);
    }
}

#[test]
fn motion_blur_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    // Axis-aligned: integer boxes, bit-exact.
    for f in [Filter::MotionBlur { angle: 0.0, distance: 40.0 }, Filter::MotionBlur { angle: 90.0, distance: 25.0 }, Filter::MotionBlur { angle: -90.0, distance: 300.0 }] {
        check_exact(&g, &f);
        check_preview_bar(&g, &f, true);
    }
    // Sheared lines (both orientations, every CPU block size): float bar.
    for f in [
        Filter::MotionBlur { angle: 30.0, distance: 40.0 },
        Filter::MotionBlur { angle: -60.0, distance: 18.0 },
        Filter::MotionBlur { angle: 75.0, distance: 120.0 },
        Filter::MotionBlur { angle: 12.0, distance: 500.0 },
    ] {
        check_float(&g, &f);
        check_preview_bar(&g, &f, false);
    }
}

#[test]
fn skin_smoothing_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    for f in [
        Filter::SkinSmoothing { radius: 8.0, smoothness: 40.0 },
        Filter::SkinSmoothing { radius: 3.0, smoothness: 0.0 },
        Filter::SkinSmoothing { radius: 30.0, smoothness: 100.0 },
    ] {
        check_float(&g, &f);
        check_preview_bar(&g, &f, false);
    }
}

#[test]
fn lens_blur_layer_mask_depth_is_bit_exact() {
    let Some(g) = gpu() else { return };
    let base = crate::engine::selection::MaskGrid::from_selection(&Selection::ellipse(RectI::new(200, 100, 800, 600)), 1013, 777).feather(60.0);
    let mips = crate::engine::masks::build_mask_mips(&base);
    let m = Arc::new(blur::DepthMask::new(&base, &mips));
    let o = blur::LensOpts { depth: blur::Depth::Mask(m), focal: 255.0, blades: 8, curvature: 50.0, ..Default::default() };
    let f = lens(25.0, 30.0, o);
    check_exact(&g, &f);
    check_preview_bar(&g, &f, true);
}

/// Which path big commits take (diagnostics for `--bench-gpu 16384`).
#[test]
#[ignore]
fn debug_big_paths() {
    let Some(g) = gpu() else { return };
    let side: u32 = std::env::var("PF_SIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(16384);
    let img = crate::engine::ops::procedural(side, side);
    let only = std::env::var("PF_ONLY").unwrap_or_default();
    for f in crate::bench_gpu::gpu_filters().into_iter().filter(|f| supports_here(f) && format!("{f:?}").contains(&only)) {
        let t = std::time::Instant::now();
        let (_, path) = on_gpu(&g, &img, &f, None, &RunOpts::default());
        println!("{f:?}: {path:?} {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    }
}

fn supports_here(f: &Filter) -> bool {
    crate::engine::gpu::kernels::lensblur::supports(f)
}

/// Whole-image resident runs (diagnostics for `--bench-gpu 16384`).
#[test]
#[ignore]
fn debug_resident() {
    let Some(g) = gpu() else { return };
    let side: u32 = std::env::var("PF_SIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(16384);
    let img = crate::engine::ops::procedural(side, side);
    let only = std::env::var("PF_ONLY").unwrap_or_default();
    for f in crate::bench_gpu::gpu_filters().into_iter().filter(|f| supports_here(f) && format!("{f:?}").contains(&only)) {
        let k = crate::engine::gpu::kernels::plan(&g, &f, 0, crate::engine::gpu::kernels::Target::Commit).unwrap();
        let r = crate::engine::gpu::run::resident_ms(&g, &img, &f, k.as_ref(), 1);
        println!("{f:?}: {r:?} failed {:?}", g.failure());
    }
}
