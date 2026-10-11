//! `pixelforge --bench-gpu [side] [filter]`: every GPU kernel against its CPU
//! path, at 16.7 MP and 268 MP by default.
//!
//! Columns: CPU ms (the reference path), GPU ms (upload, kernel, readback and
//! new tiles), GPU-resident ms (kernel only, image already on the GPU), the
//! timestamp-query kernel time, the path Auto mode picks, and whether the
//! pick is the faster one (the budget: Auto within 10 % of the best path).

use std::sync::Arc;
use std::time::Instant;

use crate::engine::filters::{self, BlurFx, Filter};
use crate::engine::gpu::{self, Gpu, kernels, run};
use crate::engine::grid::TileGrid;
use crate::engine::ops::{self, Progress};

fn best_ms<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

/// The GPU-accelerated filters at representative settings.
pub fn gpu_filters() -> Vec<Filter> {
    let mut v = vec![
        Filter::Levels { black: 12.0, white: 230.0, gamma: 1.3 },
        Filter::Curves { pts: [0.0, 80.0, 128.0, 180.0, 255.0] },
        filters::hue_saturation(25.0, 30.0, -10.0),
        Filter::Threshold { level: 128.0 },
        Filter::Desaturate,
        Filter::AddNoise { amount: 25.0, mono: false },
        Filter::GaussianBlur { radius: 2.0 },
        Filter::GaussianBlur { radius: 20.0 },
        Filter::GaussianBlur { radius: 120.0 },
        Filter::BoxBlur { radius: 10.0 },
        Filter::Blur,
        Filter::UnsharpMask { amount: 100.0, radius: 1.5, threshold: 0.0 },
        Filter::SmartSharpen { amount: 100.0, radius: 1.5, noise: 5.0, opts: Default::default() },
        Filter::HighPass { radius: 10.0 },
        Filter::LensBlur { radius: 15.0, bloom: 30.0, opts: Default::default() },
        Filter::LensBlur { radius: 60.0, bloom: 30.0, opts: filters::blur::LensOpts { blades: 6, curvature: 20.0, threshold: 200.0, ..Default::default() } },
        Filter::LensBlur { radius: 30.0, bloom: 30.0, opts: filters::blur::LensOpts { blades: 5, depth: filters::blur::Depth::Transparency, focal: 128.0, noise: 4.0, ..Default::default() } },
        Filter::ShapeBlur { radius: 15.0, shape: 3 },
        Filter::ShapeBlur { radius: 100.0, shape: 4 },
        Filter::SmartSharpen {
            amount: 200.0,
            radius: 3.0,
            noise: 3.0,
            opts: filters::blur::SharpenOpts { remove: filters::blur::Remove::Lens, shadows: filters::blur::Fade { amount: 50.0, width: 50.0, radius: 10.0 }, ..Default::default() },
        },
        Filter::SmartSharpen { amount: 200.0, radius: 8.0, noise: 3.0, opts: filters::blur::SharpenOpts { remove: filters::blur::Remove::Motion, angle: 30.0, ..Default::default() } },
        Filter::MotionBlur { angle: 0.0, distance: 40.0 },
        Filter::MotionBlur { angle: 30.0, distance: 40.0 },
        Filter::MotionBlur { angle: 12.0, distance: 500.0 },
        Filter::ShadowsHighlights { shadows: 35.0, highlights: 0.0, radius: 30.0 },
        Filter::SkinSmoothing { radius: 8.0, smoothness: 40.0 },
        Filter::Warp { name: "Twirl", warp: crate::engine::filters::Warp::Twirl { angle: 200.0, cx: 0.0, cy: 0.0, radius: 0.0 } },
        Filter::Warp { name: "Polar Coordinates", warp: crate::engine::filters::Warp::Polar { to_polar: true, w: 0.0, h: 0.0 } },
        Filter::Warp { name: "Wave", warp: crate::engine::filters::Warp::Wave { wavelength: 120.0, amplitude: 9.0 } },
        Filter::CameraRaw(raw_heavy(0.0)),
        Filter::JpegCleanup { strength: 60.0 },
        // Blur Gallery (placed by `fit`) and Radial Blur.
        Filter::FieldBlur { pins: [[0.0, 0.0, 30.0], [0.0, 0.0, 0.0], [0.0, 0.0, 15.0]], fx: BlurFx::default() },
        Filter::IrisBlur { cx: 0.0, cy: 0.0, radius: 0.0, blur: 15.0, fx: BlurFx::default() },
        Filter::IrisBlur { cx: 0.0, cy: 0.0, radius: 0.0, blur: 15.0, fx: bench_fx() },
        Filter::TiltShift { cy: 0.0, band: 0.0, transition: 0.0, blur: 15.0, fx: BlurFx::default() },
        Filter::SpinBlur { cx: 0.0, cy: 0.0, rx: 0.0, ry: 0.0, amount: 15.0, fx: BlurFx::default() },
        Filter::SpinBlur { cx: 0.0, cy: 0.0, rx: 0.0, ry: 0.0, amount: 15.0, fx: bench_fx() },
        Filter::PathBlur { x0: 0.0, y0: 0.0, x1: 0.0, y1: 0.0, bend: 0.0, length: 60.0, fx: BlurFx::default() },
        Filter::RadialBlur { amount: 10.0, zoom: false, cx: 0.0, cy: 0.0 },
        Filter::RadialBlur { amount: 30.0, zoom: true, cx: 0.0, cy: 0.0 },
        Filter::Median { radius: 3.0 },
        Filter::Median { radius: 20.0 },
        Filter::DustScratches { radius: 4.0, threshold: 10.0 },
        Filter::Despeckle,
        Filter::SurfaceBlur { radius: 5.0, threshold: 15.0 },
        Filter::SurfaceBlur { radius: 20.0, threshold: 15.0 },
        Filter::SmartBlur { radius: 5.0, threshold: 25.0 },
        Filter::ReduceNoise { strength: 5.0 },
        Filter::Maximum { radius: 3.0 },
        Filter::Maximum { radius: 30.0 },
        // Stylize, Pixelate, Render (`kernels::stylize`, `kernels::oil`).
        Filter::FindEdges,
        Filter::Emboss { angle: 135.0, height: 3.0, amount: 100.0 },
        Filter::TraceContour { level: 128.0 },
        Filter::Diffuse { radius: 2.0 },
        Filter::Wind { length: 60.0, from_left: true },
        Filter::OilPaint { radius: 4.0, oil: Default::default() },
        Filter::OilPaint { radius: 4.0, oil: crate::engine::filters::stylize::Oil { stylization: 4.0, bristle: 3.0, shine: 2.0, ..Default::default() } },
        Filter::Facet,
        Filter::Mosaic { cell: 10.0 },
        Filter::Crystallize { cell: 10.0 },
        Filter::Pointillize { cell: 10.0, bg: [255, 255, 255] },
        Filter::ColorHalftone { radius: 8.0 },
        Filter::Fragment,
        Filter::Mezzotint,
        Filter::Extrude { size: 30.0, depth: 30.0, pyramids: false },
        Filter::Tiles { size: 128.0, offset: 10.0, fill: crate::engine::px::pack(255, 255, 255, 255) },
        Filter::Custom { k: [0., -1., 0., -1., 5., -1., 0., -1., 0.], scale: 1.0, offset: 0.0 },
        Filter::DeInterlace { odd: true, interpolate: true },
        Filter::Lighting(crate::engine::filters::extra::Light { kind: 1, x: 0.0, y: 0.0, radius: 0.0, angle: -45.0, cone: 35.0, intensity: 60.0, ambient: 20.0, bump: 10.0, color: [255, 255, 255] }),
        Filter::Clouds { c0: [0, 0, 0], c1: [255, 255, 255], scale: 220.0, difference: false },
        Filter::Clouds { c0: [0, 0, 0], c1: [255, 255, 255], scale: 220.0, difference: true },
        Filter::Fibers { c0: [0, 0, 0], c1: [255, 255, 255], variance: 16.0, strength: 4.0 },
        Filter::LensFlare { x: 0.0, y: 0.0, brightness: 100.0, size: 0.0 },
        bench_gradient(crate::engine::gradient::GradKind::Linear),
        bench_gradient(crate::engine::gradient::GradKind::Angle),
        Filter::Displace { hs: 20.0, vs: 20.0, map: crate::engine::filters::extra::MapSource::Layer },
        Filter::Displace { hs: 20.0, vs: 20.0, map: crate::engine::filters::extra::MapSource::Clouds },
        Filter::LensCorrection { distortion: 40.0, ca_rc: 30.0, ca_by: -25.0, vignette: -40.0, w: 0.0, h: 0.0 },
        // Edit ▸ Transform ▸ Warp (Arc 50 %), boxed to the image in `fit`.
        Filter::Warp { name: "Warp", warp: crate::engine::filters::Warp::Mesh { m: [0.0; 32], nodes: [0.0; 50], x0: 0.0, y0: 0.0, w: 1.0, h: 1.0 } },
    ];
    // Every Filter Gallery effect at its default settings.
    for g in crate::engine::filters::gallery::G::ALL {
        let mut p = [0f32; 3];
        for (k, d) in g.params().iter().enumerate().take(3) {
            p[k] = d.default;
        }
        v.push(Filter::Gallery { g, p, fg: [0, 0, 0], bg: [255, 255, 255], doc: (0, 0) });
    }
    v
}

/// Camera Raw's heavy preset: tone, clarity, dehaze, texture, sharpening,
/// noise reduction (sized to the image by `fit`).
pub fn raw_heavy(side: f32) -> crate::engine::filters::extra::Raw {
    crate::engine::filters::extra::Raw {
        temperature: 12.0,
        tint: -6.0,
        exposure: 0.35,
        contrast: 18.0,
        highlights: -40.0,
        shadows: 35.0,
        whites: 10.0,
        blacks: -12.0,
        texture: 25.0,
        clarity: 30.0,
        dehaze: 20.0,
        vibrance: 20.0,
        saturation: 5.0,
        curve: [0.0; 4],
        hue: [0.0; 8],
        sat: [0.0; 8],
        lum: [0.0; 8],
        shadow_hue: 0.0,
        shadow_sat: 0.0,
        highlight_hue: 0.0,
        highlight_sat: 0.0,
        balance: 0.0,
        sharpen: 40.0,
        noise_reduction: 30.0,
        vignette: 0.0,
        grain: 0.0,
        linear: [0.0; 3],
        radial: [0.0; 4],
        img_w: side,
        img_h: side,
    }
}


/// Blur Gallery effects for the benchmark: bokeh and color grain.
fn bench_fx() -> BlurFx {
    BlurFx { bokeh: 60.0, bokeh_color: 40.0, noise: 20.0, color: 50.0, ..Default::default() }
}

/// A gradient placeholder, placed on the benchmark image by `fit`.
fn bench_gradient(kind: crate::engine::gradient::GradKind) -> Filter {
    use crate::engine::gradient::{Gradient, GradientFill};
    let g = Gradient::presets([0, 0, 0], [255, 255, 255]).swap_remove(5);
    Filter::Gradient(Arc::new(GradientFill::new(g, kind, (0.0, 0.0), (1.0, 1.0), false, true, 1.0)))
}

/// Centers and sizes geometric filters on the benchmark image.
fn fit(f: Filter, width: u32, height: u32) -> Filter {
    use crate::engine::filters::Warp;
    let (w, h) = (width as f32, height as f32);
    let (cx, cy, short) = (w * 0.5, h * 0.5, w.min(h));
    match f {
        Filter::Warp { name, warp: Warp::Twirl { angle, .. } } => Filter::Warp { name, warp: Warp::Twirl { angle, cx, cy, radius: short * 0.5 } },
        Filter::Warp { name, warp: Warp::Polar { to_polar, .. } } => Filter::Warp { name, warp: Warp::Polar { to_polar, w, h } },
        Filter::CameraRaw(_) => Filter::CameraRaw(crate::engine::filters::extra::Raw { img_h: h, ..raw_heavy(w) }),
        Filter::FieldBlur { pins, fx } => Filter::FieldBlur { pins: [[0.25 * w, 0.3 * h, pins[0][2]], [0.5 * w, 0.6 * h, pins[1][2]], [0.8 * w, 0.3 * h, pins[2][2]]], fx },
        Filter::IrisBlur { blur, fx, .. } => Filter::IrisBlur { cx, cy, radius: 0.35 * short, blur, fx },
        Filter::TiltShift { blur, fx, .. } => Filter::TiltShift { cy, band: 0.08 * h, transition: 0.2 * h, blur, fx },
        Filter::SpinBlur { amount, fx, .. } => Filter::SpinBlur { cx, cy, rx: 0.25 * w, ry: 0.25 * h, amount, fx },
        Filter::PathBlur { length, fx, .. } => Filter::PathBlur { x0: 0.15 * w, y0: 0.7 * h, x1: 0.85 * w, y1: 0.3 * h, bend: short * 30.0 / 200.0, length, fx },
        Filter::RadialBlur { amount, zoom, .. } => Filter::RadialBlur { amount, zoom, cx, cy },
        Filter::Lighting(lt) => Filter::Lighting(crate::engine::filters::extra::Light { x: w * 0.3, y: h * 0.3, radius: short, ..lt }),
        Filter::LensFlare { brightness, .. } => Filter::LensFlare { x: w * 0.3, y: h * 0.3, brightness, size: short * 0.5 },
        Filter::Gradient(g) => {
            use crate::engine::gradient::GradientFill;
            Filter::Gradient(Arc::new(GradientFill::new(g.grad.clone(), g.kind, (w * 0.2, h * 0.3), (w * 0.7, h * 0.9), g.reverse, g.dither, g.opacity)))
        }
        Filter::LensCorrection { distortion, ca_rc, ca_by, vignette, .. } => Filter::LensCorrection { distortion, ca_rc, ca_by, vignette, w, h },
        Filter::Warp { name, warp: Warp::Mesh { .. } } => {
            use crate::engine::warp::{BoxWarp, Warp as TextWarp, WarpStyle};
            let tw = TextWarp { style: WarpStyle::Arc, bend: 0.5, h_distort: 0.0, v_distort: 0.0, vertical: false };
            Filter::Warp { name, warp: BoxWarp::new(&tw, (0.0, 0.0, w, h)).pixel_warp() }
        }
        Filter::Gallery { g, p, fg, bg, .. } => Filter::Gallery { g, p, fg, bg, doc: (width, height) },
        f => f,
    }
}

fn label(f: &Filter) -> String {
    if let Filter::CameraRaw(_) = f {
        return "Camera Raw Filter (heavy: clarity/dehaze/NR)".into();
    }
    if let Filter::Gradient(g) = f {
        return format!("Gradient {} ({})", g.kind.label(), g.grad.name);
    }
    let mut d = format!("{f:?}");
    // Blur Gallery effects: a tag instead of the long parameter list.
    let mut name = f.name().to_string();
    if let Some(i) = d.find(", fx: BlurFx") {
        if !d[i..].contains(&format!("{:?}", BlurFx::default())) {
            name.push_str(" +fx");
        }
        d = format!("{} }}", &d[..i]);
    }
    if let Filter::Gallery { g, .. } = f {
        return format!("Gallery: {}", g.title());
    }
    let params = d.split_once('{').map(|(_, p)| p.trim_end_matches('}').trim().to_string()).unwrap_or_default();
    let params = params.replace("opts: LensOpts { ", "").replace("opts: SharpenOpts { ", "").replace("Fade { ", "").replace("blades: ", "b").replace("curvature: ", "c").replace("threshold: ", "t").replace("depth: ", "").replace("remove: ", "").replace("shadows: amount: ", "fade ");
    let mut l = format!("{name} {}", params.replace("radius: ", "r=").replace(".0,", ",").replace(".0 ", " "));
    l.truncate(44);
    l
}

pub fn run(sizes: &[crate::bench_config::Dimensions], only: Option<&str>, write_priors: bool) {
    gpu::set_blocking(true);
    let Some(mut g) = gpu::context(true) else {
        println!("GPU benchmark: no GPU ({})", gpu::describe());
        return;
    };
    print_header(&g);
    // A fresh calibration (a cached model may have been measured on a busy
    // machine), saved for the app like Preferences ▸ Benchmark GPU does.
    gpu::cost::reset();
    gpu::cost::set_model(gpu::cost::calibrate(&g));
    gpu::cost::save();
    if let Some(m) = gpu::cost::model() {
        println!(
            "  cost model: fixed {:.2} ms · transfer {:.3} ns/B · GPU probe {:.2} ms (x{:.2}) · CPU probe {:.2} ms (x{:.2})",
            m.fixed_ms, m.transfer_ns_b, m.gpu_probe_ms, m.gpu_scale, m.cpu_probe_ms, m.cpu_scale
        );
    }
    let mut priors = std::collections::HashMap::new();
    let mut over = Vec::new();
    for &dimensions in sizes {
        let (width, height) = (dimensions.w, dimensions.h);
        let _resources = crate::bench_resources::Report::start();
        let mp = width as f64 * height as f64 / 1e6;
        println!("\n{width} × {height} ({mp:.1} MP)");
        println!("{:<44} {:>9} {:>9} {:>9} {:>9}  {:<5} {}", "kernel", "CPU ms", "GPU ms", "resident", "kernel", "auto", "budget");
        let img = ops::procedural(width, height);
        if only.is_none_or(|o| "layer style".contains(&o.to_lowercase())) {
            bench_styles(&g, &img, if width.max(height) > 8192 { 2 } else { 3 });
        }
        for f in gpu_filters() {
            let f = fit(f, width, height);
            if only.is_some_and(|o| !f.name().to_lowercase().contains(&o.to_lowercase())) {
                continue;
            }
            let reps = if width.max(height) > 8192 { 2 } else { 3 };
            let row = bench_one(&g, &img, &f, reps);
            if g.is_failed() {
                // Its GPU times are partly CPU fallbacks: say so, and go on
                // with a fresh context like the app would.
                println!("  ! GPU context failed during {}: {}", label(&f), g.failure().unwrap_or_default());
                match gpu::context(true) {
                    Some(n) => g = n,
                    None => {
                        println!("  ! no GPU any more: {}", gpu::describe());
                        return;
                    }
                }
            }
            let px = width as f64 * height as f64;
            let fixed = gpu::cost::model().map_or(0.0, |m| m.fixed_ms);
            priors.insert(gpu::cost::area_key(&f, 0, px as u64), gpu::cost::Rates {
                gpu_ns_px: (row.gpu - fixed).max(0.0) * 1e6 / px,
                cpu_ns_px: row.cpu * 1e6 / px,
            });
            if px as u64 > gpu::cost::LARGE_PX {
                // Banded runs: end-to-end GPU time (see `cost::LARGE_PX`).
                priors.insert(gpu::cost::large_key(&f, 0), gpu::cost::Rates { gpu_ns_px: (row.gpu - fixed).max(0.0) * 1e6 / px, cpu_ns_px: row.cpu * 1e6 / px });
            } else if let Some(r) = row.resident {
                priors.insert(gpu::cost::op_key(&f, 0), gpu::cost::Rates { gpu_ns_px: r * 1e6 / px, cpu_ns_px: row.cpu * 1e6 / px });
            }
            let ok = row.auto_ms <= row.cpu.min(row.gpu) * 1.1 + 0.5;
            if !ok {
                over.push(format!("{} @ {width}x{height}", label(&f)));
            }
            println!(
                "{:<44} {:>9.1} {:>9.1} {:>9} {:>9}  {:<5} {}",
                label(&f),
                row.cpu,
                row.gpu,
                row.resident.map_or("—".into(), |v| format!("{v:.1}")),
                row.kernel.map_or("—".into(), |v| format!("{v:.1}")),
                row.auto,
                if ok { "ok" } else { "OVER" }
            );
        }
        if only.is_none_or(|o| "select and mask".contains(&o.to_lowercase())) {
            let row = bench_select_mask(&g, &img, if width.max(height) > 8192 { 2 } else { 3 });
            let ok = row.auto_ms <= row.cpu.min(row.gpu) * 1.1 + 0.5;
            if !ok {
                over.push(format!("Select and Mask @ {width}x{height}"));
            }
            println!(
                "{:<44} {:>9.1} {:>9.1} {:>9} {:>9}  {:<5} {}",
                "Select and Mask (r16 smart, decon)",
                row.cpu,
                row.gpu,
                "—",
                row.kernel.map_or("—".into(), |v| format!("{v:.1}")),
                row.auto,
                if ok { "ok" } else { "OVER" }
            );
        }
        interactive_rows(&g, width, height, &img, only);
        if only.is_none_or(|o| "compositing".contains(&o.to_lowercase())) {
            compositing_rows(&g, width, height, &mut over);
        }
        if only.is_none_or(|o| "stats histogram".contains(&o.to_lowercase())) {
            let layer = crate::engine::document::Layer::new(1, "s", img.clone());
            let cpu = best_ms(5, || crate::engine::filters::stats(&layer).count);
            gpu::resident::clear(&g);
            let cold = best_ms(1, || gpu::stats::stats(&g, &layer).map(|s| s.count));
            let warm = best_ms(5, || gpu::stats::stats(&g, &layer).map(|s| s.count));
            println!("\nhistogram statistics (level {} of the layer)", gpu::stats::level_of(&layer));
            println!("{:<44} {cpu:>9.2} ms\n{:<44} {cold:>9.2} ms\n{:<44} {warm:>9.2} ms", "CPU", "GPU incl. upload", "GPU resident");
        }
        drop(img);
        if only.is_none_or(|o| ["heal", "poisson", "patchmatch", "caf", "mips", "retouch"].iter().any(|k| k.contains(&o.to_lowercase()) || o.to_lowercase().contains(k))) {
            retouch(&g, width, height, only.map(str::to_lowercase).as_deref(), &mut over);
        }
        resample_rows(&g, width, height, only, &mut over);
    }
    println!("\n{} over budget", over.len());
    for o in over {
        println!("  OVER: {o}");
    }
    if write_priors {
        let path = std::path::Path::new("src/engine/gpu/priors.txt");
        let head = "# op key (name + padding bucket [+ E<n>: [2^(n-1), 2^n) pixels, GPU end to end minus fixed cost; L: legacy large]) = GPU ns/px CPU ns/px, Apple M5 Pro.\n# Without an area suffix GPU rates are resident kernel only.\n# Written by `pixelforge --bench-gpu --write-priors`.\n";
        // Merge with what's there (other sizes / filters keep their rows).
        let mut all: std::collections::HashMap<String, gpu::cost::Rates> = std::fs::read_to_string(path)
            .map(|t| gpu::cost::Model::parse(&t.lines().map(|l| format!("op {l}\n")).collect::<String>()).ops)
            .unwrap_or_default();
        all.extend(priors);
        // The probes these rates go with (see `cost::ref_probes`).
        let probes = gpu::cost::model().map_or(String::new(), |m| format!("@ref={:.3} {:.3}\n", m.gpu_probe_ms, m.cpu_probe_ms));
        match std::fs::write(path, format!("{head}{probes}{}", gpu::cost::rates_to_text(&all))) {
            Ok(()) => println!("wrote {} ({} rows)", path.display(), all.len()),
            Err(e) => println!("can't write {}: {e}", path.display()),
        }
    }
}

/// Layer Style rows: the image inside an anti-aliased ellipse with Drop
/// Shadow + Stroke + Bevel & Emboss (Photoshop's defaults). CPU = the
/// compositor's tiled path (`flatten_layers`, every core); GPU = the whole
/// padded layer through `gpu::styles` (upload, passes, readback);
/// resident = the passes alone.
fn bench_styles(g: &Arc<Gpu>, img: &TileGrid, reps: usize) {
    use crate::engine::styles::{LayerStyle, Plan, layer_bounds};
    let (w, h) = (img.w, img.h);
    let sel = crate::engine::selection::Selection::ellipse(crate::engine::grid::RectI::new(w as i32 / 10, h as i32 / 10, w as i32 * 9 / 10, h as i32 * 9 / 10));
    let shape = ops::clear(img, Some(&crate::engine::selection::Selection { inverted: true, ..sel }));
    let mut layer = crate::engine::document::Layer::new(1, "styled", shape);
    let mut st = LayerStyle::default();
    st.drop_shadow.enabled = true;
    st.stroke.enabled = true;
    st.bevel.enabled = true;
    layer.style = Some(Arc::new(st.clone()));
    let cpu = best_ms(reps, || crate::engine::composite::flatten_layers(std::slice::from_ref(&layer), w, h));
    let plan = Plan::new(Some(&st), 1.0, 0, layer_bounds(&layer));
    let pad = plan.pad as u32;
    let (pw, ph) = (w + 2 * pad, h + 2 * pad);
    let mut src = vec![0u32; (pw * ph) as usize];
    layer.grid.read_region(-(pad as i32), -(pad as i32), pw as usize, ph as usize, &mut src);
    let origin = (-(pad as i32), -(pad as i32));
    let label = "Layer Style shadow+stroke+bevel";
    match gpu::styles::run(g, &plan, &src, pw, ph, origin) {
        Ok(_) => {
            let gpu_ms = best_ms(reps, || gpu::styles::run(g, &plan, &src, pw, ph, origin));
            let (res, kern) = gpu::styles::resident_ms(g, &plan, &src, pw, ph, reps).map_or((None, None), |(r, k)| (Some(r), k));
            println!(
                "{:<44} {:>9.1} {:>9.1} {:>9} {:>9}",
                label,
                cpu,
                gpu_ms,
                res.map_or("—".into(), |v| format!("{v:.1}")),
                kern.map_or("—".into(), |v| format!("{v:.1}"))
            );
        }
        Err(e) => println!("{label:<44} {cpu:>9.1}  GPU: {e}"),
    }
}

pub fn print_header(g: &Arc<Gpu>) {
    println!("GPU benchmark — {}", gpu::describe());
    println!(
        "  backend {} · driver {} · features: f16 {} · subgroups {} ({}–{}) · timestamps {} · immediates {} · mappable primary {} · pipeline cache {}",
        g.info.backend,
        if g.info.driver.is_empty() { "—" } else { &g.info.driver },
        g.caps.f16,
        g.caps.subgroups,
        g.caps.subgroup_min,
        g.caps.subgroup_max,
        g.caps.timestamps,
        g.caps.immediates,
        g.caps.mappable_primary,
        g.caps.pipeline_cache
    );
    println!(
        "  limits: max buffer {} MB · max storage binding {} MB · workgroup storage {} KB · {} invocations · budget {} MB",
        g.limits.max_buffer_size >> 20,
        g.limits.max_storage_buffer_binding_size >> 20,
        g.limits.max_compute_workgroup_storage_size >> 10,
        g.limits.max_compute_invocations_per_workgroup,
        g.budget() >> 20
    );
    println!("  {} CPU threads", rayon::current_num_threads());
}

pub struct Row {
    pub cpu: f64,
    pub gpu: f64,
    pub resident: Option<f64>,
    pub kernel: Option<f64>,
    pub auto: &'static str,
    pub auto_ms: f64,
}

pub fn bench_one(g: &Arc<Gpu>, img: &TileGrid, f: &Filter, reps: usize) -> Row {
    let p = Progress::default();
    let cpu = best_ms(reps, || ops::apply_filter_cpu(img, f, None, &p, None));
    let opts = run::RunOpts::default();
    let gpu_ms = best_ms(reps, || gpu::apply_filter_on(g, img, f, None, &p, &opts));
    // The optional resident probe dispatches the entire image at once, unlike
    // the app's banded commit. Do not turn a valid large-image benchmark into
    // a watchdog test. Keep at least 4x headroom using the measured commit time.
    let resident_safe = !g.is_failed() && gpu_ms < gpu::exec::watchdog().as_secs_f64() * 250.0;
    if !resident_safe {
        println!("  resident probe omitted for {} (banded commit {gpu_ms:.1} ms); watchdog unchanged", label(f));
    }
    let (resident, kernel) = match resident_safe.then(|| kernels::plan(g, f, 0, kernels::Target::Commit)).flatten() {
        Some(k) => match run::resident_ms(g, img, f, k.as_ref(), reps) {
            Ok((r, k)) => (Some(r), k),
            Err(_) => (None, None),
        },
        None => (None, None),
    };
    // What Auto mode decides, from the priors and calibration (before this
    // run teaches it anything).
    let veto = gpu::cost::veto(g, f, img.w as u64 * img.h as u64);
    let (auto, auto_ms) = if veto.is_some() { ("CPU", cpu) } else { ("GPU", gpu_ms) };
    Row { cpu, gpu: gpu_ms, resident, kernel, auto, auto_ms }
}

// ----- interactive warps (Liquify, Perspective Warp) ---------------------------------------

fn row(name: &str, cpu: f64, gpu: f64, resident: Option<f64>, kernel: Option<f64>, auto: &str) {
    println!(
        "{:<44} {:>9.1} {:>9.1} {:>9} {:>9}  {:<5} {}",
        name,
        cpu,
        gpu,
        resident.map_or("—".into(), |v| format!("{v:.1}")),
        kernel.map_or("—".into(), |v| format!("{v:.1}")),
        auto,
        if auto == "GPU" && gpu > cpu * 1.1 + 0.5 || auto == "CPU" && cpu > gpu * 1.1 + 0.5 { "OVER" } else { "ok" }
    );
}

/// Liquify and Perspective Warp rows: their sessions aren't `Filter`s, so
/// they are timed here. Liquify: one large Forward Warp stroke (radius
/// side/8, eight mouse moves across the middle) — CPU vs GPU (field edits,
/// resample of the touched tiles, readback, field sync at the end), and the
/// resample alone of the stroke's tiles. Perspective Warp: a two-plane mesh
/// (straightened) over the whole layer — GPU with the source uploaded, and
/// resident. `PF_LIQ_R` overrides the Liquify brush radius.
fn interactive_rows(g: &Arc<Gpu>, width: u32, height: u32, img: &TileGrid, only: Option<&str>) {
    use crate::engine::liquify::{Backend, Brush, Mode, Session};
    use crate::engine::perspwarp::{self, Mesh, Straighten};
    let want = |n: &str| only.is_none_or(|o| n.to_lowercase().contains(&o.to_lowercase()));
    let (sx, sy) = (width as f32, height as f32);
    let reps = if width.max(height) > 8192 { 2 } else { 3 };
    if want("liquify") {
        let b = Brush { radius: std::env::var("PF_LIQ_R").ok().and_then(|v| v.parse().ok()).unwrap_or(sx.min(sy) / 8.0), pressure: 0.8, rate: 0.7 };
        let pts: Vec<(f32, f32)> = (0..=8).map(|k| (sx * (0.3 + 0.05 * k as f32), sy * (0.5 + 0.01 * k as f32))).collect();
        let stroke = |backend: Backend| -> (f64, Option<f64>, f64, Session, TileGrid) {
            let mut grid = img.clone();
            let mut ses = Session::new(1, 0, &grid);
            ses.set_backend(backend);
            ses.set_gpu_timed(true);
            let t = Instant::now();
            let mut kernel: Option<f64> = None;
            let mut waited = 0.0;
            ses.begin_stroke(pts[0].0, pts[0].1);
            for &(x, y) in &pts[1..] {
                ses.stroke_to(Mode::ForwardWarp, x, y, &b, &mut grid, false);
                if let Some(st) = ses.gpu_stats() {
                    waited += st.wait_ms;
                    if let Some(k) = st.kernel_ms {
                        kernel = Some(kernel.unwrap_or(0.0) + k);
                    }
                }
            }
            ses.end_stroke();
            (t.elapsed().as_secs_f64() * 1e3, kernel, waited, ses, grid)
        };
        // Warm-up: pipelines, the resident source.
        let _ = stroke(Backend::Gpu);
        let mut cpu = f64::MAX;
        let mut gpu_ms = f64::MAX;
        let mut kern = None::<f64>;
        let mut res = f64::MAX;
        let mut last = None;
        for _ in 0..reps {
            cpu = cpu.min(stroke(Backend::Cpu).0);
            let (ms, k, waited, ses, _) = stroke(Backend::Gpu);
            res = res.min(waited);
            if !ses.on_gpu() {
                println!("Liquify: GPU not used ({:?})", ses.gpu_off());
            }
            gpu_ms = gpu_ms.min(ms);
            if let Some(k) = k {
                kern = Some(kern.map_or(k, |b: f64| b.min(k)));
            }
            last = Some(ses);
        }
        let auto = if b.radius * 2.0 >= crate::engine::liquify::AUTO_GPU_MIN_DIAMETER { "GPU" } else { "CPU" };
        row(&format!("Liquify stroke r={} (8 moves, {} dabs)", b.radius, 8 * (0.05 * sx / (b.radius * 0.25)).ceil() as usize), cpu, gpu_ms, Some(res), kern, auto);
        // The resample alone, over the tiles the stroke touched.
        if let Some(mut ses) = last {
            ses.sync_field();
            let tiles = ses.field_tiles();
            let cpu_r = best_ms(reps, || ses.resample_cpu(&tiles));
            let mut res_r = f64::MAX;
            let mut k = None::<f64>;
            let gpu_r = best_ms(reps, || {
                let n = ses.resample_gpu(&tiles).map(|v| v.len()).unwrap_or(0);
                if let Some(st) = ses.gpu_stats() {
                    res_r = res_r.min(st.wait_ms);
                    if let Some(v) = st.kernel_ms {
                        k = Some(k.map_or(v, |b: f64| b.min(v)));
                    }
                }
                n
            });
            let mp = tiles.len() as f64 * 65536.0 / 1e6;
            row(&format!("Liquify resample ({} tiles, {mp:.1} MP)", tiles.len()), cpu_r, gpu_r, Some(res_r), k, "GPU");
        }
    }
    if want("perspective") {
        let mut m = Mesh::default();
        m.add_quad([(sx * 0.10, sy * 0.12), (sx * 0.45, sy * 0.18), (sx * 0.47, sy * 0.80), (sx * 0.08, sy * 0.88)], 4.0);
        m.add_quad([(sx * 0.45, sy * 0.18), (sx * 0.85, sy * 0.10), (sx * 0.88, sy * 0.90), (sx * 0.47, sy * 0.80)], 4.0);
        m.straighten(Straighten::Both);
        let t = Instant::now();
        let table = m.table(img.w, img.h);
        let table_ms = t.elapsed().as_secs_f64() * 1e3;
        let cpu = best_ms(reps, || perspwarp::render_cpu(img, &table));
        let up = best_ms(reps, || {
            crate::engine::gpu::resident::clear(g);
            crate::engine::gpu::kernels::perspwarp::render(g, img, &table, false).map(|o| o.tiles.len()).unwrap_or(0)
        });
        let res = best_ms(reps, || crate::engine::gpu::kernels::perspwarp::render(g, img, &table, false).map(|o| o.tiles.len()).unwrap_or(0));
        let _ = crate::engine::gpu::kernels::perspwarp::render_opts(g, img, &table, false, None, true);
        let k = crate::engine::gpu::kernels::perspwarp::last_stats().kernel_ms;
        let auto = if img.w as u64 * img.h as u64 >= perspwarp::AUTO_GPU_MIN_PX { "GPU" } else { "CPU" };
        row(&format!("Perspective Warp 2 planes (+{table_ms:.0} ms mesh)"), cpu, up, Some(res), k, auto);
    }
}

/// A selection with image edges everywhere: bright areas inside an ellipse.
pub fn bench_mask(img: &TileGrid) -> crate::engine::selection::MaskGrid {
    use crate::engine::grid::{TS, build_grid, finish_tile};
    let (w, h) = (img.w, img.h);
    let (c, r) = (w as f32 * 0.5, w as f32 * 0.45);
    let mut g = TileGrid::new(w, h);
    g.tiles = build_grid(w, h, |tx, ty| {
        let mut t = vec![0u32; TS * TS].into_boxed_slice();
        let src = img.tiles[img.idx(tx as i32, ty as i32)].as_ref();
        for y in 0..TS {
            for x in 0..TS {
                let (gx, gy) = ((tx as usize * TS + x) as f32, (ty as usize * TS + y) as f32);
                let inside = (gx - c).hypot(gy - c) < r;
                let v = if inside && src.is_some_and(|s| crate::engine::refine::guide_of(s.read()[y * TS + x]) > 120) { 255 } else { 0 };
                t[y * TS + x] = crate::engine::px::pack(v, v, v, 255);
            }
        }
        finish_tile(t)
    })
    .tiles;
    crate::engine::selection::MaskGrid::from_gray(&g)
}

pub fn select_mask_params() -> crate::engine::refine::Params {
    crate::engine::refine::Params { radius: 16.0, smart: true, smooth: 20.0, feather: 3.0, contrast: 10.0, shift: 0.0, decontaminate: true, amount: 100.0, ..Default::default() }
}

pub fn bench_select_mask(g: &Arc<Gpu>, img: &TileGrid, reps: usize) -> Row {
    use crate::engine::gpu::kernels::refine as kr;
    let mask = bench_mask(img);
    let p = select_mask_params();
    let cpu = best_ms(reps, || crate::engine::refine::refine_cpu(&mask, img, &p));
    let gpu_ms = best_ms(reps, || kr::run(g, &mask, img, &p, None).ok());
    let kernel = kr::run_timed(g, &mask, img, &p, None, true).ok().and_then(|r| r.1);
    let work = crate::engine::refine::todo_tiles(&mask, p.pad()).iter().filter(|&&t| t).count() as u64 * 65536;
    gpu::cost::observe_named(kr::OP, kr::PRIOR, work, cpu, false);
    gpu::cost::observe_named(kr::OP, kr::PRIOR, work, gpu_ms, true);
    let veto = gpu::cost::veto_named(g, kr::OP, kr::PRIOR, work);
    let (auto, auto_ms) = if veto.is_some() { ("CPU", cpu) } else { ("GPU", gpu_ms) };
    Row { cpu, gpu: gpu_ms, resident: None, kernel, auto, auto_ms }
}

fn row_opt(label: &str, cpu: f64, gpu: Option<f64>, resident: Option<f64>, auto: &'static str, over: &mut Vec<String>) {
    let auto_ms = if auto == "GPU" { gpu.unwrap_or(cpu) } else { cpu };
    let ok = auto_ms <= cpu.min(gpu.unwrap_or(cpu)) * 1.1 + 0.5;
    if !ok {
        over.push(label.to_string());
    }
    println!(
        "{:<44} {:>9.1} {:>9} {:>9} {:>9}  {:<5} {}",
        label,
        cpu,
        gpu.map_or("—".into(), |v| format!("{v:.1}")),
        resident.map_or("—".into(), |v| format!("{v:.1}")),
        "—",
        auto,
        if ok { "ok" } else { "OVER" }
    );
}

/// Retouching rows: healing blends, Content-Aware Fill (PatchMatch +
/// blend), mip pyramids.
fn retouch(g: &Arc<Gpu>, width: u32, height: u32, only: Option<&str>, over: &mut Vec<String>) {
    use crate::engine::gpu::cost;
    use crate::engine::poisson::{self, Quality, Solver};
    let want = |k: &str| only.is_none_or(|o| k.contains(o) || o == "retouch");
    let mp = width as f64 * height as f64 / 1e6;
    println!("\nretouching, {width} × {height} ({mp:.1} MP)");
    let scene = ops::demo_scene(width, height);
    if width.max(height) <= 8192 && (want("heal") || want("poisson")) {
        for (d, q) in [(100usize, Quality::Fast), (300, Quality::Fast), (500, Quality::Fast), (500, Quality::Exact)] {
            let (w, h) = (d + 6, d + 6);
            let mut src = vec![0u32; w * h];
            let mut dst = vec![0u32; w * h];
            let (x0, y0) = (width as i32 / 2, (height as f32 * 0.72) as i32);
            scene.read_region(x0 - 900, y0 + 40, w, h, &mut src);
            scene.read_region(x0, y0, w, h, &mut dst);
            let mask: Vec<u8> = (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f32 - w as f32 / 2.0, (i / w) as f32 - h as f32 / 2.0);
                    ((i % w) > 0 && (i / w) > 0 && (i % w) + 1 < w && (i / w) + 1 < h && x * x + y * y < (d as f32 / 2.0).powi(2)) as u8
                })
                .collect();
            let mut solver = Solver::new();
            let cpu = best_ms(5, || poisson::blend_rgba_cpu(&mut solver, &src, &dst, &mask, w, h, 4, q));
            let gpu = best_ms(5, || poisson::blend_rgba_gpu(g, &src, &dst, &mask, w, h, 4, q));
            let op = if q == Quality::Exact { cost::OP_HEAL_EXACT } else { cost::OP_HEAL };
            let auto = if cost::veto_op(g, op, (w * h) as u64).is_some() { "CPU" } else { "GPU" };
            row_opt(&format!("healing blend {d} px{}", if q == Quality::Exact { ", Exact" } else { "" }), cpu, Some(gpu), None, auto, over);
        }
    }

    if width.max(height) <= 8192 && (want("caf") || want("patchmatch") || want("spot")) {
        use crate::engine::gpu::{Mode, set_prefs};
        use crate::engine::retouch;
        use crate::engine::selection::{MaskGrid, Selection};
        use crate::engine::grid::RectI;
        let (sx, sy) = (width as f32, height as f32);
        let ell = |x0: f32, y0: f32, d: i32| MaskGrid::from_selection(&Selection::ellipse(RectI::new(x0 as i32, y0 as i32, x0 as i32 + d, y0 as i32 + d)), width, height);
        let adapt = crate::engine::patchmatch::Adapt { color: crate::engine::patchmatch::ColorAdapt::Default, ..Default::default() };
        let spot = ell(sx * 0.5 - 250.0, sy * 0.85 - 250.0, 500);
        let spot3 = ell(sx * 0.5 - 150.0, sy * 0.85 - 150.0, 300);
        let big = ell(sx * 0.45, sy * 0.75, 1000);
        let samp = retouch::caf_sampling(&scene, &big, retouch::SamplingArea::Auto);
        let rot = crate::engine::patchmatch::Adapt { rotation: crate::engine::patchmatch::RotationAdapt::Medium, scale: true, mirror: true, ..adapt };
        type Job<'a> = Box<dyn Fn() + 'a>;
        let jobs: Vec<(&str, Job)> = vec![
            ("spot healing 300 px (PatchMatch + blend)", Box::new(|| {
                std::hint::black_box(retouch::spot_heal(&scene, &spot3, None, retouch::SpotMode::ContentAware));
            })),
            ("spot healing 500 px (PatchMatch + blend)", Box::new(|| {
                std::hint::black_box(retouch::spot_heal(&scene, &spot, None, retouch::SpotMode::ContentAware));
            })),
            ("CAF 1 MP hole (Auto sampling)", Box::new(|| {
                std::hint::black_box(retouch::content_aware_fill(&scene, &big, &samp, adapt));
            })),
            ("CAF 1 MP hole, rot+scale+mirror", Box::new(|| {
                std::hint::black_box(retouch::content_aware_fill(&scene, &big, &samp, rot));
            })),
        ];
        {
            // PatchMatch alone (no blend), engines forced: a 2048 x 2048
            // region of the scene with a 1 MP elliptical hole.
            use crate::engine::patchmatch::{self, Engine, Params};
            let (w, h) = (2048usize, 2048usize);
            let mut img = vec![0u32; w * h];
            scene.read_region((sx * 0.3) as i32, (sy * 0.5) as i32, w, h, &mut img);
            let hole: Vec<u8> = (0..w * h)
                .map(|i| {
                    let (x, y) = ((i % w) as f32 - 1024.0, (i / w) as f32 - 1024.0);
                    (x * x + y * y < 564.0 * 564.0) as u8
                })
                .collect();
            let p = Params::default();
            let cpu = best_ms(2, || patchmatch::complete_with(&img, &hole, w, h, None, &p, Engine::Cpu));
            let gpu = best_ms(2, || patchmatch::complete_with(&img, &hole, w, h, None, &p, Engine::Gpu(g.clone())));
            let auto = if cost::veto_op(g, cost::OP_PATCHMATCH, (w * h) as u64).is_some() { "CPU" } else { "GPU" };
            row_opt("PatchMatch only, 1 MP hole in 4 MP region", cpu, Some(gpu), None, auto, over);
        }
        for (label, job) in jobs {
            if !want("caf") && !want("spot") && !want("patchmatch") {
                continue;
            }
            let reps = 2;
            set_prefs(Mode::Off, &[]);
            let cpu = best_ms(reps, &job);
            set_prefs(Mode::Force, &[]);
            let gpu = best_ms(reps, &job);
            set_prefs(Mode::Auto, &[]);
            let auto_ms = best_ms(1, &job);
            let _ = auto_ms;
            let auto = match gpu::last_of(cost::OP_PATCHMATCH).map(|o| o.path) {
                Some(gpu::PathTaken::Gpu) => "GPU",
                _ => "CPU",
            };
            row_opt(label, cpu, Some(gpu), None, auto, over);
        }
    }

    if want("mips") {
        use crate::engine::gpu::{mipgen, resident};
        let reps = if width.max(height) > 8192 { 3 } else { 5 };
        let cpu = best_ms(reps, || crate::engine::mips::build_mips(&scene));
        // Incremental update after a whole-layer edit (every tile dirty).
        let mut m = crate::engine::mips::build_mips(&scene);
        let all = vec![scene.bounds()];
        let upd = best_ms(reps, || crate::engine::mips::update_mips(&scene, &mut m, &all));
        let gpu_up = best_ms(reps, || {
            resident::clear(g);
            mipgen::build_mips(g, &scene)
        });
        // With level 0 already resident (e.g. a GPU filter's result).
        let mut best_res = f64::MAX;
        for _ in 0..reps {
            resident::clear(g);
            if let Ok(im) = resident::upload(g, &scene) {
                resident::adopt(g, &scene, im);
                let t = Instant::now();
                let _ = std::hint::black_box(mipgen::build_mips(g, &scene));
                best_res = best_res.min(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        resident::clear(g);
        println!("{:<44} {:>9.1} {:>9.1} {:>9.1} {:>9}  {:<5} {}", "mip pyramid (build_mips)", cpu, gpu_up, best_res, "—", "CPU", "ok");
        println!("{:<44} {:>9.1} {:>9} {:>9} {:>9}  {:<5} {}", "  update_mips, every tile dirty", upd, "—", "—", "—", "CPU", "ok");
    }
}

/// Display compositing of a 13-layer document (blend modes, masks, a
/// clipped layer, an isolated group with an adjustment): every display tile
/// of the view at level 0 (a 2560 x 1600 screen region), CPU vs GPU. GPU
/// cold = every tile uploaded; warm = all tiles pooled; edit = one tile of
/// the top layer changed (what painting costs per frame, with the cached
/// composite of the layers below).
fn compositing_rows(g: &Arc<Gpu>, width: u32, height: u32, over: &mut Vec<String>) {
    use crate::engine::composite::{ComposeInput, DT, Scratch, display_tile};
    use crate::engine::document::Layer;
    use rayon::prelude::*;
    let photo = ops::demo_scene(width, height);
    let mut layers = vec![Layer::new(1, "bg", photo.clone())];
    // Keep the original workload stable when new blend modes are added, so
    // timings remain comparable with the preserved Phase 1 executable.
    let modes = &crate::engine::blend::BlendMode::ALL[..18];
    for i in 0..12u64 {
        let mut l = Layer::new(2 + i, "l", ops::procedural(width, height));
        l.blend = modes[(i as usize * 5) % modes.len()];
        l.opacity = 0.2 + 0.05 * i as f32;
        if i % 4 == 1 {
            l.mask = Some(crate::engine::layers::LayerMask::new(crate::engine::selection::MaskGrid::from_selection(
                &crate::engine::selection::Selection::ellipse(crate::engine::grid::RectI::new(0, 0, width as i32 / 2, height as i32)),
                width,
                height,
            )));
        }
        layers.push(l);
    }
    let (cols, rows) = ((2560 / 256).min(width.div_ceil(256) as i32), (1600 / 256).min(height.div_ceil(256) as i32));
    let tiles: Vec<(i32, i32)> = (0..rows).flat_map(|y| (0..cols).map(move |x| (x, y))).collect();
    let view = Some((0, 0, cols, rows));
    let input = ComposeInput { layers: &layers, preview: None, extra: None };
    let cpu = best_ms(3, || {
        tiles.par_iter().map_init(Scratch::default, |scratch, &(tx, ty)| {
            let mut b = vec![0u8; DT * DT * 4];
            display_tile(&input, 0, tx, ty, &mut b, scratch);
            b[0]
        }).count()
    });
    gpu::tiles::clear(g);
    let cold = best_ms(1, || gpu::compose::display_textures(g, &input, 0, &tiles, view).map(|v| v.len()));
    let warm = best_ms(3, || gpu::compose::display_textures(g, &input, 0, &tiles, view).map(|v| v.len()));
    let mut edited = layers.clone();
    let mut k = 0u32;
    let edit = best_ms(5, || {
        k += 1;
        let top = edited.last_mut().unwrap();
        top.grid.tiles[0] = Some(std::sync::Arc::new(crate::engine::grid::Tile::solid(crate::engine::px::pack(k as u8, 9, 9, 255))));
        top.mips = crate::engine::mips::build_mips(&top.grid);
        let input = ComposeInput { layers: &edited, preview: None, extra: None };
        // Painting dirties one display tile.
        gpu::compose::display_textures(g, &input, 0, &tiles[..1.min(tiles.len())], view).map(|v| v.len())
    });
    let n = tiles.len();
    println!("\ncompositing 13 layers, {n} display tiles at level 0 (document {width} x {height})");
    for (name, ms) in [("CPU, all tiles", cpu), ("GPU cold (uploads), all tiles", cold), ("GPU warm, all tiles", warm), ("GPU after one-tile edit (1 tile)", edit)] {
        println!("{name:<44} {ms:>9.1} ms");
    }
    if warm > cpu {
        over.push(format!("compositing warm {warm:.1} ms > CPU {cpu:.1} ms"));
    }
}

// ----- non-filter resampling (Image Size, Super Zoom, transforms) ---------------------------

type Op<'a> = Box<dyn Fn(&Arc<Gpu>) -> gpu::exec::GResult<TileGrid> + 'a>;

/// Image Size (every method), Super Zoom and the transform resamplers with
/// a rectangular output: CPU, GPU (source uploaded each time), GPU with
/// the source already resident, and Auto's pick.
fn resample_rows(g: &Arc<Gpu>, width: u32, height: u32, only: Option<&str>, over: &mut Vec<String>) {
    use crate::engine::crop::{self, Interp};
    use crate::engine::document::Layer;
    use crate::engine::gpu::resample as gr;
    use crate::engine::resample::{Resample, resize_cpu};
    use crate::engine::{annot, filters::distort};
    // A filtered GPU run must not allocate two full images for an unrelated
    // suite. This matters at 150/268 MP, especially on unified-memory GPUs.
    if only.is_some_and(|o| {
        let o = o.to_lowercase();
        let resize = Resample::ALL.into_iter().any(|m| {
            [format!("Image Size {} → 60 %", m.label()), format!("Image Size {} 2×", m.label())].iter().any(|n| n.to_lowercase().contains(&o))
        });
        let rotate = Interp::ALL.into_iter().any(|m| format!("Rotate 12° {}", m.label()).to_lowercase().contains(&o));
        !resize && !rotate && !["super zoom upscale 2×", "perspective (mip bilinear)", "rotate 12° (mip bilinear, masks)"].iter().any(|n| n.contains(&o))
    }) {
        return;
    }
    let mp = width as f64 * height as f64 / 1e6;
    println!("\nresampling, {width} × {height} output ({mp:.1} MP)");
    println!("{:<44} {:>9} {:>9} {:>9}  {:<5} {}", "operation", "CPU ms", "GPU ms", "resident", "auto", "budget");
    let big = Layer::new(1, "big", ops::procedural(width, height));
    // Enlargements start from half the side; reductions go from `side` to 60 %.
    let half = Layer::new(2, "half", ops::procedural((width / 2).max(1), (height / 2).max(1)));
    let (rw, rh) = ((width * 3 / 5).max(1), (height * 3 / 5).max(1));
    let rot = annot::rotation_about(12.0, width as f32 * 0.5, height as f32 * 0.5);
    let persp = {
        let (sx, sy) = (width as f32, height as f32);
        let q = distort::square_to_quad([(0.04 * sx, 0.02 * sy), (0.9 * sx, -0.03 * sy), (1.05 * sx, 0.95 * sy), (-0.015 * sx, 0.8 * sy)]);
        distort::mat_mul(&q, &[1.0 / sx, 0.0, 0.0, 0.0, 1.0 / sy, 0.0, 0.0, 0.0, 1.0])
    };
    let mut rows: Vec<(String, String, u64, Box<dyn Fn() -> TileGrid + '_>, Op<'_>)> = Vec::new();
    for m in [Resample::Bilinear, Resample::BicubicSharper, Resample::Lanczos, Resample::NearestNeighbor] {
        let l = &big;
        rows.push((format!("Image Size {} → 60 %", m.label()), format!("Image Size ({})", m.label()), u64::from(rw) * u64::from(rh), Box::new(move || resize_cpu(l, rw, rh, m, 0.0)), Box::new(move |g| gr::resize(g, l, rw, rh, m, 0.0))));
    }
    for m in [Resample::Bilinear, Resample::Bicubic, Resample::BicubicSmoother, Resample::Lanczos, Resample::PreserveDetails, Resample::PreserveDetails2] {
        let l = &half;
        rows.push((format!("Image Size {} 2×", m.label()), format!("Image Size ({})", m.label()), u64::from(width) * u64::from(height), Box::new(move || resize_cpu(l, width, height, m, 30.0)), Box::new(move |g| gr::resize(g, l, width, height, m, 30.0))));
    }
    {
        let l = &half;
        rows.push(("Super Zoom upscale 2×".into(), "Super Zoom upscale".into(), u64::from(width) * u64::from(height), Box::new(move || ops::upscale_bicubic_cpu(l, width, height)), Box::new(move |g| gr::upscale_bicubic(g, l, width, height))));
    }
    for i in Interp::ALL {
        let (gr0, m) = (&big.grid, rot);
        rows.push((format!("Rotate 12° {}", i.label()), format!("Transform ({})", i.label()), u64::from(width) * u64::from(height), Box::new(move || crop::warp_affine_cpu(gr0, &m, width, height, i)), Box::new(move |g| gr::warp_affine(g, gr0, &m, width, height, i))));
    }
    {
        let (gr0, m) = (&big.grid, persp);
        rows.push(("Perspective (mip bilinear)".into(), "Transform (mip bilinear)".into(), u64::from(width) * u64::from(height), Box::new(move || annot::warp_grid_cpu(gr0, m, width, height)), Box::new(move |g| gr::warp_grid(g, gr0, &m, width, height))));
        let m2 = rot;
        rows.push(("Rotate 12° (mip bilinear, masks)".into(), "Transform (mip bilinear)".into(), u64::from(width) * u64::from(height), Box::new(move || annot::warp_grid_cpu(gr0, m2, width, height)), Box::new(move |g| gr::warp_grid(g, gr0, &m2, width, height))));
    }
    // The machine may be shared: more repetitions, best of.
    let reps = if width.max(height) > 8192 { 3 } else { 7 };
    for (name, op, px, cpu_f, gpu_f) in rows {
        if only.is_some_and(|o| !name.to_lowercase().contains(&o.to_lowercase())) {
            continue;
        }
        let cpu = best_ms(reps, &cpu_f);
        let gpu_ms = best_ms(reps, || {
            gpu::resident::clear(g);
            gpu_f(g).expect("GPU resample")
        });
        let resident = best_ms(reps, || gpu_f(g).expect("GPU resample"));
        gpu::resident::clear(g);
        let bytes = 8 * px;
        gpu::cost::observe_xop(&op, px, bytes, cpu, false);
        gpu::cost::observe_xop(&op, px, bytes, gpu_ms, true);
        let (auto, auto_ms) = if gpu::cost::veto_xop(g, &op, px, bytes).is_some() { ("CPU", cpu) } else { ("GPU", gpu_ms) };
        let ok = auto_ms <= cpu.min(gpu_ms) * 1.1 + 0.5;
        if !ok {
            over.push(format!("{name} @ {width}x{height}"));
        }
        println!(
            "{:<44} {:>9.1} {:>9.1} {:>9.1}  {:<5} {}   (ns/px: CPU {:.3}, GPU {:.3})",
            name,
            cpu,
            gpu_ms,
            resident,
            auto,
            if ok { "ok" } else { "OVER" },
            cpu * 1e6 / px as f64,
            gpu_ms * 1e6 / px as f64
        );
    }
}

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    #[test]
    fn rectangular_filters_use_height_for_vertical_coordinates() {
        for (w, h) in [(6000, 4000), (4000, 6000), (601, 399)] {
            let raw = fit(Filter::CameraRaw(raw_heavy(1.0)), w, h);
            let Filter::CameraRaw(raw) = raw else { panic!("expected Camera Raw") };
            assert_eq!((raw.img_w, raw.img_h), (w as f32, h as f32));
            let radial = fit(Filter::RadialBlur { amount: 10.0, zoom: true, cx: 0.0, cy: 0.0 }, w, h);
            let Filter::RadialBlur { cx, cy, .. } = radial else { panic!("expected Radial Blur") };
            assert_eq!((cx, cy), (w as f32 / 2.0, h as f32 / 2.0));
        }
    }

    #[test]
    fn rectangle_tier_filters_preserve_image_bounds_on_cpu() {
        let (w, h) = (301, 173);
        let grid = ops::procedural(w, h);
        let filter = fit(Filter::LensCorrection { distortion: 20.0, ca_rc: 0.0, ca_by: 0.0, vignette: 10.0, w: 0.0, h: 0.0 }, w, h);
        let out = ops::apply_filter_cpu(&grid, &filter, None, &Progress::default(), None);
        assert_eq!((out.w, out.h), (w, h));
        // Check that this is an exercised workload, not an empty result.
        assert!(out.tiles.iter().flatten().any(|t| t.read().iter().any(|p| *p != 0)));
    }
}
