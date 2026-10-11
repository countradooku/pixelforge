//! `pixelforge --compare-gaps <dir> GPU`: the GPU path and the CPU path
//! against Photoshop 2026 (outputs and timings from
//! `tools/photoshop_gaps.sh <dir> GPU`), on the exported scene at 16.7 MP
//! (fidelity and time) and 268 MP (time). Prints the README table.
//!
//! The GPU and CPU fidelity must agree within 0.002 SSIM: the GPU computes
//! the CPU's pixels (exactly, or within 1/255).

use std::path::Path;
use std::time::Instant;

use crate::engine::filters::catalog::{BuildCtx, Kind};
use crate::engine::filters::gallery::G;
use crate::engine::filters::{Filter, extra};
use crate::engine::gpu;
use crate::engine::grid::TileGrid;

/// Photoshop's timing for `key` from timings_GPU.txt.
fn ps_ms(dir: &Path, key: &str) -> Option<f64> {
    let t = std::fs::read_to_string(dir.join("timings_GPU.txt")).ok()?;
    t.lines().find_map(|l| {
        let (k, v) = l.split_once('\t')?;
        (k == key).then(|| v.trim().parse().ok()).flatten()
    })
}

/// The operations of the jsx's GPU block, as PixelForge filters with the
/// same settings.
fn ops(side: u32) -> Vec<(&'static str, &'static str, Filter)> {
    let g = crate::engine::grid::RectI::new(0, 0, side as i32, side as i32);
    let ctx = BuildCtx { fg: [0, 0, 0], bg: [255, 255, 255], area: g, stats: None, source: None, disp_map: None, doc_w: side, doc_h: side, layer_mask: None };
    let build = |k: Kind, v: &[f32]| {
        let e = k.entry();
        let mut vals: Vec<f32> = e.params.iter().map(|p| p.default).collect();
        vals[..v.len()].copy_from_slice(v);
        k.build(&vals, e.flag.is_some_and(|f| f.1), &ctx)
    };
    let raw = extra::Raw {
        exposure: 0.35,
        contrast: 18.0,
        highlights: -40.0,
        shadows: 35.0,
        whites: 10.0,
        blacks: -12.0,
        clarity: 30.0,
        dehaze: 20.0,
        vibrance: 20.0,
        saturation: 5.0,
        sharpen: 40.0,
        noise_reduction: 30.0,
        temperature: 0.0,
        tint: 0.0,
        ..crate::bench_gpu::raw_heavy(side as f32)
    };
    vec![
        ("gaussian", "Gaussian Blur r=20", build(Kind::GaussianBlur, &[20.0])),
        ("surface", "Surface Blur r=5 t=15", build(Kind::SurfaceBlur, &[5.0, 15.0])),
        ("median", "Median r=3", build(Kind::Median, &[3.0])),
        ("unsharp", "Unsharp Mask 100 % r=1.5", build(Kind::UnsharpMask, &[100.0, 1.5, 0.0])),
        ("smartsharpen", "Smart Sharpen 100 % r=1.5", build(Kind::SmartSharpen, &[100.0, 1.5, 10.0])),
        ("lensblur", "Lens Blur (defaults)", Kind::LensBlur.default_filter(&ctx)),
        ("oilpaint", "Oil Paint (defaults)", Kind::OilPaint.default_filter(&ctx)),
        ("cameraraw", "Camera Raw Filter (heavy)", Filter::CameraRaw(raw)),
        ("cutout", "Filter Gallery: Cutout", Kind::Gallery(G::Cutout).default_filter(&ctx)),
        ("posteredges", "Filter Gallery: Poster Edges", Kind::Gallery(G::PosterEdges).default_filter(&ctx)),
        ("neonglow", "Filter Gallery: Neon Glow", Kind::Gallery(G::NeonGlow).default_filter(&ctx)),
    ]
}

fn rgb(g: &TileGrid) -> Vec<[u8; 3]> {
    let (w, h) = (g.w as usize, g.h as usize);
    let mut b = vec![0u32; w * h];
    g.read_region(0, 0, w, h, &mut b);
    b.iter()
        .map(|&p| {
            let c = crate::engine::px::unpremultiply(p);
            [c[0], c[1], c[2]]
        })
        .collect()
}

fn best<T>(reps: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut out = None;
    let mut ms = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let r = f();
        ms = ms.min(t.elapsed().as_secs_f64() * 1e3);
        out = Some(r);
    }
    (out.unwrap(), ms)
}

pub fn run(dir: &Path) {
    gpu::set_blocking(true);
    let Some(g) = gpu::context(true) else {
        println!("no GPU: {}", gpu::describe());
        return;
    };
    let big = std::env::var_os("PF_CMP_NO_BIG").is_none();
    println!("\n## GPU — PixelForge GPU and CPU paths vs Photoshop 2026 ({})\n", gpu::describe());
    println!("Fidelity on the exported scene at 4096² (16.7 MP); times at 16.7 MP and 268 MP (16384²).\n");
    println!("| Operation | SSIM GPU | SSIM CPU | ΔE GPU | ΔE CPU | CPU 16.7 MP | GPU 16.7 MP | PS 16.7 MP | CPU 268 MP | GPU 268 MP | PS 268 MP |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    let small = crate::engine::ops::demo_scene(4096, 4096);
    let huge = big.then(|| crate::engine::ops::demo_scene(16384, 16384));
    let p = crate::engine::ops::Progress::default();
    let fmt = |v: Option<f64>| v.map_or("—".into(), |v| format!("{v:.0} ms"));
    let mut worst = 0.0f64;
    for (key, label, f) in ops(4096) {
        let (cpu_img, cpu_ms) = best(2, || crate::engine::ops::apply_filter_cpu(&small, &f, None, &p, None));
        let ((gpu_img, path), gpu_ms) = best(2, || gpu::apply_filter_on(&g, &small, &f, None, &p, &Default::default()));
        let on_gpu = path == gpu::PathTaken::Gpu;
        let (mut sg, mut sc, mut dg, mut dc) = (None, None, None, None);
        if let Ok(ps) = image::open(dir.join(format!("gpu_{key}.png"))) {
            let ps: Vec<[u8; 3]> = ps.to_rgb8().pixels().map(|q| [q[0], q[1], q[2]]).collect();
            let (a, b) = (rgb(&gpu_img), rgb(&cpu_img));
            if ps.len() == a.len() {
                sg = Some(crate::engine::metrics::ssim(&a, &ps, 4096, 4096, None));
                sc = Some(crate::engine::metrics::ssim(&b, &ps, 4096, 4096, None));
                dg = Some(crate::engine::metrics::mean_delta_e(&a, &ps, None));
                dc = Some(crate::engine::metrics::mean_delta_e(&b, &ps, None));
                worst = worst.max((sg.unwrap() - sc.unwrap()).abs());
            }
        }
        let (mut cbig, mut gbig) = (None, None);
        if let Some(hg) = &huge {
            let (_, fb) = ops(16384).into_iter().find(|o| o.0 == key).map(|o| (o.1, o.2)).unwrap();
            cbig = Some(best(1, || crate::engine::ops::apply_filter_cpu(hg, &fb, None, &p, None)).1);
            gbig = Some(best(1, || gpu::apply_filter_on(&g, hg, &fb, None, &p, &Default::default())).1);
        }
        let s = |v: Option<f64>| v.map_or("—".into(), |v| format!("{v:.3}"));
        let d = |v: Option<f64>| v.map_or("—".into(), |v| format!("{v:.2}"));
        println!(
            "| {label}{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            if on_gpu { "" } else { " (GPU declined)" },
            s(sg),
            s(sc),
            d(dg),
            d(dc),
            fmt(Some(cpu_ms)),
            fmt(Some(gpu_ms)),
            fmt(ps_ms(dir, &format!("GPU {key} 4096"))),
            fmt(cbig),
            fmt(gbig),
            fmt(ps_ms(dir, &format!("GPU {key} 16384"))),
        );
    }
    // Image Size ▸ Preserve Details (Reduce Noise 0, as the jsx scripts
    // it): 2048² → 4096² for fidelity and time, 8192² → 16384² for time.
    {
        use crate::engine::resample::{Resample, resize_cpu};
        let m = Resample::PreserveDetails;
        let src = crate::engine::document::Layer::new(1, "scene", crate::engine::ops::demo_scene(2048, 2048));
        let (cpu_img, cpu_ms) = best(2, || resize_cpu(&src, 4096, 4096, m, 0.0));
        let (gpu_img, gpu_ms) = best(2, || gpu::resample::resize(&g, &src, 4096, 4096, m, 0.0));
        let gpu_img = gpu_img.ok();
        let (mut sg, mut sc, mut dg, mut dc) = (None, None, None, None);
        if let (Ok(ps), Some(gi)) = (image::open(dir.join("gpu_preservedetails.png")), &gpu_img) {
            let ps: Vec<[u8; 3]> = ps.to_rgb8().pixels().map(|q| [q[0], q[1], q[2]]).collect();
            let (a, b) = (rgb(gi), rgb(&cpu_img));
            if ps.len() == a.len() {
                sg = Some(crate::engine::metrics::ssim(&a, &ps, 4096, 4096, None));
                sc = Some(crate::engine::metrics::ssim(&b, &ps, 4096, 4096, None));
                dg = Some(crate::engine::metrics::mean_delta_e(&a, &ps, None));
                dc = Some(crate::engine::metrics::mean_delta_e(&b, &ps, None));
                worst = worst.max((sg.unwrap() - sc.unwrap()).abs());
            }
        }
        let (mut cbig, mut gbig) = (None, None);
        if big {
            let hs = crate::engine::document::Layer::new(2, "scene", crate::engine::ops::demo_scene(8192, 8192));
            cbig = Some(best(1, || resize_cpu(&hs, 16384, 16384, m, 0.0)).1);
            gbig = Some(best(1, || gpu::resample::resize(&g, &hs, 16384, 16384, m, 0.0)).1);
        }
        let s = |v: Option<f64>| v.map_or("—".into(), |v| format!("{v:.3}"));
        let d = |v: Option<f64>| v.map_or("—".into(), |v| format!("{v:.2}"));
        println!(
            "| Image Size ▸ Preserve Details 2×{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            if gpu_img.is_some() { "" } else { " (GPU declined)" },
            s(sg),
            s(sc),
            d(dg),
            d(dc),
            fmt(Some(cpu_ms)),
            fmt(Some(gpu_ms)),
            fmt(ps_ms(dir, "GPU preservedetails 4096")),
            fmt(cbig),
            fmt(gbig),
            fmt(ps_ms(dir, "GPU preservedetails 16384")),
        );
    }
    println!("\nLargest |SSIM GPU − SSIM CPU|: {worst:.4} (bar 0.002).");
}
