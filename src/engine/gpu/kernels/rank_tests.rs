//! GPU = CPU for the rank family (Median, Dust & Scratches, Despeckle,
//! Maximum / Minimum, Smart Blur, Surface Blur, Reduce Noise).

use crate::engine::filters::Filter;
use crate::engine::gpu::tests::{check_exact, check_preview, gpu};

#[test]
fn maximum_minimum_are_bit_exact() {
    let Some(g) = gpu() else { return };
    for f in [
        Filter::Maximum { radius: 1.0 },
        Filter::Maximum { radius: 3.0 },
        Filter::Minimum { radius: 4.0 },
        Filter::Maximum { radius: 9.0 },
        Filter::Minimum { radius: 30.0 },
        Filter::Maximum { radius: 130.0 },
        Filter::Minimum { radius: 0.2 },
    ] {
        check_exact(&g, &f);
    }
    for f in [Filter::Maximum { radius: 12.0 }, Filter::Minimum { radius: 3.0 }] {
        check_preview(&g, &f);
    }
}

#[test]
fn small_medians_are_bit_exact() {
    let Some(g) = gpu() else { return };
    for f in [
        Filter::Median { radius: 1.0 },
        Filter::Median { radius: 2.0 },
        Filter::DustScratches { radius: 1.0, threshold: 6.0 },
        Filter::DustScratches { radius: 2.0, threshold: 0.0 },
        Filter::Despeckle,
    ] {
        check_exact(&g, &f);
    }
    for f in [Filter::Median { radius: 4.0 }, Filter::Despeckle, Filter::DustScratches { radius: 6.0, threshold: 4.0 }] {
        check_preview(&g, &f);
    }
}

#[test]
fn histogram_medians_are_bit_exact() {
    let Some(g) = gpu() else { return };
    for f in [
        Filter::Median { radius: 3.0 },
        Filter::Median { radius: 7.0 },
        Filter::Median { radius: 20.0 },
        Filter::Median { radius: 70.0 },
        Filter::DustScratches { radius: 4.0, threshold: 10.0 },
        Filter::DustScratches { radius: 12.0, threshold: 0.0 },
    ] {
        check_exact(&g, &f);
    }
}

/// Resident kernel timings at 4096² (PF_RANK_BENCH="median:3,surface:20:15,...":
/// name, radius, threshold).
#[test]
#[ignore]
fn time_rank_kernels() {
    use crate::engine::gpu::{kernels, run};
    let Some(g) = gpu() else { return };
    let img = crate::engine::ops::procedural(4096, 4096);
    let spec = std::env::var("PF_RANK_BENCH").unwrap_or_else(|_| "median:3,median:20".into());
    for item in spec.split(',') {
        let mut parts = item.split(':');
        let name = parts.next().unwrap();
        let a: f32 = parts.next().unwrap_or("3").parse().unwrap();
        let t: f32 = parts.next().map_or(15.0, |v| v.parse().unwrap());
        let f = match name {
            "median" => Filter::Median { radius: a },
            "dust" => Filter::DustScratches { radius: a, threshold: 10.0 },
            "smart" => Filter::SmartBlur { radius: a, threshold: t },
            "surface" => Filter::SurfaceBlur { radius: a, threshold: t },
            "noise" => Filter::ReduceNoise { strength: a },
            "max" => Filter::Maximum { radius: a },
            _ => Filter::Despeckle,
        };
        let k = kernels::plan(&g, &f, 0, kernels::Target::Commit).expect("kernel");
        let (wall, ts) = run::resident_ms(&g, &img, &f, k.as_ref(), 3).unwrap();
        println!("{item}: wall {wall:.1} ms, kernel {:.1} ms", ts.unwrap_or(0.0));
    }
}

#[test]
fn smart_blur_is_bit_exact() {
    let Some(g) = gpu() else { return };
    for f in [
        Filter::SmartBlur { radius: 3.0, threshold: 25.0 },
        Filter::SmartBlur { radius: 5.0, threshold: 0.4 },
        Filter::SmartBlur { radius: 30.0, threshold: 100.0 },
        Filter::SmartBlur { radius: 12.0, threshold: 25.0 },
        Filter::SmartBlur { radius: 100.0, threshold: 50.0 },
    ] {
        check_exact(&g, &f);
    }
    check_preview(&g, &Filter::SmartBlur { radius: 9.0, threshold: 25.0 });
}

/// The float bar (max 1/255 per channel, mean ΔE2000 <= 0.2) on every case
/// `check_exact` uses: photos, edge cases, a selection, one-tile-row bands.
fn check_close_all(g: &std::sync::Arc<crate::engine::gpu::Gpu>, f: &Filter) {
    use crate::engine::gpu::PathTaken;
    use crate::engine::gpu::run::RunOpts;
    use crate::engine::gpu::tests::{cpu, edge_case, on_gpu, photo, report};
    use crate::engine::grid::RectI;
    use crate::engine::selection::Selection;
    let ell = Selection::ellipse(RectI::new(150, 90, 900, 640));
    let cases: Vec<(&str, crate::engine::grid::TileGrid, Option<&Selection>, RunOpts)> = vec![
        ("dog", photo("dog_grass"), None, RunOpts::default()),
        ("edge", edge_case(), None, RunOpts::default()),
        ("edge+sel", edge_case(), Some(&ell), RunOpts::default()),
        ("butterfly 1-row bands", photo("butterfly"), None, RunOpts { max_band_rows: Some(1), ..Default::default() }),
    ];
    for (name, img, sel, opts) in cases {
        let c = cpu(&img, f, sel);
        let (o, path) = on_gpu(g, &img, f, sel, &opts);
        assert_eq!(path, PathTaken::Gpu, "{f:?} on {name}");
        let (mx, de, _) = report(&format!("{f:?} {name}"), &o, &c);
        assert!(mx <= 1 && de <= 0.2, "{f:?} on {name}: max {mx}, ΔE {de:.4}");
    }
}

/// Previews at mip levels 1-3 within the float bar of `Filter::render`.
fn check_preview_close(g: &std::sync::Arc<crate::engine::gpu::Gpu>, f: &Filter) {
    use crate::engine::filters;
    use crate::engine::gpu::preview;
    use crate::engine::gpu::tests::edge_case;
    use crate::engine::grid::RectI;
    use crate::engine::px;
    let mut layer = crate::engine::document::Layer::new(1, "t", edge_case());
    layer.mips = crate::engine::mips::build_mips(&layer.grid);
    for (level, rect) in [(1u32, RectI::new(-3, 5, 400, 300)), (2, RectI::new(37, 0, 260, 197)), (3, RectI::new(0, 0, 127, 98))] {
        let (w, h) = (rect.w() as usize, rect.h() as usize);
        let mut c = vec![0u32; w * h];
        f.render(&filters::Src::layer(&layer), level, rect.x0, rect.y0, w, h, &mut c, &mut Vec::new());
        let o = preview::render(g, f, &layer, level, rect).unwrap_or_else(|e| panic!("{f:?} level {level}: {e}"));
        let mx = c.iter().zip(&o).map(|(a, b)| px::unpack(*a).iter().zip(px::unpack(*b)).map(|(x, y)| x.abs_diff(y)).max().unwrap()).max().unwrap();
        let rgb = |v: &[u32]| -> Vec<[u8; 3]> { v.iter().map(|&p| { let q = px::unpremultiply(p); [q[0], q[1], q[2]] }).collect() };
        let de = crate::engine::metrics::mean_delta_e(&rgb(&c), &rgb(&o), None);
        println!("{f:?} preview level {level}: max {mx}, mean ΔE {de:.4}");
        assert!(mx <= 1 && de <= 0.2, "{f:?} preview level {level}: max {mx}, ΔE {de:.4}");
    }
}

#[test]
fn slow_surface_blurs_stay_on_the_cpu() {
    let Some(g) = gpu() else { return };
    use crate::engine::gpu::kernels::{Target, plan};
    assert!(plan(&g, &Filter::SurfaceBlur { radius: 100.0, threshold: 5.0 }, 0, Target::Commit).is_none());
    assert!(plan(&g, &Filter::SurfaceBlur { radius: 100.0, threshold: 5.0 }, 2, Target::Preview).is_some());
    assert!(plan(&g, &Filter::SurfaceBlur { radius: 20.0, threshold: 15.0 }, 0, Target::Commit).is_some());
}

#[test]
fn surface_blur_and_reduce_noise_match_the_cpu() {
    let Some(g) = gpu() else { return };
    for f in [
        Filter::SurfaceBlur { radius: 5.0, threshold: 15.0 },
        Filter::SurfaceBlur { radius: 20.0, threshold: 15.0 },
        Filter::SurfaceBlur { radius: 40.0, threshold: 6.0 },
        Filter::SurfaceBlur { radius: 20.0, threshold: 2.0 },
        Filter::SurfaceBlur { radius: 3.0, threshold: 255.0 },
        Filter::ReduceNoise { strength: 5.0 },
        Filter::ReduceNoise { strength: 0.0 },
    ] {
        check_close_all(&g, &f);
    }
    for f in [Filter::SurfaceBlur { radius: 12.0, threshold: 20.0 }, Filter::ReduceNoise { strength: 8.0 }] {
        check_preview_close(&g, &f);
    }
}
