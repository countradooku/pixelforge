//! Blur Gallery and Radial Blur kernels against the CPU.

use std::sync::Arc;

use crate::engine::filters::{self, BlurFx, Filter};
use crate::engine::gpu::run::RunOpts;
use crate::engine::gpu::tests::{check_resampled, cpu, diff_stats, edge_case, gpu, on_gpu, photo, report};
use crate::engine::gpu::{Gpu, PathTaken, preview};
use crate::engine::grid::{RectI, TileGrid};
use crate::engine::px;
use crate::engine::selection::Selection;

fn bokeh() -> BlurFx {
    BlurFx { bokeh: 80.0, bokeh_color: 50.0, range: [150.0, 255.0], ..Default::default() }
}

fn grain() -> BlurFx {
    BlurFx { noise: 30.0, ..Default::default() }
}

fn rich() -> BlurFx {
    BlurFx { bokeh: 100.0, bokeh_color: 100.0, range: [120.0, 230.0], noise: 40.0, gaussian: true, size: 6.0, roughness: 30.0, color: 70.0, highlights: 20.0 }
}

/// Field / Iris / Tilt-Shift placed on a `w x h` image.
fn variable(w: f32, h: f32) -> Vec<Filter> {
    let field = |fx| Filter::FieldBlur { pins: [[0.25 * w, 0.3 * h, 30.0], [0.5 * w, 0.6 * h, 0.0], [0.8 * w, 0.3 * h, 15.0]], fx };
    let tilt = |blur, fx| Filter::TiltShift { cy: 0.5 * h, band: 0.08 * h, transition: 0.2 * h, blur, fx };
    let iris = |blur, fx| Filter::IrisBlur { cx: 0.5 * w, cy: 0.5 * h, radius: 0.35 * w.min(h), blur, fx };
    vec![
        field(BlurFx::default()),
        field(rich()),
        tilt(15.0, BlurFx::default()),
        tilt(2.0, BlurFx::default()),
        tilt(0.3, BlurFx::default()),
        tilt(15.0, bokeh()),
        tilt(15.0, grain()),
        iris(15.0, BlurFx::default()),
        iris(60.0, rich()),
        // Tiles too small for the larger stack levels: per-pass fallback.
        iris(400.0, BlurFx::default()),
    ]
}

/// Spin / Path / Radial placed on a `w x h` image.
fn gathers(w: f32, h: f32) -> Vec<Filter> {
    let spin = |amount, fx| Filter::SpinBlur { cx: 0.5 * w, cy: 0.5 * h, rx: 0.3 * w, ry: 0.35 * h, amount, fx };
    let path = |length, fx| Filter::PathBlur { x0: 0.15 * w, y0: 0.7 * h, x1: 0.85 * w, y1: 0.3 * h, bend: w.max(h) * 30.0 / 200.0, length, fx };
    vec![
        Filter::RadialBlur { amount: 10.0, zoom: false, cx: 0.45 * w, cy: 0.55 * h },
        Filter::RadialBlur { amount: 30.0, zoom: true, cx: 0.45 * w, cy: 0.55 * h },
        spin(15.0, BlurFx::default()),
        spin(40.0, rich()),
        spin(15.0, bokeh()),
        path(60.0, BlurFx::default()),
        path(300.0, BlurFx::default()),
        path(60.0, rich()),
        path(150.0, grain()),
    ]
}

fn cases() -> Vec<(&'static str, TileGrid, Option<Selection>, RunOpts)> {
    let ell = Selection::ellipse(RectI::new(150, 90, 900, 640));
    vec![
        ("dog", photo("dog_grass"), None, RunOpts::default()),
        ("edge", edge_case(), None, RunOpts::default()),
        ("edge+sel", edge_case(), Some(ell), RunOpts::default()),
        ("butterfly 1-row bands", photo("butterfly"), None, RunOpts { max_band_rows: Some(1), ..Default::default() }),
    ]
}

/// Float pipeline bar: at most 1/255, mean ΔE2000 <= 0.2.
fn check_float(g: &Arc<Gpu>, mk: impl Fn(f32, f32) -> Vec<Filter>) {
    for (name, img, sel, opts) in cases() {
        for f in mk(img.w as f32, img.h as f32) {
            let c = cpu(&img, &f, sel.as_ref());
            let (o, path) = on_gpu(g, &img, &f, sel.as_ref(), &opts);
            assert_eq!(path, PathTaken::Gpu, "{} on {name}", f.name());
            let (mx, de, _) = report(&format!("{f:?} on {name}"), &o, &c);
            assert!(mx <= 1 && de <= 0.2, "{f:?} on {name}: max {mx}, ΔE {de:.4}");
        }
    }
}

#[test]
fn field_iris_tilt_shift_match_the_cpu() {
    let Some(g) = gpu() else { return };
    check_float(&g, variable);
}

#[test]
fn spin_path_radial_match_the_cpu() {
    let Some(g) = gpu() else { return };
    for (name, img, sel, opts) in cases() {
        for f in gathers(img.w as f32, img.h as f32) {
            let c = cpu(&img, &f, sel.as_ref());
            let (o, path) = on_gpu(&g, &img, &f, sel.as_ref(), &opts);
            assert_eq!(path, PathTaken::Gpu, "{} on {name}", f.name());
            check_resampled(&format!("{f:?} on {name}"), &o, &c);
        }
    }
}

#[test]
fn default_effects_are_the_plain_blur_on_the_gpu() {
    let Some(g) = gpu() else { return };
    let img = photo("dog_grass");
    let (w, h) = (img.w as f32, img.h as f32);
    // Effects with zero amounts take the plain path: identical pixels.
    let idle = BlurFx { bokeh_color: 60.0, range: [0.0, 80.0], gaussian: true, size: 9.0, color: 40.0, highlights: 10.0, ..Default::default() };
    for (a, b) in [
        (Filter::TiltShift { cy: 0.5 * h, band: 40.0, transition: 120.0, blur: 15.0, fx: BlurFx::default() }, Filter::TiltShift { cy: 0.5 * h, band: 40.0, transition: 120.0, blur: 15.0, fx: idle }),
        (Filter::SpinBlur { cx: 0.5 * w, cy: 0.5 * h, rx: 300.0, ry: 200.0, amount: 15.0, fx: BlurFx::default() }, Filter::SpinBlur { cx: 0.5 * w, cy: 0.5 * h, rx: 300.0, ry: 200.0, amount: 15.0, fx: idle }),
    ] {
        let (oa, _) = on_gpu(&g, &img, &a, None, &RunOpts::default());
        let (ob, _) = on_gpu(&g, &img, &b, None, &RunOpts::default());
        assert!(crate::engine::gpu::tests::diff_exact(&oa, &ob).is_none(), "{}", a.name());
        assert!(crate::engine::gpu::tests::diff_exact(&cpu(&img, &b, None), &cpu(&img, &a, None)).is_none());
    }
}

/// Previews at mip levels 1-3 against `Filter::render`: the float bar for
/// the variable blurs, the resampling bar for gathers.
fn check_preview_close(g: &Arc<Gpu>, f: &Filter, resampled: bool) {
    let mut layer = crate::engine::document::Layer::new(1, "t", edge_case());
    layer.mips = crate::engine::mips::build_mips(&layer.grid);
    for (level, rect) in [(1u32, RectI::new(-3, 5, 400, 300)), (2, RectI::new(37, 0, 260, 197)), (3, RectI::new(0, 0, 127, 98))] {
        let (w, h) = (rect.w() as usize, rect.h() as usize);
        let mut c = vec![0u32; w * h];
        f.render(&filters::Src::layer(&layer), level, rect.x0, rect.y0, w, h, &mut c, &mut Vec::new());
        let o = preview::render(g, f, &layer, level, rect).unwrap_or_else(|e| panic!("{f:?} level {level}: {e}"));
        let mut mx = 0u8;
        let mut over = 0usize;
        for (&a, &b) in o.iter().zip(&c) {
            let d = px::unpack(a).iter().zip(px::unpack(b)).map(|(p, q)| p.abs_diff(q)).max().unwrap();
            mx = mx.max(d);
            over += (d > 1) as usize;
        }
        let rgb = |v: &[u32]| -> Vec<[u8; 3]> {
            v.iter()
                .map(|&p| {
                    let c = px::unpremultiply(p);
                    [c[0], c[1], c[2]]
                })
                .collect()
        };
        let de = crate::engine::metrics::mean_delta_e(&rgb(&o), &rgb(&c), None);
        let ok = if resampled { mx <= 2 && over * 10_000 <= w * h && de <= 0.2 } else { mx <= 1 && de <= 0.2 };
        assert!(ok, "{f:?} preview level {level}: max {mx}, {over} px over 1, ΔE {de:.4}");
    }
}

#[test]
fn previews_match_the_cpu_at_mip_levels() {
    let Some(g) = gpu() else { return };
    let img = edge_case();
    let (w, h) = (img.w as f32, img.h as f32);
    for f in variable(w, h) {
        check_preview_close(&g, &f, false);
    }
    for f in gathers(w, h) {
        check_preview_close(&g, &f, true);
    }
}

#[test]
#[ignore]
fn debug_blurgallery() {
    let Some(g) = gpu() else { return };
    let img = photo("dog_grass");
    let (w, h) = (img.w as f32, img.h as f32);
    for f in variable(w, h).into_iter().chain(gathers(w, h)) {
        let c = cpu(&img, &f, None);
        let (o, _) = on_gpu(&g, &img, &f, None, &RunOpts::default());
        let (mx, de) = diff_stats(&o, &c);
        println!("{}: max {mx} ΔE {de:.4}", f.name());
    }
}
