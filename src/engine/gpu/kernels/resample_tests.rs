//! GPU = CPU for Image Size (every resampling method), Super Zoom, the
//! affine transform resamplers, the mip-aware perspective sampler, Displace
//! and Lens Correction.

use std::sync::Arc;

use crate::engine::annot;
use crate::engine::crop::{self, Interp};
use crate::engine::document::Layer;
use crate::engine::filters::distort::{mat_mul, square_to_quad};
use crate::engine::gpu::resample as gr;
use crate::engine::gpu::tests::{check_resampled, diff_exact, diff_stats, edge_case, gpu, photo, report};
use crate::engine::gpu::{Gpu, PathTaken};
use crate::engine::grid::TileGrid;
use crate::engine::ops;
use crate::engine::resample::{Resample, resize_cpu};

fn layers() -> Vec<(&'static str, Layer)> {
    vec![("dog", Layer::new(1, "dog", photo("dog_grass"))), ("edge", Layer::new(2, "edge", edge_case()))]
}

fn close(name: &str, o: &TileGrid, c: &TileGrid) {
    let (mx, de, _) = report(name, o, c);
    assert!(mx <= 1 && de <= 0.2, "{name}: max diff {mx}, mean ΔE {de:.4}");
}

fn gpu_or_skip() -> Option<Arc<Gpu>> {
    let g = gpu();
    assert!(g.is_some() || std::env::var("PF_GPU").is_ok_and(|v| v == "off"), "no GPU adapter: {}", crate::engine::gpu::describe());
    g
}

/// Output sizes for a `w x h` source: odd enlargements, reductions (one
/// past the mip threshold), mixed axes.
fn sizes(w: u32, h: u32) -> Vec<(u32, u32)> {
    vec![(w * 237 / 100, h * 237 / 100), (w * 43 / 100, h * 43 / 100), (w / 5 + 1, h / 5 + 3), (w * 3 / 2, h / 3), (w + 1, h - 1), (97, 61)]
}

#[test]
fn image_size_methods_are_bit_exact() {
    let Some(g) = gpu_or_skip() else { return };
    for (name, l) in layers() {
        for (w, h) in sizes(l.grid.w, l.grid.h) {
            for m in Resample::ALL {
                let noise = if m.has_noise() { 35.0 } else { 0.0 };
                let c = resize_cpu(&l, w, h, m, noise);
                let (o, path) = gr::on(&g, |g| gr::resize(g, &l, w, h, m, noise), || unreachable!());
                assert_eq!(path, PathTaken::Gpu);
                if let Some(d) = diff_exact(&o, &c) {
                    panic!("{m:?} {name} -> {w}x{h}: GPU != CPU at {d}");
                }
            }
        }
    }
}

#[test]
fn preserve_details_noise_levels_are_bit_exact() {
    let Some(g) = gpu_or_skip() else { return };
    let l = Layer::new(1, "b", photo("butterfly"));
    for m in [Resample::PreserveDetails, Resample::PreserveDetails2] {
        for noise in [0.0, 60.0, 100.0] {
            let (w, h) = (l.grid.w * 2 + 3, l.grid.h * 2 - 5);
            let c = resize_cpu(&l, w, h, m, noise);
            let (o, path) = gr::on(&g, |g| gr::resize(g, &l, w, h, m, noise), || unreachable!());
            assert_eq!(path, PathTaken::Gpu);
            assert!(diff_exact(&o, &c).is_none(), "{m:?} noise {noise}");
        }
    }
}

#[test]
fn super_zoom_upscale_matches_the_cpu() {
    let Some(g) = gpu_or_skip() else { return };
    for (name, l) in layers() {
        for k in [2u32, 3] {
            let (w, h) = (l.grid.w * k, l.grid.h * k);
            let c = ops::upscale_bicubic_cpu(&l, w, h);
            let (o, path) = gr::on(&g, |g| gr::upscale_bicubic(g, &l, w, h), || unreachable!());
            assert_eq!(path, PathTaken::Gpu);
            close(&format!("Super Zoom {k}x {name}"), &o, &c);
        }
    }
}

/// Rotations about the centre (with the canvas grown to fit), a scale +
/// rotation, an exact right angle, a flip and a skew.
fn affines(w: u32, h: u32) -> Vec<(String, [f32; 9], u32, u32)> {
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    let mut v = Vec::new();
    for deg in [12.5f32, -33.0, 90.0, 177.0, 0.3] {
        let r = annot::rotation_about(deg, cx, cy);
        v.push((format!("rotate {deg}°"), r, w, h));
    }
    let s = [0.61, 0.0, 0.0, 0.0, 0.61, 0.0, 0.0, 0.0, 1.0];
    v.push(("scale 0.61 + rotate 20°".into(), mat_mul(&s, &annot::rotation_about(20.0, cx, cy)), w * 61 / 100 + 7, h * 61 / 100 + 3));
    let s2 = [1.73, 0.0, -11.0, 0.0, 1.73, 5.0, 0.0, 0.0, 1.0];
    v.push(("enlarge 1.73 + rotate -8°".into(), mat_mul(&s2, &annot::rotation_about(-8.0, cx, cy)), w * 173 / 100, h * 173 / 100));
    v.push(("flip".into(), [-1.0, 0.0, w as f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], w, h));
    v.push(("skew".into(), [1.0, 0.3, -40.0, 0.1, 1.0, 3.5, 0.0, 0.0, 1.0], w + 200, h + 100));
    v
}

#[test]
fn affine_transforms_match_the_cpu() {
    let Some(g) = gpu_or_skip() else { return };
    for (name, l) in layers() {
        for (what, m, w, h) in affines(l.grid.w, l.grid.h) {
            for i in Interp::ALL {
                let c = crop::warp_affine_cpu(&l.grid, &m, w, h, i);
                let (o, path) = gr::on(&g, |g| gr::warp_affine(g, &l.grid, &m, w, h, i), || unreachable!());
                assert_eq!(path, PathTaken::Gpu);
                close(&format!("{} {what} {name}", i.label()), &o, &c);
            }
        }
    }
}

#[test]
fn perspective_and_mip_transforms_match_the_cpu() {
    let Some(g) = gpu_or_skip() else { return };
    for (name, l) in layers() {
        let (w, h) = (l.grid.w as f32, l.grid.h as f32);
        let persp = {
            // Unit square -> quad, after source -> unit square.
            let q = square_to_quad([(40.0, 20.0), (w * 0.9, -30.0), (w * 1.05, h * 0.95), (-15.0, h * 0.8)]);
            mat_mul(&q, &[1.0 / w, 0.0, 0.0, 0.0, 1.0 / h, 0.0, 0.0, 0.0, 1.0])
        };
        let shrink = mat_mul(&[0.17, 0.0, 9.0, 0.0, 0.17, 4.0, 0.0, 0.0, 1.0], &annot::rotation_about(15.0, w * 0.5, h * 0.5));
        let strong = {
            let q = square_to_quad([(0.0, 0.0), (w, h * 0.3), (w, h * 0.7), (0.0, h)]);
            mat_mul(&q, &[1.0 / w, 0.0, 0.0, 0.0, 1.0 / h, 0.0, 0.0, 0.0, 1.0])
        };
        for (what, m, ow, oh) in [
            ("perspective", persp, l.grid.w + 60, l.grid.h + 10),
            ("shrink 0.17 + rotate", shrink, l.grid.w / 4, l.grid.h / 4),
            ("strong perspective (several mips)", strong, l.grid.w, l.grid.h),
            ("rotate 30° (mask path)", annot::rotation_about(30.0, w * 0.5, h * 0.5), l.grid.w, l.grid.h),
        ] {
            let c = annot::warp_grid_cpu(&l.grid, m, ow, oh);
            let (o, path) = gr::on(&g, |g| gr::warp_grid(g, &l.grid, &m, ow, oh), || unreachable!());
            assert_eq!(path, PathTaken::Gpu);
            check_resampled(&format!("{what} {name}"), &o, &c);
        }
    }
}

#[test]
fn a_lost_device_falls_back_to_identical_cpu_pixels() {
    if crate::engine::gpu::env_mode() == Some(crate::engine::gpu::Mode::Off) {
        return;
    }
    let Ok(g) = Gpu::new() else { return };
    let g = Arc::new(g);
    let l = Layer::new(1, "dog", photo("dog_grass"));
    g.fail("simulated device loss");
    g.device.destroy();
    let c = resize_cpu(&l, 1500, 1000, Resample::Lanczos, 0.0);
    let (o, path) = gr::on(&g, |g| gr::resize(g, &l, 1500, 1000, Resample::Lanczos, 0.0), || resize_cpu(&l, 1500, 1000, Resample::Lanczos, 0.0));
    assert!(matches!(path, PathTaken::Cpu(_)), "{path:?}");
    assert!(diff_exact(&o, &c).is_none());
    let m = annot::rotation_about(10.0, 500.0, 300.0);
    let c = crop::warp_affine_cpu(&l.grid, &m, 1024, 683, Interp::Bicubic);
    let (o, path) = gr::on(&g, |g| gr::warp_affine(g, &l.grid, &m, 1024, 683, Interp::Bicubic), || crop::warp_affine_cpu(&l.grid, &m, 1024, 683, Interp::Bicubic));
    assert!(matches!(path, PathTaken::Cpu(_)));
    assert!(diff_exact(&o, &c).is_none());
}

#[test]
fn public_entry_points_give_cpu_pixels() {
    // Whatever PF_GPU says, the public entries equal the CPU reference
    // (bit-exact methods) or stay within the float bars.
    let l = Layer::new(1, "dog", photo("dog_grass"));
    let o = ops::resize_with(&l, 1777, 1185, Resample::BicubicSharper, 0.0);
    assert!(diff_exact(&o, &resize_cpu(&l, 1777, 1185, Resample::BicubicSharper, 0.0)).is_none());
    let o = ops::resize(&l, 600, 400);
    assert!(diff_exact(&o, &ops::resize_bilinear_cpu(&l, 600, 400)).is_none());
    let m = annot::rotation_about(7.0, 512.0, 341.0);
    let (mx, de) = diff_stats(&crop::warp_affine(&l.grid, &m, 1024, 683, Interp::Lanczos), &crop::warp_affine_cpu(&l.grid, &m, 1024, 683, Interp::Lanczos));
    assert!(mx <= 1 && de <= 0.2);
}

fn distort_filters() -> Vec<crate::engine::filters::Filter> {
    use crate::engine::filters::Filter;
    use crate::engine::filters::extra::{DispMap, MapSource};
    // A synthetic map file (smooth red / green ramps with a bump).
    let (mw, mh) = (300u32, 211u32);
    let rg = (0..mw * mh)
        .map(|i| {
            let (x, y) = ((i % mw) as f32, (i / mw) as f32);
            [((x / mw as f32) * 255.0) as u8, (128.0 + 100.0 * ((x * 0.05).sin() * (y * 0.07).cos())) as u8]
        })
        .collect();
    let map = Arc::new(DispMap { w: mw, h: mh, rg });
    vec![
        Filter::Displace { hs: 20.0, vs: -35.0, map: MapSource::Layer },
        Filter::Displace { hs: 60.0, vs: 60.0, map: MapSource::Clouds },
        Filter::Displace { hs: 25.0, vs: 15.0, map: MapSource::File { map: map.clone(), tile: false, img_w: 1024.0, img_h: 683.0 } },
        Filter::Displace { hs: -40.0, vs: 30.0, map: MapSource::File { map, tile: true, img_w: 1024.0, img_h: 683.0 } },
        Filter::LensCorrection { distortion: 40.0, ca_rc: 30.0, ca_by: -25.0, vignette: -40.0, w: 1024.0, h: 683.0 },
        Filter::LensCorrection { distortion: -60.0, ca_rc: -10.0, ca_by: 10.0, vignette: 50.0, w: 1013.0, h: 777.0 },
    ]
}

#[test]
fn displace_and_lens_correction_match_the_cpu() {
    use crate::engine::gpu::run::RunOpts;
    use crate::engine::gpu::tests::{cpu, on_gpu};
    use crate::engine::grid::RectI;
    use crate::engine::selection::Selection;
    let Some(g) = gpu_or_skip() else { return };
    let ell = Selection::ellipse(RectI::new(150, 90, 900, 640));
    for f in distort_filters() {
        for (name, img, sel, opts) in [
            ("dog", photo("dog_grass"), None, RunOpts::default()),
            ("edge", edge_case(), None, RunOpts::default()),
            ("edge+sel", edge_case(), Some(&ell), RunOpts::default()),
            ("butterfly 1-row bands", photo("butterfly"), None, RunOpts { max_band_rows: Some(1), ..Default::default() }),
        ] {
            let c = cpu(&img, &f, sel);
            let (o, path) = on_gpu(&g, &img, &f, sel, &opts);
            assert_eq!(path, PathTaken::Gpu, "{f:?}");
            check_resampled(&format!("{} {name}", f.name()), &o, &c);
        }
    }
}

#[test]
fn displace_and_lens_correction_previews_match_the_cpu() {
    let Some(g) = gpu_or_skip() else { return };
    for f in distort_filters() {
        check_preview_resampled(&g, &f);
    }
}

/// GPU previews at display mip levels 1–3 against `Filter::render`, with
/// the resampled bar.
fn check_preview_resampled(g: &Arc<Gpu>, f: &crate::engine::filters::Filter) {
    use crate::engine::filters;
    use crate::engine::gpu::preview;
    use crate::engine::grid::RectI;
    let mut layer = Layer::new(1, "t", edge_case());
    layer.mips = crate::engine::mips::build_mips(&layer.grid);
    for (level, rect) in [(1u32, RectI::new(-3, 5, 400, 300)), (2, RectI::new(37, 0, 260, 197)), (3, RectI::new(0, 0, 127, 98))] {
        let (w, h) = (rect.w() as usize, rect.h() as usize);
        let mut c = vec![0u32; w * h];
        f.render(&filters::Src::layer(&layer), level, rect.x0, rect.y0, w, h, &mut c, &mut Vec::new());
        let o = preview::render(g, f, &layer, level, rect).unwrap_or_else(|e| panic!("{f:?} level {level}: {e}"));
        let grid = |v: &[u32]| {
            crate::engine::grid::build_grid(w as u32, h as u32, |tx, ty| {
                let mut t = vec![0u32; 256 * 256].into_boxed_slice();
                for y in 0..256usize {
                    for x in 0..256usize {
                        let (gx, gy) = (tx as usize * 256 + x, ty as usize * 256 + y);
                        if gx < w && gy < h {
                            t[y * 256 + x] = v[gy * w + gx];
                        }
                    }
                }
                crate::engine::grid::finish_tile(t)
            })
        };
        check_resampled(&format!("{} preview level {level}", f.name()), &grid(&o), &grid(&c));
    }
}

/// Edit ▸ Transform ▸ Warp on pixels (Warp Text's patch, inverted per pixel).
fn mesh_warps() -> Vec<crate::engine::filters::Filter> {
    use crate::engine::warp::{BoxWarp, Warp, WarpStyle};
    let mut v = Vec::new();
    for (style, bend, hd, vd, b) in [
        (WarpStyle::Arc, 0.5, 0.0, 0.0, (0.0, 0.0, 1024.0, 683.0)),
        (WarpStyle::Flag, 0.4, 0.2, -0.1, (100.0, 80.0, 900.0, 600.0)),
        (WarpStyle::Bulge, -0.6, 0.0, 0.0, (0.0, 0.0, 1013.0, 777.0)),
        (WarpStyle::Fisheye, 0.5, 0.0, 0.3, (50.0, 20.0, 950.0, 700.0)),
        (WarpStyle::Twist, 0.3, 0.0, 0.0, (0.0, 0.0, 1024.0, 683.0)),
    ] {
        let w = Warp { style, bend, h_distort: hd, v_distort: vd, vertical: false };
        v.push(crate::engine::filters::Filter::Warp { name: "Warp", warp: BoxWarp::new(&w, b).pixel_warp() });
    }
    v
}

#[test]
fn mesh_warps_match_the_cpu() {
    use crate::engine::gpu::run::RunOpts;
    use crate::engine::gpu::tests::{cpu, on_gpu};
    let Some(g) = gpu_or_skip() else { return };
    for f in mesh_warps() {
        for (name, img) in [("dog", photo("dog_grass")), ("edge", edge_case())] {
            let c = cpu(&img, &f, None);
            let (o, path) = on_gpu(&g, &img, &f, None, &RunOpts::default());
            assert_eq!(path, PathTaken::Gpu);
            check_resampled(&format!("{} {name}", &format!("{f:?}")[..60]), &o, &c);
        }
        check_preview_resampled(&g, &f);
    }
}
