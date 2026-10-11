//! GPU = CPU for the Stylize, Pixelate and Render families and the Gradient
//! fill: bit-exact where the CPU is integer-defined or hash-defined (and its
//! float roundings are reproduced), else within 1/255 and mean ΔE2000 0.2;
//! previews at mip levels 1–3 under the same bars.

use std::sync::Arc;

use crate::engine::filters::extra::Light;
use crate::engine::filters::stylize::Oil;
use crate::engine::filters::{Filter, Src};
use crate::engine::gpu::run::RunOpts;
use crate::engine::gpu::tests::*;
use crate::engine::gpu::{Gpu, PathTaken, preview};
use crate::engine::gradient::{GradKind, Gradient, GradientFill};
use crate::engine::grid::{RectI, TS, TileGrid};
use crate::engine::px;
use crate::engine::selection::Selection;

fn cases() -> Vec<(&'static str, TileGrid, Option<Selection>, RunOpts)> {
    let ell = Selection::ellipse(RectI::new(150, 90, 900, 640));
    vec![
        ("dog", photo("dog_grass"), None, RunOpts::default()),
        ("edge", edge_case(), None, RunOpts::default()),
        ("edge+sel", edge_case(), Some(ell), RunOpts::default()),
        ("butterfly 1-row bands", photo("butterfly"), None, RunOpts { max_band_rows: Some(1), ..Default::default() }),
    ]
}

/// Bit-exact on every case.
fn exact(g: &Arc<Gpu>, f: &Filter) {
    for (name, img, sel, opts) in cases() {
        let c = cpu(&img, f, sel.as_ref());
        let (o, path) = on_gpu(g, &img, f, sel.as_ref(), &opts);
        assert_eq!(path, PathTaken::Gpu, "{} on {name}: {path:?}", f.name());
        if let Some(d) = diff_exact(&o, &c) {
            panic!("{f:?} on {name}: GPU != CPU at {d}");
        }
    }
}

/// At most 1/255 per channel and mean ΔE2000 <= 0.2 on every case.
fn close(g: &Arc<Gpu>, f: &Filter) {
    for (name, img, sel, opts) in cases() {
        let c = cpu(&img, f, sel.as_ref());
        let (o, path) = on_gpu(g, &img, f, sel.as_ref(), &opts);
        assert_eq!(path, PathTaken::Gpu, "{} on {name}: {path:?}", f.name());
        let (mx, de, _) = report(&format!("{} {name}", f.name()), &o, &c);
        assert!(mx <= 1 && de <= 0.2, "{f:?} on {name}: max diff {mx}, mean ΔE {de:.4}");
    }
}

/// `check_preview` with the float bar (max 1/255, mean ΔE <= 0.2).
fn preview_close(g: &Arc<Gpu>, f: &Filter) {
    let mut layer = crate::engine::document::Layer::new(1, "t", edge_case());
    layer.mips = crate::engine::mips::build_mips(&layer.grid);
    for (level, rect) in [(1u32, RectI::new(-3, 5, 400, 300)), (2, RectI::new(37, 0, 260, 197)), (3, RectI::new(0, 0, 127, 98))] {
        let (w, h) = (rect.w() as usize, rect.h() as usize);
        let mut c = vec![0u32; w * h];
        f.render(&Src::layer(&layer), level, rect.x0, rect.y0, w, h, &mut c, &mut Vec::new());
        let o = preview::render(g, f, &layer, level, rect).unwrap_or_else(|e| panic!("{f:?} level {level}: {e}"));
        let to_grid = |v: &[u32]| {
            let mut gr = TileGrid::new(w as u32, h as u32);
            gr.tiles = crate::engine::grid::build_grid(w as u32, h as u32, |tx, ty| {
                let mut t = vec![0u32; TS * TS].into_boxed_slice();
                for y in 0..TS {
                    for x in 0..TS {
                        let (gx, gy) = (tx as usize * TS + x, ty as usize * TS + y);
                        if gx < w && gy < h {
                            t[y * TS + x] = v[gy * w + gx];
                        }
                    }
                }
                crate::engine::grid::finish_tile(t)
            })
            .tiles;
            gr
        };
        let (mx, de) = diff_stats(&to_grid(&o), &to_grid(&c));
        assert!(mx <= 1 && de <= 0.2, "{f:?} preview level {level}: max {mx}, ΔE {de:.4}");
    }
}

fn light(kind: u8) -> Filter {
    Filter::Lighting(Light { kind, x: 400.0, y: 300.0, radius: 700.0, angle: 30.0, cone: 35.0, intensity: 60.0, ambient: 20.0, bump: 10.0, color: [255, 240, 220] })
}

fn oil(radius: f32, o: Oil) -> Filter {
    Filter::OilPaint { radius, oil: o }
}

fn gradient(kind: GradKind, preset: usize, dither: bool, opacity: f32) -> Filter {
    let g = Gradient::presets([200, 30, 40], [20, 60, 230]).swap_remove(preset);
    Filter::Gradient(Arc::new(GradientFill::new(g, kind, (130.3, 211.7), (811.9, 523.1), false, dither, opacity)))
}

/// Integer- or hash-defined filters at test settings (bit-exact).
fn exact_filters() -> Vec<Filter> {
    let k_frac = [0.5, -1.25, 0.0, -1.0, 6.75, -1.0, 0.25, -1.5, 0.3];
    vec![
        Filter::FindEdges,
        Filter::Emboss { angle: 135.0, height: 3.0, amount: 100.0 },
        Filter::Emboss { angle: -37.0, height: 9.6, amount: 260.0 },
        Filter::TraceContour { level: 128.0 },
        Filter::TraceContour { level: 37.5 },
        Filter::Diffuse { radius: 2.0 },
        Filter::Diffuse { radius: 7.5 },
        Filter::Wind { length: 20.0, from_left: true },
        Filter::Wind { length: 60.0, from_left: false },
        Filter::Wind { length: 3.3, from_left: true },
        Filter::Fragment,
        Filter::Custom { k: [0., -1., 0., -1., 5., -1., 0., -1., 0.], scale: 1.0, offset: 0.0 },
        Filter::Custom { k: k_frac, scale: 3.0, offset: 10.0 },
        Filter::DeInterlace { odd: true, interpolate: true },
        Filter::DeInterlace { odd: false, interpolate: false },
        Filter::Mezzotint,
        Filter::Mosaic { cell: 8.0 },
        Filter::Mosaic { cell: 33.3 },
        Filter::Extrude { size: 30.0, depth: 30.0, pyramids: false },
        Filter::Extrude { size: 17.0, depth: 80.0, pyramids: true },
        Filter::Tiles { size: 64.0, offset: 10.0, fill: px::pack(255, 255, 255, 255) },
        Filter::Tiles { size: 101.7, offset: 25.0, fill: px::pack(10, 20, 30, 255) },
        Filter::Crystallize { cell: 10.0 },
        Filter::Crystallize { cell: 3.0 },
        Filter::Crystallize { cell: 40.0 },
        Filter::Facet,
        oil(4.0, Oil::default()),
        oil(1.0, Oil::default()),
        oil(15.0, Oil::default()),
        oil(30.0, Oil::default()),
    ]
}

fn gradients() -> Vec<Filter> {
    let mut v = Vec::new();
    for kind in GradKind::ALL {
        v.push(gradient(kind, 5, false, 1.0));
    }
    v.push(gradient(GradKind::Linear, 3, true, 1.0));
    v.push(gradient(GradKind::Radial, 1, true, 0.6));
    v.push(gradient(GradKind::Angle, 4, false, 0.8));
    v
}

/// Float pipelines (within 1/255).
fn close_filters() -> Vec<Filter> {
    vec![
        Filter::Pointillize { cell: 10.0, bg: [255, 255, 255] },
        Filter::Pointillize { cell: 27.0, bg: [20, 40, 60] },
        Filter::ColorHalftone { radius: 4.0 },
        Filter::ColorHalftone { radius: 9.0 },
        light(0),
        light(1),
        light(2),
        Filter::Clouds { c0: [10, 20, 30], c1: [250, 240, 230], scale: 220.0, difference: false },
        Filter::Clouds { c0: [200, 20, 30], c1: [25, 240, 130], scale: 220.0, difference: true },
        Filter::Fibers { c0: [10, 20, 30], c1: [250, 240, 230], variance: 16.0, strength: 4.0 },
        Filter::LensFlare { x: 300.0, y: 200.0, brightness: 100.0, size: 340.0 },
        oil(4.0, Oil { stylization: 4.0, ..Default::default() }),
        oil(6.0, Oil { stylization: 10.0, cleanliness: 2.0, scale: 0.6, bristle: 5.0, angle: 30.0, shine: 3.0 }),
        oil(3.0, Oil { bristle: 8.0, ..Default::default() }),
        oil(5.0, Oil { shine: 6.0, angle: 120.0, ..Default::default() }),
    ]
}

#[test]
fn stylize_pixelate_are_bit_exact() {
    let Some(g) = gpu() else { return };
    for f in exact_filters() {
        exact(&g, &f);
    }
}

#[test]
fn gradients_are_bit_exact() {
    let Some(g) = gpu() else { return };
    for f in gradients() {
        exact(&g, &f);
    }
}

#[test]
fn float_filters_match_within_one_level() {
    let Some(g) = gpu() else { return };
    for f in close_filters() {
        close(&g, &f);
    }
}

#[test]
fn previews_match_at_mip_levels() {
    let Some(g) = gpu() else { return };
    for f in exact_filters().into_iter().chain(gradients()) {
        check_preview(&g, &f);
    }
    for f in close_filters() {
        preview_close(&g, &f);
    }
}

/// Settings the GPU declines go to the CPU, identically.
#[test]
fn declined_settings_fall_back() {
    let Some(g) = gpu() else { return };
    let img = edge_case();
    let f = Filter::ColorHalftone { radius: 20.0 };
    let (o, path) = on_gpu(&g, &img, &f, None, &RunOpts::default());
    assert!(matches!(path, PathTaken::Cpu(_)), "{path:?}");
    assert!(diff_exact(&o, &cpu(&img, &f, None)).is_none());
    // But its preview runs on the GPU.
    preview_close(&g, &f);
}

/// Whether the CPU commit equals one whole-image render (block independence).
#[test]
#[ignore]
fn cpu_block_independence() {
    let g = photo("dog_grass");
    for f in exact_filters().into_iter().chain(close_filters()) {
        let c = cpu(&g, &f, None);
        let (tw, th) = (g.cols as usize * TS, g.rows as usize * TS);
        let mut out = vec![0u32; tw * th];
        f.render(&Src::single(&g), 0, 0, 0, tw, th, &mut out, &mut Vec::new());
        let n = (0..g.h as usize).flat_map(|y| (0..g.w as usize).map(move |x| (x, y))).filter(|&(x, y)| c.pixel(x as i32, y as i32) != out[y * tw + x]).count();
        println!("{:<20} {n} px differ", f.name());
    }
}

/// Many gradient placements (sub-pixel endpoints, reversed, every preset)
/// on a 1600 x 1100 grid: the LUT index must match everywhere.
#[test]
fn gradient_placements_are_bit_exact() {
    let Some(g) = gpu() else { return };
    let img = TileGrid::new(1600, 1100);
    let mut seed = 12345u32;
    let mut rnd = |m: f32| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16777216.0 * m
    };
    for k in 0..24 {
        let kind = GradKind::ALL[k % 5];
        let preset = Gradient::presets([200, 30, 40], [20, 60, 230]).swap_remove(k % 6);
        let p0 = (rnd(1600.0) - 200.0, rnd(1100.0) - 100.0);
        let p1 = (rnd(1600.0), rnd(1100.0));
        let f = Filter::Gradient(Arc::new(GradientFill::new(preset, kind, p0, p1, k % 3 == 0, k % 4 == 0, if k % 5 == 3 { 0.7 } else { 1.0 })));
        let c = cpu(&img, &f, None);
        let (o, path) = on_gpu(&g, &img, &f, None, &RunOpts::default());
        assert_eq!(path, PathTaken::Gpu);
        if let Some(d) = diff_exact(&o, &c) {
            panic!("gradient {k} ({kind:?}, {p0:?} -> {p1:?}): GPU != CPU at {d}");
        }
    }
}

/// Cell filters at many sizes on a larger procedural image.
#[test]
fn cell_filters_at_many_sizes() {
    let Some(g) = gpu() else { return };
    let img = crate::engine::ops::procedural(1700, 1300);
    let mut ex = Vec::new();
    for cell in [2.0f32, 3.0, 5.5, 12.0, 15.0, 21.0, 64.0, 120.0, 300.0] {
        ex.push(Filter::Crystallize { cell });
        ex.push(Filter::Mosaic { cell: cell * 0.9 });
        ex.push(Filter::Extrude { size: cell * 1.3, depth: 50.0, pyramids: cell > 10.0 });
    }
    for (size, offset) in [(37.0f32, 10.0f32), (90.5, 30.0), (300.0, 9.0)] {
        ex.push(Filter::Tiles { size, offset, fill: px::pack(0, 0, 0, 255) });
    }
    for f in &ex {
        let c = cpu(&img, f, None);
        let (o, path) = on_gpu(&g, &img, f, None, &RunOpts::default());
        assert_eq!(path, PathTaken::Gpu, "{f:?}");
        if let Some(d) = diff_exact(&o, &c) {
            panic!("{f:?}: GPU != CPU at {d}");
        }
    }
    for f in [Filter::ColorHalftone { radius: 4.0 }, Filter::ColorHalftone { radius: 10.0 }, Filter::Pointillize { cell: 6.0, bg: [255; 3] }, Filter::Pointillize { cell: 90.0, bg: [0; 3] }] {
        let c = cpu(&img, &f, None);
        let (o, _) = on_gpu(&g, &img, &f, None, &RunOpts::default());
        let (mx, de, _) = report(f.name(), &o, &c);
        assert!(mx <= 1 && de <= 0.2, "{f:?}: max {mx}, ΔE {de:.4}");
    }
}

/// Paths and timings of the multi-pass kernels on a wide image.
#[test]
#[ignore]
fn wide_paths() {
    let Some(g) = gpu() else { return };
    let side: u32 = std::env::var("PF_WIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let img = crate::engine::ops::procedural(16384, side);
    for f in [oil(4.0, Oil { stylization: 4.0, bristle: 3.0, shine: 2.0, ..Default::default() }), oil(4.0, Oil::default()), Filter::ColorHalftone { radius: 8.0 }] {
        for _ in 0..2 {
            let t = std::time::Instant::now();
            let (_, path) = on_gpu(&g, &img, &f, None, &RunOpts::default());
            println!("{}: {path:?} {:.1} ms", f.name(), t.elapsed().as_secs_f64() * 1e3);
        }
    }
}
