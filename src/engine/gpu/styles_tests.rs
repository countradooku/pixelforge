//! Layer Styles: GPU against the CPU (`styles::render`).
//!
//! Every effect plane must be bit-exact; the styled result within 1/255 per
//! channel and mean ΔE2000 <= 0.2 (exact when every blend is Normal), on a
//! photo with alpha, the edge-case grid and text-like shapes, at odd sizes
//! and mip levels 0-3.

use std::sync::Arc;

use super::tests::{edge_case, gpu, photo};
use crate::engine::blend::BlendMode;
use crate::engine::document::Layer;
use crate::engine::grid::{RectI, TileGrid};
use crate::engine::px;
use crate::engine::styles::{self as st, *};

/// Thin anti-aliased strokes and dots, like small text (odd size).
fn text_like() -> TileGrid {
    let (w, h) = (731u32, 413u32);
    let segs: Vec<[f32; 5]> = (0..90)
        .map(|i| {
            let r = |k: u32| (st::hash_u(i, k as i32, 9) & 0xffff) as f32 / 65536.0;
            let (x, y) = (20.0 + r(1) * 690.0, 20.0 + r(2) * 370.0);
            let a = r(3) * 6.283;
            let l = 4.0 + r(4) * 30.0;
            [x, y, x + a.cos() * l, y + a.sin() * l, 0.8 + r(5) * 3.0]
        })
        .collect();
    let mut g = TileGrid::new(w, h);
    g.tiles = crate::engine::grid::build_grid(w, h, |tx, ty| {
        let mut t = vec![0u32; 256 * 256].into_boxed_slice();
        let mut any = false;
        for y in 0..256 {
            for x in 0..256 {
                let (gx, gy) = ((tx * 256 + x) as f32 + 0.5, (ty * 256 + y) as f32 + 0.5);
                if gx > w as f32 || gy > h as f32 {
                    continue;
                }
                let mut cov = 0f32;
                for s in &segs {
                    let (dx, dy) = (s[2] - s[0], s[3] - s[1]);
                    let t = (((gx - s[0]) * dx + (gy - s[1]) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
                    let d = ((gx - s[0] - t * dx).powi(2) + (gy - s[1] - t * dy).powi(2)).sqrt();
                    cov = cov.max((s[4] * 0.5 + 0.5 - d).clamp(0.0, 1.0));
                }
                if cov > 0.0 {
                    any = true;
                    t[(y * 256 + x) as usize] = px::premultiply(30, 60, 200, (cov * 255.0).round() as u8);
                }
            }
        }
        if any { crate::engine::grid::finish_tile(t) } else { None }
    })
    .tiles;
    g
}

/// The dog photo inside an anti-aliased ellipse.
fn photo_alpha() -> TileGrid {
    let src = photo("dog_grass");
    let sel = crate::engine::selection::Selection::ellipse(RectI::new(90, 60, src.w as i32 - 130, src.h as i32 - 40));
    crate::engine::ops::clear(&src, Some(&crate::engine::selection::Selection { inverted: true, ..sel }))
}

fn styles() -> Vec<(&'static str, LayerStyle, f32)> {
    let mut v = Vec::new();
    // Photoshop's defaults for everything at once.
    let mut all = LayerStyle::default();
    for e in Effect::ALL {
        all.set_enabled(e, true);
    }
    v.push(("all defaults", all.clone(), 1.0));
    // Normal blends only: exact.
    let mut normal = LayerStyle::default();
    normal.drop_shadow = Shadow { enabled: true, blend: BlendMode::Normal, size: 9.0, spread: 30.0, distance: 7.0, noise: 25.0, ..Shadow::drop() };
    normal.outer_glow = Glow { enabled: true, blend: BlendMode::Normal, range: 30.0, size: 12.0, ..Glow::outer() };
    normal.inner_shadow = Shadow { enabled: true, blend: BlendMode::Normal, angle: -45.0, size: 4.0, spread: 50.0, ..Shadow::inner() };
    normal.stroke = StrokeFx { enabled: true, size: 4.0, position: StrokePos::Center, opacity: 80.0, ..StrokeFx::default() };
    normal.color_overlay = ColorOverlay { enabled: true, opacity: 40.0, ..ColorOverlay::default() };
    v.push(("normal blends", normal.clone(), 0.7));
    // A linear gradient (float position: within 1/255).
    normal.gradient_overlay = GradientOverlay { enabled: true, opacity: 60.0, angle: 30.0, ..GradientOverlay::default() };
    v.push(("linear gradient", normal, 0.7));
    // The bench style: drop shadow + stroke + bevel.
    let mut bench = LayerStyle::default();
    bench.drop_shadow.enabled = true;
    bench.stroke.enabled = true;
    bench.bevel.enabled = true;
    v.push(("shadow+stroke+bevel", bench, 1.0));
    // Every blend mode somewhere, other techniques and positions.
    for (i, chunk) in BlendMode::ALL.chunks(5).enumerate() {
        let m = |k: usize| chunk[k % chunk.len()];
        let mut s = LayerStyle::default();
        s.color_overlay = ColorOverlay { enabled: true, blend: m(0), color: [200, 120, 40], opacity: 70.0 };
        s.satin = Satin { enabled: true, blend: m(1), color: [20, 200, 90], invert: i % 2 == 0, ..Satin::default() };
        s.inner_glow = Glow { enabled: true, blend: m(2), center: i % 2 == 1, range: 70.0, size: 8.0, spread: 20.0, ..Glow::inner() };
        s.bevel = Bevel {
            enabled: true,
            style: BevelStyle::ALL[i % 4],
            up: i % 2 == 0,
            soften: 2.0,
            size: 7.0,
            hi_blend: m(3),
            sh_blend: m(4),
            ..Bevel::default()
        };
        s.gradient_overlay = GradientOverlay { enabled: true, blend: m(4), radial: true, reverse: true, scale: 80.0, ..GradientOverlay::default() };
        s.stroke = StrokeFx { enabled: true, position: StrokePos::ALL[i % 3], blend: m(2), size: 3.0, ..StrokeFx::default() };
        s.drop_shadow = Shadow { enabled: true, blend: m(3), knockout: i % 2 == 0, ..Shadow::drop() };
        s.outer_glow = Glow { enabled: true, blend: m(1), ..Glow::outer() };
        v.push(("blend modes", s, 0.5 + 0.1 * i as f32));
    }
    // Fill only.
    v.push(("fill only", LayerStyle::default(), 0.4));
    v
}

fn lane_of(planes: &[Vec<u32>], lane: u8, i: usize) -> u8 {
    (planes[lane as usize / 4][i] >> ((lane % 4) * 8)) as u8
}

/// (max channel diff, mean ΔE2000, pixels over 1).
fn stats(a: &[u32], b: &[u32]) -> (u8, f64, usize) {
    let mut mx = 0u8;
    let mut over = 0;
    for (&x, &y) in a.iter().zip(b) {
        let d = px::unpack(x).iter().zip(px::unpack(y)).map(|(p, q)| p.abs_diff(q)).max().unwrap();
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
    (mx, crate::engine::metrics::mean_delta_e(&rgb(a), &rgb(b), None), over)
}

#[test]
fn styles_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let inputs = [("photo+alpha", photo_alpha()), ("edge", edge_case()), ("text", text_like())];
    let mut checked = 0;
    for (iname, grid) in inputs {
        let mut layer = Layer::new(1, iname, grid);
        layer.mips = crate::engine::mips::build_mips(&layer.grid);
        for (sname, style, fill) in styles() {
            layer.style = Some(Arc::new(style.clone()));
            layer.fill = fill;
            let bounds = layer_bounds(&layer);
            for level in 0..4u32 {
                let plan = Plan::new(Some(&style), fill, level, bounds);
                let lg = layer.level(level);
                // An odd region straddling the image edge.
                let rect = RectI::new(-5, 3, (lg.w as i32 * 3 / 4).min(415) | 1, (lg.h as i32 - 2).min(333) | 1);
                let pad = plan.pad as i32;
                let (pw, ph) = ((rect.w() + 2 * pad) as usize, (rect.h() + 2 * pad) as usize);
                let origin = (rect.x0 - pad, rect.y0 - pad);
                let mut src = vec![0u32; pw * ph];
                lg.read_region(origin.0, origin.1, pw, ph, &mut src);
                let mut cpu = vec![0u32; rect.w() as usize * rect.h() as usize];
                st::render(&plan, &src, pw, ph, origin, &mut cpu);
                let (out, planes) = super::styles::run_with_lanes(&g, &plan, &src, pw as u32, ph as u32, origin, true).unwrap_or_else(|e| panic!("{iname} / {sname} level {level}: {e}"));
                // Planes: bit-exact over the whole padded buffer.
                let lanes = st::lanes(&plan, &src, pw, ph, origin);
                for (l, cl) in lanes.iter().enumerate() {
                    let Some(cl) = cl else { continue };
                    if let Some(i) = (0..pw * ph).find(|&i| cl[i] != lane_of(&planes, l as u8, i)) {
                        panic!("{iname} / {sname} level {level}: lane {l} at ({}, {}): GPU {} CPU {}", i % pw, i / pw, lane_of(&planes, l as u8, i), cl[i]);
                    }
                }
                let (mx, de, over) = stats(&out, &cpu);
                let exact = sname == "normal blends" || sname == "fill only";
                assert!(mx <= 1 && de <= 0.2, "{iname} / {sname} level {level}: max {mx}, ΔE {de:.4}, {over} px over 1");
                if exact {
                    assert_eq!(mx, 0, "{iname} / {sname} level {level}: Normal-only stacks are integer-exact");
                }
                if level == 0 {
                    println!("{iname:12} {sname:22} max {mx} ΔE {de:.5}");
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}

/// The compositor's CPU path through `render_item` equals the plain
/// `styles::render` the GPU is checked against.
#[test]
fn compositor_hook_equals_render() {
    let mut layer = Layer::new(1, "t", text_like());
    let mut s = LayerStyle::default();
    s.drop_shadow.enabled = true;
    s.bevel.enabled = true;
    layer.style = Some(Arc::new(s));
    let layers = vec![layer];
    let input = crate::engine::composite::ComposeInput { layers: &layers, preview: None, extra: None };
    let rect = RectI::new(100, 50, 357, 307);
    let mut out = vec![0u32; 257 * 257];
    crate::engine::composite::compose_region(&input, 0, rect.x0, rect.y0, 257, 257, &mut out, &mut Default::default());
    let plan = plan_for(&layers[0], 0).unwrap();
    let pad = plan.pad as i32;
    let (pw, ph) = (257 + 2 * pad as usize, 257 + 2 * pad as usize);
    let mut src = vec![0u32; pw * ph];
    layers[0].grid.read_region(rect.x0 - pad, rect.y0 - pad, pw, ph, &mut src);
    let mut want = vec![0u32; 257 * 257];
    st::render(&plan, &src, pw, ph, (rect.x0 - pad, rect.y0 - pad), &mut want);
    assert!(out == want);
}
