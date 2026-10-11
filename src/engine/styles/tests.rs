use std::sync::Arc;

use super::*;
use crate::engine::composite::{self, ComposeInput, Scratch};
use crate::engine::grid::TileGrid;

/// A layer with an opaque red square `r` on transparency.
fn square_layer(r: RectI) -> Layer {
    let mut g = TileGrid::new(512, 512);
    g.tiles = crate::engine::grid::build_grid(512, 512, |tx, ty| {
        let mut t = vec![0u32; 256 * 256].into_boxed_slice();
        for y in 0..256 {
            for x in 0..256 {
                let (gx, gy) = (tx * 256 + x, ty * 256 + y);
                if gx >= r.x0 && gx < r.x1 && gy >= r.y0 && gy < r.y1 {
                    t[(y * 256 + x) as usize] = px::pack(255, 0, 0, 255);
                }
            }
        }
        crate::engine::grid::finish_tile(t)
    })
    .tiles;
    Layer::new(1, "sq", g)
}

fn with_style(mut l: Layer, f: impl FnOnce(&mut LayerStyle)) -> Layer {
    let mut st = LayerStyle::default();
    f(&mut st);
    l.style = Some(Arc::new(st));
    l
}

fn at(l: &Layer, x: i32, y: i32) -> [u8; 4] {
    px::unpack(composite::sample(std::slice::from_ref(l), x, y))
}

const SQ: RectI = RectI { x0: 200, y0: 200, x1: 300, y1: 300 };

#[test]
fn defaults_are_unstyled() {
    let l = square_layer(SQ);
    assert!(!is_styled(&l) && l.is_plain());
    // A style with every effect off is no style.
    let l = with_style(l, |_| {});
    assert!(!is_styled(&l));
    assert!(plan_for(&l, 0).is_none());
}

#[test]
fn drop_shadow_is_offset_by_distance_at_angle() {
    for (angle, dir) in [(90.0f32, (0, 1)), (0.0, (-1, 0)), (180.0, (1, 0)), (-90.0, (0, -1))] {
        let l = with_style(square_layer(SQ), |s| {
            s.drop_shadow = Shadow { enabled: true, blend: BlendMode::Normal, color: [0, 0, 255], opacity: 100.0, angle, distance: 20.0, spread: 0.0, size: 0.0, noise: 0.0, knockout: true };
        });
        let (cx, cy) = (250, 250);
        // The pixel `d` px beyond the square (200..300) along `dir`.
        let beyond = |s: i32, d: i32| if s > 0 { 299 + d } else if s < 0 { 200 - d } else { 250 };
        // 1 and 20 px beyond the square's edge in the shadow direction: shadow.
        for d in [1, 20] {
            assert_eq!(at(&l, beyond(dir.0, d), beyond(dir.1, d)), [0, 0, 255, 255], "angle {angle} at {d}");
        }
        // 21 px beyond: nothing; opposite side: nothing.
        assert_eq!(at(&l, beyond(dir.0, 21), beyond(dir.1, 21)), [0, 0, 0, 0], "angle {angle}");
        assert_eq!(at(&l, beyond(-dir.0, 1), beyond(-dir.1, 1)), [0, 0, 0, 0], "angle {angle}");
        // The content covers the shadow.
        assert_eq!(at(&l, cx, cy), [255, 0, 0, 255]);
    }
    // Size blurs it: the edge is soft and extends further.
    let l = with_style(square_layer(SQ), |s| {
        s.drop_shadow = Shadow { enabled: true, blend: BlendMode::Normal, color: [0, 0, 0], opacity: 100.0, angle: 90.0, distance: 20.0, spread: 0.0, size: 10.0, noise: 0.0, knockout: true };
    });
    let a = |y| at(&l, 250, y)[3];
    assert!(a(305) == 255 && a(320) > 64 && a(320) < 192 && a(331) == 0, "{} {} {}", a(305), a(320), a(331));
}

#[test]
fn stroke_width_and_position() {
    let stroke = |pos| {
        with_style(square_layer(SQ), |s| {
            s.stroke = StrokeFx { enabled: true, size: 5.0, position: pos, blend: BlendMode::Normal, opacity: 100.0, color: [0, 255, 0] };
        })
    };
    let green = [0, 255, 0, 255];
    let red = [255, 0, 0, 255];
    let out = stroke(StrokePos::Outside);
    for d in 1..=5 {
        assert_eq!(at(&out, 300 + d - 1, 250), green, "outside {d}");
        assert_eq!(at(&out, 250, 200 - d), green, "outside top {d}");
    }
    assert_eq!(at(&out, 306, 250), [0, 0, 0, 0]);
    assert_eq!(at(&out, 299, 250), red);
    // Round corners: 5 px along the diagonal is beyond the stroke.
    assert_eq!(at(&out, 304, 304)[3], 0);
    assert_eq!(at(&out, 302, 302), green);
    let ins = stroke(StrokePos::Inside);
    for d in 0..5 {
        assert_eq!(at(&ins, 299 - d, 250), green, "inside {d}");
    }
    assert_eq!(at(&ins, 294, 250), red);
    assert_eq!(at(&ins, 300, 250), [0, 0, 0, 0]);
    let mid = stroke(StrokePos::Center);
    assert_eq!(at(&mid, 298, 250), green);
    assert_eq!(at(&mid, 301, 250), green);
    assert_eq!(at(&mid, 295, 250), red);
    assert_eq!(at(&mid, 304, 250), [0, 0, 0, 0]);
}

#[test]
fn fill_zero_keeps_effects_but_hides_content() {
    let mut l = with_style(square_layer(SQ), |s| {
        s.stroke = StrokeFx { enabled: true, size: 4.0, position: StrokePos::Outside, blend: BlendMode::Normal, opacity: 100.0, color: [0, 255, 0] };
    });
    l.fill = 0.0;
    assert_eq!(at(&l, 250, 250), [0, 0, 0, 0]);
    assert_eq!(at(&l, 301, 250), [0, 255, 0, 255]);
    // Interior effects stay too.
    let mut l = with_style(square_layer(SQ), |s| s.color_overlay = ColorOverlay { enabled: true, blend: BlendMode::Normal, color: [10, 20, 30], opacity: 100.0 });
    l.fill = 0.0;
    assert_eq!(at(&l, 250, 250), [10, 20, 30, 255]);
    // Fill alone scales the content (and only the content).
    let mut l = square_layer(SQ);
    l.fill = 0.5;
    assert!(is_styled(&l) && !l.is_plain());
    assert_eq!(at(&l, 250, 250), [128, 0, 0, 128]);
    // Opacity still scales everything.
    let mut l = with_style(square_layer(SQ), |s| {
        s.stroke = StrokeFx { enabled: true, size: 4.0, position: StrokePos::Outside, blend: BlendMode::Normal, opacity: 100.0, color: [0, 255, 0] };
    });
    l.fill = 0.0;
    l.opacity = 0.5;
    assert_eq!(at(&l, 301, 250), [0, 128, 0, 128]);
}

#[test]
fn inner_effects_stay_inside_and_outer_ones_outside() {
    let l = with_style(square_layer(SQ), |s| {
        s.inner_shadow = Shadow { enabled: true, blend: BlendMode::Normal, color: [0, 0, 0], opacity: 100.0, angle: 90.0, distance: 8.0, spread: 0.0, size: 0.0, noise: 0.0, knockout: false };
        s.inner_glow = Glow { enabled: true, blend: BlendMode::Normal, color: [0, 0, 255], opacity: 100.0, spread: 0.0, size: 6.0, range: 50.0, center: false };
    });
    // Inner shadow: the top 8 rows inside are black; nothing leaks outside.
    assert_eq!(at(&l, 250, 203), [0, 0, 0, 255]);
    assert_eq!(at(&l, 250, 195), [0, 0, 0, 0]);
    assert_eq!(at(&l, 250, 250), [255, 0, 0, 255]);
    // Inner glow near the other edges, gone in the middle.
    assert!(at(&l, 299, 250)[2] > 100);
    assert_eq!(at(&l, 300, 250), [0, 0, 0, 0]);
    let l = with_style(square_layer(SQ), |s| {
        s.outer_glow = Glow { enabled: true, blend: BlendMode::Normal, color: [0, 0, 255], opacity: 100.0, spread: 50.0, size: 10.0, range: 50.0, center: false };
    });
    assert_eq!(at(&l, 250, 250), [255, 0, 0, 255]);
    // 5 px hard spread, then 5 px of blur.
    let g = at(&l, 301, 250);
    assert!(g[3] > 200 && g[2] == g[3] && g[0] == 0, "{g:?}");
    assert!(at(&l, 309, 250)[3] > 0 && at(&l, 312, 250)[3] == 0);
}

#[test]
fn bevel_satin_and_gradient_render() {
    let l = with_style(square_layer(SQ), |s| {
        s.bevel = Bevel { enabled: true, ..Bevel::default() };
    });
    // Light from 120°: the top-left edges brighten, the bottom-right darken.
    let tl = at(&l, 250, 201);
    let br = at(&l, 250, 298);
    assert!(tl[1] > 60 && br[0] < 200, "{tl:?} {br:?}");
    assert_eq!(at(&l, 250, 250), [255, 0, 0, 255]);
    assert_eq!(at(&l, 250, 190), [0, 0, 0, 0]);
    let outer = with_style(square_layer(SQ), |s| s.bevel = Bevel { enabled: true, style: BevelStyle::Outer, ..Bevel::default() });
    assert!(at(&outer, 250, 197)[3] > 0 && at(&outer, 250, 210) == [255, 0, 0, 255]);
    let g = with_style(square_layer(SQ), |s| {
        s.gradient_overlay = GradientOverlay { enabled: true, ..GradientOverlay::default() };
        s.satin = Satin { enabled: true, ..Satin::default() };
    });
    // Angle 90: black at the bottom, white at the top (before satin, which
    // only darkens).
    let (top, bottom) = (at(&g, 250, 202), at(&g, 250, 297));
    assert!(top[0] > bottom[0] + 100, "{top:?} {bottom:?}");
}

#[test]
fn dilation_transform_is_exact() {
    // Felzenszwalb with integer comparisons = brute force.
    let (w, h) = (61usize, 47usize);
    let mut p = vec![0u8; w * h];
    for (i, v) in p.iter_mut().enumerate() {
        let hsh = hash_u(i as i32, 7, 3) >> 16;
        *v = if hsh < 12 { 255 } else if hsh < 20 { (hsh * 13) as u8 } else { 0 };
    }
    for r in [0.5f32, 2.0, 3.7, 9.0] {
        let d = Dil::new(r).unwrap();
        let mut a = p.clone();
        dilate(&mut a, w, h, &d);
        for y in 0..h as i64 {
            for x in 0..w as i64 {
                let mut best = INF;
                for qy in 0..h as i64 {
                    for qx in 0..w as i64 {
                        let v = p[(qy as usize) * w + qx as usize];
                        if v > 0 {
                            let dd = (((x - qx).pow(2) + (y - qy).pow(2)) * 256 + 255 - v as i64) as u32;
                            best = best.min(dd);
                        }
                    }
                }
                let want = p[y as usize * w + x as usize].max(d.coverage(best));
                assert_eq!(a[y as usize * w + x as usize], want, "r {r} ({x},{y})");
            }
        }
    }
}

/// Any region split gives the same pixels (no tile seams), at every level.
#[test]
fn styled_regions_are_seamless() {
    let mut l = with_style(square_layer(RectI::new(150, 170, 371, 333)), |s| {
        s.drop_shadow = Shadow { enabled: true, noise: 30.0, size: 12.0, spread: 20.0, distance: 9.0, ..Shadow::drop() };
        s.stroke = StrokeFx { enabled: true, size: 6.0, position: StrokePos::Center, ..StrokeFx::default() };
        s.bevel = Bevel { enabled: true, soften: 3.0, style: BevelStyle::Pillow, ..Bevel::default() };
        s.gradient_overlay = GradientOverlay { enabled: true, blend: BlendMode::Multiply, radial: true, ..GradientOverlay::default() };
        s.inner_glow = Glow { enabled: true, center: true, range: 30.0, ..Glow::inner() };
        s.satin = Satin { enabled: true, ..Satin::default() };
    });
    l.fill = 0.6;
    let layers = vec![l];
    let input = ComposeInput { layers: &layers, preview: None, extra: None };
    for level in 0..3u32 {
        let (w, h) = (512 >> level, 512 >> level);
        let mut s = Scratch::default();
        let mut whole = vec![0u32; w * h];
        composite::compose_region(&input, level, 0, 0, w, h, &mut whole, &mut s);
        let mut part = vec![0u32; 37 * 53];
        for (x0, y0) in [(130 >> level, 150 >> level), (37, 91), (250 >> level, 300 >> level)] {
            composite::compose_region(&input, level, x0 as i32, y0 as i32, 37, 53, &mut part, &mut s);
            for y in 0..53 {
                for x in 0..37 {
                    let (gx, gy) = (x0 + x, y0 + y);
                    if gx < w && gy < h {
                        assert_eq!(part[y * 37 + x], whole[gy * w + gx], "level {level} ({gx},{gy})");
                    }
                }
            }
        }
        assert!(whole.iter().any(|&p| p != 0));
    }
}

#[test]
fn style_bytes_round_trip_and_pforge_keeps_styles() {
    let mut st = LayerStyle::default();
    st.drop_shadow = Shadow { enabled: true, color: [1, 2, 3], opacity: 42.0, angle: -30.0, noise: 7.0, knockout: false, ..Shadow::drop() };
    st.bevel = Bevel { enabled: true, style: BevelStyle::Pillow, up: false, hi_blend: BlendMode::ColorDodge, ..Bevel::default() };
    st.stroke = StrokeFx { enabled: true, position: StrokePos::Center, size: 9.0, ..StrokeFx::default() };
    st.gradient_overlay = GradientOverlay { enabled: false, radial: true, reverse: true, from: [9, 8, 7], ..GradientOverlay::default() };
    st.inner_glow.center = true;
    st.satin.invert = false;
    assert_eq!(LayerStyle::from_bytes(&st.to_bytes()), Some(st.clone()));
    // Unknown records are skipped; truncation is rejected.
    let mut b = st.to_bytes();
    b.extend_from_slice(&[99, 3, 0, 0, 0, 1, 2, 3]);
    assert_eq!(LayerStyle::from_bytes(&b), Some(st.clone()));
    assert_eq!(LayerStyle::from_bytes(&b[..b.len() - 12]), None);

    let mut a = square_layer(SQ);
    a.style = Some(Arc::new(st.clone()));
    a.fill = 0.25;
    let mut b = square_layer(SQ);
    b.id = 2;
    let path = std::env::temp_dir().join(format!("pforge-styles-{}.pforge", std::process::id()));
    crate::engine::io::save_native(&[a, b], 512, 512, 0, &Default::default(), &path).unwrap();
    let (w, h, active, layers, extras) = crate::engine::io::load_native(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let doc = crate::engine::io::native_to_document(w, h, active, layers, extras);
    assert_eq!(doc.layers[0].style.as_deref(), Some(&st));
    assert_eq!(doc.layers[0].fill, 0.25);
    assert!(doc.layers[1].style.is_none());
    assert_eq!(doc.layers[1].fill, 1.0);
}

#[test]
fn undo_restores_style_and_fill() {
    let mut doc = crate::engine::document::Document::from_grid(square_layer(SQ).grid, "L");
    doc.push_undo("Layer Style");
    doc.layers[0].style = Some(Arc::new(LayerStyle { stroke: StrokeFx { enabled: true, ..StrokeFx::default() }, ..LayerStyle::default() }));
    doc.layers[0].fill = 0.3;
    doc.undo();
    assert!(doc.layers[0].style.is_none());
    assert_eq!(doc.layers[0].fill, 1.0);
}
