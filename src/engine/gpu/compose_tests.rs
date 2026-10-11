//! GPU compositor = CPU compositor, display tile by display tile.

use std::sync::Arc;

use super::tests::{edge_case, gpu, photo};
use crate::engine::blend::BlendMode;
use crate::engine::composite::{ComposeInput, DT, Preview, Scratch, display_tile};
use crate::engine::document::Layer;
use crate::engine::filters::{self, Filter};
use crate::engine::grid::{RectI, TileGrid};
use crate::engine::layers::{LayerKind, LayerMask};
use crate::engine::selection::{MaskGrid, Selection};

fn layer(id: u64, g: TileGrid) -> Layer {
    Layer::new(id, format!("L{id}"), g)
}

/// Compares every visible display tile of `levels` (GPU vs CPU): max
/// channel difference allowed (0 for exact).
fn check(name: &str, input: &ComposeInput, w: u32, h: u32, tol: u8) {
    let Some(g) = gpu() else { return };
    for level in [-2i32, 0, 1, 2] {
        let span = if level >= 0 { 256i64 << level } else { 256 >> (-level) } as f32;
        let (cols, rows) = ((w as f32 / span).ceil() as i32, (h as f32 / span).ceil() as i32);
        // Negative levels: a few tiles only (they are tiny regions).
        let tiles: Vec<(i32, i32)> = (0..rows.min(3)).flat_map(|y| (0..cols.min(4)).map(move |x| (x, y))).collect();
        let out = super::compose::display_tiles(&g, input, level, &tiles, None).unwrap_or_else(|e| panic!("{name} level {level}: {e}"));
        let mut s = Scratch::default();
        let mut worst = 0u8;
        let mut differing = 0usize;
        for (k, &(tx, ty)) in tiles.iter().enumerate() {
            let mut c = vec![0u8; DT * DT * 4];
            display_tile(input, level, tx, ty, &mut c, &mut s);
            let mx = c.iter().zip(&out[k]).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            worst = worst.max(mx);
            differing += c.iter().zip(&out[k]).filter(|(a, b)| a != b).count();
            if mx > tol {
                let i = c.iter().zip(&out[k]).position(|(a, b)| a.abs_diff(*b) > tol).unwrap() / 4;
                panic!("{name}: level {level} tile ({tx}, {ty}) px ({}, {}): CPU {:?} GPU {:?}", i % DT, i / DT, &c[i * 4..i * 4 + 4], &out[k][i * 4..i * 4 + 4]);
            }
        }
        println!("{name} level {level}: {} tiles, max diff {worst}, {differing} bytes differ", tiles.len());
    }
}

#[test]
fn plain_stacks_match_for_every_blend_mode() {
    let base = layer(1, photo("dog_grass"));
    let top_src = edge_case();
    for mode in BlendMode::ALL {
        let mut top = layer(2, top_src.clone());
        top.blend = mode;
        top.opacity = 0.8;
        let mut third = layer(3, photo("butterfly"));
        third.opacity = 0.35;
        third.blend = if mode == BlendMode::Normal { BlendMode::Multiply } else { BlendMode::Normal };
        let layers = vec![base.clone(), top, third];
        let input = ComposeInput { layers: &layers, preview: None, extra: None };
        // The newly added PSD modes use exact integer math, including the
        // coordinate hash. Retain the historical float-mode tolerance.
        check(mode.name(), &input, 1024, 683, if mode.index() >= BlendMode::Dissolve.index() { 0 } else { 1 });
    }
    // Normal only: bit-exact.
    let mut top = layer(2, edge_case());
    top.opacity = 0.6;
    let layers = vec![base, top];
    check("normal exact", &ComposeInput { layers: &layers, preview: None, extra: None }, 1024, 683, 0);
}

#[test]
fn masks_groups_clipping_and_adjustments_match() {
    let (w, h) = (1024u32, 683u32);
    let bg = layer(1, photo("dog_grass"));
    let mut a = layer(2, edge_case());
    a.mask = Some(LayerMask::new(MaskGrid::from_selection(&Selection::ellipse(RectI::new(100, 50, 700, 600)), w, h)));
    a.blend = BlendMode::Overlay;
    // Clipped onto `a`.
    let mut c1 = layer(3, photo("butterfly"));
    c1.clip = true;
    c1.opacity = 0.7;
    c1.blend = BlendMode::Screen;
    let mut c2 = layer(4, TileGrid::new(w, h));
    c2.kind = LayerKind::Adjustment(Filter::Invert);
    c2.clip = true;
    // An isolated group with a mask, a child in Multiply, and an adjustment.
    let mut grp = layer(10, TileGrid::new(w, h));
    grp.kind = LayerKind::Group { pass_through: false, collapsed: false };
    grp.opacity = 0.75;
    grp.mask = Some(LayerMask::new(MaskGrid::from_selection(&Selection::rect(RectI::new(300, 0, 1024, 683)), w, h)));
    let mut k1 = layer(11, photo("portrait_hair"));
    k1.parent = Some(10);
    k1.blend = BlendMode::Multiply;
    let mut k2 = layer(12, TileGrid::new(w, h));
    k2.parent = Some(10);
    k2.kind = LayerKind::Adjustment(filters::hue_saturation(40.0, 20.0, 0.0));
    k2.opacity = 0.6;
    // A pass-through group.
    let mut pt = layer(20, TileGrid::new(w, h));
    pt.kind = LayerKind::Group { pass_through: true, collapsed: false };
    let mut p1 = layer(21, edge_case());
    p1.parent = Some(20);
    p1.blend = BlendMode::Difference;
    p1.offset = (37, -21);
    // A masked adjustment on top.
    let mut adj = layer(30, TileGrid::new(w, h));
    adj.kind = LayerKind::Adjustment(Filter::Levels { black: 20.0, white: 220.0, gamma: 1.2 });
    adj.mask = Some(LayerMask::new(MaskGrid::from_selection(&Selection::ellipse(RectI::new(0, 0, 600, 400)), w, h)));
    let layers = vec![bg, a, c1, c2, k1, k2, grp, p1, pt, adj];
    check("structure", &ComposeInput { layers: &layers, preview: None, extra: None }, w, h, 1);
}

#[test]
fn previews_and_extra_layers_match() {
    let (w, h) = (1024u32, 683u32);
    let layers = vec![layer(1, photo("dog_grass")), layer(2, edge_case())];
    let pv = Preview::new(1, Filter::GaussianBlur { radius: 12.0 }, Some(Selection::ellipse(RectI::new(200, 100, 800, 600))));
    let mut extra = layer(99, edge_case());
    extra.opacity = 0.4;
    let input = ComposeInput { layers: &layers, preview: Some(&pv), extra: Some(&extra) };
    check("preview", &input, w, h, 0);
    let _ = Arc::new(0);
}

/// Foreground caching: frames that edit the top layer reuse the composite
/// below it; editing a lower layer rebuilds it; every frame equals the CPU.
#[test]
fn foreground_cache_stays_exact_across_edits() {
    let Some(g) = gpu() else { return };
    let (w, h) = (1024u32, 683u32);
    let mut l2 = layer(2, edge_case());
    l2.blend = BlendMode::Multiply;
    let mut layers = vec![layer(1, photo("dog_grass")), l2, layer(3, edge_case())];
    let tiles: Vec<(i32, i32)> = (0..3).flat_map(|y| (0..4).map(move |x| (x, y))).collect();
    let view = Some((0, 0, 4, 3));
    let hits0 = super::compose::PREFIX_HITS.load(std::sync::atomic::Ordering::Relaxed);
    let paint = |l: &mut Layer, c: u32| {
        let t = Arc::new(crate::engine::grid::Tile::solid(c));
        l.grid.tiles[1] = Some(t);
        l.mips = crate::engine::mips::build_mips(&l.grid);
    };
    for step in 0..5 {
        match step {
            1 | 2 => paint(&mut layers[2], crate::engine::px::pack(10 * step as u8, 200, 30, 255)),
            3 => paint(&mut layers[0], crate::engine::px::pack(200, 0, 0, 255)),
            4 => layers[1].opacity = 0.5,
            _ => {}
        }
        let input = ComposeInput { layers: &layers, preview: None, extra: None };
        let out = super::compose::display_tiles(&g, &input, 0, &tiles, view).unwrap();
        let mut s = Scratch::default();
        for (k, &(tx, ty)) in tiles.iter().enumerate() {
            let mut c = vec![0u8; DT * DT * 4];
            display_tile(&input, 0, tx, ty, &mut c, &mut s);
            let mx = c.iter().zip(&out[k]).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(mx <= 1, "step {step} tile ({tx}, {ty}): max diff {mx}");
        }
    }
    let hits = super::compose::PREFIX_HITS.load(std::sync::atomic::Ordering::Relaxed) - hits0;
    assert!(hits >= 2, "the cached backdrop was reused {hits} times");
    let _ = (w, h);
}

/// Styled layers (Layer Style + Fill), a styled clipping base and a styled
/// pass-through group, through the GPU compositor = the CPU's.
#[test]
fn layer_styles_composite_like_the_cpu() {
    use crate::engine::styles::LayerStyle;
    let (w, h) = (1024u32, 683u32);
    let mut st = LayerStyle::default();
    st.drop_shadow.enabled = true;
    st.stroke.enabled = true;
    st.bevel.enabled = true;
    let mut a = layer(2, edge_case());
    a.mask = Some(LayerMask::new(MaskGrid::from_selection(&Selection::ellipse(RectI::new(80, 40, 900, 640)), w, h)));
    a.style = Some(Arc::new(st.clone()));
    a.fill = 0.6;
    let mut c = layer(3, photo("butterfly"));
    c.clip = true;
    c.blend = BlendMode::Multiply;
    let mut grp = layer(10, TileGrid::new(w, h));
    grp.kind = LayerKind::Group { pass_through: true, collapsed: false };
    grp.style = Some(Arc::new(st));
    let mut kid = layer(11, edge_case());
    kid.parent = Some(10);
    kid.offset = (-40, 25);
    let layers = vec![layer(1, photo("dog_grass")), a, c, kid, grp];
    check("styles", &ComposeInput { layers: &layers, preview: None, extra: None }, w, h, 1);
}

/// Consecutive adjustment layers (LUT, 3D LUT, masked, threshold,
/// desaturate) run as one fused pass, equal to the CPU.
#[test]
fn fused_adjustment_chains_match() {
    let (w, h) = (1024u32, 683u32);
    let adj = |id: u64, f: Filter, op: f32| {
        let mut l = layer(id, TileGrid::new(w, h));
        l.kind = LayerKind::Adjustment(f);
        l.opacity = op;
        l
    };
    let mut masked = adj(4, filters::vibrance(50.0, 20.0), 0.7);
    masked.mask = Some(LayerMask::new(MaskGrid::from_selection(&Selection::ellipse(RectI::new(100, 60, 800, 600)), w, h)));
    let layers = vec![
        layer(1, photo("dog_grass")),
        adj(2, Filter::Curves { pts: [0.0, 70.0, 128.0, 190.0, 255.0] }, 1.0),
        adj(3, filters::hue_saturation(30.0, 10.0, 0.0), 0.8),
        masked,
        adj(5, Filter::Desaturate, 0.5),
        layer(6, edge_case()),
        adj(7, Filter::Threshold { level: 120.0 }, 0.4),
        adj(8, Filter::Invert, 0.3),
    ];
    check("fused adjustments", &ComposeInput { layers: &layers, preview: None, extra: None }, w, h, 0);
}

/// A GPU-resident live preview region is copied into the GPU compositor's
/// frame (with its selection blended on the GPU) and equals the CPU.
#[test]
fn gpu_resident_previews_composite_like_the_cpu() {
    let Some(g) = gpu() else { return };
    let (w, h) = (1024u32, 683u32);
    let layers = vec![layer(1, photo("dog_grass")), layer(2, edge_case())];
    for sel in [None, Some(Selection::ellipse(RectI::new(200, 100, 800, 600)))] {
        let pv = Preview::new(1, Filter::GaussianBlur { radius: 12.0 }, sel);
        for level in [0u32, 1, 2] {
            let r = RectI::new(-1, -1, ((w >> level) as i32) + 300, ((h >> level) as i32) + 300);
            super::preview::prepare_on(&g, &pv.gpu, &pv.filter, &layers[1], level, r).unwrap();
        }
        let input = ComposeInput { layers: &layers, preview: Some(&pv), extra: None };
        check("resident preview", &input, w, h, 0);
    }
}

#[test]
fn advanced_blend_settings_explicitly_select_cpu_composition() {
    let mut value=layer(1,TileGrid::new(1,1));
    value.advanced_blend=Some(Arc::new(crate::engine::advanced_blend::AdvancedBlend{ranges:vec![],restricted_channels:vec![0]}));
    let mut layers=vec![value];
    assert!(!super::compose::supported(&ComposeInput{layers:&layers,preview:None,extra:None}));
    layers[0].advanced_blend=Some(Arc::new(Default::default()));
    assert!(super::compose::supported(&ComposeInput{layers:&layers,preview:None,extra:None}));
}
