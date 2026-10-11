//! `pixelforge --bench [side]`: headless engine timings on a huge canvas.

use std::time::Instant;

use crate::engine::brush::{BrushSettings, Stroke};
use crate::engine::composite::{ComposeInput, DT, Scratch, display_tile, flatten_layers};
use crate::engine::document::{Document, Layer};
use crate::engine::filters::Filter;
use crate::engine::grid::{RectI, TileGrid};
use crate::engine::mips::build_mips;
use crate::engine::ops::{self, Progress};
use crate::engine::px;
use crate::engine::selection::Selection;

thread_local! { static TARGETS: std::cell::Cell<Option<crate::bench_config::LatencyBudgets>> = const { std::cell::Cell::new(None) }; }

fn time<T>(label: &str, mp: f64, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    let s = t.elapsed().as_secs_f64();
    if mp > 0.0 {
        println!("{label:<44} {:>11.3} ms   {:>8.0} MP/s", s * 1000.0, mp / s);
    } else {
        println!("{label:<44} {:>11.3} ms", s * 1000.0);
    }
    if let Some(budget) = TARGETS.get().and_then(|b| b.target(label)) {
        println!("  latency target: {budget:.1} ms — {}", if s * 1000. <= budget { "PASS" } else { "OVER" });
    }
    r
}

pub fn run(width: u32, height: u32) {
    TARGETS.set(Some(crate::bench_config::LatencyBudgets::for_size(width, height)));
    let mp = width as f64 * height as f64 / 1e6;
    println!("PixelForge engine benchmark — {width} × {height} ({mp:.0} MP), {} threads\n", rayon::current_num_threads());

    let white = time("blank white canvas (shared tiles)", 0.0, || Document::new(width, height, Some(px::pack(255, 255, 255, 255))));
    let (live, _) = white.memory_tiles();
    println!("{:<44} {:>9} tiles ({:.2} MB)", "  memory for blank canvas", live, live as f64 * 0.25);

    let g = time("procedural fill (generate every pixel)", mp, || ops::procedural(width, height));
    let mips = time("mip pyramid build", mp, || build_mips(&g));
    println!("{:<44} {:>9} levels", "  mip levels", mips.len());

    let mut base = Layer::new(1, "base", g.clone());
    base.mips = mips;
    let mut paint = Layer::new(2, "paint", TileGrid::new(width, height));

    // Brush: a long stroke of 300 px soft dabs crossing the whole image.
    let settings = BrushSettings { size: 300.0, hardness: 0.3, opacity: 0.8, flow: 0.6, spacing: 0.1, ..Default::default() };
    let mut stroke = Stroke::brush(1, &paint.grid, settings, [220, 40, 60], false, None);
    let mut dirty = Vec::new();
    let n = 400;
    let dabs = time("brush stroke (400 events, 300 px soft)", 0.0, || {
        let mut count = 0;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let x = t * width as f32;
            let y = height as f32 * 0.5 + (t * 12.0).sin() * height as f32 * 0.3;
            let r = stroke.add_point(x, y, &mut paint.grid);
            count += r.len();
            dirty.extend(r);
        }
        count
    });
    println!("{:<44} {:>9} tile updates", "  touched", dabs);
    time("  incremental mip update for stroke", 0.0, || {
        crate::engine::mips::update_mips(&paint.grid, &mut paint.mips, &dirty)
    });

    let sel = Selection::ellipse(RectI::new(width as i32 / 8, height as i32 / 8, width as i32 * 7 / 8, height as i32 * 7 / 8));
    let p = Progress::default();
    time("hue/saturation (full image)", mp, || {
        ops::apply_filter(&g, &crate::engine::filters::hue_saturation(40.0, 20.0, 0.0), None, &p)
    });
    time("levels inside elliptical selection", mp, || {
        ops::apply_filter(&g, &Filter::Levels { black: 20.0, white: 230.0, gamma: 1.2 }, Some(&sel), &p)
    });
    time("gaussian blur r=20 (full image)", mp, || ops::apply_filter(&g, &Filter::GaussianBlur { radius: 20.0 }, None, &p));
    time("gaussian blur r=120 (full image)", mp, || ops::apply_filter(&g, &Filter::GaussianBlur { radius: 120.0 }, None, &p));
    time("unsharp mask (full image)", mp, || {
        ops::apply_filter(&g, &Filter::UnsharpMask { amount: 120.0, radius: 3.0, threshold: 2.0 }, None, &p)
    });
    time("linear gradient (full image)", mp, || {
        ops::gradient(&g, None, (0.0, 0.0), (width as f32, height as f32), [0, 0, 0], [255, 255, 255], false, 0.5)
    });
    time("flip horizontal", mp, || ops::flip_h(&g));
    time("rotate 90° clockwise", mp, || ops::rotate_cw(&g));
    time("resize to 50% (mip + bilinear)", mp, || ops::resize(&base, width / 2, height / 2));
    time("magic wand, contiguous (blank canvas)", mp, || ops::flood_mask(&white.layers[0].grid, 10, 10, 0, true));
    time("magic wand, global (procedural)", mp, || ops::flood_mask(&g, 10, 10, 30, false));

    paint.blend = crate::engine::blend::BlendMode::Multiply;
    let layers = vec![base, paint];
    time("flatten 2 layers (multiply blend)", mp, || flatten_layers(&layers, width, height));

    // Display compositing: a full 4K screen worth of tiles at several zoom levels.
    let input = ComposeInput { layers: &layers, preview: None, extra: None };
    for level in [0i32, 3, -2] {
        let tiles = 15 * 9;
        let label = format!("display: {tiles} screen tiles at mip level {level}");
        time(&label, 0.0, || {
            use rayon::prelude::*;
            (0..tiles).into_par_iter().for_each_init(Scratch::default, |s, i| {
                let mut buf = vec![0u8; DT * DT * 4];
                display_tile(&input, level, (i % 15) as i32, (i / 15) as i32, &mut buf, s);
            });
        });
    }
    let preview = crate::engine::composite::Preview::new(0, Filter::GaussianBlur { radius: 60.0 }, None);
    let input = ComposeInput { layers: &layers, preview: Some(&preview), extra: None };
    time("display: live blur r=60 preview at level 2", 0.0, || {
        use rayon::prelude::*;
        (0..135).into_par_iter().for_each_init(Scratch::default, |s, i| {
            let mut buf = vec![0u8; DT * DT * 4];
            display_tile(&input, 2, (i % 15) as i32, (i / 15) as i32, &mut buf, s);
        });
    });
    let path = std::env::temp_dir().join(format!("pixelforge-bench-{}-{width}x{height}.pforge", std::process::id()));
    time("native document save", mp, || crate::engine::io::save_native(&layers, width, height, 0, &Default::default(), &path)).expect("benchmark save");
    let loaded = time("native document open", mp, || crate::engine::io::load_native(&path)).expect("benchmark open");
    assert_eq!((loaded.0, loaded.1, loaded.3.len()), (width, height, layers.len()));
    std::fs::remove_file(path).expect("remove benchmark document");
    TARGETS.set(None);
}

/// `pixelforge --bench-filters [side]`: every catalog filter at default settings.
pub fn run_filters(width: u32, height: u32, only: Option<&str>) {
    use crate::engine::filters::catalog::{BuildCtx, Kind};
    let mp = width as f64 * height as f64 / 1e6;
    println!("Filter benchmark — {width} × {height} ({mp:.0} MP), {} threads, default parameters\n", rayon::current_num_threads());
    let g = ops::procedural(width, height);
    let layer = Layer::new(1, "bench", g.clone());
    let st = crate::engine::filters::stats(&layer);
    let ctx = BuildCtx { fg: [20, 20, 24], bg: [255, 255, 255], area: g.bounds(), stats: Some(&st), source: None, disp_map: None, layer_mask: None, doc_w: width, doc_h: height };
    let p = Progress::default();
    for k in Kind::all() {
        if k == Kind::Liquify {
            continue;
        }
        let e = k.entry();
        if only.is_some_and(|o| !e.title.to_lowercase().contains(&o.to_lowercase())) {
            continue;
        }
        let f = k.default_filter(&ctx);
        let label = format!("{:<12} {}", e.menu.title(), e.title);
        let reps: usize = std::env::var("PF_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        for _ in 1..reps {
            std::hint::black_box(ops::apply_filter(&g, &f, None, &p));
        }
        time(&label, mp, || ops::apply_filter(&g, &f, None, &p));
    }
}

// ----- --bench-tools: scripted tool gestures on a 268 MP document -------------------------

struct Budget {
    rows: Vec<(String, f64, f64)>,
}

impl Budget {
    fn check(&mut self, label: &str, ms: f64, budget: f64) {
        let ok = ms <= budget;
        println!("{label:<58} {ms:>9.2} ms   budget {budget:>7.1} ms   {}", if ok { "ok" } else { "OVER" });
        self.rows.push((label.to_string(), ms, budget));
    }

    fn summary(&self) {
        let over: Vec<_> = self.rows.iter().filter(|r| r.1 > r.2).collect();
        println!("\n{} checks, {} over budget", self.rows.len(), over.len());
        for r in over {
            println!("  OVER: {} ({:.2} ms > {:.1} ms)", r.0, r.1, r.2);
        }
    }
}

fn ms_of<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64() * 1000.0)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn p95(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() as f64 * 0.95) as usize).min(v.len() - 1)]
}

pub fn run_tools(width: u32, height: u32, only: Option<&str>) {
    use crate::engine::masks::SelOp;
    use crate::engine::segment;
    use crate::engine::selection::MaskGrid;
    let want = |k: &str| only.is_none_or(|o| k.contains(o));
    let mp = width as f64 * height as f64 / 1e6;
    println!("Tool benchmark — {width} × {height} ({mp:.0} MP), {} threads\n", rayon::current_num_threads());
    let (scene, gen_ms) = ms_of(|| ops::demo_scene(width, height));
    println!("{:<58} {gen_ms:>9.1} ms", "generate test scene");
    let base = Layer::new(1, "scene", scene.clone());
    let mut b = Budget { rows: vec![] };
    let (sx, sy) = (width as f32, height as f32);
    let (ballx, bally, ballr) = (0.30 * sx, 0.62 * sy, 0.14 * sy);

    if want("brush") {
        // A 500 px soft brush across the image, 300 mouse events.
        for (name, settings) in [
            ("brush 500 px soft", BrushSettings { size: 500.0, hardness: 0.3, spacing: 0.12, ..Default::default() }),
            ("brush 500 px, angle+roundness+texture+scatter", BrushSettings { size: 500.0, hardness: 0.5, angle: 30.0, roundness: 0.6, texture: 1, scatter: 0.2, ..Default::default() }),
            ("pencil 500 px aliased", BrushSettings { size: 500.0, hardness: 1.0, aliased: true, spacing: 0.25, ..Default::default() }),
            ("brush 500 px, every dynamic on (sampled tip, dual, colour per tip)", crate::engine::brush::all_dynamics(500.0, 7)),
        ] {
            let mut g = scene.clone();
            let mut st = Stroke::brush(1, &g, settings, [200, 30, 40], false, None);
            let mut times = Vec::new();
            for i in 0..300 {
                let t = i as f32 / 300.0;
                let (x, y) = (sx * (0.1 + 0.8 * t), sy * (0.5 + 0.2 * (t * 9.0).sin()));
                let (_, ms) = ms_of(|| st.add_point(x, y, &mut g));
                times.push(ms);
            }
            b.check(&format!("{name}: median per mouse event"), median(&mut times), 2.0);
            b.check(&format!("{name}: p95 per mouse event"), p95(&mut times), 2.0);
        }
    }

    if want("select") {
        let r1 = Selection::rect(RectI::new(width as i32 / 10, height as i32 / 10, width as i32 * 6 / 10, height as i32 * 6 / 10));
        let e1 = Selection::ellipse(RectI::new(width as i32 * 3 / 10, height as i32 * 3 / 10, width as i32 * 9 / 10, height as i32 * 9 / 10));
        let (m, ms) = ms_of(|| crate::engine::masks::combine_selection(Some(&r1), Some(e1.clone()), SelOp::Add, width, height));
        b.check("selection add (rect ∪ ellipse)", ms, 150.0);
        let sel = m.unwrap();
        let mask = MaskGrid::from_selection(&sel, width, height);
        for (label, f) in [
            ("Select ▸ Modify ▸ Expand 50 px", Box::new(|m: &MaskGrid| m.expand(50.0)) as Box<dyn Fn(&MaskGrid) -> MaskGrid>),
            ("Select ▸ Modify ▸ Contract 50 px", Box::new(|m: &MaskGrid| m.contract(50.0))),
            ("Select ▸ Modify ▸ Feather 30 px", Box::new(|m: &MaskGrid| m.feather(30.0))),
            ("Select ▸ Modify ▸ Border 40 px", Box::new(|m: &MaskGrid| m.border(40.0))),
            ("Select ▸ Modify ▸ Smooth 10 px", Box::new(|m: &MaskGrid| m.smooth(10))),
        ] {
            let (_, ms) = ms_of(|| f(&mask));
            b.check(label, ms, 1000.0);
        }
        let (_, ms) = ms_of(|| mask.to_gray());
        b.check("Quick Mask enter (selection → gray)", ms, 150.0);
        let gray = mask.to_gray();
        let (_, ms) = ms_of(|| MaskGrid::from_gray(&gray));
        b.check("Quick Mask exit (gray → selection)", ms, 300.0);
    }

    if want("wand") {
        let (_, ms) = ms_of(|| ops::flood_mask(&scene, 10, 10, 32, true));
        b.check("magic wand contiguous (sky)", ms, 1000.0);
        let paint = Layer::new(2, "paint", TileGrid::new(width, height));
        let layers = vec![base.clone(), paint];
        let (_, ms) = ms_of(|| {
            let src = ops::CompositeSource::new(&layers, width, height);
            ops::flood_mask_on(&src, 10, 10, 32, true)
        });
        b.check("magic wand Sample All Layers (2 layers)", ms, 1500.0);
    }

    if want("lasso") {
        use crate::engine::livewire::{WireParams, trace};
        use crate::engine::vector::p;
        // Trail ~12 px off the ball edge, events every ~6 px, anchors every ~250 px.
        let params = WireParams { width: 10.0, contrast: 0.1 };
        let mut times = Vec::new();
        let mut trail = vec![p(ballx + ballr, bally)];
        let n = 600;
        for k in 1..=n {
            let t = k as f32 / n as f32 * std::f32::consts::TAU * 0.25;
            let r = ballr + 6.0 + 4.0 * (t * 40.0).sin();
            trail.push(p(ballx + r * t.cos(), bally + r * t.sin()));
            let (path, ms) = ms_of(|| trace(&base, &trail, &params));
            times.push(ms);
            let len: f32 = path.windows(2).map(|w| w[0].dist(w[1])).sum();
            if len > 250.0 {
                let a = path[path.len() * 2 / 3];
                trail = vec![a, *trail.last().unwrap()];
            }
        }
        b.check("magnetic lasso: median per mouse move", median(&mut times), 4.0);
        b.check("magnetic lasso: p95 per mouse move", p95(&mut times), 4.0);
    }

    if want("quick") {
        let src = |l: u32, x: i32, y: i32, w: usize, h: usize, out: &mut [u32]| base.level(l).read_region_clamped(x, y, w, h, out);
        let mut q = segment::QuickSel::new(width, height, base.mips.len() as u32);
        let mut times = Vec::new();
        let mut last = None;
        for k in 0..12 {
            q.add_dab(ballx - ballr * 0.5 + k as f32 * ballr * 0.08, bally, ballr * 0.08);
            let (c, ms) = ms_of(|| q.solve(&src, width, height, None));
            times.push(ms);
            last = c;
        }
        b.check("quick selection: coarse result per stroke update", median(&mut times), 150.0);
        let c = last.unwrap();
        let (_, ms) = ms_of(|| segment::upsample(&c, width, height, Some(&src)));
        b.check("quick selection: full-res edge refinement", ms, 1000.0);
    }

    if want("object") {
        let src = |l: u32, x: i32, y: i32, w: usize, h: usize, out: &mut [u32]| base.level(l).read_region_clamped(x, y, w, h, out);
        let r = RectI::new((ballx - ballr * 1.3) as i32, (bally - ballr * 1.3) as i32, (ballx + ballr * 1.3) as i32, (bally + ballr * 1.3) as i32);
        let ml = base.mips.len() as u32;
        let (_, ms) = ms_of(|| segment::object_select(&src, width, height, ml, r, None, 0.12e6, false));
        b.check("object selection: first result (coarse GrabCut)", ms, 150.0);
        let (_, ms) = ms_of(|| segment::object_select(&src, width, height, ml, r, None, 1.0e6, true));
        b.check("object selection: refined (1 MP GrabCut + guided edge)", ms, 1000.0);
        // Select Subject (neural when models/select_subject.onnx exists).
        let ((_, backend), ms) = ms_of(|| crate::engine::objects::select_subject(&src, width, height, ml));
        b.check(&format!("Select Subject ({backend:?})"), ms, 800.0);
        // Object finder: background precompute, then per-move hover lookups.
        let (f, ms) = ms_of(|| crate::engine::objects::ObjectFinder::build(&src, width, height, ml));
        b.check(&format!("object finder precompute ({} objects, background)", f.objects.len()), ms, 2000.0);
        let mut times = Vec::new();
        let mut hits = 0;
        for i in 0..2000 {
            let t = i as f32 / 2000.0;
            let (x, y) = (sx * (0.05 + 0.9 * t), sy * (0.5 + 0.3 * (t * 13.0).sin()));
            let (h, ms) = ms_of(|| f.at(x, y));
            hits += h.is_some() as usize;
            times.push(ms);
        }
        b.check(&format!("object finder hover lookup ({hits}/2000 on an object): p95"), p95(&mut times), 16.0);
        let id = f.at(ballx, bally).unwrap_or(0);
        let (_, ms) = ms_of(|| f.mask(id, &src, width, height));
        b.check("object finder click → selection (guided upsample)", ms, 1000.0);
    }

    if want("layers") {
        use crate::engine::layers::{LayerKind, LayerMask, Paint, ShapeGeom, ShapeLayer};
        use crate::engine::vector::StrokeStyle;
        let mut top = Layer::new(2, "masked", ops::procedural(width, height));
        top.mask = Some(LayerMask::new(MaskGrid::from_selection(&Selection::ellipse(RectI::new(0, 0, width as i32, height as i32)), width, height)));
        let mut adj = Layer::new(3, "adj", TileGrid::new(width, height));
        adj.kind = LayerKind::Adjustment(crate::engine::filters::hue_saturation(40.0, 10.0, 0.0));
        let mut clipped = Layer::new(4, "clipped", TileGrid::filled(width, height, px::pack(255, 0, 0, 128)));
        clipped.clip = true;
        let layers = vec![base.clone(), top, clipped, adj];
        for level in [0i32, 3] {
            let input = ComposeInput { layers: &layers, preview: None, extra: None };
            let (_, ms) = ms_of(|| {
                use rayon::prelude::*;
                (0..135).into_par_iter().for_each_init(Scratch::default, |sc, i| {
                    let mut buf = vec![0u8; DT * DT * 4];
                    display_tile(&input, level, (i % 15) as i32 + 8, (i / 15) as i32 + 8, &mut buf, sc);
                });
            });
            b.check(&format!("display: 135 tiles, mask + clip + adjustment, level {level}"), ms, 26.0);
        }
        // Live shape resize: re-rasterize only the tiles the old ∪ new bounds touch.
        let mut g = TileGrid::new(width, height);
        let mut times = Vec::new();
        let mut old = None;
        // Preserve the legacy fixture wherever it fits; scale only the small rectangular tiers.
        let (shape_x, shape_y) = (if width >= 7200 { 4000. } else { sx * 0.25 }, if height >= 6400 { 4000. } else { sy * 0.25 });
        for k in 0..60 {
            let e = 2000.0 + k as f32 * 20.0;
            let shape = ShapeLayer::new(ShapeGeom::Rect { x0: shape_x, y0: shape_y, x1: shape_x + e, y1: shape_y + e * 0.75, radii: [80.0; 4] }, Paint::Solid([30, 120, 200]), Some((Paint::Solid([0, 0, 0]), StrokeStyle { width: 12.0, ..Default::default() })));
            let kind = LayerKind::Shape(std::sync::Arc::new(shape));
            let (_, ms) = ms_of(|| crate::engine::layers::rerender(&mut g, old, &kind));
            old = kind.content_bounds();
            times.push(ms);
        }
        b.check("shape live resize (≈2000–3200 px rounded rect + stroke)", median(&mut times), 8.0);
    }
    if want("retouch") {
        use crate::engine::retouch::{self, RegionFn};
        use std::sync::Arc;
        let g0 = scene.clone();
        let src: Arc<RegionFn> = Arc::new(move |x, y, w, h, out: &mut [u32]| g0.read_region(x, y, w, h, out));
        let reps: usize = std::env::var("PF_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        for _ in 1..reps {
            let mut g = scene.clone();
            let mut stroke = Stroke::new(0, &g, BrushSettings { size: 500.0, hardness: 0.6, spacing: 0.25, ..Default::default() }, Box::new(retouch::HealKernel::new(src.clone(), -900, 40, Default::default())), None);
            for i in 0..120 {
                let t = i as f32 / 120.0;
                stroke.add_point(sx * (0.35 + 0.3 * t), sy * 0.72 + (t * 20.0).sin() * 200.0, &mut g);
            }
        }
        use crate::engine::poisson::Quality;
        for (label, size, heal, q) in [
            ("clone stamp 500 px", 500.0, false, Quality::Fast),
            ("healing brush 100 px", 100.0, true, Quality::Fast),
            ("healing brush 200 px", 200.0, true, Quality::Fast),
            ("healing brush 300 px", 300.0, true, Quality::Fast),
            ("healing brush 500 px", 500.0, true, Quality::Fast),
            ("healing brush 500 px, Quality: Exact", 500.0, true, Quality::Exact),
        ] {
            let mut g = scene.clone();
            let k: Box<dyn crate::engine::brush::Kernel> = if heal { Box::new(retouch::HealKernel::new(src.clone(), -900, 40, q)) } else { Box::new(retouch::CloneKernel { src: src.clone(), dx: -900, dy: 40 }) };
            let st = BrushSettings { size, hardness: 0.6, spacing: 0.25, ..Default::default() };
            let mut stroke = Stroke::new(0, &g, st, k, None);
            let mut times = Vec::new();
            for i in 0..120 {
                let t = i as f32 / 120.0;
                let (x, y) = (sx * (0.35 + 0.3 * t), sy * 0.72 + (t * 20.0).sin() * 200.0);
                let (_, ms) = ms_of(|| stroke.add_point(x, y, &mut g));
                times.push(ms);
            }
            let per_dab = times.iter().sum::<f64>() / stroke.dab_count().max(1) as f64;
            b.check(&format!("{label}: per dab ({} dabs)", stroke.dab_count()), per_dab, if q == Quality::Exact { 16.0 } else { 8.0 });
            b.check(&format!("{label}: p95 per mouse event"), p95(&mut times), if heal { 16.0 } else { 2.0 });
        }
        // Spot Healing: a 300 px blemish on the ground texture.
        let (cx, cy) = (sx * 0.5, sy * 0.85);
        let mut cov = MaskGrid::new(width, height);
        cov = cov.combine(&MaskGrid::from_selection(&Selection::ellipse(RectI::new((cx - 150.0) as i32, (cy - 150.0) as i32, (cx + 150.0) as i32, (cy + 150.0) as i32)), width, height), SelOp::Add);
        let (_, ms) = ms_of(|| retouch::spot_heal(&scene, &cov, None, retouch::SpotMode::ContentAware));
        b.check("spot healing: 300 px blemish (PatchMatch + multigrid)", ms, 60.0);
        let (_, ms) = ms_of(|| retouch::spot_heal(&scene, &cov, None, retouch::SpotMode::CreateTexture));
        b.check("spot healing, Create Texture: 300 px (quilting + multigrid)", ms, 60.0);
        // Edit ▸ Content-Aware Fill workspace (Auto sampling, Default colour
        // adaptation): the same 300 px spot and a 1 MP ellipse.
        let adapt = crate::engine::patchmatch::Adapt { color: crate::engine::patchmatch::ColorAdapt::Default, ..Default::default() };
        let samp = retouch::caf_sampling(&scene, &cov, retouch::SamplingArea::Auto);
        let (_, ms) = ms_of(|| retouch::content_aware_fill(&scene, &cov, &samp, adapt));
        b.check("Content-Aware Fill workspace: 300 px (Auto sampling)", ms, 150.0);
        let big = MaskGrid::from_selection(&Selection::ellipse(RectI::new((sx * 0.45) as i32, (sy * 0.75) as i32, (sx * 0.45) as i32 + 1000, (sy * 0.75) as i32 + 1000)), width, height);
        let samp = retouch::caf_sampling(&scene, &big, retouch::SamplingArea::Auto);
        let (_, ms) = ms_of(|| retouch::content_aware_fill(&scene, &big, &samp, adapt));
        b.check("Content-Aware Fill workspace: 1 MP (Auto sampling)", ms, 600.0);
        let rot = crate::engine::patchmatch::Adapt { rotation: crate::engine::patchmatch::RotationAdapt::Medium, scale: true, mirror: true, ..adapt };
        let (_, ms) = ms_of(|| retouch::content_aware_fill(&scene, &big, &samp, rot));
        b.check("Content-Aware Fill workspace: 1 MP, rotation Medium + scale + mirror", ms, 2000.0);
        // Patch / content-aware move: a 1 MP (1000 × 1000) region.
        let reg = RectI::new((sx * 0.45) as i32, (sy * 0.75) as i32, (sx * 0.45) as i32 + 1000, (sy * 0.75) as i32 + 1000);
        let m = MaskGrid::from_selection(&Selection::ellipse(reg), width, height);
        let (_, ms) = ms_of(|| retouch::patch(&scene, &m, 1400, 0, retouch::PatchMode::ContentAware));
        b.check("patch (content-aware), 1 MP region", ms, 400.0);
        let (_, ms) = ms_of(|| retouch::patch(&scene, &m, 1400, 0, retouch::PatchMode::Source));
        b.check("patch (normal, source), 1 MP region", ms, 400.0);
        let (_, ms) = ms_of(|| retouch::content_aware_move(&scene, &m, 1400, -300, false, 0.6));
        b.check("content-aware move, 1 MP object", ms, 400.0);
        // A 60 px red pupil with a catch-light, painted onto the scene.
        let mut eye = scene.clone();
        let (ex, ey) = (sx * 0.15, sy * 0.2);
        let mut st = Stroke::brush(0, &eye, BrushSettings { size: 60.0, hardness: 0.9, ..Default::default() }, [200, 25, 30], false, None);
        st.add_point(ex, ey, &mut eye);
        let mut st = Stroke::brush(0, &eye, BrushSettings { size: 10.0, hardness: 1.0, ..Default::default() }, [255, 255, 255], false, None);
        st.add_point(ex + 8.0, ey - 8.0, &mut eye);
        let (_, ms) = ms_of(|| retouch::red_eye(&eye, ex as i32, ey as i32, 1.0, 0.5));
        b.check("red eye (60 px pupil: detect + fix)", ms, 30.0);
        // History brush from a packed history state.
        let mut doc = crate::engine::document::Document::from_grid(scene.clone(), "scene");
        doc.push_undo("filter");
        let inv = ops::apply_filter(&doc.layers[0].grid, &Filter::Invert, None, &Progress::default());
        doc.set_layer_grid(0, inv);
        let hs = doc.history_source(0, doc.layers[0].id).unwrap();
        let mut g = doc.layers[0].grid.clone();
        let mut stroke = Stroke::new(0, &g, BrushSettings { size: 500.0, hardness: 0.5, ..Default::default() }, Box::new(retouch::HistoryKernel { src: hs }), None);
        let mut times = Vec::new();
        for i in 0..120 {
            let t = i as f32 / 120.0;
            let (_, ms) = ms_of(|| stroke.add_point(sx * (0.2 + 0.6 * t), sy * 0.3, &mut g));
            times.push(ms);
        }
        b.check("history brush 500 px: median per mouse event", median(&mut times), 2.0);
    }
    if want("paint") {
        use crate::engine::paint::{self, Limits, Range, ReplaceMode, Sampling, Tone};
        let mk = |name: &str| -> Box<dyn crate::engine::brush::Kernel> {
            match name {
                "dodge" => Box::new(paint::ToneKernel::new(Tone::Dodge, Range::Midtones, 0.5, true, false)),
                "burn" => Box::new(paint::ToneKernel::new(Tone::Burn, Range::Shadows, 0.5, true, false)),
                "sponge" => Box::new(paint::ToneKernel::new(Tone::Saturate, Range::Midtones, 0.5, false, true)),
                "blur" => Box::new(paint::FocusKernel::new(&scene, 0.5, false)),
                "sharpen" => Box::new(paint::FocusKernel::new(&scene, 0.5, true)),
                "smudge" => Box::new(paint::SmudgeKernel::new(0.5, None)),
                "mixer" => Box::new(paint::MixerKernel::new(Some(px::pack(200, 40, 40, 255)), 0.5, 0.5, 0.5, 1.0)),
                "color replacement" => Box::new(paint::ReplaceKernel::new([0, 0, 255], [255; 3], ReplaceMode::Color, Sampling::Continuous, 40.0)),
                "color replacement, Limits: Contiguous" => Box::new(paint::ReplaceKernel::new([0, 0, 255], [255; 3], ReplaceMode::Color, Sampling::Continuous, 40.0).with_limits(Limits::Contiguous)),
                "color replacement, Limits: Find Edges" => Box::new(paint::ReplaceKernel::new([0, 0, 255], [255; 3], ReplaceMode::Color, Sampling::Continuous, 40.0).with_limits(Limits::FindEdges)),
                _ => Box::new(paint::BgEraseKernel::new(Sampling::Continuous, 40.0, [255; 3], None)),
            }
        };
        for name in ["dodge", "burn", "sponge", "blur", "sharpen", "smudge", "mixer", "color replacement", "color replacement, Limits: Contiguous", "color replacement, Limits: Find Edges", "background eraser"] {
            let mut g = scene.clone();
            let spacing = if matches!(name, "smudge" | "mixer") { 0.1 } else { 0.25 };
            let mut st = Stroke::new(0, &g, BrushSettings { size: 500.0, hardness: 0.5, spacing, ..Default::default() }, mk(name), None);
            let mut times = Vec::new();
            for i in 0..200 {
                let t = i as f32 / 200.0;
                let (x, y) = (sx * (0.1 + 0.8 * t), sy * (0.55 + 0.2 * (t * 7.0).sin()));
                let (_, ms) = ms_of(|| st.add_point(x, y, &mut g));
                times.push(ms);
            }
            b.check(&format!("{name} 500 px: p95 per mouse event"), p95(&mut times), 2.0);
        }
        let paint_layer = Layer::new(2, "p", TileGrid::new(width, height));
        let layers = vec![base.clone(), paint_layer];
        let (_, ms) = ms_of(|| {
            let src = ops::CompositeSource::new(&layers, width, height);
            let m = ops::flood_mask_on(&src, 10, 10, 32, true).feather(0.55);
            paint::erase_mask(&scene, &m, None, 1.0)
        });
        b.check("magic eraser (sky, sample all layers, AA)", ms, 1000.0);
        let (_, ms) = ms_of(|| {
            let m = ops::flood_mask(&scene, 10, 10, 32, true);
            paint::fill_mask_pattern(&scene, &m, None, 1, 1.0)
        });
        b.check("paint bucket with pattern (sky)", ms, 1000.0);
        use crate::engine::gradient::{GradKind, Gradient, GradientFill};
        for kind in GradKind::ALL {
            let f = Filter::Gradient(std::sync::Arc::new(GradientFill::new(Gradient::presets([0; 3], [255; 3])[3].clone(), kind, (sx * 0.3, sy * 0.3), (sx * 0.7, sy * 0.6), false, true, 1.0)));
            let (_, ms) = ms_of(|| ops::apply_filter(&scene, &f, None, &Progress::default()));
            b.check(&format!("gradient {} (spectrum, dither)", kind.label()), ms, 400.0);
        }
        let f = Filter::Gradient(std::sync::Arc::new(GradientFill::new(Gradient::two("bw", [0; 3], [255; 3]), GradKind::Radial, (sx * 0.3, sy * 0.3), (sx * 0.7, sy * 0.6), false, true, 1.0)));
        let preview = crate::engine::composite::Preview::new(0, f, None);
        let layers = vec![base.clone()];
        let input = ComposeInput { layers: &layers, preview: Some(&preview), extra: None };
        let (_, ms) = ms_of(|| {
            use rayon::prelude::*;
            (0..135).into_par_iter().for_each_init(Scratch::default, |sc, i| {
                let mut buf = vec![0u8; DT * DT * 4];
                display_tile(&input, 6, (i % 15) as i32, (i / 15) as i32, &mut buf, sc);
            });
        });
        b.check("live gradient preview: 135 display tiles", ms, 26.0);
    }
    if want("vector") {
        use crate::engine::layers::{self, LayerKind, Paint, ShapeGeom, ShapeLayer};
        use crate::engine::vector::{self, FillRule, Path, StrokeStyle};
        // Live shape resize: an ellipse with a stroke dragged from 1000 to 9000 px.
        let mut g = TileGrid::new(width, height);
        let mk = |r: f32| {
            LayerKind::Shape(std::sync::Arc::new(ShapeLayer::new(ShapeGeom::Ellipse { x0: sx * 0.1, y0: sy * 0.1, x1: sx * 0.1 + r, y1: sy * 0.1 + r * 0.7 }, Paint::Solid([200, 60, 40]), Some((Paint::Solid([20, 20, 20]), StrokeStyle { width: 12.0, ..Default::default() })))))
        };
        let mut old = None;
        let mut times = Vec::new();
        for i in 0..60 {
            let k = mk(1000.0 + 8000.0 * i as f32 / 59.0);
            let (_, ms) = ms_of(|| layers::rerender(&mut g, old, &k));
            old = k.content_bounds();
            times.push(ms);
        }
        b.check("shape resize (stroked ellipse to 9000 px): p95/frame", p95(&mut times), 8.0);
        let mut times = Vec::new();
        for i in 0..60 {
            let r = 1000.0 + 8000.0 * i as f32 / 59.0;
            let k = LayerKind::Shape(std::sync::Arc::new(ShapeLayer::new(ShapeGeom::Polygon { x0: sx * 0.1, y0: sy * 0.1, x1: sx * 0.1 + r, y1: sy * 0.1 + r, sides: 7, star: 0.4 }, Paint::Solid([40, 90, 200]), None)));
            let (_, ms) = ms_of(|| layers::rerender(&mut g, old, &k));
            old = k.content_bounds();
            times.push(ms);
        }
        b.check("shape resize (7-point star to 9000 px): p95/frame", p95(&mut times), 8.0);
        // Path operations on two 500-anchor smooth paths (Catmull-Rom anchors
        // on noisy circles, ~6000 flattened segments each).
        {
            use crate::engine::layers::{GradPaint, PathOp, ShapePart};
            use crate::engine::pathops;
            let (pa, pb) = (vector::noisy_loop(2500.0, 2500.0, 2000.0, 1), vector::noisy_loop(2650.0, 2600.0, 2000.0, 2));
            let (fa, fb) = (pa.flatten(0.1), pb.flatten(0.1));
            for op in [PathOp::Combine, PathOp::Subtract, PathOp::Intersect, PathOp::Exclude] {
                let bop = match op {
                    PathOp::Combine => pathops::BoolOp::Union,
                    PathOp::Subtract => pathops::BoolOp::Subtract,
                    PathOp::Intersect => pathops::BoolOp::Intersect,
                    PathOp::Exclude => pathops::BoolOp::Xor,
                };
                let mut times: Vec<f64> = (0..15).map(|_| ms_of(|| pathops::boolean(&fa, FillRule::NonZero, &fb, FillRule::NonZero, bop)).1).collect();
                let (res, _) = ms_of(|| pathops::boolean(&fa, FillRule::NonZero, &fb, FillRule::NonZero, bop));
                b.check(&format!("path op {} on two 500-anchor paths ({}+{} segs, area {:.0}) median", op.label(), fa[0].pts.len(), fb[0].pts.len(), pathops::area(&res)), median(&mut times), 8.0);
            }
            // A live two-component shape with a gradient fill, re-rendered as
            // its second component moves (each frame redoes the boolean).
            let mut g = TileGrid::new(width, height);
            let mut gr = crate::engine::gradient::Gradient::two("g", [255, 120, 0], [40, 0, 160]);
            gr.sort();
            let base = ShapeLayer::new(ShapeGeom::Path(pa.clone()), Paint::Gradient(GradPaint { grad: gr, kind: crate::engine::gradient::GradKind::Linear, angle: 30.0, scale: 1.0, reverse: false }), Some((Paint::Solid([0, 0, 0]), StrokeStyle { width: 8.0, dashes: vec![3.0, 2.0], ..Default::default() })));
            let mut old = None;
            let mut times = Vec::new();
            for i in 0..30 {
                let mut sh = base.clone();
                let mut part = ShapePart::new(ShapeGeom::Path(pb.clone()), PathOp::Subtract);
                part.geom = part.geom.translated(i as f32 * 20.0, 0.0);
                sh.parts.push(part);
                let k = LayerKind::Shape(std::sync::Arc::new(sh));
                let (_, ms) = ms_of(|| layers::rerender(&mut g, old, &k));
                old = k.content_bounds();
                times.push(ms);
            }
            b.check("live Subtract (2×500 anchors, gradient + dashed stroke), p95/frame", p95(&mut times), 16.0);
            let mut live = base.clone();
            live.parts.push(ShapePart::new(ShapeGeom::Path(pb.clone()), PathOp::Exclude));
            let (_, ms) = ms_of(|| live.merged());
            b.check("Merge Shape Components (2×500 anchors)", ms, 8.0);
            // Corner-radius widget drag: a 4000 px rounded rect, rotated 20°.
            let mut times = Vec::new();
            let mut old = None;
            for i in 0..40 {
                let mut sh = ShapeLayer::new(ShapeGeom::Rect { x0: 3000.0, y0: 3000.0, x1: 7000.0, y1: 6000.0, radii: [i as f32 * 25.0; 4] }, Paint::Solid([30, 120, 200]), None);
                sh.rot = 20f32.to_radians();
                let k = LayerKind::Shape(std::sync::Arc::new(sh));
                let (_, ms) = ms_of(|| layers::rerender(&mut g, old, &k));
                old = k.content_bounds();
                times.push(ms);
            }
            b.check("corner radius widget drag (rotated 4000 px rect): p95/event", p95(&mut times), 8.0);
            // Frame: move its placed content (4000 × 3000 image in a 3000 px frame).
            let img = crate::engine::ops::procedural(4000, 3000);
            let mut fr = crate::engine::frame::FrameLayer::new(ShapeGeom::Ellipse { x0: 2000.0, y0: 2000.0, x1: 5000.0, y1: 5000.0 });
            let so = crate::engine::layers::SmartObject::new("img", img, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            fr.content = Some(std::sync::Arc::new(fr.fitted(&so, true)));
            let mut times = Vec::new();
            let mut old = None;
            for i in 0..20 {
                let mut m = fr.content.as_ref().unwrap().m;
                m[2] += i as f32 * 15.0;
                let k = LayerKind::Frame(std::sync::Arc::new(fr.with_content_matrix(m)));
                let (_, ms) = ms_of(|| layers::rerender(&mut g, old, &k));
                old = k.content_bounds();
                times.push(ms);
            }
            b.check("frame: move placed content (3000 px frame): p95/event", p95(&mut times), 26.0);
        }
        // Paths panel: Make Selection and Stroke Path with the Brush.
        let path = Path::ellipse(sx * 0.5, sy * 0.5, sx * 0.3, sy * 0.2);
        let (_, ms) = ms_of(|| vector::rasterize_mask(&path.flatten(0.1), width, height, FillRule::NonZero));
        b.check("path → selection (ellipse at 60% width × 40% height)", ms, 200.0);
        let small = Path::star(sx * 0.5, sy * 0.5, 1500.0, 1500.0, 5, 0.5, 0.0);
        let mut g = scene.clone();
        let (_, ms) = ms_of(|| {
            let settings = BrushSettings { size: 30.0, hardness: 0.8, ..Default::default() };
            for pl in small.flatten(0.25) {
                let mut st = Stroke::brush(0, &g, settings, [255, 0, 0], false, None);
                // The dab engine spaces dabs along each segment itself.
                for q in pl.pts.iter().chain(pl.closed.then(|| &pl.pts[0])) {
                    st.add_point(q.x, q.y, &mut g);
                }
                st.finish(&mut g);
            }
        });
        b.check(&format!("stroke path (star, {:.0} px long, 30 px brush)", small.length()), ms, 100.0);
        // Freeform pen: fitting 2000 sampled points.
        let pts: Vec<_> = (0..2000).map(|i| {
            let t = i as f32 / 2000.0 * 6.28;
            vector::p(sx * 0.5 + 900.0 * t.cos() + 40.0 * (t * 13.0).sin(), sy * 0.5 + 600.0 * t.sin())
        }).collect();
        let (c, ms) = ms_of(|| vector::fit_curve(&pts, 1.0));
        b.check(&format!("freeform pen fit (2000 pts → {} cubics)", c.len()), ms, 30.0);
    }
    if want("web") {
        use crate::engine::{annot, arrange, artboard, slices, webexport, xform};
        use rayon::prelude::*;
        use crate::engine::document::Layer;
        use crate::engine::layers::LayerKind;
        // 20 layers with a 600 px blob each, spread over the canvas.
        let mut layers: Vec<Layer> = Vec::new();
        for k in 0..20u64 {
            let (x, y) = (500 + (k as i32 * 977) % (width as i32 - 1500).max(1), 400 + (k as i32 * 1531) % (height as i32 - 1500).max(1));
            let sel = crate::engine::selection::Selection::ellipse(RectI::new(x, y, x + 600, y + 450));
            let g = crate::engine::ops::fill(&TileGrid::new(width, height), Some(&sel), [(k * 12) as u8, 90, 200], 1.0);
            layers.push(Layer::new(k + 1, format!("L{k}"), g));
        }
        let rects: Vec<RectI> = layers.iter().map(|l| xform::content_rect(&l.grid).unwrap()).collect();
        let (_, ms) = ms_of(|| {
            let d = arrange::align(&rects, arrange::Align::Left, None, Some(3));
            let m = |o: (i32, i32)| [1.0, 0.0, o.0 as f32, 0.0, 1.0, o.1 as f32, 0.0, 0.0, 1.0];
            let out: Vec<Layer> = layers.par_iter().zip(&d).map(|(l, &o)| xform::layer_transformed(l, &m(o), width, height)).collect();
            out.len()
        });
        b.check("align 20 layers to a key object (document canvas)", ms, 50.0);
        let (_, ms) = ms_of(|| {
            let d = arrange::distribute(&rects, arrange::Distribute::HSpace);
            let m = |o: (i32, i32)| [1.0, 0.0, o.0 as f32, 0.0, 1.0, o.1 as f32, 0.0, 0.0, 1.0];
            let out: Vec<Layer> = layers.par_iter().zip(&d).map(|(l, &o)| xform::layer_transformed(l, &m(o), width, height)).collect();
            out.len()
        });
        b.check("distribute 20 layers (equal horizontal spacing)", ms, 50.0);
        let mut times = Vec::new();
        for k in 0..200 {
            let (x, y) = ((k * 7919) % width as i32, (k * 104729) % height as i32);
            let (_, ms) = ms_of(|| (0..layers.len()).rev().find(|&i| crate::engine::px::alpha(layers[i].grid.pixel(x, y)) > 0));
            times.push(ms);
        }
        b.check("Auto-Select hit test over 20 layers (p95)", p95(&mut times), 1.0);
        // Artboards: a 1920 × 1080 board holding a photo-sized layer.
        let mut ab = Layer::new(100, "Web 1920", TileGrid::new(width, height));
        ab.kind = LayerKind::Artboard(std::sync::Arc::new(artboard::ArtboardLayer { rect: RectI::new(1000, 1000, 2920, 2080), bg: Some([255, 255, 255]), collapsed: false }));
        let mut child = Layer::new(101, "photo", scene.clone());
        child.parent = Some(100);
        let boards = vec![child, ab];
        let (_, ms) = ms_of(|| artboard::render(&boards, 1).map(|r| r.1));
        b.check("artboard export render (1920 × 1080, one child)", ms, 50.0);
        let pages: Vec<(Vec<u8>, u32, u32)> = (0..4).map(|_| artboard::render(&boards, 1).unwrap()).collect();
        let (pdf, ms) = ms_of(|| webexport::pdf(&pages, 72.0, Some(90)).unwrap());
        b.check(&format!("artboards to PDF, 4 pages 1920 × 1080 ({:.1} MB)", pdf.len() as f32 / 1e6), ms, 150.0);
        // Save for Web: the dog photo (3840 × 2560) cut into 12 user slices.
        let photo = crate::engine::io::open_image(std::path::Path::new("testdata/photos/dog_grass.jpg")).unwrap_or_else(|_| crate::engine::ops::demo_scene(3840, 2560));
        let (w, h) = (photo.w, photo.h);
        let doc = vec![Layer::new(1, "bg", photo)];
        let user: Vec<annot::Slice> = slices::divide(RectI::new(0, 0, w as i32, h as i32), 4, 3).into_iter().enumerate().map(|(i, r)| annot::Slice::new(format!("s{i:02}"), r)).collect();
        for (name, fmt) in [("PNG-8 128 dithered", webexport::WebFormat::Png8 { colors: 128, dither: true, transparency: false }), ("PNG-24", webexport::WebFormat::Png24 { transparency: false }), ("JPEG q60", webexport::WebFormat::Jpeg { quality: 60 }), ("WebP lossless", webexport::WebFormat::Webp)] {
            let (enc, ms) = ms_of(|| slices::encode_all(&doc, w, h, &user, fmt, [255, 255, 255], "page").unwrap());
            let bytes: usize = enc.iter().map(|e| e.bytes.len()).sum();
            b.check(&format!("Save for Web {name}: 12 slices of the 3840 × 2560 photo ({:.1} MB)", bytes as f32 / 1e6), ms, 1000.0);
        }
    }
    if want("type") {
        use crate::engine::text::{self, TextLayer, TextStyle};
        text::warm_up();
        let para = "The quick brown fox jumps over the lazy dog. Sphinx of black quartz, judge my vow! ".repeat(24);
        let mut g = TileGrid::new(width, height);
        // Create: layout + 50 restyled ranges + render, as Photoshop's "make
        // text layer" with 51 style runs; best of two like the jsx timings.
        // A different first word per run, so nothing comes from the shaping cache.
        let mut create = |first: &str| {
            let mut t = TextLayer::new(para.replacen("The", first, 1), TextStyle { size: 36.0, ..Default::default() }, (sx * 0.1, sy * 0.1), Some((2400.0, 3000.0)), false);
            // 50 style runs: alternate colour / size / tracking every ~40 bytes.
            for k in 0..25 {
                let a = k * 80 + 20;
                t.set_char_style(a, a + 40, |st| {
                    st.color = [200, 20, 40];
                    st.size = 44.0;
                    st.tracking = 50.0;
                });
            }
            t.relayout();
            let b0 = t.bounds();
            t.render(&mut g, b0);
            t
        };
        let (_, cold) = ms_of(|| create("One"));
        let (mut t, ms) = ms_of(|| create("Two"));
        println!("{:<58} {:>7.2} ms (font database load included)", "text: create, first run", cold);
        println!("{:<58} {:>9} glyphs, {} runs", "paragraph", t.layout.glyphs.len(), t.runs.len());
        b.check("text: create a 2000-glyph layer with 51 runs (PS 74.5 ms)", ms, 75.0);
        let mut caret = para.len() / 2;
        let mut times = Vec::new();
        for i in 0..80 {
            let old = t;
            let (n, ms) = ms_of(|| {
                let mut n = old.clone();
                if i % 5 == 4 {
                    n.delete(caret - 1, caret);
                    caret -= 1;
                } else {
                    n.insert(caret, "x", None);
                    caret += 1;
                }
                n.relayout();
                n.rerender_edit(&mut g, &old);
                n
            });
            times.push(ms);
            t = n;
        }
        b.check("text: keystroke, 2000-glyph paragraph with 51 runs (p95; PS 59.9 ms)", p95(&mut times), 4.0);
        // Restyling a 200-byte range (split runs, relayout, re-render).
        let mut times = Vec::new();
        for i in 0..20 {
            let old = t;
            let (n, ms) = ms_of(|| {
                let mut n = old.clone();
                let a = 300 + i * 50;
                n.set_char_style(a, a + 200, |st| {
                    st.underline = i % 2 == 0;
                    st.color = [20, 60, (i * 10) as u8];
                });
                n.relayout();
                n.rerender_edit(&mut g, &old);
                n
            });
            times.push(ms);
            t = n;
        }
        b.check("text: apply a style to a 200-char range (p95; PS 54.4 ms)", p95(&mut times), 8.0);
        // Warped text: full re-render with every glyph through the patch.
        let mut w = t.clone();
        w.warp = text::Warp { style: text::WarpStyle::Flag, bend: 0.5, h_distort: 0.2, v_distort: 0.0, vertical: false };
        let bw = w.bounds();
        let (_, ms) = ms_of(|| w.render(&mut g, bw));
        b.check("text: render the paragraph warped (Flag)", ms, 150.0);
    }
    if want("nav") {
        use crate::engine::{annot, xform};
        let layers = vec![base.clone()];
        for side_ in [1u32, 11, 101] {
            let mut times = Vec::new();
            for k in 0..20 {
                let (_, ms) = ms_of(|| annot::box_average(&layers, width, height, sx * (0.1 + 0.04 * k as f32), sy * 0.5, side_));
                times.push(ms);
            }
            b.check(&format!("eyedropper / color sampler {side_}×{side_} average (median)"), median(&mut times), 4.0);
        }
        let (r, ms) = ms_of(|| xform::content_rect(&scene));
        b.check(&format!("move tool: layer content bounds ({:?})", r.map(|r| (r.w(), r.h()))), ms, 30.0);
        // Best of two runs, like tools/photoshop_tools.jsx (the first run
        // also pays for page-faulting fresh memory).
        let best2 = |f: &dyn Fn()| (0..2).map(|_| ms_of(f).1).fold(f64::MAX, f64::min);
        let crop = |dx: f32, dy: f32, w: u32, h: u32| xform::canvas_transformed(&layers, &[], &Default::default(), &[1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0, 1.0], w, h);
        let ms = best2(&|| drop(crop(1024.0, 512.0, width.saturating_sub(2048).max(1), height.saturating_sub(2048).max(1))));
        b.check("crop (tile-aligned)", ms, 400.0);
        let ms = best2(&|| drop(crop(1000.0, 517.0, width.saturating_sub(2048).max(1), height.saturating_sub(2048).max(1))));
        b.check("crop (unaligned)", ms, 1000.0);
        let (sn, cs) = 3.5f32.to_radians().sin_cos();
        let k = (sx / (sx * cs + sy * sn)).min(sy / (sy * cs + sx * sn)).min(1.0);
        let (ow, oh) = ((sx * k).round() as u32, (sy * k).round() as u32);
        let rot = annot::rotation_about(3.5, sx * 0.5, sy * 0.5);
        let m = crate::engine::filters::distort::mat_mul(&[1.0, 0.0, (ow as f32 - sx) * 0.5, 0.0, 1.0, (oh as f32 - sy) * 0.5, 0.0, 0.0, 1.0], &rot);
        let ms = best2(&|| drop(xform::canvas_transformed(&layers, &[], &Default::default(), &m, ow, oh)));
        b.check("straighten 3.5° (rotate + crop)", ms, 2000.0);
        let q = [(sx * 0.1, sy * 0.12), (sx * 0.85, sy * 0.05), (sx * 0.95, sy * 0.9), (sx * 0.05, sy * 0.8)];
        let (m, pw, ph) = annot::perspective_crop(q);
        let ms = best2(&|| drop(xform::canvas_transformed(&layers, &[], &Default::default(), &m, pw, ph)));
        b.check(&format!("perspective crop → {pw} × {ph}"), ms, 3000.0);
        let m = [0.9, 0.0, sx * 0.05, 0.0, 0.9, sy * 0.05, 0.0, 0.0, 1.0];
        let ms = best2(&|| drop(xform::layer_transformed(&base, &m, width, height)));
        b.check("free transform: scale layer 90%", ms, 2000.0);
    }
    if want("vanish") {
        use crate::engine::vanish::{self, DabSource, Plane, Scene};
        // A floor plane over most of a 4K frame.
        let g = ops::demo_scene(3840, 2160);
        let mut sc = Scene::new(3840, 2160);
        sc.planes.push(Plane { c: [(1100.0, 900.0), (2800.0, 860.0), (3700.0, 2100.0), (200.0, 2150.0)] });
        let f = sc.frame(0);
        let (_, ms) = ms_of(|| sc.tear_off(0, 0, 600.0));
        b.check("Vanishing Point: tear off a perpendicular plane", ms, 1.0);
        let mut times = Vec::new();
        for k in 0..40 {
            let (_, ms) = ms_of(|| vanish::dab(&g, &g, &f, f.sw * (0.2 + k as f32 * 0.015), f.sh * 0.8, 100.0, 0.5, DabSource::Color([200, 30, 30], 1.0)));
            times.push(ms);
        }
        b.check("Vanishing Point brush dab, 200 px on the plane (median)", median(&mut times), 4.0);
        let mut times = Vec::new();
        for k in 0..40 {
            let (_, ms) = ms_of(|| vanish::dab(&g, &g, &f, f.sw * (0.2 + k as f32 * 0.015), f.sh * 0.8, 100.0, 0.5, DabSource::Stamp { src: f, offset: (300.0, 0.0) }));
            times.push(ms);
        }
        b.check("Vanishing Point stamp dab, 200 px (median)", median(&mut times), 6.0);
        let r = (f.sw * 0.1, f.sh * 0.5, f.sw * 0.1 + 1000.0, f.sh * 0.5 + 1000.0f32.min(f.sh * 0.45));
        let (_, ms) = ms_of(|| vanish::copy_rect(&g, &g, &f, r, &f, (f.sw * 0.5, f.sh * 0.5), false));
        b.check("Vanishing Point marquee copy, 1000 × 1000 plane px", ms, 150.0);
        let (_, ms) = ms_of(|| vanish::copy_rect(&g, &g, &f, r, &f, (f.sw * 0.5, f.sh * 0.5), true));
        b.check("Vanishing Point marquee copy with Heal", ms, 400.0);
        let content = ops::demo_scene(2000, 2000);
        let (_, ms) = ms_of(|| vanish::place(&g, &content, &f, (f.sw * 0.2, f.sh * 0.2, f.sw * 0.8, f.sh * 0.8)));
        b.check("Vanishing Point: place a 2000 × 2000 image on the plane", ms, 300.0);
    }
    if want("crop") {
        use crate::engine::crop::{self, Interp};
        use crate::engine::{annot, xform};
        use crate::engine::document::Document;
        use crate::ui::viewport::{DisplayCache, Viewport};
        use gpui_kit::{Bounds, point, px, size};
        let best2 = |f: &dyn Fn()| (0..2).map(|_| ms_of(f).1).fold(f64::MAX, f64::min);
        // Rotate View: the frame plan of a 4K display (1920 × 1080 points at
        // 2×) over the 268 MP document, rotated 30°, every tile cached; the
        // GPU then draws the planned tiles as transformed sprites.
        let doc = Document::from_grid(scene.clone(), "Background");
        let mut cache = DisplayCache::default();
        let canvas = Bounds { origin: point(px(0.), px(0.)), size: size(px(1920.), px(1080.)) };
        let mut vp = Viewport::default();
        vp.fit(width, height, canvas, 2.0);
        vp.zoom_about(1.0, canvas.center(), canvas);
        vp.rot = 30f32.to_radians();
        for _ in 0..200 {
            if cache.plan(&doc, None, &vp, canvas, 2.0).pending == 0 {
                break;
            }
        }
        let mut times = Vec::new();
        let mut tiles = 0;
        for k in 0..100 {
            vp.rot = (30.0 + k as f32 * 0.01).to_radians();
            let (p, ms) = ms_of(|| cache.plan(&doc, None, &vp, canvas, 2.0));
            tiles = p.draws.len();
            times.push(ms);
        }
        let med = median(&mut times);
        b.check(&format!("Rotate View frame, 4K display ({tiles} GPU tiles, median {:.0} µs)", med * 1000.0), med, 2.0);
        // Crops of the dog photo (3840 × 2560) like the Photoshop comparison.
        let photo = crate::engine::io::open_image(std::path::Path::new("testdata/photos/dog_grass.jpg")).unwrap_or_else(|_| ops::demo_scene(3840, 2560));
        let dog = vec![Layer::new(1, "bg", photo)];
        let r = RectI::new(400, 300, 3600, 2300);
        let m = annot::rotated_crop(r, 8.0);
        for (name, interp) in [("bilinear (mip)", None), ("bicubic", Some(Interp::Bicubic)), ("Lanczos-3", Some(Interp::Lanczos))] {
            let o = xform::XformOpts { interp, ..Default::default() };
            let ms = best2(&|| drop(xform::canvas_transformed_with(&dog, &[], &Default::default(), &m, 3200, 2000, o)));
            b.check(&format!("rotated crop 8° {name}, 3840 × 2560 → 3200 × 2000"), ms, 400.0);
        }
        let o = xform::XformOpts { interp: Some(Interp::Bicubic), keep: true, ..Default::default() };
        let ms = best2(&|| drop(xform::canvas_transformed_with(&dog, &[], &Default::default(), &m, 3200, 2000, o)));
        b.check("rotated crop 8°, Delete Cropped Pixels off (keeps 3840 × 2560)", ms, 800.0);
        let layers = vec![base.clone()];
        let t = [1.0, 0.0, -400.0, 0.0, 1.0, -300.0, 0.0, 0.0, 1.0];
        let ms = best2(&|| drop(xform::canvas_transformed_with(&layers, &[], &Default::default(), &t, width.saturating_sub(2048).max(1), height.saturating_sub(2048).max(1), xform::XformOpts { keep: true, ..Default::default() })));
        b.check("crop keeping hidden pixels (unaligned)", ms, 1500.0);
        // Content-Aware crop: an 8° turn of the whole photo fills its corners.
        let rot = annot::rotation_about(8.0, 1920.0, 1280.0);
        let turned = crop::warp_affine(&dog[0].grid, &rot, 3840, 2560, Interp::Bicubic);
        let hole = crop::new_area(&rot, 3840, 2560, 3840, 2560);
        let (_, ms) = ms_of(|| crop::content_aware(&turned, &hole));
        b.check("content-aware crop fill: corners of an 8° turn, 3840 × 2560", ms, 6000.0);
        let (sn, cs) = 3.5f32.to_radians().sin_cos();
        let k = (sx / (sx * cs + sy * sn)).min(sy / (sy * cs + sx * sn)).min(1.0);
        let (ow, oh) = ((sx * k).round() as u32, (sy * k).round() as u32);
        let m = crate::engine::filters::distort::mat_mul(&[1.0, 0.0, (ow as f32 - sx) * 0.5, 0.0, 1.0, (oh as f32 - sy) * 0.5, 0.0, 0.0, 1.0], &annot::rotation_about(3.5, sx * 0.5, sy * 0.5));
        let ms = best2(&|| drop(xform::canvas_transformed_with(&layers, &[], &Default::default(), &m, ow, oh, xform::XformOpts { interp: Some(Interp::Bicubic), ..Default::default() })));
        b.check("straighten 3.5° bicubic (rotate + crop)", ms, 4000.0);
    }
    b.summary();
}
