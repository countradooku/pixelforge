//! GPU against CPU for the interactive warps: Liquify (field edits and the
//! resample through the field) and Perspective Warp.

use std::sync::Arc;

use crate::engine::gpu::tests::{check_resampled, diff_exact, edge_case, gpu, photo, report};
use crate::engine::gpu::{Gpu, Mode as GpuMode, env_mode};
use crate::engine::grid::{TileGrid, finish_tile};
use crate::engine::liquify::{Backend, Brush, Mode, Session};

/// A grid whose tiles are replaced by `tiles`.
fn with_tiles(g: &TileGrid, tiles: Vec<(usize, Box<[u32]>)>) -> TileGrid {
    let mut o = g.clone();
    for (t, px) in tiles {
        o.tiles[t] = finish_tile(px);
    }
    o
}

/// A scripted Liquify session: every tool, strokes across tile seams and
/// image edges, freeze/thaw in between.
fn liquify_script(s: &mut Session, g: &mut TileGrid) {
    let (w, h) = (g.w as f32, g.h as f32);
    let big = Brush { radius: (w.min(h) * 0.18).max(30.0), pressure: 0.8, rate: 0.7 };
    let small = Brush { radius: 23.0, pressure: 1.0, rate: 1.0 };
    let stroke = |s: &mut Session, g: &mut TileGrid, mode: Mode, b: &Brush, pts: &[(f32, f32)], tick: bool| {
        s.begin_stroke(pts[0].0, pts[0].1);
        for &(x, y) in &pts[1..] {
            s.stroke_to(mode, x, y, b, g, tick);
        }
        s.end_stroke();
    };
    let line = |a: (f32, f32), b: (f32, f32), n: usize| -> Vec<(f32, f32)> { (0..=n).map(|k| (a.0 + (b.0 - a.0) * k as f32 / n as f32, a.1 + (b.1 - a.1) * k as f32 / n as f32)).collect() };
    stroke(s, g, Mode::ForwardWarp, &big, &line((w * 0.2, h * 0.5), (w * 0.62, h * 0.43), 9), false);
    stroke(s, g, Mode::ForwardWarp, &small, &line((w - 30.0, 10.0), (w + 20.0, h * 0.3), 7), false);
    stroke(s, g, Mode::PushLeft, &big, &line((w * 0.7, h * 0.2), (w * 0.72, h * 0.8), 6), false);
    stroke(s, g, Mode::Freeze, &big, &line((w * 0.5, h * 0.5), (w * 0.52, h * 0.52), 2), true);
    stroke(s, g, Mode::Bloat, &big, &[(w * 0.45, h * 0.55); 12], true);
    stroke(s, g, Mode::Pucker, &small, &line((255.0, 250.0), (262.0, 262.0), 8), true);
    stroke(s, g, Mode::TwirlCw, &big, &[(w * 0.3, h * 0.3); 10], true);
    stroke(s, g, Mode::TwirlCcw, &small, &line((10.0, h - 10.0), (60.0, h - 40.0), 5), true);
    stroke(s, g, Mode::Thaw, &big, &[(w * 0.5, h * 0.5); 4], true);
    stroke(s, g, Mode::Smooth, &big, &line((w * 0.4, h * 0.45), (w * 0.6, h * 0.5), 6), true);
    stroke(s, g, Mode::Reconstruct, &small, &line((w * 0.25, h * 0.5), (w * 0.5, h * 0.47), 10), true);
    stroke(s, g, Mode::ForwardWarp, &big, &line((w * 0.9, h * 0.9), (w * 0.3, h * 0.95), 12), false);
}

fn images() -> Vec<(&'static str, TileGrid)> {
    vec![("dog", photo("dog_grass")), ("edge", edge_case()), ("odd 301x517", crate::engine::ops::crop(&photo("butterfly"), crate::engine::grid::RectI::wh(77, 31, 301, 517)))]
}

#[test]
fn liquify_resample_is_bit_exact() {
    let Some(_) = gpu() else { return };
    for (name, img) in images() {
        let mut g = img.clone();
        let mut s = Session::new(1, 0, &g);
        s.set_backend(Backend::Cpu);
        liquify_script(&mut s, &mut g);
        let all: Vec<usize> = (0..g.tiles.len()).collect();
        // The layer is the source resampled through the field, everywhere.
        let c = with_tiles(&img, s.resample_cpu(&all));
        if let Some(d) = diff_exact(&c, &g) {
            panic!("{name}: layer != base resampled through D at {d}");
        }
        let o = with_tiles(&img, s.resample_gpu(&all).expect("GPU resample"));
        if let Some(d) = diff_exact(&o, &c) {
            panic!("{name}: GPU resample != CPU at {d}");
        }
    }
}

#[test]
fn liquify_strokes_on_the_gpu_match_the_cpu() {
    let Some(_) = gpu() else { return };
    for (name, img) in images() {
        let mut gc = img.clone();
        let mut sc = Session::new(1, 0, &gc);
        sc.set_backend(Backend::Cpu);
        liquify_script(&mut sc, &mut gc);
        let mut gg = img.clone();
        let mut sg = Session::new(1, 0, &gg);
        sg.set_backend(Backend::Gpu);
        liquify_script(&mut sg, &mut gg);
        sg.sync_field();
        assert!(sg.on_gpu() && sg.gpu_off().is_none(), "{name}: {:?}", sg.gpu_off());
        // The fields agree (bit for bit except where Twirl's sin/cos differ).
        let mut max_d = 0f32;
        let mut differ = 0usize;
        for t in sc.field_tiles() {
            let (a, b) = (sc.field(t).unwrap(), sg.field(t).expect("GPU field synced"));
            for (x, y) in a.iter().zip(b) {
                max_d = max_d.max((x - y).abs());
                differ += (x != y) as usize;
            }
        }
        println!("{name}: field max |ΔD| {max_d:e}, {differ} components differ");
        assert!(max_d < 1e-2, "{name}: field drift {max_d}");
        check_resampled(&format!("liquify {name}"), &gg, &gc);
    }
}

/// Without Twirl (whose sin/cos are the only non-exact step) a GPU session
/// is bit-exact.
#[test]
fn liquify_without_twirl_is_bit_exact() {
    let Some(_) = gpu() else { return };
    let img = edge_case();
    let run = |backend: Backend| {
        let mut g = img.clone();
        let mut s = Session::new(1, 0, &g);
        s.set_backend(backend);
        let b = Brush { radius: 140.0, pressure: 0.9, rate: 0.6 };
        for (mode, pts) in [
            (Mode::ForwardWarp, vec![(200.0, 300.0), (330.0, 320.0), (470.0, 290.0), (600.0, 360.0)]),
            (Mode::PushLeft, vec![(700.0, 100.0), (705.0, 300.0), (690.0, 600.0)]),
            (Mode::Bloat, vec![(500.0, 400.0); 8]),
            (Mode::Pucker, vec![(250.0, 260.0); 8]),
            (Mode::Smooth, vec![(400.0, 320.0), (420.0, 330.0), (450.0, 330.0), (480.0, 340.0)]),
            (Mode::Reconstruct, vec![(300.0, 300.0), (330.0, 310.0), (360.0, 320.0)]),
        ] {
            s.begin_stroke(pts[0].0, pts[0].1);
            for &(x, y) in &pts[1..] {
                s.stroke_to(mode, x, y, &b, &mut g, mode != Mode::ForwardWarp && mode != Mode::PushLeft);
            }
            s.end_stroke();
        }
        g
    };
    let (c, o) = (run(Backend::Cpu), run(Backend::Gpu));
    report("liquify no twirl", &o, &c);
    if let Some(d) = diff_exact(&o, &c) {
        panic!("GPU != CPU at {d}");
    }
}

/// The GPU dying mid-stroke: the session finishes on the CPU with the same
/// pixels (the unsynced dabs are replayed).
#[test]
fn liquify_device_loss_mid_stroke_falls_back() {
    if env_mode() == Some(GpuMode::Off) {
        return;
    }
    let Ok(g) = Gpu::new() else { return };
    let g = Arc::new(g);
    let img = photo("dog_grass");
    let b = Brush { radius: 120.0, pressure: 0.8, rate: 0.8 };
    let pts: Vec<(f32, f32)> = (0..12).map(|k| (150.0 + k as f32 * 50.0, 200.0 + k as f32 * 20.0)).collect();
    let mut gc = img.clone();
    let mut sc = Session::new(1, 0, &gc);
    sc.set_backend(Backend::Cpu);
    let mut gg = img.clone();
    let mut sg = Session::new(1, 0, &gg);
    sg.use_context(g.clone());
    for (s, grid, lose) in [(&mut sc, &mut gc, false), (&mut sg, &mut gg, true)] {
        // A first stroke that syncs, then one that loses the device midway.
        s.begin_stroke(100.0, 100.0);
        s.stroke_to(Mode::ForwardWarp, 180.0, 140.0, &b, grid, false);
        s.end_stroke();
        s.begin_stroke(pts[0].0, pts[0].1);
        for (i, &(x, y)) in pts.iter().enumerate().skip(1) {
            if lose && i == 6 {
                g.fail("simulated device loss");
                g.device.destroy();
            }
            s.stroke_to(Mode::ForwardWarp, x, y, &b, grid, false);
        }
        s.end_stroke();
    }
    assert!(!sg.on_gpu() && sg.gpu_off().is_some());
    check_resampled("liquify after device loss", &gg, &gc);
}

// ----- Perspective Warp ---------------------------------------------------------------------

use crate::engine::perspwarp::{self, Mesh, Straighten};

/// Quads over `w × h`: two joined planes and a separate one, warped.
fn warped_mesh(w: f32, h: f32) -> Mesh {
    let mut m = Mesh::default();
    m.add_quad([(w * 0.10, h * 0.12), (w * 0.45, h * 0.18), (w * 0.47, h * 0.80), (w * 0.08, h * 0.88)], 4.0);
    m.add_quad([(w * 0.45, h * 0.18), (w * 0.85, h * 0.10), (w * 0.88, h * 0.90), (w * 0.47, h * 0.80)], 4.0);
    m.add_quad([(w * 0.55, h * 0.86), (w * 0.75, h * 0.87), (w * 0.74, h * 0.98), (w * 0.56, h * 0.97)], 4.0);
    assert_eq!(m.shared_edges(), 1);
    m.straighten(Straighten::Vertical);
    m.warp[0].0 -= w * 0.05;
    m.warp[4].1 += h * 0.04;
    m.warp[2].0 += w * 0.03;
    m
}

#[test]
fn perspective_warp_gpu_is_bit_exact() {
    let Some(g) = gpu() else { return };
    let mut cases = images();
    cases.push(("odd 257x1", crate::engine::ops::crop(&photo("dog_grass"), crate::engine::grid::RectI::wh(5, 9, 257, 1))));
    for (name, img) in cases {
        let (w, h) = (img.w as f32, img.h as f32);
        let mut meshes = vec![("warped", warped_mesh(w, h)), ("identity", Mesh::default())];
        // One quad squeezed hard (strong minification, horizon nearby).
        let mut one = Mesh::default();
        one.add_quad(Mesh::rect_quad((w * 0.2, h * 0.2), (w * 0.8, h * 0.8)), 1.0);
        one.warp = vec![(w * 0.45, h * 0.3), (w * 0.55, h * 0.3), (w * 0.95, h * 0.95), (w * 0.05, h * 0.95)];
        meshes.push(("keystone", one));
        for (mn, m) in meshes {
            let table = m.table(img.w, img.h);
            let c = perspwarp::render_cpu(&img, &table);
            for rows in [None, Some(1)] {
                let o = crate::engine::gpu::kernels::perspwarp::render_opts(&g, &img, &table, false, rows, false).expect("GPU perspective warp");
                if let Some(d) = diff_exact(&o, &c) {
                    report(&format!("perspective warp {name} {mn}"), &o, &c);
                    panic!("perspective warp {name} {mn} (bands {rows:?}): GPU != CPU at {d}");
                }
            }
        }
    }
}

#[test]
fn perspective_warp_apply_matches_whatever_the_path() {
    let img = photo("dog_grass");
    let m = warped_mesh(img.w as f32, img.h as f32);
    let c = perspwarp::render_cpu(&img, &m.table(img.w, img.h));
    let a = perspwarp::apply(&img, &m);
    assert!(diff_exact(&a, &c).is_none());
    let _ = check_resampled;
}
