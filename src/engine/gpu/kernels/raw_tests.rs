//! Camera Raw, JPEG Artifacts Removal and Photo Restoration: GPU against CPU.

use std::sync::Arc;

use crate::engine::filters::{self, Filter, extra::Raw};
use crate::engine::gpu::run::RunOpts;
use crate::engine::gpu::tests::{cpu, diff_stats, edge_case, gpu, on_gpu, photo};
use crate::engine::gpu::{Gpu, PathTaken, preview};
use crate::engine::grid::{RectI, TileGrid};
use crate::engine::px;
use crate::engine::selection::Selection;

pub fn raw_zero(w: u32, h: u32) -> Raw {
    Raw {
        temperature: 0.0,
        tint: 0.0,
        exposure: 0.0,
        contrast: 0.0,
        highlights: 0.0,
        shadows: 0.0,
        whites: 0.0,
        blacks: 0.0,
        texture: 0.0,
        clarity: 0.0,
        dehaze: 0.0,
        vibrance: 0.0,
        saturation: 0.0,
        curve: [0.0; 4],
        hue: [0.0; 8],
        sat: [0.0; 8],
        lum: [0.0; 8],
        shadow_hue: 0.0,
        shadow_sat: 0.0,
        highlight_hue: 0.0,
        highlight_sat: 0.0,
        balance: 0.0,
        sharpen: 0.0,
        noise_reduction: 0.0,
        vignette: 0.0,
        grain: 0.0,
        linear: [0.0; 3],
        radial: [0.0; 4],
        img_w: w as f32,
        img_h: h as f32,
    }
}

/// The heavy preset: tone, clarity, dehaze, texture, sharpening, noise reduction.
pub fn raw_heavy(w: u32, h: u32) -> Raw {
    Raw {
        temperature: 12.0,
        tint: -6.0,
        exposure: 0.35,
        contrast: 18.0,
        highlights: -40.0,
        shadows: 35.0,
        whites: 10.0,
        blacks: -12.0,
        texture: 25.0,
        clarity: 30.0,
        dehaze: 20.0,
        vibrance: 20.0,
        saturation: 5.0,
        sharpen: 40.0,
        noise_reduction: 30.0,
        ..raw_zero(w, h)
    }
}

pub fn raw_cases(w: u32, h: u32) -> Vec<(&'static str, Raw)> {
    let z = raw_zero(w, h);
    let mut hsl = z;
    hsl.hue = [20.0, -30.0, 15.0, 40.0, -25.0, 30.0, -10.0, 50.0];
    hsl.sat = [30.0, 20.0, -40.0, 50.0, 10.0, -20.0, 60.0, -30.0];
    hsl.lum = [-20.0, 10.0, 30.0, -30.0, 20.0, -10.0, 15.0, 25.0];
    let mut tone = z;
    tone.temperature = -25.0;
    tone.tint = 15.0;
    tone.exposure = -0.4;
    tone.contrast = -20.0;
    tone.highlights = 50.0;
    tone.shadows = -30.0;
    tone.whites = -20.0;
    tone.blacks = 25.0;
    tone.vibrance = -15.0;
    tone.saturation = 25.0;
    tone.curve = [20.0, -15.0, 30.0, -25.0];
    let mut grade = z;
    grade.shadow_hue = 210.0;
    grade.shadow_sat = 60.0;
    grade.highlight_hue = 40.0;
    grade.highlight_sat = 45.0;
    grade.balance = -30.0;
    let mut masks = z;
    masks.linear = [30.0, 45.0, 0.8];
    masks.radial = [40.0, 55.0, 35.0, -0.7];
    masks.vignette = -45.0;
    let mut grain = z;
    grain.grain = 60.0;
    grain.exposure = 0.2;
    vec![("all-zero", z), ("tone", tone), ("hsl", hsl), ("grading", grade), ("masks+vignette", masks), ("grain", grain), ("heavy", raw_heavy(w, h))]
}

/// Float bar: max 1/255 per channel, mean ΔE2000 <= 0.2, always on the GPU.
fn check_close_all(g: &Arc<Gpu>, f: &dyn Fn(u32, u32) -> Filter) {
    let ell = Selection::ellipse(RectI::new(150, 90, 900, 640));
    let cases: Vec<(&str, TileGrid, Option<&Selection>, RunOpts)> = vec![
        ("dog", photo("dog_grass"), None, RunOpts::default()),
        ("edge", edge_case(), None, RunOpts::default()),
        ("edge+sel", edge_case(), Some(&ell), RunOpts::default()),
        ("butterfly 1-row bands", photo("butterfly"), None, RunOpts { max_band_rows: Some(1), ..Default::default() }),
    ];
    for (name, img, sel, opts) in cases {
        let flt = f(img.w, img.h);
        let c = cpu(&img, &flt, sel);
        let (o, path) = on_gpu(g, &img, &flt, sel, &opts);
        assert_eq!(path, PathTaken::Gpu, "{} on {name}: {path:?}", flt.name());
        let (mx, de) = diff_stats(&o, &c);
        println!("{} on {name}: max {mx}, mean ΔE {de:.5}", flt.name());
        assert!(mx <= 1 && de <= 0.2, "{flt:?} on {name}: max diff {mx}, mean ΔE {de:.4}");
    }
}

/// GPU previews at mip levels 1-3 against `Filter::render` there (float bar).
pub fn check_preview_close(g: &Arc<Gpu>, f: &Filter) {
    let mut layer = crate::engine::document::Layer::new(1, "t", edge_case());
    layer.mips = crate::engine::mips::build_mips(&layer.grid);
    for (level, rect) in [(1u32, RectI::new(-3, 5, 400, 300)), (2, RectI::new(37, 0, 260, 197)), (3, RectI::new(0, 0, 127, 98))] {
        let (w, h) = (rect.w() as usize, rect.h() as usize);
        let mut c = vec![0u32; w * h];
        f.render(&filters::Src::layer(&layer), level, rect.x0, rect.y0, w, h, &mut c, &mut Vec::new());
        let o = preview::render(g, f, &layer, level, rect).unwrap_or_else(|e| panic!("{f:?} level {level}: {e}"));
        let mut mx = 0u8;
        for (a, b) in o.iter().zip(&c) {
            for (x, y) in px::unpack(*a).iter().zip(px::unpack(*b)) {
                mx = mx.max(x.abs_diff(y));
            }
        }
        let rgb = |v: &[u32]| -> Vec<[u8; 3]> { v.iter().map(|&p| { let q = px::unpremultiply(p); [q[0], q[1], q[2]] }).collect() };
        let de = crate::engine::metrics::mean_delta_e(&rgb(&o), &rgb(&c), None);
        println!("{} preview level {level}: max {mx}, mean ΔE {de:.5}", f.name());
        assert!(mx <= 1 && de <= 0.2, "{f:?} preview level {level}: max {mx}, ΔE {de:.4}");
    }
}

#[test]
fn luma_tie_table_is_complete() {
    let t = super::raw::lum_ties().expect("at most one tie per (r, g)");
    let n = t.iter().filter(|&&v| v != 0).count();
    println!("{n} luma ties");
    // Spot-check the integer formula plus table against the CPU's float path.
    for r in (0..256u32).step_by(3) {
        for g in 0..256u32 {
            for b in (0..256u32).step_by(5) {
                let cpu = crate::engine::filters::extra::q8(crate::engine::filters::extra::lum_of(r as u8, g as u8, b as u8)) as u32;
                let mut q = (r * 299 + g * 587 + b * 114 + 500) / 1000;
                let e = t[(r * 256 + g) as usize];
                if e != 0 && (e & 0xffff) - 1 == b {
                    q = e >> 16;
                }
                assert_eq!(q, cpu, "({r}, {g}, {b})");
            }
        }
    }
}

#[test]
fn camera_raw_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let names: Vec<&str> = raw_cases(1, 1).iter().map(|c| c.0).collect();
    for (k, name) in names.iter().enumerate() {
        println!("-- {name}");
        check_close_all(&g, &|w, h| Filter::CameraRaw(raw_cases(w, h)[k].1));
    }
}

#[test]
fn camera_raw_previews_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let img = edge_case();
    for (name, r) in raw_cases(img.w, img.h) {
        println!("-- {name}");
        check_preview_close(&g, &Filter::CameraRaw(r));
    }
}

#[test]
fn jpeg_cleanup_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    for f in [
        Filter::JpegCleanup { strength: 60.0 },
        Filter::JpegCleanup { strength: 100.0 },
    ] {
        check_close_all(&g, &|_, _| f.clone());
        check_preview_close(&g, &f);
    }
}

#[test]
fn photo_restoration_stays_on_the_cpu() {
    // Its surface blur is reproduced within 1/255 (see JPEG cleanup), but the
    // following unsharp mask and contrast LUT amplify those roundings past
    // the 1/255 bar, so there is no GPU kernel.
    assert!(!crate::engine::gpu::kernels::supports(&Filter::PhotoRestoration { strength: 50.0, scratch: 40.0 }));
}
