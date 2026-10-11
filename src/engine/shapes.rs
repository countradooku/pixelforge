//! The Custom Shape library: resolution-independent outlines normalized to
//! the unit square and stretched to the dragged box.

use super::vector::{Anchor, Path, SubPath, p};

pub const CUSTOM_SHAPES: [&str; 8] = ["Heart", "Arrow", "Star", "Check", "Speech Bubble", "Lightning", "Cloud", "Gear"];

/// Builds custom shape `which` inside the box (x0, y0)–(x1, y1).
pub fn custom_shape(which: usize, x0: f32, y0: f32, x1: f32, y1: f32) -> Path {
    let (lx, ly) = (x0.min(x1), y0.min(y1));
    let (w, h) = ((x1 - x0).abs(), (y1 - y0).abs());
    let m = |u: f32, v: f32| p(lx + u * w, ly + v * h);
    let corner = |u: f32, v: f32| Anchor::corner(m(u, v));
    let smooth = |u: f32, v: f32, iu: f32, iv: f32, ou: f32, ov: f32| Anchor { p: m(u, v), h_in: m(iu, iv), h_out: m(ou, ov), smooth: true };
    let cusp = |u: f32, v: f32, iu: f32, iv: f32, ou: f32, ov: f32| Anchor { p: m(u, v), h_in: m(iu, iv), h_out: m(ou, ov), smooth: false };
    let closed = |anchors: Vec<Anchor>| SubPath { anchors, closed: true };
    let subs = match which % CUSTOM_SHAPES.len() {
        0 => vec![closed(vec![
            cusp(0.5, 0.25, 0.5, 0.25, 0.62, 0.02),
            smooth(0.97, 0.3, 0.97, 0.08, 0.97, 0.55),
            cusp(0.5, 1.0, 0.75, 0.75, 0.5, 1.0),
            smooth(0.03, 0.3, 0.03, 0.55, 0.03, 0.08),
            cusp(0.5, 0.25, 0.38, 0.02, 0.5, 0.25),
        ])],
        1 => vec![closed(vec![corner(0.0, 0.35), corner(0.6, 0.35), corner(0.6, 0.1), corner(1.0, 0.5), corner(0.6, 0.9), corner(0.6, 0.65), corner(0.0, 0.65)])],
        2 => return Path::star(lx + w * 0.5, ly + h * 0.5, w * 0.5, h * 0.5, 5, 0.6, 0.0),
        3 => vec![closed(vec![corner(0.0, 0.55), corner(0.15, 0.4), corner(0.38, 0.62), corner(0.85, 0.1), corner(1.0, 0.25), corner(0.38, 0.92)])],
        4 => vec![closed(vec![
            smooth(0.5, 0.0, 0.2, 0.0, 0.8, 0.0),
            smooth(1.0, 0.38, 1.0, 0.15, 1.0, 0.6),
            smooth(0.55, 0.76, 0.8, 0.76, 0.45, 0.76),
            corner(0.2, 1.0),
            cusp(0.28, 0.72, 0.28, 0.72, 0.1, 0.68),
            smooth(0.0, 0.38, 0.0, 0.6, 0.0, 0.15),
        ])],
        5 => vec![closed(vec![corner(0.55, 0.0), corner(0.15, 0.55), corner(0.45, 0.55), corner(0.3, 1.0), corner(0.85, 0.4), corner(0.55, 0.4), corner(0.75, 0.0)])],
        6 => vec![closed(vec![
            cusp(0.15, 0.85, 0.15, 0.85, -0.05, 0.85),
            cusp(0.12, 0.5, -0.05, 0.55, 0.1, 0.3),
            cusp(0.4, 0.28, 0.35, 0.05, 0.6, 0.0),
            cusp(0.75, 0.35, 0.82, 0.18, 1.05, 0.35),
            cusp(0.88, 0.85, 1.05, 0.65, 0.88, 0.85),
        ])],
        _ => {
            // Gear: 10 teeth, a hole in the middle (even-odd friendly: hole reversed).
            let teeth = 10;
            let mut a = Vec::new();
            for k in 0..teeth * 4 {
                let t = k as f32 / (teeth * 4) as f32 * std::f32::consts::TAU;
                let r = if (k / 2) % 2 == 0 { 0.5 } else { 0.38 };
                a.push(corner(0.5 + r * t.cos(), 0.5 + r * t.sin()));
            }
            let hole = Path::ellipse(lx + 0.5 * w, ly + 0.5 * h, 0.15 * w, 0.15 * h);
            let mut hs = hole.subs[0].clone();
            hs.anchors.reverse();
            for an in &mut hs.anchors {
                std::mem::swap(&mut an.h_in, &mut an.h_out);
            }
            return Path { subs: vec![closed(a), hs] };
        }
    };
    Path { subs }
}
