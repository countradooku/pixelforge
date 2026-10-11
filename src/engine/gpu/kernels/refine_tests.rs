//! Select and Mask: GPU against the CPU reference (bit-exact masks and
//! decontaminated colors), per operation and combined, with band seams.

use crate::engine::gpu::tests::{edge_case, gpu, photo};
use crate::engine::grid::{TS, TileGrid, build_grid, finish_tile};
use crate::engine::px;
use crate::engine::refine::{self, Output, Params};
use crate::engine::selection::MaskGrid;

/// A mask with real image edges: the photo's bright areas inside an ellipse.
pub fn photo_mask(img: &TileGrid) -> MaskGrid {
    let (w, h) = (img.w, img.h);
    let (cx, cy, rx, ry) = (w as f32 * 0.5, h as f32 * 0.5, w as f32 * 0.42, h as f32 * 0.4);
    let mut g = TileGrid::new(w, h);
    g.tiles = build_grid(w, h, |tx, ty| {
        let mut t = vec![0u32; TS * TS].into_boxed_slice();
        for y in 0..TS {
            for x in 0..TS {
                let (gx, gy) = (tx as u32 * 256 + x as u32, ty as u32 * 256 + y as u32);
                if gx < w && gy < h {
                    let inside = ((gx as f32 - cx) / rx).powi(2) + ((gy as f32 - cy) / ry).powi(2) < 1.0;
                    let v = if inside && refine::guide_of(img.pixel(gx as i32, gy as i32)) > 110 { 255 } else { 0 };
                    t[y * TS + x] = px::pack(v, v, v, 255);
                }
            }
        }
        finish_tile(t)
    })
    .tiles;
    MaskGrid::from_gray(&g)
}

pub fn cases() -> Vec<(&'static str, Params)> {
    let d = Params::default();
    vec![
        ("radius", Params { radius: 12.0, ..d }),
        ("radius+smart", Params { radius: 24.0, smart: true, ..d }),
        ("smooth", Params { smooth: 60.0, ..d }),
        ("feather small", Params { feather: 1.5, ..d }),
        ("feather large", Params { feather: 9.0, ..d }),
        ("contrast", Params { feather: 4.0, contrast: 40.0, ..d }),
        ("shift", Params { feather: 4.0, shift: -35.0, ..d }),
        ("decontaminate", Params { decontaminate: true, amount: 80.0, ..d }),
        (
            "all",
            Params { radius: 10.0, smart: true, smooth: 30.0, feather: 2.0, contrast: 15.0, shift: 20.0, decontaminate: true, amount: 100.0, output: Output::NewLayerWithMask },
        ),
    ]
}

fn plane(m: &MaskGrid) -> Vec<u8> {
    let mut v = vec![0u8; (m.w * m.h) as usize];
    m.read_region(0, 0, m.w as usize, m.h as usize, &mut v, false);
    v
}

fn first_diff_mask(a: &MaskGrid, b: &MaskGrid) -> Option<String> {
    let (pa, pb) = (plane(a), plane(b));
    let i = (0..pa.len()).find(|&i| pa[i] != pb[i])?;
    Some(format!("({}, {}): GPU {} CPU {}", i as u32 % a.w, i as u32 / a.w, pa[i], pb[i]))
}

#[test]
fn select_and_mask_matches_the_cpu_bit_exactly() {
    let Some(g) = gpu() else { return };
    let (fm, fi) = crate::engine::refine::tests::fringe_scene(700, 600);
    let scenes = vec![("butterfly", photo("butterfly")), ("edge", edge_case())];
    let mut all: Vec<(&str, MaskGrid, TileGrid)> = scenes.into_iter().map(|(n, img)| (n, photo_mask(&img), img)).collect();
    all.push(("fringe", fm, fi));
    for (scene, mask, img) in &all {
        for (name, p) in cases() {
            let c = refine::refine_cpu(mask, img, &p);
            for rows in [None, Some(1)] {
                let (o, _) = super::refine::run(&g, mask, img, &p, rows).unwrap_or_else(|e| panic!("{name} on {scene}: {e}"));
                if let Some(d) = first_diff_mask(&o.mask, &c.mask) {
                    panic!("{name} on {scene} (bands {rows:?}): mask differs at {d}");
                }
                assert_eq!(o.colors.is_some(), c.colors.is_some());
                if let (Some(oc), Some(cc)) = (&o.colors, &c.colors) {
                    if let Some(d) = crate::engine::gpu::tests::diff_exact(oc, cc) {
                        panic!("{name} on {scene} (bands {rows:?}): colors differ at {d}");
                    }
                }
            }
            println!("{name} on {scene}: bit-exact");
        }
    }
}

#[test]
fn select_and_mask_switch_falls_back_to_identical_pixels() {
    let img = edge_case();
    let mask = photo_mask(&img);
    let p = Params { radius: 8.0, feather: 3.0, ..Default::default() };
    let c = refine::refine_cpu(&mask, &img, &p);
    // Whatever PF_GPU says (off, auto, force), the public entry gives the CPU's pixels.
    let r = refine::refine(&mask, &img, &p);
    assert!(first_diff_mask(&r.mask, &c.mask).is_none());
    // Per-op record: other tests record their own ops concurrently.
    assert!(crate::engine::gpu::last_of(super::refine::OP).is_some(), "recorded");
}
