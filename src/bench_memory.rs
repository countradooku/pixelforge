//! Reproducible storage and interaction probes. Uses only the current 8-bit engine.
use crate::engine::{
    document::Document,
    filters::Filter,
    grid::{TPX, TileGrid},
    memory, ops,
};
use std::{hint::black_box, time::Instant};

pub fn run(w: u32, h: u32, only: Option<&str>) {
    println!("Memory benchmark — {w} × {h}; RGBA8, 20 undo states");
    if only != Some("pressure") {
        compression();
    }
    if only == Some("compression") {
        return;
    }
    let g = ops::procedural(w, h);
    let mut doc = Document::from_grid(g, "Memory benchmark");
    let mut expected = Vec::new();
    for step in 0..20 {
        expected.push(fingerprint(&doc.layers[0].grid));
        doc.push_undo(format!("Revision {step}"));
        let t = Instant::now();
        let out = ops::fill(
            &doc.layers[0].grid,
            None,
            [step * 11, 230 - step * 9, 77],
            0.1,
        );
        doc.set_layer_grid(0, out);
        doc.maintain_history();
        memory::global().maintain();
        println!(
            "revision {step:02}: {:.3} ms; {}",
            t.elapsed().as_secs_f64() * 1000.,
            memory::global().stats().summary()
        );
    }
    // Paint and display with every history state still present.
    let paint = doc.insert_layer("Paint latency", TileGrid::new(w, h));
    let mut stroke = crate::engine::brush::Stroke::brush(
        1,
        &doc.layers[paint].grid,
        crate::engine::brush::BrushSettings {
            size: 300.,
            ..Default::default()
        },
        [220, 40, 60],
        false,
        None,
    );
    let mut events = Vec::new();
    for i in 0..100 {
        let t = Instant::now();
        let dirty = stroke.add_point(
            w as f32 * 0.1 + i as f32 * 8.,
            h as f32 * 0.1,
            &mut doc.layers[paint].grid,
        );
        doc.layer_pixels_changed(paint, &dirty);
        events.push(t.elapsed().as_secs_f64() * 1000.);
    }
    events.sort_by(f64::total_cmp);
    println!(
        "pressure brush including mips: median {:.3} ms; p95 {:.3} ms; max {:.3} ms",
        events[50], events[94], events[99]
    );
    let t = Instant::now();
    let input = crate::engine::composite::ComposeInput {
        layers: &doc.layers,
        preview: None,
        extra: None,
    };
    {
        use rayon::prelude::*;
        (0..135).into_par_iter().for_each_init(
            crate::engine::composite::Scratch::default,
            |scratch, i| {
                let mut out =
                    vec![0; crate::engine::composite::DT * crate::engine::composite::DT * 4];
                crate::engine::composite::display_tile(
                    &input,
                    3,
                    i % 15,
                    i / 15,
                    &mut out,
                    scratch,
                );
            },
        );
    }
    println!(
        "pressure display: {:.3} ms for 135 RGBA8 tiles",
        t.elapsed().as_secs_f64() * 1000.
    );
    let mut times = Vec::new();
    for step in (0..20).rev() {
        let t = Instant::now();
        doc.undo().expect("history retained under pressure");
        times.push(t.elapsed().as_secs_f64() * 1000.);
        assert_eq!(
            fingerprint(&doc.layers[0].grid),
            expected[step],
            "all-tile undo fingerprint at revision {step}"
        );
        memory::global().maintain();
    }
    times.sort_by(f64::total_cmp);
    println!(
        "undo all-tile fingerprints: 20/20; median {:.3} ms; p95 {:.3} ms",
        times[10], times[18]
    );
    println!(
        "32-bit document stress: unavailable until the float engine exists; no 8-bit conversion substituted"
    );
}

// Fingerprint every pixel of every tile, including edge padding; verification
// is outside the interaction timer and never retains a second image in RAM.
fn fingerprint(grid: &TileGrid) -> Vec<u64> {
    use rayon::prelude::*;
    grid.tiles
        .par_iter()
        .map(|t| {
            t.as_ref().map_or(0, |t| {
                t.read().iter().fold(0xcbf29ce484222325u64, |hash, p| {
                    (hash ^ u64::from(*p)).wrapping_mul(0x100000001b3)
                })
            })
        })
        .collect()
}

fn compression() {
    use std::io::{Read, Write};
    let g = ops::procedural(2048, 2048);
    let mut noise = 123456789u32;
    let noisy = crate::engine::grid::finish_tile(
        (0..TPX)
            .map(|_| {
                noise ^= noise << 13;
                noise ^= noise >> 17;
                noise ^= noise << 5;
                noise
            })
            .collect(),
    );
    for (name, tiles) in [
        ("procedural", &g),
        (
            "incompressible",
            &TileGrid {
                w: 256,
                h: 256,
                cols: 1,
                rows: 1,
                tiles: vec![noisy],
            },
        ),
    ] {
        for codec in ["LZ4", "deflate-1"] {
            let t = Instant::now();
            let mut compressed = Vec::new();
            let mut raw_bytes = 0usize;
            for tile in tiles.tiles.iter().flatten() {
                let p = tile.read();
                let raw: &[u8] = bytemuck::cast_slice(&p);
                raw_bytes += raw.len();
                let b = if codec == "LZ4" {
                    lz4_flex::compress(raw)
                } else {
                    let mut e =
                        flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
                    e.write_all(raw).unwrap();
                    e.finish().unwrap()
                };
                compressed.push(b);
            }
            let encode = t.elapsed().as_secs_f64() * 1000.;
            let size: usize = compressed.iter().map(Vec::len).sum();
            let t = Instant::now();
            for (b, tile) in compressed.iter().zip(tiles.tiles.iter().flatten()) {
                let p = tile.read();
                let raw: &[u8] = bytemuck::cast_slice(&p);
                let decoded = if codec == "LZ4" {
                    lz4_flex::decompress(b, TPX * 4).unwrap()
                } else {
                    let mut out = Vec::new();
                    flate2::read::DeflateDecoder::new(&b[..])
                        .read_to_end(&mut out)
                        .unwrap();
                    out
                };
                assert_eq!(decoded, raw);
                black_box(decoded);
            }
            println!(
                "{name} {codec}: {:.3} ms encode, {:.3} ms decode+verify, {raw_bytes} → {size} bytes ({:.1}%)",
                encode,
                t.elapsed().as_secs_f64() * 1000.,
                size as f64 * 100. / raw_bytes as f64
            );
        }
    }
    // A warm commit sample makes this probe useful alongside pressure runs.
    black_box(ops::apply_filter(
        &g,
        &Filter::Invert,
        None,
        &ops::Progress::default(),
    ));
}
