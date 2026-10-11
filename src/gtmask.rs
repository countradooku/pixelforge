//! `--gt-mask`: builds hand-made ground-truth masks for the segmentation
//! quality checks from a small text file of polygons.
//!
//! ```text
//! pixelforge --gt-mask photo.jpg mask.txt out.png [preview.png [x0 y0 x1 y1]]
//! ```
//!
//! Each non-empty line of `mask.txt` (applied in order; `#` starts a comment):
//!
//! - `+ x y x y …` adds a polygon,
//! - `- x y x y …` removes one,
//! - `dark T x y x y …` adds the pixels inside the polygon whose luma is
//!   below `T` (fine dark strands such as hair against a light backdrop),
//! - `light T x y x y …` removes pixels inside the polygon brighter than `T`.
//!
//! Coordinates are full-resolution pixels. The output is an 8-bit gray PNG
//! (255 = subject). The optional preview shows the photo, tinted red where
//! the mask is set, cropped to `x0 y0 x1 y1` and scaled to ≤ 1600 px.

use anyhow::{Context, Result, bail};

use crate::engine::vector::{FillRule, Poly, p, rasterize_mask};

pub fn run(args: &[String]) -> Result<()> {
    let (img_path, txt, out) = match args {
        [a, b, c, ..] => (a, b, c),
        _ => bail!("usage: --gt-mask photo.jpg mask.txt out.png [preview.png [x0 y0 x1 y1]]"),
    };
    let img = image::open(img_path).with_context(|| format!("opening {img_path}"))?.into_rgb8();
    let (w, h) = img.dimensions();
    let luma: Vec<u8> = img.pixels().map(|q| ((q[0] as u32 * 54 + q[1] as u32 * 183 + q[2] as u32 * 19) >> 8) as u8).collect();
    let mut mask = vec![0u8; (w * h) as usize];
    for (ln, line) in std::fs::read_to_string(txt)?.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let op = it.next().unwrap();
        let thr = if matches!(op, "dark" | "light") { Some(it.next().context("threshold")?.parse::<u8>()?) } else { None };
        let nums: Vec<f32> = it.map(|v| v.parse::<f32>()).collect::<Result<_, _>>().with_context(|| format!("line {}", ln + 1))?;
        if nums.len() < 6 || nums.len() % 2 != 0 {
            bail!("line {}: need at least 3 x y pairs", ln + 1);
        }
        let poly = Poly { pts: nums.chunks(2).map(|c| p(c[0], c[1])).collect(), closed: true };
        let cov = rasterize_mask(&[poly], w, h, FillRule::NonZero);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let c = cov.at(x, y);
                if c == 0 {
                    continue;
                }
                let i = (y as u32 * w + x as u32) as usize;
                match (op, thr) {
                    ("+", _) => mask[i] = mask[i].max(c),
                    ("-", _) => mask[i] = mask[i].min(255 - c),
                    ("dark", Some(t)) if luma[i] < t && c > 127 => mask[i] = 255,
                    ("light", Some(t)) if luma[i] > t && c > 127 => mask[i] = 0,
                    ("dark" | "light", _) => {}
                    _ => bail!("line {}: unknown op {op}", ln + 1),
                }
            }
        }
    }
    image::GrayImage::from_raw(w, h, mask.clone()).unwrap().save(out)?;
    let set = mask.iter().filter(|&&v| v > 127).count();
    println!("{out}: {w} × {h}, {:.1}% subject", set as f64 * 100.0 / (w * h) as f64);
    if let Some(prev) = args.get(3) {
        let r: Vec<u32> = args[4..].iter().filter_map(|v| v.parse().ok()).collect();
        let (x0, y0, x1, y1) = if r.len() == 4 { (r[0], r[1], r[2].min(w), r[3].min(h)) } else { (0, 0, w, h) };
        let k = ((x1 - x0).max(y1 - y0) as f32 / 1600.0).max(1.0);
        let (pw, ph) = (((x1 - x0) as f32 / k) as u32, ((y1 - y0) as f32 / k) as u32);
        let mut pv = image::RgbImage::new(pw, ph);
        for (px_, py, q) in pv.enumerate_pixels_mut() {
            let (sx, sy) = ((x0 as f32 + px_ as f32 * k) as u32, (y0 as f32 + py as f32 * k) as u32);
            let c = img.get_pixel(sx, sy);
            let m = mask[(sy * w + sx) as usize] as u32;
            let tint = |v: u8, t: u32| ((v as u32 * (255 - m / 2) + t * m / 2) / 255) as u8;
            *q = image::Rgb([tint(c[0], 255), tint(c[1], 0), tint(c[2], 0)]);
        }
        pv.save(prev)?;
    }
    Ok(())
}
