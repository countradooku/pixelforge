//! Slices beyond the rectangle: layer-based slices (their rect follows a
//! layer's content), Divide Slice, and Save for Web — every slice (user and
//! auto) optimized to PNG-8 / PNG-24 / JPEG / WebP, plus an HTML page laying
//! them out as a table with each user slice's link, target and alt text.

use std::path::{Path, PathBuf};

use rayon::prelude::*;

use super::annot::{Slice, all_slices, region_rgba};
use super::document::Layer;
use super::grid::RectI;
use super::webexport::{self, SliceCell, WebFormat};

/// The layer's content bounds (pixels, or the vector description's).
pub fn layer_bounds(layers: &[Layer], id: u64) -> Option<RectI> {
    let l = layers.iter().find(|l| l.id == id)?;
    l.kind.content_bounds().or_else(|| super::xform::content_rect(&l.grid))
}

/// User slices with layer-based rects resolved to the layers' current bounds
/// (clipped to the canvas; a missing layer keeps the stored rect).
pub fn resolved(slices: &[Slice], layers: &[Layer], w: u32, h: u32) -> Vec<Slice> {
    let full = RectI::wh(0, 0, w as i32, h as i32);
    slices
        .iter()
        .map(|s| {
            let rect = s.layer.and_then(|id| layer_bounds(layers, id)).unwrap_or(s.rect).intersect(&full);
            Slice { rect, ..s.clone() }
        })
        .collect()
}

/// Divide Slice: `nx` columns × `ny` rows of (nearly) equal size.
pub fn divide(r: RectI, nx: u32, ny: u32) -> Vec<RectI> {
    let (nx, ny) = (nx.max(1) as i32, ny.max(1) as i32);
    let xs: Vec<i32> = (0..=nx).map(|k| r.x0 + (r.w() * k + nx / 2) / nx).collect();
    let ys: Vec<i32> = (0..=ny).map(|k| r.y0 + (r.h() * k + ny / 2) / ny).collect();
    let mut out = Vec::new();
    for y in ys.windows(2) {
        for x in xs.windows(2) {
            out.push(RectI::new(x[0], y[0], x[1], y[1]));
        }
    }
    out
}

pub fn extension(fmt: WebFormat) -> &'static str {
    match fmt {
        WebFormat::Png8 { .. } | WebFormat::Png24 { .. } => "png",
        WebFormat::Jpeg { .. } => "jpg",
        WebFormat::Webp => "webp",
    }
}

/// One encoded slice.
pub struct WebSlice {
    pub name: String,
    pub rect: RectI,
    pub user: bool,
    pub bytes: Vec<u8>,
}

/// Encodes every slice (user slices resolved, then the auto slices that tile
/// the rest), in parallel. Auto slices are named `<base>_NN`.
pub fn encode_all(layers: &[Layer], w: u32, h: u32, user: &[Slice], fmt: WebFormat, matte: [u8; 3], base: &str) -> Result<Vec<WebSlice>, String> {
    let user = resolved(user, layers, w, h);
    let all = all_slices(&user, w, h);
    let mut encoded = Vec::with_capacity(all.len());
    // Finish composition before starting codecs: a nested Rayon composition
    // must not steal a long-running compression job from another slice. Bound
    // the staging buffers to one batch of workers rather than all slices.
    let workers = rayon::current_num_threads().max(1);
    for (batch_index, batch) in all.chunks(workers).enumerate() {
        let pixels: Vec<_> = batch.par_iter().map(|&(r, _)| region_rgba(layers, r)).collect();
        let output: Result<Vec<_>, String> = batch.par_iter().zip(pixels.into_par_iter())
            .enumerate()
            .map(|(index, (&(r, is_user), rgba))| {
                let k = batch_index * workers + index;
                let bytes = webexport::encode(&rgba, r.w() as u32, r.h() as u32, fmt, matte)?;
                let name = if is_user { user.iter().find(|s| s.rect == r).map_or_else(|| format!("{base}_{:02}", k + 1), |s| s.name.clone()) } else { format!("{base}_{:02}", k + 1) };
                Ok(WebSlice { name, rect: r, user: is_user, bytes })
            })
            .collect();
        encoded.extend(output?);
    }
    Ok(encoded)
}

/// Save for Web: writes `images/<slice>.<ext>` and `<base>.html` into `dir`;
/// returns the written files.
pub fn save_for_web(layers: &[Layer], w: u32, h: u32, user: &[Slice], fmt: WebFormat, matte: [u8; 3], dir: &Path, base: &str) -> Result<Vec<PathBuf>, String> {
    let slices = encode_all(layers, w, h, user, fmt, matte, base)?;
    let resolved_user = resolved(user, layers, w, h);
    let img_dir = dir.join("images");
    std::fs::create_dir_all(&img_dir).map_err(|e| e.to_string())?;
    let ext = extension(fmt);
    let safe = |s: &str| s.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect::<String>();
    let mut files = Vec::new();
    let mut cells = Vec::new();
    for s in &slices {
        let file = format!("{}.{ext}", safe(&s.name));
        let p = img_dir.join(&file);
        std::fs::write(&p, &s.bytes).map_err(|e| e.to_string())?;
        files.push(p);
        let opts = resolved_user.iter().find(|u| s.user && u.rect == s.rect);
        cells.push(SliceCell {
            x0: s.rect.x0,
            y0: s.rect.y0,
            x1: s.rect.x1,
            y1: s.rect.y1,
            file: format!("images/{file}"),
            url: opts.map_or(String::new(), |u| u.url.clone()),
            target: opts.map_or(String::new(), |u| u.target.clone()),
            alt: opts.map_or(String::new(), |u| u.alt.clone()),
        });
    }
    let html = webexport::html_table(base, w, h, &cells);
    let hp = dir.join(format!("{}.html", safe(base)));
    std::fs::write(&hp, html).map_err(|e| e.to_string())?;
    files.push(hp);
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::grid::TileGrid;

    #[test]
    fn divide_and_layer_based_slices() {
        let d = divide(RectI::new(0, 0, 100, 50), 3, 2);
        assert_eq!(d.len(), 6);
        assert_eq!(d.iter().map(|r| r.w() * r.h()).sum::<i32>(), 5000);
        assert_eq!((d[0].x1, d[1].x1, d[2].x1, d[3].y0), (33, 67, 100, 25));
        // A layer-based slice follows its layer's content.
        let sel = crate::engine::selection::Selection::rect(RectI::new(50, 60, 150, 90));
        let g = crate::engine::ops::fill(&TileGrid::new(400, 300), Some(&sel), [255, 0, 0], 1.0);
        let layers = vec![Layer::new(7, "logo", g)];
        let s = Slice { layer: Some(7), ..Slice::new("logo", RectI::EMPTY) };
        assert_eq!(resolved(&[s], &layers, 400, 300)[0].rect, RectI::new(50, 60, 150, 90));
    }

    #[test]
    fn batched_png_slices_preserve_order_names_and_transparent_pixels() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            let (w, h) = (123, 79);
            let mut layer = Layer::new(1, "pixels", crate::engine::ops::procedural(w, h));
            layer.opacity = 0.6;
            let layers = [layer];
            let rects = divide(RectI::new(0, 0, w as i32, h as i32), 7, 3);
            let user: Vec<_> = rects.iter().enumerate().map(|(i, &r)| Slice::new(format!("part{i}"), r)).collect();
            let expected = region_rgba(&layers, RectI::new(0, 0, w as i32, h as i32));
            let encoded = encode_all(&layers, w, h, &user, WebFormat::Png24 { transparency: true }, [255; 3], "page").unwrap();
            assert_eq!(encoded.len(), rects.len());
            for (i, slice) in encoded.iter().enumerate() {
                assert_eq!(slice.name, format!("part{i}"));
                assert_eq!(slice.rect, rects[i]);
                assert!(slice.user);
                let decoded = image::load_from_memory(&slice.bytes).unwrap().into_rgba8();
                for (x, y, pixel) in decoded.enumerate_pixels() {
                    let offset = (((slice.rect.y0 as u32 + y) * w + slice.rect.x0 as u32 + x) * 4) as usize;
                    assert_eq!(pixel.0, expected[offset..offset + 4]);
                }
            }
        });
    }

    #[test]
    fn save_for_web_writes_images_and_html() {
        let layers = vec![Layer::new(1, "bg", crate::engine::ops::procedural(400, 300))];
        let user = vec![Slice { url: "https://example.com/a?b=1&c=2".into(), alt: "Logo <x>".into(), target: "_blank".into(), ..Slice::new("logo", RectI::new(100, 100, 200, 150)) }];
        let dir = std::env::temp_dir().join(format!("pf-web-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for fmt in [WebFormat::Png8 { colors: 64, dither: true, transparency: true }, WebFormat::Png24 { transparency: true }, WebFormat::Jpeg { quality: 60 }, WebFormat::Webp] {
            let files = save_for_web(&layers, 400, 300, &user, fmt, [255, 255, 255], &dir, "page").unwrap();
            // 1 user slice + 4 auto slices around it + the page.
            assert_eq!(files.len(), 6, "{fmt:?}");
            let html = std::fs::read_to_string(dir.join("page.html")).unwrap();
            assert!(html.contains("images/logo.") && html.contains("href=\"https://example.com/a?b=1&amp;c=2\"") && html.contains("Logo &lt;x&gt;") && html.contains("_blank"));
            let img = image::open(dir.join("images").join(format!("logo.{}", extension(fmt)))).unwrap();
            assert_eq!((img.width(), img.height()), (100, 50));
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
