//! `--compare-gaps <ps_out_dir> [phases]`: quality and speed of the
//! gap-closing features against Photoshop's outputs (written by
//! `tools/photoshop_gaps.sh`) and hand-made ground truth
//! (`testdata/photos/gt/`). Prints markdown tables for the README.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::engine::grid::{RectI, TileGrid};
use crate::engine::metrics::{iou, mean_delta_e, ssim};
use crate::engine::{io, mips, objects, patchmatch, px, retouch, segment};
use crate::engine::selection::{MaskGrid, Selection};

/// A photo with its mips, as a segmentation source.
struct Photo {
    grid: TileGrid,
    mips: Vec<TileGrid>,
}

impl Photo {
    fn load(name: &'static str) -> Option<Photo> {
        let grid = io::open_image(Path::new(&format!("testdata/photos/{name}.jpg"))).ok()?;
        let mips = mips::build_mips(&grid);
        Some(Photo { grid, mips })
    }

    fn src(&self) -> impl Fn(u32, i32, i32, usize, usize, &mut [u32]) + Sync + '_ {
        move |l, x, y, bw, bh, out: &mut [u32]| {
            let lv = if l == 0 { &self.grid } else { &self.mips[(l as usize).min(self.mips.len()) - 1] };
            lv.read_region(x, y, bw, bh, out)
        }
    }

    fn rgb(&self) -> Vec<[u8; 3]> {
        let (w, h) = (self.grid.w as usize, self.grid.h as usize);
        let mut buf = vec![0u32; w * h];
        self.grid.read_region(0, 0, w, h, &mut buf);
        buf.iter().map(|&c| {
            let [r, g, b, _] = px::unpremultiply(c);
            [r, g, b]
        }).collect()
    }
}

fn gray_png(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    let img = image::open(path).ok()?.into_luma8();
    let (w, h) = img.dimensions();
    Some((img.into_raw(), w, h))
}

fn rgb_png(path: &Path) -> Option<Vec<[u8; 3]>> {
    Some(image::open(path).ok()?.into_rgb8().pixels().map(|p| p.0).collect())
}

fn mask_bytes(m: &MaskGrid) -> Vec<u8> {
    let (w, h) = (m.w as usize, m.h as usize);
    let mut v = vec![0u8; w * h];
    m.read_region(0, 0, w, h, &mut v, false);
    v
}

/// Content-aware fill holes: (label, cx, cy, rx, ry). Same as the jsx.
fn holes(name: &str) -> Vec<(&'static str, f32, f32, f32, f32)> {
    match name {
        "dog_grass" => vec![("300px", 400.0, 2100.0, 150.0, 150.0), ("1MP", 3300.0, 1900.0, 531.0, 600.0), ("daisies", 1720.0, 430.0, 240.0, 90.0)],
        "butterfly" => vec![("300px", 2200.0, 420.0, 150.0, 150.0)],
        "portrait_hair" => vec![("300px", 500.0, 600.0, 150.0, 150.0)],
        _ => vec![],
    }
}

/// Photoshop's timing for `key` from timings_A.txt.
fn ps_ms(dir: &Path, key: &str) -> Option<f64> {
    let t = std::fs::read_to_string(dir.join("timings_A.txt")).ok()?;
    t.lines().find_map(|l| {
        let (k, v) = l.split_once('\t')?;
        (k == key).then(|| v.trim().parse().ok()).flatten()
    })
}

pub fn run(args: &[String]) {
    let dir = PathBuf::from(args.first().cloned().unwrap_or_else(|| "ps_out".into()));
    let arg = args.get(1).cloned().unwrap_or_else(|| "A".into());
    let list: Vec<&str> = arg.split(',').map(str::trim).collect();
    if list.contains(&"GPU") {
        crate::compare_gpu::run(&dir);
    }
    // Single-letter phases (also accepts "ABC").
    let phases: String = list.iter().filter(|p| **p != "GPU").copied().collect::<Vec<_>>().join("");
    if phases.contains('A') {
        phase_a(&dir);
    }
    if phases.contains('B') {
        phase_b_warps(&dir);
    }
    if phases.contains('C') {
        phase_c_paths(&dir);
    }
    if phases.contains('D') {
        phase_d_web(&dir);
    }
    if phases.contains('E') {
        phase_e_crop(&dir);
    }
    if phases.contains('F') {
        phase_f_info(&dir);
    }
    if phases.contains('G') {
        phase_g_gallery(&dir);
    }
}

/// Filter Gallery fidelity: each effect at its defaults on the same
/// 1024 × 683 photo as Photoshop's references; SSIM (luma) and mean ΔE2000.
/// `PF_ONLY=n` limits it to effect `n`. Writes `pf_gallery_<n>.png`.
fn phase_g_gallery(dir: &Path) {
    use crate::engine::filters::gallery::G;
    println!("\n## Phase G — Filter Gallery vs Photoshop (dog photo at 1024 × 683, default settings)\n");
    let Ok(src) = image::open(dir.join("gallery_src.png")) else {
        return println!("(no gallery_src.png: run tools/photoshop_gaps.sh {} G)", dir.display());
    };
    let src = src.to_rgba8();
    let (w, h) = (src.width() as usize, src.height() as usize);
    let base: Vec<u32> = src.pixels().map(|p| crate::engine::px::premultiply(p[0], p[1], p[2], p[3])).collect();
    let only: Option<usize> = std::env::var("PF_ONLY").ok().and_then(|v| v.parse().ok());
    println!("| # | Effect | SSIM | mean ΔE2000 | PF time |");
    println!("|---:|---|---:|---:|---:|");
    let (mut below, mut sum) = (0, 0.0);
    for (i, g) in G::ALL.iter().enumerate() {
        if only.is_some_and(|o| o != i) {
            continue;
        }
        let Ok(ps) = image::open(dir.join(format!("gallery_{i}.png"))) else { continue };
        let ps: Vec<[u8; 3]> = ps.to_rgb8().pixels().map(|p| [p[0], p[1], p[2]]).collect();
        let defaults = g.params();
        let mut p = [0f32; 3];
        for (k, d) in defaults.iter().enumerate().take(3) {
            p[k] = d.default;
        }
        let mut buf = base.clone();
        let t0 = std::time::Instant::now();
        g.apply_in(p, [0, 0, 0], [255, 255, 255], &mut buf, w, h, 0, (0, 0), Some((w as u32, h as u32)));
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ours: Vec<[u8; 3]> = buf.iter().map(|&q| { let c = crate::engine::px::unpremultiply(q); [c[0], c[1], c[2]] }).collect();
        let ss = crate::engine::metrics::ssim(&ours, &ps, w, h, None);
        let de = crate::engine::metrics::mean_delta_e(&ours, &ps, None);
        sum += ss;
        if ss < 0.85 {
            below += 1;
        }
        let img = image::RgbImage::from_raw(w as u32, h as u32, ours.iter().flatten().copied().collect()).unwrap();
        let _ = img.save(dir.join(format!("pf_gallery_{i}.png")));
        println!("| {i} | {} | {ss:.3}{} | {de:.1} | {ms:.0} ms |", g.title(), if ss < 0.85 { " ✗" } else { "" });
    }
    println!("\nmean SSIM {:.3}; {below} below 0.85", sum / if only.is_some() { 1.0 } else { G::ALL.len() as f64 });
}

/// Info readouts and the Measurement Log vs Photoshop: Lab / HSB / CMYK of
/// 200 random colours (not on the CMYK table's grid), and a disc selection's
/// measurements plus the Gray Value of flat fills.
fn phase_f_info(dir: &Path) {
    use crate::engine::color;
    println!("\n## Phase F — Info readouts and Measurement Log vs Photoshop\n");
    let text = std::fs::read_to_string(dir.join("info_F.txt")).unwrap_or_default().replace('\r', "\n");
    let rows: Vec<Vec<f64>> = text.lines().filter_map(|l| l.split_whitespace().map(|v| v.parse().ok()).collect::<Option<Vec<f64>>>()).filter(|r| r.len() == 13).collect();
    if rows.len() < 5113 {
        println!("(no info_F.txt in {}: run tools/photoshop_gaps.sh {} F)", dir.display(), dir.display());
    } else {
        let test = &rows[4913..];
        let (mut lab, mut lab0, mut hsb, mut cmyk, mut naive) = ([0f64; 2], [0f64; 2], [0f64; 2], [0f64; 2], [0f64; 2]);
        let acc = |a: &mut [f64; 2], d: f64| {
            a[0] += d / test.len() as f64;
            a[1] = a[1].max(d);
        };
        for r in test {
            let c = [r[0] as u8, r[1] as u8, r[2] as u8];
            let l = color::srgb_to_lab(c);
            acc(&mut lab, (0..3).map(|i| (l[i] - r[7 + i]).powi(2)).sum::<f64>().sqrt());
            // Scripting's SolidColor.lab reads neutrals as (L, −0.5, −0.5).
            acc(&mut lab0, (0..3).map(|i| (l[i] - r[7 + i] - if i > 0 { 0.5 } else { 0.0 }).powi(2)).sum::<f64>().sqrt());
            let h = color::srgb_to_hsb(c);
            let dh = ((h[0] - r[10] + 540.0) % 360.0 - 180.0).abs() * if h[1] < 1.0 { 0.0 } else { 1.0 };
            acc(&mut hsb, dh.max((h[1] - r[11]).abs()).max((h[2] - r[12]).abs()));
            let k = color::srgb_to_cmyk(c);
            acc(&mut cmyk, (0..4).map(|i| (k[i] - r[3 + i]).abs()).fold(0.0, f64::max));
            let k = color::srgb_to_cmyk_naive(c);
            acc(&mut naive, (0..4).map(|i| (k[i] - r[3 + i]).abs()).fold(0.0, f64::max));
        }
        println!("| Readout (200 random sRGB colours) | mean difference | worst |");
        println!("|---|---:|---:|");
        println!("| Lab D50 (ΔE76) | {:.3} | {:.3} |", lab[0], lab[1]);
        println!("| Lab D50 (ΔE76), without scripting's −0.5 a/b offset | {:.3} | {:.3} |", lab0[0], lab0[1]);
        println!("| HSB (largest of H°, S %, B %) | {:.3} | {:.3} |", hsb[0], hsb[1]);
        println!("| CMYK, SWOP v2 table (largest ink %) | {:.3} | {:.3} |", cmyk[0], cmyk[1]);
        println!("| CMYK, naive formula (before) | {:.2} | {:.2} |", naive[0], naive[1]);
    }
    // Measurement Log: Photoshop's CSV is UTF-16 with decimal commas.
    let Ok(bytes) = std::fs::read(dir.join("measure_F.csv")) else {
        println!("\n(no measure_F.csv)");
        return;
    };
    let u16s: Vec<u16> = bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
    let csv = String::from_utf16_lossy(&u16s).replace('\r', "\n");
    let num = |f: &str| f.trim_matches(|c| c == '"' || c == '\u{feff}').replace(',', ".").parse::<f64>().ok();
    let lines: Vec<Vec<String>> = csv.lines().filter(|l| !l.trim().is_empty()).map(|l| l.split('\t').map(|s| s.to_string()).collect()).collect();
    let head = &lines[0];
    let col = |name: &str| head.iter().position(|h| h.trim_matches(|c| c == '"' || c == '\u{feff}') == name);
    let (ia, ip, ic, ig) = (col("Area").unwrap(), col("Perimeter").unwrap(), col("Circularity").unwrap(), col("Gray Value (Mean)").unwrap());
    let first = &lines[1];
    // The same disc on a gray-100 document.
    let sel = crate::engine::selection::Selection::ellipse(crate::engine::grid::RectI::new(100, 100, 500, 500));
    let mask = crate::engine::selection::MaskGrid::from_selection(&sel, 700, 600);
    let t0 = std::time::Instant::now();
    let ours = crate::engine::measure::selection(&mask, &|_, _| crate::engine::measure::ps_gray([100, 100, 100]), &Default::default(), "tmp").unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let ps_ms = std::fs::read_to_string(dir.join("timings_F.txt")).unwrap_or_default().replace('\r', "\n").lines().find(|l| l.starts_with("Record Measurements")).and_then(|l| l.split('\t').nth(1).map(|s| s.to_string())).unwrap_or_default();
    let pi = std::f64::consts::PI;
    println!("\n| Disc selection, r = 200 px (exact area 125 664, perimeter 1256.6) | PixelForge | Photoshop 2026 |");
    println!("|---|---:|---:|");
    println!("| Area | {:.0} | {:.0} |", ours.area.unwrap(), num(&first[ia]).unwrap_or(f64::NAN));
    println!("| Perimeter | {:.1} | {:.1} |", ours.perimeter.unwrap(), num(&first[ip]).unwrap_or(f64::NAN));
    println!("| Circularity | {:.4} | {:.4} |", ours.circularity.unwrap(), num(&first[ic]).unwrap_or(f64::NAN));
    println!("| Gray Value (Mean), fill 100 | {:.2} | {:.2} |", ours.gray.unwrap(), num(&first[ig]).unwrap_or(f64::NAN));
    println!("| Record Measurements time | {ms:.1} ms | {ps_ms} ms |");
    let _ = pi;
    // Gray Value of flat fills (grays 0, 16, …, 255 and four colours).
    let mut fills: Vec<[u8; 3]> = (0..=16).map(|i| [(i * 16).min(255) as u8; 3]).collect();
    fills.extend([[255, 0, 0], [0, 255, 0], [0, 0, 255], [200, 150, 50]]);
    let mut worst: (f64, [u8; 3]) = (0.0, [0; 3]);
    let mut worst_gray = 0.0f64;
    for (c, row) in fills.iter().zip(&lines[2..]) {
        let ps = num(&row[ig]).unwrap_or(f64::NAN);
        let d = (crate::engine::measure::ps_gray(*c) as f64 - ps).abs();
        if c[0] == c[1] && c[1] == c[2] {
            worst_gray = worst_gray.max(d);
        }
        if d > worst.0 {
            worst = (d, *c);
        }
    }
    println!("| Gray Value of 17 flat grays, worst difference | {worst_gray:.2} | |");
    println!("| Gray Value of 21 fills incl. colours, worst | {:.2} (at {:?}) | |", worst.0, worst.1);
}

/// Rotated crop vs Photoshop's Rotate Canvas 8° + centred crop (Bicubic
/// Automatic): SSIM of each PixelForge filter against Photoshop's pixels,
/// and the times; plus the Content-Aware corner fill's time.
fn phase_e_crop(dir: &Path) {
    use crate::engine::crop::{self, Interp};
    use crate::engine::{annot, xform};
    println!("\n## Phase E — rotated crop vs Photoshop (dog photo, 8°, 3200 × 2000)\n");
    let photo = match crate::engine::io::open_image(Path::new("testdata/photos/dog_grass.jpg")) {
        Ok(g) => g,
        Err(e) => return println!("(no dog photo: {e})"),
    };
    let ps_times: Vec<(String, String)> = std::fs::read_to_string(dir.join("timings_E.txt"))
        .unwrap_or_default()
        .replace('\r', "\n")
        .lines()
        .filter_map(|l| l.split_once('\t').map(|(a, b)| (a.to_string(), b.to_string())))
        .collect();
    let ps_time = |k: &str| ps_times.iter().find(|t| t.0.starts_with(k)).map_or("—".to_string(), |t| format!("{} ms", t.1));
    let rgb = |g: &crate::engine::grid::TileGrid| -> Vec<[u8; 3]> {
        let mut v = vec![0u32; (g.w * g.h) as usize];
        g.read_region(0, 0, g.w as usize, g.h as usize, &mut v);
        v.iter().map(|&p| { let q = crate::engine::px::unpack(p); [q[0], q[1], q[2]] }).collect()
    };
    let ps = image::open(dir.join("crop_rot8.png")).ok().map(|i| i.to_rgb8());
    let layers = vec![crate::engine::document::Layer::new(1, "bg", photo.clone())];
    // Photoshop's rotated canvas has odd dimensions, so the script's centred
    // crop can land half a pixel off ours: align on the best half-pixel shift.
    let mut shift = (0.0f32, 0.0f32);
    if let Some(p) = &ps {
        let mut best = f64::MAX;
        for dy in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            for dx in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
                let mm = crate::engine::filters::distort::mat_mul(&[1.0, 0.0, 1600.0 - 1920.0 + dx, 0.0, 1.0, 1000.0 - 1280.0 + dy, 0.0, 0.0, 1.0], &annot::rotation_about(8.0, 1920.0, 1280.0));
                let g = crop::warp_affine(&photo, &mm, 3200, 2000, Interp::Bilinear);
                let mut e = 0.0f64;
                for y in (600..1400).step_by(3) {
                    for x in (1000..2200).step_by(3) {
                        let (a, b) = (crate::engine::px::unpack(g.pixel(x, y)), p.get_pixel(x as u32, y as u32));
                        e += (0..3).map(|c| (a[c] as f64 - b[c] as f64).abs()).sum::<f64>();
                    }
                }
                if e < best {
                    best = e;
                    shift = (dx, dy);
                }
            }
        }
        println!("(Photoshop's crop sits at a ({}, {}) px offset; compared there)\n", shift.0, shift.1);
    }
    println!("| Filter | SSIM vs Photoshop | PF time | PS time |");
    println!("|---|---:|---:|---:|");
    let m = crate::engine::filters::distort::mat_mul(&[1.0, 0.0, 1600.0 - 1920.0 + shift.0, 0.0, 1.0, 1000.0 - 1280.0 + shift.1, 0.0, 0.0, 1.0], &annot::rotation_about(8.0, 1920.0, 1280.0));
    for (name, interp) in [("bilinear (mip)", None), ("bicubic", Some(Interp::Bicubic)), ("Lanczos-3", Some(Interp::Lanczos))] {
        let o = xform::XformOpts { interp, ..Default::default() };
        let mut best = f64::MAX;
        let mut r = None;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            r = Some(xform::canvas_transformed_with(&layers, &[], &Default::default(), &m, 3200, 2000, o));
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let ours = rgb(&r.unwrap().layers[0].grid);
        let ss = match &ps {
            Some(p) if p.dimensions() == (3200, 2000) => {
                let pv: Vec<[u8; 3]> = p.pixels().map(|q| [q[0], q[1], q[2]]).collect();
                format!("{:.4}", crate::engine::metrics::ssim(&ours, &pv, 3200, 2000, None))
            }
            Some(p) => format!("(PS size {:?})", p.dimensions()),
            None => "(no crop_rot8.png)".into(),
        };
        println!("| {name} | {ss} | {best:.0} ms | {} |", ps_time("Rotate canvas"));
    }
    // Content-Aware: the 8° turn's corners.
    let turned = crop::warp_affine(&photo, &annot::rotation_about(8.0, 1920.0, 1280.0), 3840, 2560, Interp::Bicubic);
    let hole = crop::new_area(&annot::rotation_about(8.0, 1920.0, 1280.0), 3840, 2560, 3840, 2560);
    let t0 = std::time::Instant::now();
    let filled = crop::content_aware(&turned, &hole);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let _ = crate::engine::io::export(&[crate::engine::document::Layer::new(1, "f", filled)], 3840, 2560, &dir.join("pf_caf_rot8.png"));
    println!("\n| Content-Aware crop fill (corners of an 8° turn, 3840 × 2560) | PF {ms:.0} ms | PS {} |", ps_time("Content-aware"));
    let t = [1.0, 0.0, -400.0, 0.0, 1.0, -300.0, 0.0, 0.0, 1.0];
    let o = xform::XformOpts { keep: true, ..Default::default() };
    let t0 = std::time::Instant::now();
    let r = xform::canvas_transformed_with(&layers, &[], &Default::default(), &t, 3200, 2000, o);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(r.layers[0].hidden.is_some());
    println!("| Crop 3200 × 2000, Delete Cropped Pixels off | PF {ms:.0} ms | PS {} |", ps_time("Crop, Delete"));
}

/// Web export vs Photoshop: the same 1920 × 1280 source, PNG-8 (128
/// colours, dithered), PNG-24 and JPEG; size, encode time and SSIM against
/// the source.
fn phase_d_web(dir: &Path) {
    use crate::engine::webexport::{self, WebFormat};
    println!("\n## Phase D — web export vs Photoshop (dog photo at 1920 × 1280)\n");
    let Ok(src) = image::open(dir.join("web_src.png")) else {
        println!("(no web_src.png in {}: run tools/photoshop_gaps.sh {} D)", dir.display(), dir.display());
        return;
    };
    let src = src.to_rgba8();
    let (w, h) = src.dimensions();
    let gray = |img: &image::RgbaImage| -> Vec<[u8; 3]> { img.pixels().map(|p| [p[0], p[1], p[2]]).collect() };
    let gs = gray(&src);
    let ps: Vec<(String, u64, f64)> = std::fs::read_to_string(dir.join("web_D.txt"))
        .unwrap_or_default()
        .replace('\r', "\n")
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Some((f.first()?.to_string(), f.get(1)?.parse().ok()?, f.get(2)?.parse().ok()?))
        })
        .collect();
    println!("| Format | PF size | PF SSIM | PF time | PS size | PS SSIM | PS time |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for (name, ext, fmt) in [("png8", "png", WebFormat::Png8 { colors: 128, dither: true, transparency: false }), ("png24", "png", WebFormat::Png24 { transparency: false }), ("jpeg60", "jpg", WebFormat::Jpeg { quality: 60 })] {
        let mut best = f64::MAX;
        let mut bytes = vec![];
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            bytes = webexport::encode(src.as_raw(), w, h, fmt, [255, 255, 255]).unwrap();
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let ours = image::load_from_memory(&bytes).unwrap().to_rgba8();
        let pf_ssim = crate::engine::metrics::ssim(&gs, &gray(&ours), w as usize, h as usize, None);
        let (ps_size, ps_ssim, ps_ms) = match (ps.iter().find(|r| r.0 == name), image::open(dir.join(format!("web_{name}.{ext}")))) {
            (Some(r), Ok(img)) => (r.1 as f64 / 1024.0, crate::engine::metrics::ssim(&gs, &gray(&img.to_rgba8()), w as usize, h as usize, None), r.2),
            _ => (f64::NAN, f64::NAN, f64::NAN),
        };
        println!("| {name} | {:.0} KB | {pf_ssim:.4} | {best:.0} ms | {ps_size:.0} KB | {ps_ssim:.4} | {ps_ms:.0} ms |", bytes.len() as f64 / 1024.0);
        // JPEG qualities aren't comparable across apps: also match Photoshop's size.
        if name == "jpeg60" && ps_size.is_finite() {
            let target = ps_size * 1024.0;
            let (mut lo, mut hi) = (1u8, 100u8);
            while lo < hi {
                let mid = (lo + hi).div_ceil(2);
                let n = webexport::encode(src.as_raw(), w, h, WebFormat::Jpeg { quality: mid }, [255, 255, 255]).unwrap().len() as f64;
                if n <= target { lo = mid } else { hi = mid - 1 }
            }
            let t0 = std::time::Instant::now();
            let b = webexport::encode(src.as_raw(), w, h, WebFormat::Jpeg { quality: lo }, [255, 255, 255]).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let img = image::load_from_memory(&b).unwrap().to_rgba8();
            let ss = crate::engine::metrics::ssim(&gs, &gray(&img), w as usize, h as usize, None);
            println!("| jpeg, size-matched (PF q{lo}) | {:.0} KB | {ss:.4} | {ms:.0} ms | {ps_size:.0} KB | {ps_ssim:.4} | {ps_ms:.0} ms |", b.len() as f64 / 1024.0);
        }
    }
}

/// Path operations vs Photoshop: the same two 500-anchor paths, each op's
/// area (exact polygon, and selection coverage at 5200² like Photoshop's
/// Make Selection) and time (boolean + selection vs Photoshop's Make Selection).
fn phase_c_paths(dir: &Path) {
    use crate::engine::pathops::{self, BoolOp};
    use crate::engine::vector::{self, FillRule};
    println!("\n## Phase C — path operations vs Photoshop (two 500-anchor paths, 5200² document)\n");
    let ps: Vec<(String, f64, f64)> = std::fs::read_to_string(dir.join("paths_C.txt"))
        .unwrap_or_default()
        .replace('\r', "\n")
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Some((f.first()?.to_string(), f.get(1)?.parse().ok()?, f.get(2)?.parse().ok()?))
        })
        .collect();
    if ps.is_empty() {
        println!("(no paths_C.txt in {}: run tools/photoshop_gaps.sh {} C)", dir.display(), dir.display());
    }
    let (a, b) = (vector::noisy_loop(2500.0, 2500.0, 2000.0, 1).flatten(0.1), vector::noisy_loop(2650.0, 2600.0, 2000.0, 2).flatten(0.1));
    println!("| Operation | PF exact area | PF selection area | PS selection area | difference | PF boolean + selection | PS Make Selection |");
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for (name, op) in [("Combine", BoolOp::Union), ("Subtract", BoolOp::Subtract), ("Intersect", BoolOp::Intersect), ("Exclude", BoolOp::Xor)] {
        let mut best = f64::MAX;
        let mut cov = 0.0;
        let mut exact = 0.0;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let res = pathops::boolean(&a, FillRule::NonZero, &b, FillRule::NonZero, op);
            let m = vector::rasterize_mask(&res, 5200, 5200, FillRule::NonZero);
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
            exact = pathops::area(&res);
            cov = (0..m.tiles.len())
                .map(|i| match &m.tiles[i] {
                    crate::engine::selection::MaskTile::Full => (256 * 256) as f64,
                    crate::engine::selection::MaskTile::Partial(t) => t.iter().map(|&v| v as f64 / 255.0).sum(),
                    crate::engine::selection::MaskTile::Empty => 0.0,
                })
                .sum::<f64>();
        }
        let p = ps.iter().find(|r| r.0 == name);
        let (pa, pt) = p.map_or((f64::NAN, f64::NAN), |r| (r.1, r.2));
        println!("| {name} | {exact:.0} | {cov:.0} | {pa:.0} | {:+.3} % | {best:.1} ms | {pt:.1} ms |", (cov - pa) / pa * 100.0);
    }
}

fn phase_a(dir: &Path) {
    println!("## Phase A — segmentation (IoU vs hand-made ground truth)\n");
    println!("| Photo | PF Select Subject | PF Object Selection (GT box) | Photoshop Select Subject | PF time | PS time |");
    println!("|---|---:|---:|---:|---:|---:|");
    let mut fills = Vec::new();
    for name in ["portrait_hair", "dog_grass", "butterfly"] {
        let Some(ph) = Photo::load(name) else {
            println!("| {name} | (photo missing) | | | | |");
            continue;
        };
        let (w, h) = (ph.grid.w, ph.grid.h);
        let src = ph.src();
        let ml = ph.mips.len() as u32;
        let t0 = Instant::now();
        let (subj, backend) = objects::select_subject(&src, w, h, ml);
        let pf_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let pf = subj.map(|m| mask_bytes(&m)).unwrap_or_else(|| vec![0; (w * h) as usize]);
        if let Ok(d) = std::env::var("PF_DUMP") {
            let _ = image::GrayImage::from_raw(w, h, pf.clone()).map(|g| g.save(Path::new(&d).join(format!("{name}_pf_subject.png"))));
        }
        let gt_name = if name == "butterfly" { "butterfly_flower" } else { name };
        let gt = gray_png(Path::new(&format!("testdata/photos/gt/{gt_name}.png"))).filter(|g| (g.1, g.2) == (w, h)).map(|g| g.0);
        let ps = gray_png(&dir.join(format!("{name}_subject.png"))).filter(|g| (g.1, g.2) == (w, h)).map(|g| g.0);
        if let (Ok(d), Some(g)) = (std::env::var("PF_DUMP"), &gt) {
            // Red: PixelForge only; blue: ground truth only; white: both.
            let diff: Vec<u8> = pf.iter().zip(g).flat_map(|(&a, &b)| match (a >= 128, b >= 128) {
                (true, true) => [255, 255, 255],
                (true, false) => [255, 0, 0],
                (false, true) => [0, 80, 255],
                _ => [0, 0, 0],
            }).collect();
            let _ = image::RgbImage::from_raw(w, h, diff).map(|g| g.save(Path::new(&d).join(format!("{name}_diff.png"))));
        }
        // Object Selection with the ground truth's bounding box (+3 %).
        let obj_gt = gray_png(Path::new(&format!("testdata/photos/gt/{name}.png"))).filter(|g| (g.1, g.2) == (w, h)).map(|g| g.0);
        let obj = obj_gt.as_ref().map(|g| {
            let (mut x0, mut y0, mut x1, mut y1) = (w as i32, h as i32, 0, 0);
            for (i, &v) in g.iter().enumerate() {
                if v >= 128 {
                    let (x, y) = ((i as u32 % w) as i32, (i as u32 / w) as i32);
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
            let pad = ((x1 - x0).max(y1 - y0) as f32 * 0.03) as i32;
            let r = RectI::new(x0 - pad, y0 - pad, x1 + pad, y1 + pad);
            let m = objects::object_in_box(&src, w, h, ml, r).or_else(|| segment::object_select(&src, w, h, ml, r, None, 1.0e6, true));
            iou(&m.map(|m| mask_bytes(&m)).unwrap_or_default(), g)
        });
        let f = |v: Option<f64>| v.map_or("—".to_string(), |v| format!("{v:.3}"));
        println!(
            "| {name} | {} ({backend:?}) | {} | {} | {pf_ms:.0} ms | {} |",
            f(gt.as_ref().map(|g| iou(&pf, g))),
            f(obj),
            f(gt.as_ref().zip(ps.as_ref()).map(|(g, p)| iou(p, g))),
            ps_ms(dir, &format!("Select Subject {name}")).map_or("—".into(), |v| format!("{v:.0} ms")),
        );
        if name == "butterfly" {
            // Photoshop may pick the butterfly alone: also score against that.
            if let Some(g2) = gray_png(Path::new("testdata/photos/gt/butterfly.png")).filter(|g| (g.1, g.2) == (w, h)).map(|g| g.0) {
                println!("| butterfly (vs butterfly-only truth) | {} | | {} | | |", f(Some(iou(&pf, &g2))), f(ps.as_ref().map(|p| iou(p, &g2))));
            }
        }
        // Content-aware fills on the same photo.
        let rgb = ph.rgb();
        for (label, cx, cy, rx, ry) in holes(name) {
            let r = RectI::new((cx - rx) as i32, (cy - ry) as i32, (cx + rx) as i32, (cy + ry) as i32);
            let hole = MaskGrid::from_selection(&Selection::ellipse(r), w, h);
            let sampling = retouch::caf_sampling(&ph.grid, &hole, retouch::SamplingArea::Auto);
            let adapt = patchmatch::Adapt { color: patchmatch::ColorAdapt::Default, ..Default::default() };
            let t0 = Instant::now();
            let res = retouch::content_aware_fill(&ph.grid, &hole, &sampling, adapt);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            let Some(res) = res else { continue };
            let filled = retouch::write_region(&ph.grid, res.region, &res.px, &res.cov);
            // SSIM inside the hole (region crop), vs the original pixels.
            let rr = res.region;
            let (rw, rh) = (rr.w() as usize, rr.h() as usize);
            let crop = |g: &dyn Fn(usize, usize) -> [u8; 3]| -> Vec<[u8; 3]> { (0..rw * rh).map(|i| g(rr.x0 as usize + i % rw, rr.y0 as usize + i / rw)).collect() };
            let orig = crop(&|x, y| rgb[y * w as usize + x]);
            let pfc = crop(&|x, y| {
                let [r, g, b, _] = px::unpremultiply(filled.pixel(x as i32, y as i32));
                [r, g, b]
            });
            let mask: Vec<u8> = (0..rw * rh)
                .map(|i| {
                    let (x, y) = ((rr.x0 as usize + i % rw) as f32 + 0.5, (rr.y0 as usize + i / rw) as f32 + 0.5);
                    ((((x - cx) / rx).powi(2) + ((y - cy) / ry).powi(2)) < 1.0) as u8
                })
                .collect();
            let s_pf = (ssim(&orig, &pfc, rw, rh, Some(&mask)), mean_delta_e(&orig, &pfc, Some(&mask)));
            let s_ps = rgb_png(&dir.join(format!("{name}_caf_{label}.png"))).filter(|v| v.len() == rgb.len()).map(|ps| {
                let psc = crop(&|x, y| ps[y * w as usize + x]);
                (ssim(&orig, &psc, rw, rh, Some(&mask)), mean_delta_e(&orig, &psc, Some(&mask)))
            });
            if let Ok(d) = std::env::var("PF_DUMP") {
                // Original | PixelForge | Photoshop, side by side.
                let ps_img = rgb_png(&dir.join(format!("{name}_caf_{label}.png"))).filter(|v| v.len() == rgb.len());
                let mut strip = image::RgbImage::new(rw as u32 * 3, rh as u32);
                for i in 0..rw * rh {
                    let (x, y) = ((i % rw) as u32, (i / rw) as u32);
                    strip.put_pixel(x, y, image::Rgb(orig[i]));
                    strip.put_pixel(x + rw as u32, y, image::Rgb(pfc[i]));
                    if let Some(ps) = &ps_img {
                        strip.put_pixel(x + 2 * rw as u32, y, image::Rgb(ps[(rr.y0 as usize + i / rw) * w as usize + rr.x0 as usize + i % rw]));
                    }
                }
                let _ = strip.save(Path::new(&d).join(format!("{name}_caf_{label}_strip.png")));
            }
            fills.push((name, label, s_pf, s_ps, ms, ps_ms(dir, &format!("Content-Aware Fill {label} {name}"))));
        }
    }
    println!("\n## Phase A — Content-Aware Fill (SSIM inside the hole vs the original pixels)\n");
    println!("| Photo, hole | PixelForge SSIM / ΔE00 | Photoshop SSIM / ΔE00 | PixelForge | Photoshop |");
    println!("|---|---:|---:|---:|---:|");
    for (name, label, s_pf, s_ps, ms, ps) in fills {
        println!(
            "| {name}, {label} | {:.3} / {:.1} | {} | {ms:.0} ms | {} |",
            s_pf.0,
            s_pf.1,
            s_ps.map_or("—".into(), |v| format!("{:.3} / {:.1}", v.0, v.1)),
            ps.map_or("—".into(), |v| format!("{v:.0} ms"))
        );
    }
}

/// Warp shapes: Photoshop's warped block outlines (`warps.txt`, one
/// subpath per line: `STYLE bend h v | x,y,lx,ly,rx,ry;...`) against
/// PixelForge's warp of the same unwarped blocks, as mask IoU.
fn phase_b_warps(dir: &Path) {
    use crate::engine::vector::{Anchor, FillRule, Path as VPath, Poly, SubPath, p, rasterize_mask};
    use crate::engine::warp::{BoxWarp, Warp, WarpStyle};
    let Ok(txt) = std::fs::read_to_string(dir.join("warps.txt")) else {
        println!("(no warps.txt in {}: run tools/photoshop_gaps.sh {} B)", dir.display(), dir.display());
        return;
    };
    let mut configs: Vec<(String, Vec<Vec<[f32; 6]>>)> = Vec::new();
    for line in txt.split(['\r', '\n']).filter(|l| !l.trim().is_empty()) {
        let Some((key, pts)) = line.split_once(" | ") else { continue };
        let sp: Vec<[f32; 6]> = pts
            .split(';')
            .filter_map(|q| {
                let v: Vec<f32> = q.split(',').filter_map(|x| x.parse().ok()).collect();
                (v.len() == 6).then(|| [v[0], v[1], v[2], v[3], v[4], v[5]])
            })
            .collect();
        match configs.last_mut() {
            Some((k, v)) if *k == key => v.push(sp),
            _ => configs.push((key.to_string(), vec![sp])),
        }
    }
    let Some((_, base)) = configs.iter().find(|(k, _)| k.starts_with("NONE")) else { return };
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for sp in base {
        for q in sp {
            x0 = x0.min(q[0]);
            x1 = x1.max(q[0]);
            y0 = y0.min(q[1]);
            y1 = y1.max(q[1]);
        }
    }
    // Photoshop warps the text's line box: for this Menlo test text it
    // extends 11.34 px above and 14.83 px below the blocks' ink.
    let bx = (x0, y0 - 11.34, x1, y1 + 14.83);
    let (cw, ch) = (3200u32, 2400u32);
    let shift = |q: crate::engine::vector::P| p(q.x + 400.0, q.y + 600.0);
    let ps_mask = |sps: &Vec<Vec<[f32; 6]>>| {
        // Photoshop path points: anchor, leftDirection (in), rightDirection (out).
        let path = VPath {
            subs: sps
                .iter()
                .map(|sp| SubPath {
                    anchors: sp.iter().map(|q| Anchor { p: shift(p(q[0], q[1])), h_in: shift(p(q[4], q[5])), h_out: shift(p(q[2], q[3])), smooth: false }).collect(),
                    closed: true,
                })
                .collect(),
        };
        rasterize_mask(&path.flatten(0.25), cw, ch, FillRule::NonZero)
    };
    let mut rows: Vec<(String, f64)> = Vec::new();
    for (key, sps) in &configs {
        if key.starts_with("NONE") {
            continue;
        }
        let f: Vec<&str> = key.split_whitespace().collect();
        let style = match f[0] {
            "ARC" => WarpStyle::Arc,
            "ARCLOWER" => WarpStyle::ArcLower,
            "ARCUPPER" => WarpStyle::ArcUpper,
            "ARCH" => WarpStyle::Arch,
            "BULGE" => WarpStyle::Bulge,
            "SHELLLOWER" => WarpStyle::ShellLower,
            "SHELLUPPER" => WarpStyle::ShellUpper,
            "FLAG" => WarpStyle::Flag,
            "WAVE" => WarpStyle::Wave,
            "FISH" => WarpStyle::Fish,
            "RISE" => WarpStyle::Rise,
            "FISHEYE" => WarpStyle::Fisheye,
            "INFLATE" => WarpStyle::Inflate,
            "SQUEEZE" => WarpStyle::Squeeze,
            _ => WarpStyle::Twist,
        };
        let num = |s: &str| s.parse::<f32>().unwrap_or(0.0) / 100.0;
        let warp = Warp { style, bend: num(f[1]), h_distort: num(f[2]), v_distort: num(f[3]), vertical: false };
        let bw = BoxWarp::new(&warp, bx);
        let ours: Vec<Poly> = base
            .iter()
            .map(|sp| {
                let n = sp.len();
                let mut pts = Vec::new();
                for k in 0..n {
                    let (a, c) = (sp[k], sp[(k + 1) % n]);
                    for t in 0..48 {
                        let t = t as f32 / 48.0;
                        pts.push(shift(bw.map(p(a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t))));
                    }
                }
                Poly { pts, closed: true }
            })
            .collect();
        let mo = rasterize_mask(&ours, cw, ch, FillRule::NonZero);
        let mp = ps_mask(sps);
        rows.push((key.clone(), iou(&mask_bytes(&mo), &mask_bytes(&mp))));
    }
    println!("## Phase B — warp styles vs Photoshop (outline IoU, 3 × 10 blocks of Menlo █)\n");
    println!("| Style | bend 50 | bend −50 | bend 100 | h 40 | v 40 | h&v 40 | worst |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for style in ["ARC", "ARCLOWER", "ARCUPPER", "ARCH", "BULGE", "SHELLLOWER", "SHELLUPPER", "FLAG", "WAVE", "FISH", "RISE", "FISHEYE", "INFLATE", "SQUEEZE", "TWIST"] {
        let get = |k: &str| rows.iter().find(|r| r.0 == format!("{style} {k}")).map_or("—".to_string(), |r| format!("{:.3}", r.1));
        let worst = rows.iter().filter(|r| r.0.starts_with(&format!("{style} "))).map(|r| r.1).fold(1.0f64, f64::min);
        println!("| {style} | {} | {} | {} | {} | {} | {} | {worst:.3} |", get("50 0 0"), get("-50 0 0"), get("100 0 0"), get("50 40 0"), get("50 0 40"), get("50 40 40"));
    }
    if let Ok(st) = std::env::var("PF_WARP_DETAIL") {
        for r in rows.iter().filter(|r| r.0.starts_with(&format!("{st} "))) {
            println!("  {:24} {:.3}", r.0, r.1);
        }
    }
    let mean = rows.iter().map(|r| r.1).sum::<f64>() / rows.len().max(1) as f64;
    println!("\n{} settings, mean IoU {mean:.4}", rows.len());
}
