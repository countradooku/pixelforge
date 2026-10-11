//! "Save for Web"-style export helpers for a raster editor.
//!
//! * [`encode`] — encode one RGBA image as PNG-8 (real indexed PNG), PNG-24, JPEG or WebP.
//! * [`quantize`] — palette quantization (median cut in Oklab + k-means refinement,
//!   optional serpentine Floyd–Steinberg dithering).
//! * [`pdf`] — minimal multi-page PDF 1.4 writer ("Export Artboards to PDF").
//! * [`html_table`] — table-layout HTML page reproducing a slice grid ("HTML and Images").
//!
//! All pixel inputs are straight (non-premultiplied) RGBA8, row-major, `w * h * 4` bytes.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::OnceLock;

use image::ExtendedColorType;
use image::codecs::webp::WebPEncoder;
use rayon::prelude::*;

/// Output format for [`encode`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WebFormat {
    /// Indexed PNG (colour type 3) with at most `colors` palette entries (clamped to 2..=256).
    /// With `transparency`, fully transparent pixels share one palette entry and partially
    /// transparent pixels keep their alpha via `tRNS`; otherwise everything is composited
    /// over the matte.
    Png8 { colors: u16, dither: bool, transparency: bool },
    /// Truecolour PNG; RGBA when `transparency` and the image actually has alpha, RGB
    /// (composited over the matte) otherwise.
    Png24 { transparency: bool },
    /// Baseline JPEG at `quality` (clamped to 1..=100), alpha composited over the matte.
    Jpeg { quality: u8 },
    /// WebP. The `image` crate only ships a *lossless* (VP8L) encoder, so this is always
    /// lossless; alpha is preserved and the matte is ignored.
    Webp,
}

// ---------------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------------

/// Validates dimensions against the buffer length; returns the pixel count.
fn check_buf(rgba: &[u8], w: u32, h: u32) -> Result<usize, String> {
    if w == 0 || h == 0 {
        return Err(format!("invalid image size {w}x{h}"));
    }
    let n = (w as usize)
        .checked_mul(h as usize)
        .filter(|n| n.checked_mul(4).is_some())
        .ok_or_else(|| format!("image size {w}x{h} overflows"))?;
    if rgba.len() != n * 4 {
        return Err(format!(
            "buffer is {} bytes, expected {} for {w}x{h} RGBA",
            rgba.len(),
            n * 4
        ));
    }
    Ok(n)
}

/// `c` over `m` with straight alpha `a`, rounded.
#[inline]
fn over(c: u8, a: u8, m: u8) -> u8 {
    let a = a as u32;
    ((c as u32 * a + m as u32 * (255 - a) + 127) / 255) as u8
}

/// Composites RGBA over an opaque matte, producing packed RGB8.
fn rgb_over(rgba: &[u8], matte: [u8; 3]) -> Vec<u8> {
    let mut out = vec![0u8; rgba.len() / 4 * 3];
    let convert = |(o, p): (&mut [u8], &[u8])| {
        for c in 0..3 {
            o[c] = over(p[c], p[3], matte[c]);
        }
    };
    // Small slices are already encoded concurrently. Avoid a nested join
    // stealing another whole codec job while this short conversion waits.
    if rgba.len() <= 4 * 1024 * 1024 {
        out.chunks_exact_mut(3).zip(rgba.chunks_exact(4)).for_each(convert);
    } else {
        out.par_chunks_exact_mut(3).zip(rgba.par_chunks_exact(4)).for_each(convert);
    }
    out
}

fn is_opaque(rgba: &[u8]) -> bool {
    if rgba.len() <= 4 * 1024 * 1024 {
        rgba.chunks_exact(4).all(|p| p[3] == 255)
    } else {
        rgba.par_chunks_exact(4).all(|p| p[3] == 255)
    }
}

/// Writes a PNG with the `png` crate.
fn png_bytes(
    w: u32,
    h: u32,
    color: png::ColorType,
    depth: png::BitDepth,
    data: &[u8],
    plte: Option<Vec<u8>>,
    trns: Option<Vec<u8>>,
) -> Result<Vec<u8>, String> {
    let err = |e: png::EncodingError| format!("PNG encoding failed: {e}");
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(color);
        enc.set_depth(depth);
        enc.set_compression(png::Compression::High);
        if let Some(p) = plte {
            enc.set_palette(p);
        }
        if let Some(t) = trns {
            enc.set_trns(t);
        }
        let mut wr = enc.write_header().map_err(err)?;
        wr.write_image_data(data).map_err(err)?;
        wr.finish().map_err(err)?;
    }
    Ok(out)
}

/// Writes an indexed PNG; uses bit depth 1/2/4 for small palettes, and emits `tRNS` only
/// when some entry is not fully opaque (truncated after the last non-opaque entry).
fn indexed_png(w: u32, h: u32, palette: &[[u8; 4]], idx: &[u8]) -> Result<Vec<u8>, String> {
    let bits: u32 = match palette.len() {
        0..=2 => 1,
        3..=4 => 2,
        5..=16 => 4,
        _ => 8,
    };
    let depth = match bits {
        1 => png::BitDepth::One,
        2 => png::BitDepth::Two,
        4 => png::BitDepth::Four,
        _ => png::BitDepth::Eight,
    };
    let data = if bits == 8 {
        idx.to_vec()
    } else {
        // Pack MSB-first, each row padded to a whole byte.
        let w = w as usize;
        let per_byte = (8 / bits) as usize;
        let stride = w.div_ceil(per_byte);
        let mut packed = vec![0u8; stride * h as usize];
        packed
            .par_chunks_exact_mut(stride)
            .zip(idx.par_chunks_exact(w))
            .for_each(|(dst, src)| {
                for (x, &i) in src.iter().enumerate() {
                    let shift = 8 - bits as usize * (x % per_byte + 1);
                    dst[x / per_byte] |= i << shift;
                }
            });
        packed
    };
    let plte: Vec<u8> = palette.iter().flat_map(|c| [c[0], c[1], c[2]]).collect();
    let trns = palette
        .iter()
        .rposition(|c| c[3] < 255)
        .map(|last| palette[..=last].iter().map(|c| c[3]).collect::<Vec<u8>>());
    png_bytes(w, h, png::ColorType::Indexed, depth, &data, Some(plte), trns)
}

// ---------------------------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------------------------

/// Encodes one image in the given web format and returns the file bytes.
///
/// `matte` is the background used wherever alpha has to be flattened (JPEG, PNG-24/PNG-8
/// without transparency).
pub fn encode(rgba: &[u8], w: u32, h: u32, fmt: WebFormat, matte: [u8; 3]) -> Result<Vec<u8>, String> {
    check_buf(rgba, w, h)?;
    match fmt {
        WebFormat::Png8 { colors, dither, transparency } => {
            let (palette, idx) = quantize(rgba, w, h, colors, dither, transparency, matte);
            indexed_png(w, h, &palette, &idx)
        }
        WebFormat::Png24 { transparency } => {
            if transparency && !is_opaque(rgba) {
                png_bytes(w, h, png::ColorType::Rgba, png::BitDepth::Eight, rgba, None, None)
            } else {
                let rgb = rgb_over(rgba, matte);
                png_bytes(w, h, png::ColorType::Rgb, png::BitDepth::Eight, &rgb, None, None)
            }
        }
        WebFormat::Jpeg { quality } => jpeg_bytes(&rgb_over(rgba, matte), w, h, quality),
        WebFormat::Webp => {
            let mut out = Vec::new();
            let enc = WebPEncoder::new_lossless(&mut out);
            let res = if is_opaque(rgba) {
                enc.encode(&rgb_over(rgba, matte), w, h, ExtendedColorType::Rgb8)
            } else {
                enc.encode(rgba, w, h, ExtendedColorType::Rgba8)
            };
            res.map_err(|e| format!("WebP encoding failed: {e}"))?;
            Ok(out)
        }
    }
}

/// Optimized-baseline JPEG (Huffman tables from the image's statistics).
fn jpeg_bytes(rgb: &[u8], w: u32, h: u32, quality: u8) -> Result<Vec<u8>, String> {
    if w == 0 || h == 0 {
        return Err("JPEG encoding failed: empty image".into());
    }
    Ok(super::jpeg::encode(rgb, w, h, quality.clamp(1, 100)))
}

// ---------------------------------------------------------------------------------------------
// Colour space (Oklab + alpha) used for all quantizer distances
// ---------------------------------------------------------------------------------------------

type Feat = [f32; 4];

#[inline]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn lin_lut() -> &'static [f32; 256] {
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))
}

/// Linear RGB + alpha (0..1) → Oklab L, a, b plus alpha as a fourth axis.
#[inline]
fn oklab(r: f32, g: f32, b: f32, a: f32) -> Feat {
    let l = (0.412_221_47 * r + 0.536_332_55 * g + 0.051_445_995 * b).cbrt();
    let m = (0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b).cbrt();
    let s = (0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b).cbrt();
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
        a,
    ]
}

#[inline]
fn feat_u8(p: [u8; 4]) -> Feat {
    let lut = lin_lut();
    oklab(lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize], p[3] as f32 / 255.0)
}

/// Feature of a (possibly fractional) sRGB colour in 0..255.
fn feat_f64(c: [f64; 4]) -> Feat {
    let l = |v: f64| srgb_to_linear((v / 255.0) as f32);
    oklab(l(c[0]), l(c[1]), l(c[2]), (c[3] / 255.0) as f32)
}

#[inline]
fn dist2(a: &Feat, b: &Feat) -> f32 {
    (0..4).map(|i| (a[i] - b[i]) * (a[i] - b[i])).sum()
}

/// Nearest-neighbour search over a small palette, sorted by lightness so the scan can
/// stop once the lightness gap alone exceeds the best distance found.
struct Nearest {
    pts: Vec<(Feat, u16)>,
}

impl Nearest {
    fn new(feats: &[Feat]) -> Self {
        let mut pts: Vec<(Feat, u16)> = feats.iter().enumerate().map(|(i, f)| (*f, i as u16)).collect();
        pts.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]));
        Nearest { pts }
    }

    fn find(&self, q: &Feat) -> u16 {
        let n = self.pts.len();
        let start = self.pts.partition_point(|p| p.0[0] < q[0]);
        let (mut best, mut best_i) = (f32::INFINITY, 0u16);
        let (mut up, mut down) = (start, start); // scan [start..) upward and (..start) downward
        let (mut up_live, mut down_live) = (true, true);
        while up_live || down_live {
            if up_live {
                if up < n && {
                    let d = self.pts[up].0[0] - q[0];
                    d * d < best
                } {
                    let d = dist2(&self.pts[up].0, q);
                    if d < best {
                        best = d;
                        best_i = self.pts[up].1;
                    }
                    up += 1;
                } else {
                    up_live = false;
                }
            }
            if down_live {
                if down > 0 && {
                    let d = self.pts[down - 1].0[0] - q[0];
                    d * d < best
                } {
                    down -= 1;
                    let d = dist2(&self.pts[down].0, q);
                    if d < best {
                        best = d;
                        best_i = self.pts[down].1;
                    }
                } else {
                    down_live = false;
                }
            }
        }
        best_i
    }
}

/// Small direct-mapped cache from packed RGBA to palette index.
struct Cache {
    slots: Vec<(u32, u16)>,
}

impl Cache {
    const BITS: u32 = 12;
    const EMPTY: u16 = u16::MAX;

    fn new() -> Self {
        Cache { slots: vec![(0, Self::EMPTY); 1 << Self::BITS] }
    }

    #[inline]
    fn get_or(&mut self, p: [u8; 4], f: impl FnOnce() -> u16) -> u16 {
        let key = u32::from_le_bytes(p);
        let slot = (key.wrapping_mul(0x9E37_79B1) >> (32 - Self::BITS)) as usize;
        let (k, v) = self.slots[slot];
        if v != Self::EMPTY && k == key {
            return v;
        }
        let v = f();
        self.slots[slot] = (key, v);
        v
    }
}

// ---------------------------------------------------------------------------------------------
// quantize
// ---------------------------------------------------------------------------------------------

/// Maximum number of distinct histogram entries before colours are bucketed more coarsely.
const HIST_LIMIT: usize = 1 << 18;
const KMEANS_PASSES: usize = 3;
/// Sentinel cluster id for the shared fully-transparent entry.
const TRANS: u16 = u16::MAX;

/// One histogram entry: mean colour of a bucket, its feature vector, and pixel count.
#[derive(Clone, Copy)]
struct Entry {
    f: Feat,
    c: [f64; 4],
    w: f64,
}

/// Weighted accumulator for centroid updates.
#[derive(Clone, Copy, Default)]
struct Acc {
    f: [f64; 4],
    c: [f64; 4],
    w: f64,
}

impl Acc {
    fn add(&mut self, e: &Entry) {
        for i in 0..4 {
            self.f[i] += e.f[i] as f64 * e.w;
            self.c[i] += e.c[i] * e.w;
        }
        self.w += e.w;
    }
    fn merge(mut self, o: &Acc) -> Acc {
        for i in 0..4 {
            self.f[i] += o.f[i];
            self.c[i] += o.c[i];
        }
        self.w += o.w;
        self
    }
}

/// Builds the colour histogram of non-transparent pixels. If there are too many distinct
/// colours, keys are coarsened (6, 5, then 4 bits per channel); bucket colours stay exact means.
fn histogram(px: &[[u8; 4]], skip_transparent: bool) -> Vec<Entry> {
    const MASKS: [u8; 4] = [0xFF, 0xFC, 0xF8, 0xF0];
    'masks: for (mi, &m) in MASKS.iter().enumerate() {
        let mask = u32::from_le_bytes([m; 4]);
        let mut map: HashMap<u32, ([u64; 4], u64)> = HashMap::new();
        for p in px {
            if skip_transparent && p[3] == 0 {
                continue;
            }
            let acc = map.entry(u32::from_le_bytes(*p) & mask).or_default();
            for (sum, &v) in acc.0.iter_mut().zip(p) {
                *sum += v as u64;
            }
            acc.1 += 1;
            if map.len() > HIST_LIMIT && mi + 1 < MASKS.len() {
                continue 'masks;
            }
        }
        let mut keyed: Vec<(u32, Entry)> = map
            .into_iter()
            .map(|(k, (s, n))| {
                let c = s.map(|v| v as f64 / n as f64);
                (k, Entry { f: feat_f64(c), c, w: n as f64 })
            })
            .collect();
        // Deterministic order regardless of HashMap iteration order.
        keyed.sort_unstable_by_key(|(k, _)| *k);
        return keyed.into_iter().map(|(_, e)| e).collect();
    }
    unreachable!("last mask always returns")
}

/// Weighted sum of squared deviations from the mean (two-pass for robustness).
fn sse(es: &[Entry]) -> f64 {
    let acc = es.iter().fold(Acc::default(), |mut a, e| {
        a.add(e);
        a
    });
    let mean = acc.f.map(|v| v / acc.w);
    es.iter()
        .map(|e| e.w * (0..4).map(|i| (e.f[i] as f64 - mean[i]).powi(2)).sum::<f64>())
        .sum()
}

/// Median-cut style split of the histogram into at most `k` boxes. Each step splits the box
/// with the largest SSE along its highest-variance axis, at the position that minimises the
/// summed SSE of both halves (Wu-style optimal cut). Returns the box ranges.
fn median_cut(es: &mut [Entry], k: usize) -> Vec<(usize, usize)> {
    struct Bx {
        s: usize,
        e: usize,
        sse: f64,
    }
    let mut boxes = vec![Bx { s: 0, e: es.len(), sse: sse(es) }];
    while boxes.len() < k {
        let Some(bi) = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.e - b.s > 1 && b.sse > 0.0)
            .max_by(|a, b| a.1.sse.total_cmp(&b.1.sse))
            .map(|(i, _)| i)
        else {
            break;
        };
        let b = boxes.swap_remove(bi);
        let seg = &mut es[b.s..b.e];

        // Axis of largest weighted variance.
        let (mut w, mut s1, mut s2) = (0.0, [0.0f64; 4], [0.0f64; 4]);
        for e in seg.iter() {
            w += e.w;
            for i in 0..4 {
                let v = e.f[i] as f64;
                s1[i] += e.w * v;
                s2[i] += e.w * v * v;
            }
        }
        let axis = (0..4)
            .max_by(|&a, &b| (s2[a] - s1[a] * s1[a] / w).total_cmp(&(s2[b] - s1[b] * s1[b] / w)))
            .unwrap_or(0);
        seg.sort_unstable_by(|a, b| a.f[axis].total_cmp(&b.f[axis]));

        // Best split position by total SSE of both halves (prefix sums over all axes).
        let (mut lw, mut l1, mut l2) = (0.0, [0.0f64; 4], [0.0f64; 4]);
        let (mut best, mut split) = (f64::INFINITY, seg.len() / 2);
        for (i, e) in seg[..seg.len() - 1].iter().enumerate() {
            lw += e.w;
            for d in 0..4 {
                let v = e.f[d] as f64;
                l1[d] += e.w * v;
                l2[d] += e.w * v * v;
            }
            let rw = w - lw;
            let cost: f64 = (0..4)
                .map(|d| {
                    let r1 = s1[d] - l1[d];
                    let r2 = s2[d] - l2[d];
                    (l2[d] - l1[d] * l1[d] / lw) + (r2 - r1 * r1 / rw)
                })
                .sum();
            if cost < best {
                best = cost;
                split = i + 1;
            }
        }
        let mid = b.s + split;
        boxes.push(Bx { s: b.s, e: mid, sse: sse(&es[b.s..mid]) });
        boxes.push(Bx { s: mid, e: b.e, sse: sse(&es[mid..b.e]) });
    }
    let mut r: Vec<(usize, usize)> = boxes.iter().map(|b| (b.s, b.e)).collect();
    r.sort_unstable();
    r
}

/// Computes `k` palette colours (sRGB means, 0..255 floats) for the histogram.
fn cluster(es: &mut [Entry], k: usize) -> Vec<[f64; 4]> {
    if es.len() <= k {
        return es.iter().map(|e| e.c).collect();
    }
    let ranges = median_cut(es, k);
    let mut cents: Vec<Acc> = ranges
        .iter()
        .map(|&(s, e)| {
            es[s..e].iter().fold(Acc::default(), |mut a, x| {
                a.add(x);
                a
            })
        })
        .collect();

    // k-means refinement: assign every histogram entry to its nearest centroid, then move
    // each centroid to the weighted mean of its members (empty clusters keep their position).
    let n = cents.len();
    for _ in 0..KMEANS_PASSES {
        let feats: Vec<Feat> = cents.iter().map(|a| a.f.map(|v| (v / a.w) as f32)).collect();
        let nn = Nearest::new(&feats);
        let sums = es
            .par_iter()
            .fold(
                || vec![Acc::default(); n],
                |mut acc, e| {
                    acc[nn.find(&e.f) as usize].add(e);
                    acc
                },
            )
            .reduce(
                || vec![Acc::default(); n],
                |a, b| a.iter().zip(&b).map(|(x, y)| x.merge(y)).collect(),
            );
        for (c, s) in cents.iter_mut().zip(sums) {
            if s.w > 0.0 {
                *c = s;
            }
        }
    }
    cents.iter().map(|a| a.c.map(|v| v / a.w)).collect()
}

/// Palette quantization. Returns `(palette RGBA, one index per pixel)`.
///
/// * `colors` is clamped to `2..=256`; the palette may be smaller if the image has fewer
///   distinct colours (unused entries are dropped).
/// * Clustering: histogram → median cut (optimal-SSE cuts) in Oklab + alpha → 3 k-means passes.
///   Images with no more distinct colours than the budget are reproduced exactly.
/// * `transparency`: fully transparent pixels share a single `[0, 0, 0, 0]` entry (counted
///   against `colors`); partially transparent pixels are clustered with alpha as a 4th axis.
///   Without it every pixel is composited over `matte` and the palette is opaque.
/// * Non-opaque entries are ordered first so a `tRNS` chunk can stay short.
/// * `dither`: serpentine Floyd–Steinberg error diffusion in sRGB (+ alpha).
///
/// # Panics
/// If `rgba.len() != w * h * 4`.
pub fn quantize(
    rgba: &[u8],
    w: u32,
    h: u32,
    colors: u16,
    dither: bool,
    transparency: bool,
    matte: [u8; 3],
) -> (Vec<[u8; 4]>, Vec<u8>) {
    assert_eq!(rgba.len() as u64, w as u64 * h as u64 * 4, "rgba buffer size mismatch");
    let (wu, hu) = (w as usize, h as usize);
    if wu * hu == 0 {
        return (Vec::new(), Vec::new());
    }

    // Normalise input: flatten onto the matte, or zero the colour of fully transparent pixels.
    let px: Vec<[u8; 4]> = rgba
        .par_chunks_exact(4)
        .map(|p| {
            if !transparency {
                [over(p[0], p[3], matte[0]), over(p[1], p[3], matte[1]), over(p[2], p[3], matte[2]), 255]
            } else if p[3] == 0 {
                [0; 4]
            } else {
                [p[0], p[1], p[2], p[3]]
            }
        })
        .collect();
    let has_trans = transparency && px.par_iter().any(|p| p[3] == 0);
    let budget = (colors.clamp(2, 256) as usize - has_trans as usize).max(1);

    let mut hist = histogram(&px, transparency);
    let pal_f = cluster(&mut hist, budget);
    let pal: Vec<[u8; 4]> = pal_f
        .iter()
        .map(|c| {
            let mut q = c.map(|v| v.round().clamp(0.0, 255.0) as u8);
            if !transparency {
                q[3] = 255;
            }
            q
        })
        .collect();
    let nn = Nearest::new(&pal.iter().map(|&c| feat_u8(c)).collect::<Vec<_>>());

    // Map pixels to cluster ids (TRANS for the shared transparent entry).
    let mut ids = vec![0u16; px.len()];
    if pal.is_empty() {
        ids.fill(TRANS); // everything transparent
    } else if dither {
        dither_fs(&px, wu, hu, &pal, &nn, transparency, &mut ids);
    } else {
        const CHUNK: usize = 1 << 14;
        ids.par_chunks_mut(CHUNK).zip(px.par_chunks(CHUNK)).for_each(|(out, inp)| {
            let mut cache = Cache::new();
            for (o, &p) in out.iter_mut().zip(inp) {
                *o = if transparency && p[3] == 0 {
                    TRANS
                } else {
                    cache.get_or(p, || nn.find(&feat_u8(p)))
                };
            }
        });
    }

    // Compact: keep used entries only; transparent first, then translucent, then opaque.
    let mut used = vec![false; pal.len()];
    let mut trans_used = false;
    for &i in &ids {
        if i == TRANS {
            trans_used = true;
        } else {
            used[i as usize] = true;
        }
    }
    let mut order: Vec<usize> = (0..pal.len()).filter(|&i| used[i]).collect();
    order.sort_by_key(|&i| pal[i][3] == 255);
    let mut palette = Vec::with_capacity(order.len() + 1);
    let mut remap = vec![0u8; pal.len()];
    let trans_idx = 0u8;
    if trans_used {
        palette.push([0, 0, 0, 0]);
    }
    for &i in &order {
        remap[i] = palette.len() as u8;
        palette.push(pal[i]);
    }
    let idx = ids
        .par_iter()
        .map(|&i| if i == TRANS { trans_idx } else { remap[i as usize] })
        .collect();
    (palette, idx)
}

/// Serpentine Floyd–Steinberg dithering into `ids`.
fn dither_fs(
    px: &[[u8; 4]],
    w: usize,
    h: usize,
    pal: &[[u8; 4]],
    nn: &Nearest,
    transparency: bool,
    ids: &mut [u16],
) {
    let palf: Vec<[f32; 4]> = pal.iter().map(|c| c.map(|v| v as f32)).collect();
    // Error rows padded by one on each side; pixel x lives at x + 1.
    let mut cur = vec![[0f32; 4]; w + 2];
    let mut next = vec![[0f32; 4]; w + 2];
    let mut cache = Cache::new();
    let chans = if transparency { 4 } else { 3 };
    for y in 0..h {
        let ltr = y % 2 == 0;
        for xi in 0..w {
            let x = if ltr { xi } else { w - 1 - xi };
            let p = px[y * w + x];
            if transparency && p[3] == 0 {
                ids[y * w + x] = TRANS;
                continue;
            }
            let e = cur[x + 1];
            let mut v = [0f32; 4];
            for c in 0..4 {
                v[c] = (p[c] as f32 + if c < chans { e[c] } else { 0.0 }).clamp(0.0, 255.0);
            }
            let q = v.map(|f| f.round() as u8);
            let i = cache.get_or(q, || nn.find(&feat_u8(q)));
            ids[y * w + x] = i;
            let (fwd, back) = if ltr { (x + 2, x) } else { (x, x + 2) };
            for c in 0..chans {
                let d = v[c] - palf[i as usize][c];
                cur[fwd][c] += d * (7.0 / 16.0);
                next[back][c] += d * (3.0 / 16.0);
                next[x + 1][c] += d * (5.0 / 16.0);
                next[fwd][c] += d * (1.0 / 16.0);
            }
        }
        std::mem::swap(&mut cur, &mut next);
        next.fill([0.0; 4]);
    }
}

// ---------------------------------------------------------------------------------------------
// PDF
// ---------------------------------------------------------------------------------------------

/// Formats a PDF real number (up to 4 decimals, no trailing zeros).
fn pdf_num(v: f64) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-" { "0".into() } else { s.to_string() }
}

/// Writes a minimal multi-page PDF 1.4: one image per page, page size in points
/// `px * 72 / dpi`. Images are embedded as `/DCTDecode` (JPEG at the given quality) when
/// `jpeg_quality` is `Some`, otherwise as zlib-compressed RGB (`/FlateDecode`). Alpha is
/// composited over white. Pages are encoded in parallel.
pub fn pdf(pages: &[(Vec<u8>, u32, u32)], dpi: f32, jpeg_quality: Option<u8>) -> Result<Vec<u8>, String> {
    if pages.is_empty() {
        return Err("PDF needs at least one page".into());
    }
    if !(dpi.is_finite() && dpi > 0.0) {
        return Err(format!("invalid dpi {dpi}"));
    }
    let images: Vec<(Vec<u8>, &str)> = pages
        .par_iter()
        .enumerate()
        .map(|(i, (rgba, w, h))| {
            check_buf(rgba, *w, *h).map_err(|e| format!("page {}: {e}", i + 1))?;
            let rgb = rgb_over(rgba, [255; 3]);
            match jpeg_quality {
                Some(q) => Ok((jpeg_bytes(&rgb, *w, *h, q)?, "/DCTDecode")),
                None => {
                    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                    z.write_all(&rgb).map_err(|e| e.to_string())?;
                    Ok((z.finish().map_err(|e| e.to_string())?, "/FlateDecode"))
                }
            }
        })
        .collect::<Result<_, String>>()?;

    // Object layout: 1 Catalog, 2 Pages, then per page i: 3+3i Page, 4+3i Image, 5+3i Contents.
    let n_obj = 2 + 3 * pages.len();
    let mut out: Vec<u8> = Vec::new();
    let mut offsets = vec![0usize; n_obj + 1];
    out.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");

    let mut obj = |out: &mut Vec<u8>, id: usize, dict: &str, stream: Option<&[u8]>| {
        offsets[id] = out.len();
        out.extend_from_slice(format!("{id} 0 obj\n{dict}\n").as_bytes());
        if let Some(s) = stream {
            out.extend_from_slice(b"stream\n");
            out.extend_from_slice(s);
            out.extend_from_slice(b"\nendstream\n");
        }
        out.extend_from_slice(b"endobj\n");
    };

    obj(&mut out, 1, "<< /Type /Catalog /Pages 2 0 R >>", None);
    let kids: Vec<String> = (0..pages.len()).map(|i| format!("{} 0 R", 3 + 3 * i)).collect();
    obj(
        &mut out,
        2,
        &format!("<< /Type /Pages /Kids [{}] /Count {} >>", kids.join(" "), pages.len()),
        None,
    );
    for (i, ((_, w, h), (data, filter))) in pages.iter().zip(&images).enumerate() {
        let (pid, iid, cid) = (3 + 3 * i, 4 + 3 * i, 5 + 3 * i);
        let pw = pdf_num(*w as f64 * 72.0 / dpi as f64);
        let ph = pdf_num(*h as f64 * 72.0 / dpi as f64);
        obj(
            &mut out,
            pid,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw} {ph}] \
                 /Resources << /XObject << /Im0 {iid} 0 R >> >> /Contents {cid} 0 R >>"
            ),
            None,
        );
        obj(
            &mut out,
            iid,
            &format!(
                "<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB \
                 /BitsPerComponent 8 /Filter {filter} /Length {} >>",
                data.len()
            ),
            Some(data),
        );
        let content = format!("q\n{pw} 0 0 {ph} 0 0 cm\n/Im0 Do\nQ\n");
        obj(
            &mut out,
            cid,
            &format!("<< /Length {} >>", content.len()),
            Some(content.as_bytes()),
        );
    }

    // Cross-reference table: every entry is exactly 20 bytes.
    let xref = out.len();
    let mut tail = format!("xref\n0 {}\n0000000000 65535 f \n", n_obj + 1);
    for off in &offsets[1..] {
        let _ = writeln!(tail, "{off:010} 00000 n ");
    }
    let _ = write!(tail, "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", n_obj + 1);
    out.extend_from_slice(tail.as_bytes());
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// HTML
// ---------------------------------------------------------------------------------------------

/// One slice for [`html_table`]: pixel rectangle `[x0, x1) × [y0, y1)` plus image file and link.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SliceCell {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    /// Image path used as `<img src>`.
    pub file: String,
    /// Link target; the image is wrapped in `<a>` when non-empty.
    pub url: String,
    /// `target` attribute of the link (omitted when empty).
    pub target: String,
    /// `alt` text of the image.
    pub alt: String,
}

/// Escapes text for HTML content and double/single-quoted attributes.
fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            c => o.push(c),
        }
    }
    o
}

/// Builds an HTML page with a table layout reproducing the slice grid, like Photoshop's
/// "HTML and Images" output, without spacer GIFs.
///
/// The grid is formed from all distinct slice edges (plus the document bounds); slices
/// spanning several grid cells get `colspan`/`rowspan`. Column widths are fixed through a
/// `<colgroup>` and row heights on each `<tr>`, so the layout is unambiguous. Slices are
/// clipped to the document; a slice overlapping an earlier one is skipped, and grid cells
/// covered by no slice become empty cells. All text is HTML-escaped.
pub fn html_table(title: &str, doc_w: u32, doc_h: u32, cells: &[SliceCell]) -> String {
    let (dw, dh) = (doc_w as i64, doc_h as i64);
    let clip = |c: &SliceCell| {
        let x0 = (c.x0 as i64).clamp(0, dw);
        let x1 = (c.x1 as i64).clamp(0, dw);
        let y0 = (c.y0 as i64).clamp(0, dh);
        let y1 = (c.y1 as i64).clamp(0, dh);
        (x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1))
    };
    let rects: Vec<(i64, i64, i64, i64)> = cells.iter().map(clip).collect();

    let edges = |a: i64, b: i64, it: &mut dyn Iterator<Item = i64>| {
        let mut v: Vec<i64> = [a, b].into_iter().chain(it).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let xs = edges(0, dw, &mut rects.iter().flat_map(|r| [r.0, r.2]));
    let ys = edges(0, dh, &mut rects.iter().flat_map(|r| [r.1, r.3]));
    let (ncol, nrow) = (xs.len().saturating_sub(1), ys.len().saturating_sub(1));
    let col_of = |x: i64| xs.binary_search(&x).unwrap_or(0);
    let row_of = |y: i64| ys.binary_search(&y).unwrap_or(0);

    // owner[r][c] = index of the slice covering that grid cell.
    let mut owner: Vec<Vec<Option<usize>>> = vec![vec![None; ncol]; nrow];
    let mut span = vec![None; cells.len()]; // (row, col, rowspan, colspan) of placed slices
    for (i, r) in rects.iter().enumerate() {
        if r.0 == r.2 || r.1 == r.3 {
            continue; // empty after clipping
        }
        let (c0, c1, r0, r1) = (col_of(r.0), col_of(r.2), row_of(r.1), row_of(r.3));
        if (r0..r1).any(|y| (c0..c1).any(|x| owner[y][x].is_some())) {
            continue; // overlaps an earlier slice
        }
        for row in owner.iter_mut().take(r1).skip(r0) {
            for o in row.iter_mut().take(c1).skip(c0) {
                *o = Some(i);
            }
        }
        span[i] = Some((r0, c0, r1 - r0, c1 - c0));
    }

    let mut h = String::new();
    let _ = writeln!(h, "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">");
    let _ = writeln!(h, "<title>{}</title>\n</head>\n<body style=\"margin:0\">", esc(title));
    let _ = writeln!(
        h,
        "<table border=\"0\" cellpadding=\"0\" cellspacing=\"0\" width=\"{doc_w}\" \
         style=\"border-collapse:collapse;table-layout:fixed;width:{doc_w}px\">"
    );
    h.push_str("<colgroup>");
    for c in 0..ncol {
        let _ = write!(h, "<col style=\"width:{}px\">", xs[c + 1] - xs[c]);
    }
    h.push_str("</colgroup>\n");
    for r in 0..nrow {
        let rh = ys[r + 1] - ys[r];
        let _ = writeln!(h, "<tr style=\"height:{rh}px\">");
        for c in 0..ncol {
            match owner[r][c] {
                Some(i) => {
                    let Some((r0, c0, rs, cs)) = span[i] else { continue };
                    if (r0, c0) != (r, c) {
                        continue; // covered by a span started elsewhere
                    }
                    let cell = &cells[i];
                    let (sw, sh) = (xs[c0 + cs] - xs[c0], ys[r0 + rs] - ys[r0]);
                    h.push_str("<td");
                    if cs > 1 {
                        let _ = write!(h, " colspan=\"{cs}\"");
                    }
                    if rs > 1 {
                        let _ = write!(h, " rowspan=\"{rs}\"");
                    }
                    h.push('>');
                    let img = format!(
                        "<img src=\"{}\" width=\"{sw}\" height=\"{sh}\" alt=\"{}\" style=\"display:block;border:0\">",
                        esc(&cell.file),
                        esc(&cell.alt)
                    );
                    if cell.url.is_empty() {
                        h.push_str(&img);
                    } else {
                        let _ = write!(h, "<a href=\"{}\"", esc(&cell.url));
                        if !cell.target.is_empty() {
                            let _ = write!(h, " target=\"{}\"", esc(&cell.target));
                        }
                        let _ = write!(h, ">{img}</a>");
                    }
                    h.push_str("</td>\n");
                }
                None => {
                    let _ = writeln!(h, "<td style=\"width:{}px;height:{rh}px\"></td>", xs[c + 1] - xs[c]);
                }
            }
        }
        h.push_str("</tr>\n");
    }
    h.push_str("</table>\n</body>\n</html>\n");
    h
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn img_from(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).flat_map(|(x, y)| f(x, y)).collect()
    }

    fn reconstruct(pal: &[[u8; 4]], idx: &[u8]) -> Vec<u8> {
        idx.iter().flat_map(|&i| pal[i as usize]).collect()
    }

    fn mae(a: &[u8], b: &[u8]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.len() as f64
    }

    /// Mean abs error between 8x8 block means (RGB) — measures banding / low-frequency error.
    fn block_mean_err(a: &[u8], b: &[u8], w: usize, h: usize) -> f64 {
        let (mut tot, mut n) = (0.0, 0);
        for by in (0..h).step_by(8) {
            for bx in (0..w).step_by(8) {
                for c in 0..3 {
                    let (mut sa, mut sb) = (0.0, 0.0);
                    for y in by..by + 8 {
                        for x in bx..bx + 8 {
                            sa += a[(y * w + x) * 4 + c] as f64;
                            sb += b[(y * w + x) * 4 + c] as f64;
                        }
                    }
                    tot += ((sa - sb) / 64.0).abs();
                    n += 1;
                }
            }
        }
        tot / n as f64
    }

    #[test]
    fn quantize_two_colours() {
        let img = img_from(16, 16, |x, _| if x < 8 { [200, 10, 30, 255] } else { [5, 90, 250, 255] });
        let (pal, idx) = quantize(&img, 16, 16, 256, false, true, [255; 3]);
        assert_eq!(pal.len(), 2);
        assert!(pal.contains(&[200, 10, 30, 255]) && pal.contains(&[5, 90, 250, 255]));
        assert_eq!(reconstruct(&pal, &idx), img);
        // Same with dithering: exact colours produce zero error, so still exact.
        let (pal, idx) = quantize(&img, 16, 16, 2, true, false, [0; 3]);
        assert_eq!(reconstruct(&pal, &idx), img);
    }

    #[test]
    fn quantize_256_roundtrip() {
        // 256 distinct, scattered colours.
        let img = img_from(16, 16, |x, y| {
            let i = y * 16 + x;
            [(i * 37 % 256) as u8, (i * 101 % 256) as u8, (255 - i) as u8, 255]
        });
        let (pal, idx) = quantize(&img, 16, 16, 256, false, true, [0; 3]);
        assert_eq!(pal.len(), 256);
        assert_eq!(reconstruct(&pal, &idx), img);
    }

    #[test]
    fn quantize_forced_clustering_low_error() {
        // 512 colours but each appears with a "twin" off by one: median cut must still give
        // 256 clusters and low error (exercises the full median-cut + k-means path).
        let img = img_from(32, 16, |x, y| {
            let i = y * 16 + x / 2;
            let t = (x % 2) as u8;
            [(i * 37 % 256) as u8 ^ t, (i * 101 % 256) as u8, (255 - i) as u8, 255]
        });
        let (pal, idx) = quantize(&img, 32, 16, 256, false, true, [0; 3]);
        assert!(pal.len() <= 256);
        assert!(mae(&reconstruct(&pal, &idx), &img) < 0.5);
    }

    #[test]
    fn quantize_gradient_error_and_dither() {
        // 1-D gradient (256 distinct colours) into 64 colours.
        let (w, h) = (256u32, 32u32);
        let g1 = img_from(w, h, |x, _| [x as u8, (x / 2 + 64) as u8, (255 - x) as u8, 255]);
        let (pal, idx) = quantize(&g1, w, h, 64, false, false, [0; 3]);
        assert!(pal.len() <= 64);
        let e = mae(&reconstruct(&pal, &idx), &g1);
        assert!(e < 2.0, "gradient MAE {e}");

        // 2-D gradient (65k colours) into 64 colours: dithering should reduce banding.
        let (w, h) = (256u32, 256u32);
        let g2 = img_from(w, h, |x, y| [x as u8, y as u8, ((x + y) / 2) as u8, 255]);
        let (p0, i0) = quantize(&g2, w, h, 64, false, false, [0; 3]);
        let (p1, i1) = quantize(&g2, w, h, 64, true, false, [0; 3]);
        let plain = reconstruct(&p0, &i0);
        let dith = reconstruct(&p1, &i1);
        assert!(mae(&plain, &g2) < 10.0, "2-D MAE {}", mae(&plain, &g2));
        let (bp, bd) = (
            block_mean_err(&plain, &g2, w as usize, h as usize),
            block_mean_err(&dith, &g2, w as usize, h as usize),
        );
        eprintln!("1-D MAE {e:.3}; 2-D MAE {:.3}; block-mean err plain {bp:.3} dithered {bd:.3}", mae(&plain, &g2));
        assert!(bd < bp * 0.6, "dithered block error {bd} vs plain {bp}");
    }

    #[test]
    fn quantize_transparency() {
        let img = img_from(8, 8, |x, y| match (x + y) % 3 {
            0 => [0, 0, 0, 0],
            1 => [255, 0, 0, 255],
            _ => [0, 0, 255, 128],
        });
        let (pal, idx) = quantize(&img, 8, 8, 16, false, true, [255; 3]);
        assert_eq!(pal[0], [0, 0, 0, 0], "transparent entry first");
        assert_eq!(pal.len(), 3);
        for (i, p) in img.chunks_exact(4).enumerate() {
            let q = pal[idx[i] as usize];
            if p[3] == 0 {
                assert_eq!(q[3], 0);
            } else {
                assert_eq!(&q[..], p);
            }
        }
        // Without transparency: composited over white, all opaque.
        let (pal, idx) = quantize(&img, 8, 8, 16, false, false, [255; 3]);
        assert!(pal.iter().all(|c| c[3] == 255));
        assert_eq!(pal[idx[0] as usize], [255, 255, 255, 255]);
        assert_eq!(pal[idx[2] as usize], [127, 127, 255, 255]);
    }

    fn decode(bytes: &[u8]) -> image::RgbaImage {
        image::load_from_memory(bytes).expect("decodes").to_rgba8()
    }

    fn photo(w: u32, h: u32) -> Vec<u8> {
        img_from(w, h, |x, y| {
            if x < 4 {
                return [0; 4];
            }
            [(x * 4) as u8, (y * 3) as u8, ((x + y) * 2) as u8, 255]
        })
    }

    #[test]
    fn encode_png8() {
        let (w, h) = (60, 40);
        let img = photo(w, h);
        let bytes = encode(&img, w, h, WebFormat::Png8 { colors: 128, dither: false, transparency: true }, [255; 3]).unwrap();
        assert_eq!(&bytes[1..4], b"PNG");
        assert_eq!(&bytes[12..16], b"IHDR");
        assert_eq!(bytes[24], 8, "bit depth");
        assert_eq!(bytes[25], 3, "colour type indexed");
        let has = |tag: &[u8]| bytes.windows(4).any(|w| w == tag);
        assert!(has(b"PLTE") && has(b"tRNS"));
        let d = decode(&bytes);
        assert_eq!(d.dimensions(), (w, h));
        assert_eq!(d.get_pixel(0, 0)[3], 0);
        assert!(mae(d.as_raw(), &img) < 3.0);

        // Opaque, no transparency: no tRNS; 2 colours → 1-bit depth.
        let two = img_from(9, 3, |x, _| if x % 2 == 0 { [0, 0, 0, 255] } else { [255, 255, 255, 255] });
        let b = encode(&two, 9, 3, WebFormat::Png8 { colors: 256, dither: true, transparency: false }, [0; 3]).unwrap();
        assert_eq!((b[24], b[25]), (1, 3));
        assert!(!b.windows(4).any(|w| w == b"tRNS"));
        assert_eq!(decode(&b).as_raw(), &two);
    }

    #[test]
    fn encode_png24_jpeg_webp() {
        let (w, h) = (60, 40);
        let img = photo(w, h);
        let matte = [255, 255, 255];
        let flat = img_from(w, h, |x, y| {
            let i = ((y * w + x) * 4) as usize;
            let p = &img[i..i + 4];
            [over(p[0], p[3], 255), over(p[1], p[3], 255), over(p[2], p[3], 255), 255]
        });

        let b = encode(&img, w, h, WebFormat::Png24 { transparency: true }, matte).unwrap();
        assert_eq!(b[25], 6);
        assert_eq!(decode(&b).as_raw(), &img);
        let b = encode(&img, w, h, WebFormat::Png24 { transparency: false }, matte).unwrap();
        assert_eq!(b[25], 2);
        assert_eq!(decode(&b).as_raw(), &flat);

        let b = encode(&img, w, h, WebFormat::Jpeg { quality: 90 }, matte).unwrap();
        assert_eq!(&b[..2], &[0xFF, 0xD8]);
        let d = decode(&b);
        assert_eq!(d.dimensions(), (w, h));
        assert!(mae(d.as_raw(), &flat) < 6.0, "jpeg mae {}", mae(d.as_raw(), &flat));

        let b = encode(&img, w, h, WebFormat::Webp, matte).unwrap();
        assert_eq!((&b[..4], &b[8..12]), (&b"RIFF"[..], &b"WEBP"[..]));
        let d = decode(&b);
        assert_eq!(d.dimensions(), (w, h));
        // Lossless; compare colour only where visible (encoder may alter hidden RGB).
        for (p, q) in d.as_raw().chunks_exact(4).zip(img.chunks_exact(4)) {
            assert_eq!(p[3], q[3]);
            if q[3] != 0 {
                assert_eq!(p, q);
            }
        }

        assert!(encode(&img, w, h + 1, WebFormat::Webp, matte).is_err());
    }

    /// Checks header/trailer and that each xref offset lands on "N 0 obj"; returns object count.
    fn check_pdf(b: &[u8]) -> usize {
        assert!(b.starts_with(b"%PDF-1.4\n"));
        assert!(b.ends_with(b"%%EOF\n"));
        // Byte-level helpers (the file contains binary image data).
        let rfind = |pat: &[u8]| b.windows(pat.len()).rposition(|w| w == pat).unwrap();
        let line_at = |at: usize| {
            let end = at + b[at..].iter().position(|&c| c == b'\n').unwrap();
            std::str::from_utf8(&b[at..end]).unwrap().to_string()
        };
        let sx = rfind(b"startxref\n") + 10;
        let xref: usize = line_at(sx).parse().unwrap();
        assert!(b[xref..].starts_with(b"xref\n0 "));
        let head = line_at(xref + 7);
        let n: usize = head.parse().unwrap();
        let table = xref + 7 + head.len() + 1;
        assert_eq!(&b[table..table + 20], b"0000000000 65535 f \n");
        for i in 1..n {
            let e = &b[table + 20 * i..table + 20 * (i + 1)];
            assert_eq!(&e[10..], b" 00000 n \n");
            let off: usize = std::str::from_utf8(&e[..10]).unwrap().parse().unwrap();
            assert!(b[off..].starts_with(format!("{i} 0 obj\n").as_bytes()), "object {i}");
        }
        let s = String::from_utf8_lossy(&b[table..]);
        assert!(s.contains(&format!("/Size {n} ")));
        n - 1
    }

    #[test]
    fn pdf_structure() {
        let pages = vec![(photo(60, 40), 60, 40), (photo(30, 50), 30, 50), (photo(10, 10), 10, 10)];
        for q in [None, Some(85)] {
            let b = pdf(&pages, 144.0, q).unwrap();
            assert_eq!(check_pdf(&b), 2 + 3 * pages.len());
            let s = String::from_utf8_lossy(&b);
            assert_eq!(s.matches("/Type /Page ").count(), 3);
            assert!(s.contains("/Count 3"));
            assert!(s.contains("/MediaBox [0 0 30 20]"), "60x40 px at 144 dpi = 30x20 pt");
            assert!(s.contains(if q.is_some() { "/DCTDecode" } else { "/FlateDecode" }));
            let name = if q.is_some() { "test_jpeg.pdf" } else { "test_flate.pdf" };
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(name), &b).unwrap();
        }
        assert!(pdf(&[], 72.0, None).is_err());
        assert!(pdf(&pages, 0.0, None).is_err());
    }

    fn cell(x0: i32, y0: i32, x1: i32, y1: i32, name: &str) -> SliceCell {
        SliceCell { x0, y0, x1, y1, file: format!("images/{name}.png"), alt: name.into(), ..Default::default() }
    }

    fn attr_sum(h: &str, prefix: &str) -> i64 {
        h.match_indices(prefix)
            .map(|(i, _)| {
                let rest = &h[i + prefix.len()..];
                rest[..rest.find("px").unwrap()].parse::<i64>().unwrap()
            })
            .sum()
    }

    #[test]
    fn html_grid_spans_and_escaping() {
        // 2x2 grid on top, one slice spanning both columns below.
        let mut cells = vec![
            cell(0, 0, 120, 50, "a"),
            cell(120, 0, 300, 50, "b"),
            cell(0, 50, 120, 100, "c"),
            cell(120, 50, 300, 100, "d"),
            cell(0, 100, 300, 160, "wide"),
        ];
        cells[0].url = "https://example.com/?a=1&b=<2>".into();
        cells[0].target = "_blank".into();
        cells[1].alt = "Tom & \"Jerry\" <x>".into();
        let h = html_table("Q&A <page>", 300, 160, &cells);

        assert!(h.contains("<title>Q&amp;A &lt;page&gt;</title>"));
        assert!(h.contains("href=\"https://example.com/?a=1&amp;b=&lt;2&gt;\" target=\"_blank\""));
        assert!(h.contains("alt=\"Tom &amp; &quot;Jerry&quot; &lt;x&gt;\""));
        assert!(h.contains("border=\"0\" cellpadding=\"0\" cellspacing=\"0\""));
        assert!(h.contains("<td colspan=\"2\"><img src=\"images/wide.png\" width=\"300\" height=\"60\""));
        assert_eq!(h.matches("<img ").count(), 5);
        assert_eq!(h.matches("<a ").count(), 1);
        assert_eq!(attr_sum(&h, "<col style=\"width:"), 300);
        assert_eq!(attr_sum(&h, "<tr style=\"height:"), 160);
        // Each row's cell widths add up to the document width.
        for row in h.split("<tr").skip(1) {
            let widths: i64 = row
                .match_indices(" width=\"")
                .map(|(i, _)| {
                    let r = &row[i + 8..];
                    r[..r.find('"').unwrap()].parse::<i64>().unwrap()
                })
                .sum();
            assert_eq!(widths, 300);
        }

        // Rowspan + uncovered gap.
        let h = html_table("t", 100, 100, &[cell(0, 0, 50, 100, "tall"), cell(50, 0, 100, 50, "tr")]);
        assert!(h.contains("<td rowspan=\"2\"><img src=\"images/tall.png\" width=\"50\" height=\"100\""));
        assert!(h.contains("<td style=\"width:50px;height:50px\"></td>"));
    }
}
