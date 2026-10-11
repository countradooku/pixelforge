//! Baseline JPEG encoder with optimized Huffman tables (Photoshop's
//! "Optimized Baseline"): JFIF YCbCr, 4:2:0 chroma below quality 90 (4:4:4
//! above), the standard Annex K quantization tables scaled IJG-style by
//! quality, and Huffman codes built from the image's own symbol statistics
//! (Annex K.2, limited to 16 bits), which makes files ~5–10 % smaller than
//! with the default tables at the same quality.
//!
//! Encoding stays on the CPU (no GPU path), like PNG/TIFF writing: the
//! DCT is cheap next to Huffman coding, which is a bit-serial stream with
//! data-dependent lengths, and the output goes to disk anyway, so moving
//! the pixels to the GPU and back would only add transfers.

use rayon::prelude::*;

const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45,
    38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

const STD_LUMA: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64,
    78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

const STD_CHROMA: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// IJG quality scaling of a base table (row-major).
fn scaled(base: &[u16; 64], quality: u8) -> [u16; 64] {
    let q = quality.clamp(1, 100) as u32;
    let s = if q < 50 { 5000 / q } else { 200 - 2 * q };
    base.map(|v| ((v as u32 * s + 50) / 100).clamp(1, 255) as u16)
}

/// Quantized coefficients of one 8 × 8 block, in zigzag order.
type Block = [i16; 64];

/// Forward DCT (separable, f32) and quantization of the 8 × 8 `px`
/// (level-shifted samples), into zigzag order.
fn fdct_quant(px: &[f32; 64], q: &[u16; 64], cos: &[[f32; 8]; 8]) -> Block {
    let mut tmp = [0f32; 64];
    for y in 0..8 {
        for u in 0..8 {
            let mut s = 0.0;
            for x in 0..8 {
                s += px[y * 8 + x] * cos[u][x];
            }
            tmp[y * 8 + u] = s;
        }
    }
    let mut out = [0i16; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut s = 0.0;
            for y in 0..8 {
                s += tmp[y * 8 + u] * cos[v][y];
            }
            let k = v * 8 + u;
            out[ZIGZAG.iter().position(|&z| z == k).unwrap()] = (s / q[k] as f32).round() as i16;
        }
    }
    out
}

fn cos_table() -> [[f32; 8]; 8] {
    std::array::from_fn(|u| {
        let c = if u == 0 { (0.125f32).sqrt() } else { 0.5 };
        std::array::from_fn(|x| c * ((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI / 16.0).cos())
    })
}

/// Bits needed for `v` (JPEG magnitude category) and its code bits.
fn category(v: i32) -> (u8, u32) {
    let a = v.unsigned_abs();
    let n = 32 - a.leading_zeros();
    let bits = if v < 0 { (v - 1) as u32 & ((1u32 << n) - 1) } else { v as u32 };
    (n as u8, bits)
}

/// Huffman code lengths (≤ 16 bits) from symbol frequencies (Annex K.2).
fn code_lengths(freq: &[u32; 257]) -> [u8; 256] {
    let mut freq: Vec<u64> = freq.iter().map(|&f| f as u64).collect();
    // A reserved symbol keeps any code from being all ones.
    freq[256] = 1;
    let mut size = [0u32; 257];
    let mut others = [-1i32; 257];
    loop {
        let mut c1 = -1i32;
        let mut v = u64::MAX;
        for (i, &f) in freq.iter().enumerate() {
            if f > 0 && f <= v {
                v = f;
                c1 = i as i32;
            }
        }
        let mut c2 = -1i32;
        v = u64::MAX;
        for (i, &f) in freq.iter().enumerate() {
            if f > 0 && f <= v && i as i32 != c1 {
                v = f;
                c2 = i as i32;
            }
        }
        if c2 < 0 {
            break;
        }
        let (c1, c2) = (c1 as usize, c2 as usize);
        freq[c1] += freq[c2];
        freq[c2] = 0;
        let mut k = c1;
        size[k] += 1;
        while others[k] >= 0 {
            k = others[k] as usize;
            size[k] += 1;
        }
        others[k] = c2 as i32;
        let mut k = c2;
        size[k] += 1;
        while others[k] >= 0 {
            k = others[k] as usize;
            size[k] += 1;
        }
    }
    // Counts per length, then limit to 16 bits.
    let mut bits = [0u32; 40];
    for &s in &size {
        if s > 0 {
            bits[s as usize] += 1;
        }
    }
    let mut i = 39;
    while i > 16 {
        while bits[i] > 0 {
            let mut j = i - 2;
            while bits[j] == 0 {
                j -= 1;
            }
            bits[i] -= 2;
            bits[i - 1] += 1;
            bits[j + 1] += 2;
            bits[j] -= 1;
        }
        i -= 1;
    }
    // Remove the reserved symbol (from the longest length).
    let mut i = 16;
    while bits[i] == 0 {
        i -= 1;
    }
    bits[i] -= 1;
    // Assign lengths to symbols by frequency order (most frequent first).
    let mut syms: Vec<usize> = (0..256).filter(|&s| size[s] > 0).collect();
    syms.sort_by_key(|&s| (size[s], s));
    let mut out = [0u8; 256];
    let mut it = syms.into_iter();
    for (len, &n) in bits.iter().enumerate().take(17).skip(1) {
        for _ in 0..n {
            if let Some(s) = it.next() {
                out[s] = len as u8;
            }
        }
    }
    out
}

/// A Huffman table: DHT `bits` / `vals` and the per-symbol (code, length).
struct Huff {
    bits: [u8; 16],
    vals: Vec<u8>,
    code: [(u16, u8); 256],
}

fn huff(freq: &[u32; 257]) -> Huff {
    let lens = code_lengths(freq);
    let mut bits = [0u8; 16];
    let mut vals = Vec::new();
    for len in 1..=16u8 {
        for s in 0..256 {
            if lens[s] == len {
                bits[len as usize - 1] += 1;
                vals.push(s as u8);
            }
        }
    }
    let mut code = [(0u16, 0u8); 256];
    let mut c = 0u16;
    let mut k = 0;
    for len in 1..=16u8 {
        for _ in 0..bits[len as usize - 1] {
            code[vals[k] as usize] = (c, len);
            c += 1;
            k += 1;
        }
        c <<= 1;
    }
    Huff { bits, vals, code }
}

struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    fn put(&mut self, bits: u32, len: u8) {
        if len == 0 {
            return;
        }
        self.acc = (self.acc << len) | (bits & ((1 << len) - 1));
        self.n += len as u32;
        while self.n >= 8 {
            let b = (self.acc >> (self.n - 8)) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0);
            }
            self.n -= 8;
        }
        self.acc &= (1 << self.n) - 1;
    }

    fn flush(&mut self) {
        if self.n > 0 {
            self.put(0x7F, (8 - self.n) as u8);
        }
    }
}

/// One component's blocks in scan order (MCU-interleaved later).
struct Plane {
    blocks: Vec<Block>,
    /// Blocks per row.
    bw: usize,
}

/// The block at `(bx, by)`, clamped to the plane (MCU padding repeats the edge).
fn block_at(p: &Plane, bx: usize, by: usize) -> &Block {
    let rows = p.blocks.len() / p.bw;
    &p.blocks[by.min(rows - 1) * p.bw + bx.min(p.bw - 1)]
}

fn plane_blocks(samples: &[f32], w: usize, h: usize, q: &[u16; 64], cos: &[[f32; 8]; 8]) -> Plane {
    let (bw, bh) = (w.div_ceil(8), h.div_ceil(8));
    let blocks = (0..bw * bh)
        .into_par_iter()
        .map(|b| {
            let (bx, by) = (b % bw, b / bw);
            let mut px = [0f32; 64];
            for y in 0..8 {
                let sy = (by * 8 + y).min(h - 1);
                for x in 0..8 {
                    let sx = (bx * 8 + x).min(w - 1);
                    px[y * 8 + x] = samples[sy * w + sx] - 128.0;
                }
            }
            fdct_quant(&px, q, cos)
        })
        .collect();
    Plane { blocks, bw }
}

/// Encodes straight RGB8 `w × h` as a baseline JPEG with optimized Huffman
/// tables.
pub fn encode(rgb: &[u8], w: u32, h: u32, quality: u8) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let sub = quality < 90;
    let (ql, qc) = (scaled(&STD_LUMA, quality), scaled(&STD_CHROMA, quality));
    let cos = cos_table();
    // Colour conversion (JFIF), then chroma subsampling.
    let mut yp = vec![0f32; w * h];
    let mut cb = vec![0f32; w * h];
    let mut cr = vec![0f32; w * h];
    yp.par_chunks_mut(w).zip(cb.par_chunks_mut(w)).zip(cr.par_chunks_mut(w)).enumerate().for_each(|(y, ((yr, br), rr))| {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (r, g, b) = (rgb[i] as f32, rgb[i + 1] as f32, rgb[i + 2] as f32);
            yr[x] = 0.299 * r + 0.587 * g + 0.114 * b;
            br[x] = -0.168_736 * r - 0.331_264 * g + 0.5 * b + 128.0;
            rr[x] = 0.5 * r - 0.418_688 * g - 0.081_312 * b + 128.0;
        }
    });
    let (cw, ch) = if sub { (w.div_ceil(2), h.div_ceil(2)) } else { (w, h) };
    let down = |p: &[f32]| -> Vec<f32> {
        if !sub {
            return p.to_vec();
        }
        (0..cw * ch)
            .into_par_iter()
            .map(|i| {
                let (x, y) = ((i % cw) * 2, (i / cw) * 2);
                let at = |dx: usize, dy: usize| p[(y + dy).min(h - 1) * w + (x + dx).min(w - 1)];
                (at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25
            })
            .collect()
    };
    let (cb, cr) = (down(&cb), down(&cr));
    // The luma plane is padded to whole MCUs (16 × 16 with subsampling).
    let mcu = if sub { 16 } else { 8 };
    let (mw, mh) = (w.div_ceil(mcu), h.div_ceil(mcu));
    let ly = plane_blocks(&yp, w, h, &ql, &cos);
    let (lcb, lcr) = (plane_blocks(&cb, cw, ch, &qc, &cos), plane_blocks(&cr, cw, ch, &qc, &cos));
    // MCU order: (component, block) visits.
    let mut seq: Vec<(usize, &Block)> = Vec::with_capacity(ly.blocks.len() + lcb.blocks.len() * 2);
    for my in 0..mh {
        for mx in 0..mw {
            if sub {
                for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                    seq.push((0, block_at(&ly, mx * 2 + dx, my * 2 + dy)));
                }
            } else {
                seq.push((0, block_at(&ly, mx, my)));
            }
            seq.push((1, block_at(&lcb, mx, my)));
            seq.push((2, block_at(&lcr, mx, my)));
        }
    }
    // Pass 1: symbol statistics (luma / chroma tables).
    let mut dcf = [[0u32; 257]; 2];
    let mut acf = [[0u32; 257]; 2];
    let mut pred = [0i32; 3];
    let walk = |blk: &Block, c: usize, pred: &mut [i32; 3], mut emit: Box<dyn FnMut(bool, u8, u32, u8) + '_>| {
        let t = (c > 0) as usize;
        let d = blk[0] as i32 - pred[c];
        pred[c] = blk[0] as i32;
        let (n, b) = category(d);
        emit(true, n, b, t as u8);
        let mut run = 0;
        let last = (1..64).rev().find(|&k| blk[k] != 0).unwrap_or(0);
        for &v in &blk[1..=last] {
            if v == 0 {
                run += 1;
                continue;
            }
            while run > 15 {
                emit(false, 0xF0, 0, t as u8);
                run -= 16;
            }
            let (n, b) = category(v as i32);
            emit(false, (run << 4) as u8 | n, b, t as u8);
            run = 0;
        }
        if last < 63 {
            emit(false, 0x00, 0, t as u8);
        }
    };
    for &(c, blk) in &seq {
        walk(
            blk,
            c,
            &mut pred,
            Box::new(|dc, sym, _, t| {
                if dc {
                    dcf[t as usize][sym as usize] += 1;
                } else {
                    acf[t as usize][sym as usize] += 1;
                }
            }),
        );
    }
    let dc = [huff(&dcf[0]), huff(&dcf[1])];
    let ac = [huff(&acf[0]), huff(&acf[1])];
    // Headers.
    let mut out = vec![0xFF, 0xD8];
    out.extend_from_slice(&[0xFF, 0xE0, 0, 16, b'J', b'F', b'I', b'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]);
    for (id, q) in [(0u8, &ql), (1, &qc)] {
        out.extend_from_slice(&[0xFF, 0xDB, 0, 67, id]);
        out.extend(ZIGZAG.iter().map(|&k| q[k] as u8));
    }
    let hs = if sub { 0x22 } else { 0x11 };
    out.extend_from_slice(&[0xFF, 0xC0, 0, 17, 8, (h >> 8) as u8, h as u8, (w >> 8) as u8, w as u8, 3, 1, hs, 0, 2, 0x11, 1, 3, 0x11, 1]);
    for (class, id, t) in [(0u8, 0u8, &dc[0]), (1, 0, &ac[0]), (0, 1, &dc[1]), (1, 1, &ac[1])] {
        let len = 2 + 1 + 16 + t.vals.len();
        out.extend_from_slice(&[0xFF, 0xC4, (len >> 8) as u8, len as u8, (class << 4) | id]);
        out.extend_from_slice(&t.bits);
        out.extend_from_slice(&t.vals);
    }
    out.extend_from_slice(&[0xFF, 0xDA, 0, 12, 3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0]);
    // Pass 2: entropy-coded data.
    let mut bw = BitWriter { out: Vec::with_capacity(w * h / 4), acc: 0, n: 0 };
    let mut pred = [0i32; 3];
    for &(c, blk) in &seq {
        walk(
            blk,
            c,
            &mut pred,
            Box::new(|isdc, sym, bits, t| {
                let tab = if isdc { &dc[t as usize] } else { &ac[t as usize] };
                let (code, len) = tab.code[sym as usize];
                bw.put(code as u32, len);
                bw.put(bits, sym & 0x0F);
            }),
        );
    }
    bw.flush();
    out.extend(bw.out);
    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn photo(w: usize, h: usize) -> Vec<u8> {
        let g = crate::engine::ops::procedural(w as u32, h as u32);
        let mut v = vec![0u32; w * h];
        g.read_region(0, 0, w, h, &mut v);
        v.iter().flat_map(|&p| {
            let c = crate::engine::px::unpremultiply(p);
            [c[0], c[1], c[2]]
        }).collect()
    }

    #[test]
    fn decodes_and_beats_default_tables() {
        let (w, h) = (517usize, 333usize);
        let rgb = photo(w, h);
        for q in [30u8, 60, 92] {
            let ours = encode(&rgb, w as u32, h as u32, q);
            let img = image::load_from_memory(&ours).expect("valid JPEG").to_rgb8();
            assert_eq!((img.width(), img.height()), (w as u32, h as u32));
            let mae: f64 = img.as_raw().iter().zip(&rgb).map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>() / rgb.len() as f64;
            assert!(mae < if q < 50 { 9.0 } else { 5.0 }, "q{q} mae {mae}");
            // The image crate's encoder with its default tables, same quality.
            let mut def = Vec::new();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut def, q).encode(&rgb, w as u32, h as u32, image::ExtendedColorType::Rgb8).unwrap();
            assert!(ours.len() < def.len(), "q{q}: {} vs {}", ours.len(), def.len());
        }
    }

    #[test]
    fn huffman_lengths_are_a_valid_prefix_code() {
        let mut f = [0u32; 257];
        for (i, v) in f.iter_mut().enumerate().take(200) {
            *v = (i as u32 * 7919) % 1000 + 1;
        }
        f[3] = 1_000_000;
        let lens = code_lengths(&f);
        assert!(lens.iter().all(|&l| l <= 16));
        let kraft: f64 = lens.iter().filter(|&&l| l > 0).map(|&l| 0.5f64.powi(l as i32)).sum();
        assert!(kraft < 1.0, "kraft {kraft}");
        assert_eq!(lens.iter().filter(|&&l| l > 0).count(), 200);
    }
}
