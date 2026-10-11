//! Image import/export and the native layered `.pforge` format.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use super::blend::BlendMode;
use super::composite::export_rgba;
use super::document::{Document, Layer};
use super::grid::{RectI, TS, Tile, TileGrid, build_grid, finish_tile};
use super::px;

/// Decodes any supported image and splits it into premultiplied tiles.
pub fn open_image(path: &Path) -> Result<TileGrid> {
    let reader = image::ImageReader::open(path)?.with_guessed_format()?;
    let mut reader = reader;
    reader.no_limits();
    let img = reader.decode().with_context(|| format!("decoding {}", path.display()))?;
    let rgba = img.into_rgba8();
    let (w, h) = rgba.dimensions();
    let raw = rgba.as_raw();
    let stride = w as usize * 4;
    Ok(build_grid(w, h, |tx, ty| {
        let x0 = tx as usize * TS;
        let y0 = ty as usize * TS;
        let cw = TS.min(w as usize - x0);
        let ch = TS.min(h as usize - y0);
        let mut out = vec![0u32; TS * TS].into_boxed_slice();
        for y in 0..ch {
            let row = &raw[(y0 + y) * stride + x0 * 4..(y0 + y) * stride + (x0 + cw) * 4];
            for (x, p) in row.chunks_exact(4).enumerate() {
                out[y * TS + x] = px::premultiply(p[0], p[1], p[2], p[3]);
            }
            // Replicate the edge so mips and filters see no seam at the border.
            if cw < TS {
                let last = out[y * TS + cw - 1];
                out[y * TS + cw..(y + 1) * TS].fill(last);
            }
        }
        for y in ch..TS {
            let (a, b) = out.split_at_mut(y * TS);
            b[..TS].copy_from_slice(&a[(ch - 1) * TS..ch * TS]);
        }
        finish_tile(out)
    }))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExportFormat {
    Png,
    Jpeg,
    Tiff,
    Webp,
    Bmp,
}

impl ExportFormat {
    pub fn from_path(p: &Path) -> Option<Self> {
        match p.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "png" => Some(Self::Png),
            "jpg" | "jpeg" => Some(Self::Jpeg),
            "tif" | "tiff" => Some(Self::Tiff),
            "webp" => Some(Self::Webp),
            "bmp" => Some(Self::Bmp),
            _ => None,
        }
    }
}

/// Flattens and writes the document as a regular image file.
pub fn export(layers: &[Layer], w: u32, h: u32, path: &Path) -> Result<()> {
    let fmt = ExportFormat::from_path(path).unwrap_or(ExportFormat::Png);
    let opaque = matches!(fmt, ExportFormat::Jpeg | ExportFormat::Bmp);
    let rgba = export_rgba(layers, w, h, opaque.then_some([255, 255, 255]));
    let file = BufWriter::with_capacity(1 << 22, File::create(path)?);
    match fmt {
        ExportFormat::Png => {
            use image::codecs::png::{CompressionType, FilterType, PngEncoder};
            use image::ImageEncoder;
            PngEncoder::new_with_quality(file, CompressionType::Fast, FilterType::Adaptive)
                .write_image(&rgba, w, h, image::ExtendedColorType::Rgba8)?;
        }
        ExportFormat::Jpeg => {
            use std::io::Write;
            let rgb: Vec<u8> = rgba.par_chunks_exact(4).flat_map_iter(|p| [p[0], p[1], p[2]]).collect();
            let mut file = file;
            file.write_all(&super::jpeg::encode(&rgb, w, h, 92))?;
        }
        ExportFormat::Tiff => {
            use image::ImageEncoder;
            image::codecs::tiff::TiffEncoder::new(file).write_image(&rgba, w, h, image::ExtendedColorType::Rgba8)?;
        }
        ExportFormat::Webp => {
            use image::ImageEncoder;
            image::codecs::webp::WebPEncoder::new_lossless(file).write_image(&rgba, w, h, image::ExtendedColorType::Rgba8)?;
        }
        ExportFormat::Bmp => {
            use image::ImageEncoder;
            let rgb: Vec<u8> = rgba.par_chunks_exact(4).flat_map_iter(|p| [p[0], p[1], p[2]]).collect();
            image::codecs::bmp::BmpEncoder::new(&mut BufWriter::new(file)).write_image(&rgb, w, h, image::ExtendedColorType::Rgb8)?;
        }
    }
    Ok(())
}

// ----- native format ------------------------------------------------------------
//
// "PFORGE01" | w u32 | h u32 | layer count u32 | active u32
// per layer: name len u32 + utf8 | visible u8 | opacity f32 | blend u8 | tile count u32
//   per non-empty tile: index u32 | kind u8 (1 = uniform: color u32,
//                                            2 = lz4: len u32 + bytes)
// Uniform tiles shared across the grid stay one 4-byte record each.

// Version 2 appends, after each layer's tiles, the layer model:
//   id u64 | parent u64 (0 = none) | clip u8 | kind u8 + payload | mask u8 [+ enabled u8 + u8 tiles]
// kind: 0 pixel, 1 group (pass-through u8, collapsed u8), 2 adjustment
// (catalog index u32, flag u8, n u32, f32 × n), 3 smart object (name, matrix
// f32 × 9, source w/h + tiles), 4 shape (geometry, fill, stroke, rule),
// 5 text (base style, origin, box, direction, warp style index + bend),
// 6 frame (geometry, rotation, transform, optional smart-object content),
// 7 artboard (rect, background colour, collapsed).
//
// After the layers come optional trailing chunks, each a 4-byte tag; older
// readers stop at the first one they don't know:
//   "PFDX" document extras (paths, annotations).
//   "PFLX" per-layer extensions: count u32, then per record
//          layer id u64 | tag u8 | len u32 | bytes. Unknown tags are skipped.
//          Tag 1: text extras (style runs, paragraph styles, vertical
//          options, full warp, type-on-path / area path).
const MAGIC_V1: &[u8; 8] = b"PFORGE01";
const MAGIC: &[u8; 8] = b"PFORGE02";

pub struct NativeLayer {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    pub grid: TileGrid,
    pub id: u64,
    pub parent: Option<u64>,
    pub clip: bool,
    pub kind: super::layers::LayerKind,
    pub adj: Option<(super::filters::catalog::Kind, Vec<f32>, bool)>,
    pub mask: Option<(MaskGridData, bool)>,
    /// Locked Background ("PFLX" tag 3).
    pub background: bool,
    /// Pixels beyond the canvas ("PFLX" tag 5).
    pub hidden: Option<Arc<super::crop::Hidden>>,
    /// Layer Style ("PFLX" tag 6).
    pub style: Option<Arc<super::styles::LayerStyle>>,
    /// Fill opacity ("PFLX" tag 7).
    pub fill: f32,
    pub advanced_blend: Option<Arc<super::advanced_blend::AdvancedBlend>>,
}

pub type MaskGridData = super::selection::MaskGrid;

fn tile_records(grid: &TileGrid) -> Vec<Vec<u8>> {
    grid.tiles
        .par_iter()
        .enumerate()
        .filter_map(|(i, t)| {
            let t = t.as_ref()?;
            let mut rec = Vec::with_capacity(16);
            rec.extend_from_slice(&(i as u32).to_le_bytes());
            if let Some(c) = t.uniform {
                rec.push(1);
                rec.extend_from_slice(&c.to_le_bytes());
            } else {
                let pixels = t.read();
                let bytes: &[u8] = bytemuck::cast_slice(&pixels);
                let c = lz4_flex::compress(bytes);
                rec.push(2);
                rec.extend_from_slice(&(c.len() as u32).to_le_bytes());
                rec.extend_from_slice(&c);
            }
            Some(rec)
        })
        .collect()
}

fn write_grid(f: &mut impl Write, grid: &TileGrid) -> Result<()> {
    let records = tile_records(grid);
    f.write_all(&(records.len() as u32).to_le_bytes())?;
    for r in records {
        f.write_all(&r)?;
    }
    Ok(())
}

struct W(Vec<u8>);
impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
    fn p(&mut self, q: super::vector::P) {
        self.f32(q.x);
        self.f32(q.y);
    }
}

fn write_path(w: &mut W, path: &super::vector::Path) {
    w.u32(path.subs.len() as u32);
    for sp in &path.subs {
        w.u8(sp.closed as u8);
        w.u32(sp.anchors.len() as u32);
        for a in &sp.anchors {
            w.p(a.p);
            w.p(a.h_in);
            w.p(a.h_out);
            w.u8(a.smooth as u8);
        }
    }
}

fn write_geom(w: &mut W, geom: &super::layers::ShapeGeom) {
    use super::layers::ShapeGeom as G;
    match geom {
        G::Rect { x0, y0, x1, y1, radii } => {
            w.u8(0);
            for v in [*x0, *y0, *x1, *y1, radii[0], radii[1], radii[2], radii[3]] {
                w.f32(v);
            }
        }
        G::Ellipse { x0, y0, x1, y1 } => {
            w.u8(1);
            for v in [*x0, *y0, *x1, *y1] {
                w.f32(v);
            }
        }
        G::Triangle { x0, y0, x1, y1 } => {
            w.u8(2);
            for v in [*x0, *y0, *x1, *y1] {
                w.f32(v);
            }
        }
        G::Polygon { x0, y0, x1, y1, sides, star } => {
            w.u8(3);
            for v in [*x0, *y0, *x1, *y1, *star] {
                w.f32(v);
            }
            w.u32(*sides);
        }
        G::Line { a, b, weight, arrow_start, arrow_end } => {
            w.u8(4);
            for v in [a.0, a.1, b.0, b.1, *weight] {
                w.f32(v);
            }
            for ar in [arrow_start, arrow_end] {
                match ar {
                    Some((x, y)) => {
                        w.u8(1);
                        w.f32(*x);
                        w.f32(*y);
                    }
                    None => w.u8(0),
                }
            }
        }
        G::Custom { x0, y0, x1, y1, which } => {
            w.u8(5);
            for v in [*x0, *y0, *x1, *y1] {
                w.f32(v);
            }
            w.u32(*which as u32);
        }
        G::Path(p) => {
            w.u8(6);
            write_path(w, p);
        }
    }
}

/// The kind-4 record: the first component with solid colours (gradients
/// and patterns write a representative colour; the full description is in
/// the "PFLX" shape extras).
fn write_shape(w: &mut W, sh: &super::layers::ShapeLayer) {
    write_geom(w, &sh.geom);
    match sh.fill.color() {
        Some(c) => {
            w.u8(1);
            w.0.extend_from_slice(&c);
        }
        None => w.u8(0),
    }
    match &sh.stroke {
        Some((paint, st)) => {
            w.u8(1);
            w.0.extend_from_slice(&paint.color().unwrap_or([0, 0, 0]));
            w.f32(st.width);
            w.u8(st.join as u8);
            w.u8(st.cap as u8);
            w.f32(st.miter_limit);
            w.u8(st.align as u8);
            w.u32(st.dashes.len() as u32);
            for &d in &st.dashes {
                w.f32(d);
            }
        }
        None => w.u8(0),
    }
    w.u8(sh.rule as u8);
}

/// The layer-model record (version 2) for one layer.
fn layer_model(l: &Layer) -> Result<Vec<u8>> {
    use super::layers::LayerKind as K;
    let mut w = W(Vec::new());
    w.u64(l.id);
    w.u64(l.parent.unwrap_or(0));
    w.u8(l.clip as u8);
    match &l.kind {
        K::Pixel => w.u8(0),
        K::Group { pass_through, collapsed } => {
            w.u8(1);
            w.u8(*pass_through as u8);
            w.u8(*collapsed as u8);
        }
        K::Adjustment(_) => {
            w.u8(2);
            let all = super::filters::catalog::Kind::all();
            match &l.adj_params {
                Some(p) => {
                    w.u32(all.iter().position(|k| *k == p.0).unwrap_or(0) as u32);
                    w.u8(p.2 as u8);
                    w.u32(p.1.len() as u32);
                    for &v in &p.1 {
                        w.f32(v);
                    }
                }
                None => {
                    w.u32(u32::MAX);
                    w.u8(0);
                    w.u32(0);
                }
            }
        }
        K::Artboard(ab) => {
            // 7: artboard (rect, background, collapsed).
            w.u8(7);
            for v in [ab.rect.x0, ab.rect.y0, ab.rect.x1, ab.rect.y1] {
                w.u32(v as u32);
            }
            match ab.bg {
                Some(c) => {
                    w.u8(1);
                    w.0.extend_from_slice(&c);
                }
                None => w.u8(0),
            }
            w.u8(ab.collapsed as u8);
        }
        K::Frame(fr) => {
            // 6: frame (geometry, rotation, transform, optional smart-object content).
            w.u8(6);
            write_geom(&mut w, &fr.shape.geom);
            w.f32(fr.shape.rot);
            fr.shape.xf.iter().for_each(|&v| w.f32(v));
            match &fr.content {
                None => w.u8(0),
                Some(so) => {
                    w.u8(1);
                    w.str(&so.name);
                    for v in so.m {
                        w.f32(v);
                    }
                    w.u32(so.src.w);
                    w.u32(so.src.h);
                    let mut g = Vec::new();
                    write_grid(&mut g, &so.src)?;
                    w.0.extend_from_slice(&g);
                }
            }
        }
        K::Smart(so) => {
            w.u8(3);
            w.str(&so.name);
            for v in so.m {
                w.f32(v);
            }
            w.u32(so.src.w);
            w.u32(so.src.h);
            let mut g = Vec::new();
            write_grid(&mut g, &so.src)?;
            w.0.extend_from_slice(&g);
        }
        K::Shape(sh) => {
            w.u8(4);
            write_shape(&mut w, sh);
        }
        K::Text(t) => {
            w.u8(5);
            w.str(&t.text);
            w.str(&t.style.family);
            w.u32(t.style.weight as u32);
            w.u8(t.style.italic as u8);
            for v in [t.style.size, t.style.leading, t.style.tracking, t.origin.0, t.origin.1, t.warp.bend] {
                w.f32(v);
            }
            w.0.extend_from_slice(&t.style.color);
            w.u8(t.style.align as u8);
            match t.box_size {
                Some((bw, bh)) => {
                    w.u8(1);
                    w.f32(bw);
                    w.f32(bh);
                }
                None => w.u8(0),
            }
            w.u8(t.vertical as u8);
            w.u8(t.warp.style.index());
        }
    }
    match &l.mask {
        None => w.u8(0),
        Some(m) => {
            w.u8(1);
            w.u8(m.enabled as u8);
            let recs: Vec<Vec<u8>> = m
                .grid
                .tiles
                .par_iter()
                .enumerate()
                .map(|(i, t)| {
                    let mut r = (i as u32).to_le_bytes().to_vec();
                    match t {
                        super::selection::MaskTile::Empty => r.push(0),
                        super::selection::MaskTile::Full => r.push(1),
                        super::selection::MaskTile::Partial(px) => {
                            let c = lz4_flex::compress(px);
                            r.push(2);
                            r.extend_from_slice(&(c.len() as u32).to_le_bytes());
                            r.extend_from_slice(&c);
                        }
                    }
                    r
                })
                .collect();
            w.u32(recs.len() as u32);
            for r in recs {
                w.0.extend_from_slice(&r);
            }
        }
    }
    Ok(w.0)
}

/// Document-level data stored after the layers (paths, annotations).
#[derive(Default)]
pub struct DocExtras {
    pub paths: Vec<super::document::PathItem>,
    pub ann: super::annot::Annotations,
    /// Selected layers (multi-selection), saved as "PFLX" tag 4.
    pub selected: Vec<u64>,
}

const EXTRAS: &[u8; 4] = b"PFDX";

fn write_extras(x: &DocExtras) -> Vec<u8> {
    let mut w = W(Vec::new());
    w.u32(x.paths.len() as u32);
    for pi in &x.paths {
        w.str(&pi.name);
        write_path(&mut w, &pi.path);
    }
    let a = &x.ann;
    let rect = |w: &mut W, r: RectI| {
        for v in [r.x0, r.y0, r.x1, r.y1] {
            w.u32(v as u32);
        }
    };
    w.u32(a.samplers.len() as u32);
    for &(x, y) in &a.samplers {
        w.f32(x);
        w.f32(y);
    }
    w.u32(a.notes.len() as u32);
    for n in &a.notes {
        w.f32(n.x);
        w.f32(n.y);
        w.str(&n.author);
        w.str(&n.text);
    }
    w.u32(a.counts.len() as u32);
    for g in &a.counts {
        w.str(&g.name);
        g.color.iter().for_each(|&c| w.u8(c));
        w.f32(g.size);
        w.u8(g.visible as u8);
        w.u32(g.pts.len() as u32);
        for &(x, y) in &g.pts {
            w.f32(x);
            w.f32(y);
        }
    }
    w.u32(a.artboards.len() as u32);
    for ab in &a.artboards {
        w.str(&ab.name);
        rect(&mut w, ab.rect);
    }
    w.u32(a.slices.len() as u32);
    for sl in &a.slices {
        w.str(&sl.name);
        rect(&mut w, sl.rect);
    }
    w.0
}

fn read_extras(r: &mut impl Read) -> Result<DocExtras> {
    use super::annot::{Artboard, CountGroup, Note, Slice};
    let rect = |r: &mut dyn Read| -> Result<RectI> {
        let mut v = [0i32; 4];
        for x in &mut v {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            *x = i32::from_le_bytes(b);
        }
        Ok(RectI::new(v[0], v[1], v[2], v[3]))
    };
    let mut x = DocExtras::default();
    for _ in 0..read_u32(r)? {
        let name = read_str(r)?;
        x.paths.push(super::document::PathItem { name, path: read_path(r)? });
    }
    let a = &mut x.ann;
    for _ in 0..read_u32(r)? {
        a.samplers.push((read_f32(r)?, read_f32(r)?));
    }
    for _ in 0..read_u32(r)? {
        let (nx, ny) = (read_f32(r)?, read_f32(r)?);
        a.notes.push(Note { x: nx, y: ny, author: read_str(r)?, text: read_str(r)? });
    }
    for _ in 0..read_u32(r)? {
        let name = read_str(r)?;
        let color = [read_u8(r)?, read_u8(r)?, read_u8(r)?];
        let mut g = CountGroup::new(name, color);
        g.size = read_f32(r)?;
        g.visible = read_u8(r)? != 0;
        for _ in 0..read_u32(r)? {
            g.pts.push((read_f32(r)?, read_f32(r)?));
        }
        a.counts.push(g);
    }
    for _ in 0..read_u32(r)? {
        let name = read_str(r)?;
        a.artboards.push(Artboard { name, rect: rect(r)? });
    }
    for _ in 0..read_u32(r)? {
        let name = read_str(r)?;
        a.slices.push(Slice::new(name, rect(r)?));
    }
    Ok(x)
}

const LAYER_EXT: &[u8; 4] = b"PFLX";
const EXT_TEXT: u8 = 1;
const EXT_SHAPE: u8 = 2;
/// Empty-payload flags: the locked Background, and a selected layer.
const EXT_BACKGROUND: u8 = 3;
const EXT_SELECTED: u8 = 4;
/// Hidden pixels: x i32, y i32, w u32, h u32, the grid's tile records, then
/// u8 has-mask and the mask's (opaque gray) tile records over the same rect.
const EXT_HIDDEN: u8 = 5;
/// Layer Style: `styles::LayerStyle::to_bytes`.
const EXT_STYLE: u8 = 6;
/// Fill opacity: f32.
const EXT_FILL: u8 = 7;
const EXT_ADVANCED_BLEND: u8 = 8;
/// Trailing chunk after "PFLX": slice options (url, target, alt, source layer)
/// by slice index.
const SLICE_EXT: &[u8; 4] = b"PFSX";
/// Measurement Log + scale, Info modes of the color samplers, Count label
/// sizes, Vanishing Point planes.
const MEASURE_EXT: &[u8; 4] = b"PFMX";

fn write_measure_ext(a: &super::annot::Annotations) -> Vec<u8> {
    let mut out = Vec::new();
    super::measure::encode(&a.measure, &mut out);
    out.extend_from_slice(&(a.sampler_modes.len() as u32).to_le_bytes());
    out.extend(a.sampler_modes.iter().map(|m| super::color::ColorMode::ALL.iter().position(|x| x == m).unwrap_or(0) as u8));
    out.extend_from_slice(&(a.counts.len() as u32).to_le_bytes());
    for g in &a.counts {
        out.extend_from_slice(&g.label_size.to_le_bytes());
    }
    out.extend_from_slice(&(a.vp_planes.len() as u32).to_le_bytes());
    for p in &a.vp_planes {
        for c in p.c {
            out.extend_from_slice(&c.0.to_le_bytes());
            out.extend_from_slice(&c.1.to_le_bytes());
        }
    }
    let mut w = (out.len() as u32).to_le_bytes().to_vec();
    w.extend(out);
    w
}

fn read_measure_ext(r: &mut impl Read, a: &mut super::annot::Annotations) -> Result<()> {
    let n = read_u32(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    let mut s = b.as_slice();
    if let Some(m) = super::measure::decode(&mut s) {
        a.measure = m;
    }
    let mut rd = |n: usize| -> Option<Vec<u8>> {
        let (h, t) = (s.get(..n)?.to_vec(), s.get(n..)?);
        s = t;
        Some(h)
    };
    if let Some(k) = rd(4).map(|v| u32::from_le_bytes(v.try_into().unwrap()) as usize) {
        if let Some(modes) = rd(k) {
            a.sampler_modes = modes.iter().map(|&i| super::color::ColorMode::ALL.get(i as usize).copied().unwrap_or_default()).collect();
        }
    }
    if let Some(k) = rd(4).map(|v| u32::from_le_bytes(v.try_into().unwrap()) as usize) {
        for g in a.counts.iter_mut().take(k) {
            if let Some(v) = rd(4) {
                g.label_size = f32::from_le_bytes(v.try_into().unwrap());
            }
        }
    }
    if let Some(k) = rd(4).map(|v| u32::from_le_bytes(v.try_into().unwrap()) as usize) {
        for _ in 0..k {
            let Some(v) = rd(32) else { break };
            let f = |i: usize| f32::from_le_bytes(v[i * 4..i * 4 + 4].try_into().unwrap());
            a.vp_planes.push(super::vanish::Plane { c: [(f(0), f(1)), (f(2), f(3)), (f(4), f(5)), (f(6), f(7))] });
        }
    }
    Ok(())
}

fn write_slice_ext(slices: &[super::annot::Slice]) -> Vec<u8> {
    let mut w = W(Vec::new());
    w.u32(slices.len() as u32);
    for s in slices {
        w.str(&s.url);
        w.str(&s.target);
        w.str(&s.alt);
        w.u64(s.layer.unwrap_or(0));
    }
    w.0
}

fn read_slice_ext(r: &mut impl Read, slices: &mut [super::annot::Slice]) -> Result<()> {
    for i in 0..read_u32(r)? as usize {
        let (url, target, alt) = (read_str(r)?, read_str(r)?, read_str(r)?);
        let layer = read_u64(r)?;
        if let Some(s) = slices.get_mut(i) {
            (s.url, s.target, s.alt) = (url, target, alt);
            s.layer = (layer != 0).then_some(layer);
        }
    }
    Ok(())
}

fn write_paint(w: &mut W, p: &super::layers::Paint) {
    use super::layers::Paint;
    match p {
        Paint::None => w.u8(0),
        Paint::Solid(c) => {
            w.u8(1);
            w.0.extend_from_slice(c);
        }
        Paint::Gradient(g) => {
            w.u8(2);
            w.u8(g.kind as u8);
            w.f32(g.angle);
            w.f32(g.scale);
            w.u8(g.reverse as u8);
            w.str(&g.grad.name);
            w.u32(g.grad.stops.len() as u32);
            for st in &g.grad.stops {
                w.f32(st.pos);
                w.0.extend_from_slice(&st.color);
                w.f32(st.mid);
            }
            w.u32(g.grad.alpha.len() as u32);
            for a in &g.grad.alpha {
                w.f32(a.pos);
                w.f32(a.alpha);
            }
        }
        Paint::Pattern(pp) => {
            w.u8(3);
            w.u32(pp.which as u32);
            w.f32(pp.scale);
            w.f32(pp.angle);
        }
    }
}

fn read_paint(r: &mut impl Read) -> Result<super::layers::Paint> {
    use super::gradient::{AlphaStop, ColorStop, GradKind, Gradient};
    use super::layers::{GradPaint, Paint, PatPaint};
    Ok(match read_u8(r)? {
        0 => Paint::None,
        1 => Paint::Solid([read_u8(r)?, read_u8(r)?, read_u8(r)?]),
        2 => {
            let kind = GradKind::ALL[(read_u8(r)? as usize).min(4)];
            let (angle, scale) = (read_f32(r)?, read_f32(r)?);
            let reverse = read_u8(r)? != 0;
            let name = read_str(r)?;
            let mut stops = Vec::new();
            for _ in 0..read_u32(r)? {
                let pos = read_f32(r)?;
                let color = [read_u8(r)?, read_u8(r)?, read_u8(r)?];
                stops.push(ColorStop { pos, color, mid: read_f32(r)? });
            }
            let mut alpha = Vec::new();
            for _ in 0..read_u32(r)? {
                alpha.push(AlphaStop { pos: read_f32(r)?, alpha: read_f32(r)? });
            }
            Paint::Gradient(GradPaint { grad: Gradient { name, stops, alpha }, kind, angle, scale, reverse })
        }
        3 => Paint::Pattern(PatPaint { which: read_u32(r)? as usize, scale: read_f32(r)?, angle: read_f32(r)? }),
        k => bail!("bad paint {k}"),
    })
}

fn write_shape_ext(sh: &super::layers::ShapeLayer) -> Vec<u8> {
    let mut w = W(Vec::new());
    w.u8(1);
    w.f32(sh.rot);
    sh.xf.iter().for_each(|&v| w.f32(v));
    w.u32(sh.parts.len() as u32);
    for pt in &sh.parts {
        w.u8(pt.op as u8);
        w.f32(pt.rot);
        pt.xf.iter().for_each(|&v| w.f32(v));
        write_geom(&mut w, &pt.geom);
    }
    write_paint(&mut w, &sh.fill);
    match &sh.stroke {
        Some((paint, st)) => {
            w.u8(1);
            write_paint(&mut w, paint);
            w.f32(st.dash_offset);
        }
        None => w.u8(0),
    }
    w.0
}

fn read_shape_ext(r: &mut impl Read, sh: &mut super::layers::ShapeLayer) -> Result<()> {
    use super::layers::{PathOp, ShapePart};
    let _version = read_u8(r)?;
    sh.rot = read_f32(r)?;
    for v in &mut sh.xf {
        *v = read_f32(r)?;
    }
    sh.parts.clear();
    for _ in 0..read_u32(r)? {
        let op = PathOp::ALL[(read_u8(r)? as usize).min(3)];
        let rot = read_f32(r)?;
        let mut xf = [0f32; 6];
        for v in &mut xf {
            *v = read_f32(r)?;
        }
        sh.parts.push(ShapePart { geom: read_geom(r)?, rot, xf, op });
    }
    sh.fill = read_paint(r)?;
    if read_u8(r)? != 0 {
        let paint = read_paint(r)?;
        let off = read_f32(r)?;
        if let Some((p, st)) = &mut sh.stroke {
            *p = paint;
            st.dash_offset = off;
        }
    }
    Ok(())
}

fn write_char_style(w: &mut W, s: &super::text::CharStyle) {
    w.str(&s.family);
    w.u32(s.weight as u32);
    w.u8(s.italic as u8);
    for v in [s.size, s.leading, s.tracking, s.baseline_shift] {
        w.f32(v);
    }
    s.color.iter().for_each(|&c| w.u8(c));
    w.u8(s.underline as u8 | (s.strike as u8) << 1 | (s.faux_bold as u8) << 2 | (s.faux_italic as u8) << 3);
    w.u8(s.caps as u8);
    w.u8(s.position as u8);
    let f = &s.features;
    w.u8(f.liga as u8 | (f.dlig as u8) << 1 | (f.onum as u8) << 2 | (f.swsh as u8) << 3 | (f.frac as u8) << 4);
    w.u32(f.ss);
}

fn read_char_style(r: &mut impl Read) -> Result<super::text::CharStyle> {
    use super::text::{Caps, CharStyle, OtFeatures, Position};
    let family = read_str(r)?;
    let weight = read_u32(r)? as u16;
    let italic = read_u8(r)? != 0;
    let mut f = [0f32; 4];
    for v in &mut f {
        *v = read_f32(r)?;
    }
    let color = [read_u8(r)?, read_u8(r)?, read_u8(r)?];
    let flags = read_u8(r)?;
    let caps = [Caps::Normal, Caps::All, Caps::Small][(read_u8(r)? as usize).min(2)];
    let position = [Position::Normal, Position::Super, Position::Sub][(read_u8(r)? as usize).min(2)];
    let of = read_u8(r)?;
    let ss = read_u32(r)?;
    Ok(CharStyle {
        family,
        weight,
        italic,
        size: f[0],
        leading: f[1],
        tracking: f[2],
        baseline_shift: f[3],
        color,
        underline: flags & 1 != 0,
        strike: flags & 2 != 0,
        faux_bold: flags & 4 != 0,
        faux_italic: flags & 8 != 0,
        caps,
        position,
        features: OtFeatures { liga: of & 1 != 0, dlig: of & 2 != 0, onum: of & 4 != 0, swsh: of & 8 != 0, frac: of & 16 != 0, ss },
    })
}

fn write_text_ext(t: &super::text::TextLayer) -> Vec<u8> {
    let mut w = W(Vec::new());
    // Version 2 adds the composer per paragraph.
    w.u8(2);
    w.u32(t.runs.len() as u32);
    for run in &t.runs {
        w.u32(run.start as u32);
        w.u32(run.end as u32);
        write_char_style(&mut w, &run.style);
    }
    w.u32(t.paras.len() as u32);
    for p in &t.paras {
        w.u8(p.align as u8);
        w.u8(p.last as u8);
        for v in [p.indent_left, p.indent_right, p.indent_first, p.space_before, p.space_after] {
            w.f32(v);
        }
        w.u8(p.hyphenate as u8);
        w.u8(p.composer as u8);
    }
    w.u8(t.upright_roman as u8);
    w.u32(t.tcy.len() as u32);
    for &(a, b) in &t.tcy {
        w.u32(a as u32);
        w.u32(b as u32);
    }
    w.u8(t.warp.style.index());
    for v in [t.warp.bend, t.warp.h_distort, t.warp.v_distort] {
        w.f32(v);
    }
    w.u8(t.warp.vertical as u8);
    match &t.path {
        None => w.u8(0),
        Some(tp) => {
            w.u8(1);
            write_path(&mut w, &tp.path);
            w.f32(tp.start);
            w.f32(tp.end);
            w.u8(tp.flip as u8 | (tp.area as u8) << 1);
        }
    }
    w.0
}

fn read_text_ext(r: &mut impl Read, t: &mut super::text::TextLayer) -> Result<()> {
    use super::text::{Composer, LastLine, ParaStyle, StyleRun, TextAlign, TextPath, WarpStyle};
    let version = read_u8(r)?;
    let n = t.text.len();
    let mut runs = Vec::new();
    for _ in 0..read_u32(r)? {
        let (a, b) = (read_u32(r)? as usize, read_u32(r)? as usize);
        let style = read_char_style(r)?;
        if a < b && b <= n && t.text.is_char_boundary(a) && t.text.is_char_boundary(b) {
            runs.push(StyleRun { start: a, end: b, style });
        }
    }
    // Only accept runs that tile the text exactly.
    let tiles = runs.first().is_none_or(|r| r.start == 0) && runs.windows(2).all(|w| w[0].end == w[1].start) && runs.last().map_or(n == 0, |r| r.end == n);
    if tiles {
        t.runs = runs;
    }
    let mut paras = Vec::new();
    for _ in 0..read_u32(r)? {
        let align = TextAlign::ALL[(read_u8(r)? as usize).min(3)];
        let last = LastLine::ALL[(read_u8(r)? as usize).min(3)];
        let mut f = [0f32; 5];
        for v in &mut f {
            *v = read_f32(r)?;
        }
        let hyphenate = read_u8(r)? != 0;
        let composer = if version >= 2 { Composer::ALL[(read_u8(r)? as usize).min(1)] } else { Composer::Every };
        paras.push(ParaStyle { align, last, indent_left: f[0], indent_right: f[1], indent_first: f[2], space_before: f[3], space_after: f[4], hyphenate, composer });
    }
    if paras.len() == t.paras.len() {
        t.paras = paras;
    }
    t.upright_roman = read_u8(r)? != 0;
    t.tcy.clear();
    for _ in 0..read_u32(r)? {
        let (a, b) = (read_u32(r)? as usize, read_u32(r)? as usize);
        if a < b && b <= n {
            t.tcy.push((a, b));
        }
    }
    t.warp.style = WarpStyle::from_index(read_u8(r)?);
    t.warp.bend = read_f32(r)?;
    t.warp.h_distort = read_f32(r)?;
    t.warp.v_distort = read_f32(r)?;
    t.warp.vertical = read_u8(r)? != 0;
    if read_u8(r)? != 0 {
        let path = read_path(r)?;
        let (start, end) = (read_f32(r)?, read_f32(r)?);
        let fl = read_u8(r)?;
        t.path = Some(Arc::new(TextPath { path, start, end, flip: fl & 1 != 0, area: fl & 2 != 0 }));
    }
    t.relayout();
    Ok(())
}

fn write_layer_ext(layers: &[Layer], selected: &[u64]) -> Vec<u8> {
    let mut recs: Vec<(u64, u8, Vec<u8>)> = Vec::new();
    for l in layers {
        if l.background {
            recs.push((l.id, EXT_BACKGROUND, vec![]));
        }
        if selected.contains(&l.id) {
            recs.push((l.id, EXT_SELECTED, vec![]));
        }
        if let Some(hd) = &l.hidden {
            let mut w = W(Vec::new());
            w.u32(hd.x as u32);
            w.u32(hd.y as u32);
            w.u32(hd.grid.w);
            w.u32(hd.grid.h);
            let mut ok = write_grid(&mut w.0, &hd.grid).is_ok();
            w.u8(hd.mask.is_some() as u8);
            if let Some(m) = &hd.mask {
                ok &= write_grid(&mut w.0, m).is_ok();
            }
            if ok {
                recs.push((l.id, EXT_HIDDEN, w.0));
            }
        }
        if let Some(st) = &l.style {
            recs.push((l.id, EXT_STYLE, st.to_bytes()));
        }
        if l.fill != 1.0 {
            recs.push((l.id, EXT_FILL, l.fill.to_le_bytes().to_vec()));
        }
        if let Some(settings)=&l.advanced_blend {
            recs.push((l.id,EXT_ADVANCED_BLEND,settings.to_bytes()));
        }
        match &l.kind {
            super::layers::LayerKind::Text(t) => recs.push((l.id, EXT_TEXT, write_text_ext(t))),
            super::layers::LayerKind::Shape(sh) => recs.push((l.id, EXT_SHAPE, write_shape_ext(sh))),
            _ => {}
        }
    }
    let mut w = W(Vec::new());
    w.u32(recs.len() as u32);
    for (id, tag, b) in recs {
        w.u64(id);
        w.u8(tag);
        w.u32(b.len() as u32);
        w.0.extend_from_slice(&b);
    }
    w.0
}

fn read_layer_ext(r: &mut impl Read, layers: &mut [NativeLayer], selected: &mut Vec<u64>) -> Result<()> {
    for _ in 0..read_u32(r)? {
        let id = read_u64(r)?;
        let tag = read_u8(r)?;
        let len = read_u32(r)? as usize;
        let mut b = vec![0u8; len];
        r.read_exact(&mut b)?;
        let Some(nl) = layers.iter_mut().find(|l| l.id == id) else { continue };
        if tag == EXT_BACKGROUND {
            nl.background = true;
        }
        if tag == EXT_SELECTED {
            selected.push(id);
        }
        if tag == EXT_HIDDEN {
            let r = &mut b.as_slice();
            let (x, y) = (read_u32(r)? as i32, read_u32(r)? as i32);
            let (hw, hh) = (read_u32(r)?, read_u32(r)?);
            let grid = read_grid(r, hw, hh)?;
            let mask = if read_u8(r)? != 0 { Some(read_grid(r, hw, hh)?) } else { None };
            nl.hidden = Some(Arc::new(super::crop::Hidden { grid, x, y, mask }));
        }
        if tag == EXT_STYLE {
            nl.style = super::styles::LayerStyle::from_bytes(&b).map(Arc::new);
        }
        if tag == EXT_FILL && b.len() == 4 {
            let f = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            nl.fill = if f.is_finite() { f.clamp(0.0, 1.0) } else { 1.0 };
        }
        if tag == EXT_ADVANCED_BLEND {
            let settings=super::advanced_blend::AdvancedBlend::from_bytes(&b)?;
            anyhow::ensure!(settings.is_identity() || !matches!(nl.kind, super::layers::LayerKind::Adjustment(_)), "Advanced blending for adjustment layers is not implemented");
            anyhow::ensure!(settings.is_identity() || !matches!(nl.kind, super::layers::LayerKind::Group{pass_through:true,..}), "Advanced blending for pass-through groups needs a backdrop mapping");
            nl.advanced_blend=Some(Arc::new(settings));
        }
        match (tag, &mut nl.kind) {
            (EXT_TEXT, super::layers::LayerKind::Text(t)) => read_text_ext(&mut b.as_slice(), Arc::make_mut(t))?,
            (EXT_SHAPE, super::layers::LayerKind::Shape(sh)) => read_shape_ext(&mut b.as_slice(), Arc::make_mut(sh))?,
            _ => {}
        }
    }
    Ok(())
}

pub fn save_native(doc_layers: &[Layer], w: u32, h: u32, active: usize, extras: &DocExtras, path: &Path) -> Result<()> {
    // Validate before touching an existing destination.
    for layer in doc_layers {
        if let Some(settings) = &layer.advanced_blend {
            settings.validate()?;
            anyhow::ensure!(settings.is_identity() || !matches!(layer.kind, super::layers::LayerKind::Group{pass_through:true,..}), "Advanced blending for pass-through groups needs a backdrop mapping");
            anyhow::ensure!(settings.is_identity() || !matches!(layer.kind, super::layers::LayerKind::Adjustment(_)), "Advanced blending for adjustment layers is not implemented");
        }
    }
    let mut f = BufWriter::with_capacity(1 << 22, File::create(path)?);
    f.write_all(MAGIC)?;
    for v in [w, h, doc_layers.len() as u32, active as u32] {
        f.write_all(&v.to_le_bytes())?;
    }
    for l in doc_layers {
        f.write_all(&(l.name.len() as u32).to_le_bytes())?;
        f.write_all(l.name.as_bytes())?;
        f.write_all(&[l.visible as u8])?;
        f.write_all(&l.opacity.to_le_bytes())?;
        f.write_all(&[l.blend as u8])?;
        write_grid(&mut f, &l.grid)?;
        let model = layer_model(l)?;
        f.write_all(&model)?;
    }
    // Optional trailing chunk: readers that predate it stop after the layers.
    f.write_all(EXTRAS)?;
    f.write_all(&write_extras(extras))?;
    f.write_all(LAYER_EXT)?;
    f.write_all(&write_layer_ext(doc_layers, &extras.selected))?;
    f.write_all(SLICE_EXT)?;
    f.write_all(&write_slice_ext(&extras.ann.slices))?;
    f.write_all(MEASURE_EXT)?;
    f.write_all(&write_measure_ext(&extras.ann))?;
    f.flush()?;
    Ok(())
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u8(r: &mut impl Read) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_f32(r: &mut impl Read) -> Result<f32> {
    Ok(f32::from_bits(read_u32(r)?))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_str(r: &mut impl Read) -> Result<String> {
    let n = read_u32(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn read_p(r: &mut impl Read) -> Result<super::vector::P> {
    Ok(super::vector::p(read_f32(r)?, read_f32(r)?))
}

/// Reads tile records into a `w x h` grid.
fn read_grid(r: &mut impl Read, w: u32, h: u32) -> Result<TileGrid> {
    let count = read_u32(r)?;
    enum Rec {
        Uniform(u32),
        Lz4(Vec<u8>),
    }
    let mut recs = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let idx = read_u32(r)? as usize;
        match read_u8(r)? {
            1 => recs.push((idx, Rec::Uniform(read_u32(r)?))),
            2 => {
                let l = read_u32(r)? as usize;
                let mut b = vec![0u8; l];
                r.read_exact(&mut b)?;
                recs.push((idx, Rec::Lz4(b)));
            }
            k => bail!("bad tile kind {k}"),
        }
    }
    let mut grid = TileGrid::new(w, h);
    let mut shared: std::collections::HashMap<u32, Arc<Tile>> = Default::default();
    let decoded: Vec<(usize, Result<Arc<Tile>>)> = recs
        .par_iter()
        .filter_map(|(i, rec)| match rec {
            Rec::Lz4(b) => Some((
                *i,
                lz4_flex::decompress(b, TS * TS * 4)
                    .map_err(anyhow::Error::from)
                    .map(|bytes| Arc::new(Tile::from_px(bytemuck::cast_slice::<u8, u32>(&bytes).to_vec().into_boxed_slice()))),
            )),
            Rec::Uniform(_) => None,
        })
        .collect();
    for (i, rec) in &recs {
        if let Rec::Uniform(c) = rec {
            let t = shared.entry(*c).or_insert_with(|| Arc::new(Tile::solid(*c))).clone();
            if *i < grid.tiles.len() {
                grid.tiles[*i] = Some(t);
            }
        }
    }
    for (i, t) in decoded {
        if i < grid.tiles.len() {
            grid.tiles[i] = Some(t?);
        }
    }
    Ok(grid)
}

fn read_path(r: &mut impl Read) -> Result<super::vector::Path> {
    let n = read_u32(r)?;
    let mut subs = Vec::new();
    for _ in 0..n {
        let closed = read_u8(r)? != 0;
        let k = read_u32(r)?;
        let mut anchors = Vec::new();
        for _ in 0..k {
            anchors.push(super::vector::Anchor { p: read_p(r)?, h_in: read_p(r)?, h_out: read_p(r)?, smooth: read_u8(r)? != 0 });
        }
        subs.push(super::vector::SubPath { anchors, closed });
    }
    Ok(super::vector::Path { subs })
}

fn read_geom(r: &mut impl Read) -> Result<super::layers::ShapeGeom> {
    use super::layers::ShapeGeom as G;
    let f = |r: &mut dyn Read| -> Result<f32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(f32::from_le_bytes(b))
    };
    Ok(match read_u8(r)? {
        0 => G::Rect { x0: f(r)?, y0: f(r)?, x1: f(r)?, y1: f(r)?, radii: [f(r)?, f(r)?, f(r)?, f(r)?] },
        1 => G::Ellipse { x0: f(r)?, y0: f(r)?, x1: f(r)?, y1: f(r)? },
        2 => G::Triangle { x0: f(r)?, y0: f(r)?, x1: f(r)?, y1: f(r)? },
        3 => {
            let (x0, y0, x1, y1, star) = (f(r)?, f(r)?, f(r)?, f(r)?, f(r)?);
            G::Polygon { x0, y0, x1, y1, star, sides: read_u32(r)? }
        }
        4 => {
            let (ax, ay, bx, by, weight) = (f(r)?, f(r)?, f(r)?, f(r)?, f(r)?);
            let mut ar = || -> Result<Option<(f32, f32)>> { Ok(if read_u8(r)? != 0 { Some((f(r)?, f(r)?)) } else { None }) };
            let (s, e) = (ar()?, ar()?);
            G::Line { a: (ax, ay), b: (bx, by), weight, arrow_start: s, arrow_end: e }
        }
        5 => {
            let (x0, y0, x1, y1) = (f(r)?, f(r)?, f(r)?, f(r)?);
            G::Custom { x0, y0, x1, y1, which: read_u32(r)? as usize }
        }
        6 => G::Path(read_path(r)?),
        k => bail!("bad shape geometry {k}"),
    })
}

fn read_shape(r: &mut impl Read) -> Result<super::layers::ShapeLayer> {
    use super::layers::Paint;
    use super::vector::{Align, Cap, FillRule, Join, StrokeStyle};
    let f = |r: &mut dyn Read| -> Result<f32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(f32::from_le_bytes(b))
    };
    let geom = read_geom(r)?;
    let rgb = |r: &mut dyn Read| -> Result<[u8; 3]> {
        let mut b = [0u8; 3];
        r.read_exact(&mut b)?;
        Ok(b)
    };
    let fill = if read_u8(r)? != 0 { Some(rgb(r)?) } else { None };
    let stroke = if read_u8(r)? != 0 {
        let c = rgb(r)?;
        let width = f(r)?;
        let join = match read_u8(r)? {
            1 => Join::Round,
            2 => Join::Bevel,
            _ => Join::Miter,
        };
        let cap = match read_u8(r)? {
            1 => Cap::Round,
            2 => Cap::Square,
            _ => Cap::Butt,
        };
        let miter_limit = f(r)?;
        let align = match read_u8(r)? {
            1 => Align::Inside,
            2 => Align::Outside,
            _ => Align::Center,
        };
        let n = read_u32(r)?;
        let mut dashes = Vec::new();
        for _ in 0..n {
            dashes.push(f(r)?);
        }
        Some((c, StrokeStyle { width, join, cap, miter_limit, align, dashes, dash_offset: 0.0 }))
    } else {
        None
    };
    let rule = if read_u8(r)? == 1 { FillRule::EvenOdd } else { FillRule::NonZero };
    let mut sh = super::layers::ShapeLayer::new(geom, fill.map_or(Paint::None, Paint::Solid), stroke.map(|(c, st)| (Paint::Solid(c), st)));
    sh.rule = rule;
    Ok(sh)
}

pub fn load_native(path: &Path) -> Result<(u32, u32, usize, Vec<NativeLayer>, DocExtras)> {
    use super::layers::{LayerKind as K, SmartObject};
    let mut r = BufReader::with_capacity(1 << 22, File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let v2 = &magic == MAGIC;
    if !v2 && &magic != MAGIC_V1 {
        bail!("not a PixelForge document");
    }
    let w = read_u32(&mut r)?;
    let h = read_u32(&mut r)?;
    let n = read_u32(&mut r)?;
    let active = read_u32(&mut r)? as usize;
    let mut layers = Vec::with_capacity(n as usize);
    for li in 0..n {
        let len = read_u32(&mut r)? as usize;
        let mut name = vec![0u8; len];
        r.read_exact(&mut name)?;
        let visible = read_u8(&mut r)? != 0;
        let opacity = f32::from_bits(read_u32(&mut r)?);
        let blend = BlendMode::from_u8(read_u8(&mut r)?);
        let grid = read_grid(&mut r, w, h)?;
        let mut nl = NativeLayer {
            name: String::from_utf8_lossy(&name).into_owned(),
            visible,
            opacity,
            blend,
            grid,
            id: li as u64 + 1,
            parent: None,
            clip: false,
            kind: K::Pixel,
            adj: None,
            mask: None,
            background: false,
            hidden: None,
            style: None,
            fill: 1.0,
            advanced_blend: None,
        };
        if v2 {
            nl.id = read_u64(&mut r)?;
            nl.parent = Some(read_u64(&mut r)?).filter(|&p| p != 0);
            nl.clip = read_u8(&mut r)? != 0;
            nl.kind = match read_u8(&mut r)? {
                0 => K::Pixel,
                1 => K::Group { pass_through: read_u8(&mut r)? != 0, collapsed: read_u8(&mut r)? != 0 },
                2 => {
                    let ki = read_u32(&mut r)?;
                    let flag = read_u8(&mut r)? != 0;
                    let cnt = read_u32(&mut r)?;
                    let mut vals = Vec::new();
                    for _ in 0..cnt {
                        vals.push(read_f32(&mut r)?);
                    }
                    let all = super::filters::catalog::Kind::all();
                    let kind = all.get(ki as usize).copied().unwrap_or(super::filters::catalog::Kind::Invert);
                    let ctx = super::filters::catalog::BuildCtx {
                        fg: [0, 0, 0],
                        bg: [255, 255, 255],
                        area: RectI::new(0, 0, w as i32, h as i32),
                        stats: None,
                        source: None,
                        disp_map: None,
                        layer_mask: None,
                        doc_w: w,
                        doc_h: h,
                    };
                    let f = if (ki as usize) < all.len() { kind.build(&vals, flag, &ctx) } else { super::filters::Filter::Invert };
                    nl.adj = Some((kind, vals, flag));
                    K::Adjustment(f)
                }
                3 => {
                    let name = read_str(&mut r)?;
                    let mut m = [0f32; 9];
                    for v in &mut m {
                        *v = read_f32(&mut r)?;
                    }
                    let (sw, sh) = (read_u32(&mut r)?, read_u32(&mut r)?);
                    let src = read_grid(&mut r, sw, sh)?;
                    K::Smart(Arc::new(SmartObject::new(name, src, m)))
                }
                7 => {
                    let mut v = [0i32; 4];
                    for x in &mut v {
                        *x = read_u32(&mut r)? as i32;
                    }
                    let bg = if read_u8(&mut r)? != 0 { Some([read_u8(&mut r)?, read_u8(&mut r)?, read_u8(&mut r)?]) } else { None };
                    let collapsed = read_u8(&mut r)? != 0;
                    K::Artboard(Arc::new(super::artboard::ArtboardLayer { rect: RectI::new(v[0], v[1], v[2], v[3]), bg, collapsed }))
                }
                6 => {
                    let geom = read_geom(&mut r)?;
                    let mut fr = super::frame::FrameLayer::new(geom);
                    fr.shape.rot = read_f32(&mut r)?;
                    for v in &mut fr.shape.xf {
                        *v = read_f32(&mut r)?;
                    }
                    if read_u8(&mut r)? != 0 {
                        let name = read_str(&mut r)?;
                        let mut m = [0f32; 9];
                        for v in &mut m {
                            *v = read_f32(&mut r)?;
                        }
                        let (sw, sh) = (read_u32(&mut r)?, read_u32(&mut r)?);
                        let src = read_grid(&mut r, sw, sh)?;
                        fr.content = Some(Arc::new(SmartObject::new(name, src, m)));
                    }
                    K::Frame(Arc::new(fr))
                }
                4 => K::Shape(Arc::new(read_shape(&mut r)?)),
                5 => {
                    use super::text::{TextAlign, TextLayer, TextStyle, WarpStyle};
                    let text = read_str(&mut r)?;
                    let family = read_str(&mut r)?;
                    let weight = read_u32(&mut r)? as u16;
                    let italic = read_u8(&mut r)? != 0;
                    let mut f = [0f32; 6];
                    for v in &mut f {
                        *v = read_f32(&mut r)?;
                    }
                    let mut color = [0u8; 3];
                    r.read_exact(&mut color)?;
                    let align = TextAlign::ALL[(read_u8(&mut r)? as usize).min(3)];
                    let box_size = if read_u8(&mut r)? != 0 { Some((read_f32(&mut r)?, read_f32(&mut r)?)) } else { None };
                    let vertical = read_u8(&mut r)? != 0;
                    let warp = WarpStyle::from_index(read_u8(&mut r)?);
                    let style = TextStyle { family, weight, italic, size: f[0], leading: f[1], tracking: f[2], color, align };
                    let mut t = TextLayer::new(text, style, (f[3], f[4]), box_size, vertical);
                    t.warp.style = warp;
                    t.warp.bend = f[5];
                    t.relayout();
                    K::Text(Arc::new(t))
                }
                k => bail!("bad layer kind {k}"),
            };
            if read_u8(&mut r)? != 0 {
                let enabled = read_u8(&mut r)? != 0;
                let cnt = read_u32(&mut r)?;
                let mut m = super::selection::MaskGrid::new(w, h);
                for _ in 0..cnt {
                    let i = read_u32(&mut r)? as usize;
                    let t = match read_u8(&mut r)? {
                        0 => super::selection::MaskTile::Empty,
                        1 => super::selection::MaskTile::Full,
                        _ => {
                            let l = read_u32(&mut r)? as usize;
                            let mut b = vec![0u8; l];
                            r.read_exact(&mut b)?;
                            super::selection::MaskTile::Partial(Arc::new(crate::engine::memory::Bytes::new(lz4_flex::decompress(&b, TS * TS)?)))
                        }
                    };
                    if i < m.tiles.len() {
                        m.tiles[i] = t;
                    }
                }
                nl.mask = Some((m, enabled));
            }
        }
        layers.push(nl);
    }
    let mut tag = [0u8; 4];
    let mut extras = if v2 && r.read_exact(&mut tag).is_ok() && &tag == EXTRAS { read_extras(&mut r)? } else { DocExtras::default() };
    if v2 && r.read_exact(&mut tag).is_ok() && &tag == LAYER_EXT {
        read_layer_ext(&mut r, &mut layers, &mut extras.selected)?;
        if r.read_exact(&mut tag).is_ok() && &tag == SLICE_EXT {
            read_slice_ext(&mut r, &mut extras.ann.slices)?;
            if r.read_exact(&mut tag).is_ok() && &tag == MEASURE_EXT {
                read_measure_ext(&mut r, &mut extras.ann)?;
            }
        }
    }
    Ok((w, h, active, layers, extras))
}

pub fn native_to_document(w: u32, h: u32, active: usize, layers: Vec<NativeLayer>, extras: DocExtras) -> Document {
    let layers: Vec<Layer> = layers
        .into_iter()
        .enumerate()
        .map(|(_, nl)| {
            let mut l = Layer::new(nl.id, nl.name, nl.grid);
            l.visible = nl.visible;
            l.opacity = nl.opacity;
            l.blend = nl.blend;
            l.parent = nl.parent;
            l.clip = nl.clip;
            l.kind = nl.kind;
            l.adj_params = nl.adj.map(std::sync::Arc::new);
            l.mask = nl.mask.map(|(m, enabled)| super::layers::LayerMask { enabled, ..super::layers::LayerMask::new(m) });
            l.background = nl.background;
            l.hidden = nl.hidden;
            l.style = nl.style;
            l.fill = nl.fill;
            l.advanced_blend = nl.advanced_blend;
            l
        })
        .collect();
    let mut layers = layers;
    let mut ann = extras.ann;
    // Older files kept artboards as annotations: they become artboard layers.
    let mut next = layers.iter().map(|l| l.id).max().unwrap_or(0) + 1;
    for ab in std::mem::take(&mut ann.artboards) {
        let mut l = Layer::new(next, ab.name, TileGrid::new(w, h));
        l.kind = super::layers::LayerKind::Artboard(Arc::new(super::artboard::ArtboardLayer { rect: ab.rect, bg: None, collapsed: false }));
        layers.push(l);
        next += 1;
    }
    let mut doc = Document::from_layers(w, h, layers, active);
    doc.paths = extras.paths;
    doc.ann = ann;
    doc.selected = extras.selected;
    doc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_advanced_blending_roundtrip_and_missing_extension_default() {
        let settings=Arc::new(super::super::advanced_blend::AdvancedBlend{
            ranges:vec![super::super::advanced_blend::Range{source:[10,20,230,240],destination:[0,0,255,255]}],
            restricted_channels:vec![2,0,2],
        });
        let plain=Layer::new(1,"Plain",TileGrid::new(1,1));
        let mut advanced=Layer::new(2,"Advanced",TileGrid::new(1,1));
        advanced.advanced_blend=Some(settings.clone());
        let path=std::env::temp_dir().join(format!("pforge-advanced-blend-{}.pforge",std::process::id()));
        save_native(&[plain,advanced],1,1,1,&DocExtras::default(),&path).unwrap();
        let (w,h,active,layers,extras)=load_native(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let document=native_to_document(w,h,active,layers,extras);
        assert!(document.layers[0].advanced_blend.is_none());
        assert_eq!(document.layers[1].advanced_blend.as_deref(),Some(settings.as_ref()));
    }

    #[test]
    fn native_roundtrip() {
        let g = super::super::ops::procedural(700, 400);
        let mut l = Layer::new(1, "A", g.clone());
        l.opacity = 0.5;
        let white = Layer::new(2, "B", TileGrid::filled(700, 400, px::pack(255, 255, 255, 255)));
        let dir = std::env::temp_dir().join(format!("pforge-test-{}.pforge", std::process::id()));
        save_native(&[white, l], 700, 400, 1, &DocExtras::default(), &dir).unwrap();
        let (w, h, active, layers, _) = load_native(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        assert_eq!((w, h, active, layers.len()), (700, 400, 1, 2));
        assert_eq!(layers[1].opacity, 0.5);
        assert_eq!(layers[1].grid.pixel(321, 123), g.pixel(321, 123));
        assert_eq!(layers[0].grid.pixel(5, 5), px::pack(255, 255, 255, 255));

        // Hidden pixels ("PFLX" tag 5) survive a save.
        let m = [1.0, 0.0, -150.0, 0.0, 1.0, -100.0, 0.0, 0.0, 1.0];
        let opts = crate::engine::xform::XformOpts { keep: true, ..Default::default() };
        let cropped = crate::engine::xform::layer_transformed_with(&Layer::new(3, "C", g.clone()), &m, 300, 200, opts);
        assert!(cropped.hidden.is_some());
        save_native(std::slice::from_ref(&cropped), 300, 200, 0, &DocExtras::default(), &dir).unwrap();
        let (_, _, _, back, _) = load_native(&dir).unwrap();
        std::fs::remove_file(&dir).ok();
        let hd = back[0].hidden.as_ref().expect("hidden pixels saved");
        assert_eq!(hd.rect(), RectI::new(-150, -100, 550, 300));
        assert_eq!(hd.grid.pixel(640, 350), g.pixel(640, 350));
    }

    #[test]
    fn native_v2_keeps_layer_model() {
        use crate::engine::layers::{LayerKind, LayerMask, ShapeGeom, ShapeLayer};
        use crate::engine::vector::StrokeStyle;
        let (w, h) = (600u32, 400u32);
        let mut base = Layer::new(1, "base", super::super::ops::procedural(w, h));
        base.background = true;
        let mut grp = Layer::new(7, "G", TileGrid::new(w, h));
        grp.kind = LayerKind::Group { pass_through: false, collapsed: true };
        // A live two-component shape: rotated rect minus a star, gradient
        // fill, dashed pattern stroke (the "PFLX" shape extras).
        use super::super::layers::{GradPaint, Paint, PatPaint, PathOp, ShapePart};
        let mut shape = ShapeLayer::new(
            ShapeGeom::Rect { x0: 50.0, y0: 40.0, x1: 300.0, y1: 200.0, radii: [20.0; 4] },
            Paint::Gradient(GradPaint { grad: super::super::gradient::Gradient::two("g", [1, 2, 3], [200, 100, 0]), kind: super::super::gradient::GradKind::Radial, angle: 30.0, scale: 0.8, reverse: true }),
            Some((Paint::Pattern(PatPaint { which: 2, scale: 1.5, angle: 10.0 }), StrokeStyle { width: 6.0, dashes: vec![2.0, 1.0], dash_offset: 0.5, ..Default::default() })),
        );
        shape.rot = 0.3;
        shape.parts.push(ShapePart { geom: ShapeGeom::Polygon { x0: 100.0, y0: 60.0, x1: 200.0, y1: 160.0, sides: 5, star: 0.5 }, rot: -0.2, xf: [1.0, 0.1, 0.0, 0.0, 1.0, 0.0], op: PathOp::Subtract });
        let mut sl = Layer::new(5, "S", TileGrid::new(w, h)).with_kind(LayerKind::Shape(Arc::new(shape.clone())));
        sl.parent = Some(7);
        sl.mask = Some(LayerMask::new(crate::engine::selection::MaskGrid::from_selection(&crate::engine::selection::Selection::ellipse(RectI::new(0, 0, 300, 300)), w, h)));
        let mut clipped = Layer::new(6, "C", TileGrid::filled(w, h, px::pack(9, 9, 9, 255)));
        clipped.clip = true;
        clipped.parent = Some(7);
        let mut adj = Layer::new(8, "A", TileGrid::new(w, h));
        let k = crate::engine::filters::catalog::Kind::Posterize;
        adj.adj_params = Some(Arc::new((k, vec![5.0], false)));
        let ctx = crate::engine::filters::catalog::BuildCtx { fg: [0, 0, 0], bg: [255, 255, 255], area: RectI::new(0, 0, w as i32, h as i32), stats: None, source: None, disp_map: None, layer_mask: None, doc_w: w, doc_h: h };
        adj.kind = LayerKind::Adjustment(k.build(&[5.0], false, &ctx));
        // A text layer with styled runs, paragraph styles, vertical options,
        // a distorted warp and a type-on-path path (the "PFLX" chunk).
        use super::super::text::{Caps, LastLine, Position, TextLayer, TextPath, TextStyle, Warp, WarpStyle};
        let mut tl = TextLayer::new("Hello styled\nworld", TextStyle { size: 30.0, ..Default::default() }, (10.0, 40.0), Some((200.0, 120.0)), false);
        tl.set_char_style(0, 5, |s| {
            s.color = [200, 10, 10];
            s.underline = true;
            s.caps = Caps::Small;
            s.features.dlig = true;
            s.features.ss = 0b101;
        });
        tl.set_char_style(6, 12, |s| {
            s.position = Position::Super;
            s.baseline_shift = 3.0;
            s.faux_italic = true;
        });
        tl.set_para_style(13, 13, |p| {
            p.last = LastLine::Center;
            p.indent_first = 12.0;
            p.space_before = 4.0;
            p.hyphenate = true;
        });
        tl.tcy = vec![(1, 3)];
        tl.upright_roman = true;
        tl.warp = Warp { style: WarpStyle::Fisheye, bend: -0.3, h_distort: 0.25, v_distort: -0.1, vertical: true };
        tl.path = Some(Arc::new(TextPath { path: super::super::vector::Path::ellipse(80.0, 80.0, 50.0, 40.0), start: 0.1, end: 0.9, flip: true, area: false }));
        tl.relayout();
        let mut txt = Layer::new(9, "text", TileGrid::new(w, h));
        txt.kind = LayerKind::Text(Arc::new(tl.clone()));
        let layers = vec![base, sl, clipped, grp, adj, txt];
        let path = std::env::temp_dir().join(format!("pforge-v2-{}.pforge", std::process::id()));
        let extras = DocExtras {
            paths: vec![super::super::document::PathItem { name: "Path 1".into(), path: super::super::vector::Path::ellipse(50.0, 60.0, 30.0, 20.0) }],
            ann: super::super::annot::Annotations {
                samplers: vec![(3.0, 4.5)],
                notes: vec![super::super::annot::Note { x: 1.0, y: 2.0, author: "me".into(), text: "hello ✓".into() }],
                counts: vec![super::super::annot::CountGroup { pts: vec![(5.0, 6.0), (7.0, 8.0)], label_size: 21.0, ..super::super::annot::CountGroup::new("G", [1, 2, 3]) }],
                artboards: vec![super::super::annot::Artboard { name: "A".into(), rect: RectI::new(-10, 0, 300, 200) }],
                slices: vec![super::super::annot::Slice { url: "https://x.y/?a=1&b=2".into(), target: "_blank".into(), alt: "alt ✓".into(), layer: Some(5), ..super::super::annot::Slice::new("s", RectI::new(1, 2, 3, 4)) }],
                // "PFMX": sampler modes, Count label size, the Measurement Log.
                sampler_modes: vec![super::super::color::ColorMode::Lab],
                vp_planes: vec![super::super::vanish::Plane { c: [(10.0, 20.0), (300.0, 40.0), (280.0, 200.0), (20.0, 180.0)] }],
                measure: super::super::measure::Measurements {
                    scale: super::super::measure::MeasureScale { px: 100.0, length: 2.54, units: "cm".into() },
                    records: vec![super::super::measure::ruler((0.0, 0.0), (30.0, 40.0), &Default::default(), "doc")],
                },
            },
            selected: vec![5, 9],
        };
        save_native(&layers, w, h, 2, &extras, &path).unwrap();
        let (_, _, _, back, bx) = load_native(&path).unwrap();
        assert_eq!(bx.paths, extras.paths);
        assert_eq!(bx.ann, extras.ann);
        std::fs::remove_file(&path).ok();
        let doc = native_to_document(w, h, 2, back, bx);
        assert!(matches!(doc.layers[3].kind, LayerKind::Group { pass_through: false, collapsed: true }));
        // Locked Background, multi-selection, and the legacy artboard annotation
        // turned into an artboard layer.
        assert!(doc.layers[0].background && !doc.layers[1].background);
        assert_eq!(doc.selected, vec![5, 9]);
        assert!(doc.ann.artboards.is_empty());
        assert!(matches!(&doc.layers.last().unwrap().kind, LayerKind::Artboard(a) if a.rect == RectI::new(-10, 0, 300, 200)));
        assert_eq!(doc.layers[1].parent, Some(7));
        assert!(doc.layers[2].clip);
        match &doc.layers[1].kind {
            LayerKind::Shape(s) => assert_eq!(**s, shape),
            _ => panic!("shape lost"),
        }
        assert_eq!(doc.layers[1].mask.as_ref().unwrap().grid.at(100, 150), 255);
        assert_eq!(doc.layers[4].adj_params.as_ref().unwrap().1, vec![5.0]);
        assert!(matches!(doc.layers[4].kind, LayerKind::Adjustment(_)));
        match &doc.layers[5].kind {
            LayerKind::Text(t) => assert_eq!(**t, tl),
            _ => panic!("text lost"),
        }
        assert_eq!(crate::engine::composite::sample(&doc.layers, 20, 20), crate::engine::composite::sample(&layers, 20, 20));
    }

    #[test]
    fn png_export_roundtrip() {
        let g = super::super::ops::procedural(300, 200);
        let mut top = Layer::new(2, "half", TileGrid::filled(300, 200, px::pack(0, 0, 255, 255)));
        top.opacity = 0.5;
        let base = Layer::new(1, "base", g);
        let path = std::env::temp_dir().join(format!("pforge-test-{}.png", std::process::id()));
        export(&[base.clone(), top.clone()], 300, 200, &path).unwrap();
        let back = open_image(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!((back.w, back.h), (300, 200));
        let want = crate::engine::composite::sample(&[base, top], 123, 77);
        assert_eq!(back.pixel(123, 77), want);
    }
}
