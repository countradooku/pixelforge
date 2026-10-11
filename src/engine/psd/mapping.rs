//! Checked mapping between PSD records and the existing tiled RGB8 engine.
//!
//! Original straight channels remain in the backing tree. An unchanged layer
//! never passes through premultiply/unpremultiply on save. Unsupported live
//! semantics are errors, so this boundary cannot silently rasterize a layer.

use std::{path::Path, sync::Arc};

use anyhow::{Context, Result, bail, ensure};

use super::{AdditionalInfo, ChannelData, Header, ImageResource, LayerRecord, Limits, PsdDocument, RawBytes, Version, compression};
use crate::engine::{blend::BlendMode, composite, document::{Document, Layer}, grid::{self, TileGrid, TS}, io::DocExtras, layers::{LayerKind, LayerMask}, px, selection::MaskGrid};

pub(crate) struct Backing {
    pub(crate) tree: PsdDocument,
    originals: Vec<Layer>,
    /// Record indices follow stable engine layer IDs, not UI row positions.
    records: Vec<(u64, usize)>,
    active: usize,
    selected: Vec<u64>,
}

/// Labels are presentation only; every original resource and channel sample
/// remains in the backing tree, including names newer than this reader.
pub(crate) fn channel_labels(backing: Option<&Backing>) -> Vec<String> {
    let mut names = vec!["Red".into(), "Green".into(), "Blue".into()];
    let Some(backing) = backing else { return names; };
    let tree = &backing.tree;
    let mut extra = Vec::new();
    if let Some(resource) = tree.resources.iter().find(|resource| resource.id == 1045) {
        let mut bytes = resource.data.as_slice();
        while bytes.len() >= 4 && extra.len() < 56 {
            let n = u32::from_be_bytes(bytes[..4].try_into().expect("four-byte prefix")) as usize;
            let Some(size) = n.checked_mul(2).and_then(|n| n.checked_add(4)) else { break; };
            let Some(value) = bytes.get(4..size) else { break; };
            let utf16: Vec<u16> = value.chunks_exact(2).map(|raw| u16::from_be_bytes([raw[0],raw[1]])).collect();
            extra.push(String::from_utf16_lossy(&utf16).trim_end_matches('\0').to_string());
            bytes = &bytes[size..];
        }
    }
    if extra.is_empty() {
        if let Some(resource) = tree.resources.iter().find(|resource| resource.id == 1006) {
            let mut bytes = resource.data.as_slice();
            while let Some((&length, tail)) = bytes.split_first() {
                let Some(name) = tail.get(..usize::from(length)) else { break; };
                if extra.len() >= 56 { break; }
                extra.push(String::from_utf8_lossy(name).into_owned());
                bytes = &tail[usize::from(length)..];
            }
        }
    }
    for index in 3..usize::from(tree.header.channels) {
        let fallback = if index == 3 && tree.layer_mask.merged_alpha { "Transparency".into() } else { format!("Extra channel {}",index-2) };
        names.push(extra.get(index-3).filter(|name| !name.is_empty()).cloned().unwrap_or(fallback));
    }
    names
}

pub(crate) fn is_psd(path: &Path) -> bool {
    path.extension().is_some_and(|s| s.eq_ignore_ascii_case("psd") || s.eq_ignore_ascii_case("psb"))
}

pub(crate) fn limits() -> Limits {
    // This is a format allocation ceiling, not a second memory budget. Tiles
    // and retained input are charged to the existing shared MemoryBudget.
    Limits { max_file_bytes: 8usize << 30, max_decoded_bytes: 4usize << 30, ..Limits::default() }
}

pub(crate) fn open(path: &Path) -> Result<Document> {
    let mut doc = from_tree(PsdDocument::read(path, &limits())?)?;
    doc.path = Some(path.to_path_buf());
    Ok(doc)
}

fn block<'a>(layer: &'a LayerRecord, key: &[u8; 4]) -> Option<&'a [u8]> {
    layer.additional_info.iter().find(|b| &b.key == key).map(|b| b.data.as_slice())
}

fn be_u32(bytes: &[u8]) -> Result<u32> {
    Ok(u32::from_be_bytes(bytes.get(..4).context("Truncated PSD integer")?.try_into()?))
}

fn group_kind(record: &LayerRecord) -> Result<u32> {
    let block = record.additional_info.iter().find(|b| matches!(&b.key,b"lsct"|b"lsdk"));
    block.map(|b|super::masks::SectionDivider::parse(&b.data).map(|d|d.kind)).transpose().map(|kind|kind.unwrap_or(0)).map_err(Into::into)
}

fn validate_editable(record: &LayerRecord) -> Result<()> {
    const LIVE: &[[u8; 4]] = &[
        *b"lfx2", *b"lmfx", *b"lrFX", *b"SoCo", *b"GdFl", *b"PtFl",
        *b"vmsk", *b"vsms", *b"vogk", *b"vscg", *b"vstk", *b"SoLd", *b"SoLE", *b"PlLd", *b"plLd",
        *b"levl", *b"curv", *b"brit", *b"hue2", *b"hue ", *b"blnc", *b"blwh", *b"mixr", *b"selc",
        *b"grdm", *b"phfl", *b"expA", *b"vibA", *b"post", *b"thrs", *b"nvrt", *b"clrL",
        *b"artb", *b"artd", *b"CgEd", *b"cinf",
    ];
    for entry in &record.additional_info {
        if LIVE.contains(&entry.key) {
            bail!("Layer {:?} contains {} whose editable mapping is not implemented; the raw codec preserves it", record.name(), String::from_utf8_lossy(&entry.key));
        }
        match &entry.key {
            b"lsct"|b"lsdk" => {
                let divider=super::masks::SectionDivider::parse(&entry.data)?;
                ensure!(divider.subtype.is_none_or(|value|value==0), "Layer {:?}: scene-group timeline mapping is not implemented",record.name());
            }
            b"knko"|b"lmgm"|b"vmgm" => {
                let value=*entry.data.as_slice().first().context("Truncated PSD blending flag")?;
                ensure!(value==0,"Layer {:?}: nondefault {} rendering is not implemented",record.name(),String::from_utf8_lossy(&entry.key));
            }
            // Photoshop's ordinary clipping stack and independent interior
            // effects are the behaviors currently implemented by the engine.
            b"clbl" => ensure!(entry.data.as_slice().first()==Some(&1),"Layer {:?}: ungrouped clipped-element blending is not implemented",record.name()),
            b"infx" => ensure!(entry.data.as_slice().first()==Some(&0),"Layer {:?}: grouped interior-effect blending is not implemented",record.name()),
            b"lspf" => {
                let protection=be_u32(entry.data.as_slice())?;
                // The existing Background abstraction already enforces this
                // lock. Arbitrary transparency/pixel/position locks do not.
                let background=record.flags&1!=0 && !record.channels.iter().any(|c|c.id==-1);
                ensure!(protection==0 || (background && protection&!5==0),"Layer {:?}: non-background protection flags require a lock mapping",record.name());
            }
            _ => {}
        }
    }
    ensure!(record.flags&1==0 || !record.channels.iter().any(|c|c.id==-1),"Layer {:?}: transparency lock requires a non-background lock mapping",record.name());
    ensure!(record.clipping<=1,"Invalid PSD clipping flag");
    super::super::advanced_blend::AdvancedBlend::from_psd(record.blending_ranges.as_slice(),block(record,b"brst").unwrap_or_default())?;
    Ok(())
}

pub(crate) fn from_tree(tree: PsdDocument) -> Result<Document> {
    ensure!(tree.header.depth == 8 && tree.header.color_mode == 3,
        "This document is {} at {} bits/channel. Colour-mode conversion is not implemented; raw PSD preservation remains available", mode_name(tree.header.color_mode), tree.header.depth);
    ensure!(tree.header.channels >= 3, "RGB PSD has fewer than three composite channels");
    let (w, h) = (tree.header.width, tree.header.height);
    // Validate compression even when layer pixels are used for the display.
    drop(tree.decode_composite(&limits())?);
    let mut layers = Vec::new();
    let mut records = Vec::new();
    let mut containers: Vec<Vec<u64>> = Vec::new();
    for (ix, record) in tree.layer_mask.layers.iter().enumerate() {
        let kind = group_kind(record)?;
        ensure!(kind <= 3, "Unknown PSD group divider {kind}");
        if kind == 3 {
            containers.push(Vec::new());
            continue;
        }
        validate_editable(record)?;
        let id = block(record, b"lyid").map(be_u32).transpose()?.map(u64::from).unwrap_or((ix + 1) as u64);
        ensure!(id != 0 && !records.iter().any(|(old, _)| *old == id), "Duplicate or zero PSD layer ID");
        let mut layer = Layer::new(id, record.name(), if kind == 1 || kind == 2 { TileGrid::new(w, h) } else { decode_pixels(record, &tree.header)? });
        layer.visible = record.flags & 2 == 0;
        layer.opacity = f32::from(record.opacity) / 255.0;
        layer.clip = record.clipping == 1;
        layer.fill = block(record, b"iOpa").and_then(|v| v.first()).copied().map(|v| f32::from(v)/255.0).unwrap_or(1.0);
        let mut blend = record.blend_mode;
        if kind == 1 || kind == 2 {
            if let Some(divider) = block(record, b"lsct").or_else(|| block(record, b"lsdk")) {
                if divider.len() >= 12 && &divider[4..8] == b"8BIM" { blend.copy_from_slice(&divider[8..12]); }
            }
            layer.kind = LayerKind::Group { pass_through: blend == *b"pass", collapsed: kind == 2 };
            let children = containers.pop().context("PSD group has no bounding divider")?;
            for child in &mut layers {
                let child: &mut Layer = child;
                if children.contains(&child.id) { child.parent = Some(id); }
            }
        }
        if kind == 0 && block(record,b"TySh").is_some() {
            // Preserve Photoshop's raster cache; with_kind would immediately
            // rerender text using a different layout engine.
            let mut text = super::text::decode(record)?;
            if tree.layer_mask.additional_info.iter().any(|block| block.key == *b"Txt2") {
                super::text::mark_global_cache(&mut text);
            }
            layer.kind = LayerKind::Text(Arc::new(text));
        }
        layer.blend = if blend == *b"pass" { BlendMode::Normal } else { blend_from_key(blend)? };
        let advanced = super::super::advanced_blend::AdvancedBlend::from_psd(record.blending_ranges.as_slice(),block(record,b"brst").unwrap_or_default())?;
        ensure!(advanced.is_identity() || !matches!(layer.kind,LayerKind::Group{pass_through:true,..}),"Layer {:?}: advanced blending on pass-through groups needs a backdrop mapping",layer.name);
        layer.advanced_blend = (!advanced.is_identity()).then(||Arc::new(advanced));
        layer.mask = decode_mask(record, &tree.header)?;
        layer.background = record.flags & 1 != 0 && !record.channels.iter().any(|channel| channel.id == -1);
        if let Some(parent) = containers.last_mut() { parent.push(id); }
        records.push((id, ix));
        layers.push(layer);
    }
    ensure!(containers.is_empty(), "Unterminated PSD group");
    if layers.is_empty() {
        let plane = tree.decode_composite(&limits())?;
        let n = (w as usize).checked_mul(h as usize).context("PSD dimensions overflow")?;
        let rgba = |x: usize, y: usize| {
            let i = y*w as usize+x;
            let alpha = if tree.layer_mask.merged_alpha && tree.header.channels > 3 { plane[3*n+i] } else { 255 };
            px::premultiply(plane[i],plane[n+i],plane[2*n+i],alpha)
        };
        layers.push(Layer::new(1, "Background", grid_from(w,h,rgba)));
    }
    let active = layers.len().saturating_sub(1);
    let mut doc = Document::from_layers(w, h, layers, active);
    read_private(&tree, &mut doc)?;
    doc.psd = Some(Arc::new(Backing { tree, originals: doc.layers.clone(), records, active: doc.active, selected: doc.selected.clone() }));
    Ok(doc)
}

pub(crate) fn mode_name(mode: u16) -> &'static str {
    match mode { 0 => "Bitmap", 1 => "Grayscale", 2 => "Indexed", 3 => "RGB", 4 => "CMYK", 7 => "Multichannel", 8 => "Duotone", 9 => "Lab", _ => "Unknown" }
}

fn grid_from(w: u32, h: u32, pixel: impl Fn(usize, usize) -> u32 + Sync) -> TileGrid {
    grid::build_grid(w, h, |tx, ty| {
        let mut pixels = vec![0; TS*TS].into_boxed_slice();
        for y in 0..TS {
            for x in 0..TS {
                let gx = (tx as usize*TS+x).min(w as usize-1);
                let gy = (ty as usize*TS+y).min(h as usize-1);
                pixels[y*TS+x] = pixel(gx,gy);
            }
        }
        grid::finish_tile(pixels)
    })
}

fn decode_pixels(record: &LayerRecord, header: &Header) -> Result<TileGrid> {
    let (rw,rh) = record.dimensions()?;
    let n = (rw as usize).checked_mul(rh as usize).context("Layer dimensions overflow")?;
    ensure!(n <= limits().max_decoded_bytes, "Layer exceeds decoded allocation limit");
    let _plane_memory=super::super::memory::global().track_cpu(n.saturating_mul(4) as u64);
    let mut planes: [Option<Vec<u8>>; 4] = Default::default();
    for channel in &record.channels {
        let ix = match channel.id { 0..=2 => channel.id as usize, -1 => 3, _ => continue };
        ensure!(planes[ix].is_none(), "Duplicate PSD colour/alpha channel");
        planes[ix] = Some(compression::decode(channel.encoded.as_slice(),rw,rh,8,header.version,&limits())?);
    }
    ensure!(n == 0 || planes[..3].iter().all(Option::is_some), "Pixel layer {:?} is missing RGB channel data", record.name());
    Ok(grid_from(header.width,header.height,|x,y| {
        let sx = x as i64-i64::from(record.bounds[1]);
        let sy = y as i64-i64::from(record.bounds[0]);
        if sx < 0 || sy < 0 || sx >= i64::from(rw) || sy >= i64::from(rh) { return 0; }
        let i = sy as usize*rw as usize+sx as usize;
        let c = |k: usize, default| planes[k].as_ref().map(|p| p[i]).unwrap_or(default);
        px::premultiply(c(0,0),c(1,0),c(2,0),c(3,255))
    }))
}

fn decode_mask(record: &LayerRecord, header: &Header) -> Result<Option<LayerMask>> {
    let has_real=record.channels.iter().any(|channel|channel.id==-3);
    let metadata=super::masks::MaskMetadata::parse(&record.mask,has_real)?;
    let Some(channel)=record.channels.iter().find(|channel|channel.id==-2) else {
        ensure!(!has_real,"Real user mask channel without composite user mask");
        ensure!(metadata.is_none(),"Mask metadata without a pixel mask channel is not editable");
        return Ok(None);
    };
    let metadata=metadata.context("Mask channel without mask metadata")?;
    ensure!(!has_real,"Layer {:?}: real and combined masks need distinct live mask representations",record.name());
    ensure!(metadata.parameters.as_ref().is_none_or(super::masks::MaskParameters::is_neutral),
        "Layer {:?}: live mask density or feather rendering is not implemented",record.name());
    let raster=metadata.channel(-2)?;
    ensure!(raster.flags&!(1|2|4|16)==0,"Layer {:?}: derived or unknown mask flags need a live mapping",record.name());
    let bounds=raster.absolute_bounds(record.bounds)?;
    let w=u32::try_from(i64::from(bounds[3])-i64::from(bounds[1]))?;
    let h=u32::try_from(i64::from(bounds[2])-i64::from(bounds[0]))?;
    let bytes=compression::decode(channel.encoded.as_slice(),w,h,8,header.version,&limits())?;
    let _mask_memory=crate::engine::memory::global().track_cpu(bytes.capacity() as u64);
    let mask=grid_from(header.width,header.height,|x,y| {
        let x=x as i64-i64::from(bounds[1]);let y=y as i64-i64::from(bounds[0]);
        let v=if x>=0 && y>=0 && x<i64::from(w) && y<i64::from(h) {bytes[y as usize*w as usize+x as usize]} else {raster.default};
        let v=if raster.inverted() {255-v} else {v};px::pack(v,v,v,255)
    });
    let mut mask=LayerMask::new(MaskGrid::from_gray(&mask));mask.enabled=raster.enabled();Ok(Some(mask))
}

pub(crate) fn blend_from_key(key: [u8;4]) -> Result<BlendMode> {
    use BlendMode::*;
    Ok(match &key {
        b"norm" => Normal, b"dark" => Darken, b"mul " => Multiply, b"idiv" => ColorBurn,
        b"lbrn" => LinearBurn, b"lite" => Lighten, b"scrn" => Screen, b"div " => ColorDodge,
        b"lddg" => LinearDodge, b"over" => Overlay, b"sLit" => SoftLight, b"hLit" => HardLight,
        b"diff" => Difference, b"smud" => Exclusion, b"hue " => Hue, b"sat " => Saturation,
        b"colr" => Color, b"lum " => Luminosity,
        b"diss" => Dissolve, b"dkCl" => DarkerColor, b"lgCl" => LighterColor,
        b"vLit" => VividLight, b"lLit" => LinearLight, b"pLit" => PinLight,
        b"hMix" => HardMix, b"fsub" => Subtract, b"fdiv" => Divide,
        _ => bail!("PSD blend mode {} has no verified engine mapping",String::from_utf8_lossy(&key)),
    })
}

fn blend_key(mode: BlendMode) -> [u8;4] {
    use BlendMode::*;
    *match mode {
        Normal => b"norm", Darken => b"dark", Multiply => b"mul ", ColorBurn => b"idiv",
        LinearBurn => b"lbrn", Lighten => b"lite", Screen => b"scrn", ColorDodge => b"div ",
        LinearDodge => b"lddg", Overlay => b"over", SoftLight => b"sLit", HardLight => b"hLit",
        Difference => b"diff", Exclusion => b"smud", Hue => b"hue ", Saturation => b"sat ", Color => b"colr", Luminosity => b"lum ",
        Dissolve => b"diss", DarkerColor => b"dkCl", LighterColor => b"lgCl",
        VividLight => b"vLit", LinearLight => b"lLit", PinLight => b"pLit",
        HardMix => b"hMix", Subtract => b"fsub", Divide => b"fdiv",
    }
}

fn same_grid(a: &TileGrid, b: &TileGrid) -> bool {
    a.w == b.w && a.h == b.h && a.tiles.iter().zip(&b.tiles).all(|(a,b)| grid::same_tile(a,b))
}

fn same_mask(a: &Option<LayerMask>, b: &Option<LayerMask>) -> bool {
    match (a,b) {
        (None,None) => true,
        (Some(a),Some(b)) => a.enabled == b.enabled && same_mask_pixels(a,b),
        _ => false,
    }
}

fn same_mask_pixels(a:&LayerMask,b:&LayerMask)->bool {
    a.grid.w==b.grid.w && a.grid.h==b.grid.h && a.grid.tiles.len()==b.grid.tiles.len()
        && a.grid.tiles.iter().zip(&b.grid.tiles).all(|(a,b)|crate::engine::masks::same_mask_tile(a,b))
}

fn update_preserved_mask(record:&mut LayerRecord,layer:&Layer,old:&Layer,w:u32,h:u32,version:Version)->Result<()> {
    if let (Some(mask),Some(previous))=(&layer.mask,&old.mask) {
        if same_mask_pixels(mask,previous) {
            // A toggle changes only the user-mask disabled flag. Keep the
            // encoded samples, relative bounds, parameters and opaque suffix.
            let mut bytes=record.mask.as_slice().to_vec();
            ensure!(bytes.len()>=18,"Preserved mask metadata is truncated");
            bytes[17]=(bytes[17]&!2) | if mask.enabled {0} else {2};
            record.mask=RawBytes::from_vec(bytes);
            return Ok(());
        }
        let metadata=super::masks::MaskMetadata::parse(&record.mask,record.channels.iter().any(|c|c.id==-3))?
            .context("Preserved mask samples have no metadata")?;
        let [top,left,bottom,right]=metadata.user.absolute_bounds(record.bounds)?;
        let has_samples=bottom>top && right>left;
        ensure!(!has_samples || (top>=0 && left>=0 && bottom<=h as i32 && right<=w as i32),
            "Editing off-canvas mask samples requires a hidden-mask mapping; the original mask is preserved");
    }
    // Deleting a mask explicitly discards all its samples, including hidden
    // ones. Adding a new mask has no preserved samples to invalidate.
    replace_mask(record,layer,w,h,version)
}

fn validate_output(layers: &[Layer], extras: &DocExtras) -> Result<()> {
    ensure!(!layers.is_empty(), "PSD needs at least one layer");
    ensure!(extras.paths.is_empty() && extras.ann == Default::default(),
        "PSD mapping for paths, annotations, measurements, slices or Vanishing Point is incomplete; keep this document as .pforge");
    for layer in layers {
        ensure!(matches!(layer.kind,LayerKind::Pixel | LayerKind::Group{..} | LayerKind::Text(_)),
            "Layer {:?} ({}) has no safe editable PSD writer; keep this document as .pforge",layer.name,layer.kind.name());
        ensure!(layer.style.is_none(), "Layer {:?}: PSD layer-style writer is incomplete",layer.name);
        ensure!(layer.offset == (0,0), "Commit the layer movement before saving");
        ensure!(layer.opacity.is_finite() && layer.fill.is_finite(), "Invalid layer opacity");
        ensure!(!matches!(layer.kind,LayerKind::Group{pass_through:true,..}) || layer.blend==BlendMode::Normal,
            "Layer {:?}: a pass-through group cannot save a separate blend mode; group isolation changes are not implemented",layer.name);
        if let Some(settings)=&layer.advanced_blend {
            settings.validate()?;
            ensure!(settings.is_identity() || !matches!(layer.kind,LayerKind::Group{pass_through:true,..}),"Layer {:?}: advanced blending on pass-through groups needs a backdrop mapping",layer.name);
        }
    }
    Ok(())
}

pub(crate) fn to_tree(layers: &[Layer], w: u32, h: u32, active: usize, extras: &DocExtras, backing: Option<&Backing>, version: Version) -> Result<PsdDocument> {
    validate_output(layers,extras)?;
    Header::rgb(w,h,4,version).validate()?;
    ensure!((w as usize).checked_mul(h as usize).and_then(|n| n.checked_mul(4)).is_some_and(|n| n <= limits().max_decoded_bytes), "PSD composite exceeds decoded allocation limit");
    let limits = limits();
    let mut tree = if let Some(backing) = backing {
        ensure!(backing.tree.header.width == w && backing.tree.header.height == h,
            "PSD canvas changes may invalidate preserved blocks; this edit cannot yet be saved safely");
        ensure!(backing.tree.header.version == version, "Changing PSD/PSB versions with preserved blocks is not implemented");
        ensure!(layers.len() == backing.originals.len() && layers.iter().zip(&backing.originals).all(|(a,b)| a.id == b.id && a.parent == b.parent),
            "PSD layer insertion, deletion or reordering may invalidate preserved references; cannot save this edit safely");
        backing.tree.clone()
    } else {
        PsdDocument::new(Header::rgb(w,h,4,version),Vec::new())
    };
    if backing.is_none() {
        ensure!(!layers.iter().any(|layer| matches!(layer.kind,LayerKind::Text(_))), "Writing a new live PSD text layer needs an authored text template; retain .pforge");
    }
    let mut changed = backing.is_none();
    if let Some(backing) = backing {
        for (layer,old) in layers.iter().zip(&backing.originals) {
            let Some((_,ix)) = backing.records.iter().find(|(id,_)| *id == layer.id) else {
                ensure!(same_grid(&layer.grid,&old.grid) && layer.name == old.name && layer.opacity == old.opacity && layer.blend == old.blend && layer.visible == old.visible && layer.fill == old.fill && layer.mask.is_none() && !layer.clip && layer.advanced_blend == old.advanced_blend,
                    "Editing a layerless PSD requires a layer-table migration which is not implemented");
                continue;
            };
            let record = &mut tree.layer_mask.layers[*ix];
            ensure!(std::mem::discriminant(&layer.kind) == std::mem::discriminant(&old.kind), "Changing a preserved PSD layer kind is not implemented");
            if let (LayerKind::Text(old_text),LayerKind::Text(new_text)) = (&old.kind,&layer.kind) {
                if old_text != new_text {
                    ensure!(!backing.tree.layer_mask.additional_info.iter().any(|block| block.key == *b"Txt2"),
                        "Text edits cannot yet update Photoshop's preserved document-level Txt2 engine data; the original data is retained");
                    changed = true;
                }
                super::text::encode(record,old_text,new_text)?;
            }
            if !same_grid(&layer.grid,&old.grid) {
                ensure!(record.bounds[0] >= 0 && record.bounds[1] >= 0 && record.bounds[2] <= h as i32 && record.bounds[3] <= w as i32,
                    "Editing off-canvas PSD pixels requires a hidden-pixel mapping");
                ensure!(layer.hidden.is_none(),"Hidden pixels cannot yet be written to PSD safely");
                replace_pixels(record,layer,w,h,version)?;
                changed = true;
            }
            if !same_mask(&layer.mask,&old.mask) { update_preserved_mask(record,layer,old,w,h,version)?; changed=true; }
            if layer.name != old.name { record.set_name(&layer.name); }
            if layer.opacity != old.opacity { record.opacity=opacity(layer.opacity);changed=true; }
            if layer.visible != old.visible { record.flags=(record.flags & !2) | if layer.visible {0} else {2};changed=true; }
            if layer.clip != old.clip { record.clipping=u8::from(layer.clip);changed=true; }
            if layer.blend != old.blend {
                record.blend_mode=blend_key(layer.blend);
                if let LayerKind::Group{pass_through,..}=layer.kind {
                    for entry in &mut record.additional_info {
                        if matches!(&entry.key,b"lsct"|b"lsdk") {
                            let mut divider=super::masks::SectionDivider::parse(&entry.data)?;
                            divider.blend_mode=Some(if pass_through {*b"pass"} else {blend_key(layer.blend)});
                            entry.data=RawBytes::from_vec(divider.encode()?);
                        }
                    }
                }
                changed=true;
            }
            if layer.fill != old.fill { set_block(record,*b"iOpa",vec![opacity(layer.fill)]);changed=true; }
            if layer.advanced_blend != old.advanced_blend {replace_advanced_blend(record,layer);changed=true;}
            if let (LayerKind::Group{pass_through,collapsed},LayerKind::Group{pass_through:old_pass,collapsed:old_collapsed}) = (&layer.kind,&old.kind) {
                ensure!(pass_through == old_pass && collapsed == old_collapsed,"Changing preserved group semantics is not implemented");
            }
            ensure!(layer.background == old.background,"Changing PSD Background lock semantics is not implemented");
        }
    } else {
        // Containers are emitted with bounding records preceding their children.
        fn append(tree: &mut PsdDocument,layers: &[Layer],parent: Option<u64>,w: u32,h: u32,depth: usize) -> Result<()> {
            ensure!(depth < 64,"PSD group nesting limit exceeded");
            for layer in layers.iter().filter(|layer| layer.parent == parent) {
                ensure!(layer.hidden.is_none(),"Hidden cropped pixels have no PSD mapping yet");
                if let LayerKind::Group{pass_through,collapsed} = layer.kind {
                    let mut end = LayerRecord::new([0;4],"</Layer group>");
                    set_block(&mut end,*b"lsct",3u32.to_be_bytes().to_vec());
                    tree.layer_mask.layers.push(end);
                    append(tree,layers,Some(layer.id),w,h,depth+1)?;
                    let mut record = make_record(layer,w,h,tree.header.version)?;
                    let mut divider = (if collapsed {2u32} else {1u32}).to_be_bytes().to_vec();
                    divider.extend_from_slice(b"8BIM");
                    let group_blend = if pass_through {*b"pass"} else {blend_key(layer.blend)};
                    divider.extend_from_slice(&group_blend);
                    set_block(&mut record,*b"lsct",divider);
                    tree.layer_mask.layers.push(record);
                } else { tree.layer_mask.layers.push(make_record(layer,w,h,tree.header.version)?); }
            }
            Ok(())
        }
        append(&mut tree,layers,None,w,h,0)?;
        ensure!(tree.layer_mask.layers.iter().filter(|r| group_kind(r).ok() != Some(3)).count() == layers.len(), "Orphaned or cyclic layer hierarchy");
        tree.layer_mask.present=true;
        tree.layer_mask.layer_info_present=true;
        tree.layer_mask.merged_alpha=true;
        tree.layer_mask.global_mask=Some(RawBytes::empty());
    }
    if changed {
        let _rgba_memory=super::super::memory::global().track_cpu(u64::from(w)*u64::from(h)*4);
        let rgba = composite::export_rgba(layers,w,h,None);
        let n = w as usize*h as usize;
        let count = tree.header.channels as usize;
        // Alpha/spot channels have their own samples. A composite edit only
        // changes RGB and the specifically flagged merged-transparency plane.
        let plane_bytes=n.checked_mul(count).context("PSD merged plane size overflow")?;
        ensure!(plane_bytes<=limits.max_decoded_bytes,"PSD merged planes exceed decoded allocation limit");
        let _plane_memory=super::super::memory::global().track_cpu(plane_bytes as u64);
        let mut planes = if backing.is_some() { tree.decode_composite(&limits)? } else { vec![0; n*count] };
        let color_channels = if tree.layer_mask.merged_alpha && count > 3 {4} else {3};
        for (i,p) in rgba.chunks_exact(4).enumerate() { for c in 0..color_channels { planes[c*n+i]=p[c]; } }
        // Photoshop 27.10 rejects our ZIP-compressed RGB8 merged image even
        // though it accepts the identical layer ZIP streams and raw composite.
        // Use interoperable per-row PackBits for newly rendered composites;
        // untouched backing streams remain byte-exact above.
        tree.image_data=RawBytes::from_vec(compression::encode(&planes,w,h.checked_mul(count as u32).context("PSD composite rows overflow")?,8,version,compression::Method::PackBits,&limits)?);
    }
    // Only PixelForge's selection/active state is private. Equivalent Photoshop
    // features must gain real mappings before they can pass validate_output.
    if backing.is_none_or(|old| old.active != active || old.selected != extras.selected) { write_private(&mut tree,layers,active,&extras.selected)?; }
    Ok(tree)
}

fn opacity(value:f32)->u8 {(value.clamp(0.0,1.0)*255.0).round() as u8}
fn set_block(record:&mut LayerRecord,key:[u8;4],data:Vec<u8>) {
    let replacement=AdditionalInfo::new(key,data);
    if let Some(entry)=record.additional_info.iter_mut().find(|entry|entry.key==key) {*entry=replacement;} else {record.additional_info.push(replacement);}
}
fn replace_advanced_blend(record:&mut LayerRecord,layer:&Layer) {
    let ranges=layer.advanced_blend.as_ref().map(|settings|settings.psd_ranges()).unwrap_or_default();
    let restrictions=layer.advanced_blend.as_ref().map(|settings|settings.psd_restrictions()).unwrap_or_default();
    record.blending_ranges=RawBytes::from_vec(ranges);
    if !restrictions.is_empty() || block(record,b"brst").is_some() {set_block(record,*b"brst",restrictions);}
}
fn make_record(layer:&Layer,w:u32,h:u32,version:Version)->Result<LayerRecord> {
    let mut record=LayerRecord::new([0,0,h as i32,w as i32],&layer.name);
    record.opacity=opacity(layer.opacity);record.blend_mode=blend_key(layer.blend);record.clipping=u8::from(layer.clip);
    record.flags=8 | if layer.visible {0} else {2} | if layer.background {1} else {0};
    set_block(&mut record,*b"lyid",u32::try_from(layer.id)?.to_be_bytes().to_vec());
    if layer.fill != 1.0 {set_block(&mut record,*b"iOpa",vec![opacity(layer.fill)]);}
    if matches!(layer.kind,LayerKind::Pixel) {replace_pixels(&mut record,layer,w,h,version)?;} else {record.bounds=[0;4];}
    replace_mask(&mut record,layer,w,h,version)?;
    replace_advanced_blend(&mut record,layer);
    Ok(record)
}
fn replace_pixels(record:&mut LayerRecord,layer:&Layer,w:u32,h:u32,version:Version)->Result<()> {
    let bounds=[0,0,h as i32,w as i32];
    if record.bounds != bounds {
        ensure!(!record.channels.iter().any(|c|!matches!(c.id,-3..=2)),
            "Resizing layer pixels with preserved extra channels needs a channel mapping");
        // Relative masks use the layer origin. Rebase their coordinates before
        // expanding the RGB planes, preserving flags, parameters and opaque tail.
        if let Some(mut mask)=super::masks::MaskMetadata::parse(&record.mask,record.channels.iter().any(|c|c.id==-3))? {
            if mask.user.flags&1!=0 {mask.user.bounds=mask.user.absolute_bounds(record.bounds)?;}
            if let Some(real)=&mut mask.real {
                if real.flags&1!=0 {real.bounds=real.absolute_bounds(record.bounds)?;}
            }
            record.mask=RawBytes::from_vec(mask.encode()?);
        }
    }
    let mut row=vec![0u32;w as usize];
    let n=w as usize*h as usize;
    let _plane_memory=super::super::memory::global().track_cpu(n.saturating_mul(4).saturating_add(w as usize*4) as u64);
    let mut planes=vec![vec![0u8;n];4];
    for y in 0..h {
        layer.grid.read_region(0,y as i32,w as usize,1,&mut row);
        for (x,&p) in row.iter().enumerate() { for (c,&v) in px::unpremultiply(p).iter().enumerate() {planes[c][y as usize*w as usize+x]=v;} }
    }
    let mut channels=Vec::new();
    for (ix,plane) in planes.iter().enumerate() {
        if layer.background && ix==3 {continue;}
        channels.push(ChannelData{id:if ix==3 {-1} else {ix as i16},encoded:RawBytes::from_vec(compression::encode(plane,w,h,8,version,compression::Method::Zip,&limits())?)});
    }
    channels.extend(record.channels.iter().filter(|c|!matches!(c.id,-1|0..=2)).cloned());
    record.channels=channels;record.bounds=bounds;
    Ok(())
}
fn replace_mask(record:&mut LayerRecord,layer:&Layer,w:u32,h:u32,version:Version)->Result<()> {
    record.channels.retain(|c|c.id != -2);
    let Some(mask)=&layer.mask else {record.mask=RawBytes::empty();return Ok(());};
    let mut metadata=super::masks::MaskMetadata::parse(&record.mask,record.channels.iter().any(|c|c.id==-3))?.unwrap_or_else(||super::masks::MaskMetadata {
        user:super::masks::RasterMask{bounds:[0;4],default:0,flags:0},
        parameters:None,real:None,tail:RawBytes::from_vec(vec![0,0]),
    });
    // The live grid already includes inversion. Store absolute canvas bounds
    // and uninverted samples, retaining the effective outside colour and the
    // neutral parameters/opaque suffix from an existing mask.
    if metadata.user.inverted() {metadata.user.default=255-metadata.user.default;}
    metadata.user.bounds=[0,0,h as i32,w as i32];
    metadata.user.flags=(metadata.user.flags&!(1|2|4)) | if mask.enabled {0} else {2};
    record.mask=RawBytes::from_vec(metadata.encode()?);
    let _mask_memory=super::super::memory::global().track_cpu(u64::from(w)*u64::from(h));
    let mut pixels=vec![0;w as usize*h as usize];
    mask.grid.read_region(0,0,w as usize,h as usize,&mut pixels,false);
    record.channels.push(ChannelData{id:-2,encoded:RawBytes::from_vec(compression::encode(&pixels,w,h,8,version,compression::Method::Zip,&limits())?)});
    Ok(())
}

const PRIVATE_ID:u16=4000;
const PRIVATE_MAGIC:&[u8]=b"PixelForge\0\x00\x01";
fn write_private(tree:&mut PsdDocument,layers:&[Layer],active:usize,selected:&[u64])->Result<()> {
    let mut bytes=PRIVATE_MAGIC.to_vec();
    bytes.extend_from_slice(&layers.get(active).map(|l|l.id).unwrap_or(0).to_be_bytes());
    bytes.extend_from_slice(&u32::try_from(selected.len())?.to_be_bytes());
    for id in selected {bytes.extend_from_slice(&id.to_be_bytes());}
    let resource=ImageResource::new(PRIVATE_ID,bytes);
    if let Some(old)=tree.resources.iter_mut().find(|r|r.id==PRIVATE_ID && r.data.as_slice().starts_with(PRIVATE_MAGIC)) {*old=resource;} else {tree.resources.push(resource);}
    Ok(())
}
fn read_private(tree:&PsdDocument,doc:&mut Document)->Result<()> {
    let Some(resource)=tree.resources.iter().find(|r|r.id==PRIVATE_ID && r.data.as_slice().starts_with(PRIVATE_MAGIC)) else {return Ok(());};
    let bytes=&resource.data.as_slice()[PRIVATE_MAGIC.len()..];
    ensure!(bytes.len()>=12,"Truncated PixelForge private resource");
    let active=u64::from_be_bytes(bytes[..8].try_into()?);let count=be_u32(&bytes[8..])? as usize;
    ensure!(count<=32767 && count.checked_mul(8).and_then(|n|n.checked_add(12))==Some(bytes.len()),"Invalid PixelForge private selection length");
    doc.active=doc.layers.iter().position(|l|l.id==active).unwrap_or(doc.active);
    for id in bytes[12..].chunks_exact(8) {
        let id=u64::from_be_bytes(id.try_into()?);
        if doc.layers.iter().any(|l|l.id==id) {doc.selected.push(id);}
    }
    Ok(())
}

pub(crate) fn save(layers:&[Layer],w:u32,h:u32,active:usize,extras:&DocExtras,backing:Option<&Backing>,path:&Path)->Result<()> {
    let version=if path.extension().is_some_and(|e|e.eq_ignore_ascii_case("psb")) {Version::Psb} else {Version::Psd};
    ensure!(version==Version::Psb || (w<=30000 && h<=30000),"This canvas requires a .psb file");
    to_tree(layers,w,h,active,extras,backing,version)?.write(path,&limits())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> Document {
        let bottom = Layer::new(7, "Opaque base", grid_from(5,3,|x,y| px::pack((x*31) as u8,(y*59) as u8,97,255)));
        let mut top = Layer::new(23, "Upper α", grid_from(5,3,|x,y| px::premultiply(170,53,(x*23+y*31) as u8,if x==0 {0} else {143})));
        top.opacity = 192.0/255.0;
        top.fill = 204.0/255.0;
        Document::from_layers(5,3,vec![bottom,top],1)
    }
    fn extras(doc: &Document) -> DocExtras {
        DocExtras { paths: doc.paths.clone(), ann: doc.ann.clone(), selected: doc.selected.clone() }
    }
    fn tree(doc: &Document, version: Version) -> PsdDocument {
        to_tree(&doc.layers,doc.w,doc.h,doc.active,&extras(doc),doc.psd.as_deref(),version).unwrap()
    }

    #[test]
    fn pixel_psd_and_psb_preserve_channels_and_private_selection() {
        for version in [Version::Psd,Version::Psb] {
            let mut original = document();original.selected=vec![7,23];
            let encoded = tree(&original,version);
            let read = from_tree(PsdDocument::parse(encoded.encode(&limits()).unwrap(),&limits()).unwrap()).unwrap();
            assert_eq!(read.layers.len(),2);
            assert_eq!(read.layers[1].name,"Upper α");
            assert_eq!(read.active,1);assert_eq!(read.selected,vec![7,23]);
            for (before,after) in original.layers.iter().zip(&read.layers) {
                assert_eq!((before.id,before.opacity,before.fill),(after.id,after.opacity,after.fill));
                for y in 0..3 {for x in 0..5 {assert_eq!(before.grid.pixel(x,y),after.grid.pixel(x,y));}}
            }
            assert_eq!(tree(&read,version).encode(&limits()).unwrap(),encoded.encode(&limits()).unwrap(),"Unedited straight channels and encoded padding must not change");
        }
    }

    #[test]
    fn new_and_edited_merged_images_use_photoshop_compatible_planar_rle() {
        for version in [Version::Psd, Version::Psb] {
            let original = document();
            let first = tree(&original, version);
            let mut edited = from_tree(first.clone()).unwrap();
            edited.layers[1].opacity = 0.25;
            for (doc, encoded) in [(&original, first), (&edited, tree(&edited, version))] {
                // This wire-method assertion guards the native Photoshop
                // rejection reproduced by migration/native-incompatible.png.
                assert_eq!(&encoded.image_data.as_slice()[..2], &[0, 1]);
                let wire = encoded.encode(&limits()).unwrap();
                let reopened = PsdDocument::parse(wire, &limits()).unwrap();
                let planes = reopened.decode_composite(&limits()).unwrap();
                let rgba = composite::export_rgba(&doc.layers, doc.w, doc.h, None);
                let pixels = (doc.w * doc.h) as usize;
                for (i, pixel) in rgba.chunks_exact(4).enumerate() {
                    for channel in 0..4 {
                        assert_eq!(planes[channel * pixels + i], pixel[channel]);
                    }
                }
            }
        }
    }

    #[test]
    fn sibling_pixel_edit_preserves_unknown_blocks_and_straight_channels() {
        let mut raw=tree(&document(),Version::Psd);
        raw.resources.push(ImageResource::new(4107,vec![0,17,255,32,7]));
        raw.layer_mask.layers[0].additional_info.push(AdditionalInfo::new(*b"zzzz",vec![0,1,9,255,0,13]));
        let untouched=raw.layer_mask.layers[0].clone();
        let resources=raw.resources.clone();
        let mut doc=from_tree(raw).unwrap();
        doc.layers[1].replace_grid(grid_from(5,3,|_,_|px::pack(43,82,197,255)));
        let saved=tree(&doc,Version::Psd);
        assert_eq!(saved.layer_mask.layers[0],untouched);
        assert_eq!(saved.resources,resources);
        let decoded=from_tree(saved).unwrap();
        assert_eq!(decoded.layers[1].grid.pixel(4,2),px::pack(43,82,197,255));
    }

    #[test]
    fn user_mask_and_nested_groups_roundtrip_stable_ids() {
        let mut doc=document();
        let mut mask=LayerMask::new(MaskGrid::from_gray(&grid_from(5,3,|x,_| {let v=(x*53) as u8;px::pack(v,v,v,255)})));
        mask.enabled=false;doc.layers[1].mask=Some(mask);
        let mut group=Layer::new(50,"Isolated",TileGrid::new(5,3));
        group.kind=LayerKind::Group{pass_through:false,collapsed:true};
        doc.layers[1].parent=Some(50);doc.layers.push(group);
        let mut outer=Layer::new(90,"Outer",TileGrid::new(5,3));
        outer.kind=LayerKind::Group{pass_through:true,collapsed:false};doc.layers[2].parent=Some(90);doc.layers.push(outer);
        doc.selected=vec![23,90];doc.active=3;
        let raw=tree(&doc,Version::Psb);let read=from_tree(raw.clone()).unwrap();
        assert_eq!(read.layers.iter().map(|l|(l.id,l.parent)).collect::<Vec<_>>(),vec![(7,None),(23,Some(50)),(50,Some(90)),(90,None)]);
        let mask=read.layers[1].mask.as_ref().unwrap();assert!(!mask.enabled);
        let mut values = [0;5];mask.grid.read_region(0,1,5,1,&mut values,false);
        assert_eq!(values,[0,53,106,159,212]);
        assert_eq!(read.selected,vec![23,90]);assert_eq!(read.active,3);
        assert_eq!(tree(&read,Version::Psb),raw);
    }

    #[test]
    fn unsupported_live_semantics_fail_without_creating_output() {
        let mut doc=document();
        doc.layers[0].kind=LayerKind::Adjustment(crate::engine::filters::Filter::Invert);
        let path=std::env::temp_dir().join(format!("pixelforge-unsafe-mapping-{}.psd",std::process::id()));
        let _=std::fs::remove_file(&path);
        assert!(save(&doc.layers,doc.w,doc.h,doc.active,&extras(&doc),None,&path).is_err());
        assert!(!path.exists());
        let mut raw=tree(&document(),Version::Psd);
        raw.layer_mask.layers[0].additional_info.push(AdditionalInfo::new(*b"TySh",vec![0;4]));
        assert!(from_tree(raw).is_err());
    }

    #[test]
    fn changed_private_state_can_clear_selection() {
        let mut doc=document();doc.selected=vec![23];
        let mut read=from_tree(tree(&doc,Version::Psd)).unwrap();
        read.selected.clear();read.active=0;
        let after=from_tree(tree(&read,Version::Psd)).unwrap();
        assert!(after.selected.is_empty());assert_eq!(after.active,0);
    }

    #[test]
    fn preserved_references_block_unsafe_geometry_or_stack_changes() {
        let mut doc=from_tree(tree(&document(),Version::Psd)).unwrap();
        doc.layers.reverse();
        assert!(to_tree(&doc.layers,doc.w,doc.h,doc.active,&extras(&doc),doc.psd.as_deref(),Version::Psd).is_err());
        doc.layers.reverse();
        assert!(to_tree(&doc.layers,6,3,doc.active,&extras(&doc),doc.psd.as_deref(),Version::Psd).is_err());
        assert!(to_tree(&doc.layers,5,3,doc.active,&extras(&doc),doc.psd.as_deref(),Version::Psb).is_err());
    }

    #[test]
    fn all_blend_keys_roundtrip_and_unknown_is_rejected() {
        for mode in BlendMode::ALL {assert_eq!(blend_from_key(blend_key(mode)).unwrap(),mode);}
        assert!(blend_from_key(*b"zzzz").is_err());
    }
    #[test]
    fn edited_group_blend_updates_both_records_and_preserves_divider_tail() {
        let mut doc=document();
        let mut group=Layer::new(50,"Group",TileGrid::new(5,3));
        group.kind=LayerKind::Group{pass_through:false,collapsed:false};
        doc.layers[1].parent=Some(50);doc.layers.push(group);
        let mut raw=tree(&doc,Version::Psd);
        let record=raw.layer_mask.layers.last_mut().unwrap();
        let entry=record.additional_info.iter_mut().find(|entry|entry.key==*b"lsct").unwrap();
        let mut divider=super::super::masks::SectionDivider::parse(&entry.data).unwrap();
        divider.subtype=Some(0);divider.tail=RawBytes::from_vec(vec![17,255,93]);
        entry.data=RawBytes::from_vec(divider.encode().unwrap());
        let mut read=from_tree(raw).unwrap();
        read.layers.last_mut().unwrap().blend=BlendMode::Multiply;
        let saved=tree(&read,Version::Psd);
        let record=saved.layer_mask.layers.last().unwrap();
        assert_eq!(record.blend_mode,*b"mul ");
        let divider=super::super::masks::SectionDivider::parse(&record.additional_info.iter().find(|entry|entry.key==*b"lsct").unwrap().data).unwrap();
        assert_eq!(divider.blend_mode,Some(*b"mul "));
        assert_eq!(divider.tail.as_slice(),&[17,255,93]);
        assert_eq!(from_tree(saved).unwrap().layers.last().unwrap().blend,BlendMode::Multiply);
    }

    #[test]
    fn cropped_pixel_edit_keeps_relative_mask_location_and_rejects_extra_channel_resize() {
        let doc=Document::from_layers(16,16,vec![Layer::new(7,"Cropped",TileGrid::new(16,16))],0);
        let mut raw=tree(&doc,Version::Psd);
        let record=&mut raw.layer_mask.layers[0];
        record.bounds=[10,10,12,12];record.channels.clear();
        for (id,values) in [(0,vec![10;4]),(1,vec![20;4]),(2,vec![30;4]),(-1,vec![255;4]),(-2,vec![0,80,160,255])] {
            record.channels.push(ChannelData{id,encoded:RawBytes::from_vec(compression::encode(&values,2,2,8,Version::Psd,compression::Method::Zip,&limits()).unwrap())});
        }
        let mask=super::super::masks::MaskMetadata {
            user:super::super::masks::RasterMask{bounds:[0,0,2,2],default:255,flags:1},
            parameters:None,real:None,tail:RawBytes::from_vec(vec![0,0]),
        };
        record.mask=RawBytes::from_vec(mask.encode().unwrap());
        let original_mask_channel=record.channels.last().unwrap().clone();
        let mut read=from_tree(raw.clone()).unwrap();
        read.layers[0].replace_grid(grid_from(16,16,|x,y|if (10..12).contains(&x)&&(10..12).contains(&y) {px::pack(60,90,120,255)} else {0}));
        let saved=tree(&read,Version::Psd);
        assert_eq!(saved.layer_mask.layers[0].channels.last(),Some(&original_mask_channel));
        let after=from_tree(saved).unwrap();
        let mut values=[0;4];after.layers[0].mask.as_ref().unwrap().grid.read_region(10,10,2,2,&mut values,false);
        assert_eq!(values,[0,80,160,255]);
        let extra=ChannelData{id:4,encoded:RawBytes::from_vec(compression::encode(&[11,22,33,44],2,2,8,Version::Psd,compression::Method::Zip,&limits()).unwrap())};
        raw.layer_mask.layers[0].channels.push(extra);
        let mut read=from_tree(raw).unwrap();read.layers[0].replace_grid(TileGrid::filled(16,16,px::pack(20,30,40,255)));
        let error=to_tree(&read.layers,16,16,0,&extras(&read),read.psd.as_deref(),Version::Psd).unwrap_err();
        assert!(error.to_string().contains("extra channels"));
    }
    fn document_with_extended_mask() -> (PsdDocument,Vec<u8>) {
        let mut raw=tree(&document(),Version::Psd);
        let metadata=super::super::masks::MaskMetadata {
            user:super::super::masks::RasterMask{bounds:[-1,-2,4,6],default:173,flags:1|4|16},
            parameters:Some(super::super::masks::MaskParameters{user_density:Some(255),user_feather_bits:Some((-0.0f64).to_bits()),..Default::default()}),
            real:None,tail:RawBytes::from_vec(vec![0xab,0xcd]),
        };
        let bytes=metadata.encode().unwrap();
        let record=&mut raw.layer_mask.layers[1];
        record.mask=RawBytes::from_vec(bytes.clone());
        record.channels.push(ChannelData{id:-2,encoded:RawBytes::from_vec(compression::encode(&(0..40).collect::<Vec<u8>>(),8,5,8,Version::Psd,compression::Method::Zip,&limits()).unwrap())});
        (raw,bytes)
    }

    #[test]
    fn toggling_an_extended_mask_preserves_samples_parameters_and_opaque_tail() {
        let (raw,mut expected_mask)=document_with_extended_mask();
        let original_channels=raw.layer_mask.layers[1].channels.clone();
        let mut read=from_tree(raw).unwrap();
        read.layers[1].mask.as_mut().unwrap().enabled=false;
        let saved=tree(&read,Version::Psd);
        expected_mask[17]|=2;
        assert_eq!(saved.layer_mask.layers[1].mask.as_slice(),expected_mask);
        assert_eq!(saved.layer_mask.layers[1].channels,original_channels);
        assert!(!from_tree(saved).unwrap().layers[1].mask.as_ref().unwrap().enabled);
    }

    #[test]
    fn extended_mask_pixel_edits_fail_but_explicit_deletion_is_allowed() {
        let (raw,_)=document_with_extended_mask();
        let mut read=from_tree(raw).unwrap();
        read.layers[1].mask.as_mut().unwrap().replace(MaskGrid::from_gray(&TileGrid::filled(5,3,px::pack(42,42,42,255))));
        let error=to_tree(&read.layers,5,3,read.active,&extras(&read),read.psd.as_deref(),Version::Psd).unwrap_err();
        assert!(error.to_string().contains("off-canvas mask samples"));
        read.layers[1].mask=None;
        let saved=tree(&read,Version::Psd);
        assert!(saved.layer_mask.layers[1].mask.is_empty());
        assert!(!saved.layer_mask.layers[1].channels.iter().any(|c|c.id==-2));
    }

    #[test]
    fn in_canvas_mask_pixel_edits_keep_effective_outside_colour_and_metadata_tail() {
        let (mut raw,_)=document_with_extended_mask();
        let record=&mut raw.layer_mask.layers[1];
        let mut metadata=super::super::masks::MaskMetadata::parse(&record.mask,false).unwrap().unwrap();
        metadata.user.bounds=[0,0,3,5];
        record.mask=RawBytes::from_vec(metadata.encode().unwrap());
        record.channels.iter_mut().find(|c|c.id==-2).unwrap().encoded=RawBytes::from_vec(compression::encode(&[17;15],5,3,8,Version::Psd,compression::Method::Zip,&limits()).unwrap());
        let mut read=from_tree(raw).unwrap();
        read.layers[1].mask.as_mut().unwrap().replace(MaskGrid::from_gray(&TileGrid::filled(5,3,px::pack(42,42,42,255))));
        let saved=tree(&read,Version::Psd);
        let metadata=super::super::masks::MaskMetadata::parse(&saved.layer_mask.layers[1].mask,false).unwrap().unwrap();
        assert_eq!(metadata.user.default,82);
        assert_eq!(metadata.user.flags,16);
        assert_eq!(metadata.tail.as_slice(),[0xab,0xcd]);
        let reopened=from_tree(saved).unwrap();
        let mut sample=[0];reopened.layers[1].mask.as_ref().unwrap().grid.read_region(2,1,1,1,&mut sample,false);
        assert_eq!(sample,[42]);
    }

    #[test]
    fn pass_through_group_blend_edits_are_rejected_before_being_lost() {
        let mut original=document();
        let mut group=Layer::new(50,"Pass Through",TileGrid::new(5,3));
        group.kind=LayerKind::Group{pass_through:true,collapsed:false};
        original.layers[1].parent=Some(50);original.layers.push(group);
        let mut imported=from_tree(tree(&original,Version::Psd)).unwrap();
        for document in [&mut original,&mut imported] {
            document.layers.last_mut().unwrap().blend=BlendMode::Multiply;
            let error=to_tree(&document.layers,5,3,document.active,&extras(document),document.psd.as_deref(),Version::Psd).unwrap_err();
            assert!(error.to_string().contains("pass-through group cannot save a separate blend mode"));
        }
    }

    #[test]
    fn composite_edit_keeps_extra_alpha_and_spot_samples_exact() {
        for merged_alpha in [false,true] {
            let mut raw=tree(&document(),Version::Psd);
            let mut planes=raw.decode_composite(&limits()).unwrap();
            raw.header.channels=6;raw.layer_mask.merged_alpha=merged_alpha;
            let n=15;planes.extend((0..n*2).map(|i| (i*7) as u8));
            if !merged_alpha {planes[3*n..4*n].fill(161);}
            raw.image_data=RawBytes::from_vec(compression::encode(&planes,5,18,8,Version::Psd,compression::Method::Zip,&limits()).unwrap());
            let mut doc=from_tree(raw).unwrap();
            doc.layers[1].replace_grid(grid_from(5,3,|_,_|px::pack(200,44,61,255)));
            let saved=tree(&doc,Version::Psd).decode_composite(&limits()).unwrap();
            let extra_start=if merged_alpha {4*n} else {3*n};
            assert_eq!(&saved[extra_start..],&planes[extra_start..]);
        }
    }

    #[test]
    fn channel_panel_prefers_unicode_names() {
        let mut raw=tree(&document(),Version::Psd);
        let mut bytes=Vec::new();let name="Mask α";let words=name.encode_utf16().collect::<Vec<_>>();
        bytes.extend_from_slice(&(words.len() as u32).to_be_bytes());
        for word in words {bytes.extend_from_slice(&word.to_be_bytes());}
        raw.resources.push(ImageResource::new(1045,bytes));
        let doc=from_tree(raw).unwrap();
        assert_eq!(channel_labels(doc.psd.as_deref()),vec!["Red","Green","Blue","Mask α"]);
    }

}

#[cfg(test)]
mod advanced_mapping_tests {
    use super::*;
    use crate::engine::advanced_blend::{AdvancedBlend,Range};
    fn encode(document:&Document,version:Version)->Result<PsdDocument> {
        to_tree(&document.layers,document.w,document.h,document.active,&DocExtras{selected:document.selected.clone(),..DocExtras::default()},document.psd.as_deref(),version)
    }
    #[test]
    fn advanced_settings_roundtrip_preserve_unknowns_and_can_be_cleared() {
        for version in [Version::Psd,Version::Psb] {
            let settings=Arc::new(AdvancedBlend{ranges:vec![Range{source:[10,20,230,240],destination:[0,0,255,255]}],restricted_channels:vec![2,0,2]});
            let mut layer=Layer::new(19,"Advanced",TileGrid::filled(2,1,px::pack(40,80,120,255)));
            layer.advanced_blend=Some(settings.clone());
            let document=Document::from_layers(2,1,vec![layer],0);
            let mut raw=encode(&document,version).unwrap();
            let unknown=AdditionalInfo::new(*b"zzzz",vec![0,17,9,255]);
            raw.layer_mask.layers[0].additional_info.push(unknown.clone());
            let original=raw.encode(&limits()).unwrap();
            let mut document=from_tree(PsdDocument::parse(original.clone(),&limits()).unwrap()).unwrap();
            assert_eq!(document.layers[0].advanced_blend.as_deref(),Some(settings.as_ref()));
            assert_eq!(encode(&document,version).unwrap().encode(&limits()).unwrap(),original);
            document.layers[0].name="Changed name".into();
            document.layers[0].advanced_blend=None;
            let saved=encode(&document,version).unwrap();
            assert!(saved.layer_mask.layers[0].blending_ranges.is_empty());
            assert_eq!(block(&saved.layer_mask.layers[0],b"brst"),Some([].as_slice()));
            assert!(saved.layer_mask.layers[0].additional_info.contains(&unknown));
            let reopened=from_tree(saved).unwrap();
            assert_eq!(reopened.layers[0].name,"Changed name");
            assert!(reopened.layers[0].advanced_blend.is_none());
        }
    }
    #[test]
    fn pass_through_advanced_semantics_are_rejected_explicitly() {
        let mut layer=Layer::new(1,"Group",TileGrid::new(1,1));
        layer.kind=LayerKind::Group{pass_through:true,collapsed:false};
        layer.advanced_blend=Some(Arc::new(AdvancedBlend{ranges:vec![],restricted_channels:vec![0]}));
        let document=Document::from_layers(1,1,vec![layer],0);
        assert!(encode(&document,Version::Psd).unwrap_err().to_string().contains("backdrop mapping"));
    }
}
