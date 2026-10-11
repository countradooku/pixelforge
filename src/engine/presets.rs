//! Brush presets (Window ▸ Brushes): folders of named brush settings with
//! their sampled tips, `.abr` import, and persistence in the user config
//! directory (`brushes.pfb`).
//!
//! File format: `PFBRUSH1`, then the tips (id u64, name, w, h, LZ4 alpha),
//! then the folders (name, open flag, presets). Each preset's settings are
//! a length-prefixed field list read in order, so later versions can append
//! fields: readers skip what they don't know and default what's missing.

use std::path::{Path, PathBuf};

use super::brush::{self, BrushSettings, ColorDynamics, Control, DualBrush, Scattering, ShapeDynamics, TexMode, TextureOpts, TipImage, Transfer};

#[derive(Clone, Debug, PartialEq)]
pub struct BrushPreset {
    pub name: String,
    pub settings: BrushSettings,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PresetFolder {
    pub name: String,
    pub open: bool,
    pub presets: Vec<BrushPreset>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct PresetLib {
    pub folders: Vec<PresetFolder>,
}

/// What an `.abr` import added and skipped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbrReport {
    pub added: usize,
    pub skipped_computed: usize,
    pub skipped_other: usize,
}

/// A procedural leaf-like tip for the default "Special Effects" folder.
fn leaf_tip() -> u64 {
    let (w, h) = (48u32, 64u32);
    let alpha = (0..w * h)
        .map(|k| {
            let (x, y) = ((k % w) as f32 / w as f32 * 2.0 - 1.0, (k / w) as f32 / h as f32 * 2.0 - 1.0);
            let width = (1.0 - y * y).max(0.0).sqrt() * 0.75;
            let vein = (x.abs() < 0.04) as u8 as f32;
            if x.abs() < width { ((1.0 - vein * 0.6) * 255.0) as u8 } else { 0 }
        })
        .collect();
    brush::register_tip(TipImage { name: "Leaf".into(), w, h, alpha })
}

impl PresetLib {
    /// The presets a fresh install starts with.
    pub fn defaults() -> PresetLib {
        let b = BrushSettings::default();
        let p = |name: &str, settings: BrushSettings| BrushPreset { name: name.into(), settings };
        PresetLib {
            folders: vec![
                PresetFolder {
                    name: "General Brushes".into(),
                    open: true,
                    presets: vec![
                        p("Soft Round", BrushSettings { size: 45.0, hardness: 0.0, ..b }),
                        p("Hard Round", BrushSettings { size: 30.0, hardness: 1.0, ..b }),
                        p("Soft Round Pressure Opacity", BrushSettings { size: 60.0, hardness: 0.0, pressure_size: false, pressure_opacity: true, ..b }),
                        p("Hard Round Pressure Size", BrushSettings { size: 30.0, hardness: 1.0, pressure_size: true, ..b }),
                        p("Airbrush Soft", BrushSettings { size: 100.0, hardness: 0.0, flow: 0.1, build_up: true, ..b }),
                    ],
                },
                PresetFolder {
                    name: "Dry Media".into(),
                    open: true,
                    presets: vec![
                        p("Chalk", BrushSettings { size: 60.0, hardness: 0.6, texture: 1, texture_depth: 0.9, tex: TextureOpts { mode: TexMode::Subtract, each_tip: true, depth_jitter: 0.3, ..Default::default() }, noise: true, ..b }),
                        p("Pencil Grain", BrushSettings { size: 8.0, hardness: 0.9, texture: 2, texture_depth: 0.6, shape: ShapeDynamics { on: true, size_ctl: Control::Pressure, ..Default::default() }, ..b }),
                    ],
                },
                PresetFolder {
                    name: "Special Effects".into(),
                    open: false,
                    presets: vec![
                        p("Spatter", BrushSettings { size: 40.0, hardness: 1.0, spacing: 0.5, scatter: 1.5, shape: ShapeDynamics { on: true, size_jitter: 0.8, ..Default::default() }, scattering: Scattering { both_axes: true, count: 4, count_jitter: 0.5, ..Default::default() }, ..b }),
                        p(
                            "Scattered Leaves",
                            BrushSettings {
                                size: 70.0,
                                tip: leaf_tip(),
                                spacing: 0.6,
                                scatter: 1.2,
                                shape: ShapeDynamics { on: true, size_jitter: 0.6, angle_jitter: 1.0, round_jitter: 0.3, min_round: 0.3, flip_x_jitter: true, ..Default::default() },
                                scattering: Scattering { both_axes: true, count: 2, ..Default::default() },
                                color: ColorDynamics { on: true, fg_bg_jitter: 1.0, hue_jitter: 0.08, bri_jitter: 0.2, bg: [200, 150, 30], ..Default::default() },
                                ..b
                            },
                        ),
                        p("Watercolor Wash", BrushSettings { size: 90.0, hardness: 0.2, flow: 0.5, wet_edges: true, transfer: Transfer { on: true, flow_jitter: 0.4, ..Default::default() }, color: ColorDynamics { on: true, hue_jitter: 0.04, sat_jitter: 0.1, ..Default::default() }, ..b }),
                        p("Dual Sponge", BrushSettings { size: 80.0, hardness: 0.8, dual: DualBrush { on: true, size: 20.0, spacing: 0.4, scatter: 1.0, both_axes: true, count: 3, ..Default::default() }, ..b }),
                    ],
                },
            ],
        }
    }

    pub fn path() -> PathBuf {
        super::prefs::config_dir().join("brushes.pfb")
    }

    /// The saved library, or the defaults.
    pub fn load_or_default() -> PresetLib {
        std::fs::read(Self::path()).ok().and_then(|b| PresetLib::decode(&b).ok()).unwrap_or_else(PresetLib::defaults)
    }

    pub fn save(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(super::prefs::config_dir())?;
        let tmp = Self::path().with_extension("tmp");
        std::fs::write(&tmp, self.encode())?;
        std::fs::rename(tmp, Self::path())
    }

    /// Imports the sampled tips of an `.abr` file as a new folder.
    pub fn import_abr(&mut self, bytes: &[u8], folder: &str) -> Result<AbrReport, String> {
        let f = super::abr::parse(bytes)?;
        let mut presets = Vec::new();
        for t in f.tips {
            let id = brush::register_tip(TipImage { name: t.name.clone(), w: t.w, h: t.h, alpha: t.alpha });
            let settings = BrushSettings { size: t.diameter.max(1.0), spacing: t.spacing.max(0.01), angle: t.angle, roundness: t.roundness.clamp(0.01, 1.0), tip: id, pressure_size: false, ..BrushSettings::default() };
            presets.push(BrushPreset { name: t.name, settings });
        }
        let report = AbrReport { added: presets.len(), skipped_computed: f.skipped_computed, skipped_other: f.skipped_other };
        self.folders.push(PresetFolder { name: folder.to_string(), open: true, presets });
        Ok(report)
    }

    pub fn import_abr_file(&mut self, path: &Path) -> Result<AbrReport, String> {
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let name = path.file_stem().map_or_else(|| "Imported".to_string(), |s| s.to_string_lossy().into_owned());
        self.import_abr(&bytes, &name)
    }

    /// Moves preset `(f, i)` one place up (`-1`) or down (`1`), across
    /// folder boundaries at the ends; returns its new position.
    pub fn move_preset(&mut self, f: usize, i: usize, by: i32) -> (usize, usize) {
        let n = self.folders[f].presets.len();
        let j = i as i32 + by;
        if j >= 0 && (j as usize) < n {
            self.folders[f].presets.swap(i, j as usize);
            return (f, j as usize);
        }
        let g = f as i32 + by;
        if g < 0 || g as usize >= self.folders.len() {
            return (f, i);
        }
        let p = self.folders[f].presets.remove(i);
        let g = g as usize;
        if by < 0 {
            self.folders[g].presets.push(p);
            (g, self.folders[g].presets.len() - 1)
        } else {
            self.folders[g].presets.insert(0, p);
            (g, 0)
        }
    }

    // ----- encoding ----------------------------------------------------------------------

    pub fn encode(&self) -> Vec<u8> {
        let mut w = Vec::new();
        w.extend_from_slice(b"PFBRUSH1");
        let mut ids: Vec<u64> = self.folders.iter().flat_map(|f| f.presets.iter()).flat_map(|p| [p.settings.tip, p.settings.dual.tip]).filter(|&id| id != 0).collect();
        ids.sort_unstable();
        ids.dedup();
        let tips: Vec<(u64, std::sync::Arc<TipImage>)> = ids.into_iter().filter_map(|id| brush::tip(id).map(|t| (id, t))).collect();
        put_u32(&mut w, tips.len() as u32);
        for (id, t) in &tips {
            w.extend_from_slice(&id.to_le_bytes());
            put_str(&mut w, &t.name);
            put_u32(&mut w, t.w);
            put_u32(&mut w, t.h);
            let z = lz4_flex::compress_prepend_size(&t.alpha);
            put_u32(&mut w, z.len() as u32);
            w.extend_from_slice(&z);
        }
        put_u32(&mut w, self.folders.len() as u32);
        for f in &self.folders {
            put_str(&mut w, &f.name);
            w.push(f.open as u8);
            put_u32(&mut w, f.presets.len() as u32);
            for p in &f.presets {
                put_str(&mut w, &p.name);
                let s = encode_settings(&p.settings);
                put_u32(&mut w, s.len() as u32);
                w.extend_from_slice(&s);
            }
        }
        w
    }

    pub fn decode(b: &[u8]) -> Result<PresetLib, String> {
        let mut r = Rd { b, i: 0 };
        if r.take(8)? != b"PFBRUSH1" {
            return Err("not a PixelForge brush file".into());
        }
        // Tips register under their content hash; remap ids in case the
        // hash function ever changes.
        let mut remap = std::collections::HashMap::new();
        for _ in 0..r.u32()? {
            let id = u64::from_le_bytes(r.take(8)?.try_into().unwrap());
            let name = r.str()?;
            let (tw, th) = (r.u32()?, r.u32()?);
            let n = r.u32()? as usize;
            let alpha = lz4_flex::decompress_size_prepended(r.take(n)?).map_err(|e| e.to_string())?;
            if alpha.len() != (tw * th) as usize {
                return Err("bad tip size".into());
            }
            remap.insert(id, brush::register_tip(TipImage { name, w: tw, h: th, alpha }));
        }
        let mut lib = PresetLib::default();
        for _ in 0..r.u32()? {
            let name = r.str()?;
            let open = r.u8()? != 0;
            let mut presets = Vec::new();
            for _ in 0..r.u32()? {
                let pname = r.str()?;
                let n = r.u32()? as usize;
                let mut settings = decode_settings(r.take(n)?);
                settings.tip = remap.get(&settings.tip).copied().unwrap_or(0);
                settings.dual.tip = remap.get(&settings.dual.tip).copied().unwrap_or(0);
                presets.push(BrushPreset { name: pname, settings });
            }
            lib.folders.push(PresetFolder { name, open, presets });
        }
        Ok(lib)
    }
}

fn put_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

fn put_str(w: &mut Vec<u8>, s: &str) {
    put_u32(w, s.len() as u32);
    w.extend_from_slice(s.as_bytes());
}

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.i..self.i + n).ok_or("truncated brush file")?;
        self.i += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String, String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
}

/// Field writer / reader for one preset's settings: a flat list of f32 /
/// u8 / u64 values in a fixed order; a reader past the end yields defaults.
struct Fw(Vec<u8>);

impl Fw {
    fn f(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn b(&mut self, v: u8) {
        self.0.push(v);
    }
    fn q(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn ctl(&mut self, c: Control) {
        let (t, n) = match c {
            Control::Off => (0, 0),
            Control::Pressure => (1, 0),
            Control::Tilt => (2, 0),
            Control::Fade(n) => (3, n),
            Control::Direction => (4, 0),
            Control::Initial => (5, 0),
        };
        self.b(t);
        self.0.extend_from_slice(&n.to_le_bytes());
    }
    fn mode(&mut self, m: TexMode) {
        self.b(TexMode::ALL.iter().position(|&x| x == m).unwrap_or(0) as u8);
    }
}

struct Fr<'a> {
    b: &'a [u8],
    i: usize,
}

impl Fr<'_> {
    fn raw<const N: usize>(&mut self) -> Option<[u8; N]> {
        let s = self.b.get(self.i..self.i + N)?;
        self.i += N;
        Some(s.try_into().unwrap())
    }
    fn f(&mut self, d: f32) -> f32 {
        self.raw::<4>().map_or(d, f32::from_le_bytes)
    }
    fn b(&mut self, d: u8) -> u8 {
        self.raw::<1>().map_or(d, |v| v[0])
    }
    fn bool(&mut self, d: bool) -> bool {
        self.b(d as u8) != 0
    }
    fn q(&mut self, d: u64) -> u64 {
        self.raw::<8>().map_or(d, u64::from_le_bytes)
    }
    fn ctl(&mut self, d: Control) -> Control {
        let Some(t) = self.raw::<1>() else { return d };
        let n = self.raw::<2>().map_or(0, u16::from_le_bytes);
        match t[0] {
            1 => Control::Pressure,
            2 => Control::Tilt,
            3 => Control::Fade(n),
            4 => Control::Direction,
            5 => Control::Initial,
            _ => Control::Off,
        }
    }
    fn mode(&mut self, d: TexMode) -> TexMode {
        self.raw::<1>().and_then(|v| TexMode::ALL.get(v[0] as usize).copied()).unwrap_or(d)
    }
}

fn encode_settings(s: &BrushSettings) -> Vec<u8> {
    let mut w = Fw(Vec::new());
    for v in [s.size, s.hardness, s.opacity, s.flow, s.spacing, s.angle, s.roundness, s.scatter] {
        w.f(v);
    }
    w.b(s.texture);
    w.f(s.texture_depth);
    w.f(s.smoothing);
    for v in [s.pressure_size, s.pressure_opacity, s.pressure_flow, s.aliased] {
        w.b(v as u8);
    }
    w.q(s.tip);
    w.b(s.flip_x as u8);
    w.b(s.flip_y as u8);
    let h = &s.shape;
    w.b(h.on as u8);
    w.f(h.size_jitter);
    w.ctl(h.size_ctl);
    w.f(h.min_size);
    w.f(h.angle_jitter);
    w.ctl(h.angle_ctl);
    w.f(h.round_jitter);
    w.ctl(h.round_ctl);
    w.f(h.min_round);
    w.b(h.flip_x_jitter as u8);
    w.b(h.flip_y_jitter as u8);
    let c = &s.scattering;
    w.b(c.both_axes as u8);
    w.ctl(c.ctl);
    w.b(c.count);
    w.f(c.count_jitter);
    w.ctl(c.count_ctl);
    let t = &s.tex;
    w.f(t.scale);
    w.mode(t.mode);
    w.b(t.invert as u8);
    w.f(t.brightness);
    w.f(t.contrast);
    w.b(t.each_tip as u8);
    w.f(t.min_depth);
    w.f(t.depth_jitter);
    w.ctl(t.depth_ctl);
    let d = &s.dual;
    w.b(d.on as u8);
    w.q(d.tip);
    w.f(d.size);
    w.f(d.spacing);
    w.f(d.scatter);
    w.b(d.both_axes as u8);
    w.b(d.count);
    w.mode(d.mode);
    w.b(d.flip as u8);
    let k = &s.color;
    w.b(k.on as u8);
    w.b(k.per_tip as u8);
    w.f(k.fg_bg_jitter);
    w.ctl(k.fg_bg_ctl);
    w.f(k.hue_jitter);
    w.f(k.sat_jitter);
    w.f(k.bri_jitter);
    w.f(k.purity);
    k.bg.iter().for_each(|&v| w.b(v));
    let x = &s.transfer;
    w.b(x.on as u8);
    w.f(x.opacity_jitter);
    w.ctl(x.opacity_ctl);
    w.f(x.flow_jitter);
    w.ctl(x.flow_ctl);
    for v in [s.wet_edges, s.build_up, s.noise, s.protect_texture] {
        w.b(v as u8);
    }
    w.0.extend_from_slice(&s.seed.to_le_bytes());
    w.0
}

fn decode_settings(b: &[u8]) -> BrushSettings {
    let d = BrushSettings::default();
    let mut r = Fr { b, i: 0 };
    let mut s = d;
    s.size = r.f(d.size);
    s.hardness = r.f(d.hardness);
    s.opacity = r.f(d.opacity);
    s.flow = r.f(d.flow);
    s.spacing = r.f(d.spacing);
    s.angle = r.f(d.angle);
    s.roundness = r.f(d.roundness);
    s.scatter = r.f(d.scatter);
    s.texture = r.b(d.texture);
    s.texture_depth = r.f(d.texture_depth);
    s.smoothing = r.f(d.smoothing);
    s.pressure_size = r.bool(d.pressure_size);
    s.pressure_opacity = r.bool(d.pressure_opacity);
    s.pressure_flow = r.bool(d.pressure_flow);
    s.aliased = r.bool(d.aliased);
    s.tip = r.q(0);
    s.flip_x = r.bool(false);
    s.flip_y = r.bool(false);
    let h = &mut s.shape;
    h.on = r.bool(false);
    h.size_jitter = r.f(0.0);
    h.size_ctl = r.ctl(Control::Off);
    h.min_size = r.f(0.0);
    h.angle_jitter = r.f(0.0);
    h.angle_ctl = r.ctl(Control::Off);
    h.round_jitter = r.f(0.0);
    h.round_ctl = r.ctl(Control::Off);
    h.min_round = r.f(0.0);
    h.flip_x_jitter = r.bool(false);
    h.flip_y_jitter = r.bool(false);
    let c = &mut s.scattering;
    c.both_axes = r.bool(false);
    c.ctl = r.ctl(Control::Off);
    c.count = r.b(1);
    c.count_jitter = r.f(0.0);
    c.count_ctl = r.ctl(Control::Off);
    let t = &mut s.tex;
    t.scale = r.f(1.0);
    t.mode = r.mode(TexMode::Multiply);
    t.invert = r.bool(false);
    t.brightness = r.f(0.0);
    t.contrast = r.f(0.0);
    t.each_tip = r.bool(false);
    t.min_depth = r.f(0.0);
    t.depth_jitter = r.f(0.0);
    t.depth_ctl = r.ctl(Control::Off);
    let du = &mut s.dual;
    let dd = DualBrush::default();
    du.on = r.bool(false);
    du.tip = r.q(0);
    du.size = r.f(dd.size);
    du.spacing = r.f(dd.spacing);
    du.scatter = r.f(0.0);
    du.both_axes = r.bool(false);
    du.count = r.b(1);
    du.mode = r.mode(TexMode::Multiply);
    du.flip = r.bool(false);
    let k = &mut s.color;
    let kd = ColorDynamics::default();
    k.on = r.bool(false);
    k.per_tip = r.bool(true);
    k.fg_bg_jitter = r.f(0.0);
    k.fg_bg_ctl = r.ctl(Control::Off);
    k.hue_jitter = r.f(0.0);
    k.sat_jitter = r.f(0.0);
    k.bri_jitter = r.f(0.0);
    k.purity = r.f(0.0);
    k.bg = [r.b(kd.bg[0]), r.b(kd.bg[1]), r.b(kd.bg[2])];
    let x = &mut s.transfer;
    x.on = r.bool(false);
    x.opacity_jitter = r.f(0.0);
    x.opacity_ctl = r.ctl(Control::Off);
    x.flow_jitter = r.f(0.0);
    x.flow_ctl = r.ctl(Control::Off);
    s.wet_edges = r.bool(false);
    s.build_up = r.bool(false);
    s.noise = r.bool(false);
    s.protect_texture = r.bool(false);
    s.seed = r.raw::<4>().map_or(d.seed, u32::from_le_bytes);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_round_trips_with_tips() {
        let mut lib = PresetLib::defaults();
        lib.folders[0].presets.push(BrushPreset { name: "All".into(), settings: brush::all_dynamics(80.0, 3) });
        lib.folders[1].open = false;
        let back = PresetLib::decode(&lib.encode()).unwrap();
        assert_eq!(back, lib);
        // The leaf tip travelled with the file.
        let leaf = &back.folders[2].presets[1].settings;
        assert!(brush::tip(leaf.tip).is_some_and(|t| t.w == 48));
        // Old (shorter) settings blobs default the missing fields.
        let short = &encode_settings(&BrushSettings { size: 77.0, ..Default::default() })[..20];
        let s = decode_settings(short);
        assert_eq!((s.size, s.shape, s.seed), (77.0, ShapeDynamics::default(), BrushSettings::default().seed));
        assert!(PresetLib::decode(b"garbage").is_err());
    }

    #[test]
    fn reorder_rename_delete() {
        let mut lib = PresetLib::defaults();
        let first = lib.folders[0].presets[0].name.clone();
        assert_eq!(lib.move_preset(0, 0, 1), (0, 1));
        assert_eq!(lib.folders[0].presets[1].name, first);
        // Past the end of a folder it moves into the next one.
        let n = lib.folders[0].presets.len();
        assert_eq!(lib.move_preset(0, n - 1, 1), (1, 0));
        assert_eq!(lib.move_preset(1, 0, -1), (0, n - 1));
        lib.folders[0].presets[0].name = "Renamed".into();
        lib.folders[0].presets.remove(1);
        assert_eq!(lib.folders[0].presets[0].name, "Renamed");
    }

    #[test]
    fn imports_real_abr_files_when_present() {
        let path = Path::new("/Applications/Adobe Photoshop 2026/Presets/Brushes/Legacy Brushes.abr");
        let Ok(bytes) = std::fs::read(path) else { return };
        let mut lib = PresetLib::default();
        let r = lib.import_abr(&bytes, "Legacy Brushes").unwrap();
        assert!(r.added > 50, "{r:?}");
        let p = &lib.folders[0].presets[0];
        let t = brush::tip(p.settings.tip).unwrap();
        assert!(t.w > 0 && t.alpha.iter().any(|&a| a > 0));
        // An imported tip paints.
        let mut g = super::super::grid::TileGrid::new(400, 400);
        let mut st = brush::Stroke::brush(0, &g, p.settings, [0, 0, 0], false, None);
        st.add_point(200.0, 200.0, &mut g);
        assert!((150..250).any(|x| super::super::px::alpha(g.pixel(x, 200)) > 0));
    }
}
