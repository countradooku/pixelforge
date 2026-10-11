//! Measurement Log (Image ▸ Analysis): records from the Ruler (and its
//! protractor), Count and the selection — area, perimeter, circularity,
//! bounds, centroid and mean gray — in a measurement scale (pixels → units),
//! exported as CSV with Photoshop's column names.

use super::grid::RectI;
use super::selection::MaskGrid;

/// `px` pixels = `length` `units` (Image ▸ Analysis ▸ Set Measurement Scale).
#[derive(Clone, Debug, PartialEq)]
pub struct MeasureScale {
    pub px: f32,
    pub length: f32,
    pub units: String,
}

impl Default for MeasureScale {
    fn default() -> Self {
        MeasureScale { px: 1.0, length: 1.0, units: "pixels".into() }
    }
}

impl MeasureScale {
    /// Units per pixel.
    pub fn factor(&self) -> f64 {
        self.length as f64 / self.px.max(1e-6) as f64
    }

    pub fn label(&self) -> String {
        format!("{} pixels = {} {}", self.px, self.length, self.units)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Ruler,
    Protractor,
    Count,
    Selection,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::Ruler => "Ruler Tool",
            Source::Protractor => "Protractor",
            Source::Count => "Count Tool",
            Source::Selection => "Selection",
        }
    }

    fn code(self) -> u8 {
        self as u8
    }

    fn from_code(c: u8) -> Source {
        match c {
            1 => Source::Protractor,
            2 => Source::Count,
            3 => Source::Selection,
            _ => Source::Ruler,
        }
    }
}

/// One row of the log; values are in scale units (areas in units²).
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub label: String,
    pub when: String,
    pub document: String,
    pub source: Source,
    pub scale: MeasureScale,
    pub count: Option<u32>,
    pub area: Option<f64>,
    pub perimeter: Option<f64>,
    pub circularity: Option<f64>,
    pub width: Option<f64>,
    pub height: Option<f64>,
    pub cx: Option<f64>,
    pub cy: Option<f64>,
    pub gray: Option<f64>,
    pub length: Option<f64>,
    pub angle: Option<f64>,
}

impl Record {
    fn new(source: Source, scale: &MeasureScale, document: &str) -> Record {
        Record {
            label: String::new(),
            when: now_text(),
            document: document.into(),
            source,
            scale: scale.clone(),
            count: None,
            area: None,
            perimeter: None,
            circularity: None,
            width: None,
            height: None,
            cx: None,
            cy: None,
            gray: None,
            length: None,
            angle: None,
        }
    }
}

/// The document's log and scale (saved in `.pforge`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Measurements {
    pub scale: MeasureScale,
    pub records: Vec<Record>,
}

impl Measurements {
    /// Appends `r` labelled "Measurement N" and returns its index.
    pub fn push(&mut self, mut r: Record) -> usize {
        if r.label.is_empty() {
            r.label = format!("Measurement {}", self.records.len() + 1);
        }
        self.records.push(r);
        self.records.len() - 1
    }
}

/// UTC "YYYY-MM-DD HH:MM:SS" (no time-zone database needed).
pub fn now_text() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Days → civil date (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC", rem / 3600, rem / 60 % 60, rem % 60)
}

/// Photoshop's Gray Value of a flat gray `v`, measured with Record
/// Measurements (its Gray working space, Dot Gain 20 %): nodes 0, 16, …, 255.
const PS_GRAY: [f32; 17] = [0.0, 12.31, 22.23, 33.58, 46.11, 59.66, 74.13, 89.42, 105.49, 122.25, 139.68, 157.73, 176.37, 195.57, 215.31, 235.55, 255.0];

/// A colour's Gray Value as Photoshop measures it: its L\* mapped through
/// the gray scale above (a gray of the same L\* has the same Gray Value).
pub fn ps_gray(c: [u8; 3]) -> f32 {
    static NODES: std::sync::OnceLock<[f32; 17]> = std::sync::OnceLock::new();
    let nodes = NODES.get_or_init(|| std::array::from_fn(|i| super::color::srgb8_to_lab([(i * 16).min(255) as u8; 3])[0]));
    let l = super::color::srgb8_to_lab(c)[0];
    let i = nodes.partition_point(|&n| n <= l).clamp(1, 16);
    let t = ((l - nodes[i - 1]) / (nodes[i] - nodes[i - 1]).max(1e-6)).clamp(0.0, 1.0);
    PS_GRAY[i - 1] + (PS_GRAY[i] - PS_GRAY[i - 1]) * t
}

/// Ruler: length and angle of `a → b` (angle counter-clockwise from +x, like
/// the Info panel), plus its width and height.
pub fn ruler(a: (f32, f32), b: (f32, f32), scale: &MeasureScale, doc: &str) -> Record {
    let (w, h, ang, l) = super::annot::measure(a, b);
    let k = scale.factor();
    Record { length: Some(l as f64 * k), angle: Some(ang as f64), width: Some(w.abs() as f64 * k), height: Some(h.abs() as f64 * k), ..Record::new(Source::Ruler, scale, doc) }
}

/// The angle between the protractor's legs `v → a` and `v → b` (0..180°).
pub fn protractor_angle(v: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (ax, ay) = (a.0 - v.0, a.1 - v.1);
    let (bx, by) = (b.0 - v.0, b.1 - v.1);
    let d = (ax * bx + ay * by) / (ax.hypot(ay) * bx.hypot(by)).max(1e-9);
    d.clamp(-1.0, 1.0).acos().to_degrees()
}

pub fn protractor(v: (f32, f32), a: (f32, f32), b: (f32, f32), scale: &MeasureScale, doc: &str) -> Record {
    let k = scale.factor();
    let l1 = (a.0 - v.0).hypot(a.1 - v.1) as f64 * k;
    Record { angle: Some(protractor_angle(v, a, b) as f64), length: Some(l1), ..Record::new(Source::Protractor, scale, doc) }
}

/// One record per visible count group.
pub fn counts(groups: &[super::annot::CountGroup], scale: &MeasureScale, doc: &str) -> Vec<Record> {
    groups
        .iter()
        .filter(|g| g.visible)
        .map(|g| Record { label: g.name.clone(), count: Some(g.pts.len() as u32), ..Record::new(Source::Count, scale, doc) })
        .collect()
}

/// Selection measurements of `mask`: area (coverage), perimeter (length of
/// the anti-aliased outline at half coverage), circularity 4πA/P², bounds,
/// centroid and the mean gray of `gray` (luminance 0..255 at a pixel) under
/// the selection.
pub fn selection(mask: &MaskGrid, gray: &(dyn Fn(i32, i32) -> f32 + Sync), scale: &MeasureScale, doc: &str) -> Option<Record> {
    use rayon::prelude::*;
    let b = super::retouch::mask_bbox(mask);
    if b.is_empty() {
        return None;
    }
    let w = b.w() as usize;
    let rows: Vec<(f64, f64, f64, f64)> = (b.y0..b.y1)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![0u8; w];
            mask.read_region(b.x0, y, w, 1, &mut row, false);
            let (mut a, mut sx, mut sy, mut g) = (0.0, 0.0, 0.0, 0.0);
            for (i, &c) in row.iter().enumerate() {
                if c == 0 {
                    continue;
                }
                let c = c as f64 / 255.0;
                let x = b.x0 + i as i32;
                a += c;
                sx += c * (x as f64 + 0.5);
                sy += c * (y as f64 + 0.5);
                g += c * gray(x, y) as f64;
            }
            (a, sx, sy, g)
        })
        .collect();
    let (a, sx, sy, g) = rows.iter().fold((0.0, 0.0, 0.0, 0.0), |t, r| (t.0 + r.0, t.1 + r.1, t.2 + r.2, t.3 + r.3));
    if a <= 0.0 {
        return None;
    }
    let per = perimeter(mask, b);
    let k = scale.factor();
    Some(Record {
        count: Some(regions(mask, b)),
        area: Some(a * k * k),
        perimeter: Some(per * k),
        circularity: Some((4.0 * std::f64::consts::PI * a / (per * per).max(1e-9)).min(1.0)),
        width: Some(b.w() as f64 * k),
        height: Some(b.h() as f64 * k),
        cx: Some(sx / a * k),
        cy: Some(sy / a * k),
        gray: Some(g / a),
        ..Record::new(Source::Selection, scale, doc)
    })
}

/// Marching-squares length of the coverage = 50 % contour inside `b`. The
/// coverage is smoothed 3 × 3 first: anti-aliased coverage isn't linear
/// across an edge, and the raw contour zigzags (≈ 1.6 % too long on a disc).
fn perimeter(mask: &MaskGrid, b: RectI) -> f64 {
    use rayon::prelude::*;
    let (x0, y0) = (b.x0 - 2, b.y0 - 2);
    let w = (b.w() + 4) as usize;
    let h = (b.h() + 4) as usize;
    let mut raw = vec![0u8; w * h];
    mask.read_region(x0, y0, w, h, &mut raw, false);
    let f = |x: usize, y: usize| raw[y * w + x] as f32;
    let field: Vec<f32> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            let (x, y) = (i % w, i / w);
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                return -0.5;
            }
            let mut s = 0.0;
            for dy in 0..3 {
                for dx in 0..3 {
                    s += f(x + dx - 1, y + dy - 1);
                }
            }
            s / (9.0 * 255.0) - 0.5
        })
        .collect();
    (0..h - 1)
        .into_par_iter()
        .map(|y| {
            let (t, u) = (&field[y * w..(y + 1) * w], &field[(y + 1) * w..(y + 2) * w]);
            let mut len = 0.0f64;
            for x in 0..w - 1 {
                // Corners: a (x, y), b (x+1, y), c (x+1, y+1), d (x, y+1).
                let v = [t[x], t[x + 1], u[x + 1], u[x]];
                let p = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
                let mut pts: Vec<(f32, f32)> = Vec::with_capacity(4);
                for e in 0..4 {
                    let (a, c) = (v[e], v[(e + 1) % 4]);
                    if (a >= 0.0) != (c >= 0.0) {
                        let s = a / (a - c);
                        let (pa, pc) = (p[e], p[(e + 1) % 4]);
                        pts.push((pa.0 + (pc.0 - pa.0) * s, pa.1 + (pc.1 - pa.1) * s));
                    }
                }
                let seg = |i: usize, j: usize| ((pts[i].0 - pts[j].0).hypot(pts[i].1 - pts[j].1)) as f64;
                len += match pts.len() {
                    2 => seg(0, 1),
                    4 => seg(0, 1) + seg(2, 3),
                    _ => 0.0,
                };
            }
            len
        })
        .sum()
}

/// Separate selected regions (4-connected, coverage ≥ 50 %).
fn regions(mask: &MaskGrid, b: RectI) -> u32 {
    let (w, h) = (b.w() as usize, b.h() as usize);
    let mut on = vec![0u8; w * h];
    mask.read_region(b.x0, b.y0, w, h, &mut on, false);
    let mut seen = vec![false; w * h];
    let mut n = 0;
    let mut stack = Vec::new();
    for s in 0..w * h {
        if seen[s] || on[s] < 128 {
            continue;
        }
        n += 1;
        stack.push(s);
        seen[s] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize| {
                if !seen[j] && on[j] >= 128 {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                visit(i - 1);
            }
            if x + 1 < w {
                visit(i + 1);
            }
            if y > 0 {
                visit(i - w);
            }
            if y + 1 < h {
                visit(i + w);
            }
        }
    }
    n
}

const COLUMNS: [&str; 18] = [
    "Label", "Date and Time", "Document", "Source", "Scale", "Scale Units", "Scale Factor", "Count", "Area", "Perimeter", "Circularity", "Height", "Width", "Centroid X", "Centroid Y", "Gray Value (Mean)", "Length", "Angle",
];

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() }
}

/// The log as CSV with Photoshop's column names.
pub fn csv(records: &[Record]) -> String {
    let mut out = COLUMNS.join(",") + "\n";
    let num = |v: Option<f64>| v.map_or(String::new(), |v| format!("{v:.6}").trim_end_matches('0').trim_end_matches('.').to_string());
    for r in records {
        let cells = [
            csv_field(&r.label),
            csv_field(&r.when),
            csv_field(&r.document),
            r.source.label().to_string(),
            csv_field(&r.scale.label()),
            csv_field(&r.scale.units),
            num(Some(r.scale.factor())),
            r.count.map_or(String::new(), |c| c.to_string()),
            num(r.area),
            num(r.perimeter),
            num(r.circularity),
            num(r.height),
            num(r.width),
            num(r.cx),
            num(r.cy),
            num(r.gray),
            num(r.length),
            num(r.angle),
        ];
        out += &cells.join(",");
        out.push('\n');
    }
    out
}

// ----- `.pforge` encoding (used by the "PFMX" chunk) ---------------------------------------

pub fn encode(m: &Measurements, out: &mut Vec<u8>) {
    let s = |out: &mut Vec<u8>, t: &str| {
        out.extend_from_slice(&(t.len() as u32).to_le_bytes());
        out.extend_from_slice(t.as_bytes());
    };
    let sc = |out: &mut Vec<u8>, k: &MeasureScale| {
        out.extend_from_slice(&k.px.to_le_bytes());
        out.extend_from_slice(&k.length.to_le_bytes());
        s(out, &k.units);
    };
    sc(out, &m.scale);
    out.extend_from_slice(&(m.records.len() as u32).to_le_bytes());
    for r in &m.records {
        s(out, &r.label);
        s(out, &r.when);
        s(out, &r.document);
        out.push(r.source.code());
        sc(out, &r.scale);
        out.extend_from_slice(&r.count.map_or(u32::MAX, |c| c).to_le_bytes());
        for v in [r.area, r.perimeter, r.circularity, r.width, r.height, r.cx, r.cy, r.gray, r.length, r.angle] {
            out.extend_from_slice(&v.unwrap_or(f64::NAN).to_le_bytes());
        }
    }
}

pub fn decode(b: &mut &[u8]) -> Option<Measurements> {
    fn take<'a>(b: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
        let (h, t) = (b.get(..n)?, b.get(n..)?);
        *b = t;
        Some(h)
    }
    let u32_ = |b: &mut &[u8]| take(b, 4).map(|v| u32::from_le_bytes(v.try_into().unwrap()));
    let f32_ = |b: &mut &[u8]| take(b, 4).map(|v| f32::from_le_bytes(v.try_into().unwrap()));
    let str_ = |b: &mut &[u8]| {
        let n = u32_(b)? as usize;
        take(b, n).map(|v| String::from_utf8_lossy(v).into_owned())
    };
    let scale = |b: &mut &[u8]| Some(MeasureScale { px: f32_(b)?, length: f32_(b)?, units: str_(b)? });
    let mut m = Measurements { scale: scale(b)?, records: vec![] };
    for _ in 0..u32_(b)? {
        let (label, when, document) = (str_(b)?, str_(b)?, str_(b)?);
        let source = Source::from_code(take(b, 1)?[0]);
        let sc = scale(b)?;
        let count = u32_(b).filter(|&c| c != u32::MAX);
        let mut v = [None; 10];
        for x in &mut v {
            let f = f64::from_le_bytes(take(b, 8)?.try_into().unwrap());
            *x = (!f.is_nan()).then_some(f);
        }
        let [area, perimeter, circularity, width, height, cx, cy, gray, length, angle] = v;
        m.records.push(Record { label, when, document, source, scale: sc, count, area, perimeter, circularity, width, height, cx, cy, gray, length, angle });
    }
    Some(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::selection::Selection;

    #[test]
    fn selection_measures_a_disc() {
        let r = 200.0f64;
        let sel = Selection::ellipse(RectI::new(100, 100, 500, 500));
        let m = MaskGrid::from_selection(&sel, 700, 600);
        let rec = selection(&m, &|_, _| 100.0, &MeasureScale::default(), "t").unwrap();
        let (a, p) = (rec.area.unwrap(), rec.perimeter.unwrap());
        let pi = std::f64::consts::PI;
        assert!((a / (pi * r * r) - 1.0).abs() < 0.002, "area {a}");
        assert!((p / (2.0 * pi * r) - 1.0).abs() < 0.005, "perimeter {p}");
        assert!((rec.circularity.unwrap() - 1.0).abs() < 0.01);
        assert!((rec.cx.unwrap() - 300.0).abs() < 0.05 && (rec.cy.unwrap() - 300.0).abs() < 0.05);
        assert_eq!((rec.width, rec.height, rec.count), (Some(400.0), Some(400.0), Some(1)));
        assert!((rec.gray.unwrap() - 100.0).abs() < 1e-9);
        // A scale of 100 px = 2.54 cm.
        let s = MeasureScale { px: 100.0, length: 2.54, units: "cm".into() };
        let rec = selection(&m, &|_, _| 0.0, &s, "t").unwrap();
        assert!((rec.area.unwrap() / (a * 0.0254 * 0.0254) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gray_value_follows_photoshop() {
        // Photoshop's readings of flat fills (tools/photoshop_gaps.sh F).
        for (c, ps) in [([100u8, 100, 100], 77.88), ([0, 255, 0], 210.54), ([0, 0, 255], 50.77), ([200, 150, 50], 139.12), ([255, 255, 255], 255.0)] {
            assert!((ps_gray(c) - ps).abs() < 1.0, "{c:?}: {} vs {ps}", ps_gray(c));
        }
    }

    #[test]
    fn ruler_protractor_counts_and_csv() {
        let r = ruler((0.0, 0.0), (300.0, -400.0), &MeasureScale::default(), "doc");
        assert_eq!(r.length, Some(500.0));
        assert!((protractor_angle((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)) - 90.0).abs() < 1e-4);
        assert!((protractor_angle((5.0, 5.0), (10.0, 5.0), (0.0, 5.0)) - 180.0).abs() < 1e-3);
        let mut g = crate::engine::annot::CountGroup::new("Cells, a \"test\"", [1, 2, 3]);
        g.pts = vec![(1.0, 1.0); 7];
        let mut hidden = crate::engine::annot::CountGroup::new("Hidden", [1, 2, 3]);
        hidden.visible = false;
        let mut log = Measurements::default();
        for rec in counts(&[g, hidden], &log.scale.clone(), "doc") {
            log.push(rec);
        }
        log.push(r);
        let text = csv(&log.records);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("Label,Date and Time,Document,Source"));
        assert!(lines[1].starts_with("\"Cells, a \"\"test\"\"\"") && lines[1].contains(",Count Tool,") && lines[1].contains(",7,"));
        assert!(lines[2].contains("Ruler Tool") && lines[2].contains(",500,"));
        // Round trip through the `.pforge` encoding.
        let mut b = Vec::new();
        encode(&log, &mut b);
        assert_eq!(decode(&mut b.as_slice()), Some(log));
        assert!(now_text().len() == 23);
    }
}
