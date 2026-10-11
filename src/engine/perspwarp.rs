//! Edit ▸ Perspective Warp.
//!
//! **Layout** mode: quads are drawn over planes of the image (a building's
//! two walls, a table top). Corners within a snap distance of another quad's
//! corner become one vertex, so quads drawn side by side join at their shared
//! edge. **Warp** mode: the vertices are dragged (or straightened) to where
//! the planes should go, and the image follows.
//!
//! The map. For each quad `q`, with layout corners `L` (where the plane is
//! in the source) and warp corners `W` (where it goes), the projective map
//! `H_q = S_L · S_W⁻¹` (`S_X` = `distort::square_to_quad(X)`) takes an output
//! position to a source position. Deep inside a warped quad that map is used
//! as is, so straight lines in the plane stay straight. Near an edge the
//! maps of the quads meeting there are blended, so the image stays
//! continuous across joined edges (two quads' projective maps agree at the
//! shared corners but not, in general, along the edge between them) and
//! outside every quad the nearest quads' maps are extrapolated:
//!
//! - `s_q(p)`: signed distance from `p` to quad `q`'s edges (positive
//!   inside), `e_q = max(δ − s_q, 0)` (0 when `δ` or more inside),
//!   `e = min_q e_q`;
//! - `w_q = κ((e_q − e) / 2δ)` with `κ(t) = 1 − smoothstep(t)` (0 for
//!   `t ≥ 1`), `source(p) = Σ w_q H_q(p) / Σ w_q`.
//!
//! So a point `δ` inside one quad uses only that quad, both quads weigh the
//! same on a shared edge, and everything is continuous. `δ` is a fraction of
//! the quads' size ([`Mesh::blend_width`]). Points a contributing map sends
//! behind its horizon are transparent.
//!
//! Rendering is a mesh warp: the map is evaluated on nodes every
//! [`MESH`] pixels of the output ([`Table`]); each pixel interpolates its
//! cell's nodes bilinearly and samples the source bicubically (Catmull-Rom,
//! transparent outside the source). The CPU ([`render_cpu`]) is the reference;
//! the GPU kernel (`gpu::kernels::perspwarp`) shares the node table and
//! evaluates the same float expressions with fusion blocked, so the two
//! agree bit for bit.

use rayon::prelude::*;

use super::filters::distort::{mat_inv, mat_mul, square_to_quad};
use super::grid::{RectI, TS, TS_I, TileGrid, build_grid, finish_tile};
use super::px;

pub type Pt = (f32, f32);

/// Output pixels between mesh nodes.
pub const MESH: u32 = 8;
const MESH_SHIFT: u32 = 3;

/// Node value for "no source here" (behind a horizon).
pub const INVALID: f32 = -1.0e30;

/// Warp mode's automatic straightening.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Straighten {
    /// Near-vertical edges become vertical.
    Vertical,
    /// Near-horizontal edges become horizontal.
    Horizontal,
    /// Both.
    Both,
}

impl Straighten {
    pub const ALL: [Straighten; 3] = [Straighten::Vertical, Straighten::Horizontal, Straighten::Both];

    pub fn label(self) -> &'static str {
        match self {
            Straighten::Vertical => "Straighten Vertical",
            Straighten::Horizontal => "Straighten Horizontal",
            Straighten::Both => "Level & Straighten",
        }
    }
}

/// The quads: shared vertices with a layout and a warp position each.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    /// Vertex positions in Layout mode (the planes in the source).
    pub layout: Vec<Pt>,
    /// Vertex positions in Warp mode (where the planes go).
    pub warp: Vec<Pt>,
    /// Corner vertices of each quad: top-left, top-right, bottom-right,
    /// bottom-left.
    pub quads: Vec<[usize; 4]>,
}

/// One quad's map and the geometry its weight needs.
#[derive(Clone, Copy, Debug)]
pub struct QuadMap {
    /// Output → source (homogeneous).
    pub h: [f32; 9],
    /// Edge lines of the warped quad: unit inward normal `(nx, ny)` and
    /// offset `c` so that `nx·x + ny·y + c` is the signed distance.
    pub edges: [[f32; 3]; 4],
}

fn cross(a: Pt, b: Pt, c: Pt) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Twice the signed area (positive for clockwise on screen, y down).
fn area2(q: &[Pt; 4]) -> f32 {
    (0..4).map(|i| q[i].0 * q[(i + 1) % 4].1 - q[(i + 1) % 4].0 * q[i].1).sum()
}

/// A usable quad: convex, non-degenerate.
pub fn quad_valid(q: &[Pt; 4]) -> bool {
    let a = area2(q);
    if a.abs() < 2.0 {
        return false;
    }
    (0..4).all(|i| cross(q[i], q[(i + 1) % 4], q[(i + 2) % 4]) * a > 0.0)
}

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Mesh {
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }

    fn corners(&self, q: usize, warp: bool) -> [Pt; 4] {
        let v = if warp { &self.warp } else { &self.layout };
        self.quads[q].map(|i| v[i])
    }

    pub fn layout_quad(&self, q: usize) -> [Pt; 4] {
        self.corners(q, false)
    }

    pub fn warp_quad(&self, q: usize) -> [Pt; 4] {
        self.corners(q, true)
    }

    /// Adds a quad (layout = warp = `q`, corners TL, TR, BR, BL). Corners
    /// within `snap` of an existing vertex reuse it: the quads join.
    pub fn add_quad(&mut self, q: [Pt; 4], snap: f32) -> usize {
        let mut idx = [0usize; 4];
        for (k, &p) in q.iter().enumerate() {
            let near = (0..self.layout.len()).filter(|&v| !idx[..k].contains(&v)).min_by(|&a, &b| dist(self.layout[a], p).total_cmp(&dist(self.layout[b], p)));
            idx[k] = match near.filter(|&v| dist(self.layout[v], p) <= snap) {
                Some(v) => v,
                None => {
                    self.layout.push(p);
                    self.warp.push(p);
                    self.layout.len() - 1
                }
            };
        }
        self.quads.push(idx);
        self.quads.len() - 1
    }

    /// The axis-aligned quad spanning two corners.
    pub fn rect_quad(a: Pt, b: Pt) -> [Pt; 4] {
        let (x0, x1, y0, y1) = (a.0.min(b.0), a.0.max(b.0), a.1.min(b.1), a.1.max(b.1));
        [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    }

    /// Moves a vertex (Layout mode moves both positions).
    pub fn move_vertex(&mut self, v: usize, p: Pt, warp: bool) {
        self.warp[v] = p;
        if !warp {
            self.layout[v] = p;
        }
    }

    /// After a Layout drag: if vertex `v` lies within `snap` of another
    /// quad's vertex, the two become one (the quads join). Returns whether
    /// it joined.
    pub fn join(&mut self, v: usize, snap: f32) -> bool {
        let p = self.layout[v];
        let mine: Vec<usize> = self.quads.iter().filter(|q| q.contains(&v)).flatten().copied().collect();
        let Some(o) = (0..self.layout.len()).filter(|&o| o != v && !mine.contains(&o) && self.used(o)).min_by(|&a, &b| dist(self.layout[a], p).total_cmp(&dist(self.layout[b], p))) else {
            return false;
        };
        if dist(self.layout[o], p) > snap {
            return false;
        }
        for q in &mut self.quads {
            for c in q.iter_mut() {
                if *c == v {
                    *c = o;
                }
            }
        }
        self.compact();
        true
    }

    fn used(&self, v: usize) -> bool {
        self.quads.iter().any(|q| q.contains(&v))
    }

    /// Drops unused vertices.
    fn compact(&mut self) {
        let mut map = vec![usize::MAX; self.layout.len()];
        let (mut l, mut w) = (Vec::new(), Vec::new());
        for v in 0..self.layout.len() {
            if self.used(v) {
                map[v] = l.len();
                l.push(self.layout[v]);
                w.push(self.warp[v]);
            }
        }
        for q in &mut self.quads {
            for c in q.iter_mut() {
                *c = map[*c];
            }
        }
        self.layout = l;
        self.warp = w;
    }

    pub fn remove_quad(&mut self, q: usize) {
        self.quads.remove(q);
        self.compact();
    }

    /// Back to the layout (Warp mode's reset).
    pub fn reset_warp(&mut self) {
        self.warp = self.layout.clone();
    }

    /// Undirected edges, each once.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let mut e: Vec<(usize, usize)> = self.quads.iter().flat_map(|q| (0..4).map(move |i| (q[i].min(q[(i + 1) % 4]), q[i].max(q[(i + 1) % 4])))).collect();
        e.sort_unstable();
        e.dedup();
        e
    }

    /// Edges two quads share.
    pub fn shared_edges(&self) -> usize {
        let mut e: Vec<(usize, usize)> = self.quads.iter().flat_map(|q| (0..4).map(move |i| (q[i].min(q[(i + 1) % 4]), q[i].max(q[(i + 1) % 4])))).collect();
        e.sort_unstable();
        e.windows(2).filter(|w| w[0] == w[1]).count()
    }

    /// Warp mode's straighten buttons: edges within 45° of vertical
    /// (horizontal) become exactly vertical (horizontal). Vertices linked by
    /// such edges take their mean x (y), so joined quads stay joined.
    pub fn straighten(&mut self, s: Straighten) {
        let edges = self.edges();
        let n = self.warp.len();
        for vertical in [true, false] {
            if (vertical && s == Straighten::Horizontal) || (!vertical && s == Straighten::Vertical) {
                continue;
            }
            let mut parent: Vec<usize> = (0..n).collect();
            fn root(p: &mut [usize], mut i: usize) -> usize {
                while p[i] != i {
                    p[i] = p[p[i]];
                    i = p[i];
                }
                i
            }
            for &(a, b) in &edges {
                let (pa, pb) = (self.warp[a], self.warp[b]);
                let (dx, dy) = ((pb.0 - pa.0).abs(), (pb.1 - pa.1).abs());
                if (vertical && dx <= dy) || (!vertical && dx > dy) {
                    let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                    parent[ra] = rb;
                }
            }
            let mut sum = vec![(0f64, 0usize); n];
            for v in 0..n {
                let r = root(&mut parent, v);
                sum[r].0 += if vertical { self.warp[v].0 } else { self.warp[v].1 } as f64;
                sum[r].1 += 1;
            }
            for v in 0..n {
                let r = root(&mut parent, v);
                let m = (sum[r].0 / sum[r].1 as f64) as f32;
                if vertical {
                    self.warp[v].0 = m;
                } else {
                    self.warp[v].1 = m;
                }
            }
        }
    }

    /// The width of the blend band at quad edges: 6 % of the median edge
    /// length of the warped quads (2 to 64 px).
    pub fn blend_width(&self) -> f32 {
        let mut l: Vec<f32> = self.edges().iter().map(|&(a, b)| dist(self.warp[a], self.warp[b])).collect();
        if l.is_empty() {
            return 8.0;
        }
        l.sort_by(f32::total_cmp);
        (l[l.len() / 2] * 0.06).clamp(2.0, 64.0)
    }

    /// The valid quads' maps.
    pub fn maps(&self) -> Vec<QuadMap> {
        (0..self.quads.len())
            .filter_map(|q| {
                let (l, w) = (self.layout_quad(q), self.warp_quad(q));
                if !quad_valid(&l) || !quad_valid(&w) {
                    return None;
                }
                let h = mat_mul(&square_to_quad(l), &mat_inv(&square_to_quad(w)));
                let sgn = area2(&w).signum();
                let edges = [0, 1, 2, 3].map(|i| {
                    let (a, b) = (w[i], w[(i + 1) % 4]);
                    let len = dist(a, b).max(1e-6);
                    // cross(b - a, p - a) / len, signed so inside is positive.
                    let (nx, ny) = (-(b.1 - a.1) / len * sgn, (b.0 - a.0) / len * sgn);
                    [nx, ny, -(nx * a.0 + ny * a.1)]
                });
                Some(QuadMap { h, edges })
            })
            .collect()
    }

    /// Source position for output position `(x, y)` (continuous pixel
    /// coordinates), None behind a horizon. Identity without quads.
    #[cfg(test)]
    pub fn map(&self, x: f32, y: f32) -> Option<Pt> {
        map_with(&self.maps(), self.blend_width(), x, y)
    }

    /// Where a source point inside layout quad `q` goes (for tests and
    /// overlays): `S_W · S_L⁻¹`.
    #[cfg(test)]
    pub fn forward(&self, q: usize, p: Pt) -> Option<Pt> {
        let m = mat_mul(&square_to_quad(self.warp_quad(q)), &mat_inv(&square_to_quad(self.layout_quad(q))));
        apply_h(&m, p.0, p.1)
    }

    /// The node table for a `w × h` output (see [`Table`]).
    pub fn table(&self, w: u32, h: u32) -> Table {
        Table::build(&self.maps(), self.blend_width(), w, h)
    }
}

fn dist(a: Pt, b: Pt) -> f32 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

#[inline]
fn apply_h(m: &[f32; 9], x: f32, y: f32) -> Option<Pt> {
    let wq = m[6] * x + m[7] * y + m[8];
    if wq <= 1e-6 {
        return None;
    }
    Some(((m[0] * x + m[1] * y + m[2]) / wq, (m[3] * x + m[4] * y + m[5]) / wq))
}

/// The blended map (see the module docs).
pub fn map_with(maps: &[QuadMap], delta: f32, x: f32, y: f32) -> Option<Pt> {
    if maps.is_empty() {
        return Some((x, y));
    }
    if maps.len() == 1 {
        return apply_h(&maps[0].h, x, y);
    }
    let mut e = [0f32; 64];
    let n = maps.len().min(64);
    let mut emin = f32::MAX;
    for (k, m) in maps.iter().take(n).enumerate() {
        let s = m.edges.iter().map(|l| l[0] * x + l[1] * y + l[2]).fold(f32::MAX, f32::min);
        e[k] = (delta - s).max(0.0);
        emin = emin.min(e[k]);
    }
    let (mut sx, mut sy, mut sw) = (0f32, 0f32, 0f32);
    for (k, m) in maps.iter().take(n).enumerate() {
        let w = 1.0 - smoothstep((e[k] - emin) / (2.0 * delta));
        if w <= 0.0 {
            continue;
        }
        let p = apply_h(&m.h, x, y)?;
        sx += p.0 * w;
        sy += p.1 * w;
        sw += w;
    }
    Some((sx / sw, sy / sw))
}

/// The map sampled on mesh nodes: node `(i, j)` is the source position of
/// output pixel `(i·MESH, j·MESH)`'s center ([`INVALID`] where there is
/// none), over the output's whole tile area.
#[derive(Clone, Debug)]
pub struct Table {
    /// Output size.
    pub w: u32,
    pub h: u32,
    pub nx: u32,
    pub ny: u32,
    /// `nx × ny` source positions.
    pub nodes: Vec<[f32; 2]>,
}

impl Table {
    pub fn build(maps: &[QuadMap], delta: f32, w: u32, h: u32) -> Table {
        let g = TileGrid::new(w, h);
        let (tw, th) = (g.cols * TS as u32, g.rows * TS as u32);
        let (nx, ny) = (tw / MESH + 1, th / MESH + 1);
        let mut nodes = vec![[0f32; 2]; (nx * ny) as usize];
        nodes.par_chunks_mut(nx as usize).enumerate().for_each(|(j, row)| {
            let y = (j as u32 * MESH) as f32 + 0.5;
            for (i, n) in row.iter_mut().enumerate() {
                let x = (i as u32 * MESH) as f32 + 0.5;
                *n = match map_with(maps, delta, x, y) {
                    Some(p) if p.0.is_finite() && p.1.is_finite() && p.0.abs() < 1e8 && p.1.abs() < 1e8 => [p.0, p.1],
                    _ => [INVALID, INVALID],
                };
            }
        });
        Table { w, h, nx, ny, nodes }
    }

    /// The identity (tests).
    #[cfg(test)]
    pub fn identity(w: u32, h: u32) -> Table {
        Table::build(&[], 1.0, w, h)
    }

    /// Source position of output pixel `(x, y)` (inside the tile area), or
    /// None. Mirrored exactly by `perspwarp.wgsl::position`.
    #[inline]
    pub fn position(&self, x: u32, y: u32) -> Option<Pt> {
        let (i, j) = ((x >> MESH_SHIFT) as usize, (y >> MESH_SHIFT) as usize);
        let nx = self.nx as usize;
        let a = self.nodes[j * nx + i];
        let b = self.nodes[j * nx + i + 1];
        let c = self.nodes[(j + 1) * nx + i];
        let d = self.nodes[(j + 1) * nx + i + 1];
        if a[0] == INVALID || b[0] == INVALID || c[0] == INVALID || d[0] == INVALID {
            return None;
        }
        let fx = (x & (MESH - 1)) as f32 * (1.0 / MESH as f32);
        let fy = (y & (MESH - 1)) as f32 * (1.0 / MESH as f32);
        let lerp = |p: f32, q: f32, t: f32| p + (q - p) * t;
        let top = (lerp(a[0], b[0], fx), lerp(a[1], b[1], fx));
        let bot = (lerp(c[0], d[0], fx), lerp(c[1], d[1], fx));
        Some((lerp(top.0, bot.0, fy), lerp(top.1, bot.1, fy)))
    }

    /// Source footprint of an output rectangle (valid nodes only), or None.
    fn footprint(&self, r: RectI) -> Option<(f32, f32, f32, f32)> {
        let (i0, j0) = ((r.x0 as u32 >> MESH_SHIFT) as usize, (r.y0 as u32 >> MESH_SHIFT) as usize);
        let (i1, j1) = ((((r.x1 - 1) as u32 >> MESH_SHIFT) + 1) as usize, (((r.y1 - 1) as u32 >> MESH_SHIFT) + 1) as usize);
        let mut b = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for j in j0..=j1.min(self.ny as usize - 1) {
            for i in i0..=i1.min(self.nx as usize - 1) {
                let n = self.nodes[j * self.nx as usize + i];
                if n[0] != INVALID {
                    b = (b.0.min(n[0]), b.1.min(n[1]), b.2.max(n[0]), b.3.max(n[1]));
                }
            }
        }
        (b.0 <= b.2).then_some(b)
    }
}

/// Catmull-Rom weights of the taps at `floor(t) - 1 ..= floor(t) + 2` for
/// fraction `f` (as `crop::Interp::Bicubic`). Mirrored by the WGSL.
#[inline]
pub fn cubic_weights(f: f32) -> [f32; 4] {
    let t2 = f * f;
    let t3 = t2 * f;
    [(-t3 + 2.0 * t2 - f) * 0.5, (3.0 * t3 - 5.0 * t2 + 2.0) * 0.5, (-3.0 * t3 + 4.0 * t2 + f) * 0.5, (t3 - t2) * 0.5]
}

/// Positions within 1/1000 px of a pixel center snap to it (exact identity).
#[inline]
fn snap(v: f32) -> f32 {
    let r = v.round();
    if (v - r).abs() < 1e-3 { r } else { v }
}

/// Rust's `f32::round` (half away from zero) for the pixel pack.
#[inline]
fn pack_acc(acc: [f32; 4]) -> u32 {
    let a = acc[3].round().clamp(0.0, 255.0);
    if a <= 0.0 {
        return 0;
    }
    let c = |v: f32| v.round().clamp(0.0, a) as u8;
    px::pack(c(acc[0]), c(acc[1]), c(acc[2]), a as u8)
}

/// Bicubic sample at source position `p` (continuous coordinates), reading
/// taps through `at` (0 outside the source).
#[inline]
fn bicubic(p: Pt, at: impl Fn(i32, i32) -> u32) -> u32 {
    let sx = snap(p.0 - 0.5);
    let sy = snap(p.1 - 0.5);
    let (ix, iy) = (sx.floor(), sy.floor());
    let wx = cubic_weights(sx - ix);
    let wy = cubic_weights(sy - iy);
    let (bx, by) = (ix as i32 - 1, iy as i32 - 1);
    let mut acc = [0f32; 4];
    for (j, &wyj) in wy.iter().enumerate() {
        for (i, &wxi) in wx.iter().enumerate() {
            let v = at(bx + i as i32, by + j as i32);
            if v == 0 {
                continue;
            }
            let q = px::unpack(v);
            let k = wxi * wyj;
            acc[0] += q[0] as f32 * k;
            acc[1] += q[1] as f32 * k;
            acc[2] += q[2] as f32 * k;
            acc[3] += q[3] as f32 * k;
        }
    }
    pack_acc(acc)
}

/// The CPU reference: `src` resampled through `table` into a
/// `table.w × table.h` grid (whole tiles, padding included, as the GPU).
pub fn render_cpu(src: &TileGrid, table: &Table) -> TileGrid {
    let sb = src.bounds();
    build_grid(table.w, table.h, |tx, ty| {
        let r = TileGrid::tile_rect(tx, ty);
        let fp = table.footprint(r)?;
        let fr = RectI::new(fp.0.floor() as i32 - 3, fp.1.floor() as i32 - 3, fp.2.ceil() as i32 + 3, fp.3.ceil() as i32 + 3);
        if !fr.intersects(&sb) {
            return None;
        }
        let mut out = vec![0u32; TS * TS].into_boxed_slice();
        let (x0, y0) = (tx * TS_I, ty * TS_I);
        let area = fr.w() as i64 * fr.h() as i64;
        if area <= 16 * (TS * TS) as i64 {
            // The footprint in one read; outside the source is transparent.
            let c = fr.intersect(&sb);
            let (bw, bh) = (c.w() as usize, c.h() as usize);
            let mut b = vec![0u32; bw * bh];
            src.read_region(c.x0, c.y0, bw, bh, &mut b);
            let at = |x: i32, y: i32| if x >= c.x0 && y >= c.y0 && x < c.x1 && y < c.y1 { b[(y - c.y0) as usize * bw + (x - c.x0) as usize] } else { 0 };
            for y in 0..TS {
                for x in 0..TS {
                    if let Some(p) = table.position((x0 + x as i32) as u32, (y0 + y as i32) as u32) {
                        out[y * TS + x] = bicubic(p, at);
                    }
                }
            }
        } else {
            // Strong compression: keep only neighboring source tiles pinned.
            let sampler = super::filters::distort::Sampler::new(src, None, TS * TS, false);
            let at = |x: i32, y: i32| if x >= 0 && y >= 0 && x < sb.x1 && y < sb.y1 { sampler.px(x, y) } else { 0 };
            for y in 0..TS {
                for x in 0..TS {
                    if let Some(p) = table.position((x0 + x as i32) as u32, (y0 + y as i32) as u32) {
                        out[y * TS + x] = bicubic(p, at);
                    }
                }
            }
        }
        finish_tile(out)
    })
}

/// Perspective Warp of a layer: the GPU when the mode allows it (Image
/// Size & Transform group) and the layer is large enough, the CPU otherwise
/// and on any GPU failure. Same pixels either way.
pub fn apply(src: &TileGrid, mesh: &Mesh) -> TileGrid {
    use super::gpu;
    let t0 = std::time::Instant::now();
    let table = mesh.table(src.w, src.h);
    let px = src.w as u64 * src.h as u64;
    let m = gpu::feature_mode(gpu::Feature::Resample);
    let why = match m {
        gpu::Mode::Off => Some("GPU off".to_string()),
        gpu::Mode::Auto if px < AUTO_GPU_MIN_PX => Some("small layer".into()),
        _ => None,
    };
    if why.is_none()
        && let Some(g) = gpu::context(gpu::blocking() || m == gpu::Mode::Force)
    {
        match gpu::kernels::perspwarp::render(&g, src, &table, false) {
            Ok(out) => {
                gpu::record("Perspective Warp", gpu::PathTaken::Gpu, t0.elapsed().as_secs_f64() * 1e3);
                return out;
            }
            Err(e) => {
                if e.is_fault() {
                    g.fail(e.to_string());
                }
                let out = render_cpu(src, &table);
                gpu::record("Perspective Warp", gpu::PathTaken::Cpu(e.to_string()), t0.elapsed().as_secs_f64() * 1e3);
                return out;
            }
        }
    }
    let out = render_cpu(src, &table);
    gpu::record("Perspective Warp", gpu::PathTaken::Cpu(why.unwrap_or_else(|| format!("GPU {}", gpu::describe()))), t0.elapsed().as_secs_f64() * 1e3);
    out
}

/// Layers at least this large go to the GPU in Auto mode.
pub const AUTO_GPU_MIN_PX: u64 = 1 << 20;

#[cfg(test)]
mod tests {
    use super::*;

    fn checker(w: u32, h: u32) -> TileGrid {
        build_grid(w, h, |tx, ty| {
            let mut t = vec![0u32; TS * TS];
            for y in 0..TS {
                for x in 0..TS {
                    let (gx, gy) = (tx * TS_I + x as i32, ty * TS_I + y as i32);
                    let v = if ((gx / 16) + (gy / 16)) % 2 == 0 { 230 } else { 30 };
                    t[y * TS + x] = px::pack(v, (gx % 256) as u8, (gy % 256) as u8, 255);
                }
            }
            finish_tile(t.into_boxed_slice())
        })
    }

    #[test]
    fn identity_quads_are_the_identity() {
        let mut m = Mesh::default();
        m.add_quad(Mesh::rect_quad((100.0, 80.0), (400.0, 300.0)), 10.0);
        m.add_quad(Mesh::rect_quad((400.0, 80.0), (700.0, 300.0)), 10.0);
        m.add_quad([(120.0, 320.0), (380.0, 340.0), (360.0, 560.0), (90.0, 520.0)], 10.0);
        assert_eq!(m.shared_edges(), 1, "the side-by-side quads join");
        for y in (0..600).step_by(7) {
            for x in (0..800).step_by(11) {
                let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
                let p = m.map(fx, fy).unwrap();
                assert!((p.0 - fx).abs() < 2e-3 && (p.1 - fy).abs() < 2e-3, "({fx}, {fy}) -> {p:?}");
            }
        }
        let src = checker(813, 611);
        let out = render_cpu(&src, &m.table(src.w, src.h));
        let mut a = vec![0u32; 813 * 611];
        let mut b = a.clone();
        src.read_region(0, 0, 813, 611, &mut a);
        out.read_region(0, 0, 813, 611, &mut b);
        assert!(a == b, "identity warp changed pixels");
        let empty = render_cpu(&src, &Table::identity(src.w, src.h));
        empty.read_region(0, 0, 813, 611, &mut b);
        assert!(a == b);
    }

    /// A keystoned facade (verticals converging upward) straightened.
    fn facade() -> Mesh {
        let mut m = Mesh::default();
        m.add_quad([(300.0, 120.0), (520.0, 130.0), (590.0, 620.0), (230.0, 610.0)], 8.0);
        m.add_quad([(520.0, 130.0), (700.0, 160.0), (790.0, 600.0), (590.0, 620.0)], 8.0);
        m
    }

    #[test]
    fn straighten_makes_verticals_vertical() {
        let mut m = facade();
        assert_eq!(m.shared_edges(), 1);
        m.straighten(Straighten::Vertical);
        for q in 0..2 {
            let w = m.warp_quad(q);
            assert!((w[0].0 - w[3].0).abs() < 1e-3 && (w[1].0 - w[2].0).abs() < 1e-3, "{w:?}");
            // Near-horizontal edges keep their slant.
            assert!((w[0].1 - w[1].1).abs() > 1.0);
        }
        // The slanted source edges come out vertical: every point along a
        // layout edge maps to one output column.
        for q in 0..2 {
            let l = m.layout_quad(q);
            for (a, b) in [(l[0], l[3]), (l[1], l[2])] {
                let xs: Vec<f32> = (0..=10).map(|k| {
                    let t = k as f32 / 10.0;
                    m.forward(q, (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)).unwrap().0
                }).collect();
                let spread = xs.iter().fold(f32::MIN, |a, &b| a.max(b)) - xs.iter().fold(f32::MAX, |a, &b| a.min(b));
                assert!(spread < 1e-2, "column spread {spread}");
            }
        }
        // And in pixels: a dark line along the left layout edge renders as
        // a vertical line.
        let (w, h) = (900u32, 700u32);
        let l = m.layout_quad(0);
        let (a, b) = (l[0], l[3]);
        let src = build_grid(w, h, |tx, ty| {
            let mut t = vec![0u32; TS * TS];
            for y in 0..TS {
                for x in 0..TS {
                    let (gx, gy) = ((tx * TS_I + x as i32) as f32 + 0.5, (ty * TS_I + y as i32) as f32 + 0.5);
                    let d = cross(a, b, (gx, gy)).abs() / dist(a, b);
                    let v = if d < 2.0 { 0 } else { 255 };
                    t[y * TS + x] = px::pack(v, v, v, 255);
                }
            }
            finish_tile(t.into_boxed_slice())
        });
        let out = render_cpu(&src, &m.table(w, h));
        let wq = m.warp_quad(0);
        let col = wq[0].0;
        for y in ((wq[0].1 + 40.0) as i32..(wq[3].1 - 40.0) as i32).step_by(10) {
            // The line's center: the darkness-weighted mean column near it.
            let (mut sw, mut sx) = (0f32, 0f32);
            for x in (col as i32 - 30)..(col as i32 + 30) {
                let d = 255.0 - px::unpack(out.pixel(x, y))[0] as f32;
                sw += d;
                sx += d * (x as f32 + 0.5);
            }
            let center = sx / sw.max(1.0);
            assert!((center - col).abs() <= 0.5, "row {y}: line at {center}, want {col}");
        }
        // Horizontal straightening levels near-horizontal edges too.
        let mut m2 = facade();
        m2.straighten(Straighten::Both);
        for q in 0..2 {
            let w = m2.warp_quad(q);
            assert!((w[0].1 - w[1].1).abs() < 1e-3 && (w[3].1 - w[2].1).abs() < 1e-3 && (w[0].0 - w[3].0).abs() < 1e-3);
        }
    }

    #[test]
    fn joined_quads_warp_continuously() {
        let mut m = facade();
        // Pull the shared edge sideways and the outer corners around.
        m.warp[1].0 += 60.0;
        m.warp[2].0 += 35.0;
        m.warp[0].1 -= 30.0;
        let maps = m.maps();
        let delta = m.blend_width();
        let mut worst = 0f32;
        for k in 0..200 {
            let t = k as f32 / 200.0;
            let (a, b) = (m.warp[1], m.warp[2]);
            let p = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            let l = map_with(&maps, delta, p.0 - 0.05, p.1).unwrap();
            let r = map_with(&maps, delta, p.0 + 0.05, p.1).unwrap();
            worst = worst.max(dist(l, r));
        }
        assert!(worst < 0.5, "jump across the joined edge: {worst}");
        // Deep inside a quad the map is that quad's projective map.
        let q0 = m.warp_quad(0);
        let c = ((q0[0].0 + q0[2].0) * 0.5, (q0[0].1 + q0[2].1) * 0.5);
        let h = apply_h(&maps[0].h, c.0, c.1).unwrap();
        let got = m.map(c.0, c.1).unwrap();
        assert!(dist(h, got) < 1e-3);
    }

    #[test]
    fn joining_by_dragging_a_corner() {
        let mut m = Mesh::default();
        m.add_quad(Mesh::rect_quad((0.0, 0.0), (100.0, 100.0)), 5.0);
        let q = m.add_quad(Mesh::rect_quad((130.0, 0.0), (230.0, 100.0)), 5.0);
        assert_eq!(m.shared_edges(), 0);
        let (tl, bl) = (m.quads[q][0], m.quads[q][3]);
        m.move_vertex(tl, (102.0, 1.0), false);
        assert!(m.join(tl, 5.0));
        let bl = m.quads[q][3].min(bl);
        m.move_vertex(m.quads[q][3], (101.0, 99.0), false);
        let _ = bl;
        assert!(m.join(m.quads[q][3], 5.0));
        assert_eq!(m.shared_edges(), 1);
        assert_eq!(m.layout.len(), 6);
    }
}
