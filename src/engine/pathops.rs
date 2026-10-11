//! Robust boolean operations on flattened polygon sets (Photoshop-style path
//! operations: Combine/Union, Subtract Front, Intersect, Exclude/XOR).
//!
//! # Algorithm
//!
//! 1. **Snap** every input vertex to a fixed integer grid of 1/1024 px. All
//!    later predicates are exact integer arithmetic (`i64`, `i128` where a
//!    product could overflow), so there are no epsilon decisions anywhere.
//! 2. **Snap rounding** (Hobby / Guibas–Marimont, then iterated): all segment
//!    crossings are found with a uniform-grid acceleration structure, each
//!    crossing is rounded to its grid point ("hot pixel"), and every segment that
//!    passes through a hot pixel (crossing or vertex) is re-routed through that
//!    pixel's centre. Fragments created by re-routing are re-checked until no
//!    fragment passes through a hot pixel other than its own endpoints. The
//!    result is a planar straight-line graph: edges only meet at shared
//!    endpoints, and overlapping collinear pieces become identical edges.
//! 3. **Dedupe** identical edges, summing their signed winding contributions
//!    per operand (edges whose contributions cancel are dropped: both sides of
//!    such an edge have the same winding).
//! 4. **Sweep** the graph in lexicographic `(y, x)` order (a symbolically tilted
//!    horizontal sweep, so horizontal edges need no special case) with a
//!    status list ordered left to right. Because the graph is planar the order
//!    never changes between events, so the winding numbers of both operands on
//!    each side of every edge follow from its left neighbour.
//! 5. **Select** edges whose two sides evaluate differently under
//!    `op(fill_a(w_a), fill_b(w_b))`, orient them so the result region lies on
//!    a fixed side, and **chain** them into closed contours. At vertices with
//!    several candidate continuations the sharpest turn towards the inside is
//!    taken, which splits regions that only touch at a point into separate
//!    contours.
//!
//! # Output convention
//!
//! Every returned [`Poly`] is closed and has no proper self-intersections, and
//! no two returned contours cross each other (they may touch at vertices).
//! Outer boundaries have **positive** signed shoelace area as computed by
//! [`area`] (`Σ x_i·y_{i+1} − x_{i+1}·y_i`), holes negative. In y-down image
//! coordinates a positive shoelace area means the contour runs **clockwise on
//! screen** (counter-clockwise in a y-up mathematical frame); the filled region
//! is always on the right-hand side when walking a contour on screen. Holes run
//! the other way, so filling the result with either `NonZero` or `EvenOdd`
//! yields exactly the boolean region, and `area(result)` is its area.
//!
//! Coordinates are snapped to 1/1024 px and clamped to ±2^19 px (524 288 px).
//!
//! Stays on the CPU (no GPU path): the sweep is a sequential event queue
//! with exact integer predicates and data-dependent splitting, operating on
//! kilobytes of vertices, so a dispatch and readback would cost more than
//! the whole operation.

use super::vector::{FillRule, P, Poly, p};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::ops::Range;

/// Path operation applied between two polygon sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    /// A ∪ B (Photoshop "Combine Shapes").
    Union,
    /// A − B (Photoshop "Subtract Front Shape").
    Subtract,
    /// A ∩ B (Photoshop "Intersect Shape Areas").
    Intersect,
    /// A ⊕ B (Photoshop "Exclude Overlapping Shapes").
    Xor,
}

/// Boolean of two polygon sets, each interpreted with its own fill rule (open
/// polys are treated as closed).
///
/// Returns closed, non-self-intersecting contours following the module's
/// output convention: outer boundaries have positive [`area`] (clockwise on
/// screen in y-down coordinates), holes negative, so filling the result with
/// `NonZero` (or `EvenOdd`) gives exactly the resulting region.
pub fn boolean(a: &[Poly], rule_a: FillRule, b: &[Poly], rule_b: FillRule, op: BoolOp) -> Vec<Poly> {
    let mut segs = Vec::new();
    push_polys(a, 1, 0, &mut segs);
    let na = segs.len();
    push_polys(b, 0, 1, &mut segs);
    let nb = segs.len() - na;
    // Trivially empty results.
    match op {
        BoolOp::Intersect if na == 0 || nb == 0 => return Vec::new(),
        BoolOp::Subtract if na == 0 => return Vec::new(),
        _ => {}
    }
    let keep = move |wa: i32, wb: i32| {
        let ia = filled(rule_a, wa);
        let ib = filled(rule_b, wb);
        match op {
            BoolOp::Union => ia || ib,
            BoolOp::Subtract => ia && !ib,
            BoolOp::Intersect => ia && ib,
            BoolOp::Xor => ia != ib,
        }
    };
    run(segs, keep)
}

/// Normalizes one polygon set (resolves self-intersections, overlaps and
/// redundant contours) under its fill rule. Same output convention as
/// [`boolean`].
pub fn simplify(a: &[Poly], rule: FillRule) -> Vec<Poly> {
    let mut segs = Vec::new();
    push_polys(a, 1, 0, &mut segs);
    run(segs, move |wa, _| filled(rule, wa))
}

/// Applies a sequence of operations left to right, like Photoshop's
/// per-subpath operations: `result = ((s0 op1 s1) op2 s2) ...`. The first
/// entry's operation is ignored. Each operand uses its own fill rule;
/// intermediate results are interpreted with `NonZero`.
pub fn combine_sequence(items: &[(BoolOp, Vec<Poly>, FillRule)]) -> Vec<Poly> {
    let Some((_, first, first_rule)) = items.first() else {
        return Vec::new();
    };
    if items.len() == 1 {
        return simplify(first, *first_rule);
    }
    let mut acc = boolean(first, *first_rule, &items[1].1, items[1].2, items[1].0);
    for (op, polys, rule) in &items[2..] {
        acc = boolean(&acc, FillRule::NonZero, polys, *rule, *op);
    }
    acc
}

/// Signed area (shoelace) summed over all polys, treating each as closed.
/// Positive for outer contours produced by this module, negative for holes.
pub fn area(polys: &[Poly]) -> f64 {
    polys
        .iter()
        .map(|poly| {
            let pts = &poly.pts;
            if pts.len() < 3 {
                return 0.0;
            }
            // Relative to the first vertex for better precision far from the origin.
            let (ox, oy) = (pts[0].x as f64, pts[0].y as f64);
            let mut s = 0.0;
            for i in 0..pts.len() {
                let a = pts[i];
                let b = pts[(i + 1) % pts.len()];
                let (ax, ay) = (a.x as f64 - ox, a.y as f64 - oy);
                let (bx, by) = (b.x as f64 - ox, b.y as f64 - oy);
                s += ax * by - bx * ay;
            }
            s * 0.5
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Integer geometry
// ---------------------------------------------------------------------------

/// Grid units per pixel.
const SCALE: f64 = 1024.0;
/// Coordinate clamp in grid units; keeps every 2×2 determinant within `i64`.
const LIMIT: f64 = (1i64 << 29) as f64;
/// Below this many segments everything runs on the calling thread.
const PAR_MIN: usize = 4096;
/// Upper bound on iterated-snap-rounding passes (converges in 1–3 in practice).
const MAX_ROUNDS: usize = 32;

/// Snapped integer point. Sweep order is lexicographic by `(y, x)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IP {
    x: i64,
    y: i64,
}

impl IP {
    #[inline]
    fn key(self) -> (i64, i64) {
        (self.y, self.x)
    }
}

#[inline]
fn cross(ax: i64, ay: i64, bx: i64, by: i64) -> i64 {
    ax * by - ay * bx
}

/// Twice the signed area of triangle `abc`; positive when `c` is on the
/// "inside" (positive-area) side of the directed line `a → b`.
#[inline]
fn orient(a: IP, b: IP, c: IP) -> i64 {
    cross(b.x - a.x, b.y - a.y, c.x - a.x, c.y - a.y)
}

#[inline]
fn snap(v: f32) -> i64 {
    let f = (v as f64 * SCALE).round();
    if f.is_nan() { 0 } else { f.clamp(-LIMIT, LIMIT) as i64 }
}

/// A directed input segment with its winding contribution per operand
/// (`+1`/`0` for the direction `a → b`).
#[derive(Clone, Copy, Debug)]
struct Seg {
    a: IP,
    b: IP,
    wa: i32,
    wb: i32,
}

#[inline]
fn filled(rule: FillRule, w: i32) -> bool {
    match rule {
        FillRule::NonZero => w != 0,
        FillRule::EvenOdd => w & 1 != 0,
    }
}

/// Appends the snapped, implicitly closed edges of `polys`.
fn push_polys(polys: &[Poly], wa: i32, wb: i32, out: &mut Vec<Seg>) {
    for poly in polys {
        let Some(last) = poly.pts.last() else { continue };
        let mut prev = IP { x: snap(last.x), y: snap(last.y) };
        for q in &poly.pts {
            let q = IP { x: snap(q.x), y: snap(q.y) };
            if q != prev {
                out.push(Seg { a: prev, b: q, wa, wb });
                prev = q;
            }
        }
    }
}

/// Rounded `n / d` (round half up), exact.
#[inline]
fn div_round(n: i128, d: i128) -> i64 {
    let (n, d) = if d < 0 { (-n, -d) } else { (n, d) };
    (2 * n + d).div_euclid(2 * d) as i64
}

/// If segments `ab` and `cd` cross at a single interior point, returns that
/// point rounded to the grid. Touching/collinear configurations return `None`:
/// they involve an endpoint, which is a hot pixel anyway.
#[inline]
fn crossing_point(a: IP, b: IP, c: IP, d: IP) -> Option<IP> {
    if a.x.max(b.x) < c.x.min(d.x)
        || c.x.max(d.x) < a.x.min(b.x)
        || a.y.max(b.y) < c.y.min(d.y)
        || c.y.max(d.y) < a.y.min(b.y)
    {
        return None;
    }
    let o1 = orient(a, b, c).signum();
    let o2 = orient(a, b, d).signum();
    if o1 == 0 || o2 == 0 || o1 == o2 {
        return None;
    }
    let o3 = orient(c, d, a);
    let o4 = orient(c, d, b);
    if o3 == 0 || o4 == 0 || o3.signum() == o4.signum() {
        return None;
    }
    // Intersection at parameter t = o3 / (o3 - o4) along a → b.
    let num = o3 as i128;
    let den = o3 as i128 - o4 as i128;
    Some(IP {
        x: a.x + div_round((b.x - a.x) as i128 * num, den),
        y: a.y + div_round((b.y - a.y) as i128 * num, den),
    })
}

/// Whether segment `ab` (non-degenerate) intersects the hot pixel of grid
/// point `h`: the half-open unit square `[h - 1/2, h + 1/2)²`, which matches the
/// round-half-up rounding of crossings. Exact.
#[inline]
fn touches(a: IP, b: IP, h: IP) -> bool {
    // Bounding-box axes (integer endpoints make the half-open bounds simple).
    if h.x < a.x.min(b.x) || h.x > a.x.max(b.x) || h.y < a.y.min(b.y) || h.y > a.y.max(b.y) {
        return false;
    }
    // Normal axis: |n·(h - a)| vs. the square's support 0.5·(|dx| + |dy|), in doubled units.
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let c2 = 2 * cross(dx, dy, h.x - a.x, h.y - a.y);
    let s = dx.abs() + dy.abs();
    if c2.abs() != s {
        return c2.abs() < s;
    }
    // The line grazes a corner/edge of the closed square. The half-open square
    // is the closed square shifted by (-ε, -ε²); decide by the sign of the shift.
    let pert = if dy != 0 { dy } else { -dx };
    c2.signum() * pert.signum() < 0
}

// ---------------------------------------------------------------------------
// Uniform grid
// ---------------------------------------------------------------------------

/// Bucket grid over segments (CSR layout). Each segment is registered in every
/// cell its (slightly inflated) footprint touches.
struct Grid {
    ox: f64,
    oy: f64,
    cs: f64,
    inv: f64,
    nx: usize,
    ny: usize,
    bbox: [i64; 4],
    start: Vec<u32>,
    items: Vec<u32>,
}

/// Inflation (grid units) of segment footprints: covers the half-unit hot
/// pixel radius plus float rounding of the cell computations.
const MARGIN: f64 = 2.0;

impl Grid {
    fn build(segs: &[Seg]) -> Grid {
        let mut bbox = [i64::MAX, i64::MAX, i64::MIN, i64::MIN];
        let mut total = 0.0;
        for s in segs {
            for q in [s.a, s.b] {
                bbox[0] = bbox[0].min(q.x);
                bbox[1] = bbox[1].min(q.y);
                bbox[2] = bbox[2].max(q.x);
                bbox[3] = bbox[3].max(q.y);
            }
            total += (s.b.x - s.a.x).abs().max((s.b.y - s.a.y).abs()) as f64;
        }
        if segs.is_empty() {
            bbox = [0; 4];
        }
        let n = segs.len().max(1) as f64;
        let w = (bbox[2] - bbox[0]) as f64 + 1.0;
        let h = (bbox[3] - bbox[1]) as f64 + 1.0;
        // Cell size ≈ average edge length, so a typical edge spans 1–3 cells,
        // but never more than ~4 cells per segment overall.
        let max_cells = 4.0 * n + 64.0;
        let mut cs = (total / n).max(16.0);
        let (nx, ny) = loop {
            let nx = (w / cs).ceil().max(1.0);
            let ny = (h / cs).ceil().max(1.0);
            if nx * ny <= max_cells {
                break (nx as usize, ny as usize);
            }
            cs *= 1.25;
        };
        let mut g = Grid {
            ox: bbox[0] as f64,
            oy: bbox[1] as f64,
            cs,
            inv: 1.0 / cs,
            nx,
            ny,
            bbox,
            start: Vec::new(),
            items: Vec::new(),
        };

        // (cell, segment) memberships, generated in parallel for big inputs.
        let pairs: Vec<(u32, u32)> = if segs.len() >= PAR_MIN {
            segs.par_chunks(2048)
                .enumerate()
                .flat_map_iter(|(ci, chunk)| {
                    let mut v = Vec::with_capacity(chunk.len() * 3);
                    for (k, s) in chunk.iter().enumerate() {
                        let id = (ci * 2048 + k) as u32;
                        g.cover(s, |c| v.push((c as u32, id)));
                    }
                    v
                })
                .collect()
        } else {
            let mut v = Vec::with_capacity(segs.len() * 3);
            for (i, s) in segs.iter().enumerate() {
                g.cover(s, |c| v.push((c as u32, i as u32)));
            }
            v
        };

        // Counting sort into CSR.
        let ncell = nx * ny;
        let mut start = vec![0u32; ncell + 1];
        for &(c, _) in &pairs {
            start[c as usize + 1] += 1;
        }
        for i in 0..ncell {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; pairs.len()];
        for &(c, s) in &pairs {
            items[fill[c as usize] as usize] = s;
            fill[c as usize] += 1;
        }
        g.start = start;
        g.items = items;
        g
    }

    #[inline]
    fn col(&self, x: f64) -> usize {
        ((x - self.ox) * self.inv).floor().clamp(0.0, (self.nx - 1) as f64) as usize
    }

    #[inline]
    fn row(&self, y: f64) -> usize {
        ((y - self.oy) * self.inv).floor().clamp(0.0, (self.ny - 1) as f64) as usize
    }

    #[inline]
    fn cell_of(&self, q: IP) -> usize {
        self.row(q.y as f64) * self.nx + self.col(q.x as f64)
    }

    #[inline]
    fn cell(&self, c: usize) -> &[u32] {
        &self.items[self.start[c] as usize..self.start[c + 1] as usize]
    }

    #[inline]
    fn contains(&self, q: IP) -> bool {
        q.x >= self.bbox[0] && q.y >= self.bbox[1] && q.x <= self.bbox[2] && q.y <= self.bbox[3]
    }

    /// Calls `f` for every cell touched by the inflated segment, column by column.
    fn cover(&self, s: &Seg, mut f: impl FnMut(usize)) {
        let (ax, ay, bx, by) = (s.a.x as f64, s.a.y as f64, s.b.x as f64, s.b.y as f64);
        let (lx, hx) = (ax.min(bx), ax.max(bx));
        let slope = if hx > lx { (by - ay) / (bx - ax) } else { 0.0 };
        for cx in self.col(lx - MARGIN)..=self.col(hx + MARGIN) {
            let colx0 = self.ox + cx as f64 * self.cs;
            let (ya, yb) = if hx > lx {
                let xa = (colx0 - MARGIN).clamp(lx, hx);
                let xb = (colx0 + self.cs + MARGIN).clamp(lx, hx);
                (ay + slope * (xa - ax), ay + slope * (xb - ax))
            } else {
                (ay, by)
            };
            let r0 = self.row(ya.min(yb) - MARGIN);
            let r1 = self.row(ya.max(yb) + MARGIN);
            for cy in r0..=r1 {
                f(cy * self.nx + cx);
            }
        }
    }
}

/// Splits `0..n` into roughly `parts` contiguous ranges.
fn ranges(n: usize, parts: usize) -> impl Iterator<Item = Range<usize>> {
    let step = n.div_ceil(parts.max(1)).max(1);
    (0..n).step_by(step).map(move |s| s..(s + step).min(n))
}

/// All proper crossings between segments, rounded to the grid. Each crossing is
/// reported once: only by the cell that contains its rounded point.
fn crossings(segs: &[Seg], g: &Grid) -> Vec<IP> {
    let work = |cells: Range<usize>| {
        let mut out = Vec::new();
        for c in cells {
            let it = g.cell(c);
            for (i, &si) in it.iter().enumerate() {
                let s = segs[si as usize];
                for &sj in &it[i + 1..] {
                    let t = segs[sj as usize];
                    if let Some(h) = crossing_point(s.a, s.b, t.a, t.b)
                        && g.cell_of(h) == c
                    {
                        out.push(h);
                    }
                }
            }
        }
        out
    };
    let ncell = g.nx * g.ny;
    if segs.len() < PAR_MIN {
        work(0..ncell)
    } else {
        let parts: Vec<Range<usize>> = ranges(ncell, 256).collect();
        parts.into_par_iter().flat_map_iter(work).collect()
    }
}

/// For every hot pixel, the segments that pass through it without having it
/// as an endpoint: `(segment index, hot pixel)`.
fn touch_splits(segs: &[Seg], g: &Grid, hot: &[IP]) -> Vec<(u32, IP)> {
    let work = |hs: &[IP]| {
        let mut out = Vec::new();
        for &h in hs {
            if !g.contains(h) {
                continue;
            }
            for &si in g.cell(g.cell_of(h)) {
                let s = &segs[si as usize];
                if h != s.a && h != s.b && touches(s.a, s.b, h) {
                    out.push((si, h));
                }
            }
        }
        out
    };
    if hot.len() < PAR_MIN {
        work(hot)
    } else {
        hot.par_chunks(1024).flat_map_iter(work).collect()
    }
}

/// Re-routes segments through their hot pixels. Returns `(unsplit, fragments)`.
fn apply_splits(segs: &[Seg], splits: Vec<(u32, IP)>) -> (Vec<Seg>, Vec<Seg>) {
    // Order split points along each segment by their projection onto it.
    let mut keyed: Vec<(u32, i64, IP)> = splits
        .into_iter()
        .map(|(si, h)| {
            let s = &segs[si as usize];
            let proj = (h.x - s.a.x) * (s.b.x - s.a.x) + (h.y - s.a.y) * (s.b.y - s.a.y);
            (si, proj, h)
        })
        .collect();
    keyed.sort_unstable_by_key(|&(si, proj, h)| (si, proj, h.x, h.y));
    let mut clean = Vec::with_capacity(segs.len());
    let mut frags = Vec::with_capacity(keyed.len() * 3);
    let mut k = 0;
    for (i, s) in segs.iter().enumerate() {
        if k >= keyed.len() || keyed[k].0 != i as u32 {
            clean.push(*s);
            continue;
        }
        let mut prev = s.a;
        while k < keyed.len() && keyed[k].0 == i as u32 {
            let h = keyed[k].2;
            if h != prev {
                frags.push(Seg { a: prev, b: h, ..*s });
                prev = h;
            }
            k += 1;
        }
        if prev != s.b {
            frags.push(Seg { a: prev, b: s.b, ..*s });
        }
    }
    (clean, frags)
}

/// Iterated snap rounding: returns segments that pairwise meet only at shared
/// endpoints or coincide exactly, and pass through no other vertex.
fn snap_round(segs: Vec<Seg>) -> Vec<Seg> {
    if segs.is_empty() {
        return segs;
    }
    // Pass 1: hot pixels are all vertices plus all rounded crossings.
    let g = Grid::build(&segs);
    let mut hot: Vec<IP> = segs.iter().map(|s| s.a).collect();
    hot.extend(crossings(&segs, &g));
    let splits = touch_splits(&segs, &g, &hot);
    if splits.is_empty() {
        return segs;
    }
    let (mut done, mut dirty) = apply_splits(&segs, splits);
    // Later passes create no new vertices, so only freshly created fragments
    // can pass through a hot pixel; re-check just those.
    for _ in 0..MAX_ROUNDS {
        if dirty.is_empty() {
            break;
        }
        let g = Grid::build(&dirty);
        let splits = touch_splits(&dirty, &g, &hot);
        if splits.is_empty() {
            break;
        }
        let (clean, frags) = apply_splits(&dirty, splits);
        done.extend(clean);
        dirty = frags;
    }
    done.extend(dirty);
    done
}

// ---------------------------------------------------------------------------
// Planar graph, winding sweep, extraction
// ---------------------------------------------------------------------------

/// Undirected graph edge `lo → hi` (vertex ids in sweep order) with summed
/// winding contributions for that direction.
#[derive(Clone, Copy, Debug)]
struct Edge {
    lo: u32,
    hi: u32,
    wa: i32,
    wb: i32,
}

struct Graph {
    verts: Vec<IP>,
    edges: Vec<Edge>,
}

fn build_graph(segs: &[Seg]) -> Graph {
    let mut verts: Vec<IP> = Vec::with_capacity(segs.len() * 2);
    for s in segs {
        verts.push(s.a);
        verts.push(s.b);
    }
    verts.sort_unstable_by_key(|q| q.key());
    verts.dedup();
    let id = |q: IP| verts.binary_search_by_key(&q.key(), |v| v.key()).unwrap() as u32;
    let mut edges: Vec<Edge> = segs
        .iter()
        .map(|s| {
            let (ia, ib) = (id(s.a), id(s.b));
            if ia < ib {
                Edge { lo: ia, hi: ib, wa: s.wa, wb: s.wb }
            } else {
                Edge { lo: ib, hi: ia, wa: -s.wa, wb: -s.wb }
            }
        })
        .collect();
    edges.sort_unstable_by_key(|e| ((e.lo as u64) << 32) | e.hi as u64);
    // Merge coincident edges; drop those whose contributions cancel.
    let mut merged: Vec<Edge> = Vec::with_capacity(edges.len());
    for e in edges {
        if let Some(l) = merged.last_mut()
            && l.lo == e.lo
            && l.hi == e.hi
        {
            l.wa += e.wa;
            l.wb += e.wb;
            continue;
        }
        merged.push(e);
    }
    merged.retain(|e| e.wa != 0 || e.wb != 0);
    Graph { verts, edges: merged }
}

/// Winding numbers `(w_a, w_b)` just left of each edge (on the side where
/// `orient(lo, hi, ·) > 0`); the right side is that plus the edge's
/// contribution.
fn sweep_windings(g: &Graph) -> Vec<(i32, i32)> {
    let (verts, edges) = (&g.verts, &g.edges);
    let mut indeg = vec![0u32; verts.len()];
    for e in edges {
        indeg[e.hi as usize] += 1;
    }
    let mut left = vec![(0i32, 0i32); edges.len()];
    let mut status: Vec<u32> = Vec::new();
    let mut fresh: Vec<u32> = Vec::new();
    let mut k = 0usize;
    for v in 0..verts.len() as u32 {
        let k0 = k;
        while k < edges.len() && edges[k].lo == v {
            k += 1;
        }
        if k0 == k && indeg[v as usize] == 0 {
            continue;
        }
        let vp = verts[v as usize];
        // Active edges strictly left of v come first in the status.
        let left_of_v = |&ei: &u32| {
            let e = &edges[ei as usize];
            e.hi != v && orient(verts[e.lo as usize], verts[e.hi as usize], vp) < 0
        };
        let mut pos = status.partition_point(left_of_v);
        let mut end = pos;
        while end < status.len() && edges[status[end] as usize].hi == v {
            end += 1;
        }
        if (end - pos) as u32 != indeg[v as usize] {
            // Unreachable for a planar graph; recover rather than corrupt state.
            status.retain(|&ei| edges[ei as usize].hi != v);
            pos = status.partition_point(left_of_v);
            end = pos;
        }
        let (mut wa, mut wb) = match pos.checked_sub(1) {
            Some(i) => {
                let ei = status[i] as usize;
                (left[ei].0 + edges[ei].wa, left[ei].1 + edges[ei].wb)
            }
            None => (0, 0),
        };
        // Edges starting at v, ordered left to right by direction.
        fresh.clear();
        fresh.extend(k0 as u32..k as u32);
        if fresh.len() > 1 {
            let dir = |ei: u32| {
                let e = &edges[ei as usize];
                let h = verts[e.hi as usize];
                (h.x - vp.x, h.y - vp.y)
            };
            fresh.sort_by(|&i, &j| {
                let (di, dj) = (dir(i), dir(j));
                0.cmp(&cross(dj.0, dj.1, di.0, di.1))
            });
        }
        for &ei in &fresh {
            left[ei as usize] = (wa, wb);
            wa += edges[ei as usize].wa;
            wb += edges[ei as usize].wb;
        }
        status.splice(pos..end, fresh.iter().copied());
    }
    left
}

/// Runs the whole pipeline with an inside predicate on `(w_a, w_b)`.
fn run(segs: Vec<Seg>, inside: impl Fn(i32, i32) -> bool) -> Vec<Poly> {
    if segs.is_empty() {
        return Vec::new();
    }
    let segs = snap_round(segs);
    let g = build_graph(&segs);
    let left = sweep_windings(&g);
    // Keep boundary edges, directed so the result lies on their positive side.
    let mut kept: Vec<(u32, u32)> = Vec::new();
    for (e, &(la, lb)) in g.edges.iter().zip(&left) {
        let il = inside(la, lb);
        if il != inside(la + e.wa, lb + e.wb) {
            kept.push(if il { (e.lo, e.hi) } else { (e.hi, e.lo) });
        }
    }
    chain(&g.verts, kept)
        .into_iter()
        .filter_map(|ring| {
            let ring = clean_ring(ring.into_iter().map(|v| g.verts[v as usize]).collect());
            if ring.len() < 3 || ring_area2(&ring) == 0 {
                return None;
            }
            let pts = ring
                .iter()
                .map(|q| p((q.x as f64 / SCALE) as f32, (q.y as f64 / SCALE) as f32))
                .collect::<Vec<P>>();
            Some(Poly { pts, closed: true })
        })
        .collect()
}

/// Turn comparison for chaining: `Greater` if `d1` turns further towards the
/// positive (inside) side relative to incoming direction `din` than `d2`.
fn turn_cmp(din: (i64, i64), d1: (i64, i64), d2: (i64, i64)) -> Ordering {
    let upper = |d: (i64, i64)| {
        let c = cross(din.0, din.1, d.0, d.1);
        c > 0 || (c == 0 && din.0 * d.0 + din.1 * d.1 < 0)
    };
    let (h1, h2) = (upper(d1), upper(d2));
    if h1 != h2 {
        return h1.cmp(&h2);
    }
    cross(d2.0, d2.1, d1.0, d1.1).cmp(&0)
}

/// Links directed boundary edges into closed vertex-id rings, taking the
/// sharpest turn towards the inside at vertices with several continuations.
fn chain(verts: &[IP], mut kept: Vec<(u32, u32)>) -> Vec<Vec<u32>> {
    kept.sort_unstable_by_key(|&(f, t)| ((f as u64) << 32) | t as u64);
    let mut start = vec![0u32; verts.len() + 1];
    for &(f, _) in &kept {
        start[f as usize + 1] += 1;
    }
    for i in 0..verts.len() {
        start[i + 1] += start[i];
    }
    let dir = |e: usize| {
        let (f, t) = kept[e];
        let (a, b) = (verts[f as usize], verts[t as usize]);
        (b.x - a.x, b.y - a.y)
    };
    let next = |e: usize| {
        let v = kept[e].1 as usize;
        let (s, t) = (start[v] as usize, start[v + 1] as usize);
        if t - s <= 1 {
            return s; // s == t only for malformed input; caught by the caller.
        }
        let din = dir(e);
        (s + 1..t).fold(s, |best, c| if turn_cmp(din, dir(c), dir(best)) == Ordering::Greater { c } else { best })
    };
    let mut used = vec![false; kept.len()];
    let mut rings = Vec::new();
    for s in 0..kept.len() {
        if used[s] {
            continue;
        }
        let mut ring = Vec::new();
        let mut e = s;
        loop {
            used[e] = true;
            ring.push(kept[e].0);
            let n = next(e);
            if n == s || n >= kept.len() || used[n] || kept[n].0 != kept[e].1 {
                break;
            }
            e = n;
        }
        rings.push(ring);
    }
    rings
}

/// Removes repeated and collinear vertices (including across the wrap-around).
fn clean_ring(pts: Vec<IP>) -> Vec<IP> {
    let mut out: Vec<IP> = Vec::with_capacity(pts.len());
    for q in pts {
        while out.len() >= 2 && orient(out[out.len() - 2], out[out.len() - 1], q) == 0 {
            out.pop();
        }
        if out.last() != Some(&q) {
            out.push(q);
        }
    }
    loop {
        let n = out.len();
        if n < 3 {
            break;
        }
        if orient(out[n - 2], out[n - 1], out[0]) == 0 {
            out.pop();
        } else if orient(out[n - 1], out[0], out[1]) == 0 {
            out.remove(0);
        } else {
            break;
        }
    }
    out
}

/// Twice the signed area of an integer ring.
fn ring_area2(ring: &[IP]) -> i128 {
    let o = ring[0];
    let mut s = 0i128;
    for i in 1..ring.len() - 1 {
        let (a, b) = (ring[i], ring[i + 1]);
        s += cross(a.x - o.x, a.y - o.y, b.x - o.x, b.y - o.y) as i128;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    const OPS: [BoolOp; 4] = [BoolOp::Union, BoolOp::Subtract, BoolOp::Intersect, BoolOp::Xor];
    const NZ: FillRule = FillRule::NonZero;
    const EO: FillRule = FillRule::EvenOdd;
    /// Supersamples per pixel along each axis for the reference rasterizer.
    const SS: usize = 16;
    /// Generic sub-pixel offsets of the sample grid. They keep sample centres
    /// away from the 1/1024 px snapping lattice and from lines of small
    /// rational slope through lattice points; otherwise a long edge displaced
    /// by sub-1/1024 px snap rounding could flip a whole run of samples that
    /// happen to sit (almost) exactly on it.
    const OFF_X: f64 = 0.0123;
    const OFF_Y: f64 = 0.0071;

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn poly(pts: &[(f32, f32)]) -> Poly {
        Poly { pts: pts.iter().map(|&(x, y)| p(x, y)).collect(), closed: true }
    }

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Poly {
        poly(&[(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    }

    fn circle(cx: f64, cy: f64, r: f64, n: usize) -> Poly {
        let pts = (0..n)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / n as f64;
                p((cx + r * t.cos()) as f32, (cy + r * t.sin()) as f32)
            })
            .collect();
        Poly { pts, closed: true }
    }

    /// Five-pointed star drawn as one self-intersecting pentagram. Slightly
    /// rotated: a long exactly horizontal edge would make the point-sampled
    /// reference area itself inaccurate (midpoint-rule error ≈ length / 2SS).
    fn pentagram(cx: f64, cy: f64, r: f64) -> Poly {
        let pts = (0..5)
            .map(|i| {
                let t = std::f64::consts::TAU * (2 * i) as f64 / 5.0 - std::f64::consts::FRAC_PI_2 + 0.1;
                p((cx + r * t.cos()) as f32, (cy + r * t.sin()) as f32)
            })
            .collect();
        Poly { pts, closed: true }
    }

    /// Star-ish polygon: alternating spikes with radial noise.
    fn noisy_star(rng: &mut Rng, cx: f64, cy: f64, r: f64, n: usize) -> Poly {
        let pts = (0..n)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / n as f64;
                let base = if i % 2 == 0 { 1.0 } else { 0.7 };
                let rr = r * base * (0.8 + 0.4 * rng.f());
                p((cx + rr * t.cos()) as f32, (cy + rr * t.sin()) as f32)
            })
            .collect();
        Poly { pts, closed: true }
    }

    fn random_poly(rng: &mut Rng, x0: f64, y0: f64, size: f64, n: usize) -> Poly {
        let pts = (0..n).map(|_| p((x0 + size * rng.f()) as f32, (y0 + size * rng.f()) as f32)).collect();
        Poly { pts, closed: true }
    }

    fn op_eval(op: BoolOp, a: bool, b: bool) -> bool {
        match op {
            BoolOp::Union => a || b,
            BoolOp::Subtract => a && !b,
            BoolOp::Intersect => a && b,
            BoolOp::Xor => a != b,
        }
    }

    /// Integer pixel window covering all inputs.
    #[derive(Clone, Copy)]
    struct Win {
        x0: f64,
        y0: f64,
        w: usize,
        h: usize,
    }

    fn window(sets: &[&[Poly]]) -> Win {
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for q in sets.iter().flat_map(|s| s.iter()).flat_map(|pl| pl.pts.iter()) {
            x0 = x0.min(q.x as f64);
            y0 = y0.min(q.y as f64);
            x1 = x1.max(q.x as f64);
            y1 = y1.max(q.y as f64);
        }
        if x0 > x1 {
            return Win { x0: 0.0, y0: 0.0, w: 1, h: 1 };
        }
        let (x0, y0) = (x0.floor() - 1.0, y0.floor() - 1.0);
        Win { x0, y0, w: (x1.ceil() + 1.0 - x0) as usize, h: (y1.ceil() + 1.0 - y0) as usize }
    }

    /// Sampled coverage: for every sample row, the sorted sample-column indices
    /// at which the inside state toggles (starting outside). Equivalent to a
    /// bitmap of SS×SS samples per pixel, but cheap to combine and count.
    type Raster = Vec<Vec<u32>>;

    /// Reference rasterizer: SS×SS point samples per pixel at sample centres,
    /// exact winding per sample via a scanline crossing list.
    fn raster(polys: &[Poly], rule: FillRule, win: Win) -> Raster {
        let (sw, sh) = (win.w * SS, win.h * SS);
        let mut xs: Vec<(f64, i32)> = Vec::new();
        let mut rows = Vec::with_capacity(sh);
        for r in 0..sh {
            let y = win.y0 + (r as f64 + 0.5) / SS as f64 + OFF_Y;
            xs.clear();
            for pl in polys {
                let n = pl.pts.len();
                for i in 0..n {
                    let (a, b) = (pl.pts[i], pl.pts[(i + 1) % n]);
                    let (ay, by) = (a.y as f64, b.y as f64);
                    if (ay <= y) != (by <= y) {
                        let t = (y - ay) / (by - ay);
                        xs.push((a.x as f64 + t * (b.x as f64 - a.x as f64), if by > ay { 1 } else { -1 }));
                    }
                }
            }
            xs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut toggles: Vec<u32> = Vec::new();
            let (mut w, mut inside) = (0, false);
            for &(x, d) in &xs {
                w += d;
                if filled(rule, w) != inside {
                    inside = !inside;
                    // First sample column whose centre lies strictly right of x.
                    let c = (((x - win.x0 - OFF_X) * SS as f64 - 0.5).floor() + 1.0).clamp(0.0, sw as f64) as u32;
                    if toggles.last() == Some(&c) {
                        toggles.pop();
                    } else {
                        toggles.push(c);
                    }
                }
            }
            rows.push(toggles);
        }
        rows
    }

    /// Per-sample combination of two rasters (`f(false, false)` must be false).
    fn combine(a: &Raster, b: &Raster, f: impl Fn(bool, bool) -> bool) -> Raster {
        a.iter()
            .zip(b)
            .map(|(ra, rb)| {
                let (mut i, mut j, mut sa, mut sb, mut cur) = (0, 0, false, false, false);
                let mut out = Vec::new();
                while i < ra.len() || j < rb.len() {
                    let pos = ra.get(i).copied().unwrap_or(u32::MAX).min(rb.get(j).copied().unwrap_or(u32::MAX));
                    while i < ra.len() && ra[i] == pos {
                        sa = !sa;
                        i += 1;
                    }
                    while j < rb.len() && rb[j] == pos {
                        sb = !sb;
                        j += 1;
                    }
                    if f(sa, sb) != cur {
                        cur = !cur;
                        out.push(pos);
                    }
                }
                out
            })
            .collect()
    }

    /// Covered area in px² (number of inside samples / SS²).
    fn sample_area(r: &Raster) -> f64 {
        let n: u64 = r.iter().map(|t| t.chunks(2).map(|c| (c[1] - c[0]) as u64).sum::<u64>()).sum();
        n as f64 / (SS * SS) as f64
    }

    fn orient_f(a: P, b: P, c: P) -> f64 {
        // Exact sign: f32 differences and their products are exact in f64.
        let (abx, aby) = (b.x as f64 - a.x as f64, b.y as f64 - a.y as f64);
        let (acx, acy) = (c.x as f64 - a.x as f64, c.y as f64 - a.y as f64);
        abx * acy - aby * acx
    }

    /// Asserts that no two result edges cross properly (touching is allowed).
    fn assert_simple(name: &str, res: &[Poly]) {
        let mut edges: Vec<(P, P)> = Vec::new();
        for pl in res {
            assert!(pl.closed, "{name}: open contour");
            assert!(pl.pts.len() >= 3, "{name}: degenerate contour");
            for i in 0..pl.pts.len() {
                edges.push((pl.pts[i], pl.pts[(i + 1) % pl.pts.len()]));
            }
        }
        edges.sort_by(|a, b| a.0.x.min(a.1.x).total_cmp(&b.0.x.min(b.1.x)));
        for i in 0..edges.len() {
            let (a, b) = edges[i];
            let maxx = a.x.max(b.x);
            for &(c, d) in &edges[i + 1..] {
                if c.x.min(d.x) > maxx {
                    break;
                }
                if a.y.max(b.y) < c.y.min(d.y) || c.y.max(d.y) < a.y.min(b.y) {
                    continue;
                }
                let (o1, o2) = (orient_f(a, b, c), orient_f(a, b, d));
                let (o3, o4) = (orient_f(c, d, a), orient_f(c, d, b));
                let proper = o1 * o2 < 0.0 && o3 * o4 < 0.0;
                assert!(!proper, "{name}: result edges cross: {a:?}-{b:?} x {c:?}-{d:?}");
            }
        }
    }

    /// Rounds coordinates to the module's documented 1/1024 px precision.
    fn quantize(polys: &[Poly]) -> Vec<Poly> {
        let q = |v: f32| ((v as f64 * SCALE).round() / SCALE) as f32;
        polys
            .iter()
            .map(|pl| Poly { pts: pl.pts.iter().map(|v| p(q(v.x), q(v.y))).collect(), closed: pl.closed })
            .collect()
    }

    /// Checks all four ops against the reference rasterizer (which samples the
    /// operands at the module's 1/1024 px input precision).
    fn check(name: &str, a: &[Poly], ra: FillRule, b: &[Poly], rb: FillRule) {
        let win = window(&[a, b]);
        let (ma, mb) = (raster(&quantize(a), ra, win), raster(&quantize(b), rb, win));
        let tol = 1e-3 * sample_area(&ma).max(sample_area(&mb)) + 1e-9;
        for op in OPS {
            let res = boolean(a, ra, b, rb, op);
            let reference = combine(&ma, &mb, |x, y| op_eval(op, x, y));
            let ref_area = sample_area(&reference);
            let got = area(&res);
            let mismatch = sample_area(&combine(&raster(&res, NZ, win), &reference, |x, y| x != y));
            assert!(
                (got - ref_area).abs() <= tol,
                "{name} {op:?}: area {got} vs reference {ref_area} (tol {tol}, raster mismatch {mismatch})"
            );
            assert!(mismatch <= tol, "{name} {op:?}: raster mismatch {mismatch} px² (tol {tol})");
            // Even-odd fill of the result must agree too (no overlapping contours).
            let mismatch_eo = sample_area(&combine(&raster(&res, EO, win), &reference, |x, y| x != y));
            assert!(mismatch_eo <= tol, "{name} {op:?}: even-odd raster mismatch {mismatch_eo} px²");
            assert_simple(&format!("{name} {op:?}"), &res);
        }
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6 * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn rect_rect_overlap_exact() {
        let a = [rect(0.0, 0.0, 10.0, 10.0)];
        let b = [rect(5.0, 5.0, 15.0, 15.0)];
        assert!(approx(area(&boolean(&a, NZ, &b, NZ, BoolOp::Union)), 175.0));
        assert!(approx(area(&boolean(&a, NZ, &b, NZ, BoolOp::Subtract)), 75.0));
        assert!(approx(area(&boolean(&a, NZ, &b, NZ, BoolOp::Intersect)), 25.0));
        assert!(approx(area(&boolean(&a, NZ, &b, NZ, BoolOp::Xor)), 150.0));
        check("rect/rect", &a, NZ, &b, NZ);
        // Opposite orientations of the operands must not matter.
        let mut br = b.clone();
        br[0].pts.reverse();
        check("rect/rect reversed", &a, NZ, &br, EO);
    }

    #[test]
    fn rect_inside_rect() {
        let a = [rect(0.0, 0.0, 20.0, 20.0)];
        let b = [rect(5.0, 5.0, 10.0, 10.0)];
        let sub = boolean(&a, NZ, &b, NZ, BoolOp::Subtract);
        assert_eq!(sub.len(), 2);
        assert!(approx(area(&sub), 375.0));
        // One positive outer and one negative hole.
        assert_eq!(sub.iter().filter(|pl| area(std::slice::from_ref(pl)) < 0.0).count(), 1);
        check("rect in rect", &a, NZ, &b, NZ);
        check("rect in rect swapped", &b, NZ, &a, NZ);
    }

    #[test]
    fn disjoint() {
        let a = [rect(0.0, 0.0, 10.0, 10.0)];
        let b = [circle(60.0, 60.0, 30.0, 64)];
        check("disjoint", &a, NZ, &b, NZ);
        assert!(boolean(&a, NZ, &b, NZ, BoolOp::Intersect).is_empty());
    }

    #[test]
    fn circles_overlap() {
        let a = [circle(50.0, 50.0, 40.0, 256)];
        let b = [circle(80.0, 60.0, 35.0, 256)];
        check("circles", &a, NZ, &b, NZ);
        check("circles eo", &a, EO, &b, EO);
        // Fractional centres → vertices off the grid.
        let a = [circle(50.3, 50.7, 40.1, 256)];
        let b = [circle(50.3, 50.7, 40.1, 255)];
        check("near-identical circles", &a, NZ, &b, NZ);
    }

    #[test]
    fn pentagram_both_rules() {
        let star = [pentagram(50.0, 50.0, 45.0)];
        let nz = simplify(&star, NZ);
        let eo = simplify(&star, EO);
        // Even-odd leaves the central pentagon empty.
        assert!(area(&nz) > area(&eo) + 100.0);
        assert_eq!(eo.len(), 5);
        let c = [circle(60.0, 40.0, 25.0, 256)];
        let r = [rect(20.0, 45.0, 90.0, 60.0)];
        for ra in [NZ, EO] {
            check("pentagram/circle", &star, ra, &c, NZ);
            check("rect/pentagram", &r, NZ, &star, ra);
            check("pentagram/pentagram", &star, ra, &[pentagram(55.0, 52.0, 40.0)], EO);
        }
    }

    #[test]
    fn figure_eight_and_bowtie() {
        let bow = [poly(&[(0.0, 0.0), (20.0, 20.0), (20.0, 0.0), (0.0, 20.0)])];
        for rule in [NZ, EO] {
            let s = simplify(&bow, rule);
            assert_eq!(s.len(), 2);
            assert!(approx(area(&s), 200.0));
        }
        // Figure-8 drawn with one loop each way.
        let eight = [poly(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0), (0.0, 10.0)])];
        check("figure-8", &eight, NZ, &[rect(5.0, 5.0, 15.0, 15.0)], NZ);
        check("bowtie", &bow, EO, &[circle(10.0, 10.0, 7.0, 256)], NZ);
    }

    #[test]
    fn random_noisy_stars() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for i in 0..4 {
            let a = [noisy_star(&mut rng, 100.0, 100.0, 90.0, 500)];
            let b = [noisy_star(&mut rng, 120.0 + 10.0 * i as f64, 110.0, 85.0, 500)];
            check("noisy stars", &a, NZ, &b, NZ);
            check("noisy stars eo", &a, EO, &b, EO);
        }
    }

    #[test]
    fn random_self_intersecting() {
        let mut rng = Rng(12345);
        let iters = std::env::var("PATHOPS_FUZZ").ok().and_then(|v| v.parse::<usize>().ok()).map_or(4, |n| n / 10 + 1);
        for _ in 0..iters {
            let a = [random_poly(&mut rng, 0.0, 0.0, 100.0, 60)];
            let b = [random_poly(&mut rng, 20.0, 10.0, 100.0, 60)];
            check("random self-intersecting nz", &a, NZ, &b, NZ);
            check("random self-intersecting eo", &a, EO, &b, NZ);
        }
    }

    #[test]
    fn lattice_degenerate_fuzz() {
        // Vertices on a coarse lattice: lots of shared vertices, collinear
        // overlaps, T-junctions and repeated edges.
        let mut rng = Rng(0xC0FFEE);
        let mut lattice = |n: usize| {
            let pts = (0..n)
                .map(|_| p((rng.f() * 9.0).floor() as f32 * 40.0, (rng.f() * 9.0).floor() as f32 * 40.0))
                .collect();
            Poly { pts, closed: true }
        };
        // PATHOPS_FUZZ=<n> raises the iteration count for longer fuzzing runs.
        let iters = std::env::var("PATHOPS_FUZZ").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
        for i in 0..iters {
            let a = vec![lattice(4 + i % 9), lattice(3 + i % 5)];
            let b = vec![lattice(4 + i % 7)];
            check("lattice nz", &a, NZ, &b, NZ);
            check("lattice eo", &a, EO, &b, if i % 2 == 0 { EO } else { NZ });
        }
    }

    #[test]
    fn shared_edge_and_partial_overlap() {
        let a = [rect(0.0, 0.0, 10.0, 10.0)];
        let b = [rect(10.0, 0.0, 20.0, 10.0)];
        let u = boolean(&a, NZ, &b, NZ, BoolOp::Union);
        assert_eq!(u.len(), 1);
        assert_eq!(u[0].pts.len(), 4, "collinear vertices should be merged: {:?}", u[0].pts);
        assert!(approx(area(&u), 200.0));
        check("shared edge", &a, NZ, &b, NZ);
        // Partially overlapping collinear edges and T-junctions.
        check("partial overlap", &a, NZ, &[rect(10.0, 3.0, 20.0, 15.0)], NZ);
        check("t-junction", &a, NZ, &[rect(2.0, 10.0, 8.0, 20.0)], NZ);
        check("collinear inside", &a, NZ, &[rect(0.0, 2.0, 5.0, 8.0)], NZ);
    }

    #[test]
    fn touching_at_vertex() {
        let a = [rect(0.0, 0.0, 10.0, 10.0)];
        let b = [rect(10.0, 10.0, 20.0, 20.0)];
        let u = boolean(&a, NZ, &b, NZ, BoolOp::Union);
        assert_eq!(u.len(), 2, "regions touching at a point stay separate contours");
        assert!(approx(area(&u), 200.0));
        assert!(boolean(&a, NZ, &b, NZ, BoolOp::Intersect).is_empty());
        check("touch at vertex", &a, NZ, &b, NZ);
        // Vertex of one touching the edge interior of the other.
        let d = [poly(&[(10.0, 5.0), (20.0, 0.0), (20.0, 10.0)])];
        check("vertex on edge", &a, NZ, &d, NZ);
    }

    #[test]
    fn far_from_origin() {
        let a = [rect(100_000.0, 50_000.0, 100_040.0, 50_030.0)];
        let b = [rect(100_020.0, 50_010.0, 100_060.0, 50_050.0)];
        assert!(approx(area(&boolean(&a, NZ, &b, NZ, BoolOp::Union)), 2400.0));
        check("far rects", &a, NZ, &b, NZ);
        // Out-of-range coordinates are clamped (±2^19 px), not overflowed.
        let huge = [rect(-1e7, -1e7, 1e7, 1e7)];
        let small = [rect(0.0, 0.0, 10.0, 10.0)];
        assert!(approx(area(&boolean(&huge, NZ, &small, NZ, BoolOp::Intersect)), 100.0));
        assert!(approx(area(&boolean(&huge, NZ, &small, NZ, BoolOp::Subtract)), 1048576.0f64.powi(2) - 100.0));
        let nan = [poly(&[(f32::NAN, 0.0), (5.0, f32::INFINITY), (3.0, 3.0)])];
        let _ = boolean(&nan, NZ, &small, NZ, BoolOp::Union);
    }

    #[test]
    fn identical_inputs() {
        let mut rng = Rng(777);
        let shapes: Vec<Vec<Poly>> = vec![
            vec![rect(0.0, 0.0, 10.0, 10.0)],
            vec![circle(30.0, 30.0, 20.0, 256)],
            vec![noisy_star(&mut rng, 100.0, 100.0, 80.0, 500)],
            vec![pentagram(50.0, 50.0, 40.0)],
        ];
        for s in &shapes {
            for rule in [NZ, EO] {
                let base = area(&simplify(s, rule));
                assert!(approx(area(&boolean(s, rule, s, rule, BoolOp::Union)), base));
                assert!(approx(area(&boolean(s, rule, s, rule, BoolOp::Intersect)), base));
                assert!(boolean(s, rule, s, rule, BoolOp::Subtract).is_empty());
                assert!(boolean(s, rule, s, rule, BoolOp::Xor).is_empty());
                check("identical", s, rule, s, rule);
            }
        }
    }

    #[test]
    fn shape_with_hole() {
        // NonZero hole: inner contour reversed.
        let mut inner = rect(10.0, 10.0, 30.0, 30.0);
        inner.pts.reverse();
        let ring_nz = [rect(0.0, 0.0, 40.0, 40.0), inner];
        assert!(approx(area(&simplify(&ring_nz, NZ)), 1200.0));
        // EvenOdd hole: same orientation.
        let ring_eo = [rect(0.0, 0.0, 40.0, 40.0), rect(10.0, 10.0, 30.0, 30.0)];
        assert!(approx(area(&simplify(&ring_eo, EO)), 1200.0));
        assert!(approx(area(&simplify(&ring_eo, NZ)), 1600.0));
        let c = [circle(25.0, 20.0, 12.0, 256)];
        check("hole nz", &ring_nz, NZ, &c, NZ);
        check("hole eo", &ring_eo, EO, &c, NZ);
        check("hole vs hole", &ring_nz, NZ, &ring_eo, EO);
        // Filling the hole exactly.
        let plug = [rect(10.0, 10.0, 30.0, 30.0)];
        let u = boolean(&ring_nz, NZ, &plug, NZ, BoolOp::Union);
        assert_eq!(u.len(), 1);
        assert!(approx(area(&u), 1600.0));
    }

    #[test]
    fn empty_and_open_inputs() {
        let a = [rect(0.0, 0.0, 10.0, 10.0)];
        let e: [Poly; 0] = [];
        assert!(boolean(&e, NZ, &e, NZ, BoolOp::Union).is_empty());
        assert!(approx(area(&boolean(&a, NZ, &e, NZ, BoolOp::Union)), 100.0));
        assert!(approx(area(&boolean(&e, NZ, &a, NZ, BoolOp::Xor)), 100.0));
        assert!(boolean(&e, NZ, &a, NZ, BoolOp::Subtract).is_empty());
        assert!(boolean(&a, NZ, &e, NZ, BoolOp::Intersect).is_empty());
        check("empty b", &a, NZ, &e, NZ);
        check("empty a", &e, NZ, &a, NZ);
        // Degenerate polys: empty, single point, two points, collinear.
        let junk = [
            Poly::default(),
            poly(&[(1.0, 1.0)]),
            poly(&[(1.0, 1.0), (5.0, 5.0)]),
            poly(&[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]),
        ];
        assert!(simplify(&junk, NZ).is_empty());
        // Open poly is treated as closed.
        let mut open = rect(0.0, 0.0, 10.0, 10.0);
        open.closed = false;
        assert!(approx(area(&simplify(&[open], NZ)), 100.0));
    }

    #[test]
    fn output_orientation_convention() {
        for reversed in [false, true] {
            let mut r = rect(0.0, 0.0, 10.0, 10.0);
            if reversed {
                r.pts.reverse();
            }
            let s = simplify(&[r], NZ);
            assert_eq!(s.len(), 1);
            assert!(area(&s) > 0.0, "outer contours have positive shoelace area");
        }
    }

    #[test]
    fn combine_sequence_matches_manual_fold() {
        let a = vec![rect(0.0, 0.0, 30.0, 30.0)];
        let b = vec![circle(30.0, 15.0, 10.0, 256)];
        let c = vec![rect(10.0, 10.0, 20.0, 20.0)];
        let d = vec![pentagram(15.0, 15.0, 8.0)];
        let seq = combine_sequence(&[
            (BoolOp::Intersect, a.clone(), NZ), // first op ignored
            (BoolOp::Union, b.clone(), NZ),
            (BoolOp::Subtract, c.clone(), NZ),
            (BoolOp::Xor, d.clone(), EO),
        ]);
        let manual = boolean(&a, NZ, &b, NZ, BoolOp::Union);
        let manual = boolean(&manual, NZ, &c, NZ, BoolOp::Subtract);
        let manual = boolean(&manual, NZ, &d, EO, BoolOp::Xor);
        assert!(approx(area(&seq), area(&manual)));
        assert!(area(&seq) > 900.0);
        assert!(combine_sequence(&[]).is_empty());
        assert!(approx(area(&combine_sequence(&[(BoolOp::Subtract, a, NZ)])), 900.0));
    }

    #[test]
    fn snap_rounding_yields_planar_graph() {
        let mut rng = Rng(424242);
        let iters = std::env::var("PATHOPS_FUZZ").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(5);
        for _ in 0..iters {
            let mut segs = Vec::new();
            push_polys(&[random_poly(&mut rng, 0.0, 0.0, 0.05, 40)], 1, 0, &mut segs);
            push_polys(&[random_poly(&mut rng, 0.0, 0.0, 0.05, 40)], 0, 1, &mut segs);
            let out = snap_round(segs);
            for (i, s) in out.iter().enumerate() {
                for t in &out[i + 1..] {
                    assert!(crossing_point(s.a, s.b, t.a, t.b).is_none(), "crossing remains");
                    for h in [t.a, t.b] {
                        assert!(h == s.a || h == s.b || !touches(s.a, s.b, h), "fragment passes a vertex {s:?} {h:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn tiny_scale_structural_fuzz() {
        // Inputs only a few snap units wide: snap rounding dominates. The exact
        // region necessarily changes, but the output must stay structurally
        // valid: simple, non-overlapping contours (NonZero fill == EvenOdd fill).
        let mut rng = Rng(0xABCDEF);
        let iters = std::env::var("PATHOPS_FUZZ").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
        for _ in 0..iters {
            let a = [random_poly(&mut rng, 0.0, 0.0, 0.03, 12), random_poly(&mut rng, 0.0, 0.0, 0.03, 5)];
            let b = [random_poly(&mut rng, 0.005, 0.0, 0.03, 12)];
            for op in OPS {
                for (ra, rb) in [(NZ, NZ), (EO, NZ), (EO, EO)] {
                    let res = boolean(&a, ra, &b, rb, op);
                    assert_simple("tiny", &res);
                    // Scale by 1024: output coordinates become integers (exact in f32).
                    let scaled: Vec<Poly> = res
                        .iter()
                        .map(|pl| Poly { pts: pl.pts.iter().map(|v| p(v.x * 1024.0, v.y * 1024.0)).collect(), closed: true })
                        .collect();
                    let win = window(&[&scaled]);
                    let diff = combine(&raster(&scaled, NZ, win), &raster(&scaled, EO, win), |x, y| x != y);
                    assert_eq!(sample_area(&diff), 0.0, "overlapping output contours");
                    assert!(area(&res) >= 0.0);
                }
            }
        }
    }

    /// Smooth closed curve with 500 anchors, Catmull-Rom interpolated and
    /// flattened to ~`per_span` segments per span.
    fn curvy_path(rng: &mut Rng, cx: f64, cy: f64, r: f64, per_span: usize) -> Poly {
        let anchors: Vec<(f64, f64)> = (0..500)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 500.0;
                let rr = r + 60.0 * (rng.f() - 0.5) + 40.0 * (7.0 * t).sin();
                (cx + rr * t.cos(), cy + rr * t.sin())
            })
            .collect();
        let n = anchors.len();
        let mut pts = Vec::with_capacity(n * per_span);
        for i in 0..n {
            let (p0, p1, p2, p3) = (anchors[(i + n - 1) % n], anchors[i], anchors[(i + 1) % n], anchors[(i + 2) % n]);
            for k in 0..per_span {
                let t = k as f64 / per_span as f64;
                let (t2, t3) = (t * t, t * t * t);
                let f = |a: f64, b: f64, c: f64, d: f64| {
                    0.5 * (2.0 * b + (-a + c) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2 + (-a + 3.0 * b - 3.0 * c + d) * t3)
                };
                pts.push(p(f(p0.0, p1.0, p2.0, p3.0) as f32, f(p0.1, p1.1, p2.1, p3.1) as f32));
            }
        }
        Poly { pts, closed: true }
    }

    /// Times all four ops on two large curved paths and validates the results
    /// via area identities and the simplicity check.
    fn perf_case(label: &str, a: &[Poly], b: &[Poly]) {
        let segs = |s: &[Poly]| s.iter().map(|pl| pl.pts.len()).sum::<usize>();
        println!("{label}: segments a = {}, b = {}", segs(a), segs(b));
        let (aa, ab) = (area(&simplify(a, NZ)), area(&simplify(b, NZ)));
        let mut areas = [0.0; 4];
        for (k, op) in OPS.into_iter().enumerate() {
            let _ = boolean(a, NZ, b, NZ, op); // warm-up
            let iters = 30;
            let mut times = Vec::with_capacity(iters);
            let mut res = Vec::new();
            for _ in 0..iters {
                let t = Instant::now();
                res = boolean(a, NZ, b, NZ, op);
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(f64::total_cmp);
            let nv: usize = res.iter().map(|pl| pl.pts.len()).sum();
            println!(
                "  {op:?}: median {:.2} ms, min {:.2} ms, max {:.2} ms; {} contours, {nv} vertices",
                times[iters / 2],
                times[0],
                times[iters - 1],
                res.len()
            );
            assert_simple(label, &res);
            areas[k] = area(&res);
        }
        let [u, s, i, x] = areas;
        let tol = 1e-6 * aa.max(ab);
        assert!((u - (aa + ab - i)).abs() < tol, "union area identity");
        assert!((s - (aa - i)).abs() < tol, "subtract area identity");
        assert!((x - (u - i)).abs() < tol, "xor area identity");
    }

    #[test]
    #[ignore = "timing; run with --release -- --ignored --nocapture"]
    fn perf_large_paths() {
        let mut rng = Rng(0xDEADBEEF);
        // 500 anchors × 12 = 6000 segments each; offset → few crossings.
        let a = [curvy_path(&mut rng, 2500.0, 2500.0, 2000.0, 12)];
        let b = [curvy_path(&mut rng, 2650.0, 2600.0, 2000.0, 12)];
        perf_case("offset curves", &a, &b);
        // Concentric, independently noisy → hundreds of crossings.
        let b2 = [curvy_path(&mut rng, 2500.0, 2500.0, 2000.0, 12)];
        perf_case("concentric curves", &a, &b2);
        // 8000 segments each.
        let a3 = [curvy_path(&mut rng, 2500.0, 2500.0, 2000.0, 16)];
        let b3 = [curvy_path(&mut rng, 2510.0, 2490.0, 2000.0, 16)];
        perf_case("8k segments", &a3, &b3);
    }
}
