//! Live-wire (intelligent scissors) for the Magnetic Lasso, and edge
//! following for Content-Aware Tracing.
//!
//! The local cost is low on strong edges: `1 - g'` where `g'` is the Sobel
//! gradient magnitude of luminance, normalized inside the search region and
//! thresholded by the Contrast option. The path from the last anchor to the
//! cursor is a Dijkstra shortest path restricted to a *band* of `width`
//! pixels around the cursor's trail (a distance transform of the rasterized
//! trail), so the search area is proportional to the trail length, never
//! the image. For long trails the search runs on a coarser mip level first
//! and is then refined at level 0 inside a band a few pixels wide around the
//! upsampled coarse path.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::document::Layer;
use super::grid::TileGrid;
use super::masks::edt;
use super::px;
use super::vector::{P, p};

#[derive(Clone, Copy, Debug)]
pub struct WireParams {
    /// Detection width (level-0 pixels).
    pub width: f32,
    /// Edge contrast threshold, 0..1.
    pub contrast: f32,
}

#[inline]
fn luma(c: u32) -> f32 {
    let [r, g, b, a] = px::unpack(c);
    // Composite over mid-gray so transparent regions have no false edges.
    let k = a as f32 / 255.0;
    let l = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
    l + 128.0 * (1.0 - k)
}

/// Edge cost (0 on the strongest edge) for a `w x h` region at `(x0, y0)`.
pub fn cost_region(grid: &TileGrid, x0: i32, y0: i32, w: usize, h: usize, contrast: f32) -> Vec<f32> {
    let (pw, ph) = (w + 2, h + 2);
    let mut buf = vec![0u32; pw * ph];
    grid.read_region_clamped(x0 - 1, y0 - 1, pw, ph, &mut buf);
    let l: Vec<f32> = buf.iter().map(|&c| luma(c)).collect();
    let mut g = vec![0f32; w * h];
    let mut gmax = 1e-3f32;
    for y in 0..h {
        for x in 0..w {
            let at = |dx: usize, dy: usize| l[(y + dy) * pw + x + dx];
            let gx = (at(2, 0) + 2.0 * at(2, 1) + at(2, 2)) - (at(0, 0) + 2.0 * at(0, 1) + at(0, 2));
            let gy = (at(0, 2) + 2.0 * at(1, 2) + at(2, 2)) - (at(0, 0) + 2.0 * at(1, 0) + at(2, 0));
            let m = (gx * gx + gy * gy).sqrt();
            g[y * w + x] = m;
            gmax = gmax.max(m);
        }
    }
    let thr = contrast.clamp(0.0, 0.95) * gmax;
    let span = (gmax - thr).max(1e-3);
    g.iter().map(|&m| 1.0 - ((m - thr) / span).clamp(0.0, 1.0)).collect()
}

/// Dijkstra on an 8-connected grid of costs, only through `allowed` pixels.
pub fn shortest_path(cost: &[f32], allowed: &[bool], w: usize, h: usize, start: (usize, usize), goal: (usize, usize)) -> Option<Vec<(usize, usize)>> {
    let n = w * h;
    let si = start.1 * w + start.0;
    let gi = goal.1 * w + goal.0;
    if si >= n || gi >= n {
        return None;
    }
    let mut dist = vec![u32::MAX; n];
    let mut prev = vec![u32::MAX; n];
    let mut heap = BinaryHeap::new();
    dist[si] = 0;
    heap.push(Reverse((0u32, si as u32)));
    // Integer weights: 1000 × cost (+ a small length term).
    let wq: Vec<u32> = cost.iter().map(|&c| (c * c * 1000.0) as u32 + 8).collect();
    while let Some(Reverse((d, i))) = heap.pop() {
        let i = i as usize;
        if d > dist[i] {
            continue;
        }
        if i == gi {
            break;
        }
        let (x, y) = (i % w, i / w);
        for (dx, dy, diag) in [(-1i32, 0i32, false), (1, 0, false), (0, -1, false), (0, 1, false), (-1, -1, true), (1, -1, true), (-1, 1, true), (1, 1, true)] {
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            if !allowed[j] && j != gi {
                continue;
            }
            let step = if diag { wq[j] * 1414 / 1000 } else { wq[j] };
            let nd = d + step;
            if nd < dist[j] {
                dist[j] = nd;
                prev[j] = i as u32;
                heap.push(Reverse((nd, j as u32)));
            }
        }
    }
    if dist[gi] == u32::MAX {
        return None;
    }
    let mut path = vec![(gi % w, gi / w)];
    let mut c = gi;
    while c != si {
        c = prev[c] as usize;
        path.push((c % w, c / w));
    }
    path.reverse();
    Some(path)
}

/// Rasterizes the polyline into `feature` (Bresenham through each segment).
fn draw_polyline(pts: &[P], x0: f32, y0: f32, w: usize, h: usize, feature: &mut [bool]) {
    let mut put = |x: f32, y: f32| {
        let (xi, yi) = ((x - x0).floor() as i32, (y - y0).floor() as i32);
        if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
            feature[yi as usize * w + xi as usize] = true;
        }
    };
    if pts.len() == 1 {
        put(pts[0].x, pts[0].y);
    }
    for s in pts.windows(2) {
        let n = (s[1].dist(s[0]).ceil() as usize).max(1);
        for k in 0..=n {
            let q = s[0].lerp(s[1], k as f32 / n as f32);
            put(q.x, q.y);
        }
    }
}

/// One live-wire solve on a single level grid. `trail` and the returned
/// path are in that level's pixel coordinates.
fn solve_level(grid: &TileGrid, trail: &[P], band: f32, contrast: f32) -> Option<Vec<P>> {
    let (mut lo, mut hi) = (p(f32::MAX, f32::MAX), p(f32::MIN, f32::MIN));
    for q in trail {
        lo = p(lo.x.min(q.x), lo.y.min(q.y));
        hi = p(hi.x.max(q.x), hi.y.max(q.y));
    }
    let pad = band.ceil() + 2.0;
    let x0 = ((lo.x - pad).floor() as i32).max(0);
    let y0 = ((lo.y - pad).floor() as i32).max(0);
    let x1 = ((hi.x + pad).ceil() as i32).min(grid.w as i32);
    let y1 = ((hi.y + pad).ceil() as i32).min(grid.h as i32);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let cost = cost_region(grid, x0, y0, w, h, contrast);
    let mut feature = vec![false; w * h];
    draw_polyline(trail, x0 as f32, y0 as f32, w, h, &mut feature);
    let d2 = edt(&feature, w, h);
    let b2 = band * band;
    let allowed: Vec<bool> = d2.iter().map(|&d| d <= b2).collect();
    let cl = |q: P| (((q.x - x0 as f32).floor().max(0.0) as usize).min(w - 1), ((q.y - y0 as f32).floor().max(0.0) as usize).min(h - 1));
    let path = shortest_path(&cost, &allowed, w, h, cl(trail[0]), cl(*trail.last().unwrap()))?;
    Some(path.into_iter().map(|(x, y)| p(x0 as f32 + x as f32 + 0.5, y0 as f32 + y as f32 + 0.5)).collect())
}

/// The magnetic path from `trail[0]` (last anchor) to the cursor (last
/// trail point), in level-0 coordinates. Pixel count of the search is
/// bounded by `trail length × width`, independent of the image size.
pub fn trace(layer: &Layer, trail: &[P], params: &WireParams) -> Vec<P> {
    if trail.len() < 2 {
        return trail.to_vec();
    }
    let len: f32 = trail.windows(2).map(|w| w[0].dist(w[1])).sum();
    let area = len * params.width * 2.0;
    // Coarse level so the band holds at most ~60k pixels.
    let mut level = 0u32;
    while area / (4u32.pow(level) as f32) > 60_000.0 && level < layer.mips.len() as u32 && params.width / (1u32 << (level + 1)) as f32 >= 2.0 {
        level += 1;
    }
    let s = (1u32 << level) as f32;
    let coarse_trail: Vec<P> = trail.iter().map(|q| p(q.x / s, q.y / s)).collect();
    let Some(coarse) = solve_level(layer.level(level), &coarse_trail, (params.width / s).max(2.0), params.contrast) else {
        return vec![trail[0], *trail.last().unwrap()];
    };
    if level == 0 {
        return coarse;
    }
    // Refine at level 0 around the upsampled path.
    let mut up: Vec<P> = coarse.iter().map(|q| p(q.x * s, q.y * s)).collect();
    up[0] = trail[0];
    *up.last_mut().unwrap() = *trail.last().unwrap();
    solve_level(&layer.grid, &up, s + 2.0, params.contrast).unwrap_or(up)
}

/// Follows the edge nearest `seed` in both directions (Content-Aware
/// Tracing). Returns the traced polyline in level-0 coordinates.
pub fn follow_edge(layer: &Layer, seed: P, max_len: usize, contrast: f32) -> Vec<P> {
    let g = &layer.grid;
    let s = super::filters::distort::Sampler::new(g, None, 1, false);
    let lum = |q: P| luma(s.sample(q.x, q.y));
    let grad = |q: P| -> (f32, f32) { ((lum(p(q.x + 1.0, q.y)) - lum(p(q.x - 1.0, q.y))) * 0.5, (lum(p(q.x, q.y + 1.0)) - lum(p(q.x, q.y - 1.0))) * 0.5) };
    let mag = |q: P| {
        let (gx, gy) = grad(q);
        gx.hypot(gy)
    };
    // Snap the seed to the strongest edge within 6 px.
    let mut best = (seed, mag(seed));
    for dy in -6..=6 {
        for dx in -6..=6 {
            let q = p(seed.x + dx as f32, seed.y + dy as f32);
            let m = mag(q);
            if m > best.1 {
                best = (q, m);
            }
        }
    }
    let start = best.0;
    let floor = (best.1 * contrast.clamp(0.05, 0.9)).max(4.0);
    let walk = |dir_sign: f32| -> Vec<P> {
        let mut pts = vec![];
        let mut q = start;
        let mut prev_t: Option<(f32, f32)> = None;
        for _ in 0..max_len {
            let (gx, gy) = grad(q);
            let m = gx.hypot(gy);
            if m < floor {
                break;
            }
            // Tangent = gradient rotated 90°, kept consistent with the walk.
            let mut t = (-gy / m, gx / m);
            if let Some(pt) = prev_t {
                if t.0 * pt.0 + t.1 * pt.1 < 0.0 {
                    t = (-t.0, -t.1);
                }
            } else {
                t = (t.0 * dir_sign, t.1 * dir_sign);
            }
            prev_t = Some(t);
            let mut nq = p(q.x + t.0, q.y + t.1);
            // Re-centre on the edge along the normal.
            let nrm = (gx / m, gy / m);
            let mut bestn = (nq, mag(nq));
            for k in [-2.0f32, -1.0, 1.0, 2.0] {
                let c = p(nq.x + nrm.0 * k * 0.5, nq.y + nrm.1 * k * 0.5);
                let mm = mag(c);
                if mm > bestn.1 {
                    bestn = (c, mm);
                }
            }
            nq = bestn.0;
            if nq.x < 1.0 || nq.y < 1.0 || nq.x > g.w as f32 - 2.0 || nq.y > g.h as f32 - 2.0 {
                break;
            }
            if pts.len() > 20 && nq.dist(start) < 1.5 {
                break;
            }
            pts.push(nq);
            q = nq;
        }
        pts
    };
    let mut back = walk(-1.0);
    back.reverse();
    back.push(start);
    back.extend(walk(1.0));
    back
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::grid::{TS, tile_mut};

    fn disk_image(n: u32, r: f32) -> TileGrid {
        let mut g = TileGrid::filled(n, n, px::pack(30, 30, 30, 255));
        let c = n as f32 * 0.5;
        for y in 0..n as i32 {
            for x in 0..n as i32 {
                let d = ((x as f32 + 0.5 - c).powi(2) + (y as f32 + 0.5 - c).powi(2)).sqrt();
                if d < r {
                    let i = g.idx(x >> 8, y >> 8);
                    tile_mut(&mut g.tiles[i]).write()[((y & 255) as usize) * TS + (x & 255) as usize] = px::pack(220, 220, 220, 255);
                }
            }
        }
        g
    }

    #[test]
    fn wire_snaps_to_circle_edge() {
        let g = disk_image(1024, 300.0);
        let layer = Layer::new(1, "d", g);
        // A sloppy trail ~8 px outside the true edge.
        let c = 512.0;
        let trail: Vec<P> = (0..=20).map(|k| {
            let t = k as f32 / 20.0 * 1.2;
            p(c + 308.0 * t.cos(), c + 308.0 * t.sin())
        }).collect();
        let path = trace(&layer, &trail, &WireParams { width: 16.0, contrast: 0.1 });
        assert!(path.len() > 50);
        // The ends approach the off-edge anchors; everything between hugs the edge.
        let inner = &path[12..path.len() - 12];
        let on_edge = inner.iter().filter(|q| (q.dist(p(c, c)) - 300.0).abs() <= 1.5).count();
        assert!(on_edge as f32 >= inner.len() as f32 * 0.97, "{on_edge}/{} points on the edge", inner.len());
    }

    #[test]
    fn coarse_then_refine_on_long_trails() {
        let g = disk_image(4096, 1500.0);
        let layer = Layer::new(1, "d", g);
        let c = 2048.0;
        let trail: Vec<P> = (0..=60).map(|k| {
            let t = k as f32 / 60.0 * 1.5;
            p(c + 1512.0 * t.cos(), c + 1512.0 * t.sin())
        }).collect();
        // The time guards against a blow-up, not for speed (--bench-tools
        // has that): the best of three runs drops stalls from the tests
        // running in parallel.
        let mut ms = f32::MAX;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let path = trace(&layer, &trail, &WireParams { width: 40.0, contrast: 0.1 });
            ms = ms.min(t0.elapsed().as_secs_f32() * 1000.0);
            let inner = &path[20..path.len() - 20];
            let on_edge = inner.iter().filter(|q| (q.dist(p(c, c)) - 1500.0).abs() <= 1.5).count();
            assert!(on_edge as f32 >= inner.len() as f32 * 0.97, "{on_edge}/{} points on the edge", inner.len());
            if ms < 400.0 {
                break;
            }
        }
        assert!(ms < 400.0, "{ms} ms");
    }

    #[test]
    fn follow_edge_traces_circle() {
        let layer = Layer::new(1, "d", disk_image(1024, 300.0));
        let pts = follow_edge(&layer, p(512.0 + 303.0, 512.0), 3000, 0.3);
        assert!(pts.len() > 1000, "only {} points", pts.len());
        let worst = pts.iter().map(|q| (q.dist(p(512.0, 512.0)) - 300.0).abs()).fold(0f32, f32::max);
        assert!(worst < 2.5, "{worst}");
    }
}
