//! Boykov–Kolmogorov max-flow / min-cut for grid graphs.
//!
//! The search-tree algorithm grows trees from both terminals, augments along
//! the paths where they meet and re-adopts orphans, reusing the trees between
//! augmentations. On image grids it is far faster than push-relabel or
//! augmenting-path methods (Boykov & Kolmogorov, PAMI 2004); a 1 MP
//! 8-connected GrabCut graph cuts in tens of milliseconds.
//!
//! Stays on the CPU (no GPU path): augmenting paths are inherently serial
//! (each augmentation changes the residual graph the next search walks), and
//! GPU push-relabel needs thousands of global sweeps to converge on the
//! sparse, uneven flows GrabCut produces; the CPU search trees finish first.

const NONE: u32 = u32::MAX;
const TERMINAL: u32 = u32::MAX - 1;
const ORPHAN: u32 = u32::MAX - 2;

#[derive(Clone, Copy)]
struct Arc {
    head: u32,
    next: u32,
    r_cap: f32,
}

#[derive(Clone, Copy)]
struct Node {
    first: u32,
    parent: u32,
    next_active: u32,
    ts: u32,
    dist: u32,
    /// Residual capacity to the terminals: > 0 from source, < 0 to sink.
    tr_cap: f32,
    sink: bool,
}

pub struct Graph {
    nodes: Vec<Node>,
    arcs: Vec<Arc>,
    flow: f64,
    // Active queue (FIFO through next_active links).
    q_first: u32,
    q_last: u32,
    orphans: std::collections::VecDeque<u32>,
    time: u32,
}

impl Graph {
    pub fn new(n: usize, edge_hint: usize) -> Self {
        Graph {
            nodes: vec![Node { first: NONE, parent: NONE, next_active: NONE, ts: 0, dist: 0, tr_cap: 0.0, sink: false }; n],
            arcs: Vec::with_capacity(edge_hint * 2),
            flow: 0.0,
            q_first: NONE,
            q_last: NONE,
            orphans: Default::default(),
            time: 0,
        }
    }

    /// Source and sink capacities of node `i` (cumulative).
    pub fn add_tweights(&mut self, i: usize, cap_source: f32, cap_sink: f32) {
        let delta = self.nodes[i].tr_cap;
        let (mut cs, mut ct) = (cap_source, cap_sink);
        if delta > 0.0 {
            cs += delta;
        } else {
            ct -= delta;
        }
        self.flow += cs.min(ct) as f64;
        self.nodes[i].tr_cap = cs - ct;
    }

    /// Edge i→j with capacity `cap`, j→i with `rev`.
    pub fn add_edge(&mut self, i: usize, j: usize, cap: f32, rev: f32) {
        let a = self.arcs.len() as u32;
        self.arcs.push(Arc { head: j as u32, next: self.nodes[i].first, r_cap: cap });
        self.nodes[i].first = a;
        self.arcs.push(Arc { head: i as u32, next: self.nodes[j].first, r_cap: rev });
        self.nodes[j].first = a + 1;
    }

    #[inline]
    fn sister(a: u32) -> u32 {
        a ^ 1
    }

    #[inline]
    fn tail(&self, a: u32) -> u32 {
        self.arcs[Self::sister(a) as usize].head
    }

    fn set_active(&mut self, i: u32) {
        if self.nodes[i as usize].next_active == NONE {
            if self.q_last != NONE {
                self.nodes[self.q_last as usize].next_active = i;
            } else {
                self.q_first = i;
            }
            self.q_last = i;
            // Self-loop marks "in queue, last".
            self.nodes[i as usize].next_active = i;
        }
    }

    fn next_active(&mut self) -> u32 {
        loop {
            let i = self.q_first;
            if i == NONE {
                return NONE;
            }
            let nx = self.nodes[i as usize].next_active;
            if nx == i {
                self.q_first = NONE;
                self.q_last = NONE;
            } else {
                self.q_first = nx;
            }
            self.nodes[i as usize].next_active = NONE;
            if self.nodes[i as usize].parent != NONE {
                return i;
            }
        }
    }

    /// Runs the max-flow; returns the flow value.
    pub fn maxflow(&mut self) -> f64 {
        let n = self.nodes.len();
        for i in 0..n {
            let nd = &mut self.nodes[i];
            nd.next_active = NONE;
            nd.ts = 0;
            if nd.tr_cap > 0.0 {
                nd.sink = false;
                nd.parent = TERMINAL;
                nd.dist = 1;
            } else if nd.tr_cap < 0.0 {
                nd.sink = true;
                nd.parent = TERMINAL;
                nd.dist = 1;
            } else {
                nd.parent = NONE;
            }
        }
        self.q_first = NONE;
        self.q_last = NONE;
        for i in 0..n as u32 {
            if self.nodes[i as usize].parent == TERMINAL {
                self.set_active(i);
            }
        }
        let mut current = NONE;
        loop {
            // Growth.
            let mut i = current;
            if i != NONE {
                self.nodes[i as usize].next_active = NONE;
                if self.nodes[i as usize].parent == NONE {
                    i = NONE;
                }
            }
            if i == NONE {
                i = self.next_active();
                if i == NONE {
                    break;
                }
            }
            let mut meet = NONE;
            let isink = self.nodes[i as usize].sink;
            let mut a = self.nodes[i as usize].first;
            while a != NONE {
                let arc = self.arcs[a as usize];
                let cap = if !isink { arc.r_cap } else { self.arcs[Self::sister(a) as usize].r_cap };
                if cap > 0.0 {
                    let j = arc.head as usize;
                    if self.nodes[j].parent == NONE {
                        self.nodes[j].sink = isink;
                        self.nodes[j].parent = Self::sister(a);
                        self.nodes[j].ts = self.nodes[i as usize].ts;
                        self.nodes[j].dist = self.nodes[i as usize].dist + 1;
                        self.set_active(j as u32);
                    } else if self.nodes[j].sink != isink {
                        meet = if !isink { a } else { Self::sister(a) };
                        break;
                    } else if self.nodes[j].ts <= self.nodes[i as usize].ts && self.nodes[j].dist > self.nodes[i as usize].dist {
                        // Shorter path heuristic.
                        self.nodes[j].parent = Self::sister(a);
                        self.nodes[j].ts = self.nodes[i as usize].ts;
                        self.nodes[j].dist = self.nodes[i as usize].dist + 1;
                    }
                }
                a = arc.next;
            }
            self.time += 1;
            if meet != NONE {
                // Keep `i` as current so it is re-scanned after augmentation.
                self.nodes[i as usize].next_active = i;
                current = i;
                self.augment(meet);
                // Adoption.
                while let Some(o) = self.orphans.pop_front() {
                    if self.nodes[o as usize].sink {
                        self.process_orphan(o, true);
                    } else {
                        self.process_orphan(o, false);
                    }
                }
            } else {
                current = NONE;
            }
        }
        self.flow
    }

    fn augment(&mut self, middle: u32) {
        // Bottleneck: source tree side.
        let mut bottleneck = self.arcs[middle as usize].r_cap;
        let mut i = self.tail(middle);
        loop {
            let a = self.nodes[i as usize].parent;
            if a == TERMINAL {
                break;
            }
            bottleneck = bottleneck.min(self.arcs[Self::sister(a) as usize].r_cap);
            i = self.arcs[a as usize].head;
        }
        bottleneck = bottleneck.min(self.nodes[i as usize].tr_cap);
        // Sink tree side.
        let mut i = self.arcs[middle as usize].head;
        loop {
            let a = self.nodes[i as usize].parent;
            if a == TERMINAL {
                break;
            }
            bottleneck = bottleneck.min(self.arcs[a as usize].r_cap);
            i = self.arcs[a as usize].head;
        }
        bottleneck = bottleneck.min(-self.nodes[i as usize].tr_cap);

        // Push.
        self.arcs[Self::sister(middle) as usize].r_cap += bottleneck;
        self.arcs[middle as usize].r_cap -= bottleneck;
        let mut i = self.tail(middle);
        loop {
            let a = self.nodes[i as usize].parent;
            if a == TERMINAL {
                break;
            }
            self.arcs[a as usize].r_cap += bottleneck;
            let s = Self::sister(a) as usize;
            self.arcs[s].r_cap -= bottleneck;
            if self.arcs[s].r_cap <= 0.0 {
                self.nodes[i as usize].parent = ORPHAN;
                self.orphans.push_front(i);
            }
            i = self.arcs[a as usize].head;
        }
        self.nodes[i as usize].tr_cap -= bottleneck;
        if self.nodes[i as usize].tr_cap <= 0.0 {
            self.nodes[i as usize].parent = ORPHAN;
            self.orphans.push_front(i);
        }
        let mut i = self.arcs[middle as usize].head;
        loop {
            let a = self.nodes[i as usize].parent;
            if a == TERMINAL {
                break;
            }
            self.arcs[Self::sister(a) as usize].r_cap += bottleneck;
            self.arcs[a as usize].r_cap -= bottleneck;
            if self.arcs[a as usize].r_cap <= 0.0 {
                self.nodes[i as usize].parent = ORPHAN;
                self.orphans.push_front(i);
            }
            i = self.arcs[a as usize].head;
        }
        self.nodes[i as usize].tr_cap += bottleneck;
        if self.nodes[i as usize].tr_cap >= 0.0 {
            self.nodes[i as usize].parent = ORPHAN;
            self.orphans.push_front(i);
        }
        self.flow += bottleneck as f64;
    }

    fn process_orphan(&mut self, i: u32, sink: bool) {
        let mut best_a = NONE;
        let mut best_d = u32::MAX;
        let mut a0 = self.nodes[i as usize].first;
        while a0 != NONE {
            let cap = if sink { self.arcs[a0 as usize].r_cap } else { self.arcs[Self::sister(a0) as usize].r_cap };
            if cap > 0.0 {
                let j = self.arcs[a0 as usize].head;
                if self.nodes[j as usize].sink == sink && self.nodes[j as usize].parent != NONE {
                    // Check that j originates from a terminal.
                    let mut d = 0u32;
                    let mut k = j;
                    let ok = loop {
                        if self.nodes[k as usize].ts == self.time {
                            d += self.nodes[k as usize].dist;
                            break true;
                        }
                        let a = self.nodes[k as usize].parent;
                        d += 1;
                        if a == TERMINAL {
                            self.nodes[k as usize].ts = self.time;
                            self.nodes[k as usize].dist = 1;
                            break true;
                        }
                        if a == ORPHAN || a == NONE {
                            break false;
                        }
                        k = self.arcs[a as usize].head;
                    };
                    if ok {
                        if d < best_d {
                            best_a = a0;
                            best_d = d;
                        }
                        // Mark distances along the path.
                        let mut k = j;
                        let mut dd = d;
                        while self.nodes[k as usize].ts != self.time {
                            self.nodes[k as usize].ts = self.time;
                            self.nodes[k as usize].dist = dd;
                            dd = dd.saturating_sub(1);
                            k = self.arcs[self.nodes[k as usize].parent as usize].head;
                        }
                    }
                }
            }
            a0 = self.arcs[a0 as usize].next;
        }
        if best_a != NONE {
            self.nodes[i as usize].parent = best_a;
            self.nodes[i as usize].ts = self.time;
            self.nodes[i as usize].dist = best_d + 1;
            return;
        }
        // No parent found: i becomes free; its children become orphans.
        self.nodes[i as usize].parent = NONE;
        let mut a0 = self.nodes[i as usize].first;
        while a0 != NONE {
            let j = self.arcs[a0 as usize].head;
            let nj = self.nodes[j as usize];
            if nj.sink == sink && nj.parent != NONE {
                let cap = if sink { self.arcs[Self::sister(a0) as usize].r_cap } else { self.arcs[a0 as usize].r_cap };
                // Note: residual toward i from j's perspective.
                let _ = cap;
                let cap_back = if sink { self.arcs[a0 as usize].r_cap } else { self.arcs[Self::sister(a0) as usize].r_cap };
                if cap_back > 0.0 {
                    self.set_active(j);
                }
                if nj.parent != TERMINAL && nj.parent != ORPHAN && self.arcs[nj.parent as usize].head == i {
                    self.nodes[j as usize].parent = ORPHAN;
                    self.orphans.push_back(j);
                }
            }
            a0 = self.arcs[a0 as usize].next;
        }
    }

    /// After `maxflow`: true when node `i` is on the source side of the cut.
    pub fn is_source(&self, i: usize) -> bool {
        let n = &self.nodes[i];
        n.parent != NONE && !n.sink
    }
}

/// Builds and cuts a 2-D grid graph. `data(i)` gives (source, sink)
/// capacities, `smooth(i, j)` the symmetric pairwise weight. 8-connected
/// when `diag`. Returns per-pixel foreground (source) labels.
pub fn cut_grid(w: usize, h: usize, diag: bool, data: impl Fn(usize) -> (f32, f32), smooth: impl Fn(usize, usize) -> f32) -> Vec<bool> {
    let n = w * h;
    let mut g = Graph::new(n, n * if diag { 4 } else { 2 });
    for i in 0..n {
        let (s, t) = data(i);
        g.add_tweights(i, s, t);
    }
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if x + 1 < w {
                let c = smooth(i, i + 1);
                g.add_edge(i, i + 1, c, c);
            }
            if y + 1 < h {
                let c = smooth(i, i + w);
                g.add_edge(i, i + w, c, c);
            }
            if diag && y + 1 < h {
                if x + 1 < w {
                    let c = smooth(i, i + w + 1) * std::f32::consts::FRAC_1_SQRT_2;
                    g.add_edge(i, i + w + 1, c, c);
                }
                if x > 0 {
                    let c = smooth(i, i + w - 1) * std::f32::consts::FRAC_1_SQRT_2;
                    g.add_edge(i, i + w - 1, c, c);
                }
            }
        }
    }
    g.maxflow();
    (0..n).map(|i| g.is_source(i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Edmonds–Karp reference on a dense adjacency matrix.
    fn reference_maxflow(n: usize, cap: &[Vec<f32>], s: usize, t: usize) -> f64 {
        let mut r: Vec<Vec<f64>> = cap.iter().map(|row| row.iter().map(|&c| c as f64).collect()).collect();
        let mut flow = 0.0;
        loop {
            let mut prev = vec![usize::MAX; n];
            prev[s] = s;
            let mut q = std::collections::VecDeque::from([s]);
            while let Some(u) = q.pop_front() {
                for v in 0..n {
                    if prev[v] == usize::MAX && r[u][v] > 1e-9 {
                        prev[v] = u;
                        q.push_back(v);
                    }
                }
            }
            if prev[t] == usize::MAX {
                return flow;
            }
            let mut b = f64::MAX;
            let mut v = t;
            while v != s {
                b = b.min(r[prev[v]][v]);
                v = prev[v];
            }
            let mut v = t;
            while v != s {
                r[prev[v]][v] -= b;
                r[v][prev[v]] += b;
                v = prev[v];
            }
            flow += b;
        }
    }

    #[test]
    fn matches_reference_on_random_graphs() {
        for seed in 0..40u32 {
            let n = 12;
            let h = |k: u32| crate::engine::filters::util::hash01(k as i32, seed as i32, 99);
            let mut g = Graph::new(n, 64);
            // Reference graph: node n = source, n+1 = sink.
            let mut cap = vec![vec![0f32; n + 2]; n + 2];
            for i in 0..n {
                let (cs, ct) = ((h(i as u32 * 3) * 10.0).floor(), (h(i as u32 * 3 + 1) * 10.0).floor());
                g.add_tweights(i, cs, ct);
                cap[n][i] += cs;
                cap[i][n + 1] += ct;
            }
            for k in 0..30u32 {
                let i = (h(1000 + k * 7) * n as f32) as usize % n;
                let j = (h(2000 + k * 11) * n as f32) as usize % n;
                if i == j {
                    continue;
                }
                let (c1, c2) = ((h(3000 + k) * 8.0).floor(), (h(4000 + k) * 8.0).floor());
                g.add_edge(i, j, c1, c2);
                cap[i][j] += c1;
                cap[j][i] += c2;
            }
            let got = g.maxflow();
            let want = reference_maxflow(n + 2, &cap, n, n + 1);
            assert!((got - want).abs() < 1e-3, "seed {seed}: {got} vs {want}");
            // The cut must separate: source-side nodes have no residual path to sink.
            let cut: f64 = (0..n)
                .flat_map(|i| (0..n + 2).map(move |j| (i, j)))
                .filter(|&(i, j)| {
                    let si = g.is_source(i);
                    let sj = if j == n { true } else if j == n + 1 { false } else { g.is_source(j) };
                    si && !sj
                })
                .map(|(i, j)| cap[i][j] as f64)
                .sum::<f64>()
                + (0..n).filter(|&i| !g.is_source(i)).map(|i| cap[n][i] as f64).sum::<f64>();
            assert!((cut - want).abs() < 1e-3, "seed {seed}: cut {cut} vs flow {want}");
        }
    }

    #[test]
    fn grid_cut_separates_two_regions() {
        let (w, h) = (60, 40);
        let img: Vec<f32> = (0..w * h).map(|i| if i % w < 30 { 0.1 } else { 0.9 }).collect();
        let labels = cut_grid(
            w,
            h,
            true,
            |i| {
                let v = img[i];
                // Likelihoods: dark = foreground.
                ((-((v - 0.1f32).powi(2)) * 20.0).exp() * 5.0, (-((v - 0.9f32).powi(2)) * 20.0).exp() * 5.0)
            },
            |a, b| 2.0 * (-(img[a] - img[b]).powi(2) * 30.0).exp(),
        );
        assert!(labels[10 * w + 5]);
        assert!(!labels[10 * w + 50]);
    }
}
