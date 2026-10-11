//! Align and Distribute for multi-layer selections (Move tool): the offset
//! each layer moves by, from the layers' content rectangles.
//!
//! - **Align** to the selection's bounds (or a marquee / the canvas), or to a
//!   **key object** that stays put (Photoshop's "Align To: Key Object").
//! - **Distribute** keeps the two outermost layers and spaces the others so
//!   their left / centre / right / top / middle / bottom edges are evenly
//!   spaced, or ("spacing") so the gaps between them are equal.

use super::grid::RectI;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Align {
    Left,
    HCenter,
    Right,
    Top,
    VCenter,
    Bottom,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Distribute {
    Left,
    HCenter,
    Right,
    Top,
    VCenter,
    Bottom,
    /// Equal horizontal gaps.
    HSpace,
    /// Equal vertical gaps.
    VSpace,
}

fn union(rects: &[RectI]) -> RectI {
    rects.iter().fold(RectI::EMPTY, |a, r| if a.is_empty() { *r } else { a.union(r) })
}

/// Offsets that align `rects` to `target` (`None` = their union; `key` =
/// that rect stays and the others align to it).
pub fn align(rects: &[RectI], how: Align, target: Option<RectI>, key: Option<usize>) -> Vec<(i32, i32)> {
    let t = match (key, target) {
        (Some(k), _) => rects[k],
        (None, Some(t)) => t,
        (None, None) => union(rects),
    };
    rects
        .iter()
        .enumerate()
        .map(|(i, r)| {
            if key == Some(i) {
                return (0, 0);
            }
            match how {
                Align::Left => (t.x0 - r.x0, 0),
                Align::Right => (t.x1 - r.x1, 0),
                Align::HCenter => ((t.x0 + t.x1 - r.x0 - r.x1) / 2, 0),
                Align::Top => (0, t.y0 - r.y0),
                Align::Bottom => (0, t.y1 - r.y1),
                Align::VCenter => (0, (t.y0 + t.y1 - r.y0 - r.y1) / 2),
            }
        })
        .collect()
}

/// Offsets that distribute `rects` (needs three or more; fewer are left as is).
pub fn distribute(rects: &[RectI], how: Distribute) -> Vec<(i32, i32)> {
    let n = rects.len();
    let mut out = vec![(0, 0); n];
    if n < 3 {
        return out;
    }
    let horizontal = matches!(how, Distribute::Left | Distribute::HCenter | Distribute::Right | Distribute::HSpace);
    // Position along the axis (doubled for exact centres) and extent.
    let key = |r: &RectI| -> i64 {
        match how {
            Distribute::Left | Distribute::HSpace => 2 * r.x0 as i64,
            Distribute::Right => 2 * r.x1 as i64,
            Distribute::HCenter => (r.x0 + r.x1) as i64,
            Distribute::Top | Distribute::VSpace => 2 * r.y0 as i64,
            Distribute::Bottom => 2 * r.y1 as i64,
            Distribute::VCenter => (r.y0 + r.y1) as i64,
        }
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| key(&rects[i]));
    let set = |out: &mut Vec<(i32, i32)>, i: usize, d: i64| {
        out[i] = if horizontal { (d as i32, 0) } else { (0, d as i32) };
    };
    match how {
        Distribute::HSpace | Distribute::VSpace => {
            let size = |r: &RectI| if horizontal { r.w() as i64 } else { r.h() as i64 };
            let lo = |r: &RectI| if horizontal { r.x0 as i64 } else { r.y0 as i64 };
            let (first, last) = (&rects[order[0]], &rects[order[n - 1]]);
            let span = lo(last) + size(last) - lo(first);
            let total: i64 = order.iter().map(|&i| size(&rects[i])).sum();
            let gap = (span - total) as f64 / (n - 1) as f64;
            let mut pos = lo(first) as f64;
            for &i in &order {
                let want = pos.round() as i64;
                set(&mut out, i, want - lo(&rects[i]));
                pos += size(&rects[i]) as f64 + gap;
            }
        }
        _ => {
            let (a, b) = (key(&rects[order[0]]), key(&rects[order[n - 1]]));
            for (k, &i) in order.iter().enumerate() {
                let want = a as f64 + (b - a) as f64 * k as f64 / (n - 1) as f64;
                set(&mut out, i, ((want - key(&rects[i]) as f64) / 2.0).round() as i64);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moved(r: &[RectI], d: &[(i32, i32)]) -> Vec<RectI> {
        r.iter().zip(d).map(|(r, &(dx, dy))| RectI::new(r.x0 + dx, r.y0 + dy, r.x1 + dx, r.y1 + dy)).collect()
    }

    #[test]
    fn align_to_selection_and_key_object() {
        let r = [RectI::new(10, 10, 50, 30), RectI::new(100, 40, 120, 90), RectI::new(60, 5, 70, 15)];
        let m = moved(&r, &align(&r, Align::Left, None, None));
        assert!(m.iter().all(|q| q.x0 == 10));
        let m = moved(&r, &align(&r, Align::Bottom, None, None));
        assert!(m.iter().all(|q| q.y1 == 90));
        // Key object: layer 1 stays, the others centre on it.
        let d = align(&r, Align::HCenter, None, Some(1));
        assert_eq!(d[1], (0, 0));
        let m = moved(&r, &d);
        assert!(m.iter().all(|q| (q.x0 + q.x1) / 2 == 110));
        // To a marquee.
        let m = moved(&r, &align(&r, Align::Top, Some(RectI::new(0, 200, 10, 300)), None));
        assert!(m.iter().all(|q| q.y0 == 200));
    }

    #[test]
    fn distribute_edges_centres_and_spacing() {
        let r = [RectI::new(0, 0, 10, 10), RectI::new(17, 0, 47, 10), RectI::new(100, 0, 120, 10), RectI::new(55, 0, 60, 10)];
        // Left edges evenly spaced between the outermost (0 and 100).
        let m = moved(&r, &distribute(&r, Distribute::Left));
        let mut xs: Vec<i32> = m.iter().map(|q| q.x0).collect();
        xs.sort();
        assert_eq!(xs, vec![0, 33, 67, 100]);
        // Equal gaps: span 0..120, widths 10 + 30 + 20 + 5 = 65 → 3 gaps of 18.33.
        let m = moved(&r, &distribute(&r, Distribute::HSpace));
        let mut s = m.clone();
        s.sort_by_key(|q| q.x0);
        let gaps: Vec<i32> = s.windows(2).map(|w| w[1].x0 - w[0].x1).collect();
        assert!(gaps.iter().all(|&g| (g - 18).abs() <= 1), "{gaps:?}");
        assert_eq!((s[0].x0, s[3].x1), (0, 120));
        // Vertical moves stay vertical; two layers don't move.
        assert!(distribute(&r, Distribute::VCenter).iter().all(|d| d.0 == 0));
        assert_eq!(distribute(&r[..2], Distribute::Left), vec![(0, 0); 2]);
    }
}
