//! Liang's hyphenation (the TeX algorithm) with the US English patterns of
//! the hyph-utf8 package (`assets/hyph`, Gerard D.C. Kuiken's ushyphmax,
//! all-permissive licence; see `assets/hyph/LICENSE-hyph-en-us.txt`).
//!
//! Each pattern is a letter string with inter-letter digits; for a word,
//! every pattern matching a substring of `.word.` raises the digits at its
//! positions, and an odd value allows a break there. The exception list
//! gives full hyphenations of words the patterns get wrong. Breaks keep at
//! least 2 letters before and 3 after (the patterns' hyphenmins).

use std::collections::HashMap;
use std::sync::OnceLock;

const PATTERNS: &str = include_str!("../../assets/hyph/hyph-en-us.pat.txt");
const EXCEPTIONS: &str = include_str!("../../assets/hyph/hyph-en-us.hyp.txt");
const LEFT_MIN: usize = 2;
const RIGHT_MIN: usize = 3;

struct Tables {
    /// Letters → digit values between them (len = letters + 1).
    patterns: HashMap<Vec<u8>, Vec<u8>>,
    max_len: usize,
    /// Word → allowed break positions (before letter index).
    exceptions: HashMap<String, Vec<usize>>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut patterns = HashMap::new();
        let mut max_len = 0;
        for pat in PATTERNS.split_whitespace() {
            let mut letters = Vec::new();
            let mut digits = vec![0u8];
            for b in pat.bytes() {
                if b.is_ascii_digit() {
                    *digits.last_mut().unwrap() = b - b'0';
                } else {
                    letters.push(b);
                    digits.push(0);
                }
            }
            max_len = max_len.max(letters.len());
            patterns.insert(letters, digits);
        }
        let mut exceptions = HashMap::new();
        for w in EXCEPTIONS.split_whitespace() {
            let mut at = Vec::new();
            let mut n = 0;
            for c in w.chars() {
                if c == '-' {
                    at.push(n);
                } else {
                    n += 1;
                }
            }
            exceptions.insert(w.replace('-', ""), at);
        }
        Tables { patterns, max_len, exceptions }
    })
}

/// Allowed break positions of `word` (before letter index `k`), or `None`
/// when the patterns don't apply (non-ASCII-letter words).
pub fn breaks(word: &str) -> Option<Vec<usize>> {
    if word.is_empty() || !word.bytes().all(|b| b.is_ascii_alphabetic()) {
        return None;
    }
    let lower = word.to_ascii_lowercase();
    let n = lower.len();
    if n < LEFT_MIN + RIGHT_MIN {
        return Some(vec![]);
    }
    let t = tables();
    if let Some(at) = t.exceptions.get(&lower) {
        return Some(at.clone());
    }
    let w: Vec<u8> = std::iter::once(b'.').chain(lower.bytes()).chain(std::iter::once(b'.')).collect();
    // points[i]: value before w[i].
    let mut points = vec![0u8; w.len() + 1];
    for i in 0..w.len() {
        for j in i + 1..=(i + t.max_len).min(w.len()) {
            if let Some(d) = t.patterns.get(&w[i..j]) {
                for (k, &v) in d.iter().enumerate() {
                    points[i + k] = points[i + k].max(v);
                }
            }
        }
    }
    // Letter k of the word is w[k + 1]; a break before it is points[k + 1].
    Some((LEFT_MIN..=n - RIGHT_MIN).filter(|&k| points[k + 1] % 2 == 1).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hyph(w: &str) -> String {
        let at = breaks(w).unwrap();
        let mut s = String::new();
        for (i, c) in w.chars().enumerate() {
            if at.contains(&i) {
                s.push('-');
            }
            s.push(c);
        }
        s
    }

    #[test]
    fn liang_matches_tex() {
        // TeX's \showhyphens with ushyphmax.
        assert_eq!(hyph("hyphenation"), "hy-phen-ation");
        assert_eq!(hyph("counterrevolutionaries"), "coun-ter-rev-o-lu-tion-ar-ies");
        assert_eq!(hyph("paragraph"), "para-graph");
        assert_eq!(hyph("internationalization"), "in-ter-na-tion-al-iza-tion");
        assert_eq!(hyph("associate"), "as-so-ciate");
        assert_eq!(hyph("table"), "ta-ble");
        assert_eq!(hyph("project"), "project");
        assert_eq!(hyph("Typography"), "Ty-pog-ra-phy");
        assert_eq!(breaks("naïve"), None);
        assert_eq!(hyph("cat"), "cat");
    }
}
