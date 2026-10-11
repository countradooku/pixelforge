//! Shared benchmark sizes and argument validation. A tier is an actual
//! rectangular document, not a square with roughly the same pixel count.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dimensions {
    pub w: u32,
    pub h: u32,
}

impl Dimensions {
    pub const fn square(side: u32) -> Self {
        Self { w: side, h: side }
    }

    pub fn pixels(self) -> u64 {
        u64::from(self.w) * u64::from(self.h)
    }
}

pub const TIERS: [(&str, Dimensions); 4] = [
    ("24", Dimensions { w: 6000, h: 4000 }),
    ("61", Dimensions { w: 9504, h: 6336 }),
    ("150", Dimensions { w: 14204, h: 10652 }),
    ("268", Dimensions::square(16384)),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Suite {
    Engine,
    Memory,
    Filters,
    Tools,
    Gpu,
}

impl Suite {
    pub fn parse(flag: &str) -> Option<Self> {
        match flag {
            "--bench" => Some(Self::Engine),
            "--bench-memory" => Some(Self::Memory),
            "--bench-filters" => Some(Self::Filters),
            "--bench-tools" => Some(Self::Tools),
            "--bench-gpu" => Some(Self::Gpu),
            _ => None,
        }
    }
}

pub const HELP: &str = "Benchmark options:
  --bench | --bench-filters | --bench-tools | --bench-gpu | --bench-memory
  [side | WIDTHxHEIGHT] [filter/section]
  --tier 24|61|150|268|all    Use the specified photographic document size
  --depth 8                  16/32-bit execution is not implemented yet
  --list-tiers               Show exact dimensions without allocating images
  --write-priors             Update calibration priors (GPU suite only)
Examples:
  pixelforge --bench --tier 24
  pixelforge --bench-filters --tier 61 invert
  pixelforge --bench-tools --tier 24 brush
  pixelforge --bench-gpu --tier 24 levels";

#[derive(Debug)]
pub struct Options {
    pub sizes: Vec<Dimensions>,
    pub only: Option<String>,
    pub write_priors: bool,
    pub list: bool,
}

impl Options {
    pub fn parse(suite: Suite, args: &[String]) -> Result<Self, String> {
        let mut sizes = None;
        let mut only = None;
        let mut write_priors = false;
        let mut list = false;
        let mut i = 0;
        while i < args.len() {
            let arg = args[i].as_str();
            match arg {
                "--list-tiers" | "--help" | "-h" => list = true,
                "--tier" => {
                    i += 1;
                    let value = args
                        .get(i)
                        .ok_or("--tier requires 24, 61, 150, 268, or all")?;
                    if sizes.is_some() {
                        return Err("choose one size or one --tier selector".into());
                    }
                    sizes = Some(if value == "all" {
                        TIERS.iter().map(|(_, d)| *d).collect()
                    } else {
                        vec![
                            TIERS
                                .iter()
                                .find(|(name, _)| name == value)
                                .ok_or_else(|| {
                                    format!(
                                        "unknown tier {value:?}; expected 24, 61, 150, 268, or all"
                                    )
                                })?
                                .1,
                        ]
                    });
                }
                "--depth" => {
                    i += 1;
                    let depth = args.get(i).ok_or("--depth requires a bit depth")?;
                    if depth != "8" {
                        return Err(format!(
                            "{depth}-bit benchmarks are not implemented; the current engine is 8-bit (no conversion was performed)"
                        ));
                    }
                }
                "--write-priors" if suite == Suite::Gpu => write_priors = true,
                flag if flag.starts_with('-') => return Err(format!("unknown option {flag:?}")),
                value => {
                    let looks_like_size = value.chars().next().is_some_and(|c| c.is_ascii_digit());
                    if looks_like_size {
                        if sizes.is_some() {
                            return Err("choose one size or one --tier selector".into());
                        }
                        let (w, h) = value.split_once('x').unwrap_or((value, value));
                        let parse =
                            |v: &str| v.parse::<u32>().ok().filter(|n| (1..=300_000).contains(n));
                        let d = Dimensions {
                            w: parse(w).ok_or_else(|| format!("invalid benchmark width {w:?}"))?,
                            h: parse(h).ok_or_else(|| format!("invalid benchmark height {h:?}"))?,
                        };
                        sizes = Some(vec![d]);
                    } else if suite == Suite::Engine || only.is_some() {
                        return Err(format!("unexpected argument {value:?}"));
                    } else {
                        only = Some(value.to_string());
                    }
                }
            }
            i += 1;
        }
        let sizes = sizes.unwrap_or_else(|| match suite {
            Suite::Filters => vec![Dimensions::square(8192)],
            Suite::Gpu => vec![Dimensions::square(4096), Dimensions::square(16384)],
            _ => vec![Dimensions::square(16384)],
        });
        Ok(Self {
            sizes,
            only,
            write_priors,
            list,
        })
    }
}

pub fn print_tiers() {
    println!("{HELP}\n\nTier    Width × height       Actual MP");
    for (name, d) in TIERS {
        println!(
            "{name:>4}    {:>6} × {:<6}      {:.3}",
            d.w,
            d.h,
            d.pixels() as f64 / 1e6
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_string()).collect()
    }

    #[test]
    fn all_suites_accept_exact_rectangular_tiers_and_filters() {
        for suite in [Suite::Engine, Suite::Filters, Suite::Tools, Suite::Gpu] {
            for (name, expected) in TIERS {
                let mut a = args(&["--tier", name, "--depth", "8"]);
                if suite != Suite::Engine {
                    a.push("brush".into());
                }
                let parsed = Options::parse(suite, &a).unwrap();
                assert_eq!(parsed.sizes, [expected]);
                assert_eq!(
                    parsed.only.as_deref(),
                    (suite != Suite::Engine).then_some("brush")
                );
            }
        }
    }

    #[test]
    fn legacy_defaults_and_explicit_square_sizes_are_preserved() {
        assert_eq!(
            Options::parse(Suite::Gpu, &[]).unwrap().sizes,
            [Dimensions::square(4096), Dimensions::square(16384)]
        );
        assert_eq!(
            Options::parse(Suite::Filters, &[]).unwrap().sizes,
            [Dimensions::square(8192)]
        );
        let p = Options::parse(Suite::Tools, &args(&["4096", "brush"])).unwrap();
        assert_eq!(p.sizes, [Dimensions::square(4096)]);
        assert_eq!(p.only.as_deref(), Some("brush"));
        assert_eq!(
            Options::parse(Suite::Engine, &args(&["601x399"]))
                .unwrap()
                .sizes,
            [Dimensions { w: 601, h: 399 }]
        );
    }

    #[test]
    fn invalid_or_unsupported_workloads_never_fall_back_to_a_huge_default() {
        for a in [
            &["0"][..],
            &["400000"],
            &["6000x0"],
            &["--tier"],
            &["--tier", "60"],
            &["--tier", "24", "4096"],
            &["--depth", "16"],
            &["--depth", "32"],
            &["--depht", "8"],
            &["--write-priors"],
        ] {
            assert!(Options::parse(Suite::Engine, &args(a)).is_err(), "{a:?}");
        }
    }

    #[test]
    fn all_tiers_run_in_increasing_order_and_report_actual_pixel_counts() {
        let p = Options::parse(Suite::Gpu, &args(&["--tier", "all", "--write-priors"])).unwrap();
        assert!(p.write_priors);
        assert_eq!(
            p.sizes.iter().map(|d| d.pixels()).collect::<Vec<_>>(),
            [24_000_000, 60_217_344, 151_301_008, 268_435_456]
        );
    }
}

/// Interaction targets in milliseconds. Commit uses the engine's full-image
/// Hue/Saturation reference; preview uses 135 RGBA8 display tiles with blur.
#[derive(Clone, Copy)]
pub struct LatencyBudgets {
    pub brush_event: f64,
    pub preview: f64,
    pub commit: f64,
    pub composite: f64,
    pub open: f64,
    pub save: f64,
}
impl LatencyBudgets {
    pub fn for_size(w: u32, h: u32) -> Self {
        let mp = w as u64 * h as u64;
        let (preview, commit, composite, io) = if mp <= 24_000_000 {
            (33., 100., 33., 1500.)
        } else if mp <= 61_000_000 {
            (33., 180., 50., 2500.)
        } else if mp <= 152_000_000 {
            (50., 400., 100., 6000.)
        } else {
            (66., 700., 180., 10000.)
        };
        Self {
            brush_event: 2.,
            preview,
            commit,
            composite,
            open: io,
            save: io,
        }
    }
    pub fn target(self, label: &str) -> Option<f64> {
        match label {
            "brush stroke (400 events, 300 px soft)" => Some(self.brush_event * 401.),
            "hue/saturation (full image)" => Some(self.commit),
            "flatten 2 layers (multiply blend)" => Some(self.composite),
            "display: live blur r=60 preview at level 2" => Some(self.preview),
            "native document open" => Some(self.open),
            "native document save" => Some(self.save),
            _ => None,
        }
    }
}
