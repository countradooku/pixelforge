//! User preferences and the per-user config directory (brush presets live
//! there too). `PF_CONFIG_DIR` overrides the location (tests, demos).

use std::path::PathBuf;

/// `~/Library/Application Support/PixelForge` on macOS, `%APPDATA%\PixelForge`
/// on Windows, `$XDG_CONFIG_HOME/pixelforge` (or `~/.config/pixelforge`) elsewhere.
pub fn config_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("PF_CONFIG_DIR") {
        return PathBuf::from(d);
    }
    // Tests never touch the user's settings (or each other's GPU health
    // and cost-model files).
    if cfg!(test) {
        return std::env::temp_dir().join(format!("pixelforge-test-{}", std::process::id()));
    }
    let home = || std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    if cfg!(target_os = "macos") {
        home().join("Library/Application Support/PixelForge")
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_else(home).join("PixelForge")
    } else {
        std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|| home().join(".config")).join("pixelforge")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Prefs {
    pub memory_percent: u8,
    pub cold_seconds: u64,
    pub scratch_dir: PathBuf,
    /// Author of new notes.
    pub author: String,
    /// Preferences ▸ Performance ▸ Use Graphics Processor.
    pub gpu: crate::engine::gpu::Mode,
    /// Per-feature overrides of `gpu` (advanced).
    pub gpu_overrides: Vec<(crate::engine::gpu::Feature, crate::engine::gpu::Mode)>,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs {
            memory_percent: 70,
            cold_seconds: 30,
            scratch_dir: std::env::temp_dir().join("pixelforge-scratch"),
            author: std::env::var("USER").or_else(|_| std::env::var("USERNAME")).unwrap_or_else(|_| "Author".into()),
            gpu: Default::default(),
            gpu_overrides: Vec::new(),
        }
    }
}

impl Prefs {
    fn path() -> PathBuf {
        config_dir().join("prefs.txt")
    }

    /// `key=value` lines; unknown keys are ignored, missing ones default.
    pub fn parse(text: &str) -> Prefs {
        use crate::engine::gpu::{Feature, Mode};
        let mut p = Prefs::default();
        for line in text.lines() {
            match line.split_once('=') {
                Some(("memory_percent", v)) => p.memory_percent = v.trim().parse::<u8>().unwrap_or(70).clamp(10, 90),
                Some(("cold_seconds", v)) => p.cold_seconds = v.trim().parse::<u64>().unwrap_or(30).clamp(1, 3600),
                Some(("scratch_dir", v)) if !v.trim().is_empty() => p.scratch_dir = PathBuf::from(v.trim()),
                Some(("author", v)) => p.author = v.trim().to_string(),
                Some(("gpu", v)) => p.gpu = Mode::parse(v).unwrap_or_default(),
                Some((k, v)) => {
                    if let (Some(f), Some(m)) = (k.strip_prefix("gpu.").and_then(|k| Feature::ALL.into_iter().find(|f| f.key() == k)), Mode::parse(v)) {
                        p.gpu_overrides.retain(|o| o.0 != f);
                        p.gpu_overrides.push((f, m));
                    }
                }
                None => {}
            }
        }
        p
    }

    pub fn to_text(&self) -> String {
        let mut s = format!("author={}\ngpu={}\n", self.author.replace('\n', " "), self.gpu.label().to_lowercase());
        s.push_str(&format!("memory_percent={}\ncold_seconds={}\nscratch_dir={}\n", self.memory_percent, self.cold_seconds, self.scratch_dir.display()));
        for (f, m) in &self.gpu_overrides {
            s.push_str(&format!("gpu.{}={}\n", f.key(), m.label().to_lowercase()));
        }
        s
    }

    /// Hands the GPU settings to the engine.
    pub fn apply_gpu(&self) {
        crate::engine::gpu::set_prefs(self.gpu, &self.gpu_overrides);
    }

    pub fn load() -> Prefs {
        std::fs::read_to_string(Self::path()).map(|t| Self::parse(&t)).unwrap_or_default()
    }

    pub fn save(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(config_dir())?;
        std::fs::write(Self::path(), self.to_text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefs_round_trip() {
        use crate::engine::gpu::{Feature, Mode};
        let p = Prefs { memory_percent: 50, cold_seconds: 90, scratch_dir: PathBuf::from("/tmp/custom scratch"), author: "Ada Lovelace".into(), gpu: Mode::Force, gpu_overrides: vec![(Feature::FilterGallery, Mode::Off), (Feature::Blur, Mode::Auto)] };
        assert_eq!(Prefs::parse(&p.to_text()), p);
        assert_eq!(Prefs::parse("other=1\nauthor= Grace \n").author, "Grace");
    }
}
