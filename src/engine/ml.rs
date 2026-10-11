//! Optional ML backend: ONNX Runtime (CoreML execution provider on macOS)
//! behind the `ml` cargo feature.
//!
//! Models are plain `.onnx` files found at runtime in a `models/` folder
//! (`$PF_MODELS`, then `./models`, then next to the executable and two
//! levels up from it, which covers `target/release`). Nothing is bundled:
//! without the feature or without a model every caller falls back to the
//! classical algorithm and the UI says so.
//!
//! Recognised files:
//!
//! | file | used by | contract |
//! |---|---|---|
//! | `select_subject.onnx` | Select Subject, Object Selection (box crop) | salient-object / matting net (BiRefNet, U²-Net, ISNet, MODNet…): input `1×3×S×S` RGB, ImageNet-normalized; output `1×1×S×S` logits or probabilities |
//! | `depth_anything_v2_small.onnx` | Neural Filters ▸ Depth Blur | Depth Anything V2: input `1×3×H×W` (multiples of 14), ImageNet-normalized; output `1×H×W` relative inverse depth |
//!
//! Inference runs at ≤ 1024 px (518 px for depth); callers upsample the
//! result with the guided filter on boundary tiles only.

use std::path::PathBuf;

pub const SUBJECT_MODEL: &str = "select_subject.onnx";
#[allow(dead_code)] // Neural Filters ▸ Depth Blur (Phase G) is its caller.
pub const DEPTH_MODEL: &str = "depth_anything_v2_small.onnx";

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Folders searched for models, in order.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(d) = std::env::var("PF_MODELS") {
        v.push(PathBuf::from(d));
    }
    v.push(PathBuf::from("models"));
    if let Ok(exe) = std::env::current_exe() {
        let mut d = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..3 {
            if let Some(p) = d {
                v.push(p.join("models"));
                d = p.parent().map(|q| q.to_path_buf());
            }
        }
    }
    v
}

/// The model file `name`, if present anywhere on the search path.
pub fn model_path(name: &str) -> Option<PathBuf> {
    model_dirs().into_iter().map(|d| d.join(name)).find(|p| p.is_file())
}

/// Whether the binary was built with the ONNX Runtime backend.
pub const fn compiled() -> bool {
    cfg!(feature = "ml")
}

/// Whether `name` can actually run (feature on and model present).
pub fn ready(name: &str) -> bool {
    compiled() && model_path(name).is_some()
}

/// One line for the UI: what backs a feature and why.
pub fn describe(name: &str, what: &str, fallback: &str) -> String {
    match (compiled(), model_path(name)) {
        (true, Some(_)) => format!("{what}: neural model ({name})"),
        (true, None) => format!("{what}: {fallback} (no models/{name})"),
        (false, _) => format!("{what}: {fallback} (built without the `ml` feature)"),
    }
}

/// Bilinear resize of straight RGB8 into planar, ImageNet-normalized CHW.
fn to_planar(rgb: &[[u8; 3]], w: usize, h: usize, ow: usize, oh: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * ow * oh];
    let (sx, sy) = (w as f32 / ow as f32, h as f32 / oh as f32);
    for y in 0..oh {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (fy as usize, fy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..ow {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (fx as usize, fx.fract());
            let x1 = (x0 + 1).min(w - 1);
            for c in 0..3 {
                let a = rgb[y0 * w + x0][c] as f32 * (1.0 - tx) + rgb[y0 * w + x1][c] as f32 * tx;
                let b = rgb[y1 * w + x0][c] as f32 * (1.0 - tx) + rgb[y1 * w + x1][c] as f32 * tx;
                let v = (a * (1.0 - ty) + b * ty) / 255.0;
                out[c * ow * oh + y * ow + x] = (v - MEAN[c]) / STD[c];
            }
        }
    }
    out
}

/// Salient-subject probability map from `select_subject.onnx`, at the
/// model's resolution. `None` when unavailable.
pub fn subject_probability(rgb: &[[u8; 3]], w: usize, h: usize) -> Option<(Vec<f32>, usize, usize)> {
    let side = imp::input_side(SUBJECT_MODEL).unwrap_or(1024).min(1024);
    let input = to_planar(rgb, w, h, side, side);
    let (shape, mut out) = imp::run(SUBJECT_MODEL, input, [1, 3, side as i64, side as i64])?;
    let (oh, ow) = (shape[shape.len() - 2] as usize, shape[shape.len() - 1] as usize);
    out.truncate(ow * oh);
    // Logits → probabilities when the net has no final sigmoid.
    if out.iter().any(|&v| !(-0.01..=1.01).contains(&v)) {
        for v in &mut out {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
    }
    Some((out, ow, oh))
}

/// Relative inverse depth (near = large) from Depth Anything V2, normalized
/// to 0..1, at ≤ 518 px on the long side. `None` when unavailable.
#[allow(dead_code)] // Neural Filters ▸ Depth Blur (Phase G) is its caller.
pub fn depth(rgb: &[[u8; 3]], w: usize, h: usize) -> Option<(Vec<f32>, usize, usize)> {
    let k = 518.0 / w.max(h) as f32;
    let m14 = |v: f32| ((v / 14.0).round().max(1.0) * 14.0) as usize;
    let (ow, oh) = (m14(w as f32 * k), m14(h as f32 * k));
    let input = to_planar(rgb, w, h, ow, oh);
    let (shape, mut out) = imp::run(DEPTH_MODEL, input, [1, 3, oh as i64, ow as i64])?;
    let (dh, dw) = (shape[shape.len() - 2] as usize, shape[shape.len() - 1] as usize);
    out.truncate(dw * dh);
    let (lo, hi) = out.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let span = (hi - lo).max(1e-6);
    for v in &mut out {
        *v = (*v - lo) / span;
    }
    Some((out, dw, dh))
}

#[cfg(feature = "ml")]
mod imp {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use ort::session::Session;
    use ort::value::Tensor;

    fn sessions() -> &'static Mutex<HashMap<String, Session>> {
        static S: OnceLock<Mutex<HashMap<String, Session>>> = OnceLock::new();
        S.get_or_init(Default::default)
    }

    /// GPU execution providers to try, best first, for this platform
    /// (`PF_ML_EP=cpu|coreml|tensorrt|cuda|directml` overrides). ONNX
    /// Runtime skips any provider whose libraries or device are missing
    /// and runs the rest of the graph on the CPU.
    fn providers() -> Vec<ort::ep::ExecutionProviderDispatch> {
        use ort::ep;
        let coreml = || {
            let cache = std::env::temp_dir().join("pixelforge-coreml");
            ep::CoreML::default().with_model_cache_dir(cache.display()).build()
        };
        match std::env::var("PF_ML_EP").as_deref() {
            Ok("cpu") => return vec![],
            Ok("coreml") => return vec![coreml()],
            Ok("tensorrt") => return vec![ep::TensorRT::default().build()],
            Ok("cuda") => return vec![ep::CUDA::default().build()],
            Ok("directml") => return vec![ep::DirectML::default().build()],
            _ => {}
        }
        let mut v = Vec::new();
        if cfg!(target_os = "macos") {
            v.push(coreml());
        }
        if cfg!(any(target_os = "linux", target_os = "windows")) {
            v.push(ep::TensorRT::default().build());
            v.push(ep::CUDA::default().build());
        }
        if cfg!(target_os = "windows") {
            v.push(ep::DirectML::default().build());
        }
        v
    }

    fn open(name: &str) -> Option<Session> {
        let path = super::model_path(name)?;
        let mut b = Session::builder().ok()?;
        let eps = providers();
        if !eps.is_empty() {
            b = b.with_execution_providers(eps).ok()?;
        }
        match b.commit_from_file(&path) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("ml: cannot load {}: {e}", path.display());
                None
            }
        }
    }

    fn with_session<T>(name: &str, f: impl FnOnce(&mut Session) -> Option<T>) -> Option<T> {
        let mut map = sessions().lock().ok()?;
        if !map.contains_key(name) {
            let s = open(name)?;
            map.insert(name.to_string(), s);
        }
        f(map.get_mut(name)?)
    }

    /// The square input side of a model with a static input shape.
    pub fn input_side(name: &str) -> Option<usize> {
        with_session(name, |s| {
            let shape = s.inputs().first()?.dtype().tensor_shape()?.to_vec();
            let side = *shape.last()?;
            (side > 0).then_some(side as usize)
        })
    }

    /// Runs a single-input, single-output image model.
    pub fn run(name: &str, input: Vec<f32>, shape: [i64; 4]) -> Option<(Vec<i64>, Vec<f32>)> {
        with_session(name, |s| {
            let input_name = s.inputs().first()?.name().to_string();
            let t = Tensor::from_array((shape.to_vec(), input)).ok()?;
            let out = match s.run(ort::inputs![input_name => t]) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("ml: {name} failed: {e}");
                    return None;
                }
            };
            let (sh, data) = out[0].try_extract_tensor::<f32>().ok()?;
            Some((sh.to_vec(), data.to_vec()))
        })
    }
}

#[cfg(not(feature = "ml"))]
mod imp {
    pub fn input_side(_name: &str) -> Option<usize> {
        None
    }

    pub fn run(_name: &str, _input: Vec<f32>, _shape: [i64; 4]) -> Option<(Vec<i64>, Vec<f32>)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planar_normalization() {
        let rgb = vec![[255u8, 128, 0]; 4];
        let p = to_planar(&rgb, 2, 2, 1, 1);
        assert!((p[0] - (1.0 - MEAN[0]) / STD[0]).abs() < 1e-4);
        assert!((p[2] - (0.0 - MEAN[2]) / STD[2]).abs() < 1e-4);
    }

    /// Runs only when the binary has the backend and the model is present.
    #[test]
    fn depth_model_orders_near_and_far() {
        if !ready(DEPTH_MODEL) {
            eprintln!("skipped: {}", describe(DEPTH_MODEL, "depth", "none"));
            return;
        }
        // A photo of a dog on a lawn: the grass at the bottom is nearer than at the top.
        let Ok(img) = image::open("testdata/photos/dog_grass.jpg") else {
            eprintln!("skipped: testdata/photos/dog_grass.jpg missing");
            return;
        };
        let img = img.into_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb: Vec<[u8; 3]> = img.pixels().map(|q| q.0).collect();
        let (d, dw, dh) = depth(&rgb, w, h).expect("depth inference");
        assert_eq!(d.len(), dw * dh);
        let row = |y: usize| d[y * dw..(y + 1) * dw].iter().sum::<f32>() / dw as f32;
        assert!(row(dh - 5) > row(5), "bottom {} top {}", row(dh - 5), row(5));
    }
}
