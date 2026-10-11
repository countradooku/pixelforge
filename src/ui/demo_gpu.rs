//! `PF_DEMO=tools PF_DEMO_PHASE=GPU`: screenshots of the GPU path — the
//! Performance settings, live filter previews and commits on the GPU, and a
//! many-layer document composited on the GPU — with the Info panel naming
//! the path each operation took. Prints `SHOT name` lines for
//! `tools/demo_shots.sh`.

use gpui_kit::{AsyncWindowContext, WeakEntity};

use crate::engine::blend::BlendMode;
use crate::engine::document::{Document, Layer};
use crate::engine::filters::catalog::Kind;
use crate::engine::layers::{LayerKind, LayerMask};
use crate::engine::selection::{MaskGrid, Selection};

use super::editor::Editor;

fn photo(name: &str) -> crate::engine::grid::TileGrid {
    crate::engine::io::open_image(std::path::Path::new(&format!("testdata/photos/{name}.jpg"))).unwrap_or_else(|_| crate::engine::ops::demo_scene(3000, 2000))
}

/// Twelve layers over the dog photo: every few blend modes, opacities,
/// masks, a clipped layer and an isolated group with an adjustment.
fn layered_doc() -> Document {
    let base = photo("dog_grass");
    let (w, h) = (base.w, base.h);
    let mut layers = vec![Layer::new(1, "Background", base)];
    let srcs = ["butterfly", "portrait_hair"];
    let modes = [BlendMode::Multiply, BlendMode::Screen, BlendMode::Overlay, BlendMode::SoftLight, BlendMode::Color, BlendMode::Difference, BlendMode::Luminosity, BlendMode::Normal];
    for (i, m) in modes.iter().enumerate() {
        let mut l = Layer::new(2 + i as u64, format!("{} {}", m.name(), i + 1), crate::engine::ops::resize(&Layer::new(0, "", photo(srcs[i % 2])), w, h));
        l.blend = *m;
        l.opacity = 0.25 + 0.08 * i as f32;
        if i % 3 == 0 {
            let r = crate::engine::grid::RectI::new((i as i32 * 97) % (w as i32 / 2), 40, w as i32 / 2 + (i as i32 * 61) % (w as i32 / 2), h as i32 - 40);
            l.mask = Some(LayerMask::new(MaskGrid::from_selection(&Selection::ellipse(r), w, h)));
        }
        if i == 4 {
            l.clip = true;
        }
        layers.push(l);
    }
    let mut grp = Layer::new(100, "Group (isolated)", crate::engine::grid::TileGrid::new(w, h));
    grp.kind = LayerKind::Group { pass_through: false, collapsed: false };
    grp.opacity = 0.8;
    let mut kid = Layer::new(101, "Hard Light in group", crate::engine::ops::resize(&Layer::new(0, "", photo("butterfly")), w, h));
    kid.parent = Some(100);
    kid.blend = BlendMode::HardLight;
    let mut adj = Layer::new(102, "Hue/Saturation", crate::engine::grid::TileGrid::new(w, h));
    adj.parent = Some(100);
    adj.kind = LayerKind::Adjustment(crate::engine::filters::hue_saturation(60.0, 20.0, 0.0));
    layers.push(kid);
    layers.push(adj);
    layers.push(grp);
    let n = layers.len();
    let mut d = Document::from_layers(w, h, layers, n - 1);
    d.path = Some("gpu_layers.pforge".into());
    d
}

pub async fn run(this: &WeakEntity<Editor>, cx: &mut AsyncWindowContext, phases: &[String]) {
    if !phases.iter().any(|p| p == "GPU") {
        return;
    }
    let exec = cx.background_executor().clone();
    let wait = move |ms: u64| exec.timer(std::time::Duration::from_millis(ms));
    macro_rules! step {
        ($body:expr) => {
            if this.update_in(cx, $body).is_err() {
                return;
            }
            let _ = this.update_in(cx, |_: &mut Editor, win, cx| {
                cx.notify();
                win.refresh();
            });
        };
    }
    macro_rules! until {
        ($cond:expr) => {
            for _ in 0..300 {
                wait(100).await;
                if this.update(cx, $cond).unwrap_or(true) {
                    break;
                }
            }
        };
    }
    macro_rules! shot {
        ($name:expr) => {
            wait(900).await;
            // One more frame so the Info panel shows what the last frame did.
            let _ = this.update_in(cx, |_: &mut Editor, win, cx| {
                cx.notify();
                win.refresh();
            });
            wait(300).await;
            let _ = this.update(cx, |_: &mut Editor, _| {
                if let Some(op) = crate::engine::gpu::last_op() {
                    eprintln!("path: {} · {} ({:.1} ms)", op.op, op.path_label(), op.ms);
                }
            });
            eprintln!("SHOT {}", $name);
            wait(2600).await;
        };
    }
    // Wait for the background GPU start-up.
    for _ in 0..100 {
        if !matches!(crate::engine::gpu::status(), crate::engine::gpu::Status::NotStarted | crate::engine::gpu::Status::Initializing) {
            break;
        }
        wait(100).await;
    }
    eprintln!("gpu: {}", crate::engine::gpu::describe());

    // GPU1: Settings ▸ Performance.
    step!(|ed: &mut Editor, _, cx| {
        ed.replace_document(doc_photo("dog_grass"));
        ed.open_preferences(cx);
    });
    shot!("pGPU_settings_performance");
    step!(|ed: &mut Editor, win, cx| ed.close_dialog(false, win, cx));

    // GPU2: Twirl live preview (whole-source gather on the GPU).
    step!(|ed: &mut Editor, win, cx| ed.open_filter_dialog(Kind::Twirl, win, cx));
    shot!("pGPU_twirl_live_preview");
    step!(|ed: &mut Editor, win, cx| ed.close_dialog(false, win, cx));

    // GPU3: Gaussian Blur preview and commit.
    step!(|ed: &mut Editor, win, cx| ed.open_filter_dialog(Kind::GaussianBlur, win, cx));
    shot!("pGPU_gaussian_live_preview");
    step!(|ed: &mut Editor, win, cx| ed.close_dialog(true, win, cx));
    until!(|ed: &mut Editor, _| ed.busy.is_none());
    shot!("pGPU_gaussian_commit");
    // Back to the sharp photo so the next previews show their own effect.
    step!(|ed: &mut Editor, win, cx| ed.act_undo(false, win, cx));


    // GPU4: Blur Gallery (Iris Blur with bokeh) and Lens Blur live
    // previews, rendered on the GPU and composited without a readback.
    step!(|ed: &mut Editor, win, cx| ed.open_filter_dialog(Kind::IrisBlur, win, cx));
    shot!("pGPU_iris_blur_live_preview");
    step!(|ed: &mut Editor, win, cx| ed.close_dialog(false, win, cx));

    step!(|ed: &mut Editor, win, cx| ed.open_filter_dialog(Kind::LensBlur, win, cx));
    shot!("pGPU_lens_blur_live_preview");
    step!(|ed: &mut Editor, win, cx| ed.close_dialog(false, win, cx));


    // GPU5: a large Forward Warp stroke in Liquify (field edits and the
    // resample on the GPU).
    step!(|ed: &mut Editor, win, cx| {
        ed.set_tool(super::editor::Tool::Liquify, win, cx);
        ed.liquify.size = 900.0;
        ed.liquify.flow = 1.0;
        ed.liquify_mode = crate::engine::liquify::Mode::ForwardWarp;
        ed.demo_press(1500.0, 1300.0, gpui_kit::Modifiers::default(), 1, win, cx);
    });
    for k in 1..=8 {
        step!(move |ed: &mut Editor, win, cx| ed.demo_move(1500.0 + 60.0 * k as f32, 1300.0 - 25.0 * k as f32, true, gpui_kit::Modifiers::default(), win, cx));
        wait(60).await;
    }
    step!(|ed: &mut Editor, win, cx| ed.demo_release(1980.0, 1100.0, win, cx));
    shot!("pGPU_liquify_forward_warp");
    step!(|ed: &mut Editor, win, cx| ed.set_tool(super::editor::Tool::Brush, win, cx));

    // GPU6: many layers, composited on the GPU into textures GPUI draws.
    step!(|ed: &mut Editor, _, _| ed.replace_document(layered_doc()));
    shot!("pGPU_compositing_13_layers");

    // Sets a dialog's sliders (in order) and refreshes its preview.
    fn set_sliders(ed: &mut Editor, vals: &[f32], win: &mut gpui_kit::Window, cx: &mut gpui_kit::Context<Editor>) {
        if let Some(d) = &ed.dialog {
            let states: Vec<_> = d.sliders.iter().map(|s| s.state.clone()).collect();
            for (st, &v) in states.iter().zip(vals) {
                st.update(cx, |s, cx| s.set_value(v, win, cx));
            }
        }
    }

    // GPU7: a Filter Gallery effect, previewed and committed on the GPU.
    step!(|ed: &mut Editor, win, cx| {
        ed.replace_document(doc_photo("dog_grass"));
        ed.open_filter_dialog(Kind::Gallery(crate::engine::filters::gallery::G::AngledStrokes), win, cx);
    });
    shot!("pGPU_gallery_angled_strokes_preview");
    step!(|ed: &mut Editor, win, cx| ed.close_dialog(true, win, cx));
    until!(|ed: &mut Editor, _| ed.busy.is_none());
    shot!("pGPU_gallery_angled_strokes_commit");
    step!(|ed: &mut Editor, win, cx| ed.act_undo(false, win, cx));

    // GPU8: Camera Raw Filter (tone, clarity, dehaze, vibrance, sharpening,
    // noise reduction) previewed and committed on the GPU.
    step!(|ed: &mut Editor, win, cx| {
        ed.open_filter_dialog(Kind::CameraRaw, win, cx);
        set_sliders(ed, &[30.0, 0.0, 0.3, 25.0, -40.0, 40.0, 0.0, 0.0, 30.0, 40.0, 10.0, 30.0], win, cx);
        ed.update_preview(cx);
    });
    shot!("pGPU_camera_raw_preview");
    step!(|ed: &mut Editor, win, cx| ed.close_dialog(true, win, cx));
    until!(|ed: &mut Editor, _| ed.busy.is_none());
    shot!("pGPU_camera_raw_commit");
    step!(|ed: &mut Editor, win, cx| ed.act_undo(false, win, cx));

    // GPU9: Oil Paint (Photoshop needs a GPU for it; here the GPU runs the
    // CPU's algorithm exactly).
    step!(|ed: &mut Editor, win, cx| ed.open_filter_dialog(Kind::OilPaint, win, cx));
    // Let the live preview render first (so its record doesn't land after
    // the commit's).
    wait(1500).await;
    step!(|ed: &mut Editor, win, cx| ed.close_dialog(true, win, cx));
    until!(|ed: &mut Editor, _| ed.busy.is_none());
    shot!("pGPU_oil_paint_commit");
    step!(|ed: &mut Editor, win, cx| ed.act_undo(false, win, cx));

    // GPU10: Image Size 2× with Preserve Details.
    step!(|ed: &mut Editor, _, cx| {
        let (w, h) = (ed.doc.w * 2, ed.doc.h * 2);
        let m = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];
        ed.run_canvas_op("Image Size", m, w, h, cx, move |l| crate::engine::ops::resize_with(l, w, h, crate::engine::resample::Resample::PreserveDetails, 0.0));
    });
    until!(|ed: &mut Editor, _| ed.busy.is_none());
    shot!("pGPU_image_size_preserve_details");
    step!(|ed: &mut Editor, win, cx| ed.act_undo(false, win, cx));

    // GPU11: Select and Mask on Select Subject's selection (radius 16, Smart
    // Radius, Decontaminate Colors).
    step!(|ed: &mut Editor, _, cx| ed.select_subject(cx));
    until!(|ed: &mut Editor, _| ed.busy.is_none());
    step!(|ed: &mut Editor, win, cx| {
        ed.open_select_mask(win, cx);
        set_sliders(ed, &[16.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 100.0, 0.0], win, cx);
    });
    until!(|ed: &mut Editor, _| ed.select_mask.as_ref().is_some_and(|s| !s.computing));
    shot!("pGPU_select_and_mask_preview");
    step!(|ed: &mut Editor, win, cx| ed.close_dialog(true, win, cx));
    until!(|ed: &mut Editor, _| ed.busy.is_none());
    shot!("pGPU_select_and_mask_applied");
}

pub fn doc_photo(name: &str) -> Document {
    let mut d = Document::from_grid(photo(name), "Background");
    d.path = Some(format!("{name}.jpg").into());
    d
}
