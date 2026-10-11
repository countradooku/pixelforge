//! Scripted demos (`PF_DEMO=…`) that drive the real input handlers, so the
//! screenshots they produce show the actual UI paths working.

use gpui_kit::{Context, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels, Point, Window};

use crate::engine::blend::BlendMode;
use crate::engine::brush::BrushSettings;

use super::editor::{Editor, FilterKind, Tool};

// ----- scripted demo (PF_DEMO=1): drives the real input handlers -----------------------

impl Editor {
    pub fn demo_drag(&mut self, pts: &[(f32, f32)], mods: gpui_kit::Modifiers, window: &mut Window, cx: &mut Context<Self>) {
        let sp: Vec<Point<Pixels>> = pts.iter().map(|&(x, y)| self.vp.to_screen(x, y, self.canvas, self.sf)).collect();
        self.on_mouse_down(
            &MouseDownEvent { button: MouseButton::Left, position: sp[0], modifiers: mods, click_count: 1, first_mouse: false },
            window,
            cx,
        );
        for &p in &sp[1..] {
            self.on_mouse_move(&MouseMoveEvent { position: p, pressed_button: Some(MouseButton::Left), modifiers: mods }, window, cx);
        }
        let last = *sp.last().unwrap();
        self.on_mouse_up(&MouseUpEvent { button: MouseButton::Left, position: last, modifiers: mods, click_count: 1 }, window, cx);
    }

    pub fn start_demo(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        cx.spawn_in(window, async move |this, cx| {
            let exec = cx.background_executor().clone();
            let wait = move |ms: u64| exec.timer(std::time::Duration::from_millis(ms));
            let checkpoint = |n: u32| eprintln!("DEMO {n}");
            macro_rules! step {
                ($body:expr) => {
                    if this.update_in(cx, $body).is_err() {
                        return;
                    }
                };
            }
            macro_rules! idle {
                () => {
                    loop {
                        wait(100).await;
                        let busy = this.update(cx, |ed, _| ed.busy.is_some()).unwrap_or(false);
                        if !busy {
                            break;
                        }
                    }
                };
            }
            wait(1500).await;
            step!(|ed: &mut Editor, _w, cx| ed.act_benchmark(cx));
            idle!();
            wait(800).await;
            checkpoint(1);
            wait(2500).await;
            step!(|ed: &mut Editor, w, cx| {
                let s = 16384.0f32;
                ed.set_tool(Tool::Brush, w, cx);
                *ed.brush_of(Tool::Brush) = BrushSettings { size: 700.0, hardness: 0.2, opacity: 0.9, flow: 0.7, spacing: 0.08, ..Default::default() };
                ed.set_fg([235, 40, 70], w, cx);
                let pts: Vec<(f32, f32)> = (0..=200)
                    .map(|i| {
                        let t = i as f32 / 200.0;
                        (s * (0.1 + 0.8 * t), s * (0.5 + 0.28 * (t * 9.0).sin()))
                    })
                    .collect();
                ed.demo_drag(&pts, Default::default(), w, cx);
                ed.brush_of(Tool::Brush).size = 250.0;
                ed.brush_of(Tool::Brush).hardness = 1.0;
                ed.set_fg([30, 120, 255], w, cx);
                let pts: Vec<(f32, f32)> = (0..=200)
                    .map(|i| {
                        let t = i as f32 / 200.0 * std::f32::consts::TAU;
                        (s * 0.5 + s * 0.3 * t.cos(), s * 0.5 + s * 0.3 * (2.0 * t).sin() * 0.6)
                    })
                    .collect();
                ed.demo_drag(&pts, Default::default(), w, cx);
                ed.sync_brush_sliders(w, cx);
            });
            wait(800).await;
            checkpoint(2);
            wait(2500).await;
            step!(|ed: &mut Editor, w, cx| {
                ed.select_layer(0, w, cx);
                ed.set_tool(Tool::EllipseSelect, w, cx);
                ed.demo_drag(&[(4000.0, 4000.0), (9000.0, 8000.0), (12400.0, 12400.0)], Default::default(), w, cx);
                ed.open_filter_dialog(FilterKind::GaussianBlur, w, cx);
                if let Some(d) = &ed.dialog {
                    let st = d.sliders[0].state.clone();
                    st.update(cx, |s, cx| s.set_value(60.0, w, cx));
                }
                ed.update_preview(cx);
            });
            wait(1200).await;
            checkpoint(3);
            wait(2500).await;
            step!(|ed: &mut Editor, w, cx| ed.close_dialog(true, w, cx));
            idle!();
            step!(|ed: &mut Editor, w, cx| {
                ed.act_select(1, cx);
                ed.act_layer("new", w, cx);
                ed.set_tool(Tool::Gradient, w, cx);
                ed.gradient_radial = true;
                ed.set_fg([255, 210, 0], w, cx);
                ed.set_bg([40, 0, 120], w, cx);
                ed.demo_drag(&[(8192.0, 8192.0), (16000.0, 16000.0)], Default::default(), w, cx);
            });
            idle!();
            step!(|ed: &mut Editor, w, cx| {
                let li = ed.doc.active;
                ed.doc.layers[li].blend = BlendMode::Overlay;
                ed.doc.layers[li].opacity = 0.75;
                ed.doc.mark_all_dirty();
                ed.sync_layer_controls(w, cx);
                ed.set_tool(Tool::Brush, w, cx);
            });
            wait(800).await;
            checkpoint(4);
            wait(2500).await;
            step!(|ed: &mut Editor, _w, cx| {
                let c = ed.canvas.center();
                ed.vp.center_at_zoom(1.0, 16384, 16384, ed.canvas, ed.sf);
                let target = ed.vp.to_screen(3300.0, 8192.0, ed.canvas, ed.sf);
                ed.vp.pan = ed.vp.pan + (c - target);
                ed.vp.zoom_about(8.0, c, ed.canvas);
                cx.notify();
            });
            wait(1200).await;
            checkpoint(5);
            wait(2500).await;
            step!(|ed: &mut Editor, w, cx| {
                ed.act_zoom(2, cx);
                // Move the paint layer right by 2000 px.
                ed.select_layer(2, w, cx);
                ed.set_tool(Tool::Move, w, cx);
                ed.demo_drag(&[(8000.0, 8000.0), (9000.0, 8000.0), (10000.0, 8000.0)], Default::default(), w, cx);
            });
            idle!();
            step!(|ed: &mut Editor, w, cx| {
                // Hide the overlay layer, rotate the canvas, then magic-wand the background.
                ed.toggle_visibility(1, cx);
                ed.select_layer(0, w, cx);
                ed.act_canvas("rot_cw", cx);
            });
            idle!();
            step!(|ed: &mut Editor, w, cx| {
                ed.act_zoom(2, cx);
                ed.set_tool(Tool::MagicWand, w, cx);
                ed.tolerance = 40.0;
                let p = ed.vp.to_screen(800.0, 800.0, ed.canvas, ed.sf);
                let mods = Default::default();
                ed.on_mouse_down(&MouseDownEvent { button: MouseButton::Left, position: p, modifiers: mods, click_count: 1, first_mouse: false }, w, cx);
                ed.on_mouse_up(&MouseUpEvent { button: MouseButton::Left, position: p, modifiers: mods, click_count: 1 }, w, cx);
                eprintln!("doc {}x{} layers={} history={:?}", ed.doc.w, ed.doc.h, ed.doc.layers.len(), ed.doc.history_labels().0);
            });
            wait(1500).await;
            checkpoint(6);
            wait(1500).await;
            step!(|ed: &mut Editor, w, cx| {
                for _ in 0..3 {
                    ed.act_undo(false, w, cx);
                }
                eprintln!("after undo: doc {}x{} history={:?}", ed.doc.w, ed.doc.h, ed.doc.history_labels().0);
            });
            wait(1500).await;
            checkpoint(7);
        })
        .detach();
    }
}

impl Editor {
    /// PF_DEMO=filters: exercises catalog dialogs with live preview + commit.
    pub fn start_filter_demo(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        cx.spawn_in(window, async move |this, cx| {
            let exec = cx.background_executor().clone();
            let wait = move |ms: u64| exec.timer(std::time::Duration::from_millis(ms));
            macro_rules! idle {
                () => {
                    loop {
                        wait(100).await;
                        if !this.update(cx, |ed, _| ed.busy.is_some()).unwrap_or(false) {
                            break;
                        }
                    }
                };
            }
            wait(1200).await;
            let _ = this.update(cx, |ed, cx| ed.act_benchmark(cx));
            idle!();
            let _ = this.update_in(cx, |ed, w, cx| ed.select_layer(0, w, cx));
            use crate::engine::filters::gallery::G;
            let steps: &[(FilterKind, &[f32])] = &[
                (FilterKind::FreeTransform, &[0.0, 0.0, 80.0, 80.0, 15.0, 0.0, 0.0, 30.0, 0.0]),
                (FilterKind::PathBlur, &[10.0, 80.0, 90.0, 20.0, 50.0, 300.0]),
                (FilterKind::CameraRaw, &[30.0, 0.0, 0.3, 25.0, -40.0, 40.0, 0.0, 0.0, 30.0, 40.0, 10.0, 30.0]),
                (FilterKind::Gallery(G::Watercolor), &[9.0, 1.0, 1.0]),
            ];
            for (n, &(kind, vals)) in steps.iter().enumerate() {
                let _ = this.update_in(cx, |ed, w, cx| {
                    ed.open_filter_dialog(kind, w, cx);
                    if let Some(d) = &ed.dialog {
                        let states: Vec<_> = d.sliders.iter().map(|s| s.state.clone()).collect();
                        for (st, &v) in states.iter().zip(vals) {
                            st.update(cx, |s, cx| s.set_value(v, w, cx));
                        }
                    }
                    ed.update_preview(cx);
                });
                wait(1500).await;
                eprintln!("DEMO {}", n * 2 + 1);
                wait(2000).await;
                let _ = this.update_in(cx, |ed, w, cx| ed.close_dialog(true, w, cx));
                idle!();
                wait(800).await;
                eprintln!("DEMO {}", n * 2 + 2);
                wait(1500).await;
            }
            // Liquify session: freeze the left half of the center, then hold
            // Twirl over the middle so it acts continuously.
            use crate::engine::liquify::Mode;
            let press = |ed: &mut Editor, x: f32, y: f32, w: &mut Window, cx: &mut Context<Editor>| {
                let p = ed.vp.to_screen(x, y, ed.canvas, ed.sf);
                ed.mouse = Some(p);
                ed.mouse_img = Some((x, y));
                ed.on_mouse_down(&MouseDownEvent { button: MouseButton::Left, position: p, modifiers: Default::default(), click_count: 1, first_mouse: false }, w, cx);
            };
            let release = |ed: &mut Editor, w: &mut Window, cx: &mut Context<Editor>| {
                let p = ed.mouse.unwrap_or_default();
                ed.on_mouse_up(&MouseUpEvent { button: MouseButton::Left, position: p, modifiers: Default::default(), click_count: 1 }, w, cx);
            };
            let _ = this.update_in(cx, |ed, w, cx| {
                ed.set_tool(Tool::Liquify, w, cx);
                ed.liquify.size = 3000.0;
                ed.liquify.flow = 1.0;
                ed.liquify.hardness = 1.0;
                ed.liquify_mode = Mode::Freeze;
                press(ed, 6500.0, 8192.0, w, cx);
            });
            wait(1200).await;
            let _ = this.update_in(cx, |ed, w, cx| {
                release(ed, w, cx);
                ed.liquify.size = 7000.0;
                ed.liquify_mode = Mode::TwirlCw;
                press(ed, 8192.0, 8192.0, w, cx);
                ed.sync_brush_sliders(w, cx);
            });
            wait(2500).await;
            let _ = this.update_in(cx, |ed, w, cx| release(ed, w, cx));
            wait(800).await;
            eprintln!("DEMO 9");
        })
        .detach();
    }
}

// ----- PF_DEMO=tools: every tool through the real input handlers ---------------------------

/// Scene geometry (matches `ops::demo_scene`).
#[derive(Clone, Copy)]
pub struct Scene {
    pub w: f32,
    pub h: f32,
}

impl Scene {
    pub fn ball(&self) -> (f32, f32, f32) {
        (0.30 * self.w, 0.62 * self.h, 0.14 * self.h)
    }
    pub fn boxr(&self) -> (f32, f32, f32, f32) {
        (0.55 * self.w, 0.42 * self.h, 0.76 * self.w, 0.80 * self.h)
    }
}

impl Editor {
    fn pt(&self, x: f32, y: f32) -> Point<Pixels> {
        self.vp.to_screen(x, y, self.canvas, self.sf)
    }

    pub fn demo_press(&mut self, x: f32, y: f32, mods: gpui_kit::Modifiers, clicks: usize, window: &mut Window, cx: &mut Context<Self>) {
        let p = self.pt(x, y);
        self.mouse = Some(p);
        self.mouse_img = Some((x, y));
        self.on_mouse_down(&MouseDownEvent { button: MouseButton::Left, position: p, modifiers: mods, click_count: clicks, first_mouse: false }, window, cx);
    }

    pub fn demo_move(&mut self, x: f32, y: f32, pressed: bool, mods: gpui_kit::Modifiers, window: &mut Window, cx: &mut Context<Self>) {
        let p = self.pt(x, y);
        self.on_mouse_move(&MouseMoveEvent { position: p, pressed_button: pressed.then_some(MouseButton::Left), modifiers: mods }, window, cx);
    }

    pub fn demo_release(&mut self, x: f32, y: f32, window: &mut Window, cx: &mut Context<Self>) {
        let p = self.pt(x, y);
        self.on_mouse_up(&MouseUpEvent { button: MouseButton::Left, position: p, modifiers: Default::default(), click_count: 1 }, window, cx);
    }

    pub fn demo_click(&mut self, x: f32, y: f32, mods: gpui_kit::Modifiers, clicks: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.demo_press(x, y, mods, clicks, window, cx);
        self.demo_release(x, y, window, cx);
    }

    pub fn start_tools_demo(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Phases: 0–5 (the toolbar tour) and A–G (the gap-closing features).
        let phases: Vec<String> = std::env::var("PF_DEMO_PHASE").ok().map(|v| v.split(',').map(|p| p.trim().to_string()).collect()).unwrap_or_else(|| ["0", "1", "2", "3", "4", "5"].map(String::from).to_vec());
        cx.spawn_in(window, async move |this, cx| {
            let exec = cx.background_executor().clone();
            let wait = move |ms: u64| exec.timer(std::time::Duration::from_millis(ms));
            macro_rules! step {
                ($body:expr) => {
                    if this.update_in(cx, $body).is_err() {
                        return;
                    }
                };
            }
            macro_rules! idle {
                () => {
                    loop {
                        wait(100).await;
                        let busy = this.update(cx, |ed, _| ed.busy.is_some()).unwrap_or(false);
                        if !busy {
                            break;
                        }
                    }
                };
            }
            macro_rules! shot {
                ($name:expr) => {
                    wait(700).await;
                    eprintln!("SHOT {}", $name);
                    wait(1300).await;
                };
            }
            let shift = gpui_kit::Modifiers { shift: true, ..Default::default() };
            let alt = gpui_kit::Modifiers { alt: true, ..Default::default() };
            let none = gpui_kit::Modifiers::default();
            let (w, h) = (4000u32, 3000u32);
            let sc = Scene { w: w as f32, h: h as f32 };
            wait(1200).await;
            step!(|ed: &mut Editor, _w, cx| {
                let g = crate::engine::ops::demo_scene(w, h);
                ed.replace_document(crate::engine::document::Document::from_grid(g, "Background"));
                cx.notify();
            });
            wait(600).await;

            if phases.iter().any(|p| p == "psd") {
                step!(|ed: &mut Editor, _, cx| {
                    let input = std::path::PathBuf::from(std::env::var_os("PF_PSD_FIXTURE").expect("PSD demo needs PF_PSD_FIXTURE"));
                    let directory = crate::engine::prefs::config_dir();
                    std::fs::create_dir_all(&directory).expect("demo directory");
                    let copy = directory.join("Phase 2 layers.psd");
                    std::fs::copy(input, &copy).expect("copy isolated PSD fixture");
                    ed.open_path(copy, cx);
                });
                idle!();
                step!(|ed: &mut Editor, _, cx| {
                    assert!(ed.doc.psd.is_some(), "PSD fixture did not open");
                    ed.dockst.tab = Some(super::dock::DockTab::Channels);
                    ed.act_zoom(2, cx);
                });
                shot!("phase2_psd_open");
                wait(12000).await;
                step!(|ed: &mut Editor, _, cx| {
                    ed.doc.push_undo("Rename PSD layer");
                    ed.doc.layers[ed.doc.active].name = "Edited in PixelForge".into();
                    ed.doc.modified = true;
                    ed.act_save(false, cx);
                });
                idle!();
                shot!("phase2_psd_saved");
                wait(12000).await;
                step!(|_, _, cx| cx.quit());
                return;
            }

            if phases.iter().any(|p| p == "memory") {
                step!(|ed: &mut Editor, _win, cx| {
                    ed.spawn_job("Memory foundation", cx, |_| {
                        let g = crate::engine::ops::procedural(14204, 10652);
                        let mut doc = crate::engine::document::Document::from_grid(g, "150 MP memory test");
                        for i in 0..4 {
                            doc.push_undo(format!("Edit {i}"));
                            let g = crate::engine::ops::fill(&doc.layers[0].grid, None, [40 + i * 20, 80, 120], 0.2);
                            doc.set_layer_grid(0, g);
                        }
                        doc.maintain_history(); crate::engine::memory::global().maintain();
                        let dir = crate::engine::prefs::config_dir();
                        std::fs::create_dir_all(&dir).expect("demo directory");
                        let path = dir.join("Phase 1 memory.pforge");
                        crate::engine::io::save_native(&doc.layers, doc.w, doc.h, doc.active, &Default::default(), &path).expect("save demo");
                        doc.path = Some(path);
                        Box::new(move |ed: &mut Editor, cx| { ed.replace_document(doc); ed.act_zoom(2, cx); })
                    });
                });
                idle!();
                shot!("phase1_memory_150mp");
                step!(|ed: &mut Editor, _win, cx| ed.open_preferences(cx));
                shot!("phase1_memory_preferences");
                step!(|_, _, cx| cx.quit());
                return;
            }

            if phases.iter().any(|p| p == "0") {
                // Layer kinds: shape layer, clipped smart object, adjustment layer with mask, group.
                step!(|ed: &mut Editor, win, cx| {
                    use crate::engine::layers::{LayerKind, Paint, ShapeGeom, ShapeLayer, SmartObject};
                    use crate::engine::vector::StrokeStyle;
                    let shape = ShapeLayer::new(ShapeGeom::Rect { x0: 300.0, y0: 300.0, x1: 1500.0, y1: 1100.0, radii: [120.0; 4] }, Paint::Solid([250, 250, 250]), Some((Paint::Solid([20, 20, 20]), StrokeStyle { width: 24.0, ..Default::default() })));
                    ed.doc.push_undo("Shape");
                    let at = ed.doc.insert_layer("Rounded Frame", crate::engine::grid::TileGrid::new(w, h));
                    let l = std::mem::replace(&mut ed.doc.layers[at], crate::engine::document::Layer::new(0, "", crate::engine::grid::TileGrid::new(1, 1)));
                    ed.doc.layers[at] = l.with_kind(LayerKind::Shape(std::sync::Arc::new(shape)));
                    // A smart object (the scene itself, scaled) clipped to the shape.
                    let src = crate::engine::ops::procedural(2048, 1536);
                    let m = [0.62, 0.0, 280.0, 0.0, 0.62, 260.0, 0.0, 0.0, 1.0];
                    ed.add_smart_layer(SmartObject::new("Placed", src, m));
                    let li = ed.doc.active;
                    ed.doc.layers[li].clip = true;
                    // Group the shape with the image clipped to it.
                    ed.select_layer(li - 1, win, cx);
                    ed.group_layer(win, cx);
                    // Hue/Saturation adjustment layer masked to an ellipse.
                    ed.set_tool(Tool::EllipseSelect, win, cx);
                    let b = sc.boxr();
                    ed.demo_drag(&[(b.0 - 100.0, b.1 - 100.0), (b.2 + 100.0, b.3 + 100.0)], none, win, cx);
                    let k = crate::engine::filters::catalog::Kind::HueSaturation;
                    ed.new_adjustment_layer(k, win, cx);
                    if let Some(d) = &ed.dialog {
                        let st = d.sliders[0].state.clone();
                        st.update(cx, |s, cx| s.set_value(150.0, win, cx));
                    }
                    ed.update_preview(cx);
                    ed.close_dialog(true, win, cx);
                    ed.act_select(1, cx);
                    eprintln!("layers: {:?}", ed.doc.layers.iter().map(|l| (l.name.clone(), l.kind.name(), l.clip, l.parent, l.mask.is_some())).collect::<Vec<_>>());
                });
                shot!("p0_layer_kinds");
                step!(|ed: &mut Editor, win, cx| {
                    // Back to a single layer for the selection tour.
                    while ed.doc.layers.len() > 1 {
                        let n = ed.doc.layers.len();
                        ed.doc.layers.remove(n - 1);
                    }
                    ed.doc.active = 0;
                    ed.doc.mark_all_dirty();
                    ed.select_layer(0, win, cx);
                });
            }

            if phases.iter().any(|p| p == "1") {
                step!(|ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::Lasso, win, cx);
                    ed.flyout = Some((Tool::Lasso.group(), 0.0));
                    cx.notify();
                });
                shot!("p1_toolbar_flyout");
                step!(|ed: &mut Editor, win, cx| {
                    ed.flyout = None;
                    ed.set_tool(Tool::RectSelect, win, cx);
                    ed.demo_drag(&[(400.0, 400.0), (1600.0, 1400.0)], none, win, cx);
                    ed.set_tool(Tool::EllipseSelect, win, cx);
                    ed.demo_drag(&[(1200.0, 1000.0), (2200.0, 1900.0)], shift, win, cx);
                    ed.set_tool(Tool::RectSelect, win, cx);
                    ed.demo_drag(&[(700.0, 700.0), (1000.0, 1000.0)], alt, win, cx);
                    ed.set_tool(Tool::RowSelect, win, cx);
                    ed.demo_click(2000.0, 2500.0, shift, 1, win, cx);
                    ed.set_tool(Tool::ColumnSelect, win, cx);
                    ed.demo_click(3500.0, 2500.0, shift, 1, win, cx);
                });
                shot!("p1_marquee_add_subtract_row_column");
                step!(|ed: &mut Editor, win, cx| {
                    ed.act_select(1, cx);
                    ed.set_tool(Tool::PolyLasso, win, cx);
                    let b = sc.boxr();
                    for &(x, y) in &[(b.0, b.1), (b.2, b.1), (b.2, b.3)] {
                        ed.demo_click(x, y, none, 1, win, cx);
                    }
                    ed.demo_move(b.0 + 50.0, b.3 - 60.0, false, none, win, cx);
                });
                shot!("p1_poly_lasso_rubber_band");
                step!(|ed: &mut Editor, win, cx| {
                    let b = sc.boxr();
                    ed.demo_click(b.0, b.3, none, 1, win, cx);
                    ed.demo_click(b.0, b.1, none, 1, win, cx);
                });
                shot!("p1_poly_lasso_closed");
                step!(|ed: &mut Editor, win, cx| {
                    ed.act_select(1, cx);
                    ed.set_tool(Tool::MagneticLasso, win, cx);
                    let (bx, by, br) = sc.ball();
                    ed.demo_click(bx + br, by, none, 1, win, cx);
                    // Trace sloppily ~15 px outside the ball for most of the way round.
                    let n = 120;
                    let mut ms = Vec::new();
                    for k in 1..=n {
                        let t = k as f32 / n as f32 * std::f32::consts::TAU * 0.8;
                        let wobble = 15.0 + 10.0 * (t * 7.0).sin();
                        ed.demo_move(bx + (br + wobble) * t.cos(), by + (br + wobble) * t.sin(), false, none, win, cx);
                        ms.push(ed.last_tool_ms);
                    }
                    ms.sort_by(|a, b| a.total_cmp(b));
                    eprintln!("magnetic lasso per move: median {:.2} ms, max {:.2} ms", ms[ms.len() / 2], ms[ms.len() - 1]);
                });
                shot!("p1_magnetic_live_wire");
                step!(|ed: &mut Editor, win, cx| {
                    ed.finish_magnetic();
                    ed.set_fg([255, 60, 200], win, cx);
                    cx.notify();
                });
                shot!("p1_magnetic_closed");
                step!(|ed: &mut Editor, win, cx| {
                    ed.act_select(1, cx);
                    ed.set_tool(Tool::ObjectSelect, win, cx);
                    let (bx, by, br) = sc.ball();
                    ed.demo_drag(&[(bx - br * 1.3, by - br * 1.3), (bx + br * 1.3, by + br * 1.3)], none, win, cx);
                    eprintln!("object selection first result {:.0} ms", ed.last_tool_ms);
                });
                idle!();
                shot!("p1_object_selection");
                step!(|ed: &mut Editor, win, cx| {
                    ed.act_select(1, cx);
                    ed.set_tool(Tool::QuickSelect, win, cx);
                    ed.brush_of(Tool::QuickSelect).size = 80.0;
                    let b = sc.boxr();
                    let pts: Vec<(f32, f32)> = (0..=30).map(|k| (b.0 + 120.0 + k as f32 * 15.0, b.1 + 150.0 + (k as f32 * 0.4).sin() * 80.0)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    eprintln!("quick selection coarse update {:.0} ms", ed.last_tool_ms);
                });
                idle!();
                shot!("p1_quick_selection");
                step!(|ed: &mut Editor, win, cx| {
                    ed.act_select(1, cx);
                    ed.set_tool(Tool::MagicWand, win, cx);
                    ed.tolerance = 40.0;
                    ed.opts.sample_all = true;
                    ed.demo_click(800.0, 300.0, none, 1, win, cx);
                });
                shot!("p1_magic_wand_sky");
                step!(|ed: &mut Editor, _win, cx| {
                    ed.apply_modify(super::select_tools::Modify::Border, 60.0, cx);
                });
                idle!();
                shot!("p1_modify_border");
                step!(|ed: &mut Editor, win, cx| {
                    ed.act_select(1, cx);
                    let (bx, by, br) = sc.ball();
                    ed.set_tool(Tool::EllipseSelect, win, cx);
                    ed.demo_drag(&[(bx - br, by - br), (bx + br, by + br)], none, win, cx);
                    ed.toggle_quick_mask(cx);
                    ed.set_tool(Tool::Brush, win, cx);
                    ed.brush_of(Tool::Brush).size = 160.0;
                    ed.brush_of(Tool::Brush).hardness = 0.3;
                    ed.set_fg([0, 0, 0], win, cx);
                    let pts: Vec<(f32, f32)> = (0..=40).map(|k| (bx - br + k as f32 * br / 20.0, by + (k as f32 * 0.3).sin() * 60.0)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                });
                shot!("p1_quick_mask_painting");
                step!(|ed: &mut Editor, win, cx| {
                    ed.toggle_quick_mask(cx);
                    ed.act_layer("new", win, cx);
                    ed.act_fill(false, cx);
                });
                idle!();
                shot!("p1_quick_mask_to_selection_filled");
            }
            if phases.iter().any(|p| p == "2") {
                // Fresh scene.
                step!(|ed: &mut Editor, _w, cx| {
                    let g = crate::engine::ops::demo_scene(w, h);
                    ed.replace_document(crate::engine::document::Document::from_grid(g, "Background"));
                    cx.notify();
                });
                wait(600).await;
                // Clone Stamp: Alt-click the ball, hover to show the overlay.
                step!(|ed: &mut Editor, win, cx| {
                    let (bx, by, _) = sc.ball();
                    ed.set_tool(Tool::CloneStamp, win, cx);
                    ed.brush_of(Tool::CloneStamp).size = 420.0;
                    ed.demo_click(bx, by, alt, 1, win, cx);
                    ed.demo_move(bx + 1500.0, by - 900.0, false, none, win, cx);
                    ed.mouse = Some(ed.vp.to_screen(bx + 1500.0, by - 900.0, ed.canvas, ed.sf));
                    ed.mouse_img = Some((bx + 1500.0, by - 900.0));
                    cx.notify();
                });
                shot!("p2_clone_overlay_preview");
                step!(|ed: &mut Editor, win, cx| {
                    let (bx, by, _) = sc.ball();
                    let pts: Vec<(f32, f32)> = (0..=40).map(|k| (bx + 1300.0 + k as f32 * 10.0, by - 1000.0 + (k as f32 * 0.4).sin() * 60.0)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    let pts: Vec<(f32, f32)> = (0..=40).map(|k| (bx + 1400.0, by - 1200.0 + k as f32 * 12.0)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                });
                shot!("p2_clone_stamp_result");
                step!(|ed: &mut Editor, _win, cx| {
                    ed.act_undo(false, _win, cx);
                    ed.act_undo(false, _win, cx);
                });
                // Healing Brush over three blemishes on the left (source: clean ground).
                let blem = crate::engine::ops::demo_blemishes(w, h);
                let heal_targets: Vec<(f32, f32, f32)> = blem.iter().copied().filter(|b| b.0 < sc.w * 0.2).take(3).collect();
                let heal_t2 = heal_targets.clone();
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::Heal, win, cx);
                    ed.ropts.aligned = false;
                    ed.demo_click(0.5 * sc.w, 0.95 * sc.h, alt, 1, win, cx);
                    for &(x, y, r) in &heal_t2 {
                        ed.brush_of(Tool::Heal).size = r * 2.0 + 40.0;
                        ed.demo_drag(&[(x - 8.0, y), (x, y), (x + 8.0, y)], none, win, cx);
                    }
                });
                shot!("p2_healing_brush");
                // Spot Healing: one click-drag per remaining blemish left of x = 0.78 w.
                step!(|ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::SpotHeal, win, cx);
                });
                for &(x, y, r) in blem.iter().filter(|b| b.0 < sc.w * 0.78 && !heal_targets.iter().any(|t| t.0 == b.0)) {
                    step!(move |ed: &mut Editor, win, cx| {
                        ed.brush_of(Tool::SpotHeal).size = r * 2.0 + 28.0;
                        ed.demo_drag(&[(x - 2.0, y), (x, y), (x + 2.0, y)], none, win, cx);
                    });
                    idle!();
                }
                shot!("p2_spot_healing");
                // Patch: lasso around the blemishes on the right, drag onto clean ground.
                let right: Vec<(f32, f32, f32)> = blem.iter().copied().filter(|b| b.0 >= sc.w * 0.78).collect();
                let (rcx, rcy) = right.iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
                let (rcx, rcy) = (rcx / right.len().max(1) as f32, rcy / right.len().max(1) as f32);
                eprintln!("patch around {} blemishes at {rcx:.0},{rcy:.0}", right.len());
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::Patch, win, cx);
                    let (cx0, cy0) = (rcx, rcy);
                    let lasso: Vec<(f32, f32)> = (0..=36).map(|k| {
                        let t = k as f32 / 36.0 * std::f32::consts::TAU;
                        (cx0 + 380.0 * t.cos(), cy0 + 330.0 * t.sin())
                    }).collect();
                    ed.demo_drag(&lasso, none, win, cx);
                    cx.notify();
                });
                shot!("p2_patch_selection");
                step!(move |ed: &mut Editor, win, cx| {
                    let (cx0, cy0) = (rcx, rcy);
                    ed.demo_drag(&[(cx0, cy0), (cx0 - 900.0, cy0 + 20.0), (cx0 - 1925.0, cy0 + 39.0)], none, win, cx);
                });
                idle!();
                shot!("p2_patch_tool");
                // Content-Aware Move: move the sun to the left.
                step!(|ed: &mut Editor, win, cx| {
                    ed.act_select(1, cx);
                    ed.set_tool(Tool::ContentAwareMove, win, cx);
                    let (sx, sy, sr) = (0.83 * sc.w, 0.17 * sc.h, 0.07 * sc.h + 30.0);
                    let lasso: Vec<(f32, f32)> = (0..=36).map(|k| {
                        let t = k as f32 / 36.0 * std::f32::consts::TAU;
                        (sx + sr * t.cos(), sy + sr * t.sin())
                    }).collect();
                    ed.demo_drag(&lasso, none, win, cx);
                    ed.demo_drag(&[(sx, sy), (sx - 600.0, sy + 50.0), (sx - 1300.0, sy + 120.0)], none, win, cx);
                });
                idle!();
                shot!("p2_content_aware_move");
                // Red Eye: paint an eye, then fix it.
                step!(|ed: &mut Editor, win, cx| {
                    ed.act_select(1, cx);
                    ed.set_tool(Tool::Brush, win, cx);
                    let (ex, ey) = (0.12 * sc.w, 0.2 * sc.h);
                    *ed.brush_of(Tool::Brush) = BrushSettings { size: 420.0, hardness: 0.8, ..Default::default() };
                    ed.set_fg([235, 200, 180], win, cx);
                    ed.demo_drag(&[(ex, ey)], none, win, cx);
                    *ed.brush_of(Tool::Brush) = BrushSettings { size: 180.0, hardness: 0.9, ..Default::default() };
                    ed.set_fg([90, 140, 90], win, cx);
                    ed.demo_drag(&[(ex, ey)], none, win, cx);
                    *ed.brush_of(Tool::Brush) = BrushSettings { size: 90.0, hardness: 0.85, ..Default::default() };
                    ed.set_fg([210, 30, 35], win, cx);
                    ed.demo_drag(&[(ex, ey)], none, win, cx);
                    *ed.brush_of(Tool::Brush) = BrushSettings { size: 18.0, hardness: 1.0, ..Default::default() };
                    ed.set_fg([255, 255, 255], win, cx);
                    ed.demo_drag(&[(ex + 14.0, ey - 14.0)], none, win, cx);
                    // Zoom in on the eye.
                    ed.vp.center_at_zoom(1.0, w, h, ed.canvas, ed.sf);
                    let c = ed.canvas.center();
                    let target = ed.vp.to_screen(ex + 300.0, ey, ed.canvas, ed.sf);
                    ed.vp.pan = ed.vp.pan + (c - target);
                    ed.vp.zoom_about(1.6, c, ed.canvas);
                    cx.notify();
                });
                shot!("p2_red_eye_before");
                step!(|ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::RedEye, win, cx);
                    let (ex, ey) = (0.12 * sc.w, 0.2 * sc.h);
                    ed.demo_click(ex, ey + 10.0, none, 1, win, cx);
                    eprintln!("red eye {:.1} ms", ed.last_tool_ms);
                    // A second, untouched copy to compare: duplicate eye on the right.
                    cx.notify();
                });
                shot!("p2_red_eye_after");
                // History brush: invert everything, then paint the original back.
                step!(|ed: &mut Editor, win, cx| {
                    ed.act_zoom(2, cx);
                    ed.ropts.history_pos = ed.doc.history_len();
                    ed.act_quick_filter(crate::engine::filters::Filter::Invert, cx);
                    let _ = win;
                });
                idle!();
                step!(|ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::HistoryBrush, win, cx);
                    ed.brush_of(Tool::HistoryBrush).size = 500.0;
                    let pts: Vec<(f32, f32)> = (0..=60).map(|k| (sc.w * (0.1 + 0.8 * k as f32 / 60.0), sc.h * 0.5 + (k as f32 * 0.3).sin() * 300.0)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    ed.set_tool(Tool::ArtHistory, win, cx);
                    ed.brush_of(Tool::ArtHistory).size = 160.0;
                    ed.ropts.art_area = 120.0;
                    let pts: Vec<(f32, f32)> = (0..=40).map(|k| (sc.w * (0.2 + 0.6 * k as f32 / 40.0), sc.h * 0.85)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    ed.set_tool(Tool::PatternStamp, win, cx);
                    ed.brush_of(Tool::PatternStamp).size = 300.0;
                    ed.ropts.pattern = 1;
                    let pts: Vec<(f32, f32)> = (0..=30).map(|k| (sc.w * (0.25 + 0.5 * k as f32 / 30.0), sc.h * 0.18)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                });
                shot!("p2_history_art_history_pattern");
            }
            if phases.iter().any(|p| p == "3") {
                step!(|ed: &mut Editor, _w, cx| {
                    let g = crate::engine::ops::demo_scene(w, h);
                    ed.replace_document(crate::engine::document::Document::from_grid(g, "Background"));
                    cx.notify();
                });
                wait(600).await;
                let (bx, by, br) = sc.ball();
                let bxr = sc.boxr();
                // Color Replacement on the ball, Dodge on its highlight, Burn under it, Sponge on the ground.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::ColorReplace, win, cx);
                    ed.set_fg([40, 90, 230], win, cx);
                    ed.brush_of(Tool::ColorReplace).size = 260.0;
                    ed.popts.paint_tolerance = 60.0;
                    let pts: Vec<(f32, f32)> = (0..=60).map(|k| {
                        let t = k as f32 / 60.0 * std::f32::consts::TAU * 2.0;
                        (bx + (br * 0.55) * (t * 0.5).cos() * (k as f32 / 60.0), by + (br * 0.55) * (t * 0.5).sin() * (k as f32 / 60.0))
                    }).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    ed.set_tool(Tool::Burn, win, cx);
                    ed.brush_of(Tool::Burn).size = 500.0;
                    ed.popts.range = crate::engine::paint::Range::Midtones;
                    ed.popts.exposure = 0.6;
                    let pts: Vec<(f32, f32)> = (0..=30).map(|k| (bx - br + k as f32 * br / 15.0, by + br + 120.0)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    ed.set_tool(Tool::Dodge, win, cx);
                    ed.brush_of(Tool::Dodge).size = 600.0;
                    let pts: Vec<(f32, f32)> = (0..=30).map(|k| (sc.w * 0.05 + k as f32 * 60.0, sc.h * 0.9)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    ed.set_tool(Tool::Sponge, win, cx);
                    ed.popts.sponge_saturate = false;
                    ed.popts.exposure = 1.0;
                    ed.brush_of(Tool::Sponge).size = 500.0;
                    let pts: Vec<(f32, f32)> = (0..=30).map(|k| (sc.w * 0.55 + k as f32 * 50.0, sc.h * 0.9)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                });
                shot!("p3_color_replace_dodge_burn_sponge");
                // Blur the box's left edge, Sharpen the ground, Smudge the sun, Mixer across the box.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::Blur, win, cx);
                    ed.popts.focus_strength = 1.0;
                    ed.brush_of(Tool::Blur).size = 200.0;
                    for _ in 0..3 {
                        let pts: Vec<(f32, f32)> = (0..=30).map(|k| (bxr.0, bxr.1 + k as f32 * (bxr.3 - bxr.1) / 30.0)).collect();
                        ed.demo_drag(&pts, none, win, cx);
                    }
                    ed.set_tool(Tool::Smudge, win, cx);
                    ed.popts.smudge_strength = 0.85;
                    ed.brush_of(Tool::Smudge).size = 180.0;
                    let (sx, sy) = (0.83 * sc.w, 0.17 * sc.h);
                    for d in [-1.0f32, 0.0, 1.0] {
                        let pts: Vec<(f32, f32)> = (0..=40).map(|k| (sx + d * 90.0 + k as f32 * 2.0, sy + k as f32 * 14.0)).collect();
                        ed.demo_drag(&pts, none, win, cx);
                    }
                    ed.set_tool(Tool::Mixer, win, cx);
                    ed.popts.mixer_color = Some([250, 220, 40]);
                    ed.popts.load_after = false;
                    ed.popts.wet = 0.8;
                    ed.popts.mix = 0.5;
                    ed.brush_of(Tool::Mixer).size = 160.0;
                    let pts: Vec<(f32, f32)> = (0..=60).map(|k| (bxr.0 - 200.0 + k as f32 * 22.0, (bxr.1 + bxr.3) * 0.5 + (k as f32 * 0.25).sin() * 120.0)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    ed.set_tool(Tool::Pencil, win, cx);
                    ed.set_fg([0, 0, 0], win, cx);
                    ed.brush_of(Tool::Pencil).size = 12.0;
                    let pts: Vec<(f32, f32)> = (0..=40).map(|k| (sc.w * 0.1 + k as f32 * 30.0, sc.h * 0.1 + (k as f32 * 0.5).sin() * 60.0)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                });
                shot!("p3_blur_smudge_mixer_pencil");
                // Erasers on a duplicate layer over a hidden background.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.act_layer("duplicate", win, cx);
                    ed.toggle_visibility(0, cx);
                    ed.set_tool(Tool::MagicEraser, win, cx);
                    ed.tolerance = 40.0;
                    ed.demo_click(300.0, 300.0, none, 1, win, cx);
                    ed.set_tool(Tool::BackgroundEraser, win, cx);
                    ed.popts.sampling = crate::engine::paint::Sampling::Continuous;
                    ed.popts.paint_tolerance = 50.0;
                    ed.brush_of(Tool::BackgroundEraser).size = 300.0;
                    let pts: Vec<(f32, f32)> = (0..=40).map(|k| (bxr.0 - 100.0 + k as f32 * 25.0, sc.h * 0.7)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    eprintln!("magic eraser {:.1} ms", ed.last_tool_ms);
                });
                shot!("p3_magic_and_background_eraser");
                // Gradient: live spectrum radial with handles and the stop editor.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.act_layer("delete", win, cx);
                    ed.toggle_visibility(0, cx);
                    ed.act_layer("new", win, cx);
                    ed.set_tool(Tool::Gradient, win, cx);
                    ed.popts.grad = crate::engine::gradient::Gradient::presets(ed.fg, ed.bg)[3].clone();
                    ed.popts.grad_kind = crate::engine::gradient::GradKind::Angle;
                    ed.popts.grad_editor = true;
                    ed.fill_opacity = 0.7;
                    ed.demo_drag(&[(sc.w * 0.5, sc.h * 0.5), (sc.w * 0.75, sc.h * 0.35)], none, win, cx);
                    // Move the end handle.
                    ed.demo_drag(&[(sc.w * 0.75, sc.h * 0.35), (sc.w * 0.8, sc.h * 0.6)], none, win, cx);
                });
                shot!("p3_live_gradient_handles");
                step!(move |ed: &mut Editor, _win, cx| ed.commit_gradient(cx));
                idle!();
                step!(move |ed: &mut Editor, win, cx| {
                    ed.popts.grad_editor = false;
                    ed.act_layer("new", win, cx);
                    ed.set_tool(Tool::Bucket, win, cx);
                    ed.popts.bucket_pattern = true;
                    ed.ropts.pattern = 0;
                    ed.opts.sample_all = true;
                    ed.fill_opacity = 1.0;
                    let (bx, by, _) = sc.ball();
                    ed.demo_click(bx, by, none, 1, win, cx);
                });
                idle!();
                shot!("p3_gradient_and_pattern_bucket");
            }
            if phases.iter().any(|p| p == "4") {
                step!(|ed: &mut Editor, _w, cx| {
                    let g = crate::engine::ops::demo_scene(w, h);
                    ed.replace_document(crate::engine::document::Document::from_grid(g, "Background"));
                    cx.notify();
                });
                wait(600).await;
                let (bx, by, br) = sc.ball();
                // Pen: a closed curved path around the ball (click-drag sets handles),
                // then an open Curvature Pen path across the sky.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::Pen, win, cx);
                    let r = br * 1.15;
                    let pts = [(bx, by - r, r * 0.55, 0.0), (bx + r, by, 0.0, r * 0.55), (bx, by + r, -r * 0.55, 0.0), (bx - r, by, 0.0, -r * 0.55)];
                    for (x, y, dx, dy) in pts {
                        ed.demo_press(x, y, none, 1, win, cx);
                        ed.demo_move(x + dx, y + dy, true, none, win, cx);
                        ed.demo_release(x + dx, y + dy, win, cx);
                    }
                    ed.demo_click(bx, by - r, none, 1, win, cx);
                    ed.set_tool(Tool::CurvaturePen, win, cx);
                    for k in 0..6 {
                        let x = sc.w * (0.1 + 0.15 * k as f32);
                        let y = sc.h * (0.12 + if k % 2 == 0 { 0.0 } else { 0.08 });
                        ed.demo_click(x, y, none, 1, win, cx);
                    }
                    ed.set_tool(Tool::DirectSelect, win, cx);
                    // Drag the right anchor of the ball path outward.
                    ed.demo_drag(&[(bx + r, by), (bx + r * 1.35, by - r * 0.1)], none, win, cx);
                    ed.vopts.sel_sub = Some(0);
                    let n: usize = ed.doc.paths.iter().map(|p| p.path.subs.iter().map(|s| s.anchors.len()).sum::<usize>()).sum();
                    eprintln!("paths {} anchors {n}", ed.doc.paths.len());
                });
                shot!("p4_pen_curvature_direct_select");
                // Paths panel: stroke the work path with a red brush, then make a selection.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.act_layer("new", win, cx);
                    *ed.brush_of(Tool::Brush) = BrushSettings { size: 24.0, hardness: 0.9, ..Default::default() };
                    ed.set_fg([220, 30, 40], win, cx);
                    ed.path_action("stroke", win, cx);
                    ed.path_action("selection", win, cx);
                    ed.set_tool(Tool::PathSelect, win, cx);
                });
                shot!("p4_stroke_path_and_selection");
                // Shapes: rounded stroked rectangle, star polygon, arrow line, custom heart, ellipse.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.act_select(1, cx);
                    ed.path_action("delete", win, cx);
                    ed.set_fg([40, 110, 220], win, cx);
                    ed.set_bg([250, 250, 250], win, cx);
                    ed.vopts.stroke = true;
                    ed.vopts.stroke_width = 18.0;
                    ed.vopts.radius = 80.0;
                    ed.set_tool(Tool::RectShape, win, cx);
                    ed.demo_drag(&[(sc.w * 0.05, sc.h * 0.05), (sc.w * 0.2, sc.h * 0.15), (sc.w * 0.3, sc.h * 0.25)], none, win, cx);
                    ed.vopts.stroke = false;
                    ed.set_fg([240, 190, 30], win, cx);
                    ed.vopts.sides = 5;
                    ed.vopts.star = 0.5;
                    ed.set_tool(Tool::PolygonShape, win, cx);
                    ed.demo_drag(&[(sc.w * 0.36, sc.h * 0.05), (sc.w * 0.5, sc.h * 0.3)], shift, win, cx);
                    ed.set_fg([20, 20, 20], win, cx);
                    ed.vopts.line_weight = 14.0;
                    ed.set_tool(Tool::LineShape, win, cx);
                    ed.demo_drag(&[(sc.w * 0.55, sc.h * 0.28), (sc.w * 0.75, sc.h * 0.08)], none, win, cx);
                    ed.set_fg([220, 40, 90], win, cx);
                    ed.vopts.custom = 0;
                    ed.set_tool(Tool::CustomShape, win, cx);
                    ed.demo_drag(&[(sc.w * 0.78, sc.h * 0.04), (sc.w * 0.95, sc.h * 0.26)], none, win, cx);
                    ed.set_fg([60, 170, 90], win, cx);
                    ed.set_tool(Tool::EllipseShape, win, cx);
                    ed.demo_drag(&[(sc.w * 0.6, sc.h * 0.35), (sc.w * 0.7, sc.h * 0.42)], none, win, cx);
                    // Live resize of the ellipse by its bottom-right handle.
                    let pts: Vec<(f32, f32)> = (0..=20).map(|k| (sc.w * (0.7 + 0.008 * k as f32), sc.h * (0.42 + 0.006 * k as f32))).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    eprintln!("shape resize {:.2} ms/frame", ed.last_tool_ms);
                });
                shot!("p4_shapes_live_resize");
                // Type: point text, paragraph text in a box, warped text.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::TypeH, win, cx);
                    ed.set_fg([255, 255, 255], win, cx);
                    ed.topts.style.size = 220.0;
                    ed.topts.style.weight = 700;
                    ed.demo_click(sc.w * 0.05, sc.h * 0.5, none, 1, win, cx);
                    ed.type_text("PixelForge");
                    ed.commit_text(cx);
                    ed.set_fg([20, 20, 30], win, cx);
                    ed.topts.style.size = 64.0;
                    ed.topts.style.weight = 400;
                    ed.demo_drag(&[(sc.w * 0.05, sc.h * 0.58), (sc.w * 0.42, sc.h * 0.95)], none, win, cx);
                    ed.type_text("Paragraph text wraps inside its box. ");
                    ed.type_text(&"The quick brown fox jumps over the lazy dog. ".repeat(6));
                    eprintln!("type {:.2} ms/keystroke", ed.last_tool_ms);
                });
                shot!("p4_type_editing_paragraph");
                step!(move |ed: &mut Editor, win, cx| {
                    ed.commit_text(cx);
                    ed.topts.style.size = 150.0;
                    ed.topts.style.weight = 700;
                    ed.topts.warp = crate::engine::text::Warp { style: crate::engine::text::WarpStyle::Arc, ..Default::default() };
                    ed.set_fg([250, 120, 20], win, cx);
                    ed.demo_click(sc.w * 0.5, sc.h * 0.62, none, 1, win, cx);
                    ed.type_text("Warped!");
                    ed.commit_text(cx);
                    ed.topts.warp = Default::default();
                    ed.set_tool(Tool::TypeMaskH, win, cx);
                    ed.topts.style.size = 260.0;
                    ed.demo_click(sc.w * 0.5, sc.h * 0.92, none, 1, win, cx);
                    ed.type_text("MASK");
                    ed.commit_text(cx);
                });
                shot!("p4_type_warp_and_mask_selection");
                // Frame: an empty frame shows the placeholder cross; placing fills it.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.act_select(1, cx);
                    ed.set_tool(Tool::Frame, win, cx);
                    ed.vopts.frame_ellipse = true;
                    ed.demo_drag(&[(sc.w * 0.76, sc.h * 0.55), (sc.w * 0.97, sc.h * 0.95)], none, win, cx);
                    ed.vopts.frame_ellipse = false;
                    ed.demo_drag(&[(sc.w * 0.52, sc.h * 0.7), (sc.w * 0.72, sc.h * 0.82)], none, win, cx);
                });
                shot!("p4_frames_empty");
                step!(move |ed: &mut Editor, win, cx| {
                    let img = crate::engine::ops::demo_scene(1600, 1200);
                    ed.add_smart_layer(crate::engine::layers::SmartObject::new("Placed scene", img, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
                    let fi = ed.doc.layers.iter().position(|l| l.name.starts_with("Frame")).unwrap();
                    ed.doc.active = fi;
                    let img = crate::engine::ops::demo_scene(800, 800);
                    ed.add_smart_layer(crate::engine::layers::SmartObject::new("Placed 2", img, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
                    ed.set_tool(Tool::Move, win, cx);
                });
                shot!("p4_frames_filled");
                // Content-Aware Tracing: hover the box edge, then click to add the path.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::ContentAwareTracing, win, cx);
                    ed.doc.active = 0;
                    let bxr = sc.boxr();
                    ed.demo_move(bxr.0 + 2.0, (bxr.1 + bxr.3) * 0.5, false, none, win, cx);
                    eprintln!("trace preview {} pts in {:.1} ms", ed.vopts.trace_preview.len(), ed.last_tool_ms);
                });
                shot!("p4_content_aware_tracing_hover");
                // Commit the trace; vertical type; Freeform Pen (shape mode);
                // Add Anchor on the traced path and Convert Point on that anchor.
                step!(move |ed: &mut Editor, win, cx| {
                    let bxr = sc.boxr();
                    ed.demo_click(bxr.0 + 2.0, (bxr.1 + bxr.3) * 0.5, none, 1, win, cx);
                    ed.set_tool(Tool::TypeV, win, cx);
                    ed.set_fg([255, 255, 255], win, cx);
                    ed.topts.style.size = 120.0;
                    ed.demo_click(sc.w * 0.97, sc.h * 0.05, none, 1, win, cx);
                    ed.type_text("縦書き TYPE");
                    ed.commit_text(cx);
                    ed.set_tool(Tool::FreeformPen, win, cx);
                    ed.vopts.pen_mode = super::vector_tools::PenMode::Shape;
                    ed.vopts.fill = true;
                    ed.vopts.stroke = false;
                    ed.set_fg([150, 60, 200], win, cx);
                    let pts: Vec<(f32, f32)> = (0..=80).map(|k| {
                        let t = k as f32 / 80.0 * std::f32::consts::TAU;
                        (sc.w * 0.12 + 260.0 * t.cos() * (1.0 + 0.25 * (5.0 * t).sin()), sc.h * 0.4 + 200.0 * t.sin() * (1.0 + 0.25 * (5.0 * t).sin()))
                    }).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    ed.vopts.pen_mode = super::vector_tools::PenMode::Path;
                    // Add an anchor midway along the traced path's top edge, then convert it to a corner.
                    ed.set_tool(Tool::AddAnchor, win, cx);
                    let n0: usize = ed.doc.paths.iter().map(|p| p.path.subs.iter().map(|s| s.anchors.len()).sum::<usize>()).sum();
                    let mid = ((bxr.0 + bxr.2) * 0.5, bxr.1);
                    ed.demo_click(mid.0, mid.1, none, 1, win, cx);
                    let n1: usize = ed.doc.paths.iter().map(|p| p.path.subs.iter().map(|s| s.anchors.len()).sum::<usize>()).sum();
                    ed.set_tool(Tool::DirectSelect, win, cx);
                    ed.demo_drag(&[mid, (mid.0, mid.1 - 250.0)], none, win, cx);
                    eprintln!("anchors {n0} -> {n1} after Add Anchor");
                });
                shot!("p4_vertical_type_freeform_add_anchor");
            }
            if phases.iter().any(|p| p == "5") {
                step!(|ed: &mut Editor, _w, cx| {
                    let g = crate::engine::ops::demo_scene(w, h);
                    ed.replace_document(crate::engine::document::Document::from_grid(g, "Background"));
                    cx.notify();
                });
                wait(600).await;
                let (bx, by, br) = sc.ball();
                let bxr = sc.boxr();
                // Rotate View: 25° with a selection and a sampler visible (overlays rotate too).
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::EllipseSelect, win, cx);
                    ed.demo_drag(&[(bx - br, by - br), (bx + br, by + br)], none, win, cx);
                    ed.set_tool(Tool::RotateView, win, cx);
                    let c = ed.vp.to_image(ed.canvas.center(), ed.canvas, ed.sf);
                    let a = 25f32.to_radians();
                    ed.demo_press(c.0 + 600.0, c.1, none, 1, win, cx);
                    ed.demo_move(c.0 + 600.0 * a.cos(), c.1 + 600.0 * a.sin(), true, none, win, cx);
                    ed.demo_release(c.0 + 600.0 * a.cos(), c.1 + 600.0 * a.sin(), win, cx);
                    eprintln!("rotate view {:.1}°", ed.vp.rot.to_degrees());
                });
                idle!();
                shot!("p5_rotate_view_25");
                step!(move |ed: &mut Editor, _win, cx| {
                    let st = ed.cache.last;
                    eprintln!("rotated frame: {} visible tiles, {:.1} ms", st.visible, st.compose_ms);
                    ed.vp.rot = 0.0;
                    ed.act_select(1, cx);
                });
                // Scrubby zoom: drag right from the ball to zoom in around it.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::Zoom, win, cx);
                    let z0 = ed.vp.zoom;
                    let p0 = ed.vp.to_screen(bx, by, ed.canvas, ed.sf);
                    ed.on_mouse_down(&MouseDownEvent { button: MouseButton::Left, position: p0, modifiers: none, click_count: 1, first_mouse: false }, win, cx);
                    for k in 1..=12 {
                        let p = p0 + gpui_kit::point(gpui_kit::px(k as f32 * 10.0), gpui_kit::px(0.));
                        ed.on_mouse_move(&MouseMoveEvent { position: p, pressed_button: Some(MouseButton::Left), modifiers: none }, win, cx);
                    }
                    ed.on_mouse_up(&MouseUpEvent { button: MouseButton::Left, position: p0, modifiers: none, click_count: 1 }, win, cx);
                    eprintln!("scrubby zoom {:.1}% -> {:.1}%", z0 * 100.0, ed.vp.zoom * 100.0);
                });
                idle!();
                shot!("p5_scrubby_zoom");
                // Color samplers, eyedropper 11×11 and a ruler; Info panel shows them.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.act_zoom(2, cx);
                    ed.set_tool(Tool::ColorSampler, win, cx);
                    ed.opts.sample_size = super::options::SampleSize::Avg11;
                    for (x, y) in [(bx, by), ((bxr.0 + bxr.2) * 0.5, (bxr.1 + bxr.3) * 0.5), (sc.w * 0.83, sc.h * 0.17), (sc.w * 0.5, sc.h * 0.9)] {
                        ed.demo_click(x, y, none, 1, win, cx);
                    }
                    ed.set_tool(Tool::Eyedropper, win, cx);
                    ed.demo_click(sc.w * 0.83, sc.h * 0.17, none, 1, win, cx);
                    ed.set_tool(Tool::Ruler, win, cx);
                    ed.demo_drag(&[(bxr.0, bxr.3), (bxr.2, bxr.1)], none, win, cx);
                });
                shot!("p5_samplers_eyedropper_ruler_info");
                // Straighten: tilt the image 6°, measure the box's top edge, straighten.
                step!(move |ed: &mut Editor, _win, cx| {
                    ed.doc.ann.samplers.clear();
                    let m = crate::engine::annot::rotation_about(-6.0, sc.w * 0.5, sc.h * 0.5);
                    ed.canvas_xform("Rotate", m, w, h, cx);
                });
                idle!();
                step!(move |ed: &mut Editor, win, cx| {
                    let m = crate::engine::annot::rotation_about(-6.0, sc.w * 0.5, sc.h * 0.5);
                    let f = |x: f32, y: f32| (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]);
                    ed.set_tool(Tool::Ruler, win, cx);
                    let (a, b) = (f(bxr.0 + 10.0, bxr.1), f(bxr.2 - 10.0, bxr.1));
                    ed.demo_drag(&[a, b], none, win, cx);
                    eprintln!("ruler angle {:.2}°", crate::engine::annot::measure(a, b).2);
                });
                shot!("p5_straighten_before");
                step!(move |ed: &mut Editor, _win, cx| ed.straighten(false, cx));
                idle!();
                wait(400).await;
                step!(move |ed: &mut Editor, _win, cx| {
                    eprintln!("straightened canvas {} × {}", ed.doc.w, ed.doc.h);
                    cx.notify();
                });
                shot!("p5_straighten_after");
                // Notes and Count.
                step!(move |ed: &mut Editor, _w, cx| {
                    let g = crate::engine::ops::demo_scene(w, h);
                    ed.replace_document(crate::engine::document::Document::from_grid(g, "Background"));
                    cx.notify();
                });
                wait(600).await;
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::Note, win, cx);
                    ed.demo_click(sc.w * 0.12, sc.h * 0.3, none, 1, win, cx);
                    ed.set_note_text(0, "Remove the dust spots on the grass", win, cx);
                    ed.demo_click(bxr.2 + 40.0, bxr.1, none, 1, win, cx);
                    ed.set_note_text(1, "Box colour approved", win, cx);
                    ed.set_tool(Tool::Count, win, cx);
                    let dots = crate::engine::ops::demo_blemishes(w, h);
                    for &(x, y, _) in dots.iter().take(6) {
                        ed.demo_click(x, y, none, 1, win, cx);
                    }
                    ed.doc.ann.counts.push(crate::engine::annot::CountGroup::new("Count Group 2", [40, 140, 240]));
                    ed.nopts.count_group = 1;
                    for &(x, y) in &[(bx, by), (sc.w * 0.83, sc.h * 0.17)] {
                        ed.demo_click(x, y, none, 1, win, cx);
                    }
                    eprintln!("count total {}", ed.doc.ann.count_total());
                });
                shot!("p5_notes_and_count");
                // Crop: 16:9 frame, rotated by dragging outside it.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.doc.ann = Default::default();
                    ed.set_tool(Tool::Crop, win, cx);
                    ed.nopts.crop_ratio = super::nav_tools::CropRatio::R16x9;
                    ed.demo_drag(&[(sc.w * 0.15, sc.h * 0.2), (sc.w * 0.85, sc.h * 0.6)], none, win, cx);
                    let r = ed.crop.unwrap();
                    let c = ((r.x0 + r.x1) as f32 * 0.5, (r.y0 + r.y1) as f32 * 0.5);
                    // Rotate: start right of the frame, drag 8° around its centre.
                    let rad = (r.w() as f32) * 0.62;
                    let a = 8f32.to_radians();
                    ed.demo_drag(&[(c.0 + rad, c.1), (c.0 + rad * a.cos(), c.1 + rad * a.sin())], none, win, cx);
                    // Pull the bottom-right handle out a bit.
                    eprintln!("crop {:?} angle {:.1}", ed.crop, ed.nopts.crop_angle);
                });
                shot!("p5_crop_rotated_16x9");
                step!(move |ed: &mut Editor, _win, cx| ed.commit_crop(cx));
                idle!();
                wait(400).await;
                step!(move |ed: &mut Editor, _win, cx| {
                    eprintln!("cropped canvas {} × {}", ed.doc.w, ed.doc.h);
                    cx.notify();
                });
                shot!("p5_crop_done");
                // Perspective crop: the box face as a skewed quad.
                step!(move |ed: &mut Editor, _w, cx| {
                    let g = crate::engine::ops::demo_scene(w, h);
                    ed.replace_document(crate::engine::document::Document::from_grid(g, "Background"));
                    cx.notify();
                });
                wait(600).await;
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::PerspectiveCrop, win, cx);
                    let q = [(bxr.0 - 150.0, bxr.1 + 60.0), (bxr.2 + 100.0, bxr.1 - 40.0), (bxr.2 + 30.0, bxr.3 + 80.0), (bxr.0 - 60.0, bxr.3 - 20.0)];
                    for (x, y) in q {
                        ed.demo_click(x, y, none, 1, win, cx);
                    }
                });
                shot!("p5_perspective_crop_quad");
                step!(move |ed: &mut Editor, _win, cx| ed.commit_perspective_crop(cx));
                idle!();
                wait(400).await;
                step!(move |ed: &mut Editor, _win, cx| {
                    eprintln!("perspective cropped canvas {} × {}", ed.doc.w, ed.doc.h);
                    cx.notify();
                });
                shot!("p5_perspective_crop_done");
                // Slices and artboards, exported.
                step!(move |ed: &mut Editor, _w, cx| {
                    let g = crate::engine::ops::demo_scene(w, h);
                    ed.replace_document(crate::engine::document::Document::from_grid(g, "Background"));
                    cx.notify();
                });
                wait(600).await;
                step!(move |ed: &mut Editor, win, cx| {
                    ed.set_tool(Tool::Slice, win, cx);
                    ed.demo_drag(&[(bx - br, by - br), (bx + br, by + br)], none, win, cx);
                    ed.demo_drag(&[(bxr.0, bxr.1), (bxr.2, bxr.3)], none, win, cx);
                    ed.set_tool(Tool::SliceSelect, win, cx);
                    ed.demo_click(bx, by, none, 1, win, cx);
                    let dir = std::env::temp_dir().join("pf_slices");
                    let _ = std::fs::create_dir_all(&dir);
                    match ed.export_regions_now(true, &dir) {
                        Ok(n) => eprintln!("exported {n} slices to {}", dir.display()),
                        Err(e) => eprintln!("slice export failed: {e:#}"),
                    }
                });
                shot!("p5_slices");
                step!(move |ed: &mut Editor, win, cx| {
                    ed.doc.ann.slices.clear();
                    ed.set_tool(Tool::Artboard, win, cx);
                    ed.demo_drag(&[(sc.w * 0.05, sc.h * 0.08), (sc.w * 0.45, sc.h * 0.92)], none, win, cx);
                    ed.demo_drag(&[(sc.w * 0.52, sc.h * 0.3), (sc.w * 0.95, sc.h * 0.7)], none, win, cx);
                    let dir = std::env::temp_dir().join("pf_artboards");
                    let _ = std::fs::create_dir_all(&dir);
                    match ed.export_regions_now(false, &dir) {
                        Ok(n) => eprintln!("exported {n} artboards to {}", dir.display()),
                        Err(e) => eprintln!("artboard export failed: {e:#}"),
                    }
                });
                shot!("p5_artboards");
                // Move tool: auto-select a shape, transform handles, align, float a selection.
                step!(move |ed: &mut Editor, win, cx| {
                    ed.doc.ann = Default::default();
                    ed.set_fg([240, 190, 30], win, cx);
                    ed.vopts.sides = 5;
                    ed.vopts.star = 0.5;
                    ed.set_tool(Tool::PolygonShape, win, cx);
                    ed.demo_drag(&[(sc.w * 0.6, sc.h * 0.05), (sc.w * 0.75, sc.h * 0.25)], shift, win, cx);
                    ed.doc.active = 0;
                    ed.set_tool(Tool::Move, win, cx);
                    ed.opts.auto_select = true;
                    ed.opts.show_transform = true;
                    // Auto-select picks the star; drag the bottom-right handle out.
                    ed.demo_click(sc.w * 0.675, sc.h * 0.15, none, 1, win, cx);
                    let r = crate::engine::xform::content_rect(&ed.doc.layers[ed.doc.active].grid).unwrap();
                    ed.demo_drag(&[(r.x1 as f32, r.y1 as f32), (r.x1 as f32 + 300.0, r.y1 as f32 + 300.0)], none, win, cx);
                    eprintln!("auto-selected layer {} ({})", ed.doc.active, ed.doc.layers[ed.doc.active].name);
                });
                idle!();
                step!(move |ed: &mut Editor, _win, cx| {
                    ed.align_layer("hcenter", cx);
                    cx.notify();
                });
                shot!("p5_move_autoselect_transform_align");
                step!(move |ed: &mut Editor, win, cx| {
                    ed.doc.active = 0;
                    ed.opts.auto_select = false;
                    ed.opts.show_transform = false;
                    ed.set_tool(Tool::EllipseSelect, win, cx);
                    ed.demo_drag(&[(bx - br, by - br), (bx + br, by + br)], none, win, cx);
                    ed.set_tool(Tool::Move, win, cx);
                    ed.demo_press(bx, by, none, 1, win, cx);
                    ed.demo_move(bx + 900.0, by - 500.0, true, none, win, cx);
                });
                shot!("p5_move_float_dragging");
                step!(move |ed: &mut Editor, win, cx| ed.demo_release(bx + 900.0, by - 500.0, win, cx));
                idle!();
                shot!("p5_move_float_dropped");
            }
            super::demo_gpu::run(&this, cx, &phases).await;
            super::demo_gaps::run(&this, cx, &phases).await;
            eprintln!("DONE");
            wait(2500).await;
            let _ = this.update(cx, |_, cx| cx.quit());
        })
        .detach();
    }
}
