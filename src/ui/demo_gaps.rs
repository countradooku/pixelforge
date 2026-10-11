//! `PF_DEMO=tools PF_DEMO_PHASE=A,…`: screenshot tour of the gap-closing
//! features (phases A–H). Like the toolbar tour it drives the real input
//! handlers and prints `SHOT name` lines for `tools/demo_shots.sh`.

use gpui_kit::{AsyncWindowContext, Context, KeyDownEvent, Keystroke, WeakEntity};

use crate::engine::document::Document;
use crate::engine::text::TextStyle;

use super::editor::{Editor, Tool};

/// Opens a test photo (or a synthetic stand-in when it is missing).
fn photo_doc(name: &str) -> Document {
    match crate::engine::io::open_image(std::path::Path::new(&format!("testdata/photos/{name}.jpg"))) {
        Ok(g) => {
            let mut d = Document::from_grid(g, "Background");
            d.path = Some(format!("{name}.jpg").into());
            d
        }
        Err(_) => Document::from_grid(crate::engine::ops::demo_scene(3000, 2000), "Background"),
    }
}

/// A white canvas.
fn blank(w: u32, h: u32) -> Document {
    Document::from_grid(crate::engine::grid::TileGrid::filled(w, h, crate::engine::px::pack(255, 255, 255, 255)), "Background")
}

/// Sends a key to the text editor (e.g. "shift-right", "cmd-a").
fn key(ed: &mut Editor, k: &str, cx: &mut Context<Editor>) {
    let ev = KeyDownEvent { keystroke: Keystroke::parse(k).unwrap(), is_held: false, prefer_character_input: false };
    ed.text_key(&ev, cx);
}

/// Image point of the caret before byte `i` of the edited text.
fn at_byte(ed: &Editor, i: usize) -> (f32, f32) {
    let t = ed.edited_text().unwrap().1;
    let (a, b) = t.caret_seg(i);
    let m = a.lerp(b, 0.5);
    (m.x, m.y)
}

fn find(ed: &Editor, s: &str, nth: usize) -> usize {
    let t = ed.edited_text().unwrap().1;
    t.text.match_indices(s).nth(nth).map(|(i, _)| i).unwrap()
}

/// Selects `s` (its `nth` occurrence) as a user would: click before it,
/// then Shift + → once per character.
fn select(ed: &mut Editor, s: &str, nth: usize, win: &mut gpui_kit::Window, cx: &mut Context<Editor>) {
    let i = find(ed, s, nth);
    let (x, y) = at_byte(ed, i);
    ed.demo_click(x + 0.5, y, gpui_kit::Modifiers::default(), 1, win, cx);
    for _ in s.chars() {
        key(ed, "shift-right", cx);
    }
}

pub async fn run(this: &WeakEntity<Editor>, cx: &mut AsyncWindowContext, phases: &[String]) {
    let has = |p: &str| phases.iter().any(|q| q == p);
    let exec = cx.background_executor().clone();
    let wait = move |ms: u64| exec.timer(std::time::Duration::from_millis(ms));
    // Every step repaints: state changes made outside input handlers don't
    // notify by themselves.
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
    // `PF_SHOT_DOC=dir` also exports each shot's document composite (a
    // check that works while the screen is locked and windows don't repaint).
    macro_rules! shot {
        ($name:expr) => {
            wait(700).await;
            if let Ok(dir) = std::env::var("PF_SHOT_DOC") {
                let _ = this.update(cx, |ed: &mut Editor, _| {
                    let p = std::path::Path::new(&dir).join(format!("{}.png", $name));
                    let _ = crate::engine::io::export(&ed.doc.layers, ed.doc.w, ed.doc.h, &p);
                });
            }
            eprintln!("SHOT {}", $name);
            wait(1300).await;
        };
    }
    let none = gpui_kit::Modifiers::default();
    let alt = gpui_kit::Modifiers { alt: true, ..Default::default() };

    if has("A") {
        // A1: hover object finder and click-to-select on the dog photo.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("dog_grass"));
            ed.set_tool(Tool::ObjectSelect, win, cx);
        });
        until!(|ed: &mut Editor, _| ed.finder.is_some() && ed.finder_building.is_none());
        step!(|ed: &mut Editor, win, cx| {
            ed.demo_move(2000.0, 1300.0, false, none, win, cx);
            let f = ed.finder.as_ref().map(|f| f.1.objects.len()).unwrap_or(0);
            eprintln!("object finder: {f} objects, hover {:?}, lookup {:.3} ms", ed.hover_object, ed.last_tool_ms);
        });
        shot!("pA_object_finder_hover");
        step!(|ed: &mut Editor, win, cx| ed.demo_click(2000.0, 1300.0, none, 1, win, cx));
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        step!(|ed: &mut Editor, win, cx| ed.demo_move(3500.0, 300.0, false, none, win, cx));
        shot!("pA_object_finder_click_selects");

        // A2: Select Subject (classical prior + GrabCut) on the portrait.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("portrait_hair"));
            ed.set_tool(Tool::QuickSelect, win, cx);
            ed.select_subject(cx);
        });
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        shot!("pA_select_subject_classical");

        // A3: Content-Aware Fill workspace: remove the daisies from the lawn.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("dog_grass"));
            ed.set_tool(Tool::EllipseSelect, win, cx);
            ed.demo_drag(&[(1480.0, 340.0), (1960.0, 520.0)], none, win, cx);
            ed.open_content_aware_fill(cx);
        });
        until!(|ed: &mut Editor, _| ed.caf.as_ref().is_none_or(|c| c.computing.is_none()));
        shot!("pA_caf_workspace_auto");
        step!(|ed: &mut Editor, win, cx| {
            if let Some(c) = &mut ed.caf {
                c.adapt.rotation = crate::engine::patchmatch::RotationAdapt::Medium;
                c.adapt.mirror = true;
                c.output = super::caf::CafOutput::NewLayer;
                c.brush = 260.0;
            }
            // Keep the dog out of the sampling area (Alt-drag subtracts).
            ed.demo_drag(&[(1500.0, 700.0), (1800.0, 820.0), (2100.0, 760.0)], alt, win, cx);
        });
        until!(|ed: &mut Editor, _| ed.caf.as_ref().is_none_or(|c| c.computing.is_none()));
        shot!("pA_caf_custom_sampling_rotation_mirror");
        step!(|ed: &mut Editor, _win, cx| ed.caf_apply(cx));
        shot!("pA_caf_applied_new_layer");

        // A4: Spot Healing ▸ Create Texture over the same daisies.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("dog_grass"));
            ed.set_tool(Tool::SpotHeal, win, cx);
            ed.ropts.spot_mode = crate::engine::retouch::SpotMode::CreateTexture;
            ed.brush_of(Tool::SpotHeal).size = 140.0;
            ed.demo_drag(&[(1520.0, 420.0), (1700.0, 440.0), (1900.0, 430.0)], none, win, cx);
        });
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        shot!("pA_spot_heal_create_texture");

        // A5: Color Replacement Limits: one dab each, Discontiguous (left)
        // vs Contiguous (middle) vs Find Edges (right), on red bands.
        step!(|ed: &mut Editor, win, cx| {
            use crate::engine::paint::{Limits, Sampling};
            let (w, h) = (2400u32, 1200u32);
            let g = crate::engine::grid::build_grid(w, h, |tx, ty| {
                let ts = crate::engine::grid::TS;
                let mut t = vec![0u32; ts * ts].into_boxed_slice();
                for y in 0..ts {
                    for x in 0..ts {
                        let gx = tx as usize * ts + x;
                        let gy = ty as usize * ts + y;
                        // Red bands separated by thin white gaps; the right
                        // third has a faint step inside a red band.
                        let band = (gy / 90) % 2 == 0 || (gy % 90) > 12;
                        let red = if gx > 1600 && gy % 180 > 90 { 165 } else { 200 };
                        t[y * ts + x] = if band { crate::engine::px::pack(red, 30, 30, 255) } else { crate::engine::px::pack(250, 250, 250, 255) };
                    }
                }
                crate::engine::grid::finish_tile(t)
            });
            ed.replace_document(Document::from_grid(g, "Bands"));
            ed.set_tool(Tool::ColorReplace, win, cx);
            ed.set_fg([40, 90, 230], win, cx);
            ed.popts.sampling = Sampling::Once;
            ed.popts.paint_tolerance = 40.0;
            ed.brush_of(Tool::ColorReplace).size = 500.0;
            for (x, lim) in [(420.0, Limits::Discontiguous), (1200.0, Limits::Contiguous), (1980.0, Limits::FindEdges)] {
                ed.popts.limits = lim;
                // y = 600 is inside a red band (the hotspot samples it).
                ed.demo_click(x, 600.0, none, 1, win, cx);
            }
        });
        shot!("pA_color_replacement_limits");

        // A6: Healing Brush, Quality: Exact.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("portrait_hair"));
            ed.set_tool(Tool::Heal, win, cx);
            ed.ropts.heal_quality = crate::engine::poisson::Quality::Exact;
            ed.brush_of(Tool::Heal).size = 160.0;
            ed.demo_click(300.0, 300.0, alt, 1, win, cx);
            ed.demo_drag(&[(700.0, 900.0), (900.0, 1000.0), (1100.0, 950.0)], none, win, cx);
            eprintln!("exact heal: last dab {:.1} ms", ed.last_tool_ms);
        });
        shot!("pA_heal_quality_exact");
    }
    if has("B") {
        use crate::engine::text::{Caps, LastLine, Position, TextAlign, Warp, WarpStyle};
        use super::type_tools::TypePanel;
        // B1: styled runs, edited with on-canvas range selections.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(blank(2600, 1100));
            ed.set_tool(Tool::TypeH, win, cx);
            ed.set_fg([30, 30, 40], win, cx);
            ed.topts.style = TextStyle { size: 104.0, ..Default::default() };
            ed.topts.panel = TypePanel::Character;
            ed.demo_click(120.0, 330.0, none, 1, win, cx);
            ed.type_text("Styled runs with Bold, Red, small caps,\nx2 + H2O, tracked and faux bold.");
            let t0 = std::time::Instant::now();
            select(ed, "runs", 0, win, cx);
            ed.apply_char(|s| {
                s.size = 150.0;
                s.color = [20, 150, 70];
            });
            select(ed, "Bold", 0, win, cx);
            ed.apply_char(|s| s.weight = 700);
            select(ed, "with", 0, win, cx);
            ed.apply_char(|s| {
                s.faux_italic = true;
                s.strike = true;
            });
            select(ed, "small caps", 0, win, cx);
            ed.apply_char(|s| {
                s.caps = Caps::Small;
                s.color = [40, 80, 220];
            });
            select(ed, "2", 0, win, cx);
            ed.apply_char(|s| s.position = Position::Super);
            select(ed, "2", 1, win, cx);
            ed.apply_char(|s| s.position = Position::Sub);
            select(ed, "tracked", 0, win, cx);
            ed.apply_char(|s| s.tracking = 300.0);
            select(ed, "faux bold", 0, win, cx);
            ed.apply_char(|s| {
                s.faux_bold = true;
                s.baseline_shift = 20.0;
            });
            select(ed, "Red", 0, win, cx);
            ed.apply_char(|s| {
                s.color = [220, 30, 30];
                s.underline = true;
            });
            // Double-click selects the word "Red" again for the shot.
            let i = find(ed, "Red", 0);
            let (x, y) = at_byte(ed, i + 1);
            ed.demo_click(x, y, none, 2, win, cx);
            let t = ed.edited_text().unwrap().1;
            eprintln!("styled runs: {} runs, {} glyphs, 9 range edits in {:.1} ms", t.runs.len(), t.layout.glyphs.len(), t0.elapsed().as_secs_f32() * 1000.0);
        });
        shot!("pB_styled_runs_selection");

        // B2: paragraph styles: justify (last line centred), first-line
        // indent, space after, hyphenation; Justify All with a left indent.
        step!(|ed: &mut Editor, win, cx| {
            ed.commit_text(cx);
            ed.replace_document(blank(2400, 1500));
            ed.set_fg([25, 25, 30], win, cx);
            ed.topts.style = TextStyle { size: 40.0, ..Default::default() };
            ed.topts.panel = TypePanel::Paragraph;
            ed.demo_drag(&[(120.0, 120.0), (1180.0, 1380.0)], none, win, cx);
            ed.type_text("Paragraph text is set by PixelForge's own line breaker: justified lines stretch their word spaces to the box edges, the last line of each paragraph follows its own alignment, and overlong words such as internationalization or counterrevolutionaries are hyphenated at the line end.\nThe second paragraph starts after the first one's space after, with a first-line indent, exactly as in Photoshop's Paragraph panel.");
            key(ed, "cmd-a", cx);
            ed.apply_para(|p| {
                p.align = TextAlign::Justify;
                p.last = LastLine::Center;
                p.indent_first = 60.0;
                p.space_after = 40.0;
                p.hyphenate = true;
            });
            ed.commit_text(cx);
            // A caption (point text, left-aligned).
            ed.topts.style = TextStyle { size: 26.0, ..Default::default() };
            ed.doc.active = 0;
            ed.demo_click(130.0, 1440.0, none, 1, win, cx);
            ed.type_text("Left: Every-line Composer with TeX hyphenation patterns (en-US).");
            ed.commit_text(cx);
            ed.topts.style = TextStyle { size: 40.0, ..Default::default() };
            ed.doc.active = 0;
            ed.demo_drag(&[(1300.0, 120.0), (2280.0, 900.0)], none, win, cx);
            ed.type_text("Justify All with a 40 px left indent sets every line flush, the last one too. Hyphenation off.");
            key(ed, "cmd-a", cx);
            ed.apply_para(|p| {
                p.align = TextAlign::Justify;
                p.last = LastLine::All;
                p.indent_left = 40.0;
            });
            ed.apply_char(|s| s.size = 52.0);
        });
        shot!("pB_paragraph_justify_hyphenate");

        // B3: OpenType features (Baskerville: dlig + onum; Seravek: liga, frac).
        step!(|ed: &mut Editor, win, cx| {
            ed.commit_text(cx);
            ed.replace_document(blank(2600, 1300));
            ed.set_fg([20, 20, 20], win, cx);
            ed.topts.panel = TypePanel::OpenType;
            let lines: [(&str, &str, f32, fn(&mut crate::engine::text::OtFeatures)); 5] = [
                ("Baskerville", "Quick fjords sty ct 0123456789", 200.0, |_| {}),
                ("Baskerville", "Quick fjords sty ct 0123456789", 400.0, |f| {
                    f.dlig = true;
                    f.onum = true;
                }),
                ("Seravek", "office affluent 1/2 3/4", 650.0, |_| {}),
                ("Seravek", "office affluent 1/2 3/4", 850.0, |f| f.liga = false),
                ("Seravek", "office affluent 1/2 3/4", 1050.0, |f| f.frac = true),
            ];
            for (fam, txt, y, f) in lines {
                ed.topts.style = TextStyle { family: fam.into(), size: 110.0, ..Default::default() };
                ed.demo_click(120.0, y, none, 1, win, cx);
                ed.type_text(txt);
                key(ed, "cmd-a", cx);
                ed.apply_char(|s| f(&mut s.features));
                ed.commit_text(cx);
            }
            ed.topts.style.size = 44.0;
            ed.topts.style.family = "Helvetica".into();
            ed.set_fg([120, 120, 130], win, cx);
            for (y, note) in [(240.0, "default"), (440.0, "dlig + onum"), (690.0, "default (liga)"), (890.0, "liga off"), (1090.0, "frac")] {
                ed.demo_click(2150.0, y, none, 1, win, cx);
                ed.type_text(note);
                ed.commit_text(cx);
            }
            // Re-enter the last sample so the panel shows its features.
            ed.set_tool(Tool::TypeH, win, cx);
            ed.doc.active = 5;
            ed.demo_click(300.0, 1020.0, none, 1, win, cx);
        });
        shot!("pB_opentype_features");

        // B4: all 15 warp styles (bend 50 %), one text layer each.
        step!(|ed: &mut Editor, win, cx| {
            ed.commit_text(cx);
            ed.replace_document(blank(3300, 2200));
            ed.topts.style = TextStyle { size: 80.0, weight: 700, ..Default::default() };
            let hues = [[220, 40, 40], [230, 120, 20], [200, 170, 0], [60, 160, 60], [20, 150, 160], [30, 100, 220], [110, 60, 210], [190, 50, 160]];
            for (k, style) in WarpStyle::ALL[1..].iter().enumerate() {
                let (cx_, cy) = (360.0 + (k % 5) as f32 * 650.0, 360.0 + (k / 5) as f32 * 720.0);
                ed.set_fg(hues[k % hues.len()], win, cx);
                ed.topts.warp = Warp { style: *style, ..Default::default() };
                ed.demo_click(cx_ - 230.0, cy, none, 1, win, cx);
                ed.type_text(&format!("{}\n{}", style.label().to_uppercase(), "WARP TEXT"));
                key(ed, "cmd-a", cx);
                ed.apply_para(|p| p.align = TextAlign::Left);
                ed.commit_text(cx);
                // Deselect it so the next click starts a new layer.
                ed.doc.active = 0;
            }
            ed.topts.warp = Warp::default();
            ed.set_tool(Tool::Move, win, cx);
        });
        shot!("pB_warp_all_15_styles");

        // B5: Flag with horizontal and vertical distortion, Warp panel open.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(blank(2400, 1400));
            ed.set_tool(Tool::TypeH, win, cx);
            ed.set_fg([30, 90, 200], win, cx);
            ed.topts.style = TextStyle { size: 230.0, weight: 700, ..Default::default() };
            ed.topts.panel = TypePanel::Warp;
            ed.topts.warp = Warp { style: WarpStyle::Flag, bend: 0.5, h_distort: 0.4, v_distort: -0.2, vertical: false };
            ed.demo_click(250.0, 650.0, none, 1, win, cx);
            ed.type_text("DISTORTED\nFLAG WARP");
            ed.topts.warp = Warp::default();
        });
        shot!("pB_warp_flag_distortion_panel");

        // B6: type on a path (and flipped on an inner circle), area type in a star.
        step!(|ed: &mut Editor, win, cx| {
            use crate::engine::document::PathItem;
            use crate::engine::vector::Path;
            ed.commit_text(cx);
            ed.replace_document(blank(2700, 1600));
            ed.set_tool(Tool::TypeH, win, cx);
            ed.topts.panel = TypePanel::Options;
            let paths = [Path::ellipse(780.0, 800.0, 560.0, 560.0), Path::ellipse(780.0, 800.0, 380.0, 380.0), Path::star(2000.0, 800.0, 620.0, 620.0, 5, 0.5, 0.0)];
            for (k, p) in paths.iter().enumerate() {
                ed.doc.paths.push(PathItem { name: format!("Path {}", k + 1), path: p.clone() });
            }
            ed.doc.active_path = Some(0);
            ed.set_fg([200, 40, 60], win, cx);
            ed.topts.style = TextStyle { size: 72.0, weight: 700, ..Default::default() };
            ed.demo_click(780.0, 240.0, none, 1, win, cx);
            ed.type_text("Type on a path follows every curve of the circle");
            ed.commit_text(cx);
            ed.doc.active_path = Some(1);
            ed.set_fg([30, 110, 200], win, cx);
            ed.topts.style.size = 60.0;
            ed.demo_click(780.0, 1180.0, none, 1, win, cx);
            ed.type_text("Flipped: the text runs inside");
            ed.apply_layer(|t| {
                if let Some(p) = &mut t.path {
                    std::sync::Arc::make_mut(p).flip = true;
                }
            });
            ed.commit_text(cx);
            ed.doc.active_path = Some(2);
            ed.set_fg([30, 30, 30], win, cx);
            ed.topts.style = TextStyle { size: 30.0, ..Default::default() };
            ed.demo_click(2000.0, 820.0, none, 1, win, cx);
            ed.type_text(&"Area type flows lines inside a closed path, here a five-pointed star. ".repeat(9));
            key(ed, "cmd-a", cx);
            ed.apply_para(|p| {
                p.align = TextAlign::Justify;
                p.hyphenate = true;
            });
            let t = ed.edited_text().unwrap().1;
            eprintln!("area type: {} lines, overflow {}", t.layout.lines.len(), t.layout.overflow);
        });
        shot!("pB_type_on_path_and_area");
        // Drag the outer path text's start marker a quarter turn along the circle.
        step!(|ed: &mut Editor, win, cx| {
            ed.commit_text(cx);
            ed.doc.active = 1;
            let crate::engine::layers::LayerKind::Text(t) = &ed.doc.layers[1].kind else { return };
            let start = t.path.as_ref().unwrap().start;
            let pts: Vec<(f32, f32)> = (0..=12).filter_map(|k| t.path_at(start + 0.02 * k as f32)).map(|(q, _)| (q.x, q.y)).collect();
            ed.demo_drag(&pts, none, win, cx);
            let crate::engine::layers::LayerKind::Text(t) = &ed.doc.layers[1].kind else { return };
            eprintln!("path text start {:.3} → {:.3}", start, t.path.as_ref().unwrap().start);
        });
        shot!("pB_path_text_start_marker_dragged");

        // B7: vertical type: rotated Roman with a tate-chu-yoko date, and
        // Standard Vertical Roman Alignment.
        step!(|ed: &mut Editor, win, cx| {
            ed.commit_text(cx);
            ed.replace_document(blank(1800, 2000));
            ed.set_tool(Tool::TypeV, win, cx);
            ed.topts.panel = TypePanel::Options;
            ed.set_fg([20, 20, 30], win, cx);
            ed.topts.style = TextStyle { size: 110.0, ..Default::default() };
            ed.demo_click(1450.0, 150.0, none, 1, win, cx);
            ed.type_text("縦書きの Roman 2026年12月");
            let i = find(ed, "12", 0);
            ed.apply_layer(|t| t.tcy = vec![(i, i + 2)]);
            ed.commit_text(cx);
            ed.set_fg([200, 50, 40], win, cx);
            ed.demo_click(950.0, 150.0, none, 1, win, cx);
            ed.type_text("縦 UPRIGHT");
            ed.apply_layer(|t| t.upright_roman = true);
            ed.commit_text(cx);
            ed.set_fg([40, 90, 200], win, cx);
            ed.topts.style.size = 80.0;
            ed.demo_drag(&[(80.0, 150.0), (620.0, 1500.0)], none, win, cx);
            ed.type_text("縦組みの段落テキストは右から左へ列を折り返します。Latin words rotate.");
        });
        shot!("pB_vertical_roman_tcy");
    }

    if has("C") {
        use crate::engine::layers::{GradPaint, LayerKind, Paint, PatPaint};
        use crate::engine::vector::{Align, StrokeStyle};
        let shift = gpui_kit::Modifiers { shift: true, ..Default::default() };
        // C1: live path operations: a rounded rectangle, ⌥-drag subtracts an
        // ellipse, ⇧-drag adds a star; the components stay editable.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(blank(2600, 1600));
            ed.set_fg([40, 110, 220], win, cx);
            ed.set_tool(Tool::RectShape, win, cx);
            ed.vopts.radius = 60.0;
            ed.demo_drag(&[(300.0, 300.0), (900.0, 700.0), (1500.0, 1200.0)], none, win, cx);
            ed.set_tool(Tool::EllipseShape, win, cx);
            // ⌥ draws from the centre and subtracts.
            ed.demo_drag(&[(1250.0, 900.0), (1350.0, 1000.0), (1500.0, 1120.0)], alt, win, cx);
            ed.set_tool(Tool::PolygonShape, win, cx);
            ed.vopts.sides = 5;
            ed.vopts.star = 0.5;
            ed.demo_drag(&[(1400.0, 200.0), (1700.0, 500.0), (2000.0, 800.0)], shift, win, cx);
            let LayerKind::Shape(s) = &ed.doc.layers[ed.doc.active].kind else { return };
            eprintln!("live shape: {} components, ops {:?}", s.parts.len() + 1, s.parts.iter().map(|p| p.op).collect::<Vec<_>>());
        });
        shot!("pC_live_path_operations");
        // C2: Merge Shape Components.
        step!(|ed: &mut Editor, _win, _cx| {
            ed.merge_shape_components();
            eprintln!("merge: {:.2} ms", ed.last_tool_ms);
        });
        shot!("pC_merged_components");

        // C3: a rotated rounded rectangle (Properties ▸ Angle keeps it a live
        // rectangle); drag one corner widget with ⌥ (one corner only).
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(blank(2600, 1600));
            ed.set_fg([230, 120, 30], win, cx);
            ed.set_tool(Tool::RectShape, win, cx);
            ed.vopts.radius = 0.0;
            ed.vopts.path_op = None;
            ed.demo_drag(&[(700.0, 400.0), (1900.0, 1200.0)], none, win, cx);
            ed.shape_edit("Rotate Shape", |s| s.rot = -20f32.to_radians());
            // The top-left widget, through the shape's rotation.
            let LayerKind::Shape(s) = &ed.doc.layers[ed.doc.active].kind else { return };
            let m = s.first().matrix();
            let from = crate::engine::shape::affine_apply(&m, crate::engine::vector::p(700.0 + 25.0, 400.0 + 25.0));
            let to = crate::engine::shape::affine_apply(&m, crate::engine::vector::p(700.0 + 260.0, 400.0 + 260.0));
            ed.demo_drag(&[(from.x, from.y), (to.x, to.y)], alt, win, cx);
            let LayerKind::Shape(s) = &ed.doc.layers[ed.doc.active].kind else { return };
            eprintln!("rotated rect: {:?}, rot {:.1}°", s.geom, s.rot.to_degrees());
        });
        shot!("pC_rotated_rect_corner_widget");
        // C4: a star, its indent dragged on canvas.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(blank(2600, 1600));
            ed.set_fg([200, 40, 120], win, cx);
            ed.set_tool(Tool::PolygonShape, win, cx);
            ed.vopts.sides = 7;
            ed.vopts.star = 0.2;
            ed.demo_drag(&[(800.0, 300.0), (1800.0, 1300.0)], none, win, cx);
            let c = (1300.0f32, 800.0f32);
            let a = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI / 7.0;
            let from = (c.0 + 500.0 * 0.8 * a.cos(), c.1 + 500.0 * 0.8 * a.sin());
            let to = (c.0 + 500.0 * 0.4 * a.cos(), c.1 + 500.0 * 0.4 * a.sin());
            ed.demo_drag(&[from, to], none, win, cx);
        });
        shot!("pC_star_indent_handle");

        // C5: gradient fill, dashed pattern stroke with an offset (Properties).
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(blank(2600, 1600));
            ed.set_fg([255, 140, 0], win, cx);
            ed.set_bg([90, 20, 170], win, cx);
            ed.set_tool(Tool::EllipseShape, win, cx);
            ed.demo_drag(&[(500.0, 300.0), (2100.0, 1300.0)], none, win, cx);
            let mut g = crate::engine::gradient::Gradient::two("FG to BG", [255, 140, 0], [90, 20, 170]);
            g.sort();
            ed.shape_edit("Fill", |s| s.fill = Paint::Gradient(GradPaint { grad: g, kind: crate::engine::gradient::GradKind::Radial, angle: 0.0, scale: 1.0, reverse: false }));
            ed.shape_edit("Stroke", |s| s.stroke = Some((Paint::Pattern(PatPaint { which: 1, scale: 1.0, angle: 0.0 }), StrokeStyle { width: 60.0, align: Align::Center, dashes: vec![3.0, 1.0], dash_offset: 1.5, ..Default::default() })));
        });
        shot!("pC_gradient_fill_pattern_dashed_stroke");

        // C6: frames: an empty one (placeholder cross), one filled by Place,
        // one filled by dropping an image file onto it.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(blank(3000, 1600));
            ed.set_tool(Tool::Frame, win, cx);
            ed.vopts.frame_ellipse = false;
            ed.demo_drag(&[(150.0, 200.0), (950.0, 1300.0)], none, win, cx);
            ed.vopts.frame_ellipse = true;
            ed.doc.active = 0;
            ed.demo_drag(&[(1100.0, 200.0), (1900.0, 1300.0)], none, win, cx);
            if let Ok(g) = crate::engine::io::open_image(std::path::Path::new("testdata/photos/dog_grass.jpg")) {
                ed.add_smart_layer(crate::engine::layers::SmartObject::new("dog_grass", g, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
            }
            ed.vopts.frame_ellipse = false;
            ed.doc.active = 0;
            ed.demo_drag(&[(2050.0, 200.0), (2850.0, 1300.0)], none, win, cx);
            // Drop the butterfly photo with the cursor over the third frame.
            ed.demo_move(2450.0, 750.0, false, none, win, cx);
            let dropped = ed.drop_into_frame(std::path::PathBuf::from("testdata/photos/butterfly.jpg"), cx);
            eprintln!("dropped into frame: {dropped}");
        });
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        step!(|ed: &mut Editor, win, cx| {
            ed.set_tool(Tool::Frame, win, cx);
            if let Some(i) = ed.doc.layers.iter().position(|l| l.name == "Frame 2") {
                ed.doc.active = i;
            }
        });
        shot!("pC_frames_empty_placed_dropped");
        // C7: Convert to Frame on a text layer, then place a photo in it.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(blank(3000, 1400));
            ed.set_tool(Tool::TypeH, win, cx);
            ed.set_fg([0, 0, 0], win, cx);
            ed.topts.style = TextStyle { size: 520.0, weight: 800, ..Default::default() };
            ed.demo_click(150.0, 1000.0, none, 1, win, cx);
            ed.type_text("FRAME");
            ed.commit_text(cx);
            ed.convert_to_frame();
            if let Ok(g) = crate::engine::io::open_image(std::path::Path::new("testdata/photos/butterfly.jpg")) {
                ed.add_smart_layer(crate::engine::layers::SmartObject::new("butterfly", g, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]));
            }
            ed.set_tool(Tool::Frame, win, cx);
        });
        shot!("pC_text_converted_to_frame");
    }

    if has("D") {
        use crate::engine::layers::{LayerKind, Paint, ShapeGeom, ShapeLayer};
        // A white Background with five coloured shape layers.
        fn shapes_doc(ed: &mut Editor) {
            ed.replace_document(blank(2600, 1500));
            let cols = [[220, 60, 60], [240, 160, 30], [60, 170, 90], [40, 120, 220], [150, 70, 200]];
            for (k, c) in cols.iter().enumerate() {
                let (x, y) = (200.0 + k as f32 * 430.0 + (k % 2) as f32 * 60.0, 250.0 + ((k * 37) % 5) as f32 * 110.0);
                let w = 220.0 + (k % 3) as f32 * 90.0;
                let shape = ShapeLayer::new(ShapeGeom::Rect { x0: x, y0: y, x1: x + w, y1: y + 260.0 + k as f32 * 30.0, radii: [24.0; 4] }, Paint::Solid(*c), None);
                let at = ed.doc.insert_layer(format!("Shape {}", k + 1), crate::engine::grid::TileGrid::new(2600, 1500));
                let l = std::mem::replace(&mut ed.doc.layers[at], crate::engine::document::Layer::new(0, "", crate::engine::grid::TileGrid::new(1, 1)));
                ed.doc.layers[at] = l.with_kind(LayerKind::Shape(std::sync::Arc::new(shape)));
            }
            ed.doc.mark_all_dirty();
        }
        // D1: ⌘-click / Shift-click selects several layers; Align ▸ Top to the
        // key object (Shape 3 stays put).
        step!(|ed: &mut Editor, win, cx| {
            shapes_doc(ed);
            ed.set_tool(Tool::Move, win, cx);
            ed.click_layer(1, false, false, win, cx);
            ed.click_layer(5, false, true, win, cx);
            ed.click_layer(3, true, false, win, cx);
            ed.click_layer(3, true, false, win, cx);
            ed.nopts.align_key = true;
            ed.align_layer("top", cx);
            eprintln!("selected {} layers, key = {}", ed.selected_layers().len(), ed.doc.layers[ed.doc.active].name);
        });
        shot!("pD_multi_select_align_to_key_object");
        // D2: Distribute ▸ equal horizontal spacing.
        step!(|ed: &mut Editor, _win, _cx| ed.distribute_layers(crate::engine::arrange::Distribute::HSpace));
        shot!("pD_distribute_horizontal_spacing");
        // D3: group two layers; Auto-Select ▸ Group picks the group, and a
        // drag moves both.
        step!(|ed: &mut Editor, win, cx| {
            shapes_doc(ed);
            ed.set_tool(Tool::Move, win, cx);
            ed.click_layer(4, false, false, win, cx);
            ed.click_layer(5, false, true, win, cx);
            ed.group_layer(win, cx);
            ed.opts.auto_select = true;
            ed.opts.auto_select_group = true;
            ed.doc.active = 0;
            ed.doc.selected.clear();
            // Shape 4 sits at x ≈ 1550.
            ed.demo_drag(&[(1700.0, 500.0), (1700.0, 800.0), (1700.0, 1000.0)], none, win, cx);
        });
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        step!(|ed: &mut Editor, win, cx| {
            ed.demo_move(1700.0, 1000.0, false, none, win, cx);
            eprintln!("auto-select group: active = {}", ed.doc.layers[ed.doc.active].name);
        });
        shot!("pD_auto_select_group_moved");
        // D4: the locked Background: Move is refused, the Eraser paints the BG colour.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("dog_grass"));
            ed.set_bg([255, 220, 120], win, cx);
            ed.set_tool(Tool::Move, win, cx);
            ed.opts.auto_select = false;
            ed.demo_drag(&[(1000.0, 1000.0), (1400.0, 1200.0)], none, win, cx);
            ed.set_tool(Tool::Eraser, win, cx);
            ed.brush_of(Tool::Eraser).size = 220.0;
            ed.demo_drag(&[(600.0, 600.0), (1500.0, 700.0), (2500.0, 650.0)], none, win, cx);
            eprintln!("background: {}, pixel {:?}", ed.doc.layers[0].background, crate::engine::px::unpack(ed.doc.layers[0].grid.pixel(1500, 700)));
        });
        shot!("pD_background_locked_eraser_paints_bg");
        // D5: artboards from presets, each holding content; Layers shows them as folders.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(blank(1200, 800));
            ed.set_tool(Tool::Artboard, win, cx);
            ed.add_preset_artboard("Instagram Post", 1080, 1080);
            ed.set_fg([40, 120, 220], win, cx);
            ed.set_tool(Tool::EllipseShape, win, cx);
            ed.demo_drag(&[(200.0, 200.0), (880.0, 880.0)], none, win, cx);
            ed.set_tool(Tool::Artboard, win, cx);
            ed.add_preset_artboard("Instagram Story", 1080, 1920);
            ed.set_fg([230, 90, 40], win, cx);
            ed.set_tool(Tool::PolygonShape, win, cx);
            ed.vopts.sides = 5;
            ed.vopts.star = 0.45;
            let x0 = ed.artboards().last().map_or(0, |a| a.2.x0) as f32;
            ed.demo_drag(&[(x0 + 140.0, 500.0), (x0 + 940.0, 1300.0)], none, win, cx);
            ed.set_tool(Tool::Artboard, win, cx);
            let dir = std::env::temp_dir().join("pf_artboards");
            let _ = std::fs::create_dir_all(&dir);
            for fmt in [0u8, 1, 2] {
                ed.nopts.artboard_format = fmt;
            }
            eprintln!("artboards: {:?}", ed.artboards().iter().map(|a| (ed.doc.layers[a.0].name.clone(), a.2)).collect::<Vec<_>>());
            match ed.export_regions_now(false, &dir) {
                Ok(n) => eprintln!("exported {n} artboard files to {}", dir.display()),
                Err(e) => eprintln!("export failed: {e}"),
            }
        });
        shot!("pD_artboards_presets_with_content");
        // D6: slices: a user slice, a layer-based slice, Divide Slice, Slice
        // Options, and the Save for Web preview (PNG-8, 64 colours).
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("dog_grass"));
            ed.set_tool(Tool::Slice, win, cx);
            ed.demo_drag(&[(300.0, 300.0), (1500.0, 900.0)], none, win, cx);
            ed.set_tool(Tool::SliceSelect, win, cx);
            ed.nopts.slice_sel = Some(0);
            ed.divide_slice(2, 1);
            ed.nopts.slice_sel = Some(1);
            if let Some(s) = ed.doc.ann.slices.get_mut(1) {
                s.name = "hero_right".into();
                s.url = "https://example.com/dogs".into();
                s.target = "_blank".into();
                s.alt = "A boxer on the lawn".into();
            }
            ed.slice_fields = None;
            ed.open_save_for_web();
            if let Some(w) = &mut ed.web {
                w.colors = 64;
            }
        });
        shot!("pD_slices_options_save_for_web");
    }

    if has("E") {
        use crate::engine::crop::{CropOverlay, Interp};
        // E1: Rotate View at 30°: the cached display tiles drawn as rotated
        // GPU sprites (no per-frame CPU resampling).
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("dog_grass"));
            ed.set_tool(Tool::RotateView, win, cx);
        });
        until!(|ed: &mut Editor, _| ed.cache.last.pending == 0);
        step!(|ed: &mut Editor, _win, _cx| ed.vp.rot = 30f32.to_radians());
        until!(|ed: &mut Editor, _| ed.cache.last.pending == 0);
        step!(|ed: &mut Editor, _win, _cx| {
            let st = ed.cache.last;
            eprintln!("rotate view 30°: {} visible tiles at level {}, plan {:.2} ms", st.visible, st.level, st.compose_ms);
        });
        shot!("pE_rotate_view_gpu_30");
        // E2: zoomed in to 200 % while rotated (nearest-neighbour magnified tiles).
        step!(|ed: &mut Editor, _win, _cx| {
            let c = ed.canvas.center();
            ed.vp.zoom_about(2.0, c, ed.canvas);
            ed.vp.rot = -12f32.to_radians();
        });
        until!(|ed: &mut Editor, _| ed.cache.last.pending == 0);
        shot!("pE_rotate_view_gpu_200pct");
        // E2b: a crop frame in a rotated view: its handles turn with the view.
        step!(|ed: &mut Editor, win, cx| {
            ed.needs_fit = true;
            ed.vp.rot = 20f32.to_radians();
            ed.set_tool(Tool::Crop, win, cx);
            ed.crop = Some(crate::engine::grid::RectI::new(600, 400, 3200, 2200));
        });
        until!(|ed: &mut Editor, _| ed.cache.last.pending == 0);
        shot!("pE_rotate_view_crop_handles");
        // E3: Crop with a rotated frame and the Golden Spiral overlay.
        step!(|ed: &mut Editor, win, cx| {
            ed.vp.rot = 0.0;
            ed.crop = None;
            ed.needs_fit = true;
            ed.set_tool(Tool::Crop, win, cx);
            ed.demo_drag(&[(500.0, 400.0), (2100.0, 900.0), (3300.0, 2200.0)], none, win, cx);
            ed.nopts.crop_angle = -6.0;
            ed.nopts.crop_overlay = CropOverlay::GoldenSpiral;
        });
        shot!("pE_crop_golden_spiral");
        step!(|ed: &mut Editor, _win, _cx| {
            ed.nopts.crop_overlay = CropOverlay::Triangle;
            ed.cycle_crop_overlay(true);
        });
        shot!("pE_crop_triangle_flipped");
        step!(|ed: &mut Editor, _win, _cx| ed.nopts.crop_overlay = CropOverlay::Diagonal);
        shot!("pE_crop_diagonal");
        step!(|ed: &mut Editor, _win, _cx| ed.nopts.crop_overlay = CropOverlay::GoldenRatio);
        shot!("pE_crop_golden_ratio");
        step!(|ed: &mut Editor, _win, _cx| ed.nopts.crop_overlay = CropOverlay::Grid);
        shot!("pE_crop_grid");
        // E4: Straighten inside Crop: drag along a tilted horizon; the frame
        // turns and shrinks to the largest level rectangle.
        step!(|ed: &mut Editor, win, cx| {
            ed.crop = None;
            ed.nopts.crop_angle = 0.0;
            ed.nopts.crop_overlay = CropOverlay::Thirds;
            ed.nopts.crop_straighten = true;
            ed.demo_press(400.0, 1500.0, none, 1, win, cx);
            ed.demo_move(2000.0, 1640.0, true, none, win, cx);
            ed.demo_move(3400.0, 1763.0, true, none, win, cx);
        });
        shot!("pE_crop_straighten_line");
        step!(|ed: &mut Editor, win, cx| {
            ed.demo_release(3400.0, 1763.0, win, cx);
            eprintln!("straighten: frame {:?} at {:.2}°", ed.crop, ed.nopts.crop_angle);
        });
        shot!("pE_crop_straighten_frame");
        // E5: Delete Cropped Pixels off + Content-Aware, a frame rotated 8°
        // and reaching past the right edge (Lanczos).
        step!(|ed: &mut Editor, win, cx| {
            ed.crop = Some(crate::engine::grid::RectI::new(900, 300, 4100, 2300));
            ed.nopts.crop_angle = 8.0;
            ed.nopts.crop_delete = false;
            ed.nopts.crop_fill = true;
            ed.nopts.crop_interp = Interp::Lanczos;
            let _ = win;
            ed.commit_crop(cx);
        });
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        step!(|ed: &mut Editor, _win, _cx| {
            let l = &ed.doc.layers[0];
            eprintln!("cropped {}×{}: layer '{}', hidden {:?}", ed.doc.w, ed.doc.h, l.name, l.hidden.as_ref().map(|h| h.rect()));
        });
        shot!("pE_crop_keep_pixels_content_aware");
        // E6: Move the layer: the hidden pixels come back into view.
        step!(|ed: &mut Editor, win, cx| {
            ed.set_tool(Tool::Move, win, cx);
            ed.demo_press(1500.0, 1000.0, none, 1, win, cx);
            ed.demo_move(1900.0, 1100.0, true, none, win, cx);
            ed.demo_move(2300.0, 1200.0, true, none, win, cx);
        });
        shot!("pE_move_drag_live_hidden_pixels");
        step!(|ed: &mut Editor, win, cx| ed.demo_release(2300.0, 1200.0, win, cx));
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        shot!("pE_move_reveals_hidden_pixels");
        // E7: Image ▸ Reveal All grows the canvas to show everything.
        step!(|ed: &mut Editor, _win, cx| ed.reveal_all(cx));
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        step!(|ed: &mut Editor, _win, _cx| eprintln!("reveal all: {}×{}", ed.doc.w, ed.doc.h));
        shot!("pE_reveal_all");
        // E8: Straighten Layer on a non-Background layer turns only that layer.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("butterfly"));
            ed.nopts.crop_delete = true;
            ed.nopts.crop_fill = false;
            let sel = crate::engine::selection::Selection::rect(crate::engine::grid::RectI::new(700, 700, 1900, 1500));
            let g = crate::engine::ops::fill(&crate::engine::grid::TileGrid::new(2664, 2660), Some(&sel), [40, 110, 220], 1.0);
            // A card tilted 8°; the ruler follows its top edge.
            let m = crate::engine::annot::rotation_about(8.0, 1300.0, 1100.0);
            let g = crate::engine::crop::warp_affine(&g, &m, 2664, 2660, Interp::Bicubic);
            let at = ed.doc.insert_layer("Card", g);
            ed.doc.active = at;
            ed.set_tool(Tool::Ruler, win, cx);
            let ap = |x: f32, y: f32| (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]);
            let (a, b) = (ap(700.0, 700.0), ap(1900.0, 700.0));
            ed.demo_drag(&[a, ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5), b], none, win, cx);
        });
        shot!("pE_straighten_layer_ruler");
        step!(|ed: &mut Editor, _win, cx| {
            ed.straighten(false, cx);
        });
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        step!(|ed: &mut Editor, _win, _cx| {
            let l = &ed.doc.layers[ed.doc.active];
            eprintln!("straighten layer: '{}' canvas {}×{}, hidden {:?}", l.name, ed.doc.w, ed.doc.h, l.hidden.as_ref().map(|h| h.rect()));
        });
        shot!("pE_straighten_layer_only");
        // E9: three documents as tabs.
        step!(|ed: &mut Editor, _win, _cx| {
            ed.doc.modified = true;
            ed.open_document(photo_doc("dog_grass"));
            ed.open_document(photo_doc("portrait_hair"));
            eprintln!("tabs: {:?}", ed.doc_tabs().iter().map(|t| t.1.clone()).collect::<Vec<_>>());
        });
        until!(|ed: &mut Editor, _| ed.cache.last.pending == 0);
        shot!("pE_tabs");
        // E10: 2-up: the active document and the next one side by side.
        step!(|ed: &mut Editor, _win, _cx| ed.set_arrange(super::docs::Arrange::TwoUp));
        wait(1500).await;
        shot!("pE_two_up");
        // E11: Tile All in Grid, then zoom / rotate the active view and Match All.
        step!(|ed: &mut Editor, _win, _cx| ed.set_arrange(super::docs::Arrange::Tile));
        wait(1500).await;
        step!(|ed: &mut Editor, _win, _cx| {
            let c = ed.canvas.center();
            ed.vp.zoom_about(0.12, c, ed.canvas);
            ed.vp.rot = 15f32.to_radians();
            ed.match_views(super::docs::Match::All);
        });
        wait(1500).await;
        shot!("pE_tile_match_all");
        // E12: Zoom All: zooming the active view zooms the others too.
        step!(|ed: &mut Editor, _win, _cx| {
            ed.docs.zoom_all = true;
            let c = ed.canvas.center();
            ed.vp.zoom_about(ed.vp.zoom * 2.0, c, ed.canvas);
        });
        wait(1500).await;
        step!(|ed: &mut Editor, _win, _cx| {
            let z: Vec<String> = ed.docs.slots.iter().map(|s| format!("{:.0}%", s.vp.zoom * 100.0)).collect();
            eprintln!("zoom all: active {:.0}%, others {z:?}", ed.vp.zoom * 100.0);
        });
        shot!("pE_tile_zoom_all");
        // E13: a background job finishes on the document it started on.
        step!(|ed: &mut Editor, win, cx| {
            ed.docs.zoom_all = false;
            ed.vp.rot = 0.0;
            let uid = ed.docs.uid;
            ed.act_canvas("rot_cw", cx);
            eprintln!("rotate 90° started on document {uid}");
            let _ = win;
        });
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        step!(|ed: &mut Editor, win, cx| {
            eprintln!("after rotate: active {}×{}", ed.doc.w, ed.doc.h);
            let first = ed.docs.order[0];
            ed.activate_doc(first, win, cx);
        });
        wait(1500).await;
        shot!("pE_tile_activate_first");
    }

    if has("F") {
        use super::dock::DockTab;
        // F1: Ruler + protractor (⌥-drag a second leg from an end point).
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("dog_grass"));
            ed.set_tool(Tool::Ruler, win, cx);
            ed.demo_drag(&[(900.0, 1900.0), (1600.0, 1700.0), (2400.0, 1500.0)], none, win, cx);
            ed.demo_drag(&[(900.0, 1900.0), (1100.0, 1500.0), (1300.0, 1000.0)], alt, win, cx);
            if let (Some((v, b)), Some(c)) = (ed.nopts.ruler, ed.nopts.ruler2) {
                eprintln!("protractor angle {:.1}°", crate::engine::measure::protractor_angle(v, b, c));
            }
            ed.record_measurements(cx);
        });
        shot!("pF_ruler_protractor");
        // F2: Measurement Log: the protractor, two count groups and an
        // elliptical selection, with a 100 px = 2.54 cm scale.
        step!(|ed: &mut Editor, win, cx| {
            ed.doc.ann.measure.scale = crate::engine::measure::MeasureScale { px: 100.0, length: 2.54, units: "cm".into() };
            ed.set_tool(Tool::Count, win, cx);
            for (x, y) in [(400.0, 400.0), (700.0, 520.0), (1000.0, 380.0), (3300.0, 600.0)] {
                ed.demo_click(x, y, none, 1, win, cx);
            }
            ed.doc.ann.counts.push(crate::engine::annot::CountGroup::new("Daisies", [250, 200, 40]));
            ed.nopts.count_group = 1;
            for (x, y) in [(1690.0, 420.0), (1760.0, 470.0), (1820.0, 430.0)] {
                ed.demo_click(x, y, none, 1, win, cx);
            }
            ed.doc.ann.counts[1].label_size = 26.0;
            ed.record_measurements(cx);
            ed.set_tool(Tool::EllipseSelect, win, cx);
            ed.demo_drag(&[(2200.0, 400.0), (2600.0, 800.0), (3000.0, 1100.0)], none, win, cx);
            ed.record_measurements(cx);
        });
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        step!(|ed: &mut Editor, _win, _cx| {
            let dir = std::env::temp_dir().join("pf_measurements.csv");
            let _ = ed.export_measurements_to(&dir);
            eprintln!("measurement log: {} records → {}", ed.doc.ann.measure.records.len(), dir.display());
        });
        shot!("pF_measurement_log");
        // F3: Count label size and a hidden group.
        step!(|ed: &mut Editor, win, cx| {
            ed.set_tool(Tool::Count, win, cx);
            ed.doc.ann.counts[0].visible = false;
            ed.nopts.count_group = 1;
        });
        shot!("pF_count_label_size_hidden_group");
        // F4: Notes panel: author from the preferences, jump to a note.
        step!(|ed: &mut Editor, win, cx| {
            ed.doc.ann.counts[0].visible = true;
            ed.dockst.prefs.author = "Radu D.".into();
            ed.set_tool(Tool::Note, win, cx);
            for (x, y) in [(500.0, 300.0), (2900.0, 1900.0), (1700.0, 1200.0)] {
                ed.demo_click(x, y, none, 1, win, cx);
            }
            let texts = ["Lift the shadows on the lawn", "Crop tighter on the right", "Collar colour: check against the brief"];
            for (i, t) in texts.iter().enumerate() {
                let k = ed.doc.ann.notes.len() - 3 + i;
                ed.set_note_text(k, t, win, cx);
            }
            ed.dockst.tab = Some(DockTab::Notes);
            let k = ed.doc.ann.notes.len() - 2;
            ed.jump_to_note(k, win, cx);
        });
        shot!("pF_notes_panel");
        // F5: Info panel: color samplers each with their own mode.
        step!(|ed: &mut Editor, win, cx| {
            ed.set_tool(Tool::ColorSampler, win, cx);
            for (x, y) in [(1500.0, 900.0), (2950.0, 900.0), (600.0, 2200.0), (1750.0, 450.0)] {
                ed.demo_click(x, y, none, 1, win, cx);
            }
            use crate::engine::color::ColorMode;
            ed.doc.ann.sampler_modes = vec![ColorMode::Rgb, ColorMode::Lab, ColorMode::Cmyk, ColorMode::Hsb];
            ed.nopts.info_mode = ColorMode::Cmyk;
            ed.dockst.tab = None;
        });
        shot!("pF_info_sampler_modes");
        // F6: Brushes panel: the default presets plus Photoshop's own
        // Legacy Brushes imported from its .abr.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(blank(3000, 1800));
            ed.set_tool(Tool::Brush, win, cx);
            let abr = std::path::Path::new("/Applications/Adobe Photoshop 2026/Presets/Brushes/Legacy Brushes.abr");
            if abr.exists() {
                ed.import_abr(abr);
            }
            ed.dockst.tab = Some(DockTab::Brushes);
        });
        shot!("pF_brush_presets_abr_import");
        // F7: strokes with each preset (and an imported sampled tip).
        step!(|ed: &mut Editor, win, cx| {
            ed.fg = [40, 90, 200];
            ed.bg = [250, 200, 40];
            let lib = ed.dockst.presets.clone();
            let mut row = 0;
            for (f, folder) in lib.folders.iter().enumerate().take(3) {
                for (i, _) in folder.presets.iter().enumerate() {
                    ed.apply_preset(f, i);
                    let y = 150.0 + row as f32 * 140.0;
                    let pts: Vec<(f32, f32)> = (0..=24).map(|k| (200.0 + k as f32 * 105.0, y + (k as f32 * 0.6).sin() * 40.0)).collect();
                    ed.demo_drag(&pts, none, win, cx);
                    row += 1;
                }
            }
            if lib.folders.len() > 3 {
                for i in [3usize, 20] {
                    if i < lib.folders[3].presets.len() {
                        ed.apply_preset(3, i);
                        ed.brush_mut().scatter = 0.8;
                        ed.brush_mut().shape = crate::engine::brush::ShapeDynamics { on: true, angle_jitter: 0.5, size_jitter: 0.4, ..Default::default() };
                        let y = 150.0 + row as f32 * 140.0;
                        let pts: Vec<(f32, f32)> = (0..=24).map(|k| (200.0 + k as f32 * 105.0, y + (k as f32 * 0.6).cos() * 40.0)).collect();
                        ed.demo_drag(&pts, none, win, cx);
                        row += 1;
                    }
                }
            }
            eprintln!("painted {row} preset strokes");
        });
        shot!("pF_preset_strokes");
        // F8: Brush Settings panel with every dynamic on.
        step!(|ed: &mut Editor, win, cx| {
            ed.set_tool(Tool::Brush, win, cx);
            *ed.brush_mut() = crate::engine::brush::all_dynamics(120.0, 7);
            ed.dockst.tab = Some(DockTab::BrushSettings);
            let pts: Vec<(f32, f32)> = (0..=30).map(|k| (300.0 + k as f32 * 80.0, 1650.0 - (k as f32 * 0.25).sin() * 80.0)).collect();
            ed.demo_drag(&pts, none, win, cx);
        });
        shot!("pF_brush_settings_all_dynamics");
    }

    if has("G") {
        use super::vp::VpTool;
        use crate::engine::grid::{RectI, TileGrid};
        use crate::engine::vanish::{self, Plane, Scene};
        // A synthetic room: a tiled floor and a brick wall in perspective.
        fn room() -> (Document, Plane) {
            let (w, h) = (2400u32, 1600u32);
            let mut g = crate::engine::ops::fill(&TileGrid::new(w, h), None, [214, 222, 230], 1.0);
            let floor = Plane { c: [(700.0, 760.0), (1750.0, 720.0), (2300.0, 1500.0), (150.0, 1560.0)] };
            let checker = crate::engine::grid::build_grid(512, 512, |tx, ty| {
                let mut t = vec![0u32; 256 * 256];
                for y in 0..256 {
                    for x in 0..256 {
                        let (gx, gy) = (tx * 256 + x as i32, ty * 256 + y as i32);
                        let dark = ((gx / 64) + (gy / 64)) % 2 == 0;
                        let v = if dark { [120, 90, 70] } else { [235, 225, 205] };
                        t[y * 256 + x] = crate::engine::px::pack(v[0], v[1], v[2], 255);
                    }
                }
                crate::engine::grid::finish_tile(t.into_boxed_slice())
            });
            let mut sc = Scene::new(w, h);
            sc.planes.push(floor);
            let f = sc.frame(0);
            g = vanish::place(&g, &checker, &f, (0.0, 0.0, f.sw, f.sh)).0;
            // A rug to clone and a scuff to heal away.
            g = vanish::dab(&g, &g.clone(), &f, f.sw * 0.3, f.sh * 0.7, f.sh * 0.12, 0.9, vanish::DabSource::Color([160, 30, 40], 1.0)).0;
            g = vanish::dab(&g, &g.clone(), &f, f.sw * 0.75, f.sh * 0.45, f.sh * 0.05, 0.2, vanish::DabSource::Color([30, 30, 30], 0.8)).0;
            (Document::from_grid(g, "Background"), floor)
        }
        // G1: Create Plane: four clicks on the floor's corners.
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(photo_doc("butterfly"));
            ed.doc.selection = Some(crate::engine::selection::Selection::rect(RectI::new(300, 200, 2300, 2400)));
            ed.act_copy(false, cx);
            ed.doc.selection = None;
            let (doc, floor) = room();
            ed.replace_document(doc);
            ed.open_vanishing_point(cx);
            for c in floor.c {
                ed.demo_click(c.0, c.1, none, 1, win, cx);
            }
            let st = ed.vpw.as_ref().unwrap();
            eprintln!("planes {} focal {:.0}", st.scene.planes.len(), st.scene.focal());
        });
        shot!("pG_vp_create_plane");
        // G2: ⌘-drag the far edge's middle handle: a perpendicular wall.
        step!(|ed: &mut Editor, win, cx| {
            let p = ed.vpw.as_ref().unwrap().scene.planes[0];
            let mid = ((p.c[0].0 + p.c[1].0) * 0.5, (p.c[0].1 + p.c[1].1) * 0.5);
            let cmd = gpui_kit::Modifiers { platform: true, ..Default::default() };
            ed.demo_drag(&[mid, (mid.0, mid.1 - 300.0), (mid.0, mid.1 - 560.0)], cmd, win, cx);
            let st = ed.vpw.as_ref().unwrap();
            eprintln!("tear-off: {} planes", st.scene.planes.len());
            // Bricks on the wall (painted through the plane, for the demo).
            if let Some(li) = ed.doc.layers.iter().position(|l| l.id == st.layer) {
                let f = st.scene.frame(1);
                let bricks = crate::engine::grid::build_grid(600, 400, |tx, ty| {
                    let mut t = vec![0u32; 256 * 256];
                    for y in 0..256 {
                        for x in 0..256 {
                            let (gx, gy) = (tx * 256 + x as i32, ty * 256 + y as i32);
                            let row = gy / 40;
                            let off = if row % 2 == 0 { 0 } else { 40 };
                            let mortar = gy % 40 < 4 || (gx + off) % 80 < 4;
                            let v = if mortar { [200, 200, 195] } else { [150 + (row % 3) as u8 * 10, 70, 50] };
                            t[y * 256 + x] = crate::engine::px::pack(v[0], v[1], v[2], 255);
                        }
                    }
                    crate::engine::grid::finish_tile(t.into_boxed_slice())
                });
                let g = vanish::place(&ed.doc.layers[li].grid, &bricks, &f, (0.0, 0.0, f.sw, f.sh)).0;
                ed.doc.set_layer_grid(li, g);
            }
        });
        shot!("pG_vp_tear_off_wall");
        // G3: Marquee on the floor around the rug, ⌥-dragged back: the copy
        // shrinks with the perspective.
        step!(|ed: &mut Editor, win, cx| {
            let f = ed.vpw.as_ref().unwrap().scene.frame(0);
            ed.vpw.as_mut().unwrap().tool = VpTool::Marquee;
            let a = f.to_img(f.sw * 0.15, f.sh * 0.55);
            let b = f.to_img(f.sw * 0.45, f.sh * 0.85);
            ed.demo_drag(&[a, b], none, win, cx);
            let c = f.to_img(f.sw * 0.3, f.sh * 0.7);
            let d = f.to_img(f.sw * 0.6, f.sh * 0.2);
            ed.demo_drag(&[c, ((c.0 + d.0) * 0.5, (c.1 + d.1) * 0.5), d], alt, win, cx);
        });
        shot!("pG_vp_marquee_copy_in_perspective");
        // G4: Stamp with Heal: the scuff is cloned over from clean floor.
        step!(|ed: &mut Editor, win, cx| {
            let f = ed.vpw.as_ref().unwrap().scene.frame(0);
            let st = ed.vpw.as_mut().unwrap();
            st.marquee = None;
            st.tool = VpTool::Stamp;
            st.radius = f.sh * 0.08;
            let src = f.to_img(f.sw * 0.75, f.sh * 0.70);
            ed.demo_click(src.0, src.1, alt, 1, win, cx);
            let pts: Vec<(f32, f32)> = (0..=10).map(|k| f.to_img(f.sw * 0.72 + k as f32 * f.sw * 0.006, f.sh * 0.45)).collect();
            ed.demo_drag(&pts, none, win, cx);
        });
        shot!("pG_vp_stamp_heal");
        // G5: Brush on the plane: a stripe that narrows with the depth.
        step!(|ed: &mut Editor, win, cx| {
            let f = ed.vpw.as_ref().unwrap().scene.frame(0);
            ed.fg = [30, 110, 220];
            let st = ed.vpw.as_mut().unwrap();
            st.tool = VpTool::Brush;
            st.radius = f.sh * 0.04;
            st.hardness = 0.8;
            let pts: Vec<(f32, f32)> = (0..=30).map(|k| f.to_img(f.sw * 0.1, f.sh * (0.95 - k as f32 * 0.03))).collect();
            ed.demo_drag(&pts, none, win, cx);
        });
        shot!("pG_vp_brush_on_plane");
        // G6: Transform: paste the butterfly onto the wall.
        step!(|ed: &mut Editor, _win, _cx| {
            let st = ed.vpw.as_mut().unwrap();
            st.sel = Some(1);
            ed.vp_paste();
        });
        shot!("pG_vp_transform_paste");
        step!(|ed: &mut Editor, _win, cx| {
            ed.vp_apply(cx);
            eprintln!("applied: history {:?}, planes saved {}", ed.doc.history_labels().0.last(), ed.doc.ann.vp_planes.len());
        });
        shot!("pG_vp_applied");
    }
    if has("H") {
        use crate::engine::perspwarp::{Mesh, Straighten};
        use super::perspwarp_ui::PwPhase;
        // A two-wall building shot from below: the walls' verticals lean in.
        let (w, h) = (3000u32, 2000u32);
        let walls: [[(f32, f32); 4]; 2] = [[(700.0, 300.0), (1500.0, 420.0), (1500.0, 1700.0), (650.0, 1800.0)], [(1500.0, 420.0), (2300.0, 280.0), (2380.0, 1820.0), (1500.0, 1700.0)]];
        let scene = || {
            let bricks = crate::engine::grid::build_grid(w, h, |tx, ty| {
                let mut t = vec![0u32; 256 * 256];
                for y in 0..256 {
                    for x in 0..256 {
                        let (gx, gy) = (tx * 256 + x as i32, ty * 256 + y as i32);
                        let win = gx % 200 > 60 && gx % 200 < 140 && gy % 250 > 70 && gy % 250 < 190;
                        let v = if win { [60, 90, 130] } else if (gx / 100 + gy / 125) % 2 == 0 { [196, 120, 90] } else { [180, 108, 80] };
                        t[y * 256 + x] = crate::engine::px::pack(v[0], v[1], v[2], 255);
                    }
                }
                crate::engine::grid::finish_tile(t.into_boxed_slice())
            });
            // Upright walls (rectangles) pushed into the keystoned view.
            let mut m = Mesh::default();
            for q in walls {
                let (x0, x1) = (q[0].0.min(q[3].0), q[1].0.max(q[2].0));
                m.add_quad(Mesh::rect_quad((x0, 300.0), (x1, 1800.0)), 2.0);
            }
            for (k, q) in walls.iter().enumerate() {
                for c in 0..4 {
                    let v = m.quads[k][c];
                    m.warp[v] = q[c];
                }
            }
            Document::from_grid(crate::engine::perspwarp::render_cpu(&bricks, &m.table(w, h)), "Background")
        };
        // H1: Layout: a quad dragged over each wall, corners onto the wall's
        // corners (the shared corners join).
        step!(|ed: &mut Editor, win, cx| {
            ed.replace_document(scene());
            ed.open_perspective_warp(cx);
            for q in walls {
                let (a, b) = ((q[0].0 + 60.0, q[0].1 + 80.0), (q[2].0 - 60.0, q[2].1 - 80.0));
                ed.demo_drag(&[a, ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5), b], none, win, cx);
                let st = ed.pww.as_ref().unwrap();
                let quad = *st.mesh.quads.last().unwrap();
                let corners: Vec<(f32, f32)> = quad.iter().map(|&v| st.mesh.layout[v]).collect();
                for (c, &target) in corners.iter().zip(q.iter()) {
                    ed.demo_drag(&[*c, target], none, win, cx);
                }
            }
            let st = ed.pww.as_ref().unwrap();
            eprintln!("perspective warp layout: {} quads, {} joined edges", st.mesh.quads.len(), st.mesh.shared_edges());
        });
        shot!("pH_perspwarp_layout");
        // H2: Warp + Straighten Vertical: the walls stand upright.
        step!(|ed: &mut Editor, _win, cx| {
            ed.pw_set_phase(PwPhase::Warp, cx);
            ed.pw_straighten(Straighten::Vertical, cx);
        });
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        shot!("pH_perspwarp_straightened");
        // H3: drag the shared corner out: the joined walls stay continuous.
        step!(|ed: &mut Editor, win, cx| {
            let p = ed.pww.as_ref().unwrap().mesh.warp[1];
            ed.demo_drag(&[p, (p.0 + 150.0, p.1 - 60.0)], none, win, cx);
        });
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        shot!("pH_perspwarp_corner_drag");
        step!(|ed: &mut Editor, _win, cx| ed.pw_apply(cx));
        until!(|ed: &mut Editor, _| ed.busy.is_none());
        step!(|ed: &mut Editor, _win, _cx| eprintln!("applied: history {:?}", ed.doc.history_labels().0.last()));
        shot!("pH_perspwarp_applied");
    }
}
