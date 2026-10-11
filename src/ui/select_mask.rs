//! Select ▸ Select and Mask…: the dialog (Edge Detection, Global
//! Refinements, Output Settings), its live overlay preview of the refined
//! selection, and the undoable output (selection, layer mask, or a new layer
//! with a layer mask, decontaminated when asked).

use std::sync::Arc;

use gpui_kit::component::slider::SliderEvent;
use gpui_kit::{Context, Window};

use crate::engine::document::Layer;
use crate::engine::grid::TileGrid;
use crate::engine::layers::LayerMask;
use crate::engine::masks;
use crate::engine::refine::{self, Output, Params, Refined};
use crate::engine::selection::MaskGrid;

use super::editor::{Dialog, DialogKind, DialogSlider, Editor, slider};

pub struct SelectMaskState {
    /// The selection being refined and the pixels guiding it.
    pub base: Arc<MaskGrid>,
    pub img: Arc<TileGrid>,
    /// Red overlay over the unselected area (like Quick Mask).
    pub overlay: Layer,
    /// Bumped per parameter change; a finished preview for an older version
    /// starts the next one instead of being shown.
    pub version: u64,
    pub computing: bool,
    /// The newest result and its parameters (reused by OK).
    pub last: Option<(Params, Arc<Refined>)>,
}

/// (label, min, max, step, default, decimals, choices) in dialog order.
type SliderSpec = (&'static str, f32, f32, f32, f32, usize, &'static [&'static str]);

pub const SLIDERS: [SliderSpec; 9] = [
    ("Edge Detection · Radius (px)", 0.0, 250.0, 1.0, 10.0, 0, &[]),
    ("Smart Radius", 0.0, 1.0, 1.0, 0.0, 0, &["Off", "On"]),
    ("Smooth", 0.0, 100.0, 1.0, 0.0, 0, &[]),
    ("Feather (px)", 0.0, 250.0, 0.1, 0.0, 1, &[]),
    ("Contrast (%)", 0.0, 100.0, 1.0, 0.0, 0, &[]),
    ("Shift Edge (%)", -100.0, 100.0, 1.0, 0.0, 0, &[]),
    ("Decontaminate Colors", 0.0, 1.0, 1.0, 0.0, 0, &["Off", "On"]),
    ("Amount (%)", 0.0, 100.0, 1.0, 100.0, 0, &[]),
    ("Output To", 0.0, 2.0, 1.0, 0.0, 0, Output::LABELS),
];

pub fn params_from(v: &[f32]) -> Params {
    let at = |i: usize| v.get(i).copied().unwrap_or(SLIDERS[i].4);
    Params {
        radius: at(0),
        smart: at(1) >= 0.5,
        smooth: at(2),
        feather: at(3),
        contrast: at(4),
        shift: at(5),
        decontaminate: at(6) >= 0.5,
        amount: at(7),
        output: Output::ALL[(at(8).round().max(0.0) as usize).min(2)],
    }
}

fn overlay_of(m: &MaskGrid) -> Layer {
    let g = m.to_gray();
    let mut ov = TileGrid::new(m.w, m.h);
    ov.tiles = g.tiles.iter().map(masks::quick_mask_overlay_tile).collect();
    Layer::new(0, "Select and Mask", ov)
}

impl Editor {
    pub fn open_select_mask(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.input_blocked() {
            return;
        }
        let Some(sel) = self.doc.selection.clone() else {
            self.flash("Make a selection first (Select and Mask refines the current selection)");
            cx.notify();
            return;
        };
        let (w, h) = (self.doc.w, self.doc.h);
        let base = Arc::new(MaskGrid::from_selection(&sel, w, h));
        let img = Arc::new(self.doc.layers[self.doc.active].grid.clone());
        let mut sliders = Vec::new();
        let mut subs = Vec::new();
        for (label, min, max, step, def, decimals, choices) in SLIDERS {
            let state = slider(cx, min, max, step, def, false);
            subs.push(cx.subscribe_in(&state, window, |this, _, _: &SliderEvent, _, cx| this.update_select_mask_preview(cx)));
            sliders.push(DialogSlider { label, state, decimals, choices });
        }
        self.select_mask = Some(SelectMaskState { overlay: overlay_of(&base), base, img, version: 0, computing: false, last: None });
        self.dialog = Some(Dialog { kind: DialogKind::SelectMask, sliders, inputs: vec![], flag: false, stats: None, source: None, disp_map: None, _subs: subs });
        self.update_select_mask_preview(cx);
    }

    fn select_mask_params(&self, cx: &Context<Self>) -> Option<Params> {
        let d = self.dialog.as_ref().filter(|d| matches!(d.kind, DialogKind::SelectMask))?;
        let vals: Vec<f32> = d.sliders.iter().map(|s| s.state.read(cx).value().start()).collect();
        Some(params_from(&vals))
    }

    /// Recomputes the overlay for the current sliders (one computation at a
    /// time; the newest settings win).
    pub fn update_select_mask_preview(&mut self, cx: &mut Context<Self>) {
        let Some(p) = self.select_mask_params(cx) else { return };
        let Some(st) = &mut self.select_mask else { return };
        st.version += 1;
        if st.computing {
            return;
        }
        if st.last.as_ref().is_some_and(|l| l.0 == p) {
            return;
        }
        st.computing = true;
        let (base, img, v) = (st.base.clone(), st.img.clone(), st.version);
        cx.spawn(async move |this, cx| {
            let r = cx.background_executor().spawn(async move { refine::refine(&base, &img, &p) }).await;
            let _ = this.update(cx, |ed, cx| {
                let Some(st) = &mut ed.select_mask else { return };
                st.computing = false;
                st.overlay = overlay_of(&r.mask);
                st.last = Some((p, Arc::new(r)));
                if st.version != v {
                    ed.update_select_mask_preview(cx);
                }
                ed.doc.mark_all_dirty();
                cx.notify();
            });
        })
        .detach();
    }

    /// OK: applies the refinement (reusing the preview's result when it is
    /// current) as one undo step.
    pub fn apply_select_mask(&mut self, p: Params, cx: &mut Context<Self>) {
        let Some(st) = self.select_mask.take() else { return };
        self.doc.mark_all_dirty();
        let reuse = st.last.filter(|l| l.0 == p).map(|l| l.1);
        let (base, img) = (st.base, st.img);
        let id = self.doc.layers[self.doc.active].id;
        if p.decon() && p.output != Output::NewLayerWithMask {
            self.flash("Decontaminate Colors outputs to a new layer with a layer mask");
        }
        self.spawn_job("Select and Mask", cx, move |_| {
            let r = reuse.unwrap_or_else(|| Arc::new(refine::refine(&base, &img, &p)));
            Box::new(move |ed: &mut Editor, _| ed.install_select_mask(&p, id, &r))
        });
    }

    pub fn cancel_select_mask(&mut self) {
        if self.select_mask.take().is_some() {
            self.doc.mark_all_dirty();
        }
    }

    fn install_select_mask(&mut self, p: &Params, id: u64, r: &Refined) {
        let output = if p.decon() { Output::NewLayerWithMask } else { p.output };
        let li = self.doc.layers.iter().position(|l| l.id == id).unwrap_or(self.doc.active);
        self.doc.push_undo("Select and Mask");
        match output {
            Output::Selection => self.doc.selection = r.mask.clone().to_selection(),
            Output::LayerMask => {
                self.doc.layers[li].mask = Some(LayerMask::new(r.mask.clone()));
                self.doc.layers[li].version += 1;
                self.doc.selection = None;
            }
            Output::NewLayerWithMask => {
                let grid = r.colors.clone().unwrap_or_else(|| self.doc.layers[li].grid.clone());
                let name = format!("{} copy", self.doc.layers[li].name);
                self.doc.active = li;
                let at = self.doc.insert_layer(name, grid);
                self.doc.layers[at].mask = Some(LayerMask::new(r.mask.clone()));
                self.doc.selection = None;
            }
        }
        self.doc.mark_all_dirty();
    }
}
