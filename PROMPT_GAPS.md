# Prompt: close every "Not implemented / differs from Photoshop" gap in PixelForge

You are working in `~/Desktop/Code/Rust/pixelforge` (Rust, GPUI via `gpui-kit` 0.7.1; not a git
repo, so back up the tree to the scratchpad before you start). PixelForge already has every tool
of Photoshop 2026's toolbar. README.md has a section called **"Not implemented / differs from
Photoshop"**. Your job is to make that section empty, or, for an item that truly can't be closed,
reduce it to one honest sentence that explains why. Photoshop 2026 is installed and scriptable
(`tools/photoshop_tools.sh`, `tools/photoshop_tools.jsx`). Use it as the reference for behaviour,
quality and speed.

## Ground rules

- **Reuse the architecture; don't fork it.** That means:
  - 256² COW tiles (`TileGrid`), mips via `layer_pixels_changed` / `set_layer_grid`, and `push_undo` before mutating.
  - The dab engine (`brush::Kernel`, `Stroke`) and the layer model (`LayerKind`, `LayerMask`, `clip`, `parent`).
  - `xform::canvas_transformed` / `layer_transformed`, `annot::Annotations`, `vector::Path` + the rasterizer + the stroker, and `text::TextLayer`.
  - `poisson::Solver`, `patchmatch::complete`, `segment::*`, `spawn_job` for anything over 30 ms, and the `.pforge` v2 format with its `PFDX` extras chunk.
- **Format compatibility:** every new piece of document state is saved in `.pforge` and is undoable. Old files must still open: add new chunks, never change existing records.
- **Quality bar:** the build stays warning-free and `cargo test --release` stays green after every phase.
- **Speed bar:** each new operation gets a `--bench-tools` row with a budget. It must also beat Photoshop wherever Photoshop has a scripting hook.
- **Proof bar:** never claim a feature works without a screenshot from a `PF_DEMO=tools` phase. Use `tools/demo_shots.sh tools <dir> <phases>`. Synthetic OS input is blocked, so drive the editor's handlers directly, as the existing demo does.
- **Downloads need approval:** don't download models, fonts or other third-party files without asking me first. Say what you need, where it comes from, its size and its licence.

## Phase A: Selections and retouching quality

1. **Neural Select Subject / Object Selection.**
   - Add an optional ML backend behind a cargo feature `ml` (ONNX Runtime via the `ort` crate, CoreML execution provider on macOS).
   - Models live in `models/` and are found at runtime. Without them, fall back to the current GrabCut path and say so in the UI.
   - Candidates: a salient-object or matting model such as BiRefNet, U²-Net or MODNet for Select Subject, and SAM / MobileSAM / EfficientSAM for Object Selection prompts (box, lasso, points). Ask me before fetching any of them.
   - Run inference at ≤ 1024 px, then upsample with the existing guided filter on boundary tiles only.
   - Add the hover **object finder**: precompute an object mask set when the tool is activated (in the background), and highlight the object under the cursor; a click selects it.
   - Budgets: hover highlight ≤ 16 ms after precompute; Select Subject ≤ 800 ms at 268 MP (Photoshop measured 2147 ms).
   - Quality: IoU against a hand-made ground-truth mask on 3 test images (one with hair). It must match or beat GrabCut, and be within 0.03 of Photoshop's result. Export Photoshop's mask through jsx.
2. **Healing exactness.**
   - Add a "Quality: Fast / Exact" option. Exact solves the full-resolution system to a 0.05 residual.
   - Fast must stay ≤ 8 ms per 500 px dab. Exact must stay ≤ 16 ms per dab.
   - Add a test that Exact matches a dense direct solve to within 0.1 of a level.
3. **Content-Aware Fill workspace** (Edit ▸ Content-Aware Fill).
   - Editable sampling area: auto, rectangular or custom, painted with a brush.
   - Color adaptation, plus rotation, scale and mirror adaptation (extend PatchMatch to search over a rotation/scale grid).
   - A live preview panel and output to the current, new or duplicate layer.
   - Compare against Photoshop's `contentAwareFill` on the 300 px and 1 MP cases: SSIM inside the hole within 0.05 of Photoshop's, and faster.
4. **Spot Healing "Create Texture".** Synthesize texture from the surrounding ring (patch-based texture synthesis seeded from the ring), then Poisson-blend it.
5. **Color Replacement "Limits".** Add Contiguous, Discontiguous and Find Edges. Contiguous does a per-dab flood within tolerance from the sampled point; Find Edges also stops at gradient edges. Budget: ≤ 2 ms per event at 500 px.

## Phase B: Type

1. **Per-character styling.** Change `TextLayer` to hold styled runs: font, weight, italic, size, colour, tracking, baseline shift, underline / strikethrough, faux bold / italic, all caps / small caps, super / subscript.
   - Selecting a range on canvas (Shift + arrows, drag, double-click word, triple-click line) applies the options bar to that range.
   - Add the Character and Paragraph panels: indents, space before / after, hyphenation on / off, and justification with last line left / centre / right / all.
2. **OpenType features panel:** ligatures, discretionary ligatures, old-style figures, swashes, stylistic sets and fractions, passed to cosmic-text / swash as font features.
3. **All 15 warp styles:** Arc, Arc Lower, Arc Upper, Arch, Bulge, Shell Lower, Shell Upper, Flag, Wave, Fish, Rise, Fisheye, Inflate, Squeeze, Twist. Each has Bend plus Horizontal and Vertical Distortion. Match Photoshop's shapes; compare outlines with jsx `warpStyle` exports.
4. **Type on a path.** Click a path with a Type tool to set text along it: start, end and flip handles; on both open and closed paths, and inside a closed path (area type).
5. **Vertical type:** Standard Vertical Roman Alignment (rotate Latin 90°) and tate-chu-yoko.
6. **Budgets:**
   - Typing stays ≤ 4 ms per keystroke for a 2000-glyph paragraph with 50 style runs.
   - Applying a style to a range ≤ 8 ms.
   - Photoshop comparison: create a layer with runs and edit one run, timed in jsx.

## Phase C: Vectors, shapes, frames

1. **Path operations:** Combine, Subtract Front, Intersect, Exclude, and Merge Shape Components.
   - Implement robust polygon booleans on flattened paths (Vatti or Martinez-Rueda, or an existing crate if its licence fits; ask first), then fit curves back (`vector::fit_curve`).
   - The operation is stored per subpath, as in Photoshop, so the result stays live until merged.
   - Tests against rasterized reference masks: area error < 0.1 %.
2. **Live shape properties.**
   - Corner-radius widgets on canvas, per corner or linked. Polygon sides and star controls on canvas.
   - A Properties panel with W, H, X, Y, rotation, fill, stroke, stroke options and radii.
   - Rotation of a parametric shape keeps it parametric: store a transform.
3. **Shape fill and stroke types:** solid, gradient (reuse `gradient.rs`), pattern (reuse `retouch::patterns`) and none, for both fill and stroke. Stroke can also be dashed with custom dash and gap values, and offset relative to the path.
4. **Dedicated Frame layer kind:** `LayerKind::Frame { geom, content: Option<SmartObject> }`, shown as one row in the Layers panel.
   - The content can be moved and scaled inside the frame.
   - Drag an image file onto a frame to fill it; text or shape layers can be converted to frames.
   - Saved in `.pforge` as a new kind id.
5. **Budget:** path booleans ≤ 8 ms for two 500-anchor paths, so shapes stay live while dragging.

## Phase D: Layers, Move, Artboards, Slices

1. **Multi-layer selection** in the Layers panel: ⌘-click, Shift-click, and ⌘A to select all layers.
   - Move, Free Transform, Align, delete, group and link all act on the selection.
   - Add **Distribute** (left, centre, right, top, middle, bottom, horizontal and vertical spacing) and **Align to key object**.
   - Auto-Select gets a **Layer / Group** option.
2. **Background layer semantics:** a locked Background layer (as in Photoshop), plus Layer from Background and Background from Layer.
   - Moving or deleting pixels on Background fills with the background colour.
   - Canvas extension fills Background with the background colour.
3. **Real artboards:** `LayerKind::Artboard { rect, background }` is a group that clips and owns its children.
   - Moving an artboard moves its layers.
   - Layers dragged across artboards change parent.
   - New-artboard presets, plus Export ▸ Artboards to Files (PNG / JPEG / PDF).
4. **Slices:**
   - Slice options (name, URL, target, alt text, dimensions).
   - Layer-based slices and Divide Slice (n × m).
   - A Save for Web-style export: per-slice format PNG-8 / PNG-24 / JPEG quality / WebP, an optimized-size preview, and an HTML table output.

## Phase E: Crop, Straighten, Rotate View, Zoom

1. **Crop:**
   - "Delete Cropped Pixels" off keeps hidden pixels: layers get an offset and extent larger than the canvas, so a later crop can bring them back.
   - **Content-Aware** crop fill: when the frame extends past the canvas, fill the new area with PatchMatch.
   - Rotated crops use bicubic (or Lanczos-3) resampling from level 0, with a mip prefilter only when downscaling.
   - Overlay options: Golden Ratio, Diagonal, Triangle, Golden Spiral and Grid.
2. **Straighten:** a Straighten tool inside Crop (drag a line to rotate the crop frame). Straighten Layer on a single non-Background layer rotates only that layer, as in Photoshop.
3. **Rotate View on the GPU.**
   - Target: ≤ 2 ms per rotated frame at 4K and a crisp image (no CPU resample), with handles drawn rotated.
   - First check whether `gpui-pre` can render a transformed image: a `PolychromeSprite` transformation, a custom `paint_surface` path, or textured `paint_path`.
   - If it can't, vendor and patch `gpui-pre` with a transformed image primitive and document the patch.
4. **Multiple documents:** open several documents as tabs (Window ▸ Arrange: tabs, 2-up, tile). Add Zoom All Windows, Pan All Windows, and Match Zoom and Location.

## Phase F: Measurement, Info, Notes, Brush presets

1. **Ruler protractor:** Alt-drag from an endpoint makes a second leg, and the angle readout is between the legs. **Measurement Log** panel: Record Measurements from the Ruler, Count, selection area, perimeter and centroid; export CSV; set measurement scale (px → units).
2. **Notes panel:** list, edit and delete notes, jump to a note, and an editable author name in preferences. **Count:** label font size, group visibility toggle, and count export to the Measurement Log.
3. **Info panel colour modes:** RGB, HSB, Lab (D50, from sRGB with exact ICC-free maths) and CMYK (a documented naive or SWOP-like conversion; say which). Show them for the cursor and for each color sampler; each sampler gets its own mode. Add a document-size and timing readout.
4. **Brush Presets panel** (Window ▸ Brushes):
   - Folders of presets with thumbnails of the tip and stroke preview.
   - New Preset from the current settings, and rename / delete / reorder.
   - Import `.abr` sampled tips (parse version 6+ sampled brushes; skip unsupported computed ones and report how many).
   - Presets persist in the user config directory.
   - **Brush Settings panel:** shape dynamics (size / angle / roundness jitter with control: off, pressure, tilt, fade), scattering (count, count jitter, both axes), texture (pattern, scale, depth, mode), dual brush, color dynamics (FG/BG, hue, saturation and brightness jitter, per tip), transfer (opacity / flow jitter), wet edges, build-up, smoothing, protect texture.
   - Every dynamic is deterministic per stroke seed, so tests can compare against reference renders.
   - Budget: ≤ 2 ms per event at 500 px with all dynamics on.

## Phase G: Filters gaps

1. **Neural Filters (ML, `ml` feature):** Colorize, Style Transfer, Smart Portrait (at least age / expression / gaze sliders, if a suitable model exists), Makeup Transfer, Landscape Mixer and Depth Blur.
   - Depth Blur uses a monocular depth model (Depth Anything V2 or MiDaS) plus the existing Lens Blur with a depth map.
   - Each runs at preview resolution live, then full resolution in `spawn_job`.
   - Same rule as Phase A: list the models, sizes and licences, and wait for my approval before downloading anything.
2. **Interactive Vanishing Point:**
   - Define planes, with tear-off perpendicular planes.
   - Marquee and Stamp on the plane in perspective, with healing on/off.
   - Brush on the plane, and Transform pasted content onto the plane.
   - Use the homographies already in `filters::distort`.
3. **Filter Gallery fidelity.** Exact pixels are impossible because Adobe doesn't publish its algorithms, so measure instead.
   - Render each of the 47 gallery effects in Photoshop via jsx at default settings on a fixed test image. Compute SSIM and mean ΔE2000 against PixelForge's.
   - Tune parameters until SSIM ≥ 0.85 for every effect. Publish the per-effect table in the README.
   - Effects that can't reach 0.85 get a one-line explanation.

## Verification per phase (same as before)

1. Unit tests against naive or reference implementations for every new algorithm. Examples: polygon booleans vs rasterized masks, styled-text layout snapshots, Lab conversion vs reference values, ABR parsing vs a known file, Exact healing vs a direct solve.
2. `--bench-tools` sections with budgets for every new interactive or heavy operation; all budgets met.
3. A Photoshop comparison through `tools/photoshop_tools.jsx`, adding a phase block. Put an honest table in the README, and say explicitly where Photoshop has no scripting hook. For ML features, also compare quality (IoU / SSIM) against Photoshop's output, not only speed.
4. A `PF_DEMO=tools` phase with screenshots of each feature in use. Look at every screenshot and fix what's wrong before moving on.
5. README updates: features, shortcuts, benchmark tables, and the "Not implemented / differs" section rewritten to what's actually left.

## Reporting

After each phase, report:
- what was built;
- benchmark numbers against their budgets;
- Photoshop comparison numbers, or why there are none;
- which screenshots you checked and what they showed;
- anything deferred, with the reason.

At the end, the README's "Not implemented / differs" section should contain only items blocked by things outside the code: models I declined to download, Adobe-proprietary algorithms, or platform limits. Each one gets a sentence explaining why.
