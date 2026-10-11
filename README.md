# PixelForge

A Photoshop-style raster editor in Rust on [GPUI](https://gpui-kit.com) + GPUI Component
(via the `gpui-kit` crate), built to stay interactive on images with hundreds of megapixels.

```bash
cargo run --release                     # the editor
cargo run --release -- --bench          # headless engine benchmark (16384² = 268 MP)
cargo run --release -- --bench 32768    # 1.07 GP
PF_DEMO=1 cargo run --release           # scripted tour driving the real input handlers
PF_DEMO=filters cargo run --release     # live-preview + commit tour of several filters
PF_DEMO=tools PF_DEMO_PHASE=0,1,2,3,4,5 cargo run --release   # toolbar tour, phase by phase
cargo test --release                    # engine, codec and CPU/GPU correctness tests
cargo run --release -- --bench-filters  # every filter at default settings, 67 MP
cargo run --release -- --bench-tools    # every toolbar tool vs its latency budget, 268 MP
cargo run --release -- --bench-tools 16384 heal    # one section (brush, select, wand, lasso, quick,
                                        #   object, layers, retouch, paint, vector, type, nav)
cargo run --release -- --export-scene 16384 scene.tif   # the benchmark scene, for Photoshop
tools/photoshop_tools.sh scene.tif 1,2,3,4,5            # the same operations timed in Photoshop 2026
tools/photoshop_gaps.sh ps_out A                        # Photoshop's outputs + timings for the gap phases (A–G)
cargo run --release -- --compare-gaps ps_out A          # IoU / SSIM / ΔE00 and speed vs those outputs
tools/demo_shots.sh tools shots/ 4                      # run the tour and screenshot every step
tools/demo_shots.sh tools shots/ A                      # the gap-closing tour (phases A–G)
cargo run --release -- --bench-gpu 16384               # every GPU kernel vs the CPU at 268 MP (see "GPU acceleration")
cargo build --release --features ml                    # optional ONNX Runtime backend (see "ML backend")
```

## Features

- **Toolbar:** every tool of Photoshop 2026's toolbar, grouped with flyouts (click and hold,
  or right-click). The same key selects a group and Shift + key cycles through it.
  Edit Toolbar (the `…` slot) hides or shows groups. The screen-mode button at the bottom of
  the toolbar (or F) cycles Standard / Full Screen with Menus / Full Screen. Details are in [The toolbar](#the-toolbar).
- **Layer model:** pixel, group (pass-through or isolated), adjustment (any point filter,
  re-editable), smart object (placed image plus matrix), shape and text layers. Each layer
  can have a layer mask (painted with any paint tool, with its own mip pyramid) and a
  clipping mask. All of them composite, display at any mip level, undo, and save in `.pforge`.
- **Layers:** new, duplicate, delete, reorder, visibility, opacity, 27 blend modes (including Dissolve, Linear Burn,
  Vivid/Linear/Pin Light, Hard Mix, Subtract and Divide), merge down, flatten, layer via copy / cut, group / ungroup (⌘G / ⇧⌘G),
  clipping mask (⌥⌘G), add / apply / delete layer masks, rasterize, Place Embedded, and live
  thumbnails.
- **Selections:** every selection tool supports New / Add (Shift) / Subtract (Alt) / Intersect,
  feather and anti-aliasing. Select ▸ Modify offers Expand, Contract, Feather, Border and
  Smooth. Quick Mask (Q) lets any paint tool edit the selection. Select Subject is included.
  Selections have anti-aliased edges and marching ants, and every paint, fill and filter
  respects them.
- **Content-Aware Fill workspace** (Edit ▸ Content-Aware Fill): the sampling area is shown in
  green and painted with a brush (Alt subtracts; `[` / `]` resize). It can be Auto, Rectangular
  or Custom. Settings: Color Adaptation (None / Default / High / Very High), Rotation Adaptation
  (None / Low ±15° / Medium ±45° / High ±90° / Full), Scale (0.8–1.25×) and Mirror. A live
  preview shows on the canvas and as a panel thumbnail. Output goes to the current layer, a new
  layer or a duplicate layer.
- **Object finder:** with Object Selection active, objects are found in the background. Hovering
  tints the object under the cursor; a click selects it.
- **Paths panel:** Work Path and saved paths, Fill, Stroke (with the Brush), Make Selection,
  Make Work Path from Selection, and Delete.
- **Info panel:** cursor, up to 10 Color Sampler readouts (sample size 1×1…101×101), each in
  its own mode (RGB, HSB, Lab D50, CMYK), saved with the document. CMYK follows Photoshop's
  default U.S. Web Coated (SWOP) v2, from a table of Photoshop's readouts. Also shows the
  Ruler / protractor, Count total, document size and the last operation's time.
- **GPU acceleration** (wgpu): filters, Blur Gallery, Lens Blur, Camera Raw, Filter
  Gallery, resampling, Liquify, Perspective Warp, Select and Mask, retouching and the layer
  compositor have GPU kernels that reproduce the CPU's pixels. A calibrated cost model picks
  the faster path per operation, and any GPU failure falls back to the CPU. See
  [GPU acceleration](#gpu-acceleration).
- **Measurement Log** (Image ▸ Analysis, ⇧⌘M): records the Ruler or protractor, each Count
  group, and the selection (area, perimeter, circularity, bounds, centroid, Gray Value, regions).
  Values use a measurement scale (px → units), are saved with the document and export as CSV
  with Photoshop's column names.
- **Panels dock** (Window ▸ Brushes / Brush Settings (F5) / Measurement Log / Notes):
  - **Brush presets:** folders, thumbnails of the tip and a stroke preview, New / Rename /
    Delete / reorder, and `.abr` import of sampled tips. Presets are kept in the user config
    directory.
  - **Brush Settings:** every dynamic (see the Brush row in the toolbar table).
  - **Notes:** list, edit, jump and delete; the author is a preference.
- **Adjustments (18):** Brightness/Contrast, Levels, Curves, Exposure, Vibrance,
  Hue/Saturation, Color Balance, Black & White, Photo Filter, Gradient Map, Invert,
  Desaturate, Posterize, Threshold, Equalize, Shadows/Highlights, Auto Tone, Auto Contrast.
- **Filters (54), every one with a live preview:**
  - **Blur:** Average, Blur, Blur More, Box, Gaussian, Lens, Motion, Radial, Smart, Surface.
  - **Distort:** Pinch, Polar Coordinates, Ripple, Shear, Spherize, Twirl, Wave, ZigZag.
  - **Noise:** Add Noise, Despeckle, Dust & Scratches, Median, Reduce Noise.
  - **Pixelate:** Color Halftone, Crystallize, Facet, Fragment, Mezzotint, Mosaic, Pointillize.
  - **Render:** Clouds, Difference Clouds, Fibers, Lens Flare.
  - **Sharpen:** Sharpen, Sharpen Edges, Sharpen More, Smart Sharpen, Unsharp Mask.
  - **Stylize:** Diffuse, Emboss, Find Edges, Oil Paint, Solarize, Trace Contour, Wind.
  - **Other:** High Pass, HSB/HSL, Maximum, Minimum, Offset, Custom (3×3 kernel).
  - **Blur Gallery:** Tilt-Shift, Iris Blur (variable-radius blur from a Gaussian stack).
  - **Video:** De-Interlace, NTSC Colors.
  - **Also:** Lighting Effects, Displace, Shape Blur (6 kernel shapes), Extrude, Tiles,
    Camera Raw Filter (13 develop sliders) and Lens Correction (distortion, fringing, vignette).
- **Filter Gallery (47):**
  - **Artistic (15):** Colored Pencil, Cutout, Dry Brush, Film Grain, Fresco, Neon Glow, Paint Daubs,
    Palette Knife, Plastic Wrap, Poster Edges, Rough Pastels, Smudge Stick, Sponge, Underpainting,
    Watercolor.
  - **Brush Strokes (8):** Accented Edges, Angled Strokes, Crosshatch, Dark Strokes, Ink Outlines,
    Spatter, Sprayed Strokes, Sumi-e.
  - **Distort (3):** Diffuse Glow, Glass, Ocean Ripple.
  - **Sketch (14):** Bas Relief, Chalk & Charcoal, Charcoal, Chrome, Conté Crayon, Graphic Pen,
    Halftone Pattern, Note Paper, Photocopy, Plaster, Reticulation, Stamp, Torn Edges, Water Paper.
  - **Stylize (1):** Glowing Edges.
  - **Texture (6):** Craquelure, Grain, Mosaic Tiles, Patchwork, Stained Glass, Texturizer.
- **More adjustments:** Channel Mixer, Selective Color, Replace Color, Match Color (from the layer below).
- **Liquify** (⇧⌘X): Forward Warp, Reconstruct, Smooth, Twirl ↻/↺, Pucker, Bloat, Push Left,
  Freeze Mask (drawn in red) and Thaw Mask, with Size, Pressure and Rate. A session keeps the
  original layer and one displacement field, so Reconstruct can undo earlier warps. Twirl,
  Pucker, Bloat, Reconstruct and Smooth keep acting while the button is held.
- **Camera Raw Filter** (⇧⌘A, 57 controls):
  - Basic: white balance, exposure, contrast, highlights, shadows, whites, blacks, texture,
    clarity, dehaze, vibrance, saturation.
  - Parametric tone curve.
  - 8-band HSL.
  - Color grading: shadows and highlights, with balance.
  - Detail: sharpening and noise reduction.
  - Effects: vignette and grain.
  - Masks: linear and radial gradient exposure.
- **Blur Gallery:** Field (3 pins), Iris, Tilt-Shift, Path (curved, mip-prefiltered) and Spin
  (elliptical).
- **Edit ▸ Transform ▸ Free Transform** (⌘T): move, scale, rotate, skew and perspective as one
  homography, live-previewed.
- **Vanishing Point** (⌥⌘V), an interactive workspace on the active layer:
  - **Planes:** Create Plane (click four corners) and Edit Plane (drag corners). ⌘-dragging an
    edge's middle handle tears off a perpendicular plane, using the focal length estimated from
    the first plane.
  - **Editing in perspective:** Marquee (⌥-drag copies, with Heal on or off), Stamp (⌥-click
    the source), and Brush (circles on the plane).
  - **Transform:** pastes the clipboard onto a plane, where it can be moved and scaled.
  - OK applies everything as one undo step, and the planes are saved with the document.
- **Adaptive Wide Angle** (⌥⇧⌘A): fisheye correction.
- **Lighting Effects:** point, spot and infinite lights with a bump map.
- **Displace:** maps from the layer, from clouds, or from any image file, stretched or tiled.
- **Neural Filters (classical, no ML):** Skin Smoothing, JPEG Artifacts Removal, Photo
  Restoration, Super Zoom (bicubic upscale of the whole document plus detail sharpening) and
  Harmonization.
- **Compressed history:** steps older than the newest two are LZ4-compressed on a background
  thread. Only tiles that nothing live still references are compressed, and duplicates are
  stored once. A budget (`PF_HISTORY_MB`, default 6144) drops the oldest steps when exceeded.
  Undo and redo stay bit-exact.
- **Image operations:** Image Size (resampling), Crop / Crop to Selection, Reveal All, Rotate
  90° CW and CCW, Flip Horizontal / Vertical. Every canvas operation moves every layer kind:
  shapes stay vector, smart objects compose their matrix, and masks, hidden pixels, paths and
  annotations follow. Flips and right-angle turns stay pixel-exact.
- **Documents:** any number of open documents as tabs (⌃Tab / ⌃⇧Tab, ⌘W). Window ▸ Arrange
  shows 2-up or tiles them all. Match Zoom / Location / Rotation / All and Zoom All / Scroll
  All keep the views together. Background jobs finish on the document they started on.
- **Edit:** Undo/redo with a clickable History panel, Cut / Copy / Paste as layer, Fill
  with foreground or background, Clear.
- **Files:** open PNG, JPEG, TIFF, WebP and BMP (by dialog or drag and drop). Export
  PNG/JPEG/TIFF/WebP/BMP. The native layered `.pforge` format (v2) compresses tiles with
  parallel LZ4. It stores the full layer model plus paths, color samplers, notes, counts,
  artboards and slices. Export Artboards / Export Slices write one PNG per region.

## PSD / PSB status (Phase 2 in progress)

The in-repository codec reads and writes 8-bit PSD/PSB blocks, PackBits and ZIP
channels (including prediction), descriptors and text EngineData. Original opaque
blocks, resources, channel streams and descriptor keys remain in the backing
file tree. PSD/PSB Open and Save are connected to background jobs for the supported
RGB8 layer subset. Window ▸ Channels lists RGB and preserved extra channels; Info
shows the document mode, depth and file format.

**Phase 2 is incomplete.** Editable mapping currently covers RGB pixel layers,
ordinary groups, clipping, opacity/fill, basic raster masks, 27 blend mode keys,
Blend If ranges and RGB channel restrictions. Photoshop fidelity is a separate
acceptance gate; the presence of a mapping is not a compatibility guarantee.
Imported text displays Photoshop's stored pixels and retains its descriptors and
EngineData. Text edits with a document-level `Txt2` cache cannot yet be saved safely.
Live adjustment/fill/style/vector/smart-object mappings, complete text editing,
non-RGB conversion and several metadata/UI features remain unfinished. Unsupported
operations return an error rather than discarding or rasterizing editable content.
Preserved PSDs also reject stack/canvas changes that could invalidate opaque
references. See the [Phase 2 evidence and failures](docs/benchmarks/2026-10-11-phase-2/REPORT.md).

`.pforge` remains the native format because the full native document model does
not yet have safe PSD representations. The migration reader is retained:

```bash
./target/release/pixelforge --convert-pforge input.pforge output.psd
./target/release/pixelforge --convert-pforge input.pforge output.psb
./target/release/pixelforge --psd-check input.psd roundtrip.psd report.json
./target/release/pixelforge --psd-corpus locally-generated-corpus
```

Migration refuses unsupported native features before replacing the destination.
`--psd-check` distinguishes editable document-model checks from raw preservation
checks; raw preservation alone does not pass the corpus. The corpus runner uses
locally authored Photoshop fixtures, flattened and per-layer exports, structural
checks, actual PixelForge renders, native Photoshop reopening and an independent
dialog watchdog. Missing evidence and unsupported cases fail acceptance.

PixelForge-only active/selected layer state uses image resource **4000** with
13-byte signature `PixelForge\0\x00\x01`, followed by big-endian active-layer `u64`,
selected count `u32` and that many layer-ID `u64` values. A foreign resource 4000
without this signature is preserved independently. This resource does not replace
any Photoshop feature. Photoshop does not interpret this selection state; its
retention on re-save must be checked against the installed version. Loss of this
private state must never imply loss of document pixels or editable features.

## The toolbar

| Group (key) | Tools | What they do |
|---|---|---|
| Move (V) | Move, Artboard | Move drags every selected layer (⌘-click / Shift-click in Layers, Select ▸ All Layers ⌥⌘A). Groups and artboards move with their contents, and vector, text and frame layers stay live. **Auto-Select** (or ⌘-click) picks the **Layer** or its outermost **Group**. Show Transform Controls has 8 handles, proportional by default; Shift frees, Alt scales from the centre. **Align** (left / centre / right / top / middle / bottom) works on the selected layers' bounds, a marquee, the canvas, or **To Key Object** (the active layer stays). **Distribute** handles left, centre, right, top, middle, bottom and equal horizontal / vertical spacing for three or more layers. Dragging inside a selection floats the selected pixels. **Artboard** creates artboard layers: top-level containers with a white background that clip their layers. Draw them, use the 12 presets (iPhone, iPad, Android, Web, MacBook, Watch, Instagram, A4 / Letter), or move and resize them; the canvas grows to hold them. They export as PNG, JPEG or a multi-page PDF, with only each artboard's own layers. |
| Marquee (M) | Rectangular, Elliptical, Single Row, Single Column | Shift / Alt / Shift+Alt boolean modes, feather and anti-alias. |
| Lasso (L) | Lasso, Polygonal, Magnetic | Magnetic runs live-wire Dijkstra in a band around the cursor, coarse level first and then refined, with auto anchors, width, contrast and frequency. |
| Object (W) | Object Selection, Quick Selection, Magic Wand | Object Selection runs GrabCut (GMM, max-flow) from a rectangle or lasso, coarse-to-fine, with a guided-filter edge. With a salient-object model installed (see [ML backend](#ml-backend-optional)) it runs the model on the box instead. Its **Object Finder** hover tint is precomputed in the background (subject components plus merged Felzenszwalb regions); hovering is a label lookup and a click selects the object. Select ▸ Subject uses a boundary-connectivity background prior plus GrabCut, or the model when present. Quick Selection is a graph cut that grows with the stroke. Magic Wand uses tolerance, contiguous and Sample All Layers. All three do a tile-parallel flood. |
| Crop (C) | Crop, Perspective Crop, Slice, Slice Select | Crop has 8 handles, move, rotate (drag outside, Shift snaps 15°), ratio presets and a size readout. Overlays: Thirds, Grid, Diagonal, Triangle, Golden Ratio and Golden Spiral (O cycles, Shift+O turns them). **Straighten** (or ⌘-drag) levels a drawn line by turning the frame. **Delete Cropped Pixels** off keeps what falls outside as hidden pixels, and **Content-Aware** fills canvas area the crop adds. Rotated crops resample with Bilinear, Bicubic or Lanczos-3. Perspective Crop takes 4 corners with a perspective grid and outputs a homography-rectified image. Slice draws user slices, auto slices fill the rest, and they are numbered like Photoshop's. Slice Select moves, resizes and deletes them. The **Slices panel** has Slice Options (name, URL, target, alt text), **New Layer-based Slice** (the slice follows the layer's bounds) and **Divide Slice** (columns × rows). **Save for Web** offers PNG-8 (2–256 colours, dither, transparency), PNG-24, JPEG (quality) or WebP, with a live optimized preview, size and encode time. It writes every slice plus an HTML table page with the links and alt text. |
| Frame (K) | Frame | **Frame layers**: a rectangular or elliptical frame (placeholder cross while empty) holding a placed image clipped to it. File ▸ Place into the active frame, or drop an image file onto a frame, fits the image to cover it. The Frame panel moves and scales the content, with Fit / Fill / Clear. Layer ▸ Convert to Frame turns a shape layer, or a text layer's glyphs, into a frame. Frames rotate and scale parametrically and save as their own layer kind. |
| Eyedropper (I) | Eyedropper, Color Sampler, Ruler, Note, Count | Eyedropper takes a sample size (point…101×101) from all layers or the current one, and live-drags. Color Sampler places ≤ 10 samplers (drag to move, ⌥-click deletes) that show in Info. Ruler measures W/H/angle/length (Shift snaps 45°); ⌥-drag from an end point adds the **protractor** leg, and the readout becomes the angle between the legs with L1 and L2. Straighten Layer rotates and crops to the largest same-aspect rectangle; ⌥ keeps the whole image. Note adds notes with text, moving and deleting, signed with the author preference. Count places numbered markers in coloured groups with marker and label size and per-group visibility. Record sends counts or the ruler to the Measurement Log. |
| Healing (J) | Spot Healing, Healing, Patch, Content-Aware Move, Red Eye | Spot Healing offers Content-Aware (multi-scale PatchMatch, then Poisson blend), Create Texture (image quilting seeded from the ring around the spot, then Poisson blend) and Proximity Match. Healing is a Poisson (multigrid) blend of an Alt-clicked source per dab, aligned or not, with a sample layer option and **Quality: Fast / Exact**. Exact solves at full resolution to a max Laplacian residual of 0.05; a test checks it against a direct banded-Cholesky solve. Patch works in Source, Destination or Content-Aware mode. Content-Aware Move has Move and Extend modes. Red Eye detects the pupil and darkens it. |
| Brush (B) | Brush, Pencil, Color Replacement, Mixer Brush | One dab engine, with computed round or sampled (`.abr`) tips and the full Brush Settings: Shape Dynamics, Scattering, Texture, Dual Brush, Color Dynamics, Transfer, Noise, Wet Edges, Build-up, Smoothing and Protect Texture. Details: Shape Dynamics covers size, angle and roundness jitter with Off / Pressure / Tilt / Fade / Direction controls, minimums and flip jitter. Scattering has both axes, count and count jitter. Texture has 10 modes, scale, brightness, contrast, invert, and per-tip depth with jitter. Color Dynamics varies FG/BG, hue, saturation, brightness and purity, per tip or per stroke. Transfer jitters opacity and flow. Smoothing is a lazy mouse plus Catmull-Rom. Each stroke's randomness comes from its seed, so a seeded stroke always renders the same pixels. Pencil paints aliased. Color Replacement has Hue / Saturation / Color / Luminosity modes, sampling continuous / once / background, tolerance and **Limits**. Discontiguous changes every match under the tip. Contiguous does a per-dab scanline flood from the hotspot. Find Edges also stops at gradient edges inside the tolerance. Background Eraser shares the same Limits. Mixer Brush has wet / load / mix / flow and load-after-stroke. |
| Stamp (S) | Clone Stamp, Pattern Stamp | Clone Stamp has a live source overlay, aligned or not, and current / below / all layers. Pattern Stamp uses 5 procedural patterns, aligned. |
| History (Y) | History Brush, Art History Brush | Both paint from any history state (◉ column in History), read lazily from the compressed history. Art History has 4 stroke styles and an area setting. |
| Eraser (E) | Eraser, Background Eraser, Magic Eraser | Background Eraser has sampling, tolerance and protect foreground. Magic Eraser offers tolerance, contiguous, Sample All Layers and anti-alias. |
| Gradient (G) | Gradient, Paint Bucket | Gradient has Linear / Radial / Angle / Reflected / Diamond, a stop editor (colour and opacity stops), presets, reverse and dither. The gradient stays live with draggable handles until Enter. Paint Bucket fills with the foreground colour or a pattern, with tolerance, contiguous and all layers. |
| Focus | Blur, Sharpen, Smudge | Strength. Smudge has finger painting. |
| Toning (O) | Dodge, Burn, Sponge | Dodge and Burn take shadows / midtones / highlights, exposure and protect tones. Sponge saturates or desaturates, with vibrance. |
| Pen (P) | Pen, Freeform Pen, Content-Aware Tracing, Curvature Pen, Add / Delete Anchor, Convert Point | Pen clicks for corners and drags for smooth points, with a rubber-band preview. Freeform Pen fits curves (Schneider). Curvature Pen produces Catmull-Rom smooth anchors, and double-click makes a corner. Content-Aware Tracing follows the edge under the cursor and click adds it. Each pen works in Path or Shape mode. |
| Type (T) | Horizontal / Vertical Type, Horizontal / Vertical Type Mask | Point, paragraph (box), **type on a path** (click the active path; drag the start / end markers along it, or the start marker across it to flip) and **area type** (click inside a closed path). **Per-character style runs**: font, weight, italic, size, leading, tracking, colour, baseline shift, underline / strikethrough, faux bold / italic, all caps / small caps, super / subscript. On-canvas **range selection**: drag, Shift + arrows, ⌥ word jumps, double-click word, triple-click line, ⌘A. The options bar and the **Type panel** (Character / Paragraph / OpenType / Warp / Path & Vertical tabs) apply to the selection, to the typing style at a bare caret, or to the whole layer. Paragraphs: alignment with last-line Left / Center / Right / Justify All, left / right / first-line indents, space before / after, TeX-pattern hyphenation, and the Every-line or Single-line Composer. OpenType: ligatures, discretionary ligatures, old-style figures, swashes, fractions, stylistic sets 1–8. All **15 warp styles** with Bend and H / V Distortion, horizontal or vertical. Vertical type rotates Roman text (or keeps it upright with Standard Vertical Roman Alignment) and supports tate-chu-yoko. Type Mask commits as a selection. |
| Path Selection (A) | Path Selection, Direct Selection | Path Selection selects and moves a subpath. Direct Selection drags anchors and handles; smooth handles stay mirrored, and Alt breaks them. |
| Shape (U) | Rectangle (corner radius), Ellipse, Triangle, Polygon (sides, star indent), Line (weight, arrowheads), Custom Shape (8 shapes) | Live, parametric shape layers. **Path operations** (options bar, or ⇧ / ⌥ / ⇧⌥ at mouse-down): Combine, Subtract Front, Intersect, Exclude. Each operation applies to one component and stays live until **Merge Shape Components**. On-canvas **corner-radius widgets** change all corners, or one with ⌥; Polygons get a **star-indent handle**. The **Properties** panel has W, H, X, Y and angle (rotation is stored, so a rotated rectangle stays a rectangle), plus per-corner or linked radii, sides and star ratio. Fill and stroke can each be none, solid, gradient (five kinds, angle, scale, reverse) or pattern (scale, angle). Strokes have width, inside / centre / outside alignment, caps, joins, solid / dashed / dotted / custom dashes and a dash offset. Shift constrains and Alt draws from the centre. Handles follow the rotation, and edits re-render only changed tiles. |
| Hand (H) | Hand, Rotate View (R) | Rotate View rotates the display only, with Shift snapping 15° and a double-click reset. The GPU draws the cached display tiles as rotated sprites. Painting, selections and every overlay keep working rotated. |
| Zoom (Z) | Zoom | Scrubby zoom (drag left or right), click / ⌥-click, a zoom-rectangle mode, 100 %, Fit and Fill Screen. |

## Why it stays fast at hundreds of megapixels

| Technique | Effect |
|---|---|
| 256×256 **copy-on-write tiles** (`Arc`) | Undo snapshots, background jobs and stroke bases are pointer copies. Each history step costs only the tiles it changed. |
| **Sparse + uniform tiles** | Transparent tiles don't exist, and flat tiles are shared. A blank 268 MP canvas is one 256 KB tile, and filling it with a color is still one tile. |
| **Per-layer mip pyramids**, updated incrementally from dirty rects | The viewer composites only the mip level that matches the zoom, so screen cost is bounded by screen pixels, not image pixels. |
| **Display tile cache** on the GPU (258² with a 1 px gutter for seamless bilinear) | Panning and zooming reuse textures. Only dirty or new tiles are composited, in parallel, under a ~26 ms budget, with coarser cached levels drawn as stand-ins. |
| **Nearest-neighbour magnified levels** (−1…−5) | Crisp pixels up to 3200 %. |
| **Live preview at display resolution** | Filter dialogs preview on the visible mip tiles, with the blur radius scaled per level. That's instant even at 268 MP. OK then commits at full resolution off the UI thread. |
| **Separable box-blur ×3**, strip-wise | Gaussian blur cost is independent of radius (r=20 and r=120 both take ~0.35 s at 268 MP). |
| **Hierarchical flood fill** | Uniform tiles are flooded whole and the per-pixel scanline runs only in mixed tiles. A magic wand on a blank 268 MP canvas takes 0.3 ms. |
| **SWAR premultiplied pixel math** | `over`, `lerp`, `scale` and mip averaging work on two channels per 16-bit lane. |
| **Brush: stroke-mask accumulation** | Dabs are bucketed per tile and processed in parallel, touching only each dab's bounding box. A LUT on squared distance avoids a `sqrt` per pixel. |
| Everything heavy runs on **rayon**, and long ops run on a **background executor** | The UI never blocks. A progress HUD shows while jobs run. |

### Photographic benchmark tiers and shared memory

The 24–61 MP rows are the everyday targets. The larger rows stress the same
8-bit engine; 16/32-bit requests are rejected until those document modes exist.
All four benchmark suites accept `--tier 24|61|150|268|all`, `WIDTHxHEIGHT`, or
an existing square side argument. Original defaults remain available.

| Tier | Dimensions | Actual megapixels |
|---|---|---:|
| 24 | 6000 × 4000 | 24.000 |
| 61 | 9504 × 6336 | 60.217 |
| 150 | 14204 × 10652 | 151.301 |
| 268 (stress) | 16384 × 16384 | 268.435 |

Release measurements on 2026-10-11, Apple M5 Pro, 48 GiB RAM, 18 Rayon workers,
`PF_GPU=off`. Each suite/tier runs in a fresh process, serially. Filter and
file rows below are individual engine-suite timings; brush rows are gesture
p95 values. These describe the procedural fixture, not all photographs.

| Operation | **24 MP** | **61 MP** | 150 MP | 268 MP stress |
|---|---:|---:|---:|---:|
| Hue/Saturation, full image | **16.9 ms** | **36.9 ms** | 82.1 ms | 140.8 ms |
| Gaussian blur r=20, full image | **17.3 ms** | **38.9 ms** | 89.7 ms | 144.4 ms |
| Live blur preview, 135 display tiles | **5.8 ms** | **5.9 ms** | 6.5 ms | 7.2 ms |
| 500 px soft brush, p95 per mouse event | **0.33 ms** | **0.36 ms** | 0.42 ms | 0.54 ms |
| Composite base + sparse paint layer | **2.1 ms** | **2.8 ms** | 4.5 ms | 6.7 ms |
| Native `.pforge` save | **14.4 ms** | **28.0 ms** | 55.3 ms | 87.0 ms |
| Native `.pforge` open | **6.4 ms** | **13.3 ms** | 23.0 ms | 39.2 ms |
| Engine suite peak RSS | **801 MiB** | **1446 MiB** | 2966 MiB | 5062 MiB |

The per-tier latency targets are checked in the engine output. Brush gestures
also retain their existing per-event checks in the tools suite.

| Target (ms) | 24 MP | 61 MP | 150 MP | 268 MP |
|---|---:|---:|---:|---:|
| Brush event | 2 | 2 | 2 | 2 |
| Filter preview | 33 | 33 | 50 | 66 |
| Hue/Saturation commit | 100 | 180 | 400 | 700 |
| Composite | 33 | 50 | 100 | 180 |
| Open / save, each | 1500 | 2500 | 6000 | 10000 |

**Memory and scratch.** One `MemoryBudget` charges each shared pixel allocation
once across layer pixels, undo/redo, smart-object sources, mips, and previews.
Coverage masks, CPU display images/atlas estimates, GPU buffers (including idle
pooled buffers), resident images, and GPU display textures share the allowance.
Settings exposes the RAM percentage (70% by default, adjustable from 10–90%)
and scratch folder. Info shows used/budget, compressed bytes, live scratch
records including their recovery copies, and storage errors.

The worker compresses hidden layers (including descendants of hidden groups),
older history, and tiles idle for 30 seconds. The idle threshold is configurable
with `cold_seconds` in preferences. Empty and uniform tiles stay shared.
Access transparently restores a tile; read/write guards pin it while in use.
Under pressure, stable snapshots of access times choose LRU tiles for spilling
to scratch until usage reaches a 90% low-water mark. Undo retains cached mips,
so it does not rebuild the entire document merely to restore a state. Memory
pressure no longer deletes history; the existing 80-step history count remains.

Scratch uses private files, checksums, a verified recovery copy, reused free
extents, and automatic file removal on process exit. Failed writes keep the
source pixels in RAM. A damaged primary copy can be restored from recovery;
if both copies fail verification, reads return an error rather than substitute
pixels (infallible pixel-access calls fail loudly on unrecoverable reads). Scratch is temporary
storage, not a crash-recovery document. Freed extents can remain allocated until
reuse or until the last record is released; the Info count is live records.

The allowance controls tracked image storage, **not an absolute RSS ceiling**.
Pinned work can briefly exceed it; masks and external previews remain resident.
Transient filter/file buffers, allocator retention, metadata, and graphics
runtime overhead also contribute to RSS. Each benchmark logs both tracked
usage and OS RSS/peak RSS. The macOS process page-in counter includes file
reads; it alone does not establish whether swapping occurred.

**Measured pressure:** a 14204 × 10652 RGBA8 document with 20 full-image history
states, `PF_MEMORY_MB=512`, retained every state. Tracked storage peaked at
516.6 MiB, process RSS at 925.5 MiB, and live scratch data at about 5.0 GiB
including recovery copies. Brush p95 including mip updates was 0.352 ms;
135 display tiles took 0.983 ms; undo p95 was 0.966 ms. Fingerprints covered
every stored tile of all 20 restored images, outside the undo latency timer.
System page-outs, swap-outs, and memory-compression counts did not increase;
swap usage stayed at 2.44 MiB. The system page-in count increased from ordinary
file activity. Raw before/after `vm_stat` and swap readings accompany the log.
The requested **150 MP 32-bit/channel** pressure scenario remains gated on the
future float document engine; RGBA8 was not presented as that measurement.

On a 16 MiB procedural tile corpus, LZ4 encoded in 19.4 ms and decoded/verified
in 4.6 ms, versus 64.1/44.0 ms for deflate level 1. Compressed sizes were 64.3%
and 49.1% of the input respectively. LZ4's faster access won for cold storage.
Incompressible tiles remain hot unless pressure requires spilling. Both codecs
were existing dependencies; no new crate or download was needed. The existing
`block 0.1.6` dependency is locally patched for current Rust's foreign-static
and explicit-ABI checks (see `vendor/README.md`).

```bash
cargo run --release -- --bench --tier 24
cargo run --release -- --bench-tools --tier 61 brush
cargo run --release -- --bench-filters --tier 24 invert
cargo run --release -- --bench-gpu --tier 24 levels
cargo run --release -- --bench --list-tiers
PF_MEMORY_MB=512 PF_GPU=off cargo run --release -- --bench-memory --tier 150 pressure
cargo run --release -- --bench-memory --tier 24 compression
python3 tools/phase1_benchmarks.py
python3 tools/phase1_memory.py
python3 tools/phase1_demo.py
```

The [Phase 1 verification report](docs/benchmarks/2026-10-11-phase1/REPORT.md)
links all tests, regression comparisons, system counters, UI captures, and
Photoshop timings at every tier. The [performance follow-up](docs/benchmarks/2026-10-11-phase1-followup/REPORT.md)
records the Object Finder fixes, size-specific GPU Auto calibration, independent
validation and retained failed measurements. A successful process exit alone
does not certify every performance target.
The initial benchmark work and skill provenance remain in
[`docs/benchmarks/2026-10-11/`](docs/benchmarks/2026-10-11/).
PSD/PSB work is in progress in Phase 2, as detailed above. 16/32-bit engine work, HDR, and raw decoding remain outside this phase.

### Legacy square benchmark (Apple M5 Pro, 18 threads, 16384 × 16384 = 268 MP)

```
blank white canvas (shared tiles)                  1.0 ms   (1 tile, 0.25 MB)
procedural fill (generate every pixel)           109.0 ms    2462 MP/s
mip pyramid build                                 13.4 ms   19982 MP/s
brush stroke (400 events, 300 px soft)           141.2 ms
hue/saturation (full image)                      206.3 ms    1301 MP/s
gaussian blur r=20 (full image)                  354.7 ms     757 MP/s
gaussian blur r=120 (full image)                 384.0 ms     699 MP/s
unsharp mask (full image)                        373.1 ms     719 MP/s
linear gradient (full image)                      81.3 ms    3303 MP/s
flip horizontal                                   20.0 ms
rotate 90° clockwise                              27.5 ms
resize to 50% (mip + bilinear)                    19.3 ms
magic wand, contiguous (blank canvas)              0.3 ms
display: 135 screen tiles at mip level 0           1.2 ms
display: live blur r=60 preview at level 2        13.8 ms
```

(The flatten figure printed by `--bench` is small because that benchmark's paint layer is
sparse, and untouched tiles are shared rather than recomposited.)

### Toolbar vs Photoshop 2026 (same machine, same 16384² scene)

The Photoshop timings come from `tools/photoshop_tools.jsx` (the best of two runs, with
`$.hiresTimer` around the scripted call). PixelForge's come from `--bench-tools`. Many tools
have no scripting hook in Photoshop. Brush strokes, healing / clone / spot-heal strokes,
Patch, Content-Aware Move, Red Eye, Rotate View, scrubby zoom and Perspective Crop can't be
timed from a script, so those rows show only PixelForge against its own budget.

| Operation (268 MP) | PixelForge | Photoshop 2026 |
|---|---:|---:|
| Selection add (rect ∪ ellipse) | 1.4 ms | 112 ms |
| Select ▸ Modify ▸ Expand / Contract 50 px | 32 / 30 ms | 607 / 653 ms |
| Select ▸ Modify ▸ Feather 30 / Border 40 / Smooth 10 | 7 / 55 / 3 ms | 115 / 696 / 140 ms |
| Quick Mask enter / exit | 0.5 / 0.5 ms | 57 / 16 ms |
| Magic Wand, contiguous (sky) | 34 ms | 280 ms |
| Select Subject (PS: ML model; PF: boundary prior + GrabCut, no model installed) | 75 ms | 2147 ms |
| Content-Aware Fill 300 px (PF: Spot Healing) | 45 ms | 268 ms |
| Content-Aware Fill 300 px (PF: Content-Aware Fill workspace, Auto sampling) | 69 ms | 268 ms |
| Content-Aware Fill 1 MP (PF: Content-Aware Patch) | 262 ms | 563 ms |
| Content-Aware Fill 1 MP (PF: Content-Aware Fill workspace, Auto sampling) | 415 ms | 563 ms |
| Gradient Linear / Radial / Angle / Reflected / Diamond (spectrum, dither) | 46 / 42 / 55 / 40 / 42 ms | 183 / 395 / 210 / 191 / 193 ms |
| Paint Bucket (sky, tol 32) | 40 ms | 270 ms |
| Magic Eraser (PS: wand + delete) | 38 ms | 297 ms |
| Text: create a 2000-glyph paragraph layer with 51 style runs (PS: ActionManager `make`) | 2.3 ms | 74.5 ms |
| Text: insert one character in that layer (PS: `set` textKey + runs) | 1.2 ms | 59.9 ms |
| Text: restyle a range (PS: edit one run's colour) | 1.5 ms | 54.4 ms |
| Shape: resize one frame (≈9000 px stroked ellipse) | 2.8 ms | 6.3 ms |
| Path → selection (9800 × 6500 ellipse) | 1.2 ms | 208 ms |
| Stroke Path (star, 10 000 px, brush) | 2.5 ms | 16 ms |
| Color Sampler: add + read | 0.01 ms | 16 ms |
| Crop 268 → 205 MP, tile-aligned / unaligned | 8 / 18 ms | 77 / 49 ms |
| Straighten 3.5° (rotate + crop) | 241 ms | 820 ms |
| Free Transform: scale a layer to 90 % | 220 ms | 729 ms |

Per-event latency on the 268 MP scene, against the budgets in `src/bench.rs`:

| Tool (500 px tip unless noted) | p95 / median per event | Budget |
|---|---:|---:|
| Brush (soft; with angle+roundness+texture+scatter) / Pencil | 0.36 / 0.37 / 0.27 ms | 2 ms |
| Dodge, Burn, Sponge, Blur, Sharpen | 0.35–0.43 ms | 2 ms |
| Smudge / Mixer Brush / Color Replacement / Background Eraser | 1.39 / 1.37 / 0.79 / 0.39 ms | 2 ms |
| Color Replacement, Limits: Contiguous / Find Edges | 1.4 / 1.9 ms | 2 ms |
| Clone Stamp, History Brush | 0.30 / 0.24 ms | 2 ms |
| Healing Brush per dab, 100 / 200 / 300 / 500 px (Quality: Fast) | 0.75 / 1.4 / 2.3 / 4.6 ms | 8 ms |
| Healing Brush per dab, 500 px, Quality: Exact | 7.8 ms | 16 ms |
| Spot Healing, 300 px blemish: Content-Aware / Create Texture | 45 / 35 ms | 60 ms |
| Patch (content-aware) / Content-Aware Move, 1 MP | 262 / 320 ms | 400 ms |
| Content-Aware Fill workspace, 300 px / 1 MP / 1 MP with rotation + scale + mirror | 69 / 415 / 477 ms | 150 / 600 / 2000 ms |
| Object finder: background precompute / hover lookup (p95) / click → selection | 708 / 0.002 / 10 ms | 2 s / 16 ms / 1 s |
| Quick Selection, coarse result / edge refinement | 9.9 / 7.7 ms | 150 ms / 1 s |
| Object Selection, first / refined | 13.8 / 132 ms | 150 ms / 1 s |
| Magnetic Lasso per mouse move (p95) | 0.62 ms | 4 ms |
| Text keystroke in a 2000-glyph paragraph with 51 runs (p95) | 1.2 ms | 4 ms |
| Text: apply a style to a 200-character range (p95) | 1.5 ms | 8 ms |
| Text: re-render that paragraph warped (Flag) | 28 ms | 150 ms |
| Shape live resize per frame (p95) | 2.1–2.8 ms | 8 ms |
| Path operation on two 500-anchor paths (Combine / Subtract / Intersect / Exclude, median) | 2.5–2.8 ms | 8 ms |
| Merge Shape Components (2 × 500 anchors) | 2.9 ms | 8 ms |
| Live Subtract re-render (2 × 500 anchors, gradient fill, dashed stroke, p95) | 8.9 ms | 16 ms |
| Corner-radius widget drag, rotated 4000 px rectangle (p95) | 1.4 ms | 8 ms |
| Frame: move placed content, 3000 px frame (p95) | 12.3 ms | 26 ms |
| Align / Distribute 20 layers (268 MP canvas) | 7.1 / 4.3 ms | 50 ms |
| Auto-Select hit test over 20 layers (p95) | < 0.01 ms | 1 ms |
| Artboard export render, 1920 × 1080 / 4-page PDF | 1.8 / 12 ms | 50 / 150 ms |
| Save for Web, 12 slices of a 3840 × 2560 photo: PNG-8 128 dithered / PNG-24 / JPEG 60 / WebP | 496 / 262 / 14 / 12 ms | 1 s |
| Eyedropper / Color Sampler, 101×101 average | 0.01 ms | 4 ms |
| Rotate View frame plan, 4K display (160 GPU-rotated tiles) | 0.003 ms | 2 ms |
| Rotated crop 8°, 3840 × 2560 → 3200 × 2000: bilinear / bicubic / Lanczos-3 | 8 / 10 / 34 ms | 400 ms |
| Crop keeping hidden pixels: rotated photo / 268 MP unaligned | 18 / 25 ms | 800 / 1500 ms |
| Content-aware crop fill, corners of an 8° turn (3840 × 2560) | 1.01 s | 6 s |
| Straighten 3.5° bicubic, 268 MP | 301 ms | 4 s |
| Vanishing Point dab, 200 px on a 4K plane: brush / stamp (median) | 0.39 / 1.15 ms | 4 / 6 ms |
| Vanishing Point marquee copy, 1000 × 1000 plane px: plain / healed | 17 / 56 ms | 150 / 400 ms |
| Vanishing Point: place a 2000 × 2000 image on a plane | 12 ms | 300 ms |

Anything over ~30 ms (crops, Straighten, Perspective Crop, Free Transform, Patch, Object
Selection refinement) runs on the background executor with the progress HUD.

## GPU acceleration

Filters, resampling, retouching, Liquify, Perspective Warp, Select and Mask and the layer
compositor have a GPU path on [wgpu](https://wgpu.rs) 29 (Metal on macOS, Vulkan or D3D12
elsewhere). The CPU path stays the reference: every GPU kernel reproduces its pixels,
bit-exactly or within 1/255 per channel (mean ΔE00 ≤ 0.2), and the tests compare the two on
photos, an edge-case image and live-preview levels. If the GPU is missing, fails, runs out of
memory or hangs, the work finishes on the CPU with the same result.

```bash
cargo run --release -- --bench-gpu 4096           # every GPU kernel vs the CPU (16.8 MP)
cargo run --release -- --bench-gpu 16384          # the same at 268 MP
cargo run --release -- --bench-gpu 4096 "lens"    # one family
cargo run --release -- --bench-gpu 16384 --write-priors   # refresh the cost model's priors
tools/photoshop_gaps.sh ps_out GPU                # Photoshop's outputs + timings for the same ops
cargo run --release -- --compare-gaps ps_out GPU  # GPU and CPU paths vs Photoshop
PF_GPU=off cargo test --release                   # the suite with the GPU off / forced on
PF_GPU=force cargo test --release
tools/demo_shots.sh tools shots/ GPU              # screenshots of the GPU paths in the app
```

**Adapter on the reference machine:** Apple M5 Pro · Metal · 37 GB unified memory (f16,
subgroups 4–64, timestamp queries, immediates, mappable primary buffers). PixelForge uses half
of the unified working set (18 GB, less 600 MB headroom): the rest is the RAM the CPU-side
document, history and fallback need.

### How it works

- **Start-up.** The device is created off the UI thread at launch. Every pipeline is then
  precompiled in the background, and a ~0.2 s micro-benchmark calibrates the cost model. Until
  that is done, the app uses the CPU.
- **Cost model (Auto).** For each operation and size class, Auto predicts GPU time (fixed
  submit cost + transfers + kernel) and CPU time. The rates come from this machine's `priors.txt`
  (measured with `--bench-gpu --write-priors`), scaled by the start-up probes, and real runs
  refine them. The GPU runs when the predicted CPU/GPU speed ratio reaches 1.1.
  End-to-end rates are separated by power-of-two image-area ranges, so small images,
  photographic documents and stress images do not overwrite one another. Gradient
  algorithms also have separate rates. Unmeasured sizes retain the transfer/kernel
  estimate and legacy large-image fallback. The model is cached per adapter, driver,
  app version and cost-model schema.
- **Data flow.** Filters run in pipelined tile-row bands (two in flight) with the apron each
  kernel needs. A GPU-resident copy of each layer (LRU) lets previews, gather filters and
  filter chains skip the upload. Display tiles are composited on the GPU straight into the
  textures GPUI draws, with no CPU round trip, through a small vendored GPUI patch for external
  textures.
- **Exactness.** Metal compiles WGSL with fast math, so the kernels use integer math,
  CPU-built tables (LUTs, unpremultiply, thresholds, kernels, rank discs) and correctly rounded
  `p_*` helpers wherever a float feeds a threshold, quantization, rank or sample position.
- **Preferences ▸ Performance (⌘,):** Use Graphics Processor (Auto / Off / Force), per-feature
  overrides (point ops, blurs, sharpening, distort, Blur Gallery, Lens Blur, Camera Raw,
  stylize, Filter Gallery, resampling, Select and Mask, compositing, retouching), and
  Benchmark GPU, which recalibrates and measures every kernel on this machine. `PF_GPU=off|auto|force`
  overrides the setting.
- **Info panel:** the adapter, mode and resident memory; the path the last operation took
  (`GPU`, `CPU (cost model)`, `GPU → CPU fallback`) with its time; and the compositor's path.

### GPU vs Photoshop 2026

`tools/photoshop_gaps.sh ps_out GPU` runs the operations Photoshop accelerates on the GPU,
via scripting, on the exported benchmark scene. It saves 4096² outputs and times each run
(best of two at 4096², one at 16384²). `--compare-gaps ps_out GPU` runs PixelForge's CPU and
GPU paths on the same scene, scores both against Photoshop's output, and times them.
Photoshop 2026 ran with Use Graphics Processor on, in Metal mode.

| Operation | SSIM GPU | SSIM CPU | ΔE00 GPU | ΔE00 CPU | CPU 16.7 MP | GPU 16.7 MP | PS 16.7 MP | CPU 268 MP | GPU 268 MP | PS 268 MP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian Blur r=20 | 1.000 | 1.000 | 0.03 | 0.03 | 10 ms | 10 ms | 21 ms | 136 ms | 160 ms | 318 ms |
| Surface Blur r=5 t=15 | 1.000 | 1.000 | 0.02 | 0.02 | 40 ms | **20 ms** | 104 ms | 456 ms | **261 ms** | 1712 ms |
| Median r=3 | 1.000 | 1.000 | 0.00 | 0.00 | 41 ms | **19 ms** | 31 ms | 631 ms | **256 ms** | 452 ms |
| Unsharp Mask 100 % r=1.5 | 1.000 | 1.000 | 0.02 | 0.02 | **6 ms** | 8 ms | 15 ms | **107 ms** | 134 ms | 221 ms |
| Smart Sharpen 100 % r=1.5 | 0.999 | 0.999 | 0.10 | 0.10 | 8 ms | 8 ms | 28 ms | **119 ms** | 180 ms | 419 ms |
| Lens Blur (defaults) | 0.997 | 0.997 | 0.17 | 0.17 | 24 ms | **18 ms** | 246 ms | 380 ms | **297 ms** | 4472 ms |
| Oil Paint (defaults) | 0.227 | 0.227 | 5.15 | 5.15 | 14 ms | **8 ms** | 144 ms | 222 ms | **131 ms** | 2162 ms |
| Camera Raw Filter (12 sliders) | 0.948 | 0.948 | 8.98 | 8.98 | 101 ms | **14 ms** | 308 ms | 1506 ms | **266 ms** | 4338 ms |
| Filter Gallery: Cutout | 0.987 | 0.987 | 12.58 | 12.58 | **118 ms** | 179 ms | 300 ms | **1340 ms** | 2658 ms | 5326 ms |
| Filter Gallery: Poster Edges | 0.777 | 0.777 | 3.69 | 3.69 | **18 ms** | 23 ms | 295 ms | **279 ms** | 445 ms | 5013 ms |
| Filter Gallery: Neon Glow | 0.992 | 0.992 | 3.30 | 3.30 | **14 ms** | 21 ms | 507 ms | **233 ms** | 289 ms | 7887 ms |
| Image Size ▸ Preserve Details 2× | 0.994 | 0.994 | 0.16 | 0.16 | 26 ms | **7 ms** | 427 ms | 421 ms | **153 ms** | 2743 ms |

- **GPU = CPU.** The two paths score identically against Photoshop. The largest
  |SSIM GPU − SSIM CPU| is 0.0000, against a bar of 0.002.
- **Speed.** The faster PixelForge path (bold) beats Photoshop on every row at both sizes:
  Lens Blur by 15×, Camera Raw by 16×, Oil Paint by 16×, Preserve Details by 18× at 268 MP,
  and the Filter Gallery effects by 4–34×. Where the GPU loses (Cutout, Poster Edges, Neon Glow,
  Unsharp Mask), Auto keeps the CPU.
- **Fidelity.** These scores come from the algorithms, not the GPU. Oil Paint's strokes don't
  match Photoshop's unpublished stroke model (SSIM 0.23). Camera Raw's output on this scene is
  much more saturated than its slider values would give, probably because it reads the
  untagged TIFF in a wider working space (ΔE 9). The Filter Gallery scores on photos are in
  [Filter Gallery fidelity](#filter-gallery-fidelity).
- **Scripting gaps.** Scripted Camera Raw needs `CMod = "Filter"` and custom settings in the
  descriptor, or it opens its dialog and blocks the script. After the Camera Raw filter has run,
  opening another TIFF can be intercepted by the Camera Raw dialog too, so the jsx opens a PNG
  for the small Preserve Details run.

### GPU vs CPU in PixelForge (Apple M5 Pro)

End to end (gather, upload, kernel, readback, new tiles), best of 2–3 runs, procedural image.
**Auto** is what the cost model picks.

| Operation | CPU 16.8 MP | GPU 16.8 MP | CPU 268 MP | GPU 268 MP | Auto (268 MP) |
|---|---:|---:|---:|---:|---|
| Levels / Curves (LUT) | 1.4 ms | 3.2 ms | 21 ms | 39 ms | CPU |
| Hue/Saturation | 9.6 ms | 3.6 ms | 145 ms | 44 ms | GPU |
| Gaussian Blur r=20 | 9.0 ms | 8.0 ms | 188 ms | 151 ms | GPU |
| Gaussian Blur r=120 | 17.6 ms | 16.0 ms | 414 ms | 301 ms | GPU |
| Unsharp Mask r=1.5 | 6.1 ms | 7.2 ms | 139 ms | 128 ms | GPU |
| Smart Sharpen 200 %, Lens Blur removal r=3 | 79 ms | 12 ms | 2285 ms | 223 ms | GPU |
| Lens Blur r=30, pentagon iris | 233 ms | 17 ms | 6556 ms | 271 ms | GPU |
| Iris Blur | 76 ms | 21 ms | 1713 ms | 322 ms | GPU |
| Spin Blur | 198 ms | 5.9 ms | 2300 ms | 85 ms | GPU |
| Path Blur | 320 ms | 14 ms | 5202 ms | 227 ms | GPU |
| Camera Raw Filter (clarity, dehaze, NR, sharpening) | 126 ms | 14 ms | 2239 ms | 243 ms | GPU |
| Median r=3 | 47 ms | 16 ms | 882 ms | 265 ms | GPU |
| Surface Blur r=5 | 82 ms | 27 ms | 2416 ms | 404 ms | GPU |
| Oil Paint (stylization 4) | 223 ms | 26 ms | 3701 ms | 459 ms | GPU |
| Twirl | 17 ms | 2.5 ms | 335 ms | 49 ms | GPU |
| Lens Correction | 85 ms | 4.4 ms | 1374 ms | 53 ms | GPU |
| Warp (Arc 50 %, mesh) | 382 ms | 48 ms | 5551 ms | 875 ms | GPU |
| Filter Gallery: Angled Strokes | 249 ms | 50 ms | 3203 ms | 885 ms | GPU |
| Filter Gallery: Chalk & Charcoal | 340 ms | 22 ms | 3835 ms | 347 ms | GPU |
| Filter Gallery: Poster Edges | 18 ms | 21 ms | 355 ms | 416 ms | CPU |
| Image Size Preserve Details 2× | 47 ms | 8.4 ms | 730 ms | 130 ms | GPU |
| Rotate 12° Lanczos | 175 ms | 16 ms | 2209 ms | 259 ms | GPU |
| Select and Mask (radius 16, smart, decontaminate) | 702 ms | 155 ms | 5981 ms | 1839 ms | GPU |
| Liquify Forward Warp stroke (r = side/8) | 16 ms | 7.6 ms | 160 ms | 56 ms | GPU |
| Perspective Warp, 2 planes | 30 ms | 7.1 ms | 855 ms | 107 ms | GPU |
| Content-Aware Fill, 1 MP hole | 437 ms | 229 ms | — | — | GPU |
| Compositing 13 layers, 60 display tiles (warm) | 35 ms | 6.2 ms | 36 ms | 5.9 ms | GPU |

In the legacy measurements above, Auto landed within 10 % (+0.5 ms) of the faster
path on all 157 rows at 16.8 MP and on 146 of 148 at 268 MP. The newer rectangular
tier results and their misses are recorded in the Phase 1 verification report. The exceptions are Filter Gallery Film Grain and Neon
Glow, light effects within 14 % of parity. CPU timings of bandwidth-bound operations vary by
up to ±40 % between runs on this machine, so near-parity rows can flip.

### What stays on the CPU, and why

- **Point operations at full size** (Levels, Curves, Threshold, Desaturate, Invert…): the CPU
  runs them at ~13 GP/s, at memory bandwidth, so uploading and reading back costs more than the
  work. They still run on the GPU inside fused adjustment chains in the compositor, and on
  resident previews.
- **Small-radius blurs, box blurs, Unsharp Mask at 16.8 MP, De-Interlace, Poster Edges,
  Cutout and other light Filter Gallery effects:** these are near parity, and the cost model
  decides per size.
- **Flood fill / Magic Wand, GrabCut graph cuts, live-wire:** serial frontier or
  augmenting-path algorithms. A GPU version iterates whole-image passes until nothing changes.
- **Vector booleans, text shaping and layout:** sequential, exact-integer, and only kilobytes
  of data.
- **Encoding** (JPEG, PNG, TIFF, WebP): bit-serial entropy coding that ends on disk anyway.
- **Brush dabs:** each dab touches a few thousand pixels and depends on the previous one, so a
  submit per dab costs more than the dab. The stroke's layer is still composited on the GPU.
- **Mip pyramids and histogram statistics:** at 268 MP the pyramid takes 11 ms on the CPU
  (27 ms on the GPU with the upload), and the Histogram panel's statistics take 0.24 ms
  (they read a mip level). The GPU only ties once the image is already resident.

Each of these has a comment in the code saying why.

### Fallback

- A validation error blocklists that kernel. Every later call of it goes to the CPU.
- A timeout (watchdog 2 s per submission, `PF_GPU_WATCHDOG_MS`), device loss or uncaptured
  error marks the context failed. The current operation finishes its remaining bands on the
  CPU, merged seamlessly, and the next one creates a new context. After three failures on
  the same adapter and driver, the GPU is disabled until Preferences ▸ Benchmark GPU
  re-enables it.
- An out-of-memory error falls back for that operation only.
- Tests simulate a lost device partway through a filter, a Liquify stroke, a retouching
  operation and a resample, and check that the result equals the CPU's. The whole suite
  passes with `PF_GPU=off` and `PF_GPU=force`.

### What Photoshop needs a GPU for

Adobe's [GPU usage page](https://helpx.adobe.com/photoshop/desktop/get-started/technical-requirements-installation/photoshop-and-graphics-processor-gpu-card-usage.html)
(February 2026) lists Oil Paint, Perspective Warp, Render ▸ Flame / Picture Frame / Tree,
3D, Birds Eye View, Flick Panning, Scrubby Zoom and Smooth Brush Resizing as not working
without a GPU. Blur Gallery, Camera Raw, Image Size ▸ Preserve Details, Lens Blur, Neural
Filters, Select Focus, Select and Mask and Smart Sharpen are listed as GPU-accelerated. In
PixelForge everything works without a GPU: Oil Paint, Perspective Warp and Scrubby Zoom have
CPU paths that the GPU reproduces exactly. Render Flame, Picture Frame, Tree and 3D aren't
implemented.

Blur Gallery, Liquify, Perspective Warp and Select and Mask have no scripting hook in
Photoshop (their events need the full on-canvas widget descriptor), so they have no
Photoshop timing above. Their PixelForge numbers come from `--bench-gpu`.

## ML backend (optional)

`cargo build --release --features ml` links ONNX Runtime through the `ort` crate. Execution
providers are tried best first, and any that are missing at run time fall back to the CPU:

| Platform | Providers |
|---|---|
| macOS | CoreML |
| Linux | TensorRT, CUDA |
| Windows | TensorRT, CUDA, DirectML |

`ort`'s `lax-feature-matching` downloads the closest prebuilt runtime for each platform at build
time. `PF_ML_EP=cpu|coreml|cuda|tensorrt|directml` forces one provider. Models are plain
`.onnx` files that are not bundled. They are looked up in `$PF_MODELS`, then `./models`, then
next to the executable:

| File | Used by | Status |
|---|---|---|
| `select_subject.onnx` (BiRefNet / U²-Net / ISNet / MODNet-style: `1×3×S×S` in, `1×1×S×S` out) | Select Subject, Object Selection (box), object finder | not installed; the classical path is used and the options bar says so |
| `depth_anything_v2_small.onnx` (Apache-2.0, 99 MB) | Neural Filters ▸ Depth Blur | installed |

Inference runs at ≤ 1024 px (518 px for depth), and the result is upsampled with the guided
filter on boundary tiles only.

## Gap-closing features vs Photoshop 2026

`tools/photoshop_gaps.sh ps_out A` runs Photoshop on the CC0 test photos in
`testdata/photos` (licences in `LICENSES.md`) and saves its masks, fills and timings.
`pixelforge --compare-gaps ps_out A` runs PixelForge on the same inputs. The ground-truth
masks in `testdata/photos/gt/` were traced for this project. Their edges were placed at the
half-way colour change and then corrected by hand, so they slightly favour colour-contrast
methods.

**Segmentation (IoU against ground truth).** No segmentation model is installed, so
PixelForge's Select Subject is the classical path.

| Photo | PixelForge Select Subject | PixelForge Object Selection (box) | Photoshop Select Subject | PixelForge | Photoshop |
|---|---:|---:|---:|---:|---:|
| Portrait (hair) | 0.533 | 0.880 | 0.957 | 645 ms | 1264 ms |
| Dog on grass | 0.804 | 0.807 | 0.986 | 58 ms | 1171 ms |
| Butterfly + thistle | 0.815 | 0.954 | 0.485 (it picks the butterfly alone) | 459 ms | 1122 ms |
| Butterfly only | 0.419 | | 0.976 | | |

Before this work, Select Subject was plain inset-box GrabCut, which scored 0.221 on the portrait
(it treats the frame as background). The boundary-connectivity prior fixes that, but hair and
cast shadows still need the model.

**Content-Aware Fill (inside the hole, against the original pixels).** Photoshop's fill is
`Edit ▸ Fill ▸ Content-Aware`. PixelForge's is the Content-Aware Fill workspace with defaults
(Auto sampling, Default colour adaptation).

| Photo, hole | PixelForge SSIM / ΔE00 | Photoshop SSIM / ΔE00 | PixelForge | Photoshop |
|---|---:|---:|---:|---:|
| Portrait backdrop, 300 px | 0.952 / 5.5 | 0.969 / 2.9 | 84 ms | 283 ms |
| Lawn, 300 px | 0.162 / 10.1 | 0.204 / 9.3 | 69 ms | 276 ms |
| Lawn, 1 MP | 0.119 / 12.7 | 0.132 / 11.9 | 364 ms | 623 ms |
| Lawn, daisies | 0.239 / 9.3 | 0.275 / 8.8 | 80 ms | 275 ms |
| Butterfly bokeh, 300 px | 0.803 / 3.5 | 0.821 / 2.7 | 74 ms | 269 ms |

Grass can't be restored exactly, so SSIM is low for both apps there. It also rewards plausible
blur: an averaging fill scored up to 0.03 higher but looked like a smudge. PixelForge's default
therefore takes final pixels from the best matches, and a focus prior keeps it from copying
defocused areas. In the screenshots its lawn fills are less visible than Photoshop's.

**Type (phase B).** Text layers carry per-character style runs and per-paragraph styles. Each
paragraph can be set by the Every-line Composer (Knuth–Plass optimal breaking, Photoshop's
default) or the Single-line Composer (greedy). Justification uses Photoshop's default word
spacing of 80 %, 100 % and 133 % (minimum, desired, maximum), with all four last-line modes.
Hyphenation is Liang's TeX algorithm on the en-US patterns of the hyph-utf8 package, in
`assets/hyph` under an all-permissive licence. Words the patterns don't cover, such as
non-ASCII words, fall back to a syllable heuristic. Everything is saved in the `.pforge` "PFLX"
chunk. Files without the chunk still open, and older readers skip it.

| Operation, 2000-glyph paragraph with 51 style runs | PixelForge | Photoshop 2026 (ActionManager) |
|---|---:|---:|
| Create the layer | 2.3 ms | 74.5 ms |
| Insert one character | 1.2 ms | 59.9 ms |
| Restyle a range / edit one run's colour | 1.5 ms | 54.4 ms |

`tools/photoshop_gaps.sh ps_out B` makes Photoshop convert each warp style, at 16 settings
(bend, H and V distortion), into a path. `--compare-gaps ps_out B` rasterizes those outlines
and PixelForge's warp of the same text box, then reports the IoU. Photoshop's warps are exact
bicubic Bézier patches. PixelForge rebuilds the 4 × 4 control mesh with the laws measured
from Photoshop's output:

- **Bend:** circular-arc control points for each style.
- **Vertical distortion:** applied first, by a per-style row law. The Shells scale each row
  about the midpoint of its end points. Inflate and Squeeze scale offsets by the factor of the
  nearer edge row (or, horizontally, the nearer edge column).
- **Horizontal distortion:** applied next. It scales each column about the midpoint of its
  end points.

| Style | bend 50 | bend −50 | bend 100 | h 40 | v 40 | h&v 40 | worst of 16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Arc | 0.998 | 0.998 | 0.998 | 0.998 | 0.995 | 0.995 | 0.964 |
| Arc Lower / Upper | 0.999 / 0.999 | 0.998 / 0.998 | 0.998 / 0.999 | 0.999 / 0.999 | 0.998 / 0.999 | 0.998 / 0.999 | 0.995 / 0.998 |
| Arch | 0.998 | 0.998 | 0.997 | 0.998 | 0.996 | 0.996 | 0.989 |
| Bulge | 0.999 | 0.996 | 0.999 | 0.999 | 0.999 | 0.999 | 0.996 |
| Shell Lower / Upper | 0.998 / 0.999 | 0.998 / 0.998 | 0.998 / 0.998 | 0.998 / 0.999 | 0.998 / 0.998 | 0.998 / 0.998 | 0.995 / 0.997 |
| Flag / Wave | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 0.998 / 0.999 |
| Fish / Rise | 0.998 / 0.999 | 0.998 / 0.998 | 0.998 / 0.997 | 0.998 / 0.999 | 0.998 / 0.998 | 0.998 / 0.998 | 0.998 / 0.997 |
| Fisheye | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.999 | 0.999 |
| Inflate / Squeeze | 0.998 / 0.998 | 0.997 / 0.997 | 0.999 / 0.999 | 0.999 / 0.999 | 0.999 / 0.999 | 0.992 / 0.989 | 0.989 / 0.984 |
| Twist | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |

Mean over all 240 settings: **0.9981**. The worst case is Arc at bend 100 % with V 40 % (0.964).
Its control mesh matches Photoshop's fitted mesh, so the difference comes from rasterizing a
nearly closed, self-overlapping outline.

**Vectors, shapes and frames (phase C).** Path operations use exact integer predicates on a
1/1024 px snapped grid. Each operation is planarized by snap rounding and evaluated by a
winding-number sweep, so results are always valid, non-crossing contours. On two discs the
four operations land within 0.1 % of the analytic areas (`live_path_operations_match_analytic_areas`).
`tools/photoshop_gaps.sh ps_out C` builds the same two 500-anchor paths in Photoshop and
makes a selection with each operation, and `--compare-gaps ps_out C` compares the results:

| Operation | PixelForge selection area | Photoshop selection area | Difference | PixelForge boolean + selection | Photoshop Make Selection |
|---|---:|---:|---:|---:|---:|
| Combine | 13 298 679 | 13 296 245 | +0.018 % | 4.8 ms | 41.4 ms |
| Subtract | 730 705 | 728 242 | +0.338 % | 4.1 ms | 17.0 ms |
| Intersect | 11 837 269 | 11 834 944 | +0.020 % | 3.5 ms | 39.0 ms |
| Exclude | 1 461 410 | 1 456 672 | +0.325 % | 3.7 ms | 19.1 ms |

PixelForge's selection coverage equals the exact polygon area. Photoshop's sits about 0.19 px
inside the outline along the whole boundary, a constant ≈ 2,400 px² over ≈ 12,600 px of
boundary. That gap is only visible in percentage terms on the small Subtract and Exclude
results.

**Layers, Move, artboards and slices (phase D).**
- **Multi-layer selection** is saved with the document, as is the **locked Background**:
  - The Background can't move or be reordered, and its opacity and blend mode are locked.
  - The Eraser, Delete and Layer via Cut give the background colour, and moved pixels leave it
    behind. Canvas growth fills with it too.
  - Magic Eraser and Background Eraser turn it into "Layer 0", as Photoshop does;
    double-clicking it also does.
- **Artboards** are a layer kind. Older files' artboard annotations open as artboard layers.
- **Slice options** are saved in a new `.pforge` chunk.

Photoshop 2026's legacy Save for Web cancels under scripting, so `tools/photoshop_gaps.sh
ps_out D` times its scriptable equivalents on the same 1920 × 1280 photo: Indexed Color
(local adaptive, 128 colours, diffusion) + Save PNG, Save PNG, and Save JPEG (quality 6 of
12). `--compare-gaps ps_out D` encodes the identical source and scores it with SSIM against
that source:

| Format | PixelForge size / SSIM / time | Photoshop size / SSIM / time |
|---|---:|---:|
| PNG-8, 128 colours, dithered | 1727 KB / 0.967 / 409 ms | 1626 KB / 0.950 / 1581 ms |
| PNG-24 | 4893 KB / 1.000 / 169 ms | 4809 KB / 1.000 / 1086 ms |
| JPEG at Photoshop's file size (PixelForge q73, optimized Huffman) | 540 KB / 0.938 / 22 ms | 546 KB / 0.936 / 300 ms |

JPEGs (Save for Web, PDF pages, Export) come from PixelForge's own optimized-baseline encoder
(`src/engine/jpeg.rs`), the same as Photoshop's "Optimized Baseline": Huffman tables built from
each image's statistics, ~5–10 % smaller than the default tables at the same quality.

**Crop, Rotate View and documents (phase E).**
- **Rotate View** now runs on the GPU. GPUI is vendored in `vendor/` (see its README)
  so image sprites can take a transformation, with matching Metal, WGSL and HLSL shaders and
  culling by the transformed bounds. Each frame draws the cached display tiles rotated, with no
  CPU resampling. Tiles overlap by half a device pixel and clamp their sampling to their own
  texels, so no seams show. Planning a 4K frame takes 3 µs; the old CPU path took 7.4 ms.
- **Crop:**
  - With Delete Cropped Pixels off, a pixel layer keeps everything outside the canvas as
    hidden pixels, together with its mask's hidden part. They are saved in `.pforge` ("PFLX"
    tag 5) and are undoable. They come back with Move (already while dragging), a larger crop,
    any canvas transform or Image ▸ Reveal All. The Background becomes "Layer 0", as in
    Photoshop.
  - Content-Aware fills new canvas area. Border holes are filled on a padded copy in
    overlapping windows, so the fill keeps the grass texture right to the edge.
  - Straighten inside Crop turns the frame and fits the largest level rectangle.
  - Straighten Layer on a non-Background layer turns only that layer.

`tools/photoshop_gaps.sh ps_out E` times Photoshop on the dog photo. `--compare-gaps ps_out E`
does the same in PixelForge and compares pixels. Photoshop's crop of its rotated canvas sits
half a pixel off the geometric centre, so the comparison aligns on the best half-pixel shift
(−0.5, −0.5):

| Operation (dog photo, 3840 × 2560) | PixelForge | Photoshop 2026 |
|---|---:|---:|
| Rotate 8° + crop 3200 × 2000, bicubic: SSIM vs Photoshop's pixels | 0.982 (Lanczos-3 0.982, bilinear 0.975) | — |
| Same, time | 10 ms (Lanczos-3 34 ms) | 51 ms |
| Crop 3200 × 2000 with Delete Cropped Pixels off | 1 ms | 43 ms |
| Content-Aware fill of an 8° turn's corners | 1.0 s | 1.1 s |

**Measurement, Info, notes and brushes (phase F).** `tools/photoshop_gaps.sh ps_out F` reads
Photoshop's Lab / HSB / CMYK for a 17³ sRGB grid plus 200 random colours. It also records
Photoshop's measurements of a disc selection and of 21 flat fills. The grid became the CMYK table
(`assets/cmyk`); the random colours, which are off the grid, check it. `--compare-gaps ps_out F`:

| Info readout, 200 random colours | mean difference | worst |
|---|---:|---:|
| Lab D50, ΔE76 (Photoshop's scripting adds −0.5 to a and b) | 0.22 | 0.49 |
| HSB, largest of H°, S %, B % | 0.005 | 0.03 |
| CMYK, SWOP v2 (largest ink %) | 0.46 | 3.4 |
| CMYK, the naive formula used before | 27 | 65 |

| Disc selection, r = 200 px (exact: area 125 664, perimeter 1256.6) | PixelForge | Photoshop 2026 |
|---|---:|---:|
| Area | 125 667 | 125 692 |
| Perimeter | 1258.6 | 1324.7 |
| Circularity | 0.997 | 0.900 |
| Gray Value (Mean), gray-100 fill | 77.99 | 77.88 |
| Record Measurements | 2.8 ms | 30 ms |

- **Perimeter:** PixelForge measures the anti-aliased outline, within 0.2 % of the true circle.
  Photoshop's follows the pixel boundary (+5.4 %), so its circularity reads 0.90.
- **Gray Value** follows Photoshop's gray scale (Dot Gain 20 %, as a function of L\*). It is
  within 0.17 of Photoshop on all 21 fills.
- **Brushes:**
  - With every dynamic on, a 500 px stroke costs 0.84 ms median / 1.47 ms p95 per mouse event
    (budget 2 ms).
  - Photoshop 2026's own *Legacy Brushes.abr* imports as 296 sampled tips (165 computed brushes
    are skipped and reported).
  - A seeded all-dynamics stroke is a reference render
    (`docs/shots/phaseF/reference_all_dynamics_seed7.png`). The test checks its exact pixels.
  - Photoshop has no scripting hook for painting with dynamics, so brush speed can't be timed
    against it.

## Not implemented / differs from Photoshop

- **ML-only Neural Filters:** Colorize, Style Transfer, Smart Portrait, Makeup Transfer,
  Landscape Mixer and Depth Blur need trained models; they were left out of this round by request,
  and the candidate colorization and style models were not approved for download.
- **Neural Select Subject / Object Selection:** the ONNX backend is built in (`--features ml`), but
  no segmentation model ships, because the candidate models (BiRefNet, MobileSAM) were not
  approved for download. Without one, the classical boundary prior plus GrabCut scores IoU 0.53 /
  0.80 / 0.82 on the three test photos, against Photoshop's 0.96 / 0.99 / 0.98, and misses
  fine hair and cast shadows.
- **Filter Gallery fidelity:** 9 of 47 effects reach SSIM 0.85 against Photoshop; the rest
  depend on Adobe's unpublished random patterns or texture bitmaps (see
  [Filter Gallery fidelity](#filter-gallery-fidelity)).
- **Content-Aware Fill** is PatchMatch with adaptation plus a Poisson blend, not Adobe's
  unpublished algorithm. Its SSIM is within 0.05 of Photoshop's on every test hole, but the pixels
  differ.
- **Type:** line endings can differ from Photoshop's in a few places. Adobe's composer weights
  and hyphenation dictionaries are proprietary, so PixelForge uses Knuth–Plass with TeX's
  patterns.
- **Rotate View on Windows:** the HLSL shader change is written but has not been compiled,
  because this Mac has no DirectX compiler and the Windows target was declined.
- **Tilt control:** GPUI reports no pen tilt, so brush dynamics set to Tilt behave as with a
  mouse (upright pen).

## Filters vs Photoshop 2026

Same 8192 × 8192 (67 MP) RGB image, default or matching parameters, best of two runs, on an
M5 Pro (18 cores). Photoshop was timed in-process with `$.hiresTimer` through its own scripting
([tools/photoshop_bench.jsx](tools/photoshop_bench.jsx)). PixelForge was timed with
`pixelforge --bench-filters 8192` and includes building the new tile grid. **PixelForge is faster
on all 43 filters both apps could run, with a geometric-mean speed-up of 5.1×.** Lens Blur,
Fibers and ZigZag failed under Photoshop's scripting API, so they aren't compared.

| Filter | Photoshop 2026 | PixelForge | Speed-up |
|---|---:|---:|---:|
| Mosaic | 4,593.5 ms | 25.1 ms | **183.0×** |
| Smart Blur | 30,329.3 ms | 477.5 ms | **63.5×** |
| Solarize | 132.3 ms | 5.2 ms | **25.4×** |
| Average | 82.7 ms | 5.2 ms | **15.9×** |
| Equalize | 77.0 ms | 5.0 ms | **15.4×** |
| Auto Contrast | 74.9 ms | 4.9 ms | **15.3×** |
| Levels | 80.2 ms | 6.1 ms | **13.1×** |
| Twirl | 794.4 ms | 62.3 ms | **12.8×** |
| Minimum | 73.6 ms | 7.8 ms | **9.4×** |
| Maximum | 68.1 ms | 7.7 ms | **8.8×** |
| Pinch | 510.1 ms | 62.5 ms | **8.2×** |
| Posterize | 36.4 ms | 4.8 ms | **7.6×** |
| Dust & Scratches | 161.1 ms | 21.8 ms | **7.4×** |
| Clouds | 754.2 ms | 103.7 ms | **7.3×** |
| Difference Clouds | 751.9 ms | 106.7 ms | **7.0×** |
| Crystallize | 752.1 ms | 106.8 ms | **7.0×** |
| Radial Blur | 7,604.8 ms | 1,139.0 ms | **6.7×** |
| Brightness/Contrast | 107.5 ms | 22.4 ms | **4.8×** |
| Invert | 22.1 ms | 4.9 ms | **4.5×** |
| Sharpen Edges | 74.5 ms | 17.7 ms | **4.2×** |
| Add Noise | 44.9 ms | 10.7 ms | **4.2×** |
| Fragment | 49.3 ms | 12.8 ms | **3.9×** |
| Gaussian Blur | 82.3 ms | 23.5 ms | **3.5×** |
| Box Blur | 51.0 ms | 14.6 ms | **3.5×** |
| Threshold | 33.7 ms | 9.7 ms | **3.5×** |
| Despeckle | 76.9 ms | 22.6 ms | **3.4×** |
| Ripple | 196.2 ms | 58.4 ms | **3.4×** |
| Desaturate | 33.8 ms | 10.3 ms | **3.3×** |
| Spherize | 197.0 ms | 61.7 ms | **3.2×** |
| Polar Coordinates | 211.0 ms | 67.4 ms | **3.1×** |
| High Pass | 88.1 ms | 31.0 ms | **2.8×** |
| Emboss | 55.5 ms | 19.6 ms | **2.8×** |
| Blur | 25.2 ms | 9.5 ms | **2.7×** |
| Blur More | 28.5 ms | 11.4 ms | **2.5×** |
| Shadows/Highlights | 258.6 ms | 112.0 ms | **2.3×** |
| Motion Blur | 40.1 ms | 18.0 ms | **2.2×** |
| Median | 140.7 ms | 63.2 ms | **2.2×** |
| Surface Blur | 432.6 ms | 262.0 ms | **1.7×** |
| Unsharp Mask | 43.3 ms | 27.0 ms | **1.6×** |
| Facet | 53.3 ms | 34.0 ms | **1.6×** |
| Sharpen More | 30.2 ms | 21.6 ms | **1.4×** |
| Find Edges | 47.5 ms | 35.3 ms | **1.3×** |
| Sharpen | 25.9 ms | 20.6 ms | **1.3×** |

### Filter Gallery vs Photoshop 2026

Same method on 4096 × 4096 (16.7 MP), because the Photoshop gallery is slow. Photoshop ran each
effect with its default settings ([tools/photoshop_gallery_bench.jsx](tools/photoshop_gallery_bench.jsx)).
**PixelForge is faster on all 47 effects, with a geometric-mean speed-up of 20×.** These effects
have no published algorithms, so PixelForge's versions are built from its own primitives (relief
shading, stroke noise, Kuwahara, Voronoi, procedural textures) and tuned against Photoshop's output.

| Filter Gallery effect | Photoshop 2026 | PixelForge | Speed-up |
|---|---:|---:|---:|
| Colored Pencil | 2,952.2 ms | 20.8 ms | **141.9×** |
| Dry Brush | 1,445.8 ms | 21.3 ms | **67.9×** |
| Underpainting | 3,830.2 ms | 56.7 ms | **67.6×** |
| Stamp | 817.2 ms | 15.0 ms | **54.5×** |
| Film Grain | 347.0 ms | 6.7 ms | **51.8×** |
| Water Paper | 2,464.3 ms | 49.1 ms | **50.2×** |
| Sponge | 2,515.2 ms | 58.9 ms | **42.7×** |
| Fresco | 1,315.1 ms | 31.2 ms | **42.2×** |
| Ink Outlines | 1,752.3 ms | 41.8 ms | **41.9×** |
| Smudge Stick | 532.5 ms | 14.3 ms | **37.2×** |
| Neon Glow | 493.7 ms | 13.8 ms | **35.8×** |
| Diffuse Glow | 662.6 ms | 18.7 ms | **35.4×** |
| Sprayed Strokes | 817.2 ms | 27.5 ms | **29.7×** |
| Palette Knife | 3,036.5 ms | 102.4 ms | **29.7×** |
| Reticulation | 643.9 ms | 22.2 ms | **29.0×** |
| Photocopy | 420.9 ms | 14.7 ms | **28.6×** |
| Craquelure | 2,533.2 ms | 99.2 ms | **25.5×** |
| Poster Edges | 460.5 ms | 18.1 ms | **25.4×** |
| Mosaic Tiles | 1,116.5 ms | 47.3 ms | **23.6×** |
| Torn Edges | 370.2 ms | 16.2 ms | **22.9×** |
| Patchwork | 421.5 ms | 19.4 ms | **21.7×** |
| Spatter | 619.8 ms | 29.5 ms | **21.0×** |
| Grain | 144.3 ms | 7.0 ms | **20.6×** |
| Accented Edges | 851.9 ms | 42.1 ms | **20.2×** |
| Graphic Pen | 598.8 ms | 29.8 ms | **20.1×** |
| Sumi-e | 1,240.2 ms | 63.0 ms | **19.7×** |
| Dark Strokes | 1,961.9 ms | 101.8 ms | **19.3×** |
| Watercolor | 1,979.7 ms | 103.8 ms | **19.1×** |
| Rough Pastels | 989.5 ms | 52.4 ms | **18.9×** |
| Plastic Wrap | 590.5 ms | 35.6 ms | **16.6×** |
| Angled Strokes | 2,053.9 ms | 134.1 ms | **15.3×** |
| Halftone Pattern | 182.9 ms | 12.8 ms | **14.3×** |
| Glowing Edges | 781.0 ms | 55.1 ms | **14.2×** |
| Plaster | 615.4 ms | 47.2 ms | **13.0×** |
| Bas Relief | 303.4 ms | 24.9 ms | **12.2×** |
| Crosshatch | 1,795.4 ms | 154.5 ms | **11.6×** |
| Charcoal | 1,531.7 ms | 132.2 ms | **11.6×** |
| Note Paper | 153.9 ms | 13.6 ms | **11.3×** |
| Conté Crayon | 219.7 ms | 22.5 ms | **9.8×** |
| Ocean Ripple | 483.6 ms | 52.4 ms | **9.2×** |
| Chrome | 1,078.5 ms | 121.7 ms | **8.9×** |
| Glass | 277.9 ms | 39.4 ms | **7.1×** |
| Paint Daubs | 676.5 ms | 98.3 ms | **6.9×** |
| Cutout | 843.7 ms | 127.8 ms | **6.6×** |
| Stained Glass | 398.2 ms | 61.3 ms | **6.5×** |
| Chalk & Charcoal | 781.1 ms | 145.1 ms | **5.4×** |
| Texturizer | 101.1 ms | 37.9 ms | **2.7×** |

#### Filter Gallery fidelity

Each effect ran at its default settings on the 1024 × 683 dog photo in Photoshop 2026
(`tools/photoshop_gaps.sh <dir> G`) and in PixelForge
(`pixelforge --compare-gaps /tmp/pf_ps_out G`), with foreground black and background white. SSIM
is on luminance. **9 of 47 reach SSIM 0.85 (up from 0 before tuning); the mean is 0.531 (was
0.304).** Most of the rest are built
from Photoshop's own random strokes, grain or cells, which no reimplementation can line up pixel for
pixel. Those were tuned to match Photoshop's statistics and look (checked side by side) rather
than to raise the score: a blurrier version usually scores higher but looks wrong.

Side-by-side renders (Photoshop left, PixelForge right, four effects per sheet in table order) are in
[docs/shots/phaseG/gallery_vs_ps_*.jpg](docs/shots/phaseG/).

| # | Effect | SSIM | mean ΔE2000 | Why below 0.85 |
|---:|---|---:|---:|---|
| 0 | Colored Pencil | 0.470 | 7.6 | Random hatch strokes; tone, paper colour and stroke direction match. |
| 1 | Cutout | **0.889** | 3.8 | — |
| 2 | Dry Brush | 0.532 | 4.4 | Photoshop's two-tone dab pattern doesn't follow local image structure (57% agreement); Kuwahara is the closest model found. |
| 3 | Film Grain | 0.809 | 4.7 | Independent per-pixel noise from Photoshop's generator; the noise model and highlight boost match. |
| 4 | Fresco | 0.232 | 15.3 | A random light/dark field over the whole image (±70 levels); per-tone means match within 10 levels. |
| 5 | Neon Glow | **0.987** | 2.0 | — |
| 6 | Paint Daubs | 0.702 | 4.1 | Photoshop's fine streaky texture inside the daubs isn't tied to the image. |
| 7 | Palette Knife | 0.797 | 2.9 | Large slabs match; the fine blocky mottling inside them does not. |
| 8 | Plastic Wrap | 0.674 | 5.6 | Highlight worms follow Photoshop's own height field; a flat haze scores 0.72 but loses them. |
| 9 | Poster Edges | **0.980** | 1.0 | — |
| 10 | Rough Pastels | 0.358 | 7.6 | Random stroke streaks and canvas texture; no texture would score 0.51 but not look like pastel. |
| 11 | Smudge Stick | 0.776 | 2.8 | Fine shape of the dark smears inside dark fur and grass. |
| 12 | Sponge | 0.359 | 6.8 | Random sponge blotches; blotch size matched by autocorrelation. |
| 13 | Underpainting | 0.586 | 3.1 | Random canvas rows; a plain blur scores 0.62. |
| 14 | Watercolor | 0.336 | 11.5 | Photoshop's blotch segmentation and black gaps in grass can't be aligned; palette matches. |
| 15 | Accented Edges | **0.998** | 0.3 | — |
| 16 | Angled Strokes | 0.749 | 3.8 | Photoshop's stroke rendering adds structure beyond the per-direction medians (0.87 ceiling even in pure stroke areas). |
| 17 | Crosshatch | 0.604 | 7.8 | Pencil texture correlates only ~0.5 with the image. |
| 18 | Dark Strokes | 0.666 | 6.5 | A near-threshold tone curve amplifies small stroke differences. |
| 19 | Ink Outlines | 0.359 | 19.0 | Channel-independent speckle; no neighbourhood function tried exceeds 0.32. |
| 20 | Spatter | 0.210 | 7.9 | Random displacement. |
| 21 | Sprayed Strokes | 0.239 | 7.5 | Random displacement. |
| 22 | Sumi-e | 0.712 | 6.2 | A near-threshold tone curve (≈7× contrast) amplifies small stroke differences. |
| 23 | Diffuse Glow | 0.653 | 3.8 | Uniform random grain on the glow; even with no grain the ceiling is 0.735. |
| 24 | Glass | 0.164 | 8.5 | Random displacement field; size, strength and mean shift match. |
| 25 | Ocean Ripple | 0.166 | 9.1 | Random displacement field; size, strength and mean shift match. |
| 26 | Bas Relief | **0.921** | 5.2 | — |
| 27 | Chalk & Charcoal | 0.473 | 5.8 | Random stroke placement; stroke shape, directions and tone curve match. |
| 28 | Charcoal | 0.270 | 14.5 | Shadow hatching is uncorrelated with the image; stroke length, width and spacing match. |
| 29 | Chrome | 0.242 | 11.8 | A chaotic process; no function of the blurred image explains Photoshop's output (best 0.33). |
| 30 | Conté Crayon | 0.054 | 15.2 | Photoshop overlays a fixed 128 × 120 texture tile (Adobe's asset); ours is procedural. |
| 31 | Graphic Pen | 0.172 | 34.7 | Random strokes; stroke statistics match Photoshop's within 0.01 correlation. |
| 32 | Halftone Pattern | **0.969** | 2.3 | — |
| 33 | Note Paper | 0.359 | 10.3 | Random paper grain (statistics match); no grain would score 0.41. |
| 34 | Photocopy | **0.992** | 0.5 | — |
| 35 | Plaster | 0.506 | 11.6 | Flat areas follow Photoshop's document-height light ramp; its ripple-like contour lines inside the plaster are not reproduced. |
| 36 | Reticulation | 0.050 | 16.7 | Random grain covers the whole image. |
| 37 | Stamp | 0.529 | 12.0 | Photoshop thresholds with ~12 levels of noise; no deterministic threshold beats 88.5% agreement. |
| 38 | Torn Edges | 0.554 | 5.6 | Ragged edges come from random grain. |
| 39 | Water Paper | 0.603 | 4.2 | Each cell picks a random fibre direction; a deterministic average scores 0.62 but looks soft. |
| 40 | Glowing Edges | **0.999** | 0.0 | — |
| 41 | Craquelure | 0.153 | 21.0 | Random crack network. |
| 42 | Grain | 0.390 | 21.8 | Independent ±50 uniform noise per channel. |
| 43 | Mosaic Tiles | 0.392 | 17.5 | Random tile joints. |
| 44 | Patchwork | **0.940** | 2.6 | — |
| 45 | Stained Glass | 0.126 | 22.5 | Random Voronoi cells; the centre light spot and the leaded frame match Photoshop. |
| 46 | Texturizer | 0.279 | 11.1 | Photoshop's Canvas is a fixed 128 × 120 bitmap (Adobe's asset); ours is a procedural weave of the same spacing and strength. |

### How the filters are built

| Technique | Used by |
|---|---|
| **Three classes, one code path:** point, local (padded window) and warp/gather (mapped sampling). The live preview runs the same code on the visible mip tiles with spatial parameters scaled by 2^-level. | All filters |
| **Planar + transposed separable passes:** column running sums vectorize (0.1 ns/px). Horizontal passes run as column passes on an 8×8-blocked transpose instead of as a serial row chain (1.1 ns/px). | Gaussian, Box, High Pass, Unsharp, Shadows/Highlights |
| **Direct u16 fixed-point taps** for small sigma, each tap one 16-lane slice op | Blur, Blur More, Sharpen family, small Gaussians |
| **Radius-independent strips** (full-width, then full-height) for huge radii | Gaussian r > ~60 and the other blur-derived filters |
| **Perreault–Hébert O(1) median** (coarse + lazy fine histograms) | Median, Dust & Scratches (r ≥ 3), Smart Blur |
| **Sorting networks on rows:** 3×3 column-sort median, 5×5 forgetful selection | Median / Despeckle / Dust & Scratches r ≤ 2 |
| **Yang's O(1) bilateral (PBFIC):** box-filtered intensity levels, only the levels a tile actually spans | Surface Blur, Reduce Noise |
| **van Herk / Gil-Werman** (large r) and **shifted-slice min/max** (small r) | Maximum, Minimum |
| **Sheared-line running sums:** O(1) motion blur at any angle, with exact box for axis angles | Motion Blur |
| **Row-vectorized disk sums** over prefix rows plus a gamma LUT for bokeh | Lens Blur |
| **Mip-prefiltered sampling:** taps spaced along the path read the matching mip level, so long arcs need ~24 taps | Radial Blur |
| **Integral/box-sum Kuwahara** with row-vectorized variance keys | Oil Paint, Facet |
| **33³ 3D LUT with tetrahedral interpolation:** any color transform costs four taps | Hue/Sat, Vibrance, Color Balance, B&W, Photo Filter, Gradient Map, HSB/HSL |
| **Histogram stats from the mip pyramid:** sampled from ≥1 MP of the pyramid, no full-image pass | Equalize, Auto Tone/Contrast, Average |
| **Flat-tile shortcut:** a uniform tile in a uniform neighborhood is filtered once per color and shared | All position-independent filters |
| **Adaptive blocks:** tiles render in 256–2048 px blocks sized to the filter's padding | All local filters |
| **mimalloc:** per-tile scratch buffers recycle instead of page-faulting fresh memory | All filters |

## Shortcuts

Tool keys follow Photoshop's. A key selects its group's current tool, Shift + key cycles the
group, and a long press or right-click on a toolbar slot opens the flyout.

| Key | Tool group | Key | Action |
|---|---|---|---|
| V | Move · Artboard | ⌘Z / ⇧⌘Z | Undo / Redo |
| M | Marquee (rect, ellipse, row, column) | ⌘A / ⌘D / ⇧⌘I | Select all / Deselect / Inverse |
| L | Lasso · Polygonal · Magnetic | ⌘X ⌘C ⌘V | Cut / Copy / Paste as layer |
| W | Object · Quick Selection · Magic Wand | ⌥⌫ / ⌘⌫ / ⌫ | Fill FG / Fill BG / Clear (⌫ deletes while typing) |
| C | Crop · Perspective Crop · Slice · Slice Select | ⇧⌘N / ⌘J / ⇧⌘J | New layer / via Copy / via Cut |
| K | Frame | ⌘G / ⇧⌘G / ⌥⌘G | Group / Ungroup / Clipping mask |
| I | Eyedropper · Color Sampler · Ruler · Note · Count | ⌘E / ⇧⌘E | Merge down / Flatten |
| J | Spot Healing · Healing · Patch · Content-Aware Move · Red Eye | ⌘] / ⌘[ | Layer up / down |
| B | Brush · Pencil · Color Replacement · Mixer | ⌘I ⇧⌘U ⌘U ⌘L ⌘M ⌘B | Invert, Desaturate, Hue/Sat, Levels, Curves, Color Balance |
| S / Y | Clone · Pattern Stamp / History · Art History | ⌥⌘I | Image Size |
| E | Eraser · Background · Magic Eraser | ⌘0 / ⌘1 / ⌘= / ⌘- | Fit / 100 % / Zoom in / out |
| G | Gradient · Paint Bucket | [ ] / ⇧[ ⇧] | Brush size / hardness |
| O | Dodge · Burn · Sponge | 1…9, 0 | Opacity 10…90 %, 100 % |
| P | Pen family (7 tools) | X / D | Swap / Default colors |
| T | Type (4 tools) | Q | Quick Mask |
| A | Path · Direct Selection | F | Cycle screen mode |
| U | Shapes (6 tools) | Enter / Esc | Commit / cancel (crop, perspective crop, gradient, pen, polygon lasso) |
| H / R | Hand / Rotate View | Esc or ⌘↩ | Commit text |
| O / ⇧O (Crop) | Cycle crop overlay / its orientation | ⌃Tab / ⌃⇧Tab / ⌘W | Next / previous document, close |
| Z (⌥ = out) | Zoom | ⌘N ⌘O ⌘S ⇧⌘S | New, Open, Save, Save As / Export |

Scroll pans, and ⌘/Alt + scroll or a trackpad pinch zooms about the cursor. Middle-drag or
holding Space pans with any tool.

## Layout

```
src/engine/   UI-independent pixel engine
  px.rs         SWAR premultiplied RGBA8 math
  grid.rs       COW sparse tile grids, region reads
  mips.rs       mip pyramids, incremental updates
  blend.rs      18 blend modes
  selection.rs  shape / mask selections with per-tile classification
  brush.rs      dab engine (Kernel trait: paint, erase, clone, heal, tone, focus, smudge, mixer…)
  masks.rs      MaskGrid booleans, EDT expand/contract, feather, border, smooth, quick mask
  layers.rs     layer kinds, layer masks, smart objects
  shape.rs      live shape layers: components + path operations, rotation, paints
  pathops.rs    exact polygon booleans (snap rounding, integer predicates)
  frame.rs      frame layers (shape + clipped placed content)
  artboard.rs   artboard layers (presets, clipping, per-artboard export)
  arrange.rs    Align (to selection / key object) and Distribute
  slices.rs     layer-based slices, Divide Slice, Save for Web (images + HTML)
  measure.rs    Measurement Log: ruler / protractor / count / selection records, scale, CSV
  jpeg.rs       baseline JPEG encoder with optimized Huffman tables
  vanish.rs     Vanishing Point planes, tear-off, copy / stamp / brush / place in perspective
  presets.rs    brush preset library (folders, .pfb persistence, .abr import via abr.rs)
  prefs.rs      user config directory and preferences (note author)
  webexport.rs  PNG-8 quantizer (Oklab median cut + k-means, dither), encoders, PDF, HTML table
  vector.rs     paths, flattening, exact-area rasterizer, stroker, curve fitting
  text.rs       text layers: style runs, paragraph styles, own line breaker (justify,
                hyphenation), area / path / vertical layout, swash glyphs, caret / hit / ranges
  warp.rs       Photoshop's 15 warp styles as bicubic Bézier patches (with distortions)
  hyphen.rs     Liang (TeX) hyphenation with the en-US patterns in assets/hyph
  shapes.rs     custom shape library
  graphcut.rs   Boykov–Kolmogorov max-flow
  segment.rs    GrabCut, quick selection, guided-filter edges (Object / Quick Selection)
  livewire.rs   magnetic lasso and content-aware tracing
  poisson.rs    4-lane multigrid Poisson solver (healing, patch, seamless blends)
  patchmatch.rs multi-scale PatchMatch completion with rotation / scale / mirror / colour
                adaptation, sharp final pass and focus prior (healing, patch, Content-Aware Fill)
  retouch.rs    clone / pattern / history / heal kernels, spot heal (incl. Create Texture),
                patch, red eye, Content-Aware Fill sampling + fill
  objects.rs    Select Subject (boundary prior / model), object finder (FH segmentation)
  ml.rs         optional ONNX Runtime backend (`ml` feature), model discovery, depth / saliency
  color.rs      sRGB → Lab (D50, Bradford) / HSB / naive CMYK, Info readout modes
  metrics.rs    SSIM, CIEDE2000, IoU
  paint.rs      toning, focus, smudge, mixer, color replacement, erasers
  gradient.rs   gradients (5 kinds, stops, dither)
  annot.rs      samplers, notes, counts, artboards, slices, ruler / perspective math
  xform.rs      layer and canvas transforms (crop, rotate, perspective, free transform)
  crop.rs       crop resampling (bicubic / Lanczos), hidden pixels, content-aware crop fill, overlays
  filters/      ~70 filters: mod (dispatch), catalog (menus/params), fast (planar SIMD paths),
                blur, rank (O(1) median/bilateral), stylize, distort, util (LUTs, 3D LUT)
  ops.rs        fills, flood fill, gradient, transforms, strip blur
  composite.rs  layer compositing, display tiles, export
  document.rs   layers, document, history
  io.rs         image I/O, native .pforge format
  gpu/          wgpu compute path: device (context, health, budget), exec (pool, dispatch,
                watchdog), shaders + wgsl/ (pipelines, precompile), run (banded runner,
                CPU fallback), cost (Auto cost model, priors.txt), kernels/ (one per filter
                family), compose (GPU compositor → GPUI textures), resident (GPU image cache),
                preview, resample, poisson / patchmatch, stats, styles
src/ui/       GPUI front end
  viewport.rs   zoom/pan math, GPU display-tile cache
  docs.rs       document tabs, 2-up / tile, Match and Zoom / Scroll All
  dock.rs       Brushes, Brush Settings, Measurement Log and Notes panels
  editor.rs     state, input dispatch, jobs, dialogs
  tools.rs      the toolbar: tool list, groups, keys, help
  options.rs    options bar and bound sliders
  select_tools.rs / retouch_tools.rs / paint_tools.rs / vector_tools.rs / type_tools.rs /
  nav_tools.rs  tool families (input, overlays, options)
  layers_ui.rs  layers panel, masks, groups, adjustment / smart / place
  render.rs     widgets and canvas painting
  actions.rs    menus and key bindings
  caf.rs        Edit ▸ Content-Aware Fill workspace
  demo.rs       scripted tours (PF_DEMO)
  demo_gaps.rs  gap-closing tour (PF_DEMO=tools PF_DEMO_PHASE=A…G)
src/bench.rs  --bench, --bench-filters, --bench-tools
src/compare.rs --compare-gaps: quality / speed vs Photoshop outputs and ground truth
src/gtmask.rs --gt-mask: builds ground-truth masks from polygon files
testdata/     CC0 test photos and hand-made ground-truth masks
tools/        Photoshop comparison scripts and screenshot helpers
vendor/       GPUI with transformed image sprites (see vendor/README.md)
```
