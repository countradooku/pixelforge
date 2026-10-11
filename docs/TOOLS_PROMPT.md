# Prompt: implement the full Photoshop toolbar in PixelForge

You are working in `~/Desktop/Code/Rust/pixelforge`. It is a Rust + GPUI (gpui-kit 0.7.1) raster
editor built for images of hundreds of megapixels. Read `README.md` first, then the code listed
under "Architecture you must build on". Your job is to implement **every tool in Photoshop's
toolbar**, including each flyout sub-tool, at Photoshop quality or better and **measurably
faster than Photoshop 2026** (installed on this Mac and scriptable). Work in order of the phases
below, and keep the build warning-free with all tests passing after every phase.

## Architecture you must build on (do not reinvent)

- **Pixels:** `src/engine/grid.rs` holds `TileGrid`: 256² copy-on-write `Arc<Tile>` tiles.
  `None` means transparent, `uniform` means one flat color. Also there: `tile_mut`, `RectI`,
  `SyncPtr` for disjoint parallel tile writes, `build_grid`, and `read_region[_clamped]`.
  Pixels are premultiplied RGBA8 in a `u32`, and `src/engine/px.rs` has SWAR helpers
  (`scale`, `lerp`, `over`, `avg4`).
- **Mips:** `mips.rs` is updated incrementally from dirty rects. Every pixel edit must end in
  `Document::layer_pixels_changed(li, &rects)` or `set_layer_grid`.
- **Undo:** `document.rs` provides `push_undo(label)` *before* mutating. Snapshots are pointer
  copies, and old steps are LZ4-packed in the background.
- **Painting engine pattern:** `brush.rs` `Stroke`. The base snapshot plus a per-stroke u16
  coverage mask hold dabs, which are bucketed by tile and processed with rayon. A falloff LUT on
  d² avoids `sqrt`. Opacity caps the stroke; flow builds up. **Every new painting tool (clone,
  heal, dodge, smudge, …) must reuse or generalize this engine, not copy it.**
- **Persistent tool session pattern:** `liquify.rs` `Session` (base, a sparse per-tile field, and
  a version check against outside edits).
- **Selections:** `selection.rs` has `Selection`, `Shape` (Rect, Ellipse, Poly, `Mask(MaskGrid)`)
  and per-tile `TileCoverage` classification. **`ops.rs`** has `flood_mask` (a hierarchical
  tile-level BFS plus scanline), `map_tiles`, `fill_mask`, the remap/transform helpers and
  `apply_filter` (with block sizes chosen from padding).
- **Filters:** `filters/` contains `fast.rs` (planar vectorized blur passes, transposes,
  `gauss_plane`), `rank.rs` (O(1) histogram median and PBFIC bilateral), `distort.rs` (`Sampler`,
  homographies, mip-prefiltered gather) and `util.rs` (`Lut3d`, hashing, `fbm`).
- **Display:** `composite.rs` and `ui/viewport.rs` composite the screen at the zoom's mip level.
  `ComposeInput.extra` takes a display-only overlay layer. `Document.preview` holds a live
  filter preview.
- **UI:** `ui/editor.rs` has the `Tool` enum, `Drag` enum, `on_mouse_down/move/up`,
  `prepare_frame` (overlays, continuous tools), `spawn_job` / `run_layer_op` (background jobs
  with progress) and `start_demo` / `start_filter_demo` (`PF_DEMO`). `ui/render.rs` builds the
  per-tool options bar. `ui/actions.rs` has the menus, `RunFilter` and the key bindings.
- **Measurement:** `src/bench.rs` (`--bench`, `--bench-filters`) and `tools/*.jsx` (Photoshop
  timing via `osascript … do javascript`).

## The tools (Photoshop 2026 toolbar, with flyouts)

Existing tools to keep and upgrade where noted: Move, Rectangular / Elliptical Marquee, Lasso,
Magic Wand, Crop, Eyedropper, Brush, Eraser, Liquify, Paint Bucket, Gradient, Hand, Zoom.

1. **Move / Artboard.** Move gets Auto-Select layer (pick the topmost layer with alpha under the
   cursor), Show Transform Controls (on-canvas handles feeding the existing Free Transform
   homography), align/distribute, and moving selected pixels when a selection exists (float,
   commit, undo).
   **Artboard** is a document-level list of named rects; export each one.
2. **Marquees.** Add Single Row / Single Column. **All selection tools** get new / add /
   subtract / intersect (Shift / Alt / Shift+Alt), feather and anti-alias. Add `MaskGrid`
   boolean ops computed per tile; uniform tiles short-circuit. Add Select ▸ Modify
   (Expand / Contract / Border / Smooth / Feather) on `MaskGrid`, built from the existing O(1)
   max/min and Gaussian code.
3. **Lasso, Polygonal Lasso, Magnetic Lasso.** Magnetic Lasso is a live-wire / intelligent
   scissors on a gradient-magnitude cost. Use Dijkstra restricted to a band around the cursor,
   computed on the display mip level for speed and refined at level 0.
4. **Object Selection, Quick Selection, Magic Wand.**
   - **Quick Selection:** brush-driven region growing over SLIC superpixels or a graph cut on a
     coarse mip, refined at full resolution only near the boundary band.
   - **Object Selection:** a classical rectangle- or lasso-seeded GrabCut-style GMM plus graph
     cut, run at about 1 MP and upsampled with edge-aware refinement. Optional ONNX hook for
     BiRefNet (MIT), behind a feature flag; never required.
   - **Magic Wand:** add Sample All Layers and Anti-alias.
5. **Crop, Perspective Crop, Slice, Slice Select.** Crop gets on-canvas handles, rotation,
   aspect presets and Delete Cropped Pixels. Perspective Crop reuses `distort::square_to_quad`.
   Slices are metadata with export.
6. **Frame tool.** Rect / ellipse placeholder frames that clip a placed image. Implement it as a
   clipping-mask layer type (see "Shared infrastructure").
7. **Eyedropper, Color Sampler (up to 10 persistent points, shown in Info), Ruler (distance,
   angle, Straighten Layer), Note, Count.** Add sample size (point, 3×3 … 101×101 average) to the
   Eyedropper. Read box averages from the mip pyramid, so a 101×101 average is O(1).
8. **Spot Healing Brush, Healing Brush, Patch, Content-Aware Move, Red Eye.** This is the
   flagship; it must be excellent.
   - **Healing:** source texture plus a Poisson (Laplacian) solve for the boundary, using a
     **multigrid** solver on the dab's bounding region (V-cycles, f32 planes, SIMD-friendly). It
     must not be a naive Jacobi solver.
   - **Spot Healing / Content-Aware / Patch:** **PatchMatch** (Barnes et al.) with multi-scale
     coarse-to-fine over the mip pyramid, randomized search plus propagation, rayon
     parallelized by scanline parity. Run it only over the hole plus a margin; never a full-image
     pass.
   - **Red Eye:** detection inside the clicked region on the redness ratio, with a pupil mask
     and luminance-preserving desaturation.
9. **Brush, Pencil, Color Replacement, Mixer Brush.**
   - **Pencil:** aliased hard edge with Auto Erase.
   - **Color Replacement:** hue, color, saturation or luminosity modes with tolerance sampling.
   - **Mixer Brush:** wet / load / mix with a paint reservoir that picks up canvas color.
   - **All brushes:** pressure via `MousePressureEvent` (size / opacity / flow), smoothing
     (lazy-mouse plus Catmull-Rom resampling of input), angle, roundness, scatter and texture.
     Keep the per-event cost proportional to the dab area.
10. **Clone Stamp, Pattern Stamp.** Alt-click sets the source. Support aligned / non-aligned,
    a sample of current / current-and-below / all layers, and a source overlay preview in
    `prepare_frame`. Sampling reads the source through `TileGrid` with a constant offset, so
    plain row copies are possible when the offset is integer.
11. **History Brush, Art History Brush.** Paint from a chosen history state. Unpack only the
    tiles touched (packed history is per-tile LZ4), never the whole state.
12. **Eraser, Background Eraser, Magic Eraser.**
    - **Background Eraser:** sample the color under the hotspot (continuous / once /
      background swatch) with tolerance, and protect the foreground color.
    - **Magic Eraser:** `flood_mask` plus clear, sharing tiles exactly like the bucket.
13. **Gradient, Paint Bucket.** The gradient gets an editor (multiple stops plus opacity stops),
    linear, radial, angle, reflected and diamond types, dither, and on-canvas handles. The
    bucket gets Pattern fill and Sample All Layers.
14. **Blur, Sharpen, Smudge.** These are brush-driven local filters. Each dab reads a padded
    neighborhood from the stroke base and runs the planar kernels in `filters/fast.rs` only
    inside the dab. Smudge carries a pickup buffer along the path, with Finger Painting mode.
15. **Dodge, Burn, Sponge.** Range (shadows / midtones / highlights) and exposure, with
    Protect Tones. Sponge saturates or desaturates with Vibrance. Implement them as 256-entry
    LUT weights per range, so the cost is one LUT per pixel per dab.
16. **Pen, Freeform Pen, Content-Aware Tracing, Curvature Pen, Add / Delete / Convert Anchor
    Point, Path Selection, Direct Selection.** Paths are a document-level vector model (cubic
    Béziers, multiple subpaths) with a Paths panel. Add Stroke Path (drives the brush engine
    along the arc-length resampled path), Fill Path, and Make Selection (exact analytic coverage
    rasterization). Content-Aware Tracing is an edge-following trace from the magnetic lasso's
    cost map.
17. **Type: Horizontal / Vertical Type, Horizontal / Vertical Type Mask.** Text layers are
    *vector* layers re-rasterized on change: shaping with `cosmic-text` / `rustybuzz`, glyph
    outlines through `swash` or `ttf-parser`. Include font, size, tracking, leading, alignment,
    color, point and paragraph text, on-canvas editing, and Warp Text via the existing warps.
    Rasterize only the tiles the text bounds touch, and cache glyph coverage per (font, size,
    subpixel offset).
18. **Shapes: Rectangle (with corner radii), Ellipse, Triangle, Polygon (sides, star), Line
    (arrowheads), Custom Shape.** These are vector shape layers (fill, stroke, stroke alignment
    and dashes) rasterized with **exact area-coverage scanline AA** (signed-area accumulation),
    sparse per tile. Live-resize on the canvas.
19. **Hand, Rotate View, Zoom.** Rotate View rotates the *view* only. GPUI's `paint_image`
    cannot rotate, so add a rotated display path: composite visible tiles into a screen-aligned
    buffer through the inverse rotation from the mip pyramid (bounded by screen pixels), or draw
    textured quads via a custom element. Zoom gets scrubby zoom (drag) and Zoom All Windows.
20. **Edit Toolbar, foreground / background swatches, swap and reset (already exist).**
21. **Quick Mask mode (Q).** The selection becomes an editable red overlay layer (reuse
    `ComposeInput.extra`). Painting with any paint tool edits the mask, and exiting converts it
    back to a `MaskGrid` selection.
22. **Screen modes (F):** standard, full screen with menus, full screen.

Every tool also gets: a toolbar entry with a flyout (long-press or right-click, like
Photoshop), its Photoshop shortcut plus Shift-cycling within the group, an options bar, tooltips
with a one-line explanation, an on-canvas cursor or overlay, and one undo step per gesture.

## Shared infrastructure to add first (Phase 0)

- **Layer kinds:** pixel (existing), text, shape (vector), smart-object-lite (placed image plus
  a transform), group (pass-through or isolated, using `composite.rs` recursion), adjustment
  layers (reuse `Filter` as a non-destructive layer evaluated in `compose_region`), and layer
  masks and clipping masks (a `MaskGrid` per layer, applied in `compose_region` with per-tile
  Full/Empty/Partial classification so unmasked tiles cost nothing). Vector layers cache their
  rasterization as a normal `TileGrid` plus mips, re-rasterizing only dirty tiles.
- **A generalized dab engine** (`brush.rs`) with a pluggable per-pixel kernel trait:
  `fn apply(&self, base: TileView, mask_coverage, out)`. Brush, eraser, clone, history, dodge /
  burn / sponge, blur / sharpen, color replacement, pencil, background eraser and quick-mask
  painting are all kernels on this engine. Input smoothing and pressure live here once.
- **Vector core:** paths, an analytic coverage rasterizer (sparse, tile-parallel; pixels
  identical at tile seams), stroking (joins, caps, dashes) via `lyon` or your own offsetter, and
  hit testing.
- **Panels:** Paths, Character / Paragraph, Info (color samplers, ruler), and a brush presets
  panel.

## Performance requirements (non-negotiable; verify with benchmarks)

- **Never scale with image size when the edit is local.** Every tool's per-event cost must be
  O(affected pixels) and touch only the tiles it needs. Copy-on-write tiles stay shared;
  untouched tiles are never copied.
- **Interactive tools on a 268 MP document (`File ▸ New 256 MP Test Image`):**
  - Brush-engine tools with a 500 px dab: ≤ 2 ms per mouse event.
  - Clone / heal dab: ≤ 8 ms.
  - Spot Healing a 300 px blemish: ≤ 60 ms.
  - Content-Aware Move / Patch on 1 MP: ≤ 400 ms.
  - Quick / Object Selection: first result ≤ 150 ms (coarse level), refined ≤ 1 s.
  - Magnetic Lasso: under 4 ms per mouse move.
  - Text edits: under 4 ms per keystroke for a 2,000-glyph paragraph.
  - Shape resize: under 8 ms per frame.
- **Background jobs:** anything over ~30 ms runs through `spawn_job` with progress. Live
  previews always run at the display mip level.
- **Hot loops:** SIMD-friendly, planar `u8` / `u16` / `f32` with contiguous slices (follow
  `filters/fast.rs`: vectorized column passes and blocked transposes instead of serial row
  chains), rayon over tiles, no per-pixel allocation, scratch buffers reused per thread, and
  mimalloc already enabled. Profile with `/usr/bin/sample` (see the `PF_REPEAT` pattern in
  `bench.rs`) before declaring anything optimized.
- **Algorithms:** use the best known one, not the textbook one: multigrid Poisson, multi-scale
  PatchMatch, superpixels or graph cut at coarse resolution with boundary-band refinement,
  analytic AA rasterization, O(1) window statistics from the mip pyramid / integral images, and
  LUT-based tone kernels.

## Verification (required for every phase)

1. **Unit tests:** correctness against naive reference implementations (Poisson residual
   below a tolerance, PatchMatch NNF convergence, rasterizer coverage equal to the exact polygon
   area, selection boolean-op truth tables, a text layout snapshot). Keep
   `cargo test --release` green.
2. **Benchmarks:** add `--bench-tools` to `src/bench.rs`. It replays scripted gestures on a
   268 MP document and prints ms per event and per gesture, and every number is checked
   against the budgets above.
3. **Photoshop comparison:** extend `tools/*.jsx` with scriptable equivalents (Content-Aware
   Fill via `executeAction(stringIDToTypeID("contentAwareFill"))`, Spot Healing via recorded
   descriptors, text layer creation, shape creation, Select Subject). Time them on the same
   image sizes, and put an honest table in the README. Where Photoshop has no scripting hook,
   say so.
4. **Real UI check:** extend `PF_DEMO` with a `tools` scenario that drives each tool through
   the real input handlers (`demo_drag`, `on_mouse_down`, …). Screenshot the window by ID
   (`screencapture -l`) and visually check each result. Synthetic OS input is blocked, so drive
   the handlers directly. Do not claim a tool works without a screenshot of it working.
5. **README:** update the feature list, shortcuts, benchmarks and a "Not implemented / differs
   from Photoshop" section. Be explicit about anything approximated.

## Phases

0. Shared infrastructure: layer kinds, masks, generalized dab engine, vector core, selection
   boolean ops.
1. Selection tools: marquee variants, lassos, Quick / Object Selection, Select ▸ Modify, Quick
   Mask.
2. Retouching: Clone / Pattern Stamp, History brushes, Healing family plus Content-Aware Move
   and Patch, Red Eye.
3. Painting: Pencil, Color Replacement, Mixer, Blur / Sharpen / Smudge, Dodge / Burn / Sponge,
   Background / Magic Eraser, gradient editor, bucket patterns.
4. Vector: Pen family, paths panel, shapes, Type tools and text layers, Frame tool.
5. Navigation and measurement: Rotate View, scrubby zoom, Color Sampler, Ruler / Straighten,
   Note, Count, Artboards, Slices, Perspective Crop, crop handles, screen modes, Edit Toolbar.

At the end of each phase, report what was built, the benchmark numbers against the budgets, the
Photoshop comparison, the screenshots you checked, and anything deferred and why.
