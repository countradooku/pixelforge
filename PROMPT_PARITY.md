# Prompt: Feature parity with Photoshop 2026, with a better UX

Run this **after** `PROMPT_PSD.md` is done: PSD and PSB are the native format, documents can be
8, 16 or 32-bit, HDR and raw work, and memory and speed are tuned per size tier.

You are working in `~/Desktop/Code/Rust/pixelforge`. It is Rust with GPUI via `gpui-kit` 0.7.1 /
`gpui-pre` 0.3.8, vendored and patched in `vendor/`. It is not a git repo, so back up the tree to
the scratchpad before you start, and again before each phase.

Your job: **close every remaining feature gap with Photoshop 2026**, except the AI features
listed below, and give every feature **a better user experience than Photoshop's**. This is not
a reskin of Photoshop. Keep what Photoshop users rely on (terminology, default shortcuts, the
document / layer / mask / adjustment model, PSD semantics). Redesign how people reach and use
it, so a Photoshop user feels at home in an hour, and everyone does common work in fewer steps,
with less waiting and fewer modal dialogs.

Photoshop 2026 is installed and scriptable (`tools/photoshop_*.jsx`, with the dialog watchdog
from the PSD work). It is the reference for **what** exists and for result fidelity, never for
**how** the UI looks.

## Out of scope: AI features (for now)

Leave out:
- Generative Fill, Generative Expand, Generate Image, Generate Background and Generative
  Upscale;
- Neural Filters that need trained models;
- Remove tool's and Distraction Removal's AI modes;
- Sky Replacement's sky generation;
- Select Subject, Object Selection, Select Sky and Select People in their AI forms;
- Match Font;
- Harmonize, and anything else Adobe labels as Firefly or Sensei.

Classical versions that already exist (GrabCut Select Subject, PatchMatch Content-Aware Fill,
classical Neural Filters) stay and can be improved, but no new trained models or downloads.
Every out-of-scope item still appears in the parity matrix, marked "AI, deferred", so nothing is
silently missing.

## Ground rules

- **Dependencies:** ask me before any new crate or download (name, source, version, size,
  licence). This includes scripting engines, video codecs, colour-management libraries, fonts
  and test assets. Prefer what's already in `Cargo.lock`.
- **No Adobe trade dress:** don't copy Adobe's icons, artwork, panel visuals, sample content,
  textures, presets or proprietary data files.
  - Use your own icons, or the icon set already in the project, and your own presets.
  - Keep names and shortcuts that are industry vocabulary (Levels, Curves, ⌘J, ⌘T, ⌥-drag…)
    so muscle memory works.
- **Everything else stays as it is:**
  - The CPU path is the reference and the GPU path matches it.
  - PSD round trips stay lossless (every new feature gets its PSD mapping and joins the
    `--psd-corpus`).
  - All bit depths are supported, or the feature is greyed out with a reason.
  - Benchmark budgets hold.
  - The build is warning-free and `cargo test --release` is green with `PF_GPU=off` and
    `force`.
- **Proof bar:** never claim a feature works without:
  - a test;
  - a `PF_DEMO=tools` screenshot of the feature in use;
  - a Photoshop comparison of the result wherever the result is measurable (SSIM / ΔE00 / IoU /
    timing, by jsx).
  Look at every screenshot and fix what's wrong, including UX problems: clutter, clipping,
  inconsistency, unclear state.
- **UX claims are measured too** (see Phase 1).

## Phase 0: The parity inventory

You can't close gaps you haven't listed. Build `PARITY.md` and a machine-readable
`docs/parity.json` from Photoshop itself, not from memory.

1. **Sources**, in this order:
   - (a) **Photoshop's own command list.** Edit ▸ Keyboard Shortcuts ▸ Summarize exports every
     application menu command, panel menu command and tool to HTML. Try to script it. If it
     can't be scripted, ask me to export it once (Application Menus, Panel Menus, Tools,
     Task Spaces) and give you the files.
   - (b) **Photoshop's scripting surface.** Enumerate menu commands through the UXP `core` API
     (menu command titles and states by ID) if available from a UXP script, plus the event IDs
     used in the existing jsx files.
   - (c) **The Photoshop User Guide** table of contents on helpx.adobe.com: every tool, panel,
     dialog, option and workflow page.
   - (d) **Every tool's options bar** and **every panel's flyout menu**, opened by script
     where possible, otherwise taken from (a).
   - (e) **Preferences**, every pane and option.
   - (f) **File formats** in Open / Save As / Export.
2. **One row per item** (command, tool, tool option, panel, panel option, dialog option,
   preference, file format, workflow):
   - its Photoshop location;
   - PixelForge status: done / partial / missing / AI-deferred / obsolete (removed from
     Photoshop, e.g. 3D, which Adobe discontinued; confirm);
   - the evidence (test name, screenshot, comparison number);
   - for anything not done, the plan.
   - "Partial" must say exactly what's missing: options, bit depths, PSD mapping, GPU path,
     preview.
3. **Check what already exists** against the code and README, not the README alone: open the
   feature in the app, use it, take a screenshot.
4. **Prioritise:** frequency of use (common photo, design and retouch workflows first), then
   dependency order, then effort.
5. **Report** the totals per category and the top gaps before Phase 1.

Known gap candidates, to verify in Phase 0 (not a substitute for it):

- **Panels:** Channels, Color, Swatches, Gradients, Patterns, Styles, Shapes, Character,
  Paragraph, Character and Paragraph Styles, Glyphs, Navigator, Layer Comps, Clone Source, Tool
  Presets, Actions, Timeline, Adjustments presets, Histogram (check), Comments, Libraries
  (local only; no cloud).
- **Colour:**
  - management: ICC profiles, Assign / Convert to Profile, Proof Setup, soft proofing and
    gamut warning;
  - editing in Lab, CMYK and Grayscale modes; Duotone, Indexed and Bitmap conversions;
  - the full Color Picker (HSB / Lab / CMYK / hex, colour libraries), Color Settings, and
    printing with colour management.
- **Non-destructive editing:**
  - Smart Filters on smart objects (any filter, re-editable, masked, blend and opacity per
    filter), Convert for Smart Filters;
  - linked smart objects and Edit Contents;
  - layer comps; adjustment presets.
- **Selection:** Color Range (sampled colours, highlights / midtones / shadows / skin tones,
  localized clusters), Focus Area (classical, from local sharpness), Grow, Similar, Transform
  Selection, Save / Load Selection to channels, selection from channels.
- **Image and Layer commands:** Apply Image, Calculations, Trim, Canvas Size (with anchor),
  Image Rotation variants, Duplicate, Variables / Data Sets, Auto-Align Layers, Auto-Blend
  Layers (panorama and focus stack), Stack Modes (median, mean…), Photomerge, Load Files into
  Stack, Merge Visible / Stamp Visible, Layer ▸ Matting (Defringe, Color Decontaminate),
  Arrange, Lock options.
- **Edit:**
  - transforms: Puppet Warp, Content-Aware Scale (with protect mask), Content-Aware Move
    (check), Transform ▸ Warp split modes (check), Auto-Align;
  - commands: Define Brush / Pattern / Custom Shape, Paste Special (Paste in Place, Paste Into,
    Paste Outside), Purge;
  - customisation: Keyboard Shortcuts editor, Menus customisation, Presets manager and
    migration.
- **Filters:** Shake Reduction, Offset, HSB/HSL, Render ▸ Flame / Picture Frame / Tree, Video
  filters (De-Interlace exists; NTSC Colors), Pattern Preview (View), Other ▸ Custom (check
  options), any filter in the inventory not yet present.
- **Type:** Character and Paragraph panels in full (OpenType features, kerning methods,
  composers, hyphenation settings, justification), Character and Paragraph Styles, Glyphs, Find
  and Replace Text, Check Spelling (ask before any dictionary download), Convert to Shape /
  Work Path, variable fonts.
- **Automation:**
  - Actions: record and play every command with its settings, conditional actions, modal
    controls, action sets saved and loaded;
  - Batch, Droplets (here: a CLI batch runner), Image Processor, Scripts (a scripting API:
    choose the engine with me), Generate ▸ Image Assets, Export layers and artboards to files.
- **Timeline:**
  - frame animation (frames, tweening, onion skin, GIF export);
  - video timeline and video layers (ask before any codec dependency; at minimum import an image
    sequence and export an image sequence or GIF).
- **Export and print:** Export As (formats, scale, multiple sizes, metadata, colour space),
  Quick Export, Artboards to PDF / files, PDF Presentation, Contact Sheet, Print (page setup,
  colour management, marks, bleed).
- **Input:**
  - pen tilt, rotation, eraser end and barrel buttons: GPUI doesn't report them yet; patch the
    vendored GPUI on macOS through `NSEvent` tablet events;
  - trackpad gestures: pinch zoom, rotate and two-finger pan;
  - fully customisable shortcuts.
- **View:** Proof Colors, Gamut Warning, Pixel Grid, Smart Guides, Guide Layout, New Guides
  from Shape, Snap To options, Extras, multiple windows per document, Navigator, Match Location.
- **Existing "differs" items:** Filter Gallery textures (let users load their own texture
  images), Type composer (improve line breaking against Photoshop's on a larger corpus), Rotate
  View on Windows (still blocked by the declined target; keep it as is).

## Phase 1: The UX foundation (before adding features)

Design these once so every later feature uses them. Write `docs/UX.md` with the principles,
components and rules, and keep it current.

1. **Principles** (measurable where possible):
   - **Non-modal by default.** Adjustments, filters and transforms edit live, on the canvas,
     with the document still usable (zoom, pan, compare). A modal dialog needs a written
     reason.
   - **Preview is never slower than 1 frame for small documents and the 61 MP tier's budget for
     large ones.** Previews never block input.
   - **Non-destructive first.** Filters and adjustments offer "as a layer / smart filter" as the
     default where Photoshop makes them destructive, with one shortcut to apply destructively.
   - **One inspector:** a single contextual Properties panel for the active layer, tool,
     selection, adjustment, filter or object, instead of dozens of dialogs with different
     layouts.
   - **Everything is findable:** a command palette (⌘K) searches every command, tool, panel,
     preference, preset, recent action and help topic, shows the shortcut, and can run
     anything. Menus still exist and use Photoshop's grouping where it's sensible.
   - **Before / after everywhere:** split view, toggle and side by side for any preview.
   - **Undo everything** (including panel and preference changes where sensible), with named
     history steps.
   - **State is always visible:** active tool, mode, bit depth, colour mode, selection, mask
     editing, quick mask, smart object context, current target (layer, mask, filter mask).
   - **Defaults that work:** sensible defaults with presets that show live thumbnails on the
     current image.
   - **Keyboard-first and accessible:** every action reachable by keyboard, focus visible,
     scalable UI (75–200 %), colour-blind-safe status colours, contrast meeting WCAG AA for
     text.
2. **Components:**
   - Build the inspector, property rows, numeric scrubbers (drag on labels, arithmetic in
     fields, units), curve and gradient editors, colour fields, preset grids, the command
     palette, toasts, progress (cancellable everywhere) and the on-canvas widget system
     (handles, gizmos, HUDs).
   - Retrofit the existing dialogs (filters, adjustments, Image Size, Liquify, Blur Gallery,
     Camera Raw, Select and Mask, Content-Aware Fill, Vanishing Point, Perspective Warp) onto
     them.
3. **Workspaces:**
   - A dockable panel system: dock, undock, tab, collapse, stack; saved named workspaces;
     a per-task default set (Photography, Design, Painting, Retouch); reset.
   - Panels remember their state per workspace.
4. **Preferences:** one searchable preferences window with every option from the inventory that
   applies, grouped by task, not by history. Each preference shows what it affects. Import
   Photoshop's settings where the format is readable (shortcuts, swatches, brushes, gradients,
   patterns, styles, actions); ask me for sample files from my Photoshop install, or generate
   them with Photoshop by script.
5. **Onboarding:**
   - First run asks "coming from Photoshop?" and keeps Photoshop shortcuts and terminology on by
     default.
   - A "What's this?" hover help that names the Photoshop equivalent.
   - No tutorial wall.
6. **UX measurement:**
   - For each common task in a task list (a list of 30 or more in `docs/UX.md`), drive the task
     through the real input handlers in a demo script and count the steps: clicks, keystrokes
     and dialogs opened, and the time spent waiting.
     - Examples: retouch a portrait, cut out a subject with hair and add a shadow, batch-resize
       100 images, make a social post from a template, colour-grade a photo, focus-stack five
       shots, make a panorama, prepare a CMYK print file, animate a GIF.
   - Do the same tasks in Photoshop by documented steps (and by script where possible, for
     timings).
   - Report steps and waiting time side by side. PixelForge must need fewer or equal steps
     for every task, and fewer on most. Explain each exception.
7. **Heuristic review:** after each phase, review the new UI against Nielsen's ten heuristics
   and your own principles. List the problems found and fixed, with before / after
   screenshots.

## Phase 2: Panels and colour

1. Build every missing panel from the inventory on the Phase 1 components, with PSD mappings
   (swatches, gradients, patterns and styles also read Photoshop's preset files `.aco`, `.ase`,
   `.grd`, `.pat`, `.asl`, `.csh` and `.tpl` where the formats are documented or
   well-understood; ask me for samples).
2. **Colour management:**
   - Use ICC profiles end to end: document profile, working spaces, display profile (macOS
     ColorSync), and rendering intents with black-point compensation.
   - Assign / Convert to Profile, Proof Setup with soft proofing and gamut warning, Color
     Settings.
   - Propose the colour engine (in-repo or a crate; ask), and compare conversions against
     Photoshop's on a colour-target image (ΔE00 ≤ 1 mean, ≤ 3 max for RGB↔CMYK with the same
     profiles and intent).
3. **Colour modes:**
   - Edit in Lab, CMYK and Grayscale at every bit depth, with the tools and filters Photoshop
     allows in each mode (check by script).
   - Duotone, Indexed and Bitmap conversions with Photoshop's options.
4. **Color Picker:** HSB, RGB, Lab, CMYK and hex; colour libraries where licensing allows
   (don't ship proprietary libraries such as PANTONE data; let users load their own); an eye
   dropper anywhere on screen, which needs a macOS screen-capture permission (explain it and
   ask before requesting).

## Phase 3: Non-destructive editing

1. **Smart Filters:**
   - Any filter on a smart object, with a stack per object; each filter has a blending mode,
     opacity and visibility; a shared filter mask; reorder; re-edit live.
   - GPU-accelerated through the existing kernels, and cached so editing one filter doesn't
     recompute the ones below.
   - PSD round trip.
2. **Smart objects:** Edit Contents (opens the embedded document as a linked tab; saving updates
   the parent), linked smart objects (relink, update modified, package), convert to layers,
   replace contents, export contents, and the stack modes used by Load Files into Stack.
3. **Layer comps, adjustment presets,** and "apply as smart filter" by default (from the UX
   principles).

## Phase 4: Missing tools, commands and filters

Implement everything left in the inventory's tool, command and filter categories, in the
priority order from Phase 0. Each item needs:

- the result compared against Photoshop by jsx where it's measurable;
- every bit depth and colour mode Photoshop supports for it;
- its PSD mapping;
- a GPU path where `--bench-gpu` shows a win;
- its benchmark row with a budget;
- a screenshot of the redesigned UI.

Examples: Puppet Warp, Content-Aware Scale, Color Range, Focus Area, Apply Image, Calculations,
Auto-Align and Auto-Blend Layers (panorama and focus stacking), Photomerge, Shake Reduction,
Render ▸ Flame / Picture Frame / Tree, Offset, HSB/HSL, Paste Special and Define presets.

## Phase 5: Automation

1. **Actions:**
   - Every undoable command is recordable with its parameters (the command model already exists
     through history labels; make it a real command bus), including tool strokes where
     Photoshop records them.
   - Playback with modal controls (stop at a dialog), conditional steps and insert-menu-item.
   - Action sets save and load in PixelForge's own format, and import Photoshop's `.atn` where
     the commands map, with a report of unmapped steps.
2. **Batch and Droplets:** Batch over folders with naming rules and error logs. A CLI
   `pixelforge --batch <action> <in> <out>` plays the role of Droplets.
3. **Image Processor, Export Layers / Artboards to Files, Generate ▸ Image Assets,** and
   Variables / Data Sets (CSV-driven templates).
4. **Scripting:** a documented scripting API over the same command bus (propose the engine,
   e.g. an embedded JavaScript or Lua runtime, with its licence and size, and ask). It needs
   enough coverage to rewrite the repo's own demo and benchmark drivers as scripts.

## Phase 6: Timeline and animation

1. **Frame animation:** a frames timeline, per-frame layer visibility, position and opacity,
   tweening, onion skin, timing, loop count, and animated GIF / APNG / WebP export.
2. **Video timeline:** layer keyframes (position, opacity, style, transform), image-sequence
   import and export, and video layers through a codec dependency, which must be approved by me
   first. If not approved, report the gap.
3. **PSD mapping** of the timeline and animation data (preserved byte-exact if not
   interpreted).

## Phase 7: Export, print and input

1. **Export As and Quick Export:** formats, multiple scales and suffixes, colour space
   conversion, metadata options, size preview; Artboards to PDF / files; PDF Presentation;
   Contact Sheet.
2. **Print:** page setup, scaling, position, colour management (printer profile, rendering
   intent, proof), marks, bleed, and a print preview through macOS printing.
3. **Input:** pen tilt, rotation, eraser end and buttons through a vendored GPUI patch (NSEvent
   tablet events), brush dynamics using them, trackpad pinch, rotate and pan gestures, and a
   full shortcut editor with conflict detection and import of Photoshop `.kys` files where
   possible.

## Verification per phase

1. **Tests:** every feature, every bit depth and colour mode it supports, its PSD round trip
   (in `--psd-corpus`), and the GPU against the CPU where there's a GPU path.
2. **Photoshop comparisons** by jsx, with the dialog watchdog: fidelity (SSIM / ΔE00 / IoU) and
   timing, in honest README tables.
3. **UX:** the task-step comparison table and the heuristic review, with screenshots.
4. **`PF_DEMO=tools` phases** for every new feature, through the real input handlers. Look at
   every screenshot.
5. **`PARITY.md` updated:** every row done, AI-deferred, or obsolete with evidence, or blocked
   with a precise reason outside the code.
6. **README:** features, panels, colour management, automation, timeline, export and print, the
   UX section (principles, command palette, workspaces, task-step table), benchmark tables, and
   "Not implemented / differs" limited to AI-deferred items and things outside the code.

## Reporting

After each phase, report:
- the parity totals per category (done / partial / missing / AI-deferred / obsolete) and what
  changed;
- what was built, with its fidelity and timing against Photoshop and the budgets;
- the UX measurements (steps and waiting per task against Photoshop) and the heuristic review
  findings;
- which screenshots you checked and what they showed;
- anything deferred or blocked, with the reason.

## Sources

- Adobe, Photoshop User Guide (table of contents for the inventory):
  https://helpx.adobe.com/photoshop/desktop.html
- Adobe, keyboard shortcuts; Edit ▸ Keyboard Shortcuts ▸ Summarize writes every command and shortcut, assigned or not, to an HTML file (described by third-party guides, e.g. https://photoshoptrainingchannel.com/tips/view-print-list-keyboard-shortcuts/):
  https://helpx.adobe.com/photoshop/desktop/get-started/settings-and-preferences/view-keyboard-shortcuts.html
- Adobe, UXP for Photoshop (scripting surface, menu commands):
  https://developer.adobe.com/photoshop/uxp/2022/
- Adobe, Photoshop File Format Specification (preset and settings formats):
  https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/
- Nielsen Norman Group, 10 usability heuristics:
  https://www.nngroup.com/articles/ten-usability-heuristics/
- W3C, WCAG 2.2 (contrast, keyboard access): https://www.w3.org/TR/WCAG22/
