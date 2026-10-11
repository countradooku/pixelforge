# Prompt: PSD as the native format, 16-bit and 32-bit documents, HDR, raw, and memory

You are working in `~/Desktop/Code/Rust/pixelforge`. It is Rust with GPUI via `gpui-kit` 0.7.1 /
`gpui-pre` 0.3.8, vendored and patched in `vendor/`. It is not a git repo, so back up the tree to
the scratchpad before you start, and again before each phase.

Today PixelForge:

- is **8 bits per channel** throughout: premultiplied RGBA packed in one `u32` (`engine/px.rs`
  SWAR math), 256² COW tiles (`engine/grid.rs`), per-layer mips, and a wgpu GPU path
  (`engine/gpu/`) that reproduces the CPU bit-exactly;
- saves documents in its own **`.pforge`** format (`engine/io.rs` `save_native` / `load_native`)
  and imports and exports only flat PNG, JPEG, TIFF, WebP and BMP through the `image` crate;
- has **no PSD support, no 16 or 32-bit documents, no HDR and no raw decoding**. The Camera Raw
  Filter works on 8-bit layer pixels.

Your job:

1. **Replace `.pforge` with PSD / PSB** as the document format, with complete Photoshop
   compatibility in both directions (defined below).
2. **16-bit and 32-bit float documents** (Image ▸ Mode), like Photoshop.
3. **HDR**: 32-bit linear documents, HDR display on EDR screens, and HDR file formats.
4. **Raw files**: open camera raw and DNG files into a 16 or 32-bit document.
5. **Memory and speed** at realistic sizes: tiered benchmarks, memory budgets, compressed tiles
   and a scratch disk, so 16 and 32-bit work stays fast.

Photoshop 2026 is installed and scriptable (`tools/photoshop_gaps.sh`, `tools/photoshop_gaps.jsx`,
`tools/photoshop_tools.jsx`). It is the reference for file compatibility, rendering, behaviour
and speed. Known scripting quirks are recorded in the memory notes and README: scripted Camera
Raw needs `CMod = "Filter"` and custom settings, or its dialog blocks the script; after the
Camera Raw filter runs, opening a TIFF can be intercepted by the Camera Raw dialog. Every
Photoshop run needs a watchdog that detects dialog windows (CoreGraphics window list), because a
modal dialog blocks the scripting bridge.

## What "complete PSD compatibility" means here

It's a measurable bar, not a slogan. A feature is done only when all of these hold:

1. **Read:** every PSD and PSB that Photoshop 2026 writes opens in PixelForge. Every feature
   PixelForge has is editable (layer, mask, style, text, shape, adjustment, smart object…). The
   composite matches Photoshop's within the bars below.
2. **Write:** every file PixelForge writes opens in Photoshop with no warning, no "some data
   could not be read" or compatibility dialog, and no rasterised or missing layer. Photoshop's
   rendering of it matches PixelForge's within the bars below.
3. **Lossless round trip:** PS → PixelForge → save → PS loses nothing.
   - Data PixelForge interprets is re-encoded equivalently.
   - Data it doesn't interpret is preserved byte-exact and written back: unknown image
     resources, unknown additional layer information keys, unknown descriptor keys inside known
     blocks, Adobe-private blobs such as Liquify meshes, smart filter internals and generator
     settings.
   - A test proves this with a byte-level structural diff of the parsed blocks.
4. **Edits stay safe:** editing one layer doesn't drop or corrupt the preserved data of others.
   When an edit makes preserved data stale (for example, pixels of a smart object changed while
   its smart filters are preserved but not interpreted), handle it the way Photoshop's own
   "data could not be updated" cases work: keep the data, warn in the Info panel, never write
   an inconsistent file.
5. **Everything Photoshop stores, PixelForge stores:** all colour modes, bit depths, channels,
   paths, guides, slices, annotations, metadata (XMP, EXIF, IPTC), ICC profiles, print settings,
   timeline and animation data, comps.
   - PixelForge must at least read and preserve all of it.
   - It must render what it can, at least through the stored composite.
   - Where PixelForge lacks the feature (3D, video layers, CMYK editing…), the layer shows its
     stored pixels, is marked read-only in the Layers panel, and round-trips byte-exact.

Bars:

- Pixel layers, masks, blend modes, opacity / fill, clipping, groups: **bit-exact** composites
  at 8 and 16-bit, against Photoshop's flattened export, on the test corpus.
- Adjustment layers, fill layers, layer styles, shapes, text: SSIM ≥ 0.995 and mean ΔE00 ≤ 0.5
  against Photoshop's flattened export. Report every miss with the reason. Text line breaks are
  the known exception (Adobe's composer is proprietary), so measure them separately.
- 32-bit: per-channel relative error ≤ 1e-3 for pixel layers, against Photoshop's 32-bit export
  (EXR or 32-bit TIFF).

## What the research says (read this first; confirm on the current pages)

**The PSD format.**

- Adobe publishes the *Photoshop File Format Specification*. Its sections: header, colour mode
  data, image resources, layer and mask information, image data.
- PSD is limited to 30,000 px per side and 2 GB; PSB to 300,000 px per side.
- Compression types: raw, RLE (PackBits, with per-row byte counts), ZIP without prediction, ZIP
  with prediction (delta per row; for 32-bit, bytes are interleaved before deflate).
- Colour modes: Bitmap, Grayscale, Indexed, RGB, CMYK, Multichannel, Duotone, Lab.
- Photoshop's 16-bit mode stores and computes 0…32768 ("15+1-bit"), not 0…65535. Its 32-bit
  mode is linear float.
- Layers live in "additional layer information" blocks keyed by four characters. Important ones:
  - Names and structure: `luni` (Unicode name), `lsct` / `lsdk` (group dividers), `lyid`.
  - Styles: `lfx2` / `lmfx` (layer styles as descriptors).
  - Text: `TySh` (text, whose `EngineData` is a separately structured, *undocumented* blob).
  - Smart objects: `PlLd` / `SoLd` / `SoLE` (placed and smart-object layers, smart filters inside
    the descriptor); `lnk2` / `lnk3` / `lnkD` (embedded and linked files).
  - Vectors: `vmsk` / `vsms` (vector masks), `vogk` (live-shape origination), `vscg` / `vstk`
    (shape fill and stroke).
  - Other: `artb` / `artd` (artboards), `Lr16` / `Lr32` / `Mt16` / `Mt32` (16 and 32-bit layer
    data), `FMsk` (filter mask), `shmd` (metadata), `cinf`, `CgEd`, `Txt2`, `Patt` / `Pat2` /
    `Pat3` (patterns).
  - Fill layers: `SoCo` / `GdFl` / `PtFl`.
  - Adjustment layers: `levl`, `curv`, `brit`, `hue2`, `blnc`, `blwh`, `mixr`, `selc`, `grdm`,
    `phfl`, `expA`, `vibA`, `post`, `thrs`, `nvrt`, `clrL`.
- Image resources include:
  - 1005 resolution, 1039 ICC profile, 1060 XMP, 1058 EXIF.
  - 1050 / 7000+ slices, 1032 guides, 2000–2997 paths, 2999 clipping path.
  - 1065 layer comps, 4000+ plug-in data, 1036 thumbnail.
- Descriptors (`ActionDescriptor` binary) appear in many blocks. One correct descriptor
  reader and writer (all types: `obj `, `Objc`, `VlLs`, `doub`, `UntF`, `TEXT`, `enum`, `long`,
  `comp`, `bool`, `GlbO`, `type`, `GlbC`, `alis`, `tdta`, `ObAr`, `UnFl`, `Pth `) covers most of
  them. Keep key order and unknown keys.

**Prior art** (read for structure and edge cases; don't depend on them, and don't copy GPL code).

- **psd-tools** (Python, MIT): the most complete open reader. It documents most blocks, blend
  modes and compositing, including where its compositing differs from Photoshop.
- **ag-psd** (TypeScript, MIT): reads *and writes* layers, effects, text (including
  `EngineData`), smart objects, vector shapes and artboards, and documents what Photoshop
  accepts when writing.
- **Photopea**: the best-known third-party PSD compatibility (closed source). It shows what is
  achievable.
- **GIMP / Krita**: partial; text and styles are rasterised or lost. That's the failure mode to
  avoid.
- **Rust `psd` crate**: read-only, pixels and basic layers only. Don't depend on it.

**Bit depths and HDR.**

- Photoshop Image ▸ Mode: 8, 16 (0…32768) and 32 bits per channel float (linear).
  - In 32-bit, many filters and some tools are unavailable. Photoshop shows a 32-bit Preview
    exposure/gamma control and offers HDR Toning and Merge to HDR Pro.
  - Confirm the exact list of 32-bit-capable filters and tools in Photoshop 2026 by checking the
    menus by script.
- Photoshop's HDR display is a Technology Preview ("precise color management for HDR display").
  Camera Raw can edit in HDR and save AVIF / JPEG XL with gain maps. Gain maps are ISO 21496-1,
  and are also used by Apple and Android HDR photos.
- macOS EDR: `CAMetalLayer.wantsExtendedDynamicRangeContent`, an `rgba16Float` layer in an
  extended-linear colour space (for example Display P3), and
  `NSScreen.maximumExtendedDynamicRangeColorComponentValue` for the current headroom.
  Screenshots may clip EDR content to SDR, so verify HDR output from the texture values as well.

**Raw.**

- Adobe's DNG specification is public.
- Proprietary raws (CR2, CR3, NEF, ARW, RAF, ORF, RW2…) need a decoder library. Candidates:
  `rawler` (Rust, from dnglab, LGPL-2.1, CR3 support), `rawloader` (Rust, LGPL-2.1), and LibRaw
  bindings (C, LGPL-2.1 / CDDL). Check versions, maintenance, size and licence before asking
  me.
- Development: black / white levels, white balance, demosaic (implement from the published
  papers, e.g. RCD or AMaZE descriptions, without copying GPL code), camera-to-working-space
  matrix, highlight reconstruction, then the existing Camera Raw controls in linear float.

**Memory** (how others stay fast).

- Photoshop: tile cache levels and tile size preferences, a scratch disk, compressed history.
- Affinity: GPU-resident documents, with 16-bit roughly doubling VRAM needs.
- darktable: tiling plus a reserved headroom.
- PixelForge's own lesson (README, GPU section): a full-budget GPU pool on unified memory pushed
  macOS into paging, and a 250 ms dispatch past the 2 s watchdog. Memory budgets must cover CPU
  tiles, history, mips and the GPU together.

## Ground rules

- **Dependencies:** ask me before any new crate or download (name, source, version, size,
  licence), including adding an existing transitive crate (for example `flate2`) as a direct
  dependency. Expect to ask for:
  - deflate for PSD ZIP;
  - EXR;
  - AVIF / JPEG XL with gain maps;
  - a raw decoder;
  - possibly a fast compressor for cold tiles and the scratch disk.
  Prefer what's already in `Cargo.lock`. The PSD/PSB reader and writer and the descriptor codec
  are written in this repo, not taken from a crate.
- **Replace the format:**
  - Save / Save As / Open use PSD (PSB when a side exceeds 30,000 px, or when the user picks it).
  - `.pforge` writing is removed. Keep a `.pforge` reader only as a migration path:
    Open converts the file into a PSD document and says so; `--convert-pforge <in> <out.psd>`
    does it from the command line.
  - Every feature `.pforge` stored must have a PSD representation. PixelForge-only data without
    a Photoshop equivalent (Measurement Log, Count labels, notes if not mapped to Photoshop's
    annotations…) goes in one private, documented image resource. Photoshop is expected to drop
    it on re-save; the README says so.
- **The composite is always written** (image data section, plus the `Lr16`/`Lr32` conventions at
  16 and 32-bit), so Finder, Preview and every other reader show the right image. This includes
  Photoshop's "Maximize Compatibility" behaviour.
- **Reuse the architecture:** 256² COW tiles, mips, `push_undo`, `spawn_job` for anything over
  30 ms, the `Filter` / `catalog::Kind` model, the GPU runtime with its CPU fallback and cost
  model.
- **The 8-bit path stays the reference:**
  - 16-bit and 32-bit implementations are tested against it, quantised (16-bit results reduced
    to 8 bits must equal the 8-bit result within 1/255 where the operation is defined that way).
  - GPU paths at every depth are tested against the CPU at that depth: bit-exact for integer
    16-bit; for float, relative error ≤ 1e-5 or documented per kernel (Metal compiles with fast
    math, so use the same techniques as `engine/gpu`).
- **Proof bar:** never claim a feature works without:
  - a test;
  - a `PF_DEMO=tools` screenshot showing it (document mode and bit depth visible in the Info
    panel, the file format in the title or status bar);
  - for files, the Photoshop round-trip result.
  Look at every screenshot and fix what's wrong.
- **Build:** warning-free, `cargo test --release` green, and the suite green with `PF_GPU=off`
  and `PF_GPU=force`. No existing benchmark budget may regress by more than 5 % at 8-bit.

## Phase 1: Foundations — realistic benchmarks and memory

1. **Benchmark tiers:**
   - Add tiers to `--bench`, `--bench-tools`, `--bench-filters` and `--bench-gpu`: 24 MP
     (6000×4000), 61 MP (9504×6336), 150 MP (14204×10652), and 268 MP as the stress row. Run
     each at 8, 16 and 32-bit once those exist.
   - Set latency budgets per tier for the interactive operations: brush stroke latency, filter
     preview, filter commit, layer composite, open, save.
   - The 24–61 MP tiers are the ones users feel; make them the headline numbers in the README.
2. **Memory accounting:**
   - One `MemoryBudget` (preference: % of RAM, default like Photoshop's ~70 %) covering layer
     tiles, history, mips, previews, the GPU pool and the resident cache.
   - Show it in the Info panel (used / budget, compressed, on disk).
   - Log it in every benchmark.
3. **Compressed cold tiles:**
   - History snapshots, tiles untouched for N seconds, and tiles of hidden layers are compressed
     in memory (fast LZ-class or deflate level 1; benchmark the options and ask for any crate).
   - Decompress on access, transparently through the grid API.
   - Uniform and empty tiles stay shared as today.
4. **Scratch disk:**
   - When the budget is exceeded, the least-recently-used compressed tiles spill to a scratch
     file in the config or temp directory (a preference, like Photoshop's), read back on demand.
   - A 150 MP 32-bit document with 20 history states must stay interactive without macOS paging.
     Measure it.
5. **Display precision:** display tiles and the display path stay 8-bit (half float for HDR)
   whatever the document depth. Mips for display are kept at display precision; full-depth mips
   only where an operation needs them.

## Phase 2: PSD / PSB, 8-bit, complete

1. **Codec layer** (`engine/psd/`):
   - Header, colour mode data, image resources, layer and mask info and image data.
   - PackBits and ZIP (with and without prediction), PSB 8-byte lengths.
   - The full descriptor reader and writer, with key order and unknown keys preserved.
   - A raw "block tree" model that holds every byte that isn't interpreted.
2. **Mapping to the document model**, both ways:
   - Pixel layers, masks (user and vector), blend modes (all 27 of Photoshop's, including the
     ones PixelForge doesn't have yet — add them: Linear Burn, Darker Color, Lighter Color, Vivid
     / Linear / Pin Light, Hard Mix, Subtract, Divide, Dissolve…), opacity, fill, clipping,
     groups (pass-through, isolated, knockout), blend-if ranges, channel restrictions, locks,
     colour tags, visibility.
   - Adjustment and fill layers: every type Photoshop has. Add the missing adjustments to the
     engine rather than rasterising them.
   - Layer styles: every effect and parameter in `lfx2`, multiple instances per effect, global
     light, "scale effects", contours, gradients and patterns. Match Photoshop's rendering (the
     Photoshop export is the reference).
   - Text (`TySh` plus `EngineData`): parse and write `EngineData`; map style runs, paragraph
     styles, warp, text on a path and area text to `engine/text.rs`. Text stays live both ways.
     When PixelForge's layout differs, the stored Photoshop rendering is shown until the user
     edits the layer.
   - Shapes and paths: live shapes (`vogk`), vector masks, fill and stroke (`vscg`/`vstk`),
     path components and operations, paths panel, clipping path.
   - Smart objects: embedded and linked (PSD, PSB, raster, raw), transform and warp, smart
     filters (interpret the ones PixelForge has; preserve the others byte-exact and keep their
     stored output), "Edit Contents" opening the embedded document.
   - Artboards, slices, guides, grids, layer comps, annotations and notes, the count tool, XMP,
     EXIF, IPTC, ICC profiles (convert or assign on open like Photoshop), resolution, print
     settings, timeline (preserve), and 3D / video layers (preserve, show stored pixels,
     read-only).
3. **Colour modes:**
   - RGB is editable.
   - Grayscale, Bitmap, Indexed, Duotone, Lab, CMYK and Multichannel open and preserve; editing
     converts the way Photoshop's Image ▸ Mode does, with a prompt.
   - Extra channels (spot colours, alpha channels) are kept, listed in a Channels panel, and
     written back.
4. **Test corpus:**
   - A `tools/photoshop_psd.jsx` that makes, in Photoshop, one PSD per feature and parameter
     family (hundreds of small files), plus several large realistic documents. It also exports
     each file's flattened composite and each layer's pixels.
   - A `--psd-corpus <dir>` command that runs every round-trip check below and writes a report.
   - Never commit Adobe sample files without a licence; generate everything locally.
5. **Checks, per corpus file:**
   - (a) PixelForge's composite against Photoshop's export.
   - (b) PS → PF → PS: open PixelForge's saved file in Photoshop by script, export, and compare.
     Read Photoshop's layer tree back by script (names, kinds, blend modes, opacities, styles,
     text contents, smart objects) and compare it to the original.
   - (c) Structural diff of all blocks: interpreted ones equal after re-encoding, preserved ones
     byte-exact.
   - (d) No Photoshop warning dialogs. Watch the window list and fail the check if a dialog
     appears.
6. **Switch-over:** Save / Open / Recent use PSD and PSB, `.pforge` is migration-only,
   autosave and crash recovery write PSD, and the README's format notes are rewritten.

## Phase 3: 16-bit documents

1. **Pixel type:**
   - Tiles become generic over the pixel format (RGBA8 premultiplied, RGBA16 in Photoshop's
     0…32768 range, RGBA32F linear).
   - Keep the 8-bit SWAR fast paths. Add 16-bit SWAR (one channel per 32-bit lane, or 4 × 16 in
     64-bit) and SIMD where it pays off.
   - Decide premultiplied versus straight per depth by measurement and Photoshop equality, and
     document why.
2. **Image ▸ Mode:**
   - 8 ↔ 16 ↔ 32 conversion with Photoshop's rules: dithering when going down (Photoshop's
     "Use Dither (8-bit/channel images)" preference) and the 32 → 16/8 HDR Toning dialog.
   - The Info panel shows depth and mode.
3. **Engine, in order:**
   - compositing, blend modes, masks and styles;
   - mips;
   - brushes and all paint tools;
   - selections and masks;
   - every adjustment and filter (including the Filter Gallery);
   - retouching;
   - resampling and transforms;
   - Liquify and Perspective Warp;
   - Select and Mask;
   - Content-Aware Fill;
   - gradients (16-bit gradients without banding).
   Anything not yet 16-bit-capable is greyed out with a tooltip, never silently 8-bit.
4. **GPU:** a 16-bit variant of every kernel (`rgba16uint` or packed storage), bit-exact
   against the 16-bit CPU path, with cost-model priors per depth.
5. **Files:** PSD and PSB 16-bit read and write (`Lr16`, 16-bit image data, Photoshop's 32768
   scale), 16-bit PNG and TIFF import and export.

## Phase 4: 32-bit float and HDR

1. **32-bit documents:**
   - Linear-light float (f32; benchmark storing tiles as f16 with f32 math and keep it where
     precision holds).
   - Offer the same tools and filters Photoshop allows in 32-bit (checked by script against
     Photoshop 2026's enabled menu items), plus any PixelForge can do correctly in float.
   - Add Exposure, the 32-bit Preview options (exposure / gamma on the view), HDR Toning (local
     adaptation, equalize histogram, exposure and gamma, highlight compression) and Merge to HDR
     (from bracketed files: alignment, ghost removal, response curve).
2. **HDR display:**
   - Patch the vendored GPUI Metal layer for EDR: an `rgba16Float` drawable in an
     extended-linear space and half-float display tiles.
   - HDR documents show highlights above SDR white up to the screen's current EDR headroom.
     They are tone-mapped on SDR screens and in screenshots.
   - A preference to turn this off.
   - Info panel: current headroom.
   - Verify by reading the display texture's values (above 1.0 where expected), since
     screenshots may clip.
3. **HDR files:**
   - OpenEXR (half and float, multipart where needed) and Radiance `.hdr`.
   - 32-bit TIFF and 32-bit PSD and PSB (`Lr32`, ZIP with prediction byte layout).
   - AVIF and JPEG XL with ISO 21496-1 gain maps, and HDR-to-SDR export with tone mapping.
   - Ask before each crate.
4. **Photoshop comparison:** the same 32-bit operations in Photoshop by script (Exposure, HDR
   Toning, Merge to HDR, 32-bit filters) on the same bracketed and EXR inputs, compared with
   relative error and SSIM of the tone-mapped outputs, and timed.

## Phase 5: Raw

1. **Decoding:** propose a decoder (from the candidates above) with its licence, size and
   camera coverage, and wait for my approval. DNG is decoded in-repo from the spec if the chosen
   library doesn't cover all of it.
2. **Development pipeline:**
   - Run it in float, on the GPU with the CPU reference: black/white levels, white balance (as
     shot, presets, custom), demosaic (a high-quality algorithm, benchmarked against the others),
     camera matrix to a linear working space, highlight reconstruction, lens profile corrections
     where available, then the existing Camera Raw controls.
   - The open dialog is the Camera Raw Filter UI extended with raw-only controls.
   - Open into 16-bit or 32-bit (a preference, like Camera Raw's workflow options), or as a
     smart object that keeps the raw and the settings (re-editable, saved inside the PSD the way
     Photoshop does).
3. **Photoshop comparison:**
   - Open the same raws with Camera Raw defaults and with a fixed set of slider values by script
     (watching for dialogs).
   - Report SSIM and ΔE00 against PixelForge, and time.
   - Use test raws generated or captured locally, or ask me before downloading samples (source
     and licence).

## Phase 6: Speed at depth

For every operation, at each tier and depth:

- **Benchmark:** CPU, GPU and Photoshop at 24, 61 and 150 MP, in 8, 16 and 32-bit, plus the
  268 MP stress row.
- **Speed budgets:** at 16-bit, each operation takes at most 2× its 8-bit time, and beats
  Photoshop's 16-bit time. At 32-bit, it beats Photoshop's 32-bit time.
- **Memory budgets:** peak resident memory per tier and depth, staying within the MemoryBudget
  with no macOS paging. Measure page-ins during the benchmark.
- **Optimise:** fused passes, half-float GPU storage where precision allows, read-once
  pipelines, prefetching from the scratch disk during strokes, and GPU-resident 16 and 32-bit
  documents where the cost model wins.
- **README:** the tiered tables (headline 24 / 61 MP), with memory columns.

## Verification per phase

1. Unit tests for every codec block, the descriptor codec, every mapping, every depth path
   (against the 8-bit reference and the GPU), Image ▸ Mode conversions, memory budget and
   compression, and the scratch disk (including a simulated full disk and a corrupted scratch
   file, falling back cleanly).
2. `--psd-corpus` green: every corpus file passes (a)–(d) above, or appears in the README with a
   precise reason that is outside the code (e.g. undocumented proprietary data that is preserved
   but not rendered).
3. Photoshop round trips and timings by jsx, with dialog watchdogs.
4. `PF_DEMO=tools` phases:
   - opening a complex Photoshop-authored PSD (text, styles, smart objects, adjustments) and
     editing it;
   - saving and reopening it in Photoshop (screenshot of Photoshop by window ID);
   - a 16-bit gradient without banding;
   - a 32-bit HDR document on the EDR display (with the texture-value check);
   - a raw open dialog;
   - the Info panel memory row under a large document.
   Look at every screenshot.
5. README: a "File format" section (PSD and PSB as the native format, the compatibility table
   per feature with the measured round-trip results, what is preserved but not editable, the
   migration from `.pforge`), "Bit depth and HDR", "Raw", the tiered benchmarks, and the "Not
   implemented / differs" section limited to things outside the code.

## Reporting

After each phase, report:
- what was built;
- the corpus pass rate (per feature family), the round-trip and fidelity numbers against the
  bars;
- CPU, GPU and Photoshop times and memory per tier and depth against the budgets;
- which screenshots you checked and what they showed;
- anything deferred, preserved-but-not-editable, or blocked, with the reason.

## Sources

- Adobe, Photoshop File Format Specification:
  https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/
- Adobe, Photoshop 16-bit and 32-bit / HDR images (Image ▸ Mode, 32-bit Preview, HDR Toning,
  Merge to HDR Pro): https://helpx.adobe.com/photoshop/using/high-dynamic-range-images.html
- Adobe, Camera Raw HDR output (AVIF / JPEG XL, HDR display):
  https://helpx.adobe.com/camera-raw/desktop/hdr-and-advanced-output/hdr-output.html
- Adobe, Photoshop GPU card usage (technology previews, HDR display):
  https://helpx.adobe.com/photoshop/desktop/get-started/technical-requirements-installation/photoshop-and-graphics-processor-gpu-card-usage.html
- Adobe, DNG specification: https://helpx.adobe.com/camera-raw/desktop/dng-and-file-formats/digital-negative.html
- ISO 21496-1 gain maps; Adobe gain map specification:
  https://helpx.adobe.com/camera-raw/desktop/hdr-and-advanced-output/gain-map.html
- Apple, EDR (Metal): https://developer.apple.com/documentation/metal/hdr-content
- psd-tools (Python, MIT): https://github.com/psd-tools/psd-tools
- ag-psd (TypeScript, MIT): https://github.com/Agamnentzar/ag-psd
- Photopea documentation (PSD-first editor): https://www.photopea.com/learn/psd
- rawler / dnglab (LGPL-2.1): https://github.com/dnglab/dnglab
- rawloader (LGPL-2.1): https://github.com/pedrocr/rawloader
- LibRaw: https://www.libraw.org
- darktable memory and performance: https://docs.darktable.org/usermanual/5.6/en/special-topics/mem-performance/
