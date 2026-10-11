# Prompt: GPU filters and compositing on wgpu, with a CPU fallback

You are working in `~/Desktop/Code/Rust/pixelforge`. It is Rust with GPUI via `gpui-kit` 0.7.1 /
`gpui-pre` 0.3.8, vendored and patched in `vendor/`. It is not a git repo, so back up the tree to
the scratchpad before you start.

Every filter and adjustment in PixelForge already runs on the CPU (`src/engine/filters/*`, rayon
over 256² COW tiles), and README.md compares each one with Photoshop 2026. Your job is to add a
**GPU compute path on wgpu** for:

- every operation Photoshop runs on the GPU;
- every other operation where a GPU path is measurably faster.

The CPU path stays as the reference and the automatic fallback. The GPU path must produce the
same result as the CPU path, which is already measured against Photoshop. It must be faster than
both the CPU path and Photoshop by a wide margin.

Photoshop 2026 is installed and scriptable (`tools/photoshop_gaps.sh`, `tools/photoshop_gaps.jsx`,
`tools/photoshop_tools.jsx`). It is the reference for behaviour, quality and speed.

## What the research says (read this first)

**Photoshop's GPU use.** These are Adobe's GPU FAQ and Community Expert lists; confirm on the
current pages before you rely on them.

- **Won't run without a GPU:** Oil Paint, Perspective Warp, Liquify (it needs a compatible GPU
  per Adobe's filter troubleshooting page), Smooth Brush Resizing, Birds-Eye View and scrubby zoom.
- **GPU or OpenCL accelerated:**
  - Blur Gallery (Field, Iris, Tilt-Shift, Path and Spin; OpenCL) and Lens Blur.
  - Smart Sharpen (OpenCL) and Select and Mask (OpenCL).
  - Image Size ▸ Preserve Details, and the Camera Raw filter (Camera Raw *requires* a GPU for editing).
  - Neural Filters, Select Focus and Artboards.
- **Canvas:** Photoshop draws the canvas through "Native Canvas" (Metal / DirectX): zoom, pan and
  rotate.
- **Compositing:** Photoshop 2026 adds "GPU Compositing" and "Foreground Composite Caching".
  Users report artifacts and high WindowServer load with them, so get both right.
- **CPU mode:** with the GPU off, features that have a CPU pipeline still run but slower. The
  GPU-only ones are disabled.

**What the alternatives do.**

- **Affinity Photo:** end-to-end Metal compute (OpenCL on Windows) for tools, adjustments, live
  filters, brushes and RAW. Vector work and blend ranges stay on the CPU. It supports multiple
  GPUs. The gain is biggest with stacked live filters. Its cost is VRAM: 1–2 GB is enough for 4K,
  and 4 GB+ is better at 16-bit.
- **darktable** is the best-documented fallback design:
  - OpenCL per module, with a runtime estimate of CPU versus GPU work and a user "advantage hint".
  - Tiling when VRAM is short, with a reserved headroom (600 MB by default). Heavy tiling is
    often slower than the CPU path, so it goes to the CPU instead.
  - A watchdog: about 2 s before it falls back to the CPU.
  - Devices that keep failing are disabled automatically.
  - An async mode that syncs after each module when kernels misbehave.
- **Pixelmator Pro:** Metal and Core Image throughout, including painting.
- **GIMP / GEGL:** OpenCL was never stable or precise enough to be on by default (memory leaks,
  imprecision). Some builds removed it. The lesson is to test GPU results against the CPU
  reference, always.
- **Krita:** GPU for the canvas only; its filters are CPU-bound. Users notice filters lag behind
  Photoshop's.
- **Photopea:** layers kept in WebGL memory, GPU blending of every blend mode, masks and layer
  styles.
- **Graphite (Rust):** wgpu plus Vello, with a node-graph compositor.

**wgpu facts that matter.**

- WGSL is translated by naga to MSL, HLSL and SPIR-V, so naga can validate shaders for every
  backend offline.
- `shader-f16` (`enable f16;`) speeds up compute considerably on Apple GPUs (Chrome measured
  28–41 % on an M1 Pro).
- Subgroups beat workgroup shared memory for reductions and scans, but subgroup size varies by
  GPU; guard for size 1.
- Timestamp queries work on Metal, but not *inside* passes on Apple GPUs, so time whole passes.
- Push constants are native-only.
- `MAPPABLE_PRIMARY_BUFFERS` helps only on unified-memory machines (Apple Silicon) and hurts
  elsewhere.

## Ground rules

- **Dependency:** use `wgpu` **29.0.4**, the version already in `Cargo.lock` through
  `gpui-pre-wgpu`, so there aren't two copies; naga comes with it. wgpu and its dependency tree are
  approved. Ask me before any other new crate or download (name, source, size, licence).
- **Reuse the architecture:**
  - 256² COW tiles, mips via `layer_pixels_changed` / `set_layer_grid`, and `push_undo` before
    mutating.
  - `spawn_job` for anything over 30 ms, and the `Filter` / `catalog::Kind` model.
  - The display-tile cache (`ui/viewport.rs`) and the GPU sprite path in `vendor/`.
- **One switch:** a GPU path is an implementation detail behind the same filter entry, never a new
  menu item. `filters::apply` chooses the path.
- **CPU fallback everywhere:**
  - No adapter, an unsupported feature, a VRAM shortage, a validation error, device loss or a
    watchdog timeout each fall back to the CPU, mid-operation if needed, with an identical result.
  - A device that fails repeatedly is disabled and remembered in the config directory (see
    `engine::prefs`).
- **Same results:** for every GPU kernel, a test compares GPU against CPU on real photos and on
  edge cases (alpha, tile seams, image borders, mip levels, odd sizes).
  - Integer-defined operations (LUTs, blend modes, posterize, threshold) must be bit-exact.
  - Float pipelines may differ by at most 1/255 per channel, with mean ΔE2000 ≤ 0.2.
  - Random effects use the same hash as the CPU path, so the noise is identical.
  - The Photoshop fidelity numbers in the README (SSIM / ΔE) must not change by more than 0.002.
- **Speed bar:**
  - Every GPU kernel gets `--bench-gpu` rows (CPU ms, GPU ms including upload and readback,
    GPU-resident ms) at 16.7 MP and 268 MP, with budgets.
  - Each kernel must beat the CPU path and Photoshop's own time. Photoshop's numbers are in the
    README tables, or measure them with jsx.
  - Where the GPU loses, the cost model must keep the CPU path, and the README says so.
- **Proof bar:** never claim a feature works without a `PF_DEMO=tools` screenshot showing the GPU
  path running, with the adapter name and the path used in the Info panel. Look at every screenshot
  and fix what's wrong.
- **Platforms:**
  - Metal is primary on this Mac. Vulkan and DX12 must compile through `cfg` and be validated
    offline. The cross targets were declined, so don't install them.
  - Add a test that runs every WGSL module through naga's validator and its MSL, HLSL and SPIR-V
    writers.
  - Build stays warning-free and `cargo test --release` stays green.

## Phase 1: The GPU runtime (`src/engine/gpu/`)

1. **Device.**
   - Create a lazy, off-UI-thread `GpuContext`: one adapter (high performance), device and queue.
   - Record the features (shader-f16, subgroups, timestamp queries, push constants, mappable
     primary buffers) and the limits.
   - Register device-lost and uncaptured-error handlers that flip the context to "failed" and
     trigger the CPU fallback.
   - Expose the adapter name, backend and VRAM estimate in the Info panel and in `--bench-gpu`.
2. **Memory.**
   - Upload and download whole 256² tiles plus an apron (the filter's padding) into texture
     arrays or storage buffers. Benchmark `rgba8unorm` textures against packed `u32` storage
     buffers for your access patterns and keep the faster one.
   - Keep a GPU-resident layer cache (LRU, VRAM budget with darktable-style headroom) so chains
     of operations and previews don't re-upload.
   - Use a ring of staging buffers with `map_async`, pipelined so upload, compute and readback of
     consecutive tile batches overlap.
   - On Apple Silicon, test mappable primary buffers to skip copies.
3. **Scheduling.**
   - Build a cost model per operation: `upload + kernel(MP) + readback` against the CPU time.
     Calibrate it with a startup micro-benchmark, cache it per adapter in the config directory,
     and recalibrate when the driver or app version changes.
   - Small regions stay on the CPU: brush dabs, sub-tile previews, and anything under the
     measured break-even.
   - Large-radius operations that would tile heavily also stay on the CPU when the model says so.
   - Use a watchdog per submission (default 2 s) that cancels and falls back to the CPU.
   - Cache compiled pipelines (wgpu `PipelineCache` where supported) and precompile all shaders
     in the background at startup.
4. **User control.**
   - Preferences ▸ Performance: Use Graphics Processor (Auto / Off / Force), advanced per-feature
     overrides, and a "Benchmark GPU" button.
   - The environment variable `PF_GPU=off|auto|force` overrides them for tests.
   - The Info panel's Timing row says which path the last operation took.

## Phase 2: Where Photoshop uses the GPU

Match Photoshop's results through the existing CPU implementations, port each one to WGSL, and
beat Photoshop's time.

1. **Blur Gallery** (Field, Iris, Tilt-Shift, Path, Spin), including bokeh, light range and noise.
   **Lens Blur**, including the depth map, iris shapes, specular highlights and noise.
2. **Smart Sharpen** (Gaussian / Lens / Motion removal, shadows and highlights fade) and
   **Unsharp Mask**.
3. **Oil Paint**, **Liquify** and **Perspective Warp**:
   - Liquify: displacement-field brushes on the GPU, then a resample.
   - Perspective Warp: a mesh warp with bicubic resampling.
4. **Image Size**: bicubic variants, Lanczos and Preserve Details. Also free transform and warp
   resampling, and rotated crops (bicubic / Lanczos in `crop::warp_affine`).
5. **Camera Raw filter** pipeline: tone, HSL, curves, clarity / texture / dehaze, noise reduction,
   lens corrections. This needs no ML.
6. **Select and Mask:** edge refinement (guided filter, smooth / feather / contrast / shift edge,
   decontaminate colours).
7. **Compositing:**
   - A GPU compositor for display tiles and mips: every blend mode, opacity / fill, masks,
     clipping, groups (pass-through and isolated), adjustment layers and layer styles.
   - Feed it into the GPU sprite path so live previews never round-trip through the CPU.
   - Replace `prepare_rotated`-style CPU work wherever any is left.
   - Implement foreground composite caching (the layers below the edited one cached as a
     texture) without the artifacts Photoshop users report.
8. **Neural Filters are out of scope** (no ML in this prompt).

## Phase 3: Where Photoshop doesn't use the GPU, but should

Do these only where `--bench-gpu` proves a win; each needs its benchmark row.

1. **Blur family:**
   - Gaussian at every radius; keep the CPU algorithm so the results match.
   - Box, Motion, Radial, Smart and Surface blur (bilateral), and Shape Blur.
   - Median, Dust & Scratches and Reduce Noise.
2. **Pixel-local operations as fused LUT or matrix kernels:**
   - Adjustments: Levels, Curves, Hue/Saturation, Color Balance, Black & White, Channel Mixer,
     Selective Color, Gradient Map, Photo Filter, Vibrance, Exposure, Posterize, Threshold.
   - Fuse chained adjustment layers into one kernel per tile.
   - Histograms and auto-tone statistics via subgroup reductions.
3. **Distort:** Displace, Polar Coordinates, Twirl, Wave, Ripple, ZigZag, Spherize, Pinch,
   Lens Correction and Adaptive Wide Angle (all resampling-bound).
4. **Filter Gallery:** all 47 effects. They are CPU-only and slow in Photoshop (2–4 s at 16.7 MP),
   so they are big wins here. Keep the tuned fidelity from Phase G.
5. **Stylize / Render / Noise:** Emboss, Find Edges, Wind, Diffuse, Lighting Effects, Clouds,
   Fibers, Add Noise, High Pass, Maximum / Minimum, Custom.
6. **Retouching:**
   - The Poisson membrane and multigrid solver (healing, Content-Aware Fill blend, Vanishing Point
     heal).
   - PatchMatch for Content-Aware Fill (GPU PatchMatch with jump flooding is well established).
   - Mip-pyramid builds after edits.
7. **Keep on the CPU** (document why in a code comment):
   - flood fills (Magic Wand, Paint Bucket), graph cuts, vector booleans, text shaping;
   - PNG / JPEG / WebP encoding;
   - single small brush dabs, which are latency-bound. Measure big brushes (≥ 500 px with
     dynamics); Photoshop uses the GPU for brush resizing only.

## Phase 4: Making the kernels as fast as possible

For each kernel, try these techniques and keep what the benchmark proves:

- Tile the work into workgroup shared memory: load the tile plus its apron once.
- Run separable filters as two passes over shared-memory rows and columns.
- Pick workgroup sizes per vendor (8 × 8, 16 × 16, 32 × 8) from the calibration.
- Use f16 storage and arithmetic where precision allows (shader-f16). Keep f32 where the
  CPU-equality tests fail with f16.
- Use subgroup operations for reductions, histograms and prefix sums, with a shared-memory
  fallback.
- Fuse kernels:
  - adjustment chains into one pass;
  - blur → blend → mask;
  - filter → mip-level downsample in the same dispatch.
- Avoid readback: keep results GPU-resident until the document needs the pixels, such as save,
  history or a CPU-only operation.
- Overlap work: async compute, multiple queue submissions in flight, and double-buffered staging.
- On Apple GPUs, prefer `textureLoad` over `textureSample` for integer coordinates, avoid
  divergent branches in inner loops, and use precomputed LUTs over transcendental functions.
- Profile with timestamp queries around whole passes and record per-kernel GPU time in
  `--bench-gpu`.

## Phase 5: Comparison with Photoshop and the other editors

1. Add a phase block `GPU` to `tools/photoshop_gaps.jsx`. On the same 16.7 MP and 268 MP images,
   time:
   - Blur Gallery, Lens Blur, Smart Sharpen, Oil Paint, Image Size ▸ Preserve Details and the
     Camera Raw filter, all with descriptors;
   - Gaussian Blur, Surface Blur, Median, Unsharp Mask and three Filter Gallery effects.
   Save their outputs.
2. `--compare-gaps <dir> GPU` reports, per operation:
   - SSIM and ΔE2000 of the GPU path against Photoshop, and the same for the CPU path, which
     must agree within 0.002;
   - CPU ms, GPU ms and Photoshop ms.
3. README:
   - a "GPU acceleration" section with that table and the adapter used;
   - which operations stay on the CPU and why;
   - the fallback behaviour (when it triggers, and that it's tested);
   - a note on what Photoshop requires the GPU for that PixelForge doesn't, because PixelForge
     always has the CPU path.
   Where Photoshop has no scripting hook (Liquify, Perspective Warp, Select and Mask), say so in
   the README.

## Verification per phase

1. Unit tests:
   - GPU against CPU for every kernel, with the thresholds above and on seam and edge cases;
   - naga validation of every WGSL module for all backends;
   - forced fallback: `PF_GPU=off`, `force`, and a simulated device loss mid-filter, each giving
     identical pixels;
   - cost-model unit tests.
   The whole suite runs with `PF_GPU=off` and with `PF_GPU=force` (skipping GPU tests when no
   adapter exists).
2. `--bench-gpu` rows with budgets, all met; the `--bench-tools` and `--bench-filters` budgets
   still met.
3. Photoshop comparisons through jsx, with an honest README table and explicit notes where
   there's no scripting hook.
4. A `PF_DEMO=tools` phase showing GPU previews (filter dialogs, Blur Gallery live preview,
   compositing many layers, Liquify) with the Info panel naming the path. Look at every
   screenshot.
5. README updates: features, Preferences ▸ Performance, benchmark tables, the GPU section, and
   the "Not implemented / differs" section kept to items blocked by things outside the code.

## Reporting

After each phase, report:
- what was built;
- the GPU, CPU and Photoshop numbers against their budgets;
- the GPU-against-CPU equality results;
- which screenshots you checked and what they showed;
- anything left on the CPU or deferred, with the reason.

## Sources

- Adobe, Photoshop GPU FAQ and "Graphics processor (GPU) card usage":
  https://helpx.adobe.com/photoshop/kb/photoshop-cs6-gpu-faq1.html ·
  https://helpx.adobe.com/photoshop/desktop/get-started/technical-requirements-installation/photoshop-and-graphics-processor-gpu-card-usage.html
- Adobe, "Issues while using filters and effects" (Liquify / Oil Paint need a GPU):
  https://helpx.adobe.com/photoshop/kb/filter-effect-issues.html
- Adobe, Camera Raw GPU FAQ: https://helpx.adobe.com/photoshop/kb/acr-gpu-faq.html
- Adobe Community, which features need the GPU:
  https://community.adobe.com/questions-712/how-do-i-permanently-disable-use-graphics-processor-1110088
- Adobe Community, Native Canvas and GPU compositing in 2026:
  https://community.adobe.com/t5/photoshop-ecosystem-discussions/what-is-native-canvas-and-what-happens-when-i-disable-it/m-p/13156343 ·
  https://community.adobe.com/bug-reports-711/foreground-composite-caching-causes-viewport-compositing-artifacts-while-painting-beneath-semi-transparent-groups-photoshop-27-x-1632965
- Affinity Photo, hardware acceleration:
  https://affinity.help/photo2/en-US.lproj/pages/Extras/hardwareAcceleration.html
- darktable, memory and performance tuning (OpenCL scheduling, tiling, fallback):
  https://docs.darktable.org/usermanual/5.6/en/special-topics/mem-performance/
- GIMP 2.10 (OpenCL off by default): https://linuxfr.org/news/gimp-2-10-roule-au-gegl
- Krita, GPU acceleration (canvas only):
  https://mail.kde.org/pipermail/kimageshop/2018-May/015233.html
- Photopea 1.3, WebGL blending: https://blog.photopea.com/photopea-1-3.html
- Graphite (wgpu and Vello): https://summerofcode.withgoogle.com/archive/2024/projects/v5z2Psnc
- WebGPU subgroups and f16: https://developer.chrome.com/blog/new-in-webgpu-128 ·
  https://developer.chrome.com/blog/new-in-webgpu-120
- wgpu features (timestamp queries, push constants, mappable primary buffers):
  https://wgpu.rs/doc/wgpu/struct.FeaturesWGPU.html
- Intel, fast GPU blur algorithms:
  https://intel.com/content/www/us/en/developer/articles/technical/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms.html
