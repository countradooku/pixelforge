# Phase 2 PSD/PSB implementation and verification

Status: **INCOMPLETE — acceptance has not passed.** This report is being updated as verification continues. No later phase has begun. `.pforge` remains native for new documents, autosave and recovery because not every native feature has a safe PSD representation.

## Preservation and prerequisites

The workspace has no Git repository. Before modification, source and the working release executable were copied to `/private/tmp/pixelforge-psd-scratchpad/phase-2-2026-10-11-153448/`. The archive SHA-256 is `c21aacc09e9a4359c77fc9efb3be1d1df30d1e0216cff9eb494042e4604e6bfb`; the baseline executable SHA-256 is `59db5eeadb9c09b6a13b91c53425075b5da9d5046463de369e6f5542f08ac041`. Its manifest records individual source hashes.

Phase 1 and its follow-up were checked against their logs: 383 passed / 18 ignored in each GPU mode, a clean release build, and matching follow-up source fingerprints. No prerequisite repair was needed. The previously reported Clippy failures (9 errors, 164 warnings) are separate from the release/test gates; this phase has not claimed them fixed. No dependencies were added, promoted or downloaded. ML assets were not downloaded. `PROMPT_MCP.md` remains outside the changes.

The machine is an Apple M5 Pro, 18 CPU cores, 48 GiB RAM, macOS 27.0.1 arm64. See [environment.json](environment.json), the backup manifest and [change-inventory.json](change-inventory.json). Heavy builds, tests and measurements run serially with isolated application configuration directories.

## Implemented scope

* In-repository PSD/PSB framing, raw block tree, image resources, channel streams, PackBits, ZIP and ZIP prediction; bounded lengths, allocation limits, shared backing storage and MemoryBudget integration. Unsupported engine depths fail explicitly.
* Ordered descriptors and EngineData preserve unknown keys, duplicate keys, lexical data and opaque bounded parents. Unsupported unframed descriptor values preserve the enclosing block instead of guessing their length.
* RGB8 document mapping for pixels, ordinary groups, opacity/fill, clipping, basic raster masks, 27 blend keys, Blend If and RGB channel restrictions. Advanced blend state participates in native persistence and undo/redo. Unsupported advanced CPU operations explicitly bypass the GPU path.
* Imported text retains TySh/EngineData and Photoshop's cached raster. Document-level Txt2 makes changed-text saves unsafe, so those saves are rejected. This does not satisfy live text editing acceptance.
* Worker-based PSD/PSB Open/Save for the supported subset; Channels lists retained extra channels; Info exposes format, mode and depth. Unsupported stack/canvas edits and unsafe channel/mask changes are rejected before replacing files.
* Migration `--convert-pforge`, structural/editable `--psd-check`, strict `--psd-metrics`, and an evidence-producing `--psd-corpus` runner. Output alias protections cover canonical paths, existing hard links/symlinks and case aliases.

A real Photoshop PSB exposed `8B64` framing on `cinf` with a 410-byte payload. The parser now honors the signature's 64-bit length in addition to the specification's PSB key rules. Literal framing/alignment, truncation and oversized-length tests were added; byte-exact re-encoding alone would not have detected the old opaque-tail misparse.

Other corrected save paths include group blend keys inside section dividers, relative mask rebasing when pixel bounds expand, byte-preserving mask toggles, and rejection of unrepresentable pass-through group blend edits. Merge Down refuses advanced blend settings that would otherwise be applied twice.

## Release and correctness evidence

[verification-psb/runs.json](verification-psb/runs.json) records the final build/test commands and exit codes:

| Check | Result |
|---|---|
| Offline locked release build | Pass, no warnings |
| Default GPU tests | 498 passed, 19 ignored |
| PF_GPU=off tests | 498 passed, 19 ignored |
| PF_GPU=force tests | 498 passed, 19 ignored |
| Explicit Photoshop-authored text test | 1 passed |

The authored-text test is normally ignored because it depends on locally generated Photoshop fixtures. Other ignored tests remain ignored and are not counted as passes. New blend modes retain strict CPU/GPU equality tests; legacy tolerances and the integer CPU reference were not relaxed.

`document.rs` SHA-256 stayed `06e009615dc1589e409b4f5dd93b1d138e3c26e0d3212379d7448b75b4c99807` before and after every final command. Earlier unexpected disappearance of its advanced-blend fields was investigated; no writer was identified, and the user knew of no concurrent edits. The fields were restored and have remained stable through the recorded verification. Earlier failed logs are retained rather than overwritten.

## Photoshop corpus and first full run

All fixtures were authored locally. No Adobe sample files were downloaded. The planned corpus contains 323 cases; 304 were generated successfully and 19 failed authoring. Four failed contour/effect combinations were mistakes in the catalog, not evidence that Photoshop lacks those effects. Coverage inspection also found ignored generator parameters; corrected retries retain the earlier evidence and do not count duplicated output as parameter coverage.

The first full run is [corpus/pixelforge-results-m7ovphyt/REPORT.md](corpus/pixelforge-results-m7ovphyt/REPORT.md), with raw per-file results and fingerprints alongside it. It precedes the final 8B64 framing correction. It found 307 byte-exact raw round trips (including three retained historical fixtures), 187 document mappings and 88 passing PixelForge composite comparisons. **0/323 passed all gates.** Opaque preservation does not count as editable support.

The independent native-open watchdog stopped on the first file after observing a `Pixel Layer Bar` layer-8 window. This may be a contextual toolbar, but its identity was not visually established during that run, so the strict failure remains. Three subsequent standalone native-open runs passed, restoring preferences and closing their own test documents. They do not substitute for a full corpus pass. No blanket toolbar exemption was added.

Both deliberately truncated and corrupt negative controls triggered watchdog failures; the named test dialogs were inspected and dismissed, without closing user documents. Their evidence is in `corpus/warning-controls/`. Each Photoshop invocation retains its staged JSX, configuration and independent CoreGraphics watch log. Known narrow Progress-window classification fixes preserve the original failed logs and their disposition.

## Confirmed fidelity failures

The acceptance thresholds remain integer-exact RGBA for integer paths and SSIM >= 0.995 / mean DeltaE00 <= 0.5 for perceptual paths. Full-resolution comparison is used. Text line breaks are not independently measured yet, and therefore text acceptance fails even where cached pixels match.

Opaque Normal and Multiply at full opacity/fill matched. Partial opacity/fill exposed one-level rounding differences: Normal at 37% opacity differed in 576 pixels and opacity+fill at 37% in 2,112; Multiply at 37% differed in 672 and both at 37% in 384. An isolated Multiply group at 43% differed in 1,632 pixels. Photoshop's fixture result blends the mode result using a combined rounded opacity/fill; the current premultiplied engine path rounds at different stages. A fixture-only arithmetic change was not substituted for a validated general compositing model.

Partial-opacity Dissolve has different spatial behavior, with maximum errors around 202–203 and SSIM around 0.65–0.75. Transparent fixtures can fail exact RGBA because hidden RGB differs even where alpha matches and white-matted SSIM is 1. These remain failures; the integer gate was not weakened.

## Memory, timings and workload limitations

The four large files retain their actual dimensions: 6000×4000, 9504×6336, 14204×10652 and 16384×16384. They are six-layer synthetic rectangle/backdrop fixtures, **not realistic production documents**. This requirement remains unmet. Their first-run codec process peak RSS was approximately 0.48 / 1.01 / 2.29 / 3.83 GB. The repeated 268 MP engine/tool/GPU runs peaked at 5,096.5 / 2,827.0 / 20,233.5 MiB process RSS respectively; tracked budget peaks differ and are recorded in [benchmark-memory.json](benchmark-memory.json). Photoshop's separate process memory is not included. `/usr/bin/time` process swap counters were zero; ordinary file page-ins are not reported as swapping, and this is not a claim that system-wide swap was absent.

Repeated alternating baseline comparisons completed for all four tiers: engine, layer tools and CPU/GPU compositing (216 runs, nine repetitions per executable). See [all medians](BENCHMARKS.md). Five initial medians exceeded 5%: 24 MP blank canvas +15.21% (0.388 → 0.447 ms), 24 MP selected levels +7.94% (2.469 → 2.665 ms), 268 MP horizontal flip +6.60% (13.878 → 14.794 ms), 150 MP GPU one-tile edit +7.14% (9.8 → 10.5 ms), and 268 MP cold GPU uploads +5.31% (139.3 → 146.7 ms). No run reported a budget miss. The focused engine follow-up (15 repetitions per executable at 24/268 MP) found blank canvas -5.98%, selected levels +7.59% and flip +0.69%. It also found display medians +14.07% at 24 MP/mip 0, +5.84% at 24 MP/mip -2, and +5.04% at 268 MP/mip 0. These are retained as unresolved performance findings; the display path gained advanced-state and Dissolve dispatch checks, while selected-levels code is unchanged. The GPU follow-up found 150 MP one-tile edit unchanged (9.6 → 9.6 ms), and 268 MP cold uploads +0.75% (134.2 → 135.2 ms). It also found a new 268 MP one-tile-edit median regression of +14.29%; this remains unresolved. See `focus-engine/report.json`, `focus-gpu/report.json` and their raw logs. The changing affected measurements show run variability, but are not grounds to discard above-threshold results. The unchanged source hashes for benchmark/operation/grid/memory/selection/filter paths are recorded in [regression-source-audit.json](regression-source-audit.json); unchanged source alone does not establish unchanged compiled performance. The compositing benchmark pins the original 18-mode sequence so adding blend modes does not silently change the baseline workload. New modes have separate correctness tests. Raw samples, latency-budget misses and every median regression above 5% are retained for investigation.

## Private resource and outstanding gates

Image resource 4000 with the 13-byte `PixelForge\0\x00\x01` signature stores active layer ID (big-endian u64), selected count (u32), and selected IDs (u64). A foreign resource with the same numeric ID and different signature is preserved separately. This is only PixelForge selection state, not an alternative encoding of Photoshop features. Installed-Photoshop re-save retention and independent native-writer reopening are pending verification.

Required work remains: all adjustment/fill/style mappings and rendering; complete live text including Txt2, path/area text and measured line breaks; shapes/paths; smart objects and filters/Edit Contents; artboards and metadata UI; non-RGB conversion and ICC assign/convert; channel editing; read-only stored output for 3D/video; realistic large workloads; passing full native reopen/tree/structural/fidelity gates; screenshot-verified application workflows; regression investigations; and safe native-format switch-over. These are implementation and verification gaps, not externally blocked successes.

No Phase 3 work has begun. The phase must remain open until these requirements pass.
