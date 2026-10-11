# Phase 1 verification — 2026-10-11

The later [performance follow-up](../2026-10-11-phase1-followup/REPORT.md) resolves the Object Finder and GPU Auto misses recorded below and contains the final 383-test results. This report retains the original memory-pressure and Photoshop evidence.

RGBA8 memory foundations and tiered benchmarks. The document engine remains RGB, 8 bits/channel; `.pforge` remains native until Phase 2. No PSD, HDR, raw, 16-bit or float compatibility claim is made here.

## Implementation

- One RAM allowance (70% default) for shared layer/history/mip pixels, masks, previews and GPU allocations, with settings and Info counters.
- LZ4 cold storage, access guards, stable LRU spilling, private checksummed scratch records with a verified recovery copy, and reused disk extents. Failed writes retain RAM data.
- History retains managed tiles and cached mips; pressure no longer removes history states. Display remains RGBA8.
- Reduced per-pixel locking, cached sampling, direct tile transforms, blocked rotation, deduplicated mip updates, and staged slice export that avoids nested worker waits behind compression.

## Tests and build

- [tests-default](tests-default.log): ok. 376 passed; 0 failed; 18 ignored; 0 measured; 0 filtered out; finished in 87.71s
- [tests-off](tests-off.log): ok. 376 passed; 0 failed; 18 ignored; 0 measured; 0 filtered out; finished in 1.81s
- [tests-force](tests-force.log): ok. 376 passed; 0 failed; 18 ignored; 0 measured; 0 filtered out; finished in 86.79s
- [Release build](build.log): finished without compiler warnings.
- [Clippy](clippy.log): fails on nine existing lint errors (approximate constants, large enums, `SyncPtr::at`, a manual `contains`, and a boxed UI argument); its 164 warnings are separate from the warning-free compiler build. Optional ML/all-features validation was not run because its ONNX build can download a runtime.

## Pressure and compression

The 150 MP RGBA8 run used a 512 MiB allowance and 20 full-image history states. Every tile in every restored image was fingerprinted; all 20 states matched. Undo timers exclude this verification pass.

[Pressure log](memory-150-512.log) and [system counters](memory-150-512-system.json):

- Peak tracked: 516.6 MiB; peak RSS: 925.5 MiB; live scratch, including recovery copies: about 5.0 GiB.
- Brush including mips: p95 0.352 ms, max 0.402 ms. Display: 135 RGBA8 tiles in 0.983 ms. Undo: median 0.808 ms, p95 0.966 ms.
- Page-outs 216→216; swap-outs 156→156; memory compressions 4,889,632→4,889,632. Swap usage 2.44→2.44 MiB. Page-ins 2,853,489→2,853,513 include ordinary file activity.
- The allowance is for tracked image storage, not an RSS cap. Pins can temporarily exceed it; masks/previews, allocator retention, temporary processing buffers, metadata and driver overhead limit reclamation.

[Compression comparison](compression.log): LZ4 encoded/decoded the 16 MiB procedural corpus in 19.376/4.575 ms, versus deflate level 1 at 64.117/44.006 ms. Sizes were 64.3% and 49.1%; LZ4 was chosen for restore latency. Incompressible data was also checked bit-for-bit.

## Tier runs

All suites run serially in fresh processes. Engine/tools/filters use CPU mode; GPU runs measure CPU and forced GPU paths and the model’s Auto selection. Full raw timings and resource counters are in each log. `runs.json` records exit status and fixed-budget misses.

| Suite | 24 MP | 61 MP | 150 MP | 268 MP |
|---|---|---|---|---|
| engine | [0, 0 misses](engine-24.log) | [0, 0 misses](engine-61.log) | [0, 0 misses](engine-150.log) | [0, 0 misses](engine-268.log) |
| tools | [0, 1 misses](tools-24.log) | [0, 0 misses](tools-61.log) | [0, 1 misses](tools-150.log) | [0, 0 misses](tools-268.log) |
| filters | [0, 0 misses](filters-24.log) | [0, 0 misses](filters-61.log) | [0, 0 misses](filters-150.log) | [0, 0 misses](filters-268.log) |
| gpu | [0, 6 misses](gpu-24.log) | [0, 15 misses](gpu-61.log) | [0, 8 misses](gpu-150.log) | [0, 4 misses](gpu-268.log) |

Cells show process exit code and reported budget misses; exit 0 does not mean every latency target passed. All new engine latency targets pass. Object-finder precomputation exceeds its existing 2 s target at 24 and 150 MP. GPU misses mean Auto selected a path outside its “within 10% + 0.5 ms of fastest” target; the measured CPU and GPU paths still completed. Full-image optional resident probes are omitted when the measured banded commit is too close to the dispatch watchdog. The watchdog itself is unchanged.

| Forced GPU commit | 24 MP | 61 MP | 150 MP | 268 MP |
|---|---:|---:|---:|---:|
| Hue/Saturation | 5.3 ms | 11.1 ms | 28.7 ms | 45.9 ms |
| Gaussian r20 | 13.8 ms | 29.4 ms | 76.8 ms | 131.3 ms |
| Gaussian r120 | 25.1 ms | 56.4 ms | 142.8 ms | 281.3 ms |

GPU runs were collected before the last CPU sampling/export/mip optimizations. Their compute kernels are unchanged; the final test modes rerun CPU/GPU equivalence.

## Comparison with the original executable

The preserved pre-change executable is `/private/tmp/pixelforge-psd-scratchpad/pixelforge-before`. `tools/phase1_compare.py` alternates original/new processes at 8192². Full suites and focused repeats are retained separately. The original engine/filter output rounds to 0.1 ms, and tool output to 0.01 ms; small deltas need that precision limit and run variance considered. Legacy fixture coordinates are preserved wherever they fit.

The completed comparison set covers 283 matching operation labels. The engine uses 25 alternating pairs; focused tool groups use 9–31 pairs and focused filters use 9. Remaining median deltas above 5% are listed below, with focused repeats replacing the corresponding full-suite row. Raw samples are retained, including slower observations; fixed latency budgets do not waive relative regressions.

| Operation | Original ms | Current ms | Change | Evidence |
|---|---:|---:|---:|---|
| None in the completed repeated comparisons | | | | |

## Photoshop comparison

Photoshop 2026 27.10.0 completed all four tiers using the same opaque procedural RGBA8 pixels exported as PNG. Filter values are best of three applications, restoring the source history state each time; open/save are single observations. PNG open and PSD save measure different formats from PixelForge’s native `.pforge` rows and are not direct codec comparisons. These are speed measurements, not PSD round-trip or pixel-equivalence certification.

| Photoshop operation (ms) | 24 MP | 61 MP | 150 MP | 268 MP |
|---|---:|---:|---:|---:|
| Fixture open | 268.6 | 504.9 | 1027.6 | 1455.4 |
| Hue/Saturation | 16.4 | 36.5 | 68.9 | 117.8 |
| Levels | 72.9 | 186.3 | 455.3 | 808.4 |
| Gaussian blur r20 | 53.8 | 136.4 | 297.8 | 500.4 |
| Gaussian blur r120 | 73.5 | 163.7 | 365.9 | 635.0 |
| Invert | 11.8 | 22.2 | 46.1 | 78.5 |
| PSD save | 438.0 | 332.4 | 732.2 | 1289.4 |

Logs: [24 MP](photoshop-24.log) / [watchdog](photoshop-24.watchdog.json), [61 MP](photoshop-61.log) / [watchdog](photoshop-61.watchdog.json), [150 MP](photoshop-150.log) / [watchdog](photoshop-150.watchdog.json), [268 MP](photoshop-268.log) / [watchdog](photoshop-268.watchdog.json). Each successful run restored the previous active document and dialog preference and closed only its temporary benchmark document. The first TIFF attempt was intercepted by Camera Raw; the watchdog detected it, the user cancelled it, and subsequent runs used PNG. Initial failure evidence is retained under `initial/`.


## Visual checks

The four current captures were inspected after unlocking the Mac:

- [512 MiB pressure document](phase1_memory_150mp.png): 14204×10652, RGB 8 bits/channel, `.pforge` title/status, history and 509/512 MiB usage with 1719.9 MiB scratch visible.
- [Pressure preferences](phase1_memory_preferences.png): RAM percentage controls and scratch folder chooser visible without clipping. The 512 MiB test override applies independently of the saved 70% preference.
- [Cold storage document](cold-demo/phase1_memory_150mp.png): 2677/4096 MiB, 381.6 MiB compressed and 127.4 MiB scratch visible.
- [Cold storage preferences](cold-demo/phase1_memory_preferences.png): controls and document state visible.

Initial locked-screen captures are retained under `initial/locked-screen/` as failed verification evidence, not as proof of the feature.

## Outstanding acceptance gates

- The requested 150 MP **32-bit/channel** / 20-history-state stress test needs the float engine in Phase 4. The tested RGBA8 run is not a substitute for it.
- PSD corpus and Photoshop round trips are not applicable to this memory-only phase; no PSD reader/writer was added.
- The original CPU comparison set has no remaining median regressions above 5%. Full performance sign-off still excludes the documented rectangular-tier Object Finder and GPU Auto-selection budget misses.

No new dependencies were added or downloaded. `block 0.1.6` was copied from the existing local Cargo cache and patched for the current compiler; source and license are preserved in `vendor/block`.
