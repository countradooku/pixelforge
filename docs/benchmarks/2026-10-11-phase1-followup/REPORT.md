# Phase 1 performance follow-up — 2026-10-11

This follow-up fixes Object Finder latency and CPU/GPU Auto selection. It does not start Phase 2. The existing unified memory budget, LZ4 cold tiles and scratch storage remain in place; their pressure, corruption/recovery, compression and Photoshop timing evidence is in the [original Phase 1 report](../2026-10-11-phase1/REPORT.md).

## Changes

- Object Finder shares Select Subject’s boundary/colour prior instead of learning the entire inset rectangle as foreground. It rejects frame-dominated background components and locally refines replacement region candidates to recover shaded object edges. The guided edge filter now fuses four moment passes and two coefficient passes into channel batches, preserving each floating-point operation’s order; bitwise tests compare against the retained scalar reference. Click masks are cropped to cached candidate bounds with the full interpolation/filter halo; pixel-by-pixel tests compare them with the former uncropped path. Resolution and the two GrabCut iterations are retained. The graph-cut solver itself is unchanged.

- GPU Auto learns end-to-end rates separately for power-of-two pixel-area ranges, including distinct 150 MP and 268 MP ranges. Gradient algorithms have separate keys. Learning one range no longer overwrites another; old cached models are invalidated by a schema key. Unmeasured ranges keep the previous transfer/kernel and large-image fallback estimates.

- Added selection-mask quality tests, including the actual 150 MP textured fixture, frame contact and blank images; added cost-model isolation, boundary, serialization and gradient-variant checks. Benchmark gradient labels now identify the algorithm.

- No dependencies were added or downloaded. Source and executable backups are in `/private/tmp/pixelforge-psd-scratchpad/`.

## Build and tests

- [off](final/tests-off.log): test result: ok. 383 passed; 0 failed; 18 ignored; 0 measured; 0 filtered out; finished in 2.63s

- [default](final/tests-default.log): test result: ok. 383 passed; 0 failed; 18 ignored; 0 measured; 0 filtered out; finished in 76.23s

- [force](final/tests-force.log): test result: ok. 383 passed; 0 failed; 18 ignored; 0 measured; 0 filtered out; finished in 83.82s

[Release build](final/build.log) completed without compiler warnings. The previously documented Clippy findings are not claimed fixed. The optional ML backend/models were not downloaded.

## Object Finder and tool budgets

All four fresh tool suites are recorded below. Times are milliseconds; the precompute budget is 2000 ms.

| Tier | Previous precompute | New precompute | Previous click | New click | Tool budget misses |
|---|---:|---:|---:|---:|---:|
| [24 MP](final/tiers/tools-24.log) | 2487.10 | 81.62 | 2.81 | 1.83 | 0 |
| [61 MP](final/tiers/tools-61.log) | 560.77 | 53.88 | 3.98 | 3.01 | 0 |
| [150 MP](final/tiers/tools-150.log) | 4494.35 | 1463.04 | 6.64 | 5.36 | 0 |
| [268 MP](final/tiers/tools-268.log) | 674.88 | 61.91 | 9.00 | 6.95 | 0 |

Repeated alternating comparisons use the preserved pre-follow-up executable. Raw samples and changing candidate counts are retained; matching labels normalize the counts, not the operations.

| Image / operation | Before median | After median | Change |
|---|---:|---:|---:|
| 8192², nine pairs: generate test scene | 114.70 | 118.00 | +2.9% |
| 8192², nine pairs: object selection: first result (coarse GrabCut) | 13.11 | 13.61 | +3.8% |
| 8192², nine pairs: object selection: refined (1 MP GrabCut + guided edge) | 135.54 | 136.43 | +0.7% |
| 8192², nine pairs: Select Subject (GrabCut) | 67.48 | 64.96 | -3.7% |
| 8192², nine pairs: object finder precompute (objects, background) | 696.01 | 61.65 | -91.1% |
| 8192², nine pairs: object finder click → selection (guided upsample) | 4.99 | 3.80 | -23.8% |
| 150 MP, seven pairs: generate test scene | 282.40 | 278.10 | -1.5% |
| 150 MP, seven pairs: object selection: first result (coarse GrabCut) | 16.47 | 16.28 | -1.2% |
| 150 MP, seven pairs: object selection: refined (1 MP GrabCut + guided edge) | 105.61 | 102.35 | -3.1% |
| 150 MP, seven pairs: Select Subject (GrabCut) | 147.44 | 140.21 | -4.9% |
| 150 MP, seven pairs: object finder precompute (objects, background) | 4927.76 | 1491.55 | -69.7% |
| 150 MP, seven pairs: object finder click → selection (guided upsample) | 5.80 | 5.06 | -12.8% |

Comparisons above 5%: none. The earlier 283-operation comparison remains linked in the original report; unrelated CPU implementations were not changed by this follow-up.

## GPU Auto validation

The full four-tier calibration and independent full GPU runs are retained in [training](training/runs.json) and [validation](validation/runs.json). The first independent run had 0/2/3/2 misses at 24/61/150/268 MP, compared with the original 6/15/8/4. Seven affected tier/filter combinations were then calibrated from seven fresh runs each, using medians; the resulting priors were frozen and rebuilt before a separate seven-run validation. Both gradient variants were checked. The other full-run rows used unchanged estimates. All benchmarks ran serially.

| Final focused validation | CPU median ms | GPU median ms | Auto median ms | 10% + 0.5 ms budget | Pass | Individual misses / runs |
|---|---:|---:|---:|---:|---|---:|
| 150 MP / Gradient Linear (Chrome) | 24.5 | 29.7 | 24.5 | 27.5 | True | 0/7 |
| 150 MP / Gradient Angle (Chrome) | 31.4 | 26.1 | 26.1 | 29.2 | True | 0/7 |
| 150 MP / Gallery: Plaster | 376.6 | 302.9 | 302.9 | 333.7 | True | 0/7 |
| 150 MP / Gallery: Sponge | 527.8 | 433.7 | 433.7 | 477.6 | True | 0/7 |
| 268 MP / Gallery: Diffuse Glow | 329.4 | 281.5 | 281.5 | 310.2 | True | 0/7 |
| 268 MP / Gallery: Stamp | 230.2 | 243.9 | 230.2 | 253.7 | True | 0/7 |
| 61 MP / Gallery: Glowing Edges | 194.2 | 151.4 | 151.4 | 167.0 | True | 0/7 |
| 61 MP / Gallery: Note Paper | 44.7 | 41.6 | 44.7 | 46.3 | True | 0/7 |

See [all final samples](final/auto-validation.json), [calibration samples](repeat-calibration/samples.json) and [GPU checks](final/checks.json) and [final source checks](final/fused-checks.json). Individual outliers are retained rather than hidden by the median. This is a measured decision model, not a guarantee under arbitrary concurrent machine load.

Intermediate sizes, not used to train their ranges:

- [intermediate-12000x12000-gradient](final/intermediate-12000x12000-gradient.log): no misses.

- [intermediate-8000x6000-gaussian-blur](final/intermediate-8000x6000-gaussian-blur.log): no misses.

- [intermediate-8000x6000-glowing-edges](final/intermediate-8000x6000-glowing-edges.log): no misses.

Training used the then-current inclusive upper boundary; retained per-tier snapshots were remapped to the final half-open bins before rebuilding. Two gradient variants were separated using the retained 0.1 ms rows; subsequent 150 MP gradient calibration uses the full-precision per-run rate records. The calibration and verification scripts and snapshots are saved alongside the logs.

## Selection quality and visual checks

The whole-mask tests require over 97% IoU for two discs on a gradient backdrop and for a subject touching the frame, and over 90% sampled IoU on the actual 150 MP shaded-ball/texture fixture. The latter initially failed, caught a merged-background regression, and passes after candidate rejection plus local refinement. A 1×1 and a uniform image must not invent objects.

The two final Object Finder captures and seven earlier, unchanged phase A smoke captures were inspected: hover and click selection, classical Select Subject, automatic and custom Content-Aware Fill, output to a new layer, texture healing, colour replacement and exact healing. RGB/8-bit mode and the image name are visible. These are UI/selection checks, not PSD interoperability proof. The classical photo selector includes the dog’s shadow and misses much of the portrait’s dark hair; these heuristic limits are visible and are not advertised as Photoshop-equivalent selection.

- [pA_object_finder_click_selects](final/demo-fused/pA_object_finder_click_selects.png)

- [pA_object_finder_hover](final/demo-fused/pA_object_finder_hover.png)

- [pA_caf_applied_new_layer](demo/pA_caf_applied_new_layer.png)

- [pA_caf_custom_sampling_rotation_mirror](demo/pA_caf_custom_sampling_rotation_mirror.png)

- [pA_caf_workspace_auto](demo/pA_caf_workspace_auto.png)

- [pA_color_replacement_limits](demo/pA_color_replacement_limits.png)

- [pA_heal_quality_exact](demo/pA_heal_quality_exact.png)

- [pA_select_subject_classical](demo/pA_select_subject_classical.png)

- [pA_spot_heal_create_texture](demo/pA_spot_heal_create_texture.png)

## Scope and next phases

The 150 MP **32-bit** / 20-history-state memory stress gate still belongs to Phase 4. Phase 1’s RGBA8 pressure result is not a substitute. `.pforge` remains native; no PSD/PSB, 16-bit, HDR or raw implementation was begun here. Photoshop comparison timings from the preceding Phase 1 run remain applicable as historical measurements; no new Photoshop round-trip claim is made.

Separate ready-to-use prompts: [Phase 2](../../prompts/phase-2-psd-psb.txt), [Phase 3](../../prompts/phase-3-16-bit.txt), [Phase 4](../../prompts/phase-4-float-hdr.txt), [Phase 5](../../prompts/phase-5-raw.txt), [Phase 6](../../prompts/phase-6-speed-at-depth.txt).
