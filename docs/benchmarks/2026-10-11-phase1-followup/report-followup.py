from pathlib import Path
import json,re,hashlib,tarfile,difflib
out=Path('docs/benchmarks/2026-10-11-phase1-followup');fin=out/'final';old=Path('docs/benchmarks/2026-10-11-phase1')
s=['# Phase 1 performance follow-up — 2026-10-11\n', 'This follow-up fixes Object Finder latency and CPU/GPU Auto selection. It does not start Phase 2. The existing unified memory budget, LZ4 cold tiles and scratch storage remain in place; their pressure, corruption/recovery, compression and Photoshop timing evidence is in the [original Phase 1 report](../2026-10-11-phase1/REPORT.md).\n', '## Changes\n', '- Object Finder shares Select Subject’s boundary/colour prior instead of learning the entire inset rectangle as foreground. It rejects frame-dominated background components and locally refines replacement region candidates to recover shaded object edges. The guided edge filter now fuses four moment passes and two coefficient passes into channel batches, preserving each floating-point operation’s order; bitwise tests compare against the retained scalar reference. Click masks are cropped to cached candidate bounds with the full interpolation/filter halo; pixel-by-pixel tests compare them with the former uncropped path. Resolution and the two GrabCut iterations are retained. The graph-cut solver itself is unchanged.\n- GPU Auto learns end-to-end rates separately for power-of-two pixel-area ranges, including distinct 150 MP and 268 MP ranges. Gradient algorithms have separate keys. Learning one range no longer overwrites another; old cached models are invalidated by a schema key. Unmeasured ranges keep the previous transfer/kernel and large-image fallback estimates.\n- Added selection-mask quality tests, including the actual 150 MP textured fixture, frame contact and blank images; added cost-model isolation, boundary, serialization and gradient-variant checks. Benchmark gradient labels now identify the algorithm.\n- No dependencies were added or downloaded. Source and executable backups are in `/private/tmp/pixelforge-psd-scratchpad/`.\n', '## Build and tests\n']
for mode in ['off','default','force']:
 text=(fin/f'tests-{mode}.log').read_text(); result=next(l for l in text.splitlines() if l.startswith('test result:'))
 s.append(f'- [{mode}](final/tests-{mode}.log): {result}\n')
s.append('\n[Release build](final/build.log) completed without compiler warnings. The previously documented Clippy findings are not claimed fixed. The optional ML backend/models were not downloaded.\n')
s.append('\n## Object Finder and tool budgets\n\nAll four fresh tool suites are recorded below. Times are milliseconds; the precompute budget is 2000 ms.\n\n| Tier | Previous precompute | New precompute | Previous click | New click | Tool budget misses |\n|---|---:|---:|---:|---:|---:|\n')
runs=json.loads((fin/'tiers/runs.json').read_text())
def timing(text,stem):
 line=next(l for l in text.splitlines() if l.startswith(stem));return float(re.search(r'\s([\d.]+) ms',line)[1])
for tier in ['24','61','150','268']:
 before=(old/f'tools-{tier}.log').read_text();after=(fin/f'tiers/tools-{tier}.log').read_text();r=next(r for r in runs if r['tier']==tier)
 vals=[timing(t,stem) for stem in ['object finder precompute','object finder click'] for t in [before,after]]
 s.append(f'| [{tier} MP](final/tiers/tools-{tier}.log) | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} | {vals[3]:.2f} | {len(r["budget_misses"])} |\n')
s.append('\nRepeated alternating comparisons use the preserved pre-follow-up executable. Raw samples and changing candidate counts are retained; matching labels normalize the counts, not the operations.\n\n| Image / operation | Before median | After median | Change |\n|---|---:|---:|---:|\n')
reg=[]
for folder,label in [('object-comparison-fused','8192², nine pairs'),('object-comparison-150-fused','150 MP, seven pairs')]:
 data=json.loads((fin/folder/'legacy-tools-comparison.json').read_text())
 for key,v in data.items():
  if 'change_percent' not in v or v['change_percent'] is None:continue
  s.append(f'| {label}: {key} | {v["before_median"]:.2f} | {v["after_median"]:.2f} | {v["change_percent"]:+.1f}% |\n')
  if v['change_percent']>5:reg.append((label,key,v['change_percent']))
s.append(f'\nComparisons above 5%: {reg if reg else "none"}. The earlier 283-operation comparison remains linked in the original report; unrelated CPU implementations were not changed by this follow-up.\n')
s.append('\n## GPU Auto validation\n\nThe full four-tier calibration and independent full GPU runs are retained in [training](training/runs.json) and [validation](validation/runs.json). The first independent run had 0/2/3/2 misses at 24/61/150/268 MP, compared with the original 6/15/8/4. Seven affected tier/filter combinations were then calibrated from seven fresh runs each, using medians; the resulting priors were frozen and rebuilt before a separate seven-run validation. Both gradient variants were checked. The other full-run rows used unchanged estimates. All benchmarks ran serially.\n\n| Final focused validation | CPU median ms | GPU median ms | Auto median ms | 10% + 0.5 ms budget | Pass | Individual misses / runs |\n|---|---:|---:|---:|---:|---|---:|\n')
auto=json.loads((fin/'auto-validation.json').read_text())
for name,r in auto.items():
 bad=sum(v['status']=='OVER' for v in r['samples'])
 s.append(f'| {name} | {r["cpu_median_ms"]:.1f} | {r["gpu_median_ms"]:.1f} | {r["auto_median_ms"]:.1f} | {r["budget_ms"]:.1f} | {r["pass"]} | {bad}/{len(r["samples"])} |\n')
s.append('\nSee [all final samples](final/auto-validation.json), [calibration samples](repeat-calibration/samples.json) and [GPU checks](final/checks.json) and [final source checks](final/fused-checks.json). Individual outliers are retained rather than hidden by the median. This is a measured decision model, not a guarantee under arbitrary concurrent machine load.\n\nIntermediate sizes, not used to train their ranges:\n')
for p in sorted(fin.glob('intermediate-*.log')):
 misses=[l.strip() for l in p.read_text().splitlines() if 'OVER' in l]
 s.append(f'- [{p.stem}](final/{p.name}): {"no misses" if not misses else "; ".join(misses)}.\n')
s.append('\nTraining used the then-current inclusive upper boundary; retained per-tier snapshots were remapped to the final half-open bins before rebuilding. Two gradient variants were separated using the retained 0.1 ms rows; subsequent 150 MP gradient calibration uses the full-precision per-run rate records. The calibration and verification scripts and snapshots are saved alongside the logs.\n')
s.append('\n## Selection quality and visual checks\n\nThe whole-mask tests require over 97% IoU for two discs on a gradient backdrop and for a subject touching the frame, and over 90% sampled IoU on the actual 150 MP shaded-ball/texture fixture. The latter initially failed, caught a merged-background regression, and passes after candidate rejection plus local refinement. A 1×1 and a uniform image must not invent objects.\n\nThe two final Object Finder captures and seven earlier, unchanged phase A smoke captures were inspected: hover and click selection, classical Select Subject, automatic and custom Content-Aware Fill, output to a new layer, texture healing, colour replacement and exact healing. RGB/8-bit mode and the image name are visible. These are UI/selection checks, not PSD interoperability proof. The classical photo selector includes the dog’s shadow and misses much of the portrait’s dark hair; these heuristic limits are visible and are not advertised as Photoshop-equivalent selection.\n\n')
for p in sorted((fin/'demo-fused').glob('pA_object_finder*.png')):s.append(f'- [{p.stem}](final/demo-fused/{p.name})\n')
for p in sorted((out/'demo').glob('*.png')):
 if not p.name.startswith('pA_object_finder'):s.append(f'- [{p.stem}](demo/{p.name})\n')
s.append('\n## Scope and next phases\n\nThe 150 MP **32-bit** / 20-history-state memory stress gate still belongs to Phase 4. Phase 1’s RGBA8 pressure result is not a substitute. `.pforge` remains native; no PSD/PSB, 16-bit, HDR or raw implementation was begun here. Photoshop comparison timings from the preceding Phase 1 run remain applicable as historical measurements; no new Photoshop round-trip claim is made.\n\nSeparate ready-to-use prompts: [Phase 2](../../prompts/phase-2-psd-psb.txt), [Phase 3](../../prompts/phase-3-16-bit.txt), [Phase 4](../../prompts/phase-4-float-hdr.txt), [Phase 5](../../prompts/phase-5-raw.txt), [Phase 6](../../prompts/phase-6-speed-at-depth.txt).\n')
report = ''.join(s)
report = re.sub(r'(?m)^(#{1,6} .+)\n(?!\n)', r'\1\n\n', report)
report = re.sub(r'(?m)([^\n])\n(?=## |-[ ])', r'\1\n\n', report)
(out/'REPORT.md').write_text(report)
files=['src/engine/objects.rs','src/engine/graphcut.rs','src/engine/segment.rs','src/engine/gpu/cost.rs','src/engine/gpu/tests.rs','src/engine/gpu/priors.txt','src/bench_gpu.rs','tools/phase1_compare.py','tools/phase1_demo.py','README.md','target/release/pixelforge']
(out/'verified-files.json').write_text(json.dumps({f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},indent=2))
with tarfile.open('/private/tmp/pixelforge-psd-scratchpad/before-performance-followup.tar.gz') as t:
 diff=[]
 for f in files:
  if f.startswith('target/'):continue
  oldtext=t.extractfile('./'+f).read().decode().splitlines(keepends=True)
  diff.extend(difflib.unified_diff(oldtext,Path(f).read_text().splitlines(keepends=True),fromfile='before/'+f,tofile='after/'+f))
 (out/'changes.patch').write_text(''.join(diff))
print('Report written. Median misses:',[k for k,v in auto.items() if not v['pass']], 'CPU regressions:',reg)
