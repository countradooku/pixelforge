#!/usr/bin/env python3
"""Alternating before/after medians against a preserved pre-change executable."""
import argparse, json, os, pathlib, re, statistics, subprocess
p=argparse.ArgumentParser()
p.add_argument('before',type=pathlib.Path)
p.add_argument('--suite',choices=['engine','tools','filters'],default='engine')
p.add_argument('--reps',type=int,default=15)
p.add_argument('--size',default='8192')
p.add_argument('--only',help='Existing tools/filter group selector, passed to both executables')
p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('docs/benchmarks/2026-10-11-phase1'))
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);result={}
env=os.environ|{'PF_GPU':'off','PF_CONFIG_DIR':str((a.out/'compare-config').resolve())}
flag={'engine':'--bench','tools':'--bench-tools','filters':'--bench-filters'}[a.suite]
for i in range(a.reps):
 for name in (['before','after'] if i%2==0 else ['after','before']):
  binary=str(a.before.resolve()) if name=='before' else 'target/release/pixelforge'
  r=subprocess.run([binary,flag,a.size]+([a.only] if a.only else []),env=env,capture_output=True,text=True,check=True,timeout=1800)
  (a.out/f'legacy-{a.suite}-{name}-{i}.log').write_text(r.stdout+r.stderr)
  for line in r.stdout.splitlines():
   m=re.match(r'(.+?)\s+([\d.]+) ms\b',line)
   if m:
    label=m[1].strip().replace(', 268 MP','')
    # Segmentation improvements can change the reported candidate/hit count;
    # compare the operation, retaining those counts in the raw logs.
    label=re.sub(r'\(\d+ objects, background\)', '(objects, background)', label)
    label=re.sub(r'\(\d+/2000 on an object\)', '(sampled objects)', label)
    result.setdefault(label,{}).setdefault(name,[]).append(float(m[2]))
for name,data in result.items():
 if 'before' in data and 'after' in data:
  old,new=(statistics.median(data[k]) for k in ('before','after'))
  data.update(before_median=old,after_median=new,change_percent=(new/old-1)*100 if old else None)
  if old and new>old*1.05: print(f'{name}: {old:.3f} -> {new:.3f} ms (+{data["change_percent"]:.1f}%)',flush=True)
(a.out/f'legacy-{a.suite}-comparison.json').write_text(json.dumps(result,indent=2))
