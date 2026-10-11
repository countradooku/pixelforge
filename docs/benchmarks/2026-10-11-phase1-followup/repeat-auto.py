import pathlib,json,time,subprocess,os,statistics,re
out=pathlib.Path('docs/benchmarks/2026-10-11-phase1-followup'); dest=out/'repeat-calibration';dest.mkdir(exist_ok=True)
while True:
 try: done=any(r['name']=='demo-runner' for r in json.loads((out/'checks.json').read_text()))
 except (FileNotFoundError,json.JSONDecodeError): done=False
 if done: break
 time.sleep(1)
runs=json.loads((out/'validation/runs.json').read_text());targets=set()
for run in runs:
 if run['suite']!='gpu':continue
 for miss in run['budget_misses']:
  label=miss.removeprefix('OVER: ').split(' @ ')[0]
  if label.startswith('Gallery: '): name=label.removeprefix('Gallery: ')
  elif label.startswith('Gradient '): name='Gradient'
  elif label.startswith('Gaussian Blur '): name='Gaussian Blur'
  else: raise RuntimeError('Need selector for '+label)
  targets.add((run['tier'],name))
(dest/'targets.json').write_text(json.dumps(sorted(targets),indent=2))
p=pathlib.Path('src/engine/gpu/priors.txt');original=p.read_text();(dest/'priors-before.txt').write_text(original)
def rates(s):return dict(l.rsplit('=',1) for l in s.splitlines() if '=' in l and not l.startswith(('#','@')))
base=rates(original);samples={};results=[]
for tier,name in sorted(targets):
 for rep in range(7):
  slug=name.lower().replace(' ','-');log=dest/f'{tier}-{slug}-{rep}.log'
  with log.open('w') as f:
   proc=subprocess.run(['target/release/pixelforge','--bench-gpu','--tier',tier,name,'--write-priors'],stdout=f,stderr=subprocess.STDOUT,env=os.environ|{'PF_GPU':'force','PF_CONFIG_DIR':str((dest/'config').resolve())},timeout=300)
  if proc.returncode:raise SystemExit(proc.returncode)
  current=rates(p.read_text())
  area={'24':25,'61':26,'150':28,'268':29}[tier]
  prefix=name.lower()+' '
  for key,value in current.items():
   if key.lower().startswith(prefix) and key.endswith(f' E{area}'):
    samples.setdefault(key,[]).append(tuple(map(float,value.split())))
  results.append({'tier':tier,'filter':name,'repeat':rep,'log':str(log)})
  print(results[-1],flush=True)
  p.write_text(original)
for key,observed in samples.items():
 base[key]=' '.join(f'{statistics.median(v[i] for v in observed):.4f}' for i in (0,1))
heads=[l for l in original.splitlines() if l.startswith(('#','@'))]
p.write_text('\n'.join(heads+[key+'='+value for key,value in sorted(base.items())])+'\n')
(dest/'samples.json').write_text(json.dumps(samples,indent=2));(dest/'priors-after.txt').write_bytes(p.read_bytes())
for name,cmd in [('build-final-2',['cargo','build','--release','--locked']),('tests-off-2',['cargo','test','--release','--locked'])]:
 with (out/(name+'.log')).open('w') as f:
  proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ|{'PF_GPU':'off','PF_CONFIG_DIR':str((dest/'config').resolve())},timeout=600)
 print(name,proc.returncode,flush=True)
 if proc.returncode:raise SystemExit(proc.returncode)
