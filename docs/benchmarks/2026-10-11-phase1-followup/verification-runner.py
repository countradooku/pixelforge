import pathlib,time,json,subprocess,os
out=pathlib.Path('docs/benchmarks/2026-10-11-phase1-followup');train=out/'training'
while not (train/'priors-after-268.txt').exists(): time.sleep(1)
p=pathlib.Path('src/engine/gpu/priors.txt')
s=p.read_text(); lines=[]
# The training executable used ceil(log2(px)); the final half-open intervals
# move exact powers of two to the next bin. Only the 268 MP row is affected.
for line in s.splitlines():
 if ' E28=' in line: line=line.replace(' E28=',' E29=')
 if line.startswith('# op key'): line='# op key (name + padding bucket [+ E<n>: [2^(n-1), 2^n) pixels, GPU end to end minus fixed cost; L: legacy large]) = GPU ns/px CPU ns/px, Apple M5 Pro.'
 lines.append(line)
for line in (train/'priors-after-150.txt').read_text().splitlines():
 if ' E28=' in line: lines.append(line)
heads=[l for l in lines if l.startswith(('#','@'))]; rates=sorted(l for l in lines if l and not l.startswith(('#','@')))
assert len({l.split('=')[0] for l in rates})==len(rates)
p.write_text('\n'.join(heads+rates)+'\n')
# Gradient algorithms previously shared one key. Recover separate linear
# and angle measurements from the retained training rows (0.1 ms precision).
import re
ops={l.split('=')[0]:l.split('=')[1] for l in rates}
for tier,px in [('24',6000*4000),('61',9504*6336),('150',14204*10652),('268',16384*16384)]:
 log=(train/f'gpu-{tier}.log').read_text()
 fixed=float(re.search(r'cost model: fixed ([0-9.]+)',log)[1])
 rows=[l for l in log.splitlines() if l.startswith('Gradient grad:')]
 assert len(rows)==2
 for variant,line in zip(['',' angle'],rows):
  nums=line[44:].split();cpu,gpu,res=map(float,nums[:3]);key=f'Gradient{variant} p0'
  ops[f'{key} E{px.bit_length()}']=f'{max(0,gpu-fixed)*1e6/px:.4f} {cpu*1e6/px:.4f}'
  if tier=='61':ops[key]=f'{res*1e6/px:.4f} {cpu*1e6/px:.4f}'
  if tier=='268':ops[f'{key} L']=f'{max(0,gpu-fixed)*1e6/px:.4f} {cpu*1e6/px:.4f}'
p.write_text('\n'.join(heads+[k+'='+v for k,v in sorted(ops.items())])+'\n')
(out/'trained-priors.txt').write_bytes(p.read_bytes())
results=[]
def run(name,cmd,mode=None):
 t=time.monotonic();env=os.environ|{'PF_CONFIG_DIR':str((out/'config').resolve())}
 env.pop('PF_GPU',None)
 if mode:env['PF_GPU']=mode
 with (out/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 results.append({'name':name,'exit_code':p.returncode,'seconds':round(time.monotonic()-t,2)})
 (out/'checks.json').write_text(json.dumps(results,indent=2));print(results[-1],flush=True)
 if p.returncode:raise SystemExit(p.returncode)
run('build-final',['cargo','build','--release','--locked'])
run('tests-off',['cargo','test','--release','--locked'],'off')
run('tests-default',['cargo','test','--release','--locked'])
run('tests-force',['cargo','test','--release','--locked'],'force')
run('tier-runner',['python3','tools/phase1_benchmarks.py','--suites','tools,gpu','--out',str(out/'validation')])
run('object-comparison',['python3','tools/phase1_compare.py','/private/tmp/pixelforge-psd-scratchpad/pixelforge-before-followup','--suite','tools','--only','object','--reps','9','--out',str(out/'object-comparison')])
run('demo-runner',['python3','tools/phase1_demo.py','--phase','A','--mb','4096','--out',str(out/'demo')])
