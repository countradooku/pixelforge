import pathlib,subprocess,os,json,time,re,statistics
out=pathlib.Path('docs/benchmarks/2026-10-11-phase1-followup');dest=out/'final';dest.mkdir(exist_ok=True)
records=[]
def run(name,cmd,mode=None):
 env=os.environ|{'PF_CONFIG_DIR':str((dest/'config').resolve())};env.pop('PF_GPU',None)
 if mode:env['PF_GPU']=mode
 t=time.monotonic()
 with (dest/(name+'.log')).open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 records.append({'name':name,'exit_code':r.returncode,'seconds':round(time.monotonic()-t,2)})
 (dest/'checks.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
run('build',['cargo','build','--release','--locked'])
run('tests-off',['cargo','test','--release','--locked'],'off')
run('tests-default',['cargo','test','--release','--locked'])
run('tests-force',['cargo','test','--release','--locked'],'force')
run('tools',['python3','tools/phase1_benchmarks.py','--suites','tools','--out',str(dest/'tiers')])
run('object-comparison',['python3','tools/phase1_compare.py','/private/tmp/pixelforge-psd-scratchpad/pixelforge-before-followup','--suite','tools','--only','object','--reps','9','--out',str(dest/'object-comparison')])
run('object-comparison-150',['python3','tools/phase1_compare.py','/private/tmp/pixelforge-psd-scratchpad/pixelforge-before-followup','--suite','tools','--only','object','--size','14204x10652','--reps','7','--out',str(dest/'object-comparison-150')])
targets=json.loads((out/'repeat-calibration/targets.json').read_text());rows={}
for tier,name in targets:
 for rep in range(7):
  slug=name.lower().replace(' ','-');key=f'gpu-{tier}-{slug}-{rep}'
  run(key,['target/release/pixelforge','--bench-gpu','--tier',tier,name],'force')
  for line in (dest/(key+'.log')).read_text().splitlines():
   m=re.match(r'(.{44})\s+([\d.]+)\s+([\d.]+)\s+(?:[\d.]+|—)\s+(?:[\d.]+|—)\s+(CPU|GPU)\s+(ok|OVER)',line)
   if m:
    label,cpu,gpu,auto,status=m.groups();cpu=float(cpu);gpu=float(gpu)
    rows.setdefault(f'{tier} MP / {label.strip()}',[]).append({'cpu_ms':cpu,'gpu_ms':gpu,'auto':auto,'auto_ms':cpu if auto=='CPU' else gpu,'status':status})
summary={}
for name,samples in rows.items():
 cpu,gpu,auto=(statistics.median(r[k] for r in samples) for k in ['cpu_ms','gpu_ms','auto_ms'])
 summary[name]={'cpu_median_ms':cpu,'gpu_median_ms':gpu,'auto_median_ms':auto,'budget_ms':min(cpu,gpu)*1.1+.5,'pass':auto<=min(cpu,gpu)*1.1+.5,'samples':samples}
(dest/'auto-validation.json').write_text(json.dumps(summary,indent=2))
print('Auto median misses:',[k for k,v in summary.items() if not v['pass']],flush=True)
# Intermediate sizes exercise the same trained ranges away from the tier sizes.
for size,name in [('8000x6000','Gaussian Blur'),('12000x12000','Gradient'),('8000x6000','Glowing Edges')]:
 run('intermediate-'+size+'-'+name.lower().replace(' ','-'),['target/release/pixelforge','--bench-gpu',size,name],'force')
run('demo',['python3','tools/phase1_demo.py','--phase','A','--mb','4096','--out',str(dest/'demo')])
