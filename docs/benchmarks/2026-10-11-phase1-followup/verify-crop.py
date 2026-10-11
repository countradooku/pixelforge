import pathlib,subprocess,os,json,time
out=pathlib.Path('docs/benchmarks/2026-10-11-phase1-followup');fin=out/'final'
records=[]
def run(name,cmd,mode=None):
 env=os.environ|{'PF_CONFIG_DIR':str((fin/'config').resolve())};env.pop('PF_GPU',None)
 if mode:env['PF_GPU']=mode
 t=time.monotonic()
 with (fin/(name+'.log')).open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 records.append({'name':name,'exit_code':r.returncode,'seconds':round(time.monotonic()-t,2)})
 (fin/'crop-checks.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
 if r.returncode:raise SystemExit(r.returncode)
run('build',['cargo','build','--release','--locked'])
run('object-comparison-150',['python3','tools/phase1_compare.py','/private/tmp/pixelforge-psd-scratchpad/pixelforge-before-followup','--suite','tools','--only','object','--size','14204x10652','--reps','7','--out',str(fin/'object-comparison-150-crop')])
run('tests-off',['cargo','test','--release','--locked'],'off')
run('tests-default',['cargo','test','--release','--locked'])
run('tests-force',['cargo','test','--release','--locked'],'force')
run('tools',['python3','tools/phase1_benchmarks.py','--suites','tools','--out',str(fin/'tiers')])
run('object-comparison',['python3','tools/phase1_compare.py','/private/tmp/pixelforge-psd-scratchpad/pixelforge-before-followup','--suite','tools','--only','object','--reps','9','--out',str(fin/'object-comparison-crop')])
run('demo',['python3','tools/phase1_demo.py','--phase','A','--mb','4096','--out',str(fin/'demo-crop')])
