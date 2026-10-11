import subprocess,json,time
from pathlib import Path
out=Path(__file__).parent
before='/private/tmp/pixelforge-psd-scratchpad/phase-2-2026-10-11-153448/pixelforge-before'
runs=[]
for suite,only in [('engine',None),('tools','layers'),('gpu','compositing')]:
 folder=out/('comparison-'+suite)
 command=['python3','tools/phase2_compare.py',before,'--out',str(folder),'--suite',suite,'--reps','9']
 if only: command+=['--only',only]
 t=time.monotonic()
 with (out/('comparison-'+suite+'.log')).open('w') as f: result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
 runs.append(dict(suite=suite,command=command,exit_code=result.returncode,seconds=time.monotonic()-t))
 (out/'comparison-runs.json').write_text(json.dumps(runs,indent=2));print(suite,result.returncode,flush=True)
 if result.returncode:break
