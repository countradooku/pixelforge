import json,subprocess,time
from pathlib import Path
root=Path(__file__).parent
before='/private/tmp/pixelforge-psd-scratchpad/phase-2-2026-10-11-153448/pixelforge-before'
runs=[]
for suite in ['engine','gpu']:
 r=json.loads((root/f'comparison-{suite}/report.json').read_text())
 tiers=sorted({x['tier'] for x in r['regressions_over_5_percent']},key=int)
 if not tiers:continue
 command=['python3','tools/phase2_compare.py',before,'--out',str(root/f'focus-{suite}'),'--suite',suite,'--tiers',','.join(tiers),'--reps','15']
 if suite=='gpu':command+=['--only','compositing']
 t=time.monotonic()
 with (root/f'focus-{suite}.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 runs.append(dict(suite=suite,command=command,exit_code=result.returncode,seconds=time.monotonic()-t))
 (root/'focus-runs.json').write_text(json.dumps(runs,indent=2));print(suite,result.returncode,flush=True)
 if result.returncode:break
