#!/usr/bin/env python3
"""Run serially: benchmarks must not compete with one another for CPU/GPU/RAM."""
import argparse, json, os, pathlib, subprocess, time
p=argparse.ArgumentParser()
p.add_argument('--out', type=pathlib.Path, default=pathlib.Path('docs/benchmarks/2026-10-11-phase1'))
p.add_argument('--tiers', default='24,61,150,268')
p.add_argument('--suites', default='engine,tools,filters,gpu')
p.add_argument('--timeout', type=int, default=1800)
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
env=os.environ|{'PF_CONFIG_DIR':str((a.out/'config').resolve())}
results=json.loads((a.out/'runs.json').read_text()) if (a.out/'runs.json').exists() else []
for suite in a.suites.split(','):
 for tier in a.tiers.split(','):
  flag={'engine':'--bench','tools':'--bench-tools','filters':'--bench-filters','gpu':'--bench-gpu'}[suite]
  cmd=['target/release/pixelforge',flag,'--tier',tier]
  started=time.monotonic(); filename=a.out/f'{suite}-{tier}.log'
  print(f'Running {suite} {tier} MP',flush=True)
  try:
   with filename.open('w') as f:
    proc=subprocess.run(cmd,env=env|{'PF_GPU':'force' if suite=='gpu' else 'off'},stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout)
   code=proc.returncode
  except subprocess.TimeoutExpired: code='timeout'
  over=[line.strip() for line in filename.read_text().splitlines() if 'OVER:' in line or '— OVER' in line]
  record={'budget_misses':over,'suite':suite,'tier':tier,'exit_code':code,'seconds':round(time.monotonic()-started,2),'log':str(filename)}
  results=[r for r in results if (r['suite'],r['tier']) != (suite,tier)]; results.append(record); (a.out/'runs.json').write_text(json.dumps(results,indent=2))
  print(json.dumps(record),flush=True)
  if code != 0: raise SystemExit(1)
