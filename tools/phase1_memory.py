#!/usr/bin/env python3
"""Capture process memory and macOS paging counters around a pressure run."""
import argparse, json, os, pathlib, subprocess
p=argparse.ArgumentParser()
p.add_argument('--tier',default='150');p.add_argument('--mb',default='512')
p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('docs/benchmarks/2026-10-11-phase1'))
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
def counters():
 return {key:subprocess.check_output(cmd,text=True) for key,cmd in [('vm_stat',['vm_stat']),('swapusage',['sysctl','vm.swapusage']),('ram',['sysctl','hw.memsize'])]}
record={'tier':a.tier,'budget_mb':a.mb,'before':counters()}
with (a.out/f'memory-{a.tier}-{a.mb}.log').open('w') as f:
 r=subprocess.run(['target/release/pixelforge','--bench-memory','--tier',a.tier,'pressure'],env=os.environ|{'PF_GPU':'off','PF_MEMORY_MB':a.mb,'PF_CONFIG_DIR':str((a.out/'config').resolve())},stdout=f,stderr=subprocess.STDOUT,timeout=1800)
record['after']=counters();record['exit_code']=r.returncode
(a.out/f'memory-{a.tier}-{a.mb}-system.json').write_text(json.dumps(record,indent=2))
raise SystemExit(r.returncode)
