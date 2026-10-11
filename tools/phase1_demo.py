#!/usr/bin/env python3
"""Capture only the newly started demo's window; never close other instances."""
import argparse, os, pathlib, selectors, subprocess, time
parser=argparse.ArgumentParser();parser.add_argument('--mb',default='512');parser.add_argument('--phase',default='memory');parser.add_argument('--out',type=pathlib.Path,default=pathlib.Path('docs/benchmarks/2026-10-11-phase1'));args=parser.parse_args()
out=args.out;out.mkdir(parents=True,exist_ok=True)
config=out/'demo-config';config.mkdir(exist_ok=True)
p=subprocess.Popen(['target/release/pixelforge'],env=os.environ|{'PF_DEMO':'tools','PF_DEMO_PHASE':args.phase,'PF_MEMORY_MB':args.mb,'PF_CONFIG_DIR':str(config.resolve()),'PF_GPU':'off'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
watch=f'''ObjC.import('CoreGraphics'); ObjC.import('Foundation'); ObjC.import('AppKit');
$.NSRunningApplication.runningApplicationWithProcessIdentifier({p.pid}).activateWithOptions(2);
var list=ObjC.deepUnwrap(ObjC.castRefToObject($.CGWindowListCopyWindowInfo($.kCGWindowListOptionOnScreenOnly | $.kCGWindowListExcludeDesktopElements,0)));
var w=list.filter(function(w){{return w.kCGWindowOwnerPID=={p.pid} && w.kCGWindowLayer==0 && w.kCGWindowBounds.Width>500;}})[0];
w ? String(w.kCGWindowNumber) : '';'''
started=time.monotonic(); sel=selectors.DefaultSelector();sel.register(p.stdout,selectors.EVENT_READ)
try:
 with (out/'demo.log').open('w') as log:
  while p.poll() is None:
   if time.monotonic()-started > 180: raise TimeoutError('demo timed out')
   for key,_ in sel.select(1):
    line=key.fileobj.readline();log.write(line);log.flush()
    if line.startswith('SHOT '):
     name=line.strip().split()[1]
     wid=subprocess.check_output(['osascript','-l','JavaScript','-e',watch],text=True).strip()
     if not wid: raise RuntimeError('demo window not found')
     time.sleep(.25)
     subprocess.run(['screencapture','-x','-o','-l'+wid,str(out/(name+'.png'))],check=True)
     print(name,flush=True)
  log.write(p.stdout.read())
 if p.returncode: raise RuntimeError('demo failed: '+str(p.returncode))
 # This file was generated solely to prove the native-format status label.
 (config/'Phase 1 memory.pforge').unlink(missing_ok=True)
finally:
 if p.poll() is None:
  p.terminate();p.wait(timeout=5)
