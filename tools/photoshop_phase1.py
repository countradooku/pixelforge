#!/usr/bin/env python3
"""Run the Phase 1 comparison with an independent CoreGraphics dialog watchdog."""
import argparse, json, pathlib, subprocess, tempfile, time
p = argparse.ArgumentParser()
p.add_argument('fixture', type=pathlib.Path)
p.add_argument('output', type=pathlib.Path)
p.add_argument('--timeout', type=int, default=300)
a = p.parse_args()
# Stage the locally authored script outside Desktop, as the existing Photoshop
# harness does, so opening it does not require a Desktop-folder permission.
staged = tempfile.TemporaryDirectory(prefix='pixelforge-photoshop-')
script = pathlib.Path(staged.name) / 'benchmark.jsx'
script.write_text(pathlib.Path(__file__).with_suffix('.jsx').read_text())
# JSON strings quote both paths as JavaScript data, never shell code.
js = f'var INPUT={json.dumps(str(a.fixture.resolve()))}; var OUTPUT={json.dumps(str(a.output.resolve()))}; $.evalFile(new File({json.dumps(str(script))}));'
apple = f'with timeout of {a.timeout + 30} seconds\n' + 'tell application "Adobe Photoshop 2026" to do javascript ' + json.dumps(js) + '\nend timeout'
watch = '''ObjC.import('CoreGraphics'); ObjC.import('Foundation');
var list=ObjC.deepUnwrap(ObjC.castRefToObject($.CGWindowListCopyWindowInfo($.kCGWindowListOptionOnScreenOnly | $.kCGWindowListExcludeDesktopElements,0)));
JSON.stringify(list.filter(function(w){return (w.kCGWindowOwnerName||'').indexOf('Photoshop')>=0;}).map(function(w){return {id:w.kCGWindowNumber,name:w.kCGWindowName||'',layer:w.kCGWindowLayer,bounds:w.kCGWindowBounds};}));'''
def windows():
    r = subprocess.run(['osascript', '-l', 'JavaScript', '-e', watch], text=True, capture_output=True, timeout=10, check=True)
    return json.loads(r.stdout)
baseline = {w['id'] for w in windows()}
process = subprocess.Popen(['osascript', '-e', apple], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
started=time.monotonic(); observations=[]; seen={}
try:
    while process.poll() is None:
        current = windows()
        # Dialogs use either a modal layer or a centered non-document window.
        dialogs = [w for w in current if w['id'] not in baseline and (
            w['layer'] > 0 or any(s in w['name'].lower() for s in ['camera raw','warning','error','color profile','missing profile','mismatch']) or
            (w['bounds']['Width'] < 950 and w['bounds']['Height'] < 650 and w['bounds']['Width'] > 220 and w['bounds']['Height'] > 100 and not any(s in w['name'].lower() for s in ['.tif','.png','.psd','.psb']))) ]
        observations.append({'seconds':round(time.monotonic()-started,1),'windows':current,'dialogs':dialogs})
        seen={w['id']:seen.get(w['id'],time.monotonic()) for w in dialogs}
        persistent=[w for w in dialogs if time.monotonic()-seen[w['id']] >= 3]
        if persistent: raise RuntimeError('Photoshop dialog detected; scripting bridge terminated: '+json.dumps(persistent))
        if time.monotonic()-started > a.timeout: raise TimeoutError('Photoshop benchmark exceeded watchdog timeout')
        time.sleep(1)
    out,err=process.communicate(timeout=10)
    print(out,end=''); print(err,end='')
    if process.returncode or 'ERROR\t' in out: raise RuntimeError('Photoshop reported a benchmark failure')
finally:
    if process.poll() is None:
        process.terminate()
        try: process.wait(timeout=3)
        except subprocess.TimeoutExpired: process.kill()
    a.output.with_suffix('.watchdog.json').write_text(json.dumps(observations,indent=2))
