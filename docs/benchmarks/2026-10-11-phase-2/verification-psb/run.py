import hashlib,json,os,subprocess,time
from pathlib import Path
out=Path(__file__).parent
runs=[]
commands=[('build',['cargo','build','--release','--offline','--locked'],None)]
for mode,gpu in [('default',None),('off','off'),('force','force')]:
 commands.append(('tests-'+mode,['cargo','test','--release','--offline','--locked','--','--test-threads=1'],gpu))
commands.append(('authored-text',['cargo','test','--release','--offline','--locked','authored_photoshop_text_files_decode_preserve_and_refuse_stale_global_cache','--','--ignored','--nocapture'],None))
for name,command,gpu in commands:
 env=os.environ|{'PF_CONFIG_DIR':str((out/('config-'+name)).resolve()),'PIXELFORGE_PSD_TEXT_CORPUS':str(Path('docs/benchmarks/2026-10-11-phase-2/corpus/originals').resolve())}
 env.pop('PF_GPU',None)
 if gpu: env['PF_GPU']=gpu
 start=time.monotonic();before=hashlib.sha256(Path('src/engine/document.rs').read_bytes()).hexdigest()
 with (out/(name+'.log')).open('w') as f: code=subprocess.run(['/usr/bin/time','-l']+command,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 after=hashlib.sha256(Path('src/engine/document.rs').read_bytes()).hexdigest()
 runs.append(dict(name=name,command=command,exit_code=code,seconds=time.monotonic()-start,PF_GPU=gpu or 'default',config=env['PF_CONFIG_DIR'],document_before=before,document_after=after))
 (out/'runs.json').write_text(json.dumps(runs,indent=2));print(name,code,flush=True)
 if code or before!=after: break
