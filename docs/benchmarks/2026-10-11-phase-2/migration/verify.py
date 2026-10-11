"""Native writer/re-save verification; each Photoshop call has its own watchdog."""
import hashlib,json,os,struct,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
corpus=root.parent/'corpus'
records=[]
def run(name,cmd):
 t=time.monotonic()
 with (root/(name+'.log')).open('w') as f:
  p=subprocess.run(cmd,env=os.environ|{'PF_CONFIG_DIR':str(root/'config'/name)},stdout=f,stderr=subprocess.STDOUT)
 row=dict(name=name,command=cmd,exit_code=p.returncode,seconds=time.monotonic()-t)
 records.append(row);(root/'runs.json').write_text(json.dumps(records,indent=2));print(name,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
def private(path):
 data=path.read_bytes();pos=26
 n=struct.unpack_from('>I',data,pos)[0];pos+=4+n
 n=struct.unpack_from('>I',data,pos)[0];pos+=4;end=pos+n;out=[]
 while pos<end:
  assert data[pos:pos+4]==b'8BIM';rid=struct.unpack_from('>H',data,pos+4)[0];pos+=6
  n=data[pos]+1;pos+=(n+1)&~1
  n=struct.unpack_from('>I',data,pos)[0];pos+=4;payload=data[pos:pos+n];pos+=(n+1)&~1
  if rid==4000:out.append(payload.hex())
 assert pos==end
 return out
for ext in ['psd','psb']:
 path=root/('two-pixel-layers.'+ext)
 run('convert-'+ext,['./target/release/pixelforge','--convert-pforge',str(root/'two-pixel-layers.pforge'),str(path)])
 run('check-'+ext,['./target/release/pixelforge','--psd-check',str(path),str(root/('checked.'+ext)),str(root/('check-'+ext+'.json'))])
 resave=root/('resave-'+ext)
 run('native-'+ext,['python3','tools/photoshop_psd.py','native-verify',str(corpus),'--input-file',str(path),'--output',str(root/('native-'+ext)),'--reference-output',str(root/('native-'+ext+'-reference')),'--resave-output',str(resave),'--defer-fidelity'])
 saved=[p for p in resave.rglob('*') if p.suffix.lower() in ['.psd','.psb']]
 assert len(saved)==1,saved
 run('resaved-native-'+ext,['python3','tools/photoshop_psd.py','native-verify',str(corpus),'--input-file',str(saved[0]),'--output',str(root/('resaved-native-'+ext)),'--reference-output',str(root/('resaved-native-'+ext+'-reference')),'--defer-fidelity'])
 before,after=private(path),private(saved[0]);assert before,'Missing writer private resource'
 result=dict(original=str(path),resaved=str(saved[0]),original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),resaved_sha256=hashlib.sha256(saved[0].read_bytes()).hexdigest(),private_original=before,private_resaved=after,private_retained=before==after)
 (root/('private-'+ext+'.json')).write_text(json.dumps(result,indent=2));print('private-'+ext,result['private_retained'],flush=True)
