"""CPU diagnostic: original mapping and complete distribution-file content identity.
No Transformers/tokenizer/model imports; external total cap required.
"""
import hashlib,importlib.metadata as md,json,os,sys,time
from pathlib import Path
out=Path(sys.argv[1]);assert not out.exists();out.mkdir()
def mark(stage,**kw):print(json.dumps({'stage':stage,'at':time.time(),**kw}),flush=True)
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
mark('start',pid=os.getpid(),python=sys.executable,sys_path=sys.path)
mark('original_mapping_start');mapping=md.packages_distributions();mark('original_mapping_complete',packages=len(mapping))
(out/'mapping_partial.json').write_text(json.dumps(mapping,sort_keys=True)+'\n')
records=[];seen=set();total=0;versions={}
for dist in md.distributions():
 name=dist.metadata['Name'];versions[name]=dist.version
 files=dist.files
 if files is None:raise RuntimeError('distribution without complete file manifest:'+name)
 mark('distribution_identity_start',name=name,files=len(files))
 for f in files:
  p=Path(dist.locate_file(f)).resolve()
  if str(p) in seen:continue
  seen.add(str(p))
  if not p.is_file():raise RuntimeError('missing/nonregular manifest file:'+str(p))
  size=p.stat().st_size;records.append({'path':str(p),'bytes':size,'sha256':digest(p)});total+=size
 mark('distribution_identity_complete',name=name,hashed_files=len(records),bytes=total)
# All searchable roots must be closed inventories, including undeclared modules.
for rootstr in sys.path:
 if not rootstr:rootstr=os.getcwd()
 root=Path(rootstr)
 if not root.exists():continue
 if root.is_file():
  if str(root.resolve()) not in seen:records.append({'path':str(root.resolve()),'bytes':root.stat().st_size,'sha256':digest(root)})
  continue
 for p in root.rglob('*'):
  if p.is_symlink():raise RuntimeError('unclosed symlink search path:'+str(p))
  if not p.is_file() or '__pycache__' in p.parts:continue
  resolved=str(p.resolve())
  if resolved in seen:continue
  seen.add(resolved);records.append({'path':resolved,'bytes':p.stat().st_size,'sha256':digest(p)})
# Identity is meaningful only with no environment mutation throughout capture/use.
identity={'sys_path':sys.path,'python':sys.version,'executable_sha256':digest(sys.executable),'files':sorted(records,key=lambda r:r['path']),'versions':versions}
identity_digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
receipt={'producer':'original_importlib.metadata.packages_distributions','environment_digest':identity_digest,'metadata_source_digest':digest(md.__file__),'mapping':mapping,'mapping_sha256':hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'versions':versions,'complete':True,'caveat':'Read-only capture requires immutable environment and same sys.path for use; diagnostic entrypoint sys.path differs from production until explicitly reconciled'}
(out/'environment_identity.json').write_text(json.dumps(identity)+'\n');(out/'authoritative_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');mark('complete')
