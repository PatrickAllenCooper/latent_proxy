"""Review-only CPU dependency staging builder. Does not import staged packages.
Caller must enforce finite CPU/time/memory caps; preserve partial tree on failure.
Python source is copied, other assets are explicit source links with content hashes.
"""
import hashlib,json,os
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def stage_packages(site,destination,packages):
 site=Path(site).resolve();destination=Path(destination);destination.mkdir(exist_ok=False);records=[]
 for package in packages:
  if package not in ('torch','sympy','accelerate'):raise ValueError('unregistered package')
  source=site/package
  if not source.is_dir() or source.is_symlink():raise ValueError('missing/symlink package root')
  for parent,dirs,files in os.walk(source,followlinks=False):
   if any((Path(parent)/d).is_symlink() for d in dirs):raise ValueError('symlink directory needs separate custody')
   dirs[:]=[d for d in dirs if d!='__pycache__']
   for name in files:
    p=Path(parent)/name
    if p.suffix in ('.pyc','.pyo'):continue
    target=destination/p.relative_to(site);target.parent.mkdir(parents=True,exist_ok=True)
    before=p.stat();digest=sha(p)
    if p.suffix=='.py':
     # Copy exact bytes; hash-check again after copy and reject source mutation.
     with p.open('rb') as src,target.open('xb') as dst:
      for b in iter(lambda:src.read(2**20),b''):dst.write(b)
     mode='copied_python'
    else:target.symlink_to(p);mode='linked_asset'
    after=p.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or sha(target)!=digest:raise ValueError('source mutated during staging')
    records.append({'source':str(p),'staged':str(target),'sha256':digest,'bytes':after.st_size,'mode':mode})
    with (destination/'staging_progress.jsonl').open('a') as journal:journal.write(json.dumps(records[-1])+'\n')
    if len(records)%100==0:print(json.dumps({'stage':'staging_progress','files':len(records),'bytes':sum(r['bytes'] for r in records)}),flush=True)
 receipt={'packages':list(packages),'files':records,'model_calls':0,'import_equivalence_verified':False,'full_environment_identity':False,'qualification_ready':False}
 (destination/'staging_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

def verify_staged(receipt):
 for r in receipt['files']:
  if sha(r['source'])!=r['sha256'] or sha(r['staged'])!=r['sha256']:raise ValueError('stale or tampered dependency')
 return True
