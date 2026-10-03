"""Offline CPU cache integrity receipt; no model or GPU allocation."""
import hashlib,json,os,time
from pathlib import Path
root=Path('/projects/paco0228/.caches/hf/models--Qwen--Qwen2.5-1.5B-Instruct')
revision=(root/'refs/main').read_text().strip();snapshot=root/'snapshots'/revision
report={'revision':revision,'snapshot':str(snapshot.resolve()),'files':[],'started':time.time()}
for p in sorted(snapshot.iterdir()):
 if not p.is_file():continue
 start=time.time();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 digest=h.hexdigest();target=p.resolve()
 if p.suffix=='.safetensors':
  assert len(target.name)==64 and digest==target.name,(p,digest,target.name)
  from safetensors import safe_open
  with safe_open(str(p),framework='pt',device='cpu') as f:assert len(f.keys())>0
 report['files'].append({'file':p.name,'size':p.stat().st_size,'sha256':digest,'read_seconds':time.time()-start})
from transformers import AutoTokenizer
start=time.time();tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
assert tokenizer.encode('Preference check')
report.update(tokenizer_seconds=time.time()-start,completed=time.time(),valid=True)
Path(os.environ['RECEIPT']).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
