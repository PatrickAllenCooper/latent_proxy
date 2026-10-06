"""CPU-only exact frozen code/runtime/case/token/model gate before GPU submission."""
import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);started=time.time();m=json.loads((root/'execution_freeze.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(2**20):h.update(b)
 return h.hexdigest()
for name,want in m['remote_hashes'].items():assert sha(root/name)==want,name
cache=json.loads((root/'cache_receipt.json').read_text());assert cache['valid'];token=json.loads((root/'token_receipt.json').read_text());assert token['complete'] and token['serialization_equality'] and token['restoration_valid'];assert len(token['prompt_ids'])==16
for f in cache['files']:
 assert sha(Path(cache['snapshot'])/f['file'])==f['sha256'],f['file']
print(json.dumps({'event':'model_files_verified','at':time.time()}),flush=True)
archive=Path('/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/transformers.tar');assert sha(archive)=='8e91c3771d157877b4deb2492e9149f3e802bc71886e4d5e2ee2f4fcdbc1f282'
assert token['manifest_sha256']==sha(root/'manifest.jsonl') and token['cache_receipt_sha256']==sha(root/'cache_receipt.json') and token['normalizer_sha256']==sha(root/'prompt_token_ids.py')
r={'complete':True,'started':started,'completed':time.time(),'freeze_sha256':sha(root/'execution_freeze.json'),'snapshot':cache['snapshot'],'files_verified':cache['files'],'remote_hashes':m['remote_hashes'],'token_receipt_sha256':sha(root/'token_receipt.json'),'CPU_only':True,'model_calls':0};(root/'hash_gate.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'event':'hash_gate_complete','elapsed':time.time()-started}),flush=True)
