"""CPU-only package archive for node-local import staging; preserve shared source."""
import hashlib,json,os,tarfile,time
from pathlib import Path
start=time.time();package=Path('/projects/paco0228/software/anaconda/envs/latent-proxy-env/lib/python3.12/site-packages/transformers')
assert package.is_dir()
root=Path('/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39');root.mkdir(exist_ok=True)
archive=root/'transformers.tar'
with tarfile.open(archive,'w') as tar:tar.add(package,arcname='transformers',filter=lambda info:None if '__pycache__' in info.name else info)
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
report={'source':str(package),'archive':str(archive),'sha256':h.hexdigest(),'archive_bytes':archive.stat().st_size,'elapsed_seconds':time.time()-start,'valid':True,'purpose':'Same installed Transformers code staged as archive for node-local extraction; no dependency version change'}
(root/'runtime_receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
