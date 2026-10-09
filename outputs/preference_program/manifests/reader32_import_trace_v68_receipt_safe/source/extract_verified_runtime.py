"""Extract a hash-verified package archive into a per-job node-local directory."""
import hashlib,json,os,tarfile,time
from pathlib import Path
start=time.time();receipt=json.loads(Path(os.environ['RUNTIME_RECEIPT']).read_text());archive=Path(receipt['archive']);h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==receipt['sha256']
out=Path(os.environ['LOCAL_RUNTIME']);out.mkdir(parents=True,exist_ok=False)
with tarfile.open(archive) as tar:tar.extractall(out,filter='data')
assert (out/'transformers/__init__.py').is_file()
print(json.dumps({'event':'node_local_runtime_ready','at_unix':time.time(),'elapsed_seconds':time.time()-start,'path':str(out),'archive_sha256':h.hexdigest()}),flush=True)
