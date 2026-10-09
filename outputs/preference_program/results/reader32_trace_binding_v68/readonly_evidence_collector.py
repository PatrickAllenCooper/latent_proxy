import hashlib,json,os,time
from pathlib import Path
ROOT=Path('/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-diagnostic-v67/source')
ENV=Path('/projects/paco0228/software/anaconda/envs/latent-proxy-env')
RUNTIME=Path('/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39')
budget=64*1024*1024;used=0

def evidence(p,maximum):
 global used
 st=p.stat();resolved=p.resolve()
 result={'path':str(p),'resolved_path':str(resolved),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'device':st.st_dev,'inode':st.st_ino,'is_symlink':p.is_symlink()}
 if st.st_size<=maximum and used+st.st_size<=budget:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b);used+=len(b)
  result['sha256']=h.hexdigest()
 else:result['sha256']=None;result['hash_not_read']='bounded-read limit'
 return result
r={'read_only':True,'at_unix':time.time(),'read_bytes_cap':budget,'source':[],'runtime':[],'package_files':[],'package_roots':[]}
for p in sorted(ROOT.iterdir()):
 if p.is_file():r['source'].append(evidence(p,2*1024*1024))
for p in (RUNTIME/'runtime_receipt.json',RUNTIME/'transformers.tar',ENV/'bin/python'):
 r['runtime'].append(evidence(p,53*1024*1024))
receipt=json.loads((ROOT/'cpu_receipt.json').read_text())
r['prior_python_sha256']=receipt['python_sha256']
site=ENV/'lib/python3.12/site-packages'
for n in ('PIL','torch'):
 p=site/n;st=p.stat();r['package_roots'].append({'path':str(p),'resolved_path':str(p.resolve()),'device':st.st_dev,'inode':st.st_ino,'full_content_verified':False})
for n in ('PIL/Image.py','torch/__init__.py','torch/distributed/rpc/__init__.py','torch/_jit_internal.py','torch/nn/functional.py'):
 r['package_files'].append(evidence(site/n,1024*1024))
# Exact known Pillow native module only; bounded directory listing, no recursive scan.
for p in sorted((site/'PIL').glob('_imaging.cpython-312-*.so'))[:2]:r['package_files'].append(evidence(p,8*1024*1024))
# Native Torch closure is large: metadata only for these known library paths.
for n in ('torch/lib/libtorch_global_deps.so','torch/lib/libtorch_python.so','torch/lib/libtorch_cpu.so'):
 p=site/n
 if p.is_file():r['package_files'].append(evidence(p,0))
r['read_bytes']=used;r['finished_at_unix']=time.time()
print(json.dumps(r,indent=2))
