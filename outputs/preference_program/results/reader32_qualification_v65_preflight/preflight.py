import json,hashlib,sys,time
from pathlib import Path
r=Path('/scratch/alpine/paco0228/latent_proxy_runs/reader32-qualification-v65');s=r/'source'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((s/'execution_freeze.json').read_text());assert f['GPU_admission'] and f['resource_amendment_approved']
for n,h in f['source_hashes'].items():assert sha(s/n)==h,n
assert sha(s/'approval_binding.json')==f['approval_binding_sha256']
assert f['GPU']=={'type':'h200_3g.71gb','count':1,'cpu':6,'host_mem_GiB':64,'wall_seconds':300,'startup_seconds':90,'generation_end_seconds':270,'complete_seconds':285,'first_generation_seconds':30,'remaining_generation_seconds':10}
c=json.loads((s/'cpu_receipt.json').read_text());assert c['complete'] and sha(s/'cpu_receipt.json')==f['CPU_receipt_sha256'];assert c['python']==sys.version and sha(sys.executable)==c['python_sha256']
assert sha(s/'startup_receipt.json')==f['startup_receipt_sha256']
snap=Path('/scratch/alpine/paco0228/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd')
for n,e in c['files'].items():
 p=snap/n;t=p.stat();assert t.st_size==e['bytes'] and t.st_mtime_ns==e['mtime_ns'] and str(p.resolve())==e['target'],n
 if not n.endswith('.safetensors'):assert sha(p)==e['sha256'],n
runtime=Path('/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json');assert sha(runtime)==c['runtime_receipt_sha256']
reg=json.loads((s/'registration.json').read_text())
for n,h in reg['hashes'].items():assert sha(s/n)==h,n
assert sha(s/'prepare_eligibility_priority.py')==reg['scorer_sha256']
assert len((s/'cases.jsonl').read_text().splitlines())==len((s/'prompts.jsonl').read_text().splitlines())==16
assert not (r/'spool/smoke').exists() and not (r/'spool/submission_claim.json').exists()
receipt={'complete':True,'at_unix':time.time(),'CPU_only':True,'GPU_jobs':0,'model_calls':0,'exact_source_hashes_valid':True,'Python_identity_valid':True,'cached_snapshot_metadata_valid':True,'cached_nonweight_digests_valid':True,'prior_full_weight_hash_custody_retained':True,'weight_digest_rehashed_now':False,'runtime_receipt_valid':True,'frozen16_scientific_hashes_valid':True,'one_attempt_approval_bound':True,'freeze_sha256':sha(s/'execution_freeze.json')}
(r/'spool/preflight.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
