import hashlib,json,time
from pathlib import Path
R=Path('/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68');V=R/'revision-receipt-safe'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((V/'controller/manifest.json').read_text());mismatches=[Path(p).name for p,h in m['source_hashes'].items() if sha(p)!=h]
files={}
for name in ('capture_receipt.json','runtime_stage.json','wrapper_terminal.json','CPU_import_receipt.json','imports.jsonl','syscalls.jsonl','cpu-33635575.out','cpu-33635575.err','submission_claim','submitted_job_id'):
 p=R/'spool'/name
 files[name]={'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else 0,'sha256':sha(p) if p.exists() else None}
metadata=sum(files[name]['bytes'] for name in ('capture_receipt.json','runtime_stage.json','wrapper_terminal.json','CPU_import_receipt.json'))
trace=sum(files[name]['bytes'] for name in ('imports.jsonl','syscalls.jsonl'))
assert metadata<=65536 and trace<=10485760 and not mismatches
print(json.dumps({'at_unix':time.time(),'job_id':'33635575','files':files,'source_hash_mismatches':mismatches,'manifest_sha256':sha(V/'controller/manifest.json'),'trace_bytes':trace,'captured_metadata_bytes':metadata,'metadata_cap_passed':True,'trace_cap_passed':True,'original_remote_source_preserved':all(sha(p)==sha(V/'source'/p.name) for p in (R/'source').iterdir() if p.is_file()),'CPU_completion_receipt_present':files['CPU_import_receipt.json']['exists']},indent=2))
