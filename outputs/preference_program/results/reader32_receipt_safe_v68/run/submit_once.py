import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68');V=R/'revision-receipt-safe'
m_path=V/'controller/manifest.json';m=json.loads(m_path.read_text())
assert hashlib.sha256(m_path.read_bytes()).hexdigest()=='892d2420fd5f7cddaf7644a8d348a97d5372484ce774582b1cf676c28b596333'
assert m['cpu_admission'] and m['remote_materialization_ready'] and m['approved_attempt_unused']
sys.path.insert(0,str(V/'controller'))
from reader32_trace_capture_v68_receipt_safe import verify_binding
verify_binding(m)
for e in m['shared_exact_files']:
 p=Path(e['path']);st=p.stat();assert (st.st_size,st.st_mtime_ns)==(e['bytes'],e['mtime_ns']);assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
queue=subprocess.run(['squeue','-u','paco0228','-h','-n','lp-reader32-import-v68','-o','%i'],capture_output=True,text=True,check=True)
assert not queue.stdout.strip(),'existing project job blocks submission'
claim={'at_unix':time.time(),'source_commit':'dcf4a17','approval_message_id':m['approval_message_id'],'manifest_sha256':hashlib.sha256(m_path.read_bytes()).hexdigest(),'resources':m['resources'],'attempt':1,'maximum_attempts':1}
with (R/'spool/submission_claim').open('x') as f:f.write(json.dumps(claim)+'\n');f.flush();os.fsync(f.fileno())
r=subprocess.run(['sbatch','--parsable','--account=ucb736_asc1',str(V/'controller/reader32_import_trace_v68_receipt_safe.slurm')],capture_output=True,text=True,check=True)
job=r.stdout.strip().split(';')[0];assert job.isdigit()
with (R/'spool/submitted_job_id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
fd=os.open(str(R/'spool'),os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps({'job_id':job,'submitted_at_unix':time.time(),'resources':m['resources'],'source_commit':'dcf4a17','manifest_sha256':claim['manifest_sha256'],'retry':False},indent=2))
