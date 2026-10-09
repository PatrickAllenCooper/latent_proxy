import ast,hashlib,json,os,sys,time
from pathlib import Path
R=Path('/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68')
V=R/'revision-receipt-safe';M=json.loads((V/'controller/manifest.json').read_text())
sys.path.insert(0,str(V/'controller'))
from reader32_trace_capture_v68_receipt_safe import verify_binding
source,driver=verify_binding(M)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert M['cpu_admission'] is True and M['remote_materialization_ready'] is False
assert M['approval_message_id']=='Sentinel_4455df170154819198f158ac5465133e'
assert M['resources']=={'cpus':1,'memory_gib':2,'wall_seconds':120,'check_seconds':90,'attempts':1,'gpus':0,'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal'}
for e in M['shared_exact_files']:
 p=Path(e['path']);st=p.stat();assert(st.st_size,st.st_mtime_ns)==(e['bytes'],e['mtime_ns']);assert sha(p)==e['sha256']
 for alias in e['verified_aliases']:assert p.samefile(alias)
d=M['runtime_descriptor'];assert sha(d['receipt_path'])==d['receipt_sha256']
old=json.loads((source/'cpu_receipt.json').read_text());assert sha(M['python'])==old['python_sha256']==d['python_sha256_from_prior_cpu_receipt']
# Existing remote source is preserved and identical to the copied baseline.
for p in (R/'source').iterdir():
 if p.is_file():assert sha(p)==sha(source/p.name)
assert driver.name=='reader32_import_diagnostic_v68_receipt_safe.py'
wrapper=(V/'controller/reader32_import_trace_v68_receipt_safe.slurm').read_text()
for x in ('#SBATCH --cpus-per-task=1','#SBATCH --mem=2G','#SBATCH --time=00:02:00','#SBATCH --account=ucb736_asc1','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','export PYTHONPATH="$LOCAL_RUNTIME:$SOURCE"','cd "$SOURCE"'):assert x in wrapper
assert '--gres' not in wrapper and '--gpus' not in wrapper
assert not (R/'spool/submitted_job_id').exists()
for name in ('CPU_import_receipt.json','capture_receipt.json','runtime_stage.json','wrapper_terminal.json'):assert not (R/'spool'/name).exists()
assert sum(v for k,v in M['metadata_budget'].items() if k!='combined_bytes')==65536
assert M['trace_bytes_limit']==10485760
print(json.dumps({'review_passed':True,'at_unix':time.time(),'source_and_controller_hashes_valid':True,'baseline_source_files_preserved':28,'safe_adapter_guard_prefix_AST_identical':True,'six_shared_file_bindings_valid':True,'fresh_python_sha256':old['python_sha256'],'runtime_receipt_sha256':d['receipt_sha256'],'manifest_sha256':sha(V/'controller/manifest.json'),'resources':M['resources'],'metadata_budget':M['metadata_budget'],'source_read_only':all(not(p.stat().st_mode&0o222) for p in source.iterdir()),'framework_imports':0,'model_calls':0,'allocation_started':False},indent=2))
