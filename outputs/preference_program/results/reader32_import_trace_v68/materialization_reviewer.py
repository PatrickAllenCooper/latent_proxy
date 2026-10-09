import ast,hashlib,json,os,re,time
from pathlib import Path
R=Path('/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68')
M=json.loads((R/'controller/manifest.json').read_text());issues=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in M['source_hashes'].items():
 if sha(p)!=h:issues.append('source_hash:'+Path(p).name)
expected={'cpus':1,'memory_gib':2,'wall_seconds':120,'check_seconds':90,'attempts':1,'gpus':0,'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal'}
assert M['resources']==expected and M['cpu_admission'] is True and M['remote_materialization_ready'] is False
assert M['approval_message_id']=='Sentinel_4455df170154819198f158ac5465133e'
wrapper=(R/'controller/reader32_import_trace_v68_held.slurm').read_text()
for line in ('#SBATCH --cpus-per-task=1','#SBATCH --mem=2G','#SBATCH --time=00:02:00','#SBATCH --account=ucb736_asc1','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal','export PYTHONPATH="$LOCAL_RUNTIME:$SOURCE"','cd "$SOURCE"'):
 assert line in wrapper,line
assert '--gres' not in wrapper and '--gpus' not in wrapper
assert 'trace='+'' not in '' # no runtime trace or model import executed
controller=(R/'controller/reader32_trace_capture_v68.py').read_text()
assert "'-e','raw='+RAW" in controller and "'-p'" not in controller and "'sudo'" not in controller
assert M['trace_bytes_limit']==10485760 and sum(M['metadata_budget'][k] for k in ('runtime_stage_max','capture_receipt_max','wrapper_receipt_max'))==65536
for e in M['shared_exact_files']:
 p=Path(e['path']);st=p.stat()
 if (st.st_size,st.st_mtime_ns)!=(e['bytes'],e['mtime_ns']) or sha(p)!=e['sha256']:issues.append('shared_exact_file:'+e['label'])
 for alias in e['verified_aliases']:
  if not p.samefile(alias):issues.append('alias:'+e['label'])
d=M['runtime_descriptor'];assert sha(d['receipt_path'])==d['receipt_sha256']
receipt=json.loads((R/'source/cpu_receipt.json').read_text())
assert sha(M['python'])==receipt['python_sha256']==d['python_sha256_from_prior_cpu_receipt']
# AST-only comparison: no runtime module/package loads.
def imports(p,name):
 t=ast.parse(p.read_text());f=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
 return [ast.dump(n,include_attributes=False) for n in ast.walk(f) if isinstance(n,(ast.Import,ast.ImportFrom))]
assert imports(R/'source/reader32_import_diagnostic_v67_cpu.py','dependencies')==imports(R/'source/run_reader32_v65.py','deps')
assert not any((R/'spool').glob('*submitted*'))
print(json.dumps({'at_unix':time.time(),'review_passed':not issues,'issues':issues,'source_hashes_valid':True,'fresh_python_sha256':receipt['python_sha256'],'six_exact_file_bindings_valid':True,'runtime_receipt_sha256':d['receipt_sha256'],'manifest_pre_materialization_sha256':sha(R/'controller/manifest.json'),'resources':expected,'no_framework_imports':True,'no_model_calls':True,'trace_privacy_limits_valid':True},indent=2))
assert not issues
