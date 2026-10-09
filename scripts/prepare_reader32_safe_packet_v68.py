"""Freeze a LOCAL held output-safe revision; no remote writes or admission."""
import hashlib,json
from pathlib import Path
from prepare_reader32_safe_driver_v68 import main as derive_driver,OUT,BASE,ROOT

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 derive_driver()
 prior=ROOT/'outputs/preference_program/manifests/reader32_import_trace_v68_local'
 manifest=json.loads((prior/'manifest.json').read_text())
 controller=OUT/'controller';controller.mkdir(exist_ok=True)
 copies=(ROOT/'scripts/reader32_trace_capture_v68_receipt_safe.py',ROOT/'scripts/reader32_trace_privacy_v68.py',ROOT/'scripts/slurm/reader32_import_trace_v68_receipt_safe.slurm')
 for p in copies:(controller/p.name).write_bytes(p.read_bytes())
 manifest.update(cpu_admission=False,remote_materialization_ready=False,allocation_submitted=False,
  diagnostic_source=str(OUT/'source'),driver_name='reader32_import_diagnostic_v68_receipt_safe.py',
  source_hashes={str(p):sha(p) for folder in (OUT/'source',controller) for p in folder.iterdir() if p.is_file()},
  metadata_budget={'combined_bytes':65536,'runtime_stage_max':15360,'capture_receipt_max':46080,'CPU_import_receipt_max':3072,'wrapper_receipt_max':1024},
  runtime_roots=[{'path':str(OUT/'source'),'verified':True,'identity':'all baseline and diagnostic-only files bound'}],
  baseline_driver_sha256=sha(BASE/'reader32_import_diagnostic_v67_cpu.py'),
  derivation_receipt_sha256=sha(OUT/'derivation_receipt.json'),
  approval_message_id='Sentinel_4455df170154819198f158ac5465133e',approved_attempt_unused=True,
  binding_status='LOCAL revision only; existing remote v68 source/controller/archive untouched',
  remaining_gates=['independent review of output-only derivative and full packet','authorized materialization of separately versioned revision','live source/runtime/identity/admission checks before original approved attempt','own-child tracing permissions/syntax unknown until sole approved attempt'])
 (OUT/'manifest_local_held.json').write_text(json.dumps(manifest,indent=2)+'\n')
 remote=json.loads(json.dumps(manifest));prefix='/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68/revision-receipt-safe'
 remote['diagnostic_source']=prefix+'/source'
 remote['source_hashes']={prefix+('/source/' if '/source/' in p else '/controller/')+Path(p).name:h for p,h in manifest['source_hashes'].items()}
 remote['runtime_roots']=[{'path':prefix+'/source','verified':True,'identity':'requires live frozen source digest verification before admission'}]
 remote['binding_status']='PROPOSED revision paths only; not remotely materialized'
 (OUT/'manifest_remote_held.json').write_text(json.dumps(remote,indent=2)+'\n')
 proof={'source_files':len(list((OUT/'source').iterdir())),'original_baseline_files_preserved':28,'baseline_source_mismatches':[],
  'driver_output_only_derivation_sha256':sha(OUT/'derivation_receipt.json'),'manifest_remote_sha256':sha(OUT/'manifest_remote_held.json'),
  'controller_hashes':{p.name:sha(p) for p in controller.iterdir()},'cpu_admission':False,'remote_materialization_ready':False,'approved_attempt_unused':True,'remote_writes':0,'model_calls':0,'framework_imports':0}
 (OUT/'packet_receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
if __name__=='__main__':main()
