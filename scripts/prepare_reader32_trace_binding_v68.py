"""Local-only held remote-path packet from existing read-only evidence."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/preference_program/manifests/reader32_import_diagnostic_v67_approved/source'
OUT=ROOT/'outputs/preference_program/manifests/reader32_import_trace_v68_local'
REMOTE='/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 evidence_path=ROOT/'outputs/preference_program/results/reader32_trace_binding_v68/remote_readonly_evidence.json'
 evidence=json.loads(evidence_path.read_text())
 prior=json.loads((OUT/'manifest.json').read_text())
 localhashes={str(p):digest(p) for p in BASE.iterdir() if p.is_file()}
 for name in ('reader32_trace_privacy_v68.py','reader32_trace_capture_v68.py'):
  p=ROOT/'scripts'/name;localhashes[str(p)]=digest(p)
 wrapper=ROOT/'scripts/slurm/reader32_import_trace_v68_held.slurm'
 localhashes[str(wrapper)]=digest(wrapper)
 # No modified baseline or invented remote source identity.
 remote_seen={Path(e['path']).name:e['sha256'] for e in evidence['source']}
 for path,h in localhashes.items():
  if str(BASE) in path and remote_seen.get(Path(path).name)!=h:raise ValueError('existing source mismatch')
 shared=[]
 for e in evidence['package_files']:
  if e['sha256']:
   entry={k:e[k] for k in ('path','sha256','bytes','mtime_ns')}
   entry['label']=e['path'].split('/site-packages/')[1]
   entry['verified_aliases']=[] if e['resolved_path']==e['path'] else [e['resolved_path']]
   shared.append(entry)
 runtime={Path(e['path']).name:e for e in evidence['runtime']}
 descriptor={'receipt_path':runtime['runtime_receipt.json']['path'],'receipt_sha256':runtime['runtime_receipt.json']['sha256'],
  'archive_path':runtime['transformers.tar']['path'],'archive_sha256':runtime['transformers.tar']['sha256'],
  'archive_bytes':runtime['transformers.tar']['bytes'],'extractor_sha256':localhashes[str(BASE/'extract_verified_runtime.py')],
  'node_local_recipe':'${SLURM_TMPDIR:-/tmp}/lp-reader32-import-v68-${SLURM_JOB_ID}/transformers',
  'stage_receipt':'spool/runtime_stage.json; pinned extractor JSON checked before root admission',
  'python_sha256_from_prior_cpu_receipt':evidence['prior_python_sha256'],
  'python_fresh_hash':False,'python_live_metadata':runtime['python'],
  'scope':'archive and extractor identity verified; full shared native dependency closure NOT verified'}
 prior.update(source_hashes=localhashes,runtime_descriptor=descriptor,shared_exact_files=shared,
  source_baseline='All 28 existing remote v67 source files match local frozen source; source remains unchanged',
  evidence_sha256=digest(evidence_path),metadata_budget={'combined_bytes':65536,'runtime_stage_max':15360,'capture_receipt_max':49152,'wrapper_receipt_max':1024},
  remaining_gates=['independent review','fresh single CPU approval','permitted future remote materialization and hash verification','compute-node strace permissions/syntax tested only in approved sole attempt'])
 (OUT/'manifest.json').write_text(json.dumps(prior,indent=2)+'\n')
 remote=json.loads(json.dumps(prior))
 remote['diagnostic_source']=REMOTE+'/source'
 remote['source_hashes']={REMOTE+('/source/' if str(BASE) in p else '/controller/')+Path(p).name:h for p,h in localhashes.items()}
 remote['runtime_roots']=[{'path':REMOTE+'/source','verified':True,'identity':'admitted only after verify_binding hashes all frozen copied source files'}]
 remote['binding_status']='PROPOSED paths; no remote v68 materialization exists or has been checked'
 remote['existing_source_origin']='/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-diagnostic-v67/source'
 remote['verified_symlink_aliases']=[{'path':e['path'],'alias':a,'label':e['label']} for e in shared for a in e['verified_aliases']]
 (OUT/'remote_manifest_held.json').write_text(json.dumps(remote,indent=2)+'\n')
 # Frozen local controller copies; no remote upload or write.
 controller=OUT/'controller';controller.mkdir(exist_ok=True)
 for p in (ROOT/'scripts/reader32_trace_capture_v68.py',ROOT/'scripts/reader32_trace_privacy_v68.py',wrapper):
  (controller/p.name).write_bytes(p.read_bytes())
 receipt={'remote_writes':0,'jobs_submitted':0,'cpu_admission':False,'remote_materialization_ready':False,
  'source_files_verified':len(remote_seen),'source_mismatches':[], 'shared_exact_files_hashed':len(shared),
  'evidence_sha256':digest(evidence_path),'remote_manifest_sha256':digest(OUT/'remote_manifest_held.json'),
  'controller_hashes':{p.name:digest(p) for p in controller.iterdir() if p.is_file()},
  'full_shared_package_content_verified':False,'archive_verified':True,'unchanged_prior_charges':{'allocated_cpu_seconds':2417,'MIG_slice_seconds':254}}
 (OUT/'binding_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
