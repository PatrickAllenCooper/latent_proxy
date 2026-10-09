"""Build local held source/controller packet. No SSH, jobs or runtime imports."""
import hashlib,json
from pathlib import Path
from reader32_calibration_contract_v69 import PAYLOAD,PAYLOAD_SHA256,RESOURCES,BUDGET
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/preference_program/manifests/reader32_tracer_calibration_v69_local'
REMOTE='/scratch/alpine/paco0228/latent_proxy_runs/tracer-calibration-v69'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=OUT/'source';controller=OUT/'controller'
    source.mkdir(parents=True,exist_ok=True);controller.mkdir(exist_ok=True)
    for name in ('reader32_calibration_toy_v69.py','reader32_calibration_contract_v69.py'):
        (source/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    (source/'calibration_input.bin').write_bytes(PAYLOAD)
    for name in ('reader32_tracer_calibration_v69.py','reader32_calibration_terminal_v69.py','reader32_calibration_contract_v69.py','reader32_trace_capture_v69_calibration.py','reader32_trace_privacy_v68.py'):
        (controller/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    wrapper=ROOT/'scripts/slurm/reader32_tracer_calibration_v69_held.slurm'
    (controller/wrapper.name).write_bytes(wrapper.read_bytes())
    manifest={'purpose':'stdlib_owned_tracer_calibration','calibration_admission':False,'allocation_approved':False,'remote_materialization_ready':False,'fresh_approval_message_id':None,'attempts_used':0,'retry':False,'resources':RESOURCES,'metadata_budget':BUDGET,'trace_cap_bytes':10485760,
      'source_dir':str(source),'controller_dir':str(controller),
      'source_hashes':{str(p):sha(p) for folder in (source,controller) for p in folder.iterdir() if p.is_file()},
      'runtime':{'python':'/projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python','python_sha256':'9d27cfcde7e3128a1b0cd86e7bf8020fc1c8210dbc485bcf878ab7ef8a5c7516','python_digest_origin':'verified live before33635575; not freshly verified this local session','strace':'/usr/bin/strace','strace_sha256':None,'strace_identity_live_verified':False},
      'payload_sha256':PAYLOAD_SHA256,'payload_bytes':len(PAYLOAD),'source_max_bytes':262144,'qualification_established':False,'model_calls':0,'framework_imports':0,
      'required_future_gates':['independent review','fresh explicit one-attempt allocation approval','live site and allocation headroom','bounded read-only tracer/runtime identity binding','separate materialization and all frozen hash checks','compute-node tracing permission/syntax established only by approved calibration'],
      'historical_accounting_preserved':{'CPU_seconds':2425,'MIG_slice_seconds':254},'no_remote_writes':True}
    (OUT/'manifest_local_held.json').write_text(json.dumps(manifest,indent=2)+'\n')
    remote=json.loads(json.dumps(manifest));remote['source_dir']=REMOTE+'/source';remote['controller_dir']=REMOTE+'/controller'
    remote['source_hashes']={REMOTE+('/source/' if Path(p).parent==source else '/controller/')+Path(p).name:h for p,h in manifest['source_hashes'].items()}
    (OUT/'manifest_remote_held.json').write_text(json.dumps(remote,indent=2)+'\n')
    (OUT/'packet_receipt.json').write_text(json.dumps({'source_files':len(list(source.iterdir())),'controller_files':len(list(controller.iterdir())),'source_bytes':sum(p.stat().st_size for folder in (source,controller) for p in folder.iterdir()),'manifest_sha256':sha(OUT/'manifest_remote_held.json'),'source_hashes':manifest['source_hashes'],'admission':False,'allocation_approved':False,'remote_writes':0,'new_jobs':0},indent=2)+'\n')
if __name__=='__main__':main()
