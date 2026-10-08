"""Proposed CPU-only literal-path gate. No frameworks, weight reads, or fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

SNAPSHOT=Path('/scratch/alpine/paco0228/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd')
RESOURCES={'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal','CPUs':1,'host_mem_GiB':2,'wall_seconds':120,'check_seconds':90,'attempts':1,'GPUs':0}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def compute_context(env,hostname):
    assert env.get('SLURM_JOB_ID','').isdigit(),'Slurm compute job required'
    assert env.get('SLURM_JOB_PARTITION')=='acpu','acpu CPU partition required'
    assert env.get('SLURM_JOB_ACCOUNT')=='ucb736_asc1','approved account required'
    assert env.get('SLURM_CPUS_PER_TASK')=='1','one CPU required'
    assert env.get('SLURM_MEM_PER_NODE')=='2048','2GiB host request required'
    assert env.get('SLURMD_NODENAME','').split('.')[0]==hostname.split('.')[0],'compute-node hostname mismatch'
    assert not hostname.lower().startswith('login'),'login-node evidence is insufficient'
    for key in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE'):
        assert env.get(key,'') in ('','0'),'GPU allocation forbidden'
    return {'hostname':hostname,'job_id':env['SLURM_JOB_ID'],'partition':'acpu','account':env['SLURM_JOB_ACCOUNT'],'GPUs':0}

def audit_paths(snapshot,files):
    rows=[]
    for name,expected in files.items():
        assert Path(name).name==name,'unsafe snapshot member'
        path=Path(snapshot)/name
        try:
            stat=path.stat();actual={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'target':str(path.resolve())}
            changed=[k for k in actual if actual[k]!=expected[k]]
            try:alias_samefile=os.path.samefile(path,expected['target'])
            except OSError:alias_samefile=None
            # Preserve the exact literal guard; alias equality never upgrades it.
            digest_match=None if name.endswith('.safetensors') else sha(path)==expected['sha256']
            rows.append({'file':name,'expected':{k:expected[k] for k in actual},'actual':actual,'changed':changed,'alias_samefile':alias_samefile,'nonweight_digest_match':digest_match,'literal_identity_matches':not changed and digest_match is not False,'weight_content_rehashed':False})
        except OSError as exc:
            rows.append({'file':name,'literal_identity_matches':False,'error':type(exc).__name__+': '+str(exc)})
    assert rows,'empty custody cannot pass'
    return {'literal_gate_matches_on_this_CPU_node':all(r['literal_identity_matches'] for r in rows),'files':rows,'weight_content_rehashed':False,'GPU_node_namespace_verified':False,'scientific_qualification':None,'model_calls':0,'GPU_jobs':0}

def write_new(path,receipt):
    with Path(path).open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n');f.flush()

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--approval',type=Path,required=True);p.add_argument('--spool',type=Path,required=True);args=p.parse_args()
    src=args.source.resolve();spool=args.spool.resolve()
    assert src!=spool and src not in spool.parents,'spool must be outside frozen source'
    approval=json.loads(args.approval.read_text())
    assert approval.get('CPU_admission') is True,'CPU allocation not approved'
    assert approval['resources']==RESOURCES
    assert approval['checker_sha256']==sha(__file__)
    context=compute_context(os.environ,socket.gethostname())
    output=spool/('path-check-'+context['job_id']+'.json')
    assert not output.exists(),'never overwrite a prior check'
    start=float(os.environ['STUDY_WALL_START']);done=threading.Event()
    def watch():
        if not done.wait(max(0,start+90-time.time())):
            try:write_new(spool/('path-check-'+context['job_id']+'-stop.json'),{'complete':False,'reason':'90_second_CPU_check_deadline','at_unix':time.time(),'context':context,'GPU_jobs':0,'model_calls':0})
            finally:os._exit(2)
    threading.Thread(target=watch,daemon=True).start()
    freeze_path=src/'execution_freeze.json';assert sha(freeze_path)==approval['source_freeze_sha256']
    freeze=json.loads(freeze_path.read_text())
    for name,digest in freeze['source_hashes'].items():assert sha(src/name)==digest,name
    receipt_path=src/'cpu_receipt.json';assert sha(receipt_path)==freeze['CPU_receipt_sha256']
    prior=json.loads(receipt_path.read_text());assert prior['complete']
    assert prior['python']==sys.version and sha(sys.executable)==prior['python_sha256']
    assert len(prior['files'])==24 and sum(n.endswith('.safetensors') for n in prior['files'])==17
    result=audit_paths(SNAPSHOT,prior['files']);result.update(complete=True,context=context,started_unix=start,completed_unix=time.time(),prior_receipt_sha256=sha(receipt_path),source_freeze_sha256=sha(freeze_path),resources=RESOURCES)
    assert time.time()<start+90,'CPU check deadline exhausted'
    write_new(output,result);done.set();print(json.dumps(result),flush=True)
    return 0 if result['literal_gate_matches_on_this_CPU_node'] else 2

if __name__=='__main__':sys.exit(main())
