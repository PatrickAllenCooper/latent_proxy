"""Stdlib-only fixed calibration contract; no model/qualification code."""
import hashlib
import json
import math
import os
from pathlib import Path
import re

RESOURCES={'cpus':1,'memory_MiB':256,'wall_seconds':30,'check_seconds':20,
 'attempts':1,'gpus':0,'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal'}
BUDGET={'total':65536,'preparation':15360,'capture':46080,'toy':3072,'terminal':1024}
PAYLOAD=(b'latent_proxy owned tracer calibration\n')*2
PAYLOAD_SHA256=hashlib.sha256(PAYLOAD).hexdigest()
PHASES=('calibration_toy_start','calibration_toy_complete')
MAX_SOURCE_BYTES=256*1024

class CalibrationStop(RuntimeError):pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def admission(manifest, env, now):
    if manifest.get('calibration_admission') is not True or manifest.get('allocation_approved') is not True:
        raise CalibrationStop('calibration held')
    if manifest.get('purpose')!='stdlib_owned_tracer_calibration':raise CalibrationStop('wrong purpose')
    if manifest.get('resources')!=RESOURCES or manifest.get('metadata_budget')!=BUDGET:
        raise CalibrationStop('contract mismatch')
    if manifest.get('attempts_used')!=0 or manifest.get('retry') is not False:
        raise CalibrationStop('attempt hold')
    approval=manifest.get('fresh_approval_message_id')
    if type(approval) is not str or not re.fullmatch('[A-Za-z0-9_-]{1,128}',approval) or approval=='Sentinel_4455df170154819198f158ac5465133e':raise CalibrationStop('fresh approval required')
    start=float(env.get('STUDY_WALL_START','0'))
    if not math.isfinite(start) or now>=start+20:raise CalibrationStop('shell deadline')
    expected={'SLURM_JOB_PARTITION':'acpu','SLURM_CPUS_PER_TASK':'1','SLURM_MEM_PER_NODE':'256','SLURM_JOB_ACCOUNT':'ucb736_asc1'}
    if any(env.get(k)!=v for k,v in expected.items()):raise CalibrationStop('allocation mismatch')
    if not str(env.get('SLURM_JOB_ID','')).isdigit():raise CalibrationStop('job identity')
    if any(env.get(k,'') not in ('','0') for k in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE')):
        raise CalibrationStop('GPU forbidden')
    return start

def validate_source(manifest):
    roots=(Path(manifest['source_dir']),Path(manifest['controller_dir']))
    if any(not p.is_absolute() or '..' in p.parts or p.is_symlink() for p in roots):raise CalibrationStop('source roots')
    declared={p:set() for p in roots}
    for name in manifest['source_hashes']:
        path=Path(name)
        if path.parent not in roots or path.is_symlink():raise CalibrationStop('source scope')
        declared[path.parent].add(path.name)
    for root in roots:
        allowed=declared[root] | ({'manifest.json'} if root==roots[1] else set())
        for p in root.iterdir():
            if p.name not in allowed or not p.is_file() or p.is_symlink():raise CalibrationStop('unexpected source entry')
    total=0
    for name,expected in manifest['source_hashes'].items():
        path=Path(name);total+=path.stat().st_size
        if total>MAX_SOURCE_BYTES:raise CalibrationStop('source byte bound')
        if sha(path)!=expected:raise CalibrationStop('source identity')
    return total

# The schema accepts only fixed fields, finite numeric measurements and hashes.
FIELDS={'preparation':{'kind':1,'source_sha256':str,'python_sha256':str,'strace_sha256':str,'at':float,'source_bytes':int,'model_calls':0,'framework_imports':0,'GPU_allocations':0},
 'toy':{'kind':2,'payload_sha256':str,'bytes_read':int,'started':float,'completed':float,'calibration_only':True,'qualification_established':False,'model_calls':0,'framework_imports':0,'GPU_allocations':0},
 'terminal':{'kind':3,'exit_code':int,'at':float,'calibration_only':True,'qualification_established':False,'model_calls':0,'GPU_allocations':0}}

def encode_metadata(kind, record, limit=None):
    if kind not in FIELDS or type(record) is not dict or set(record)!=set(FIELDS[kind]):
        raise CalibrationStop('metadata schema')
    for key,expected in FIELDS[kind].items():
        value=record[key]
        if expected is str:
            if type(value) is not str or not re.fullmatch('[0-9a-f]{64}',value):raise CalibrationStop('metadata hash')
        elif expected is float:
            if type(value) not in (float,int) or not 0<=value<=1e12 or not math.isfinite(value):raise CalibrationStop('metadata time')
        elif expected is int:
            if type(value) is not int or not 0<=value<=MAX_SOURCE_BYTES:raise CalibrationStop('metadata numeric')
        elif type(value) is not type(expected) or value!=expected:raise CalibrationStop('metadata fixed field')
    if kind=='toy' and (record['bytes_read']!=len(PAYLOAD) or record['completed']<record['started']):raise CalibrationStop('toy result')
    cap=BUDGET[kind] if limit is None else min(limit,BUDGET[kind])
    data=(json.dumps(record,allow_nan=False,sort_keys=True,separators=(',',':'))+'\n').encode()
    if len(data)>cap:raise CalibrationStop('metadata byte cap')
    return data

def write_metadata(path, kind, record, limit=None):
    data=encode_metadata(kind,record,limit)
    with Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(str(Path(path).parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return len(data)
