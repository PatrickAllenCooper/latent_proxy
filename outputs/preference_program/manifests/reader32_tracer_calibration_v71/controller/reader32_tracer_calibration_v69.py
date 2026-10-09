"""Separate held stdlib-only calibration. Never invokes a qualification driver."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from reader32_calibration_contract_v69 import (
    RESOURCES,BUDGET,PHASES,PAYLOAD_SHA256,CalibrationStop,admission,
    validate_source,sha,write_metadata,encode_metadata)
from reader32_trace_capture_v69_calibration import Parser,TraceStop,capture

class CalibrationParser(Parser):
    def __init__(self,sink):
        super().__init__(sink);self.toy_events=[]
    def line(self,channel,line):
        if channel!='syscalls' and line.startswith('{'):
            self.mark(channel,'phase_json')
            event=json.loads(line)
            if type(event) is not dict or set(event)!={'event','at'} or event['event'] not in PHASES:
                raise TraceStop('calibration phase rejected')
            if type(event['at']) not in (int,float) or not 0<=event['at']<=1e12:
                raise TraceStop('calibration phase time')
            expected=PHASES[len(self.toy_events)] if len(self.toy_events)<2 else None
            if event['event']!=expected:raise TraceStop('calibration phase order')
            self.toy_events.append(event['event'])
            self.sink.retain('imports','import',timestamp=time.monotonic(),result=PHASES.index(event['event']))
            return
        super().line(channel,line)
    def finish(self):
        super().finish()
        if tuple(self.toy_events)!=PHASES:raise TraceStop('calibration phases incomplete')

def build_command(manifest,source,spool,fd):
    if type(fd) is not int or fd<3:raise CalibrationStop('own pipe fd')
    from reader32_trace_capture_v69_calibration import CALLS,RAW
    return ['/usr/bin/strace','-f','-ttt','-T','-s','8192','-e','trace='+CALLS,
      '-e','raw='+RAW,'-o','/proc/self/fd/'+str(fd),manifest['runtime']['python'],
      '-S','-X','importtime','-u',str(source/'reader32_calibration_toy_v69.py'),str(source),str(spool)]

def measure_trace(spool):
    # Read only already sanitized records; no raw tracer output or line hashes.
    path=spool/'syscalls.jsonl';opens=reads=0;lines=0
    if not path.exists():return {'lines':0,'input_opens':0,'input_reads':0}
    if path.stat().st_size>10485760:raise CalibrationStop('trace cap')
    with path.open('rb') as f:
        for line in f:
            if len(line)>8192:raise CalibrationStop('sanitized line cap')
            row=json.loads(line);lines+=1
            if row.get('path')=='root0/calibration_input.bin' and not row.get('unfinished'):
                opens+=int(row['kind']=='openat' and row.get('result',-1)>=0)
                reads+=int(row['kind'] in ('read','pread64') and row.get('result',0)>0)
    return {'lines':lines,'input_opens':opens,'input_reads':reads}

def validate_outcome(spool):
    receipt=spool/'calibration_toy_receipt.json'
    if not receipt.exists() or receipt.stat().st_size>BUDGET['toy']:raise CalibrationStop('toy receipt absent or oversized')
    record=json.loads(receipt.read_text());encode_metadata('toy',record)
    if record['payload_sha256']!=PAYLOAD_SHA256:raise CalibrationStop('toy digest mismatch')
    measurements=measure_trace(spool)
    if not measurements['input_opens'] or not measurements['input_reads']:raise CalibrationStop('useful file trace unverified')
    return measurements

STAGE=0

def execute(manifest,source,spool,env):
    global STAGE
    STAGE=1
    # Held before filesystem reads or launches; consumed v68 flags cannot admit.
    start=admission(manifest,env,time.time())
    if manifest.get('remote_materialization_ready') is not True:raise CalibrationStop('materialization held')
    if str(source)!=manifest['source_dir'] or str(Path(__file__).parent)!=manifest['controller_dir']:
        raise CalibrationStop('source location')
    if source==spool or source in spool.parents:raise CalibrationStop('separate spool required')
    if env.get('PYTHONDONTWRITEBYTECODE')!='1':raise CalibrationStop('bytecode forbidden')
    if any((spool/name).exists() for name in ('calibration_preparation.json','capture_receipt.json','calibration_toy_receipt.json','calibration_terminal.json')):raise CalibrationStop('attempt artifacts already exist')
    STAGE=2
    count=validate_source(manifest)
    if manifest['runtime']['python']!='/projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python' or manifest['runtime']['strace']!='/usr/bin/strace':raise CalibrationStop('runtime path scope')
    STAGE=3
    for name in ('python','strace'):
        digest=manifest['runtime'].get(name+'_sha256')
        if type(digest) is not str or len(digest)!=64:raise CalibrationStop('runtime binding incomplete')
        if sha(manifest['runtime'][name])!=digest:raise CalibrationStop('runtime identity')
    if time.time()>=start+20:raise CalibrationStop('preparation deadline')
    STAGE=4
    write_metadata(spool/'calibration_preparation.json','preparation',{'kind':1,'source_sha256':hashlib.sha256(json.dumps(manifest['source_hashes'],sort_keys=True).encode()).hexdigest(),'python_sha256':manifest['runtime']['python_sha256'],'strace_sha256':manifest['runtime']['strace_sha256'],'at':time.time(),'source_bytes':count,'model_calls':0,'framework_imports':0,'GPU_allocations':0})
    STAGE=5
    r,w=os.pipe();reason=capture(build_command(manifest,source,spool,w),spool,(str(source),),
        time.monotonic()+max(0,start+20-time.time()),(r,w),parser_factory=CalibrationParser,phase_codes=list(PHASES))
    validate_source(manifest)
    if reason!='complete':raise CalibrationStop('calibration capture stopped')
    STAGE=6
    validate_outcome(spool)
    if time.time()>=start+20:raise CalibrationStop('final deadline')
    return 0

def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('source',type=Path);parser.add_argument('spool',type=Path);args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    try:
        status=execute(manifest,args.source,args.spool,os.environ)
    except Exception as error:
        # Numeric enums and equality checks only; never serialize exception text.
        known={'allocation mismatch':11,'runtime binding incomplete':12,'runtime identity':13,'source identity':14,'calibration capture stopped':15,'useful file trace unverified':16,'shell deadline':17,'unexpected source entry':18}
        code=known.get(str(error),1 if isinstance(error,CalibrationStop) else 2 if isinstance(error,OSError) else 3)
        expected={'SLURM_JOB_PARTITION':'acpu','SLURM_CPUS_PER_TASK':'1','SLURM_MEM_PER_NODE':'256','SLURM_JOB_ACCOUNT':'ucb736_asc1'}
        data=(json.dumps({'stage':STAGE,'code':code,'allocation_checks':[os.environ.get(k)==v for k,v in expected.items()]},sort_keys=True)+'\n').encode()
        if len(data)>256:raise RuntimeError('fixed diagnostic cap')
        with (args.spool/'calibration_failure.json').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        status=1
    raise SystemExit(status)
if __name__=='__main__':main()
