"""Reads only the frozen small synthetic payload; stdlib-only calibration target."""
import json
import sys
import time
from pathlib import Path
from reader32_calibration_contract_v69 import PAYLOAD,PAYLOAD_SHA256,sha,write_metadata

def main():
    source=Path(sys.argv[1]);spool=Path(sys.argv[2]);start=time.time()
    print(json.dumps({'event':'calibration_toy_start','at':start}),flush=True)
    payload=source/'calibration_input.bin'
    if payload.stat().st_size!=len(PAYLOAD):raise RuntimeError('toy input size')
    with payload.open('rb') as f:data=f.read(len(PAYLOAD)+1)
    if data!=PAYLOAD or sha(payload)!=PAYLOAD_SHA256:raise RuntimeError('toy input identity')
    write_metadata(spool/'calibration_toy_receipt.json','toy',{'kind':2,'payload_sha256':PAYLOAD_SHA256,'bytes_read':len(data),'started':start,'completed':time.time(),'calibration_only':True,'qualification_established':False,'model_calls':0,'framework_imports':0,'GPU_allocations':0})
    print(json.dumps({'event':'calibration_toy_complete','at':time.time()}),flush=True)
if __name__=='__main__':main()
