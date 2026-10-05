"""Proposed instrumented CPU preparation. Not executed; requires corrected-attempt decision.
Use unchanged verified runtime archive staged in CPU node-local storage first.
Invoke with external timeout --kill-after=2s 90s; no GPU/model imports.
"""
import argparse,faulthandler,hashlib,json,os,time
from pathlib import Path
from prompt_token_ids import normalize_prompt_ids
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def mark(stage,**kw):print(json.dumps({'stage':stage,'at_unix':time.time(),**kw}),flush=True)
assert not a.output.exists();started=time.time();mark('CPU_start',pid=os.getpid())
faulthandler.enable();faulthandler.dump_traceback_later(30,repeat=True)
mark('transformers_import_start')
from transformers import AutoTokenizer
mark('transformers_import_complete')
cache=json.loads(a.cache_receipt.read_text());assert cache['valid'];mark('tokenizer_load_start')
t=AutoTokenizer.from_pretrained(cache['snapshot'],local_files_only=True,trust_remote_code=True)
mark('tokenizer_load_complete')
ids={}
for r in map(json.loads,a.manifest.read_text().splitlines()):
 mark('chat_template_start',case_id=r['case_id']);x=t.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True)
 x=normalize_prompt_ids(x);ids[r['case_id']]=x;mark('chat_template_complete',case_id=r['case_id'],tokens=len(x))
assert len(ids)==16
receipt={'runtime_receipt_sha256':hashlib.sha256(Path(os.environ['RUNTIME_RECEIPT']).read_bytes()).hexdigest(),'normalizer_sha256':hashlib.sha256(Path(__file__).with_name('prompt_token_ids.py').read_bytes()).hexdigest(),'preparation_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cache_receipt_sha256':hashlib.sha256(a.cache_receipt.read_bytes()).hexdigest(),'CPU_only':True,'model_calls':0,'started_at_unix':started,'completed_at_unix':time.time(),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'snapshot':cache['snapshot'],'template_sha256':hashlib.sha256(t.chat_template.encode()).hexdigest(),'prompt_ids':ids,'max_prompt_tokens':max(map(len,ids.values()))}
a.output.write_text(json.dumps(receipt,indent=2)+'\n');faulthandler.cancel_dump_traceback_later();mark('CPU_complete')
