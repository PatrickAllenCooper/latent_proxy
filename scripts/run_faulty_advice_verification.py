"""Registered visible-score verification smoke; no preference discovery claims."""
import argparse,faulthandler,hashlib,json,os,time
from pathlib import Path
import numpy as np

def event(name,**kwargs):print(json.dumps({'event':name,'at_unix':time.time(),**kwargs}),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache-receipt',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--limit',type=int,default=20);a=p.parse_args()
 rows=[json.loads(x) for x in a.manifest.read_text().splitlines()][:a.limit]
 assert len(rows)==a.limit==20 and {r['user_id'] for r in rows}=={0}
 cache=json.loads(a.cache_receipt.read_text());assert cache['valid']
 faulthandler.enable();event('imports_start')
 import torch
 from src.training.model_utils import load_model_with_optional_checkpoint
 from scripts.run_finite_menu_llm_pilot import parse_action
 event('model_load_start',snapshot=cache['snapshot']);faulthandler.dump_traceback_later(90,exit=True)
 model,tokenizer=load_model_with_optional_checkpoint(cache['snapshot'])
 faulthandler.cancel_dump_traceback_later();event('model_loaded',gpu_device_count=torch.cuda.device_count(),gpu_memory_allocated_bytes=torch.cuda.memory_allocated())
 a.output_dir.mkdir(parents=True,exist_ok=True)
 with (a.output_dir/'base.jsonl').open('w') as f:
  for i,row in enumerate(rows):
   messages=[{'role':'user','content':row['prompt']}]
   encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt')
   if hasattr(encoded,'input_ids'):encoded=encoded['input_ids']
   encoded=encoded.to(model.device)
   with torch.inference_mode():generated=model.generate(encoded,max_new_tokens=24,do_sample=False,pad_token_id=tokenizer.eos_token_id)
   completion=tokenizer.decode(generated[0,encoded.shape[-1]:],skip_special_tokens=True);action=parse_action(completion)
   v=np.array(row['expected_utilities']);span=float(v.max()-v.min())
   regret=None if action is None else float((v.max()-v[action])/span)
   record=dict(row,messages=messages,rendered_prompt=tokenizer.decode(encoded[0],skip_special_tokens=False),completion=completion,final_action=None if action is None else chr(65+action),parse_failure=action is None,normalized_regret=regret)
   f.write(json.dumps(record)+'\n');f.flush();event('generation_progress',records=i+1,gpu_memory_allocated_bytes=torch.cuda.memory_allocated(),arm=row['arm'],completion=completion)
 receipt={'records':len(rows),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'snapshot':cache['snapshot'],'completed_at_unix':time.time(),'max_new_tokens':24,'do_sample':False}
 (a.output_dir/'base_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
