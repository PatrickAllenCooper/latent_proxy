"""Frozen eight-response output-format diagnostic with token-level custody."""
import argparse,faulthandler,hashlib,json,re,time
from pathlib import Path

def event(name,**kw):print(json.dumps({'event':name,'at_unix':time.time(),**kw}),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache-receipt',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 rows=[json.loads(x) for x in a.manifest.read_text().splitlines()];assert len(rows)==8
 cache=json.loads(a.cache_receipt.read_text());assert cache['valid']
 faulthandler.enable();event('imports_start')
 import torch
 from src.training.model_utils import load_model_with_optional_checkpoint
 event('model_load_start',snapshot=cache['snapshot']);faulthandler.dump_traceback_later(90,exit=True)
 model,tokenizer=load_model_with_optional_checkpoint(cache['snapshot']);faulthandler.cancel_dump_traceback_later()
 event('model_loaded',gpu_device_count=torch.cuda.device_count(),gpu_memory_allocated_bytes=torch.cuda.memory_allocated())
 eos=model.generation_config.eos_token_id;eos_ids=eos if isinstance(eos,list) else [eos] if eos is not None else []
 a.output_dir.mkdir(parents=True,exist_ok=True)
 with (a.output_dir/'base.jsonl').open('w') as f:
  for i,row in enumerate(rows):
   messages=[{'role':'user','content':row['prompt']}]
   encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt')
   if hasattr(encoded,'input_ids'):encoded=encoded['input_ids']
   encoded=encoded.to(model.device);prompt_length=int(encoded.shape[-1])
   with torch.inference_mode():generated=model.generate(encoded,max_new_tokens=24,do_sample=False,pad_token_id=tokenizer.eos_token_id)
   ids=generated[0,prompt_length:].tolist();completion=tokenizer.decode(ids,skip_special_tokens=True)
   pattern=r'\s*([ABCD])\s*' if row['response_mode']=='label' else r'\s*([0-9])\s*'
   match=re.fullmatch(pattern,completion);parsed=None if match is None else match.group(1)
   record=dict(row,messages=messages,rendered_prompt=tokenizer.decode(encoded[0],skip_special_tokens=False),completion=completion,parsed_response=parsed,parse_failure=match is None,correct=parsed==row['expected_response'],prompt_length=prompt_length,generated_ids=ids,eos_ids=eos_ids,tokenizer_eos_id=tokenizer.eos_token_id,generated_length=len(ids),last_token=ids[-1] if ids else None,length_cap_reached=len(ids)==24,eos_observed=bool(ids and ids[-1] in eos_ids))
   f.write(json.dumps(record)+'\n');f.flush();event('generation_progress',records=i+1,gpu_memory_allocated_bytes=torch.cuda.memory_allocated(),mode=row['response_mode'],completion=completion,generated_length=len(ids),eos_observed=record['eos_observed'])
 (a.output_dir/'base_receipt.json').write_text(json.dumps({'records':8,'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'snapshot':cache['snapshot'],'max_new_tokens':24,'do_sample':False,'completed_at_unix':time.time()},indent=2)+'\n')
if __name__=='__main__':main()
