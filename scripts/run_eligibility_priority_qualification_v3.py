"""One bounded sequential precision qualification; preserve partials on failure."""
import argparse,collections,faulthandler,gc,hashlib,json,os,re,threading,time
from pathlib import Path
from prompt_token_ids import normalize_prompt_ids
from indexed_startup import indexed_import
from indexed_source_inspection import inspect_with_index
from shutdown_trace import install
exit_observer=install()

def event(name,**kw):print(json.dumps({'event':name,'at_unix':time.time(),**kw}),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--token-receipt',type=Path,required=True);p.add_argument('--cache-receipt',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 root=a.manifest.parent;freeze=json.loads((root/'execution_freeze.json').read_text());gate=json.loads((root/'hash_gate.json').read_text());assert gate['complete'] and gate['freeze_sha256']==hashlib.sha256((root/'execution_freeze.json').read_bytes()).hexdigest()
 for name,want in freeze['remote_hashes'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want,name
 a.output_dir.mkdir(parents=True,exist_ok=False)
 wall_start=float(os.environ['STUDY_WALL_START']);wall_deadline=wall_start+285
 rows=[json.loads(x) for x in a.manifest.read_text().splitlines()];assert len(rows)==16
 tokens=json.loads(a.token_receipt.read_text());assert tokens['normalizer_sha256']==hashlib.sha256(Path(__file__).with_name('prompt_token_ids.py').read_bytes()).hexdigest();assert tokens['cache_receipt_sha256']==hashlib.sha256(a.cache_receipt.read_bytes()).hexdigest();assert tokens['manifest_sha256']==hashlib.sha256(a.manifest.read_bytes()).hexdigest()
 cache=json.loads(a.cache_receipt.read_text());assert cache['valid'];snapshot=cache['snapshot']
 faulthandler.enable()
 def arm_deadline(seconds):
  remaining=wall_deadline-time.time()
  assert remaining>0,'complete-wall budget exhausted'
  faulthandler.dump_traceback_later(min(seconds,remaining),exit=True)
 arm_deadline(min(90,wall_start+90-time.time()));event('imports_start')
 def import_dependencies():
  import torch
  from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
  return torch,AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 (dependencies,captures),inspection=inspect_with_index(lambda:indexed_import(import_dependencies))
 torch,AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig=dependencies
 faulthandler.cancel_dump_traceback_later()
 budget=12*2**30;done=threading.Event()
 def stop(reason,**details):
  failure={'reason':reason,'at_unix':time.time(),**details};(a.output_dir/'budget_stop.json').write_text(json.dumps(failure,indent=2)+'\n');event('budget_stop',**failure);os._exit(2)
 def monitor():
  while not done.wait(.1):
   if time.time()>wall_deadline:stop('285-second complete-wall budget')
   allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
   if max(allocated,reserved)>budget:stop('12GiB planning budget',allocated=allocated,reserved=reserved)
 threading.Thread(target=monitor,daemon=True).start()
 arm_deadline(90);event('tokenizer_load_start',snapshot=snapshot)
 tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=True)
 if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
 tokenizer.padding_side='left';faulthandler.cancel_dump_traceback_later()
 template_hash=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest();assert template_hash==tokens['template_sha256'];assert snapshot==tokens['snapshot'];prompt_tokens={};arm_configs={};records=[]
 with (a.output_dir/'base.jsonl').open('w') as f:
  for precision in ('BF16',):
   armrows=rows;assert len(armrows)==16
   torch.cuda.reset_peak_memory_stats();arm_deadline(min(90,wall_start+90-time.time()));event('model_load_start',precision=precision,snapshot=snapshot)
   kwargs={'pretrained_model_name_or_path':snapshot,'local_files_only':True,'trust_remote_code':True,'torch_dtype':torch.bfloat16,'device_map':{'':0}}
   if precision=='NF4':kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
   model=AutoModelForCausalLM.from_pretrained(**kwargs);model.config.use_cache=False;model.eval();faulthandler.cancel_dump_traceback_later()
   assert all(p.device.type=='cuda' for p in model.parameters())
   loaded4=bool(getattr(model,'is_loaded_in_4bit',False));assert loaded4==(precision=='NF4')
   dtypes=dict(collections.Counter(str(p.dtype) for p in model.parameters()))
   compute=sorted({str(m.compute_dtype) for m in model.modules() if hasattr(m,'compute_dtype')})
   if precision=='NF4':assert compute==['torch.bfloat16']
   else:assert set(dtypes)=={'torch.bfloat16'}
   eos=model.generation_config.eos_token_id;eos_ids=eos if isinstance(eos,list) else [eos] if eos is not None else []
   config={'attention_implementation':model.config._attn_implementation,'parameter_dtype_counts':dtypes,'quantized':loaded4,'quantization_compute_dtypes':compute,'eos_ids':eos_ids,'tokenizer_eos_id':tokenizer.eos_token_id,'chat_template_sha256':template_hash,'use_cache':False}
   if arm_configs:
    prior=arm_configs['NF4']
    for key in ('attention_implementation','eos_ids','tokenizer_eos_id','chat_template_sha256','use_cache'):assert config[key]==prior[key],key
   assert config['attention_implementation']=='sdpa' and template_hash=='cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f'
   arm_configs[precision]=config;event('model_loaded',precision=precision,config=config,gpu_memory_allocated_bytes=torch.cuda.memory_allocated(),gpu_memory_reserved_bytes=torch.cuda.memory_reserved())
   arm_deadline(10)
   for row in armrows:
    messages=row['messages'];encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt')
    ids_in=normalize_prompt_ids(encoded);assert ids_in==normalize_prompt_ids(tokens['prompt_ids'][row['case_id']])
    encoded=torch.tensor([ids_in],dtype=torch.long)
    rendered=tokenizer.decode(ids_in,skip_special_tokens=False);encoded=encoded.to('cuda:0');prompt_length=len(ids_in)
    gpu_start=torch.cuda.Event(enable_timing=True);gpu_end=torch.cuda.Event(enable_timing=True);gpu_start.record()
    with torch.inference_mode():generated=model.generate(encoded,max_new_tokens=24,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    gpu_end.record();torch.cuda.synchronize();gpu_elapsed_ms=gpu_start.elapsed_time(gpu_end)
    ids=generated[0,prompt_length:].tolist();completion=tokenizer.decode(ids,skip_special_tokens=True);match=re.fullmatch(r'\s*([ABCD])\s*',completion);parsed=None if match is None else match.group(1)
    r=dict(row,gpu_elapsed_ms=gpu_elapsed_ms,rendered_prompt=rendered,prompt_ids=ids_in,prompt_length=prompt_length,generated_ids=ids,generated_length=len(ids),eos_ids=eos_ids,last_token=ids[-1] if ids else None,eos_observed=bool(ids and ids[-1] in eos_ids),length_cap_reached=len(ids)==24,completion=completion,parsed_response=parsed,parse_failure=match is None,correct=parsed==row['gold_action'])
    f.write(json.dumps(r)+'\n');f.flush();records.append(r);event('generation_progress',precision=precision,records=len(records),completion=completion,gpu_memory_allocated_bytes=torch.cuda.memory_allocated(),gpu_memory_reserved_bytes=torch.cuda.memory_reserved(),eos_observed=r['eos_observed'],gpu_elapsed_ms=gpu_elapsed_ms,generated_tokens=len(ids))
   faulthandler.cancel_dump_traceback_later();peak_alloc=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved();assert max(peak_alloc,peak_reserved)<=budget
   event('arm_complete',precision=precision,peak_allocated_bytes=peak_alloc,peak_reserved_bytes=peak_reserved)
   del generated,encoded,model;gc.collect();torch.cuda.empty_cache();event('cleanup_complete',precision=precision,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
   assert torch.cuda.memory_allocated()<2**28,'model allocation remains after cleanup'
 assert len(records)==16;done.set()
 receipt={'token_receipt_sha256':hashlib.sha256(a.token_receipt.read_bytes()).hexdigest(),'normalizer_sha256':hashlib.sha256(Path(__file__).with_name('prompt_token_ids.py').read_bytes()).hexdigest(),'records':16,'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'snapshot':snapshot,'arm_configs':arm_configs,'max_new_tokens':24,'do_sample':False,'wall_elapsed_seconds':time.time()-wall_start,'completed_at_unix':time.time(),'resource_budget_GiB':12,'complete':True}
 (a.output_dir/'base_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');event('study_complete',records=16,wall_elapsed_seconds=receipt['wall_elapsed_seconds']);exit_observer()
if __name__=='__main__':main()
