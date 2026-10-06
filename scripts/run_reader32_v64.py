"""Bounded CPU integrity or GPU qualification; no download/fallback/retry."""
import argparse,hashlib,json,os,sys,time,threading
from pathlib import Path
from prompt_token_ids import normalize_prompt_ids
from reader32_v60_contracts import memory_budget,violation,admit_next,result,export_incomplete
SNAP=Path('/scratch/alpine/paco0228/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def emit(event,**data):print(json.dumps(dict(event=event,at=time.time(),**data)),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['gpu']);p.add_argument('root',type=Path);p.add_argument('--spool',type=Path,required=True);a=p.parse_args();r=a.root
 assert r.resolve()!=a.spool.resolve() and r.resolve() not in a.spool.resolve().parents,'spool must be outside source'
 assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
 global FAILURE_OUT
 FAILURE_OUT=a.spool/'smoke'
 if a.stage=='gpu':
  start=float(os.environ['STUDY_WALL_START']);out=a.spool/'smoke';out.mkdir(exist_ok=False)
 if a.stage=='gpu':
  startup_done=threading.Event()
  def startup_watch():
   if not startup_done.wait(max(0,start+90-time.time())):
    export_incomplete(out,'90_second_startup_limit',time.time());os._exit(2)
  threading.Thread(target=startup_watch,daemon=True).start()
 reg=json.loads((r/'registration.json').read_text());cases=[json.loads(x) for x in (r/'cases.jsonl').read_text().splitlines()];rows=[json.loads(x) for x in (r/'prompts.jsonl').read_text().splitlines()]
 assert len(cases)==len(rows)==16
 for name,want in reg['hashes'].items():assert sha(r/name)==want,name
 assert sha(r/'prepare_eligibility_priority.py')==reg['scorer_sha256']
 freeze=json.loads((r/'execution_freeze.json').read_text())
 for name,want in freeze['source_hashes'].items():assert sha(r/name)==want,name
 if a.stage=='cpu':
  assert not (r/'cpu_receipt.json').exists();start=time.time();emit('CPU_integrity_start')
  idx=json.loads((SNAP/'model.safetensors.index.json').read_text());names=sorted(set(idx['weight_map'].values()))
  assert len(names)==17
  names+=['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','generation_config.json']
  files={}
  for name in names:
   f=SNAP/name;before=f.stat();digest=sha(f);after=f.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
   files[name]={'sha256':digest,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'target':str(f.resolve())}
   with (r/'integrity_journal.jsonl').open('a') as journal:journal.write(json.dumps({'file':name,**files[name]})+'\n');journal.flush()
   emit('CPU_file_hashed',name=name,bytes=after.st_size)
  assert files['config.json']['sha256']=='9c6772f138ef9e5b3d1c18f2c87e451bbc01f5f1a4eabb36f9bf4f53829b903e'
  assert files['model.safetensors.index.json']['sha256']=='0183543f2e6e40d3d1e863ed0b2a8c9cdaf2ee4045ce724a45cea30e9995a0af'
  emit('CPU_tokenizer_start')
 else:
  receipt=json.loads((r/'cpu_receipt.json').read_text());assert receipt['complete'];assert sha(r/'cpu_receipt.json')==freeze['CPU_receipt_sha256'];assert freeze['GPU_admission'];files=receipt['files'];emit('GPU_process_start')
  for name,entry in files.items():
   f=SNAP/name;s=f.stat();assert s.st_size==entry['bytes'] and s.st_mtime_ns==entry['mtime_ns'] and str(f.resolve())==entry['target']
  assert receipt['freeze_sha256']==freeze['CPU_execution_freeze_sha256'] and receipt['runtime_receipt_sha256']==sha(os.environ['RUNTIME_RECEIPT'])
 from indexed_startup import indexed_import
 from indexed_source_inspection import inspect_with_index
 def deps():
  from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
  if a.stage=='cpu':return Qwen2Tokenizer,None,None
  import torch
  from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
  return Qwen2Tokenizer,torch,Qwen2ForCausalLM
 (dependencies,captures),inspection=inspect_with_index(lambda:indexed_import(deps));Tok,torch,Model=dependencies
 t=Tok.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=False)
 template=hashlib.sha256(t.chat_template.encode()).hexdigest();assert template=='cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f'
 tokens={}
 for row in rows:
  ids=normalize_prompt_ids(t.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True))
  rendered=t.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
  assert ids==normalize_prompt_ids(t.encode(rendered,add_special_tokens=False)) and len(ids)<=512
  tokens[row['case_id']]={'ids':ids,'rendered':rendered}
 if a.stage=='cpu':
  receipt={'complete':True,'files':files,'tokens':tokens,'template_sha256':template,'freeze_sha256':sha(r/'execution_freeze.json'),'runtime_receipt_sha256':sha(os.environ['RUNTIME_RECEIPT']),'python':sys.version,'python_sha256':sha(sys.executable),'started':start,'completed':time.time(),'model_calls':0,'captures':captures,'inspection':inspection}
  (r/'cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');emit('CPU_complete',elapsed=time.time()-start);return
 assert tokens==receipt['tokens'] and sys.version==receipt['python'] and sha(sys.executable)==receipt['python_sha256']
 done=threading.Event();phase={'name':'startup','deadline':start+90};total=torch.cuda.get_device_properties(0).total_memory;budget=memory_budget(total)
 assert torch.cuda.device_count()==1 and total<80*2**30,'unexpected CUDA device visibility/capacity for one71GB MIG'
 device_name=torch.cuda.get_device_name(0)
 def stop(reason):
  export_incomplete(out,reason,time.time(),phase=dict(phase),allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved());emit('STOP_incomplete',reason=reason);os._exit(2)
 def monitor():
  last_memory_event=0
  while not done.wait(.05):
   if time.time()-last_memory_event>=2:
    emit('GPU_memory_progress',phase=phase['name'],allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved());last_memory_event=time.time()
   reason=violation(time.time(),start,phase['deadline'],torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved(),budget)
   if reason:stop(reason)
 threading.Thread(target=monitor,daemon=True).start();emit('model_load_start',total_memory=total,budget=budget,device_name=device_name,device_count=torch.cuda.device_count(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
 assert total>65*2**30,'71GB slice usable memory insufficient for proposed cap'
 model=Model.from_pretrained(str(SNAP),local_files_only=True,trust_remote_code=False,torch_dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='sdpa');model.eval();model.config.use_cache=False
 assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters())
 assert not getattr(model,'is_loaded_in_4bit',False) and model.config._attn_implementation=='sdpa'
 startup_done.set();emit('model_loaded',allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved());durations=[];records=[]
 import runpy
 score=runpy.run_path(str(r/'prepare_eligibility_priority.py'))['score'];byid={c['case_id']:c for c in cases}
 with (out/'responses.jsonl').open('w') as f:
  for row in rows:
   remaining=start+285-time.time()
   if not admit_next(time.time(),start,durations,len(records)):stop('insufficient_remaining_time')
   phase.update(name='generation',deadline=time.time()+10);beg=time.time();emit('generation_start',case_id=row['case_id'])
   inp=torch.tensor([tokens[row['case_id']]['ids']],device='cuda');ev1=torch.cuda.Event(enable_timing=True);ev2=torch.cuda.Event(enable_timing=True);ev1.record()
   with torch.inference_mode():generated=model.generate(inp,max_new_tokens=24,do_sample=False,use_cache=False,pad_token_id=t.eos_token_id)
   ev2.record();torch.cuda.synchronize();ids=generated[0,inp.shape[1]:].tolist();raw=t.decode(ids,skip_special_tokens=True);duration=time.time()-beg
   reason=violation(time.time(),start,phase['deadline'],torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved(),budget)
   if reason:stop(reason)
   durations.append(duration)
   record={**row,'gold':byid[row['case_id']]['gold'],'prompt_ids':tokens[row['case_id']]['ids'],'rendered_prompt':tokens[row['case_id']]['rendered'],'generated_ids':ids,'completion':raw,'score':score(byid[row['case_id']],raw),'elapsed':duration,'cuda_elapsed_ms':ev1.elapsed_time(ev2),'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'length_cap_reached':len(ids)==24}
   records.append(record);f.write(json.dumps(record)+'\n');f.flush();emit('generation_progress',case_id=row['case_id'],records=len(records),tokens=len(ids),cuda_ms=record['cuda_elapsed_ms'],allocated=record['allocated'],reserved=record['reserved']);phase.update(name='between_cases',deadline=start+285)
 phase.update(name='export',deadline=start+285)
 assert len(records)==16
 reason=violation(time.time(),start,phase['deadline'],torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved(),budget)
 if reason:stop(reason)
 final={**result(records,[row['case_id'] for row in rows],True),'cpu_receipt_sha256':sha(r/'cpu_receipt.json'),'freeze_sha256':sha(r/'execution_freeze.json'),'started':start,'completed':time.time(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'actual_memory':total,'budget':budget}
 (out/'receipt.json').write_text(json.dumps(final,indent=2)+'\n');emit('GPU_complete',**final);done.set()
if __name__=='__main__':
 try:main()
 except Exception as exc:
  if 'FAILURE_OUT' in globals():export_incomplete(FAILURE_OUT,type(exc).__name__+': '+str(exc),time.time())
  raise
