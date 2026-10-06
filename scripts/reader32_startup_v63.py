"""CPU-only exact GPU import admission under immutable-source layout."""
import argparse,ast,hashlib,json,os,sys,threading,time
from pathlib import Path
from prompt_token_ids import normalize_prompt_ids
from indexed_startup import indexed_import
from indexed_source_inspection import inspect_with_index

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def import_dependencies():
 from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
 import torch
 from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
 return Qwen2Tokenizer,torch,Qwen2ForCausalLM

def import_signature(source,name):
 tree=ast.parse(source);f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
 return [ast.dump(n,include_attributes=False) for n in ast.walk(f) if isinstance(n,(ast.Import,ast.ImportFrom))]

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('spool',type=Path);a=p.parse_args();source=a.source;spool=a.spool;start=float(os.environ['STUDY_WALL_START']);done=threading.Event()
 assert source.resolve()!=spool.resolve() and source.resolve() not in spool.resolve().parents,'spool must be outside source'
 assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1'
 assert not (spool/'startup_receipt.json').exists()
 def watchdog():
  if not done.wait(max(0,start+90-time.time())):
   (spool/'startup_stop.json').write_text(json.dumps({'complete':False,'reason':'90_second_startup_limit','at':time.time(),'model_calls':0})+'\n');os._exit(2)
 threading.Thread(target=watchdog,daemon=True).start()
 freeze=json.loads((source/'startup_freeze.json').read_text())
 for n,h in freeze['files'].items():assert sha(source/n)==h,n
 assert import_signature(Path(__file__).read_text(),'import_dependencies')==import_signature((source/'run_reader32_v62.py').read_text(),'deps')
 original=json.loads((source/'cpu_receipt.json').read_text());assert original['complete'] and sha(source/'cpu_receipt.json')==freeze['CPU_receipt_sha256']
 assert original['runtime_receipt_sha256']==sha(os.environ['RUNTIME_RECEIPT'])
 snap=Path(original['snapshot']) if 'snapshot' in original else Path(freeze['snapshot'])
 for n,e in original['files'].items():
  f=snap/n;s=f.stat();assert s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns'] and str(f.resolve())==e['target']
  if not n.endswith('.safetensors'):assert sha(f)==e['sha256']
 print(json.dumps({'event':'exact_GPU_imports_start','at':time.time()}),flush=True)
 (dependencies,captures),inspection=inspect_with_index(lambda:indexed_import(import_dependencies));Tok,torch,Model=dependencies
 print(json.dumps({'event':'exact_GPU_imports_complete','at':time.time()}),flush=True)
 assert Model.__name__=='Qwen2ForCausalLM' and Tok.__name__=='Qwen2Tokenizer'
 assert sys.version==original['python'] and sha(sys.executable)==original['python_sha256']
 t=Tok.from_pretrained(str(snap),local_files_only=True,trust_remote_code=False)
 assert hashlib.sha256(t.chat_template.encode()).hexdigest()==original['template_sha256']
 rows=list(map(json.loads,(source/'prompts.jsonl').read_text().splitlines()));assert len(rows)==16
 for row in rows:
  ids=normalize_prompt_ids(t.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True));rendered=t.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
  assert ids==original['tokens'][row['case_id']]['ids']==normalize_prompt_ids(t.encode(rendered,add_special_tokens=False)) and rendered==original['tokens'][row['case_id']]['rendered']
 for n,h in freeze['files'].items():assert sha(source/n)==h,n
 assert time.time()<start+90,'startup window exhausted'
 receipt={'complete':True,'CPU_only':True,'model_loads':0,'model_calls':0,'GPU_allocations':0,'exact_GPU_dependency_imports_valid':True,'startup_directory_guard_valid':True,'frozen16_tokens_identical':True,'source_spool_separated':True,'source_hashes':freeze['files'],'captures':captures,'inspection':inspection,'torch_version':torch.__version__,'python':sys.version,'start':start,'completed':time.time(),'startup_elapsed_seconds':time.time()-start,'GPU_model_load_remaining_startup_seconds':start+90-time.time(),'GPU_model_load_feasibility':'Unmeasured; CPU import success does not establish32B CUDA load/response feasibility','prior_CPU_receipt_sha256':sha(source/'cpu_receipt.json')}
 (spool/'startup_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'event':'CPU_startup_admission_complete','elapsed':receipt['startup_elapsed_seconds']}),flush=True);done.set()
if __name__=='__main__':
 try:main()
 except Exception as exc:
  if len(sys.argv)==3:(Path(sys.argv[2])/'startup_stop.json').write_text(json.dumps({'complete':False,'reason':type(exc).__name__+': '+str(exc),'at':time.time(),'model_calls':0})+'\n')
  raise
