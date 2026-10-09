"""Approved one CPU-only pinned guarded import; no model loading or inference."""
import argparse,ast,hashlib,json,os,sys,time
from pathlib import Path
from indexed_startup import indexed_import
from indexed_source_inspection import inspect_with_index
from reader32_import_trace import import_trace

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(event,**kw):print(json.dumps({'event':event,'at':time.time(),**kw}),flush=True)
def imports(source,name):
 t=ast.parse(source);f=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
 return [ast.dump(n,include_attributes=False) for n in ast.walk(f) if isinstance(n,(ast.Import,ast.ImportFrom))]
def dependencies():
 emit('dependency_callback_enter');emit('tokenizer_class_import_start')
 from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
 emit('tokenizer_class_import_complete');emit('torch_import_start')
 import torch
 emit('torch_import_complete');emit('model_class_import_start')
 from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
 emit('model_class_import_complete')
 return Qwen2Tokenizer,torch,Qwen2ForCausalLM

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('spool',type=Path);a=p.parse_args();s=a.source;o=a.spool
 assert s.resolve()!=o.resolve() and s.resolve() not in o.resolve().parents
 assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'
 start=float(os.environ['STUDY_WALL_START']);emit('CPU_diagnostic_start',shell_start=start)
 f=json.loads((s/'diagnostic_freeze.json').read_text());assert f['CPU_admission'] is True
 assert f['resources']=={'CPUs':1,'mem_GiB':2,'wall_seconds':120,'check_seconds':90,'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal','GPUs':0,'attempts':1}
 assert os.environ['SLURM_JOB_ID'].isdigit() and os.environ['SLURM_JOB_PARTITION']=='acpu'
 assert os.environ['SLURM_CPUS_PER_TASK']=='1' and os.environ['SLURM_MEM_PER_NODE']=='2048'
 for k in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE'):assert os.environ.get(k,'') in ('','0')
 for n,h in f['files'].items():assert sha(s/n)==h,n
 assert imports(Path(__file__).read_text(),'dependencies')==imports((s/'run_reader32_v65.py').read_text(),'deps')
 old=json.loads((s/'cpu_receipt.json').read_text());assert old['complete'] and sys.version==old['python'] and sha(sys.executable)==old['python_sha256']
 assert sha(os.environ['RUNTIME_RECEIPT'])==old['runtime_receipt_sha256']
 snap=Path(f['snapshot'])
 for n,e in old['files'].items():
  path=snap/n;stat=path.stat();assert stat.st_size==e['bytes'] and stat.st_mtime_ns==e['mtime_ns'] and str(path.resolve())==e['target'],n
  if not n.endswith('.safetensors'):assert sha(path)==e['sha256'],n
 assert time.time()<start+90
 emit('guarded_dependency_import_start')
 with import_trace():
  (deps,captures),inspection=inspect_with_index(lambda:indexed_import(dependencies))
 emit('guarded_dependency_import_complete')
 Tok,torch,Model=deps
 assert Tok.__name__=='Qwen2Tokenizer' and Model.__name__=='Qwen2ForCausalLM'
 for n,h in f['files'].items():assert sha(s/n)==h,n
 assert time.time()<start+90
 from reader32_safe_receipt_v68 import write_receipt
 receipt={'complete':True,'CPU_only':True,'model_loads':0,'model_calls':0,'GPU_allocations':0,'guarded_imports_completed':True,'started':start,'completed':time.time(),'source_hashes_valid':True,'CPU_receipt_sha256':sha(s/'cpu_receipt.json'),'runtime_receipt_sha256':sha(os.environ['RUNTIME_RECEIPT']),'freeze_sha256':sha(s/'diagnostic_freeze.json'),'baseline_driver_sha256':sha(s/'reader32_import_diagnostic_v67_cpu.py'),'adapter_driver_sha256':sha(Path(__file__)),'qualification_established':False}
 write_receipt(o/'CPU_import_receipt.json',receipt)
 emit('CPU_diagnostic_complete',elapsed=time.time()-start)
if __name__=='__main__':main()
