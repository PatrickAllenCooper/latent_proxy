"""Local output-only derivation of immutable v67 driver; exact AST guard checks."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/preference_program/manifests/reader32_import_diagnostic_v67_approved/source'
OUT=ROOT/'outputs/preference_program/manifests/reader32_import_trace_v68_receipt_safe'
TAIL=''' from reader32_safe_receipt_v68 import write_receipt
 receipt={'complete':True,'CPU_only':True,'model_loads':0,'model_calls':0,'GPU_allocations':0,'guarded_imports_completed':True,'started':start,'completed':time.time(),'source_hashes_valid':True,'CPU_receipt_sha256':sha(s/'cpu_receipt.json'),'runtime_receipt_sha256':sha(os.environ['RUNTIME_RECEIPT']),'freeze_sha256':sha(s/'diagnostic_freeze.json'),'baseline_driver_sha256':sha(s/'reader32_import_diagnostic_v67_cpu.py'),'adapter_driver_sha256':sha(Path(__file__)),'qualification_established':False}
 write_receipt(o/'CPU_import_receipt.json',receipt)
 emit('CPU_diagnostic_complete',elapsed=time.time()-start)
if __name__=='__main__':main()
'''

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main_nodes(text):
 tree=ast.parse(text)
 f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 return tree,f

def verify_output_only(original,adapted):
 old,of=main_nodes(original);new,nf=main_nodes(adapted)
 marker=next(i for i,n in enumerate(of.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='receipt' for t in n.targets))
 dump=lambda nodes: [ast.dump(n,include_attributes=False) for n in nodes]
 if dump(of.body[:marker])!=dump(nf.body[:marker]):raise ValueError('guard prefix changed')
 if not isinstance(nf.body[marker],ast.ImportFrom) or nf.body[marker].module!='reader32_safe_receipt_v68':raise ValueError('adapter boundary changed')
 old_prefix=old.body[:old.body.index(of)]
 new_prefix=new.body[:new.body.index(nf)]
 if dump(old_prefix)!=dump(new_prefix):raise ValueError('imports or dependencies changed')
 expected=original[:original.index(' receipt=')]+TAIL
 if adapted!=expected:raise ValueError('unexpected adapter derivation')
 return {'guard_prefix_statements':marker,'dependency_AST_identical':True,'baseline_top_level_AST_identical':True,'output_tail_only_changed':True}

def main():
 source=OUT/'source';source.mkdir(parents=True,exist_ok=True)
 for p in BASE.iterdir():
  if p.is_file():(source/p.name).write_bytes(p.read_bytes())
 original=(BASE/'reader32_import_diagnostic_v67_cpu.py').read_text()
 adapted=original[:original.index(' receipt=')]+TAIL
 proof=verify_output_only(original,adapted)
 driver=source/'reader32_import_diagnostic_v68_receipt_safe.py';driver.write_text(adapted)
 (source/'reader32_safe_receipt_v68.py').write_bytes((ROOT/'scripts/reader32_safe_receipt_v68.py').read_bytes())
 proof.update(baseline_driver_sha256=sha(BASE/'reader32_import_diagnostic_v67_cpu.py'),adapter_driver_sha256=sha(driver),safe_writer_sha256=sha(source/'reader32_safe_receipt_v68.py'),baseline_source_mismatches=[p.name for p in BASE.iterdir() if p.is_file() and sha(p)!=sha(source/p.name)],allocation_submitted=False)
 (OUT/'derivation_receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
 if proof['baseline_source_mismatches']:raise ValueError('baseline source copy changed')
if __name__=='__main__':main()
