"""Bounded CPU token receipt with indexed import and observed normal cleanup."""
import argparse,faulthandler,hashlib,importlib.metadata as md,json,os,time,sys
from pathlib import Path
from indexed_startup import verify_metadata,indexed_import
from indexed_source_inspection import inspect_with_index
from prompt_token_ids import normalize_prompt_ids
from shutdown_trace import install
exit_observer=install()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--cache-receipt',type=Path,required=True);p.add_argument('--environment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def mark(stage,**kw):print(json.dumps({'stage':stage,'at_unix':time.time(),**kw}),flush=True)
assert not a.output.exists();started=time.time();mark('CPU_start',pid=os.getpid())
expected=json.loads(a.environment.read_text());verify_metadata(expected)
faulthandler.enable();faulthandler.dump_traceback_later(15,repeat=True)
original=md.packages_distributions;original_exists=Path.exists
import genericpath
original_os=os.path.exists;original_generic=genericpath.exists
source_paths=[Path(md.__file__),Path('/projects/paco0228/software/anaconda/envs/latent-proxy-env/lib/python3.12/inspect.py'),Path('/projects/paco0228/software/anaconda/envs/latent-proxy-env/lib/python3.12/site-packages/torch/_library/utils.py')];source_hashes={str(s):sha(s) for s in source_paths}
def import_cls():
 from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
 return Qwen2Tokenizer
mark('transformers_import_start');(AutoTokenizer,captures),inspection=inspect_with_index(lambda:indexed_import(import_cls));mark('transformers_import_complete')
assert md.packages_distributions is original and Path.exists is original_exists and os.path.exists is original_os and genericpath.exists is original_generic
cache=json.loads(a.cache_receipt.read_text());assert cache['valid'] and cache['revision']=='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
tokenizer_hashes={}
for f in cache['files']:
 if f['file']!='model.safetensors':
  path=Path(cache['snapshot'])/f['file'];assert sha(path)==f['sha256'];tokenizer_hashes[str(path)]=f['sha256']
cfg=json.loads((Path(cache['snapshot'])/'tokenizer_config.json').read_text());assert cfg['tokenizer_class']=='Qwen2Tokenizer' and not cfg.get('auto_map')
assert 'transformers.models.auto.auto_factory' not in sys.modules
mark('tokenizer_load_start');t=AutoTokenizer.from_pretrained(cache['snapshot'],local_files_only=True,trust_remote_code=True);mark('tokenizer_load_complete')
assert type(t).__name__=='Qwen2Tokenizer'
reference=json.loads(Path(__file__).with_name('reference_token_receipt.json').read_text());assert reference['complete'] and reference['snapshot']==cache['snapshot']
reference_rows=[json.loads(l) for l in Path(__file__).with_name('reference_manifest.jsonl').read_text().splitlines()];assert len(reference_rows)==16
for row in reference_rows:
 assert normalize_prompt_ids(t.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True))==reference['prompt_ids'][row['case_id']]
mark('reference16_ID_equality_valid')
rows=[json.loads(r) for r in a.manifest.read_text().splitlines()];assert len(rows)==16 and len({r['case_id'] for r in rows})==16
ids={};equality={};vocabulary_size=len(t)
for r in rows:
 mark('chat_template_start',case_id=r['case_id'])
 x=normalize_prompt_ids(t.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True))
 y=normalize_prompt_ids(t.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True,return_dict=True))
 rendered=t.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True)
 z=normalize_prompt_ids(t.encode(rendered,add_special_tokens=False))
 assert x==y==z and all(i<vocabulary_size for i in x)
 ids[r['case_id']]=x;equality[r['case_id']]=True
 with a.output.with_suffix('.partial.jsonl').open('a') as journal:
  journal.write(json.dumps({'case_id':r['case_id'],'prompt_ids':x,'three_path_equality':True})+'\n');journal.flush()
 mark('chat_template_complete',case_id=r['case_id'],tokens=len(x))
assert len(t)==vocabulary_size
assert json.loads(json.dumps(ids))==ids
verify_metadata(expected);assert all(sha(p)==h for p,h in {**source_hashes,**tokenizer_hashes}.items())
assert hashlib.sha256(t.chat_template.encode()).hexdigest()=='cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f'
receipt={'complete':True,'tokenizer_class':type(t).__name__,'reference16_ID_equality_valid':True,'reference_token_receipt_sha256':sha(Path(__file__).with_name('reference_token_receipt.json')),'auto_factory_imported':False,'runtime_receipt_sha256':sha(os.environ['RUNTIME_RECEIPT']),'normalizer_sha256':sha(Path(__file__).with_name('prompt_token_ids.py')),'preparation_script_sha256':sha(__file__),'cache_receipt_sha256':sha(a.cache_receipt),'CPU_only':True,'model_calls':0,'started_at_unix':started,'completed_at_unix':time.time(),'manifest_sha256':sha(a.manifest),'snapshot':cache['snapshot'],'template_sha256':hashlib.sha256(t.chat_template.encode()).hexdigest(),'prompt_ids':ids,'max_prompt_tokens':max(map(len,ids.values())),'three_path_equality':equality,'serialization_equality':True,'metadata_before_after_valid':True,'restoration_valid':True,'source_hashes_before_after':source_hashes,'tokenizer_files_before_after':tokenizer_hashes,'indexed_captures':captures,'inspection_receipt':inspection,'qualification_ready':False,'full_package_content_identity':False}
a.output.write_text(json.dumps(receipt,indent=2)+'\n');assert json.loads(a.output.read_text())==receipt;mark('CPU_complete');exit_observer()
