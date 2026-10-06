"""One changed, CPU-only cause-specific preparation attempt; no model load."""
import hashlib,json,os,runpy,sys,time
from pathlib import Path
from indexed_startup import indexed_import
from indexed_source_inspection import inspect_with_index
from diagnose_reader32_token_gate import diagnose
r=Path(sys.argv[1]);start=time.time()
freeze=json.loads((r/'execution_freeze.json').read_text())
for name,want in freeze['source_hashes'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==want,name
snap='/scratch/alpine/paco0228/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'
print(json.dumps({'event':'pinned_token_diagnostic_start','at':start}),flush=True)
def cls():
 from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
 return Qwen2Tokenizer
(Tok,captures),inspection=inspect_with_index(lambda:indexed_import(cls));t=Tok.from_pretrained(snap,local_files_only=True,trust_remote_code=False)
assert type(t).__name__=='Qwen2Tokenizer'
assert hashlib.sha256(t.chat_template.encode()).hexdigest()=='cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f'
rows=list(map(json.loads,(r/'prompts.jsonl').read_text().splitlines()));assert len(rows)==16
report=diagnose(t,rows,r/'token_diagnostic');report.update(started=start,completed=time.time(),snapshot=snap,tokenizer_class=type(t).__name__,source_hashes=freeze['source_hashes'],captures=captures,inspection=inspection)
(r/'token_diagnostic_receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
assert report['complete'],'diagnostic token gate failed; stop without integrity continuation'
observed=list(map(json.loads,(r/'token_diagnostic/tokens.jsonl').read_text().splitlines()))
assert observed[0]['normalized_equality'] and not observed[0]['raw_equality'],'initial failure not reproduced as container-only mismatch; no integrity retry'
# A demonstrated first-case container mismatch permits only representation normalization.
# IDs, rendering, lengths, checkpoint, scorer and prompts are unchanged.
sys.argv=[str(r/'reader32_v61.py'),'cpu',str(r)]
runpy.run_path(str(r/'reader32_v61.py'),run_name='__main__')
