"""CPU diagnostic only: retain raw shapes/IDs before enforcing the same gate."""
import hashlib,json
from pathlib import Path
from collections.abc import Mapping
from prompt_token_ids import normalize_prompt_ids

def describe(value):
 if isinstance(value,Mapping):return {'type':type(value).__name__,'keys':list(value),'input_ids':describe(value.get('input_ids'))}
 if hasattr(value,'tolist'):return {'type':type(value).__name__,'shape':list(value.shape),'value':value.tolist()}
 return {'type':type(value).__name__,'value':value}

def check_gate(case_id,chat_output,render_encode_output):
 record={'case_id':case_id,'raw_chat':describe(chat_output),'raw_render_encode':describe(render_encode_output),'raw_equality':chat_output==render_encode_output}
 try:
  left=normalize_prompt_ids(chat_output);right=normalize_prompt_ids(render_encode_output)
  record.update(chat_ids=left,render_ids=right,chat_length=len(left),render_length=len(right),normalized_equality=left==right,first_difference=next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),None),gate_valid=left==right)
 except (ValueError,TypeError) as exc:record.update(gate_valid=False,error=str(exc))
 return record

def diagnose(tokenizer,rows,out):
 """Call with pinned CPU tokenizer only; no model creation or API calls."""
 out=Path(out);out.mkdir(exist_ok=False);records=[]
 for row in rows:
  rendered=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True)
  record=check_gate(row['case_id'],tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True),tokenizer.encode(rendered,add_special_tokens=False))
  record.update(rendered=rendered,render_sha256=hashlib.sha256(rendered.encode()).hexdigest())
  records.append(record)
  with (out/'tokens.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush()
  if not record['gate_valid']:return {'complete':False,'model_calls':0,'records':len(records),'reason':'token_gate_failed_operands_preserved'}
 return {'complete':len(records)==16,'model_calls':0,'records':len(records),'normalized_ID_gate_valid':all(r['gate_valid'] for r in records)}
