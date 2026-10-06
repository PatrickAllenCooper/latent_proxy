"""Existing-output-only audit: parse prose independently; no Transformer/model imports."""
import hashlib,itertools,json,re,time
from pathlib import Path
import jinja2,tokenizers
from tokenizers import Tokenizer
OUT=Path('outputs/preference_program/results/semantic_reader_audit_v58')
TPL=Path('outputs/preference_program/results/dialogue_phase2_checkpoint/chat_template.jinja')
TOK=Path('outputs/preference_program/results/dialogue_phase2_checkpoint/tokenizer.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def parse(prompt):
 priority=re.search(r'Priority list: ([^.]+)\.',prompt).group(1).split(', then ')
 options=[dict(label=m[0],attribute=m[1],eligible=m[2]=='yes',rank=int(m[3]) if m[3] else None) for m in re.findall(r'^([ABCD]): attribute=(\w+); eligible=(yes|no)\.(?: priority_rank=(\d+)\.)?$',prompt,re.M)]
 assert len(priority)==len(set(priority))==4 and len(options)==4 and len({o['label'] for o in options})==4 and {o['attribute'] for o in options}==set(priority) and sum(o['eligible'] for o in options)==3
 assert all(o['rank'] is None or o['rank']==priority.index(o['attribute'])+1 for o in options)
 gold=min((o for o in options if o['eligible']),key=lambda o:priority.index(o['attribute']))['label']
 return priority,options,gold

def main():
 started=time.time();OUT.mkdir(parents=True,exist_ok=False)
 assert sha(TPL)=='cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f'
 template=jinja2.Environment().from_string(TPL.read_text());tokenizer=Tokenizer.from_file(str(TOK));allrows=[];sources={str(p):sha(p) for p in [TPL,TOK]}
 manifests={'v55':Path('outputs/preference_program/manifests/eligibility_priority_v45/execution_manifest.jsonl'),'v56':Path('outputs/preference_program/manifests/preference_factorial_v56/execution_manifest.jsonl')}
 bases={c['semantic_case_id']:c for c in map(json.loads,Path('outputs/preference_program/manifests/preference_factorial_v56/semantic_cases.jsonl').read_text().splitlines())}
 for study,root,tokenpath in [('v55',Path('outputs/preference_program/results/eligibility_qualification_v55'),Path('outputs/preference_program/results/token_bound_v54/token_receipt.json')),('v56',Path('outputs/preference_program/results/preference_factorial_direct_v57'),Path('outputs/preference_program/results/preference_factorial_direct_v57/token_receipt.json'))]:
  raw=root/'smoke/base.jsonl';sources.update({str(p):sha(p) for p in [raw,manifests[study],tokenpath]});design={r['case_id']:r for r in map(json.loads,manifests[study].read_text().splitlines())};cpu=json.loads(tokenpath.read_text());records=list(map(json.loads,raw.read_text().splitlines()));assert len(records)==len(design)==16
  for r in records:
   d=design[r['case_id']];assert r['prompt']==d['prompt'] and r['messages']==[dict(role='user',content=r['prompt'])];assert r['prompt_sha256']==hashlib.sha256(r['prompt'].encode()).hexdigest();priority,options,gold=parse(r['prompt']);assert gold==r['gold_action']==d['gold_action'];assert template.render(messages=r['messages'],add_generation_prompt=True)==r['rendered_prompt'];assert tokenizer.encode(r['rendered_prompt'],add_special_tokens=False).ids==r['prompt_ids']==cpu['prompt_ids'][r['case_id']];assert tokenizer.decode(r['prompt_ids'],skip_special_tokens=False)==r['rendered_prompt'];assert tokenizer.decode(r['generated_ids'],skip_special_tokens=True)==r['completion'];assert r['prompt_length']==len(r['prompt_ids']) and r['generated_length']==len(r['generated_ids']);assert r['last_token']==r['generated_ids'][-1] in r['eos_ids'] and r['eos_observed'] and not r['length_cap_reached'];m=re.fullmatch(r'\s*([ABCD])\s*',r['completion']);assert m;answer=m[1];chosen=next(o for o in options if o['label']==answer);assert r['correct']==(answer==gold) and r['parsed_response']==answer and not r['parse_failure']
   if study=='v56':
    base=bases[r['semantic_case_id']];expected=[dict(o,label='ABCD'[('ABCD'.index(o['label'])+r['label_rotation'])%4]) for o in base['options']]
    if r['row_order']=='reverse':expected.reverse()
    assert [{k:o[k] for k in ['label','attribute','eligible']} for o in options]==expected==r['options'];assert priority==base['priority'];assert all((o['rank'] is not None)==(r['format']=='explicit_rank') for o in options)
   allrows.append(dict(study=study,case_id=r['case_id'],gold=gold,response=answer,correct=answer==gold,eligibility_violation=not chosen['eligible'],priority_error=chosen['eligible'] and answer!=gold,rendering_verified=True,tokens_reencoded=True,strict_scoring_verified=True,EOS_verified=True))
 # Pure deterministic constrained policy control across every finite task assignment.
 enumerated=0
 for attrs,priority,excluded in itertools.product(itertools.permutations(['quiet','spacious','portable','durable']),itertools.permutations(['quiet','spacious','portable','durable']),range(4)):
  values=[-float('inf') if i==excluded else -priority.index(a) for i,a in enumerate(attrs)];chosen=max(range(4),key=values.__getitem__);walk=next(i for a in priority for i,x in enumerate(attrs) if x==a and i!=excluded);assert chosen==walk and chosen!=excluded and values.count(values[chosen])==1;enumerated+=1
 report=dict(existing_output_records_verified=32,all_semantic_gold_unambiguous=True,label_row_rank_mapping_valid=True,full_prompt_rendering_verified=True,full_prompt_ID_reencoding_verified=True,strict_score_and_EOS_verified=True,local_tokenizer_SHA256=sha(TOK),local_tokenizer_not_byte_identical_to_pinned_JSON=True,token_reencoding_scope='Independent consistency on these32 prompts/completions using preservedcheckpointtokenizer; original pinnedmodel/runtime integrity rests on productionhashreceipts',local_versions=dict(tokenizers=tokenizers.__version__,jinja2=jinja2.__version__),studies={s:dict(n=16,correct=sum(r['correct'] for r in allrows if r['study']==s),eligibility_violations=sum(r['eligibility_violation'] for r in allrows if r['study']==s),priority_errors=sum(r['priority_error'] for r in allrows if r['study']==s)) for s in ['v55','v56']},deterministic_policy_control_cases=enumerated,deterministic_policy_control_errors=0,policy_control_does_not_test_LLM_tool_use=True,model_calls=0,GPU_jobs=0,started=started,elapsed_seconds=time.time()-started,source_hashes=sources,rows=allrows)
 (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['rows','source_hashes']},indent=2))
if __name__=='__main__':main()
