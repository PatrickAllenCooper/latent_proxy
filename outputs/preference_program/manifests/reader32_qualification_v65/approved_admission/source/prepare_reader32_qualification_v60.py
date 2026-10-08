"""Freeze a local CPU-only proposal; no model or remote calls."""
import hashlib, itertools, json, random, runpy
from pathlib import Path
R=Path('outputs/preference_program/manifests/reader32_qualification_v60')
S=Path('scripts/prepare_eligibility_priority.py')
api=runpy.run_path(str(S)); rng=random.Random(2026100602)
def signature(c):
    return (tuple(c['priority']),next(o['attribute'] for o in c['options'] if not o['eligible']))
paths=list(Path('outputs/preference_program/manifests/eligibility_priority_v45').glob('*_cases.jsonl'))+[Path('outputs/preference_program/manifests/preference_factorial_v56/semantic_cases.jsonl')]
old={signature(json.loads(line)) for p in paths for line in p.read_text().splitlines()}
pool=[]
for pr in itertools.permutations(api['ATTR']):
 for attrs in itertools.permutations(api['ATTR']):
  for excluded in range(4):
   c={'priority':list(pr),'options':[{'label':l,'attribute':a,'eligible':i!=excluded} for i,(l,a) in enumerate(zip('ABCD',attrs))]}
   if signature(c) in old:continue
   c.update(gold=api['oracle'](c),top_priority_ineligible=pr[0]==attrs[excluded]);pool.append(c)
rng.shuffle(pool);chosen=[];used=set(old)
for label in 'ABCD':
 for topbad in (False,True):
  for _ in range(2):
   c=next(c for c in pool if c['gold']==label and c['top_priority_ineligible']==topbad and signature(c) not in used)
   used.add(signature(c));chosen.append(dict(c,case_id=f'reader32-fresh-{len(chosen):02d}'))
rng.shuffle(chosen); R.mkdir(parents=True,exist_ok=True)
(R/'cases.jsonl').write_text(''.join(json.dumps(c,sort_keys=True)+'\n' for c in chosen))
rows=[{'case_id':c['case_id'],'messages':[{'role':'user','content':api['prompt'](c)}],'prompt':api['prompt'](c)} for c in chosen]
(R/'prompts.jsonl').write_text(''.join(json.dumps(c,sort_keys=True)+'\n' for c in rows))
assert len({signature(c) for c in chosen})==16
assert all(api['score'](c,c['gold'])['correct'] and api['score'](c,'Answer: '+c['gold'])['parse_failure'] for c in chosen)
reg={'status':'proposal_only_no_execution_authorized','preparation_approval':'2026-10-06T16:52:00Z Sentinel_6dff92097b348191ac29747811496cd5','model':'Qwen/Qwen2.5-32B-Instruct','revision':'5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd','seed':2026100602,'cases':16,'independent_semantic_cases':16,'prior_semantics_excluded':len(old),'gold_counts':{l:sum(c['gold']==l for c in chosen) for l in 'ABCD'},'excluded_top_count':sum(c['top_priority_ineligible'] for c in chosen),'gate':'16/16 strict correct; no parse/eligibility/priority errors; all custody/runtime/resource gates valid','scorer_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'generation':{'precision':'BF16','adapters':False,'do_sample':False,'max_new_tokens':24,'max_prompt_tokens':512,'use_cache':False,'attention':'sdpa','responses':16},'advice_hold':True,'automatic_retry':False,'calls':0,'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'cases.jsonl',R/'prompts.jsonl',Path(__file__)]},'exclusion_source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(R/'registration.json').write_text(json.dumps(reg,indent=2)+'\n');print(json.dumps(reg,indent=2))
