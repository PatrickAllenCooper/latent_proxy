"""Local validation of frozen cases/scoring and runner budget declarations."""
import ast,hashlib,json,runpy
from pathlib import Path
r=Path('outputs/preference_program/manifests/reader32_qualification_v60');reg=json.loads((r/'registration.json').read_text());api=runpy.run_path('scripts/prepare_eligibility_priority.py');cases=[json.loads(l) for l in (r/'cases.jsonl').read_text().splitlines()];checks=0
for c in cases:
 eligible=[o for o in c['options'] if o['eligible']];gold=min(eligible,key=lambda o:c['priority'].index(o['attribute']))['label'];assert gold==c['gold']
 for l in 'ABCD':
  s=api['score'](c,l);o=next(o for o in c['options'] if o['label']==l)
  assert s['correct']==(l==gold) and s['eligibility_violation']==(not o['eligible']);checks+=1
 for invalid in ['', 'Answer: '+gold, 'AB',gold.lower()]:assert api['score'](c,invalid)['parse_failure'];checks+=1
 assert api['score'](c,' \n'+gold+'\n')['correct'];checks+=1
s=Path('scripts/reader32_v60.py').read_text();ast.parse(s)
for guard in ['start+90','start+285','time.time()+10','memory_budget(total)','violation(','len(ids)<=512','max_new_tokens=24','local_files_only=True','use_cache=False','low_cpu_mem_usage=True']:assert guard in s,guard
receipt={'complete':True,'model_calls':0,'case_scoring_assertions':checks,'runner_syntax_valid':True,'shell_syntax_checked_separately':True,'GPU_resource_guards_present':True,'live_runtime_not_validated':True,'runner_sha256':hashlib.sha256(s.encode()).hexdigest()};(r/'local_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
