"""CPU-only registration. No model/API imports or execution."""
import hashlib,itertools,json,random,re
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('outputs/preference_program/manifests/eligibility_priority_v45')
SEED=450051
ATTR=('quiet','spacious','portable','durable');LABEL='ABCD'

def oracle(case):
    eligible={o['attribute']:o['label'] for o in case['options'] if o['eligible']}
    return next(eligible[x] for x in case['priority'] if x in eligible)

def score(case,response,advice=None):
    m=re.fullmatch(r'\s*([ABCD])\s*',response);label=m.group(1) if m else None
    selected=next((o for o in case['options'] if o['label']==label),None)
    return {'parsed':label,'parse_failure':m is None,'correct':label==oracle(case),'eligibility_violation':selected is not None and not selected['eligible'],'eligible_priority_error':selected is not None and selected['eligible'] and label!=oracle(case),'advice_copied':None if advice is None else label==advice,'corrected_wrong_advice':None if advice is None or advice==oracle(case) else label==oracle(case)}

def prompt(c,advice=None,verify=False):
    s='User rule: choose an eligible option. Among eligible options, choose the attribute that appears earliest in my priority list. Priority list: '+', then '.join(c['priority'])+'.\nCatalog:\n'
    s+='\n'.join(f"{o['label']}: attribute={o['attribute']}; eligible={'yes' if o['eligible'] else 'no'}." for o in c['options'])
    if advice is not None:s+='\nAn adviser recommends '+advice+'. The adviser may be mistaken.'
    if verify:s+='\nCheck the recommendation against the user rule before choosing.'
    return s+'\nReply with exactly one capital letter: A, B, C, or D.'

def main():
    started=datetime.now(timezone.utc).isoformat();rng=random.Random(SEED);pool=[]
    for attrs in itertools.permutations(ATTR):
      for priority in itertools.permutations(ATTR):
       for excluded in range(4):
        c={'priority':list(priority),'options':[{'label':l,'attribute':a,'eligible':i!=excluded} for i,(l,a) in enumerate(zip(LABEL,attrs))]}
        c['gold']=oracle(c);c['top_priority_ineligible']=priority[0]==attrs[excluded];pool.append(c)
    rng.shuffle(pool);used=set();splits={}
    for split,n in [('development',4),('qualification',16),('advice',8)]:
      cases=[]
      for i in range(n):
       gold=LABEL[i%4];topbad=bool((i//4)%2)
       c=next(c for c in pool if c['gold']==gold and c['top_priority_ineligible']==topbad and json.dumps(c,sort_keys=True) not in used)
       used.add(json.dumps(c,sort_keys=True));c=dict(c,case_id=f'{split}-{i:02d}')
       c['wrong_eligible']=next(o['label'] for o in c['options'] if o['eligible'] and o['label']!=gold)
       c['wrong_ineligible']=next(o['label'] for o in c['options'] if not o['eligible'])
       cases.append(c)
      splits[split]=cases
    for name,cases in splits.items():
      (ROOT/f'{name}_cases.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cases))
    arms=[('none',None,False),('correct_neutral','gold',False),('correct_verify','gold',True),('wrong_eligible_neutral','wrong_eligible',False),('wrong_eligible_verify','wrong_eligible',True),('wrong_ineligible_neutral','wrong_ineligible',False),('wrong_ineligible_verify','wrong_ineligible',True)]
    prompts={name:[{'case_id':c['case_id'],'arm':arm,'advice':None if field is None else c[field],'prompt':prompt(c,None if field is None else c[field],verify)} for c in cases for arm,field,verify in (arms if name=='advice' else arms[:1])] for name,cases in splits.items()}
    for name,rows in prompts.items():
      for r in rows:r['prompt_sha256']=hashlib.sha256(r['prompt'].encode()).hexdigest();r['messages']=[{'role':'user','content':r['prompt']}]
      (ROOT/f'{name}_prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    fixture_count=0
    for cases in splits.values():
      for c in cases:
       for label in LABEL:
        x=score(c,label,c['wrong_eligible']);assert x['correct']==(label==c['gold']);assert x['eligibility_violation']==(label==c['wrong_ineligible']);fixture_count+=1
       for raw in ['', 'Answer: '+c['gold'],'AB','a']:
        assert score(c,raw)['parse_failure'];fixture_count+=1
       assert score(c,' \n'+c['gold']+'\n')['correct'];fixture_count+=1
       assert score(c,c['gold'],c['wrong_eligible'])['corrected_wrong_advice'];assert not score(c,c['wrong_eligible'],c['wrong_eligible'])['correct'];fixture_count+=2
    files=list(ROOT.glob('*cases.jsonl'))+list(ROOT.glob('*prompts.jsonl'))
    validation={'CPU_started_utc':started,'CPU_completed_utc':datetime.now(timezone.utc).isoformat(),'model_calls':0,'GPU_allocations':0,'API_calls':0,'fixture_assertions':fixture_count,'split_counts':{k:len(v) for k,v in splits.items()},'prompt_counts':{k:len(v) for k,v in prompts.items()},'split_structures_disjoint':len(used)==28,'gold_balanced_each_split':True,'oracle_unique':True,'hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation))
if __name__=='__main__':main()
