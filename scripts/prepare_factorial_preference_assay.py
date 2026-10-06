"""Local-only frozen 2(case) x 2(format) x 2(label) x 2(row order) design."""
import hashlib,itertools,json,random,re
from pathlib import Path
from prepare_eligibility_priority import score,oracle,ATTR,LABEL
SEED=2026100601
ROOT=Path('outputs/preference_program/manifests/preference_factorial_v56')
PRIOR=Path('outputs/preference_program/manifests/eligibility_priority_v45')
def signature(c):
 return (tuple(c['priority']),tuple(sorted(o['attribute'] for o in c['options'] if o['eligible'])))
def render(c,ranked):
 text='User rule: choose an eligible option. Among eligible options, choose the attribute that appears earliest in my priority list. Priority list: '+', then '.join(c['priority'])+'.\nCatalog:\n'
 text+='\n'.join(f"{o['label']}: attribute={o['attribute']}; eligible={'yes' if o['eligible'] else 'no'}."+(f" priority_rank={1+c['priority'].index(o['attribute'])}." if ranked else '') for o in c['options'])
 return text+'\nReply with exactly one capital letter: A, B, C, or D.'
def main():
 assert not ROOT.exists(),'Never overwrite a registered design';ROOT.mkdir(parents=True)
 prior=[c for split in ['development','qualification','advice'] for c in map(json.loads,(PRIOR/f'{split}_cases.jsonl').read_text().splitlines())];used={signature(c) for c in prior};pool=[]
 for attrs,priority,excluded in itertools.product(itertools.permutations(ATTR),itertools.permutations(ATTR),range(4)):
  c=dict(priority=list(priority),options=[dict(label=l,attribute=a,eligible=i!=excluded) for i,(l,a) in enumerate(zip(LABEL,attrs))]);c['top_priority_ineligible']=priority[0]==attrs[excluded]
  if signature(c) not in used:pool.append(c)
 rng=random.Random(SEED);bases=[]
 for i,(topbad,gold) in enumerate([(False,'A'),(True,'B')]):
  candidates=[c for c in pool if c['top_priority_ineligible']==topbad and oracle(c)==gold and signature(c) not in used]
  c=rng.choice(candidates);c=dict(c,semantic_case_id=f'fresh-{i:02d}');bases.append(c);used.add(signature(c))
 rows=[];fixtures=0
 for base,ranked,rotation,reverse in itertools.product(bases,[False,True],[0,2],[False,True]):
  options=[dict(o,label=LABEL[(LABEL.index(o['label'])+rotation)%4]) for o in base['options']]
  if reverse:options.reverse()
  c=dict(base,options=options);case_id=f"{base['semantic_case_id']}-{'rank' if ranked else 'prose'}-l{rotation}-{'reverse' if reverse else 'forward'}";prompt=render(c,ranked);gold=oracle(c)
  # Independent rank-minimization oracle, semantic equivalence, unchanged scorer fixtures.
  assert gold==min((o for o in options if o['eligible']),key=lambda o:base['priority'].index(o['attribute']))['label'];assert signature(c)==signature(base)
  for label in LABEL:
   s=score(c,label);chosen=next(o for o in options if o['label']==label);assert s['correct']==(label==gold);assert s['eligibility_violation']==(not chosen['eligible']);fixtures+=1
  for invalid in ['', 'a','AB','Answer: '+gold]:assert score(c,invalid)['parse_failure'];fixtures+=1
  row=dict(c,case_id=case_id,format='explicit_rank' if ranked else 'prose',label_rotation=rotation,row_order='reverse' if reverse else 'forward',gold_action=gold,prompt=prompt,messages=[dict(role='user',content=prompt)],prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest());rows.append(row)
 rng.shuffle(rows);assert len(rows)==16;assert {g:sum(r['gold_action']==g for r in rows) for g in LABEL}==dict.fromkeys(LABEL,4)
 pairs={}
 for factor,others in [('format',['label_rotation','row_order']),('label_rotation',['format','row_order']),('row_order',['format','label_rotation'])]:
  grouped={}
  for r in rows:grouped.setdefault(tuple([r['semantic_case_id']]+[r[k] for k in others]),[]).append(r)
  assert len(grouped)==8 and all(len(v)==2 for v in grouped.values());pairs[factor]=[[r['case_id'] for r in v] for v in grouped.values()]
 files={'semantic_cases.jsonl':''.join(json.dumps(c)+'\n' for c in bases),'execution_manifest.jsonl':''.join(json.dumps(r)+'\n' for r in rows),'paired_contrasts.json':json.dumps(pairs,indent=2)+'\n'}
 for name,content in files.items():(ROOT/name).write_text(content)
 validation=dict(seed=SEED,model_calls=0,GPU_jobs=0,rows=16,semantic_cases=2,independent_fresh_semantic_signatures=True,prior_semantic_signatures=len({signature(c) for c in prior}),oracle_fixture_assertions=fixtures,paired_contrasts_per_factor=8,gold_letters_balanced=True,best_catalog_positions_balanced=True,scorer_sha256=hashlib.sha256(Path('scripts/prepare_eligibility_priority.py').read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},execution_authorized=False,advice_panel_blocked=True)
 positions=[next(i for i,o in enumerate(r['options']) if o['label']==r['gold_action']) for r in rows];assert {i:positions.count(i) for i in range(4)}==dict.fromkeys(range(4),4)
 (ROOT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print(json.dumps(validation,indent=2))
if __name__=='__main__':main()
