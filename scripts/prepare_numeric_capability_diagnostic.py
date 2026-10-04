"""Freeze16 no-advice numeric/order diagnostics; no model calls."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path('outputs/preference_program');source=root/'manifests/faulty_advice_verification_v36.jsonl'
base=[json.loads(x) for x in source.read_text().splitlines() if json.loads(x)['user_id']==0 and json.loads(x)['arm']=='no_advice'];assert len(base)==4
protocol={'status':'proposed frozen diagnostic; no generation or allocation authorized by preparation','source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'model':'Qwen/Qwen2.5-1.5B-Instruct','snapshot':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306','design':'2numeric representations x2listing orders x4cyclic label assignments=16','representations':['original signed six-decimal scores','rank-preserving positive integers1..4'],'orders':['canonical ABCD','reversed DCBA'],'advice':'none','decoding':{'do_sample':False,'max_new_tokens':24},'metrics':['exact argmax agreement','parse failures','A-response rate','first-listed response rate','paired correctness changes'],'outcome_control':'deterministic argmax of displayed numbers;original argmax identical in every matched pair','inference':'descriptive diagnostic on one reused smoke case; no user-level CI, preference recovery, arithmetic reward calculation or general capability claim','resource_limit_if_later_authorized':{'gpu':'one H200 MIG2g.35gb','walltime':'five minutes','load_watchdog_seconds':90,'runtime_staging_seconds':45},'compute_basis':'validated20-response smoke42s,peak framework1.19GB;queue/runtime uncertain','gate':'review protocol; no launch from preparation;320-record study remains gated'}
rows=[]
for b in base:
 values=np.array(b['expected_utilities']);ranks=np.empty(4,dtype=int);ranks[np.argsort(values)]=np.arange(1,5)
 for representation in ('decimals','integers'):
  nums=values if representation=='decimals' else ranks
  for order in ('canonical','reversed'):
   indices=list(range(4)) if order=='canonical' else list(range(3,-1,-1))
   scores=', '.join(f'{chr(65+i)}={nums[i]:+.6f}' if representation=='decimals' else f'{chr(65+i)}={nums[i]:d}' for i in indices)
   prompt='User preference scores: '+scores+'. Choose the action with the highest score. Reply with exactly one capital letter: A, B, C, or D.'
   gold=chr(65+int(np.argmax(nums)));assert gold==b['gold_action']
   rows.append({'user_id':0,'scenario_id':b['scenario_id'],'permutation_id':b['permutation_id'],'representation':representation,'listing_order':order,'prompt':prompt,'displayed_values':nums.tolist(),'original_utilities':values.tolist(),'gold_action':gold,'first_listed_action':chr(65+indices[0])})
assert len(rows)==16
p=root/'manifests/numeric_capability_v40_protocol.json';p.write_text(json.dumps(protocol,indent=2)+'\n')
m=p.with_name('numeric_capability_v40.jsonl');m.write_text(''.join(json.dumps(r)+'\n' for r in rows))
review=p.with_name('numeric_capability_v40_review.txt');review.write_text('Proposed only:16 responses; no generation performed.\n\n'+'\n\n'.join(f"{i+1}. {r['representation']}/{r['listing_order']}/rotation{r['permutation_id']}\n{r['prompt']}\nDeterministic expected answer: {r['gold_action']}" for i,r in enumerate(rows)))
print('Prepared16 exact prompts and deterministic answer controls;no calls')
