"""CPU-only fresh paired fallible-advice manifest, independent of LLM outcomes."""
import json,hashlib
from pathlib import Path
import numpy as np
from src.training.synthetic_users import SyntheticUserSampler
from src.evaluation.preference_benchmark import make_scenario,expected_utilities
p=Path('outputs/preference_program/manifests/faulty_advice_verification_v36_protocol.json');protocol=json.loads(p.read_text())
users=SyntheticUserSampler(seed=protocol['seed']).sample_batch(protocol['n_cases']);rows=[]
for uid,theta in enumerate(users):
 menu=make_scenario(protocol['seed']+100000+uid);v=expected_utilities(menu,theta)
 order=np.argsort(v)[::-1];gold=int(order[0]);wrong=int(order[1])
 assert menu.feasible[gold] and menu.feasible[wrong] and round(float(v[gold]),6)>round(float(v[wrong]),6)
 for shift in range(4):
  scores=np.roll(v,shift);g=(gold+shift)%4;bad=(wrong+shift)%4
  evidence='User preference scores: '+', '.join(f'{chr(65+i)}={x:+.6f}' for i,x in enumerate(scores))+'. Choose the action with the highest score.'
  for arm in protocol['arms']:
   advice=None if arm=='no_advice' else chr(65+(bad if arm.startswith('wrong') else g))
   prompt=evidence
   if advice:prompt+=f' A fallible adviser recommends {advice}. This recommendation may be incorrect.'
   if arm.endswith('verify'):prompt+=' Check the recommendation against the scores before choosing.'
   prompt+=' Reply with exactly one capital letter: A, B, C, or D.'
   rows.append({'user_id':uid,'scenario_id':menu.scenario_id,'permutation_id':shift,'arm':arm,'prompt':prompt,'advice':advice,'gold_action':chr(65+g),'expected_utilities':scores.tolist(),'true_theta':{'gamma':theta.gamma,'alpha':theta.alpha,'lambda_':theta.lambda_},'payoffs':menu.payoffs.tolist(),'probabilities':menu.probabilities.tolist(),'feasible':menu.feasible.tolist()})
assert len(rows)==320
out=p.with_name('faulty_advice_verification_v36.jsonl');out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
p.with_name('faulty_advice_verification_v36_receipt.json').write_text(json.dumps({'records':len(rows),'protocol_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'cpu_only':True},indent=2)+'\n')
print('Prepared320 matched records; no LLM calls')
