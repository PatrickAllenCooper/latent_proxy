"""Replay saved answers, retain posterior actions, and verify original metrics."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from src.agents.finite_menu_inference import UserParticles,prepared_menus,update_weights,posterior_mean
from src.training.synthetic_users import UserType
from src.evaluation.preference_benchmark import make_scenario,expected_utilities,evaluate_action

root=Path('outputs/preference_program/results'); source=root/'finite_menu_discovery_replication_v2'
c=json.loads((source/'summary.json').read_text())['config']
saved={(int(r['user_id']),r['arm'],int(r['budget'])):r for r in csv.DictReader((source/'per_user.csv').open())}
traces={}
for line in (source/'queries.jsonl').read_text().splitlines():
 r=json.loads(line);traces.setdefault((r['user_id'],r['arm']),[]).append(r)
menus=[make_scenario(c['seed']+300000+i) for i in range(c['n_targets'])]
queries=[make_scenario(c['seed']+100000+i) for i in range(c['n_queries'])]
rows=[];max_error=0.;max_profile_error=0.;disagreements={}
for u in range(c['n_users']):
 particles=UserParticles.sample(c['n_particles'],c['seed']+500000+u)
 _,qb=prepared_menus(queries,particles);tu,_=prepared_menus(menus,particles)
 for arm in c['arms']:
  w=np.ones(c['n_particles'])/c['n_particles'];answers=sorted(traces[u,arm],key=lambda r:r['round'])
  assert len(answers)==8
  for b in range(9):
   if b in c['budgets']:
    s=saved[u,arm,b];theta=UserType(*[float(s['true_'+k]) for k in ('gamma','alpha','lambda')]);mean=posterior_mean(w,particles)
    max_profile_error=max(max_profile_error,*[abs(v-float(s['estimated_'+k])) for k,v in zip(('gamma','alpha','lambda'),(mean.gamma,mean.alpha,mean.lambda_))])
    posterior=np.einsum('n,tna->ta',w,tu);regrets=[]
    for i,m in enumerate(menus):
     point=expected_utilities(m,mean);pa=int(posterior[i].argmax());pe=int(point.argmax())
     assert m.feasible[pa] and m.feasible[pe]
     pr=evaluate_action(m,theta,pa)['normalized_regret'];er=evaluate_action(m,theta,pe)['normalized_regret'];regrets.append(pr)
     rows.append({'user_id':u,'arm':arm,'budget':b,'scenario_id':m.scenario_id,'posterior_action':pa,'point_action':pe,'disagree':pa!=pe,'posterior_regret':pr,'point_regret':er,'posterior_utilities':json.dumps(posterior[i].tolist()),'point_utilities':json.dumps(point.tolist()),'payoffs':json.dumps(m.payoffs.tolist()),'probabilities':json.dumps(m.probabilities.tolist()),'feasible':json.dumps(m.feasible.tolist())})
    max_error=max(max_error,abs(np.mean(regrets)-float(s['reward_regret'])))
   if b<8:
    r=answers[b];assert r['round']==b+1 and r['scenario_id']==queries[r['query_id']].scenario_id
    w=update_weights(w,qb[r['query_id'],:,r['choice']])
assert max_error<1e-10 and max_profile_error<1e-10
for arm in c['arms']:
 for b in c['budgets']:
  group=[r for r in rows if r['arm']==arm and r['budget']==b];different=[r for r in group if r['disagree']]
  disagreements[f'{arm}/{b}']={'decisions':len(group),'disagreements':len(different),'mean_point_minus_posterior_regret':float(np.mean([r['point_regret']-r['posterior_regret'] for r in group]))}
out=root/'discovery_decision_replay_v34';out.mkdir(exist_ok=True)
with (out/'per_decision.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
report={'source_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('queries.jsonl','per_user.csv','summary.json')},'records':len(rows),'max_saved_regret_error':max_error,'max_saved_profile_error':max_profile_error,'cells':disagreements,'method':'Deterministic particle prior reconstructed from recorded seed; only saved answers replayed, no new response sampling. Shared environment state and feasibility retained in every row.'}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
