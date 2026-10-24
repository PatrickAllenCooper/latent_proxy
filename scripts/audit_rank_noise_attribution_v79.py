"""Independent v79 answer/posterior/action replay and paired contrast audit."""
import hashlib,itertools,json,math,random,sys
from pathlib import Path
ATTR=('quiet','spacious','portable','durable');ORDERS=list(itertools.permutations(ATTR));PAIRS=list(itertools.combinations(ATTR,2));CFG={'update10_plan10':(.1,.1),'update20_plan10':(.2,.1),'update10_plan20':(.1,.2),'update20_plan20':(.2,.2)}
def main():
 assert sys.version_info>=(3,12),'Exact replay uses Python3.12+ summation'
 root=Path('outputs/preference_program/results/rank_noise_attribution_v79');out=root/'results';receipt=json.loads((out/'application_receipt.json').read_text());m=json.loads(Path('outputs/preference_program/manifests/rank_noise_attribution_v79/manifest.json').read_text());custody=(root/'scheduler_and_custody.txt').read_text();hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('results','spool') for p in (root/folder).iterdir()}
 for n,h in hashes.items():assert h+'  /scratch/alpine/paco0228/latent_proxy_runs/rank-noise-attribution-v79/'+n in custody
 for n,h in receipt['artifacts'].items():assert hashes['results/'+n]==h
 assert receipt['source_sha256']==m['hashes']['source/rank_noise_attribution_v79.py'];assert m['hashes']['protocol.json']==hashes['results/protocol.json'];assert receipt['audit']['protocol_opens']>=1;assert receipt['native_syscalls_observed'] is False;assert receipt['elapsed_seconds']<20
 rows=[json.loads(x) for x in (out/'per_user.jsonl').read_text().splitlines()];traces=[json.loads(x) for x in (out/'question_traces.jsonl').read_text().splitlines()];assert len(rows)==11520 and len(traces)==18432
 key=lambda r:(r['seed'],r['user_id'],r['condition'],r['filter'],r['arm'])
 endpoints={(key(r),r['questions']):r for r in rows};assert len(endpoints)==len(rows);states={};randomqueries={};observedflips={};questionpairs={};checked=0
 def check(k,q,w):
  nonlocal checked
  seed,user,condition,filt,arm=k;r=endpoints[k,q];truth=ORDERS[user];assert r['true_order_index']==user;assert abs(w[user]-r['true_order_mass'])<1e-12
  rng=random.Random(seed*100+user);attrs=list(ATTR);rng.shuffle(attrs);excluded=rng.randrange(4);labels=[];regrets=[]
  for ex in range(4):
   eligible=[i for i in range(4) if i!=ex];a=max(eligible,key=lambda i:sum(w[j]*(3-order.index(attrs[i])) for j,order in enumerate(ORDERS)));labels.append('ABCD'[a]);best=max(3-truth.index(attrs[i]) for i in eligible);regrets.append((best-3+truth.index(attrs[a]))/(best+1))
  assert r['actions_by_excluded_option']==labels and r['target_action']==labels[excluded];assert abs(r['target_regret']-regrets[excluded])<1e-12;assert abs(r['transfer_regret']-sum(regrets[i] for i in range(4) if i!=excluded)/3)<1e-12;assert r['correct']==(regrets[excluded]==0);checked+=1
 for r in traces:
  k=key(r);seed,user,condition,filt,arm=k;q=r['question'];prior=states.get(k,[1/24]*24)
  if q==1:check(k,0,prior)
  h=.1 if filt=='hazard10' else 0;prior=[(1-h)*v+h/24 for v in prior];a,b=PAIRS[r['pair_index']];truth=ORDERS[user];latent=truth.index(a)<truth.index(b);flipped=random.Random(f'{seed}:{user}:{q}:{r["pair_index"]}').random()<.2;assert r['latent_answer']==latent and r['flipped']==flipped and r['answer']==(latent!=flipped)
  eps=CFG[condition][0];post=[v*((1-eps) if (order.index(a)<order.index(b))==r['answer'] else eps) for v,order in zip(prior,ORDERS)];z=sum(post);post=[v/z for v in post];assert len(r['posterior_weights'])==24 and all(abs(a-b)<1e-12 for a,b in zip(post,r['posterior_weights']));states[k]=post
  flipkey=(seed,user,q,r['pair_index']);observedflips.setdefault(flipkey,set()).add(r['flipped']);questionpairs[(*k,q)]=r['pair_index']
  if arm=='random':randomqueries.setdefault((seed,user,q),set()).add(r['pair_index'])
  if q in (2,4,6,8):check(k,q,post)
 assert checked==11520;assert all(len(v)==1 for v in randomqueries.values());assert all(len(v)==1 for v in observedflips.values())
 s=json.loads((out/'summary.json').read_text());get=lambda seed,user,c,f,a:endpoints[(seed,user,c,f,a),8]['transfer_regret']
 for group in s['conditions']:
  selected=[r for r in rows if all(r[k]==group[k] for k in ('condition','filter','arm','questions'))];assert len(selected)==72
  for metric in ('target_regret','transfer_regret','correct','entropy','true_order_mass'):assert abs(sum(r[metric] for r in selected)/72-group[metric])<1e-12
 def interval(d,seed):
  rng=random.Random(seed);v=sorted(sum(rng.choice(d) for _ in d)/24 for _ in range(1000));return [v[24],v[974]]
 for c in s['primary_contrasts']:
  other=c['contrast'].removeprefix('AIF-minus-');d=[sum(get(seed,user,c['condition'],c['filter'],'AIF')-get(seed,user,c['condition'],c['filter'],other) for seed in (7101,8101,9101))/3 for user in range(24)]
  assert abs(sum(d)/24-c['mean'])<1e-12;assert all(abs(a-b)<1e-12 for a,b in zip(interval(d,'v79:'+c['condition']+c['filter']+other),c['order_cluster_interval95']))
 for c in s['factorial_contrasts']:
  d=[sum(sum(coef*(get(seed,user,name,c['filter'],'AIF')-get(seed,user,name,c['filter'],'EIG')) for name,coef in c['coefficients'].items()) for seed in (7101,8101,9101))/3 for user in range(24)];assert abs(sum(d)/24-c['mean'])<1e-12;assert all(abs(a-b)<1e-12 for a,b in zip(interval(d,'v79factorial:'+c['filter']+c['contrast']),c['order_cluster_interval95']))
 for c in s['per_policy_matching']:
  d=[sum(get(seed,user,'update20_plan20',c['filter'],c['arm'])-get(seed,user,'update10_plan10',c['filter'],c['arm']) for seed in (7101,8101,9101))/3 for user in range(24)];assert abs(sum(d)/24-c['mean'])<1e-12;assert all(abs(a-b)<1e-12 for a,b in zip(interval(d,'v79matching:'+c['filter']+c['arm']),c['order_cluster_interval95']))
 agreement=[]
 for filt in ('static','hazard10'):
  for arm in ('EIG','decision_focused','AIF'):
   for name in ('update20_plan10','update10_plan20','update20_plan20'):
    keys=[(seed,user,q) for seed in (7101,8101,9101) for user in range(24) for q in range(1,9)];same=sum(questionpairs[seed,user,'update10_plan10',filt,arm,q]==questionpairs[seed,user,name,filt,arm,q] for seed,user,q in keys);agreement.append({'filter':filt,'arm':arm,'changed_configuration':name,'query_slot_agreement':same/len(keys),'interpretation':'adaptive trajectories may diverge; not fixed-history equivalence'})
 report={'custody_and_frozen_protocol_verified':True,'posterior_updates_and_potential_noise_replayed':18432,'raw_endpoint_actions_and_scores_verified':11520,'common_response_streams_verified':True,'random_query_invariance_verified':True,'all_primary_factorial_and_matching_contrasts_recomputed':True,'query_agreement':agreement,'source_hashes':hashes,'audit_python':sys.version,'compute_python':receipt['python_version'],'unique_latent_orders':24,'seed_repeats':3,'interval_scope':'exploratory unadjusted order-cluster bootstrap; not human population inference','model_calls':0,'GPU_allocations':0}
 (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print('all answers,24-weight posteriors,actions,scores and32 contrasts verified')
if __name__=='__main__':main()
