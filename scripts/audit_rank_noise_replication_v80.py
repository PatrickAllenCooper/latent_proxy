"""Independent30stream posterior/action replay, paired estimates and intervals."""
import hashlib,itertools,json,math,random,sys
from pathlib import Path
ATTR=('quiet','spacious','portable','durable');ORDERS=list(itertools.permutations(ATTR));PAIRS=list(itertools.combinations(ATTR,2));SEEDS=list(range(20101,20131));CONFIGS={'update10_plan10':.1,'update20_plan20':.2}
def main():
 assert sys.version_info>=(3,12)
 root=Path('outputs/preference_program/results/rank_noise_replication_v80');out=root/'results';m=json.loads(Path('outputs/preference_program/manifests/rank_noise_replication_v80/manifest.json').read_text());receipt=json.loads((out/'application_receipt.json').read_text());custody=(root/'scheduler_and_custody.txt').read_text();hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('results','spool') for p in (root/folder).iterdir()}
 for name,h in hashes.items():assert h+'  /scratch/alpine/paco0228/latent_proxy_runs/rank-noise-replication-v80/'+name in custody
 for name,h in receipt['artifacts'].items():assert hashes['results/'+name]==h
 assert receipt['source_sha256']==m['hashes']['source/rank_noise_replication_v80.py'];assert receipt['kernel_sha256']==m['hashes']['source/rank_noise_attribution_v79.py'];assert receipt['artifacts']['protocol.json']==m['hashes']['protocol.json'];assert receipt['elapsed_seconds']<20 and receipt['audit']['protocol_opens']>=1;assert sum(p.stat().st_size for p in out.iterdir())<33554432
 protocol=json.loads((out/'protocol.json').read_text());assert protocol['seeds']==SEEDS;assert protocol['AIF_weight']==.1 and protocol['hazard']==.1
 rows=[json.loads(x) for x in (out/'per_user.jsonl').read_text().splitlines()];traces=[json.loads(x) for x in (out/'question_traces.jsonl').read_text().splitlines()];assert len(rows)==14400 and len(traces)==23040
 key=lambda r:(r['seed'],r['user_id'],r['condition'],r['arm']);indexed={(key(r),r['questions']):r for r in rows};assert len(indexed)==len(rows);states={};flips={};checked=0
 def check(k,q,w):
  nonlocal checked
  seed,user,c,a=k;r=indexed[k,q];truth=ORDERS[user];rng=random.Random(seed*100+user);attrs=list(ATTR);rng.shuffle(attrs);ex=rng.randrange(4);actions=[];regrets=[]
  for excluded in range(4):
   eligible=[i for i in range(4) if i!=excluded];chosen=max(eligible,key=lambda i:sum(p*(3-order.index(attrs[i])) for p,order in zip(w,ORDERS)));actions.append('ABCD'[chosen]);best=max(3-truth.index(attrs[i]) for i in eligible);regrets.append((best-3+truth.index(attrs[chosen]))/(best+1))
  assert r['target_action']==actions[ex] and r['actions_by_excluded_option']==actions;assert abs(r['target_regret']-regrets[ex])<1e-12;assert abs(r['transfer_regret']-sum(regrets[i] for i in range(4) if i!=ex)/3)<1e-12;assert abs(w[user]-r['true_order_mass'])<1e-12;assert r['correct']==(regrets[ex]==0);checked+=1
 for r in traces:
  k=key(r);seed,user,c,a=k;q=r['question'];w=states.get(k,[1/24]*24)
  if q==1:check(k,0,w)
  w=[.9*p+.1/24 for p in w];x,y=PAIRS[r['pair_index']];truth=ORDERS[user];latent=truth.index(x)<truth.index(y);flip=random.Random(f'{seed}:{user}:{q}:{r["pair_index"]}').random()<.2;assert (latent,flip,latent!=flip)==(r['latent_answer'],r['flipped'],r['answer']);eps=CONFIGS[c];post=[p*((1-eps) if (order.index(x)<order.index(y))==r['answer'] else eps) for p,order in zip(w,ORDERS)];z=sum(post);post=[p/z for p in post];assert all(abs(a-b)<1e-12 for a,b in zip(post,r['posterior_weights']));states[k]=post;flips.setdefault((seed,user,q,r['pair_index']),set()).add(flip)
  if q in (2,4,6,8):check(k,q,post)
 assert checked==14400 and all(len(v)==1 for v in flips.values())
 summary=json.loads((out/'summary.json').read_text());raw=[json.loads(x) for x in (out/'paired_effects.jsonl').read_text().splitlines()];assert len(raw)==720
 get=lambda seed,user,c,a:indexed[(seed,user,c,a),8]['transfer_regret']
 for r in raw:
  seed,user=r['seed'],r['user_id'];a0=get(seed,user,'update10_plan10','AIF');e0=get(seed,user,'update10_plan10','EIG');a1=get(seed,user,'update20_plan20','AIF');e1=get(seed,user,'update20_plan20','EIG');expected={'joint_gap_correction':(a1-e1)-(a0-e0),'baseline_AIF_minus_EIG':a0-e0,'matched_AIF_minus_EIG':a1-e1,'AIF_absolute_correction':a1-a0,'EIG_absolute_correction':e1-e0}
  assert all(abs(r[k]-v)<1e-12 for k,v in expected.items())
 for c in summary['estimates']:
  name=c['contrast'];by_seed=[sum(r[name] for r in raw if r['seed']==seed)/24 for seed in SEEDS];by_order=[sum(r[name] for r in raw if r['user_id']==u)/30 for u in range(24)];assert abs(sum(by_seed)/30-c['mean'])<1e-12
  for key_,values in [('stream',by_seed),('order',by_order)]:
   rng=random.Random('v80:'+key_+':'+name);boot=sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(10000));field='stream_block_interval95' if key_=='stream' else 'order_cluster_interval95';assert all(abs(a-b)<1e-12 for a,b in zip([boot[249],boot[9749]],c[field]))
 for c in summary['conditions']:
  selected=[r for r in rows if all(r[k]==c[k] for k in ('condition','arm','questions'))];assert len(selected)==720
  for metric in ('target_regret','transfer_regret','correct','entropy','true_order_mass'):assert abs(sum(r[metric] for r in selected)/720-c[metric])<1e-12
 report={'fresh_streams':30,'finite_enumerated_orders':24,'paired_scenarios':720,'raw_actions_and_scores_replayed':14400,'answers_and_full_posteriors_replayed':23040,'paired_contrasts_and_both_interval_types_recomputed':True,'common_potential_noise_verified':True,'frozen_source_protocol_kernel_and_custody_verified':True,'no_AIF_setting_change':True,'source_hashes':hashes,'audit_python':sys.version,'compute_python':receipt['python_version'],'primary_inference_scope':'MonteCarlo uncertainty across30freshstreamblocks over the fixed complete24order domain; not human inference','model_calls':0,'GPU_allocations':0};(root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print('validated14400scores,23040updates,720paired effects and all intervals')
if __name__=='__main__':main()
