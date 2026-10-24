"""Independent full-factorial replay and primary-family interval verification."""
import hashlib,itertools,json,math,random,sys
from pathlib import Path
ATTR=('quiet','spacious','portable','durable');ORDERS=list(itertools.permutations(ATTR));PAIRS=list(itertools.combinations(ATTR,2));CONFIGS=('update10_plan10','update20_plan10','update10_plan20','update20_plan20');EPS=dict(zip(CONFIGS,(.1,.2,.1,.2)))
COEF={'average_update_effect':(-.5,.5,-.5,.5),'average_planner_effect':(-.5,-.5,.5,.5),'interaction':(1,-1,-1,1),'joint_correction':(-1,0,0,1),'update_at_plan10':(-1,1,0,0),'planner_at_update10':(-1,0,1,0),'update_at_plan20':(0,0,-1,1),'planner_at_update20':(0,-1,0,1)}
def quantile(v,p):
 x=(len(v)-1)*p;i=math.floor(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def main():
 assert sys.version_info>=(3,12)
 root=Path('outputs/preference_program/results/rank_noise_components_v81');manifest=json.loads(Path('outputs/preference_program/manifests/rank_noise_components_v81/manifest.json').read_text());custody=(root/'scheduler_and_custody.txt').read_text();hashes={};indexed={};receipts=[]
 key=lambda r:(r['seed'],r['user_id'],r['condition'],r['arm'])
 for shard in (0,1):
  folder=root/f'shard{shard}/results';a=json.loads((folder/'application_receipt.json').read_text());receipts.append(a)
  for place in (folder,root/f'shard{shard}/spool'):
   for p in place.iterdir():
    h=hashlib.sha256(p.read_bytes()).hexdigest();rel=str(p.relative_to(root));hashes[rel]=h;assert h+'  /scratch/alpine/paco0228/latent_proxy_runs/rank-noise-components-v81/'+rel in custody
  for name,h in a['artifacts'].items():assert hashes[f'shard{shard}/results/'+name]==h
  assert a['source_sha256']==manifest['hashes']['source/rank_noise_components_v81.py'];assert a['kernel_sha256']==manifest['hashes']['source/rank_noise_attribution_v79.py'];assert hashes[f'shard{shard}/results/protocol.json']==manifest['hashes'][f'protocol_shard{shard}.json'];assert a['elapsed_seconds']<20 and a['records']==14400 and a['trace_records']==23040 and a['audit']['protocol_opens']>=1;assert a['native_syscalls_observed'] is False
  for line in (folder/'per_user.jsonl').open():
   r=json.loads(line);k=(key(r),r['questions']);assert k not in indexed;indexed[k]=r
 assert len(indexed)==28800 and {k[0][0] for k in indexed}==set(range(40101,40131));states={};flips={};checked=0;updates=0
 def check(k,q,w):
  nonlocal checked
  seed,user,c,arm=k;r=indexed[k,q];rng=random.Random(seed*100+user);attrs=list(ATTR);rng.shuffle(attrs);ex=rng.randrange(4);truth=ORDERS[user];actions=[];regrets=[]
  for e in range(4):
   eligible=[i for i in range(4) if i!=e];chosen=max(eligible,key=lambda i:sum(p*(3-order.index(attrs[i])) for p,order in zip(w,ORDERS)));actions.append('ABCD'[chosen]);best=max(3-truth.index(attrs[i]) for i in eligible);regrets.append((best-3+truth.index(attrs[chosen]))/(best+1))
  assert r['target_action']==actions[ex] and r['actions_by_excluded_option']==actions;assert abs(regrets[ex]-r['target_regret'])<1e-12 and abs(sum(regrets[i] for i in range(4) if i!=ex)/3-r['transfer_regret'])<1e-12;assert abs(w[user]-r['true_order_mass'])<1e-12 and r['correct']==(regrets[ex]==0);checked+=1
 for shard in (0,1):
  for line in (root/f'shard{shard}/results/question_traces.jsonl').open():
   r=json.loads(line);k=key(r);seed,user,c,arm=k;q=r['question'];w=states.get(k,[1/24]*24)
   if q==1:check(k,0,w)
   prior=[.9*p+.1/24 for p in w];x,y=PAIRS[r['pair_index']];truth=ORDERS[user];latent=truth.index(x)<truth.index(y);flip=random.Random(f'{seed}:{user}:{q}:{r["pair_index"]}').random()<.2;assert (latent,flip,latent!=flip)==(r['latent_answer'],r['flipped'],r['answer']);eps=EPS[c];post=[p*((1-eps) if (order.index(x)<order.index(y))==r['answer'] else eps) for p,order in zip(prior,ORDERS)];z=sum(post);post=[v/z for v in post];assert len(r['posterior_weights'])==24 and all(abs(a-b)<1e-12 for a,b in zip(post,r['posterior_weights']));states[k]=post;flips.setdefault((seed,user,q,r['pair_index']),set()).add(flip);updates+=1
   if q in (2,4,6,8):check(k,q,post)
 assert checked==28800 and updates==46080 and all(len(v)==1 for v in flips.values())
 raw=[json.loads(x) for x in (root/'paired_effects.jsonl').read_text().splitlines()];assert len(raw)==720
 get=lambda seed,user,c,a:indexed[(seed,user,c,a),8]['transfer_regret']
 for r in raw:
  seed,user=r['seed'],r['user_id']
  for scope in ('AIF','EIG','gap'):
   ys=[get(seed,user,c,'AIF')-get(seed,user,c,'EIG') if scope=='gap' else get(seed,user,c,scope) for c in CONFIGS]
   for name,coef in COEF.items():assert abs(r[scope+':'+name]-sum(v*c for v,c in zip(ys,coef)))<1e-12
  for c in CONFIGS:assert abs(r['gapcell:'+c]-(get(seed,user,c,'AIF')-get(seed,user,c,'EIG')))<1e-12
 summary=json.loads((root/'summary.json').read_text());rng=random.Random('v81-stream-bootstrap-fixed');resamples=[rng.choices(range(30),k=30) for _ in range(10000)]
 for e in summary['estimates']:
  key_=e['contrast'];values=[sum(r[key_] for r in raw if r['seed']==seed)/24 for seed in range(40101,40131)];boots=sorted(sum(values[i] for i in ids)/30 for ids in resamples);assert abs(sum(values)/30-e['mean'])<1e-12;primary=key_ in ('AIF:average_update_effect','AIF:average_planner_effect','AIF:interaction');assert e['primary']==primary;alpha=.05/3 if primary else .05
  for field,probs in [('nominal_interval95',(.025,.975)),('decision_interval',(alpha/2,1-alpha/2))]:assert all(abs(a-b)<1e-12 for a,b in zip(e[field],[quantile(boots,p) for p in probs]))
 for g in summary['conditions']:
  selected=[r for (k,q),r in indexed.items() if (k[2],k[3],q)==(g['condition'],g['arm'],g['questions'])];assert len(selected)==720
  for metric in ('target_regret','transfer_regret','correct','entropy','true_order_mass'):assert abs(sum(r[metric] for r in selected)/720-g[metric])<1e-12
 a=json.loads((root/'analysis_receipt.json').read_text());assert a['source_sha256']==manifest['hashes']['source/analyze_rank_noise_components_v81.py']
 for name,h in a['outputs'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
 report={'frozen_sources_protocols_kernel_and_custody_verified':True,'raw_actions_and_scores_replayed':checked,'all24weight_posterior_updates_and_noisy_answers_replayed':updates,'all720paired_effects_and28intervals_recomputed':True,'primary_family3_and_Bonferroni_intervals_verified':True,'common_potential_answers_verified':True,'streams':30,'finite_orders':24,'model_calls':0,'GPU_allocations':0,'audit_python':sys.version,'raw_artifact_hashes':hashes,'scope':'finite synthetic domain;stream-block uncertainty;nothuman/LLM/RL evidence'};(root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print('validated28800scores,46080updates,720paired factorials and28contrasts')
if __name__=='__main__':main()
