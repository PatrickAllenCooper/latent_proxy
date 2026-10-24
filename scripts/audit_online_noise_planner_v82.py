"""Independent joint-noise replay, leakage/timeline controls and scoring audit."""
import hashlib,itertools,json,math,random,sys
from pathlib import Path
ATTR=('quiet','spacious','portable','durable');ORDERS=list(itertools.permutations(ATTR));PAIRS=list(itertools.combinations(ATTR,2));GRID=[k/20 for k in range(9)];PRIOR=[math.comb(8,k)*.25**k*.75**(8-k) for k in range(9)]
def state(j):
 ps=[sum(row) for row in j];mean=round(sum(p*e for p,e in zip(ps,GRID)),12)
 def qt(q):
  a=0
  for e,p in zip(GRID,ps):
   a+=p
   if a>=q:return e
  return GRID[-1]
 return mean,ps,[qt(.05),qt(.95)]
def noise_update(j,pair,answer):
 a,b=PAIRS[pair];v=[[p*((1-e) if (o.index(a)<o.index(b))==answer else e) for p,o in zip(row,ORDERS)] for e,row in zip(GRID,j)];z=sum(sum(row) for row in v);return [[p/z for p in row] for row in v]
def qt(v,p):
 x=(len(v)-1)*p;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def main():
 assert sys.version_info>=(3,12)
 root=Path('outputs/preference_program/results/online_noise_planner_v82');m=json.loads(Path('outputs/preference_program/manifests/online_noise_planner_v82/manifest.json').read_text());custody=(root/'scheduler_and_custody.txt').read_text();hashes={};indexed={};receipts=[]
 key=lambda r:(r['seed'],r['user_id'],r['true_noise'],r['mode'])
 for shard in range(3):
  folder=root/f'shard{shard}/results';a=json.loads((folder/'application_receipt.json').read_text());receipts.append(a)
  for place in (folder,root/f'shard{shard}/spool'):
   for p in place.iterdir():
    h=hashlib.sha256(p.read_bytes()).hexdigest();rel=str(p.relative_to(root));hashes[rel]=h;assert h+'  /scratch/alpine/paco0228/latent_proxy_runs/online-noise-planner-v82/'+rel in custody
  for n,h in a['artifacts'].items():assert hashes[f'shard{shard}/results/'+n]==h
  assert a['source_sha256']==m['hashes']['source/online_noise_planner_v82.py'];assert a['kernel_sha256']==m['hashes']['source/rank_noise_attribution_v79.py'];assert hashes[f'shard{shard}/results/protocol.json']==m['hashes'][f'protocol_shard{shard}.json'];assert a['records']==10800 and a['trace_records']==17280 and a['elapsed_seconds']<20 and a['audit']['protocol_opens']>=1
  for line in (folder/'per_user.jsonl').open():
   r=json.loads(line);k=(key(r),r['questions']);assert k not in indexed;indexed[k]=r
 assert len(indexed)==32400 and {k[0][0] for k in indexed}==set(range(60101,60131));checked=updates=0;noise10_paths={}
 def check(k,q,w,j):
  nonlocal checked
  seed,user,noise,mode=k;r=indexed[k,q];mean,marginal,interval=state(j);assert r['estimated_noise']==mean;assert all(abs(a-b)<1e-12 for a,b in zip(marginal,r['noise_marginal']));assert r['noise_interval90']==interval and r['noise_interval_covers_truth']==(interval[0]<=noise<=interval[1]);assert abs(r['noise_bias']-(mean-noise))<1e-12 and abs(r['noise_absolute_error']-abs(mean-noise))<1e-12;assert r['noise_estimator_used_for_planning']==(mode=='online')
  rng=random.Random(seed*100+user);attrs=list(ATTR);rng.shuffle(attrs);ex=rng.randrange(4);truth=ORDERS[user];actions=[];regrets=[]
  for e in range(4):
   eligible=[i for i in range(4) if i!=e];a=max(eligible,key=lambda i:sum(p*(3-o.index(attrs[i])) for p,o in zip(w,ORDERS)));actions.append('ABCD'[a]);best=max(3-truth.index(attrs[i]) for i in eligible);regrets.append((best-3+truth.index(attrs[a]))/(best+1))
  assert r['target_action']==actions[ex] and r['actions_by_excluded_option']==actions;assert abs(r['target_regret']-regrets[ex])<1e-12 and abs(r['transfer_regret']-sum(regrets[i] for i in range(4) if i!=ex)/3)<1e-12;assert abs(r['true_order_mass']-w[user])<1e-12;assert r['correct']==(regrets[ex]==0);checked+=1
 for shard in range(3):
  previous=None;lastq=8;w=None;j=None
  for line in (root/f'shard{shard}/results/question_traces.jsonl').open():
   r=json.loads(line);k=key(r);seed,user,noise,mode=k;q=r['question']
   if q==1:
    assert previous!=k and lastq==8;w=[1/24]*24;j=[[p/24]*24 for p in PRIOR];check(k,0,w,j);lastq=0
   else:assert previous==k
   assert q==lastq+1;previous=k;lastq=q
   before,_,_=state(j);eps=.1 if mode=='fixed10' else noise if mode=='oracle' else before;assert r['estimated_noise_before_query']==before and r['planner_noise_before_query']==eps
   # The online estimate is derived solely from the prior observed history.
   if mode=='online':assert eps==before
   a,b=PAIRS[r['pair_index']];truth=ORDERS[user];latent=truth.index(a)<truth.index(b);flip=random.Random(f'{seed}:{user}:{q}:{r["pair_index"]}').random()<noise;assert (latent,flip,latent!=flip)==(r['latent_answer'],r['flipped'],r['answer'])
   prior=[.9*p+.1/24 for p in w];v=[p*(.9 if (o.index(a)<o.index(b))==r['answer'] else .1) for p,o in zip(prior,ORDERS)];z=sum(v);w=[p/z for p in v];assert all(abs(a-b)<1e-12 for a,b in zip(w,r['posterior_weights']))
   j=noise_update(j,r['pair_index'],r['answer']);after,post,_=state(j);assert after==r['estimated_noise_after_answer'] and all(abs(a-b)<1e-12 for a,b in zip(post,r['noise_marginal_after_answer']));updates+=1
   if noise==.1 and mode in ('fixed10','oracle'):noise10_paths.setdefault((seed,user,mode),[]).append((r['pair_index'],r['answer']))
   if q in (2,4,6,8):check(k,q,w,j)
  assert lastq==8
 assert checked==32400 and updates==51840
 for seed in range(60101,60131):
  for user in range(24):assert noise10_paths[seed,user,'fixed10']==noise10_paths[seed,user,'oracle']
 s=json.loads((root/'summary.json').read_text());rng=random.Random('v82stream-bootstrap');ids=[rng.choices(range(30),k=30) for _ in range(10000)]
 for e in s['estimates']:
  first,second=e['contrast'].split('-minus-');noise=e['true_noise'];values=[sum(indexed[(seed,u,noise,first),8]['transfer_regret']-indexed[(seed,u,noise,second),8]['transfer_regret'] for u in range(24))/24 for seed in range(60101,60131)];boots=sorted(sum(values[i] for i in rr)/30 for rr in ids);assert abs(sum(values)/30-e['mean'])<1e-12;primary=noise==.2 and first=='online';assert primary==e['primary'];alpha=.025 if primary else .05
  for field,ps in [('nominal_interval95',(.025,.975)),('decision_interval',(alpha/2,1-alpha/2))]:assert all(abs(a-b)<1e-12 for a,b in zip(e[field],[qt(boots,p) for p in ps]))
 for g in s['conditions']:
  rows=[r for (k,q),r in indexed.items() if (k[2],k[3],q)==(g['true_noise'],g['mode'],g['questions'])];assert len(rows)==720
  for name in ('target_regret','transfer_regret','correct','entropy','true_order_mass','estimated_noise','noise_bias','noise_absolute_error','noise_interval_covers_truth'):assert abs(sum(r[name] for r in rows)/720-g[name])<1e-12
 ar=json.loads((root/'analysis_receipt.json').read_text());assert ar['source_sha256']==m['hashes']['source/analyze_online_noise_planner_v82.py']
 for n,h in ar['outputs'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
 report={'frozen_source_protocol_kernel_and_custody_verified':True,'raw_actions_and_scores_verified':32400,'joint_noise_and_main_order_updates_replayed':51840,'online_before_answer_timeline_verified':True,'no_true_noise_or_preferences_needed_to_reconstruct_online_estimates':True,'fixed10_and_oracle10_queries_identical_in_all720trajectories':True,'all_primary_and_secondary_intervals_recomputed':True,'primary_family2_Bonferroni97_5percent_verified':True,'streams':30,'finite_orders':24,'true_noise_rates':[.1,.2,.3],'source_hashes':hashes,'audit_python':sys.version,'model_calls':0,'GPU_allocations':0,'scope':'eight-question certainty-equivalent plug-in online learner;notallpossibleonline estimators'};(root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print('validated32400scores,51840joint-noiseupdates,timeline,controls and intervals')
if __name__=='__main__':main()
