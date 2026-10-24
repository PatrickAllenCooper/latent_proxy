"""Post-run record/custody and independent posterior/action replay. No inference."""
import hashlib,itertools,json,math,random,sys
from pathlib import Path
ATTR=('quiet','spacious','portable','durable');ORDERS=list(itertools.permutations(ATTR));PAIRS=list(itertools.combinations(ATTR,2))
def main():
 if sys.version_info<(3,12):raise RuntimeError('Exact replay requires Python3.12+ summation; Python3.9 changes near-tied choices')
 root=Path('outputs/preference_program/results/rank_stress_v78');out=root/'results';receipt=json.loads((out/'application_receipt.json').read_text());manifest=json.loads(Path('outputs/preference_program/manifests/rank_stress_v78/manifest.json').read_text());custody=(root/'scheduler_and_custody.txt').read_text()
 hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('results','spool') for p in (root/folder).iterdir()}
 for n,d in hashes.items():assert d+'  /scratch/alpine/paco0228/latent_proxy_runs/rank-stress-v78/'+n in custody
 for n,d in receipt['artifacts'].items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==d
 assert receipt['source_sha256']==manifest['hashes']['source/rank_stress_v78.py'];assert manifest['hashes']['protocol.json']==receipt['artifacts']['protocol.json'];assert receipt['audit']['protocol_opens']>=1
 assert [x['phase'] for x in receipt['events']]==['protocol_frozen','simulation_complete','analysis_complete'];assert receipt['events'][-1]['at']-receipt['events'][0]['at']<20;assert sum(p.stat().st_size for p in out.iterdir())<manifest['artifact_budget_bytes']
 rows=[json.loads(x) for x in (out/'per_user.jsonl').read_text().splitlines()];traces=[json.loads(x) for x in (out/'question_traces.jsonl').read_text().splitlines()];assert len(rows)==11520 and len(traces)==18432
 key=lambda r:(r['seed'],r['user_id'],r['condition'],r['filter'],r['arm'])
 indexed={(key(r),r['questions']):r for r in rows};assert len(indexed)==len(rows);states={};reconstructed=[]
 def check(k,q,w):
  seed,user,condition,filt,arm=k;r=indexed[k,q];truth=ORDERS[user];truth=tuple(reversed(truth)) if condition=='change_after4' and q>4 else truth;assert ORDERS.index(truth)==r['true_order_index'];assert abs(w[ORDERS.index(truth)]-r['true_order_mass'])<1e-12
  rng=random.Random(seed*100+user);attrs=list(ATTR);rng.shuffle(attrs);excluded=rng.randrange(4);actions=[];regrets=[]
  for ex in range(4):
   eligible=[i for i in range(4) if i!=ex];chosen=max(eligible,key=lambda i:sum(w[j]*(3-order.index(attrs[i])) for j,order in enumerate(ORDERS)));best=max(3-truth.index(attrs[i]) for i in eligible);regret=(best-(3-truth.index(attrs[chosen])))/(best+1);actions.append('ABCD'[chosen]);regrets.append(regret)
  assert abs(regrets[excluded]-r['target_regret'])<1e-12,(k,q,'target',regrets[excluded],r['target_regret'],actions);assert abs(sum(regrets[i] for i in range(4) if i!=excluded)/3-r['transfer_regret'])<1e-12,(k,q,'transfer',regrets,r['transfer_regret'],actions);assert r['correct']==(regrets[excluded]==0)
  reconstructed.append({'seed':seed,'user_id':user,'condition':condition,'filter':filt,'arm':arm,'questions':q,'reconstructed_target_action':actions[excluded],'reconstructed_actions_by_excluded_option':actions,'derivation':'independent posterior replay; original run did not serialize final labels'})
 for r in traces:
  k=key(r);w=states.get(k,[1/24]*24);q=r['question'];seed,user,condition,filt,arm=k
  if q==1:check(k,0,w)
  hazard=.1 if filt=='hazard10' else 0;prior=[(1-hazard)*v+hazard/24 for v in w];a,b=PAIRS[r['pair_index']];truth=ORDERS[user];truth=tuple(reversed(truth)) if condition=='change_after4' and q>4 else truth;latent=truth.index(a)<truth.index(b);noise={'noise10':.1,'noise20':.2}.get(condition,0.);flipped=random.Random(f'{seed}:{user}:{q}:{r["pair_index"]}').random()<noise;assert latent==r['latent_answer'] and flipped==r['flipped'] and (latent!=flipped)==r['answer'];post=[v*(.9 if (order.index(a)<order.index(b))==r['answer'] else .1) for v,order in zip(prior,ORDERS)];z=sum(post);post=[x/z for x in post];assert abs(post[ORDERS.index(truth)]-r['posterior_true_mass'])<1e-12;states[k]=post
  if q in (2,4,6,8):check(k,q,post)
 assert len(reconstructed)==11520
 summary=json.loads((out/'summary.json').read_text())
 for g in summary['conditions']:
  selected=[r for r in rows if all(r[k]==g[k] for k in ('condition','filter','arm','questions'))];assert len(selected)==g['rows']==72
  for metric in ('target_regret','transfer_regret','correct','entropy','true_order_mass'):assert abs(sum(r[metric] for r in selected)/72-g[metric])<1e-12
 for c in summary['primary_contrasts']:
  other=c['contrast'].removeprefix('AIF-minus-');delta=[]
  for user in range(24):
   values={a:[r['transfer_regret'] for r in rows if (r['condition'],r['filter'],r['arm'],r['questions'],r['user_id'])==(c['condition'],c['filter'],a,8,user)] for a in ('AIF',other)}
   delta.append(sum(values['AIF'])/3-sum(values[other])/3)
  assert abs(sum(delta)/24-c['mean'])<1e-12
  rng=random.Random('v78:'+c['condition']+c['filter']+other);boot=sorted(sum(rng.choice(delta) for _ in delta)/24 for _ in range(1000))
  assert all(abs(a-b)<1e-12 for a,b in zip([boot[24],boot[974]],c['order_cluster_interval95']))
 (root/'reconstructed_actions.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in reconstructed))
 report={'custody_verified':True,'frozen_source_and_protocol_verified':True,'all18432_noisy_answers_and_posterior_updates_replayed':True,'all11520_endpoint_scores_recomputed':True,'aggregate_metrics_recomputed':True,'all16_primary_contrasts_and_intervals_recomputed':True,'audit_python_version':sys.version,'compute_python_version':'3.12.13','old_python39_replay':'failed on near-tied action because floating sum implementation differs; no scientific outputs were changed','unique_preference_orders':24,'seed_repeats':3,'independent_users_not72':True,'native_syscall_claim':False,'confidence_scope':'exploratory order-cluster intervals over finite synthetic panel;16 contrasts, not multiplicity adjusted','source_hashes':hashes,'model_calls':0,'GPU_allocations':0}
 (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print('custody,18432 updates,11520 scores,160 aggregates verified')
if __name__=='__main__':main()
