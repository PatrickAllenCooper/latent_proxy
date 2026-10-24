"""Frozen paired stream-block evaluation of unknown online noise."""
import hashlib,json,math,random,sys,time
from pathlib import Path
SEEDS=list(range(60101,60131));MODES=('fixed10','online','oracle');NOISES=(.1,.2,.3)
def qtile(v,p):
 x=(len(v)-1)*p;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def run(root):
 started=time.time();assert sys.version_info>=(3,12);endpoints={};groups={};sources={}
 for shard in range(3):
  p=root/f'shard{shard}/results/per_user.jsonl';sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
  for line in p.open():
   r=json.loads(line);k=(r['true_noise'],r['mode'],r['questions']);g=groups.setdefault(k,[]);g.append(r)
   if r['questions']==8:
    key=(r['seed'],r['user_id'],r['true_noise'],r['mode']);assert key not in endpoints;endpoints[key]=r
 assert len(endpoints)==6480 and {k[0] for k in endpoints}==set(SEEDS)
 curves=[]
 metrics=('target_regret','transfer_regret','correct','entropy','true_order_mass','estimated_noise','noise_bias','noise_absolute_error','noise_interval_covers_truth')
 for (noise,mode,budget),rows in sorted(groups.items()):
  assert len(rows)==720;curves.append({'true_noise':noise,'mode':mode,'questions':budget,'rows':720,**{k:sum(r[k] for r in rows)/720 for k in metrics}})
 rng=random.Random('v82stream-bootstrap');resamples=[rng.choices(range(30),k=30) for _ in range(10000)];estimates=[];paired=[]
 for noise in NOISES:
  for first,second in [('online','fixed10'),('online','oracle'),('oracle','fixed10')]:
   d=[]
   for seed in SEEDS:
    for user in range(24):
     a=endpoints[seed,user,noise,first];b=endpoints[seed,user,noise,second];paired.append({'seed':seed,'user_id':user,'true_noise':noise,'contrast':first+'-minus-'+second,'transfer_regret_difference':a['transfer_regret']-b['transfer_regret']})
    d.append(sum(endpoints[seed,u,noise,first]['transfer_regret']-endpoints[seed,u,noise,second]['transfer_regret'] for u in range(24))/24)
   boot=sorted(sum(d[i] for i in ids)/30 for ids in resamples);primary=noise==.2 and first=='online';alpha=.05/2 if primary else .05
   estimates.append({'true_noise':noise,'contrast':first+'-minus-'+second,'mean':sum(d)/30,'nominal_interval95':[qtile(boot,.025),qtile(boot,.975)],'decision_interval':[qtile(boot,alpha/2),qtile(boot,1-alpha/2)],'decision_interval_level':1-alpha,'primary':primary,'stream_blocks':30,'per_stream_means':d})
 result={'conditions':curves,'estimates':estimates,'streams':30,'finite_orders':24,'paired_scenarios_per_true_noise':720,'endpoint_rows':sum(len(v) for v in groups.values()),'query_traces_expected':51840,'primary_noise':.2,'primary_family':['online-minus-fixed10','online-minus-oracle'],'intervals':'paired30streamblocks,10000bootstrapresamples,2primaryBonferroni97.5percent','model_calls':0,'GPU_allocations':0}
 (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(root/'paired_effects.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in paired));(root/'analysis_receipt.json').write_text(json.dumps({'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_hashes':sources,'outputs':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('summary.json','paired_effects.jsonl')},'python_version':sys.version,'elapsed_seconds':time.time()-started,'model_calls':0,'GPU_allocations':0},indent=2)+'\n');print('complete30stream online noise comparison')
if __name__=='__main__':run(Path(sys.argv[1]))
