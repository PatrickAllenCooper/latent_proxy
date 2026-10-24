"""Frozen factorial analysis; complete all30streams before reading effects."""
import hashlib,json,math,random,sys,time
from pathlib import Path
CONFIGS=('update10_plan10','update20_plan10','update10_plan20','update20_plan20')
SPECS={'average_update_effect':(-.5,.5,-.5,.5),'average_planner_effect':(-.5,-.5,.5,.5),'interaction':(1,-1,-1,1),'joint_correction':(-1,0,0,1),'update_at_plan10':(-1,1,0,0),'planner_at_update10':(-1,0,1,0),'update_at_plan20':(0,0,-1,1),'planner_at_update20':(0,-1,0,1)}
PRIMARY=('average_update_effect','average_planner_effect','interaction');SEEDS=list(range(40101,40131))
def quantile(v,p):
 h=(len(v)-1)*p;i=int(h);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(h-i)
def run(root):
 started=time.time();assert sys.version_info>=(3,12);lookup={};sums={};counts={};source_hashes={}
 for shard in (0,1):
  p=root/f'shard{shard}/results/per_user.jsonl';source_hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
  for line in p.open():
   r=json.loads(line);key=(r['seed'],r['user_id'],r['condition'],r['arm'])
   if r['questions']==8:assert key not in lookup;lookup[key]=r
   group=(r['condition'],r['arm'],r['questions']);counts[group]=counts.get(group,0)+1;totals=sums.setdefault(group,{k:[] for k in ('target_regret','transfer_regret','correct','entropy','true_order_mass')})
   for k in totals:totals[k].append(r[k])
 assert len(lookup)==5760 and {k[0] for k in lookup}==set(SEEDS)
 get=lambda seed,user,c,a:lookup[seed,user,c,a]['transfer_regret']
 raw=[]
 for seed in SEEDS:
  for user in range(24):
   values={c:{a:get(seed,user,c,a) for a in ('AIF','EIG')} for c in CONFIGS};r={'seed':seed,'user_id':user}
   for scope in ('AIF','EIG','gap'):
    ys=[values[c]['AIF']-values[c]['EIG'] if scope=='gap' else values[c][scope] for c in CONFIGS]
    for name,coef in SPECS.items():r[scope+':'+name]=sum(v*w for v,w in zip(ys,coef))
   for c in CONFIGS:r['gapcell:'+c]=values[c]['AIF']-values[c]['EIG']
   raw.append(r)
 rng=random.Random('v81-stream-bootstrap-fixed');indices=[rng.choices(range(30),k=30) for _ in range(10000)];estimates=[]
 for key in [k for k in raw[0] if k not in ('seed','user_id')]:
  by_seed=[sum(r[key] for r in raw if r['seed']==seed)/24 for seed in SEEDS];boots=sorted(sum(by_seed[i] for i in ids)/30 for ids in indices);primary=key.startswith('AIF:') and key.split(':')[1] in PRIMARY;alpha=.05/3 if primary else .05
  estimates.append({'contrast':key,'mean':sum(by_seed)/30,'nominal_interval95':[quantile(boots,.025),quantile(boots,.975)],'decision_interval':[quantile(boots,alpha/2),quantile(boots,1-alpha/2)],'decision_interval_level':1-alpha,'primary':primary,'stream_blocks':30,'per_stream_means':by_seed})
 curves=[]
 for (c,a,q),totals in sorted(sums.items()):
  n=counts[c,a,q];assert n==720;curves.append({'condition':c,'arm':a,'questions':q,'rows':n,**{k:sum(v)/n for k,v in totals.items()}})
 result={'estimates':estimates,'conditions':curves,'streams':30,'finite_orders':24,'paired_scenarios':720,'endpoint_rows':sum(counts.values()),'query_traces_expected':46080,'primary_family':list(PRIMARY),'bootstrap':'paired30streamblocks;10000 common resamples;linear interpolated percentile;3primary Bonferroni intervals98.333percent','all_settings_fixed':True,'model_calls':0,'GPU_allocations':0}
 (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(root/'paired_effects.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in raw));(root/'analysis_receipt.json').write_text(json.dumps({'python_version':sys.version,'elapsed_seconds':time.time()-started,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_hashes':source_hashes,'outputs':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('summary.json','paired_effects.jsonl')},'model_calls':0,'GPU_allocations':0},indent=2)+'\n');print('complete30stream factorial analysis')
if __name__=='__main__':run(Path(sys.argv[1]))
