"""Two bounded sequential shards; unchanged v79 kernels, full noise2x2."""
import hashlib,json,math,os,random,resource,signal,sys,time
from pathlib import Path
import rank_noise_attribution_v79 as core
ARMS=('EIG','AIF');CONFIGS={'update10_plan10':(.1,.1),'update20_plan10':(.2,.1),'update10_plan20':(.1,.2),'update20_plan20':(.2,.2)};BUDGETS=(0,2,4,6,8)
def protocol(seeds=None,users=24):
 return {'version':'v81','seeds':list(range(40101,40131)) if seeds is None else seeds,'users':users,'arms':list(ARMS),'configurations':{k:{'update_noise':v[0],'planner_noise':v[1]} for k,v in CONFIGS.items()},'true_flip_probability':.2,'true_preferences':'static','hazard':.1,'AIF_weight':.1,'question_budgets':list(BUDGETS),'kernel':'frozen v79, unchanged native Python3.12 sum/argmax','primary_outcome':'AIF transfer normalized regret at8questions','primary_contrasts':['average_update_effect','average_planner_effect','interaction'],'primary_interval':'30pairedstreamblocks each averaging24orders;10000bootstrap resamples;Bonferroni98.333percent intervals for3primary contrasts,nominal95percent intervals also reported','secondary':['conditional update/planner effects at each other level','joint correction','EIG effects','AIF-minus-EIG gap effects','final target and transfer curves'],'interpretation':'negative mean reduces regret;interaction identifies dependence,not a unique causal mediation;closed-loop updater changes also change future queries','scope':'finite24order synthetic domain,not human/LLM/RL evidence','stopping':'fixed30freshstreams;2sequential15streamshards;no tuning or significance-based stopping','resources':{'cpus':1,'memory_MiB':256,'scheduler_wall_seconds':60,'application_wall_seconds':30,'check_seconds':20,'gpus':0},'artifact_cap_per_shard':33554432,'model_calls':0}
def simulate(seed,user,arm,config):
 truth=core.ORDERS[user];rng=random.Random(seed*100+user);attrs=list(core.ATTR);rng.shuffle(attrs);excluded=rng.randrange(4);menus=[core.utilities(attrs,e) for e in range(4)];_,u,r=menus[excluded];update_eps,plan_eps=CONFIGS[config];w=core.UNIFORM[:];questions=random.Random(seed*100+user);rows=[];traces=[]
 for q in range(9):
  if q in BUDGETS:
   actions=['ABCD'[eligible[core.action(w,uu)]] for eligible,uu,_ in menus];target=r[user][core.action(w,u)];transfer=sum(rr[user][core.action(w,uu)] for e,(_,uu,rr) in enumerate(menus) if e!=excluded)/3
   rows.append({'seed':seed,'user_id':user,'arm':arm,'condition':config,'questions':q,'target_action':actions[excluded],'actions_by_excluded_option':actions,'target_regret':target,'transfer_regret':transfer,'correct':target==0,'entropy':core.entropy(w),'true_order_mass':w[user]})
  if q==8:break
  w=core.predict(w,.1);pair=core.select(arm,w,u,r,questions,plan_eps);a,b=core.PAIRS[pair];latent=truth.index(a)<truth.index(b);flip=random.Random(f'{seed}:{user}:{q+1}:{pair}').random()<.2;answer=latent!=flip;w=core.update(w,pair,answer,update_eps)
  assert all(math.isfinite(p) and p>0 for p in w) and abs(sum(w)-1)<1e-10
  traces.append({'seed':seed,'user_id':user,'arm':arm,'condition':config,'question':q+1,'pair_index':pair,'latent_answer':latent,'flipped':flip,'answer':answer,'posterior_weights':w})
 return rows,traces

def run(out,d):
 started=time.time();events=[];audit={'protocol_opens':0}
 def hook(event,args):
  if event=='open' and isinstance(args[0],str) and args[0]==str(out/'protocol.json'):audit['protocol_opens']+=1
 sys.addaudithook(hook);out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(d,indent=2)+'\n');assert json.loads((out/'protocol.json').read_text())==d;events.append({'phase':'protocol_frozen','at':time.time()});rows=[];traces=[]
 for seed in d['seeds']:
  for user in range(d['users']):
   for c in CONFIGS:
    for a in ARMS:
     rr,tt=simulate(seed,user,a,c);rows.extend(rr);traces.extend(tt)
 events.append({'phase':'simulation_complete','at':time.time()})
 for name,data in [('per_user',rows),('question_traces',traces)]:
  with (out/(name+'.jsonl')).open('x') as f:
   for r in data:f.write(json.dumps(r,sort_keys=True)+'\n')
 events.append({'phase':'artifacts_written','at':time.time()});assert sum(p.stat().st_size for p in out.iterdir())<33554432
 receipt={'method':'application_phase_and_Python_audit','native_syscalls_observed':False,'python_version':sys.version,'events':events,'audit':audit,'elapsed_seconds':time.time()-started,'peakRSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform=='darwin' else 1),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'kernel_sha256':hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},'records':len(rows),'trace_records':len(traces),'model_calls':0,'GPU_allocations':0};(out/'application_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'elapsed':receipt['elapsed_seconds'],'rows':len(rows),'peakRSS_KiB':receipt['peakRSS_KiB']}))
if __name__=='__main__':
 if sys.platform.startswith('linux'):resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456))
 signal.signal(signal.SIGALRM,lambda *_:sys.exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,min(20,float(os.environ.get('STUDY_WALL_START',time.time()))+20-time.time())))
 run(Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text()))
