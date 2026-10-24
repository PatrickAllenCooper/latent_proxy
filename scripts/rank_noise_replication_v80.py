"""Fixed30-stream focused replication; unchanged v79 kernels/settings."""
import hashlib,json,math,os,random,resource,signal,sys,time
from pathlib import Path
import rank_noise_attribution_v79 as core
ARMS=('EIG','AIF');CONFIGS={'update10_plan10':.1,'update20_plan20':.2};BUDGETS=(0,2,4,6,8)
def protocol(smoke=False):
 return {'version':'v80','seeds':[20101] if smoke else list(range(20101,20131)),'users':2 if smoke else 24,'arms':list(ARMS),'configurations':{k:{'update_noise':v,'planner_noise':v} for k,v in CONFIGS.items()},'true_flip_probability':.2,'true_preferences':'static','hazard':.1,'AIF_weight':.1,'question_budgets':list(BUDGETS),'kernel':'unchanged frozen v79 select/update/predict/utilities/action/risk; pinned Python3.12.13 summation and argmax ties','primary_estimand':'mean over uniform finite24orders and fresh menu/noise streams of joint-matching change in AIF-minus-EIG transfer-regret gap at8questions','primary_contrast':'(AIF20-EIG20)-(AIF10-EIG10); negative means correction narrows deficit','primary_interval':'95% percentile paired stream-block bootstrap,10000resamples; each independent seed block averages all24enumerated orders','secondary_intervals':'95% preference-order cluster bootstrap,10000resamples after averaging streams; retained for v79 comparison, not human-population inference','secondary':['AIF-minus-EIG before/after correction','AIF absolute correction','EIG absolute correction','target regret','all question-budget curves'],'stopping':'fixed30streams; no significance-based stopping or tuning; only resource watchdog can terminate','interpretation':'primary interval below0 supports reduction within this benchmark; crossing0 inconclusive; matched gap crossing0 is not equivalence','scope':'focused replication of primary hazard10 regime; not a new full factorial, not LLM/RL/human evidence','resources':{'cpus':1,'memory_MiB':256,'scheduler_wall_seconds':60,'application_wall_seconds':30,'check_seconds':20,'gpus':0},'artifact_cap_bytes':33554432,'model_calls':0,'smoke':smoke}
def simulate(seed,user,arm,config):
 truth=core.ORDERS[user];rng=random.Random(seed*100+user);attrs=list(core.ATTR);rng.shuffle(attrs);excluded=rng.randrange(4);menus=[core.utilities(attrs,e) for e in range(4)];_,u,r=menus[excluded];eps=CONFIGS[config];w=core.UNIFORM[:];questions=random.Random(seed*100+user);rows=[];traces=[]
 for q in range(9):
  if q in BUDGETS:
   actions=['ABCD'[eligible[core.action(w,uu)]] for eligible,uu,_ in menus];target=r[user][core.action(w,u)];transfer=sum(rr[user][core.action(w,uu)] for e,(_,uu,rr) in enumerate(menus) if e!=excluded)/3
   rows.append({'seed':seed,'user_id':user,'arm':arm,'condition':config,'questions':q,'target_action':actions[excluded],'actions_by_excluded_option':actions,'target_regret':target,'transfer_regret':transfer,'correct':target==0,'entropy':core.entropy(w),'true_order_mass':w[user]})
  if q==8:break
  w=core.predict(w,.1);pair=core.select(arm,w,u,r,questions,eps);a,b=core.PAIRS[pair];latent=truth.index(a)<truth.index(b);flip=random.Random(f'{seed}:{user}:{q+1}:{pair}').random()<.2;answer=latent!=flip;w=core.update(w,pair,answer,eps)
  assert all(math.isfinite(p) and p>0 for p in w) and abs(sum(w)-1)<1e-10
  traces.append({'seed':seed,'user_id':user,'arm':arm,'condition':config,'question':q+1,'pair_index':pair,'latent_answer':latent,'flipped':flip,'answer':answer,'posterior_weights':w})
 return rows,traces

def intervals(values,key):
 rng=random.Random('v80:'+key);n=len(values);boots=sorted(sum(rng.choice(values) for _ in values)/n for _ in range(10000));return [boots[249],boots[9749]]
def run(out,d):
 started=time.time();events=[];audit={'protocol_opens':0}
 def hook(event,args):
  if event=='open' and isinstance(args[0],str) and args[0]==str(out/'protocol.json'):audit['protocol_opens']+=1
 sys.addaudithook(hook);out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(d,indent=2)+'\n');assert json.loads((out/'protocol.json').read_text())==d;events.append({'phase':'protocol_frozen','at':time.time()});rows=[];traces=[]
 for seed in d['seeds']:
  for user in range(d['users']):
   for config in CONFIGS:
    for arm in ARMS:
     a,b=simulate(seed,user,arm,config);rows.extend(a);traces.extend(b)
 events.append({'phase':'simulation_complete','at':time.time()})
 for name,data in [('per_user',rows),('question_traces',traces)]:
  with (out/(name+'.jsonl')).open('x') as f:
   for r in data:f.write(json.dumps(r,sort_keys=True)+'\n')
 lookup={(r['seed'],r['user_id'],r['condition'],r['arm']):r for r in rows if r['questions']==8};get=lambda seed,user,c,a:lookup[seed,user,c,a]['transfer_regret']
 def effects(seed,user):
  a0=get(seed,user,'update10_plan10','AIF');e0=get(seed,user,'update10_plan10','EIG');a1=get(seed,user,'update20_plan20','AIF');e1=get(seed,user,'update20_plan20','EIG')
  return {'joint_gap_correction':(a1-e1)-(a0-e0),'baseline_AIF_minus_EIG':a0-e0,'matched_AIF_minus_EIG':a1-e1,'AIF_absolute_correction':a1-a0,'EIG_absolute_correction':e1-e0}
 raw=[dict(seed=seed,user_id=user,**effects(seed,user)) for seed in d['seeds'] for user in range(d['users'])];estimates=[]
 for name in ('joint_gap_correction','baseline_AIF_minus_EIG','matched_AIF_minus_EIG','AIF_absolute_correction','EIG_absolute_correction'):
  by_seed=[sum(r[name] for r in raw if r['seed']==seed)/d['users'] for seed in d['seeds']];by_order=[sum(r[name] for r in raw if r['user_id']==user)/len(d['seeds']) for user in range(d['users'])]
  estimates.append({'contrast':name,'mean':sum(by_seed)/len(by_seed),'stream_block_interval95':intervals(by_seed,'stream:'+name),'order_cluster_interval95':intervals(by_order,'order:'+name),'independent_stream_blocks':len(by_seed),'enumerated_orders':len(by_order),'per_stream_means':by_seed,'per_order_means':by_order})
 curves=[]
 for c in CONFIGS:
  for a in ARMS:
   for q in BUDGETS:
    selected=[r for r in rows if (r['condition'],r['arm'],r['questions'])==(c,a,q)];curves.append({'condition':c,'arm':a,'questions':q,'rows':len(selected),**{k:sum(r[k] for r in selected)/len(selected) for k in ('target_regret','transfer_regret','correct','entropy','true_order_mass')}})
 events.append({'phase':'analysis_complete','at':time.time()});summary={'estimates':estimates,'conditions':curves,'records':len(rows),'trace_records':len(traces),'unique_orders':d['users'],'fresh_streams':len(d['seeds']),'paired_scenarios':d['users']*len(d['seeds']),'elapsed_seconds':time.time()-started,'model_calls':0,'GPU_allocations':0};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'paired_effects.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in raw));assert sum(p.stat().st_size for p in out.iterdir())<33554432
 receipt={'method':'application_phase_and_Python_audit','native_syscalls_observed':False,'python_version':sys.version,'events':events,'audit':audit,'elapsed_seconds':time.time()-started,'peakRSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform=='darwin' else 1),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'kernel_sha256':hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},'model_calls':0,'GPU_allocations':0};(out/'application_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'elapsed':receipt['elapsed_seconds'],'rows':len(rows),'peakRSS_KiB':receipt['peakRSS_KiB']}))
if __name__=='__main__':
 if sys.platform.startswith('linux'):resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456))
 signal.signal(signal.SIGALRM,lambda *_:sys.exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,min(20,float(os.environ.get('STUDY_WALL_START',time.time()))+20-time.time())))
 run(Path(sys.argv[1]),json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv)>3 and sys.argv[2]=='--protocol' else protocol('--smoke' in sys.argv))
