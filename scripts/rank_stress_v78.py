"""CPU-only noisy/drifting finite-order discovery; no model/framework/ptrace."""
import hashlib,itertools,json,math,os,random,resource,signal,sys,time
from pathlib import Path
from preference_use_cycle_v70 import ATTR
ORDERS=list(itertools.permutations(ATTR));PAIRS=list(itertools.combinations(ATTR,2));ARMS=('random','EIG','decision_focused','AIF')
YES=[[p.index(a)<p.index(b) for p in ORDERS] for a,b in PAIRS]
CONDITIONS={'clean':(0.,False),'noise10':(.1,False),'noise20':(.2,False),'change_after4':(0.,True)}
BUDGETS=(0,2,4,6,8);UNIFORM=[1/24]*24

def entropy(w):return -sum(p*math.log(p) for p in w if p>0)
def update(w,pair,answer,epsilon=.1):
 v=[p*((1-epsilon) if yes==answer else epsilon) for p,yes in zip(w,YES[pair])];z=sum(v)
 return [p/z for p in v]
def predict(w,hazard):return [(1-hazard)*p+hazard/24 for p in w]
def utilities(attrs,excluded):
 eligible=[i for i in range(4) if i!=excluded];utility=[[3-p.index(attrs[i]) for i in eligible] for p in ORDERS]
 regrets=[[(max(row)-x)/(max(row)+1) for x in row] for row in utility]
 return eligible,utility,regrets
def action(w,utility):return max(range(3),key=lambda a:sum(p*row[a] for p,row in zip(w,utility)))
def risk(w,utility,regrets):
 a=action(w,utility);return sum(p*row[a] for p,row in zip(w,regrets))
def select(arm,w,utility,regrets,rng):
 if arm=='random':return rng.randrange(6)
 h=entropy(w);base=risk(w,utility,regrets);values=[]
 for pair in range(6):
  yes=sum(p*(.9 if y else .1) for p,y in zip(w,YES[pair]));posts=[update(w,pair,b) for b in (False,True)];probs=[1-yes,yes]
  eig=h-sum(pr*entropy(post) for pr,post in zip(probs,posts))
  gain=base-sum(pr*risk(post,utility,regrets) for pr,post in zip(probs,posts))
  values.append(eig if arm=='EIG' else gain if arm=='decision_focused' else gain+.1*eig/math.log(24))
 return max(range(6),key=lambda i:values[i])
def protocol(smoke=False):
 return {'version':'v78','seeds':[4101] if smoke else [4101,5101,6101],'users':2 if smoke else 24,'arms':list(ARMS),'conditions':{k:{'answer_flip_probability':v[0],'reverse_order_before_question5':v[1]} for k,v in CONDITIONS.items()},'filters':{'static':0.,'hazard10':.1},'inference_answer_flip_probability':.1,'AIF_weight':.1,'AIF_scope':'operational expected decision-regret reduction plus normalized information gain; not full generative AIF','question_budgets':list(BUDGETS),'changing_preferences':'only change condition; no answer noise there; reversal after fourth answer, hidden from inference','common_random_numbers':'potential answer noise fixed by seed/user/question/pair; menus and true trajectories common across arms/filters','primary_endpoint':'transfer normalized regret at8questions; paired AIF-minus-decision_focused and AIF-minus-EIG in every condition/filter','secondary':['target_regret','correct','entropy','true_order_mass','question_count'],'intervals':'95% percentile resampling across24preference-order clusters, seed repetitions averaged within cluster; exploratory finite-order uncertainty, not human population inference','qualification':'synthetic finite preference discovery only; exact executor, no LLM/tool assimilation/RL training','resource_limits':{'CPU':1,'memory_MiB':256,'scheduler_wall_seconds':60,'application_wall_seconds':30,'check_seconds':20,'GPU':0},'model_calls':0,'smoke':smoke}
def run(out,design):
 started=time.time();events=[];audit={'protocol_opens':0}
 def hook(event,args):
  if event=='open' and isinstance(args[0],str) and args[0]==str(out/'protocol.json'):audit['protocol_opens']+=1
 sys.addaudithook(hook)
 out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(design,indent=2)+'\n')
 registered=json.loads((out/'protocol.json').read_text());assert registered==design
 events.append({'phase':'protocol_frozen','at':time.time()});rows=[];traces=[]
 for seed in design['seeds']:
  for user in range(design['users']):
   truth=ORDERS[user];rng=random.Random(seed*100+user);attrs=list(ATTR);rng.shuffle(attrs);excluded=rng.randrange(4);menus=[utilities(attrs,e) for e in range(4)];_,utility,regrets=menus[excluded]
   flips={(q,p):random.Random(f'{seed}:{user}:{q}:{p}').random() for q in range(1,9) for p in range(6)}
   for condition,(noise,change) in CONDITIONS.items():
    for filt,hazard in design['filters'].items():
     for arm in ARMS:
      w=UNIFORM[:];questions=random.Random(seed*100+user)
      for q in range(9):
       current=truth if not change or q<=4 else tuple(reversed(truth));truth_index=ORDERS.index(current)
       if q in BUDGETS:
        a=action(w,utility);transfer=[r[truth_index][action(w,u)] for e,(_,u,r) in enumerate(menus) if e!=excluded]
        rows.append({'seed':seed,'user_id':user,'condition':condition,'filter':filt,'arm':arm,'questions':q,'true_order_index':truth_index,'target_regret':regrets[truth_index][a],'transfer_regret':sum(transfer)/3,'correct':regrets[truth_index][a]==0,'entropy':entropy(w),'true_order_mass':w[truth_index]})
       if q==8:break
       qnext=q+1;current=truth if not change or qnext<=4 else tuple(reversed(truth));w=predict(w,hazard);pair=select(arm,w,utility,regrets,questions);a,b=PAIRS[pair];latent=current.index(a)<current.index(b);flipped=flips[qnext,pair]<noise;answer=latent!=flipped;w=update(w,pair,answer)
       assert all(math.isfinite(p) and p>0 for p in w) and abs(sum(w)-1)<1e-10
       traces.append({'seed':seed,'user_id':user,'condition':condition,'filter':filt,'arm':arm,'question':qnext,'pair_index':pair,'latent_answer':latent,'flipped':flipped,'answer':answer,'posterior_true_mass':w[ORDERS.index(current)]})
 events.append({'phase':'simulation_complete','at':time.time()})
 for name,data in [('per_user',rows),('question_traces',traces)]:
  with (out/(name+'.jsonl')).open('x') as f:
   for r in data:f.write(json.dumps(r,sort_keys=True)+'\n')
 aggregate=[];contrasts=[]
 for condition in CONDITIONS:
  for filt in design['filters']:
   for arm in ARMS:
    for q in BUDGETS:
     selected=[r for r in rows if (r['condition'],r['filter'],r['arm'],r['questions'])==(condition,filt,arm,q)]
     aggregate.append({'condition':condition,'filter':filt,'arm':arm,'questions':q,'rows':len(selected),**{k:sum(r[k] for r in selected)/len(selected) for k in ('target_regret','transfer_regret','correct','entropy','true_order_mass')}})
   for comparator in ('decision_focused','EIG'):
    delta=[]
    for user in range(design['users']):
     by_arm={a:[r['transfer_regret'] for r in rows if (r['condition'],r['filter'],r['arm'],r['questions'],r['user_id'])==(condition,filt,a,8,user)] for a in ('AIF',comparator)}
     delta.append(sum(by_arm['AIF'])/len(by_arm['AIF'])-sum(by_arm[comparator])/len(by_arm[comparator]))
    boot=random.Random('v78:'+condition+filt+comparator);samples=sorted(sum(boot.choice(delta) for _ in delta)/len(delta) for _ in range(1000))
    contrasts.append({'condition':condition,'filter':filt,'contrast':'AIF-minus-'+comparator,'endpoint':'transfer_regret_at8','mean':sum(delta)/len(delta),'order_cluster_interval95':[samples[24],samples[974]],'order_clusters':len(delta),'seed_repeats':len(design['seeds']),'lower_is_better':True})
 events.append({'phase':'analysis_complete','at':time.time()})
 summary={'conditions':aggregate,'primary_contrasts':contrasts,'records':len(rows),'trace_records':len(traces),'unique_orders':design['users'],'seed_repeats':len(design['seeds']),'elapsed_seconds':time.time()-started,'model_calls':0,'GPU_allocations':0}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 receipt={'method':'application_phase_and_Python_audit','native_syscalls_observed':False,'events':events,'audit':audit,'elapsed_seconds':time.time()-started,'peakRSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024 if sys.platform=='darwin' else 1),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},'model_calls':0,'GPU_allocations':0}
 (out/'application_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'elapsed':receipt['elapsed_seconds'],'rows':len(rows),'peakRSS_native_units':receipt['peakRSS_KiB']}))
if __name__=='__main__':
 if sys.platform.startswith('linux'):resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,256*1024*1024))
 signal.signal(signal.SIGALRM,lambda *_:sys.exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,min(20,float(os.environ.get('STUDY_WALL_START',time.time()))+20-time.time())))
 run(Path(sys.argv[1]),json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv)>3 and sys.argv[2]=='--protocol' else protocol('--smoke' in sys.argv))
