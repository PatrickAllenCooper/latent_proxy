"""Unknown per-user noise sidecar; fixed AIF core/belief updater, no leakage."""
import hashlib,json,math,os,random,resource,signal,sys,time
from pathlib import Path
import rank_noise_attribution_v79 as core
GRID=tuple(k/20 for k in range(9));PRIOR=tuple(math.comb(8,k)*(.25**k)*(.75**(8-k)) for k in range(9));MODES=('fixed10','online','oracle');NOISES=(.1,.2,.3);BUDGETS=(0,2,4,6,8)
def noise_prior():return [[p/24]*24 for p in PRIOR]
def noise_update(joint,pair,answer):
 v=[[p*((1-eps) if yes==answer else eps) for p,yes in zip(row,core.YES[pair])] for eps,row in zip(GRID,joint)];z=sum(sum(row) for row in v)
 return [[p/z for p in row] for row in v]
def noise_state(joint):
 marginal=[sum(row) for row in joint];mean=round(sum(eps*p for eps,p in zip(GRID,marginal)),12)
 def quantile(q):
  total=0
  for eps,p in zip(GRID,marginal):
   total+=p
   if total>=q:return eps
  return GRID[-1]
 return mean,marginal,[quantile(.05),quantile(.95)]
def planner_noise(mode,joint,true_noise=None):
 if mode=='fixed10':return .1
 if mode=='online':return noise_state(joint)[0]
 if mode=='oracle':return true_noise
 raise ValueError('mode')
def protocol(seeds=None,users=24):
 return {'version':'v82','seeds':list(range(60101,60131)) if seeds is None else seeds,'users':users,'modes':list(MODES),'true_noise_rates':list(NOISES),'AIF_weight':.1,'main_belief_update_noise':.1,'main_hazard':.1,'core':'frozen v79 selection/action/update; native Python3.12 ties unchanged','question_budgets':list(BUDGETS),'online_estimator':'sidecar exact joint posterior over24staticunknown preference orders and9noise rates; only past pair/answer input; independent reset per trajectory','noise_grid':list(GRID),'noise_prior':list(PRIOR),'noise_prior_mean':.1,'prior_construction':'epsilon=.05*K,K~Binomial(8,.25); fixed before outputs,no tuning','planner_estimate':'posterior predictive mean rounded12decimals,used before next answer; no truth-order/flip/noise input','online_scope':'certainty-equivalent plug-in mean; existing questions only,no added noise-identification objective or extra calibration probes','shadow_estimator':'recorded in all modes for diagnosis; only online mode uses its mean for planning','sidecar_assumptions':'static latent order and constant symmetric response noise,correct for these environments; does not modify main AIF belief state','common_random_numbers':'potential uniform fixed by fresh seed/user/question/pair across modes/rates; thresholded at actualnoise for response generator only','primary_endpoint':'transfer normalized regret at8questions,true20percentnoise','primary_contrasts':['online-minus-fixed10','online-minus-oracle'],'primary_interval':'30pairedstreamblocks averaging24orders;10000bootstrapresamples;two-primary Bonferroni97.5percent intervals','secondary':['10/30percentnoise robustness','target regret','noise estimate bias/MAE and90percentcredible-interval coverage at every budget','estimated noise before each query','oracle-minus-fixed10'],'interpretation':'interval crossing0 is not oracle equivalence; oracle knows noise only,not preferences,and is a reference rather than guaranteed optimal transfer policy','stopping':'fixed30freshstreams;three sequential10streamshards;no tuning or optional stopping','resources':{'cpus':1,'memory_MiB':256,'scheduler_wall_seconds':60,'application_wall_seconds':30,'check_seconds':20,'gpus':0},'artifact_cap_per_shard':33554432,'scope':'finite synthetic known-static users;no LLM/RL/human evidence','model_calls':0}
def simulate(seed,user,true_noise,mode):
 truth=core.ORDERS[user];rng=random.Random(seed*100+user);attrs=list(core.ATTR);rng.shuffle(attrs);excluded=rng.randrange(4);menus=[core.utilities(attrs,e) for e in range(4)];_,u,r=menus[excluded];w=core.UNIFORM[:];joint=noise_prior();questions=random.Random(seed*100+user);rows=[];traces=[]
 for q in range(9):
  mean,marginal,interval=noise_state(joint)
  if q in BUDGETS:
   actions=['ABCD'[eligible[core.action(w,uu)]] for eligible,uu,_ in menus];target=r[user][core.action(w,u)];transfer=sum(rr[user][core.action(w,uu)] for e,(_,uu,rr) in enumerate(menus) if e!=excluded)/3
   rows.append({'seed':seed,'user_id':user,'true_noise':true_noise,'mode':mode,'questions':q,'target_action':actions[excluded],'actions_by_excluded_option':actions,'target_regret':target,'transfer_regret':transfer,'correct':target==0,'entropy':core.entropy(w),'true_order_mass':w[user],'estimated_noise':mean,'noise_bias':mean-true_noise,'noise_absolute_error':abs(mean-true_noise),'noise_interval90':interval,'noise_interval_covers_truth':interval[0]<=true_noise<=interval[1],'noise_marginal':marginal,'noise_estimator_used_for_planning':mode=='online'})
  if q==8:break
  w=core.predict(w,.1);eps=planner_noise(mode,joint,true_noise);pair=core.select('AIF',w,u,r,questions,eps);a,b=core.PAIRS[pair];latent=truth.index(a)<truth.index(b);flip=random.Random(f'{seed}:{user}:{q+1}:{pair}').random()<true_noise;answer=latent!=flip;w=core.update(w,pair,answer,.1);joint=noise_update(joint,pair,answer);after,post,_=noise_state(joint)
  assert abs(sum(w)-1)<1e-10 and abs(sum(post)-1)<1e-10
  traces.append({'seed':seed,'user_id':user,'true_noise':true_noise,'mode':mode,'question':q+1,'pair_index':pair,'latent_answer':latent,'flipped':flip,'answer':answer,'planner_noise_before_query':eps,'estimated_noise_before_query':mean,'estimated_noise_after_answer':after,'noise_marginal_after_answer':post,'posterior_weights':w})
 return rows,traces

def run(out,d):
 started=time.time();events=[];audit={'protocol_opens':0}
 def hook(event,args):
  if event=='open' and isinstance(args[0],str) and args[0]==str(out/'protocol.json'):audit['protocol_opens']+=1
 sys.addaudithook(hook);out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(d,indent=2)+'\n');assert json.loads((out/'protocol.json').read_text())==d;events.append({'phase':'protocol_frozen','at':time.time()});rows=[];traces=[]
 for seed in d['seeds']:
  for user in range(d['users']):
   for noise in NOISES:
    for mode in MODES:
     rr,tt=simulate(seed,user,noise,mode);rows.extend(rr);traces.extend(tt)
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
