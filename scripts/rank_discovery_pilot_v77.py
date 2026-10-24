"""Finite synthetic-user discovery pilot; no LLM, RL training or framework imports."""
import hashlib,itertools,json,math,random,time
from pathlib import Path
from preference_use_cycle_v70 import ATTR,oracle,score
ORDERS=list(itertools.permutations(ATTR));PAIRS=list(itertools.combinations(ATTR,2))
ARMS=('random','EIG','decision_focused','AIF')
def entropy(n):return math.log(n) if n else 0.
def action(posterior,case):
 eligible=[o for o in case['options'] if o['eligible']]
 return max(eligible,key=lambda o:sum(3-order.index(o['attribute']) for order in posterior))['label']
def risk(posterior,case):
 chosen=action(posterior,case)
 return sum(score(dict(case,priority=list(order)),chosen)['normalized_regret'] for order in posterior)/len(posterior)
def groups(posterior,pair):
 a,b=pair;return [[p for p in posterior if (p.index(a)<p.index(b))==answer] for answer in (False,True)]
def select(arm,posterior,case,rng):
 if arm=='random':return rng.choice(PAIRS)
 def value(pair):
  split=groups(posterior,pair);weights=[len(g)/len(posterior) for g in split]
  eig=entropy(len(posterior))-sum(w*entropy(len(g)) for w,g in zip(weights,split))
  gain=risk(posterior,case)-sum(w*risk(g,case) for w,g in zip(weights,split) if g)
  return eig if arm=='EIG' else gain if arm=='decision_focused' else gain+.1*eig/math.log(24)
 return max(PAIRS,key=value)
def run(out):
 started=time.time();out.mkdir(parents=True,exist_ok=False)
 protocol={'version':'v77','registered_before_outputs':True,'arms':list(ARMS),'seeds':[1001,2001,3001],'question_budgets':[0,1,2,4,8],'unique_latent_orders':24,'users':'same24 fully enumerated orders in each seed; not72 independent users','answer_model':'deterministic truthful pairwise preference','prior':'uniform over24 total orders','AIF':'expected terminal normalized-regret reduction +0.1*EIG/log(24); operational active-inference mixture, not a full generative AIF agent','selection_ties':'lexicographic pair order','random':'with replacement; all arms share initial prior/menu/answers and maximum8questions','deployment':'exact posterior expected reward action; no LLM reader','evaluation':['targeted final decision','transfer to other three excluded-option menus'],'intervals':'none; finite enumerated diagnostic, not a population sample','model_calls':0,'GPU_allocations':0,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');rows=[];traces=[]
 for seed in protocol['seeds']:
  for u,truth in enumerate(ORDERS):
   menu_rng=random.Random(seed*100+u);attrs=list(ATTR);menu_rng.shuffle(attrs);excluded=menu_rng.randrange(4)
   case={'priority':list(truth),'options':[{'label':l,'attribute':a,'eligible':i!=excluded} for i,(l,a) in enumerate(zip('ABCD',attrs))]}
   for arm in ARMS:
    posterior=ORDERS[:];rng=random.Random(seed*100+u)
    for q in range(9):
     if q in protocol['question_budgets']:
      chosen=action(posterior,case);transfer=[]
      for other in range(4):
       if other==excluded:continue
       c=dict(case,options=[dict(o,eligible=i!=other) for i,o in enumerate(case['options'])]);transfer.append(score(c,action(posterior,c))['normalized_regret'])
      rows.append({'seed':seed,'user_id':u,'arm':arm,'questions':q,'posterior_orders':len(posterior),'posterior_true_mass':1/len(posterior),'posterior_entropy':entropy(len(posterior)),'correct':chosen==oracle(case),'normalized_regret':score(case,chosen)['normalized_regret'],'transfer_regret':sum(transfer)/3})
     if q==8:break
     pair=select(arm,posterior,case,rng);answer=truth.index(pair[0])<truth.index(pair[1]);before=len(posterior);posterior=[p for p in posterior if (p.index(pair[0])<p.index(pair[1]))==answer];assert truth in posterior
     traces.append({'seed':seed,'user_id':u,'arm':arm,'question':q+1,'pair':list(pair),'answer':answer,'prior_orders':before,'posterior_orders':len(posterior)})
 (out/'per_user.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows));(out/'question_traces.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in traces))
 summary=[]
 for arm in ARMS:
  for q in protocol['question_budgets']:
   selected=[r for r in rows if r['arm']==arm and r['questions']==q]
   summary.append({'arm':arm,'questions':q,'rows':len(selected),**{key:sum(r[key] for r in selected)/len(selected) for key in ('correct','normalized_regret','transfer_regret','posterior_entropy')}})
 (out/'summary.json').write_text(json.dumps({'conditions':summary,'elapsed_seconds':time.time()-started,'records':len(rows),'question_trace_records':len(traces),'model_calls':0,'GPU_allocations':0},indent=2)+'\n')
 return summary
if __name__=='__main__':
 run(Path('outputs/preference_program/results/rank_discovery_pilot_v77'))
