"""Posthoc shared-history planner diagnostic; no new response streams or tuning."""
import hashlib,json,math,random,sys
from pathlib import Path
import rank_noise_attribution_v79 as c

def values(arm,w,u,r,noise):
 h=c.entropy(w);base=c.risk(w,u,r);out=[]
 for pair in range(6):
  prob=sum(p*((1-noise) if yes else noise) for p,yes in zip(w,c.YES[pair]));posts=[c.update(w,pair,b,noise) for b in (False,True)];ps=[1-prob,prob];eig=h-sum(p*c.entropy(v) for p,v in zip(ps,posts));gain=base-sum(p*c.risk(v,u,r) for p,v in zip(ps,posts));out.append(eig if arm=='EIG' else gain+.1*eig/math.log(24))
 return out

def main():
 root=Path('outputs/preference_program/results/rank_noise_components_v81');paths={}
 for shard in (0,1):
  for line in (root/f'shard{shard}/results/question_traces.jsonl').open():
   r=json.loads(line);k=(r['seed'],r['user_id'],r['condition'],r['arm']);paths.setdefault(k,[]).append((r['pair_index'],r['answer']))
 rows=[]
 for arm in ('AIF','EIG'):
  for eps,pair in [(.1,('update10_plan10','update10_plan20')),(.2,('update20_plan10','update20_plan20'))]:
   changed=strict=ties=slots=0;margins=[]
   for seed in range(40101,40131):
    for user in range(24):
     a=paths[seed,user,pair[0],arm];b=paths[seed,user,pair[1],arm];slots+=sum(x[0]==y[0] for x,y in zip(a,b));w=c.UNIFORM[:]
     for q,((old,answer),(new,other_answer)) in enumerate(zip(a,b),1):
      prior=c.predict(w,.1)
      if old!=new:
       changed+=1;rng=random.Random(seed*100+user);attrs=list(c.ATTR);rng.shuffle(attrs);ex=rng.randrange(4);_,u,r=c.utilities(attrs,ex);v0=values(arm,prior,u,r,.1);v1=values(arm,prior,u,r,.2);m0=v0[old]-v0[new];m1=v1[new]-v1[old];assert m0>-1e-10 and m1>-1e-10
       if min(m0,m1)>1e-12:strict+=1
       else:ties+=1
       margins.append({'seed':seed,'user_id':user,'question':q,'old_pair':old,'new_pair':new,'old_margin':m0,'new_margin':m1});break
      assert answer==other_answer;w=c.update(prior,old,answer,eps)
   rows.append({'arm':arm,'update_noise':eps,'paired_scenarios':720,'query_slot_agreement':slots/(720*8),'divergent_trajectories':changed,'first_divergence_strict_ranking_reversal':strict,'first_divergence_near_tie_in_one_or_both_models':ties,'threshold':1e-12,'margins':margins})
 result={'status':'posthoc diagnostic only; no new inference or tuning','scope':'paired planner assumptions at identical observed history until first query divergence; later histories may differ','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':rows,'model_calls':0,'GPU_allocations':0};(root/'query_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='margins'} for r in rows],indent=2))
if __name__=='__main__':main()
