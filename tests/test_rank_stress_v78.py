import json,math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'));import rank_stress_v78 as p
class Stress(unittest.TestCase):
 def test_soft_likelihood_and_hazard(self):
  w=p.update(p.UNIFORM,0,True);self.assertAlmostEqual(sum(w),1);self.assertTrue(all(x>0 for x in w));self.assertAlmostEqual(sum(v for v,y in zip(w,p.YES[0]) if y),.9)
  v=p.predict(w,.1);self.assertAlmostEqual(sum(v),1);self.assertGreater(min(v),min(w))
 def test_exact_oracle_and_information(self):
  _,u,r=p.utilities(p.ATTR,0);w=[float(i==0) for i in range(24)];self.assertEqual(p.action(w,u),0);self.assertEqual(p.risk(w,u,r),0);self.assertAlmostEqual(p.entropy(p.UNIFORM),math.log(24))
 def test_smoke_paired_baselines_and_transition(self):
  rows=[json.loads(x) for x in Path('outputs/preference_program/results/rank_stress_v78_local_smoke/per_user.jsonl').read_text().splitlines()]
  for user in range(2):
   baseline=[x for x in rows if x['user_id']==user and x['questions']==0];self.assertEqual(len({x['target_regret'] for x in baseline}),1)
   for arm in p.ARMS:
    for filt in ('static','hazard10'):
     clean=next(x for x in rows if (x['user_id'],x['arm'],x['filter'],x['condition'],x['questions'])==(user,arm,filt,'clean',4));changed=next(x for x in rows if (x['user_id'],x['arm'],x['filter'],x['condition'],x['questions'])==(user,arm,filt,'change_after4',4));self.assertEqual(clean['target_regret'],changed['target_regret']);self.assertEqual(clean['true_order_index'],changed['true_order_index'])
  for x in rows:self.assertGreater(x['true_order_mass'],0);self.assertTrue(0<=x['target_regret']<=1)
if __name__=='__main__':unittest.main()
