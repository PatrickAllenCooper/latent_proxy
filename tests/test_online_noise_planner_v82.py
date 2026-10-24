import inspect,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'));import online_noise_planner_v82 as n
class OnlineNoise(unittest.TestCase):
 def test_prior_and_no_truth_interface(self):
  j=n.noise_prior();self.assertAlmostEqual(sum(sum(r) for r in j),1);self.assertEqual(n.noise_state(j)[0],.1);self.assertEqual(list(inspect.signature(n.noise_update).parameters),['joint','pair','answer'])
  self.assertEqual(n.planner_noise('online',j,.1),n.planner_noise('online',j,.3));self.assertEqual(n.planner_noise('fixed10',j,.3),.1);self.assertEqual(n.planner_noise('oracle',j,.3),.3)
 def test_noise_posterior_uses_evidence(self):
  j=n.noise_prior();one=n.noise_update(j,0,True);self.assertAlmostEqual(n.noise_state(one)[0],.1)
  consistent=n.noise_update(one,0,True);conflicting=n.noise_update(one,0,False);self.assertLess(n.noise_state(consistent)[0],n.noise_state(conflicting)[0])
 def test_tenpercent_oracle_equals_fixed(self):
  a,at=n.simulate(80101,0,.1,'fixed10');b,bt=n.simulate(80101,0,.1,'oracle')
  for x,y in zip(a,b):
   for k in ('target_action','transfer_regret','estimated_noise'):self.assertEqual(x[k],y[k])
  self.assertEqual([r['pair_index'] for r in at],[r['pair_index'] for r in bt])
 def test_fixed_and_oracle_reproduce_prior_cells(self):
  old=[json.loads(x) for x in Path('outputs/preference_program/results/rank_noise_attribution_v79/results/per_user.jsonl').read_text().splitlines()]
  for mode,condition in [('fixed10','update10_plan10'),('oracle','update10_plan20')]:
   rows,_=n.simulate(7101,0,.2,mode);saved=[r for r in old if (r['seed'],r['user_id'],r['condition'],r['arm'],r['filter'])==(7101,0,condition,'AIF','hazard10')]
   for r,t in zip(rows,saved):
    for k in ('target_action','transfer_regret','true_order_mass'):self.assertEqual(r[k],t[k])
 def test_online_has_no_future_answer_access(self):
  rows,traces=n.simulate(80102,1,.2,'online');self.assertEqual(traces[0]['planner_noise_before_query'],.1)
  for i,r in enumerate(traces):
   self.assertEqual(r['planner_noise_before_query'],.1 if i==0 else traces[i-1]['estimated_noise_after_answer']);self.assertEqual(r['estimated_noise_before_query'],r['planner_noise_before_query'])
if __name__=='__main__':unittest.main()
