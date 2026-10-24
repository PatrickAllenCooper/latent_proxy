import json,random,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'));import rank_noise_attribution_v79 as n;import rank_stress_v78 as old
class Attribution(unittest.TestCase):
 def test_10percent_planner_matches_frozen_prior_algorithm(self):
  rng=random.Random(19);_,u,r=n.utilities(n.ATTR,0)
  for _ in range(8):
   w=[rng.random() for _ in range(24)];w=[p/sum(w) for p in w]
   for arm in n.ARMS:self.assertEqual(n.select(arm,w,u,r,random.Random(1),.1),old.select(arm,w,u,r,random.Random(1)))
 def test_matched_update_not_fixed10(self):
  w=n.update(n.UNIFORM,0,True,.2);self.assertAlmostEqual(sum(p for p,y in zip(w,n.YES[0]) if y),.8)
  self.assertEqual(n.protocol()['seeds'],[7101,8101,9101]);self.assertFalse(set(n.protocol()['seeds']) & {4101,5101,6101})
 def test_smoke_records_actions_and_random_queries_invariant(self):
  root=Path('outputs/preference_program/results/rank_noise_attribution_v79_local_smoke');traces=[json.loads(x) for x in (root/'question_traces.jsonl').read_text().splitlines()];choices={}
  for r in traces:
   self.assertEqual(len(r['posterior_weights']),24);self.assertAlmostEqual(sum(r['posterior_weights']),1)
   if r['arm']=='random':choices.setdefault((r['seed'],r['user_id'],r['question']),set()).add(r['pair_index'])
  self.assertTrue(all(len(x)==1 for x in choices.values()))
  for r in map(json.loads,(root/'per_user.jsonl').read_text().splitlines()):self.assertIn(r['target_action'],'ABCD');self.assertEqual(len(r['actions_by_excluded_option']),4)
if __name__=='__main__':unittest.main()
