import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'));import rank_discovery_pilot_v77 as p
class Pilot(unittest.TestCase):
 def case(self):return {'priority':list(p.ATTR),'options':[{'label':l,'attribute':a,'eligible':i!=0} for i,(l,a) in enumerate(zip('ABCD',p.ATTR))]}
 def test_partition_and_oracle(self):
  g=p.groups(p.ORDERS,p.PAIRS[0]);self.assertEqual([len(x) for x in g],[12,12]);c=self.case();self.assertEqual(p.risk([tuple(c['priority'])],c),0)
 def test_saved_counts_and_common_baseline(self):
  root=Path('outputs/preference_program/results/rank_discovery_pilot_v77');rows=[json.loads(x) for x in (root/'per_user.jsonl').read_text().splitlines()];self.assertEqual(len(rows),1440)
  for seed in (1001,2001,3001):
   for user in range(24):
    v=[r for r in rows if r['seed']==seed and r['user_id']==user and r['questions']==0];self.assertEqual(len(v),4);self.assertEqual(len({r['normalized_regret'] for r in v}),1)
  for r in rows:self.assertTrue(0<=r['normalized_regret']<=1);self.assertGreaterEqual(r['posterior_orders'],1)
 def test_trace_answers_and_posterior_replay(self):
  root=Path('outputs/preference_program/results/rank_discovery_pilot_v77');rows=[json.loads(x) for x in (root/'question_traces.jsonl').read_text().splitlines()];self.assertEqual(len(rows),2304);states={}
  for r in rows:
   key=(r['seed'],r['user_id'],r['arm']);prior=states.get(key,p.ORDERS[:]);truth=p.ORDERS[r['user_id']];a,b=r['pair'];self.assertEqual(truth.index(a)<truth.index(b),r['answer']);self.assertEqual(len(prior),r['prior_orders']);post=[o for o in prior if (o.index(a)<o.index(b))==r['answer']];self.assertEqual(len(post),r['posterior_orders']);self.assertIn(truth,post);states[key]=post
if __name__=='__main__':unittest.main()
