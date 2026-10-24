import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'));import rank_noise_components_v81 as n;import analyze_rank_noise_components_v81 as a
class Components(unittest.TestCase):
 def test_fresh_streams_and_settings(self):
  d=n.protocol();self.assertEqual(len(d['seeds']),30);self.assertFalse(set(d['seeds']) & set(range(20101,20131)));self.assertEqual(d['hazard'],.1);self.assertEqual(d['AIF_weight'],.1)
 def test_all4cells_reproduce_v79_exactly(self):
  old=[json.loads(x) for x in Path('outputs/preference_program/results/rank_noise_attribution_v79/results/per_user.jsonl').read_text().splitlines()]
  for c in n.CONFIGS:
   for arm in n.ARMS:
    rows,_=n.simulate(7101,0,arm,c);saved=[r for r in old if (r['seed'],r['user_id'],r['condition'],r['arm'],r['filter'])==(7101,0,c,arm,'hazard10')]
    for r,s in zip(rows,saved):
     for k in ('target_action','actions_by_excluded_option','target_regret','transfer_regret','true_order_mass'):self.assertEqual(r[k],s[k])
 def test_factorial_coefficients(self):
  ys=[1,3,4,6];effects={k:sum(v*c for v,c in zip(ys,coef)) for k,coef in a.SPECS.items()};self.assertEqual(effects['average_update_effect'],2);self.assertEqual(effects['average_planner_effect'],3);self.assertEqual(effects['interaction'],0);self.assertEqual(a.quantile([0.,1.],.5),.5)
if __name__=='__main__':unittest.main()
