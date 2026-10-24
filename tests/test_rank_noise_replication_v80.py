import json,random,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'));import rank_noise_replication_v80 as p
class Replication(unittest.TestCase):
 def test_settings_and_freshness(self):
  d=p.protocol();self.assertEqual(len(d['seeds']),30);self.assertFalse(set(d['seeds']) & {4101,5101,6101,7101,8101,9101,30101,30102,30103});self.assertEqual(d['AIF_weight'],.1);self.assertEqual(d['hazard'],.1)
 def test_exact_old_kernel_outputs_on_saved_test_case(self):
  root=Path('outputs/preference_program/results/rank_noise_attribution_v79/results');old=[json.loads(x) for x in (root/'per_user.jsonl').read_text().splitlines()];traces=[json.loads(x) for x in (root/'question_traces.jsonl').read_text().splitlines()]
  for c in p.CONFIGS:
   for a in p.ARMS:
    rr,tt=p.simulate(7101,0,a,c)
    saved=[x for x in old if (x['seed'],x['user_id'],x['filter'],x['arm'],x['condition'])==(7101,0,'hazard10',a,c)]
    for r,s in zip(rr,saved):
     for k in ('target_action','actions_by_excluded_option','target_regret','transfer_regret','true_order_mass'):self.assertEqual(r[k],s[k])
    st=[x for x in traces if (x['seed'],x['user_id'],x['filter'],x['arm'],x['condition'])==(7101,0,'hazard10',a,c)]
    self.assertEqual([x['pair_index'] for x in tt],[x['pair_index'] for x in st])
 def test_stream_interval_pairing_and_smoke_integrity(self):
  self.assertEqual(p.intervals([0.]*30,'constant'),[0.,0.]);root=Path('outputs/preference_program/results/rank_noise_replication_v80_local_smoke');s=json.loads((root/'summary.json').read_text());self.assertEqual(s['records'],40);self.assertEqual(s['trace_records'],64)
if __name__=='__main__':unittest.main()
