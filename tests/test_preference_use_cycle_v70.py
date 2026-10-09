import itertools,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import preference_use_cycle_v70 as p
class Fixtures(unittest.TestCase):
 def case(self):return {'priority':list(p.ATTR),'options':[{'label':l,'attribute':a,'eligible':i!=0} for i,(l,a) in enumerate(zip('ABCD',p.ATTR))]}
 def test_excluded_top(self):self.assertEqual(p.oracle(self.case()),'B')
 def test_strict_parse_and_penalty(self):
  for raw in ('Answer: B','B or C','',None):self.assertTrue(p.score(self.case(),raw)['parse_failure'])
  self.assertTrue(p.score(self.case(),'A')['constraint_violation']);self.assertEqual(p.score(self.case(),'A')['normalized_regret'],1)
 def test_tool_budget_and_protocol(self):
  s=p.ProxySession(self.case());self.assertEqual(s.call({'operation':'recommend_constrained'})['action'],'B');s.call({'operation':'evaluate_candidates','labels':['A','B']})
  with self.assertRaises(ValueError):s.call({'operation':'recommend_constrained'})
 def test_invalid_requests(self):
  for r in ({'operation':'unknown'},{'operation':'recommend_constrained','secret':1},{'operation':'evaluate_candidates','labels':['A','A']}):
   with self.assertRaises(ValueError):p.ProxySession(self.case()).call(r)
 def test_hybrid_preserves_original(self):
  r=p.hybrid(self.case(),'A');self.assertFalse(r['original']['correct']);self.assertTrue(r['final']['correct']);self.assertTrue(r['corrected'])
 def test_private_profile_not_in_no_info_or_tool(self):
  for arm in ('no_preferences','proxy_tool','validated_hybrid'):self.assertNotIn('Priority list:',p.prompt(self.case(),arm))
 def test_discovery_separate_from_execution(self):
  r=p.discovery_score(self.case(),list(p.ATTR));self.assertEqual(r['pairwise_order_error'],0);self.assertTrue(r['true_preference_decision']['correct'])
 def test_frozen_disjoint_and_semantic_diversity(self):
  root=Path('outputs/preference_program/manifests/preference_use_cycle_v70');rows=json.loads((root/'cases.json').read_text());splits=json.loads((root/'splits.json').read_text())
  for a,b in itertools.combinations(splits,2):
   self.assertFalse({tuple(x) for x in splits[a]['users']} & {tuple(x) for x in splits[b]['users']})
   self.assertFalse({(tuple(x['attributes']),x['excluded_index']) for x in splits[a]['menus']} & {(tuple(x['attributes']),x['excluded_index']) for x in splits[b]['menus']})
  smoke=[c for c in rows if c['user_id'] in ('heldout-00','heldout-01')]
  self.assertEqual(len({(tuple(c['priority']),next(o['attribute'] for o in c['options'] if not o['eligible'])) for c in smoke}),8)
 def test_saved_trace_integrity_and_separate_hybrid_score(self):
  r={'case_id':'fixture','arm':'validated_hybrid','model_revision':p.MODEL['revision'],'EOS_observed':True,'length_cap_reached':False,'generated_token_ids':[1,2],'tool_trace':[{'request':{'operation':'recommend_constrained'},'result':{'action':'B'}}],'raw_final_output':'A'};c=dict(self.case(),case_id='fixture')
  out=p.audit_response(c,'validated_hybrid',r);self.assertFalse(out['reader']['correct']);self.assertTrue(out['hybrid']['final']['correct'])
  r['tool_trace'][0]['result']['action']='A'
  with self.assertRaises(ValueError):p.audit_response(c,'validated_hybrid',r)
 def test_exhaustive_oracle(self):
  n=0
  for priority,attrs,ex in itertools.product(itertools.permutations(p.ATTR),itertools.permutations(p.ATTR),range(4)):
   c={'priority':list(priority),'options':[{'label':l,'attribute':a,'eligible':i!=ex} for i,(l,a) in enumerate(zip('ABCD',attrs))]};self.assertEqual(p.oracle(c),p.independent_oracle(c));n+=1
  self.assertEqual(n,2304)
if __name__=='__main__':unittest.main()
