import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('contracts',Path(__file__).resolve().parents[1]/'scripts/reader32_v60_contracts.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
class Contracts(unittest.TestCase):
 def setUp(self):self.ids=[f'case-{i}' for i in range(16)]
 def records(self,n,correct=True):return [{'case_id':i,'score':{'correct':correct}} for i in self.ids[:n]]
 def test_capacity_admission(self):
  for size in [0,35*c.GIB,65*c.GIB]:
   with self.assertRaises(ValueError):c.memory_budget(size)
  self.assertEqual(c.memory_budget(71*c.GIB),65*c.GIB)
  self.assertEqual(c.memory_budget(66*c.GIB),int(66*c.GIB*.95))
 def test_clock_and_memory_boundaries(self):
  self.assertIsNone(c.violation(89.99,0,90,10,10,10))
  self.assertEqual(c.violation(90,0,90,0,0,10),'time_limit')
  self.assertEqual(c.violation(285,0,400,0,0,10),'time_limit')
  self.assertEqual(c.violation(20,0,30,11,0,10),'framework_memory_limit')
  self.assertEqual(c.violation(20,0,30,0,11,10),'framework_memory_limit')
 def test_response_caps_and_estimated_remaining_time(self):
  self.assertTrue(c.admit_next(89,0,[],0))
  self.assertFalse(c.admit_next(276,0,[],0))
  self.assertFalse(c.admit_next(200,0,[9],1))
  self.assertTrue(c.admit_next(100,0,[2],1))
  self.assertFalse(c.admit_next(10,0,[1],16))
 def test_first_case_probe_exactly_sixteen(self):
  calls=[]
  for i in self.ids:
   self.assertTrue(c.admit_next(90+len(calls),0,[1] if calls else [],len(calls)));calls.append(i)
  self.assertEqual(calls,self.ids);self.assertEqual(len(calls),16)
  self.assertTrue(c.result(self.records(16),self.ids,True)['qualified'])
 def test_partial_never_semantic_failure(self):
  for n in [0,1,15]:
   x=c.result(self.records(n,False),self.ids,False);self.assertFalse(x['complete']);self.assertIsNone(x['qualified'])
   with self.assertRaises(ValueError):c.result(self.records(n),self.ids,True)
 def test_order_duplicate_fail_closed(self):
  for rs in [[{'case_id':'other','score':{'correct':True}}],self.records(1)*2,self.records(16)+self.records(1)]:
   with self.assertRaises(ValueError):c.result(rs,self.ids,False)
 def test_completed_negative_separate_from_resource_failure(self):
  x=c.result(self.records(16,False),self.ids,True);self.assertTrue(x['complete']);self.assertFalse(x['qualified']);self.assertEqual(x['classification'],'completed_qualification')
 def test_partial_export_preserves_raw_and_tolerates_truncated_tail(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'responses.jsonl';raw=json.dumps(self.records(1)[0])+'\n{"case_id":';p.write_text(raw)
   x=c.export_incomplete(d,'time_limit',285);self.assertEqual(x['records'],1);self.assertIsNone(x['qualified']);self.assertEqual(p.read_text(),raw)
   self.assertEqual(json.loads((Path(d)/'budget_stop.json').read_text()),x)
if __name__=='__main__':unittest.main()
