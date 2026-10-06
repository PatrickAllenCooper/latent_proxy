import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_reader32_token_gate import check_gate
class TokenGate(unittest.TestCase):
 def test_equal_list_unchanged(self):
  x=check_gate('x',[1,2],[1,2]);self.assertTrue(x['gate_valid']);self.assertEqual(x['chat_ids'],[1,2])
 def test_container_normalization_only(self):
  for value in [{'input_ids':[1,2]},[[1,2]]]:
   x=check_gate('x',value,[1,2]);self.assertFalse(x['raw_equality']);self.assertTrue(x['gate_valid']);self.assertEqual(x['chat_ids'],[1,2])
 def test_true_difference_never_corrected(self):
  x=check_gate('x',{'input_ids':[1,3]},[1,2]);self.assertFalse(x['gate_valid']);self.assertEqual(x['first_difference'],1)
 def test_oversized_invalid_types_or_batch_fail(self):
  for x in [[1]*513,[],['1'],{'bad':[1]},[[1],[2]],[-1]]:self.assertFalse(check_gate('x',x,[1])['gate_valid'])
if __name__=='__main__':unittest.main()
