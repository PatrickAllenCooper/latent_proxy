"""Exercise actual CPU wrapper with a synthetic tokenizer, never a model."""
import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
class Continuation(unittest.TestCase):
 def exercise(self,mode):
  calls=[];template=(ROOT/'outputs/preference_program/results/dialogue_phase2_checkpoint/chat_template.jinja').read_text()
  class Qwen2Tokenizer:
   chat_template=template
   @classmethod
   def from_pretrained(cls,snapshot,**kw):
    assert snapshot.endswith('5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd') and kw=={'local_files_only':True,'trust_remote_code':False};return cls()
   def apply_chat_template(self,messages,tokenize=True,**kw):
    if not tokenize:return 'rendered'
    return [1,2] if mode=='raw_equal' else {'input_ids':[1,3] if mode=='real_difference' else [1,2]}
   def encode(self,rendered,**kw):return [1,2]
  modules={}
  for name in ['transformers','transformers.models','transformers.models.qwen2','transformers.models.qwen2.tokenization_qwen2']:
   modules[name]=types.ModuleType(name);modules[name].__path__=[]
  modules['transformers.models.qwen2.tokenization_qwen2'].Qwen2Tokenizer=Qwen2Tokenizer
  i=types.ModuleType('indexed_startup');i.indexed_import=lambda f:(f(),[]);modules[i.__name__]=i
  i=types.ModuleType('indexed_source_inspection');i.inspect_with_index=lambda f:(f(),{});modules[i.__name__]=i
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'execution_freeze.json').write_text(json.dumps({'source_hashes':{}}));rows=[{'case_id':str(n),'messages':[{'role':'user','content':'frozen'}]} for n in range(16)];(r/'prompts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
   with patch.dict(sys.modules,modules),patch.object(sys,'argv',['prepare_reader32_v61.py',str(r)]),patch('runpy.run_path',side_effect=lambda *args,**kwargs:calls.append(args)),patch('builtins.print'):
    error=None
    try:exec(compile((ROOT/'scripts/prepare_reader32_v61.py').read_text(),'prepare_reader32_v61.py','exec'),{'__name__':'fixture'})
    except AssertionError as exc:error=str(exc)
   receipt=json.loads((r/'token_diagnostic_receipt.json').read_text());records=list(map(json.loads,(r/'token_diagnostic/tokens.jsonl').read_text().splitlines()))
   return calls,error,receipt,records
 def test_proven_container_only_continues_once(self):
  calls,error,receipt,records=self.exercise('container');self.assertIsNone(error);self.assertEqual(len(calls),1);self.assertTrue(receipt['complete']);self.assertEqual(len(records),16);self.assertTrue(all(x['chat_ids']==[1,2] for x in records))
 def test_true_ID_difference_stops_and_preserves_operand(self):
  calls,error,receipt,records=self.exercise('real_difference');self.assertEqual(calls,[]);self.assertIsNotNone(error);self.assertFalse(receipt['complete']);self.assertEqual(records[0]['chat_ids'],[1,3]);self.assertEqual(records[0]['render_ids'],[1,2])
 def test_unreproduced_failure_cannot_trigger_integrity_retry(self):
  calls,error,receipt,records=self.exercise('raw_equal');self.assertEqual(calls,[]);self.assertIsNotNone(error);self.assertTrue(receipt['complete']);self.assertTrue(records[0]['raw_equality'])
if __name__=='__main__':unittest.main()
