"""Exercise actual GPU CLI failure routing before any framework import."""
import json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Layout(unittest.TestCase):
 def test_preimport_failure_exports_only_to_sibling_spool(self):
  with tempfile.TemporaryDirectory() as d:
   base=Path(d);src=base/'source';spool=base/'spool';src.mkdir();spool.mkdir()
   for n,txt in [('registration.json','{}'),('cases.jsonl',''),('prompts.jsonl','')]:p=src/n;p.write_text(txt);p.chmod(0o444)
   src.chmod(0o555);env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','STUDY_WALL_START':str(time.time())}
   try:
    p=subprocess.run([sys.executable,str(ROOT/'scripts/run_reader32_v64.py'),'gpu',str(src),'--spool',str(spool)],env=env,capture_output=True,timeout=10)
    self.assertNotEqual(p.returncode,0);x=json.loads((spool/'smoke/budget_stop.json').read_text());self.assertFalse(x['complete']);self.assertIsNone(x['qualified']);self.assertEqual(x['records'],0);self.assertEqual(sorted(y.name for y in src.iterdir()),['cases.jsonl','prompts.jsonl','registration.json'])
   finally:src.chmod(0o755)
 def test_nested_spool_rejected_before_any_source_write(self):
  with tempfile.TemporaryDirectory() as d:
   src=Path(d)/'source';src.mkdir();nested=src/'spool';env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','STUDY_WALL_START':str(time.time())}
   p=subprocess.run([sys.executable,str(ROOT/'scripts/run_reader32_v64.py'),'gpu',str(src),'--spool',str(nested)],env=env,capture_output=True,timeout=10)
   self.assertNotEqual(p.returncode,0);self.assertIn('spool must be outside source',p.stderr.decode());self.assertEqual(list(src.iterdir()),[])
if __name__=='__main__':unittest.main()
