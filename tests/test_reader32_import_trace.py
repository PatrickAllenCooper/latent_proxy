"""Local-only diagnostic fixtures; no models, installs, Slurm or GPU calls."""
import ast,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from reader32_import_trace import import_trace

class Trace(unittest.TestCase):
 def test_schedule_and_cancel_on_success(self):
  with patch('reader32_import_trace.faulthandler.dump_traceback_later') as begin,patch('reader32_import_trace.faulthandler.cancel_dump_traceback_later') as end:
   with import_trace():pass
   begin.assert_called_once_with(20,repeat=True,file=sys.stderr);end.assert_called_once_with()
 def test_cancel_after_import_error(self):
  with patch('reader32_import_trace.faulthandler.dump_traceback_later'),patch('reader32_import_trace.faulthandler.cancel_dump_traceback_later') as end:
   with self.assertRaises(RuntimeError):
    with import_trace():raise RuntimeError('fixture')
   end.assert_called_once_with()
 def test_bad_interval_rejected(self):
  with self.assertRaises(ValueError):
   with import_trace(interval_seconds=0):pass
 def test_real_stderr_stack_retained_and_cancellation(self):
  code="import sys,time;sys.path.insert(0,sys.argv[1]);from reader32_import_trace import import_trace\nwith import_trace(interval_seconds=.02):time.sleep(.055)\nprint('CANCELLED',file=sys.stderr,flush=True);time.sleep(.05)"
  p=subprocess.run([sys.executable,'-c',code,str(ROOT/'scripts')],capture_output=True,timeout=3)
  self.assertEqual(p.returncode,0);raw=p.stderr.decode();self.assertIn('Timeout',raw);self.assertIn('File "<string>"',raw);self.assertNotIn('Timeout',raw.split('CANCELLED')[1])
 def test_proposal_model_and_token_calls_identical(self):
  def calls(path,name):
   tree=ast.parse(path.read_text())
   return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr==name]
  for name in ['from_pretrained','generate','apply_chat_template','encode']:
   self.assertEqual(calls(ROOT/'scripts/run_reader32_v65.py',name),calls(ROOT/'scripts/run_reader32_v67_proposal.py',name))
 def test_proposal_imports_and_numeric_guards_identical(self):
  def deps(path):
   t=ast.parse(path.read_text());f=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='deps')
   return [ast.dump(n,include_attributes=False) for n in ast.walk(f) if isinstance(n,(ast.Import,ast.ImportFrom))]
  self.assertEqual(deps(ROOT/'scripts/run_reader32_v65.py'),deps(ROOT/'scripts/run_reader32_v67_proposal.py'))
  text=(ROOT/'scripts/run_reader32_v67_proposal.py').read_text()
  for value in ["start+90", "generation_end_seconds':270", "complete_seconds':285", "first_generation_seconds':30", "remaining_generation_seconds':10"]:self.assertIn(value,text)
if __name__=='__main__':unittest.main()
