import ast,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class CPUOnly(unittest.TestCase):
 def test_exact_import_list(self):
  def imports(path,name):
   t=ast.parse(path.read_text());f=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
   return [ast.dump(n,include_attributes=False) for n in ast.walk(f) if isinstance(n,(ast.Import,ast.ImportFrom))]
  self.assertEqual(imports(ROOT/'scripts/reader32_import_diagnostic_v67_cpu.py','dependencies'),imports(ROOT/'scripts/run_reader32_v65.py','deps'))
 def test_no_model_token_or_GPU_calls(self):
  t=ast.parse((ROOT/'scripts/reader32_import_diagnostic_v67_cpu.py').read_text())
  forbidden={'generate','from_pretrained','apply_chat_template','cuda','is_available','load_state_dict'}
  self.assertFalse([n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden])
 def test_unapproved_stops_before_dependency_import(self):
  with tempfile.TemporaryDirectory() as d:
   s=Path(d)/'source';o=Path(d)/'spool';s.mkdir();o.mkdir();(s/'diagnostic_freeze.json').write_text('{"CPU_admission":false}')
   p=subprocess.run([sys.executable,str(ROOT/'scripts/reader32_import_diagnostic_v67_cpu.py'),str(s),str(o)],env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','STUDY_WALL_START':str(time.time())},capture_output=True,timeout=3)
   self.assertNotEqual(p.returncode,0);self.assertNotIn(b'guarded_dependency_import_start',p.stdout);self.assertEqual(list(o.iterdir()),[])
if __name__=='__main__':unittest.main()
