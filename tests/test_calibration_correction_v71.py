import json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
ROOT=Path('outputs/preference_program/manifests/reader32_tracer_calibration_v71')
class Correction(unittest.TestCase):
 def test_early_failure_is_safe_and_bounded(self):
  with tempfile.TemporaryDirectory() as d:
   env=dict(os.environ,STUDY_WALL_START=str(time.time()),SLURM_JOB_ID='123',SLURM_JOB_PARTITION='acpu',SLURM_CPUS_PER_TASK='1',SLURM_MEM_PER_NODE='256',PYTHONDONTWRITEBYTECODE='1');env.pop('SLURM_JOB_ACCOUNT',None)
   r=subprocess.run([sys.executable,'-S',str(ROOT/'controller/reader32_tracer_calibration_v69.py'),str(ROOT/'manifest.json'),'/unused',d],env=env,capture_output=True,timeout=3)
   self.assertEqual(r.returncode,1);p=Path(d,'calibration_failure.json');self.assertLessEqual(p.stat().st_size,256);self.assertEqual(json.loads(p.read_text()),{'stage':1,'code':11,'allocation_checks':[True,True,True,False]});self.assertEqual(r.stderr,b'')
 def test_outer_budget_and_frozen_identity(self):
  import hashlib
  m=json.loads((ROOT/'manifest.json').read_text());s=(ROOT/'controller/reader32_tracer_calibration_v69_held.slurm').read_text();self.assertIn('29s bash "$0"',s);self.assertIn('--kill-after=1s',s);self.assertEqual(m['scheduler_wall_seconds'],60)
  for path,digest in m['source_hashes'].items():self.assertEqual(hashlib.sha256((ROOT/Path(path).parent.name/Path(path).name).read_bytes()).hexdigest(),digest)
if __name__=='__main__':unittest.main()
