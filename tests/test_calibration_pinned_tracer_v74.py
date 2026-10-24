import hashlib,importlib.util,json,sys,unittest
from pathlib import Path
ROOT=Path('outputs/preference_program/manifests/reader32_tracer_calibration_v74')
sys.path.insert(0,str(ROOT/'controller'))
spec=importlib.util.spec_from_file_location('pinned_calibration',ROOT/'controller/reader32_tracer_calibration_v69.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
class Pinned(unittest.TestCase):
 def test_command_uses_exact_pinned_manifest_path(self):
  m=json.loads((ROOT/'manifest.json').read_text());cmd=h.build_command(m,Path(m['source_dir']),Path('/spool'),4);self.assertEqual(cmd[0],'/scratch/alpine/paco0228/latent_proxy_runs/tracer-calibration-v74/runtime/strace');self.assertNotIn('-p',cmd);self.assertNotIn('sudo',cmd);self.assertEqual(m['runtime']['strace_sha256'],'e85f6a7cc0d125339704a397eac1a2b2d2eeb243d57608dce277cd6373f3f1b7')
 def test_source_inventory_and_limits_preserved(self):
  m=json.loads((ROOT/'manifest.json').read_text())
  for path,digest in m['source_hashes'].items():self.assertEqual(hashlib.sha256((ROOT/Path(path).parent.name/Path(path).name).read_bytes()).hexdigest(),digest)
  self.assertEqual(m['resources']['gpus'],0);self.assertEqual(m['resources']['memory_MiB'],256);self.assertEqual(m['scheduler_wall_seconds'],60);self.assertEqual(m['outer_watchdog_seconds'],30)
if __name__=='__main__':unittest.main()
