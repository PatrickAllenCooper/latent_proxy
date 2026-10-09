"""Local binding fixtures only; no package imports, remote writes or allocations."""
import hashlib,json,os
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reader32_trace_capture_v68 import verify_runtime_descriptor,verify_binding
from reader32_trace_privacy_v68 import TraceSink,TraceStop

class BindingFixtures(unittest.TestCase):
 def fixture(self,d):
  p=Path(d);receipt=p/'receipt.json';receipt.write_text('{}')
  known=p/'known.so';known.write_bytes(b'toy bytes')
  st=known.stat();tmp=p/'node';runtime=tmp/'lp-reader32-import-v68-123'
  (p/'runtime_stage.json').write_text(json.dumps({'event':'node_local_runtime_ready','path':str(runtime),'archive_sha256':'a'*64}))
  manifest={'runtime_descriptor':{'receipt_path':str(receipt),'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'archive_sha256':'a'*64},
   'shared_exact_files':[{'path':str(known),'sha256':hashlib.sha256(known.read_bytes()).hexdigest(),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'label':'PIL/toy.so','verified_aliases':[]}]}
  env={'SLURM_TMPDIR':str(tmp),'SLURM_JOB_ID':'123','LOCAL_RUNTIME':str(runtime)}
  return manifest,env,known
 def test_bound_runtime_recipe_exact_file_redaction(self):
  with tempfile.TemporaryDirectory() as d:
   m,e,k=self.fixture(d);root,exact=verify_runtime_descriptor(m,e,d)
   s=TraceSink(d,(root,),exact_paths=exact)
   self.assertEqual(s.path_label(str(k)),'PIL/toy.so')
   self.assertEqual(s.path_label(str(k.parent/'unknown.so')),'[redacted]')
   self.assertEqual(s.path_label(root+'/x.py'),'root0/x.py');s.close()
 def test_bad_stage_path_hash_and_job_rejected(self):
  for mutation in ('path','hash','job','oversize','shared'):
   with tempfile.TemporaryDirectory() as d:
    m,e,k=self.fixture(d)
    if mutation=='path':e['LOCAL_RUNTIME']='/private/unrelated'
    if mutation=='hash':m['runtime_descriptor']['receipt_sha256']='0'*64
    if mutation=='job':e['SLURM_JOB_ID']='../bad'
    if mutation=='oversize':Path(d,'runtime_stage.json').write_text('x'*15361)
    if mutation=='shared':k.write_bytes(b'changed')
    with self.assertRaises(TraceStop):verify_runtime_descriptor(m,e,d)
 def test_verified_alias_only(self):
  with tempfile.TemporaryDirectory() as d:
   m,e,k=self.fixture(d);alias=Path(d,'alias.so');alias.symlink_to(k)
   m['shared_exact_files'][0]['verified_aliases']=[str(alias)]
   _,exact=verify_runtime_descriptor(m,e,d);self.assertEqual(exact[str(alias)],'PIL/toy.so')
   alias.unlink();alias.write_bytes(b'wrong')
   with self.assertRaises(TraceStop):verify_runtime_descriptor(m,e,d)
 def test_frozen_packet_source_and_controller_hashes(self):
  root=Path(__file__).resolve().parents[1];folder=root/'outputs/preference_program/manifests/reader32_import_trace_v68_local'
  local=json.loads((folder/'manifest.json').read_text());verify_binding(local)
  remote=json.loads((folder/'remote_manifest_held.json').read_text())
  self.assertFalse(remote['cpu_admission']);self.assertFalse(remote['remote_materialization_ready'])
  self.assertNotIn('/Users/',json.dumps(remote['source_hashes']))
  for p in (folder/'controller').iterdir():
   key='/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68/controller/'+p.name
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),remote['source_hashes'][key])
  self.assertEqual(sum(remote['metadata_budget'][k] for k in ('runtime_stage_max','capture_receipt_max','wrapper_receipt_max')),65536)

if __name__=='__main__':unittest.main()

class BaselineEnvironmentFixture(unittest.TestCase):
 def test_child_import_path_and_working_directory_preserved(self):
  root=Path(__file__).resolve().parents[1]
  wrapper=(root/'scripts/slurm/reader32_import_trace_v68_held.slurm').read_text()
  baseline=(root/'scripts/slurm/reader32_import_diagnostic_v67_cpu.slurm').read_text()
  expected='export PYTHONPATH="$LOCAL_RUNTIME:$SOURCE"'
  self.assertIn(expected,baseline);self.assertIn(expected,wrapper)
  self.assertNotIn('$SOURCE:$ROOT/controller',wrapper)
  self.assertIn('cd "$SOURCE"',baseline);self.assertIn('cd "$SOURCE"',wrapper)
