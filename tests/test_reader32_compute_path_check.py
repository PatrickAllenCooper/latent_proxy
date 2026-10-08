"""Local synthetic filesystem tests only. No Slurm, model or GPU calls."""
import ast,hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import reader32_compute_path_check as c

class PathCheck(unittest.TestCase):
 def fixture(self,d):
  p=Path(d)/'weights.safetensors';p.write_bytes(b'fixture-not-model');q=Path(d)/'config.json';q.write_text('{}')
  return {x.name:{'bytes':x.stat().st_size,'mtime_ns':x.stat().st_mtime_ns,'target':str(x.resolve()),'sha256':c.sha(x)} for x in [p,q]}
 def env(self):return {'SLURM_JOB_ID':'123','SLURM_JOB_PARTITION':'acpu','SLURM_JOB_ACCOUNT':'ucb736_asc1','SLURM_CPUS_PER_TASK':'1','SLURM_MEM_PER_NODE':'2048','SLURMD_NODENAME':'compute-fixture'}
 def test_literal_match_passes_only_CPU_namespace(self):
  with tempfile.TemporaryDirectory() as d:
   r=c.audit_paths(d,self.fixture(d));self.assertTrue(r['literal_gate_matches_on_this_CPU_node']);self.assertFalse(r['GPU_node_namespace_verified']);self.assertIsNone(r['scientific_qualification'])
 def test_alias_equality_does_not_rescue_literal_mismatch(self):
  with tempfile.TemporaryDirectory() as d:
   files=self.fixture(d);p=Path(d)/'alias';p.symlink_to(Path(d)/'weights.safetensors');files['weights.safetensors']['target']=str(p)
   r=c.audit_paths(d,files);self.assertFalse(r['literal_gate_matches_on_this_CPU_node']);self.assertTrue(r['files'][0]['alias_samefile']);self.assertEqual(r['files'][0]['changed'],['target'])
 def test_size_and_timestamp_drift_fail(self):
  for key in ['bytes','mtime_ns']:
   with tempfile.TemporaryDirectory() as d:
    files=self.fixture(d);files['weights.safetensors'][key]+=1;r=c.audit_paths(d,files);self.assertFalse(r['literal_gate_matches_on_this_CPU_node']);self.assertIn(key,r['files'][0]['changed'])
 def test_nonweight_digest_drift_fails(self):
  with tempfile.TemporaryDirectory() as d:
   files=self.fixture(d);files['config.json']['sha256']='0'*64;r=c.audit_paths(d,files);self.assertFalse(r['literal_gate_matches_on_this_CPU_node'])
 def test_missing_file_explicit(self):
  with tempfile.TemporaryDirectory() as d:
   files=self.fixture(d);(Path(d)/'weights.safetensors').unlink();r=c.audit_paths(d,files);self.assertFalse(r['literal_gate_matches_on_this_CPU_node']);self.assertIn('error',r['files'][0])
 def test_weights_never_bulk_read(self):
  with tempfile.TemporaryDirectory() as d:
   files=self.fixture(d);original=c.sha;seen=[]
   def observe(p):seen.append(Path(p).name);return original(p)
   with patch.object(c,'sha',observe):r=c.audit_paths(d,files)
   self.assertEqual(seen,['config.json']);self.assertFalse(r['weight_content_rehashed'])
 def test_empty_and_unsafe_members_rejected(self):
  with self.assertRaises(AssertionError):c.audit_paths('.',{})
  with self.assertRaises(AssertionError):c.audit_paths('.',{'../outside':{}})
 def test_compute_context_required(self):
  self.assertEqual(c.compute_context(self.env(),'compute-fixture')['GPUs'],0)
  for field,value in [('SLURM_JOB_ID',''),('SLURM_JOB_PARTITION','ah200'),('SLURM_JOB_ACCOUNT','other'),('SLURM_CPUS_PER_TASK','2'),('SLURM_MEM_PER_NODE','1024'),('SLURM_JOB_GPUS','0,1'),('SLURM_GPUS_ON_NODE','1')]:
   env=self.env();env[field]=value
   with self.assertRaises(AssertionError):c.compute_context(env,'compute-fixture')
  env=self.env();env['SLURMD_NODENAME']='login-ci4'
  with self.assertRaises(AssertionError):c.compute_context(env,'login-ci4')
 def test_never_overwrites_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'receipt.json';c.write_new(p,{'first':True})
   with self.assertRaises(FileExistsError):c.write_new(p,{'replacement':True})
   self.assertEqual(json.loads(p.read_text()),{'first':True})
 def test_unapproved_CLI_stops_without_cluster_or_models(self):
  with tempfile.TemporaryDirectory() as d:
   src=Path(d)/'source';spool=Path(d)/'spool';src.mkdir();spool.mkdir();approval=Path(d)/'approval.json';approval.write_text('{"CPU_admission":false}')
   p=subprocess.run([sys.executable,str(ROOT/'scripts/reader32_compute_path_check.py'),str(src),'--approval',str(approval),'--spool',str(spool)],capture_output=True,timeout=5)
   self.assertNotEqual(p.returncode,0);self.assertIn(b'CPU allocation not approved',p.stderr);self.assertEqual(list(spool.iterdir()),[])
 def test_only_standard_library_imports(self):
  tree=ast.parse((ROOT/'scripts/reader32_compute_path_check.py').read_text());names=[]
  for n in ast.walk(tree):
   if isinstance(n,ast.Import):names.extend(x.name for x in n.names)
   if isinstance(n,ast.ImportFrom):names.append(n.module)
  self.assertEqual(set(names),{'argparse','hashlib','json','os','pathlib','socket','sys','threading','time'})

if __name__=='__main__':unittest.main()
