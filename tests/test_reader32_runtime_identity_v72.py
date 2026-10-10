import hashlib,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reader32_runtime_identity_v72 as d
class IdentityFixtures(unittest.TestCase):
 def manifest(self):return {'script_sha256':hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest(),'purpose':'observed_runtime_identity_only','identity_diagnostic_admission':True,'allocation_approved':True,'fresh_approval_message_id':'fixture_fresh','resources':dict(d.RESOURCES),'attempts_used':0,'retry':False,'runtime_paths':dict(d.PATHS),'expected_sha256':{k:'a'*64 for k in d.PATHS}}
 def env(self):return dict(STUDY_WALL_START=str(time.time()),SLURM_JOB_PARTITION='acpu',SLURM_CPUS_PER_TASK='1',SLURM_MEM_PER_NODE='256',SLURM_JOB_ACCOUNT='ucb736_asc1',SLURM_JOB_ID='123')
 def test_held_no_file_reads(self):
  with tempfile.TemporaryDirectory() as root,patch.object(d,'observe') as read:
   m=self.manifest();m['identity_diagnostic_admission']=False
   with self.assertRaises(d.Held):d.execute(m,Path(root),self.env())
   read.assert_not_called();self.assertEqual(list(Path(root).iterdir()),[])
 def test_consumed_approvals_and_resource_changes(self):
  for approval in d.CONSUMED:
   m=self.manifest();m['fresh_approval_message_id']=approval
   with self.assertRaises(d.Held):d.admit(m,self.env(),time.time())
  m=self.manifest();m['resources']['gpus']=1
  with self.assertRaises(d.Held):d.admit(m,self.env(),time.time())
 def test_actual_streamed_hash_match_mismatch_missing_cap_deadline(self):
  with tempfile.TemporaryDirectory() as root:
   p=Path(root,'input');p.write_bytes(b'toy');good=hashlib.sha256(b'toy').hexdigest()
   self.assertEqual(d.observe(p,good,time.time()+1,3)[0]['code'],0)
   self.assertEqual(d.observe(p,'a'*64,time.time()+1,3)[0]['code'],1)
   self.assertEqual(d.observe(p,'a'*64,time.time()+1,2)[0]['code'],3)
   self.assertEqual(d.observe(p,'a'*64,0,3)[0]['code'],4)
   self.assertEqual(d.observe(Path(root,'missing'),'a'*64,time.time()+1,3)[0]['code'],2)
 def test_records_both_bindings_without_rebaselining(self):
  m=self.manifest();original=json.dumps(m,sort_keys=True);calls=[]
  def fake(path,expected,deadline,budget):
   calls.append(path);return {'expected':expected,'observed':('a' if len(calls)==1 else 'b')*64,'bytes':3,'code':int(len(calls)==2)},3
  with tempfile.TemporaryDirectory() as root,patch.object(d,'observe',side_effect=fake):
   self.assertEqual(d.execute(m,Path(root),self.env()),1);r=json.loads(Path(root,'runtime_identity.json').read_text());self.assertEqual(r['python']['code'],0);self.assertEqual(r['strace']['code'],1);self.assertFalse(r['trusted_bindings_changed']);self.assertLessEqual(len(d.encode(r)),1024)
   r['python']['observed']='/private/secret'
   with self.assertRaises(d.Held):d.encode(r)
  self.assertEqual(json.dumps(m,sort_keys=True),original)
if __name__=='__main__':unittest.main()
