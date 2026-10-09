"""Local stdlib-only toy fixtures; synthetic traces are never CURC evidence."""
import ast,hashlib,json,os,shutil,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reader32_calibration_contract_v69 as c
import reader32_tracer_calibration_v69 as h
import reader32_trace_capture_v69_calibration as e
from reader32_trace_privacy_v68 import TraceSink,TraceStop
from prepare_reader32_tracer_calibration_v69 import OUT

class ContractFixtures(unittest.TestCase):
 def manifest(self):return {'purpose':'stdlib_owned_tracer_calibration','calibration_admission':True,'allocation_approved':True,'fresh_approval_message_id':'Sentinel_fixture_fresh','resources':dict(c.RESOURCES),'metadata_budget':dict(c.BUDGET),'attempts_used':0,'retry':False}
 def env(self):return {'STUDY_WALL_START':str(time.time()),'SLURM_JOB_ID':'123','SLURM_JOB_PARTITION':'acpu','SLURM_CPUS_PER_TASK':'1','SLURM_MEM_PER_NODE':'256','SLURM_JOB_ACCOUNT':'ucb736_asc1','PYTHONDONTWRITEBYTECODE':'1'}
 def test_hold_and_consumed_v68_approval(self):
  for mutation in ({'calibration_admission':False},{'allocation_approved':False},{'fresh_approval_message_id':'Sentinel_4455df170154819198f158ac5465133e'},{'attempts_used':1},{'retry':True},{'purpose':'framework_qualification'}):
   m=self.manifest();m.update(mutation)
   with self.assertRaises(c.CalibrationStop):c.admission(m,self.env(),time.time())
 def test_exact_allocation_no_gpu_and_shell_deadline(self):
  c.admission(self.manifest(),self.env(),time.time())
  for key,value in (('SLURM_CPUS_PER_TASK','2'),('SLURM_MEM_PER_NODE','2048'),('SLURM_JOB_PARTITION','ah200'),('SLURM_JOB_GPUS','1'),('SLURM_JOB_ACCOUNT','other'),('STUDY_WALL_START','0')):
   env=self.env();env[key]=value
   with self.assertRaises(c.CalibrationStop):c.admission(self.manifest(),env,time.time())
 def test_frozen_packet_scope_and_hashes(self):
  m=json.loads((OUT/'manifest_local_held.json').read_text());c.validate_source(m)
  self.assertFalse(m['allocation_approved']);self.assertFalse(m['calibration_admission'])
  self.assertEqual(sum(c.BUDGET[k] for k in ('preparation','capture','toy','terminal')),65536)
  self.assertEqual((OUT/'source/calibration_input.bin').read_bytes(),c.PAYLOAD)
  self.assertLess((OUT/'source/calibration_input.bin').stat().st_size,128)
 def test_source_mutation_extra_file_and_symlink_rejected(self):
  for kind in ('mutation','extra','symlink'):
   with tempfile.TemporaryDirectory() as d:
    source=Path(d,'s');control=Path(d,'c');source.mkdir();control.mkdir();p=source/'own.py';p.write_text('known')
    m={'source_dir':str(source),'controller_dir':str(control),'source_hashes':{str(p):c.sha(p)}}
    if kind=='mutation':p.write_text('different')
    elif kind=='extra':(source/'json.py').write_text('unapproved')
    else:p.unlink();p.symlink_to('/outside/private')
    with self.assertRaises(c.CalibrationStop):c.validate_source(m)
 def toy_record(self):return {'kind':2,'payload_sha256':c.PAYLOAD_SHA256,'bytes_read':len(c.PAYLOAD),'started':1.,'completed':2.,'calibration_only':True,'qualification_established':False,'model_calls':0,'framework_imports':0,'GPU_allocations':0}
 def test_receipt_privacy_and_cap_before_open(self):
  for mutation in ({'secret':'credential'},{'payload_sha256':'/private/file'},{'framework_imports':1},{'started':float('nan')},{'bytes_read':True},{'qualification_established':True}):
   with tempfile.TemporaryDirectory() as d:
    r=self.toy_record();r.update(mutation);p=Path(d,'out.json')
    with self.assertRaises(c.CalibrationStop):c.write_metadata(p,'toy',r)
    self.assertFalse(p.exists())
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,'out.json')
   with self.assertRaises(c.CalibrationStop):c.write_metadata(p,'toy',self.toy_record(),limit=1)
   self.assertFalse(p.exists())
 def test_actual_stdlib_toy_receipt_success_and_input_failure(self):
  for valid in (True,False):
   with tempfile.TemporaryDirectory() as d:
    spool=Path(d);source=OUT/'source'
    if not valid:
     source=spool/'source';shutil.copytree(OUT/'source',source);(source/'calibration_input.bin').write_bytes(b'bad')
    r=subprocess.run([sys.executable,'-S',str(source/'reader32_calibration_toy_v69.py'),str(source),str(spool)],capture_output=True)
    self.assertEqual(r.returncode==0,valid);self.assertEqual((spool/'calibration_toy_receipt.json').exists(),valid)
    if valid:
     data=json.loads((spool/'calibration_toy_receipt.json').read_text());c.encode_metadata('toy',data)
     self.assertFalse(data['qualification_established'])
 def test_exact_toy_phase_protocol(self):
  with tempfile.TemporaryDirectory() as d:
   p=h.CalibrationParser(TraceSink(d))
   p.chunk('stdout',b'{"event":"calibration_toy_start","at":1}\n{"event":"calibration_toy_complete","at":2}\n');p.finish();p.sink.close()
  for text in ('{"event":"CPU_diagnostic_complete","at":1}','{"event":"calibration_toy_complete","at":1}','{"event":"calibration_toy_start","at":1,"secret":"credential"}'):
   with tempfile.TemporaryDirectory() as d:
    p=h.CalibrationParser(TraceSink(d))
    with self.assertRaises(TraceStop):p.chunk('stdout',(text+'\n').encode())
    p.sink.close()
 def test_toy_receipt_alone_does_not_prove_tracing(self):
  with tempfile.TemporaryDirectory() as d:
   c.write_metadata(Path(d,'calibration_toy_receipt.json'),'toy',self.toy_record())
   with self.assertRaisesRegex(c.CalibrationStop,'unverified'):h.validate_outcome(Path(d))
 def test_command_fixed_own_child_numeric_read_and_stdlib(self):
  command=h.build_command({'runtime':{'python':'/pinned/python'}},Path('/source'),Path('/spool'),4)
  self.assertIn('-S',command);self.assertIn('raw='+e.RAW,command);self.assertIn('/proc/self/fd/4',command)
  self.assertNotIn('-p',command);self.assertNotIn('sudo',command)
  self.assertIn('/source/reader32_calibration_toy_v69.py',command)
  self.assertNotIn('reader32_import_diagnostic_v68_receipt_safe.py',' '.join(command))
 def test_cli_held_has_no_outputs_or_launch(self):
  with tempfile.TemporaryDirectory() as d:
   r=subprocess.run([sys.executable,'-S',str(OUT/'controller/reader32_tracer_calibration_v69.py'),str(OUT/'manifest_local_held.json'),str(OUT/'source'),d],capture_output=True)
   self.assertNotEqual(r.returncode,0);self.assertEqual(list(Path(d).iterdir()),[])
 def test_imports_stdlib_or_frozen_local_helpers_only(self):
  allowed={'argparse','hashlib','json','math','os','pathlib','re','selectors','signal','subprocess','sys','time','reader32_calibration_contract_v69','reader32_trace_capture_v69_calibration','reader32_trace_privacy_v68'}
  for folder in (OUT/'source',OUT/'controller'):
   for p in folder.glob('*.py'):
    for n in ast.walk(ast.parse(p.read_text())):
     if isinstance(n,ast.Import):self.assertTrue(all(a.name in allowed for a in n.names),p.name)
     if isinstance(n,ast.ImportFrom):self.assertIn(n.module,allowed,p.name)
 def test_shell_resources_and_deadline(self):
  text=(OUT/'controller/reader32_tracer_calibration_v69_held.slurm').read_text()
  for line in ('#SBATCH --cpus-per-task=1','#SBATCH --mem=256M','#SBATCH --time=00:00:30','#SBATCH --account=ucb736_asc1','#SBATCH --partition=acpu','#SBATCH --qos=cpu-normal'):
   self.assertIn(line,text)
  self.assertIn('+20-time.time()',text);self.assertIn('--kill-after=2s',text);self.assertNotIn('--gres',text)

if __name__=='__main__':unittest.main()
