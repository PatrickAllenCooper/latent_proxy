import ast,hashlib,json
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reader32_safe_receipt_v68 as m
from prepare_reader32_safe_driver_v68 import verify_output_only,BASE,OUT

class SafeReceiptFixtures(unittest.TestCase):
 def receipt(self):
  return dict({k:True for k in m.TRUE_KEYS},**{k:False for k in m.FALSE_KEYS},**{k:0 for k in m.ZERO_KEYS},**{'started':1.,'completed':2.},**{k:'a'*64 for k in m.HASH_KEYS})
 def test_success_custody_cap(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,'CPU_import_receipt.json');r=m.write_receipt(p,self.receipt())
   self.assertLessEqual(p.stat().st_size,3072);self.assertEqual(r['sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
   self.assertEqual(set(json.loads(p.read_text())),m.KEYS)
 def test_raw_fields_rejected_before_any_file(self):
  for key,value in [('captures',{'/private/secret':'credentials'}),('inspection',{'directory_signatures':'private'}),('hostname','secret'),('torch_version','text'),('environment',{'TOKEN':'secret'})]:
   with tempfile.TemporaryDirectory() as d:
    r=self.receipt();r[key]=value;p=Path(d,'out.json')
    with self.assertRaises(ValueError):m.write_receipt(p,r)
    self.assertFalse(p.exists())
 def test_oversize_before_any_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,'out.json')
   with self.assertRaises(ValueError):m.write_receipt(p,self.receipt(),max_bytes=10)
   self.assertFalse(p.exists())
 def test_types_privacy_success_failure(self):
  for k,v in [('model_calls',True),('complete',False),('started',float('nan')),('completed',float('inf')),('CPU_receipt_sha256','/private/secret'),('qualification_established',True)]:
   with tempfile.TemporaryDirectory() as d:
    r=self.receipt();r[k]=v;p=Path(d,'out.json')
    with self.assertRaises(ValueError):m.write_receipt(p,r)
    self.assertFalse(p.exists())
 def test_exclusive_write(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,'out.json');p.write_text('preserved')
   with self.assertRaises(FileExistsError):m.write_receipt(p,self.receipt())
   self.assertEqual(p.read_text(),'preserved')
 def test_exact_output_only_derivation_and_guard_mutation(self):
  old=(BASE/'reader32_import_diagnostic_v67_cpu.py').read_text()
  new=(OUT/'source/reader32_import_diagnostic_v68_receipt_safe.py').read_text()
  self.assertTrue(verify_output_only(old,new)['dependency_AST_identical'])
  for mutation in ('assert time.time()<start+90','assert f[\'CPU_admission\'] is True','assert sha(os.environ[\'RUNTIME_RECEIPT\'])==old[\'runtime_receipt_sha256\']'):
   with self.assertRaises(ValueError):verify_output_only(old,new.replace(mutation,'pass'))
 def test_no_raw_receipt_generation(self):
  text=(OUT/'source/reader32_import_diagnostic_v68_receipt_safe.py').read_text()
  tree=ast.parse(text);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  receipt=next(n.value for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='receipt' for t in n.targets))
  self.assertEqual({k.value for k in receipt.keys},m.KEYS)
  tail=text[text.index(' from reader32_safe_receipt_v68'):]
  for unsafe in ('captures','inspection','hostname','torch.__version__','directory_signatures','json.dumps'):self.assertNotIn(unsafe,tail)

if __name__=='__main__':unittest.main()

class GuardedDriverToyFixtures(unittest.TestCase):
 def run_toy(self,folder,corrupt_runtime=False,mutate_after_import=False):
  from types import SimpleNamespace
  from contextlib import nullcontext
  from unittest.mock import Mock
  root=Path(folder);source=root/'source';source.mkdir();spool=root/'spool';spool.mkdir();snapshot=root/'snapshot';snapshot.mkdir()
  baseline=BASE/'reader32_import_diagnostic_v67_cpu.py'
  (source/baseline.name).write_bytes(baseline.read_bytes())
  (source/'run_reader32_v65.py').write_text('# toy AST guard input')
  runtime=root/'runtime.json';runtime.write_text('{}')
  binary=root/'python';binary.write_bytes(b'toy binary')
  digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
  old={'complete':True,'python':'fixture','python_sha256':digest(binary),'runtime_receipt_sha256':'0'*64 if corrupt_runtime else digest(runtime),'files':{}}
  (source/'cpu_receipt.json').write_text(json.dumps(old))
  freeze={'CPU_admission':True,'resources':{'CPUs':1,'mem_GiB':2,'wall_seconds':120,'check_seconds':90,'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal','GPUs':0,'attempts':1},'snapshot':str(snapshot),'files':{baseline.name:digest(source/baseline.name)}}
  (source/'diagnostic_freeze.json').write_text(json.dumps(freeze))
  env={'PYTHONDONTWRITEBYTECODE':'1','STUDY_WALL_START':'0','SLURM_JOB_ID':'123','SLURM_JOB_PARTITION':'acpu','SLURM_CPUS_PER_TASK':'1','SLURM_MEM_PER_NODE':'2048','RUNTIME_RECEIPT':str(runtime)}
  parser=Mock();parser.parse_args.return_value=SimpleNamespace(source=source,spool=spool)
  phases=[]
  def dependencies():
   if mutate_after_import:(source/baseline.name).write_text('changed')
   return type('Qwen2Tokenizer',(),{}),SimpleNamespace(),type('Qwen2ForCausalLM',(),{})
  namespace={'argparse':SimpleNamespace(ArgumentParser=lambda:parser),'Path':Path,'os':SimpleNamespace(environ=env),'sys':SimpleNamespace(version='fixture',executable=str(binary)), 'time':SimpleNamespace(time=lambda:10.),'json':json,'sha':digest,'imports':lambda *a:['same AST fixture'],'emit':lambda event,**kw:phases.append(event),'dependencies':dependencies,'import_trace':nullcontext,'indexed_import':lambda callback:(callback(),[{'directory_signatures':{'/private/secret':'credentials'}}]),'inspect_with_index':lambda callback:(callback(),{'directory_signatures':{'/outside/private':'secret'}}),'__file__':str(OUT/'source/reader32_import_diagnostic_v68_receipt_safe.py')}
  tree=ast.parse(Path(namespace['__file__']).read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  exec(compile(ast.Module(body=[node],type_ignores=[]),'<toy-main>','exec'),namespace)
  if corrupt_runtime or mutate_after_import:
   with self.assertRaises(AssertionError):namespace['main']()
   self.assertFalse((spool/'CPU_import_receipt.json').exists())
  else:
   namespace['main']();raw=(spool/'CPU_import_receipt.json').read_text()
   self.assertNotIn('secret',raw);self.assertNotIn('directory_signatures',raw)
   self.assertTrue(json.loads(raw)['guarded_imports_completed']);self.assertEqual(phases[-1],'CPU_diagnostic_complete')
 def test_success_raw_metadata_never_serialized(self):
  with tempfile.TemporaryDirectory() as d:self.run_toy(d)
 def test_guard_failure_no_receipt(self):
  with tempfile.TemporaryDirectory() as d:self.run_toy(d,corrupt_runtime=True)
 def test_post_import_source_guard_still_enforced(self):
  with tempfile.TemporaryDirectory() as d:self.run_toy(d,mutate_after_import=True)

class PacketFixtures(unittest.TestCase):
 def test_binding_guards_budget_and_frozen_controllers(self):
  import reader32_trace_capture_v68_receipt_safe as c
  manifest=json.loads((OUT/'manifest_local_held.json').read_text());c.verify_binding(manifest)
  self.assertFalse(manifest['cpu_admission']);self.assertFalse(manifest['remote_materialization_ready'])
  budget=manifest['metadata_budget'];self.assertEqual(sum(v for k,v in budget.items() if k!='combined_bytes'),65536)
  self.assertEqual(budget['CPU_import_receipt_max'],m.MAX_BYTES)
  remote=json.loads((OUT/'manifest_remote_held.json').read_text())
  self.assertNotIn('/Users/',json.dumps(remote['source_hashes']))
  for p in (OUT/'controller').iterdir():
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),remote['source_hashes']['/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68/revision-receipt-safe/controller/'+p.name])
 def test_environment_resource_and_final_guard_checks(self):
  wrapper=(OUT/'controller/reader32_import_trace_v68_receipt_safe.slurm').read_text()
  self.assertIn('export PYTHONPATH="$LOCAL_RUNTIME:$SOURCE"',wrapper);self.assertIn('cd "$SOURCE"',wrapper)
  for x in ('--cpus-per-task=1','--mem=2G','--time=00:02:00','--account=ucb736_asc1','--qos=cpu-normal'):self.assertIn(x,wrapper)
  source=(OUT/'controller/reader32_trace_capture_v68_receipt_safe.py').read_text()
  self.assertIn('verify_binding(m)  # Recheck',source)
  self.assertIn("driver=source/'reader32_import_diagnostic_v68_receipt_safe.py'",source)
  self.assertNotIn("driver=source/'reader32_import_diagnostic_v67_cpu.py'",source)
