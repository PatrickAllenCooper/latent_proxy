"""Stdlib-only Python audit calibration; does not observe native syscalls."""
import hashlib,json,math,os,re,subprocess,sys,time
from pathlib import Path
RESOURCES={'cpus':1,'memory_MiB':256,'scheduler_wall_seconds':60,'application_wall_seconds':30,'check_seconds':20,'attempts':1,'gpus':0}
PAYLOAD=b'latent_proxy owned tracer calibration\n'*2
DIGEST=hashlib.sha256(PAYLOAD).hexdigest()
class Stop(RuntimeError):pass

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def save(path,record,cap):
 data=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()
 if len(data)>cap:raise Stop('metadata cap')
 with Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 fd=os.open(str(Path(path).parent),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def child(source,spool):
 target=source/'calibration_input.bin';count=[0]
 def audit(event,args):
  if event=='open' and isinstance(args[0],str) and args[0]==str(target):count[0]+=1
 sys.addaudithook(audit)
 start=time.time();print(json.dumps({'event':'application_start','at':start}),flush=True)
 with target.open('rb') as f:data=f.read(len(PAYLOAD)+1)
 if data!=PAYLOAD:raise Stop('input mismatch')
 receipt={'method':'python_audit_and_owned_read','audit_open_events':count[0],'bytes_read':len(data),'payload_sha256':hashlib.sha256(data).hexdigest(),'started':start,'completed':time.time(),'native_syscalls_observed':False,'model_calls':0,'framework_imports':0,'GPU_allocations':0}
 save(spool/'application_child.json',receipt,2048)
 print(json.dumps({'event':'application_complete','at':time.time()}),flush=True)

def validate_child(r,events):
 expected={'method','audit_open_events','bytes_read','payload_sha256','started','completed','native_syscalls_observed','model_calls','framework_imports','GPU_allocations'}
 if set(r)!=expected or r['method']!='python_audit_and_owned_read':raise Stop('schema')
 if type(r['audit_open_events']) is not int or r['audit_open_events']<1 or type(r['bytes_read']) is not int or r['bytes_read']!=len(PAYLOAD) or r['payload_sha256']!=DIGEST:raise Stop('read evidence')
 if r['native_syscalls_observed'] is not False or any(type(r[k]) is not int or r[k]!=0 for k in ('model_calls','framework_imports','GPU_allocations')):raise Stop('scope')
 times=[r['started'],r['completed']]
 if any(type(x) not in (int,float) or not math.isfinite(x) for x in times) or times[1]<times[0]:raise Stop('times')
 if len(events)!=2 or [x.get('event') for x in events]!=['application_start','application_complete'] or any(set(x)!={'event','at'} for x in events):raise Stop('phases')
 if any(type(x['at']) not in (float,int) or not math.isfinite(x['at']) for x in events) or events[1]['at']<events[0]['at']:raise Stop('phase times')

def admit(m,env):
 if m.get('admission') is not True or m.get('resources')!=RESOURCES or m.get('approval')!='user_20261024_application_instrumentation' or m.get('attempts_used')!=0:raise Stop('held')
 expected={'SLURM_JOB_PARTITION':'acpu','SLURM_CPUS_PER_TASK':'1','SLURM_MEM_PER_NODE':'256','SLURM_JOB_ACCOUNT':'ucb736_asc1'}
 if any(env.get(k)!=v for k,v in expected.items()):raise Stop('allocation')
 if any(env.get(k,'') not in ('','0') for k in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE')):raise Stop('GPU')
 start=float(env.get('STUDY_WALL_START','0'))
 if not math.isfinite(start) or time.time()>=start+20:raise Stop('deadline')
 return start+20

def execute(m,source,spool,env):
 deadline=admit(m,env);started=time.time();stage='source_check'
 if source.is_symlink() or not source.is_absolute() or '..' in source.parts:raise Stop('source root')
 if set(p.name for p in source.iterdir())!=set(m['source_hashes']):raise Stop('inventory')
 for name,digest in m['source_hashes'].items():
  p=source/name
  if p.is_symlink() or p.stat().st_size>65536 or sha(p)!=digest:raise Stop('source identity')
 python=m['python']
 if python!='/projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python' or sha(python)!=m['python_sha256']:raise Stop('python identity')
 if any(spool.iterdir()):
  # Slurm creates empty stdout/stderr before application launch.
  if any(p.name in ('application_child.json','application_result.json') for p in spool.iterdir()):raise Stop('attempt exists')
 stage='child';result={'method':'python_audit_and_owned_read','native_syscalls_observed':False,'model_calls':0,'framework_imports':0,'GPU_allocations':0,'started':started,'success':False}
 try:
  run=subprocess.run([python,'-S','-B',str(source/'application_calibration_v76.py'),'child',str(source),str(spool)],capture_output=True,timeout=max(.001,deadline-time.time()))
  result['child_exit_code']=run.returncode
  diagnostic=run.stderr[:2048].decode('utf8','replace');diagnostic=re.sub(r'/[A-Za-z0-9_.+/@=-]+','[path]',diagnostic);result['stderr']=diagnostic
  if run.returncode:raise Stop('child failure')
  if len(run.stdout)>4096:raise Stop('phase cap')
  events=[json.loads(line) for line in run.stdout.splitlines()]
  p=spool/'application_child.json'
  if p.stat().st_size>2048:raise Stop('child cap')
  r=json.loads(p.read_text());validate_child(r,events)
  if time.time()>=deadline:raise Stop('final deadline')
  result.update(success=True,events=events,child_receipt_sha256=sha(p),source_rechecked=all(sha(source/n)==d for n,d in m['source_hashes'].items()))
  if not result['source_rechecked']:raise Stop('source changed')
 except (Stop,ValueError,OSError,subprocess.TimeoutExpired) as error:
  result.update(success=False,error_category=type(error).__name__,stage=stage)
 result['completed']=time.time();save(spool/'application_result.json',result,8192)
 return int(not result['success'])
if __name__=='__main__':
 if sys.argv[1]=='child':child(Path(sys.argv[2]),Path(sys.argv[3]))
 else:
  try:status=execute(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]),Path(sys.argv[3]),os.environ)
  except Exception:status=2
  raise SystemExit(status)
