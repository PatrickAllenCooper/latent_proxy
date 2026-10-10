"""Held stdlib-only observed identity diagnostic. Never updates trusted bindings."""
import hashlib,json,math,os,re,sys,time
from pathlib import Path
RESOURCES={'cpus':1,'memory_MiB':256,'scheduler_wall_seconds':60,'application_wall_seconds':30,'check_seconds':20,'attempts':1,'gpus':0}
PATHS={'python':'/projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python','strace':'/usr/bin/strace'}
CAP=1024
READ_CAP=64*1024*1024
CONSUMED={'user_20261009_correct_and_proceed','user_20261009_implement_next_cycle'}
class Held(RuntimeError):pass

def admit(m,env,now):
 if m.get('identity_diagnostic_admission') is not True or m.get('allocation_approved') is not True:raise Held('held')
 if m.get('purpose')!='observed_runtime_identity_only' or m.get('resources')!=RESOURCES or m.get('attempts_used')!=0 or m.get('retry') is not False:raise Held('contract')
 a=m.get('fresh_approval_message_id')
 if type(a) is not str or not re.fullmatch('[A-Za-z0-9_-]{1,128}',a) or a in CONSUMED:raise Held('fresh approval')
 start=float(env.get('STUDY_WALL_START','0'))
 if not math.isfinite(start) or now>=start+20:raise Held('deadline')
 expected={'SLURM_JOB_PARTITION':'acpu','SLURM_CPUS_PER_TASK':'1','SLURM_MEM_PER_NODE':'256','SLURM_JOB_ACCOUNT':'ucb736_asc1'}
 if any(env.get(k)!=v for k,v in expected.items()) or not str(env.get('SLURM_JOB_ID','')).isdigit():raise Held('allocation')
 if any(env.get(k,'') not in ('','0') for k in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS','SLURM_GPUS_ON_NODE')):raise Held('GPU')
 if m.get('runtime_paths')!=PATHS:raise Held('runtime scope')
 if set(m.get('expected_sha256',{}))!=set(PATHS) or any(type(x) is not str or not re.fullmatch('[0-9a-f]{64}',x) for x in m['expected_sha256'].values()):raise Held('trusted bindings absent')
 expected_source=m.get('script_sha256')
 if type(expected_source) is not str or not re.fullmatch('[0-9a-f]{64}',expected_source):raise Held('source binding absent')
 if Path(__file__).stat().st_size>16384 or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=expected_source:raise Held('source identity')
 return start+20

def observe(path,expected,deadline,budget):
 # Error enums: 0 match; 1 mismatch; 2 IO failure; 3 read cap; 4 deadline.
 h=hashlib.sha256();count=0
 try:
  size=Path(path).stat().st_size
  if size>budget:return {'expected':expected,'observed':None,'bytes':0,'code':3},0
  with open(path,'rb') as f:
   while True:
    if time.time()>=deadline:return {'expected':expected,'observed':None,'bytes':count,'code':4},count
    chunk=f.read(min(1024*1024,budget-count+1))
    if not chunk:break
    count+=len(chunk)
    if count>budget:return {'expected':expected,'observed':None,'bytes':count,'code':3},count
    h.update(chunk)
 except OSError:return {'expected':expected,'observed':None,'bytes':count,'code':2},count
 digest=h.hexdigest()
 return {'expected':expected,'observed':digest,'bytes':count,'code':int(digest!=expected)},count

def encode(r):
 if set(r)!={'kind','started','completed','python','strace','trusted_bindings_changed','qualification_established','model_calls','GPU_allocations'}:raise Held('schema')
 if r['kind']!=72 or r['trusted_bindings_changed'] is not False or r['qualification_established'] is not False or type(r['model_calls']) is not int or r['model_calls']!=0 or type(r['GPU_allocations']) is not int or r['GPU_allocations']!=0:raise Held('fixed values')
 if any(type(r[k]) not in (int,float) or not math.isfinite(r[k]) or not 0<=r[k]<=1e12 for k in ('started','completed')) or r['completed']<r['started']:raise Held('time')
 for name in PATHS:
  row=r[name]
  if set(row)!={'expected','observed','bytes','code'}:raise Held('row')
  if type(row['expected']) is not str or not re.fullmatch('[0-9a-f]{64}',row['expected']):raise Held('hash')
  if row['observed'] is not None and (type(row['observed']) is not str or not re.fullmatch('[0-9a-f]{64}',row['observed'])):raise Held('hash')
  if type(row['bytes']) is not int or not 0<=row['bytes']<=READ_CAP+1 or type(row['code']) is not int or not 0<=row['code']<=4:raise Held('numeric')
  if row['code'] in (0,1) and (row['observed'] is None or row['code']!=int(row['observed']!=row['expected'])):raise Held('comparison')
 data=(json.dumps(r,sort_keys=True,allow_nan=False)+'\n').encode()
 if len(data)>CAP:raise Held('receipt cap')
 return data

def execute(m,spool,env):
 deadline=admit(m,env,time.time());started=time.time();rows={};budget=READ_CAP
 for name,path in PATHS.items():
  rows[name],count=observe(path,m['expected_sha256'][name],deadline,budget);budget-=count
 record=dict(kind=72,started=started,completed=time.time(),**rows,trusted_bindings_changed=False,qualification_established=False,model_calls=0,GPU_allocations=0)
 data=encode(record)
 with (Path(spool)/'runtime_identity.json').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 fd=os.open(str(spool),os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return int(any(r['code'] for r in rows.values()))
if __name__=='__main__':
 try:status=execute(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]),os.environ)
 except Exception:status=2
 raise SystemExit(status)
