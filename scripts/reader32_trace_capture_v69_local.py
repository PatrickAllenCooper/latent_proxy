"""LOCAL compatibility revision. No allocation or framework execution admission."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time
from reader32_trace_privacy_v68 import TraceSink, TraceStop, MAX_LINE, require_admission, MODULE_LABELS

CALLS = 'openat,newfstatat,statx,read,pread64,mmap,futex,close'
RAW = 'read,pread64,mmap,futex,close'
EVENTS = {'CPU_diagnostic_start','dependency_callback_enter','tokenizer_class_import_start',
 'tokenizer_class_import_complete','torch_import_start','torch_import_complete',
 'model_class_import_start','model_class_import_complete','guarded_dependency_import_start',
 'guarded_dependency_import_complete','CPU_diagnostic_complete'}

CHANNEL_CODES = {'stdout':1,'stderr':2,'syscalls':3,'imports':4}
TYPE_CODES = {'tool_attach':1,'tool_permission_error':2,'tool_open_error':3,
 'tool_selector_error':4,'tool_error':5,'import_header':6,'import_row':7,
 'phase_json':8,'stack_frame':9,'python_bootstrap_error':10,'warning':11,
 'unknown':12,'syscall_row':13,'oversize':14,'invalid_utf8':15}


def classify_line(line, channel):
    if line.startswith(('strace:', '/usr/bin/strace:')):
        if re.fullmatch(r'(?:strace:|/usr/bin/strace:) Process [1-9]\d* attached(?: with [1-9]\d* threads)?',line): return 'tool_attach'
        if 'Operation not permitted' in line or 'Permission denied' in line: return 'tool_permission_error'
        if "Can't fopen" in line or "Can't stat" in line: return 'tool_open_error'
        if 'invalid system call' in line or 'invalid argument' in line: return 'tool_selector_error'
        return 'tool_error'
    if line.startswith('import time:'):
        return 'import_header' if line=='import time: self [us] | cumulative | imported package' else 'import_row'
    if line.startswith('{'): return 'phase_json'
    if line.startswith(('Fatal Python error:', 'Python path configuration:')): return 'python_bootstrap_error'
    if 'Warning:' in line or 'warning:' in line: return 'warning'
    if re.fullmatch(r'Timeout \(0:00:20\)!|Thread 0x[0-9a-f]+ \(most recent call first\):|  File ".*", line \d+ in [\w<>]+|\s*',line): return 'stack_frame'
    if channel=='syscalls': return 'syscall_row'
    return 'unknown'


class Parser:
    def __init__(self, sink):
        self.sink=sink; self.fds={}; self.pending={}; self.buffers={}
        self.channel_bytes={code:0 for code in CHANNEL_CODES.values()}
        self.channel_lines={code:0 for code in CHANNEL_CODES.values()}
        self.type_counts={code:0 for code in TYPE_CODES.values()}
        self.last_channel=0;self.last_type=0;self.control_lines=0
    def diagnostics(self):
        return {'channel_bytes':self.channel_bytes,'channel_lines':self.channel_lines,
          'line_type_counts':self.type_counts,'last_channel_code':self.last_channel,
          'last_type_code':self.last_type}
    def mark(self, channel, kind):
        self.last_channel=CHANNEL_CODES[channel]
        self.last_type=TYPE_CODES[kind]
        self.type_counts[self.last_type]+=1
    def chunk(self, channel, data):
        label=channel.split(':')[-1]
        if label not in CHANNEL_CODES: raise TraceStop('unknown channel')
        self.channel_bytes[CHANNEL_CODES[label]]+=len(data)
        buf=self.buffers.get(channel,b'')+data
        while b'\n' in buf:
            line,buf=buf.split(b'\n',1)
            if len(line)>MAX_LINE:
                self.mark(label,'oversize');raise TraceStop('oversize line')
            self.channel_lines[CHANNEL_CODES[label]]+=1
            try: decoded=line.decode('utf8',errors='strict')
            except UnicodeError:
                self.mark(label,'invalid_utf8');raise TraceStop('invalid encoding')
            self.line(label,decoded)
        if len(buf)>MAX_LINE:
            self.mark(label,'oversize');raise TraceStop('oversize line')
        self.buffers[channel]=buf
    def finish(self):
        if any(self.buffers.values()) or self.pending: raise TraceStop('incomplete trace')
    def line(self, channel, line):
        now=time.monotonic()
        kind=classify_line(line,channel);self.mark(channel,kind)
        if kind=='tool_attach':
            if channel!='stderr':raise TraceStop('tool notification wrong channel')
            # Fixed grammar from this launched tracer; no PID inspection/attach.
            # Notification alone establishes neither ownership nor useful tracing.
            self.control_lines+=1
            if self.control_lines>4096:raise TraceStop('control notification cap')
            return
        if kind.startswith('tool_'):raise TraceStop(kind)
        if kind in ('python_bootstrap_error','warning'):raise TraceStop(kind)
        if channel!='syscalls':
            if line.startswith('import time:'):
                if kind=='import_header': return
                match=re.fullmatch(r'import time:\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\w.]+)\s*',line)
                if not match: raise TraceStop('malformed importtime')
                # Only self time retained: nested cumulative times are not additive.
                self.sink.retain('imports','import',duration=int(match[1])/1e6,timestamp=now,module=match[3] if match[3] in MODULE_LABELS else 'other')
                return
            if line.startswith('{'):
                event=json.loads(line)
                if event.get('event') not in EVENTS: raise TraceStop('unknown phase')
                self.sink.retain('imports','import',timestamp=now,result=sorted(EVENTS).index(event['event']))
                return
            # Only traceback framing is accepted; never persist stack text/code.
            if re.fullmatch(r'Timeout \(0:00:20\)!|Thread 0x[0-9a-f]+ \(most recent call first\):|  File ".*", line \d+ in [\w<>]+|\s*',line):
                return
            raise TraceStop('unexpected child output')
        if re.search(r'Permission denied|Operation not permitted|ptrace|strace:',line):
            raise TraceStop('trace unavailable')
        match=re.fullmatch(r'(\d+)\s+(\d+\.\d+)\s+(.*)',line)
        if not match: raise TraceStop('malformed syscall prefix')
        pid,stamp,body=int(match[1]),float(match[2]),match[3]
        if re.fullmatch(r'\+\+\+ (?:exited with \d+|killed by SIG\w+(?: \(core dumped\))?) \+\+\+',body): return
        if body.endswith('<unfinished ...>'):
            if pid in self.pending: raise TraceStop('nested unfinished')
            
            if len(self.pending)>=4096: raise TraceStop('pending cap')
            prefix=body[:-len('<unfinished ...>')].strip()
            start=re.fullmatch(r'(\w+)\((.*)',prefix)
            if not start or start[1] not in CALLS.split(','): raise TraceStop('unexpected unfinished syscall')
            kind,args=start[1],start[2].rstrip().rstrip(')').rstrip()
            path=None;fd=None
            if kind in RAW.split(','):
                if not re.fullmatch(r'-?(?:0x[0-9a-f]+|\d+)(?:, -?(?:0x[0-9a-f]+|\d+))*,?',args): raise TraceStop('non-numeric unfinished arguments')
                if kind in ('read','pread64'):
                    fd=int(args.split(',')[0],0);path=self.fds.get((pid,fd),'[unknown]')
            else:
                quoted=re.findall(r'"((?:[^"\\]|\\.)*)"',args)
                if len(quoted)!=1 or '\\' in quoted[0] or '...' in args: raise TraceStop('unsafe unfinished path grammar')
                path=quoted[0]
            if kind!='close':self.sink.retain('syscalls',kind,path=path,timestamp=stamp,fd=fd,pid=pid,unfinished=True)
            self.pending[pid]=body[:-len('<unfinished ...>')]; return
        resumed=re.fullmatch(r'<\.\.\. (\w+) resumed>(.*)',body)
        if resumed:
            prior=self.pending.pop(pid,None)
            if prior is None or not prior.startswith(resumed[1]+'('): raise TraceStop('unmatched resume')
            body=prior+resumed[2]
        call=re.fullmatch(r'(\w+)\((.*)\)\s+=\s+(-?(?:0x[0-9a-f]+|\d+))(?:\s+[A-Z0-9_]+\s+\([^\n]*\))?\s+<(\d+\.\d+)>',body)
        if not call or call[1] not in CALLS.split(','): raise TraceStop('unexpected syscall')
        kind,args,result,duration=call[1],call[2],int(call[3],0) if call[3].startswith(('0x','-0x')) else int(call[3]),float(call[4])
        path=None; fd=None
        if kind in RAW.split(','):
            # raw format prevents any decoded buffers/addresses from retention.
            if not re.fullmatch(r'-?(?:0x[0-9a-f]+|\d+)(?:, -?(?:0x[0-9a-f]+|\d+))*',args):
                raise TraceStop('non-numeric raw arguments')
            fd=int(args.split(',')[0],0)
            if kind in ('read','pread64'): path=self.fds.get((pid,fd),'[unknown]')
            if kind=='close': self.fds.pop((pid,fd),None); return
            if kind not in ('read','pread64'): fd=None
        else:
            quoted=re.findall(r'"((?:[^"\\]|\\.)*)"',args)
            if len(quoted)!=1 or '\\' in quoted[0] or '...' in args: raise TraceStop('unsafe path grammar')
            path=quoted[0]
            if kind=='openat' and result>=0:
                
                if len(self.fds)>=4096: raise TraceStop('fd cap')
                fd=result; self.fds[(pid,fd)]=path
        self.sink.retain('syscalls',kind,duration=duration,result=result if kind!='mmap' else (0 if result>=0 else -1),path=path,timestamp=stamp,fd=fd,pid=pid)


def durable_json(path, data):
    encoded=(json.dumps(data,indent=2)+'\n').encode()
    if len(encoded)>46080: raise TraceStop('metadata cap')
    with Path(path).open('xb') as out:
        out.write(encoded);out.flush();os.fsync(out.fileno())
    directory=os.open(str(Path(path).parent),os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def capture(command, spool, roots, deadline, trace_fd=None, limit=10*1024*1024, exact_paths=None):
    """No shell, no attach; only this child's newly created group is signalled."""
    sink=TraceSink(spool, roots, limit, exact_paths); parser=Parser(sink)
    proc=None; reason='complete';started=time.monotonic(); sel=selectors.DefaultSelector()
    owned_trace_fds=set(() if trace_fd is None else trace_fd)
    previous=signal.getsignal(signal.SIGTERM)
    def deadline_signal(signum,frame): raise TraceStop('external deadline')
    signal.signal(signal.SIGTERM,deadline_signal)
    try:
        if time.monotonic()>=deadline: raise TraceStop('startup timeout')
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            start_new_session=True,pass_fds=(() if trace_fd is None else (trace_fd[1],)))
        if trace_fd is not None:
            os.close(trace_fd[1]);owned_trace_fds.discard(trace_fd[1]); sel.register(trace_fd[0],selectors.EVENT_READ,'syscalls')
        sel.register(proc.stdout,selectors.EVENT_READ,'stdout')
        sel.register(proc.stderr,selectors.EVENT_READ,'stderr')
        while sel.get_map():
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TraceStop('startup timeout')
            for key,_ in sel.select(min(.05,remaining)):
                data=os.read(key.fd,4096)
                if not data: sel.unregister(key.fileobj)
                else: parser.chunk(str(key.fd)+':'+key.data,data)
        parser.finish()
        if proc.wait(timeout=max(.001,deadline-time.monotonic())): raise TraceStop('child failure')
    except TraceStop as error:
        reason=str(error)
    except (ValueError,UnicodeError,subprocess.TimeoutExpired,OSError):
        # Do not retain exception text: it may contain raw paths/output.
        reason='capture_stopped'
    finally:
        if proc is not None:
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
            # Parent exit does not establish that descendants exited. This is
            # only the group created by start_new_session above.
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        signal.signal(signal.SIGTERM,previous)
        sel.close()
        if proc is not None:
            proc.stdout.close();proc.stderr.close()
        if trace_fd is not None:
            for fd in owned_trace_fds:
                try: os.close(fd)
                except OSError: pass
        custody=sink.close()
        durable_json(Path(spool)/'capture_receipt.json',dict(custody,reason=reason,
          elapsed=time.monotonic()-started,wall_time=time.time(),monotonic_time=time.monotonic(),phase_codes=sorted(EVENTS),exit_code=None if proc is None else proc.returncode,
          metadata_limit_bytes=46080,raw_trace_persisted=False,diagnostics=parser.diagnostics()))
    return reason


def verify_binding(m):
    for filename,digest in m['source_hashes'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest: raise TraceStop('source binding')
    source=Path(m['diagnostic_source']); driver=source/'reader32_import_diagnostic_v68_receipt_safe.py'
    def imports(path,name):
        tree=ast.parse(path.read_text());f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        return [ast.dump(n,include_attributes=False) for n in ast.walk(f) if isinstance(n,(ast.Import,ast.ImportFrom))]
    if imports(driver,'dependencies')!=imports(source/'run_reader32_v65.py','deps'): raise TraceStop('import sequence changed')
    baseline=source/'reader32_import_diagnostic_v67_cpu.py'
    old=ast.parse(baseline.read_text());new=ast.parse(driver.read_text())
    of=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nf=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    marker=next(i for i,n in enumerate(of.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='receipt' for t in n.targets))
    dump=lambda nodes:[ast.dump(n,include_attributes=False) for n in nodes]
    if dump(of.body[:marker])!=dump(nf.body[:marker]) or dump(old.body[:old.body.index(of)])!=dump(new.body[:new.body.index(nf)]):raise TraceStop('baseline guard prefix changed')
    if not isinstance(nf.body[marker],ast.ImportFrom) or nf.body[marker].module!='reader32_safe_receipt_v68':raise TraceStop('safe receipt boundary changed')
    if m['metadata_budget']!={'combined_bytes':65536,'runtime_stage_max':15360,'capture_receipt_max':46080,'CPU_import_receipt_max':3072,'wrapper_receipt_max':1024}:raise TraceStop('metadata budget changed')
    return source,driver

def verify_runtime_descriptor(m, env, spool):
    descriptor=m['runtime_descriptor']
    receipt=Path(descriptor['receipt_path'])
    if hashlib.sha256(receipt.read_bytes()).hexdigest()!=descriptor['receipt_sha256']:
        raise TraceStop('runtime receipt binding')
    job=env.get('SLURM_JOB_ID','')
    if not job.isdigit(): raise TraceStop('job identity')
    temporary=Path(env.get('SLURM_TMPDIR') or '/tmp')
    if not temporary.is_absolute() or '..' in temporary.parts: raise TraceStop('temporary root')
    expected=temporary/('lp-reader32-import-v68-'+job)
    if env.get('LOCAL_RUNTIME')!=str(expected): raise TraceStop('staged root binding')
    stage=Path(spool)/'runtime_stage.json'
    if stage.stat().st_size>15360: raise TraceStop('stage metadata cap')
    event=json.loads(stage.read_text())
    if event.get('event')!='node_local_runtime_ready' or event.get('path')!=str(expected) or event.get('archive_sha256')!=descriptor['archive_sha256']:
        raise TraceStop('stage receipt binding')
    # Flush the bounded known extractor receipt; no raw stderr is stored.
    with stage.open('rb') as f: os.fsync(f.fileno())
    exact={}
    for entry in m['shared_exact_files']:
        path=Path(entry['path']);st=path.stat()
        if st.st_size!=entry['bytes'] or st.st_mtime_ns!=entry['mtime_ns']:
            raise TraceStop('shared file metadata changed')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise TraceStop('shared file content changed')
        exact[str(path)]=entry['label']
        # Only aliases explicitly observed as identical are admitted, not discovery.
        for alias in entry.get('verified_aliases',[]):
            if not path.samefile(alias): raise TraceStop('alias binding')
            exact[alias]=entry['label']
    return str(expected/'transformers'),exact

def build_command(manifest, source, driver, spool, output_fd):
    if type(output_fd) is not int or output_fd<3:raise TraceStop('invalid own pipe fd')
    return ['/usr/bin/strace','-f','-ttt','-T','-s','8192','-e','trace='+CALLS,
      '-e','raw='+RAW,'-o','/proc/self/fd/'+str(output_fd),manifest['python'],
      '-X','importtime','-u',str(driver),str(source),str(spool)]

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('spool',type=Path);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    m=json.loads(a.manifest.read_text()); start=float(os.environ.get('STUDY_WALL_START','0'))
    if m.get('local_compatibility_revision_admission') is not True: raise TraceStop('local revision held')
    require_admission(m,time.time(),start)
    if m.get('remote_materialization_ready') is not True: raise TraceStop('materialization held')
    source,driver=verify_binding(m)
    if a.preflight:
        receipt=Path(m['runtime_descriptor']['receipt_path'])
        if hashlib.sha256(receipt.read_bytes()).hexdigest()!=m['runtime_descriptor']['receipt_sha256']: raise TraceStop('runtime receipt binding')
        return
    roots=[entry['path'] for entry in m['runtime_roots'] if entry.get('verified') is True]
    runtime_root,exact=verify_runtime_descriptor(m,os.environ,a.spool)
    roots.append(runtime_root)
    r,w=os.pipe()
    command=build_command(m,source,driver,a.spool,w)
    result=capture(command,a.spool,roots,time.monotonic()+max(0,start+90-time.time()),(r,w),exact_paths=exact)
    verify_binding(m)  # Recheck every frozen source/controller file after capture.
    raise SystemExit(0 if result=='complete' else 124)

if __name__=='__main__': main()
