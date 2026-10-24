"""Separate calibration capture engine; original v69 compatibility snapshot preserved."""
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time
from reader32_trace_privacy_v68 import TraceSink, TraceStop, MAX_LINE, MODULE_LABELS

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


def capture(command, spool, roots, deadline, trace_fd=None, limit=10*1024*1024, exact_paths=None, parser_factory=Parser, phase_codes=None):
    """No shell, no attach; only this child's newly created group is signalled."""
    sink=TraceSink(spool, roots, limit, exact_paths); parser=parser_factory(sink)
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
          elapsed=time.monotonic()-started,wall_time=time.time(),monotonic_time=time.monotonic(),phase_codes=sorted(EVENTS) if phase_codes is None else phase_codes,exit_code=None if proc is None else proc.returncode,
          metadata_limit_bytes=46080,raw_trace_persisted=False,diagnostics=parser.diagnostics()))
    return reason


# Library only: execution is gated by the separate calibration harness.
