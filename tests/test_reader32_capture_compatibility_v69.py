import importlib.util,json,os
from pathlib import Path
import sys,tempfile,time,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reader32_trace_capture_v69_local as m
import reader32_trace_capture_v68_receipt_safe as old
from reader32_trace_privacy_v68 import TraceSink,TraceStop

class CompatibilityFixtures(unittest.TestCase):
 def parser(self,d):return m.Parser(TraceSink(d,('/verified',)))
 def test_absolute_tool_prefix_reproduces_old_branch(self):
  with tempfile.TemporaryDirectory() as d:
   s=TraceSink(d)
   with self.assertRaisesRegex(TraceStop,'unexpected child output'):
    old.Parser(s).chunk('stderr:imports',b'/usr/bin/strace: ptrace: Operation not permitted\n')
   s.close()
 def test_denial_open_selector_classified_and_stopped_no_raw(self):
  lines=[('/usr/bin/strace: ptrace: Operation not permitted','tool_permission_error'),('strace: Permission denied /private/secret','tool_permission_error'),('/usr/bin/strace: Can\'t fopen /private/credential','tool_open_error'),('/usr/bin/strace: invalid system call secret','tool_selector_error'),('/usr/bin/strace: mystery credentials','tool_error')]
  for text,reason in lines:
   with tempfile.TemporaryDirectory() as d:
    p=self.parser(d)
    with self.assertRaisesRegex(TraceStop,reason):p.chunk('7:stderr',(text+'\n').encode())
    self.assertEqual(p.diagnostics()['last_channel_code'],2)
    self.assertEqual(p.diagnostics()['last_type_code'],m.TYPE_CODES[reason])
    self.assertNotIn('secret',json.dumps(p.diagnostics()));p.sink.close()
 def test_bounded_exact_bootstrap_not_general_ignore(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.parser(d)
   p.chunk('stderr',b'/usr/bin/strace: Process 123 attached\nstrace: Process 456 attached with 2 threads\n')
   self.assertEqual(p.type_counts[m.TYPE_CODES['tool_attach']],2)
   self.assertEqual(p.sink.used,0)
   with self.assertRaises(TraceStop):p.chunk('stderr',b'/usr/bin/strace: Process 123 attached secret\n')
   p.sink.close()
 def test_import_headers_and_rows_are_strict(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.parser(d)
   for text in ('import time: self [us] | cumulative | imported package','import time:      12 |         42 |   torch.distributed.rpc','import time: 1 | 2 | _io'):
    p.chunk('stderr',(text+'\n').encode())
   with self.assertRaisesRegex(TraceStop,'malformed importtime'):p.chunk('stderr',b'import time: secret self [us]\n')
   p.sink.close()
 def test_phase_bootstrap_warning_unknown_privacy(self):
  for text,reason in [('Fatal Python error: secret','python_bootstrap_error'),('/private/secret.py:1: RuntimeWarning: credentials','warning'),('unknown credential','unexpected child output'),('{"event":"evil"}','unknown phase')]:
   with tempfile.TemporaryDirectory() as d:
    p=self.parser(d)
    with self.assertRaisesRegex(TraceStop,reason):p.chunk('stdout',(text+'\n').encode())
    p.sink.close()
 def test_chunk_stream_separation_oversize_invalid_utf8(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.parser(d);p.chunk('stdout',b'{"event":');p.chunk('stderr',b'import time: 1 | 2 | toy\n');p.chunk('stdout',b'"CPU_diagnostic_start"}\n');p.finish()
   with self.assertRaises(TraceStop):p.chunk('stderr',b'x'*8193)
   self.assertEqual(p.last_type,m.TYPE_CODES['oversize']);p.sink.close()
  with tempfile.TemporaryDirectory() as d:
   p=self.parser(d)
   with self.assertRaises(TraceStop):p.chunk('stderr',b'\xff\n')
   self.assertEqual(p.last_type,m.TYPE_CODES['invalid_utf8']);p.sink.close()
 def test_toy_capture_records_numeric_diagnostics_only(self):
  with tempfile.TemporaryDirectory() as d:
   code='import sys;sys.stderr.write("/usr/bin/strace: Can\\\'t fopen /private/secret\\n")'
   reason=m.capture([sys.executable,'-c',code],d,(),time.monotonic()+2)
   self.assertEqual(reason,'tool_open_error')
   receipt=Path(d,'capture_receipt.json').read_text()
   self.assertNotIn('private',receipt);self.assertNotIn('secret',receipt)
   self.assertEqual(json.loads(receipt)['diagnostics']['last_channel_code'],2)

if __name__=='__main__':unittest.main()

class PipeAndEnvelopeFixtures(unittest.TestCase):
 def test_real_own_child_pipe_startup_all_channels(self):
  with tempfile.TemporaryDirectory() as d:
   r,w=os.pipe()
   code='import os,sys;os.write(int(sys.argv[1]),b\'123 10.000001 openat(AT_FDCWD, "/verified/mock.py", O_RDONLY) = 3 <0.000001>\\n\');os.close(int(sys.argv[1]));sys.stderr.write("import time: 1 | 2 | toy\\n");print(\'{"event":"CPU_diagnostic_start"}\')'
   self.assertEqual(m.capture([sys.executable,'-c',code,str(w)],d,('/verified',),time.monotonic()+2,(r,w)),'complete')
   receipt=json.loads(Path(d,'capture_receipt.json').read_text());obs=receipt['diagnostics']
   self.assertEqual(obs['channel_lines']['1'],1);self.assertEqual(obs['channel_lines']['2'],1);self.assertEqual(obs['channel_lines']['3'],1)
   self.assertLess(Path(d,'capture_receipt.json').stat().st_size,46080)
 def test_command_exact_numeric_raw_own_fd_no_attach(self):
  command=m.build_command({'python':'/pinned/python'},'/source','/source/safe_driver.py','/spool',4)
  self.assertEqual(command,['/usr/bin/strace','-f','-ttt','-T','-s','8192','-e','trace='+m.CALLS,'-e','raw='+m.RAW,'-o','/proc/self/fd/4','/pinned/python','-X','importtime','-u','/source/safe_driver.py','/source','/spool'])
  self.assertNotIn('-p',command);self.assertNotIn('sudo',command)
  with self.assertRaises(TraceStop):m.build_command({'python':'x'},'s','d','o',2)
 def test_local_revision_hold_overrides_old_approved_manifest(self):
  import subprocess
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,'manifest.json');p.write_text('{"cpu_admission":true,"remote_materialization_ready":true}')
   result=subprocess.run([sys.executable,m.__file__,str(p),d],capture_output=True)
   self.assertNotEqual(result.returncode,0);self.assertFalse(Path(d,'capture_receipt.json').exists())
 def test_control_notifications_have_a_cap(self):
  with tempfile.TemporaryDirectory() as d:
   p=m.Parser(TraceSink(d));p.control_lines=4096
   with self.assertRaisesRegex(TraceStop,'control notification cap'):p.chunk('stderr',b'strace: Process 12 attached\n')
   p.sink.close()
 def test_deadline_cleanup_and_unknown_warning_not_ignored(self):
  with tempfile.TemporaryDirectory() as d:
   self.assertEqual(m.capture([sys.executable,'-c','import time;time.sleep(2)'],d,(),time.monotonic()+.05),'startup timeout')
   self.assertTrue(Path(d,'capture_receipt.json').exists())
 def test_scientific_and_identity_guards_unchanged_in_observer(self):
  import ast
  prior=ast.parse(Path(old.__file__).read_text());new=ast.parse(Path(m.__file__).read_text())
  for name in ('verify_binding','verify_runtime_descriptor'):
   a=next(n for n in prior.body if isinstance(n,ast.FunctionDef) and n.name==name)
   b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))

class RealImporttimeAndBootstrapFixtures(unittest.TestCase):
 def test_actual_stdlib_only_importtime_output(self):
  with tempfile.TemporaryDirectory() as d:
   code='import json;print(json.dumps({"event":"CPU_diagnostic_start"}))'
   reason=m.capture([sys.executable,'-S','-X','importtime','-c',code],d,(),time.monotonic()+2)
   self.assertEqual(reason,'complete')
   receipt=json.loads(Path(d,'capture_receipt.json').read_text())
   self.assertGreater(receipt['diagnostics']['line_type_counts'][str(m.TYPE_CODES['import_row'])],0)
   self.assertEqual(receipt['diagnostics']['line_type_counts'][str(m.TYPE_CODES['import_header'])],1)
 def test_tool_bootstrap_cannot_be_spoofed_on_stdout(self):
  with tempfile.TemporaryDirectory() as d:
   p=m.Parser(TraceSink(d))
   with self.assertRaisesRegex(TraceStop,'wrong channel'):p.chunk('stdout',b'strace: Process 12 attached\n')
   p.sink.close()
 def test_unknown_cached_importtime_still_fails(self):
  with tempfile.TemporaryDirectory() as d:
   p=m.Parser(TraceSink(d))
   with self.assertRaisesRegex(TraceStop,'malformed importtime'):p.chunk('stderr',b'import time: cached | cached | torch\n')
   p.sink.close()
