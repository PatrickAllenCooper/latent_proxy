import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reader32_trace_capture_v68 as m
from reader32_trace_privacy_v68 import TraceSink,TraceStop

class CaptureFixtures(unittest.TestCase):
 def test_parser_fd_timestamp_redaction(self):
  with tempfile.TemporaryDirectory() as d:
   sink=TraceSink(d,('/verified/package',));p=m.Parser(sink)
   p.chunk('1:syscalls',b'12 123.000001 openat(AT_FDCWD, "/verified/package/x.so", O_RDONLY) = 3 <0.250000>\n')
   p.chunk('1:syscalls',b'12 123.300001 read(0x3, 0xdeadbeef, 0x100) = 32 <0.100000>\n')
   p.finish();sink.close();rows=[json.loads(x) for x in Path(d,'syscalls.jsonl').read_text().splitlines()]
   self.assertEqual(rows[1]['path'],'root0/x.so');self.assertEqual(rows[1]['fd'],3)
   self.assertEqual(rows[1]['timestamp'],123.300001)
   self.assertNotIn('deadbeef',Path(d,'syscalls.jsonl').read_text())
 def test_reject_buffers_denial_oversize_unexpected(self):
  for text in ('12 1.000000 read(3, "secret", 6) = 6 <0.001000>\n','strace: ptrace: Operation not permitted\n','x'*8193,'12 1.000000 execve("secret") = 0 <0.000001>\n'):
   with tempfile.TemporaryDirectory() as d:
    s=TraceSink(d);p=m.Parser(s)
    with self.assertRaises(TraceStop):p.chunk('syscalls',text.encode())
    s.close()
 def test_no_filesystem_probe(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   s=TraceSink(d,('/verified',))
   with patch.object(Path,'resolve',side_effect=AssertionError('probe')):
    self.assertEqual(s.path_label('/outside/private'),'[redacted]')
    self.assertEqual(s.path_label('/verified/x'),'root0/x')
   s.close()
 def test_streams_separate_and_complete(self):
  with tempfile.TemporaryDirectory() as d:
   code='import sys;print(\'{"event":"CPU_diagnostic_start"}\',flush=True);sys.stderr.write("import time: 1 | 2 | toy.module\\n")'
   self.assertEqual(m.capture([sys.executable,'-c',code],d,(),time.monotonic()+2),'complete')
   receipt=json.loads(Path(d,'capture_receipt.json').read_text())
   self.assertEqual(receipt['sha256']['imports'],hashlib.sha256(Path(d,'imports.jsonl').read_bytes()).hexdigest())
 def test_timeout_group_and_unrelated_survive(self):
  outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
  try:
   with tempfile.TemporaryDirectory() as d:
    t=time.monotonic()
    code='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(10)"]);time.sleep(10)'
    self.assertEqual(m.capture([sys.executable,'-c',code],d,(),t+.15),'startup timeout')
    self.assertLess(time.monotonic()-t,2.5);self.assertIsNone(outsider.poll())
    self.assertTrue(Path(d,'capture_receipt.json').exists())
  finally: outsider.terminate();outsider.wait()
 def test_cap_and_denial_durable(self):
  for code,limit,expected in [('print("import time: 1 | 2 | toy.module")',1,'combined trace cap'),('print("denied secret credential")',1000,'unexpected child output'),('print("x"*10000)',1000,'oversize line')]:
   with tempfile.TemporaryDirectory() as d:
    self.assertEqual(m.capture([sys.executable,'-c',code],d,(),time.monotonic()+2,limit=limit),expected)
    data=Path(d,'capture_receipt.json').read_text();self.assertNotIn('credential',data)
 def test_command_no_attach_or_elevation(self):
  source=Path(m.__file__).read_text()
  self.assertIn("'-e','raw='+RAW",source)
  self.assertNotIn("'sudo'",source);self.assertNotIn("'-p'",source)

if __name__=='__main__': unittest.main()

class BindingAndDeadlineFixtures(unittest.TestCase):
 def test_unchanged_import_binding_and_mutation(self):
  path=Path(__file__).resolve().parents[1]/'outputs/preference_program/manifests/reader32_import_trace_v68_local/manifest.json'
  manifest=json.loads(path.read_text());m.verify_binding(manifest)
  bad=dict(manifest,source_hashes=dict(manifest['source_hashes']))
  bad['source_hashes'][next(iter(bad['source_hashes']))]='0'*64
  with self.assertRaises(TraceStop): m.verify_binding(bad)
 def test_external_term_durable_and_child_group_stopped(self):
  with tempfile.TemporaryDirectory() as d:
   script=Path(d,'fixture.py')
   script.write_text('import sys,time;sys.path.insert(0,sys.argv[1]);import reader32_trace_capture_v68 as m; m.capture([sys.executable,"-c","import time;time.sleep(10)"],sys.argv[2],(),time.monotonic()+10)')
   controller=subprocess.Popen([sys.executable,str(script),str(Path(m.__file__).parent),d])
   time.sleep(.12);controller.terminate();controller.wait(timeout=3)
   receipt=json.loads(Path(d,'capture_receipt.json').read_text())
   self.assertEqual(receipt['reason'],'external deadline');self.assertLess(receipt['elapsed'],3)
 def test_hold_cli_no_child(self):
  with tempfile.TemporaryDirectory() as d:
   manifest=Path(d,'held.json');manifest.write_text('{"cpu_admission":false}')
   p=subprocess.run([sys.executable,m.__file__,str(manifest),d],capture_output=True)
   self.assertNotEqual(p.returncode,0);self.assertFalse(Path(d,'capture_receipt.json').exists())

class DescendantFixture(unittest.TestCase):
 def test_reaped_leader_ignoring_descendant_cleanup(self):
  outsider=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
  try:
   with tempfile.TemporaryDirectory() as d:
    marker=Path(d,'leaked.txt')
    child='import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(.5);pathlib.Path('+repr(str(marker))+').write_text("leaked");time.sleep(5)'
    parent='import subprocess,sys;subprocess.Popen([sys.executable,"-c",'+repr(child)+']);sys.exit(0)'
    self.assertEqual(m.capture([sys.executable,'-c',parent],d,(),time.monotonic()+.2),'startup timeout')
    time.sleep(.6)
    self.assertFalse(marker.exists());self.assertIsNone(outsider.poll())
  finally:outsider.terminate();outsider.wait()
 def test_import_label_attribution(self):
  with tempfile.TemporaryDirectory() as d:
   s=TraceSink(d);p=m.Parser(s)
   p.chunk('stderr:imports',b'import time: 100 | 200 | torch.distributed.rpc\n')
   p.finish();s.close()
   self.assertEqual(json.loads(Path(d,'imports.jsonl').read_text())['module_label'],'torch.distributed.rpc')

class PendingAndDenialFixtures(unittest.TestCase):
 def test_unfinished_wait_retained_privately(self):
  with tempfile.TemporaryDirectory() as d:
   s=TraceSink(d,('/verified',));p=m.Parser(s)
   p.chunk('syscalls',b'12 123.000001 openat(AT_FDCWD, "/verified/x.so", O_RDONLY <unfinished ...>\n')
   s.close();row=json.loads(Path(d,'syscalls.jsonl').read_text())
   self.assertTrue(row['unfinished']);self.assertEqual(row['pid'],12)
   self.assertNotIn('seconds',row);self.assertEqual(row['path'],'root0/x.so')
 def test_actual_child_denial_channel(self):
  with tempfile.TemporaryDirectory() as d:
   code='import sys;sys.stderr.write("strace: ptrace: Operation not permitted\\n")'
   self.assertEqual(m.capture([sys.executable,'-c',code],d,(),time.monotonic()+2),'trace unavailable')
