import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('trace_v68', Path(__file__).resolve().parents[1] / 'scripts/reader32_trace_privacy_v68.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class PrivacyFixtures(unittest.TestCase):
    def test_redaction(self):
        with tempfile.TemporaryDirectory() as d:
            s=m.TraceSink(d, (d,))
            for p in ('/private/credentials', '../secret', d+'/../secret'):
                self.assertEqual(s.path_label(p), '[redacted]')
            self.assertEqual(s.path_label(d+'/known.py'), 'root0/known.py')
            s.retain('syscalls','openat',path='/private/credentials')
            s.close()
            self.assertNotIn('credentials',Path(d,'syscalls.jsonl').read_text())
    def test_shared_cap_and_custody(self):
        with tempfile.TemporaryDirectory() as d:
            s=m.TraceSink(d,limit=50)
            s.retain('syscalls','read',result=1)
            with self.assertRaises(m.TraceStop): s.retain('imports','import',path='/secret')
            receipt=s.close()
            data=Path(d,'syscalls.jsonl').read_bytes()
            self.assertEqual(receipt['retained_bytes'],len(data))
            self.assertEqual(receipt['sha256']['syscalls'],hashlib.sha256(data).hexdigest())
    def test_denied_and_no_buffers(self):
        with tempfile.TemporaryDirectory() as d:
            s=m.TraceSink(d)
            s.retain('syscalls','denied',result=-1)
            with self.assertRaises(m.TraceStop): s.retain('syscalls','read',result='secret buffer')
            with self.assertRaises(m.TraceStop): s.retain('syscalls','execve')
            with self.assertRaises(m.TraceStop): s.retain('syscalls','read',path='/secret')
            s.close()
    def test_hold_timeout_envelope(self):
        r={'cpus':1,'memory_gib':2,'wall_seconds':120,'check_seconds':90,'attempts':1,'gpus':0,'account':'ucb736_asc1','partition':'acpu','qos':'cpu-normal'}
        with self.assertRaises(m.TraceStop): m.require_admission({'cpu_admission':False,'resources':r},0,0)
        with self.assertRaises(m.TraceStop): m.require_admission({'cpu_admission':True,'resources':r},90,0)
        m.require_admission({'cpu_admission':True,'resources':r},89,0)
    def test_stdlib_only_no_launch(self):
        import ast
        tree=ast.parse(Path(m.__file__).read_text())
        imports=[n.names[0].name for n in ast.walk(tree) if isinstance(n,ast.Import)]
        self.assertEqual(imports,['hashlib','json'])

if __name__=='__main__': unittest.main()
