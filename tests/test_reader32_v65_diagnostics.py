"""CPU-only deadline, diagnostic protocol, and frozen-runner checks. No models."""
import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import reader32_v65_diagnostics as d
from reader32_v60_contracts import result, export_incomplete, violation

class TensorFixture:
    def __init__(self, ids): self.ids=ids
    def tolist(self): return self.ids

class DiagnosticTests(unittest.TestCase):
    def test_budget_arithmetic(self):
        self.assertEqual(90+30+15*10,270)
        self.assertEqual(d.GENERATION_END_SECONDS+15,d.EXPORT_END_SECONDS)
        self.assertEqual(d.EXPORT_END_SECONDS+15,d.JOB_END_SECONDS)
    def test_deadlines_and_caps(self):
        self.assertEqual(d.generation_deadline(77,0,0),107)
        self.assertEqual(d.generation_deadline(108,0,1),118)
        self.assertEqual(d.generation_deadline(265,0,15),270)
        with self.assertRaises(ValueError): d.phase_cap(16)
    def test_reserve_declared_caps(self):
        self.assertTrue(d.admit_next(90,0,0))
        self.assertFalse(d.admit_next(90.001,0,0))
        self.assertTrue(d.admit_next(120,0,1))
        self.assertFalse(d.admit_next(120.001,0,1))
        self.assertTrue(d.admit_next(260,0,15))
        self.assertFalse(d.admit_next(260.001,0,15))
        self.assertFalse(d.admit_next(0,0,16))
    def test_full_schedule_and_no_extra_case(self):
        now=77
        for count in range(16):
            self.assertTrue(d.admit_next(now,0,count))
            now=d.generation_deadline(now,0,count)
        self.assertEqual(now,257)
        self.assertFalse(d.admit_next(now,0,16))
    def test_export_global_cap_unchanged(self):
        self.assertIsNone(violation(284.99,0,285,1,1,2))
        self.assertEqual(violation(285,0,285,1,1,2),'time_limit')
    def test_prompt_new_and_end_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            log=d.DiagnosticLog(Path(td)/'partial.jsonl','first',clock=lambda:123)
            s=d.TokenIDObserver([7,8],log);s.put(TensorFixture([[7,8]]))
            s.put(TensorFixture([9]));s.put(TensorFixture([10]));s.end()
            rows=list(map(json.loads,log.path.read_text().splitlines()))
            self.assertEqual([x['event'] for x in rows],['stream_prompt','stream_new_token','stream_new_token','stream_end'])
            self.assertEqual(s.new_ids,[9,10]);self.assertTrue(s.ended)
            self.assertTrue(all(x['eligible_for_scoring'] is False for x in rows))
    def test_protocol_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            log=d.DiagnosticLog(Path(td)/'partial.jsonl','first');s=d.TokenIDObserver([7],log)
            with self.assertRaises(ValueError): s.put(TensorFixture([[8]]))
            with self.assertRaises(ValueError): s.end()
            s.put(TensorFixture([[7]]))
            for malformed in ([[9]],[9,10],['9']):
                with self.assertRaises(ValueError): s.put(TensorFixture(malformed))
            for i in range(24): s.put(TensorFixture([i]))
            with self.assertRaises(ValueError): s.put(TensorFixture([25]))
            s.end()
            with self.assertRaises(ValueError): s.put(TensorFixture([26]))
    def test_partial_stream_never_scored(self):
        with tempfile.TemporaryDirectory() as td:
            log=d.DiagnosticLog(Path(td)/'first_call_diagnostics.jsonl','first')
            s=d.TokenIDObserver([1],log);s.put(TensorFixture([[1]]));s.put(TensorFixture([2]))
            r=export_incomplete(td,'time_limit',30)
            self.assertEqual(r['records'],0);self.assertIsNone(r['qualified'])
            self.assertFalse(s.ended);self.assertEqual(s.new_ids,[2])
    def test_stack_samples_5_10_20_and_cancel(self):
        class Wait:
            def __init__(self,cancel=False):self.targets=[];self.now=100;self.cancel=cancel
            def wait(self,delay):
                self.targets.append(delay);self.now+=delay
                return self.cancel and len(self.targets)==2
        for cancel,expected in [(False,[5,10,20]),(True,[5])]:
            wait=Wait(cancel);sampler=d.FirstCallStacks(None,123,100,clock=lambda:wait.now);sampler.done=wait
            with patch.object(d,'sample_stack') as sampled:
                sampler._watch();self.assertEqual([x.args[2] for x in sampled.call_args_list],expected)
            self.assertEqual(wait.targets,[5,5] if cancel else [5,5,10])
    def test_missing_stack_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            log=d.DiagnosticLog(Path(td)/'d.jsonl','first');d.sample_stack(log,123,5,frames={})
            row=json.loads(log.path.read_text());self.assertFalse(row['available']);self.assertEqual(row['target_seconds'],5)
    def test_strict_scientific_gate_preserved(self):
        ids=[str(i) for i in range(16)];rs=[{'case_id':i,'score':{'correct':True}} for i in ids]
        self.assertTrue(result(rs,ids,True)['qualified'])
        rs[-1]['score']['correct']=False;self.assertFalse(result(rs,ids,True)['qualified'])
        self.assertIsNone(result(rs[:-1],ids,False)['qualified'])
    def test_model_and_generation_calls_unchanged_except_observer(self):
        def calls(path,name):
            tree=ast.parse(path.read_text());out=[]
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr==name:
                    node.keywords=[k for k in node.keywords if k.arg!='streamer']
                    out.append(ast.dump(node,include_attributes=False))
            return out
        for name in ['from_pretrained','generate','apply_chat_template','encode']:
            self.assertEqual(calls(ROOT/'scripts/run_reader32_v64.py',name),calls(ROOT/'scripts/run_reader32_v65.py',name))
    def test_nested_spool_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'source';src.mkdir();nested=src/'spool'
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','STUDY_WALL_START':str(time.time())}
            p=subprocess.run([sys.executable,str(ROOT/'scripts/run_reader32_v65.py'),'gpu',str(src),'--spool',str(nested)],env=env,capture_output=True,timeout=5)
            self.assertNotEqual(p.returncode,0);self.assertIn(b'spool must be outside source',p.stderr)
            self.assertEqual(list(src.iterdir()),[])

    def test_unapproved_runner_stops_before_framework_import(self):
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/'source';spool=Path(td)/'spool';src.mkdir();spool.mkdir()
            reg={'hashes':{},'scorer_sha256':'unused'}
            # Reach approval gate through an authentic frozen scientific manifest.
            base=ROOT/'outputs/preference_program/manifests/reader32_qualification_v64'
            for name in ['registration.json','cases.jsonl','prompts.jsonl']:(src/name).write_bytes((base/name).read_bytes())
            for name in json.loads((src/'registration.json').read_text())['hashes']:
                if not (src/name).exists():(src/name).write_bytes((ROOT/'scripts'/name).read_bytes())
            (src/'prepare_eligibility_priority.py').write_bytes((ROOT/'scripts/prepare_eligibility_priority.py').read_bytes())
            (src/'execution_freeze.json').write_text(json.dumps({'GPU_admission':False,'resource_amendment_approved':False}))
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','STUDY_WALL_START':str(time.time())}
            p=subprocess.run([sys.executable,str(ROOT/'scripts/run_reader32_v65.py'),'gpu',str(src),'--spool',str(spool)],env=env,capture_output=True,timeout=5)
            self.assertNotEqual(p.returncode,0);self.assertIn(b'30_second_first_case_amendment_not_approved',p.stderr)
            receipt=json.loads((spool/'smoke/budget_stop.json').read_text());self.assertEqual(receipt['records'],0);self.assertIsNone(receipt['qualified'])
            self.assertNotIn(b'dependency_import_start',p.stdout)

if __name__=='__main__':unittest.main()
