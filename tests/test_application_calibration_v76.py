import json,os,sys,tempfile,subprocess,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'));import application_calibration_v76 as a
class Audit(unittest.TestCase):
 def test_real_audit_and_read(self):
  with tempfile.TemporaryDirectory() as d:
   source=Path(d,'source');source.mkdir();spool=Path(d,'spool');spool.mkdir();(source/'calibration_input.bin').write_bytes(a.PAYLOAD)
   r=subprocess.run([sys.executable,'-S','-B',a.__file__,'child',str(source),str(spool)],capture_output=True,timeout=3);self.assertEqual(r.returncode,0);receipt=json.loads((spool/'application_child.json').read_text());a.validate_child(receipt,[json.loads(x) for x in r.stdout.splitlines()]);self.assertFalse(receipt['native_syscalls_observed'])
   receipt['audit_open_events']=0
   with self.assertRaises(a.Stop):a.validate_child(receipt,[])
 def test_bad_payload_stops(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'calibration_input.bin').write_bytes(b'bad');r=subprocess.run([sys.executable,'-S','-B',a.__file__,'child',d,d],capture_output=True,timeout=3);self.assertNotEqual(r.returncode,0);self.assertFalse((p/'application_child.json').exists())
 def test_held_and_metadata_cap(self):
  with self.assertRaises(a.Stop):a.admit({'admission':False},{})
  with tempfile.TemporaryDirectory() as d:
   p=Path(d,'receipt')
   with self.assertRaises(a.Stop):a.save(p,{'text':'x'*5000},100)
   self.assertFalse(p.exists())
if __name__=='__main__':unittest.main()
