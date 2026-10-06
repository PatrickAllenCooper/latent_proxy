"""Reproduce startup guard race and bookkeeping isolation locally."""
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from indexed_metadata_exists import ExistsIndex,DirectoryChanged
class SourceDirectory(unittest.TestCase):
 def test_ledger_in_indexed_source_root_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p=root/'runner.py';p.write_text('frozen')
   i=ExistsIndex();self.assertTrue(i.exists(p));(root/'jobs.json').write_text('{}')
   with self.assertRaises(DirectoryChanged):i.validate()
 def test_bookkeeping_outside_source_root_preserves_guard(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);src=root/'source';src.mkdir();book=root/'bookkeeping';book.mkdir();p=src/'runner.py';p.write_text('frozen')
   i=ExistsIndex();self.assertTrue(i.exists(p));(book/'jobs.json').write_text('{}');i.validate();self.assertEqual(p.read_text(),'frozen')
 def test_guard_still_rejects_source_entry_mutation(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);src=root/'source';src.mkdir();p=src/'runner.py';p.write_text('frozen');i=ExistsIndex();i.exists(p);(src/'extra.py').write_text('changed')
   with self.assertRaises(DirectoryChanged):i.validate()
if __name__=='__main__':unittest.main()
