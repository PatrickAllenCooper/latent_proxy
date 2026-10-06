import os,genericpath,tempfile
from pathlib import Path
from indexed_source_inspection import inspect_with_index
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);(root/'source.py').write_text('');(root/'live').symlink_to(root/'source.py');(root/'dead').symlink_to(root/'absent')
 paths=[str(root/'source.py'),str(root/'absent'),str(root/'live'),str(root/'dead'),root,os.fsencode(root/'source.py')]
 original_os=os.path.exists;original_generic=genericpath.exists;expected=[original_os(p) for p in paths]
 actual,_=inspect_with_index(lambda:[os.path.exists(p) for p in paths]);assert actual==expected
 assert os.path.exists is original_os and genericpath.exists is original_generic
 try:inspect_with_index(lambda:(_ for _ in ()).throw(RuntimeError('fixture')))
 except RuntimeError:pass
 assert os.path.exists is original_os and genericpath.exists is original_generic
print('Inspection exists equivalence, symlink/bytes fallback and restoration fixtures passed')
