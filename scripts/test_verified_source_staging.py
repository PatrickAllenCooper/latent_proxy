"""Local real-file fixtures, no torch/sympy/accelerate imports."""
import tempfile
from pathlib import Path
from stage_verified_python_sources import stage_packages,verify_staged
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);site=root/'site';(site/'torch/sub').mkdir(parents=True)
 (site/'torch/__init__.py').write_text('unchanged source\n');(site/'torch/sub/module.py').write_bytes(b'x=1\n');(site/'torch/native.so').write_bytes(b'fixture-binary')
 receipt=stage_packages(site,root/'staged',('torch',));assert verify_staged(receipt)
 assert (root/'staged/torch/native.so').is_symlink() and not (root/'staged/torch/__init__.py').is_symlink()
 (site/'torch/native.so').write_bytes(b'tampered')
 try:verify_staged(receipt)
 except ValueError:pass
 else:raise AssertionError('changed linked binary accepted')
print('Exact Python copy, explicit linked asset, content hash and mutation rejection fixtures passed; no dependency imports')
