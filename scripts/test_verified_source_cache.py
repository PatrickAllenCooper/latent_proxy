"""Local exact-source/origin/staleness fixtures; no dependency imports."""
import tempfile
from pathlib import Path
from verified_source_cache import capture_sources,verified_source,compile_preserving_origin
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);source=root/'module.py';source.write_text('value = 7\n')
 receipt=capture_sources([source],root/'cache');row=receipt['sources'][0]
 assert verified_source(row)==source.read_bytes();code=compile_preserving_origin(row)
 assert code.co_filename==str(source.resolve());namespace={};exec(code,namespace);assert namespace['value']==7
 originalcode=compile(source.read_bytes(),str(source.resolve()),'exec',dont_inherit=True)
 assert code.co_code==originalcode.co_code and code.co_consts==originalcode.co_consts
 source.write_text('value = 8\n')
 try:verified_source(row)
 except ValueError:pass
 else:raise AssertionError('stale original accepted')
 source.write_text('value = 7\n');Path(row['cache_path']).write_text('value = 9\n')
 try:verified_source(row)
 except ValueError:pass
 else:raise AssertionError('tampered cache accepted')
print('Exactbytes/code/originalfilename and original/cache mutation fixtures passed; no real package imports')
