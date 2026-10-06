"""Local cache-contract fixtures, no actual Transformers/package/model import."""
import hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from verified_distribution_map import load_verified_map,import_with_verified_map
mapping={'PIL':['Pillow'],'torch':['torch'],'namespace':['dist-a','dist-b']}
receipt={'producer':'original_importlib.metadata.packages_distributions','environment_digest':'env','metadata_source_digest':'source','mapping':mapping,'mapping_sha256':hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'cache.json';p.write_text(json.dumps(receipt));assert load_verified_map(p,'env','source')==mapping
 for e,s in [('wrong','source'),('env','wrong')]:
  try:load_verified_map(p,e,s)
  except ValueError:pass
  else:raise AssertionError('stale cache accepted')
 receipt['mapping']['torch']=['invented'];p.write_text(json.dumps(receipt))
 try:load_verified_map(p,'env','source')
 except ValueError:pass
 else:raise AssertionError('tamper accepted')
original=lambda:{'original':['original']};module=SimpleNamespace(packages_distributions=original)
assert import_with_verified_map(lambda:module.packages_distributions(),module,mapping)==mapping
assert module.packages_distributions is original
try:import_with_verified_map(lambda:(_ for _ in ()).throw(RuntimeError('fixture')),module,mapping)
except RuntimeError:pass
assert module.packages_distributions is original
print('Local cache identity/staleness/tamper/restoration fixtures passed; no production cache or model imports')
