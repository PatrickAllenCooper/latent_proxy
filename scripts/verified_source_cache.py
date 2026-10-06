"""Narrow exact-source cache contract, not a deployed importer.
Original module filenames/package search paths/native references remain unchanged.
Missing/untraced modules always use the original runtime; no tree pruning.
"""
import hashlib,json
from pathlib import Path

def sha_bytes(data):return hashlib.sha256(data).hexdigest()

def capture_sources(paths,destination):
 destination=Path(destination);destination.mkdir(exist_ok=False);rows=[]
 for source in paths:
  source=Path(source)
  if source.suffix!='.py' or source.is_symlink():raise ValueError('only explicit regular Python sources')
  before=source.stat();data=source.read_bytes();after=source.stat()
  if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('source changed during capture')
  digest=sha_bytes(data);stored=destination/(digest+'.py')
  if stored.exists():
   if stored.read_bytes()!=data:raise ValueError('cache collision')
  else:stored.write_bytes(data)
  rows.append({'original_path':str(source.resolve()),'cache_path':str(stored.resolve()),'sha256':digest,'bytes':len(data),'stat_signature':[after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns]})
 receipt={'sources':rows,'runtime_paths_changed':False,'native_assets_changed':False,'importer_deployed':False,'full_environment_identity':False}
 (destination/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

def verified_source(row):
 # Both source and cache verified: stale mapping or mutated data fails closed.
 original=Path(row['original_path']);cached=Path(row['cache_path'])
 data=cached.read_bytes()
 if sha_bytes(data)!=row['sha256'] or sha_bytes(original.read_bytes())!=row['sha256']:raise ValueError('changed original or cached source')
 return data

def compile_preserving_origin(row,optimize=-1):
 return compile(verified_source(row),row['original_path'],'exec',dont_inherit=True,optimize=optimize)
