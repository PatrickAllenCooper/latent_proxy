"""Review-only inspection exists index; no filenames/package contents changed.
Single-threaded startup only, same mutation/symlink rules as metadata index.
"""
import genericpath,os
from pathlib import Path
from indexed_metadata_exists import ExistsIndex

def inspect_with_index(callable_):
 original_os=os.path.exists;original_generic=genericpath.exists;index=ExistsIndex(Path.exists)
 def exists(path):
  # Preserve fd/bytes/custom path behavior through original implementation.
  if not isinstance(path,(str,Path)):return original_os(path)
  try:return index.exists(path)
  except (TypeError,ValueError):return original_os(path)
 os.path.exists=exists;genericpath.exists=exists
 try:
  result=callable_();index.validate();return result,index.receipt()
 finally:os.path.exists=original_os;genericpath.exists=original_generic
