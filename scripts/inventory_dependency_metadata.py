"""Bounded metadata inventory, standard library only; no package imports or models."""
import argparse,csv,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site-packages',type=Path,required=True);p.add_argument('--seconds',type=float,default=25);p.add_argument('--output',type=Path,required=True);a=p.parse_args();start=time.monotonic();rows=[]
for d in sorted(a.site_packages.glob('*.dist-info')):
 if time.monotonic()-start>a.seconds:break
 def read(name):
  try:return (d/name).read_bytes()
  except FileNotFoundError:return b''
 top=read('top_level.txt');record=read('RECORD');metadata=read('METADATA')
 rows.append({'distribution_directory':d.name,'declared_top_level':top.decode().split(),'record_entries':sum(1 for _ in csv.reader(record.decode().splitlines())),'record_bytes':len(record),'record_sha256':hashlib.sha256(record).hexdigest(),'metadata_sha256':hashlib.sha256(metadata).hexdigest(),'requires_inferred_files':not bool(top.strip())})
 print(json.dumps(rows[-1]),flush=True)
receipt={'scope':'dist-info only, no imported distribution/file existence checks','elapsed_seconds':time.monotonic()-start,'distributions':rows,'complete':len(rows)==len(list(a.site_packages.glob('*.dist-info'))),'model_calls':0}
a.output.write_text(json.dumps(receipt,indent=2)+'\n')
