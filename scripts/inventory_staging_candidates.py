"""Finite CPU filesystem inventory only; no package imports, model or execution."""
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=25);a=p.parse_args();start=time.monotonic();rows=[];complete=True
for package in ('torch','sympy','accelerate'):
 root=a.site/package;python_bytes=0;python_files=0;other_bytes=0;other_files=0;symlinks=[]
 for parent,dirs,files in os.walk(root,followlinks=False):
  dirs[:]=[d for d in dirs if d!='__pycache__']
  if time.monotonic()-start>a.seconds:complete=False;break
  for name in files:
   f=Path(parent)/name
   if f.is_symlink():symlinks.append(str(f));continue
   size=f.stat().st_size
   if f.suffix=='.py':python_files+=1;python_bytes+=size
   else:other_files+=1;other_bytes+=size
 rows.append({'package':package,'root':str(root),'python_files':python_files,'python_bytes':python_bytes,'other_files':other_files,'other_bytes':other_bytes,'symlinks':symlinks,'complete':complete});print(json.dumps(rows[-1]),flush=True)
 if not complete:break
receipt={'complete':complete,'elapsed_seconds':time.monotonic()-start,'packages':rows,'no_package_imports':True,'model_calls':0};a.output.write_text(json.dumps(receipt,indent=2)+'\n')
