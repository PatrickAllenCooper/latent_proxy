"""Descriptive audit of frozen16response representation/order diagnostic."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
manifest=Path('outputs/preference_program/manifests/numeric_capability_v40.jsonl');source=[json.loads(x) for x in manifest.read_text().splitlines()];rows=[json.loads(x) for x in (a.results/'base.jsonl').read_text().splitlines()];assert len(rows)==16
receipt=json.loads((a.results/'base_receipt.json').read_text());assert receipt['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest() and receipt['max_new_tokens']==24 and receipt['do_sample'] is False
for r,s in zip(rows,source):
 for k in s:assert r[k]==s[k]
 assert r['messages']==[{'role':'user','content':r['prompt']}] and r['prompt'] in r['rendered_prompt']
 assert r['gold_action']=='ABCD'[int(np.argmax(r['displayed_values']))]
 if not r['parse_failure']:
  v=np.array(r['original_utilities']);act=ord(r['final_action'])-65;assert abs((v.max()-v[act])/(v.max()-v.min())-r['normalized_regret'])<1e-10
cells={}
for rep in ('decimals','integers'):
 for order in ('canonical','reversed'):
  g=[r for r in rows if r['representation']==rep and r['listing_order']==order]
  cells[f'{rep}/{order}']={'records':len(g),'correct':sum(r['final_action']==r['gold_action'] for r in g),'parse_failures':sum(r['parse_failure'] for r in g),'A_responses':sum(r['final_action']=='A' for r in g),'first_listed_responses':sum(r['final_action']==r['first_listed_action'] for r in g),'responses':[{k:r[k] for k in ('permutation_id','gold_action','completion','final_action')} for r in g]}
report={'valid':True,'records':16,'cells':cells,'limitation':'Descriptive one reused case; representation changes sign/scale/precision together; cannot infer preference recovery, reward arithmetic or general capability','next_gate':'No automatic repeat or expansion;review matched error pattern'}
(a.results/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
