"""Validate registered20-response smoke; descriptive only, one underlying case."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
manifest=Path('outputs/preference_program/manifests/faulty_advice_verification_v36.jsonl');source=[json.loads(x) for x in manifest.read_text().splitlines()][:20];rows=[json.loads(x) for x in (a.results/'base.jsonl').read_text().splitlines()];assert len(rows)==20
receipt=json.loads((a.results/'base_receipt.json').read_text());assert receipt['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
for row,s in zip(rows,source):
 for k in s:assert row[k]==s[k]
 assert 'TOOL' not in row['prompt'] and 'copy' not in row['prompt']
 scores=np.array(row['expected_utilities']);assert row['gold_action']=='ABCD'[int(scores.argmax())]
 if not row['parse_failure']:
  action=ord(row['final_action'])-65;regret=float((scores.max()-scores[action])/(scores.max()-scores.min()));assert abs(regret-row['normalized_regret'])<1e-10
arms={}
for arm in sorted({r['arm'] for r in rows}):
 group=[r for r in rows if r['arm']==arm]
 arms[arm]={'records':len(group),'parse_failures':sum(r['parse_failure'] for r in group),'optimal':sum(r['final_action']==r['gold_action'] for r in group),'mean_regret_parse_penalty1':float(np.mean([1 if r['parse_failure'] else r['normalized_regret'] for r in group])),'corrected_wrong_advice':sum(r['advice']!=r['gold_action'] and r['advice'] is not None and r['final_action']==r['gold_action'] for r in group),'spoiled_correct_advice':sum(r['advice']==r['gold_action'] and not r['parse_failure'] and r['final_action']!=r['gold_action'] for r in group)}
report={'valid':True,'arms':arms,'cases':1,'records':20,'limitation':'Pipeline smoke on one underlying case; no inferential CI or population claim. Visible utility-score verification, not preference discovery.'}
(a.results/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
