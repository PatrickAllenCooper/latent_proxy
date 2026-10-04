"""Audit all frozen factorial cells and paired correctness contrasts."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
root=Path('outputs/preference_program/manifests');m=root/'framing_factorial_v42.jsonl';source=[json.loads(x) for x in m.read_text().splitlines()];rows=[json.loads(x) for x in (a.results/'base.jsonl').read_text().splitlines()];assert len(rows)==16
receipt=json.loads((a.results/'base_receipt.json').read_text());assert receipt['records']==16 and receipt['manifest_sha256']==hashlib.sha256(m.read_bytes()).hexdigest() and receipt['runner_sha256']==hashlib.sha256(Path('scripts/run_framing_factorial.py').read_bytes()).hexdigest() and receipt['max_new_tokens']==24 and receipt['do_sample'] is False
for r,s in zip(rows,source):
 for k in s:assert r[k]==s[k]
 assert hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256']
 assert r['messages']==[{'role':'user','content':r['prompt']}] and r['prompt'] in r['rendered_prompt']
 assert len(r['generated_ids'])==r['generated_length']<=24 and r['prompt_length']>0
 assert r['last_token']==(r['generated_ids'][-1] if r['generated_ids'] else None) and r['length_cap_reached']==(r['generated_length']==24) and r['eos_observed']==(r['last_token'] in r['eos_ids'])
 match=re.fullmatch(r'\s*([ABCD])\s*',r['completion']);parsed=match.group(1) if match else None
 assert parsed==r['parsed_response'] and r['parse_failure']==(match is None) and r['correct']==(parsed==r['gold_action'])
lookup={(r['header'],r['instruction'],r['permutation_id']):r for r in rows};assert len(lookup)==16
cells={}
for h in ('preference','numbers'):
 for i in ('score','label'):
  g=[lookup[h,i,k] for k in range(4)];passed=all(r['correct'] and not r['parse_failure'] and not r['length_cap_reached'] for r in g)
  cells[h+'/'+i]={'correct':sum(r['correct'] for r in g),'parse_failures':sum(r['parse_failure'] for r in g),'length_caps':sum(r['length_cap_reached'] for r in g),'measurement_gate_pass':passed,'responses':[{key:r[key] for key in ('permutation_id','gold_action','completion','correct','generated_length','eos_observed')} for r in g]}
def c(h,i,k):return int(lookup[h,i,k]['correct'])
header=[c('numbers',i,k)-c('preference',i,k) for i in ('score','label') for k in range(4)]
instruction=[c(h,'label',k)-c(h,'score',k) for h in ('preference','numbers') for k in range(4)]
interaction=[(c('numbers','label',k)-c('preference','label',k))-(c('numbers','score',k)-c('preference','score',k)) for k in range(4)]
report={'valid':True,'records':16,'cells':cells,'paired_contrasts':{name:{'differences':v,'mean':sum(v)/len(v)} for name,v in [('numbers_minus_preference',header),('label_minus_score',instruction),('interaction',interaction)]},'eos_observed':sum(r['eos_observed'] for r in rows),'measurement_gate_pass':any(cell['measurement_gate_pass'] for cell in cells.values()),'limitation':'one reusedcase;passingcell4/4 ismeasurementgate only,freshcase qualification required;no general/preference discovery claim','stop':'No retry/tuning/enlargement/320study'}
(a.results/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
