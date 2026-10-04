"""Audit frozen8response paired outputs and token-level termination evidence."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
m=Path('outputs/preference_program/manifests/output_mapping_v41.jsonl');source=[json.loads(x) for x in m.read_text().splitlines()];rows=[json.loads(x) for x in (a.results/'base.jsonl').read_text().splitlines()];assert len(rows)==8
receipt=json.loads((a.results/'base_receipt.json').read_text());assert receipt['records']==8 and receipt['manifest_sha256']==hashlib.sha256(m.read_bytes()).hexdigest() and receipt['runner_sha256']==hashlib.sha256(Path('scripts/run_output_mapping.py').read_bytes()).hexdigest() and receipt['max_new_tokens']==24 and receipt['do_sample'] is False
for r,s in zip(rows,source):
 for k in s:assert r[k]==s[k]
 assert r['messages']==[{'role':'user','content':r['prompt']}] and r['prompt'] in r['rendered_prompt']
 assert len(r['generated_ids'])==r['generated_length']<=24 and r['prompt_length']>0
 assert r['last_token']==(r['generated_ids'][-1] if r['generated_ids'] else None)
 assert r['length_cap_reached']==(r['generated_length']==24) and r['eos_observed']==(r['last_token'] in r['eos_ids'])
 match=re.fullmatch(r'\s*([ABCD])\s*' if r['response_mode']=='label' else r'\s*([0-9])\s*',r['completion']);parsed=match.group(1) if match else None
 assert parsed==r['parsed_response'] and r['parse_failure']==(match is None) and r['correct']==(parsed==r['expected_response'])
lookup={(r['response_mode'],r['permutation_id']):r for r in rows}
report={'valid':True,'records':8,'modes':{mode:{'correct':sum(r['correct'] for r in rows if r['response_mode']==mode),'parse_failures':sum(r['parse_failure'] for r in rows if r['response_mode']==mode),'responses':[{k:r[k] for k in ('permutation_id','expected_response','completion','correct','generated_length','eos_observed','length_cap_reached')} for r in rows if r['response_mode']==mode]} for mode in ('label','number')},'number_minus_label_accuracy':sum(int(lookup['number',k]['correct'])-int(lookup['label',k]['correct']) for k in range(4))/4,'eos_observed':sum(r['eos_observed'] for r in rows),'length_caps':sum(r['length_cap_reached'] for r in rows),'limitation':'One reusedcase;numericgoldalways4,so number agreement may beconstantoutput;taskframing/outputmappingonly;no preference/generalcomparison claim','stop':'No automatic retry,tuning,enlargement or320record launch'}
(a.results/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
