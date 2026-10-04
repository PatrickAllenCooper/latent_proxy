"""Audit frozen numeric qualification, preserving failures as valid evidence."""
import argparse, hashlib, json, re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
m=Path('outputs/preference_program/manifests/fresh_bf16_v44.jsonl');source=[json.loads(x) for x in m.read_text().splitlines()]
rows=[json.loads(x) for x in (a.results/'base.jsonl').read_text().splitlines()];receipt=json.loads((a.results/'base_receipt.json').read_text())
assert len(rows)==16 and receipt['complete'] and receipt['records']==16
assert receipt['manifest_sha256']==hashlib.sha256(m.read_bytes()).hexdigest()
assert receipt['runner_sha256']==hashlib.sha256(Path('scripts/run_fresh_bf16_qualification.py').read_bytes()).hexdigest()
assert receipt['snapshot'].endswith('989aa7980e4cf806f80c7fef2b1adb7bc71aa306') and receipt['max_new_tokens']==24 and receipt['do_sample'] is False
cfg=receipt['arm_configs']['BF16'];assert not cfg['quantized'] and set(cfg['parameter_dtype_counts'])=={'torch.bfloat16'} and cfg['attention_implementation']=='sdpa' and cfg['use_cache'] is False
assert cfg['chat_template_sha256']=='cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f'
for r,s in zip(rows,source):
 for k in s:assert r[k]==s[k]
 assert s['gold_action']=='ABCD'[s['displayed_values'].index(max(s['displayed_values']))]
 assert r['prompt'] in r['rendered_prompt'] and len(r['prompt_ids'])==r['prompt_length']<=512
 assert len(r['generated_ids'])==r['generated_length']<=24 and r['last_token']==r['generated_ids'][-1]
 assert r['eos_ids']==cfg['eos_ids'] and r['eos_observed']==(r['last_token'] in r['eos_ids']) and r['length_cap_reached']==(r['generated_length']==24)
 match=re.fullmatch(r'\s*([ABCD])\s*',r['completion']);parsed=match.group(1) if match else None
 assert parsed==r['parsed_response'] and r['parse_failure']==(match is None) and r['correct']==(parsed==r['gold_action'])
report={'valid':True,'records':16,'correct':sum(r['correct'] for r in rows),'parse_failures':sum(r['parse_failure'] for r in rows),'eos_observed':sum(r['eos_observed'] for r in rows),'length_caps':sum(r['length_cap_reached'] for r in rows),'measurement_gate_pass':all(r['correct'] and r['eos_observed'] and not r['parse_failure'] and not r['length_cap_reached'] for r in rows),'by_stratum':{s:sum(r['correct'] for r in rows if r['stratum']==s) for s in sorted({r['stratum'] for r in rows})},'wrong_cases':[{'case_id':r['case_id'],'values':r['displayed_values'],'gold':r['gold_action'],'completion':r['completion']} for r in rows if not r['correct']],'limitation':'Balanced non-iid positive integer selection; not preference discovery/advice correction; does not resolve prior precision order/kernel confounds','next_gate':'Failure stops for conceptual assay review; pass permits separately registered verification only; no retry/tuning/enlargement/320 study'}
(a.results/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
