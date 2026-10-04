"""Complete-pair precision audit; incomplete artifacts cannot support inference."""
import argparse,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args()
root=Path('outputs/preference_program/manifests');m=root/'precision_control_v43.jsonl';source=[json.loads(x) for x in m.read_text().splitlines()];rows=[json.loads(x) for x in (a.results/'base.jsonl').read_text().splitlines()]
assert len(rows)==8,'incomplete qualification:no paired scientific contrast'
receipt=json.loads((a.results/'base_receipt.json').read_text());assert receipt['complete'] and receipt['records']==8 and receipt['manifest_sha256']==hashlib.sha256(m.read_bytes()).hexdigest() and receipt['runner_sha256']==hashlib.sha256(Path('scripts/run_precision_control.py').read_bytes()).hexdigest() and receipt['max_new_tokens']==24 and receipt['do_sample'] is False
lookup={}
for r,s in zip(rows,source):
 for key in s:assert r[key]==s[key]
 assert r['prompt'] in r['rendered_prompt'] and len(r['prompt_ids'])==r['prompt_length']
 assert len(r['generated_ids'])==r['generated_length']<=24 and r['last_token']==(r['generated_ids'][-1] if r['generated_ids'] else None) and r['eos_observed']==(r['last_token'] in r['eos_ids']) and r['length_cap_reached']==(r['generated_length']==24)
 match=re.fullmatch(r'\s*([ABCD])\s*',r['completion']);parsed=None if match is None else match.group(1);assert parsed==r['parsed_response'] and r['parse_failure']==(match is None) and r['correct']==(parsed==r['gold_action'])
 lookup[r['precision'],r['permutation_id']]=r
assert len(lookup)==8
for k in range(4):assert lookup['NF4',k]['prompt_ids']==lookup['BF16',k]['prompt_ids']
configs=receipt['arm_configs'];assert configs['NF4']['quantized'] and not configs['BF16']['quantized'] and configs['NF4']['quantization_compute_dtypes']==['torch.bfloat16'] and set(configs['BF16']['parameter_dtype_counts'])=={'torch.bfloat16'}
for key in ('attention_implementation','eos_ids','tokenizer_eos_id','chat_template_sha256','use_cache'):assert configs['NF4'][key]==configs['BF16'][key]
arms={}
for precision in ('NF4','BF16'):
 g=[lookup[precision,k] for k in range(4)]
 arms[precision]={'correct':sum(r['correct'] for r in g),'parse_failures':sum(r['parse_failure'] for r in g),'length_caps':sum(r['length_cap_reached'] for r in g),'measurement_gate_pass':all(r['correct'] and not r['parse_failure'] and not r['length_cap_reached'] for r in g),'outputs':[{key:r[key] for key in ('permutation_id','gold_action','completion','generated_length','eos_observed')} for r in g]}
delta=[int(lookup['BF16',k]['correct'])-int(lookup['NF4',k]['correct']) for k in range(4)]
report={'valid':True,'records':8,'arms':arms,'BF16_minus_NF4':{'differences':delta,'mean':sum(delta)/4,'changed_responses':sum(lookup['BF16',k]['completion']!=lookup['NF4',k]['completion'] for k in range(4))},'eos_observed':sum(r['eos_observed'] for r in rows),'wall_elapsed_seconds':receipt['wall_elapsed_seconds'],'limitation':'One reusedcase;precisionpackage includesquantizedkernels/loadingcodepaths;NF4thenBF16 confounds order/allocator/time;freshcase gate required;no general/preference conclusion','stop':'No automatic retry,tuning,enlargement/320study'}
(a.results/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
