"""Frozen future analysis; no model/API imports, no fitting or threshold tuning."""
import hashlib,json
from pathlib import Path
from prepare_eligibility_priority import score
ROOT=Path('outputs/preference_program/manifests/preference_factorial_v56')
def analyze(records):
 validation=json.loads((ROOT/'validation.json').read_text())
 for name,h in validation['hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 assert hashlib.sha256(Path('scripts/prepare_eligibility_priority.py').read_bytes()).hexdigest()==validation['scorer_sha256']
 design={r['case_id']:r for r in map(json.loads,(ROOT/'execution_manifest.jsonl').read_text().splitlines())}
 assert len(records)==16 and len({r['case_id'] for r in records})==16 and {r['case_id'] for r in records}==set(design)
 scored={};attributes={}
 for r in records:
  c=design[r['case_id']];assert r['prompt']==c['prompt'];s=score(c,r['completion']);scored[r['case_id']]=s
  attributes[r['case_id']]=next((o['attribute'] for o in c['options'] if o['label']==s['parsed']),None)
 pairs=json.loads((ROOT/'paired_contrasts.json').read_text());contrasts={}
 levels={'format':('prose','explicit_rank'),'label_rotation':(0,2),'row_order':('forward','reverse')}
 for factor,ps in pairs.items():
  first,second=levels[factor];diff=[];agreement=[];pair_details=[]
  for ids in ps:
   a=next(i for i in ids if design[i][factor]==first);b=next(i for i in ids if design[i][factor]==second)
   diff.append(int(scored[b]['correct'])-int(scored[a]['correct']))
   agreement.append(attributes[a] is not None and attributes[b] is not None and attributes[a]==attributes[b])
   pair_details.append(dict(semantic_case_id=design[a]['semantic_case_id'],first_case_id=a,second_case_id=b,correct_difference=diff[-1],same_semantic_choice=agreement[-1]))
  contrasts[factor]=dict(first=first,second=second,paired_correct_difference=diff,mean_paired_accuracy_difference=sum(diff)/8,semantic_choice_agreement_pairs=sum(agreement),total_pairs=8,pairs=pair_details)
 by_format={f:dict(n=8,correct=sum(scored[i]['correct'] for i in design if design[i]['format']==f)) for f in levels['format']}
 valid=all(not s['parse_failure'] for s in scored.values()) and all(r['eos_observed'] and not r['length_cap_reached'] and 0<len(r['generated_ids'])<24 and r['generated_ids'][-1] in r['eos_ids'] and all(type(i) is int and i>=0 for i in r['generated_ids']) for r in records)
 return dict(n=16,correct=sum(s['correct'] for s in scored.values()),eligibility_violations=sum(s['eligibility_violation'] for s in scored.values()),eligible_priority_errors=sum(s['eligible_priority_error'] for s in scored.values()),parse_failures=sum(s['parse_failure'] for s in scored.values()),by_format=by_format,by_semantic_case={case:dict(n=8,correct=sum(scored[i]['correct'] for i in design if design[i]['semantic_case_id']==case),by_format={f:dict(n=4,correct=sum(scored[i]['correct'] for i in design if design[i]['semantic_case_id']==case and design[i]['format']==f)) for f in levels['format']}) for case in sorted({c['semantic_case_id'] for c in design.values()})},contrasts=contrasts,response_format_valid=valid,new_smoke_passed=valid and all(s['correct'] for s in scored.values()),original_v55_qualification_still_failed=True,advice_panel_blocked=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();records=[json.loads(l) for l in a.results.read_text().splitlines()];a.output.write_text(json.dumps(analyze(records),indent=2)+'\n')
