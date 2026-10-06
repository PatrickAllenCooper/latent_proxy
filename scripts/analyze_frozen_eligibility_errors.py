"""Post-outcome descriptive analysis; never mutates frozen cases/scorer/results."""
import collections,hashlib,json,re
from pathlib import Path
ROOT=Path('outputs/preference_program/results/eligibility_qualification_v55')
CASES=Path('outputs/preference_program/manifests/eligibility_priority_v45/qualification_cases.jsonl')
MANIFEST=Path('outputs/preference_program/manifests/eligibility_priority_v45/execution_manifest.jsonl')
RAW=ROOT/'smoke/base.jsonl'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cases={c['case_id']:c for c in map(json.loads,CASES.read_text().splitlines())};rows=list(map(json.loads,RAW.read_text().splitlines()));assert len(rows)==len(cases)==16
 diag=[];confusion={g:{r:0 for r in 'ABCD'} for g in 'ABCD'}
 for row in rows:
  c=cases[row['case_id']];eligible=[o for o in c['options'] if o['eligible']];gold=min(eligible,key=lambda o:c['priority'].index(o['attribute']))['label'];assert gold==c['gold']==row['gold_action']
  m=re.fullmatch(r'\s*([ABCD])\s*',row['completion']);assert m;answer=m[1];assert answer==row['parsed_response'];chosen=next(o for o in c['options'] if o['label']==answer);confusion[gold][answer]+=1
  first_eligible=eligible[0]['label'];unfiltered=next(o['label'] for o in c['options'] if o['attribute']==c['priority'][0])
  diag.append(dict(case_id=c['case_id'],gold=gold,response=answer,correct=answer==gold,eligibility_violation=not chosen['eligible'],eligible_priority_error=chosen['eligible'] and answer!=gold,chosen_priority_rank=1+c['priority'].index(chosen['attribute']),top_priority_ineligible=c['top_priority_ineligible'],first_catalog_eligible=first_eligible,unfiltered_top_priority=unfiltered,priority=c['priority'],options=c['options']))
 report=dict(analysis_type='Post-outcome descriptive interpretation of full frozen16; not a preregistered new trial',source_hashes={str(p):sha(p) for p in [CASES,MANIFEST,RAW]},n=16,correct=sum(d['correct'] for d in diag),eligibility_violations=sum(d['eligibility_violation'] for d in diag),eligible_priority_errors=sum(d['eligible_priority_error'] for d in diag),response_counts=dict(collections.Counter(d['response'] for d in diag)),confusion=confusion,diagnostics=diag,heuristic_descriptive_checks=dict(first_catalog_eligible_gold_accuracy=sum(d['first_catalog_eligible']==d['gold'] for d in diag),model_first_catalog_eligible_agreement=sum(d['first_catalog_eligible']==d['response'] for d in diag),unfiltered_top_priority_model_agreement=sum(d['unfiltered_top_priority']==d['response'] for d in diag)),qualification_threshold=16,qualification_passed=False,advice_panel_blocked=True,model_calls=0,GPU_allocations=0,limitations=['Deterministic hand-constructed16casepanel, not random independent population sample','Gold letter tied to presentation order; no causal label-position claim','Single checkpoint/template/greedy decoding; no preference discovery or proxy/tool-use evidence','Post-outcome slices and heuristic agreement are diagnostic, not a rescued qualification subset'])
 (ROOT/'frozen_error_analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['diagnostics','source_hashes']},indent=2))
if __name__=='__main__':main()
