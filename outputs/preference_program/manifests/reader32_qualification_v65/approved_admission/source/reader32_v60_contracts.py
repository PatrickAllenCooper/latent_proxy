"""Pure admission/budget rules shared by live runner and synthetic fixtures."""
import json,os,threading
from pathlib import Path
GIB=2**30

def memory_budget(total):
 if total<=65*GIB:raise ValueError('insufficient_actual_capacity')
 return min(65*GIB,int(total*.95))

def violation(now,start,phase_deadline,allocated,reserved,budget):
 if now>=min(start+285,phase_deadline):return 'time_limit'
 if max(allocated,reserved)>budget:return 'framework_memory_limit'
 return None

def admit_next(now,start,durations,count):
 if not 0<=count<16:return False
 remaining=start+285-now
 return remaining>=10 and (not durations or remaining>=(16-count)*max(durations)*1.5+15)

def result(records,expected_ids,complete):
 ids=[r['case_id'] for r in records]
 if ids!=expected_ids[:len(ids)] or len(ids)>16:raise ValueError('unexpected_or_duplicate_response')
 if complete and len(ids)!=16:raise ValueError('partial_cannot_complete')
 return {'complete':complete,'records':len(ids),'strict_correct_observed':sum(r['score']['correct'] for r in records),'qualified':all(r['score']['correct'] for r in records) if complete else None,'classification':'completed_qualification' if complete else 'incomplete_resource_or_runtime'}

def export_incomplete(out,reason,at,**details):
 out=Path(out);out.mkdir(exist_ok=True)
 records=[];path=out/'responses.jsonl'
 if path.exists():
  for line in path.read_text().splitlines():
   try:records.append(json.loads(line))
   except json.JSONDecodeError:break
 receipt={'complete':False,'qualified':None,'classification':'incomplete_resource_or_runtime','records':len(records),'reason':reason,'at':at,**details}
 tmp=out/f'.budget-stop-{os.getpid()}-{threading.get_ident()}.json';tmp.write_text(json.dumps(receipt,indent=2)+'\n');os.replace(tmp,out/'budget_stop.json')
 return receipt
