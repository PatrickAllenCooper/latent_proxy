"""One CPU diagnostic budget for token prep plus full model/runtime hashes."""
import json,runpy,sys,time
from pathlib import Path
root=Path(sys.argv[1]);print(json.dumps({'event':'combined_CPU_preparation_start','at':time.time()}),flush=True)
sys.argv=[str(root/'prepare_eligibility_tokens_v5.py'),'--manifest',str(root/'manifest.jsonl'),'--cache-receipt',str(root/'cache_receipt.json'),'--environment',str(root/'environment_before.json'),'--output',str(root/'token_receipt.json')]
runpy.run_path(str(root/'prepare_eligibility_tokens_v5.py'),run_name='__main__')
sys.argv=[str(root/'recheck_factorial_hashes_v56.py'),str(root)]
runpy.run_path(str(root/'recheck_factorial_hashes_v56.py'),run_name='__main__')
print(json.dumps({'event':'combined_CPU_preparation_complete','at':time.time()}),flush=True)
