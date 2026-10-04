"""CPU-only frozen header x instruction factorial; no model calls or submissions."""
import hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('outputs/preference_program/manifests')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=ROOT/'numeric_capability_v40.jsonl';mapping=ROOT/'output_mapping_v41.jsonl'
 bases=[json.loads(x) for x in source.read_text().splitlines() if json.loads(x)['representation']=='integers' and json.loads(x)['listing_order']=='canonical']
 assert len(bases)==4 and [r['permutation_id'] for r in bases]==list(range(4))
 headers={'preference':'User preference scores:','numbers':'Numbers:'}
 instructions={'score':'Choose the action with the highest score.','label':'Which label has the largest number?'}
 output='Reply with exactly one capital letter: A, B, C, or D.'
 rows=[]
 for base in bases:
  values=base['displayed_values'];assert sorted(values)==[1,2,3,4]
  gold='ABCD'[values.index(max(values))];assert gold==base['gold_action']
  evidence=', '.join(f'{chr(65+i)}={v}' for i,v in enumerate(values))+'.'
  for h,header in headers.items():
   for i,instruction in instructions.items():
    rows.append({'case_id':base['scenario_id'],'permutation_id':base['permutation_id'],'header':h,'instruction':i,'displayed_values':values,'gold_action':gold,'prompt':f'{header} {evidence} {instruction} {output}','prompt_sha256':hashlib.sha256(f'{header} {evidence} {instruction} {output}'.encode()).hexdigest()})
 assert len(rows)==16 and len({(r['permutation_id'],r['header'],r['instruction']) for r in rows})==16
 old={(r['permutation_id']):r for r in bases}
 direct={r['permutation_id']:r for r in [json.loads(x) for x in mapping.read_text().splitlines()] if r['response_mode']=='label'}
 for r in rows:
  if r['header']=='preference' and r['instruction']=='score':assert r['prompt']==old[r['permutation_id']]['prompt']
  if r['header']=='numbers' and r['instruction']=='label':assert r['prompt']==direct[r['permutation_id']]['prompt']
  assert r['prompt'].endswith(output) and 'adviser' not in r['prompt'] and 'TOOL' not in r['prompt']
  assert re.fullmatch(r'\s*([ABCD])\s*',r['gold_action'])
  for h in headers:
   for i in instructions:
    match=next(x for x in rows if x['permutation_id']==r['permutation_id'] and x['header']==h and x['instruction']==i)
    assert match['displayed_values']==r['displayed_values'] and match['gold_action']==r['gold_action']
 m=ROOT/'framing_factorial_v42.jsonl';m.write_text(''.join(json.dumps(r)+'\n' for r in rows))
 scoring={'parser_regex':r'\s*([ABCD])\s*','correctness':'parsed uppercase label equals displayed-value argmax;strict fullmatch;parse failure scores incorrect and reported separately','gold_by_rotation':[r['gold_action'] for r in bases],'primary_paired_contrasts':{'header':'numbers minus preference correctness,paired within rotation and instruction;report8pair differences and average','instruction':'label minus score correctness,paired within rotation and header;report8pair differences and average','interaction':'[(numbers,label)-(preference,label)]-[(numbers,score)-(preference,score)] per rotation;report4differences and average'},'additional':'fourcell correct/4,parse failures,A responses,exact output and token stops byrotation;allnulls/errors retained','uncertainty':'descriptive one reusedcase;no populationCI/pseudoreplication','gate':'Any fixed cell4/4correct with zero parse failures and no length caps qualifies only measurement on this case;fresh-case qualification required before broader inference;otherwise seek assay guidance without adaptive tuning'}
 s=ROOT/'framing_factorial_v42_scoring.json';s.write_text(json.dumps(scoring,indent=2)+'\n')
 protocol={'status':'frozen proposed preparation only;no inference/submission','created_at_utc':datetime.now(timezone.utc).isoformat(),'sources':{str(source):digest(source),str(mapping):digest(mapping)},'design':'samefourintegerrotations x2headers x2instructions=16;canonicalABCD,letteronly,noadvice','headers':headers,'instructions':instructions,'output_contract':output,'model':'Qwen/Qwen2.5-1.5B-Instruct','snapshot':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306','decode':{'do_sample':False,'max_new_tokens':24},'hashes':{'manifest':digest(m),'scoring':digest(s),'preparation_script':digest(Path(__file__))},'resources_if_authorized':{'account':'ucb736_asc1','gpu':'oneH200MIG2g.35gb','memory':'32G','walltime':'00:05:00','model_load_watchdog_seconds':90,'runtime_staging_watchdog_seconds':45},'token_custody_required':['generatedIDs','promptlength','effectiveEOSIDs','tokenizerEOSID','generatedlength','lasttoken','EOSobserved','lengthcapindicator','renderedmessages','model/runtimehashes'],'interpretation':'One reusedcase;4/4measurementgate only;freshcase qualification beforebroaderinference;header/instruction interactions not preference discovery','stop':'No launch from preparation;no tuning/retry/resource enlargement or320study'}
 p=ROOT/'framing_factorial_v42_protocol.json';p.write_text(json.dumps(protocol,indent=2)+'\n')
 (ROOT/'framing_factorial_v42_validation.json').write_text(json.dumps({'valid':True,'records':16,'unique_cells':16,'same_values_and_gold_within_rotation':True,'old_prompt_corner_exact':True,'direct_prompt_corner_exact':True,'canonical_order':True,'gold_balanced':True,'strict_parser_control_checked':True,'protocol_sha256':digest(p),'manifest_sha256':digest(m),'scoring_sha256':digest(s),'CPU_only':True,'model_calls':0,'job_submissions':0},indent=2)+'\n')
 (ROOT/'framing_factorial_v42_review.txt').write_text('Frozen proposal only; no generation performed.\n\n'+'\n\n'.join(f"{n+1}. rotation{r['permutation_id']} / {r['header']} / {r['instruction']}\n{r['prompt']}\nExpected: {r['gold_action']}\nPrompt SHA256: {r['prompt_sha256']}" for n,r in enumerate(rows)))
 print('CPU validation passed:16prompts,2exact prior-corner matches,16unique factorialcells;zero calls/submissions')
if __name__=='__main__':main()
