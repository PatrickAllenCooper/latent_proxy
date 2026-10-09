"""CPU-only finite preference protocol, constrained tool and strict scorer.
No model calls, training, optimizer execution or scheduler operations.
"""
import hashlib,itertools,json,random,re
from pathlib import Path
ATTR=('quiet','spacious','portable','durable')
ARMS=('no_preferences','explicit_profile','proxy_tool','validated_hybrid')
MODEL={'name':'Qwen/Qwen2.5-32B-Instruct','revision':'5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd','precision':'BF16','do_sample':False,'max_new_tokens':24,'adapters':False}

def validate(c):
    assert len(c['priority'])==4 and set(c['priority'])==set(ATTR)
    assert len(c['options'])==4 and {o['label'] for o in c['options']}==set('ABCD')
    assert {o['attribute'] for o in c['options']}==set(ATTR)
    assert all(type(o['eligible']) is bool for o in c['options']) and sum(o['eligible'] for o in c['options'])==3

def rewards(c):
    validate(c)
    return {o['label']:{'reward':3-c['priority'].index(o['attribute']),'eligible':o['eligible']} for o in c['options']}

def oracle(c):
    r=rewards(c);return max((l for l in r if r[l]['eligible']),key=lambda l:r[l]['reward'])

def independent_oracle(c):
    return next(o['label'] for a in c['priority'] for o in c['options'] if o['attribute']==a and o['eligible'])

class ProxySession:
    def __init__(self,case,budget=2):
        validate(case);self.case=case;self.budget=budget;self.trace=[]
    def call(self,request):
        if len(self.trace)>=self.budget:raise ValueError('tool budget exhausted')
        if type(request) is not dict or request.get('operation') not in ('evaluate_candidates','recommend_constrained'):raise ValueError('invalid tool operation')
        if request['operation']=='recommend_constrained':
            if set(request)!={'operation'}:raise ValueError('unexpected field')
            result={'action':oracle(self.case)}
        else:
            if set(request)!={'operation','labels'} or not isinstance(request['labels'],list) or not request['labels'] or len(set(request['labels']))!=len(request['labels']) or any(l not in 'ABCD' or len(l)!=1 for l in request['labels']):raise ValueError('invalid candidate labels')
            result={l:rewards(self.case)[l] for l in request['labels']}
        self.trace.append({'request':request,'result':result});return result

def score(c,raw):
    r=rewards(c);m=re.fullmatch(r'\s*([ABCD])\s*',raw) if isinstance(raw,str) else None
    action=m[1] if m else None;gold=oracle(c)
    # Ineligible actions receive finite prespecified penalty -1; oracle excludes them.
    utility=-1 if action is None or not r[action]['eligible'] else r[action]['reward']
    worst=min([-1]+[v['reward'] for v in r.values() if v['eligible']]);best=r[gold]['reward']
    return {'action':action,'gold':gold,'correct':action==gold,'parse_failure':action is None,'constraint_violation':action is not None and not r[action]['eligible'],'normalized_regret':(best-utility)/(best-worst)}

def discovery_score(c,inferred_priority):
    if len(inferred_priority)!=4 or set(inferred_priority)!=set(ATTR):raise ValueError('invalid inferred preference order')
    pairs=list(itertools.combinations(ATTR,2))
    error=sum((c['priority'].index(a)<c['priority'].index(b))!=(inferred_priority.index(a)<inferred_priority.index(b)) for a,b in pairs)/len(pairs)
    inferred=dict(c,priority=list(inferred_priority))
    return {'pairwise_order_error':error,'proxy_action':oracle(inferred),'true_preference_decision':score(c,oracle(inferred))}

def audit_response(c,arm,record):
    if arm not in ARMS or record['case_id']!=c['case_id'] or record['arm']!=arm:raise ValueError('record identity')
    if record['model_revision']!=MODEL['revision']:raise ValueError('model identity')
    if not record['EOS_observed'] or record['length_cap_reached']:raise ValueError('incomplete response')
    if not record['generated_token_ids'] or len(record['generated_token_ids'])>MODEL['max_new_tokens']:raise ValueError('token bound')
    session=ProxySession(c,budget=2 if arm in ('proxy_tool','validated_hybrid') else 0)
    for event in record['tool_trace']:
        if session.call(event['request'])!=event['result']:raise ValueError('tool result mismatch')
    result={'reader':score(c,record['raw_final_output']),'tool_calls':len(session.trace)}
    if arm=='validated_hybrid':result['hybrid']=hybrid(c,record['raw_final_output'])
    return result

def hybrid(c,raw):
    original=score(c,raw);action=oracle(c)
    return {'original':original,'final':score(c,action),'corrected':original['action']!=action,'final_action':action,'architecture':'forced_oracle_override_not_LLM_capability'}

def prompt(c,arm):
    if arm not in ARMS:raise ValueError('arm')
    menu='\n'.join(f"{o['label']}: attribute={o['attribute']}; eligible={'yes' if o['eligible'] else 'no'}." for o in c['options'])
    head='Choose an eligible option. Output exactly one capital letter A, B, C, or D.'
    if arm=='explicit_profile':head+=' Priority list: '+', then '.join(c['priority'])+'. Choose the highest-priority eligible option.'
    if arm in ('proxy_tool','validated_hybrid'):head+=' You may call evaluate_candidates or recommend_constrained at most twice to learn this user\'s preferences. Use the returned information to choose.'
    return head+'\n'+menu

def freeze(root=Path('outputs/preference_program/manifests/preference_use_cycle_v70')):
    rng=random.Random(2026100907);users=list(itertools.permutations(ATTR));rng.shuffle(users)
    groups={a:[] for a in ATTR}
    for attrs in itertools.permutations(ATTR):
        for ex in range(4):groups[attrs[ex]].append((list(attrs),ex))
    for group in groups.values():rng.shuffle(group)
    menus=[groups[a][i] for i in range(24) for a in ATTR]
    rows=[];splits={};bounds={'development':(users[:12],menus[:48]),'validation':(users[12:18],menus[48:72]),'heldout':(users[18:],menus[72:])}
    for split,(profiles,catalog) in bounds.items():
        splits[split]={'users':[list(p) for p in profiles],'menus':[{'attributes':a,'excluded_index':e} for a,e in catalog]}
        for u,priority in enumerate(profiles):
            for k in range(4):
                attrs,ex=catalog[u*4+k];c={'case_id':f'{split}-{u:02d}-{k}','user_id':f'{split}-{u:02d}','task_id':f'{split}-{u*4+k:02d}','split':split,'priority':list(priority),'options':[{'label':l,'attribute':a,'eligible':i!=ex} for i,(l,a) in enumerate(zip('ABCD',attrs))]};c['gold']=oracle(c);rows.append(c)
    smoke=[c for c in rows if c['split']=='heldout' and c['user_id'] in ('heldout-00','heldout-01')]
    root.mkdir(parents=True,exist_ok=False)
    def save(name,data): (root/name).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    save('cases.json',rows);save('splits.json',splits)
    # Private case/reward truth is absent from these model-visible initial messages.
    save('smoke_requests.json',[{'case_id':c['case_id'],'arm':a,'messages':[{'role':'user','content':prompt(c,a)}],'tool_budget':2 if a in ('proxy_tool','validated_hybrid') else 0} for c in smoke for a in ARMS])
    qualification=Path('outputs/preference_program/manifests/reader32_qualification_v60')
    save('protocol.json',{'status':'frozen_preparation_only','model':MODEL,'qualification':{'cases_path':str(qualification/'cases.jsonl'),'cases_sha256':hashlib.sha256((qualification/'cases.jsonl').read_bytes()).hexdigest(),'gate':'16/16; zero parse failures/truncation; valid runtime/custody; no prompt tuning','reuse':'unchanged frozen panel; no completed 32B responses'},'smoke':{'independent_users':2,'independent_cases':8,'arms':list(ARMS),'final_answers':32,'max_tool_calls_per_answer':2,'model_call_ceiling':64,'interpretation':'paired descriptive mechanism pilot; not population evidence'},'discovery':{'arms':['random','EIG','decision_focused_AIF'],'paired_users':True,'questions_per_user':8,'noise_conditions':'separate future stress experiment','metrics':['parameter_or_order_error','decision_regret','question_count'],'status':'design_only; existing domain elicitors retained'},'proxy':{'kind':'deterministic constrained rank reward; not learned RL','learned_policy':'separate future approximation arm'},'optimization':{'optimizer':'GEPA','status':'not executed; no dependency installed','candidate_ceiling':32,'development_feedback_evaluation_ceiling':256,'validation_evaluations_after_selection':24,'heldout_feedback_allowed':False,'budget_admission':'future resource proposal required'},'fine_tuning':{'status':'dataset prepared; no training','starting_checkpoint':MODEL,'new_adapter_arm':True,'training_examples':48,'weight_change_check_required':True,'stated_profile_regression_required':True,'training_recipe':'must be frozen in a separate proposal before training'},'enforcement_metrics':['raw_choice_correctness','normalized_regret','eligibility_violation','parse_failure','tool_calls','hybrid_correction'],'required_output_fields':['case_id','arm','raw_final_output','tool_trace','generated_token_ids','EOS_observed','length_cap_reached','model_revision','config_hash','seed','timestamps'],'statistics':'user-cluster paired intervals only after adequately powered confirmation; no CI on this two-user smoke','GPU_admission':False,'automatic_retry':False})
    save('training_examples.json',[{'case_id':c['case_id'],'messages':[{'role':'user','content':prompt(c,'explicit_profile')},{'role':'assistant','content':c['gold']}]} for c in rows if c['split']=='development'])
    exhaustive=0
    for priority in itertools.permutations(ATTR):
        for attrs in itertools.permutations(ATTR):
            for ex in range(4):
                c={'priority':list(priority),'options':[{'label':l,'attribute':a,'eligible':i!=ex} for i,(l,a) in enumerate(zip('ABCD',attrs))]}
                assert oracle(c)==independent_oracle(c);assert score(c,oracle(c))['normalized_regret']==0;exhaustive+=1
    save('CPU_validation.json',{'oracle_cases':exhaustive,'errors':0,'model_calls':0,'qualification_passed':False,'datasets':{s:sum(c['split']==s for c in rows) for s in bounds},'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}})
    return root
if __name__=='__main__':print(freeze())
