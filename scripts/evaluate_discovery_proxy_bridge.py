"""Reanalyze saved discovery estimates through frozen CPU decision mechanisms."""
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from src.evaluation.preference_benchmark import make_scenario, expected_utilities, evaluate_action
from src.training.synthetic_users import UserType
from src.training.finite_menu_ppo import FiniteMenuPolicy

ROOT = Path('outputs/preference_program/results')

def main():
    torch.set_num_threads(2)
    source = ROOT / 'finite_menu_discovery_replication_v2'
    config = json.loads((source / 'summary.json').read_text())['config']
    inputs = list(csv.DictReader((source / 'per_user.csv').open()))
    menus = [make_scenario(config['seed'] + 300000 + i) for i in range(config['n_targets'])]
    paths = {'frozen': ROOT / 'finite_menu_ppo_pilot_v2/policy.pt',
             'natural_imitation': ROOT / 'natural_imitation_cpu_probe_v27/policy.pt'}
    policies = {}
    for name, path in paths.items():
        policy = FiniteMenuPolicy()
        policy.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        policy.eval(); policies[name] = policy
    decisions, users = [], []
    for r in inputs:
        true = UserType(*[float(r['true_' + k]) for k in ('gamma','alpha','lambda')])
        estimate = UserType(*[float(r['estimated_' + k]) for k in ('gamma','alpha','lambda')])
        base = {k:r[k] for k in ('seed','user_id','arm','budget')}
        for name in (*policies, 'point_exact'):
            regrets = []
            for menu in menus:
                action = policies[name].act(menu, estimate) if name in policies else int(expected_utilities(menu, estimate).argmax())
                regret = evaluate_action(menu, true, action)['normalized_regret']
                assert 0 <= regret <= 1 + 1e-10
                regrets.append(regret)
                decisions.append(dict(base, model=name, scenario_id=menu.scenario_id, action=action, normalized_regret=regret))
            users.append(dict(base, model=name, normalized_regret=float(np.mean(regrets))))
        users.append(dict(base, model='posterior_exact', normalized_regret=float(r['reward_regret'])))
    assert len(decisions) == 3200 * 3 * config['n_targets']
    out = ROOT / 'discovery_proxy_bridge_v32'; out.mkdir(exist_ok=True)
    for filename, rows in [('per_decision.csv',decisions),('per_user.csv',users)]:
        with (out/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rng=np.random.default_rng(32001)
    idx=rng.integers(0,config['n_users'],size=(10000,config['n_users']))
    cells={}
    for arm in config['arms']:
        for budget in config['budgets']:
            arrays={model:np.array([r['normalized_regret'] for r in users if r['model']==model and r['arm']==arm and int(r['budget'])==budget]) for model in (*policies,'point_exact','posterior_exact')}
            assert all(len(v)==config['n_users'] for v in arrays.values())
            contrasts={}
            for a,b in [('natural_imitation','frozen'),('point_exact','posterior_exact'),('natural_imitation','posterior_exact')]:
                delta=arrays[a]-arrays[b]
                contrasts[a+'_minus_'+b]={'mean':float(delta.mean()),'user_bootstrap_95':np.quantile(delta[idx].mean(axis=1),[.025,.975]).tolist()}
            cells[f'{arm}/{budget}']={'mean_regret':{k:float(v.mean()) for k,v in arrays.items()},'paired':contrasts}
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),'config':config,'checkpoints':{k:str(v) for k,v in paths.items()},'cells':cells,'validation':{'decision_records':len(decisions),'user_records':len(users),'bounded_regret':True},'limitations':'Reanalysis of the existing analytic discovery panel; posterior actions are unavailable, so posterior reference uses saved per-user regret. No conversational LLM discovery or independent replication.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    for key in ('random/0','random/8','eig/8','decision_value/8','aif_50/8'): print(key,cells[key])

if __name__=='__main__':main()
