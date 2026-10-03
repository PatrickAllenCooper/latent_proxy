"""Paired CPU ablation of the preference inputs to frozen proxy policies."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

from src.evaluation.preference_benchmark import make_scenario, expected_utilities
from src.training.finite_menu_ppo import FiniteMenuPolicy
from src.training.synthetic_users import SyntheticUserSampler, UserType


def run(checkpoints, output, seed, noise_study=False):
    torch.set_num_threads(2)
    users = SyntheticUserSampler(seed=seed).sample_batch(200)
    menus = [make_scenario(seed + 1000000 + i) for i in range(20)]
    rows = []
    model_paths = dict(checkpoints)
    if noise_study:
        model_paths['exact_reward'] = None
    arms = ('true', 'noise_025', 'noise_075', 'generic') if noise_study else ('true', 'swapped', 'generic')
    for name, path in model_paths.items():
        policy = None
        if path is not None:
            policy = FiniteMenuPolicy()
            policy.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            policy.eval()
        for uid, theta in enumerate(users):
            inputs = {'true': theta, 'swapped': users[(uid + 1) % len(users)],
                      'generic': UserType(.5, 1., 2.)}
            if noise_study:
                z = np.random.default_rng(seed + 200000 + uid).normal(size=3)
                for label, sigma in [('noise_025', .25), ('noise_075', .75)]:
                    logit = np.log(theta.gamma / (1 - theta.gamma)) + sigma * z[0]
                    inputs[label] = UserType(float(1 / (1 + np.exp(-logit))),
                        float(theta.alpha * np.exp(sigma * z[1])),
                        float(1 + (theta.lambda_ - 1) * np.exp(sigma * z[2])))
                inputs = {arm: inputs[arm] for arm in arms}
            for menu in menus:
                values = expected_utilities(menu, theta)
                span = max(float(values.max() - values.min()), 1e-10)
                for arm, supplied in inputs.items():
                    action = policy.act(menu, supplied) if policy is not None else int(expected_utilities(menu, supplied).argmax())
                    rows.append({'model': name, 'arm': arm, 'user_id': uid,
                        'scenario_id': menu.scenario_id, 'gamma': theta.gamma,
                        'alpha': theta.alpha, 'lambda': theta.lambda_,
                        'supplied_gamma': supplied.gamma, 'supplied_alpha': supplied.alpha,
                        'supplied_lambda': supplied.lambda_, 'action': action,
                        'gold_action': int(values.argmax()),
                        'normalized_regret': float((values.max() - values[action]) / span)})
    assert len(rows) == len(model_paths) * len(arms) * 4000
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_decision.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r['model'], r['arm'], r['user_id']].append(r)
    rng = np.random.default_rng(26606)
    idx = rng.integers(0, 200, size=(10000, 200))
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': seed,
              'n_users': 200, 'n_menus': 20, 'checkpoints': {k: str(v) for k,v in checkpoints.items()},
              'noise_coordinates': 'Gaussian perturbations to logit(gamma), log(alpha), log(lambda-1), common direction across noise levels' if noise_study else None,
              'models': {}}
    for model in model_paths:
        means = {arm: np.array([np.mean([r['normalized_regret'] for r in groups[model,arm,u]])
                                for u in range(200)]) for arm in arms}
        result = {'mean_regret': {a: float(v.mean()) for a,v in means.items()}, 'paired': {}}
        for arm in arms[1:]:
            delta = means[arm] - means['true']
            changed = sum(groups[model,arm,u][i]['action'] != groups[model,'true',u][i]['action']
                          for u in range(200) for i in range(20))
            result['paired'][arm + '_minus_true'] = {'mean_regret_difference': float(delta.mean()),
                'user_cluster_bootstrap_95ci': [float(x) for x in np.quantile(delta[idx].mean(axis=1), [.025,.975])],
                'changed_actions': changed, 'decisions': 4000}
        report['models'][model] = result
    report['candidate_minus_frozen'] = {}
    for arm in arms:
        delta = np.array([np.mean([r['normalized_regret'] for r in groups['natural_imitation',arm,u]])
                          - np.mean([r['normalized_regret'] for r in groups['frozen',arm,u]])
                          for u in range(200)])
        report['candidate_minus_frozen'][arm] = {'mean_regret_difference': float(delta.mean()),
            'user_cluster_bootstrap_95ci': [float(x) for x in np.quantile(delta[idx].mean(axis=1), [.025,.975])]}
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['models']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=9601)
    p.add_argument('--noise-study', action='store_true')
    a = p.parse_args()
    run({'frozen': a.frozen, 'natural_imitation': a.candidate}, a.output_dir, a.seed, a.noise_study)
