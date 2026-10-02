"""Paired CPU evaluation of frozen and imitation-updated proxy policies."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.evaluation.preference_benchmark import expected_utilities, make_scenario
from src.training.finite_menu_ppo import FiniteMenuPolicy
from src.training.synthetic_users import SyntheticUserSampler


def evaluate(checkpoints: dict[str, Path], output_dir: Path, seed: int,
             n_users: int, n_scenarios: int) -> None:
    torch.set_num_threads(2)
    models = {}
    for name, path in checkpoints.items():
        model = FiniteMenuPolicy()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model.eval()
        models[name] = model
    users = SyntheticUserSampler(seed=seed).sample_batch(n_users)
    scenarios = [make_scenario(seed + 1_000_000 + i) for i in range(n_scenarios)]
    rows = []
    for user_id, theta in enumerate(users):
        for scenario in scenarios:
            values = expected_utilities(scenario, theta)
            gold = int(np.argmax(values))
            span = max(float(values.max() - values.min()), 1e-10)
            for name, model in models.items():
                action = model.act(scenario, theta)
                rows.append({'seed': seed, 'user_id': user_id,
                    'scenario_id': scenario.scenario_id, 'arm': name,
                    'gamma': theta.gamma, 'alpha': theta.alpha, 'lambda': theta.lambda_,
                    'gold_action': gold, 'action': action,
                    'normalized_regret': float((values.max() - values[action]) / span)})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'per_decision.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_user = defaultdict(lambda: defaultdict(list))
    by_arm = defaultdict(list)
    for row in rows:
        by_user[row['user_id']][row['arm']].append(row)
        by_arm[row['arm']].append(row)
    rng = np.random.default_rng(26603)
    idx = rng.integers(0, n_users, size=(10000, n_users))
    user_means = {arm: np.array([np.mean([r['normalized_regret'] for r in by_user[u][arm]])
                                  for u in range(n_users)]) for arm in checkpoints}
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'seed': seed, 'n_users': n_users, 'n_scenarios': n_scenarios,
        'checkpoints': {name: str(path) for name, path in checkpoints.items()},
        'arms': {}, 'paired_vs_frozen': {}}
    for arm, subset in by_arm.items():
        vals = user_means[arm]
        summary['arms'][arm] = {'decisions': len(subset),
            'mean_normalized_regret': float(vals.mean()),
            'user_cluster_bootstrap_95ci': [float(x) for x in np.quantile(vals[idx].mean(axis=1), [.025, .975])],
            'oracle_match_rate': float(np.mean([r['action'] == r['gold_action'] for r in subset])),
            'action_counts': dict(Counter(str(r['action']) for r in subset)),
            'by_gold': {str(g): {'decisions': sum(r['gold_action'] == g for r in subset),
                'oracle_match_rate': float(np.mean([r['action'] == g for r in subset if r['gold_action'] == g])),
                'mean_normalized_regret': float(np.mean([r['normalized_regret'] for r in subset if r['gold_action'] == g]))}
                for g in range(4)}}
    for arm in checkpoints:
        if arm == 'frozen':
            continue
        delta = user_means[arm] - user_means['frozen']
        summary['paired_vs_frozen'][arm] = {'mean_regret_difference': float(delta.mean()),
            'user_cluster_bootstrap_95ci': [float(x) for x in np.quantile(delta[idx].mean(axis=1), [.025, .975])]}
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'arms': summary['arms'], 'paired_vs_frozen': summary['paired_vs_frozen']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--balanced', type=Path, required=True)
    p.add_argument('--mixed', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=9401)
    p.add_argument('--n-users', type=int, default=200)
    p.add_argument('--n-scenarios', type=int, default=20)
    a = p.parse_args()
    evaluate({'frozen': a.frozen, 'balanced': a.balanced, 'mixed': a.mixed},
             a.output_dir, a.seed, a.n_users, a.n_scenarios)
