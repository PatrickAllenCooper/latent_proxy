"""Prepare a balanced, paired pilot isolating numeric proxy-score reading."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.run_finite_menu_llm_pilot import format_prompt
from src.evaluation.preference_benchmark import BENCHMARK_VERSION, evaluate_action, expected_utilities, make_scenario
from src.training.synthetic_users import SyntheticUserSampler

ARMS = ('baseline', 'full_scores', 'scores_only', 'exact_advice')


def prepare(output: Path, n_users: int, seed: int) -> None:
    users = SyntheticUserSampler(seed=seed).sample_batch(max(200, 8 * n_users))
    rows = []
    selected = {a: [] for a in range(4)}
    used_users = set()
    for user_id, theta in enumerate(users):
        scenarios = [make_scenario(seed + 1_000_000 + user_id * 2000 + i) for i in range(2000)]
        for scenario in scenarios:
            values = expected_utilities(scenario, theta)
            gold = int(np.argmax(values))
            if len(selected[gold]) < n_users and user_id not in used_users:
                selected[gold].append((user_id, theta, scenario, values))
                used_users.add(user_id)
                break
        if all(len(v) == n_users for v in selected.values()):
            break
    assert all(len(v) == n_users for v in selected.values()), {k: len(v) for k, v in selected.items()}
    for gold, group in selected.items():
        for user_id, theta, scenario, values in group:
            base = format_prompt(scenario, theta, None)
            head, instruction = base.rsplit('\n', 1)
            scores = ', '.join(f'{chr(65 + a)}={values[a]:+.6f}' for a in range(4))
            variants = {
                'baseline': base,
                'full_scores': f'{head}\nPreference proxy supplied these expected user utilities: {scores}. These are signed decimal numbers. Compare their numeric values and choose the action with the largest number.\n{instruction}',
                'scores_only': f'Preference proxy scores for four actions are signed decimals: {scores}. Choose the letter with the largest numeric score. Reply with exactly one capital letter: A, B, C, or D.',
                'exact_advice': f'{head}\nPreference proxy recommendation: {chr(65 + gold)}. Follow this recommendation.\n{instruction}',
            }
            metrics = [evaluate_action(scenario, theta, a) for a in range(4)]
            for arm in ARMS:
                rows.append({'benchmark_version': BENCHMARK_VERSION, 'seed': seed,
                             'user_id': user_id, 'scenario_id': scenario.scenario_id,
                             'arm': arm, 'true_theta': theta.__dict__,
                             'expected_utilities': [float(v) for v in values],
                             'gold_action': chr(65 + gold), 'wrong_action': chr(65 + int(np.argmin(values))),
                             'metrics_by_action': metrics, 'prompt': variants[arm]})
    assert Counter(r['gold_action'] for r in rows if r['arm'] == 'baseline') == {x: n_users for x in 'ABCD'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    print(json.dumps({'event': 'cpu_preparation_complete', 'at_utc': datetime.now(timezone.utc).isoformat(),
                      'records': len(rows), 'path': str(output), 'gold_per_action': n_users}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-users', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7001)
    args = parser.parse_args()
    prepare(args.output, args.n_users, args.seed)
