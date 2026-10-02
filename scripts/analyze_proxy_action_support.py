"""Audit action support and conditional regret of a frozen finite-menu proxy."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def analyze(source: Path, output: Path) -> None:
    by_key = defaultdict(dict)
    with source.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['user_id'], row['scenario_id']
            assert row['arm'] not in by_key[key]
            by_key[key][row['arm']] = row
    assert len(by_key) == 4000
    assert all('gold_reward_tool' in d and 'learned_rl_proxy' in d for d in by_key.values())
    panel = []
    for (user, scenario), arms in by_key.items():
        gold = arms['gold_reward_tool']
        learned = arms['learned_rl_proxy']
        assert float(gold['normalized_regret']) < 1e-12
        panel.append({'user_id': user, 'scenario_id': scenario,
                      'gold_action': int(gold['action']), 'policy_action': int(learned['action']),
                      'policy_regret': float(learned['normalized_regret'])})
    rng = np.random.default_rng(26602)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(source),
              'decisions': len(panel), 'users': len({r['user_id'] for r in panel}),
              'gold_counts': dict(Counter(str(r['gold_action']) for r in panel)),
              'policy_counts': dict(Counter(str(r['policy_action']) for r in panel)),
              'by_gold': {}}
    for gold in range(4):
        subset = [r for r in panel if r['gold_action'] == gold]
        users = sorted({r['user_id'] for r in subset})
        # Bootstrap conditional regrets by users contributing at least one case.
        user_totals = np.array([sum(r['policy_regret'] for r in subset if r['user_id'] == u)
                                for u in users])
        user_counts = np.array([sum(r['user_id'] == u for r in subset) for u in users])
        idx = rng.integers(0, len(users), size=(10000, len(users)))
        draws = user_totals[idx].sum(axis=1) / user_counts[idx].sum(axis=1)
        report['by_gold'][str(gold)] = {
            'decisions': len(subset), 'users': len(users),
            'policy_action_counts': dict(Counter(str(r['policy_action']) for r in subset)),
            'policy_optimal': sum(r['policy_action'] == gold for r in subset),
            'mean_normalized_regret': float(np.mean([r['policy_regret'] for r in subset])),
            'user_cluster_bootstrap_95ci': [float(x) for x in np.quantile(draws, [.025, .975])],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'decisions': report['decisions'], 'gold_counts': report['gold_counts'],
                      'policy_counts': report['policy_counts'], 'by_gold': report['by_gold']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    analyze(a.input, a.output)
