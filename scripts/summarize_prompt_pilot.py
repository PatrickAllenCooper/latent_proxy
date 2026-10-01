"""Summarize paired finite-menu prompt arms with user-cluster bootstrap intervals."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def summarize(path: Path, output: Path, *, bootstrap_seed: int = 43117, draws: int = 10000) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    present = {row['arm'] for row in rows}
    assert 'baseline' in present
    arms = ('baseline', *sorted(present - {'baseline'}))
    keys = defaultdict(dict)
    for row in rows:
        key = (row['user_id'], row['scenario_id'])
        assert row['arm'] in arms and row['arm'] not in keys[key]
        assert row['parse_failure'] is False
        assert row['metrics']['action'] == ord(row['completion']) - 65
        assert np.isfinite(row['metrics']['normalized_regret'])
        keys[key][row['arm']] = row
    assert all(set(group) == set(arms) for group in keys.values())
    users = sorted({key[0] for key in keys})
    out = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(path),
           'records': len(rows), 'paired_cases': len(keys), 'users': len(users),
           'parse_failures': sum(row['parse_failure'] for row in rows), 'arms': {}, 'paired': {}}
    for arm in arms:
        group = [row for row in rows if row['arm'] == arm]
        out['arms'][arm] = {
            'records': len(group), 'mean_normalized_regret': float(np.mean([r['metrics']['normalized_regret'] for r in group])),
            'optimal_count': sum(r['completion'] == r['gold_action'] for r in group),
            'choices': dict(Counter(r['completion'] for r in group)),
            'per_user_mean_normalized_regret': {str(u): float(np.mean([r['metrics']['normalized_regret'] for r in group if r['user_id'] == u])) for u in users},
        }
    rng = np.random.default_rng(bootstrap_seed)
    for arm in arms[1:]:
        user_delta = np.array([np.mean([group[arm]['metrics']['normalized_regret'] - group['baseline']['metrics']['normalized_regret'] for (u2, _), group in keys.items() if u2 == u]) for u in users])
        draws_idx = rng.integers(0, len(users), size=(draws, len(users)))
        means = user_delta[draws_idx].mean(axis=1)
        out['paired'][f'{arm}_minus_baseline'] = {
            'mean_normalized_regret_difference': float(user_delta.mean()),
            'user_cluster_bootstrap_95ci': [float(x) for x in np.quantile(means, [.025, .975])],
            'user_differences': {str(u): float(v) for u, v in zip(users, user_delta)},
            'changed_cases': [dict(user_id=u, scenario_id=s, baseline=group['baseline']['completion'], treatment=group[arm]['completion'], gold=group[arm]['gold_action']) for (u, s), group in keys.items() if group['baseline']['completion'] != group[arm]['completion']],
        }
    output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'records': out['records'], 'paired_cases': out['paired_cases'], 'paired': out['paired']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
