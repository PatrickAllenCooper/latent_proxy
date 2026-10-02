"""Validate and summarize the paired proxy-result handoff comparison."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ARMS = ('concatenated', 'compact_user', 'tool_role')


def summarize(source: Path, output: Path) -> None:
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    groups = defaultdict(dict)
    for r in rows:
        key = (r['user_id'], r['scenario_id'], r['permutation_id'])
        assert r['arm'] not in groups[key]
        assert r['gold_action'] == chr(65 + max(range(4), key=lambda i: r['expected_utilities'][i]))
        if r['final_action'] is not None:
            assert r['metrics']['action'] == ord(r['final_action']) - 65
        groups[key][r['arm']] = r
    assert all(set(g) == set(ARMS) for g in groups.values())
    rng = np.random.default_rng(26021)
    case_ids = sorted({(k[0], k[1]) for k in groups})
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(source),
              'records': len(rows), 'paired_rotations': len(groups), 'cases': len(case_ids),
              'parse_failures': sum(r['parse_failure'] for r in rows), 'arms': {}, 'paired': {}}
    case_regrets = {}
    for arm in ARMS:
        subset = [r for r in rows if r['arm'] == arm]
        case_regrets[arm] = np.array([np.mean([r['metrics']['normalized_regret'] if r['metrics'] else 1.0
            for k, g in groups.items() if k[:2] == case for r in [g[arm]]]) for case in case_ids])
        idx = rng.integers(0, len(case_ids), size=(10000, len(case_ids)))
        report['arms'][arm] = {'records': len(subset), 'correct': sum(r['final_action'] == r['gold_action'] for r in subset),
            'parse_failures': sum(r['parse_failure'] for r in subset),
            'final_action_counts': dict(Counter(r['final_action'] for r in subset)),
            'mean_normalized_regret': float(case_regrets[arm].mean()),
            'regret_95ci': [float(x) for x in np.quantile(case_regrets[arm][idx].mean(axis=1), [.025, .975])],
            'correct_by_gold': {gold: sum(r['final_action'] == gold for r in subset if r['gold_action'] == gold)
                                for gold in 'ABCD'}}
    for arm in ARMS[1:]:
        delta = case_regrets[arm] - case_regrets['concatenated']
        idx = rng.integers(0, len(case_ids), size=(10000, len(case_ids)))
        report['paired'][f'{arm}_minus_concatenated'] = {'mean_regret_difference': float(delta.mean()),
            'case_cluster_bootstrap_95ci': [float(x) for x in np.quantile(delta[idx].mean(axis=1), [.025, .975])]}
    report['failures'] = [{k: r[k] for k in ('user_id', 'scenario_id', 'permutation_id', 'arm', 'gold_action',
                                          'final_action', 'completion')} for r in rows
                          if r['final_action'] != r['gold_action']]
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('records', 'arms', 'paired')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    summarize(a.input, a.output)
