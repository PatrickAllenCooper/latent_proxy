"""Summarize controlled voluntary textual proxy-tool calls and adherence."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ARMS = ('direct', 'optional_tool', 'forced_tool')


def summarize(input_path: Path, output_path: Path, seed: int = 24401, draws: int = 10000) -> None:
    rows = [json.loads(s) for s in input_path.read_text().splitlines() if s.strip()]
    groups = defaultdict(dict)
    for row in rows:
        key = row['user_id'], row['scenario_id']
        subkey = row['permutation_id'], row['arm']
        assert subkey not in groups[key]
        assert row['parse_failure'] is False
        assert row['metrics']['action'] == ord(row['final_action']) - 65
        groups[key][subkey] = row
    assert len(rows) == 192 and len(groups) == 16 and all(len(g) == 12 for g in groups.values())
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(input_path),
              'records': len(rows), 'cases': len(groups), 'parse_failures': sum(r['parse_failure'] for r in rows),
              'protocol': 'Textual TOOL request routed by a controlled harness to an exact argmax proxy; this is not native API function calling.',
              'arms': {}, 'paired': {}}
    rng = np.random.default_rng(seed)
    for arm in ARMS:
        subset = [r for r in rows if r['arm'] == arm]
        case_regret = np.array([np.mean([g[(p, arm)]['metrics']['normalized_regret'] for p in range(4)]) for g in groups.values()])
        idx = rng.integers(0, len(groups), size=(draws, len(groups)))
        report['arms'][arm] = {'records': len(subset), 'correct': sum(r['final_action'] == r['gold_action'] for r in subset),
            'mean_normalized_regret': float(case_regret.mean()),
            'regret_95ci': [float(x) for x in np.quantile(case_regret[idx].mean(axis=1), [.025, .975])],
            'first_completion_counts': dict(Counter(r['first_completion'] for r in subset)),
            'final_action_counts': dict(Counter(r['final_action'] for r in subset)),
            'tool_calls': sum(r['tool_called'] for r in subset),
            'correct_by_gold': {gold: sum(r['final_action'] == gold for r in subset if r['gold_action'] == gold) for gold in 'ABCD'}}
    optional = [r for r in rows if r['arm'] == 'optional_tool']
    report['optional_tool_adherence'] = {'called': sum(r['tool_called'] for r in optional),
        'followed_result': sum(r['tool_called'] and r['final_action'] == r['tool_result']['recommended_action'] for r in optional),
        'failures': [{'user_id': r['user_id'], 'scenario_id': r['scenario_id'],
                      'permutation_id': r['permutation_id'], 'tool_advice': r['tool_result']['recommended_action'],
                      'final_action': r['final_action'], 'raw_final': r['final_completion']} for r in optional
                     if r['tool_called'] and r['final_action'] != r['tool_result']['recommended_action']]}
    for arm, reference in (('optional_tool', 'direct'), ('forced_tool', 'optional_tool')):
        delta = np.array([np.mean([g[(p, arm)]['metrics']['normalized_regret'] - g[(p, reference)]['metrics']['normalized_regret'] for p in range(4)]) for g in groups.values()])
        idx = rng.integers(0, len(groups), size=(draws, len(groups)))
        report['paired'][f'{arm}_minus_{reference}_regret'] = {'mean': float(delta.mean()),
            'case_cluster_bootstrap_95ci': [float(x) for x in np.quantile(delta[idx].mean(axis=1), [.025, .975])]}
    output_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'arms': report['arms'], 'optional_tool_adherence': report['optional_tool_adherence'], 'paired': report['paired']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
