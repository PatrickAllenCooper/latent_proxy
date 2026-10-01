"""Summarize paired label-rotation responses with case-cluster intervals."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def summarize(input_path: Path, output_path: Path, seed: int = 22991, draws: int = 10000) -> None:
    rows = [json.loads(s) for s in input_path.read_text().splitlines() if s.strip()]
    cases = defaultdict(dict)
    for row in rows:
        key = (row['user_id'], row['scenario_id'])
        subkey = (row['permutation_id'], row['arm'])
        assert subkey not in cases[key]
        assert row['completion'] in 'ABCD' and len(row['completion']) == 1
        assert row['metrics']['action'] == ord(row['completion']) - 65
        cases[key][subkey] = row
    assert len(cases) == 32 and all(len(v) == 8 for v in cases.values())
    rng = np.random.default_rng(seed)
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(input_path),
              'records': len(rows), 'cases': len(cases), 'parse_failures': sum(r['parse_failure'] for r in rows),
              'arms': {}, 'paired': {}}
    for arm in ('scores_only', 'exact_advice'):
        arm_rows = [r for r in rows if r['arm'] == arm]
        case_accuracy = np.array([np.mean([group[(p, arm)]['completion'] == group[(p, arm)]['gold_action'] for p in range(4)]) for group in cases.values()])
        case_regret = np.array([np.mean([group[(p, arm)]['metrics']['normalized_regret'] for p in range(4)]) for group in cases.values()])
        idx = rng.integers(0, len(cases), size=(draws, len(cases)))
        report['arms'][arm] = {
            'records': len(arm_rows), 'correct': int(sum(r['completion'] == r['gold_action'] for r in arm_rows)),
            'accuracy': float(case_accuracy.mean()), 'accuracy_95ci': [float(x) for x in np.quantile(case_accuracy[idx].mean(axis=1), [.025, .975])],
            'mean_normalized_regret': float(case_regret.mean()), 'regret_95ci': [float(x) for x in np.quantile(case_regret[idx].mean(axis=1), [.025, .975])],
            'choice_counts': dict(Counter(r['completion'] for r in arm_rows)),
            'correct_by_gold': {gold: int(sum(r['completion'] == gold for r in arm_rows if r['gold_action'] == gold)) for gold in 'ABCD'},
            'correct_by_permutation': {str(p): int(sum(r['completion'] == r['gold_action'] for r in arm_rows if r['permutation_id'] == p)) for p in range(4)},
        }
    accuracy_delta = np.array([np.mean([int(group[(p, 'scores_only')]['completion'] == group[(p, 'scores_only')]['gold_action']) - int(group[(p, 'exact_advice')]['completion'] == group[(p, 'exact_advice')]['gold_action']) for p in range(4)]) for group in cases.values()])
    regret_delta = np.array([np.mean([group[(p, 'scores_only')]['metrics']['normalized_regret'] - group[(p, 'exact_advice')]['metrics']['normalized_regret'] for p in range(4)]) for group in cases.values()])
    idx = rng.integers(0, len(cases), size=(draws, len(cases)))
    report['paired'] = {'scores_only_minus_exact_accuracy': float(accuracy_delta.mean()),
                        'accuracy_difference_95ci': [float(x) for x in np.quantile(accuracy_delta[idx].mean(axis=1), [.025, .975])],
                        'scores_only_minus_exact_regret': float(regret_delta.mean()),
                        'regret_difference_95ci': [float(x) for x in np.quantile(regret_delta[idx].mean(axis=1), [.025, .975])]}
    output_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cases': report['cases'], 'arms': report['arms'], 'paired': report['paired']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
