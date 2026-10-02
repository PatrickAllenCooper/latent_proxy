"""Summarize prompt development with parse failures as explicit non-decisions."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def summarize(input_path: Path, output_path: Path, seed: int = 44117, draws: int = 10000) -> None:
    rows = [json.loads(s) for s in input_path.read_text().splitlines() if s.strip()]
    cases = defaultdict(dict)
    for row in rows:
        key = (row['user_id'], row['scenario_id'])
        subkey = (row['permutation_id'], row['arm'])
        assert subkey not in cases[key]
        cases[key][subkey] = row
    assert len(cases) == 16 and all(len(group) == 16 for group in cases.values())
    arms = ('scores_only', 'vertical_table', 'pairwise', 'code_argmax')
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source': str(input_path),
              'records': len(rows), 'cases': len(cases), 'arms': {}, 'paired': {}}
    rng = np.random.default_rng(seed)
    for arm in arms:
        subset = [r for r in rows if r['arm'] == arm]
        valid = [r for r in subset if not r['parse_failure']]
        report['arms'][arm] = {'records': len(subset), 'parse_failures': len(subset) - len(valid),
                               'correct_of_all': sum(r['completion'] == r['gold_action'] for r in valid),
                               'choice_counts_valid': dict(Counter(r['completion'] for r in valid)),
                               'mean_regret_valid_only': float(np.mean([r['metrics']['normalized_regret'] for r in valid])) if valid else None}
    for arm in ('code_argmax',):
        case_delta = np.array([np.mean([group[(p, arm)]['metrics']['normalized_regret'] - group[(p, 'scores_only')]['metrics']['normalized_regret'] for p in range(4)]) for group in cases.values()])
        idx = rng.integers(0, len(cases), size=(draws, len(cases)))
        report['paired'][f'{arm}_minus_scores_only_regret'] = {
            'mean': float(case_delta.mean()),
            'case_cluster_bootstrap_95ci': [float(x) for x in np.quantile(case_delta[idx].mean(axis=1), [.025, .975])]}
    report['selection_rule'] = 'Exclude arms with any parse failures. Among eligible arms, minimize mean normalized regret on development cases. Submit held-out evaluation only if an alternative improves on scores_only.'
    report['eligible'] = [a for a in arms if report['arms'][a]['parse_failures'] == 0]
    report['selected'] = min(report['eligible'], key=lambda a: report['arms'][a]['mean_regret_valid_only'])
    report['holdout_submitted'] = report['selected'] != 'scores_only'
    output_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
