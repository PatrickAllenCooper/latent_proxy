"""Analyze label-by-row-position effects using the frozen table phrase parser."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.reparse_numeric_prompt import parse_completion


def summarize(raw_path: Path, manifest_path: Path, output_dir: Path, seed: int = 78211, draws: int = 10000) -> None:
    raw = [json.loads(s) for s in raw_path.read_text().splitlines() if s.strip()]
    manifest = [json.loads(s) for s in manifest_path.read_text().splitlines() if s.strip()]
    def key(x): return x['user_id'], x['scenario_id'], x['label_rotation'], x['row_rotation']
    lookup = {key(x): x for x in manifest}
    assert len(lookup) == len(raw) == 256
    rows = []
    for row in raw:
        action, method = parse_completion(row['completion'], row['arm'])
        source = lookup[key(row)]
        assert action is not None
        metric = source['metrics_by_action'][ord(action) - 65]
        rows.append({'user_id': row['user_id'], 'scenario_id': row['scenario_id'],
                     'label_rotation': row['label_rotation'], 'row_rotation': row['row_rotation'],
                     'gold_row_position': row['gold_row_position'], 'gold_action': row['gold_action'],
                     'parsed_action': action, 'parse_method': method, 'raw_completion': row['completion'],
                     'correct': action == row['gold_action'], 'metrics': metric})
    cases = defaultdict(dict)
    for row in rows:
        cases[(row['user_id'], row['scenario_id'])][(row['label_rotation'], row['row_rotation'])] = row
    assert len(cases) == 16 and all(len(g) == 16 for g in cases.values())
    rng = np.random.default_rng(seed)
    case_accuracy_delta = np.array([np.mean([g[(l, 0)]['correct'] for l in range(4)]) - np.mean([g[(l, p)]['correct'] for l in range(4) for p in (1, 2, 3)]) for g in cases.values()])
    case_regret_delta = np.array([np.mean([g[(l, 0)]['metrics']['normalized_regret'] for l in range(4)]) - np.mean([g[(l, p)]['metrics']['normalized_regret'] for l in range(4) for p in (1, 2, 3)]) for g in cases.values()])
    idx = rng.integers(0, len(cases), size=(draws, len(cases)))
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'raw_source': str(raw_path),
              'manifest': str(manifest_path), 'records': len(rows), 'cases': len(cases),
              'original_parse_failures': sum(x['parse_failure'] for x in raw),
              'parse_methods': dict(Counter(x['parse_method'] for x in rows)),
              'choice_counts': dict(Counter(x['parsed_action'] for x in rows)),
              'correct': sum(x['correct'] for x in rows),
              'correct_by_gold': {g: sum(x['correct'] for x in rows if x['gold_action'] == g) for g in 'ABCD'},
              'correct_by_gold_row_position': {str(p): sum(x['correct'] for x in rows if x['gold_row_position'] == p) for p in range(4)},
              'correct_by_row_rotation': {str(p): sum(x['correct'] for x in rows if x['row_rotation'] == p) for p in range(4)},
              'mean_regret_by_row_rotation': {str(p): float(np.mean([x['metrics']['normalized_regret'] for x in rows if x['row_rotation'] == p])) for p in range(4)},
              'correct_by_gold_and_position': {g: {str(p): sum(x['correct'] for x in rows if x['gold_action'] == g and x['gold_row_position'] == p) for p in range(4)} for g in 'ABCD'},
              'fixed_order_minus_rotated': {'accuracy_difference': float(case_accuracy_delta.mean()),
                 'accuracy_95ci': [float(x) for x in np.quantile(case_accuracy_delta[idx].mean(axis=1), [.025, .975])],
                 'regret_difference': float(case_regret_delta.mean()),
                 'regret_95ci': [float(x) for x in np.quantile(case_regret_delta[idx].mean(axis=1), [.025, .975])]}}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'reparsed.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    (output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.raw, args.manifest, args.output_dir)
