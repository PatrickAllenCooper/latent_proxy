"""Prepare label and table-row rotations to isolate output letter vs position bias."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.prepare_numeric_comparison_pilot import prepare as prepare_balanced


def prepare(output: Path, source: Path, users_per_gold: int, seed: int) -> None:
    prepare_balanced(source, users_per_gold, seed)
    originals = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    originals = [row for row in originals if row['arm'] == 'baseline']
    rows = []
    for old in originals:
        for label_rotation in range(4):
            values = [old['expected_utilities'][(a - label_rotation) % 4] for a in range(4)]
            metrics = [old['metrics_by_action'][(a - label_rotation) % 4].copy() for a in range(4)]
            for a, metric in enumerate(metrics):
                metric['action'] = a
            for row_rotation in range(4):
                order = [(i + row_rotation) % 4 for i in range(4)]
                table = '\n'.join(f'{chr(65 + a)} | {values[a]:+.6f}' for a in order)
                prompt = ('Four signed decimal utilities are shown below. The best action is the row with the numerically largest utility.\n'
                          f'Action | Utility\n{table}\n'
                          'Reply with exactly the best action letter: A, B, C, or D.')
                rows.append({'benchmark_version': old['benchmark_version'], 'seed': seed,
                             'user_id': old['user_id'], 'scenario_id': old['scenario_id'],
                             'label_rotation': label_rotation, 'row_rotation': row_rotation,
                             'arm': 'vertical_table', 'true_theta': old['true_theta'],
                             'expected_utilities': values, 'gold_action': chr(65 + values.index(max(values))),
                             'wrong_action': chr(65 + values.index(min(values))),
                             'gold_row_position': order.index(values.index(max(values))),
                             'metrics_by_action': metrics, 'prompt': prompt})
    assert len({(r['user_id'], r['scenario_id'], r['label_rotation'], r['row_rotation']) for r in rows}) == len(rows)
    assert Counter(r['gold_action'] for r in rows) == dict.fromkeys('ABCD', 4 * len(originals))
    assert Counter(r['gold_row_position'] for r in rows) == dict.fromkeys(range(4), 4 * len(originals))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    print(json.dumps({'event': 'cpu_preparation_complete', 'at_utc': datetime.now(timezone.utc).isoformat(),
                      'records': len(rows), 'cases': len(originals), 'path': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--users-per-gold', type=int, default=4)
    parser.add_argument('--seed', type=int, default=11001)
    args = parser.parse_args()
    prepare(args.output, args.source, args.users_per_gold, args.seed)
