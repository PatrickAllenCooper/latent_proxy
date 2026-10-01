"""Rotate action labels while holding each user's underlying menu utilities fixed."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ARMS = ('scores_only', 'exact_advice')


def prepare(source: Path, output: Path) -> None:
    base = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    base = [row for row in base if row['arm'] == 'baseline']
    assert len(base) == 32 and len({(r['user_id'], r['scenario_id']) for r in base}) == 32
    rows = []
    for old in base:
        values = old['expected_utilities']
        for shift in range(4):
            new_values = [values[(new - shift) % 4] for new in range(4)]
            gold = chr(65 + new_values.index(max(new_values)))
            wrong = chr(65 + new_values.index(min(new_values)))
            new_metrics = [old['metrics_by_action'][(new - shift) % 4].copy() for new in range(4)]
            for new, metric in enumerate(new_metrics):
                metric['action'] = new
            scores = ', '.join(f'{chr(65 + a)}={new_values[a]:+.6f}' for a in range(4))
            prompts = {
                'scores_only': f'Preference proxy scores for four actions are signed decimals: {scores}. Choose the letter with the largest numeric score. Reply with exactly one capital letter: A, B, C, or D.',
                'exact_advice': f'Preference proxy recommendation: {gold}. Follow this recommendation. Reply with exactly one capital letter: A, B, C, or D.',
            }
            for arm in ARMS:
                rows.append({'benchmark_version': old['benchmark_version'], 'seed': old['seed'],
                             'user_id': old['user_id'], 'scenario_id': old['scenario_id'],
                             'permutation_id': shift, 'arm': arm,
                             'true_theta': old['true_theta'], 'expected_utilities': new_values,
                             'gold_action': gold, 'wrong_action': wrong,
                             'metrics_by_action': new_metrics, 'prompt': prompts[arm]})
    assert Counter(r['gold_action'] for r in rows if r['arm'] == 'scores_only') == dict.fromkeys('ABCD', 32)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    print(json.dumps({'event': 'cpu_preparation_complete', 'at_utc': datetime.now(timezone.utc).isoformat(),
                      'records': len(rows), 'cases': len(base), 'source': str(source), 'path': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
