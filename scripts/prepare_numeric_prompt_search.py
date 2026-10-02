"""CPU-prepare label-rotated prompt development and held-out pools."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.prepare_numeric_comparison_pilot import prepare as prepare_balanced

CANDIDATES = ('scores_only', 'vertical_table', 'pairwise', 'code_argmax')


def prompt_for(arm: str, values: list[float]) -> str:
    scores = ', '.join(f'{chr(65 + a)}={values[a]:+.6f}' for a in range(4))
    if arm == 'scores_only':
        return f'Preference proxy scores for four actions are signed decimals: {scores}. Choose the letter with the largest numeric score. Reply with exactly one capital letter: A, B, C, or D.'
    if arm == 'vertical_table':
        table = '\n'.join(f'{chr(65 + a)} | {values[a]:+.6f}' for a in range(4))
        return f'Four signed decimal utilities are shown below. The best action is the row with the numerically largest utility.\nAction | Utility\n{table}\nReply with exactly the best action letter: A, B, C, or D.'
    if arm == 'pairwise':
        return f'Signed decimal utilities: {scores}. Compare A against B and retain the larger numeric value. Compare that winner against C, then against D. Return the final winner as exactly one capital letter: A, B, C, or D.'
    if arm == 'code_argmax':
        mapping = ', '.join(f"'{chr(65 + a)}': {values[a]:+.6f}" for a in range(4))
        return f'For the signed decimal utility dictionary {{{mapping}}}, return the key selected by max(scores, key=scores.get). Compare numeric values, not letter order. Reply with exactly one capital letter: A, B, C, or D.'
    raise ValueError(arm)


def prepare_split(output: Path, source: Path, users_per_gold: int, seed: int) -> None:
    prepare_balanced(source, users_per_gold, seed)
    base = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    base = [x for x in base if x['arm'] == 'baseline']
    rows = []
    for old in base:
        for shift in range(4):
            values = [old['expected_utilities'][(a - shift) % 4] for a in range(4)]
            gold = chr(65 + values.index(max(values)))
            metrics = [old['metrics_by_action'][(a - shift) % 4].copy() for a in range(4)]
            for a, metric in enumerate(metrics):
                metric['action'] = a
            for arm in CANDIDATES:
                rows.append({'benchmark_version': old['benchmark_version'], 'seed': seed,
                             'user_id': old['user_id'], 'scenario_id': old['scenario_id'],
                             'permutation_id': shift, 'arm': arm,
                             'true_theta': old['true_theta'], 'expected_utilities': values,
                             'gold_action': gold, 'wrong_action': chr(65 + values.index(min(values))),
                             'metrics_by_action': metrics, 'prompt': prompt_for(arm, values)})
    assert len({(r['user_id'], r['scenario_id'], r['permutation_id'], r['arm']) for r in rows}) == len(rows)
    assert Counter(r['gold_action'] for r in rows if r['arm'] == 'scores_only') == dict.fromkeys('ABCD', 4 * users_per_gold)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    print(json.dumps({'event': 'cpu_preparation_complete', 'at_utc': datetime.now(timezone.utc).isoformat(),
                      'seed': seed, 'cases': len(base), 'records': len(rows), 'path': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--users-per-gold', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    prepare_split(args.output, args.source, args.users_per_gold, args.seed)
