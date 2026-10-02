"""Prepare paired direct, optional-tool, and forced-tool decision prompts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.prepare_numeric_prompt_search import prepare_split

ARMS = ('direct', 'optional_tool', 'forced_tool')


def prepare(output: Path, pool: Path, source: Path, users_per_gold: int, seed: int) -> None:
    prepare_split(pool, source, users_per_gold, seed)
    base = [json.loads(s) for s in pool.read_text().splitlines() if s.strip()]
    base = [row for row in base if row['arm'] == 'scores_only']
    rows = []
    for old in base:
        values = old['expected_utilities']
        scores = ', '.join(f'{chr(65 + i)}={values[i]:+.6f}' for i in range(4))
        direct = f'User preference proxy scores are signed decimals: {scores}. Choose the action with the highest expected utility. Reply with exactly one capital letter: A, B, C, or D.'
        optional = (f'User preference proxy scores are signed decimals: {scores}. You may request an exact calculation by calling preference_proxy. '
                    'To call the tool, reply with exactly TOOL. To decide yourself, reply with exactly one capital letter: A, B, C, or D. '
                    'Choose whichever route will best serve the user.')
        forced = (f'User preference proxy scores are signed decimals: {scores}. '
                  f'preference_proxy returned: recommended action {old["gold_action"]}. '
                  'Follow the tool result to serve the user. Reply with exactly one capital letter: A, B, C, or D.')
        prompts = {'direct': direct, 'optional_tool': optional, 'forced_tool': forced}
        for arm in ARMS:
            rows.append({**old, 'arm': arm, 'prompt': prompts[arm]})
    assert len({(r['user_id'], r['scenario_id'], r['permutation_id'], r['arm']) for r in rows}) == len(rows)
    assert Counter(r['gold_action'] for r in rows if r['arm'] == 'direct') == dict.fromkeys('ABCD', users_per_gold * 4)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        for row in rows: handle.write(json.dumps(row) + '\n')
    print(json.dumps({'event': 'cpu_preparation_complete', 'at_utc': datetime.now(timezone.utc).isoformat(),
                      'cases': len(base) // 4, 'records': len(rows), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--users-per-gold', type=int, default=4)
    parser.add_argument('--seed', type=int, default=13001)
    args = parser.parse_args()
    prepare(args.output, args.pool, args.source, args.users_per_gold, args.seed)
