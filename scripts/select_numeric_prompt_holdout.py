"""Freeze baseline versus table candidate from the sealed held-out pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = {'scores_only', 'vertical_table'}


def select(source: Path, output: Path) -> None:
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    selected = [row for row in rows if row['arm'] in ARMS]
    cases = len({(r['user_id'], r['scenario_id']) for r in selected})
    assert len(selected) == 8 * cases
    assert len({(r['user_id'], r['scenario_id'], r['permutation_id'], r['arm']) for r in selected}) == len(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        for row in selected:
            handle.write(json.dumps(row) + '\n')
    print(json.dumps({'records': len(selected), 'source': str(source), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    select(args.source, args.output)
