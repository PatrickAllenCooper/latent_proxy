"""Diagnostic strict/fixed-phrase parsing without changing raw model results."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LETTER = re.compile(r'[ABCD]')
TABLE_PHRASE = re.compile(r'The best action is:\s*([ABCD])\.?', re.IGNORECASE)


def parse_completion(text: str, arm: str) -> tuple[str | None, str]:
    stripped = text.strip()
    if LETTER.fullmatch(stripped):
        return stripped, 'single_letter'
    if arm == 'vertical_table':
        match = TABLE_PHRASE.fullmatch(stripped)
        if match:
            return match.group(1).upper(), 'table_phrase'
    return None, 'unparsed'


def summarize(raw_path: Path, manifest_path: Path, output_dir: Path, seed: int = 55317, draws: int = 10000) -> None:
    raw = [json.loads(s) for s in raw_path.read_text().splitlines() if s.strip()]
    manifest = [json.loads(s) for s in manifest_path.read_text().splitlines() if s.strip()]
    def key(row):
        return row['user_id'], row['scenario_id'], row['permutation_id'], row['arm']
    lookup = {key(row): row for row in manifest}
    assert len(lookup) == len(manifest) == len(raw)
    rows = []
    for row in raw:
        source = lookup[key(row)]
        action, method = parse_completion(row['completion'], row['arm'])
        if row['parse_failure'] is False:
            assert method == 'single_letter' and action == row['completion']
        metrics = None if action is None else source['metrics_by_action'][ord(action) - 65]
        rows.append({'user_id': row['user_id'], 'scenario_id': row['scenario_id'],
                     'permutation_id': row['permutation_id'], 'arm': row['arm'],
                     'raw_completion': row['completion'], 'original_parse_failure': row['parse_failure'],
                     'parsed_action': action, 'parse_method': method, 'gold_action': row['gold_action'],
                     'metrics': metrics})
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'reparsed.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    arms = sorted({row['arm'] for row in rows})
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'raw_source': str(raw_path),
              'manifest': str(manifest_path), 'records': len(rows), 'arms': {},
              'parser': 'exact single capital letter for all arms; additionally exact fixed table phrase for vertical_table only'}
    for arm in arms:
        subset = [r for r in rows if r['arm'] == arm]
        valid = [r for r in subset if r['parsed_action'] is not None]
        report['arms'][arm] = {'records': len(subset), 'parsed': len(valid),
                               'parse_methods': dict(Counter(r['parse_method'] for r in subset)),
                               'choice_counts': dict(Counter(r['parsed_action'] for r in valid)),
                               'correct': sum(r['parsed_action'] == r['gold_action'] for r in valid),
                               'mean_normalized_regret_valid': float(np.mean([r['metrics']['normalized_regret'] for r in valid])) if valid else None}
    if {'scores_only', 'vertical_table'}.issubset(arms):
        groups = defaultdict(dict)
        for row in rows:
            if row['arm'] in ('scores_only', 'vertical_table'):
                groups[(row['user_id'], row['scenario_id'])][(row['permutation_id'], row['arm'])] = row
        if all(len(group) == 8 and all(r['parsed_action'] is not None for r in group.values()) for group in groups.values()):
            delta = np.array([np.mean([group[(p, 'vertical_table')]['metrics']['normalized_regret'] - group[(p, 'scores_only')]['metrics']['normalized_regret'] for p in range(4)]) for group in groups.values()])
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(delta), size=(draws, len(delta)))
            report['paired_table_minus_scores_only'] = {'mean_normalized_regret_difference': float(delta.mean()),
                'case_cluster_bootstrap_95ci': [float(x) for x in np.quantile(delta[idx].mean(axis=1), [.025, .975])],
                'cases': len(delta)}
    (output_dir / 'reparse_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.raw, args.manifest, args.output_dir)
