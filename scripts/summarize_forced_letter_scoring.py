"""Summarize forced-letter logits versus greedy response decisions."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.reparse_numeric_prompt import parse_completion


def summarize(raw_path: Path, manifest_path: Path, output_dir: Path, seed: int = 77419, draws: int = 10000) -> None:
    raw = [json.loads(s) for s in raw_path.read_text().splitlines() if s.strip()]
    manifest = [json.loads(s) for s in manifest_path.read_text().splitlines() if s.strip()]
    def key(x): return x['user_id'], x['scenario_id'], x['permutation_id'], x['arm']
    lookup = {key(x): x for x in manifest}
    assert len(raw) == len(lookup) == 128
    records = []
    for row in raw:
        src = lookup[key(row)]
        greedy, method = parse_completion(row['completion'], row['arm'])
        forced = row['forced_action']
        assert forced in 'ABCD' and forced == max('ABCD', key=lambda x: row['candidate_logits'][x])
        assert greedy is not None
        assert row['forced_metrics']['action'] == ord(forced) - 65
        greedy_metrics = src['metrics_by_action'][ord(greedy) - 65]
        records.append({'user_id': row['user_id'], 'scenario_id': row['scenario_id'],
                        'permutation_id': row['permutation_id'], 'arm': row['arm'],
                        'gold_action': row['gold_action'], 'forced_action': forced,
                        'greedy_action': greedy, 'greedy_parse_method': method,
                        'raw_completion': row['completion'],
                        'candidate_probabilities': row['candidate_probabilities'],
                        'forced_metrics': row['forced_metrics'], 'greedy_metrics': greedy_metrics})
    groups = defaultdict(dict)
    for row in records:
        groups[(row['user_id'], row['scenario_id'])][(row['permutation_id'], row['arm'])] = row
    assert len(groups) == 16 and all(len(g) == 8 for g in groups.values())
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'raw_source': str(raw_path),
              'manifest': str(manifest_path), 'records': len(records), 'cases': len(groups),
              'note': 'Forced scoring restricts the first assistant token to A/B/C/D. Greedy table answers often begin with a phrase; these are distinct decoding conditions.',
              'arms': {}, 'paired': {}}
    rng = np.random.default_rng(seed)
    for arm in ('scores_only', 'vertical_table'):
        subset = [r for r in records if r['arm'] == arm]
        arm_cases = [[g[(p, arm)] for p in range(4)] for g in groups.values()]
        report['arms'][arm] = {}
        for mode in ('forced', 'greedy'):
            case_accuracy = np.array([np.mean([r[f'{mode}_action'] == r['gold_action'] for r in group]) for group in arm_cases])
            case_regret = np.array([np.mean([r[f'{mode}_metrics']['normalized_regret'] for r in group]) for group in arm_cases])
            idx = rng.integers(0, len(groups), size=(draws, len(groups)))
            report['arms'][arm][mode] = {'correct': int(sum(r[f'{mode}_action'] == r['gold_action'] for r in subset)),
                'choice_counts': dict(Counter(r[f'{mode}_action'] for r in subset)),
                'mean_normalized_regret': float(case_regret.mean()),
                'accuracy_95ci': [float(x) for x in np.quantile(case_accuracy[idx].mean(axis=1), [.025, .975])],
                'regret_95ci': [float(x) for x in np.quantile(case_regret[idx].mean(axis=1), [.025, .975])],
                'correct_by_gold': {gold: int(sum(r[f'{mode}_action'] == gold for r in subset if r['gold_action'] == gold)) for gold in 'ABCD'}}
        report['arms'][arm]['greedy_parse_methods'] = dict(Counter(r['greedy_parse_method'] for r in subset))
        delta = np.array([np.mean([r['forced_metrics']['normalized_regret'] - r['greedy_metrics']['normalized_regret'] for r in group]) for group in arm_cases])
        idx = rng.integers(0, len(groups), size=(draws, len(groups)))
        report['paired'][f'{arm}_forced_minus_greedy_regret'] = {'mean': float(delta.mean()),
            'case_cluster_bootstrap_95ci': [float(x) for x in np.quantile(delta[idx].mean(axis=1), [.025, .975])]}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'paired_records.jsonl').open('w') as handle:
        for row in records: handle.write(json.dumps(row) + '\n')
    (output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.raw, args.manifest, args.output_dir)
