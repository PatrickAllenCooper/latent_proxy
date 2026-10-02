"""Compare two frozen proxy checkpoints on the same labeled menus."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def compare(reference: Path, candidate: Path, output: Path) -> None:
    old = [json.loads(s) for s in reference.read_text().splitlines() if s.strip()]
    new = [json.loads(s) for s in candidate.read_text().splitlines() if s.strip()]
    assert len(old) == len(new) == 64
    by_case = defaultdict(list)
    changes = []
    for x, y in zip(old, new):
        key = x['user_id'], x['scenario_id'], x['permutation_id']
        assert key == (y['user_id'], y['scenario_id'], y['permutation_id'])
        assert x['gold_action'] == y['gold_action'] and x['expected_utilities'] == y['expected_utilities']
        old_regret = x['metrics_by_action'][ord(x['learned_advice']) - 65]['normalized_regret']
        new_regret = y['metrics_by_action'][ord(y['learned_advice']) - 65]['normalized_regret']
        by_case[key[:2]].append(new_regret - old_regret)
        if x['learned_advice'] != y['learned_advice']:
            changes.append({'user_id': key[0], 'scenario_id': key[1], 'permutation_id': key[2],
                            'gold_action': x['gold_action'], 'reference_advice': x['learned_advice'],
                            'candidate_advice': y['learned_advice'],
                            'reference_regret': old_regret, 'candidate_regret': new_regret})
    assert len(by_case) == 16 and all(len(v) == 4 for v in by_case.values())
    case_delta = np.array([np.mean(v) for v in by_case.values()])
    rng = np.random.default_rng(26605)
    idx = rng.integers(0, len(case_delta), size=(10000, len(case_delta)))
    def arm_report(rows: list[dict]) -> dict:
        return {'optimal': sum(r['learned_advice'] == r['gold_action'] for r in rows),
                'mean_normalized_regret': float(np.mean([
                    r['metrics_by_action'][ord(r['learned_advice']) - 65]['normalized_regret'] for r in rows])),
                'advice_counts': dict(Counter(r['learned_advice'] for r in rows)),
                'checkpoint_sha256': rows[0]['policy_checkpoint_sha256']}
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'reference_manifest': str(reference), 'candidate_manifest': str(candidate),
              'cases': len(by_case), 'records_per_arm': len(old),
              'reference': arm_report(old), 'candidate': arm_report(new),
              'candidate_minus_reference_regret': {'mean': float(case_delta.mean()),
                'case_cluster_bootstrap_95ci': [float(v) for v in np.quantile(case_delta[idx].mean(axis=1), [.025, .975])]},
              'changed_advice': changes}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('reference', 'candidate', 'candidate_minus_reference_regret')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    compare(a.reference, a.candidate, a.output)
