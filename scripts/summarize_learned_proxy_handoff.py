"""Separate frozen proxy error from LLM handoff error on paired menus."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def summarize(learned_path: Path, exact_path: Path, output_path: Path) -> None:
    learned = [json.loads(s) for s in learned_path.read_text().splitlines() if s.strip()]
    exact = [json.loads(s) for s in exact_path.read_text().splitlines() if s.strip()]
    exact_by_key = {(r['user_id'], r['scenario_id'], r['permutation_id']): r
                    for r in exact if r['arm'] == 'compact_user'}
    assert len(exact_by_key) == 64 and learned
    seen = set()
    paired = []
    for r in learned:
        key = r['user_id'], r['scenario_id'], r['permutation_id']
        assert key not in seen and key in exact_by_key
        seen.add(key)
        oracle = exact_by_key[key]
        assert r['gold_action'] == oracle['gold_action']
        assert r['expected_utilities'] == oracle['expected_utilities']
        assert r['policy_metrics']['action'] == ord(r['learned_advice']) - 65
        if not r['parse_failure']:
            assert r['metrics']['action'] == ord(r['final_action']) - 65
        paired.append((r, oracle))
    rng = np.random.default_rng(26601)
    cases = sorted({(r['user_id'], r['scenario_id']) for r in learned})
    components = {}
    for name, fn in {
        'learned_proxy': lambda r, o: r['policy_metrics']['normalized_regret'],
        'learned_proxy_plus_llm': lambda r, o: r['metrics']['normalized_regret'] if r['metrics'] else 1.0,
        'exact_proxy_plus_llm': lambda r, o: o['metrics']['normalized_regret'] if o['metrics'] else 1.0,
    }.items():
        vals = np.array([np.mean([fn(r, o) for r, o in paired
                                   if (r['user_id'], r['scenario_id']) == case]) for case in cases])
        idx = rng.integers(0, len(cases), size=(10000, len(cases)))
        components[name] = {'mean_normalized_regret': float(vals.mean()),
                            'case_cluster_bootstrap_95ci': [float(x) for x in np.quantile(vals[idx].mean(axis=1), [.025, .975])]}
    delta = np.array([np.mean([(r['metrics']['normalized_regret'] if r['metrics'] else 1.0)
                               - r['policy_metrics']['normalized_regret'] for r, _ in paired
                               if (r['user_id'], r['scenario_id']) == case]) for case in cases])
    idx = rng.integers(0, len(cases), size=(10000, len(cases)))
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'learned_source': str(learned_path), 'exact_source': str(exact_path),
        'records': len(learned), 'cases': len(cases),
        'parse_failures': sum(r['parse_failure'] for r in learned),
        'policy_optimal': sum(r['learned_advice'] == r['gold_action'] for r in learned),
        'llm_followed_policy': sum(r['final_action'] == r['learned_advice'] for r in learned),
        'llm_optimal': sum(r['final_action'] == r['gold_action'] for r in learned),
        'final_action_counts': dict(Counter(r['final_action'] for r in learned)),
        'components': components,
        'llm_minus_policy_regret': {'mean': float(delta.mean()),
            'case_cluster_bootstrap_95ci': [float(x) for x in np.quantile(delta[idx].mean(axis=1), [.025, .975])]},
        'advice_mismatches': [{k: r[k] for k in ('user_id', 'scenario_id', 'permutation_id',
                                               'learned_advice', 'final_action', 'completion')}
                              for r in learned if r['final_action'] != r['learned_advice']]}
    output_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('records', 'parse_failures', 'policy_optimal',
                                           'llm_followed_policy', 'llm_optimal', 'components',
                                           'llm_minus_policy_regret')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--learned', type=Path, required=True)
    p.add_argument('--exact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    summarize(a.learned, a.exact, a.output)
