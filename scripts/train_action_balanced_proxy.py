"""CPU pilot: recover rare optimal actions with balanced imitation updates."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.preference_benchmark import expected_utilities, make_scenario
from src.training.finite_menu_ppo import FiniteMenuPolicy, encode_menu
from src.training.synthetic_users import SyntheticUserSampler


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def balanced_batch(rng: np.random.Generator, sampler: SyntheticUserSampler,
                   per_action: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    selected: dict[int, list[np.ndarray]] = {a: [] for a in range(4)}
    attempts = 0
    while any(len(v) < per_action for v in selected.values()):
        attempts += 1
        scenario = make_scenario(int(rng.integers(0, 2**31)))
        theta = sampler.sample()
        gold = int(np.argmax(expected_utilities(scenario, theta)))
        if len(selected[gold]) < per_action:
            selected[gold].append(encode_menu(scenario, theta))
        if attempts > per_action * 400:
            raise RuntimeError('Could not fill balanced action batch')
    states = np.stack([x for a in range(4) for x in selected[a]])
    labels = np.repeat(np.arange(4), per_action)
    order = rng.permutation(len(labels))
    return torch.from_numpy(states[order]), torch.from_numpy(labels[order]), attempts


def natural_batch(rng: np.random.Generator, sampler: SyntheticUserSampler,
                  size: int) -> tuple[torch.Tensor, torch.Tensor]:
    states = []
    labels = []
    for _ in range(size):
        scenario = make_scenario(int(rng.integers(0, 2**31)))
        theta = sampler.sample()
        states.append(encode_menu(scenario, theta))
        labels.append(int(np.argmax(expected_utilities(scenario, theta))))
    return torch.from_numpy(np.stack(states)), torch.tensor(labels)


def audit(policy: FiniteMenuPolicy, seed: int, n_users: int, n_scenarios: int) -> dict:
    sampler = SyntheticUserSampler(seed=seed)
    scenarios = [make_scenario(seed + 1_000_000 + i) for i in range(n_scenarios)]
    counts = Counter()
    by_gold = defaultdict(list)
    for theta in sampler.sample_batch(n_users):
        for scenario in scenarios:
            scores = expected_utilities(scenario, theta)
            gold = int(np.argmax(scores))
            action = policy.act(scenario, theta)
            regret = float((scores.max() - scores[action]) / max(float(scores.max() - scores.min()), 1e-10))
            counts[action] += 1
            by_gold[gold].append((action, regret))
    all_rows = [item for v in by_gold.values() for item in v]
    return {'seed': seed, 'decisions': len(all_rows),
            'policy_action_counts': {str(a): counts[a] for a in range(4)},
            'mean_normalized_regret': float(np.mean([regret for _, regret in all_rows])),
            'oracle_action_match_rate': float(sum(a == g for g, rows in by_gold.items() for a, _ in rows) / len(all_rows)),
            'by_gold': {str(g): {'decisions': len(rows),
                                'policy_action_counts': {str(a): sum(action == a for action, _ in rows) for a in range(4)},
                                'mean_normalized_regret': float(np.mean([regret for _, regret in rows]))}
                        for g, rows in sorted(by_gold.items())}}


def run(checkpoint: Path, output_dir: Path, seed: int, updates: int,
        per_action: int, lr: float, dev_seed: int, natural_fraction: float) -> None:
    if not 0 <= natural_fraction < 1:
        raise ValueError('natural_fraction must be in [0, 1)')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sampler = SyntheticUserSampler(seed=seed + 10_000)
    policy = FiniteMenuPolicy()
    initial = torch.load(checkpoint, map_location='cpu', weights_only=True)
    policy.load_state_dict(initial)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    report = {'started_at_utc': now(), 'source_checkpoint': str(checkpoint),
              'source_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'seed': seed, 'updates': updates, 'per_action_per_batch': per_action,
              'learning_rate': lr, 'development_seed': dev_seed,
              'natural_fraction': natural_fraction,
              'baseline_development': audit(policy, dev_seed, 50, 20), 'history': []}
    for step in range(1, updates + 1):
        t0 = time.monotonic()
        states, labels, attempts = balanced_batch(rng, sampler, per_action)
        if natural_fraction:
            natural_size = round(len(labels) * natural_fraction / (1 - natural_fraction))
            natural_states, natural_labels = natural_batch(rng, sampler, natural_size)
            states = torch.cat((states, natural_states))
            labels = torch.cat((labels, natural_labels))
            order = torch.randperm(len(labels))
            states, labels = states[order], labels[order]
        t1 = time.monotonic()
        logits, _ = policy(states)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.monotonic()
        if step in {1, updates} or step % 10 == 0:
            report['history'].append({'step': step, 'loss': float(loss.item()),
                                      'sampling_attempts': attempts,
                                      'cpu_preparation_seconds': t1 - t0,
                                      'cpu_update_seconds': t2 - t1,
                                      'development': audit(policy, dev_seed, 50, 20)})
    report['max_parameter_change'] = max(float((policy.state_dict()[k] - initial[k]).abs().max())
                                         for k in initial)
    assert report['max_parameter_change'] > 0
    report['completed_at_utc'] = now()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output_dir / 'policy.pt')
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'baseline': report['baseline_development'],
                      'final': report['history'][-1]['development'],
                      'max_parameter_change': report['max_parameter_change']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=17101)
    p.add_argument('--updates', type=int, default=20)
    p.add_argument('--per-action', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dev-seed', type=int, default=9301)
    p.add_argument('--natural-fraction', type=float, default=0.0)
    a = p.parse_args()
    run(a.checkpoint, a.output_dir, a.seed, a.updates, a.per_action, a.lr,
        a.dev_seed, a.natural_fraction)
