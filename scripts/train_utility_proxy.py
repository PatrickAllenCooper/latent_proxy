"""CPU pilot comparing natural imitation with full-information utility learning."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.train_action_balanced_proxy import audit
from scripts.train_finite_menu_ppo import sample_batch
from src.training.finite_menu_ppo import FiniteMenuPolicy
from src.training.synthetic_users import SyntheticUserSampler

import numpy as np


def run(checkpoint: Path, output_dir: Path, objective: str, seed: int,
        updates: int, batch_size: int, learning_rate: float, dev_seed: int) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sampler = SyntheticUserSampler(seed=seed + 10_000)
    policy = FiniteMenuPolicy()
    initial = torch.load(checkpoint, map_location='cpu', weights_only=True)
    policy.load_state_dict(initial)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_checkpoint': str(checkpoint),
              'source_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'objective': objective, 'seed': seed, 'updates': updates,
              'batch_size': batch_size, 'learning_rate': learning_rate,
              'dev_seed': dev_seed, 'baseline_development': audit(policy, dev_seed, 50, 20),
              'history': []}
    for step in range(1, updates + 1):
        t0 = time.monotonic()
        states, utilities = sample_batch(rng, sampler, batch_size)
        t1 = time.monotonic()
        logits, _ = policy(states)
        if objective == 'expected_utility':
            loss = -(F.softmax(logits, dim=-1) * utilities).sum(dim=-1).mean()
        elif objective == 'natural_imitation':
            loss = F.cross_entropy(logits, utilities.argmax(dim=-1))
        else:
            raise ValueError(objective)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t2 = time.monotonic()
        if step in {1, updates} or step % 20 == 0:
            report['history'].append({'step': step, 'loss': float(loss.item()),
                'cpu_preparation_seconds': t1 - t0, 'cpu_update_seconds': t2 - t1,
                'development': audit(policy, dev_seed, 50, 20)})
    report['max_parameter_change'] = max(float((policy.state_dict()[k] - initial[k]).abs().max())
                                         for k in initial)
    assert report['max_parameter_change'] > 0
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output_dir / 'policy.pt')
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'objective': objective,
        'baseline': report['baseline_development'],
        'final': report['history'][-1]['development'],
        'max_parameter_change': report['max_parameter_change']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--objective', choices=('expected_utility', 'natural_imitation'), required=True)
    p.add_argument('--seed', type=int, default=17201)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--dev-seed', type=int, default=9301)
    a = p.parse_args()
    run(a.checkpoint, a.output_dir, a.objective, a.seed, a.updates,
        a.batch_size, a.learning_rate, a.dev_seed)
