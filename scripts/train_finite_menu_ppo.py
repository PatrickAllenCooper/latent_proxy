"""Train and audit a preference-conditioned proxy policy using PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from src.evaluation.preference_benchmark import (
    BENCHMARK_VERSION,
    expected_utilities,
    make_scenario,
)
from src.training.finite_menu_ppo import FiniteMenuPolicy, encode_menu
from src.training.synthetic_users import SyntheticUserSampler


def sample_batch(
    rng: np.random.Generator,
    sampler: SyntheticUserSampler,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    rewards = []
    for _ in range(size):
        scenario = make_scenario(int(rng.integers(0, 2**31)))
        theta = sampler.sample()
        utility = expected_utilities(scenario, theta)
        span = max(float(utility.max() - utility.min()), 1e-6)
        features.append(encode_menu(scenario, theta))
        rewards.append(((utility - utility.mean()) / span).astype(np.float32))
    return torch.from_numpy(np.stack(features)), torch.from_numpy(np.stack(rewards))


def audit(policy: FiniteMenuPolicy, seed: int, n_users: int, n_scenarios: int) -> dict[str, float]:
    sampler = SyntheticUserSampler(seed=seed)
    scenarios = [make_scenario(seed + 1_000_000 + i) for i in range(n_scenarios)]
    regrets = []
    matches = []
    with torch.no_grad():
        for theta in sampler.sample_batch(n_users):
            for scenario in scenarios:
                values = expected_utilities(scenario, theta)
                choice = policy.act(scenario, theta)
                span = max(float(values.max() - values.min()), 1e-10)
                regrets.append(float((values.max() - values[choice]) / span))
                matches.append(float(choice == int(np.argmax(values))))
    return {
        "mean_normalized_regret": float(np.mean(regrets)),
        "oracle_action_match_rate": float(np.mean(matches)),
        "n_decisions": len(regrets),
    }


def train(
    seed: int, updates: int, batch_size: int, warmstart_updates: int
) -> tuple[FiniteMenuPolicy, list[dict], dict[str, float]]:
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sampler = SyntheticUserSampler(seed=seed + 10_000)
    policy = FiniteMenuPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    history = []
    for _ in range(warmstart_updates):
        states, all_rewards = sample_batch(rng, sampler, batch_size)
        logits, _ = policy(states)
        loss = torch.nn.functional.cross_entropy(logits, all_rewards.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    warmstart_audit = audit(policy, seed=3001, n_users=50, n_scenarios=20)
    for group in optimizer.param_groups:
        group["lr"] = 1e-4
    for update in range(updates):
        states, all_rewards = sample_batch(rng, sampler, batch_size)
        with torch.no_grad():
            old_logits, old_values = policy(states)
            actions = Categorical(logits=old_logits).sample()
            old_logp = Categorical(logits=old_logits).log_prob(actions)
            rewards = all_rewards.gather(1, actions[:, None]).squeeze(1)
            advantage = rewards - old_values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        for _ in range(4):
            logits, values = policy(states)
            distribution = Categorical(logits=logits)
            logp = distribution.log_prob(actions)
            ratio = (logp - old_logp).exp()
            surrogate = torch.minimum(
                ratio * advantage,
                torch.clamp(ratio, 0.8, 1.2) * advantage,
            )
            loss = -surrogate.mean() + 0.5 * torch.square(values - rewards).mean()
            loss -= 0.01 * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        if update == 0 or (update + 1) % 20 == 0 or update + 1 == updates:
            history.append({"update": update + 1, "loss": float(loss.item())})
    return policy, history, warmstart_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmstart-updates", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    policy, history, warmstart_audit = train(
        args.seed, args.updates, args.batch_size, args.warmstart_updates
    )
    pilot = audit(policy, seed=1001, n_users=50, n_scenarios=20)
    held_out = audit(policy, seed=3001, n_users=50, n_scenarios=20)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), args.output_dir / "policy.pt")
    report = {
        "benchmark_version": BENCHMARK_VERSION,
        "algorithm": "one-step clipped PPO actor-critic",
        "training_seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "sampled_training_decisions": args.updates * args.batch_size,
        "warmstart_updates": args.warmstart_updates,
        "warmstart_audit": warmstart_audit,
        "pilot": pilot,
        "held_out": held_out,
        "history": history,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pilot": pilot, "held_out": held_out}, indent=2))


if __name__ == "__main__":
    main()
