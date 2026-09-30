"""Paired CPU pilot for the new reward/behavior benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.evaluation.preference_benchmark import (
    BENCHMARK_VERSION,
    behavior_probabilities,
    evaluate_action,
    expected_utilities,
    make_scenario,
)
from src.training.synthetic_users import SyntheticUserSampler, UserType


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    arms = sorted({str(row["arm"]) for row in records})
    by_user: dict[str, dict[int, list[dict[str, object]]]] = {}
    for arm in arms:
        by_user[arm] = {}
    for row in records:
        by_user[str(row["arm"])].setdefault(int(row["user_id"]), []).append(row)
    rng = np.random.default_rng(91_337)
    result: dict[str, object] = {}
    for arm in arms:
        users = sorted(by_user[arm])
        fields = ("normalized_regret", "behavioral_agreement", "feasible")
        summary: dict[str, object] = {"users": len(users), "decisions": len(records) // len(arms)}
        for field in fields:
            values = np.array([
                np.mean([float(row[field]) for row in by_user[arm][user]])
                for user in users
            ])
            draws = rng.integers(0, len(values), size=(2000, len(values)))
            means = values[draws].mean(axis=1)
            summary[field] = {
                "mean": float(values.mean()),
                "user_bootstrap_95": [float(x) for x in np.quantile(means, [.025, .975])],
            }
        result[arm] = summary
    comparisons = (
        ("gold_reward_tool", "generic"),
        ("gold_behavior_tool", "gold_reward_tool"),
        ("swapped_reward_tool", "generic"),
        ("learned_rl_proxy", "generic"),
    )
    paired: dict[str, object] = {}
    for first, second in comparisons:
        if first not in by_user or second not in by_user:
            continue
        users = sorted(set(by_user[first]) & set(by_user[second]))
        contrasts: dict[str, object] = {}
        for field in ("normalized_regret", "behavioral_agreement"):
            differences = np.array([
                np.mean([float(row[field]) for row in by_user[first][user]])
                - np.mean([float(row[field]) for row in by_user[second][user]])
                for user in users
            ])
            draws = rng.integers(0, len(users), size=(2000, len(users)))
            contrasts[field] = {
                "mean_difference": float(differences.mean()),
                "user_bootstrap_95": [
                    float(x) for x in np.quantile(differences[draws].mean(axis=1), [.025, .975])
                ],
            }
        paired[f"{first}_minus_{second}"] = contrasts
    result["paired_contrasts"] = paired
    return result


def run(
    n_users: int,
    n_scenarios: int,
    seed: int,
    policy_checkpoint: Path | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if n_users < 2 or n_scenarios < 1:
        raise ValueError("need at least two users and one scenario")
    users = SyntheticUserSampler(seed=seed).sample_batch(n_users)
    scenarios = [make_scenario(seed + 1_000_000 + i) for i in range(n_scenarios)]
    rng = np.random.default_rng(seed + 2_000_000)
    generic = UserType(gamma=.6, alpha=1.0, lambda_=1.5)
    policy = None
    if policy_checkpoint is not None:
        import torch

        from src.training.finite_menu_ppo import FiniteMenuPolicy

        policy = FiniteMenuPolicy()
        policy.load_state_dict(torch.load(policy_checkpoint, map_location="cpu", weights_only=True))
        policy.eval()
    records: list[dict[str, object]] = []
    for user_id, theta in enumerate(users):
        # Stable user-specific preference for the safe option is behavioral,
        # not part of the user's ground-truth outcome utility.
        safe_bias = float(rng.uniform(-.06, .06))
        bias = rng.normal(0.0, .018, size=4)
        bias[0] += safe_bias
        swapped_theta = users[(user_id + 1) % n_users]
        for scenario in scenarios:
            utility = expected_utilities(scenario, theta)
            behavior = behavior_probabilities(scenario, theta, bias=bias)
            decisions = {
                "generic": int(np.argmax(expected_utilities(scenario, generic))),
                "gold_reward_tool": int(np.argmax(utility)),
                "gold_behavior_tool": int(np.argmax(behavior)),
                "swapped_reward_tool": int(np.argmax(
                    expected_utilities(scenario, swapped_theta)
                )),
            }
            if policy is not None:
                decisions["learned_rl_proxy"] = policy.act(scenario, theta)
            for arm, action in decisions.items():
                metrics = evaluate_action(
                    scenario, theta, action, behavior_bias=bias
                )
                records.append({
                    "benchmark_version": BENCHMARK_VERSION,
                    "seed": seed,
                    "user_id": user_id,
                    "scenario_id": scenario.scenario_id,
                    "arm": arm,
                    "true_gamma": theta.gamma,
                    "true_alpha": theta.alpha,
                    "true_lambda": theta.lambda_,
                    "safe_behavior_bias": safe_bias,
                    **metrics,
                })
    diagnostics: dict[str, object] = {}
    extremes = SyntheticUserSampler().sample_extreme_types()
    for name, theta in extremes.items():
        choices = [int(np.argmax(expected_utilities(scenario, theta))) for scenario in scenarios]
        diagnostics[name] = {
            "action_counts": np.bincount(choices, minlength=4).tolist(),
        }
    return records, {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_users": n_users,
        "n_scenarios": n_scenarios,
        "policy_checkpoint": str(policy_checkpoint) if policy_checkpoint else None,
        "summary": summarize(records),
        "extreme_type_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--n-scenarios", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path)
    args = parser.parse_args()
    records, summary = run(
        args.n_users, args.n_scenarios, args.seed, args.policy_checkpoint
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_decision.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["summary"], indent=2))


if __name__ == "__main__":
    main()
