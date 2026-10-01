"""Prepare paired prompt-only and numeric-proxy comparison menus on CPU."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.run_finite_menu_llm_pilot import format_prompt
from src.evaluation.preference_benchmark import (
    BENCHMARK_VERSION, evaluate_action, expected_utilities, make_scenario,
)
from src.training.synthetic_users import SyntheticUserSampler


ARMS = ("baseline", "explicit_formula", "score_comparison")


def prepare(output: Path, n_users: int, n_scenarios: int, seed: int) -> None:
    users = SyntheticUserSampler(seed=seed).sample_batch(n_users)
    scenarios = [make_scenario(seed + 1_000_000 + i) for i in range(n_scenarios)]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for user_id, theta in enumerate(users):
            for scenario in scenarios:
                values = expected_utilities(scenario, theta)
                gold = int(np.argmax(values))
                wrong = int(np.argmin(values))
                base = format_prompt(scenario, theta, None)
                head, instruction = base.rsplit("\n", 1)
                scores = ", ".join(
                    f"{chr(65 + a)}={values[a]:+.6f}" for a in range(4)
                )
                variants = {
                    "baseline": base,
                    "explicit_formula": (
                        "Select the action with the greatest expected user utility. "
                        "For each outcome, transform a positive payoff x to x^alpha and "
                        "a negative payoff x to -lambda*(-x)^alpha. Discount each "
                        "period t by gamma^t, sum periods, weight outcomes by their "
                        "probabilities, and compare all four actions. Think through "
                        f"the calculation before giving one letter.\n{head}\n{instruction}"
                    ),
                    "score_comparison": (
                        f"{head}\nPreference proxy supplied these expected user "
                        f"utilities: {scores}. These are signed decimal numbers. "
                        "Compare their numeric values and choose the action with the "
                        f"largest number.\n{instruction}"
                    ),
                }
                metrics = [evaluate_action(scenario, theta, a) for a in range(4)]
                for arm in ARMS:
                    row = {
                        "benchmark_version": BENCHMARK_VERSION, "seed": seed,
                        "user_id": user_id, "scenario_id": scenario.scenario_id,
                        "arm": arm, "true_theta": theta.__dict__,
                        "expected_utilities": [float(v) for v in values],
                        "gold_action": chr(65 + gold),
                        "wrong_action": chr(65 + wrong),
                        "metrics_by_action": metrics, "prompt": variants[arm],
                    }
                    handle.write(json.dumps(row) + "\n")
    print(json.dumps({"event": "cpu_preparation_complete",
                      "at_utc": datetime.now(timezone.utc).isoformat(),
                      "path": str(output),
                      "records": n_users * n_scenarios * len(ARMS)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-users", type=int, default=4)
    parser.add_argument("--n-scenarios", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6001)
    args = parser.parse_args()
    prepare(args.output, args.n_users, args.n_scenarios, args.seed)


if __name__ == "__main__":
    main()
