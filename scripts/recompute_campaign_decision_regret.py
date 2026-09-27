"""Recompute decision regret with a true-type expected-utility optimizer."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.expected_utility_optimizer import (
    expected_allocation_utility,
    optimize_expected_allocation,
)
from src.evaluation.generalization_protocol import DOMAIN_FACTORIES
from src.training.synthetic_users import UserType


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def recompute(input_dir: Path) -> dict[str, Any]:
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    config = manifest.get("config", {})
    reference_point_mode = config.get("reference_point_mode", "zero")
    utility_form = config.get("utility_form", "absolute")
    paths = sorted(input_dir.glob("*/*/s*_u*.json"))
    if not paths:
        raise FileNotFoundError(f"No per-user campaign records found under {input_dir}")

    for path in paths:
        record = json.loads(path.read_text())
        domain = record["domain"]
        seed = int(record["seed"])
        user_idx = int(record["user_idx"])
        env = DOMAIN_FACTORIES[domain]()
        env.reset(seed=seed + user_idx)
        obs = env._get_obs()
        stats = env.get_channel_stats()
        wealth = float(np.asarray(obs["wealth"]).sum())
        rounds_remaining = int(env.config.n_rounds - obs["round"])
        reference_point = wealth if reference_point_mode == "current_wealth" else 0.0
        theta_record = record["true_theta"]
        theta = UserType(
            gamma=float(theta_record["gamma"]),
            alpha=float(theta_record["alpha"]),
            lambda_=float(theta_record["lambda_"]),
        )
        heuristic = env.get_optimal_action(theta_record)
        optimal_action, optimal_utility = optimize_expected_allocation(
            theta, stats["means"], stats["variances"], wealth,
            rounds_remaining, reference_point=reference_point,
            utility_form=utility_form, initial_actions=[heuristic],
        )
        recommended = np.asarray(
            record["alignment"]["optimal_action_inferred"], dtype=np.float64,
        )
        recommended_utility = expected_allocation_utility(
            recommended, theta, stats["means"], stats["variances"], wealth,
            rounds_remaining, reference_point=reference_point,
            utility_form=utility_form,
        )
        regret = max(0.0, optimal_utility - recommended_utility)
        passes, reasons = env.check_quality_floor(recommended)
        alignment = record["alignment"]
        alignment.update({
            "optimal_action_true": [float(x) for x in optimal_action],
            "decision_regret": float(regret),
            "decision_regret_relative": float(regret / max(abs(optimal_utility), 1e-12)),
            "quality_floor_violation": not passes,
            "quality_floor_reasons": reasons,
            "action_l2_distance": float(np.linalg.norm(optimal_action - recommended)),
        })
        record["alignment"] = alignment
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(path)

    manifest.setdefault("config", {})["decision_regret_metric"] = (
        "true-type expected prospect utility at a multistart SLSQP optimum on "
        "the long-only simplex minus utility of the inferred-type action; both "
        "evaluated with 48-point Gauss-Hermite quadrature"
    )
    manifest["decision_regret_recomputed_with_git_sha"] = _git_sha()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"records_updated": len(paths), "input_dir": str(input_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(recompute(Path(args.input_dir)), indent=2))


if __name__ == "__main__":
    main()
