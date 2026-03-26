from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.game_variants import create_variant_a, create_variant_b
from src.evaluation.elicitation_metrics import run_elicitation_benchmark
from src.evaluation.experiment_runner import (
    _alignment_inferred_vs_true,
    blend_inferred_theta,
    prior_mean_theta,
)


@dataclass
class AblationResults:
    """Maps swept parameter values to metric dicts."""

    param_name: str
    values: list[Any]
    metrics_by_value: dict[str, dict[str, float]]
    wall_clock_s: dict[str, float] = field(default_factory=dict)


def _sync_convergence(ec: ElicitationConfig) -> ElicitationConfig:
    c = ec.convergence
    new_c = ConvergenceConfig(
        gamma_variance_threshold=c.gamma_variance_threshold,
        alpha_variance_threshold=c.alpha_variance_threshold,
        lambda_variance_threshold=c.lambda_variance_threshold,
        robust_action_level=c.robust_action_level,
        max_rounds=ec.max_rounds,
    )
    return ElicitationConfig(
        posterior_type=ec.posterior_type,
        n_particles=ec.n_particles,
        max_rounds=ec.max_rounds,
        n_scenarios_per_round=ec.n_scenarios_per_round,
        n_eig_samples=ec.n_eig_samples,
        temperature=ec.temperature,
        convergence=new_c,
        seed=ec.seed,
    )


def run_sweep(
    param_name: str,
    values: list[Any],
    base_elicitation: ElicitationConfig,
    *,
    n_users: int = 12,
    seed: int = 42,
    variant: str = "a",
) -> AblationResults:
    """Run elicitation benchmark for each parameter value (hold others fixed).

    Supported ``param_name``:
    - ``max_rounds`` / ``query_budget``: query budget sweep
    - ``posterior_type``: ``gaussian`` vs ``particle``
    - ``n_eig_samples``: EIG Monte Carlo sample count
    - ``beta``: post-hoc blend between prior mean and inferred theta (proxy for
      curriculum alignment weight); one benchmark run, then sweep blend values
    """
    metrics: dict[str, dict[str, float]] = {}
    clocks: dict[str, float] = {}

    env_factory = (
        create_variant_b if variant.lower() in ("b", "variant_b") else create_variant_a
    )

    if param_name == "beta":
        ec = _sync_convergence(base_elicitation)
        t0 = time.perf_counter()
        bench = run_elicitation_benchmark(
            env_factory(), n_users=n_users, config=ec, seed=seed,
        )
        clocks["_benchmark"] = time.perf_counter() - t0
        active = bench["active_results"]
        env = env_factory()
        prior = prior_mean_theta()
        for b in values:
            aligns: list[float] = []
            for i, r in enumerate(active):
                if r.true_theta is None:
                    continue
                blended = blend_inferred_theta(r.inferred_theta, float(b), prior)
                aligns.append(
                    _alignment_inferred_vs_true(
                        env, blended, r.true_theta, seed=seed + i + 11,
                    )
                )
            eff = bench["efficiency"]
            metrics[str(b)] = {
                "mean_alignment_blended": float(np.mean(aligns)) if aligns else 0.0,
                "mean_active_error": eff["mean_active_error"],
                "error_reduction_pct": eff["error_reduction_pct"],
            }
            clocks[str(b)] = 0.0
        return AblationResults("beta", list(values), metrics, clocks)

    for v in values:
        ec = ElicitationConfig(
            posterior_type=base_elicitation.posterior_type,
            n_particles=base_elicitation.n_particles,
            max_rounds=base_elicitation.max_rounds,
            n_scenarios_per_round=base_elicitation.n_scenarios_per_round,
            n_eig_samples=base_elicitation.n_eig_samples,
            temperature=base_elicitation.temperature,
            convergence=base_elicitation.convergence,
            seed=base_elicitation.seed,
        )
        if param_name in ("max_rounds", "query_budget"):
            ec.max_rounds = int(v)
            ec = _sync_convergence(ec)
        elif param_name == "posterior_type":
            ec.posterior_type = str(v)
        elif param_name == "n_eig_samples":
            ec.n_eig_samples = int(v)
        else:
            raise ValueError(f"Unknown ablation param_name: {param_name}")

        ec = _sync_convergence(ec)
        t0 = time.perf_counter()
        bench = run_elicitation_benchmark(
            env_factory(), n_users=n_users, config=ec, seed=seed,
        )
        clocks[str(v)] = time.perf_counter() - t0
        eff = bench["efficiency"]
        env = env_factory()
        aligns: list[float] = []
        for i, r in enumerate(bench["active_results"]):
            if r.true_theta is None:
                continue
            aligns.append(
                _alignment_inferred_vs_true(
                    env, r.inferred_theta, r.true_theta, seed=seed + i + 3,
                )
            )
        metrics[str(v)] = {
            "mean_active_rounds": eff["mean_active_rounds"],
            "mean_random_rounds": eff["mean_random_rounds"],
            "error_reduction_pct": eff["error_reduction_pct"],
            "round_reduction_pct": eff["round_reduction_pct"],
            "mean_active_error": eff["mean_active_error"],
            "mean_random_error": eff["mean_random_error"],
            "mean_alignment_active": float(np.mean(aligns)) if aligns else 0.0,
            "wall_clock_s": clocks[str(v)],
        }

    return AblationResults(param_name, list(values), metrics, clocks)


def default_beta_values() -> list[float]:
    return [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


def default_query_budget_values() -> list[int]:
    return [3, 5, 8, 10, 15, 20]


def default_posterior_types() -> list[str]:
    return ["gaussian", "particle"]
