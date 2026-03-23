from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop, ElicitationResult
from src.environments.resource_game import ResourceStrategyGame
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler

logger = logging.getLogger(__name__)


def compute_elicitation_efficiency(
    active_results: list[ElicitationResult],
    random_results: list[ElicitationResult],
) -> dict[str, float]:
    """Compare elicitation efficiency between active and random query strategies.

    The key metric is the reduction in rounds required to achieve comparable
    preference recovery accuracy. Target: 30%+ reduction (README Section 7.1).
    """
    active_rounds = [r.n_rounds for r in active_results]
    random_rounds = [r.n_rounds for r in random_results]

    active_errors = []
    random_errors = []
    for r in active_results:
        err = r.preference_recovery_error()
        if err:
            active_errors.append(err["total"])
    for r in random_results:
        err = r.preference_recovery_error()
        if err:
            random_errors.append(err["total"])

    mean_active_rounds = float(np.mean(active_rounds)) if active_rounds else 0.0
    mean_random_rounds = float(np.mean(random_rounds)) if random_rounds else 0.0
    mean_active_error = float(np.mean(active_errors)) if active_errors else 0.0
    mean_random_error = float(np.mean(random_errors)) if random_errors else 0.0

    round_reduction = 0.0
    if mean_random_rounds > 0:
        round_reduction = (mean_random_rounds - mean_active_rounds) / mean_random_rounds

    error_reduction = 0.0
    if mean_random_error > 0:
        error_reduction = (mean_random_error - mean_active_error) / mean_random_error

    return {
        "mean_active_rounds": mean_active_rounds,
        "mean_random_rounds": mean_random_rounds,
        "round_reduction_pct": round_reduction * 100,
        "mean_active_error": mean_active_error,
        "mean_random_error": mean_random_error,
        "error_reduction_pct": error_reduction * 100,
    }


def compute_recovery_curve(
    result: ElicitationResult,
) -> dict[str, Any]:
    """Per-round trajectory of posterior variance and recovery error."""
    trajectory = result.variance_trajectory

    per_round_error = []
    if result.true_theta is not None:
        true = {
            "gamma": result.true_theta.gamma,
            "alpha": result.true_theta.alpha,
            "lambda_": result.true_theta.lambda_,
        }
        for var_dict in trajectory:
            round_err = sum(abs(var_dict.get(k, 0.5) - true.get(k, 0.5))
                           for k in true) / len(true)
            per_round_error.append(round_err)

    return {
        "variance_trajectory": trajectory,
        "n_rounds": result.n_rounds,
        "convergence_reason": result.convergence_reason,
        "final_error": result.preference_recovery_error(),
    }


def run_elicitation_benchmark(
    env: ResourceStrategyGame,
    n_users: int = 20,
    config: ElicitationConfig | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Full benchmark comparing active vs random elicitation.

    Samples n_users from the prior, runs both strategies on each,
    and aggregates efficiency metrics.
    """
    config = config or ElicitationConfig()
    sampler = SyntheticUserSampler(seed=seed)

    active_results: list[ElicitationResult] = []
    random_results: list[ElicitationResult] = []

    for i in range(n_users):
        ut = sampler.sample()
        user_active = SyntheticUser(ut, temperature=config.temperature, seed=seed + i)
        user_random = SyntheticUser(ut, temperature=config.temperature, seed=seed + i)

        env.reset(seed=seed + i)

        active_config = ElicitationConfig(
            posterior_type=config.posterior_type,
            n_particles=config.n_particles,
            max_rounds=config.max_rounds,
            n_scenarios_per_round=config.n_scenarios_per_round,
            n_eig_samples=config.n_eig_samples,
            temperature=config.temperature,
            convergence=config.convergence,
            seed=seed + i * 1000,
        )

        loop_active = ElicitationLoop(active_config)
        result_active = loop_active.run(env, user_active, query_type="active")
        active_results.append(result_active)

        random_config = ElicitationConfig(
            posterior_type=config.posterior_type,
            n_particles=config.n_particles,
            max_rounds=config.max_rounds,
            temperature=config.temperature,
            convergence=config.convergence,
            seed=seed + i * 2000,
        )

        loop_random = ElicitationLoop(random_config)
        result_random = loop_random.run(env, user_random, query_type="random")
        random_results.append(result_random)

        logger.info(
            "User %d/%d: active=%d rounds (err=%.3f), random=%d rounds (err=%.3f)",
            i + 1, n_users,
            result_active.n_rounds,
            (result_active.preference_recovery_error() or {}).get("total", -1),
            result_random.n_rounds,
            (result_random.preference_recovery_error() or {}).get("total", -1),
        )

    efficiency = compute_elicitation_efficiency(active_results, random_results)

    return {
        "efficiency": efficiency,
        "active_results": active_results,
        "random_results": random_results,
        "n_users": n_users,
    }
