"""Theta stability: test-retest reliability of preference inference (README Section 7.2)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import stats as sp_stats

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.base import BaseEnvironment
from src.evaluation.statistical_analysis import icc_2_1
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler, UserType

logger = logging.getLogger(__name__)


@dataclass
class ThetaStabilityResult:
    """Test-retest reliability across multiple elicitation sessions."""

    per_user_theta: list[list[dict[str, float]]]
    icc_per_param: dict[str, float]
    pearson_per_param: dict[str, float]
    n_users: int = 0
    n_sessions: int = 2


def run_user_sessions(
    ut: UserType,
    env_factory: Callable[[], BaseEnvironment],
    elicitation: ElicitationConfig,
    seed: int,
    user_index: int,
    n_sessions: int,
) -> list[dict[str, float]]:
    """Run ``n_sessions`` independent elicitation sessions for one user.

    Uses the same per-session seeding scheme as ``run_theta_stability_test``
    so results are reproducible and comparable across callers.
    """
    session_thetas: list[dict[str, float]] = []
    for s in range(n_sessions):
        env = env_factory()
        session_seed = seed + user_index * 1000 + s * 100 + 7
        user = SyntheticUser(
            ut, temperature=elicitation.temperature, seed=session_seed + 1,
        )
        conv = elicitation.convergence
        if conv.max_rounds < elicitation.max_rounds:
            conv = ConvergenceConfig(
                gamma_variance_threshold=conv.gamma_variance_threshold,
                alpha_variance_threshold=conv.alpha_variance_threshold,
                lambda_variance_threshold=conv.lambda_variance_threshold,
                robust_action_level=conv.robust_action_level,
                max_rounds=elicitation.max_rounds,
            )
        cfg = ElicitationConfig(
            posterior_type=elicitation.posterior_type,
            n_particles=elicitation.n_particles,
            max_rounds=elicitation.max_rounds,
            n_scenarios_per_round=elicitation.n_scenarios_per_round,
            n_eig_samples=elicitation.n_eig_samples,
            temperature=elicitation.temperature,
            convergence=conv,
            seed=session_seed,
            scenario_library=elicitation.scenario_library,
            reference_point_mode=elicitation.reference_point_mode,
        )
        loop = ElicitationLoop(cfg)
        env.reset(seed=session_seed + 2)
        res = loop.run(env, user, query_type="active")
        session_thetas.append(res.inferred_theta)
    return session_thetas


def compute_stability_metrics(
    per_user: list[list[dict[str, float]]],
    param_names: list[str] | None = None,
    all_pairs: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """ICC(2,1) and Pearson correlation per parameter across sessions.

    Args:
        per_user: per_user[i][s] is the inferred theta dict for user i,
            session s.
        all_pairs: If False (default, historical behavior) only the first
            two sessions are compared. If True, ICC/Pearson are averaged
            over all session pairs (useful when n_sessions > 2).
    """
    param_names = param_names or ["gamma", "alpha", "lambda_"]
    n_users = len(per_user)
    n_sessions = min(len(u) for u in per_user) if per_user else 0

    if all_pairs and n_sessions > 2:
        pairs = [
            (a, b)
            for a in range(n_sessions)
            for b in range(a + 1, n_sessions)
        ]
    else:
        pairs = [(0, 1)]

    icc_per_param: dict[str, float] = {}
    pearson_per_param: dict[str, float] = {}

    for p in param_names:
        icc_vals: list[float] = []
        pearson_vals: list[float] = []
        for a, b in pairs:
            s1 = np.array(
                [per_user[i][a].get(p, 0.0) for i in range(n_users)],
                dtype=np.float64,
            )
            s2 = np.array(
                [per_user[i][b].get(p, 0.0) for i in range(n_users)],
                dtype=np.float64,
            )
            icc_vals.append(icc_2_1(s1, s2))
            if n_users >= 3 and np.std(s1) > 1e-10 and np.std(s2) > 1e-10:
                corr, _ = sp_stats.pearsonr(s1, s2)
                pearson_vals.append(float(corr))
            else:
                pearson_vals.append(0.0)
        icc_per_param[p] = float(np.mean(icc_vals))
        pearson_per_param[p] = float(np.mean(pearson_vals))

    return icc_per_param, pearson_per_param


def run_theta_stability_test(
    env_factory: Callable[[], BaseEnvironment],
    n_users: int = 10,
    n_sessions: int = 2,
    elicitation: ElicitationConfig | None = None,
    seed: int = 42,
) -> ThetaStabilityResult:
    """Run multiple independent elicitation sessions per user and measure reliability.

    For each synthetic user, ``n_sessions`` independent elicitation loops are run
    on the same environment type (different RNG seeds for scenario selection).
    ICC(2,1) and per-parameter Pearson correlation measure stability.
    """
    elicitation = elicitation or ElicitationConfig()
    sampler = SyntheticUserSampler(seed=seed)

    per_user: list[list[dict[str, float]]] = []
    param_names = ["gamma", "alpha", "lambda_"]

    for i in range(n_users):
        ut = sampler.sample()
        session_thetas = run_user_sessions(
            ut, env_factory, elicitation, seed=seed,
            user_index=i, n_sessions=n_sessions,
        )
        per_user.append(session_thetas)

    icc_per_param, pearson_per_param = compute_stability_metrics(
        per_user, param_names=param_names, all_pairs=False,
    )

    return ThetaStabilityResult(
        per_user_theta=per_user,
        icc_per_param=icc_per_param,
        pearson_per_param=pearson_per_param,
        n_users=n_users,
        n_sessions=n_sessions,
    )
