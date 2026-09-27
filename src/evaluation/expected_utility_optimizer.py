"""Deterministic expected-utility optimization over the allocation simplex."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.training.synthetic_users import UserType, prospect_utility


def expected_allocation_utility(
    allocation: NDArray[np.floating],
    theta: UserType,
    channel_means: NDArray[np.floating],
    channel_variances: NDArray[np.floating],
    current_wealth: float,
    rounds_remaining: int,
    reference_point: float = 0.0,
    utility_form: str = "absolute",
    n_quadrature_points: int = 48,
) -> float:
    """Evaluate expected one-period prospect utility with Gauss-Hermite nodes."""
    allocation = np.asarray(allocation, dtype=np.float64)
    means = np.asarray(channel_means, dtype=np.float64)
    variances = np.asarray(channel_variances, dtype=np.float64)
    port_mean = float(np.dot(allocation, means))
    port_var = max(0.0, float(np.dot(np.square(allocation), variances)))
    nodes, weights = np.polynomial.hermite.hermgauss(n_quadrature_points)
    returns = port_mean + np.sqrt(2.0 * port_var) * nodes
    outcomes = returns if utility_form == "return_normalized" else current_wealth * (1.0 + returns)
    utility = prospect_utility(
        outcomes, theta.alpha, theta.lambda_, reference_point,
    )
    return float(np.dot(weights / np.sqrt(np.pi), utility) * theta.gamma ** rounds_remaining)


def optimize_expected_allocation(
    theta: UserType,
    channel_means: NDArray[np.floating],
    channel_variances: NDArray[np.floating],
    current_wealth: float,
    rounds_remaining: int,
    reference_point: float = 0.0,
    utility_form: str = "absolute",
    initial_actions: list[NDArray[np.floating]] | None = None,
    n_quadrature_points: int = 48,
) -> tuple[NDArray[np.float64], float]:
    """Find a multi-start SLSQP optimum on the long-only unit simplex.

    The optimizer is deliberately transparent and deterministic. Starts include
    equal allocation, each simplex vertex, and any caller-provided candidate
    actions; all are evaluated with the same quadrature objective.
    """
    n_channels = len(channel_means)
    uniform = np.full(n_channels, 1.0 / n_channels, dtype=np.float64)
    starts = [uniform, *np.eye(n_channels, dtype=np.float64)]
    if initial_actions:
        starts.extend(np.asarray(a, dtype=np.float64) for a in initial_actions)

    def value(action: NDArray[np.floating]) -> float:
        return expected_allocation_utility(
            action, theta, channel_means, channel_variances, current_wealth,
            rounds_remaining, reference_point, utility_form, n_quadrature_points,
        )

    objective: Callable[[NDArray[np.float64]], float] = lambda x: -value(x)
    best_action = uniform.copy()
    best_value = value(best_action)
    constraints = ({"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)},)
    bounds = [(0.0, 1.0)] * n_channels
    for start in starts:
        start = np.clip(np.asarray(start, dtype=np.float64), 0.0, 1.0)
        if start.sum() <= 1e-12:
            start = uniform.copy()
        else:
            start /= start.sum()
        result = minimize(
            objective, start, method="SLSQP", bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-11, "disp": False},
        )
        candidate = np.clip(result.x, 0.0, 1.0)
        candidate /= max(float(candidate.sum()), 1e-12)
        candidate_value = value(candidate)
        if candidate_value > best_value:
            best_action, best_value = candidate, candidate_value
    return best_action.astype(np.float64), float(best_value)
