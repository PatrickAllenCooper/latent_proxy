"""Versioned finite-menu benchmark for utility and behavioral agreement.

Payoffs are fractions of initial resources. A signed payoff at time t has
value gamma**t * prospect_utility(payoff, alpha, lambda). This objective is
deliberately separate from the project's historical terminal-wealth scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.training.synthetic_users import UserType, prospect_utility

BENCHMARK_VERSION = "finite_menu_v2"


@dataclass(frozen=True)
class MenuScenario:
    """One decision with shared outcome probabilities across all actions."""

    scenario_id: str
    labels: tuple[str, ...]
    payoffs: NDArray[np.float64]  # action, outcome, period
    probabilities: NDArray[np.float64]  # action, outcome
    feasible: NDArray[np.bool_]

    def __post_init__(self) -> None:
        n_actions, n_outcomes, _ = self.payoffs.shape
        if self.probabilities.shape != (n_actions, n_outcomes):
            raise ValueError("probability shape does not match payoffs")
        if len(self.labels) != n_actions or self.feasible.shape != (n_actions,):
            raise ValueError("labels or feasibility shape does not match actions")
        if not np.any(self.feasible):
            raise ValueError("scenario must have at least one feasible action")
        if not np.all(np.isfinite(self.payoffs)):
            raise ValueError("payoffs must be finite")
        if np.any(self.probabilities < 0) or not np.allclose(
            self.probabilities.sum(axis=1), 1.0
        ):
            raise ValueError("each action needs normalized probabilities")


def make_scenario(seed: int, scenario_id: str | None = None) -> MenuScenario:
    """Produce varied timing, risk, and loss tradeoffs with four actions."""
    rng = np.random.default_rng(seed)
    scale = float(rng.uniform(0.75, 1.35))
    safe = float(rng.uniform(0.035, 0.065)) * scale
    delayed = safe * float(rng.uniform(2.0, 9.5))
    risky_gain = safe * float(rng.uniform(3.0, 14.0))
    risky_loss = safe * float(rng.uniform(0.1, 3.0))
    balanced_gain = safe * float(rng.uniform(2.0, 8.0))
    balanced_loss = safe * float(rng.uniform(0.1, 2.0))
    delayed_t = int(rng.integers(2, 5))
    payoffs = np.zeros((4, 2, 5), dtype=np.float64)
    payoffs[0, :, 0] = safe
    payoffs[1, :, delayed_t] = delayed
    payoffs[2, 0, 0] = risky_gain
    payoffs[2, 1, 0] = -risky_loss
    payoffs[3, 0, 1] = balanced_gain
    payoffs[3, 1, 1] = -balanced_loss
    risky_p = float(rng.uniform(0.4, 0.78))
    balanced_p = float(rng.uniform(0.65, 0.94))
    probabilities = np.array(
        [[1.0, 0.0], [1.0, 0.0], [risky_p, 1.0 - risky_p],
         [balanced_p, 1.0 - balanced_p]], dtype=np.float64
    )
    return MenuScenario(
        scenario_id=scenario_id or f"menu-{seed}",
        labels=("safe_now", "larger_later", "risky_now", "balanced_later"),
        payoffs=payoffs,
        probabilities=probabilities,
        feasible=np.ones(4, dtype=np.bool_),
    )


def expected_utilities(scenario: MenuScenario, theta: UserType) -> NDArray[np.float64]:
    """Compute independent ground-truth expected utility per action."""
    periods = np.arange(scenario.payoffs.shape[-1], dtype=np.float64)
    values = prospect_utility(scenario.payoffs, theta.alpha, theta.lambda_)
    outcome_values = np.sum(values * theta.gamma**periods, axis=-1)
    result = np.sum(outcome_values * scenario.probabilities, axis=-1)
    return np.where(scenario.feasible, result, -np.inf)


def behavior_probabilities(
    scenario: MenuScenario,
    theta: UserType,
    *,
    temperature: float = 0.025,
    bias: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Bounded-rational policy; bias represents behavior beyond utility."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = expected_utilities(scenario, theta)
    if bias is not None:
        if bias.shape != scores.shape:
            raise ValueError("bias shape does not match actions")
        scores = scores + bias
    logits = (scores - np.max(scores)) / temperature
    weights = np.exp(logits)
    weights[~scenario.feasible] = 0.0
    return weights / weights.sum()


def evaluate_action(
    scenario: MenuScenario,
    theta: UserType,
    action: int,
    *,
    behavior_bias: NDArray[np.float64] | None = None,
    temperature: float = 0.025,
) -> dict[str, float | int | bool]:
    """Evaluate a decision using hidden truth, independently of its proxy."""
    if not 0 <= action < len(scenario.labels):
        raise ValueError("action is outside the menu")
    utilities = expected_utilities(scenario, theta)
    behavior = behavior_probabilities(
        scenario, theta, temperature=temperature, bias=behavior_bias
    )
    feasible_values = utilities[scenario.feasible]
    utility_range = float(np.max(feasible_values) - np.min(feasible_values))
    regret = float(np.max(feasible_values) - utilities[action])
    return {
        "action": action,
        "feasible": bool(scenario.feasible[action]),
        "utility": float(utilities[action]),
        "regret": regret,
        "normalized_regret": regret / utility_range if utility_range > 1e-10 else 0.0,
        "indifferent_menu": utility_range <= 1e-10,
        "behavioral_agreement": float(behavior[action]),
    }
