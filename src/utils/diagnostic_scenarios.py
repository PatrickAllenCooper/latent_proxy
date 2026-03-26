from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.resource_game import ResourceStrategyGame


@dataclass
class DiagnosticScenario:
    """A binary choice scenario designed to discriminate preference parameters.

    Attributes:
        game_state: Environment observation dict at the time of the query.
        option_a: First candidate allocation (numpy array summing to 1).
        option_b: Second candidate allocation (numpy array summing to 1).
        target_param: Which parameter this scenario most discriminates.
        description: Human-readable description of the tradeoff.
        channel_means: Channel expected returns in the current regime.
        channel_variances: Channel return variances in the current regime.
        current_wealth: Total portfolio value at query time.
        rounds_remaining: Rounds left in the game.
        multiperiod_horizon: If set (>1), expected utility uses multi-period
            compounding (for gamma-identifiable scenarios). None uses single-period
            evaluation with rounds_remaining discounting.
    """

    game_state: dict[str, Any]
    option_a: NDArray[np.floating[Any]]
    option_b: NDArray[np.floating[Any]]
    target_param: str
    description: str
    channel_means: NDArray[np.floating[Any]]
    channel_variances: NDArray[np.floating[Any]]
    current_wealth: float
    rounds_remaining: int
    multiperiod_horizon: int | None = None


class ScenarioLibraryBase(ABC):
    """Abstract scenario library for structured elicitation (game, stock, etc.)."""

    @abstractmethod
    def generate_gamma_scenarios(
        self, env: Any, n: int = 10,
    ) -> list[DiagnosticScenario]:
        """Scenarios targeting discount factor (gamma)."""

    @abstractmethod
    def generate_alpha_scenarios(
        self, env: Any, n: int = 10,
    ) -> list[DiagnosticScenario]:
        """Scenarios targeting risk aversion (alpha)."""

    @abstractmethod
    def generate_lambda_scenarios(
        self, env: Any, n: int = 10,
    ) -> list[DiagnosticScenario]:
        """Scenarios targeting loss aversion (lambda_)."""

    def generate_all(
        self, env: Any, n_per_param: int = 10,
    ) -> list[DiagnosticScenario]:
        scenarios: list[DiagnosticScenario] = []
        scenarios.extend(self.generate_gamma_scenarios(env, n_per_param))
        scenarios.extend(self.generate_alpha_scenarios(env, n_per_param))
        scenarios.extend(self.generate_lambda_scenarios(env, n_per_param))
        return scenarios


class ScenarioLibrary(ScenarioLibraryBase):
    """Generates diagnostic scenarios from the resource strategy game.

    Each generator produces scenarios where the two options differ along a
    specific preference dimension (gamma, alpha, or lambda_), so the user's
    choice is maximally informative about that parameter.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_gamma_scenarios(
        self, env: ResourceStrategyGame, n: int = 10,
    ) -> list[DiagnosticScenario]:
        """Scenarios discriminating time-horizon preference (gamma).

        High-gamma (patient) users prefer conservative allocations that compound
        well over many rounds. Low-gamma (impatient) users prefer aggressive
        allocations that maximize immediate returns.
        """
        scenarios = []
        for i in range(n):
            env.reset(seed=int(self._rng.integers(0, 2**31)))
            n_steps = int(self._rng.integers(0, env.config.n_rounds // 2))
            uniform = np.ones(env.config.n_channels) / env.config.n_channels
            for _ in range(n_steps):
                env.step(uniform)

            obs = env._get_obs()
            stats = env.get_channel_stats()

            K = env.config.n_channels
            safe_heavy = np.zeros(K)
            safe_heavy[0] = 0.70
            safe_heavy[1] = 0.30
            aggressive_heavy = np.zeros(K)
            aggressive_heavy[2] = 0.60
            aggressive_heavy[3] = 0.40

            rounds_left = env.config.n_rounds - obs["round"]
            mp_horizon = max(10, min(rounds_left, 20))

            scenarios.append(DiagnosticScenario(
                game_state=obs,
                option_a=safe_heavy,
                option_b=aggressive_heavy,
                target_param="gamma",
                description=(
                    "Option A favors long-term compounding with lower-risk channels. "
                    "Option B favors high immediate returns from aggressive channels."
                ),
                channel_means=stats["means"],
                channel_variances=stats["variances"],
                current_wealth=float(obs["wealth"].sum()),
                rounds_remaining=rounds_left,
                multiperiod_horizon=mp_horizon,
            ))

        return scenarios

    def generate_alpha_scenarios(
        self, env: ResourceStrategyGame, n: int = 10,
    ) -> list[DiagnosticScenario]:
        """Scenarios discriminating risk aversion (alpha).

        High-alpha (risk-averse) users prefer low-variance allocations.
        Low-alpha (risk-seeking) users tolerate high variance for higher returns.
        """
        scenarios = []
        for i in range(n):
            env.reset(seed=int(self._rng.integers(0, 2**31)))
            n_steps = int(self._rng.integers(0, env.config.n_rounds // 2))
            uniform = np.ones(env.config.n_channels) / env.config.n_channels
            for _ in range(n_steps):
                env.step(uniform)

            obs = env._get_obs()
            stats = env.get_channel_stats()

            K = env.config.n_channels
            low_risk = np.zeros(K)
            low_risk[0] = 0.75
            low_risk[1] = 0.25
            high_risk = np.zeros(K)
            high_risk[1] = 0.15
            high_risk[2] = 0.50
            high_risk[3] = 0.35

            scenarios.append(DiagnosticScenario(
                game_state=obs,
                option_a=low_risk,
                option_b=high_risk,
                target_param="alpha",
                description=(
                    "Option A is a conservative, low-volatility allocation. "
                    "Option B is an aggressive, high-return but high-risk allocation."
                ),
                channel_means=stats["means"],
                channel_variances=stats["variances"],
                current_wealth=float(obs["wealth"].sum()),
                rounds_remaining=env.config.n_rounds - obs["round"],
            ))

        return scenarios

    def generate_lambda_scenarios(
        self, env: ResourceStrategyGame, n: int = 10,
    ) -> list[DiagnosticScenario]:
        """Scenarios discriminating loss aversion (lambda_).

        High-lambda (loss-averse) users avoid allocations with downside risk.
        Low-lambda users are less sensitive to potential losses.
        """
        scenarios = []
        for i in range(n):
            env.reset(seed=int(self._rng.integers(0, 2**31)))
            n_steps = int(self._rng.integers(0, env.config.n_rounds // 2))
            uniform = np.ones(env.config.n_channels) / env.config.n_channels
            for _ in range(n_steps):
                env.step(uniform)

            obs = env._get_obs()
            stats = env.get_channel_stats()

            loss_averse_action = env.get_optimal_action(
                {"gamma": 0.7, "alpha": 1.5, "lambda_": 3.0}
            )
            loss_neutral_action = env.get_optimal_action(
                {"gamma": 0.7, "alpha": 0.3, "lambda_": 1.01}
            )

            if np.allclose(loss_averse_action, loss_neutral_action, atol=0.02):
                continue

            scenarios.append(DiagnosticScenario(
                game_state=obs,
                option_a=loss_averse_action,
                option_b=loss_neutral_action,
                target_param="lambda_",
                description=(
                    "Option A minimizes downside loss exposure at the cost of upside. "
                    "Option B accepts more loss risk for higher expected returns."
                ),
                channel_means=stats["means"],
                channel_variances=stats["variances"],
                current_wealth=float(obs["wealth"].sum()),
                rounds_remaining=env.config.n_rounds - obs["round"],
            ))

        return scenarios
