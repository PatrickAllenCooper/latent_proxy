from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.diagnostic_scenarios import DiagnosticScenario, ScenarioLibraryBase


class StockScenarioLibrary(ScenarioLibraryBase):
    """Diagnostic binary-choice scenarios for the stock backtest environment."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_gamma_scenarios(
        self, env: Any, n: int = 10,
    ) -> list[DiagnosticScenario]:
        scenarios: list[DiagnosticScenario] = []
        K = env.config.n_channels
        if K < 2:
            return scenarios

        for _ in range(n):
            env.reset(seed=int(self._rng.integers(0, 2**31)))
            n_steps = int(self._rng.integers(0, max(1, env.config.n_periods // 2)))
            uniform = np.ones(K) / K
            for _ in range(n_steps):
                env.step(uniform)

            obs = env._get_obs()
            stats = env.get_channel_stats()

            treasury_heavy = np.zeros(K)
            treasury_heavy[0] = 0.55
            treasury_heavy[1] = 0.25
            if K > 3:
                treasury_heavy[3] = 0.10
            treasury_heavy[K - 1] += 0.05
            treasury_heavy[2] = max(0.0, 1.0 - treasury_heavy.sum())
            treasury_heavy /= treasury_heavy.sum()

            equity_heavy = np.zeros(K)
            equity_heavy[2] = 0.35
            equity_heavy[4] = 0.25 if K > 4 else 0.0
            equity_heavy[1] = 0.20
            equity_heavy[3] = 0.15 if K > 3 else 0.0
            equity_heavy[0] = max(0.0, 1.0 - equity_heavy.sum())
            equity_heavy /= equity_heavy.sum()

            rounds_left = env.config.n_periods - obs["round"]
            mp_horizon = max(10, min(rounds_left, 20))

            scenarios.append(DiagnosticScenario(
                game_state=obs,
                option_a=treasury_heavy,
                option_b=equity_heavy,
                target_param="gamma",
                description=(
                    "Option A emphasizes fixed income and quality equities for "
                    "smoother compounding. Option B tilts toward higher-beta equities "
                    "and real assets for higher expected return with more path risk."
                ),
                channel_means=stats["means"],
                channel_variances=stats["variances"],
                current_wealth=float(obs["wealth"].sum()),
                rounds_remaining=rounds_left,
                multiperiod_horizon=mp_horizon,
            ))

        return scenarios

    def generate_alpha_scenarios(
        self, env: Any, n: int = 10,
    ) -> list[DiagnosticScenario]:
        scenarios: list[DiagnosticScenario] = []
        K = env.config.n_channels
        if K < 4:
            return scenarios

        for _ in range(n):
            env.reset(seed=int(self._rng.integers(0, 2**31)))
            n_steps = int(self._rng.integers(0, max(1, env.config.n_periods // 2)))
            uniform = np.ones(K) / K
            for _ in range(n_steps):
                env.step(uniform)

            obs = env._get_obs()
            stats = env.get_channel_stats()

            low_vol = np.zeros(K)
            low_vol[0] = 0.58
            low_vol[1] = 0.32
            low_vol[2] = 0.04
            low_vol[3] = 0.04
            low_vol[4] = 0.02

            high_vol = np.zeros(K)
            high_vol[0] = 0.04
            high_vol[1] = 0.08
            high_vol[2] = 0.38
            high_vol[3] = 0.26
            high_vol[4] = 0.24

            scenarios.append(DiagnosticScenario(
                game_state=obs,
                option_a=low_vol,
                option_b=high_vol,
                target_param="alpha",
                description=(
                    "Option A is a defensive mix (bonds and large caps). "
                    "Option B adds small caps, international, and REITs for higher risk."
                ),
                channel_means=stats["means"],
                channel_variances=stats["variances"],
                current_wealth=float(obs["wealth"].sum()),
                rounds_remaining=env.config.n_periods - obs["round"],
            ))

        return scenarios

    def generate_lambda_scenarios(
        self, env: Any, n: int = 10,
    ) -> list[DiagnosticScenario]:
        scenarios: list[DiagnosticScenario] = []
        max_attempts = max(n * 8, n + 5)
        attempts = 0
        while len(scenarios) < n and attempts < max_attempts:
            attempts += 1
            env.reset(seed=int(self._rng.integers(0, 2**31)))
            n_steps = int(self._rng.integers(0, max(1, env.config.n_periods // 2)))
            K = env.config.n_channels
            uniform = np.ones(K) / K
            for _ in range(n_steps):
                env.step(uniform)

            obs = env._get_obs()
            stats = env.get_channel_stats()

            loss_averse = env.get_optimal_action(
                {"gamma": 0.75, "alpha": 1.2, "lambda_": 3.5}
            )
            loss_neutral = env.get_optimal_action(
                {"gamma": 0.75, "alpha": 1.2, "lambda_": 1.05}
            )

            if np.allclose(loss_averse, loss_neutral, atol=0.02):
                tail_safe = np.array([0.48, 0.32, 0.08, 0.08, 0.04][:K], dtype=np.float64)
                tail_risky = np.array([0.06, 0.12, 0.28, 0.24, 0.30][:K], dtype=np.float64)
                if K < 5:
                    tail_safe = np.ones(K) / K
                    tail_risky = np.ones(K) / K
                tail_safe = tail_safe / tail_safe.sum()
                tail_risky = tail_risky / tail_risky.sum()
                loss_averse, loss_neutral = tail_safe, tail_risky

            scenarios.append(DiagnosticScenario(
                game_state=obs,
                option_a=loss_averse,
                option_b=loss_neutral,
                target_param="lambda_",
                description=(
                    "Option A tilts away from downside tail risk. "
                    "Option B accepts more drawdown risk for upside."
                ),
                channel_means=stats["means"],
                channel_variances=stats["variances"],
                current_wealth=float(obs["wealth"].sum()),
                rounds_remaining=env.config.n_periods - obs["round"],
            ))

        return scenarios
