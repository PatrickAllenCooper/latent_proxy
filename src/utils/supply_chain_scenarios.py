from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.diagnostic_scenarios import DiagnosticScenario, ScenarioLibraryBase


class SupplyChainScenarioLibrary(ScenarioLibraryBase):
    """Diagnostic binary-choice scenarios for the supply chain environment."""

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

            domestic_heavy = np.zeros(K)
            domestic_heavy[0] = 0.50
            domestic_heavy[1] = 0.30
            domestic_heavy[2] = 0.05
            if K > 3:
                domestic_heavy[3] = 0.10
            if K > 4:
                domestic_heavy[4] = 0.05

            budget_heavy = np.zeros(K)
            budget_heavy[2] = 0.35
            if K > 4:
                budget_heavy[4] = 0.25
            budget_heavy[1] = 0.20
            if K > 3:
                budget_heavy[3] = 0.15
            budget_heavy[0] = max(0.0, 1.0 - budget_heavy.sum())

            domestic_heavy /= domestic_heavy.sum()
            budget_heavy /= budget_heavy.sum()

            rounds_left = env.config.n_periods - obs["round"]
            mp_horizon = max(8, min(rounds_left, 20))

            scenarios.append(DiagnosticScenario(
                game_state=obs,
                option_a=domestic_heavy,
                option_b=budget_heavy,
                target_param="gamma",
                description=(
                    "Option A favors reliable domestic suppliers for steady long-term "
                    "procurement. Option B tilts toward cheaper overseas and spot market "
                    "sources for higher expected savings with more supply risk."
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

            low_risk = np.zeros(K)
            low_risk[0] = 0.55
            low_risk[1] = 0.30
            low_risk[2] = 0.05
            low_risk[3] = 0.05
            low_risk[4] = 0.05 if K > 4 else 0.0
            low_risk /= low_risk.sum()

            high_risk = np.zeros(K)
            high_risk[0] = 0.05
            high_risk[1] = 0.10
            high_risk[2] = 0.35
            high_risk[3] = 0.25
            high_risk[4] = 0.25 if K > 4 else 0.0
            high_risk /= high_risk.sum()

            scenarios.append(DiagnosticScenario(
                game_state=obs,
                option_a=low_risk,
                option_b=high_risk,
                target_param="alpha",
                description=(
                    "Option A concentrates on reliable domestic suppliers. "
                    "Option B uses cheaper overseas and spot market suppliers "
                    "with higher delivery variance."
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
                safe_mix = np.array(
                    [0.45, 0.30, 0.08, 0.10, 0.07][:K], dtype=np.float64,
                )
                risky_mix = np.array(
                    [0.08, 0.12, 0.30, 0.22, 0.28][:K], dtype=np.float64,
                )
                safe_mix = safe_mix / safe_mix.sum()
                risky_mix = risky_mix / risky_mix.sum()
                loss_averse, loss_neutral = safe_mix, risky_mix

            scenarios.append(DiagnosticScenario(
                game_state=obs,
                option_a=loss_averse,
                option_b=loss_neutral,
                target_param="lambda_",
                description=(
                    "Option A mitigates supply chain disruption risk. "
                    "Option B accepts more downside exposure for cost savings."
                ),
                channel_means=stats["means"],
                channel_variances=stats["variances"],
                current_wealth=float(obs["wealth"].sum()),
                rounds_remaining=env.config.n_periods - obs["round"],
            ))

        return scenarios
