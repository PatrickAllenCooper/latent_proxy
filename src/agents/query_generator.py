from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.resource_game import ResourceStrategyGame
from src.utils.diagnostic_scenarios import DiagnosticScenario, ScenarioLibrary
from src.utils.information_gain import compute_eig_batch
from src.utils.posterior import PosteriorBase


class StructuredQueryGenerator:
    """Selects diagnostic queries by maximizing Expected Information Gain.

    Option A from README Section 6.1: a structured elicitation module computes
    EIG over a predefined set of diagnostic game scenarios and selects the
    highest-scoring one.
    """

    def __init__(
        self,
        n_scenarios_per_round: int = 50,
        n_eig_samples: int = 500,
        temperature: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_scenarios_per_round = n_scenarios_per_round
        self.n_eig_samples = n_eig_samples
        self.temperature = temperature
        self._library = ScenarioLibrary(seed=seed)
        self._rng = np.random.default_rng(seed)

    def select_query(
        self,
        env: ResourceStrategyGame,
        posterior: PosteriorBase,
    ) -> DiagnosticScenario:
        """Select the diagnostic scenario with the highest EIG."""
        n_per_param = max(
            self.n_scenarios_per_round // 3, 3,
        )
        scenarios = self._library.generate_all(env, n_per_param)

        if not scenarios:
            scenarios = self._library.generate_all(env, n_per_param * 2)
        if not scenarios:
            return self._fallback_scenario(env)

        eig_scores = compute_eig_batch(
            scenarios, posterior, self.n_eig_samples,
            self.temperature, self._rng,
        )

        best_idx = int(np.argmax(eig_scores))
        return scenarios[best_idx]

    def _fallback_scenario(
        self, env: ResourceStrategyGame,
    ) -> DiagnosticScenario:
        """Generate a simple fallback scenario when the library is empty."""
        env.reset(seed=int(self._rng.integers(0, 2**31)))
        obs = env._get_obs()
        stats = env.get_channel_stats()
        K = env.config.n_channels
        a = np.zeros(K)
        a[0] = 0.6
        a[1] = 0.4
        b = np.zeros(K)
        b[2] = 0.5
        b[3] = 0.5
        return DiagnosticScenario(
            game_state=obs,
            option_a=a, option_b=b,
            target_param="alpha",
            description="Safe vs risky allocation.",
            channel_means=stats["means"],
            channel_variances=stats["variances"],
            current_wealth=float(obs["wealth"].sum()),
            rounds_remaining=env.config.n_rounds,
        )


class RandomQueryGenerator:
    """Baseline that picks diagnostic scenarios at random."""

    def __init__(self, seed: int = 42) -> None:
        self._library = ScenarioLibrary(seed=seed)
        self._rng = np.random.default_rng(seed)

    def select_query(
        self,
        env: ResourceStrategyGame,
        posterior: PosteriorBase,
    ) -> DiagnosticScenario:
        scenarios = self._library.generate_all(env, n_per_param=5)
        if not scenarios:
            scenarios = self._library.generate_all(env, n_per_param=10)
        if not scenarios:
            K = env.config.n_channels
            env.reset(seed=int(self._rng.integers(0, 2**31)))
            obs = env._get_obs()
            stats = env.get_channel_stats()
            a = self._rng.dirichlet(np.ones(K))
            b = self._rng.dirichlet(np.ones(K))
            return DiagnosticScenario(
                game_state=obs, option_a=a, option_b=b,
                target_param="unknown",
                description="Random allocation pair.",
                channel_means=stats["means"],
                channel_variances=stats["variances"],
                current_wealth=float(obs["wealth"].sum()),
                rounds_remaining=env.config.n_rounds,
            )
        idx = int(self._rng.integers(0, len(scenarios)))
        return scenarios[idx]
