from __future__ import annotations

import numpy as np
import pytest

from src.environments.resource_game import ResourceStrategyGame
from src.utils.diagnostic_scenarios import DiagnosticScenario, ScenarioLibrary
from src.utils.information_gain import compute_eig_batch, compute_eig_mc
from src.utils.posterior import GaussianPosterior, ParticlePosterior


@pytest.fixture
def env() -> ResourceStrategyGame:
    game = ResourceStrategyGame()
    game.reset(seed=42)
    return game


@pytest.fixture
def discriminating_scenario(env: ResourceStrategyGame) -> DiagnosticScenario:
    stats = env.get_channel_stats()
    obs = env._get_obs()
    return DiagnosticScenario(
        game_state=obs,
        option_a=np.array([0.6, 0.3, 0.05, 0.05]),
        option_b=np.array([0.05, 0.15, 0.4, 0.4]),
        target_param="alpha",
        description="Safe vs risky",
        channel_means=stats["means"],
        channel_variances=stats["variances"],
        current_wealth=float(obs["wealth"].sum()),
        rounds_remaining=env.config.n_rounds,
    )


@pytest.fixture
def nondiscriminating_scenario(env: ResourceStrategyGame) -> DiagnosticScenario:
    stats = env.get_channel_stats()
    obs = env._get_obs()
    return DiagnosticScenario(
        game_state=obs,
        option_a=np.array([0.25, 0.25, 0.25, 0.25]),
        option_b=np.array([0.25, 0.25, 0.25, 0.25]),
        target_param="none",
        description="Identical options",
        channel_means=stats["means"],
        channel_variances=stats["variances"],
        current_wealth=float(obs["wealth"].sum()),
        rounds_remaining=env.config.n_rounds,
    )


class TestEIGPositive:
    def test_discriminating_scenario_positive_eig(
        self, discriminating_scenario: DiagnosticScenario,
    ) -> None:
        posterior = ParticlePosterior(n_particles=200)
        eig = compute_eig_mc(
            discriminating_scenario, posterior, n_samples=200,
            rng=np.random.default_rng(0),
        )
        assert eig > 0, f"EIG should be positive for discriminating scenario: {eig}"

    def test_nondiscriminating_low_eig(
        self, nondiscriminating_scenario: DiagnosticScenario,
    ) -> None:
        posterior = ParticlePosterior(n_particles=200)
        eig = compute_eig_mc(
            nondiscriminating_scenario, posterior, n_samples=200,
            rng=np.random.default_rng(0),
        )
        assert eig < 0.1, f"EIG should be near zero for identical options: {eig}"


class TestEIGProperties:
    def test_eig_nonnegative(
        self, discriminating_scenario: DiagnosticScenario,
    ) -> None:
        posterior = GaussianPosterior()
        eig = compute_eig_mc(
            discriminating_scenario, posterior, n_samples=200,
            rng=np.random.default_rng(42),
        )
        assert eig >= 0

    def test_reproducible_with_seed(
        self, discriminating_scenario: DiagnosticScenario,
    ) -> None:
        posterior = ParticlePosterior(n_particles=200)
        eig1 = compute_eig_mc(
            discriminating_scenario, posterior, n_samples=100,
            rng=np.random.default_rng(99),
        )
        posterior2 = ParticlePosterior(n_particles=200)
        eig2 = compute_eig_mc(
            discriminating_scenario, posterior2, n_samples=100,
            rng=np.random.default_rng(99),
        )
        np.testing.assert_allclose(eig1, eig2, atol=1e-6)


class TestEIGBatch:
    def test_batch_length(
        self, discriminating_scenario: DiagnosticScenario,
        nondiscriminating_scenario: DiagnosticScenario,
    ) -> None:
        posterior = ParticlePosterior(n_particles=200)
        scenarios = [discriminating_scenario, nondiscriminating_scenario]
        eigs = compute_eig_batch(
            scenarios, posterior, n_samples=100,
            rng=np.random.default_rng(0),
        )
        assert eigs.shape == (2,)

    def test_batch_ordering(
        self, discriminating_scenario: DiagnosticScenario,
        nondiscriminating_scenario: DiagnosticScenario,
    ) -> None:
        posterior = ParticlePosterior(n_particles=200)
        scenarios = [discriminating_scenario, nondiscriminating_scenario]
        eigs = compute_eig_batch(
            scenarios, posterior, n_samples=200,
            rng=np.random.default_rng(0),
        )
        assert eigs[0] >= eigs[1], (
            f"Discriminating should have higher EIG: {eigs[0]} vs {eigs[1]}"
        )
