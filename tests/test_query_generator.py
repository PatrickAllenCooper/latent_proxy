from __future__ import annotations

import numpy as np
import pytest

from src.agents.query_generator import RandomQueryGenerator, StructuredQueryGenerator
from src.environments.resource_game import ResourceStrategyGame
from src.utils.posterior import ParticlePosterior


@pytest.fixture
def env() -> ResourceStrategyGame:
    game = ResourceStrategyGame()
    game.reset(seed=42)
    return game


@pytest.fixture
def posterior() -> ParticlePosterior:
    return ParticlePosterior(n_particles=200)


class TestStructuredQueryGenerator:
    def test_selects_valid_scenario(
        self, env: ResourceStrategyGame, posterior: ParticlePosterior,
    ) -> None:
        gen = StructuredQueryGenerator(
            n_scenarios_per_round=15,
            n_eig_samples=100,
            seed=42,
        )
        scenario = gen.select_query(env, posterior)
        assert scenario is not None
        np.testing.assert_allclose(scenario.option_a.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(scenario.option_b.sum(), 1.0, atol=1e-6)

    def test_selects_different_from_random_on_average(
        self, env: ResourceStrategyGame, posterior: ParticlePosterior,
    ) -> None:
        active_gen = StructuredQueryGenerator(
            n_scenarios_per_round=15, n_eig_samples=100, seed=42,
        )
        random_gen = RandomQueryGenerator(seed=42)

        active_params = []
        random_params = []
        for _ in range(5):
            a = active_gen.select_query(env, posterior)
            r = random_gen.select_query(env, posterior)
            active_params.append(a.target_param)
            random_params.append(r.target_param)

        assert len(active_params) == 5
        assert len(random_params) == 5


class TestRandomQueryGenerator:
    def test_selects_valid_scenario(
        self, env: ResourceStrategyGame, posterior: ParticlePosterior,
    ) -> None:
        gen = RandomQueryGenerator(seed=42)
        scenario = gen.select_query(env, posterior)
        assert scenario is not None
        assert scenario.option_a.shape[0] == env.config.n_channels

    def test_variety_over_multiple_calls(
        self, env: ResourceStrategyGame, posterior: ParticlePosterior,
    ) -> None:
        gen = RandomQueryGenerator(seed=42)
        scenarios = [gen.select_query(env, posterior) for _ in range(10)]
        params = {s.target_param for s in scenarios}
        assert len(params) >= 1
