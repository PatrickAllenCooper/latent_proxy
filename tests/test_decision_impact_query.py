from __future__ import annotations

import numpy as np

from src.agents.query_generator import DecisionImpactQueryGenerator
from src.environments.game_variants import create_variant_a
from src.utils.posterior import ParticlePosterior


def test_decision_impact_generator_returns_simplex_options() -> None:
    env = create_variant_a()
    env.reset(seed=17)
    posterior = ParticlePosterior(n_particles=48)
    gen = DecisionImpactQueryGenerator(
        n_scenarios_per_round=9, n_particles=24, seed=19,
    )

    scenario = gen.select_query(env, posterior)

    assert scenario.option_a.shape == (env.config.n_channels,)
    assert scenario.option_b.shape == (env.config.n_channels,)
    np.testing.assert_allclose(scenario.option_a.sum(), 1.0)
    np.testing.assert_allclose(scenario.option_b.sum(), 1.0)


def test_action_disagreement_is_zero_for_identical_actions() -> None:
    actions = np.tile(np.array([[0.5, 0.5]]), (4, 1))
    weights = np.full(4, 0.25)
    assert DecisionImpactQueryGenerator._weighted_variance(actions, weights) == 0.0
