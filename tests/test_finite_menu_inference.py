"""Checks for the finite-menu inference and acquisition calculations."""

import numpy as np

from src.agents.finite_menu_inference import (
    UserParticles,
    acquisition_scores,
    particle_behavior,
    particle_utilities,
    update_weights,
)
from src.evaluation.preference_benchmark import MenuScenario, expected_utilities
from src.training.synthetic_users import UserType


def test_particle_utility_matches_scalar_oracle() -> None:
    payoffs = np.array([[[.04, 0.0], [.04, 0.0]],
                        [[0.0, .2], [0.0, -.03]]])
    scenario = MenuScenario("two", ("safe", "late"), payoffs,
                            np.array([[1.0, 0.0], [.7, .3]]),
                            np.ones(2, dtype=bool))
    types = [UserType(.3, .5, 1.5), UserType(.9, 1.5, 3.0)]
    particles = UserParticles(
        np.array([t.gamma for t in types]),
        np.array([t.alpha for t in types]),
        np.array([t.lambda_ for t in types]),
        np.zeros((2, 2)),
    )
    expected = np.stack([expected_utilities(scenario, t) for t in types])
    assert np.allclose(particle_utilities(scenario, particles), expected)


def test_query_value_is_nonnegative_and_posterior_updates() -> None:
    query_utilities = np.array([[[.3, 0.0], [0.0, .3]]])
    query_behavior = np.stack([particle_behavior(query_utilities[0], np.zeros((2, 2)))])
    target_utilities = np.array([[[.3, 0.0], [0.0, .3]]])
    weights = np.array([.5, .5])
    information, decision_value = acquisition_scores(
        weights, query_behavior, target_utilities
    )
    assert information[0] > 0.5
    assert decision_value[0] > 0.1
    updated = update_weights(weights, query_behavior[0, :, 0])
    assert updated[0] > .99
    assert np.isclose(updated.sum(), 1.0)
