from __future__ import annotations

import numpy as np

from src.environments.game_variants import create_variant_a
from src.evaluation.adherence_study import _simplex_lattice, project_to_quality_floor
from src.training.dpo_data import DPOPairConfig, DPOPairGenerator


def test_simplex_lattice_is_normalized() -> None:
    actions = _simplex_lattice(4, 8)
    assert len(actions) == 165
    for action in actions:
        assert np.all(action >= 0)
        np.testing.assert_allclose(action.sum(), 1.0)


def test_quality_projection_returns_feasible_action() -> None:
    env = create_variant_a()
    env.reset(seed=3)
    projected = project_to_quality_floor(np.array([1.0, 0.0, 0.0, 0.0]), env, units=8)
    assert env.check_quality_floor(projected)[0]
    np.testing.assert_allclose(projected.sum(), 1.0)


def test_dialogue_phase2_pairs_hide_explicit_profile() -> None:
    cfg = DPOPairConfig(
        n_pairs=1, n_candidates=4, n_game_states=2, curriculum_phase=2,
        seed=23, dialogue_context_rounds=2,
    )
    pair = DPOPairGenerator(cfg).generate_dataset()[0]
    assert "Preference elicitation dialogue:" in pair.prompt
    assert "User chose Option" in pair.prompt
    assert "discount factor:" not in pair.prompt
    assert pair.user_type is not None
