"""Tests for the DPO study runner using mock LLM conditions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.evaluation.dpo_study import (
    ConditionResult,
    DPOStudyConfig,
    DPOStudyResult,
    _run_analytical_condition,
)
from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.game_variants import create_variant_a


def test_analytical_condition_produces_results():
    conv = ConvergenceConfig(max_rounds=2)
    elic = ElicitationConfig(
        posterior_type="particle",
        n_particles=64,
        max_rounds=2,
        n_scenarios_per_round=6,
        n_eig_samples=24,
        convergence=conv,
        seed=10,
    )
    cr = _run_analytical_condition(create_variant_a, n_users=2, elicitation=elic, max_rounds=2, seed=11)
    assert isinstance(cr, ConditionResult)
    assert cr.condition == "analytical"
    assert len(cr.alignment_scores) == 2
    assert all(np.isfinite(s) for s in cr.alignment_scores)
    assert cr.mean_alignment >= -1.0


def test_condition_result_stats():
    cr = ConditionResult(
        condition="test",
        alignment_scores=[0.8, 0.9, 0.7],
        violation_rates=[0.0, 0.0, 1.0],
        per_round_alignments=[[0.5, 0.7], [0.6, 0.8], [0.4, 0.6]],
    )
    assert abs(cr.mean_alignment - 0.8) < 0.01
    assert abs(cr.mean_violation - 1/3) < 0.01


def test_study_config_defaults():
    cfg = DPOStudyConfig()
    assert cfg.n_users == 5
    assert "game" in cfg.environments
    assert cfg.base_model_path == "Qwen/Qwen2.5-1.5B-Instruct"
