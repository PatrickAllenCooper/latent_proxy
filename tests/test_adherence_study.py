"""Tests for the adherence study (elicited/true-profile x condition), offline only.

Uses a fixed-response fake generator instead of a real model, so these never
touch a GPU or the network -- mirrors tests/test_dpo_study.py's pattern of
exercising the non-LLM plumbing directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.llm_elicitation import LLMElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.game_variants import create_variant_a
from src.evaluation.adherence_study import (
    AdherenceConditionResult,
    AdherenceStudyConfig,
    _append_format_instruction,
    _run_adherence_condition,
    _theta_from_inferred,
    compare_conditions,
    run_adherence_study,
)
from src.training.serialization import AllocationSerializer
from src.training.synthetic_users import UserType

CHANNEL_NAMES = ["safe", "growth", "aggressive", "volatile"]


class _FixedTextGenerator:
    """Always returns the same allocation text, regardless of prompt."""

    def __init__(self, allocation: list[float]) -> None:
        self._text = AllocationSerializer(CHANNEL_NAMES).serialize(np.asarray(allocation))
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._text


# ---------------------------------------------------------------------------
# _theta_from_inferred
# ---------------------------------------------------------------------------


def test_theta_from_inferred_within_range_passes_through():
    ut = _theta_from_inferred({"gamma": 0.6, "alpha": 1.2, "lambda_": 1.8})
    assert ut == UserType(gamma=0.6, alpha=1.2, lambda_=1.8)


def test_theta_from_inferred_clips_out_of_range_values():
    # gamma > 1, alpha < 0, lambda_ < 1: none of these are reachable from a
    # real posterior mean, but the conversion must not crash on them.
    ut = _theta_from_inferred({"gamma": 1.5, "alpha": -2.0, "lambda_": 0.2})
    assert 0 < ut.gamma <= 1.0
    assert ut.alpha >= 0.0
    assert ut.lambda_ >= 1.0


def test_theta_from_inferred_defaults_on_missing_keys():
    ut = _theta_from_inferred({})
    assert 0 < ut.gamma <= 1.0
    assert ut.alpha >= 0.0
    assert ut.lambda_ >= 1.0


# ---------------------------------------------------------------------------
# _run_adherence_condition
# ---------------------------------------------------------------------------


def test_true_mode_rejects_bad_theta_mode():
    with pytest.raises(ValueError, match="Unknown theta_mode"):
        _run_adherence_condition(
            create_variant_a, None, None, "base", "bogus", 1,
            LLMElicitationConfig(), ElicitationConfig(), seed=1,
            generator=_FixedTextGenerator([0.4, 0.3, 0.2, 0.1]),
        )


def test_true_mode_produces_one_score_per_user():
    gen = _FixedTextGenerator([0.4, 0.3, 0.2, 0.1])
    result = _run_adherence_condition(
        create_variant_a, None, None, "base", "true", 3,
        LLMElicitationConfig(), ElicitationConfig(), seed=1, generator=gen,
    )
    assert isinstance(result, AdherenceConditionResult)
    assert result.condition == "base"
    assert result.theta_mode == "true"
    assert len(result.alignment_scores) == 3
    assert len(result.violation_rates) == 3
    assert all(np.isfinite(s) for s in result.alignment_scores)
    assert result.mean_alignment == pytest.approx(float(np.mean(result.alignment_scores)))
    assert len(gen.prompts) == 3
    # true mode: the prompt must render the user's actual numeric profile.
    assert "preference profile" in gen.prompts[0]


def test_violation_logs_diagnostic_reasons(caplog):
    """Regression test for the elicited-mode investigation: env.check_quality_floor
    already returns human-readable failure reasons, but the scoring path
    (compute_quality_floor_violation_rate) only keeps the boolean. This
    verifies the diagnostic warning surfaces the actual reasons instead of
    just a violation count, using an undiversified (single-channel) response
    that's known to fail the diversification check.
    """
    gen = _FixedTextGenerator([1.0, 0.0, 0.0, 0.0])
    with caplog.at_level("WARNING"):
        result = _run_adherence_condition(
            create_variant_a, None, None, "dpo_phase2", "true", 1,
            LLMElicitationConfig(), ElicitationConfig(), seed=1, generator=gen,
        )
    assert result.mean_violation == pytest.approx(1.0)
    warnings = [r for r in caplog.records if "QUALITY FLOOR VIOLATION" in r.message]
    assert len(warnings) == 1
    assert "true_theta=" in warnings[0].message
    assert "profile_theta=" in warnings[0].message
    assert "allocation=" in warnings[0].message


def test_append_format_instruction_gives_an_explicit_template():
    """Regression test: without this, raw generations from an Instruct model
    are free-form chain-of-thought reasoning that never reaches parseable
    numbers, confirmed by direct inspection of real generations from the
    trained Phase 2 checkpoint (every response silently fell back to a
    uniform allocation, i.e. alignment_score exactly 0.0 for every user).
    """
    prompt = _append_format_instruction("Some game state.", CHANNEL_NAMES)
    assert "Some game state." in prompt
    assert "Format your response EXACTLY as" in prompt
    assert "Recommended allocation:" in prompt
    for name in CHANNEL_NAMES:
        assert f"{name}: __%" in prompt


def test_prompt_sent_to_generator_includes_format_instruction():
    gen = _FixedTextGenerator([0.4, 0.3, 0.2, 0.1])
    _run_adherence_condition(
        create_variant_a, None, None, "base", "true", 1,
        LLMElicitationConfig(), ElicitationConfig(), seed=1, generator=gen,
    )
    assert "Format your response EXACTLY as" in gen.prompts[0]


def test_parse_failure_rate_tracks_unparseable_responses():
    class _AlternatingGenerator:
        def __init__(self) -> None:
            self.n_calls = 0

        def generate(self, prompt: str) -> str:
            self.n_calls += 1
            if self.n_calls % 2 == 1:
                return "Let me think step by step about this allocation problem..."
            return AllocationSerializer(CHANNEL_NAMES).serialize(np.array([0.4, 0.3, 0.2, 0.1]))

    result = _run_adherence_condition(
        create_variant_a, None, None, "base", "true", 4,
        LLMElicitationConfig(), ElicitationConfig(), seed=1, generator=_AlternatingGenerator(),
    )
    assert result.parse_failure_rate == pytest.approx(0.5)
    assert all(np.isfinite(s) for s in result.alignment_scores)


def test_elicited_mode_runs_full_elicitation_loop():
    """Small but real particle-filter + EIG loop, not mocked."""
    conv = ConvergenceConfig(max_rounds=2)
    elic = ElicitationConfig(
        posterior_type="particle", n_particles=32, max_rounds=2,
        n_scenarios_per_round=6, n_eig_samples=16, convergence=conv, seed=3,
    )
    gen = _FixedTextGenerator([0.25, 0.25, 0.25, 0.25])
    result = _run_adherence_condition(
        create_variant_a, None, None, "dpo_phase2", "elicited", 2,
        LLMElicitationConfig(), elic, seed=3, generator=gen,
    )
    assert result.theta_mode == "elicited"
    assert len(result.alignment_scores) == 2
    assert all(np.isfinite(s) for s in result.alignment_scores)
    assert "preference profile" in gen.prompts[0]


# ---------------------------------------------------------------------------
# AdherenceStudyConfig / run_adherence_study validation
# ---------------------------------------------------------------------------


def test_adherence_study_config_defaults():
    cfg = AdherenceStudyConfig()
    assert cfg.n_users == 20
    assert cfg.environment == "game"
    assert cfg.base_model_path == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.conditions == ["base", "dpo_phase1", "dpo_phase2"]


def test_run_adherence_study_rejects_missing_checkpoint():
    cfg = AdherenceStudyConfig(conditions=["dpo_phase1"], phase1_checkpoint=None)
    with pytest.raises(ValueError, match="requires a checkpoint"):
        run_adherence_study(cfg)


def test_run_adherence_study_rejects_unknown_condition():
    cfg = AdherenceStudyConfig(conditions=["not_a_real_condition"])
    with pytest.raises(ValueError, match="Unknown condition"):
        run_adherence_study(cfg)


def test_run_adherence_study_rejects_unknown_environment():
    cfg = AdherenceStudyConfig(environment="not_a_real_env", conditions=["base"])
    with pytest.raises(ValueError, match="Unknown environment"):
        run_adherence_study(cfg)


# ---------------------------------------------------------------------------
# compare_conditions
# ---------------------------------------------------------------------------


def test_compare_conditions_builds_expected_label_and_direction():
    results = {
        "dpo_phase2": {
            "true": AdherenceConditionResult(
                condition="dpo_phase2", theta_mode="true",
                alignment_scores=[0.9, 0.85, 0.95, 0.8, 0.9],
            ),
        },
        "base": {
            "true": AdherenceConditionResult(
                condition="base", theta_mode="true",
                alignment_scores=[0.1, 0.2, 0.0, 0.15, 0.1],
            ),
        },
    }
    t = compare_conditions(results, "dpo_phase2", "base", "true")
    assert t.hypothesis == "dpo_phase2_vs_base_true"
    assert np.isfinite(t.p_value)
    assert np.isfinite(t.effect_size)
