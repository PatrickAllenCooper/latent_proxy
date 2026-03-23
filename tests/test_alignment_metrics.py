from __future__ import annotations

import numpy as np
import pytest

from src.environments.resource_game import ResourceStrategyGame
from src.evaluation.alignment_metrics import (
    compute_alignment_score,
    compute_preference_recovery_error,
    compute_quality_floor_violation_rate,
    evaluate_actions,
    evaluate_model_outputs,
)
from src.training.synthetic_users import UserType


class TestAlignmentScore:
    def test_perfect_alignment(self) -> None:
        actions = [np.array([0.3, 0.3, 0.2, 0.2])] * 5
        optimal = [np.array([0.3, 0.3, 0.2, 0.2])] * 5
        score = compute_alignment_score(actions, optimal)
        assert score == 1.0

    def test_random_low_alignment(self) -> None:
        rng = np.random.default_rng(42)
        actions = [rng.dirichlet(np.ones(4)) for _ in range(50)]
        optimal = [rng.dirichlet(np.ones(4)) for _ in range(50)]
        score = compute_alignment_score(actions, optimal)
        assert -0.5 < score < 0.8

    def test_opposite_ranking(self) -> None:
        actions = [np.array([0.5, 0.3, 0.15, 0.05])]
        optimal = [np.array([0.05, 0.15, 0.3, 0.5])]
        score = compute_alignment_score(actions, optimal)
        assert score < 0

    def test_empty_returns_zero(self) -> None:
        assert compute_alignment_score([], []) == 0.0

    def test_constant_actions(self) -> None:
        actions = [np.array([0.25, 0.25, 0.25, 0.25])]
        optimal = [np.array([0.4, 0.3, 0.2, 0.1])]
        score = compute_alignment_score(actions, optimal)
        assert isinstance(score, float)


class TestQualityFloorViolationRate:
    def test_good_actions_zero_violations(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        good_actions = [np.array([0.3, 0.3, 0.2, 0.2])] * 10
        rate = compute_quality_floor_violation_rate(good_actions, env)
        assert rate == 0.0

    def test_bad_actions_nonzero_violations(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        bad_actions = [np.array([1.0, 0.0, 0.0, 0.0])] * 10
        rate = compute_quality_floor_violation_rate(bad_actions, env)
        assert rate > 0.0

    def test_mixed_actions(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        mixed = [
            np.array([0.3, 0.3, 0.2, 0.2]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        ]
        rate = compute_quality_floor_violation_rate(mixed, env)
        assert rate == 0.5

    def test_empty_returns_zero(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        rate = compute_quality_floor_violation_rate([], env)
        assert rate == 0.0


class TestPreferenceRecoveryError:
    def test_perfect_recovery(self) -> None:
        ut = UserType(gamma=0.8, alpha=1.0, lambda_=2.0)
        inferred = {"gamma": 0.8, "alpha": 1.0, "lambda_": 2.0}
        errors = compute_preference_recovery_error(inferred, ut)
        assert errors["gamma"] == 0.0
        assert errors["alpha"] == 0.0
        assert errors["lambda_"] == 0.0
        assert errors["total"] == 0.0

    def test_imperfect_recovery(self) -> None:
        ut = UserType(gamma=0.8, alpha=1.0, lambda_=2.0)
        inferred = {"gamma": 0.7, "alpha": 1.2, "lambda_": 2.5}
        errors = compute_preference_recovery_error(inferred, ut)
        np.testing.assert_allclose(errors["gamma"], 0.1)
        np.testing.assert_allclose(errors["alpha"], 0.2)
        np.testing.assert_allclose(errors["lambda_"], 0.5)
        assert errors["total"] > 0

    def test_missing_params_default_to_zero(self) -> None:
        ut = UserType(gamma=0.8, alpha=1.0, lambda_=2.0)
        errors = compute_preference_recovery_error({}, ut)
        assert errors["gamma"] == 0.8
        assert errors["alpha"] == 1.0
        assert errors["lambda_"] == 2.0


class TestEvaluateActions:
    def test_optimal_actions_high_alignment(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        user_types = [
            UserType(gamma=0.9, alpha=1.0, lambda_=1.5),
            UserType(gamma=0.5, alpha=0.5, lambda_=1.2),
        ]
        optimal_actions = [
            env.get_optimal_action(
                {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
            )
            for ut in user_types
        ]
        results = evaluate_actions(optimal_actions, env, user_types)
        assert results["alignment_score"] == 1.0
        assert results["quality_floor_violation_rate"] == 0.0
        assert results["n_episodes"] == 2

    def test_results_structure(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        user_types = [UserType(gamma=0.7, alpha=1.0, lambda_=1.5)]
        actions = [np.array([0.3, 0.3, 0.2, 0.2])]
        results = evaluate_actions(actions, env, user_types)
        assert "alignment_score" in results
        assert "quality_floor_violation_rate" in results
        assert "mean_quality_score" in results
        assert "n_episodes" in results


class TestEvaluateModelOutputs:
    def test_parses_and_evaluates(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        user_types = [UserType(gamma=0.7, alpha=1.0, lambda_=1.5)]
        responses = ["Safe: 30%\nGrowth: 30%\nAggressive: 20%\nVolatile: 20%"]
        results = evaluate_model_outputs(responses, env, user_types)
        assert "alignment_score" in results
        assert results["quality_floor_violation_rate"] == 0.0

    def test_garbage_response_still_evaluates(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        user_types = [UserType(gamma=0.7, alpha=1.0, lambda_=1.5)]
        responses = ["I do not know what to recommend."]
        results = evaluate_model_outputs(responses, env, user_types)
        assert isinstance(results["alignment_score"], float)
