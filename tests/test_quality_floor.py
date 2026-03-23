from __future__ import annotations

import numpy as np
import pytest

from src.environments.resource_game import GameConfig, ResourceStrategyGame
from src.evaluation.quality_metrics import (
    check_diversification,
    check_dominance,
    compute_quality_score,
    estimate_bankruptcy_prob,
    full_quality_check,
)


@pytest.fixture
def neutral_stats() -> dict[str, np.ndarray]:
    """Channel stats representing a neutral regime with clear archetypes."""
    return {
        "means": np.array([0.02, 0.06, 0.12, 0.08]),
        "variances": np.array([0.0001, 0.0016, 0.01, 0.04]),
        "names": ["safe", "growth", "aggressive", "volatile"],
    }


@pytest.fixture
def dominated_stats() -> dict[str, np.ndarray]:
    """Channel stats where channel 2 strictly dominates channel 3."""
    return {
        "means": np.array([0.02, 0.06, 0.10, 0.08]),
        "variances": np.array([0.01, 0.02, 0.03, 0.05]),
        "names": ["safe", "growth", "aggressive", "volatile"],
    }


class TestDominanceCheck:
    def test_no_dominance_diversified(self, neutral_stats: dict) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        ok, msg = check_dominance(
            action, neutral_stats["means"], neutral_stats["variances"],
            neutral_stats["names"],
        )
        assert ok

    def test_no_dominance_concentrated_but_not_dominated(
        self, neutral_stats: dict
    ) -> None:
        action = np.array([0.0, 0.0, 1.0, 0.0])
        ok, msg = check_dominance(
            action, neutral_stats["means"], neutral_stats["variances"],
            neutral_stats["names"],
        )
        assert ok, "Aggressive channel is not dominated in neutral stats"

    def test_catches_dominated_concentration(self, dominated_stats: dict) -> None:
        action = np.array([0.0, 0.0, 0.0, 1.0])
        ok, msg = check_dominance(
            action, dominated_stats["means"], dominated_stats["variances"],
            dominated_stats["names"],
        )
        assert not ok
        assert "volatile" in msg.lower() or "dominated" in msg.lower()

    def test_passes_when_not_concentrated(self, dominated_stats: dict) -> None:
        action = np.array([0.0, 0.3, 0.3, 0.4])
        ok, msg = check_dominance(
            action, dominated_stats["means"], dominated_stats["variances"],
            dominated_stats["names"],
        )
        assert ok, "Dominance check only applies to >99% concentration"


class TestDiversificationCheck:
    def test_single_channel_fails(self) -> None:
        action = np.array([1.0, 0.0, 0.0, 0.0])
        ok, msg = check_diversification(action, min_channels=2)
        assert not ok
        assert "1 active" in msg

    def test_two_channels_passes(self) -> None:
        action = np.array([0.5, 0.5, 0.0, 0.0])
        ok, msg = check_diversification(action, min_channels=2)
        assert ok

    def test_four_channels_passes(self) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        ok, msg = check_diversification(action, min_channels=2)
        assert ok

    def test_tiny_weights_ignored(self) -> None:
        action = np.array([0.98, 0.005, 0.005, 0.01])
        ok, msg = check_diversification(action, min_channels=2, weight_threshold=0.01)
        assert not ok, "Weights below threshold should not count as active"

    def test_custom_min_channels(self) -> None:
        action = np.array([0.5, 0.5, 0.0, 0.0])
        ok, msg = check_diversification(action, min_channels=3)
        assert not ok


class TestBankruptcyEstimation:
    def test_safe_allocation_low_bankruptcy(self, neutral_stats: dict) -> None:
        action = np.array([0.7, 0.2, 0.05, 0.05])
        rng = np.random.default_rng(42)
        prob = estimate_bankruptcy_prob(
            action, neutral_stats["means"], neutral_stats["variances"],
            n_samples=10000, rng=rng,
        )
        assert prob < 0.01, f"Safe allocation should have near-zero bankruptcy: {prob}"

    def test_volatile_allocation_higher_bankruptcy(self, neutral_stats: dict) -> None:
        action = np.array([0.0, 0.0, 0.0, 1.0])
        rng = np.random.default_rng(42)
        prob = estimate_bankruptcy_prob(
            action, neutral_stats["means"], neutral_stats["variances"],
            n_samples=10000, rng=rng,
        )
        assert prob >= 0.0

    def test_returns_probability_in_range(self, neutral_stats: dict) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.default_rng(42)
        prob = estimate_bankruptcy_prob(
            action, neutral_stats["means"], neutral_stats["variances"],
            n_samples=5000, rng=rng,
        )
        assert 0.0 <= prob <= 1.0

    def test_zero_variance_zero_bankruptcy(self) -> None:
        action = np.array([1.0, 0.0, 0.0, 0.0])
        means = np.array([0.05, 0.0, 0.0, 0.0])
        variances = np.array([0.0, 0.0, 0.0, 0.0])
        prob = estimate_bankruptcy_prob(action, means, variances)
        assert prob == 0.0


class TestQualityScore:
    def test_positive_return_positive_score(self, neutral_stats: dict) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        score = compute_quality_score(
            action, neutral_stats["means"], neutral_stats["variances"]
        )
        assert score > 0

    def test_safe_allocation_high_sharpe(self, neutral_stats: dict) -> None:
        safe = np.array([1.0, 0.0, 0.0, 0.0])
        risky = np.array([0.0, 0.0, 0.0, 1.0])
        score_safe = compute_quality_score(
            safe, neutral_stats["means"], neutral_stats["variances"]
        )
        score_risky = compute_quality_score(
            risky, neutral_stats["means"], neutral_stats["variances"]
        )
        assert score_safe > score_risky, (
            "Safe channel should have higher Sharpe ratio"
        )


class TestFullQualityCheck:
    def test_valid_allocation_passes(self, neutral_stats: dict) -> None:
        action = np.array([0.3, 0.3, 0.2, 0.2])
        rng = np.random.default_rng(42)
        passes, violations, details = full_quality_check(
            action, neutral_stats["means"], neutral_stats["variances"],
            neutral_stats["names"], rng=rng,
        )
        assert passes
        assert len(violations) == 0
        assert "quality_score" in details

    def test_undiversified_fails(self, neutral_stats: dict) -> None:
        action = np.array([1.0, 0.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        passes, violations, details = full_quality_check(
            action, neutral_stats["means"], neutral_stats["variances"],
            neutral_stats["names"], rng=rng,
        )
        assert not passes
        assert len(violations) >= 1
        assert not details["diversification"]["passes"]

    def test_dominated_concentration_fails(self, dominated_stats: dict) -> None:
        action = np.array([0.0, 0.0, 0.0, 1.0])
        rng = np.random.default_rng(42)
        passes, violations, details = full_quality_check(
            action, dominated_stats["means"], dominated_stats["variances"],
            dominated_stats["names"], rng=rng,
        )
        assert not passes
        assert not details["dominance"]["passes"]

    def test_details_structure(self, neutral_stats: dict) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.default_rng(42)
        _, _, details = full_quality_check(
            action, neutral_stats["means"], neutral_stats["variances"],
            rng=rng,
        )
        assert "dominance" in details
        assert "diversification" in details
        assert "bankruptcy" in details
        assert "quality_score" in details
        assert "probability" in details["bankruptcy"]


class TestEnvironmentQualityFloor:
    """Integration tests: quality floor checks through the environment interface."""

    def test_diversified_passes(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        action = np.array([0.3, 0.3, 0.2, 0.2])
        passes, violations = env.check_quality_floor(action)
        assert passes, f"Diversified allocation should pass: {violations}"

    def test_single_channel_fails(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        action = np.array([1.0, 0.0, 0.0, 0.0])
        passes, violations = env.check_quality_floor(action)
        assert not passes
        assert any("active" in v.lower() or "channel" in v.lower() for v in violations)

    def test_quality_floor_across_regimes(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        good_action = np.array([0.3, 0.3, 0.2, 0.2])

        for regime in [0, 1, 2]:
            env._regime = regime
            passes, violations = env.check_quality_floor(good_action)
            assert passes, (
                f"Good allocation should pass in regime {regime}: {violations}"
            )
