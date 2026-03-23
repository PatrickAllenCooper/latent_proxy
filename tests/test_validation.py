"""Milestone 1 validation tests.

These go beyond unit tests to verify the two core README requirements:
1. Optimal strategies differ meaningfully across user types.
2. Quality floor constraints are well-calibrated.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.environments.resource_game import (
    BEAR,
    BULL,
    NEUTRAL,
    GameConfig,
    ResourceStrategyGame,
)
from src.evaluation.quality_metrics import full_quality_check
from src.training.synthetic_users import (
    SyntheticUser,
    SyntheticUserSampler,
    UserType,
)


class TestStrategyDifferentiation:
    """Validates that the game environment produces meaningfully different
    optimal strategies for different user types, across regimes."""

    @pytest.fixture
    def env(self) -> ResourceStrategyGame:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        return env

    EXTREME_TYPES = {
        "patient_cautious": {"gamma": 0.95, "alpha": 2.0, "lambda_": 2.5},
        "patient_aggressive": {"gamma": 0.95, "alpha": 0.3, "lambda_": 1.1},
        "impatient_cautious": {"gamma": 0.3, "alpha": 2.0, "lambda_": 2.5},
        "impatient_aggressive": {"gamma": 0.3, "alpha": 0.3, "lambda_": 1.1},
    }

    def test_all_pairs_differ(self, env: ResourceStrategyGame) -> None:
        """Every pair of extreme user types produces different optimal actions."""
        actions = {
            name: env.get_optimal_action(theta)
            for name, theta in self.EXTREME_TYPES.items()
        }
        names = list(actions.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = np.linalg.norm(actions[names[i]] - actions[names[j]])
                assert diff > 0.01, (
                    f"{names[i]} vs {names[j]}: L2 distance {diff:.4f} "
                    f"is too small"
                )

    def test_risk_aversion_shifts_toward_safe(
        self, env: ResourceStrategyGame
    ) -> None:
        """Higher alpha should increase allocation to the safe channel."""
        low_alpha = env.get_optimal_action(
            {"gamma": 0.7, "alpha": 0.1, "lambda_": 1.5}
        )
        high_alpha = env.get_optimal_action(
            {"gamma": 0.7, "alpha": 3.0, "lambda_": 1.5}
        )
        assert high_alpha[0] > low_alpha[0], (
            f"Safe channel allocation: alpha=3.0 gives {high_alpha[0]:.3f}, "
            f"alpha=0.1 gives {low_alpha[0]:.3f}"
        )

    def test_strategies_differ_across_regimes(self) -> None:
        """The same user type should get different recommendations in
        different market regimes."""
        env = ResourceStrategyGame()
        env.reset(seed=42)
        theta = {"gamma": 0.7, "alpha": 1.0, "lambda_": 1.5}

        regime_actions = {}
        for regime, name in [(BULL, "bull"), (BEAR, "bear"), (NEUTRAL, "neutral")]:
            env._regime = regime
            regime_actions[name] = env.get_optimal_action(theta)

        bull_bear_diff = np.linalg.norm(
            regime_actions["bull"] - regime_actions["bear"]
        )
        assert bull_bear_diff > 0.01, (
            f"Bull vs bear L2 distance: {bull_bear_diff:.4f}"
        )

    def test_sampled_users_produce_varied_strategies(self) -> None:
        """Randomly sampled users from the prior should produce a range of
        different optimal strategies."""
        env = ResourceStrategyGame()
        env.reset(seed=42)
        sampler = SyntheticUserSampler(seed=0)

        actions = []
        for _ in range(50):
            ut = sampler.sample()
            theta = {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
            actions.append(env.get_optimal_action(theta))

        actions_arr = np.array(actions)
        per_channel_std = actions_arr.std(axis=0)
        assert np.any(per_channel_std > 0.02), (
            f"Channel allocation std across 50 users: {per_channel_std}, "
            f"expected meaningful variation"
        )


class TestQualityFloorCalibration:
    """Validates that quality floor constraints are well-calibrated:
    - Good allocations pass
    - Bad allocations fail
    - Edge cases are handled
    """

    @pytest.fixture
    def env(self) -> ResourceStrategyGame:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        return env

    def test_optimal_actions_pass_quality_floor(self) -> None:
        """Optimal actions for all user types should pass quality floor."""
        env = ResourceStrategyGame()
        env.reset(seed=42)
        sampler = SyntheticUserSampler(seed=0)

        for _ in range(30):
            ut = sampler.sample()
            theta = {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
            action = env.get_optimal_action(theta)
            passes, violations = env.check_quality_floor(action)
            assert passes, (
                f"Optimal action for theta=({ut.gamma:.2f}, {ut.alpha:.2f}, "
                f"{ut.lambda_:.2f}) failed quality floor: {violations}"
            )

    def test_random_valid_allocations_mostly_pass(
        self, env: ResourceStrategyGame
    ) -> None:
        """Uniformly random simplex allocations should mostly pass (they're
        naturally diversified)."""
        rng = np.random.default_rng(42)
        pass_count = 0
        n_trials = 100

        for _ in range(n_trials):
            raw = rng.dirichlet(np.ones(4))
            passes, _ = env.check_quality_floor(raw)
            if passes:
                pass_count += 1

        pass_rate = pass_count / n_trials
        assert pass_rate > 0.7, (
            f"Only {pass_rate:.1%} of random allocations pass quality floor"
        )

    def test_degenerate_allocations_fail(self, env: ResourceStrategyGame) -> None:
        """All single-channel allocations should fail diversification."""
        for i in range(4):
            action = np.zeros(4)
            action[i] = 1.0
            passes, violations = env.check_quality_floor(action)
            assert not passes, f"Single channel {i} should fail"

    def test_quality_floor_stable_across_episodes(self) -> None:
        """Quality floor should give consistent results for the same
        allocation across different episode initializations."""
        good_action = np.array([0.3, 0.3, 0.2, 0.2])
        bad_action = np.array([1.0, 0.0, 0.0, 0.0])

        for seed in range(20):
            env = ResourceStrategyGame()
            env.reset(seed=seed)
            good_passes, _ = env.check_quality_floor(good_action)
            bad_passes, _ = env.check_quality_floor(bad_action)
            assert good_passes, f"Good action failed on seed {seed}"
            assert not bad_passes, f"Bad action passed on seed {seed}"

    def test_full_quality_check_integration(self, env: ResourceStrategyGame) -> None:
        """Full quality check should agree with environment check."""
        stats = env.get_channel_stats()
        names = [ch.name for ch in env.config.channels]
        rng = np.random.default_rng(42)

        action = np.array([0.3, 0.3, 0.2, 0.2])
        env_passes, env_violations = env.check_quality_floor(action)
        full_passes, full_violations, details = full_quality_check(
            action, stats["means"], stats["variances"], names,
            min_channels=env.config.min_channels,
            max_bankruptcy_prob=env.config.max_bankruptcy_prob,
            rng=rng,
        )

        assert env_passes == full_passes
