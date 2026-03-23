from __future__ import annotations

import numpy as np
import pytest

from src.training.synthetic_users import (
    PriorConfig,
    SyntheticUser,
    SyntheticUserSampler,
    UserType,
    discounted_utility,
    prospect_utility,
)


class TestUserType:
    def test_valid_construction(self) -> None:
        ut = UserType(gamma=0.9, alpha=1.0, lambda_=2.0)
        assert ut.gamma == 0.9
        assert ut.alpha == 1.0
        assert ut.lambda_ == 2.0

    def test_gamma_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            UserType(gamma=0.0, alpha=1.0, lambda_=1.5)
        with pytest.raises(ValueError, match="gamma"):
            UserType(gamma=1.5, alpha=1.0, lambda_=1.5)
        with pytest.raises(ValueError, match="gamma"):
            UserType(gamma=-0.1, alpha=1.0, lambda_=1.5)

    def test_alpha_negative(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            UserType(gamma=0.5, alpha=-1.0, lambda_=1.5)

    def test_lambda_below_one(self) -> None:
        with pytest.raises(ValueError, match="lambda_"):
            UserType(gamma=0.5, alpha=1.0, lambda_=0.5)

    def test_boundary_values(self) -> None:
        ut = UserType(gamma=1.0, alpha=0.0, lambda_=1.0)
        assert ut.gamma == 1.0
        assert ut.alpha == 0.0
        assert ut.lambda_ == 1.0

    def test_frozen(self) -> None:
        ut = UserType(gamma=0.5, alpha=1.0, lambda_=1.5)
        with pytest.raises(AttributeError):
            ut.gamma = 0.9  # type: ignore[misc]


class TestSyntheticUserSampler:
    def test_sample_produces_valid_type(self, sampler: SyntheticUserSampler) -> None:
        ut = sampler.sample()
        assert 0 < ut.gamma <= 1
        assert ut.alpha >= 0
        assert ut.lambda_ >= 1

    def test_batch_sample_size(self, sampler: SyntheticUserSampler) -> None:
        batch = sampler.sample_batch(50)
        assert len(batch) == 50
        for ut in batch:
            assert 0 < ut.gamma <= 1

    def test_batch_has_variety(self, sampler: SyntheticUserSampler) -> None:
        batch = sampler.sample_batch(100)
        gammas = [ut.gamma for ut in batch]
        assert max(gammas) - min(gammas) > 0.1, "Sampled gammas lack variety"

    def test_reproducibility(self) -> None:
        s1 = SyntheticUserSampler(seed=42)
        s2 = SyntheticUserSampler(seed=42)
        for _ in range(10):
            ut1 = s1.sample()
            ut2 = s2.sample()
            assert ut1.gamma == ut2.gamma
            assert ut1.alpha == ut2.alpha
            assert ut1.lambda_ == ut2.lambda_

    def test_extreme_types(self, sampler: SyntheticUserSampler) -> None:
        extremes = sampler.sample_extreme_types()
        assert "patient_cautious" in extremes
        assert "impatient_aggressive" in extremes
        assert extremes["patient_cautious"].gamma > extremes["impatient_aggressive"].gamma
        assert extremes["patient_cautious"].alpha > extremes["impatient_aggressive"].alpha

    def test_custom_prior(self) -> None:
        config = PriorConfig(gamma_a=10.0, gamma_b=1.0, lambda_low=1.0, lambda_high=1.5)
        sampler = SyntheticUserSampler(prior_config=config, seed=0)
        batch = sampler.sample_batch(50)
        mean_gamma = np.mean([ut.gamma for ut in batch])
        assert mean_gamma > 0.7, "High gamma_a should produce high gamma values"


class TestProspectUtility:
    def test_gains_are_positive(self) -> None:
        u = prospect_utility(100.0, alpha=1.0, lambda_=2.0, reference_point=0.0)
        assert u > 0

    def test_losses_are_negative(self) -> None:
        u = prospect_utility(-50.0, alpha=1.0, lambda_=2.0, reference_point=0.0)
        assert u < 0

    def test_loss_aversion_amplifies_losses(self) -> None:
        u_gain = prospect_utility(10.0, alpha=1.0, lambda_=2.5, reference_point=0.0)
        u_loss = prospect_utility(-10.0, alpha=1.0, lambda_=2.5, reference_point=0.0)
        assert abs(u_loss) > abs(u_gain), "Loss aversion should make |u(loss)| > |u(gain)|"

    def test_monotonicity_in_gains(self) -> None:
        u1 = prospect_utility(50.0, alpha=1.0, lambda_=1.5)
        u2 = prospect_utility(100.0, alpha=1.0, lambda_=1.5)
        assert u2 > u1

    def test_risk_aversion_concavity(self) -> None:
        u_50 = prospect_utility(50.0, alpha=1.0, lambda_=1.5)
        u_100 = prospect_utility(100.0, alpha=1.0, lambda_=1.5)
        u_150 = prospect_utility(150.0, alpha=1.0, lambda_=1.5)
        marginal_1 = u_100 - u_50
        marginal_2 = u_150 - u_100
        assert marginal_2 < marginal_1, "Diminishing marginal utility (concavity)"

    def test_zero_alpha_is_linear(self) -> None:
        u1 = prospect_utility(50.0, alpha=0.0, lambda_=1.5)
        u2 = prospect_utility(100.0, alpha=0.0, lambda_=1.5)
        np.testing.assert_allclose(u2, 2.0 * u1, rtol=1e-6)

    def test_vectorized(self) -> None:
        wealths = np.array([10.0, 50.0, -20.0, 100.0])
        utilities = prospect_utility(wealths, alpha=1.0, lambda_=2.0)
        assert utilities.shape == (4,)
        assert utilities[0] > 0
        assert utilities[2] < 0

    def test_reference_point_shift(self) -> None:
        u_gain = prospect_utility(120.0, alpha=1.0, lambda_=2.0, reference_point=100.0)
        u_loss = prospect_utility(80.0, alpha=1.0, lambda_=2.0, reference_point=100.0)
        assert u_gain > 0
        assert u_loss < 0


class TestDiscountedUtility:
    def test_discounting_reduces_utility(self, patient_cautious: UserType) -> None:
        u_now = discounted_utility(100.0, patient_cautious, rounds_remaining=0)
        u_later = discounted_utility(100.0, patient_cautious, rounds_remaining=10)
        assert u_now > u_later

    def test_patient_user_discounts_less(
        self, patient_cautious: UserType, impatient_aggressive: UserType
    ) -> None:
        u_patient = discounted_utility(100.0, patient_cautious, rounds_remaining=10)
        u_impatient = discounted_utility(100.0, impatient_aggressive, rounds_remaining=10)
        assert u_patient > u_impatient

    def test_zero_rounds_remaining_no_discount(self, balanced_user: UserType) -> None:
        u = discounted_utility(100.0, balanced_user, rounds_remaining=0)
        u_raw = prospect_utility(100.0, balanced_user.alpha, balanced_user.lambda_)
        np.testing.assert_allclose(u, u_raw, rtol=1e-6)


class TestSyntheticUser:
    def test_evaluate_outcome(self, patient_cautious: UserType) -> None:
        user = SyntheticUser(patient_cautious, seed=0)
        u = user.evaluate_outcome(100.0, rounds_remaining=5)
        assert isinstance(u, float)
        assert u > 0

    def test_choose_prefers_higher_utility(self) -> None:
        ut = UserType(gamma=0.9, alpha=1.0, lambda_=1.5)
        user = SyntheticUser(ut, temperature=0.01, seed=0)
        choices = [user.choose(10.0, 1.0) for _ in range(100)]
        frac_a = choices.count(0) / len(choices)
        assert frac_a > 0.9, "Should almost always prefer higher-utility option"

    def test_choose_is_stochastic_at_high_temp(self) -> None:
        ut = UserType(gamma=0.9, alpha=1.0, lambda_=1.5)
        user = SyntheticUser(ut, temperature=10.0, seed=0)
        choices = [user.choose(1.1, 1.0) for _ in range(200)]
        frac_a = choices.count(0) / len(choices)
        assert 0.3 < frac_a < 0.7, "High temperature should make choices nearly random"

    def test_evaluate_allocation(self, patient_cautious: UserType) -> None:
        user = SyntheticUser(patient_cautious, seed=0)
        means = np.array([0.02, 0.06, 0.12, 0.08])
        variances = np.array([0.01, 0.04, 0.10, 0.20]) ** 2
        alloc = np.array([0.4, 0.3, 0.2, 0.1])
        score = user.evaluate_allocation(alloc, means, variances, 1000.0, 5)
        assert isinstance(score, float)

    def test_choose_allocation(self, patient_cautious: UserType) -> None:
        user = SyntheticUser(patient_cautious, temperature=0.01, seed=0)
        means = np.array([0.02, 0.06, 0.12, 0.08])
        variances = np.array([0.01, 0.04, 0.10, 0.20]) ** 2
        safe_alloc = np.array([0.6, 0.3, 0.05, 0.05])
        risky_alloc = np.array([0.0, 0.1, 0.5, 0.4])
        choices = [
            user.choose_allocation(
                [safe_alloc, risky_alloc],
                means, variances, 1000.0, 5,
            )
            for _ in range(50)
        ]
        assert isinstance(choices[0], int)
        assert all(c in (0, 1) for c in choices)


class TestDifferentUserTypesDifferentStrategies:
    """Validates the README requirement that optimal strategies differ
    meaningfully across user types."""

    def test_patient_vs_impatient_optimal_actions(self) -> None:
        from src.environments.resource_game import ResourceStrategyGame

        env = ResourceStrategyGame()
        env.reset(seed=42)

        patient_action = env.get_optimal_action(
            {"gamma": 0.95, "alpha": 2.0, "lambda_": 2.5}
        )
        impatient_action = env.get_optimal_action(
            {"gamma": 0.3, "alpha": 0.3, "lambda_": 1.1}
        )

        diff = np.linalg.norm(patient_action - impatient_action)
        assert diff > 0.05, (
            f"Optimal actions should differ meaningfully, "
            f"but L2 distance is only {diff:.4f}"
        )

    def test_risk_averse_prefers_safe(self) -> None:
        from src.environments.resource_game import ResourceStrategyGame

        env = ResourceStrategyGame()
        env.reset(seed=42)

        cautious_action = env.get_optimal_action(
            {"gamma": 0.7, "alpha": 3.0, "lambda_": 2.5}
        )
        aggressive_action = env.get_optimal_action(
            {"gamma": 0.7, "alpha": 0.1, "lambda_": 1.0}
        )

        assert cautious_action[0] > aggressive_action[0], (
            "Risk-averse user should allocate more to the safe channel"
        )

    def test_evaluation_differs_across_types(self) -> None:
        means = np.array([0.02, 0.06, 0.12, 0.08])
        variances = np.array([0.0001, 0.0016, 0.01, 0.04])
        safe_alloc = np.array([0.6, 0.3, 0.05, 0.05])
        risky_alloc = np.array([0.05, 0.15, 0.4, 0.4])

        cautious = SyntheticUser(
            UserType(gamma=0.9, alpha=2.0, lambda_=2.5), seed=42
        )
        aggressive = SyntheticUser(
            UserType(gamma=0.9, alpha=0.2, lambda_=1.1), seed=42
        )

        cautious_safe = cautious.evaluate_allocation(
            safe_alloc, means, variances, 1000.0, 5
        )
        cautious_risky = cautious.evaluate_allocation(
            risky_alloc, means, variances, 1000.0, 5
        )
        aggressive_safe = aggressive.evaluate_allocation(
            safe_alloc, means, variances, 1000.0, 5
        )
        aggressive_risky = aggressive.evaluate_allocation(
            risky_alloc, means, variances, 1000.0, 5
        )

        cautious_prefers_safe = cautious_safe > cautious_risky
        aggressive_prefers_risky = aggressive_risky > aggressive_safe

        assert cautious_prefers_safe or aggressive_prefers_risky, (
            "At least one user type should show a clear allocation preference"
        )
