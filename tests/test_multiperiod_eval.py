from __future__ import annotations

import numpy as np
import pytest

from src.training.synthetic_users import SyntheticUser, UserType
from src.utils.diagnostic_scenarios import DiagnosticScenario
from src.utils.information_gain import compute_eig_mc
from src.utils.posterior import ParticlePosterior


def test_multiperiod_differs_from_single_period_discount_path() -> None:
    """Multi-period scoring sums discounted per-period utilities; single-period differs."""
    ut = UserType(gamma=0.85, alpha=1.0, lambda_=1.5)
    user = SyntheticUser(ut, seed=123)
    means = np.array([0.02, 0.06, 0.12, 0.08], dtype=np.float64)
    variances = np.array([0.0001, 0.0016, 0.01, 0.04], dtype=np.float64)
    wealth = 1000.0
    safe = np.array([0.7, 0.3, 0.0, 0.0])
    risky = np.array([0.0, 0.0, 0.6, 0.4])

    mp = user.evaluate_allocation_multiperiod(
        safe, means, variances, wealth, n_periods=12, n_samples=400,
    )
    sp = user.evaluate_allocation(
        safe, means, variances, wealth, rounds_remaining=12,
    )
    assert mp != pytest.approx(sp, abs=1e-9)


def test_gamma_changes_safe_vs_risky_tradeoff_multiperiod() -> None:
    """Discounting the path makes the safe-vs-risky contrast depend on gamma."""
    alpha, lam = 1.0, 1.5
    high_g = UserType(gamma=0.95, alpha=alpha, lambda_=lam)
    low_g = UserType(gamma=0.25, alpha=alpha, lambda_=lam)

    means = np.array([0.02, 0.06, 0.12, 0.08], dtype=np.float64)
    variances = np.array([0.0001, 0.0016, 0.01, 0.04], dtype=np.float64)
    wealth = 1000.0
    safe = np.array([0.7, 0.3, 0.0, 0.0])
    risky = np.array([0.0, 0.0, 0.6, 0.4])

    seed = 7
    u_high_safe = SyntheticUser(high_g, seed=seed).evaluate_allocation_multiperiod(
        safe, means, variances, wealth, n_periods=15, n_samples=400,
    )
    u_high_risky = SyntheticUser(high_g, seed=seed).evaluate_allocation_multiperiod(
        risky, means, variances, wealth, n_periods=15, n_samples=400,
    )
    u_low_safe = SyntheticUser(low_g, seed=seed).evaluate_allocation_multiperiod(
        safe, means, variances, wealth, n_periods=15, n_samples=400,
    )
    u_low_risky = SyntheticUser(low_g, seed=seed).evaluate_allocation_multiperiod(
        risky, means, variances, wealth, n_periods=15, n_samples=400,
    )

    gap_high = u_high_safe - u_high_risky
    gap_low = u_low_safe - u_low_risky
    assert abs(gap_high - gap_low) > 1e-3, (
        "Multi-period utility contrast should vary with gamma (identifiability)"
    )


def test_gamma_scenario_multiperiod_eig_positive() -> None:
    """EIG under multi-period gamma scenario should be positive with a spread posterior."""
    post = ParticlePosterior(n_particles=400, ess_threshold_ratio=0.2)
    means = np.array([0.02, 0.06, 0.12, 0.08], dtype=np.float64)
    variances = np.array([0.0001, 0.0016, 0.01, 0.04], dtype=np.float64)
    scenario_mp = DiagnosticScenario(
        game_state={},
        option_a=np.array([0.7, 0.3, 0.0, 0.0]),
        option_b=np.array([0.0, 0.0, 0.6, 0.4]),
        target_param="gamma",
        description="test",
        channel_means=means,
        channel_variances=variances,
        current_wealth=1000.0,
        rounds_remaining=15,
        multiperiod_horizon=12,
    )
    scenario_sp = DiagnosticScenario(
        game_state={},
        option_a=scenario_mp.option_a,
        option_b=scenario_mp.option_b,
        target_param="gamma",
        description="test",
        channel_means=means,
        channel_variances=variances,
        current_wealth=1000.0,
        rounds_remaining=15,
        multiperiod_horizon=None,
    )
    rng = np.random.default_rng(0)
    eig_mp = compute_eig_mc(
        scenario_mp, post, n_samples=200, temperature=0.1, rng=rng,
    )
    rng2 = np.random.default_rng(0)
    eig_sp = compute_eig_mc(
        scenario_sp, post, n_samples=200, temperature=0.1, rng=rng2,
    )
    assert eig_mp >= 0.0
    assert eig_sp >= 0.0
    assert np.isfinite(eig_mp) and np.isfinite(eig_sp)
