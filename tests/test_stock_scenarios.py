from __future__ import annotations

import numpy as np
import pytest

from src.environments.stock_backtest import StockBacktestEnv
from src.training.synthetic_users import SyntheticUser, UserType
from src.utils.stock_scenarios import StockScenarioLibrary


def _env_short() -> StockBacktestEnv:
    from src.environments.stock_backtest import StockBacktestConfig

    return StockBacktestEnv(
        config=StockBacktestConfig(n_periods=40, drawdown_mc_samples=300),
    )


def test_gamma_scenarios_target_gamma() -> None:
    lib = StockScenarioLibrary(seed=1)
    env = _env_short()
    scenarios = lib.generate_gamma_scenarios(env, n=4)
    assert len(scenarios) == 4
    for s in scenarios:
        assert s.target_param == "gamma"
        assert s.multiperiod_horizon is not None
        assert s.multiperiod_horizon > 1
        assert abs(float(s.option_a.sum()) - 1.0) < 1e-6
        assert abs(float(s.option_b.sum()) - 1.0) < 1e-6


def test_alpha_scenarios_target_alpha() -> None:
    lib = StockScenarioLibrary(seed=2)
    env = _env_short()
    scenarios = lib.generate_alpha_scenarios(env, n=4)
    assert len(scenarios) == 4
    for s in scenarios:
        assert s.target_param == "alpha"


def test_lambda_scenarios_target_lambda() -> None:
    lib = StockScenarioLibrary(seed=3)
    env = _env_short()
    scenarios = lib.generate_lambda_scenarios(env, n=4)
    assert len(scenarios) >= 1
    for s in scenarios:
        assert s.target_param == "lambda_"


def test_stock_gamma_multiperiod_eu_differs_by_gamma() -> None:
    lib = StockScenarioLibrary(seed=4)
    env = _env_short()
    scenarios = lib.generate_gamma_scenarios(env, n=1)
    assert scenarios
    s = scenarios[0]
    assert s.multiperiod_horizon is not None
    h = int(s.multiperiod_horizon)

    high_g = UserType(gamma=0.95, alpha=1.0, lambda_=1.5)
    low_g = UserType(gamma=0.25, alpha=1.0, lambda_=1.5)
    seed = 99
    u_high_a = SyntheticUser(high_g, seed=seed).evaluate_for_query(
        s.option_a,
        s.channel_means,
        s.channel_variances,
        s.current_wealth,
        s.rounds_remaining,
        multiperiod_horizon=h,
    )
    u_high_b = SyntheticUser(high_g, seed=seed).evaluate_for_query(
        s.option_b,
        s.channel_means,
        s.channel_variances,
        s.current_wealth,
        s.rounds_remaining,
        multiperiod_horizon=h,
    )
    u_low_a = SyntheticUser(low_g, seed=seed).evaluate_for_query(
        s.option_a,
        s.channel_means,
        s.channel_variances,
        s.current_wealth,
        s.rounds_remaining,
        multiperiod_horizon=h,
    )
    u_low_b = SyntheticUser(low_g, seed=seed).evaluate_for_query(
        s.option_b,
        s.channel_means,
        s.channel_variances,
        s.current_wealth,
        s.rounds_remaining,
        multiperiod_horizon=h,
    )
    gap_high = u_high_a - u_high_b
    gap_low = u_low_a - u_low_b
    assert abs(gap_high - gap_low) > 1e-4


def test_alpha_scenarios_preference_direction_high_alpha() -> None:
    lib = StockScenarioLibrary(seed=5)
    env = _env_short()
    scenarios = lib.generate_alpha_scenarios(env, n=4)
    cautious = UserType(gamma=0.7, alpha=3.5, lambda_=2.2)
    wins = 0
    for s in scenarios:
        user = SyntheticUser(cautious, seed=7)
        eu_a = user.evaluate_for_query(
            s.option_a,
            s.channel_means,
            s.channel_variances,
            s.current_wealth,
            s.rounds_remaining,
        )
        eu_b = user.evaluate_for_query(
            s.option_b,
            s.channel_means,
            s.channel_variances,
            s.current_wealth,
            s.rounds_remaining,
        )
        if eu_a > eu_b:
            wins += 1
    assert wins >= len(scenarios) // 2
