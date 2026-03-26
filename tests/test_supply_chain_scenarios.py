from __future__ import annotations

import numpy as np

from src.environments.supply_chain import SupplyChainConfig, SupplyChainEnv
from src.training.synthetic_users import SyntheticUser, UserType
from src.utils.supply_chain_scenarios import SupplyChainScenarioLibrary


def _env_short():
    return SupplyChainEnv(
        config=SupplyChainConfig(n_periods=30, resilience_mc_samples=300),
    )


def test_gamma_scenarios_target_gamma():
    lib = SupplyChainScenarioLibrary(seed=1)
    env = _env_short()
    scenarios = lib.generate_gamma_scenarios(env, n=4)
    assert len(scenarios) == 4
    for s in scenarios:
        assert s.target_param == "gamma"
        assert s.multiperiod_horizon is not None
        assert abs(float(s.option_a.sum()) - 1.0) < 1e-6


def test_alpha_scenarios_target_alpha():
    lib = SupplyChainScenarioLibrary(seed=2)
    env = _env_short()
    scenarios = lib.generate_alpha_scenarios(env, n=4)
    assert len(scenarios) == 4
    for s in scenarios:
        assert s.target_param == "alpha"


def test_lambda_scenarios_target_lambda():
    lib = SupplyChainScenarioLibrary(seed=3)
    env = _env_short()
    scenarios = lib.generate_lambda_scenarios(env, n=4)
    assert len(scenarios) >= 1
    for s in scenarios:
        assert s.target_param == "lambda_"


def test_gamma_eu_differs_by_gamma():
    lib = SupplyChainScenarioLibrary(seed=4)
    env = _env_short()
    scenarios = lib.generate_gamma_scenarios(env, n=1)
    assert scenarios
    s = scenarios[0]
    h = int(s.multiperiod_horizon)
    high_g = UserType(gamma=0.95, alpha=1.0, lambda_=1.5)
    low_g = UserType(gamma=0.25, alpha=1.0, lambda_=1.5)
    seed = 99
    gap_high = (
        SyntheticUser(high_g, seed=seed).evaluate_for_query(
            s.option_a, s.channel_means, s.channel_variances,
            s.current_wealth, s.rounds_remaining, multiperiod_horizon=h,
        )
        - SyntheticUser(high_g, seed=seed).evaluate_for_query(
            s.option_b, s.channel_means, s.channel_variances,
            s.current_wealth, s.rounds_remaining, multiperiod_horizon=h,
        )
    )
    gap_low = (
        SyntheticUser(low_g, seed=seed).evaluate_for_query(
            s.option_a, s.channel_means, s.channel_variances,
            s.current_wealth, s.rounds_remaining, multiperiod_horizon=h,
        )
        - SyntheticUser(low_g, seed=seed).evaluate_for_query(
            s.option_b, s.channel_means, s.channel_variances,
            s.current_wealth, s.rounds_remaining, multiperiod_horizon=h,
        )
    )
    assert abs(gap_high - gap_low) > 1e-4
