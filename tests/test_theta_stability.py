from __future__ import annotations

import numpy as np

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.supply_chain import SupplyChainConfig, SupplyChainEnv
from src.environments.stock_backtest import StockBacktestConfig, StockBacktestEnv
from src.environments.game_variants import create_variant_a
from src.evaluation.theta_stability import run_theta_stability_test


def _fast_elicitation():
    return ElicitationConfig(
        posterior_type="particle",
        n_particles=64,
        max_rounds=2,
        n_scenarios_per_round=6,
        n_eig_samples=24,
        convergence=ConvergenceConfig(max_rounds=2),
        seed=100,
    )


def test_stability_game():
    elic = _fast_elicitation()
    res = run_theta_stability_test(
        create_variant_a, n_users=3, n_sessions=2, elicitation=elic, seed=101,
    )
    assert res.n_users == 3
    assert res.n_sessions == 2
    for p in ["gamma", "alpha", "lambda_"]:
        assert p in res.icc_per_param
        assert np.isfinite(res.icc_per_param[p])
        assert p in res.pearson_per_param
        assert np.isfinite(res.pearson_per_param[p])


def test_stability_stock():
    elic = _fast_elicitation()
    factory = lambda: StockBacktestEnv(
        config=StockBacktestConfig(n_periods=12, drawdown_mc_samples=200),
    )
    res = run_theta_stability_test(
        factory, n_users=3, n_sessions=2, elicitation=elic, seed=102,
    )
    assert res.n_users == 3
    for p in ["gamma", "alpha", "lambda_"]:
        assert np.isfinite(res.icc_per_param[p])


def test_stability_supply_chain():
    elic = _fast_elicitation()
    factory = lambda: SupplyChainEnv(
        config=SupplyChainConfig(n_periods=10, resilience_mc_samples=200),
    )
    res = run_theta_stability_test(
        factory, n_users=3, n_sessions=2, elicitation=elic, seed=103,
    )
    assert res.n_users == 3
    for p in ["gamma", "alpha", "lambda_"]:
        assert np.isfinite(res.icc_per_param[p])


def test_stability_icc_nonneg_on_correlated():
    elic = _fast_elicitation()
    res = run_theta_stability_test(
        create_variant_a, n_users=4, n_sessions=2, elicitation=elic, seed=104,
    )
    for p in ["gamma", "alpha", "lambda_"]:
        assert res.icc_per_param[p] >= -1.0
