from __future__ import annotations

import numpy as np

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.game_variants import create_variant_a
from src.environments.stock_backtest import StockBacktestConfig
from src.evaluation.experiment_runner import run_cross_domain_transfer


def test_cross_domain_three_conditions_with_ci() -> None:
    conv = ConvergenceConfig(max_rounds=2)
    elic = ElicitationConfig(
        posterior_type="particle",
        n_particles=80,
        max_rounds=2,
        n_scenarios_per_round=6,
        n_eig_samples=32,
        convergence=conv,
        seed=40,
    )
    stock_cfg = StockBacktestConfig(n_periods=20, drawdown_mc_samples=250)
    tr = run_cross_domain_transfer(
        n_users=2,
        elicitation=elic,
        seed=41,
        stock_config=stock_cfg,
    )
    assert tr.n_users == 2
    for block in (tr.generic, tr.within_domain, tr.cross_domain):
        assert "mean" in block
        assert "ci_low" in block
        assert "ci_high" in block
    assert len(tr.per_user["generic"]) == 2


def test_game_inferred_theta_stock_allocation_valid() -> None:
    from src.agents.elicitation_loop import ElicitationLoop
    from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler
    from src.utils.diagnostic_scenarios import ScenarioLibrary

    env_game = create_variant_a()
    stock = StockBacktestConfig(n_periods=12, drawdown_mc_samples=200)
    from src.environments.stock_backtest import StockBacktestEnv

    stock_env = StockBacktestEnv(config=stock)
    sampler = SyntheticUserSampler(seed=50)
    ut = sampler.sample()
    lib = ScenarioLibrary(seed=51)
    elic = ElicitationConfig(
        posterior_type="particle",
        n_particles=64,
        max_rounds=2,
        n_scenarios_per_round=6,
        n_eig_samples=24,
        convergence=ConvergenceConfig(max_rounds=2),
        seed=52,
        scenario_library=lib,
    )
    loop = ElicitationLoop(elic)
    env_game.reset(seed=53)
    user = SyntheticUser(ut, seed=54)
    res = loop.run(env_game, user, query_type="active")
    stock_env.reset(seed=55)
    alloc = stock_env.get_optimal_action(res.inferred_theta)
    assert alloc.shape == (5,)
    assert np.isclose(alloc.sum(), 1.0, atol=1e-5)
    ok, _ = stock_env.check_quality_floor(alloc)
    assert ok
