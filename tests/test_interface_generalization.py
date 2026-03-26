from __future__ import annotations

import numpy as np

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop
from src.agents.preference_tracker import ConvergenceConfig, PreferenceTracker
from src.agents.query_generator import StructuredQueryGenerator
from src.environments.stock_backtest import StockBacktestConfig, StockBacktestEnv
from src.training.synthetic_users import SyntheticUser, UserType
from src.utils.posterior import ParticlePosterior
from src.utils.stock_scenarios import StockScenarioLibrary


def test_elicitation_loop_runs_on_stock_env() -> None:
    env = StockBacktestEnv(
        config=StockBacktestConfig(n_periods=18, drawdown_mc_samples=200),
    )
    lib = StockScenarioLibrary(seed=60)
    cfg = ElicitationConfig(
        posterior_type="particle",
        n_particles=72,
        max_rounds=2,
        n_scenarios_per_round=6,
        n_eig_samples=28,
        temperature=0.1,
        convergence=ConvergenceConfig(max_rounds=2),
        seed=61,
        scenario_library=lib,
    )
    ut = UserType(gamma=0.65, alpha=1.0, lambda_=1.6)
    user = SyntheticUser(ut, seed=62)
    env.reset(seed=63)
    res = ElicitationLoop(cfg).run(env, user, query_type="active")
    assert {"gamma", "alpha", "lambda_"} <= set(res.inferred_theta.keys())
    assert res.n_rounds >= 1


def test_structured_query_generator_stock_library() -> None:
    env = StockBacktestEnv(
        config=StockBacktestConfig(n_periods=14, drawdown_mc_samples=200),
    )
    env.reset(seed=64)
    lib = StockScenarioLibrary(seed=65)
    gen = StructuredQueryGenerator(
        n_scenarios_per_round=9,
        n_eig_samples=24,
        seed=66,
        library=lib,
    )
    post = ParticlePosterior(n_particles=80, ess_threshold_ratio=0.2)
    scenario = gen.select_query(env, post)
    assert scenario.option_a.shape == (5,)
    assert abs(float(scenario.option_a.sum()) - 1.0) < 1e-5


def test_preference_tracker_robust_action_stock() -> None:
    env = StockBacktestEnv(
        config=StockBacktestConfig(n_periods=10, drawdown_mc_samples=200),
    )
    env.reset(seed=67)
    tracker = PreferenceTracker(
        posterior_type="particle",
        n_particles=80,
        temperature=0.1,
        convergence=ConvergenceConfig(max_rounds=2),
    )
    ok = tracker.check_robust_action(env)
    assert isinstance(ok, bool)
