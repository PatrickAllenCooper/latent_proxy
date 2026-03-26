from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.environments.stock_backtest import (
    StockBacktestConfig,
    StockBacktestEnv,
    create_default_stock_env,
    load_stock_config_yaml,
)
from src.training.synthetic_users import UserType


def test_stock_env_from_default_config(stock_config_short: StockBacktestConfig) -> None:
    env = StockBacktestEnv(config=stock_config_short)
    assert env.config.n_channels == 5
    obs, info = env.reset(seed=0)
    assert obs["wealth"].shape == (5,)
    assert obs["market_state"].shape == (5, 2)
    assert 0 <= obs["round"] < env.config.n_periods
    assert "total_wealth" in info


def test_stock_env_from_yaml() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "stock" / "default.yaml"
    cfg = load_stock_config_yaml(path)
    env = StockBacktestEnv(config=cfg)
    assert len(cfg.assets) == 5
    obs, _ = env.reset(seed=1)
    assert obs["wealth"].shape == (5,)


def test_create_default_stock_env() -> None:
    env = create_default_stock_env()
    assert isinstance(env, StockBacktestEnv)
    obs, _ = env.reset(seed=2)
    assert obs["market_state"].shape[0] == 5


def test_step_uniform_updates_wealth(stock_env: StockBacktestEnv) -> None:
    obs0, _ = stock_env.reset(seed=3)
    w0 = float(obs0["wealth"].sum())
    uniform = np.ones(5) / 5
    obs1, reward, term, trunc, info = stock_env.step(uniform)
    assert not term
    assert "returns" in info
    assert float(obs1["wealth"].sum()) > 0


def test_terminal_reward_matches_final_wealth(stock_config_short: StockBacktestConfig) -> None:
    cfg = StockBacktestConfig(
        n_periods=4,
        initial_capital=10_000.0,
        drawdown_mc_samples=200,
        assets=stock_config_short.assets,
        regime_transition_matrix=stock_config_short.regime_transition_matrix,
    )
    env = StockBacktestEnv(config=cfg)
    env.reset(seed=4)
    uniform = np.ones(5) / 5
    total_r = 0.0
    for _ in range(3):
        _, r, term, _, _ = env.step(uniform)
        total_r += r
        assert not term
    _, r_last, term, _, _ = env.step(uniform)
    assert term
    assert r_last == pytest.approx(float(env._wealth.sum()), rel=1e-9)


def test_regime_transitions_respect_matrix(stock_config_short: StockBacktestConfig) -> None:
    det = np.eye(3)
    cfg = StockBacktestConfig(
        n_periods=30,
        regime_transition_matrix=det,
        drawdown_mc_samples=200,
        assets=stock_config_short.assets,
    )
    env = StockBacktestEnv(config=cfg)
    env.reset(seed=5)
    uniform = np.ones(5) / 5
    for _ in range(10):
        _, _, _, _, info = env.step(uniform)
        assert info["regime"] == env._regime


def test_quality_score_finite(stock_env: StockBacktestEnv) -> None:
    stock_env.reset(seed=6)
    a = np.ones(5) / 5
    q = stock_env.quality_score(a)
    assert np.isfinite(q)


def test_quality_floor_rejects_concentrated(stock_env: StockBacktestEnv) -> None:
    stock_env.reset(seed=7)
    concentrated = np.array([1.0, 0, 0, 0, 0])
    ok, violations = stock_env.check_quality_floor(concentrated)
    assert not ok
    assert any("active" in v.lower() or "minimum" in v.lower() for v in violations)


def test_quality_floor_passes_diversified(stock_env: StockBacktestEnv) -> None:
    stock_env.reset(seed=8)
    div = np.array([0.25, 0.25, 0.2, 0.15, 0.15])
    ok, violations = stock_env.check_quality_floor(div)
    assert ok
    assert violations == []


def test_optimal_action_differs_by_preferences(stock_env: StockBacktestEnv) -> None:
    stock_env.reset(seed=9)
    patient = stock_env.get_optimal_action(
        {"gamma": 0.95, "alpha": 2.0, "lambda_": 2.0},
    )
    impatient = stock_env.get_optimal_action(
        {"gamma": 0.25, "alpha": 0.35, "lambda_": 1.1},
    )
    risk_averse = stock_env.get_optimal_action(
        {"gamma": 0.75, "alpha": 2.5, "lambda_": 2.5},
    )
    risk_seeking = stock_env.get_optimal_action(
        {"gamma": 0.75, "alpha": 0.2, "lambda_": 1.05},
    )
    spread_time = float(np.linalg.norm(patient - impatient))
    spread_risk = float(np.linalg.norm(risk_averse - risk_seeking))
    assert max(spread_time, spread_risk) > 0.08


def test_sampled_user_optimals_pass_floor(stock_env: StockBacktestEnv) -> None:
    stock_env.reset(seed=10)
    rng = np.random.default_rng(11)
    for _ in range(8):
        g = float(rng.uniform(0.35, 0.95))
        a = float(rng.lognormal(0.0, 0.4))
        lam = float(rng.uniform(1.05, 2.8))
        ut = UserType(gamma=g, alpha=a, lambda_=lam)
        act = stock_env.get_optimal_action(
            {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_},
        )
        ok, _ = stock_env.check_quality_floor(act)
        assert ok


def test_channel_stats_shift_with_regime(stock_config_short: StockBacktestConfig) -> None:
    cfg = StockBacktestConfig(
        n_periods=5,
        drawdown_mc_samples=200,
        assets=stock_config_short.assets,
        regime_transition_matrix=np.eye(3),
    )
    env = StockBacktestEnv(config=cfg)
    from src.environments.resource_game import BEAR, BULL

    env.reset(seed=12)
    env._regime = BULL
    s_bull = env.get_channel_stats()
    env._regime = BEAR
    s_bear = env.get_channel_stats()
    assert not np.allclose(s_bull["means"], s_bear["means"])
    assert not np.allclose(s_bull["variances"], s_bear["variances"])
