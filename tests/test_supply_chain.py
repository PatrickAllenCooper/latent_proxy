from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.environments.supply_chain import (
    SupplyChainConfig,
    SupplyChainEnv,
    create_default_supply_chain_env,
    load_supply_chain_config_yaml,
)
from src.environments.resource_game import BEAR, BULL
from src.training.synthetic_users import UserType


def test_supply_chain_from_default_config(supply_chain_config_short):
    env = SupplyChainEnv(config=supply_chain_config_short)
    assert env.config.n_channels == 5
    obs, info = env.reset(seed=0)
    assert obs["wealth"].shape == (5,)
    assert obs["market_state"].shape == (5, 2)
    assert "total_wealth" in info


def test_supply_chain_from_yaml():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "supply_chain" / "default.yaml"
    cfg = load_supply_chain_config_yaml(path)
    env = SupplyChainEnv(config=cfg)
    assert len(cfg.suppliers) == 5
    obs, _ = env.reset(seed=1)
    assert obs["wealth"].shape == (5,)


def test_create_default_supply_chain_env():
    env = create_default_supply_chain_env()
    assert isinstance(env, SupplyChainEnv)
    obs, _ = env.reset(seed=2)
    assert obs["market_state"].shape[0] == 5


def test_step_updates_wealth(supply_chain_env):
    supply_chain_env.reset(seed=3)
    uniform = np.ones(5) / 5
    obs1, reward, term, trunc, info = supply_chain_env.step(uniform)
    assert not term
    assert "returns" in info
    assert float(obs1["wealth"].sum()) > 0


def test_terminal_reward(supply_chain_config_short):
    cfg = SupplyChainConfig(
        n_periods=4,
        initial_budget=10_000.0,
        resilience_mc_samples=200,
        suppliers=supply_chain_config_short.suppliers,
        regime_transition_matrix=supply_chain_config_short.regime_transition_matrix,
    )
    env = SupplyChainEnv(config=cfg)
    env.reset(seed=4)
    uniform = np.ones(5) / 5
    for _ in range(3):
        _, r, term, _, _ = env.step(uniform)
        assert not term
    _, r_last, term, _, _ = env.step(uniform)
    assert term
    assert r_last == pytest.approx(float(env._wealth.sum()), rel=1e-9)


def test_quality_score_finite(supply_chain_env):
    supply_chain_env.reset(seed=5)
    q = supply_chain_env.quality_score(np.ones(5) / 5)
    assert np.isfinite(q)


def test_quality_floor_rejects_concentrated(supply_chain_env):
    supply_chain_env.reset(seed=6)
    concentrated = np.array([1.0, 0, 0, 0, 0])
    ok, violations = supply_chain_env.check_quality_floor(concentrated)
    assert not ok
    assert len(violations) >= 1


def test_quality_floor_passes_diversified(supply_chain_env):
    supply_chain_env.reset(seed=7)
    div = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
    ok, violations = supply_chain_env.check_quality_floor(div)
    assert ok
    assert violations == []


def test_optimal_action_differs_by_preferences(supply_chain_env):
    supply_chain_env.reset(seed=8)
    risk_averse = supply_chain_env.get_optimal_action(
        {"gamma": 0.75, "alpha": 2.5, "lambda_": 2.5},
    )
    risk_seeking = supply_chain_env.get_optimal_action(
        {"gamma": 0.75, "alpha": 0.2, "lambda_": 1.05},
    )
    spread = float(np.linalg.norm(risk_averse - risk_seeking))
    assert spread > 0.05


def test_sampled_user_optimals_pass_floor(supply_chain_env):
    supply_chain_env.reset(seed=9)
    rng = np.random.default_rng(10)
    for _ in range(8):
        g = float(rng.uniform(0.35, 0.95))
        a = float(rng.lognormal(0.0, 0.4))
        lam = float(rng.uniform(1.05, 2.8))
        act = supply_chain_env.get_optimal_action(
            {"gamma": g, "alpha": a, "lambda_": lam},
        )
        ok, _ = supply_chain_env.check_quality_floor(act)
        assert ok


def test_channel_stats_shift_with_regime(supply_chain_config_short):
    cfg = SupplyChainConfig(
        n_periods=5,
        resilience_mc_samples=200,
        suppliers=supply_chain_config_short.suppliers,
        regime_transition_matrix=np.eye(3),
    )
    env = SupplyChainEnv(config=cfg)
    env.reset(seed=11)
    env._regime = BULL
    s_bull = env.get_channel_stats()
    env._regime = BEAR
    s_bear = env.get_channel_stats()
    assert not np.allclose(s_bull["means"], s_bear["means"])
