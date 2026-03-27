"""Tests for LLM-in-the-loop elicitation using a mock model (no GPU required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any

import numpy as np
import pytest

from src.agents.llm_elicitation import (
    LLMElicitationConfig,
    LLMElicitationLoop,
    LLMElicitationResult,
    parse_two_options,
    _serialize_state,
    _get_channel_names,
)
from src.environments.resource_game import ResourceStrategyGame
from src.environments.stock_backtest import StockBacktestConfig, StockBacktestEnv
from src.environments.supply_chain import SupplyChainConfig, SupplyChainEnv
from src.training.synthetic_users import SyntheticUser, UserType


def _mock_generate_fn(n_channels: int):
    """Build a mock _generate_text that returns valid allocation text."""
    rng = np.random.default_rng(99)

    def _gen(model, tokenizer, prompt, config):
        alloc_a = rng.dirichlet(np.ones(n_channels))
        alloc_b = rng.dirichlet(np.ones(n_channels))
        if "Recommend" in prompt or "final" in prompt.lower():
            lines = ["Recommended allocation:"]
            for i in range(n_channels):
                lines.append(f"  channel_{i}: {alloc_a[i]*100:.0f}%")
            return "\n".join(lines)
        lines = ["Option A:"]
        for i in range(n_channels):
            lines.append(f"  channel_{i}: {alloc_a[i]*100:.0f}%")
        lines.append("Option B:")
        for i in range(n_channels):
            lines.append(f"  channel_{i}: {alloc_b[i]*100:.0f}%")
        return "\n".join(lines)

    return _gen


def test_parse_two_options_basic():
    text = "Option A:\n  safe: 60%\n  growth: 40%\nOption B:\n  safe: 20%\n  growth: 80%"
    a, b = parse_two_options(text, 2)
    assert a.shape == (2,)
    assert b.shape == (2,)
    assert np.isclose(a.sum(), 1.0, atol=0.01)
    assert np.isclose(b.sum(), 1.0, atol=0.01)
    assert a[0] > a[1]
    assert b[1] > b[0]


def test_parse_two_options_fallback():
    text = "No valid allocations here."
    a, b = parse_two_options(text, 4)
    assert np.allclose(a, np.ones(4) / 4)


@patch("src.agents.llm_elicitation._generate_text")
def test_llm_loop_game_env(mock_gen):
    env = ResourceStrategyGame()
    K = env.config.n_channels
    mock_gen.side_effect = _mock_generate_fn(K)

    model = MagicMock()
    tokenizer = MagicMock()
    cfg = LLMElicitationConfig(max_rounds=3)
    loop = LLMElicitationLoop(model, tokenizer, config=cfg)

    ut = UserType(gamma=0.7, alpha=1.0, lambda_=1.5)
    user = SyntheticUser(ut, seed=1)

    result = loop.run(env, user, seed=42)
    assert isinstance(result, LLMElicitationResult)
    assert result.n_rounds == 3
    assert result.recommendation.shape == (K,)
    assert np.isclose(result.recommendation.sum(), 1.0, atol=0.01)
    assert len(result.per_round_recommendations) == 3
    assert result.elapsed_seconds > 0


@patch("src.agents.llm_elicitation._generate_text")
def test_llm_loop_stock_env(mock_gen):
    env = StockBacktestEnv(config=StockBacktestConfig(n_periods=10, drawdown_mc_samples=200))
    K = env.config.n_channels
    mock_gen.side_effect = _mock_generate_fn(K)

    model = MagicMock()
    tokenizer = MagicMock()
    cfg = LLMElicitationConfig(max_rounds=2)
    loop = LLMElicitationLoop(model, tokenizer, config=cfg)

    ut = UserType(gamma=0.5, alpha=0.8, lambda_=1.3)
    user = SyntheticUser(ut, seed=2)

    result = loop.run(env, user, seed=43)
    assert result.n_rounds == 2
    assert result.recommendation.shape == (5,)


@patch("src.agents.llm_elicitation._generate_text")
def test_llm_loop_supply_chain_env(mock_gen):
    env = SupplyChainEnv(config=SupplyChainConfig(n_periods=8, resilience_mc_samples=200))
    K = env.config.n_channels
    mock_gen.side_effect = _mock_generate_fn(K)

    model = MagicMock()
    tokenizer = MagicMock()
    cfg = LLMElicitationConfig(max_rounds=2)
    loop = LLMElicitationLoop(model, tokenizer, config=cfg)

    ut = UserType(gamma=0.8, alpha=1.5, lambda_=2.0)
    user = SyntheticUser(ut, seed=3)

    result = loop.run(env, user, seed=44)
    assert result.n_rounds == 2
    assert result.recommendation.shape == (5,)


def test_serialize_state_game():
    env = ResourceStrategyGame()
    env.reset(seed=0)
    text = _serialize_state(env)
    assert "Total value" in text
    assert "%" in text


def test_get_channel_names_all_envs():
    game = ResourceStrategyGame()
    assert len(_get_channel_names(game)) == 4
    stock = StockBacktestEnv(config=StockBacktestConfig(n_periods=5, drawdown_mc_samples=100))
    assert len(_get_channel_names(stock)) == 5
    sc = SupplyChainEnv(config=SupplyChainConfig(n_periods=5, resilience_mc_samples=100))
    assert len(_get_channel_names(sc)) == 5
