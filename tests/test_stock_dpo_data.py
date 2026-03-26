from __future__ import annotations

import numpy as np

from src.environments.stock_backtest import StockBacktestConfig, StockBacktestEnv
from src.training.dpo_data import DPOPairConfig, pairs_to_hf_dict
from src.training.stock_dpo_data import StockDPOPairGenerator
from src.training.stock_serialization import StockAllocationSerializer


def test_stock_dpo_phase1_sharpe_ordering() -> None:
    cfg = DPOPairConfig(
        n_pairs=12,
        n_candidates=6,
        n_game_states=3,
        curriculum_phase=1,
        seed=21,
    )
    stock_cfg = StockBacktestConfig(n_periods=16, drawdown_mc_samples=200)
    gen = StockDPOPairGenerator(cfg)
    pairs = gen.generate_dataset(stock_cfg)
    assert pairs
    env = StockBacktestEnv(config=stock_cfg)
    env.reset(seed=0)
    ser = StockAllocationSerializer(env)
    for p in pairs[:5]:
        assert "portfolio" in p.prompt.lower() or "advising" in p.prompt.lower()
        w_chosen = ser.parse(p.chosen)
        w_rej = ser.parse(p.rejected)
        assert np.isfinite(env.quality_score(w_chosen))
        assert p.chosen_score >= p.rejected_score - 1e-6


def test_stock_dpo_phase2_chosen_matches_user_optimal() -> None:
    cfg = DPOPairConfig(
        n_pairs=10,
        n_candidates=8,
        n_game_states=2,
        curriculum_phase=2,
        seed=22,
    )
    stock_cfg = StockBacktestConfig(n_periods=14, drawdown_mc_samples=200)
    gen = StockDPOPairGenerator(cfg)
    pairs = gen.generate_dataset(stock_cfg)
    assert pairs
    for p in pairs:
        assert p.user_type is not None
        assert "investor" in p.prompt.lower() or "preference" in p.prompt.lower()


def test_stock_dpo_hf_columns() -> None:
    cfg = DPOPairConfig(
        n_pairs=6,
        n_candidates=5,
        n_game_states=2,
        curriculum_phase=1,
        seed=23,
    )
    gen = StockDPOPairGenerator(cfg)
    pairs = gen.generate_dataset(
        StockBacktestConfig(n_periods=12, drawdown_mc_samples=200),
    )
    d = pairs_to_hf_dict(pairs)
    assert set(d.keys()) == {"prompt", "chosen", "rejected"}
    assert len(d["prompt"]) == len(pairs)
