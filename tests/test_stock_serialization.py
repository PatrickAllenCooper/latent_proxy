from __future__ import annotations

import numpy as np

from src.training.stock_serialization import (
    StockAllocationSerializer,
    StockStateSerializer,
)


def test_stock_state_serializer_content(stock_env) -> None:
    stock_env.reset(seed=0)
    obs = stock_env._get_obs()
    text = StockStateSerializer().serialize(obs, stock_env)
    assert "stock" in text.lower() or "portfolio" in text.lower()
    assert "Sharpe" in text
    assert "trading period" in text.lower() or "period" in text.lower()
    assert "game" not in text.lower()
    assert "channel" not in text.lower()


def test_stock_allocation_round_trip(stock_env) -> None:
    stock_env.reset(seed=1)
    ser = StockAllocationSerializer(stock_env)
    w = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    blob = ser.serialize(w)
    assert "%" in blob
    parsed = ser.parse(blob)
    assert parsed.shape == (5,)
    assert np.allclose(parsed, w / w.sum(), atol=0.02)
