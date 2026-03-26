from __future__ import annotations

import numpy as np

from src.training.supply_chain_serialization import (
    SupplyChainAllocationSerializer,
    SupplyChainStateSerializer,
)


def test_supply_chain_state_serializer(supply_chain_env):
    supply_chain_env.reset(seed=0)
    obs = supply_chain_env._get_obs()
    text = SupplyChainStateSerializer().serialize(obs, supply_chain_env)
    assert "procurement" in text.lower() or "supplier" in text.lower()
    assert "efficiency" in text.lower()
    assert "game" not in text.lower()
    assert "portfolio" not in text.lower()
    assert "channel" not in text.lower()


def test_supply_chain_allocation_round_trip(supply_chain_env):
    supply_chain_env.reset(seed=1)
    ser = SupplyChainAllocationSerializer(supply_chain_env)
    w = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    blob = ser.serialize(w)
    assert "%" in blob
    parsed = ser.parse(blob)
    assert parsed.shape == (5,)
    assert np.allclose(parsed, w / w.sum(), atol=0.02)


def test_supply_chain_no_stock_terms(supply_chain_env):
    supply_chain_env.reset(seed=2)
    obs = supply_chain_env._get_obs()
    text = SupplyChainStateSerializer().serialize(obs, supply_chain_env)
    for forbidden in ["stock", "bond", "equity", "treasury", "sharpe", "portfolio"]:
        assert forbidden not in text.lower()
