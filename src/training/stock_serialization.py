from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.resource_game import REGIME_NAMES
from src.environments.stock_backtest import StockBacktestEnv
from src.training.serialization import AllocationSerializer, UserProfileSerializer
from src.training.synthetic_users import UserType


class StockStateSerializer:
    """Renders stock backtest observations as natural-language context."""

    def serialize(
        self,
        obs: dict[str, Any],
        env: StockBacktestEnv,
        include_instruction: bool = True,
    ) -> str:
        wealth = obs["wealth"]
        market_state = obs["market_state"]
        current_period = obs["round"]
        total_periods = env.config.n_periods
        total_value = float(wealth.sum())
        asset_names = [a.name.replace("_", " ").title() for a in env.config.assets]

        lines: list[str] = [
            "You are advising on a diversified stock and bond portfolio.",
            f"This is trading period {current_period + 1} of {total_periods}.",
            "",
            f"Total portfolio value: ${total_value:,.2f}",
        ]
        for i, label in enumerate(asset_names):
            w = float(wealth[i])
            pct = (w / total_value * 100) if total_value > 0 else 0.0
            lines.append(f"  {label:18s} ${w:>12,.2f} ({pct:.1f}%)")
        lines.append("")

        regime_name = REGIME_NAMES.get(env._regime, "unknown")
        lines.append(f"Current macro regime: {regime_name}.")
        lines.append("Per-asset expected return, volatility, and Sharpe ratio (per period):")
        for i, label in enumerate(asset_names):
            mu = float(market_state[i, 0])
            var = float(market_state[i, 1])
            vol = np.sqrt(max(var, 0.0))
            sharpe = mu / vol if vol > 1e-12 else 0.0
            lines.append(
                f"  {label:18s}  E[r]={mu*100:.3f}%  vol={vol*100:.3f}%  "
                f"Sharpe={sharpe:.3f}"
            )

        if include_instruction:
            listed = ", ".join(asset_names)
            lines.append("")
            lines.append(
                f"Recommend portfolio weights across these asset classes: {listed}. "
                "Weights must sum to 100%."
            )

        return "\n".join(lines)


class StockAllocationSerializer(AllocationSerializer):
    """Allocation text for stock asset names."""

    def __init__(self, env: StockBacktestEnv) -> None:
        names = [a.name.replace("_", " ") for a in env.config.assets]
        super().__init__(names)


def build_stock_prompt(
    obs: dict[str, Any],
    env: StockBacktestEnv,
    user_type: UserType | None = None,
) -> str:
    """Full prompt: portfolio state plus optional investor profile."""
    ss = StockStateSerializer()
    prompt = ss.serialize(obs, env)
    if user_type is not None:
        ups = UserProfileSerializer()
        prompt = prompt + "\n\n" + ups.serialize(user_type)
    return prompt
