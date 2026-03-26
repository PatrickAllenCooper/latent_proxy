from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.supply_chain import SUPPLY_REGIME_NAMES, SupplyChainEnv
from src.training.serialization import AllocationSerializer, UserProfileSerializer
from src.training.synthetic_users import UserType


class SupplyChainStateSerializer:
    """Renders supply chain procurement state as natural-language context."""

    def serialize(
        self,
        obs: dict[str, Any],
        env: SupplyChainEnv,
        include_instruction: bool = True,
    ) -> str:
        wealth = obs["wealth"]
        market_state = obs["market_state"]
        current_period = obs["round"]
        total_periods = env.config.n_periods
        total_budget = float(wealth.sum())
        supplier_names = [
            s.name.replace("_", " ").title() for s in env.config.suppliers
        ]

        lines: list[str] = [
            "You are advising on a multi-supplier procurement strategy.",
            f"This is procurement period {current_period + 1} of {total_periods}.",
            "",
            f"Total procurement budget: ${total_budget:,.2f}",
        ]
        for i, label in enumerate(supplier_names):
            w = float(wealth[i])
            pct = (w / total_budget * 100) if total_budget > 0 else 0.0
            lines.append(f"  {label:22s} ${w:>12,.2f} ({pct:.1f}%)")
        lines.append("")

        regime_name = SUPPLY_REGIME_NAMES.get(env._regime, "unknown")
        lines.append(f"Current supply chain conditions: {regime_name}.")
        lines.append(
            "Per-supplier expected cost savings, delivery volatility, and efficiency:"
        )
        for i, label in enumerate(supplier_names):
            mu = float(market_state[i, 0])
            var = float(market_state[i, 1])
            vol = np.sqrt(max(var, 0.0))
            eff = mu / vol if vol > 1e-12 else 0.0
            lines.append(
                f"  {label:22s}  savings={mu*100:.3f}%  vol={vol*100:.3f}%  "
                f"efficiency={eff:.3f}"
            )

        if include_instruction:
            listed = ", ".join(supplier_names)
            lines.append("")
            lines.append(
                f"Recommend procurement allocation across these suppliers: "
                f"{listed}. Allocations must sum to 100%."
            )

        return "\n".join(lines)


class SupplyChainAllocationSerializer(AllocationSerializer):
    """Allocation text using supplier names."""

    def __init__(self, env: SupplyChainEnv) -> None:
        names = [s.name.replace("_", " ") for s in env.config.suppliers]
        super().__init__(names)


def build_supply_chain_prompt(
    obs: dict[str, Any],
    env: SupplyChainEnv,
    user_type: UserType | None = None,
) -> str:
    ss = SupplyChainStateSerializer()
    prompt = ss.serialize(obs, env)
    if user_type is not None:
        ups = UserProfileSerializer()
        prompt = prompt + "\n\n" + ups.serialize(user_type)
    return prompt
