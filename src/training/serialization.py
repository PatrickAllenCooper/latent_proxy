from __future__ import annotations

import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.resource_game import REGIME_NAMES, ResourceStrategyGame
from src.training.synthetic_users import UserType


class GameStateSerializer:
    """Converts environment observations into natural language prompts."""

    def serialize(
        self,
        obs: dict[str, Any],
        env: ResourceStrategyGame,
        include_instruction: bool = True,
    ) -> str:
        wealth = obs["wealth"]
        market_state = obs["market_state"]
        current_round = obs["round"]
        total_rounds = env.config.n_rounds
        total_wealth = float(wealth.sum())
        channel_names = [ch.name for ch in env.config.channels]

        lines = []
        lines.append(
            f"You are advising on a resource allocation strategy. "
            f"This is round {current_round + 1} of {total_rounds}."
        )
        lines.append("")

        lines.append(f"Current portfolio value: ${total_wealth:,.2f}")
        for i, name in enumerate(channel_names):
            w = float(wealth[i])
            pct = (w / total_wealth * 100) if total_wealth > 0 else 0.0
            lines.append(f"  {name.capitalize():12s} ${w:>10,.2f} ({pct:.1f}%)")
        lines.append("")

        regime_name = REGIME_NAMES.get(env._regime, "unknown")
        lines.append(f"Current market conditions (regime: {regime_name}):")
        for i, name in enumerate(channel_names):
            mean_pct = float(market_state[i, 0]) * 100
            vol_pct = np.sqrt(float(market_state[i, 1])) * 100
            lines.append(
                f"  {name.capitalize():12s} "
                f"expected return {mean_pct:.1f}%, volatility {vol_pct:.1f}%"
            )

        if include_instruction:
            lines.append("")
            channels_str = ", ".join(n.capitalize() for n in channel_names)
            lines.append(
                f"Recommend a percentage allocation across these channels: "
                f"{channels_str}. Allocations must sum to 100%."
            )

        return "\n".join(lines)


class UserProfileSerializer:
    """Converts UserType parameters to natural language descriptions."""

    _GAMMA_BANDS = [
        (0.0, 0.3, "Very short-term"),
        (0.3, 0.5, "Short-term"),
        (0.5, 0.7, "Medium-term"),
        (0.7, 0.9, "Long-term"),
        (0.9, 1.01, "Very long-term"),
    ]

    _ALPHA_BANDS = [
        (0.0, 0.5, "High", "low risk aversion"),
        (0.5, 1.0, "Moderate", "moderate risk aversion"),
        (1.0, 2.0, "Low", "high risk aversion"),
        (2.0, 100.0, "Very low", "very high risk aversion"),
    ]

    _LAMBDA_BANDS = [
        (1.0, 1.3, "Low"),
        (1.3, 1.8, "Moderate"),
        (1.8, 2.3, "High"),
        (2.3, 100.0, "Very high"),
    ]

    def serialize(self, user_type: UserType) -> str:
        horizon_label = "Medium-term"
        for lo, hi, label in self._GAMMA_BANDS:
            if lo <= user_type.gamma < hi:
                horizon_label = label
                break

        risk_tolerance = "Moderate"
        risk_label = "moderate risk aversion"
        for lo, hi, tol, rl in self._ALPHA_BANDS:
            if lo <= user_type.alpha < hi:
                risk_tolerance = tol
                risk_label = rl
                break

        loss_label = "Moderate"
        for lo, hi, label in self._LAMBDA_BANDS:
            if lo <= user_type.lambda_ < hi:
                loss_label = label
                break

        lines = [
            "The investor has the following preference profile:",
            f"- Time horizon: {horizon_label} (discount factor: {user_type.gamma:.2f})",
            f"- Risk tolerance: {risk_tolerance} -- {risk_label} "
            f"(alpha: {user_type.alpha:.2f})",
            f"- Loss sensitivity: {loss_label} (loss aversion: {user_type.lambda_:.2f})",
        ]
        return "\n".join(lines)


class AllocationSerializer:
    """Converts between numpy allocation arrays and structured text."""

    def __init__(self, channel_names: list[str] | None = None) -> None:
        self.channel_names = channel_names or ["safe", "growth", "aggressive", "volatile"]

    def serialize(self, allocation: NDArray[np.floating[Any]]) -> str:
        allocation = np.asarray(allocation, dtype=np.float64)
        total = allocation.sum()
        if total > 0:
            allocation = allocation / total

        lines = ["Recommended allocation:"]
        for i, name in enumerate(self.channel_names):
            pct = float(allocation[i]) * 100
            lines.append(f"  {name.capitalize()}: {pct:.0f}%")
        return "\n".join(lines)

    def parse(self, text: str) -> NDArray[np.floating[Any]]:
        """Parse an allocation from LLM text output.

        Tries multiple patterns to handle varied LLM output formatting.
        """
        allocations = np.zeros(len(self.channel_names), dtype=np.float64)

        for i, name in enumerate(self.channel_names):
            pattern = rf"{re.escape(name)}\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                allocations[i] = float(match.group(1)) / 100.0

        if allocations.sum() <= 0:
            numbers = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
            if len(numbers) == len(self.channel_names):
                for i, n in enumerate(numbers):
                    allocations[i] = float(n) / 100.0

        total = allocations.sum()
        if total > 0:
            allocations = allocations / total
        else:
            allocations = np.ones(len(self.channel_names)) / len(self.channel_names)

        return allocations


def build_prompt(
    obs: dict[str, Any],
    env: ResourceStrategyGame,
    user_type: UserType | None = None,
) -> str:
    """Build a complete prompt from game state and optional user profile."""
    gs = GameStateSerializer()
    prompt = gs.serialize(obs, env)

    if user_type is not None:
        ups = UserProfileSerializer()
        profile = ups.serialize(user_type)
        prompt = prompt + "\n\n" + profile

    return prompt
