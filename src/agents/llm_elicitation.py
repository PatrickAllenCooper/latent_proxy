"""LLM-in-the-loop preference elicitation (DPO study).

The LLM drives the full elicitation cycle: generates diagnostic queries,
observes user choices, and produces a final recommendation. Compared against
the analytical baseline (EIG + particle filter) to measure DPO's effect.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.base import BaseEnvironment
from src.training.serialization import AllocationSerializer
from src.training.synthetic_users import SyntheticUser, UserType

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert financial advisor. Your task is to understand an investor's "
    "preferences through a structured dialogue, then recommend an optimal allocation. "
    "Always format allocations as percentage lists that sum to 100%."
)

QUERY_TEMPLATE = """{system}

{state_description}

{history_block}

Propose two distinct allocation strategies for the investor to choose between.
Option A should be more conservative (lower risk, steadier returns).
Option B should be more aggressive (higher expected return, more volatility).

Format your response EXACTLY as:
Option A:
  {channel_list_a}
Option B:
  {channel_list_b}
"""

RECOMMEND_TEMPLATE = """{system}

{state_description}

The investor made the following choices during our conversation:
{history_summary}

Based on these revealed preferences, recommend a single final allocation.
Format your response EXACTLY as:
Recommended allocation:
  {channel_list}
"""


@dataclass
class LLMElicitationConfig:
    max_rounds: int = 5
    max_new_tokens: int = 256
    temperature: float = 0.3
    do_sample: bool = True


@dataclass
class LLMElicitationResult:
    recommendation: NDArray[np.floating[Any]]
    history: list[dict[str, Any]]
    n_rounds: int
    elapsed_seconds: float
    per_round_recommendations: list[NDArray[np.floating[Any]]]
    true_theta: UserType | None = None


def _serialize_state(env: BaseEnvironment) -> str:
    obs = env._get_obs()
    wealth = obs["wealth"]
    market_state = obs["market_state"]
    total = float(wealth.sum())
    K = env.config.n_channels
    names = _get_channel_names(env)

    lines = [f"Total value: ${total:,.2f}"]
    for i in range(K):
        w = float(wealth[i])
        pct = (w / total * 100) if total > 0 else 0.0
        mu = float(market_state[i, 0])
        var = float(market_state[i, 1])
        vol = np.sqrt(max(var, 0.0))
        lines.append(
            f"  {names[i]:20s} ${w:>10,.2f} ({pct:.1f}%)  "
            f"E[r]={mu*100:.2f}% vol={vol*100:.2f}%"
        )
    return "\n".join(lines)


def _get_channel_names(env: BaseEnvironment) -> list[str]:
    if hasattr(env, "config"):
        cfg = env.config
        if hasattr(cfg, "channels"):
            return [ch.name for ch in cfg.channels]
        if hasattr(cfg, "assets"):
            return [a.name.replace("_", " ") for a in cfg.assets]
        if hasattr(cfg, "suppliers"):
            return [s.name.replace("_", " ") for s in cfg.suppliers]
    K = env.config.n_channels
    return [f"channel_{i}" for i in range(K)]


def _format_channel_template(names: list[str]) -> str:
    return "\n  ".join(f"{n}: __%" for n in names)


def _build_history_block(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No prior interactions yet."
    lines = ["Prior interaction history:"]
    for i, h in enumerate(history):
        choice_label = "A" if h["choice"] == 0 else "B"
        lines.append(f"  Round {i+1}: Investor chose Option {choice_label}")
    return "\n".join(lines)


def _build_history_summary(history: list[dict[str, Any]], names: list[str]) -> str:
    if not history:
        return "No interactions recorded."
    lines = []
    for i, h in enumerate(history):
        choice_label = "A" if h["choice"] == 0 else "B"
        chosen = h["option_a"] if h["choice"] == 0 else h["option_b"]
        alloc_str = ", ".join(
            f"{names[j]}: {float(chosen[j])*100:.0f}%" for j in range(len(names))
        )
        lines.append(f"  Round {i+1}: Chose Option {choice_label} ({alloc_str})")
    return "\n".join(lines)


def parse_two_options(
    text: str, n_channels: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract Option A and Option B allocations from LLM output."""
    option_a = np.ones(n_channels) / n_channels
    option_b = np.ones(n_channels) / n_channels

    parts = re.split(r"Option\s*B\s*[:\-]?", text, flags=re.IGNORECASE)
    block_a = parts[0] if parts else text
    block_b = parts[1] if len(parts) > 1 else text

    nums_a = re.findall(r"(\d+(?:\.\d+)?)\s*%", block_a)
    nums_b = re.findall(r"(\d+(?:\.\d+)?)\s*%", block_b)

    if len(nums_a) >= n_channels:
        vals = np.array([float(x) / 100 for x in nums_a[:n_channels]])
        if vals.sum() > 0:
            option_a = vals / vals.sum()
    if len(nums_b) >= n_channels:
        vals = np.array([float(x) / 100 for x in nums_b[:n_channels]])
        if vals.sum() > 0:
            option_b = vals / vals.sum()

    return option_a, option_b


def _generate_text(model: Any, tokenizer: Any, prompt: str, config: LLMElicitationConfig) -> str:
    """Run a single generation pass."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_new_tokens * 2)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=max(config.temperature, 0.01),
            do_sample=config.do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


class LLMElicitationLoop:
    """Full LLM-driven elicitation: the model generates queries and recommendations."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: LLMElicitationConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or LLMElicitationConfig()

    def run(
        self,
        env: BaseEnvironment,
        user: SyntheticUser,
        *,
        seed: int = 42,
    ) -> LLMElicitationResult:
        env.reset(seed=seed)
        names = _get_channel_names(env)
        K = env.config.n_channels
        serializer = AllocationSerializer(names)
        channel_tpl = _format_channel_template(names)

        history: list[dict[str, Any]] = []
        per_round_recs: list[NDArray[np.floating[Any]]] = []
        t0 = time.perf_counter()

        for rnd in range(self.config.max_rounds):
            state_desc = _serialize_state(env)
            history_block = _build_history_block(history)

            query_prompt = QUERY_TEMPLATE.format(
                system=SYSTEM_PROMPT,
                state_description=state_desc,
                history_block=history_block,
                channel_list_a=channel_tpl,
                channel_list_b=channel_tpl,
            )

            raw_query = _generate_text(self.model, self.tokenizer, query_prompt, self.config)
            option_a, option_b = parse_two_options(raw_query, K)

            stats = env.get_channel_stats()
            means = stats["means"]
            variances = stats["variances"]
            wealth = float(env._get_obs()["wealth"].sum())
            rounds_left = env.config.n_rounds - env._get_obs()["round"]

            eu_a = user.evaluate_allocation(option_a, means, variances, wealth, rounds_left)
            eu_b = user.evaluate_allocation(option_b, means, variances, wealth, rounds_left)
            choice = user.choose(eu_a, eu_b)

            history.append({
                "round": rnd,
                "option_a": option_a,
                "option_b": option_b,
                "choice": choice,
                "raw_query": raw_query,
            })

            rec_prompt = RECOMMEND_TEMPLATE.format(
                system=SYSTEM_PROMPT,
                state_description=state_desc,
                history_summary=_build_history_summary(history, names),
                channel_list=channel_tpl,
            )
            raw_rec = _generate_text(self.model, self.tokenizer, rec_prompt, self.config)
            rec = serializer.parse(raw_rec)
            per_round_recs.append(rec)

            logger.debug("Round %d: choice=%d, rec=%s", rnd + 1, choice, rec)

        elapsed = time.perf_counter() - t0
        final_rec = per_round_recs[-1] if per_round_recs else np.ones(K) / K

        return LLMElicitationResult(
            recommendation=final_rec,
            history=history,
            n_rounds=len(history),
            elapsed_seconds=elapsed,
            per_round_recommendations=per_round_recs,
            true_theta=user.user_type,
        )
