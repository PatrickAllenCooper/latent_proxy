"""Gamma-only DPO pair generation.

Fixes alpha and lambda at constant values and varies only gamma (discount
factor / time preference). This simplifies the learning problem for small
models by focusing on a single preference dimension.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.base import BaseEnvironment
from src.environments.resource_game import GameConfig, ResourceStrategyGame
from src.training.dpo_data import DPOPair, DPOPairConfig, pairs_to_hf_dict
from src.training.serialization import AllocationSerializer, GameStateSerializer
from src.training.synthetic_users import SyntheticUser, UserType

logger = logging.getLogger(__name__)

FIXED_ALPHA = 1.0
FIXED_LAMBDA = 1.5

GAMMA_PROFILE_TEMPLATE = (
    "The investor's time horizon preference:\n"
    "- Patience level: {label} (discount factor gamma = {gamma:.2f})\n"
    "A patient investor (high gamma) values long-term compounding growth.\n"
    "An impatient investor (low gamma) prioritizes near-term returns."
)

GAMMA_LABELS = [
    (0.0, 0.3, "Very impatient"),
    (0.3, 0.5, "Impatient"),
    (0.5, 0.7, "Moderate"),
    (0.7, 0.9, "Patient"),
    (0.9, 1.01, "Very patient"),
]


def _gamma_label(gamma: float) -> str:
    for lo, hi, label in GAMMA_LABELS:
        if lo <= gamma < hi:
            return label
    return "Moderate"


def _gamma_profile(gamma: float) -> str:
    return GAMMA_PROFILE_TEMPLATE.format(label=_gamma_label(gamma), gamma=gamma)


class GammaDPOPairGenerator:
    """DPO pairs that discriminate gamma only (alpha and lambda held fixed)."""

    def __init__(self, config: DPOPairConfig | None = None) -> None:
        self.config = config or DPOPairConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def generate_dataset(
        self,
        game_config: GameConfig | None = None,
    ) -> list[DPOPair]:
        env = ResourceStrategyGame(config=game_config)
        gs = GameStateSerializer()
        alloc_ser = AllocationSerializer([ch.name for ch in env.config.channels])

        pairs: list[DPOPair] = []
        states_per_round = max(1, self.config.n_pairs // self.config.n_game_states)

        for state_idx in range(self.config.n_game_states):
            seed = self.config.seed + state_idx
            env.reset(seed=seed)
            n_steps = int(self._rng.integers(0, env.config.n_rounds // 2))
            uniform = np.ones(env.config.n_channels) / env.config.n_channels
            for _ in range(n_steps):
                env.step(uniform)
            obs = env._get_obs()

            for _ in range(states_per_round):
                gamma_user = float(self._rng.beta(2, 2))
                gamma_user = max(0.05, min(gamma_user, 0.99))
                ut = UserType(gamma=gamma_user, alpha=FIXED_ALPHA, lambda_=FIXED_LAMBDA)

                if self.config.curriculum_phase == 1:
                    pair = self._make_quality_pair(obs, env, alloc_ser, gs)
                else:
                    pair = self._make_gamma_pair(obs, env, ut, alloc_ser, gs)

                if pair is not None:
                    pairs.append(pair)
                if len(pairs) >= self.config.n_pairs:
                    break
            if len(pairs) >= self.config.n_pairs:
                break

        logger.info(
            "Generated %d gamma-only DPO pairs (phase %d)",
            len(pairs), self.config.curriculum_phase,
        )
        return pairs[: self.config.n_pairs]

    def _make_quality_pair(
        self,
        obs: dict[str, Any],
        env: BaseEnvironment,
        alloc_ser: AllocationSerializer,
        gs: GameStateSerializer,
    ) -> DPOPair | None:
        prompt = gs.serialize(obs, env)
        K = env.config.n_channels

        candidates = [
            self._rng.dirichlet(np.ones(K)) for _ in range(self.config.n_candidates)
        ]
        for g in [0.3, 0.5, 0.7, 0.9]:
            candidates.append(env.get_optimal_action(
                {"gamma": g, "alpha": FIXED_ALPHA, "lambda_": FIXED_LAMBDA},
            ))

        scored = [(c, env.quality_score(c), env.check_quality_floor(c)[0]) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        chosen, chosen_score, _ = scored[0]
        rejected = None
        rejected_score = 0.0
        for alloc, score, passes in reversed(scored):
            if not passes:
                rejected, rejected_score = alloc, score
                break
        if rejected is None:
            rejected, rejected_score = scored[-1][0], scored[-1][1]
        if np.allclose(chosen, rejected):
            return None

        return DPOPair(
            prompt=prompt,
            chosen=alloc_ser.serialize(chosen),
            rejected=alloc_ser.serialize(rejected),
            chosen_score=chosen_score,
            rejected_score=rejected_score,
        )

    def _make_gamma_pair(
        self,
        obs: dict[str, Any],
        env: BaseEnvironment,
        ut: UserType,
        alloc_ser: AllocationSerializer,
        gs: GameStateSerializer,
    ) -> DPOPair | None:
        state_text = gs.serialize(obs, env, include_instruction=True)
        profile = _gamma_profile(ut.gamma)
        prompt = state_text + "\n\n" + profile

        theta_target = {"gamma": ut.gamma, "alpha": FIXED_ALPHA, "lambda_": FIXED_LAMBDA}
        chosen = env.get_optimal_action(theta_target)
        chosen_score = env.quality_score(chosen)

        contrast_gamma = ut.gamma + (0.4 if ut.gamma < 0.5 else -0.4)
        contrast_gamma = max(0.05, min(contrast_gamma, 0.99))
        theta_contrast = {"gamma": contrast_gamma, "alpha": FIXED_ALPHA, "lambda_": FIXED_LAMBDA}
        rejected = env.get_optimal_action(theta_contrast)
        rejected_score = env.quality_score(rejected)

        if np.allclose(chosen, rejected):
            return None

        return DPOPair(
            prompt=prompt,
            chosen=alloc_ser.serialize(chosen),
            rejected=alloc_ser.serialize(rejected),
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            user_type=ut,
        )
