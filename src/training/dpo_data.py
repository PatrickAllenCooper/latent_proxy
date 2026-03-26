from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.base import BaseEnvironment
from src.environments.resource_game import GameConfig, ResourceStrategyGame
from src.training.serialization import (
    AllocationSerializer,
    build_prompt,
)
from src.training.synthetic_users import (
    SyntheticUser,
    SyntheticUserSampler,
    UserType,
)

logger = logging.getLogger(__name__)


@dataclass
class DPOPairConfig:
    """Configuration for DPO pair generation."""

    n_pairs: int = 10000
    n_candidates: int = 8
    n_game_states: int = 500
    curriculum_phase: int = 1
    seed: int = 42
    dirichlet_alpha: float = 1.0
    perturbation_std: float = 0.1


@dataclass
class DPOPair:
    """A single DPO training pair."""

    prompt: str
    chosen: str
    rejected: str
    chosen_score: float = 0.0
    rejected_score: float = 0.0
    user_type: UserType | None = None


class CandidateGenerator:
    """Generates diverse candidate allocations for DPO pair construction."""

    def __init__(
        self,
        env: BaseEnvironment,
        sampler: SyntheticUserSampler,
        rng: np.random.Generator,
    ) -> None:
        self._env = env
        self._sampler = sampler
        self._rng = rng
        self._K = env.config.n_channels

    def generate(
        self,
        n: int,
        target_theta: dict[str, float] | None = None,
    ) -> list[NDArray[np.floating[Any]]]:
        """Generate n diverse candidate allocations."""
        candidates: list[NDArray[np.floating[Any]]] = []

        if target_theta is not None:
            candidates.append(self._env.get_optimal_action(target_theta))

        n_contrasting = min(3, n // 3)
        for _ in range(n_contrasting):
            ut = self._sampler.sample()
            theta = {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
            candidates.append(self._env.get_optimal_action(theta))

        n_random = min(n // 3, n - len(candidates))
        for _ in range(n_random):
            raw = self._rng.dirichlet(np.ones(self._K))
            candidates.append(raw)

        while len(candidates) < n:
            if target_theta is not None:
                base = self._env.get_optimal_action(target_theta)
            else:
                base = np.ones(self._K) / self._K
            noise = self._rng.normal(0, 0.1, size=self._K)
            perturbed = np.maximum(base + noise, 0.0)
            total = perturbed.sum()
            if total > 0:
                perturbed = perturbed / total
            else:
                perturbed = np.ones(self._K) / self._K
            candidates.append(perturbed)

        return candidates[:n]


class DPOPairGenerator:
    """Generates DPO preference pairs from the game environment and synthetic users.

    Phase 1 (quality-only): y_w has the highest quality score, y_l is a low-quality
    or floor-violating candidate. No user profile in prompt.

    Phase 2 (type-conditioned): y_w has the highest utility for the sampled user type,
    y_l has high utility for a contrasting type. User profile included in prompt.
    """

    def __init__(self, config: DPOPairConfig | None = None) -> None:
        self.config = config or DPOPairConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def generate_dataset(
        self,
        game_config: GameConfig | None = None,
    ) -> list[DPOPair]:
        """Generate a full dataset of DPO pairs."""
        env = ResourceStrategyGame(config=game_config)
        sampler = SyntheticUserSampler(seed=self.config.seed)
        candidate_gen = CandidateGenerator(env, sampler, self._rng)
        alloc_serializer = AllocationSerializer(
            [ch.name for ch in env.config.channels]
        )

        pairs: list[DPOPair] = []
        states_per_round = max(1, self.config.n_pairs // self.config.n_game_states)

        for state_idx in range(self.config.n_game_states):
            seed = self.config.seed + state_idx
            env.reset(seed=seed)

            n_steps = self._rng.integers(0, env.config.n_rounds // 2)
            uniform_action = np.ones(env.config.n_channels) / env.config.n_channels
            for _ in range(n_steps):
                env.step(uniform_action)

            obs = env._get_obs()

            for _ in range(states_per_round):
                if self.config.curriculum_phase == 1:
                    pair = self._make_phase1_pair(
                        obs, env, candidate_gen, alloc_serializer
                    )
                else:
                    user_type = sampler.sample()
                    pair = self._make_phase2_pair(
                        obs, env, candidate_gen, alloc_serializer,
                        user_type, sampler,
                    )

                if pair is not None:
                    pairs.append(pair)

                if len(pairs) >= self.config.n_pairs:
                    break

            if len(pairs) >= self.config.n_pairs:
                break

        logger.info(
            "Generated %d DPO pairs for phase %d",
            len(pairs), self.config.curriculum_phase,
        )
        return pairs[:self.config.n_pairs]

    def _make_phase1_pair(
        self,
        obs: dict[str, Any],
        env: BaseEnvironment,
        candidate_gen: CandidateGenerator,
        alloc_serializer: AllocationSerializer,
    ) -> DPOPair | None:
        """Phase 1: quality-only pair. No user profile in prompt."""
        prompt = build_prompt(obs, env, user_type=None)
        candidates = candidate_gen.generate(self.config.n_candidates)

        scored = []
        for c in candidates:
            q_score = env.quality_score(c)
            passes, _ = env.check_quality_floor(c)
            scored.append((c, q_score, passes))

        scored.sort(key=lambda x: x[1], reverse=True)

        chosen_alloc, chosen_score, _ = scored[0]

        rejected_alloc = None
        rejected_score = 0.0

        for alloc, score, passes in reversed(scored):
            if not passes:
                rejected_alloc = alloc
                rejected_score = score
                break

        if rejected_alloc is None:
            rejected_alloc = scored[-1][0]
            rejected_score = scored[-1][1]

        if np.allclose(chosen_alloc, rejected_alloc):
            return None

        return DPOPair(
            prompt=prompt,
            chosen=alloc_serializer.serialize(chosen_alloc),
            rejected=alloc_serializer.serialize(rejected_alloc),
            chosen_score=chosen_score,
            rejected_score=rejected_score,
        )

    def _make_phase2_pair(
        self,
        obs: dict[str, Any],
        env: BaseEnvironment,
        candidate_gen: CandidateGenerator,
        alloc_serializer: AllocationSerializer,
        user_type: UserType,
        sampler: SyntheticUserSampler,
    ) -> DPOPair | None:
        """Phase 2: type-conditioned pair. User profile in prompt."""
        prompt = build_prompt(obs, env, user_type=user_type)
        target_theta = {
            "gamma": user_type.gamma,
            "alpha": user_type.alpha,
            "lambda_": user_type.lambda_,
        }
        candidates = candidate_gen.generate(
            self.config.n_candidates, target_theta=target_theta
        )

        user = SyntheticUser(user_type, seed=int(self._rng.integers(0, 2**31)))
        channel_stats = env.get_channel_stats()
        means = channel_stats["means"]
        variances = channel_stats["variances"]
        current_wealth = float(obs["wealth"].sum())
        rounds_left = env.config.n_rounds - obs["round"]

        scored = []
        for c in candidates:
            utility = user.evaluate_allocation(
                c, means, variances, current_wealth, rounds_left,
            )
            scored.append((c, utility))

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen_alloc, chosen_score = scored[0]

        contrasting = sampler.sample()
        while (
            abs(contrasting.gamma - user_type.gamma) < 0.2
            and abs(contrasting.alpha - user_type.alpha) < 0.3
        ):
            contrasting = sampler.sample()

        contrast_theta = {
            "gamma": contrasting.gamma,
            "alpha": contrasting.alpha,
            "lambda_": contrasting.lambda_,
        }
        rejected_alloc = env.get_optimal_action(contrast_theta)
        contrast_user = SyntheticUser(
            contrasting, seed=int(self._rng.integers(0, 2**31))
        )
        rejected_score = contrast_user.evaluate_allocation(
            rejected_alloc, means, variances, current_wealth, rounds_left,
        )

        if np.allclose(chosen_alloc, rejected_alloc):
            if len(scored) > 1:
                rejected_alloc = scored[-1][0]
                rejected_score = scored[-1][1]
            else:
                return None

        return DPOPair(
            prompt=prompt,
            chosen=alloc_serializer.serialize(chosen_alloc),
            rejected=alloc_serializer.serialize(rejected_alloc),
            chosen_score=chosen_score,
            rejected_score=rejected_score,
            user_type=user_type,
        )


def pairs_to_hf_dict(pairs: list[DPOPair]) -> dict[str, list[str]]:
    """Convert DPO pairs to a dict suitable for HuggingFace Dataset.from_dict()."""
    return {
        "prompt": [p.prompt for p in pairs],
        "chosen": [p.chosen for p in pairs],
        "rejected": [p.rejected for p in pairs],
    }


def save_pairs(pairs: list[DPOPair], path: str | Path) -> None:
    """Save DPO pairs to disk as a HuggingFace Dataset."""
    from datasets import Dataset

    ds = Dataset.from_dict(pairs_to_hf_dict(pairs))
    ds.save_to_disk(str(path))
    logger.info("Saved %d pairs to %s", len(pairs), path)


def load_pairs(path: str | Path) -> Any:
    """Load a saved DPO dataset from disk."""
    from datasets import load_from_disk

    return load_from_disk(str(path))
