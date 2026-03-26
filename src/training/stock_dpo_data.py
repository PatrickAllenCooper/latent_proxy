from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.stock_backtest import StockBacktestConfig, StockBacktestEnv
from src.training.dpo_data import (
    CandidateGenerator,
    DPOPair,
    DPOPairConfig,
    DPOPairGenerator,
)
from src.training.stock_serialization import (
    StockAllocationSerializer,
    build_stock_prompt,
)
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler, UserType

logger = logging.getLogger(__name__)


class StockDPOPairGenerator(DPOPairGenerator):
    """DPO pairs for the stock backtest domain (quality then alignment phases)."""

    def generate_dataset(
        self,
        stock_config: StockBacktestConfig | None = None,
    ) -> list[DPOPair]:
        cfg = stock_config or StockBacktestConfig(n_periods=120)
        env = StockBacktestEnv(config=cfg)
        sampler = SyntheticUserSampler(seed=self.config.seed)
        candidate_gen = CandidateGenerator(env, sampler, self._rng)
        alloc_serializer = StockAllocationSerializer(env)

        pairs: list[DPOPair] = []
        states_per_round = max(1, self.config.n_pairs // self.config.n_game_states)

        for state_idx in range(self.config.n_game_states):
            seed = self.config.seed + state_idx
            env.reset(seed=seed)

            n_steps = int(self._rng.integers(0, max(1, env.config.n_rounds // 2)))
            uniform_action = np.ones(env.config.n_channels) / env.config.n_channels
            for _ in range(n_steps):
                env.step(uniform_action)

            obs = env._get_obs()

            for _ in range(states_per_round):
                if self.config.curriculum_phase == 1:
                    pair = self._make_phase1_pair_stock(
                        obs, env, candidate_gen, alloc_serializer,
                    )
                else:
                    user_type = sampler.sample()
                    pair = self._make_phase2_pair_stock(
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
            "Generated %d stock DPO pairs for phase %d",
            len(pairs), self.config.curriculum_phase,
        )
        return pairs[: self.config.n_pairs]

    def _make_phase1_pair_stock(
        self,
        obs: dict[str, Any],
        env: StockBacktestEnv,
        candidate_gen: CandidateGenerator,
        alloc_serializer: StockAllocationSerializer,
    ) -> DPOPair | None:
        prompt = build_stock_prompt(obs, env, user_type=None)
        candidates = candidate_gen.generate(self.config.n_candidates)

        scored: list[tuple[NDArray[np.floating[Any]], float, bool]] = []
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

    def _make_phase2_pair_stock(
        self,
        obs: dict[str, Any],
        env: StockBacktestEnv,
        candidate_gen: CandidateGenerator,
        alloc_serializer: StockAllocationSerializer,
        user_type: UserType,
        sampler: SyntheticUserSampler,
    ) -> DPOPair | None:
        prompt = build_stock_prompt(obs, env, user_type=user_type)
        target_theta = {
            "gamma": user_type.gamma,
            "alpha": user_type.alpha,
            "lambda_": user_type.lambda_,
        }
        candidates = candidate_gen.generate(
            self.config.n_candidates, target_theta=target_theta,
        )

        user = SyntheticUser(user_type, seed=int(self._rng.integers(0, 2**31)))
        channel_stats = env.get_channel_stats()
        means = channel_stats["means"]
        variances = channel_stats["variances"]
        current_wealth = float(obs["wealth"].sum())
        rounds_left = env.config.n_rounds - obs["round"]

        scored: list[tuple[NDArray[np.floating[Any]], float]] = []
        for c in candidates:
            utility = user.evaluate_allocation(
                c, means, variances, current_wealth, rounds_left,
            )
            scored.append((c, utility))

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen_alloc, chosen_score = scored[0]

        contrasting = sampler.sample()
        guard = 0
        while (
            abs(contrasting.gamma - user_type.gamma) < 0.2
            and abs(contrasting.alpha - user_type.alpha) < 0.3
            and guard < 50
        ):
            contrasting = sampler.sample()
            guard += 1

        contrast_theta = {
            "gamma": contrasting.gamma,
            "alpha": contrasting.alpha,
            "lambda_": contrasting.lambda_,
        }
        rejected_alloc = env.get_optimal_action(contrast_theta)
        contrast_user = SyntheticUser(
            contrasting, seed=int(self._rng.integers(0, 2**31)),
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
