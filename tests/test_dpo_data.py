from __future__ import annotations

import numpy as np
import pytest

from src.environments.resource_game import GameConfig, ResourceStrategyGame
from src.training.dpo_data import (
    CandidateGenerator,
    DPOPair,
    DPOPairConfig,
    DPOPairGenerator,
    pairs_to_hf_dict,
)
from src.training.serialization import AllocationSerializer
from src.training.synthetic_users import SyntheticUserSampler, UserType


class TestCandidateGenerator:
    def test_generates_correct_count(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        sampler = SyntheticUserSampler(seed=42)
        rng = np.random.default_rng(42)
        gen = CandidateGenerator(env, sampler, rng)
        candidates = gen.generate(8)
        assert len(candidates) == 8

    def test_candidates_are_simplex(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        sampler = SyntheticUserSampler(seed=42)
        rng = np.random.default_rng(42)
        gen = CandidateGenerator(env, sampler, rng)
        candidates = gen.generate(10)
        for c in candidates:
            np.testing.assert_allclose(c.sum(), 1.0, atol=1e-6)
            assert np.all(c >= 0)

    def test_with_target_theta(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        sampler = SyntheticUserSampler(seed=42)
        rng = np.random.default_rng(42)
        gen = CandidateGenerator(env, sampler, rng)
        theta = {"gamma": 0.9, "alpha": 1.0, "lambda_": 1.5}
        candidates = gen.generate(6, target_theta=theta)
        assert len(candidates) == 6

    def test_candidates_have_variety(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=42)
        sampler = SyntheticUserSampler(seed=42)
        rng = np.random.default_rng(42)
        gen = CandidateGenerator(env, sampler, rng)
        candidates = gen.generate(10)
        arr = np.array(candidates)
        assert arr.std(axis=0).max() > 0.01


class TestDPOPairGeneratorPhase1:
    def test_generates_pairs(self) -> None:
        config = DPOPairConfig(
            n_pairs=20, n_game_states=10, curriculum_phase=1, seed=42,
        )
        gen = DPOPairGenerator(config)
        pairs = gen.generate_dataset()
        assert len(pairs) > 0
        assert len(pairs) <= 20

    def test_pair_structure(self) -> None:
        config = DPOPairConfig(
            n_pairs=10, n_game_states=5, curriculum_phase=1, seed=42,
        )
        gen = DPOPairGenerator(config)
        pairs = gen.generate_dataset()
        for pair in pairs:
            assert isinstance(pair.prompt, str)
            assert isinstance(pair.chosen, str)
            assert isinstance(pair.rejected, str)
            assert len(pair.prompt) > 0
            assert len(pair.chosen) > 0
            assert len(pair.rejected) > 0

    def test_phase1_no_user_profile(self) -> None:
        config = DPOPairConfig(
            n_pairs=10, n_game_states=5, curriculum_phase=1, seed=42,
        )
        gen = DPOPairGenerator(config)
        pairs = gen.generate_dataset()
        for pair in pairs:
            assert "preference profile" not in pair.prompt

    def test_chosen_higher_quality(self) -> None:
        config = DPOPairConfig(
            n_pairs=20, n_game_states=10, curriculum_phase=1, seed=42,
        )
        gen = DPOPairGenerator(config)
        pairs = gen.generate_dataset()
        higher_count = sum(1 for p in pairs if p.chosen_score >= p.rejected_score)
        assert higher_count / len(pairs) > 0.8

    def test_chosen_rejected_differ(self) -> None:
        config = DPOPairConfig(
            n_pairs=10, n_game_states=5, curriculum_phase=1, seed=42,
        )
        gen = DPOPairGenerator(config)
        pairs = gen.generate_dataset()
        for pair in pairs:
            assert pair.chosen != pair.rejected


class TestDPOPairGeneratorPhase2:
    def test_generates_pairs(self) -> None:
        config = DPOPairConfig(
            n_pairs=20, n_game_states=10, curriculum_phase=2, seed=42,
        )
        gen = DPOPairGenerator(config)
        pairs = gen.generate_dataset()
        assert len(pairs) > 0

    def test_phase2_has_user_profile(self) -> None:
        config = DPOPairConfig(
            n_pairs=10, n_game_states=5, curriculum_phase=2, seed=42,
        )
        gen = DPOPairGenerator(config)
        pairs = gen.generate_dataset()
        for pair in pairs:
            assert "preference profile" in pair.prompt

    def test_phase2_has_user_type(self) -> None:
        config = DPOPairConfig(
            n_pairs=10, n_game_states=5, curriculum_phase=2, seed=42,
        )
        gen = DPOPairGenerator(config)
        pairs = gen.generate_dataset()
        for pair in pairs:
            assert pair.user_type is not None
            assert isinstance(pair.user_type, UserType)


class TestPairsToHFDict:
    def test_correct_keys(self) -> None:
        pairs = [
            DPOPair(prompt="p1", chosen="c1", rejected="r1"),
            DPOPair(prompt="p2", chosen="c2", rejected="r2"),
        ]
        d = pairs_to_hf_dict(pairs)
        assert set(d.keys()) == {"prompt", "chosen", "rejected"}
        assert len(d["prompt"]) == 2

    def test_correct_values(self) -> None:
        pairs = [DPOPair(prompt="p", chosen="c", rejected="r")]
        d = pairs_to_hf_dict(pairs)
        assert d["prompt"][0] == "p"
        assert d["chosen"][0] == "c"
        assert d["rejected"][0] == "r"

    def test_empty(self) -> None:
        d = pairs_to_hf_dict([])
        assert d["prompt"] == []


class TestReproducibility:
    def test_same_seed_same_pairs(self) -> None:
        config1 = DPOPairConfig(
            n_pairs=10, n_game_states=5, curriculum_phase=1, seed=99,
        )
        config2 = DPOPairConfig(
            n_pairs=10, n_game_states=5, curriculum_phase=1, seed=99,
        )
        gen1 = DPOPairGenerator(config1)
        gen2 = DPOPairGenerator(config2)
        pairs1 = gen1.generate_dataset()
        pairs2 = gen2.generate_dataset()
        assert len(pairs1) == len(pairs2)
        for p1, p2 in zip(pairs1, pairs2):
            assert p1.prompt == p2.prompt
            assert p1.chosen == p2.chosen
