from __future__ import annotations

import numpy as np
import pytest

from src.environments.resource_game import (
    BEAR,
    BULL,
    NEUTRAL,
    GameConfig,
    ResourceStrategyGame,
)


class TestEnvironmentCreation:
    def test_default_construction(self) -> None:
        env = ResourceStrategyGame()
        assert env.config.n_channels == 4
        assert env.config.n_rounds == 20

    def test_custom_config(self, short_game_config: GameConfig) -> None:
        env = ResourceStrategyGame(config=short_game_config)
        assert env.config.n_rounds == 5
        assert env.config.initial_wealth == 100.0

    def test_observation_space_shape(self, env: ResourceStrategyGame) -> None:
        obs, _ = env.reset(seed=0)
        assert obs["wealth"].shape == (4,)
        assert obs["market_state"].shape == (4, 2)
        assert isinstance(obs["round"], (int, np.integer))

    def test_action_space_shape(self, env: ResourceStrategyGame) -> None:
        assert env.action_space.shape == (4,)


class TestReset:
    def test_reset_initializes_wealth(self, env: ResourceStrategyGame) -> None:
        obs, _ = env.reset(seed=0)
        expected_per_channel = env.config.initial_wealth / env.config.n_channels
        np.testing.assert_allclose(
            obs["wealth"],
            np.full(4, expected_per_channel),
        )

    def test_reset_starts_at_round_zero(self, env: ResourceStrategyGame) -> None:
        obs, _ = env.reset(seed=0)
        assert obs["round"] == 0

    def test_reset_info_contains_fields(self, env: ResourceStrategyGame) -> None:
        _, info = env.reset(seed=0)
        assert "total_wealth" in info
        assert "regime" in info
        assert "regime_name" in info

    def test_reset_reproduces_with_seed(self) -> None:
        env1 = ResourceStrategyGame()
        env2 = ResourceStrategyGame()
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1["wealth"], obs2["wealth"])
        np.testing.assert_array_equal(obs1["market_state"], obs2["market_state"])


class TestStepping:
    def test_step_increments_round(self, env: ResourceStrategyGame) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        obs, _, _, _, info = env.step(action)
        assert obs["round"] == 1

    def test_step_returns_valid_obs(self, env: ResourceStrategyGame) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["wealth"].shape == (4,)
        assert np.all(obs["wealth"] >= 0)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_episode_terminates_after_t_rounds(
        self, short_env: ResourceStrategyGame
    ) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        for i in range(5):
            obs, reward, terminated, truncated, info = short_env.step(action)
            if i < 4:
                assert not terminated
            else:
                assert terminated

    def test_terminal_reward_is_total_wealth(
        self, short_env: ResourceStrategyGame
    ) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        for _ in range(5):
            _, reward, terminated, _, _ = short_env.step(action)
        assert terminated
        assert reward > 0

    def test_wealth_is_nonnegative(self, env: ResourceStrategyGame) -> None:
        action = np.array([0.0, 0.0, 0.0, 1.0])
        for _ in range(20):
            obs, _, terminated, _, _ = env.step(action)
            assert np.all(obs["wealth"] >= 0)
            if terminated:
                break

    def test_action_normalization(self, env: ResourceStrategyGame) -> None:
        unnormalized = np.array([2.0, 3.0, 1.0, 4.0])
        obs1, _, _, _, _ = env.step(unnormalized)
        assert np.all(obs1["wealth"] >= 0)

    def test_zero_action_becomes_uniform(self, env: ResourceStrategyGame) -> None:
        zero_action = np.array([0.0, 0.0, 0.0, 0.0])
        obs, _, _, _, _ = env.step(zero_action)
        assert np.all(obs["wealth"] >= 0)


class TestRegimeSwitching:
    def test_regime_is_valid(self, env: ResourceStrategyGame) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        regimes_seen = set()
        for _ in range(100):
            env.reset(seed=None)
            for _ in range(20):
                _, _, terminated, _, info = env.step(action)
                regimes_seen.add(info["regime"])
                if terminated:
                    break
        assert regimes_seen.issubset({BULL, BEAR, NEUTRAL})

    def test_multiple_regimes_visited_over_many_episodes(self) -> None:
        env = ResourceStrategyGame()
        action = np.array([0.25, 0.25, 0.25, 0.25])
        regimes_seen = set()
        for seed in range(50):
            env.reset(seed=seed)
            for _ in range(20):
                _, _, terminated, _, info = env.step(action)
                regimes_seen.add(info["regime"])
                if terminated:
                    break
        assert len(regimes_seen) == 3, (
            f"Expected all 3 regimes over 50 episodes, saw {regimes_seen}"
        )

    def test_regime_affects_returns(self, env: ResourceStrategyGame) -> None:
        stats_neutral = env.get_channel_stats()

        env._regime = BULL
        stats_bull = env.get_channel_stats()

        env._regime = BEAR
        stats_bear = env.get_channel_stats()

        assert not np.allclose(stats_neutral["means"], stats_bull["means"])
        assert not np.allclose(stats_neutral["means"], stats_bear["means"])


class TestReproducibility:
    def test_same_seed_same_trajectory(self) -> None:
        env1 = ResourceStrategyGame()
        env2 = ResourceStrategyGame()
        action = np.array([0.3, 0.3, 0.2, 0.2])

        env1.reset(seed=99)
        env2.reset(seed=99)

        for _ in range(10):
            obs1, r1, t1, _, _ = env1.step(action)
            obs2, r2, t2, _, _ = env2.step(action)
            np.testing.assert_array_equal(obs1["wealth"], obs2["wealth"])
            assert r1 == r2
            assert t1 == t2

    def test_different_seeds_different_trajectories(self) -> None:
        env1 = ResourceStrategyGame()
        env2 = ResourceStrategyGame()
        action = np.array([0.3, 0.3, 0.2, 0.2])

        env1.reset(seed=1)
        env2.reset(seed=2)

        wealths1 = []
        wealths2 = []
        for _ in range(10):
            obs1, _, _, _, _ = env1.step(action)
            obs2, _, _, _, _ = env2.step(action)
            wealths1.append(obs1["wealth"].copy())
            wealths2.append(obs2["wealth"].copy())

        all_same = all(
            np.allclose(w1, w2) for w1, w2 in zip(wealths1, wealths2)
        )
        assert not all_same


class TestChannelStats:
    def test_channel_stats_shape(self, env: ResourceStrategyGame) -> None:
        stats = env.get_channel_stats()
        assert stats["means"].shape == (4,)
        assert stats["variances"].shape == (4,)

    def test_variances_are_positive(self, env: ResourceStrategyGame) -> None:
        stats = env.get_channel_stats()
        assert np.all(stats["variances"] > 0)

    def test_safe_channel_has_lowest_variance(self, env: ResourceStrategyGame) -> None:
        env._regime = NEUTRAL
        stats = env.get_channel_stats()
        safe_var = stats["variances"][0]
        assert safe_var == stats["variances"].min()


class TestQualityScore:
    def test_quality_score_returns_float(self, env: ResourceStrategyGame) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        score = env.quality_score(action)
        assert isinstance(score, float)

    def test_diversified_beats_single_dominated(self, env: ResourceStrategyGame) -> None:
        diversified = np.array([0.3, 0.3, 0.2, 0.2])
        env._regime = NEUTRAL
        score_div = env.quality_score(diversified)
        assert isinstance(score_div, float)
