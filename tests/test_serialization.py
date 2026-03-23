from __future__ import annotations

import numpy as np
import pytest

from src.environments.resource_game import GameConfig, ResourceStrategyGame
from src.training.serialization import (
    AllocationSerializer,
    GameStateSerializer,
    UserProfileSerializer,
    build_prompt,
)
from src.training.synthetic_users import UserType


@pytest.fixture
def env() -> ResourceStrategyGame:
    game = ResourceStrategyGame()
    game.reset(seed=42)
    return game


@pytest.fixture
def obs(env: ResourceStrategyGame) -> dict:
    return env._get_obs()


class TestGameStateSerializer:
    def test_contains_round_info(self, obs: dict, env: ResourceStrategyGame) -> None:
        gs = GameStateSerializer()
        text = gs.serialize(obs, env)
        assert "round 1 of 20" in text

    def test_contains_wealth(self, obs: dict, env: ResourceStrategyGame) -> None:
        gs = GameStateSerializer()
        text = gs.serialize(obs, env)
        assert "$" in text
        assert "Safe" in text
        assert "Growth" in text
        assert "Aggressive" in text
        assert "Volatile" in text

    def test_contains_market_conditions(
        self, obs: dict, env: ResourceStrategyGame
    ) -> None:
        gs = GameStateSerializer()
        text = gs.serialize(obs, env)
        assert "expected return" in text
        assert "volatility" in text
        assert "regime" in text

    def test_contains_instruction(self, obs: dict, env: ResourceStrategyGame) -> None:
        gs = GameStateSerializer()
        text = gs.serialize(obs, env, include_instruction=True)
        assert "Recommend" in text
        assert "100%" in text

    def test_no_instruction(self, obs: dict, env: ResourceStrategyGame) -> None:
        gs = GameStateSerializer()
        text = gs.serialize(obs, env, include_instruction=False)
        assert "Recommend" not in text

    def test_after_steps(self, env: ResourceStrategyGame) -> None:
        action = np.array([0.25, 0.25, 0.25, 0.25])
        for _ in range(5):
            env.step(action)
        obs = env._get_obs()
        gs = GameStateSerializer()
        text = gs.serialize(obs, env)
        assert "round 6 of 20" in text

    def test_zero_wealth_handled(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=0)
        env._wealth = np.zeros(4)
        obs = env._get_obs()
        gs = GameStateSerializer()
        text = gs.serialize(obs, env)
        assert "$0.00" in text


class TestUserProfileSerializer:
    def test_patient_cautious(self) -> None:
        ups = UserProfileSerializer()
        ut = UserType(gamma=0.95, alpha=2.0, lambda_=2.5)
        text = ups.serialize(ut)
        assert "Long-term" in text or "Very long-term" in text
        assert "0.95" in text
        assert "2.00" in text
        assert "2.50" in text

    def test_impatient_aggressive(self) -> None:
        ups = UserProfileSerializer()
        ut = UserType(gamma=0.2, alpha=0.3, lambda_=1.1)
        text = ups.serialize(ut)
        assert "Short-term" in text or "Very short-term" in text
        assert "High" in text

    def test_all_fields_present(self) -> None:
        ups = UserProfileSerializer()
        ut = UserType(gamma=0.6, alpha=1.0, lambda_=1.5)
        text = ups.serialize(ut)
        assert "Time horizon" in text
        assert "Risk tolerance" in text
        assert "Loss sensitivity" in text
        assert "discount factor" in text.lower() or "0.60" in text

    def test_boundary_gamma(self) -> None:
        ups = UserProfileSerializer()
        ut = UserType(gamma=1.0, alpha=0.0, lambda_=1.0)
        text = ups.serialize(ut)
        assert "1.00" in text


class TestAllocationSerializer:
    def test_serialize(self) -> None:
        alloc = AllocationSerializer()
        action = np.array([0.3, 0.3, 0.2, 0.2])
        text = alloc.serialize(action)
        assert "Safe: 30%" in text
        assert "Growth: 30%" in text
        assert "Aggressive: 20%" in text
        assert "Volatile: 20%" in text

    def test_parse_standard_format(self) -> None:
        alloc = AllocationSerializer()
        text = "Safe: 30%\nGrowth: 30%\nAggressive: 20%\nVolatile: 20%"
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, [0.3, 0.3, 0.2, 0.2], atol=0.01)

    def test_round_trip(self) -> None:
        alloc = AllocationSerializer()
        original = np.array([0.35, 0.25, 0.25, 0.15])
        text = alloc.serialize(original)
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, original, atol=0.02)

    def test_parse_with_equals(self) -> None:
        alloc = AllocationSerializer()
        text = "Safe = 40%\nGrowth = 30%\nAggressive = 20%\nVolatile = 10%"
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, [0.4, 0.3, 0.2, 0.1], atol=0.01)

    def test_parse_normalizes(self) -> None:
        alloc = AllocationSerializer()
        text = "Safe: 50%\nGrowth: 50%\nAggressive: 50%\nVolatile: 50%"
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed.sum(), 1.0)

    def test_parse_fallback_on_garbage(self) -> None:
        alloc = AllocationSerializer()
        text = "I recommend nothing specific."
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, [0.25, 0.25, 0.25, 0.25])

    def test_parse_percentage_only(self) -> None:
        alloc = AllocationSerializer()
        text = "Allocate 40%, 30%, 20%, 10% respectively."
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed.sum(), 1.0)
        assert parsed[0] > parsed[-1]

    def test_unnormalized_input(self) -> None:
        alloc = AllocationSerializer()
        action = np.array([2.0, 3.0, 1.0, 4.0])
        text = alloc.serialize(action)
        assert "%" in text
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed.sum(), 1.0)

    def test_custom_channel_names(self) -> None:
        alloc = AllocationSerializer(["bonds", "stocks", "crypto"])
        action = np.array([0.5, 0.3, 0.2])
        text = alloc.serialize(action)
        assert "Bonds: 50%" in text
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, [0.5, 0.3, 0.2], atol=0.02)


class TestBuildPrompt:
    def test_without_user_type(self, obs: dict, env: ResourceStrategyGame) -> None:
        prompt = build_prompt(obs, env, user_type=None)
        assert "round" in prompt
        assert "preference profile" not in prompt

    def test_with_user_type(self, obs: dict, env: ResourceStrategyGame) -> None:
        ut = UserType(gamma=0.9, alpha=1.5, lambda_=2.0)
        prompt = build_prompt(obs, env, user_type=ut)
        assert "round" in prompt
        assert "preference profile" in prompt
        assert "0.90" in prompt
