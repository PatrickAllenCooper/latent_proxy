from __future__ import annotations

import numpy as np
import pytest

from src.environments.resource_game import ChannelConfig, GameConfig, ResourceStrategyGame
from src.training.synthetic_users import (
    PriorConfig,
    SyntheticUser,
    SyntheticUserSampler,
    UserType,
)


SEED = 42


@pytest.fixture
def default_config() -> GameConfig:
    return GameConfig()


@pytest.fixture
def short_game_config() -> GameConfig:
    """A shorter game for faster test execution."""
    return GameConfig(n_rounds=5, initial_wealth=100.0)


@pytest.fixture
def env(default_config: GameConfig) -> ResourceStrategyGame:
    game = ResourceStrategyGame(config=default_config)
    game.reset(seed=SEED)
    return game


@pytest.fixture
def short_env(short_game_config: GameConfig) -> ResourceStrategyGame:
    game = ResourceStrategyGame(config=short_game_config)
    game.reset(seed=SEED)
    return game


@pytest.fixture
def patient_cautious() -> UserType:
    return UserType(gamma=0.95, alpha=2.0, lambda_=2.5)


@pytest.fixture
def impatient_aggressive() -> UserType:
    return UserType(gamma=0.3, alpha=0.3, lambda_=1.1)


@pytest.fixture
def balanced_user() -> UserType:
    return UserType(gamma=0.6, alpha=1.0, lambda_=1.5)


@pytest.fixture
def sampler() -> SyntheticUserSampler:
    return SyntheticUserSampler(seed=SEED)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)
