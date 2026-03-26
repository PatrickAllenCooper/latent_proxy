from __future__ import annotations

import numpy as np
from src.environments.game_variants import (
    VARIANT_CONFIGS,
    create_variant_a,
    create_variant_b,
    get_variant_config,
    load_game_config_yaml,
)


def test_variant_a_four_channels() -> None:
    env = create_variant_a()
    assert env.config.n_channels == 4
    assert env.config.n_rounds == 20


def test_variant_b_six_channels_fifty_rounds() -> None:
    env = create_variant_b()
    assert env.config.n_channels == 6
    assert env.config.n_rounds == 50
    names = [c.name for c in env.config.channels]
    assert "balanced" in names
    assert "hedge" in names


def test_get_variant_config_matches_factories() -> None:
    ca = get_variant_config("a")
    cb = get_variant_config("b")
    assert ca.n_channels == 4
    assert cb.n_channels == 6


def test_variant_configs_lookup() -> None:
    assert VARIANT_CONFIGS["a"].n_channels == 4
    assert VARIANT_CONFIGS["b"].n_channels == 6


def test_both_pass_quality_floor_on_uniform() -> None:
    for factory in (create_variant_a, create_variant_b):
        env = factory()
        env.reset(seed=42)
        uniform = np.ones(env.config.n_channels) / env.config.n_channels
        ok, _ = env.check_quality_floor(uniform)
        assert ok, f"{factory.__name__} uniform should pass quality floor"


def test_optimal_actions_differ_across_variants() -> None:
    theta = {"gamma": 0.75, "alpha": 1.0, "lambda_": 1.5}
    env_a = create_variant_a()
    env_b = create_variant_b()
    env_a.reset(seed=1)
    env_b.reset(seed=1)
    a_a = env_a.get_optimal_action(theta)
    a_b = env_b.get_optimal_action(theta)
    assert a_a.shape[0] == 4 and a_b.shape[0] == 6
    assert np.isclose(a_a.sum(), 1.0) and np.isclose(a_b.sum(), 1.0)


def test_load_default_yaml() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "game" / "default.yaml"
    cfg = load_game_config_yaml(path)
    assert cfg.n_channels == 4
