from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.environments.resource_game import (
    ChannelConfig,
    GameConfig,
    ResourceStrategyGame,
)


def _as_float_matrix(rows: list[list[Any]]) -> NDArray[np.floating[Any]]:
    return np.asarray(rows, dtype=np.float64)


def game_config_from_yaml_dict(d: dict[str, Any]) -> GameConfig:
    """Build a GameConfig from a YAML-loaded ``game`` section dict."""
    g = d["game"]
    channels_raw = g["channels"]
    channels: list[ChannelConfig] = []
    for c in channels_raw:
        channels.append(
            ChannelConfig(
                name=str(c["name"]),
                mu=float(c["mu"]),
                sigma=float(c["sigma"]),
                regime_sensitivity=float(c.get("regime_sensitivity", 1.0)),
                correlation_sign=float(c.get("correlation_sign", 1.0)),
            )
        )

    qf = g.get("quality_floor", {})
    matrix = _as_float_matrix(g["regime_transition_matrix"])

    return GameConfig(
        n_rounds=int(g["n_rounds"]),
        initial_wealth=float(g["initial_wealth"]),
        channels=channels,
        regime_transition_matrix=matrix,
        min_channels=int(qf.get("min_channels", 2)),
        max_bankruptcy_prob=float(qf.get("max_bankruptcy_prob", 0.05)),
        bankruptcy_mc_samples=int(qf.get("bankruptcy_mc_samples", 5000)),
        channel_correlation=float(g.get("channel_correlation", 0.3)),
    )


def load_game_config_yaml(path: str | Path) -> GameConfig:
    """Load game configuration from a YAML file (same schema as ``default.yaml``)."""
    p = Path(path)
    cfg = OmegaConf.load(p)
    container: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return game_config_from_yaml_dict(container)


def _package_config(name: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "configs" / "game" / name


VARIANT_CONFIG_PATHS: dict[str, Path] = {
    "a": _package_config("default.yaml"),
    "b": _package_config("variant_b.yaml"),
    "default": _package_config("default.yaml"),
}

_VARIANT_ALIASES: dict[str, str] = {
    "variant_a": "a",
    "variant_b": "b",
    "default": "default",
}


def create_variant_a(config_path: str | Path | None = None) -> ResourceStrategyGame:
    """Four-channel, T=20 style game (default config)."""
    path = Path(config_path) if config_path else VARIANT_CONFIG_PATHS["a"]
    cfg = load_game_config_yaml(path)
    return ResourceStrategyGame(config=cfg)


def create_variant_b(config_path: str | Path | None = None) -> ResourceStrategyGame:
    """Six-channel, T=50 transfer target variant."""
    path = Path(config_path) if config_path else VARIANT_CONFIG_PATHS["b"]
    cfg = load_game_config_yaml(path)
    return ResourceStrategyGame(config=cfg)


def get_variant_config(name: str) -> GameConfig:
    """Resolve a variant label to a ``GameConfig`` (without constructing an env)."""
    key = _VARIANT_ALIASES.get(name, name)
    if key not in VARIANT_CONFIG_PATHS:
        key = "default"
    return load_game_config_yaml(VARIANT_CONFIG_PATHS[key])


VARIANT_CONFIGS: dict[str, GameConfig] = {
    "a": load_game_config_yaml(VARIANT_CONFIG_PATHS["a"]),
    "b": load_game_config_yaml(VARIANT_CONFIG_PATHS["b"]),
    "default": load_game_config_yaml(VARIANT_CONFIG_PATHS["default"]),
}
