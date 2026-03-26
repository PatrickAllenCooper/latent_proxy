from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.environments.base import BaseEnvironment
from src.environments.env_utils import certainty_equivalent, nearest_positive_definite
from src.environments.resource_game import BULL, BEAR, NEUTRAL, REGIME_NAMES

PERIODS_PER_YEAR = 252


@dataclass
class AssetConfig:
    """Single asset class in the backtest (e.g. US large cap)."""

    name: str
    sector: str = ""
    annual_mu: float = 0.08
    annual_sigma: float = 0.16
    regime_sensitivity: float = 1.0
    correlation_sign: float = 1.0

    def per_period_mu_sigma(self) -> tuple[float, float]:
        mu_p = self.annual_mu / PERIODS_PER_YEAR
        sigma_p = self.annual_sigma / np.sqrt(PERIODS_PER_YEAR)
        return float(mu_p), float(sigma_p)


@dataclass
class StockBacktestConfig:
    """Configuration for multi-asset stock backtest environment."""

    n_periods: int = 252
    initial_capital: float = 100_000.0
    assets: list[AssetConfig] = field(default_factory=lambda: [
        AssetConfig(
            name="us_treasury", sector="fixed_income",
            annual_mu=0.03, annual_sigma=0.04, regime_sensitivity=0.1,
        ),
        AssetConfig(
            name="us_large_cap", sector="equity",
            annual_mu=0.10, annual_sigma=0.16, regime_sensitivity=0.8,
        ),
        AssetConfig(
            name="us_small_cap", sector="equity",
            annual_mu=0.12, annual_sigma=0.22, regime_sensitivity=1.0,
        ),
        AssetConfig(
            name="international", sector="equity",
            annual_mu=0.08, annual_sigma=0.18, regime_sensitivity=0.9,
            correlation_sign=-0.3,
        ),
        AssetConfig(
            name="reits", sector="real_estate",
            annual_mu=0.09, annual_sigma=0.20, regime_sensitivity=1.2,
            correlation_sign=0.5,
        ),
    ])
    regime_transition_matrix: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([
            [0.70, 0.15, 0.15],
            [0.15, 0.70, 0.15],
            [0.15, 0.15, 0.70],
        ])
    )
    min_assets: int = 2
    max_catastrophic_loss_prob: float = 0.10
    catastrophic_return_threshold: float = -0.50
    drawdown_mc_samples: int = 5000
    channel_correlation: float = 0.25
    rebalance_frequency: int = 1

    @property
    def n_channels(self) -> int:
        return len(self.assets)

    @property
    def n_rounds(self) -> int:
        return self.n_periods


class StockBacktestEnv(BaseEnvironment):
    """Multi-asset portfolio backtest with regime switching (README Milestone 5).

    Observations match the resource game layout (wealth, market_state, round)
    so elicitation and evaluation code can share the same tensor shapes.
    Per-period returns use annual parameters scaled to daily-equivalent steps.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: StockBacktestConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or StockBacktestConfig()
        self.render_mode = render_mode
        K = self.config.n_channels
        T = self.config.n_periods

        self.observation_space = spaces.Dict({
            "wealth": spaces.Box(low=0.0, high=np.inf, shape=(K,), dtype=np.float64),
            "market_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(K, 2), dtype=np.float64,
            ),
            "round": spaces.Discrete(T),
        })
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(K,), dtype=np.float64,
        )

        self._rng: np.random.Generator = np.random.default_rng()
        self._regime: int = NEUTRAL
        self._wealth: NDArray[np.floating[Any]] = np.zeros(K)
        self._round: int = 0
        self._total_wealth_history: list[float] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        K = self.config.n_channels
        self._regime = NEUTRAL
        self._wealth = np.full(K, self.config.initial_capital / K)
        self._round = 0
        self._total_wealth_history = [float(self._wealth.sum())]
        return self._get_obs(), self._get_info()

    def step(
        self, action: NDArray[np.floating[Any]],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = self._normalize_action(action)
        total_wealth = float(self._wealth.sum())
        self._wealth = action * total_wealth

        returns = self._sample_returns()
        self._wealth = self._wealth * (1.0 + returns)
        self._wealth = np.maximum(self._wealth, 0.0)

        self._transition_regime()
        self._round += 1
        self._total_wealth_history.append(float(self._wealth.sum()))

        terminated = self._round >= self.config.n_periods
        truncated = False
        reward = 0.0
        if terminated:
            reward = float(self._wealth.sum())

        obs = self._get_obs()
        info = self._get_info()
        info["returns"] = returns
        info["regime"] = self._regime
        return obs, reward, terminated, truncated, info

    def quality_score(self, action: NDArray[np.floating[Any]]) -> float:
        action = self._normalize_action(action)
        stats = self.get_channel_stats()
        means = stats["means"]
        variances = stats["variances"]
        expected_return = float(np.dot(action, means))
        portfolio_variance = float(np.dot(action**2, variances))
        if portfolio_variance <= 0:
            return expected_return
        return float(expected_return / np.sqrt(portfolio_variance))

    def check_quality_floor(
        self, action: NDArray[np.floating[Any]],
    ) -> tuple[bool, list[str]]:
        action = self._normalize_action(action)
        violations: list[str] = []

        dom_ok, dom_msg = self._check_dominance(action)
        if not dom_ok:
            violations.append(dom_msg)

        div_ok, div_msg = self._check_diversification(action)
        if not div_ok:
            violations.append(div_msg)

        cat_ok, cat_msg = self._check_catastrophic_tail(action)
        if not cat_ok:
            violations.append(cat_msg)

        return len(violations) == 0, violations

    def get_optimal_action(
        self, theta: dict[str, float],
    ) -> NDArray[np.floating[Any]]:
        gamma = theta.get("gamma", 0.95)
        alpha = theta.get("alpha", 1.0)
        lambda_ = theta.get("lambda_", 1.5)

        stats = self.get_channel_stats()
        means = stats["means"]
        variances = stats["variances"]
        K = self.config.n_channels

        effective_horizon = 1.0 / (1.0 - gamma + 1e-6)

        scores = np.zeros(K)
        for i in range(K):
            scores[i] = certainty_equivalent(
                means[i], variances[i], alpha, lambda_, effective_horizon,
            )
        pos = np.maximum(scores, 0.0)
        total = pos.sum()
        if total > 1e-15:
            return pos / total
        shifted = scores - float(np.min(scores)) + 1e-10
        return shifted / float(np.sum(shifted))

    def get_channel_stats(self) -> dict[str, NDArray[np.floating[Any]]]:
        K = self.config.n_channels
        means = np.zeros(K)
        variances = np.zeros(K)
        for i, asset in enumerate(self.config.assets):
            adj_mu, adj_sigma = self._regime_adjusted_params(asset)
            means[i] = adj_mu
            variances[i] = adj_sigma**2
        return {"means": means, "variances": variances}

    def _normalize_action(
        self, action: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        action = np.asarray(action, dtype=np.float64)
        action = np.maximum(action, 0.0)
        total = action.sum()
        if total <= 0:
            return np.ones(self.config.n_channels) / self.config.n_channels
        return action / total

    def _get_obs(self) -> dict[str, Any]:
        stats = self.get_channel_stats()
        market_state = np.stack(
            [stats["means"], stats["variances"]], axis=-1,
        )
        return {
            "wealth": self._wealth.copy(),
            "market_state": market_state,
            "round": self._round,
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "total_wealth": float(self._wealth.sum()),
            "regime": self._regime,
            "regime_name": REGIME_NAMES[self._regime],
            "round": self._round,
        }

    def _regime_adjusted_params(self, asset: AssetConfig) -> tuple[float, float]:
        base_mu, base_sigma = asset.per_period_mu_sigma()
        sensitivity = asset.regime_sensitivity
        corr_sign = asset.correlation_sign

        if self._regime == BULL:
            mu_mult = 1.0 + 0.5 * sensitivity * corr_sign
            sigma_mult = 1.0 - 0.2 * sensitivity * abs(corr_sign)
        elif self._regime == BEAR:
            mu_mult = 1.0 - 0.5 * sensitivity * corr_sign
            sigma_mult = 1.0 + 0.4 * sensitivity * abs(corr_sign)
        else:
            mu_mult = 1.0
            sigma_mult = 1.0

        sigma_mult = max(sigma_mult, 0.1)
        return base_mu * mu_mult, base_sigma * sigma_mult

    def _sample_returns(self) -> NDArray[np.floating[Any]]:
        K = self.config.n_channels
        means = np.zeros(K)
        stds = np.zeros(K)
        for i, asset in enumerate(self.config.assets):
            adj_mu, adj_sigma = self._regime_adjusted_params(asset)
            means[i] = adj_mu
            stds[i] = adj_sigma

        rho = self.config.channel_correlation
        corr_matrix = np.full((K, K), rho)
        np.fill_diagonal(corr_matrix, 1.0)
        cov_matrix = np.outer(stds, stds) * corr_matrix
        cov_matrix = nearest_positive_definite(cov_matrix)
        return self._rng.multivariate_normal(means, cov_matrix)

    def _transition_regime(self) -> None:
        probs = self.config.regime_transition_matrix[self._regime]
        self._regime = int(self._rng.choice(3, p=probs))

    def _check_dominance(
        self, action: NDArray[np.floating[Any]],
    ) -> tuple[bool, str]:
        stats = self.get_channel_stats()
        means = stats["means"]
        variances = stats["variances"]
        concentrated = np.where(action > 0.99)[0]
        if len(concentrated) != 1:
            return True, ""
        idx = int(concentrated[0])
        for j in range(self.config.n_channels):
            if j == idx:
                continue
            if means[j] >= means[idx] and variances[j] <= variances[idx]:
                if means[j] > means[idx] or variances[j] < variances[idx]:
                    return False, (
                        f"Asset {idx} ({self.config.assets[idx].name}) is strictly "
                        f"dominated by asset {j} ({self.config.assets[j].name})"
                    )
        return True, ""

    def _check_diversification(
        self, action: NDArray[np.floating[Any]],
    ) -> tuple[bool, str]:
        active = int(np.sum(action > 0.01))
        if active < self.config.min_assets:
            return False, (
                f"Only {active} active asset(s); minimum is {self.config.min_assets}"
            )
        return True, ""

    def _check_catastrophic_tail(
        self, action: NDArray[np.floating[Any]],
    ) -> tuple[bool, str]:
        stats = self.get_channel_stats()
        means = stats["means"]
        variances = stats["variances"]
        port_mean = float(np.dot(action, means))
        port_var = float(np.dot(action**2, variances))
        if port_var <= 0:
            return True, ""
        port_std = np.sqrt(port_var)
        n_samples = self.config.drawdown_mc_samples
        sim_returns = self._rng.normal(port_mean, port_std, size=n_samples)
        thr = self.config.catastrophic_return_threshold
        p_cat = float(np.mean(sim_returns <= thr))
        if p_cat > self.config.max_catastrophic_loss_prob:
            return False, (
                f"Catastrophic loss P(r <= {thr:.2f}) = {p_cat:.3f} exceeds "
                f"{self.config.max_catastrophic_loss_prob:.3f}"
            )
        return True, ""


def stock_config_from_yaml_dict(d: dict[str, Any]) -> StockBacktestConfig:
    """Build StockBacktestConfig from YAML-loaded dict (key ``stock``)."""
    s = d["stock"]
    assets_raw = s["assets"]
    assets: list[AssetConfig] = []
    for a in assets_raw:
        assets.append(
            AssetConfig(
                name=str(a["name"]),
                sector=str(a.get("sector", "")),
                annual_mu=float(a["annual_mu"]),
                annual_sigma=float(a["annual_sigma"]),
                regime_sensitivity=float(a.get("regime_sensitivity", 1.0)),
                correlation_sign=float(a.get("correlation_sign", 1.0)),
            )
        )
    matrix = np.asarray(s["regime_transition_matrix"], dtype=np.float64)
    return StockBacktestConfig(
        n_periods=int(s["n_periods"]),
        initial_capital=float(s["initial_capital"]),
        assets=assets,
        regime_transition_matrix=matrix,
        min_assets=int(s.get("quality_floor", {}).get("min_assets", 2)),
        max_catastrophic_loss_prob=float(
            s.get("quality_floor", {}).get("max_catastrophic_loss_prob", 0.10)
        ),
        catastrophic_return_threshold=float(
            s.get("quality_floor", {}).get("catastrophic_return_threshold", -0.50)
        ),
        drawdown_mc_samples=int(
            s.get("quality_floor", {}).get("drawdown_mc_samples", 5000)
        ),
        channel_correlation=float(s.get("channel_correlation", 0.25)),
        rebalance_frequency=int(s.get("rebalance_frequency", 1)),
    )


def load_stock_config_yaml(path: str | Path) -> StockBacktestConfig:
    p = Path(path)
    cfg = OmegaConf.load(p)
    container: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return stock_config_from_yaml_dict(container)


def create_default_stock_env(config_path: str | Path | None = None) -> StockBacktestEnv:
    if config_path is None:
        root = Path(__file__).resolve().parents[2]
        config_path = root / "configs" / "stock" / "default.yaml"
    cfg = load_stock_config_yaml(config_path)
    return StockBacktestEnv(config=cfg)
