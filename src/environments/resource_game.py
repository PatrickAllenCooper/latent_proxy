from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.environments.base import BaseEnvironment

BULL = 0
BEAR = 1
NEUTRAL = 2
REGIME_NAMES = {BULL: "bull", BEAR: "bear", NEUTRAL: "neutral"}


@dataclass
class ChannelConfig:
    """Configuration for a single investment channel."""

    name: str
    mu: float
    sigma: float
    regime_sensitivity: float = 1.0
    correlation_sign: float = 1.0


@dataclass
class GameConfig:
    """Full configuration for the resource strategy game."""

    n_rounds: int = 20
    initial_wealth: float = 1000.0
    channels: list[ChannelConfig] = field(default_factory=lambda: [
        ChannelConfig(name="safe", mu=0.02, sigma=0.01, regime_sensitivity=0.2),
        ChannelConfig(name="growth", mu=0.06, sigma=0.04, regime_sensitivity=0.6),
        ChannelConfig(name="aggressive", mu=0.12, sigma=0.10, regime_sensitivity=1.0),
        ChannelConfig(
            name="volatile", mu=0.08, sigma=0.20,
            regime_sensitivity=1.5, correlation_sign=-1.0,
        ),
    ])
    regime_transition_matrix: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([
            [0.70, 0.15, 0.15],
            [0.15, 0.70, 0.15],
            [0.15, 0.15, 0.70],
        ])
    )
    min_channels: int = 2
    max_bankruptcy_prob: float = 0.05
    bankruptcy_mc_samples: int = 5000
    channel_correlation: float = 0.3

    @property
    def n_channels(self) -> int:
        return len(self.channels)


class ResourceStrategyGame(BaseEnvironment):
    """Sequential resource allocation game over T rounds with K investment channels.

    The player manages a portfolio across channels with different risk-return
    profiles. Returns are regime-dependent (bull/bear/neutral) with Markov
    switching dynamics. Quality floor constraints prevent dominated, undiversified,
    or bankruptcy-risking allocations.

    See README Sections 4.2-4.3 for the full specification.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: GameConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GameConfig()
        self.render_mode = render_mode
        K = self.config.n_channels
        T = self.config.n_rounds

        self.observation_space = spaces.Dict({
            "wealth": spaces.Box(
                low=0.0, high=np.inf, shape=(K,), dtype=np.float64,
            ),
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
        self._wealth = np.full(K, self.config.initial_wealth / K)
        self._round = 0
        self._total_wealth_history = [float(self._wealth.sum())]

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: NDArray[np.floating[Any]]
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

        terminated = self._round >= self.config.n_rounds
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
        channel_stats = self.get_channel_stats()
        means = channel_stats["means"]
        variances = channel_stats["variances"]

        expected_return = float(np.dot(action, means))
        portfolio_variance = float(np.dot(action**2, variances))
        if portfolio_variance <= 0:
            return expected_return
        sharpe = expected_return / np.sqrt(portfolio_variance)
        return float(sharpe)

    def check_quality_floor(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[bool, list[str]]:
        action = self._normalize_action(action)
        violations: list[str] = []

        dom_ok, dom_msg = self._check_dominance(action)
        if not dom_ok:
            violations.append(dom_msg)

        div_ok, div_msg = self._check_diversification(action)
        if not div_ok:
            violations.append(div_msg)

        bank_ok, bank_msg = self._check_bankruptcy(action)
        if not bank_ok:
            violations.append(bank_msg)

        passes = len(violations) == 0
        return passes, violations

    def get_optimal_action(
        self, theta: dict[str, float]
    ) -> NDArray[np.floating[Any]]:
        gamma = theta.get("gamma", 0.95)
        alpha = theta.get("alpha", 1.0)
        lambda_ = theta.get("lambda_", 1.5)

        channel_stats = self.get_channel_stats()
        means = channel_stats["means"]
        variances = channel_stats["variances"]
        K = self.config.n_channels

        effective_horizon = 1.0 / (1.0 - gamma + 1e-6)

        scores = np.zeros(K)
        for i in range(K):
            ce = _certainty_equivalent(
                means[i], variances[i], alpha, lambda_, effective_horizon
            )
            scores[i] = ce

        scores = np.maximum(scores, 0.0)
        total = scores.sum()
        if total <= 0:
            return np.ones(K) / K

        action = scores / total
        return action

    def get_channel_stats(self) -> dict[str, NDArray[np.floating[Any]]]:
        K = self.config.n_channels
        means = np.zeros(K)
        variances = np.zeros(K)

        for i, ch in enumerate(self.config.channels):
            adj_mu, adj_sigma = self._regime_adjusted_params(ch)
            means[i] = adj_mu
            variances[i] = adj_sigma**2

        return {"means": means, "variances": variances}

    # ---- internal methods ----

    def _normalize_action(
        self, action: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        action = np.asarray(action, dtype=np.float64)
        action = np.maximum(action, 0.0)
        total = action.sum()
        if total <= 0:
            return np.ones(self.config.n_channels) / self.config.n_channels
        return action / total

    def _get_obs(self) -> dict[str, Any]:
        channel_stats = self.get_channel_stats()
        market_state = np.stack(
            [channel_stats["means"], channel_stats["variances"]], axis=-1
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

    def _regime_adjusted_params(
        self, channel: ChannelConfig
    ) -> tuple[float, float]:
        sensitivity = channel.regime_sensitivity
        corr_sign = channel.correlation_sign

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

        return channel.mu * mu_mult, channel.sigma * sigma_mult

    def _sample_returns(self) -> NDArray[np.floating[Any]]:
        K = self.config.n_channels
        means = np.zeros(K)
        stds = np.zeros(K)

        for i, ch in enumerate(self.config.channels):
            adj_mu, adj_sigma = self._regime_adjusted_params(ch)
            means[i] = adj_mu
            stds[i] = adj_sigma

        rho = self.config.channel_correlation
        corr_matrix = np.full((K, K), rho)
        np.fill_diagonal(corr_matrix, 1.0)

        cov_matrix = np.outer(stds, stds) * corr_matrix
        cov_matrix = _nearest_positive_definite(cov_matrix)

        returns = self._rng.multivariate_normal(means, cov_matrix)
        return returns

    def _transition_regime(self) -> None:
        probs = self.config.regime_transition_matrix[self._regime]
        self._regime = int(self._rng.choice(3, p=probs))

    def _check_dominance(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[bool, str]:
        channel_stats = self.get_channel_stats()
        means = channel_stats["means"]
        variances = channel_stats["variances"]

        concentrated = np.where(action > 0.99)[0]
        if len(concentrated) != 1:
            return True, ""

        idx = concentrated[0]
        for j in range(self.config.n_channels):
            if j == idx:
                continue
            if means[j] >= means[idx] and variances[j] <= variances[idx]:
                if means[j] > means[idx] or variances[j] < variances[idx]:
                    return False, (
                        f"Channel {idx} ({self.config.channels[idx].name}) is "
                        f"strictly dominated by channel {j} "
                        f"({self.config.channels[j].name})"
                    )
        return True, ""

    def _check_diversification(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[bool, str]:
        active_channels = int(np.sum(action > 0.01))
        if active_channels < self.config.min_channels:
            return False, (
                f"Only {active_channels} active channel(s); "
                f"minimum is {self.config.min_channels}"
            )
        return True, ""

    def _check_bankruptcy(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[bool, str]:
        channel_stats = self.get_channel_stats()
        means = channel_stats["means"]
        variances = channel_stats["variances"]

        port_mean = float(np.dot(action, means))
        port_var = float(np.dot(action**2, variances))

        if port_var <= 0:
            return True, ""

        port_std = np.sqrt(port_var)
        n_samples = self.config.bankruptcy_mc_samples
        sim_returns = self._rng.normal(port_mean, port_std, size=n_samples)
        bankruptcy_prob = float(np.mean(sim_returns <= -1.0))

        if bankruptcy_prob > self.config.max_bankruptcy_prob:
            return False, (
                f"Bankruptcy probability {bankruptcy_prob:.3f} exceeds "
                f"threshold {self.config.max_bankruptcy_prob:.3f}"
            )
        return True, ""


def _certainty_equivalent(
    mu: float, var: float, alpha: float, lambda_: float,
    effective_horizon: float = 1.0,
) -> float:
    """Approximate certainty equivalent under prospect-theory utility.

    Returns grow linearly with horizon (mu * h) but the risk penalty
    scales as sqrt(var * h) -- the standard deviation of cumulative
    returns under i.i.d. assumptions. This sub-linear scaling means
    patient users (large h) see relatively more risk penalty on volatile
    channels compared to their return advantage, producing meaningfully
    different allocations from impatient users.
    """
    compounded_return = mu * effective_horizon
    if alpha <= 0:
        return compounded_return

    cumulative_risk = np.sqrt(max(var * effective_horizon, 0.0))
    risk_penalty = 0.5 * alpha * cumulative_risk
    ce = compounded_return - risk_penalty
    if ce < 0:
        ce *= lambda_
    return ce


def _nearest_positive_definite(
    matrix: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Find the nearest positive-definite matrix via symmetric polar decomposition."""
    sym = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
