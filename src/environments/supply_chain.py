from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from omegaconf import OmegaConf

from src.environments.base import BaseEnvironment
from src.environments.env_utils import certainty_equivalent, nearest_positive_definite
from src.environments.resource_game import BULL, BEAR, NEUTRAL, REGIME_NAMES

SUPPLY_REGIME_NAMES = {BULL: "normal", BEAR: "disrupted", NEUTRAL: "crisis"}


@dataclass
class SupplierConfig:
    """Single supplier in the procurement environment."""

    name: str
    sector: str = ""
    cost_per_unit: float = 0.06
    delivery_variance: float = 0.05
    reliability_rating: float = 0.90
    regime_sensitivity: float = 0.5
    correlation_sign: float = 1.0


@dataclass
class SupplyChainConfig:
    """Configuration for the supply chain procurement environment."""

    n_periods: int = 60
    initial_budget: float = 500_000.0
    suppliers: list[SupplierConfig] = field(default_factory=lambda: [
        SupplierConfig(
            name="premium_domestic", sector="domestic",
            cost_per_unit=0.03, delivery_variance=0.02,
            reliability_rating=0.98, regime_sensitivity=0.1,
        ),
        SupplierConfig(
            name="standard_domestic", sector="domestic",
            cost_per_unit=0.06, delivery_variance=0.05,
            reliability_rating=0.92, regime_sensitivity=0.5,
        ),
        SupplierConfig(
            name="overseas_budget", sector="international",
            cost_per_unit=0.12, delivery_variance=0.12,
            reliability_rating=0.78, regime_sensitivity=1.0,
        ),
        SupplierConfig(
            name="specialty_niche", sector="specialty",
            cost_per_unit=0.08, delivery_variance=0.10,
            reliability_rating=0.85, regime_sensitivity=0.8,
            correlation_sign=-0.3,
        ),
        SupplierConfig(
            name="spot_market", sector="commodities",
            cost_per_unit=0.10, delivery_variance=0.18,
            reliability_rating=0.70, regime_sensitivity=1.3,
            correlation_sign=0.5,
        ),
    ])
    regime_transition_matrix: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([
            [0.75, 0.15, 0.10],
            [0.20, 0.60, 0.20],
            [0.15, 0.15, 0.70],
        ])
    )
    min_suppliers: int = 2
    max_single_supplier_share: float = 0.70
    resilience_mc_samples: int = 5000
    resilience_threshold: float = -0.40
    max_disruption_prob: float = 0.10
    channel_correlation: float = 0.20

    @property
    def n_channels(self) -> int:
        return len(self.suppliers)

    @property
    def n_rounds(self) -> int:
        return self.n_periods


class SupplyChainEnv(BaseEnvironment):
    """Multi-supplier procurement with regime switching (README Section 8.2, domain 3).

    Observations share the same layout as game and stock environments (wealth,
    market_state, round) so the full elicitation/evaluation stack works unchanged.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: SupplyChainConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SupplyChainConfig()
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
        self._regime: int = BULL
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
        self._regime = BULL
        self._wealth = np.full(K, self.config.initial_budget / K)
        self._round = 0
        self._total_wealth_history = [float(self._wealth.sum())]
        return self._get_obs(), self._get_info()

    def step(
        self, action: NDArray[np.floating[Any]],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = self._normalize_action(action)
        total_budget = float(self._wealth.sum())
        self._wealth = action * total_budget

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
        expected_savings = float(np.dot(action, means))
        delivery_risk = float(np.dot(action**2, variances))
        if delivery_risk <= 0:
            return expected_savings
        return float(expected_savings / np.sqrt(delivery_risk))

    def check_quality_floor(
        self, action: NDArray[np.floating[Any]],
    ) -> tuple[bool, list[str]]:
        action = self._normalize_action(action)
        violations: list[str] = []

        active = int(np.sum(action > 0.01))
        if active < self.config.min_suppliers:
            violations.append(
                f"Only {active} active supplier(s); minimum is {self.config.min_suppliers}"
            )

        max_share = float(np.max(action))
        if max_share > self.config.max_single_supplier_share:
            idx = int(np.argmax(action))
            violations.append(
                f"Supplier {self.config.suppliers[idx].name} has {max_share:.0%} share, "
                f"exceeding {self.config.max_single_supplier_share:.0%} limit"
            )

        dom_ok, dom_msg = self._check_dominance(action)
        if not dom_ok:
            violations.append(dom_msg)

        tail_ok, tail_msg = self._check_tail_risk(action)
        if not tail_ok:
            violations.append(tail_msg)

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
            action = pos / total
        else:
            shifted = scores - float(np.min(scores)) + 1e-10
            action = shifted / float(np.sum(shifted))

        max_share = self.config.max_single_supplier_share
        min_sup = self.config.min_suppliers
        active = int(np.sum(action > 0.01))
        if active < min_sup:
            order = np.argsort(scores)[::-1]
            action = np.zeros(K)
            for j in range(min(min_sup, K)):
                action[order[j]] = max(scores[order[j]], 1e-6)
            action /= action.sum()

        capped = np.minimum(action, max_share)
        residual = 1.0 - capped.sum()
        if residual > 1e-10:
            below_cap = capped < max_share
            if np.any(below_cap):
                capped[below_cap] += residual * (
                    capped[below_cap] / max(capped[below_cap].sum(), 1e-15)
                )
        capped = np.maximum(capped, 0.0)
        capped_total = capped.sum()
        if capped_total > 1e-15:
            return capped / capped_total
        return np.ones(K) / K

    def get_channel_stats(self) -> dict[str, NDArray[np.floating[Any]]]:
        K = self.config.n_channels
        means = np.zeros(K)
        variances = np.zeros(K)
        for i, supplier in enumerate(self.config.suppliers):
            adj_mu, adj_sigma = self._regime_adjusted_params(supplier)
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
            "regime_name": SUPPLY_REGIME_NAMES[self._regime],
            "round": self._round,
        }

    def _regime_adjusted_params(
        self, supplier: SupplierConfig,
    ) -> tuple[float, float]:
        base_mu = supplier.cost_per_unit
        base_sigma = np.sqrt(supplier.delivery_variance)
        sensitivity = supplier.regime_sensitivity
        corr_sign = supplier.correlation_sign

        if self._regime == BULL:
            mu_mult = 1.0 + 0.4 * sensitivity * corr_sign
            sigma_mult = 1.0 - 0.15 * sensitivity * abs(corr_sign)
        elif self._regime == BEAR:
            mu_mult = 1.0 - 0.4 * sensitivity * corr_sign
            sigma_mult = 1.0 + 0.35 * sensitivity * abs(corr_sign)
        else:
            mu_mult = 1.0 - 0.1 * sensitivity
            sigma_mult = 1.0 + 0.2 * sensitivity

        sigma_mult = max(sigma_mult, 0.1)
        return base_mu * mu_mult, base_sigma * sigma_mult

    def _sample_returns(self) -> NDArray[np.floating[Any]]:
        K = self.config.n_channels
        means = np.zeros(K)
        stds = np.zeros(K)
        for i, supplier in enumerate(self.config.suppliers):
            adj_mu, adj_sigma = self._regime_adjusted_params(supplier)
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
                        f"Supplier {idx} ({self.config.suppliers[idx].name}) "
                        f"dominated by {j} ({self.config.suppliers[j].name})"
                    )
        return True, ""

    def _check_tail_risk(
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
        n_samples = self.config.resilience_mc_samples
        sim_returns = self._rng.normal(port_mean, port_std, size=n_samples)
        thr = self.config.resilience_threshold
        p_disrupt = float(np.mean(sim_returns <= thr))
        if p_disrupt > self.config.max_disruption_prob:
            return False, (
                f"Supply disruption P(r <= {thr:.2f}) = {p_disrupt:.3f} exceeds "
                f"{self.config.max_disruption_prob:.3f}"
            )
        return True, ""


def supply_chain_config_from_yaml_dict(d: dict[str, Any]) -> SupplyChainConfig:
    s = d["supply_chain"]
    suppliers_raw = s["suppliers"]
    suppliers: list[SupplierConfig] = []
    for sup in suppliers_raw:
        suppliers.append(
            SupplierConfig(
                name=str(sup["name"]),
                sector=str(sup.get("sector", "")),
                cost_per_unit=float(sup["cost_per_unit"]),
                delivery_variance=float(sup["delivery_variance"]),
                reliability_rating=float(sup.get("reliability_rating", 0.90)),
                regime_sensitivity=float(sup.get("regime_sensitivity", 0.5)),
                correlation_sign=float(sup.get("correlation_sign", 1.0)),
            )
        )
    matrix = np.asarray(s["regime_transition_matrix"], dtype=np.float64)
    floor = s.get("quality_floor", {})
    return SupplyChainConfig(
        n_periods=int(s["n_periods"]),
        initial_budget=float(s["initial_budget"]),
        suppliers=suppliers,
        regime_transition_matrix=matrix,
        min_suppliers=int(floor.get("min_suppliers", 2)),
        max_single_supplier_share=float(floor.get("max_single_supplier_share", 0.70)),
        resilience_mc_samples=int(floor.get("resilience_mc_samples", 5000)),
        resilience_threshold=float(floor.get("resilience_threshold", -0.40)),
        max_disruption_prob=float(floor.get("max_disruption_prob", 0.10)),
        channel_correlation=float(s.get("channel_correlation", 0.20)),
    )


def load_supply_chain_config_yaml(path: str | Path) -> SupplyChainConfig:
    p = Path(path)
    cfg = OmegaConf.load(p)
    container: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return supply_chain_config_from_yaml_dict(container)


def create_default_supply_chain_env(
    config_path: str | Path | None = None,
) -> SupplyChainEnv:
    if config_path is None:
        root = Path(__file__).resolve().parents[2]
        config_path = root / "configs" / "supply_chain" / "default.yaml"
    cfg = load_supply_chain_config_yaml(config_path)
    return SupplyChainEnv(config=cfg)
