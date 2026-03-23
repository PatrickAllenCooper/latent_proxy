from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UserType:
    """Latent preference parameters for a user.

    Attributes:
        gamma: Discount factor over future outcomes, in (0, 1].
        alpha: Risk aversion coefficient (>= 0). Higher values = more risk averse.
        lambda_: Loss aversion coefficient (>= 1). Ratio of loss sensitivity to gain sensitivity.
    """

    gamma: float
    alpha: float
    lambda_: float

    def __post_init__(self) -> None:
        if not 0 < self.gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.lambda_ < 1:
            raise ValueError(f"lambda_ must be >= 1, got {self.lambda_}")


@dataclass
class PriorConfig:
    """Configuration for the prior distributions over user type parameters."""

    gamma_a: float = 2.0
    gamma_b: float = 2.0
    alpha_mu: float = 0.0
    alpha_sigma: float = 0.5
    lambda_low: float = 1.0
    lambda_high: float = 3.0


class SyntheticUserSampler:
    """Draws user types from configurable prior distributions.

    Prior distributions (from README Section 4.4):
        gamma ~ Beta(a, b)
        alpha ~ LogNormal(mu, sigma)
        lambda_ ~ Uniform(low, high)
    """

    def __init__(
        self,
        prior_config: PriorConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = prior_config or PriorConfig()
        self._rng = np.random.default_rng(seed)

    def sample(self) -> UserType:
        """Draw a single user type from the prior."""
        gamma = float(self._rng.beta(self.config.gamma_a, self.config.gamma_b))
        gamma = np.clip(gamma, 1e-6, 1.0)

        alpha = float(self._rng.lognormal(
            self.config.alpha_mu, self.config.alpha_sigma
        ))

        lambda_ = float(self._rng.uniform(
            self.config.lambda_low, self.config.lambda_high
        ))

        return UserType(gamma=gamma, alpha=alpha, lambda_=lambda_)

    def sample_batch(self, n: int) -> list[UserType]:
        """Draw n user types from the prior."""
        return [self.sample() for _ in range(n)]

    def sample_extreme_types(self) -> dict[str, UserType]:
        """Return canonical extreme user types for testing and validation.

        These well-separated types should produce meaningfully different
        optimal strategies in any well-designed environment.
        """
        return {
            "patient_cautious": UserType(gamma=0.95, alpha=2.0, lambda_=2.5),
            "patient_aggressive": UserType(gamma=0.95, alpha=0.3, lambda_=1.1),
            "impatient_cautious": UserType(gamma=0.3, alpha=2.0, lambda_=2.5),
            "impatient_aggressive": UserType(gamma=0.3, alpha=0.3, lambda_=1.1),
            "balanced": UserType(gamma=0.6, alpha=1.0, lambda_=1.5),
        }


def prospect_utility(
    wealth: float | NDArray[np.floating[Any]],
    alpha: float,
    lambda_: float,
    reference_point: float = 0.0,
) -> float | NDArray[np.floating[Any]]:
    """Prospect-theory value function.

    u(w) = (w - ref)^(1/(1+alpha))              if w >= reference_point
    u(w) = -lambda_ * |w - ref|^(1/(1+alpha))   if w < reference_point

    The exponent 1/(1+alpha) gives concavity for gains (risk aversion)
    and convexity for losses (risk seeking in the loss domain), consistent
    with Kahneman-Tversky prospect theory. Higher alpha = more curvature.
    """
    w = np.asarray(wealth, dtype=np.float64)
    ref = reference_point
    deviation = w - ref

    exponent = 1.0 / (1.0 + alpha)

    gains_mask = deviation >= 0
    result = np.where(
        gains_mask,
        np.power(np.maximum(deviation, 0.0), exponent),
        -lambda_ * np.power(np.maximum(-deviation, 0.0), exponent),
    )

    if np.ndim(wealth) == 0:
        return float(result)
    return result


def discounted_utility(
    terminal_wealth: float,
    theta: UserType,
    rounds_remaining: int,
    reference_point: float = 0.0,
) -> float:
    """Compute discounted prospect-theory utility for terminal wealth."""
    u = prospect_utility(
        terminal_wealth, theta.alpha, theta.lambda_, reference_point
    )
    discount = theta.gamma ** rounds_remaining
    return float(u) * discount


class SyntheticUser:
    """A simulated user with fixed preference parameters.

    Responds to choice queries using softmax-rational decision making:
    P(choose A) = sigma((EU_A - EU_B) / tau)
    """

    def __init__(
        self,
        user_type: UserType,
        temperature: float = 0.1,
        reference_point: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.user_type = user_type
        self.temperature = temperature
        self.reference_point = reference_point
        self._rng = np.random.default_rng(seed)

    @property
    def theta(self) -> UserType:
        return self.user_type

    def evaluate_outcome(self, wealth: float, rounds_remaining: int = 0) -> float:
        """Compute this user's utility for a given terminal wealth."""
        return discounted_utility(
            wealth, self.user_type, rounds_remaining, self.reference_point
        )

    def evaluate_allocation(
        self,
        allocation: NDArray[np.floating[Any]],
        channel_means: NDArray[np.floating[Any]],
        channel_variances: NDArray[np.floating[Any]],
        current_wealth: float,
        rounds_remaining: int = 1,
    ) -> float:
        """Score an allocation under this user's utility using Monte Carlo.

        Simulates outcomes from a single-period return model and averages
        the discounted prospect-theory utility.
        """
        allocation = np.asarray(allocation, dtype=np.float64)
        n_samples = 1000

        port_mean = float(np.dot(allocation, channel_means))
        port_var = float(np.dot(allocation**2, channel_variances))
        port_std = max(np.sqrt(port_var), 1e-10)

        sim_returns = self._rng.normal(port_mean, port_std, size=n_samples)
        sim_wealth = current_wealth * (1.0 + sim_returns)

        utilities = prospect_utility(
            sim_wealth, self.user_type.alpha,
            self.user_type.lambda_, self.reference_point,
        )
        discount = self.user_type.gamma ** rounds_remaining
        return float(np.mean(utilities)) * discount

    def choose(
        self,
        utility_a: float,
        utility_b: float,
    ) -> int:
        """Softmax-rational binary choice between two options.

        Returns 0 for option A, 1 for option B.
        """
        diff = (utility_a - utility_b) / max(self.temperature, 1e-10)
        diff = np.clip(diff, -500, 500)
        prob_a = 1.0 / (1.0 + np.exp(-diff))
        return 0 if self._rng.random() < prob_a else 1

    def choose_allocation(
        self,
        allocations: list[NDArray[np.floating[Any]]],
        channel_means: NDArray[np.floating[Any]],
        channel_variances: NDArray[np.floating[Any]],
        current_wealth: float,
        rounds_remaining: int = 1,
    ) -> int:
        """Choose among multiple allocations using softmax over expected utilities."""
        utilities = np.array([
            self.evaluate_allocation(
                a, channel_means, channel_variances,
                current_wealth, rounds_remaining,
            )
            for a in allocations
        ])

        scaled = (utilities - utilities.max()) / max(self.temperature, 1e-10)
        scaled = np.clip(scaled, -500, 500)
        exp_scaled = np.exp(scaled)
        probs = exp_scaled / exp_scaled.sum()

        return int(self._rng.choice(len(allocations), p=probs))
