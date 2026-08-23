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


_VALID_REFERENCE_POINT_MODES = ("zero", "current_wealth")
_VALID_UTILITY_FORMS = ("absolute", "return_normalized")


def validate_utility_axes(reference_point_mode: str, utility_form: str) -> None:
    """Validate the (reference_point_mode, utility_form) combination.

    In return space the only sensible reference is 0.0 ("no change from
    status quo") -- a dollar-valued reference_point="current_wealth" is a
    type mismatch against a fractional return domain, so that combination
    is rejected rather than silently reinterpreted.
    """
    if reference_point_mode not in _VALID_REFERENCE_POINT_MODES:
        raise ValueError(
            f"Unknown reference_point_mode: {reference_point_mode!r} "
            f"(expected one of {_VALID_REFERENCE_POINT_MODES})"
        )
    if utility_form not in _VALID_UTILITY_FORMS:
        raise ValueError(
            f"Unknown utility_form: {utility_form!r} "
            f"(expected one of {_VALID_UTILITY_FORMS})"
        )
    if utility_form == "return_normalized" and reference_point_mode == "current_wealth":
        raise ValueError(
            "utility_form='return_normalized' is incompatible with "
            "reference_point_mode='current_wealth': in return space the "
            "reference is always 0.0 (no change from status quo); a "
            "dollar-valued reference point does not apply."
        )


def resolve_reference_point(
    current_wealth: float,
    reference_point_mode: str,
    utility_form: str = "absolute",
) -> float:
    """Derive the concrete reference point for a scenario's current_wealth.

    Validates the (reference_point_mode, utility_form) combination first.
    Under "return_normalized", the reference is always 0.0 regardless of
    reference_point_mode (see validate_utility_axes).
    """
    validate_utility_axes(reference_point_mode, utility_form)
    if utility_form == "return_normalized":
        return 0.0
    if reference_point_mode == "current_wealth":
        return float(current_wealth)
    return 0.0


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
        utility_form: str = "absolute",
    ) -> None:
        if utility_form not in _VALID_UTILITY_FORMS:
            raise ValueError(
                f"Unknown utility_form: {utility_form!r} "
                f"(expected one of {_VALID_UTILITY_FORMS})"
            )
        self.user_type = user_type
        self.temperature = temperature
        self.reference_point = reference_point
        self.utility_form = utility_form
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
        reference_point: float | None = None,
    ) -> float:
        """Score an allocation under this user's utility using Monte Carlo.

        Simulates outcomes from a single-period return model and averages
        the discounted prospect-theory utility.

        Args:
            reference_point: Optional per-call override for the prospect-theory
                reference point (e.g. the scenario's current wealth so negative
                returns register as losses). Uses ``self.reference_point``
                when None, preserving prior behavior.
        """
        allocation = np.asarray(allocation, dtype=np.float64)
        n_samples = 400
        ref = self.reference_point if reference_point is None else float(reference_point)

        port_mean = float(np.dot(allocation, channel_means))
        port_var = float(np.dot(allocation**2, channel_variances))
        port_std = max(np.sqrt(port_var), 1e-10)

        sim_returns = self._rng.normal(port_mean, port_std, size=n_samples)

        if self.utility_form == "return_normalized":
            outcome = sim_returns
        else:
            outcome = current_wealth * (1.0 + sim_returns)

        utilities = prospect_utility(
            outcome, self.user_type.alpha,
            self.user_type.lambda_, ref,
        )
        discount = self.user_type.gamma ** rounds_remaining
        return float(np.mean(utilities)) * discount

    def evaluate_allocation_multiperiod(
        self,
        allocation: NDArray[np.floating[Any]],
        channel_means: NDArray[np.floating[Any]],
        channel_variances: NDArray[np.floating[Any]],
        current_wealth: float,
        n_periods: int,
        n_samples: int = 256,
        reference_point: float | None = None,
    ) -> float:
        """Monte Carlo expected utility over multiple compounding rounds.

        For each simulated path, wealth compounds with i.i.d. portfolio returns.
        Per-round utility uses prospect theory on end-of-period wealth; each
        period is discounted by gamma**t so patience changes the relative
        value of stable vs volatile allocation paths (gamma is identifiable).

        Args:
            reference_point: Optional per-call override for the prospect-theory
                reference point. Uses ``self.reference_point`` when None.
        """
        if n_periods < 1:
            raise ValueError(f"n_periods must be >= 1, got {n_periods}")

        allocation = np.asarray(allocation, dtype=np.float64)
        port_mean = float(np.dot(allocation, channel_means))
        port_var = float(np.dot(allocation**2, channel_variances))
        port_std = max(np.sqrt(port_var), 1e-10)

        gamma = self.user_type.gamma
        alpha = self.user_type.alpha
        lambda_ = self.user_type.lambda_
        ref = self.reference_point if reference_point is None else float(reference_point)

        returns = self._rng.normal(
            port_mean, port_std, size=(n_samples, n_periods),
        )
        path_u = np.zeros(n_samples, dtype=np.float64)

        if self.utility_form == "return_normalized":
            # Each period is normalized by its own preceding wealth (i.e.
            # scored directly on that period's return), not by the path's
            # initial wealth -- avoids reintroducing a scale artifact across
            # periods within one compounding path (see plan rationale).
            for t in range(n_periods):
                u_step = prospect_utility(returns[:, t], alpha, lambda_, ref)
                path_u += (gamma**t) * np.asarray(u_step, dtype=np.float64)
        else:
            wealth = np.full(n_samples, current_wealth, dtype=np.float64)
            for t in range(n_periods):
                wealth *= 1.0 + returns[:, t]
                u_step = prospect_utility(wealth, alpha, lambda_, ref)
                path_u += (gamma**t) * np.asarray(u_step, dtype=np.float64)

        return float(np.mean(path_u))

    def evaluate_for_query(
        self,
        allocation: NDArray[np.floating[Any]],
        channel_means: NDArray[np.floating[Any]],
        channel_variances: NDArray[np.floating[Any]],
        current_wealth: float,
        rounds_remaining: int,
        multiperiod_horizon: int | None = None,
        reference_point: float | None = None,
    ) -> float:
        """Expected utility for a diagnostic query (single- or multi-period).

        Args:
            reference_point: Optional per-call override for the prospect-theory
                reference point (e.g. the query's own current wealth). Uses
                ``self.reference_point`` when None.
        """
        if multiperiod_horizon is not None and multiperiod_horizon > 1:
            return self.evaluate_allocation_multiperiod(
                allocation,
                channel_means,
                channel_variances,
                current_wealth,
                int(multiperiod_horizon),
                reference_point=reference_point,
            )
        return self.evaluate_allocation(
            allocation,
            channel_means,
            channel_variances,
            current_wealth,
            rounds_remaining,
            reference_point=reference_point,
        )

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
