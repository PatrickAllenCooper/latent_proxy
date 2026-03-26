from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


class PosteriorBase(ABC):
    """Abstract base for posterior distributions over user preference parameters."""

    param_names: list[str]

    @property
    @abstractmethod
    def n_params(self) -> int: ...

    @property
    @abstractmethod
    def mean(self) -> NDArray[np.floating[Any]]: ...

    @property
    @abstractmethod
    def variance(self) -> NDArray[np.floating[Any]]: ...

    @property
    def std(self) -> NDArray[np.floating[Any]]:
        return np.sqrt(self.variance)

    @abstractmethod
    def sample(
        self, n: int = 1, rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating[Any]]: ...

    @abstractmethod
    def entropy(self) -> float: ...

    @abstractmethod
    def update_from_choice(
        self,
        choice: int,
        option_a_alloc: NDArray[np.floating[Any]],
        option_b_alloc: NDArray[np.floating[Any]],
        channel_means: NDArray[np.floating[Any]],
        channel_variances: NDArray[np.floating[Any]],
        current_wealth: float,
        rounds_remaining: int,
        temperature: float,
        multiperiod_horizon: int | None = None,
    ) -> None: ...

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(self.mean[i])
            for i, name in enumerate(self.param_names)
        }

    def credible_region(
        self, level: float = 0.9, n_samples: int = 10000,
        rng: np.random.Generator | None = None,
    ) -> dict[str, tuple[float, float]]:
        samples = self.sample(n_samples, rng)
        alpha = (1.0 - level) / 2.0
        intervals = {}
        for i, name in enumerate(self.param_names):
            lo = float(np.quantile(samples[:, i], alpha))
            hi = float(np.quantile(samples[:, i], 1.0 - alpha))
            intervals[name] = (lo, hi)
        return intervals

    def converged(self, thresholds: dict[str, float]) -> bool:
        """Check whether all parameter variances are below their thresholds."""
        for i, name in enumerate(self.param_names):
            key = f"{name}_variance_threshold"
            if key in thresholds:
                if self.variance[i] > thresholds[key]:
                    return False
        return True


def _clamp_theta(samples: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Enforce parameter constraints: gamma in (0,1], alpha >= 0, lambda >= 1."""
    out = samples.copy()
    if out.ndim == 1:
        out = out.reshape(1, -1)
    out[:, 0] = np.clip(out[:, 0], 1e-6, 1.0)
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    out[:, 2] = np.maximum(out[:, 2], 1.0)
    return out.squeeze() if samples.ndim == 1 else out


def _choice_log_likelihood(
    theta: NDArray[np.floating[Any]],
    choice: int,
    option_a_alloc: NDArray[np.floating[Any]],
    option_b_alloc: NDArray[np.floating[Any]],
    channel_means: NDArray[np.floating[Any]],
    channel_variances: NDArray[np.floating[Any]],
    current_wealth: float,
    rounds_remaining: int,
    temperature: float,
    rng: np.random.Generator | None = None,
    multiperiod_horizon: int | None = None,
) -> float:
    """Log-probability of the observed choice under softmax-rational model for a given theta."""
    from src.training.synthetic_users import SyntheticUser, UserType

    gamma, alpha, lambda_ = float(theta[0]), float(theta[1]), float(theta[2])
    gamma = np.clip(gamma, 1e-6, 1.0)
    alpha = max(alpha, 0.0)
    lambda_ = max(lambda_, 1.0)

    ut = UserType(gamma=gamma, alpha=alpha, lambda_=lambda_)
    user = SyntheticUser(ut, temperature=temperature, seed=None)
    user._rng = rng or np.random.default_rng(0)

    eu_a = user.evaluate_for_query(
        option_a_alloc,
        channel_means,
        channel_variances,
        current_wealth,
        rounds_remaining,
        multiperiod_horizon=multiperiod_horizon,
    )
    eu_b = user.evaluate_for_query(
        option_b_alloc,
        channel_means,
        channel_variances,
        current_wealth,
        rounds_remaining,
        multiperiod_horizon=multiperiod_horizon,
    )

    diff = (eu_a - eu_b) / max(temperature, 1e-10)
    diff = np.clip(diff, -500, 500)
    log_prob_a = -np.log1p(np.exp(-diff))
    log_prob_b = -np.log1p(np.exp(diff))

    return float(log_prob_a) if choice == 0 else float(log_prob_b)


@dataclass
class GaussianPosterior(PosteriorBase):
    """Parametric Gaussian posterior over user preference parameters."""

    param_names: list[str] = field(
        default_factory=lambda: ["gamma", "alpha", "lambda_"]
    )
    _mean: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([0.5, 1.0, 1.5])
    )
    covariance: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.diag([0.08, 0.25, 0.33])
    )

    def __post_init__(self) -> None:
        self._mean = np.asarray(self._mean, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        n = len(self.param_names)
        if self._mean.shape != (n,):
            raise ValueError(f"mean shape {self._mean.shape} != ({n},)")
        if self.covariance.shape != (n, n):
            raise ValueError(f"covariance shape {self.covariance.shape} != ({n},{n})")

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def mean(self) -> NDArray[np.floating[Any]]:
        return self._mean

    @property
    def variance(self) -> NDArray[np.floating[Any]]:
        return np.diag(self.covariance)

    def sample(
        self, n: int = 1, rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating[Any]]:
        rng = rng or np.random.default_rng()
        samples = rng.multivariate_normal(self._mean, self.covariance, size=n)
        return _clamp_theta(samples)

    def update(
        self,
        observation: NDArray[np.floating[Any]],
        observation_variance: NDArray[np.floating[Any]] | float = 0.1,
    ) -> None:
        """Kalman-filter style update with a noisy observation of theta."""
        observation = np.asarray(observation, dtype=np.float64)
        if np.isscalar(observation_variance):
            R = np.eye(self.n_params) * float(observation_variance)
        else:
            R = np.diag(np.asarray(observation_variance, dtype=np.float64))

        S = self.covariance + R
        K = self.covariance @ np.linalg.inv(S)
        self._mean = self._mean + K @ (observation - self._mean)
        self.covariance = (np.eye(self.n_params) - K) @ self.covariance
        self.covariance = (self.covariance + self.covariance.T) / 2.0

    def update_from_choice(
        self,
        choice: int,
        option_a_alloc: NDArray[np.floating[Any]],
        option_b_alloc: NDArray[np.floating[Any]],
        channel_means: NDArray[np.floating[Any]],
        channel_variances: NDArray[np.floating[Any]],
        current_wealth: float,
        rounds_remaining: int,
        temperature: float,
        multiperiod_horizon: int | None = None,
    ) -> None:
        """Approximate Bayesian update from an observed binary choice.

        Uses importance-weighted samples to shift the Gaussian mean and covariance
        toward theta values consistent with the observed choice.
        """
        rng = np.random.default_rng()
        n_samples = 500
        samples = self.sample(n_samples, rng)

        log_weights = np.array([
            _choice_log_likelihood(
                s, choice, option_a_alloc, option_b_alloc,
                channel_means, channel_variances,
                current_wealth, rounds_remaining, temperature,
                rng=np.random.default_rng(i),
                multiperiod_horizon=multiperiod_horizon,
            )
            for i, s in enumerate(samples)
        ])

        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum() + 1e-30

        self._mean = np.average(samples, weights=weights, axis=0)
        diff = samples - self._mean
        self.covariance = np.einsum("i,ij,ik->jk", weights, diff, diff)
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        min_var = 1e-4
        np.fill_diagonal(
            self.covariance,
            np.maximum(np.diag(self.covariance), min_var),
        )

    def entropy(self) -> float:
        n = self.n_params
        sign, logdet = np.linalg.slogdet(self.covariance)
        if sign <= 0:
            return float("-inf")
        return float(0.5 * n * (1 + np.log(2 * np.pi)) + 0.5 * logdet)


@dataclass
class ParticlePosterior(PosteriorBase):
    """Particle-based posterior over user preference parameters.

    Maintains a weighted set of theta particles. Updates by reweighting
    particles according to the likelihood of observed choices, then
    resamples when the effective sample size drops too low.
    """

    param_names: list[str] = field(
        default_factory=lambda: ["gamma", "alpha", "lambda_"]
    )
    n_particles: int = 1000
    particles: NDArray[np.floating[Any]] | None = field(default=None, repr=False)
    weights: NDArray[np.floating[Any]] | None = field(default=None, repr=False)
    ess_threshold_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.particles is None:
            rng = np.random.default_rng(0)
            self.particles = np.column_stack([
                np.clip(rng.beta(2, 2, size=self.n_particles), 1e-6, 1.0),
                np.maximum(rng.lognormal(0, 0.5, size=self.n_particles), 0.0),
                np.clip(rng.uniform(1, 3, size=self.n_particles), 1.0, None),
            ])
        else:
            self.particles = np.asarray(self.particles, dtype=np.float64)
            self.n_particles = len(self.particles)

        if self.weights is None:
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights = np.asarray(self.weights, dtype=np.float64)
            self.weights /= self.weights.sum()

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def mean(self) -> NDArray[np.floating[Any]]:
        return np.average(self.particles, weights=self.weights, axis=0)

    @property
    def variance(self) -> NDArray[np.floating[Any]]:
        mu = self.mean
        diff = self.particles - mu
        return np.einsum("i,ij,ij->j", self.weights, diff, diff)

    @property
    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights ** 2))

    def sample(
        self, n: int = 1, rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating[Any]]:
        rng = rng or np.random.default_rng()
        indices = rng.choice(
            self.n_particles, size=n, p=self.weights, replace=True,
        )
        samples = self.particles[indices].copy()
        noise = rng.normal(0, 0.01, size=samples.shape)
        samples += noise
        return _clamp_theta(samples)

    def update_from_choice(
        self,
        choice: int,
        option_a_alloc: NDArray[np.floating[Any]],
        option_b_alloc: NDArray[np.floating[Any]],
        channel_means: NDArray[np.floating[Any]],
        channel_variances: NDArray[np.floating[Any]],
        current_wealth: float,
        rounds_remaining: int,
        temperature: float,
        multiperiod_horizon: int | None = None,
    ) -> None:
        log_likelihoods = np.array([
            _choice_log_likelihood(
                self.particles[i], choice,
                option_a_alloc, option_b_alloc,
                channel_means, channel_variances,
                current_wealth, rounds_remaining, temperature,
                rng=np.random.default_rng(i),
                multiperiod_horizon=multiperiod_horizon,
            )
            for i in range(self.n_particles)
        ])

        log_likelihoods -= log_likelihoods.max()
        likelihood = np.exp(log_likelihoods)
        self.weights *= likelihood
        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles

        if self.effective_sample_size < self.ess_threshold_ratio * self.n_particles:
            self._systematic_resample()

    def _systematic_resample(self) -> None:
        """Systematic resampling to combat particle degeneracy."""
        n = self.n_particles
        positions = (np.arange(n) + np.random.default_rng().random()) / n
        cumsum = np.cumsum(self.weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, n - 1)
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(n) / n

        noise = np.random.default_rng().normal(0, 0.005, size=self.particles.shape)
        self.particles += noise
        self.particles = _clamp_theta(self.particles)

    def entropy(self) -> float:
        """Approximate entropy via weighted variance determinant."""
        var = self.variance
        pseudo_det = float(np.prod(np.maximum(var, 1e-20)))
        n = self.n_params
        return float(0.5 * n * (1 + np.log(2 * np.pi)) + 0.5 * np.log(pseudo_det))


def compute_eig(
    query_utilities: NDArray[np.floating[Any]],
    posterior: PosteriorBase,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate Expected Information Gain for a binary query.

    Legacy stub kept for backward compatibility.
    Use information_gain.compute_eig_mc for proper estimation.
    """
    rng = rng or np.random.default_rng()
    query_utilities = np.asarray(query_utilities, dtype=np.float64)
    spread = abs(float(query_utilities[0] - query_utilities[1]))
    current_entropy = posterior.entropy()
    max_var = float(np.max(posterior.variance))
    eig_approx = spread * max_var / (spread + 1e-6)
    return min(float(eig_approx), current_entropy)
