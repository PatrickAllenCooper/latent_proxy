from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class GaussianPosterior:
    """Parametric Gaussian posterior over user preference parameters.

    Maintains a multivariate Gaussian N(mean, covariance) over theta = (gamma, alpha, lambda_).
    Provides Bayesian update via conjugate-like approximation and sampling for
    downstream EIG computation and decision making.

    This is the Milestone 1 stub. Full Bayesian tracking with NumPyro
    (particle-based or MCMC) will replace this in Milestone 3.
    """

    param_names: list[str] = field(
        default_factory=lambda: ["gamma", "alpha", "lambda_"]
    )
    mean: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.array([0.5, 1.0, 1.5])
    )
    covariance: NDArray[np.floating[Any]] = field(
        default_factory=lambda: np.diag([0.08, 0.25, 0.33])
    )

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64)
        self.covariance = np.asarray(self.covariance, dtype=np.float64)
        n = len(self.param_names)
        if self.mean.shape != (n,):
            raise ValueError(
                f"mean shape {self.mean.shape} does not match "
                f"{n} parameters"
            )
        if self.covariance.shape != (n, n):
            raise ValueError(
                f"covariance shape {self.covariance.shape} does not match "
                f"({n}, {n})"
            )

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def variance(self) -> NDArray[np.floating[Any]]:
        """Marginal variances for each parameter."""
        return np.diag(self.covariance)

    @property
    def std(self) -> NDArray[np.floating[Any]]:
        return np.sqrt(self.variance)

    def sample(
        self,
        n: int = 1,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Draw n samples from the posterior.

        Returns array of shape (n, n_params).
        """
        rng = rng or np.random.default_rng()
        samples = rng.multivariate_normal(self.mean, self.covariance, size=n)
        samples[:, 0] = np.clip(samples[:, 0], 1e-6, 1.0)
        samples[:, 1] = np.maximum(samples[:, 1], 0.0)
        samples[:, 2] = np.maximum(samples[:, 2], 1.0)
        return samples

    def update(
        self,
        observation: NDArray[np.floating[Any]],
        observation_variance: NDArray[np.floating[Any]] | float = 0.1,
    ) -> None:
        """Bayesian update with a noisy observation of theta.

        Uses the Kalman-filter update equations for a Gaussian prior
        with Gaussian observation noise.

        Args:
            observation: Observed value of theta (possibly noisy).
            observation_variance: Observation noise variance. Scalar or diagonal.
        """
        observation = np.asarray(observation, dtype=np.float64)
        if np.isscalar(observation_variance):
            R = np.eye(self.n_params) * float(observation_variance)
        else:
            R = np.diag(np.asarray(observation_variance, dtype=np.float64))

        S = self.covariance + R
        K = self.covariance @ np.linalg.inv(S)
        innovation = observation - self.mean
        self.mean = self.mean + K @ innovation
        self.covariance = (np.eye(self.n_params) - K) @ self.covariance

        self.covariance = (self.covariance + self.covariance.T) / 2.0

    def entropy(self) -> float:
        """Differential entropy of the Gaussian posterior."""
        n = self.n_params
        return float(
            0.5 * n * (1 + np.log(2 * np.pi))
            + 0.5 * np.log(np.linalg.det(self.covariance))
        )

    def to_dict(self) -> dict[str, float]:
        """Return the posterior mean as a named parameter dict."""
        return {
            name: float(self.mean[i])
            for i, name in enumerate(self.param_names)
        }

    def credible_region(
        self, level: float = 0.9, n_samples: int = 10000,
        rng: np.random.Generator | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Compute marginal credible intervals for each parameter."""
        samples = self.sample(n_samples, rng)
        alpha = (1.0 - level) / 2.0
        intervals = {}
        for i, name in enumerate(self.param_names):
            lo = float(np.quantile(samples[:, i], alpha))
            hi = float(np.quantile(samples[:, i], 1.0 - alpha))
            intervals[name] = (lo, hi)
        return intervals


def compute_eig(
    query_utilities: NDArray[np.floating[Any]],
    posterior: GaussianPosterior,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate Expected Information Gain for a binary query.

    Placeholder for Milestone 3. Currently uses a variance-reduction
    heuristic: EIG is approximated as the expected reduction in posterior
    entropy from observing the query response.

    Args:
        query_utilities: Shape (2,) utilities of option A vs B under the
            current posterior mean. The spread indicates how diagnostic
            the query is.
        posterior: Current Gaussian posterior over theta.
        n_samples: Number of MC samples for the approximation.
        rng: Random generator.

    Returns:
        Approximate EIG in nats.
    """
    rng = rng or np.random.default_rng()
    query_utilities = np.asarray(query_utilities, dtype=np.float64)

    spread = abs(float(query_utilities[0] - query_utilities[1]))
    current_entropy = posterior.entropy()
    max_var = float(np.max(posterior.variance))

    eig_approx = spread * max_var / (spread + 1e-6)
    return min(float(eig_approx), current_entropy)
