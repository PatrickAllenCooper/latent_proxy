from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.training.synthetic_users import SyntheticUser, UserType
from src.utils.diagnostic_scenarios import DiagnosticScenario
from src.utils.posterior import PosteriorBase


def _choice_probability(
    theta: NDArray[np.floating[Any]],
    scenario: DiagnosticScenario,
    temperature: float = 0.1,
    rng: np.random.Generator | None = None,
) -> float:
    """P(choose option A | theta, scenario) under the softmax-rational model."""
    gamma = float(np.clip(theta[0], 1e-6, 1.0))
    alpha = float(max(theta[1], 0.0))
    lambda_ = float(max(theta[2], 1.0))

    ut = UserType(gamma=gamma, alpha=alpha, lambda_=lambda_)
    user = SyntheticUser(ut, temperature=temperature, seed=None)
    user._rng = rng or np.random.default_rng(0)

    eu_a = user.evaluate_for_query(
        scenario.option_a,
        scenario.channel_means,
        scenario.channel_variances,
        scenario.current_wealth,
        scenario.rounds_remaining,
        multiperiod_horizon=scenario.multiperiod_horizon,
    )
    eu_b = user.evaluate_for_query(
        scenario.option_b,
        scenario.channel_means,
        scenario.channel_variances,
        scenario.current_wealth,
        scenario.rounds_remaining,
        multiperiod_horizon=scenario.multiperiod_horizon,
    )

    diff = (eu_a - eu_b) / max(temperature, 1e-10)
    diff = np.clip(diff, -500, 500)
    prob_a = 1.0 / (1.0 + np.exp(-diff))
    return float(prob_a)


def _weighted_entropy(
    particles: NDArray[np.floating[Any]],
    weights: NDArray[np.floating[Any]],
) -> float:
    """Estimate entropy of a weighted particle distribution.

    Uses the determinant of the weighted covariance as a proxy.
    """
    n_params = particles.shape[1]
    mu = np.average(particles, weights=weights, axis=0)
    diff = particles - mu
    cov = np.einsum("i,ij,ik->jk", weights, diff, diff)
    var_diag = np.maximum(np.diag(cov), 1e-20)
    pseudo_det = float(np.prod(var_diag))
    return float(0.5 * n_params * (1 + np.log(2 * np.pi)) + 0.5 * np.log(pseudo_det))


def compute_eig_mc(
    scenario: DiagnosticScenario,
    posterior: PosteriorBase,
    n_samples: int = 500,
    temperature: float = 0.1,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate Expected Information Gain for a diagnostic scenario.

    Uses nested Monte Carlo: sample theta particles from the posterior,
    compute response probabilities under each, then estimate the expected
    reduction in posterior entropy.

    EIG = H[theta] - E_r[ H[theta | r] ]
    """
    rng = rng or np.random.default_rng()
    particles = posterior.sample(n_samples, rng)

    probs_a = np.array([
        _choice_probability(
            particles[i], scenario, temperature,
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        )
        for i in range(n_samples)
    ])
    probs_b = 1.0 - probs_a

    marginal_a = float(np.mean(probs_a))
    marginal_b = 1.0 - marginal_a

    marginal_a = np.clip(marginal_a, 1e-10, 1.0 - 1e-10)
    marginal_b = 1.0 - marginal_a

    prior_entropy = _weighted_entropy(
        particles, np.ones(n_samples) / n_samples,
    )

    weights_a = probs_a / (marginal_a * n_samples)
    weights_a /= weights_a.sum() + 1e-30
    entropy_a = _weighted_entropy(particles, weights_a)

    weights_b = probs_b / (marginal_b * n_samples)
    weights_b /= weights_b.sum() + 1e-30
    entropy_b = _weighted_entropy(particles, weights_b)

    eig = prior_entropy - marginal_a * entropy_a - marginal_b * entropy_b
    return max(float(eig), 0.0)


def compute_eig_batch(
    scenarios: list[DiagnosticScenario],
    posterior: PosteriorBase,
    n_samples: int = 500,
    temperature: float = 0.1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.floating[Any]]:
    """Score a batch of scenarios by EIG. Returns array of EIG values."""
    rng = rng or np.random.default_rng()
    return np.array([
        compute_eig_mc(
            s, posterior, n_samples, temperature,
            rng=np.random.default_rng(int(rng.integers(0, 2**31))),
        )
        for s in scenarios
    ])
