from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.resource_game import ResourceStrategyGame
from src.utils.diagnostic_scenarios import DiagnosticScenario
from src.utils.posterior import GaussianPosterior, ParticlePosterior, PosteriorBase


@dataclass
class ConvergenceConfig:
    """Thresholds for elicitation convergence."""

    gamma_variance_threshold: float = 0.005
    alpha_variance_threshold: float = 0.05
    lambda_variance_threshold: float = 0.05
    robust_action_level: float = 0.9
    max_rounds: int = 10


class PreferenceTracker:
    """Manages posterior updates and convergence checking.

    Wraps either a GaussianPosterior or ParticlePosterior and provides
    a uniform interface for the elicitation loop.
    """

    def __init__(
        self,
        posterior_type: str = "particle",
        n_particles: int = 1000,
        temperature: float = 0.1,
        convergence: ConvergenceConfig | None = None,
    ) -> None:
        self.temperature = temperature
        self.convergence = convergence or ConvergenceConfig()
        self.n_observations = 0

        if posterior_type == "particle":
            self.posterior: PosteriorBase = ParticlePosterior(n_particles=n_particles)
        elif posterior_type == "gaussian":
            self.posterior = GaussianPosterior()
        else:
            raise ValueError(f"Unknown posterior type: {posterior_type}")

    def observe(
        self,
        choice: int,
        scenario: DiagnosticScenario,
    ) -> None:
        """Update the posterior after observing the user's choice."""
        self.posterior.update_from_choice(
            choice=choice,
            option_a_alloc=scenario.option_a,
            option_b_alloc=scenario.option_b,
            channel_means=scenario.channel_means,
            channel_variances=scenario.channel_variances,
            current_wealth=scenario.current_wealth,
            rounds_remaining=scenario.rounds_remaining,
            temperature=self.temperature,
            multiperiod_horizon=scenario.multiperiod_horizon,
        )
        self.n_observations += 1

    def get_recommendation_theta(self) -> dict[str, float]:
        """Return the posterior mean as a named parameter dict."""
        return self.posterior.to_dict()

    def check_convergence(self) -> tuple[bool, str]:
        """Check all convergence criteria.

        Returns (converged, reason) where reason describes which
        criterion triggered convergence.
        """
        if self.n_observations >= self.convergence.max_rounds:
            return True, "max_rounds"

        thresholds = {
            "gamma_variance_threshold": self.convergence.gamma_variance_threshold,
            "alpha_variance_threshold": self.convergence.alpha_variance_threshold,
            "lambda__variance_threshold": self.convergence.lambda_variance_threshold,
        }
        if self.posterior.converged(thresholds):
            return True, "variance_threshold"

        return False, ""

    def check_robust_action(
        self, env: ResourceStrategyGame, level: float | None = None,
    ) -> bool:
        """Check whether the optimal action is robust to posterior uncertainty.

        The action is robust if it is the same (within tolerance) across
        the credible region of theta.
        """
        level = level or self.convergence.robust_action_level
        region = self.posterior.credible_region(level, n_samples=500)

        corners = []
        for name in self.posterior.param_names:
            if name in region:
                corners.append(region[name])

        if not corners:
            return False

        lo_theta = {
            self.posterior.param_names[i]: corners[i][0]
            for i in range(len(corners))
        }
        hi_theta = {
            self.posterior.param_names[i]: corners[i][1]
            for i in range(len(corners))
        }
        mean_theta = self.posterior.to_dict()

        action_mean = env.get_optimal_action(mean_theta)
        action_lo = env.get_optimal_action(lo_theta)
        action_hi = env.get_optimal_action(hi_theta)

        dist_lo = float(np.linalg.norm(action_mean - action_lo))
        dist_hi = float(np.linalg.norm(action_mean - action_hi))

        return dist_lo < 0.1 and dist_hi < 0.1

    def summary(self) -> dict[str, Any]:
        """Return a summary of the current tracker state."""
        return {
            "mean": self.posterior.to_dict(),
            "variance": {
                name: float(self.posterior.variance[i])
                for i, name in enumerate(self.posterior.param_names)
            },
            "entropy": self.posterior.entropy(),
            "n_observations": self.n_observations,
        }
