from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.agents.preference_tracker import ConvergenceConfig, PreferenceTracker
from src.agents.query_generator import RandomQueryGenerator, StructuredQueryGenerator
from src.agents.response_generator import ResponseGenerator
from src.environments.resource_game import ResourceStrategyGame
from src.evaluation.alignment_metrics import compute_preference_recovery_error
from src.training.synthetic_users import SyntheticUser, UserType
from src.utils.diagnostic_scenarios import DiagnosticScenario

logger = logging.getLogger(__name__)


@dataclass
class ElicitationConfig:
    """Configuration for the elicitation loop."""

    posterior_type: str = "particle"
    n_particles: int = 1000
    max_rounds: int = 10
    n_scenarios_per_round: int = 50
    n_eig_samples: int = 500
    temperature: float = 0.1
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    seed: int = 42


@dataclass
class ElicitationResult:
    """Result of running the elicitation loop on a single user."""

    inferred_theta: dict[str, float]
    true_theta: UserType | None
    history: list[tuple[DiagnosticScenario, int]]
    n_rounds: int
    convergence_reason: str
    variance_trajectory: list[dict[str, float]]

    def preference_recovery_error(self) -> dict[str, float] | None:
        if self.true_theta is None:
            return None
        return compute_preference_recovery_error(
            self.inferred_theta, self.true_theta,
        )


class ElicitationLoop:
    """Orchestrates the active learning cycle for preference elicitation.

    In each round:
    1. Generates candidate diagnostic scenarios
    2. Scores them by EIG and selects the best
    3. Presents the query to the user (synthetic or real)
    4. Observes the response and updates the posterior
    5. Checks convergence criteria
    """

    def __init__(self, config: ElicitationConfig | None = None) -> None:
        self.config = config or ElicitationConfig()

    def run(
        self,
        env: ResourceStrategyGame,
        user: SyntheticUser,
        query_type: str = "active",
    ) -> ElicitationResult:
        """Run the full elicitation loop with a synthetic user.

        Args:
            env: The game environment.
            user: A synthetic user with known theta.
            query_type: "active" for EIG-based selection, "random" for baseline.
        """
        tracker = PreferenceTracker(
            posterior_type=self.config.posterior_type,
            n_particles=self.config.n_particles,
            temperature=self.config.temperature,
            convergence=self.config.convergence,
        )

        if query_type == "active":
            query_gen = StructuredQueryGenerator(
                n_scenarios_per_round=self.config.n_scenarios_per_round,
                n_eig_samples=self.config.n_eig_samples,
                temperature=self.config.temperature,
                seed=self.config.seed,
            )
        else:
            query_gen = RandomQueryGenerator(seed=self.config.seed)

        history: list[tuple[DiagnosticScenario, int]] = []
        variance_trajectory: list[dict[str, float]] = []
        convergence_reason = "max_rounds"

        variance_trajectory.append({
            name: float(tracker.posterior.variance[i])
            for i, name in enumerate(tracker.posterior.param_names)
        })

        for round_idx in range(self.config.max_rounds):
            converged, reason = tracker.check_convergence()
            if converged:
                convergence_reason = reason
                break

            scenario = query_gen.select_query(env, tracker.posterior)

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
            choice = user.choose(eu_a, eu_b)

            tracker.observe(choice, scenario)
            history.append((scenario, choice))

            variance_trajectory.append({
                name: float(tracker.posterior.variance[i])
                for i, name in enumerate(tracker.posterior.param_names)
            })

            logger.debug(
                "Round %d: choice=%d, mean=%s, var=%s",
                round_idx + 1, choice,
                tracker.posterior.to_dict(),
                {n: f"{v:.4f}" for n, v in zip(
                    tracker.posterior.param_names, tracker.posterior.variance
                )},
            )

        converged, reason = tracker.check_convergence()
        if converged and convergence_reason == "max_rounds":
            convergence_reason = reason

        inferred = tracker.get_recommendation_theta()

        return ElicitationResult(
            inferred_theta=inferred,
            true_theta=user.user_type,
            history=history,
            n_rounds=len(history),
            convergence_reason=convergence_reason,
            variance_trajectory=variance_trajectory,
        )
