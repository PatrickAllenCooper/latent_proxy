from __future__ import annotations

from abc import abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class BaseEnvironment(gym.Env):
    """Abstract base for all decision environments in the latent preference system.

    Extends gymnasium.Env with methods required for preference-aware recommendation:
    quality scoring, quality floor constraint checking, and optimal action computation
    given known user preference parameters.
    """

    @abstractmethod
    def quality_score(self, action: NDArray[np.floating[Any]]) -> float:
        """Compute R_quality for an action in the current environment state.

        This measures objective performance independent of user preferences.
        Higher values indicate objectively better actions.
        """

    @abstractmethod
    def check_quality_floor(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[bool, list[str]]:
        """Check whether an action violates hard quality constraints.

        Returns:
            A tuple of (passes, violations) where passes is True if the action
            satisfies all constraints, and violations is a list of human-readable
            descriptions of any violated constraints.
        """

    @abstractmethod
    def get_optimal_action(
        self, theta: dict[str, float]
    ) -> NDArray[np.floating[Any]]:
        """Compute the optimal action for a user with the given preference parameters.

        Used for evaluation: comparing agent recommendations against the
        true optimum under known user types.

        Args:
            theta: Dictionary of user preference parameters (e.g. gamma, alpha, lambda_).
        """

    @abstractmethod
    def get_channel_stats(self) -> dict[str, NDArray[np.floating[Any]]]:
        """Return current channel statistics (means, variances) given the regime.

        Used by quality floor checks and the active learning module.
        """
