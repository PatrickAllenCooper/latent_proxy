from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.base import BaseEnvironment
from src.training.serialization import AllocationSerializer
from src.utils.posterior import PosteriorBase


class ResponseGenerator:
    """Generates recommendations from the current posterior estimate.

    Given the posterior mean theta, computes the optimal allocation via
    the environment's get_optimal_action, checks quality floor compliance,
    and serializes as text.
    """

    def __init__(self, channel_names: list[str] | None = None) -> None:
        self._serializer = AllocationSerializer(channel_names)

    def generate(
        self,
        env: BaseEnvironment,
        posterior: PosteriorBase,
    ) -> NDArray[np.floating[Any]]:
        """Compute the recommended allocation for the current posterior mean."""
        theta = posterior.to_dict()
        action = env.get_optimal_action(theta)

        passes, violations = env.check_quality_floor(action)
        if not passes:
            K = env.config.n_channels
            action = np.ones(K) / K

        return action

    def generate_text(
        self,
        env: BaseEnvironment,
        posterior: PosteriorBase,
    ) -> str:
        """Generate a text recommendation."""
        action = self.generate(env, posterior)
        return self._serializer.serialize(action)
