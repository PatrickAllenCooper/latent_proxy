from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environments.base import BaseEnvironment
from src.utils.diagnostic_scenarios import DiagnosticScenario
from src.utils.posterior import PosteriorBase


class BaseAgent(ABC):
    """Abstract interface for a preference-eliciting recommendation agent."""

    @abstractmethod
    def select_query(
        self,
        env: BaseEnvironment,
        posterior: PosteriorBase,
    ) -> DiagnosticScenario:
        """Select the next diagnostic query to pose to the user."""

    @abstractmethod
    def generate_recommendation(
        self,
        env: BaseEnvironment,
        posterior: PosteriorBase,
    ) -> NDArray[np.floating[Any]]:
        """Generate a recommended allocation given current beliefs."""

    @abstractmethod
    def update_beliefs(
        self,
        choice: int,
        scenario: DiagnosticScenario,
        posterior: PosteriorBase,
    ) -> None:
        """Update the posterior after observing the user's choice."""
