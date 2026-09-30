"""Small contextual PPO policy for the versioned finite-menu benchmark."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from src.evaluation.preference_benchmark import MenuScenario
from src.training.synthetic_users import UserType


def encode_menu(scenario: MenuScenario, theta: UserType) -> NDArray[np.float32]:
    """Encode only information available to a gold-profile policy."""
    return np.concatenate([
        (scenario.payoffs.ravel() * 10.0).astype(np.float32),
        scenario.probabilities.ravel().astype(np.float32),
        np.array([
            theta.gamma, np.log1p(theta.alpha), theta.lambda_ - 1.0
        ], dtype=np.float32),
    ])


class FiniteMenuPolicy(nn.Module):
    """Actor-critic network; policy inference never sees evaluator scores."""

    def __init__(self, input_dim: int = 51, n_actions: int = 4) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(states)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    @torch.no_grad()
    def act(self, scenario: MenuScenario, theta: UserType) -> int:
        state = torch.from_numpy(encode_menu(scenario, theta)).unsqueeze(0)
        logits, _ = self(state)
        logits[:, ~torch.from_numpy(scenario.feasible)] = -1e9
        return int(torch.argmax(logits[0]).item())
