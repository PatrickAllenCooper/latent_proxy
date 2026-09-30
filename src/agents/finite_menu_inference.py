"""Particle inference and acquisition for the finite-menu user model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.evaluation.preference_benchmark import MenuScenario
from src.training.synthetic_users import SyntheticUserSampler, UserType


@dataclass(frozen=True)
class UserParticles:
    gamma: NDArray[np.float64]
    alpha: NDArray[np.float64]
    lambda_: NDArray[np.float64]
    behavior_bias: NDArray[np.float64]

    @classmethod
    def sample(cls, n: int, seed: int) -> UserParticles:
        if n < 2:
            raise ValueError("at least two particles are required")
        types = SyntheticUserSampler(seed=seed).sample_batch(n)
        rng = np.random.default_rng(seed + 10_000)
        bias = rng.normal(0.0, .018, size=(n, 4))
        bias[:, 0] += rng.uniform(-.06, .06, size=n)
        return cls(
            np.array([t.gamma for t in types], dtype=np.float64),
            np.array([t.alpha for t in types], dtype=np.float64),
            np.array([t.lambda_ for t in types], dtype=np.float64),
            bias,
        )

    @property
    def n(self) -> int:
        return len(self.gamma)


def particle_utilities(
    scenario: MenuScenario, particles: UserParticles
) -> NDArray[np.float64]:
    """Expected utility, indexed by particle then action."""
    payoff = scenario.payoffs[None, :, :, :]
    exponent = 1.0 / (1.0 + particles.alpha[:, None, None, None])
    lam = particles.lambda_[:, None, None, None]
    values = np.where(
        payoff >= 0,
        np.maximum(payoff, 0) ** exponent,
        -lam * np.maximum(-payoff, 0) ** exponent,
    )
    periods = np.arange(scenario.payoffs.shape[-1], dtype=np.float64)
    discount = particles.gamma[:, None, None, None] ** periods[None, None, None, :]
    outcome_values = np.sum(values * discount, axis=-1)
    expected = np.sum(outcome_values * scenario.probabilities[None, :, :], axis=-1)
    return np.where(scenario.feasible[None, :], expected, -np.inf)


def particle_behavior(
    utilities: NDArray[np.float64],
    bias: NDArray[np.float64],
    temperature: float = .025,
) -> NDArray[np.float64]:
    logits = (utilities + bias) / temperature
    logits -= np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(logits)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def prepared_menus(
    scenarios: list[MenuScenario], particles: UserParticles
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    utilities = np.stack([particle_utilities(s, particles) for s in scenarios])
    behavior = np.stack([
        particle_behavior(utilities[i], particles.behavior_bias)
        for i in range(len(scenarios))
    ])
    return utilities, behavior


def acquisition_scores(
    weights: NDArray[np.float64],
    query_behavior: NDArray[np.float64],
    target_utilities: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mutual information and exact finite-menu decision value of information."""
    predictive = np.einsum("n,qna->qa", weights, query_behavior, optimize=True)
    eps = 1e-12
    predicted_entropy = -np.sum(predictive * np.log(predictive + eps), axis=-1)
    conditional_entropy = -np.sum(
        query_behavior * np.log(query_behavior + eps), axis=-1
    ) @ weights
    information_gain = np.maximum(predicted_entropy - conditional_entropy, 0.0)
    current = np.einsum("n,tna->ta", weights, target_utilities, optimize=True)
    current_value = float(np.max(current, axis=-1).mean())
    # Multiplying each prospective posterior action value by P(response)
    # cancels the posterior normalizer. This computes Bayesian EVSI directly.
    future_weighted = np.einsum(
        "n,qnr,tna->qrta", weights, query_behavior, target_utilities, optimize=True
    )
    future_value = np.max(future_weighted, axis=-1).sum(axis=1).mean(axis=-1)
    decision_value = np.maximum(future_value - current_value, 0.0)
    return information_gain, decision_value


def choose_query(
    arm: str,
    weights: NDArray[np.float64],
    query_behavior: NDArray[np.float64],
    target_utilities: NDArray[np.float64],
    available: NDArray[np.bool_],
    rng: np.random.Generator,
) -> tuple[int, float, float]:
    candidates = np.flatnonzero(available)
    if not len(candidates):
        raise ValueError("no queries remain")
    if arm == "random":
        return int(rng.choice(candidates)), float("nan"), float("nan")
    information, value = acquisition_scores(
        weights, query_behavior[candidates], target_utilities
    )
    if arm == "eig":
        score = information
    elif arm == "decision_value":
        score = value
    elif arm == "aif_50":
        def scale(x: NDArray[np.float64]) -> NDArray[np.float64]:
            span = float(np.max(x) - np.min(x))
            return (x - np.min(x)) / span if span > 1e-12 else np.zeros_like(x)

        score = .5 * scale(information) + .5 * scale(value)
    else:
        raise ValueError(f"unknown acquisition arm: {arm}")
    selected = int(np.argmax(score))
    return int(candidates[selected]), float(information[selected]), float(value[selected])


def update_weights(
    weights: NDArray[np.float64],
    choice_likelihood: NDArray[np.float64],
) -> NDArray[np.float64]:
    updated = weights * choice_likelihood
    mass = float(updated.sum())
    if mass <= 1e-300 or not np.isfinite(mass):
        raise ValueError("posterior likelihood underflow")
    return updated / mass


def posterior_mean(weights: NDArray[np.float64], particles: UserParticles) -> UserType:
    return UserType(
        gamma=float(np.dot(weights, particles.gamma)),
        alpha=float(np.dot(weights, particles.alpha)),
        lambda_=float(np.dot(weights, particles.lambda_)),
    )
