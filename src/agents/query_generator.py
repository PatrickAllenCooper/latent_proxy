from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from src.environments.base import BaseEnvironment
from src.utils.diagnostic_scenarios import (
    DiagnosticScenario,
    ScenarioLibrary,
    ScenarioLibraryBase,
)
from src.utils.information_gain import compute_eig_batch
from src.utils.posterior import ParticlePosterior, PosteriorBase
from src.environments.env_utils import certainty_equivalent


class StructuredQueryGenerator:
    """Selects diagnostic queries by maximizing Expected Information Gain.

    Option A from README Section 6.1: a structured elicitation module computes
    EIG over a predefined set of diagnostic game scenarios and selects the
    highest-scoring one.
    """

    def __init__(
        self,
        n_scenarios_per_round: int = 50,
        n_eig_samples: int = 500,
        temperature: float = 0.1,
        seed: int = 42,
        library: ScenarioLibraryBase | None = None,
        reference_point_mode: str = "zero",
        utility_form: str = "absolute",
    ) -> None:
        self.n_scenarios_per_round = n_scenarios_per_round
        self.n_eig_samples = n_eig_samples
        self.temperature = temperature
        self.reference_point_mode = reference_point_mode
        self.utility_form = utility_form
        self._library: ScenarioLibraryBase = (
            library if library is not None else ScenarioLibrary(seed=seed)
        )
        self._rng = np.random.default_rng(seed)

    def select_query(
        self,
        env: BaseEnvironment,
        posterior: PosteriorBase,
    ) -> DiagnosticScenario:
        """Select the diagnostic scenario with the highest EIG."""
        n_per_param = max(
            self.n_scenarios_per_round // 3, 3,
        )
        scenarios = self._library.generate_all(env, n_per_param)

        if not scenarios:
            scenarios = self._library.generate_all(env, n_per_param * 2)
        if not scenarios:
            return self._fallback_scenario(env)

        eig_scores = compute_eig_batch(
            scenarios, posterior, self.n_eig_samples,
            self.temperature, self._rng,
            reference_point_mode=self.reference_point_mode,
            utility_form=self.utility_form,
        )

        best_idx = int(np.argmax(eig_scores))
        return scenarios[best_idx]

    def _fallback_scenario(
        self, env: BaseEnvironment,
    ) -> DiagnosticScenario:
        """Generate a simple fallback scenario when the library is empty."""
        env.reset(seed=int(self._rng.integers(0, 2**31)))
        obs = env._get_obs()
        stats = env.get_channel_stats()
        K = env.config.n_channels
        a = np.zeros(K)
        b = np.zeros(K)
        if K >= 4:
            a[0] = 0.6
            a[1] = 0.4
            b[2] = 0.5
            b[3] = 0.5
        elif K >= 2:
            a[0] = 0.7
            a[1] = 0.3
            b[0] = 0.3
            b[1] = 0.7
        else:
            a[0] = 1.0
            b[0] = 1.0
        return DiagnosticScenario(
            game_state=obs,
            option_a=a, option_b=b,
            target_param="alpha",
            description="Safe vs risky allocation.",
            channel_means=stats["means"],
            channel_variances=stats["variances"],
            current_wealth=float(obs["wealth"].sum()),
            rounds_remaining=env.config.n_rounds,
        )


class RandomQueryGenerator:
    """Baseline that picks diagnostic scenarios at random."""

    def __init__(
        self, seed: int = 42, library: ScenarioLibraryBase | None = None,
    ) -> None:
        self._library: ScenarioLibraryBase = (
            library if library is not None else ScenarioLibrary(seed=seed)
        )
        self._rng = np.random.default_rng(seed)

    def select_query(
        self,
        env: BaseEnvironment,
        posterior: PosteriorBase,
    ) -> DiagnosticScenario:
        scenarios = self._library.generate_all(env, n_per_param=5)
        if not scenarios:
            scenarios = self._library.generate_all(env, n_per_param=10)
        if not scenarios:
            K = env.config.n_channels
            env.reset(seed=int(self._rng.integers(0, 2**31)))
            obs = env._get_obs()
            stats = env.get_channel_stats()
            a = self._rng.dirichlet(np.ones(K))
            b = self._rng.dirichlet(np.ones(K))
            return DiagnosticScenario(
                game_state=obs, option_a=a, option_b=b,
                target_param="unknown",
                description="Random allocation pair.",
                channel_means=stats["means"],
                channel_variances=stats["variances"],
                current_wealth=float(obs["wealth"].sum()),
                rounds_remaining=env.config.n_rounds,
            )
        idx = int(self._rng.integers(0, len(scenarios)))
        return scenarios[idx]


class DecisionImpactQueryGenerator:
    """Choose queries expected to reduce uncertainty in the eventual action.

    This is a one-step value-of-information proxy: for each candidate scenario,
    estimate the posterior-weighted variance of the scenario's optimal action
    before and after either possible response. The score is the expected
    reduction in that variance. It targets decision disagreement, rather than
    parameter entropy, and is intentionally reported as a proxy rather than
    literal utility regret.
    """

    def __init__(
        self,
        n_scenarios_per_round: int = 50,
        n_particles: int = 256,
        temperature: float = 0.1,
        seed: int = 42,
        library: ScenarioLibraryBase | None = None,
        reference_point_mode: str = "zero",
        utility_form: str = "absolute",
    ) -> None:
        self.n_scenarios_per_round = n_scenarios_per_round
        self.n_particles = n_particles
        self.temperature = temperature
        self.reference_point_mode = reference_point_mode
        self.utility_form = utility_form
        self._library = library or ScenarioLibrary(seed=seed)
        self._rng = np.random.default_rng(seed)

    def _weighted_actions(
        self, scenario: DiagnosticScenario, particles: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        means = np.asarray(scenario.channel_means, dtype=np.float64)
        variances = np.asarray(scenario.channel_variances, dtype=np.float64)
        actions = []
        for gamma, alpha, loss_aversion in particles:
            horizon = 1.0 / (1.0 - float(gamma) + 1e-6)
            scores = np.asarray([
                certainty_equivalent(mu, var, float(alpha), float(loss_aversion), horizon)
                for mu, var in zip(means, variances)
            ])
            positive = np.maximum(scores, 0.0)
            if positive.sum() > 1e-15:
                actions.append(positive / positive.sum())
            else:
                shifted = scores - scores.min() + 1e-10
                actions.append(shifted / shifted.sum())
        return np.asarray(actions, dtype=np.float64)

    @staticmethod
    def _weighted_variance(actions: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
        mean = np.average(actions, axis=0, weights=weights)
        return float(np.average(np.sum((actions - mean) ** 2, axis=1), weights=weights))

    def _score(
        self, scenario: DiagnosticScenario, particles: NDArray[np.float64],
        weights: NDArray[np.float64], action_cache: NDArray[np.float64],
    ) -> float:
        likelihood_a = np.empty(len(particles), dtype=np.float64)
        for i, (gamma, alpha, lambda_) in enumerate(particles):
            horizon = float(
                scenario.multiperiod_horizon
                if scenario.multiperiod_horizon is not None
                else 1.0 / (1.0 - float(gamma) + 1e-6)
            )
            channel_values = np.asarray([
                certainty_equivalent(mu, var, float(alpha), float(lambda_), horizon)
                for mu, var in zip(scenario.channel_means, scenario.channel_variances)
            ])
            ua = float(np.dot(scenario.option_a, channel_values))
            ub = float(np.dot(scenario.option_b, channel_values))
            diff = np.clip((ua - ub) / max(self.temperature, 1e-10), -500, 500)
            likelihood_a[i] = 1.0 / (1.0 + np.exp(-diff))

        p_a = float(np.dot(weights, likelihood_a))
        p_b = 1.0 - p_a
        current = self._weighted_variance(action_cache, weights)
        expected_after = 0.0
        for likelihood, probability in ((likelihood_a, p_a), (1.0 - likelihood_a, p_b)):
            if probability <= 1e-12:
                continue
            updated = weights * likelihood / probability
            expected_after += probability * self._weighted_variance(action_cache, updated)
        return max(0.0, current - expected_after)

    def select_query(self, env: BaseEnvironment, posterior: PosteriorBase) -> DiagnosticScenario:
        n_per_param = max(self.n_scenarios_per_round // 3, 3)
        scenarios = self._library.generate_all(env, n_per_param)
        if not scenarios:
            scenarios = self._library.generate_all(env, n_per_param * 2)
        if not scenarios:
            return RandomQueryGenerator(seed=int(self._rng.integers(2**31))).select_query(env, posterior)

        if isinstance(posterior, ParticlePosterior):
            n = min(self.n_particles, posterior.n_particles)
            indices = self._rng.choice(posterior.n_particles, size=n, replace=True, p=posterior.weights)
            particles = posterior.particles[indices]
            weights = np.full(n, 1.0 / n)
        else:
            particles = posterior.sample(self.n_particles, self._rng)
            weights = np.full(len(particles), 1.0 / len(particles))

        scores = []
        for scenario in scenarios:
            actions = self._weighted_actions(scenario, particles)
            scores.append(self._score(scenario, particles, weights, actions))
        return scenarios[int(np.argmax(scores))]


class DirichletQueryGenerator:
    """Fully-uninformed floor: each round is a completely random scenario.

    Draws both allocation options i.i.d. from Dirichlet(ones(K)) on a freshly
    reset environment. Unlike RandomQueryGenerator, it never touches the
    scenario library, so it carries no design knowledge at all -- the floor
    for the learning-curve decomposition.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def select_query(
        self,
        env: BaseEnvironment,
        posterior: PosteriorBase,
    ) -> DiagnosticScenario:
        K = env.config.n_channels
        env.reset(seed=int(self._rng.integers(0, 2**31)))
        obs = env._get_obs()
        stats = env.get_channel_stats()
        a = self._rng.dirichlet(np.ones(K))
        b = self._rng.dirichlet(np.ones(K))
        return DiagnosticScenario(
            game_state=obs, option_a=a, option_b=b,
            target_param="unknown",
            description="Fully random Dirichlet allocation pair.",
            channel_means=stats["means"],
            channel_variances=stats["variances"],
            current_wealth=float(obs["wealth"].sum()),
            rounds_remaining=env.config.n_rounds,
        )


class FixedQueryGenerator:
    """Non-adaptive but well-designed static questionnaire.

    On the first call it generates the scenario-library pool once (same
    n_per_param logic as StructuredQueryGenerator), scores it a single time
    by EIG against the *prior* posterior, then orders queries by ranking
    scenarios within each target_param group and interleaving the groups
    round-robin (balancing gamma/alpha/lambda_ targeting). The resulting
    sequence is served in that static order regardless of responses,
    isolating "good static design" from "adaptivity".
    """

    def __init__(
        self,
        n_scenarios_per_round: int = 50,
        n_eig_samples: int = 500,
        temperature: float = 0.1,
        seed: int = 42,
        library: ScenarioLibraryBase | None = None,
        n_particles: int = 1000,
        posterior_factory: Callable[[], PosteriorBase] | None = None,
        reference_point_mode: str = "zero",
        utility_form: str = "absolute",
    ) -> None:
        self.n_scenarios_per_round = n_scenarios_per_round
        self.n_eig_samples = n_eig_samples
        self.temperature = temperature
        self.n_particles = n_particles
        self.reference_point_mode = reference_point_mode
        self.utility_form = utility_form
        self._library: ScenarioLibraryBase = (
            library if library is not None else ScenarioLibrary(seed=seed)
        )
        self._rng = np.random.default_rng(seed)
        self._posterior_factory = posterior_factory
        self._sequence: list[DiagnosticScenario] | None = None
        self._next_idx = 0

    def select_query(
        self,
        env: BaseEnvironment,
        posterior: PosteriorBase,
    ) -> DiagnosticScenario:
        """Serve the next query of the static questionnaire (cycling if exhausted)."""
        if self._sequence is None:
            self._sequence = self._build_sequence(env)
        if not self._sequence:
            return self._fallback_scenario(env)
        scenario = self._sequence[self._next_idx % len(self._sequence)]
        self._next_idx += 1
        return scenario

    def _prior_posterior(self) -> PosteriorBase:
        """A fresh posterior of the configured type, untouched by any choices."""
        if self._posterior_factory is not None:
            return self._posterior_factory()
        return ParticlePosterior(n_particles=self.n_particles)

    def _build_sequence(self, env: BaseEnvironment) -> list[DiagnosticScenario]:
        """Generate the pool once, score it by prior EIG, and order it."""
        n_per_param = max(
            self.n_scenarios_per_round // 3, 3,
        )
        scenarios = self._library.generate_all(env, n_per_param)
        if not scenarios:
            scenarios = self._library.generate_all(env, n_per_param * 2)
        if not scenarios:
            return []

        prior = self._prior_posterior()
        eig_scores = compute_eig_batch(
            scenarios, prior, self.n_eig_samples,
            self.temperature, self._rng,
            reference_point_mode=self.reference_point_mode,
            utility_form=self.utility_form,
        )

        groups: dict[str, list[DiagnosticScenario]] = {}
        for idx in np.argsort(-eig_scores, kind="stable"):
            scenario = scenarios[int(idx)]
            groups.setdefault(scenario.target_param, []).append(scenario)

        canonical = ("gamma", "alpha", "lambda_")
        param_order = [p for p in canonical if p in groups]
        param_order += [p for p in groups if p not in canonical]

        sequence: list[DiagnosticScenario] = []
        while any(groups[p] for p in param_order):
            for param in param_order:
                if groups[param]:
                    sequence.append(groups[param].pop(0))
        return sequence

    def _fallback_scenario(
        self, env: BaseEnvironment,
    ) -> DiagnosticScenario:
        """Generate a simple fallback scenario when the library is empty."""
        env.reset(seed=int(self._rng.integers(0, 2**31)))
        obs = env._get_obs()
        stats = env.get_channel_stats()
        K = env.config.n_channels
        a = np.zeros(K)
        b = np.zeros(K)
        if K >= 4:
            a[0] = 0.6
            a[1] = 0.4
            b[2] = 0.5
            b[3] = 0.5
        elif K >= 2:
            a[0] = 0.7
            a[1] = 0.3
            b[0] = 0.3
            b[1] = 0.7
        else:
            a[0] = 1.0
            b[0] = 1.0
        return DiagnosticScenario(
            game_state=obs,
            option_a=a, option_b=b,
            target_param="alpha",
            description="Safe vs risky allocation.",
            channel_means=stats["means"],
            channel_variances=stats["variances"],
            current_wealth=float(obs["wealth"].sum()),
            rounds_remaining=env.config.n_rounds,
        )
