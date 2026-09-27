from __future__ import annotations

import copy
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
from src.training.synthetic_users import prospect_utility, resolve_reference_point


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


class ActiveInferenceQueryGenerator:
    """Balance epistemic information and pragmatic decision value.

    For every candidate question, this estimates (1) mutual information between
    the user's preference type and answer, and (2) expected value of sample
    information for the downstream allocation decision. The two scores are
    normalized within the candidate pool and combined with an explicit
    epistemic weight, following the epistemic/pragmatic split in active
    inference.

    The decision value is estimated over a finite menu of allocations induced
    by posterior particles. Utility uses deterministic Gauss-Hermite quadrature
    for the final one-period allocation and common-random-number Monte Carlo for
    multi-period query scenarios. This makes it a practical decision-aware
    acquisition function, rather than a claim of exact Bayesian optimality.
    """

    def __init__(
        self,
        n_scenarios_per_round: int = 50,
        n_value_particles: int = 96,
        n_query_samples: int = 64,
        n_quadrature_points: int = 20,
        epistemic_weight: float = 0.5,
        temperature: float = 0.1,
        seed: int = 42,
        library: ScenarioLibraryBase | None = None,
        reference_point_mode: str = "zero",
        utility_form: str = "absolute",
    ) -> None:
        if not 0.0 <= epistemic_weight <= 1.0:
            raise ValueError("epistemic_weight must be in [0, 1]")
        self.n_scenarios_per_round = n_scenarios_per_round
        self.n_value_particles = n_value_particles
        self.n_query_samples = n_query_samples
        self.temperature = temperature
        self.epistemic_weight = epistemic_weight
        self.reference_point_mode = reference_point_mode
        self.utility_form = utility_form
        self._library = library or ScenarioLibrary(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._decision_env: BaseEnvironment | None = None
        self._gh_nodes, self._gh_weights = np.polynomial.hermite.hermgauss(
            n_quadrature_points,
        )
        self._gh_weights = self._gh_weights / np.sqrt(np.pi)
        self.last_score_components: dict[str, float] = {}

    @staticmethod
    def _entropy(probabilities: NDArray[np.float64]) -> float:
        p = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        return float(-np.sum(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))

    def _sample_posterior(
        self, posterior: PosteriorBase,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = self.n_value_particles
        if isinstance(posterior, ParticlePosterior):
            n = min(n, posterior.n_particles)
            idx = self._rng.choice(
                posterior.n_particles, size=n, replace=True, p=posterior.weights,
            )
            particles = posterior.particles[idx].copy()
        else:
            particles = posterior.sample(n, self._rng)
        return np.asarray(particles, dtype=np.float64), np.full(n, 1.0 / n)

    def _expected_allocation_utility(
        self,
        particles: NDArray[np.float64],
        allocation: NDArray[np.float64],
        means: NDArray[np.float64],
        variances: NDArray[np.float64],
        current_wealth: float,
        rounds_remaining: int,
        reference_point: float,
        *,
        noise: NDArray[np.float64] | None = None,
        multiperiod_horizon: int | None = None,
    ) -> NDArray[np.float64]:
        """Vectorized approximation of SyntheticUser's expected utility."""
        port_mean = float(np.dot(allocation, means))
        port_var = float(np.dot(np.square(allocation), variances))
        alpha = particles[:, 1]
        loss_aversion = particles[:, 2]
        gamma = np.clip(particles[:, 0], 1e-6, 1.0)

        if multiperiod_horizon is not None and multiperiod_horizon > 1:
            horizon = int(multiperiod_horizon)
            if noise is None:
                noise = self._rng.standard_normal((self.n_query_samples, horizon))
            simulated_returns = port_mean + np.sqrt(max(port_var, 0.0)) * noise
            wealth_paths = current_wealth * np.cumprod(1.0 + simulated_returns, axis=1)
            total = np.zeros((len(particles), simulated_returns.shape[0]))
            for t in range(horizon):
                outcomes = (
                    simulated_returns[:, t]
                    if self.utility_form == "return_normalized"
                    else wealth_paths[:, t]
                )
                deviation = outcomes[None, :] - reference_point
                exponent = 1.0 / (1.0 + alpha[:, None])
                utility = np.where(
                    deviation >= 0,
                    np.maximum(deviation, 0.0) ** exponent,
                    -loss_aversion[:, None]
                    * np.maximum(-deviation, 0.0) ** exponent,
                )
                total += np.power(gamma[:, None], t) * utility
            return np.mean(total, axis=1)

        returns = (
            port_mean
            + np.sqrt(max(2.0 * port_var, 0.0)) * self._gh_nodes
        )
        outcomes = (
            returns
            if self.utility_form == "return_normalized"
            else current_wealth * (1.0 + returns)
        )
        deviation = outcomes[None, :] - reference_point
        exponent = 1.0 / (1.0 + alpha[:, None])
        utility = np.where(
            deviation >= 0,
            np.maximum(deviation, 0.0) ** exponent,
            -loss_aversion[:, None] * np.maximum(-deviation, 0.0) ** exponent,
        )
        expectation = utility @ self._gh_weights
        return expectation * np.power(gamma, max(1, rounds_remaining))

    def _decision_utility_matrix(
        self,
        particles: NDArray[np.float64],
        actions: NDArray[np.float64],
        env: BaseEnvironment,
    ) -> NDArray[np.float64]:
        obs = env._get_obs()
        stats = env.get_channel_stats()
        wealth = float(np.asarray(obs["wealth"]).sum())
        rounds_left = max(1, int(env.config.n_rounds - obs["round"]))
        ref = resolve_reference_point(
            wealth, self.reference_point_mode, self.utility_form,
        )
        return np.stack([
            self._expected_allocation_utility(
                particles, action, stats["means"], stats["variances"],
                wealth, rounds_left, ref,
            )
            for action in actions
        ], axis=1)

    def _candidate_actions(
        self, particles: NDArray[np.float64], posterior: PosteriorBase,
    ) -> NDArray[np.float64]:
        assert self._decision_env is not None
        actions = [
            self._decision_env.get_optimal_action({
                "gamma": float(theta[0]),
                "alpha": float(theta[1]),
                "lambda_": float(theta[2]),
            })
            for theta in particles
        ]
        actions.append(self._decision_env.get_optimal_action(posterior.to_dict()))
        actions.append(np.ones(self._decision_env.config.n_channels) / self._decision_env.config.n_channels)
        return np.unique(np.round(np.asarray(actions, dtype=np.float64), decimals=7), axis=0)

    def _choice_probabilities(
        self,
        particles: NDArray[np.float64],
        scenario: DiagnosticScenario,
    ) -> NDArray[np.float64]:
        ref = resolve_reference_point(
            scenario.current_wealth, self.reference_point_mode, self.utility_form,
        )
        horizon = scenario.multiperiod_horizon
        common_noise = None
        if horizon is not None and horizon > 1:
            common_noise = self._rng.standard_normal((self.n_query_samples, int(horizon)))
        utility_a = self._expected_allocation_utility(
            particles, scenario.option_a, scenario.channel_means,
            scenario.channel_variances, scenario.current_wealth,
            scenario.rounds_remaining, ref, noise=common_noise,
            multiperiod_horizon=horizon,
        )
        utility_b = self._expected_allocation_utility(
            particles, scenario.option_b, scenario.channel_means,
            scenario.channel_variances, scenario.current_wealth,
            scenario.rounds_remaining, ref, noise=common_noise,
            multiperiod_horizon=horizon,
        )
        logit = np.clip(
            (utility_a - utility_b) / max(self.temperature, 1e-10),
            -500.0, 500.0,
        )
        return 1.0 / (1.0 + np.exp(-logit))

    def _score_candidates(
        self,
        scenarios: list[DiagnosticScenario],
        particles: NDArray[np.float64],
        weights: NDArray[np.float64],
        decision_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        current_value = float(np.max(weights @ decision_values))
        information_scores = []
        decision_scores = []
        for scenario in scenarios:
            p_choose_a = self._choice_probabilities(particles, scenario)
            marginal_a = float(np.dot(weights, p_choose_a))
            marginal_a = float(np.clip(marginal_a, 1e-12, 1.0 - 1e-12))
            information = self._entropy(np.asarray([marginal_a])) - float(
                np.dot(weights, self._binary_entropies(p_choose_a))
            )
            information_scores.append(max(information, 0.0))

            expected_after = 0.0
            for likelihood, probability in (
                (p_choose_a, marginal_a),
                (1.0 - p_choose_a, 1.0 - marginal_a),
            ):
                updated = weights * likelihood / max(probability, 1e-12)
                best_after = float(np.max(updated @ decision_values))
                expected_after += probability * best_after
            decision_scores.append(max(expected_after - current_value, 0.0))

        info = np.asarray(information_scores, dtype=np.float64)
        evsi = np.asarray(decision_scores, dtype=np.float64)
        if info.max(initial=0.0) > 0:
            info = info / info.max()
        if evsi.max(initial=0.0) > 0:
            evsi = evsi / evsi.max()
        epistemic = self.epistemic_weight * info
        pragmatic = (1.0 - self.epistemic_weight) * evsi
        scores = epistemic + pragmatic
        best_idx = int(np.argmax(scores))
        self.last_score_components = {
            "epistemic_weight": float(self.epistemic_weight),
            "selected_mutual_information_nats": float(information_scores[best_idx]),
            "selected_normalized_evsi": float(evsi[best_idx]),
            "selected_epistemic_score": float(epistemic[best_idx]),
            "selected_pragmatic_score": float(pragmatic[best_idx]),
            "selected_combined_score": float(scores[best_idx]),
        }
        return scores

    @staticmethod
    def _binary_entropies(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
        p = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)

    def select_query(
        self, env: BaseEnvironment, posterior: PosteriorBase,
    ) -> DiagnosticScenario:
        if self._decision_env is None:
            # Scenario-library generation mutates env state. Preserve the
            # actual deployment decision context before generating questions.
            self._decision_env = copy.deepcopy(env)

        particles, weights = self._sample_posterior(posterior)
        actions = self._candidate_actions(particles, posterior)
        decision_values = self._decision_utility_matrix(
            particles, actions, self._decision_env,
        )

        n_per_param = max(self.n_scenarios_per_round // 3, 3)
        scenarios = self._library.generate_all(env, n_per_param)
        if not scenarios:
            scenarios = self._library.generate_all(env, n_per_param * 2)
        if not scenarios:
            return RandomQueryGenerator(
                seed=int(self._rng.integers(2**31)),
            ).select_query(env, posterior)

        scores = self._score_candidates(
            scenarios, particles, weights, decision_values,
        )
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
