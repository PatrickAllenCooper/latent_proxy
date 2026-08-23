"""Sanity experiment: the reference point controls whether lambda_ is identifiable.

With the historical reference point of 0.0 and positive wealth, simulated
outcomes never fall below the reference, so the loss branch of the prospect
value function never fires and loss aversion (lambda_) has exactly zero
effect on choices. With reference_point = the scenario's current wealth,
negative returns register as losses and lambda_ becomes identifiable.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop
from src.agents.preference_tracker import ConvergenceConfig, PreferenceTracker
from src.environments.resource_game import ResourceStrategyGame
from src.training.synthetic_users import SyntheticUser, UserType
from src.utils.diagnostic_scenarios import DiagnosticScenario, ScenarioLibraryBase

# A scenario with a meaningful downside channel: the risky option has a
# high-variance return stream, so ~38% of simulated returns are negative.
CURRENT_WEALTH = 100.0
CHANNEL_MEANS = np.array([0.01, 0.06])
CHANNEL_VARIANCES = np.array([1e-6, 0.04])
SAFE = np.array([1.0, 0.0])
RISKY = np.array([0.0, 1.0])


def _eu_difference(
    lambda_: float,
    reference_point: float | None,
    utility_form: str = "absolute",
) -> float:
    """EU(risky) - EU(safe) for a user with the given lambda_.

    A fixed seed makes the Monte Carlo draws identical across calls, so
    any difference between users is due to preferences, not noise.
    """
    user = SyntheticUser(
        UserType(gamma=0.9, alpha=0.8, lambda_=lambda_),
        temperature=0.1,
        seed=123,
        utility_form=utility_form,
    )
    eu_safe = user.evaluate_for_query(
        SAFE, CHANNEL_MEANS, CHANNEL_VARIANCES, CURRENT_WEALTH,
        rounds_remaining=1, reference_point=reference_point,
    )
    eu_risky = user.evaluate_for_query(
        RISKY, CHANNEL_MEANS, CHANNEL_VARIANCES, CURRENT_WEALTH,
        rounds_remaining=1, reference_point=reference_point,
    )
    return eu_risky - eu_safe


class TestReferencePointControlsLambda:
    def test_zero_mode_lambda_has_no_effect(self) -> None:
        """With ref=0 and positive wealth, lambda_=1 and lambda_=3 users are
        indistinguishable: identical EU difference between risky and safe."""
        d_lo = _eu_difference(lambda_=1.0, reference_point=None)
        d_hi = _eu_difference(lambda_=3.0, reference_point=None)
        assert d_lo == pytest.approx(d_hi, abs=1e-10)

    def test_current_wealth_mode_lambda_matters(self) -> None:
        """With ref=current_wealth, negative returns are losses and the
        loss-averse user penalizes the risky option strictly more."""
        d_lo = _eu_difference(lambda_=1.0, reference_point=CURRENT_WEALTH)
        d_hi = _eu_difference(lambda_=3.0, reference_point=CURRENT_WEALTH)
        # The high-lambda user dislikes the risky option strictly more,
        # by a clear margin (not MC noise: draws are seed-matched).
        assert d_hi < d_lo - 0.5

    def test_default_reference_point_unchanged(self) -> None:
        """Omitting reference_point must reproduce the historical behavior."""
        user = SyntheticUser(
            UserType(gamma=0.9, alpha=0.8, lambda_=2.0), temperature=0.1, seed=7,
        )
        eu_default = user.evaluate_for_query(
            RISKY, CHANNEL_MEANS, CHANNEL_VARIANCES, CURRENT_WEALTH,
            rounds_remaining=1,
        )
        user2 = SyntheticUser(
            UserType(gamma=0.9, alpha=0.8, lambda_=2.0), temperature=0.1, seed=7,
        )
        eu_explicit_zero = user2.evaluate_for_query(
            RISKY, CHANNEL_MEANS, CHANNEL_VARIANCES, CURRENT_WEALTH,
            rounds_remaining=1, reference_point=0.0,
        )
        assert eu_default == pytest.approx(eu_explicit_zero, abs=1e-12)

    def test_return_normalized_also_makes_lambda_identifiable(self) -> None:
        """utility_form='return_normalized' (default reference_point_mode="zero")
        achieves the same lambda-identifiability fix as current_wealth mode --
        without the cross-scenario wealth-scale contamination that mode alone
        does not address (see outputs/canonical vs outputs/canonical_cw)."""
        d_lo = _eu_difference(
            lambda_=1.0, reference_point=None, utility_form="return_normalized",
        )
        d_hi = _eu_difference(
            lambda_=3.0, reference_point=None, utility_form="return_normalized",
        )
        assert d_hi < d_lo - 0.01


class TestTrackerModeValidation:
    def test_bad_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="reference_point_mode"):
            PreferenceTracker(reference_point_mode="wealth")

    def test_return_normalized_with_current_wealth_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="return_normalized"):
            PreferenceTracker(
                reference_point_mode="current_wealth",
                utility_form="return_normalized",
            )


class DownsideScenarioLibrary(ScenarioLibraryBase):
    """Handcrafted scenarios with a genuine negative-return channel.

    Lambda scenarios pit a near-certain small gain against a high-variance
    option whose indifference point sweeps across the lambda_ prior range,
    so (under a wealth reference point) each choice acts like a threshold
    query on lambda_. Alpha scenarios are gains-only contrasts that pin
    curvature without touching the loss branch.
    """

    W = 100.0

    def _scenario(
        self,
        option_a: np.ndarray,
        option_b: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
        target: str,
    ) -> DiagnosticScenario:
        return DiagnosticScenario(
            game_state={},
            option_a=option_a,
            option_b=option_b,
            target_param=target,
            description=f"downside test scenario ({target})",
            channel_means=means,
            channel_variances=variances,
            current_wealth=self.W,
            rounds_remaining=1,
        )

    def generate_gamma_scenarios(self, env, n=10):  # noqa: ANN001, ANN201
        return []

    def generate_alpha_scenarios(self, env, n=10):  # noqa: ANN001, ANN201
        scenarios = []
        for w in np.linspace(0.3, 0.7, n):
            scenarios.append(self._scenario(
                np.array([1.0, 0.0]),
                np.array([1.0 - w, w]),
                means=np.array([0.01, 0.10]),
                variances=np.array([1e-6, 0.0009]),
                target="alpha",
            ))
        return scenarios

    def generate_lambda_scenarios(self, env, n=10):  # noqa: ANN001, ANN201
        scenarios = []
        for mu in np.linspace(0.06, 0.18, n):
            scenarios.append(self._scenario(
                np.array([1.0, 0.0]),
                np.array([0.0, 1.0]),
                means=np.array([0.01, float(mu)]),
                variances=np.array([1e-6, 0.0625]),
                target="lambda_",
            ))
        return scenarios


def _run_mini_elicitation(mode: str) -> tuple[float, float]:
    """Run a 10-round elicitation; return (initial, final) lambda_ variance."""
    env = ResourceStrategyGame()
    env.reset(seed=42)
    user = SyntheticUser(
        UserType(gamma=0.7, alpha=1.0, lambda_=2.2), temperature=0.1, seed=11,
    )
    config = ElicitationConfig(
        posterior_type="particle",
        n_particles=400,
        max_rounds=10,
        n_scenarios_per_round=15,
        n_eig_samples=60,
        temperature=0.1,
        convergence=ConvergenceConfig(
            # Tiny thresholds so the loop always runs all 10 rounds.
            gamma_variance_threshold=1e-9,
            alpha_variance_threshold=1e-9,
            lambda_variance_threshold=1e-9,
            max_rounds=10,
        ),
        seed=5,
        scenario_library=DownsideScenarioLibrary(),
        reference_point_mode=mode,
    )
    loop = ElicitationLoop(config)
    result = loop.run(env, user, query_type="active")
    assert result.n_rounds == 10
    v0 = result.variance_trajectory[0]["lambda_"]
    vf = result.variance_trajectory[-1]["lambda_"]
    return v0, vf


class TestMiniElicitationLambdaIdentifiability:
    """Smoke assertion: under current_wealth mode the lambda_ posterior
    contracts over 10 rounds; under zero mode it stays ~flat (only
    resampling noise moves it, since the likelihood is lambda-free)."""

    def test_lambda_variance_contracts_only_with_wealth_reference(self) -> None:
        v0_zero, vf_zero = _run_mini_elicitation("zero")
        v0_cw, vf_cw = _run_mini_elicitation("current_wealth")

        ratio_zero = vf_zero / v0_zero
        ratio_cw = vf_cw / v0_cw

        # current_wealth mode: clear contraction of the lambda_ posterior.
        assert ratio_cw < 0.75, (
            f"expected lambda_ variance contraction under current_wealth mode, "
            f"got final/initial = {ratio_cw:.3f}"
        )
        # zero mode: approximately flat (tolerant band for resampling noise).
        assert ratio_zero > 0.75, (
            f"expected ~flat lambda_ variance under zero mode, "
            f"got final/initial = {ratio_zero:.3f}"
        )
        # And the two modes differ by a clear margin.
        assert ratio_cw < ratio_zero - 0.15


class TestUtilityFormMismatchGuard:
    """ElicitationLoop must reject a user/config utility_form disagreement --
    the generative process (synthetic user) and the inference model
    (posterior/EIG, driven by config) must agree on which formula is used."""

    def test_mismatched_user_and_config_raises(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=1)
        user = SyntheticUser(
            UserType(gamma=0.7, alpha=1.0, lambda_=1.5),
            temperature=0.1, seed=1, utility_form="absolute",
        )
        config = ElicitationConfig(
            n_particles=50, max_rounds=2, n_scenarios_per_round=6,
            n_eig_samples=20, utility_form="return_normalized",
        )
        loop = ElicitationLoop(config)
        with pytest.raises(ValueError, match="utility_form mismatch"):
            loop.run(env, user, query_type="active")

    def test_matched_user_and_config_runs(self) -> None:
        env = ResourceStrategyGame()
        env.reset(seed=1)
        user = SyntheticUser(
            UserType(gamma=0.7, alpha=1.0, lambda_=1.5),
            temperature=0.1, seed=1, utility_form="return_normalized",
        )
        config = ElicitationConfig(
            n_particles=50, max_rounds=2, n_scenarios_per_round=6,
            n_eig_samples=20, utility_form="return_normalized",
        )
        loop = ElicitationLoop(config)
        result = loop.run(env, user, query_type="active")
        assert result.n_rounds == 2
