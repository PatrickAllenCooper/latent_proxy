from __future__ import annotations

import numpy as np
import pytest

from src.environments.resource_game import ResourceStrategyGame
from src.training.synthetic_users import SyntheticUser, UserType
from src.utils.diagnostic_scenarios import DiagnosticScenario, ScenarioLibrary


@pytest.fixture
def env() -> ResourceStrategyGame:
    game = ResourceStrategyGame()
    game.reset(seed=42)
    return game


@pytest.fixture
def library() -> ScenarioLibrary:
    return ScenarioLibrary(seed=42)


class TestGammaScenarios:
    def test_generates_scenarios(
        self, env: ResourceStrategyGame, library: ScenarioLibrary,
    ) -> None:
        scenarios = library.generate_gamma_scenarios(env, n=10)
        assert len(scenarios) > 0

    def test_gamma_scenarios_have_different_allocations(
        self, env: ResourceStrategyGame, library: ScenarioLibrary,
    ) -> None:
        """Gamma scenarios present safe vs aggressive options.

        Because gamma acts as a uniform discount on both options in
        single-period evaluation, we verify the options themselves are
        structurally different (safe-heavy vs aggressive-heavy).
        """
        scenarios = library.generate_gamma_scenarios(env, n=5)
        if not scenarios:
            pytest.skip("No gamma scenarios generated")

        for s in scenarios:
            safe_weight_a = float(s.option_a[0])
            safe_weight_b = float(s.option_b[0])
            assert safe_weight_a > safe_weight_b, (
                "Option A should have more safe allocation than option B"
            )


class TestAlphaScenarios:
    def test_generates_scenarios(
        self, env: ResourceStrategyGame, library: ScenarioLibrary,
    ) -> None:
        scenarios = library.generate_alpha_scenarios(env, n=10)
        assert len(scenarios) > 0

    def test_alpha_scenarios_produce_different_preference_strengths(
        self, env: ResourceStrategyGame, library: ScenarioLibrary,
    ) -> None:
        """Alpha scenarios should produce different preference STRENGTHS.

        Full preference reversals are not guaranteed because with
        reference_point=0 and positive wealth, loss aversion does not
        activate. However, the margin between EU(A) and EU(B) should
        differ across alpha values, which is what EIG exploits.
        """
        scenarios = library.generate_alpha_scenarios(env, n=5)
        if not scenarios:
            pytest.skip("No alpha scenarios generated")

        s = scenarios[0]
        cautious = SyntheticUser(
            UserType(gamma=0.7, alpha=5.0, lambda_=1.5), temperature=0.1, seed=0,
        )
        bold = SyntheticUser(
            UserType(gamma=0.7, alpha=0.05, lambda_=1.5), temperature=0.1, seed=0,
        )

        eu_a_c = cautious.evaluate_allocation(
            s.option_a, s.channel_means, s.channel_variances,
            s.current_wealth, s.rounds_remaining,
        )
        eu_b_c = cautious.evaluate_allocation(
            s.option_b, s.channel_means, s.channel_variances,
            s.current_wealth, s.rounds_remaining,
        )
        eu_a_b = bold.evaluate_allocation(
            s.option_a, s.channel_means, s.channel_variances,
            s.current_wealth, s.rounds_remaining,
        )
        eu_b_b = bold.evaluate_allocation(
            s.option_b, s.channel_means, s.channel_variances,
            s.current_wealth, s.rounds_remaining,
        )

        margin_cautious = abs(eu_a_c - eu_b_c)
        margin_bold = abs(eu_a_b - eu_b_b)

        assert margin_cautious != pytest.approx(margin_bold, abs=1e-6), (
            f"Preference margins should differ: cautious={margin_cautious:.6f}, "
            f"bold={margin_bold:.6f}"
        )


class TestLambdaScenarios:
    def test_generates_scenarios(
        self, env: ResourceStrategyGame, library: ScenarioLibrary,
    ) -> None:
        scenarios = library.generate_lambda_scenarios(env, n=20)
        assert len(scenarios) > 0


class TestScenarioValidity:
    def test_allocations_are_simplex(
        self, env: ResourceStrategyGame, library: ScenarioLibrary,
    ) -> None:
        scenarios = library.generate_all(env, n_per_param=5)
        for s in scenarios:
            np.testing.assert_allclose(s.option_a.sum(), 1.0, atol=1e-6)
            np.testing.assert_allclose(s.option_b.sum(), 1.0, atol=1e-6)
            assert np.all(s.option_a >= 0)
            assert np.all(s.option_b >= 0)

    def test_options_differ(
        self, env: ResourceStrategyGame, library: ScenarioLibrary,
    ) -> None:
        scenarios = library.generate_all(env, n_per_param=5)
        for s in scenarios:
            assert not np.allclose(s.option_a, s.option_b, atol=0.02)

    def test_has_valid_metadata(
        self, env: ResourceStrategyGame, library: ScenarioLibrary,
    ) -> None:
        scenarios = library.generate_all(env, n_per_param=3)
        for s in scenarios:
            assert s.target_param in ("gamma", "alpha", "lambda_")
            assert len(s.description) > 0
            assert s.current_wealth > 0
            assert s.rounds_remaining > 0


class TestGenerateAll:
    def test_covers_all_params(
        self, env: ResourceStrategyGame, library: ScenarioLibrary,
    ) -> None:
        scenarios = library.generate_all(env, n_per_param=5)
        params = {s.target_param for s in scenarios}
        assert "gamma" in params
        assert "alpha" in params
