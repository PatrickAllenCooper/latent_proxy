from __future__ import annotations

import numpy as np
import pytest

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop, ElicitationResult
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.resource_game import ResourceStrategyGame
from src.training.synthetic_users import SyntheticUser, UserType


@pytest.fixture
def env() -> ResourceStrategyGame:
    game = ResourceStrategyGame()
    game.reset(seed=42)
    return game


def _make_config(
    posterior_type: str = "particle",
    max_rounds: int = 8,
    n_particles: int = 300,
    n_scenarios: int = 15,
    n_eig: int = 100,
    seed: int = 42,
) -> ElicitationConfig:
    return ElicitationConfig(
        posterior_type=posterior_type,
        n_particles=n_particles,
        max_rounds=max_rounds,
        n_scenarios_per_round=n_scenarios,
        n_eig_samples=n_eig,
        temperature=0.1,
        convergence=ConvergenceConfig(
            gamma_variance_threshold=0.003,
            alpha_variance_threshold=0.03,
            lambda_variance_threshold=0.03,
            max_rounds=max_rounds,
        ),
        seed=seed,
    )


class TestElicitationLoopBasics:
    def test_runs_without_error(self, env: ResourceStrategyGame) -> None:
        config = _make_config(max_rounds=3, n_particles=100, n_scenarios=6, n_eig=50)
        user = SyntheticUser(
            UserType(gamma=0.8, alpha=1.0, lambda_=1.5), temperature=0.1, seed=0,
        )
        loop = ElicitationLoop(config)
        result = loop.run(env, user, query_type="active")
        assert isinstance(result, ElicitationResult)
        assert result.n_rounds > 0
        assert result.n_rounds <= 3

    def test_history_recorded(self, env: ResourceStrategyGame) -> None:
        config = _make_config(max_rounds=3, n_particles=100, n_scenarios=6, n_eig=50)
        user = SyntheticUser(
            UserType(gamma=0.7, alpha=1.5, lambda_=2.0), temperature=0.1, seed=0,
        )
        loop = ElicitationLoop(config)
        result = loop.run(env, user)
        assert len(result.history) == result.n_rounds
        for scenario, choice in result.history:
            assert choice in (0, 1)

    def test_variance_trajectory(self, env: ResourceStrategyGame) -> None:
        config = _make_config(max_rounds=3, n_particles=100, n_scenarios=6, n_eig=50)
        user = SyntheticUser(
            UserType(gamma=0.6, alpha=1.0, lambda_=1.5), temperature=0.1, seed=0,
        )
        loop = ElicitationLoop(config)
        result = loop.run(env, user)
        assert len(result.variance_trajectory) == result.n_rounds + 1

    def test_recovery_error_computable(self, env: ResourceStrategyGame) -> None:
        config = _make_config(max_rounds=3, n_particles=100, n_scenarios=6, n_eig=50)
        user = SyntheticUser(
            UserType(gamma=0.8, alpha=1.0, lambda_=1.5), temperature=0.1, seed=0,
        )
        loop = ElicitationLoop(config)
        result = loop.run(env, user)
        err = result.preference_recovery_error()
        assert err is not None
        assert "gamma" in err
        assert "total" in err


class TestElicitationTermination:
    def test_terminates_at_max_rounds(self, env: ResourceStrategyGame) -> None:
        config = _make_config(
            max_rounds=4, n_particles=100, n_scenarios=6, n_eig=50,
        )
        config.convergence.gamma_variance_threshold = 0.0
        config.convergence.alpha_variance_threshold = 0.0
        config.convergence.lambda_variance_threshold = 0.0
        user = SyntheticUser(
            UserType(gamma=0.5, alpha=1.0, lambda_=1.5), temperature=0.1, seed=0,
        )
        loop = ElicitationLoop(config)
        result = loop.run(env, user)
        assert result.n_rounds == 4
        assert result.convergence_reason == "max_rounds"


class TestRandomVsActive:
    def test_both_strategies_run(self, env: ResourceStrategyGame) -> None:
        config = _make_config(max_rounds=3, n_particles=100, n_scenarios=6, n_eig=50)
        user_type = UserType(gamma=0.8, alpha=1.0, lambda_=1.5)

        user_a = SyntheticUser(user_type, temperature=0.1, seed=0)
        loop_a = ElicitationLoop(config)
        result_a = loop_a.run(env, user_a, query_type="active")

        user_r = SyntheticUser(user_type, temperature=0.1, seed=0)
        loop_r = ElicitationLoop(config)
        result_r = loop_r.run(env, user_r, query_type="random")

        assert result_a.n_rounds > 0
        assert result_r.n_rounds > 0

    def test_active_lower_error_trend(self, env: ResourceStrategyGame) -> None:
        """Over multiple users, active should tend toward lower error.

        This is a soft test -- we check that active is not dramatically
        worse than random, not that it always wins on every user.
        """
        config = _make_config(
            max_rounds=6, n_particles=200, n_scenarios=12, n_eig=100,
        )

        active_errors = []
        random_errors = []

        for seed in range(5):
            ut = UserType(gamma=0.8, alpha=1.0, lambda_=1.5)

            user_a = SyntheticUser(ut, temperature=0.1, seed=seed)
            loop_a = ElicitationLoop(ElicitationConfig(
                **{**config.__dict__, "seed": seed * 100},
            ))
            res_a = loop_a.run(env, user_a, query_type="active")
            err_a = res_a.preference_recovery_error()
            if err_a:
                active_errors.append(err_a["total"])

            user_r = SyntheticUser(ut, temperature=0.1, seed=seed)
            loop_r = ElicitationLoop(ElicitationConfig(
                **{**config.__dict__, "seed": seed * 200},
            ))
            res_r = loop_r.run(env, user_r, query_type="random")
            err_r = res_r.preference_recovery_error()
            if err_r:
                random_errors.append(err_r["total"])

        mean_active = np.mean(active_errors) if active_errors else 999
        mean_random = np.mean(random_errors) if random_errors else 999

        assert mean_active < mean_random * 2.0, (
            f"Active ({mean_active:.3f}) should not be much worse than "
            f"random ({mean_random:.3f})"
        )


class TestBothPosteriorTypes:
    @pytest.mark.parametrize("posterior_type", ["particle", "gaussian"])
    def test_runs_with_posterior_type(
        self, env: ResourceStrategyGame, posterior_type: str,
    ) -> None:
        config = _make_config(
            posterior_type=posterior_type,
            max_rounds=3, n_particles=100, n_scenarios=6, n_eig=50,
        )
        user = SyntheticUser(
            UserType(gamma=0.7, alpha=1.0, lambda_=1.5), temperature=0.1, seed=0,
        )
        loop = ElicitationLoop(config)
        result = loop.run(env, user)
        assert result.n_rounds > 0
        err = result.preference_recovery_error()
        assert err is not None
