from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from src.agents.query_generator import DirichletQueryGenerator, FixedQueryGenerator
from src.environments.resource_game import ResourceStrategyGame
from src.utils.posterior import ParticlePosterior

FIXED_KWARGS = dict(
    n_scenarios_per_round=6,
    n_eig_samples=25,
    n_particles=80,
    seed=7,
)


@pytest.fixture
def env() -> ResourceStrategyGame:
    game = ResourceStrategyGame()
    game.reset(seed=123)
    return game


@pytest.fixture
def posterior() -> ParticlePosterior:
    return ParticlePosterior(n_particles=100)


def _fresh_env() -> ResourceStrategyGame:
    game = ResourceStrategyGame()
    game.reset(seed=123)
    return game


class TestDirichletQueryGenerator:
    def test_options_form_valid_simplex(
        self, env: ResourceStrategyGame, posterior: ParticlePosterior,
    ) -> None:
        gen = DirichletQueryGenerator(seed=7)
        scenario = gen.select_query(env, posterior)
        K = env.config.n_channels
        assert scenario.option_a.shape == (K,)
        assert scenario.option_b.shape == (K,)
        assert np.all(scenario.option_a >= 0.0)
        assert np.all(scenario.option_b >= 0.0)
        np.testing.assert_allclose(scenario.option_a.sum(), 1.0, atol=1e-9)
        np.testing.assert_allclose(scenario.option_b.sum(), 1.0, atol=1e-9)
        assert scenario.target_param == "unknown"

    def test_deterministic_given_seed(
        self, posterior: ParticlePosterior,
    ) -> None:
        gen1 = DirichletQueryGenerator(seed=7)
        gen2 = DirichletQueryGenerator(seed=7)
        env1 = _fresh_env()
        env2 = _fresh_env()
        for _ in range(3):
            s1 = gen1.select_query(env1, posterior)
            s2 = gen2.select_query(env2, posterior)
            np.testing.assert_allclose(s1.option_a, s2.option_a)
            np.testing.assert_allclose(s1.option_b, s2.option_b)
            np.testing.assert_allclose(s1.channel_means, s2.channel_means)
            assert s1.current_wealth == s2.current_wealth

    def test_varies_across_rounds(
        self, env: ResourceStrategyGame, posterior: ParticlePosterior,
    ) -> None:
        gen = DirichletQueryGenerator(seed=7)
        s1 = gen.select_query(env, posterior)
        s2 = gen.select_query(env, posterior)
        assert not np.allclose(s1.option_a, s2.option_a)


class TestFixedQueryGenerator:
    def test_identical_sequence_for_fresh_instances(
        self, posterior: ParticlePosterior,
    ) -> None:
        gen1 = FixedQueryGenerator(**FIXED_KWARGS)
        gen2 = FixedQueryGenerator(**FIXED_KWARGS)
        env1 = _fresh_env()
        env2 = _fresh_env()
        # gen2 sees a *different* posterior object: the questionnaire must
        # not depend on it.
        other_posterior = ParticlePosterior(n_particles=50)
        for _ in range(6):
            s1 = gen1.select_query(env1, posterior)
            s2 = gen2.select_query(env2, other_posterior)
            assert s1.target_param == s2.target_param
            np.testing.assert_allclose(s1.option_a, s2.option_a)
            np.testing.assert_allclose(s1.option_b, s2.option_b)

    def test_static_regardless_of_responses(
        self, posterior: ParticlePosterior,
    ) -> None:
        gen1 = FixedQueryGenerator(**FIXED_KWARGS)
        gen2 = FixedQueryGenerator(**FIXED_KWARGS)
        env1 = _fresh_env()
        env2 = _fresh_env()
        updated = ParticlePosterior(n_particles=100)
        seq1 = []
        seq2 = []
        for _ in range(4):
            seq1.append(gen1.select_query(env1, posterior))
            s2 = gen2.select_query(env2, updated)
            seq2.append(s2)
            # Simulate an observed choice between rounds for gen2 only.
            updated.update_from_choice(
                choice=0,
                option_a_alloc=s2.option_a,
                option_b_alloc=s2.option_b,
                channel_means=s2.channel_means,
                channel_variances=s2.channel_variances,
                current_wealth=s2.current_wealth,
                rounds_remaining=s2.rounds_remaining,
                temperature=0.1,
                multiperiod_horizon=s2.multiperiod_horizon,
            )
        for s1, s2 in zip(seq1, seq2):
            assert s1.target_param == s2.target_param
            np.testing.assert_allclose(s1.option_a, s2.option_a)
            np.testing.assert_allclose(s1.option_b, s2.option_b)

    def test_round_robin_balances_target_params(
        self, env: ResourceStrategyGame, posterior: ParticlePosterior,
    ) -> None:
        gen = FixedQueryGenerator(**FIXED_KWARGS)
        sequence = [gen.select_query(env, posterior) for _ in range(6)]
        distinct = {s.target_param for s in sequence}
        head = [s.target_param for s in sequence[: len(distinct)]]
        assert len(set(head)) == len(distinct)

    def test_options_are_valid(
        self, env: ResourceStrategyGame, posterior: ParticlePosterior,
    ) -> None:
        gen = FixedQueryGenerator(**FIXED_KWARGS)
        scenario = gen.select_query(env, posterior)
        np.testing.assert_allclose(scenario.option_a.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(scenario.option_b.sum(), 1.0, atol=1e-6)


def _load_campaign_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "run_canonical_campaign.py"
    )
    spec = importlib.util.spec_from_file_location("run_canonical_campaign", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMiniCampaign:
    def test_end_to_end(self, tmp_path: Path) -> None:
        mod = _load_campaign_module()
        argv = [
            "--domains", "game_a",
            "--arms", "fixed,dirichlet",
            "--n-users", "2",
            "--seeds", "42",
            "--max-rounds", "2",
            "--n-particles", "60",
            "--n-eig-samples", "25",
            "--n-scenarios-per-round", "6",
            "--workers", "1",
            "--output-dir", str(tmp_path),
        ]
        args = mod.parse_args(argv)
        summary = mod.run_campaign(args)
        assert summary["n_completed"] == 4
        assert summary["n_failed"] == 0

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["config"]["n_users"] == 2
        assert manifest["config"]["max_rounds"] == 2
        assert "git_sha" in manifest

        records: dict[tuple[str, int], dict] = {}
        for arm in ("fixed", "dirichlet"):
            for idx in (0, 1):
                p = tmp_path / "game_a" / arm / f"s42_u{idx}.json"
                assert p.exists(), f"missing {p}"
                records[(arm, idx)] = json.loads(p.read_text())

        # Paired design: same users across arms for a given (domain, seed).
        for idx in (0, 1):
            assert (
                records[("fixed", idx)]["true_theta"]
                == records[("dirichlet", idx)]["true_theta"]
            )
        assert (
            records[("fixed", 0)]["true_theta"]
            != records[("fixed", 1)]["true_theta"]
        )

        rec = records[("fixed", 0)]
        assert rec["n_rounds"] == 2
        assert rec["convergence_reason"] == "max_rounds"
        assert len(rec["mean_trajectory"]) == 3  # prior + one per round
        assert len(rec["variance_trajectory"]) == 3
        assert len(rec["per_round_error"]) == 3
        assert len(rec["history"]) == 2
        round0 = rec["history"][0]
        for key in (
            "option_a", "option_b", "channel_means", "channel_variances",
            "current_wealth", "rounds_remaining", "target_param",
            "multiperiod_horizon", "choice",
        ):
            assert key in round0
        assert round0["choice"] in (0, 1)
        assert -1.0 <= rec["alignment"]["spearman"] <= 1.0
        assert len(rec["alignment"]["optimal_action_true"]) > 0

        # Resumable: a second run skips every existing user file.
        summary2 = mod.run_campaign(args)
        assert summary2["n_completed"] == 0
        assert summary2["n_skipped"] == 4
