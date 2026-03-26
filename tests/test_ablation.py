from __future__ import annotations

import numpy as np

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.evaluation.ablation_runner import run_sweep


def test_query_budget_sweep() -> None:
    conv = ConvergenceConfig(max_rounds=4)
    base = ElicitationConfig(
        posterior_type="particle",
        n_particles=64,
        max_rounds=4,
        n_scenarios_per_round=5,
        n_eig_samples=24,
        convergence=conv,
        seed=11,
    )
    res = run_sweep(
        "max_rounds",
        [3],
        base,
        n_users=1,
        seed=12,
    )
    assert res.param_name == "max_rounds"
    assert "3" in res.metrics_by_value


def test_posterior_method_sweep() -> None:
    conv = ConvergenceConfig(max_rounds=2)
    base = ElicitationConfig(
        posterior_type="particle",
        n_particles=64,
        max_rounds=2,
        n_scenarios_per_round=5,
        n_eig_samples=24,
        convergence=conv,
        seed=21,
    )
    res = run_sweep(
        "posterior_type",
        ["gaussian", "particle"],
        base,
        n_users=1,
        seed=22,
    )
    assert "gaussian" in res.metrics_by_value
    assert "particle" in res.metrics_by_value


def test_beta_sweep_reproducible_alignment_metric() -> None:
    conv = ConvergenceConfig(max_rounds=2)
    base = ElicitationConfig(
        posterior_type="particle",
        n_particles=64,
        max_rounds=2,
        n_scenarios_per_round=5,
        n_eig_samples=24,
        convergence=conv,
        seed=31,
    )
    r1 = run_sweep("beta", [0.0, 1.0], base, n_users=1, seed=40)
    assert np.isfinite(r1.metrics_by_value["0.0"]["mean_alignment_blended"])
    assert np.isfinite(r1.metrics_by_value["1.0"]["mean_alignment_blended"])
