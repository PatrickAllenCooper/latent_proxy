from __future__ import annotations

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.evaluation.experiment_runner import (
    ExperimentConfig,
    check_targets,
    run_full_evaluation,
    run_transfer_experiment,
)


def test_check_targets_flags() -> None:
    t = check_targets(
        mean_alignment=0.85,
        violation_rate=0.0,
        mean_gamma_mae=0.05,
        error_reduction_pct=35.0,
    )
    assert t.alignment_above_threshold
    assert t.quality_floor_clean
    assert t.gamma_mae_below_threshold
    assert t.elicitation_efficiency_met


def test_run_full_evaluation_smoke() -> None:
    conv = ConvergenceConfig(max_rounds=3)
    elic = ElicitationConfig(
        posterior_type="particle",
        n_particles=80,
        max_rounds=2,
        n_scenarios_per_round=6,
        n_eig_samples=32,
        convergence=conv,
        seed=0,
    )
    cfg = ExperimentConfig(
        variants=["a"],
        n_users=1,
        seed=1,
        elicitation=elic,
    )
    out = run_full_evaluation(cfg)
    assert "variants" in out
    assert "a" in out["variants"]
    block = out["variants"]["a"]
    assert "efficiency" in block
    assert "targets" in block
    assert "mean_alignment_active" in block
    assert "mean_total_recovery_error" in block


def test_transfer_experiment_three_conditions() -> None:
    conv = ConvergenceConfig(max_rounds=2)
    elic = ElicitationConfig(
        posterior_type="particle",
        n_particles=80,
        max_rounds=2,
        n_scenarios_per_round=6,
        n_eig_samples=32,
        convergence=conv,
        seed=2,
    )
    tr = run_transfer_experiment(n_users=1, elicitation=elic, seed=3)
    assert tr.n_users == 1
    for key in ("mean", "ci_low", "ci_high"):
        assert key in tr.generic
        assert key in tr.within_domain
        assert key in tr.cross_domain
    assert len(tr.per_user["generic"]) == 1
    assert len(tr.per_user["within_domain"]) == 1
    assert len(tr.per_user["cross_domain"]) == 1
