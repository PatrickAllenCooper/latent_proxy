from __future__ import annotations

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.evaluation.generalization_protocol import (
    GeneralizationStudyConfig,
    run_generalization_study,
)


def _tiny_config():
    conv = ConvergenceConfig(max_rounds=2)
    elic = ElicitationConfig(
        posterior_type="particle",
        n_particles=64,
        max_rounds=2,
        n_scenarios_per_round=6,
        n_eig_samples=24,
        convergence=conv,
        seed=200,
    )
    return GeneralizationStudyConfig(
        domain_pairs=[("game_a", "stock"), ("game_a", "supply_chain")],
        stability_domains=["game_a"],
        n_users=2,
        n_stability_sessions=2,
        elicitation=elic,
        seed=201,
        run_h4=False,
    )


def test_generalization_produces_all_pairs():
    cfg = _tiny_config()
    result = run_generalization_study(cfg)
    assert "game_a->stock" in result.per_domain_pair
    assert "game_a->supply_chain" in result.per_domain_pair


def test_per_parameter_transfer_populated():
    cfg = _tiny_config()
    result = run_generalization_study(cfg)
    for param in ["gamma", "alpha", "lambda_"]:
        assert param in result.per_parameter_transfer
        assert len(result.per_parameter_transfer[param]) >= 2


def test_hypothesis_results_keyed():
    cfg = _tiny_config()
    result = run_generalization_study(cfg)
    for h in ["H1", "H2", "H3"]:
        assert h in result.hypothesis_results
        assert len(result.hypothesis_results[h]) >= 1


def test_stability_present():
    cfg = _tiny_config()
    result = run_generalization_study(cfg)
    assert "game_a" in result.stability
    stab = result.stability["game_a"]
    assert stab.n_users == 2
    for p in ["gamma", "alpha", "lambda_"]:
        assert p in stab.icc_per_param
