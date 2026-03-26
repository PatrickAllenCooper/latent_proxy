"""Unified generalization study protocol (README Section 8, Milestone 6).

Runs all domain-pair transfer experiments, per-parameter decomposition,
theta stability, and H1-H4 statistical tests in one coordinated pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.base import BaseEnvironment
from src.environments.game_variants import create_variant_a, create_variant_b
from src.environments.stock_backtest import StockBacktestConfig, StockBacktestEnv
from src.environments.supply_chain import SupplyChainConfig, SupplyChainEnv
from src.evaluation.alignment_metrics import (
    compute_alignment_score,
    compute_preference_recovery_error,
)
from src.evaluation.elicitation_metrics import run_elicitation_benchmark
from src.evaluation.experiment_runner import TransferExperimentResult, _normal_ci
from src.evaluation.statistical_analysis import (
    HypothesisTestResult,
    holm_bonferroni,
    run_test_h1_within_domain,
    run_test_h2_cross_domain,
    run_test_h3_parameter_transfer,
    run_test_h4_active_vs_random,
)
from src.evaluation.theta_stability import ThetaStabilityResult, run_theta_stability_test
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler
from src.utils.diagnostic_scenarios import ScenarioLibrary
from src.utils.stock_scenarios import StockScenarioLibrary
from src.utils.supply_chain_scenarios import SupplyChainScenarioLibrary

logger = logging.getLogger(__name__)

DOMAIN_FACTORIES: dict[str, Callable[[], BaseEnvironment]] = {
    "game_a": create_variant_a,
    "game_b": create_variant_b,
    "stock": lambda: StockBacktestEnv(config=StockBacktestConfig()),
    "supply_chain": lambda: SupplyChainEnv(config=SupplyChainConfig()),
}

SCENARIO_LIBRARIES: dict[str, Callable[[int], Any]] = {
    "game_a": lambda seed: ScenarioLibrary(seed=seed),
    "game_b": lambda seed: ScenarioLibrary(seed=seed),
    "stock": lambda seed: StockScenarioLibrary(seed=seed),
    "supply_chain": lambda seed: SupplyChainScenarioLibrary(seed=seed),
}

STRUCTURAL_SIMILARITY: dict[tuple[str, str], float] = {
    ("game_a", "game_b"): 0.95,
    ("game_a", "stock"): 0.55,
    ("game_a", "supply_chain"): 0.40,
    ("stock", "supply_chain"): 0.50,
}


@dataclass
class GeneralizationStudyConfig:
    domain_pairs: list[tuple[str, str]] = field(default_factory=lambda: [
        ("game_a", "game_b"),
        ("game_a", "stock"),
        ("game_a", "supply_chain"),
        ("stock", "supply_chain"),
    ])
    stability_domains: list[str] = field(default_factory=lambda: [
        "game_a", "stock", "supply_chain",
    ])
    n_users: int = 15
    n_stability_sessions: int = 2
    elicitation: ElicitationConfig = field(default_factory=ElicitationConfig)
    seed: int = 42
    run_h4: bool = True


@dataclass
class GeneralizationStudyResult:
    per_domain_pair: dict[str, TransferExperimentResult]
    per_parameter_transfer: dict[str, dict[str, float]]
    stability: dict[str, ThetaStabilityResult]
    hypothesis_results: dict[str, list[HypothesisTestResult]]
    h4_details: dict[str, Any] | None = None


def _theta_dict(ut: Any) -> dict[str, float]:
    return {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}


def _run_pair_transfer(
    source_name: str,
    target_name: str,
    n_users: int,
    elicitation: ElicitationConfig,
    seed: int,
) -> tuple[TransferExperimentResult, dict[str, float]]:
    """Run generic / within-target / cross (source->target) and collect per-param MAE."""
    source_factory = DOMAIN_FACTORIES[source_name]
    target_factory = DOMAIN_FACTORIES[target_name]
    sampler = SyntheticUserSampler(seed=seed)

    g_scores: list[float] = []
    w_scores: list[float] = []
    c_scores: list[float] = []
    param_errors: dict[str, list[float]] = {"gamma": [], "alpha": [], "lambda_": []}

    convergence = elicitation.convergence
    if convergence.max_rounds < elicitation.max_rounds:
        convergence = ConvergenceConfig(
            gamma_variance_threshold=convergence.gamma_variance_threshold,
            alpha_variance_threshold=convergence.alpha_variance_threshold,
            lambda_variance_threshold=convergence.lambda_variance_threshold,
            robust_action_level=convergence.robust_action_level,
            max_rounds=elicitation.max_rounds,
        )

    for i in range(n_users):
        ut = sampler.sample()
        theta_d = _theta_dict(ut)

        target_env = target_factory()
        target_env.reset(seed=seed + i)
        opt_target = target_env.get_optimal_action(theta_d)
        kt = target_env.config.n_channels
        uniform = np.ones(kt, dtype=np.float64) / kt
        g_scores.append(compute_alignment_score([uniform], [opt_target]))

        lib_target = SCENARIO_LIBRARIES[target_name](seed + i + 4000)
        user_w = SyntheticUser(ut, temperature=elicitation.temperature, seed=seed + i + 333)
        cfg_w = ElicitationConfig(
            posterior_type=elicitation.posterior_type,
            n_particles=elicitation.n_particles,
            max_rounds=elicitation.max_rounds,
            n_scenarios_per_round=elicitation.n_scenarios_per_round,
            n_eig_samples=elicitation.n_eig_samples,
            temperature=elicitation.temperature,
            convergence=convergence,
            seed=seed + i * 1000 + 7,
            scenario_library=lib_target,
        )
        loop_w = ElicitationLoop(cfg_w)
        target_env.reset(seed=seed + i + 9000)
        res_w = loop_w.run(target_env, user_w, query_type="active")
        opt_hat_w = target_env.get_optimal_action(res_w.inferred_theta)
        w_scores.append(compute_alignment_score([opt_hat_w], [opt_target]))

        lib_source = SCENARIO_LIBRARIES[source_name](seed + i + 5000)
        user_c = SyntheticUser(ut, temperature=elicitation.temperature, seed=seed + i + 777)
        cfg_c = ElicitationConfig(
            posterior_type=elicitation.posterior_type,
            n_particles=elicitation.n_particles,
            max_rounds=elicitation.max_rounds,
            n_scenarios_per_round=elicitation.n_scenarios_per_round,
            n_eig_samples=elicitation.n_eig_samples,
            temperature=elicitation.temperature,
            convergence=convergence,
            seed=seed + i * 1000 + 99,
            scenario_library=lib_source,
        )
        source_env = source_factory()
        loop_c = ElicitationLoop(cfg_c)
        source_env.reset(seed=seed + i + 8000)
        res_c = loop_c.run(source_env, user_c, query_type="active")
        opt_hat_c = target_env.get_optimal_action(res_c.inferred_theta)
        c_scores.append(compute_alignment_score([opt_hat_c], [opt_target]))

        recovery = compute_preference_recovery_error(res_c.inferred_theta, ut)
        for p_name in param_errors:
            param_errors[p_name].append(recovery.get(p_name, 0.0))

    def pack(xs: list[float]) -> dict[str, float]:
        m, lo, hi = _normal_ci(xs)
        return {"mean": m, "ci_low": lo, "ci_high": hi}

    transfer = TransferExperimentResult(
        generic=pack(g_scores),
        within_domain=pack(w_scores),
        cross_domain=pack(c_scores),
        per_user={"generic": g_scores, "within_domain": w_scores, "cross_domain": c_scores},
        n_users=n_users,
    )
    mean_param_mae = {p: float(np.mean(v)) for p, v in param_errors.items()}
    return transfer, mean_param_mae


def run_generalization_study(
    config: GeneralizationStudyConfig | None = None,
) -> GeneralizationStudyResult:
    config = config or GeneralizationStudyConfig()

    per_domain_pair: dict[str, TransferExperimentResult] = {}
    per_param_by_pair: dict[tuple[str, str], dict[str, float]] = {}

    for src, tgt in config.domain_pairs:
        key = f"{src}->{tgt}"
        logger.info("Running pair %s (n_users=%d)", key, config.n_users)
        tr, param_mae = _run_pair_transfer(
            src, tgt, config.n_users, config.elicitation,
            seed=config.seed + hash(key) % 10000,
        )
        per_domain_pair[key] = tr
        per_param_by_pair[(src, tgt)] = param_mae

    per_parameter_transfer: dict[str, dict[str, float]] = {}
    for param in ["gamma", "alpha", "lambda_"]:
        per_parameter_transfer[param] = {
            f"{s}->{t}": per_param_by_pair[(s, t)][param]
            for s, t in per_param_by_pair
        }

    stability: dict[str, ThetaStabilityResult] = {}
    if config.n_stability_sessions >= 2:
        for domain in config.stability_domains:
            if domain not in DOMAIN_FACTORIES:
                continue
            factory = DOMAIN_FACTORIES[domain]
            lib_factory = SCENARIO_LIBRARIES.get(domain)
            elic = config.elicitation
            if lib_factory is not None:
                lib = lib_factory(config.seed + 7777)
                elic = ElicitationConfig(
                    posterior_type=elic.posterior_type,
                    n_particles=elic.n_particles,
                    max_rounds=elic.max_rounds,
                    n_scenarios_per_round=elic.n_scenarios_per_round,
                    n_eig_samples=elic.n_eig_samples,
                    temperature=elic.temperature,
                    convergence=elic.convergence,
                    seed=elic.seed,
                    scenario_library=lib,
                )
            stab = run_theta_stability_test(
                factory,
                n_users=config.n_users,
                n_sessions=config.n_stability_sessions,
                elicitation=elic,
                seed=config.seed + 3000,
            )
            stability[domain] = stab
            logger.info("Stability %s: ICC=%s", domain, stab.icc_per_param)

    hypothesis_results: dict[str, list[HypothesisTestResult]] = {
        "H1": [], "H2": [], "H3": [], "H4": [],
    }

    for key, tr in per_domain_pair.items():
        h1 = run_test_h1_within_domain(
            tr.per_user["within_domain"],
            tr.per_user["generic"],
            hypothesis_label=f"H1_{key}",
        )
        hypothesis_results["H1"].append(h1)

        h2_list = run_test_h2_cross_domain(
            tr.per_user["cross_domain"],
            tr.per_user["generic"],
            tr.per_user["within_domain"],
            hypothesis_label=f"H2_{key}",
        )
        hypothesis_results["H2"].extend(h2_list)

    h3_results = run_test_h3_parameter_transfer(
        per_param_by_pair,
        structural_similarity=STRUCTURAL_SIMILARITY,
        hypothesis_label="H3",
    )
    hypothesis_results["H3"] = list(h3_results.values())

    h4_details: dict[str, Any] | None = None
    if config.run_h4:
        h4_details = {}
        for domain in config.stability_domains:
            if domain not in DOMAIN_FACTORIES:
                continue
            factory = DOMAIN_FACTORIES[domain]
            lib_factory = SCENARIO_LIBRARIES.get(domain)
            elic = config.elicitation
            if lib_factory is not None:
                lib = lib_factory(config.seed + 8888)
                elic = ElicitationConfig(
                    posterior_type=elic.posterior_type,
                    n_particles=elic.n_particles,
                    max_rounds=elic.max_rounds,
                    n_scenarios_per_round=elic.n_scenarios_per_round,
                    n_eig_samples=elic.n_eig_samples,
                    temperature=elic.temperature,
                    convergence=elic.convergence,
                    seed=elic.seed,
                    scenario_library=lib,
                )
            env = factory()
            bench = run_elicitation_benchmark(
                env,
                n_users=min(config.n_users, 5),
                config=elic,
                seed=config.seed + 6000,
            )
            active_errs = [
                (r.preference_recovery_error() or {}).get("total", 0.0)
                for r in bench["active_results"]
            ]
            random_errs = [
                (r.preference_recovery_error() or {}).get("total", 0.0)
                for r in bench["random_results"]
            ]
            h4 = run_test_h4_active_vs_random(
                active_errs, random_errs, hypothesis_label=f"H4_{domain}",
            )
            hypothesis_results["H4"].append(h4)
            h4_details[domain] = bench["efficiency"]

    all_pvals: list[float] = []
    all_results_flat: list[HypothesisTestResult] = []
    for group in hypothesis_results.values():
        for r in group:
            all_pvals.append(r.p_value)
            all_results_flat.append(r)

    if all_pvals:
        adjusted = holm_bonferroni(all_pvals)
        for r, adj_p in zip(all_results_flat, adjusted):
            r.p_value = float(adj_p)
            r.conclusion = "reject_null" if adj_p < 0.05 else "fail_to_reject"

    return GeneralizationStudyResult(
        per_domain_pair=per_domain_pair,
        per_parameter_transfer=per_parameter_transfer,
        stability=stability,
        hypothesis_results=hypothesis_results,
        h4_details=h4_details,
    )
