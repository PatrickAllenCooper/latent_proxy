from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop, ElicitationResult
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.game_variants import create_variant_a, create_variant_b
from src.environments.resource_game import ResourceStrategyGame
from src.evaluation.alignment_metrics import (
    compute_alignment_score,
    compute_quality_floor_violation_rate,
    evaluate_actions,
)
from src.evaluation.elicitation_metrics import run_elicitation_benchmark
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler

logger = logging.getLogger(__name__)

TransferCondition = Literal["generic", "within_domain", "cross_domain"]


def _normal_ci(values: list[float], z: float = 1.96) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    m = float(np.mean(arr))
    if arr.size < 2:
        return m, m, m
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return m, m - z * se, m + z * se


def _theta_dict(ut: Any) -> dict[str, float]:
    return {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}


def _make_env(variant: str) -> ResourceStrategyGame:
    v = variant.lower()
    if v in ("b", "variant_b"):
        return create_variant_b()
    return create_variant_a()


@dataclass
class ExperimentConfig:
    """Orchestration config for Milestone 4 evaluation."""

    variants: list[str] = field(default_factory=lambda: ["a", "b"])
    n_users: int = 20
    seed: int = 42
    conditions: list[TransferCondition] = field(
        default_factory=lambda: ["generic", "within_domain", "cross_domain"],
    )
    elicitation: ElicitationConfig = field(default_factory=ElicitationConfig)
    ablation_params: dict[str, list[Any]] = field(default_factory=dict)


@dataclass
class TargetCheckResult:
    """README Section 7.1 targets vs observed aggregates."""

    alignment_above_threshold: bool
    quality_floor_clean: bool
    gamma_mae_below_threshold: bool
    elicitation_efficiency_met: bool
    details: dict[str, Any]


def check_targets(
    mean_alignment: float,
    violation_rate: float,
    mean_gamma_mae: float,
    error_reduction_pct: float,
    *,
    alignment_threshold: float = 0.8,
    gamma_mae_threshold: float = 0.1,
    efficiency_threshold_pct: float = 30.0,
) -> TargetCheckResult:
    return TargetCheckResult(
        alignment_above_threshold=mean_alignment >= alignment_threshold,
        quality_floor_clean=violation_rate <= 0.0,
        gamma_mae_below_threshold=mean_gamma_mae < gamma_mae_threshold,
        elicitation_efficiency_met=error_reduction_pct >= efficiency_threshold_pct,
        details={
            "mean_alignment": mean_alignment,
            "violation_rate": violation_rate,
            "mean_gamma_mae": mean_gamma_mae,
            "error_reduction_pct": error_reduction_pct,
            "alignment_threshold": alignment_threshold,
            "gamma_mae_threshold": gamma_mae_threshold,
            "efficiency_threshold_pct": efficiency_threshold_pct,
        },
    )


def _alignment_inferred_vs_true(
    env: ResourceStrategyGame,
    inferred: dict[str, float],
    true_theta: Any,
    seed: int,
) -> float:
    env.reset(seed=seed)
    opt_true = env.get_optimal_action(_theta_dict(true_theta))
    opt_hat = env.get_optimal_action(inferred)
    return compute_alignment_score([opt_hat], [opt_true])


def run_full_evaluation(
    config: ExperimentConfig | None = None,
) -> dict[str, Any]:
    """Per-variant elicitation benchmark plus README 7.1-style aggregates."""
    config = config or ExperimentConfig()
    out: dict[str, Any] = {"variants": {}, "config": asdict(config)}

    for variant in config.variants:
        env = _make_env(variant)
        bench = run_elicitation_benchmark(
            env,
            n_users=config.n_users,
            config=config.elicitation,
            seed=config.seed,
        )
        eff = bench["efficiency"]
        active_results: list[ElicitationResult] = bench["active_results"]

        gamma_maes: list[float] = []
        total_errs: list[float] = []
        alignments: list[float] = []
        rec_actions: list[NDArray[np.floating[Any]]] = []
        true_types: list[Any] = []

        for i, r in enumerate(active_results):
            err = r.preference_recovery_error()
            if err:
                gamma_maes.append(err["gamma"])
                total_errs.append(err["total"])
            if r.true_theta is not None:
                alignments.append(
                    _alignment_inferred_vs_true(
                        env, r.inferred_theta, r.true_theta,
                        seed=config.seed + i * 17 + 1,
                    )
                )
                rec_actions.append(env.get_optimal_action(r.inferred_theta))
                true_types.append(r.true_theta)

        violation_rate = 0.0
        if rec_actions:
            violation_rate = compute_quality_floor_violation_rate(rec_actions, env)

        mean_quality = 0.0
        if rec_actions and true_types:
            q_eval = evaluate_actions(rec_actions, env, true_types)
            mean_quality = float(q_eval["mean_quality_score"])

        mean_gamma_mae = float(np.mean(gamma_maes)) if gamma_maes else 0.0
        mean_align = float(np.mean(alignments)) if alignments else 0.0

        targets = check_targets(
            mean_align,
            violation_rate,
            mean_gamma_mae,
            float(eff["error_reduction_pct"]),
        )
        targets_dict = {
            "alignment_above_threshold": targets.alignment_above_threshold,
            "quality_floor_clean": targets.quality_floor_clean,
            "gamma_mae_below_threshold": targets.gamma_mae_below_threshold,
            "elicitation_efficiency_met": targets.elicitation_efficiency_met,
            "details": targets.details,
        }

        out["variants"][variant] = {
            "efficiency": eff,
            "mean_alignment_active": mean_align,
            "mean_gamma_mae": mean_gamma_mae,
            "mean_total_recovery_error": float(np.mean(total_errs)) if total_errs else 0.0,
            "quality_floor_violation_rate": violation_rate,
            "mean_quality_score": mean_quality,
            "targets": targets_dict,
            "n_users": config.n_users,
        }
        logger.info(
            "Variant %s: align=%.3f gamma_mae=%.3f err_reduct=%.1f%% targets_met=%s",
            variant,
            mean_align,
            mean_gamma_mae,
            eff["error_reduction_pct"],
            all(
                [
                    targets.alignment_above_threshold,
                    targets.quality_floor_clean,
                    targets.gamma_mae_below_threshold,
                    targets.elicitation_efficiency_met,
                ],
            ),
        )

    return out


@dataclass
class TransferExperimentResult:
    """Three-condition transfer protocol (README Section 8.1)."""

    generic: dict[str, float]
    within_domain: dict[str, float]
    cross_domain: dict[str, float]
    per_user: dict[str, list[float]]
    n_users: int


def run_transfer_experiment(
    n_users: int = 15,
    elicitation: ElicitationConfig | None = None,
    seed: int = 42,
) -> TransferExperimentResult:
    """Generic / Within-Domain / Cross-Domain on variant A -> B."""
    elicitation = elicitation or ElicitationConfig()
    convergence = elicitation.convergence
    if convergence.max_rounds < elicitation.max_rounds:
        convergence = ConvergenceConfig(
            gamma_variance_threshold=convergence.gamma_variance_threshold,
            alpha_variance_threshold=convergence.alpha_variance_threshold,
            lambda_variance_threshold=convergence.lambda_variance_threshold,
            robust_action_level=convergence.robust_action_level,
            max_rounds=elicitation.max_rounds,
        )
        elicitation = ElicitationConfig(
            posterior_type=elicitation.posterior_type,
            n_particles=elicitation.n_particles,
            max_rounds=elicitation.max_rounds,
            n_scenarios_per_round=elicitation.n_scenarios_per_round,
            n_eig_samples=elicitation.n_eig_samples,
            temperature=elicitation.temperature,
            convergence=convergence,
            seed=elicitation.seed,
        )

    env_a = create_variant_a()
    env_b = create_variant_b()
    sampler = SyntheticUserSampler(seed=seed)

    g_scores: list[float] = []
    w_scores: list[float] = []
    c_scores: list[float] = []

    for i in range(n_users):
        ut = sampler.sample()
        theta_d = _theta_dict(ut)

        env_b.reset(seed=seed + i)
        opt_b = env_b.get_optimal_action(theta_d)
        kb = env_b.config.n_channels
        uniform = np.ones(kb, dtype=np.float64) / kb
        g_scores.append(compute_alignment_score([uniform], [opt_b]))

        user_w = SyntheticUser(ut, temperature=elicitation.temperature, seed=seed + i + 333)
        cfg_w = ElicitationConfig(
            posterior_type=elicitation.posterior_type,
            n_particles=elicitation.n_particles,
            max_rounds=elicitation.max_rounds,
            n_scenarios_per_round=elicitation.n_scenarios_per_round,
            n_eig_samples=elicitation.n_eig_samples,
            temperature=elicitation.temperature,
            convergence=elicitation.convergence,
            seed=seed + i * 1000 + 7,
        )
        loop_b = ElicitationLoop(cfg_w)
        env_b.reset(seed=seed + i + 9000)
        res_b = loop_b.run(env_b, user_w, query_type="active")
        opt_hat_b = env_b.get_optimal_action(res_b.inferred_theta)
        w_scores.append(compute_alignment_score([opt_hat_b], [opt_b]))

        user_c = SyntheticUser(ut, temperature=elicitation.temperature, seed=seed + i + 777)
        cfg_c = ElicitationConfig(
            posterior_type=elicitation.posterior_type,
            n_particles=elicitation.n_particles,
            max_rounds=elicitation.max_rounds,
            n_scenarios_per_round=elicitation.n_scenarios_per_round,
            n_eig_samples=elicitation.n_eig_samples,
            temperature=elicitation.temperature,
            convergence=elicitation.convergence,
            seed=seed + i * 1000 + 99,
        )
        loop_a = ElicitationLoop(cfg_c)
        env_a.reset(seed=seed + i + 8000)
        res_a = loop_a.run(env_a, user_c, query_type="active")
        opt_hat_cross = env_b.get_optimal_action(res_a.inferred_theta)
        c_scores.append(compute_alignment_score([opt_hat_cross], [opt_b]))

    def pack(xs: list[float]) -> dict[str, float]:
        m, lo, hi = _normal_ci(xs)
        return {"mean": m, "ci_low": lo, "ci_high": hi}

    return TransferExperimentResult(
        generic=pack(g_scores),
        within_domain=pack(w_scores),
        cross_domain=pack(c_scores),
        per_user={
            "generic": g_scores,
            "within_domain": w_scores,
            "cross_domain": c_scores,
        },
        n_users=n_users,
    )


def prior_mean_theta() -> dict[str, float]:
    """Central prior defaults for blending (ablation / proxy for curriculum beta)."""
    return {"gamma": 0.5, "alpha": 1.0, "lambda_": 2.0}


def blend_inferred_theta(
    inferred: dict[str, float],
    beta: float,
    prior_mean: dict[str, float] | None = None,
) -> dict[str, float]:
    """Interpolate between prior mean and inferred theta (beta = alignment emphasis)."""
    prior_mean = prior_mean or prior_mean_theta()
    b = float(np.clip(beta, 0.0, 1.0))
    out: dict[str, float] = {}
    for k in prior_mean:
        out[k] = (1.0 - b) * prior_mean[k] + b * float(inferred.get(k, prior_mean[k]))
    out["gamma"] = float(np.clip(out["gamma"], 1e-6, 1.0))
    out["alpha"] = float(max(out["alpha"], 0.0))
    out["lambda_"] = float(max(out["lambda_"], 1.0))
    return out
