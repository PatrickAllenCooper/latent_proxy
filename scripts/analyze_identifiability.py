"""Diagnose per-parameter identifiability of the diagnostic scenario libraries.

CPU-only script. For every domain, it enumerates the scenario library, samples
theta from the prior, and computes the softmax choice probability
p(theta) = sigmoid((EU_A - EU_B) / tau) along with central finite-difference
sensitivities dp/dtheta_j under common random numbers (CRN). Per-choice Fisher
information I_j = (dp/dtheta_j)^2 / (p (1 - p)) then quantifies how much a
single binary choice can tell us about each parameter.

Also probes whether the prospect-theory loss branch (wealth < reference point)
ever fires: if simulated wealth never drops below the reference point of 0,
lambda_ cannot influence any choice and I_lambda must be identically zero.

CRN is essential here: EU is estimated by ~400-draw Monte Carlo inside
SyntheticUser, so without reseeding the user RNG identically before every
evaluate_for_query call, MC noise (O(1/sqrt(400))) would swamp the O(1e-2)
finite-difference steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.generalization_protocol import DOMAIN_FACTORIES, SCENARIO_LIBRARIES
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler, UserType
from src.utils.diagnostic_scenarios import DiagnosticScenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PARAM_NAMES = ["gamma", "alpha", "lambda_"]
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "gamma": (1e-6, 1.0),
    "alpha": (0.0, np.inf),
    "lambda_": (1.0, np.inf),
}
SATURATION_LO = 0.02
SATURATION_HI = 0.98
FISHER_FLOOR = 1e-6


def _make_user(theta: NDArray[np.floating[Any]], temperature: float) -> SyntheticUser:
    """Build a SyntheticUser from a (possibly out-of-bounds) theta vector."""
    gamma = float(np.clip(theta[0], 1e-6, 1.0))
    alpha = float(max(theta[1], 0.0))
    lambda_ = float(max(theta[2], 1.0))
    return SyntheticUser(
        UserType(gamma=gamma, alpha=alpha, lambda_=lambda_),
        temperature=temperature,
    )


def _choice_prob_crn(
    theta: NDArray[np.floating[Any]],
    scenario: DiagnosticScenario,
    temperature: float,
    crn_seed: int,
) -> tuple[float, float]:
    """Return (P(choose A), EU_A - EU_B) using common random numbers.

    The user RNG is reseeded identically immediately before EVERY
    evaluate_for_query call so that the same underlying standard-normal draws
    are reused across both options and across all finite-difference stencil
    points for a given (scenario, theta).
    """
    user = _make_user(theta, temperature)

    user._rng = np.random.default_rng(crn_seed)
    eu_a = user.evaluate_for_query(
        scenario.option_a,
        scenario.channel_means,
        scenario.channel_variances,
        scenario.current_wealth,
        scenario.rounds_remaining,
        multiperiod_horizon=scenario.multiperiod_horizon,
    )
    user._rng = np.random.default_rng(crn_seed)
    eu_b = user.evaluate_for_query(
        scenario.option_b,
        scenario.channel_means,
        scenario.channel_variances,
        scenario.current_wealth,
        scenario.rounds_remaining,
        multiperiod_horizon=scenario.multiperiod_horizon,
    )

    diff = float(np.clip((eu_a - eu_b) / max(temperature, 1e-10), -500, 500))
    prob_a = 1.0 / (1.0 + np.exp(-diff))
    return float(prob_a), float(eu_a - eu_b)


def _verify_crn(scenario: DiagnosticScenario, temperature: float, crn_seed: int) -> float:
    """Verify CRN determinism: identical seeds must give bitwise-equal EU.

    Returns the max absolute EU discrepancy (must be exactly 0.0).
    """
    theta = np.array([0.6, 1.0, 1.5])
    user = _make_user(theta, temperature)

    evals: list[float] = []
    for _ in range(2):
        user._rng = np.random.default_rng(crn_seed)
        evals.append(user.evaluate_for_query(
            scenario.option_a,
            scenario.channel_means,
            scenario.channel_variances,
            scenario.current_wealth,
            scenario.rounds_remaining,
            multiperiod_horizon=scenario.multiperiod_horizon,
        ))
    max_diff = abs(evals[0] - evals[1])

    p1, d1 = _choice_prob_crn(theta, scenario, temperature, crn_seed)
    p2, d2 = _choice_prob_crn(theta, scenario, temperature, crn_seed)
    if max_diff != 0.0 or p1 != p2 or d1 != d2:
        raise RuntimeError(
            f"CRN check failed: EU diff={max_diff}, p diff={abs(p1 - p2)}. "
            "Reseeding the user RNG did not make evaluations deterministic."
        )
    return float(max_diff)


def _sensitivities(
    theta: NDArray[np.floating[Any]],
    scenario: DiagnosticScenario,
    temperature: float,
    crn_seed: int,
    rel_step: float,
) -> tuple[float, float, dict[str, float]]:
    """Central finite-difference dp/dtheta_j under CRN, respecting bounds.

    Returns (p_center, eu_diff_center, {param: dp/dtheta}).
    """
    p0, eu_diff = _choice_prob_crn(theta, scenario, temperature, crn_seed)

    grads: dict[str, float] = {}
    for j, name in enumerate(PARAM_NAMES):
        lo_bound, hi_bound = PARAM_BOUNDS[name]
        x = float(theta[j])
        h = rel_step * max(abs(x), 0.05)
        x_hi = min(x + h, hi_bound)
        x_lo = max(x - h, lo_bound)
        denom = x_hi - x_lo
        if denom <= 0.0:
            grads[name] = 0.0
            continue
        th = theta.astype(np.float64).copy()
        th[j] = x_hi
        p_hi, _ = _choice_prob_crn(th, scenario, temperature, crn_seed)
        th[j] = x_lo
        p_lo, _ = _choice_prob_crn(th, scenario, temperature, crn_seed)
        grads[name] = (p_hi - p_lo) / denom

    return p0, eu_diff, grads


def _loss_branch_fraction(scenario: DiagnosticScenario, crn_seed: int) -> float:
    """Fraction of MC wealth draws below the reference point (0.0).

    Replicates SyntheticUser.evaluate_allocation / evaluate_allocation_multiperiod
    inline (same draw shapes, same RNG seeding discipline). Wealth draws do not
    depend on theta, so this probe is a property of (scenario, option, seed).
    Returns the max fraction over the two options.
    """
    fractions: list[float] = []
    for alloc in (scenario.option_a, scenario.option_b):
        a = np.asarray(alloc, dtype=np.float64)
        port_mean = float(np.dot(a, scenario.channel_means))
        port_var = float(np.dot(a**2, scenario.channel_variances))
        port_std = max(np.sqrt(port_var), 1e-10)
        rng = np.random.default_rng(crn_seed)

        mp = scenario.multiperiod_horizon
        if mp is not None and mp > 1:
            n_samples = 256
            returns = rng.normal(port_mean, port_std, size=(n_samples, int(mp)))
            wealth = np.full(n_samples, scenario.current_wealth, dtype=np.float64)
            below = 0
            for t in range(int(mp)):
                wealth = wealth * (1.0 + returns[:, t])
                below += int(np.sum(wealth < 0.0))
            fractions.append(below / (n_samples * int(mp)))
        else:
            returns = rng.normal(port_mean, port_std, size=400)
            wealth = scenario.current_wealth * (1.0 + returns)
            fractions.append(float(np.mean(wealth < 0.0)))
    return float(max(fractions))


def _stats_block(
    fisher: dict[str, list[float]],
    probs: list[float],
    scaled_eu_diffs: list[float],
) -> dict[str, Any]:
    """Aggregate Fisher-information and choice-probability stats."""
    out: dict[str, Any] = {}
    for name in PARAM_NAMES:
        arr = np.asarray(fisher[name], dtype=np.float64)
        out[name] = {
            "median_fisher": float(np.median(arr)),
            "p10_fisher": float(np.percentile(arr, 10)),
            "p90_fisher": float(np.percentile(arr, 90)),
            "frac_fisher_above_1e-6": float(np.mean(arr > FISHER_FLOOR)),
        }
    p = np.asarray(probs, dtype=np.float64)
    d = np.asarray(scaled_eu_diffs, dtype=np.float64)
    out["median_abs_eu_diff_over_tau"] = float(np.median(np.abs(d)))
    out["saturated_fraction"] = float(np.mean((p < SATURATION_LO) | (p > SATURATION_HI)))
    out["n_pairs"] = int(len(p))
    return out


def _analyze_domain(
    domain: str,
    n_per_param: int,
    thetas: list[NDArray[np.floating[Any]]],
    temperature: float,
    rel_step: float,
    seed: int,
    n_probe_seeds: int,
) -> dict[str, Any]:
    """Run the full identifiability sweep for one domain."""
    env = DOMAIN_FACTORIES[domain]()
    library = SCENARIO_LIBRARIES[domain](seed)
    scenarios = library.generate_all(env, n_per_param)
    logger.info("[%s] %d scenarios, %d thetas", domain, len(scenarios), len(thetas))
    if not scenarios:
        return {"error": "scenario library produced no scenarios"}

    crn_diff = _verify_crn(scenarios[0], temperature, crn_seed=seed + 555)

    overall_fisher: dict[str, list[float]] = {name: [] for name in PARAM_NAMES}
    overall_probs: list[float] = []
    overall_diffs: list[float] = []
    by_target: dict[str, dict[str, Any]] = {}

    for s_idx, scenario in enumerate(scenarios):
        tp = scenario.target_param
        if tp not in by_target:
            by_target[tp] = {
                "fisher": {name: [] for name in PARAM_NAMES},
                "probs": [],
                "diffs": [],
            }
        for t_idx, theta in enumerate(thetas):
            crn_seed = seed + 100003 * s_idx + 7919 * t_idx
            p0, eu_diff, grads = _sensitivities(
                theta, scenario, temperature, crn_seed, rel_step,
            )
            scaled_diff = eu_diff / max(temperature, 1e-10)
            denom = p0 * (1.0 - p0) + 1e-12
            for name in PARAM_NAMES:
                fisher_j = grads[name] ** 2 / denom
                overall_fisher[name].append(fisher_j)
                by_target[tp]["fisher"][name].append(fisher_j)
            overall_probs.append(p0)
            overall_diffs.append(scaled_diff)
            by_target[tp]["probs"].append(p0)
            by_target[tp]["diffs"].append(scaled_diff)
        if (s_idx + 1) % 10 == 0:
            logger.info("[%s] scenario %d/%d done", domain, s_idx + 1, len(scenarios))

    probe_fractions: list[float] = []
    probe_details: list[dict[str, Any]] = []
    for s_idx, scenario in enumerate(scenarios):
        seeds = [seed + 100003 * s_idx + 7919 * t for t in range(n_probe_seeds)]
        fracs = [_loss_branch_fraction(scenario, cs) for cs in seeds]
        probe_fractions.extend(fracs)
        probe_details.append({
            "scenario_idx": s_idx,
            "target_param": scenario.target_param,
            "multiperiod_horizon": scenario.multiperiod_horizon,
            "max_frac_below_ref": float(max(fracs)),
        })

    lambda_fisher = np.asarray(overall_fisher["lambda_"], dtype=np.float64)
    result: dict[str, Any] = {
        "n_scenarios": len(scenarios),
        "n_thetas": len(thetas),
        "crn_max_abs_eu_discrepancy": crn_diff,
        "target_param_counts": {
            tp: len(block["probs"]) // len(thetas) for tp, block in by_target.items()
        },
        "overall": _stats_block(overall_fisher, overall_probs, overall_diffs),
        "by_target_param": {
            tp: _stats_block(block["fisher"], block["probs"], block["diffs"])
            for tp, block in by_target.items()
        },
        "loss_branch_probe": {
            "n_probes": len(probe_fractions),
            "max_fraction_below_ref": float(np.max(probe_fractions)),
            "mean_fraction_below_ref": float(np.mean(probe_fractions)),
            "any_draw_below_ref": bool(np.max(probe_fractions) > 0.0),
            "per_scenario_max": probe_details,
        },
        "lambda_identifiable_anywhere": bool(np.any(lambda_fisher > FISHER_FLOOR)),
        "max_lambda_fisher": float(np.max(lambda_fisher)),
    }
    return result


def _format_stats_row(name: str, block: dict[str, Any]) -> str:
    s = block[name]
    return (
        f"| {name} | {s['median_fisher']:.3e} | {s['p10_fisher']:.3e} | "
        f"{s['p90_fisher']:.3e} | {s['frac_fisher_above_1e-6']:.3f} |"
    )


def _write_markdown(results: dict[str, Any], path: Path) -> None:
    lines: list[str] = ["# Identifiability diagnostics", ""]
    lines.append(
        "Per-choice Fisher information I_j = (dp/dtheta_j)^2 / (p(1-p)) computed with "
        "common random numbers over (scenario, theta) pairs."
    )
    lines.append("")
    for domain, res in results["domains"].items():
        lines.append(f"## {domain}")
        if "error" in res:
            lines.append(f"ERROR: {res['error']}")
            lines.append("")
            continue
        ov = res["overall"]
        lines.append(
            f"{res['n_scenarios']} scenarios x {res['n_thetas']} thetas "
            f"({ov['n_pairs']} pairs). "
            f"Median |EU_A-EU_B|/tau = {ov['median_abs_eu_diff_over_tau']:.3f}; "
            f"saturated (p<{SATURATION_LO} or p>{SATURATION_HI}): "
            f"{ov['saturated_fraction']:.3f}."
        )
        lines.append("")
        lines.append("| param | median I | p10 I | p90 I | frac I > 1e-6 |")
        lines.append("|---|---|---|---|---|")
        for name in PARAM_NAMES:
            lines.append(_format_stats_row(name, ov))
        lines.append("")
        lines.append("### By target_param subgroup")
        lines.append("")
        for tp, block in res["by_target_param"].items():
            lines.append(
                f"**{tp}-targeted** ({block['n_pairs']} pairs, "
                f"median |dEU|/tau = {block['median_abs_eu_diff_over_tau']:.3f}, "
                f"saturated = {block['saturated_fraction']:.3f})"
            )
            lines.append("")
            lines.append("| param | median I | p10 I | p90 I | frac I > 1e-6 |")
            lines.append("|---|---|---|---|---|")
            for name in PARAM_NAMES:
                lines.append(_format_stats_row(name, block))
            lines.append("")
        probe = res["loss_branch_probe"]
        lines.append(
            f"**Loss-branch probe**: max fraction of MC wealth draws below the "
            f"reference point (0.0) across {probe['n_probes']} probes = "
            f"{probe['max_fraction_below_ref']:.5f} "
            f"(any draw below ref: {probe['any_draw_below_ref']})."
        )
        lines.append(
            f"**Lambda identifiable anywhere** (I_lambda > 1e-6): "
            f"{res['lambda_identifiable_anywhere']} "
            f"(max I_lambda = {res['max_lambda_fisher']:.3e})."
        )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-parameter identifiability diagnostics for scenario libraries",
    )
    parser.add_argument(
        "--domains", nargs="+", default=list(DOMAIN_FACTORIES.keys()),
        choices=list(DOMAIN_FACTORIES.keys()),
    )
    parser.add_argument("--n-thetas", type=int, default=200,
                        help="Thetas sampled from the prior per domain")
    parser.add_argument("--n-per-param", type=int, default=17,
                        help="Scenarios per target parameter (~3x this total)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--rel-step", type=float, default=1e-2,
                        help="Relative finite-difference step")
    parser.add_argument("--n-probe-seeds", type=int, default=3,
                        help="CRN seeds per scenario for the loss-branch probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/diagnostics"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sampler = SyntheticUserSampler(seed=args.seed + 17)
    thetas = [
        np.array([ut.gamma, ut.alpha, ut.lambda_], dtype=np.float64)
        for ut in sampler.sample_batch(args.n_thetas)
    ]

    results: dict[str, Any] = {
        "config": {
            "domains": args.domains,
            "n_thetas": args.n_thetas,
            "n_per_param": args.n_per_param,
            "temperature": args.temperature,
            "rel_step": args.rel_step,
            "n_probe_seeds": args.n_probe_seeds,
            "seed": args.seed,
        },
        "domains": {},
    }

    for domain in args.domains:
        logger.info("=== Domain: %s ===", domain)
        results["domains"][domain] = _analyze_domain(
            domain, args.n_per_param, thetas, args.temperature,
            args.rel_step, args.seed, args.n_probe_seeds,
        )

    json_path = args.output_dir / "identifiability.json"
    md_path = args.output_dir / "identifiability.md"
    json_path.write_text(json.dumps(results, indent=2))
    _write_markdown(results, md_path)
    logger.info("Wrote %s and %s", json_path, md_path)

    for domain, res in results["domains"].items():
        if "error" in res:
            continue
        logger.info(
            "[%s] lambda identifiable anywhere: %s (max I_lambda=%.3e); "
            "loss branch ever fires: %s",
            domain,
            res["lambda_identifiable_anywhere"],
            res["max_lambda_fisher"],
            res["loss_branch_probe"]["any_draw_below_ref"],
        )


if __name__ == "__main__":
    main()
