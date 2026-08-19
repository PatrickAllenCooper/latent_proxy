"""Diagnose Monte Carlo stability of EIG-based query selection.

CPU-only script. For each domain it runs a short elicitation (PreferenceTracker
+ StructuredQueryGenerator, mirroring ElicitationLoop.run) to obtain a
realistic mid-session particle posterior, freezes that posterior, enumerates
the candidate scenario pool exactly as StructuredQueryGenerator would, then:

1. Computes a "gold" EIG ranking with a very large sample budget (default
   20000 samples, one fixed RNG).
2. For each smaller sample budget, runs R repetitions with different RNGs and
   reports P(argmax == gold argmax), P(argmax in gold top-5), and the mean
   Kendall tau of the repetition ranking vs the gold ranking.
3. Reports the gold pool spread (max EIG - median EIG) / (max EIG + 1e-12)
   and the top-5 gold EIG values.

Interpretation: if the argmax is unstable at the production sample budget,
EIG-based selection is effectively random (which would explain a missing
active-learning gain over the random baseline). If the gold pool spread is
~0, the candidate pool itself is the ceiling: no selector can help.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.preference_tracker import ConvergenceConfig, PreferenceTracker
from src.agents.query_generator import StructuredQueryGenerator
from src.evaluation.generalization_protocol import DOMAIN_FACTORIES, SCENARIO_LIBRARIES
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler
from src.utils.information_gain import compute_eig_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class JobSpec:
    """One (domain, user) stability job; must stay picklable for workers."""

    domain: str
    user_idx: int
    seed: int
    n_particles: int
    elicitation_rounds: int
    elicit_eig_samples: int
    n_scenarios_per_round: int
    n_per_param: int
    gold_samples: int
    sample_grid: list[int]
    n_reps: int
    temperature: float


def _run_elicitation_session(
    spec: JobSpec,
) -> tuple[PreferenceTracker, Any, Any, dict[str, float]]:
    """Run a short active elicitation and return the (unfrozen) tracker.

    Mirrors ElicitationLoop.run but keeps the tracker so the posterior can be
    frozen and reused for the stability sweep. Convergence thresholds are set
    to 0 so the session always runs the full number of rounds.
    """
    env = DOMAIN_FACTORIES[spec.domain]()
    library = SCENARIO_LIBRARIES[spec.domain](spec.seed + 100 * spec.user_idx)

    sampler = SyntheticUserSampler(seed=spec.seed + spec.user_idx)
    user_type = sampler.sample()
    user = SyntheticUser(
        user_type,
        temperature=spec.temperature,
        seed=spec.seed + 999 + spec.user_idx,
    )

    tracker = PreferenceTracker(
        posterior_type="particle",
        n_particles=spec.n_particles,
        temperature=spec.temperature,
        convergence=ConvergenceConfig(
            gamma_variance_threshold=0.0,
            alpha_variance_threshold=0.0,
            lambda_variance_threshold=0.0,
            max_rounds=spec.elicitation_rounds,
        ),
    )
    query_gen = StructuredQueryGenerator(
        n_scenarios_per_round=spec.n_scenarios_per_round,
        n_eig_samples=spec.elicit_eig_samples,
        temperature=spec.temperature,
        seed=spec.seed + spec.user_idx,
        library=library,
    )

    env.reset(seed=spec.seed + spec.user_idx)
    for round_idx in range(spec.elicitation_rounds):
        scenario = query_gen.select_query(env, tracker.posterior)
        eu_a = user.evaluate_for_query(
            scenario.option_a,
            scenario.channel_means,
            scenario.channel_variances,
            scenario.current_wealth,
            scenario.rounds_remaining,
            multiperiod_horizon=scenario.multiperiod_horizon,
        )
        eu_b = user.evaluate_for_query(
            scenario.option_b,
            scenario.channel_means,
            scenario.channel_variances,
            scenario.current_wealth,
            scenario.rounds_remaining,
            multiperiod_horizon=scenario.multiperiod_horizon,
        )
        choice = user.choose(eu_a, eu_b)
        tracker.observe(choice, scenario)
        logger.debug(
            "[%s u%d] round %d: choice=%d mean=%s",
            spec.domain, spec.user_idx, round_idx + 1, choice,
            tracker.posterior.to_dict(),
        )

    true_theta = {
        "gamma": user_type.gamma,
        "alpha": user_type.alpha,
        "lambda_": user_type.lambda_,
    }
    return tracker, env, library, true_theta


def run_domain_user_job(spec: JobSpec) -> dict[str, Any]:
    """Full stability analysis for one (domain, user): elicit, freeze, sweep."""
    logger.info("[%s u%d] running %d-round elicitation (n_particles=%d)",
                spec.domain, spec.user_idx, spec.elicitation_rounds, spec.n_particles)
    tracker, env, library, true_theta = _run_elicitation_session(spec)
    posterior = tracker.posterior  # frozen: only read from here on

    pool = library.generate_all(env, spec.n_per_param)
    if not pool:
        return {"domain": spec.domain, "user_idx": spec.user_idx,
                "error": "empty candidate pool"}
    logger.info("[%s u%d] candidate pool: %d scenarios; computing gold EIG "
                "(n_samples=%d)", spec.domain, spec.user_idx, len(pool),
                spec.gold_samples)

    gold_rng = np.random.default_rng([spec.seed, spec.user_idx, 31337])
    gold_eig = compute_eig_batch(
        pool, posterior, n_samples=spec.gold_samples,
        temperature=spec.temperature, rng=gold_rng,
    )
    gold_argmax = int(np.argmax(gold_eig))
    gold_order = np.argsort(gold_eig)[::-1]
    gold_top5_idx = [int(i) for i in gold_order[:5]]
    gold_top5_values = [float(gold_eig[i]) for i in gold_top5_idx]
    gold_max = float(np.max(gold_eig))
    gold_median = float(np.median(gold_eig))
    gold_spread = float((gold_max - gold_median) / (gold_max + 1e-12))

    per_budget: dict[str, Any] = {}
    for n_s in spec.sample_grid:
        argmax_hits: list[float] = []
        top5_hits: list[float] = []
        taus: list[float] = []
        for rep in range(spec.n_reps):
            rng = np.random.default_rng([spec.seed, spec.user_idx, n_s, rep])
            eig = compute_eig_batch(
                pool, posterior, n_samples=n_s,
                temperature=spec.temperature, rng=rng,
            )
            am = int(np.argmax(eig))
            argmax_hits.append(1.0 if am == gold_argmax else 0.0)
            top5_hits.append(1.0 if am in gold_top5_idx else 0.0)
            tau, _ = kendalltau(eig, gold_eig)
            taus.append(float(tau) if np.isfinite(tau) else np.nan)
        tau_arr = np.asarray(taus, dtype=np.float64)
        n_valid = int(np.sum(np.isfinite(tau_arr)))
        per_budget[str(n_s)] = {
            "p_argmax_match": float(np.mean(argmax_hits)),
            "p_argmax_in_gold_top5": float(np.mean(top5_hits)),
            "mean_kendall_tau": (
                float(np.nanmean(tau_arr)) if n_valid > 0 else None
            ),
            "std_kendall_tau": (
                float(np.nanstd(tau_arr)) if n_valid > 0 else None
            ),
            "n_reps": spec.n_reps,
            "n_undefined_tau": spec.n_reps - n_valid,
        }
        logger.info("[%s u%d] n_samples=%d: P(argmax=gold)=%.2f, mean tau=%s",
                    spec.domain, spec.user_idx, n_s,
                    per_budget[str(n_s)]["p_argmax_match"],
                    per_budget[str(n_s)]["mean_kendall_tau"])

    posterior_summary = {
        "mean": posterior.to_dict(),
        "variance": {
            name: float(posterior.variance[i])
            for i, name in enumerate(posterior.param_names)
        },
    }
    return {
        "domain": spec.domain,
        "user_idx": spec.user_idx,
        "true_theta": true_theta,
        "posterior_after_elicitation": posterior_summary,
        "n_candidates": len(pool),
        "gold": {
            "n_samples": spec.gold_samples,
            "argmax_idx": gold_argmax,
            "argmax_target_param": pool[gold_argmax].target_param,
            "top5_idx": gold_top5_idx,
            "top5_values": gold_top5_values,
            "max": gold_max,
            "median": gold_median,
            "pool_spread": gold_spread,
        },
        "per_budget": per_budget,
    }


def _aggregate_domain(user_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Average per-budget metrics over users within a domain."""
    valid = [r for r in user_results if "error" not in r]
    if not valid:
        return {"error": "no valid user results"}
    budgets = list(valid[0]["per_budget"].keys())
    agg: dict[str, Any] = {}
    for b in budgets:
        p_arg = [r["per_budget"][b]["p_argmax_match"] for r in valid]
        p_top5 = [r["per_budget"][b]["p_argmax_in_gold_top5"] for r in valid]
        taus = [
            r["per_budget"][b]["mean_kendall_tau"] for r in valid
            if r["per_budget"][b]["mean_kendall_tau"] is not None
        ]
        agg[b] = {
            "p_argmax_match": float(np.mean(p_arg)),
            "p_argmax_in_gold_top5": float(np.mean(p_top5)),
            "mean_kendall_tau": float(np.mean(taus)) if taus else None,
        }
    return {
        "n_users": len(valid),
        "gold_pool_spreads": [r["gold"]["pool_spread"] for r in valid],
        "per_budget": agg,
    }


def _write_markdown(results: dict[str, Any], path: Path) -> None:
    lines: list[str] = ["# EIG selection stability", ""]
    lines.append(
        "Gold ranking uses one large-budget EIG pass per (domain, user) over a frozen "
        "mid-session posterior. Each smaller budget is repeated with fresh RNGs."
    )
    lines.append("")
    for domain, block in results["domains"].items():
        lines.append(f"## {domain}")
        agg = block["aggregate"]
        if "error" in agg:
            lines.append(f"ERROR: {agg['error']}")
            lines.append("")
            continue
        spreads = ", ".join(f"{s:.3f}" for s in agg["gold_pool_spreads"])
        lines.append(
            f"{agg['n_users']} users. Gold pool spread "
            f"(max-median)/max per user: [{spreads}]."
        )
        lines.append("")
        lines.append("| n_samples | P(argmax = gold) | P(argmax in gold top-5) "
                     "| mean Kendall tau |")
        lines.append("|---|---|---|---|")
        for b, m in agg["per_budget"].items():
            tau_s = f"{m['mean_kendall_tau']:.3f}" if m["mean_kendall_tau"] is not None else "n/a"
            lines.append(
                f"| {b} | {m['p_argmax_match']:.2f} | "
                f"{m['p_argmax_in_gold_top5']:.2f} | {tau_s} |"
            )
        lines.append("")
        for r in block["per_user"]:
            if "error" in r:
                lines.append(f"- user {r['user_idx']}: ERROR {r['error']}")
                continue
            top5 = ", ".join(f"{v:.4f}" for v in r["gold"]["top5_values"])
            lines.append(
                f"- user {r['user_idx']}: pool={r['n_candidates']}, gold argmax targets "
                f"`{r['gold']['argmax_target_param']}`, gold top-5 EIG = [{top5}], "
                f"spread = {r['gold']['pool_spread']:.3f}"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EIG selection stability vs Monte Carlo sample budget",
    )
    parser.add_argument(
        "--domains", nargs="+", default=["game_a", "stock"],
        choices=list(DOMAIN_FACTORIES.keys()),
    )
    parser.add_argument("--n-users", type=int, default=3)
    parser.add_argument("--n-particles", type=int, default=2000)
    parser.add_argument("--elicitation-rounds", type=int, default=5)
    parser.add_argument("--elicit-eig-samples", type=int, default=200,
                        help="EIG samples used during the warm-up elicitation")
    parser.add_argument("--n-scenarios-per-round", type=int, default=50)
    parser.add_argument("--n-per-param", type=int, default=17,
                        help="Pool size per target parameter (~3x this total)")
    parser.add_argument("--gold-samples", type=int, default=20000)
    parser.add_argument("--sample-grid", type=int, nargs="+",
                        default=[100, 200, 500, 800, 1600])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Process-parallelism over (domain, user) jobs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/diagnostics"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        JobSpec(
            domain=domain,
            user_idx=u,
            seed=args.seed,
            n_particles=args.n_particles,
            elicitation_rounds=args.elicitation_rounds,
            elicit_eig_samples=args.elicit_eig_samples,
            n_scenarios_per_round=args.n_scenarios_per_round,
            n_per_param=args.n_per_param,
            gold_samples=args.gold_samples,
            sample_grid=list(args.sample_grid),
            n_reps=args.reps,
            temperature=args.temperature,
        )
        for domain in args.domains
        for u in range(args.n_users)
    ]

    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            job_results = list(pool.map(run_domain_user_job, specs))
    else:
        job_results = [run_domain_user_job(s) for s in specs]

    results: dict[str, Any] = {
        "config": {
            **{k: v for k, v in vars(args).items() if k != "output_dir"},
            "output_dir": str(args.output_dir),
        },
        "domains": {},
    }
    for domain in args.domains:
        per_user = [r for r in job_results if r["domain"] == domain]
        results["domains"][domain] = {
            "per_user": per_user,
            "aggregate": _aggregate_domain(per_user),
        }

    json_path = args.output_dir / "eig_stability.json"
    md_path = args.output_dir / "eig_stability.md"
    json_path.write_text(json.dumps(results, indent=2))
    _write_markdown(results, md_path)
    logger.info("Wrote %s and %s", json_path, md_path)

    for domain, block in results["domains"].items():
        agg = block["aggregate"]
        if "error" in agg:
            continue
        for b, m in agg["per_budget"].items():
            logger.info(
                "[%s] n_samples=%s: P(argmax=gold)=%.2f, mean tau=%s",
                domain, b, m["p_argmax_match"], m["mean_kendall_tau"],
            )


if __name__ == "__main__":
    main()
