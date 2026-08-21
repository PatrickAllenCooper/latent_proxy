"""Canonical learning-curve campaign: recovery vs. rounds, by query strategy.

Runs the paired experiment showing HOW preference recovery improves with each
binary choice, decomposed by query strategy (arm):

    active    -- EIG-adaptive selection over the scenario-library pool
    fixed     -- static questionnaire, prior-EIG-ranked, round-robin balanced
    random    -- uniform draws from the scenario library
    dirichlet -- fully random Dirichlet option pairs (uninformed floor)

Paired design: for a given (domain, seed), the n_users thetas are sampled ONCE
(SyntheticUserSampler(seed=seed)); every arm sees the same users in the same
order with the same choice-noise seeds and identical env.reset(seed=seed+i).
Early stopping is disabled so every user answers exactly max_rounds queries.

CPU-only, no GPU or LLM required. Resumable: existing per-user JSON files are
skipped.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop
from src.agents.preference_tracker import ConvergenceConfig
from src.evaluation.alignment_metrics import compute_alignment_score
from src.evaluation.elicitation_metrics import compute_recovery_curve
from src.evaluation.generalization_protocol import DOMAIN_FACTORIES, SCENARIO_LIBRARIES
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ARMS = ("active", "fixed", "random", "dirichlet")

# Distinct generator seed offsets per arm so query-selection rng streams do
# not collide across arms (user choice-noise and env seeds stay identical).
ARM_SEED_OFFSETS = {"active": 0, "fixed": 1, "random": 2, "dirichlet": 3}

TEMPERATURE = 0.1


def _worker_init() -> None:
    """Keep each worker single-threaded so 12 workers use ~12 cores."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _user_theta(seed: int, user_idx: int) -> Any:
    """The user_idx-th theta of the shared (domain, seed) user panel.

    The panel is defined purely by SyntheticUserSampler(seed=seed), so every
    arm reconstructs the identical user in the identical order.
    """
    sampler = SyntheticUserSampler(seed=seed)
    thetas = [sampler.sample() for _ in range(user_idx + 1)]
    return thetas[user_idx]


def _no_early_stop_convergence(max_rounds: int) -> ConvergenceConfig:
    """A ConvergenceConfig no criterion can trigger before max_rounds.

    Variance thresholds of 0.0 can never be reached (posterior variance stays
    strictly positive), so check_convergence only fires via its
    n_observations >= max_rounds branch.
    """
    return ConvergenceConfig(
        gamma_variance_threshold=0.0,
        alpha_variance_threshold=0.0,
        lambda_variance_threshold=0.0,
        max_rounds=max_rounds,
    )


def _serialize_history(
    history: list[tuple[Any, int]],
) -> list[dict[str, Any]]:
    """Full per-round query/choice record (arrays as lists)."""
    rounds = []
    for scenario, choice in history:
        rounds.append({
            "option_a": [float(x) for x in scenario.option_a],
            "option_b": [float(x) for x in scenario.option_b],
            "channel_means": [float(x) for x in scenario.channel_means],
            "channel_variances": [float(x) for x in scenario.channel_variances],
            "current_wealth": float(scenario.current_wealth),
            "rounds_remaining": int(scenario.rounds_remaining),
            "target_param": scenario.target_param,
            "multiperiod_horizon": (
                int(scenario.multiperiod_horizon)
                if scenario.multiperiod_horizon is not None else None
            ),
            "choice": int(choice),
        })
    return rounds


def run_user_task(task: dict[str, Any]) -> dict[str, Any]:
    """Run one (domain, arm, seed, user_idx) elicitation and write its JSON.

    Constructs env/user/loop inside the worker (nothing heavy is pickled).
    """
    domain = task["domain"]
    arm = task["arm"]
    seed = task["seed"]
    user_idx = task["user_idx"]
    out_path = Path(task["out_path"])

    t0 = time.monotonic()

    ut = _user_theta(seed, user_idx)
    true_theta = {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}

    env = DOMAIN_FACTORIES[domain]()
    library = SCENARIO_LIBRARIES[domain](seed + user_idx + 4000)
    user = SyntheticUser(ut, temperature=TEMPERATURE, seed=seed + user_idx)

    config = ElicitationConfig(
        posterior_type="particle",
        n_particles=task["n_particles"],
        max_rounds=task["max_rounds"],
        n_scenarios_per_round=task["n_scenarios_per_round"],
        n_eig_samples=task["n_eig_samples"],
        temperature=TEMPERATURE,
        convergence=_no_early_stop_convergence(task["max_rounds"]),
        seed=seed + user_idx * 1000 + ARM_SEED_OFFSETS.get(arm, 9),
        scenario_library=library,
        reference_point_mode=task.get("reference_point_mode", "zero"),
    )
    loop = ElicitationLoop(config)

    env.reset(seed=seed + user_idx)
    result = loop.run(env, user, query_type=arm)
    curve = compute_recovery_curve(result)

    # Re-reset so the alignment computation sees the same env state in
    # every arm (query generators mutate env state during the loop).
    env.reset(seed=seed + user_idx)
    opt_true = env.get_optimal_action(true_theta)
    opt_inferred = env.get_optimal_action(result.inferred_theta)
    spearman = compute_alignment_score([opt_inferred], [opt_true])

    record = {
        "domain": domain,
        "arm": arm,
        "seed": seed,
        "user_idx": user_idx,
        "true_theta": true_theta,
        "inferred_theta": result.inferred_theta,
        "n_rounds": result.n_rounds,
        "convergence_reason": result.convergence_reason,
        "mean_trajectory": result.mean_trajectory,
        "variance_trajectory": result.variance_trajectory,
        "per_round_error": curve["per_round_error"],
        "final_error": curve["final_error"],
        "alignment": {
            "optimal_action_true": [float(x) for x in opt_true],
            "optimal_action_inferred": [float(x) for x in opt_inferred],
            "spearman": float(spearman),
        },
        "history": _serialize_history(result.history),
        "elapsed_seconds": time.monotonic() - t0,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(record, f)
    tmp_path.rename(out_path)

    return {
        "domain": domain, "arm": arm, "seed": seed, "user_idx": user_idx,
        "elapsed_seconds": record["elapsed_seconds"],
        "final_total_error": (curve["final_error"] or {}).get("total"),
    }


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_manifest(output_dir: Path, args: argparse.Namespace) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
        "config": {
            "domains": args.domains,
            "arms": args.arms,
            "n_users": args.n_users,
            "seeds": args.seeds,
            "max_rounds": args.max_rounds,
            "n_particles": args.n_particles,
            "n_eig_samples": args.n_eig_samples,
            "n_scenarios_per_round": args.n_scenarios_per_round,
            "workers": args.workers,
            "reference_point_mode": args.reference_point_mode,
            "temperature": TEMPERATURE,
            "posterior_type": "particle",
            "arm_seed_offsets": ARM_SEED_OFFSETS,
            "early_stopping": "disabled (variance thresholds 0.0)",
        },
        "command": " ".join(sys.argv),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def build_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    """Enumerate pending (domain, arm, seed, user_idx) tasks, skipping done ones."""
    output_dir = Path(args.output_dir)
    tasks: list[dict[str, Any]] = []
    n_skipped = 0
    for domain in args.domains:
        if domain not in DOMAIN_FACTORIES:
            raise ValueError(
                f"Unknown domain {domain!r}; known: {sorted(DOMAIN_FACTORIES)}"
            )
        for arm in args.arms:
            if arm not in ARMS:
                raise ValueError(f"Unknown arm {arm!r}; known: {ARMS}")
            for seed in args.seeds:
                for user_idx in range(args.n_users):
                    out_path = output_dir / domain / arm / f"s{seed}_u{user_idx}.json"
                    if out_path.exists():
                        n_skipped += 1
                        continue
                    tasks.append({
                        "domain": domain,
                        "arm": arm,
                        "seed": seed,
                        "user_idx": user_idx,
                        "max_rounds": args.max_rounds,
                        "n_particles": args.n_particles,
                        "n_eig_samples": args.n_eig_samples,
                        "n_scenarios_per_round": args.n_scenarios_per_round,
                        "reference_point_mode": args.reference_point_mode,
                        "out_path": str(out_path),
                    })
    return tasks, n_skipped


def run_campaign(args: argparse.Namespace) -> dict[str, int]:
    """Run all pending tasks; returns {n_tasks, n_completed, n_skipped, n_failed}."""
    output_dir = Path(args.output_dir)
    _write_manifest(output_dir, args)

    tasks, n_skipped = build_tasks(args)
    n_total = len(tasks) + n_skipped
    logger.info(
        "Campaign: %d tasks total, %d already done, %d to run (workers=%d)",
        n_total, n_skipped, len(tasks), args.workers,
    )

    n_completed = 0
    n_failed = 0
    t0 = time.monotonic()

    if args.workers <= 1:
        for task in tasks:
            try:
                summary = run_user_task(task)
                n_completed += 1
                _log_progress(summary, n_completed, len(tasks), t0)
            except Exception:
                n_failed += 1
                logger.exception("Task failed: %s", task)
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_worker_init,
        ) as pool:
            futures = {pool.submit(run_user_task, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    summary = future.result()
                    n_completed += 1
                    _log_progress(summary, n_completed, len(tasks), t0)
                except Exception:
                    n_failed += 1
                    logger.exception("Task failed: %s", task)

    logger.info(
        "Done: %d completed, %d skipped (resume), %d failed in %.1f min",
        n_completed, n_skipped, n_failed, (time.monotonic() - t0) / 60.0,
    )
    return {
        "n_tasks": n_total,
        "n_completed": n_completed,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
    }


def _log_progress(
    summary: dict[str, Any], done: int, total: int, t0: float,
) -> None:
    elapsed = time.monotonic() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta_min = (total - done) / rate / 60.0 if rate > 0 else float("inf")
    logger.info(
        "[%d/%d] %s/%s s%d u%d: err=%s (%.1fs) ETA %.1f min",
        done, total, summary["domain"], summary["arm"],
        summary["seed"], summary["user_idx"],
        (f"{summary['final_total_error']:.3f}"
         if summary["final_total_error"] is not None else "n/a"),
        summary["elapsed_seconds"], eta_min,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical learning-curve campaign (paired arms x domains)"
    )
    parser.add_argument(
        "--domains", default=",".join(DOMAIN_FACTORIES),
        help="Comma-separated domain names",
    )
    parser.add_argument(
        "--arms", default=",".join(ARMS),
        help="Comma-separated query strategies",
    )
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument(
        "--seeds", default="42,43,44,45,46",
        help="Comma-separated seeds (one user panel per (domain, seed))",
    )
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--n-particles", type=int, default=2000)
    parser.add_argument("--n-eig-samples", type=int, default=800)
    parser.add_argument("--n-scenarios-per-round", type=int, default=50)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", default="outputs/canonical")
    parser.add_argument(
        "--reference-point-mode", default="zero", choices=["zero", "current_wealth"],
        help="Prospect-theory reference point: 'zero' (historical default) or "
             "'current_wealth' (fixes lambda unidentifiability, see refpoint study)",
    )

    args = parser.parse_args(argv)
    args.domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_campaign(args)


if __name__ == "__main__":
    main()
