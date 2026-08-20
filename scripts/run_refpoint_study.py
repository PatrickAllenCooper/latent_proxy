"""A/B study: prospect-theory reference-point mode (zero vs current_wealth).

For each mode x domain cell:
  (a) Recovery: run one elicitation per fresh user; record per-parameter
      absolute error (inferred vs true) and posterior contraction ratio
      (final/initial variance from the variance trajectory).
  (b) Stability: n_sessions independent elicitations per user; ICC(2,1)
      and Pearson per parameter, reusing the theta_stability machinery.

Seeds are mode-independent, so the two modes see identical users and
scenario-library seeds: the only difference between arms is the mode.

Example:
    python scripts/run_refpoint_study.py \
        --modes zero,current_wealth --domains game_a,stock,supply_chain \
        --n-users 30 --n-sessions 3 --max-rounds 10 --n-particles 1000 \
        --n-eig-samples 500 --seed 42 --workers 8 --output-dir outputs/refpoint
"""

from __future__ import annotations

import os

# Keep BLAS single-threaded inside workers (set before numpy import; under
# the spawn start method this module is re-imported in each worker).
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop
from src.agents.preference_tracker import ConvergenceConfig
from src.evaluation.alignment_metrics import compute_preference_recovery_error
from src.evaluation.generalization_protocol import DOMAIN_FACTORIES, SCENARIO_LIBRARIES
from src.evaluation.theta_stability import compute_stability_metrics, run_user_sessions
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler, UserType

logger = logging.getLogger("refpoint_study")

PARAMS = ["gamma", "alpha", "lambda_"]
TEMPERATURE = 0.1


def _domain_seed(base_seed: int, domain: str) -> int:
    """Deterministic, mode-independent seed for a domain cell."""
    return int(base_seed + zlib.crc32(domain.encode()) % 100_000)


def _make_config(
    mode: str,
    knobs: dict[str, Any],
    seed: int,
    scenario_library: Any,
) -> ElicitationConfig:
    return ElicitationConfig(
        posterior_type="particle",
        n_particles=int(knobs["n_particles"]),
        max_rounds=int(knobs["max_rounds"]),
        n_scenarios_per_round=int(knobs["n_scenarios_per_round"]),
        n_eig_samples=int(knobs["n_eig_samples"]),
        temperature=TEMPERATURE,
        convergence=ConvergenceConfig(max_rounds=int(knobs["max_rounds"])),
        seed=seed,
        scenario_library=scenario_library,
        reference_point_mode=mode,
    )


def recovery_task(payload: dict[str, Any]) -> dict[str, Any]:
    """One user's elicitation: per-parameter |error| and contraction ratio."""
    mode = payload["mode"]
    domain = payload["domain"]
    i = int(payload["user_index"])
    dseed = int(payload["domain_seed"])
    knobs = payload["knobs"]

    ut = UserType(**payload["true_theta"])
    lib = SCENARIO_LIBRARIES[domain](dseed + i + 4000)
    cfg = _make_config(mode, knobs, seed=dseed + i * 1000 + 7, scenario_library=lib)

    env = DOMAIN_FACTORIES[domain]()
    env.reset(seed=dseed + i + 9000)
    user = SyntheticUser(ut, temperature=TEMPERATURE, seed=dseed + i + 333)

    res = ElicitationLoop(cfg).run(env, user, query_type="active")

    recovery = compute_preference_recovery_error(res.inferred_theta, ut)
    v0 = res.variance_trajectory[0]
    vf = res.variance_trajectory[-1]
    contraction = {p: float(vf[p] / max(v0[p], 1e-30)) for p in PARAMS}

    return {
        "kind": "recovery",
        "mode": mode,
        "domain": domain,
        "user_index": i,
        "true_theta": payload["true_theta"],
        "inferred_theta": res.inferred_theta,
        "abs_error": {p: float(recovery[p]) for p in PARAMS},
        "contraction_ratio": contraction,
        "n_rounds": res.n_rounds,
        "convergence_reason": res.convergence_reason,
    }


def stability_task(payload: dict[str, Any]) -> dict[str, Any]:
    """One user's n_sessions independent sessions (theta_stability seeding)."""
    mode = payload["mode"]
    domain = payload["domain"]
    i = int(payload["user_index"])
    dseed = int(payload["domain_seed"])
    knobs = payload["knobs"]

    ut = UserType(**payload["true_theta"])
    lib = SCENARIO_LIBRARIES[domain](dseed + 7777)
    elic = _make_config(mode, knobs, seed=dseed, scenario_library=lib)

    thetas = run_user_sessions(
        ut,
        DOMAIN_FACTORIES[domain],
        elic,
        seed=dseed + 3000,
        user_index=i,
        n_sessions=int(payload["n_sessions"]),
    )
    return {
        "kind": "stability",
        "mode": mode,
        "domain": domain,
        "user_index": i,
        "true_theta": payload["true_theta"],
        "session_thetas": thetas,
    }


def _run_task(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["kind"] == "recovery":
        return recovery_task(payload)
    return stability_task(payload)


def _task_key(p: dict[str, Any]) -> str:
    return f"{p['kind']}__{p['mode']}__{p['domain']}__{p['user_index']}"


def _checkpoint_path(out_dir: Path) -> Path:
    return out_dir / "task_checkpoints.jsonl"


def _load_checkpoints(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Load previously completed task results, keyed by task key.

    Tolerates a truncated last line (process killed mid-write).
    """
    path = _checkpoint_path(out_dir)
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[_task_key(rec)] = rec
    return done


def _sample_users(seed: int, n_users: int) -> list[dict[str, float]]:
    sampler = SyntheticUserSampler(seed=seed)
    return [
        {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
        for ut in sampler.sample_batch(n_users)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", type=str, default="zero,current_wealth")
    parser.add_argument(
        "--domains", type=str, default="game_a,stock,supply_chain",
        help=f"Comma list from: {', '.join(DOMAIN_FACTORIES)}",
    )
    parser.add_argument("--n-users", type=int, default=30)
    parser.add_argument("--n-sessions", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--n-particles", type=int, default=1000)
    parser.add_argument("--n-eig-samples", type=int, default=500)
    parser.add_argument("--n-scenarios-per-round", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs/refpoint")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    for m in modes:
        if m not in ("zero", "current_wealth"):
            parser.error(f"unknown mode: {m}")
    for d in domains:
        if d not in DOMAIN_FACTORIES:
            parser.error(f"unknown domain: {d} (choose from {list(DOMAIN_FACTORIES)})")

    out_dir = Path(args.output_dir)
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)

    knobs = {
        "max_rounds": args.max_rounds,
        "n_particles": args.n_particles,
        "n_eig_samples": args.n_eig_samples,
        "n_scenarios_per_round": args.n_scenarios_per_round,
    }

    # Mode-independent users per domain: both arms see the same population.
    payloads: list[dict[str, Any]] = []
    for domain in domains:
        dseed = _domain_seed(args.seed, domain)
        recovery_users = _sample_users(dseed + 500, args.n_users)
        stability_users = _sample_users(dseed + 3000, args.n_users)
        for mode in modes:
            for i, theta in enumerate(recovery_users):
                payloads.append({
                    "kind": "recovery", "mode": mode, "domain": domain,
                    "user_index": i, "true_theta": theta,
                    "domain_seed": dseed, "knobs": knobs,
                })
            for i, theta in enumerate(stability_users):
                payloads.append({
                    "kind": "stability", "mode": mode, "domain": domain,
                    "user_index": i, "true_theta": theta,
                    "domain_seed": dseed, "knobs": knobs,
                    "n_sessions": args.n_sessions,
                })

    checkpointed = _load_checkpoints(out_dir)
    remaining = [p for p in payloads if _task_key(p) not in checkpointed]
    logger.info(
        "Submitting %d tasks (%d modes x %d domains x %d users x {recovery, stability}) "
        "on %d workers -- %d already checkpointed, %d to run",
        len(payloads), len(modes), len(domains), args.n_users, args.workers,
        len(checkpointed), len(remaining),
    )

    t0 = time.time()
    results: list[dict[str, Any]] = list(checkpointed.values())
    n_done = len(results)
    if remaining:
        with open(_checkpoint_path(out_dir), "a", buffering=1) as ckpt_f, \
                ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_task, p): p for p in remaining}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    r = fut.result()
                except Exception:
                    logger.exception(
                        "Task failed: %s %s/%s user %s",
                        p["kind"], p["mode"], p["domain"], p["user_index"],
                    )
                    raise
                results.append(r)
                ckpt_f.write(json.dumps(r) + "\n")
                n_done += 1
                if n_done % 10 == 0 or n_done == len(payloads):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    remaining_n = len(payloads) - n_done
                    logger.info(
                        "%d/%d tasks done (%.1fs elapsed, est remaining %.1fs)",
                        n_done, len(payloads), elapsed,
                        remaining_n / rate if rate > 0 else float("nan"),
                    )

    # ---- Aggregate per cell -------------------------------------------------
    summary: dict[str, Any] = {
        "config": {
            "modes": modes, "domains": domains,
            "n_users": args.n_users, "n_sessions": args.n_sessions,
            "seed": args.seed, "temperature": TEMPERATURE, **knobs,
        },
        "cells": {},
    }

    for mode in modes:
        summary["cells"][mode] = {}
        for domain in domains:
            rec = sorted(
                (r for r in results
                 if r["kind"] == "recovery"
                 and r["mode"] == mode and r["domain"] == domain),
                key=lambda r: r["user_index"],
            )
            stab = sorted(
                (r for r in results
                 if r["kind"] == "stability"
                 and r["mode"] == mode and r["domain"] == domain),
                key=lambda r: r["user_index"],
            )
            per_user_sessions = [r["session_thetas"] for r in stab]
            icc, pearson = compute_stability_metrics(
                per_user_sessions,
                param_names=PARAMS,
                all_pairs=args.n_sessions > 2,
            )

            cell_metrics = {}
            for p in PARAMS:
                cell_metrics[p] = {
                    "mae": float(np.mean([r["abs_error"][p] for r in rec])),
                    "contraction_ratio": float(
                        np.mean([r["contraction_ratio"][p] for r in rec])
                    ),
                    "icc": icc[p],
                    "pearson": pearson[p],
                }
            summary["cells"][mode][domain] = cell_metrics

            cell_path = out_dir / "cells" / f"{mode}__{domain}.json"
            with open(cell_path, "w") as f:
                json.dump({
                    "mode": mode,
                    "domain": domain,
                    "metrics": cell_metrics,
                    "recovery": rec,
                    "stability": stab,
                }, f, indent=2)
            logger.info(
                "[%s/%s] %s",
                mode, domain,
                {p: {k: round(v, 4) for k, v in m.items()}
                 for p, m in cell_metrics.items()},
            )

    summary["wall_clock_seconds"] = time.time() - t0
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Done in %.1fs. Summary: %s", summary["wall_clock_seconds"],
        out_dir / "summary.json",
    )


if __name__ == "__main__":
    main()
