"""Adherence study: does a DPO-tuned checkpoint act on a stated preference profile?

See src/evaluation/adherence_study.py for the two theta_modes this compares
(explicit true profile vs. an elicited-via-active-questioning estimate).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.evaluation.adherence_study import (
    THETA_MODES,
    AdherenceStudyConfig,
    compare_conditions,
    run_adherence_study,
)
from src.utils.visualization import save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO adherence study")
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--phase1-checkpoint", default=None)
    parser.add_argument("--phase2-checkpoint", default=None)
    parser.add_argument("--environment", default="game")
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--output-dir", default="outputs/adherence_study")
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--n-eig-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument(
        "--conditions", default="base,dpo_phase1,dpo_phase2",
        help="Comma-separated conditions. Valid: base, dpo_phase1, dpo_phase2.",
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conv = ConvergenceConfig(max_rounds=args.max_rounds)
    analytical_cfg = ElicitationConfig(
        posterior_type="particle",
        n_particles=args.n_particles,
        max_rounds=args.max_rounds,
        n_scenarios_per_round=15,
        n_eig_samples=args.n_eig_samples,
        convergence=conv,
        seed=args.seed,
    )

    study_cfg = AdherenceStudyConfig(
        n_users=args.n_users,
        environment=args.environment,
        base_model_path=args.base_model,
        phase1_checkpoint=args.phase1_checkpoint,
        phase2_checkpoint=args.phase2_checkpoint,
        conditions=conditions,
        analytical_elicitation=analytical_cfg,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    logger.info(
        "Starting adherence study (n_users=%d, conditions=%s)", args.n_users, conditions,
    )
    results = run_adherence_study(study_cfg)

    bundle: dict = {
        "config": {
            "n_users": args.n_users,
            "environment": args.environment,
            "base_model": args.base_model,
            "phase1_checkpoint": args.phase1_checkpoint,
            "phase2_checkpoint": args.phase2_checkpoint,
            "conditions": conditions,
            "seed": args.seed,
        },
        "results": {},
        "hypothesis_tests": {},
    }
    for condition, by_mode in results.items():
        bundle["results"][condition] = {
            mode: {
                "mean_alignment": r.mean_alignment,
                "mean_violation": r.mean_violation,
                "alignment_scores": r.alignment_scores,
                "violation_rates": r.violation_rates,
            }
            for mode, r in by_mode.items()
        }

    for mode in THETA_MODES:
        for other in ("base", "dpo_phase1"):
            if "dpo_phase2" in results and other in results:
                t = compare_conditions(results, "dpo_phase2", other, mode)
                bundle["hypothesis_tests"][f"dpo_phase2_vs_{other}_{mode}"] = {
                    "hypothesis": t.hypothesis,
                    "p_value": t.p_value,
                    "effect_size": t.effect_size,
                    "conclusion": t.conclusion,
                }
                logger.info(
                    "dpo_phase2 vs %s (%s): p=%.4f, effect=%.3f, %s",
                    other, mode, t.p_value, t.effect_size, t.conclusion,
                )

    save_results(bundle, out_dir / "adherence_study_results.json")
    logger.info("Wrote results to %s", out_dir / "adherence_study_results.json")

    summary = json.dumps(
        {c: {m: r["mean_alignment"] for m, r in modes.items()} for c, modes in bundle["results"].items()},
        indent=2,
    )
    logger.info("Summary (mean alignment score by condition x theta_mode):\n%s", summary)


if __name__ == "__main__":
    main()
