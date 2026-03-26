"""Milestone 4 full evaluation: variants, transfer, ablations, plots, exports."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.evaluation.ablation_runner import (
    AblationResults,
    default_beta_values,
    default_posterior_types,
    run_sweep,
)
from src.evaluation.experiment_runner import ExperimentConfig, run_full_evaluation, run_transfer_experiment
from src.utils.visualization import (
    format_results_table,
    plot_ablation_sweep,
    plot_transfer_comparison,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4 full evaluation pipeline")
    parser.add_argument("--n-users", type=int, default=15, help="Users per benchmark / transfer")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--n-particles", type=int, default=1500)
    parser.add_argument("--n-eig-samples", type=int, default=800)
    parser.add_argument("--posterior-type", default="particle", choices=["particle", "gaussian"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation_m4")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    convergence = ConvergenceConfig(max_rounds=args.max_rounds)
    elicitation = ElicitationConfig(
        posterior_type=args.posterior_type,
        n_particles=args.n_particles,
        max_rounds=args.max_rounds,
        n_eig_samples=args.n_eig_samples,
        convergence=convergence,
        seed=args.seed,
    )

    exp_cfg = ExperimentConfig(
        variants=["a", "b"],
        n_users=args.n_users,
        seed=args.seed,
        elicitation=elicitation,
    )

    logger.info("Running full evaluation (variants a, b)...")
    full_eval = run_full_evaluation(exp_cfg)

    bundle: dict = {"full_evaluation": full_eval, "transfer": None, "ablations": {}}

    if not args.skip_transfer:
        logger.info("Running transfer experiment (A -> B)...")
        transfer = run_transfer_experiment(
            n_users=min(args.n_users, 12),
            elicitation=elicitation,
            seed=args.seed + 1,
        )
        bundle["transfer"] = {
            "generic": transfer.generic,
            "within_domain": transfer.within_domain,
            "cross_domain": transfer.cross_domain,
            "n_users": transfer.n_users,
        }
        if not args.skip_plots:
            fig_t = plot_transfer_comparison(transfer)
            fig_t.savefig(out_dir / "transfer_comparison.png", dpi=150)
            logger.info("Wrote transfer plot to %s", out_dir / "transfer_comparison.png")

    if not args.skip_ablation:
        logger.info("Running ablation sweeps (reduced user count)...")
        n_ab = max(5, min(8, args.n_users // 2))
        bundle["ablations"]["query_budget"] = asdict(
            run_sweep(
                "max_rounds",
                [3, 5, 8, 10],
                elicitation,
                n_users=n_ab,
                seed=args.seed + 2,
            ),
        )
        bundle["ablations"]["posterior_type"] = asdict(
            run_sweep(
                "posterior_type",
                default_posterior_types(),
                elicitation,
                n_users=n_ab,
                seed=args.seed + 3,
            ),
        )
        bundle["ablations"]["beta_blend"] = asdict(
            run_sweep(
                "beta",
                default_beta_values(),
                elicitation,
                n_users=n_ab,
                seed=args.seed + 4,
            ),
        )

        if not args.skip_plots:
            qb = AblationResults(**bundle["ablations"]["query_budget"])
            fig_q = plot_ablation_sweep(qb, "mean_active_error")
            fig_q.savefig(out_dir / "ablation_query_budget.png", dpi=150)

    md = format_results_table(full_eval)
    (out_dir / "results_summary.md").write_text(md, encoding="utf-8")
    save_results(bundle, out_dir / "results_bundle.json")
    logger.info("Wrote %s and results_bundle.json", out_dir / "results_summary.md")


if __name__ == "__main__":
    main()
