"""Milestone 6: full generalization study across game, stock, supply chain."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.evaluation.generalization_protocol import (
    GeneralizationStudyConfig,
    run_generalization_study,
)
from src.utils.visualization import (
    plot_hypothesis_panel,
    plot_parameter_transfer_heatmap,
    plot_transfer_matrix,
    plot_stability_scatter,
    save_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 6 generalization study",
    )
    parser.add_argument("--n-users", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--n-particles", type=int, default=80)
    parser.add_argument("--n-eig-samples", type=int, default=32)
    parser.add_argument("--n-scenarios-per-round", type=int, default=6)
    parser.add_argument("--posterior-type", default="particle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/generalization")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--skip-h4", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--quick", action="store_true",
        help="Tiny settings for smoke test",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_users = 2
        args.max_rounds = 2
        args.n_particles = 64
        args.n_eig_samples = 24
        args.n_scenarios_per_round = 6

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    convergence = ConvergenceConfig(max_rounds=args.max_rounds)
    elicitation = ElicitationConfig(
        posterior_type=args.posterior_type,
        n_particles=args.n_particles,
        max_rounds=args.max_rounds,
        n_scenarios_per_round=args.n_scenarios_per_round,
        n_eig_samples=args.n_eig_samples,
        convergence=convergence,
        seed=args.seed,
    )

    config = GeneralizationStudyConfig(
        n_users=args.n_users,
        n_stability_sessions=0 if args.skip_stability else 2,
        elicitation=elicitation,
        seed=args.seed,
        run_h4=not args.skip_h4,
    )

    logger.info("Running generalization study (n_users=%d) ...", config.n_users)
    result = run_generalization_study(config)

    bundle: dict = {
        "per_domain_pair": {},
        "per_parameter_transfer": result.per_parameter_transfer,
        "hypothesis_results": {},
        "stability": {},
    }
    for key, tr in result.per_domain_pair.items():
        bundle["per_domain_pair"][key] = {
            "generic": tr.generic,
            "within_domain": tr.within_domain,
            "cross_domain": tr.cross_domain,
            "n_users": tr.n_users,
        }
    for h_key, h_list in result.hypothesis_results.items():
        bundle["hypothesis_results"][h_key] = [
            {
                "hypothesis": r.hypothesis,
                "p_value": r.p_value,
                "effect_size": r.effect_size,
                "conclusion": r.conclusion,
                "method": r.method_name,
            }
            for r in h_list
        ]
    for domain, stab in result.stability.items():
        bundle["stability"][domain] = {
            "icc": stab.icc_per_param,
            "pearson": stab.pearson_per_param,
            "n_users": stab.n_users,
        }
    if result.h4_details:
        bundle["h4_efficiency"] = result.h4_details

    save_results(bundle, out_dir / "results_bundle.json")
    logger.info("Wrote results to %s", out_dir / "results_bundle.json")

    summary = json.dumps(
        {k: v for k, v in bundle.items() if k != "per_domain_pair"},
        indent=2,
        default=str,
    )
    logger.info("Summary:\n%s", summary)

    if not args.skip_plots:
        fig_h = plot_hypothesis_panel(result.hypothesis_results)
        fig_h.savefig(out_dir / "hypothesis_panel.png", dpi=150)
        logger.info("Wrote hypothesis_panel.png")

        fig_heat = plot_parameter_transfer_heatmap(result.per_parameter_transfer)
        fig_heat.savefig(out_dir / "parameter_heatmap.png", dpi=150)
        logger.info("Wrote parameter_heatmap.png")

        cross_align = {
            k: tr.cross_domain["mean"] for k, tr in result.per_domain_pair.items()
        }
        fig_mat = plot_transfer_matrix(cross_align)
        fig_mat.savefig(out_dir / "transfer_matrix.png", dpi=150)
        logger.info("Wrote transfer_matrix.png")

        import matplotlib
        matplotlib.pyplot.close("all")


if __name__ == "__main__":
    main()
