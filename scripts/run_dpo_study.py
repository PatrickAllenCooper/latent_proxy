"""DPO elicitation study: base LLM vs DPO Phase 1 vs Phase 2 vs analytical."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.llm_elicitation import LLMElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.evaluation.dpo_study import DPOStudyConfig, run_dpo_study
from src.utils.visualization import save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO elicitation study")
    parser.add_argument("--n-users", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--phase1-checkpoint", default=None)
    parser.add_argument("--phase2-checkpoint", default=None)
    parser.add_argument("--envs", nargs="+", default=["game", "stock", "supply_chain"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/dpo_study")
    parser.add_argument("--n-particles", type=int, default=200)
    parser.add_argument("--n-eig-samples", type=int, default=100)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

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

    llm_cfg = LLMElicitationConfig(
        max_rounds=args.max_rounds,
        max_new_tokens=256,
        temperature=0.3,
    )

    study_cfg = DPOStudyConfig(
        n_users=args.n_users,
        max_rounds=args.max_rounds,
        environments=args.envs,
        base_model_path=args.base_model,
        phase1_checkpoint=args.phase1_checkpoint,
        phase2_checkpoint=args.phase2_checkpoint,
        llm_config=llm_cfg,
        analytical_elicitation=analytical_cfg,
        seed=args.seed,
    )

    logger.info("Starting DPO elicitation study (n_users=%d, max_rounds=%d)", args.n_users, args.max_rounds)
    result = run_dpo_study(study_cfg)

    bundle: dict = {"config": result.config, "environments": {}, "hypothesis_tests": {}}
    for env_name, conditions in result.per_env.items():
        env_block: dict = {}
        for cond_name, cr in conditions.items():
            env_block[cond_name] = {
                "mean_alignment": cr.mean_alignment,
                "mean_violation": cr.mean_violation,
                "alignment_scores": cr.alignment_scores,
                "violation_rates": cr.violation_rates,
            }
        bundle["environments"][env_name] = env_block

    for env_name, tests in result.hypothesis_tests.items():
        bundle["hypothesis_tests"][env_name] = [
            {
                "hypothesis": t.hypothesis,
                "p_value": t.p_value,
                "effect_size": t.effect_size,
                "conclusion": t.conclusion,
            }
            for t in tests
        ]

    save_results(bundle, out_dir / "dpo_study_results.json")
    logger.info("Wrote results to %s", out_dir / "dpo_study_results.json")

    if not args.skip_plots:
        try:
            from src.utils.visualization import plot_dpo_comparison, plot_convergence_by_condition
            fig = plot_dpo_comparison(result.per_env)
            fig.savefig(out_dir / "dpo_comparison.png", dpi=150)
            logger.info("Wrote dpo_comparison.png")

            fig2 = plot_convergence_by_condition(result.per_env)
            fig2.savefig(out_dir / "convergence_by_condition.png", dpi=150)
            logger.info("Wrote convergence_by_condition.png")

            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

    summary = json.dumps(
        {env: {c: d["mean_alignment"] for c, d in conds.items()}
         for env, conds in bundle["environments"].items()},
        indent=2,
    )
    logger.info("Summary:\n%s", summary)


if __name__ == "__main__":
    main()
