"""Run the elicitation benchmark: active learning vs random baseline.

This is a CPU-only script that uses synthetic users. No GPU or LLM required.
Validates that EIG-based query selection recovers user preferences faster
than random query selection.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.resource_game import ResourceStrategyGame
from src.evaluation.elicitation_metrics import run_elicitation_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elicitation benchmark: active vs random"
    )
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--n-particles", type=int, default=1000)
    parser.add_argument("--n-scenarios", type=int, default=50)
    parser.add_argument("--n-eig-samples", type=int, default=500)
    parser.add_argument("--posterior-type", default="particle",
                        choices=["particle", "gaussian"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    convergence = ConvergenceConfig(max_rounds=args.max_rounds)
    config = ElicitationConfig(
        posterior_type=args.posterior_type,
        n_particles=args.n_particles,
        max_rounds=args.max_rounds,
        n_scenarios_per_round=args.n_scenarios,
        n_eig_samples=args.n_eig_samples,
        convergence=convergence,
        seed=args.seed,
    )

    env = ResourceStrategyGame()
    logger.info("Running elicitation benchmark with %d users...", args.n_users)

    results = run_elicitation_benchmark(
        env, n_users=args.n_users, config=config, seed=args.seed,
    )

    eff = results["efficiency"]
    logger.info("=== Benchmark Results ===")
    logger.info("Active:  %.1f mean rounds, %.4f mean error",
                eff["mean_active_rounds"], eff["mean_active_error"])
    logger.info("Random:  %.1f mean rounds, %.4f mean error",
                eff["mean_random_rounds"], eff["mean_random_error"])
    logger.info("Error reduction: %.1f%%", eff["error_reduction_pct"])
    logger.info("Round reduction: %.1f%%", eff["round_reduction_pct"])


if __name__ == "__main__":
    main()
