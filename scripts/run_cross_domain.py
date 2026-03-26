"""Cross-domain transfer: resource game elicitation vs stock recommendations (Milestone 5)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.elicitation_loop import ElicitationConfig
from src.agents.preference_tracker import ConvergenceConfig
from src.environments.stock_backtest import load_stock_config_yaml
from src.evaluation.experiment_runner import run_cross_domain_transfer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Game-to-stock cross-domain transfer experiment",
    )
    parser.add_argument("--n-users", type=int, default=3, help="Synthetic users")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--n-particles", type=int, default=80)
    parser.add_argument("--n-eig-samples", type=int, default=32)
    parser.add_argument("--n-scenarios-per-round", type=int, default=6)
    parser.add_argument("--posterior-type", default="particle", choices=["particle", "gaussian"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stock-config",
        type=str,
        default=None,
        help="Optional path to stock YAML (default: configs/stock/default.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON path for results",
    )
    args = parser.parse_args()

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

    stock_cfg = None
    if args.stock_config is not None:
        stock_cfg = load_stock_config_yaml(args.stock_config)

    result = run_cross_domain_transfer(
        n_users=args.n_users,
        elicitation=elicitation,
        seed=args.seed,
        stock_config=stock_cfg,
    )

    payload = {
        "generic": result.generic,
        "within_domain": result.within_domain,
        "cross_domain": result.cross_domain,
        "n_users": result.n_users,
    }
    text = json.dumps(payload, indent=2)
    logger.info("Cross-domain transfer summary:\n%s", text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
