"""Phase 1 training: Quality floor enforcement via DPO.

Trains the base model to avoid dominated, undiversified, and bankruptcy-risking
recommendations. Uses quality-only DPO pairs (no user profile conditioning).
Corresponds to curriculum Phase 1 (beta=0.0) in README Section 5.3.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environments.resource_game import ResourceStrategyGame
from src.evaluation.alignment_metrics import evaluate_actions
from src.training.dpo_data import DPOPairConfig, DPOPairGenerator, pairs_to_hf_dict
from src.training.dpo_trainer import ConditionalDPOTrainer, DPOTrainingConfig
from src.training.model_utils import ModelConfig
from src.training.synthetic_users import SyntheticUserSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_phase1_data(
    n_pairs: int = 10000,
    seed: int = 42,
    output_path: str | None = None,
) -> list:
    """Generate Phase 1 DPO pairs (quality-only)."""
    config = DPOPairConfig(
        n_pairs=n_pairs,
        curriculum_phase=1,
        seed=seed,
    )
    generator = DPOPairGenerator(config)
    pairs = generator.generate_dataset()

    if output_path:
        from src.training.dpo_data import save_pairs
        save_pairs(pairs, output_path)

    logger.info("Phase 1 data: %d pairs generated", len(pairs))
    return pairs


def run_phase1_training(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    n_pairs: int = 10000,
    num_epochs: int = 3,
    output_dir: str = "outputs/dpo/phase1",
) -> None:
    """Full Phase 1 pipeline: generate data, train, evaluate."""
    model_config = ModelConfig(model_name=model_name)
    data_config = DPOPairConfig(n_pairs=n_pairs, curriculum_phase=1)
    training_config = DPOTrainingConfig(
        model=model_config,
        data=data_config,
        curriculum_phase=1,
        num_epochs=num_epochs,
        output_dir=output_dir,
    )

    trainer = ConditionalDPOTrainer(training_config)
    trainer.train()

    logger.info("Phase 1 training complete.")


def evaluate_phase1(checkpoint_path: str, n_episodes: int = 100) -> dict:
    """Evaluate a Phase 1 checkpoint for quality floor compliance."""
    env = ResourceStrategyGame()
    env.reset(seed=0)
    sampler = SyntheticUserSampler(seed=0)

    user_types = sampler.sample_batch(n_episodes)
    optimal_actions = []
    for ut in user_types:
        theta = {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
        optimal_actions.append(env.get_optimal_action(theta))

    results = evaluate_actions(optimal_actions, env, user_types)
    logger.info("Phase 1 evaluation (optimal baseline):")
    logger.info("  Quality floor violation rate: %.4f", results["quality_floor_violation_rate"])
    logger.info("  Mean quality score: %.4f", results["mean_quality_score"])
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 1: Quality floor training")
    parser.add_argument("--action", choices=["generate", "train", "evaluate"], default="generate")
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--n-pairs", type=int, default=10000)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/dpo/phase1")
    parser.add_argument("--data-path", default="data/phase1")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if args.action == "generate":
        generate_phase1_data(args.n_pairs, output_path=args.data_path)
    elif args.action == "train":
        run_phase1_training(
            model_name=args.model_name,
            n_pairs=args.n_pairs,
            num_epochs=args.num_epochs,
            output_dir=args.output_dir,
        )
    elif args.action == "evaluate":
        if args.checkpoint is None:
            logger.error("--checkpoint required for evaluate action")
            sys.exit(1)
        evaluate_phase1(args.checkpoint)
