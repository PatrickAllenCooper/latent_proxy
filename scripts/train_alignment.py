"""Phase 2 training: Basic alignment via type-conditioned DPO.

Loads the Phase 1 checkpoint and continues training with user-type-conditioned
DPO pairs. Prompts now include the user preference profile. Corresponds to
curriculum Phase 2 (beta=0.3) in README Section 5.3.
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


def generate_phase2_data(
    n_pairs: int = 20000,
    seed: int = 42,
    output_path: str | None = None,
) -> list:
    """Generate Phase 2 DPO pairs (type-conditioned)."""
    config = DPOPairConfig(
        n_pairs=n_pairs,
        curriculum_phase=2,
        seed=seed,
    )
    generator = DPOPairGenerator(config)
    pairs = generator.generate_dataset()

    if output_path:
        from src.training.dpo_data import save_pairs
        save_pairs(pairs, output_path)

    logger.info("Phase 2 data: %d pairs generated", len(pairs))
    return pairs


def run_phase2_training(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    phase1_checkpoint: str | None = None,
    n_pairs: int = 20000,
    num_epochs: int = 3,
    output_dir: str = "outputs/dpo/phase2",
) -> None:
    """Full Phase 2 pipeline: load Phase 1, generate data, train, evaluate."""
    model_config = ModelConfig(model_name=model_name)
    data_config = DPOPairConfig(n_pairs=n_pairs, curriculum_phase=2)
    training_config = DPOTrainingConfig(
        model=model_config,
        data=data_config,
        curriculum_phase=2,
        beta=0.1,
        num_epochs=num_epochs,
        output_dir=output_dir,
    )

    trainer = ConditionalDPOTrainer(training_config)

    if phase1_checkpoint:
        trainer.load_checkpoint(phase1_checkpoint)
        logger.info("Loaded Phase 1 checkpoint: %s", phase1_checkpoint)

    trainer.train()
    logger.info("Phase 2 training complete.")


def evaluate_phase2(
    checkpoint_path: str,
    n_episodes: int = 100,
) -> dict:
    """Evaluate a Phase 2 checkpoint for alignment on extreme user types."""
    env = ResourceStrategyGame()
    env.reset(seed=0)
    sampler = SyntheticUserSampler(seed=0)

    extreme_types = sampler.sample_extreme_types()
    user_types = list(extreme_types.values())
    type_names = list(extreme_types.keys())

    optimal_actions = []
    for ut in user_types:
        theta = {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
        optimal_actions.append(env.get_optimal_action(theta))

    results = evaluate_actions(optimal_actions, env, user_types)
    logger.info("Phase 2 evaluation (optimal baseline):")
    logger.info("  Alignment score: %.4f", results["alignment_score"])
    logger.info("  Quality floor violation rate: %.4f", results["quality_floor_violation_rate"])
    logger.info("  Mean quality score: %.4f", results["mean_quality_score"])

    for name, ut in zip(type_names, user_types):
        theta = {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
        action = env.get_optimal_action(theta)
        logger.info("  %s: %s", name, [f"{x:.2f}" for x in action])

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: Alignment training")
    parser.add_argument("--action", choices=["generate", "train", "evaluate"], default="generate")
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--phase1-checkpoint", default=None)
    parser.add_argument("--n-pairs", type=int, default=20000)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/dpo/phase2")
    parser.add_argument("--data-path", default="data/phase2")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if args.action == "generate":
        generate_phase2_data(args.n_pairs, output_path=args.data_path)
    elif args.action == "train":
        run_phase2_training(
            model_name=args.model_name,
            phase1_checkpoint=args.phase1_checkpoint,
            n_pairs=args.n_pairs,
            num_epochs=args.num_epochs,
            output_dir=args.output_dir,
        )
    elif args.action == "evaluate":
        if args.checkpoint is None:
            logger.error("--checkpoint required for evaluate action")
            sys.exit(1)
        evaluate_phase2(args.checkpoint)
