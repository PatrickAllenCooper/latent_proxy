"""Gamma-only DPO study: train and evaluate on discount factor only."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.agents.llm_elicitation import LLMElicitationConfig, LLMElicitationLoop
from src.environments.resource_game import ResourceStrategyGame
from src.evaluation.alignment_metrics import compute_alignment_score
from src.training.dpo_data import DPOPairConfig, pairs_to_hf_dict
from src.training.dpo_trainer import ConditionalDPOTrainer, DPOTrainingConfig
from src.training.gamma_dpo_data import FIXED_ALPHA, FIXED_LAMBDA, GammaDPOPairGenerator
from src.training.model_utils import ModelConfig
from src.training.synthetic_users import SyntheticUser, UserType
from src.utils.visualization import save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def generate_gamma_data(n_pairs: int, phase: int, seed: int, output_path: str) -> None:
    cfg = DPOPairConfig(n_pairs=n_pairs, curriculum_phase=phase, seed=seed)
    gen = GammaDPOPairGenerator(cfg)
    pairs = gen.generate_dataset()
    from src.training.dpo_data import save_pairs
    save_pairs(pairs, output_path)
    logger.info("Saved %d gamma pairs (phase %d) to %s", len(pairs), phase, output_path)


def train_gamma_dpo(
    model_name: str,
    phase: int,
    n_pairs: int,
    num_epochs: int,
    output_dir: str,
    seed: int,
) -> None:
    mcfg = ModelConfig(model_name=model_name)
    dcfg = DPOPairConfig(n_pairs=n_pairs, curriculum_phase=phase, seed=seed)
    tcfg = DPOTrainingConfig(
        model=mcfg,
        data=dcfg,
        curriculum_phase=phase,
        num_epochs=num_epochs,
        output_dir=output_dir,
    )

    logger.info("Generating gamma-only DPO pairs (phase %d)...", phase)
    gen = GammaDPOPairGenerator(dcfg)
    pairs = gen.generate_dataset()
    from datasets import Dataset
    dataset = Dataset.from_dict(pairs_to_hf_dict(pairs))
    logger.info("Dataset: %d pairs", len(dataset))

    from src.training.model_utils import (
        apply_lora,
        load_base_model,
        load_tokenizer,
        prepare_model_for_training,
    )

    model = load_base_model(mcfg)
    model = prepare_model_for_training(model)
    model = apply_lora(model, mcfg)
    tokenizer = load_tokenizer(mcfg)

    trainer = ConditionalDPOTrainer(tcfg)
    trainer._model = model
    trainer._tokenizer = tokenizer
    trainer._dataset = dataset
    trainer.train()
    logger.info("Phase %d training complete.", phase)


def run_gamma_eval(
    model_name: str,
    checkpoint_path: str | None,
    condition_name: str,
    n_users: int,
    max_rounds: int,
    seed: int,
) -> dict:
    """Run LLM elicitation on gamma-only users and measure alignment."""
    from src.training.model_utils import load_base_model, load_tokenizer

    mcfg = ModelConfig(model_name=model_name)
    model = load_base_model(mcfg)
    tokenizer = load_tokenizer(mcfg)

    if checkpoint_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        logger.info("Loaded checkpoint: %s", checkpoint_path)

    model.eval()

    llm_cfg = LLMElicitationConfig(max_rounds=max_rounds, max_new_tokens=200, temperature=0.3)
    loop = LLMElicitationLoop(model, tokenizer, config=llm_cfg)

    env = ResourceStrategyGame()
    rng = np.random.default_rng(seed)
    align_scores: list[float] = []
    gamma_errors: list[float] = []

    for i in range(n_users):
        gamma = float(rng.beta(2, 2))
        gamma = max(0.05, min(gamma, 0.99))
        ut = UserType(gamma=gamma, alpha=FIXED_ALPHA, lambda_=FIXED_LAMBDA)
        user = SyntheticUser(ut, temperature=0.1, seed=seed + i + 100)

        res = loop.run(env, user, seed=seed + i)

        theta_true = {"gamma": gamma, "alpha": FIXED_ALPHA, "lambda_": FIXED_LAMBDA}
        opt_true = env.get_optimal_action(theta_true)
        align = compute_alignment_score([res.recommendation], [opt_true])
        align_scores.append(align)

        safe_weight = float(res.recommendation[0])
        inferred_gamma = safe_weight * 1.2
        inferred_gamma = max(0.05, min(inferred_gamma, 0.99))
        gamma_errors.append(abs(inferred_gamma - gamma))

        logger.info(
            "%s user %d/%d: gamma=%.2f align=%.3f",
            condition_name, i + 1, n_users, gamma, align,
        )

    del model, tokenizer
    import torch
    torch.cuda.empty_cache()

    return {
        "condition": condition_name,
        "mean_alignment": float(np.mean(align_scores)),
        "alignment_scores": align_scores,
        "mean_gamma_error": float(np.mean(gamma_errors)),
        "gamma_errors": gamma_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gamma-only DPO study")
    parser.add_argument("--action", choices=["generate", "train-p1", "train-p2", "eval", "full"],
                        default="full")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--n-users", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/gamma_study")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.action in ("generate", "full"):
        generate_gamma_data(args.n_pairs, 1, args.seed, str(out / "data_p1"))
        generate_gamma_data(args.n_pairs, 2, args.seed + 1, str(out / "data_p2"))

    if args.action in ("train-p1", "full"):
        train_gamma_dpo(
            args.model_name, phase=1, n_pairs=args.n_pairs,
            num_epochs=args.num_epochs, output_dir=str(out / "dpo_p1"),
            seed=args.seed,
        )

    if args.action in ("train-p2", "full"):
        train_gamma_dpo(
            args.model_name, phase=2, n_pairs=args.n_pairs,
            num_epochs=args.num_epochs, output_dir=str(out / "dpo_p2"),
            seed=args.seed + 1,
        )

    if args.action in ("eval", "full"):
        p1_ckpt = str(out / "dpo_p1" / "phase1" / "final")
        p2_ckpt = str(out / "dpo_p2" / "phase2" / "final")

        results: dict = {}

        logger.info("=== Evaluating base model ===")
        results["base"] = run_gamma_eval(
            args.model_name, None, "base",
            args.n_users, args.max_rounds, args.seed + 10,
        )

        if Path(p1_ckpt).exists():
            logger.info("=== Evaluating DPO Phase 1 ===")
            results["dpo_phase1"] = run_gamma_eval(
                args.model_name, p1_ckpt, "dpo_phase1",
                args.n_users, args.max_rounds, args.seed + 10,
            )

        if Path(p2_ckpt).exists():
            logger.info("=== Evaluating DPO Phase 2 ===")
            results["dpo_phase2"] = run_gamma_eval(
                args.model_name, p2_ckpt, "dpo_phase2",
                args.n_users, args.max_rounds, args.seed + 10,
            )

        summary = {
            cond: {"mean_alignment": r["mean_alignment"], "mean_gamma_error": r["mean_gamma_error"]}
            for cond, r in results.items()
        }
        save_results(results, out / "gamma_study_results.json")
        logger.info("Results:\n%s", json.dumps(summary, indent=2))
        logger.info("Wrote %s", out / "gamma_study_results.json")


if __name__ == "__main__":
    main()
