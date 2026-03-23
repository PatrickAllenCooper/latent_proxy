from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.training.dpo_data import (
    DPOPairConfig,
    DPOPairGenerator,
    pairs_to_hf_dict,
)
from src.training.model_utils import (
    ModelConfig,
    apply_lora,
    load_base_model,
    load_tokenizer,
    prepare_model_for_training,
)

logger = logging.getLogger(__name__)


@dataclass
class DPOTrainingConfig:
    """Configuration for conditional DPO training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DPOPairConfig = field(default_factory=DPOPairConfig)
    curriculum_phase: int = 1
    beta: float = 0.1
    learning_rate: float = 5e-5
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_length: int = 1024
    max_prompt_length: int = 768
    output_dir: str = "outputs/dpo"
    logging_steps: int = 10
    save_steps: int = 500
    use_wandb: bool = True
    wandb_project: str = "latent-proxy"


class ConditionalDPOTrainer:
    """Wraps TRL's DPOTrainer with curriculum-aware training.

    Phase 1 (quality floor): Trains on quality-only pairs where the chosen
    response has a higher Sharpe ratio and the rejected may violate quality
    constraints. No user profile in prompts.

    Phase 2 (basic alignment): Trains on type-conditioned pairs where the
    chosen response is optimal for the sampled user type and the rejected
    is optimal for a contrasting type. User profiles included in prompts.
    """

    def __init__(self, config: DPOTrainingConfig | None = None) -> None:
        self.config = config or DPOTrainingConfig()
        self.config.data.curriculum_phase = self.config.curriculum_phase
        self._model = None
        self._tokenizer = None
        self._trainer = None

    def prepare(self) -> None:
        """Load model, tokenizer, and generate training data."""
        self._model = load_base_model(self.config.model)
        self._model = prepare_model_for_training(self._model)
        self._model = apply_lora(self._model, self.config.model)
        self._tokenizer = load_tokenizer(self.config.model)

        logger.info(
            "Generating DPO pairs for phase %d...", self.config.curriculum_phase
        )
        generator = DPOPairGenerator(self.config.data)
        pairs = generator.generate_dataset()

        from datasets import Dataset

        self._dataset = Dataset.from_dict(pairs_to_hf_dict(pairs))
        logger.info("Dataset ready: %d pairs", len(self._dataset))

    def train(self) -> None:
        """Run DPO training."""
        if self._model is None or self._tokenizer is None:
            self.prepare()

        from transformers import TrainingArguments
        from trl import DPOTrainer

        phase_dir = (
            f"{self.config.output_dir}/phase{self.config.curriculum_phase}"
        )
        Path(phase_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=phase_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            bf16=True,
            remove_unused_columns=False,
            report_to="wandb" if self.config.use_wandb else "none",
        )

        self._trainer = DPOTrainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset,
            processing_class=self._tokenizer,
            beta=self.config.beta,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )

        logger.info(
            "Starting DPO training (phase %d, beta=%.2f, %d epochs)...",
            self.config.curriculum_phase,
            self.config.beta,
            self.config.num_epochs,
        )
        self._trainer.train()

        self._model.save_pretrained(f"{phase_dir}/final")
        self._tokenizer.save_pretrained(f"{phase_dir}/final")
        logger.info("Phase %d training complete. Saved to %s/final", self.config.curriculum_phase, phase_dir)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a previously saved LoRA checkpoint for continued training."""
        from peft import PeftModel

        if self._model is None:
            self._model = load_base_model(self.config.model)
            self._model = prepare_model_for_training(self._model)

        self._model = PeftModel.from_pretrained(self._model, checkpoint_path)
        self._tokenizer = load_tokenizer(self.config.model)
        logger.info("Loaded checkpoint from %s", checkpoint_path)
