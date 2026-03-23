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
    load_base_model,
    load_tokenizer,
    prepare_model_for_training,
    get_lora_config,
)

logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """Configuration for type-conditioned reward model training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DPOPairConfig = field(default_factory=lambda: DPOPairConfig(
        curriculum_phase=2, n_pairs=20000,
    ))
    learning_rate: float = 1e-5
    num_epochs: int = 2
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    output_dir: str = "outputs/reward_model"
    logging_steps: int = 10
    save_steps: int = 500
    use_wandb: bool = True
    wandb_project: str = "latent-proxy"


def prepare_reward_dataset(
    config: RewardModelConfig | None = None,
) -> Any:
    """Generate the preference dataset for reward model training.

    Uses the same DPO pair format: (prompt, chosen, rejected). The reward
    model learns to assign higher scalar rewards to chosen over rejected.
    User profile is included in the prompt (Phase 2 data) to make the
    reward type-conditioned.
    """
    config = config or RewardModelConfig()
    generator = DPOPairGenerator(config.data)
    pairs = generator.generate_dataset()

    from datasets import Dataset

    dataset = Dataset.from_dict(pairs_to_hf_dict(pairs))
    logger.info("Reward model dataset: %d pairs", len(dataset))
    return dataset


class TypeConditionedRewardModel:
    """Type-conditioned reward model r(y | x, theta_user) -> R.

    Trains on the same preference data as DPO. The prompt includes the
    user profile, which makes the learned reward function implicitly
    conditioned on user type. Used for:
    - Online PPO training (Milestone 3)
    - Evaluation: scoring model outputs without environment simulation
    """

    def __init__(self, config: RewardModelConfig | None = None) -> None:
        self.config = config or RewardModelConfig()
        self._model = None
        self._tokenizer = None
        self._trainer = None

    def prepare(self) -> None:
        """Load model, tokenizer, and generate training data."""
        from transformers import AutoModelForSequenceClassification

        bnb_config = None
        if self.config.model.quantization_bits in (4, 8):
            from src.training.model_utils import get_bnb_config
            bnb_config = get_bnb_config(self.config.model.quantization_bits)

        load_kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.config.model.model_name,
            "num_labels": 1,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config

        self._model = AutoModelForSequenceClassification.from_pretrained(
            **load_kwargs
        )
        self._model = prepare_model_for_training(self._model)

        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            r=self.config.model.lora_rank,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            target_modules=self.config.model.lora_target_modules,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        self._model = get_peft_model(self._model, lora_config)

        self._tokenizer = load_tokenizer(self.config.model)
        self._dataset = prepare_reward_dataset(self.config)

    def train(self) -> None:
        """Run reward model training."""
        if self._model is None or self._tokenizer is None:
            self.prepare()

        from transformers import TrainingArguments
        from trl import RewardTrainer

        output_dir = self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            bf16=True,
            remove_unused_columns=False,
            report_to="wandb" if self.config.use_wandb else "none",
        )

        self._trainer = RewardTrainer(
            model=self._model,
            args=training_args,
            train_dataset=self._dataset,
            processing_class=self._tokenizer,
            max_length=self.config.max_length,
        )

        logger.info("Starting reward model training (%d epochs)...", self.config.num_epochs)
        self._trainer.train()

        self._model.save_pretrained(f"{output_dir}/final")
        self._tokenizer.save_pretrained(f"{output_dir}/final")
        logger.info("Reward model training complete. Saved to %s/final", output_dir)
