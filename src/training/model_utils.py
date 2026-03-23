from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for base model loading and LoRA setup."""

    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization_bits: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] | None = None
    max_length: int = 1024
    padding_side: str = "left"

    def __post_init__(self) -> None:
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


def get_bnb_config(bits: int = 4) -> Any:
    """Create a BitsAndBytes quantization config."""
    import torch
    from transformers import BitsAndBytesConfig

    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None


def get_lora_config(model_config: ModelConfig | None = None) -> Any:
    """Create a PEFT LoRA configuration."""
    from peft import LoraConfig, TaskType

    config = model_config or ModelConfig()
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_base_model(
    model_config: ModelConfig | None = None,
) -> Any:
    """Load a base model with optional quantization.

    Returns the model ready for LoRA adapter attachment.
    """
    from transformers import AutoModelForCausalLM

    config = model_config or ModelConfig()
    bnb_config = get_bnb_config(config.quantization_bits)

    load_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": config.model_name,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config

    logger.info("Loading model: %s (quantization: %d-bit)", config.model_name, config.quantization_bits)
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    model.config.use_cache = False

    return model


def load_tokenizer(model_config: ModelConfig | None = None) -> Any:
    """Load and configure the tokenizer for the base model."""
    from transformers import AutoTokenizer

    config = model_config or ModelConfig()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = config.padding_side

    return tokenizer


def prepare_model_for_training(model: Any) -> Any:
    """Prepare a quantized model for LoRA training."""
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model: Any, model_config: ModelConfig | None = None) -> Any:
    """Apply LoRA adapters to the model."""
    from peft import get_peft_model

    lora_config = get_lora_config(model_config)
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA applied: %d trainable / %d total parameters (%.2f%%)",
        trainable, total, 100.0 * trainable / total,
    )
    return model
