from __future__ import annotations

import pytest

from src.training.dpo_data import DPOPair, DPOPairConfig, DPOPairGenerator, pairs_to_hf_dict
from src.training.model_utils import ModelConfig
from src.training.reward_model import RewardModelConfig


class TestRewardModelConfig:
    def test_defaults(self) -> None:
        config = RewardModelConfig()
        assert config.data.curriculum_phase == 2
        assert config.data.n_pairs == 20000
        assert config.learning_rate == 1e-5
        assert config.num_epochs == 2

    def test_custom_values(self) -> None:
        config = RewardModelConfig(
            learning_rate=5e-6,
            num_epochs=4,
            output_dir="custom_dir",
        )
        assert config.learning_rate == 5e-6
        assert config.num_epochs == 4
        assert config.output_dir == "custom_dir"

    def test_model_config_embedded(self) -> None:
        model_cfg = ModelConfig(model_name="test-model", lora_rank=8)
        config = RewardModelConfig(model=model_cfg)
        assert config.model.model_name == "test-model"
        assert config.model.lora_rank == 8


class TestRewardDataPreparation:
    def test_phase2_data_for_reward(self) -> None:
        config = DPOPairConfig(
            n_pairs=10, n_game_states=5, curriculum_phase=2, seed=42,
        )
        gen = DPOPairGenerator(config)
        pairs = gen.generate_dataset()
        d = pairs_to_hf_dict(pairs)

        assert len(d["prompt"]) == len(pairs)
        for prompt in d["prompt"]:
            assert "preference profile" in prompt

    def test_reward_data_has_chosen_rejected(self) -> None:
        config = DPOPairConfig(
            n_pairs=5, n_game_states=3, curriculum_phase=2, seed=42,
        )
        gen = DPOPairGenerator(config)
        pairs = gen.generate_dataset()
        d = pairs_to_hf_dict(pairs)

        assert len(d["chosen"]) == len(pairs)
        assert len(d["rejected"]) == len(pairs)
        for i in range(len(pairs)):
            assert d["chosen"][i] != d["rejected"][i]


class TestModelConfig:
    def test_defaults(self) -> None:
        config = ModelConfig()
        assert config.model_name == "mistralai/Mistral-7B-Instruct-v0.3"
        assert config.quantization_bits == 4
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert "q_proj" in config.lora_target_modules

    def test_custom_target_modules(self) -> None:
        config = ModelConfig(lora_target_modules=["q_proj", "v_proj"])
        assert config.lora_target_modules == ["q_proj", "v_proj"]

    def test_default_target_modules_on_none(self) -> None:
        config = ModelConfig(lora_target_modules=None)
        assert len(config.lora_target_modules) == 4
