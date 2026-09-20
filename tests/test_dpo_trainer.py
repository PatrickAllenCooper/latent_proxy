"""Tests for ConditionalDPOTrainer's reference-model construction.

Regression coverage for a real bug: trl's automatic PEFT reference handling
(cloning the current adapter into a frozen "ref" adapter on the SAME
PeftModel) produced a reference that never diverged from the policy when the
model was loaded via PeftModel.from_pretrained for continued training
(confirmed empirically on a real Phase 2 run: rewards/accuracies/margins
stayed exactly zero for 2700+ real gradient steps). _build_ref_model sidesteps
this with an explicit, independently loaded and frozen reference model.

Mocked throughout -- no real model download/load, matching this project's
existing pattern for testing model-loading orchestration without a GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from src.training.dpo_trainer import ConditionalDPOTrainer, DPOTrainingConfig


def test_build_ref_model_returns_none_without_a_loaded_checkpoint():
    """Fresh (Phase 1 style) training keeps trl's own automatic handling."""
    trainer = ConditionalDPOTrainer(DPOTrainingConfig())
    assert trainer._loaded_checkpoint_path is None
    assert trainer._build_ref_model() is None


@patch("peft.PeftModel.from_pretrained")
@patch("src.training.dpo_trainer.load_base_model")
def test_build_ref_model_loads_independent_frozen_copy(mock_load_base, mock_from_pretrained):
    base = MagicMock(name="fresh_base_model")
    mock_load_base.return_value = base

    real_param = torch.nn.Parameter(torch.zeros(3))
    assert real_param.requires_grad is True  # sanity: starts trainable

    ref_model = MagicMock(name="loaded_ref_model")
    ref_model.parameters.return_value = [real_param]
    mock_from_pretrained.return_value = ref_model

    trainer = ConditionalDPOTrainer(DPOTrainingConfig())
    trainer._loaded_checkpoint_path = "outputs/dpo/phase1/phase1/final"

    result = trainer._build_ref_model()

    # A fresh base load, not a reuse of self._model -- independence is the
    # whole point of the fix.
    mock_load_base.assert_called_once_with(trainer.config.model)
    mock_from_pretrained.assert_called_once_with(base, "outputs/dpo/phase1/phase1/final")
    ref_model.eval.assert_called_once()
    assert real_param.requires_grad is False
    assert result is ref_model


def test_load_checkpoint_records_path_for_build_ref_model():
    trainer = ConditionalDPOTrainer(DPOTrainingConfig())
    with patch("src.training.dpo_trainer.load_base_model", return_value=MagicMock()), \
         patch("src.training.dpo_trainer.prepare_model_for_training", side_effect=lambda m: m), \
         patch("src.training.dpo_trainer.load_tokenizer", return_value=MagicMock()), \
         patch("peft.PeftModel.from_pretrained", return_value=MagicMock()):
        trainer.load_checkpoint("some/checkpoint/path")

    assert trainer._loaded_checkpoint_path == "some/checkpoint/path"
