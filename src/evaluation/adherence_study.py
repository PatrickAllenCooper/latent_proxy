"""Adherence study: does a DPO-tuned model act on a stated preference profile?

Decomposes the "elicit then adhere" question into two conditions instead of
conflating them (contrast with dpo_study.py's LLM condition, which has the
model run its own elicitation dialogue and never sees an explicit profile):

- theta_mode="true": the prompt states the user's *true* profile, in exactly
  the format Phase 2 was trained on (``build_prompt`` with a ``UserType``).
  Tests whether fine-tuning taught the model to follow an explicit profile
  at all, in-distribution for its own training.
- theta_mode="elicited": the profile fed into the prompt is instead the
  posterior mean produced by the existing analytical active-questioning loop
  (``ElicitationLoop``), and the generated allocation is still scored against
  the *true* optimum. Tests whether adherence holds up when the stated
  profile is itself a noisy estimate rather than ground truth -- the
  realistic deployment case.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop
from src.agents.llm_elicitation import LLMElicitationConfig, _get_channel_names
from src.agents.text_backends import LocalHFGenerator, TextGenerator
from src.evaluation.dpo_study import _rec_parse_failed
from src.environments.base import BaseEnvironment
from src.environments.game_variants import create_variant_a
from src.evaluation.alignment_metrics import evaluate_model_outputs
from src.evaluation.statistical_analysis import HypothesisTestResult, run_test_h1_within_domain
from src.training.serialization import build_prompt
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler, UserType

logger = logging.getLogger(__name__)

ADHERENCE_ENV_FACTORIES: dict[str, Callable[[], BaseEnvironment]] = {
    "game": create_variant_a,
}

THETA_MODES = ("true", "elicited")


@dataclass
class AdherenceStudyConfig:
    n_users: int = 20
    environment: str = "game"
    base_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    phase1_checkpoint: str | None = None
    phase2_checkpoint: str | None = None
    conditions: list[str] = field(default_factory=lambda: ["base", "dpo_phase1", "dpo_phase2"])
    analytical_elicitation: ElicitationConfig = field(default_factory=ElicitationConfig)
    # _generate_text (src/agents/llm_elicitation.py) truncates the INPUT
    # prompt to max_new_tokens*2 tokens -- confirmed by direct measurement
    # that this study's prompts (game state + full preference profile +
    # format instruction) run ~300-400 tokens, so anything below ~200 here
    # silently truncates the prompt (including the format instruction this
    # module appends) before the model ever sees it. 300 gives a ~600-token
    # input budget with comfortable margin, and enough generation room for
    # the model's reasoning-before-answering tendency to still reach the
    # formatted allocation.
    max_new_tokens: int = 300
    generation_temperature: float = 0.3
    seed: int = 9001


@dataclass
class AdherenceConditionResult:
    condition: str
    theta_mode: str
    alignment_scores: list[float] = field(default_factory=list)
    violation_rates: list[float] = field(default_factory=list)
    # Fraction of generations the parser couldn't extract a real allocation
    # from (fell back to a uniform default) -- see _rec_parse_failed. High
    # values mean alignment_scores are dominated by the zero-variance
    # fallback case, not genuine measurements.
    parse_failure_rate: float = 0.0
    mean_alignment: float = 0.0
    mean_violation: float = 0.0

    def __post_init__(self) -> None:
        if self.alignment_scores:
            self.mean_alignment = float(np.mean(self.alignment_scores))
        if self.violation_rates:
            self.mean_violation = float(np.mean(self.violation_rates))


def _append_format_instruction(prompt: str, channel_names: list[str]) -> str:
    """Append an explicit output-format instruction, mirroring
    llm_elicitation.RECOMMEND_TEMPLATE's established pattern for getting a
    parseable response out of free generation.

    Confirmed necessary by direct inspection of raw generations: DPO training
    rewards preferring one well-formatted completion over another, but that's
    a comparative signal between two *given* completions -- it doesn't by
    itself teach the model to always lead with that format under free
    generation. Without this instruction, Qwen2.5-1.5B-Instruct's base
    "reason through it first" tendency dominates and the response never
    reaches parseable numbers within a normal token budget, silently falling
    back to a uniform allocation on every single call (a zero-variance
    alignment score of exactly 0.0, not a genuine measurement).
    """
    channel_tpl = "\n  ".join(f"{n}: __%" for n in channel_names)
    return (
        f"{prompt}\n\n"
        f"Format your response EXACTLY as:\n"
        f"Recommended allocation:\n"
        f"  {channel_tpl}"
    )


def _theta_from_inferred(inferred: dict[str, float]) -> UserType:
    """Convert a posterior-mean dict into a validated UserType.

    The posterior lives in the same constrained space as UserType, so this
    should not clip in practice -- but crossing from a numeric estimate into
    a strictly-validated dataclass is a real boundary, so guard it rather
    than let one edge-case posterior mean crash an entire run.
    """
    gamma = float(np.clip(inferred.get("gamma", 0.5), 1e-6, 1.0))
    alpha = max(float(inferred.get("alpha", 1.0)), 0.0)
    lambda_ = max(float(inferred.get("lambda_", 1.0)), 1.0)
    return UserType(gamma=gamma, alpha=alpha, lambda_=lambda_)


def _run_adherence_condition(
    env_factory: Callable[[], BaseEnvironment],
    model: Any,
    tokenizer: Any,
    condition_name: str,
    theta_mode: str,
    n_users: int,
    generation_config: LLMElicitationConfig,
    analytical_elicitation: ElicitationConfig,
    seed: int,
    generator: TextGenerator | None = None,
) -> AdherenceConditionResult:
    """Run one (condition, theta_mode) cell.

    ``generator`` is injectable so tests can exercise this without loading a
    real model (mirroring dpo_study.py's ``_run_llm_condition``); production
    callers leave it None and get the real local-GPU generation path.
    """
    if theta_mode not in THETA_MODES:
        raise ValueError(f"Unknown theta_mode: {theta_mode!r} (expected one of {THETA_MODES})")

    sampler = SyntheticUserSampler(seed=seed)
    if generator is None:
        generator = LocalHFGenerator(model, tokenizer, config=generation_config)
    align_scores: list[float] = []
    violations: list[float] = []
    parse_fails: list[bool] = []

    for i in range(n_users):
        theta_true = sampler.sample()
        env = env_factory()
        obs, _ = env.reset(seed=seed + i)
        channel_names = _get_channel_names(env)

        if theta_mode == "elicited":
            user = SyntheticUser(
                theta_true,
                temperature=analytical_elicitation.temperature,
                seed=seed + i + 500,
            )
            loop = ElicitationLoop(analytical_elicitation)
            elicit_res = loop.run(env, user, query_type="active")
            profile_theta = _theta_from_inferred(elicit_res.inferred_theta)
        else:
            profile_theta = theta_true

        prompt = _append_format_instruction(
            build_prompt(obs, env, user_type=profile_theta), channel_names,
        )
        response = generator.generate(prompt)
        parse_fails.append(_rec_parse_failed(response, channel_names))

        result = evaluate_model_outputs([response], env, [theta_true], channel_names)
        align_scores.append(result["alignment_score"])
        violations.append(result["quality_floor_violation_rate"])

        logger.info(
            "%s/%s user %d/%d: align=%.3f viol=%.0f parse_fail=%s",
            condition_name, theta_mode, i + 1, n_users,
            align_scores[-1], violations[-1], parse_fails[-1],
        )

    return AdherenceConditionResult(
        condition=condition_name,
        theta_mode=theta_mode,
        alignment_scores=align_scores,
        violation_rates=violations,
        parse_failure_rate=float(np.mean(parse_fails)) if parse_fails else 0.0,
    )


def run_adherence_study(
    config: AdherenceStudyConfig,
) -> dict[str, dict[str, AdherenceConditionResult]]:
    """Run every configured condition x theta_mode. Returns {condition: {theta_mode: result}}."""
    from src.training.model_utils import load_model_with_optional_checkpoint

    if config.environment not in ADHERENCE_ENV_FACTORIES:
        raise ValueError(
            f"Unknown environment: {config.environment!r} "
            f"(expected one of {list(ADHERENCE_ENV_FACTORIES)})"
        )
    factory = ADHERENCE_ENV_FACTORIES[config.environment]

    gen_cfg = LLMElicitationConfig(
        max_new_tokens=config.max_new_tokens,
        temperature=config.generation_temperature,
    )
    checkpoints = {
        "base": None,
        "dpo_phase1": config.phase1_checkpoint,
        "dpo_phase2": config.phase2_checkpoint,
    }

    results: dict[str, dict[str, AdherenceConditionResult]] = {}
    for condition in config.conditions:
        if condition not in checkpoints:
            raise ValueError(
                f"Unknown condition: {condition!r} (expected one of {list(checkpoints)})"
            )
        ckpt = checkpoints[condition]
        if condition != "base" and not ckpt:
            raise ValueError(f"condition {condition!r} requires a checkpoint path")

        logger.info("Loading model for condition %s...", condition)
        model, tokenizer = load_model_with_optional_checkpoint(config.base_model_path, ckpt)

        results[condition] = {}
        for theta_mode in THETA_MODES:
            results[condition][theta_mode] = _run_adherence_condition(
                factory, model, tokenizer, condition, theta_mode,
                config.n_users, gen_cfg, config.analytical_elicitation, config.seed,
            )

        del model, tokenizer

    return results


def compare_conditions(
    results: dict[str, dict[str, AdherenceConditionResult]],
    condition_a: str,
    condition_b: str,
    theta_mode: str,
) -> HypothesisTestResult:
    """One-sided paired test: does condition_a score higher than condition_b?

    Same theta_mode for both sides (e.g. dpo_phase2 vs. base, both "elicited").
    """
    return run_test_h1_within_domain(
        results[condition_a][theta_mode].alignment_scores,
        results[condition_b][theta_mode].alignment_scores,
        hypothesis_label=f"{condition_a}_vs_{condition_b}_{theta_mode}",
    )
