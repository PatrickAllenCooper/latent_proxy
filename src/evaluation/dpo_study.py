"""DPO elicitation study: compare base LLM vs DPO-trained vs analytical baseline.

Runs the full LLM-in-the-loop elicitation across environments and conditions,
computes alignment, quality floor violations, and convergence curves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from src.agents.elicitation_loop import ElicitationConfig, ElicitationLoop
from src.agents.llm_elicitation import LLMElicitationConfig, LLMElicitationLoop, LLMElicitationResult
from src.environments.base import BaseEnvironment
from src.environments.game_variants import create_variant_a
from src.environments.stock_backtest import StockBacktestConfig, StockBacktestEnv
from src.environments.supply_chain import SupplyChainConfig, SupplyChainEnv
from src.evaluation.alignment_metrics import compute_alignment_score
from src.evaluation.statistical_analysis import HypothesisTestResult, run_test_h1_within_domain
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler

logger = logging.getLogger(__name__)

STUDY_ENV_FACTORIES: dict[str, Callable[[], BaseEnvironment]] = {
    "game": create_variant_a,
    "stock": lambda: StockBacktestEnv(config=StockBacktestConfig()),
    "supply_chain": lambda: SupplyChainEnv(config=SupplyChainConfig()),
}

# Conditions that run purely on CPU (no text generation involved).
NON_LLM_CONDITIONS = ("analytical", "random")
# Conditions that require a text-generation backend.
LLM_CONDITIONS = ("base", "dpo_phase1", "dpo_phase2")
ALL_CONDITIONS = NON_LLM_CONDITIONS + LLM_CONDITIONS


@dataclass
class DPOStudyConfig:
    n_users: int = 5
    max_rounds: int = 5
    environments: list[str] = field(default_factory=lambda: ["game", "stock", "supply_chain"])
    base_model_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    phase1_checkpoint: str | None = None
    phase2_checkpoint: str | None = None
    llm_config: LLMElicitationConfig = field(default_factory=LLMElicitationConfig)
    analytical_elicitation: ElicitationConfig = field(default_factory=ElicitationConfig)
    seed: int = 42
    # Text-generation backend for LLM conditions: "local" (transformers,
    # historical path), "azure" (Azure OpenAI API), or "stub" (offline).
    backend: str = "local"
    # Azure deployment name (azure backend only).
    deployment: str | None = None
    # JSONL path for raw API completions (azure backend only).
    generation_log: str | None = None
    # Which conditions to run; None = historical default (analytical + base,
    # plus dpo_phase1/dpo_phase2 when checkpoints are provided).
    conditions: list[str] | None = None


@dataclass
class ConditionResult:
    condition: str
    alignment_scores: list[float]
    violation_rates: list[float]
    per_round_alignments: list[list[float]]
    mean_alignment: float = 0.0
    mean_violation: float = 0.0

    def __post_init__(self) -> None:
        if self.alignment_scores:
            self.mean_alignment = float(np.mean(self.alignment_scores))
        if self.violation_rates:
            self.mean_violation = float(np.mean(self.violation_rates))


@dataclass
class DPOStudyResult:
    per_env: dict[str, dict[str, ConditionResult]]
    hypothesis_tests: dict[str, list[HypothesisTestResult]]
    config: dict[str, Any] | None = None


def _run_analytical_condition(
    env_factory: Callable[[], BaseEnvironment],
    n_users: int,
    elicitation: ElicitationConfig,
    max_rounds: int,
    seed: int,
    query_type: str = "active",
    condition_name: str = "analytical",
) -> ConditionResult:
    """Run a non-LLM baseline: particle filter with EIG ("active") or random queries."""
    from src.agents.preference_tracker import ConvergenceConfig

    sampler = SyntheticUserSampler(seed=seed)
    align_scores: list[float] = []
    violations: list[float] = []
    per_round: list[list[float]] = []

    conv = ConvergenceConfig(max_rounds=max_rounds)
    cfg = ElicitationConfig(
        posterior_type=elicitation.posterior_type,
        n_particles=elicitation.n_particles,
        max_rounds=max_rounds,
        n_scenarios_per_round=elicitation.n_scenarios_per_round,
        n_eig_samples=elicitation.n_eig_samples,
        temperature=elicitation.temperature,
        convergence=conv,
        seed=seed,
    )

    for i in range(n_users):
        ut = sampler.sample()
        theta_d = {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
        env = env_factory()
        env.reset(seed=seed + i)

        user = SyntheticUser(ut, temperature=cfg.temperature, seed=seed + i + 100)
        loop = ElicitationLoop(cfg)
        res = loop.run(env, user, query_type=query_type)

        opt_true = env.get_optimal_action(theta_d)
        opt_hat = env.get_optimal_action(res.inferred_theta)
        align_scores.append(compute_alignment_score([opt_hat], [opt_true]))

        passes, _ = env.check_quality_floor(opt_hat)
        violations.append(0.0 if passes else 1.0)
        per_round.append([])

    return ConditionResult(
        condition=condition_name,
        alignment_scores=align_scores,
        violation_rates=violations,
        per_round_alignments=per_round,
    )


def _run_llm_condition(
    env_factory: Callable[[], BaseEnvironment],
    model: Any,
    tokenizer: Any,
    condition_name: str,
    n_users: int,
    llm_config: LLMElicitationConfig,
    seed: int,
    generator: Any | None = None,
) -> ConditionResult:
    """Run a single LLM condition (base, phase1, or phase2)."""
    sampler = SyntheticUserSampler(seed=seed)
    align_scores: list[float] = []
    violations: list[float] = []
    per_round: list[list[float]] = []

    loop = LLMElicitationLoop(model, tokenizer, config=llm_config, generator=generator)

    for i in range(n_users):
        ut = sampler.sample()
        theta_d = {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
        env = env_factory()

        user = SyntheticUser(ut, temperature=0.1, seed=seed + i + 200)
        res = loop.run(env, user, seed=seed + i)

        opt_true = env.get_optimal_action(theta_d)
        align_scores.append(compute_alignment_score([res.recommendation], [opt_true]))

        passes, _ = env.check_quality_floor(res.recommendation)
        violations.append(0.0 if passes else 1.0)

        round_aligns: list[float] = []
        for rec in res.per_round_recommendations:
            round_aligns.append(compute_alignment_score([rec], [opt_true]))
        per_round.append(round_aligns)

        logger.info(
            "%s user %d/%d: align=%.3f viol=%.0f",
            condition_name, i + 1, n_users,
            align_scores[-1], violations[-1],
        )

    return ConditionResult(
        condition=condition_name,
        alignment_scores=align_scores,
        violation_rates=violations,
        per_round_alignments=per_round,
    )


def _load_model_and_tokenizer(
    model_path: str,
    checkpoint_path: str | None = None,
) -> tuple[Any, Any]:
    """Load base model with optional LoRA checkpoint."""
    from src.training.model_utils import ModelConfig, load_base_model, load_tokenizer

    cfg = ModelConfig(model_name=model_path)
    model = load_base_model(cfg)
    tokenizer = load_tokenizer(cfg)

    if checkpoint_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        logger.info("Loaded LoRA checkpoint from %s", checkpoint_path)

    model.eval()
    return model, tokenizer


def _resolve_conditions(config: DPOStudyConfig) -> list[str]:
    """Determine which conditions to run, preserving historical defaults."""
    if config.conditions is None:
        conditions = ["analytical", "base"]
        if config.phase1_checkpoint:
            conditions.append("dpo_phase1")
        if config.phase2_checkpoint:
            conditions.append("dpo_phase2")
        return conditions

    unknown = [c for c in config.conditions if c not in ALL_CONDITIONS]
    if unknown:
        raise ValueError(
            f"Unknown conditions: {unknown}. Valid conditions: {list(ALL_CONDITIONS)}"
        )
    return list(config.conditions)


def run_dpo_study(config: DPOStudyConfig) -> DPOStudyResult:
    """Execute the full DPO elicitation study."""
    per_env: dict[str, dict[str, ConditionResult]] = {}
    hypothesis_tests: dict[str, list[HypothesisTestResult]] = {}

    llm_cfg = config.llm_config
    llm_cfg.max_rounds = config.max_rounds

    conditions = _resolve_conditions(config)
    llm_conditions = [c for c in conditions if c in LLM_CONDITIONS]
    logger.info("Conditions: %s (backend=%s)", conditions, config.backend)

    base_model, base_tok = None, None
    p1_model, p1_tok = None, None
    p2_model, p2_tok = None, None
    generator = None

    if llm_conditions:
        if config.backend == "local":
            base_model, base_tok = _load_model_and_tokenizer(config.base_model_path)
            if "dpo_phase1" in llm_conditions:
                if not config.phase1_checkpoint:
                    raise ValueError("dpo_phase1 condition requires phase1_checkpoint")
                p1_model, p1_tok = _load_model_and_tokenizer(
                    config.base_model_path, config.phase1_checkpoint,
                )
            if "dpo_phase2" in llm_conditions:
                if not config.phase2_checkpoint:
                    raise ValueError("dpo_phase2 condition requires phase2_checkpoint")
                p2_model, p2_tok = _load_model_and_tokenizer(
                    config.base_model_path, config.phase2_checkpoint,
                )
        else:
            dpo_conds = [c for c in llm_conditions if c != "base"]
            if dpo_conds:
                raise ValueError(
                    f"Conditions {dpo_conds} need LoRA checkpoints and are only "
                    f"available with backend='local' (got backend={config.backend!r})"
                )
            from src.agents.text_backends import create_generator

            generator = create_generator(
                config.backend,
                deployment=config.deployment,
                log_path=config.generation_log,
                temperature=llm_cfg.temperature,
                max_tokens=llm_cfg.max_new_tokens,
                seed=config.seed,
            )

    for env_name in config.environments:
        if env_name not in STUDY_ENV_FACTORIES:
            logger.warning("Unknown environment: %s", env_name)
            continue

        factory = STUDY_ENV_FACTORIES[env_name]
        env_results: dict[str, ConditionResult] = {}

        logger.info("=== Environment: %s ===", env_name)

        for condition in conditions:
            logger.info("Running condition: %s...", condition)
            if condition == "analytical":
                env_results["analytical"] = _run_analytical_condition(
                    factory, config.n_users, config.analytical_elicitation,
                    config.max_rounds, seed=config.seed,
                )
            elif condition == "random":
                env_results["random"] = _run_analytical_condition(
                    factory, config.n_users, config.analytical_elicitation,
                    config.max_rounds, seed=config.seed,
                    query_type="random", condition_name="random",
                )
            elif condition == "base":
                env_results["base"] = _run_llm_condition(
                    factory, base_model, base_tok, "base",
                    config.n_users, llm_cfg, seed=config.seed,
                    generator=generator,
                )
            elif condition == "dpo_phase1":
                env_results["dpo_phase1"] = _run_llm_condition(
                    factory, p1_model, p1_tok, "dpo_phase1",
                    config.n_users, llm_cfg, seed=config.seed,
                )
            elif condition == "dpo_phase2":
                env_results["dpo_phase2"] = _run_llm_condition(
                    factory, p2_model, p2_tok, "dpo_phase2",
                    config.n_users, llm_cfg, seed=config.seed,
                )

        per_env[env_name] = env_results

        if "dpo_phase2" in env_results and "base" in env_results:
            h_test = run_test_h1_within_domain(
                env_results["dpo_phase2"].alignment_scores,
                env_results["base"].alignment_scores,
                hypothesis_label=f"DPO_vs_base_{env_name}",
            )
            hypothesis_tests.setdefault(env_name, []).append(h_test)
            logger.info(
                "%s DPO vs base: p=%.4f, effect=%.3f, %s",
                env_name, h_test.p_value, h_test.effect_size, h_test.conclusion,
            )

    del base_model, base_tok
    if p1_model is not None:
        del p1_model, p1_tok
    if p2_model is not None:
        del p2_model, p2_tok

    return DPOStudyResult(
        per_env=per_env,
        hypothesis_tests=hypothesis_tests,
        config={
            "n_users": config.n_users,
            "max_rounds": config.max_rounds,
            "environments": config.environments,
            "conditions": conditions,
            "backend": config.backend,
            "deployment": config.deployment,
            "base_model": config.base_model_path,
            "phase1_checkpoint": config.phase1_checkpoint,
            "phase2_checkpoint": config.phase2_checkpoint,
        },
    )
