from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from src.environments.base import BaseEnvironment
from src.training.serialization import AllocationSerializer, build_prompt
from src.training.synthetic_users import UserType


def compute_alignment_score(
    agent_actions: list[NDArray[np.floating[Any]]],
    optimal_actions: list[NDArray[np.floating[Any]]],
) -> float:
    """Spearman rank correlation between agent and optimal allocations.

    For each action pair, ranks the channel allocations and computes
    the average rank correlation. A score of 1.0 means the agent
    perfectly matches the optimal ranking across all channels.
    """
    if len(agent_actions) == 0:
        return 0.0

    correlations = []
    for agent_act, opt_act in zip(agent_actions, optimal_actions):
        agent_flat = np.asarray(agent_act).flatten()
        opt_flat = np.asarray(opt_act).flatten()
        if np.std(agent_flat) < 1e-10 or np.std(opt_flat) < 1e-10:
            correlations.append(1.0 if np.allclose(agent_flat, opt_flat) else 0.0)
            continue
        corr, _ = stats.spearmanr(agent_flat, opt_flat)
        correlations.append(float(corr))

    return float(np.mean(correlations))


def compute_quality_floor_violation_rate(
    agent_actions: list[NDArray[np.floating[Any]]],
    env: BaseEnvironment,
) -> float:
    """Fraction of agent actions that violate quality floor constraints."""
    if len(agent_actions) == 0:
        return 0.0

    violations = 0
    for action in agent_actions:
        passes, _ = env.check_quality_floor(action)
        if not passes:
            violations += 1

    return violations / len(agent_actions)


def compute_preference_recovery_error(
    theta_inferred: dict[str, float],
    theta_true: UserType,
) -> dict[str, float]:
    """Per-parameter mean absolute error between inferred and true theta.

    Stub for Milestone 3 when the active learning loop produces inferred theta.
    """
    errors: dict[str, float] = {}
    true_dict = {
        "gamma": theta_true.gamma,
        "alpha": theta_true.alpha,
        "lambda_": theta_true.lambda_,
    }

    for param, true_val in true_dict.items():
        inferred_val = theta_inferred.get(param, 0.0)
        errors[param] = abs(inferred_val - true_val)

    errors["total"] = float(np.mean(list(errors.values())))
    return errors


def evaluate_actions(
    agent_actions: list[NDArray[np.floating[Any]]],
    env: BaseEnvironment,
    user_types: list[UserType],
) -> dict[str, Any]:
    """Evaluate a set of agent actions against optimal actions for given user types.

    Computes alignment score, quality score, and violation rate in one pass.
    """
    optimal_actions = []
    for ut in user_types:
        theta = {"gamma": ut.gamma, "alpha": ut.alpha, "lambda_": ut.lambda_}
        optimal_actions.append(env.get_optimal_action(theta))

    alignment = compute_alignment_score(agent_actions, optimal_actions)
    violation_rate = compute_quality_floor_violation_rate(agent_actions, env)

    quality_scores = [env.quality_score(a) for a in agent_actions]

    return {
        "alignment_score": alignment,
        "quality_floor_violation_rate": violation_rate,
        "mean_quality_score": float(np.mean(quality_scores)) if quality_scores else 0.0,
        "n_episodes": len(agent_actions),
    }


def evaluate_model_outputs(
    model_responses: list[str],
    env: BaseEnvironment,
    user_types: list[UserType],
    channel_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate raw text outputs from a trained model.

    Parses allocation text into arrays, then runs the standard evaluation.
    """
    serializer = AllocationSerializer(channel_names)
    agent_actions = [serializer.parse(response) for response in model_responses]
    return evaluate_actions(agent_actions, env, user_types)
