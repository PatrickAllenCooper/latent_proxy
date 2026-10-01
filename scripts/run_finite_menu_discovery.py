"""Paired analytic-user experiment for preference discovery and AIF acquisition."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.agents.finite_menu_inference import (
    UserParticles,
    choose_query,
    posterior_mean,
    prepared_menus,
    update_weights,
)
from src.evaluation.preference_benchmark import (
    BENCHMARK_VERSION,
    behavior_probabilities,
    evaluate_action,
    make_scenario,
)
from src.training.synthetic_users import SyntheticUserSampler


ARMS = ("random", "eig", "decision_value", "aif_50")
ADAPTIVE_ARMS = ("random", "eig", "eig_hazard_10", "eig_hazard_25")
TRIGGER_ARMS = ("random", "eig", "eig_hazard_10", "eig_surprise_05",
                "eig_surprise_10")
BUDGETS = (0, 2, 4, 8)


def response_seed(seed: int, user_id: int, query_id: int) -> int:
    return seed + 3_000_000 + user_id * 10_000 + query_id


def evaluate_posterior(
    *,
    true_theta: object,
    true_bias: np.ndarray,
    weights: np.ndarray,
    particles: UserParticles,
    target_utilities: np.ndarray,
    target_behavior: np.ndarray,
    targets: list,
    stress: str,
) -> dict[str, float]:
    estimated_reward = np.einsum("n,tna->ta", weights, target_utilities)
    estimated_behavior = np.einsum("n,tna->ta", weights, target_behavior)
    regret_reward = []
    agreement_reward = []
    regret_behavior = []
    agreement_behavior = []
    for i, scenario in enumerate(targets):
        reward_choice = int(np.argmax(estimated_reward[i]))
        behavior_choice = int(np.argmax(estimated_behavior[i]))
        first = evaluate_action(scenario, true_theta, reward_choice,
                                behavior_bias=true_bias)
        second = evaluate_action(scenario, true_theta, behavior_choice,
                                 behavior_bias=true_bias)
        observed_policy = behavior_probabilities(
            scenario, true_theta, bias=true_bias,
            temperature=.06 if stress == "noisy" else .025,
        )
        if stress == "inconsistent":
            observed_policy = .8 * observed_policy + .05
        regret_reward.append(first["normalized_regret"])
        agreement_reward.append(float(observed_policy[reward_choice]))
        regret_behavior.append(second["normalized_regret"])
        agreement_behavior.append(float(observed_policy[behavior_choice]))
    estimate = posterior_mean(weights, particles)
    estimated_bias = np.dot(weights, particles.behavior_bias)
    return {
        "estimated_gamma": estimate.gamma,
        "estimated_alpha": estimate.alpha,
        "estimated_lambda": estimate.lambda_,
        "estimated_bias_safe": float(estimated_bias[0]),
        "estimated_bias_delayed": float(estimated_bias[1]),
        "estimated_bias_risky": float(estimated_bias[2]),
        "estimated_bias_balanced": float(estimated_bias[3]),
        "gamma_abs_error": abs(estimate.gamma - true_theta.gamma),
        "alpha_abs_error": abs(estimate.alpha - true_theta.alpha),
        "lambda_abs_error": abs(estimate.lambda_ - true_theta.lambda_),
        "reward_regret": float(np.mean(regret_reward)),
        "reward_agreement": float(np.mean(agreement_reward)),
        "behavior_regret": float(np.mean(regret_behavior)),
        "behavior_agreement": float(np.mean(agreement_behavior)),
        "posterior_ess": float(1.0 / np.square(weights).sum()),
    }


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.integers(0, len(values), size=(2000, len(values)))
    return [float(x) for x in np.quantile(values[draws].mean(axis=1), [.025, .975])]


def summarize(rows: list[dict], arms: tuple[str, ...] = ARMS) -> dict:
    rng = np.random.default_rng(91_339)
    fields = (
        "gamma_abs_error", "alpha_abs_error", "lambda_abs_error",
        "reward_regret", "reward_agreement", "behavior_regret",
        "behavior_agreement", "posterior_ess",
    )
    by_key = {(r["arm"], r["budget"], r["user_id"]): r for r in rows}
    users = sorted({r["user_id"] for r in rows})
    result: dict = {"benchmark_version": BENCHMARK_VERSION, "n_users": len(users),
                    "cells": {}, "paired_contrasts": {}}
    for budget in BUDGETS:
        for arm in arms:
            cell = {}
            for field in fields:
                values = np.array([by_key[arm, budget, u][field] for u in users])
                cell[field] = {"mean": float(values.mean()),
                               "user_bootstrap_95": bootstrap(values, rng)}
            result["cells"][f"{arm}/{budget}"] = cell
        if budget == 0:
            continue
        for arm in arms:
            if arm == "random":
                continue
            contrast = {}
            for field in fields[:-1]:
                values = np.array([
                    by_key[arm, budget, u][field]
                    - by_key["random", budget, u][field]
                    for u in users
                ])
                contrast[field] = {"mean_difference": float(values.mean()),
                                   "user_bootstrap_95": bootstrap(values, rng)}
            result["paired_contrasts"][f"{arm} minus random/{budget}"] = contrast
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=30)
    parser.add_argument("--n-particles", type=int, default=256)
    parser.add_argument("--n-queries", type=int, default=24)
    parser.add_argument("--n-targets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument(
        "--stress", choices=("matched", "noisy", "inconsistent", "shift"),
        default="matched",
    )
    parser.add_argument("--arm-set", choices=("standard", "change_adaptation",
                                               "change_trigger"),
                        default="standard")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.n_queries < max(BUDGETS):
        parser.error("query pool must fit the largest budget")
    if args.n_users < 2 or args.n_targets < 1:
        parser.error("need at least two users and one target")
    arms = {"standard": ARMS, "change_adaptation": ADAPTIVE_ARMS,
            "change_trigger": TRIGGER_ARMS}[args.arm_set]
    users = SyntheticUserSampler(seed=args.seed).sample_batch(args.n_users)
    shifted_users = SyntheticUserSampler(seed=args.seed + 800_000).sample_batch(args.n_users)
    queries = [make_scenario(args.seed + 100_000 + i) for i in range(args.n_queries)]
    acquisition_targets = [make_scenario(args.seed + 200_000 + i) for i in range(4)]
    evaluation_targets = [make_scenario(args.seed + 300_000 + i)
                          for i in range(args.n_targets)]
    rng = np.random.default_rng(args.seed + 400_000)
    rows = []
    traces = []
    for user_id, true_theta in enumerate(users):
        true_bias = rng.normal(0.0, .018, size=4)
        true_bias[0] += rng.uniform(-.06, .06)
        shifted_bias = rng.normal(0.0, .018, size=4)
        shifted_bias[0] += rng.uniform(-.06, .06)
        particles = UserParticles.sample(args.n_particles, args.seed + 500_000 + user_id)
        _, query_behavior = prepared_menus(queries, particles)
        acquisition_utility, _ = prepared_menus(acquisition_targets, particles)
        target_utility, target_behavior = prepared_menus(evaluation_targets, particles)
        true_query_probabilities = [behavior_probabilities(
            s, true_theta, bias=true_bias,
            temperature=.06 if args.stress == "noisy" else .025,
        ) for s in queries]
        shifted_query_probabilities = [behavior_probabilities(
            s, shifted_users[user_id], bias=shifted_bias,
        ) for s in queries]
        for arm in arms:
            weights = np.ones(args.n_particles, dtype=np.float64) / args.n_particles
            available = np.ones(args.n_queries, dtype=np.bool_)
            arm_rng = np.random.default_rng(args.seed + 600_000 + user_id * 100 + arms.index(arm))
            for step in range(max(BUDGETS) + 1):
                after_shift = args.stress == "shift" and step >= 4
                evaluation_theta = shifted_users[user_id] if after_shift else true_theta
                evaluation_bias = shifted_bias if after_shift else true_bias
                if step in BUDGETS:
                    metrics = evaluate_posterior(
                        true_theta=evaluation_theta, true_bias=evaluation_bias, weights=weights,
                        particles=particles, target_utilities=target_utility,
                        target_behavior=target_behavior, targets=evaluation_targets,
                        stress=args.stress,
                    )
                    rows.append({
                        "seed": args.seed, "user_id": user_id, "arm": arm,
                        "budget": step, "stress": args.stress,
                        "true_gamma": evaluation_theta.gamma,
                        "true_alpha": evaluation_theta.alpha,
                        "true_lambda": evaluation_theta.lambda_,
                        "true_bias_safe": float(evaluation_bias[0]),
                        "true_bias_delayed": float(evaluation_bias[1]),
                        "true_bias_risky": float(evaluation_bias[2]),
                        "true_bias_balanced": float(evaluation_bias[3]),
                        **metrics,
                    })
                if step == max(BUDGETS):
                    break
                query_id, information, decision_value = choose_query(
                    "eig" if arm.startswith(("eig_hazard_", "eig_surprise_")) else arm,
                    weights, query_behavior, acquisition_utility, available, arm_rng
                )
                available[query_id] = False
                response_rng = np.random.default_rng(response_seed(args.seed, user_id, query_id))
                true_probabilities = (
                    shifted_query_probabilities if after_shift else true_query_probabilities
                )
                choice = int(response_rng.choice(4, p=true_probabilities[query_id]))
                if args.stress == "inconsistent" and response_rng.random() < .15:
                    choice = int(response_rng.choice([a for a in range(4) if a != choice]))
                predictive_choice_probability = float(np.dot(
                    weights, query_behavior[query_id, :, choice]
                ))
                hazard = {"eig_hazard_10": .10, "eig_hazard_25": .25}.get(arm, 0.0)
                surprise_threshold = {"eig_surprise_05": .05,
                                      "eig_surprise_10": .10}.get(arm)
                if (surprise_threshold is not None
                        and predictive_choice_probability < surprise_threshold):
                    hazard = .25
                predictive_weights = ((1.0 - hazard) * weights
                                      + hazard / args.n_particles)
                weights = update_weights(predictive_weights,
                                         query_behavior[query_id, :, choice])
                traces.append({
                    "seed": args.seed, "user_id": user_id, "arm": arm,
                    "round": step + 1, "query_id": query_id,
                    "scenario_id": queries[query_id].scenario_id,
                    "choice": choice,
                    "stress": args.stress,
                    "after_shift": after_shift,
                    "predictive_choice_probability": predictive_choice_probability,
                    "refresh_applied": hazard,
                    "selected_information_gain": None if np.isnan(information) else information,
                    "selected_decision_value": None if np.isnan(decision_value) else decision_value,
                    "posterior_ess": float(1.0 / np.square(weights).sum()),
                })
        if (user_id + 1) % 5 == 0:
            print(f"users={user_id + 1}/{args.n_users}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_user.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "queries.jsonl").open("w") as handle:
        for trace in traces:
            handle.write(json.dumps(trace) + "\n")
    report = summarize(rows, arms)
    report["config"] = {"seed": args.seed, "n_users": args.n_users,
                        "n_particles": args.n_particles, "n_queries": args.n_queries,
                        "n_targets": args.n_targets, "budgets": BUDGETS,
                        "arms": arms, "arm_set": args.arm_set,
                        "records": len(rows), "query_records": len(traces),
                        "stress": args.stress,
                        "simulator": "analytic stochastic 4-way choice with behavioral bias"}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"records": len(rows), "query_records": len(traces)}, indent=2))


if __name__ == "__main__":
    main()
