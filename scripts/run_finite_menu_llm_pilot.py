"""Paired gold-profile LLM pilot for proxy tool use and adapter transfer."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.evaluation.preference_benchmark import (
    BENCHMARK_VERSION,
    evaluate_action,
    expected_utilities,
    make_scenario,
)
from src.training.synthetic_users import SyntheticUserSampler


def format_prompt(scenario: object, theta: object, tool_advice: str | None) -> str:
    lines = [
        "You are advising a user on one resource allocation decision.",
        "Payoffs are fractions of the current resource budget. A payoff can be negative.",
        "Periods are numbered from 0 (now) onward. Choose one action.",
        f"User preferences: gamma={theta.gamma:.4f} (future discount), "
        f"alpha={theta.alpha:.4f} (outcome curvature), "
        f"lambda={theta.lambda_:.4f} (loss sensitivity).",
        "Actions:",
    ]
    for action, label in enumerate(scenario.labels):
        outcomes = []
        for outcome, probability in enumerate(scenario.probabilities[action]):
            if probability < 1e-12:
                continue
            events = [
                f"period {period}: {payoff:+.4f}"
                for period, payoff in enumerate(scenario.payoffs[action, outcome])
                if abs(payoff) > 1e-12
            ]
            outcomes.append(f"{probability:.2%} chance of ({', '.join(events)})")
        lines.append(f"{chr(65 + action)} [{label}]: {'; '.join(outcomes)}")
    if tool_advice is not None:
        lines.append(f"Preference proxy tool result: recommended action {tool_advice}.")
    lines.append("Reply with exactly one capital letter: A, B, C, or D.")
    return "\n".join(lines)


def parse_action(completion: str) -> int | None:
    match = re.fullmatch(r"\s*([ABCD])\s*[.\n]*\s*", completion)
    return ord(match.group(1)) - ord("A") if match else None


def generate(model: object, tokenizer: object, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"]
    encoded = encoded.to(model.device)
    with torch.inference_mode():
        output = model.generate(
            encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if not getattr(generate, "_reported_gpu", False):
        print(json.dumps({
            "event": "first_generation",
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "input_tokens": int(encoded.shape[-1]),
            "output_tokens": int(output.shape[-1] - encoded.shape[-1]),
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "gpu_memory_peak_bytes": int(torch.cuda.max_memory_allocated()),
        }), flush=True)
        generate._reported_gpu = True
    return tokenizer.decode(output[0, encoded.shape[-1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--condition", choices=("base", "dialogue"), required=True)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--n-users", type=int, default=12)
    parser.add_argument("--n-scenarios", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.condition == "dialogue" and args.checkpoint is None:
        parser.error("dialogue requires --checkpoint")

    from src.training.model_utils import load_model_with_optional_checkpoint

    print(json.dumps({"event": "model_load_start", "condition": args.condition,
                      "at_utc": datetime.now(timezone.utc).isoformat()}), flush=True)
    model, tokenizer = load_model_with_optional_checkpoint(
        args.model_name, str(args.checkpoint) if args.checkpoint else None
    )
    import torch
    print(json.dumps({
        "event": "model_loaded", "condition": args.condition,
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_device_count": int(torch.cuda.device_count()),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
    }), flush=True)
    policy = None
    if args.policy_checkpoint:
        import torch

        from src.training.finite_menu_ppo import FiniteMenuPolicy

        policy = FiniteMenuPolicy()
        policy.load_state_dict(torch.load(
            args.policy_checkpoint, map_location="cpu", weights_only=True
        ))
        policy.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.condition}.jsonl"
    users = SyntheticUserSampler(seed=args.seed).sample_batch(args.n_users)
    scenarios = [make_scenario(args.seed + 1_000_000 + i) for i in range(args.n_scenarios)]
    arms = ["no_tool", "gold_reward_tool"]
    if policy is not None:
        arms.append("learned_rl_tool")
    n_rows = 0
    with output_path.open("w") as handle:
        for user_id, theta in enumerate(users):
            for scenario in scenarios:
                values = expected_utilities(scenario, theta)
                for arm in arms:
                    advice = None
                    if arm == "gold_reward_tool":
                        advice = chr(65 + int(np.argmax(values)))
                    elif arm == "learned_rl_tool":
                        advice = chr(65 + policy.act(scenario, theta))
                    prompt = format_prompt(scenario, theta, advice)
                    completion = generate(model, tokenizer, prompt, args.max_new_tokens)
                    action = parse_action(completion)
                    metrics = (
                        evaluate_action(scenario, theta, action)
                        if action is not None else None
                    )
                    row = {
                        "benchmark_version": BENCHMARK_VERSION,
                        "condition": args.condition,
                        "arm": arm,
                        "seed": args.seed,
                        "user_id": user_id,
                        "scenario_id": scenario.scenario_id,
                        "true_theta": theta.__dict__,
                        "tool_advice": advice,
                        "prompt": prompt,
                        "completion": completion,
                        "parse_failure": action is None,
                        "metrics": metrics,
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    n_rows += 1
                    if n_rows <= 6:
                        print(json.dumps({
                            "condition": args.condition, "arm": arm,
                            "completion": completion, "parse_failure": action is None,
                        }), flush=True)
                if n_rows % 30 == 0:
                    print(f"records={n_rows}", flush=True)
    receipt = {
        "benchmark_version": BENCHMARK_VERSION,
        "condition": args.condition,
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "policy_checkpoint": str(args.policy_checkpoint) if args.policy_checkpoint else None,
        "n_users": args.n_users,
        "n_scenarios": args.n_scenarios,
        "arms": arms,
        "expected_records": args.n_users * args.n_scenarios * len(arms),
        "records": n_rows,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / f"{args.condition}_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
