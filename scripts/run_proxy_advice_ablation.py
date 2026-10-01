"""Prepare CPU prompts and run a small GPU advice-reliability ablation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.run_finite_menu_llm_pilot import format_prompt, generate, parse_action
from src.evaluation.preference_benchmark import (
    BENCHMARK_VERSION, evaluate_action, expected_utilities, make_scenario,
)
from src.training.synthetic_users import SyntheticUserSampler


ARMS = ("wrong_direct", "wrong_caveated", "exact_scores")


def prepare(path: Path, n_users: int, n_scenarios: int, seed: int) -> None:
    users = SyntheticUserSampler(seed=seed).sample_batch(n_users)
    scenarios = [make_scenario(seed + 1_000_000 + i) for i in range(n_scenarios)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for user_id, theta in enumerate(users):
            for scenario in scenarios:
                values = expected_utilities(scenario, theta)
                gold = int(np.argmax(values))
                wrong = int(np.argmin(values))
                assert gold != wrong
                base = format_prompt(scenario, theta, None)
                head, final_instruction = base.rsplit("\n", 1)
                metrics = [evaluate_action(scenario, theta, a) for a in range(4)]
                for arm in ARMS:
                    if arm == "wrong_direct":
                        tool = f"Preference proxy tool result: recommended action {chr(65 + wrong)}."
                    elif arm == "wrong_caveated":
                        tool = ("A preliminary preference proxy recommends action "
                                f"{chr(65 + wrong)}, but it may be inaccurate. "
                                "Verify it against the payoffs and user preferences before deciding.")
                    else:
                        scores = ", ".join(
                            f"{chr(65 + a)}={values[a]:+.6f}" for a in range(4)
                        )
                        tool = ("Preference proxy tool result: expected user utility "
                                f"by action is {scores}. Choose the highest utility.")
                    row = {
                        "benchmark_version": BENCHMARK_VERSION,
                        "seed": seed, "user_id": user_id,
                        "scenario_id": scenario.scenario_id, "arm": arm,
                        "true_theta": theta.__dict__,
                        "expected_utilities": [float(x) for x in values],
                        "gold_action": chr(65 + gold),
                        "wrong_action": chr(65 + wrong),
                        "metrics_by_action": metrics,
                        "prompt": f"{head}\n{tool}\n{final_instruction}",
                    }
                    handle.write(json.dumps(row) + "\n")
    print(json.dumps({"event": "cpu_preparation_complete", "at_utc":
                      datetime.now(timezone.utc).isoformat(), "path": str(path),
                      "records": n_users * n_scenarios * len(ARMS)}), flush=True)


def run(manifest: Path, output_dir: Path, model_name: str, checkpoint: Path | None) -> None:
    import torch
    from src.training.model_utils import load_model_with_optional_checkpoint

    rows = [json.loads(x) for x in manifest.open()]
    if not rows:
        raise ValueError("empty CPU-prepared manifest")
    condition = "dialogue" if checkpoint else "base"
    print(json.dumps({"event": "model_load_start", "condition": condition,
                      "at_utc": datetime.now(timezone.utc).isoformat()}), flush=True)
    model, tokenizer = load_model_with_optional_checkpoint(
        model_name, str(checkpoint) if checkpoint else None
    )
    print(json.dumps({"event": "model_loaded", "condition": condition,
                      "at_utc": datetime.now(timezone.utc).isoformat(),
                      "gpu_device_count": torch.cuda.device_count(),
                      "gpu_name": torch.cuda.get_device_name(0),
                      "gpu_memory_allocated_bytes": torch.cuda.memory_allocated()}), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{condition}.jsonl"
    with output_path.open("w") as handle:
        for i, prepared in enumerate(rows):
            completion = generate(model, tokenizer, prepared["prompt"], 24)
            action = parse_action(completion)
            row = {k: v for k, v in prepared.items() if k != "metrics_by_action"}
            row.update(condition=condition, completion=completion,
                       parse_failure=action is None,
                       metrics=None if action is None else
                       prepared["metrics_by_action"][action])
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if i < 6 or (i + 1) % 30 == 0:
                print(json.dumps({"event": "generation_progress", "at_utc":
                                  datetime.now(timezone.utc).isoformat(),
                                  "records": i + 1, "arm": prepared["arm"],
                                  "completion": completion,
                                  "parse_failure": action is None}), flush=True)
    receipt = {"condition": condition, "model_name": model_name,
               "checkpoint": str(checkpoint) if checkpoint else None,
               "manifest": str(manifest), "records": len(rows),
               "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (output_dir / f"{condition}_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-users", type=int, default=12)
    p.add_argument("--n-scenarios", type=int, default=8)
    p.add_argument("--seed", type=int, default=5001)
    g = sub.add_parser("generate")
    g.add_argument("--manifest", type=Path, required=True)
    g.add_argument("--output-dir", type=Path, required=True)
    g.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    g.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.n_users, args.n_scenarios, args.seed)
    else:
        run(args.manifest, args.output_dir, args.model_name, args.checkpoint)


if __name__ == "__main__":
    main()
