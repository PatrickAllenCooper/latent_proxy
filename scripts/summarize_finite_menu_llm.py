"""Validate and summarize paired finite-menu LLM generations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.preference_benchmark import BENCHMARK_VERSION


def user_means(rows: list[dict], field: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if field == "parse_failure":
            value = float(row["parse_failure"])
        elif row["parse_failure"]:
            value = 1.0 if field == "normalized_regret" else 0.0
        else:
            value = float(row["metrics"][field])
        grouped[int(row["user_id"])].append(value)
    return {user: float(np.mean(values)) for user, values in grouped.items()}


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.integers(0, len(values), size=(2000, len(values)))
    return [float(x) for x in np.quantile(values[draws].mean(axis=1), [.025, .975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for condition in ("base", "dialogue"):
        receipt = json.loads((args.input_dir / f"{condition}_receipt.json").read_text())
        condition_rows = [json.loads(line) for line in
                          (args.input_dir / f"{condition}.jsonl").read_text().splitlines()]
        if len(condition_rows) != receipt["expected_records"]:
            raise ValueError(f"{condition} count mismatch")
        if receipt["benchmark_version"] != BENCHMARK_VERSION:
            raise ValueError(f"{condition} benchmark version mismatch")
        rows.extend(condition_rows)
    keys = [(r["condition"], r["arm"], r["seed"], r["user_id"], r["scenario_id"])
            for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate paired decision keys")
    if any(r["benchmark_version"] != BENCHMARK_VERSION for r in rows):
        raise ValueError("mixed benchmark versions")
    if any(not r["prompt"] or not isinstance(r["completion"], str) for r in rows):
        raise ValueError("missing raw prompt or completion")
    if any(not r["parse_failure"] and not r["metrics"]["feasible"] for r in rows):
        raise ValueError("infeasible parsed recommendation")

    rng = np.random.default_rng(91_338)
    cells = {(r["condition"], r["arm"]) for r in rows}
    by_cell = {cell: [r for r in rows if (r["condition"], r["arm"]) == cell]
               for cell in sorted(cells)}
    summary: dict[str, object] = {"benchmark_version": BENCHMARK_VERSION,
                                  "records": len(rows), "cells": {}, "contrasts": {}}
    fields = ("normalized_regret", "behavioral_agreement", "parse_failure")
    for cell, cell_rows in by_cell.items():
        entry: dict[str, object] = {"records": len(cell_rows)}
        for field in fields:
            means = user_means(cell_rows, field)
            vals = np.asarray(list(means.values()), dtype=np.float64)
            entry[field] = {"mean": float(vals.mean()),
                            "user_bootstrap_95": bootstrap(vals, rng)}
        summary["cells"]["/".join(cell)] = entry
    comparisons = []
    for condition in ("base", "dialogue"):
        for tool in ("gold_reward_tool", "learned_rl_tool"):
            comparisons.append(((condition, tool), (condition, "no_tool")))
    for arm in ("no_tool", "gold_reward_tool", "learned_rl_tool"):
        comparisons.append((("dialogue", arm), ("base", arm)))
    for first, second in comparisons:
        if first not in by_cell or second not in by_cell:
            continue
        entry = {}
        for field in fields:
            first_users = user_means(by_cell[first], field)
            second_users = user_means(by_cell[second], field)
            if first_users.keys() != second_users.keys():
                raise ValueError(f"unpaired users: {first}, {second}")
            delta = np.array([first_users[u] - second_users[u] for u in sorted(first_users)])
            entry[field] = {"mean_difference": float(delta.mean()),
                            "user_bootstrap_95": bootstrap(delta, rng)}
        summary["contrasts"]["/".join(first) + " minus " + "/".join(second)] = entry
    (args.input_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["cells"], indent=2))


if __name__ == "__main__":
    main()
