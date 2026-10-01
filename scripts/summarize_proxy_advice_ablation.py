"""Summarize paired proxy-advice conditions with user-level bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.integers(0, len(values), size=(5000, len(values)))
    return [float(v) for v in np.quantile(values[draws].mean(axis=1), [.025, .975])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ablation = load(args.ablation)
    baseline = [r for r in load(args.baseline) if r["arm"] == "no_tool"]
    if len(ablation) != 288 or len(baseline) != 96:
        raise ValueError("expected 288 ablation and 96 paired baseline records")
    rows = []
    for r in ablation + baseline:
        action = r["completion"] if not r["parse_failure"] else None
        rows.append({
            "user_id": int(r["user_id"]), "scenario_id": r["scenario_id"],
            "arm": r["arm"], "normalized_regret":
                float(r["metrics"]["normalized_regret"]) if r["metrics"] else 1.0,
            "gold_choice": float(action == r.get("gold_action"))
                if "gold_action" in r else float(r["metrics"]["normalized_regret"] == 0),
            "wrong_choice": float(action == r["wrong_action"])
                if "wrong_action" in r else float("nan"),
            "parse_failure": float(r["parse_failure"]),
        })
    by_key = {(r["user_id"], r["scenario_id"], r["arm"]): r for r in rows}
    if len(by_key) != 384:
        raise ValueError("missing or duplicate paired case")
    users = sorted({r["user_id"] for r in rows})
    scenarios = sorted({r["scenario_id"] for r in rows})
    if len(users) != 12 or len(scenarios) != 8:
        raise ValueError("unexpected user or scenario count")
    arms = ("no_tool", "wrong_direct", "wrong_caveated", "exact_scores")
    user_rows = []
    for user in users:
        for arm in arms:
            subset = [by_key[user, scenario, arm] for scenario in scenarios]
            user_rows.append({"user_id": user, "arm": arm,
                              **{field: float(np.mean([r[field] for r in subset]))
                                 for field in ("normalized_regret", "gold_choice",
                                               "parse_failure")}})
    user_map = {(r["user_id"], r["arm"]): r for r in user_rows}
    rng = np.random.default_rng(91777)
    fields = ("normalized_regret", "gold_choice", "parse_failure")
    report = {"n_users": len(users), "n_scenarios": len(scenarios),
              "records": len(rows), "cells": {}, "paired_contrasts": {}}
    for arm in arms:
        report["cells"][arm] = {}
        for field in fields:
            values = np.array([user_map[u, arm][field] for u in users])
            report["cells"][arm][field] = {
                "mean": float(values.mean()), "user_bootstrap_95": bootstrap(values, rng)}
    for a, b in (("wrong_direct", "no_tool"), ("wrong_caveated", "wrong_direct"),
                 ("exact_scores", "no_tool"), ("exact_scores", "wrong_caveated")):
        contrast = {}
        for field in fields:
            differences = np.array([user_map[u, a][field]
                                    - user_map[u, b][field] for u in users])
            contrast[field] = {"mean_difference": float(differences.mean()),
                               "user_bootstrap_95": bootstrap(differences, rng)}
        report["paired_contrasts"][f"{a} minus {b}"] = contrast
    report["wrong_direct_follow_rate"] = float(np.mean([
        r["wrong_choice"] for r in rows if r["arm"] == "wrong_direct"
    ]))
    report["wrong_caveated_follow_rate"] = float(np.mean([
        r["wrong_choice"] for r in rows if r["arm"] == "wrong_caveated"
    ]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_user.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(user_rows[0]))
        writer.writeheader()
        writer.writerows(user_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
