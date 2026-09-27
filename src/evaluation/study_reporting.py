"""Tabular summaries for paired model-condition studies."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_ci(values: list[float], seed: int, n_boot: int) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def _paired_rows(
    cell: dict[str, dict[str, dict[str, Any]]],
    metric_getters: dict[str, Callable[[dict[str, Any], int], float | None]],
    comparisons: list[tuple[str, str]],
    *,
    seed: int,
    n_boot: int,
) -> list[dict[str, Any]]:
    rows = []
    for condition_a, condition_b in comparisons:
        if condition_a not in cell or condition_b not in cell:
            continue
        user_ids = sorted(set(cell[condition_a]) & set(cell[condition_b]))
        for metric, getter in metric_getters.items():
            differences = []
            for uid in user_ids:
                a = getter(cell[condition_a][uid], int(uid))
                b = getter(cell[condition_b][uid], int(uid))
                if a is not None and b is not None:
                    differences.append(float(a) - float(b))
            mean, lo, hi = _mean_ci(differences, seed, n_boot)
            sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
            rows.append({
                "contrast": f"{condition_a}-{condition_b}", "metric": metric,
                "n_paired": len(differences), "mean_difference": mean,
                "ci95_low": lo, "ci95_high": hi,
                "paired_cohens_dz": mean / sd if sd > 0 else float("nan"),
            })
    return rows


def write_adherence_reports(
    results: dict[str, dict[str, Any]], output_dir: str | Path,
    *, seed: int = 42, n_boot: int = 10000,
) -> None:
    root = Path(output_dir)
    user_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    comparisons = [("dpo_phase2", "base"), ("dpo_phase2", "dpo_phase1"), ("dpo_dialogue", "dpo_phase2")]
    getters = {
        "alignment": lambda r, i: r.get("alignment"),
        "quality_floor_violation": lambda r, i: float(r.get("quality_floor_violation", False)),
        "parse_failure": lambda r, i: float(r.get("parse_failure", False)),
        "quality_constrained_alignment": lambda r, i: r.get("quality_constrained_alignment"),
        "decision_regret": lambda r, i: r.get("decision_regret"),
        "quality_constrained_decision_regret": lambda r, i: r.get("quality_constrained_decision_regret"),
    }
    for mode in next(iter(results.values())).keys():
        cell = {}
        for condition, by_mode in results.items():
            result = by_mode[mode]
            rows = result.per_user
            cell[condition] = {str(row["user_idx"]): row for row in rows}
            user_rows.extend({"theta_mode": mode, "condition": condition, **row} for row in rows)
            for metric in getters:
                values = [getters[metric](r, i) for i, r in enumerate(rows)]
                values = [float(v) for v in values if v is not None]
                mean, lo, hi = _mean_ci(values, seed, n_boot)
                summary_rows.append({
                    "theta_mode": mode, "condition": condition, "metric": metric,
                    "n": len(values), "mean": mean, "ci95_low": lo, "ci95_high": hi,
                })
        contrasts.extend({"theta_mode": mode, **r} for r in _paired_rows(
            cell, getters, comparisons, seed=seed, n_boot=n_boot,
        ))
    _write_csv(root / "per_user.csv", user_rows)
    _write_csv(root / "summary.csv", summary_rows)
    _write_csv(root / "paired_contrasts.csv", contrasts)


def write_dpo_reports(
    result: Any, output_dir: str | Path, *, seed: int = 42, n_boot: int = 10000,
) -> None:
    root = Path(output_dir)
    users: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    getters = {
        "alignment": lambda r, i: r.get("alignment"),
        "quality_floor_violation": lambda r, i: float(r.get("quality_floor_violation", False)),
        "query_parse_failure_rate": lambda r, i: r.get("query_parse_failure_rate"),
        "recommendation_parse_failure_rate": lambda r, i: r.get("recommendation_parse_failure_rate"),
    }
    comparisons = [("dpo_phase2", "base"), ("dpo_phase2", "dpo_phase1"), ("dpo_dialogue", "dpo_phase2")]
    for domain, by_condition in result.per_env.items():
        cell = {}
        for condition, condition_result in by_condition.items():
            rows = condition_result.per_user
            cell[condition] = {str(row["user_idx"]): row for row in rows}
            users.extend({"environment": domain, "condition": condition, **row} for row in rows)
            for metric in getters:
                values = [getters[metric](row, i) for i, row in enumerate(rows)]
                values = [float(value) for value in values if value is not None]
                mean, lo, hi = _mean_ci(values, seed, n_boot)
                summaries.append({
                    "environment": domain, "condition": condition, "metric": metric,
                    "n": len(values), "mean": mean, "ci95_low": lo, "ci95_high": hi,
                })
        contrasts.extend({"environment": domain, **r} for r in _paired_rows(
            cell, getters, comparisons, seed=seed, n_boot=n_boot,
        ))
    _write_csv(root / "per_user.csv", users)
    _write_csv(root / "summary.csv", summaries)
    _write_csv(root / "paired_contrasts.csv", contrasts)
