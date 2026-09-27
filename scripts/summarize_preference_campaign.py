"""Summarize paired per-user canonical campaign results with uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "total_error", "alignment", "decision_regret", "decision_regret_relative",
    "question_count", "quality_floor_violation",
)


def _read_record(path: Path) -> dict[str, Any]:
    with path.open() as f:
        record = json.load(f)
    error = record.get("final_error") or {}
    align = record.get("alignment") or {}
    return {
        "domain": record["domain"], "arm": record["arm"],
        "seed": int(record["seed"]), "user_idx": int(record["user_idx"]),
        "total_error": error.get("total"), "alignment": align.get("spearman"),
        "decision_regret": align.get("decision_regret"),
        "decision_regret_relative": align.get("decision_regret_relative"),
        "question_count": record.get("question_count"),
        "quality_floor_violation": align.get("quality_floor_violation"),
    }


def _bootstrap_mean_ci(values: list[float], seed: int, n_boot: int) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(input_dir: Path, output_dir: Path, n_boot: int = 10000) -> dict[str, int]:
    records = [_read_record(p) for p in sorted(input_dir.glob("*/*/s*_u*.json"))]
    if not records:
        raise FileNotFoundError(f"No per-user campaign records under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    matched: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        grouped[(r["domain"], r["arm"])].append(r)
        matched[(r["domain"], r["seed"], r["user_idx"])][r["arm"]] = r

    summary_rows = []
    for (domain, arm), rows in sorted(grouped.items()):
        row: dict[str, Any] = {"domain": domain, "arm": arm, "n": len(rows)}
        for metric in METRICS:
            values = [float(r[metric]) for r in rows if r[metric] is not None]
            mean, lo, hi = _bootstrap_mean_ci(values, 1701, n_boot)
            row.update({f"{metric}_mean": mean, f"{metric}_ci95_low": lo, f"{metric}_ci95_high": hi})
        summary_rows.append(row)

    paired_rows = []
    arms = sorted({r["arm"] for r in records} - {"random"})
    for domain in sorted({r["domain"] for r in records}):
        for arm in arms:
            pairs = [g for (d, _, _), g in matched.items() if d == domain and arm in g and "random" in g]
            if not pairs:
                continue
            for metric in METRICS:
                diffs = [float(g[arm][metric]) - float(g["random"][metric]) for g in pairs
                         if g[arm][metric] is not None and g["random"][metric] is not None]
                mean, lo, hi = _bootstrap_mean_ci(diffs, 2917, n_boot)
                sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")
                paired_rows.append({
                    "domain": domain, "contrast": f"{arm}-random", "metric": metric,
                    "n_paired": len(diffs), "mean_difference": mean,
                    "ci95_low": lo, "ci95_high": hi,
                    "paired_cohens_dz": mean / sd if sd > 0 else float("nan"),
                })

    for filename, rows in (("per_user.csv", records), ("summary.csv", summary_rows), ("paired_contrasts.csv", paired_rows)):
        if not rows:
            continue
        with (output_dir / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = json.loads((input_dir / "manifest.json").read_text()) if (input_dir / "manifest.json").exists() else {}
    (output_dir / "analysis.json").write_text(json.dumps({
        "input_dir": str(input_dir), "n_records": len(records), "n_bootstrap": n_boot,
        "campaign_manifest": manifest,
        "interpretation": {
            "paired_contrast_sign": "negative is favorable for error/regret/rounds/violations; positive is favorable for alignment",
            "decision_regret": "common-random-number expected-utility gap versus the true-type optimal action in the common initial state",
            "intervals": "percentile bootstrap 95% intervals over matched user observations",
        },
    }, indent=2))
    return {"n_records": len(records), "n_summary_rows": len(summary_rows), "n_paired_rows": len(paired_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    args = parser.parse_args()
    source = Path(args.input_dir)
    target = Path(args.output_dir) if args.output_dir else source / "analysis"
    print(json.dumps(summarize(source, target, args.n_bootstrap), indent=2))


if __name__ == "__main__":
    main()
