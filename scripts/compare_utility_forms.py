"""Paired per-user comparison of absolute and return-normalized campaigns."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load(root: Path, domain: str, arm: str, seeds: list[int]) -> dict[tuple[int, int], dict[str, Any]]:
    result = {}
    for seed in seeds:
        for path in (root / domain / arm).glob(f"s{seed}_u*.json"):
            with path.open() as f:
                row = json.load(f)
            idx = int(row["user_idx"])
            result[(seed, idx)] = row
    return result


def compare(old: Path, new: Path, domain: str, arm: str, seeds: list[int], n_boot: int) -> list[dict[str, Any]]:
    a, b = load(old, domain, arm, seeds), load(new, domain, arm, seeds)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise FileNotFoundError("No matched seed/user records in both campaign directories")
    metrics = {
        "total_error": lambda r: (r.get("final_error") or {}).get("total"),
        "alpha_error": lambda r: (r.get("final_error") or {}).get("alpha"),
        "alignment": lambda r: (r.get("alignment") or {}).get("spearman"),
        "decision_regret": lambda r: (r.get("alignment") or {}).get("decision_regret"),
        "quality_floor_violation": lambda r: (r.get("alignment") or {}).get("quality_floor_violation"),
        "question_count": lambda r: r.get("question_count"),
    }
    rng = np.random.default_rng(51041)
    rows = []
    for metric, getter in metrics.items():
        diffs = np.asarray([float(getter(b[k])) - float(getter(a[k])) for k in keys
                            if getter(a[k]) is not None and getter(b[k]) is not None])
        if not len(diffs):
            continue
        boot = np.mean(rng.choice(diffs, (n_boot, len(diffs)), replace=True), axis=1)
        mean = float(diffs.mean())
        sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else float("nan")
        rows.append({
            "domain": domain, "arm": arm, "metric": metric,
            "comparison": "return_normalized - absolute", "n_paired": len(diffs),
            "mean_difference": mean,
            "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)),
            "paired_cohens_dz": mean / sd if sd > 0 else float("nan"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute-dir", default="outputs/canonical")
    parser.add_argument("--normalized-dir", default="outputs/canonical_renorm_stage3")
    parser.add_argument("--domain", default="supply_chain")
    parser.add_argument("--arm", default="active")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--output", default="outputs/canonical_renorm_stage3/utility_form_comparison.csv")
    args = parser.parse_args()
    rows = compare(Path(args.absolute_dir), Path(args.normalized_dir), args.domain, args.arm,
                   [int(x) for x in args.seeds.split(",")], args.n_bootstrap)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"matched_users": rows[0]["n_paired"], "output": str(destination), "metrics": rows}, indent=2))


if __name__ == "__main__":
    main()
