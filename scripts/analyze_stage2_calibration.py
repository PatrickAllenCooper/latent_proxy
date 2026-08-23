"""Stage 2 gate: per-seed (not pooled) comparison of return_normalized vs absolute.

Deliberately does NOT flat-pool all 5 seeds into one Wilcoxon test -- that
methodology is exactly what produced a false-positive result earlier in
this project (a807a47 -> 21806ab: a single-seed n=30 spot-check showed
improvement that reversed at full 5-seed x n=50 scale). Each seed is
treated as an independent replicate; promotion to Stage 3 requires
consistent-direction improvement across >=4-of-5 seeds, not just a
favorable average.

Usage:
    .venv/bin/python scripts/analyze_stage2_calibration.py \
        --new-dir outputs/canonical_renorm_stage2 \
        --old-dir outputs/canonical \
        --domains supply_chain,stock \
        --seeds 42,43,44,45,46
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_active_errors(
    base_dir: Path, domain: str, seed: int, user_idxs: range,
) -> dict[int, dict[str, float]]:
    """user_idx -> {total, alpha} error, active arm only."""
    out: dict[int, dict[str, float]] = {}
    for u in user_idxs:
        path = base_dir / domain / "active" / f"s{seed}_u{u}.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        fe = d.get("final_error") or {}
        out[u] = {"total": fe.get("total"), "alpha": fe.get("alpha")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new-dir", default="outputs/canonical_renorm_stage2")
    parser.add_argument("--old-dir", default="outputs/canonical")
    parser.add_argument("--domains", default="supply_chain,stock")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--n-users", type=int, default=20)
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    new_dir = Path(args.new_dir)
    old_dir = Path(args.old_dir)

    print(f"{'domain':14s} {'seed':>5s} {'n':>4s} {'old_mean_total':>15s} "
          f"{'new_mean_total':>15s} {'delta':>8s} {'direction':>10s}")

    summary: dict[str, Any] = {}
    for domain in domains:
        per_seed_deltas = []
        for seed in seeds:
            old = load_active_errors(old_dir, domain, seed, range(args.n_users))
            new = load_active_errors(new_dir, domain, seed, range(args.n_users))
            common = sorted(set(old) & set(new))
            if not common:
                print(f"{domain:14s} {seed:5d}   -- no matched users found --")
                continue
            old_vals = [old[u]["total"] for u in common]
            new_vals = [new[u]["total"] for u in common]
            old_mean = sum(old_vals) / len(old_vals)
            new_mean = sum(new_vals) / len(new_vals)
            delta = new_mean - old_mean  # negative = improvement (lower error)
            direction = "IMPROVED" if delta < 0 else "worse"
            per_seed_deltas.append(delta)
            print(f"{domain:14s} {seed:5d} {len(common):4d} {old_mean:15.4f} "
                  f"{new_mean:15.4f} {delta:8.4f} {direction:>10s}")

        n_improved = sum(1 for d in per_seed_deltas if d < 0)
        mean_delta = sum(per_seed_deltas) / len(per_seed_deltas) if per_seed_deltas else float("nan")
        summary[domain] = {
            "per_seed_deltas": per_seed_deltas,
            "n_seeds_improved": n_improved,
            "n_seeds_total": len(per_seed_deltas),
            "mean_delta": mean_delta,
            "gate_passed_4_of_5": n_improved >= 4,
        }
        print(f"  --> {domain}: {n_improved}/{len(per_seed_deltas)} seeds improved, "
              f"mean delta={mean_delta:.4f}, gate(>=4/5)={'PASS' if n_improved >= 4 else 'FAIL'}")
        print()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
