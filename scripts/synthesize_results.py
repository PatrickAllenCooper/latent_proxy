"""Pool genM6's 5 seeds and compute paired stats on the canonical campaign.

genM6 only persists per-seed aggregated (mean, 95% CI) per domain pair, not
raw per-user arrays, so seeds are combined via inverse-variance fixed-effect
meta-analysis (mean/CI) and Stouffer's method (hypothesis p-values), with
Holm-Bonferroni across the pooled hypothesis family.

The canonical campaign DOES have raw per-user records with users matched
across arms (same seed => same synthetic-user panel), so arm-vs-arm
comparisons use a real paired Wilcoxon signed-rank test on final total error.

Usage:
    python scripts/synthesize_results.py --output-dir outputs/synthesis
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.statistical_analysis import (
    _wilcoxon_safe,
    cohens_d_paired,
    holm_bonferroni,
    rank_biserial_r,
)

GENM6_SEEDS = [42, 43, 44, 45, 46]
CAMPAIGN_ARM_COMPARISONS = [
    ("active", "dirichlet"),
    ("active", "random"),
    ("active", "fixed"),
    ("fixed", "dirichlet"),
]


def _se_from_ci(mean: float, ci_low: float, ci_high: float, n: int) -> float:
    """Back out a standard error from a reported 95% CI, with an n-based floor."""
    width = ci_high - ci_low
    if width > 1e-12:
        return width / (2 * 1.959964)
    # Degenerate CI (e.g. all-zero generic baseline): fall back to a
    # conservative placeholder so it doesn't dominate the pooled weight.
    return 1.0 / max(np.sqrt(n), 1.0)


def inverse_variance_pool(entries: list[tuple[float, float, float, int]]) -> dict[str, float]:
    """entries: list of (mean, ci_low, ci_high, n). Fixed-effect IV pooling."""
    means = np.array([e[0] for e in entries])
    ses = np.array([_se_from_ci(*e) for e in entries])
    weights = 1.0 / np.maximum(ses**2, 1e-12)
    pooled_mean = float(np.sum(weights * means) / np.sum(weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(weights)))
    return {
        "pooled_mean": pooled_mean,
        "pooled_ci_low": pooled_mean - 1.959964 * pooled_se,
        "pooled_ci_high": pooled_mean + 1.959964 * pooled_se,
        "n_seeds": len(entries),
        "total_n": sum(e[3] for e in entries),
        "per_seed_means": means.tolist(),
    }


def stouffer_pool(p_values: list[float]) -> float:
    """Combine one-sided p-values across independent seeds via Stouffer's Z."""
    ps = np.clip(np.array(p_values), 1e-300, 1 - 1e-16)
    z = stats.norm.ppf(1 - ps)
    pooled_z = float(np.sum(z) / np.sqrt(len(z)))
    return float(1 - stats.norm.cdf(pooled_z))


def pool_genm6(genm6_dir: Path) -> dict[str, Any]:
    bundles = {}
    for seed in GENM6_SEEDS:
        path = genm6_dir / f"seed{seed}" / "results_bundle.json"
        if path.exists():
            bundles[seed] = json.load(open(path))

    domain_pairs: dict[str, dict[str, Any]] = {}
    all_keys = set()
    for b in bundles.values():
        all_keys.update(b["per_domain_pair"].keys())

    for key in sorted(all_keys):
        entry: dict[str, Any] = {}
        for branch in ("within_domain", "cross_domain"):
            vals = [
                (
                    b["per_domain_pair"][key][branch]["mean"],
                    b["per_domain_pair"][key][branch]["ci_low"],
                    b["per_domain_pair"][key][branch]["ci_high"],
                    b["per_domain_pair"][key]["n_users"],
                )
                for b in bundles.values()
                if key in b["per_domain_pair"]
            ]
            entry[branch] = inverse_variance_pool(vals)
        domain_pairs[key] = entry

    hyp_families: dict[str, list[float]] = defaultdict(list)
    hyp_effects: dict[str, list[float]] = defaultdict(list)
    for b in bundles.values():
        for h_key, results in b.get("hypothesis_results", {}).items():
            for r in results:
                hyp_families[r["hypothesis"]].append(r["p_value"])
                hyp_effects[r["hypothesis"]].append(r["effect_size"])

    pooled_hyps = {}
    for hyp_name, pvals in hyp_families.items():
        pooled_hyps[hyp_name] = {
            "n_seeds_present": len(pvals),
            "per_seed_p": pvals,
            "pooled_p_stouffer": stouffer_pool(pvals) if len(pvals) >= 2 else pvals[0],
            "mean_effect_size": float(np.mean(hyp_effects[hyp_name])),
        }

    names = sorted(pooled_hyps.keys())
    raw_p = [pooled_hyps[n]["pooled_p_stouffer"] for n in names]
    adjusted = holm_bonferroni(raw_p)
    for name, adj_p in zip(names, adjusted):
        pooled_hyps[name]["holm_adjusted_p"] = float(adj_p)
        pooled_hyps[name]["conclusion"] = "reject_null" if adj_p < 0.05 else "fail_to_reject"

    return {
        "n_seeds_found": len(bundles),
        "seeds": list(bundles.keys()),
        "domain_pairs_pooled": domain_pairs,
        "hypotheses_pooled": pooled_hyps,
    }


def load_campaign(campaign_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """domain -> arm -> list of {user_key, total_error, alignment}."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(str(campaign_dir / "*" / "*" / "*.json")):
        if "manifest" in path:
            continue
        parts = Path(path).parts
        domain, arm = parts[-3], parts[-2]
        d = json.load(open(path))
        fe = d.get("final_error") or {}
        align = (d.get("alignment") or {}).get("spearman")
        user_key = (d["seed"], d["user_idx"])
        out[domain][arm].append({
            "user_key": user_key,
            "total_error": fe.get("total"),
            "alpha_error": fe.get("alpha"),
            "gamma_error": fe.get("gamma"),
            "lambda_error": fe.get("lambda_"),
            "alignment": align,
        })
    return out


def paired_campaign_stats(campaign: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for domain, arms in campaign.items():
        by_arm_keyed = {
            arm: {r["user_key"]: r for r in records}
            for arm, records in arms.items()
        }
        domain_results = {}
        raw_p = []
        comparisons = []
        for arm_a, arm_b in CAMPAIGN_ARM_COMPARISONS:
            if arm_a not in by_arm_keyed or arm_b not in by_arm_keyed:
                continue
            common_keys = sorted(
                set(by_arm_keyed[arm_a]) & set(by_arm_keyed[arm_b])
            )
            a_err = np.array([by_arm_keyed[arm_a][k]["total_error"] for k in common_keys])
            b_err = np.array([by_arm_keyed[arm_b][k]["total_error"] for k in common_keys])
            # one-sided: is arm_a's error LOWER (better) than arm_b's?
            stat, pval = _wilcoxon_safe(b_err, a_err, alternative="greater")
            es = rank_biserial_r(stat, len(common_keys))
            d = cohens_d_paired(list(b_err), list(a_err))
            key = f"{arm_a}_lt_{arm_b}"
            comparisons.append(key)
            raw_p.append(pval)
            domain_results[key] = {
                "n_paired": len(common_keys),
                "mean_error_a": float(a_err.mean()),
                "mean_error_b": float(b_err.mean()),
                "p_value": pval,
                "rank_biserial_r": es,
                "cohens_d": d,
            }
        adjusted = holm_bonferroni(raw_p) if raw_p else []
        for key, adj_p in zip(comparisons, adjusted):
            domain_results[key]["holm_adjusted_p"] = float(adj_p)
            domain_results[key]["conclusion"] = (
                "reject_null" if adj_p < 0.05 else "fail_to_reject"
            )
        results[domain] = domain_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--genm6-dir", default="outputs/genM6")
    parser.add_argument("--campaign-dir", default="outputs/canonical")
    parser.add_argument("--output-dir", default="outputs/synthesis")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Pooling genM6 across seeds...")
    genm6_pooled = pool_genm6(Path(args.genm6_dir))
    with open(out_dir / "genm6_pooled.json", "w") as f:
        json.dump(genm6_pooled, f, indent=2, default=str)

    print("\n=== genM6 pooled hypotheses (Stouffer + Holm across 5 seeds) ===")
    for name, r in sorted(genm6_pooled["hypotheses_pooled"].items()):
        print(
            f"  {name:45s} pooled_p={r['pooled_p_stouffer']:.4g} "
            f"holm_p={r['holm_adjusted_p']:.4g} eff={r['mean_effect_size']:.3f} "
            f"{r['conclusion']}"
        )

    print("\nLoading canonical campaign...")
    campaign = load_campaign(Path(args.campaign_dir))
    campaign_stats = paired_campaign_stats(campaign)
    with open(out_dir / "campaign_paired_stats.json", "w") as f:
        json.dump(campaign_stats, f, indent=2, default=str)

    print("\n=== Campaign paired arm comparisons (Wilcoxon + Holm per domain) ===")
    for domain, comps in campaign_stats.items():
        print(f"-- {domain} --")
        for key, r in comps.items():
            print(
                f"  {key:22s} n={r['n_paired']:4d} err_a={r['mean_error_a']:.4f} "
                f"err_b={r['mean_error_b']:.4f} holm_p={r['holm_adjusted_p']:.4g} "
                f"d={r['cohens_d']:.3f} {r['conclusion']}"
            )

    print(f"\nWrote {out_dir}/genm6_pooled.json and {out_dir}/campaign_paired_stats.json")


if __name__ == "__main__":
    main()
