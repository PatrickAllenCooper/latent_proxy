"""Statistical hypothesis testing for the generalization study (H1-H4).

Provides paired non-parametric tests, effect sizes, multiple comparison
correction, and ICC for theta stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class HypothesisTestResult:
    """Result of a single statistical hypothesis test."""

    hypothesis: str
    statistic: float
    p_value: float
    effect_size: float
    ci: tuple[float, float] | None
    method_name: str
    conclusion: str
    details: dict[str, Any] | None = None


def rank_biserial_r(statistic: float, n: int) -> float:
    """Rank-biserial correlation (effect size for Wilcoxon signed-rank)."""
    if n <= 0:
        return 0.0
    t_max = n * (n + 1) / 2
    if t_max == 0:
        return 0.0
    return float(1.0 - 2.0 * statistic / t_max)


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for paired samples."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-15:
        return 0.0
    return float(np.mean(diff) / sd)


def _wilcoxon_safe(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """scipy.stats.wilcoxon with fallback for zero-diff or tiny samples."""
    diff = x - y
    if np.all(diff == 0):
        return 0.0, 1.0
    if len(diff) < 6:
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(x, y, alternative=alternative)
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return 0.0, 1.0


def holm_bonferroni(p_values: Sequence[float]) -> NDArray[np.float64]:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    ps = np.asarray(p_values, dtype=np.float64)
    m = len(ps)
    if m == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(ps)
    adjusted = np.empty(m, dtype=np.float64)
    cummax = 0.0
    for rank, idx in enumerate(order):
        corrected = ps[idx] * (m - rank)
        corrected = max(corrected, cummax)
        cummax = corrected
        adjusted[idx] = min(corrected, 1.0)
    return adjusted


def _conclude(p: float, alpha: float = 0.05) -> str:
    return "reject_null" if p < alpha else "fail_to_reject"


def run_test_h1_within_domain(
    within_scores: Sequence[float],
    generic_scores: Sequence[float],
    *,
    hypothesis_label: str = "H1",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """H1: within-domain personalization > generic (paired Wilcoxon, one-sided)."""
    w = np.asarray(within_scores, dtype=np.float64)
    g = np.asarray(generic_scores, dtype=np.float64)
    stat, pval = _wilcoxon_safe(w, g, alternative="greater")
    n = len(w)
    es = rank_biserial_r(stat, n)
    d = cohens_d_paired(within_scores, generic_scores)
    return HypothesisTestResult(
        hypothesis=hypothesis_label,
        statistic=stat,
        p_value=pval,
        effect_size=es,
        ci=None,
        method_name="Wilcoxon signed-rank (greater)",
        conclusion=_conclude(pval, alpha),
        details={"cohens_d": d, "n": n},
    )


def run_test_h2_cross_domain(
    cross_scores: Sequence[float],
    generic_scores: Sequence[float],
    within_scores: Sequence[float] | None = None,
    *,
    hypothesis_label: str = "H2",
    alpha: float = 0.05,
) -> list[HypothesisTestResult]:
    """H2: cross > generic AND cross < within (two one-sided Wilcoxon tests)."""
    results: list[HypothesisTestResult] = []
    c = np.asarray(cross_scores, dtype=np.float64)
    g = np.asarray(generic_scores, dtype=np.float64)
    n = len(c)

    stat_cg, pval_cg = _wilcoxon_safe(c, g, alternative="greater")
    es_cg = rank_biserial_r(stat_cg, n)
    results.append(HypothesisTestResult(
        hypothesis=f"{hypothesis_label}_cross_gt_generic",
        statistic=stat_cg,
        p_value=pval_cg,
        effect_size=es_cg,
        ci=None,
        method_name="Wilcoxon signed-rank (greater)",
        conclusion=_conclude(pval_cg, alpha),
        details={"n": n},
    ))

    if within_scores is not None:
        w = np.asarray(within_scores, dtype=np.float64)
        stat_cw, pval_cw = _wilcoxon_safe(w, c, alternative="greater")
        es_cw = rank_biserial_r(stat_cw, n)
        results.append(HypothesisTestResult(
            hypothesis=f"{hypothesis_label}_within_gt_cross",
            statistic=stat_cw,
            p_value=pval_cw,
            effect_size=es_cw,
            ci=None,
            method_name="Wilcoxon signed-rank (greater)",
            conclusion=_conclude(pval_cw, alpha),
            details={"n": n},
        ))

    return results


def run_test_h3_parameter_transfer(
    per_pair_param_mae: dict[tuple[str, str], dict[str, float]],
    structural_similarity: dict[tuple[str, str], float] | None = None,
    *,
    hypothesis_label: str = "H3",
) -> dict[str, HypothesisTestResult]:
    """H3: gamma transfers better than alpha/lambda; transfer correlates with similarity.

    ``per_pair_param_mae`` maps (source, target) -> {"gamma": mae, "alpha": mae, ...}.
    ``structural_similarity`` maps (source, target) -> float (0-1, higher = more similar).
    """
    results: dict[str, HypothesisTestResult] = {}
    gamma_maes: list[float] = []
    other_maes: list[float] = []
    for _, mae_dict in per_pair_param_mae.items():
        gamma_maes.append(mae_dict.get("gamma", 0.0))
        alpha_mae = mae_dict.get("alpha", 0.0)
        lambda_mae = mae_dict.get("lambda_", 0.0)
        other_maes.append((alpha_mae + lambda_mae) / 2.0)

    mean_gamma = float(np.mean(gamma_maes)) if gamma_maes else 0.0
    mean_other = float(np.mean(other_maes)) if other_maes else 0.0
    d = cohens_d_paired(other_maes, gamma_maes) if len(gamma_maes) >= 2 else 0.0

    results["gamma_vs_other"] = HypothesisTestResult(
        hypothesis=f"{hypothesis_label}_gamma_transfers_better",
        statistic=mean_gamma,
        p_value=1.0,
        effect_size=d,
        ci=None,
        method_name="descriptive (mean MAE comparison)",
        conclusion="gamma_lower" if mean_gamma < mean_other else "no_difference",
        details={"mean_gamma_mae": mean_gamma, "mean_other_mae": mean_other},
    )

    if structural_similarity is not None and len(structural_similarity) >= 3:
        pairs = sorted(structural_similarity.keys())
        sim_vals = [structural_similarity[p] for p in pairs]
        total_mae = [
            sum(per_pair_param_mae.get(p, {}).values())
            for p in pairs
        ]
        if len(sim_vals) >= 3:
            corr, pval = stats.spearmanr(sim_vals, total_mae)
            results["similarity_correlation"] = HypothesisTestResult(
                hypothesis=f"{hypothesis_label}_similarity_correlates",
                statistic=float(corr),
                p_value=float(pval),
                effect_size=float(corr),
                ci=None,
                method_name="Spearman correlation",
                conclusion=_conclude(float(pval)),
                details={"n_pairs": len(pairs)},
            )

    return results


def run_test_h4_active_vs_random(
    active_errors: Sequence[float],
    random_errors: Sequence[float],
    *,
    hypothesis_label: str = "H4",
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """H4: active learning converges faster (lower error) than random."""
    r = np.asarray(random_errors, dtype=np.float64)
    a = np.asarray(active_errors, dtype=np.float64)
    n = len(a)
    stat, pval = _wilcoxon_safe(r, a, alternative="greater")
    es = rank_biserial_r(stat, n)
    d = cohens_d_paired(list(random_errors), list(active_errors))
    return HypothesisTestResult(
        hypothesis=hypothesis_label,
        statistic=stat,
        p_value=pval,
        effect_size=es,
        ci=None,
        method_name="Wilcoxon signed-rank (random > active)",
        conclusion=_conclude(pval, alpha),
        details={"cohens_d": d, "n": n},
    )


def icc_2_1(
    session1: NDArray[np.float64],
    session2: NDArray[np.float64],
) -> float:
    """ICC(2,1) -- two-way random, single measures (ANOVA-based).

    Rows = subjects, two columns (sessions).
    """
    s1 = np.asarray(session1, dtype=np.float64)
    s2 = np.asarray(session2, dtype=np.float64)
    n = len(s1)
    if n < 2:
        return 0.0
    k = 2
    data = np.column_stack([s1, s2])
    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_rows = k * float(np.sum((row_means - grand_mean) ** 2))
    ss_cols = n * float(np.sum((col_means - grand_mean) ** 2))
    ss_total = float(np.sum((data - grand_mean) ** 2))
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / max(n - 1, 1)
    ms_error = ss_error / max((n - 1) * (k - 1), 1)
    ms_cols = ss_cols / max(k - 1, 1)

    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if abs(denom) < 1e-15:
        return 0.0
    return float((ms_rows - ms_error) / denom)
