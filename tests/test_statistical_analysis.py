from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.statistical_analysis import (
    HypothesisTestResult,
    cohens_d_paired,
    holm_bonferroni,
    icc_2_1,
    rank_biserial_r,
    run_test_h1_within_domain,
    run_test_h2_cross_domain,
    run_test_h3_parameter_transfer,
    run_test_h4_active_vs_random,
)


def test_h1_returns_valid_result():
    within = [0.9, 0.85, 0.88, 0.92, 0.87, 0.91]
    generic = [0.4, 0.35, 0.42, 0.38, 0.41, 0.39]
    r = run_test_h1_within_domain(within, generic)
    assert isinstance(r, HypothesisTestResult)
    assert r.hypothesis == "H1"
    assert 0.0 <= r.p_value <= 1.0
    assert np.isfinite(r.effect_size)
    assert r.conclusion == "reject_null"


def test_h1_no_signal():
    same = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    r = run_test_h1_within_domain(same, same)
    assert r.conclusion == "fail_to_reject"


def test_h2_returns_two_results():
    cross = [0.7, 0.65, 0.72, 0.68, 0.71, 0.69]
    generic = [0.4, 0.35, 0.42, 0.38, 0.41, 0.39]
    within = [0.9, 0.85, 0.88, 0.92, 0.87, 0.91]
    results = run_test_h2_cross_domain(cross, generic, within)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, HypothesisTestResult)
        assert 0.0 <= r.p_value <= 1.0


def test_h3_gamma_vs_other():
    per_pair = {
        ("game_a", "stock"): {"gamma": 0.05, "alpha": 0.20, "lambda_": 0.18},
        ("game_a", "supply_chain"): {"gamma": 0.06, "alpha": 0.25, "lambda_": 0.22},
    }
    results = run_test_h3_parameter_transfer(per_pair)
    assert "gamma_vs_other" in results
    r = results["gamma_vs_other"]
    assert r.details is not None
    assert r.details["mean_gamma_mae"] < r.details["mean_other_mae"]
    assert r.conclusion == "gamma_lower"


def test_h4_active_better():
    active = [0.10, 0.12, 0.09, 0.11, 0.08, 0.10]
    random = [0.30, 0.28, 0.32, 0.29, 0.31, 0.27]
    r = run_test_h4_active_vs_random(active, random)
    assert isinstance(r, HypothesisTestResult)
    assert r.conclusion == "reject_null"


def test_holm_bonferroni_correction():
    raw = [0.01, 0.04, 0.03, 0.50]
    adjusted = holm_bonferroni(raw)
    assert len(adjusted) == 4
    assert all(a >= r for a, r in zip(adjusted, raw))
    assert all(0 <= a <= 1.0 for a in adjusted)


def test_icc_perfect_agreement():
    s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert icc_2_1(s1, s2) == pytest.approx(1.0, abs=0.01)


def test_icc_no_agreement():
    s1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    icc = icc_2_1(s1, s2)
    assert icc < 0.5


def test_effect_sizes_finite():
    assert np.isfinite(rank_biserial_r(10.0, 20))
    assert np.isfinite(cohens_d_paired([1, 2, 3], [4, 5, 6]))
