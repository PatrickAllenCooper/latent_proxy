"""Measure whether temperature=0.1 remains appropriate under utility_form="return_normalized".

The fixed softmax temperature used everywhere (SyntheticUser.choose,
_choice_log_likelihood, EIG) was implicitly calibrated against the OLD
absolute-wealth-scale EU-difference magnitudes. Under return_normalized,
typical |EU_a - EU_b| magnitudes are computed over dimensionless returns
instead of dollar-denominated wealth, so this must be re-verified rather
than assumed.

Read-only measurement script -- makes no production code changes. Modeled
on scripts/analyze_eig_stability.py's sampling pattern.

Usage:
    .venv/bin/python scripts/calibrate_temperature.py --n-thetas 200 --n-scenarios 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.generalization_protocol import DOMAIN_FACTORIES, SCENARIO_LIBRARIES
from src.training.synthetic_users import SyntheticUser, SyntheticUserSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("calibrate_temperature")

TEMPERATURE_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sample_abs_eu_diffs(
    domain: str, utility_form: str, n_thetas: int, n_scenarios: int, seed: int,
) -> dict[str, np.ndarray]:
    """|EU_a - EU_b| across n_thetas x (scenarios drawn per theta), by target_param.

    Uses common random numbers between the option_a/option_b evaluations
    (reseed to the same value before each), matching the pattern
    _choice_probability/_choice_log_likelihood use during real posterior
    updates and EIG -- the quantity that actually determines choice-model
    informativeness, not the raw ground-truth-user choice-generation noise
    (which advances one shared _rng across a whole session and would
    swamp genuine signal with independent-sampling MC noise here).

    Returned per target_param: the pool is dominated by near-null
    alpha/lambda_ pairs (consistent with the earlier Fisher-information
    audit, outputs/diagnostics/identifiability.md) while gamma-targeted
    pairs are strongly informative -- an "all scenarios" median would be
    misleading since EIG-driven selection picks the best candidate, not
    a random one.
    """
    sampler = SyntheticUserSampler(seed=seed)
    env_factory = DOMAIN_FACTORIES[domain]
    lib_factory = SCENARIO_LIBRARIES[domain]

    diffs_by_param: dict[str, list[float]] = {"gamma": [], "alpha": [], "lambda_": [], "all": []}
    n_per_param = max(n_scenarios // 3, 3)

    for i in range(n_thetas):
        ut = sampler.sample()
        env = env_factory()
        env.reset(seed=seed + i)
        library = lib_factory(seed + i + 4000)
        scenarios = library.generate_all(env, n_per_param)
        if not scenarios:
            continue

        for j, scenario in enumerate(scenarios):
            common_seed = seed + i * 10_000 + j
            user_a = SyntheticUser(
                ut, temperature=0.1, seed=common_seed, utility_form=utility_form,
            )
            eu_a = user_a.evaluate_for_query(
                scenario.option_a, scenario.channel_means, scenario.channel_variances,
                scenario.current_wealth, scenario.rounds_remaining,
                multiperiod_horizon=scenario.multiperiod_horizon,
            )
            user_b = SyntheticUser(
                ut, temperature=0.1, seed=common_seed, utility_form=utility_form,
            )
            eu_b = user_b.evaluate_for_query(
                scenario.option_b, scenario.channel_means, scenario.channel_variances,
                scenario.current_wealth, scenario.rounds_remaining,
                multiperiod_horizon=scenario.multiperiod_horizon,
            )
            d = abs(eu_a - eu_b)
            diffs_by_param["all"].append(d)
            key = scenario.target_param if scenario.target_param in diffs_by_param else "all"
            if key != "all":
                diffs_by_param[key].append(d)

    return {k: np.array(v) for k, v in diffs_by_param.items()}


def informativeness_band(diffs: np.ndarray, temperature: float) -> dict[str, float] | None:
    """Softmax-implied P(correct) at this temperature; summary of the band."""
    if len(diffs) == 0:
        return None
    p_correct = np.maximum(sigmoid(diffs / temperature), 1.0 - sigmoid(diffs / temperature))
    return {
        "median_p_correct": float(np.median(p_correct)),
        "p90_p_correct": float(np.percentile(p_correct, 90)),
        "frac_near_coinflip_lt_0.6": float(np.mean(p_correct < 0.6)),
        "frac_saturated_gt_0.95": float(np.mean(p_correct > 0.95)),
    }


def percentiles(diffs: np.ndarray) -> dict[str, float | None]:
    if len(diffs) == 0:
        return {"p10": None, "median": None, "p90": None, "max": None}
    return {
        "p10": float(np.percentile(diffs, 10)),
        "median": float(np.median(diffs)),
        "p90": float(np.percentile(diffs, 90)),
        "max": float(np.max(diffs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-thetas", type=int, default=200)
    parser.add_argument("--n-scenarios", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/diagnostics")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for domain in DOMAIN_FACTORIES:
        logger.info("Domain %s: sampling absolute-mode EU diffs...", domain)
        diffs_abs = sample_abs_eu_diffs(
            domain, "absolute", args.n_thetas, args.n_scenarios, args.seed,
        )
        logger.info("Domain %s: sampling return_normalized-mode EU diffs...", domain)
        diffs_rn = sample_abs_eu_diffs(
            domain, "return_normalized", args.n_thetas, args.n_scenarios, args.seed,
        )

        domain_result: dict[str, Any] = {
            "n_pairs_absolute": {k: len(v) for k, v in diffs_abs.items()},
            "n_pairs_return_normalized": {k: len(v) for k, v in diffs_rn.items()},
            "diff_percentiles_absolute": {k: percentiles(v) for k, v in diffs_abs.items()},
            "diff_percentiles_return_normalized": {k: percentiles(v) for k, v in diffs_rn.items()},
            "temperature_grid": {},
        }

        for T in TEMPERATURE_GRID:
            domain_result["temperature_grid"][str(T)] = {
                "absolute": {k: informativeness_band(v, T) for k, v in diffs_abs.items()},
                "return_normalized": {k: informativeness_band(v, T) for k, v in diffs_rn.items()},
            }

        results[domain] = domain_result

        for param in ("gamma", "alpha", "lambda_", "all"):
            b_abs = domain_result["temperature_grid"]["0.1"]["absolute"][param]
            b_rn = domain_result["temperature_grid"]["0.1"]["return_normalized"][param]
            logger.info(
                "%s/%s: absolute@T=0.1 median=%.3f p90=%.3f | return_normalized@T=0.1 median=%.3f p90=%.3f",
                domain, param,
                b_abs["median_p_correct"] if b_abs else float("nan"),
                b_abs["p90_p_correct"] if b_abs else float("nan"),
                b_rn["median_p_correct"] if b_rn else float("nan"),
                b_rn["p90_p_correct"] if b_rn else float("nan"),
            )

    with open(out_dir / "temperature_calibration.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Temperature calibration @ T=0.1 (median / p90 of P(correct), by target_param) ===")
    print(f"{'domain':14s} {'param':8s} {'form':18s} {'median':>8s} {'p90':>8s}")
    for domain, r in results.items():
        for param in ("gamma", "alpha", "lambda_", "all"):
            for form in ("absolute", "return_normalized"):
                cell = r["temperature_grid"]["0.1"][form][param]
                med = cell["median_p_correct"] if cell else float("nan")
                p90 = cell["p90_p_correct"] if cell else float("nan")
                print(f"{domain:14s} {param:8s} {form:18s} {med:8.3f} {p90:8.3f}")

    logger.info("Wrote %s", out_dir / "temperature_calibration.json")


if __name__ == "__main__":
    main()
