from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def compute_quality_score(
    action: NDArray[np.floating[Any]],
    means: NDArray[np.floating[Any]],
    variances: NDArray[np.floating[Any]],
) -> float:
    """Compute R_quality as the Sharpe ratio of the portfolio allocation.

    Args:
        action: Allocation weights over K channels (must sum to 1).
        means: Expected returns per channel in current regime.
        variances: Return variances per channel in current regime.

    Returns:
        Sharpe ratio of the portfolio. Higher is objectively better.
    """
    action = np.asarray(action, dtype=np.float64)
    port_return = float(np.dot(action, means))
    port_var = float(np.dot(action**2, variances))
    if port_var <= 0:
        return port_return
    return port_return / np.sqrt(port_var)


def check_dominance(
    action: NDArray[np.floating[Any]],
    means: NDArray[np.floating[Any]],
    variances: NDArray[np.floating[Any]],
    channel_names: list[str] | None = None,
    threshold: float = 0.99,
) -> tuple[bool, str]:
    """Check whether the allocation concentrates in a strictly dominated channel.

    A channel i is strictly dominated if there exists another channel j with
    mean_j >= mean_i and variance_j <= variance_i (with at least one strict inequality).

    Args:
        action: Allocation weights.
        means: Expected returns per channel.
        variances: Return variances per channel.
        channel_names: Optional names for violation messages.
        threshold: Weight above which a channel is considered concentrated.

    Returns:
        (passes, message) where passes is True if no dominance violation.
    """
    action = np.asarray(action, dtype=np.float64)
    K = len(action)
    names = channel_names or [str(i) for i in range(K)]

    concentrated = np.where(action > threshold)[0]
    if len(concentrated) != 1:
        return True, ""

    idx = int(concentrated[0])
    for j in range(K):
        if j == idx:
            continue
        weakly_better = means[j] >= means[idx] and variances[j] <= variances[idx]
        strictly_better = means[j] > means[idx] or variances[j] < variances[idx]
        if weakly_better and strictly_better:
            return False, (
                f"Channel '{names[idx]}' is strictly dominated by "
                f"channel '{names[j]}' (mean {means[j]:.4f} vs {means[idx]:.4f}, "
                f"var {variances[j]:.6f} vs {variances[idx]:.6f})"
            )

    return True, ""


def check_diversification(
    action: NDArray[np.floating[Any]],
    min_channels: int = 2,
    weight_threshold: float = 0.01,
) -> tuple[bool, str]:
    """Check whether the allocation is sufficiently diversified.

    Args:
        action: Allocation weights.
        min_channels: Minimum number of channels with nonzero allocation.
        weight_threshold: Weight below which a channel is considered inactive.

    Returns:
        (passes, message) where passes is True if diversification is met.
    """
    action = np.asarray(action, dtype=np.float64)
    active = int(np.sum(action > weight_threshold))
    if active < min_channels:
        return False, (
            f"Only {active} active channel(s); minimum required is {min_channels}"
        )
    return True, ""


def estimate_bankruptcy_prob(
    action: NDArray[np.floating[Any]],
    means: NDArray[np.floating[Any]],
    variances: NDArray[np.floating[Any]],
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate the probability that the portfolio loses all value in one round.

    Uses Monte Carlo: sample returns from N(port_mean, port_std), count
    fraction with total return <= -1 (i.e., wealth goes to zero).

    Args:
        action: Allocation weights.
        means: Expected returns per channel.
        variances: Return variances per channel.
        n_samples: Number of Monte Carlo samples.
        rng: Random generator for reproducibility.
    """
    action = np.asarray(action, dtype=np.float64)
    rng = rng or np.random.default_rng()

    port_mean = float(np.dot(action, means))
    port_var = float(np.dot(action**2, variances))
    if port_var <= 0:
        return 0.0

    port_std = np.sqrt(port_var)
    sim_returns = rng.normal(port_mean, port_std, size=n_samples)
    return float(np.mean(sim_returns <= -1.0))


def full_quality_check(
    action: NDArray[np.floating[Any]],
    means: NDArray[np.floating[Any]],
    variances: NDArray[np.floating[Any]],
    channel_names: list[str] | None = None,
    min_channels: int = 2,
    max_bankruptcy_prob: float = 0.05,
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Run all quality floor checks and return structured results.

    Returns:
        (passes, violations, details) where:
        - passes: True if all checks pass
        - violations: list of human-readable violation strings
        - details: dict with per-check results for auditing
    """
    violations: list[str] = []
    details: dict[str, Any] = {}

    dom_ok, dom_msg = check_dominance(action, means, variances, channel_names)
    details["dominance"] = {"passes": dom_ok, "message": dom_msg}
    if not dom_ok:
        violations.append(dom_msg)

    div_ok, div_msg = check_diversification(action, min_channels)
    details["diversification"] = {"passes": div_ok, "message": div_msg}
    if not div_ok:
        violations.append(div_msg)

    bankruptcy_prob = estimate_bankruptcy_prob(
        action, means, variances, n_samples, rng
    )
    bank_ok = bankruptcy_prob <= max_bankruptcy_prob
    bank_msg = "" if bank_ok else (
        f"Bankruptcy probability {bankruptcy_prob:.4f} exceeds "
        f"threshold {max_bankruptcy_prob:.4f}"
    )
    details["bankruptcy"] = {
        "passes": bank_ok,
        "probability": bankruptcy_prob,
        "threshold": max_bankruptcy_prob,
        "message": bank_msg,
    }
    if not bank_ok:
        violations.append(bank_msg)

    quality_score = compute_quality_score(action, means, variances)
    details["quality_score"] = quality_score

    return len(violations) == 0, violations, details
