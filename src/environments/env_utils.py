from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def certainty_equivalent(
    mu: float,
    var: float,
    alpha: float,
    lambda_: float,
    effective_horizon: float = 1.0,
) -> float:
    """Approximate certainty equivalent under prospect-theory-style risk penalty.

    Shared by resource game and stock backtest for ``get_optimal_action``.
    """
    compounded_return = mu * effective_horizon
    if alpha <= 0:
        return compounded_return

    cumulative_risk = np.sqrt(max(var * effective_horizon, 0.0))
    risk_penalty = 0.5 * alpha * cumulative_risk
    ce = compounded_return - risk_penalty
    if ce < 0:
        ce *= lambda_
    return float(ce)


def nearest_positive_definite(
    matrix: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Nearest positive-definite matrix via eigenvalue clipping."""
    sym = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
