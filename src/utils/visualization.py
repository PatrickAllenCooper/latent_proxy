from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.agents.elicitation_loop import ElicitationResult
from src.evaluation.ablation_runner import AblationResults
from src.evaluation.experiment_runner import TransferExperimentResult


def plot_recovery_curves(
    active_results: list[ElicitationResult],
    random_results: list[ElicitationResult] | None = None,
    *,
    max_curves: int = 5,
) -> Figure:
    """Posterior variance / proxy recovery vs round for active (and optional random)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def _plot_panel(ax: Any, results: list[ElicitationResult], label: str) -> None:
        for i, r in enumerate(results[:max_curves]):
            traj = r.variance_trajectory
            if not traj:
                continue
            xs = np.arange(len(traj))
            total_var = [sum(traj[t].values()) for t in range(len(traj))]
            ax.plot(xs, total_var, alpha=0.7, label=f"{label} user {i}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Sum posterior variance")
        ax.set_title(label)
        ax.legend(fontsize=7)

    _plot_panel(axes[0], active_results, "Active")
    if random_results:
        _plot_panel(axes[1], random_results, "Random")
    else:
        axes[1].set_visible(False)
    fig.tight_layout()
    return fig


def plot_ablation_sweep(
    ablation: AblationResults,
    metric_key: str = "mean_active_error",
) -> Figure:
    """Metric vs ablation parameter with simple point plot."""
    xs: list[float] = []
    ys: list[float] = []
    for k in ablation.values:
        sk = str(k)
        if sk not in ablation.metrics_by_value:
            continue
        m = ablation.metrics_by_value[sk]
        if metric_key not in m:
            continue
        try:
            xs.append(float(k))
        except (TypeError, ValueError):
            xs.append(float(xs[-1] + 1) if xs else 0.0)
        ys.append(float(m[metric_key]))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(ablation.param_name)
    ax.set_ylabel(metric_key)
    ax.set_title(f"Ablation: {metric_key}")
    fig.tight_layout()
    return fig


def plot_transfer_comparison(transfer: TransferExperimentResult) -> Figure:
    """Bar chart: Generic / Within-Domain / Cross-Domain alignment means."""
    labels = ["Generic", "Within-Domain", "Cross-Domain"]
    means = [
        transfer.generic["mean"],
        transfer.within_domain["mean"],
        transfer.cross_domain["mean"],
    ]
    lows = [
        transfer.generic["mean"] - transfer.generic["ci_low"],
        transfer.within_domain["mean"] - transfer.within_domain["ci_low"],
        transfer.cross_domain["mean"] - transfer.cross_domain["ci_low"],
    ]
    highs = [
        transfer.generic["ci_high"] - transfer.generic["mean"],
        transfer.within_domain["ci_high"] - transfer.within_domain["mean"],
        transfer.cross_domain["ci_high"] - transfer.cross_domain["mean"],
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, means, yerr=[lows, highs], capsize=4, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Alignment vs true optimal")
    ax.set_title("Transfer experiment (variant A -> B)")
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    return fig


def format_results_table(full_eval: dict[str, Any], targets: dict[str, float] | None = None) -> str:
    """Markdown table of per-variant metrics vs optional targets."""
    targets = targets or {
        "alignment": 0.8,
        "gamma_mae": 0.1,
        "efficiency_pct": 30.0,
        "quality_violation": 0.0,
    }
    lines = [
        "| Variant | Alignment | Gamma MAE | Err reduction % | Q-violation |",
        "|---------|-----------|-----------|-----------------|-------------|",
    ]
    variants = full_eval.get("variants", {})
    for name, block in variants.items():
        eff = block.get("efficiency", {})
        lines.append(
            f"| {name} | {block.get('mean_alignment_active', 0):.3f} | "
            f"{block.get('mean_gamma_mae', 0):.3f} | "
            f"{eff.get('error_reduction_pct', 0):.1f} | "
            f"{block.get('quality_floor_violation_rate', 0):.3f} |",
        )
    lines.append("")
    lines.append("| Target | Value |")
    lines.append("|--------|-------|")
    for k, v in targets.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def save_results(data: dict[str, Any], path: str | Path) -> None:
    """JSON export (numpy arrays converted to lists)."""

    def _convert(obj: Any) -> Any:
        if is_dataclass(obj) and not isinstance(obj, type):
            return _convert(asdict(obj))
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, tuple):
            return [_convert(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_convert(data), f, indent=2)
