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
from src.evaluation.statistical_analysis import HypothesisTestResult


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


def plot_hypothesis_panel(
    hypothesis_results: dict[str, list[HypothesisTestResult]],
) -> Figure:
    """2x2 grid summarizing H1-H4 test results with significance markers."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [("H1", axes[0, 0]), ("H2", axes[0, 1]), ("H3", axes[1, 0]), ("H4", axes[1, 1])]

    for h_key, ax in panels:
        results = hypothesis_results.get(h_key, [])
        if not results:
            ax.set_title(f"{h_key}: no data")
            ax.set_visible(False)
            continue
        labels = [r.hypothesis.replace(f"{h_key}_", "") for r in results]
        effects = [r.effect_size for r in results]
        colors = ["#55a868" if r.conclusion == "reject_null" else "#c44e52" for r in results]
        x = np.arange(len(labels))
        ax.barh(x, effects, color=colors, edgecolor="white")
        for j, r in enumerate(results):
            marker = "*" if r.conclusion == "reject_null" else ""
            ax.text(
                effects[j] + 0.02, j,
                f"p={r.p_value:.3f}{marker}", va="center", fontsize=8,
            )
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Effect size")
        ax.set_title(h_key)

    fig.tight_layout()
    return fig


def plot_parameter_transfer_heatmap(
    per_parameter_transfer: dict[str, dict[str, float]],
) -> Figure:
    """Heatmap: domain pairs (rows) x parameters (columns), cell = cross-domain MAE."""
    params = sorted(per_parameter_transfer.keys())
    if not params:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    pair_keys = sorted(next(iter(per_parameter_transfer.values())).keys())
    data = np.array([
        [per_parameter_transfer[p].get(pk, 0.0) for p in params]
        for pk in pair_keys
    ])

    fig, ax = plt.subplots(figsize=(6, max(3, len(pair_keys) * 0.6 + 1)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(params)))
    ax.set_xticklabels(params, fontsize=9)
    ax.set_yticks(np.arange(len(pair_keys)))
    ax.set_yticklabels(pair_keys, fontsize=9)
    for i in range(len(pair_keys)):
        for j in range(len(params)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="MAE")
    ax.set_title("Per-parameter cross-domain transfer error")
    fig.tight_layout()
    return fig


def plot_transfer_matrix(
    pairwise_alignment: dict[str, float],
    domain_order: list[str] | None = None,
) -> Figure:
    """NxN alignment matrix across domain pairs."""
    if domain_order is None:
        names: set[str] = set()
        for key in pairwise_alignment:
            parts = key.split("->")
            if len(parts) == 2:
                names.update(parts)
        domain_order = sorted(names)

    n = len(domain_order)
    mat = np.full((n, n), np.nan)
    idx_map = {d: i for i, d in enumerate(domain_order)}
    for key, val in pairwise_alignment.items():
        parts = key.split("->")
        if len(parts) == 2 and parts[0] in idx_map and parts[1] in idx_map:
            mat[idx_map[parts[0]], idx_map[parts[1]]] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(domain_order, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(domain_order, fontsize=9)
    ax.set_xlabel("Target")
    ax.set_ylabel("Source")
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Cross-domain alignment")
    ax.set_title("Cross-domain transfer alignment matrix")
    fig.tight_layout()
    return fig


def plot_stability_scatter(
    session1: dict[str, list[float]],
    session2: dict[str, list[float]],
    param_names: list[str] | None = None,
) -> Figure:
    """Per-parameter test-retest scatter (session 1 vs session 2)."""
    if param_names is None:
        param_names = sorted(session1.keys())
    n_params = len(param_names)
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]

    for ax, p in zip(axes, param_names):
        s1 = np.asarray(session1.get(p, []))
        s2 = np.asarray(session2.get(p, []))
        mn = min(len(s1), len(s2))
        if mn == 0:
            ax.set_title(f"{p}: no data")
            continue
        ax.scatter(s1[:mn], s2[:mn], alpha=0.7, edgecolor="white", s=40)
        lo = min(float(s1[:mn].min()), float(s2[:mn].min())) - 0.05
        hi = max(float(s1[:mn].max()), float(s2[:mn].max())) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel(f"{p} (session 1)")
        ax.set_ylabel(f"{p} (session 2)")
        ax.set_title(p)

    fig.suptitle("Theta stability: test-retest", fontsize=12)
    fig.tight_layout()
    return fig


def plot_dpo_comparison(
    per_env: dict[str, dict[str, Any]],
) -> Figure:
    """Bar chart: alignment by condition (analytical / base / P1 / P2) per environment."""
    env_names = list(per_env.keys())
    all_conditions = set()
    for conds in per_env.values():
        all_conditions.update(conds.keys())
    condition_order = [c for c in ["analytical", "base", "dpo_phase1", "dpo_phase2"] if c in all_conditions]

    n_envs = len(env_names)
    n_conds = len(condition_order)
    x = np.arange(n_envs)
    width = 0.8 / max(n_conds, 1)
    colors = ["#4c72b0", "#c44e52", "#dd8452", "#55a868"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, cond in enumerate(condition_order):
        vals = []
        for env_name in env_names:
            cr = per_env[env_name].get(cond)
            if cr is None:
                vals.append(0.0)
            elif hasattr(cr, "mean_alignment"):
                vals.append(cr.mean_alignment)
            elif isinstance(cr, dict):
                vals.append(cr.get("mean_alignment", 0.0))
            else:
                vals.append(0.0)
        ax.bar(x + j * width, vals, width, label=cond, color=colors[j % len(colors)])

    ax.set_xticks(x + width * (n_conds - 1) / 2)
    ax.set_xticklabels(env_names, fontsize=10)
    ax.set_ylabel("Mean alignment score")
    ax.set_title("DPO elicitation study: alignment by condition")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.2, 1.05)
    fig.tight_layout()
    return fig


def plot_convergence_by_condition(
    per_env: dict[str, dict[str, Any]],
) -> Figure:
    """Per-round alignment curves showing convergence for each condition."""
    env_names = list(per_env.keys())
    n_envs = len(env_names)
    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 4), squeeze=False)

    for col, env_name in enumerate(env_names):
        ax = axes[0, col]
        conditions = per_env[env_name]
        for cond_name, cr in conditions.items():
            per_round = None
            if hasattr(cr, "per_round_alignments"):
                per_round = cr.per_round_alignments
            elif isinstance(cr, dict):
                per_round = cr.get("per_round_alignments")
            if not per_round or not any(per_round):
                continue
            max_len = max(len(r) for r in per_round if r)
            if max_len == 0:
                continue
            means = []
            for t in range(max_len):
                vals = [r[t] for r in per_round if len(r) > t]
                means.append(float(np.mean(vals)))
            ax.plot(range(1, max_len + 1), means, marker="o", markersize=4, label=cond_name)
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean alignment")
        ax.set_title(env_name)
        ax.legend(fontsize=7)
        ax.set_ylim(-0.3, 1.05)

    fig.suptitle("Convergence by condition", fontsize=12)
    fig.tight_layout()
    return fig


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
