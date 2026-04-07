#!/bin/bash
# scripts/azure/collect_results.sh
# Aggregate all experiment results into a single summary, regenerate paper
# figures, and copy key files to outputs/results/ for easy download.

set -euo pipefail

PROJECT_DIR="${HOME}/latent_proxy"
CONDA_DIR="${HOME}/miniconda3"
ENV_NAME="latent-proxy"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${PROJECT_DIR}"

RESULTS_DIR="outputs/results"
mkdir -p "${RESULTS_DIR}"/{figures,json}

log() { echo "[collect] $(date '+%H:%M:%S') $*"; }

log "Collecting results..."

# --------------------------------------------------------------------------
# Copy JSON bundles
# --------------------------------------------------------------------------
declare -A JSON_FILES=(
    ["gamma_scaling_1.5B"]="outputs/gamma_scaling/1.5B/gamma_study_results.json"
    ["gamma_scaling_3B"]="outputs/gamma_scaling/3B/gamma_study_results.json"
    ["gamma_scaling_7B"]="outputs/gamma_scaling/7B/gamma_study_results.json"
    ["gamma_scaling_14B"]="outputs/gamma_scaling/14B/gamma_study_results.json"
    ["dpo_full_study"]="outputs/dpo_study_azure/results/dpo_study_results.json"
    ["m4_evaluation"]="outputs/evaluation_m4_azure/results_bundle.json"
    ["m5_cross_domain"]="outputs/evaluation_m5_azure/cross_domain.json"
    ["m6_generalization"]="outputs/generalization_azure/results_bundle.json"
)

for NAME in "${!JSON_FILES[@]}"; do
    SRC="${JSON_FILES[$NAME]}"
    if [ -f "${SRC}" ]; then
        cp "${SRC}" "${RESULTS_DIR}/json/${NAME}.json"
        log "  Copied ${NAME}.json"
    else
        log "  WARNING: ${SRC} not found"
    fi
done

# --------------------------------------------------------------------------
# Copy figures
# --------------------------------------------------------------------------
find outputs/ -name "*.png" -newer outputs/ | while read -r FIG; do
    cp "${FIG}" "${RESULTS_DIR}/figures/" 2>/dev/null || true
done
log "Figures copied to ${RESULTS_DIR}/figures/"

# --------------------------------------------------------------------------
# Generate consolidated summary
# --------------------------------------------------------------------------
log "Generating consolidated summary..."

python - <<'PYEOF'
import json
import os
from pathlib import Path

results_dir = Path("outputs/results/json")
summary = {}

# Gamma scaling summary
for label in ["1.5B", "3B", "7B", "14B"]:
    p = results_dir / f"gamma_scaling_{label}.json"
    if p.exists():
        d = json.loads(p.read_text())
        summary[f"gamma_{label}"] = {
            cond: {
                "mean_alignment": d[cond]["mean_alignment"],
                "mean_gamma_error": d[cond]["mean_gamma_error"],
            }
            for cond in ["base", "dpo_phase1", "dpo_phase2"]
            if cond in d
        }

# DPO full study
p = results_dir / "dpo_full_study.json"
if p.exists():
    d = json.loads(p.read_text())
    summary["dpo_full_study"] = {
        env: {
            cond: v["mean_alignment"]
            for cond, v in conds.items()
        }
        for env, conds in d.get("environments", {}).items()
    }

# M4 evaluation
p = results_dir / "m4_evaluation.json"
if p.exists():
    d = json.loads(p.read_text())
    variants = d.get("full_evaluation", {}).get("variants", {})
    summary["m4_evaluation"] = {
        v: {
            "alignment": b.get("mean_alignment_active"),
            "gamma_mae": b.get("mean_gamma_mae"),
            "error_reduction_pct": b.get("efficiency", {}).get("error_reduction_pct"),
            "q_violation": b.get("quality_floor_violation_rate"),
        }
        for v, b in variants.items()
    }

# M5 cross-domain
p = results_dir / "m5_cross_domain.json"
if p.exists():
    d = json.loads(p.read_text())
    summary["m5_cross_domain"] = {
        "generic": d.get("generic", {}).get("mean"),
        "within_domain": d.get("within_domain", {}).get("mean"),
        "cross_domain": d.get("cross_domain", {}).get("mean"),
        "n_users": d.get("n_users"),
    }

# M6 generalization
p = results_dir / "m6_generalization.json"
if p.exists():
    d = json.loads(p.read_text())
    summary["m6_generalization"] = {
        "hypothesis_results": {
            h: [
                {"hypothesis": r["hypothesis"], "p_value": r["p_value"], "conclusion": r["conclusion"]}
                for r in tests
            ]
            for h, tests in d.get("hypothesis_results", {}).items()
        },
        "stability_icc": {
            domain: s["icc"]
            for domain, s in d.get("stability", {}).items()
        },
    }

out_path = Path("outputs/results/consolidated_summary.json")
out_path.write_text(json.dumps(summary, indent=2))
print(f"Wrote {out_path}")

# Print key metrics
print("\n========== RESULTS SUMMARY ==========")
if "gamma_scaling" := {k: v for k, v in summary.items() if k.startswith("gamma_")}:
    print("\nGamma Scaling (DPO Phase 2 gamma error):")
    for label in ["1.5B", "3B", "7B", "14B"]:
        key = f"gamma_{label}"
        if key in summary and "dpo_phase2" in summary[key]:
            err = summary[key]["dpo_phase2"]["mean_gamma_error"]
            align = summary[key]["dpo_phase2"]["mean_alignment"]
            print(f"  {label:6s}  gamma_err={err:.3f}  alignment={align:.3f}")

if "m4_evaluation" in summary:
    print("\nM4 Evaluation (analytical):")
    for v, m in summary["m4_evaluation"].items():
        print(f"  variant {v}: align={m['alignment']:.3f}  "
              f"gamma_mae={m['gamma_mae']:.3f}  "
              f"err_reduct={m['error_reduction_pct']:.1f}%  "
              f"q_viol={m['q_violation']:.3f}")

if "m5_cross_domain" in summary:
    m = summary["m5_cross_domain"]
    print(f"\nM5 Cross-Domain: generic={m['generic']:.3f}  "
          f"within={m['within_domain']:.3f}  cross={m['cross_domain']:.3f}")
PYEOF

log "Results consolidated at outputs/results/consolidated_summary.json"
log "Figures at outputs/results/figures/"
log "JSON files at outputs/results/json/"
log ""
log "To download results:"
log "  scp -r azureuser@20.57.36.243:~/latent_proxy/outputs/results ./azure_results"
