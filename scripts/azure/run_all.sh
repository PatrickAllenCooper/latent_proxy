#!/bin/bash
# scripts/azure/run_all.sh
# Master orchestrator for the full Azure experiment suite.
#
# Execution order:
#   1. Phase 2 (gamma scaling)  +  Phase 4 (analytical) -- in parallel
#   2. Phase 3 (full DPO study) -- after gamma scaling completes
#   3. Phase 5 (collect results) -- after everything completes
#
# Usage:
#   Inside a tmux session:
#     tmux new -s experiments
#     bash scripts/azure/run_all.sh 2>&1 | tee logs/run_all.log
#
# Prerequisites: bash scripts/azure/setup.sh must have been run first.

set -euo pipefail

PROJECT_DIR="${HOME}/latent_proxy"
CONDA_DIR="${HOME}/miniconda3"
ENV_NAME="latent-proxy"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${PROJECT_DIR}"

mkdir -p logs

log() { echo "[run_all] $(date '+%H:%M:%S') $*"; }

START_TIME=$(date '+%s')

log "======================================================"
log "Azure Experiment Suite -- $(date)"
log "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//') "
log "CPUs: $(nproc)"
log "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
log "======================================================"

# --------------------------------------------------------------------------
# Phase 2 + Phase 4 in parallel
# --------------------------------------------------------------------------
log "Starting Phase 2 (gamma scaling, GPU) and Phase 4 (analytical, CPU) in parallel..."

bash scripts/azure/run_gamma_scaling.sh \
    > logs/phase2_gamma_scaling.log 2>&1 &
PID_P2=$!
log "Phase 2 started (PID ${PID_P2})"

bash scripts/azure/run_analytical.sh \
    > logs/phase4_analytical.log 2>&1 &
PID_P4=$!
log "Phase 4 started (PID ${PID_P4})"

# --------------------------------------------------------------------------
# Wait for both to complete
# --------------------------------------------------------------------------
wait "${PID_P2}"
P2_STATUS=$?
if [ "${P2_STATUS}" -eq 0 ]; then
    log "Phase 2 (gamma scaling) COMPLETE"
else
    log "WARNING: Phase 2 exited with status ${P2_STATUS} -- check logs/phase2_gamma_scaling.log"
fi

wait "${PID_P4}"
P4_STATUS=$?
if [ "${P4_STATUS}" -eq 0 ]; then
    log "Phase 4 (analytical) COMPLETE"
else
    log "WARNING: Phase 4 exited with status ${P4_STATUS} -- check logs/phase4_analytical.log"
fi

# --------------------------------------------------------------------------
# Phase 3: Full DPO study (uses best model from Phase 2)
# --------------------------------------------------------------------------
log "Starting Phase 3 (full DPO study)..."
bash scripts/azure/run_dpo_full.sh \
    > logs/phase3_dpo_full.log 2>&1
P3_STATUS=$?
if [ "${P3_STATUS}" -eq 0 ]; then
    log "Phase 3 (DPO full study) COMPLETE"
else
    log "WARNING: Phase 3 exited with status ${P3_STATUS} -- check logs/phase3_dpo_full.log"
fi

# --------------------------------------------------------------------------
# Phase 5: Collect results
# --------------------------------------------------------------------------
log "Starting Phase 5 (collect results)..."
bash scripts/azure/collect_results.sh
log "Phase 5 (results collection) COMPLETE"

# --------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------
END_TIME=$(date '+%s')
ELAPSED=$(( END_TIME - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

log "======================================================"
log "All experiments complete in ${HOURS}h ${MINS}m"
log "Key outputs:"
log "  outputs/gamma_scaling/*/gamma_study_results.json"
log "  outputs/dpo_study_azure/results/dpo_study_results.json"
log "  outputs/evaluation_m4_azure/results_bundle.json"
log "  outputs/evaluation_m5_azure/cross_domain.json"
log "  outputs/generalization_azure/results_bundle.json"
log "  outputs/results/consolidated_summary.json"
log ""
log "To download all results:"
log "  scp -r azureuser@20.57.36.243:~/latent_proxy/outputs/results ./azure_results"
log "======================================================"
