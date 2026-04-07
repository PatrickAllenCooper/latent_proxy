#!/bin/bash
# scripts/azure/run_dpo_full.sh
# Full 3-parameter DPO elicitation study using the best model from gamma scaling.
#
# Trains DPO (Phase 1 quality + Phase 2 alignment) with the standard 3-parameter
# preference model (gamma, alpha, lambda) across all three environments.
# Then runs the 4-condition comparison: analytical / base / P1 / P2.
#
# Reads the best model from outputs/gamma_scaling/best_model.txt if available,
# falls back to Qwen2.5-7B-Instruct if not.

set -euo pipefail

PROJECT_DIR="${HOME}/latent_proxy"
CONDA_DIR="${HOME}/miniconda3"
ENV_NAME="latent-proxy"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p outputs/dpo_study_azure

log() { echo "[dpo-full] $(date '+%H:%M:%S') $*"; }

# --------------------------------------------------------------------------
# Determine best model
# --------------------------------------------------------------------------
if [ -f "outputs/gamma_scaling/best_model.txt" ]; then
    BEST_MODEL=$(head -1 outputs/gamma_scaling/best_model.txt | tr -d '[:space:]')
    BEST_LABEL=$(sed -n '2p' outputs/gamma_scaling/best_model.txt | tr -d '[:space:]')
    log "Using best model from gamma scaling: ${BEST_MODEL} (${BEST_LABEL})"
else
    BEST_MODEL="Qwen/Qwen2.5-7B-Instruct"
    BEST_LABEL="7B"
    log "best_model.txt not found; defaulting to ${BEST_MODEL}"
fi

DPO_OUT="outputs/dpo_study_azure/${BEST_LABEL}"
mkdir -p "${DPO_OUT}"

# --------------------------------------------------------------------------
# Phase 1: Generate + train quality-floor DPO pairs
# --------------------------------------------------------------------------
P1_CKPT="${DPO_OUT}/phase1/phase1/final"
P2_CKPT="${DPO_OUT}/phase2/phase2/final"

if [ -f "${P1_CKPT}/adapter_config.json" ]; then
    log "Phase 1 checkpoint already exists -- skipping Phase 1 training"
else
    log "Generating Phase 1 DPO pairs (quality, n=10000)..."
    python scripts/train_quality.py \
        --action generate \
        --model-name "${BEST_MODEL}" \
        --n-pairs 10000 \
        --data-path "${DPO_OUT}/data_phase1"

    log "Training Phase 1 DPO (quality floor)..."
    CUDA_VISIBLE_DEVICES="0" python scripts/train_quality.py \
        --action train \
        --model-name "${BEST_MODEL}" \
        --n-pairs 10000 \
        --num-epochs 3 \
        --output-dir "${DPO_OUT}/phase1" \
        > "${LOG_DIR}/dpo_full_phase1.log" 2>&1
    log "Phase 1 training complete."
fi

# --------------------------------------------------------------------------
# Phase 2: Generate + train alignment DPO pairs
# --------------------------------------------------------------------------
if [ -f "${P2_CKPT}/adapter_config.json" ]; then
    log "Phase 2 checkpoint already exists -- skipping Phase 2 training"
else
    log "Generating Phase 2 DPO pairs (alignment, n=20000)..."
    python scripts/train_alignment.py \
        --action generate \
        --model-name "${BEST_MODEL}" \
        --n-pairs 20000 \
        --data-path "${DPO_OUT}/data_phase2"

    log "Training Phase 2 DPO (alignment)..."
    CUDA_VISIBLE_DEVICES="0" python scripts/train_alignment.py \
        --action train \
        --model-name "${BEST_MODEL}" \
        --n-pairs 20000 \
        --num-epochs 3 \
        --output-dir "${DPO_OUT}/phase2" \
        > "${LOG_DIR}/dpo_full_phase2.log" 2>&1
    log "Phase 2 training complete."
fi

# --------------------------------------------------------------------------
# Run the 4-condition DPO study (analytical / base / P1 / P2)
# --------------------------------------------------------------------------
STUDY_RESULTS="outputs/dpo_study_azure/results/dpo_study_results.json"
if [ -f "${STUDY_RESULTS}" ]; then
    log "DPO study results already exist -- skipping"
else
    log "Running full DPO elicitation study (n_users=15, max_rounds=5)..."
    python scripts/run_dpo_study.py \
        --n-users 15 \
        --max-rounds 5 \
        --base-model "${BEST_MODEL}" \
        --phase1-checkpoint "${P1_CKPT}" \
        --phase2-checkpoint "${P2_CKPT}" \
        --envs game stock supply_chain \
        --seed 42 \
        --n-particles 500 \
        --n-eig-samples 200 \
        --output-dir "outputs/dpo_study_azure/results" \
        > "${LOG_DIR}/dpo_full_study.log" 2>&1
    log "DPO study complete. Results: outputs/dpo_study_azure/results/"
fi
