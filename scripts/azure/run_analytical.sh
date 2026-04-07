#!/bin/bash
# scripts/azure/run_analytical.sh
# CPU-only analytical experiments at production scale.
#
# Runs in parallel where possible using background processes.
# CPU-only -- no GPU required -- so these can run alongside GPU experiments.
#
# Experiments:
#   A1. Elicitation benchmark (active vs random baseline)
#   A2. M4 full evaluation (variants a/b, A->B transfer, ablations)
#   A3. M5 cross-domain transfer (game -> stock)
#   A4. M6 generalization study (4 domain pairs, H1-H4, stability)

set -euo pipefail

PROJECT_DIR="${HOME}/latent_proxy"
CONDA_DIR="${HOME}/miniconda3"
ENV_NAME="latent-proxy"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p outputs/{evaluation_m4_azure,evaluation_m5_azure,generalization_azure}

log() { echo "[analytical] $(date '+%H:%M:%S') $*"; }

# --------------------------------------------------------------------------
# A1: Elicitation benchmark (runs fastest, ~30 min on 48 cores)
# --------------------------------------------------------------------------
log "A1: Elicitation benchmark..."
python scripts/run_elicitation.py \
    --n-users 50 \
    --max-rounds 10 \
    --n-particles 2000 \
    --n-eig-samples 800 \
    --n-scenarios 50 \
    --posterior-type particle \
    --seed 42 \
    > "${LOG_DIR}/analytical_elicitation.log" 2>&1 &
PID_A1=$!
log "A1 started (PID ${PID_A1})"

# --------------------------------------------------------------------------
# A2: M4 full evaluation (variants + transfer + ablations, ~4 hours)
# --------------------------------------------------------------------------
log "A2: M4 full evaluation..."
python scripts/run_full_evaluation.py \
    --n-users 30 \
    --max-rounds 10 \
    --n-particles 2000 \
    --n-eig-samples 800 \
    --posterior-type particle \
    --seed 42 \
    --output-dir outputs/evaluation_m4_azure \
    > "${LOG_DIR}/analytical_m4.log" 2>&1 &
PID_A2=$!
log "A2 started (PID ${PID_A2})"

# --------------------------------------------------------------------------
# A3: M5 game-to-stock cross-domain (starts after A1 to avoid CPU contention)
# --------------------------------------------------------------------------
log "Waiting for A1 to finish before starting A3..."
wait "${PID_A1}" && log "A1 complete" || log "WARNING: A1 exited non-zero"

log "A3: M5 cross-domain transfer (game -> stock)..."
mkdir -p outputs/evaluation_m5_azure
python scripts/run_cross_domain.py \
    --n-users 30 \
    --max-rounds 10 \
    --n-particles 2000 \
    --n-eig-samples 800 \
    --n-scenarios-per-round 50 \
    --posterior-type particle \
    --seed 43 \
    --output outputs/evaluation_m5_azure/cross_domain.json \
    > "${LOG_DIR}/analytical_m5.log" 2>&1 &
PID_A3=$!
log "A3 started (PID ${PID_A3})"

# --------------------------------------------------------------------------
# A4: M6 generalization study (starts after A2 to avoid CPU contention)
# --------------------------------------------------------------------------
log "Waiting for A2 to finish before starting A4..."
wait "${PID_A2}" && log "A2 complete" || log "WARNING: A2 exited non-zero"

log "A4: M6 generalization study..."
python scripts/run_generalization_study.py \
    --n-users 30 \
    --max-rounds 10 \
    --n-particles 2000 \
    --n-eig-samples 800 \
    --n-scenarios-per-round 50 \
    --posterior-type particle \
    --seed 44 \
    --output-dir outputs/generalization_azure \
    > "${LOG_DIR}/analytical_m6.log" 2>&1 &
PID_A4=$!
log "A4 started (PID ${PID_A4})"

# --------------------------------------------------------------------------
# Wait for remaining experiments
# --------------------------------------------------------------------------
wait "${PID_A3}" && log "A3 complete" || log "WARNING: A3 exited non-zero"
wait "${PID_A4}" && log "A4 complete" || log "WARNING: A4 exited non-zero"

log "All analytical experiments complete."
