#!/bin/bash
# scripts/azure/run_gamma_scaling.sh
# Gamma-only DPO model scaling study on the Azure 2x A100 VM.
#
# Tests whether larger models improve discount-factor (gamma) elicitation by
# training and evaluating at four model sizes:
#   1.5B  (GPU 0)  |  3B   (GPU 1)   -- first wave, in parallel
#   7B    (GPU 0)  |  14B  (GPU 1)   -- second wave, in parallel
#
# Each model: 5,000 DPO pairs, 3 epochs, evaluate 15 users x 5 rounds.
# Results written to outputs/gamma_scaling/<size>/gamma_study_results.json.

set -euo pipefail

PROJECT_DIR="${HOME}/latent_proxy"
CONDA_DIR="${HOME}/miniconda3"
ENV_NAME="latent-proxy"

source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p outputs/gamma_scaling/{1.5B,3B,7B,14B}

N_PAIRS=5000
NUM_EPOCHS=3
N_USERS=15
MAX_ROUNDS=5
SEED=42

log() { echo "[gamma-scaling] $(date '+%H:%M:%S') $*"; }

run_model() {
    local MODEL="$1"
    local LABEL="$2"
    local GPU_ID="$3"
    local OUT="outputs/gamma_scaling/${LABEL}"
    local LOGFILE="${LOG_DIR}/gamma_${LABEL}.log"
    local DONE_MARKER="${OUT}/.done"

    if [ -f "${DONE_MARKER}" ]; then
        log "  ${LABEL}: already complete (${DONE_MARKER} exists) -- skipping"
        return 0
    fi

    log "Starting ${LABEL} on GPU ${GPU_ID}..."
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/run_gamma_study.py \
        --action full \
        --model-name "${MODEL}" \
        --n-pairs "${N_PAIRS}" \
        --num-epochs "${NUM_EPOCHS}" \
        --n-users "${N_USERS}" \
        --max-rounds "${MAX_ROUNDS}" \
        --seed "${SEED}" \
        --output-dir "${OUT}" \
        > "${LOGFILE}" 2>&1
    touch "${DONE_MARKER}"
    log "Finished ${LABEL}. Log: ${LOGFILE}"
}

# --------------------------------------------------------------------------
# Wave 1: 1.5B (GPU 0) and 3B (GPU 1) in parallel
# --------------------------------------------------------------------------
log "====== Wave 1: 1.5B + 3B (parallel) ======"
run_model "Qwen/Qwen2.5-1.5B-Instruct" "1.5B" "0" &
PID_1B=$!
run_model "Qwen/Qwen2.5-3B-Instruct"   "3B"   "1" &
PID_3B=$!

wait "${PID_1B}" && log "1.5B complete" || log "WARNING: 1.5B exited non-zero"
wait "${PID_3B}" && log "3B complete"   || log "WARNING: 3B exited non-zero"

# --------------------------------------------------------------------------
# Wave 2: 7B (GPU 0) and 14B (GPU 1) in parallel
# --------------------------------------------------------------------------
log "====== Wave 2: 7B + 14B (parallel) ======"
run_model "Qwen/Qwen2.5-7B-Instruct"  "7B"  "0" &
PID_7B=$!
run_model "Qwen/Qwen2.5-14B-Instruct" "14B" "1" &
PID_14B=$!

wait "${PID_7B}"  && log "7B complete"  || log "WARNING: 7B exited non-zero"
wait "${PID_14B}" && log "14B complete" || log "WARNING: 14B exited non-zero"

# --------------------------------------------------------------------------
# Print summary table and pick best model
# --------------------------------------------------------------------------
log "====== Gamma Scaling Results ======"
BEST_MODEL=""
BEST_ERR=999
BEST_LABEL=""

for LABEL in 1.5B 3B 7B 14B; do
    RESULTS="outputs/gamma_scaling/${LABEL}/gamma_study_results.json"
    if [ -f "${RESULTS}" ]; then
        python - <<PYEOF
import json
label = "${LABEL}"
with open("${RESULTS}") as f:
    d = json.load(f)
print(f"\n--- {label} ---")
for cond in ['base', 'dpo_phase1', 'dpo_phase2']:
    if cond in d:
        a = d[cond]['mean_alignment']
        e = d[cond]['mean_gamma_error']
        print(f"  {cond}: alignment={a:.3f}  gamma_err={e:.3f}")
# Write best model for downstream use
p2 = d.get('dpo_phase2', {})
err = p2.get('mean_gamma_error', 999)
with open(f"outputs/gamma_scaling/{label}_dpo_p2_err.txt", "w") as g:
    g.write(str(err))
PYEOF
    fi
done

# Identify best model by lowest Phase 2 gamma error
for LABEL in 1.5B 3B 7B 14B; do
    ERR_FILE="outputs/gamma_scaling/${LABEL}_dpo_p2_err.txt"
    if [ -f "${ERR_FILE}" ]; then
        ERR=$(cat "${ERR_FILE}")
        python - <<PYEOF
best_err = ${BEST_ERR}
err = float("${ERR}")
label = "${LABEL}"
if err < best_err:
    with open("outputs/gamma_scaling/best_model.txt", "w") as f:
        models = {
            "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
            "3B": "Qwen/Qwen2.5-3B-Instruct",
            "7B": "Qwen/Qwen2.5-7B-Instruct",
            "14B": "Qwen/Qwen2.5-14B-Instruct",
        }
        f.write(f"{models[label]}\n{label}\n{err}\n")
        print(f"New best: {label} (gamma_err={err:.3f})")
PYEOF
    fi
done

if [ -f "outputs/gamma_scaling/best_model.txt" ]; then
    BEST_LINE=$(head -2 outputs/gamma_scaling/best_model.txt | tail -1)
    log "Best model size: ${BEST_LINE}"
fi

log "Gamma scaling study complete."
