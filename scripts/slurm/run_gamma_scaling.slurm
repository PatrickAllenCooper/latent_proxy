#!/bin/bash
#SBATCH --job-name=gamma-scaling
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gamma-scaling-%j.out
#SBATCH --error=logs/gamma-scaling-%j.err

# Gamma-only DPO model scaling study on CURC A100.
#
# Tests whether larger models improve preference elicitation by training
# and evaluating at three model sizes on the gamma (discount factor) task:
#   1. Qwen2.5-1.5B-Instruct  (baseline, already tested locally)
#   2. Qwen2.5-3B-Instruct    (fits A100 easily)
#   3. Qwen2.5-7B-Instruct    (full-size, 4-bit on A100)
#
# Each model: generate gamma-focused DPO data, train Phase 1 + Phase 2,
# evaluate with 15 users x 5 rounds.
#
# Expected runtime: ~10 hours (3 models x ~3h training + eval each)

set -euo pipefail

PROJECT_DIR="/projects/paco0228/latent_proxy"
CONDA_ENV="/projects/paco0228/software/anaconda/envs/latent-proxy-env"
HF_CACHE="/scratch/alpine/paco0228/hf_cache"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
mkdir -p logs outputs/gamma_scaling

source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

echo "========================================"
echo "Gamma scaling study -- $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================"

N_PAIRS=5000
NUM_EPOCHS=3
N_USERS=15
MAX_ROUNDS=5
SEED=42

MODELS=(
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

LABELS=(
    "1.5B"
    "3B"
    "7B"
)

for idx in "${!MODELS[@]}"; do
    MODEL="${MODELS[$idx]}"
    LABEL="${LABELS[$idx]}"
    OUT_DIR="outputs/gamma_scaling/${LABEL}"

    echo ""
    echo "========================================"
    echo "Model: ${MODEL} (${LABEL})"
    echo "Started: $(date)"
    echo "========================================"

    mkdir -p "${OUT_DIR}"

    python scripts/run_gamma_study.py \
        --action full \
        --model-name "${MODEL}" \
        --n-pairs ${N_PAIRS} \
        --num-epochs ${NUM_EPOCHS} \
        --n-users ${N_USERS} \
        --max-rounds ${MAX_ROUNDS} \
        --seed ${SEED} \
        --output-dir "${OUT_DIR}" \
        2>&1 | tee "${OUT_DIR}/run.log"

    echo "Finished ${LABEL}: $(date)"
done

echo ""
echo "========================================"
echo "Gamma scaling study complete -- $(date)"
echo "========================================"

echo "Results summary:"
for LABEL in "${LABELS[@]}"; do
    RESULTS="outputs/gamma_scaling/${LABEL}/gamma_study_results.json"
    if [ -f "${RESULTS}" ]; then
        echo ""
        echo "--- ${LABEL} ---"
        python -c "
import json
with open('${RESULTS}') as f:
    d = json.load(f)
for cond in ['base', 'dpo_phase1', 'dpo_phase2']:
    if cond in d:
        print(f'  {cond}: alignment={d[cond][\"mean_alignment\"]:.3f}  gamma_err={d[cond][\"mean_gamma_error\"]:.3f}')
"
    fi
done
