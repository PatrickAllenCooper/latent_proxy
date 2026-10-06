#!/bin/bash
#SBATCH --job-name=lp-factorial-v56
#SBATCH --account=ucb736_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --exclude=c3gpu-e5-u5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/preference-factorial-direct-v57/gpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/preference-factorial-direct-v57/gpu-%j.err
set -eo pipefail
export STUDY_WALL_START=$(date +%s)
source ~/.bashrc
conda activate latent-proxy-env
set -u
ROOT=/scratch/alpine/paco0228/latent_proxy_runs/preference-factorial-direct-v57
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export RUNTIME_RECEIPT=/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json
export LOCAL_RUNTIME="${SLURM_TMPDIR:-/tmp}/latent-proxy-runtime-$SLURM_JOB_ID"
echo "job=$SLURM_JOB_ID started=$(date -Is) node=$(hostname)"
test ! -e "$ROOT/smoke"
test -e "$ROOT/hash_gate.json"
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv -l 2 > "$ROOT/gpu-$SLURM_JOB_ID.csv" 2>&1 &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
timeout --kill-after=2s 45s python /scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/extract_verified_runtime.py
export PYTHONPATH="$LOCAL_RUNTIME"
timeout --kill-after=2s 285s python -u "$ROOT/run_preference_factorial_direct_v57.py" --manifest "$ROOT/manifest.jsonl" --token-receipt "$ROOT/token_receipt.json" --cache-receipt "$ROOT/cache_receipt.json" --output-dir "$ROOT/smoke"
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
