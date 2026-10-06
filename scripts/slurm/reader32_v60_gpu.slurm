#!/bin/bash
#SBATCH --job-name=lp-reader32-v60
#SBATCH --account=ucb736_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/reader32-qualification-v60/gpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/reader32-qualification-v60/gpu-%j.err
set -eo pipefail
export STUDY_WALL_START=$(date +%s)
source ~/.bashrc
conda activate latent-proxy-env
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ROOT=/scratch/alpine/paco0228/latent_proxy_runs/reader32-qualification-v60
export RUNTIME_RECEIPT=/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json
export LOCAL_RUNTIME="${SLURM_TMPDIR:-/tmp}/lp-reader32-gpu-$SLURM_JOB_ID"
export PYTHONPATH="$LOCAL_RUNTIME:$ROOT"
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv -l 2 > "$ROOT/gpu-$SLURM_JOB_ID.csv" 2>&1 &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
# Stage the already verified package archive; startup includes this node-local copy.
timeout --kill-after=2s 45s python "$ROOT/extract_verified_runtime.py"
python -u "$ROOT/reader32_v60.py" gpu "$ROOT"
