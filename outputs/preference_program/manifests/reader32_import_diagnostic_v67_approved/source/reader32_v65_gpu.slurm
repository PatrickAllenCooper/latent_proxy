#!/bin/bash
#SBATCH --job-name=lp-reader32-v65
#SBATCH --account=ucb736_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/reader32-qualification-v65/spool/gpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/reader32-qualification-v65/spool/gpu-%j.err
set -eo pipefail
export STUDY_WALL_START=$(date +%s.%N)
echo "study_shell_start=$STUDY_WALL_START generation_deadline=+270 export_deadline=+285 cleanup_deadline=+300"
source ~/.bashrc
conda activate latent-proxy-env
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export SOURCE=/scratch/alpine/paco0228/latent_proxy_runs/reader32-qualification-v65/source
export SPOOL=/scratch/alpine/paco0228/latent_proxy_runs/reader32-qualification-v65/spool
export RUNTIME_RECEIPT=/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json
export LOCAL_RUNTIME="${SLURM_TMPDIR:-/tmp}/lp-reader32-v65-$SLURM_JOB_ID"
export PYTHONPATH="$LOCAL_RUNTIME:$SOURCE"
cd "$SOURCE"
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv -l 2 > "$SPOOL/gpu-$SLURM_JOB_ID.csv" 2>&1 &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
echo "GPU qualification begins $(date -Is) job=$SLURM_JOB_ID"
timeout --kill-after=2s 45s python "$SOURCE/extract_verified_runtime.py"
# Runner deadlines include shell setup/runtime staging, not queue time.
# Never submit until execution_freeze.json has explicit 30-second amendment approval.
RUNNER_REMAINING=$(python -c 'import os,time; print(max(.001,float(os.environ["STUDY_WALL_START"])+285-time.time()))')
# External bound protects export window if Python watchdogs cannot run.
timeout --signal=TERM --kill-after=2s "${RUNNER_REMAINING}s" python -u "$SOURCE/run_reader32_v65.py" gpu "$SOURCE" --spool "$SPOOL"
echo "GPU qualification ends $(date -Is)"
