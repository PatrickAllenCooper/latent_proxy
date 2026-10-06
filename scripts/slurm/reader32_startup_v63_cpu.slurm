#!/bin/bash
#SBATCH --job-name=lp-reader32-startup-v63
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/reader32-startup-v63/spool/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/reader32-startup-v63/spool/cpu-%j.err
set -eo pipefail
export STUDY_WALL_START=$(date +%s)
source ~/.bashrc
conda activate latent-proxy-env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export SOURCE=/scratch/alpine/paco0228/latent_proxy_runs/reader32-startup-v63/source
export SPOOL=/scratch/alpine/paco0228/latent_proxy_runs/reader32-startup-v63/spool
export RUNTIME_RECEIPT=/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json
export LOCAL_RUNTIME="${SLURM_TMPDIR:-/tmp}/lp-reader32-startup-v63-$SLURM_JOB_ID"
export PYTHONPATH="$LOCAL_RUNTIME:$SOURCE"
cd "$SOURCE"
timeout --kill-after=2s 590s bash -c 'python "$SOURCE/extract_verified_runtime.py" && python -u "$SOURCE/reader32_startup_v63.py" "$SOURCE" "$SPOOL"'
