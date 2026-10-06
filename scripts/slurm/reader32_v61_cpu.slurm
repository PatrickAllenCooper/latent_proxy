#!/bin/bash
#SBATCH --job-name=lp-reader32-v61-prep
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/reader32-preparation-v61/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/reader32-preparation-v61/cpu-%j.err
set -eo pipefail
source ~/.bashrc
conda activate latent-proxy-env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export ROOT=/scratch/alpine/paco0228/latent_proxy_runs/reader32-preparation-v61
export RUNTIME_RECEIPT=/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json
export LOCAL_RUNTIME="${SLURM_TMPDIR:-/tmp}/lp-reader32-v61-$SLURM_JOB_ID"
export PYTHONPATH="$LOCAL_RUNTIME:$ROOT"
echo "changed CPU preparation begins $(date -Is) job=$SLURM_JOB_ID"
timeout --kill-after=2s 590s bash -c 'python "$ROOT/extract_verified_runtime.py" && python -u "$ROOT/prepare_reader32_v61.py" "$ROOT"'
echo "changed CPU preparation ends $(date -Is)"
