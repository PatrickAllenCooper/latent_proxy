#!/bin/bash
#SBATCH --job-name=lp-elig-cpu-v3
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/eligibility-priority-v45/repair-v3/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/eligibility-priority-v45/repair-v3/cpu-%j.err
set -eo pipefail
source ~/.bashrc
conda activate latent-proxy-env
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ROOT=/scratch/alpine/paco0228/latent_proxy_runs/eligibility-priority-v45
export RUNTIME_RECEIPT=/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json
export LOCAL_RUNTIME="${SLURM_TMPDIR:-/tmp}/latent-proxy-cpu-repair-$SLURM_JOB_ID"
echo "CPUjob=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
test ! -e "$ROOT/token_receipt.json"
/usr/bin/time -v timeout --kill-after=2s 45s python /scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/extract_verified_runtime.py
export PYTHONPATH="$LOCAL_RUNTIME"
/usr/bin/time -v timeout --kill-after=2s 90s python -u "$ROOT/repair-v3/prepare_eligibility_tokens_v3.py" --manifest "$ROOT/manifest.jsonl" --cache-receipt /scratch/alpine/paco0228/latent_proxy_runs/startup-diagnostic-v37/cache_receipt.json --output "$ROOT/token_receipt.json"
echo "CPUjob=$SLURM_JOB_ID complete=$(date -Is)"
