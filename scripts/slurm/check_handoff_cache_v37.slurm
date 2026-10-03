#!/bin/bash
#SBATCH --job-name=lp-cache-check
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/startup-diagnostic-v37/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/startup-diagnostic-v37/cpu-%j.err
set -euo pipefail
set +u
source ~/.bashrc
set -u
conda activate latent-proxy-env
export HF_HUB_OFFLINE=1
export RECEIPT=/scratch/alpine/paco0228/latent_proxy_runs/startup-diagnostic-v37/cache_receipt.json
python /scratch/alpine/paco0228/latent_proxy_runs/startup-diagnostic-v37/check_handoff_cache.py
