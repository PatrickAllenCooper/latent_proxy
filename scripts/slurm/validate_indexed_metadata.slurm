#!/bin/bash
#SBATCH --job-name=lp-metadata-index
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/metadata-index-v47/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/metadata-index-v47/cpu-%j.err
set -eo pipefail
source ~/.bashrc
conda activate latent-proxy-env
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
ROOT=/scratch/alpine/paco0228/latent_proxy_runs/metadata-index-v47
echo "CPUdiagnostic job=$SLURM_JOB_ID started=$(date -Is)"
/usr/bin/time -v timeout --kill-after=2s 90s python -u "$ROOT/validate_indexed_metadata_integration.py" "$ROOT/integration"
echo "CPUdiagnostic complete=$(date -Is)"
