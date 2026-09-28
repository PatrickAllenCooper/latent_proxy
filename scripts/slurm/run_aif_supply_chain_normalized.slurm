#!/bin/bash
#SBATCH --job-name=lp-aif-sc-norm
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/aif-sc-normalized/logs/aif-sc-normalized-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/aif-sc-normalized/logs/aif-sc-normalized-%j.err

set -euo pipefail
PROJECT_DIR="/home/paco0228/latent_proxy_aif_pilot"
CONDA_ENV_NAME="latent-proxy-env"
RESULTS_DIR="/scratch/alpine/paco0228/latent_proxy_runs/aif-sc-normalized"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR/outputs/supply_chain"
set +u
source ~/.bashrc
set -u
conda activate "$CONDA_ENV_NAME"
echo "job=$SLURM_JOB_ID revision=${LATENT_PROXY_REVISION:-unknown} started=$(date -Is)"
python scripts/run_canonical_campaign.py \
  --domains supply_chain \
  --arms active,random,decision_impact,aif_50,aif_80 \
  --n-users 50 --seeds 42,43,44,45,46 --max-rounds 8 \
  --n-particles 500 --n-eig-samples 200 \
  --n-scenarios-per-round 15 --workers 12 \
  --utility-form return_normalized \
  --output-dir "$RESULTS_DIR/outputs/supply_chain"
python scripts/summarize_preference_campaign.py \
  --input-dir "$RESULTS_DIR/outputs/supply_chain"
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
