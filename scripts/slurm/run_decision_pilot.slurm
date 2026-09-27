#!/bin/bash
#SBATCH --job-name=lp-decision-pilot
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/decision-pilot-%j.out
#SBATCH --error=logs/decision-pilot-%j.err

set -euo pipefail
PROJECT_DIR="/projects/paco0228/latent_proxy_experiments"
CONDA_ENV_NAME="latent-proxy-env"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJECT_DIR"
mkdir -p logs outputs/preference_pilot
set +u
source ~/.bashrc
set -u
conda activate "$CONDA_ENV_NAME"
echo "job=$SLURM_JOB_ID revision=${LATENT_PROXY_REVISION:-unknown} started=$(date -Is)"
python scripts/run_canonical_campaign.py \
  --domains game_a,stock,supply_chain \
  --arms active,fixed,random,decision_impact \
  --n-users 50 --seeds 42 --max-rounds 8 \
  --n-particles 500 --n-eig-samples 200 \
  --n-scenarios-per-round 15 --workers 12 \
  --output-dir outputs/preference_pilot
python scripts/summarize_preference_campaign.py \
  --input-dir outputs/preference_pilot
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
