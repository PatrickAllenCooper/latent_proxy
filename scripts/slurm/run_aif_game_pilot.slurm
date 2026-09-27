#!/bin/bash
#SBATCH --job-name=lp-aif-game
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/aif-game-pilot/logs/aif-game-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/aif-game-pilot/logs/aif-game-%j.err

set -euo pipefail
PROJECT_DIR="/home/paco0228/latent_proxy_aif_pilot"
CONDA_ENV_NAME="latent-proxy-env"
RESULTS_DIR="/scratch/alpine/paco0228/latent_proxy_runs/aif-game-pilot"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR/outputs/game"
set +u
source ~/.bashrc
set -u
conda activate "$CONDA_ENV_NAME"
echo "job=$SLURM_JOB_ID revision=${LATENT_PROXY_REVISION:-unknown} started=$(date -Is)"
python scripts/run_canonical_campaign.py \
  --domains game_a \
  --arms active,random,decision_impact,aif_20,aif_50,aif_80 \
  --n-users 50 --seeds 42 --max-rounds 8 \
  --n-particles 500 --n-eig-samples 200 \
  --n-scenarios-per-round 15 --workers 12 \
  --output-dir "$RESULTS_DIR/outputs/game"
python scripts/summarize_preference_campaign.py \
  --input-dir "$RESULTS_DIR/outputs/game"
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
