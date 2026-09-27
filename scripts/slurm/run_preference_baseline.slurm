#!/bin/bash
#SBATCH --job-name=lp-pref-baseline
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/preference-baseline-%j.out
#SBATCH --error=logs/preference-baseline-%j.err

set -euo pipefail
PROJECT_DIR="/projects/paco0228/latent_proxy_experiments"
CONDA_ENV_NAME="latent-proxy-env"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HOME="/scratch/alpine/paco0228/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
cd "$PROJECT_DIR"
mkdir -p logs outputs/canonical_renorm_stage3 outputs/preference_pilot
set +u
source ~/.bashrc
set -u
conda activate "$CONDA_ENV_NAME"
echo "job=$SLURM_JOB_ID revision=${LATENT_PROXY_REVISION:-unknown} started=$(date -Is)"
python scripts/run_canonical_campaign.py \
  --domains supply_chain --arms active --n-users 50 --seeds 42,43,44,45,46 \
  --max-rounds 12 --n-particles 2000 --n-eig-samples 800 \
  --n-scenarios-per-round 50 --workers 12 \
  --utility-form return_normalized \
  --output-dir outputs/canonical_renorm_stage3
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
