#!/bin/bash
#SBATCH --job-name=latent-proxy-ablation
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ablation-%j.out
#SBATCH --error=logs/ablation-%j.err

set -euo pipefail

PROJECT_DIR="/projects/paco0228/latent_proxy"
CONDA_ENV_NAME="latent-proxy-env"

cd "$PROJECT_DIR"
mkdir -p logs outputs/evaluation_m4

source ~/.bashrc
conda activate "$CONDA_ENV_NAME"

python scripts/run_full_evaluation.py \
    --n-users 20 \
    --max-rounds 10 \
    --n-particles 2000 \
    --n-eig-samples 800 \
    --posterior-type particle \
    --output-dir outputs/evaluation_m4_ablation \
    --skip-transfer

echo "Ablation sweep job complete."
