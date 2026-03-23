#!/bin/bash
#SBATCH --job-name=latent-proxy-dpo
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dpo-training-%j.out
#SBATCH --error=logs/dpo-training-%j.err

set -euo pipefail

PROJECT_DIR="/projects/paco0228/latent_proxy"
CONDA_ENV="/projects/paco0228/software/anaconda/envs/latent-proxy-env"
HF_CACHE="/scratch/alpine/paco0228/hf_cache"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

cd "$PROJECT_DIR"
mkdir -p logs

source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

echo "Phase 1: Quality floor training"
python scripts/train_quality.py --action train \
    --n-pairs 10000 \
    --num-epochs 3 \
    --output-dir outputs/dpo/phase1

echo "Phase 2: Alignment training"
python scripts/train_alignment.py --action train \
    --phase1-checkpoint outputs/dpo/phase1/final \
    --n-pairs 20000 \
    --num-epochs 3 \
    --output-dir outputs/dpo/phase2

echo "DPO training complete."
