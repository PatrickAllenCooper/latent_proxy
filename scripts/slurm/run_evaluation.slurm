#!/bin/bash
#SBATCH --job-name=latent-proxy-eval
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluation-%j.out
#SBATCH --error=logs/evaluation-%j.err

set -euo pipefail

PROJECT_DIR="/projects/paco0228/latent_proxy"
CONDA_ENV="/projects/paco0228/software/anaconda/envs/latent-proxy-env"
HF_CACHE="/scratch/alpine/paco0228/hf_cache"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

cd "$PROJECT_DIR"
mkdir -p logs

source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

echo "Running elicitation benchmark (CPU portion)"
python scripts/run_elicitation.py \
    --n-users 50 \
    --max-rounds 10 \
    --n-particles 2000 \
    --n-eig-samples 800 \
    --posterior-type particle

echo "Running Milestone 4 full evaluation (variants, transfer, ablations)"
python scripts/run_full_evaluation.py \
    --n-users 30 \
    --max-rounds 10 \
    --n-particles 2000 \
    --n-eig-samples 800 \
    --posterior-type particle \
    --output-dir outputs/evaluation_m4

echo "Running Milestone 5 cross-domain transfer (game -> stock)"
mkdir -p outputs/evaluation_m5
python scripts/run_cross_domain.py \
    --n-users 30 \
    --max-rounds 10 \
    --n-particles 2000 \
    --n-eig-samples 800 \
    --n-scenarios-per-round 50 \
    --posterior-type particle \
    --seed 43 \
    --output outputs/evaluation_m5/cross_domain_transfer.json

echo "Running DPO evaluation (Phase 1)"
python scripts/train_quality.py --action evaluate \
    --checkpoint outputs/dpo/phase1/final

echo "Running alignment evaluation (Phase 2)"
python scripts/train_alignment.py --action evaluate \
    --checkpoint outputs/dpo/phase2/final

echo "Evaluation complete."
