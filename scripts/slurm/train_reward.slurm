#!/bin/bash
#SBATCH --job-name=latent-proxy-reward
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/reward-training-%j.out
#SBATCH --error=logs/reward-training-%j.err

set -euo pipefail

PROJECT_DIR="/projects/paco0228/latent_proxy"
CONDA_ENV="/projects/paco0228/software/anaconda/envs/latent-proxy-env"
HF_CACHE="/scratch/alpine/paco0228/hf_cache"

export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

cd "$PROJECT_DIR"
mkdir -p logs

source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

echo "Training type-conditioned reward model"
python -c "
from src.training.reward_model import TypeConditionedRewardModel, RewardModelConfig
config = RewardModelConfig(output_dir='outputs/reward_model')
model = TypeConditionedRewardModel(config)
model.train()
"

echo "Reward model training complete."
