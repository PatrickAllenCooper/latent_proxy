#!/bin/bash
#SBATCH --job-name=lp-dialogue-eval
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dialogue-eval-%j.out
#SBATCH --error=logs/dialogue-eval-%j.err

set -euo pipefail
PROJECT_DIR="/projects/paco0228/latent_proxy_experiments"
CONDA_ENV_NAME="latent-proxy-env"
export HF_HOME="/scratch/alpine/paco0228/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_DIR"
mkdir -p logs outputs/dialogue_comparison
set +u
source ~/.bashrc
set -u
conda activate "$CONDA_ENV_NAME"
echo "job=$SLURM_JOB_ID revision=${LATENT_PROXY_REVISION:-unknown} started=$(date -Is)"
nvidia-smi
python scripts/run_adherence_study.py \
  --n-users 50 --max-rounds 5 --environment game --seed 9001 \
  --n-particles 200 --n-eig-samples 100 \
  --phase1-checkpoint checkpoints/phase1 \
  --phase2-checkpoint checkpoints/phase2 \
  --dialogue-checkpoint outputs/dpo/dialogue_phase2/phase2/final \
  --conditions base,dpo_phase1,dpo_phase2,dpo_dialogue \
  --output-dir outputs/dialogue_comparison/adherence
python scripts/run_dpo_study.py \
  --n-users 50 --max-rounds 5 --envs game --seed 9001 \
  --n-particles 200 --n-eig-samples 100 \
  --phase1-checkpoint checkpoints/phase1 \
  --phase2-checkpoint checkpoints/phase2 \
  --dialogue-checkpoint outputs/dpo/dialogue_phase2/phase2/final \
  --conditions base,dpo_phase1,dpo_phase2,dpo_dialogue \
  --output-dir outputs/dialogue_comparison/self_elicitation --skip-plots
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
