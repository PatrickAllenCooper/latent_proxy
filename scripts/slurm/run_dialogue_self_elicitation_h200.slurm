#!/bin/bash
#SBATCH --job-name=lp-dialogue-self
#SBATCH --account=ucb736_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/77add19/logs/dialogue-self-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/77add19/logs/dialogue-self-%j.err

set -euo pipefail
PROJECT_DIR="/home/paco0228/latent_proxy_experiments"
CONDA_ENV_NAME="latent-proxy-env"
RESULTS_DIR="/scratch/alpine/paco0228/latent_proxy_runs/77add19"
export HF_HOME="/scratch/alpine/paco0228/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_DIR"
mkdir -p "$RESULTS_DIR/outputs/dialogue_comparison/self_elicitation"
set +u
source ~/.bashrc
set -u
conda activate "$CONDA_ENV_NAME"
echo "job=$SLURM_JOB_ID revision=${LATENT_PROXY_REVISION:-unknown} started=$(date -Is)"
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
version = tuple(int(x) for x in torch.version.cuda.split('.')[:2])
assert version > (12, 0), f"H200 requires CUDA > 12.0, found {torch.version.cuda}"
assert "H200" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print(f"Torch {torch.__version__}; CUDA {torch.version.cuda}; GPU {torch.cuda.get_device_name(0)}")
PY
python scripts/run_dpo_study.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --n-users 50 --max-rounds 5 --envs game --seed 9001 \
  --n-particles 200 --n-eig-samples 100 \
  --phase1-checkpoint "$PROJECT_DIR/checkpoints/phase1" \
  --phase2-checkpoint "$PROJECT_DIR/checkpoints/phase2" \
  --dialogue-checkpoint "$RESULTS_DIR/outputs/dpo/dialogue_phase2/phase2/final" \
  --conditions base,dpo_phase1,dpo_phase2,dpo_dialogue \
  --output-dir "$RESULTS_DIR/outputs/dialogue_comparison/self_elicitation" --skip-plots
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
