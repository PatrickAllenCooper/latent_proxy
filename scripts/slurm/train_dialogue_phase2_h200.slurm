#!/bin/bash
#SBATCH --job-name=lp-dialogue-dpo
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dialogue-phase2-%j.out
#SBATCH --error=logs/dialogue-phase2-%j.err

set -euo pipefail
PROJECT_DIR="/projects/paco0228/latent_proxy_experiments"
CONDA_ENV_NAME="latent-proxy-env"
export HF_HOME="/scratch/alpine/paco0228/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_DIR"
mkdir -p logs outputs/dpo/dialogue_phase2
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
python scripts/train_alignment.py --action train \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --phase1-checkpoint checkpoints/phase1 \
  --n-pairs 20000 --num-epochs 3 \
  --dialogue-context-rounds 4 \
  --output-dir outputs/dpo/dialogue_phase2 \
  --no-gradient-checkpointing
python - <<'PY'
from safetensors.torch import load_file
import torch
start = load_file("checkpoints/phase1/adapter_model.safetensors")
end = load_file("outputs/dpo/dialogue_phase2/phase2/final/adapter_model.safetensors")
assert start.keys() == end.keys(), "Adapter tensor keys changed"
diffs = [torch.max(torch.abs(start[k] - end[k])).item() for k in start]
assert any(d > 0 for d in diffs), "Dialogue Phase 2 adapter is identical to Phase 1"
print(f"Updated tensors: {sum(d > 0 for d in diffs)}/{len(diffs)}; max_abs_diff={max(diffs):.8g}")
PY
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
