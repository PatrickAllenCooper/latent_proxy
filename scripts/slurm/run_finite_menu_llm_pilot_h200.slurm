#!/bin/bash
#SBATCH --job-name=lp-proxy-llm
#SBATCH --account=ucb736_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/proxy-program-v2/logs/llm-pilot-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/proxy-program-v2/logs/llm-pilot-%j.err

set -euo pipefail
PROJECT_DIR="/home/paco0228/latent_proxy_proxy_program"
RUN_ROOT="/scratch/alpine/paco0228/latent_proxy_runs/proxy-program-v2"
export HF_HOME="/scratch/alpine/paco0228/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$PROJECT_DIR"
mkdir -p "$RUN_ROOT/outputs"
set +u
source ~/.bashrc
set -u
conda activate latent-proxy-env
echo "job=$SLURM_JOB_ID revision=$(cat REVISION) started=$(date -Is)"
python - <<'PY'
import torch
assert torch.cuda.is_available()
assert "H200" in torch.cuda.get_device_name(0)
assert tuple(int(x) for x in torch.version.cuda.split('.')[:2]) > (12, 0)
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
python scripts/run_finite_menu_llm_pilot.py \
  --condition base --n-users 12 --n-scenarios 8 --seed 5001 \
  --policy-checkpoint "$PROJECT_DIR/policy.pt" \
  --output-dir "$RUN_ROOT/outputs"
python scripts/run_finite_menu_llm_pilot.py \
  --condition dialogue --n-users 12 --n-scenarios 8 --seed 5001 \
  --checkpoint /scratch/alpine/paco0228/latent_proxy_runs/77add19/outputs/dpo/dialogue_phase2/phase2/final \
  --policy-checkpoint "$PROJECT_DIR/policy.pt" \
  --output-dir "$RUN_ROOT/outputs"
python scripts/summarize_finite_menu_llm.py --input-dir "$RUN_ROOT/outputs"
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
