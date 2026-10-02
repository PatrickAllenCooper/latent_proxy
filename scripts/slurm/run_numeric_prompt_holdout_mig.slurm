#!/bin/bash
#SBATCH --job-name=lp-prompt-holdout
#SBATCH --account=ucb736_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --exclude=c3gpu-e5-u5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/prompt-holdout-v14/logs/pilot-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/prompt-holdout-v14/logs/pilot-%j.err

set -euo pipefail
PROJECT_DIR=/home/paco0228/latent_proxy_proxy_program_v14
RUN_ROOT=/scratch/alpine/paco0228/latent_proxy_runs/prompt-holdout-v14
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/alpine/paco0228/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$PROJECT_DIR"
mkdir -p "$RUN_ROOT/pilot" "$RUN_ROOT/logs"
set +u
source ~/.bashrc
set -u
conda activate latent-proxy-env
echo "job=$SLURM_JOB_ID node=$(hostname) revision=$(cat REVISION) started=$(date -Is)"
python - <<'PY'
import json, torch
assert torch.cuda.is_available(), 'CUDA unavailable at GPU preflight'
x = torch.ones(1024, device='cuda')
assert float(x.sum().item()) == 1024.0
print(json.dumps({'event': 'gpu_preflight', 'device_count': torch.cuda.device_count(),
                  'device_name': torch.cuda.get_device_name(0),
                  'allocated_bytes': torch.cuda.memory_allocated()}), flush=True)
PY
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv -l 15 \
  > "$RUN_ROOT/logs/pilot-$SLURM_JOB_ID-gpu.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
python scripts/run_proxy_advice_ablation.py generate \
  --manifest "$RUN_ROOT/manifests/pilot.jsonl" \
  --output-dir "$RUN_ROOT/pilot"
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
