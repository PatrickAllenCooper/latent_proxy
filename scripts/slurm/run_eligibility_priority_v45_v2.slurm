#!/bin/bash
#SBATCH --job-name=lp-elig-priority
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
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/eligibility-priority-v45/logs/pilot-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/eligibility-priority-v45/logs/pilot-%j.err

set -euo pipefail
export STUDY_WALL_START=$(date +%s)
PROJECT_DIR=/home/paco0228/latent_proxy_proxy_program_v22
RUN_ROOT=/scratch/alpine/paco0228/latent_proxy_runs/startup-diagnostic-v37
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/alpine/paco0228/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$PROJECT_DIR"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/${RUN_LABEL:?}"
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
export RUNTIME_RECEIPT=/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json
export LOCAL_RUNTIME="${SLURM_TMPDIR:-/tmp}/latent-proxy-runtime-$SLURM_JOB_ID"
timeout 45s python /scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/extract_verified_runtime.py
export PYTHONPATH="$LOCAL_RUNTIME:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python /scratch/alpine/paco0228/latent_proxy_runs/eligibility-priority-v45/repair-v3/run_eligibility_priority_qualification_v2.py \
 --manifest /scratch/alpine/paco0228/latent_proxy_runs/eligibility-priority-v45/manifest.jsonl \
 --token-receipt /scratch/alpine/paco0228/latent_proxy_runs/eligibility-priority-v45/token_receipt.json \
 --cache-receipt /scratch/alpine/paco0228/latent_proxy_runs/startup-diagnostic-v37/cache_receipt.json \
 --output-dir /scratch/alpine/paco0228/latent_proxy_runs/eligibility-priority-v45/smoke
echo "job=$SLURM_JOB_ID completed=$(date -Is)"
