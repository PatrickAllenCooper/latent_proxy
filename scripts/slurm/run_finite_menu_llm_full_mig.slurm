#!/bin/bash
#SBATCH --job-name=lp-proxy-full
#SBATCH --account=ucb736_asc1
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200_2g.35gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/proxy-program-v4/logs/full-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/proxy-program-v4/logs/full-%j.err

set -euo pipefail
PROJECT_DIR=/home/paco0228/latent_proxy_proxy_program_v4
RUN_ROOT=/scratch/alpine/paco0228/latent_proxy_runs/proxy-program-v4
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/scratch/alpine/paco0228/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$PROJECT_DIR"
mkdir -p "$RUN_ROOT/outputs" "$RUN_ROOT/logs"
set +u
source ~/.bashrc
set -u
conda activate latent-proxy-env
echo "job=$SLURM_JOB_ID revision=$(cat REVISION) started=$(date -Is)"
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv -l 30 \
  > "$RUN_ROOT/logs/full-$SLURM_JOB_ID-gpu.csv" &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
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
