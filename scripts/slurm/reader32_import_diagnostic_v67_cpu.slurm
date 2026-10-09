#!/bin/bash
#SBATCH --job-name=lp-reader32-import-v67
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-diagnostic-v67/spool/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-diagnostic-v67/spool/cpu-%j.err
set -eo pipefail
export STUDY_WALL_START=$(date +%s.%N)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export SOURCE=/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-diagnostic-v67/source
export SPOOL=/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-diagnostic-v67/spool
export RUNTIME_RECEIPT=/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json
export LOCAL_RUNTIME="${SLURM_TMPDIR:-/tmp}/lp-reader32-import-v67-$SLURM_JOB_ID"
export PYTHONPATH="$LOCAL_RUNTIME:$SOURCE"
PYTHON=/projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python
export PYTHON
cd "$SOURCE"
REMAINING=$("$PYTHON" -c 'import os,time;print(max(.001,float(os.environ["STUDY_WALL_START"])+90-time.time()))')
set +e
timeout --signal=TERM --kill-after=2s "${REMAINING}s" bash -c '"$PYTHON" "$SOURCE/extract_verified_runtime.py" && "$PYTHON" -u "$SOURCE/reader32_import_diagnostic_v67_cpu.py" "$SOURCE" "$SPOOL"'
STATUS=$?
set -e
export CPU_DIAGNOSTIC_EXIT_CODE="$STATUS"
"$PYTHON" -c 'import json,os,time;from pathlib import Path; p=Path(os.environ["SPOOL"])/"CPU_terminal.json";p.open("x").write(json.dumps({"exit_code":int(os.environ["CPU_DIAGNOSTIC_EXIT_CODE"]),"at":time.time(),"model_calls":0,"GPU_allocations":0})+"\n")'
exit "$STATUS"
