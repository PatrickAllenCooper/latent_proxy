#!/bin/bash
#SBATCH --job-name=lp-reader32-import-v68
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68/spool/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68/spool/cpu-%j.err
set -eo pipefail
# Held before extraction/import; previous CPU approval cannot admit this wrapper.
PYTHON=/projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python
ROOT=/scratch/alpine/paco0228/latent_proxy_runs/reader32-import-trace-v68
export STUDY_WALL_START=$(date +%s.%N)
"$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1]));sys.exit(0 if m["cpu_admission"] is True else 78)' "$ROOT/revision-receipt-safe/controller/manifest.json"
export SOURCE="$ROOT/revision-receipt-safe/source" SPOOL="$ROOT/spool"
CONTROLLER="$ROOT/revision-receipt-safe/controller"
export RUNTIME_RECEIPT=/scratch/alpine/paco0228/latent_proxy_runs/verification-runtime-v39/runtime_receipt.json
export LOCAL_RUNTIME="${SLURM_TMPDIR:-/tmp}/lp-reader32-import-v68-$SLURM_JOB_ID"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$LOCAL_RUNTIME:$SOURCE"
cd "$SOURCE"
export PYTHON ROOT
REMAINING=$("$PYTHON" -c 'import os,time;print(max(.001,float(os.environ["STUDY_WALL_START"])+90-time.time()))')
# No raw trace/stderr reaches Slurm logs: capture consumes its child's pipes.
# Extraction output is suppressed; a failed preparation is inconclusive.
set +e
timeout --signal=TERM --kill-after=2s "${REMAINING}s" bash -c '"$PYTHON" "$ROOT/revision-receipt-safe/controller/reader32_trace_capture_v68_receipt_safe.py" "$ROOT/revision-receipt-safe/controller/manifest.json" "$SPOOL" --preflight >/dev/null 2>&1 && "$PYTHON" "$SOURCE/extract_verified_runtime.py" >"$SPOOL/runtime_stage.json" 2>/dev/null && exec "$PYTHON" "$ROOT/revision-receipt-safe/controller/reader32_trace_capture_v68_receipt_safe.py" "$ROOT/revision-receipt-safe/controller/manifest.json" "$SPOOL" >/dev/null 2>&1'

STATUS=$?
set -e
export STATUS
"$PYTHON" -c 'import json,os,time;from pathlib import Path;p=Path(os.environ["SPOOL"])/"wrapper_terminal.json";f=p.open("x");f.write(json.dumps({"exit_code":int(os.environ["STATUS"]),"at":time.time(),"GPU_allocations":0,"model_calls":0})+"\n");f.flush();os.fsync(f.fileno());f.close();d=os.open(os.environ["SPOOL"],os.O_RDONLY);os.fsync(d);os.close(d)' >/dev/null 2>&1
exit "$STATUS"
