#!/bin/bash
#SBATCH --job-name=lp-tracer-calibration-v71
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:01:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/tracer-calibration-v71/spool/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/tracer-calibration-v71/spool/cpu-%j.err
set -eo pipefail
# Slurm resolution is one minute; application process tree is bounded to 30s.
if [[ ${CALIBRATION_OUTER_ACTIVE:-0} != 1 ]]; then
  export CALIBRATION_OUTER_ACTIVE=1
  exec timeout --signal=TERM --kill-after=1s 29s bash "$0"
fi
ROOT=/scratch/alpine/paco0228/latent_proxy_runs/tracer-calibration-v71
SOURCE="$ROOT/source"
CONTROLLER="$ROOT/controller"
SPOOL="$ROOT/spool"
PYTHON=/projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python
export STUDY_WALL_START=$(date +%s.%N)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$CONTROLLER"
cd "$SOURCE"
# Independent admission; no v68 approval fields or qualification driver.
"$PYTHON" -S -c 'import json,sys;m=json.load(open(sys.argv[1]));sys.exit(0 if m.get("calibration_admission") is True and m.get("allocation_approved") is True else 78)' "$CONTROLLER/manifest.json" >/dev/null 2>&1
REMAINING=$("$PYTHON" -S -c 'import os,time;print(max(.001,float(os.environ["STUDY_WALL_START"])+20-time.time()))')
set +e
timeout --signal=TERM --kill-after=2s "${REMAINING}s" "$PYTHON" -S -u "$CONTROLLER/reader32_tracer_calibration_v69.py" "$CONTROLLER/manifest.json" "$SOURCE" "$SPOOL" >/dev/null 2>&1
STATUS=$?
set -e
"$PYTHON" -S "$CONTROLLER/reader32_calibration_terminal_v69.py" "$SPOOL" "$STATUS" >/dev/null 2>&1
exit "$STATUS"
