#!/bin/bash
#SBATCH --job-name=lp-app-calibration-v76
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:01:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/application-calibration-v76/spool/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/application-calibration-v76/spool/cpu-%j.err
set -eo pipefail
if [[ ${APP_AUDIT_OUTER:-0} != 1 ]]; then
 export APP_AUDIT_OUTER=1
 exec timeout --signal=TERM --kill-after=1s 29s bash "$0"
fi
ROOT=/scratch/alpine/paco0228/latent_proxy_runs/application-calibration-v76
export STUDY_WALL_START=$(date +%s.%N)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python -S -B "$ROOT/source/application_calibration_v76.py" "$ROOT/manifest.json" "$ROOT/source" "$ROOT/spool" >/dev/null 2>&1
