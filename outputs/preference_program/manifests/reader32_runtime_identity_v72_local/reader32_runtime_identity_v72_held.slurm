#!/bin/bash
#SBATCH --job-name=lp-runtime-identity-v72
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:01:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/runtime-identity-v72/spool/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/runtime-identity-v72/spool/cpu-%j.err
set -eo pipefail
if [[ ${IDENTITY_OUTER_ACTIVE:-0} != 1 ]]; then
 export IDENTITY_OUTER_ACTIVE=1
 exec timeout --signal=TERM --kill-after=1s 29s bash "$0"
fi
ROOT=/scratch/alpine/paco0228/latent_proxy_runs/runtime-identity-v72
export STUDY_WALL_START=$(date +%s.%N)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python -S -B "$ROOT/source/reader32_runtime_identity_v72.py" "$ROOT/manifest.json" "$ROOT/spool" >/dev/null 2>&1
