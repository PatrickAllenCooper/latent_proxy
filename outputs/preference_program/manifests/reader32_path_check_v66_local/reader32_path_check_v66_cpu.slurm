#!/bin/bash
# PROPOSAL ONLY. Requires separately bound CPU allocation approval.
#SBATCH --job-name=lp-reader32-path-v66
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/reader32-path-check-v66/spool/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/reader32-path-check-v66/spool/cpu-%j.err
set -eo pipefail
export STUDY_WALL_START=$(date +%s.%N)
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=/projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python
CHECKER=/scratch/alpine/paco0228/latent_proxy_runs/reader32-path-check-v66/source/reader32_compute_path_check.py
APPROVAL=/scratch/alpine/paco0228/latent_proxy_runs/reader32-path-check-v66/source/approval.json
SOURCE=/scratch/alpine/paco0228/latent_proxy_runs/reader32-qualification-v65/source
SPOOL=/scratch/alpine/paco0228/latent_proxy_runs/reader32-path-check-v66/spool
REMAINING=$("$PYTHON" -c 'import os,time; print(max(.001,float(os.environ["STUDY_WALL_START"])+90-time.time()))')
timeout --signal=TERM --kill-after=2s "${REMAINING}s" "$PYTHON" -u "$CHECKER" "$SOURCE" --approval "$APPROVAL" --spool "$SPOOL"
