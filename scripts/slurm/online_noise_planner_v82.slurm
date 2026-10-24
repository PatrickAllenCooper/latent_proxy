#!/bin/bash
#SBATCH --job-name=lp-online-noise-planner-v82
#SBATCH --account=ucb736_asc1
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:01:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/alpine/paco0228/latent_proxy_runs/online-noise-planner-v82/spool/cpu-%j.out
#SBATCH --error=/scratch/alpine/paco0228/latent_proxy_runs/online-noise-planner-v82/spool/cpu-%j.err
set -eo pipefail
if [[ ${RANK_STRESS_OUTER:-0} != 1 ]]; then
 export RANK_STRESS_OUTER=1
 exec timeout --signal=TERM --kill-after=1s 29s bash "$0"
fi
ROOT=/scratch/alpine/paco0228/latent_proxy_runs/online-noise-planner-v82
case "${SHARD:-}" in 0|1|2) ;; *) exit 78 ;; esac
export STUDY_WALL_START=$(date +%s.%N)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=/projects/paco0228/software/anaconda/envs/latent-proxy-env/bin/python
"$PYTHON" -S -B -c 'import hashlib,json,pathlib,sys;m=json.load(open(sys.argv[1]));assert m["approval"]=="user_20261024_online_noise_planner";assert m["resources"]=={"cpus":1,"memory_MiB":256,"scheduler_wall_seconds":60,"application_wall_seconds":30,"check_seconds":20,"gpus":0};r=pathlib.Path(sys.argv[1]).parent;assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in m["hashes"].items());h=hashlib.sha256();f=open(sys.executable,"rb");[h.update(b) for b in iter(lambda:f.read(1048576),b"")];assert h.hexdigest()==m["python_sha256"]' "$ROOT/manifest.json"
exec "$PYTHON" -S -B "$ROOT/source/online_noise_planner_v82.py" "$ROOT/shard$SHARD/results" "$ROOT/protocol_shard$SHARD.json"
