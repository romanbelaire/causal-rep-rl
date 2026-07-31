#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=procgen-perf-agg
#SBATCH -o logs/%x_%a.%j.out
#SBATCH -e logs/%x_%a.%j.err
#SBATCH --array=0-0
#SBATCH --requeue

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
export PYTHONUNBUFFERED=1

EXP_NAME="${EXP_NAME:?EXP_NAME required}"
PROCGEN_ROOT="results/perf_eval/procgen_easy/${EXP_NAME}"
TABLE_DIR="results/perf_eval/tables/${EXP_NAME}"
mkdir -p "${TABLE_DIR}"
python -m src.experiments.aggregate_performance_eval "${PROCGEN_ROOT}" \
  --output "${TABLE_DIR}/procgen_easy.txt"
