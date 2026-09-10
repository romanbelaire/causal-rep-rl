#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=2
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=actro-agg
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%j.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%j.err
#SBATCH --requeue
#
# CPU aggregation of MiniGrid E0/E6. Fails if any seed is missing.
# Submit after the MiniGrid array: sbatch --dependency=afterok:<JOBID> this script.

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/active_ctro/minigrid/tables

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=""

python -m src.experiments.aggregate_active_ctro \
  --results-root results/active_ctro/minigrid
