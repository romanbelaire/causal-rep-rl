#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 04:00:00
#SBATCH --ntasks-per-node=4
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=plots
#SBATCH -o logs/%x_%a.%j.out

#SBATCH --array=0-0
#SBATCH --requeue

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/slurm

# PyTorch + deps: Ocean conda env (Python 3.10)
export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH

export PYTHONUNBUFFERED=1

python scripts/plot_metrics.py --output-dir plots
