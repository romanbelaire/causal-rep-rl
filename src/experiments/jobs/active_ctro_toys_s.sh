#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=4
#SBATCH -A cis260223p
#SBATCH --mail-type=END
#SBATCH --mail-user=rbelaire@andrew.cmu.edu
#SBATCH --job-name=actro-toys
#SBATCH -o /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%j.out
#SBATCH -e /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/%x.%j.err
#SBATCH --requeue
#
# CPU toys + invariants for active CTRO (E1–E5). No GPU.
# Submit: sbatch src/experiments/jobs/active_ctro_toys_s.sh

REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs results/active_ctro/toys

export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=""

python -c "import torch; print('cuda_available', torch.cuda.is_available()); assert not torch.cuda.is_available(), 'Toys must run on CPU'"

python -m src.experiments.run_active_ctro_toys \
  --results-root results/active_ctro/toys
