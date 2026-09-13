#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 08:00:00
#SBATCH --ntasks-per-node=16
#SBATCH -A cis260223p
#SBATCH --job-name=sep-predictiveness
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail
REPO=/ocean/projects/cis260223p/rbelaire/causal-rep-rl
cd "$REPO" || exit 1
mkdir -p logs
export PATH=/ocean/projects/cis260223p/rbelaire/envs/causal-rep/bin:$PATH
export PYTHONUNBUFFERED=1
python -m src.experiments.analyze_aliasing_predictiveness \
  --suite procgen_easy \
  --reroll --n-steps 512 --device cpu \
  --output-dir /ocean/projects/cis260223p/rbelaire/causal-rep-rl/results/aliasing_predictiveness/procgen_easy
