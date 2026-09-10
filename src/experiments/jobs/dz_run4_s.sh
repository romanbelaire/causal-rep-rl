#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=2-00:00:00
#SBATCH --constraint=v100|l40|l40s|a100|a40
#SBATCH --partition=GPU-shared
#SBATCH --output=/ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/dz_run4_%j.out
#SBATCH --error=/ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs/dz_run4_%j.err
#SBATCH --job-name=dz-run4
#
# Run 4: rate test (prioritize after Run 0 passes).

set -euo pipefail
module load pytorch/26.05-2.11-py3 2>/dev/null || true
cd /jet/home/rbelaire/causal-rep-rl
if [ -d .venv ]; then source .venv/bin/activate; fi
export RESULTS_ROOT="${RESULTS_ROOT:-/ocean/projects/cis260223p/rbelaire/causal-rep-rl/results}"
export DEVICE=cuda
mkdir -p "${RESULTS_ROOT}/slurm" /ocean/projects/cis260223p/rbelaire/causal-rep-rl/logs
bash src/experiments/jobs/dz_phase3.sh run4
