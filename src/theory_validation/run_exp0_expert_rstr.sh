#!/bin/bash
# Exp 0: one RSTR expert run (single seed). Submit via submit_exp0_experts.sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --constraint=h100nvl|h200|h100|l40|l40s|a100|a40
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --job-name=tv2-exp0-rstr

set -euo pipefail
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"
module purge
module load Python/3.10.16-GCCcore-13.3.0
module load CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 OpenMPI/5.0.3-GCC-13.3.0
source rl-venv/bin/activate
source src/theory_validation/env_pytorch_alloc.sh

: "${SEED:?Set SEED (e.g. sbatch --export=ALL,SEED=42)}"
echo "RSTR expert training seed=${SEED}"
srun --gres=gpu:1 python -m src.main \
  --config configs/theory_validation/expert_rstr_minigrid.json \
  --seed "$SEED"
