#!/bin/bash
# Exp 3: repr_coef 0.1 vs 0.0 collapse ablation, 3 seeds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --constraint=nopreempt #h100nvl|h200|h100|l40|l40s|a100|a40
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --job-name=tv2-exp3

set -euo pipefail
REPO=/common/home/users/r/rbelaire.2021/causal-rep
cd "$REPO"
module purge
module load Python/3.10.16-GCCcore-13.3.0
module load CUDA/12.6.0 cuDNN/9.5.0.50-CUDA-12.6.0 OpenMPI/5.0.3-GCC-13.3.0
source rl-venv/bin/activate
source src/theory_validation/env_pytorch_alloc.sh

SEEDS=(42 43 44)
for S in "${SEEDS[@]}"; do
  srun --gres=gpu:1 python -m src.main --config configs/theory_validation/exp3_rstr_repr_on.json --seed "$S"
  srun --gres=gpu:1 python -m src.main --config configs/theory_validation/exp3_rstr_repr_off.json --seed "$S"
done

python -m src.theory_validation.analyze_checkpoints
python -m src.theory_validation.plot_validation_v2 --exp 3
