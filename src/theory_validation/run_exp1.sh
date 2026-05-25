#!/bin/bash
# Theory validation v2 — Exp 1 (self-contained):
#   1. Train RSTR (lconv on) + VAE (no repr loss), 3 seeds
#   2. Plot μ_concave vs rank (PR + PCA)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv2-exp1.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv2-exp1

set -euo pipefail
source src/theory_validation/slurm_env.sh

echo "=== Exp 1: training ==="
tv2_exp12_train

echo "=== Exp 1: plots ==="
python -m src.theory_validation.plot_validation_v2 --exp 1
echo "Exp 1 complete."
