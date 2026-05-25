#!/bin/bash
# Theory validation v2 — Exp 2 (self-contained):
#   1. Train RSTR + VAE (same runs as Exp 1; safe to re-run after Exp 1)
#   2. Plot lagged CCF
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv2-exp2.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv2-exp2

set -euo pipefail
source src/theory_validation/slurm_env.sh

echo "=== Exp 2: training (skipped if Exp 1 already finished) ==="
tv2_exp12_train_if_needed

echo "=== Exp 2: plots ==="
python -m src.theory_validation.plot_validation_v2 --exp 2
echo "Exp 2 complete."
