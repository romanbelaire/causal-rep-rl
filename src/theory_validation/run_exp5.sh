#!/bin/bash
# Theory validation v2 — Exp 5 (self-contained):
#   1. RSTR + VAE + vanilla transfer (3 seeds)
#   2. Plots
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv2-exp5.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv2-exp5

set -euo pipefail
source src/theory_validation/slurm_env.sh

echo "=== Exp 5: training ==="
for S in "${TV2_SEEDS[@]}"; do
  tv2_srun_train configs/theory_validation/exp5_rstr_transfer.json "$S"
  tv2_srun_train configs/theory_validation/exp5_vae_transfer.json "$S"
  tv2_srun_train configs/theory_validation/exp5_vanilla_transfer.json "$S"
done

echo "=== Exp 5: plots ==="
python -m src.theory_validation.plot_validation_v2 --exp 5
echo "Exp 5 complete."
