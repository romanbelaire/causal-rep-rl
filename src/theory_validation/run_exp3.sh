#!/bin/bash
# Theory validation v2 — Exp 3 (self-contained):
#   1. RSTR repr_coef on vs off (3 seeds)
#   2. Checkpoint bootstrap analysis + plots
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv2-exp3.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv2-exp3

set -euo pipefail
source src/theory_validation/slurm_env.sh

echo "=== Exp 3: training ==="
for S in "${TV2_SEEDS[@]}"; do
  tv2_srun_train configs/theory_validation/exp3_rstr_repr_on.json "$S"
  tv2_srun_train configs/theory_validation/exp3_rstr_repr_off.json "$S"
done

echo "=== Exp 3: analysis + plots ==="
python -m src.theory_validation.analyze_checkpoints
python -m src.theory_validation.plot_validation_v2 --exp 3
echo "Exp 3 complete."
