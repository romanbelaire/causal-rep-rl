#!/bin/bash
# tv3 kappa arm — exp1: κ-directed RSTR + VAE baseline (3 seeds each)
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv3-kappa-exp1.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv3-kappa-exp1

set -euo pipefail
source src/theory_validation/slurm_env_v3.sh

echo "=== tv3 kappa exp1: training ==="
tv3_kappa_exp1_train

echo "=== tv3 kappa exp1: plots ==="
python -m src.theory_validation.plot_validation_v3 --arm kappa --exp 1
python -m src.theory_validation.plot_validation_v3 --arm kappa --exp training_loss
echo "tv3 kappa exp1 complete."
