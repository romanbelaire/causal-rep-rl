#!/bin/bash
# tv3 distill arm — exp1: Z* distillation RSTR + VAE baseline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv3-distill-exp1.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv3-distill-exp1

set -euo pipefail
source src/theory_validation/slurm_env_v3.sh

echo "=== tv3 distill exp1: training ==="
tv3_distill_exp1_train

echo "=== tv3 distill exp1: plots ==="
python -m src.theory_validation.plot_validation_v3 --arm distill --exp 1
python -m src.theory_validation.plot_validation_v3 --arm distill --exp training_loss
echo "tv3 distill exp1 complete."
