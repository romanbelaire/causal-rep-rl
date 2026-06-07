#!/bin/bash
# tv4 exp1 MICo arm: baseline + alpha sweep x 3 seeds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=02-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv4-exp1-mico.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv4-exp1-mico

set -euo pipefail
source src/theory_validation/slurm_env_v4.sh

echo "=== tv4 exp1 MICo: training ==="
tv4_mico_exp1_train

echo "=== tv4 exp1 MICo: plots ==="
python -m src.theory_validation.plot_validation_v4 --arm mico --exp 1
python -m src.theory_validation.plot_validation_v4 --arm mico --exp training_loss
echo "tv4 exp1 MICo complete."
