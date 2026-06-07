#!/bin/bash
# tv3 kappa exp2 remainder — kappa_all_off seed44
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=1-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv3-kexp2-alloff-s44.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv3-kexp2-alloff-s44

set -euo pipefail
source src/theory_validation/slurm_env_v3.sh

echo "=== tv3 kappa exp2 remainder: kappa_all_off seed44 ==="
tv3_require_expert_weights rstr
tv3_srun_train configs/theory_validation_v3/exp3_kappa_all_off.json 44
echo "tv3 kappa exp2 kappa_all_off seed44 complete."
