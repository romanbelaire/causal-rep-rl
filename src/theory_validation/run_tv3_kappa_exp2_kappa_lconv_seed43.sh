#!/bin/bash
# tv3 kappa exp2 remainder — kappa_lconv seed43
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200gb
#SBATCH --time=1-0:00:0
#SBATCH --mail-type=END
#SBATCH --output=%u.tv3-kexp2-klconv-s43.%j.out
#SBATCH --requeue
#SBATCH --partition=researchshort
#SBATCH --account=pradeepresearch
#SBATCH --qos=research-1-qos
#SBATCH --mail-user=rbelaire.2021@phdcs.smu.edu.sg
#SBATCH --job-name=tv3-kexp2-klconv-s43

set -euo pipefail
source src/theory_validation/slurm_env_v3.sh

echo "=== tv3 kappa exp2 remainder: kappa_lconv seed43 ==="
tv3_require_expert_weights rstr
tv3_srun_train configs/theory_validation_v3/exp3_kappa_lconv.json 43
echo "tv3 kappa exp2 kappa_lconv seed43 complete."
